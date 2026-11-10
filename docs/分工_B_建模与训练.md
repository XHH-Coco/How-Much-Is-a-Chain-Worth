# 分工 B · 建模与训练（Modelling）

> **对应**：`docs/Proposal_HowMuchIsAChainWorth_v1.3.md` §6（实验设计）、§7.1 的 Owner 列 `B`、§9 的 `B — Modelling`
> **工时**：§7.1 逐项归属 **37.5 人时** + 返工分摊 4 = **41.5 人时**（占 105 的 40%，全队最重）
> **你负责的目录**：`src/chainworth/{chem,arms}/`、`configs/{arms,experiments}/`、`results/raw_runs/`、`results/aggregate/predictions/`
> **一句话**：同一批样品，用五台仪器各测一遍，把每台仪器的读数原样记下来交给 C。

---

## 0. 先用大白话讲：你这份差事是干什么的

### 你要干的活，是"把样品用五台仪器各测一遍"

四个人合写一篇实验报告：

- **A** 在实验开始前把试剂配好、天平校准、并在记录本上签字（数据、划分、描述符、嵌入）；
- **你** 拿着同一批样品，分别用五台仪器测，每测一次就在卡片上记一行读数；
- **C** 拿你那一叠卡片算"哪台仪器多称出了多少"，并配上误差范围；
- **D** 把这件事讲成 8 分钟。

"臂"（arm）就是**仪器的分支**，不是五个不同的模型比赛。同一批样品、同一套划分、同一批种子，只换一个变量：

| 臂 | 换的是什么变量 | 大白话 |
|:--|:--|:--|
| **A1 地板** | 什么都不学 | 先量一下"瞎猜"有多差。忘了这一步，后面所有提升都没有参照 |
| **A2 经典** | 只给重复单元的描述符 | 标准秤：RDKit 描述符 + XGBoost / 随机森林 |
| **A3 图** | 只给图结构和消息传递 | 更贵的秤：官方 polymer_chemprop，**必须 GPU** |
| **A4 语言模型** | 只给预训练嵌入 | 别人校准过的秤：polyBERT |
| **A5 链信息（我们的方法）** | 在 A2/A3 的输入里塞进链的自由度 | 给秤加一个"链长"刻度：重复次数、wPSMILES 权重、多单元拼接 |

**关键的一点，也是最容易做错的一点**：A5 的定位是"**改输入，不改模型**"。如果你在 A5 里顺手把 XGBoost 换成了更好的模型，那 A5 相比 A2 的增益就分不清是"信息"带来的还是"模型"带来的——而整篇 proposal 的 H3 就是"信息 vs 架构"这个对比。**A5 越界，H3 就作废。**

### 你产出的东西长什么样

不是"一个训练好的模型"，而是**一叠运行卡片**。每次训练落一个 JSON，最小长这样：

```json
{
  "run_id": "pb_t1_tg__a3_graph__seed0",
  "benchmark": "polybench", "task": "pb_t1_tg", "split_version": "official_70_30__sha256:8f2c…",
  "arm": "a3_graph", "seed": 0, "config_hash": "sha256:1a9d…",
  "n_train": 5638, "n_test": 2417,
  "metrics": {"r2": 0.812, "mae": 21.4, "rmse": 33.9},
  "wall_clock_s": 96, "device": "cuda:0",
  "env_fingerprint": "results/env_fingerprints/gnn.json"
}
```

加一份**测试集预测**（`predictions/<run_id>.parquet`，两列：`row_id, y_pred`）。**没有预测文件，C 就没法算 bootstrap 区间** —— 只能拿一个点估计算结论，等于白跑。

### 做完的标志（一句话）

> **C 拿到 `results/aggregate/` 之后，不需要再问你任何一句话，就能算出 Λ、κ 和它们的置信区间。**
> 也就是说：每次运行都留了 run 记录、留了预测、留了 split 版本、留了配置哈希。

---

## 1. 你在流水线上的位置

| 阶段 | A | B（你） | C | D |
|:--|:--|:--|:--|:--|
| **P0 环境** | ● 装包锁版本 | ○ 同时把 `polymer_chemprop` 跑通一个 fold | – | ○ 读 Proposal |
| **P1 获取数据** | ● 下载 + 完整性 | – | – | – |
| **P2 清洗与划分** | ● 标签调和 + 划分快照 | ○ 写特征装配，等 A 的产物 | ○ 先写指标单测 | – |
| **P3 配置** | ● `configs/tasks/` 冻结 | ● **主责**：`configs/arms/`、`configs/experiments/` | – | ○ 搭幻灯片骨架 |
| **P4 代码** | ○ 划分/泄漏测试 | ● **主责**：五个臂 | ● **主责**：Λ/κ | – |
| **P5 运行** | ○ 审计 | ● **主责**：全部拟合 | ○ 等 run 记录 | – |
| **P6 图表与文档** | ○ 审计签字 | ○ 补跑与答疑 | ● **主责**：5 张图 | ● **主责**：讲稿/彩排 |

● 主责　○ 参与/等待

**你是唯一一个"卡在中间"的人**：上面等 A 的数据，下面 C 等你跑完。所以你的策略必须是**先用小样本把管线跑通，再放大**——不要等 8 个任务的数据全齐了才开始写第一行代码。

### 与 A 的接口（划清边界，避免重复劳动）

| 谁 | 产出什么 | 放在哪 |
|:--|:--|:--|
| **A** | 结构级：规范化 PSMILES、RDKit 描述符矩阵、Morgan 指纹、polyBERT 嵌入 | `data/processed/{structures,descriptors,polybert}/` |
| **你** | 任务级：把结构级特征 join 成 `X`、wPSMILES 权重、图对象、A5 注入器 | `src/chainworth/{chem,arms}/` + `data/processed/features/` |

**一句话分工**：A 保证"每个结构算对一次"，你保证"每个任务的每一行对上正确的结构，并且行序由 split 文件决定"。

---

## 2. 任务清单（拿什么 → 怎么做 → 验收标准）

> 每项标了 pH（人时，含 agent 辅助）与是否关键路径。**8 项合计 37.5 pH**，与 §7.1 一致。
> 说明：§7.1 把"表示管线"记成 4.5 pH 一行，这里拆成 B-1（3.0）+ B-5（1.5），合计不变，拆开是因为 A5 有独立的验收标准。

### B-1 · 表示装配与图构建（3.0 pH｜关键）

| 项 | 内容 |
|:--|:--|
| **拿什么** | A 交的 `data/processed/structures/psmiles.parquet`、`descriptors/<task>.parquet`、`polybert/{embeddings.npy,index.parquet}`；`configs/tasks/*.yaml`；`splits/`（只读） |
| **怎么做** | ① 按 `row_id`（不是 DataFrame 行序）join 结构级特征，得到任务级 `X`；② 构建 PyG `Data` 对象（原子特征、键特征、wPSMILES 权重作为节点/边权重）；③ 复现 chemprop 的 polymer 模式图构建，**用官方脚本对拍一次**，不要凭记忆写 |
| **产出** | `src/chainworth/chem/{features.py,graph.py}`、`src/chainworth/io/loaders.py`、`data/processed/features/<task>.npz` + `feature_manifest.csv`（列数、指纹位数、缺失数） |
| **验收标准** | ① 行数 = 该任务调和后可用标签数，且**行序由 split 文件决定**（测试断言：`X.index` 与 `splits/…/test.json` 的索引逐位相同）；② 随机抽 100 个结构，特征向量与 A 的描述符矩阵逐位相等；③ 同一结构两次构图 `edge_index` 完全相同（原子排序稳定，不许依赖字典序）；④ 特征矩阵里**不含任何用全量数据算的统计量**（标准化只在 train 上 fit，见 A-6 的泄漏测试）；⑤ 解析失败的结构计数并列出 ID，不许静默 `dropna` |

> **红线**：`X` 的行序如果按 DataFrame 顺序走，划分就失效了——而代码不会报错。这是 §7.5 点名的第 (i) 类失败模式。

### B-2 · A1 地板 + A2 经典四臂（5.0 pH｜关键）

| 项 | 内容 |
|:--|:--|
| **拿什么** | B-1 的任务级特征矩阵 + `splits/`（官方 70/30 或 5 折，二者都用官方文件） |
| **怎么做** | 四个学习器：**Dummy（均值）/ kNN / XGBoost / RandomForest**。统一一套超参空间、统一预算 **≤30 trials**，搜索用 Murcko-grouped CV 且**只在 train 内做**；每个配置跑 **3 个模型种子**；预测落盘 |
| **产出** | `configs/arms/{a1_floor,a2_conventional}.yaml`（含搜索空间与预算表）+ 每 run 一份 `results/raw_runs/<run_id>.json` + `results/aggregate/predictions/<run_id>.parquet` |
| **验收标准** | ① run 记录字段齐全（见 §0 那张 JSON），缺一项不算完成；② 同 seed 连跑两遍，`r2` 差异 **< 1e-9**（经典模型是确定性的，达不到说明有随机源没固定）；③ 预测行数 = 测试集行数，`row_id` 与 split 文件逐位一致；④ PolyBench26 上把 `RDKit_RF` 与官方基线数对比，同任务差距 **< 2σ**，> 2σ 时先查协议再报数（不许直接当"我们更好/更差"）；⑤ 预算表写进 `docs/`，被问"调了多少次参"能立刻答 |

> **口径待统一**：§7.1 写"150 fits"，§6 写"96 classical + 48 GNN"。150 − 96 = 54 的来源 proposal 没写明（可能是 A1 与 A5 的复跑）。**B-2 开工前与 A 一起定死一个数**，写进 `configs/experiments/main_table.yaml`，之后文档只许引用这一个数。

### B-3 · A3 官方图神经网络主扫（11.0 pH｜**GPU 必需**｜关键）

| 项 | 内容 |
|:--|:--|
| **拿什么** | B-1 的图对象、`environment/gnn.yml`（`chemprop==2.1.0`、`torch==2.3.1+cu121`、`torch-geometric==2.5.3`）、A 的划分 |
| **怎么做** | 官方 `polymer_chemprop`：depth 3、hidden 300、batch 64（12 GB 上 OOM 就改 32 + 梯度累积 2，墙钟 ≈1.2×）、lr 1e-3、≤100 epoch、early stopping 15。**共 48 runs** = PolyBench 6 任务 × 3 种子（18）+ PolyBench26 2 任务 × 5 折 × 3 种子（30）。单 run 1–8 min，合计 **3–5 GPU-h**，一夜跑完 |
| **产出** | `configs/arms/a3_graph.yaml` + 48 条 run 记录 + 48 份预测 + `results/aggregate/gpu_hours.csv`（每 run 实测墙钟） |
| **验收标准** | ① **D0 24:00 前**单任务单折跑通（硬门，§7.2）；② 每条 run 的实测耗时 ≤ §7.4 预测值的 **1.5×**，超标立即走 triage (i) 而不是等；③ 记录里带 GPU 型号 + 驱动版本 + `torch.version.cuda`；④ 同 seed 两次差异 **< 1e-3**（GPU 非确定性允许，但要开 `cudnn.deterministic` 并记录；不许只写"大致一致"）；⑤ `torch.cuda.set_per_process_memory_fraction(0.9)`，一个进程一张卡 |

> **为什么值得单列 11 pH**：真正花时间的不是 48 次运行，而是把官方图管线调通（featurizer 版本、`wPSMILES` 权重怎么进图、early stopping 的监控量选哪个）。**48 次运行省不下多少，管线调错一次就是半天。**

### B-4 · A4 polyBERT 迁移臂（3.5 pH｜GPU 建议）

| 项 | 内容 |
|:--|:--|
| **拿什么** | A 的 `polybert/embeddings.npy` + `index.parquet`（42k 结构，768 维左右）；`environment/polybert.yml` |
| **怎么做** | **两种模式都要写**：(a) **冻结嵌入 + ridge 头**（几十分钟，CPU 也能跑）；(b) **微调**（2.5–4 GPU-h）。默认跑 (a)，时间够再跑 (b) |
| **产出** | `configs/arms/a4_polybert.yaml` + 两条（或一条）run 记录 + `docs/DECISIONS.md` 里一条决策记录：为什么跑/不跑微调 |
| **验收标准** | ① 嵌入行与结构一一对应（抽 50 个结构反查，见 A-8）；② (a) 与 (b) 若都跑了，同一 split/seed 上比较，差异写进主表备注；③ 若只跑 (a)，**必须**在决策记录里写明触发了 §7.4 的 triage (ii)，并在幻灯片上标注——不许悄悄降级 |

### B-5 · A5 链信息注入臂（1.5 pH｜关键，H3 的一半）

这是"我们自己的方法"，也是最容易被做歪的一项。

| 项 | 内容 |
|:--|:--|
| **拿什么** | B-1 的特征矩阵与图对象；Proposal §6 的 A5 定义（重复次数作为显式特征、wPSMILES 权重、多单元序列拼接） |
| **怎么做** | 三条注入**各自可单独开关**，所以实际产出三组：A5a（重复次数）、A5b（wPSMILES 权重）、A5c（多单元拼接）+ A5all（三条全开）。学习器、超参空间、划分、种子**与 A2（或 A3）完全相同**，唯一差别是输入矩阵 |
| **产出** | `src/chainworth/arms/chain_info.py`、`configs/arms/a5_chain.yaml`、4 组 run 记录 |
| **验收标准** | ① **同学习器同超参**：`config_hash` 除输入特征外必须与 A2 一致，用测试断言；② 三组消融齐全，只开一条也要能跑；③ 加一个**打乱对照**（把链特征在训练集内随机置换后重跑）：若打乱后增益没有下降，说明增益来自"多了几列"而不是"链信息"，**这条必须如实报**；④ 与 A3 的增益在同一 split/seed 上可直接相减（H3 需要"谁降得多"） |

> **为什么加打乱对照**：A5 相对 A2 多了几列特征，任何模型都会因为维度增加而略有变化。没有一个对照，你无法回答"这 0.02 是信息还是维度"。这个对照成本极低（一次重跑），但它是 H3 能不能站住的关键。

### B-6 · 主表汇总与 leaderboard 对齐（2.5 pH｜关键）

| 项 | 内容 |
|:--|:--|
| **拿什么** | 全部 run 记录 + 预测；PolyBench 排行榜上的公开数 |
| **怎么做** | 汇总 8 任务 × 5 臂的表（均值 ± std，附 bootstrap CI 由 C 提供）；逐格标注 run_id；与排行榜的差，写成"差在协议 / 差在实现"两类，不许混为一谈 |
| **产出** | `results/aggregate/main_table.csv` + `docs/Main Table Notes.md` |
| **验收标准** | ① 表中每一格都能点回具体 run_id；② 与排行榜对比时注明"官方数来自哪一版、哪个模型、哪种划分"；③ 差距 > 10% 的格子必须给出解释或标注 `unexplained`（诚实标注比编解释好） |

### B-7 · 机制实验：数据规模 + OpenPoly 扫描（6.0 pH｜部分 GPU）

| 项 | 内容 |
|:--|:--|
| **拿什么** | PolyBench26 官方 `scaling_splits`（300 / 1000 / 3000 / 10000；homopolymer MD_5000 只有 300 / 1000）；OpenPoly 的 3 个性质（Tg 443 / 介电常数 295 / Tm 168） |
| **怎么做** | ① **scaling**：5 档 × 5 折 × 2 性质 × 2 臂 = **100 fits**（CPU，XGBoost/RF）；② **OpenPoly 扫描**：3 性质 × 5 个 n 档（30/60/120/240/all）× 5 折 × 3 种子 = **225 GNN runs**（GPU 4–11 h），同网格经典臂另 **225 CPU fits** |
| **产出** | `results/aggregate/scaling.csv`、`results/aggregate/openpoly_scan.csv` + 对应 run 记录 |
| **验收标准** | ① 断言使用的是官方 `scaling_splits` 文件哈希，不是自己按行数截的（测试卡死）；② 5 档逐档齐全，缺档要报"哪一档不可能建成"（homopolymer MD_5000 没有 3000 档，这是事实不是失败）；③ OpenPoly 一律 **n ≤ 443**，不许外推；④ 225 个 run 必须与主扫**错夜跑**，不许抢同一张卡 |

### B-8 · 架构迁移实验（5.0 pH｜CPU）

| 项 | 内容 |
|:--|:--|
| **拿什么** | PolyBench26 官方 task 4 的划分（homopolymer → alternating / random / block） |
| **怎么做** | 2 性质 × 2 臂 × 5 折 × 3 种子 = **60 runs**，经典臂为主；有条件再加 GNN 臂（属可选项，+4–7 GPU-h，不进核心预算） |
| **产出** | `results/aggregate/arch_transfer.csv` + run 记录 |
| **验收标准** | ① 四个架构逐档齐全；② 报告**实际排序**（block > random > alternating > homopolymer 是否成立），而不是只报"通过/不通过"；③ 若排序不成立，直接写入 §5 的证伪条件，不许重调超参再试 |

---

## 3. 交付物清单（文件级）

```
configs/arms/a1_floor.yaml  a2_conventional.yaml  a3_graph.yaml  a4_polybert.yaml  a5_chain.yaml
configs/experiments/main_table.yaml  scaling.yaml  arch_transfer.yaml  openpoly_scan.yaml
src/chainworth/chem/features.py  graph.py
src/chainworth/arms/{floor,conventional,graph,polybert,chain_info}.py
src/chainworth/io/loaders.py
results/raw_runs/<run_id>.json                      全部 run，入库
results/aggregate/predictions/<run_id>.parquet       预测，入库（小文件）
results/aggregate/{main_table,scaling,arch_transfer,openpoly_scan,gpu_hours}.csv
data/processed/features/<task>.npz + feature_manifest.csv
docs/Main Table Notes.md
```

**不入库**：模型权重（`.pt`/`.ckpt`）、`logs/`、`outputs/`。权重可由 `config_hash` + seed 重建，没有入库价值。

---

## 4. 没有 GPU 怎么办

你的活有 **3 项必须 GPU**、**1 项建议 GPU**，其余全 CPU。**没有卡不等于没事做，只等于这 4 项改成"交代码 + 交请求"。**

| 任务 | GPU | 没卡的处置 |
|:--|:--|:--|
| B-1 表示装配 | 无 | 全做完 |
| B-2 四臂经典 | 无 | 全做完（这是主表的一半，**没有卡也必须完成**） |
| **B-3 GNN 主扫 48 runs** | **必需** | 交代码 + 交请求，由仓库所有者代跑 |
| B-4 polyBERT | 建议 | 只跑 (a) 冻结+ridge（CPU 可完成）；微调写进请求单 |
| B-5 A5 注入器 | 无 | 全做完（A5a/b/c 都可只用 XGBoost/RF 跑） |
| B-6 主表 | 无 | 全做完（缺的格子标 `blocked_no_gpu`） |
| B-7 scaling | 无 | scaling 的 100 fits 全做完 |
| **B-7 OpenPoly 225 runs** | **必需** | 交代码 + 交请求 |
| B-8 架构迁移 | 无 | 经典臂全做完（60 runs） |

### 交接协议（三条，缺一条不算交接完成）

1. **代码能在 CPU 上跑通一小片**：所有训练入口必须有 `--limit`（或 `--tasks/--folds` 子集参数）与 `--dry-run`，CPU 上 2 分钟内跑完一片并打印形状与指标。**没通过 smoke 的代码不要提交请求单**，那只会让代跑的人替你调错。
2. **不许写死 GPU**：`device = "cuda" if torch.cuda.is_available() else "cpu"`；CPU 下能跑完（慢无所谓）。`torch.cuda` 相关的调用必须包在 `if device == "cuda"` 里。
3. **在 `docs/RUN_REQUESTS.md` 里一行一条**，字段固定：

```
| run_id 规则 | 命令 | 预计 GPU-h | 依赖的 config_hash | 期望产出路径 | 提交时间 |
```

例：`pb_t1_tg__a3_graph__seed{0,1,2} | python -m chainworth.train --config configs/arms/a3_graph.yaml --task pb_t1_tg --seeds 0 1 2 | 0.2 | sha256:1a9d… | results/raw_runs/ | 11-13 20:00`

### 交接后你要做的（远程验收，不是甩手）

代跑完成后，run 记录会回到 `results/raw_runs/`。**你仍然负责验收**：

- [ ] 条数对不对（48 / 225 / 60，缺一条都要说清是哪条、为什么）
- [ ] 每条记录里 `device` 是 `cuda`、`env_fingerprint` 指向 `gnn.json`
- [ ] 预测文件行数与 split 测试集行数一致
- [ ] 随机抽 3 条，用同一命令在 CPU 上 `--limit` 重跑，指标量级合理（不要求相同，只要求不是明显坏掉）
- [ ] **发现异常就报，不要"帮忙"改数**

> **不许做的事**：手写 JSON 充数、把 `--limit 200` 的试跑产物改名成全量、把 CPU 跑的结果标成 `cuda`。占位请写 `{"status": "blocked_no_gpu", "requested_at": "…", "command": "…"}`，让主表能区分"没跑"和"跑了没成"。

---

## 5. 依赖关系与冻结边界

**你依赖**：A 的 `data/processed/`、`configs/tasks/`、`splits/`（全部是只读输入）。
**依赖你的**：C（run 记录 + 预测 → Λ/κ）、D（主表与图的数据源）、A（审计对象）。

| 冻结项 | 冻结在 | 谁能改 |
|:--|:--|:--|
| 超参空间与预算（≤30 trials） | `configs/arms/*.yaml` | 只能整体换版本并全表重跑，不能"这个任务多调两次" |
| run 记录格式 | `results/raw_runs/_schema.json`（B-2 开工前定） | 只能加字段不能改字段语义，改要迁移旧记录 |
| 臂的定义 | `configs/arms/` | A5 只能改输入，不能换学习器（否则 H3 作废） |
| 随机种子集合 | `configs/experiments/` | `{0,1,2}`；triage 时只许砍到 `{0}` 并在文档标注 |

---

## 6. 时间线与里程碑（以 8 个工作日 / 11-19 汇报为准）

| 时间 | 你要交的 | 卡点 |
|:--|:--|:--|
| **D0 18:00** | B-1 管线在 1 个任务上跑通；`polymer_chemprop` 能在一折上开跑 | 硬门（§7.2） |
| **D0 24:00** | **GNN 单任务单折跑完并记录耗时**；B-2 的第一个基线数出来 | 硬门；跑不完就触发 §7.2 的降级判断 |
| **D1 12:00** | 主扫 48 runs 已启动，**实测每 run 耗时与 §7.4 对照**；超标立刻 triage (i) | 硬门 |
| **D1 24:00** | 48 条 run 记录齐全且经你验收；A2 四臂 8 任务跑完 | 主表能不能做就看这个 |
| **P2 结束** | scaling 100 fits + OpenPoly 225 runs 完成（可与 P1 重叠，取决于卡数） | — |
| **P3 结束** | 主表逐格能点回 run_id；与排行榜的差已分类 | 交 D 前 |

> **夜间是给别人用的**：主扫在你手上那一夜，OpenPoly 扫描必须排在**另一夜**，不许抢卡。有第二张卡时两件事可以并行。

---

## 7. 红线与常见坑

**红线（做了等于重做）**

1. **自己切数据**：评估必须读 `splits/` 里的官方索引。`train_test_split` 出现在评估路径上就是红线（调参可以用，但必须在 train 内部，且代码里能看出来）。
2. **A5 换模型**：A5 的唯一定义是"同学习器、不同输入"。换了学习器，H3 直接从"信息 vs 架构"塌成"信息 vs 我喜欢的模型"。
3. **只在测试集上评估却用测试集调参**：超参搜索只能在 train 内做 Murcko-grouped CV，一次都不许碰 test。
4. **run 记录手写**：记录只能由训练脚本落盘，手改等于伪造。
5. **GPU 结果与 CPU 结果混报**：同一条 run 只允许一个 device，混着报的话没人能复现。

**常见坑**

| 坑 | 现象 | 正确做法 |
|:--|:--|:--|
| 特征按 DataFrame 行序 join | 划分错位，R² 反而变好 | 用 `row_id` join，断言 `X.index == split.test_index` |
| kNN 不做标准化 | kNN 结果极差，被解读成"经典方法不行" | 距离度量前标准化，且只在 train 上 fit |
| `X` 里混入全量统计量 | 泄漏，指标好得不像话 | 标准化/编码器只在 train 上 fit |
| 12 GB 上 hidden 600 + batch 64 | OOM，跑到一半崩 | hidden 300、batch 32 + 梯度累积 2 |
| 两个进程抢一张卡 | 都变慢或都 OOM | 一进程一卡，`set_per_process_memory_fraction(0.9)` |
| 只存了指标没存预测 | C 无法算 CI，只能报点估计 | 每次运行落 `predictions/<run_id>.parquet` |
| 拿 leaderboard 数直接对比 | 协议不同，被 Q&A 抓住 | 注明版本/模型/划分，差距 > 10% 标 `unexplained` |

---

## 8. 交接前自检（逐条打勾再 push）

- [ ] `git status` 里没有权重文件、没有 `logs/`、没有 `outputs/`
- [ ] 所有训练入口都有 `--limit` 与 `--dry-run`，CPU 上 2 分钟能跑完一片
- [ ] `pytest tests/` 全绿，含 `test_splits_identity.py`、`test_no_leakage.py`
- [ ] run 记录条数与该跑的数目对得上（48 / 225 / 60 / 100），缺的都有 `blocked_no_gpu` 占位
- [ ] 每份 run 记录都带 `split_version`、`config_hash`、`env_fingerprint`
- [ ] 每份 run 记录都有对应 `predictions/*.parquet`，行数与测试集一致
- [ ] `configs/arms/*.yaml` 里的预算表 ≤30 trials，且与 `docs/` 里的说明一致
- [ ] A5 的 `config_hash` 与 A2 只在输入特征上不同（断言过）
- [ ] 打乱对照跑过了，结论写进 `docs/Main Table Notes.md`
- [ ] `docs/RUN_REQUESTS.md` 里，所有需要 GPU 的项都写清了命令、预计耗时、期望产出
- [ ] 提交信息写清"做了什么 / 没做什么 / 下一步是什么"（README 第七节）

---

## 9. 工时口径核对（全队共用一张表）

| 角色 | §7.1 硬归属 | 返工池分摊 | 合计 | 占 105 pH |
|:--|--:|--:|--:|--:|
| A 数据与合规 | 29.5（22.5 + 审计 7） | +4 | 33.5 | 32% |
| **B 建模（你）** | **37.5** | **+4** | **41.5** | **40%** |
| C 分析 | 17.0（10 + 审计 7） | +4 | 21.0 | 20% |
| D 写作 | 7.0 | +2 | 9.0 | 9% |
| 合计 | 91.0 | 14 | 105 | 100% |

> **口径说明**：Proposal §9 写的是 28 / 32 / 28 / 12（百分比粗估），与 §7.1 逐项归属重算的结果（32 / 40 / 20 / 9）不一致——B 被低估、C 被高估。**以逐项归属为准**；若被问到，标准回答是"§9 是比例估计，§7.1 是逐项分解，两者以逐项分解为准"。这条差异需要记进 `docs/DECISIONS.md`（P3 建该文件时补一行）。

> **那 4 人时是什么**：B 的 8 项加总已是 37.5（3.0+5.0+11.0+3.5+1.5+2.5+6.0+5.0），与 §7.1 完全对上；这 4 人时是**未预先分配的余量**——数据缺陷返工、OOM 重跑、任务替换。别提前花掉。
