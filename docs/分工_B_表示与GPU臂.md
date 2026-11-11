# 分工 B · 表示与 GPU 臂（Representations & GPU Arms）

> **对应**：`docs/Proposal_HowMuchIsAChainWorth_v1.3.md` §6（实验设计）、§7.1 的负责人列 `B`、§9 的 `B — GPU arms`
> **工时**：固定项 **21.5 人时** + 弹性余量 **2.5** = **24.0 人时**
> **你负责的目录**：`src/chainworth/{chem,arms}/`（A3 / A4 部分）、`configs/arms/{a3_graph,a4_polybert}.yaml`、`configs/experiments/openpoly_scan.yaml`、`results/raw_runs/`（GPU 部分）、`results/aggregate/predictions/`（GPU 部分）、`data/processed/polybert/`
> **一句话**：全队只有你一个人排队用显卡，所以你把所有要卡的活打包排进夜里，白天做装配和调试。
>
> **四人均分口径**：全队技术工时 96 pH（原 105 pH 减去写作与汇报的 9 pH，那 9 pH 由仓库所有者承担），四个人各 **24.0 pH**，固定项落在 21.5–22.0，极差 0.5 pH，差额是各自的弹性余量。

---

## 0. 先用大白话讲：你这份差事是干什么的

### 你要干的活，是"开两台要大电的仪器，而且钥匙只有你一个人有"

四个人合写一篇实验报告（写报告本身由仓库所有者负责，不在四个人的分工里）：

- **A** 把样品配好、编号贴好、天平校准，并在记录本上签好字；
- **你** 手上是两台**最耗电**的仪器：一台图神经网络（A3），一台语言模型（A4）。它们只能夜里开，白天你要做的是接线和试机；
- **C** 用不用电的那几台仪器测同样的样品，然后把所有读数汇总成一张主表；
- **D** 拿表上的数字算结论、画图，并且审你交出去的记录。

先把"臂"这个字说清楚：**臂（arm）不是四个不同的模型在比赛，而是同一批样品、同一套划分、同一批种子，只换一个变量**：

| 臂 | 换的是什么 | 大白话 | 谁跑 |
|:--|:--|:--|:--|
| **A1 地板** | 什么都不学 | 先量一下"瞎猜"有多差 | C |
| **A2 经典** | 只给重复单元的描述符 | 标准秤：RDKit 描述符 + XGBoost / 随机森林 | C |
| **A3 图** | 只给图结构和消息传递 | 更贵的秤：官方 polymer_chemprop，**必须 GPU** | **你** |
| **A4 语言模型** | 只给预训练嵌入 | 别人校准过的秤：polyBERT | **你** |
| **A5 链信息（我们的方法）** | 在 A2/A3 的输入里塞进链的自由度 | 给秤加一个"链长"刻度 | C（在 A2 上注入） |

**你只负责 A3 与 A4 两条臂**，但这恰恰是 proposal 里"架构"那一列的支柱：H3（信息 vs 架构）要拿 A5 的增益去减 A3 的增益，**如果 A3 的数不可信，H3 就没得算**。所以你的活不是"跑完就行"，是"跑出来的每条记录都能被 D 直接拿去做区间估计"。

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
  "env_fingerprint": "results/env_fingerprints/env_gnn.json"
}
```

加一份**测试集预测**（`predictions/<run_id>.parquet`，两列：`row_id, y_pred`）。**没有预测文件，D 就没法算 bootstrap 区间**——只能拿一个点估计算结论，等于白跑。

### 做完的标志（一句话）

> **D 拿到你的 `results/raw_runs/` 与 `predictions/` 之后，不需要再问你任何一句话，就能算出 Λ、κ 和它们的置信区间。**
> 也就是说：每次运行都留了 run 记录、留了预测、留了 split 版本、留了配置哈希、留了实测墙钟。

---

## 1. 你在流水线上的位置

| 阶段 | A 数据 | B（你） | C CPU 批量 | D 判据与图 |
|:--|:--|:--|:--|:--|
| **P0 环境** | ● 装包锁版本 | ○ 预取 polyBERT 权重 + `polymer_chemprop` 跑通一折 | ○ 写 A1/A2 骨架 | ○ 合成数据写指标单测 |
| **P1 获取数据** | ● 下载 + 完整性 | – | ○ 写超参空间 | ○ 把 Λ/κ 代码写完 |
| **P2 清洗与划分** | ● 调和 + 划分快照 | ○ 写装配，等 A 的产物 | ○ 等特征 | ● 指标与 Λ 计算器 |
| **P3 配置** | ● `configs/tasks/` 冻结 | ● **主责**：`a3_graph.yaml`、`a4_polybert.yaml`、`openpoly_scan.yaml` | ● `a1/a2/a5.yaml` + 两个机制实验配置 | ○ 汇总表契约 |
| **P4 代码** | ○ 划分/泄漏测试 | ● **主责**：B-1 装配、A3、A4 | ● A1/A2/A5 + 三批实验入口 | ● Λ/κ/CI/图 |
| **P5 运行** | ○ 交数据 | ● **主责**：GPU 批量（48 + 225） | ● **主责**：CPU 批量（100 + 225 + 60） | ○ 等 run 记录 |
| **P6 图表与对账** | ○ 答疑 | ○ 补跑与答疑 | ● **主责**：主表 | ● **主责**：5 图 + 结论 + 审计 |

● 主责　○ 参与/等待

**你是全队唯一的 GPU 队列**，所以你的排期就是全队的排期：主扫 48 → 隔一夜 → OpenPoly 225。**白天不要占卡跑长任务**，夜里一次跑完再看结果。

> 不要等 8 个任务的数据全齐了才开始写第一行代码。**先用 `--limit` 在 1 个任务 1 折上把管线跑通**，再放大——这一步在 P0 就能做，用的是 A 的 1 个任务。

### 与 A、C 的边界（划清，避免重复劳动）

| 谁 | 产出什么 | 放在哪 |
|:--|:--|:--|
| **A** | 结构级：规范化 PSMILES、RDKit 描述符矩阵、Morgan 指纹 | `data/processed/{structures,descriptors}/` |
| **你** | polyBERT 嵌入（原属 A，现归你）+ 任务级装配：`X`、图对象 | `data/processed/{polybert,features}/`、`src/chainworth/chem/` |
| **C** | A1/A2/A5 三条臂的 run 记录 + 三批机制实验 + 主表 | `results/raw_runs/`（CPU 部分）、`results/aggregate/main_table.csv` |

**一句话分工**：A 保证"每个结构算对一次"，你保证"每个任务的每一行对上正确的结构，并且行序由 split 文件决定"，C 保证"每个能不用卡算的数都算出来了"。

---

## 2. 任务清单（拿什么 → 怎么做 → 验收标准）

> 每项标了 pH（人时，含 agent 辅助）与是否关键路径。**5 项固定合计 21.5 pH**。
>
> **两处安排先说清楚**：① polyBERT 嵌入（2.0 pH）在 A 那边，现在划到你这里——它和 A4 臂共用同一套 7.3 GB 权重缓存与 `polybert` 环境，分两个人做只会重复下载、重复踩环境坑；② A1/A2/A5 三条臂与全部 CPU 批量在 C 那里，你专心做 GPU 那两条。

### B-1 · 表示装配与图构建（3.0 pH｜关键）

| 项 | 内容 |
|:--|:--|
| **拿什么** | A 交的 `data/processed/structures/psmiles.parquet`、`descriptors/<task>.parquet`、你算的 `polybert/{embeddings.npy,index.parquet}`；`configs/tasks/*.yaml`；`splits/`（只读） |
| **怎么做** | ① 按 `row_id`（不是 DataFrame 行序）join 结构级特征，得到任务级 `X`；② 构建 PyG `Data` 对象（原子特征、键特征、wPSMILES 权重作为节点/边权重）；③ 复现 chemprop 的 polymer 模式图构建，**用官方脚本对拍一次**，不要凭记忆写 |
| **产出** | `src/chainworth/chem/{features.py,graph.py}`、`src/chainworth/io/loaders.py`、`data/processed/features/<task>.npz` + `feature_manifest.csv`（列数、指纹位数、缺失数） |
| **验收标准** | ① 行数 = 该任务调和后可用标签数，且**行序由 split 文件决定**（断言：`X.index` 与 `splits/…/test.json` 的索引逐位相同）；② 随机抽 100 个结构，特征向量与 A 的描述符矩阵逐位相等；③ 同一结构两次构图 `edge_index` 完全相同（原子排序稳定，不许依赖字典序）；④ 特征矩阵里**不含任何用全量数据算的统计量**（标准化只在 train 上 fit）；⑤ 解析失败的结构计数并列出 ID，不许静默 `dropna` |

> **红线**：`X` 的行序如果按 DataFrame 顺序走，划分就失效了——而代码不会报错。这是 §7.5 点名的第 (i) 类失败模式。
>
> 【接口】 C 会复用你的 `feature_manifest.csv` 与装配函数跑 A2/A5。**装配函数一旦被 C 引用就冻结**，改字段要提前一天说。
>
> 【状态】 **已交付 2026-11-12**：10 个 PolyBench 任务 × 20,660 行 × 2,253 列，行序由划分文件决定，
> 图（72/14）与 chemprop 2.1.0 逐位对拍 200 例。产物是 `scripts/03b_assemble.py` +
> `data/processed/features/<task>.npz`（不入库）+ `data/manifests/{task_features_manifest,graph_manifest}.csv`，
> 五条验收全过。凭据见 [`交付_P2_B1_表示装配.md`](交付_P2_B1_表示装配.md)。
> **没做的两条**：PolyBench26 / OpenPoly 的装配（它们的标签不在 A-3 调和表里，是 B-3 开工前的第一件事）、
> A3/A4 的训练代码（B-3/B-4）。清单命名从分工书的 `feature_manifest.csv` 改为
> `task_features_manifest.csv`——A-6 已有一份 `features_manifest.csv`，两份同名会互相覆盖（D-29）。

### B-2 · polyBERT 嵌入（2.0 pH｜GPU 建议，CPU 可跑）

| 项 | 内容 |
|:--|:--|
| **拿什么** | 42k PSMILES（或 A 去重后的结构集）+ `environment/polybert.yml`；需从 HuggingFace 取权重（缓存约 **7.3 GB**） |
| **怎么做** | 批量前向，输出定长向量；**用 `--limit` 支持小样本试跑**。GPU 0.5–1.5 h；**纯 CPU 约 6–10 h，可以挂着跑一晚** |
| **产出** | `data/processed/polybert/embeddings.npy` + `.../index.parquet`（行号 ↔ 结构 ID）+ manifest（维度、覆盖数、模型版本） |
| **验收标准** | ① 嵌入行数 = 输入结构数，**顺序对齐可校验**（抽 50 行用结构 ID 反查一致）；② ndim 与模型一致并记录；③ 网络断了可续跑（分片落盘） |

> 【注意】 这项 **CPU 也能做完**，只是慢。没有 GPU 就挂着过夜，不要把自己卡死在这儿，也不要把 `--limit 200` 的试跑结果当成全量交上来。
>
> 【为什么在你手上】 它和 A4 臂是同一套权重、同一个环境、同一批 PSMILES。放在 A 那边，你和 A 会各自下载一次 7.3 GB、各自踩一次 transformers 版本坑。

### B-3 · A3 官方图神经网络主扫（11.0 pH｜**GPU 必需**｜关键）

| 项 | 内容 |
|:--|:--|
| **拿什么** | B-1 的图对象、`environment/gnn.yml`（`chemprop`、`torch`、`torch-geometric` 版本以 yml 为准）、A 的划分 |
| **怎么做** | 官方 `polymer_chemprop`：depth 3、hidden 300、batch 64（12 GB 上 OOM 就改 32 + 梯度累积 2，墙钟 ≈1.2×）、lr 1e-3、≤100 epoch、early stopping 15。**共 48 runs** = PolyBench 6 任务 × 3 种子（18）+ PolyBench26 2 任务 × 5 折 × 3 种子（30）。单 run 1–8 min，合计 **3–5 GPU-h**，一夜跑完 |
| **产出** | `configs/arms/a3_graph.yaml` + 48 条 run 记录 + 48 份预测 + `results/aggregate/gpu_hours.csv`（每 run 实测墙钟） |
| **验收标准** | ① **D0 24:00 前**单任务单折跑通（硬门，§7.2）；② 每条 run 的实测耗时 ≤ §7.4 预测值的 **1.5×**，超标立即走 triage (i) 而不是等；③ 记录里带 GPU 型号 + 驱动版本 + `torch.version.cuda`；④ 同 seed 两次差异 **< 1e-3**（GPU 非确定性允许，但要开 `cudnn.deterministic` 并记录；不许只写"大致一致"）；⑤ `torch.cuda.set_per_process_memory_fraction(0.9)`，一个进程一张卡 |

> **为什么值得单列 11 pH**：真正花时间的不是 48 次运行，而是把官方图管线调通（featurizer 版本、wPSMILES 权重怎么进图、early stopping 的监控量选哪个）。**48 次运行省不下多少，管线调错一次就是半天。**
>
> 【本机】 **一张 `NVIDIA GeForce RTX 4070`，12,282 MiB（12 GB），驱动 591.86，capability 8.9，46 个 SM**（实测于 2026-11-12，D-30；P0 指纹里那张「Laptop 8 GB」录错了，已重录）。所以上一段按 12 GB 写的降级梯**照用不改**：`hidden 300 + batch 64` 先试，OOM 再 `batch 32 + 累积 2`（墙钟 ≈1.2×）。注意 `set_per_process_memory_fraction(0.9)` ≈ 11.0 GB 比实测空闲可用量（10.6 GB）还高，它不是 OOM 保险。

### B-4 · A4 polyBERT 迁移臂（3.5 pH｜GPU 建议）

| 项 | 内容 |
|:--|:--|
| **拿什么** | 你自己的 `polybert/embeddings.npy` + `index.parquet`（42k 结构，768 维左右）；`environment/polybert.yml` |
| **怎么做** | **两种模式都要写**：(a) **冻结嵌入 + ridge 头**（几十分钟，CPU 也能跑）；(b) **微调**（2.5–4 GPU-h）。默认跑 (a)，时间够再跑 (b) |
| **产出** | `configs/arms/a4_polybert.yaml` + 两条（或一条）run 记录 + `docs/DECISIONS.md` 里一条决策记录：为什么跑/不跑微调 |
| **验收标准** | ① 嵌入行与结构一一对应（抽 50 个结构反查，见 B-2）；② (a) 与 (b) 若都跑了，同一 split/seed 上比较，差异写进主表备注；③ 若只跑 (a)，**必须**在决策记录里写明触发了 §7.4 的 triage (ii)，并在汇报材料上标注——**不许悄悄降级** |

### B-5 · OpenPoly 扫描的 GPU 部分（2.0 pH｜**GPU 必需**｜关键）

| 项 | 内容 |
|:--|:--|
| **拿什么** | A-7 复核过的 OpenPoly 三性质（Tg 443 / 介电常数 295 / Tm 168）+ 官方 5 折；B-3 已调通的 GNN 管线与配置 |
| **怎么做** | 3 性质 × 5 个 n 档（30 / 60 / 120 / 240 / all）× 5 折 × 3 种子 = **225 runs**。复用 B-3 的配置，**只改训练集大小**，别的都不动；排在 P2 那一夜，**不许和主扫抢卡**（4–11 GPU-h） |
| **产出** | 225 条 `results/raw_runs/` + 225 份预测 + `results/aggregate/openpoly_gpu_index.csv`（`run_id, property, n_level, fold, seed, wall_clock_s`） |
| **验收标准** | ① 225 条齐全，缺的写 `blocked_no_gpu` 并说明是哪条；② 一律 **n ≤ 443**，不许外推；③ 每 run 都落 `predictions/<run_id>.parquet`（D 要拿它算 bootstrap 与过零点，没有预测这 225 条等于白跑）；④ 实测墙钟写进索引，超 §7.4 预测 1.5× 就立刻报 |

> 【边界】 同一网格上的 **经典臂 225 fits 归 C**（`分工_C` 的 C-3）。你只管 GNN 这 225 条。两批用的是**同一套 n 档与折索引**，所以谁先跑完都别改档位——档位是 A 冻结的。

---

## 3. 交付物清单（文件级）

```
configs/arms/a3_graph.yaml  a4_polybert.yaml
configs/experiments/openpoly_scan.yaml
src/chainworth/chem/features.py  graph.py
src/chainworth/arms/{graph,polybert}.py
src/chainworth/io/loaders.py
data/processed/polybert/embeddings.npy + index.parquet + manifest
data/processed/features/<task>.npz + feature_manifest.csv
results/raw_runs/<run_id>.json                       GPU 的 273 条，入库
results/aggregate/predictions/<run_id>.parquet        预测，入库（小文件）
results/aggregate/{openpoly_gpu_index,gpu_hours}.csv
docs/DECISIONS.md（A4 微调那条决策）
docs/RUN_REQUESTS.md（没有卡时才需要填）
```

**不入库**：模型权重（`.pt`/`.ckpt`）、`logs/`、`outputs/`、HuggingFace 缓存（`~/.cache/huggingface`）。权重可由 `config_hash` + seed 重建，没有入库价值。

---

## 4. 没有 GPU 怎么办

**你这一份是全队最依赖显卡的**：5 项里有 2 项 GPU 必需、2 项 GPU 建议。所以"没有卡"这件事必须在**开工第一天**说清楚，而不是等到 D1 晚上。

| 任务 | GPU | 没卡的处置 |
|:--|:--|:--|
| B-1 表示装配 | 无 | 全做完（这是你和 C 的共同底座，**没有卡也必须完成**） |
| B-2 polyBERT 嵌入 | 建议 | 挂着过夜自己跑完（CPU 6–10 h）；或交代码 + 请求单 |
| **B-3 GNN 主扫 48 runs** | **必需** | 交代码 + 交请求，由仓库所有者代跑 |
| B-4 polyBERT 迁移臂 | 建议 | 只跑 (a) 冻结 + ridge（CPU 可完成），微调写进请求单 |
| **B-5 OpenPoly 225 GNN runs** | **必需** | 交代码 + 交请求（C 会同时用 CPU 臂跑同一网格） |

### 交接协议（三条，缺一条不算交接完成）

1. **代码能在 CPU 上跑通一小片**：所有训练入口必须有 `--limit`（或 `--tasks/--folds` 子集参数）与 `--dry-run`，CPU 上 2 分钟内跑完一片并打印形状与指标。**没通过 smoke 的代码不要提交请求单**，那只会让代跑的人替你调错。
2. **不许写死 GPU**：`device = "cuda" if torch.cuda.is_available() else "cpu"`；CPU 下能跑完（慢无所谓）。`torch.cuda` 相关调用必须包在 `if device == "cuda"` 里。
3. **在 `docs/RUN_REQUESTS.md` 里一行一条**，字段固定：

```
| run_id 规则 | 命令 | 预计 GPU-h | 依赖的 config_hash | 期望产出路径 | 提交时间 |
```

例：`pb_t1_tg__a3_graph__seed{0,1,2} | python -m chainworth.train --config configs/arms/a3_graph.yaml --task pb_t1_tg --seeds 0 1 2 | 0.2 | sha256:1a9d… | results/raw_runs/ | 11-13 20:00`

### 交接后你要做的（远程验收，不是甩手）

代跑完成后，run 记录会回到 `results/raw_runs/`。**你仍然负责验收**：

- [ ] 条数对不对（273 条 = 48 + 225，缺一条都要说清是哪条、为什么）
- [ ] 每条记录里 `device` 是 `cuda`、`env_fingerprint` 指向 `env_gnn.json`
- [ ] 预测文件行数与 split 测试集行数一致
- [ ] 随机抽 3 条，用同一命令在 CPU 上 `--limit` 重跑，指标量级合理（不要求相同，只要求不是明显坏掉）
- [ ] **发现异常就报，不要"帮忙"改数**

> **不许做的事**：手写 JSON 充数、把 `--limit 200` 的试跑产物改名成全量、把 CPU 跑的结果标成 `cuda`。占位请写 `{"status": "blocked_no_gpu", "requested_at": "…", "command": "…"}`，让主表能区分"没跑"和"跑了没成"。

---

## 5. 依赖关系与冻结边界

**你依赖**：A 的 `data/processed/{structures,descriptors}`、`configs/tasks/`、`splits/`（全部是只读输入）。
**依赖你的**：C（你的装配函数与 `feature_manifest.csv`，他跑 A2/A5 要用；A3 的 `config_hash` 他跑机制实验要复用）、D（run 记录 + 预测 → Λ/κ，没有你的预测他算不出区间）、仓库所有者（代跑与汇报素材）。

| 冻结项 | 冻结在 | 谁能改 |
|:--|:--|:--|
| 超参空间与预算（≤30 trials） | `configs/arms/*.yaml` | 只能整体换版本并全表重跑，不能"这个任务多调两次" |
| run 记录格式 | `results/raw_runs/_schema.json`（B-1 开工前定） | 只能加字段不能改字段语义，改要迁移旧记录 |
| 装配函数与行序规则 | `src/chainworth/io/loaders.py` | 被 C 引用后冻结；改要提前一天说 |
| 随机种子集合 | `configs/experiments/` | `{0,1,2}`；triage 时只许砍到 `{0}` 并在文档标注 |

> **全队只有一张冻结登记表**：`README` 第九节（版本号、冻结日期、怎么验、当前状态）。本节只列与你直接相关的那几项；口径冲突时以 README 为准。冻结物被改动 = 该阶段重做，不是"改一下就好"。

---

## 6. 时间线与里程碑（以 8 个工作日 / 11-19 汇报为准）

| 时间 | 你要交的 | 卡点 |
|:--|:--|:--|
| **D0 18:00** | B-1 管线在 1 个任务上跑通；`polymer_chemprop` 能在一折上开跑；polyBERT 权重下完 | 硬门（§7.2） |
| **D0 24:00** | **GNN 单任务单折跑完并记录耗时**（D 要用这个数判断 48 runs 来不来得及） | 硬门；跑不完就触发 §7.2 的降级判断 |
| **D1 12:00** | 主扫 48 runs 已启动，**实测每 run 耗时与 §7.4 对照**；超标立刻 triage (i) | 硬门 |
| **D1 24:00** | 48 条 run 记录齐全且自查过（字段、预测、device） | C 的主表能不能做就看这个 |
| **P2 结束** | OpenPoly 225 GNN runs 完成（与主扫**分两夜**，不许抢卡） | — |
| **P3 结束** | 交给 C 与 D 的 run 记录、预测**一条不缺**，缺的都有 `blocked_no_gpu` 占位 | 交付前 |

> **夜间是稀缺资源**：主扫在你手上那一夜，OpenPoly 扫描必须排在**另一夜**。有第二张卡时两件事可以并行。**别在白天起长任务**——那不是你的时间，是 C 的 CPU 批量和 D 写代码的时间。

---

## 7. 红线与常见坑

**红线（做了等于重做）**

1. **自己切数据**：评估必须读 `splits/` 里的官方索引。`train_test_split` 出现在评估路径上就是红线（调参可以用，但必须在 train 内部）。
2. **A3 偷换配置**：A3 是"官方 polymer_chemprop 的标准配置"，不是"我调到最好的一版"。为了好看而调，H3 就变成"信息 vs 我调的参数"。
3. **只在测试集上评估却用测试集调参**：超参搜索只能在 train 内做 Murcko-grouped CV，一次都不许碰 test。
4. **run 记录手写**：记录只能由训练脚本落盘，手改等于伪造。
5. **GPU 结果与 CPU 结果混报**：同一条 run 只允许一个 device，混着报的话没人能复现。

**代码怎么写**：P4 起所有提交的代码按 [`docs/CODE_STYLE.md`](CODE_STYLE.md)——中英双语注释，每个函数写清输入输出、数据来源、配置来源，数字必须有来源。审阅时不看风格，只看"能不能从注释追到数据来源"；追不到的退回重写。

**常见坑**

| 坑 | 现象 | 正确做法 |
|:--|:--|:--|
| 特征按 DataFrame 行序 join | 划分错位，R² 反而变好 | 用 `row_id` join，断言 `X.index == split.test_index` |
| `X` 里混入全量统计量 | 泄漏，指标好得不像话 | 标准化/编码器只在 train 上 fit |
| 12 GB 上 hidden 600 + batch 64 | OOM，跑到一半崩 | hidden 300、batch 32 + 梯度累积 2 |
| 两个进程抢一张卡 | 都变慢或都 OOM | 一进程一卡，`set_per_process_memory_fraction(0.9)` |
| 只存了指标没存预测 | D 无法算 CI，只能报点估计 | 每次运行落 `predictions/<run_id>.parquet` |
| 把 CPU 结果标成 cuda | 汇报时被问住，整张表作废 | `device` 字段由脚本自动填，人不可改 |
| 拿 leaderboard 数直接对比 | 协议不同，被 Q&A 抓住 | 注明版本/模型/划分（这条主要归 C） |

---

## 8. 交接前自检（逐条打勾再 push）

- [ ] `git status` 里没有权重文件、没有 `logs/`、没有 HuggingFace 缓存
- [ ] 所有训练入口都有 `--limit` 与 `--dry-run`，CPU 上 2 分钟能跑完一片
- [ ] `pytest tests/` 全绿，含 `test_splits_identity.py`、`test_no_leakage.py`
- [ ] run 记录条数对得上（48 + 225 = 273），缺的都有 `blocked_no_gpu` 占位
- [ ] 每份 run 记录都带 `split_version`、`config_hash`、`env_fingerprint`、`device`、`wall_clock_s`
- [ ] 每份 run 记录都有对应 `predictions/*.parquet`，行数与测试集一致
- [ ] `configs/arms/*.yaml` 里的预算表 ≤30 trials，且与 `docs/` 里的说明一致
- [ ] polyBERT 嵌入行数 = 结构数，抽 50 行反查一致
- [ ] `gpu_hours.csv` 与 `openpoly_gpu_index.csv` 的实测墙钟已写全
- [ ] `docs/RUN_REQUESTS.md` 里，所有需要代跑的项都写清了命令、预计耗时、期望产出
- [ ] `git status` 里没有 agent 自建目录（`.workbuddy/`、`.tools/` 等）或 `scripts/_scratch/`；有就自己补进 `.gitignore`（README 第五节硬规则 4）
- [ ] 提交信息写清"做了什么 / 没做什么 / 下一步是什么"（README 第十节）

---

## 9. 工时口径核对（全队共用一张表）

| 角色 | 固定项 | 弹性余量 | **合计** |
|:--|--:|--:|--:|
| A 数据与合规 | 21.5 | 2.5 | 24.0 |
| **B 表示与 GPU 臂（你）** | **21.5** | **2.5** | **24.0** |
| C 经典臂、机制实验与主表 | 22.0 | 2.0 | 24.0 |
| D 评测、图表与对账 | 21.5 | 2.5 | 24.0 |
| 合计 | 86.5 | 9.5 | **96.0** |

**你这 21.5 是怎么来的**：B-1 3.0 + B-2 2.0 + B-3 11.0 + B-4 3.5 + B-5 2.0 = 21.5。

> **你原来是 41.5（三人制时 32.0），现在 24.0，动的是哪几笔**：A1/A2/A5 三条臂（6.5）、scaling 与 OpenPoly 的 CPU 臂（4.0）、架构迁移（5.0）、主表（2.5）、运行记录审计（4.0）全部移出 → 剩 21.5；polyBERT 嵌入（2.0）从 A 移进来 → 净值不变；余量 2.5 → 24.0。
>
> **你为什么不留审计**：全队的 run 记录里 505 条是 C 跑的、273 条是你跑的。审计交给**用这些记录的人**——C 拿它们建主表，所以运行记录审计在 C 手上；你交完就接受他的抽查。**自己审自己等于没审**，这条比多背 4 pH 划算。
>
> **为什么你的余量只有 2.5**：你的主要风险是"GPU 排队"和"OOM"，这两种风险已经被排期吸收了（分两夜跑、可重试、降级梯）；真正需要临时补工时的是数据缺陷（归 A）和图与口径（归 D）。别把这 2.5 提前花在"再调一轮超参"上，那不属于返工，属于改协议。
