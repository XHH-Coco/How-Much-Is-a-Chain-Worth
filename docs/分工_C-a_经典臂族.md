# 分工 C-a · 经典臂族：A1 地板、A2 经典、A5 链信息

> **对应**：`docs/分工_C_经典臂与机制实验.md`（母本）的 **C-1**（5.0）、**C-2**（1.5）、**C-5 的"排行榜对齐·经典三列"**（1.0）
> **工时**：固定项 **7.5 人时** = 三份 7.5 / 7.5 / 7.0 里的第一份（合计 22.0，与母本一致）；**弹性余量 2.0 不进任何一份**，留在 C 组名下共用——谁那一批要重跑，从这里出
> **你负责的目录**：`src/chainworth/arms/{floor,conventional,chain_info}.py`、`configs/arms/{a1_floor,a2_conventional,a5_chain}.yaml`、`configs/experiments/main_table.yaml`、`results/raw_runs/`（`a1_*` / `a2_*` / `a5_*`）、`results/aggregate/predictions/`（同上）、`docs/Leaderboard Gap.md`
> **一句话**：全队不用显卡的那三台"秤"都在你这儿——你要用**一套骨架、三个开关**做出 A1/A2/A5，并让 A5 相对 A2 的增益**只能是"多了链信息"，不能是"换了个模型"**。

> **三人切分口径（三份文档一字不差地写着同一张表）**
>
> | 分工 | 拿的是母本哪几条 | 固定项 | 交给谁什么 |
> |:--|:--|--:|:--|
> | **C-a 经典臂族（本文件）** | C-1 5.0 + C-2 1.5 + C-5 的经典三列对齐 1.0 | **7.5** | 冻结的 A2 超参 → C-b；A1/A2/A5 的 run_id → C-c |
> | **C-b 规模与迁移前半** | C-3 4.0 + C-4 的物化与官方地板 2.0 + C-6 的批量执行 1.5 | **7.5** | 物化划分 + 组分特征 → C-c；退单台账 → C-c |
> | **C-c 迁移主跑、主表与审计** | C-4 的 30 runs 与 OOD 3.0 + C-5 定稿 1.5 + C-6 套件 2.5 | **7.0** | 主表 → D；审计结论 → 全队 |
>
> 切法只有一条原则：**谁的数字谁做，谁的接口谁定**。三人共用同一批特征、同一份划分、同一组模型种子 `seed0/1/2`——**谁的活都不许改这三样**。

---

## 0. 先用大白话讲：你这份差事是干什么的

### 你负责"三个开关"，不是"三套代码"

四个人合写一份实验报告，同一批样品、同一套划分、同一批种子，只换一个变量：

| 臂 | 换的是什么 | 大白话 | 谁跑 |
|:--|:--|:--|:--|
| **A1 地板** | 什么都不学 | 先量一下"瞎猜"有多差。**没有它，后面所有提升都没有参照** | **你** |
| **A2 经典** | 只给重复单元的描述符 | 标准秤：205 个 RDKit 描述符 + 2048 位指纹，配 XGBoost / 随机森林 | **你** |
| A3 图 | 只给图结构 + 消息传递 | 更贵的秤，必须 GPU | B（已交付 48 条） |
| A4 语言模型 | 只给预训练嵌入 | 别人校准过的秤 | B（已交付 48 条） |
| **A5 链信息（我们的方法）** | 在 A2 的输入里塞进链的自由度 | 给标准秤加一个"链长"刻度 | **你**（在 A2 上加列，不换模型） |

**A5 是全篇最容易被做歪的一项**：它的定义是"**改输入，不改模型**"。如果你在 A5 里顺手把 XGBoost 换成更好的模型，A5 相比 A2 的增益就分不清是"信息"带来的还是"模型"带来的——而 proposal 的 H3 就是"信息 vs 架构"这个对比。**A5 越界，H3 就作废。**

### 你还要回答"我们比官方差多少、差在哪"

主表里属于你的三列（A1/A2/A5），与 PolyBench 排行榜的差距由**你**解释：是协议不同（划分、指标、预处理），还是实现不同（调参、编码、随机源）。C-c 只负责把表拼起来，**他不会替你说这句话**——因为他没跑过。

### 做完的标志（一句话）

> **C-b 拿到你的 `a2_conventional.yaml` 与超参表就能不问你一句地开跑 scaling 与 OpenPoly；C-c 拿到你的 run_id 索引就能把 A1/A2/A5 三列填进主表；D 拿到 A2 与 A5 在同一 (任务, 折, 种子) 上的预测就能直接相减。**

---

## 1. 你在流水线上的位置

| 阶段 | A 数据（已完成） | B 表示与 GPU 臂（已完成） | **C-a（你）** | C-b / C-c |
|:--|:--|:--|:--|:--|
| P3 配置 | `configs/tasks/` 已冻结 18 份 | `a3/a4/openpoly_scan` 已冻结 | ● **主责**：`a1/a2/a5` + `main_table` | ○ 等你的超参 |
| P4 代码 | 划分/泄漏测试 | 装配 + A3 + A4 + B-5 | ● **主责**：三臂骨架 + 打乱对照 | ○ 等你的接口 |
| P5 运行 | 交数据 | 546 条已落盘 | ● **主责**：A1/A2/A5 全量 | ● C-b 跑三批机制 / C-c 跑 30 runs |
| P6 对账 | 答疑 | 已验收 | ● **主责**：三列的排行榜差异 | ● C-c 拼主表 |

● 主责　○ 参与/等待

**顺序不能反**：先把 A1+A2 在**一两个任务**上跑通 → 出第一批真数字给 D（他只用 A2 就能算出第一个 Λ，整条链就通了）→ 再放大到 8 个任务 → **最后**才做 A5。反过来做（先写 A5 再想汇总）会让你在最后一天才发现某个字段没落盘。

### 你手上的现成东西（不用碰 `data/raw/`，一个字节都不用）

| 你有什么 | 在哪 | 怎么用 |
|:--|:--|:--|
| 15 个任务的特征矩阵 | `data/processed/features/<task>.npz`（合计 11 MiB） | 每**测量**一行：`row_id` + `y` + `psmiles` + 205 描述符 float64 + 2048 位指纹 uint8；**行序 = `row_id` = `splits/` 的 `row_position` 同一坐标系** |
| 逐份哈希与列序说明 | `data/manifests/task_features_manifest.csv` | `pack_sha256` 逐份；读之前先核，核不过就停（别自己重算） |
| 结构层画像（A5 要用） | `data/processed/structures/psmiles.parquet`（18,152 结构 + wPSMILES） | 重复单元数、wPSMILES 权重从这儿取，**不要从 SMILES 里自己解析** |
| 官方划分 | `splits/polybench/*/official_70_30.json`（10 份）、`splits/polybench26/*/seed_42/`（PB26 5 折） | 只读。`official_70_30` 只有 `test_index`（无 val），PB26 的 `fold_NN.npz` 里有 `train_pool / val / test` |
| run 记录的字段契约 | `results/raw_runs/_schema.json` | **你的记录必须过这份 jsonschema**；`arm` 的枚举里 `a1_floor / a2_conventional / a5_chain` 都已就位 |
| B 已落的 546 条 | `results/raw_runs/*.json` | 抄格式、抄 `split_version` 写法（`<12 位 hex>@splits/...`），**不要抄数值** |

> **一条硬事实**：`data/processed/` 与 `data/raw/` 都已在 `.gitignore` 里，全靠阶段脚本重建。所以你**不需要**等任何人发数据包——特征已经在本机，`pack_sha256` 核过就能开工。

---

## 2. 任务清单（拿什么 → 怎么做 → 验收标准）

> 四项固定合计 **7.5 pH**。

### C-a-1 · 骨架、三份配置与超参预算表（1.5 pH｜关键，卡三个人）

| 项 | 内容 |
|:--|:--|
| **拿什么** | `docs/CODE_STYLE.md`、`results/raw_runs/_schema.json`、B-3/B-5 的两份臂配置当样板（`configs/arms/a3_graph.yaml`、`configs/experiments/openpoly_scan.yaml`） |
| **怎么做** | 一个 `fit_one(...)` 入口，把"臂"写成**输入构造 + 学习器**两段：A1 = 空输入 + {Dummy 均值, kNN}；A2 = 描述符+指纹 + {XGBoost, RF}；A5 = A2 的输入 + 链列 + {**与 A2 完全相同的** XGBoost, RF}。超参搜索用 **Murcko-grouped CV 且只在 train 内**，统一预算 **≤30 trials**，搜索空间与预算表写进各 yaml。**先把"主表单元格口径"定死**（见下方口径框），再写代码 |
| **产出** | `src/chainworth/arms/{floor,conventional,chain_info}.py`、`configs/arms/{a1_floor,a2_conventional,a5_chain}.yaml`、`configs/experiments/main_table.yaml`（含 `expected_runs`）、`results/aggregate/a2_best_params.csv`（一任务一学习器一行：`task,learner,best_params,cv_score,trials`） |
| **验收标准** | ① 三份 yaml 有搜索空间 + 预算表，**≤30 trials**，被问"调了多少次参"能立刻答；② 任一 run 记录过 `_schema.json`（jsonschema，不是肉眼比对）；③ `--limit` 冒烟只许写 `*.partial.*`，不许进 `results/raw_runs/`；④ 同 seed 连跑两遍，`r2` 差 **< 1e-9**（经典模型是确定性的，达不到说明有随机源没固定：`n_jobs`、`random_state`、pandas 排序都算随机源）；⑤ `main_table.yaml` 的 `expected_runs` 与后文口径框算出的一致 |

> **【口径必须先定死·母本已标注】** 母本 §7.1 写"150 fits"，C-1 又说"§6 写 96 classical + 48 GNN"。按现在的配置算：
> **评估单元 = 6 个 PolyBench 单折 + 2 个 PB26 × 5 折 = 16 个**；每个单元 × 3 臂 × 3 种子 = **144 条**。
> 与 150 差 6 条。差额只可能来自"哪个学习器进主表"（A1 要不要两个地板、A2/A5 用 XGB 还是 XGB+RF）。
> **建议口径**：主表每臂只放 **1 个学习器**（A1=Dummy、A2=XGBoost、A5=XGBoost），把 kNN 与 RF 记为"学习器敏感性"另存一张表、不进主表 → **144 条**。
> **不许倒推**：先把口径写进 `configs/experiments/main_table.yaml` 的 `expected_runs`，之后**所有文档只引这一个数**。

### C-a-2 · A1 地板 + A2 经典全量（3.5 pH｜关键，D 的图 1 靠它）

| 项 | 内容 |
|:--|:--|
| **拿什么** | 上面那 16 个评估单元与 15 份 npz；`splits/` 官方索引（只读） |
| **怎么做** | 16 单元 × 3 臂中属于你的 2 臂（A1/A2）× 3 种子；PolyBench 走 `official_70_30`（**只有 test**，没有 val——别自己切一个 val 出来调参，要调就在 train 内做 CV）；PB26 走 `fold_00..04` 的 `train_pool` 训练、`test` 评估（`val` 列存在但主表口径不用它，用了要写进 `main_table.yaml`）。每次落三样：run 记录 + 预测 parquet + 一行实测墙钟 |
| **产出** | `results/raw_runs/{a1_floor,a2_conventional}*.json`、`results/aggregate/predictions/<run_id>.parquet`（两列 `row_id,y_pred`）、`results/aggregate/{a1_a2_index}.csv`（`run_id,task,arm,learner,fold,seed,final_r2,n_train,n_test`，**给 C-c 的唯一入口**） |
| **验收标准** | ① 每 run 都有预测，**行数 = 该 split 的 `test_index` 长度**，`row_id` 与 split 文件逐位一致；② 同 seed 两遍 `r2` 差 < 1e-9；③ PolyBench26 两个任务上与官方 `RDKit_RF` 基线对比，同任务差距 **< 2σ**；> 2σ 先查协议（特征列、标签标准化、划分来源）再报数，**不许直接当"我们更好/更差"**；④ `device=cpu`、`env_fingerprint=results/env_fingerprints/env_core.json`（`_schema.json` 只认这三个文件）；⑤ D0 硬门：**至少 1 个任务**跑通并把粗数交给 D（他要用它算第一个 Λ） |

> **kNN 的坑**：不做标准化 = 结果极差 = 被解读成"经典方法不行"。距离度量前标准化，且**只在 train 上 fit**。同理任何编码器、任何缺失值填充，`fit` 只在 train 上做。

### C-a-3 · A5 三注入 + A5all + 打乱对照（1.5 pH｜关键，H3 的一半）

| 项 | 内容 |
|:--|:--|
| **拿什么** | A2 的输入 + `psmiles.parquet` 的 `wPSMILES` 与重复单元数；Proposal §6 的 A5 定义 |
| **怎么做** | 三条注入**各自可单独开关**：**A5a** 重复次数（重复单元数作为显式特征）、**A5b** wPSMILES 权重、**A5c** 多单元序列拼接；外加 **A5all**（三条全开）与**打乱对照**（把链列在 train 内随机置换后重跑）。学习器、超参、划分、种子**与 A2 完全相同**。分布：A5all 跑满 16 个单元（进主表），A5a/b/c 与打乱对照只在 **2 个代表任务**上跑全（`pb_t1_tg` 单折 + `pb26_polymetrix_tg` 5 折 = 6 单元） |
| **产出** | `src/chainworth/arms/chain_info.py`、`configs/arms/a5_chain.yaml`、四组 run 记录、`results/aggregate/scramble_control.csv`（一任务一臂一行：`with_chain / scrambled / delta / delta_after_scramble`） |
| **验收标准** | ① **同学习器同超参**：`a5_chain.yaml` 与 `a2_conventional.yaml` **解析后除输入段外逐键相同**（写成断言，不是肉眼比对）；② 记录里新增 `a2_config_hash` 字段，值 = `configs/arms/a2_conventional.yaml` 的字节 sha256（`_schema.json` 的 `additionalProperties: true` 允许加字段，**但只许加不许改**）；③ 只开一条也能跑（三个开关相互独立）；④ **打乱对照必须跑、必须如实报**：若打乱后增益没有下降，说明增益来自"多了几列"而不是"链信息"；⑤ 与 A3 的增益能在同 split/seed 上直接相减 |

> **为什么必须有打乱对照**：A5 相对 A2 多了几列，任何模型都会因为维度增加而略有变化。没有对照，你无法回答"这 0.02 是信息还是维度"。成本极低（一次重跑），但它是 H3 能不能站住的关键——**D 画图 4 时那根柱子必须有它**。

> **【字节哈希的两面】** 记录的 `config_hash` 是**配置文件字节哈希**，所以 `a5_chain.yaml` 的哈希**必然**不等于 `a2_conventional.yaml`（文件本身就不同）。母本那句"`config_hash` 除输入特征外必须与 A2 一致"**不能按字面实现**。正确做法是双写：`config_hash` = 本臂配置字节哈希（协议身份），`a2_config_hash` = A2 配置字节哈希（"我派生自哪一版 A2"）。断言的落点是"解析后除输入段外逐键相同"，不是"哈希相等"。

### C-a-4 · 排行榜对齐：经典三列（1.0 pH｜D 的图 1 与 Q&A）

| 项 | 内容 |
|:--|:--|
| **拿什么** | 你的 `a1_a2_index.csv` 与预测；PolyBench 排行榜的公开数（**记清是哪一版、哪个模型、哪种划分**） |
| **怎么做** | 逐个格算"我们 − 官方"，然后**分类**：差在**协议**（划分来源、指标定义、标签标准化、预处理）/ 差在**实现**（调参预算、编码方式、随机源）/ `unexplained`。差距 > 10% 的格必须给解释或老实标 `unexplained` |
| **产出** | `docs/Leaderboard Gap.md`（经典三列）+ 主表对应三列的值交给 C-c |
| **验收标准** | ① 每一格都能点回 `run_id`；② 官方数的来源三要素（版本 / 模型 / 划分）写清；③ **不许把"协议差异"和"实现差异"混成一类**（这是 Q&A 最容易抓的点）；④ 缺格留灰标 `blocked_no_gpu`，**不许填 0** |

---

## 3. 交付物清单（文件级）

```
configs/arms/{a1_floor,a2_conventional,a5_chain}.yaml     ← 三份，字节哈希一改就全批作废
configs/experiments/main_table.yaml                       ← 含 expected_runs（口径唯一出处）
src/chainworth/arms/{floor,conventional,chain_info}.py
results/raw_runs/<a1_*|a2_*|a5_*>.json                    ← 约 144 + 消融/打乱，入库
results/aggregate/predictions/<run_id>.parquet            ← 每 run 一份，两列 row_id,y_pred
results/aggregate/a2_best_params.csv                      ← 交给 C-b 的超参
results/aggregate/a1_a2_index.csv                         ← 交给 C-c 的入口
results/aggregate/scramble_control.csv                    ← 打乱对照，H3 的命门
docs/Leaderboard Gap.md                                   ← 经典三列的差异分类
```

**不入库**：模型权重（`.pkl` / `.joblib`）、`logs/`、`outputs/`。经典模型的权重可由 `config_hash` + seed 重建。

---

## 4. 没有 GPU 怎么办

**你一项都不需要 GPU**，这一份分工里 0 条 GPU 记录。一台普通笔记本能跑完：

| 任务 | GPU | 量级 |
|:--|:--|:--|
| C-a-1 骨架与调参 | 无 | 16 单元 × ≤30 trials，挂机 1–2 h |
| C-a-2 A1/A2 全量 | 无 | 约 144 条，单条 5–30 s（最大任务 8,055 行 × 2,253 列） |
| C-a-3 A5 四组 | 无 | 主表 48 条 + 消融 108 + 打乱 36，合计约 2 h |
| C-a-4 排行榜对齐 | 无 | pandas 级别 |

你被卡的方式只有一种：**A2 超参没定，C-b 的 scaling 与 OpenPoly 就开不了工**。所以 C-a-1 与 C-a-2 的"1 个任务跑通"是**当天必须交**的东西，别攒到最后。

---

## 5. 依赖关系与冻结边界

**你依赖**：A 的 `data/processed/features/`（已就位）与 `splits/`（只读）；B 的 `task_features_manifest.csv` 列序字典（已就位）。**你不依赖任何 GPU 产物**。

**依赖你的**：C-b（`a2_conventional.yaml` + `a2_best_params.csv`）；C-c（`a1_a2_index.csv` 与预测）；D（主表三列 → 图 1 与 H3 的一半）。

| 交接点 | 交付什么 | 对方拿它干什么 | 什么时候 |
|:--|:--|:--|:--|
| **→ C-b** | `configs/arms/a2_conventional.yaml`（冻结）+ `a2_best_params.csv` | scaling 与 OpenPoly 只换训练集大小/子集，**超参一个都不许动** | 11-13 收工前 |
| **→ C-c** | `a1_a2_index.csv` + 全部预测 parquet | 填主表三列、算配对差 | 11-16 |
| **→ C-c** | `scramble_control.csv` | H3 的证据链；主表备注列 | 11-17 |
| **→ D** | 第一个任务的 A2 真数字 | 算第一个 Λ，打通整条链 | 11-13（硬门） |

| 冻结项 | 冻结在 | 谁能改 |
|:--|:--|:--|
| 超参空间与预算（≤30 trials） | `configs/arms/{a1,a2,a5}.yaml` | 只能整体换版本 + 全表重跑，不能"这个任务多调两次" |
| 臂的定义 | `configs/arms/` | A5 只能改输入，不能换学习器（否则 H3 作废） |
| 主表单元格口径与 `expected_runs` | `configs/experiments/main_table.yaml` | 定死后只引这一个数 |
| 主表列名 | `results/aggregate/main_table.csv` | 只加不改（D 的对账脚本按列名读） |

> **全队只有一张冻结登记表**：`README` 第九节（版本号、冻结日期、怎么验、当前状态）。三份臂配置**登记完才算冻结**；冻结物被改动 = 该阶段重做。

---

## 6. 时间线与里程碑（剩余 5 个工作日，11-19 汇报）

| 时间 | 你要交的 | 卡点 |
|:--|:--|:--|
| **11-13（五）** | 三份 yaml + `expected_runs` 定死；A1/A2 在 **1 个任务**上跑通，第一个基线数交给 D | **硬门**，卡 C-b 开工 |
| **11-14~15（周末）** | A1/A2 全量挂机（16 单元 × 2 臂 × 3 种子）；超参表定稿交 C-b | 无人值守也要能续跑 |
| **11-16（一）** | A1/A2 全量收口 + `a1_a2_index.csv` 交 C-c | D 的图 1 靠它 |
| **11-17（二）** | A5 all 全量 + A5a/b/c 消融 + **打乱对照** | H3 的证据 |
| **11-18（三）** | `Leaderboard Gap.md` 三列写完；主表三列的值确认进表 | Q&A 会问"差在哪" |
| **11-19（四）** | 冻结、答疑、补格 | 交付 |

> **别把 144 条堆到最后一天**：它们是"可并行但总量大"的活。正确姿势是**每跑完一批就落盘一批、汇总一批**，而不是"全部跑完再写汇总脚本"。

---

## 7. 红线与常见坑

**红线（做了等于重做）**

1. **自己切数据**：评估必须读 `splits/` 里的官方索引。`train_test_split` 出现在评估路径上就是红线（调参可以用，但必须在 train 内部）。
2. **A5 换模型**：A5 的唯一定义是"同学习器、不同输入"。换了学习器，H3 直接从"信息 vs 架构"塌成"信息 vs 我喜欢的模型"。
3. **用测试集调参**：超参搜索只能在 train 内做 Murcko-grouped CV，一次都不许碰 test。
4. **run 记录手写**：记录只能由训练脚本落盘，手改等于伪造。
5. **主表填 0 充缺格**：缺就留灰并标 `blocked_no_gpu`。
6. **打乱对照结果不好看就删掉**：这条对照的意义就是"可能打脸"，删了 H3 就没有立锥之地。

**常见坑**

| 坑 | 现象 | 正确做法 |
|:--|:--|:--|
| kNN 不做标准化 | 结果极差，被解读成"经典方法不行" | 距离度量前标准化，且只在 train 上 fit |
| `X` 里混入全量统计量 | 泄漏，指标好得不像话 | 标准化/编码器只在 train 上 fit |
| 特征行序按自己排 | 预测对不上 `row_id`，指标变噪声 | 行序 = `row_id` = split 的 `row_position`，**不排序、不重排、不删行** |
| 没核 `pack_sha256` 就读 npz | 用了别人改过的特征，全表作废 | 读之前核 manifest 里的哈希，核不过就停 |
| 只存指标不存预测 | D 算不出 CI，只能报点估计 | 每 run 落 `predictions/<run_id>.parquet` |
| `--limit` 的产物混进全量 | 主表里混进调试数 | 只许 `*.partial.*`，审计会抓（C-c 的测试会红） |
| A5 的增益只看均值 | 分不清"信息"与"维度" | 打乱对照跑起来 |
| 拿 leaderboard 数直接对比 | 协议不同，被 Q&A 抓住 | 注明版本/模型/划分，> 10% 标 `unexplained` |

**代码怎么写**：按 [`docs/CODE_STYLE.md`](CODE_STYLE.md)——中英双语注释，每个函数写清输入输出、数据来源、配置来源，数字必须有来源。审阅时不看风格，只看"能不能从注释追到数据来源"；追不到的退回重写。

---

## 8. 交接前自检（逐条打勾再 push）

- [ ] `git status` 里没有模型权重、没有 `logs/`、没有 `.ipynb_checkpoints`、没有 `*.partial.*`
- [ ] `pytest tests/` 全绿（基线 280 passed / 7 skipped，只许多不许少）
- [ ] 每条 run 都过 `results/raw_runs/_schema.json`（用 jsonschema 跑，不是肉眼）
- [ ] 每条 run 都有对应 `predictions/*.parquet`，行数 = 该 split 的 `test_index` 长度
- [ ] 同 seed 两遍 `r2` 差 < 1e-9（留一条凭据 JSON，像 `a3_repeatability.json` 那样）
- [ ] `a5_chain.yaml` 与 `a2_conventional.yaml` 除输入段外逐键相同（断言过）
- [ ] 记录里的 `a2_config_hash` = `a2_conventional.yaml` 的字节 sha256（逐条核一遍）
- [ ] 打乱对照跑过了，数字与一句话结论写进 `scramble_control.csv`
- [ ] 三份 yaml 的预算表 ≤30 trials，且与 README 第九节的登记一致
- [ ] `expected_runs` 与实际条数相等（差一条就说明有格没跑或跑了没落盘）
- [ ] `Leaderboard Gap.md` 里每一格都能点回 `run_id`，差异已分类
- [ ] 提交信息写清"做了什么 / 没做什么 / 下一步是什么"（README 第十节）

---

## 9. 工时口径核对

| 项 | pH |
|:--|--:|
| C-a-1 骨架、三份配置与预算表 | 1.5 |
| C-a-2 A1 地板 + A2 经典全量 | 3.5 |
| C-a-3 A5 三注入 + all + 打乱对照 | 1.5 |
| C-a-4 排行榜对齐（经典三列） | 1.0 |
| **合计（固定项）** | **7.5** |

**7.5 是怎么来的**：母本 C-1 的 5.0（= 1.5 骨架 + 3.5 主跑）+ C-2 的 1.5 + C-5 的 2.5 里划 1.0 给"差异分类" = 7.5。C-5 剩下的 1.5 在 C-c（汇总与定稿）。

> **弹性余量 2.0 不在你名下**：它是 C 组的共用池。你的风险不在算力，而在**"A2 超参没定死，两个人就得干等"**——所以余量优先花在"某一批要重跑"，其次花在"把 A2 的第一版数字提前到 11-13"。
