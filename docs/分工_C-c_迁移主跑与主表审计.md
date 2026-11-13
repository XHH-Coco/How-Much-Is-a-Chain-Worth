# 分工 C-c · 迁移主跑、主表与审计：30 runs、OOD 合成、8×5 总表

> **对应**：`docs/分工_C_经典臂与机制实验.md`（母本）的 **C-4 后三步**（3.0）、**C-5 的汇总定稿**（1.5）、**C-6 的审计套件**（2.5）
> **工时**：固定项 **7.0 人时** = 三份 7.5 / 7.5 / 7.0 里的第三份（合计 22.0，与母本一致）；**弹性余量 2.0 不进任何一份**，留在 C 组名下共用
> **你负责的目录**：`configs/experiments/arch_transfer.yaml` 的 `eval` 段、`results/aggregate/{arch_transfer.csv,arch_component_preds/,arch_ood/,main_table.csv}`、`tests/test_run_records.py`、`docs/{Main Table Notes.md,Audit Log.md}`（套件与结论部分）
> **一句话**：全队所有**结论的总入口**在你这儿——你要让 `main_table.csv` 的每一格都能点回一次运行，并让"数字的出生证明"这件事有测试守着。

> **三人切分口径（三份文档一字不差地写着同一张表）**
>
> | 分工 | 拿的是母本哪几条 | 固定项 | 交给谁什么 |
> |:--|:--|--:|:--|
> | **C-a 经典臂族** | C-1 5.0 + C-2 1.5 + C-5 的经典三列对齐 1.0 | **7.5** | 冻结的 A2 超参 → C-b；A1/A2/A5 的 run_id → 你 |
> | **C-b 规模与迁移前半** | C-3 4.0 + C-4 的物化与官方地板 2.0 + C-6 的批量执行 1.5 | **7.5** | 物化划分 + 组分特征 → 你；退单台账 → 你 |
> | **C-c 迁移主跑、主表与审计（本文件）** | C-4 的 30 runs 与 OOD 3.0 + C-5 定稿 1.5 + C-6 套件 2.5 | **7.0** | 主表 → D；审计结论 → 全队 |
>
> 切法只有一条原则：**谁的数字谁做，谁的接口谁定**。三人共用同一批特征、同一份划分、同一组模型种子 `seed0/1/2`——**谁的活都不许改这三样**。

---

## 0. 先用大白话讲：你这份差事是干什么的

### 你是三个人里唯一"要给结论交货"的那个

C-a 把三个秤校准了，C-b 把横轴拉开了，你收的是**两样东西**：

1. **架构迁移的 30 条 run**——这是全篇唯一"预测一个分子对在两个不同构型下的表现"的实验，也是 §5 架构排序的唯一来源；
2. **一张 8 任务 × 5 臂的总表**——D 画图 1、算 Λ、写结论全靠它。

外面还有一个**看门**的活：运行记录审计。母本把 4.0 pH 拆成两半，**套件由你写**（C-b 在各批收尾时跑、把不合格的记录退回去）。分开的理由是：写套件的人要拿这张表对账，跑套件的人不是写套件的人，**测试才不会为了"跑绿"而放松断言**。

### 最容易被做歪的一处：OOD 不是"直接预测"

那 36,828 行留出集**不是**"喂进去出 36,828 个预测再算 RMSE"。正确做法是官方三段式：

```
从 holdout 取全部唯一组分（psmiles_a ∪ psmiles_b）
   → 用该折该臂该种子训好的模型，对每个组分各预测一次（一个 psmiles 一条）
   → summarize_weighted_predictions.py 按 pred_a·f_a + pred_b·f_b 合成
   → 得 random / block / combined 各一个点值
```

**别绕开那个脚本自己写加权**：它会对「组分缺预测」「摩尔分数不合 1」「psmiles 重复」直接抛错——那三种错自己写就一定漏。

### 做完的标志（一句话）

> **D 拿到 `main_table.csv` 就能一个人把 Λ、κ、H1–H4 全算完，不需要回来问任何一句"这个格子是哪次跑的"；`pytest tests` 里那六条审计断言，故意注入错误时会红。**

---

## 1. 你在流水线上的位置

| 阶段 | A 数据（已完成） | B 表示与 GPU 臂（已完成） | C-a | C-b | **C-c（你）** |
|:--|:--|:--|:--|:--|:--|
| P3 配置 | 任务清单已冻结 | 三份配置已冻结 | `a1/a2/a5` | `scaling` / `openpoly` | ● `arch_transfer` 的 `eval` 段 |
| P4 代码 | 划分/泄漏测试 | 装配 + 图 + 嵌入 | 三臂骨架 | 档位与物化 | ● **主责**：30 runs + OOD 三段式 + 主表 + 审计套件 |
| P5 运行 | 交数据 | 546 条已落盘 | A1/A2/A5 | 60–80 + 225 | ● **主责**：30 runs + OOD 合成 |
| P6 对账 | 答疑 | 已验收 | 排行榜三列 | 审计执行 | ● **主责**：主表定稿 + 审计结论 |

● 主责　○ 参与/等待

**你的顺序**：审计套件**先写**（它是别人每天要用的东西，而且写完你自己立刻能校验手上的记录）→ 等 C-b 的地基（11-16）→ 30 runs → OOD → 主表粗版 → 定稿。

### 你手上的现成东西

| 你有什么 | 在哪 | 怎么用 |
|:--|:--|:--|
| 546 条 GPU 记录（B 已交付） | `results/raw_runs/*.json` + `results/aggregate/predictions/` | 主表的 A3/A4 两列；审计的第一批样本 |
| B 的三份索引 | `openpoly_gpu_index.csv`（450 行）、`openpoly_n_index.csv`（150 行）、`gpu_hours.csv`（546 行） | 主表填 OpenPoly 那部分的入口；**`gpu_hours.csv` 是共用表，不许整表求和**（D-37） |
| run 记录的字段契约 | `results/raw_runs/_schema.json` | 审计套件直接拿它跑 jsonschema，**别手写字段清单** |
| 架构迁移的任务配置 | `configs/tasks/pb26_arch_transfer_ea.yaml`（已冻结） | 行数 6,478 / 36,828、许可、折协议都写在这里；**但你只引用，不改字节**（改了就 30 条全作废） |
| 主表列名契约 | D 的对账脚本按列名读 | **只加不改**；粗版先给，CI 列留空由 D 回填 |

---

## 2. 任务清单（拿什么 → 怎么做 → 验收标准）

> 四项固定合计 **7.0 pH**。

### C-c-1 · 架构迁移 30 runs（ID 两档）（1.5 pH｜关键）

| 项 | 内容 |
|:--|:--|
| **拿什么** | C-b 的 `generated/shared_splits/`（canonical 6,478）+ `docs/Arch Materialization.md`（行数与地板凭据）+ `configs/experiments/arch_transfer.yaml` 的 `data` 段 |
| **怎么做** | 1 性质（EA）× 2 臂 × 5 折 × 3 种子 = **30 runs**。ID 评估**直接用每折 test 预测**（交替 **611** 行 / 均聚 **37** 行），不重训、不重采样。每次落三样：run 记录 + 预测 + 墙钟 |
| **产出** | `results/raw_runs/*__arch__*.json`、`results/aggregate/predictions/<run_id>.parquet`、`results/aggregate/arch_transfer.csv`（一 run 一行，`dataset` 列区分 alternating / homopolymer） |
| **验收标准** | ① **ID 两档分开报**（交替 611 / 均聚 37），不许合并成一个"ID"；② 每 run 有预测，行数 = 该折 test 长度；③ 同 seed 两遍逐位相同（经典臂是确定的）；④ 报告**实际排序**（block > random > alternating > homopolymer 是否成立），而不是只报"通过/不通过"；⑤ **排序不成立就照实写**，写进 `docs/DECISIONS.md` 并由 D 在结论里说——**不许重调超参再试** |

> **【口径冲突·必须当天定，不许两边都当没说】** `configs/tasks/pb26_arch_transfer_ea.yaml` 里 `arms: [a2_conventional, a3_graph]`，但母本 C-4 的 30 runs 是"**全程 CPU、你一项都不需要 GPU**"，而 GNN 臂是"可选、由 B 代跑、+4–7 GPU-h、不计入 24 pH"。
> 两句话不能同时成立：按配置跑，30 条里有 **15 条是 GPU**。
> **建议**：把 arch 的 30 条定为 **`a2_conventional` + `a5_chain`**（都在 CPU，且正好服务 H3 ——"同一 (任务, 折, 种子) 上 A5 与 A2 可相减"）。改臂 = 改冻结配置 = **换版本 + 全批重跑 + 登记 D-xx**，不是"顺手改一行"。开工前 30 分钟内与仓库所有者定死。

### C-c-2 · OOD 组分预测与官方三段式合成（1.5 pH｜关键，D 的置信区间靠它）

| 项 | 内容 |
|:--|:--|
| **拿什么** | C-b 的 `data/processed/arch/component_features.parquet` + `component_index.csv`（`psmiles → 特征行号`）；官方 `summarize_weighted_predictions.py`；36,828 行 holdout（含 `psmiles_a/psmiles_b/f_a/f_b` 与 random/block 标签） |
| **怎么做** | ① **先断言组分覆盖完备**：holdout 里每个唯一 `psmiles` 在 `component_index.csv` 里有且只有一行（缺就停，回去找 C-b，别自己补）；② 对 30 个 run 各预测一遍组分（每个 psmiles **一次**，不许重复）；③ 组件预测落成**官方要的两列** `psmiles,prediction`；④ 交给 `summarize_weighted_predictions.py` 合成 random / block / combined；⑤ **保留 5 折的 OOD 预测向量**（集成要用，D-09） |
| **产出** | `results/aggregate/arch_component_preds/<run_id>.csv`（两列，官方输入格式）、`results/aggregate/arch_ood/<run_id>.csv` + `..._metrics.json`、`results/aggregate/arch_ood_ensemble.json`（五折集成后的三档点值） |
| **验收标准** | ① 组分覆盖断言通过（缺一条就红）；② 只走官方脚本，**不许自己写加权**；③ **OOD 那一栏不得给"均值 ± 标准误"，不得对五折 OOD RMSE 做 t 检验**——折固定、不是独立重复；④ 五折预测向量全部留档（`arch_ood_vectors/<run_id>.parquet`）；⑤ 与官方地板（random 0.536 / block 0.581 / combined 0.559）的差写进 `arch_transfer.csv` 的备注 |

> 【口径警告·必读】 官方 README 原话：*The OOD holdout is fixed across folds, so fold metrics are not independent replicates. ... Do not apply a t-test to the five fold-level OOD RMSEs.* **正确做法**：保留 5 折的 OOD 预测向量 → 按聚合物取平均 → 从五折集成算出 random / block / combined 各**一个** RMSE/MAE/MSE。ID（交替 / 均聚）才允许按折汇总均值与总体标准差。这条同时约束 D 的置信区间：OOD 那栏给的是集成点值 + 重采样区间，不是"5 个折的均值 ± SE"。**做错了不会报错，只会给出一个假显著的结论。**

### C-c-3 · 主表汇总与定稿（1.5 pH｜关键，D 的图 1）

| 项 | 内容 |
|:--|:--|
| **拿什么** | 全部 run 记录 + 预测：B 的 546 条（A3 48 / A4 48 / OpenPoly 450）+ C-a 的 `a1_a2_index.csv` 与 A1/A2/A5 记录 + C-b 的 `scaling_cpu.csv` / `openpoly_cpu.csv` + 你自己的 30 条；PolyBench 排行榜公开数（由 C-a 提供差异分类） |
| **怎么做** | ① **先出粗版**：只填 `run_id` 与均值，CI 列留空，当天给 D（他画图 1 只用 A2 就能起步）；② 填满 **8 任务 × 5 臂 = 40 格**，每格记 `mean / std / n_runs / run_id 列表 / 缺格原因`；③ 与排行榜的差 C-a 已分类，你只做合并与格式；④ 缺格留灰标 `blocked_no_gpu`，**不许填 0** |
| **产出** | `results/aggregate/main_table.csv`（列名只加不改）、`docs/Main Table Notes.md`（口径、缺格说明、CI 由 D 回填的占位约定） |
| **验收标准** | ① 每一格都能点回具体 `run_id`（D 的对账脚本会自动查）；② 缺格是 `blocked_no_gpu` 而不是 0 或均值；③ 行数/格数对得上：8 × 5 = 40 格，缺几格写几条；④ 与排行榜的差**已分类**（协议 / 实现 / `unexplained`），>10% 的格必须解释或老实标；⑤ 粗版在 11-18 中午前发出，**不许等 CI 齐了再给**（那会把 D 绑在你的批量后面） |
| **记账纪律** | `gpu_hours.csv` 是**共用表**（A3 + A4 冻结 + B-5 混在一起），任何"某项花销"必须按 `arm` 或 `run_id` 归属筛，**不能整表求和**（D-37 的错法：整表 96 行 = 0.698，其中 0.696 是 A3，于是 B-4 被算成超预算 11%，实际 91%） |

### C-c-4 · 运行记录审计：套件与结论（2.5 pH｜关键）

| 项 | 内容 |
|:--|:--|
| **拿什么** | `results/raw_runs/_schema.json`（**字段契约的唯一出处**）、全队记录与预测、`splits/`（判"预测行数对不对"要用） |
| **怎么做** | 写 `tests/test_run_records.py`：**第 0 条**用 jsonschema 校验每条记录过 `_schema.json`；再六条断言：① 字段齐全（`run_id, task, arm, seed, split_version, config_hash, env_fingerprint`，`status=ok` 时另有 `metrics/wall_clock_s/device/predictions/n_train/n_test`）；② 每个 run 有对应预测文件；③ 预测行数 = 该 split 的 test 长度；④ 同一 `run_id` 不出现两次；⑤ `--limit` 试跑产物不得出现在全量目录（`*.partial.*` 一律不许进 `results/raw_runs/`）；⑥ `device` 与 `env_fingerprint` 对得上（`cuda` ↔ `env_gnn.json` / `env_polybert.json`，`cpu` ↔ `env_core.json`） |
| **产出** | `tests/test_run_records.py`、`docs/Audit Log.md` 的"运行侧结论"章节（条数、退单数、结论） |
| **验收标准** | ① **四处注入必须变红**：(a) 删掉一条记录的 `split_version`；(b) 把一份 `--limit 200` 的产物改名当全量；(c) 把 `device` 改成 `cuda` 但 `env_fingerprint` 仍指 `env_core.json`；(d) 复制一份记录让同一 `run_id` 出现两次。**四处都红才算写完**；② `pytest tests` 全绿，且**只许多不许少**（基线 **280 passed / 7 skipped**）；③ 每次批量结束记录条数，缺哪条说清哪条、为什么；④ 发现异常**退回**给产出人（交 C-b 执行），不许"帮忙"改数 |

> 【注意】 套件的价值不在"跑绿"，在于**你敢打回别人的记录**。`run` 记录错了不会抛异常，只会让结论对不上账。

---

## 3. 交付物清单（文件级）

```
results/raw_runs/<arch 30 条>.json
results/aggregate/predictions/<run_id>.parquet
results/aggregate/arch_transfer.csv                    ← ID 两档
results/aggregate/arch_component_preds/<run_id>.csv    ← 两列 psmiles,prediction
results/aggregate/arch_ood/<run_id>.csv + _metrics.json
results/aggregate/arch_ood_vectors/<run_id>.parquet    ← 5 折向量，集成用
results/aggregate/main_table.csv                       ← 40 格，列名只加不改
docs/Main Table Notes.md
docs/Audit Log.md（运行侧结论）
tests/test_run_records.py
```

**不入库**：`generated/`、模型权重、`logs/`、`outputs/`。

---

## 4. 没有 GPU 怎么办

**你一项都不需要 GPU**（前提：C-c-1 的臂口径按建议定成 `a2 + a5`）。

| 任务 | GPU | 量级 |
|:--|:--|:--|
| C-c-1 30 runs（ID） | 无 | 6,478 行 / 2,253 列，XGB 单条 10–40 s → 合计 15–25 分钟 |
| C-c-2 OOD 组分预测 | 无 | 30 个 run × 唯一组分数（一次一组分），**这是本份最重的一步**，先算一次组分特征再批量预测，1–2 h |
| C-c-3 主表 | 无 | pandas 级别 |
| C-c-4 审计 | 无 | 秒级，但**注入测试要手写四份坏样本** |

被卡的方式只有一种：**C-b 的地基没到（11-16）**。对策：11-13 就把套件写完（它不依赖任何人），11-16 当天先做 C-c-3 的粗表骨架（用 B 已有的 546 条就能先跑通一遍表结构），别干等。

---

## 5. 依赖关系与冻结边界

**你依赖**：B 的 546 条记录与三份索引；C-a 的 `a1_a2_index.csv` 与三列差异分类；C-b 的 `generated/shared_splits/`、组分特征、`arch_transfer.yaml` 的 `data` 段、`audit_backlog.csv`。

**依赖你的**：D（主表 → 图 1、Λ、CI；审计结论 → 能不能发布）；全队（审计套件是他们每天自查用的）。

| 交接点 | 交付什么 | 对方拿它干什么 | 什么时候 |
|:--|:--|:--|:--|
| **→ 全队** | `tests/test_run_records.py` | C-b 每批收尾时跑、退单 | 11-13 |
| **→ D** | `main_table.csv` 粗版（run_id + 均值，CI 空） | 画图 1 起步 | 11-18 中午 |
| **→ D** | `arch_transfer.csv` + `arch_ood_ensemble.json` | 架构排序、H4 | 11-18 |
| **→ D** | `main_table.csv` 定稿 | 算 Λ、写结论 | 11-19 |
| **← C-b** | `generated/`、组分特征、`arch_transfer.yaml` 的 `data` 段 | 你的 30 runs 全部输入 | 11-16 |

| 冻结项 | 冻结在 | 谁能改 |
|:--|:--|:--|
| 架构迁移的折协议与 holdout | `configs/tasks/pb26_arch_transfer_ea.yaml`（**已冻结**） | 改臂/改折 = 换版本 + 全批重跑 + 登记 D-xx，**不许顺手改一行** |
| OOD 的合成方式 | 官方 `summarize_weighted_predictions.py` | 谁都不许另写一套 |
| 主表列名 | `results/aggregate/main_table.csv` | 只加不改 |
| 审计断言的严格度 | `tests/test_run_records.py` | 只许加断言；为了让测试变绿而删断言 = 审计作废 |

> **全队只有一张冻结登记表**：`README` 第九节。`arch_transfer.yaml` 登记完才算冻结。

---

## 6. 时间线与里程碑（剩余 5 个工作日，11-19 汇报）

| 时间 | 你要交的 | 卡点 |
|:--|:--|:--|
| **11-13（五）** | `test_run_records.py` 写完（含四处注入全红）；`arch_transfer` 的臂口径定死 | 全队每天要用 |
| **11-14~15（周末）** | 主表骨架跑通一遍（用 B 的 546 条，先把列与格填对）；等 C-b 地基 | 别空转 |
| **11-16（一）** | 收到 `generated/` 与组分特征，**先断言组分覆盖完备**；30 runs 开跑 | 缺组分就当天退回 |
| **11-17（二）** | 30 runs 跑完 + OOD 三段式合成；`arch_transfer.csv` 出数 | 排序不成立就照实写 |
| **11-18（三）** | **主表粗版中午前给 D**；晚些出定稿（CI 由 D 回填） | D 的图 1 |
| **11-19（四）** | 审计结论章节 + 条数对账；冻结、答疑 | 交付 |

> **主表粗版一定要先出**：D 的进度不能绑在你的批量后面。只有 `run_id + 均值` 也比"等齐"有用。

---

## 7. 红线与常见坑

**红线（做了等于重做）**

1. **自己写 OOD 加权**：官方脚本会对三种错直接抛错，自己写的版本会静默给错数。
2. **对五折 OOD 做 t 检验 / 报均值 ± SE**：折固定，不是独立重复（D-09）。做错了不报错，只给假显著。
3. **架构排序反了就"再试一次"**：polymer 的架构排序是预注册的。反了就是反了，写进 `DECISIONS.md` 照实说。
4. **主表填 0 充缺格**：缺就留灰标 `blocked_no_gpu`。
5. **改 `run` 记录让测试变绿**：记录只能由训练脚本落盘；你只有"退回"的权力。
6. **共用表整表求和**：`gpu_hours.csv` 必须按 `arm` / `run_id` 筛（D-37）。

**常见坑**

| 坑 | 现象 | 正确做法 |
|:--|:--|:--|
| 组分少几条 | `summarize` 抛"组分缺预测"，或权重和 ≠ 1 | 断言覆盖完备，缺就回 C-b 补特征 |
| 把 OOD 当 ID 的平均 | 报出一个"5 折平均 ± SE"的假显著 | 五折向量 → 按聚合物取平均 → 集成点值 |
| ID 两档合并 | 交替 611 与均聚 37 混成一体，噪声掩盖信号 | **分开报**，别看总数好看 |
| 主表先等 CI | D 干等到最后一天 | 粗版先给，CI 留空占位 |
| 审计只测"能过" | 注入错误也不红 | 四处注入必须变红，否则等于没写 |
| `pytest` 少了几条 | 有人 `skip` 掉了 GPU 那几项以外的 | 基线 280 passed / 7 skipped，数目对不上要查 |
| 用 `ls results/raw_runs/*.json \| wc -l` 减掉 `_schema.json` 时数错 | 条数差 1，对账卡住 | `_schema.json` 不是记录，统计时排除 |

**代码怎么写**：按 [`docs/CODE_STYLE.md`](CODE_STYLE.md)——中英双语注释，每个函数写清输入输出、数据来源、配置来源，数字必须有来源。

---

## 8. 交接前自检（逐条打勾再 push）

- [ ] `git status` 里没有 `generated/`、模型权重、`logs/`、`*.partial.*`
- [ ] `pytest tests/` **280 passed / 7 skipped**（数目对不上要解释）
- [ ] 四处注入（删字段 / 改名 partial / device 与 env 不符 / 重复 run_id）**全部变红**
- [ ] `main_table.csv` 40 格齐全或明确标 `blocked_no_gpu`，**没有一格是 0**
- [ ] 每格都能点回 `run_id`，且该 `run_id` 的记录与预测都在
- [ ] `arch_transfer.csv` 的 ID 两档（611 / 37）**分开**；OOD 三档是**集成点值**
- [ ] `arch_ood_vectors/` 里 30 条五折向量都在（缺一条集成就不成立）
- [ ] `arch_component_preds/<run_id>.csv` 是两列 `psmiles,prediction`，且每个 psmiles 只出现一次
- [ ] 与官方地板的差（random / block / combined）写进表或 `Main Table Notes.md`
- [ ] `Main Table Notes.md` 写清了缺格原因与 CI 占位约定
- [ ] 条数对账：记录数 = 546 + C 组各批 `expected_*` 之和，差额逐条说明
- [ ] 提交信息写清"做了什么 / 没做什么 / 下一步是什么"（README 第十节）

---

## 9. 工时口径核对

| 项 | pH |
|:--|--:|
| C-c-1 架构迁移 30 runs（ID 两档） | 1.5 |
| C-c-2 OOD 组分预测与三段式合成 | 1.5 |
| C-c-3 主表汇总与定稿 | 1.5 |
| C-c-4 运行记录审计：套件与结论 | 2.5 |
| **合计（固定项）** | **7.0** |

**7.0 是怎么来的**：母本 C-4 的 5.0 里划走 2.0 给 C-b（物化与地板），剩 3.0（1.5 + 1.5）；C-5 的 2.5 里划 1.0 给 C-a（差异分类），剩 1.5；C-6 的 4.0 里划 1.5 给 C-b（批量执行），剩 2.5。

> **你的余量花在哪**：`C-c-2` 是三个人里**唯一可能在 11-18 才暴雷**的一步（组分覆盖、摩尔分数、重复 psmiles 三种错都只在那一步现形）。所以余量优先给它：**11-16 一收到地基就先断言覆盖完备**，不要等到 30 runs 全跑完才发现组分缺了几百个。
