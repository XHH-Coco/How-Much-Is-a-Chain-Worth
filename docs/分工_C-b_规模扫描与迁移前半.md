# 分工 C-b · 规模扫描与迁移前半：scaling、OpenPoly 经典臂、架构迁移的物化

> **对应**：`docs/分工_C_经典臂与机制实验.md`（母本）的 **C-3**（4.0）、**C-4 的前两步**（2.0）、**C-6 的批量执行**（1.5）
> **工时**：固定项 **7.5 人时** = 三份 7.5 / 7.5 / 7.0 里的第二份（合计 22.0，与母本一致）；**弹性余量 2.0 不进任何一份**，留在 C 组名下共用
> **你负责的目录**：`configs/experiments/{scaling,openpoly_classic}.yaml`、`src/chainworth/experiments/scaling.py`、`scripts/06_run_cpu.py`（新增）、`data/processed/features/`（`pb26_scaling_cp` 两份**新增**）、`results/aggregate/{scaling_cpu,openpoly_cpu}.csv`、`docs/Arch Materialization.md`、`docs/Audit Log.md`（运行侧执行记录）
> **一句话**：你负责**把"数据变多，误差会不会变小"和"换一份划分，数字会不会跳"变成两条能交叉验证的曲线**，并给 C-c 把架构迁移的地基（物化 + 官方地板）打好。

> **三人切分口径（三份文档一字不差地写着同一张表）**
>
> | 分工 | 拿的是母本哪几条 | 固定项 | 交给谁什么 |
> |:--|:--|--:|:--|
> | **C-a 经典臂族** | C-1 5.0 + C-2 1.5 + C-5 的经典三列对齐 1.0 | **7.5** | 冻结的 A2 超参 → 你；A1/A2/A5 的 run_id → C-c |
> | **C-b 规模与迁移前半（本文件）** | C-3 4.0 + C-4 的物化与官方地板 2.0 + C-6 的批量执行 1.5 | **7.5** | 物化划分 + 组分特征 → C-c；退单台账 → C-c |
> | **C-c 迁移主跑、主表与审计** | C-4 的 30 runs 与 OOD 3.0 + C-5 定稿 1.5 + C-6 套件 2.5 | **7.0** | 主表 → D；审计结论 → 全队 |
>
> 切法只有一条原则：**谁的数字谁做，谁的接口谁定**。三人共用同一批特征、同一份划分、同一组模型种子 `seed0/1/2`——**谁的活都不许改这三样**。

---

## 0. 先用大白话讲：你这份差事是干什么的

### 你干的是"把横轴拉开"的活

四个人合写一份报告。C-a 在 8 个任务上把秤校准了，你接着要回答两个**只有你能回答**的问题：

| 你要回答的 | 怎么回答 | 给谁用 |
|:--|:--|:--|
| **数据变多，误差会不会变小？变多快？** | PolyBench26 官方的 `scaling_indices`——同一份数据，**只换训练集大小**，看 R² 怎么爬 | D 算 **κ**（H2 的判据） |
| **"OpenPoly 上我们不如官方 XGBoost"是不是划分的锅？** | 在**官方同样的规则复现划分**上跑经典臂，与 B 的 GNN 在**同一批训练行**上对比 | D 求 **H4 的交叉点** |
| **架构排序在 ID 与 OOD 上是否一样？** | 先把官方 task4 的留出集**物化**出来、先复现官方地板，再交给 C-c 跑 30 runs | C-c（他跑，你打地基） |

**最关键的一句**：这三批全部**复用 C-a 调好的超参**、**只换训练集大小或换数据子集**。你一旦"为了让曲线好看"另调一套超参，κ 就和主表不是同一个世界了，H2 直接作废。

### 你还负责"第一批守门"

母本把运行记录审计 4.0 pH 拆给了两个人：C-c 写**套件**（`tests/test_run_records.py`），你负责**在各批收尾时真的跑一遍、并把不合格的记录退回去**。理由很实在——你的三批最早收工，且你不产 GPU 记录，查 B 的 273 条最"无立场"。

### 做完的标志（一句话）

> **D 拿到 `scaling_cpu.csv` 能直接拟合 κ、拿到 `openpoly_cpu.csv` 能与 B 的 450 条按 (性质, 档位, 折, 种子) 逐格相减；C-c 拿到 `generated/shared_splits/` 与组分特征表就能不问你一句地开跑 30 runs。**

---

## 1. 你在流水线上的位置

| 阶段 | A 数据（已完成） | B 表示与 GPU 臂（已完成） | C-a 经典臂族 | **C-b（你）** | C-c |
|:--|:--|:--|:--|:--|:--|
| P3 配置 | 任务清单已冻结 | 三份配置已冻结 | `a1/a2/a5` | ● `scaling` / `openpoly_classic` | ○ 等你的地基 |
| P4 代码 | 划分/泄漏测试 | 装配 + 图 + 嵌入 | 三臂骨架 | ● **主责**：Cp 双数据集装配、档位切分、物化管线 | ● 30 runs |
| P5 运行 | 交数据 | 546 条已落盘 | A1/A2/A5 | ● **主责**：60–80 + 225 fits | ● 30 runs + OOD |
| P6 对账 | 答疑 | 已验收 | 排行榜三列 | ● **主责**：审计批量执行与退单 | ● 主表 + 审计套件 |

● 主责　○ 参与/等待

**你的顺序是硬的**：先做**不依赖任何人**的 `C-b-3`（架构迁移物化，只依赖上游官方仓库）→ 再做 `C-b-1` 的第一个数据集（顺手把装配路径打通）→ C-a 的超参一到就跑 `C-b-2`（225 条，可挂机）→ 最后 `C-b-4`（审计）。反过来做（先等超参）你会白等半天。

### 你手上的现成东西（以及一处**必须先补的洞**）

| 你有什么 | 在哪 | 怎么用 |
|:--|:--|:--|
| OpenPoly 三性质的特征 | `data/processed/features/{op_tg,op_tm,op_dielectric_constant_total}.npz` | 现成，`pack_sha256` 已登记 → **225 条今天就能开跑** |
| PB26 两个主表任务的特征 | `.../pb26_{polymetrix_tg,homopolymer_density}.npz` | 只作参照，**scaling 用的不是这两个任务** |
| 官方 scaling 划分快照 | `splits/polybench26/pb26_scaling_cp/{md5000_alternating_cp,md5000_homopolymer_cp}/seed_42/` | 每份 `manifest.json` + `fold_00..04.npz`；npz 的键是 **`train_pool / val / test`** |
| B-5 的 n 档索引（**对账唯一依据**） | `results/aggregate/openpoly_n_index.csv`（150 行 = 3 性质 × 10 折 × 5 档） | 含 `subset_sha256` 与 `n_bins_covered`；**你的 225 条必须与它逐档同子集** |
| run 记录字段契约 | `results/raw_runs/_schema.json` | `arm` 枚举含 `a2_conventional`；`task` 必须是 `configs/tasks/<task>.yaml` 的 `task_id` |

> **【洞 · 实测，别以为是环境问题】`pb26_scaling_cp` 现在装不了特征。**
> ① `data/processed/features/` 里**没有** `pb26_scaling_cp*.npz`（15 份里没有它）；
> ② `chainworth.io.loaders.load_split('pb26_scaling_cp')` 直接抛 `PermissionError`——它把**目录**当文件打开，因为这个任务有 **2 个数据集**、每个数据集各有 5 折，现有 `SplitSpec` 是"一任务一划分"。
> 所以 **C-b-1 的第一件事不是跑 fit，是给这个任务做两份 pack（`md5000_alternating_cp` / `md5000_homopolymer_cp`）并让 loader 认得"双数据集 + 档位"这一层**。这一步是这一份分工里唯一有"要改公共代码"风险的地方，**排在最前面做**。

---

## 2. 任务清单（拿什么 → 怎么做 → 验收标准）

> 四项固定合计 **7.5 pH**。

### C-b-1 · 数据量扫描 scaling（1.5 pH｜关键，H2 的判据）

| 项 | 内容 |
|:--|:--|
| **拿什么** | `splits/polybench26/pb26_scaling_cp/*/seed_42/`（A-5 快照，**只读**，字节级 sha256 已登记在 `data/manifests/splits_manifest.csv`）；`configs/tasks/pb26_scaling_cp.yaml`（已冻结）；C-a 的 `a2_best_params.csv`；`data/raw/polybench26/Datasets/PSMILES/MD_5000/Cp/{alternating,homopolymer}_Cp.csv`（造特征用） |
| **怎么做** | ① 给两个数据集各造一份特征 pack（走 `02_features.py` + `03b_assemble.py` 的现有路径；`canonical_sha256` 已在 `manifest.json` 里，造完必须核）；② 打通"任务 → 数据集 → 档位"三层寻址（见下方口径框）；③ **档位 = `train_pool[:N]` 的前缀**，不是"自己按行数截前 n 行"、也不是"随机抽 n 行"；④ 每档 × 5 折 × 2 臂（XGBoost / RF，超参抄 C-a 的），预测逐条落盘 |
| **产出** | `configs/experiments/scaling.yaml`（含 `expected_fits`）、`data/processed/features/pb26_scaling_cp_*.npz` 两份（**不入库**，靠脚本重建）、`results/aggregate/scaling_cpu.csv`（一 fit 一行：`task,dataset,tier,n_train,fold,arm,seed,r2,mae,rmse,wall_clock_s,run_id`）、run 记录 + 预测 |
| **验收标准** | ① 断言**读的是快照**：逐份核 `seed_42/*.npz` 的字节 sha256 与 `splits_manifest.csv` 相符；② 断言训练集 = `train_pool[:N]`（写成测试，不是注释）；③ 档位逐档齐全，缺档写明"这是事实不是失败"；④ `openpoly_n_index.csv` 那种"子集身份"也要有——**每档的 `subset_sha256` 写进 `scaling_cpu.csv`**，否则 D 无法证明五折用的是同一批行；⑤ 与 A2 的超参除"训练集规模"外逐字段一致（`a2_config_hash` 字段 + 解析后逐键断言） |

> **【口径先定死·母本算术对不上配置】** 母本写"5 档 × 5 折 × 2 性质 × 2 臂 = **100 fits**"。按实测的 `manifest.json`：
> **交替 Cp 有 4 个档（300 / 1000 / 3000 / 10000）、均聚 Cp 有 2 个档（300 / 1000）**，`train_pool` 分别是 **22,242 / 1,440**。
> 所以 **6 个档位 × 5 折 × 2 臂 = 60 fits**；若把 `all`（`train_pool` 全长）当成第 7、第 8 档，则是 **80 fits**。
> **100 在这个配置上算不出来**（5 档 × 2 数据集 = 10 个档位，均聚只有 2 个档）。开工 30 分钟内定死，写进 `scaling.yaml` 的 `expected_fits`，之后**只引这一个数**。建议取 **80**：`all` 档是 κ 曲线右端的锚点，缺了 D 只能外推。

> **【run_id 装不下第四个维度】** `_schema.json` 的三段式是 `<task>__<arm>__<fold>__seed<N>`，而这里需要区分 **数据集 × 折 × 档位** 三个值。B-5 的做法是把档位塞进折段（`seed0_n120`，D-35）。照抄同一手法：`run_id = pb26_scaling_cp__a2_conventional__md5000_alternating_cp_fold_00_n1000__seed0`，`split_fold` 仍记 `fold_00`，另加 `dataset_id` 与 `n_level` 两个字段（`additionalProperties: true` 允许加，**只加不改**）。**别自己发明第四段**——审计套件按 pattern 卡，卡不过就是红。

### C-b-2 · OpenPoly 经典臂 225 fits（2.5 pH｜关键，H4 的交叉点）

| 项 | 内容 |
|:--|:--|
| **拿什么** | `data/processed/features/op_{tg,tm,dielectric_constant_total}.npz`（现成）；`splits/openpoly/<task>/rule_replicated_seed{0..4}.json`；**`results/aggregate/openpoly_n_index.csv`（150 行，含 `subset_sha256`）**；C-a 的超参 |
| **怎么做** | 3 性质 × 5 个 n 档（30 / 60 / 120 / 240 / all）× 折 `seed0–4` × 3 种子 = **225 fits**，2 臂（XGBoost / RF）。**n 档语义按 D-35**：n = 训练集行数，**固定种子打乱后取前缀**、五档**嵌套**；`n=240` 在 `op_dielectric`（池 235）与 `op_tm`（池 134）上取不满 → 如实记 `n_level_saturated`，与 `all` 档逐位相同是**预期的自检**，不是错误。**先核 `subset_sha256`，再跑**——你与 B 的 GNN 必须落在同一批训练行上，否则 H4 的交叉点没有意义 |
| **产出** | `configs/experiments/openpoly_classic.yaml`（`expected_fits: 225`）、`results/aggregate/openpoly_cpu.csv`（一 fit 一行，含 `n_level,subset_sha256,n_bins_covered,n_level_saturated`）、225 条 run 记录 + 预测 |
| **验收标准** | ① 逐档核 `subset_sha256` 与 B-5 的 `openpoly_n_index.csv` **完全相同**（一条不同就停，别"大概一样"）；② 一律 **n ≤ 443**，不许外推；③ `n_bins_covered` 记下来（D 要判断"低 n 端覆盖不到全值域"是不是现象）；④ 与 A2 的 `config_hash` 关系按 `a2_config_hash` + 逐键断言处理；⑤ 每 fit 都落预测——D 的 κ/CI 重采样与 H4 过零点都要用 |
| **与 B 的边界** | 同一网格上 GNN 的 450 条（225 主表 + 225 敏感性）是 B 的，**已交付完毕**。你只跑经典臂这 225 条，**n 档与折索引一个字都不许改**——改了就不是同一张网格，H4 直接作废 |

### C-b-3 · 架构迁移的物化与官方地板复现（2.0 pH｜关键，C-c 的地基）

| 项 | 内容 |
|:--|:--|
| **拿什么** | 上游 `Datasets/task4_architecture_transfer/` 的**三个脚本**（`create_task4_shared_splits.py` / `summarize_weighted_predictions.py` / `evaluate_rdkit_ood.py`）+ `task4_all_model_scores.csv`（已发布的对账基准）。**注意：本机 `data/raw/polybench26/Datasets/` 里现在只有 `PSMILES/`、`wPSMILES/`、`scaling_indices/`、`Dataset_construction_scripts/`，没有 task4 目录**——第一步是按 A-2 的方式从上游取（并记 commit 与逐文件 sha256，写进 `SOURCES.md` 的同一套格式） |
| **怎么做** | ① **取件**：取 task4 目录，登记 commit / 文件 sha256；② **物化**：跑 `create_task4_shared_splits.py`（那个 224 MB 的 polyBERT 字典 pkl **只有 polyBERT 臂需要，经典臂可省**），产物落 `generated/shared_splits/`（gitignored）；③ **核行数**：canonical **6,478 = 交替 6,110 + 均聚 368**，OOD **36,828 = random 18,414 + block 18,414**，与官方 README 对不上就停；④ **先复现官方地板**：用 `evaluate_rdkit_ood.py` 跑 RDKit RF，与 `task4_all_model_scores.csv` 对照（官方：ID 交替 **0.148–0.170**、均聚 **0.37–0.64**；OOD ensemble **random 0.536 / block 0.581 / combined 0.559**）；⑤ **造组分特征**：从 36,828 行 holdout 取**全部唯一组分**（`psmiles_a` ∪ `psmiles_b`），每个 psmiles **只算一次** 205 描述符 + 2048 位指纹（复用 A-6 的 `structures.py` / `descriptors.py`，别另写一套） |
| **产出** | `docs/Arch Materialization.md`（取件凭据 + 行数核对 + 地板复现三档数 + 差在哪）、`generated/shared_splits/`（物化产物 + 逐文件 sha256）、`data/processed/arch/component_features.parquet` + `component_index.csv`（`psmiles → 特征行号`，含 `feature_sha256`）、`configs/experiments/arch_transfer.yaml` 的 `data` 段（档位与行数） |
| **验收标准** | ① 行数四条全对（6,478 / 6,110 / 368 / 36,828），对不上**停**；② 官方地板复现差距要在文档里**写出具体数值**，OOD combined 差 **> 0.1 就先查物化，不许调模型**；③ **组分覆盖完备**：holdout 里每个唯一 `psmiles` 在 `component_index.csv` 里**有且只有一行**（写成断言，缺一条就红）——这是 C-c 唯一会踩的坑；④ 组分里含 3/4/6 个 `*` 的结构（A-6 有 116 个拒出 wPSMILES）**不许静默丢**，要么给特征、要么列进 `component_failures.csv` 并让 C-c 知道会缺哪几条；⑤ 三脚本的 `--help` 与用法抄进 `Arch Materialization.md`，C-c 照着跑不需要问你 |

> **【口径已定·D-10】** 母本原写"2 性质 × 2 臂 × 5 折 × 3 种子 = 60 runs"。官方 task4 **只做 electron affinity 一个性质**，正确的是 **30 runs**（C-c 跑）。**5.0 pH 不降**的理由：少掉的 30 次拟合并未让工作变少，因为 OOD 那 36,828 行不是直接预测，而是"逐组分预测 + 按摩尔分数加权合成"，这条管线的成本恰好是 2 pH 量级。
>
> **【许可·必须随结果走】** 这个任务的训练源是 polyVERSE（EA），属 **GTRC General Public Use License**，不是主表那个 CC BY 4.0。结果表须**随附 GTRC 全文、免费分发、不得商用**，且**不能把 GTRC 的条款套到 PolyBench 的任务头上**。物化产物 `generated/shared_splits/` 属派生数据，**不入库**，只在文档里记 sha256。

### C-b-4 · 运行记录审计：批量执行与退单（1.5 pH｜关键）

| 项 | 内容 |
|:--|:--|
| **拿什么** | C-c 写的 `tests/test_run_records.py`；全队 `results/raw_runs/*.json`（B 的 546 条 + C 组陆续新增）与 `results/aggregate/predictions/*.parquet` |
| **怎么做** | 每批跑完立刻跑一遍套件（不是最后一天跑一遍），把不合格的记录**退回给产出人**（B 已验收，退回对象主要是 C-a / C-c 与你自己），并逐条记在 `results/aggregate/audit_backlog.csv`（`run_id,问题,退回给谁,退回时间,状态`）。**你只退回，不改数** |
| **产出** | `results/aggregate/audit_backlog.csv`、`docs/Audit Log.md` 的"运行侧执行记录"章节（每批一节：何时跑、多少条、几条不合格、退了谁） |
| **验收标准** | ① 每一批收尾都有一次执行记录（`scaling` / `openpoly` / `a1_a2` / `a5` / `arch` 五批不缺）；② 注入错误的测试会红（C-c 的套件自测，你要在他的凭据上复跑一次确认）；③ 不合格的记录**必须退**，不许"帮忙改一下就好"；④ 条数对账：`ls results/raw_runs/*.json \| wc -l` 与各批 `expected_*` 之和相符，差额逐条说清 |

> **为什么执行给你**：你的三批最早收工，且你不产 GPU 记录——查 B 的 546 条最"无立场"。C-c 写套件是因为他要拿这张表对账；**写的人与跑的人分开，测试才不会为了"跑绿"而放松断言**。

---

## 3. 交付物清单（文件级）

```
configs/experiments/{scaling,openpoly_classic,arch_transfer}.yaml
src/chainworth/experiments/scaling.py          ← 档位切分 + 双数据集寻址
data/processed/features/pb26_scaling_cp_*.npz  ← 两份，不入库，脚本重建
results/raw_runs/<a2_*>.json                   ← scaling 60–80 + openpoly 225，入库
results/aggregate/predictions/<run_id>.parquet
results/aggregate/{scaling_cpu,openpoly_cpu}.csv
results/aggregate/audit_backlog.csv
generated/shared_splits/                       ← 物化产物，不入库（GTRC）
data/processed/arch/{component_features.parquet,component_index.csv}
docs/Arch Materialization.md
docs/Audit Log.md（运行侧执行记录）
```

**不入库**：物化产物、模型权重、`logs/`、`outputs/`、`.pkl`。

---

## 4. 没有 GPU 怎么办

**你一项都不需要 GPU。** 这一份分工里 0 条 GPU 记录。

| 任务 | GPU | 量级 |
|:--|:--|:--|
| C-b-1 scaling | 无 | 60–80 fits，最大 22,242 行 × 2,253 列，单条 20–60 s → 挂机 1–2 h；**另有造特征（29,602 行）约 3–5 分钟** |
| C-b-2 OpenPoly 225 | 无 | 单条 1–5 s（143–651 行），总计 10–20 分钟，可分三次挂机 |
| C-b-3 物化与地板 | 无 | 物化 + RDKit RF 地板，20–40 分钟；组分特征（36,828 行去重后的分子）是**大头**，先用缓存再全量 |
| C-b-4 审计 | 无 | pytest 级别，秒级 |

被卡的方式有两种：**C-a 的超参没定**（`C-b-2` 开不了工）与**上游 task4 取不到**（`C-b-3` 开不了工）。对策：`C-b-3` 的取件放在第一个小时做，`C-b-1` 的装配放在第二个小时做——这两件事都不依赖任何人。

---

## 5. 依赖关系与冻结边界

**你依赖**：A 的 `splits/polybench26/pb26_scaling_cp/*/seed_42/`（只读）、`data/raw/polybench26/Datasets/PSMILES/MD_5000/Cp/`（造特征）；C-a 的 `a2_conventional.yaml` + `a2_best_params.csv`；B-5 的 `openpoly_n_index.csv`（对账靶子）；C-c 的 `tests/test_run_records.py`（审计）。

**依赖你的**：C-c（物化划分、组分特征、30 runs 的全部输入）；D（κ、H4 交叉点）。

| 交接点 | 交付什么 | 对方拿它干什么 | 什么时候 |
|:--|:--|:--|:--|
| **→ C-c** | `generated/shared_splits/` + `component_features.parquet` + `component_index.csv` + `docs/Arch Materialization.md` | 跑 30 runs 与 OOD 三段式；**先断言组分覆盖完备** | 11-16 |
| **→ C-c** | `configs/experiments/arch_transfer.yaml` 的 `data` 段（行数、折名、holdout 身份） | 他的配置引用你的数，不自己数 | 11-16 |
| **→ D** | `scaling_cpu.csv`（含 `subset_sha256`） | 拟合 κ（H2） | 11-16 |
| **→ D** | `openpoly_cpu.csv`（含 `subset_sha256`、`n_level_saturated`） | 与 B 的 GNN 逐格相减，求 H4 交叉点 | 11-17 |
| **→ C-c** | `audit_backlog.csv` | 主表定稿前必须清零 | 每批收尾 |

| 冻结项 | 冻结在 | 谁能改 |
|:--|:--|:--|
| 档位与折索引 | `configs/experiments/scaling.yaml` + `splits/` 快照 | 档位来自官方文件，**不许按行数截**；要改就是新版本并全批重跑 |
| OpenPoly 的 n 档网格 | `configs/experiments/openpoly_classic.yaml` + `openpoly_n_index.csv` | n 档与折是**数据**，不进配置（D-35 / D-36 的同一条道理） |
| 物化产物 | `generated/shared_splits/` | 谁都不改；C-c 只读，重跑要你确认 |
| 主表列名 | `results/aggregate/main_table.csv` | 只加不改 |

> **全队只有一张冻结登记表**：`README` 第九节。两份实验配置**登记完才算冻结**；冻结物被改动 = 该阶段重做。

---

## 6. 时间线与里程碑（剩余 5 个工作日，11-19 汇报）

| 时间 | 你要交的 | 卡点 |
|:--|:--|:--|
| **11-13（五）** | ① task4 取件完成、物化跑起来；② `pb26_scaling_cp` 两份 pack 造出来且 loader 认得；③ `scaling.yaml` 的 `expected_fits` 定死 | 两个洞都在这天堵 |
| **11-14~15（周末）** | 物化行数四条核对完 + 官方地板复现出数；scaling 60–80 fits 挂机跑完 | 地板不对就停，别往下走 |
| **11-16（一）** | `scaling_cpu.csv` 交 D；**组分特征 + 索引表**交 C-c（他 11-17 要开跑） | C-c 的硬依赖 |
| **11-17（二）** | OpenPoly 225 fits 跑完、`subset_sha256` 逐档对上 B-5；`openpoly_cpu.csv` 交 D | H4 的判据 |
| **11-18（三）** | 五批审计全部执行完、`audit_backlog.csv` 清零；`Arch Materialization.md` 定稿 | 交付前 |
| **11-19（四）** | 答疑、补跑、冻结 | 交付 |

> **别把三批堆到最后一天**：每跑完一批就落盘一批、交一批、审计一批，而不是"全部跑完再写汇总脚本"。

---

## 7. 红线与常见坑

**红线（做了等于重做）**

1. **scaling 自己按行数截**：官方档位 ≠ 取前 n 行。读官方 `train_pool` 前缀，并且**断言哈希**。
2. **OpenPoly 换一套子集**：你与 B 必须落在同一批训练行上（`subset_sha256` 相等），否则 H4 的交叉点不是"臂的差"而是"子集的差"。
3. **机制实验另起超参**：为了"让曲线好看"换个模型或调个学习器，κ 就和主表不是同一个世界。
4. **地板不对还往下跑**：官方地板差一个量级说明物化或合成错了，继续跑只会得到一堆假数。
5. **run 记录手写 / 手改**：退回可以，改数等于伪造。
6. **组分静默丢弃**：含 3/4/6 个 `*` 的结构算不出 wPSMILES 时，**列出来**交给 C-c，不许悄悄少几行——少的那几条会在 `summarize_weighted_predictions.py` 里变成"组分缺预测"的报错，或者更糟：不报错但权重和不等于 1。

**常见坑**

| 坑 | 现象 | 正确做法 |
|:--|:--|:--|
| 把 `pb26_scaling_cp` 当成"一任务一划分" | `PermissionError`，看着像权限问题 | 它是**双数据集**任务，每个数据集各有 5 折；先做两份 pack |
| npz 里没找到"档位数组" | 以为档位丢了 | 档位**不在** npz 里：`manifest.json` 的 `train_sizes` 是档位，`train_pool[:N]` 是切法 |
| 均聚 Cp 硬凑 5 档 | 造出官方没有的档位，κ 曲线失真 | 只有 300 / 1000（+ all），**缺档写明是事实** |
| 只存指标没存预测 | D 无法重采样 | 每 fit 落 `predictions/<run_id>.parquet` |
| 用了别人的 `generated/` | 物化版本不同，行数对不上 | 自己物化、自己核四条行数 |
| 审计只跑绿不抓错 | 测试永远通过，等于没写 | 故意注入错误，看它红不红（C-c 的套件，你复跑） |
| 共用表整表求和 | GPU 小时数算错（D-37 的错法） | 记账按 `arm` / `run_id` 筛，**别整表 sum** |

**代码怎么写**：按 [`docs/CODE_STYLE.md`](CODE_STYLE.md)——中英双语注释，每个函数写清输入输出、数据来源、配置来源，数字必须有来源。

---

## 8. 交接前自检（逐条打勾再 push）

- [ ] `git status` 里没有 `generated/`、模型权重、`logs/`、`*.partial.*`
- [ ] `pytest tests/` 全绿（基线 280 passed / 7 skipped，只许多不许少）
- [ ] `scaling_cpu.csv` 的档位数 = `scaling.yaml` 的 `expected_fits` 对应的档位，缺档有说明
- [ ] 每档的 `subset_sha256` 都在 CSV 里（D 要证明五折同一批行）
- [ ] OpenPoly 225 条的 `subset_sha256` 与 `openpoly_n_index.csv` **逐档相等**
- [ ] 物化四条行数（6,478 / 6,110 / 368 / 36,828）写在 `Arch Materialization.md` 里
- [ ] 官方地板复现的具体数值写出来了，OOD combined 的差也写了
- [ ] `component_index.csv` 里每个唯一 `psmiles` 只有一行；缺特征的列进 `component_failures.csv` 并告知 C-c
- [ ] `audit_backlog.csv` 每条都有"退回给谁 / 什么时候 / 现在什么状态"
- [ ] 提交信息写清"做了什么 / 没做什么 / 下一步是什么"（README 第十节）
- [ ] 没往 `docs/SOURCES.md` 里漏掉 task4 的取件 commit 与 sha256

---

## 9. 工时口径核对

| 项 | pH |
|:--|--:|
| C-b-1 数据量扫描（含两块 pack 与寻址打通） | 1.5 |
| C-b-2 OpenPoly 经典臂 225 fits | 2.5 |
| C-b-3 架构迁移物化 + 官方地板 + 组分特征 | 2.0 |
| C-b-4 运行记录审计：批量执行与退单 | 1.5 |
| **合计（固定项）** | **7.5** |

**7.5 是怎么来的**：母本 C-3 的 4.0（1.5 + 2.5 拆成两项）+ C-4 的 5.0 里划 2.0 给"物化与地板" + C-6 的 4.0 里划 1.5 给"批量执行"。C-4 剩下的 3.0（30 runs + OOD）与 C-6 的 2.5（套件）在 C-c。

> **你的余量花在哪**：`pb26_scaling_cp` 的双数据集装配是这一份里唯一"要动公共代码"的地方——如果它拖到 11-15，宁可从 C 组共用余量里挪 0.5 出来，也不要挤掉 `C-b-3` 的取件（那是 C-c 的前置，晚了会连环塌）。
