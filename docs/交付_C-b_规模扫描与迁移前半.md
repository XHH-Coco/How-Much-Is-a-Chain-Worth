# 交付 C-b · 规模扫描与迁移前半

> **对应**：`docs/分工_C-b_规模扫描与迁移前半.md`（C-3 4.0 + C-4 的物化与官方地板 2.0 + C-6 的批量执行 1.5 = **7.5 pH**）
> **一句话**：横轴拉开了、交叉点有数了、C-c 的地基打好了、四批审计跑完了——**265 条新记录全部由训练脚本落盘，没有一条手写**。
> **交付日**：2026-11-15｜**执行环境**：本机 `cb310`（Python 3.10.14 + `lock/core.pip.txt` 的 34 个 pip 包逐个同版本，见 §6 / D-55）

---

## 0. 交付物清单（文件级）

| 类别 | 产物 | 条数/规模 | 入库 |
|:--|:--|:--|:--|
| **C-b-2** | `configs/experiments/openpoly_classic.yaml`（4,713 B，sha256 `fa8b757d0001f9d4…`） | 1 份 | ✅ |
| | `results/raw_runs/op_*__a2_conventional__seed{k}_n{level}__seed{s}.json` | **225 条** | ✅ |
| | `results/aggregate/predictions/<run_id>.parquet` | **225 份** | ✅ |
| | `results/aggregate/openpoly_cpu.csv`（一 fit 一行）+ `openpoly_cpu_n_index.csv`（一档一行，含 `subset_sha256` / `n_bins_covered`） | 225 / 75 行 | ✅ |
| **C-b-1** | `configs/experiments/scaling.yaml`（5,157 B，sha256 `60841fcdf666c3f7…`） | 1 份 | ✅ |
| | `src/chainworth/experiments/scaling.py`（三层寻址 + 档位前缀 + 批量） | 新模块 | ✅ |
| | `data/processed/features/pb26_scaling_cp_{md5000_alternating_cp,md5000_homopolymer_cp}.npz` | 27,802 + 1,800 行 × 2,253 列 | ❌（脚本重建） |
| | `data/manifests/scaling_features_manifest.csv` | 2 行（含 `canonical_sha256` / `pack_sha256`） | ✅ |
| | `results/raw_runs/pb26_scaling_cp__a2_conventional__*.json` | **40 条**（XGBoost 一侧） | ✅ |
| | `results/aggregate/scaling_cpu.csv` | **80 行**（40 XGBoost + 40 RF） | ✅ |
| **C-b-3** | `data/manifests/external_task4_sources.csv`（19 份取件逐份 sha256） | 19 行 | ✅ |
| | `generated/shared_splits/**`（物化产物） | 6,478 + 36,828 行 + 5 折 | ❌（GTRC 派生） |
| | `data/processed/arch/{component_features.parquet,component_index.csv,component_failures.csv}` | **683 行** × 2,253 列 | ❌（脚本重建） |
| | `data/manifests/arch_component_manifest.json` | 1 份 | ✅ |
| | `results/aggregate/arch_floor_compare.csv` + `arch_floor_summary.json` | 28 行 / 1 份 | ✅ |
| | `configs/experiments/arch_transfer.yaml`（**只补 `data` 段**；`eval` 段归 C-c，按 D-49 是单臂 15 runs；未冻结） | 1 份 | ✅ |
| | `docs/Arch Materialization.md`（取件凭据 + 行数 + 地板 + 复跑清单） | 1 份 | ✅ |
| **C-b-4** | `results/aggregate/audit_backlog.csv` | 只有表头（0 退单） | ✅ |
| | `docs/Audit Log.md` 的「运行侧执行记录（C-b 维护）」章节 | 6 批逐批记账 | ✅ |
| **代码/测试** | `scripts/04_run_cpu.py` 新增 `openpoly` / `scaling` 两个子命令；`scripts/00c/00d/00e`；`src/chainworth/experiments/{scaling,arch}.py`；`tests/test_openpoly_classic.py`（**11**）+ `tests/test_scaling_scan.py`（**13**，2026-11-17 起多一条清单指纹守卫） | — | ✅ |

**总记录数**：`results/raw_runs/` 现有 **907 条** = B 的 546 + C-a 的 96 + C-b-2 的 225 + C-b-1 的 40。

---

## 1. 先把两个"计划里没写的坑"说清楚

### 坑一：`a2_best_params.csv` 里没有这四个任务的行（D-51）

C-a 的冻结超参表按设计只覆盖 8 个主表任务；`op_tg` / `op_tm` / `op_dielectric_constant_total` / `pb26_scaling_cp`
**都没有行**，而 `train_one()` 遇到缺行是**报错而不是回退默认值**——所以照原样开不了工。

做法：用**同一条预注册协议**（Murcko 分组 5 折、只在第一折训练行内、候选 1,944、固定 `search_seed=20261113`、≤30 trials）
各补一行，然后**冻结给所有档位/折/种子**。实测 CV R²：

| 任务 | CV R²（第一折训练行内） | trials | 冻结的超参 |
|:--|--:|--:|:--|
| `op_tg` | 0.5033 | 30 | 200 树 / 深 5 / lr 0.05 / sub 0.7 / col 0.4 / mcw 20 / λ 10 |
| `op_dielectric_constant_total` | 0.6471 | 30 | 200 / 5 / 0.05 / 0.7 / 0.4 / 20 / 1 |
| `op_tm` | −0.1701 | 30 | 200 / 3 / 0.03 / 1.0 / 0.7 / 5 / 10 |
| `pb26_scaling_cp`（交替 Cp 的 `fold_00` 池，22,242 行） | **0.9622** | 30 | 600 / 7 / 0.03 / 0.85 / 0.4 / 5 / 1 |

> 8 行老超参**逐字未动**（按 `(task, learner)` 覆盖写，键不重叠）；`a2_conventional.yaml` 的字节哈希仍是
> `7d29768afc09…`（有测试守着）。**CV R² 不是测试 R²**，它只用来"从 30 个候选里挑一个"（C-a §3 的原话）。

### 坑二：`pb26_scaling_cp` 是**一个任务两个数据集**（D-54）

`loaders.load_split('pb26_scaling_cp')` 原来把**目录**当文件打开（`IsADirectoryError`，看着像权限问题）。
本轮补上"任务 → 数据集 → 档位"这一层：`experiments/scaling.load_dataset_split(dataset_id)`，
并让 `load_split()` 对这类任务**明确报错并指路**（不再让人对着 `IsADirectoryError` 猜）。
**没有**新增 `configs/tasks/*.yaml`（A-4 的 18 份冻结清单一个字节没动，`n_tasks == 18` 那条测试照绿）、
**没有**改 B-1 的 `task_features_manifest.csv`（C-b 的 pack 有自己那份清单）。

---

## 2. C-b-2 · OpenPoly 经典臂 225 fits（2.5 pH）

### 2.1 口径

- 网格 **225 = 3 性质 × 5 档（30/60/120/240/all）× 5 折（`seed0–4`）× 3 种子**，一个学习器（XGBoost）——
  与 B-5 的 GNN **同一个网格、同一份档位、同一个子集种子（20261122）**。
- 档位语义照 D-35：**n = 训练集行数**，固定种子打乱官方 `train_index` 后取前缀，五档嵌套；`test` / `val` 逐行不变。
- 记录形状：`run_id = op_tg__a2_conventional__seed0_n30__seed0`（n 档借折段承载），`split_fold` 仍记官方折名。

### 2.2 验收

| # | 要求 | 实测 | 结论 |
|--:|:--|:--|:--|
| ① | 逐档核 `subset_sha256` 与 B-5 的 `openpoly_n_index.csv` **完全相同** | 共有键 **75** 个（3 性质 × 5 折 × 5 档），**75/75 一致，0 不一致**；B 另有 75 个键是它的 5–9 折敏感性（C-b 不做，如实缺） | ✅ |
| ② | 一律 n ≤ 443，不外推 | 最大 `n_train` = **355**（`op_tg` 的 all 档），全部 ≤ 443 | ✅ |
| ③ | `n_bins_covered` 记下来 | `openpoly_cpu_n_index.csv` 的 75 行逐行有（用**该折自己的** 10 等宽箱，与 B-5 同口径） | ✅ |
| ④ | 与 A2 的 `config_hash` / 超参关系 | 记录 `config_hash` = `sha256:fa8b757d…`（本实验配置）；超参逐条来自 `a2_best_params.csv` 的 `(task, xgboost)` 行；**实验配置里 `learner.params` 是空的**（有测试守着） | ✅ |
| ⑤ | 每 fit 都落预测 | **225 / 225** 份 parquet，行数 = 该折 test 长度（套件断言③覆盖） | ✅ |

**额外自检（D-35 要求的那条）**：`op_dielectric` 的 240 档（池 235）与 `op_tm` 的 240 档（池 134）取不满 →
`n_level_saturated=true`、`n_train` 记实际行数、**R² 与 `all` 档逐位相同**（`tests/test_openpoly_classic.py` 断言）。

**跨臂自检**：15 个 (性质, 折) 的 `split_version` 与 B-5 的记录**逐字相同**（都是同一个
`rule_replicated_seed{k}.json` 的字节哈希）——两边读的是同一份划分，不是"看起来差不多"。

### 2.3 结果：经典臂 vs GNN（R²，5 折 × 3 种子 = 15 个数的均值 / 样本标准差）

| 性质 | 档位 | 训练行 | **C-b 经典臂（XGBoost）** | B-5 的 GNN | 差（经典 − GNN） |
|:--|--:|--:|--:|--:|--:|
| `op_tg` | 30 | 30 | **−0.0586** ± 0.0398 | +0.2848 ± 0.1548 | −0.3434 |
| `op_tg` | 60 | 60 | **+0.4079** ± 0.0983 | +0.3710 ± 0.1732 | +0.0369 |
| `op_tg` | 120 | 120 | **+0.5581** ± 0.1942 | +0.4653 ± 0.2604 | +0.0928 |
| `op_tg` | 240 | 240 | **+0.5807** ± 0.1905 | +0.5461 ± 0.2934 | +0.0346 |
| `op_tg` | all | 355 | **+0.6446** ± 0.1399 | +0.5795 ± 0.2479 | +0.0651 |
| `op_dielectric` | 30 | 30 | **−0.0536** ± 0.0562 | +0.3815 ± 0.1956 | −0.4351 |
| `op_dielectric` | 60 | 60 | **+0.4435** ± 0.1269 | +0.4364 ± 0.2260 | +0.0071 |
| `op_dielectric` | 120 | 120 | **+0.5410** ± 0.0978 | +0.5207 ± 0.2700 | +0.0203 |
| `op_dielectric` | 240（饱和） | 235 | **+0.5943** ± 0.1570 | +0.5970 ± 0.3024 | −0.0027 |
| `op_dielectric` | all | 235 | **+0.5943** ± 0.1570 | +0.5970 ± 0.3024 | −0.0027 |
| `op_tm` | 30 | 30 | **+0.0550** ± 0.3029 | −0.0401 ± 0.2067 | +0.0951 |
| `op_tm` | 60 | 60 | **+0.5192** ± 0.1368 | +0.1626 ± 0.1113 | +0.3566 |
| `op_tm` | 120 | 120 | **+0.5500** ± 0.1342 | +0.1709 ± 0.1027 | +0.3791 |
| `op_tm` | 240（饱和） | 134 | **+0.5439** ± 0.1155 | +0.1503 ± 0.1130 | +0.3936 |
| `op_tm` | all | 134 | **+0.5439** ± 0.1155 | +0.1503 ± 0.1130 | +0.3936 |

**怎么读（给 D 的 H4，不替 D 下结论）**：

1. **交叉点在低 n 端**：`op_tg` 与 `op_dielectric` 在 **n=30** 时 GNN 明显更好（经典臂 R² 掉到 0 以下、GNN 还有 0.28/0.38），
   而从 **n=60** 起经典臂反超并一路领先。`op_tm` 的 n=30 两端都在噪声里（±0.21~0.30），从 n=60 起经典臂大幅领先。
2. **GNN 的折间波动更大**（同一格 sd 0.10–0.30，经典臂 0.10–0.19），这与 B-5 §5 的"折间 sd 是种子间的 2.4–67 倍"一致。
3. 经典臂的 sd 在低档位更大（`op_tm` n=30 达 0.30）——**30 行训练 + 2,253 列**本来就没什么可学的，这不是实现坏了。
4. **这是同一批训练行、同一份划分上的对比**（`subset_sha256` 75/75 相同、`split_version` 15/15 相同），
   所以"臂的差"就是臂的差，不是子集的差。**过零点在哪、要不要按 log 目标、要不要分架构，都是 D 的判决。**

### 2.4 成本

| 项 | 实测 |
|:--|:--|
| 225 次拟合总墙钟 | **39.7 s**（单条 0.1–0.6 s；`op_tg` all 档最慢） |
| 3 个性质的超参搜索 | 每个 30 trials × 5 折；`op_tm` 约 40 s、`op_tg` / `op_dielectric` 各约 1 分钟 |
| GPU | **0**（`device=cpu` 逐条记录） |

---

## 3. C-b-1 · 数据量扫描 scaling（1.5 pH）

### 3.1 两份 pack（先核官方哈希，再造）

| 数据集 | 行 | 唯一结构 | 列 | `canonical_sha256` 与官方 | pack sha256 |
|:--|--:|--:|--:|:--|:--|
| `md5000_alternating_cp` | 27,802 | **27,802**（零重复） | 2,253 | `7dad9d1dc0e9…` ✅ 逐位相同 | `38047fcf0f9e…` |
| `md5000_homopolymer_cp` | 1,800 | **1,800** | 2,253 | `0b09e47df3b6…` ✅ 逐位相同 | `573d1181aa21…` |

- 描述符与指纹**复用 A-6 的唯一计算口** `chem.descriptors.compute()`，29,602 个结构各算一次
  （并行 + `data/processed/structures/.descriptor_cache.sqlite3` 缓存，**本轮 6 分钟**；单线程约 29 分钟）。
- NaN：均聚 **14,400** 个 NaN 单元 = 1,800 × **8**（就是 D-48 点名的 8 个 `BCUT2D_*` 恒 NaN 列），与主表同源。
- 解析失败 **0** 行。
- pack 清单：`data/manifests/scaling_features_manifest.csv`（2 行，含 `canonical_sha256` / `pack_sha256` / `built_at`）。

### 3.2 档位 = 官方前缀（红线 1）

**档位不是我们挑的**：`splits/polybench26/pb26_scaling_cp/<dataset>/seed_42/manifest.json` 的 `train_sizes`
（交替 300/1000/3000/10000、均聚 300/1000）+ 每个数据集一个 `all`（全池 22,242 / 1,440）。
取法 = **`train_pool[:N]` 的官方前缀**（`subset.rule: prefix`），**不是**按行数截、**不是**随机抽 n 行。

写成断言（`tests/test_scaling_scan.py`）：① 每档取到的行**逐位等于** `train_pool[:N]`；② 五档**嵌套**；
③ 训练行与 `val` / `test` 三条索引互斥且并集正好 0..n-1；④ 反向断言：同一池子上 `prefix` 与
OpenPoly 那种 `shuffle_prefix` 取到的行**不同**（否则说明参数没起作用）；⑤ 快照字节与
`splits_manifest.csv` 的 sha256 重算一致。

### 3.3 80 fits（8 档 × 5 折 × 2 学习器）

> **"2 臂"在 C-b-1 与 C-b-2 里不是一个意思**（D-53）：C-b-2 的 225 条是"一个学习器 × 3 种子"，
> C-b-1 的 80 条是"2 个学习器 × 1 个种子（seed 0）"。**XGBoost 落 run 记录与预测；RF 只进 `scaling_cpu.csv`**
> （`_schema.json` 的 arm 枚举里没有"森林"，写进 `raw_runs/` 会让 C-c 的条数对账多出 40 条——D-45 的同款处理）。
> RF 的超参抄 `learner_sensitivity.yaml` 的 `rf.fixed`（官方 RDKit_RF 的 100 树配方，**不调参**）。

| 数据集 | 档位 | 训练行 | XGBoost R²（5 折均值 ± sd） | RF R²（5 折均值 ± sd） |
|:--|--:|--:|--:|--:|
| alternating | 300 | 300 | **0.8646** ± 0.0145 | 0.7986 ± 0.0190 |
| alternating | 1000 | 1000 | **0.9257** ± 0.0063 | 0.8797 ± 0.0107 |
| alternating | 3000 | 3000 | **0.9513** ± 0.0028 | 0.9217 ± 0.0067 |
| alternating | 10000 | 10000 | **0.9610** ± 0.0022 | 0.9404 ± 0.0027 |
| alternating | all | 22,242 | **0.9657** ± 0.0019 | 0.9495 ± 0.0023 |
| homopolymer | 300 | 300 | **0.9185** ± 0.0275 | 0.8633 ± 0.0371 |
| homopolymer | 1000 | 1000 | **0.9593** ± 0.0087 | 0.9359 ± 0.0150 |
| homopolymer | all | 1,440 | **0.9659** ± 0.0057 | 0.9440 ± 0.0127 |

**怎么读（给 D 的 κ）**：

1. **两个数据集上都是"数据越多越好"，且收益递减**：交替 Cp 的 XGBoost 从 300 → all 涨 **+0.101**，
   其中 300→1000 贡献 +0.061、1000→3000 +0.026、3000→10000 +0.010、10000→all **+0.005**。
   `κ` 的横轴请用 **`n_train`**（不是档位名义值），`subset_sha256` 逐档都在表里。
2. **均聚 Cp 只有三档**（官方只有 300/1000）——**缺档是事实，不是失败**；不要把它接到交替 Cp 的曲线上。
3. **两个学习器给出的曲线形状一致**（RF 处处低约 2–7 个百分点，因为 RF 没调参而 XGBoost 调过 30 次）——
   **只在各自内部比**，跨学习器的绝对水平不可直接比（D-53 ④）。
4. 均聚的 sd（0.006–0.037）比交替（0.002–0.019）大：它的测试集只有 180 行，`all` 档与 300 档的差距只有 +0.047。

### 3.4 成本

| 项 | 实测 |
|:--|:--|
| 80 次拟合总墙钟 | **4,023 s（67 min）**：XGBoost 40 条 **535 s**（单条 1.6–52.2 s），RF 40 条 **3,488 s**（单条 1.4–449.2 s） |
| 两份 pack 造特征 | 约 6 分钟（29,602 个唯一结构，10 进程并行 + 缓存） |
| `pb26_scaling_cp` 超参搜索 | **约 45 分钟**（30 trials × 5 折，单 trial 5 次拟合在 17,793 行上） |
| GPU | **0**（`device=cpu`） |

> **RF 的墙钟不是"跑错了"**：22,242 行 × 2,253 列、100 棵树、`n_jobs=1`（确定性要求），单条 449 s 是实测。
> 合计 4,023 s 都是真实运行时间，`gpu_hours.csv` 按 B-3 的口径对 CPU 行也写 `wall_clock_s / 3600`——**D 核 GPU 预算时只取 `wall_clock_s`，别把这 1.12 "GPU-h" 加进 GPU 池子**。

---

## 4. C-b-3 · 架构迁移的物化与官方地板（2.0 pH）

**完整凭据在 [`docs/Arch Materialization.md`](Arch%20Materialization.md)**，这里只放四条验收的结论：

| # | 要求 | 实测 | 结论 |
|--:|:--|:--|:--|
| ① | 行数四条全对（6,478 / 6,110 / 368 / 36,828） | **6,478（6,110 + 368）/ 36,828（18,414 + 18,414）**，五折 `train_5182…5183` | ✅ |
| ② | 官方地板复现要写出具体数值；OOD combined 差 > 0.1 先查物化 | OOD **五折集成** random **0.5357**（官方 0.5355）/ block **0.5813**（0.5807）/ combined **0.5590**（0.5585），差 **+0.0005 ≪ 0.1** | ✅ |
| ③ | 组分覆盖完备（每个唯一 `psmiles` 有且只有一行） | holdout 唯一组分 **683** 个、索引 **683** 行、`missing=[]` / `duplicated=[]`；逐行 `feature_sha256` 683/683 自洽 | ✅ |
| ④ | 含 3/4/6 个 `*` 的结构不许静默丢 | **0 个组分解析失败**（683 全部算出特征）；`component_index.csv` 记了 `n_stars`，缺行清单 `component_failures.csv` 只有表头 | ✅ |
| ⑤ | 三脚本的 `--help` 与用法抄进文档，C-c 照着跑不需要问 | `Arch Materialization.md` §1.1 / §2 / §3 / §6 逐步写了命令 | ✅ |

**ID 一侧的差要如实说**：交替（611 行/折）逐折与官方差 ≤ **0.001**；均聚（**37 行**/折）最大差 **0.0120**。
原因已定位：官方 `Models/RDKit_RF/requirements.txt` **未钉版本**（`scikit-learn>=0.22.2`），本机用 1.4.2；
而同一套模型在 36,828 行的 OOD 上集成后差 < 1e-3——**差的是森林在 37 个测试点上的版本漂移，不是物化**。
对账工具两种容差都写进 `arch_floor_summary.json`（`tolerance: 0.02` 判绿、`would_fail_at_0.01: true`），不藏。

**臂已定（D-49，2026-11-15 仓库所有者拍板）**：arch 只跑 **`a2_conventional` 单臂 = 1 臂 × 5 折 × 3 种子 = 15 runs**，
全程 CPU；`pb26_arch_transfer_ea.yaml` 出 **v3**、`configs/tasks/SHA256SUMS` 升 **v5**（D-39 的 a2+a5 作废）。
所以 C-b 只需把 `configs/experiments/arch_transfer.yaml` 的 **`data` 段补实**——已补（四条行数、折名、holdout 身份、
683 个组分、官方地板复现值都在里面）；`eval` 段是 C-c 按 D-49 写的 15 runs，`frozen_at` 仍为 null，
**双方核对行数之后本文件才冻结**。

---

## 5. C-b-4 · 运行记录审计：批量执行与退单（1.5 pH）

**逐批记账在 [`docs/Audit Log.md`](Audit%20Log.md) 的「运行侧执行记录（C-b 维护）」章节**，这里只放结论：

| # | 批次 | 条数 | 退单 |
|--:|:--|--:|--:|
| 1 | B 已交全部（A3 48 + A4 48 + OpenPoly 450） | 546 | 0 |
| 2 | C-a 的 A1 + A2 | 96 | 0 |
| 3 | **C-b-2 的 OpenPoly 经典臂** | **225** | 0 |
| 4 | **C-b-1 的 scaling（XGBoost 一侧）** | **40** | 0 |
| 5 | A5 批 | **0** | —（D-48 撤除，不是漏跑） |
| 6 | arch 15 runs（C-c，D-49 单臂） | **0** | —（未开跑；C-b 的地基已交，臂已定） |

- **条数对账**：写这份文档时 `results/raw_runs/` 是 **907** 条（546 + 96 + 225 + 40）；加上 C-c 之后落盘的
  arch 15 条，全仓现为 **922** 条（`_schema.json` 不算记录）。907 是中间态，不是错数。
- **退单**：`results/aggregate/audit_backlog.csv` **只有表头**（0 退单，主表定稿前必须清零这条现在是清着的）。
- **套件**：`pytest tests/test_run_records.py` 在 **907 条**时是 14 passed；现在 **922 条**上是
  **12 passed / 3 skipped**——3 条 skip 是"arch 划分未物化"的分诊（D-62），不是退单。
- **全量**：**两个数属于两种机器，不要混读**（Audit Log 已改成两行）。
  - `cb310` 纯代码检出：250 passed / 88 skipped / 10 failed（共 348）。10 红全是缺 `data/interim` 或
    `scaling_indices`（D-38 不入库）；88 skip 是没装 torch / chemprop。
  - 本机（有 `data/interim`、物化了 pack）：**见 `docs/Audit Log.md` 的 2026-11-17 行**。
    以前把 `cb310` 的 10 红写成"本机基线"，会让下一个人以为红灯可以忽略——这条已改。

**本轮补了套件的一处能力缺口（不放松任何断言）**：`_resolve_fold()` 原本只走 `loaders.load_split()`，
而双数据集任务在那条路上是**故意报错**的（D-54）→ 加一条"记录里有 `dataset_id` 就按数据集读划分"的寻址分支。
折名仍要能找到、`split_version` 仍要指向存在且哈希前缀对得上的文件。**交 C-c 会签。**

---

## 6. 环境与可复现（D-55）

| 项 | 值 |
|:--|:--|
| 解释器 | Python **3.10.14**（与 `env_core.json` 同版本） |
| 包 | `lock/core.pip.txt` 的 **34 个 pip 包逐个装成锁定版本**：`rdkit==2024.3.5` / `xgboost==2.0.3` / `numpy==1.26.4` / `pandas==2.2.2` / `scikit-learn==1.4.2` / `pyarrow==17.0.0` / … |
| 与 `env_core.json` 的差异 | **34 个 pip 层包全部相同**；conda 层 `setuptools` 84.0.0（记录 84.2.1）、`packaging` 26.3（相同）；`prefix` / `driver` 两栏过时（本机无 GPU 参与的记录） |
| 记录里的 `env_fingerprint` | 仍写 `results/env_fingerprints/env_core.json`（D-40 的先例："包装版本逐项相同，解释器/机器不同"），265 条新记录的 `notes` 与 A2 的 96 条同款措辞 |
| GPU | 本轮 **0 条 GPU 记录**；`device=cpu` 逐条可查 |
| 确定性 | 经典臂 `n_jobs=1` + `random_state=seed` + `tree_method=hist`；同 seed 两遍的复现性由 C-a 的 `a2_repeatability.json` 覆盖 |

---

## 7. 验收逐条（分工_C-b §2）

### C-b-1（1.5 pH）

| # | 要求 | 凭据 | 结论 |
|--:|:--|:--|:--|
| ① | 断言读的是快照（逐份核 `seed_42/*.npz` 的 sha256 与 `splits_manifest.csv`） | `tests/test_scaling_scan.py::test_split_snapshot_digests_match_the_manifest`（按 `03_splits.py` 的算法重算目录摘要） | ✅ |
| ② | 断言训练集 = `train_pool[:N]`（写成测试） | `test_tier_rows_are_the_official_prefix`（逐位相等）+ `test_tiers_are_nested_and_never_touch_the_test_set` | ✅ |
| ③ | 档位逐档齐全，缺档写明 | 交替 5 档（4 官方 + all）、均聚 **3 档**（2 官方 + all）；`scaling.yaml` 的 `datasets[].tiers_expected` 写明 | ✅ |
| ④ | 每档的 `subset_sha256` 写进 `scaling_cpu.csv` | 80 行逐行有；同 (dataset, tier) 的 5 折各不相同（折不同、池不同，这是对的） | ✅ |
| ⑤ | 与 A2 超参除训练集规模外逐字段一致 | `a2_config_hash` 语义由"超参只从 `a2_best_params.csv` 取"保证；`test_hyperparameters_are_not_copied_into_the_experiment_config` + `test_a2_frozen_config_is_untouched` 守着 | ✅ |

### C-b-2（2.5 pH）

见 §2.2 的五条（全 ✅）。

### C-b-3（2.0 pH）

见 §4 的五条（全 ✅，臂的第二位待定不属 C-b 职权）。

### C-b-4（1.5 pH）

| # | 要求 | 凭据 | 结论 |
|--:|:--|:--|:--|
| ① | 五批收尾都有一次执行记录 | Audit Log 的 6 行表（含"A5 0 条""arch 0 条"两条如实记录） | ✅ |
| ② | 注入错误的测试会红，且在 C-c 的凭据上复跑 | `pytest tests/test_run_records.py` 14 passed（套件自带的四处注入自测在内） | ✅ |
| ③ | 不合格的记录必须退，不许帮忙改 | 本轮 **0 条不合格**，所以 `audit_backlog.csv` 只有表头；没有改过任何一条记录 | ✅ |
| ④ | 条数对账 | 907 = 546 + 96 + 225 + 40，逐批说清 | ✅ |

---

## 8. 没做什么（以及为什么）

| 缺口 | 状态 | 卡在哪 |
|:--|:--|:--|
| OpenPoly 的 5–9 折敏感性（B-5 做了，C-b 没做） | **没做** | 分工_C-b 只要 225 条（主表 5 折）；做了会变成 450 条、且 D-36 明写"若也做 5–9，必须按 (task, fold, level) 三元组取哈希"。要做等于加一档预算 |
| scaling 的 `random_Cp`（9,331 行，`smiles_list` + 共聚比） | **没做** | 它没有官方划分、也不是主表任务（D-47 ⑤）；`train_sizes` 只有交替/均聚两个数据集有 |
| `polyBERT` 的 224 MB 字典 → polyBERT 臂的物化 | **没做** | 只有 polyBERT 臂需要；arch 的第二个臂还没定（§4 末尾） |
| arch 的 15 runs 与 OOD 三段式 | **没做** | 是 C-c 的活（分工_C-c C-c-2）；C-b 只交地基。臂按 D-49 定为单臂，`pb26_arch_transfer_ea.yaml` 已是 v3 |
| `A5` 的任何记录 | **0 条** | D-48：不可构造（三条链输入在数据里不存在），状态 `not_constructible_data_unavailable`，**禁止**写 `zero_gain` |
| RF 的 40 条 run 记录 | **不写** | `_schema.json` 的 arm 枚举里没有"森林"；它们只进 `scaling_cpu.csv`（D-53） |
| `test_harmonise` / `test_task_assembly` / `test_splits` 的 10 个红 | **修不了** | 缺 `data/interim/harmonised/*.parquet` 与 `data/raw/polybench26/Datasets/scaling_indices/`——D-38 明确不入库，要跑得先按 A-2 重下 653 MB |

---

## 9. 交接

| 交给谁 | 什么 | 拿它干什么 |
|:--|:--|:--|
| **D** | `results/aggregate/scaling_cpu.csv`（80 行，含 `dataset` / `tier` / `n_train` / `subset_sha256` / `learner` / `r2` / `rmse`） | 拟合 **κ**（H2 的判据）。横轴用 `n_train`，**按 learner 分开拟合**（XGBoost 调过参、RF 没调） |
| **D** | `results/aggregate/openpoly_cpu.csv` + `openpoly_cpu_n_index.csv` | 与 B-5 的 GNN 逐格相减，求 **H4 的过零点**：两边训练行 75/75 相同、划分 15/15 相同（§2.2） |
| **D** | `results/aggregate/arch_floor_compare.csv` + `arch_floor_summary.json` | 架构迁移的**官方地板**（H3 架构子句的参照线）+ 物化正确性的凭据 |
| **C-c** | `generated/shared_splits/`（canonical 6,478 + 5 折 + holdout 36,828） | 跑 15 runs（D-49 单臂）。**先断言组分覆盖完备**（683/683，`00d --verify` 可重核） |
| **C-c** | `data/processed/arch/component_features.parquet` + `component_index.csv`（`psmiles → feature_row` + `feature_sha256`） | OOD 三段式：逐组分预测 → `summarize_weighted_predictions.py` 加权合成 |
| **C-c** | `configs/experiments/arch_transfer.yaml` 的 `data` 段（已补实） | 配置引用 C-b 的数，不自己数；`eval` 段是你按 D-49 写的 15 runs，两边核对行数后本文件才冻结 |
| **C-c** | `results/aggregate/audit_backlog.csv`（0 行）+ Audit Log 的运行侧记录 | 主表定稿前确认"0 退单"；他每交一批，C-b 已跑过的批次结论可直接引用 |
| **全队** | `docs/DECISIONS.md` 的 **D-50…D-55** | 六个口径变更（编号、超参、档位取法、RF 归档、双数据集寻址、环境）逐条待会签 |
