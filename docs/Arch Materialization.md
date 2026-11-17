# Arch Materialization（架构迁移的物化与官方地板复现）

> **对应**：`docs/分工_C-b_规模扫描与迁移前半.md` 的 **C-b-3**（2.0 pH）
> **一句话**：官方只发**脚本**不发数据，所以这一步是"按官方脚本在本地把留出集物化出来、先用官方地板证明物化没做错"，
> 然后把 C-c 跑 **15 runs**（D-49 单臂）需要的三样东西（划分、组分特征、配置的 `data` 段）交出去。
> **许可**：task4 的训练源是 polyVERSE / VIPEA，属 **GTRC General Public Use License**（D-08），
> 不是主表那个 CC BY 4.0。结果表须随附 GTRC 全文、免费分发、**不得商用**；GTRC 的条款不能套到 PolyBench 的任务头上。

---

## 1. 取件：钉 commit、逐文件 sha256

| 项 | 值 |
|:--|:--|
| 上游 | `rlearsch/PolymerBenchmark2026` |
| commit | `83655bc25901a8cabe3a1e5474b7ec26f01b919f`（与 `外部官方来源` 里 scaling 结果表同一个 commit） |
| 取件清单 | `data/manifests/external_task4_sources.csv`（**19 份、17.77 MB**，逐份 `bytes` + `sha256` + `source_url`） |
| 落点 | `data/raw/polybench26/Datasets/**`（`data/` → `data/` 除 D-38 那 20 份外不入库，**字节不进版本库**） |
| 重跑 | `python scripts/00c_fetch_task4.py`（幂等：已缓存的份不重下，清单照写） |

取的三类东西，以及**为什么必须取**：

1. **task4 本体**（`Datasets/task4_architecture_transfer/`）：`README.md`、`evaluate_rdkit_ood.py`、
   `summarize_weighted_predictions.py`、`task4_all_model_scores.csv`（官方已发布的成绩，对账靶子）、
   `train_models.sh`、`legacy_train.csv`。官方**只发布物化工作流**，划出集与 canonical 表要本地跑脚本才出现（`SOURCES.md`【坑七】）。
2. **物化脚本与依赖**（`Datasets/Dataset_construction_scripts/`）：`create_task4_shared_splits.py` 与它
   `import` 的 `polymer_conversions.py` / `rdkit_descriptor_cache.py` / `task4_architecture_transfer.py`，以及生成 VIPEA 表的 `process_Vipea_data.py`。
3. **训练源表**：`PSMILES/polyVERSE/electron_affinity/*.csv`（369 行）、
   `files/polymer-chemprop-data/{dataset-poly_chemprop.csv,dataset.csv}`（共 17 MB，VIPEA 的 EA/IP 源）。

### 1.1 两步"生成物"（官方 README 明写是 ignored outputs，必须自己跑一次）

```bash
# ① VIPEA：从 tracked 的 chemprop 源表生成 EA/IP × {alternating, random, block}
cd data/raw/polybench26/Datasets/Dataset_construction_scripts
python process_Vipea_data.py
# → wPSMILES/Vipea/{EA,IP}/{alternating,random,block}_{EA,IP}.csv + PSMILES/Vipea/EA/alternating_EA.csv

# ② polyVERSE 的 wPSMILES：从 PSMILES 表转一遍
python PSMILES_to_wPSMILES.py <Datasets>/PSMILES/polyVERSE/electron_affinity <Datasets>/wPSMILES/polyVERSE/electron_affinity
```

实测产物：`alternating_EA.csv` 847 KB、`random_EA.csv` 4.26 MB、`block_EA.csv` 4.63 MB、
`wPSMILES/polyVERSE/electron_affinity/*.csv` 24.5 KB。

---

## 2. 物化：一条命令、四条行数

```bash
python Datasets/Dataset_construction_scripts/create_task4_shared_splits.py \
  --output-dir generated/shared_splits --overwrite
# --polybert-dictionary 只有 polyBERT 臂需要（那个 224 MB 的 pkl）；经典臂可省，本次省略
```

**验收①：四条行数四条全对（不是"差不多"）**

| 项 | 要求 | 实测 | 结论 |
|:--|--:|--:|:--|
| canonical 合计 | 6,478 | **6,478** | ✅ |
| ├ 交替（alternating） | 6,110 | **6,110** | ✅ |
| └ 均聚（homopolymer） | 368 | **368** | ✅ |
| OOD holdout | 36,828 | **36,828** | ✅ |
| ├ random | 18,414 | **18,414** | ✅ |
| └ block | 18,414 | **18,414** | ✅ |
| 五折 train 行数 | 80% | **5,182** / 折（`fold_00/train_5182` … `fold_04/train_5183`） | ✅ |
| OOD 唯一组分 | —（本轮新增的观测） | **683** | 供 C-c 用 |

物化产物（`generated/shared_splits/`，**不入库**，GTRC 派生数据）：

```
manifest.json              源表 sha256、seed 42、五折行数、架构计数
canonical_train.csv        6,478 行（row_index + psmiles + wpsmiles + electron_affinity + architecture + 溯源）
architecture_holdout.csv  36,828 行（psmiles_a/psmiles_b + molar_fraction_a/b + architecture）
ood_components.csv           683 行（全部唯一组分，官方给的是"逐组分预测"的口径）
ood_wpsmiles.csv           8.9 MB（chemprop 臂直接吃的那份）
fold_0X/train_*/{indices,PSMILES,wPSMILES,RDKit_descriptors,polyBERT}/
```

> 物化脚本自带的三条口径（照抄官方 README，不另立规矩）：① 交替 VIPEA 的 wPSMILES 先转成 PSMILES；
> ② 合并后**按 PSMILES 去重**，28 对重复用**平均 EA** 与**最早的源 wPSMILES**；
> ③ 五折是 **PSMILES 分组**的 80/10/10、`seed 42`，OOD holdout **固定、永不进入训练或调参**。

---

## 3. 官方地板复现：先证明物化没做错，再谈 15 runs

**思路**：官方给了它自己的 RDKit RF 成绩（`task4_all_model_scores.csv` 的 `RDKit RF` 15 行）。用它当靶子——
如果我在**自己的物化产物**上跑官方训练脚本，能复现官方的数，就说明"物化 + 组分加权合成"这条链路没错；
差一个量级就说明物化错了，那时**不许调模型**（分工_C-b 验收②、红线 4）。

```bash
# 训练官方的 RDKit RF（5 折 × 100 树，官方配方，不调参）
cd data/raw/polybench26/Models/RDKit_RF
python train_rf_scaling.py --split-root <repo>/generated/shared_splits \
       --save-root <repo>/generated/models/rdkit_rf --seed 42

# 官方 OOD 评估（逐组分预测 → 按摩尔分数加权合成）
python <repo>/data/raw/polybench26/Datasets/task4_architecture_transfer/evaluate_rdkit_ood.py \
  --split-root <repo>/generated/shared_splits \
  --result-root <repo>/generated/models/rdkit_rf \
  --output-root <repo>/generated/models/rdkit_rf/ood \
  --datasets-root <repo>/data/raw/polybench26/Datasets

# 对账（本仓库脚本）：ID 逐折 + OOD 单折留档 + OOD 五折集成三个点值 vs 官方
python scripts/00e_arch_floor.py
```

### 3.1 OOD 五折集成（**唯一可报口径**，D-09）

| 架构 | 本轮复现 RMSE | 官方 RMSE | Δ | 结论 |
|:--|--:|--:|--:|:--|
| random | **0.535743** | 0.535517 | **+0.000226** | ✅ |
| block | **0.581319** | 0.580658 | **+0.000661** | ✅ |
| combined | **0.558996** | 0.558544 | **+0.000452** | ✅ |

**与验收②的报警线比**：combined 差 **0.00045 ≪ 0.1** → 物化与合成链路复现得上，**可以交给 C-c 跑 15 runs（D-49 单臂）**。
`results/aggregate/arch_floor_summary.json` 里 `ood_ensemble.alarm = false`。

> **为什么必须是"五折集成后一个点值"**：OOD holdout 在五折之间**固定不变**，所以五折的 OOD 指标不是独立重复，
> 官方 README 原文禁止对五个折级 OOD RMSE 做 t 检验。做法：每个聚合物把 5 条预测取平均，再算 random / block / combined
> 各一个 RMSE/MAE/MSE（`scripts/00e_arch_floor.py` 的 `ensemble` 段，含逐折留档但**不作为判决**）。

### 3.2 ID（分布内）：逐折对账，差在哪

| 架构 | 测试行/折 | 本轮 vs 官方 的最大 \|ΔRMSE\| | 结论 |
|:--|--:|--:|:--|
| alternating | 611 | **0.000953** | ✅ 逐折差 ≤ 0.001 |
| homopolymer | 37 | **0.012049** | ⚠️ **在 37 个测试点上**的森林抖动 |

**差在哪，说清楚（不粉饰）**：

1. **官方环境没钉版本**：`Models/RDKit_RF/requirements.txt` 只写 `scikit-learn>=0.22.2`、`numpy>=1.18.1`，
   没有一个上界也没有一个 pin。本机用 scikit-learn **1.4.2**（P0 冻结）。随机森林的分裂在"37 行测试集"这种量级上
   会随版本漂移，**这一项无法逐位复现，也不该假装能**。
2. **同一套模型在 36,828 行的 OOD 上集成后与官方差 < 1e-3**：如果物化错了（行数、组分覆盖、加权、去重顺序），
   这么大的样本上会立刻暴露成量级差；它没有。所以判据是"物化对不对"，不是"逐折 611/37 行能不能逐位对齐"。
3. **对账工具两种口径都记**：`arch_floor_summary.json` 的 `id` 段同时写 `tolerance: 0.02`（本轮判绿）与
   `would_fail_at_0.01: true`（按 0.01 会判红）。**不藏**：ID 逐折不是判决口径（D-09 只认五折集成的 OOD），
   它是一次体检。

产物：`results/aggregate/arch_floor_compare.csv`（**28 行** = ID 10 + OOD 单折 15 + OOD 集成 3）与
`results/aggregate/arch_floor_summary.json`。

---

## 4. 组分特征：C-c 唯一会踩的坑

官方 OOD 不是"逐行预测"，而是**逐组分预测 + 按摩尔分数加权合成**，所以 C-c 需要一张"组分 → 特征"的表。

```bash
python scripts/00d_arch_components.py            # 算（683 个组分，并行；RDKit 缓存命中 385 / 新算 298）
python scripts/00d_arch_components.py --verify   # 只读复核：覆盖完备 + 逐行指纹
```

| 产物 | 内容 |
|:--|:--|
| `data/processed/arch/component_features.parquet` | **683 行 × 2,253 列**（205 描述符 + 2048 位指纹，列序与 A-6/B-1 逐列相同），**不入库** |
| `data/processed/arch/component_index.csv` | `psmiles, feature_row, n_stars, feature_sha256, n_nan_descriptors`（**一个 psmiles 有且只有一行**） |
| `data/processed/arch/component_failures.csv` | 算不出特征的组分 + 它在 holdout 里出现多少次（本轮 **0 行**） |
| `data/manifests/arch_component_manifest.json` | 行数、列数、NaN 单元、`features_sha256`、覆盖结论 |

**验收③（唯一性）实测**：holdout 里唯一组分 **683** 个，索引 **683** 行，`missing = []`、`duplicated = []` →
**覆盖完备**。这条不是注释，是断言：`--verify` 逐行重算 `feature_sha256` 并核 `feature_row` 不越界，**683/683 全对**。

> **给 C-c 的一句话**：拿 `architecture_holdout.csv` 的 `psmiles_a/psmiles_b` 去 `component_index.csv` 取行号、
> 到 `component_features.parquet` 取特征；**缺一条就停，回来找 C-b，别自己补**。
> `feature_sha256` 是"你读到的这一行就是表里那一行"的凭据。

**验收④（不许静默丢）**：本轮 **0 个**组分解析失败（683 个全部算出来了），所以 `component_failures.csv` 只有表头。
`n_stars` = 3/4/6 的支化结构也在其中：它们**能**算 POD 描述符与指纹（A-6 拒的是 wPSMILES，不是特征），
所以这里不缺行——但 `component_index.csv` 把 `n_stars` 记下来了，C-c 若要按星号数分层可以逐行筛。

---

## 5. 交给 C-c 的 `data` 段（`configs/experiments/arch_transfer.yaml`）

| 键 | 值 |
|:--|:--|
| `canonical_rows` | 6,478（alternating 6,110 / homopolymer 368） |
| `folds` | `fold_00` … `fold_04`，每折 `train_5182`（80%）/ val（10%）/ in-distribution test（10%），**PSMILES 分组**、`seed 42` |
| `holdout` | `generated/shared_splits/architecture_holdout.csv`，36,828 行（random 18,414 / block 18,414），**固定、不参与任何拟合** |
| `holdout_components` | `ood_components.csv` 683 个唯一 psmiles → `data/processed/arch/component_index.csv` |
| `oof_reporting` | **五折集成后一个点值**（D-09）；禁止对五个折级 OOD RMSE 做 t 检验 |
| `target` | `electron_affinity`（EA，单位 eV），单一性质 |
| `licence` | GTRC General Public Use License（polyVERSE / VIPEA） |

**臂已定（D-49）**：arch 只跑 **`a2_conventional` 单臂 = 15 runs**（1 臂 × 5 折 × 3 种子，全程 CPU），
`pb26_arch_transfer_ea.yaml` 出 v3、`SHA256SUMS` 升 v5；D-39 的 a2+a5 作废（a5 的三条链输入在数据里不存在，D-47）。
物化与臂无关，所以 `data` 段不受这条决定影响——C-b 已在 `configs/experiments/arch_transfer.yaml` 把 `data` 段补实。

---

## 6. 复跑清单（C-c 照着跑，不需要问 C-b）

```bash
# 0) 取件（幂等）+ 生成 VIPEA/polyVERSE 的 wPSMILES（在 Dataset_construction_scripts 里）
python scripts/00c_fetch_task4.py
python process_Vipea_data.py
python PSMILES_to_wPSMILES.py <Datasets>/PSMILES/polyVERSE/electron_affinity <Datasets>/wPSMILES/polyVERSE/electron_affinity

# 1) 物化（约 3 分钟；RDKit 描述符走它自己的缓存）
python Datasets/Dataset_construction_scripts/create_task4_shared_splits.py --output-dir generated/shared_splits --overwrite

# 2) 组分特征（约 10 秒，并行）
python scripts/00d_arch_components.py && python scripts/00d_arch_components.py --verify

# 3) 地板（可选，但"物化没做错"的凭据就是它）
python .../train_rf_scaling.py --split-root generated/shared_splits --save-root generated/models/rdkit_rf
python .../evaluate_rdkit_ood.py --split-root generated/shared_splits --result-root generated/models/rdkit_rf \
       --output-root generated/models/rdkit_rf/ood --datasets-root data/raw/polybench26/Datasets
python scripts/00e_arch_floor.py
```

三个官方脚本的 `--help` 已逐条抄在上面 §1.1 / §2 / §3 的命令里；不要自己重写"加权合成"，
`summarize_weighted_predictions.py` 会对「组分缺预测」「摩尔分数不合 1」「psmiles 重复」直接抛错。

**三条不明显的约束（2026-11-17 本机整链复跑时实测，见 `docs/重跑_arch_异机复现_2026-11-17.md`）**：

1. **第 0 步不能省**。`00c` 的清单只记上游字节，`wPSMILES/polyVERSE/...` 是**派生**数据，不在清单里，
   而上游也没有这个路径（直接取会拿到 404 正文）。缺了它，第 1 步报
   `FileNotFoundError: ...wPSMILES\polyVERSE\electron_affinity\...csv`。
2. **第 2 步不能省，也不能并进 `arch --setup`**。`--setup` 只做折与 ID 特征；跳过 `00d` 会让
   `arch --ood` 的第一行断言就退：`RuntimeError: 组分覆盖不完备：missing=[...]`。
   这个断言是设计好的闸，不是在报环境问题。
3. **第 1 步不幂等**。`create_shared_splits()` 用的是 `output_dir.mkdir(parents=True)`，没有 `exist_ok`：
   上一次失败留下的**空目录**会让重跑直接 `FileExistsError: [WinError 183]`。先 `rmdir` 再跑，或用 `--overwrite`
   （后者会 `shutil.rmtree` 整个输出目录，想清楚再用）。

---

## 7. 本轮没做（以及为什么）

| 缺口 | 状态 | 卡在哪 |
|:--|:--|:--|
| `--polybert-dictionary`（224 MB 的 pkl） | **没做** | 只有 polyBERT 臂需要；`a4_polybert` 不在 15 runs 的臂里（D-49 定的是 a2_conventional 单臂） |
| chemprop / polyBERT / periodic-graph 三条臂的地板 | **没做** | 分工_C-b 只要"官方地板"一条（RDKit RF）；另外三条要 GPU 与各自环境，且不属于 C-b |
| OOD 的 `IP`（电离能）一侧 | **没做** | task4 只做 `electron_affinity` 一个性质（`SOURCES.md`【坑七】已核实） |
| `legacy_train.csv` 的用途 | 只取未用 | 官方 README 未把它并入 task4 协议（5 折与 holdout 都来自 VIPEA + polyVERSE）；留着是为了核对时能追 |

---

## 8. 交接状态（2026-11-17 补）：45 份行级产物现在在哪

`results/aggregate/arch_ood_artifacts_manifest.csv` 里 45 条（30 份 `arch_component_preds/*.csv` +
15 份 `arch_ood_vectors/*.parquet`）都写着 `committed: no`。以前的信息只到这一步——"留本地"，
但**没写哪台机器、什么时候能删**，于是 D 要算 OOD 的 bootstrap 区间时没有任何落点（D-62）。

现在清单多了三列：`where`、`created_at`、`deletable_after`。当前状态是 **45/45 `unrecorded`**——
这是如实记录的缺口，不是填好了。**需要 C-c 本人在物化机上补一次**：

```bash
python scripts/04_run_cpu.py arch --verify-ood
```

这条命令分三件事报，不要混读：

| 报什么 | 含义 | 该怎么办 |
|:--|:--|:--|
| 缺 N 份 | 这台机器上没有这些文件 | 按打印的 `reconstruct` 命令重建，或向 C-c 取件 |
| 哈希不符 N 份 | 文件被改过（或清单是旧版） | 先查是谁改的，**不要**用改过的文件算区间 |
| `where` 未登记 N 份 | 文件在，但不知道它在哪台机器上 | 在清单里填机器标识与 `deletable_after` |

**取件另有两条路，都不依赖 C-c 在线**：① 重建——`arch --setup` 需要上游 19 份文件，
`00c_fetch_task4.py` 按 `data/manifests/external_task4_sources.csv` 的 URL + sha256 幂等拉取；
② 重跑 `arch --ood`——但它同样需要物化输入，所以在没有 `generated/` 的机器上等价于重建整条链。
两条路都需要那份 163 MB 级别的上游数据，**这就是为什么清单里的 `where` 比哈希更重要**。

### 8.1 2026-11-17：本机重建完成，45 份不再"下落不明"

在一台 `DSC6020_Project_Core`（Python 3.10.14 / rdkit 2024.3.5 / xgboost 2.0.3 / sklearn 1.4.2 /
numpy 1.26.4 / pandas 2.2.2 / pyarrow 17.0.0，七项与 `env_core.json` 全同）的机器上，按 §6 的五步
从上游重建了整条链，然后跑 `arch --run` 与 `arch --ood`：

| 判据 | 结果 |
|:--|:--|
| `canonical_sha256` | `632aa02d…`，与记录一致 |
| `features_sha256` / `pack_sha256` | `fed6173b…`，与记录一致 |
| `split_version` | `ad424d72…`，与记录一致 |
| 15 条记录指标 | Δ=**0.000e+00**（`git diff` 只有 `wall_clock_s` 与两个时间戳三行） |
| 135 个 OOD 指标值 | Δ=**0.000e+00** |
| 45 份产物 sha256 | **45/45 与清单一致** |

结论：**在环境可校验的前提下，确定性重跑可以替代"取件"**。清单的 `where` 已由 `unrecorded` 改为
`local:DESKTOP-6C1JF93（E:/How-Much-Is-a-Chain-Worth）`，`arch --verify-ood` 四项归零（D-65）。
