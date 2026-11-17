# Main Table Notes（主表口径说明）

> C-c-3 的配套文档。**列名只加不改**（`分工_D` §5 冻结项）；D 的对账脚本按列名读，改列名 = 该阶段重做。
> 本文件记录：列名契约、32 格口径、缺格约定、CI 占位约定、与排行榜对齐的入口。

---

## 1. 列名契约（v0，2026-11-13 由 C-c 定，冻结后只加不改）

| 列 | 含义 | 谁填 |
|:--|:--|:--|
| `task` | `configs/tasks/<task>.yaml` 的 `task_id` | C-c |
| `benchmark` | `polybench` / `polybench26`（OpenPoly 不进这 32 格，见 §4） | C-c |
| `arm` | `a1_floor` / `a2_conventional` / `a3_graph` / `a4_polybert`（`a5_chain` 已撤除，D-48） | C-c |
| `n_runs` | 该格包含的 run 条数；缺格留**空**，不许填 0 | C-c |
| `mean_rmse` / `std_rmse` / `mean_mae` / `std_mae` / `mean_r2` / `std_r2` | 该格全部 run 的均值与**样本**标准差（`main_table.yaml` **v3** `spread: sample_std_over_cell_runs`；PolyBench 格 n=3、PB26 格 n=15，与 交付_C-a §4 的 ± 同口径）；缺格留空，不许填 0 或均值充数 | C-c |
| `run_ids` | 该格全部 `run_id`，`;` 相连；每一格都能点回 `results/raw_runs/<run_id>.json` 与 `results/aggregate/predictions/<run_id>.parquet` | C-c |
| `missing_reason` | 缺格原因：`blocked_no_gpu`（占位没跑）/ `not_constructible_data_unavailable`（无输入可喂，D-48）；已填格留空 | C-c |
| `leaderboard_value` | PolyBench 排行榜公开数（含出处版本，C-a 提供）；**只落在 `a2_conventional` 行**（比较臂），其余行留空 | C-a → C-c 合并 |
| `leaderboard_diff_class` | 与排行榜的差分类：`implementation` / `scale_only` / `unexplained`（**v3 起 `protocol` 拆成后两类**，D-63）；同样只落在 `a2_conventional` 行 | C-a → C-c 合并 |
| `ci_rmse_low` / `ci_rmse_high` | RMSE 的重采样置信区间，**留空由 D 回填**（粗版不许等 CI） | D |
| `notes` | 口径备注（每格一句，写清"这个数怎么来的"） | C-c |

## 2. 32 格口径（2026-11-14 按 D-48 修订：A5 撤除）

- 8 任务 × **4 臂** = **32 格**（D-48 修订；D-17 的"5 臂"含已被判定**不可构造**的 A5）。任务与评估单元：
  - 6 个 PolyBench 任务 × `official_70_30` 单折 = 6 单元；
  - 2 个 PolyBench26 任务（`pb26_homopolymer_density`、`pb26_polymetrix_tg`）× `fold_00..04` = 10 单元；
  - 合计 **16 评估单元**；每格 = 该单元 × 3 种子（PolyBench 每格 3 条，PB26 每格 15 条）。
- 每臂只放 **1 个学习器**（C-a 建议口径，`configs/experiments/main_table.yaml` v2 的 `expected_runs` 为准）：
  A1 = Dummy、A2 = XGBoost；kNN / RF 是"学习器敏感性"，另存一张表、不进主表。
- **A5 不进表，也不标成"没跑"**：它是 `not_constructible_data_unavailable`（三条链输入在数据里不存在，D-47），
  不是 `zero_gain`——这两句话对读者的含义完全不同。替代证据是 D-46 的零信息对照（+3 列置换的 R² 地板）。
- A3 = chemprop 2.1.0 MPNN（B-3）；A4 = 冻结 polyBERT 嵌入 + ridge 头（B-4，闭式解，同 seed 两次逐位相同）。
- 汇总口径：先按 (task, arm) 收集全部 run 的 `metrics`，再算均值与**样本**标准差。ID 才允许这样汇总；
  **OOD 不得给"均值 ± SE"**（D-09），OOD 在 `arch_ood_ensemble.json` 里是集成点值，不进这张表。

## 3. 缺格约定（组规，不许破）

- 缺格**留灰**：`n_runs` 与六个数字列留空，`missing_reason` 写 `blocked_no_gpu`
  （或 `not_constructible_data_unavailable`，只用于 A5 这种"输入不存在"的情况）。
- **不许填 0、不许填均值**：填 0 不会报错，但会一路传进 D 的结论。
- **32 格已全部填实（2026-11-15）**：A1 / A2 的 16 格来自 C-a 的 96 条记录（`a1_a2_index.csv`），
  A3 / A4 的 16 格来自 B 的 96 条（各 48；A3 的另外 450 条属 OpenPoly 扫描，不进这 32 格）。
  无缺格、无 `blocked_no_gpu`；A5 的 8 格已按 D-48 从表中删除（删除 ≠ 填 0，A5 的去向写在 §2）。
- 两个枚举不要混用：`missing_reason` 答"这格为什么空"，run 记录的 `status`（`ok` / `blocked_no_gpu` /
  `failed`）答"这次运行怎么了"（D-48 ⑥）。

## 4. OpenPoly 的 450 条不进这 32 格

- B-5 的 OpenPoly 规模扫描（主表 225 + 5–9 折敏感性 225）是 **3 性质 × 5 档**的另一张网格，
  入口是 `results/aggregate/openpoly_gpu_index.csv`（450 行）与 `openpoly_n_index.csv`（150 行）；
  C-b 的 CPU 经典臂 225 条落 `openpoly_cpu.csv`。D 的 H4 从这些表读，不从这里读。
- 记账纪律（D-37）：`gpu_hours.csv` 是共用表，任何"某项花销"必须按 `arm` / `run_id` 归属筛，
  **不能整表求和**。

## 5. 已知口径备注（每格一句）

- `pb_t2_permeability`：标签与 RMSE/MAE 同尺度（约 1e-11 量级），RMSE≈0 是尺度效应不是满分；
  读这一格请以 `mean_r2` 为准（A3 ≈ −0.01，A4 ≈ 0.002）。
- `pb_t3_iwlt` 的 RMSE 量级（数百）同样由标签单位决定，跨任务比 RMSE 前先看单位表
  `data/manifests/unit_decisions.csv`。
- A4 的 `std_*` ≈ 0（闭式解确定性），不是"特别稳"，是"不掷骰子"（B-4 交付文档 §5 原话）。
- **排行榜两列的口径**：`leaderboard_value` = 官方数（PolyBench 六格取平台"最佳树" R²；PB26 两格取
  `RDKit_RF` 折间均值 RMSE，同划分）。`leaderboard_diff_class`：Tg / Tm 与 PB26 两格 = `implementation`
  （同尺度同划分，差在训练量/编码/调参）；modulus / permeability 两格 = **`scale_only`**；
  dielectric_loss / iwlt 两格 = **`unexplained`**（D-63 分类，D-66 把支撑它的数字换成 3 种子可复算值）。
- **四格的同尺度实测（D-63 分类，D-66 入账）**：把同一套冻结超参放回官方的 **ln(y)** 空间复跑，
  4 任务 × 2 臂 × 3 种子 = **24 行**，全部落在 `results/aggregate/log_target_sensitivity.csv`
  （`python scripts/04_run_cpu.py log-target` 重建）。这张表**不进主表、不写 `raw_runs/`**，
  `expected_runs` 仍是 96。A1 地板在同一空间一起算，否则两列不能相减：

  | 任务 | 我方 raw（A2） | A1 地板（ln） | 我方 **ln**（A2，均值±sd） | 官方（ln） | ln 尺度差 | 判读 |
  |:--|--:|--:|--:|--:|--:|:--|
  | `pb_t2_permeability` | −0.0018 | −0.0002 | 0.6203 ± 0.0020 | 0.6600 | −0.040 | 纯尺度问题（−6.0%） |
  | `pb_t2_modulus` | +0.0310 | −0.0004 | 0.3261 ± 0.0155 | 0.3485 | −0.022 | 纯尺度问题（−6.4%） |
  | `pb_t3_iwlt` | −0.0530 | −0.0048 | 0.4782 ± 0.0273 | 0.7519 | **−0.274** | 仍是实质差距（−36.4%） |
  | `pb_t3_dielectric_loss` | +0.5931 | −0.0171 | 0.1737 ± 0.0000 | 0.4195 | **−0.246** | **方向反转**：raw 看着赢、同尺度其实输 |

  所以"`protocol` 不可减、不归因"这句话只对一半——**两格可以减**（换到同一把尺子后差 6%），
  另两格确实还没解释。D 写 Λ 时按 `scale_only` / `unexplained` 分开处理，不要把这四格混成一类。
- **别把这两格当"只差 2%"引用（D-66 的修正）**：D-63 初版只有 seed 0 单点，modulus 记的是 0.3413（−2.1%）；
  跑满 3 个种子是 0.3413 / 0.3267 / 0.3104，均值 **0.3261（−6.4%）**——单点恰好是最高值。
  iwlt 同理（0.4640 是三种子里最低的，均值 0.4782）。凡引用幅度，引 `log_target_sensitivity.csv` 的均值，
  不引 D-63 那四个单点。
- **这 24 行只答 A2 一臂，不是 Q-10 的答案**：Q-10 要求"四臂同时施加对数变换"，那会改 `config_hash`
  并让 96 条全部重跑。这里只是"同一把尺子上我们站在哪儿"，不构成协议改版。
- **ID 指标只覆盖一半样本**：PB26 的官方划分把训练集切成 10 个 partition，第 `k` 折 test 取 `k`、
  val 取 `k+5`，所以 5 折 test 互不重叠、**并集 = 全量的一半**（6,478 → 3,240；交替 3,055、均聚 185）。
  详见 `docs/验收_P3_分工C-c.md` §5.5。报告里写"在 6,478 行上"是错的，必须写"在 3,240 行（半数样本）上"。
- **重尾任务另看 Spearman**：R² 由少数极值主导（permeability 的 A2 输出甚至是常数，R² 为负）。
  `results/aggregate/spearman.csv`（`05b_report_tables.py spearman`）给出 32 格逐 run 的秩相关均值/极值，
  回答"排序还灵不灵"。它**不回填记录**——记录是冻结的协议身份（D-64）。
- **`n_runs` 的含义**：该格实际包含的 run 条数，PolyBench 格 **3**（1 折 × 3 种子）、PB26 格 **15**
  （5 折 × 3 种子）。`n_runs` 不是"单元数"，也不是格数；32 格一共 **192** 条 run。
- **宽度不要混读**：主表的 `a2_conventional` 走 XGBoost，入模 **2,253** 列；`learner_sensitivity` 里的
  RF / kNN 经 `SimpleImputer(strategy="median")` 后是 **2,245** 列（8 个 `BCUT2D_*` 列全 NaN，fit 时被丢弃
  并告警）。这是设计如此，不是缺列（D-48 ③）。

## 6. 更新流程

- 骨架生成：`PYTHONUTF8=1 python scripts/_scratch/build_main_table.py`（scratch，等 C-a 的
  `configs/experiments/main_table.yaml` 落地后并入正式入口）。
- C-a / C-b 每交一批，重跑一次、更新本文 §2/§5；**11-18 中午前粗版给 D**（CI 列留空），
  11-18 晚定稿，CI 由 D 回填。
