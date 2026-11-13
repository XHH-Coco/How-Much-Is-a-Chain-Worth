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
| `mean_rmse` / `std_rmse` / `mean_mae` / `std_mae` / `mean_r2` / `std_r2` | 该格全部 run 的均值与总体标准差；缺格留空，不许填 0 或均值充数 | C-c |
| `run_ids` | 该格全部 `run_id`，`;` 相连；每一格都能点回 `results/raw_runs/<run_id>.json` 与 `results/aggregate/predictions/<run_id>.parquet` | C-c |
| `missing_reason` | 缺格原因：`blocked_no_gpu`（占位没跑）；已填格留空 | C-c |
| `leaderboard_value` | PolyBench 排行榜公开数（含出处版本，C-a 提供） | C-a → C-c 合并 |
| `leaderboard_diff_class` | 与排行榜的差分类：`protocol` / `implementation` / `unexplained`（C-a 提供） | C-a → C-c 合并 |
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
- 汇总口径：先按 (task, arm) 收集全部 run 的 `metrics`，再算均值与总体标准差。ID 才允许这样汇总；
  **OOD 不得给"均值 ± SE"**（D-09），OOD 在 `arch_ood_ensemble.json` 里是集成点值，不进这张表。

## 3. 缺格约定（组规，不许破）

- 缺格**留灰**：`n_runs` 与六个数字列留空，`missing_reason` 写 `blocked_no_gpu`
  （或 `not_constructible_data_unavailable`，只用于 A5 这种"输入不存在"的情况）。
- **不许填 0、不许填均值**：填 0 不会报错，但会一路传进 D 的结论。
- 32 格在**数据上已无真缺格**：A1 / A2 的 16 格有 96 条记录（`a1_a2_index.csv`），A3 / A4 的 16 格
  有 96 条（各 48，B 组；A3 的另外 450 条属 OpenPoly 扫描，不进这 32 格）。
  但 `main_table.csv` 现在仍是 **40 行旧骨架**：A1 / A2 的 16 格还挂着 `blocked_no_gpu` 占位、
  A5 的 8 格尚未删除——**C-c 重跑一次骨架即可填实**，不用等 C-a。
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
- **宽度不要混读**：主表的 `a2_conventional` 走 XGBoost，入模 **2,253** 列；`learner_sensitivity` 里的
  RF / kNN 经 `SimpleImputer(strategy="median")` 后是 **2,245** 列（8 个 `BCUT2D_*` 列全 NaN，fit 时被丢弃
  并告警）。这是设计如此，不是缺列（D-48 ③）。

## 6. 更新流程

- 骨架生成：`PYTHONUTF8=1 python scripts/_scratch/build_main_table.py`（scratch，等 C-a 的
  `configs/experiments/main_table.yaml` 落地后并入正式入口）。
- C-a / C-b 每交一批，重跑一次、更新本文 §2/§5；**11-18 中午前粗版给 D**（CI 列留空），
  11-18 晚定稿，CI 由 D 回填。
