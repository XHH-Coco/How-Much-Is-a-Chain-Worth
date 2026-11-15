# Audit Log（审计台账）

> 全队审计 14 pH 按"谁的数字依赖它谁来查"分配：入口 5（D：划分与泄漏）、**运行记录 4（C：本文件）**、出口 5（D，自查）。
> 本文件只有结论与台账，没有结果——run 记录只能由训练脚本落盘，审计只有"退回"的权力，没有"帮忙改"的权力。
> 发现异常 → 退回产出人（由 C-b 执行退单）→ 台账记一条 → 产出人重跑后回传。

---

## 运行侧结论（C-c 维护）

### 套件（2026-11-13 交付）

- 实现：`tests/test_run_records.py`，14 条测试 = 第 0 条（schema）+ 六条断言 + 四处注入（各自必须变红）+ 2 条守护。
- 第 0 条：逐条记录过 `results/raw_runs/_schema.json`。**jsonschema 不在三套冻结环境里**（P0 冻结，现有代码刻意不引入），
  经用户 2026-11-13 拍板，套件内置 draft-07 **子集**校验器：字段清单与约束全部读 `_schema.json`（不手写），
  子集范围由 `SUPPORTED_KEYWORDS` 声明，并有测试守着"schema 只能用受支持关键字"。
- 六条断言：① 字段齐全（七项基础 + ok 六项）② 每个 ok run 有 `predictions/<run_id>.parquet`
  ③ 预测行数 = 该折 test 长度（含 `split_version` 哈希前缀 = 文件字节）④ run_id 唯一且文件名一致
  ⑤ `*.partial.*` 一律不许进 `results/raw_runs/` ⑥ device↔env 指纹（cuda↔gnn/polybert，cpu↔core）。
- 四处注入（删 `split_version` / partial 混入 / device 与 env 不符 / 重复 run_id）逐条验证**变红**；
  任何人为了让测试变绿而删断言 = 审计作废（红线）。

### 审计运行记录

| 日期 | 范围 | 条数 | 退单 | 结论 |
|:--|:--|--:|--:|:--|
| 2026-11-13 | B 已交全部（A3 48 + A4 48 + OpenPoly 450） | **546** | **0** | 全绿：546 条全部过第 0 条与六条断言，预测文件 546 份齐，行数与折对得上，无 partial 混入，device/env 全部一致 |
| 2026-11-15 | C-a 已交 A1/A2 批（`main_table.yaml` v2：16 单元 × 2 臂 × 3 种子） | **96** | **0** | 全绿：A1 48 + A2 48 全部过第 0 条与六条断言，预测 96 份齐、行数 = 官方测试行、device=cpu ↔ env_core（D-40 的 venv 备注不影响判定） |
| 2026-11-15 | 全量复跑（B 546 + C-a 96） | **642** | **0** | 套件 14/14 绿；主表 32 格引用 192 条（A1/A2/A3/A4 各 48），其余 450 条属 OpenPoly 扫描 |
| 2026-11-16 | C-c 已交 arch 批（15 条，D-49 单臂） | **15** | **0** | 全绿：15 条全部过第 0 条与六条断言；预测 15 份齐、行数 = 648（交替 611 + 均聚 37）；`split_version` 指向本地物化的 `splits/polybench26/arch/seed_42/fold_XX.npz` 且哈希前缀逐条对上；device=cpu ↔ env_core；`id_tier_metrics`/`id_tier_n` 两档分开落盘（验收①）；OOD 走官方三段式（验收②），五折只报集成点值（D-09） |

### 条数对账（657 的构成）

- 657 = B 546（A3 48 + A4 48 + OpenPoly 450）+ C-a 96（A1 48 + A2 48，`main_table.yaml` v2 口径）+ **C-c 15（arch 单臂 15 runs，D-49）**。
- A5：**0 条**，按 D-48 是 `not_constructible_data_unavailable`（D-47 证明三条链输入在数据里不存在），
  **不是** `zero_gain`、也不标 `blocked_no_gpu`——主表 32 格没有 A5 行（已删，见 Main Table Notes §2）。
- 对账口径：记录数 = 546 + C 组各批 `expected_*` 之和；`results/raw_runs/_schema.json` **不是记录**，统计时排除。
- 不在审计范围内：`results/aggregate/a4_finetune/`（B-4 微调对照 48 条，与冻结臂同 run_id，按 D-34 落独立目录、不进 `raw_runs/`，由 B 自管）。

### 测试基线说明

- 任务书基线：**280 passed / 7 skipped**（数据齐全的机器，`pytest tests`）。
- 2026-11-15 本机（缺 `data/interim/harmonised/*.parquet`——仓库有意不入库）：
  **230 passed / 86 skipped / 9 failed**。9 个失败全部是 `test_harmonise.py` / `test_task_assembly.py`
  里依赖 `data/interim` 的既有用例，与审计套件无关；86 skipped 同理（缺数据即跳过，或 core 环境无 chemprop）。
  套件本体 `pytest tests/test_run_records.py`：**14 passed**（642 条）。数据齐全机上应为 280 + 14 = 294 passed / 7 skipped。
- **一处移植修复（2026-11-15）**：`tests/test_arms_classical.py::test_record_validates_against_the_schema`
  原本在函数内 `import jsonschema`——只有 C-a 私装过该包的 venv 能跑（D-40），三套冻结环境里都没有它。
  按 2026-11-13 已拍板的口径改为复用本套件的内置子集校验器（`_schema_problems`），断言不删、校验语义不变；
  改完 `pytest tests` 从 10 failed 回到 9 failed 基线。改动在 C-a 的文件里，已如实登记，C-a 可在下次合并时认领。

### 待办

- C-b 每批收尾跑 `pytest tests/test_run_records.py`，退单记入上表；
- ~~C-c 自己的 arch 15 条（D-49 单臂）~~ **已交（2026-11-16，15 条 0 退单）**；任务配置按 D-57 保持 D-49 的 v3 / SHA256SUMS v5 一个字节没动（present+文件会弄红 A-5 的 `test_splits.py`，划分解析由 C-c 的 `arch.py::_load_arch_split()` 与套件 `_resolve_fold` 新分支承担）；
- 11-19 定稿：补齐上表、写最终结论（发布前必须 0 退单、0 未解释缺口）。

---

## 运行侧执行记录（C-b 维护，2026-11-15）

> 分工_C-b 的 C-b-4：**每批收尾就跑一遍套件、逐批记账**，只退回、不改数。C-c 在上面已审过
> **642 条**（B 546 + C-a 96）；C-b 在两批新数据交付后追加了两批，现总 **907 条**。
> 本机没有 `data/interim/harmonised/` 与 `data/raw/polybench26/Datasets/scaling_indices/`（D-38 明确不入库），
> 所以套件里依赖那两处的既有用例按 `skipped` 计——**套件本体（`tests/test_run_records.py`，14 条）全程可跑**。

### 逐批执行（C-b 追加的两批用粗体）

| # | 批次 | 何时跑 | 记录条数 | 退单 | 结论 |
|--:|:--|:--|--:|--:|:--|
| 1 | B 已交全部（A3 48 + A4 48 + OpenPoly 450） | 11-15 04:47 | **546** | 0 | 复核：与 C-c 11-13 的结论一致，全绿 |
| 2 | C-a 的 A1 + A2 | 11-15 04:47 | **96** | 0 | 全绿：`a1_floor` 48 + `a2_conventional` 48，`config_hash` 逐条等于冻结臂配置 |
| 3 | **C-b-2 的 OpenPoly 经典臂 225** | 11-15 04:47 | **225** | 0 | 全绿：225 条 `ok`、225 份预测、`arm=a2_conventional`、`device=cpu`；每条带 `n_level` / `subset_sha256` / `n_train_pool` / `n_level_saturated` |
| 4 | **C-b-1 的 scaling（XGBoost 一侧 40 条）** | 11-15 07:26 | **40** | 0 | 全绿：`dataset_id` / `fold_index` / `n_level` 齐全；`split_version` 指向该折自己那份 `fold_XX.npz`（哈希前缀逐条对上）。**另有 40 条 RF 行只进 `scaling_cpu.csv`、不进 `raw_runs/`**（`_schema.json` 的 arm 枚举里没有"森林"，D-53） |
| 5 | A5 批 | — | **0** | — | **没有记录可审**：A5 已按 D-48 撤除（三条链输入在现有数据里不存在，D-47），状态 `not_constructible_data_unavailable`。不是"漏跑"，是"不可构造" |
| 6 | arch（C-c） | — | **0** | — | C-c 尚未交批。臂已由 **D-49** 定为 `a2_conventional` 单臂 = **15 runs**（`pb26_arch_transfer_ea.yaml` v3 / `SHA256SUMS` v5）；C-b 的地基（物化 + 组分特征 + 官方地板）已交，等 C-c 开跑 |

### 条数对账（907 的构成）

- 907 = C-c 已审的 **642**（B 546 + C-a 96）+ C-b 追加的 **265**（C-b-2 225 + C-b-1 40）。
- `results/raw_runs/` 文件数 908 = 907 条记录 + `_schema.json`（**不是记录**，统计时排除）。
- **0 退单**，所以 `results/aggregate/audit_backlog.csv` **只有表头**（"主表定稿前必须清零"这条现在是清着的）。
  退单会留痕：每退一条就多一行（`run_id,问题,退回给谁,退回时间,状态`）。

### 本机测试基线（2026-11-15，`pytest tests`）

- **250 passed / 88 skipped / 10 failed**（共 348 条）。10 个失败**全部**是"缺数据"的既有用例：
  `test_harmonise.py` 1 条、`test_task_assembly.py` 8 条（缺 `data/interim/harmonised/*.parquet`）、
  `test_splits.py` 1 条（缺 `data/raw/polybench26/Datasets/scaling_indices/`）。
  两处都是 **D-38 明确不入库**的东西，在这台"数据包机"上跑不出来，与 C-b 的改动无关。
- 套件本体 `pytest tests/test_run_records.py`：**14 passed**（在 907 条真实记录上）。
- C-b 新增的测试：`tests/test_openpoly_classic.py`（14）+ `tests/test_scaling_scan.py`（9）= **23 条**，全绿。

### C-b 给套件补的一处能力缺口（不放松任何断言）

`_resolve_fold()` 原本只走 `loaders.load_split()`，而**双数据集任务**（PB26 的 scaling）在那条路上是
**故意报错**的（D-54：`split_ref` 是目录，一任务一划分不成立）。C-b 加了一条寻址分支：记录里有
`dataset_id` 就走 `experiments.scaling.load_dataset_split()`。**断言一条都没放松**：折名仍要在 spec 里
找得到、`split_version` 仍要指向存在且哈希前缀对得上的文件。**交 C-c 会签。**

---

## 入口侧结论（D 维护）

（占位：D 的划分与泄漏审计章节。）

## 出口侧结论（D 维护）

（占位：D 的出口审计章节，属自查。）
