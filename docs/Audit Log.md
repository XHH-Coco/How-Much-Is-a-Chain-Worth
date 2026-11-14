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

### 条数对账（642 的构成）

- 642 = B 546（A3 48 + A4 48 + OpenPoly 450）+ C-a 96（A1 48 + A2 48，`main_table.yaml` v2 口径）。
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
- C-b 的 scaling / openpoly CPU 批、C-c 自己的 arch 30 条，逐批过套件；
- 11-19 定稿：补齐上表、写最终结论（发布前必须 0 退单、0 未解释缺口）。

---

## 入口侧结论（D 维护）

（占位：D 的划分与泄漏审计章节。）

## 出口侧结论（D 维护）

（占位：D 的出口审计章节，属自查。）
