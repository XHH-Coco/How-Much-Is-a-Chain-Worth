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

### 条数对账（**962**，2026-11-17 23:40 追平；上一版 922）

- **962 = 922 + 40（D-69 的 scaling 链臂 A3）**。922 的构成仍是
  **546（B）+ 96（C-a）+ 225（C-b OpenPoly）+ 40（C-b scaling 的 XGBoost）+ 15（C-c arch）**；
  新增的 40 条只加在 `pb26_scaling_cp` 这个机制实验上，主表 8 任务一条未动。
  文件数 963 = 962 条 + `_schema.json`（**不是记录**，统计时排除）。核法只有一条：
  `ls results/raw_runs/*.json | grep -v _schema | wc -l`。
- 本文件以前同时出现 **657**（= 546+96+15，漏了 C-b 的 265）与 **907**（= 642+265，漏了 C-c 的 15），
  两个数各自"算得通"，所以谁也没发现。**现已合并为 922 一个数**，旧数保留在本行的括号里只为追溯。
- 另有 40 条 **RF** 拟合只在 `results/aggregate/scaling_cpu.csv` 里（D-53：森林只进表、不写 run 记录），
  拿 80 去对账会以为少了 40 条。
- A5：**0 条**，按 D-48 是 `not_constructible_data_unavailable`（D-47 证明三条链输入在数据里不存在），
  **不是** `zero_gain`、也不标 `blocked_no_gpu`——主表 32 格没有 A5 行（已删，见 Main Table Notes §2）。
- 对账口径：记录数 = 546 + C 组各批 `expected_*` 之和；`results/raw_runs/_schema.json` **不是记录**，统计时排除。
- 不在审计范围内：`results/aggregate/a4_finetune/`（B-4 微调对照 48 条，与冻结臂同 run_id，按 D-34 落独立目录、不进 `raw_runs/`，由 B 自管）。

### 测试基线说明

- 任务书基线：**280 passed / 7 skipped**（数据齐全的机器，`pytest tests`）。
- **2026-11-18 07:35 最新基线：498 passed / 7 skipped**（core 环境，209.73s，退出码 0）。
  相对 03:00 的 488，+10 条是答辩补算的守门（`tests/test_defense_checks.py`：ρ 与已发布列逐位一致、
  精确置换 p 的两端、段斜率按 dex、窗内不过零返回 `None`、log 空间插值、翻号里近零格的区分、
  地板取两条件较大者、不稳定行不进稳定格、汇总带 `sources`）。
  同轮 `provenance_check.csv` 由 4,896 行刷新到 **4,897 行 / 4,832 ok / 65 exempt / 0 fail**
  （新增两张 `defense_*` 派生表已登记在 `DERIVED_TABLES`，否则 `aggregate_table_registered` 会红——
  D-67 ③「未登记即红」在本地抓到的正是这一类）。
- **2026-11-18 03:00 基线：488 passed / 7 skipped**（core 环境，193.12s，退出码 0）。
  相对 02:25 的 487，+1 条是报告勘误的守门
  （`test_only_properties_that_change_sign_get_an_n_star`：3 个性质里只有 2 个有 n\*、Tm 必须返回 `None`）。
  7 skipped 仍是那几条要 GPU / 大模型 / `CHAINWORTH_SLOW=1` 的。
- **2026-11-18 02:25 基线：487 passed / 7 skipped**（core 环境，182.98s，退出码 0）。
  相对 00:12 的 464，+23 条全是本轮两次探针的守门：
  `test_metric_sensitivity.py` +11（H2′ 的两把尺子：平坦曲线比 1、四分之一判 halved、
  恰好 0.5 不算减半、首档为负写 `undefined`、档位按行数排不按字符串、折均值先算再比、
  幂律的 `decay_per_dex`、nan 不许污染整格、缺件要抛、真仓库 0.650 vs 0.167、区间包住点估计）、
  `test_n_matched.py` +12（Λ 定义与不裁剪、同种子配对、缺链臂要丢、层号从任务名推、
  Δ 的符号、没有 tier2 的档要留行、区间、缺表要抛、真表 4 任务 × 2 档、匹配后 Δ 仍为负）。
- **2026-11-18 00:12 基线：464 passed / 7 skipped**（core 环境，184.16s，退出码 0）。
  相对 D-69 落地后的 440，+24 条全是本轮新增的守门：
  `test_kappa.py` +6（不报斜率的两个门槛、真仓库两条、杠杆三条）、
  `test_ordering.py` +11（秩/ρ/τ-b/倒序/判据/真仓库形状/混淆/单一来源）、
  `test_aggregate_contract.py` +7（重画逐值一致、两次打乱实验、n/a 不许是 0、
  部分折要报可用行数、两份 R² 实现逐位对拍、常数标签 nan）。
  7 skipped 仍是那几条要 GPU / 大模型 / `CHAINWORTH_SLOW=1` 的。
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

- 907 是**中间态**：642（B 546 + C-a 96）+ C-b 追加 265（C-b-2 225 + C-b-1 40），写这一行时 C-c 的
  arch 批还没落盘。加上那 15 条才是总数 **922**（见上一节）。
- **0 退单**，所以 `results/aggregate/audit_backlog.csv` **只有表头**（"主表定稿前必须清零"这条现在是清着的）。
  退单会留痕：每退一条就多一行（`run_id,问题,退回给谁,退回时间,状态`）。

### 测试基线（2026-11-17 起：两种机器分开写）

以前这里只有一行"250 passed / 88 skipped / 10 failed"，而那是 **`cb310` 纯代码检出机**的数。
本机（有 `data/interim/`、装了 torch/chemprop、物化了 scaling pack）跑出来是另一组数——
一个数按两种机器读，就会把"环境缺件"当成"记录有问题"。

| 机器 | `pytest tests` | `test_run_records.py` | 说明 |
|:--|:--|:--|:--|
| 本机（2026-11-17 22:35，D-68 后） | **416 passed / 7 skipped / 0 failed**（423） | **15 passed / 0 skipped / 0 failed** | 比 365 多的 51 条全是 D 的：`test_provenance.py` **13**、`test_entry_audit.py` **7**、`test_metrics.py` **10**、`test_lambda.py` **13**、`test_bootstrap_repro.py` **8**。skip 仍是同样 7 条环境缺件 |
| 本机（2026-11-17 21:45，D-67 后） | **384 passed / 7 skipped / 0 failed**（391） | **15 passed / 0 skipped / 0 failed** | 比 365 多的 19 条是 D 的审计：`test_provenance.py` 12 + `test_entry_audit.py` 7 |
| 本机（2026-11-17 21:00，D-66 后） | **365 passed / 7 skipped / 0 failed**（372） | **15 passed / 0 skipped / 0 failed** | 比 359 多的 6 条全是 D-66 新加的 ln 尺度守门（`test_arms_classical.py` 24 → **30**）；skip 仍是同样 7 条环境缺件，一条没多 |
| 本机（2026-11-17 晚，arch 物化后） | **359 passed / 7 skipped / 0 failed**（366） | **15 passed / 0 skipped / 0 failed** | 物化后 5 条与"arch 未物化"相关的 skip 全部转 pass；剩 7 条全是环境：`test_arms_graph` 3 + `test_task_assembly` 1（chemprop）、`test_arms_polybert` 1（transformers）、`test_polybert` 1（torch）、`test_features` 1（`CHAINWORTH_SLOW=1`） |
| 本机（2026-11-17 下午，物化前） | 354 passed / 12 skipped / 0 failed（366） | 12 passed / 3 skipped / 0 failed | 12 条 skip 里 3 条是"arch 划分未物化"的分诊（D-62），6 条是 core 环境缺 torch/chemprop，2 条是 arch 物化，1 条是 `CHAINWORTH_SLOW` 未开——**没有一条是失败** |
| `cb310` 纯代码检出 | 250 passed / 88 skipped / 10 failed（348） | 14 passed（在 907 条时） | 88 条 skip 是没装 torch/chemprop；10 条红是缺 `data/interim/` 与 `scaling_indices/` |

- 计数口径：`pytest tests -q` 的 `passed + skipped + failed`。**用例数不等于记录数**，两者不要混。
- `cb310` 那 10 条红的归因只对那台机器成立。本机 `data/interim/` 有 16 个文件、`scaling_indices/`
  有 `MD_5000` 与 `PolyMetriX`，所以那 10 条在这里全绿——不要再把"红灯=缺数据"当成通用解释。
- 逐文件用例数（2026-11-17 22:35 实测）：`test_openpoly_classic.py` **11**、`test_scaling_scan.py` **13**、
  `test_arch_transfer.py` **13**（13 passed + 0 skipped，物化后两条 skipif 转 pass）、
  `test_run_records.py` **15**（15 + 0）、`test_arms_classical.py` **30**（24 + D-66 的 6 条 ln 尺度守门）、
  `test_provenance.py` **13**（D-67）、`test_entry_audit.py` **7**（D-67）、
  `test_metrics.py` **10** + `test_lambda.py` **13** + `test_bootstrap_repro.py` **8**（D-68）。
  旧文档里"openpoly 14 / scaling 9"是把两个数字调了个位置。
- **物化状态的判据曾经写错**：`test_materialisation_present_means_green` 用 `glob("fold_*.npz")` 查
  `splits/polybench26/arch/`，而实际物化落在 `arch/seed_42/fold_XX.npz`（记录 `split_version` 写的就是这个路径）。
  扁平 glob 永远不匹配，于是这条"防分诊放水"的反向测试在**任何**机器上都不跑。2026-11-17 改 `rglob` 后，
  本机 15 passed / 0 skipped 且断言全空（D-65）。与 D-62 是同一类病的两面：一个把缺件判成错，一个有件也不跑。

### C-b 给套件补的一处能力缺口（不放松任何断言）

`_resolve_fold()` 原本只走 `loaders.load_split()`，而**双数据集任务**（PB26 的 scaling）在那条路上是
**故意报错**的（D-54：`split_ref` 是目录，一任务一划分不成立）。C-b 加了一条寻址分支：记录里有
`dataset_id` 就走 `experiments.scaling.load_dataset_split()`。**断言一条都没放松**：折名仍要在 spec 里
找得到、`split_version` 仍要指向存在且哈希前缀对得上的文件。**交 C-c 会签。**

---

## 入口侧结论（D 维护）

> **审计的边界**：这一节查的是"数据怎么进来的"——官方划分有没有被绕过、全量统计量有没有泄进
> 特征、单位换算有没有证据。**性质**：这一节是**交叉复核**（A 交数据、D 查），不是自查；
> 出口侧那一节才是自查。两节的结论不许混着引用（分工 D §10）。

**首轮完成时间**：2026-11-17 21:40。判据全部落在能跑的断言上，逐条见下表；"我查过了"不算。

### 1. 逐项结论

| # | 检查项 | 检查方式（可跑的断言） | 结论 | 发现的问题 |
|:--|:--|:--|:--|:--|
| 1 | 评估不许自己切划分 | `test_splits.py::test_no_self_cut_split_in_src`（扫 `src/` 的 `train_test_split` / `KFold` / `ShuffleSplit` 调用）+ 反向 `test_self_cut_scan_catches_an_injected_call` | **过** | 无。实测注入一次真调用后**当场红**，见 §2 ① |
| 2 | 划分快照完整、深校验通过 | `test_verify_splits` / `test_verify_splits_deep` / `test_snapshot_manifest_covers_every_split_task` | **过** | 无 |
| 3 | 记录里的划分身份对得上真文件 | `06b_audit.py provenance` 的 `split_digest_matches` | **过：962/962** | 无。962 = 922 + D-69 的 40 条链臂（2026-11-17 晚追加，见下节）；引用 **61 份**划分文件（openpoly 30、polybench26 25、polybench 6），字节哈希逐条对上。链臂那 10 份折文件**不是新增**——A2 的 40 条 scaling 记录早就指向同一批文件 |
| 4 | 跨集结构泄漏要量出来 | `test_no_leakage.py::test_openpoly_leakage_matches_recomputation` / `test_leakage_basis_is_canonical_and_strictly_larger_for_tg_seed0` | **过（口径为规范式）** | 无新问题。比字符串会把 Tg 的 28 个报成 15 个（D-28），这条已守着 |
| 5 | 特征里没有标签列 | `test_no_leakage.py::test_feature_files_carry_no_label_columns`（只读 parquet schema） | **过** | **但只看列名**，见 §3 缺口 |
| 6 | 特征里没有全量统计量 | **新增** `test_entry_audit.py::test_no_feature_column_encodes_a_label_statistic`（17 份 pack × 2,253 列） | **过** | 无。判据见 §3：常数 + 等于标签的五个统计量之一 |
| 7 | 单位决策都有证据 | **新增** `test_entry_audit.py::test_every_unit_decision_carries_evidence` | **过：43/43** | `evidence` 最短 **45** 字符、中位 126、最长 367；`rule` 与 `decided_at` 无空值 |
| 8 | 换算能逐行复算 | **新增** `test_celsius_conversions_are_exact_on_every_row` | **过：15 任务 30,733 行** | 声明 `+273.15` 的三个任务 `label − label_raw` **逐行**恰为 273.15；identity 的逐行相等 |
| 9 | 抽 20 行回查原始表 | **新增** `test_twenty_rows_match_the_raw_csv` | **过：20/20** | 抽法**按任务分层**（15 个任务各 1 行 + 5 行全表等距），否则 8,055 行的任务独占名额、168 行的任务一行都抽不到。实测踩到两处，见 §3 |
| 10 | 代码里不许全量 fit | `test_no_leakage.py::test_src_fits_nothing_on_full_data` + 反向 `test_scanner_catches_a_full_data_fit` / `test_partial_fit_counts_as_fit` | **过（0 处）** | 无 |
| 11 | 预测行数 = 记录的 `n_test` | `06b_audit.py provenance` 的 `prediction_rows_equal_n_test` | **过：922/922** | 无。列恒为 `row_id,y_pred`，`row_id` 无重复（922/922） |

### 2. 两次注入实验（实测输出，不是"应该会红"）

**① 往 `src/` 注入一次自切划分**（D-9 ①）：

```
$ cat > src/chainworth/analysis/_injected_bad_split.py   # train_test_split(x, y, test_size=0.3)
$ pytest tests/test_splits.py -k self_cut
FAILED tests/test_splits.py::test_no_self_cut_split_in_src
E  AssertionError: src/ 里出现了自己切数据的调用（评估必须用官方划分）：
E    src\chainworth\analysis\_injected_bad_split.py:9:train_test_split
```

**结论：红，且点名文件与行号。** 注入文件当即删除（`src/chainworth/analysis/` 现在只有 `provenance.py`）。

**② 往特征里加一列"全量标签均值"**（D-9 ②）：

- **旧闸门不红，这是真缺口。** `test_feature_files_carry_no_label_columns` 的判据是列名白名单
  `FORBIDDEN_FEATURE_COLUMNS = {label, label_raw, y, target, target_value, row_id}`，
  一列叫 `y_mean` 的常数列**不在名单里**——它的名字看起来像特征，值却是标签的统计量。
- **补上的闸门红。** 新增判据不看列名，只看两件事同时成立：① 这一列在全体样本上取值相同
  （极差 ≤ 1e-9 相对）；② 那个常数值等于标签的 `mean`/`median`/`std`/`min`/`max` 之一
  （纯相对容差 1e-9）。反向用例 `test_detector_catches_an_injected_full_data_mean` 每次 CI 都跑。

### 3. 首轮发现的四个问题（都已修，全部"不报错"型）

| # | 问题 | 为什么危险 | 处置 |
|:--|:--|:--|:--|
| 1 | 泄漏闸门只比列名 | 改个名字就绕过：`y_mean` / `target_smooth` / `label_avg` 都不在名单里，而值就是标签本身 | 新增**按值**判据（上表第 6 条）；**不放松**原断言 |
| 2 | 常数判定用 `std == 0` **会漏** | 实测 `np.full(64, 396.33)` 的 `np.nanstd` = **6.25e-13**，不是 0。用 `std == 0` 判常数，真正的常数列会被当成"有方差"，判据当场空转 | 改用**极差**（`max − min`）与相对容差；全 NaN 列显式跳过（8 个 `BCUT2D_*`） |
| 3 | 原始表头有**尾随空格** | `Tensile stress strength at break.csv` 的列名是 `TSSAB(GPa) `（带一个空格），而 `unit_decisions.property` 记的是 trim 后的名字。按字面匹配 → `KeyError`，会把"名字差一个空格"报成"这行数据不对" | 回查时按 **strip 后**匹配，并把实际列名打进失败信息 |
| 4 | 统计量为 0 时判据**没有信息量** | `pb_t2_permeability` 的标签在 **1e-13** 量级（实测最小 `3.4e-13`），均值≈0；任何恒为 0 的指纹位都会"等于均值"。首版把这 15 列全判红了 | 纯相对容差 + **统计量为 0 时跳过**；这不是放宽，而是承认这条判据在零均值标签上没有分辨力 |

### 4. 明确没解决的缺口（不假装已经解决）

1. **强相关但非常数的泄漏抓不住。** 一列若与标签相关系数 0.9，可能合法（`MolWt` 与 `Tg` 本来
   就相关），也可能是标签的线性变换。任何阈值都会变成"凭手感退单"，所以**不设**。
   真要查只能人工看图；本节点只覆盖"整列换成常数"这种最省事、也最容易被顺手写进去的形态。
2. **`data/processed/descriptors/` 不入库**，所以判据落在 **pack** 上而不是描述符目录。
   往描述符目录注入的列必然流进 pack（pack 由那批描述符拼成），所以覆盖了两个注入点；
   但反过来"只在 pack 上查"意味着**pack 之外**的中间产物不受这条保护。
3. **`unit_decisions.csv` 只查"有没有证据"，不查"证据对不对"。** `evidence` 里的物理区间论断
   （如"若按 K 读则中位 138 K 低于任何聚合物的 Tg"）是无监督的判断题，机器判不了；能自动核的
   只有"换算在 30,733 行上逐行成立"与"20 行能从原始表找回"。这两条都做了，剩下的是人的事。

### 5. 修复提交号

见本轮提交 `fix(D8-D9): 四向对账 + 入口审计首轮`（`docs/DECISIONS.md` D-67 同步登记）。
四个问题全部在**同一轮**发现并修掉，没有跨轮悬空。

## H2 补臂与 κ（D 维护，2026-11-17 23:40）

> **这一节回答一件事**：H2 说"数据涨上去也补不回链信息"，那么 κ = dΛ/dlog₁₀n 到底是多少。
> 决策链：D-69（链臂用 A3、独立配置、预算 6→8）、D-70（`train_one` 接口对齐）、
> D-71（重采样形状 + 主表任务边界）、D-72（测试）、D-73（口径敏感性必须随数字一起报）。

### 1. 交付的 40 条

| 项 | 值 | 核法 |
|:--|:--|:--|
| 条数 | **40** = 8 个 (数据集, 档位) × 5 折 × seed0 | `ls results/raw_runs/*scaling_cp*a3_graph*.json \| wc -l` |
| 配置身份 | 单一份 `config_hash` = `sha256:f42c2adcdd76…`（`scaling_chain.yaml`） | 40 条全同 |
| 划分身份 | 10 份（2 数据集 × 5 折），与 A2 逐格相同 | `n_subset_sha256` 逐格相等（`test_both_arms_sit_on_the_same_training_rows`） |
| 环境 | `env_gnn.json` | 记录字段 |
| 成本 | **1.629 GPU-h** / 97.7 分钟，0 条失败 | `gpu_hours.csv` 按 `device=cuda, arm=a3_graph, task=pb26_scaling_cp` 筛 |
| 对账 | **4832 ok / 56 exempt / 0 fail**（4,888 行，通过率 100.00%） | `06b_audit.py provenance`（`06d` 开头跑的那次是 4830/55——当时 `kappa.csv` 与 `kappa_cells.csv` 还没写出来） |

### 2. κ 的数

| 数据集 | κ | 95% 区间 | Λ：300 档 → 最大档 | 减半比 | `verdict` |
|:--|--:|:--|:--|--:|:--|
| `md5000_alternating_cp` | **−0.0863** | [−0.1053, −0.0672] | +0.2946 → +0.1915 | 0.650 | `kappa_negative_but_lambda_not_halved` |
| `md5000_homopolymer_cp` | **+0.6949** | [+0.1876, +1.3605] | +0.0415 → +0.4824 | 11.620 | `H2_not_refuted` |

**结论（按预注册口径）**：H2 在交替集上**没有被推翻**——κ 的区间整段 < 0，但 Λ 只降到 0.65，
没到"一半"；均聚集反向，只能说"没看到下降"。四个口径里有两个都不满足，这是最稳的一条。

**但"κ < 0"这句不能单独引**：40 格里 **23 格** `1 − R²_unit < 0.05`（交替集从 n=3000 起），
Λ 是驼峰形（0.295 → **0.503** → 0.192），去掉最小档 κ = **−0.224**，只用稳定两档 κ = **+0.398**。
逐条读数见 `docs/Kappa Table Notes.md` §4–§5。

### 3. 基线

`pytest tests` = **440 passed / 7 skipped**（182.73s，core 环境）。口径从 416 → 440（+24）：
新增 `test_kappa.py` **14**、`test_scaling_chain.py` **9**（D-69）、`test_lambda.py` **13 → 14**
（①双数据集 pack 寻址）。其余文件一条未改。skip 的 7 条与上一轮同一批（要 GPU / 大模型 / `CHAINWORTH_SLOW=1`）。
逐文件用例数：splits 41 / openpoly_scan 38 / features 35 / arms_classical 30 / table_s27 26 /
arms_polybert 26 / arms_graph 23 / harmonise 22 / task_assembly 20 / openpoly_counts 16 /
task_configs 15 / run_records 15 / polybert 15 / lambda 14 / kappa 14 / scaling_scan 13 /
provenance 13 / arch_transfer 13 / openpoly_classic 11 / no_leakage 10 / metrics 10 /
scaling_chain 9 / bootstrap_repro 8 / entry_audit 7 / data_package 3。

### 4. 没解决的

1. **κ 的符号活在放大区里。** 只用 `R²_unit ≤ 0.95` 的档位，斜率是 +0.398；负号来自分母 < 0.05
   的格子。要把它彻底说清，需要 A2 那一侧在这个任务上**更好**（R²_unit 不到 0.96），
   或者换一条在低 n 端更稳的链臂——都不是本轮的范围。
2. **只有 seed 0。** A2 那一批也只有 seed 0，所以两条臂是对称的，但 GNN 在 n=300 的初始化方差
   完全没进区间（行 bootstrap 看不到它）。要收窄这一点，得补 seed 1/2（约 +1.6 GPU-h）。
3. **均聚集只有 3 档、1.35 个数量级**，且 5 折里有一折斜率 2.07；它的 κ 只能当"量级/方向"读。

## 出口侧结论（D 维护，2026-11-18 00:20）

> **先声明独立性**：这一节是**自查**。出图的人和审图的人是同一个，独立性天然不足（分工 D-10 的
> 原话）。补偿只有两条：① 测试要能抓错——打乱数据后图必须变，这条测试存在且是本节的核心证据；
> ② 仓库所有者在写汇报时会拿图对照表，那是事实上的第二双眼睛。**本节不写成"独立审计"。**

### 1. 三项必查

| # | 检查项 | 方式 | 结论 | 发现 |
|:--|:--|:--|:--|:--|
| 1 | 指标是否只有一份实现 | `grep` 全仓找 R² 公式 + 对拍测试 | **有两份，已逐位对齐** | `analysis/ci.py:_r2_rows`（向量化，bootstrap 用）与 `metrics/regression.py:r2`（单次）。两份实现本身不是错——2,000 次重采样必须向量化——但必须逐位一致。新增 `test_the_two_r2_implementations_agree_bit_for_bit`（3 组规模 × `abs=1e-12`）与常数标签的 `nan` 对拍。**这条测试是本节唯一能自动跑的"单一实现"代理** |
| 2 | 汇总是否混了划分版本 | 逐行核 `unit_run` 与 `chain_run` 的 `split_version` | **321 行全部一致，0 例混用** | 同一 (task, arm) 出现 5 个 `split_version` 是**正常的**：那是 5 折各自的 npz 文件，不是混用。判据必须是"**同一折内**两臂同源"，不是"同一任务只有一个版本" |
| 2b | 是否混了来源 | 读 6 个 PolyBench 主表任务的 `source_kind` | **只有 1 种（`experimental`）** | 提案要求的跨来源 τ 与 leave-one-source-out **不可评**，`h1_heterogeneity.csv` 写 `not_evaluable_single_source`。把"做不了"写成"通过"是审计里最常见的假绿 |
| 3 | 图与表是否同源 | 五张图的 `build()` 全部只读 `results/aggregate/`；入口 `07_figures.py` 先跑对账闸门 | **同源，闸门 4832 ok / 59 exempt / 0 fail 才出图** | 图不重算任何数：`ρ` 读 `h1_ordering.csv` 的汇总行、κ 读 `kappa.csv`、Λ 读 `lambda_ci.csv`、ΔR² 读 `openai…` 两张扫描表 |

### 2. 打乱实验（实测，不是"应该会红"）

| 注入什么 | 哪条测试 | 结果 | 说明 |
|:--|:--|:--|:--|
| 把 `pb_t1_tg` 的 A3 三行 λ 改成 +0.999 | `test_shuffling_one_task_moves_the_heatmap` | **图 1 里出现 +0.999** | 若图是硬编码的，这条会红。**注入的是数据，不是代码**——这样测的正是"图读不读表" |
| 把 `h1_ordering.csv` 的逐点 λ 与汇总 ρ 一起改成 +0.777 | `test_shuffling_the_lambda_table_moves_the_tier_panel` | **图 2 里出现 +0.777** | 第一次只改逐点行时**这条红了**：图 2 的 ρ 读的是汇总行。这不是图的问题，是"改一半的表自相矛盾"。修正方式是把两处一起改（模拟一次真正的重算），并把这个约束写进测试 docstring |
| 同一份表连画两次 | `test_redrawing_is_value_identical` | 图 1 / 2 / 3 的数值文本逐个相同 | 守的是"图里没有随机量"。第一版图 2 用随机抖动错开两臂，每次重画位置都变——那种图没法当证据 |
| 缺 A4 的 OpenPoly 三格 | `test_missing_cells_read_na_never_zero` | `n/a` ≥ 3 次，且全文无 `+0.000` | D-6 验收标准④ |

**最有价值的一条**：`test_a_cell_averaging_partial_folds_says_how_many`。图 1 的第一版把
`pb26_homopolymer_density` 画成 `+nan`——因为该任务 15 行里只有 6 行可算（另 9 行 A2 的 R² ≥ 0.95，
按 D-1 不输出 Λ），而 `np.mean` 让一个 nan 污染了整格。修完之后补了这条测试。
**诚实地说：这个 bug 是我看图看出来的，不是测试抓的**——测试是事后补的，它守住的是同类回归。

### 3. 没解决的缺口

1. **H3 无法判定**：`a5_chain` 在 `raw_runs/` 里 **0 条**，`scramble_control.csv` **从未产出**。
   图 4 把这两根画成灰色空位，caption 写明原因。这是**数据缺口**，不是审计能补的。
2. **`+0.000` 那条断言是"这批数据恰好没有"**：若将来某个任务的 Λ 精确为 0，这条会假红。
   它守的是"没有格子被写成 0"，不是"Λ 不可能为 0"。
3. **图 5 的 n\* 只有 3 个值**，且三个性质的观测窗宽度不同（Tg 到 365、Tm 到 134、介电常数到 255），
   **n\* 之间不可直接比较**。图上已标注，但这条限制只能靠 caption 传达，测试守不住。
4. **五张图的 PDF 没有像素级回归测试**：`test_redrawing_is_value_identical` 比的是图里的文本，
   不是渲染结果。字体换一台机器就可能不同（D-6 允许"字体渲染差异"），所以像素比对反而会假红。
   代价是：布局层面的回归（比如标签又压到色标上）测试抓不到——第一版图 1 就是这样，靠人眼看出来。


---

## 出口侧追加：两处探针与一处地板扩样（D 维护，2026-11-18 02:20）

> 同样是**自查**。三件事都不动已交付的 962 条记录，只动分析层与派生表。

### 1. 自检：新表能不能锚在旧交付上

| 新产物 | 自检方式 | 结果 |
|:--|:--|:--|
| `null_calibration.csv`（18 → 48 行） | 旧 18 行按 (task, condition, fold, seed) 逐行对比 | **r2 最大差 0.0**（逐位复现） |
| 同上，8 个任务的 `baseline` | 与 `raw_runs/*__a2_conventional__*` 的 seed-0 记录比 | 差 ≤ **4e-11**（CSV 十位小数所致） |
| `n_matched.csv` 的 tier2 全量跑 | `--no-subset` 应等价于原主表 run | permeability R² = −0.0018、modulus = 0.0247，与既有 A2 记录一致 |

这三条是"扩样而不是换协议"的全部依据：**同一批折、同一批测试行、同一组超参，只多跑了任务**。

### 2. 本轮抓到的三个真问题（都不报错）

1. **`decay_per_dex` 混用了自然对数与常用对数**（`metric_sensitivity.py` 初版）。合成测试种了一条
   `y ∝ n^(−0.5)` 的曲线，期望 −0.5，实得 **−1.151**（= −2.3026/2）。若只用真数据，这个数"看着像个数"，
   永远不会有人发现它错了 2.3 倍。这正是 D-71 那条纪律的用处：**先写不依赖真数据的测试**。
2. **n 匹配的档位取错了坐标系**。初版用 711 / 1034 当档位——那是**总行数**（含测试集），
   而 tier2 的训练行数是 **497 / 723**。照初版跑，等于"对齐了测试集、对错了训练集"，
   而 `n_level` 机制不会报错，只会安静地在错误的 n 上训练。已删掉那 24 行重跑（D-78）。
3. **两条测试把地板表的行数与单元数写死了 18 / 6**。地板扩到 8 个任务后它们当场红——
   这次是好事（假红暴露了出处重复）。已改成从表里派生：单元数 = 逐任务折数之和、
   行数 = 3 × 单元数、三个条件的 (task, fold) 集合必须相同。

### 3. 两处探针的结论与边界

| 探针 | 结论 | 边界（不能说过头的地方） |
|:--|:--|:--|
| n 匹配（D-78） | 同 n 下 Δ = Λ(t1) − Λ(t2) = **−0.033 / −0.087**，区间排除 0 → 层间差不是样本量造成的 | 只覆盖 t1 vs t2；t3 只有 228/279 行，**无法往上匹配**；tier2 的 A3 建图丢行（723→651、497→447），A4 按设计拒绝 n 档 |
| H2′（D-77） | ΔR² 首末比 **0.167 [0.138, 0.198]**，与 Λ 的 **0.650 [0.570, 0.739]** 不重叠 | 未预注册，**只并列报、不进结论**；换指标等于换问题，"减半"的含义变了 |

### 4. 仍然没解决的

1. **tier3 的混淆没法解**：往上匹配要造数据，往下匹配会把 279 行再砍到几十行，A3 直接崩。
   这一条只能写在局限里，等新数据。
2. **H2′ 需要重新预注册才能进结论**。现在的写法是"两把尺子并排"，读者要自己判断；
   若把 ΔR² 写进提案，"减半"的定义与 §4.2 的口径都要重写。
3. **`n_matched.csv` 没有逐 run 的预测落盘**（只有 R²）。要复核只能重跑（20–35 s/条），
   这对"派生表可复算"是可接受的，但不如 `raw_runs/` 那样能直接对账。

---

## 出口侧勘误：报告把 Tm 的 n\* 写错了（D 维护，2026-11-18 03:10）

> 起因：做汇报材料时逐页核数字，发现报告 §5.5 给 Tm 报了 `n* ≈ 50`，而图 5 的实现与它的说明都写「全窗为负、停止外推」。

### 1. 事实

| 性质 | 窗内各档 ΔR²（30 / 60 / 120 / 上界） | n\* | 依据 |
|:--|:--|:--|:--|
| Tg | +0.344 / −0.037 / −0.093 / −0.065 | **≈ 56** | `_crossover()` 报第一个过零点（插值） |
| Tm | −0.095 / −0.357 / −0.379 / −0.394 | **不报** | 全窗同号，按 D-5 验收标准②只报窗上界、不外推 |
| 介电常数 | +0.435 / −0.007 / −0.020 / +0.003 | **≈ 59** | 同上 |

复算方式：`openpoly_cpu.csv` 与 `openpoly_gpu_index.csv` 按 `(property, fold, seed, n_level)` 配对，得 225 个差，
逐档取均值后用 `fig5_crossover._crossover()` 求过零点 —— 与报告写的 56 / 59 一致，**只有 Tm 的 50 没有出处**。

### 2. 改了什么

- `docs/研究报告.md` §5.5：表格重写为「窗内各档 ΔR² + n\*」，标题从"三个性质的过零点都在 n ≈ 50–60"
  改成"只有两个性质过零，第三个全窗为负"；摘要第 5 条与 §7 局限第 7 条同步。
- `README.md` 脚本表 07 行：`n* ≈ 56 / 50 / 59` → `n* ≈ 56 / 不报 / 59`。
- **新增断言** `tests/test_aggregate_contract.py::test_only_properties_that_change_sign_get_an_n_star`：
  钉住"3 个性质里只有 2 个有 n\*、Tm 必须返回 None、两个值四舍五入是 56 与 59"。

### 3. 为什么这条值得记

图的代码本来是对的，错的是**正文**：`_crossover()` 全窗同号时返回 `None`，所以它从来不会给 Tm 一个数。
这类错误不会被任何测试抓到（没有人测"文档里的数"），只能在**逐页核对引用**时暴露。
它的教训与 D-71 同型但更窄：**合成测试守得住代码，守不住转述**；转述的守门方式是"每个数写回它的表名与列名"，
这次就是靠 `results/figures/fig5_crossover.caption.md` 里那句"窗上界 134 行处仍为负"对上的。
