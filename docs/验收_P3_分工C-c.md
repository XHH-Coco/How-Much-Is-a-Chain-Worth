# 分工 C-c · 整体验收记录（C-c-1 → C-c-4，2026-11-17）

> 验收对象：`docs/交付_C-c_架构迁移与主表审计.md`、两个提交（`67cd836` 主表+审计 642 条、`72b2208` arch 15 runs + OOD + 657 条），以及它引用的 **15 条记录 / 15 份预测 / 4 张汇总表 / 1 份 45 行产物清单 / 11 条守门测试**、D-56…D-58 三条决定。
> 验收方式：**不采信交付文档的结论**——记录 → 预测 → 表逐格重算；主表 32 格从 192 条记录反算；配置字节哈希独立重算；干净工作副本上跑全套测试。
> 环境：`DSC6020_Project_Core`（Python 3.10.14，`xgboost 2.0.3` / `sklearn 1.4.2` / `pandas 2.2.2`），与本批记录 `env_core.json` 同口径。
> **本机状态须先声明**：这台工作副本上**没有** `generated/`、`data/processed/arch/`、`splits/polybench26/arch/seed_42/*.npz`，也没有清单里那 45 份行级产物。下面凡是"核不了"的地方都是这个原因，不是数据错。

---

## 1. 一句话结论

**有条件通过。** 数字我全核了：15 条记录的指标、4 张汇总表、主表 32 格、15 行记账，**逐格重算零差**；配置哈希 **8,794 B / `fd41221650…` 独立重算吻合**；方法上三条红线（不自己写加权、不对五折做 t 检验、排序反了不重试）一条没踩，D-58 把"后半不成立"照实登记了。

**但有五条必须处理**，其中前两条会直接卡住 D：

1. **【阻断 D】45 份行级产物"留本地"落空** —— 清单 `committed:no` 说有 45 份留在本地，本机一份都没有；重建所需的输入（`generated/`、`id_features.parquet`、折 npz）也一份都没有。所以交付 §8 给 D 的两条取件路（①重建 ②取件核 sha256）**在本机都不通**。D 要算 bootstrap 区间，只能找你本人要文件。
2. **【阻断全队】干净 clone 上审计套件红 2 条** —— `pytest tests` 实测 **345 passed / 11 skipped / 3 failed**，其中 2 条红是 `test_run_records.py`，原因全是"15 条 arch 记录的 `split_version` 指向的折 npz 不存在"（D-57 决定物化不入库）。审计套件是**全队每天自查**的工具，新机器上直接红 = 大家学会忽略它。
3. **【守门虚设】4 条 skip 里有 2 条与物化无关** —— `test_arch_transfer.py` 实测 **7 passed / 4 skipped**（文档写"11 passed"）。被 skip 的 4 条中，`test_records_share_the_frozen_config_hash` 与 `test_records_carry_id_tier_metrics` **只读 `raw_runs/*.json`，根本不碰物化**，却被物化的 `skipif` 一起带走了。后果：干净 clone 上"**15 条记录的 config_hash 是否等于冻结配置字节哈希**"这条最核心的守门**不跑**——有人改了 `arch_transfer.yaml` 一个空格，测试不会红。
4. **【文档自相矛盾】** `Audit Log.md` 里 C-c 段说 657 条，C-b 段的表 #6 却写"arch 0 条、C-c 尚未交批"、总数 907。同一文件里两个数并存，且后者明确说这批没交。实际现在是 **922 条**。
5. **【口径漏了半句】** ID 指标只覆盖 **3,240 / 6,478 行**（官方 10-partition：test 取 5 份、val 取另 5 份），不是全量。文档写"5 折 × 648"没错，但读者会以为 6,110 个交替样本全测过。D 照现有文字写"在 6,478 行 ID 上"就会错。

交给 D 的三个结论**都站得住**（详见 §4、§7 的通俗版）：

- **六折（5 折 × 3 种子）上 ID 两档分开报**：整体 RMSE 0.1431（R² 0.9569）、交替 0.1099、均聚 0.3952（R² 0.8169）——均聚那档只有 37 点/折，本来就抖。
- **OOD 三档全部低于官方地板**：0.4777 / 0.5260 / 0.5024（seed0）对 0.5357 / 0.5813 / 0.5590，Δ −0.037 ~ −0.058。这是"臂比官方 RDKit_RF 好"，不是"OOD 容易"。
- **架构排序只成立前半**：`block > random` 成立；`alternating > homopolymer` 实测是反的（0.1099 < 0.3952）。D-58 照实登记、没换口径再试，这是正确动作。

---

## 2. 怎么核的（四层）

| 层 | 做什么 | 为什么这一层必须单独做 |
|:--|:--|:--|
| ① 计数 | `raw_runs/` 全局与逐臂数、预测份数、`arch_ood/` 文件数、主表格数与 `run_ids` 个数 | 文档里的条数是"作者说的"；D-37 的教训是**不分组数就不知道对不对** |
| ② 数值 | 15 条记录 → 均值/极值手算复现；`arch_transfer.csv` 与记录逐列对；主表 32 格从 192 条记录重算均值与样本标准差；OOD 逐折指标与汇总表逐格对 | 记录里的 `metrics` 是训练脚本写的，不清算一遍就只是"又抄了一遍" |
| ③ 字节 | `arch_transfer.yaml` 独立算 sha256；与 15 条记录的 `config_hash` 逐条比；核文件是否 CRLF | 配置哈希就是协议——差一个字节，15 条全作废且**不报错** |
| ④ 反证 | 干净工作副本跑 `pytest tests`，看**别人 clone 下来是什么样**；单跑 `test_arch_transfer.py -rs` 看 skip 理由 | 交付文档的"全绿"是在有物化的机器上；验收方必须站在"新机器"的角度看一遍 |

> **这一轮核不了的一项**（如实声明）：`arch_ood_ensemble.json` 的集成点值我**无法独立重算**，因为它依赖 `arch_ood_vectors/*.parquet`，而 45 份行级产物不在本机。我只能核到"逐折 `arch_ood/*_metrics.json` 的 15 份都在、三档 `n_samples` 是 18414/18414/36828、与 `arch_transfer.csv` 的 `ood_*_rmse` 逐格相同"。集成本身的算术（5 折按行平均再算 RMSE）读了代码，逻辑对。

---

## 3. 结果一览

| 交付物 | 文档自称 | 我实测 | 判定 |
|:--|:--|:--|:--|
| arch 记录数 | 15 | **15**（`a2_conventional` × fold_00..04 × seed0..2，网格无缺无重） | ✅ |
| `raw_runs/` 总数 | 657（另段写 907） | **922**（= B 546 + C-a 96 + C-b 265 + C-c 15） | ⚠️ 文档两处数字均过期 |
| arch 预测 | 15 份，行数 = 648 | **15/15 份齐、15/15 行数 = 648、无重复 `row_id`、无 NaN 预测** | ✅ |
| 同折 3 种子 test 行集 | 同一折同 test | **5/5 折的 3 个种子 `row_id` 集合完全相同** | ✅ |
| ID 整体 RMSE（15 run 均值） | 0.1431（0.1277–0.1587） | **0.143113**，极值 **0.127728 / 0.158652** | ✅ |
| 交替档 RMSE | 0.1099（0.0998–0.1200） | **0.109939**，极值 **0.099755 / 0.119987** | ✅ |
| 均聚档 RMSE | 0.3952（0.3007–0.4832） | **0.395185**，极值 **0.300704 / 0.483182** | ✅ |
| R²（整体/交替/均聚） | 0.9569 / 0.9734 / 0.8169 | **0.956862 / 0.973350 / 0.816932** | ✅ |
| `config_hash` | 全部 = 配置字节哈希 | **15/15 = `sha256:fd41221650…b87d8`**；文件 **8,794 B**、独立算得 `fd4122165029a7…`、**无 CRLF** | ✅ |
| 记账 | 15 行 CPU，墙钟 131.9 s | **15 行、`device=cpu`、`wall_clock_s` 合计 131.931**、`gpu_hours` 合计 0.0439（CPU 小时） | ✅ |
| OOD 逐折指标 | 15 份 metrics | **15/15 份在**，`n_samples` = 18414/18414/36828；与 `arch_transfer.csv` 的 `ood_*_rmse` **逐格相同** | ✅ |
| OOD 三档 vs 地板 | 全低于地板，Δ −0.05~−0.06 | seed0 **−0.0580 / −0.0553 / −0.0566**；seed1 **−0.0370 / −0.0353 / −0.0361**；seed2 **−0.0448 / −0.0426 / −0.0436** | ✅（最差信号 −0.035，仍全负） |
| 主表 32 格 | 8 任务 × 4 臂全填，无缺格 | **32/32 格**；`n_runs` 与 `run_ids` 个数 **32/32 一致**；192 个 `run_id` **全部有记录 + 有预测** | ✅ |
| 主表均值/样本标准差 | 从记录汇总 | **均值 96/96 列零差、样本标准差 96/96 列零差**（容差 1e-12） | ✅ |
| 排行榜两列 | 只落 `a2_conventional` | **8/8 行落在 `a2_conventional`**，其余 24 行留空 | ✅ |
| 缺格 | 无 `blocked_no_gpu` | **`missing_reason` 全空**，无 0、无均值充数 | ✅ |
| 检测样本覆盖 | "5 折 × 648" | **5 折 test 并集 = 3,240 行**（交替 3,055/6,110、均聚 185/368）——官方 10-partition 设计，非缺陷，但文档没写 | ⚠️ 口径需补一句 |
| `pytest tests`（本机干净副本） | 全绿 | **345 passed / 11 skipped / 3 failed**（359）；3 红 = 审计 2 + scaling pack 1 | ⚠️ 见 §5.2 |
| `test_arch_transfer.py` | 11 passed | **7 passed / 4 skipped**（skip 理由全是"本地未物化"） | ⚠️ 见 §5.3 |
| 决策登记 | D-56 / D-57 / D-58 | 三条表格行**齐全、编号无重复**（全文 D-39…D-58 各编号唯一） | ✅ |
| 冻结登记（README 第九节） | arch_transfer v1 / PROVENANCE / 守门测试 | **三项都登记了**，哈希与实测一致 | ✅ |

---

## 4. 逐项凭据

### 4.1 C-c-1 · 15 runs（ID 两档）

- 网格：`fold_00..04 × seed0..2`，15 个组合**不重不漏**；`n_train` 集合 = `{5182, 5183}`（fold_03/04 为 5183），与配置 `train_rows: [5182,5182,5182,5183,5183]` 一致；`n_test` 全 648。
- 两档分开：每条记录都有 `id_tier_metrics{alternating, homopolymer}` 与 `id_tier_n{611, 37}`，**从未合并**（验收①）。
- 我按记录里的 15 个 RMSE 手算均值：整体 2.14670/15 = **0.143113**、交替 1.64909/15 = **0.109939**、均聚 5.92777/15 = **0.395185** —— 与文档三个数逐位对上。
- 我按记录里的 15 个 OOD 逐折 RMSE 手算：random 7.40931/15 = **0.493954**、block 8.11882/15 = **0.541255** —— 与 D-58 表述的 0.4940 / 0.5413 一致。
- 超参：15 条记录内联的 `hyperparams` 完全相同（`colsample 0.4 / lr 0.1 / depth 5 / min_child_weight 20 / n_estimators 600 / reg_lambda 10 / subsample 1.0`），`search_cv_r2 = 0.9284251854`、`search_trials = 30`；配置里 `learner.params` 留空（D-56：超参表是数据不是配置）——**这一条我看得很重**，因为它让 15 条记录的 `config_hash` 与搜索数值解耦，将来重搜不会作废已有记录。
- 时间：`started_at` 最早 2026-11-16T01:02:33，`finished_at` 最晚 01:04:33 前后；单条 8.47–10.57 s，合计 131.931 s，全 CPU。

### 4.2 C-c-2 · OOD 三段式

代码层核到的（`src/chainworth/experiments/arch.py`）：

- **① 覆盖断言**：`run_ood()` 开头先 `assert_coverage_complete()`，`not complete` 直接 `raise`（不是 warn）。这正是分工书要求"缺一条就停"的实现。
- **② 只走官方脚本**：`synthesize_ood()` 用 `subprocess` 调 `Datasets/task4_architecture_transfer/summarize_weighted_predictions.py`，参数 `--holdout / --component-predictions / --output / --metrics` 四个，**自己不写一行加权**；非零退出即 `raise`（验收②）。
- **③ 防作弊的重拟合比对**：`component_predictions()` 用同训练行、同超参、同种子重 fit，然后 `np.array_equal(pred_id, stored["y_pred"])`，**不一致就 raise，不许往下走**。这是这套流程里最有价值的一道闸。
- **④ 集成不算 t 检验**：`ensemble()` 只输出 `rmse/mae/mse` 三个点值 + 与地板的差，**代码里没有任何 `scipy.stats`、没有 `sem`、没有 `std`**；`protocol` 字段把"折固定、非独立重复、未做 t 检验"写进了 JSON 本体（D-09）。
- **⑤ 地板未硬编码**：`floors` 从配置的 `data.floor_reproduction.ood_ensemble` 读，不是我上次 C-b 验收里看到的那种"写死一个数"。地板值 0.5357429224 / 0.5813189825 / 0.5589956343 与 `arch_floor_summary.json` 的 `ood_ensemble` 三档**逐位相同**。

### 4.3 C-c-3 · 主表 32 格

这是本轮我核得最细的一块，因为它是 D 画图 1 的唯一入口：

- 32 格 = 8 任务 × 4 臂（`a1_floor` / `a2_conventional` / `a3_graph` / `a4_polybert`），**A5 的 8 格确实删了**（不是留空、不是填 0）。
- 我从主表的 `run_ids` 把 192 条记录全读出来，**重算每格的 `mean_rmse/mean_mae/mean_r2` 与 `std_*`（样本标准差，`ddof=1`）：96 个均值零差、96 个标准差零差**，容差 1e-12。
- 每格的 `n_runs` 与 `run_ids` 用 `;` 分出来的个数 **32/32 一致**；每格每个 `run_id` 都能在 `results/raw_runs/` 与 `results/aggregate/predictions/` 找到对应文件（**验收①：每格点得回去**）。
- 排行榜两列只落在 `a2_conventional` 的 8 行（Tg 0.881944 / Tm 0.715289 / modulus 0.348486 / permeability 0.660010 / dielectric_loss 0.419500 / iwlt 0.751941 / PB26 两格 0.06406 / 44.639），分类是 4 `implementation` + 4 `protocol` —— 与 `Leaderboard Gap.md §3` 一致。
- `pb_t1_tg` 那格的 `notes` 写"A2−官方=+0.004184"：0.8861276 − 0.881944 = **0.0041836** ✅。
- **A4 的 `std_*` = 0 是设计**（ridge 闭式解、不掷骰子），`a1_floor` 的 `std_*` = 0 同理（Dummy）。这两列读的时候不能当"特别稳"。

### 4.4 C-c-4 · 审计套件

- 套件本体 `tests/test_run_records.py` = 14 条（第 0 条 schema + 六条断言 + 四处注入 + 2 条守护），`_resolve_fold()` 的 D-57 分支我看过了：`split_ref` 是目录时按记录自己的 `split_version` npz 解析，**哈希前缀仍要在前段逐条核对**，没有放宽（这一点与交付 §6 的描述一致）。
- 台账口径对：`657 = 546 + 96 + 15`，A5 = 0 条且注明"不可构造"而不是"零增益"。
- **但台账在干净副本上复现不出来**：本机 2 条红，退单 15 条（全部是"折 npz 不存在"）。见 §5.2。

---

## 5. 必须处理的五条

### 5.1 【阻断 D】45 份行级产物"留本地"落空，重建输入也不在

> **2026-11-17 已闭环（D-65）**：本机从上游重跑整条链路，三项身份值（`canonical_sha256` /
> `features_sha256` / `split_version`）逐位复现，15 条记录指标 Δ=0、135 个 OOD 值 Δ=0、
> 45 份产物 sha256 **45/45 与清单一致**，`arch_ood_audit.json` 首次生成（`refit_all_match: true`）。
> 45 份的 `where` 已登记为本机，`arch --verify-ood` 报"到位 45、缺 0、哈希不符 0、`where` 未登记 0"。
> 完整过程见 `docs/重跑_arch_异机复现_2026-11-17.md`。下面这段原始记录保留，作为问题为什么严重的凭据。

**事实**（逐条实测）：

| 清单里写的路径 | 清单说 | 本机实际 |
|:--|:--|:--|
| `results/aggregate/arch_component_preds/*.csv` | 15 份、`committed:no`、留本地 | **目录不存在，0 份** |
| `results/aggregate/arch_ood/*.csv` | 15 份、留本地 | **0 份**（只有 15 份 `_metrics.json`，那是入库的） |
| `results/aggregate/arch_ood_vectors/*.parquet` | 15 份、留本地 | **目录不存在，0 份** |

重建所需的输入，本机同样一份没有：

| 输入 | 谁造的 | 本机 |
|:--|:--|:--|
| `generated/shared_splits/` | C-b 物化（`external_task4_sources.csv` 取件） | **不存在** |
| `data/processed/arch/id_features.parquet` | C-c（6,478 × 2,253） | **目录不存在** |
| `data/processed/arch/component_features.parquet` | C-b（683 × 2,253） | **不存在** |
| `splits/polybench26/arch/seed_42/fold_0X.npz` | C-c `arch --setup` | **只剩 `PROVENANCE.json` + `README.md`** |

**后果**：交付 §8 给 D 的两条路——

- ① "本机跑 `arch --ood` 重建"：**要在有 `generated/` 的机器上跑**，本机不行；而 `arch --setup` 本身还要先取件。
- ② "向 C-c 取 45 份、逐份核清单 sha256"：**45 份不在任何入库路径上**，"本地"是哪台机器、什么时候会被清掉，全仓库没有一处记着。

D 要算 bootstrap 区间，现在只能私下找你。这不是数据错，是**交件断了**。

**改法**（三选一，按成本从低到高）：

1. 至少把"这 45 份现在在哪台机器、什么时候可以删"写进 `data/manifests/arch_ood_artifacts_manifest.csv` 的 `note`（或 README 第九节 arch 那三行），让下一个人知道该找谁；
2. 给 `04_run_cpu.py arch` 加一个 `--verify-ood`：读清单、逐份算 sha256、报 45/45 或列出缺哪几份。这样"取件后能一键核"这件事才成立；
3. 把 45 份打成一个受控归档（`tar.zst` + 清单已在仓库里），交 D 或放到约定的外部位置。**不建议入库**——GTRC 派生数据，D-08 的理由没变（400 MB 级别，远超 20 MiB 的入库闸门）。

我倾向先做 1 + 2（半小时内能完成），3 由你与 D 之间直接交接。

### 5.2 【阻断全队】干净副本上审计套件红 2 条

**实测**（本机 = 干净工作副本，`git status` 只有两份验收文档）：

```
345 passed, 11 skipped, 3 failed
FAILED tests/test_run_records.py::test_3_prediction_rows_match_split_test_length
FAILED tests/test_run_records.py::test_composed_audit_is_clean_on_real_records
FAILED tests/test_scaling_scan.py::test_packs_are_built_from_the_official_canonical_bytes
```

前两条的失败信息**逐字相同**：15 条 arch 记录的 `split_version` 指向 `splits/polybench26/arch/seed_42/fold_0X.npz`，文件不存在。第三条是 C-b 的账（两份 scaling pack 未入库，我上一轮已提）。

**根因是制度，不是测试写得不好**：D-57 把任务配置保持在 `deferred` + 目录，物化产物不入库——这是对的（它挡住了 A-5 的 `test_splits.py` 变红）。但审计套件 `_resolve_fold()` 把"折文件不存在"直接判成**退单**（`problems.append`），于是：

- 任何 clone 这份仓库的人，跑 `pytest tests` 都会看到 2 条红；
- 而这 2 条红**每次都是同样的 15 条、同样的原因**，与"这批记录有没有问题"无关；
- 分工书原话是"套件由你写，C-b 在各批收尾时跑"——如果工具从一开始就是红的，**退单信号会被噪声淹没**。

**改法（不动断言，只分诊）**：

```python
# _resolve_fold 里，文件不存在时区分两种情形
if not split_path.exists():
    if str(record.get("task", "")) in DEFERRED_MATERIALISED_TASKS:   # 只有 arch 这一类
        pytest.skip(f"本地未物化：{rel}（先跑 04_run_cpu.py arch --setup）")
    return None, [f"{name}: split_version 指向的 {rel} 不存在"]
```

配一条**反向测试**守住"物化存在时必须全绿"：`test_arch_folds_resolve_when_materialised` —— 在 `splits/polybench26/arch/` 有 npz 的机器上，15 条记录必须全部解析成功、行数全对。这样有物化的机器上严格度一点没降，没物化的机器上给出的是"未做"而不是"做错了"。

> 两个不要：(a) **不要**把 arch 的 15 条从审计里排除——那等于放弃审计；(b) **不要**为了绿而 `skipif` 掉整条断言（见 §5.3 的反例）。

### 5.3 【守门虚设】4 条 skip 里 2 条与物化无关

`tests/test_arch_transfer.py` 尾部四条都挂着同一句：

```python
@pytest.mark.skipif(not SPLIT_MANIFEST.exists(), reason="本地未物化：先跑 ... arch --setup")
```

但这四条实际读的是什么：

| 测试 | 读什么 | 该不该受物化影响 |
|:--|:--|:--|
| `test_materialised_manifest_row_counts` | 物化 `manifest.json` | **该**（合理） |
| `test_load_split_resolves_via_arch_loader` | 物化 npz | **该**（合理） |
| `test_records_share_the_frozen_config_hash` | `raw_runs/*.json` **只读记录** | **不该** |
| `test_records_carry_id_tier_metrics` | `raw_runs/*.json` **只读记录** | **不该** |

**后果很具体**：干净副本上，"**15 条记录的 `config_hash` 是否等于冻结配置的字节哈希**"这条守门**不跑**。谁把 `arch_transfer.yaml` 改了一个空格（比如升 v2 时忘了重跑），测试不会红——而这正是 C-c 整份交付最核心的承诺（"config_hash 就是文件字节哈希"）。同理，"每条记录都带 ID 两档"也不跑。

**这不是故意的**：同文件里 `test_train_rows_in_config_match_materialisation()`（第 143 行）**也不碰物化、也没挂 skipif**，跑绿了。说明后面两条是复制粘贴时把装饰器带过来的。

**改法**：删掉这两条的 `@pytest.mark.skipif`（**断言一个字不动**，符合"只许加、不许删"）。改完预期 `9 passed / 2 skipped`；在物化齐全的机器上是 `11 passed`。

**顺带把文档改对**：交付 §6 与 README 第九节第 474 行都写"**11 passed / 11 条断言**"，实测是 **7 + 4**。应写成"11 条断言；有物化 11 passed，干净副本 9 passed / 2 skipped（另 2 条与 §5.2 同因）"。

### 5.4 【文档自相矛盾】Audit Log 里 907 与 657 并存

`docs/Audit Log.md` 现在有两段结论：

- 上段（C-c 维护）：`2026-11-16 C-c 已交 arch 批（15 条）… 全绿`，条数对账写 **657**；
- 下段（C-b 维护）：表 #6 写 **"arch（C-c）| — | 0 | — | C-c 尚未交批"**，条数对账写 **907**。

同一份文件里，一段说交了、一段说没交；一个总数 657、一个 907。**实际现在是 922**（`ls results/raw_runs/*.json | grep -v _schema | wc -l`）。

D 发布前要引用"0 退单"这个结论，看到两个总数会先来问你一遍——这正好是分工书想避免的那类追问。

**改法**：把 C-b 段的 #6 改成"已交 15 条 / 0 退单 / 11-16"，"907 的构成"改成 922（= 907 + 15），并在两段之间加一行"本表口径以此段为准，上段只记 C-c 自己那一批"。或者干脆合并成一张表——现在两段的分工读者看不出来。

### 5.5 【口径漏半句】ID 指标只覆盖一半样本

官方 `create_task4_shared_splits.py` 的划分逻辑（我读了源码）：

```python
train["partition"] = assign_folds(train, num_folds * 2, seed)      # 10 份
for fold in range(num_folds):
    train_frame = train[~isin([fold, fold + num_folds])]            # 8 份
    val_frame   = train[partition == fold + num_folds]              # 1 份（+5）
    test_frame  = train[partition == fold]                          # 1 份
```

**10 个 partition → 5 折的 test 取 `0..4`、val 取 `5..9`**。所以：

- 5 折 test 并集 = **3,240 行**（交替 3,055 / 6,110、均聚 185 / 368），**正好一半**；
- 另 3,238 行**只当 val、从不进 test**；
- 这与"标准 5 折交叉覆盖全量"不是一回事。

配置里其实写对了（`in_distribution_test_fraction: 0.10`），交付 §2 与 README 也都写"每折 test 648"，**数字没有错**。漏的是那句话：**读者会以为 6,110 个交替样本全被评估过**。D 若写"在 6,478 行 ID 上 R² = 0.9569"，就是错的（实际 3,240 行）；写"5 折各 648 行（交替 611 + 均聚 37）"才是对的。

**改法**：在交付 §2 与 `Main Table Notes` 各加一句：

> ID 指标覆盖 3,240 / 6,478 行。官方 `create_task4_shared_splits.py` 切 10 个 partition，每折 test 取 1 份、val 取 1 份，5 折的 test 互不重叠——所以交替 3,055 / 6,110、均聚 185 / 368 进过 test，其余只做 val。跨折聚合是"5 个折级 RMSE 的均值"，不是"全量 6,478 行的 RMSE"。

---

## 6. 建议补的五条（不阻塞验收）

1. **`refit_matches_id` 从不落盘**。`component_predictions()` 算了 `identical`（重拟合 vs 落盘 ID 预测逐位相同），不一致会 `raise`——这是全流程最硬的防作弊闸。但它**只活在内存里**（`summary["refit_matches_id"]`），`grep` 全仓只出现在 `arch.py` 两行，`arch_ood_ensemble.json` 与 `arch_transfer.csv` 里都没有。**第三方无从核对"15 条全部逐位一致"这句话**，只能信你的日志。建议在 `arch_ood_ensemble.json` 加一个 `refit_matches_id: {"fold_00__seed0": true, ...}`（15 个键）。成本 5 行代码，收益是"这条结论从此可核"。
2. **集成求均值时没校验 5 个 vector 的行序**。`ensemble()` 里 `merged` 只取 `vectors[0]` 的 `holdout_row_index`，然后 `np.mean([...], axis=0)` 直接按位置平均。若某折 parquet 的行序不同（官方脚本输出顺序依赖输入顺序），**结果会串行且不报错**。加一行 `assert (v["holdout_row_index"].to_numpy() == vectors[0]["holdout_row_index"].to_numpy()).all()` 即可。这是典型的"不报错、数字看起来合理"的坑——和 D-37 同一类。
3. **OOD 那一步没进 `gpu_hours.csv`**。表里 arch 只有 15 行、合计 131.9 s（ID 训练）。而分工书说 OOD 是"**本份最重的一步，1–2 h**"（15 次重拟合 + 683 组分预测 + 45 万行官方合成）。D 报总成本时这 1–2 h 会**整个漏掉**。建议加一行聚合记录（如 `pb26_arch_transfer_ea__ood__all`，`device=cpu`，`wall_clock_s` = OOD 阶段实测），或在 `arch_ood_ensemble.json` 记 `wall_clock_s_ood`。
4. **`arch_floor_summary.json` 的 verdict 仍写"可以交给 C-c 跑 30 runs"**。D-49 已经把它改成 15 条了。这是 C-b 的产物、我上轮提过，还没改；一句话的事。
5. **主表 4 格 `protocol` 仍悬在 Q-10 上**（modulus / permeability / dielectric_loss / iwlt）。`leaderboard_diff_class = protocol` 已经把"不可减、不归因"写清楚了，这比含糊过去好得多；但 Q-10 一天不定，D 的 Λ 就一天不能算。这条不归 C-c，只是提醒它是 D 的前置。

---

## 7. 不是问题、但一定会被问的三条

1. **"交替 0.1099 < 均聚 0.3952，是不是说均聚更难预测？"** 不能这么读。均聚每折只有 **37 个** test 点，15 条的 RMSE 在 **0.3007–0.4832** 之间跳（极差 0.18，是交替档极差 0.020 的 9 倍）。这 37 点还是 10-partition 里那 5 份（共 185 / 368 个均聚样本）。**样本量决定了这一档的 RMSE 主要是噪声**，它只能支持"均聚不比交替容易"，不能支持"均聚难多少"。
2. **"三档都低于官方地板，是不是我们赢了？"** 是"臂比官方 RDKit_RF 好"，不是"OOD 容易"。地板本身在 36,828 行上，随机档 18,414、block 档 18,414——**两个子集的分母不同，不能互相加减**。D-58 里 `block > random` 成立是"同域内比大小"（合法），而 `alternating` vs `homopolymer` 是**跨域**（ID vs ID 也不算，一个是 611 点一个是 37 点）——config 里 `ordering_note` 已经写了"跨域只作陈述不作判决"，这句话在交付 §4 也抄了，**是对的**。
3. **"131.9 秒跑完 15 条，这数可信吗？"** 可信，但只覆盖 ID。6,478 × 2,253 的 XGBoost、600 棵树、CPU，单条 8–10 s 是正常的（C-a 的 A2 也是这个量级）。OOD 的 1–2 h 不在这个数里（见 §6 第 3 条）。

---

## 8. 交接与待办

**给 D**（三个结论可以直接用，但两条前置）：

| 你要的 | 现在能不能用 | 前置 |
|:--|:--|:--|
| 图 1（32 格主表） | **能**，32/32 格、每格可点回 run_id、均值零差 | 无。CI 两列等你回填 |
| H1 架构子句 | **能，但措辞要绕开 D-58 的"后半不成立"** | 读 `DECISIONS.md` D-58 + §7 第 1 条 |
| OOD 的 bootstrap 区间 | **能**（2026-11-17 起） | 45 份向量已落在本机并登记 `where`；逐份核清单 sha256 即可（D-65） |
| Λ / κ | 能算，但 4 个 `protocol` 格（Q-10）一天不定就一天算不了 | Q-10 |

**给仓库所有者**（要拍板的三件）：

1. §5.2 的 skip 分诊方案——动的是审计套件，分工书说"只许加断言"，我这条是**加一条反向测试 + 把环境问题与记录问题分开**，不删任何断言，但需要你点头；
2. §5.3 删两条 `skipif`——收益明确（核心守门能跑了），风险是"11 passed"变成"9 passed / 2 skipped"，文档要同步改；
3. `main_table.yaml` 的 `cell` 段仍写"一格 =（任务, 折, 臂）3 种子"，实际一格 =（任务, 臂），PB26 是 15 条 —— 这是我上轮 C-a 验收提的第 3 条，**还没改**，它动的是冻结口径表，得你拍。

**给 C-c 本人**（按优先级；1–4 已在 2026-11-17 完成，见 D-62/D-65）：

1. ~~§5.1 至少写清 45 份在哪（+ 加 `--verify-ood`）~~ → **已完成**：45 份落到本机、`where` 全部登记，
   `arch --verify-ood` 三分类（缺件 / 哈希不符 / `where` 未登记）分开报；
2. ~~§5.3 删两条 `skipif`，跑一遍确认 9+2~~ → **已完成**：实测 **13 passed / 0 skipped**（折物化后两条也转 pass）；
3. ~~§5.4 把 Audit Log 两个数合成一个（922）~~ → **已完成**；
4. ~~§5.5 补那句"覆盖 3,240 / 6,478"~~ → **已完成**；
5. §6 第 1、2 条各 5 行代码，把"可核"补上 → 第 1 条（`refit_matches_id` 落盘）已随 `arch_ood_audit.json`
   完成；第 2 条（集成前校验行序）已加断言。**5 条全闭环。**

---

## 附：本轮操作与残留

- 本次验收**全程只读**：没有改任何记录、预测、配置、清单；`git status` 只有新增的 `docs/验收_P3_分工C-a.md`、`docs/验收_P3_分工C-b.md`、本文件。
- 跑测试用了三个新临时目录（`.pytest_tmp_cc1`…`cc4`），均已 `--basetemp` 指到 gitignore 的 `.pytest_tmp*` 下，收尾随 shell 清理。
- 本机仍缺 `data/processed/arch/`、`generated/`、`splits/polybench26/arch/seed_42/`（C-c 的物化产物），以及 C-b 的两份 scaling pack —— 这三处是本轮 3 条红的直接原因，**都不是本轮的改动造成的**。
