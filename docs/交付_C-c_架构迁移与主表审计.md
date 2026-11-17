# 交付 C-c · 架构迁移主跑（15 runs + OOD 三段式）与主表/审计收口

> 本文件是 C-c 的交付说明：C-c-1（arch 15 runs）、C-c-2（OOD 逐组分预测 + 官方三段式合成）。
> 主表 32 格与审计套件在 2026-11-15 的收口见 `docs/Main Table Notes.md` 与 `docs/Audit Log.md`；
> 本文件只讲 **2026-11-16 新增的 arch 批**（642 → 657 条；全仓总数当时还没算进 C-b 的 265，
> 现统一为 **922**，见 Audit Log「条数对账」）。
> 数字全部由训练脚本落盘，审计只有退回权（红线 4）；本文档不手写任何 run 数字。

---

## 1. 交付了什么

| 项 | 内容 | 位置 |
|:--|:--|:--|
| C-c-1 | 15 条 run（a2_conventional 单臂 × 5 折 × 3 种子，全程 CPU）+ 15 份预测 + 墙钟 | `results/raw_runs/pb26_arch_transfer_ea__*.json`、`results/aggregate/predictions/`、`results/aggregate/gpu_hours.csv`（CPU 记录） |
| C-c-2 | OOD 三段式：覆盖断言 → 逐组分预测（683 组分 × 15 run）→ 官方 `summarize_weighted_predictions.py` 合成 | `results/aggregate/arch_component_preds/`、`arch_ood/`（csv + metrics json）；**行级产物不入库（GTRC，D-08），45 份逐份 sha256 入 `data/manifests/arch_ood_artifacts_manifest.csv`** |
| 集成 | 五折集成点值（D-09：无 t 检验、无均值 ± SE） | `results/aggregate/arch_ood_ensemble.json` |
| 汇总表 | ID 两档 + OOD 三档，一 run 一行 | `results/aggregate/arch_transfer.csv` |
| 代码 | arch 子命令 + 实验模块 | `scripts/04_run_cpu.py arch`、`src/chainworth/experiments/arch.py` |
| 配置 | 冻结实验配置 v1（15 条记录的 `config_hash` 就是它的字节哈希） | `configs/experiments/arch_transfer.yaml`（8,794 B，sha256 `fd4122165029a7…`） |
| 任务配置 | v3（D-49）保持原样 + 指纹表 v5——D-57 裁定：present+文件会弄红 A-5 的 `test_splits.py`，故不升版，划分解析由 C-c 自己的 `_load_arch_split()` 与套件 `_resolve_fold` 分支承担 | `configs/tasks/pb26_arch_transfer_ea.yaml`、`configs/tasks/SHA256SUMS` |
| 划分 | 本地物化 5 折 npz + manifest（不入库，`arch --setup` 重建） | `splits/polybench26/arch/seed_42/`（入库的只有 README.md + PROVENANCE.json） |
| 特征 | ID 特征 6,478 × 2,253（不入库，清单入库） | `data/processed/arch/id_features.parquet`、`data/manifests/arch_id_manifest.json` |
| 守门测试 | 11 条断言 | `tests/test_arch_transfer.py` |

## 2. C-c-1 结果：ID 两档（15 runs）

超参：D-51 预注册协议搜索一次（fold_00 训练行 5,182 行内 Murcko 5 折 CV，30 trials，
`search_seed=20261113`，cv_r2 = 0.9284），落 `a2_best_params.csv` 第 13 行数据记录（D-56）。
**配置里 `learner.params` 留空**——超参表是数据不是配置，15 条记录的 `config_hash` 与搜索数值解耦。

| 口径 | RMSE（15 run 均值） | 范围 | R²（均值） |
|:--|--:|--:|--:|
| ID 整体（每折 test 648 行） | **0.1431** | 0.1277 – 0.1587 | 0.9569 |
| 交替共聚（每折 611 行，单独报） | **0.1099** | 0.0998 – 0.1200 | 0.9734（交替档均值） |
| 均聚（每折 37 行，单独报） | **0.3952** | 0.3007 – 0.4832 | 0.8169（均聚档均值） |

- 两档**从不合并**成一个 "ID"（验收①）；每折 test = 交替 611 + 均聚 37，行数与官方物化逐折核过。
- **覆盖口径必须写清**：官方脚本把训练集切成 **10 个 partition**，第 `k` 折 test 取 `k`、val 取 `k+5`，
  所以 5 折 test 互不重叠、**并集 = 全量的一半**：**3,240 / 6,478 行**（交替 3,055 / 6,110、均聚 185 / 368）。
  上表的 0.1431 / 0.9569 是"每折 648 行的平均"，**不是**"在 6,478 行上"的值；报告里写后者就是错的
  （`tests/step` 的 10-partition 事实见 `create_task4_shared_splits.py` 的 `assign_folds(train, num_folds*2)`）。
- 训练行数逐折 = [5,182, 5,182, 5,182, 5,183, 5,183]、验证 [648, 648, 648, 647, 647]，
  与官方 `create_task4_shared_splits.py` 的产物一致。
- 墙钟合计 131.9 秒（15 条，每条 8.5–10.6 秒，CPU）；0 条 GPU 记录。

## 3. C-c-2 结果：OOD 三段式

① 覆盖断言：holdout 36,828 行的全部唯一组分（683）在 `component_index.csv` 有且仅一行（missing 0 / duplicated 0）。
② 逐组分预测：每个 (折, 种子) 用同一批训练行、同一份冻结超参、同一种子**重拟合**，并先与落盘的 ID 测试集
预测逐位比对（`n_jobs=1` + `tree_method=hist`，CPU 确定性），**15 条全部逐位一致**后才预测 683 个组分。
③ 合成：只走官方 `summarize_weighted_predictions.py`（`--holdout --component-predictions --output --metrics`），
自己一行加权都没写（验收②）。

五折集成点值（按聚合物平均，**不做 t 检验、不给均值 ± SE**，D-09）：

| 种子 | random RMSE | block RMSE | combined RMSE |
|--:|--:|--:|--:|
| 0 | 0.4777 | 0.5260 | 0.5024 |
| 1 | 0.4987 | 0.5460 | 0.5229 |
| 2 | 0.4910 | 0.5387 | 0.5154 |
| 官方 RDKit_RF 地板（C-b 复现） | 0.5357 | 0.5813 | 0.5590 |

- 三个种子、三档全部**低于官方地板**，Δ ≈ −0.05 ~ −0.06（a2 XGBoost 对 RDKit_RF 的差距）。
- 逐折 OOD 指标（`arch_ood/*_metrics.json`）只是过程留档，**不是判决口径**；判决只看上表的点值。
- 五折预测向量逐 run 留档 `arch_ood_vectors/<run_id>.parquet`（验收④），D 算置信区间直接可用。

## 4. 架构排序：预注册期望只成立前半（D-58，照实登记）

预注册口径（`arch_transfer.yaml` eval 段）：`block > random > alternating > homopolymer`。
实测（跨域只作陈述不作判决）：

| 档 | mean RMSE | 来源 |
|:--|--:|:--|
| block（OOD） | 0.5413 | 15 个折级 RMSE 的均值 |
| random（OOD） | 0.4940 | 同上 |
| homopolymer（ID） | 0.3952 | 15 run 的均聚档 RMSE 均值 |
| alternating（ID） | 0.1099 | 15 run 的交替档 RMSE 均值 |

- **block > random 成立**（OOD 两档内部比大小是合法口径）。
- **"alternating > homopolymer" 不成立，实测是反的**（0.1099 < 0.3952）。注意均聚测试点每折只有
  37 个，RMSE 噪声大（范围 0.30–0.48），且 OOD 与 ID 是不同分布——跨域链式比较本身只能陈述。
- 按 config 里 ordering_note 的既定条款：**不重调超参、不换口径再试**，实测数字原样交 D（D-58 待会签）。

## 5. 与 C-b 的接口处（联合核对）

- **train_rows 改正**：C-b 的 data 段原写折 3 为 5,182；2026-11-16 物化后核对 manifest，折 3 实为
  5,183（val 647）。已在冻结配置 v1 里改正并在冻结头注明，**请 C-b 会签 D-57**。
- 组分特征跨机器一致：本地重造 683 × 2,253，`features_sha256` 与 C-b 的清单逐字节相同（`fd7ed9af…`），
  其清单文件本身保持 C-b 原字节未动。
- ID 特征（C-c 造，6,478 × 2,253）列序与组分表、与 A-6/B-1 的 2,253 列逐列相同（规则 2）；
  哈希入 `data/manifests/arch_id_manifest.json`（`fed6173b82…`），本体不入库（GTRC 派生数据，D-08）。

## 6. 审计

- `pytest tests/test_run_records.py`：交付时 **14 passed**（在 657 条上）。2026-11-17 起分诊落地后是
  **12 passed / 3 skipped**（在 **922** 条上）：3 条 skip 全部是"arch 划分未物化"的显式跳过，
  不是退单；在物化机上必须 15 passed（`test_materialisation_present_means_green` 守着）。套件加了
  `_resolve_fold` 一个分支（D-57）：任务配置的 `split_ref` 是目录、记录 `split_version` 指向 npz 文件时，
  按该 npz 的 `test` 数组核对折名与测试行（字节哈希仍在前段逐条核过，无放宽）；其余断言一条未动。
- `pytest tests/test_arch_transfer.py`：交付时写的是 **11 passed**，实测是 **7 passed / 4 skipped**——
  其中两条（`test_records_share_the_frozen_config_hash`、`test_records_carry_id_tier_metrics`）只读
  `raw_runs/*.json`，与物化无关，却被 `skipif` 一起带走了。**2026-11-17 已删掉那两条装饰器**，
  现在 **11 passed / 2 skipped**（剩余 2 条确实需要物化产物的 skip；D-62）。
- Audit Log 已加 2026-11-16 一行（15 条 0 退单），条数对账 642 → 657。

## 7. 环境与许可

- 环境：`DSC6020_Project_Core`（Python 3.10.14，包版本与 `results/env_fingerprints/env_core.json`
  的 `packages` 逐项相同，与 D-40/D-55 同口径）；记录 `device: cpu`、`env_fingerprint: env_core.json`。
- 训练源 polyVERSE EA 属 **GTRC General Public Use License**（D-08）：结果表随附 GTRC 全文、
  免费分发、不得商用；派生数据（特征 parquet、物化划分）不入库，只入库哈希清单。

## 8. 留给 D 与仓库所有者

- **D 要 bootstrap 区间时的取件路径**：五折预测向量是行级 GTRC 派生产物，不入库（D-08，同 C-b 的
  组分表先例）。三条路，按省事排序：
  ① **直接取本机现成的 45 份**（2026-11-17 已落在 `local:DESKTOP-6C1JF93（E:/How-Much-Is-a-Chain-Worth）`，
  清单的 `where` 已登记），逐份核 `data/manifests/arch_ood_artifacts_manifest.csv` 的 sha256；
  ② 在任意一台 RDKit 构建相同的机器上重跑 `arch --ood` 重建——**已实测逐位复现**（见
  `docs/重跑_arch_异机复现_2026-11-17.md`：15 条记录 Δ=0、135 个 OOD 值 Δ=0、45 份产物哈希 45/45 一致）；
  ③ 向 C-c 取件，同样逐份核 sha256。
- **OOD 的两项"内存事实"已落盘**：`results/aggregate/arch_ood_audit.json` 记 `refit_all_match: true`
  （15/15 重拟合与落盘 ID 预测逐位相同）与 `wall_clock_s_total: 190.702`；`arch_ood_ensemble.json` 引用它们，
  不再是 `null`（D-60、D-65）。
- H1 架构子句的措辞需处理 D-58 的"后半不成立"：建议表述交给所有者拍板（Q-11 同款流程）。
- 主表不动：arch 是机制实验，不进 32 格（Proposal §3.1 表 C）；32 格定稿见 `docs/Main Table Notes.md`。
