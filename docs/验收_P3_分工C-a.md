# 分工 C-a · 整体验收记录（C-a-1 → C-a-5 + 三张附加表，2026-11-16）

> 验收对象：`docs/交付_C-a_经典臂族.md`、它引用的 **96 条 run 记录 / 96 份预测 / 16 份产物**，以及 D-45 / D-46 / D-48 三条决定。
> 验收方式：**不采信交付文档的结论**，从记录 → 预测 → 标签重算一遍，再做字节与表间交叉。

---

## 1. 一句话结论

**通过。** 96 条记录逐条自洽（我从预测 + 标签重算 R²/RMSE，18 位小数逐位相同），96 份预测逐份齐、三份配置的字节哈希与记录里的 `config_hash` 逐位相同，三张附加表都做过"与主表同一把尺"的自证。

**两个遗留必须处理**，否则下一批人会被误导：

1. **`04_run_cpu.py index` 重建会把 arch 的 15 条吞进 `a1_a2_index.csv`**（实测：dry-run 报 **111 条**）。现在的 96 行只是因为索引生成于 arch 开跑之前——这是一个**不报错、看起来完全正常**的静默错误（与 D-37 同型）。
2. **Q-10（重尾任务取不取对数）仍未决**，而它在 4 个任务上直接决定"我们比不比得过官方"这句话能不能说。我按 ln 尺度实测了四个数，见 §5.2——**"尺度不同"这个解释只对其中两个任务成立**。

---

## 2. 怎么核的（四层）

| 层 | 做法 | 结果 |
|:--|:--|:--|
| ① 数值可复算 | 从 `predictions/<run_id>.parquet` 的 `row_id` 取特征包里的 `y`，重算 R² / RMSE | 6 条抽样（A1/A2 × 3 任务）**ΔR² = 0.00e+00**，RMSE 逐位相同 |
| ② 字节可证 | 三份配置文件的 sha256 vs 交付文档写的哈希 vs 96 条记录里的 `config_hash` | 三者一致（§4-1） |
| ③ 表间交叉 | 主表 / 排行榜对齐表 / 敏感性表 / 空值对照共用同一批数字，逐格比 | 8 个任务的 A2 均值与 `polybench_official_compare.csv` 逐格相同 |
| ④ 文档自称 vs 磁盘 | 条数、行数、断言、测试 | 仅 2 处过期（§6），无虚报 |

复核命令（任何人可重跑）：

```bash
python -m pytest tests -q                      # 345 passed / 3 failed（3 条都不在 C-a 名下，见 §6.3）
python scripts/04_run_cpu.py index --dry-run   # 应报 96；现在报 111 ← 问题 §5.1
python scripts/04b_official_compare.py polybench-compare   # 6 任务排行榜对齐（不出网）
```

---

## 3. 结果一览

| 项 | 声称 | 实测 | 判定 |
|:--|:--|:--|:--|
| A1 记录 | 48 | 48（8 任务 × 3 种子；PB26 每任务 5 折） | ✅ |
| A2 记录 | 48 | 48，**同一 `config_hash`**（`7d29768a…`） | ✅ 一把尺 |
| 预测 | 96 | 96 份，`row_id + y_pred` 两列，行数 = 该折 test 长度 | ✅ |
| `a1_a2_index.csv` | 96 行 | 96 行（但重建会变 111，见 §5.1） | ⚠️ |
| `main_table.yaml` | 96 / 32 格 | `expected_runs.total: 96`，`tests/test_arms_classical.py` 断言已改 96 | ✅ |
| A5 | 0 条、`draft_blocked_pending_data` | 0 条，`status: draft_blocked_pending_data`、`frozen_at: null` | ✅ 与 D-48 一致 |
| BCUT2D 八列 | 不删 + 守卫测试 | `tests/test_arms_classical.py:93` 的 `BCUT2D_ALL_NAN` 名单 + 断言在 | ✅ D-48 ⑤ 已落 |
| 敏感性表 | 96 行 | 96 行（6 任务 ×1 折 ×3 种子 + 2 任务 ×5 折 ×3 种子）×2 学习器 | ✅ |
| 受控对照 | 50 行 | 50 行 = 2 任务 × 5 变体 × 5 折 × seed0；`baseline` 10 行齐 | ✅ |
| 零信息对照 | 18 行 | 18 行 = 6 单元 × 3 条件 × seed0 | ✅ |
| 重复性 | 同 seed 两遍逐位相同 | `max_abs_prediction_delta = 0.0` | ✅（覆盖面见 §6.1） |
| 墙钟 | — | A1 48 条 ≈ 0 秒（闭式均值）｜A2 48 条 **3.8 分钟** | ✅ 全程 CPU |

---

## 4. 逐项凭据

### 4.1 配置与字节（C-a-1）

| 文件 | 字节 | sha256 | 与记录 |
|:--|--:|:--|:--|
| `configs/arms/a1_floor.yaml` | 3,098 | `871836748cd89e13…b64edc` | 48 条 `config_hash` 全等 |
| `configs/arms/a2_conventional.yaml` | 6,324 | `7d29768afc09624e…4b22d7f` | 48 条全等 |
| `configs/experiments/main_table.yaml` | 5,372 | `06cd9e1efe82e1f1…db0251b` | v2（96 口径） |

**A2 的 48 条同哈希**是这次验收最想确认的一条：同哈希 = 同一套预处理、同一个学习器、同一份超参来源；主表 A2 那一列才能与敏感性表、受控对照、arch 的 15 条**分得开**（arch 用自己的 `arch_transfer.yaml`，哈希不同，D-57 已登记）。

### 4.2 数值（C-a-2）

抽样重算（官方划分、seed 0）：

| 任务 | 臂 | 记录 R² | 重算 R² | Δ |
|:--|:--|--:|--:|--:|
| `pb_t1_tg` | a1_floor | −0.0008828133 | −0.0008828133 | 0 |
| `pb_t1_tg` | a2_conventional | 0.8862028074 | 0.8862028074 | 0 |
| `pb_t2_permeability` | a2_conventional | −0.0018136130 | −0.0018136130 | 0 |
| `pb26_polymetrix_tg` | a2_conventional | 0.8968939547 | 0.8968939547 | 0 |

A1 的预测是常数（= `train_mean` 419.2639），符合"不读特征"的定义；`pb_t1_tg` 的 0.8862 与 D-48 里那次独立复现的 `0.8862028074232996` 逐位相同——**同一套复现管线，两种入口**。

### 4.3 三张附加表的"同一把尺"自证

- `null_calibration.csv` 的 `baseline` 与 A2 seed0 记录逐折相等（交付文档记最大差 `3.0e−11`，已固化成测试）；
- `pb26_controlled_control.csv` 的 `baseline` 与 A2 同任务同折 seed0 相等（10/10，≤4.5e−11）；
- `learner_sensitivity.csv` 的网格与主表**同折同种子**（96 行 = 主表网格 × 2 学习器）。

三张表都能"减"——这是它们敢叫"对照"的唯一理由。

---

## 5. 必须改的三条

### 5.1 【静默错误】`index` 重建会吞进 arch 的 15 条

```text
$ python scripts/04_run_cpu.py index --dry-run
磁盘上 A1/A2 的 ok 记录 111 条（不含 n 档记录）
```

`index` 的过滤只排除了**带 `n_level` 的记录**（C-b 的 scaling / OpenPoly，D-52），但 `pb26_arch_transfer_ea` 的 15 条 A2 记录**既没有 `n_level`、arm 又是 `a2_conventional`**，于是会被 `write_index` 按 `run_id` 并进索引。

- **现在为什么没炸**：`a1_a2_index.csv` 生成于 arch 开跑（`72b2208`）之前，仍是 96 行。
- **谁会踩**：任何一次 `04_run_cpu.py run`（跑完自动刷索引）或 `index`——索引就变 111 行，而主表填表、`96 行`断言、`Main Table Notes` 的条数三者同时失真，**且不报错**。
- **修法**（二选一）：① `index` 的过滤加一条"`task` 必须属于 `main_table.yaml` 的 16 个单元"；② 按 `config_hash ∈ {a1 的 871836…, a2 的 7d29768…}` 收。①更稳（新增任务不会自动混进来），②更严（能挡"改了配置的 A2"）。
- **归属**：`04_run_cpu.py` 是 C-a 写的、C-b 加的 `n_level` 过滤，arch 记录是 C-c 落盘的——**接口漏了，三方都合理**。建议 C-c 落（他最熟 arch 的 run_id 形态），C-a 会签。

### 5.2 【口径未决】Q-10：4 个重尾任务，只有 2 个是"纯尺度问题"

主表现标 `protocol`（差因分类）的 4 格，官方都在 **ln 尺度**、我们在原始尺度（`D-44` / S27 已写明）。我把同一套冻结超参**在 ln 空间重跑**了一遍（官方划分、seed 0、单折、未入账）：

| 任务 | 我们（原始） | 我们（**ln**） | 官方最佳树（ln） | ln 尺度差 | 判读 |
|:--|--:|--:|--:|--:|:--|
| `pb_t2_permeability` | −0.0018 | **0.6226** | 0.6600 | −0.037 | 纯尺度问题：同尺度下只差 5.6% |
| `pb_t2_modulus` | +0.0247 | **0.3413** | 0.3485 | −0.007 | 纯尺度问题：差 2.1% |
| `pb_t3_iwlt` | −0.0419 | **0.4640** | 0.7519 | **−0.288** | **仍是实质差距（38%）** |
| `pb_t3_dielectric_loss` | +0.5931 | **0.1737** | 0.4195 | **−0.246** | **方向反转**：raw 看着赢 0.17，同尺度其实输 0.25 |

**为什么这条必须改**：现在主表里 `pb_t3_dielectric_loss` 的 A2（0.593）摆在一个 0.4195 的官方数旁边、标 `protocol`，读者会读成"我们更好"；换成同一把尺子是 **0.174 vs 0.4195**。这不是精度问题，是**结论方向**问题。

**建议**（三件，按代价排）：

1. `main_table.csv` 的 `leaderboard_diff_class` 把这 4 格拆开：`scale_only`（permeability / modulus）与 `unexplained`（iwlt / dielectric_loss）。**不改任何数字，只改标签**；
2. `docs/Leaderboard Gap.md` 补上表这 4 个 ln 数（标明"单折 seed0 复核值，未入账"）；
3. 若要正式入账：跑 **12 条附录**（4 任务 × 3 种子，`04_run_cpu.py` 需加 `--log-target`，约 5 分钟），落 `results/aggregate/log_target_sensitivity.csv`，**不进主表、不进 `expected_runs`**。

> **补记（2026-11-17 晚，D-66）：第 3 条已落实，上面这张表是历史快照，不要再用它的数。**
> 实跑是 **24 行**（4 任务 × **2 臂** × 3 种子，多出来的一臂是 A1 地板——不同空间的两列不能相减），
> 落 `results/aggregate/log_target_sensitivity.csv`，命令 `python scripts/04_run_cpu.py log-target`。
> 跑完发现单点确实会骗人：**modulus 从 −2.1% 修正为 −6.4%**（0.3413 恰是三种子里最高的，均值 0.3261 ± 0.0155），
> **iwlt 从 −38% 修正为 −36.4%**（0.4640 是三种子里最低的，均值 0.4782 ± 0.0273）；permeability 与
> dielectric_loss 的种子间几乎不动（0.6203 ± 0.0020、0.1737 ± 0.0000）。**`scale_only` / `unexplained`
> 的分类不变**，但凡是引用幅度的地方一律改引 CSV 的均值。当前口径见 `docs/Leaderboard Gap.md` §3。

### 5.3 【文档与产物不一致】`main_table.yaml` 的 `cell` 段

```yaml
cell:
  aggregation: mean_over_seeds   # 主表一格 = 同一 (任务, 折, 臂) 三个种子的均值
  spread: sample_std_3_seeds     # n=3
```

实际产物是 **一格 = (任务, 臂)**、PB26 那两格 **n=15**（5 折 × 3 种子）。`Main Table Notes` 与交付文档 §4 都写对了，但 `main_table.yaml` 自称"唯一出处"，它的 `cell` 段仍停在 v1 的写法。D 画误差棒时按 YAML 读会算错（PB26 的 ±0.03 主要来自折，不是种子）。

**修法**：`main_table.yaml` 出 v3，把 `cell` 段改成 `aggregation: mean_over_folds_and_seeds` / `spread: sample_std_over_cell_runs`（PolyBench n=3、PB26 n=15），并同步 `tests/test_arms_classical.py` 里引用该段的断言。这是**文档修订，不动任何 run**。

---

## 6. 建议补的四条（不阻塞验收）

1. **重复性只测了 1 个任务**：`a2_repeatability.json` 只有 `pb_t1_tg`。A2 是确定性的（0.0），但"确定性"这条断言最好覆盖两类边界：`pb_t2_permeability`（sd=0 那条、输出常数）与一个 PB26 折（多折数据装配）。约 30 秒。
2. **`a2_best_params.csv` 已从 8 行变 13 行**（C-b 的 scaling / OpenPoly 与 C-c 的 arch 追加）。交付文档 §1 写的"8 行"已过期；文件本身没问题（按 `(task, learner)` 取），但**这张表成了共用表**，建议在表头写明"谁追加过"，或像 `gpu_hours.csv` 那样加 `owner` 列（D-37 的教训：共用表不写归属，后面没人敢用）。
3. **全仓 `pytest` 现在 3 红**，都不在 C-a 名下：
   - `test_run_records::test_3_prediction_rows_match_split_test_length` 与 `::test_composed_audit_is_clean_on_real_records`：arch 的 15 条记录指向 `splits/polybench26/arch/seed_42/fold_*.npz`，而那是**不入库的物化产物**——陌生人的克隆上审计套件必红。归属 **C-c**（要么让审计对"物化未就位"显式 skip，要么把划分索引入库并登记 D-xx）。
   - `test_scaling_scan::test_packs_are_built_from_the_official_canonical_bytes`：归属 **C-b**。
4. **4 个重尾任务的 Spearman**：不必改记录，事后从 `predictions/*.parquet` 算一张聚合表即可（重尾目标上 R² 由少数极值主导，Spearman 才是"排序还行不行"的证据）。

---

## 7. 不是问题、但一定会被问的三条

1. **`pb26_homopolymer_density` 的 A1 是 −0.0178 而不是 0**：Dummy 用训练行均值，测试集均值天然不同。不是 bug（交付文档 §4 第 4 条已写）。
2. **两格 sd 恰好 0.0000**：`pb_t3_dielectric_loss` 选中的超参 `subsample=1.0` 且 `colsample_bytree=1.0`（无随机源）；`pb_t2_permeability` 输出常数（种子无从改变）。都不是"特别稳"。
3. **PB26 两格 ±0.03**：95% 来自折间（density 跨折 sd 0.032 vs 折内 0.0015），不是模型不稳。**任何小于 0.03 的"臂间差"在 density 上都不可区分**——D 写结论时必须带这句。

---

## 8. 交接与待办

| # | 事项 | 归属 | 何时 |
|:--|:--|:--|:--|
| 1 | `index` 过滤加"主表 16 单元"（§5.1） | C-c（C-a 会签） | 下一批记录落盘前 |
| 2 | 4 格差因拆 `scale_only` / `unexplained` + Gap 文档补 ln 数（§5.2 1–2） | C-a | 报告定稿前 |
| 3 | `main_table.yaml` 出 v3，`cell` 段改对（§5.3） | C-a | 同批 |
| 4 | 12 条 ln 附录（§5.2-3） | C-a，需 `--log-target` | 可选，5 分钟 |
| 5 | arch 记录 → 物化划分的测试归属（§6.3） | C-c | 本周 |
| 6 | `a2_best_params.csv` 加 `owner` 列或表头注明追加者（§6.2） | C-b / C-c | 顺手 |

**不影响**：96 条记录、96 份预测、三份配置哈希、任何已写进主表的 R²/RMSE、B 的 546 条、C-b 的 265 条。
