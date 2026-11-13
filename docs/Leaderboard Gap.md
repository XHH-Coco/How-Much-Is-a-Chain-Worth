# Leaderboard Gap：经典三列（A1 / A2 / A5）

> **状态（2026-11-13 晚更新）：8 个任务的官方数都拿到了。A5 仍然缺数据。**
> - **PB26 两格**：官方仓库 `rlearsch/PolymerBenchmark2026`（commit `83655bc`）同时提交了分数表与划分索引，
>   10 份索引与本地**逐数组相等**（`official_split_check.json`）→ 同划分、不同训练量与模型。
> - **PolyBench 六格**：平台公开接口 `/api/com_result_data` 有 39 数据集 × 287 格的 Test R²
>   （快照 sha256 `fce27fd0…`，11,188 行）。三项核对通过：**6/6 测试集行数相等**、
>   **6/6 R² 分档计数逐档相等**、划分同源于平台的 `test-download`。
>   **但行身份核不了**（平台只回位置序号），所以这六格只能写"同来源、同大小"，不写"逐行相同"。
> - **尺度警告**：SI Table S27 写明官方对 permeability / modulus / dielectric_loss / iwlt 用 **ln(y)**；
>   我们在原始空间（Q-10 未决）。这四格的 R² **不可相减**，产物里 `r2_comparable=False`。
>   只有 Tg / Tm 两行可比。
> - **A5** 照旧 `blocked_pending_data`。一格都不填 0、不填均值。

- 交付人：C-a
- 交付日：2026-11-13
- 对应分工：`docs/分工_C-a_经典臂族.md` §2 **C-a-4**
- 数据源：`results/aggregate/a1_a2_index.csv`（96 行）+ `results/aggregate/predictions/*.parquet`
  + `results/aggregate/official_baseline_compare.csv`（10 行）+ `results/aggregate/polybench_official_compare.csv`（6 行）
  + `results/aggregate/pb26_controlled_control.csv`（50 行，受控对照）
  （全部由 `scripts/04b_official_compare.py` 与 `scripts/04_run_cpu.py` 生成）

---

## 1. 来源三要素，现在各自是什么状态

分工书要求官方数写清**版本 / 模型 / 划分**。8 个任务现在都能填，差别在"能核到什么程度"：

| 要素 | PB26（两个任务） | PolyBench（六个任务） |
|:--|:--|:--|
| 版本 | 仓库 `rlearsch/PolymerBenchmark2026`，commit `83655bc25901a8cabe3a1e5474b7ec26f01b919f`；文件 sha256 见 `data/manifests/external_official_sources.csv` | 平台 `http://202.98.16.11`，`/api/com_result_data` 快照 sha256 `fce27fd0d539…aa98f1be`；论文 DOI `10.1021/polymscitech.6c00094`。见 `data/manifests/external_polybench_sources.csv` |
| 模型 | `RDKit_RF` = 200+ RDKit 描述符 + RandomForest（100 树，中位数填充，seed = 42 + 折号），见仓库 `Models/RDKit_RF/README.md` | 每格一个 `(Algorithm, Descriptor)`；论文基线 30 类模型 × 7 类编码。比较同时给"最佳格"与"最佳树集成格" |
| 划分 | **与本地完全一致**：seed_42 五折，`test / train_pool / val` 三个数组逐位相等 | **同来源、同大小**：平台 `test-download` 是唯一官方划分，6/6 任务测试集行数与平台 `n_samples` 相等；**行身份不可核**（接口只回位置序号） |
| 标签尺度 | 原始空间（Tg/Tm 按 D-15 换成 K） | **Tg/Tm = raw，其余四个 = ln**（SI Table S27 `target_scale`）；我们全部 raw |

> 取数方式：
> - PB26：`python scripts/04b_official_compare.py fetch`（走 `raw.githubusercontent.com/<repo>/<commit>/<path>`，
>   与 GitHub contents API 的 base64 内容 sha256 逐位相同，但不吃 60 次/小时的限额）。
> - PolyBench：`python scripts/04b_official_compare.py polybench`（先 `polybench-fetch` 再 `polybench-compare`）。
> 产物只落在 `data/raw/external/`（gitignored）与 `data/manifests/external_*_sources.csv`（入库）。

**没有这三样就填数，等于把"协议差异"和"实现差异"混成一类**——分工书把这句列为 Q&A 最容易抓的点（C-a-4 ③）。
PolyBench 六格的三样现在都有出处，但"划分同源"不等于"行身份相同"、`ln` 尺度也不等于 `raw`；
这两条都写进了产物（`r2_comparable` 列与 summary 的 `caveats`），不靠口头补。

> 与分工书 §2 的 `blocked_no_gpu` 说明：那个灰标针对"缺 GPU 跑不出的格"。**本文件的缺口不是 GPU**，
> 所以用 `blocked_no_source`；A5 用 `blocked_pending_data`。两者都遵守同一条红线：**不许填 0**。

---

## 2. 逐格现状（8 任务 × 3 列经典臂）

`r2` 原单位、原始空间；`±` 见 `docs/交付_C-a_经典臂族.md` §4 的口径说明（PB26 的 ± 含折间差异）。

| 任务 | 单元数 | A1 地板（我方） | A2 经典（我方） | A5 链信息（我方） | 官方 | 我们 − 官方 |
|:--|--:|--:|--:|--:|:--|:--|
| `pb26_homopolymer_density` | 5 折 | −0.0178 ± 0.0149 | **0.9467 ± 0.0297** | `blocked_pending_data` | `RDKit_RF` **RMSE 0.0641 ± 0.0166**（train 1000） | 见 §2.1：RMSE 比 0.724 |
| `pb26_polymetrix_tg` | 5 折 | −0.0014 ± 0.0027 | **0.8926 ± 0.0079** | `blocked_pending_data` | `RDKit_RF` **RMSE 44.64 ± 1.83**（train 3000） | 见 §2.1：RMSE 比 0.832 |
| `pb_t1_tg` | 1 折 | −0.0009 ± 0.0000 | **0.8861 ± 0.0002** | `blocked_pending_data` | `0.8819` 最佳树（HistGBDT/Mordred，**raw**） | **+0.0042**（R² 可比） |
| `pb_t1_tm` | 1 折 | −0.0013 ± 0.0000 | **0.7197 ± 0.0046** | `blocked_pending_data` | `0.7153` 最佳树（XGBoost/RDKit_2048，**raw**） | **+0.0044**（R² 可比） |
| `pb_t2_modulus` | 1 折 | −0.0001 ± 0.0000 | 0.0310 ± 0.0096 | `blocked_pending_data` | `0.3485` 最佳树（ExtraTrees/RDKit_2048，**ln**） | 不可减（尺度不同） |
| `pb_t2_permeability` | 1 折 | −0.0018 ± 0.0000 | **−0.0018** ± 0.0000 | `blocked_pending_data` | `0.6600` 最佳树（CatBoost/Mordred，**ln**） | 不可减（尺度不同） |
| `pb_t3_dielectric_loss` | 1 折 | −0.0006 ± 0.0000 | 0.5931 ± 0.0000 | `blocked_pending_data` | `0.4195` 最佳树（HistGBDT/RDKit_512，**ln**） | 不可减（尺度不同） |
| `pb_t3_iwlt` | 1 折 | **−0.0329** ± 0.0000 | −0.0530 ± 0.0098 | `blocked_pending_data` | `0.7519` 最佳树（HistGBDT/Morgan_2048，**ln**） | 不可减（尺度不同） |

### 每一格怎么点回 run_id

- 命名规则：`{task}__{arm}__{fold}__seed{N}`，`arm ∈ {a1_floor, a2_conventional}`，
  PolyBench 的 `fold = official_70_30`，PB26 的 `fold ∈ {fold_00 … fold_04}`，`N ∈ {0,1,2}`。
- 例：`pb_t1_tg__a2_conventional__official_70_30__seed0`（R² = 0.8862028074232996，就是 D 那个 Λ 的输入）。
- **96 行全在 `results/aggregate/a1_a2_index.csv`**（`run_id, task, arm, learner, fold, seed, final_r2, n_train, n_test`），
  表里每个均值都是这 96 行按 `(task, arm)` 求的，没有手抄数字。

### 2.1 PB26 两格：逐折 RMSE 并排（同划分）

官方表给的是 **RMSE**，不是 R²；把两边换成同一个量才谈得上"差多少"。
我方 RMSE 由 `scripts/04b_official_compare.py compare` 从预测 parquet 与 `y` 重算（不手抄）：

| 任务 | 折 | 官方 `RDKit_RF` | 我方 A2（seed0/1/2） | 我方均值 | 我方 ÷ 官方 |
|:--|:--|--:|:--|--:|--:|
| `pb26_polymetrix_tg` | fold_00 | 44.937 | 36.635 / 37.903 / 37.777 | 37.438 | 0.833 |
| | fold_01 | 43.005 | 33.982 / 33.940 / 34.234 | 34.052 | 0.792 |
| | fold_02 | 45.061 | 38.481 / 38.637 / 39.052 | 38.723 | 0.859 |
| | fold_03 | 47.683 | 39.174 / 38.725 / 38.929 | 38.942 | 0.817 |
| | fold_04 | 42.510 | 36.576 / 36.585 / 36.727 | 36.629 | 0.862 |
| | **折间均值** | **44.639 ± 1.829** | — | **37.157 ± 1.769** | **0.832** |
| `pb26_homopolymer_density` | fold_00 | 0.06374 | 0.04206 / 0.04196 / 0.04047 | 0.04149 | 0.651 |
| | fold_01 | 0.04188 | 0.03083 / 0.03127 / 0.03149 | 0.03120 | 0.745 |
| | fold_02 | 0.09348 | 0.07242 / 0.07259 / 0.07214 | 0.07238 | 0.774 |
| | fold_03 | 0.05993 | 0.04280 / 0.04431 / 0.04481 | 0.04397 | 0.734 |
| | fold_04 | 0.06128 | 0.04274 / 0.04262 / 0.04342 | 0.04292 | 0.700 |
| | **折间均值** | **0.06406 ± 0.01663** | — | **0.04639 ± 0.01377** | **0.724** |

`±` 是**折间总体标准差**（`ddof=0`），与官方 aggregate 表的定义一致。

**2σ 判定（C-a-2 验收③ 用的就是这一条）**

| 任务 | 官方 2σ 带 | 我方均值 | 在带内？ |
|:--|:--|--:|:--|
| `pb26_polymetrix_tg` | [40.98, 48.30] | 37.157 | **否**（比带下沿还低 3.82） |
| `pb26_homopolymer_density` | [0.0308, 0.0973] | 0.04639 | **是** |

**Tg 落在带外，按分工书的规矩不能直接说"我们更好"。** 查过了，差不在协议，在三处实现：

1. **训练量不同**：官方只用了 `train_pool` 里的 3000 行（Tg）/ 1000 行（density），我方用**整个** `train_pool`（5893 / 1440）。
   官方自己的 scaling 文档说"小训练集是大训练集的严格子集"，也就是它**故意**少给样本。
2. **模型不同**：官方是 RandomForest(100 树) + 中位数填充；我方是 XGBoost（`hist`，30 次分组 CV 选参）。
3. **输入不同**：官方只有 200+ RDKit 描述符；我方是 205 描述符 **+ 2048 位指纹**。

**因此这两格只能写成"同划分、不同实现的差距"，不能写成"我们赢了 RDKit_RF"。**
反过来说也一样：若有人拿这两行说"官方基线太弱"，那是在比较 3000 行 RF 与 5893 行 XGB+指纹，不是一个受控实验。

### 2.1b 受控对照：三处差异已经拆开（2026-11-13 补跑）

三个单改变量 + 一个"全部改成官方"的复算，每项 5 折 × seed 0（另有 `baseline` 作零点）。
产物 `results/aggregate/pb26_controlled_control.csv`（50 行）与 `_summary.json`；
`baseline` 与 `raw_runs` 里同任务同折 seed0 的 A2 记录**逐折相等**（最大差 4.5e−11，CSV 十位小数所致）。

| 任务 | baseline（A2） | 只改训练量（3000/1000 行） | 只去指纹（205 列） | 只换 RF | 三项全改 | 官方 | 残差 |
|:--|--:|--:|--:|--:|--:|--:|--:|
| `pb26_polymetrix_tg` | 36.969 | 39.889（**+2.92**） | 38.542（**+1.57**） | 39.182（**+2.21**） | 44.611 | 44.639 | **−0.028** |
| `pb26_homopolymer_density` | 0.046169 | 0.051529（**+0.00536**） | 0.047077（**+0.00091**） | 0.057042（**+0.01087**） | 0.063699 | 0.064063 | **−0.00036** |

读法：

1. **"三项全改"几乎逐位复现官方**（Tg 差 0.06%，密度差 0.6% 的 RMSE 量级）——说明"同划分"这条链没有暗差，
   剩下的差就是这三处实现。
2. **Tg 上三处都重要**：训练量 2.92 / 学习器 2.21 / 特征 1.57，三项之和 6.70 ≈ 联合差 7.64（非严格可加）。
3. **密度上学习器是主因**：RF 一项就 +0.0109，占联合差 0.0175 的 62%；指纹只值 +0.0009。
4. 因此 §2.1 里的"并列候选"可以删掉：差**已经分到三项**，不是"三者之一，没拆开"。
   仍要保留的限定：这四项是**单改**，交互项未单独测（联合差 7.64 ≠ 单改和 6.70 就是交互量级）。

### 2.2 PolyBench 六格：平台排行榜并排（新）

平台每格一行 `(Dataset, Algorithm, Descriptor) → Test R²`，6 个任务各 287 行。
下表"最佳树"只取 8 类树集成（XGBoost / RF / CatBoost / GBDT / HistGBDT / ExtraTrees / AdaBoost / DecisionTree）；
"可比"一列由 SI Table S27 的 `target_scale` 决定。

| 任务 | 尺度 | 平台最佳格 | 平台最佳树 | A2（我方） | A2 − 最佳树 |
|:--|:--|:--|:--|--:|--:|
| `pb_t1_tg` | raw | 0.886449（SVM/Morgan_2048） | 0.881944（HistGBDT/Mordred） | **0.886128 ± 0.000153** | **+0.004184** |
| `pb_t1_tm` | raw | 0.719161（SVM/Morgan_2048） | 0.715289（XGBoost/RDKit_2048） | **0.719714 ± 0.003773** | **+0.004425** |
| `pb_t2_modulus` | ln | 0.395284（SVM/Morgan_2048） | 0.348486（ExtraTrees/RDKit_2048） | 0.031010 ± 0.007814 | 不可减 |
| `pb_t2_permeability` | ln | 0.672768（SVM/Maccs） | 0.660010（CatBoost/Mordred） | −0.001814 ± 0.000000 | 不可减 |
| `pb_t3_dielectric_loss` | ln | 0.419500（HistGBDT/RDKit_512） | 同左 | 0.593074 ± 0.000000 | 不可减 |
| `pb_t3_iwlt` | ln | 0.751941（HistGBDT/Morgan_2048） | 同左 | −0.053041 ± 0.007991 | 不可减 |

**四条必须一起写的话**

1. **两行可比**（Tg/Tm）：A2 比平台最佳树高 +0.0042 / +0.0044，比平台最佳格低 0.0003 / 高 0.0006。
   即"等于最强单模型、略高于最强树"，不是"大幅领先"。
2. **四行不可比**：官方在 `ln(y)` 上算 R²，我们在原始空间。B-3 的诊断（Q-10）已经证明换空间能从 −0.03 变 0.61，
   所以这四行任何"我们比官方差 0.6"的说法都是把尺度差当模型差。
3. **划分同源、大小相同，但行身份核不了**：平台预测接口只回 `index ∈ [0, n)`。我们不声称"逐行相同"，
   只声称"来自同一 `test-download` 划分 + 测试集行数 6/6 相等"。
4. **完整性有硬证据**：把 11,188 行按任务重算 R² 三档计数，与平台 `/api/com_test_r2` **6/6 任务的每一档都相等**
   （如 Tg 116 / 158 / 13）。只核行数不够，这条才证明"287 格完整取回且解析正确"。

---

## 3. 差异分类：PB26 已有结论，PolyBench 分两类

一旦官方数到位，格子只能落进三类：**协议**（划分 / 指标定义 / 标签变换 / 预处理）、**实现**（调参预算 / 编码 / 随机源）、
`unexplained`（差距 > 10% 且前两类都解释不了，必须老实写）。

**PB26 两格：归"实现"。** 依据是划分逐位相同（`official_split_check.json`：10 份 npz、每份三个数组全等），
指标都是原单位 RMSE，标签都没取对数；剩下的训练量 / 模型 / 输入三处差异属于实现层。

**PolyBench 六格：两类。** Tg / Tm → **实现**（官方 `target_scale=raw`，同划分来源、同测试集大小；
差的 0.0003/0.0006 落在编码与调参）。其余四格 → **协议**（官方在 `ln(y)` 上评分，我们在原始空间；
这是 P3 从条件句变成陈述句的一刻，依据是 S27 的 `target_scale` 列本身，不是我们的推断）。
四格的具体数值差**不归因**：尺度不同，差值没有意义。

| # | 我方协议事实 | 出处 | 如果官方不同，差在"协议" |
|--:|:--|:--|:--|
| P1 | PolyBench 用**一份固定的官方 70/30**，三个种子只换模型初始化 | D-20、`splits/polybench/*/official_70_30.json` | 官方若按别的种子/重切，两列不可直接相减 |
| P2 | PB26 训练只用 `train_pool`，**`val` 不参与训练** | `main_table.yaml`、`loaders.py:468-471` | **已核实官方也一样**：scaling 的 `train_size` 是 `train_pool` 的子集，与 `val` 无关 |
| P3 | 指标是**原始空间 R²**；四个重尾任务**没有取对数** | Q-10（`docs/DECISIONS.md`） | **已核实官方不同**：S27 `target_scale` 写明这四个任务用 `ln`；因此四格标 `r2_comparable=False`。B-3 实测同一臂换空间可从 −0.03 变 0.61，正说明这个尺度差有多大 |
| P4 | 输入 = **205 描述符 + 2048 位指纹 = 2,253 列**，其中 8 个 `BCUT2D_*` 恒为 NaN | Q-08、A-6/B-1 | 官方 `RDKit_RF` 只用 200+ 描述符、**没有指纹** → 这是"实现"差异 |
| P5 | 超参 = **30 个固定种子抽的候选**，Murcko 分组 5 折 CV，只在第一个训练折内 | `a2_conventional.yaml` | 官方基线**没调参**（RF 默认 100 树）→ "实现"差异 |
| P6 | `n_jobs=1`、`tree_method=hist`、`random_state=seed` | 同上 | 复现性差异，不算"我们更好/更差" |
| P7 | PB26 的划分是**官方索引本身** | `official_split_check.json`（本次新增） | **已排除**：官方不可能"用了别的划分" |

**现在只剩 P1 / P4 / P5 是条件句**：平台固定 70/30 与我们的划分同源（`test-download`），但不是逐行可核；
平台的 Mordred / Morgan 等编码与我们的 205+2048 不是同一实现；平台各格的超参由它自己的搜索给出，没有发布预算。
P2 与 P3 能改成陈述句，是因为这次真的读到了官方文件（官方仓库的划分索引、SI 的 `target_scale`）。
**P1 保留的原因**：不能核行身份时，"同划分"是来源级证据，不是逐行级证据。

---

## 4. 还缺什么（按优先级）

1. ~~PolyBench 六个任务的排行榜快照~~ **已解决（D-44）**：`data/raw/external/polybench_leaderboard.json`
   + `external_polybench_sources.csv`（sha256、取回时间、论文 DOI）+ `polybench_split_sizes.json`。
2. ~~PB26 的受控对照~~ **已解决（D-45，§2.1b）**：训练量 / 特征 / 学习器三项分别量化，
   `official_recipe` 复现官方 RMSE 到 0.06%（Tg）与 0.6%（密度）。
3. **仍未解决**：① Q-10 四个重尾任务的 `ln` 尺度要不要全臂统一（决定 PolyBench 四格能否填数）；
   ② PolyBench 的行身份（平台不回行号，除非拿到平台的预测行索引映射）；
   ③ A5 的三条口径（见 `docs/交付_C-a_经典臂族.md` §9）。

---

## 5. 一句话给 Q&A

> 两个"倍数"的分子不同，别当成矛盾：§2.1b 受控表只用 **seed 0**（baseline 36.969 / 0.046169 →
> 0.828 / 0.721 倍）；下面那句 0.832 / 0.724 用的是**三 seed 的折均 RMSE 再比官方**（37.157 / 0.046394）。
> 全表三 seed 的逐折值在 `official_baseline_compare.csv`。

> 我们**不声称超过已发表的基线**（Proposal 第 39 行）；我们要量的是它们没有量的东西。
> PB26 两格：**同一份五折划分**上，A2 的 RMSE 是官方 `RDKit_RF` 的 0.832（Tg）/ 0.724（密度）倍（三 seed 折均）。
> 受控对照把差拆成了三项——Tg：训练量 +2.92 / 学习器 +2.21 / 特征 +1.57；
> 密度：学习器 +0.0109 是主因。三项一起改成官方配方后，RMSE 与官方只差 0.06% / 0.6%，
> 说明差**全部**来自这三处实现，不是协议或数据。
> PolyBench 六格：Tg / Tm 两行可比，A2 比平台最佳树高 +0.0042 / +0.0044；
> 其余四行官方在 `ln(y)` 上评分，与我们的原始空间**不可相减**（Q-10）。
> A5 仍灰——灰的原因写在 `docs/交付_C-a_经典臂族.md` §9，不是没跑。
