# B-4 交付凭据：A4 polyBERT 迁移臂

> 一句话：把 B-2 那份**已经读过约一千万条聚合物串**的编码器接上任务，看它能不能直接迁移。
> 主表走 (a) 冻结嵌入 + ridge 头（48 条，6.0 秒），对照走 (b) 微调（48 条，2.684 GPU-h，两种都跑了）。
> 结论三条：**① 四个"A3 学不动"的任务，A4 冻结臂全部不比 A3 差，其中两个明显更好**；
> **② 放开权重后 6 个任务涨 +0.06 ~ +0.13**；**③ 微调后 48 条 R² 中位数 0.8956 高于 A3 的 0.8899
> ——"polyBERT 不如 GNN"只在冻结时成立，"表示不如人"不成立。**

- 交付人：B（代仓库所有者执行）
- 交付日：2026-11-12
- 对应分工：`docs/分工_B_表示与GPU臂.md` §2 B-4
- 配置：`configs/arms/a4_polybert.yaml`（冻结 v1，它的字节哈希就是每条记录的 `config_hash`）

---

## 1. 产物

| 文件 | 是什么 | 数量 |
|:--|:--|--:|
| `configs/arms/a4_polybert.yaml` | 冻结的臂配置：表示来源与哈希、ridge 头、微调段、网格、预算 | 1 |
| `results/raw_runs/<task>__a4_polybert__<fold>__seed<N>.json` | 一条 run 一份记录（主表，`mode=frozen_ridge`） | 48 |
| `results/aggregate/predictions/<同名>.parquet` | 测试集预测，两列 `row_id` / `y_pred` | 48 |
| `results/aggregate/gpu_hours.csv` | 实测墙钟（与 A3 共用一张表，按 `run_id` 覆盖，现 96 行） | 见文件 |
| `results/aggregate/a4_repeatability.json` | 冻结臂同 seed 两次的可复现性实测（验收④） | 1 |
| `results/aggregate/a4_finetune_repeatability.json` | **微调**同 seed 两次的可复现性实测（验收④；跨进程另测一次，见 §4） | 1 |
| `results/aggregate/a4_finetune/` | 微调（对照）的记录、预测、墙钟——**刻意不写进 `raw_runs/`** | 48 条 |
| `results/aggregate/a4_finetune_vs_frozen.csv` | (a) 与 (b) 同任务同折同种子的逐条对照（验收②凭据） | 1 |
| `src/chainworth/arms/polybert.py` | 本臂的全部逻辑（数据对齐、ridge、微调、对照表） | 1 |
| `scripts/05_run_gpu.py` | 入口：`--arm a4_polybert`（`--finetune` / `--compare` / `--repeat-check`） | 改 |
| `tests/test_arms_polybert.py` | 23 项断言（22 纯函数 + 1 需要 transformers） | 1 |

> **为什么微调不写 `results/raw_runs/`**：`raw_runs/` 里是主表的 48 条，文件名由 `run_id` 唯一确定。
> 微调用的是同一套任务/折/种子，`run_id` 会完全一样——写进去就是**覆盖**，主表与对照从此分不清
> 哪条是哪条。所以它落在 `results/aggregate/a4_finetune/`，由 `05_run_gpu.py --finetune` 自动
> 指向那里，`--compare` 再把两边并成一张 CSV。

---

## 2. 48 条是怎么点出来的

与 B-3（A3）**逐条对齐**：同任务、同折、同种子、同一份训练内部留出。

| 基准 | 任务 | 折 | 种子 | run 数 |
|:--|:--|:--|:--|--:|
| PolyBench | `pb_t1_tg` / `pb_t1_tm` / `pb_t2_modulus` / `pb_t2_permeability` / `pb_t3_dielectric_loss` / `pb_t3_iwlt` | `official_70_30` | 0/1/2 | 6 × 1 × 3 = **18** |
| PolyBench26 | `pb26_polymetrix_tg` / `pb26_homopolymer_density` | `fold_00` … `fold_04` | 0/1/2 | 2 × 5 × 3 = **30** |
| | | | 合计 | **48** |

这条对齐不是形式主义。主表要把 A3 与 A4 两列相减，**相减的前提是两列算的是同一件事**：

- 同一个 `split_version`（同折）——`tests/test_arms_polybert.py::test_grid_mirrors_a3_task_fold_seed` 与
  `test_a4_and_a3_share_the_same_split_version` 各守一半；
- 同一份训练内部 10% 留出（同一个 `val_split_seed=20261121`）——A3 用它选 epoch，A4 用它选 alpha。
  两臂在"哪份样本上做选择"这件事上必须一样，否则差值里混进了"两份不同验证集"的差；
- 同一个指标函数（`arms/graph.py::metrics`，A4 直接 import，不重写）。

---

## 3. 五个口径（改任何一个都等于换了一条臂）

### ① 对齐**只认规范式**，不认行号

B-1 的 `features/<task>.npz` 按 `row_id` 排；B-2 的 `embeddings.npy` 按 `psmiles_canonical` 升序排。
**两份文件的行序来源不同**，按行号拼会得到"第 i 行结构配第 j 行标签"的矩阵——不报错、不告警，
只是让 R² 变成噪声。这是最难查的一类错，所以：

- 唯一的对齐入口是 `EmbeddingTable.row_of_or_none(canonical)`；
- `tests/test_arms_polybert.py::test_embeddings_join_on_canonical_string_not_row_number` 逐行核
  "这一行的嵌入是不是它规范式那一行"——**按行号取会在这里立刻红**。

### ② 标准化只用训练行

特征与标签的均值/标准差都只从 `train` 算（`_standardise` / `target_scaler`），验证与测试行只被
变换、不贡献统计量。`test_standardisation_ignores_validation_rows` 把验证块整体平移 1000，断言
`x_mean`/`x_scale` 一位不变。

### ③ alpha 从写死的候选网格里选，选在训练内部

`head.alpha_grid` 是配置里写死的九个值（`1e-2 … 1e6`），判据是**验证集原单位 RMSE**——
与 A3 早停监控的 `val_loss` 同一个量。并列时取更小的 alpha（规则写死，不靠遍历顺序）。

> **这里有一件事必须如实写**：第一版网格只到 `1000`。跑完 48 条发现**四个小样本任务的
> 全部 12 条都选中了上界 1000**（`pb_t2_modulus` / `pb_t2_permeability` /
> `pb_t3_dielectric_loss` / `pb_t3_iwlt`，训练行 143–651，而特征是 600 维）。
> "选到边界"意味着**候选范围不够宽**，不是"这里最好"。所以把上界延到 `1e6` 重跑——代价 6 秒，
> 48 条全部重跑（不是挑着改哪一条）。**判据是"选中边界"这个客观信号，不是"结果不好看"**：
> 前者是缺陷，后者是调参。
>
> 改完之后 `pb_t2_modulus` 从 0.0046 升到 0.0583、`pb_t2_permeability` 从 −0.1238 升到 0.0016，
> 另两个不变。新网格的选中分布是 `{10: 18, 100: 18, 1000: 6, 10000: 3, 1e6: 3}`。

### ④ 线性头**没有随机性**，所以三个种子必然给出同一个数

不装作有方差。48 条里 `pb_t1_tg` / `pb_t1_tm` / `pb_t2_*` / `pb_t3_*` 这 18 条的三个种子是
**同一个数的三份副本**，`r2` 的样本标准差是 0。

**这不是"结果特别稳"，是"闭式解不掷骰子"。** 写在这里是防止有人把主表的 `±0.0000` 读成好消息。
PB26 那 30 条的 sd 非零，那是**折间**差异（5 折内容不同），不是种子间差异。

### ⑤ 两种模式跑在**两个环境**里，因为 rdkit

| | (a) 冻结 + ridge | (b) 微调 |
|:--|:--|:--|
| 需要 | rdkit（规范式对齐）+ torch（`linalg.solve`） | transformers |
| 环境 | `DSC6020_Project_GNN`（两者都有） | `DSC6020_Project_Polybert` |
| 记录里的 `env_fingerprint` | `env_gnn.json` | `env_polybert.json` |

**为什么不能一个环境跑完**：`DSC6020_Project_Polybert` 里有 torch 与 transformers，但**没有 rdkit
也没有 pandas**（`lock/polybert.pip.txt` 可查）；而 `DSC6020_Project_GNN` 有 rdkit/pandas，却没有
transformers。两边都不完整，所以分两条。

代价是"微调路径不能查嵌入覆盖"（`build_task_data(need_embedding=False)`）：那条路只拿 PSMILES
原文去 tokenize，用不到冻结向量，也就不需要规范化。本次实测冻结臂覆盖 **100%**（8 个任务
`n_rows_without_embedding` 全为 0），所以两条路径看到的样本完全相同。

> 顺带修了三处**依赖裁剪**（都是为了让 A4 能在 polybert 环境里读数据而不往冻结环境加包）：
> `chem/features.py`、`io/loaders.py` 的 `pandas` 与 `chem/descriptors` 改成函数内导入；
> `arms/graph.py` 的 `features`/`chem.graph` 改成 `build_task_data()` 内导入；
> `arms/graph.py::write_predictions` 改用 pyarrow 直写（那个环境没有 pandas）。
> 这些改动**不改任何数值**：A3 的 48 条记录一条没动，`pytest tests` 全绿。

---

## 4. 三条验收，各自的凭据

| # | 分工书要求 | 凭据 | 结论 |
|:--|:--|:--|:--|
| ① | 嵌入行与结构一一对应（抽 50 个结构反查） | `test_embeddings_join_on_canonical_string_not_row_number`（**全量逐行**，不是抽 50）+ `test_embedding_hash_uses_the_manifest_recipe`（与 B-2 的 manifest 同哈希）+ 8 个任务覆盖 100% | **通过** |
| ② | (a) 与 (b) 若都跑了，同 split/seed 比较，差异写进主表备注 | `results/aggregate/a4_finetune_vs_frozen.csv`（48 条配对、0 缺边）+ 本文 §5.3 的对照表 + **§5.3 末尾那段可照抄的"给主表的备注"** | **通过**（两条都跑了） |
| ③ | 若只跑 (a)，必须在决策记录里写明触发 §7.4 的 triage (ii)，不许悄悄降级 | 不适用：**(b) 跑了**。决策记录 D-34 写的是"为什么两条都跑、主表为什么仍用 (a)" | **不触发** |

额外一条（分工书没要求，但 B-3 立了规矩）：**同 seed 两次逐位相同**。两种模式各一份凭据：

| 模式 | 文件 | 最大预测差 | 说明 |
|:--|:--|--:|:--|
| (a) 冻结 + ridge | `a4_repeatability.json` | **0.0** | 闭式解，差为 0 是必然；跑它是把必然变成可核对的文件 |
| (b) 微调 | `a4_finetune_repeatability.json` | **0.0** | 有随机初始化、批次顺序、dropout，**不是必然**，是实测 |

(b) 那条测了两次，两次都是 0：

1. **同进程连跑两次**（`--finetune --repeat-check 2`，`pb_t3_iwlt` seed0）：`r2` 两次都是
   `0.20704386436667221`，69 个测试点的预测差全为 0，轮数 14 / best_epoch 9 也一致；
2. **两个独立进程各跑一次**（`--out-root` 指到两个空目录，避免 `resume` 跳过）：同上，
   `r2` 逐位相同。跨进程这一点很重要——它把"进程级状态残留"这种非确定性也排除掉了。

所以 `cudnn.deterministic` + `use_deterministic_algorithms` 这两个开关在微调路径上也是够的。
**但要说清口径**：2.684 GPU-h 是**在确定性模式下**测的。B-3 量过确定性相对非确定性的代价是
墙钟 ×1.92（A3）；A4 没有测非确定性版本，所以这里不能反推"关掉能省多少"，只能说这个数偏保守。

---

## 5. 实测数字

### 5.1 主表：A4 (a) 冻结嵌入 + ridge

**看板：48 条全部 `status=ok`、全部 `device=cuda`（RTX 4070）、`config_hash` 单一
（`sha256:ca4261c3…`）、16 个 `split_version`、缺嵌入 0 行、48 份预测行数与记录 `n_test` 逐一相等、
48 条逐条过 `record_problems()`（0 条报错）。**

| 任务 | 基准 | run | 训练/测试 | 选中的 alpha | R² | MAE | RMSE |
|:--|:--|--:|--:|--:|:--|--:|--:|
| `pb26_homopolymer_density` | PB26 | 15 | 1,440 / 180 | 10 | **0.9000 ± 0.0256** | 0.0447 | 0.0648 |
| `pb26_polymetrix_tg` | PB26 | 15 | 5,893 / 737 | 100 | **0.8418 ± 0.0139** | 33.11 | 45.07 |
| `pb_t1_tg` | PolyBench | 3 | 5,074 / 2,417 | 100 | **0.8236 ± 0.0000** | 33.97 | 46.00 |
| `pb_t1_tm` | PolyBench | 3 | 2,391 / 1,139 | 10 | 0.6367 ± 0.0000 | 46.61 | 63.53 |
| `pb_t2_modulus` | PolyBench | 3 | 651 / 311 | 10,000 | 0.0583 ± 0.0000 | 1.316 | 3.398 |
| `pb_t2_permeability` | PolyBench | 3 | 447 / 214 | 1,000,000 | 0.0016 ± 0.0000 | 3.7e−11 | 1.9e−10 |
| `pb_t3_dielectric_loss` | PolyBench | 3 | 175 / 84 | 1,000 | 0.2250 ± 0.0000 | 0.3158 | 0.5774 |
| `pb_t3_iwlt` | PolyBench | 3 | 143 / 69 | 1,000 | 0.1256 ± 0.0000 | 157.4 | 474.0 |

> `±` 的含义：PolyBench 六行是 0，因为线性头无随机性（见 §3 ④）；PB26 两行是**15 条
> （5 折 × 3 种子）合并**的样本标准差。**只算折间波动会更小一点**——先按折平均三个种子、再在
> 5 个折上算 sd，`homo` 是 0.0277、`polymetrix` 是 0.0150。两个口径都写在这里，是因为
> 论文里说"跨折波动"时要用后者，而"这条 run 有多稳"要用前者，混用会把 0.026 与 0.028 说成
> 两个不同的事实。（原稿把 0.0256 标成"折间"，是标错了，这里是勘误。）

**墙钟**：48 条合计 **6.0 秒 = 0.0017 GPU-h**（单条 0.04–0.45 秒）。分工书给 B-4 的预算是
0.5–3.5 GPU-h——**(a) 把它花掉了不到千分之一**。原因很简单：解一次 600×600 的线性方程组，
在 RTX 4070 上是亚毫秒级；那笔预算原本是按"微调"估的。

### 5.2 A4 与 A3 逐任务对比（主表的两个可减列）

| 任务 | A3 官方 MPNN | A4 冻结 + ridge | 差（A4 − A3） |
|:--|--:|--:|--:|
| `pb26_homopolymer_density` | 0.9776 | 0.9000 | −0.0776 |
| `pb26_polymetrix_tg` | 0.8932 | 0.8418 | −0.0514 |
| `pb_t1_tg` | 0.8857 | 0.8236 | −0.0622 |
| `pb_t1_tm` | 0.6660 | 0.6367 | −0.0293 |
| `pb_t2_modulus` | 0.0153 | 0.0583 | **+0.0430** |
| `pb_t2_permeability` | −0.0097 | 0.0016 | **+0.0113** |
| `pb_t3_dielectric_loss` | −0.0087 | 0.2250 | **+0.2336** |
| `pb_t3_iwlt` | 0.0187 | 0.1256 | **+0.1069** |

**这张表的形状比任何一个数都重要**：

- **四个"信息充足"的任务（前四行），A3 全赢，差距 0.03–0.08。** 这是应该的：图神经网络能表示
  拓扑与局部化学环境，而 600 维的平均池化向量是"整根链的一个摘要"，丢掉了位置信息。差 0.03–0.08
  正是"摘要 vs 全图"的代价。
- **四个"信息不足"的任务（后四行），A4 全赢，两个赢很多。** 这值得单开一节。

### 5.3 微调（对照，不进主表）

**(b) 跑了 48 条**，与 (a) **逐条同任务、同折、同种子、同一份训练内部 10% 留出**。超参取一次就不再动：
`lr=2e-5`、`batch 32`、`max_epochs=30`、`patience=5`、`weight_decay=0.01`、骨干不冻结、线性头 600→1、
目标仍用 train 上的 z-score（与 (a) 同一个缩放口径）。合计 **9,660.6 秒 = 2.684 GPU-h**（预算 0.5–3.5），
单条 6–1,163 秒，轮数 6–30（中位 15），**46/48 触发早停**。

| 任务 | (a) 冻结 + ridge | (b) 微调 | 差 (b) − (a) | 微调种子间 sd |
|:--|--:|--:|--:|--:|
| `pb26_homopolymer_density` | 0.9000 ± 0.0256 | **0.9599 ± 0.0141** | **+0.0599** | 0.0012 |
| `pb26_polymetrix_tg` | 0.8418 ± 0.0139 | **0.8984 ± 0.0054** | **+0.0566** | 0.0013 |
| `pb_t1_tg` | 0.8236 | **0.8831** | **+0.0595** | 0.0052 |
| `pb_t1_tm` | 0.6367 | **0.7031** | **+0.0663** | 0.0063 |
| `pb_t2_modulus` | 0.0583 | 0.0191 | −0.0392 | 0.0256 |
| `pb_t2_permeability` | 0.0016 | −0.0450 | −0.0466 | 0.0471 |
| `pb_t3_dielectric_loss` | 0.2250 | **0.3541** | **+0.1291** | 0.0974 |
| `pb_t3_iwlt` | 0.1256 | **0.2026** | **+0.0770** | 0.0401 |
| **48 条中位数** | 0.8408 | **0.8956** | **+0.0548**（两个中位数之差） | — |

> `±` 的口径与 §5.1 **逐字相同**：PB26 两行是 15 条（5 折 × 3 种子）**合并**的样本标准差
> （只算折间波动是 0.0146 / 0.0049）；PolyBench 六行对 (a) 是 0，因为线性头无随机性。
> 最后一列单独给"微调种子间 sd"——它是 (b) 独有的量，(a) 没有这个数。**不要把最后两列混着读**：
> 合并 sd 说的是"这一格有多不稳"，种子 sd 说的是"换个随机种子结果变多少"，折间 sd 说的是
> "换一批样品结果变多少"——三个问题，三个数。

**三个观察，按重要性排：**

1. **微调在 6 个任务上更好，其中四个的幅度很一致（+0.057 ~ +0.066），另两个赢更多**
   （`iwlt` +0.077、`dielectric_loss` +0.129）。四个"信息充足"的任务挤在同一个窄区间，
   不像是巧合：同一个骨干、同一份数据、同一条早停规则，只是把权重放开，收益就基本恒定。
2. **输的两个任务，恰好是 (a) 也没有学到东西的两个**（`modulus` 0.058、`permeability` 0.002，
   两者都≈0）。它们的微调种子间 sd 反而是全表最大（0.026 / 0.047），均值落到 0 附近甚至为负。
   合理的读法是：**表示里没有可提取的信号时，放开权重只会把噪声也放大**——不是"微调有害"，
   而是"这两个任务本来就没有可学的，两条路都只能给出与猜均值无法区分的答案"。§6 的 alpha
   诊断（1e4 ~ 1e12 一路拉到 −0.003 ~ +0.027）从另一侧印证了同一件事。
3. **微调之后，polyBERT 与官方 GNN 打平了**：48 条 R² 中位数 (b) **0.8956** vs A3 **0.8899**
   （(a) 是 0.8408）。逐任务看，四个"信息充足"的任务两臂差距都在 ±0.04 内（homo −0.018、
   polymetrix +0.005、tg −0.003、tm **+0.037**），差别落在种子波动量级；而 A3 塌掉的两个任务
   （`dielectric_loss`、`iwlt`），微调版分别高出 **+0.363 / +0.184**。
   这一条对 H3 很重要：**"polyBERT 不如 GNN"只在冻结时成立，"表示不如人"并不成立**。

**主表为什么仍用 (a)、不用 (b)**：分工书写的是"默认跑 (a)"，且 (a) 是闭式解——没有随机种子、
没有批次顺序、6 秒可完全重算，(b) 的种子间 sd 在小样本任务上到 0.05–0.10。主表的每一格要能被
独立复核，(a) 满足，(b) 只能复现到"同 seed 逐位相同"（见 §4 第四条）。所以 (b) 的数字出现在这里
和 `a4_finetune_vs_frozen.csv` 里，作为"同一份表示配上训练能走多远"的上限说明。

**两条必须写下来的运行事实**：

- **`max_epochs=30` 对最大的任务偏紧**：`pb_t1_tg` 的 seed2 跑满 30 轮、best 落在第 29 轮，
  即还在降；两条跑满上限的 run 都是最大任务。所以 (b) 的数字是"30 轮预算内的微调"，不是
  "收敛后的微调"。要改就是网格级改动（48 条全部重跑），不单方面改。
- **同进程连跑会变慢，最终按每条 run 一个进程跑完**：同一条 `pb26_polymetrix_tg` 在"已经跑过
  18 个其他任务"的进程里是 37–39 秒/轮，在刚启动的进程里是 27–31 秒/轮（差约 30%，12 GB 卡上
  该任务本身就占 11.9 GB）。所以 48 条后半段改成逐条独立进程 + `resume` 跳过已完成。
  **墙钟 300 秒与 1,163 秒的差别几乎全来自 `epochs_run` 是 11 还是 30，不是显存**——同一进程内
  `s/ep` 是平的，第一版把这条归因写错了，已勘误。

**给主表的备注（验收②要的"差异写进主表备注"，这是可照抄的一段）**：

> A4 列取**冻结嵌入 + ridge**：无随机性，三个种子给出同一个数，任何人可重算到逐位相同。
> 同一份网格上的**微调对照**（48 条，2.684 GPU-h，`results/aggregate/a4_finetune_vs_frozen.csv`）
> 在 8 个任务里 **6 个更高**（+0.057 ~ +0.129）、2 个更低（`pb_t2_modulus` −0.039、
> `pb_t2_permeability` −0.047）——后两个正是两臂都"与猜均值无法区分"的任务，差值落在种子波动内。
> 48 条的 R² 中位数：微调 **0.8956** > A3 **0.8899** > 冻结 **0.8408**。
> **主表列用冻结臂的理由是"可重算"，不是"更好看"。**

---

## 6. 四个"A3 学不动"的任务，A4 反而更好——这不是矛盾

B-3 的交付文档里写过：`pb_t2_modulus` / `pb_t2_permeability` / `pb_t3_dielectric_loss` /
`pb_t3_iwlt` 这四个任务的 R² 接近 0，因为 IwLT 依赖测试温度与时间、介电损耗依赖频率、渗透率依赖
气体与温度——而网络只看到重复单元的化学结构。当时的原话是：

> 如果 A4/A5 在这四个任务上突然变好，那才是信号。

现在的实测是：**A4 在四个上都 ≥ A3，其中 `pb_t3_dielectric_loss`（0.225 vs −0.009）与
`pb_t3_iwlt`（0.126 vs 0.019）明显更好。**

这与"缺条件"不矛盾，两者说的是不同的事：

1. **这四个任务的标签噪声占比极高**，样本又小（84 / 69 / 214 / 311 个测试点）。在这种设定下，
   **容量越大越容易过拟合**。A3 的 MPNN 有约 60 万参数、要在 143–651 行上训练，早停救命也只能
   停在"跟均值差不多"；A4 的 ridge 在同等数据上只有 600 个自由度、还有强正则（选中的 alpha
   是 1e3–1e6），所以它退得**更稳**——R² 略高于 0 而不是略低于 0。
2. **"略高于 0"不等于"学到了"**。0.225 的 dielectric_loss 仍然远不能做预测，它的意义是
   "polyBERT 嵌入里确实有**一点点**与介电损耗相关的结构信号，而 A3 的表达没能把它稳住"。

`pb_t2_permeability` 值得单独说，因为它的 alpha 选到了网格上界（1e6）。我为此做了一次诊断
（不进主表），把 alpha 一路拉到 1e12：

| alpha | 验证 RMSE | 测试 R² |
|--:|--:|--:|
| 1e4 | 5.254e−11 | +0.0267 |
| 1e5 | 4.272e−11 | +0.0203 |
| **1e6（选中）** | **4.068e−11** | **+0.0016** |
| 1e7 | 4.075e−11 | −0.0023 |
| 1e8 … 1e12 | 4.077e−11 | −0.0027 |

两个结论：

- **选到 1e6 不是网格掐住了**：1e7 之后验证 RMSE 反而略差（4.075 > 4.068），1e6 是真正的验证
  最优。网格够宽。
- **无论怎么选，测试 R² 都在 −0.003 ~ +0.027 之间**，即"与直接猜均值无法区分"。线性头在原始
  空间对这个任务无可为——标签跨 16 个数量级（1e−17 ~ 3.8e−09），在原始空间做最小二乘，
  那几个大值会把权重全拽过去。

这正是 **Q-10**（四个重尾任务是否改对数目标）要回答的问题。B-3 的 ln 空间诊断显示 MPNN 在
permeability 上能从 −0.03 提到 0.61；A4 的这张表从另一半印证了同一件事：**问题在目标空间，
不在表示**。此事仍待 C/D 与仓库所有者定，**没有单方面改 config**。

---

## 7. 怎么重建、怎么验

```bash
# 0) 前置：B-1 的装配产物与 B-2 的嵌入都在（都不入库）
ls data/processed/polybert/{embeddings.npy,index.parquet}

# 1) 只打印 48 条网格，不训练、不写文件
PYTHONPATH= D:/Miniconda3/envs/DSC6020_Project_GNN/python.exe \
    scripts/05_run_gpu.py --arm a4_polybert --dry-run

# 2) 主表：48 条冻结 + ridge（约 6 秒 + 每个任务一次对齐，合计不到 1 分钟）
PYTHONPATH= D:/Miniconda3/envs/DSC6020_Project_GNN/python.exe \
    scripts/05_run_gpu.py --arm a4_polybert --require-gpu --no-resume

# 3) 可复现性：同 seed 两次，最大预测差必须是 0
PYTHONPATH= D:/Miniconda3/envs/DSC6020_Project_GNN/python.exe \
    scripts/05_run_gpu.py --arm a4_polybert --tasks pb_t1_tg --seeds 0 --repeat-check 2
cat results/aggregate/a4_repeatability.json

# 4) 对照：48 条微调（实测 2.684 GPU-h，polybert 环境），再并成一张 CSV
#    注意：同进程连跑会变慢（见 §5.3 末条），48 条建议按任务分进程，或按每条一个进程跑
PYTHONPATH= D:/Miniconda3/envs/DSC6020_Project_Polybert/python.exe \
    scripts/05_run_gpu.py --arm a4_polybert --finetune --require-gpu --tasks pb26_polymetrix_tg
PYTHONPATH= D:/Miniconda3/envs/DSC6020_Project_GNN/python.exe \
    scripts/05_run_gpu.py --arm a4_polybert --compare

# 5) 微调的可复现性：同 seed 两次，跨进程再验一次（换两个空目录，别让 resume 跳过）
PYTHONPATH= D:/Miniconda3/envs/DSC6020_Project_Polybert/python.exe \
    scripts/05_run_gpu.py --arm a4_polybert --finetune --tasks pb_t3_iwlt --seeds 0 --repeat-check 2
cat results/aggregate/a4_finetune_repeatability.json

# 6) 测试（core 环境，不需要 GPU；239 passed / 7 skipped，其中 skipped 含下面这条要 transformers 的）
PYTHONPATH= D:/Miniconda3/envs/DSC6020_Project_Core/python.exe -m pytest tests -q

#    那条"要 transformers"的微调用例（test_finetune_smoke_writes_a_consistent_record）
#    **两个环境都跑不了 pytest**：core/GNN 有 pytest 没 transformers，polybert 有 transformers 没 pytest
#    （`environment/polybert.yml` 里不含 pytest，按仓库纪律不擅自往冻结环境里装）。
#    等价的直调断言（实测通过，1 轮，r2 −0.048 是只训 1 轮的预期值）：
PYTHONPATH=src D:/Miniconda3/envs/DSC6020_Project_Polybert/python.exe -c "
from chainworth.arms import polybert as P
cfg = P.load_arm_config('configs/arms/a4_polybert.yaml')
o = P.train_one_finetune(P.build_task_data_finetune('pb_t3_iwlt'), cfg,
                         fold='official_70_30', seed=0, device='cuda', max_epochs=1, patience=0)
assert o.record['mode'] == 'finetune'
assert o.record['env_fingerprint'] == 'results/env_fingerprints/env_polybert.json'
assert o.y_pred.size == o.y_true.size == o.record['n_test']
assert P.record_problems(o.record) == []
print('OK')"
```

**核对一句一条**：

| 想知道 | 命令 | 合格的样子 |
|:--|:--|:--|
| 网格对不对 | `--dry-run \| wc -l` | 48 |
| 记录齐不齐 | `ls results/raw_runs/*a4_polybert*.json \| wc -l` | 48 |
| 预测齐不齐 | `ls results/aggregate/predictions/*a4_polybert*.parquet \| wc -l` | 48 |
| 配置有没有被改过 | `sha256sum configs/arms/a4_polybert.yaml` 与记录里的 `config_hash` 比 | 一致 |
| 两臂是不是同一折 | 记录的 `split_version` 与 A3 同名任务比 | 一致 |
| 微调有没有偷偷覆盖主表 | `ls results/aggregate/a4_finetune/raw_runs \| wc -l` | 48，且 `raw_runs/` 里没有 `mode=finetune` |
| (a)(b) 配对全不全 | `--compare` 的打印 | `48 条配对；只有微调 0 条 / 只有冻结 0 条` |
| 微调墙钟 | 微调记录里 `wall_clock_s` 求和 | 9,660.6 秒 = 2.684 GPU-h |

---

## 8. 没做到的

1. **没有对 A4 做任何超参搜索**。alpha 是唯一的可调量，它在一份写死的候选网格上按验证集选，
   且选择规则写死在配置里。`hidden`、池化方式、max_tokens 全部取 B-2 的既有设定，一个没动——
   动了就不再是"同一份表示"。这与 A3 的"不调参"是同一条纪律（红线 2）。
2. **(b) 的超参没有调**：`lr=2e-5`、`batch 32`、`≤30 epoch`、`patience 5` 是 BERT 类微调的常用
   量级，取一次就不再动。所以 (b) 的数字应当读作"**一次常规微调能到什么水平**"，不是"最好的微调"。
   其中 `max_epochs=30` 对最大任务**确实偏紧**：`pb_t1_tg` seed2 与 `pb26_polymetrix_tg` fold_00
   seed2 都跑满 30 轮，best 分别落在第 29 轮与第 25 轮（都还在降），另 46 条都在 30 轮内早停。
   改它要 48 条全重跑，属于网格级改动，没单方面动。
3. **没有做"冻结前 N 层 + 微调后 N 层"的梯度**：那需要额外的 48×k 次运行，收益不清。要做的话
   是 B-5 之后的事。
4. **没有碰 Q-10**（对数目标）。§6 的诊断只是**证据**，改口径要四臂同时改（Proposal §5 第 4 条），
   不是一条臂能决定的。
5. **没有把微调权重入库**：`.ckpt` / `.pt` 都没留，`config_hash` + seed 能重建。可复现性**已实测**
   （同进程两次 + 跨进程两次，`r2` 全精度相同、预测最大差 0，见 §4），但重建要重新花 2.684 GPU-h。
6. **微调的可复现性只测了最小的任务**（`pb_t3_iwlt`，228 行，约 10 秒一条）。选它是因为"能反复跑
   而不占卡"，代价是：大任务（`pb26_polymetrix_tg`，单条 300–1,163 秒）没做同样的两次比对。
   确定性开关是进程级的、与任务大小无关，所以推断上应当一致——但这是推断，不是实测，如实记在这里。
7. **微调记录的 `batch_size` / `max_epochs` / `patience` / `lr` 只在 `hyperparameters` 里，不在顶层**
   （A3 的记录两处都有）。主表不需要这几个字段，但**若 C 的汇总脚本按顶层取，会取到空值，要改成
   读 `hyperparameters`**。不改的原因：改记录结构要 48 条全部重跑，为一处冗余不值得。
8. **同进程连跑的墙钟变慢只测到现象，没查根因**（37–39 秒/轮 vs 27–31 秒/轮，见 §5.3）。
   规避手段是"每条 run 一个进程"，而不是修好它；`empty_cache()` 加在每轮结束时，但没做
   "加与不加"的对照实验，所以不能声称它解决了多少。
9. **有一条用例在任何环境里都不会自动跑**：`test_finetune_smoke_writes_a_consistent_record` 要
   transformers，而装了 transformers 的 `DSC6020_Project_Polybert` **没有 pytest**（另外两个环境有
   pytest 但没 transformers）。按仓库纪律没往冻结环境里擅自装 pytest，所以这条一直是被 skip 的。
   本轮用等价的直调断言在 polybert 环境里跑过（§7 第 6 条），但它没有进入任何一次 `pytest tests`
   ——**这是测试覆盖的真实缺口，不是"跑过就算"**。要补的话是给 polybert 环境加一条 dev 依赖并
   在 `lock/` 里登记，属于分工 A 的范围。
