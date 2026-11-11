# 任务清单快照 D0（Task List Snapshot D0）

`冻结时刻：2026-11-12T03:48:00+08:00` · 冻结人 **A**（任务 `A-4`）· 版本 **v1（D0）** · 阶段门 `P1 → P2`
配套文件：`configs/tasks/*.yaml`（18 份）+ `configs/tasks/SHA256SUMS` + `src/chainworth/io/tasks.py`（读取与校验）

> **这份快照的作用只有一个：让任务集在 D0 之后不能悄悄漂移。**
> 主表是"8 个任务、每层 2 个"这个排序主张的全部承重结构。任务换了、层号换了、n 换了，而没人记下来，
> 后面所有 Λ 的排序结论都会变成一个查不出原因的错。所以：**此后的任何替换，都要在 §7 追加一行，
> 写清时间戳、替换了谁、为什么、谁批准的。没有这一行的替换，视为没发生，也不许出现在主表里。**

---

## 1. 一句话

**8 个主表任务（PolyBench 每层 2 个 + PolyBench26 2 个）、5 个预注册备用、3 个 OpenPoly 边界性质、2 个机制实验任务，共 18 份配置，全部落到 `configs/tasks/`，每份的 `n` 与单位都能追到上游交付物，任何字段缺失或任务名拼错都会在读取时当场报错。**

打个比方：这是给全队发的**一份 18 行的"用料清单"**。清单上每一行写清"用哪种料（数据集）、要哪个指标（目标性质）、按什么单位记账、有多少可用样本、按哪份图纸下料（官方划分）"。
清单本身不做菜，但它一旦贴到墙上，就没人能悄悄换掉某种料——**换了必须写在变更栏里，签名、写日期**。

---

## 2. 冻结清单（**由脚本打印，不要手抄**）

下表由 `PYTHONPATH=src python -m chainworth.io.tasks --snapshot` 生成；`tests/test_task_configs.py::test_snapshot_doc_matches_configs` 会逐行比对，
**手改这张表会让测试变红**。`层` 为 `-` 表示该基准没有 PolyBench 的三层划分（PolyBench26 / OpenPoly / 机制实验）。
`单位` 为"未证实"表示官方没声明、我们也拒绝猜（§5）；"不需要"表示该任务只比较 RMSE、单位不进入判决（§5）。

| task_id | 基准 | 角色 | 层 | n | 单位 | 来源 | 状态 |
|---|---|---|---:|---:|---|---|---|
| `pb_t1_tg` | polybench | main | 1 | 8055 | K | experimental | 可用 |
| `pb_t1_tm` | polybench | main | 1 | 3796 | K | experimental | 可用 |
| `pb_t2_modulus` | polybench | main | 2 | 1034 | GPa | experimental | 可用 |
| `pb_t2_permeability` | polybench | main | 2 | 711 | cm3(STP)*cm/(cm2*s*Pa) | experimental | 可用 |
| `pb_t3_dielectric_loss` | polybench | main | 3 | 279 | 无量纲 | experimental | 可用 |
| `pb_t3_iwlt` | polybench | main | 3 | 228 | hour | experimental | 可用 |
| `pb26_homopolymer_density` | polybench26 | main | - | 1800 | 未证实 | md | 可用 |
| `pb26_polymetrix_tg` | polybench26 | main | - | 7367 | K | experimental | 可用 |
| `pb_t1_bandgap_chain` | polybench | reserve | 1 | 0 | 未证实 | experimental | 不可用 |
| `pb_t1_td` | polybench | reserve | 1 | 4898 | K | experimental | 可用 |
| `pb_t2_tssab` | polybench | reserve | 2 | 1034 | GPa | experimental | 可用 |
| `pb_t3_dlt` | polybench | reserve | 3 | 238 | 无量纲 | experimental | 可用 |
| `pb_t3_dmp_sm` | polybench | reserve | 3 | 387 | GPa | experimental | 可用 |
| `op_dielectric_constant_total` | openpoly | boundary | - | 295 | 无量纲 | experimental | 可用 |
| `op_tg` | openpoly | boundary | - | 443 | K | experimental | 可用 |
| `op_tm` | openpoly | boundary | - | 168 | K | experimental | 可用 |
| `pb26_arch_transfer_ea` | polybench26 | mechanism | - | 6478 | 不需要 | dft | 可用 |
| `pb26_scaling_cp` | polybench26 | mechanism | - | 29602 | 不需要 | md | 可用 |

**读表要点（三句，避免串行）**

1. **`n` 是"可用标签数"，不是文件行数。** `pb_t1_bandgap_chain` 原始有 3,380 行、可用 0 行（整列弃用），表里就写 0——把 3,380 写成可用数，D 算 Λ 的分母就错了 3,380。
2. **`pb26_scaling_cp` 的 29,602 = 27,802 + 1,800**，是两个数据集相加（交替 + 均聚），逐数据集的行数与档位在 YAML 的 `datasets` 里，且与官方 `manifest.json` 逐项相等。
3. **`pb26_arch_transfer_ea` 的 6,478 是训练集**（交替 6,110 + 均聚 368）；另有固定的 OOD holdout 36,828（random 18,414 + block 18,414），**它永远不进训练或调参**，所以不计入 `n`。

---

## 3. 冻结了哪些字段（schema）

字段由 `src/chainworth/io/tasks.py` 的 `TASK_KEYS_REQUIRED` 定义，**20 个键必须存在**（值可以是 `null`，键不能少）。

| 字段 | 含义 | 谁填 / 从哪来 |
|---|---|---|
| `task_id` | 任务 ID，等于文件名主干 | A-4，命名见 README §8 |
| `dataset_name` | 稳定数据集名；PolyBench 即平台数据集名 | A-2 的 39 行标准表（拼错即报错） |
| `benchmark` | `polybench` / `polybench26` / `openpoly` | Proposal §3.1 |
| `role` | `main` / `reserve` / `boundary` / `mechanism` | 表 A–E |
| `tier` + `tier_basis` | 层号与其依据；无三层划分的基准写 `null` + 理由 | 表 A（层号在正文被点名）、表 E |
| `property` / `target_column` | 人读的性质名 / 下游要读的那一列 | A-3 的调和表（`label`）或官方 CSV 列名 |
| `target_unit` + `unit_status` + `unit_evidence` | 单位、单位状态、依据 | `unit_decisions.csv` 或官方表头（§5） |
| `source_kind` | `experimental` / `md` / `dft` / `mixed` | 发布方文档 |
| `n_labels` + `n_source` | 可用标签数 + 出处 | A-3 清单或官方 `manifest.json` |
| `data_ref` | 下游该读哪个文件 | A-3 的 parquet 或官方 CSV |
| `eval_unit` / `split_protocol` / `split_ref` / `split_status` | 官方评估单元 / 协议 / 划分文件 / 是否已到位 | A-5（当前全为 `pending`） |
| `arms` | 在这个任务上跑哪些臂 | 见 §4 |
| `status` + `status_note` | 可用性；不可用必须写原因 | A-3 / D-16 |

---

## 4. 口径锁定：主表是 **8 任务 × 5 臂**（结案 Q-01）

**结论：主表 = 8 个任务 × 5 条臂（A1–A5）。`configs/tasks/*.yaml` 的 `arms` 字段按这个写死，之后不许两种说法并存。**

为什么是 5 而不是 4（三处独立证据，不是投票）：

1. **臂的集合是 Proposal §6 定义的**：A1 地板、A2 常规、A3 图、A4 语言模型、**A5 链信息**。§6 的表是臂的唯一定义处。
2. **§5 的 H3 需要 A5**："信息胜过架构"这条主张的整个意义就是"注入链信息 vs 换更强的架构"。把 A5 从主表拿掉，H3 就从主表里消失了。
3. **§8 的主表（图 6）与 §7.1 的分工都按 5 条臂写**：C 跑 A1/A2/A5、B 跑 A3/A4；若只有 4 条臂，C 的活里会莫名少一条。

那 §6 与 §7.1 里的"4 臂 / 96 次经典 + 48 次 GNN / 150 次常规"怎么办：**它们是旧稿残留，按 Q-02 处理，不进本快照**。
本快照**只冻结"哪些任务、哪些臂"，不冻结"总共多少次拟合"**。理由如下，且这条理由本身也要记下来，免得后人以为我们算不明白：

| 数 | 出处 | 按官方评估单元复算 | 结论 |
|---|---|---|---|
| 每臂 48 次 | —— | PolyBench 6 任务 × 1 个官方 70/30 × 3 种子 = 18；PolyBench26 2 任务 × 5 折 × 3 种子 = 30；18 + 30 = **48** | 48 是**每臂**的量 |
| 144 = 96 + 48 | §6 机制实验表"主表"行 | 96 经典 = 2 臂 × 48，48 GNN = 1 臂 × 48 → **恰好 3 臂** | 与"4 臂""5 臂"都对不上，是旧稿按 3 条臂写的 |
| 150 | §7.1 工时表"常规基线 ×4"、§7.4"四条常规臂" | 150 − 144 = **6 次无出处** | Q-02 原话即"差 6 次没有出处" |
| 778（全队总拟合） | DECISIONS D-10 | 150 + 48 + 100 + 30 + 225 + 225 = **778** | 778 里含的是 150，不是 144 |

**所以：`778` 这个总数暂时是"按 150 记的"，而 150 本身没有被复算过。** 这不是 A-4 能单方面改的数（它同时牵动 C 的 22.0 pH 与 D 的审计口径），
留给 **Q-02（C 定）**：C 在 P2 的超参空间与预算表定稿时，按"每臂 48 次 × 实际臂数"重算一遍，并说明 A1 地板与 A5 注入是否各自独立成格。
**在那之前，任何文档引用"107 次""192 次""240 次"都是编的**——本快照不写这种数。

> 这一条同时也是对 Q-05 的回应：主表**不加**架构迁移那一栏。架构迁移（GTRC 许可）只在机制的 C-4 里出现，
> 与主表 8 个任务不混（表 C 原文：机制实验不进主表）。

---

## 5. 单位：3 种状态，2 处"未证实"，0 处"猜"

`unit_status` 只有三种取值，且各有硬条件（`tasks.py` 会检查）：

| 状态 | 含义 | 本快照里有谁 |
|---|---|---|
| `verified` | 有证据（列名、官方文档），且 `target_unit` 非空 | 16 个任务 |
| `unverified` | **官方没声明，我们拒绝猜**：`target_unit` 必须为 `null` | `pb26_homopolymer_density`、`pb_t1_bandgap_chain` |
| `not_required` | 该任务只比较同一性质内的 RMSE，单位不进入 κ / Λ / 排序判决 | `pb26_scaling_cp`、`pb26_arch_transfer_ea` |

两处"未证实"要单独说清，因为它们以后一定会被问：

1. **`pb26_homopolymer_density`（密度）**：官方 `README.md`、`DATA_PROVENANCE.md`、`SCALING_EXPERIMENTS_METHODS.md` 三处都**没有**写密度的单位，
   源 `Cleaned_OMersBench_v3_final.jsonl` 的 `outputs.density` 只有一个数值（首条 `1.1222…`）。数值量级像是 g/cm³，但**"像"不是证据**——
   A-3 在 `bandgap chain_Egc` 上已经立过这个规矩（D-16），这条不能破例。**报告该列时必须写"官方未声明单位"**；补到一手证据后追加一条决策（D-19），不许就地改本文件。
2. **`pb_t1_bandgap_chain`（带隙）**：表头 `bandgap_chain` 不写单位，整任务弃用（D-16），可用标签 0。它仍在名单里、标 `role: reserve`，
   是因为"预注册的备用名单"本身是证据——**删掉它，等于假装我们没选过它**；标 `unavailable` 才是诚实的写法。

---

## 6. 与上游的逐项对账（谁的数、追到哪）

| 配置字段 | 对账对象 | 谁在查 |
|---|---|---|
| PolyBench 6+4 个任务的 `n_labels` | `data/manifests/harmonised_manifest.csv` 的 `n_labels` | `cross_check_upstream()` / `pytest` |
| PolyBench 6+4 个任务的 `target_unit` | 同一清单的 `label_unit`（即 `unit_decisions.csv` 的 `target_unit`） | 同上（D-15：A-3 是唯一口径） |
| PolyBench 的 `dataset_name` | `polybench_expected_files.csv` 的 39 行（拼错即报错） | `load_task_config()` |
| PolyBench26 的 `n_rows` / `train_sizes` / `target_column` / `canonical_source` | 官方 `scaling_indices/**/seed_42/manifest.json` | `cross_check_upstream()` |
| OpenPoly 的 `n`（443 / 295 / 168） | Proposal §2.3（2026-11-11 直接数发布 CSV） | **A-7 复核**；复核后若变，追加决策并改 YAML |
| 机制实验的训练量 | 官方 task4 与 scaling 的文档（`docs/分工_C` C-3/C-4） | C（他是使用者） |

**怎么验（三条命令，任一条红就说明冻结物被动了）**

```bash
# 1) 校验 18 份配置：字段、枚举、层数、与上游清单的相等性
PYTHONPATH=src python -m chainworth.io.tasks --verify

# 2) 打印快照表（改配置后重跑这一条，把 §2 的表整块替换）
PYTHONPATH=src python -m chainworth.io.tasks --snapshot

# 3) 回归测试（含"故意改坏配置必须报错"的 6 条负向断言）
python -m pytest tests/test_task_configs.py -q

# 4) 冻结物自身的指纹
cd configs/tasks && sha256sum -c SHA256SUMS
```

---

## 7. 替换登记（**只追加，不覆盖；没有这一行就不算替换**）

当前：**无替换**（v1 即初版）。任何替换按下面格式追加一行，并按 README §9 出 `configs/tasks` 的 v2 快照。

| 时间 | 换出 | 换入 | 触发（QA 不合格的具体原因） | 批准 | 对结论的影响 |
|---|---|---|---|---|---|
| —— | —— | —— | —— | —— | —— |

**替换规则（预注册，不许临场发明）**

1. **先求同层，再求同族。** 第 3 层只有 `pb_t3_dlt`（介电损耗）是干净替换；换 `pb_t3_dmp_sm`（DMA 模量）会削弱"降解"那条轴，**必须在主表 Notes 里写明**（表 E 原文就要求这样标）。
2. **第 1 层的电子类备选已经没有了。** `bandgap chain_Egc` 因单位弃用（D-16），所以第 1 层若出问题，只剩 `pb_t1_td`（热学，同族）。
3. **换任务必须同时改三处**：`configs/tasks/*.yaml`、本快照 §2 的表（重跑 `--snapshot`）、`configs/tasks/SHA256SUMS`；漏一处 = 快照作废。
4. **不许因为"跑出来不好看"而换。** 触发替换的只有 A-3 的三条 QA：缺 PSMILES、单位歧义无法消解、可用标签 < 200。

---

## 8. 没做到的事（写在这里，不要等被问）

| # | 没做到 | 影响 | 谁来补 |
|---|---|---|---|
| 1 | **`split_ref` 全是 `pending`**：18 个任务的划分文件都还没落盘 | 谁也不能开始正式训练；B/C 只能先写装配与实验入口 | A-5（P2） |
| 2 | **PolyBench26 与 OpenPoly 没有调和表**：`pb26_*` 直接读官方 CSV（Tg 已是 K），OpenPoly 的 `n` 还是 Proposal 的数 | OpenPoly 的 443/295/168 未独立复核 | A-7（1.0 pH） |
| 3 | **密度单位未证实**（§5 第 1 条） | 主表报告该列时必须标"官方未声明" | 谁找到一手证据谁追加决策 |
| 4 | **拟合次数未冻结**（§4 末） | 主表"多少次 run"暂时只有 778 这个含 150 的总数 | Q-02：C 在 P2 预算表定稿时结案 |
| 5 | **`config_hash` 还没生成** | P3 之前，run 记录无法与配置哈希对齐（README §9 第 7 行） | P3 随 `configs/arms/*.yaml` 一起做 |
| 6 | **A-8（Table S27 转录）没做** | 不影响主表：所有主表任务的层号都已在正文被点名 | 可选，不计入 24 pH 配额 |

---

## 9. 变更记录

| 日期 | 变更 | 谁 |
|---|---|---|
| 2026-11-12 | 建档 v1：18 份任务配置冻结（8 主表 + 5 备用 + 3 边界 + 2 机制）；Q-01 结案为"8 任务 × 5 臂"；记录 §5 两处单位未证实与 §4 拟合次数未冻结 | 仓库所有者（代 A 执行 A-4） |
