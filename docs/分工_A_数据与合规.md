# 分工 A · 数据与合规（Data & Compliance）

> **对应**：`docs/Proposal_HowMuchIsAChainWorth_v1.3.md` §7.1 的负责人列 `A`、§9 的 `A — 数据与合规`
> **工时**：硬项 **23.5 人时** + 审计（入口侧）**5.0** + 弹性余量 **3.5** = **32.0 人时**
> **你负责的目录**：`data/`、`splits/`、`configs/tasks/`、`results/env_fingerprints/`、以及 `tests/` 里与划分/泄漏相关的那部分
> **一句话**：让别人算出来的每一个数，都能追到某一行、某一次运行、某一条换算规则。
>
> **三人均分口径**：全队技术工时 96 pH（原 105 pH 减去写作与汇报的 9 pH，那 9 pH 由仓库所有者承担）。A / B / C 各 **32.0 pH**，差值 ≤ 2 pH。

---

## 0. 先用大白话讲：你这份差事是干什么的

### 你要干的活，就是"配试剂 + 校准天平"

想象三个人要合写一篇实验报告（写报告本身由仓库所有者负责，不在三个人的分工里）。

- B 负责把五个模型放上去跑（像做实验的人）；
- C 负责把跑出来的数字算成结论（像记账和当裁判的人）；
- **你负责的是在他们动手之前，把试剂配好、把天平校准、并且在实验记录本上签字。**

配试剂听起来最没意思，但它是唯一能让"我们的结果"和"一堆没法验证的数字"区分开的东西。举三个真实的翻车方式，全部**不会报错**：

1. 有人把某个任务的 Tg 当成摄氏度，其实文件里是开尔文。模型照样训练，R² 照样好看，只是那个任务的结论从根上就是错的。
2. 有人图省事自己 `train_test_split` 切了一刀。代码跑得通、结果很漂亮，但**官方划分**被绕过了——按课程红线，这张表直接作废。
3. 有人在算描述符时先对全量数据做了标准化，再切分。这叫把测试集信息泄漏进训练，误差会好看得不像话。

这三种情况，**一个异常都不会抛**。能拦住它们的只有两样东西：一份锁死的划分文件，和一套自动跑的断言测试。**这两样就是你的产出。**

### 你的东西被谁用、什么时候用

| 谁 | 用你的什么 | 什么时候 |
|:--|:--|:--|
| B 建模 | `data/processed/` 下的描述符与嵌入、`configs/tasks/*.yaml`、`splits/` | P1 一开工就要 |
| C 评测 | 每个任务的 `n`、单位、`split_ref`、scaling 的档位与折索引、以及"哪条换算规则影响了多少行" | 算 Λ 与 κ 时必须知道分母是什么、档位怎么截 |
| 仓库所有者（写作与汇报，不占分工） | `data/manifests/` 里的条数与 sha256、单位换算决策表 | 被问"你们的数据可信吗"时 |

### 做完的标志（一句话）

> **另一个人拿到你的提交，只用 `git clone` + 两条命令，就能把 8 个主表任务的数据原样重建出来，且划分与官方逐位一致。**

---

## 1. 你在整条流水线上的位置

README 定的是**阶段串行**：上一阶段没验收，下一阶段不开工。**分工不是绕过阶段门**——同一个阶段内三个人可以同时干活，跨阶段必须等门。

| 阶段 | A（你） | B | C |
|:--|:--|:--|:--|
| **P0 环境** | ● **主责**：三套环境装包锁版本 | ○ 等 | ○ 先用合成数据写指标单测 |
| **P1 获取数据** | ● **主责**：四基准下载 + 完整性 | – | ○ 把 Λ/κ 代码写完 |
| **P2 清洗与划分** | ● **主责**：标签调和 + 官方划分快照 | ○ 等你的 `data/processed/` | ● **主责**：指标与 Λ 计算器 |
| **P3 配置** | ● **主责**：`configs/tasks/` 冻结 | ● 写 `configs/arms/`、`configs/experiments/` | ○ 定义汇总表与图的数据契约 |
| **P4 代码** | ○ 写划分/泄漏测试 | ● **主责**：五个臂 | ● **主责**：Λ/κ + 规模与架构实验入口 |
| **P5 运行** | ○ 审计 | ● **主责**：GPU 批量（主扫 48 + OpenPoly 225） | ● **主责**：CPU 批量（scaling 100 + OpenPoly 225 + 架构 60） |
| **P6 图表与文档** | ○ 审计签字 | ○ 补跑与答疑 | ● **主责**：5 张图 + 主表 + 三条结论 |

● 主责　○ 参与/等待

**你在最前面，所以你是最容易被卡住的人，也是最容易卡住全队的人。** P0–P2 的 22.5 人时必须压住，否则后面两个人一起等。

---

## 2. 任务清单（拿什么 → 怎么做 → 验收标准）

> 每项都标了 **pH**（人时，含 agent 辅助）与**是否关键路径**。关键路径上的项拖一天，全队拖一天。

### A-1 · 三套 conda 环境装包并留指纹（3 pH｜关键）

| 项 | 内容 |
|:--|:--|
| **拿什么** | `environment/core.yml`、`environment/gnn.yml`、`environment/polybert.yml`（已写好版本，别改版本号） |
| **怎么做** | 逐个 `conda env create -f`；装完在各自环境里跑 `pip freeze`、记录 `python -V`、`nvidia-smi` 的驱动与 CUDA 版本、GPU 型号 |
| **产出** | `results/env_fingerprints/core.json` / `gnn.json` / `polybert.json`（**要入库**） |
| **验收标准** | ① 三个环境各自能 `import` 自己的核心包（core：`rdkit` `xgboost` `pysmiles`；gnn：`torch` `torch_geometric` `chemprop`；polybert：`transformers`）；② 指纹文件里版本号与 `.yml` 逐行一致；③ 旁人按 `.yml` 重建，`pip freeze` 与本机 diff 只剩平台相关包 |

> 【注意】 **为什么要留指纹**：同一份 `.yml` 在不同机器上装出来可能差几个传递依赖。结论要能复现，就得知道当时到底装了什么。这也是被问"你们的实验能重跑吗"时的证据。
>
> 【注意】 **装包时补一行**：配置与产物用 YAML（`configs/tasks/*.yaml`、`configs/arms/*.yaml`），但 `environment/core.yml` 目前**没有列 `pyyaml`**。装包时顺手加 `pyyaml==6.0.1`，并同步更新三个 `.yml` 里被它牵动的传递依赖；否则 P3 写配置的那一刻才会发现读不了 YAML，而那时 A 已经在做 P4 的事了。

### A-2 · PolyBench 39 个任务下载 + 完整性核对（3 pH｜关键）

| 项 | 内容 |
|:--|:--|
| **拿什么** | PolyBench 官方平台（39 个 CSV，稳定 ID = 数据集名，同时是 figshare 上的文件名；CC BY 4.0，DOI `10.6084/m9.figshare.30917717`） |
| **怎么做** | 写成 `scripts/00_download_polybench.py`：逐个下载 → 算 sha256 → 数行数 → 写 manifest。**不要手工点下载**，手工下载没法追 |
| **产出** | `data/raw/polybench/*.csv`（**不入库**）+ `data/manifests/polybench_manifest.csv`（**要入库**，列：`task_id, n_rows, sha256, source_url, downloaded_at`） |
| **验收标准** | ① manifest 恰好 39 行；② `n_rows` 合计 **42,169**；③ 与 Proposal **附录 A** 的 39 行逐行对齐（名字、条数都要对上）；④ 每个文件 sha256 可在 manifest 查到；⑤ `data/raw/` 不在 git 里（README 硬规则 1） |

> 【红线】 **别把 `n` 抄错**：附录 A 的条数是从平台 `/api/data-counts` 读的，是"记录数"不是"结构数"。两个数字在 PolyBench 里不等（14,717 个重复单元 vs 42,169 条记录），报告里不能说混。

### A-3 · Table S27 转录（**可选，不计入 32 pH 配额**｜3 pH｜非关键）

| 项 | 内容 |
|:--|:--|
| **拿什么** | `ps6c00094_si_001.pdf`（**156 MB**，md5 `fc7861f0839a03f601aaba4a6c4ce4a0`），其中 Table S27 = 39 行"任务 → 分层" |
| **怎么做** | 转录成 CSV；**只做验证，不做设计依据**。主文已给 Tier 大小 14/15/10 |
| **产出** | `data/manifests/polybench_tiers_s27.csv` |
| **验收标准** | ① 39 行齐全；② Tier 计数与主文 14/15/10 一致；③ **若与主文冲突，以主文为准**，把冲突任务替换为备用任务并记录日期与理由 |

> 【注意】 这一项**故意不放在关键路径上**：Proposal §3.1 只允许"主文点名了 tier 的"任务进主表，所以设计不依赖这份 PDF。你要是有余力就做，做不完不影响主表；但**不许假装读过**（§2.3 已经写明没读到，别在汇报里翻供）。
>
> 因为它是验证性的，**这 3 pH 不计入你的 32 pH 配额**。A-1 到 A-10 全部交验之前不要碰它。

### A-4 · 标签调和（7 pH｜关键，全队最重的单项之一）

这是你最重要的一项，也是最没法被 agent 加速的一项。

| 项 | 内容 |
|:--|:--|
| **拿什么** | A-2 下载的 39 个 CSV，共 42,169 条；重点是进主表的 6 个（Tg 8,055 / Tm 3,796 / 拉伸模量 1,034 / 气体渗透系数 711 / 介电损耗因子 279 / 等温失重时间 228） |
| **怎么做** | ① **单位判定**：温度 K vs °C、模量 GPa vs MPa、介电的频率与湿度条件——每一条判定都要有依据（列名、文献、数值量级），不允许"看着像"；② **去重**：42k 条里同名结构重复的要去重，去重键与规则写进决策表；③ **不可调和的冲突**：同一物理量用不同方法测（如 Tg 用 DSC vs DMA）→ **不换算，直接丢掉该任务并记录理由**；④ 全程脚本化，产出可重建 |
| **产出** | `data/interim/harmonised/<task>.parquet` + **`data/manifests/unit_decisions.csv`**（列：`property, raw_unit, target_unit, rule, n_rows_affected, evidence, decided_by, decided_at`） |
| **验收标准** | ① 每一条换算在决策表里有且仅有一行，`evidence` 列非空；② 每个主表任务调和后**可用标签 ≥ 200**（低于 200 触发备用任务替换）；③ 调和前后行数差 = 决策表 `n_rows_affected` 之和；④ 抽 20 行人工回查原始 CSV，全对 |

> 【红线】 **"不许猜"的具体含义**：遇到 `Tg = 500` 这种值是 K 还是 °C 分不清的，不许取"看起来合理"的那个。要么找到列名/单位字段的证据，要么整列弃用。**一个靠猜换来的任务，比少一个任务糟得多。**

### A-5 · 任务清单冻结 + 附录快照（2 pH｜关键）

| 项 | 内容 |
|:--|:--|
| **拿什么** | A-4 调和后的数据、Proposal §3.1 的 Table A–E |
| **怎么做** | 每个任务写一个 YAML：数据集名、目标性质、单位、`split_ref`、tier、source（experimental/DFT/MD）、可用标签数；再写一份带时间戳的冻结快照 |
| **产出** | `configs/tasks/{pb_t1_tg, pb_t1_tm, pb_t2_modulus, pb_t2_permeability, pb_t3_dielectric_loss, pb_t3_iwlt, pb26_polymetrix_tg, pb26_homopolymer_density}.yaml` + 3 个 OpenPoly + 2 个机制任务 + `docs/Task List Snapshot D0.md` |
| **验收标准** | ① 8 个主表任务配置齐全，**2 个/分层**（T1×2, T2×2, T3×2）+ PB26×2；② 每个 YAML 能被读取器校验，字段缺失/任务名拼错**当场报错**；③ 快照里写明冻结时刻与"此后任何替换都要记时间戳+理由"；④ 备用名单（5 个）也写进配置，标 `role: reserve` |

> 【口径】 **顺手锁定一处口径矛盾**：Proposal §6 写主表"8 tasks × **4** arms"，§8 写"8 tasks × **5** arms"（A1–A5 共五条臂，A5 是注入版）。**这处由你在 `configs/experiments/` 与 B 一起定死，写进快照，之后不许两种说法并存。**

### A-6 · 官方划分对齐 + 断言测试（4 pH｜关键）

| 项 | 内容 |
|:--|:--|
| **拿什么** | PolyBench 官方 70/30 划分索引（版本控制下发的索引文件）；PolyBench26 官方 5 折（base seed 42，val/test 各 10%）+ 官方 `scaling_splits` |
| **怎么做** | 把官方索引**原样快照**成只读文件；写 `tests/test_splits.py`：读快照 → 断言 train/test 索引集合与官方逐位一致、无交集、覆盖全量、fold 数与比例正确；再写一条**泄漏测试**：断言任何特征变换只 fit 在 train 上 |
| **产出** | `splits/polybench/<task>/official_70_30.json`、`splits/polybench26/<task>/fold_*.json`、`splits/polybench26/scaling/*`（**全部入库、只读**）+ `tests/test_splits.py` |
| **验收标准** | ① `pytest tests/test_splits.py` 全绿；② 划分文件带 sha256 与来源 URL；③ **任何代码不得重算划分**——写测试扫描 `src/` 里有没有 `train_test_split` 或 `KFold` 被用于评估（只允许出现在超参搜索里）；④ PolyBench 上**不出现 k 折评估**（官方只用固定 70/30） |

> 【红线】 这是课程红线。Proposal §7.5 把"agent 自己切数据"列为**头号失败模式**，而这类错误不抛异常。

### A-7 · 表征数据生成（PSMILES / wPSMILES / RDKit 描述符，1.5 pH｜关键）

| 项 | 内容 |
|:--|:--|
| **拿什么** | `data/interim/harmonised/` 的结构列（PSMILES）+ `environment/core.yml`（`rdkit==2024.3.5`、`pysmiles==1.1.2`） |
| **怎么做** | 三个产物：① PSMILES 规范化；② wPSMILES（带重复单元权重）；③ RDKit 描述符矩阵 + Morgan 指纹。全部按结构去重后算一次，再 join 回任务表 |
| **产出** | `data/processed/descriptors/<task>.parquet`、`data/processed/structures/psmiles.parquet` |
| **验收标准** | ① 每个主表任务一行一结构，join 后无意外行丢失（丢失数写进日志）；② 解析失败的结构**计数并列出**（RDKit 会给 warning，别让它静默吞掉）；③ 描述符列数、指纹位数记录在 manifest；④ 同一结构重复计算结果一致（缓存命中） |

### A-8 · polyBERT 嵌入（2 pH｜GPU 建议，CPU 可跑）

| 项 | 内容 |
|:--|:--|
| **拿什么** | 42k PSMILES（或去重后的结构集）+ `environment/polybert.yml`；需从 HuggingFace 取权重（缓存约 **7.3 GB**） |
| **怎么做** | 批量前向，输出定长向量；**用 `--limit` 支持小样本试跑**。GPU 0.5–1.5 h；**纯 CPU 约 6–10 h，可以挂着跑一晚** |
| **产出** | `data/processed/polybert/embeddings.npy` + `.../index.parquet`（行号 ↔ 结构 ID）+ manifest（维度、覆盖数、模型版本） |
| **验收标准** | ① 嵌入行数 = 输入结构数，**顺序对齐可校验**（抽 50 行用结构 ID 反查一致）；② ndim 与模型一致并记录；③ 网络断了可续跑（分片落盘） |

> 【注意】 这项 **CPU 也能做完**，只是慢。没有 GPU 就挂着过夜，不要把自己卡死在这儿，也不要把 `--limit 200` 的试跑结果当成全量交上来。

### A-9 · OpenPoly 三性质复核（1 pH）

| 项 | 内容 |
|:--|:--|
| **拿什么** | OpenPoly 发布 CSV（**741 行 / 741 个不同 PSMILES / 3,985 条结构-性质标注**） |
| **怎么做** | 直接数，不要读摘要。逐列统计非空数，重点核 Tg、Dielectric Constant Total、Tm |
| **产出** | `data/manifests/openpoly_manifest.csv`（列：`property, n_nonnull, usable_after_harmonisation`） |
| **验收标准** | ① 741 行、3,985 个标注、Tg **443**、介电常数 **295**、Tm **168** —— 五项与 §2.3 逐个对上；② 明确写下"**不要用** `final_polymer_property_counts_from_literature.csv`（Tg 58,755 是文献计数，不是可用样本）" |

### A-10 · 入口侧审计（5.0 pH｜关键）

全队审计共 14 pH，按"数据怎么进来的（A）/ 跑出来的记录对不对（B）/ 数字怎么出去的（C）"切成 **5 + 4 + 5**。你只管第一段，别把 C 的活也接过来——重复审计等于没人审计。

| 项 | 内容 |
|:--|:--|
| **拿什么** | 全队的提交，重点是任何"数据怎么进来"的代码（`data/`、`splits/`、`src/chainworth/io/`） |
| **怎么做** | 只查两类失败模式：**(i)** 自己切数据而不用官方划分；**(ii)** 全局统计量泄进特征。每条都要落到一个**能自动跑的断言**上，而不是"我看过了"。指标口径（第三类）归 C，运行记录与臂配置归 B |
| **产出** | `docs/Audit Log.md` 的入口侧章节（列：检查项、检查方式、结论、发现的问题、修复提交号）+ `tests/test_splits.py`、`tests/test_no_leakage.py` |
| **验收标准** | ① 两类模式各有对应测试，且**故意注入一个错误时测试会红**；② 每次合并到 `main` 前跑一遍，日志留痕；③ 审计发现的问题都有修复提交号 |

> 【注意】 审计是**新增成本**，不是浪费：Proposal §7.5 明确写了"agent 最常犯的三个错都不会报错，但都会让主表失效"。但它是三个人的事，不是一个人的事——你查入口，B 查 run 记录，C 查汇总与图。

---

## 3. 交付物清单（文件级）

```
configs/tasks/*.yaml                    8 主表 + 3 OpenPoly + 2 机制 + 5 备用
data/manifests/polybench_manifest.csv   39 行，合计 42,169
data/manifests/openpoly_manifest.csv    741 / 3,985 / 443 / 295 / 168
data/manifests/unit_decisions.csv       每条换算一行，带 evidence
data/manifests/polybench_tiers_s27.csv  可选（验证性）
data/processed/descriptors/*.parquet
data/processed/polybert/embeddings.npy
splits/polybench/<task>/official_70_30.json      只读
splits/polybench26/<task>/fold_*.json            只读
results/env_fingerprints/{core,gnn,polybert}.json
tests/test_splits.py  tests/test_no_leakage.py
docs/Task List Snapshot D0.md
docs/Audit Log.md
```

**不入库**：`data/raw/`、`data/interim/`、任何原始 CSV（README 硬规则 1：原始数据永不入库，仓库里只有指针）。

---

## 4. 没有 GPU 怎么办

你的活**绝大部分是 CPU**，做不完的只有一件：

| 任务 | GPU 需求 | 没卡的处置 |
|:--|:--|:--|
| A-1 环境 | 无（但 `gnn.yml`/`polybert.yml` 装了也跑不了） | 照装，指纹照留；跑不了不影响装包 |
| A-2–A-7、A-9、A-10 | **无** | 全做完 |
| **A-8 polyBERT 嵌入** | 建议 GPU（0.5–1.5 h）；**CPU 6–10 h** | 两条路任选：① 挂着过夜自己跑完；② 只交代码（`scripts/0X_embed_polybert.py` + 一条 `--limit 200` 的 smoke 结果），在 `RUN_REQUESTS.md` 里写明"全量 42k，预计 CPU 8 h / GPU 1 h"，由仓库所有者代跑 |

**交代码的卫生要求**（三条，缺一条就不算交接完成）：

1. 脚本必须有 `--limit` / `--dry-run`，能在 CPU 上 2 分钟内跑完一小片并打印形状；
2. 不写死 GPU：`device = "cuda" if torch.cuda.is_available() else "cpu"`，CPU 下能跑通（慢无所谓）；
3. 在 `RUN_REQUESTS.md` 里一行一条：run id、命令、预计耗时、期望产出文件路径、依赖哪个数据版本。

> 【红线】 **不许**手写任何结果文件充数，也不许把 `--limit 200` 的试跑产物改名成全量。占位请写 `{"status": "blocked_no_gpu", "requested_at": ..., "command": ...}`，让主表能区分"没跑"和"跑了没成"。

---

## 5. 依赖关系与冻结边界

**你依赖谁**：没人。你在最前面。
**谁依赖你**：B（描述符/嵌入/配置/划分）、C（`n`/单位/换算影响/scaling 档位）、仓库所有者（条数与证据，他自己去 `data/manifests/` 取，不需要你转交）。

**你冻结的东西**（一旦冻结，别人不得改，改要出报告）：

| 冻结项 | 冻结在 | 谁能改 |
|:--|:--|:--|
| 官方划分索引 | `splits/`（只读） | 谁都不能改；要换划分就出 `v2` 文件 |
| 任务清单（8 主表 + 备用） | `configs/tasks/` + `docs/Task List Snapshot D0.md` | 只有触发"任务 QA 不合格"才能按备用名单替换，且必须记时间戳+理由 |
| 单位换算规则 | `data/manifests/unit_decisions.csv` | 只能追加，不能覆盖；覆盖等于毁证据 |
| 三套环境版本 | `environment/*.yml` + `results/env_fingerprints/` | 只能升版本并全表重跑，不能"我这儿装了新版" |

---

## 6. 时间线与里程碑（以 8 个工作日 / 11-19 汇报为准）

| 时间 | 你要交的 | 卡点 |
|:--|:--|:--|
| **D0 18:00** | 三套环境装好 + A-2 下载完成 + A-7 数据生成**启动并抽检过**（不是"正在下"） | 硬门 |
| **D0 24:00** | A-6 划分快照 + `test_splits.py` 全绿；A-4 至少 2 个任务调和完 | 硬门 |
| **D1 12:00** | 8 个主表任务全部调和完 + `configs/tasks/` 冻结 + 快照落盘 | 主表能不能开工就看这个 |
| **D1 24:00** | A-5 全部完成；A-8 全量嵌入（自己跑或已交给 owner） | — |
| **P2 结束** | A-9、A-10 首轮审计 | — |
| **P3 结束** | 审计签字：主表每个数字都能追到 run 记录与数据版本 | 交付前 |

> 15 个工作日那套（11-26）把上面每一格整体后移约一周，P0 从 11/11–11/15。

---

## 7. 红线与常见坑

**红线（做了等于重做）**

1. **原始数据入库**：`data/raw/` 必须被 `.gitignore` 挡住。仓库里只放 URL、sha256、条数。
2. **自己切数据**：评估必须用官方划分。写在 `src/` 里的 `train_test_split`，只要参与评估就是红线。
3. **静默填补缺失**：调和时不许 `fillna(0)`、不许跨方法换算（DSC 的 Tg 不能换成 DMA 的 Tg）。
4. **靠猜定单位**：没有证据的换算，宁可丢任务。
5. **手改 manifest**：manifest 只能由脚本重生成，手改等于伪造。

**常见坑（都有人踩过）**

| 坑 | 现象 | 正确做法 |
|:--|:--|:--|
| PolyBench 用 k 折评估 | 结果看着更"稳"，但官方协议里不存在这个数 | 固定 70/30 + 3 个模型种子 + 残差 bootstrap；Murcko CV 只用于调参 |
| 把 `n=39` 的任务全当候选 | 主表膨胀、分层失衡 | 只允许"主文点名 tier"的任务进主表，其余 `pending S27` |
| OpenPoly 用文献计数 | Tg 报成 58,755，虚高 50 倍 | 只用 741 行那张表 |
| 描述符在全量上标准化 | 泄漏，R² 虚高 | 只在 train 上 fit，transform 到 val/test |
| 环境版本"差不多就行" | 换台机器结果差一点，查不出来 | 锁版本 + 留 fingerprint |

---

## 8. 交接前自检（逐条打勾再 push）

- [ ] `git status` 里没有 `data/raw/`、没有大文件、没有 `.ipynb_checkpoints`
- [ ] 三个环境指纹文件已入库，版本与 `.yml` 一致
- [ ] `polybench_manifest.csv` 39 行，条数合计 42,169，与附录 A 逐行对齐
- [ ] `pytest tests/test_splits.py tests/test_no_leakage.py` 全绿
- [ ] 8 个 `configs/tasks/*.yaml` 齐全，2 个/分层，字段校验通过
- [ ] `unit_decisions.csv` 每条都有 `evidence`，行数差能对上
- [ ] `docs/Task List Snapshot D0.md` 有冻结时间戳，且写明了 arm 数量的口径（4 还是 5）
- [ ] `docs/Audit Log.md` 的入口侧章节完成首轮，划分与泄漏两类各有测试（第三类"指标口径"归 C）
- [ ] `RUN_REQUESTS.md` 里没有 GPU 的项已写清命令与预计耗时
- [ ] 提交信息写清"做了什么 / 没做什么 / 下一步是什么"（README 第七节）

---

## 9. 工时口径核对（全队共用一张表）

**按人时不按百分比**：三个人各拿 **32.0 pH**；固定项（含审计）落在 28.0–30.0，极差 2.0 pH，剩下的差额放在各自的弹性余量里。

| 角色 | 硬项 | 审计 | 弹性余量 | **合计** |
|:--|--:|--:|--:|--:|
| **A 数据与合规（你）** | **23.5** | **5.0** | **3.5** | **32.0** |
| B 建模与训练 | 26.0 | 4.0 | 2.0 | 32.0 |
| C 评测与图表 | 23.0 | 5.0 | 4.0 | 32.0 |
| 合计 | 72.5 | 14.0 | 9.5 | **96.0** |

**你这 23.5 是怎么来的**：A-1 3.0 + A-2 3.0 + A-4 7.0 + A-5 2.0 + A-6 4.0 + A-7 1.5 + A-8 2.0 + A-9 1.0 = 23.5。A-3（S27 转录 3.0）不进配额，做与不做都不影响这张表。

> **总量为什么是 96 而不是 105**：原 105 pH 里有 9 pH 是写作与汇报（幻灯、Q&A、彩排）。这部分现在由仓库所有者自己承担，不进组员的分工池。105 − 9 = 96，96 ÷ 3 = 32.0。
>
> **和原 §7.1 逐项归属的差异**：B 原来一人扛 37.5（主表汇总 2.5 + scaling/OpenPoly 6.0 + 架构迁移 5.0 都在他手上），比另外两人重 40%。现在把这三块（合计 13.5 pH）移给 C——因为这三块产出的东西（Λ 的散点、κ 的斜率、H4 的交叉点）正是 C 自己判据的输入，"谁判谁跑"比"一个人跑、另一个人等"更快，也更少扯皮。
>
> **审计 14 pH 改成三分**：入口 5（A）+ 运行记录 4（B）+ 出口 5（C）。原先是 A 7 + C 7，B 不查自己落在 `results/raw_runs/` 的东西——那恰好是最容易出错的地方（device 写错、预测没落盘、config_hash 对不上）。
