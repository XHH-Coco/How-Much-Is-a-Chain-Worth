# 分工 A · 数据与合规（Data & Compliance）

> **对应**：`docs/Proposal_HowMuchIsAChainWorth_v1.3.md` §1–§3（数据与协议）、§7.1 的负责人列 `A`、§9 的 `A — Data & compliance`
> **工时**：固定项 **21.5 人时** + 弹性余量 **2.5** = **24.0 人时**
> **你负责的目录**：`data/`、`splits/`、`configs/tasks/`、`results/env_fingerprints/`
> **一句话**：让别人算出来的每一个数，都能追到某一行、某一次运行、某一条换算规则。
>
> **四人均分口径**：全队技术工时 96 pH（原 105 pH 减去写作与汇报的 9 pH，那 9 pH 由仓库所有者承担），四个人各 **24.0 pH**，固定项落在 21.5–22.0，极差 0.5 pH，差额是各自的弹性余量。

---

## 0. 先用大白话讲：你这份差事是干什么的

### 你要干的活，就是"配试剂 + 校准天平"

想象四个人要合写一篇实验报告（写报告本身由仓库所有者负责，不在四个人的分工里）。

- B 负责把需要显卡的那几台仪器跑起来（图神经网络、语言模型）；
- C 负责把不用显卡的那几台仪器跑起来，并把所有读数汇总成一张主表；
- D 负责把表上的数字算成结论、画成图，并审别人交来的东西；
- **你负责的是在他们动手之前，把试剂配好、把天平校准、把样品编号贴对。**

配试剂听起来最没意思，但它是唯一能让"我们的结果"和"一堆没法验证的数字"区分开的东西。举三个真实的翻车方式，全部**不会报错**：

1. 有人把某个任务的 Tg 当成摄氏度，其实文件里是开尔文。模型照样训练，R² 照样好看，只是那个任务的结论从根上就是错的。
2. 有人图省事自己 `train_test_split` 切了一刀。代码跑得通、结果很漂亮，但**官方划分**被绕过了——按课程红线，这张表直接作废。
3. 有人在算描述符时先对全量数据做了标准化，再切分。这叫把测试集信息泄漏进训练，误差会好看得不像话。

这三种情况，**一个异常都不会抛**。能拦住它们的只有两样东西：一份锁死的划分文件，和一套自动跑的断言测试。**这两样就是你的产出。**

### 你的东西被谁用、什么时候用

| 谁 | 用你的什么 | 什么时候 |
|:--|:--|:--|
| B（GPU 臂） | `data/processed/structures`、`descriptors`、`configs/tasks/*.yaml`、`splits/` | P1 一开工就要 |
| C（CPU 批量） | 同一批特征 + 超参可复用的任务配置 + 官方 `scaling_indices` 档位 | 跑 100 + 225 + 60 之前 |
| D（判据与图） | 每个任务的 `n`、单位、`split_ref`、以及"哪条换算规则影响了多少行" | 算 Λ 与 κ 时必须知道分母是什么 |
| 仓库所有者（写作与汇报，不占分工） | `data/manifests/` 里的条数与 sha256、单位换算决策表 | 被问"你们的数据可信吗"时 |

### 做完的标志（一句话）

> **另一个人拿到你的提交，只用 `git clone` + 两条命令，就能把 8 个主表任务的数据原样重建出来，且划分与官方逐位一致。**

---

## 1. 你在整条流水线上的位置

README 定的是**阶段串行**：上一阶段没验收，下一阶段不开工。**分工不是绕过阶段门**——同一个阶段内四个人可以同时干活，跨阶段必须等门。

| 阶段 | A（你） | B GPU 臂 | C CPU 批量 | D 判据与图 |
|:--|:--|:--|:--|:--|
| **P0 环境** | ● **主责**：三套环境装包锁版本 | ○ 预取 polyBERT 权重（7.3 GB） | ○ 写 A1/A2 的骨架 | ○ 用合成数据写指标单测 |
| **P1 获取数据** | ● **主责**：三基准下载 + 完整性 | – | ○ 写超参空间与预算表 | ○ 把 Λ/κ 代码写完 |
| **P2 清洗与划分** | ● **主责**：标签调和 + 官方划分快照 | ○ 等你的 `processed/`，先写装配 | ○ 等你的特征，先写实验入口 | ● **主责**：指标与 Λ 计算器 |
| **P3 配置** | ● **主责**：`configs/tasks/` 冻结 | ● `configs/arms/{a3,a4}.yaml` | ● `configs/arms/{a1,a2,a5}.yaml` + 两个实验配置 | ○ 定义汇总表与图的数据契约 |
| **P4 代码** | ○ 写划分/泄漏测试 | ● **主责**：A3 / A4 臂 | ● **主责**：A1 / A2 / A5 + 三批实验入口 | ● **主责**：Λ / κ / CI / 图 |
| **P5 运行** | ○ 交数据，接受审计 | ● **主责**：GPU 批量（48 + 225） | ● **主责**：CPU 批量（100 + 225 + 60） | ○ 等 run 记录 |
| **P6 图表与对账** | ○ 答疑与补数据 | ○ 补跑与答疑 | ● **主责**：主表 | ● **主责**：5 张图 + 三条结论 + 两端口审计 |

● 主责　○ 参与/等待

**你在最前面，所以你是最容易被卡住的人，也是最容易卡住全队的人。** P0–P2 的 21.5 人时必须压住，否则后面的 B、C、D 三个人一起等。好在你只做 CPU 的事，不用排队等卡，也不用等别人的产物——**这是全队唯一一段"完全靠自己就能推进"的时间，别浪费。**

---

## 2. 任务清单（拿什么 → 怎么做 → 验收标准）

> 每项都标了 **pH**（人时，含 agent 辅助）与**是否关键路径**。关键路径上的项拖一天，全队拖一天。
> **7 项固定合计 21.5 pH**（A-8 是可选验证项，不计入配额）。

### A-1 · 三套 conda 环境装包并留指纹（3.0 pH｜关键）

> **状态：已交付 2026-11-11**（仓库所有者代做）。凭据见 `docs/交付_P0_环境.md`。你的活变成"读一遍它、在本机把三条验收命令跑通"，不是重装。

| 项 | 内容 |
|:--|:--|
| **拿什么** | `environment/core.yml`、`environment/gnn.yml`、`environment/polybert.yml`（已冻结 v2，别改版本号）+ `lock/` 下六个锁文件 |
| **怎么做** | 装就按 `lock/`：`conda create -n <env> --file lock/<env>.conda.txt` 再 `pip install -r lock/<env>.pip.txt`；装完跑 `pip freeze`、`python -V`，与 `results/env_fingerprints/` 里那份对比 |
| **产出** | `results/env_fingerprints/env_core.json` / `env_gnn.json` / `env_polybert.json`（**已入库**） |
| **验收标准** | ① 三个环境各自能 `import` 自己的核心包（core：`rdkit` `xgboost` `pysmiles` `yaml` `pyarrow`；gnn：`torch` `torch_geometric` `chemprop`；polybert：`transformers`）；② 指纹里版本号与 `lock/` 逐行一致；③ `cd lock && sha256sum -c SHA256SUMS` 全 OK；④ 若你重装过，`pip freeze` 与指纹的 diff 只剩平台相关包 |

> 【注意】 **为什么要留指纹**：同一份 `.yml` 在不同机器上装出来可能差几个传递依赖。结论要能复现，就得知道当时到底装了什么。这也是被问"你们的实验能重跑吗"时的证据。
>
> 【注意】 **PyYAML 已补**：Core 原来漏了它（P3 读 `configs/**/*.yaml` 会当场失败），现在三个环境统一 `PyYAML==6.0.3`，已写进 `lock/core.pip.txt`。你若发现 `import yaml` 报错，说明装的是旧锁。
>
> 【注意】 **parquet 引擎已补（v2）**：三个环境都有 `pyarrow==17.0.0`。若 `to_parquet()` 报 `ImportError: Unable to find a usable engine`，说明装的是 v1 的锁，按 v2 重装即可——**不要就地 `pip install`**，那会让你手里的锁对别人失效。polybert 环境没有 pandas（只做编码），写 `index.parquet` 用 `pyarrow.Table` 直接构造。

### A-2 · 三基准下载 + 完整性核对（3.0 pH｜关键）

> **状态：已交付 2026-11-12**（仓库所有者代做，人工搬运 + 脚本离线验真）。**三个基准全部完成**：PolyBench26（63 文件 / 66.0 MiB，两个官方 sha256 逐位相等）、OpenPoly（741 行）、PolyBench（39 个 CSV / 5,937,499 B，**39/39 与 figshare 公布的 `supplied_md5` 逐个相等**）。`data/manifests/` 下三份清单齐备，PolyBench 那份 **39 行 / 条数合计 42,169 / 与 Proposal 附录 A 逐行对齐**。六条验收标准逐条见 `docs/交付_P1_A2_下载与完整性.md` **§9**。
>
> **figshare 403 仍未解**：本机出口被 `awselb/2.0` 拦截，39 个 CSV 是人工下载后由 `python scripts/00_download.py --verify-only` 离线核 md5 采纳的（`DECISIONS.md` D-13 / D-14）。**字节可证，传输过程无机器日志**——审计时以 md5 相等为凭。

> **口径**：Proposal §2.1 数的是**承重资源**，不是数据集个数——四个资源里承重的只有 PolyBench 与 PolyBench26，OpenPoly 仅边界，第四个（MoleculeNet / PCQM4Mv2）**不承重**。所以只下**三个**，第四个只在 `SOURCES.md` 登记、不下载、不建 manifest。若将来有人要用它，先改 `SOURCES.md` §3 + 记 `docs/DECISIONS.md`，再动手。

| 项 | 内容 |
|:--|:--|
| **拿什么** | [`data/manifests/SOURCES.md`](../data/manifests/SOURCES.md)——**入口与许可的唯一权威，`source_url` 只能取自这里**。三个基准：PolyBench 平台（39 个 CSV，稳定 ID = 数据集名；CC BY 4.0，DOI `10.6084/m9.figshare.30917717`）、PolyBench26（`rlearsch/PolymerBenchmark2026`）、OpenPoly（`WangGroupFDU/Openpoly_benchmark`） |
| **怎么做** | 写成 `scripts/00_download.py`：逐个下载 → 算 sha256 → 数行数 → 写 manifest。**不要手工点下载**，手工下载没法追。`version` 列对 git 仓库记 commit 短 sha，对 figshare 记存档版本号 |
| **产出** | `data/raw/**`（**不入库**）+ `data/manifests/{polybench,polybench26,openpoly}_download_manifest.csv`（**要入库**，列：`task_id, n_rows, sha256, source_url, version, downloaded_at`） |
| **验收标准** | ① PolyBench 那份恰好 39 行；② `n_rows` 合计 **42,169**；③ 与 Proposal **附录 A** 的 39 行逐行对齐（名字、条数都要对上）；④ 每个文件 sha256 可在 manifest 查到；⑤ `data/raw/` 不在 git 里（README 硬规则 1）；⑥ 每行 `source_url` 能在 `SOURCES.md` 找到、`version` 非空 |

> 【红线】 **别把 `n` 抄错**：附录 A 的条数是从平台 `/api/data-counts` 读的，是"记录数"不是"结构数"。两个数字在 PolyBench 里不等（14,717 个重复单元 vs 42,169 条记录），报告里不能说混。

### A-3 · 标签调和（7.0 pH｜关键，全队最重的单项之一）

> **状态：已交付 2026-11-12**（仓库所有者代做）。**6 个主表任务 + 4 个预备队任务**已调和入库，共 20,660 行；`bandgap chain_Egc` 因表头不写单位按红线整列弃用（3,380 行）；决策表 **30 行**，四条验收标准全过（`python scripts/01_harmonise.py --verify` 退出码 0，`pytest tests/test_harmonise.py` 17 项全绿），抽检 20 行与原始 CSV 逐项相同。凭据见 [`docs/交付_P1_A3_标签调和.md`](交付_P1_A3_标签调和.md)。
>
> **三条已验证、会改变后来做法的事实**（细节在该文档 §2–§3）：① 单位证据就在列名里，热学两任务已统一换到 **K**（`T + 273.15`），其余四类按发布单位原样保留并显式记为 identity；② `property_log` 是**自然对数**不是 log10（逐行等于 `ln(label)`，残差 5e−10），别拿它当对数目标；③ 去重键 `(task, 规范 SMILES)` 实测**删 0 行**——每个任务内部本来就无重复，但跨任务有 9,166 行共享同一结构，那是合法数据。
>
> **你的下一步**：A-4 冻结任务清单时，`unit` 字段直接抄 `unit_decisions.csv` 的 `target_unit`，`n` 抄 `harmonised_manifest.csv` 的 `n_labels`；**PolyBench26 的 2 个任务与 OpenPoly 的 3 个性质还没调和**，要用它们之前先补交付文档 §6 第 2 条。

这是你最重要的一项，也是最没法被 agent 加速的一项。

| 项 | 内容 |
|:--|:--|
| **拿什么** | A-2 下载的 39 个 CSV，共 42,169 条；重点是进主表的 6 个（Tg 8,055 / Tm 3,796 / 拉伸模量 1,034 / 气体渗透系数 711 / 介电损耗因子 279 / 等温失重时间 228） |
| **怎么做** | ① **单位判定**：温度 K vs °C、模量 GPa vs MPa、介电的频率与湿度条件——每一条判定都要有依据（列名、文献、数值量级），不允许"看着像"；② **去重**：42k 条里同名结构重复的要去重，去重键与规则写进决策表；③ **不可调和的冲突**：同一物理量用不同方法测（如 Tg 用 DSC vs DMA）→ **不换算，直接丢掉该任务并记录理由**；④ 全程脚本化，产出可重建 |
| **产出** | `data/interim/harmonised/<task>.parquet` + **`data/manifests/unit_decisions.csv`**（列：`property, raw_unit, target_unit, rule, n_rows_affected, evidence, decided_by, decided_at`） |
| **验收标准** | ① 每一条换算在决策表里有且仅有一行，`evidence` 列非空；② 每个主表任务调和后**可用标签 ≥ 200**（低于 200 触发备用任务替换）；③ 调和前后行数差 = 决策表 `n_rows_affected` 之和；④ 抽 20 行人工回查原始 CSV，全对 |

> 【红线】 **"不许猜"的具体含义**：遇到 `Tg = 500` 这种值是 K 还是 °C 分不清的，不许取"看起来合理"的那个。要么找到列名/单位字段的证据，要么整列弃用。**一个靠猜换来的任务，比少一个任务糟得多。**

### A-4 · 任务清单冻结 + 附录快照（2.0 pH｜关键）

> **状态：已交付 2026-11-12**（仓库所有者代做）。**18 份配置**落 `configs/tasks/`：8 主表（PolyBench 每层 2 个 + PolyBench26 2 个）+ 5 备用 + 3 OpenPoly 边界 + 2 机制实验；四项验收标准全过（`PYTHONPATH=src python -m chainworth.io.tasks --verify` 退出码 0，`pytest tests/test_task_configs.py` **15 项全绿**，`cd configs/tasks && sha256sum -c SHA256SUMS` 18 个 OK）。快照 [`docs/Task List Snapshot D0.md`](Task%20List%20Snapshot%20D0.md) 冻结于 **2026-11-12T03:48:00+08:00**。凭据见 [`docs/交付_P1_A4_任务清单冻结.md`](交付_P1_A4_任务清单冻结.md)。
>
> **三条已验证、会改变后来做法的事实**（细节在该文档 §3–§4）：① **Q-01 结案 = 主表「8 任务 × 5 臂」**（D-17）——按官方评估单元每臂 48 次，§6 的"96 经典 + 48 GNN = 144"只等于 **3 臂**，§7.4 的 150 = 144 + 6 次无出处，所以**拟合次数仍归 Q-02（C 定）**，快照里不写任何"192/240"；② **PolyMetriX Tg 本来就是开尔文**（列名 `Tg_K`、7,367 行 0 缺失、134.15–768.15 K），A-3 文档 §7 第 4 条那句"也必须加 273.15"是预设、已勘误（D-18）；③ **密度单位官方三处文档均未声明**，故 `unit_status: unverified` 且 `target_unit: null`（D-19），机制实验两个任务用第三种状态 `not_required`。
>
> **你的下一步（A-5）**：18 行的 `split_ref` 就是 A-5 的落盘清单（PolyBench 11 个 / PB26 2 个 / OpenPoly 3 个 / 机制 2 个，路径逐条写在 YAML 里）。**落盘后把 `split_status` 由 `pending` 改成 `present` 并重出 `SHA256SUMS`**——只改状态不重出指纹，等于冻结物被改而不留痕。（✅ 2026-11-12 已按此执行：17 份改了状态、`SHA256SUMS` 出到 v2，见下方 A-5 状态块。）

| 项 | 内容 |
|:--|:--|
| **拿什么** | A-3 调和后的数据、Proposal §3.1 的 Table A–E |
| **怎么做** | 每个任务写一个 YAML：数据集名、目标性质、单位、`split_ref`、tier、source（experimental/DFT/MD）、可用标签数；再写一份带时间戳的冻结快照 |
| **产出** | `configs/tasks/{pb_t1_tg, pb_t1_tm, pb_t2_modulus, pb_t2_permeability, pb_t3_dielectric_loss, pb_t3_iwlt, pb26_polymetrix_tg, pb26_homopolymer_density}.yaml` + 3 个 OpenPoly + 2 个机制任务 + `docs/Task List Snapshot D0.md` |
| **验收标准** | ① 8 个主表任务配置齐全，**2 个/分层**（T1×2, T2×2, T3×2）+ PB26×2；② 每个 YAML 能被读取器校验，字段缺失/任务名拼错**当场报错**；③ 快照里写明冻结时刻与"此后任何替换都要记时间戳+理由"；④ 备用名单（5 个）也写进配置，标 `role: reserve` |

> 【口径】 **顺手锁定一处口径矛盾**：Proposal §6 写主表"8 tasks × **4** arms"，§8 写"8 tasks × **5** arms"（A1–A5 共五条臂，A5 是注入版）。**这处由你牵头、C 与 D 会签，写进快照，之后不许两种说法并存。**

### A-5 · 官方划分对齐 + 断言测试（4.0 pH｜关键）

> **状态：已交付 2026-11-12**（仓库所有者代做）。**14 份快照**落 `splits/`：PolyBench **10 份**行号 JSON（6 主表 + 4 备用）、PolyBench26 **4 份**官方 `seed_42/` 原样拷贝（2 主表 + 机制 Cp 的 2 个数据集）；OpenPoly **按红线停住**，只落一份"官方没发布划分"的结论。八条验收标准：**7 条通过、第 ⑤ 条明确不通过且原因在上游**。凭据见 [`docs/交付_P1_A5_官方划分快照.md`](交付_P1_A5_官方划分快照.md)。
>
> **四条已验证、会改变后来做法的事实**（细节在该文档 §2–§4）：① **PolyBench 的官方划分只有一份固定的 70/30 测试集，没有 seed 维度**（`&seed=0/1/2` 返回字节完全相同），`eval_unit` 里的"×3"是**模型种子**不是三份划分（D-20）；② **canonical PSMILES 表要本地物化**，且必须写成 LF——Windows 下 pandas 默认 CRLF，官方 `manage_scaling_indices.py verify` 会直接报 checksum mismatch，归一后与官方 `canonical_sha256` 逐位相等（D-21）；③ **OpenPoly 的 `.pt` 划分没发布**，notebook 只读不切，自己按 8:1:1 重切等于踩头号红线（D-22）；④ **划分状态变四态**：`present` / `blocked_upstream` / `deferred` / `pending`（后者只允许不可用任务）（D-23）。
>
> **你的下一步（A-6）**：结构产物（PSMILES / wPSMILES / 描述符）**不入库**，交付的是"能重建它们的脚本 + manifest"。注意 `data/raw/polybench26/Datasets/PSMILES/` 现在有 A-5 物化出来的 canonical 表（49 张），**可以复用它**，但不要把"本地物化"当成"官方发布"——两类文件在 `PROVENANCE` 上不一样。


| 项 | 内容 |
|:--|:--|
| **拿什么** | ① PolyBench 官方 70/30 划分索引（版本控制下发的索引文件）；② PolyBench26 的 **`Datasets/scaling_indices/`**（**不是** `scaling_splits/`——那份被官方 gitignore、仓库里没有实体，见 `SOURCES.md` §2.2【更正】）；③ OpenPoly 发布仓库里的 `train_and_test_code/` 划分。**`task3_complexity/` 与 `task4_architecture_transfer/` 不快照**：前者是事后分层不是留出任务，后者官方只发布物化脚本（`SOURCES.md`【坑八】【坑七】） |
| **怎么做** | **四步，顺序不能换**：<br>① 跑官方 `Datasets/Dataset_construction_scripts/generate_basic_datasets.sh` 生成 **canonical PSMILES 表**——官方不发布这张表，只发布 compact 源（【坑六】）；<br>② 用 `scaling_indices/<实验>/seed_42/manifest.json` 里的 **`canonical_sha256`** 校验这张表，**不通过就停**（行号会指向别的分子，且**不报错**，只会算出一组错的 R²）；<br>③ 把主表两个实验的 `seed_42/`（`manifest.json` + `fold_00..04.npz`）**原样拷**进 `splits/`，跑官方 `manage_scaling_indices.py verify --indices-root splits/polybench26 --datasets-root <canonical 表所在> ` 自检；<br>④ 写 `tests/test_splits.py`：断言索引集合与官方逐位一致、无交集、覆盖全量、fold 数与比例正确、**`train_pool[:N]` 确为嵌套前缀**（官方 verify 不查这条，只写在它的 `export` 分支里），再写一条**泄漏测试**：断言任何特征变换只 fit 在 train 上 |
| **产出** | `splits/polybench/<task>/official_70_30.json`、`splits/polybench26/<task>/seed_42/{manifest.json, fold_00..04.npz}`（**原样拷贝，不改字节**）、`splits/polybench26/README.md`（记仓库 commit、源文件 sha256、canonical sha256、校验命令）、`splits/openpoly/<task>_official.json`（**全部入库、只读**）+ `tests/test_splits.py`。**`splits/polybench26/arch/` 留空并注明"本地物化、不入库"** |
| **验收标准** | ① `pytest tests/test_splits.py` 全绿；② 划分文件带 sha256 与来源 URL，且 `splits/polybench26/README.md` 里能追到仓库 commit；③ **任何代码不得重算划分**——写测试扫描 `src/` 里有没有 `train_test_split` 或 `KFold` 被用于评估（只允许出现在超参搜索里）；④ PolyBench 上**不出现 k 折评估**（官方只用固定 70/30）；⑤ OpenPoly 那份直接快照仓库 `train_and_test_code/` 的划分结果，test 比例与官方一致，不得自己重切；⑥ **canonical PSMILES 表的 sha256 与 `manifest.json` 的 `canonical_sha256` 相等**（这条不通过，后面全作废）；⑦ **`train_pool[:N]` 前缀断言**：对每个档位 N 断言 `array_equal(materialized[:N], train_pool[:N])`——这是 B、C 共用的不变量，官方 verify 兜不住；⑧ `manage_scaling_indices.py verify` 在部分快照上跑通（缺 canonical 源时加 `--allow-missing-sources`） |

> 【红线】 这是课程红线。Proposal §7.5 把"agent 自己切数据"列为**头号失败模式**，而这类错误不抛异常。
>
> 【红线·发布物更正】 Proposal §2.1 与 §7.5 原文写的是"必须使用官方 `scaling_splits`"，**这条按字面做不到**：`Datasets/scaling_splits/` 被官方 gitignore，只存在于跑过生成脚本的机器上。等价的官方发布物是 **`Datasets/scaling_indices/`**。**判定标准一个字没变**——"划分必须可追到官方发布物，不得由我们的代码决定谁进训练集"；变的只是取哪个文件。这条也解释了为什么 ①（生成 canonical 表）不能省：索引存的是**行号**，行号只在那张表的特定顺序下指向正确分子，换版本不报错、只出错结果。
>
> 【接口】 **D 会拿这条测试来复核你**（入口审计归 D，见 `分工_D`）。他复核的方式不是读你的代码，而是往 `src/` 里故意注入一次 `train_test_split`，看你的测试会不会红。所以"测试写了"不够，**得是能抓住真错的测试**。
>
> 【工时】 OpenPoly 那份快照**计入本项 4.0 pH，不单列任务项**（约 0.3 pH 的量）：它的划分随发布仓库下发，不需要向平台申请，也不做断言以外的加工。加这一项是因为 README 第九节第 6 行把 `splits/openpoly/` 也算进了"A 的官方划分快照"，而原来只有 PolyBench 与 PolyBench26 两项——目录留着没人填，C 的 OpenPoly 经典臂就没有划分可用。
>
> 【工时·新增工序】 上面 ①②⑦ 三步是 2026-11-11 深夜才定下来的（canonical PSMILES 生成、`canonical_sha256` 校验、前缀断言），原条目没有，合计约 **0.5–0.75 pH**。**A-5 仍是 4.0 pH、固定项仍是 21.5**：这笔从你的 **2.5 弹性余量**里出，余量降到 1.75–2.0。不调 A 的总工时，也不动 96 pH 口径——因为这三步不是"新增任务"，是"原来那条工序漏掉的前置条件"，不补就得不出可信划分。

### A-6 · 表征数据生成：PSMILES / wPSMILES / RDKit 描述符（1.5 pH｜关键）

| 项 | 内容 |
|:--|:--|
| **拿什么** | `data/interim/harmonised/` 的结构列（PSMILES）+ `environment/core.yml`（`rdkit`、`pysmiles`） |
| **怎么做** | 三个产物：① PSMILES 规范化；② wPSMILES（带重复单元权重）；③ RDKit 描述符矩阵 + Morgan 指纹。全部按结构去重后算一次，再 join 回任务表 |
| **产出** | `data/processed/structures/psmiles.parquet`、`data/processed/descriptors/<task>.parquet`（**均不入库**，见下）+ 生成脚本 `scripts/02_features.py` + 一份记录列数与失败结构的 manifest |
| **验收标准** | ① 每个主表任务一行一结构，join 后无意外行丢失（丢失数写进日志）；② 解析失败的结构**计数并列出 ID**（RDKit 会给 warning，别让它静默吞掉）；③ 描述符列数、指纹位数记录在 manifest；④ 同一结构重复计算结果一致（缓存命中） |

> 【边界】 你只保证"每个结构算对一次"。**"每个任务的每一行对上正确的结构"是 B 的事**（B-1 用 `row_id` join，并断言行序由 split 文件决定）。别替他做装配，否则他的验收标准就没法独立成立了。
>
> 【交付方式】 这两个 parquet **不入库**：`.gitignore` 把 `data/processed/*` 整目录忽略，README §4.3 也是这个口径。所以你交的不是数据文件，而是**"能重建它们的脚本 + manifest"**——B 拿到提交后跑一次 `python scripts/02_features.py` 就得到同一批文件。**交付时在 `#data` 里明确写这一句**，不要只说"已生成"，否则 B 会一直等一个永远不会出现在仓库里的文件。

### A-7 · OpenPoly 三性质复核（1.0 pH）

| 项 | 内容 |
|:--|:--|
| **拿什么** | OpenPoly 发布 CSV（**741 行 / 741 个不同 PSMILES / 3,985 条结构-性质标注**） |
| **怎么做** | 直接数，不要读摘要。逐列统计非空数，重点核 Tg、Dielectric Constant Total、Tm |
| **产出** | `data/manifests/openpoly_property_counts.csv`（列：`property, n_nonnull, usable_after_harmonisation`）**——不要叫 `openpoly_manifest.csv`**：那个名字归 A-2 的下载清单（`openpoly_download_manifest.csv`），两份东西列完全不同，同名会互相覆盖 |
| **验收标准** | ① 741 行、3,985 个标注、Tg **443**、介电常数 **295**、Tm **168** —— 五项与 §2.3 逐个对上；② 明确写下"**不要用** `final_polymer_property_counts_from_literature.csv`（Tg 58,755 是文献计数，不是可用样本）" |

> 【为什么归你】 这不是"顺手数一下"：C 的 OpenPoly 经典臂要在 n ≤ 443 的网格上跑，D 的 H4 要在同一批样本上求过零点。**分母错了，他们俩的数一起错。** 数完在 `#data` 里报一声，别让他们自己去猜。

### A-8 · Table S27 转录（**可选，不计入 24 pH 配额**｜3.0 pH｜非关键）

| 项 | 内容 |
|:--|:--|
| **拿什么** | `ps6c00094_si_001.pdf`（**156 MB**，md5 `fc7861f0839a03f601aaba4a6c4ce4a0`），其中 Table S27 = 39 行"任务 → 分层" |
| **怎么做** | 转录成 CSV；**只做验证，不做设计依据**。主文已给 Tier 大小 14/15/10 |
| **产出** | `data/external/table_s27.csv`（**不入库**，与 README §4.2 的 `data/external/` 定名一致；它是从 PDF 转录出来的第三方数据，不是我们生成的清单，所以不放 `manifests/`） |
| **验收标准** | ① 39 行齐全；② Tier 计数与主文 14/15/10 一致；③ **若与主文冲突，以主文为准**，把冲突任务替换为备用任务并记录日期与理由 |

> 【注意】 这一项**故意不放在关键路径上**：Proposal §3.1 只允许"主文点名了 tier 的"任务进主表，所以设计不依赖这份 PDF。你要是有余力就做，做不完不影响主表；但**不许假装读过**（§2.3 已经写明没读到，别在汇报里翻供）。
>
> 因为它是验证性的，**这 3.0 pH 不计入你的 24 pH 配额**。A-1 到 A-7 全部交验之前不要碰它。

---

## 3. 交付物清单（文件级）

```
【入库】
configs/tasks/*.yaml                            8 主表 + 3 OpenPoly + 2 机制 + 5 备用（共 18 份，已冻结 v1）
configs/tasks/SHA256SUMS                        上面 18 份的指纹（`cd configs/tasks && sha256sum -c`）
data/manifests/SOURCES.md                       四个资源入口 + 许可（source_url 的唯一来源）
data/manifests/polybench_download_manifest.csv  39 行，合计 42,169
data/manifests/polybench26_download_manifest.csv
data/manifests/openpoly_download_manifest.csv
data/manifests/openpoly_property_counts.csv     741 / 3,985 / 443 / 295 / 168
data/manifests/unit_decisions.csv               每条换算一行，带 evidence
splits/polybench/<task>/official_70_30.json     只读
splits/polybench26/<task>/seed_42/manifest.json 官方原样拷贝，不改字节
splits/polybench26/<task>/seed_42/fold_0[0-4].npz  官方原样拷贝，不改字节
splits/polybench26/README.md                    来源 commit + 三组 sha256 + 校验命令
splits/openpoly/<task>_official.json            只读
results/env_fingerprints/env_{core,gnn,polybert}.json
tests/test_splits.py  tests/test_no_leakage.py
docs/Task List Snapshot D0.md

【不入库，靠脚本重建——交的是脚本 + manifest，不是文件】
data/raw/**                                     原样下载        scripts/00_download.py
data/interim/harmonised/<task>.parquet          标签调和        scripts/01_harmonise.py
data/processed/structures/psmiles.parquet       结构规范化      scripts/02_features.py
data/processed/descriptors/<task>.parquet       描述符与指纹    scripts/02_features.py
data/external/table_s27.csv                     可选验证性转录   （A-8，手工转录）
```

**不入库的理由不是"文件大"，是硬规则 1：数据本体永不入库，仓库里只有指针。** B/C/D 拿到你的提交后跑对应脚本重建——所以脚本必须可重跑（`--dry-run` / `--limit`）、manifest 必须能证明"重出来的和原来一致"（sha256 与条数）。**交付时写清"跑哪条命令重建"，不要只说"已生成"。**

---

## 4. 没有 GPU 怎么办

**你不需要 GPU，一项都不需要。** 这是四个角色里唯一"有台笔记本就能独立完成"的一个：

| 任务 | GPU | 说明 |
|:--|:--|:--|
| A-1 环境 | 无 | `gnn.yml`/`polybert.yml` 装了跑不了也不影响你，指纹照留 |
| A-2 下载 | 无 | 只受网速影响；断点续传写好 |
| A-3 标签调和 | 无 | pandas 级别，这是你最需要专注力的地方 |
| A-4–A-7 | 无 | 纯 CPU |

所以你没有"等卡"这个借口，也没有"等别人"这个借口——**P0–P2 的进度只取决于你自己。** 这也意味着：如果你慢了，全队都在等你，而不是在等机器。

> 【注意】 polyBERT 嵌入（原属 A 的 A-8，2.0 pH）**已经划给 B**，因为它和 A4 臂共用同一套权重缓存与环境。你不用管那 7.3 GB 的下载。

---

## 5. 依赖关系与冻结边界

**你依赖谁**：没人。你在最前面。
**谁依赖你**：B（结构、描述符、配置、划分）、C（特征、`n`、官方 scaling 档位）、D（`n`/单位/换算影响，算 Λ 的分母）、仓库所有者（条数与证据，他自己去 `data/manifests/` 取）。

| 冻结项 | 冻结在 | 谁能改 |
|:--|:--|:--|
| 官方划分索引 | `splits/`（只读） | 谁都不能改；要换划分就出 `v2` 文件 |
| 任务清单（8 主表 + 备用） | `configs/tasks/` + `docs/Task List Snapshot D0.md` | 只有触发"任务 QA 不合格"才能按备用名单替换，且必须记时间戳+理由 |
| 单位换算规则 | `data/manifests/unit_decisions.csv` | 只能追加，不能覆盖；覆盖等于毁证据 |
| 三套环境版本 | `environment/*.yml` + `lock/` + `results/env_fingerprints/` | **已冻结 v2（2026-11-11，v2 = v1 + pyarrow）**；只能升版本并全表重跑，不能"我这儿装了新版" |

> **全队只有一张冻结登记表**：`README` 第九节（版本号、冻结日期、怎么验、当前状态）。本节只列与你直接相关的那几项；口径冲突时以 README 为准。冻结物被改动 = 该阶段重做，不是"改一下就好"。

---

## 6. 时间线与里程碑（以 8 个工作日 / 11-19 汇报为准）

| 时间 | 你要交的 | 卡点 |
|:--|:--|:--|
| **D0 18:00** | 三套环境装好 + A-2 下载完成 + A-6 数据生成**启动并抽检过**（不是"正在下"） | 硬门（环境这半已于 11-11 提前交付） |
| **D0 24:00** | A-5 划分快照 + `test_splits.py` 全绿；A-3 至少 2 个任务调和完 | 硬门 |
| **D1 12:00** | 8 个主表任务全部调和完 + `configs/tasks/` 冻结 + 快照落盘 | 主表能不能开工就看这个 |
| **D1 24:00** | A-4、A-6 全部完成；结构级产物交 B | — |
| **P2 结束** | A-7 复核数交 C 与 D | 他们的分母 |
| **P3 结束** | 配合 D 的入口审计：注入错误时测试必须红 | 交付前 |

> 15 个工作日那套（11-26）把上面每一格整体后移约一周，P0 从 11/11–11/15。

---

## 7. 红线与常见坑

**红线（做了等于重做）**

1. **原始数据入库**：`data/raw/` 必须被 `.gitignore` 挡住。仓库里只放 URL、sha256、条数。
2. **自己切数据**：评估必须用官方划分。写在 `src/` 里的 `train_test_split`，只要参与评估就是红线。
3. **静默填补缺失**：调和时不许 `fillna(0)`、不许跨方法换算（DSC 的 Tg 不能换成 DMA 的 Tg）。
4. **靠猜定单位**：没有证据的换算，宁可丢任务。
5. **手改 manifest**：manifest 只能由脚本重生成，手改等于伪造。

**代码怎么写**：P4 起所有提交的代码按 [`docs/CODE_STYLE.md`](CODE_STYLE.md)——中英双语注释，每个函数写清输入输出、数据来源、配置来源，数字必须有来源。审阅时不看风格，只看"能不能从注释追到数据来源"；追不到的退回重写。

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
- [ ] 三个环境指纹已入库，版本与 `lock/` 逐行一致；`cd lock && sha256sum -c SHA256SUMS` 全 OK
- [ ] `polybench_download_manifest.csv` 39 行，条数合计 42,169，与附录 A 逐行对齐；每行 `source_url` 能在 `SOURCES.md` 找到、`version` 非空
- [ ] `pytest tests/test_splits.py tests/test_no_leakage.py` 全绿，且**故意注入错误时会红**；`splits/openpoly/` 里已有官方划分快照
- [ ] canonical PSMILES 表的 sha256 与 `splits/polybench26/*/seed_42/manifest.json` 的 `canonical_sha256` 相等；`manage_scaling_indices.py verify` 通过；`train_pool[:N]` 前缀断言有单独测试且能抓错
- [ ] 8 个 `configs/tasks/*.yaml` 齐全，2 个/分层，字段校验通过 ｜ **已过（18 份；`PYTHONPATH=src python -m chainworth.io.tasks --verify`）**
- [ ] `unit_decisions.csv` 每条都有 `evidence`，行数差能对上
- [ ] `docs/Task List Snapshot D0.md` 有冻结时间戳，且写明了 arm 数量的口径（4 还是 5）｜ **已过（5 臂，D-17）**
- [ ] 结构级产物（structures / descriptors）**不入库**，已确认 `python scripts/02_features.py` 能从 `data/interim/` 重建，并在 `#data` 里写明重建命令（不是"已生成"）
- [ ] OpenPoly 五个数（741 / 3,985 / 443 / 295 / 168）已公布
- [ ] `git status` 里没有 agent 自建目录（`.workbuddy/`、`.tools/` 等）或 `scripts/_scratch/`；有就自己补进 `.gitignore`（README 第五节硬规则 4）
- [ ] 提交信息写清"做了什么 / 没做什么 / 下一步是什么"（README 第十节）

---

## 9. 工时口径核对（全队共用一张表）

**按人时不按百分比**：四个人各拿 **24.0 pH**；固定项落在 21.5–22.0，**极差 0.5 pH**，剩下的差额放在各自的弹性余量里。

| 角色 | 固定项 | 弹性余量 | **合计** |
|:--|--:|--:|--:|
| **A 数据与合规（你）** | **21.5** | **2.5** | **24.0** |
| B 表示与 GPU 臂 | 21.5 | 2.5 | 24.0 |
| C 经典臂、机制实验与主表 | 22.0 | 2.0 | 24.0 |
| D 评测、图表与对账 | 21.5 | 2.5 | 24.0 |
| 合计 | 86.5 | 9.5 | **96.0** |

**你这 21.5 是怎么来的**：A-1 3.0 + A-2 3.0 + A-3 7.0 + A-4 2.0 + A-5 4.0 + A-6 1.5 + A-7 1.0 = 21.5。A-8（S27 转录 3.0）不进配额。

> **总量为什么是 96 而不是 105**：原 105 pH 里有 9 pH 是写作与汇报（幻灯 7 + 余量 2）。这部分现在由仓库所有者自己承担，不进组员的分工池。105 − 9 = 96，96 ÷ 4 = **24.0**。
>
> **为什么本版不再有"审计"这一栏**：四人制下审计按"**谁的数字依赖它，谁来查**"分配——入口（划分与泄漏）归 D，因为 Λ 的分母就是 `n` 和 `split_ref`；运行记录归 C，因为他跑出了全队 3/4 的 fit；出口（汇总与图）归 D。你不再自审入口，**改由 D 拿你 A-5 的测试去抓错**——自己审自己等于没审，这一条比多背 5 pH 重要。
>
> **你不能省的那一项**：A-3 的 7.0 pH。它是全队唯一"agent 帮不上忙"的活（换算证据要人看），也是唯一能让 42,169 条记录变成"可用标签"的一步。想快一点的办法不是跳步，是**先做进主表的 6 个任务**，其余 33 个留到主表跑起来之后再补。
