# 基准来源登记（SOURCES） / Benchmark Sources

> **这一份文件是数据入口的唯一权威。** `*_download_manifest.csv` 里的 `source_url` 和 `version` 两列，只能取自本文件；发现本文件里没有的入口，先在这里登记（带核实时点与方式），再去下载。
>
> 为什么单独一份：`Proposal` 只说到"平台上的 CSV 文件名"和 DOI，没写可点击的地址；而 `data/raw/` 不入库、`data/manifests/` 里的 manifest 又是别人重建的唯一线索。**入口不落地，manifest 就是无源之水**——这正是分工_A A-3 红线「不许猜」要拦的东西。
>
> 维护人：分工 A。改动走 `docs/DECISIONS.md` 记一条。

---

## 1. 四个资源，三种角色

Proposal §2.1 的结论是「四个资源，其中两个承载结果」。A-2 只下**三个**，第四个只登记不复现——理由在 §3。

| 资源 | 角色 | 承载 | 本次是否下载 | 入口 |
|---|---|---|---|---|
| **PolyBench** | 承重 | H1 | ✅ 下（39 个 CSV） | <http://polybench.ciac.jl.cn>（备用 <http://202.98.16.11>） |
| **PolyBench26** | 承重 | H2、H1 架构子句 | ✅ 下 | <https://github.com/rlearsch/PolymerBenchmark2026> |
| **OpenPoly** | 仅边界 | H4 | ✅ 下 | <https://github.com/WangGroupFDU/Openpoly_benchmark> |
| MoleculeNet / PCQM4Mv2 | 不承重 | 无（仅背景） | ❌ 不下，仅登记 | <https://moleculenet.org> / <https://ogb.stanford.edu/docs/l2/pcqm4mv2/> |

---

## 2. 逐个登记

### 2.1 PolyBench（承重 · H1）

| 项 | 内容 |
|:--|:--|
| 入口 | 平台 <http://polybench.ciac.jl.cn>，论文给出的备用地址 <http://202.98.16.11> |
| 规模 | 39 个任务 / **42,169 条记录** / 14,717 个重复单元 / 7 类表征 / 30 个基线模型 |
| 论文 | *PolyBench: A Reproducible Open Benchmark Platform for Repeat-Unit-Based Polymer Property Prediction*，Polym. Sci. Technol. **2026**, 2(10), 836–854，DOI `10.1021/polymscitech.6c00094`，2026-08-31 上线（开放获取） |
| 存档 | figshare，DOI `10.6084/m9.figshare.30917717`，**CC BY 4.0**；39 个 CSV 的文件名 = 数据集名（PolyBench 不发数字任务 ID） |
| `version` 取值 | figshare 存档版本号（`v1`/`v2`…）；若从平台直取，记抓取日期 + 平台页脚版本串 |
| 核实时点 | URL、规模、论文元数据：**2026-11-11 公开检索核实**。figshare DOI：转录自 Proposal §2，**本次未重新解析** |
| 待确认（A 在 D0 前做） | ① 平台是否需要注册/登录才能批量取 CSV；② `/api/data-counts` 是否公开（附录 A 的 39 个条数即从此接口读，需要能复读一次）；③ figshare 版本号，以及存档里 39 个文件名与平台是否逐一对应 |

> 【坑】42,169 是**记录数**，14,717 是**重复单元数**。两者在 PolyBench 里不相等，报告里不能说混（Proposal 附录 A 已按记录数给）。

### 2.2 PolyBench26（承重 · H2 与 H1 架构子句）

| 项 | 内容 |
|:--|:--|
| 入口 | <https://github.com/rlearsch/PolymerBenchmark2026> |
| 规模 | 约 **25 万**条性质数据点 / **8** 个性质 / 实验 + DFT + MD 三类来源；均聚物与交替、无规、嵌段共聚物 |
| 四个官方任务 | 分布内预测、数据集规模 scaling、重复单元复杂度、架构迁移到留出架构 |
| 官方训练档位 | 300 / 1000 / 3000 / 10000（交替 MD_5000）；同聚物 MD_5000 只有 300 / 1000；PolyMetriX Tg 为 300 / 1000 / 3000。5 折，base seed 42，val/test 各 10%，训练集为嵌套前缀 |
| 论文 | *An open benchmark for machine learning-based polymer property prediction*，arXiv:2609.27036（2026-09-22 提交 / 09-24 发布）；LLNL 与 FAIR at Meta |
| `version` 取值 | **git commit 短 sha**（该仓库没有 tag 时以 commit 为准），另记抓取日期 |
| 核实时点 | URL、规模、任务与档位：**2026-11-11 公开检索核实** |
| 待确认 | ① 仓库 LICENSE 类型（本次检索未取到，**不能用"大概是开源"代替**）；② `scaling_splits` 与 5 折索引的确切目录与文件名——A-5 要"原样快照"，得先知道快照什么；③ 是否需要 Git LFS 或额外数据下载脚本（仓库大小可能超预期） |

> 【红线】这是唯一提供**官方**规模操纵与架构操纵的资源。自己造子采样、自己造 wPSMILES 加权，结果就成了我们的产物而不是这个领域的属性（Proposal §2.1）。所以 `scaling_splits` 必须原样取，不许重算。

### 2.3 OpenPoly（仅边界 · H4）

| 项 | 内容 |
|:--|:--|
| 入口 | <https://github.com/WangGroupFDU/Openpoly_benchmark>（代码 **MIT**） |
| 规模 | **741 行 / 741 个不同 PSMILES / 3,985 条结构-性质标注 / 26 个性质**（2026-11-11 直接数发布 CSV） |
| 论文 | *OpenPoly: A Polymer Database Empowering Benchmarking and Multi-property Predictions*，Chin. J. Polym. Sci. **2025**, 43(10), 1749–1760，DOI `10.1007/s10118-025-3402-y` |
| 数据许可 | 论文 Data Availability 声明为 **CC BY 4.0**（代码 MIT，两者不是一回事） |
| 官方划分 | 仓库自带 `train_and_test_code/`（含数据预处理、超参搜索与评估协议）。**这就是 OpenPoly 的官方协议**，我们只快照、不自造 |
| 预训练权重 | Zenodo DOI `10.5281/zenodo.15551637`。**本仓库不需要**——我们只跑自己的五个臂，不加载别人的权重 |
| `version` 取值 | git commit 短 sha（仓库 2025-05-30 首次上传） |
| 核实时点 | URL、规模、许可、论文元数据：**2026-11-11 公开检索核实**；741 / 3,985 两个数：2026-11-11 直接数发布 CSV（Proposal §2.3） |
| 待确认 | `data/` 下 CSV 的确切文件名与列名——A-7 逐列数之前先把这一层记下来，否则数的对象都可能不是同一张表 |

> 【坑一】仓库 README 写的是 "3,985 polymer entries"，而实测是 **741 行**、3,985 条**标注**。措辞与结构对不上，**以实测为准**（A-7 复核项）。
>
> 【坑二】同目录下 `final_polymer_property_counts_from_literature.csv` 报 Tg **58,755**、折射率 7,878——那是**匹配结构之前**的文献记录数，比训练表高 1–2 个数量级。**不许用**（Proposal §2.3 已把它列为需要更正的典型口径错误）。
>
> 【坑三】网页版数据库 <https://cleanenergymaterials.cn/polymer/polymer_database/experiment_polymer_database> 需要注册登录，且不保证与仓库 CSV 同步。**以仓库 CSV 为准**，网页只用来交叉看一眼。

### 2.4 MoleculeNet / PCQM4Mv2（背景，不下载）

| 项 | 内容 |
|:--|:--|
| 入口 | <https://moleculenet.org> / <https://ogb.stanford.edu/docs/l2/pcqm4mv2/> |
| 用途 | 只在汇报里作跨层级背景（Proposal §2.1：「不承重」） |
| 决定 | **不下载、不建 manifest、不进任何表**。PCQM4Mv2 测试标签不公开，任何主张都不能落在它身上 |

---

## 3. 为什么"四基准"只下三个

Proposal §2.1 数的是**承重资源**，不是"数据集个数"。四个资源里承重的只有两个（PolyBench、PolyBench26），OpenPoly 只在边界上承重，第四个（MoleculeNet / PCQM4Mv2）**完全不承重**。

所以 A-2 的下载体量定为 **3 个**：少下一个不承重的资源不影响任何一条假设，多下一个则要付出数 GB 磁盘与数小时网络，却换不来一个能进表的数。**分工_A A-2 已按此改口径；若将来有人要用第四个，先改本节 + 记 `docs/DECISIONS.md`，再下载。**

---

## 4. A 在 D0 前必须确认的四件事

1. **PolyBench 平台要不要账号**；`/api/data-counts` 能否复读（附录 A 的 39 个条数的来源）。
2. **PolyBench26 的官方索引在哪**——`scaling_splits` 与 5 折文件的目录/文件名；没有它 A-5 无法"原样快照"，而 A-5 是课程红线相关项。
3. **PolyBench26 的 LICENSE**，以及 OpenPoly 的"代码 MIT / 数据 CC BY 4.0"是否足以支撑我们入库派生表（`data/manifests/`、`splits/`、`configs/tasks/` 都含派生信息）。
4. **磁盘**：PolyBench 39 个 CSV 与 OpenPoly 都很小，PolyBench26 是大头（约 25 万点）。A 侧合计预计 < 5 GB；polyBERT 的 7.3 GB 权重归 B，不用管。

---

## 5. 变更记录

| 日期 | 改了什么 | 谁 |
|:--|:--|:--|
| 2026-11-11 | 建档：四个资源入口、许可、版本钉法与三条已知坑 | 仓库所有者（开工前补齐） |
