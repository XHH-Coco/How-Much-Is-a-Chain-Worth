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
| 入口 | 平台 <http://polybench.ciac.jl.cn>，论文给出的备用地址 <http://202.98.16.11>。**仓库所有者 2026-11-11 实际访问并完成注册用的是 IP 地址**；域名那条是否可达**未验证**，别当着能用。账号本身不入库（见 §4 第 1 条） |
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
| 规模 | 约 **25 万**条性质数据点 / **8** 个性质 / 实验 + DFT + MD 三类来源；均聚物与交替、无规、嵌段共聚物。**结构级条数另有一组官方数字，别混**：`Cleaned_OMersBench_v3_final.jsonl` 82,990 条（交替 65,221 / 无规 10,379 / 均聚 7,390）+ PolyMetriX Tg 7,367 条 + polyVERSE 738 条（EA 368 / IP 370）。「25 万点」是**性质点**口径，「82,990」是**记录**口径，A-2 数行数时以文件内条数为准并在 manifest 注明是哪一种 |
| 四个官方任务 | 分布内预测、数据集规模 scaling、重复单元复杂度、架构迁移到留出架构 |
| 官方训练档位 | 300 / 1000 / 3000 / 10000（交替 MD_5000）；同聚物 MD_5000 只有 300 / 1000；PolyMetriX Tg 为 300 / 1000 / 3000。5 折，base seed 42，val/test 各 10%，训练集为嵌套前缀 |
| 论文 | *An open benchmark for machine learning-based polymer property prediction*，arXiv:2609.27036（2026-09-22 提交 / 09-24 发布）；LLNL 与 FAIR at Meta |
| 仓库许可 | **MIT——但只覆盖仓库软件**。README 原文：licence applies to repository software; individual source datasets remain subject to their own terms。**数据另有分层许可，见下方【坑四】** |
| 数据许可（分层） | OPoly26 / OMersBench：**CC BY 4.0**（源 `facebook/OMol25`，钉在 commit `47146a3ac4a3451741993a1605ca1b1050c6b9bc`）· VIPEA：**MIT**（可再分发）· PolyMetriX Tg：**CC BY 4.0**（Zenodo `10.5281/zenodo.14980914`，v1）· polyVERSE（EA/IP）：**GTRC General Public Use License，非 MIT** · PoLyInfo：**不得再分发**（不在公开源清单内） |
| 官方索引（A-5 快照对象） | **`Datasets/scaling_indices/`**——这是**入库的**紧凑划分索引：每折存整数 `train_pool` / `val` / `test` 数组 + canonical PSMILES 的 SHA-256；训练集 size N = `train_pool[:N]` 前缀。`Datasets/scaling_splits/` 那些展开好的 CSV 是 **gitignored、本地生成**（`generate_scaling_datasets.sh`），**不需要也不应该快照** |
| 协议参数（已核实） | 5 折，base seed **42**；val / test 各占全量 10%，train_pool 为余下 80%；交替 MD_5000 档位 300/1000/3000/10000，**同聚物 MD_5000 只有 300/1000**（全量仅约 1,715–1,800 行，3000 档不可行），PolyMetriX Tg 档位 300/1000/3000 |
| 上游源文件 sha256（官方公布，A-2 可直接核对） | `Datasets/Dataset_construction_scripts/files/Cleaned_OMersBench_v3_final.jsonl` = `7e8a17…2c68567e`（全文 `7e8a175866cc7166e589fbca1bb125bdece31c398b654a8848f5410f2c68567e`，82,990 条：交替 65,221 / 无规 10,379 / 均聚 7,390）· `Datasets/PSMILES/PolyMetriX/Tg/Tg.csv` = `b6d2b9f83419c94c5b24200b0288d9a7b53256931d87ffc1533c8884119a09d3`（7,367 条 PSMILES） |
| `version` 取值 | **git commit 短 sha**（该仓库没有 tag 时以 commit 为准），另记抓取日期 |
| 核实时点 | URL、规模、任务与档位：2026-11-11 公开检索核实；**许可分层、`scaling_indices/` 目录、5 折参数、上游 sha256：2026-11-11 当晚读 `README.md` / `DATA_PROVENANCE.md` / `SCALING_EXPERIMENTS_METHODS.md` 原文核实** |
| 待确认 | ① 是否需要 Git LFS 或额外下载脚本（仓库体积可能超预期）；② `Datasets/scaling_indices/` 下按数据集分的**确切子目录名与文件名**（目录已定位，命名待 clone 后登记）；③ 我们的两个主表任务（PolyMetriX Tg、同聚物 MD_5000 density）在 `scaling_indices/` 里对应的实验名逐一登记 |

> 【红线】这是唯一提供**官方**规模操纵与架构操纵的资源。自己造子采样、自己造 wPSMILES 加权，结果就成了我们的产物而不是这个领域的属性（Proposal §2.1）。所以划分**必须**来自官方发布物，不许自行洗牌。
>
> 【2026-11-11 当晚更正】红线原文写的是"`scaling_splits` 必须原样取"，**这条按字面做不到**：`Datasets/scaling_splits/` 被官方 gitignore，只存在于跑过 `generate_scaling_datasets.sh` 的本地机器上，仓库里没有可取的实体。可行且等价的解法是取 **`Datasets/scaling_indices/`**——官方入库的紧凑索引（整数行号 + PSMILES sha256），再用官方前缀规则 `train_pool[:N]` 展开。**判定标准不变：划分必须可追到官方发布物，不得由我们的代码决定谁进训练集。** A-5 的产出路径与工序需据此调整，见 §4 第 2 条。
>
> 【坑四】**MIT 不等于数据可以随便用。** 仓库 LICENSE 是 MIT，但 README 明说 MIT 只管软件；`DATA_PROVENANCE.md` 逐源列了许可，四档不同：OPoly26/OMersBench 与 PolyMetriX 是 CC BY 4.0（可入库派生，须署名）、VIPEA 是 MIT、**polyVERSE 的 EA/IP 是 GTRC 协议**（须随附 GTRC 全文、免费分发、不得商用——所以 `bandgap`、`Ionization Energy_Ei` 这类来自 polyVERSE 的任务，其结果表能不能公开要先看 GTRC）、**PoLyInfo 完全不得再分发**。我们的三个主表任务都落在 CC BY 4.0 档，入库 `data/manifests/` 与 `splits/` 没有问题，但**署名与许可声明必须随表走**。
>
> 【坑五】**索引是行号，行号没有独立意义。** `scaling_indices/` 存的是整数下标，它绑定的是"官方脚本从 tracked 源文件生成 PSMILES 表时的那个顺序"。换了源文件版本、换了生成脚本，同样的行号指向的就是别的分子，而且**不会报错**——只会安静地算出一组错的 R²。所以快照索引时必须同时钉住三样：① 仓库 commit sha；② `Cleaned_OMersBench_v3_final.jsonl` 与 `Tg.csv` 的 sha256（官方已公布，见上表，A-2 下载后直接核对）；③ `generate_basic_datasets.sh` 的 commit。三者缺一，快照就是不可复现的。

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

## 4. A 在 D0 前必须确认的四件事（2026-11-11 当晚更新状态）

| # | 事项 | 状态 | 还需要谁做什么 |
|:--|:--|:--|:--|
| 1 | PolyBench 平台要不要账号 / `/api/data-counts` 能不能复读 | 🟡 **账号已注册**（仓库所有者，2026-11-11）；**接口未验证** | A 在登录态下确认：接口路径是否仍为 `/api/data-counts`、返回结构、以及 39 个 n 相加是否仍为 42,169。**结论只写数与时间，不写账号** |
| 2 | PolyBench26 的官方索引在哪 | ✅ **已定位** | 取 `Datasets/scaling_indices/`（入库发布物），不要找 `scaling_splits/`（gitignored）。**A-5 工序需按 §2.2【2026-11-11 当晚更正】改**，改法见该条 |
| 3 | 许可是否足以支撑入库派生表 | ✅ **已解决** | 三主表任务均 CC BY 4.0，`data/manifests/` 与 `splits/` 可入库，**须随表带署名与许可声明**；避开 polyVERSE（GTRC）与 PoLyInfo（不可分发） |
| 4 | 磁盘 | ⬜ 未实测 | A 侧预计 < 5 GB（PolyBench 39 CSV + OpenPoly + PolyBench26 源文件 ~83k 条 jsonl）；polyBERT 的 7.3 GB 归 B。**下载前先 `df` 一遍**，PolyBench26 若走 LFS 会翻倍 |

**仍然悬空的一件事（不影响 A-2，影响 A-5 的写法）**：`scaling_indices/` 的具体子目录与命名要 clone 之后才能登记。在它落地之前，`splits/polybench26/scaling/` 保持为空，**不要用 `sklearn` 现造一个划分顶上**——那正好是红线拦的东西。

---

## 5. 变更记录

| 日期 | 改了什么 | 谁 |
|:--|:--|:--|
| 2026-11-11 | 建档：四个资源入口、许可、版本钉法与三条已知坑 | 仓库所有者（开工前补齐） |
| 2026-11-11 当晚 | PolyBench26 许可落定：仓库 **MIT**（仅软件），数据分层（OPoly26 / PolyMetriX = CC BY 4.0、VIPEA = MIT、polyVERSE = GTRC、PoLyInfo 不可分发），新增【坑四】 | 仓库所有者（GitHub 仓库页确认） |
| 2026-11-11 当晚 | 官方索引定位到 `Datasets/scaling_indices/`（入库）/ `scaling_splits/`（gitignored）；补 5 折与档位参数、上游源文件 sha256；**更正红线措辞**（"原样取 scaling_splits"按字面做不到），新增【坑五】行号绑定；§4 四件事改状态表 | 读官方 `README.md`、`DATA_PROVENANCE.md`、`SCALING_EXPERIMENTS_METHODS.md` 原文 |
