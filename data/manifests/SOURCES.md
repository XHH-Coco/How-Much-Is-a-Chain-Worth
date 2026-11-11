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
| 计数接口（**已复读**） | `http://202.98.16.11/api/data-counts`，**无需登录即可 GET**，返回 `{"data_counts":[{"Dataset":…, "Count":…}, …]}`。2026-11-11 20:58 实测：**39 条，`Count` 相加 = 42,169**；与 Proposal 附录 A 的 39 行**逐行比对 0 处不一致**（名字与条数全对，集合双向差集均为空）。原始响应已入库：`data/manifests/polybench_api_data_counts_20261111.json`（1,978 字节，sha256 `bae4eb5c32c4ff9786e22c8db9dd63e0cdb215b858b3f0618982dc0e9d9be766`）。**这份响应就是附录 A 的引证**——不要重跑覆盖它，要更新就另存新文件并在本表登记 |
| 论文 | *PolyBench: A Reproducible Open Benchmark Platform for Repeat-Unit-Based Polymer Property Prediction*，Polym. Sci. Technol. **2026**, 2(10), 836–854，DOI `10.1021/polymscitech.6c00094`，2026-08-31 上线（开放获取） |
| 存档 | figshare，DOI `10.6084/m9.figshare.30917717`，**CC BY 4.0**；39 个 CSV 的文件名 = 数据集名（PolyBench 不发数字任务 ID） |
| `version` 取值 | figshare 存档版本号（`v1`/`v2`…）；若从平台直取，记抓取日期 + 平台页脚版本串 |
| 核实时点 | URL、规模、论文元数据：**2026-11-11 公开检索核实**。figshare DOI：转录自 Proposal §2，**本次未重新解析** |
| 待确认（A 在 D0 前做） | ① **39 个 CSV 的批量获取方式**——平台有导出按钮/接口，还是只能逐个点；② **figshare 存档能否匿名取**（DOI `10.6084/m9.figshare.30917717`），版本号是 `v1` 还是更高；③ 存档里 39 个文件名与平台提示名是否逐一对应。**账号与接口这两条已闭合**，不要再当未解决问题 |

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
| 官方索引（A-5 快照对象） | **`Datasets/scaling_indices/`**——官方**入库**的紧凑划分。目录 README 原话：*intentionally tracked in Git so a new model can use the published folds **without generating polyBERT embeddings, RDKit descriptors, or the 7.6 GB materialized split tree***。布局 `<数据集族>/<性质>/<实验名>/seed_42/{manifest.json, fold_00.npz … fold_04.npz}`：`manifest.json` 记 canonical dataset 路径与其 SHA-256、target、fold 设置、training sizes、行数；`fold_NN.npz` 存整数数组 `train_pool` / `val` / `test`，训练 size N 取 `train_pool[:N]`。官方自带校验 `manage_scaling_indices.py verify`（在 `Datasets/Dataset_construction_scripts/`，用它自己的 venv） |
| 主表对应路径（**已核实**） | PolyMetriX Tg → `Datasets/scaling_indices/PolyMetriX/Tg/Tg/seed_42/`（`Tg` 连两层是官方原样，不是笔误）；同聚物 density → `Datasets/scaling_indices/MD_5000/density/homopolymer_density/seed_42/`。`MD_5000/` 下共五个性质 `Cp` `Cv` `Rg` `density` `refractive_index`，其中 `density` 再分 `alternating_density` / `homopolymer_density` |
| 磁盘（实测量级） | 官方 README 要 **~15 GB 空闲**，但那是**完整复现**（含 7.6 GB 物化划分树 + 四套表示 + 全部生成结果）。**我们不需要这些**：只要 `Cleaned_OMersBench_v3_final.jsonl`（66,878,709 B ≈ 63.8 MiB）+ `PolyMetriX/Tg/Tg.csv`（558,386 B）+ `scaling_indices/`（npz 与 manifest，几十 MB），合计 **< 200 MB**。`scaling_splits/`、polyBERT embeddings、RDKit descriptors 一律不生成本地副本 |
| 协议参数（已核实） | 5 折，base seed **42**；val / test 各占全量 10%，train_pool 为余下 80%；交替 MD_5000 档位 300/1000/3000/10000，**同聚物 MD_5000 只有 300/1000**（全量仅约 1,715–1,800 行，3000 档不可行），PolyMetriX Tg 档位 300/1000/3000 |
| 上游源文件 sha256（官方公布，A-2 可直接核对） | `Datasets/Dataset_construction_scripts/files/Cleaned_OMersBench_v3_final.jsonl` = `7e8a17…2c68567e`（全文 `7e8a175866cc7166e589fbca1bb125bdece31c398b654a8848f5410f2c68567e`，82,990 条：交替 65,221 / 无规 10,379 / 均聚 7,390）· `Datasets/PSMILES/PolyMetriX/Tg/Tg.csv` = `b6d2b9f83419c94c5b24200b0288d9a7b53256931d87ffc1533c8884119a09d3`（7,367 条 PSMILES） |
| `version` 取值 | **git commit 短 sha**（该仓库没有 tag 时以 commit 为准），另记抓取日期 |
| 核实时点 | URL、规模、任务与档位：2026-11-11 公开检索核实；**许可分层、`scaling_indices/` 目录、5 折参数、上游 sha256：2026-11-11 当晚读 `README.md` / `DATA_PROVENANCE.md` / `SCALING_EXPERIMENTS_METHODS.md` 原文核实** |
| 待确认 | ① `manage_scaling_indices.py verify` 在"只取部分文件"时能否跑通（它可能要求完整目录树）；② 仓库是否走 Git LFS——API 报的 jsonl 是 66.9 MB 实际字节数、不像 LFS，但**首次 clone 时确认一次**；③ **架构迁移任务（H1 架构子句）的官方划分在哪**——`scaling_indices/` 只覆盖数据集规模实验，架构留出集的文件位置本次未定位，这是 A-5 的第二块缺口 |

> 【红线】这是唯一提供**官方**规模操纵与架构操纵的资源。自己造子采样、自己造 wPSMILES 加权，结果就成了我们的产物而不是这个领域的属性（Proposal §2.1）。所以划分**必须**来自官方发布物，不许自行洗牌。
>
> 【2026-11-11 当晚更正】红线原文写的是"`scaling_splits` 必须原样取"，**这条按字面做不到**：`Datasets/scaling_splits/` 被官方 gitignore，只存在于跑过 `generate_scaling_datasets.sh` 的本地机器上，仓库里没有可取的实体。可行且等价的解法是取 **`Datasets/scaling_indices/`**——官方入库的紧凑索引（整数行号 + PSMILES sha256），再用官方前缀规则 `train_pool[:N]` 展开。**判定标准不变：划分必须可追到官方发布物，不得由我们的代码决定谁进训练集。** A-5 的产出路径与工序需据此调整，见 §4 第 2 条。
>
> 【坑四】**MIT 不等于数据可以随便用。** 仓库 LICENSE 是 MIT，但 README 明说 MIT 只管软件；`DATA_PROVENANCE.md` 逐源列了许可，四档不同：OPoly26/OMersBench 与 PolyMetriX 是 CC BY 4.0（可入库派生，须署名）、VIPEA 是 MIT、**polyVERSE 的 EA/IP 是 GTRC 协议**（须随附 GTRC 全文、免费分发、不得商用）、**PoLyInfo 完全不得再分发**。**我们主表选的两个 PolyBench26 任务（PolyMetriX Tg、同聚物 MD density）都落在 CC BY 4.0 档，入库 `data/manifests/` 与 `splits/` 没有问题**，但署名与许可声明必须随表走。落在 GTRC 档的是 PolyBench26 的 EA/IP 两个任务，我们没选；`PoLyInfo` 全程不碰。
>
> 【注意·别串】PolyBench（平台那 39 个任务）里也有 `bandgap`、`Ionization Energy_Ei` 这些名字，但它们属于**另一个数据集**、走 PolyBench 自己的 CC BY 4.0。不要把 PolyBench26 的 polyVERSE 许可套到 PolyBench 的任务头上，反之亦然。
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
| 1 | PolyBench 平台要不要账号 / `/api/data-counts` 能不能复读 | ✅ **已闭合** | 平台**无需登录**即可 GET `http://202.98.16.11/api/data-counts`；2026-11-11 20:58 实测 39 条、合计 **42,169**，与附录 A 逐行 0 差异。原始响应已入库（见 §2.1）。**剩下的是 CSV 批量下载方式**，属 A-2 执行细节 |
| 2 | PolyBench26 的官方索引在哪 | ✅ **已定位到文件级** | 取 `Datasets/scaling_indices/<族>/<性质>/<实验>/seed_42/{manifest.json, fold_NN.npz}`；主表两条路径见 §2.2。**A-5 工序需按 §2.2【更正】改**，改法见该条 |
| 3 | 许可是否足以支撑入库派生表 | ✅ **已解决** | 主表两个 PB26 任务均 CC BY 4.0，`data/manifests/` 与 `splits/` 可入库，**须随表带署名与许可声明**；避开 polyVERSE（GTRC）与 PoLyInfo（不可分发） |
| 4 | 磁盘 | ✅ **已实测** | 本机 D 盘 489 GB 可用，而**我们只需 < 200 MB**（jsonl 63.8 MiB + Tg.csv 0.53 MiB + indices 数十 MB）。官方要的 ~15 GB 是完整复现口径，**不要照那个准备** |

**仍然悬空的两件事（不影响 A-2，影响 A-5 的写法）**：① `manage_scaling_indices.py verify` 在部分目录下能否跑通；② **架构迁移任务的官方划分文件位置未定位**。在这两条落地之前，`splits/polybench26/scaling/` 保持为空，**不要用 `sklearn` 现造一个划分顶上**——那正好是红线拦的东西。

---

## 5. 变更记录

| 日期 | 改了什么 | 谁 |
|:--|:--|:--|
| 2026-11-11 | 建档：四个资源入口、许可、版本钉法与三条已知坑 | 仓库所有者（开工前补齐） |
| 2026-11-11 当晚 | PolyBench26 许可落定：仓库 **MIT**（仅软件），数据分层（OPoly26 / PolyMetriX = CC BY 4.0、VIPEA = MIT、polyVERSE = GTRC、PoLyInfo 不可分发），新增【坑四】 | 仓库所有者（GitHub 仓库页确认） |
| 2026-11-11 当晚 | 官方索引定位到 `Datasets/scaling_indices/`（入库）/ `scaling_splits/`（gitignored）；补 5 折与档位参数、上游源文件 sha256；**更正红线措辞**（"原样取 scaling_splits"按字面做不到），新增【坑五】行号绑定；§4 四件事改状态表 | 读官方 `README.md`、`DATA_PROVENANCE.md`、`SCALING_EXPERIMENTS_METHODS.md` 原文 |
| 2026-11-11 当晚 | **`/api/data-counts` 复读成功**：39 条、合计 42,169、与附录 A 逐行 0 差异；原始响应入库 `polybench_api_data_counts_20261111.json`。§2.1「待确认」由"要不要账号 / 接口能不能复读"改为只剩"CSV 批量下载方式 + figshare 可否匿名" | 仓库所有者在本机 GET 接口，仓库所有者转交响应 |
| 2026-11-11 当晚 | `scaling_indices/` 细化到文件级：布局 `<族>/<性质>/<实验>/seed_42/{manifest.json, fold_NN.npz}`，主表两条路径已核实；补磁盘量级（我们 <200 MB vs 官方 ~15 GB 完整复现）、官方校验命令 `manage_scaling_indices.py verify`；新增"架构迁移划分未定位"这条缺口 | 读 `Datasets/scaling_indices/README.md` 与 GitHub contents API |
