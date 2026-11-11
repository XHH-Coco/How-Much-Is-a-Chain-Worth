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
| 存档（**已解析，v2**） | figshare，DOI `10.6084/m9.figshare.30917717`，**实际版本 `v2`**（`doi` 字段为 `.v2`），CC BY 4.0（`license.value=1`），公开、**无需登录、未禁用下载**（`download_disabled: false`、`is_embargoed: false`）。**存档页**（浏览器里点 *Download all* 用这条，slug 很长，直接用 DOI 跳转更省事）：<https://figshare.com/articles/dataset/PolyBench_A_Standardized_Benchmark_Platform_for_Artificial_Intelligence_in_Polymer_Science/30917717>，等价短入口 <https://doi.org/10.6084/m9.figshare.30917717>；整包 zip 形如 <https://figshare.com/ndownloader/articles/30917717/versions/2>（按 figshare 固定格式拼，本机 403 未验证）。**39 个文件**，合计 **5,937,499 B ≈ 5.66 MiB**，全部 `mimetype: text/csv`，`folder_structure` 均为 `SMILES`。文件名 = 数据集名 + `.csv`（含空格原样），**39 个与 `/api/data-counts` 的 39 条、附录 A 的 39 行三方对齐**。每个文件自带 `supplied_md5` 且与 `computed_md5` **全部相同**，可直接用于二次校验。下载地址形态：`https://ndownloader.figshare.com/files/<file_id>`，清单从 `GET https://api.figshare.com/v2/articles/30917717` 取 |
| `version` 取值 | 存档取 **`v2`**（写进 manifest 的 `version` 列，形如 `figshare:v2`）；若从平台直取，记抓取日期 + 平台页脚版本串 |
| 核实时点 | URL、规模、论文元数据：2026-11-11 公开检索核实；**figshare 版本号、39 文件清单、总字节、md5、免登录：2026-11-11 当晚调 figshare API v2 核实** |
| 39 个文件的标准表 | `data/manifests/polybench_expected_files.csv`：39 行，列为 `file_name, task_id, size_bytes, md5, figshare_file_id, download_url`；由原始响应快照 `data/manifests/polybench_api_article_20261112.json`（v2、CC BY 4.0）机械派生，字节合计 5,937,499 = 存档 `size` 字段。**任何路线（换网络 / 浏览器 `Download all` / 平台手动 / 人代下）拿到的 CSV 都以这张表的 `md5` 为唯一验收标准**，见 `DECISIONS.md` D-13 |
| 待确认（A 在 D0 前做） | ✅ **已全部闭合（2026-11-12）**。**平台的批量获取方式：没有。** `GET /api/dataset/test-download?file_name=<名>.csv` 只返回两列 `polymer_name,SMILES`（前端文案即 *Download Test Set (SMILES only)*，标签不给）；`/api/polymers-data` 是平台展示库（分页 JSON，每页 10 条），**不是**这 39 个官方 CSV；`/api/dataset/download`、`/api/datasets`、`/data/SMILES/<名>.csv` 均 404 或回落前端。**所以 figshare 存档是唯一可脚本化的正规入口**——但它对本机出口一律 403，见【坑十】。账号、接口、版本号、文件名对应四条均已闭合 |

> 【坑】42,169 是**记录数**，14,717 是**重复单元数**。两者在 PolyBench 里不相等，报告里不能说混（Proposal 附录 A 已按记录数给）。
>
> 【坑十】**figshare 从我们这台机器的出口一律 403，不是脚本问题。** 2026-11-12 实测：`figshare.com` / `api.figshare.com` / `ndownloader.figshare.com` 的**全部**路径都返回 `403`，响应头 `Server: awselb/2.0`——拒绝方是 figshare 自己的负载均衡。走代理、不走代理、IPv4、IPv6、`http://`、伪装 UA 与 Referer，结果一致；`doi.org` 能 `302` 落到存档页，随后仍是 `403`。**同一时刻换一条网络路径读同一个 API 是通的**（能列出 39 个文件与各自 `supplied_md5`，`version: 2`、CC BY 4.0、`download_disabled: false`），所以存档没下架、也没改版本。**处理办法只有三条**：换网络重跑 `python scripts/00_download.py --source polybench`；或在能打开存档页的机器上点 *Download all* 把 zip 拷回来解压（39 个 CSV 平铺进 `data/raw/polybench/`）；或登录平台/manual 逐个下，文件名照 `polybench_expected_files.csv` 抄。**不管哪条，收尾都是 `python scripts/00_download.py --verify-only`**：不出网、逐份核官方 md5、39 份全过才写清单（本地已就位也可以让脚本直接跑第一条）。**不许用平台的 `/api/polymers-data` 或 test-download 的 SMILES 表顶替**——那样 sha256 与发布方对不上，A-3 之后每个数都会带一个查不出来的来源。详见 `docs/交付_P1_A2_下载与完整性.md` §5。
>
> 【坑十二】**`Isothermal weight loss time.csv` 的表头有一处编码损伤**：列名 `IWLT(â„ƒ)` 是 `℃` 被二次编码，同表的 `IWLT(hour)` 正常。**A-3 做单位判定时不要把这一列当"温度单位证据"**——证据取 `IWLT(hour)` 与文献口径，并在 `unit_decisions.csv` 的 `evidence` 里写明"原始表头存在编码损伤"。其余 38 个文件严格 UTF-8 可解、无乱码特征。
>
> 【坑十一】**`*.conda.txt` 那种"一次请求拿全"的思路，在 GitHub 上要靠 tarball。** 实测本机 `raw.githubusercontent.com` 会在传输中途被 TLS 重协商打断（`curl` 退出码 56 / 20 秒断连），而 `codeload.github.com/<repo>/tar.gz/<commit>` 稳定且一个请求就能把整个仓库取回。所以 `scripts/00_download.py` 对两个 GitHub 基准一律走 tarball + 白名单解包，**不改字节、只挑文件**；两个官方公布过 sha256 的文件解出来当场核对，逐位相等才算过。这条不影响可追溯性：清单里的 `source_url` 指向 `github.com/<repo>/blob/<commit>/<path>`，`version` 是 commit 短 sha。

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
| 待确认 | ① 仓库是否走 Git LFS（API 报的字节数是真实大小、不像 LFS，首次 clone 时确认一次）；② `PSMILES_pBERT_dict.pkl`（224 MB 生成物）是否必需——只有 polyBERT 臂需要，其余臂可绕开。**③④ 已于 2026-11-11 深夜闭合，见下方【坑八】与【坑七】** |

> 【红线】这是唯一提供**官方**规模操纵与架构操纵的资源。自己造子采样、自己造 wPSMILES 加权，结果就成了我们的产物而不是这个领域的属性（Proposal §2.1）。所以划分**必须**来自官方发布物，不许自行洗牌。
>
> 【2026-11-11 当晚更正】红线原文写的是"`scaling_splits` 必须原样取"，**这条按字面做不到**：`Datasets/scaling_splits/` 被官方 gitignore，只存在于跑过 `generate_scaling_datasets.sh` 的本地机器上，仓库里没有可取的实体。可行且等价的解法是取 **`Datasets/scaling_indices/`**——官方入库的紧凑索引（整数行号 + PSMILES sha256），再用官方前缀规则 `train_pool[:N]` 展开。**判定标准不变：划分必须可追到官方发布物，不得由我们的代码决定谁进训练集。** A-5 的产出路径与工序需据此调整，见 §4 第 2 条。
>
> 【坑四】**MIT 不等于数据可以随便用。** 仓库 LICENSE 是 MIT，但 README 明说 MIT 只管软件；`DATA_PROVENANCE.md` 逐源列了许可，四档不同：OPoly26/OMersBench 与 PolyMetriX 是 CC BY 4.0（可入库派生，须署名）、VIPEA 是 MIT、**polyVERSE 的 EA/IP 是 GTRC 协议**（须随附 GTRC 全文、免费分发、不得商用）、**PoLyInfo 完全不得再分发**。**我们主表选的两个 PolyBench26 任务（PolyMetriX Tg、同聚物 MD density）都落在 CC BY 4.0 档，入库 `data/manifests/` 与 `splits/` 没有问题**，但署名与许可声明必须随表走。落在 GTRC 档的是 PolyBench26 的 EA/IP 两个任务，我们没选；`PoLyInfo` 全程不碰。
>
> 【注意·别串】PolyBench（平台那 39 个任务）里也有 `bandgap`、`Ionization Energy_Ei` 这些名字，但它们属于**另一个数据集**、走 PolyBench 自己的 CC BY 4.0。不要把 PolyBench26 的 polyVERSE 许可套到 PolyBench 的任务头上，反之亦然。
>
> 【坑五】**索引是行号，行号没有独立意义。** `scaling_indices/` 存的是整数下标，它绑定的是"官方脚本从 tracked 源文件生成 PSMILES 表时的那个顺序"。换了源文件版本、换了生成脚本，同样的行号指向的就是别的分子，而且**不会报错**——只会安静地算出一组错的 R²。所以快照索引时必须同时钉住三样：① 仓库 commit sha；② `Cleaned_OMersBench_v3_final.jsonl` 与 `Tg.csv` 的 sha256（官方已公布，见上表，A-2 下载后直接核对）；③ `generate_basic_datasets.sh` 的 commit。三者缺一，快照就是不可复现的。

**四个官方任务各自落在哪个目录。** `Datasets/README.md` 把「入库」与「生成」分得很清，这张表决定 A-5 能快照什么：

| 官方任务 | 目录 | 是否入库 | 快照可行性 |
|:--|:--|:--|:--|
| 数据集规模 scaling | `Datasets/scaling_indices/`（约 2 MB） | ✅ 入库 | **可直接快照** |
| 重复单元复杂度 | `Datasets/task3_complexity/`（README 1.6 KB + `alternating_refractive_index.csv` 3,344,980 B） | ✅ 入库 | **不适用——它不是一个留出任务，是事后分层评估**，见【坑八】。我们不用，A-5 不快照 |
| 架构迁移 | `Datasets/task4_architecture_transfer/`（约 0.4 MB） | ✅ 入库，但只是**「物化工作流」** | **不能快照留出集本身**，只能快照脚本 + 生成参数 |
| 分布内预测 | 无独立目录 | — | 由各模型自己的 5 折决定，不走这两个目录 |

> 【坑六】**索引指向的那张表，官方不发布。** `Datasets/README.md` 明确 `PSMILES/MD_300/`、`MD_5000/`、`Vipea/`、`Tg_combined/` 全是 *generated via scripts and excluded from Git*（PSMILES 约 90 MB、wPSMILES 约 206 MB），入库的只有 compact 源（`Dataset_construction_scripts/files/`，约 81 MB）与 `PSMILES/polyVERSE/`、`PSMILES/PolyMetriX/` 这些小的网页源。**所以 A-5 的正确工序是**：先跑官方 `generate_basic_datasets.sh` 生成 canonical PSMILES → 拿 `scaling_indices/<实验>/seed_42/manifest.json` 里的 `canonical_sha256` 校验这张表 → 校验通过后行号才有意义。跳过这步，索引就是一组随机数。
>
> 【坑七】**架构迁移走的是 GTRC 数据，不是 CC BY 4.0。** `task4_architecture_transfer/README.md` 写明该任务预测的是 **electron affinity**，训练源为 `PSMILES/polyVERSE/electron_affinity/…` + `wPSMILES/Vipea/EA/alternating_EA.csv`。polyVERSE 属 **GTRC General Public Use License**（见【坑四】）。**架构迁移已由 `分工_C` 的 C-4 纳入本次范围（5.0 pH）**，所以这条不是"要不要做"的问题，而是"结果表怎么写"的问题：**须随附 GTRC 全文、免费分发、不得商用**，且 `bandgap` 那类 PolyBench 任务不能跟着一起套用。
>
> 【task4 的关键数字（已核实）】**目标只有 1 个性质——electron affinity（EA）**，不是两个。canonical 训练集 **6,478** 条（交替 6,110 + 均聚 368）；OOD holdout **36,828** 行（random 18,414 + block 18,414），**固定、永不进入训练或调参**。五折 80/10/10、PSMILES 分组、**seed 42**；物化命令 `create_task4_shared_splits.py --polybert-dictionary Datasets/Dataset_construction_scripts/files/PSMILES_pBERT_dict.pkl`（该 pkl 224 MB，本身是生成物；**省略该参数可先物化 PSMILES / wPSMILES / RDKit 三套，只有 polyBERT 臂才需要它**）；输出落在 `generated/shared_splits/`，gitignored。VIPEA 的 `wPSMILES/Vipea/EA/alternating_EA.csv` 也是生成物，需先跑 `process_Vipea_data.py` 从 tracked 源重建。
>
> 【坑八】**"重复单元复杂度"不是留出任务，是事后分层——所以 A-5 没有"复杂度划分"可快照。** `task3_complexity/README.md` 原话：*Task 3 is a post-hoc stratification of the legacy alternating refractive-index test predictions, not a complexity-held-out training task.* 目录里只有两个文件（README + `alternating_refractive_index.csv`，3,344,980 B，git blob `ed085bc1…`）。它的用法是：**用 legacy 交替折射率协议训练 → 评估其留出测试预测 → 按 `original_index` 与本表 join → 按 `n_distinct_monomers`（5–10）分别报 RMSE**。表里的 `legacy_test_fold`（1–5）是 **legacy 协议的折号**，不是 PB26 主表协议的折号，也不是"官方复杂度划分"。另外两条：① 论文的 Task 3 聚合**只用 5 ≤ n_distinct_monomers ≤ 10 的行**，2 或 4 个单元的行只为溯源保留、**不得进聚合**；② 迁移前文件 sha256 = `6d099ee5dd158c52448741859843a7eaa5c8c9ff7d4dc43cdc483bbb7d7d469a`（2026-09-25 从 legacy OpenPolymerBench Task 3 准备产物迁入）。**结论：我们主表没有 refractive_index、机制实验也不碰复杂度分层，所以这个任务不进 A-5，也不占工时。** 将来若要做，它要的是"按折号分组报 RMSE"，不是"切一份新划分"。
>
> 【坑九】**架构迁移的 OOD 折不是独立重复，不能做 t 检验。** 官方 README 原话：*The OOD holdout is fixed across folds, so fold metrics are not independent replicates. ... Do not apply a t-test to the five fold-level OOD RMSEs.* 正确做法：**保留全部 5 折的 OOD 预测向量 → 按聚合物取平均 → 从五折集成算出 random / block / combined 各一个 RMSE/MAE/MSE**。这条直接影响 `分工_D` 的置信区间与 H1 判决：**OOD 那一栏不能给"均值 ± 标准误"**，只能给一个集成点值 + 由 5 个预测向量重采样得到的区间。ID（交替 / 均聚两档）才允许按折汇总均值与总体标准差。**若 C-4 或 D 把 OOD 的 5 个折值当 5 个独立样本算显著性，结论就是假的，且不会有任何报错。**
>
> 【官方 verify 的真实能力（读了源码）】`manage_scaling_indices.py verify` 用 `rglob("manifest.json")` 遍历，**只校验存在的实验、不要求目录齐全**，所以部分快照也能跑；缺 canonical 源时必须加 `--allow-missing-sources`，否则抛 `FileNotFoundError`。支持 `--indices-root` / `--datasets-root` 指向任意目录，**A-5 可以直接用它校验我们自己的快照**。它检查：schema 版本、canonical sha256（若源存在）、fold 数与逐折 `seed = base + fold`、npz 恰含三个数组、`train_pool ∪ val ∪ test` 无重复且覆盖 `0..n_rows-1`、各 `n_*` 与数组长度一致、`max(train_sizes) ≤ len(train_pool)`。**但注意：它不检查「`train_pool[:N]` 是嵌套前缀」——那个断言只在 `export` 分支里有。所以前缀规则必须由我们自己的 `tests/test_splits.py` 兜，官方 verify 兜不住。**

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
| 待确认 | ✅ **已落定（2026-11-12）**：`data/` 下只有两个 CSV，**我们用的是 `data/final_polymer_properties_fromliterature.csv`**（178,587 B，741 行 × 32 列，前 12 列依次为 `Name` `PSMILES` `Bandgap (eV)` `CO2 Permeability (Barrer)` `Compressive Strength (MPa)` `Crystallization Temperature (K)` `Elongation at Break (%)` `Flexural Strength (MPa)` `Tg (K)` `H2 Permeability (Barrer)` `Hardness (MPa)` `Impact Strength (kJ/m2)`；sha256 `8de5baf7…9db5eb3`，commit `ccaa669`）；同目录的 `final_polymer_property_counts_fromliterature.csv`（849 B）就是【坑二】那张**不许用**的文献计数表，`scripts/00_download.py` 因此**故意不下载它** |

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
| 2 | PolyBench26 的官方索引在哪 | ✅ **四个任务全部定位，且"能不能用"也已判完** | scaling → `scaling_indices/`（**A-5 快照它**）；复杂度 → `task3_complexity/`（**不是留出任务，是事后分层，我们不用**，见【坑八】）；架构迁移 → `task4_architecture_transfer/`（**只发布物化脚本，不快照留出集**，只登记参数，见【坑七】）；分布内 → 各模型 5 折。**A-5 工序已按 §2.2【更正】与【坑六】改完，见 `分工_A`** |
| 3 | 许可是否足以支撑入库派生表 | ✅ **已解决，但多一条边** | 主表两个 PB26 任务均 CC BY 4.0，`data/manifests/` 与 `splits/` 可入库，**须随表带署名**；**架构迁移已纳入（C-4），走 polyVERSE = GTRC**（见【坑七】），结果表须随附 GTRC 全文、不得商用 |
| 4 | 磁盘 | ✅ **已实测** | 本机 D 盘 489 GB 可用，而**我们只需 < 200 MB**（jsonl 63.8 MiB + Tg.csv 0.53 MiB + indices 约 2 MB 的子集）。官方要的 ~15 GB 是完整复现口径，**不要照那个准备** |

**四件事全部闭合，A-2 / A-5 无阻塞项。** 剩下两条不是"待确认事实"，而是**已定口径的执行提醒**：

1. **`splits/polybench26/` 只放两类东西**：`<task>/seed_42/` 下从 `scaling_indices/` 原样拷来的 `manifest.json` + `fold_00..04.npz`；以及 A-5 自己写的 `README.md`（记来源 commit、源文件 sha256、校验命令）。**不要用 `sklearn` 现造任何划分顶上**——那是红线拦的东西。
2. **架构迁移不进 `splits/`**：它的留出集官方不发布，只发布物化脚本。`splits/polybench26/arch/` 保持为空并在 README 里注明"由 `create_task4_shared_splits.py` 本地物化，不入库"，C-4 用的时候自己生成、自己记 commit。

---

## 5. 变更记录

| 日期 | 改了什么 | 谁 |
|:--|:--|:--|
| 2026-11-11 | 建档：四个资源入口、许可、版本钉法与三条已知坑 | 仓库所有者（开工前补齐） |
| 2026-11-11 当晚 | PolyBench26 许可落定：仓库 **MIT**（仅软件），数据分层（OPoly26 / PolyMetriX = CC BY 4.0、VIPEA = MIT、polyVERSE = GTRC、PoLyInfo 不可分发），新增【坑四】 | 仓库所有者（GitHub 仓库页确认） |
| 2026-11-11 当晚 | 官方索引定位到 `Datasets/scaling_indices/`（入库）/ `scaling_splits/`（gitignored）；补 5 折与档位参数、上游源文件 sha256；**更正红线措辞**（"原样取 scaling_splits"按字面做不到），新增【坑五】行号绑定；§4 四件事改状态表 | 读官方 `README.md`、`DATA_PROVENANCE.md`、`SCALING_EXPERIMENTS_METHODS.md` 原文 |
| 2026-11-11 当晚 | **`/api/data-counts` 复读成功**：39 条、合计 42,169、与附录 A 逐行 0 差异；原始响应入库 `polybench_api_data_counts_20261111.json`。§2.1「待确认」由"要不要账号 / 接口能不能复读"改为只剩"CSV 批量下载方式 + figshare 可否匿名" | 仓库所有者在本机 GET 接口，仓库所有者转交响应 |
| 2026-11-11 当晚 | `scaling_indices/` 细化到文件级：布局 `<族>/<性质>/<实验>/seed_42/{manifest.json, fold_NN.npz}`，主表两条路径已核实；补磁盘量级（我们 <200 MB vs 官方 ~15 GB 完整复现）、官方校验命令 `manage_scaling_indices.py verify`；新增"架构迁移划分未定位"这条缺口 | 读 `Datasets/scaling_indices/README.md` 与 GitHub contents API |
| 2026-11-11 当晚 | **figshare 存档解析完成**：`v2`、39 文件、合计 5,937,499 B、全部 text/csv、`supplied_md5 == computed_md5`、**免登录且未禁用下载**、下载地址形态 `ndownloader.figshare.com/files/<id>`；§2.1「待确认」收敛为只剩"平台批量下载方式" | 调 figshare API v2 `/articles/30917717` |
| 2026-11-11 当晚 | **四个官方任务的落点全部定位**：新增任务目录表、【坑六】canonical PSMILES 需脚本生成（索引指向的表官方不发布）、【坑七】架构迁移走 polyVERSE = GTRC、task4 关键数字（6,478 / 36,828 / seed 42 / 5 折 80-10-10）、官方 verify 的真实能力（部分目录可跑、需 `--allow-missing-sources`、**不检查前缀规则**）；§4 悬空项改为三条 | 读 `Datasets/README.md`、`task4_architecture_transfer/README.md`、`manage_scaling_indices.py` 源码、GitHub contents API |
| 2026-11-11 深夜 | **task3 定性闭合**：读 `task3_complexity/README.md` 原文，确认它是 *post-hoc stratification*、**不是复杂度留出任务**，`legacy_test_fold` 是 legacy 协议折号而非官方划分；新增【坑八】并把它移出 A-5；任务目录表该行由"待确认"改为"不适用" | 读 `task3_complexity/README.md` 原文 + GitHub contents API（目录仅 2 文件，CSV 3,344,980 B） |
| 2026-11-11 深夜 | **task4 结构与统计口径闭合**：目标只有 **1 个性质（EA）**；补 VIPEA 生成步骤（`process_Vipea_data.py`）、`--polybert-dictionary` 可省略的说明；新增【坑九】**OOD 折固定、不是独立重复、禁止对五折 OOD RMSE 做 t 检验**，只能五折集成后报一个点值——直接影响 D 的置信区间与 H1 判决 | 读 `task4_architecture_transfer/README.md` 全文 |
| 2026-11-11 深夜 | §4 四件事全部标为闭合；末尾两条由"悬空项"改为"已定口径的执行提醒"（`splits/polybench26/` 只放 indices 快照、架构迁移不进 `splits/`） | — |
| 2026-11-12 | **§2.1 最后一条「待确认」闭合**：平台**没有**批量导出/已标注数据接口（只有 *test set, SMILES only* 与展示库），figshare 存档是唯一可脚本化的正规入口；新增【坑十】figshare 对本站出口一律 403（`awselb/2.0`）的证据与三条出路、新增【坑十一】GitHub 走 tarball 而不是 `raw.`（TLS 重协商断连）；**§2.3「待确认」落定**：OpenPoly 用 `data/final_polymer_properties_fromliterature.csv`（741 行 × 32 列），文献计数那张故意不下载 | 执行 A-2：跑 `scripts/00_download.py`、实测 39 个文件清单与平台端点 |
| 2026-11-12 | **39 个 CSV 到位并验收通过**：人工下载后 `--verify-only` 离线核 md5，39/39 与 `supplied_md5` 相等，清单 39 行 / 42,169 / 与附录 A 逐行对齐；新增【坑十二】`Isothermal weight loss time.csv` 表头编码损伤（`IWLT(â„ƒ)`），A-3 做单位判定时不许拿它当温度证据 | 仓库所有者（代 A 执行 A-2） |
| 2026-11-12 | **§2.1 新增「39 个文件的标准表」一行**：把 figshare 原始响应冻成快照 + 派生出 39 行 md5 表，作为任何下载路线的唯一验收标准（D-13）；【坑十】的处理办法同步改成"三条路 + 同一个 `--verify-only` 收尾" | 同上 |
