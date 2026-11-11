# 决策记录（DECISIONS）

> **谁改口径谁登记。** README 第九节规定：冻结物不得原地改，要改就出新版本文件，并在这里记一条（为什么改、影响哪些 run）。**没写进这里的口径变更，视为没发生过**——后人只会看到两个互相矛盾的数字，查不出是哪一个被改过。
>
> 这份文件同时收两种东西：**已定**（D-xx，只追加不删除）和**待决**（Q-xx，定了之后转成 D-xx）。待决项不许被当成已定项引用。

---

## 1. 怎么记

| 要求 | 说明 |
|:--|:--|
| 编号 | `D-01` 起三位递增，永不复用、永不重排。删掉一条决策等于伪造历史 |
| 日期 | 实际做出决定的那天，不是写下来的那天 |
| 决策 | 一句话说清"改成了什么"，可执行、可检验 |
| 触发 / 原因 | 为什么非改不可。写"文档自相矛盾"要指出哪两处 |
| 影响范围 | 会波及哪些任务编号、哪些已产出的数字、哪些人的验收标准 |
| 定的人 / 会签 | 谁拍板、谁复核。**口头同意不算会签**，会签人要在提交或 `#data` 里留下痕迹 |
| 追加不覆盖 | 决策被推翻时，追加一条新的 D-xx 并写明"取代 D-yy"，老那条留着 |

---

## 2. 已定

| 编号 | 日期 | 决策 | 触发 / 原因 | 影响范围 | 定的人 | 会签 |
|:--|:--|:--|:--|:--|:--|:--|
| D-01 | 2026-11-11 | 装环境以 `lock/` 精确重放为准，`environment/*.yml` 降为声明与说明书 | README §一第 1 步标题写"严格按 yml"，同节备注却写"以 lock 为准"，新组员照标题会走错 | 三套环境的重装流程；分工_A A-1 的"怎么做" | 仓库所有者 | 待会签 |
| D-02 | 2026-11-11 | 基准下载只做 3 个（PolyBench、PolyBench26、OpenPoly）；第四个 MoleculeNet / PCQM4Mv2 仅登记、不下载、不建 manifest | Proposal §2.1 明确第四个资源"不承重，仅作跨层级背景"，而分工_A A-2 原写"四基准下载" | 分工_A A-2 的体量与验收；`data/manifests/SOURCES.md` §3 | 仓库所有者 | 待会签（A） |
| D-03 | 2026-11-11 | 下载清单统一为 `<bench>_download_manifest.csv`；OpenPoly 的性质计数改名 `openpoly_property_counts.csv` | A-2 与 A-7 原本都产出 `openpoly_manifest.csv`，列完全不同，后写的会覆盖前写的证据 | 分工_A A-2 / A-7 / §3 / §8；README §4.2 `data/manifests/` 行 | 仓库所有者 | 待会签（A） |
| D-04 | 2026-11-11 | `data/processed/` 保持不入库；结构与描述符的交付物是**脚本 + manifest**，不是 parquet | `.gitignore` 把 `data/processed/*` 整目录忽略（README §4.3 同口径），但分工_A §3 把两个 parquet 列进了"交付物"，B 会等一个不会出现在仓库里的文件 | B 获取输入的方式（跑 `scripts/02_features.py` 重建）；分工_A A-6 的交付口径 | 仓库所有者 | 待会签（A、B） |
| D-05 | 2026-11-11 | `splits/openpoly/` 归 A-5，计入其 4.0 pH，不单列任务项 | README 第九节第 6 行把 `splits/openpoly/` 算作"A 的官方划分快照"，但分工_A A-5 只列 PolyBench 与 PolyBench26，目录空着没人填 | 分工_A A-5 的范围与工时构成；C 的 OpenPoly 经典臂 | 仓库所有者 | 待会签（A、C） |
| D-06 | 2026-11-11 | A-5 的快照对象由官方 `scaling_splits/` 改为 **`Datasets/scaling_indices/`**，并按「生成 canonical PSMILES → `canonical_sha256` 校验 → 拷 `seed_42/` → 官方 `verify` → 自测前缀规则」五步执行 | `Datasets/scaling_splits/` 被官方 gitignore，只存在于跑过生成脚本的本地机器，**仓库里没有可取的实体**；Proposal §2.1 / §7.5 与分工_A A-5 都按这个不存在的发布物写了工序。等价且官方入库的替代物是 `scaling_indices/`（整数行号 + PSMILES sha256） | 分工_A A-5 全条、§3 清单、§8 自检；分工_C C-3 的输入与验收①；README §4.2 `splits/` 行与第九节冻结登记第 6 行；`CODE_STYLE.md` 的 `SCALING_TIERS` 注释；Proposal 中英各 4 处措辞 + 版本行勘误 | 仓库所有者 | 待会签（A、B、C） |
| D-07 | 2026-11-11 | `task3_complexity`（重复单元复杂度）**不纳入 A-5，不占工时，不建 `splits/polybench26/complexity/`** | 读官方 `task3_complexity/README.md` 原文：*Task 3 is a post-hoc stratification of the legacy alternating refractive-index test predictions, **not** a complexity-held-out training task.* 它不是留出任务，没有"复杂度划分"可快照；表里的 `legacy_test_fold` 是 **legacy 协议**的折号，不是 PB26 主表协议的折号。且我们主表没有 refractive_index | 分工_A A-5 的产出路径（原 `splits/polybench26/<task>/fold_*.json` 中的复杂度部分取消）；`data/manifests/SOURCES.md` §2.2 任务目录表与【坑八】 | 仓库所有者 | 待会签（A） |
| D-08 | 2026-11-11 | `task4_architecture_transfer` **不进 `splits/`**：留出集官方不发布，只发布物化脚本；C-4 自己本地生成、自己记 commit。结果表按 **GTRC** 标许可 | 官方 README 说明该目录是 *reproducible architecture-held-out materialization workflow*，产出落在 gitignored 的 `generated/shared_splits/`；训练源为 polyVERSE EA，属 GTRC 而非主表的 CC BY 4.0 | 分工_A A-5 产出（`splits/polybench26/arch/` 留空并注明）；分工_C C-4 的输入与许可义务；D 的结论表署名与分发条款 | 仓库所有者 | 待会签（A、C、D） |
| D-09 | 2026-11-11 | 架构迁移的 **OOD 指标用五折集成后报一个点值**（5 个预测向量按聚合物取平均，再算 random / block / combined 各一个 RMSE/MAE/MSE）；**禁止**对五折 OOD RMSE 做 t 检验、**禁止**给"均值 ± 标准误"。ID（交替 / 均聚）才允许按折汇总均值与总体标准差 | 官方 `task4_architecture_transfer/README.md` 原话：*The OOD holdout is fixed across folds, so fold metrics are not independent replicates. ... Do not apply a t-test to the five fold-level OOD RMSEs.* 折固定 ≠ 独立重复 | 分工_C C-4 验收⑤与产出（必须留 5 折预测向量）；分工_D 的 CI 构造与 H1 判决；D 的五图里任何"OOD ± 误差棒" | 仓库所有者（依官方协议，非自选） | 待会签（C、D） |
| D-10 | 2026-11-11 | 架构迁移（C-4）定为 **1 性质（electron affinity）× 2 臂 × 5 折 × 3 种子 = 30 runs**，**5.0 pH 不降**；省下的 2 pH 用在 OOD 的**逐组件预测 + 摩尔分数加权合成**上（官方 `summarize_weighted_predictions.py`）。全队 fit 口径同步由 535/808 改为 **505/778** | Q-04 的两个问题分别定：① runs 数不是口味问题——官方 task4 只做 electron affinity**一个**性质，"2 性质 × ... = 60" 按官方结构算不出来，30 是唯一正确值；② 工时不能跟着降，因为原计划漏了一整条管线：官方 OOD 评估不是直接预测 36,828 行，而是先对每个唯一组分 PSMILES 各预测一次、再按 `pred_a·f_a + pred_b·f_b` 合成，脚本对"组分缺预测""摩尔分数不合 1""psmiles 重复"三种情况直接抛错。这条管线原来没写进 C-4，60 runs 的旧口径把它掩盖了 | 分工_C C-4 全条（怎么做 / 产出 / 验收⑤⑥）；Proposal 中英 §6 机制实验表、§7.1 P2 行与 §7.3 段落、§7.1 的 808→778；分工 B/C/D 与 README 的 535→505 | 仓库所有者 | 待会签（C、D） |
| D-11 | 2026-11-11 | 三套环境由 **v1 升 v2**：三个 `environment/*.yml` 与三个 `lock/*.pip.txt` 各加 `pyarrow==17.0.0`；**conda 层字节不动**；`SHA256SUMS` 与三份指纹同步重算 | pandas 2.2.2 不自带 parquet 引擎，`pyarrow` / `fastparquet` 在三环境的 pip 层与 conda 层命中均为 0；实跑 `pd.DataFrame(...).to_parquet()` 报 `ImportError: Unable to find a usable engine`。而 parquet 是 A 的 `harmonised/structures/descriptors`、B/C 的 `predictions/*.parquet`、D 的 bootstrap 输入，**A-3 第一步就会停**。改 CSV 要动 20 多处文档，且 `predictions` 是三方交接格式，CSV 浮点往返不保证逐位一致 | 三个 `environment/*.yml`、三个 `lock/*.pip.txt`、`lock/SHA256SUMS`、三份 `results/env_fingerprints/env_*.json`、README §4.1/§4.2/§九第 1 行、`docs/交付_P0_环境.md` §8；**其余组员需按新锁重装一次**（现在还没人装过，此刻改成本最低） | 仓库所有者 | A（本人执行） |
| D-12 | 2026-11-12 | A-2 的下载通道与清单口径定死三条：① **PolyBench 只走 figshare API**（枚举 39 文件 + 逐份校验 `supplied_md5`），**不许**用平台的展示库或 *test set, SMILES only* 接口顶替；② **两个 GitHub 基准走 `codeload` tarball + 白名单解包**，不用 `raw.githubusercontent.com`；③ 清单在分工书那六列之后**追加 `bytes` 与 `n_rows_basis` 两列**，且 `--limit` 只写 `*_download_manifest.partial.csv` | ① 平台实测没有批量导出（`/api/dataset/test-download` 只给 `polymer_name,SMILES` 两列、`/api/polymers-data` 是每页 10 条的展示库），用它顶替会让 sha256 与发布方对不上；② `raw.githubusercontent.com` 从本机在传输中被打断（TLS 重协商，`curl` 退出码 56 / 20 秒），tarball 一个请求取全且稳定；③ 分工书 A-2 的六列里没有列能表达"文件多大"（截断时唯一能早发现的数），而 `SOURCES.md` §2.2 明确要求注明"82,990 是哪一种条数"；④ 限量跑若写正名清单，3 行会静静取代 39 行，下游分不出来 | A-2 的产出格式与脚本 `scripts/00_download.py`；A-3 / A-7 与 D 的入口审计**读清单时多两列**（下游只按列名取值即可，不受影响）；`docs/交付_P1_A2_下载与完整性.md` §2；`.gitignore` 新增 `data/manifests/*.partial.csv` | 仓库所有者 | A（本人执行 A-2） |

---

## 3. 待决（**还没定，不许当已定引用**）

| 编号 | 问题 | 原文冲突 | 该谁定 | 什么时候必须定掉 |
|:--|:--|:--|:--|:--|
| Q-01 | 主表是「8 任务 × 4 臂」还是「8 任务 × 5 臂」 | Proposal §7.1 与 P1 阶段行写 **4 臂**（96 次经典 + 48 次 GNN）；§8 写 **5 臂** | A 牵头，C 与 D 会签 | A-4 冻结 `docs/Task List Snapshot D0.md` 之前——快照里只能有一种说法 |
| Q-02 | 常规臂的总拟合数到底是 150 还是 144 | §7.1 工时表写「常规基线 ×4（150 次拟合）」，§7.4 又写「四条常规臂（150 次）」；而主表行给的是 **96 经典 + 48 GNN = 144**，差 6 次没有出处 | C（他跑这批） | P2 超参空间与预算表定稿前；C 的 22.0 pH 里含这批 |
| Q-03 | §9 的占用率口径与 §7.1 的分项工时是否自洽 | §9 写「95/128 = 74%，毛 82% / 净 74%」（净需求 95、毛 105、可复用 10），而 §7.1 的分项工时列与这组数没有逐项核对过 | 仓库所有者（他承担写作与汇报，占用率是汇报里的数） | P3 结束前 |
| Q-05 | 主表是否要加进架构迁移（GTRC）那一栏 | Proposal §8 的主表是 8 任务（6 PolyBench + 2 PolyBench26）。架构迁移在 §5 是 H1 的架构子句、在 C-4 单列，**不在主表**。若要进主表，许可栏与署名就跟别人不一样 | 仓库所有者（对外交付的口径） | P3 定表时，与 Q-01 一起 |

> Q-01 是**唯一会卡住别人**的一条：A-4 的快照一旦冻结，B 和 C 的配置就按那个臂数写死。其余两条可以并行核。
>
> **Q-04 已结案为 D-10（2026-11-11）**：runs 定 30、5.0 pH 不降、差额写在 OOD 的逐组件加权管线上。`分工_C` 与 Proposal 中英已同步，全队 fit 口径由 535/808 改为 505/778。
