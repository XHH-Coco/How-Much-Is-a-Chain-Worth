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
| D-13 | 2026-11-12 | 把 figshare `/articles/30917717` 的原始响应冻结成 `data/manifests/polybench_api_article_20261112.json`，并由它派生 **39 行** `data/manifests/polybench_expected_files.csv`（`file_name, task_id, size_bytes, md5, figshare_file_id, download_url`）；**任何人任何路线拿到的 CSV，一律以这张表的 md5 为唯一验收标准**；`--verify-only` 增加「本地采纳」：`data/raw/polybench/` 里 39 份逐份核过 md5 就直接写清单，不出网 | 本机出口被 figshare 403（D-12 之后仍无解），而换网络 / 换人代下 / 平台手动下这三条路线都会产生同一个问题：**凭什么说手上这份文件就是官方那份**。没有冻结表就只能靠人抄 39 个 md5，且 API 不可达时连"要下哪 39 个文件"都说不清（文件名含空格、大小写、下划线，抄错一个下游就缺一个任务）。同时，冻结表让"只有 API 被拦、ndownloader 可达"的机器仍能照常下载，不依赖先问 API | A-2 的完成判据从"脚本跑通"改为"清单 39 行且每行 md5 与表一致"；A-3 的输入合法性以该表为准；**表与线上 API 不一致时脚本报错停下、不许自动跟随**（上游换版本要出新 D 条目）；`--verify-only` 由"只读"变为"可写清单"，但只在 39 份全部核过 md5 时才写 | 仓库所有者 | A（本人执行） |
| D-14 | 2026-11-12 | A-2 验收通过，并把验收中查清的三件事写成口径：① **PolyBench 39 个 CSV 的入库凭据是“md5 相等”**——人工下载 + `--verify-only` 离线逐份核官方 `supplied_md5`（39/39 相等），不要求有机器传输日志；② **42,169 从此称“本地数出来的数”**，出处是 `polybench_download_manifest.csv` 的 `n_rows` 列（可由 `data/raw/polybench/` 重算），不再是 `/api/data-counts` 的复读；③ **A-3 的去重键定为 `(task, SMILES)`**，禁止全局按 SMILES 去重 | 三条都是验收时才发现的事实，不写下来就会被误用：① figshare 对本机出口 403 是长期状态，若坚持“必须脚本直下”才能入库，这项永远完不成，而 md5 相等已经能证明字节身份；② 之前文档里 42,169 只能标“接口复读”，现在本地文件就摆着，再那样写等于自我贬低证据等级；③ 全库 42,169 行只有 15,728 个唯一 SMILES（26,441 条重复），但**每个任务内部零重复**——重复全部来自“同一结构被测了多个任务”，全局去重会把合法的跨任务样本删掉 | A-3 / A-4 的做法；报告与主表里 42,169 的出处写法；D 审计数据入口时的判据；`docs/交付_P1_A2_下载与完整性.md` §9 为执行凭据 | 仓库所有者 | A（本人执行） |
| D-15 | 2026-11-12 | A-3 的目标单位与决策表口径定死：热学（Tg/Tm/Td）统一 `T[K] = T[℃] + 273.15`，模量 / 渗透 / 介电损耗 / 时间四类**按发布单位原样保留并显式记为 identity**（不换 Barrer、不换 MPa）；`unit_decisions.csv` 在分工书八列之上加三列：`task_id`、`decision_kind`、`n_values_converted`，并把 `n_rows_affected` 的语义钉为**该条判定删掉的行数**（换算不删行写 0） | 两件事必须在动手前定，否则下游会各写一套：① 若渗透系数换 Barrer、模量换 MPa，同一性质在不同来源之间就会出现两套数字，而 Λ 本身对单位不敏感（Proposal §4.1）——多换一次只多一个出错点；② 分工书的验收标准③要求「行数差 = `n_rows_affected` 之和」，但八列无法同时表达「换算改写了多少行」与「剔除了多少行」，不拆开就永远算不平 | 所有下游以 `unit_decisions.csv` 为唯一单位口径；A-4 的 YAML `unit` 字段抄 `target_unit`；PolyBench26 / OpenPoly 的同类性质（如 PolyMetriX Tg）**必须用同一条 +273.15**，并新增决策行而不是默默跟随 | 仓库所有者 | A（本人执行） |
| D-16 | 2026-11-12 | 两处「不许猜」的落地：① **`bandgap chain_Egc` 整任务弃用**（3,380 行）——表头只写 `bandgap_chain`，未声明单位，数值 0.0205 ~ 9.863 只在 eV 口径下说得通；② **`property_log` 一律不当标签、不当特征**——A-2/A-3 两次核对证明它逐行等于 `ln(label)`（最大绝对残差 5e−10），不是 log10 | ① 红线原文：“遇到分不清 K 还是 °C 的，不许取看起来合理的那个；要么找到证据，要么整列弃用”。bandgap 只有①种解释符合量级，但“唯一说得通”仍是推断，不是证据；② 若当 log10 用会得到“差 2.3 倍”的假异常（ln10 = 2.302585），若当特征用则把标签直接喂进模型，属于泄漏 | A-4 若要把 bandgap 当预备队，先补该数据集论文/元数据的单位声明并**新增一条决策**解锁；任何任务要用对数目标，由训练侧现算 `log(label)`，不复用发布方的 `property_log` | 仓库所有者 | A（本人执行） |
| D-17 | 2026-11-12 | **Q-01 结案：主表是「8 任务 × 5 臂」（A1–A5）。** `configs/tasks/*.yaml` 的 `arms` 字段按此写死；**本决策不冻结"总共多少次拟合"**，该数归 Q-02（C 定）。同时把 §6/§7.1 的"4 臂、96 经典 + 48 GNN = 144"判为**旧稿残留**：按官方评估单元复算，每臂 = PolyBench 6×3 + PB26 2×5×3 = 48 次，96+48 只等于**3 臂**；§7.4 的 150 = 144 + 6 次无出处；778（D-10）里含的是 150 而不是 144 | 同一份 Proposal 里三处写法互相矛盾（§6 机制表与 §7.1 P1 行写 4 臂，§8 主表与图 6 写 5 臂），而 A-4 一旦冻结，B、C 的配置就按那个臂数写死。选 5 臂的三条独立证据：① 臂的集合定义在 §6（A1–A5，含 A5 链信息）；② §5 的 H3「信息胜过架构」必须有 A5，否则该主张不在主表内；③ §8 主表与 §7.1 分工（C 跑 A1/A2/A5、B 跑 A3/A4）都按五条写。而 144 这个数在任何臂数下都算不出来，只能算作 3 臂的旧稿 | 18 份 `configs/tasks/*.yaml` 的 `arms`；`docs/Task List Snapshot D0.md` §4；B/C 的 run 入口与 `configs/experiments/*.yaml`；C 的 P2 预算表（Q-02）；D 的主表口径与图 6 | 仓库所有者（代 A 执行 A-4） | 待会签（C、D） |
| D-18 | 2026-11-12 | **PolyMetriX Tg 的源数据本来就是开尔文：`target_unit: K`、`unit_status: verified`，禁止再加 273.15。** 同时更正 `docs/交付_P1_A3_标签调和.md` §7 第 4 条那句预设（原文要求"PolyBench26 的 PolyMetriX Tg 也必须用同一条 `T + 273.15`"） | 三条一手证据：① 源表 `Datasets/PSMILES/PolyMetriX/Tg/Tg.csv` 只有两列 `smiles, Tg_K`；② `DATA_PROVENANCE.md` 记该文件是 `labels.Exp_Tg(K)` 列的精确投影，**列名自带 (K)**；③ 实测 7,367 行 / 0 缺失 / 134.15 ~ 768.15 K / 中位 410.15 K（=137 ℃），若当 ℃ 读则最大值 768 ℃ 已超过任何聚合物的分解温度。A-3 当时是"同类性质应同口径换算"的**预设**，不是实测 | `pb26_polymetrix_tg` 的 YAML；主表第 7 行与第 1 行的对比（同物理量换发布版本）——若误加 273 K，两行不可比且不会报错；A-3 交付文档 §7 加勘误；`tests/test_task_configs.py::test_polymetrix_tg_is_already_kelvin` 守着这条 | 仓库所有者（代 A 执行 A-4） | 待会签（A、D） |
| D-19 | 2026-11-12 | **单位有三种状态，不是两种**：`verified`（有证据）/ `unverified`（官方未声明、我们拒绝猜，`target_unit` 必须为 `null`）/ `not_required`（该任务只比较同一性质内的 RMSE，单位不进入 κ、Λ 或排序判决）。据此：`pb26_homopolymer_density` 标 `unverified`（密度单位官方三处文档均未声明）；`pb26_scaling_cp` 与 `pb26_arch_transfer_ea` 标 `not_required` | 密度：`README.md` / `DATA_PROVENANCE.md` / `SCALING_EXPERIMENTS_METHODS.md` 三处均无单位，源 jsonl 的 `outputs.density` 只有数值（首条 `1.1222596573`）。量级像 g/cm³，但"像"不是证据（D-16 已立此规矩）。机制实验：κ 是对 log10 n 的斜率、Λ 是 R² 的函数，二者对单位不敏感——为这种任务猜单位是纯风险无收益 | 18 份 YAML 的 `unit_status`；`src/chainworth/io/tasks.py` 的校验规则（`unverified` 时 `target_unit` 必须为 null）；D 的主表报告（密度列须标"官方未声明单位"）；解锁方式=找到一手证据后**追加决策**，不许就地改 YAML | 仓库所有者（代 A 执行 A-4） | 待会签（A、D） |

---

## 3. 待决（**还没定，不许当已定引用**）

| 编号 | 问题 | 原文冲突 | 该谁定 | 什么时候必须定掉 |
|:--|:--|:--|:--|:--|
| Q-02 | 常规臂的总拟合数到底是 150 还是 144 | §7.1 工时表写「常规基线 ×4（150 次拟合）」，§7.4 又写「四条常规臂（150 次）」；而主表行给的是 **96 经典 + 48 GNN = 144**，差 6 次没有出处。**D-17 已复算：每臂 = PolyBench 6×3 + PB26 2×5×3 = 48 次；144 只等于 3 臂，150 = 144 + 6。所以"4 臂 / 5 臂"下正确值分别是 192 / 240（A1 地板与 A5 注入是否各自独立成格，请 C 明确）** | C（他跑这批） | P2 超参空间与预算表定稿前；C 的 22.0 pH 里含这批 |
| Q-03 | §9 的占用率口径与 §7.1 的分项工时是否自洽 | §9 写「95/128 = 74%，毛 82% / 净 74%」（净需求 95、毛 105、可复用 10），而 §7.1 的分项工时列与这组数没有逐项核对过 | 仓库所有者（他承担写作与汇报，占用率是汇报里的数） | P3 结束前 |
| Q-05 | 主表是否要加进架构迁移（GTRC）那一栏 | Proposal §8 的主表是 8 任务（6 PolyBench + 2 PolyBench26）。架构迁移在 §5 是 H1 的架构子句、在 C-4 单列，**不在主表**。若要进主表，许可栏与署名就跟别人不一样 | 仓库所有者（对外交付的口径） | P3 定表时 |

> **Q-01 已结案为 D-17（2026-11-12）**：主表 = **8 任务 × 5 臂**；`configs/tasks/*.yaml` 的 `arms` 按此写死，快照见 `docs/Task List Snapshot D0.md` §4。**D-17 不冻结拟合次数**，那个数仍归 Q-02。
>
> **Q-04 已结案为 D-10（2026-11-11）**：runs 定 30、5.0 pH 不降、差额写在 OOD 的逐组件加权管线上。`分工_C` 与 Proposal 中英已同步，全队 fit 口径由 535/808 改为 505/778。
