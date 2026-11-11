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

---

## 3. 待决（**还没定，不许当已定引用**）

| 编号 | 问题 | 原文冲突 | 该谁定 | 什么时候必须定掉 |
|:--|:--|:--|:--|:--|
| Q-01 | 主表是「8 任务 × 4 臂」还是「8 任务 × 5 臂」 | Proposal §7.1 与 P1 阶段行写 **4 臂**（96 次经典 + 48 次 GNN）；§8 写 **5 臂** | A 牵头，C 与 D 会签 | A-4 冻结 `docs/Task List Snapshot D0.md` 之前——快照里只能有一种说法 |
| Q-02 | 常规臂的总拟合数到底是 150 还是 144 | §7.1 工时表写「常规基线 ×4（150 次拟合）」，§7.4 又写「四条常规臂（150 次）」；而主表行给的是 **96 经典 + 48 GNN = 144**，差 6 次没有出处 | C（他跑这批） | P2 超参空间与预算表定稿前；C 的 22.0 pH 里含这批 |
| Q-03 | §9 的占用率口径与 §7.1 的分项工时是否自洽 | §9 写「95/128 = 74%，毛 82% / 净 74%」（净需求 95、毛 105、可复用 10），而 §7.1 的分项工时列与这组数没有逐项核对过 | 仓库所有者（他承担写作与汇报，占用率是汇报里的数） | P3 结束前 |
| Q-04 | 架构迁移（C-4）到底是 30 runs 还是 60 runs，5.0 pH 要不要跟着降 | Proposal §7.1 与分工_C 原写「**2 性质** × 2 臂 × 5 折 × 3 种子 = **60 runs**」；但官方 task4 **只做 electron affinity 一个性质**。按 1 性质算，正确值是 1 × 2 × 5 × 3 = **30 runs**。目前 `分工_C` 已按 30 写、Proposal 仍写 60 | C（他跑这批），与 D 会签；工时口径由仓库所有者定 | **P2 排期前**。C 的 22.0 pH 里 5.0 是这项。定法只有两种：① runs 降到 30、工时降到约 2.5–3.0，**腾出的 2.0–2.5 pH 要还给 C 或被 96 pH 口径吸收**；② 维持 5.0 pH，理由写成"OOD 加权集成（组件预测 + molar fractions 合成）与 ID/OOD 双栏报告补足差额"——但这需要在 C 的验收里真写进这两项工作，否则就是虚记 |
| Q-05 | 主表是否要加进架构迁移（GTRC）那一栏 | Proposal §8 的主表是 8 任务（6 PolyBench + 2 PolyBench26）。架构迁移在 §5 是 H1 的架构子句、在 C-4 单列，**不在主表**。若要进主表，许可栏与署名就跟别人不一样 | 仓库所有者（对外交付的口径） | P3 定表时，与 Q-01 一起 |

> Q-01 是**唯一会卡住别人**的一条：A-4 的快照一旦冻结，B 和 C 的配置就按那个臂数写死。**Q-04 同样卡人**：C 的 P2 排期要按 runs 数刷格子，30 与 60 差一倍；D 的架构排序图也等这个数。其余两条可以并行核。
>
> **待决项的现役处理方式**：`分工_C` 已经把 C-4 写成 **30 runs** 并在条目里标注「原记 60，见 Q-04」，这样 C 开工不会按错的数排期；**Proposal 仍写 60，未改**——runs 数属于方案层，不能在定案前单方面改一处。两边不一致是**已知且已登记**的，不是漏改。
