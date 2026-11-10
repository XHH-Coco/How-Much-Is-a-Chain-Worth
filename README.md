# 链值多少？

**How Much Is a Chain Worth?**

**高分子性质预测中的链信息回收率，以及更多数据能否买到它**

> 把一个高分子链压缩成单个重复单元之后，还有多大比例的物性方差变得无法解释 ——
> 而当数据量再涨 10 倍时，这个比例会缩小吗？

`状态：骨架，D0 之前` · 代码 MIT · 仓库内不含任何基准数据

---

## 为什么做这件事

晶体可以用「组分 + 晶胞」描述完，高分子不行：分子量分布、序列、立构规整度、凝聚态形态与加工历史都会推动实测性质，而这些一个都塞不进一条 PSMILES 里。PolyBench 的作者在自己的 limitation 段落里承认了这一点，应对办法是把 39 个任务划成三个暂定的解释层级（tier）。

**限制被承认了，分层也给了，但「缺口到底有多大」从来没有被测量过。** 这就是切口。我们只测两个量：

| 记号 | 是什么 | 取值范围 |
|---|---|---|
| **Λ** | 重复单元解释不掉的那部分方差里，链信息能回收的比例 | [0, 1] |
| **κ** | `dΛ / d log₁₀ n` —— 数据涨 10 倍时，这个比例是否移动 | 实数 |

关于「天花板」这个词：汇报里用到它时，指 `R²_unit`，即重复单元已经解释掉的那份方差。Λ 测的是这道界**之上**的空间。所以 **Λ ≈ 0 的含义是天花板已经封顶，而不是不存在**（详见 `docs/PROTOCOL.md`）。

## 八任务主表

每个 PolyBench 层各两个，外加两个 PolyBench26 数据集。这里每一个 tier 都在基准自己的正文里被点名过，因此没有任何一项需要等那份 156 MB 的 Supporting Information。

| # | 任务 | 基准 | n | 层 | 族 |
|---|---|---|---|---|---|
| 1 | 玻璃化转变温度 | polybench | 8,055 | 1 | 热学 |
| 2 | 熔点 | polybench | 3,796 | 1 | 热学 |
| 3 | 拉伸模量 | polybench | 1,034 | 2 | 力学 |
| 4 | 气体渗透系数 | polybench | 711 | 2 | 输运 |
| 5 | 介电损耗因子 | polybench | 279 | 3 | 介电损耗 |
| 6 | 等温热失重时间 | polybench | 228 | 3 | 降解 |
| 7 | PolyMetriX Tg | polybench26 | — | — | 热学 |
| 8 | homopolymer density | polybench26 | — | — | 密度 |

另外还有一个 **OpenPoly 扫描**，在 n = 30 / 60 / 120 / 240 / 全量 上跑 Tg（443）、Dielectric Constant Total（295）、Tm（168）—— 只服务 H4，从不进主表。选任务的规则、替补名单、以及每一行为什么选它：`docs/PROTOCOL.md` 与 `data/manifests/candidate_tasks.csv`。

## 五条臂

| 臂 | 内容 | 环境 | GPU |
|---|---|---|---|
| A1 floor | dummy mean + kNN | Core | 否 |
| A2 conventional | RDKit 描述符 / Morgan FP + XGBoost、RandomForest | Core | 否 |
| A3 graph | 官方 `polymer_chemprop` | GNN | **是** |
| A4 language model | polyBERT 官方 embedding | Polybert | 是 |
| A5 chain information | 重复单元计数特征、wPSMILES 加权、多单元拼接 | Core | 否 |

A5 是 A2 的改造，不是第六个模型族。它存在的目的只有一个：回答 H3 —— **信息**能否胜过**架构**。

## 仓库结构

本节是仓库结构的唯一定稿。**任何新增、删除、重命名、移动文件的操作，都必须让本节跟着变**，判定方式见本节末尾。

### 目录树

<!-- TREE:BEGIN -->
```
How-Much-Is-a-Chain-Worth/
├─ .github/
│  └─ workflows/
│     └─ ci.yml                  ← push/PR 跑结构自检与单测；env-pins 需手动触发
├─ configs/                      ← 协议的可执行形态：任务、臂、实验全部是 JSON
│  ├─ arms/                      ← 五条臂 A1–A5，一条一个文件
│  │  ├─ a1_floor.json
│  │  ├─ a2_conventional.json
│  │  ├─ a3_graph.json
│  │  ├─ a4_polybert.json
│  │  └─ a5_chain.json
│  ├─ experiments/               ← 主表 / scaling / 架构迁移 / OpenPoly 扫描
│  │  ├─ arch_transfer.json
│  │  ├─ main_table.json
│  │  ├─ openpoly_scan.json
│  │  └─ scaling.json
│  ├─ tasks/                     ← 11 个任务：PolyBench 6 + PolyBench26 2 + OpenPoly 扫描 3
│  │  ├─ openpoly_dielectric_total.json
│  │  ├─ openpoly_tg.json
│  │  ├─ openpoly_tm.json
│  │  ├─ pb26_homopolymer_density.json
│  │  ├─ pb26_polymetrix_tg.json
│  │  ├─ pb_t1_tg.json
│  │  ├─ pb_t1_tm.json
│  │  ├─ pb_t2_modulus.json
│  │  ├─ pb_t2_permeability.json
│  │  ├─ pb_t3_dielectric_loss.json
│  │  └─ pb_t3_iwlt.json
│  └─ README.md                  ← 为什么用 JSON 而不是 YAML（ADR-001）
├─ data/                         ← 只放指针与数字，不放数据本体
│  ├─ manifests/                 ← 入库：任务表、候选表、摘要 ——「用的是哪份数据」的物证
│  │  ├─ candidate_tasks.csv     ← §3.1 的 A–E 表：8 主 + 3 扫描 + 5 替补
│  │  ├─ MANIFEST.lock.json      ← manifest 自身的哈希锁
│  │  ├─ openpoly_scan.csv       ← 443 / 295 / 168 三个性质 × 五档 n
│  │  ├─ polybench_tasks.csv     ← 39 行，加总 42,169，测试守着这两个数
│  │  └─ sha256.txt              ← 下载件摘要，D0 填充；现为占位
│  └─ README.md                  ← 数据卡：四资源对照、741 与 3,985、许可状态
├─ docs/                         ← 预注册与裁决，都带版本与理由
│  ├─ APPENDIX_A.md              ← 39 个 PolyBench 任务全表与层号状态
│  ├─ DECISIONS.md               ← ADR：为什么 JSON、为什么 3.10、为什么 PolyBench 不做 k 折
│  ├─ PROTOCOL.md                ← 预注册：任务、臂、假设、证伪判据；D0 冻结后才算数
│  ├─ PROVENANCE.md              ← 每个数字追到文件与行，含三个曾经写错的
│  ├─ RUNBOOK.md                 ← D0 七项清单、闸门、四个角色、出问题怎么办
│  └─ tree_notes.json            ← 上面那棵树的注释来源；路径失效即报错
├─ environment/                  ← 三个环境的声明与校验；装包发生在 D0
│  ├─ lock/                      ← conda-lock 产物落这里并入库；现为空
│  │  └─ .gitkeep
│  ├─ b2-core.yml                ← 直接依赖 + 上下界；全量锁定在 lock/
│  ├─ b2-gnn.yml                 ← RDKit 与 torch 分居两个环境，避开 Windows libgomp 冲突
│  ├─ b2-polybert.yml            ← transformers 卡死；7.3 GB 权重缓存可独立删除
│  ├─ pins.json                  ← 版本的唯一权威，verify_env.py 与之双向比对
│  ├─ README.md                  ← 三个环境为什么必须分开
│  ├─ requirements-core.txt      ← pip 侧精确版本，供 conda-lock 之外的复核
│  ├─ requirements-gnn.txt       ← torch 只从 download.pytorch.org 装，不走 conda
│  ├─ requirements-polybert.txt  ← 同上
│  └─ verify_env.py              ← 既防低也防高：3.10 下 pip -U 不报错，只会让数变（ADR-004）
├─ results/                      ← 证据：小体积、可追溯到一次具体运行
│  ├─ aggregate/                 ← 跨 run 聚合：Λ、κ 与 CI
│  │  └─ .gitkeep
│  ├─ env_fingerprints/          ← 每 run 一份环境指纹（含驱动版本）
│  │  └─ .gitkeep
│  ├─ figures/                   ← 图 1–5 的产物
│  │  └─ .gitkeep
│  ├─ raw_runs/                  ← 每 run 一个 JSON —— 结论的审计线索
│  │  ├─ .gitkeep
│  │  └─ _schema.json            ← 记录格式：没有 CI 的格子不许进表
│  └─ README.md                  ← 什么进 results/、什么不进
├─ scripts/                      ← 按编号排列的流水线；03–05 仍是 stub
│  ├─ 00_fetch_data.sh           ← 下载四个基准；需要三个环境变量，不编造 URL
│  ├─ 01_build_manifests.py      ← 重建 manifests 并算摘要（--hash）
│  ├─ 02_build_splits.py         ← 复制官方索引，绝不自己重新生成
│  ├─ 03_run_main_table.sh       ← 八任务主表
│  ├─ 04_run_scaling.sh          ← n 档扫描，算 κ 的来源
│  ├─ 05_run_openpoly_scan.sh    ← OpenPoly 小样本扫描，只服务 H4
│  ├─ 06_aggregate.py            ← Λ 与 κ 只在这里算，单个 run 自己不算
│  ├─ 07_make_figures.sh         ← 图 1–5
│  ├─ check_repo.py              ← 仓库自检，含 README 结构树是否过期
│  └─ gen_tree.py                ← 重画 README 里那棵树（make tree）
├─ splits/                       ← 冻结的划分索引 —— 比 src/ 更不可动
│  ├─ openpoly/                  ← 自建分层扫描，随附生成脚本的 hash；现为空
│  │  └─ .gitkeep
│  ├─ polybench/                 ← 官方固定 70/30 单划分；现为空，D0 填充
│  │  └─ .gitkeep
│  ├─ polybench26/               ← 官方 5 折；现为空，D0 填充
│  │  └─ .gitkeep
│  ├─ README.md                  ← 三类划分的性质差异，一页说清
│  └─ SCHEMA.json                ← 划分文件的字段约束
├─ src/
│  └─ b2/                        ← 零第三方依赖的协议层：装包之前就能全绿
│     ├─ analysis/               ← Λ、κ、CI、排序一致性
│     │  ├─ __init__.py
│     │  ├─ lambda_kappa.py      ← Λ 与 κ 的唯一计算处
│     │  └─ ordering.py          ← H1 的排序一致性（Spearman）
│     ├─ arms/                   ← A1–A5 的实现；P1，需要 RDKit
│     │  └─ __init__.py
│     ├─ chem/                   ← PSMILES 解析、描述符、指纹；P1
│     │  └─ __init__.py
│     ├─ figures/                ← 图 1–5 的绘制
│     │  └─ __init__.py
│     ├─ io/                     ← 读入与内容哈希
│     │  ├─ __init__.py
│     │  └─ hash.py
│     ├─ metrics/                ← 回归指标与区间估计
│     │  ├─ __init__.py
│     │  └─ bootstrap.py         ← 百分位 bootstrap：CI 的唯一来源
│     ├─ protocol/               ← 划分校验与运行前断言
│     │  ├─ __init__.py
│     │  ├─ assertions.py        ← 跑之前该拒绝什么
│     │  └─ splits.py            ← 读划分、查泄漏、验身份
│     ├─ __init__.py
│     ├─ cli.py                  ← 命令行入口：fingerprint 等
│     ├─ config.py               ← 读取并校验 configs/，非法组合直接拒
│     └─ py.typed
├─ tests/                        ← stdlib unittest：没有 pytest 也必须全绿
│  ├─ fixtures/
│  │  └─ tiny/                   ← 12 条合成记录：没有基准数据也能跑通五臂
│  │     ├─ records.json         ← 合成记录
│  │     └─ split.json           ← 与之配套的划分
│  ├─ __init__.py
│  ├─ test_bootstrap.py          ← CI 覆盖率的对拍
│  ├─ test_config.py             ← 非法任务组合必须被拒（如 PolyBench 用 5 折）
│  ├─ test_lambda_kappa.py       ← Λ 的边界：R²_unit = 1、负值、方向
│  ├─ test_manifests.py          ← 39 行与 42,169 的加总守卫
│  ├─ test_readme_tree.py        ← README 结构树与磁盘一致
│  ├─ test_repo_layout.py        ← 必需的文件与目录
│  └─ test_splits_schema.py      ← 划分 schema 与 fixtures 的对应
├─ .gitattributes                ← 一律 LF：CRLF 的 .sh 会让每条 recipe 报 command not found
├─ .gitignore                    ← 只按目录忽略，绝不按扩展名（ADR-005）
├─ CITATION.cff                  ← 引用信息：本仓库 + 四个基准各自的 DOI
├─ DATA_LICENSES.md              ← 四个撞名资源的许可对照
├─ LICENSE                       ← MIT，仅覆盖代码；数据沿用各基准自己的许可
├─ Makefile                      ← check / test / tree / env-verify 的快捷入口
├─ pyproject.toml                ← src layout，单包 b2
└─ README.md                     ← 本文件：结构说明的唯一定稿
```
<!-- TREE:END -->

### 五个结构性判断

树本身只说明「有什么」，下面五条说明「为什么这样分」—— 它们才是这个布局不可随意改动的理由。

1. **`splits/` 比 `src/` 更不可动。** 代码可以重写，协议不能漂。PolyBench 是官方固定 70/30 单划分，PolyBench26 是官方 5 折，OpenPoly 是我们自建的分层扫描；三种东西混在一起最容易出错，所以各占一个目录、各有 README、各有摘要，并由 `tests/test_splits_schema.py` 断言盘上文件与记录逐字节一致。
2. **`data/manifests/` 入库，`data/raw/` 不入库。** 数据本体几百 MB 到几 GB，且随时可从 DOI 取回；但「用的是哪一份」必须留在仓库里。§2.3 那个 50 倍的虚高（58,755 是文献计数，不是记录数）就是靠 manifest 的行数与摘要挡住的。
3. **`results/raw_runs/` 入库，模型权重不入库。** 每个 run 一个 JSON，1–5 KB，全部加起来不到 2 MB，但它是「结论可追溯」的物理载体；权重动辄几百 MB，且完全可由 config + seed 重建，没有入库价值。
4. **协议全部用 JSON，不用 YAML。** `1e3`、`on`、`no` 在 YAML 里会静默变成别的类型，而协议文件一旦被静默重写就再也对不上（ADR-001）。
5. **`docs/` 分成两份，不合并。** `PROTOCOL.md` 是预注册（我们打算做什么、什么情况下算被证伪），`DECISIONS.md` 是 ADR（为什么当时这么定）。前者 D0 之后只许追加，后者只许新增；混成一份就没法说明「这个决定是在看到数据之前做的」。

### 结构怎么保持同步

树不是手写的，是生成的：

```bash
make tree                        # 重画 README 里那棵树
python scripts/check_repo.py     # 树与磁盘不一致会直接 FAIL
python -m unittest discover -s tests -t .   # 同样会红
```

三处把关：`scripts/gen_tree.py` 负责生成，`scripts/check_repo.py` 在 CI 里比对，`tests/test_readme_tree.py` 让 `make test` 也拦住。文件清单来自 `git ls-files --cached --others --exclude-standard`，所以树里只出现 clone 之后真能拿到的东西 —— 被 `.gitignore` 忽略的数据与权重不会混进来。

每行右侧的注释来自 `docs/tree_notes.json`。**注释的键是路径**：文件一改名，旧注释立刻被报为失效，不会悄悄留在树里指着一个不存在的文件。

## 复现

以下命令既不需要基准数据，也不需要标准库以外的任何包：

```bash
python scripts/check_repo.py                  # 布局、配置、manifest 算术、README 树
python -m unittest discover -s tests -t .     # 55 个单测，纯 stdlib
python environment/verify_env.py              # 三个 conda 环境与 pins.json 比对
```

数据到位（见 `scripts/00_fetch_data.sh`）、环境装好之后：

```bash
python scripts/01_build_manifests.py --hash   # 为每份下载件算摘要
python scripts/02_build_splits.py             # 复制官方索引，绝不自己生成
bash   scripts/03_run_main_table.sh           # 每 run 一个 JSON 落到 results/raw_runs/
python scripts/06_aggregate.py                # Λ 与 κ；单个 run 自己不算
```

**当前状态**：协议层、单测、配置已完整且全绿。五条臂（`src/b2/arms/`）还没实现 —— 那是 P1，需要 RDKit，而 RDKit 目前是刻意不装的。

## 结果

还没有。这一节在 P2 才填，也不允许提前填。填的时候必须带上：Λ 与 κ **及其 bootstrap 区间**、方向与 H1 一致的任务数、以及与 PolyBench leaderboard 之间诚实的差距。没有区间的格子不许进表。

## 什么情况会证明我们错了

| 如果 | 那么 |
|---|---|
| κ 的 95% CI 显著低于 0，且 Λ 从最小到最大官方 n 减半 | H2 被证伪：数据**确实**买到了链信息 |
| Λ 的排序给出 Spearman ρ < 0.5，或在两个层之间反转 | H1 被证伪：分层并不对应可回收的信息量 |
| 换架构的收益超过显式链信息 | H3 被证伪：杠杆是表征，不是信息 |
| n\* 的移动与标签离散度无关 | H4 被证伪：n\* 不是噪声驱动的 |

我们报告证伪，而不是把参数调到「看起来对」。**Λ 处处 ≈ 0 本身也是一个结果**：天花板已经封顶，重复单元范式成立，而这个领域靠堆数据的策略买到的是插值，不是信息。

## 数据与许可

仓库不提交任何基准数据。`data/raw/` 在 `.gitignore` 里；`data/manifests/` 保存行数与摘要，用来证明我们用的是哪一份文件。存在四个容易撞名的资源 —— **OpenPoly**（741 结构 / 3,985 标注 / 26 性质）、**OPoly26**（Meta，657 万 DFT）、**PolyBench26**（LLNL，约 25 万数据点）、以及一个无关的 **PolyBench**（约 12.5 万条 LLM 问答对）。引用时只报 DOI 与规模，不要只报名字。逐个的细节与许可状态：`DATA_LICENSES.md` 与 `data/README.md`。

## 三个环境

三个环境统一 Python 3.10.14。3.10 已经过了上游 EOL（2026-10-31），我们仍然留在 3.10，因为已发表的基线是在这一代栈上评的，而 `environment/pins.json` 里每一个 pin 都是**最后一个**还提供 cp310 wheel 的版本。RDKit 与 PyTorch 塞进同一个环境会撞上 Windows 上经典的 `libgomp` 崩溃，所以必须拆开。torch 一律从 `download.pytorch.org` 装，不走 conda。

`pins.json` 是唯一权威；`verify_env.py` 对**版本过高**同样报错，因为在 3.10 上 `pip install -U` 不会失败，只会让数变（ADR-004）。

## 团队分工

| 角色 | 负责 | 占比 |
|---|---|---|
| A — 数据与合规 | 基准接入、口径统一、划分正确性、审计 | 28% |
| B — 建模 | 五条臂、GPU 扫描、机制实验 | 32% |
| C — 分析 | Λ / κ、区间、图、三条结论 | 28% |
| D — 写作与汇报 | slides、Q&A 准备、排练 | 12% |

## 引用

见 `CITATION.cff`。基准也要一并引用：PolyBench（figshare `10.6084/m9.figshare.30917717`）、PolyBench26 与 OpenPoly 各按各自的 DOI。
