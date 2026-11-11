# 链值多少？ / How Much Is a Chain Worth?

> 把一条高分子链压缩成一个重复单元之后，还有多少性质方差解释不了？这个比例会随着数据量增加十倍而缩小吗？

`status: P0 已完成（环境已冻结）` · 代码 MIT · 仓库内不含任何基准数据

---

## 一、第一次克隆之后：按顺序做完这七件事

**先读这一节，再动手。** 这是一个空骨架：没有数据、没有代码、没有配置。下面的顺序是刻意的——跳过第 1 步会在装包时炸，跳过第 3 步会在跑完之后返工。

### 1. 装环境：按 `lock/` 精确重放，`environment/*.yml` 只用来读

```bash
# 三个环境各走一遍，把 gnn 换成 core / polybert
conda create -n DSC6020_Project_GNN --file lock/gnn.conda.txt   # 1) conda 层：按 URL 重放，不重新求解
conda activate DSC6020_Project_GNN
pip install -r lock/gnn.pip.txt                                 # 2) pip 层：索引地址在文件头，pip 会自动带上
sha256sum -c lock/SHA256SUMS                                    # 3) 校验锁文件自身完好，6 个文件全 OK 才继续
python -V                                                       # Core/Polybert 3.10.x；GNN 3.11.x（见下）
```

- **为什么不用 `environment/*.yml` 装。** `.yml` 只写直接依赖（`gnn.yml` 13 条），`lock/` 连间接依赖一起写（`gnn` 环境 = 18 个 conda 构建产物 + 70 个 pip 包）。按 `.yml` 装，conda 求解器会**现场重算一遍**，间接依赖（`torchmetrics`、`sympy`、`filelock` 这些）跟着当时的索引走，结果是"能跑，但和别人不是同一套版本"，后面主表的数字就对不上。
- **`.yml` 的用途是读，不是装。** 它头部注释解释了每条钉版的原因：为什么 GNN 是 3.11、为什么必须 `setuptools==75.8.0`、为什么索引只能用 `--index-url cu121`。要动版本之前先读那里。
- **重放有两个前提**：① 同操作系统——`lock/*.conda.txt` 头部写死 `platform: win-64`，换成 Linux/Mac 第 1 步就会失败；② 网络能到 `mirrors.tuna.tsinghua.edu.cn`，conda 锁里 18 个 URL 全指这个镜像。
- **退路只有一条**：`lock/` 重放失败时才退回 `conda env update -n <env> -f environment/<env>.yml --prune`。这条路装出来的环境**不保证与锁一致**，装完必须重新导出锁（`conda list --explicit` + `pip freeze`）并告知分工 A 登记，否则你和别人就不是同一套版本了。
- **Python 版本：Core / Polybert 用 3.10.14，GNN 用 3.11.18。** 不是随手混的：`chemprop 2.x` 强制要求 ≥3.11（PyPI 上 2.0.0–2.3.1 全部如此），而 1.x 只支持 <3.9，所以 **3.10 上不存在任何可安装的 chemprop**，而 A3 臂正是建立在它上面。三套包版本各自按对应 Python 的 wheel 兼容性钉死；谁单独升大版本，别人拿到的 `lock` 就对他失效。（3.10 已于 2026-10-31 EOL，目前是已知取舍，记在 `environment/*.yml` 的头注释里。）
- **三个环境不许互相 install。** 把 RDKit 塞进 GNN 环境，`libgomp` 会让训练半路崩，而且报错与化学库毫无关系，极难查。
- **不要往系统 Python 里装，也不要新建 `requirements.txt`。** 本仓库只有 conda 一套声明。
- **`environment/` 三份已冻结**（第九节第 1 行），不要自己改；发现它和 `lock/` 不一致时**找分工 A**，不要自行"修"其中一边。装法、验收命令与已知局限见 [`docs/交付_P0_环境.md`](docs/交付_P0_环境.md)。

### 2. 按这个顺序读文档

| 顺序 | 读什么 | 解决什么问题 |
|---:|---|---|
| 1 | 本文件 | 阶段门、目录归属、硬规则、冻结清单、代码规范 |
| 2 | `docs/Proposal_HowMuchIsAChainWorth_v1.3.md` | 研究问题、五个臂、假设与判据（**英文版为准**，中文版逐节对照） |
| 3 | `docs/分工_X_*.md` 你自己那份 | 每条任务、验收标准、工时 |
| 4 | 自己那份的第 4 节 | 没有 GPU 时交什么、由谁代跑 |

四份分工文档各自独立可读，不需要先读别人的。但**不要只看自己那份就开工**：判据（Λ、κ、H4 的交叉点）写在 Proposal 里，你的验收标准是从它倒推出来的。

### 3. 认领任务编号，而不仅是"角色"

分工文档第 2 节是任务清单，每条形如 `A-1`…`D-10`，写了四件事：**拿什么 → 怎么做 → 产出 → 验收标准**。交上来的东西对不上"验收标准"那一栏，就是不收。

同时认清你的**只读输入**：`data/raw/`、`splits/`、`configs/tasks/` 里被冻结的部分（见第九节），对你都是只读的。缺数据找 A，缺划分找 A，缺配置找你的上游。冻结物谁都不得改。

### 4. 数据自己下载，不要从别人机器上拷

仓库永远不含基准数据（第五节硬规则 1）。按分工文档里 P1 的步骤下载，落到 `data/raw/`，算 sha256 写进 `data/manifests/`。**拷贝来的文件没有 sha256，等于没有来源，一律退回。**

**P1 起的下载不要手点**：`python scripts/00_download.py --source polybench26|openpoly|polybench` 就是这一步的唯一入口（`--dry-run` 空跑、`--limit` 限量、`--verify-only` 只校验清单）。它逐份校验发布方公布的 md5/sha256、数条数、写清单，并拒绝在哈希对不上时落盘。当前进度与 PolyBench 的网络阻塞见 [`docs/交付_P1_A2_下载与完整性.md`](docs/交付_P1_A2_下载与完整性.md)。

### 5. 配好 git，按格式提交

```bash
git config user.name  "你的名字"
git config user.email "你的邮箱"
git remote -v          # 必须是 XHH-Coco/How-Much-Is-a-Chain-Worth
```

提交信息三段式（第十节）：**做了什么 / 没做什么 / 下一步**。缺任何一段，别人没法判断能不能接你的活。

### 6. 用 agent 写代码的，自己补 `.gitignore`

`.gitignore` 已经挡掉了 `.workbuddy/`、`.tools/`、`.claude/`、`.codex/` 这类常见 agent 目录，但**你的 agent 可能生成别的东西**（工具缓存、会话日志、临时索引、`.bak`）。判断标准只有一条：

> **别人克隆之后拿不到它，也不影响复现——那它就不该入库。**

符合这条的，你自己补进 `.gitignore` 的「agent scaffolding」一节，并在提交信息里写明补了什么（第五节硬规则 4）。

### 7. 动手前先确认自己在哪个阶段门内

阶段之间不并行（第三节）。上一阶段没验收，不要开下一阶段的工——包括"先写着玩玩"。写了不验收，等于给后面的人一个看起来能跑、其实没人验过的地基。

### 30 秒自检

| 检查 | 命令 | 期望 |
|---|---|---|
| Python 版本 | `python -V` | Core/Polybert 3.10.x；GNN 3.11.x |
| 三个环境在 | `conda env list` | `DSC6020_Project_{Core,GNN,Polybert}` |
| 远端正确 | `git remote -v` | `github.com/XHH-Coco/How-Much-Is-a-Chain-Worth` |
| 没把数据带进来 | `git status --short` | 不出现 `data/raw/` 下的文件 |
| 冻结物没动 | `git status --short splits/ configs/` | 空 |
| 没把 agent 目录带进来 | `git status --short` | 不出现 `.workbuddy/`、`.tools/` |

---

## 二、仓库现在是什么状态

**骨架 + 已冻结的环境。**

| 有什么 | 还没有什么 |
|---|---|
| 目录约定（空目录各放一个 `.gitkeep`） | 没有实现代码 |
| 元数据：`LICENSE`、`CITATION.cff`、`.gitignore`、`.gitattributes` | 没有配置文件（任务、实验一个都没写） |
| 三份环境声明 + **两层锁**：`environment/*.yml`、`lock/*.{conda,pip}.txt`、`lock/SHA256SUMS` | 没有数据、没有划分索引、没有运行结果 |
| **三份环境指纹**：`results/env_fingerprints/env_{core,gnn,polybert}.json` | 没有测试 |
| 文档：两份 Proposal、四份分工任务书、代码规范、运行请求单、P0 交付记录 | — |

**P0 已完成并冻结**（2026-11-11）：三个 conda 环境按声明装好，导出了两层锁与三份指纹，验收凭据在 [`docs/交付_P0_环境.md`](docs/交付_P0_环境.md)。

**P1 完成**（2026-11-12）：A-2 的三个基准全部落地——PolyBench26（63 文件 / 66.0 MiB，两个官方 sha256 逐位相等）、OpenPoly（741 行 / 174 KiB）、PolyBench（39 个 CSV / 5,937,499 B，**39/39 与 figshare 公布的 `supplied_md5` 逐个相等**）。三份清单齐备，PolyBench 那份 39 行、条数合计 42,169、与 Proposal 附录 A 逐行对齐。figshare 对本机出口仍 403，39 个 CSV 由仓库所有者人工下载、脚本离线核 md5 采纳（`DECISIONS.md` D-13/D-14）。验收凭据见 [`docs/交付_P1_A2_下载与完整性.md`](docs/交付_P1_A2_下载与完整性.md) §9。

**P2 进行中**（2026-11-12）：A-3 标签调和已交付——**6 个主表任务（14,103 条）+ 4 个预备队任务（6,557 条）**已调和入库，`bandgap chain_Egc` 因表头不写单位按红线整列弃用（3,380 条），决策表 30 行、10 个任务去重删 **0** 行。验收凭据见 [`docs/交付_P1_A3_标签调和.md`](docs/交付_P1_A3_标签调和.md)。

**官方划分快照已交付**（2026-11-12，A-5）：`splits/` 下 **14 份只读快照**——PolyBench 10 个任务的官方 70/30 测试集（从平台接口取回，按 RDKit 规范 SMILES 对回行号，比例 30.01%–30.26%）+ PolyBench26 的 4 份官方索引（原样拷贝、逐文件核 sha256、官方自检 `Verified 4 compact scaling-index packages.`）。**OpenPoly 0 份**：官方只发布了读 `.pt` 的 notebook，划分文件与生成代码都不在发布物里，因此按红线**明确停住、不自己切**（D-22）。8 条验收标准 **7 过 1 不过**（第 ⑤ 条因上游缺失不通过）。凭据见 [`docs/交付_P1_A5_官方划分快照.md`](docs/交付_P1_A5_官方划分快照.md)。

**任务清单已冻结**（2026-11-12，A-4）：`configs/tasks/` 下 **18 份**配置——8 个主表（PolyBench 每层 2 个 + PolyBench26 2 个）、5 个预注册备用、3 个 OpenPoly 边界性质、2 个机制实验任务，另有 `SHA256SUMS` 与快照 [`docs/Task List Snapshot D0.md`](docs/Task%20List%20Snapshot%20D0.md)（冻结于 `2026-11-12T03:48:00+08:00`）。校验器在 `src/chainworth/io/tasks.py`：字段缺失、任务名拼错、n 与上游清单不符**一律当场报错**。同一次交付里结案 Q-01（**主表 = 8 任务 × 5 臂**，D-17），并纠正两处口径：PolyMetriX Tg 源数据已是 K（D-18）、密度单位官方未声明故不写（D-19）。凭据见 [`docs/交付_P1_A4_任务清单冻结.md`](docs/交付_P1_A4_任务清单冻结.md)。

上一版仓库曾在初始化阶段一次性写入了代码、配置、任务清单与测试。那不是本次初始化的范围，已被整体移除；那些内容保留在提交 `21be74f` 里，需要时按阶段逐项取回，而不是一次全塞进来。

**原则：一个阶段只做一件事，做完、验完、提交，再进下一阶段。**

---

## 三、阶段计划

| 阶段 | 做什么 | 产出落在 | 验收标准 | 状态 |
|---|---|---|---|---|
| **P0 环境** | 三个 conda 环境装包、锁版本 | `environment/` + `lock/` | 每个环境能 import 自己的核心包，版本可复现 | **已交付 2026-11-11**，见 `docs/交付_P0_环境.md` |
| **P1 获取数据** | 下载**三个**基准（第四个 MoleculeNet/PCQM4Mv2 只登记不复现，见 `docs/DECISIONS.md` D-02），算摘要 | `data/raw/` + `data/manifests/` | 条数与官方发布一致（PolyBench 必须 39 条、合计 42,169），每个文件有 sha256 | **已交付 2026-11-12**：三份清单齐备；PolyBench 39 行 / 42,169 / 与附录 A 逐行对齐，39/39 核过官方 md5，见 `docs/交付_P1_A2_下载与完整性.md` §9 |
| **P2 清洗与划分** | 统一记录格式；官方划分做成只读快照 | `data/interim/` + `splits/` | 划分文件带校验，任何人不得改写 | **◐ 标签调和已交付 2026-11-12**（A-3：10 个任务调和入库、4 条验收标准全过，见 `docs/交付_P1_A3_标签调和.md` §5）；**官方划分快照已交付 2026-11-12**（A-5：14 份只读快照、8 条验收 7 过 1 停，OpenPoly 按 D-22 明确停住，见 `docs/交付_P1_A5_官方划分快照.md`） |
| **P3 配置** | 任务、实验、处理的配置文件 | `configs/` | 每个配置能被读取并校验，错配即报错 | **◐ 任务清单已冻结 2026-11-12**（A-4：18 份 `configs/tasks/*.yaml` + 快照，见 `docs/交付_P1_A4_任务清单冻结.md`）；`configs/arms/`、`configs/experiments/` 未开始 |
| **P4 代码** | 按分层实现 | `src/chainworth/` | 单元测试通过，同 seed 两次结果逐位相同 | 未开始 |
| **P5 运行** | 按配置跑，逐次落记录 | `results/raw_runs/` | 每个数字都能追到某一次运行 | 未开始 |
| **P6 图表与文档** | 出图、写方法与结论 | `results/figures/` + `docs/` | 图能由脚本一键重画 | 未开始 |

阶段之间不并行。上一阶段没验收，下一阶段不开工。

---

## 四、文件结构与目录约定

**这一节是结构说明书，和磁盘必须一致。** 树的唯一来源是 `git ls-files`（不是手写记忆）。**改了结构却不在同一次提交里改本节，这次提交就算没做完**（见第十节）。

### 4.1 当前结构（142 个跟踪文件）

```text
How-Much-Is-a-Chain-Worth/
├── README.md            本文件：结构、规则、冻结清单的唯一权威
├── LICENSE              MIT
├── CITATION.cff         引用信息
├── .gitignore           入库 / 不入库的分界线
├── .gitattributes       换行符与语言统计
│
├── environment/         三套环境的「声明」（已冻结 v2，精确版本见 lock/）
│   ├── core.yml
│   ├── gnn.yml
│   └── polybert.yml
│
├── lock/                ← P0 已填：两层锁，复现环境的权威依据
│   ├── core.conda.txt   conda 层：包 URL + 构建号
│   ├── core.pip.txt     pip 层：== 精确版本 + 索引地址
│   ├── gnn.conda.txt
│   ├── gnn.pip.txt
│   ├── polybert.conda.txt
│   ├── polybert.pip.txt
│   └── SHA256SUMS       上面 6 个文件的校验和
│
├── configs/             ← P3 填；A-4 已先落 tasks/（18 份，冻结 v1）
│   ├── tasks/           一个任务一份（A-4 已填 18 份 + `SHA256SUMS`）
│   ├── experiments/     一次实验一份
│   └── arms/            一个臂一份
│
├── data/                ← P1–P2 填：数据本体不入库，只留清单
│   ├── raw/             原样下载，只读
│   ├── interim/         中间产物，可重建
│   │                    ← A-3 已填：harmonised/<task_id>_harmonised.parquet（10 张，不入库）
│   ├── processed/       喂模型前的最终表
│   ├── external/        基准之外的补充数据
│   └── manifests/       清单、条数、sha256（入库）+ SOURCES.md（入口与许可）+ 接口响应快照
│                        ← A-2 已填：三份 `*_download_manifest.csv` + figshare 快照
│                        ← A-3 已填：`unit_decisions.csv`（30 行）+ `harmonised_manifest.csv` + `harmonise_spotcheck.csv`
│
├── splits/              ← P2 填：划分索引，只读（入库）—— A-5 已填 14 份
│   ├── polybench/       A-5：10 份 `official_70_30.json` + `README.md`（官方测试集来源与对号规则）
│   ├── polybench26/     A-5：4 份 `seed_42/{manifest.json, fold_00..04.npz}` 原样拷贝 + `PROVENANCE.json` + `README.md`
│   │   └── arch/        A-5：只有 `README.md`——官方只发物化脚本，索引谁用谁本地生成，不入库
│   └── openpoly/        A-5：只有 `NOT_AVAILABLE.json`（D-22：官方未发布划分，按红线停住）
│
├── src/chainworth/      ← P4 填：包代码
│   ├── io/              数据与配置的唯一出入口（A-3 填 `harmonise.py`；A-4 填 `tasks.py`）
│   ├── chem/            结构解析、描述符、分子图
│   ├── arms/            五个臂各一个模块
│   ├── metrics/         指标与 bootstrap
│   ├── analysis/        Λ、κ、假设检验
│   ├── figures/         五张图
│   └── protocol/        种子、常量、划分校验（A-3 填 `constants.py`；A-5 填 `splits.py`：快照生成 + 四层校验 + 自切扫描）
│
├── scripts/             ← P1 起填：阶段入口 00–07（入库，必须可重跑）
│                        ← 已填 `00_download.py`（A-2：三基准下载 + 完整性核对）、`01_harmonise.py`（A-3：标签调和）、`03_splits.py`（A-5：官方划分快照 + `--verify --deep`）
├── tests/               ← P4 起填（A-3 已先落 `test_harmonise.py`：17 项；A-4 落 `test_task_configs.py`：15 项；A-5 落 `test_splits.py`：28 项）
│
├── results/             ← P5 起填（P0 已先落三份环境指纹）
│   ├── raw_runs/        一次运行一份 JSON（入库）
│   ├── aggregate/       汇总表与预测（入库）
│   ├── figures/         图（可由脚本重画）
│   └── env_fingerprints/ 指纹（入库）：P0 三份，P5 起每次运行一份
│       ├── env_core.json
│       ├── env_gnn.json
│       └── env_polybert.json
│
└── docs/
    ├── Proposal_HowMuchIsAChainWorth_v1.3.md        英文版，判据以此为准
    ├── Proposal_HowMuchIsAChainWorth_中文版_v1.3.md  逐节对照
    ├── 分工_A_数据与合规.md
    ├── 分工_B_表示与GPU臂.md
    ├── 分工_C_经典臂与机制实验.md
    ├── 分工_D_评测与图表.md
    ├── CODE_STYLE.md      代码怎么写
    ├── RUN_REQUESTS.md    没有 GPU 时怎么请求代跑
    ├── DECISIONS.md       决策记录（D-xx 已定）与待决问题（Q-xx）
    ├── 交付_P0_环境.md     P0 交付记录：装了什么、怎么验、哪些没做到
    ├── 交付_P1_A2_下载与完整性.md  A-2 交付记录：下了什么、哈希对没对上、PolyBench 为什么没下成
    ├── 交付_P1_A3_标签调和.md      A-3 交付记录：单位凭什么这么定、删了几行、哪些任务没做
    ├── 交付_P1_A4_任务清单冻结.md  A-4 交付记录：18 份配置怎么点出来的、臂数口径、三处新事实
    ├── 交付_P1_A5_官方划分快照.md  A-5 交付记录：14 份快照怎么来的、8 条验收 7 过 1 停、OpenPoly 为什么停住
    └── Task List Snapshot D0.md   **任务清单快照（冻结物）**：冻结时刻、清单表、替换登记
```

**每个空目录里有一个 `.gitkeep`。** git 不记录空目录，删掉它，别人克隆下来这个目录就消失了。`← Pn 填` 的意思是：在这个阶段之前，该目录应该是空的；谁提前往里塞东西，就是越阶段。

### 4.2 每个目录放哪些文件

| 目录 | 放哪些文件（命名规则 / 例子） | 明确不放 |
|---|---|---|
| `environment/` | 一个环境一份声明，首行带 `# FROZEN v2`：`core.yml`、`gnn.yml`、`polybert.yml` | 锁文件（放 `lock/`）、装包产物 |
| `lock/` | 一律成对：`<env>.conda.txt`（conda 层，URL+构建号）+ `<env>.pip.txt`（pip 层，`==` 版本）+ `SHA256SUMS` | 手改的锁；`conda-lock` 的 `.lock.yml`（本项目不用，原因见 `docs/交付_P0_环境.md` §4） |
| `configs/tasks/` | 一个任务一份 `<dataset>_t<n>_<property>.yaml`，如 `pb_t1_tg.yaml`。**A-4 已填 18 份**（8 主表 + 5 备用 + 3 OpenPoly + 2 机制），schema 与校验在 `src/chainworth/io/tasks.py`，指纹在 `configs/tasks/SHA256SUMS`，清单快照是 `docs/Task List Snapshot D0.md` | 实验编排、超参数、臂的名单（臂在 `configs/arms/`，这里的 `arms` 只是"这个任务要跑哪些"） |
| `configs/arms/` | 一个臂一份 `<arm>.yaml`：`a1_floor`…`a5_chain` | 任务定义、种子列表 |
| `configs/experiments/` | 一次实验一份，写清跑哪些任务 × 哪些臂 × 几个种子：`main_table.yaml`、`scaling.yaml`、`arch_transfer.yaml`、`openpoly_scan.yaml` | 单任务字段（放 `tasks/`） |
| `data/raw/` | 原样下载，**保留发布方原文件名** | 清洗、改名、合并、补列 |
| `data/interim/` | 中间产物，文件名带中间步骤：`harmonised/<task_id>_harmonised.parquet`（A-3 已出 10 张，**不入库**，靠 `scripts/01_harmonise.py` 重建；行数与哈希登记在 `data/manifests/harmonised_manifest.csv`） | 唯一副本 |
| `data/processed/` | 一个任务一张最终表 `<task_id>.parquet`；特征放 `features/<task_id>.npz` | 原始文件 |
| `data/external/` | 基准之外补充的数据：`table_s27.csv` | 主基准（那在 `raw/`） |
| `data/manifests/` | 清单与校验：`SOURCES.md`（入口、许可、版本钉法，**唯一权威**）、`polybench_api_data_counts_20261111.json`（`/api/data-counts` 原始响应，39 条 / 42,169 的引证）、`polybench_api_article_20261112.json` + `polybench_expected_files.csv`（figshare 原始响应快照与派生的 **39 行标准表**：文件名 / 字节数 / 官方 md5 / 直链，任何下载路线的验收标准，见 `DECISIONS.md` D-13）、`<bench>_download_manifest.csv`（一文件一行，列见 `docs/交付_P1_A2_下载与完整性.md` §2）、`*.sha256`、**`unit_decisions.csv`**（A-3：每条单位/去重/弃列判定一行，**只可追加**）、**`harmonised_manifest.csv`**（A-3：一任务一行，`rows_in/rows_out/source_sha256/parquet_sha256`）、**`harmonise_spotcheck.csv`**（A-3：抽检 20 行，原始/调和/重算三列并排）、**`splits_manifest.csv`**（A-5：14 份快照一文件一行，`sha256/bytes/n_train/n_test/created_at`）、**`polybench_official_test_manifest.csv`**（A-5：10 份官方测试集 CSV 一行一任务，`n_rows/n_total/test_ratio/sha256/source_url`） | 数据本体；`.partial.csv`（限量跑的调试产物，已 gitignore） |
| `splits/polybench/`、`splits/polybench26/`、`splits/openpoly/` | 划分索引，**只读**：PolyBench 是 `<task>/official_70_30.json`（A-5 出 10 份：`n_total`/`n_test`/`test_index`/`device_label`/`source_url`，行号绑定 A-3 那张表）；PolyBench26 是官方 `scaling_indices/` 的**原样拷贝** `<task>/seed_42/{manifest.json, fold_00..04.npz}`（不改字节，A-5 出 4 份）+ `PROVENANCE.json`（commit、源目录、逐文件 sha256、`canonical_sha256`、折数、档位），外加一份 `README.md` 记来源与三组 sha256。OpenPoly 只有 `NOT_AVAILABLE.json`（D-22）；`splits/polybench26/arch/` 只有一份说明（本地物化、不入库） | 生成脚本的临时输出、自己切的划分 |
| `src/chainworth/io/` | `loaders.py`（读数据与配置）、`writers.py`（写 run 记录）、`harmonise.py`（A-3：原始 CSV → 调和表 + 决策行）、`tasks.py`（A-4：读并校验 `configs/tasks/*.yaml`，CLI 见 `python -m chainworth.io.tasks --verify`） | 业务逻辑、硬编码路径 |
| `src/chainworth/chem/` | `features.py`（描述符）、`graph.py`（分子图构建） | 模型定义 |
| `src/chainworth/arms/` | 一臂一文件：`floor.py`、`conventional.py`、`graph.py`、`polybert.py`、`chain_info.py` | 训练循环（`train.py`） |
| `src/chainworth/metrics/` | `regression.py`（R²、Spearman）、`bootstrap.py`（残差重采样） | 图、表格 |
| `src/chainworth/analysis/` | `lambda_.py`、`kappa.py`、`hypotheses.py` | 画图 |
| `src/chainworth/figures/` | 一图一文件：`fig1_*.py`…`fig5_*.py` | 硬编码数字、截图 |
| `src/chainworth/protocol/` | `seeds.py`、`constants.py`、`splits.py`（A-5：官方划分快照的唯一实现——PolyBench 取测试集并对行号、PB26 canonical 物化与快照、OpenPoly 停住结论、四层校验 `verify_splits(deep=)`、`find_self_cut_calls()` 自切扫描） | 实验结果 |
| `scripts/` | 阶段入口 `00_download.py`…`07_figures.py`，必须支持 `--dry-run` / `--limit` | 临时脚本（`scripts/_scratch/`，已忽略）、可复用库代码（放 `src/`） |
| `tests/` | `test_<模块>.py`；小型合成样例放 `fixtures/` | 真实数据副本、必须 GPU 才能跑的测试 |
| `results/raw_runs/` | 一次运行一份 `<run_id>.json` | 手写或手改的数字 |
| `results/aggregate/` | 汇总 `<name>.csv`（`main_table` / `scaling` / `arch_transfer` / `openpoly_scan`）+ `predictions/<run_id>.parquet` | 手改的数字 |
| `results/figures/` | 脚本导出的图 `<figN>_<name>.pdf` / `.png` | 截图、从 PPT 里抠出来的图 |
| `results/env_fingerprints/` | P0 已填 `env_core.json`、`env_gnn.json`、`env_polybert.json`；P5 起一次运行一份 `<run_id>.json` | 环境说明文档（写 `docs/`） |
| `docs/` | Proposal、四份分工书、`CODE_STYLE.md`、`RUN_REQUESTS.md`、`DECISIONS.md`（决策记录 D-xx 与待决问题 Q-xx，**已建**，口径变更必须登记）、`交付_<阶段>_<主题>.md`（已交付 `交付_P0_环境.md`、`交付_P1_A2_下载与完整性.md`、`交付_P1_A3_标签调和.md`、`交付_P1_A4_任务清单冻结.md`、`交付_P1_A5_官方划分快照.md`）、`Task List Snapshot D0.md`（**任务清单快照，冻结物**）；后续还会有 `Audit Log.md`（审计日志）、`Conclusions.md`（结论）、`Main Table Notes.md`（主表口径） | 与 README 重复的内容、数据、运行结果 |
| 仓库根目录 | 只允许 `README.md`、`LICENSE`、`CITATION.cff`、`.gitignore`、`.gitattributes` 五个文件 + 上面九个目录 | 散落文件（`test.py`、`tmp.csv`、`notes.md`、`新建文本文档.txt`） |

#### 4.2.1 阶段入口 00–07 归谁（**写代码之前先看这张表**）

`scripts/` 已落地 `00_download.py`、`01_harmonise.py`、`03_splits.py` 三个（对应行标了"已交付"），其余五个仍是**预先约定**。编号只表示"流水线第几步"，不表示谁的私产；谁接手别人的编号，先改这张表并到 `docs/DECISIONS.md` 记一条——否则两个人会各自写出一个 `01_*.py`。

| 编号 | 脚本 | 主责 | 输入 | 输出 |
|:--|:--|:--|:--|:--|
| 00 | `00_download.py` | A | `data/manifests/SOURCES.md` | `data/raw/**` + `*_download_manifest.csv` |
| 01 | `01_harmonise.py` | A | `data/raw/**` | `data/interim/harmonised/*.parquet` + `unit_decisions.csv` + `harmonised_manifest.csv`（**已交付 2026-11-12**，见 `docs/交付_P1_A3_标签调和.md`） |
| 02 | `02_features.py` | A | `data/interim/harmonised/` | `data/processed/structures/`、`descriptors/`（不入库，靠它重建） |
| 03 | `03_splits.py` | A | 各基准官方划分索引 | `splits/**`（只读快照）（**已交付 2026-11-12**：`--source {all,canonical,polybench,polybench26,openpoly}` + `--verify --deep`，见 `docs/交付_P1_A5_官方划分快照.md`） |
| 04 | `04_run_cpu.py` | C | `configs/**` + `splits/` | `results/raw_runs/*.json`（经典臂批量） |
| 05 | `05_run_gpu.py` | B | `configs/**` + `splits/` | `results/raw_runs/*.json`（GNN / polyBERT 臂） |
| 06 | `06_aggregate.py` | C | `results/raw_runs/` | `results/aggregate/main_table.csv` 等 |
| 07 | `07_figures.py` | D | `results/aggregate/` | `results/figures/figN_*.pdf` / `.png` |

两条附加约定：每个入口必须支持 `--dry-run` / `--limit`（第七节第 5 条）；06 → 07 之间不允许出现"手改过的中间表"，图只能从 `results/aggregate/` 里的文件画。

### 4.3 不入库的东西

数据本体（`data/raw|interim|processed` 里的内容）、模型权重与缓存、日志、`scripts/_scratch/`、agent 自建目录（`.workbuddy/`、`.tools/` 等）、`__pycache__` 与各类缓存。

完整规则见 `.gitignore`。判断标准就是硬规则 4 那一句：**别人克隆之后拿不到它，也不影响复现——那它就不该入库。**

### 4.4 改结构的时候怎么办

1. 目录增、删、改名，或根目录加文件 → **同一次提交**里改本节两处：4.1 的树、4.2 的表。
2. 提交信息加一行「结构变更：加了什么、为什么」。
3. 树用 `git ls-files` 重新生成，**不要手抄**；手抄的树一周之内就会错。
4. 新目录必须带 `.gitkeep`。若该目录内容被忽略（如 `data/raw/`），同时在 `.gitignore` 里加 `!<dir>/.gitkeep`，否则克隆后目录消失。
5. 结构改动会牵动别的文档（分工书里的 `scripts/00_download.py`、Proposal 的目录描述）→ 同一提交内一并改；漏改按"提交未完成"处理，不能"下次再说"。
6. 一次提交只做一类结构变更。别把"加目录"和"挪数据"混在同一个 commit 里，否则回退时两件事都退不掉。

> 为什么这么较真：四个人的项目里，"我按老结构去找文件、找不到"是最常见的浪费。README 是唯一权威，结构一旦漂移，就没人敢信它写的任何东西。

---

## 五、四条硬规则

1. **原始数据永不入库。** 仓库里只有指针（URL、sha256、条数）。
2. **`splits/` 只读。** 划分一旦确定，任何代码不得重算；要换划分，出新版本文件。
3. **结论必须对应一条运行记录。** 没有 run 记录的数字，不写进任何文档。
4. **agent 产物不入库。** `.workbuddy/`、`.tools/` 这类由 agent 建起来的工作目录，一律进 `.gitignore`。组员若使用 agent，**需自行在 `.gitignore` 补全由 agent 生成、不应参与版本控制的所有文件或文件夹**，并在提交信息里写明补了什么。判断标准：别人克隆后拿不到它，也不影响复现。

---

## 六、环境

| 文件 | conda 环境 | Python | GPU | 为什么单独一个 |
|---|---|---|---|---|
| `environment/core.yml` | `DSC6020_Project_Core` | 3.10 | 否 | 化学信息学工具 + 经典模型 + 统计 |
| `environment/gnn.yml` | `DSC6020_Project_GNN` | 3.11 | 是 | 图神经网络框架 + chemprop 2.x（该包强制要求 ≥3.11），且与上者的 OpenMP 运行库冲突 |
| `environment/polybert.yml` | `DSC6020_Project_Polybert` | 3.10 | 是 | 预训练嵌入，权重缓存约 7 GB，需可独立删除 |

**两张表是什么关系**：`environment/*.yml` 是**声明**——写"要哪个版本"，只列直接依赖，装的时候由 conda 求解器现场解析；`lock/*.txt` 是**快照**——从装好的环境反向导出，"实际装出了哪些文件"，连间接依赖一起写死。**装环境按 `lock/`，`.yml` 留作读。** 理由与失败后果见第一节第 1 步。

2026-11-11 已按锁装好并冻结（Core / Polybert 为 Python 3.10.14，GNN 为 Python 3.11.18）。新机器或重装：

```bash
# 1) conda 层：按 URL + 构建号精确重放，不重新求解
conda create -n DSC6020_Project_Core      --file lock/core.conda.txt
conda create -n DSC6020_Project_GNN       --file lock/gnn.conda.txt
conda create -n DSC6020_Project_Polybert  --file lock/polybert.conda.txt

# 2) pip 层：索引地址（cu121 / PyPI / PyG wheel）写在文件头，pip 会自动带上
conda activate DSC6020_Project_Core      && pip install -r lock/core.pip.txt
conda activate DSC6020_Project_GNN       && pip install -r lock/gnn.pip.txt
conda activate DSC6020_Project_Polybert  && pip install -r lock/polybert.pip.txt

# 3) 校验锁文件自身没被改坏，6 个文件全 OK 才继续
sha256sum -c lock/SHA256SUMS
```

**退路（只在锁重放失败时走）**——例如换了操作系统、或镜像不可达。这条路会让求解器重新解析，**装出来的环境不保证与锁一致**；走完必须重新导出锁并告知分工 A 登记：

```bash
conda env update -n DSC6020_Project_Core     -f environment/core.yml     --prune
conda env update -n DSC6020_Project_GNN      -f environment/gnn.yml      --prune
conda env update -n DSC6020_Project_Polybert -f environment/polybert.yml --prune
```

三条禁忌（环境不许互相 install、不许装进系统 Python、不许另建 `requirements.txt`）见第一节第 1 步。`environment/` 三份已冻结（第九节第 1 行）。

**P0 已完成并冻结（2026-11-11）**：`lock/` 里每个环境两份锁——`<env>.conda.txt` 是 conda 层（包 URL + 构建号，可 `conda create --file` 直接重建），`<env>.pip.txt` 是 pip 层（`==` 精确版本 + 索引地址，可 `pip install -r`）。`lock/SHA256SUMS` 给这 6 个文件做校验。三份指纹在 `results/env_fingerprints/`。

**能力边界要说清**：这套锁能复现到**版本级**（同平台、同 Python 版本，装出来版本号逐个相同），不能复现到**哈希级**（pip 层没有 wheel 的 sha256；补哈希要下载全部 wheel，torch 单包 2.4 GB）。为什么没用 `conda-lock`、以及其余四条已知局限，写在 [`docs/交付_P0_环境.md`](docs/交付_P0_环境.md) 第 4、5 节。

---

## 七、代码规范

**P4 起，凡进入 `src/`、`scripts/`、`tests/` 的代码必须遵守 [`docs/CODE_STYLE.md`](docs/CODE_STYLE.md)。** P1/P2 写下载与清洗脚本时已经适用——那些脚本是后面所有数字的源头。摘要：

1. **中英双语注释。** 中文在前、英文在后，用 ` / ` 分隔；长注释先中文整段、再英文整段，不交错。
2. **每个函数都必须有 docstring**（含私有函数），写清七项：用途、输入（含义+单位+来源）、输出（形状/dtype/索引含义）、数据来源、配置来源、副作用、异常。
3. **每个功能块前有分隔注释**，格式 `# ---- 中文 / English ----`；函数超过约 20 行必须分块。
4. **数字必须有来源。** 除 0/1 之类的结构性常量，任何字面量要么从 `configs/` 读，要么行尾注释写明来源（Proposal 章节、官方文件名、文档 URL）。
5. **一次性脚本不进仓库。** 放 `scripts/_scratch/`（已忽略）；`scripts/` 里只留能重跑的阶段入口，且必须支持 `--dry-run` / `--limit`。

> 为什么这么严：这个项目最贵的错误是"没人报错但结果全错"（Proposal §7.5 的三类失败模式）。注释写清数据来源与配置来源，是事后唯一能查出这类错误的办法。

---

## 八、命名约定

- 包名统一用 `chainworth`。文档、配置、产物一律用可读名称，不使用内部方案编号。
- 文档与文件用标题命名，不用代号：`Proposal_HowMuchIsAChainWorth_v1.3.md`。
- 配置与产物用可读 ID（`pb_t1_tg`、`a2_conventional`），不用序号。
- 代码标识符用英文；注释按第七节双语。

---

## 九、冻结登记（谁改谁负责）

**冻结 = 只读。** 一旦某项被标为冻结，任何人不得在原地改；要改就出新版本文件，并在 `docs/DECISIONS.md` 记一条（为什么改、影响哪些 run）。**冻结物被改动 = 该阶段重做。**

### 登记规则

1. 冻结发生在**阶段验收那一刻**，由该阶段负责人在下表登记：版本号、冻结日期、校验方式。只写"已冻结"不算。
2. 冻结物必须**能验**：带 sha256、带条数、带版本号。验不了的不叫冻结，叫"大概定了"。
3. 能写注释的文件（`.yml`、`.py`、`.md`），第一行加标记：`# FROZEN v<n> | <日期> | sha256=<前 12 位>`。**自指的情况**（对文件自己算哈希，加完这行哈希就变了）改为登记**它所配套的、可独立校验的那个文件**的哈希——例如 `environment/core.yml` 登记 `lock/core.pip.txt` 的哈希。
4. 不能写注释的（`.json`、`.csv`、`.parquet`），在同目录 `MANIFEST.sha256` 里登记，一行一个文件。
5. 冻结后**本表就是唯一权威副本**。分工文档 §5 只列"与你直接相关的那几项"，口径冲突时以本表为准。

### 登记表

| # | 冻结点 | 位置 | 冻结于 | 怎么验 | 状态 |
|---:|---|---|---|---|---|
| 1 | 三套环境版本 | `environment/*.yml`（声明）+ `lock/*.{conda,pip}.txt`（精确） | **P0 · 2026-11-11** | `lock/SHA256SUMS`（6 个文件）+ 三份 `results/env_fingerprints/env_*.json`；重建时 `pip install -r --dry-run` 应全部 `already satisfied` | **已冻结 v2（2026-11-11 21:26，`280f60506e9e` / `a91f0654a867` / `f636dab33d98`；v1 为 `f67dec4a853e` / `90e3560567ca` / `e6b7bbcc639b`）。v2 = v1 + `pyarrow==17.0.0`（parquet 引擎），conda 层字节未变** |
| 2 | 原始下载文件 | `data/raw/` | P1 | `data/manifests/*.sha256`，一个文件一行 | 未冻结（待 P1） |
| 3 | 任务清单与条数 | `data/manifests/tasks_*.csv` | P1 | 条数与官方发布逐行对齐 + sha256 | 未冻结（待 P1） |
| 4 | 单位换算决策 | `data/manifests/unit_decisions.csv` | P2 | 只能追加、不能覆盖；每行带 evidence；`python scripts/01_harmonise.py --verify` 逐条核 | **A-3 已交 v1（2026-11-12，30 行，sha256 见 `git log` 该文件首条提交）**，P2 划分完成后一并冻结 |
| 5 | 清洗后的规范表 | `data/interim/`、`data/processed/` | P2 | 行数/列数断言 + `MANIFEST.sha256`（A-3 用 `harmonised_manifest.csv` 的 `rows_out` + `parquet_sha256` 顶上） | **A-3 已交 v1（2026-11-12，10 张表）**，P2 划分完成后一并冻结 |
| 6 | 官方划分快照 | `splits/polybench/`、`splits/polybench26/`、`splits/openpoly/` | P2 | 每份文件的 sha256（json 与 npz 都算），`tests/test_splits.py` 断言；PolyBench26 另需 `canonical_sha256` 与 `train_pool[:N]` 前缀断言 | **A-5 已交 v1（2026-11-12，14 份：PolyBench 10 + PolyBench26 4，指纹表 `data/manifests/splits_manifest.csv`，`python scripts/03_splits.py --verify --deep` 通过）**；OpenPoly 0 份，按 D-22 记 `blocked_upstream`，不属本行冻结范围 |
| 7 | 任务配置 | `configs/tasks/*.yaml` | P3 | `configs/tasks/SHA256SUMS`（`cd configs/tasks && sha256sum -c`）+ 每次 run 记录里的 `config_hash` 必须对得上 | **A-4 已交 v1（2026-11-12，18 份，指纹表 `configs/tasks/SHA256SUMS`）**；`config_hash` 与臂/实验配置在 P3 一并登记 |
| 8 | 臂的超参数 | `configs/arms/*.yaml` | P3 | 同上 | 未冻结（待 P3） |

> 每次冻结后**立即更新本表**，并在提交信息里写"冻结了什么、版本号是什么"。这张表是组员判断"我能不能动这个文件"的唯一依据。

---

## 十、提交约定

一个阶段一个提交。提交信息必须写清三件事：**做了什么、没做什么、下一步是什么**。跨阶段的改动拆开提交，便于回退到任意阶段边界。

冻结内容、`.gitignore` 的改动、配置改动必须在"做了什么"里单独点名。

**结构变更必须同一次提交内同步第四节。** 增删目录、改目录名、在根目录加文件，都要在同一个 commit 里更新 4.1 的树与 4.2 的表，并在提交信息里单列一行「结构变更：加了什么、为什么」。**结构变了而 README 没变，这次提交不算完成**，不要等"下次一起改"——下周没人记得改过什么。

---

## 十一、分工文档

四份任务书，各自独立可读（不需要先读别的文件）。第 2 节统一是"拿什么 → 怎么做 → 产出 → 验收标准"，第 4 节统一写"没有 GPU 时交什么、由谁代跑"，第 5 节写冻结边界。

**写作与汇报不单列角色**：叙事、幻灯片、Q&A 题库、彩排由仓库所有者承担，不计入下表。

| 文件 | 角色 | 固定项 | 余量 | 工时 |
|---|---|---|---|---|
| `docs/分工_A_数据与合规.md` | 环境、基准接入、标签调和、任务清单冻结、官方划分、结构级表征 | 21.5 | 2.5 | 24.0 pH |
| `docs/分工_B_表示与GPU臂.md` | 表示装配与图构建、polyBERT 嵌入、A3 主扫、A4 迁移臂、OpenPoly 的 GNN 部分 | 21.5 | 2.5 | 24.0 pH |
| `docs/分工_C_经典臂与机制实验.md` | A1/A2/A5、scaling 与 OpenPoly 经典臂、架构迁移、主表、运行记录审计 | 22.0 | 2.0 | 24.0 pH |
| `docs/分工_D_评测与图表.md` | Λ / κ / CI、H1–H4、五张图、三条结论、对账、入口与出口审计 | 21.5 | 2.5 | 24.0 pH |

四人的名义工时相同（各 24.0 pH；固定项 21.5–22.0，极差 0.5 pH，差额是各自的弹性余量）：技术工时合计 96 pH，即原 105 pH 减去写作与汇报的 9 pH。

切法按流水线走：**A 出数据 → B 跑要卡的（A3/A4 + OpenPoly GNN）→ C 跑不要卡的（A1/A2/A5 + 三批机制实验）并汇总主表 → D 下判决并出图**。审计 14 pH 按"谁的数字依赖它谁来查"分配：入口 5（D：划分与泄漏，因为 Λ 的分母就是 `n` 与 `split_ref`）、运行记录 4（C：他跑出 505/778 的 fit）、出口 5（D，属自查，已在 `Audit Log.md` 里写明性质）。

任务项编号为 `A-1`…`A-8`（A-8 为可选验证项，不计配额）、`B-1`…`B-5`、`C-1`…`C-6`、`D-1`…`D-10`；Proposal §6 的五个臂仍记作 `A1`–`A5`（地板 / 经典 / 图 / 语言模型 / 链信息）。两套编号刻意不同形，避免混读。

两份配套文档：`docs/CODE_STYLE.md`（代码怎么写）、`docs/RUN_REQUESTS.md`（没有 GPU 时怎么请求代跑）。
