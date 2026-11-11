# 链值多少？ / How Much Is a Chain Worth?

> 把一条高分子链压缩成一个重复单元之后，还有多少性质方差解释不了？这个比例会随着数据量增加十倍而缩小吗？

`status: P0 已完成（环境已冻结）` · 代码 MIT · 仓库内不含任何基准数据

---

## 一、第一次克隆之后：按顺序做完这七件事

**先读这一节，再动手。** 这是一个空骨架：没有数据、没有代码、没有配置。下面的顺序是刻意的——跳过第 1 步会在装包时炸，跳过第 3 步会在跑完之后返工。

### 1. 装环境：严格按 `environment/*.yml`，不要 `pip install` 最新版

```bash
conda env create -f environment/core.yml       # 数据、化学信息学、经典模型
conda env create -f environment/gnn.yml        # 图神经网络（与上者 OpenMP 冲突，必须分开）
conda env create -f environment/polybert.yml   # 预训练嵌入，权重缓存约 7 GB
conda activate DSC6020_Project_Core
python -V                                      # Core/Polybert 3.10.x；GNN 3.11.x（见下）
```

- **Python 版本：Core / Polybert 用 3.10.14，GNN 用 3.11.18。** 不是随手混的：`chemprop 2.x` 强制要求 ≥3.11（PyPI 上 2.0.0–2.3.1 全部如此），而 1.x 只支持 <3.9，所以 **3.10 上不存在任何可安装的 chemprop**，而 A3 臂正是建立在它上面。三套包版本各自按对应 Python 的 wheel 兼容性钉死；谁单独升大版本，别人拿到的 `lock` 就对他失效。（3.10 已于 2026-10-31 EOL，目前是已知取舍，记在 `environment/*.yml` 的头注释里。）
- **三个环境不许互相 install。** 把 RDKit 塞进 GNN 环境，`libgomp` 会让训练半路崩，而且报错与化学库毫无关系，极难查。
- **不要往系统 Python 里装，也不要新建 `requirements.txt`。** 本仓库只有 conda 一套声明。
- **P0 已验收，现在以 `lock/*.conda.txt` + `lock/*.pip.txt` 为准**（`environment/*.yml` 是声明，`lock/` 是从装好的环境反向导出的精确版本表）。两者不一致时找分工 A，不要自己改——`environment/` 三份文件已冻结（第九节第 1 行）。装法、验收命令与已知局限见 [`docs/交付_P0_环境.md`](docs/交付_P0_环境.md)。

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

**P0 已完成并冻结**（2026-11-11）：三个 conda 环境按声明装好，导出了两层锁与三份指纹，验收凭据在 [`docs/交付_P0_环境.md`](docs/交付_P0_环境.md)。下一阶段是 P1 获取数据。

上一版仓库曾在初始化阶段一次性写入了代码、配置、任务清单与测试。那不是本次初始化的范围，已被整体移除；那些内容保留在提交 `21be74f` 里，需要时按阶段逐项取回，而不是一次全塞进来。

**原则：一个阶段只做一件事，做完、验完、提交，再进下一阶段。**

---

## 三、阶段计划

| 阶段 | 做什么 | 产出落在 | 验收标准 | 状态 |
|---|---|---|---|---|
| **P0 环境** | 三个 conda 环境装包、锁版本 | `environment/` + `lock/` | 每个环境能 import 自己的核心包，版本可复现 | **已交付 2026-11-11**，见 `docs/交付_P0_环境.md` |
| **P1 获取数据** | 下载四个基准，算摘要 | `data/raw/` + `data/manifests/` | 条数与官方发布一致，每个文件有 sha256 | ← 下一个 |
| **P2 清洗与划分** | 统一记录格式；官方划分做成只读快照 | `data/interim/` + `splits/` | 划分文件带校验，任何人不得改写 | 未开始 |
| **P3 配置** | 任务、实验、处理的配置文件 | `configs/` | 每个配置能被读取并校验，错配即报错 | 未开始 |
| **P4 代码** | 按分层实现 | `src/chainworth/` | 单元测试通过，同 seed 两次结果逐位相同 | 未开始 |
| **P5 运行** | 按配置跑，逐次落记录 | `results/raw_runs/` | 每个数字都能追到某一次运行 | 未开始 |
| **P6 图表与文档** | 出图、写方法与结论 | `results/figures/` + `docs/` | 图能由脚本一键重画 | 未开始 |

阶段之间不并行。上一阶段没验收，下一阶段不开工。

---

## 四、文件结构与目录约定

**这一节是结构说明书，和磁盘必须一致。** 树的唯一来源是 `git ls-files`（不是手写记忆）。**改了结构却不在同一次提交里改本节，这次提交就算没做完**（见第十节）。

### 4.1 当前结构（52 个跟踪文件）

```text
How-Much-Is-a-Chain-Worth/
├── README.md            本文件：结构、规则、冻结清单的唯一权威
├── LICENSE              MIT
├── CITATION.cff         引用信息
├── .gitignore           入库 / 不入库的分界线
├── .gitattributes       换行符与语言统计
│
├── environment/         三套环境的「声明」（已冻结 v1，精确版本见 lock/）
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
├── configs/             ← P3 填：配置文件
│   ├── tasks/           一个任务一份
│   ├── experiments/     一次实验一份
│   └── arms/            一个臂一份
│
├── data/                ← P1–P2 填：数据本体不入库，只留清单
│   ├── raw/             原样下载，只读
│   ├── interim/         中间产物，可重建
│   ├── processed/       喂模型前的最终表
│   ├── external/        基准之外的补充数据
│   └── manifests/       清单、条数、sha256（入库）
│
├── splits/              ← P2 填：划分索引，只读（入库）
│   ├── polybench/
│   ├── polybench26/
│   └── openpoly/
│
├── src/chainworth/      ← P4 填：包代码
│   ├── io/              数据与配置的唯一出入口
│   ├── chem/            结构解析、描述符、分子图
│   ├── arms/            五个臂各一个模块
│   ├── metrics/         指标与 bootstrap
│   ├── analysis/        Λ、κ、假设检验
│   ├── figures/         五张图
│   └── protocol/        种子、常量、划分校验
│
├── scripts/             ← P1 起填：阶段入口 00–07（入库，必须可重跑）
├── tests/               ← P4 起填
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
    └── 交付_P0_环境.md     P0 交付记录：装了什么、怎么验、哪些没做到
```

**每个空目录里有一个 `.gitkeep`。** git 不记录空目录，删掉它，别人克隆下来这个目录就消失了。`← Pn 填` 的意思是：在这个阶段之前，该目录应该是空的；谁提前往里塞东西，就是越阶段。

### 4.2 每个目录放哪些文件

| 目录 | 放哪些文件（命名规则 / 例子） | 明确不放 |
|---|---|---|
| `environment/` | 一个环境一份声明，首行带 `# FROZEN v1`：`core.yml`、`gnn.yml`、`polybert.yml` | 锁文件（放 `lock/`）、装包产物 |
| `lock/` | 一律成对：`<env>.conda.txt`（conda 层，URL+构建号）+ `<env>.pip.txt`（pip 层，`==` 版本）+ `SHA256SUMS` | 手改的锁；`conda-lock` 的 `.lock.yml`（本项目不用，原因见 `docs/交付_P0_环境.md` §4） |
| `configs/tasks/` | 一个任务一份 `<dataset>_t<n>_<property>.yaml`，如 `pb_t1_tg.yaml` | 实验编排、超参数 |
| `configs/arms/` | 一个臂一份 `<arm>.yaml`：`a1_floor`…`a5_chain` | 任务定义、种子列表 |
| `configs/experiments/` | 一次实验一份，写清跑哪些任务 × 哪些臂 × 几个种子：`main_table.yaml`、`scaling.yaml`、`arch_transfer.yaml`、`openpoly_scan.yaml` | 单任务字段（放 `tasks/`） |
| `data/raw/` | 原样下载，**保留发布方原文件名** | 清洗、改名、合并、补列 |
| `data/interim/` | 中间产物，文件名带中间步骤：`*_harmonised.parquet` | 唯一副本 |
| `data/processed/` | 一个任务一张最终表 `<task_id>.parquet`；特征放 `features/<task_id>.npz` | 原始文件 |
| `data/external/` | 基准之外补充的数据：`table_s27.csv` | 主基准（那在 `raw/`） |
| `data/manifests/` | 清单与校验：`*_manifest.csv`、`*.sha256`、`unit_decisions.csv` | 数据本体 |
| `splits/polybench/`、`splits/polybench26/`、`splits/openpoly/` | 划分索引 `<task_id>_<fold>.json`，**只读** | 生成脚本的临时输出 |
| `src/chainworth/io/` | `loaders.py`（读数据与配置）、`writers.py`（写 run 记录） | 业务逻辑、硬编码路径 |
| `src/chainworth/chem/` | `features.py`（描述符）、`graph.py`（分子图构建） | 模型定义 |
| `src/chainworth/arms/` | 一臂一文件：`floor.py`、`conventional.py`、`graph.py`、`polybert.py`、`chain_info.py` | 训练循环（`train.py`） |
| `src/chainworth/metrics/` | `regression.py`（R²、Spearman）、`bootstrap.py`（残差重采样） | 图、表格 |
| `src/chainworth/analysis/` | `lambda_.py`、`kappa.py`、`hypotheses.py` | 画图 |
| `src/chainworth/figures/` | 一图一文件：`fig1_*.py`…`fig5_*.py` | 硬编码数字、截图 |
| `src/chainworth/protocol/` | `seeds.py`、`constants.py`、`splits.py`（划分校验） | 实验结果 |
| `scripts/` | 阶段入口 `00_download.py`…`07_figures.py`，必须支持 `--dry-run` / `--limit` | 临时脚本（`scripts/_scratch/`，已忽略）、可复用库代码（放 `src/`） |
| `tests/` | `test_<模块>.py`；小型合成样例放 `fixtures/` | 真实数据副本、必须 GPU 才能跑的测试 |
| `results/raw_runs/` | 一次运行一份 `<run_id>.json` | 手写或手改的数字 |
| `results/aggregate/` | 汇总 `<name>.csv`（`main_table` / `scaling` / `arch_transfer` / `openpoly_scan`）+ `predictions/<run_id>.parquet` | 手改的数字 |
| `results/figures/` | 脚本导出的图 `<figN>_<name>.pdf` / `.png` | 截图、从 PPT 里抠出来的图 |
| `results/env_fingerprints/` | P0 已填 `env_core.json`、`env_gnn.json`、`env_polybert.json`；P5 起一次运行一份 `<run_id>.json` | 环境说明文档（写 `docs/`） |
| `docs/` | Proposal、四份分工书、`CODE_STYLE.md`、`RUN_REQUESTS.md`、`交付_<阶段>_<主题>.md`（已交付 `交付_P0_环境.md`）；后续还会有 `DECISIONS.md`（决策记录）、`Audit Log.md`（审计日志）、`Conclusions.md`（结论）、`Task List Snapshot D0.md`（任务清单快照）、`Main Table Notes.md`（主表口径） | 与 README 重复的内容、数据、运行结果 |
| 仓库根目录 | 只允许 `README.md`、`LICENSE`、`CITATION.cff`、`.gitignore`、`.gitattributes` 五个文件 + 上面九个目录 | 散落文件（`test.py`、`tmp.csv`、`notes.md`、`新建文本文档.txt`） |

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

三份 `.yml` 既是声明，也是**安装凭据**：2026-11-11 已按此装好（Core / Polybert 为 Python 3.10.14，GNN 为 Python 3.11.18）。重装命令：

```bash
conda env update -n DSC6020_Project_Core    -f environment/core.yml    --prune
conda env update -n DSC6020_Project_GNN     -f environment/gnn.yml     --prune
conda env update -n DSC6020_Project_Polybert -f environment/polybert.yml --prune
```

装法与三条禁忌见第一节第 1 步。

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
3. 能写注释的文件（`.yml`、`.py`、`.md`），第一行加标记：`# FROZEN v1 | 2026-11-15 | sha256=<前 12 位>`。**自指的情况**（对文件自己算哈希，加完这行哈希就变了）改为登记**它所配套的、可独立校验的那个文件**的哈希——例如 `environment/core.yml` 登记 `lock/core.pip.txt` 的哈希。
4. 不能写注释的（`.json`、`.csv`、`.parquet`），在同目录 `MANIFEST.sha256` 里登记，一行一个文件。
5. 冻结后**本表就是唯一权威副本**。分工文档 §5 只列"与你直接相关的那几项"，口径冲突时以本表为准。

### 登记表

| # | 冻结点 | 位置 | 冻结于 | 怎么验 | 状态 |
|---:|---|---|---|---|---|
| 1 | 三套环境版本 | `environment/*.yml`（声明）+ `lock/*.{conda,pip}.txt`（精确） | **P0 · 2026-11-11** | `lock/SHA256SUMS`（6 个文件）+ 三份 `results/env_fingerprints/env_*.json`；重建时 `pip install -r --dry-run` 应全部 `already satisfied` | **已冻结 v1（`f67dec4a853e` / `90e3560567ca` / `e6b7bbcc639b`）** |
| 2 | 原始下载文件 | `data/raw/` | P1 | `data/manifests/*.sha256`，一个文件一行 | 未冻结（待 P1） |
| 3 | 任务清单与条数 | `data/manifests/tasks_*.csv` | P1 | 条数与官方发布逐行对齐 + sha256 | 未冻结（待 P1） |
| 4 | 单位换算决策 | `data/manifests/unit_decisions.csv` | P2 | 只能追加、不能覆盖；每行带 evidence | 未冻结（待 P2） |
| 5 | 清洗后的规范表 | `data/interim/`、`data/processed/` | P2 | 行数/列数断言 + `MANIFEST.sha256` | 未冻结（待 P2） |
| 6 | 官方划分快照 | `splits/polybench/`、`splits/polybench26/`、`splits/openpoly/` | P2 | 每份 json 的 sha256，`tests/test_splits.py` 断言 | 未冻结（待 P2） |
| 7 | 任务配置 | `configs/tasks/*.yaml` | P3 | 每次 run 记录里的 `config_hash` 必须对得上 | 未冻结（待 P3） |
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

切法按流水线走：**A 出数据 → B 跑要卡的（A3/A4 + OpenPoly GNN）→ C 跑不要卡的（A1/A2/A5 + 三批机制实验）并汇总主表 → D 下判决并出图**。审计 14 pH 按"谁的数字依赖它谁来查"分配：入口 5（D：划分与泄漏，因为 Λ 的分母就是 `n` 与 `split_ref`）、运行记录 4（C：他跑出 535/808 的 fit）、出口 5（D，属自查，已在 `Audit Log.md` 里写明性质）。

任务项编号为 `A-1`…`A-8`（A-8 为可选验证项，不计配额）、`B-1`…`B-5`、`C-1`…`C-6`、`D-1`…`D-10`；Proposal §6 的五个臂仍记作 `A1`–`A5`（地板 / 经典 / 图 / 语言模型 / 链信息）。两套编号刻意不同形，避免混读。

两份配套文档：`docs/CODE_STYLE.md`（代码怎么写）、`docs/RUN_REQUESTS.md`（没有 GPU 时怎么请求代跑）。
