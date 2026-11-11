# 代码规范（Code Style）

> **适用范围**：`src/chainworth/`、`scripts/`、`tests/` 下的全部 `.py`，以及其他语言（`.yaml` 注释、`.R`、shell）按同一精神执行。
> **生效时间**：P4 起强制；P1/P2 写下载与清洗脚本时**已经适用**（那些脚本是后面所有数字的源头）。

**一句话**：半年后打开这个文件的人——包括你自己——只靠注释就应该知道：这段代码**吃什么、吐什么、数字从哪来、配置读的哪一份**。

---

## 0. 为什么要求这么细

Proposal §7.5 列了三类"不会报错、但会让主表作废"的失败：**(i)** 自己切数据而不用官方划分；**(ii)** 全局统计量泄进特征；**(iii)** 指标各写一份、口径不一。

这三类错误都发生在**代码与数据的边界**上，而边界只有注释能说清楚。所以审阅时不看代码风格、不看命名是否漂亮，只看两件事：

1. 能不能从注释**追到数据来源**（哪个文件的哪一版）；
2. 能不能从注释**追到配置来源**（哪个 yaml 的哪个键）。

追不到的，退回重写。

---

## 1. 注释一律中英双语

| 规则 | 说明 |
|---|---|
| 顺序 | **中文在前，英文在后**，用 ` / ` 分隔 |
| 长注释 | 先中文整段（连续若干行），再英文整段；**不要一句中文一句英文交错** |
| 不翻译的东西 | 标识符、文件名、字段名、指标名保留英文原样：`split_ref`、`run_id`、`R²_unit`、`Murcko` |
| docstring 域标记 | `Args` / `Returns` / `Raises` / `Yields` 保留英文 |

```python
# 正确 / correct
# 只在 train 上 fit 标准化器；否则测试集的均值会漏进特征。 / Fit the scaler on train only; otherwise test-set means leak into features.

# 正确（长注释） / correct (long form)
# 官方 scaling 档位是现成文件，不是"取前 n 行"。
# 读文件、校验哈希，任何按行数截断的做法都会让 κ 偏移。
# The official scaling tiers ship as files, not as "first n rows".
# Read the file and verify its hash; truncating by row count shifts kappa.

# 错误（中英交错） / wrong (interleaved)
# 只在 train 上 fit / Fit the scaler on train only / 否则泄漏 / otherwise leakage
```

---

## 2. 模块级 docstring（每个 `.py` 文件必须有）

放在 `import` 之前，必须写清五项：

```python
"""按任务读取已冻结的规范表，并按官方划分切成训练/测试集。
Read frozen per-task tables and slice them with the official split.

中文：
    用途：本模块是"数据 -> 特征矩阵"的唯一入口，禁止在别处直接 read_csv。
    依赖：data/processed/{task}.parquet（P2 冻结）、splits/（P2 冻结）。
    配置：configs/tasks/{task}.yaml 的 data / split 两节。
    对外：load_task()、iter_tasks()。
    被谁用：src/chainworth/arms/*、src/chainworth/analysis/*。

English:
    Purpose: the single entry point from data to feature matrices; no other
    module may call read_csv directly.
    Depends on: data/processed/{task}.parquet and splits/ (both frozen at P2).
    Config: the `data` and `split` sections of configs/tasks/{task}.yaml.
    Public API: load_task(), iter_tasks().
    Callers: src/chainworth/arms/*, src/chainworth/analysis/*.
"""
```

---

## 3. 函数与方法：一个都不能漏（含私有函数）

每个函数必须有 docstring，且必须出现下面七项。可以合并成段落，但不许缺项：

1. **用途** / Purpose —— 一句话说这是干什么的，不要复述函数名；
2. **输入** / Args —— 每个参数：含义 + 单位 + 取值来源（谁传的、从哪读的）；
3. **输出** / Returns —— 形状、dtype、单位、索引含义；
4. **数据来源** / Data source —— 具体到文件路径与版本（`splits/polybench/*.json`，P2 冻结）；
5. **配置来源** / Config —— `configs/**/*.yaml` 的哪个键；没有就写 `无 / none`；
6. **副作用** / Side effects —— 写哪个目录、建什么文件、改什么状态；纯函数写 `无 / none`；
7. **异常** / Raises —— 什么条件下抛什么，尤其是"配置缺键""条数对不上"。

### 模板

```python
def build_split(task_id: str, split_ref: str) -> tuple[np.ndarray, np.ndarray]:
    """从官方划分快照里读出 train/test 行号，不做任何重算。
    Read train/test row indices from the official split snapshot; nothing is recomputed.

    中文：
        用途：把只读索引文件变成两个行号数组，供特征矩阵切片。
        输入：task_id —— 任务 ID（如 "pb_t1_tg"），必须能在 configs/tasks/ 里找到；
              split_ref —— 相对仓库根的划分文件路径（如 "splits/polybench/pb_t1_tg_seed0.json"）。
        输出：(train_idx, test_idx)，int64 一维数组，元素是 0-based 行号，二者不相交。
        数据来源：splits/polybench/*.json（P2 冻结，登记于 README 第九节）。
        配置来源：configs/tasks/{task_id}.yaml 的 split 字段。
        副作用：无（纯函数，不写盘）。
        异常：FileNotFoundError —— 快照不存在；ValueError —— 两个索引有交集。

    English:
        Purpose: turn the read-only index file into two row-index arrays for slicing.
        Args: task_id - task id resolvable under configs/tasks/; split_ref - repo-relative path to the split json.
        Returns: (train_idx, test_idx), 1-D int64 arrays of 0-based row positions, disjoint.
        Data source: splits/polybench/*.json (frozen at P2, registered in README section 9).
        Config: the `split` key of configs/tasks/{task_id}.yaml.
        Side effects: none.
        Raises: FileNotFoundError if the snapshot is missing; ValueError if the two index sets overlap.
    """
```

> 私有函数（`_helper`）一样要写，只是可以压成两段：中文一段、英文一段，七项用分号连着写。

---

## 4. 功能块：用分隔注释标出来

一个函数超过约 20 行，必须分块；每块写清"这一步在做什么、为什么不能省"。

```python
# ---- 1. 读配置并校验版本 / Read config and verify version ----
cfg = load_task_config(task_id)          # 配置来源：configs/tasks/{task_id}.yaml
assert cfg["split"]["frozen"], "划分未冻结，不许开工 / split not frozen, refusing to run"

# ---- 2. 切特征（只对 train fit） / Slice features (fit on train only) ----
scaler = StandardScaler().fit(X[train_idx])

# ---- 3. 落盘 run 记录 / Write the run record ----
```

模块内的大段落（常量区、类、注册表）同样用 `# ---- 中文 / English ----` 分隔。

---

## 5. 数字必须有来源

- **禁止裸字面量**。除了 `0`、`1`、`2`、`-1` 这类结构性常量，任何数字要么从 `configs/` 读，要么在行尾注释写明来源。
- 反复使用的数字（阈值、档位、种子、最大 n）统一放 `src/chainworth/protocol/constants.py`，**每个常量写一行来源注释**。

```python
# 300 / 1000 / 3000 / 10000：官方 scaling_splits 的档位，不是等比切分。
# 来源：PolyBench26 release, scaling_splits/README（哈希见 data/manifests/）。
SCALING_TIERS = (300, 1000, 3000, 10000)

MAX_N = 443          # OpenPoly 的 Tg 标注数，超过就是外推。来源：data/manifests/openpoly_counts.csv
```

---

## 6. 数据与配置只走一个入口

- 读数据：一律 `src/chainworth/io/loaders.py`，**不许**在业务函数里写 `pd.read_csv("data/processed/...")`。路径集中一处，才查得动"这个数字是哪张表算出来的"。
- 读配置：一律 `load_task_config()` / `load_arm_config()`，函数内部做字段校验；**不许**散落 `yaml.safe_load(open(...))`。
- 写结果：一律走 `io/` 下的 run 记录写入器，由它自动填 `run_id`、`config_hash`、`device`、时间戳。人不可手写这些字段。

---

## 7. 禁止事项

| 不要 | 因为 | 改成 |
|---|---|---|
| `print()` 当日志 | 跑 200 个 run 时无法过滤 | `logging`，级别由 config 控制 |
| 在业务代码里 `np.random.seed()` | 三处各设一次，结果互相覆盖 | `protocol/seeds.py` 统一派生 |
| 注释复述代码（"i 加 1"） | 零信息量，还会过期 | 写"为什么"，不写"是什么" |
| 在 notebook 里改数据再存回 | 改动无提交、无 sha256 | 改写脚本进 `scripts/`，notebook 只读 |
| 在 `src/` 里硬编码绝对路径 | 换机器就崩，且没人知道原来指哪 | 相对仓库根，或由 config 传 |
| 用 `**kwargs` 吞掉参数 | 输入含义不可见，无法写 docstring | 显式列参数 |

---

## 8. 一次性脚本：放 `scripts/_scratch/`

- `scripts/` 里**只留阶段入口**（`00`–`07`），必须可重跑：幂等、带 `--dry-run` 和 `--limit`。
- 临时调试、一次性统计、试验性画图 → `scripts/_scratch/`，已在 `.gitignore` 里，**永不提交**。
- 判断标准：**别人克隆后能不能靠这一个文件把你的结果重跑出来？** 不能，就别放进 `scripts/`。

---

## 9. 提交前自检

- [ ] 每个新增函数都有双语 docstring，且"数据来源""配置来源"两项都不是空话
- [ ] 模块顶部有模块级 docstring，写了依赖与被谁调用
- [ ] 没有裸数字（`grep -nE "[^a-zA-Z_][0-9]{2,}" src/ | grep -v "#"` 抽查）
- [ ] 没有 `print()` 当日志
- [ ] `src/` 里没有 `read_csv(` / `open(` 的裸路径
- [ ] `scripts/` 的新入口能 `--dry-run` 通过，且重复运行结果一致
- [ ] 没有把 `scripts/_scratch/` 或 agent 目录带进提交（`git status --short` 自查）
