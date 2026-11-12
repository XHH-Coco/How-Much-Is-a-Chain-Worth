"""A3 图臂：在 B-1 的分子图上跑 chemprop 官方 MPNN，一次运行落一条记录 + 一份预测。
A3 graph arm: run chemprop's official MPNN over B-1's molecular graphs; one run writes one record
plus one predictions file.

中文：
    用途：分工_B 的 B-3/B-5 共用的训练器。一件事情的两种说法：
          ① 对一个"任务 × 折 × 种子"训练一次图神经网络，在官方测试集上算 r2/mae/rmse；
          ② 把这次训练写成 `results/raw_runs/<run_id>.json` 与
             `results/aggregate/predictions/<run_id>.parquet`，让 D 不必问任何人就能算 Λ/κ 与区间。
    为什么不用 `chemprop train` 命令行：官方 CLI 读的是 CSV、自己切分、自己决定输出目录。
          本仓库的划分是**只读快照**（`splits/`），行序由 `row_id` 决定，评估只能读官方索引
          （红线 1）。所以这里只借官方的**模型与训练循环**（`chemprop.models.MPNN` +
          `BatchMolGraph` + `BondMessagePassing`/`NormAggregation`/`RegressionFFN` +
          Noam 式学习率），数据与划分一律来自 B-1 的装配产物。
    依赖：torch 2.3.1+cu121 / torch-geometric 2.5.3 / chemprop 2.1.0 / pytorch-lightning
          （`environment/gnn.yml`）；numpy / pandas / pyarrow / PyYAML。
    配置：`configs/arms/a3_graph.yaml`（冻结 v1）。**它的字节哈希就是 `config_hash`**——
          改一个空格，48 条 run 的哈希全部对不上。
    对外：`ArmConfig`、`load_arm_config()`、`config_hash()`、`enumerate_runs()`、`run_id_for()`、
          `carve_validation()`、`metrics()`、`build_task_data()`、`train_one()`、`run_grid()`、
          `write_run_record()`、`write_predictions()`、`record_problems()`。
    被谁用：`scripts/05_run_gpu.py`（B-3/B-5 的唯一入口）、`tests/test_arms_graph.py`。

English:
    Purpose: the trainer shared by B-3 and B-5. One thing said twice: (1) train a GNN on one
    "task x fold x seed" and score r2/mae/rmse on the official test split; (2) write that run out as
    `results/raw_runs/<run_id>.json` plus `results/aggregate/predictions/<run_id>.parquet`, so D can
    compute Lambda/kappa and its intervals without asking anyone anything.
    Why not the `chemprop train` CLI: the official CLI reads a CSV, splits it itself and picks its
    own output layout. Here the split is a **read-only snapshot** (`splits/`), row order comes from
    `row_id`, and evaluation may only read the official indices (red line 1). So this module borrows
    the official **model and training loop** (`chemprop.models.MPNN`, `BatchMolGraph`,
    `BondMessagePassing`/`NormAggregation`/`RegressionFFN`, the Noam-like schedule) and takes data
    and splits from B-1's assembly, always.
    Depends on: torch 2.3.1+cu121, torch-geometric 2.5.3, chemprop 2.1.0, pytorch-lightning
    (environment/gnn.yml); numpy, pandas, pyarrow, PyYAML.
    Config: `configs/arms/a3_graph.yaml` (frozen v1). **Its byte hash is `config_hash`.**
    Public API: `ArmConfig`, `load_arm_config()`, `config_hash()`, `enumerate_runs()`,
    `run_id_for()`, `carve_validation()`, `metrics()`, `build_task_data()`, `train_one()`,
    `run_grid()`, `write_run_record()`, `write_predictions()`, `record_problems()`.
    Callers: `scripts/05_run_gpu.py` and `tests/test_arms_graph.py`.

四条不能搞错的口径 ／ Four conventions that must not slip
-----------------------------------------------------------
1. **评估只读官方索引。** `train_one()` 的 train/val/test 全部来自 `SplitSpec`，函数体里没有
   任何 `random_split`/`train_test_split`（红线 1）。PolyBench 官方没有验证集，早停用**训练集内部**
   的固定 10% 留出（`carve_validation()`），这份留出永远不碰测试集，也不进指标。
   Evaluation reads the official indices only; there is no `random_split`/`train_test_split` here.
   PolyBench ships no validation set, so early stopping uses a **fixed 10% of the official train
   split** - never the test split, and it never enters the reported metrics.
2. **标签标准化只用 train 的统计量。** 均值/标准差在 `train_index` 上算一次，写进
   `UnscaleTransform`；验证集与测试集直接套用（红线 3 的近亲：全量统计量会泄漏）。
   Target standardisation uses train statistics only, written into `UnscaleTransform`; val and test
   reuse them.
3. **`device` 由脚本填，人不可改。** CPU 上跑出来的是 `cpu`，GPU 上是 `cuda`；混报等于让整张表
   作废（分工_B §7 常见坑）。`device` 与环境指纹一起进记录。
   `device` is filled by the script, never by hand; mixing CPU and GPU results voids the table.
4. **确定性要两条腿，且两条由同一个变量驱动。** `cudnn.deterministic` 只管卷积类算子，
   管不到 chemprop 图聚合用的 `scatter_reduce_(reduce="sum")`（CUDA 上走原子加）；
   而 `pl.Trainer(deterministic=False)` 会在构造时**主动调用** `use_deterministic_algorithms(False)`，
   把外面设的开关抹掉。实测：只钉 cudnn 时同 seed 两次差 **65 K**；两条都钉后**逐位相同**，
   代价约 1.6 倍墙钟。细节见 `train_one()` 里那段注释。
   Determinism needs both legs, driven by one variable: cudnn covers conv-like ops only, not the
   `scatter_reduce` that chemprop's aggregation uses, and `Trainer(deterministic=False)` actively
   disables `use_deterministic_algorithms` in its constructor. Measured: cudnn-only differs by 65 K
   between same-seed runs; both legs are bit-identical at ~1.6x wall clock.

关于标签标准化：为什么必须做，以及踩过的坑 ／ Target standardisation: why, and the trap
--------------------------------------------------------------------------------
`chemprop.nn.UnscaleTransform.forward` 在 `self.training` 为真时**原样返回输入**，只有 `eval()`
之后才做 `x * scale + mean`（`transforms.py` 源码，不是猜的）。于是训练/验证阶段网络输出活在
**标准化空间**里，喂给 criterion 的标签也必须是 `(y − mean) / scale`；喂原单位会让网络去追原单位
的数值，而预测时再乘一次 scale——实测把 618 K 预测成 98,564 K（约 180 倍），R² 变成 −189,524。
这也是 chemprop 官方流程在建 dataloader 前调 `normalize_targets()` 的原因。
**代价要写明**：`val_loss` 是标准化单位的 RMSE，记录里的 `best_val_rmse` 已乘回 `scale`、
与原单位同尺；`val_loss_normalized` 原样保留，便于和 chemprop 的日志对拍。
`UnscaleTransform.forward` returns its input unchanged while `self.training` is true; only after
`eval()` does it apply `x * scale + mean` (from `transforms.py`, not guessed). Training and
validation therefore live in **standardised space**, so the labels handed to the criterion must be
`(y - mean) / scale`. Raw labels make the network chase raw values and then multiply by scale once
more at prediction time - measured at ~180x (618 K predicted as 98,564 K, R^2 = -189,524), which is
exactly why chemprop's own pipeline calls `normalize_targets()` first.
The cost is stated in the record: `val_loss` is an RMSE in standardised units, `best_val_rmse` is
multiplied back by `scale` to sit in original units, and `val_loss_normalized` keeps the raw value
for comparison against chemprop's own logs.
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import yaml

# cuBLAS 在 `torch.use_deterministic_algorithms(True)` 下要求给 workspace 一个固定大小，并且
# **必须在本进程第一次调用 cuBLAS 之前**设好——之后设无效，PyTorch 会抛
# "CUDA tensors with cuBLAS require CUBLAS_WORKSPACE_CONFIG to be set"。所以放在模块层而不是
# 函数里：调用方可能先建图、先摸显卡，等进到 `train_one` 时就晚了。`:4096:8` 是 PyTorch
# 文档给的值，含义是"用 4096 字节 × 8 的 workspace"，只影响显存里一小块临时区。
# cuBLAS needs a fixed workspace size under deterministic algorithms, and it must be set before
# the first cuBLAS call in the process (setting it later raises). Module level, not function level:
# callers may touch the GPU while building data, so `train_one` would be too late.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

# `features` 与 `chem.graph` **不在模块层导入**：它们分别要 pandas / rdkit，而本模块的通用部分
# （网格、指标、记录读写、`run_grid`）要被 A4（`arms/polybert.py`）复用——A4 跑在
# `DSC6020_Project_Polybert` 环境里，那个环境**没有 pandas 也没有 rdkit**（`lock/polybert.pip.txt`）。
# 真正需要建图的地方只有 `build_task_data()`，它在函数体内导入这两者。
# `features` and `chem.graph` are not imported at module level: they need pandas/rdkit, while the
# generic half of this module (grid, metrics, record I/O, `run_grid`) is reused by the A4 arm, which
# runs in an environment without either. Only `build_task_data()` needs them, and imports them there.
from chainworth.io import loaders as L

LOGGER = logging.getLogger("chainworth.arms.graph")

REPO_ROOT = Path(__file__).resolve().parents[3]    # src/chainworth/arms/graph.py -> 仓库根
DEFAULT_CONFIG_PATH = REPO_ROOT / "configs" / "arms" / "a3_graph.yaml"
RAW_RUNS_DIR = REPO_ROOT / "results" / "raw_runs"
PREDICTIONS_DIR = REPO_ROOT / "results" / "aggregate" / "predictions"
AGGREGATE_DIR = REPO_ROOT / "results" / "aggregate"
GPU_HOURS_PATH = AGGREGATE_DIR / "gpu_hours.csv"
ENV_FINGERPRINT = "results/env_fingerprints/env_gnn.json"

ARM = "a3_graph"

# 五个臂的名字，抄自 `results/raw_runs/_schema.json` 的 `arm.enum`（测试会拿文件和这份常量对拍，
# 所以这里不是"第二份真相"，而是"schema 变了我必须知道"）。`record_problems()` 用它判臂名，
# 这样 A4 不必再往这个模块里塞一个 `if arm == ...`。
# The five arm names, copied from the schema's `arm.enum`; a test diffs this tuple against the file,
# so it is a tripwire rather than a second source of truth.
KNOWN_ARMS = ("a1_floor", "a2_conventional", "a3_graph", "a4_polybert", "a5_chain")

# run_id 的格式由 `results/raw_runs/_schema.json` 的 pattern 钉死；
# 这里的两个常量只是把那条规则写在代码里，免得拼错要到写文件时才发现。
# The run_id shape is pinned by the schema's pattern; these two constants restate it so a typo
# fails here rather than at write time.
RUN_ID_SEPARATOR = "__"

# 记录里必须出现的字段（与 `_schema.json` 的 `required` 一致）。校验器在 `record_problems()`。
# Fields every record must carry (same list as the schema's `required`, checked by
# `record_problems()`).
RECORD_REQUIRED = (
    "run_id",
    "status",
    "benchmark",
    "task",
    "arm",
    "seed",
    "split_version",
    "config_hash",
    "env_fingerprint",
)
RECORD_REQUIRED_OK = ("metrics", "wall_clock_s", "device", "predictions", "n_train", "n_test")

# `gpu_hours.csv` 的列。只增不改（同 run 记录规则）。
# Columns of gpu_hours.csv; additive only, like the run records.
GPU_HOURS_COLUMNS = (
    "run_id",
    "benchmark",
    "task",
    "arm",
    "split_fold",
    "seed",
    "device",
    "n_train",
    "n_val",
    "n_test",
    "epochs_run",
    "best_epoch",
    "best_val_rmse",
    "wall_clock_s",
    "gpu_hours",
    "config_hash",
    "split_version",
)

# B-5 的两份索引。`openpoly_gpu_index.csv` 是分工_B §2 B-5 点名的产物
# （`run_id, property, n_level, fold, seed, wall_clock_s` 六列都在里面，另外多给几列是为了
# 让 D 不必再开 225 个 JSON）；`openpoly_n_index.csv` 是"每一档到底用了哪些训练行"的凭据，
# 每 (task, fold, level) 一行，C 的经典臂可以拿它核对两边用的是同一套子集。
# The two B-5 indexes. The first is the artefact 分工_B names; the six required columns are all
# there, plus a few more so D need not open 225 JSON files. The second records which training rows
# each level used, one row per (task, fold, level), so C's classical arm can check the subsets match.
N_INDEX_COLUMNS = (
    "run_id",
    "property",
    "n_level",
    "n_train",
    "n_train_pool",
    "n_level_saturated",
    "fold",
    "fold_index",
    "seed",
    "wall_clock_s",
    "gpu_hours",
    "r2",
    "rmse",
    "mae",
    "n_test",
    "n_val",
    "epochs_run",
    "early_stopped",
    "best_epoch",
    "device",
    "subset_sha256",
    "config_hash",
    "split_version",
)

SUBSET_INDEX_COLUMNS = (
    "task",
    "fold",
    "n_level",
    "n_train",
    "n_train_pool",
    "n_level_saturated",
    "n_train_requested",
    "subset_seed",
    "subset_sha256",
    "y_mean",
    "y_std",
    "y_min",
    "y_max",
    "n_bins_covered",
    "bin_counts",
)


class ArmError(RuntimeError):
    """A3 臂无法进行时抛这一个；消息里必须写清"缺什么、谁该补"。
    The single exception raised when the arm cannot proceed; the message must be actionable.

    中文：
        用途：失败原因只有三类——缺上游产物（装配产物 / 配置）、配置字段不对、数据与划分不配套。
              三类都用这一个异常，调用方只捕一种；但**消息要能直接指导下一步**，
              否则跑到一半停下的人只看到 `KeyError: 'depth'`。
    English:
        Purpose: three failure kinds only - a missing upstream artefact, a bad config field, or data
        that does not match its split. One exception type, but the message must say what to do next.
    """


# ---- 1. 配置 / Configuration ----


@dataclass(frozen=True)
class ArmConfig:
    """冻结的臂配置 + 它的字节哈希。配置一改哈希就变，48 条记录的 `config_hash` 全部作废。
    The frozen arm config plus its byte hash; edit the file and every `config_hash` breaks.

    中文：
        字段：`path`（配置文件路径）、`raw`（解析后的字典）、`sha256`（文件字节哈希，64 位）。
        数据来源：`configs/arms/a3_graph.yaml`。配置来源：它自己。副作用：无。
        English: fields are the path, the parsed dict and the 64-hex byte digest.
    """

    path: Path
    raw: dict[str, Any]
    sha256: str

    @property
    def hash(self) -> str:
        """`sha256:<64 位>`，run 记录里 `config_hash` 的格式。／ The `sha256:...` form."""
        return f"sha256:{self.sha256}"

    def section(self, name: str) -> dict[str, Any]:
        """取一个子表（`architecture` / `training` / `device` / `records`）。
        Return one sub-table (architecture / training / device / records).

        中文：用途：配置是分节的，取小节时**缺失就报错**而不是静默给空字典——静默的空字典会让
              `depth` 变成 `None`，然后在模型里变成一个看不出根因的 TypeError。
              输入：name。输出：dict。异常：ArmError —— 缺这一节。
        English: a missing section raises instead of returning `{}`, because a silent empty dict
        turns `depth` into None and surfaces as an unexplained TypeError inside the model.
        """
        value = self.raw.get(name)
        if not isinstance(value, dict):
            raise ArmError(f"{self.path.name} 缺 `{name}` 一节：配置格式不对，别猜着读")
        return value


def _sha256_of_bytes(payload: bytes) -> str:
    """字节的 sha256。／ sha256 of some bytes.

    中文：用途：`config_hash` 用。输入：bytes。输出：64 位十六进制。
          数据来源：调用方。配置来源：无。副作用：无。异常：无。
    English: used by `config_hash`. Args: bytes. Returns: a hex digest.
    """
    return hashlib.sha256(payload).hexdigest()


def _sha256_of_file(path: Path) -> str:
    """文件字节的 sha256（分块读，别把 100 MB 读进内存）。／ sha256 of a file, read in chunks.

    中文：用途：折文件的 `split_version` 用。这里自己实现一份而不去调 `loaders` 的私有函数——
          `loaders` 的装配接口冻结后不该再被别的模块伸手进去。
          输入：path。输出：64 位十六进制。数据来源：磁盘。副作用：只读。异常：FileNotFoundError。
    English: used for a fold file's `split_version`. Implemented here rather than reaching into
    `loaders`' private helper, because loaders' assembly API is frozen.
    """
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _json_safe(value: Any) -> Any:
    """把 NaN/Inf 换成 None，让 JSON 是**合法 JSON**（`json.dumps` 默认会写出 `NaN`）。
    Turn NaN/Inf into None so the JSON is **valid JSON** (`json.dumps` writes bare `NaN` by default).

    中文：
        用途：测试集只有 1 行时 r2 是 NaN（见 `metrics()`）。`json.dumps` 会把它写成裸 `NaN`，
              而 `NaN` 不是合法 JSON——Python 读得回来，`jq`、`jsonschema`、别的语言都读不回来。
              这里递归处理容器，把非有限浮点变成 None（= 这一项不可用），其余原样。
        输入：value —— 任意可序列化结构。输出：同构结构。副作用：无。异常：无。
    English:
        With one test row, r2 is NaN. `json.dumps` writes a bare `NaN`, which Python reads back but
        `jq`, `jsonschema` and every other language reject. This recurses through containers and
        turns non-finite floats into None ("not available"), leaving everything else alone.
    """
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.floating):
        return _json_safe(float(value))
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    return value


def load_arm_config(path: str | Path | None = None) -> ArmConfig:
    """读并校验一份臂配置；哈希在**验证之前**算，保证它是磁盘上那一份字节的哈希。
    Load and validate an arm config; the hash is taken before validation, so it is the bytes on disk.

    中文：
        用途：一次读完、一次校验。校验只查"必须有且类型对"的键，不做值域猜测——值域错了应该在
              配置评审时被发现，而不是在训练到第 60 个 epoch 时。
        输入：path —— 配置路径（默认 `configs/arms/a3_graph.yaml`）。
        输出：`ArmConfig`。
        数据来源：`configs/arms/`。配置来源：它自己。副作用：只读。
        异常：ArmError —— 文件不存在、YAML 不合法、缺节缺键。
    English:
        Args: path (defaults to configs/arms/a3_graph.yaml). Returns: an ArmConfig.
        Side effects: read-only. Raises ArmError on a missing file, bad YAML or a missing key.
    """
    target = Path(path) if path is not None else DEFAULT_CONFIG_PATH
    if not target.is_file():
        raise ArmError(f"缺臂配置 {target}：B-3 的入口是 configs/arms/a3_graph.yaml")
    payload = target.read_bytes()
    digest = _sha256_of_bytes(payload)
    try:
        raw = yaml.safe_load(payload)
    except yaml.YAMLError as error:                       # pragma: no cover - 只在手改坏时触发
        raise ArmError(f"{target.name} 不是合法 YAML：{error}") from error
    if not isinstance(raw, dict):
        raise ArmError(f"{target.name} 顶层不是映射：配置格式不对")
    # 通用键所有臂都有；`architecture`/`training` 是 A3 的位置，`representation`/`head` 是 A4 的。
    # 按 arm 分派检查，而不是要求每份配置都带一堆用不上的节——A4 没有 `training.loss`，
    # 硬要它写一个就是逼人填假值。
    # Common keys for every arm; `architecture`/`training` belong to A3, `representation`/`head` to
    # A4. Checks are dispatched by arm instead of demanding a section each config never reads.
    for key in ("arm", "version", "device", "seeds", "runs", "records"):
        if key not in raw:
            raise ArmError(f"{target.name} 缺顶层键 `{key}`")
    arm_name = str(raw["arm"])
    if arm_name not in KNOWN_ARMS:
        raise ArmError(f"{target.name} 的 arm={arm_name!r} 不在 {list(KNOWN_ARMS)} 里")
    if arm_name == ARM:
        for key in ("architecture", "training"):
            if key not in raw:
                raise ArmError(f"{target.name} 缺顶层键 `{key}`")
        architecture = raw["architecture"]
        training = raw["training"]
        for key in ("depth", "hidden", "atom_feature_dim", "bond_feature_dim"):
            if key not in architecture:
                raise ArmError(f"{target.name} 的 architecture 缺 `{key}`")
        for key in ("batch_size", "max_epochs", "patience", "monitor", "max_lr", "loss"):
            if key not in training:
                raise ArmError(f"{target.name} 的 training 缺 `{key}`")
    elif arm_name == "a4_polybert":
        for key in ("representation", "head", "validation"):
            if key not in raw:
                raise ArmError(f"{target.name} 缺顶层键 `{key}`（A4 要表示来源、头、验证留出）")
        representation = raw["representation"]
        for key in ("embeddings", "index", "emb_dim", "join_key"):
            if key not in representation:
                raise ArmError(f"{target.name} 的 representation 缺 `{key}`")
        if "alpha_grid" not in raw["head"]:
            raise ArmError(f"{target.name} 的 head 缺 `alpha_grid`：候选正则强度必须写死在配置里")
    seeds = raw["seeds"]
    if not isinstance(seeds, list) or not seeds or not all(isinstance(s, int) for s in seeds):
        raise ArmError(f"{target.name} 的 seeds 必须是整数列表，现在是 {seeds!r}")
    return ArmConfig(path=target, raw=raw, sha256=digest)


def config_hash(config: ArmConfig) -> str:
    """`sha256:<64 位>`——run 记录里 `config_hash` 的值。
    The run record's `config_hash`, in `sha256:<64 hex>` form.

    中文：用途：记录必须能回答"这一条是用哪份配置跑的"；手写版本号在文件被换时不会变。
          输入：config。输出：字符串。数据来源：`ArmConfig.sha256`。副作用：无。异常：无。
    English: a hand-written version number does not change when the file is swapped; a byte hash
    does. Args: config. Returns: a string.
    """
    return config.hash


# ---- 2. 运行网格 / The run grid ----


@dataclass(frozen=True)
class RunKey:
    """一条 run 的身份：任务 × 折 × 种子。／ One run's identity: task x fold x seed.

    中文：
        用途：网格的枚举结果。`benchmark` 从任务配置里取，不从名字猜——`pb26_*` 这个前缀
              是约定，不是判据。
        字段：`task`、`benchmark`、`fold`、`seed`。
        English: `benchmark` comes from the task config, never from the name prefix.
    """

    task: str
    benchmark: str
    fold: str
    seed: int
    # `arm` 默认取本模块的 A3；A4（`arms/polybert.py`）复用同一个 `RunKey` 与同一份编排
    # （`run_grid`），只把这个字段换成 `a4_polybert`——两条臂的网格、白名单校验、run_id 形状
    # 因此是**同一份代码**，不会各写一套然后悄悄不一致。
    # `arm` defaults to A3; the polyBERT arm reuses this same RunKey and the same `run_grid`,
    # swapping only this field, so both arms share one implementation of the grid and its checks.
    arm: str = ARM
    # n 档（只有 B-5 的 OpenPoly 规模扫描用）：`"30"` / `"60"` / `"120"` / `"240"` / `"all"`。
    # **None 表示"这个网格没有 n 档这个概念"**（B-3/B-4 的 96 条），与 `"all"` 不是一回事：
    # 前者 run_id 不带后缀（历史记录逐字不变），后者带 `_nall`（225 条全靠这个后缀一眼可数）。
    # n-level (only B-5's OpenPoly scan). None means "this grid has no n dimension" and keeps the
    # run_id byte-identical to the 96 records already on disk; "all" is a real level and appends
    # `_nall`, which is what makes the 225 OpenPoly files countable with a single grep.
    n_level: str | None = None

    def run_id(self) -> str:
        """`<task>__a3_graph__<fold>__seed<N>`（schema 只允许小写字母数字与下划线）。
        The run id in the schema's shape.

        中文：用途：文件名必须等于它，`ls results/raw_runs/ | wc -l` 才能用来对账。
              有 n 档时折段写成 `<fold>_n<档>`（如 `seed0_n30`、`seed0_nall`）——schema 的正则
              只有三段 + seed，n 档没地方放，所以**借折段承载**，`fold` 字段本身仍是官方折名
              （`seed0`），测试集与验证集照旧按官方折取。
              输入：无。输出：字符串。副作用：无。异常：ArmError —— 折名里有非法字符。
        English: the file name must equal it so `ls | wc -l` reconciles. With an n-level the fold
        segment becomes `<fold>_n<level>` because the schema's pattern has exactly three segments
        plus the seed - the n-level rides along in the fold segment, while the `fold` field itself
        stays the official fold name (the test/validation sets keep coming from that official fold).
        """
        fold = self.fold.lower()
        bad = [c for c in fold if not (c.isalnum() or c == "_")]
        if bad:
            raise ArmError(f"折名 {self.fold!r} 含 schema 不允许的字符 {bad}：run_id 会不合法")
        if self.n_level is not None:
            level = str(self.n_level).lower()
            bad_level = [c for c in level if not (c.isalnum() or c == "_")]
            if bad_level:
                raise ArmError(f"n 档 {self.n_level!r} 含 schema 不允许的字符 {bad_level}")
            fold = f"{fold}_n{level}"
        return RUN_ID_SEPARATOR.join([self.task, self.arm, fold, f"seed{self.seed}"])


def run_id_for(task: str, fold: str, seed: int, *, arm: str = ARM, n_level: str | None = None) -> str:
    """按任务/折/种子拼 run_id（不查配置，供测试与对账用）。
    Build a run id from task/fold/seed without touching the config (for tests and reconciliation).

    中文：输入：task、fold、seed、arm（默认 A3）、n_level（默认 None = 无 n 档）。
          输出：字符串。副作用：无。异常：ArmError —— 名字不合法。
    English: Args: task, fold, seed, arm (A3 by default), n_level (None = no n dimension).
    Returns: a string. Raises on an illegal name.
    """
    return RunKey(task=task, benchmark="", fold=fold, seed=seed, arm=arm, n_level=n_level).run_id()


def n_level_rows(
    train_rows: "np.ndarray", n_level: str | None, *, seed: int
) -> tuple["np.ndarray", dict[str, Any]]:
    """从官方训练行里取出 n 档要用的那些行；档位之间**嵌套**，顺序无偏。
    Take the rows an n-level uses out of the official training rows; levels nest and stay unbiased.

    中文：
        用途：B-5 的 OpenPoly 规模扫描要在 30/60/120/240/全部 五个训练集大小上各训一次。三个
              决定，理由都写在 D-35：
              ① **不用"取前 n 行"**：官方 `train_index` 是升序的（0,1,2,…），前 n 行就是原始 CSV
                 的前 n 行，而同一条文献往往连着几十行、测的是同一批样品——那会把"样本量效应"
                 换成"文献顺序效应"，正好是 H4 要排除的东西（C-3 对 PolyBench 有同款警告）；
              ② **固定种子打乱后取前缀**：`numpy` 的 `default_rng(seed).permutation` 是唯一随机源，
                 每行等概率，所以任何前缀都是训练集的无偏子集；种子写死在实验配置里；
              ③ **嵌套**：`n=30` 的行集合是 `n=60` 的真子集，五档只差"加进去的行"，不换样品——
                 这样的曲线才能读成"数据变多"的效应。
              **不做值域分层**，这一条反直觉但要说明：OpenPoly 的训练行本身已经是按 10 个等宽箱
              分层抽出来的（`rule_replicated` 协议），它的无偏子集不会新增偏差；而 n=30 时按比例
              分层每箱连 1 行都分不到（Tg 最高那箱全量只有 4 行），分层在低档上只是更精细的噪声。
              「低 n 端天然覆盖不到全值域」是**要观察的现象**，不是要修掉的缺陷（见 D-35）。
        输入：train_rows —— 该折该次实际可用的训练行（已剔除建图失败）；n_level —— `"30"`…`"all"`
              或 None；seed —— 子集种子（来自实验配置）。
        输出：(取出的行, 说明字典)。说明里有 `n_train_pool` / `n_train_requested` /
              `n_level_saturated` / `n_subset_seed` / `n_subset_sha256`。
        数据来源：`splits/**`（只读）。配置来源：实验配置的 `subset.seed`。
        副作用：无（纯函数）。异常：ArmError —— 档位不是正整数也不是 `all`。
    English:
        Five training-set sizes for the OpenPoly learning curve, with three decisions argued in
        D-35: (1) **no "first n rows"** - the official train_index is ascending, so a prefix is the
        head of the raw CSV, and one paper's samples sit together - that would trade a sample-size
        effect for a literature-order effect; (2) **a fixed-seed permutation, then a prefix** - one
        RNG, every row equally likely, so every prefix is an unbiased subset; (3) **levels nest**, so
        a level adds rows instead of swapping them. **No value-bin stratification**: the training
        rows are already a 10-bin stratified draw, and at n=30 the sparsest bin cannot even supply
        one row - low-n bins being under-covered is the phenomenon H4 measures, not a bug to hide.
        Returns: (rows, info). Raises ArmError when a level is neither a positive int nor "all".
    """
    rows = np.asarray(train_rows, dtype=np.int64)
    pool = int(rows.size)
    info: dict[str, Any] = {"n_train_pool": pool}
    if n_level is None:
        return rows, info
    text = str(n_level).strip().lower()
    if text == "all":
        requested = pool
        saturated = False
    else:
        try:
            requested = int(text)
        except ValueError as error:
            raise ArmError(f"n 档 {n_level!r} 既不是正整数也不是 `all`") from error
        if requested <= 0:
            raise ArmError(f"n 档 {n_level!r} 必须为正（0 行训练的 R² 无定义）")
        # 请求值顶到池子大小就是"这一档已经是全量"：op_dielectric 的 train 只有 235 行、
        # op_tm 只有 134 行，所以两者的 240 档都取不满。**如实记 saturated**，不假装取到了 240：
        # 这两档的 R² 应当与 all 档逐位相同（同一条数据、同一个种子），可以当自检用。
        # A request at or above the pool means "this level is already everything": dielectric has
        # 235 training rows and Tm has 134, so neither can fill a 240 level. `saturated` says so
        # instead of pretending; those runs must score bit-identically to the "all" level.
        saturated = requested >= pool
    used = min(requested, pool)
    order = np.random.default_rng(int(seed)).permutation(pool)
    picked = rows[order[:used]]
    digest = hashlib.sha256(np.sort(picked).astype("<i8").tobytes()).hexdigest()
    info.update(
        {
            "n_level": text,
            "n_train_requested": int(requested),
            "n_level_saturated": bool(saturated),
            "n_subset_seed": int(seed),
            # 排过序再哈希：定的是"哪些行"，与 DataLoader 的批序无关，换台机器也一致。
            # Hash the sorted rows: this pins *which* rows, independent of batch order or machine.
            "n_subset_sha256": f"sha256:{digest}",
        }
    )
    return picked, info


def enumerate_runs(
    config: ArmConfig,
    *,
    tasks: Sequence[str] | None = None,
    folds: Sequence[str] | None = None,
    seeds: Sequence[int] | None = None,
    n_levels: Sequence[str] | None = None,
    extra_folds: Sequence[str] | None = None,
) -> list[RunKey]:
    """把配置里的网格展开成 run 列表；可按任务/折/种子/n 档打子集（冒烟与三行重跑用）。
    Expand the configured grid into a run list, filterable by task/fold/seed/n-level.

    中文：
        用途：48 条 = 6 个 PolyBench × 1 折 × 3 种子 + 2 个 PB26 × 5 折 × 3 种子。
              有 `n_levels` 的配置（B-5 的 OpenPoly 扫描）多一个维度：45 条 = 3 性质 × 5 折 ×
              3 种子，再乘 5 个 n 档 = 225 条。**一个 `n_levels` 键决定有没有这一维**，不靠
              benchmark 名字猜——猜错了会静默少跑 180 条。
              **折名不做匹配猜测**：`--folds` 给的名字必须在配置里存在，否则报错——
              静默跳过会让人以为"跑完了"，而实际一条没跑。
              `extra_folds` 是**显式**的例外：只对 openpoly 生效，用来跑配置里没列的划分
              （B-5 的 5–9 折敏感性分析）。为什么不能把 5–9 写进配置？配置的字节哈希就是
              225 条的 `config_hash`，加两个折名等于换了一份配置，`write_n_index()` 会把
              已交付的 225 条当成"旧协议"全部剔除。折是**数据**，不是超参。
        输入：config；tasks / folds / seeds / n_levels / extra_folds —— 可选白名单。
        输出：`RunKey` 列表（按配置顺序，n 档在折之外那一层）。
        数据来源：`configs/arms/a3_graph.yaml` 与 `configs/tasks/*.yaml`。
        配置来源：`runs`、`seeds`、`n_levels`（可在 `runs.<benchmark>` 里覆盖顶层）。
        副作用：只读。
        异常：ArmError —— 白名单里出现配置中没有的任务/折/n 档。
    English:
        Purpose: 48 = 6 PolyBench x 1 fold x 3 seeds + 2 PB26 x 5 folds x 3 seeds. A config with
        `n_levels` (B-5's OpenPoly scan) gains one dimension: 45 x 5 levels = 225. **Presence of the
        key decides whether the dimension exists**, never the benchmark name - guessing wrong would
        silently skip 180 runs. The fold filter is **not** a prefix match: a name missing from the
        config raises, because a silent skip looks exactly like "all done" while nothing ran.
        `extra_folds` is the explicit exception: openpoly only, for folds the config does not list
        (B-5's 5-9 sensitivity run). They cannot go into the config because its byte hash *is* the
        `config_hash` of the 225 delivered records - adding two names would make `write_n_index()`
        treat those 225 as an older protocol and drop them. Folds are data, not hyper-parameters.
        Returns: RunKeys in config order. Raises: ArmError on an unknown task/fold/n-level.
    """
    grid = config.raw["runs"]
    if not isinstance(grid, dict):
        raise ArmError("配置的 `runs` 必须是映射（polybench / polybench26 两条）")
    # 臂名从配置读，不从模块常量猜：A3/A4 共用这份网格展开，只有 `arm` 不同。
    # The arm name comes from the config, not from a module constant: A3 and A4 share this
    # expansion and differ only in `arm`.
    arm_name = str(config.raw.get("arm") or ARM)
    if arm_name not in KNOWN_ARMS:
        raise ArmError(f"配置里的 arm={arm_name!r} 不在 schema 的枚举 {list(KNOWN_ARMS)} 里")
    wanted_tasks = set(tasks) if tasks else None
    wanted_folds = set(folds) if folds else None
    wanted_seeds = set(seeds) if seeds is not None and len(seeds) else None
    wanted_levels = {str(x).lower() for x in n_levels} if n_levels else None
    # 配置外的折：只有调用方显式给出才追加，且只在 openpoly 上追加（别的基准没有第二套文件）。
    # Folds outside the config: appended only when the caller names them, and only for openpoly.
    extra_folds_list = [str(name) for name in (extra_folds or [])]

    top_levels = config.raw.get("n_levels")
    if top_levels is not None and (not isinstance(top_levels, list) or not top_levels):
        raise ArmError(f"`n_levels` 必须是非空列表，现在是 {top_levels!r}")

    keys: list[RunKey] = []
    known_tasks: set[str] = set()
    known_folds: set[str] = set()
    known_levels: set[str] = set()
    for benchmark, block in grid.items():
        if not isinstance(block, dict):
            raise ArmError(f"`runs.{benchmark}` 必须是映射（tasks / folds 两个键）")
        block_tasks = list(block.get("tasks", []))
        block_folds = [str(name) for name in block.get("folds", [])]
        if extra_folds_list and benchmark == "openpoly":
            block_folds.extend(name for name in extra_folds_list if name not in block_folds)
        if not block_tasks or not block_folds:
            raise ArmError(f"`runs.{benchmark}` 的 tasks/folds 不能为空")
        block_levels = block.get("n_levels", top_levels)
        # `None` 是"这一块没有 n 维"，`"all"` 是"有 n 维且全量档"——两者 run_id 不同形，
        # 所以这里必须分开，不能把 None 当成一个档位塞进去。
        # None means "no n dimension here"; "all" is a real level. Their run_ids differ, so the
        # two can never be collapsed into one value.
        levels: list[str | None] = (
            [None] if block_levels is None else [str(x).lower() for x in block_levels]
        )
        for task in block_tasks:
            task_config = L.load_task_config(task)
            actual = str(task_config["benchmark"])
            if actual != benchmark:
                raise ArmError(
                    f"{task} 在配置里挂在 `runs.{benchmark}` 下，但 configs/tasks 说它是 {actual}"
                )
            known_tasks.add(task)
            for level in levels:
                if level is not None:
                    known_levels.add(level)
                if wanted_tasks is not None and task not in wanted_tasks:
                    continue
                if wanted_levels is not None and level not in wanted_levels:
                    continue
                for fold in block_folds:
                    known_folds.add(fold)
                    if wanted_folds is not None and fold not in wanted_folds:
                        continue
                    for seed in config.raw["seeds"]:
                        if wanted_seeds is not None and int(seed) not in wanted_seeds:
                            continue
                        keys.append(
                            RunKey(
                                task=task,
                                benchmark=benchmark,
                                fold=fold,
                                seed=int(seed),
                                arm=arm_name,
                                n_level=level,
                            )
                        )

    if wanted_tasks is not None:
        unknown = sorted(wanted_tasks - known_tasks)
        if unknown:
            raise ArmError(f"--tasks 里这些任务不在配置网格中：{unknown}；配置里的是 {sorted(known_tasks)}")
    if wanted_folds is not None:
        unknown = sorted(wanted_folds - known_folds)
        if unknown:
            raise ArmError(f"--folds 里这些折不在配置网格中：{unknown}；配置里的是 {sorted(known_folds)}")
    if wanted_levels is not None:
        unknown = sorted(wanted_levels - known_levels)
        if unknown:
            raise ArmError(f"--n-levels 里这些档不在配置网格中：{unknown}；配置里的是 {sorted(known_levels)}")
    if not keys:
        raise ArmError("网格为空：检查 --tasks/--folds/--seeds/--n-levels 是否互相排除掉了")
    return keys


# ---- 3. 早停用的验证集 / The validation set used for early stopping ----


def carve_validation(
    train_index: np.ndarray, *, fraction: float, seed: int
) -> tuple[np.ndarray, np.ndarray]:
    """从**训练索引内部**切出一份固定留出，给早停用；返回 (训练, 验证)，两者都升序。
    Carve a fixed hold-out **inside the training indices** for early stopping; both sorted ascending.

    中文：
        用途：PolyBench 官方只发 70/30，没有验证集，而 `patience 15` 必须有东西可监控。
              三条约束：
              ① **只动 train**——测试集永远不参与任何选择（红线 3）；
              ② **与模型种子无关**（`seed` 是配置里的固定数，不是 run 的 seed），三个种子共用
                 同一份留出，否则"种子间的差"里会混进"验证集不同"；
              ③ 升序返回——批序由 DataLoader 的 shuffle 决定，索引本身的顺序不该再引入一次随机。
              切完训练侧为空就报错：留出 100% 等于没训练。
        输入：train_index —— 官方训练索引；fraction —— 留出比例（0,1)；seed —— 固定种子。
        输出：(train, val)，两个 int64 数组，升序、互斥。
        数据来源：调用方（`SplitSpec`）。配置来源：`training.val_fraction_if_missing` 与
        `training.val_split_seed`。副作用：无（不修改入参）。异常：ArmError —— 比例不合法或切空。
    English:
        Purpose: PolyBench ships 70/30 only and `patience 15` needs something to monitor. Three
        constraints: (1) **train only** - the test split never informs any choice (red line 3);
        (2) **independent of the model seed** - the split seed is a fixed config number, so all three
        model seeds share one hold-out and a seed-to-seed difference cannot be a different val set;
        (3) ascending output, so the only randomness in batch order is the loader's shuffle.
        Args: train_index; fraction; seed. Returns: (train, val), int64, sorted, disjoint.
        Raises: ArmError on an illegal fraction or an empty side.
    """
    index = np.asarray(train_index, dtype=np.int64)
    if index.size < 2:
        raise ArmError(f"训练集只有 {index.size} 行，切不出验证集")
    if not 0.0 < float(fraction) < 1.0:
        raise ArmError(f"val_fraction_if_missing={fraction} 必须在 (0,1) 内")
    n_val = int(round(index.size * float(fraction)))
    n_val = min(max(n_val, 1), index.size - 1)
    rng = np.random.default_rng(int(seed))
    order = rng.permutation(index.size)
    val = np.sort(index[order[:n_val]])
    train = np.sort(index[order[n_val:]])
    if train.size == 0 or val.size == 0:
        raise ArmError("留出之后训练集或验证集为空：比例不对")
    return train, val


# ---- 4. 指标 / Metrics ----


def metrics(y_true: Iterable[float], y_pred: Iterable[float]) -> dict[str, float]:
    """测试集三个指标：r2 / mae / rmse（键名被 schema 钉死，别改名）。
    The three test metrics; the key names are pinned by the schema, do not rename.

    中文：
        用途：手写而不是 `sklearn.metrics`——三个公式各有 6 行，手写让"用哪个 R² 定义"这件事
              留在代码里可读（`1 - SS_res/SS_tot`，总体均值做分母），且不引入对 sklearn 版本的
              依赖（两个环境各钉了一个版本）。
              `r2` 在测试集只有 1 行时是 NaN，**如实给 NaN 而不是 0**：0 会被当成"和均值一样差"，
              NaN 会被下游当成"这一条不可用"，后者才是真的。
        输入：y_true / y_pred —— 等长序列。
        输出：dict，键 `r2` / `mae` / `rmse`，值是 Python float。
        数据来源：调用方。配置来源：无。副作用：无。异常：ArmError —— 长度不一致或为空。
    English:
        Purpose: hand-written rather than sklearn's, because three formulas of six lines each keep
        "which R^2" readable in the code and drop a version dependency (each environment pins its
        own). With one test row `r2` is NaN and is reported as NaN, not 0: 0 reads as "as bad as the
        mean", NaN reads as "not available", and the latter is the truth.
        Args: y_true, y_pred. Returns: {'r2','mae','rmse'} as Python floats.
        Raises: ArmError on a length mismatch or an empty input.
    """
    truth = np.asarray(list(y_true), dtype=np.float64)
    pred = np.asarray(list(y_pred), dtype=np.float64)
    if truth.size == 0:
        raise ArmError("指标算不了：测试集为空")
    if truth.shape != pred.shape:
        raise ArmError(f"指标算不了：y_true {truth.shape} 与 y_pred {pred.shape} 不等长")
    finite = np.isfinite(truth) & np.isfinite(pred)
    if not finite.all():
        truth, pred = truth[finite], pred[finite]
    if truth.size == 0:
        raise ArmError("指标算不了：过滤掉非有限值后为空")
    residual = truth - pred
    mae = float(np.mean(np.abs(residual)))
    rmse = float(math.sqrt(float(np.mean(residual**2))))
    total = float(np.sum((truth - truth.mean()) ** 2))
    r2 = float("nan") if total == 0.0 else float(1.0 - float(np.sum(residual**2)) / total)
    return {"r2": r2, "mae": mae, "rmse": rmse}


# ---- 5. 任务数据：装配产物 + 图 + 划分 / Task data: pack + graphs + split ----


@dataclass
class TaskData:
    """一个任务的全部输入：行表、划分、每行一张图（或 None）、标签。
    Everything one task needs: the row table, the split, one graph per row (or None) and labels.

    中文：
        用途：`train_one()` 的输入。图的构建是**任务级**的（7,367 行只建一次，5 折 × 3 种子复用），
              所以它不放在 `train_one()` 里面——否则同一个任务的 15 条 run 会把同样的图重建 15 次。
        字段：`task_id`、`benchmark`、`n_rows`、`split`（`SplitSpec`）、`y`（float64，原单位）、
              `row_id`（0..n-1）、`mgs`（每行一个 chemprop `MolGraph`，解析失败为 None）、
              `psmiles`（每行的规范式）、`n_failed`、`failures`（前若干条失败原因）、`built_in_s`。
        数据来源：`data/processed/features/<task>.npz`（B-1）+ `splits/**`（只读）。
        配置来源：`configs/tasks/<task>.yaml`（经 `loaders`）。副作用：只读。
        English: fields are the task id, benchmark, row count, SplitSpec, labels in original units,
        row ids, one MolGraph per row (None on a parse failure), canonical strings, failure counts
        and the graph-building wall clock. Graphs are built **per task**, not per run, because
        15 runs of one task would otherwise rebuild the same 7,367 graphs 15 times.
    """

    task_id: str
    benchmark: str
    n_rows: int
    split: L.SplitSpec
    y: np.ndarray
    row_id: np.ndarray
    mgs: list[Any]
    psmiles: list[str]
    n_failed: int
    failures: list[tuple[str, str]]
    built_in_s: float

    def fold(self, name: str) -> L.FoldSpec:
        """按名字取一折；名字不在就报错并列出可用的折。
        Fetch one fold by name, raising with the available names.

        中文：输入：name。输出：`FoldSpec`。异常：ArmError —— 折名不存在。
        English: Args: name. Returns: a FoldSpec. Raises ArmError on an unknown name.
        """
        for fold in self.split.folds:
            if fold.name == name:
                return fold
        raise ArmError(
            f"{self.task_id} 没有折 {name!r}；可用的折：{list(self.split.fold_names)}"
        )


# 图缓存：同一进程里连续跑多个折/种子时复用，键是任务 ID。
# Graph cache, keyed by task id, so successive folds/seeds in one process reuse the graphs.
_GRAPH_CACHE: dict[str, TaskData] = {}


def build_task_data(task_id: str, *, limit: int | None = None, use_cache: bool = True) -> TaskData:
    """读装配产物、建每行的图、读划分，打包成 `TaskData`。
    Read the pack, build one graph per row, read the split, and return a TaskData.

    中文：
        用途：把 B-1/B-1.5 的三件产物接到一起。三件事值得说明：
              ① **行序不重排**：`row_id` 就是 `pack["row_id"]`（0..n-1），与 `splits` 的
                 `test_index` 同一坐标系；这里只做切片（`--limit`），从不排序或过滤；
              ② **解析失败不删行**：`mgs[i]` 为 None 的行留在表里，标签留着，测试时跳过并在记录里
                 记账（`n_graph_failed`）——删行会让后面全部前移，测试集悄悄换样品；
              ③ **同一结构只建一次图**：按规范式去重，行表里重复的结构共享同一个 `MolGraph`
                 （OpenPoly 443 行只有 304 个结构）。`BatchMolGraph` 只读入参，共享是安全的。
        输入：task_id；limit —— 只取前 N 行（冒烟；调用方负责不写正式产物）；use_cache ——
              同进程内复用。
        输出：`TaskData`。
        数据来源：`data/processed/features/`、`splits/`。配置来源：`configs/tasks/`。
        副作用：读盘、写日志。异常：ArmError —— 缺装配产物或图解析全失败。
    English:
        Three things worth stating: (1) **row order is never rebuilt** - `row_id` is the pack's own
        0..n-1 index, the same coordinate system as `test_index`; this function only slices for
        `--limit`, never sorts or filters; (2) **a parse failure does not drop the row** - `mgs[i]`
        stays None, the label stays, the test step skips it and the record counts it, because
        dropping shifts every later row and silently swaps the test set; (3) **one graph per unique
        structure** - repeated rows share one MolGraph (OpenPoly's 443 rows hold 304 structures),
        which is safe because `BatchMolGraph` only reads its inputs.
        Args: task_id; limit (first N rows, smoke; the caller keeps it out of official artefacts);
        use_cache. Returns: a TaskData. Side effects: reads disk, logs.
        Raises: ArmError on a missing pack or when every graph fails to parse.
    """
    key = f"{task_id}@{limit}"
    if use_cache and key in _GRAPH_CACHE:
        return _GRAPH_CACHE[key]

    # 见模块头的说明：这两个模块要 pandas / rdkit，只在这里导入。
    # See the module header: these two need pandas/rdkit, so they are imported right here.
    from chainworth.chem import features as F
    from chainworth.chem import graph as G

    started = time.perf_counter()
    pack = F.load_pack(task_id)
    split = L.load_split(task_id)
    n_rows = int(len(pack["row_id"]))
    if limit is not None:
        n_rows = min(int(limit), n_rows)
        split = L.limit_split(split, n_rows)
        pack = {name: (value[:n_rows] if isinstance(value, np.ndarray) and value.ndim and value.shape[0] == len(pack["row_id"]) else value)
                for name, value in pack.items()}
    problems = L.check_split_matches_rows(split, n_rows)
    if problems:
        raise ArmError("；".join(problems))

    # 图级缓存：规范式 -> MolGraph。同一结构在行表里出现多次时只建一次。
    # Per-task cache: canonical string -> MolGraph, built once per structure.
    unique: dict[str, Any] = {}
    failures: list[tuple[str, str]] = []
    mgs: list[Any] = []
    psmiles = [str(value) for value in pack["psmiles"]]
    for text in dict.fromkeys(psmiles):
        if not text:
            failures.append(("", "no_structure"))
            continue
        tensors = G.build_graph(text)
        if tensors is None:
            failures.append((text, "parse_failed"))
            continue
        unique[text] = _to_molgraph(tensors)
    for text in psmiles:
        mgs.append(unique.get(text))
    n_failed = sum(1 for item in mgs if item is None)
    if n_failed == len(mgs):
        raise ArmError(f"{task_id} 的 {len(mgs)} 行一张图都建不出来：结构表或装配产物有问题")
    built_in_s = time.perf_counter() - started
    LOGGER.info(
        "%s：%d 行 / %d 个唯一结构 / 图失败 %d 行；建图 %.1f 秒",
        task_id,
        n_rows,
        len(unique),
        n_failed,
        built_in_s,
    )
    data = TaskData(
        task_id=task_id,
        benchmark=split.benchmark,
        n_rows=n_rows,
        split=split,
        y=np.asarray(pack["y"], dtype=np.float64),
        row_id=np.asarray(pack["row_id"], dtype=np.int64),
        mgs=mgs,
        psmiles=psmiles,
        n_failed=n_failed,
        failures=failures[:50],
        built_in_s=built_in_s,
    )
    if use_cache:
        _GRAPH_CACHE[key] = data
    return data


def _to_molgraph(tensors: G.GraphTensors) -> Any:
    """`GraphTensors`（numpy）→ chemprop 的 `MolGraph`（唯一转换点，torch 在这里才被导入）。
    GraphTensors (numpy) -> chemprop's MolGraph; the only conversion point, where torch is imported.

    中文：
        用途：B-1 的建图结果与 chemprop 的批处理之间只允许这一处转换。**不许在别处自己拼
              `MolGraph`**：`edge_index` 的 dtype 或 `rev_edge_index` 少一处，模型照样能训、
              指标照样出，只是没有可比性（同 `graph.to_pyg()` 的理由）。
              `MolGraph` 是纯 dataclass、不 import torch，但它的字段会被 `BatchMolGraph` 拼成张量，
              所以转换放在训练模块里而不是 `chem/graph.py`。
        输入：tensors —— `build_graph()` 的输出。
        输出：`chemprop.data.MolGraph`。
        数据来源：调用方。配置来源：无。副作用：无。异常：ImportError —— 没装 chemprop。
    English:
        The only allowed conversion between B-1's graphs and chemprop's batching; building a
        `MolGraph` elsewhere lets a wrong dtype or a missing `rev_edge_index` still train and still
        produce metrics that are not comparable. Args: tensors. Returns: a MolGraph.
        Raises: ImportError when chemprop is absent.
    """
    from chemprop.data import MolGraph

    return MolGraph(
        V=np.ascontiguousarray(tensors.atom),
        E=np.ascontiguousarray(tensors.bond),
        edge_index=np.ascontiguousarray(tensors.edge_index),
        rev_edge_index=np.ascontiguousarray(tensors.rev_edge_index),
    )


# ---- 6. 设备与环境 / Device and environment ----


def resolve_device(prefer: str = "cuda") -> str:
    """选设备：`cuda` 优先，没有卡就退到 `cpu`（不许写死 GPU）。
    Pick the device: cuda if available, otherwise cpu. Never hard-code the GPU.

    中文：用途：分工_B §4 交接协议第 2 条。输入：prefer。输出："cuda" 或 "cpu"。
          数据来源：torch。配置来源：`device.prefer`。副作用：无。异常：ImportError —— 没装 torch。
    English: division B section 4, handover protocol item 2. Args: prefer. Returns: 'cuda'/'cpu'.
    """
    import torch

    if prefer == "cpu":
        return "cpu"
    return "cuda" if torch.cuda.is_available() else "cpu"


def _driver_version() -> str:
    """从环境指纹里取驱动版本（不现场跑 nvidia-smi：记录要与指纹文件说同一件事）。
    Read the driver version from the environment fingerprint instead of shelling out to nvidia-smi.

    中文：用途：验收③要求 run 记录带驱动版本，而 `env_fingerprint` 指向的那份文件里就有；
              两处各查一次会出现"记录说 591.86、指纹说 572.83"的分歧。
          输入：无。输出：字符串（取不到就 "unknown"）。数据来源：`results/env_fingerprints/`。
          副作用：只读。异常：无（取不到不致命）。
    English: acceptance item 3 wants the driver version, and the fingerprint file already holds it;
    querying twice invites "record says 591.86, fingerprint says 572.83". Returns "unknown" if the
    file is unreadable.
    """
    path = REPO_ROOT / ENV_FINGERPRINT
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return "unknown"
    driver = payload.get("driver") or {}
    return str(driver.get("driver_version") or "unknown")


def gpu_block(device: str, *, cudnn_deterministic: bool) -> dict[str, Any] | None:
    """`cuda` 时给出记录里的 `gpu` 块（schema 要求 device=cuda 必填）；CPU 返回 None。
    The record's `gpu` block when the device is cuda (the schema requires it there); None on CPU.

    中文：输入：device、cudnn_deterministic。输出：dict 或 None。数据来源：torch + 环境指纹。
          配置来源：`device.cudnn_deterministic`。副作用：无。异常：无。
    English: Args: device, cudnn_deterministic. Returns: a dict or None.
    """
    if device != "cuda":
        # 先判设备再 import torch：`DSC6020_Project_Core` 里没有 torch，而 A4 的线性头可以用
        # numpy 在 CPU 上解（见 `arms/polybert.py` 的 `_ridge_solve`），那条路径不该因为"记录里
        # 要写个 gpu 块"就崩掉。
        # Test the device before importing torch: the core environment has no torch, and the A4 linear
        # head can be solved with numpy on the CPU - that path must not die for a gpu block it will
        # never write.
        return None
    import torch

    props = torch.cuda.get_device_properties(0)
    return {
        "model": torch.cuda.get_device_name(0),
        "driver": _driver_version(),
        "torch_cuda": str(torch.version.cuda),
        "cudnn_deterministic": bool(cudnn_deterministic),
        # 记的是**实际生效**的严格确定性开关，不是配置里的意图：这条 run 到底可不可复现，
        # 看这一个布尔值就够，不必回读代码去猜。判据见 `record_problems()`。
        # The *effective* strict-determinism flag, not the configured intent, so a reader can tell
        # whether the run was repeatable without going back to the code.
        "deterministic_algorithms": bool(torch.are_deterministic_algorithms_enabled()),
        "capability": f"{props.major}.{props.minor}",
        "memory_total_mib": int(round(props.total_memory / (1024**2))),
    }


# ---- 7. 训练一条 run / Train one run ----


@dataclass
class TrainOutcome:
    """一条 run 的训练结果：记录草稿 + 测试集上的真值与预测（写文件的活在外面）。
    One run's outcome: a record draft plus test-set truth and predictions (writing happens outside).

    中文：
        字段：`record`（还没写盘的记录 dict）、`row_id` / `y_true` / `y_pred`（测试集，等长）、
              `history`（每轮的 train_loss/val_loss，只进日志，不进记录——记录只放结论）。
        English: `record` is the un-written record draft; the three arrays are the test set;
        `history` goes to the log only, not into the record.
    """

    record: dict[str, Any]
    row_id: np.ndarray
    y_true: np.ndarray
    y_pred: np.ndarray
    history: list[dict[str, float]] = field(default_factory=list)


def _make_collate(mean: float, scale: float) -> Any:
    """造一个 collate：把 (MolGraph, y) 列表拼成 chemprop 的 `TrainingBatch`。
    Build a collate that turns a list of (MolGraph, y) into chemprop's `TrainingBatch`.

    中文：
        用途：官方 `collate` 读的是 `MoleculeDatapoint`（会重新解析 SMILES、重新 featurize），
              而我们手上已经是建好的图，重跑一遍既慢又多一个"两套建图代码"的机会。
              这里直接拼 `BatchMolGraph`——它就是官方批处理类，只是入口不同。
        为什么 `Y` 要标准化：`chemprop.nn.UnscaleTransform.forward` 在 `self.training` 为真时
              **原样返回**（只看 `transforms.py` 源码，不是猜的）。也就是说训练与验证阶段网络输出
              是标准化空间里的量，只有 `eval()` 之后才反变换回原单位。所以喂给 criterion 的标签
              必须是 `(y - mean) / scale`；喂原单位会让网络去追原单位、而在预测时再乘一次 scale，
              结果是预测值放大约 180 倍（实测：真值 618 K、预测 98,564 K）。这也正是 chemprop
              官方流程在建 dataloader 前调 `normalize_targets()` 的原因。
        输入：mean / scale —— **训练集**标签的均值与标准差（不许用全量）。
        输出：可给 `DataLoader(collate_fn=...)` 的函数。
        数据来源：无。配置来源：`training.target_scaling`。副作用：无。
        异常：ImportError —— 没装 chemprop/torch。
    English:
        The official collate reads `MoleculeDatapoint`, which re-parses SMILES and re-featurises;
        we already hold the graphs. This assembles `BatchMolGraph` (the official batching class)
        directly.
        Why `Y` must be standardised: `UnscaleTransform.forward` returns its input unchanged while
        `self.training` is true (read from `transforms.py`, not guessed). Training and validation
        therefore live in standardised space and only `eval()` unscales. Feeding raw labels makes the
        network chase raw values and then multiplies by scale once more at prediction time - measured
        at ~180x (truth 618 K, prediction 98,564 K). That is exactly why chemprop's own pipeline
        calls `normalize_targets()` before building the dataloader.
        Args: mean/scale of **training** labels. Returns: a collate function.
    """
    import torch
    from chemprop.data import BatchMolGraph, TrainingBatch

    def collate(batch: Sequence[tuple[Any, float]]) -> Any:
        graphs = [item[0] for item in batch]
        raw = np.asarray([float(item[1]) for item in batch], dtype=np.float64)
        scaled = (raw - float(mean)) / float(scale)
        targets = torch.tensor(scaled.reshape(-1, 1), dtype=torch.float32)
        weights = torch.ones(len(batch), dtype=torch.float32)
        return TrainingBatch(
            bmg=BatchMolGraph(graphs),
            V_d=None,
            X_d=None,
            Y=targets,
            w=weights,
            lt_mask=None,
            gt_mask=None,
        )

    return collate


def _make_history_callback() -> Any:
    """造一个只记逐轮 loss 的 Lightning 回调（日志用；不进记录）。
    Build a Lightning callback that records per-epoch losses (for the log, not the record).

    中文：
        用途：验收④要"同 seed 两次差异 < 1e-3"，而且调不通时要能看出"是没收敛还是早停太早"。
              逐轮 loss 是这两件事的共同证据，但它有 100 行、不该塞进 run 记录（记录只放结论）。
              回调**在这里才定义**，因为它的基类来自 lightning，而本模块在 core 环境（无 torch）
              里也要能被 import（测试与 `--dry-run` 都走 core）。
        输入：无。输出：一个带 `history()` 方法的回调实例。
        数据来源：Lightning 的 `callback_metrics`。配置来源：无。副作用：无。
        异常：ImportError —— 没装 lightning（只在真要训练时才可能）。
    English:
        Criterion 4 needs "same seed twice within 1e-3" and a failed run needs "did not converge" vs
        "early stopping fired too soon"; per-epoch losses are the shared evidence for both, but 100
        rows do not belong in a run record. The class is defined here, not at module level, because
        its base class comes from lightning while this module must stay importable in the core
        environment (tests and `--dry-run` both run there).
    """
    import lightning.pytorch as pl

    class EpochHistory(pl.Callback):
        """把每轮 `train_loss` / `val_loss` 抄下来的最小回调。／ Minimal per-epoch loss recorder."""

        def __init__(self) -> None:
            super().__init__()
            self._rows: list[dict[str, Any]] = []

        def on_validation_epoch_end(self, trainer: Any, pl_module: Any) -> None:
            metrics_now = trainer.callback_metrics
            row: dict[str, Any] = {"epoch": int(trainer.current_epoch)}
            for name in ("train_loss", "val_loss"):
                value = metrics_now.get(name)
                if value is not None:
                    row[name] = float(value.detach().cpu())
            if "val_loss" in row:
                self._rows.append(row)

        def history(self) -> list[dict[str, Any]]:
            """逐轮 loss 列表（只含有验证的那些轮）。／ Per-epoch losses (validation rounds only)."""
            return list(self._rows)

    return EpochHistory()


def target_scaler(y_train: np.ndarray) -> tuple[float, float]:
    """训练标签的（均值, 标准差）——**只用训练集**，测试集永远不参与。
    The (mean, std) of the **training** labels only; the test set never participates.

    中文：
        用途：`build_model()` 的反变换与 `_make_collate()` 的标准化必须用同一对数，所以抽成一个函数；
              两处各算一遍迟早会有一处忘了"只用 train"。
              方差为 0 时标准差取 1（退化为不减不除）并留一条警告：常数标签的任务本来就学不出东西，
              但让它因为除零崩掉没有意义。
        输入：y_train —— 训练集标签（原单位；NaN 允许，会先过滤）。
        输出：(mean, std)。数据来源：调用方。配置来源：`training.target_scaling`。
        副作用：无。异常：ArmError —— 全部是 NaN。
        English:
        The unscaling in `build_model()` and the standardisation in `_make_collate()` must use the
        same pair, so it is computed once; two copies would eventually disagree about "train only".
        A zero variance falls back to std=1 with a warning: a constant-label task teaches nothing,
        but crashing on a division by zero helps nobody.
    """
    finite = np.asarray(y_train, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        raise ArmError("训练标签全部是 NaN：这一折没法训")
    mean = float(finite.mean())
    std = float(finite.std())
    if not np.isfinite(std) or std == 0.0:
        LOGGER.warning("训练标签方差为 0，标准化退化为不减不除（std=1）")
        std = 1.0
    return mean, std


def build_model(config: ArmConfig, y_train: np.ndarray) -> Any:
    """按配置造官方 MPNN；标签标准化参数只用 `y_train` 算。
    Build the official MPNN from the config; the target scaler uses `y_train` only.

    中文：
        用途：`BondMessagePassing` + `NormAggregation` + `RegressionFFN` 就是 chemprop 的默认
              MPNN，`MPNN` 负责 Noam 式学习率与指标。逐项来源见 `configs/arms/a3_graph.yaml` 的注释。
              **标签标准化**：均值/标准差取自 `y_train`（不是全量），写进 `UnscaleTransform`；
              这一步是"不泄漏"的关键——用全量统计量等于把测试集的标签分布提前告诉模型。
        输入：config；y_train —— 训练集标签（原单位）。
        输出：`chemprop.models.MPNN`。
        数据来源：`configs/arms/a3_graph.yaml`。配置来源：`architecture` 与 `training`。
        副作用：无。异常：ArmError —— 训练标签全是 NaN 或方差 0。
    English:
        `BondMessagePassing` + `NormAggregation` + `RegressionFFN` is chemprop's default MPNN and
        `MPNN` supplies the Noam-like schedule and metrics. The target scaler is fitted on
        `y_train` only and written into `UnscaleTransform`: using all-label statistics would tell
        the model the test distribution in advance.
        Args: config; y_train. Returns: an MPNN. Raises: ArmError when every training label is NaN
        or the variance is zero.
    """
    import torch
    from chemprop.models import MPNN
    from chemprop.nn import (
        MAE,
        R2Score,
        RMSE,
        BondMessagePassing,
        MeanAggregation,
        NormAggregation,
        RegressionFFN,
        SumAggregation,
        UnscaleTransform,
    )

    arch = config.section("architecture")
    train_cfg = config.section("training")

    mean, std = target_scaler(y_train)

    message_passing = BondMessagePassing(
        d_v=int(arch["atom_feature_dim"]),
        d_e=int(arch["bond_feature_dim"]),
        d_h=int(arch["hidden"]),
        depth=int(arch["depth"]),
        dropout=float(arch["dropout"]),
        activation=str(arch["activation"]),
        undirected=bool(arch["undirected"]),
    )
    # 配置里写的是类名（`NormAggregation`），这里只认"聚合方式"那一段，大小写与后缀都不计较：
    # 让 `norm` / `NormAggregation` / `norm_aggregation` 三种写法指向同一件事。
    # The config carries the class name; only the "which aggregation" part matters here.
    aggregation_name = str(arch.get("aggregation", "norm")).lower().replace("aggregation", "").strip("_")
    if aggregation_name == "norm":
        aggregation = NormAggregation(norm=float(arch["aggregation_norm"]))
    elif aggregation_name == "mean":
        aggregation = MeanAggregation()
    elif aggregation_name == "sum":
        aggregation = SumAggregation()
    else:
        raise ArmError(f"不认识的 aggregation={aggregation_name!r}（norm / mean / sum）")

    loss_name = str(train_cfg["loss"]).lower()
    criterion = {"rmse": RMSE, "mae": MAE}[loss_name]()
    predictor = RegressionFFN(
        n_tasks=1,
        input_dim=int(arch["hidden"]),
        hidden_dim=int(arch["ffn_hidden"]),
        n_layers=int(arch["ffn_layers"]),
        dropout=float(arch["dropout"]),
        activation=str(arch["activation"]),
        criterion=criterion,
        output_transform=UnscaleTransform(
            mean=torch.tensor([mean], dtype=torch.float32),
            scale=torch.tensor([std], dtype=torch.float32),
        ),
    )
    return MPNN(
        message_passing=message_passing,
        agg=aggregation,
        predictor=predictor,
        batch_norm=bool(arch["batch_norm"]),
        metrics=[RMSE(), MAE(), R2Score()],
        warmup_epochs=int(train_cfg["warmup_epochs"]),
        init_lr=float(train_cfg["init_lr"]),
        max_lr=float(train_cfg["max_lr"]),
        final_lr=float(train_cfg["final_lr"]),
    )


def train_one(
    data: TaskData,
    config: ArmConfig,
    *,
    fold: str,
    seed: int,
    device: str | None = None,
    max_epochs: int | None = None,
    patience: int | None = None,
    n_level: str | None = None,
) -> TrainOutcome:
    """训一条 run，返回记录草稿与测试集预测（不写盘）。
    Train one run and return the record draft plus test predictions (no writing here).

    中文：
        用途：B-3 的核心。步骤：
              ① 取折（官方索引）→ 需要时切出训练内部 10% 当早停验证；
              ①' 有 n 档时（B-5）把训练行截到这一档——**只动训练集**，验证集与测试集
                 在所有档位上逐行相同，否则 R² 会与 n 一起变，H4 的曲线读不出因果；
              ② 建模型（标签标准化只用训练标签）；
              ③ 训练到 ≤`max_epochs` 或早停触发，保留 `val_loss` 最好的一版权重；
              ④ 用最好权重在**官方测试集**上预测一次，算 r2/mae/rmse；
              ⑤ 记录草稿：`run_id`、`split_version`（折文件的字节哈希）、`config_hash`、
                 `device`、`gpu` 块、实测墙钟、`best_epoch`、图失败行数、验证集来源。
              **测试集只在这一步被读一次**，且只用于算指标，不参与任何选择。
        输入：data —— `TaskData`；config；fold —— 官方折名；seed —— 模型种子；
              device —— 覆盖自动选择（测试用）；max_epochs / patience —— 冒烟用覆盖；
              n_level —— 训练集档位（`"30"`…`"all"` / None = 无这一维，B-3/B-4 走 None）。
        输出：`TrainOutcome`。
        数据来源：`TaskData`。配置来源：`config`（n 档种子取 `subset.seed`）。
        副作用：训练模型、占显存、写日志。异常：ArmError —— 数据与折不配套、全为失败图。
    English:
        The core of B-3: (1) take the official fold, carving the fixed 10% inner validation when the
        benchmark ships none; (1') with an n-level (B-5) trim the **training rows only** - the
        validation and test sets are row-for-row identical at every level, otherwise R² would move
        with n for two reasons at once and the H4 curve would lose its causal reading; (2) build the
        model (target scaler on training labels only); (3) train to `max_epochs` or until early
        stopping; (4) predict once on the **official test split**; (5) draft the record.
        Args: data; config; official fold name; seed; device; max_epochs/patience overrides;
        n_level (None = no such dimension, as in B-3/B-4). Returns: a TrainOutcome.
        Raises: ArmError when the fold does not fit the data or every graph failed.
    """
    import lightning.pytorch as pl
    import torch
    from torch.utils.data import DataLoader

    train_cfg = config.section("training")
    device_cfg = config.section("device")
    chosen = device or resolve_device(str(device_cfg.get("prefer", "cuda")))

    if chosen == "cuda":
        # 显存比例：一个进程一张卡，让显存泄漏杀掉这次运行而不是整个桌面会话（分工_B §2 B-3 ⑤）。
        # One process per card; a leak kills this run instead of the desktop session.
        fraction = float(device_cfg.get("per_process_memory_fraction", 0.9))
        torch.cuda.set_per_process_memory_fraction(fraction, 0)
    deterministic = bool(device_cfg.get("cudnn_deterministic", True))
    torch.backends.cudnn.deterministic = deterministic
    torch.backends.cudnn.benchmark = False
    # 确定性要**两条腿**，只钉 cudnn 那条不够：cudnn 只管卷积那类算子，而这套 MPNN 的
    # 图上聚合是 `scatter_reduce_(reduce="sum")`（chemprop 的 SumAggregation / NormAggregation），
    # 它在 CUDA 上默认走原子加，同一次加法的先后顺序每次不同。第二条腿是
    # `use_deterministic_algorithms`。实测（pb_t1_tg，2 轮，同 seed 两次）：
    #   只钉 cudnn          → 预测中位差 0.56 K、2417 个测试点里 2416 个 > 1e-3；
    #   两条腿都钉          → 逐位相同（max 差 0），2 轮与 6 轮都成立，代价约 2 倍墙钟。
    # 为什么之前一直"没通过"也记在这里：`pl.Trainer(deterministic=False)` 会在构造时**主动调用**
    # `torch.use_deterministic_algorithms(False)`，把外面设的开关抹掉。所以这个参数必须与
    # `use_deterministic_algorithms` 的取值一致，否则上面的设置等于没写。
    # Determinism needs both legs: cudnn covers conv-like ops only, while chemprop's graph
    # aggregation is `scatter_reduce_(reduce="sum")`, which uses atomics on CUDA. Measured on
    # pb_t1_tg (2 epochs, same seed twice): cudnn-only gives a 0.56 K median gap; both legs give a
    # bit-identical result at ~2x wall clock. And `pl.Trainer(deterministic=False)` *actively
    # disables* `use_deterministic_algorithms` in its constructor - which is why the switch had to
    # be passed through here as well, not just set beforehand.
    torch.use_deterministic_algorithms(deterministic)

    spec_fold = data.fold(fold)
    train_index = np.asarray(spec_fold.train_index, dtype=np.int64)
    val_index = np.asarray(spec_fold.val_index, dtype=np.int64)
    test_index = np.asarray(spec_fold.test_index, dtype=np.int64)
    val_source = "official"
    if val_index.size == 0:
        train_index, val_index = carve_validation(
            train_index,
            fraction=float(train_cfg["val_fraction_if_missing"]),
            seed=int(train_cfg["val_split_seed"]),
        )
        val_source = f"inner_{float(train_cfg['val_fraction_if_missing']):.0%}_of_train"
    if test_index.size == 0:
        raise ArmError(f"{data.task_id}/{fold} 的测试集为空：这条 run 不该存在")

    # 只保留建图成功的行；失败的行走记录里的计数，不静默丢掉（验收：解析失败要计数并列 ID）。
    # Keep only rows whose graph exists; failures are counted in the record, never dropped silently.
    def usable(index: np.ndarray) -> np.ndarray:
        return np.asarray([i for i in index if data.mgs[int(i)] is not None], dtype=np.int64)

    train_rows = usable(train_index)
    val_rows = usable(val_index)
    test_rows = usable(test_index)
    if train_rows.size == 0 or val_rows.size == 0 or test_rows.size == 0:
        raise ArmError(
            f"{data.task_id}/{fold}：去掉建图失败的行后 train/val/test = "
            f"{train_rows.size}/{val_rows.size}/{test_rows.size}，有空的，这条 run 不能跑"
        )
    # n 档只截**训练**行。验证行与测试行在这里一格都不动：H4 要读的是"训练集变大"的效应，
    # 测试集若跟着变小，R² 的分子分母同时变，曲线就没有解释力了（D-35）。
    # The n-level trims **training** rows only. Validation and test rows are untouched here: H4
    # reads the effect of more training data, and a shrinking test set would move both numerator
    # and denominator of R² at once, leaving the curve uninterpretable.
    n_info: dict[str, Any] = {}
    if n_level is not None:
        subset_cfg = config.raw.get("subset") or {}
        if "seed" not in subset_cfg:
            raise ArmError(
                f"{config.path.name} 有 n 档但没有 `subset.seed`：子集取法必须钉死一个种子，"
                "否则同一条 run 换个进程就换一批训练行（D-35）"
            )
        train_rows, n_info = n_level_rows(train_rows, n_level, seed=int(subset_cfg["seed"]))
        if train_rows.size == 0:
            raise ArmError(f"{data.task_id}/{fold}/n={n_level}：截完训练行为 0，这条 run 不能跑")

    pl.seed_everything(int(seed), workers=True)
    mean, std = target_scaler(data.y[train_rows])
    collate = _make_collate(mean, std)
    batch_size = int(train_cfg["batch_size"])
    num_workers = int(device_cfg.get("num_workers", 0))
    pin = chosen == "cuda"

    train_loader = DataLoader(
        [(data.mgs[int(i)], float(data.y[int(i)])) for i in train_rows],
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate,
        num_workers=num_workers,
        pin_memory=pin,
    )
    val_loader = DataLoader(
        [(data.mgs[int(i)], float(data.y[int(i)])) for i in val_rows],
        batch_size=128,
        shuffle=False,
        collate_fn=collate,
        num_workers=num_workers,
        pin_memory=pin,
    )
    test_loader = DataLoader(
        [(data.mgs[int(i)], float(data.y[int(i)])) for i in test_rows],
        batch_size=128,
        shuffle=False,
        collate_fn=collate,
        num_workers=num_workers,
        pin_memory=pin,
    )

    epochs = int(max_epochs if max_epochs is not None else train_cfg["max_epochs"])
    wait = int(patience if patience is not None else train_cfg["patience"])
    callbacks: list[Any] = [_make_history_callback()]
    checkpoint_dir = None
    if wait > 0:
        from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint

        checkpoint_dir = REPO_ROOT / "outputs" / "a3_ckpt" / (
            f"{data.task_id}_{fold}_seed{seed}" + (f"_n{n_level}" if n_level is not None else "")
        )
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        monitor = str(train_cfg["monitor"])
        callbacks.append(
            EarlyStopping(monitor=monitor, patience=wait, mode="min", check_finite=True)
        )
        callbacks.append(
            ModelCheckpoint(
                dirpath=checkpoint_dir,
                filename="best",
                monitor=monitor,
                mode="min",
                save_top_k=1,
                save_weights_only=True,
                auto_insert_metric_name=False,
            )
        )

    trainer = pl.Trainer(
        max_epochs=epochs,
        accelerator=chosen,
        devices=1,
        logger=False,
        enable_checkpointing=bool(checkpoint_dir),
        enable_progress_bar=False,
        enable_model_summary=False,
        callbacks=callbacks,
        # 关掉开场自检：它会在第 0 轮之前跑一次验证，让 `len(history)` 多出一行、
        # `epochs_run` 与 `best_epoch` 全部偏 1。关掉之后"验证轮数 = 训练轮数"。
        # Disable the sanity check: it validates once before epoch 0, adding a phantom row to the
        # history and skewing `epochs_run`/`best_epoch` by one.
        num_sanity_val_steps=0,
        # 确定性：**必须与上面 `torch.use_deterministic_algorithms(deterministic)` 一致**。
        # Lightning 在 `__init__` 里就会调用 `torch.use_deterministic_algorithms(deterministic)`，
        # 传 False 等于把上面刚设的开关当场关掉——这是本臂"同 seed 两次差 65 K"的真正原因，
        # 而不是硬件做不到（同机实测 strict 下逐位相同）。代价是约 2 倍墙钟，已计入预算。
        # Must match `torch.use_deterministic_algorithms(deterministic)` above: Lightning calls it
        # inside `__init__`, so passing False silently undoes the setting - that was the real cause
        # of the 65 K gap between same-seed runs, not a hardware limit (strict is bit-identical on
        # this machine). Costs ~2x wall clock, accounted for in the budget.
        deterministic=deterministic,
    )

    model = build_model(config, data.y[train_rows])
    started = time.perf_counter()
    try:
        trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)
        best_path = None
        if checkpoint_dir is not None:
            candidate = checkpoint_dir / "best.ckpt"
            best_path = str(candidate) if candidate.is_file() else None
        predicted = trainer.predict(model, dataloaders=test_loader, ckpt_path=best_path)
    finally:
        wall = time.perf_counter() - started
        if checkpoint_dir is not None:
            for leftover in checkpoint_dir.glob("*.ckpt"):
                # 权重不入库（.gitignore 有 *.ckpt），也没有入库价值：config_hash + seed 能重建。
                # Weights are not committed and have no record value: config_hash + seed rebuild them.
                try:
                    leftover.unlink()
                except OSError:                     # pragma: no cover - 文件被占用时留一句日志
                    LOGGER.warning("删不掉临时权重 %s（不影响记录）", leftover)

    y_pred = torch.cat([item.reshape(-1) for item in predicted]).detach().cpu().numpy()
    y_pred = np.asarray(y_pred, dtype=np.float64).reshape(-1)
    y_true = np.asarray(data.y[test_rows], dtype=np.float64)
    if y_pred.size != y_true.size:
        raise ArmError(
            f"{data.task_id}/{fold}/seed{seed}：预测 {y_pred.size} 条 ≠ 测试集 {y_true.size} 条"
        )

    best_score = None
    for callback in callbacks:
        if hasattr(callback, "best_model_score") and callback.best_model_score is not None:
            best_score = float(callback.best_model_score.detach().cpu())
    history = callbacks[0].history()          # `_make_history_callback()` 的结果永远是第一个
    # 轮数用"验证跑过几轮"计数，不用 `trainer.current_epoch`：后者在 Lightning 2.6 里是
    # 0 基且收尾会多 1，实测 max_epochs=2 时它给 3。验证每轮都跑，所以 `len(history)` 就是
    # 真正训过的轮数，`best_epoch` 也用它换算成 1 基，两个数在同一把尺子上。
    # Epoch count comes from "how many validation rounds ran", not `trainer.current_epoch`
    # (0-based in Lightning 2.6 and one too high at the end: max_epochs=2 reported 3).
    epochs_run = len(history)
    best_epoch = None
    finite_history = [row for row in history if row.get("val_loss") is not None]
    if finite_history:
        best_epoch = int(min(finite_history, key=lambda row: row["val_loss"])["epoch"]) + 1

    split_version, fold_sha = fold_version(data.split, fold)
    this_run_id = run_id_for(data.task_id, fold, seed, n_level=n_level)
    record: dict[str, Any] = {
        "run_id": this_run_id,
        "status": "ok",
        "benchmark": data.benchmark,
        "task": data.task_id,
        "arm": ARM,
        "seed": int(seed),
        "split_version": split_version,
        "split_fold": fold,
        "config_hash": config_hash(config),
        "n_train": int(train_rows.size),
        "n_test": int(test_rows.size),
        "n_val": int(val_rows.size),
        "metrics": _json_safe(metrics(y_true, y_pred)),
        "wall_clock_s": round(wall, 3),
        "device": chosen,
        "env_fingerprint": ENV_FINGERPRINT,
        "predictions": f"results/aggregate/predictions/{this_run_id}.parquet",
        "epochs_run": epochs_run,
        "early_stopped": epochs_run < epochs,
        "best_epoch": best_epoch,
        "best_val_rmse": None if best_score is None else _json_safe(best_score * std),
        "val_loss_normalized": None if best_score is None else _json_safe(best_score),
        "val_loss_unit": "rmse_original（已乘回 scale；val_loss_normalized 是标准化单位）",
        "val_source": val_source,
        "n_graph_failed": int(data.n_failed),
        "graph_failed_ids": [text for text, _ in data.failures[:20]],
        "target_scaling": str(train_cfg["target_scaling"]),
        "loss": str(train_cfg["loss"]),
        "batch_size": batch_size,
        "max_epochs": epochs,
        "patience": wait,
        "hyperparameters": {
            "depth": int(config.section("architecture")["depth"]),
            "hidden": int(config.section("architecture")["hidden"]),
            "aggregation": str(config.section("architecture").get("aggregation")),
            "max_lr": float(train_cfg["max_lr"]),
        },
        "notes": (
            "chemprop 2.1.0 MPNN（官方默认组件，depth/hidden/lr 见 hyperparameters）；"
            "测试集只用于打分，不参与任何选择；分子图来自 chem/graph.py（与官方 featurizer 逐位对拍）"
        ),
    }
    gpu = gpu_block(chosen, cudnn_deterministic=deterministic)
    if gpu is not None:
        record["gpu"] = gpu
    if n_info:
        # 五个字段只在有 n 档时出现：`n_train_pool` 是这一折可用的全部训练行（355/235/134），
        # `n_train_requested` 是档位名义值，`n_train`（上面）是实际喂进去的行数。
        # 三者不等就说明这一档被池子大小卡住了——op_dielectric 与 op_tm 的 240 档都是这种情况，
        # 必须让它显形，否则 D 会把 240 画在一个只有 134 行的点上（D-35）。
        # These five fields appear only when there is an n-level: the pool, the nominal request and
        # the actual count. When they disagree the level was capped by the pool - exactly what
        # happens to dielectric and Tm at 240 - and hiding it would put a 240 on a 134-row point.
        record.update(n_info)
        record["notes"] = (
            str(record["notes"])
            + "；n 档只截训练集，验证集与测试集在 5 个档位上逐行相同（D-35）；"
            "x 轴请用 `n_train`（实际行数），不要用档位名义值"
        )
    if fold_sha:
        record["split_fold_sha256"] = fold_sha
    if data.n_failed:
        LOGGER.warning(
            "%s/%s：%d 行没有图（前几条：%s），已从 train/val/test 中跳过并记进记录",
            data.task_id,
            fold,
            data.n_failed,
            [text[:24] for text, _ in data.failures[:3]],
        )

    return TrainOutcome(
        record=record,
        row_id=np.asarray(data.row_id[test_rows], dtype=np.int64),
        y_true=y_true,
        y_pred=y_pred,
        history=history,
    )


def fold_version(spec: L.SplitSpec, fold: str) -> tuple[str, str | None]:
    """一折的 `split_version`：优先用**这一折自己的文件**哈希，取不到再退回整份划分的哈希。
    A fold's `split_version`: prefer the fold file's own hash, falling back to the split's digest.

    中文：
        用途：`split_version` 是"划分文件的身份"（schema）。PolyBench 一折就一个 JSON，整份哈希
              就是这一折的哈希；PolyBench26 的 `manifest.json` 只描述五折，真正定义折的是
              `fold_XX.npz`；OpenPoly 是五个 JSON 各一折、`sha256` 是五份拼起来的。
              所以这里按基准算出**这一折**的文件。取不到文件就退回 `spec.sha256`，
              并在返回值里给出 None，让调用方知道"这条记录只有整份哈希"。
        输入：spec；fold —— 折名。
        输出：(split_version, 折文件哈希或 None)。格式 `<12 位>@<仓库相对路径>`。
        数据来源：`splits/**`（只读）。副作用：只读。异常：无。
    English:
        `split_version` is "the identity of the split file". PolyBench has one JSON per task, so the
        whole-file digest is that fold's digest; PolyBench26's `manifest.json` only describes five
        folds and `fold_XX.npz` is what defines a fold; OpenPoly has five JSONs and `spec.sha256`
        is their concatenation. This returns the **fold's own** file. When it cannot be located the
        split's digest is used and None is returned, so the caller knows the record carries the
        whole-split hash only.
        Returns: (split_version, fold digest or None) as `<12 hex>@<repo-relative path>`.
    """
    candidates: list[Path] = []
    ref = REPO_ROOT / spec.ref
    if spec.benchmark == "polybench":
        candidates.append(ref)
    elif spec.benchmark == "polybench26":
        candidates.append(ref.parent / f"{fold}.npz")
    elif spec.benchmark == "openpoly":
        candidates.append(ref / f"rule_replicated_{fold}.json")
    for path in candidates:
        if path.is_file():
            digest = _sha256_of_file(path)
            return f"{digest[:12]}@{_repo_relative(path)}", digest
    return f"{spec.sha256[:12]}@{spec.ref}", None


# ---- 8. 落盘 / Writing ----


def _repo_relative(path: Path) -> str:
    """绝对路径 → 仓库相对的 POSIX 路径（与 `features.repo_relative` 同一口径）。
    Turn an absolute path into a repo-relative POSIX path, same convention as `features.repo_relative`.

    中文：
        用途：`split_version` 里那条路径要在记录里跨机器可读，所以不能带盘符与反斜杠。
              这里自己实现一份而不是调 `features.repo_relative`：`features` 要 pandas / rdkit，
              而本函数属于 A4 也要复用的通用部分（见模块头）。
        输入：path。输出：字符串。数据来源：无。副作用：无。异常：无（不在仓库内的路径原样返回）。
    English: `split_version` carries this path and records must be readable on any machine, so no
    drive letter and no backslashes. Implemented here rather than calling `features`, which needs
    pandas/rdkit, because this helper is part of the generic half A4 also reuses.
    """
    try:
        return path.resolve().relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return path.as_posix()


def write_run_record(record: dict[str, Any], *, directory: Path | None = None) -> Path:
    """把一条记录写成 `<run_id>.json`（UTF-8、LF、两空格缩进），返回路径。
    Write one record to `<run_id>.json` (UTF-8, LF, two-space indent) and return the path.

    中文：
        用途：记录只能由训练脚本落盘（红线 4），所以写入口只有这一处。
              `newline="\\n"` 是必须的：Windows 默认写 CRLF，而仓库规则是 `eol=lf`，
              一次提交就会把整份文件标成"全部改动"（`* text=auto eol=lf` 也会警告）。
              写完立刻回读并校验（`record_problems`），不合格就抛——**不许留下半条记录**。
        输入：record —— `TrainOutcome.record`；directory —— 目标目录（默认 `results/raw_runs/`）。
        输出：写出的路径。
        数据来源：调用方。配置来源：`records.raw_runs`。副作用：建目录、写一个 JSON。
        异常：ArmError —— 记录缺必填字段或文件名与 run_id 不一致。
    English:
        Records are written by the training script only (red line 4), so this is the single writer.
        `newline="\\n"` is mandatory: Windows writes CRLF by default while the repo pins `eol=lf`,
        and one commit would then show the whole file as changed. The record is read back and
        validated immediately; a bad record raises instead of leaving half an entry behind.
    """
    run_id = str(record.get("run_id", ""))
    problems = record_problems(record)
    if problems:
        raise ArmError(f"记录不合格，未写盘：{problems}")
    target_dir = Path(directory) if directory is not None else RAW_RUNS_DIR
    target_dir.mkdir(parents=True, exist_ok=True)
    path = target_dir / f"{run_id}.json"
    payload = json.dumps(record, ensure_ascii=False, indent=2, sort_keys=False)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(payload + "\n")
    return path


def write_predictions(
    outcome: TrainOutcome, *, directory: Path | None = None, run_id: str | None = None
) -> Path:
    """把测试集预测写成 parquet（两列：`row_id` int64、`y_pred` float64）。
    Write the test predictions as a parquet with exactly two columns: row_id int64, y_pred float64.

    中文：
        用途：没有预测文件，D 就算不出 bootstrap 区间，48 条 run 等于白跑（分工_B §0）。
              列名固定为 `row_id` / `y_pred`，行序 = 官方 `test_index` 的顺序**去掉建图失败的行**；
              `row_id` 是任务行位置，D 用它回到调和表取标签，所以**不许写成 0..n-1 的序号**。
        输入：outcome；directory；run_id（默认取记录里的）。
        输出：写出的路径。
        数据来源：`TrainOutcome`。配置来源：`records.predictions`。
        副作用：建目录、写一个小 parquet。异常：ArmError —— 缺 pyarrow 或长度不一致。
    English:
        Without this file D cannot bootstrap and all 48 runs are wasted. Column names are fixed;
        row order follows the official `test_index` with failed-graph rows removed; `row_id` is the
        task row position (never a 0..n-1 ordinal), because D uses it to look the label back up.
        Raises: ArmError when pyarrow is missing or the arrays disagree in length.
    """
    # 直接走 pyarrow 而不用 pandas：polyBERT 环境里**没有 pandas**（`lock/polybert.pip.txt` 只有
    # pyarrow），而 A4 要复用这个写入口。多一层 pandas 就等于让一条臂多一个用不上的依赖。
    # Straight pyarrow rather than pandas: the polyBERT environment ships pyarrow but no pandas,
    # and A4 reuses this writer. Going through pandas would add a dependency to one arm only.
    import pyarrow as pa
    import pyarrow.parquet as pq

    if outcome.row_id.size != outcome.y_pred.size:
        raise ArmError(
            f"预测写不了：row_id {outcome.row_id.size} 条 ≠ y_pred {outcome.y_pred.size} 条"
        )
    identifier = run_id or str(outcome.record["run_id"])
    target_dir = Path(directory) if directory is not None else PREDICTIONS_DIR
    target_dir.mkdir(parents=True, exist_ok=True)
    path = target_dir / f"{identifier}.parquet"
    table = pa.table(
        {
            "row_id": pa.array(np.asarray(outcome.row_id, dtype=np.int64)),
            "y_pred": pa.array(np.asarray(outcome.y_pred, dtype=np.float64)),
        }
    )
    pq.write_table(table, path)
    return path


def append_gpu_hours(rows: Sequence[dict[str, Any]], *, path: Path | None = None) -> Path:
    """把若干条 run 的墙钟汇总进 `gpu_hours.csv`（按 run_id 覆盖同名的旧行）。
    Merge several runs' wall clocks into `gpu_hours.csv`, replacing rows with the same run_id.

    中文：
        用途：分工_B §2 B-3 的产出之一。**按 run_id 覆盖**而不是追加：重跑一条 run 时不能同时留下
              两行（D 会把它算成两次 GPU 预算）。`gpu_hours` 一律按 `wall_clock_s / 3600` 算，
              CPU 上也记（值为 0 的 GPU 时间没有意义，但墙钟有），`device` 列标明是哪种。
        输入：rows —— 至少含 `run_id` 与 `wall_clock_s` 的字典序列；path —— 目标文件。
        输出：写出的路径。
        数据来源：调用方。配置来源：`records.gpu_hours`。
        副作用：读旧文件、写新文件（UTF-8 / LF）。异常：ArmError —— 缺列。
    English:
        One of B-3's artefacts. Rows are replaced **by run_id**, never appended: a re-run must not
        leave two rows or D counts the budget twice. `gpu_hours` is always `wall_clock_s / 3600`,
        recorded on CPU too (zero GPU time is meaningless but wall clock is not), with `device`
        saying which it was.
    """
    import csv

    target = Path(path) if path is not None else GPU_HOURS_PATH
    target.parent.mkdir(parents=True, exist_ok=True)
    existing: dict[str, dict[str, Any]] = {}
    if target.is_file():
        with target.open("r", encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle):
                existing[str(row.get("run_id", ""))] = dict(row)
    for row in rows:
        missing = [key for key in ("run_id", "wall_clock_s") if key not in row]
        if missing:
            raise ArmError(f"gpu_hours 行缺列 {missing}：{row}")
        payload = {key: row.get(key, "") for key in GPU_HOURS_COLUMNS}
        seconds = float(row["wall_clock_s"])
        payload["gpu_hours"] = f"{seconds / 3600.0:.6f}"
        existing[str(row["run_id"])] = payload
    ordered = sorted(existing.values(), key=lambda item: str(item.get("run_id", "")))
    with target.open("w", encoding="utf-8", newline="\n") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(GPU_HOURS_COLUMNS), lineterminator="\n")
        writer.writeheader()
        for row in ordered:
            writer.writerow(row)
    return target


def write_n_index(config: ArmConfig) -> dict[str, Any]:
    """把带 n 档的网格刷成两份索引：一条 run 一行 + 一个 (task, fold, 档位) 一行。
    Refresh two indexes for an n-level grid: one row per run, and one per (task, fold, level).

    中文：
        用途：B-5 的 225 条如果只能靠 `ls | wc -l` 对账，D 算 κ 时要开 225 个 JSON，C 想核对
              "两边用的是不是同一套训练行"也无从下手。所以这里把**磁盘上现有的记录**汇总成：
              ① `openpoly_gpu_index.csv`——分工_B §2 B-5 点名的产物，六列
                 （`run_id, property, n_level, fold, seed, wall_clock_s`）都在，另加 R²/RMSE/轮数
                 等，让 D 不必再开文件；
              ② `openpoly_n_index.csv`——每 (task, fold, 档位) 一行，写**这一档到底用了哪些行**
                 （`subset_sha256`）、训了多少（`n_train`）、池子多大（`n_train_pool`）、有没有
                 被池子卡住（`n_level_saturated`），以及训练行的标签分布与值域箱覆盖。
              两份都按**全量记录的现状**生成，不是"这次写了哪些"——`resume` 跳过已有记录时索引
              依然完整。**只收 `status=ok` 且 `config_hash` 与本配置一致的记录**：换过配置的旧
              记录不在同一协议里，混进来会让 D 以为是同一批数。
        输入：config。
        输出：`{path, n_index_path, n_runs, n_tasks, n_levels, n_folds, n_seeds, n_records_seen}`。
        数据来源：`results/raw_runs/*.json`、`splits/openpoly/**`、`data/processed/features/**.npz`。
        配置来源：`records.index` / `records.n_index`（缺省用下面的常量）。
        副作用：写两个 CSV（UTF-8 / LF）。
        异常：ArmError —— 配置里没有 `n_levels`，或 `n_index` 与 `index` 没配。
    English:
        With only `ls | wc -l` to reconcile 225 runs, D would have to open 225 JSON files to get κ,
        and C could not check that both arms used the same training rows. So this folds the records
        already on disk into (1) the artefact 分工_B names - the six required columns are all there,
        plus R²/RMSE/epochs so D need not open files - and (2) one row per (task, fold, level) saying
        which rows that level used (`subset_sha256`), how many, the pool size, whether the pool
        capped it, plus the label distribution and value-bin coverage of its training rows. Both are
        built from **every record on disk** so a resumed run still leaves a complete index, and both
        take only `status=ok` records whose `config_hash` matches this config - an older record from
        a different config is not the same protocol.
        Returns: a dict describing what was written. Raises ArmError when the config has no
        `n_levels` or the two index paths are not configured.
        Side effects: writes two CSVs (UTF-8 / LF).
    """
    import csv

    if not config.raw.get("n_levels"):
        raise ArmError(
            "这份配置没有 `n_levels`：n 索引只对 B-5 的 OpenPoly 规模扫描有意义。"
            "要生成 48 条的索引请用 C 的汇总脚本"
        )
    records_cfg = config.section("records")
    index_path = REPO_ROOT / str(
        records_cfg.get("index", "results/aggregate/openpoly_gpu_index.csv")
    )
    n_index_path = REPO_ROOT / str(
        records_cfg.get("n_index", "results/aggregate/openpoly_n_index.csv")
    )
    wanted_hash = config_hash(config)
    arm_name = str(config.raw.get("arm") or ARM)
    subset_seed = int((config.raw.get("subset") or {}).get("seed", 0))

    runs: list[dict[str, Any]] = []
    seen = 0
    for path in sorted(RAW_RUNS_DIR.glob("*.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:                        # pragma: no cover - 只在半截文件时触发
            continue
        seen += 1
        if payload.get("arm") != arm_name or payload.get("n_level") is None:
            continue
        if payload.get("status") != "ok" or payload.get("config_hash") != wanted_hash:
            continue
        runs.append(payload)
    if not runs:
        raise ArmError(
            f"`{RAW_RUNS_DIR.name}/` 里没有一条能用（arm={arm_name}、有 n 档、status=ok、"
            f"config_hash={wanted_hash[:16]}…）：先跑网格再来刷索引"
        )

    def fmt(value: Any, digits: int = 6) -> str:
        return "" if value is None else f"{float(value):.{digits}f}"

    index_path.parent.mkdir(parents=True, exist_ok=True)
    with index_path.open("w", encoding="utf-8", newline="\n") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(N_INDEX_COLUMNS), lineterminator="\n")
        writer.writeheader()
        for item in sorted(runs, key=lambda row: str(row["run_id"])):
            metrics = item.get("metrics") or {}
            writer.writerow(
                {
                    "run_id": item["run_id"],
                    "property": item["task"],
                    "n_level": item["n_level"],
                    "n_train": item.get("n_train", ""),
                    "n_train_pool": item.get("n_train_pool", ""),
                    "n_level_saturated": str(bool(item.get("n_level_saturated"))).lower(),
                    "fold": item.get("split_fold", ""),
                    "fold_index": str(item.get("split_fold", "")).replace("seed", ""),
                    "seed": item.get("seed", ""),
                    "wall_clock_s": fmt(item.get("wall_clock_s"), 3),
                    "gpu_hours": fmt(float(item.get("wall_clock_s") or 0.0) / 3600.0),
                    "r2": fmt(metrics.get("r2")),
                    "rmse": fmt(metrics.get("rmse")),
                    "mae": fmt(metrics.get("mae")),
                    "n_test": item.get("n_test", ""),
                    "n_val": item.get("n_val", ""),
                    "epochs_run": item.get("epochs_run", ""),
                    "early_stopped": str(bool(item.get("early_stopped"))).lower(),
                    "best_epoch": item.get("best_epoch", ""),
                    "device": item.get("device", ""),
                    "subset_sha256": item.get("n_subset_sha256", ""),
                    "config_hash": item.get("config_hash", ""),
                    "split_version": item.get("split_version", ""),
                }
            )

    # 子集索引：按 (task, fold, 档位) 去重（三个模型种子用的是同一批训练行，所以三行的
    # `subset_sha256` 必须完全一样——这一条同时是一个自检：如果种子之间哈希不同，说明子集
    # 依赖了模型种子，那 n 档就不只是"训练集大小"了）。
    # One row per (task, fold, level): the three model seeds share the same training rows, so their
    # `subset_sha256` must be identical - that equality doubles as a self-check that the subset does
    # not depend on the model seed, which would make the level something other than a size.
    subsets: dict[tuple[str, str, str], dict[str, Any]] = {}
    for item in runs:
        sig = (str(item["task"]), str(item.get("split_fold", "")), str(item["n_level"]))
        if sig in subsets:
            if subsets[sig]["n_subset_sha256"] != item.get("n_subset_sha256"):
                LOGGER.warning(
                    "%s/%s/n=%s：三个种子的训练行哈希不一致（%s vs %s），档位不只是大小",
                    sig[0],
                    sig[1],
                    sig[2],
                    subsets[sig]["n_subset_sha256"],
                    item.get("n_subset_sha256"),
                )
            continue
        subsets[sig] = item

    with n_index_path.open("w", encoding="utf-8", newline="\n") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(SUBSET_INDEX_COLUMNS), lineterminator="\n")
        writer.writeheader()
        for sig in sorted(subsets, key=lambda key: (key[0], key[2], key[1])):
            item = subsets[sig]
            stats = _subset_label_stats(
                str(item["task"]), str(item.get("split_fold", "")), item, subset_seed
            )
            writer.writerow(
                {
                    "task": sig[0],
                    "fold": sig[1],
                    "n_level": sig[2],
                    "n_train": item.get("n_train", ""),
                    "n_train_pool": item.get("n_train_pool", ""),
                    "n_level_saturated": str(bool(item.get("n_level_saturated"))).lower(),
                    "n_train_requested": item.get("n_train_requested", ""),
                    "subset_seed": subset_seed,
                    "subset_sha256": item.get("n_subset_sha256", ""),
                    **stats,
                }
            )

    tasks = sorted({str(item["task"]) for item in runs})
    levels = sorted({str(item["n_level"]) for item in runs}, key=lambda text: (text != "all", text))
    folds = sorted({str(item.get("split_fold", "")) for item in runs})
    seeds = sorted({int(item["seed"]) for item in runs})
    LOGGER.info(
        "索引已刷新：%d 条 run、%d 个子集（%s）",
        len(runs),
        len(subsets),
        ", ".join(tasks),
    )
    return {
        "path": index_path,
        "n_index_path": n_index_path,
        "n_runs": len(runs),
        "n_records_seen": seen,
        "n_tasks": len(tasks),
        "n_levels": len(levels),
        "n_folds": len(folds),
        "n_seeds": len(seeds),
    }


def _subset_label_stats(
    task: str, fold: str, record: dict[str, Any], subset_seed: int
) -> dict[str, str]:
    """算一档训练行的标签统计与值域箱覆盖；算不出来就留空，不猜。
    Label statistics and value-bin coverage for one level's training rows, blank when unknowable.

    中文：
        用途：H4 要判断"低 n 端 R² 掉下去，是因为样本少还是因为值域没覆盖"。后者只能靠数：
              这一档的训练行落进了 10 个等宽箱里的几个。**只做事后诊断，不参与训练**——
              算不出来（缺 npz / 缺折文件 / 两边 y 对不上）就留空并记一条 warning。
              `n_bins_covered` 与 `bin_counts` 用**该折自己的** `bin_edges`（五个种子各一套），
              所以同一任务不同折的箱不完全可比；这是数据事实，不加注释会被读错。
        输入：task；fold（`seed0`…）；record（提供档位与 `n_train`）；subset_seed（与训练同一个）。
        输出：六个字符串字段。数据来源：`splits/openpoly/`、`data/processed/features/`。
        副作用：读盘。异常：无（失败即留空）。
    English:
        H4 has to separate "R² falls because there are fewer rows" from "R² falls because the value
        range is not covered"; the second is counted, not asserted: into how many of the ten
        equal-width bins this level's training rows fall. Diagnostics only - never feeds training -
        and blank (with a warning) when the arrays are missing or disagree. Bins use **that fold's
        own** `bin_edges`, which differ per seed, so the same task's folds are not strictly
        comparable; that is a fact about the data, not an oversight.
    """
    blank = {
        "y_mean": "",
        "y_std": "",
        "y_min": "",
        "y_max": "",
        "n_bins_covered": "",
        "bin_counts": "",
    }
    try:
        with np.load(REPO_ROOT / "data" / "processed" / "features" / f"{task}.npz") as pack:
            y = np.asarray(pack["y"], dtype=np.float64)
        split_path = REPO_ROOT / "splits" / "openpoly" / task / f"rule_replicated_{fold}.json"
        payload = json.loads(split_path.read_text(encoding="utf-8"))
        edges = np.asarray(payload["bin_edges"], dtype=np.float64)
        index = np.concatenate(
            [
                np.asarray(payload["train_index"], dtype=np.int64),
                np.asarray(payload["val_index"], dtype=np.int64),
                np.asarray(payload["test_index"], dtype=np.int64),
            ]
        )
        # 自检：三份索引合起来必须正好覆盖 0..n-1，否则 y 的行序与折文件不是一个坐标系，
        # 算出来的箱覆盖就是编的。
        # Self-check: the three index arrays must cover 0..n-1 exactly, otherwise the labels and the
        # split are not in the same coordinate system and any bin count would be fiction.
        if index.size != y.size or not np.array_equal(np.sort(index), np.arange(y.size)):
            raise ArmError("三份索引没覆盖全部行")
        rows = np.asarray(payload["train_index"], dtype=np.int64)
        picked, _info = n_level_rows(rows, str(record["n_level"]), seed=subset_seed)
        # 训练时先剔除建图失败的行再截断，这里做不到（要点图才知道），所以按行数核对：
        # 对不上就返回空，而不是拿一个"差不多的"集合去算覆盖数。
        # The trainer drops parse failures before trimming; we cannot know those without building
        # graphs, so compare counts and return blank rather than compute coverage on a near-miss set.
        if picked.size != int(record.get("n_train") or -1):
            raise ArmError(f"重放得到 {picked.size} 行，记录说 {record.get('n_train')} 行")
        values = y[picked]
        n_bins = max(len(edges) - 1, 1)
        counts = np.bincount(
            np.clip(np.digitize(values, edges[1:-1]), 0, n_bins - 1), minlength=n_bins
        )
        return {
            "y_mean": f"{float(np.mean(values)):.6f}",
            "y_std": f"{float(np.std(values, ddof=1)) if values.size > 1 else 0.0:.6f}",
            "y_min": f"{float(np.min(values)):.6f}",
            "y_max": f"{float(np.max(values)):.6f}",
            "n_bins_covered": str(int((counts > 0).sum())),
            "bin_counts": " ".join(str(int(c)) for c in counts),
        }
    except (OSError, KeyError, ArmError, ValueError) as error:
        LOGGER.warning(
            "%s/%s/n=%s 的标签统计算不出来（%s），索引里留空",
            task,
            fold,
            record.get("n_level"),
            error,
        )
        return blank


# ---- 9. 记录校验与整轮编排 / Record validation and the whole grid ----


def record_problems(record: dict[str, Any]) -> list[str]:
    """按 `results/raw_runs/_schema.json` 的核心约束检查一条记录（不引入 jsonschema 依赖）。
    Check one record against the core constraints of `_schema.json` (no jsonschema dependency).

    中文：
        用途：schema 是冻结物，但训练脚本不该为了校验它去装一个包（GNN 环境里没有 jsonschema，
              也不该为这个加）。这里实现的是**会真的被违反**的那几条：必填字段、run_id 形状、
              `sha256:` 前缀、`split_version` 的 `@splits/` 段、`ok` 状态下的附加必填、
              `cuda` 必须有 `gpu` 块且块里两个确定性开关都为 true、预测路径与 run_id 一致、
              指标键齐全。
        输入：record。输出：问题字符串列表；空列表表示合格。
        数据来源：调用方。配置来源：`_schema.json`（本函数是它的手工落版）。副作用：无。
        异常：无（收集问题而非抛出）。
    English:
        The schema is frozen, but the trainer should not install a package to check it. This
        implements the constraints that actually get violated: required fields, the run_id shape,
        the `sha256:` prefix, the `@splits/` segment, the extra requirements for `status=ok`, the
        `gpu` block on cuda, a predictions path that matches the run_id, and the metric keys.
        Returns: problem strings; empty means valid.
    """
    problems: list[str] = []
    for key in RECORD_REQUIRED:
        if key not in record:
            problems.append(f"缺必填字段 `{key}`")
    if problems:
        return problems
    run_id = str(record["run_id"])
    arm_name = str(record.get("arm", ""))
    if arm_name not in KNOWN_ARMS:
        problems.append(f"arm={arm_name!r} 不在 schema 的枚举 {list(KNOWN_ARMS)} 里")
        return problems
    expected = run_id_for(
        str(record["task"]),
        str(record.get("split_fold", "")),
        int(record["seed"]),
        arm=arm_name,
        n_level=record.get("n_level"),
    )
    if run_id != expected:
        problems.append(f"run_id {run_id!r} 与 task/fold/seed 拼出来的 {expected!r} 不一致")
    if record.get("status") not in ("ok", "blocked_no_gpu", "failed"):
        problems.append(f"status={record.get('status')!r} 不在 ok/blocked_no_gpu/failed 里")
    if not str(record.get("config_hash", "")).startswith("sha256:"):
        problems.append("config_hash 必须以 sha256: 开头")
    elif len(str(record["config_hash"])) != len("sha256:") + 64:
        problems.append("config_hash 的十六进制不是 64 位")
    if "@splits/" not in str(record.get("split_version", "")):
        problems.append("split_version 必须是 `<12 位>@splits/...`")
    elif len(str(record["split_version"]).split("@", 1)[0]) != 12:
        problems.append("split_version 的哈希前缀不是 12 位")
    if not str(record.get("env_fingerprint", "")).startswith("results/env_fingerprints/env_"):
        problems.append("env_fingerprint 必须是 results/env_fingerprints/env_{core,gnn,polybert}.json")
    if record.get("status") == "ok":
        for key in RECORD_REQUIRED_OK:
            if key not in record:
                problems.append(f"status=ok 缺字段 `{key}`")
        metric_keys = set((record.get("metrics") or {}).keys())
        if not {"r2", "mae", "rmse"} <= metric_keys:
            problems.append(f"metrics 缺键，现在只有 {sorted(metric_keys)}")
        if record.get("device") == "cuda":
            gpu = record.get("gpu")
            if not isinstance(gpu, dict):
                problems.append("device=cuda 必须有 `gpu` 块（型号/驱动/torch.version.cuda）")
            elif gpu.get("cudnn_deterministic") is not True:
                problems.append("gpu.cudnn_deterministic 必须为 true（验收④要可复现，不是'大致一致'）")
            elif gpu.get("deterministic_algorithms") is not True:
                problems.append(
                    "gpu.deterministic_algorithms 必须为 true：只钉 cudnn 不够，"
                    "scatter_reduce 的原子加会让同 seed 两次相差几十 K"
                )
        expected_pred = f"results/aggregate/predictions/{run_id}.parquet"
        if str(record.get("predictions")) != expected_pred:
            problems.append(f"predictions 应为 {expected_pred}，现在是 {record.get('predictions')!r}")
        # n 档的三条自洽关系（B-5）。这三条**都会真的被违反**：档位名写成名义值、实际行数被池子
        # 卡住却标 unsaturated、或者反过来标了 saturated 却没顶到池子。任何一条错，D 画出来的
        # 学习曲线就会把"没有的数据"算成一个点（D-35）。
        # Three self-consistency rules for an n-level (B-5), each of which does get violated: a level
        # named by its nominal value, a capped level not marked saturated, or the reverse. Any of
        # them puts a point on the curve that the data does not support.
        if record.get("n_level") is not None:
            pool = record.get("n_train_pool")
            requested = record.get("n_train_requested")
            used = record.get("n_train")
            saturated = record.get("n_level_saturated")
            if not isinstance(pool, int) or not isinstance(requested, int) or not isinstance(used, int):
                problems.append("有 n 档就必须记 n_train_pool / n_train_requested / n_train（三个整数）")
            else:
                if used != min(requested, pool):
                    problems.append(
                        f"n_train={used} 应等于 min(n_train_requested {requested}, n_train_pool {pool})"
                    )
                if bool(saturated) != bool(requested >= pool and str(record["n_level"]) != "all"):
                    problems.append(
                        f"n_level_saturated={saturated!r} 与 requested {requested} / pool {pool} "
                        f"/ level {record['n_level']!r} 不自洽"
                    )
            if not str(record.get("n_subset_sha256", "")).startswith("sha256:"):
                problems.append("有 n 档就要记 n_subset_sha256（哪些训练行必须可核对）")
    return problems


def run_grid(
    config: ArmConfig,
    *,
    tasks: Sequence[str] | None = None,
    folds: Sequence[str] | None = None,
    seeds: Sequence[int] | None = None,
    n_levels: Sequence[str] | None = None,
    extra_folds: Sequence[str] | None = None,
    limit: int | None = None,
    device: str | None = None,
    max_epochs: int | None = None,
    patience: int | None = None,
    resume: bool = True,
    out_root: Path | None = None,
    builder: Any = None,
    trainer: Any = None,
) -> list[dict[str, Any]]:
    """按网格逐条训练、落记录与预测、汇总墙钟；返回记录列表。
    Train the grid run by run, writing records, predictions and wall clocks; return the records.

    中文：
        用途：`scripts/05_run_gpu.py` 的全部实际工作都在这里（脚本只做参数解析）。
              两点设计：
              ① **按任务分组**：同一个任务的 15 条 run 共享一份 `TaskData`（图只建一次）；
              ② **`resume` 只跳过 `ok` 的记录**：`failed` / `blocked_no_gpu` 的占位不会被当成"跑过了"。
                  跳过的判据是"文件名在、且 JSON 里 status=ok 且 config_hash 与当前一致"——
                  配置换了就必须重跑，否则记录会声称自己用的是旧配置。
        输入：config；tasks/folds/seeds —— 子集；extra_folds —— 配置外的折（只对 openpoly，
              B-5 的 5–9 敏感性分析）；limit —— 冒烟；device；max_epochs/patience ——
              覆盖；resume —— 跳过已完成的；out_root —— 测试用输出根（None 走仓库默认）；
              builder / trainer —— **注入点**：A4 传 `polybert.build_task_data` 与
              `polybert.train_one`，于是"按任务分组 + resume 语义 + 写记录/预测/墙钟"这一整套
              编排只有一份实现，两条臂不会在"哪些算跑完了"这种细节上分叉。
        输出：本次写出的记录列表（按网格顺序）。
        数据来源：`TaskData`。配置来源：`config`。
        副作用：训练、写记录/预测/`gpu_hours.csv`。异常：ArmError —— 见各步。
    English:
        All the real work; the script only parses arguments. Two design points: (1) runs are grouped
        by task so one TaskData (and therefore one graph build) serves all 15 runs of that task;
        (2) `resume` skips only `ok` records **whose config_hash matches**, so a failed placeholder
        is retried and a config change forces a re-run instead of a record claiming an old config.
        `builder`/`trainer` are injection points: the polyBERT arm passes its own two functions, so
        the grouping / resume / writing orchestration exists once, not twice.
        Returns: the records written this time, in grid order.
    """
    builder = builder or build_task_data
    trainer = trainer or train_one
    keys = enumerate_runs(
        config, tasks=tasks, folds=folds, seeds=seeds, n_levels=n_levels, extra_folds=extra_folds
    )
    raw_dir = (out_root / "raw_runs") if out_root else RAW_RUNS_DIR
    pred_dir = (out_root / "predictions") if out_root else PREDICTIONS_DIR
    hours_path = (out_root / "gpu_hours.csv") if out_root else GPU_HOURS_PATH
    raw_dir.mkdir(parents=True, exist_ok=True)
    pred_dir.mkdir(parents=True, exist_ok=True)

    written: list[dict[str, Any]] = []
    hours_rows: list[dict[str, Any]] = []
    by_task: dict[str, list[RunKey]] = {}
    for key in keys:
        by_task.setdefault(key.task, []).append(key)

    for task, task_keys in by_task.items():
        pending = []
        for key in task_keys:
            path = raw_dir / f"{key.run_id()}.json"
            if resume and path.is_file():
                try:
                    existing = json.loads(path.read_text(encoding="utf-8"))
                except json.JSONDecodeError:
                    existing = {}
                if (
                    existing.get("status") == "ok"
                    and existing.get("config_hash") == config_hash(config)
                ):
                    LOGGER.info("跳过已完成的 %s", key.run_id())
                    continue
            pending.append(key)
        if not pending:
            continue
        data = builder(task, limit=limit)
        for key in pending:
            LOGGER.info("开始 %s", key.run_id())
            outcome = trainer(
                data,
                config,
                fold=key.fold,
                seed=key.seed,
                device=device,
                max_epochs=max_epochs,
                patience=patience,
                n_level=key.n_level,
            )
            write_predictions(outcome, directory=pred_dir)
            path = write_run_record(outcome.record, directory=raw_dir)
            written.append(outcome.record)
            hours_rows.append(outcome.record)
            LOGGER.info(
                "%s → r2=%.4f rmse=%.4f %.1f 秒（%s）",
                path.name,
                outcome.record["metrics"]["r2"],
                outcome.record["metrics"]["rmse"],
                outcome.record["wall_clock_s"],
                outcome.record["device"],
            )
            if hours_rows:
                append_gpu_hours(hours_rows, path=hours_path)
                hours_rows = []
    return written
