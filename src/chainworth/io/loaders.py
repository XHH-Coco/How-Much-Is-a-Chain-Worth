"""按任务取数：把"调和表 + 官方划分 + 结构级特征"读成一个可对齐的三元组。
Per-task loading: read the harmonised table, the official split and the structure-level
features into one alignable triple.

中文：
    用途：B-1 的取数层，也是"行序由划分文件决定"这条红线的**唯一实现处**。
          三件事：① 给每个任务一个 `row_id`（= A-3 调和表的行位置，按 `source_row` 升序，
          也就是 `splits/polybench/*.json` 的 `row_order` 所指的那个口径）；
          ② 把三种基准的划分文件（PolyBench JSON / PolyBench26 NPZ / OpenPoly JSON）
          解析成同一个 `SplitSpec`，下游不再各自 `json.load`；
          ③ 读 A-6 的结构级特征表，**按键 join**，绝不按 DataFrame 行序。
    依赖：`data/interim/harmonised/*.parquet`（A-3）、`configs/tasks/*.yaml`（A-4）、
          `splits/**`（A-5，只读）、`data/processed/{structures,descriptors}`（A-6，不入库）。
    配置：`configs/tasks/<task>.yaml` 的 `benchmark` / `data_ref` / `target_column` /
          `split_ref` / `split_protocol` 五个键。
    对外：`load_task_config()`、`available_tasks()`、`load_rows()`、`load_split()`、
          `load_descriptor_table()`、`load_structure_table_index()`、`SplitSpec`、`FoldSpec`。
    被谁用：`src/chainworth/chem/features.py`、`scripts/03b_assemble.py`、
            `tests/test_task_assembly.py`；C 的 A2/A5 臂与 B 的 A3/A4 臂都从这里取数。

English:
    Purpose: B-1's data layer and the **single implementation** of the "row order comes from the
    split file" red line. Three jobs: (1) give every task a `row_id` - the row position of A-3's
    harmonised table sorted by `source_row`, which is exactly what
    `splits/polybench/*.json`'s `row_order` refers to; (2) parse all three benchmarks' split
    files (PolyBench JSON / PolyBench26 NPZ / OpenPoly JSON) into one `SplitSpec`, so nothing
    downstream calls `json.load` on a split again; (3) read A-6's structure-level feature table
    and **join on the key**, never on DataFrame row order.
    Depends on: data/interim/harmonised/*.parquet (A-3), configs/tasks/*.yaml (A-4),
    splits/** (A-5, read-only), data/processed/{structures,descriptors} (A-6, not committed).
    Config: `benchmark` / `data_ref` / `target_column` / `split_ref` / `split_protocol` of
    configs/tasks/<task>.yaml.
    Public API: `load_task_config()`, `available_tasks()`, `load_rows()`, `load_split()`,
    `load_descriptor_table()`, `load_structure_table_index()`, `SplitSpec`, `FoldSpec`.
    Callers: src/chainworth/chem/features.py, scripts/03b_assemble.py,
    tests/test_task_assembly.py, C's A2/A5 arms and B's A3/A4 arms.

为什么 row_id 是这个定义 ／ Why row_id is defined this way
---------------------------------------------------------
A-3 的调和表是按 `source_row` 升序写的，A-5 的 PolyBench 划分里 `row_order` 写的就是
"`<task>_harmonised.parquet#row_position（按 source_row 升序）`"，`test_index` 里的整数是**位置**
而不是 `source_row`。所以两个口径必须一起看：位置 = 行号 = `row_id`，而 `source_row` 只是原始
CSV 的行号，是**另一个数**。把两者混用不会报错，只会让测试集悄悄错位——这是 Proposal §7.5 的
第 (i) 类失败模式。
A-3 writes the harmonised table sorted by `source_row`, and A-5's PolyBench splits say
`row_order` is "`<task>_harmonised.parquet#row_position` (ascending `source_row`)"; the integers in
`test_index` are **positions**, not `source_row` values. The two must be read together: position =
row number = `row_id`, while `source_row` is the original CSV line and is a **different number**.
Mixing them raises nothing and silently misplaces the test set - failure mode (i) of Proposal §7.5.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from chainworth.io.tasks import load_all_task_configs

# `pandas` 刻意不在模块层导入：本模块有一半函数只读 JSON/YAML（`load_task_config`、
# `load_split`、`SplitSpec`），而 A4 臂跑的 `DSC6020_Project_Polybert` 环境**没有 pandas**。
# 需要它的三个函数各自在函数体内导入，注解因为 `from __future__ import annotations` 不求值。
# `pandas` is deliberately not imported at module level: half of this module only reads JSON/YAML,
# and the polyBERT environment (where the A4 arm runs) has no pandas. The three functions that
# need it import it locally; annotations are strings thanks to `from __future__ import annotations`.

REPO_ROOT = Path(__file__).resolve().parents[3]     # src/chainworth/io/loaders.py -> 仓库根
CONFIGS_DIR = REPO_ROOT / "configs" / "tasks"
HARMONISED_DIR = REPO_ROOT / "data" / "interim" / "harmonised"
PROCESSED_DIR = REPO_ROOT / "data" / "processed"
STRUCTURES_PATH = PROCESSED_DIR / "structures" / "psmiles.parquet"
DESCRIPTORS_DIR = PROCESSED_DIR / "descriptors"
FEATURES_DIR = PROCESSED_DIR / "features"
SPLITS_ROOT = REPO_ROOT / "splits"
MANIFEST_DIR = REPO_ROOT / "data" / "manifests"

# 任务行表的列。`row_id` 是调和表的位置，**不是** `source_row`。
# The task row table's columns. `row_id` is the harmonised row position, **not** `source_row`.
ROW_COLUMNS = ("row_id", "psmiles_canonical", "label")

# 一次装配要读的调和表列：行号、结构键、标签。`smiles_raw` 只在需要重新规范化时才读。
# The harmonised columns one assembly needs: row number, structure key, label. `smiles_raw` is
# read only when a re-canonicalisation is required.
HARMONISED_COLUMNS = ("source_row", "smiles_canonical", "label")

# OpenPoly 的 `eval_unit` 用 `split seeds 0-4` 当五折（D-24）；这里写死是因为它不是实验参数，
# 而是"哪几份划分算折"的口径。
# OpenPoly's `eval_unit` treats split seeds 0-4 as its five folds (D-24); hard-coded because that
# is not an experiment parameter but the definition of which split files count as folds.
OPENPOLY_FOLD_SEEDS = (0, 1, 2, 3, 4)


class AssemblyError(RuntimeError):
    """装配无法进行时抛这一个；消息里必须写清"缺什么、谁该补"。
    The single exception raised when assembly cannot proceed; the message must say what is
    missing and who supplies it.

    中文：
        用途：装配失败的原因只有三类——缺上游产物、任务类型不支持、划分与行数不配套。三类都用
              这一个异常，调用方只需捕获一种；但**消息必须能直接指导下一步**，否则跑到一半停下
              的人只会看到"KeyError"。
    English:
        Purpose: assembly fails for exactly three reasons - a missing upstream artefact, an
        unsupported task kind, or a split that does not match the row count. One exception type
        keeps the caller simple, but the message must be actionable on its own.
    """


@dataclass(frozen=True)
class FoldSpec:
    """一折的索引（训练 / 验证 / 测试），索引值是**任务行位置**。
    One fold's indices (train / val / test); the values are **task row positions**.

    中文：
        用途：把三种基准的折结构压成一种。PolyBench 没有验证集（官方只发 70/30），
              `val_index` 因此是空数组而不是 None——空数组在下游是"合法但为空"，None 是"没这一项"，
              两者含义不同，混用会让"没有验证集"和"漏读了验证集"看起来一样。
        字段：`name`（折名，如 `official_70_30` / `fold_00` / `seed0`）、`train_index`、
              `val_index`、`test_index`、`train_pool`（PolyBench26 的档位池；其余基准为 None）。
        数据来源：`splits/**`。配置来源：无。副作用：无。

    English:
        Purpose: collapse three benchmarks' fold shapes into one. PolyBench has no validation set
        (the platform ships 70/30 only), so `val_index` is an empty array and never None - an
        empty array is "legitimately empty", None would be "this field does not exist", and
        conflating them makes "no val set" look like "the val set was not read".
        Fields: `name`, `train_index`, `val_index`, `test_index`, `train_pool` (PolyBench26's
        scaling pool; None elsewhere). Data source: splits/**. Config: none. Side effects: none.
    """

    name: str
    train_index: np.ndarray
    val_index: np.ndarray
    test_index: np.ndarray
    train_pool: np.ndarray | None = None


@dataclass(frozen=True)
class SplitSpec:
    """一个任务的全部折，外加它来自哪个文件（哈希进清单，供 D 复核）。
    Every fold of one task, plus which file it came from (the hash goes into the manifest).

    中文：
        用途：`split_version` 是 run 记录里必填的字段（分工_B §0）。它的值必须是**文件字节的
              sha256**，不是"v1"这种手写版本号——手写版本号在文件被替换时不会变。
        字段：`task_id`、`benchmark`、`protocol`、`ref`（仓库相对路径）、`sha256`、`n_total`、
              `row_order`（口径说明，原样抄自划分文件）、`folds`（`FoldSpec` 元组）。
        数据来源：`splits/`。配置来源：`configs/tasks/<task>.yaml` 的 `split_ref`。
        副作用：无。异常：AssemblyError —— 划分文件缺失或格式不认识。

    English:
        Purpose: `split_version` is a required run-record field (division B, section 0), and its
        value must be the **sha256 of the file bytes**, not a hand-written "v1" - a hand-written
        version does not change when the file is swapped.
        Fields: task_id, benchmark, protocol, ref, sha256, n_total, row_order (copied verbatim
        from the split file) and folds.
        Data source: splits/. Config: `split_ref`. Side effects: none.
        Raises: AssemblyError when the split file is missing or its shape is unrecognised.
    """

    task_id: str
    benchmark: str
    protocol: str
    ref: str
    sha256: str
    n_total: int
    row_order: str
    folds: tuple[FoldSpec, ...]

    @property
    def fold_names(self) -> tuple[str, ...]:
        """折名的元组，用于日志与清单（不返回对象，免得调用方去改折）。
        The fold names, for logs and manifests (objects stay inside the spec).
        """
        return tuple(fold.name for fold in self.folds)


# ---- 1. 配置与任务清单 / Config and task inventory ----


def load_task_config(task_id: str) -> dict[str, Any]:
    """读一份 `configs/tasks/<task>.yaml`（复用 A-4 的校验器，不自己 yaml.safe_load）。
    Read one task config through A-4's validator instead of calling yaml.safe_load here.

    中文：
        用途：装配要用的四个键（`benchmark` / `data_ref` / `target_column` / `split_ref`）
              都在这份配置里。**复用**校验器是刻意的：自己读一遍就等于绕过 A-4 的字段检查，
              一个拼错的 `split_ref` 会变成"文件不存在"这种查不出根因的错。
        输入：task_id —— 任务 ID（`configs/tasks/<task_id>.yaml`）。
        输出：配置 dict（键见 `chainworth.io.tasks.TASK_KEYS_REQUIRED`）。
        数据来源：`configs/tasks/`。配置来源：无。副作用：只读。
        异常：TaskConfigError —— 文件不合法或不存在。

    English:
        Purpose: the four keys assembly needs all live in this file. Reusing the validator is
        deliberate: reading it by hand would bypass A-4's field checks, and a typo in `split_ref`
        would surface as a bare "file not found" with no traceable cause.
        Args: task_id. Returns: the config dict. Data source: configs/tasks/.
        Side effects: read-only. Raises: TaskConfigError on an invalid or missing file.
    """
    configs = load_all_task_configs(CONFIGS_DIR)
    if task_id not in configs:
        raise AssemblyError(
            f"不认识的任务 {task_id!r}；configs/tasks 下现有：{sorted(configs)}"
        )
    return configs[task_id]


def available_tasks() -> list[str]:
    """列出**能装配**的任务：有调和表的那些（当前是 10 个 PolyBench 任务）。
    List the tasks that can actually be assembled: those with a harmonised table.

    中文：
        用途："能装配"的判据是上游产物在不在，不是配置文件在不在。PolyBench26 与 OpenPoly 的
              配置齐全，但 A-3 没有为它们产出调和表（那两类任务的标签在 `data/raw/` 的原始 CSV
              里，结构也不在 A-6 的结构表里），所以它们暂时装不出来——这一条必须由本函数说出
              真话，而不是让 `load_rows()` 在半个小时后抛 FileNotFoundError。
        输入：无。输出：任务 ID 列表（字典序）。
        数据来源：`data/interim/harmonised/*.parquet`。配置来源：无。副作用：只读。异常：无。

    English:
        Purpose: "can be assembled" is decided by whether the upstream artefact exists, not by
        whether a config exists. PolyBench26 and OpenPoly have configs, but A-3 wrote no
        harmonised table for them (their labels live in the raw CSVs and their structures are not
        in A-6's structure table), so they cannot be assembled yet - this function has to say so
        up front rather than have `load_rows()` raise a bare FileNotFoundError half an hour later.
        Args: none. Returns: task ids, sorted. Data source: data/interim/harmonised/.
        Side effects: read-only. Raises: none.
    """
    if not HARMONISED_DIR.is_dir():
        return []
    suffix = "_harmonised.parquet"
    return sorted(path.name[: -len(suffix)] for path in HARMONISED_DIR.glob(f"*{suffix}"))


def harmonised_path(task_id: str) -> Path:
    """给一个任务调和表的路径；不存在就抛并指出该跑哪一步。
    Return a task's harmonised path, raising with the stage that produces it.

    中文：
        用途：路径只在这里拼一次。缺文件时的消息要能直接回答"下一步做什么"。
        输入：task_id。输出：Path。数据来源：`data/interim/harmonised/`。
        配置来源：无。副作用：只读。异常：AssemblyError —— 文件不存在。

    English:
        Purpose: the only place the path is built; the failure message answers "what next".
        Args: task_id. Returns: a Path. Data source: data/interim/harmonised/.
        Side effects: read-only. Raises: AssemblyError when the file is missing.
    """
    path = HARMONISED_DIR / f"{task_id}_harmonised.parquet"
    if not path.is_file():
        raise AssemblyError(
            f"缺调和表 {path}：先跑 `python scripts/01_harmonise.py`（A-3）。"
            f"{task_id} 若是 PolyBench26 / OpenPoly 任务，A-3 从未为它产出调和表，"
            f"B-1 只装配有调和表的任务（见分工_B §2 B-1 的『拿什么』）"
        )
    return path


# ---- 2. 任务行表 / The task row table ----


def load_rows(task_id: str, *, limit: int | None = None) -> pd.DataFrame:
    """读一个任务的行表：`row_id` + 结构键 + 标签，索引是 `row_id`。
    Read a task's row table: row_id, structure key and label, indexed by row_id.

    中文：
        用途：这是"一行 = 一份样品"的那张表。三件事必须同时成立：
              ① `row_id` = 调和表的**行位置**（0 起），与 `splits/polybench/*.json` 的
                 `test_index` / `train_index` 同一口径——这是验收①的立足点；
              ② 结构键是 A-3 的 `smiles_canonical`（B-1 不重新规范化：A-6 已复核过 20,660 行
                 与 A-3 逐行一致，两处都再算一遍只会制造第二个口径）；
              ③ 标签**按配置里的 `target_column` 取**，不写死 `label`——PolyBench26 的列名是
                 `Tg_K`、`density` 之类，写死会在那两类任务上静默取错列。
              结构键为空的行走**不删**：删行会让后面的行全部前移，而划分文件里的位置不会跟着动，
              于是测试集悄悄换了一批样品。空键在这里如实保留（特征为 NaN），并由调用方记账。
        输入：task_id —— 任务 ID；limit —— 只取前 N 行（冒烟跑，调用方负责不写正式产物）。
        输出：DataFrame，索引名 `row_id`，列见 `ROW_COLUMNS`。
        数据来源：`data/interim/harmonised/<task>_harmonised.parquet`。
        配置来源：`target_column`（决定取哪一列当标签）。
        副作用：只读。异常：AssemblyError —— 缺表或缺 `target_column` 列。

    English:
        Purpose: the table where one row is one sample. Three things must hold together:
        (1) `row_id` is the harmonised **row position** (from 0), the same coordinate system as
        `test_index` / `train_index` in the PolyBench split JSONs - this is what criterion 1
        stands on; (2) the structure key is A-3's `smiles_canonical` (B-1 does not
        re-canonicalise: A-6 already re-checked all 20,660 rows against A-3, and a third pass
        would create a second convention); (3) the label is read via the config's
        `target_column`, never a hard-coded `label` - PolyBench26 calls it `Tg_K`, `density` and
        so on, and a hard-coded name would silently pick the wrong column there.
        Rows with an empty structure key are **kept**: dropping them would shift every later row
        while the split's positions stay put, quietly swapping the test set. They stay, with NaN
        features, and are accounted for by the caller.
        Args: task_id; limit (first N rows for a smoke run; the caller keeps it out of the official
        manifest). Returns: a DataFrame indexed by `row_id`.
        Data source: data/interim/harmonised/<task>_harmonised.parquet.
        Config: `target_column`. Side effects: read-only.
        Raises: AssemblyError on a missing table or a missing `target_column`.
    """
    import pandas as pd

    config = load_task_config(task_id)
    target_column = str(config["target_column"])
    columns = list(HARMONISED_COLUMNS)
    if target_column not in columns:
        columns.append(target_column)
    frame = pd.read_parquet(harmonised_path(task_id), columns=columns)
    if limit is not None:
        frame = frame.head(limit)
    if "source_row" not in frame.columns:
        raise AssemblyError(f"{task_id} 的调和表没有 `source_row`：无法确认行序口径")
    # 行序必须由 `source_row` 升序定义——这是 A-5 划分文件里 `row_order` 的原话。
    # 若哪天调和表被重新排序而划分没跟着改，这一行会当场炸出来，而不是几周后从 R² 里发现。
    # The order must be ascending `source_row`, exactly as A-5's `row_order` says. If the
    # harmonised table is ever re-sorted without the split following, this line fails loudly here
    # instead of surfacing as a weird R^2 weeks later.
    if not frame["source_row"].is_monotonic_increasing:
        raise AssemblyError(
            f"{task_id} 的调和表不是按 `source_row` 升序：与 splits 的 `row_order` 口径不符，"
            "位置索引会整体错位"
        )
    labels = pd.to_numeric(frame[target_column], errors="coerce").to_numpy(dtype="float64")
    psmiles = frame["smiles_canonical"].astype("object").where(
        frame["smiles_canonical"].notna(), None
    )
    rows = pd.DataFrame(
        {
            "row_id": np.arange(len(frame), dtype=np.int64),
            "psmiles_canonical": list(psmiles),
            "label": labels,
        },
        columns=list(ROW_COLUMNS),
    ).set_index("row_id")
    rows.index.name = "row_id"
    return rows


# ---- 3. 划分 / Splits ----


def _sha256_of(path: Path) -> str:
    """文件字节的 sha256（分块读）。
    sha256 of a file's bytes, read in chunks.

    中文：用途：`split_version` 要的是文件身份。输入：path。输出：64 位十六进制。
          数据来源：磁盘。配置来源：无。副作用：只读。异常：FileNotFoundError —— 文件不在。
    English: Purpose: `split_version` needs file identity. Args: path. Returns: a hex digest.
    """
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _relative(path: Path) -> str:
    """绝对路径转仓库相对的 POSIX 路径（清单里一律正斜杠）。
    Turn an absolute path into a repo-relative POSIX path (manifests use forward slashes).

    中文：用途：Windows 的反斜杠会让清单在不同机器上长得不一样（A-5 踩过）。
          输入：path。输出：字符串。数据来源：无。副作用：无。异常：无。
    English: Purpose: Windows backslashes make a manifest look different per machine (A-5 hit it).
    """
    try:
        return path.resolve().relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return path.as_posix()


def _polybench_split(task_id: str, config: dict[str, Any]) -> SplitSpec:
    """解析一份 PolyBench 划分 JSON（官方 70/30，没有验证集）。
    Parse one PolyBench split JSON (official 70/30, no validation set).

    中文：
        用途：PolyBench 的验证集是**空的**，不是"缺了没读"——官方只发 70/30，训练与测试互补。
              这里显式造一个空数组，让下游不必分支判断 `val_index` 是否存在。
        输入：task_id；config —— 任务配置。
        输出：`SplitSpec`（一折，名为 `official_70_30`）。
        数据来源：`splits/polybench/<task>/official_70_30.json`。
        配置来源：`split_ref` / `split_protocol`。
        副作用：只读。异常：AssemblyError —— 文件缺失或键不全。

    English:
        Purpose: PolyBench's validation set is **empty**, not "present but unread" - the platform
        ships 70/30 and train/test are complements. An explicit empty array keeps downstream free
        of `if val_index` branches.
        Args: task_id; config. Returns: a SplitSpec with one fold named `official_70_30`.
        Data source: splits/polybench/<task>/official_70_30.json.
        Config: `split_ref` / `split_protocol`. Side effects: read-only.
        Raises: AssemblyError on a missing file or missing keys.
    """
    path = REPO_ROOT / str(config["split_ref"])
    if not path.is_file():
        raise AssemblyError(f"缺划分文件 {path}：先跑 `python scripts/03_splits.py`（A-5）")
    payload = json.loads(path.read_text(encoding="utf-8"))
    missing = {"n_total", "test_index", "train_index"} - set(payload)
    if missing:
        raise AssemblyError(f"{path.name} 缺键 {sorted(missing)}：格式不对，别猜着读")
    fold = FoldSpec(
        name=str(payload.get("protocol") or config.get("split_protocol") or "official_70_30"),
        train_index=np.asarray(payload["train_index"], dtype=np.int64),
        val_index=np.zeros((0,), dtype=np.int64),
        test_index=np.asarray(payload["test_index"], dtype=np.int64),
    )
    return SplitSpec(
        task_id=task_id,
        benchmark="polybench",
        protocol=str(config.get("split_protocol") or fold.name),
        ref=_relative(path),
        sha256=_sha256_of(path),
        n_total=int(payload["n_total"]),
        row_order=str(payload.get("row_order", "")),
        folds=(fold,),
    )


def _polybench26_split(task_id: str, config: dict[str, Any]) -> SplitSpec:
    """解析 PolyBench26 的官方五折（`seed_42/fold_XX.npz` + `manifest.json`）。
    Parse PolyBench26's official five folds (`seed_42/fold_XX.npz` + `manifest.json`).

    中文：
        用途：这一类的索引指向**官方 canonical PSMILES 表的行号**，不是调和表位置——这里只把
              索引读出来，`row_order` 字段会如实标明它是"官方 canonical 表行号"，绝不冒充调和表
              位置。装配 PB26 还要一张结构表（A-6 没覆盖那些结构），因此本函数只负责"划分这一半"。
        输入：task_id；config。
        输出：`SplitSpec`（五折，名为 `fold_00`…`fold_04`）。
        数据来源：`splits/polybench26/<task>/seed_42/`。
        配置来源：`split_ref`（指向该目录下的 `manifest.json`）。
        副作用：只读。异常：AssemblyError —— 目录/文件缺失。

    English:
        Purpose: these indices address rows of the **official canonical PSMILES table**, not
        harmonised positions, and the `row_order` field says exactly that rather than pretending
        otherwise. Assembling PB26 also needs a structure table (A-6 does not cover those
        structures), so this function delivers the split half only.
        Args: task_id; config. Returns: a SplitSpec with five folds named fold_00..fold_04.
        Data source: splits/polybench26/<task>/seed_42/. Config: `split_ref`.
        Side effects: read-only. Raises: AssemblyError when the directory or a fold is missing.
    """
    manifest_path = REPO_ROOT / str(config["split_ref"])
    directory = manifest_path.parent
    if not directory.is_dir():
        raise AssemblyError(f"缺 PolyBench26 划分目录 {directory}：先跑 scripts/03_splits.py")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    folds: list[FoldSpec] = []
    for entry in manifest.get("folds", []):
        fold_path = directory / str(entry["indices_file"])
        with np.load(fold_path) as payload:
            folds.append(
                FoldSpec(
                    name=f"fold_{int(entry['fold']):02d}",
                    train_index=payload["train_pool"].astype(np.int64),
                    val_index=payload["val"].astype(np.int64),
                    test_index=payload["test"].astype(np.int64),
                    train_pool=payload["train_pool"].astype(np.int64),
                )
            )
    if not folds:
        raise AssemblyError(f"{manifest_path} 里没有 folds：格式不对，别猜着读")
    return SplitSpec(
        task_id=task_id,
        benchmark="polybench26",
        protocol=str(config.get("split_protocol") or "official_scaling_indices_5fold_seed42"),
        ref=_relative(manifest_path),
        sha256=_sha256_of(manifest_path),
        n_total=int(manifest.get("n_rows", 0)),
        row_order="官方 canonical PSMILES 表的行号（不是 A-3 调和表位置）",
        folds=tuple(folds),
    )


def _openpoly_split(task_id: str, config: dict[str, Any]) -> SplitSpec:
    """解析 OpenPoly 的规则复现划分（种子 0-4 当五折，D-24）。
    Parse OpenPoly's rule-replicated splits (seeds 0-4 act as the five folds, D-24).

    中文：
        用途：**这不是作者那一份划分**（D-24），所以 `protocol` 一律带 `rule_replicated`，
              且 `row_order` 抄成"发布顺序的基数据行号"，让下游知道索引指向原始 CSV 而不是调和表。
              五个折文件各有一份哈希，`sha256` 取五份按种子排序后的拼接哈希——单取一份会把
              "五份里有一份被换过"这种事藏起来。
        输入：task_id；config（`split_ref` 指向 `splits/openpoly/<task>/` 目录）。
        输出：`SplitSpec`（五折，名为 `seed0`…`seed4`）。
        数据来源：`splits/openpoly/<task>/rule_replicated_seed{0..4}.json`。
        配置来源：`split_ref` / `split_protocol`。
        副作用：只读。异常：AssemblyError —— 缺文件。

    English:
        Purpose: this is **not the authors' split** (D-24), so `protocol` always carries
        `rule_replicated` and `row_order` is copied as "published-order base-data row numbers", so
        downstream knows the indices address the raw CSV, not a harmonised table. The hash is the
        concatenated digest of all five files in seed order - taking one would hide "one of the
        five was swapped".
        Args: task_id; config (`split_ref` is the directory). Returns: a SplitSpec with five folds.
        Data source: splits/openpoly/<task>/rule_replicated_seed{0..4}.json.
        Config: `split_ref` / `split_protocol`. Side effects: read-only.
        Raises: AssemblyError when a file is missing.
    """
    directory = REPO_ROOT / str(config["split_ref"])
    folds: list[FoldSpec] = []
    digests: list[str] = []
    for seed in OPENPOLY_FOLD_SEEDS:
        path = directory / f"rule_replicated_seed{seed}.json"
        if not path.is_file():
            raise AssemblyError(f"缺 OpenPoly 划分 {path}：先跑 scripts/03_splits.py（A-5）")
        payload = json.loads(path.read_text(encoding="utf-8"))
        digests.append(_sha256_of(path))
        folds.append(
            FoldSpec(
                name=f"seed{seed}",
                train_index=np.asarray(payload["train_index"], dtype=np.int64),
                val_index=np.asarray(payload["val_index"], dtype=np.int64),
                test_index=np.asarray(payload["test_index"], dtype=np.int64),
            )
        )
    combined = hashlib.sha256("".join(digests).encode("ascii")).hexdigest()
    return SplitSpec(
        task_id=task_id,
        benchmark="openpoly",
        protocol=str(config.get("split_protocol") or "replicated_8_1_1_stratified10"),
        ref=_relative(directory),
        sha256=combined,
        n_total=int(json.loads((directory / "rule_replicated_seed0.json").read_text())["n_total"]),
        row_order="原始 CSV 的发布顺序行号（不是 A-3 调和表位置）",
        folds=tuple(folds),
    )


def limit_split(spec: SplitSpec, n_rows: int) -> SplitSpec:
    """把划分裁到前 `n_rows` 行——**只给冒烟跑用**，正式产物一律用完整划分。
    Trim a split to the first `n_rows` rows - **smoke runs only**; official artefacts always use
    the full split.

    中文：
        用途：`--limit 50` 是把行表截了 50 行，而划分文件说的还是 228 行、索引最大到 227。
              若不裁，装配会正确地拒绝（"索引越界"），于是冒烟跑根本跑不起来；若悄悄跳过这项检查，
              冒烟跑的产物就会被当成正式产物。折中办法：把每折的索引过滤成 `< n_rows`，
              并**保留同一个坐标系**（仍是行位置），只是这一折少了一些行。
              裁完若某折空了，说明 limit 太小，当场报错——空训练集跑出来的东西没有意义。
        输入：spec —— 完整划分；n_rows —— 截断后的行数。
        输出：新的 `SplitSpec`（`n_total=n_rows`，折名不变）。
        数据来源：调用方。配置来源：无。副作用：无（新对象）。
        异常：AssemblyError —— 某一折裁完为空。

    English:
        Purpose: `--limit 50` truncates the row table to 50 rows while the split still describes
        228 rows with indices up to 227. Without trimming, assembly correctly refuses
        ("index out of range") and the smoke run cannot run at all; skipping the check instead would
        let a smoke artefact pass as an official one. The compromise: filter each fold's indices to
        `< n_rows` while keeping the same coordinate system (still row positions), just with fewer
        rows in that fold. If a fold ends up empty the limit is too small and this raises - training
        on nothing means nothing.
        Args: spec; n_rows. Returns: a new SplitSpec with `n_total=n_rows`, same fold names.
        Data source: the caller. Side effects: none (a new object).
        Raises: AssemblyError when a fold becomes empty.
    """
    folds: list[FoldSpec] = []
    for fold in spec.folds:
        trimmed = FoldSpec(
            name=fold.name,
            train_index=fold.train_index[fold.train_index < n_rows],
            val_index=fold.val_index[fold.val_index < n_rows],
            test_index=fold.test_index[fold.test_index < n_rows],
            train_pool=None if fold.train_pool is None else fold.train_pool[fold.train_pool < n_rows],
        )
        if trimmed.train_index.size == 0 or trimmed.test_index.size == 0:
            raise AssemblyError(
                f"{spec.task_id}/{fold.name}：--limit {n_rows} 太小，裁完训练集或测试集为空；"
                "冒烟请给到四位数"
            )
        folds.append(trimmed)
    return SplitSpec(
        task_id=spec.task_id,
        benchmark=spec.benchmark,
        protocol=spec.protocol,
        ref=spec.ref,
        sha256=spec.sha256,
        n_total=n_rows,
        row_order=spec.row_order,
        folds=tuple(folds),
    )


def load_split(task_id: str) -> SplitSpec:
    """按基准类型分派，给出统一形态的划分。
    Dispatch by benchmark and return a uniform split.

    中文：
        用途：三种基准的划分文件格式完全不同（JSON / NPZ / JSON×N），凡是"读划分"的地方都走这里，
              避免第二份解析实现。**只读**：本函数不写 `splits/`，也不重算任何划分（红线 2）。
        输入：task_id。
        输出：`SplitSpec`。
        数据来源：`splits/**`。
        配置来源：`benchmark` + `split_ref`。
        副作用：只读。异常：AssemblyError —— 基准不认识或文件缺失。

    English:
        Purpose: the three benchmarks ship completely different split formats; every reader goes
        through here so there is never a second parser. **Read-only**: nothing under `splits/` is
        written or recomputed (red line 2).
        Args: task_id. Returns: a SplitSpec. Data source: splits/**.
        Config: `benchmark` + `split_ref`. Side effects: read-only.
        Raises: AssemblyError on an unknown benchmark or a missing file.
    """
    config = load_task_config(task_id)
    benchmark = str(config["benchmark"])
    if benchmark == "polybench":
        return _polybench_split(task_id, config)
    if benchmark == "polybench26":
        return _polybench26_split(task_id, config)
    if benchmark == "openpoly":
        return _openpoly_split(task_id, config)
    raise AssemblyError(f"{task_id} 的 benchmark={benchmark!r} 不认识")


def check_split_matches_rows(spec: SplitSpec, n_rows: int) -> list[str]:
    """核划分与行表是否配套（折内互斥、覆盖全表、不越界）。
    Check that a split fits the row table: folds are disjoint, cover it, and stay in range.

    中文：
        用途：验收①要求"行数 = 调和后可用标签数，且行序由划分文件决定"。这条只能在**跑装配之前**
              核，因为错位的划分不会让装配报错——它只会给出一组好看的指标。三类错各查一遍：
              ① 索引越界（划分文件与调和表版本不配套）；② 折内有重叠（同一行既是训练又是测试）；
              ③ 并集不等于 0..n-1（有行不属于任何一侧，通常是调和表被重跑过）。
              缺验证集不算错：PolyBench 官方就没有。
        输入：spec —— 划分；n_rows —— 行表行数。
        输出：问题字符串列表；空列表表示配套。
        数据来源：调用方。配置来源：无。副作用：只读。异常：无（收集问题而非抛出）。

    English:
        Purpose: criterion 1 says "row count equals the usable-label count and the row order is
        decided by the split file", and that can only be checked *before* assembly, because a
        misaligned split never raises - it just produces pretty metrics. Three failure modes are
        checked: out-of-range indices (split and harmonised table are from different versions),
        overlap between folds (one row is both train and test), and a union that is not 0..n-1 (a
        row belongs to neither side, usually because the harmonised table was rebuilt). A missing
        validation set is not an error: PolyBench officially has none.
        Args: spec; n_rows. Returns: problem strings; empty means it fits.
        Data source: the caller. Side effects: read-only. Raises: none.
    """
    problems: list[str] = []
    if spec.n_total and spec.n_total != n_rows:
        problems.append(
            f"{spec.task_id}：划分文件 n_total={spec.n_total} 与行表 {n_rows} 不一致"
        )
    for fold in spec.folds:
        sides = {"train": fold.train_index, "val": fold.val_index, "test": fold.test_index}
        for name, index in sides.items():
            if index.size == 0:
                continue
            if int(index.min()) < 0 or int(index.max()) >= n_rows:
                problems.append(
                    f"{spec.task_id}/{fold.name}：{name} 索引越界（最大 {int(index.max())} ≥ {n_rows}）"
                )
        flat = np.concatenate([index for index in sides.values() if index.size])
        if flat.size and len(np.unique(flat)) != flat.size:
            problems.append(f"{spec.task_id}/{fold.name}：折内有行同时落在两侧（索引重复）")
        expected = int(sum(index.size for index in sides.values()))
        if expected != n_rows:
            problems.append(
                f"{spec.task_id}/{fold.name}：三侧合计 {expected} 行 ≠ 行表 {n_rows} 行"
            )
    return problems


# ---- 4. 结构级特征 / Structure-level features ----


def load_structure_table_index() -> pd.DataFrame:
    """读 A-6 的结构表，按 `psmiles_canonical` 建索引（不用 `structure_id`）。
    Load A-6's structure table indexed by `psmiles_canonical` (never by `structure_id`).

    中文：
        用途：`structure_id` 是全局行号，任务集一变就整体平移（A-6 交付文档 §6 第 5 条），
              所以对外一律按规范式索引。
        输入：无。
        输出：DataFrame（列含 `psmiles_canonical` / `wpsmiles` / `n_stars`）。
        数据来源：`data/processed/structures/psmiles.parquet`（不入库，靠 02 重建）。
        配置来源：无。副作用：只读。异常：AssemblyError —— 缺结构表。

    English:
        Purpose: `structure_id` is a global row number and shifts whenever the task set changes
        (A-6 delivery note, section 6, item 5), so the public index is always the canonical form.
        Args: none. Returns: a DataFrame with psmiles_canonical / wpsmiles / n_stars.
        Data source: data/processed/structures/psmiles.parquet (not committed; rebuild with 02).
        Side effects: read-only. Raises: AssemblyError when the table is missing.
    """
    if not STRUCTURES_PATH.is_file():
        raise AssemblyError(
            f"缺结构表 {_relative(STRUCTURES_PATH)}：先跑 "
            "`python scripts/02_features.py --source all`（约 2 分 40 秒）"
        )
    import pandas as pd

    frame = pd.read_parquet(STRUCTURES_PATH)
    return frame.set_index("psmiles_canonical", drop=False)


def load_descriptor_table(task_id: str) -> pd.DataFrame:
    """读 A-6 给某任务落的"结构键 + 205 描述符 + 2048 位指纹"表，按结构键索引。
    Load A-6's per-task structure-key + descriptors + fingerprint table, indexed by structure key.

    中文：
        用途：这是装配的原料。**按 `psmiles_canonical` 建索引**，调用方用 `reindex` 一次取齐，
              绝不按行序对齐——A-6 的表里一行是一**个结构**，而任务行表里一行是一**次测量**，
              同一个结构可能占多行，行序对齐在数学上就是错的。
        输入：task_id。
        输出：DataFrame（索引为规范式；列含 205 描述符 + 2048 位指纹）。
        数据来源：`data/processed/descriptors/<task>.parquet`。
        配置来源：无。副作用：只读。异常：AssemblyError —— 缺文件。

    English:
        Purpose: the raw material for assembly. Indexed by `psmiles_canonical` so the caller can
        `reindex` in one shot; row-order alignment is never used, because A-6's rows are
        *structures* while the task's rows are *measurements*, and one structure may occupy
        several rows - aligning by row order is mathematically wrong.
        Args: task_id. Returns: a DataFrame indexed by canonical form.
        Data source: data/processed/descriptors/<task>.parquet.
        Side effects: read-only. Raises: AssemblyError when the file is missing.
    """
    path = DESCRIPTORS_DIR / f"{task_id}.parquet"
    if not path.is_file():
        raise AssemblyError(
            f"缺特征文件 {_relative(path)}：先跑 `python scripts/02_features.py --source all`"
        )
    import pandas as pd

    frame = pd.read_parquet(path)
    return frame.set_index("psmiles_canonical", drop=False)


def feature_pack_path(task_id: str) -> Path:
    """给任务级装配产物的路径（`.npz`）。
    Return the path of a task-level assembly artefact (`.npz`).

    中文：用途：路径只在这里拼一次。输入：task_id。输出：Path（文件可能还不存在）。
          数据来源：`data/processed/features/`（不入库）。配置来源：无。副作用：无。异常：无。
    English: Purpose: one place for the path. Args: task_id. Returns: a Path (may not exist yet).
    """
    return FEATURES_DIR / f"{task_id}.npz"
