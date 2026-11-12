"""任务级特征装配：把结构级特征按 `row_id` 摊到每一行上，行序由划分文件决定。
Task-level feature assembly: spread structure-level features onto every row by `row_id`, with the
row order decided by the split file.

中文：
    用途：B-1 的主体。上游 A-6 给的是"每个**结构**一行"的画像（11,494 个结构），而模型要的是
          "每次**测量**一行"的矩阵（20,660 行）。本模块做这一次投影，并把四件事钉死：
          ① **行序**：`row_id` = A-3 调和表的行位置，与 `splits/polybench/*.json` 的
             `test_index` 同一坐标系；`X` 的第 k 行就是 `row_id=k`，**不重排、不 dropna**；
          ② **join 键**：`psmiles_canonical`，一次 `reindex` 取齐，绝不按行序对齐；
          ③ **不掺全量统计量**：落盘的矩阵是**原始值**（NaN 照留），标准化只提供"在训练集上 fit"
             的函数，由调用方按折传入训练索引——验收④查的就是这条；
          ④ **丢了多少、丢了谁**：结构键为空 / 不在 A-6 表里的行逐条记进
             `data/manifests/task_feature_unresolved.csv`，不许静默少一行。
    依赖：`chainworth.io.loaders`（取数）、`chainworth.chem.descriptors`（列名与列序）、
          numpy/pandas/pyarrow。
    配置：无 yaml——特征空间是口径（记 D-25），不是实验参数；任务清单来自
          `chainworth.io.loaders.load_task_config()`。
    对外：`assemble_task()`、`save_pack()`、`load_pack()`、`standardize_fit()`、
          `apply_standardizer()`、`verify_features()`、`FEATURE_COLUMNS_*` 常量。
    被谁用：`scripts/03b_assemble.py`（阶段入口）、`tests/test_task_assembly.py`（断言）、
            C 的 A2/A5 臂与 B 的 A3/A4 臂（`load_pack()`）。

English:
    Purpose: the body of B-1. A-6 ships one row per **structure** (11,494 of them); a model needs
    one row per **measurement** (20,660). This module performs that projection and pins four
    things: (1) row order - `row_id` is A-3's harmonised row position, the same coordinate system
    as `test_index` in the PolyBench split JSONs; row k of `X` is `row_id=k`, nothing is reordered
    and nothing is dropped; (2) join key - `psmiles_canonical` via a single `reindex`, never row
    order; (3) no global statistics - the matrix on disk holds **raw** values with NaN intact,
    and standardisation is offered only as "fit on the training index" so the caller passes the
    fold's train index; that is exactly what criterion 4 checks; (4) what was lost - rows with an
    empty structure key or a key missing from A-6's table are listed one by one in
    data/manifests/task_feature_unresolved.csv, never silently dropped.
    Depends on: chainworth.io.loaders, chainworth.chem.descriptors (column names and order),
    numpy/pandas/pyarrow.
    Config: no yaml - the feature space is a definition (D-25), not an experiment parameter; the
    task list comes from chainworth.io.loaders.load_task_config().
    Public API: `assemble_task()`, `save_pack()`, `load_pack()`, `standardize_fit()`,
    `apply_standardizer()`, `verify_features()` and the FEATURE_COLUMNS_* constants.
    Callers: scripts/03b_assemble.py, tests/test_task_assembly.py, C's A2/A5 arms and B's A3/A4
    arms (through `load_pack()`).

为什么把描述符与指纹分两块存 ／ Why descriptors and fingerprints are two blocks
--------------------------------------------------------------------------------
描述符是 float64 的连续量（205 列），指纹是 0/1 的位（2048 列）。合成一个 float32 矩阵会同时
坏两件事：描述符的数值精度（验收②要求与 A 的表**逐位**相等，float32 过不了），以及指纹"这是
一位而不是一个数"的语义（下游若做标准化，位会被当成连续量缩放）。
分开存，`stack()` 在需要时拼——拼一次 20,660 × 2,253 的 float32 只要几十毫秒。
Descriptors are float64 continuous values (205 columns); fingerprints are 0/1 bits (2048 columns).
Merging them into one float32 matrix breaks two things at once: descriptor precision (criterion 2
requires **bit-for-bit** equality with A's table, which float32 fails) and the "a bit is not a
number" semantics of the fingerprint (a downstream scaler would rescale bits). Kept apart, with
`stack()` to join them on demand - joining 20,660 x 2,253 float32 takes milliseconds.
"""

from __future__ import annotations

import csv
import hashlib
import json
import logging
import os
import warnings
from collections.abc import Mapping, Sequence
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

from chainworth.io import loaders as L

# `pandas`、`chainworth.chem.descriptors`（要 rdkit）与 `chainworth.chem.structures`（要 pandas）
# **都刻意不在模块层导入**：A4 臂跑在 `DSC6020_Project_Polybert` 环境里，那个环境有 numpy/pyarrow
# 但**没有 pandas、没有 rdkit**（`lock/polybert.pip.txt`），而它要的就是本模块的 `load_pack()`——
# 读一个 `.npz`。把这些依赖留在顶层，等于让"读一个 npz"也要先装好化学信息学全家桶。
# `pandas`, `descriptors` (rdkit) and `structures` (pandas) are deliberately **not** imported at
# module level: the A4 arm runs in the polyBERT environment, which ships numpy/pyarrow but neither
# pandas nor rdkit, and all it needs here is `load_pack()` - a plain `.npz` read.

LOGGER = logging.getLogger("chainworth.features")

REPO_ROOT = Path(__file__).resolve().parents[3]     # src/chainworth/chem/features.py -> 仓库根
FEATURES_DIR = REPO_ROOT / "data" / "processed" / "features"
MANIFEST_DIR = REPO_ROOT / "data" / "manifests"
MANIFEST_PATH = MANIFEST_DIR / "task_features_manifest.csv"
UNRESOLVED_PATH = MANIFEST_DIR / "task_feature_unresolved.csv"

# 清单列。顺序即写入顺序，读的人不用猜。
# The manifest's columns, in write order so a reader never has to guess.
MANIFEST_COLUMNS = (
    "task_id",
    "benchmark",
    "protocol",
    "n_rows",
    "n_label_rows",
    "n_unique_structures",
    "n_descriptors",
    "n_fingerprint_bits",
    "n_columns_total",
    "n_nan_cells",
    "n_rows_without_structure",
    "n_structures_without_wpsmiles",
    "feature_stats",
    "row_order_basis",
    "split_ref",
    "split_sha256",
    "pack_file",
    "pack_sha256",
    "built_at",
)

# 未解析清单列：一行一**行**（不是一结构），因为这里要能回答"哪个任务的第几行没特征"。
# The unresolved manifest is one row per **task row** (not per structure), because the question it
# must answer is "which task's which row has no features".
UNRESOLVED_COLUMNS = ("task_id", "row_id", "psmiles_canonical", "reason")

# 落盘矩阵里出现这些值就说明有人把全量统计量写进了特征（验收④）。
# These values in the on-disk matrix would mean somebody baked full-data statistics into features.
FEATURE_STATS_NONE = "none（原始值；标准化/插补由调用方在 train 上 fit）"


def sha256_of(path: Path) -> str:
    """文件字节的 sha256（分块读，别把 20 MB 的 npz 一次读进内存）。
    sha256 of a file's bytes, read in chunks.

    中文：用途：产物身份。输入：path。输出：64 位十六进制。数据来源：磁盘。
          配置来源：无。副作用：只读。异常：FileNotFoundError —— 文件不在。
    English: Purpose: artefact identity. Args: path. Returns: a hex digest.
    """
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def repo_relative(path: Path) -> str:
    """绝对路径转仓库相对的 POSIX 路径。
    Turn an absolute path into a repo-relative POSIX path.

    中文：用途：清单跨机器可读（A-5 踩过反斜杠的坑）。输入：path。输出：字符串。
          数据来源：无。配置来源：无。副作用：无。异常：无。
    English: Purpose: manifests stay readable across machines (A-5 hit the backslash trap).
    """
    try:
        return path.resolve().relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return path.as_posix()


def now_iso() -> str:
    """当前本地时间（带时区、秒级）。/ The current local timestamp with offset, second precision.

    中文：用途：清单的时间列。输入：无。输出：ISO 字符串。数据来源：系统时钟。
          副作用：无。异常：无。
    English: Purpose: the manifest's time column. Args: none. Returns: an ISO string.
    """
    return datetime.now().astimezone().isoformat(timespec="seconds")


def write_csv_atomic(rows: Sequence[Mapping[str, object]], path: Path, columns: Sequence[str]) -> None:
    """原子写 CSV，行尾固定 LF（`.gitattributes` 钉死）。
    Write a CSV atomically with LF endings (pinned by .gitattributes).

    中文：用途：清单是证据，写一半比没有更糟。输入：rows / path / columns。
          数据来源：调用方。配置来源：无。副作用：建目录、写一个文件。
          异常：KeyError —— 某行缺列。
    English: Purpose: a manifest is evidence and half of one is worse than none.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(columns), lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column, "") for column in columns})
    os.replace(temporary, path)


def read_manifest_rows(path: Path) -> list[dict[str, str]]:
    """把清单读成行字典列表；不存在返回空。/ Read a manifest; a missing file reads as empty.

    中文：用途：重跑时判断产物是否同一份字节，从而沿用旧的 `built_at`（A-2/A-3 同一个坑：
          每次刷新时间戳会让冻结物在 git 里显得被人动过）。
          输入：path。输出：行字典列表。数据来源：磁盘 CSV。副作用：只读。异常：无。
    English: Purpose: decide whether a re-run produced the same bytes so the old `built_at` can be
             kept (A-2/A-3 hit this: refreshing timestamps makes frozen files look hand-edited).
    """
    if not path.is_file():
        return []
    try:
        with path.open("r", encoding="utf-8", newline="") as handle:
            return list(csv.DictReader(handle))
    except (OSError, csv.Error):
        return []


# ---- 1. 特征空间 / The feature space ----


def descriptor_columns() -> list[str]:
    """205 个描述符列名与列序（直接取 A-6 的定义，不另写一份）。
    The 205 descriptor names and order, taken from A-6's definition rather than re-typed.

    中文：
        用途：列序一变、模型就不能复用（同名不同序的矩阵乘出来是垃圾且不报错）。所以这里**不**
              定义列表，而是转调 `descriptors.descriptor_columns()`——两处各写一份，迟早不一致。
        输入：无。输出：列名列表。数据来源：`chainworth.chem.descriptors`。
        配置来源：`descriptors.DROPPED_DESCRIPTORS`。副作用：无。异常：无。

    English:
        Purpose: a changed column order silently invalidates every trained model. So this function
        does **not** define its own list; it delegates to `descriptors.descriptor_columns()` - two
        copies would drift apart sooner or later.
        Args: none. Returns: the column names. Data source: chainworth.chem.descriptors.
        Config: `descriptors.DROPPED_DESCRIPTORS`. Side effects: none.
    """
    # 局部导入：本函数要 rdkit，而本模块的 `load_pack()` 不该因为读一个 npz 就要求装 rdkit。
    # Local import: this function needs rdkit, while `load_pack()` must not.
    from chainworth.chem import descriptors as D

    return D.descriptor_columns()


def fingerprint_columns() -> list[str]:
    """2048 个指纹列名（`morgan_0000` … `morgan_2047`），与 A-6 的列名逐字相同。
    The 2048 fingerprint column names (`morgan_0000` ... `morgan_2047`), identical to A-6's.

    中文：
        用途：列名里带位号，读的人不用翻文档；与 A-6 逐字相同才保证两份产物可以直接对照。
        输入：无。输出：列名列表。数据来源：`structures.FINGERPRINT_PREFIX` + `descriptors.FINGERPRINT_BITS`。
        配置来源：同上两个常量。副作用：无。异常：无。

    English:
        Purpose: the bit index is in the name and the spelling matches A-6 exactly, so the two
        artefacts can be compared field by field.
        Args: none. Returns: the column names. Data source: `FINGERPRINT_PREFIX` + `FINGERPRINT_BITS`.
    """
    from chainworth.chem import descriptors as D
    from chainworth.chem.structures import FINGERPRINT_PREFIX

    return [f"{FINGERPRINT_PREFIX}{index:04d}" for index in range(D.FINGERPRINT_BITS)]


# ---- 2. 装配 / Assembly ----


def assemble_task(
    task_id: str, *, limit: int | None = None, flow: str = "depth"
) -> dict[str, Any]:
    """把一个任务装配成"每次测量一行"的特征矩阵，返回含未解析清单与统计的摘要。
    Assemble one task into a per-measurement feature matrix; return a summary with the unresolved
    list and the counts.

    中文：
        用途：B-1 的核心函数。步骤与每一步的理由：
              ① 读行表（`load_rows()`）——`row_id` 就是行位置，与划分同一坐标系；
              ② 读 A-6 的特征表并按 `psmiles_canonical` **reindex**——一次取齐，行序与行表对齐；
                 取不到的键（结构不在 A-6 表里）留 NaN 并记账，**不删行**；
              ③ 结构键为空的行同样留 NaN 并记账（`reason=no_structure`）——删行会让后面全部前移；
              ④ 记 NaN 单元数：这不是错误，是"某些描述符对某些结构天然无定义"（A-6 §6 第 4 条），
                 下游要能分清"NaN 本来就有"和"我们算漏了"。
        输入：task_id —— 任务 ID；limit —— 只取前 N 行（冒烟）；flow —— 预留的流程标签，
              当前不参与计算（A3/A4 的输入完全一样，臂的差别在训练脚本里，不在这里）。
        输出：摘要 dict，键含 `row_id` / `y` / `psmiles` / `desc` / `fp` / `unresolved` / 统计。
        数据来源：调和表 + A-6 特征表 + 结构表。
        配置来源：`target_column`（行表）、`benchmark`（划分）。
        副作用：只读（不写盘；写盘是 `save_pack()` 的事）。
        异常：AssemblyError —— 缺上游产物或划分与行数不配套。

    English:
        Purpose: B-1's core function, step by step and why:
        (1) read the row table - `row_id` is the row position, the split's own coordinate system;
        (2) read A-6's feature table and **reindex** by `psmiles_canonical` - one shot, row order
        matching the row table; keys that cannot be found stay NaN and are recorded, never dropped;
        (3) rows whose structure key is empty also stay NaN and are recorded
        (`reason=no_structure`) - dropping a row shifts every later one;
        (4) count NaN cells: that is not an error but "some descriptors are undefined for some
        structures" (A-6 section 6 item 4), and downstream must tell "NaN was always there" from
        "we missed something".
        Args: task_id; limit (first N rows, smoke); flow (reserved label, unused today - A3 and A4
        consume the same matrix, the arm difference lives in the training script).
        Returns: a summary dict with row_id / y / psmiles / desc / fp / unresolved and the counts.
        Data source: harmonised table + A-6 feature table + structure table.
        Config: `target_column`, `benchmark`. Side effects: none (saving is `save_pack()`'s job).
        Raises: AssemblyError on a missing upstream artefact or a split that does not fit.
    """
    del flow  # 见 docstring：臂的差别不在装配层。
    rows = L.load_rows(task_id, limit=limit)
    split = L.load_split(task_id)
    if limit is not None:
        split = L.limit_split(split, len(rows))
    problems = L.check_split_matches_rows(split, len(rows))
    if problems:
        raise L.AssemblyError("；".join(problems))

    table = L.load_descriptor_table(task_id)
    columns = descriptor_columns()
    bits = fingerprint_columns()
    expected = ["psmiles_canonical", *columns, *bits]
    missing = [name for name in expected if name not in table.columns]
    if missing:
        raise L.AssemblyError(
            f"{task_id} 的 A-6 特征表缺列 {missing[:4]}…：特征空间与当前 descriptors 模块不一致，"
            "按 `python scripts/02_features.py --source all --force` 重出"
        )

    keys = list(rows["psmiles_canonical"])
    # 空键不能进 `reindex`：pandas 会把 None 当成"取这一行"，语义与"缺这个键"不同。
    # Empty keys must not go into `reindex`: pandas reads None as "this row", which is not the same
    # thing as "this key is missing".
    lookup = [key if isinstance(key, str) else "__missing__" for key in keys]
    picked = table.reindex(lookup)
    desc = picked[columns].to_numpy(dtype="float64")
    fp = picked[bits].to_numpy(dtype="uint8")
    found = picked["psmiles_canonical"].notna().to_numpy()

    unresolved: list[dict[str, object]] = []
    for position, (key, ok) in enumerate(zip(keys, found)):
        if ok:
            continue
        unresolved.append(
            {
                "task_id": task_id,
                "row_id": position,
                "psmiles_canonical": "" if key is None else str(key),
                "reason": "no_structure" if key is None else "not_in_structure_table",
            }
        )
    if unresolved:
        LOGGER.warning(
            "%s：%d 行拿不到结构级特征（%s）——行**未删除**，特征留 NaN，逐条记进 %s",
            task_id,
            len(unresolved),
            "/".join(sorted({str(item["reason"]) for item in unresolved})),
            repo_relative(UNRESOLVED_PATH),
        )

    structures = L.load_structure_table_index()
    known = [key for key in keys if isinstance(key, str)]
    without_wpsmiles = int(structures.reindex(list(dict.fromkeys(known)))["wpsmiles"].isna().sum())
    labels = rows["label"].to_numpy(dtype="float64")
    n_nan = int(np.isnan(desc).sum())

    summary: dict[str, Any] = {
        "task_id": task_id,
        "benchmark": split.benchmark,
        "protocol": split.protocol,
        "n_rows": int(len(rows)),
        "n_label_rows": int(np.isfinite(labels).sum()),
        "n_unique_structures": int(len(set(known))),
        "n_rows_without_structure": len(unresolved),
        "n_structures_without_wpsmiles": without_wpsmiles,
        "n_nan_cells": n_nan,
        "n_descriptors": len(columns),
        "n_fingerprint_bits": len(bits),
        "n_columns_total": len(columns) + len(bits),
        "row_order_basis": f"{split.ref}#row_position（与 splits 的 test_index 同一坐标系）",
        "split_ref": split.ref,
        "split_sha256": split.sha256,
        "row_id": rows.index.to_numpy(dtype=np.int64),
        "y": labels,
        "psmiles": np.asarray(["" if key is None else str(key) for key in keys], dtype=str),
        "desc": desc,
        "fp": fp,
        "unresolved": unresolved,
        "desc_columns": columns,
        "fp_columns": bits,
    }
    LOGGER.info(
        "%s：%d 行 × %d 列（%d 描述符 + %d 位）；NaN 单元 %d；无结构 %d 行；唯一结构 %d",
        task_id,
        summary["n_rows"],
        summary["n_columns_total"],
        summary["n_descriptors"],
        summary["n_fingerprint_bits"],
        n_nan,
        summary["n_rows_without_structure"],
        summary["n_unique_structures"],
    )
    return summary


def stack(summary: Mapping[str, Any]) -> np.ndarray:
    """把描述符块与指纹块拼成一个 float32 矩阵（训练脚本用的形状）。
    Join the descriptor and fingerprint blocks into one float32 matrix (the training shape).

    中文：
        用途：模型只认一个矩阵。拼的时候把指纹从 0/1 转成 0.0/1.0，描述符保持值不变——
              **这里不做任何缩放**，缩放要用训练集的统计量，见 `standardize_fit()`。
        输入：summary —— `assemble_task()` 的输出（或 `load_pack()` 的同名字典）。
        输出：shape (n, 2253) 的 float32 数组。
        数据来源：调用方。配置来源：无。副作用：无（新数组）。
        异常：KeyError —— 缺 `desc` / `fp`。

    English:
        Purpose: a model takes one matrix. The fingerprint becomes 0.0/1.0 and the descriptors keep
        their values - **no scaling happens here**, because scaling needs training-set statistics
        (see `standardize_fit()`).
        Args: summary. Returns: a (n, 2253) float32 array. Data source: the caller.
        Side effects: none (a new array). Raises: KeyError when `desc` or `fp` is absent.
    """
    return np.hstack(
        [np.asarray(summary["desc"], dtype=np.float32), np.asarray(summary["fp"], dtype=np.float32)]
    )


# ---- 3. 落盘与读回 / Saving and loading ----


def save_pack(summary: Mapping[str, Any], *, partial: bool = False) -> Path:
    """把装配结果写成 `.npz`（压缩、无 pickle），返回文件路径。
    Write the assembly to a compressed, pickle-free `.npz` and return the path.

    中文：
        用途：`data/processed/*` 不入库（D-04），所以这里存的是"重跑一次就有的东西"，格式要
              稳、要能被别的语言读。**不用 pickle**：`np.load(..., allow_pickle=True)` 等于允许
              在加载时执行代码，而这份文件会被三个人各自的脚本打开。
              存**明文结构串**而不是"结构索引"：索引要另一张表才能解释，多一张表就多一次错位机会。
        输入：summary —— `assemble_task()` 的输出；partial —— 限量跑（写 `*.partial.npz`）。
        输出：写出的路径。
        数据来源：调用方。配置来源：无。副作用：建目录、写一个文件（约 20 MB）。
        异常：OSError —— 无写权限或磁盘满。

    English:
        Purpose: data/processed is not committed (D-04), so this stores "something a re-run
        reproduces"; the format must be stable and readable from another language. **No pickle**:
        `np.load(..., allow_pickle=True)` means executing code at load time, and three people's
        scripts open this file. Plain structure strings are stored instead of structure indices -
        an index needs a second table to interpret, and a second table is a second chance to
        misalign.
        Args: summary; partial (write `*.partial.npz`). Returns: the written path.
        Side effects: creates the directory and writes one ~20 MB file. Raises: OSError.
    """
    task_id = str(summary["task_id"])
    target = (
        FEATURES_DIR / f"{task_id}.partial.npz" if partial else L.feature_pack_path(task_id)
    )
    target.parent.mkdir(parents=True, exist_ok=True)
    meta = {
        "task_id": task_id,
        "benchmark": summary["benchmark"],
        "protocol": summary["protocol"],
        "split_ref": summary["split_ref"],
        "split_sha256": summary["split_sha256"],
        "row_order_basis": summary["row_order_basis"],
        "feature_stats": FEATURE_STATS_NONE,
        "format_version": 1,
    }
    np.savez_compressed(
        target,
        row_id=np.asarray(summary["row_id"], dtype=np.int64),
        y=np.asarray(summary["y"], dtype=np.float64),
        psmiles=np.asarray(summary["psmiles"], dtype=str),
        desc=np.asarray(summary["desc"], dtype=np.float64),
        fp=np.asarray(summary["fp"], dtype=np.uint8),
        desc_columns=np.asarray(summary["desc_columns"], dtype=str),
        fp_columns=np.asarray(summary["fp_columns"], dtype=str),
        meta=np.asarray(json.dumps(meta, ensure_ascii=False)),
    )
    return target


def load_pack(task_id: str) -> dict[str, Any]:
    """读回一个任务的装配产物（C 的 A2/A5 臂与 B 的 A3/A4 臂的唯一入口）。
    Load one task's assembly artefact - the single entry point for C's and B's arms.

    中文：
        用途：下游不许自己去 `np.load`：那样每处都要记得 `allow_pickle=False`、记得 `row_id`
              是索引而不是特征列。这里一次说清，并顺手核"行号是不是 0..n-1 连续"——
              不是的话说明文件被拼过或截过，**这时报错比给出一个错位矩阵好**。
        输入：task_id。
        输出：dict，键为 `row_id` / `y` / `psmiles` / `desc` / `fp` / `desc_columns` /
              `fp_columns` / `meta`；`stack()` 可直接吃。
        数据来源：`data/processed/features/<task>.npz`。
        配置来源：无。副作用：只读。
        异常：AssemblyError —— 文件缺失；ValueError —— 行号不连续（文件不可信）。

    English:
        Purpose: downstream must not call `np.load` itself - every call site would have to
        remember `allow_pickle=False` and that `row_id` is an index rather than a feature column.
        This says it once and also checks that the row numbers are 0..n-1; if they are not, the
        file was spliced or truncated, and **failing here beats handing over a misaligned matrix**.
        Args: task_id. Returns: a dict with row_id / y / psmiles / desc / fp / columns / meta.
        Data source: data/processed/features/<task>.npz. Side effects: read-only.
        Raises: AssemblyError when missing; ValueError when the row numbers are not contiguous.
    """
    path = L.feature_pack_path(task_id)
    if not path.is_file():
        raise L.AssemblyError(
            f"缺装配产物 {repo_relative(path)}：先跑 `python scripts/03b_assemble.py`（B-1）"
        )
    with np.load(path, allow_pickle=False) as payload:
        pack = {name: payload[name] for name in payload.files}
    row_id = pack["row_id"]
    if not np.array_equal(row_id, np.arange(len(row_id), dtype=row_id.dtype)):
        raise ValueError(
            f"{path.name} 的 row_id 不是 0..{len(row_id) - 1} 连续：文件被拼过或截过，不能当行序用"
        )
    pack["meta"] = json.loads(str(pack["meta"]))
    return pack


# ---- 4. 标准化：只在 train 上 fit / Standardisation: fit on train only ----


def standardize_fit(X_train: np.ndarray) -> dict[str, np.ndarray]:
    """在**训练集**上算均值与标准差，返回可直接用于 `apply_standardizer()` 的参数。
    Compute mean and scale on the **training set** and return parameters for `apply_standardizer()`.

    中文：
        用途：验收④要求"特征矩阵里不含任何用全量数据算的统计量"。于是本函数只接受训练集矩阵，
              并在函数名里就写明 `X_train`——`tests/test_no_leakage.py` 的扫描器按"调用表达式里
              有没有 train 标记"判红黑，名字骗不过它也骗不过读代码的人。
              **只对描述符块标准化**：2048 个指纹位是 0/1，缩放会把它们变成没有语义的连续量。
        输入：X_train —— 训练集矩阵（`stack()` 的输出切片），shape (n_train, 2253)。
        输出：dict，`mean` / `scale` / `n_train` / `n_all_nan_columns`；常数列的 `scale` 置 1
              （避免除零），整列全 NaN 的列 `mean` 记 0 且该列表现在 `n_all_nan_columns` 里。
        数据来源：调用方。配置来源：无。副作用：无。异常：ValueError —— 形状不像 `stack()` 的输出。

    English:
        Purpose: criterion 4 forbids full-data statistics in the feature matrix. This function
        therefore accepts only the training matrix, and says `X_train` in its name -
        tests/test_no_leakage.py decides red/green from whether the call expression mentions a
        training marker, and a name fools neither the scanner nor a reader.
        **Only the descriptor block is standardised**: the 2048 fingerprint bits are 0/1 and
        scaling turns them into meaningless continuous values.
        Args: X_train - a slice of `stack()`'s output, shape (n_train, 2253).
        Returns: a dict with `mean` / `scale` / `n_train`; constant columns get scale 1.
        Data source: the caller. Side effects: none. Raises: ValueError on an unexpected shape.
    """
    from chainworth.chem import descriptors as D

    matrix = np.asarray(X_train, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[1] != len(descriptor_columns()) + D.FINGERPRINT_BITS:
        raise ValueError(
            f"X_train 的形状 {matrix.shape} 不像 stack() 的输出"
            f"（应为 (n, {len(descriptor_columns()) + D.FINGERPRINT_BITS})）"
        )
    n_descriptors = len(descriptor_columns())
    block = matrix[:, :n_descriptors]
    if len(block) == 0:
        raise ValueError("X_train 是空的：标准化需要至少一行训练样本")
    # NaN 不参与统计：`nanmean` 会跳过缺失，`nanstd` 同理；否则某一列有一个 NaN 就整列变 NaN。
    # 整列全 NaN 的列（实测有 8 个 `BCUT2D_*`，对每个含 `*` 的重复单元都算不出来）会让
    # `nanmean` 报 "Mean of empty slice"——那不是错，是数据本身如此，所以这里屏蔽警告并**记下来**，
    # 让下游能一眼看到"这 8 列没有信息"，而不是让警告在日志里刷 10 遍没人看。
    # NaN must not enter the statistics: `nanmean`/`nanstd` skip it, otherwise a single NaN turns
    # the whole column into NaN. Columns that are entirely NaN (8 `BCUT2D_*` ones measured here,
    # undefined for every `*`-containing repeat unit) make `nanmean` warn about an empty slice -
    # that is data, not an error, so the warning is silenced and the fact is **recorded**, letting
    # downstream see "these 8 columns carry no information" instead of a wall of warnings.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        mean = np.nanmean(block, axis=0)
        scale = np.nanstd(block, axis=0, ddof=0)
    all_nan = ~np.isfinite(mean)
    scale = np.where((scale == 0) | ~np.isfinite(scale), 1.0, scale)
    mean = np.where(all_nan, 0.0, mean)
    return {
        "mean": mean,
        "scale": scale,
        "n_train": np.asarray(len(block), dtype=np.int64),
        "n_all_nan_columns": np.asarray(int(all_nan.sum()), dtype=np.int64),
    }


def apply_standardizer(params: Mapping[str, np.ndarray], X: np.ndarray) -> np.ndarray:
    """用训练集统计量变换矩阵；指纹块原样保留，NaN 仍是 NaN。
    Transform a matrix with training statistics; the fingerprint block passes through and NaN
    stays NaN.

    中文：
        用途：训练集与测试集必须用**同一组**统计量，所以变换与拟合分开两个函数：拟合只能拿
              train，变换可以拿任何一侧。函数名里没有 `fit` 是刻意的——它不是拟合。
        输入：params —— `standardize_fit()` 的输出；X —— 要变换的矩阵。
        输出：同形状的 float32 矩阵。
        数据来源：调用方。配置来源：无。副作用：无（新数组）。异常：KeyError —— 参数缺键。

    English:
        Purpose: train and test must be transformed by the **same** statistics, so fitting and
        applying are separate functions: only fitting is restricted to train. The name carries no
        `fit` on purpose - it is not fitting.
        Args: params; X. Returns: a float32 matrix of the same shape.
        Data source: the caller. Side effects: none. Raises: KeyError on missing parameters.
    """
    matrix = np.asarray(X, dtype=np.float64).copy()
    n_descriptors = int(params["mean"].size)
    matrix[:, :n_descriptors] = (matrix[:, :n_descriptors] - params["mean"]) / params["scale"]
    return matrix.astype(np.float32)


# ---- 5. 落清单与校验 / Manifest and verification ----


def manifest_row(summary: Mapping[str, Any], pack_path: Path, *, built_at: str = "") -> dict[str, object]:
    """把一次装配压成清单的一行（数字全部来自本次结果，不手填）。
    Turn one assembly into a manifest row; every number comes from this run, none is typed by hand.

    中文：
        用途：分工书要求 `feature_manifest.csv` 写"列数、指纹位数、缺失数"。这里在此之上补了
              `split_sha256`（D 复核划分版本要用）、`pack_sha256`（文件身份）、`row_order_basis`
              （写着"行序凭什么"）——三者都是"别人重出来是不是同一份"的凭据。
        输入：summary；pack_path —— 落盘的 npz；built_at —— 沿用旧时间戳时传入。
        输出：dict，键见 `MANIFEST_COLUMNS`。
        数据来源：`summary` 与磁盘。配置来源：无。副作用：算一次文件哈希（读盘）。
        异常：OSError —— 文件不可读。

    English:
        Purpose: division B asks the manifest for column counts, fingerprint width and missing
        counts; on top of that this adds `split_sha256` (D needs it to pin the split version),
        `pack_sha256` (file identity) and `row_order_basis` (why the row order is what it is) -
        all three are evidence for "is your rebuild the same as mine".
        Args: summary; pack_path; built_at. Returns: a dict keyed by MANIFEST_COLUMNS.
        Data source: the summary and the disk. Side effects: one file hash. Raises: OSError.
    """
    return {
        "task_id": summary["task_id"],
        "benchmark": summary["benchmark"],
        "protocol": summary["protocol"],
        "n_rows": summary["n_rows"],
        "n_label_rows": summary["n_label_rows"],
        "n_unique_structures": summary["n_unique_structures"],
        "n_descriptors": summary["n_descriptors"],
        "n_fingerprint_bits": summary["n_fingerprint_bits"],
        "n_columns_total": summary["n_columns_total"],
        "n_nan_cells": summary["n_nan_cells"],
        "n_rows_without_structure": summary["n_rows_without_structure"],
        "n_structures_without_wpsmiles": summary["n_structures_without_wpsmiles"],
        "feature_stats": FEATURE_STATS_NONE,
        "row_order_basis": summary["row_order_basis"],
        "split_ref": summary["split_ref"],
        "split_sha256": summary["split_sha256"],
        "pack_file": repo_relative(pack_path),
        "pack_sha256": sha256_of(pack_path),
        "built_at": built_at or now_iso(),
    }


def write_manifests(
    rows: Sequence[Mapping[str, object]],
    unresolved: Sequence[Mapping[str, object]],
    *,
    partial: bool = False,
) -> tuple[Path, Path | None]:
    """写两份清单（正式名或 `*.partial.*`），返回写出的路径。
    Write both manifests (official names or `*.partial.*`) and return their paths.

    中文：
        用途：正式清单是证据，限量跑的清单是调试——`.gitignore` 只忽略 `data/manifests/*.partial.csv`，
              所以限量跑必须换名，否则 3 行会静静取代 10 行（D-12 踩过同一个坑）。
        输入：rows —— 每任务一行；unresolved —— 未解析行；partial —— 限量跑。
        输出：(清单路径, 未解析清单路径或 None)。
        数据来源：调用方。配置来源：无。副作用：写 1–2 个文件。异常：OSError。

    English:
        Purpose: the official manifest is evidence while a limited run's is debugging;
        .gitignore only ignores data/manifests/*.partial.csv, so a limited run must be renamed or
        three rows quietly replace ten (D-12 hit exactly this).
        Args: rows; unresolved; partial. Returns: (manifest path, unresolved path or None).
        Side effects: writes one or two files. Raises: OSError.
    """
    if partial:
        target = MANIFEST_PATH.with_name("task_features_manifest.partial.csv")
        write_csv_atomic(rows, target, MANIFEST_COLUMNS)
        if not unresolved:
            return target, None
        unsettled = UNRESOLVED_PATH.with_name("task_feature_unresolved.partial.csv")
        write_csv_atomic(unresolved, unsettled, UNRESOLVED_COLUMNS)
        return target, unsettled
    write_csv_atomic(rows, MANIFEST_PATH, MANIFEST_COLUMNS)
    if not unresolved:
        return MANIFEST_PATH, None
    write_csv_atomic(unresolved, UNRESOLVED_PATH, UNRESOLVED_COLUMNS)
    return MANIFEST_PATH, UNRESOLVED_PATH


def verify_features(task_ids: Sequence[str] | None = None, *, sample: int = 100) -> list[str]:
    """把 B-1 的五条验收标准跑成问题清单（空 = 全过），`scripts` 与 `pytest` 共用这一份。
    Turn B-1's five acceptance criteria into a list of problems (empty means pass); shared by the
    stage entry and pytest.

    中文：
        用途：这是"人跑的"和"CI 跑的"必须同一份的实现。逐条对应分工书：
              ① 行数 = 调和后可用标签数，且行序由 split 决定（`row_id` 连续、`X.index` 与
                 `test_index` 逐位相同、三侧合计 = n_rows）；
              ② 随机抽 `sample` 个结构，特征向量与 A-6 的描述符矩阵**逐位相等**（np.array_equal，
                 不是 allclose——allclose 会放过 float32 的截断）；
              ③ 同一结构两次构图 `edge_index` 完全相同（在 `graph.py` 的测试里，这里只核数据侧）；
              ④ 矩阵里没有全量统计量（两条独立证据：清单 `feature_stats` 必须是 `none`；
                 **整矩阵**必须与 A-6 的原始描述符/指纹逐位相等——标准化或插补都会改变数值）；
              ⑤ 未解析的行逐条列在清单里，且**行未被删除**（n_rows 与调和表一致）。
        输入：task_ids —— 要核的任务（None = 盘上全部 npz）；sample —— 抽多少个结构核特征值。
        输出：问题字符串列表；空列表表示通过。
        数据来源：`data/processed/features/*.npz`、`data/manifests/**`、`splits/**`、A-6 产物。
        配置来源：`configs/tasks/*.yaml`（经 `loaders`）。
        副作用：只读。异常：无（收集问题而非抛出）。

    English:
        Purpose: the single implementation shared by the human run and CI. Criterion by criterion:
        (1) row count equals the usable-label count and the order comes from the split
        (`row_id` contiguous, `X.index` bit-identical to `test_index`, three sides summing to
        n_rows); (2) a random `sample` of structures has feature vectors **bit-identical** to A-6's
        descriptor matrix (`np.array_equal`, not `allclose`, which forgives float32 truncation);
        (3) two graph builds of one structure give an identical `edge_index` (covered in graph.py's
        tests; here only the data side); (4) no full-data statistics in the matrix, evidenced twice
        - the manifest's `feature_stats` must read `none`, and the **whole matrix** must remain
        bit-identical to A-6's raw values, which no standardisation or imputation survives;
        (5) unresolved rows are listed one by one and were **not dropped** (n_rows matches the
        harmonised table).
        Args: task_ids (None = every npz on disk); sample. Returns: problem strings.
        Data source: data/processed/features/*.npz, data/manifests/**, splits/**, A-6 artefacts.
        Side effects: read-only. Raises: none (collects instead of raising).
    """
    problems: list[str] = []
    on_disk = sorted(path.stem for path in FEATURES_DIR.glob("*.npz"))
    tasks = list(task_ids) if task_ids else on_disk
    if not tasks:
        return [f"{repo_relative(FEATURES_DIR)} 下没有任何 .npz：先跑 scripts/03b_assemble.py"]
    manifest = {row.get("task_id", ""): row for row in read_manifest_rows(MANIFEST_PATH)}
    unresolved_rows = read_manifest_rows(UNRESOLVED_PATH)
    rng = np.random.default_rng(0)

    for task_id in tasks:
        record = manifest.get(task_id)
        if record is None:
            problems.append(f"{task_id}：清单里没有这一行（03b 没跑全，或跑了 .partial）")
            continue
        pack_path = REPO_ROOT / str(record.get("pack_file", ""))
        if not pack_path.is_file():
            problems.append(f"{task_id}：清单指向的 {record.get('pack_file')} 不存在")
            continue
        if sha256_of(pack_path) != record.get("pack_sha256"):
            problems.append(f"{task_id}：{record.get('pack_file')} 的 sha256 与清单不符")
        try:
            pack = load_pack(task_id)
        except (L.AssemblyError, ValueError) as error:
            problems.append(f"{task_id}：{error}")
            continue

        rows = L.load_rows(task_id)
        split = L.load_split(task_id)
        n_rows = len(rows)

        # ① 行数与行序 / row count and row order
        if len(pack["row_id"]) != n_rows:
            problems.append(
                f"{task_id}：装配 {len(pack['row_id'])} 行 ≠ 调和表 {n_rows} 行（有行被删或加了）"
            )
        if list(pack["psmiles"]) != list(rows["psmiles_canonical"].fillna("").astype(str)):
            problems.append(f"{task_id}：装配的结构键序列与调和表行序不一致（行被重排过）")
        for fold in split.folds:
            index = fold.test_index
            if index.size and pack["row_id"][index].tolist() != index.tolist():
                problems.append(
                    f"{task_id}/{fold.name}：X 的第 k 行不是 row_id=k——按位置取测试集会取到别的样品"
                )
        # 标签也要逐位对上：装配时若换了列，指标会整体漂移而不会报错。
        # The labels must match bit for bit too; a swapped column shifts every metric silently.
        expected_y = rows["label"].to_numpy(dtype="float64")
        if not np.array_equal(pack["y"], expected_y, equal_nan=True):
            problems.append(f"{task_id}：装配的标签与调和表 `target_column` 不一致")

        # ② 抽 100 个结构与 A-6 的表逐位比 / sample 100 structures, compare bit for bit
        table = L.load_descriptor_table(task_id)
        columns = list(pack["desc_columns"])
        bits = list(pack["fp_columns"])
        keys = [key for key in pack["psmiles"] if key]
        if keys:
            picks = rng.choice(len(keys), size=min(sample, len(keys)), replace=False)
            positions = np.flatnonzero(pack["psmiles"] != "")
            chosen = positions[np.sort(picks)]
            source = table.reindex([pack["psmiles"][position] for position in chosen])
            if not np.array_equal(
                source[columns].to_numpy(dtype="float64"), pack["desc"][chosen], equal_nan=True
            ):
                problems.append(
                    f"{task_id}：抽 {len(chosen)} 个结构的描述符与 A-6 的表不逐位相等"
                )
            if not np.array_equal(
                source[bits].to_numpy(dtype="uint8"), pack["fp"][chosen]
            ):
                problems.append(f"{task_id}：抽 {len(chosen)} 个结构的指纹与 A-6 的表不逐位相等")

        # ③ 同一结构出现多次时必须给同一组特征（跨行一致性，与图无关的那一半）
        # (3) one structure appearing several times must get one feature vector (the non-graph half)
        # 用 numpy 数频次而不用 `pd.Series(...).value_counts()`：本模块已不再依赖 pandas
        # （A4 环境没有它）。排序按频次降序、并列时按字典序，抽检集合与原来的行为等价。
        # numpy instead of pandas: this module no longer depends on it, and the A4 environment has
        # no pandas. Frequency-descending order keeps the sampled set equivalent to before.
        values, counts = np.unique(np.asarray(pack["psmiles"]).astype(str), return_counts=True)
        order = np.argsort(-counts, kind="stable")
        repeated = [
            str(values[i]) for i in order if values[i] and int(counts[i]) > 1
        ]
        for key in repeated[:20]:
            index = np.flatnonzero(pack["psmiles"] == key)
            block = pack["desc"][index]
            # 必须把首行**广播成整块的形状**再比：`np.array_equal(block, block[:1])` 在块有
            # 两行以上时是形状不等（(k,205) vs (1,205)），会假报"不一致"——PolyBench 的 10 个
            # 任务每任务内已去重，这个分支从没被触发过；OpenPoly 不去重（D-26）才把它暴露出来。
            # The first row must be **broadcast to the block's shape**: comparing `block` with
            # `block[:1]` is a shape mismatch for k >= 2 ((k,205) vs (1,205)) and reports a false
            # inconsistency. PolyBench dedupes inside every task, so this branch never ran; OpenPoly
            # deliberately does not dedupe (D-26) and surfaced it.
            expected = np.broadcast_to(block[0], block.shape)
            if not np.array_equal(block, expected, equal_nan=True):
                problems.append(f"{task_id}：结构 {key[:24]}… 的多行特征不一致")
                break

        # ④ 没有全量统计量：两条独立的证据。
        # （a）清单自报 `feature_stats=none`——这是"作者声明"，可以撒谎；
        # （b）**整矩阵**与 A-6 的原始值逐位相等——这条不能撒谎：任何标准化/插补都会改变数值。
        # 用 `equal_nan=True` 是必须的：NaN 是"该描述符对这个结构无定义"，不是缺失值待填。
        # (4) No full-data statistics, evidenced twice: (a) the manifest self-reports
        # `feature_stats=none`, which is a claim and could lie; (b) the **whole matrix** is
        # bit-identical to A-6's raw values, which cannot lie - any standardisation or imputation
        # changes the numbers. `equal_nan=True` is required: NaN means "this descriptor is
        # undefined for this structure", not a missing value waiting to be filled.
        if str(record.get("feature_stats", "")).strip() != FEATURE_STATS_NONE:
            problems.append(
                f"{task_id}：清单里 feature_stats 不是 `none`（{record.get('feature_stats')!r}）——"
                "特征矩阵里可能混进了全量统计量"
            )
        lookup = [key if key else "__missing__" for key in pack["psmiles"]]
        whole = table.reindex(lookup)
        if not np.array_equal(
            whole[columns].to_numpy(dtype="float64"), pack["desc"], equal_nan=True
        ):
            problems.append(
                f"{task_id}：整矩阵与 A-6 的原始描述符不逐位相等——要么被标准化/插补过，"
                "要么落盘时改过数值"
            )
        if not np.array_equal(whole[bits].to_numpy(dtype="uint8"), pack["fp"]):
            problems.append(f"{task_id}：整矩阵的指纹与 A-6 的原始指纹不逐位相等")

        # ⑤ 未解析的行逐条在册，且没有被删 / unresolved rows are listed and not dropped
        listed = [row for row in unresolved_rows if row.get("task_id") == task_id]
        if len(listed) != int(record.get("n_rows_without_structure", -1)):
            problems.append(
                f"{task_id}：未解析清单 {len(listed)} 条 ≠ 汇总的 "
                f"{record.get('n_rows_without_structure')} 条"
            )
        for row in listed:
            position = int(str(row.get("row_id", "-1")))
            if 0 <= position < n_rows and pack["psmiles"][position] != "":
                problems.append(
                    f"{task_id}：第 {position} 行被记成未解析，但装配里它有结构键"
                )
                break
    return problems


def nan_report(summary: Mapping[str, Any], *, top: int = 5) -> list[tuple[str, int]]:
    """列出 NaN 最多的几个描述符列（写进日志，方便判断是"天然无定义"还是"算错了"）。
    List the descriptors with the most NaN cells, for the log.

    中文：
        用途：A-6 报告过"约 8 列对某些结构天然 NaN"。这条信息在装配层被放大成 20,660 行之后，
              更需要能一眼看到是哪几列——否则"3.9% 的 NaN"永远是个不解释的数字。
        输入：summary —— 装配摘要；top —— 列几个。
        输出：[(列名, NaN 单元数)]，按数量降序。
        数据来源：`summary["desc"]`。配置来源：无。副作用：无。异常：无。

    English:
        Purpose: A-6 reported "about 8 columns are naturally NaN for some structures". At 20,660
        rows that number needs a name, or "3.9% NaN" stays an unexplained figure.
        Args: summary; top. Returns: [(column, NaN cells)] in descending order.
    """
    desc = np.asarray(summary["desc"])
    counts = np.isnan(desc).sum(axis=0)
    order = np.argsort(-counts)[:top]
    return [(str(summary["desc_columns"][index]), int(counts[index])) for index in order]
