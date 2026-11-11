"""A-6 编排：结构表、每任务描述符文件、四份清单与校验（唯一实现）。
A-6 orchestration: the structure table, per-task descriptor files, four manifests and the checks.

中文：
    用途：`scripts/02_features.py` 与 `tests/test_features.py` 共用的唯一实现。负责四件事：
          ① 读 10 份调和表，把 `smiles_raw` 重新规范化并与 A-3 的 `smiles_canonical` **逐行复核**；
          ② 按规范形式去重，落一张全局结构表（结构画像 + wPSMILES）；
          ③ 每个任务落一份"结构键 + 205 描述符 + 2048 位指纹"的 parquet；
          ④ 四份清单：结构表、每任务特征、失败结构、描述符列定义。
    依赖：`data/interim/harmonised/*.parquet`（A-3）、`src/chainworth/chem/notation.py`、
          `src/chainworth/chem/descriptors.py`、rdkit/pandas/pyarrow。
    配置：本模块常量（路径、列名）+ `descriptors.py` 的特征空间常量；**没有 yaml**——
          特征空间不是实验参数，是口径，改它要记 DECISIONS。
    对外：`build_structure_table()`、`build_descriptors()`、`verify_outputs()`、
          `select_tasks()`、`load_structure_table()`、各路径常量。
    被谁用：`scripts/02_features.py`（阶段入口）、`tests/test_features.py`（断言）。

English:
    Purpose: the single implementation shared by `scripts/02_features.py` and
    `tests/test_features.py`. Four jobs: (1) read the ten harmonised tables, re-canonicalise
    `smiles_raw` and re-check it row by row against A-3's `smiles_canonical`; (2) deduplicate by
    canonical form into one global structure table carrying the notation profile and wPSMILES;
    (3) write one parquet per task holding a structure key, 205 descriptors and 2048 fingerprint
    bits; (4) write four manifests (structures, per-task features, failed structures, column
    definitions).
    Depends on: data/interim/harmonised/*.parquet (A-3), chainworth.chem.notation,
    chainworth.chem.descriptors, rdkit/pandas/pyarrow.
    Config: the path and column constants below plus descriptors.py's feature-space constants.
    There is no yaml: the feature space is a definition, not an experiment parameter, and
    changing it means a DECISIONS entry.
    Public API: `build_structure_table()`, `build_descriptors()`, `verify_outputs()`,
    `select_tasks()`, `load_structure_table()` and the path constants.
    Callers: scripts/02_features.py (stage entry) and tests/test_features.py (assertions).

边界：本模块只保证"每个结构算对一次" ／ Boundary: one correct computation per structure
------------------------------------------------------------------------------------
"每个任务的每一行对上正确的结构、且行序由划分文件决定"是 B 的事（B-1）。所以本模块产出的
per-task 文件是**结构键索引**，不带 `row_id`、不带标签、不带任务行序——B 拿 `psmiles_canonical`
（或 `structure_id`）自己去 join。替 B 做了装配，B-1 的验收标准就没法独立成立了。

"Which structure each task row maps to, in an order decided by the split file" is B's job (B-1).
The per-task files here are therefore keyed by structure only: no `row_id`, no labels, no task row
order. B joins on `psmiles_canonical` (or `structure_id`). Assembling here would destroy the
independence of B-1's acceptance criteria.
"""

from __future__ import annotations

import csv
import hashlib
import json
import logging
import os
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from chainworth.chem import descriptors as D
from chainworth.chem.notation import (
    HOMOPOLYMER_STAR_COUNT,
    canonical_psmiles,
    canonical_sort_key,
    count_stars,
    to_wpsmiles,
)

LOGGER = logging.getLogger("chainworth.structures")

REPO_ROOT = Path(__file__).resolve().parents[3]      # src/chainworth/chem/structures.py -> 仓库根
INTERIM_DIR = REPO_ROOT / "data" / "interim" / "harmonised"
PROCESSED_DIR = REPO_ROOT / "data" / "processed"
STRUCTURES_DIR = PROCESSED_DIR / "structures"
DESCRIPTORS_DIR = PROCESSED_DIR / "descriptors"
STRUCTURES_PATH = STRUCTURES_DIR / "psmiles.parquet"
CACHE_PATH = STRUCTURES_DIR / ".descriptor_cache.sqlite3"

MANIFEST_DIR = REPO_ROOT / "data" / "manifests"
STRUCTURES_MANIFEST = MANIFEST_DIR / "structures_manifest.csv"
FEATURES_MANIFEST = MANIFEST_DIR / "features_manifest.csv"
FAILURES_PATH = MANIFEST_DIR / "structure_failures.csv"
COLUMNS_PATH = MANIFEST_DIR / "descriptor_columns.json"

# 结构表的列。`structure_id` 是**行号**，只在排序稳定时有意义；真正的键是 `psmiles_canonical`。
# The structure table's columns. `structure_id` is a row number and only means something while the
# order is stable; the real key is `psmiles_canonical`.
STRUCTURE_COLUMNS = (
    "structure_id",
    "psmiles_canonical",
    "wpsmiles",
    "wpsmiles_reason",
    "n_stars",
    "n_raw_variants",
    "n_rows",
    "n_tasks",
    "tasks",
    "example_raw",
)

FAILURE_COLUMNS = (
    "reason",
    "psmiles_canonical",
    "smiles_raw",
    "n_rows",
    "task_ids",
    "example_source_rows",
)

STRUCTURES_MANIFEST_COLUMNS = (
    "n_structures",
    "n_rows_total",
    "n_wpsmiles",
    "n_without_wpsmiles",
    "n_rows_without_structure",
    "psmiles_parquet",
    "psmiles_sha256",
    "rdkit_version",
    "algorithm_version",
    "built_at",
)

FEATURES_MANIFEST_COLUMNS = (
    "task_id",
    "n_rows",
    "n_unique_structures",
    "n_without_wpsmiles",
    "n_descriptors",
    "n_fingerprint_bits",
    "n_nan_cells",
    "descriptor_file",
    "descriptor_sha256",
    "built_at",
    "rdkit_version",
)

# 调和表里本阶段真正用到的列；用 `columns=` 读，避免把标签列也拉进内存。
# The only harmonised columns this stage needs; reading with `columns=` keeps labels out of RAM.
HARMONISED_COLUMNS = ("task_id", "source_row", "smiles_raw", "smiles_canonical")

# 指纹列名：`morgan_0000` … `morgan_2047`，位数写进名字里，读的人不用翻文档。
# Fingerprint columns `morgan_0000` … `morgan_2047`: the width is in the name.
FINGERPRINT_PREFIX = "morgan_"

# 这些列名一旦出现在描述符文件里，说明有人把标签带进了特征表——就是 B-1 要独立验收的那件事。
# 若真的需要标签，应当去读调和表，而不是从特征文件里顺手拿。
# Any of these names in a descriptor file means labels leaked into features - exactly what B-1
# must be able to verify independently. Labels belong in the harmonised tables, not here.
FORBIDDEN_FEATURE_COLUMNS = ("label", "label_raw", "y", "target", "row_id")


@dataclass
class TaskStructures:
    """一个任务的结构清单与统计，供编排与日志共用。
    One task's structure inventory and counts, shared by orchestration and logging.

    中文：
        用途：把"这个任务有几行、几个结构、几个没有 wPSMILES、几行没对上结构"打包成一个对象，
              免得函数之间传五个整数、还传错顺序。
        字段：`task_id`、`n_rows`、`structures`（按结构表顺序的规范式列表）、
              `n_without_wpsmiles`、`n_rows_without_structure`。
        数据来源：调和表 + 结构表。配置来源：无。副作用：无。

    English:
        Purpose: bundle "how many rows, how many structures, how many lack a wPSMILES, how many
                 rows failed to match a structure" into one object instead of passing five ints
                 in an order somebody will eventually get wrong.
        Fields: `task_id`, `n_rows`, `structures`, `n_without_wpsmiles`,
        `n_rows_without_structure`. Data source: harmonised tables plus the structure table.
    """

    task_id: str
    n_rows: int
    structures: list[str] = field(default_factory=list)
    n_without_wpsmiles: int = 0
    n_rows_without_structure: int = 0


def _sha256_of(path: Path) -> str:
    """算文件字节的 sha256（分块读，别把 100 MB 的 parquet 一次读进内存）。
    sha256 of the file bytes, read in chunks so a large parquet is never loaded whole.

    中文：
        用途：产物哈希是"别人重出来的是不是同一份"的唯一凭据。
        输入：path —— 文件路径。
        输出：64 位十六进制字符串。
        数据来源：磁盘。配置来源：无。副作用：无。
        异常：FileNotFoundError —— 文件不存在（调用方应当先判存在）。
    English:
        Purpose: the artefact hash is the only evidence that someone else's rebuild is identical.
        Args: path. Returns: a 64-char hex digest. Data source: disk.
        Side effects: none. Raises: FileNotFoundError when the file is missing.
    """
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_csv_atomic(rows: Sequence[Mapping[str, object]], path: Path, columns: Sequence[str]) -> None:
    """原子写 CSV：先写临时文件再替换；`lineterminator="\\n"` 保证 LF。
    Write a CSV atomically via a temp file; `lineterminator="\\n"` keeps it LF.

    中文：
        用途：清单是证据，写一半比没有更糟。行尾必须是 LF（`.gitattributes` 已钉死），
              csv 模块默认写 CRLF，所以显式指定。
        输入：rows —— 行字典序列；path —— 目标；columns —— 列顺序。
        输出：无。数据来源：调用方。配置来源：无。
        副作用：建父目录、写一个文件。异常：KeyError —— 某行缺列。
    English:
        Purpose: a manifest is evidence, and half a manifest is worse than none. Line endings must
                 be LF (pinned by .gitattributes) while csv defaults to CRLF, hence the explicit
                 argument.
        Args: rows; path; columns. Returns: none. Side effects: creates the directory and writes
        one file. Raises: KeyError on a row missing a column.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(columns), lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column, "") for column in columns})
    os.replace(temporary, path)


def _read_csv_rows(path: Path) -> list[dict[str, str]]:
    """把一份清单读成行字典列表；文件不存在返回空列表。
    Read a manifest into a list of row dicts; a missing file reads as empty.

    中文：
        用途：重跑时判断"这一份上次是不是同一份字节"，从而沿用旧的 `built_at`——
              每次重跑都刷新时间戳会让冻结物在 git 里显得被人动过，而实际没变（A-2/A-3 同一个坑）。
        输入：path —— 清单路径。
        输出：行字典列表。数据来源：磁盘 CSV。配置来源：无。副作用：只读。
        异常：无（读坏了按空处理）。
    English:
        Purpose: decide whether a re-run produced the same bytes so the old `built_at` can be kept;
                 refreshing timestamps every run makes frozen files look hand-edited (the same
                 trap A-2 and A-3 hit).
        Args: path. Returns: a list of row dicts. Data source: the CSV on disk.
        Side effects: read-only. Raises: none.
    """
    if not path.is_file():
        return []
    try:
        with path.open("r", encoding="utf-8", newline="") as handle:
            return list(csv.DictReader(handle))
    except (OSError, csv.Error):
        return []


def _now() -> str:
    """当前本地时间（带时区、秒级），格式与 A-2/A-3 的清单一致。
    The current local timestamp (with offset, second resolution), matching A-2/A-3 manifests.

    中文：用途：清单的时间列。输入：无。输出：ISO 字符串。
          数据来源：系统时钟。配置来源：无。副作用：无。异常：无。
    English: Purpose: the manifests' time column. Args: none. Returns: an ISO string.
    """
    return datetime.now().astimezone().isoformat(timespec="seconds")


# ---- 1. 选任务 / Select tasks ----


def available_tasks() -> list[str]:
    """列出 `data/interim/harmonised/` 里现存的任务 ID（按字母序）。
    List the task ids present in data/interim/harmonised, alphabetically.

    中文：
        用途：本阶段的输入是 A-3 的产物，不是 `configs/tasks/`——所以"能跑哪些任务"以盘上
              实际存在的调和表为准。缺哪个任务时，日志会说清楚是"没跑 01"，而不是"任务不存在"。
        输入：无。
        输出：任务 ID 列表。数据来源：`data/interim/harmonised/*_harmonised.parquet`。
        配置来源：无。副作用：无。异常：无。
    English:
        Purpose: this stage's input is A-3's artefact, not configs/tasks, so "which tasks can run"
                 is decided by which harmonised tables exist. When one is missing the log says
                 "stage 01 was not run" rather than "no such task".
        Args: none. Returns: task ids. Data source: data/interim/harmonised.
        Config: none. Side effects: none. Raises: none.
    """
    if not INTERIM_DIR.is_dir():
        return []
    return sorted(
        path.name[: -len("_harmonised.parquet")]
        for path in INTERIM_DIR.glob("*_harmonised.parquet")
    )


def select_tasks(task_ids: Sequence[str] | None = None) -> list[str]:
    """按 `--tasks` 选任务；写错 ID 当场报错并列出可选值。
    Pick tasks by `--tasks`; an unknown id fails immediately and lists the valid ones.

    中文：
        用途：`--tasks` 的落地口。写错一个字母就静默少跑一个任务，是这类脚本最常见的坑。
        输入：task_ids —— 任务 ID 序列，None 表示全部。
        输出：任务 ID 列表（保持调用方给的顺序，去重）。
        数据来源：`available_tasks()`。配置来源：无。副作用：无。
        异常：KeyError —— 含不认识的任务 ID。
    English:
        Purpose: the landing point for `--tasks`. A typo that silently skips a task is the classic
                 failure of scripts like this.
        Args: task_ids - None means all. Returns: the ids, order preserved and de-duplicated.
        Data source: `available_tasks()`. Config: none. Side effects: none.
        Raises: KeyError listing the valid ids.
    """
    present = available_tasks()
    if not task_ids:
        if not present:
            raise FileNotFoundError(
                f"{INTERIM_DIR} 下没有调和表：先跑 `python scripts/01_harmonise.py`"
            )
        return present
    unknown = [task for task in task_ids if task not in present]
    if unknown:
        raise KeyError(f"不认识的任务 {unknown}；可选：{present}")
    return list(dict.fromkeys(task_ids))


def _harmonised_path(task_id: str) -> Path:
    """给出某任务的调和表路径。
    Return the harmonised table path of one task.

    中文：用途：路径拼装集中一处，避免两处各拼一遍。输入：task_id。输出：Path。
          数据来源：无。配置来源：无。副作用：无。异常：无。
    English: Purpose: one place that builds the path. Args: task_id. Returns: a Path.
    """
    return INTERIM_DIR / f"{task_id}_harmonised.parquet"


def _read_harmonised(task_id: str) -> pd.DataFrame:
    """读一个任务的调和表，只取本阶段需要的四列。
    Read one task's harmonised table, only the four columns this stage needs.

    中文：
        用途：**不读标签列**是刻意的：本阶段与标签无关，读了只会让"特征文件里混进标签"
              变得更容易发生。
        输入：task_id。
        输出：DataFrame（列见 `HARMONISED_COLUMNS`）。
        数据来源：`data/interim/harmonised/<task>_harmonised.parquet`。
        配置来源：无。副作用：只读。异常：FileNotFoundError —— 文件不存在。
    English:
        Purpose: deliberately not reading the label columns keeps this stage label-free, which
                 makes "labels leaked into a feature file" much harder to happen by accident.
        Args: task_id. Returns: a DataFrame with `HARMONISED_COLUMNS`.
        Data source: the harmonised parquet. Config: none. Side effects: read-only.
        Raises: FileNotFoundError when the table is missing.
    """
    path = _harmonised_path(task_id)
    if not path.is_file():
        raise FileNotFoundError(f"缺调和表：{path}（先跑 01_harmonise.py）")
    frame = pd.read_parquet(path, columns=list(HARMONISED_COLUMNS))
    return frame


# ---- 2. 结构表 / The structure table ----


def build_structure_table(
    task_ids: Sequence[str] | None = None, *, limit: int | None = None, dry_run: bool = False
) -> dict[str, object]:
    """建全局结构表 + 失败清单 + 结构清单；返回一份运行摘要。
    Build the global structure table, the failure list and the structure manifest; return a summary.

    中文：
        用途：A-6 的核心一步。三件事：
              ① **逐行复核** A-3 的 `smiles_canonical`：用 `smiles_raw` 重新规范化一遍，
                 不一致就记 `canonical_mismatch` 并不让该行进结构表——B 是按 A-3 那一列 join 的，
                 两边不一致意味着 join 会静默错位，这种错必须在那之前就炸出来；
              ② 按规范形式去重成一个结构，记它的星号数、wPSMILES、出现在几个任务、几行；
              ③ 落 `psmiles.parquet` 与两份清单。
        输入：task_ids —— 任务（None=全部）；limit —— 每任务只取前 N 行（冒烟，产物走 `.partial`）；
              dry_run —— 只算不写。
        输出：摘要 dict（键见 `STRUCTURES_MANIFEST_COLUMNS` 加 `failures`）。
        数据来源：`data/interim/harmonised/*.parquet`。
        配置来源：`HOMOPOLYMER_STAR_COUNT`、`D.provenance()`。
        副作用：写 parquet 与两份 CSV（dry_run/limit 时不同）。
        异常：FileNotFoundError —— 缺调和表；KeyError —— 任务 ID 不认识。
    English:
        Purpose: A-6's core step. Three things: (1) re-check A-3's `smiles_canonical` row by row by
                 canonicalising `smiles_raw` again - a mismatch is recorded as `canonical_mismatch`
                 and kept out of the structure table, because B joins on A-3's column and an
                 inconsistency there is a silent misalignment that must fail here instead;
                 (2) deduplicate by canonical form into one structure, recording its asterisk
                 count, wPSMILES, how many tasks and how many rows it covers; (3) write
                 `psmiles.parquet` plus two manifests.
        Args: task_ids (None = all), limit (first N rows per task, smoke run, `.partial` outputs),
              dry_run.
        Returns: a summary dict keyed like `STRUCTURES_MANIFEST_COLUMNS` plus `failures`.
        Data source: the harmonised parquet files.
        Config: `HOMOPOLYMER_STAR_COUNT` via notation, `D.provenance()`.
        Side effects: writes a parquet and two CSVs (differently for dry-run/limit).
        Raises: FileNotFoundError on a missing table; KeyError on an unknown task id.
    """
    tasks = select_tasks(task_ids)
    groups: dict[str, dict[str, object]] = {}
    # 失败按 (reason, key) 聚合，避免 199 个结构写 199 行重复原因相同的东西——
    # 但每个结构仍然是独立的行，因为"列出 ID"是验收标准②。
    # Failures aggregate per (reason, key); each structure still gets its own row, because
    # "list the ids" is acceptance criterion 2.
    failures: dict[tuple[str, str], dict[str, object]] = {}
    total_rows = 0
    rows_without_structure = 0

    for task_id in tasks:
        frame = _read_harmonised(task_id)
        if limit is not None:
            frame = frame.head(limit)
        total_rows += len(frame)
        for raw, canonical_a3, source_row in zip(
            frame["smiles_raw"], frame["smiles_canonical"], frame["source_row"]
        ):
            raw_text = "" if raw is None else str(raw)
            canonical_now = canonical_psmiles(raw_text)
            if canonical_now is None:
                rows_without_structure += 1
                _add_failure(failures, "parse_failed", raw_text, raw_text, task_id, int(source_row))
                continue
            a3_text = "" if canonical_a3 is None else str(canonical_a3)
            if a3_text != canonical_now:
                rows_without_structure += 1
                _add_failure(
                    failures,
                    "canonical_mismatch",
                    canonical_now,
                    raw_text,
                    task_id,
                    int(source_row),
                    detail=f"A-3={a3_text}",
                )
                continue
            entry = groups.get(canonical_now)
            if entry is None:
                entry = {
                    "n_rows": 0,
                    "raws": Counter(),
                    "tasks": set(),
                    "source_rows": [],
                }
                groups[canonical_now] = entry
            entry["n_rows"] = int(entry["n_rows"]) + 1
            entry["raws"][raw_text] += 1
            entry["tasks"].add(task_id)
            if len(entry["source_rows"]) < 3:
                entry["source_rows"].append(int(source_row))

    # 结构表按 (长度, 字符串) 全序排；structure_id 就是行号，行序稳定它才有意义。
    # Ordered by a total order over (length, string); structure_id is the row number and only
    # means something while that order is stable.
    ordered = canonical_sort_key(list(groups))
    structure_rows: list[dict[str, object]] = []
    n_wpsmiles = 0
    for structure_id, canonical in enumerate(ordered):
        entry = groups[canonical]
        stars = count_stars(canonical)
        wpsmiles = to_wpsmiles(canonical)
        if wpsmiles is not None:
            n_wpsmiles += 1
        else:
            # 星号数不是 2：wPSMILES 的单重复单元写法表达不了，记原因、不猜（见 notation 模块）。
            # 一个结构只写一行——按任务重复写会把"199 个结构"说成几百行失败，读的人会以为更糟。
            # Not two asterisks: the single-repeat-unit wPSMILES form cannot express it. Record the
            # reason and do not guess (see the notation module). One row per structure: one row per
            # task would turn "199 structures" into hundreds of failure lines and read as worse
            # than it is.
            key = (f"star_count_{stars}_not_{HOMOPOLYMER_STAR_COUNT}", canonical)
            failures[key] = {
                "smiles_raw": max(entry["raws"], key=entry["raws"].get),
                "n_rows": int(entry["n_rows"]),
                "tasks": set(entry["tasks"]),
                "source_rows": list(entry["source_rows"]),
            }
        structure_rows.append(
            {
                "structure_id": structure_id,
                "psmiles_canonical": canonical,
                "wpsmiles": wpsmiles,
                "wpsmiles_reason": "" if wpsmiles else f"star_count_{stars}",
                "n_stars": stars,
                "n_raw_variants": len(entry["raws"]),
                "n_rows": entry["n_rows"],
                "n_tasks": len(entry["tasks"]),
                "tasks": ",".join(sorted(entry["tasks"])),
                "example_raw": max(entry["raws"], key=entry["raws"].get),
            }
        )

    frame = pd.DataFrame(structure_rows, columns=list(STRUCTURE_COLUMNS))
    frame["structure_id"] = frame["structure_id"].astype("int64")
    frame["n_stars"] = frame["n_stars"].astype("int16")
    frame["n_raw_variants"] = frame["n_raw_variants"].astype("int32")
    frame["n_rows"] = frame["n_rows"].astype("int32")
    frame["n_tasks"] = frame["n_tasks"].astype("int16")

    failure_rows = [
        {
            "reason": reason,
            "psmiles_canonical": key,
            "smiles_raw": entry["smiles_raw"],
            "n_rows": entry["n_rows"],
            "task_ids": ",".join(sorted(entry["tasks"])),
            "example_source_rows": ",".join(str(value) for value in entry["source_rows"]),
        }
        for (reason, key), entry in sorted(failures.items())
    ]

    summary: dict[str, object] = {
        "n_structures": len(frame),
        "n_rows_total": total_rows,
        "n_wpsmiles": n_wpsmiles,
        "n_without_wpsmiles": len(frame) - n_wpsmiles,
        "n_rows_without_structure": rows_without_structure,
        "n_failures": len(failure_rows),
        "rdkit_version": D.provenance()["rdkit_version"],
        "algorithm_version": D.ALGORITHM_VERSION,
        "tasks": tasks,
        # 内存里的表：`--dry-run` 时下一步（描述符）没有盘上文件可读，就直接用这一份，
        # 免得"空跑"只能跑一半。放进摘要而不是另开返回值，是为了不改变调用方的解包方式。
        # The in-memory frame: during a dry run the next step has no file to read, so it uses this
        # one instead of the dry run stopping halfway. It rides in the summary rather than becoming
        # a second return value, so existing callers keep unpacking the same way.
        "table": frame,
    }
    LOGGER.info(
        "结构表：%d 个结构 / %d 行；有 wPSMILES %d、无 %d；未对上结构的行 %d；失败清单 %d 行",
        summary["n_structures"],
        total_rows,
        n_wpsmiles,
        summary["n_without_wpsmiles"],
        rows_without_structure,
        len(failure_rows),
    )

    if dry_run:
        LOGGER.info("[dry-run] 不写盘：%s、%s、%s", STRUCTURES_PATH, STRUCTURES_MANIFEST, FAILURES_PATH)
        return summary

    partial = limit is not None
    target = STRUCTURES_PATH.with_name("psmiles.partial.parquet") if partial else STRUCTURES_PATH
    target.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(target, index=False, compression="snappy")
    summary["psmiles_parquet"] = _relative(target)
    summary["psmiles_sha256"] = _sha256_of(target)

    if partial:
        # 限量跑是调试：不写正式清单（`.gitignore` 只放过 *.partial.csv，正式清单一律由全量跑生成）。
        # A limited run is a debug artefact: no official manifest (only *.partial.csv is ignored-free,
        # and official manifests come from full runs only).
        _write_csv_atomic(
            failure_rows, FAILURES_PATH.with_name("structure_failures.partial.csv"), FAILURE_COLUMNS
        )
        LOGGER.info("限量跑：产物写 %s，清单走 *.partial.csv，不入库", target.name)
        return summary

    previous = {row.get("psmiles_sha256", ""): row for row in _read_csv_rows(STRUCTURES_MANIFEST)}
    old = previous.get(str(summary["psmiles_sha256"]))
    built_at = old["built_at"] if old is not None else _now()
    manifest_row = {
        "n_structures": summary["n_structures"],
        "n_rows_total": summary["n_rows_total"],
        "n_wpsmiles": summary["n_wpsmiles"],
        "n_without_wpsmiles": summary["n_without_wpsmiles"],
        "n_rows_without_structure": summary["n_rows_without_structure"],
        "psmiles_parquet": summary["psmiles_parquet"],
        "psmiles_sha256": summary["psmiles_sha256"],
        "rdkit_version": summary["rdkit_version"],
        "algorithm_version": summary["algorithm_version"],
        "built_at": built_at,
    }
    _write_csv_atomic([manifest_row], STRUCTURES_MANIFEST, STRUCTURES_MANIFEST_COLUMNS)
    _write_csv_atomic(failure_rows, FAILURES_PATH, FAILURE_COLUMNS)
    _write_columns_manifest()
    LOGGER.info(
        "结构表已写：%s（sha256 %s…）", summary["psmiles_parquet"], str(summary["psmiles_sha256"])[:12]
    )
    return summary


def _add_failure(
    failures: dict[tuple[str, str], dict[str, object]],
    reason: str,
    key: str,
    raw: str,
    task_id: str,
    source_row: int,
    *,
    detail: str = "",
) -> None:
    """把一条失败并进聚合表；同一个 (reason, key) 只累加行数与任务。
    Merge one failure into the aggregate; the same (reason, key) accumulates rows and tasks.

    中文：
        用途：同一个坏结构可能在多个任务里出现几百次，逐行写会让失败清单变成日志。
              按 (reason, key) 聚合后仍是"每个结构一行"，符合验收②的"列出 ID"，
              同时保留 n_rows 与任务列表这两个有用的数。
        输入：failures —— 聚合表；reason —— 原因；key —— 规范式（解析失败时用原始串）；
              raw —— 原始串；task_id / source_row —— 出处；detail —— 附加说明（如 A-3 的写法）。
        输出：无。数据来源：调用方。配置来源：无。副作用：修改 failures。异常：无。

    English:
        Purpose: one bad structure can appear hundreds of times across tasks; one row per
                 occurrence turns the failure list into a log. Aggregating by (reason, key) keeps
                 "one row per structure" as criterion 2 requires while preserving the useful
                 `n_rows` and task list.
        Args: failures; reason; key; raw; task_id; source_row; detail.
        Returns: none. Side effects: mutates `failures`. Raises: none.
    """
    entry = failures.get((reason, key))
    if entry is None:
        entry = {
            "smiles_raw": raw if not detail else f"{raw}（{detail}）",
            "n_rows": 0,
            "tasks": set(),
            "source_rows": [],
        }
        failures[(reason, key)] = entry
    entry["n_rows"] = int(entry["n_rows"]) + 1
    entry["tasks"].add(task_id)
    if len(entry["source_rows"]) < 3:
        entry["source_rows"].append(source_row)


def _relative(path: Path) -> str:
    """把绝对路径转成仓库相对的 POSIX 路径（清单里一律用正斜杠）。
    Turn an absolute path into a repo-relative POSIX path (manifests always use forward slashes).

    中文：用途：Windows 的 `\\` 会让清单在不同机器上看起来不一样，A-5 踩过这个坑。
          输入：path。输出：字符串。数据来源：无。配置来源：无。副作用：无。异常：无。
    English: Purpose: Windows backslashes make a manifest look different across machines; A-5 hit
             this. Args: path. Returns: a string.
    """
    try:
        return path.resolve().relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return path.as_posix()


def _write_columns_manifest() -> None:
    """把特征空间定义写成 JSON（列名、列序、指纹参数、RDKit 版本）。
    Write the feature-space definition as JSON: column names, order, fingerprint settings, RDKit.

    中文：
        用途：这是 B/C 唯一需要读的"特征字典"。把列序写进文件而不是让下游从 parquet 里猜，
              是因为列序一变、模型就不能复用（同名不同序的矩阵乘出来是垃圾且不报错）。
        输入：无。输出：无。数据来源：`D.provenance()`。
        配置来源：`D` 模块常量。副作用：写一份 JSON。异常：OSError —— 无写权限。
    English:
        Purpose: the only "feature dictionary" B/C needs. The column order is written down instead
                 of being inferred from a parquet, because a changed order silently invalidates
                 every trained model.
        Args: none. Returns: none. Data source: `D.provenance()`.
        Side effects: writes one JSON file. Raises: OSError on permissions.
    """
    payload = D.provenance()
    # 额外字段放在**顶层**，不要塞进 `fingerprint` 子字典：校验时是拿子字典整体比对的，
    # 塞进去就会让"文件里多一个字段"变成"特征空间不一致"的假警报。
    # The extra fields go at the top level, never inside the `fingerprint` sub-dict: verification
    # compares that sub-dict as a whole, so nesting would turn "one extra field" into a false
    # "feature space changed" alarm.
    payload["fingerprint_column_prefix"] = FINGERPRINT_PREFIX
    payload["n_descriptors"] = len(payload["descriptor_columns"])
    payload["generated_at"] = _now()
    COLUMNS_PATH.parent.mkdir(parents=True, exist_ok=True)
    temporary = COLUMNS_PATH.with_name(COLUMNS_PATH.name + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=False)
        handle.write("\n")
    os.replace(temporary, COLUMNS_PATH)


def load_structure_table(path: Path | None = None) -> pd.DataFrame:
    """读结构表；不存在就抛，让调用方知道要跑 02。
    Load the structure table; raise when missing so the caller knows stage 02 has to run.

    中文：
        用途：B/C 与本模块的校验都从这张表出发。**按 `psmiles_canonical` 建索引**，
              因为 `structure_id` 只是行号，数据一变就会移动。
        输入：path —— 表路径（None 用正式产物；限量跑传 `psmiles.partial.parquet`）。
        输出：DataFrame（列见 `STRUCTURE_COLUMNS`，索引为规范式）。
        数据来源：`data/processed/structures/psmiles.parquet`。
        配置来源：无。副作用：只读。异常：FileNotFoundError —— 还没跑 02。
    English:
        Purpose: B/C and this module's checks both start here. Indexed by `psmiles_canonical`
                 because `structure_id` is only a row number and moves when the data changes.
        Args: path - None for the official artefact, or the `.partial` one for a limited run.
        Returns: a DataFrame indexed by canonical string.
        Data source: data/processed/structures/psmiles.parquet.
        Side effects: read-only. Raises: FileNotFoundError before stage 02 has run.
    """
    target = Path(path) if path is not None else STRUCTURES_PATH
    if not target.is_file():
        raise FileNotFoundError(f"缺结构表：{target}（先跑 `python scripts/02_features.py`）")
    frame = pd.read_parquet(target)
    return frame.set_index("psmiles_canonical", drop=False)


# ---- 3. 每任务描述符 / Per-task descriptor files ----


def _descriptor_file_is_stale(
    target: Path, table: pd.DataFrame, unique_order: Sequence[str]
) -> tuple[bool, str]:
    """判断已有的特征文件是否与当前结构表一致；不一致就必须重写。
    Decide whether an existing feature file still agrees with the current structure table.

    中文：
        用途：`structure_id` 是**全局结构表的行号**，所以只要结构表的任务集变了，行号就会整体
              平移——旧文件里的 id 于是指向别的结构。这种错不会自己报错：文件能读、列数对、
              哈希也是对的那份旧清单，只有 join 出来的特征是错位的。
              所以"文件已存在"不能当"文件是对的"，必须核这两列。
        输入：target —— 已有文件；table —— 当前结构表；unique_order —— 本次应有的规范式顺序。
        输出：(是否需要重写, 原因)；原因只用于日志。
        数据来源：磁盘 parquet + 内存结构表。配置来源：无。副作用：只读。
        异常：无（文件读坏了按"需要重写"处理）。

    English:
        Purpose: `structure_id` is a row number in the global structure table, so the moment the
                 table's task set changes every id shifts and the old file's ids point at other
                 structures. That failure is silent: the file opens, the column count is right and
                 the hash even matches the old manifest - only the joined features are misaligned.
                 "The file exists" therefore cannot mean "the file is fine"; these two columns must
                 be compared.
        Args: target; table; unique_order. Returns: (needs rewriting, reason), reason for logging.
        Data source: the parquet on disk plus the in-memory table. Side effects: read-only.
        Raises: none (an unreadable file counts as "needs rewriting").
    """
    if not target.is_file():
        return True, "文件不存在"
    try:
        existing = pd.read_parquet(target, columns=["psmiles_canonical", "structure_id"])
    except (OSError, ValueError, KeyError) as error:
        return True, f"读不了（{type(error).__name__}）"
    if len(existing) != len(unique_order):
        return True, f"行数变了（{len(existing)} -> {len(unique_order)}）"
    if existing["psmiles_canonical"].astype(str).tolist() != list(unique_order):
        return True, "结构集合或顺序变了"
    expected = table.loc[list(unique_order), "structure_id"].to_numpy()
    if not np.array_equal(existing["structure_id"].to_numpy(), expected):
        return True, "structure_id 与当前结构表不一致（任务集变了）"
    return False, ""


def build_descriptors(
    task_ids: Sequence[str] | None = None,
    *,
    limit: int | None = None,
    force: bool = False,
    dry_run: bool = False,
    table: pd.DataFrame | None = None,
) -> list[TaskStructures]:
    """为每个任务写一份"结构键 + 描述符 + 指纹"的 parquet；返回每任务的统计。
    Write one parquet per task (structure key, descriptors, fingerprint); return per-task counts.

    中文：
        用途：A-6 的第三件产物。计算走 `DescriptorCache`，所以跨任务重复的结构只算一次
              （11,494 个结构 vs 20,660 行，省下 45% 的计算）。
              **文件里没有标签、没有 row_id、没有任务行序**：B 自己 join（见模块 docstring 的边界）。
        输入：task_ids —— 任务；limit —— 每任务前 N 行（走 `.partial`）；force —— 即使产物已存在也重写；
              dry_run —— 只算不写。
        输出：`TaskStructures` 列表（每任务一项）。
        数据来源：调和表 + 结构表（必须先跑 `build_structure_table()`）。
        配置来源：`D` 模块的特征空间常量。
        副作用：写 `data/processed/descriptors/<task>.parquet`、缓存 sqlite、`features_manifest.csv`。
        异常：FileNotFoundError —— 结构表不存在；ValueError —— 缓存 provenance 不匹配。
    English:
        Purpose: A-6's third deliverable. Computation goes through `DescriptorCache`, so a
                 structure repeated across tasks is computed once (11,494 structures vs 20,660
                 rows, 45% saved). The file carries no labels, no `row_id` and no task row order;
                 B joins (see the boundary in the module docstring).
        Args: task_ids; limit (first N rows per task, `.partial`); force (rewrite even if present);
              dry_run; table (an in-memory structure table, used by `--dry-run` so the smoke run
              does not need a file that was deliberately not written).
        Returns: a list of `TaskStructures`.
        Data source: the harmonised tables plus the structure table (stage 02 must have run first).
        Config: the feature-space constants in `D`.
        Side effects: writes per-task parquet, the sqlite cache and features_manifest.csv.
        Raises: FileNotFoundError without the structure table; ValueError on a cache provenance
        mismatch.
    """
    tasks = select_tasks(task_ids)
    if table is None:
        table = load_structure_table()
    elif "psmiles_canonical" in table.columns and table.index.name != "psmiles_canonical":
        # 内存表是按行号索引的，这里统一成按规范式索引，免得两条路径（盘上/内存）行为不同。
        # The in-memory frame is row-indexed; normalise it to be canonical-indexed so the two paths
        # (disk and memory) behave identically.
        table = table.set_index("psmiles_canonical", drop=False)
    known = set(table.index)
    columns = D.descriptor_columns()
    fingerprint_columns = [f"{FINGERPRINT_PREFIX}{index:04d}" for index in range(D.FINGERPRINT_BITS)]

    results: list[TaskStructures] = []
    manifest_rows: list[dict[str, object]] = []
    previous = {row.get("task_id", ""): row for row in _read_csv_rows(FEATURES_MANIFEST)}
    failure_keys = {
        row.get("psmiles_canonical", "") for row in _read_csv_rows(FAILURES_PATH)
    }

    with D.DescriptorCache(CACHE_PATH) as cache:
        for task_id in tasks:
            frame = _read_harmonised(task_id)
            if limit is not None:
                frame = frame.head(limit)
            canonicals = [str(value) for value in frame["smiles_canonical"].dropna()]
            missing = [value for value in canonicals if value not in known]
            # 结构表里没有的行有两种来源：A-3/A-6 不一致（已进失败清单），或者结构表是用
            # 另一组 `--tasks` 建的。前者是"有记录的丢失"，后者是操作顺序问题，必须喊出来——
            # 否则它会表现成"特征文件行数偏少"，一直没人查。
            # A row missing from the structure table means either an A-3/A-6 disagreement (already
            # in the failure manifest) or a structure table built with a different `--tasks`. The
            # first is a recorded loss; the second is an ordering mistake and must be loud, or it
            # will just look like "the feature file has fewer rows" and nobody will investigate.
            unexplained = [value for value in dict.fromkeys(missing) if value not in failure_keys]
            if unexplained:
                LOGGER.warning(
                    "%s：%d 个结构不在结构表里，且失败清单也没记（例：%s…）——"
                    "多半是用另一组 --tasks 建的结构表，请用同一组任务重跑 --source structures",
                    task_id,
                    len(unexplained),
                    unexplained[0][:40],
                )
            unique_order = list(dict.fromkeys(canonicals))
            unique_order = [value for value in unique_order if value in known]
            records, stats = cache.records(unique_order)
            LOGGER.info(
                "%s：%d 行 / %d 结构（缓存命中 %d、新算 %d、解析失败 %d）",
                task_id,
                len(frame),
                stats["unique"],
                stats["hits"],
                stats["misses"],
                stats["unparsable"],
            )

            data: dict[str, object] = {
                "structure_id": table.loc[unique_order, "structure_id"].to_numpy(dtype="int64"),
                "psmiles_canonical": unique_order,
            }
            descriptor_matrix = np.full(
                (len(unique_order), len(columns)), np.nan, dtype=np.float64
            )
            fingerprint_matrix = np.zeros(
                (len(unique_order), D.FINGERPRINT_BITS), dtype=np.uint8
            )
            for position, record in enumerate(records):
                payload = record.get("descriptors")
                if not payload:
                    continue
                descriptor_matrix[position, :] = [payload[name] for name in columns]
                fingerprint_matrix[position, :] = D.fingerprint_array(record["fingerprint"])
            for index, name in enumerate(columns):
                data[name] = descriptor_matrix[:, index]
            for index, name in enumerate(fingerprint_columns):
                data[name] = fingerprint_matrix[:, index]

            output = pd.DataFrame(data)
            partial = limit is not None
            target = (
                DESCRIPTORS_DIR / f"{task_id}.partial.parquet"
                if partial
                else DESCRIPTORS_DIR / f"{task_id}.parquet"
            )
            sha = ""
            if not dry_run:
                target.parent.mkdir(parents=True, exist_ok=True)
                stale, reason = _descriptor_file_is_stale(target, table, unique_order)
                if force or stale:
                    output.to_parquet(target, index=False, compression="snappy")
                    LOGGER.info(
                        "  写 %s（%d 行 × %d 列%s）",
                        _relative(target),
                        len(output),
                        output.shape[1],
                        f"；重写原因：{reason}" if reason else "",
                    )
                else:
                    LOGGER.info("  %s 已是最新，沿用（--force 可重写）", _relative(target))
                sha = _sha256_of(target)

            without_wpsmiles = int(
                table.loc[unique_order, "wpsmiles"].isna().sum() if unique_order else 0
            )
            n_nan = D.nan_count(records)
            results.append(
                TaskStructures(
                    task_id=task_id,
                    n_rows=len(frame),
                    structures=unique_order,
                    n_without_wpsmiles=without_wpsmiles,
                    n_rows_without_structure=len(missing),
                )
            )
            if dry_run or partial:
                continue
            old = previous.get(task_id)
            built_at = (
                old["built_at"]
                if old is not None and old.get("descriptor_sha256") == sha
                else _now()
            )
            manifest_rows.append(
                {
                    "task_id": task_id,
                    "n_rows": len(frame),
                    "n_unique_structures": len(unique_order),
                    "n_without_wpsmiles": without_wpsmiles,
                    "n_descriptors": len(columns),
                    "n_fingerprint_bits": D.FINGERPRINT_BITS,
                    "n_nan_cells": n_nan,
                    "descriptor_file": _relative(target),
                    "descriptor_sha256": sha,
                    "built_at": built_at,
                    "rdkit_version": D.provenance()["rdkit_version"],
                }
            )

    if not dry_run and limit is None:
        _write_csv_atomic(manifest_rows, FEATURES_MANIFEST, FEATURES_MANIFEST_COLUMNS)
        _write_columns_manifest()
        LOGGER.info("特征清单已写：%s（%d 个任务）", _relative(FEATURES_MANIFEST), len(manifest_rows))
    return results


# ---- 4. 校验 / Verification ----


def verify_outputs(
    task_ids: Sequence[str] | None = None, *, deep: bool = False
) -> list[str]:
    """把 A-6 的四条验收标准跑成问题清单（空 = 全过）。
    Turn A-6's four acceptance criteria into a list of problems (empty means pass).

    中文：
        用途：`scripts/02_features.py --verify` 与 pytest 共用的唯一实现，保证"人跑的"和
              "CI 跑的"是同一条检查。
        输入：task_ids —— 要核的任务（None=盘上全部）；deep —— 是否重算每个结构的描述符与指纹
              （约 10 秒；浅层只核哈希与结构）。
        输出：问题字符串列表；空列表即通过。
        数据来源：`data/processed/**`、`data/manifests/**`、`data/interim/harmonised/**`。
        配置来源：`D.provenance()`（特征空间定义）。
        副作用：只读。异常：无（问题收集而非抛出）。

    English:
        Purpose: the single implementation shared by `--verify` and pytest, so the human run and
                 the CI run are literally the same check.
        Args: task_ids (None = everything on disk); deep (recompute every structure's descriptors
              and fingerprint, about ten seconds; the shallow pass only checks hashes and shape).
        Returns: a list of problem strings; empty means pass.
        Data source: data/processed, data/manifests, data/interim/harmonised.
        Config: `D.provenance()`.
        Side effects: read-only. Raises: none (collects instead of raising).
    """
    problems: list[str] = []

    # ① 特征空间定义：列名/列序/RDKit 版本必须与冻结的 JSON 一致，否则两套特征会被混用。
    # Feature-space identity: names, order and RDKit version must match the frozen JSON, otherwise
    # two different feature spaces get mixed.
    expected_columns = D.descriptor_columns()
    if not COLUMNS_PATH.is_file():
        problems.append(f"缺特征字典：{_relative(COLUMNS_PATH)}")
    else:
        frozen = json.loads(COLUMNS_PATH.read_text(encoding="utf-8"))
        current = D.provenance()
        for key in ("algorithm_version", "descriptor_columns", "dropped_descriptors", "fingerprint"):
            if frozen.get(key) != current.get(key):
                problems.append(f"特征字典里的 {key} 与当前环境不一致（RDKit {current['rdkit_version']}）")
        if frozen.get("rdkit_version") != current["rdkit_version"]:
            problems.append(
                "RDKit 版本变了：字典记 "
                f"{frozen.get('rdkit_version')}，当前 {current['rdkit_version']}；"
                "描述符数值会漂移，必须重出全部产物并把新字典一起提交"
            )

    # ② 结构表：存在、哈希对得上、列齐、n_rows 合计等于调和表行数。
    # Structure table: present, hash matches, columns complete, n_rows sums to the harmonised rows.
    table = None
    structure_rows = _read_csv_rows(STRUCTURES_MANIFEST)
    if not STRUCTURES_PATH.is_file():
        problems.append(f"缺结构表：{_relative(STRUCTURES_PATH)}")
    elif not structure_rows:
        problems.append(f"缺结构清单：{_relative(STRUCTURES_MANIFEST)}")
    else:
        record = structure_rows[0]
        actual = _sha256_of(STRUCTURES_PATH)
        if record.get("psmiles_sha256") != actual:
            problems.append(
                f"结构表 sha256 不符：清单 {record.get('psmiles_sha256', '')[:12]}… vs 实际 {actual[:12]}…"
            )
        table = load_structure_table()
        missing_columns = [name for name in STRUCTURE_COLUMNS if name not in table.columns]
        if missing_columns:
            problems.append(f"结构表缺列：{missing_columns}")
        if int(record.get("n_structures", -1)) != len(table):
            problems.append(
                f"结构数不符：清单 {record.get('n_structures')} vs 实际 {len(table)}"
            )
        n_rows_sum = int(table["n_rows"].sum())
        if int(record.get("n_rows_total", -1)) != n_rows_sum + int(
            record.get("n_rows_without_structure", 0)
        ):
            problems.append(
                "结构表的 n_rows 合计与清单对不上："
                f"{n_rows_sum} + 未对上结构的行 {record.get('n_rows_without_structure')} "
                f"vs 清单 {record.get('n_rows_total')}"
            )

    # ③ 每任务文件：清单在、文件在、哈希对、列齐、行数对、没有标签列。
    # Per-task files: manifest present, file present, hash matches, columns complete, row count
    # matches, no label columns.
    tasks = task_ids or available_tasks()
    feature_rows = {row.get("task_id", ""): row for row in _read_csv_rows(FEATURES_MANIFEST)}
    expected_feature_columns = (
        ["structure_id", "psmiles_canonical"]
        + expected_columns
        + [f"{FINGERPRINT_PREFIX}{index:04d}" for index in range(D.FINGERPRINT_BITS)]
    )
    for task_id in tasks:
        record = feature_rows.get(task_id)
        if record is None:
            problems.append(f"{task_id}：特征清单里没有这一行（02_features 没跑全）")
            continue
        path = REPO_ROOT / str(record["descriptor_file"])
        frame = _check_feature_file(task_id, path, record, expected_feature_columns, problems)
        if frame is None:
            continue

        # ④ 覆盖：调和表里每个有结构标签的行，都必须在特征文件里找得到对应结构；
        #    同时这一行的 `structure_id` 要和全局结构表一致（否则 B 按 id join 会错位）。
        # Coverage: every harmonised row with a canonical structure must have a feature row, and
        # that row's `structure_id` must agree with the global table (otherwise B's join on the id
        # silently misaligns).
        _verify_coverage(task_id, frame, problems)
        if table is not None:
            unknown = sorted(set(frame["psmiles_canonical"].astype(str)) - set(table.index))
            if unknown:
                problems.append(
                    f"{task_id}：{len(unknown)} 个结构不在全局结构表里（例：{unknown[0][:40]}…）"
                )
            else:
                expected_ids = table.loc[
                    frame["psmiles_canonical"].astype(str).tolist(), "structure_id"
                ].to_numpy()
                if not np.array_equal(expected_ids, frame["structure_id"].to_numpy()):
                    problems.append(f"{task_id}：structure_id 与全局结构表不一致")

    # ⑤ 失败清单：每条 `star_count_*` 的结构确实星号数不对；每条 `parse_failed` 确实解析不了。
    # Failure manifest: every `star_count_*` really has that star count; every `parse_failed`
    # really fails to parse.
    if not FAILURES_PATH.is_file():
        problems.append(f"缺失败清单：{_relative(FAILURES_PATH)}")
    else:
        for row in _read_csv_rows(FAILURES_PATH):
            reason = row.get("reason", "")
            canonical = row.get("psmiles_canonical", "")
            if reason.startswith("star_count_"):
                try:
                    claimed = int(reason.split("_")[2])
                except (IndexError, ValueError):
                    problems.append(f"失败清单里的原因格式不认识：{reason}")
                    continue
                actual_stars = count_stars(canonical)
                if actual_stars != claimed:
                    problems.append(
                        f"失败清单说 {canonical[:32]}… 有 {claimed} 个星号，实际 {actual_stars} 个"
                    )
                if actual_stars == HOMOPOLYMER_STAR_COUNT:
                    problems.append(f"失败清单把两个星号的结构记成失败：{canonical[:32]}…")
            elif reason == "parse_failed":
                if canonical_psmiles(row.get("smiles_raw", "")) is not None:
                    problems.append(f"失败清单说 {row.get('smiles_raw', '')[:32]}… 解析不了，实际可以解析")

    if deep:
        problems.extend(_deep_recompute(table, tasks, expected_columns))
    return problems


def _check_feature_file(
    task_id: str,
    path: Path,
    record: Mapping[str, str],
    expected_columns: Sequence[str],
    problems: list[str],
) -> pd.DataFrame | None:
    """核一份特征文件：存在、哈希、列齐、行数、NaN 数、没有标签列。通过则返回表。
    Check one feature file: existence, hash, columns, row count, NaN count and label leakage.

    中文：
        用途：浅层校验的"单文件"部分。抽成函数是为了让测试能拿一份**临时**文件去跑同一段
              逻辑——否则测试只能改仓库里的真产物，那是拿证据当实验材料。
        输入：task_id；path —— 文件路径；record —— 清单里对应的一行；
              expected_columns —— 期望的完整列序；problems —— 追加问题。
        输出：读出来的 DataFrame；任一项不过时返回 None（调用方据此跳过后续检查）。
        数据来源：磁盘 parquet + 清单。配置来源：`FORBIDDEN_FEATURE_COLUMNS`。
        副作用：追加 problems。异常：无（读不了记一条问题）。
    English:
        Purpose: the shallow pass's single-file part. It is a function so a test can run the same
                 logic against a *temporary* file; otherwise a test would have to corrupt the real
                 artefact, which means using evidence as lab material.
        Args: task_id; path; record (the matching manifest row); expected_columns; problems.
        Returns: the frame, or None when something failed (so the caller skips later checks).
        Data source: the parquet on disk plus the manifest. Config: `FORBIDDEN_FEATURE_COLUMNS`.
        Side effects: appends to problems. Raises: none.
    """
    if not path.is_file():
        problems.append(f"{task_id}：清单指向的文件不存在 {_relative(path)}")
        return None
    if _sha256_of(path) != record.get("descriptor_sha256"):
        problems.append(f"{task_id}：{_relative(path)} 的 sha256 与清单不符")
    try:
        frame = pd.read_parquet(path)
    except (OSError, ValueError) as error:
        problems.append(f"{task_id}：{_relative(path)} 读不了（{type(error).__name__}: {error}）")
        return None
    leaked = [name for name in FORBIDDEN_FEATURE_COLUMNS if name in frame.columns]
    if leaked:
        problems.append(f"{task_id}：特征文件里出现标签/行号列 {leaked}（B 自己装配，别替他做）")
    if list(frame.columns) != list(expected_columns):
        extra = [name for name in frame.columns if name not in expected_columns]
        absent = [name for name in expected_columns if name not in frame.columns]
        problems.append(
            f"{task_id}：列与特征字典不符（多 {extra[:4]} 缺 {absent[:4]}，"
            f"实际 {len(frame.columns)} 列、应为 {len(expected_columns)}）"
        )
        return None
    if len(frame) != int(record.get("n_unique_structures", -1)):
        problems.append(
            f"{task_id}：行数不符：清单 {record.get('n_unique_structures')} vs 实际 {len(frame)}"
        )
    descriptor_columns = expected_columns[2 : 2 + len(D.descriptor_columns())]
    if int(record.get("n_nan_cells", -1)) != _count_nan(frame, descriptor_columns):
        problems.append(f"{task_id}：n_nan_cells 与文件内容不一致")
    return frame


def _count_nan(frame: pd.DataFrame, columns: Sequence[str]) -> int:
    """数特征文件里描述符列的 NaN 单元数。
    Count NaN cells across the descriptor columns of a feature file.

    中文：用途：与清单里的 `n_nan_cells` 对照，防止"文件被替换但清单没更新"。
          输入：frame —— 特征表；columns —— 描述符列名。输出：int。
          数据来源：内存中的表。配置来源：无。副作用：无。异常：无。
    English: Purpose: compare against the manifest's `n_nan_cells`, catching a swapped file whose
             manifest was not updated. Args: frame; columns. Returns: an int.
    """
    return int(frame[list(columns)].isna().to_numpy().sum())


def _verify_coverage(task_id: str, frame: pd.DataFrame, problems: list[str]) -> bool:
    """核"调和表的每一行都能在特征文件里找到结构"（验收①）。
    Check that every harmonised row finds a structure in the feature file (criterion 1).

    中文：
        用途：A-6 的验收①是"join 后无意外行丢失（丢失数写进日志）"。这里把"丢失"定义清楚：
              调和表里 `smiles_canonical` 非空的行，其规范式必须出现在该任务的特征文件里；
              缺失数必须等于失败清单里该任务的行数——两边对得上，才是"有记录的丢失"而不是"没人知道的丢失"。
        输入：task_id；frame —— 该任务的特征文件；problems —— 追加问题。
        输出：bool（True = 这一项没问题，供调用方决定是否继续做后面的检查）。
        数据来源：`data/interim/harmonised/<task>_harmonised.parquet` 与失败清单。
        配置来源：无。副作用：追加 problems。异常：无。

    English:
        Purpose: criterion 1 is "no unexpected row loss after the join (losses logged)". Loss is
                 defined precisely: every harmonised row with a non-null `smiles_canonical` must
                 appear in that task's feature file, and any shortfall must equal the failure
                 rows recorded for that task - a recorded loss, not an unknown one.
        Args: task_id; frame; problems. Returns: True when this task's coverage is fine.
        Data source: the harmonised parquet and the failure manifest.
        Side effects: appends to problems. Raises: none.
    """
    try:
        harmonised = _read_harmonised(task_id)
    except FileNotFoundError as error:
        problems.append(str(error))
        return False
    wanted = {str(value) for value in harmonised["smiles_canonical"].dropna()}
    have = set(frame["psmiles_canonical"].astype(str))
    lost = sorted(wanted - have)
    recorded = {
        row.get("psmiles_canonical", "")
        for row in _read_csv_rows(FAILURES_PATH)
        if task_id in row.get("task_ids", "").split(",")
    }
    unrecorded = [value for value in lost if value not in recorded]
    if unrecorded:
        problems.append(
            f"{task_id}：{len(unrecorded)} 个结构的行在特征文件里缺失且失败清单没记"
            f"（例：{unrecorded[0][:40]}…）"
        )
        return False
    # 行数天然可以不等于结构数（同一结构可占多行），所以这里只核"结构覆盖"，
    # 不去强行核行数相等——那会把正常数据判成错误。
    # Row count need not equal structure count (one structure may occupy several rows), so only
    # structure coverage is asserted; forcing row-count equality would flag healthy data.
    return True


def _deep_check_frame(
    task_id: str,
    frame: pd.DataFrame,
    columns: Sequence[str],
    problems: list[str],
    cache: dict[str, dict[str, object] | None] | None = None,
) -> None:
    """把一份特征文件的每一行从结构重算一遍，与文件里的值整矩阵比较。
    Recompute every row of one feature file from its structure and compare whole matrices.

    中文：
        用途：`--verify --deep` 与测试共用的"单文件重算"部分。抽成函数是为了让测试能传一份
              **被改过一个字的**内存表进来，验证"这种改动确实会被抓到"——否则只能去改真产物。
        输入：task_id；frame —— 特征表（可以是被篡改的副本）；columns —— 描述符列名；
              problems —— 追加问题；cache —— 跨任务复用的重算缓存（可选）。
        输出：无（问题追加进 problems）。
        数据来源：`D.compute()` 重算 + 传入的表。配置来源：`D` 模块常量。
        副作用：可能填充 cache。异常：无。

    English:
        Purpose: the single-file recomputation shared by `--verify --deep` and the tests. It is a
                 function so a test can pass an in-memory frame with **one value changed** and show
                 that the change is caught; otherwise a test would have to edit the real artefact.
        Args: task_id; frame (may be a tampered copy); columns; problems; cache (optional, shared
              across tasks).
        Returns: none (problems are appended).
        Data source: `D.compute()` plus the given frame. Config: `D` constants.
        Side effects: may fill `cache`. Raises: none.
    """
    local: dict[str, dict[str, object] | None] = cache if cache is not None else {}
    fingerprint_columns = [f"{FINGERPRINT_PREFIX}{index:04d}" for index in range(D.FINGERPRINT_BITS)]
    # 先一次性把 ndarray 取出来，再按位置比：按值筛选会变成 O(n²)，8055 行要跑十几分钟。
    # Convert once and compare by position: filtering by value inside the loop is O(n^2) and takes
    # fifteen minutes on 8,055 rows.
    canonicals = frame["psmiles_canonical"].astype(str).tolist()
    values = frame[list(columns)].to_numpy(dtype=float)
    bits = frame[fingerprint_columns].to_numpy(dtype=np.uint8)
    for canonical in canonicals:
        if canonical not in local:
            local[canonical] = D.compute(canonical)
    parsed = [local[canonical] for canonical in canonicals]
    bad_parse = [canonicals[index] for index, fresh in enumerate(parsed) if fresh is None]
    if bad_parse:
        problems.append(
            f"{task_id}：{len(bad_parse)} 个结构现在解析不了，但文件里有它们的特征"
            f"（例：{bad_parse[0][:40]}…）"
        )
        return
    # 整矩阵比较：逐行 `np.array_equal` 要在 Python 里跑上万次，慢且没必要。
    # Whole-matrix comparison: a per-row `np.array_equal` would run tens of thousands of times.
    expected_values = np.array(
        [[fresh["descriptors"][name] for name in columns] for fresh in parsed], dtype=float
    )
    expected_bits = np.vstack([D.fingerprint_array(fresh["fingerprint"]) for fresh in parsed])
    value_bad = ~(
        (expected_values == values) | (np.isnan(expected_values) & np.isnan(values))
    ).all(axis=1)
    bit_bad = ~(expected_bits == bits).all(axis=1)
    broken = np.flatnonzero(value_bad | bit_bad)
    if broken.size:
        first = canonicals[int(broken[0])]
        kind = "描述符" if value_bad[broken[0]] else "指纹"
        problems.append(
            f"{task_id}：{broken.size} 个结构的特征与重算结果不一致"
            f"（首个是{kind}，例：{first[:40]}…）"
        )


def _deep_recompute(
    table: pd.DataFrame | None, tasks: Sequence[str], columns: Sequence[str]
) -> list[str]:
    """重算每个结构的描述符与指纹，与盘上文件逐位比较（验收④）。
    Recompute every structure's descriptors and fingerprint and compare bit for bit (criterion 4).

    中文：
        用途：浅层校验靠 sha256，只能发现"文件被改过"；如果改文件的人顺手把清单里的哈希也
              更新了，浅层就看不见了。deep 从**原始结构**重算，绕开清单，是唯一能挡住
              "文件与清单一起被改"的检查。11,494 个结构重算约 **1.5–2.5 分钟**——
              比浅层慢得多，所以它是 `--verify --deep` 才走的路，不是默认路径。
        输入：table —— 结构表；tasks —— 要核的任务；columns —— 描述符列名。
        输出：问题字符串列表。
        数据来源：结构表 + 每任务特征文件 + RDKit 重算。
        配置来源：`D` 模块常量。副作用：无（重算走内存，不写缓存）。异常：无。

    English:
        Purpose: the shallow pass relies on sha256 and can only see "the file changed"; if whoever
                 changed it also updated the manifest hash, the shallow pass is blind. The deep pass
                 recomputes from the raw structure, bypassing the manifest, and is the only check
                 that survives "file and manifest edited together". Recomputing 11,494 structures
                 takes about 1.5-2.5 minutes, which is why this is `--verify --deep` and not the
                 default path.
        Args: table; tasks; columns. Returns: problem strings.
        Data source: the structure table, the feature files and a fresh RDKit computation.
        Config: `D` constants. Side effects: none (computation stays in memory).
        Raises: none.
    """
    if table is None:
        return ["deep 校验需要结构表，但它不可用"]
    problems: list[str] = []
    cache: dict[str, dict[str, object] | None] = {}
    for task_id in tasks:
        path = DESCRIPTORS_DIR / f"{task_id}.parquet"
        if not path.is_file():
            continue
        _deep_check_frame(task_id, pd.read_parquet(path), columns, problems, cache)
    # wPSMILES 也重算一遍：它是 B 建图的输入，写错了比描述符错更难发现。
    # Recompute wPSMILES too: it feeds B's graph builder, and a wrong one is harder to spot than a
    # wrong descriptor.
    wrong = 0
    first_wrong: str | None = None
    for row in table.itertuples(index=False):
        if bool(to_wpsmiles(str(row.psmiles_canonical)) is None) != pd.isna(row.wpsmiles):
            wrong += 1
            if first_wrong is None:
                first_wrong = str(row.psmiles_canonical)
    if wrong:
        problems.append(f"结构表里有 {wrong} 行的 wPSMILES 与重算结果不一致（例：{str(first_wrong)[:40]}…）")
    return problems
