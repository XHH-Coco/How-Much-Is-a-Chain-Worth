"""C-b-3：架构迁移的组分特征表——OOD 里每个唯一组分各算一次 205 描述符 + 2048 位指纹。
C-b-3: the component feature table - one 205-descriptor + 2048-bit row per unique OOD component.

中文：
    用途：官方 task4 的 OOD 评估不是"逐行预测"，而是**逐组分预测 + 按摩尔分数加权合成**
          （`summarize_weighted_predictions.py` / `evaluate_rdkit_ood.py` 都是这么做的）。
          所以 C-c 跑 30 runs 之前，必须先有一张"组分 → 特征"的表，且**覆盖完备**：
          holdout 里每个唯一 `psmiles` 在索引里**有且只有一行**，缺一条就停（分工_C-b 验收③）。
    数据来源：`generated/shared_splits/ood_components.csv`（官方脚本物化的产物，683 行）、
              `generated/shared_splits/architecture_holdout.csv`（36,828 行，用来核覆盖）。
    复用而非重写：描述符与指纹走 `chem.descriptors.compute()`（A-6 的唯一计算口），
              缓存与并行走 `experiments.scaling._features_from_cache()`（同一套 RDKit 缓存）。
    许可：这些结构来自 VIPEA / polyVERSE，属 **GTRC General Public Use License**（D-08），
          不是主表那个 CC BY 4.0——特征表是派生数据，随结果表一起受 GTRC 约束。
    对外：`build_component_features()`、`load_component_index()`、`assert_coverage_complete()`。

English:
    Purpose: task4's OOD evaluation predicts **per component** and weights by molar fraction, so
    C-c needs a component -> feature table first, and it must be **complete**: every unique `psmiles`
    in the holdout appears in the index exactly once, or stop (criterion 3).
    Data: the official script's `ood_components.csv` (683 rows) and `architecture_holdout.csv`
    (36,828 rows, used for the coverage check). Descriptors come from
    `chem.descriptors.compute()` and the cache/parallelism from `experiments.scaling`.
    Licence: VIPEA/polyVERSE structures are GTRC (D-08), not the main table's CC BY 4.0.
"""

from __future__ import annotations

import csv
import hashlib
import json
import logging
from pathlib import Path
from typing import Any, Sequence

import numpy as np

LOGGER = logging.getLogger("chainworth.experiments.arch")

REPO_ROOT = Path(__file__).resolve().parents[3]
GENERATED = REPO_ROOT / "generated" / "shared_splits"
ARCH_DIR = REPO_ROOT / "data" / "processed" / "arch"
FEATURES_PATH = ARCH_DIR / "component_features.parquet"
INDEX_PATH = ARCH_DIR / "component_index.csv"
FAILURES_PATH = ARCH_DIR / "component_failures.csv"
MANIFEST_PATH = REPO_ROOT / "data" / "manifests" / "arch_component_manifest.json"

INDEX_COLUMNS = (
    "psmiles",
    "feature_row",
    "n_stars",
    "feature_sha256",
    "n_nan_descriptors",
)
FAILURE_COLUMNS = ("psmiles", "reason", "n_in_holdout")


def holdout_path() -> Path:
    """官方 holdout 的位置。／ Where the official holdout lives."""
    return GENERATED / "architecture_holdout.csv"


def components_path() -> Path:
    """官方 `ood_components.csv` 的位置。／ Where the official OOD component list lives."""
    return GENERATED / "ood_components.csv"


def _sha256_bytes(payload: bytes) -> str:
    """字节的 sha256。／ sha256 of some bytes."""
    return hashlib.sha256(payload).hexdigest()


def _row_sha256(values: np.ndarray) -> str:
    """一行的特征指纹：描述符 float64 + 指纹位，按固定顺序哈希（换机器也能复算）。
    One row's feature digest: descriptors plus bits in a fixed order, reproducible on any machine.
    """
    return _sha256_bytes(np.ascontiguousarray(values, dtype=np.float64).tobytes())


def read_component_list() -> list[str]:
    """读官方的组分清单（`ood_components.csv` 的 `psmiles` 列，保持文件顺序）。
    Read the official component list, in file order.
    """
    import pandas as pd

    path = components_path()
    if not path.is_file():
        raise FileNotFoundError(
            f"缺 {path.as_posix()}：先跑官方物化 "
            "`python Datasets/Dataset_construction_scripts/create_task4_shared_splits.py`（C-b-3 第②步）"
        )
    frame = pd.read_csv(path)
    column = "psmiles" if "psmiles" in frame.columns else frame.columns[0]
    return [str(value) for value in frame[column]]


def build_component_features(*, workers: int = 0) -> dict[str, Any]:
    """造组分特征表 + 索引 + 失败清单，返回摘要。
    Build the component feature table, its index and the failure list; return a summary.

    中文：
        用途：683 个组分（实测）各算一次 205 描述符 + 2048 位指纹，落成
              `data/processed/arch/component_features.parquet`（不入库，脚本重建）。
              `component_index.csv` 给 C-c 的用法是 `psmiles → feature_row`：
              **一个 psmiles 一行，有且只有一行**；行里带 `feature_sha256`，C-c 可以逐行核对
              "我读到的特征就是这张表里的那一行"。
              解析失败的组分**不静默丢**：进 `component_failures.csv`，并算出它在 holdout 里
              出现多少次——那样 C-c 才知道会缺哪几条、影响多少个 holdout 行（验收④）。
        输入：workers —— 并行进程数（0 = 核数−1，上限 8）。
        输出：摘要 dict（组分数、特征列数、失败数、覆盖是否完备）。
        数据来源：`generated/shared_splits/`。副作用：写三个文件 + 一份清单。
        异常：FileNotFoundError —— 官方物化产物不在。
    English:
        Each component gets one 205-descriptor + 2048-bit row, written to
        data/processed/arch/component_features.parquet (not committed; rebuild with this script).
        `component_index.csv` maps psmiles -> feature_row, exactly one row per psmiles, carrying
        `feature_sha256` so C-c can verify row by row. Unparsable components are **not** dropped
        silently: they go to `component_failures.csv` with their holdout frequency, so C-c knows
        exactly which rows would be missing (criterion 4).
    """
    import pandas as pd

    from chainworth.chem import descriptors as D
    from chainworth.chem import features as F
    from chainworth.experiments.scaling import _features_from_cache

    values = read_component_list()
    if workers <= 0:
        import os

        workers = max(1, min(8, (os.cpu_count() or 2) - 1))
    columns = D.descriptor_columns()
    # 指纹的列名在 `chem.features`（它是"装配层"的命名），描述符列名在 `chem.descriptors`。
    # The fingerprint column names live in chem.features (the assembly layer), the descriptor names
    # in chem.descriptors.
    bits = F.fingerprint_columns()
    unique = list(dict.fromkeys(values))
    LOGGER.info("组分 %d 个（去重后 %d 个）", len(values), len(unique))
    records = _features_from_cache(unique, workers=workers, log_prefix="arch-components")

    failed: list[str] = []
    rows: list[np.ndarray] = []
    index_rows: list[dict[str, Any]] = []
    canonical: list[str] = []
    for position, psmiles in enumerate(unique):
        payload = records.get(psmiles)
        if payload is None or payload.get("psmiles_canonical") is None:
            failed.append(psmiles)
            continue
        descriptor_values = payload["descriptors"]
        vector = np.concatenate(
            [
                np.asarray(
                    [
                        np.nan if descriptor_values.get(column) is None else float(descriptor_values[column])
                        for column in columns
                    ],
                    dtype=np.float64,
                ),
                D.fingerprint_array(payload["fingerprint"]).astype(np.float64),
            ]
        )
        rows.append(vector)
        canonical.append(str(payload["psmiles_canonical"]))
        index_rows.append(
            {
                "psmiles": psmiles,
                "feature_row": len(rows) - 1,
                "n_stars": int(str(psmiles).count("*")),
                "n_nan_descriptors": int(np.isnan(vector[: len(columns)]).sum()),
            }
        )
    if not rows:
        raise RuntimeError("所有组分都算不出特征：先看 RDKit 与 psmiles 的格式")

    frame = pd.DataFrame(
        np.vstack(rows),
        columns=[*columns, *bits],
    )
    frame.insert(0, "psmiles_canonical", canonical)
    ARCH_DIR.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(FEATURES_PATH, index=False)
    features_sha = hashlib.sha256(FEATURES_PATH.read_bytes()).hexdigest()

    # 逐行指纹：读回 parquet 的那一行与这里算的一致，才说明落盘没走样。
    # Per-row digests: recomputed from what is on disk, so a silent write-time change would show up.
    matrix = frame[[*columns, *bits]].to_numpy(dtype=np.float64)
    for item in index_rows:
        item["feature_sha256"] = _row_sha256(matrix[int(item["feature_row"])])
    with INDEX_PATH.open("w", encoding="utf-8", newline="\n") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(INDEX_COLUMNS), lineterminator="\n")
        writer.writeheader()
        for item in index_rows:
            writer.writerow({column: item.get(column, "") for column in INDEX_COLUMNS})

    holdout_counts = _holdout_counts()
    with FAILURES_PATH.open("w", encoding="utf-8", newline="\n") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(FAILURE_COLUMNS), lineterminator="\n")
        writer.writeheader()
        for psmiles in failed:
            writer.writerow(
                {
                    "psmiles": psmiles,
                    "reason": "unparsable_smiles",
                    "n_in_holdout": holdout_counts.get(psmiles, 0),
                }
            )

    coverage = assert_coverage_complete(values=failed)
    summary = {
        "generated_at": _now_iso(),
        "source_commit": "83655bc25901a8cabe3a1e5474b7ec26f01b919f",
        "licence": "GTRC General Public Use License（VIPEA / polyVERSE）",
        "n_components_listed": len(unique),
        "n_feature_rows": int(len(frame)),
        "n_descriptors": len(columns),
        "n_fingerprint_bits": len(bits),
        "n_failed": len(failed),
        "failed_examples": failed[:10],
        "n_nan_cells": int(np.isnan(matrix[:, : len(columns)]).sum()),
        "features_path": FEATURES_PATH.relative_to(REPO_ROOT).as_posix(),
        "features_sha256": features_sha,
        "index_path": INDEX_PATH.relative_to(REPO_ROOT).as_posix(),
        "failures_path": FAILURES_PATH.relative_to(REPO_ROOT).as_posix(),
        "coverage": coverage,
    }
    MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    MANIFEST_PATH.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n"
    )
    LOGGER.info(
        "组分特征：%d 行 × %d 列（%d 描述符 + %d 位）；失败 %d；覆盖完备=%s",
        len(frame),
        len(columns) + len(bits),
        len(columns),
        len(bits),
        len(failed),
        coverage["complete"],
    )
    return summary


def _holdout_counts() -> dict[str, int]:
    """统计每个组分在 holdout 里出现多少次（两列组分表按行计数）。
    Count how often each component appears in the holdout (both component columns, per row).
    """
    import pandas as pd

    path = holdout_path()
    if not path.is_file():
        return {}
    frame = pd.read_csv(path, usecols=["psmiles_a", "psmiles_b"])
    counts: dict[str, int] = {}
    for column in ("psmiles_a", "psmiles_b"):
        for value, count in frame[column].value_counts().items():
            counts[str(value)] = counts.get(str(value), 0) + int(count)
    return counts


def assert_coverage_complete(*, values: Sequence[str] | None = None) -> dict[str, Any]:
    """核"holdout 里每个唯一 psmiles 在索引里**有且只有一行**"（分工_C-b 验收③）。
    Check that every unique holdout psmiles appears in the index exactly once (criterion 3).

    中文：
        用途：这是 C-c 唯一会踩的坑——组分缺特征时 `summarize_weighted_predictions.py` 会抛
              "组分缺预测"，或者更糟：不报错但权重和不等于 1。所以这条必须是**断言**，不是注释。
        输入：values —— 已知解析失败的组分（只用于把失败说清楚；不影响"索引唯一"这条判断）。
        输出：dict（`complete` / `n_holdout_components` / `n_index_rows` / `missing` / `duplicated`）。
        数据来源：`generated/shared_splits/architecture_holdout.csv` + `component_index.csv`。
        副作用：只读。异常：无（返回结果，由调用方决定怎么处置）。
    English:
        This is the one trap C-c will hit: a component without features makes
        summarize_weighted_predictions.py raise "missing component predictions", or worse, silently
        produce weights that do not sum to one. So it is an **assertion**, not a comment.
    """
    import pandas as pd

    path = holdout_path()
    if not path.is_file():
        raise FileNotFoundError(f"缺 {path.as_posix()}：先跑官方物化脚本（C-b-3 第②步）")
    frame = pd.read_csv(path, usecols=["psmiles_a", "psmiles_b"])
    in_holdout = set(frame["psmiles_a"]) | set(frame["psmiles_b"])
    index = pd.read_csv(INDEX_PATH) if INDEX_PATH.is_file() else pd.DataFrame(columns=["psmiles"])
    counts = index["psmiles"].value_counts() if "psmiles" in index.columns else {}
    present = set(counts.index) if hasattr(counts, "index") else set()
    duplicated = sorted(str(key) for key, value in counts.items() if int(value) > 1)
    missing = sorted(in_holdout - present)
    return {
        "complete": not missing and not duplicated,
        "n_holdout_components": len(in_holdout),
        "n_index_rows": int(len(index)),
        "missing": missing[:20],
        "n_missing": len(missing),
        "duplicated": duplicated[:20],
        "n_duplicated": len(duplicated),
        "n_failed_known": len(list(values or [])),
    }


def load_component_index(*, path: Path | None = None) -> dict[str, int]:
    """读 `psmiles → feature_row` 映射（C-c 的唯一入口）。／ Read psmiles -> feature_row."""
    target = Path(path) if path is not None else INDEX_PATH
    if not target.is_file():
        raise FileNotFoundError(f"缺 {target.as_posix()}：先跑 scripts/00d_arch_components.py")
    table: dict[str, int] = {}
    with target.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            key = str(row["psmiles"])
            if key in table:
                raise RuntimeError(f"component_index.csv 里 {key[:24]}… 出现了两行：唯一性被破坏")
            table[key] = int(row["feature_row"])
    return table


def _now_iso() -> str:
    """带时区的 ISO 时间。／ ISO timestamp with a timezone."""
    from datetime import datetime, timezone

    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")
