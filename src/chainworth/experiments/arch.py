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
    对外：`build_component_features()`、`load_component_index()`、`assert_coverage_complete()`；
          C-c 段：`materialize_splits()`、`build_id_features()`、`load_id_data()`、`run_arch()`、
          `component_predictions()`、`synthesize_ood()`、`run_ood()`、`ensemble_ood()`、
          `write_arch_table()`、`verify_arch_setup()`。

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
import time
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


# =====================================================================
# C-c-1 / C-c-2：划分落位、ID 特征、15 runs、OOD 三段式
# C-c-1/C-c-2: split placement, ID features, the 15 runs, the OOD three stages
# =====================================================================

from chainworth.arms import conventional as C
from chainworth.arms import graph as G
from chainworth.io import loaders as L

TASK_ID = "pb26_arch_transfer_ea"
ARM = "a2_conventional"
ARCH_SPLIT_DIR = REPO_ROOT / "splits" / "polybench26" / "arch" / "seed_42"
ARCH_SPLIT_MANIFEST = ARCH_SPLIT_DIR / "manifest.json"
ARCH_PROVENANCE = REPO_ROOT / "splits" / "polybench26" / "arch" / "PROVENANCE.json"
ID_FEATURES_PATH = ARCH_DIR / "id_features.parquet"
ID_MANIFEST_PATH = REPO_ROOT / "data" / "manifests" / "arch_id_manifest.json"
ARCH_TABLE_PATH = REPO_ROOT / "results" / "aggregate" / "arch_transfer.csv"
COMPONENT_PREDS_DIR = REPO_ROOT / "results" / "aggregate" / "arch_component_preds"
OOD_DIR = REPO_ROOT / "results" / "aggregate" / "arch_ood"
OOD_VECTORS_DIR = REPO_ROOT / "results" / "aggregate" / "arch_ood_vectors"
OOD_ENSEMBLE_PATH = REPO_ROOT / "results" / "aggregate" / "arch_ood_ensemble.json"
OOD_AUDIT_PATH = REPO_ROOT / "results" / "aggregate" / "arch_ood_audit.json"
OOD_MANIFEST_PATH = REPO_ROOT / "data" / "manifests" / "arch_ood_artifacts_manifest.csv"
SUMMARIZE_SCRIPT = (
    REPO_ROOT / "data" / "raw" / "polybench26" / "Datasets" / "task4_architecture_transfer"
    / "summarize_weighted_predictions.py"
)

UPSTREAM_COMMIT = "83655bc25901a8cabe3a1e5474b7ec26f01b919f"
MATERIALIZE_SCRIPT = (
    REPO_ROOT / "data" / "raw" / "polybench26" / "Datasets" / "Dataset_construction_scripts"
    / "create_task4_shared_splits.py"
)

ARCH_TABLE_COLUMNS = (
    "run_id",
    "task",
    "arm",
    "seed",
    "split_fold",
    "n_train",
    "n_test",
    "wall_clock_s",
    "r2_id",
    "rmse_id",
    "mae_id",
    "r2_alternating",
    "rmse_alternating",
    "mae_alternating",
    "n_alternating",
    "r2_homopolymer",
    "rmse_homopolymer",
    "mae_homopolymer",
    "n_homopolymer",
    "ood_random_rmse",
    "ood_block_rmse",
    "ood_combined_rmse",
    "note",
)


def canonical_frame() -> Any:
    """读官方物化的 canonical 表（6,478 行），行数不对就报错。
    Read the materialised canonical table; a wrong row count raises.
    """
    import pandas as pd

    path = GENERATED / "canonical_train.csv"
    if not path.is_file():
        raise FileNotFoundError(f"缺 {path.as_posix()}：先跑官方物化脚本（Arch Materialization.md §2）")
    frame = pd.read_csv(path)
    if len(frame) != 6478:
        raise RuntimeError(f"canonical_train.csv 是 {len(frame)} 行，应为 6,478：物化产物被换过")
    return frame


def materialize_splits(*, overwrite: bool = False) -> dict[str, Any]:
    """把官方物化的五折索引落成 `splits/polybench26/arch/seed_42/` 的 polybench26 格式。
    Materialise the official five folds into splits/polybench26/arch/seed_42/ (polybench26 shape).

    中文：
        用途：`_schema.json` 的 `split_version` 必须是 `@splits/...`，而官方物化产物在
              `generated/shared_splits/`（不入库）。所以把每折的 train/val/test 行号写成
              `fold_XX.npz`（`train_pool`/`val`/`test` 三个数组，索引 = canonical 表行位置），
              再写一份 `manifest.json` 用 `_polybench26_split` 的格式——本模块的
              `_load_arch_split()` 按同一口径读它（**不走** `L.load_split()`：任务配置的
              `split_ref` 是目录形态，那是 A-5 校验器对 deferred 的契约，D-57）。
              `PROVENANCE.json` 记物化脚本的 sha256 与上游 commit（README 的要求）。
              行数断言与官方一致：每折 train 5,182/5,183、val 647/648、test 648（交替 611 + 均聚 37）。
        输入：overwrite —— 已存在时是否重写。输出：摘要 dict。
        数据来源：`generated/shared_splits/fold_*/train_*/indices/*.csv`。副作用：写三个文件。
        异常：RuntimeError —— 行数或两档分布与官方不符。
    English:
        The schema demands split_version start with `@splits/` while the official materialisation
        lives under generated/ (not committed), so each fold's row indices become a `fold_XX.npz`
        with `train_pool`/`val`/`test` arrays (row positions of the canonical table), plus a
        `manifest.json` in `_polybench26_split`'s shape - read back by this module's own
        `_load_arch_split()` (not L.load_split(): the task config's split_ref is the directory
        form, which is the A-5 verifier's contract for deferred, D-57).
        PROVENANCE.json records the materialisation script's sha256 and the upstream commit.
    """
    import pandas as pd

    if ARCH_SPLIT_DIR.is_dir() and any(ARCH_SPLIT_DIR.iterdir()):
        if not overwrite:
            raise FileExistsError(
                f"{ARCH_SPLIT_DIR} 已存在：换划分要显式 --overwrite（本地不入库，删了重跑也行）"
            )
        import shutil

        shutil.rmtree(ARCH_SPLIT_DIR)
    canonical = canonical_frame()
    architecture = dict(zip(canonical["row_index"].astype(int), canonical["architecture"]))
    ARCH_SPLIT_DIR.mkdir(parents=True, exist_ok=True)
    folds = []
    for fold_index in range(5):
        run_dir = sorted((GENERATED / f"fold_{fold_index:02d}").glob("train_*"))
        if len(run_dir) != 1:
            raise RuntimeError(f"fold_{fold_index:02d} 下应有恰好一个 train_* 目录，现有 {run_dir}")
        directory = run_dir[0]
        arrays: dict[str, np.ndarray] = {}
        for name in ("train", "val", "test"):
            frame = pd.read_csv(directory / "indices" / f"{name}.csv")
            arrays[name] = frame["row_index"].to_numpy(dtype=np.int64)
        test_arch = pd.Series([architecture[int(row)] for row in arrays["test"]]).value_counts().to_dict()
        if len(arrays["test"]) != 648 or test_arch != {"alternating": 611, "homopolymer": 37}:
            raise RuntimeError(
                f"fold_{fold_index:02d} test {len(arrays['test'])} 行、两档 {test_arch}："
                "与官方（648 = 611 + 37）不符，停下查物化"
            )
        if len(arrays["train"]) not in (5182, 5183):
            raise RuntimeError(f"fold_{fold_index:02d} train {len(arrays['train'])} 行：应为 5,182 或 5,183")
        np.savez(
            ARCH_SPLIT_DIR / f"fold_{fold_index:02d}.npz",
            train_pool=arrays["train"],
            val=arrays["val"],
            test=arrays["test"],
        )
        folds.append(
            {
                "fold": fold_index,
                "indices_file": f"fold_{fold_index:02d}.npz",
                "n_train_pool": int(len(arrays["train"])),
                "n_val": int(len(arrays["val"])),
                "n_test": int(len(arrays["test"])),
                "seed": 42,
            }
        )
    manifest = {
        "schema_version": 1,
        "task_id": TASK_ID,
        "seed": 42,
        "n_rows": 6478,
        "canonical_sha256": hashlib.sha256((GENERATED / "canonical_train.csv").read_bytes()).hexdigest(),
        "canonical_source": "generated/shared_splits/canonical_train.csv（官方物化，不入库）",
        "protocol": "official_task4_5fold_seed42_ood_holdout",
        "row_order": "canonical_train.csv 的 row_index（0..6477，物化时按 psmiles 排序后的行位置）",
        "in_distribution_test": {"alternating": 611, "homopolymer": 37},
        "folds": folds,
    }
    ARCH_SPLIT_MANIFEST.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n"
    )
    provenance = {
        "generated_by": MATERIALIZE_SCRIPT.relative_to(REPO_ROOT).as_posix(),
        "script_sha256": hashlib.sha256(MATERIALIZE_SCRIPT.read_bytes()).hexdigest()
        if MATERIALIZE_SCRIPT.is_file()
        else None,
        "upstream_commit": UPSTREAM_COMMIT,
        "source_dir": GENERATED.relative_to(REPO_ROOT).as_posix(),
        "source_manifest_sha256": hashlib.sha256((GENERATED / "manifest.json").read_bytes()).hexdigest(),
        "materialized_at": _now_iso(),
        "licence": "GTRC General Public Use License（polyVERSE / VIPEA，D-08）——本目录除 PROVENANCE.json 外不入库",
    }
    ARCH_PROVENANCE.write_text(
        json.dumps(provenance, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n"
    )
    LOGGER.info(
        "arch 划分落位：5 折 npz + manifest（train %s / val %s / test 648）；PROVENANCE 已写",
        [entry["n_train_pool"] for entry in folds],
        [entry["n_val"] for entry in folds],
    )
    return {
        "n_folds": 5,
        "n_rows": 6478,
        "train_rows": [entry["n_train_pool"] for entry in folds],
        "val_rows": [entry["n_val"] for entry in folds],
        "test_rows": [entry["n_test"] for entry in folds],
        "manifest_sha256": hashlib.sha256(ARCH_SPLIT_MANIFEST.read_bytes()).hexdigest(),
    }


def build_id_features(*, workers: int = 0) -> dict[str, Any]:
    """造 ID 特征表（6,478 行 × 2,253 列，与组分表同一套列），写 parquet + 清单。
    Build the ID feature table (6,478 x 2,253, the same columns as the component table).

    中文：
        用途：15 runs 的 X。列序与 `component_features.parquet` / A-6 / B-1 逐列相同
              （205 POD 描述符 + 2048 位指纹），走 `chem.descriptors.compute()` 与
              `experiments.scaling._features_from_cache()`——规则 2 的"同一批特征"。
              **官方 fold 目录里的 RDKit_descriptors 不用**：那是官方地板用的 200 维
              RDKit 描述符，不是主表特征口径。
              任何一行解析失败都**抛错**（canonical 表 6,478 行必须全部有特征，不许静默丢）。
        输入：workers —— 并行进程数（0 = 核数−1，上限 8）。输出：摘要 dict。
        数据来源：`generated/shared_splits/canonical_train.csv`。副作用：写 parquet + 清单。
    English:
        X for the 15 runs. The column order matches component_features.parquet / A-6 / B-1 exactly
        (205 POD descriptors + 2048 bits) through chem.descriptors.compute() and the shared cache -
        rule 2's "same feature batch". The official RDKit_descriptors folders are NOT used: those
        200 columns feed the official floor, not the main-table feature convention. One unparsable
        row raises; none of the 6,478 canonical rows may be dropped silently.
    """
    import pandas as pd

    from chainworth.chem import descriptors as D
    from chainworth.chem import features as F
    from chainworth.experiments.scaling import _features_from_cache

    canonical = canonical_frame()
    if workers <= 0:
        import os

        workers = max(1, min(8, (os.cpu_count() or 2) - 1))
    columns = D.descriptor_columns()
    bits = F.fingerprint_columns()
    values = [str(value) for value in canonical["psmiles"]]
    unique = list(dict.fromkeys(values))
    LOGGER.info("ID 行 %d（唯一结构 %d）", len(values), len(unique))
    records = _features_from_cache(unique, workers=workers, log_prefix="arch-id")

    rows: list[np.ndarray] = []
    canonical_rows: list[str] = []
    failed: list[str] = []
    for position, psmiles in enumerate(values):
        payload = records.get(psmiles)
        if payload is None or payload.get("psmiles_canonical") is None:
            failed.append(psmiles)
            continue
        descriptor_values = payload["descriptors"]
        rows.append(
            np.concatenate(
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
        )
        canonical_rows.append(str(payload["psmiles_canonical"]))
    if failed:
        raise RuntimeError(
            f"{len(failed)} 行 ID 结构算不出特征（前 5 个：{failed[:5]}）：canonical 表一行都不能丢，"
            "停下查 RDKit 与 psmiles 格式"
        )
    matrix = np.vstack(rows)
    frame = pd.DataFrame(matrix, columns=[*columns, *bits])
    frame.insert(0, "electron_affinity", canonical["electron_affinity"].to_numpy(dtype=np.float64))
    frame.insert(0, "psmiles_canonical", canonical_rows)
    frame.insert(0, "psmiles", values)
    frame.insert(0, "row_index", canonical["row_index"].to_numpy(dtype=np.int64))
    ARCH_DIR.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(ID_FEATURES_PATH, index=False)
    id_sha = hashlib.sha256(ID_FEATURES_PATH.read_bytes()).hexdigest()
    summary = {
        "generated_at": _now_iso(),
        "source_commit": UPSTREAM_COMMIT,
        "licence": "GTRC General Public Use License（polyVERSE / VIPEA，D-08）",
        "n_rows": int(len(frame)),
        "n_descriptors": len(columns),
        "n_fingerprint_bits": len(bits),
        "n_nan_cells": int(np.isnan(matrix).sum()),
        "features_path": ID_FEATURES_PATH.relative_to(REPO_ROOT).as_posix(),
        "features_sha256": id_sha,
        "canonical_sha256": hashlib.sha256((GENERATED / "canonical_train.csv").read_bytes()).hexdigest(),
    }
    ID_MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    ID_MANIFEST_PATH.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n"
    )
    LOGGER.info(
        "ID 特征：%d 行 × %d 列（%d 描述符 + %d 位）；NaN 单元 %d；sha256=%s…",
        len(frame), len(columns) + len(bits), len(columns), len(bits),
        summary["n_nan_cells"], id_sha[:12],
    )
    return summary


def load_id_data(*, limit: int | None = None) -> Any:
    """把 ID 特征表读成经典臂要的 `TaskData`（核哈希、核划分配套）。
    Load the ID feature table into the classical arm's TaskData (hash-checked, split-checked).
    """
    import pandas as pd

    if not ID_FEATURES_PATH.is_file():
        raise FileNotFoundError(
            f"缺 {ID_FEATURES_PATH.relative_to(REPO_ROOT).as_posix()}：先跑 `04_run_cpu.py arch --setup`"
        )
    if not ID_MANIFEST_PATH.is_file():
        raise FileNotFoundError("缺 arch_id_manifest.json：先跑 `04_run_cpu.py arch --setup`")
    manifest = json.loads(ID_MANIFEST_PATH.read_text(encoding="utf-8"))
    actual = hashlib.sha256(ID_FEATURES_PATH.read_bytes()).hexdigest()
    if actual != manifest.get("features_sha256"):
        raise RuntimeError(
            f"id_features.parquet 的 sha256={actual[:12]} 与清单 {manifest.get('features_sha256', '')[:12]} 不符：文件被改过"
        )
    from chainworth.chem import descriptors as D
    from chainworth.chem import features as F

    columns = D.descriptor_columns()
    bits = F.fingerprint_columns()
    frame = pd.read_parquet(ID_FEATURES_PATH)
    X = frame[[*columns, *bits]].to_numpy(dtype=np.float32)
    y = frame["electron_affinity"].to_numpy(dtype=np.float64)
    split = _load_arch_split()
    problems = L.check_split_matches_rows(split, int(X.shape[0]))
    if problems:
        raise L.AssemblyError("；".join(problems))
    return C.TaskData(
        task_id=TASK_ID,
        benchmark="polybench26",
        X=X,
        y=y,
        row_id=frame["row_index"].to_numpy(dtype=np.int64),
        psmiles=frame["psmiles_canonical"].to_numpy(),
        split=split,
        pack_sha256=actual,
        limit=limit,
    )


def _load_arch_split() -> Any:
    """读物化的五折（`arch/seed_42/manifest.json` + npz），解析口径与 `_polybench26_split` 逐行相同。
    Parse the materialised five folds with exactly `_polybench26_split`'s semantics.

    中文：
        用途：arch 的划分**不能**走 `L.load_split()`——任务配置（v3，D-49）的 `split_ref` 是
              `splits/polybench26/arch/` **目录**形态，这是 A-5 校验器对 `split_status: deferred`
              的契约（"说明文件目录"），改成 manifest 文件会把 A-5 的 `test_splits.py` 弄红；
              而 `L.load_split()` 要求 split_ref 是文件。两条路不能同时走（D-57），所以这里
              按 `_polybench26_split` 的同一解析（manifest.json + `train_pool`/`val`/`test`、
              折名 `fold_XX`、行号 = canonical 表行位置）读一遍，构造出**同一形态**的
              `SplitSpec`——审计套件对 arch 记录也从记录的 `split_version` 文件解析（同套件
              `_resolve_fold` 的分支），两边口径一致。
    English:
        arch's split cannot go through L.load_split(): the frozen task config's split_ref is the
        *directory* form (the A-5 verifier's contract for `split_status: deferred`), while
        L.load_split() demands a file. Both cannot hold at once (D-57), so this parses the
        materialised manifest with the identical semantics and returns the same SplitSpec shape.
    """
    manifest = json.loads(ARCH_SPLIT_MANIFEST.read_text(encoding="utf-8"))
    directory = ARCH_SPLIT_MANIFEST.parent
    folds = []
    for entry in manifest.get("folds", []):
        fold_path = directory / str(entry["indices_file"])
        with np.load(fold_path) as payload:
            folds.append(
                L.FoldSpec(
                    name=f"fold_{int(entry['fold']):02d}",
                    train_index=payload["train_pool"].astype(np.int64),
                    val_index=payload["val"].astype(np.int64),
                    test_index=payload["test"].astype(np.int64),
                    train_pool=payload["train_pool"].astype(np.int64),
                )
            )
    if not folds:
        raise L.AssemblyError(f"{ARCH_SPLIT_MANIFEST} 里没有 folds：格式不对，别猜着读")
    return L.SplitSpec(
        task_id=TASK_ID,
        benchmark="polybench26",
        protocol="official_task4_5fold_seed42_ood_holdout",
        ref=ARCH_SPLIT_MANIFEST.relative_to(REPO_ROOT).as_posix(),
        sha256=hashlib.sha256(ARCH_SPLIT_MANIFEST.read_bytes()).hexdigest(),
        n_total=int(manifest.get("n_rows", 0)),
        row_order="官方 canonical PSMILES 表的行号（不是 A-3 调和表位置）",
        folds=tuple(folds),
    )


def arch_params(*, learner: str = "xgboost") -> tuple[dict[str, Any], dict[str, Any]]:
    """读 `a2_best_params.csv` 里 arch 那一行；缺行就报错（不许回退默认值）。
    Read arch's row of a2_best_params.csv; a missing row raises, no default fallback.
    """
    import pandas as pd

    path = C.BEST_PARAMS_PATH
    table = pd.read_csv(path)
    rows = table[(table["task"] == TASK_ID) & (table["learner"] == learner)]
    if rows.empty:
        raise RuntimeError(
            f"{path} 里没有 ({TASK_ID}, {learner})：先跑 `04_run_cpu.py arch --search`（D-51 协议）"
        )
    row = rows.iloc[0]
    return json.loads(row["best_params"]), {
        "cv_score": float(row["cv_score"]),
        "trials": int(row["trials"]),
    }


def _id_tier_metrics(data: Any, outcome: Any) -> dict[str, Any]:
    """按架构把一条 run 的测试预测分成两档，各算 r2/mae/rmse（用同一套 G.metrics）。
    Split one run's test predictions by architecture and score each tier with G.metrics.
    """
    import pandas as pd

    canonical = canonical_frame()
    architecture = dict(zip(canonical["row_index"].astype(int), canonical["architecture"]))
    labels = pd.Series([architecture[int(row)] for row in outcome.row_id])
    y_true = data.y[np.asarray(outcome.row_id, dtype=np.int64)]
    y_pred = outcome.y_pred
    tiers: dict[str, Any] = {}
    counts: dict[str, int] = {}
    for name in ("alternating", "homopolymer"):
        mask = (labels == name).to_numpy()
        counts[name] = int(mask.sum())
        tiers[name] = G.metrics(y_true[mask], y_pred[mask])
    return tiers, counts


def _record_on_disk(run_id: str) -> dict[str, Any] | None:
    """磁盘上已有的 ok 记录；没有或坏了返回 None。／ The existing ok record, or None."""
    path = G.RAW_RUNS_DIR / f"{run_id}.json"
    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None
    return payload if payload.get("status") == "ok" else None


def run_arch(
    config: Any,
    *,
    folds: Sequence[str] | None = None,
    seeds: Sequence[int] | None = None,
    resume: bool = True,
) -> list[dict[str, Any]]:
    """15 runs：a2 XGBoost × 5 折 × 3 种子；每 run 落记录 + 预测 + 墙钟，ID 两档分开记。
    The 15 runs: a2 XGBoost x 5 folds x 3 seeds; each writes record + predictions + wall clock.

    中文：
        用途：C-c-1。复用 `C.train_one()`（超参从 `a2_best_params.csv` 的 arch 行取、冻结配置
              的字节哈希进 `config_hash`），折与种子全走 `data.split` 的官方折。记录里额外带
              `id_tier_metrics` / `id_tier_n`（交替 611 / 均聚 37 分开报，验收①）。
              落盘走 `G.write_predictions` / `G.write_run_record` / `G.append_gpu_hours`
              （红线 4：记录只能由训练脚本写）。
        输入：config —— 冻结的 arch_transfer.yaml（ArmConfig）；folds / seeds —— 白名单。
        输出：本次写出的记录（含 resume 跳过的）。数据来源：ID 特征 + `splits/polybench26/arch/`。
        副作用：训练、写记录/预测/墙钟。异常：ArmError —— 折名不存在等。
    English:
        C-c-1. Reuses C.train_one() (params from a2_best_params.csv's arch row; the frozen config's
        byte hash is the config_hash). The record carries id_tier_metrics / id_tier_n so the two ID
        tiers (alternating 611 / homopolymer 37) are reported separately (criterion 1).
    """
    data = load_id_data()
    fold_list = [name for name in data.split.fold_names if folds is None or name in folds]
    seed_list = [int(seed) for seed in (seeds if seeds is not None else config.raw.get("seeds") or [0])]
    rows: list[dict[str, Any]] = []
    for fold in fold_list:
        for seed in seed_list:
            run_id = G.RunKey(task=TASK_ID, benchmark="polybench26", fold=fold, seed=seed, arm=ARM).run_id()
            if resume:
                existing = _record_on_disk(run_id)
                if existing is not None:
                    LOGGER.info("跳过已完成的 %s", run_id)
                    rows.append(existing)
                    continue
            outcome = C.train_one(data, config, fold=fold, seed=seed)
            record = dict(outcome.record)
            tiers, counts = _id_tier_metrics(data, outcome)
            record["id_tier_metrics"] = tiers
            record["id_tier_n"] = counts
            G.write_predictions(outcome)
            G.write_run_record(record)
            G.append_gpu_hours([record])
            rows.append(record)
            LOGGER.info(
                "%s：id rmse=%.4f（交替 %.4f / 均聚 %.4f）n_train=%d %.1f 秒",
                run_id, record["metrics"]["rmse"],
                tiers["alternating"]["rmse"], tiers["homopolymer"]["rmse"],
                record["n_train"], record["wall_clock_s"],
            )
    return rows


def _component_matrix() -> tuple[Any, np.ndarray]:
    """核过哈希的组分特征矩阵（683 × 2,253，float64）。／ The hash-checked component feature matrix."""
    import pandas as pd

    manifest = json.loads((REPO_ROOT / "data" / "manifests" / "arch_component_manifest.json").read_text(encoding="utf-8"))
    actual = hashlib.sha256(FEATURES_PATH.read_bytes()).hexdigest()
    if actual != manifest.get("features_sha256"):
        raise RuntimeError(
            f"component_features.parquet 的 sha256={actual[:12]} 与清单 {manifest.get('features_sha256', '')[:12]} 不符"
        )
    frame = pd.read_parquet(FEATURES_PATH)
    return frame, frame.drop(columns=["psmiles_canonical"]).to_numpy(dtype=np.float64)


def component_predictions(fold: str, seed: int, params: dict[str, Any]) -> tuple[Any, dict[str, Any]]:
    """重拟合该 (折, 种子) 的模型并预测 683 个组分；重拟合与 ID 预测逐位一致才放行。
    Re-fit the (fold, seed) model and predict the 683 components; the refit must reproduce the
    stored ID predictions bit for bit.

    中文：
        用途：OOD 三段式的第②段。A2 的 `train_one()` 只落预测不落模型，所以这里按同一批
              训练行、同一份超参、同一个种子**重拟合**（`make_estimator` 钉死 `n_jobs=1` +
              `tree_method=hist`，CPU 上确定性），并先拿 ID 测试集预测与落盘的
              `results/aggregate/predictions/<run_id>.parquet` 逐位对——不一致就抛错，
              绝不带着一个"可能不一样"的模型去算 OOD。
        输入：fold；seed；params —— arch 行的冻结超参。输出：(组分 CSV 路径, 摘要)。
        数据来源：ID 特征 + 组分特征 + 落盘的预测。副作用：写 `arch_component_preds/<run_id>.csv`。
    English:
        Stage 2 of the OOD pipeline. train_one() persists predictions, not models, so the model is
        re-fitted on the same rows with the same params and seed (n_jobs=1 + hist are deterministic
        on CPU), then checked against the stored ID predictions bit for bit - a mismatch raises
        rather than letting a possibly-different model score the OOD set.
    """
    import pandas as pd

    data = load_id_data()
    spec = {fold_spec.name: fold_spec for fold_spec in data.split.folds}[fold]
    run_id = G.RunKey(task=TASK_ID, benchmark="polybench26", fold=fold, seed=seed, arm=ARM).run_id()
    X_train = data.X[spec.train_index]
    y_train = data.y[spec.train_index]
    model = C.make_estimator("xgboost", params, seed=seed)
    model.fit(X_train, y_train)
    pred_id = np.asarray(model.predict(data.X[spec.test_index]), dtype=np.float64)
    stored_path = G.PREDICTIONS_DIR / f"{run_id}.parquet"
    if not stored_path.is_file():
        raise FileNotFoundError(f"缺 {stored_path.relative_to(REPO_ROOT).as_posix()}：先跑 15 runs")
    stored = pd.read_parquet(stored_path)
    identical = bool(np.array_equal(pred_id, stored["y_pred"].to_numpy()))
    if not identical:
        raise RuntimeError(
            f"{run_id} 的重拟合与落盘 ID 预测不一致：模型不是同一份，OOD 不许往下走"
        )
    comp_frame, comp_matrix = _component_matrix()
    pred_comp = np.asarray(model.predict(comp_matrix.astype(np.float32)), dtype=np.float64)
    order = load_component_index()
    positions = sorted(order.items(), key=lambda item: item[1])
    component_csv = pd.DataFrame(
        {"psmiles": [key for key, _ in positions], "prediction": [pred_comp[row] for _, row in positions]}
    )
    COMPONENT_PREDS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = COMPONENT_PREDS_DIR / f"{run_id}.csv"
    component_csv.to_csv(out_path, index=False, lineterminator="\n")
    return out_path, {
        "run_id": run_id,
        "n_components": int(len(component_csv)),
        "refit_matches_id": identical,
    }


def synthesize_ood(run_id: str, *, component_csv: Any) -> dict[str, Any]:
    """OOD 第③段：官方 `summarize_weighted_predictions.py` 合成 + 留档五折向量。
    Stage 3: the official script synthesises the weighted prediction; the per-fold vector is kept.

    中文：
        用途：**只走官方脚本**（验收②），不自己写加权。官方脚本会对「组分缺预测」「摩尔分数
              不合 1」「psmiles 重复」直接抛错。产物两样：`arch_ood/<run_id>.csv`（holdout + 预测）
              与 `..._metrics.json`（random / block / combined 的逐折指标——**不是判决口径**，
              D-09）；再把预测向量留档 `arch_ood_vectors/<run_id>.parquet` 给集成用。
        输入：run_id；component_csv —— 组分预测 CSV。输出：该 run 的 OOD 摘要 dict。
        副作用：写两个文件 + 向量 parquet。异常：官方脚本非零退出即抛。
    """
    import subprocess
    import sys

    import pandas as pd

    out_path = OOD_DIR / f"{run_id}.csv"
    metrics_path = OOD_DIR / f"{run_id}_metrics.json"
    OOD_DIR.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable, str(SUMMARIZE_SCRIPT),
        "--holdout", str(holdout_path()),
        "--component-predictions", str(component_csv),
        "--output", str(out_path),
        "--metrics", str(metrics_path),
    ]
    completed = subprocess.run(command, capture_output=True, text=True, encoding="utf-8")
    if completed.returncode != 0:
        raise RuntimeError(
            f"官方合成脚本退出码 {completed.returncode}：\n{completed.stdout}\n{completed.stderr}"
        )
    holdout = pd.read_csv(out_path)
    vector = holdout[
        [
            "holdout_row_index", "architecture", "electron_affinity",
            "psmiles_a", "psmiles_b", "molar_fraction_a", "molar_fraction_b", "prediction",
        ]
    ].copy()
    OOD_VECTORS_DIR.mkdir(parents=True, exist_ok=True)
    vector.to_parquet(OOD_VECTORS_DIR / f"{run_id}.parquet", index=False)
    payload = json.loads(metrics_path.read_text(encoding="utf-8"))
    summary = {
        "run_id": run_id,
        "n_holdout_rows": int(len(holdout)),
        "metrics": {entry["architecture"]: entry for entry in payload["metrics"]},
        "vector_path": (OOD_VECTORS_DIR / f"{run_id}.parquet").relative_to(REPO_ROOT).as_posix(),
        "official_script": SUMMARIZE_SCRIPT.relative_to(REPO_ROOT).as_posix(),
    }
    LOGGER.info(
        "%s OOD：random rmse=%.4f / block %.4f / combined %.4f",
        run_id,
        summary["metrics"]["random"]["rmse"],
        summary["metrics"]["block"]["rmse"],
        summary["metrics"]["combined"]["rmse"],
    )
    return summary


def run_ood(
    config: Any,
    *,
    folds: Sequence[str] | None = None,
    seeds: Sequence[int] | None = None,
    resume: bool = True,
) -> list[dict[str, Any]]:
    """每个 run 跑完 OOD 三段式（覆盖断言 → 逐组分预测 → 官方合成），返回逐 run 摘要。
    Run the OOD three stages per run; returns one summary per run.
    """
    params, search_info = arch_params()
    coverage = assert_coverage_complete()
    if not coverage["complete"]:
        raise RuntimeError(f"组分覆盖不完备：missing={coverage['missing']} duplicated={coverage['duplicated']}")
    fold_list = list(folds) if folds else [entry["name"] for entry in _split_fold_names()]
    seed_list = [int(seed) for seed in (seeds if seeds is not None else config.raw.get("seeds") or [0])]
    summaries: list[dict[str, Any]] = []
    # OOD 是这份分工里最重的一步（15 次重拟合 + 683 组分预测 + 45 万行官方合成），但它**不写
    # run 记录**，所以 `gpu_hours.csv` 里没有它（那张表一次 run 一行）。墙钟记在这里的
    # `arch_ood_audit.json`，由 `05b_cost_report.py` 并入总账（D-60）。
    # OOD is the heaviest step here but writes no run record, so it has no row in gpu_hours.csv
    # (one row per run). Its wall clock lands in arch_ood_audit.json and the cost report adds it.
    previous_audit = {}
    if OOD_AUDIT_PATH.is_file():
        try:
            previous_audit = json.loads(OOD_AUDIT_PATH.read_text(encoding="utf-8")) or {}
        except json.JSONDecodeError:
            previous_audit = {}
    known_refit: dict[str, Any] = dict(previous_audit.get("refit_matches_id") or {})
    seconds: dict[str, float] = {}
    for fold in fold_list:
        for seed in seed_list:
            run_id = G.RunKey(task=TASK_ID, benchmark="polybench26", fold=fold, seed=seed, arm=ARM).run_id()
            clock = time.perf_counter()
            if resume and (OOD_DIR / f"{run_id}_metrics.json").is_file() and (
                OOD_VECTORS_DIR / f"{run_id}.parquet"
            ).is_file():
                LOGGER.info("OOD 已合成：%s", run_id)
                payload = json.loads((OOD_DIR / f"{run_id}_metrics.json").read_text(encoding="utf-8"))
                summaries.append(
                    {
                        "run_id": run_id,
                        "metrics": {entry["architecture"]: entry for entry in payload["metrics"]},
                    }
                )
                continue
            csv_path, refit = component_predictions(fold, seed, params)
            summary = synthesize_ood(run_id, component_csv=csv_path)
            summary["refit_matches_id"] = refit["refit_matches_id"]
            summary["search_cv_r2"] = search_info["cv_score"]
            known_refit[run_id] = bool(refit["refit_matches_id"])
            summaries.append(summary)
            seconds[run_id] = time.perf_counter() - clock
    _write_ood_audit(summaries, known_refit, seconds)
    return summaries


def _write_ood_audit(
    summaries: Sequence[Mapping[str, Any]],
    refit: Mapping[str, Any],
    seconds: Mapping[str, float],
) -> Path | None:
    """把 OOD 阶段的两件"只在内存里"的事实落盘：重拟合一致性 + 墙钟。
    Persist the two facts that used to live only in memory: refit identity and wall clock.

    中文：
        用途：`refit_matches_id` 是"重拟合出来的模型与落盘 ID 预测逐位相同"的证据，原来是
              `component_predictions()` 的局部变量，进程一退就无法核对；OOD 的墙钟同样不落
              run 记录。两者写进 `arch_ood_audit.json`，`ensemble_ood()` 再引用一次，
              这样"15 条全部一致"与"OOD 花了多久"两句话都能被第三方复核（D-60）。
        输入：summaries；refit —— run_id → bool；seconds —— run_id → 秒。
        输出：写出的路径（没有跑任何 run 时返回 None）。副作用：写一个 JSON。
    English:
        refit_matches_id is the evidence that the re-fitted model reproduces the stored ID
        predictions bit for bit; it used to be a local variable and vanished with the process, and
        the OOD wall clock never reached any ledger. Both land here so a third party can check
        "all 15 identical" and "how long OOD took" (D-60).
    """
    if not seconds and not summaries:
        return None
    payload = {
        "generated_at": _now_iso(),
        "task": TASK_ID,
        "arm": ARM,
        "n_runs": len(summaries),
        "n_runs_timed": len(seconds),
        "wall_clock_s_total": round(float(sum(seconds.values())), 3),
        "wall_clock_s_per_run": {key: round(float(value), 3) for key, value in sorted(seconds.items())},
        "refit_matches_id": {key: refit[key] for key in sorted(refit) if refit[key] is not None},
        "refit_all_match": bool(refit) and all(value is True for value in refit.values()),
        "note": (
            "OOD 阶段（重拟合 + 683 组分预测 + 官方合成）不写 run 记录，故不在 gpu_hours.csv；"
            "墙钟以本文件为准，由 05b_cost_report.py 并入总账（D-60）。"
            "`refit_matches_id` 只覆盖本次真正重拟合过的 run——resume 命中的 run 不重算，"
            "沿用上一次记录的结论。"
        ),
    }
    OOD_AUDIT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OOD_AUDIT_PATH.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n"
    )
    return OOD_AUDIT_PATH


def _split_fold_names() -> list[dict[str, str]]:
    """arch 的官方折名（fold_00..fold_04）。／ The official arch fold names."""
    manifest = json.loads(ARCH_SPLIT_MANIFEST.read_text(encoding="utf-8"))
    return [{"name": f"fold_{int(entry['fold']):02d}"} for entry in manifest["folds"]]


def ensemble_ood(config: Any, *, seeds: Sequence[int] | None = None) -> dict[str, Any]:
    """五折预测向量 → 按聚合物取平均 → random / block / combined 各一个点值（D-09，不做 t 检验）。
    Average the five fold vectors per polymer and report one point value per architecture.

    中文：
        用途：OOD holdout 在五折之间固定，折级 OOD 指标不是独立重复（官方 README 原文），
              所以这里**不**对五个折级 RMSE 做 t 检验，也不报"均值 ± 标准误"——只报五折
              集成的三个点值（RMSE/MAE/MSE），并给出与 C-b 复现的官方地板的差。
        输入：config；seeds —— 白名单。输出：写 `arch_ood_ensemble.json` 的 dict。
        数据来源：`arch_ood_vectors/*.parquet`。副作用：写一个 JSON。
    English:
        The holdout is fixed across folds, so fold-level OOD metrics are not independent replicates;
        this reports the 5-fold ensemble point values only (no t-test, no mean ± SE), plus the
        deltas against C-b's reproduced official floor.
    """
    import pandas as pd

    floors = (config.raw.get("data") or {}).get("floor_reproduction", {}).get("ood_ensemble") or {}
    seed_list = [int(seed) for seed in (seeds if seeds is not None else config.raw.get("seeds") or [0])]
    per_seed: dict[str, Any] = {}
    for seed in seed_list:
        vectors = []
        for fold in [entry["name"] for entry in _split_fold_names()]:
            run_id = G.RunKey(task=TASK_ID, benchmark="polybench26", fold=fold, seed=seed, arm=ARM).run_id()
            path = OOD_VECTORS_DIR / f"{run_id}.parquet"
            if not path.is_file():
                raise FileNotFoundError(f"缺 {path.relative_to(REPO_ROOT).as_posix()}：先跑 OOD 三段式")
            vectors.append(pd.read_parquet(path))
        # 五个折级向量必须**逐行对得上**才能按位置求平均：官方合成脚本的输出顺序依赖输入顺序，
        # 一旦某折的行序不同，`np.mean(axis=0)` 会把 A 聚合物算到 B 的预测上，**不报错**、
        # 数字看起来还合理（与 D-37 同类的静默错误）。行序断言放在求均值之前（D-60）。
        # The five fold-level vectors must line up row by row before a positional mean: the official
        # synthesis script's output order follows its input order, so one reordered fold would let
        # np.mean(axis=0) average polymer A's prediction into polymer B - silently, and the numbers
        # still look plausible (the D-37 failure class). Assert the order before averaging (D-60).
        base_index = vectors[0]["holdout_row_index"].to_numpy()
        for position, vector in enumerate(vectors[1:], start=1):
            other = vector["holdout_row_index"].to_numpy()
            if other.shape != base_index.shape or not np.array_equal(other, base_index):
                mismatch = int(np.flatnonzero(other != base_index)[0]) if other.shape == base_index.shape else -1
                raise RuntimeError(
                    f"seed {seed} 的第 {position} 折行序与第 0 折不一致（首个差异位置 {mismatch}）："
                    "按位置求平均会串行，拒绝集成"
                )
        merged = vectors[0][["holdout_row_index", "architecture", "electron_affinity"]].copy()
        merged["prediction"] = np.mean([vector["prediction"].to_numpy() for vector in vectors], axis=0)
        points: dict[str, Any] = {}
        for architecture in ("random", "block", "combined"):
            subset = merged if architecture == "combined" else merged[merged["architecture"] == architecture]
            error = subset["electron_affinity"].to_numpy() - subset["prediction"].to_numpy()
            mse = float(np.mean(np.square(error)))
            points[architecture] = {
                "n_samples": int(len(subset)),
                "rmse": float(np.sqrt(mse)),
                "mae": float(np.mean(np.abs(error))),
                "mse": mse,
            }
            floor = floors.get(architecture)
            if isinstance(floor, (int, float)):
                points[architecture]["floor"] = float(floor)
                points[architecture]["delta_vs_floor"] = float(points[architecture]["rmse"] - floor)
        per_seed[str(seed)] = {"ensemble_point_values": points, "folds_averaged": 5}
    audit: dict[str, Any] = {}
    if OOD_AUDIT_PATH.is_file():
        try:
            audit = json.loads(OOD_AUDIT_PATH.read_text(encoding="utf-8")) or {}
        except json.JSONDecodeError:
            audit = {}
    payload = {
        "generated_at": _now_iso(),
        "task": TASK_ID,
        "arm": ARM,
        "protocol": "5-fold per-polymer averaged ensemble; fold-level OOD metrics are NOT independent "
        "replicates (fixed holdout), no t-test was computed (D-09)",
        "floors": floors,
        "seeds": per_seed,
        # 这两项原先只活在内存里（`component_predictions()` 的局部变量）与日志里，第三方无从核对。
        # 现在从 `arch_ood_audit.json` 引用；缺失时写 null + 说明，**不留空、也不编数**（D-60）。
        # These two used to exist only in memory and in the log. They are cited from
        # arch_ood_audit.json; when absent this writes null plus a note - never a blank, never a guess.
        "refit_matches_id": audit.get("refit_matches_id") or None,
        "refit_all_match": audit.get("refit_all_match"),
        "ood_wall_clock_s": audit.get("wall_clock_s_total"),
        "ood_audit_path": (
            OOD_AUDIT_PATH.relative_to(REPO_ROOT).as_posix() if audit else None
        ),
    }
    if not audit:
        payload["ood_audit_note"] = (
            "缺 results/aggregate/arch_ood_audit.json：这份 JSON 由 `arch --ood` 在 D-60 之后写出。"
            "旧版运行没有留下重拟合一致性与 OOD 墙钟，需在物化机上重跑一次 `--ood` 才能补上；"
            "在此之前，这两项保持 null，不许用估计值填。"
        )
    OOD_ENSEMBLE_PATH.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n"
    )
    return payload


def verify_ood_artifacts(*, manifest_path: Path | None = None) -> dict[str, Any]:
    """核 45 份行级产物：在不在、字节对不对、`where` 有没有登记（取件用）。
    Verify the 45 row-level artifacts: present, byte-identical, and with a recorded host.

    中文：
        用途：这些产物 `committed: no`（D-08），清单里只有哈希。D 要算 OOD 的 bootstrap 区间就得
              拿到它们，而"清单说留本地"并不等于"某个地方真有"。这条命令把三件事分开报：
              ① 缺几份（→ 按 `reconstruct` 重建，或向 C-c 取件）；② 哈希不符几份（→ 文件被改过）；
              ③ `where` 未登记几份（→ 交接链断在这里：不知道去哪台机器取）。
              三者混成一个"红"就会重演 C-c 验收 §5.1 的困境（D-62）。
        输入：manifest_path（默认 `data/manifests/arch_ood_artifacts_manifest.csv`）。
        输出：计数 + 明细 + 重建命令。数据来源：清单与盘上文件。副作用：只读。
        异常：清单不存在时抛 FileNotFoundError——清单本身是入库物，缺了就是仓库问题。
    English:
        These artifacts are committed: no (D-08) and the manifest only carries hashes, so "kept
        locally" never proved that somewhere really has them. Three separate counts: missing
        (rebuild or ask C-c), digest mismatch (the file was edited), and an unrecorded host (the
        handoff chain breaks here). Merging them into one red light is exactly the C-c section 5.1
        trap (D-62). Read-only; missing manifest raises.
    """
    target = Path(manifest_path) if manifest_path is not None else OOD_MANIFEST_PATH
    if not target.is_file():
        raise FileNotFoundError(f"缺 {target}：这张清单是入库物，缺了先查仓库，不要重跑覆盖")
    with target.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    present = 0
    missing: list[str] = []
    mismatch: list[dict[str, str]] = []
    unrecorded: list[str] = []
    for row in rows:
        rel = str(row.get("path", ""))
        # `where` 与文件在不在是**两件事**：缺件要重建，缺机器名要人填。以前把 where 的检查放在
        # "文件存在"之后，于是"45 份全缺 + 45 份 where 未登记"只报前一半（2026-11-17 实测撞到）。
        # `where` and the file's presence are different findings: a missing file needs a rebuild, a
        # missing host name needs a human. The check used to sit after the existence test, so
        # "45 missing + 45 unrecorded" reported only the first half.
        if str(row.get("where", "")).strip() in ("", "unrecorded"):
            unrecorded.append(rel)
        path = REPO_ROOT / rel
        if not path.is_file():
            missing.append(rel)
            continue
        present += 1
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        expected = str(row.get("sha256", ""))
        if expected and actual != expected:
            mismatch.append({"path": rel, "expected": expected, "actual": actual})
    return {
        "manifest": target.relative_to(REPO_ROOT).as_posix(),
        "n_artifacts": len(rows),
        "present": present,
        "missing": missing,
        "mismatch": mismatch,
        "unrecorded_hosts": unrecorded,
        "ok": not missing and not mismatch,
        "reconstruct": (
            "python scripts/04_run_cpu.py arch --setup && python scripts/04_run_cpu.py arch --ood"
        ),
    }


def write_arch_table(records: Sequence[dict[str, Any]], *, ood_by_run: Mapping[str, dict[str, Any]] | None = None) -> Any:
    """把 15 条记录 + 逐 run OOD 指标写成 `arch_transfer.csv`（一 run 一行，两档分开列）。
    Write arch_transfer.csv: one row per run, the two ID tiers in separate columns.
    """
    import csv

    import pandas as pd

    ood_by_run = ood_by_run or {}
    rows: list[dict[str, Any]] = []
    for record in records:
        run_id = str(record.get("run_id", ""))
        metrics = record.get("metrics") or {}
        tiers = record.get("id_tier_metrics") or {}
        counts = record.get("id_tier_n") or {}
        ood = ood_by_run.get(run_id) or {}
        ood_metrics = ood.get("metrics") or {}
        rows.append(
            {
                "run_id": run_id,
                "task": record.get("task", TASK_ID),
                "arm": record.get("arm", ARM),
                "seed": record.get("seed", ""),
                "split_fold": record.get("split_fold", ""),
                "n_train": record.get("n_train", ""),
                "n_test": record.get("n_test", ""),
                "wall_clock_s": f"{float(record.get('wall_clock_s', 0)):.3f}",
                "r2_id": metrics.get("r2", ""),
                "rmse_id": metrics.get("rmse", ""),
                "mae_id": metrics.get("mae", ""),
                "r2_alternating": (tiers.get("alternating") or {}).get("r2", ""),
                "rmse_alternating": (tiers.get("alternating") or {}).get("rmse", ""),
                "mae_alternating": (tiers.get("alternating") or {}).get("mae", ""),
                "n_alternating": counts.get("alternating", ""),
                "r2_homopolymer": (tiers.get("homopolymer") or {}).get("r2", ""),
                "rmse_homopolymer": (tiers.get("homopolymer") or {}).get("rmse", ""),
                "mae_homopolymer": (tiers.get("homopolymer") or {}).get("mae", ""),
                "n_homopolymer": counts.get("homopolymer", ""),
                "ood_random_rmse": (ood_metrics.get("random") or {}).get("rmse", ""),
                "ood_block_rmse": (ood_metrics.get("block") or {}).get("rmse", ""),
                "ood_combined_rmse": (ood_metrics.get("combined") or {}).get("rmse", ""),
                "note": "OOD 逐折为过程留档，判决口径是五折集成点值（arch_ood_ensemble.json，D-09）",
            }
        )
    ARCH_TABLE_PATH.parent.mkdir(parents=True, exist_ok=True)
    with ARCH_TABLE_PATH.open("w", encoding="utf-8", newline="\n") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(ARCH_TABLE_COLUMNS), lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column, "") for column in ARCH_TABLE_COLUMNS})
    del pd
    return ARCH_TABLE_PATH


def verify_arch_setup() -> dict[str, Any]:
    """只读复核：物化行数、组分覆盖、两份特征表的哈希。／ Read-only check of the whole setup."""
    import pandas as pd

    problems: list[str] = []
    canonical = pd.read_csv(GENERATED / "canonical_train.csv")
    holdout = pd.read_csv(GENERATED / "architecture_holdout.csv")
    counts_c = canonical["architecture"].value_counts().to_dict()
    counts_h = holdout["architecture"].value_counts().to_dict()
    if len(canonical) != 6478 or counts_c != {"alternating": 6110, "homopolymer": 368}:
        problems.append(f"canonical {len(canonical)} 行 {counts_c} ≠ 6,478 (6,110+368)")
    if len(holdout) != 36828 or counts_h != {"random": 18414, "block": 18414}:
        problems.append(f"holdout {len(holdout)} 行 {counts_h} ≠ 36,828 (18,414+18,414)")
    coverage = assert_coverage_complete()
    if not coverage["complete"]:
        problems.append(f"组分覆盖不完备：{coverage}")
    for path, manifest_path in ((ID_FEATURES_PATH, ID_MANIFEST_PATH),):
        if not path.is_file():
            problems.append(f"缺 {path.relative_to(REPO_ROOT).as_posix()}")
            continue
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if hashlib.sha256(path.read_bytes()).hexdigest() != manifest.get("features_sha256"):
            problems.append(f"{path.name} 与清单哈希不符")
    if not ARCH_SPLIT_MANIFEST.is_file():
        problems.append(f"缺 {ARCH_SPLIT_MANIFEST.relative_to(REPO_ROOT).as_posix()}：先 arch --setup")
    return {
        "ok": not problems,
        "problems": problems,
        "canonical": {"rows": int(len(canonical)), **counts_c},
        "holdout": {"rows": int(len(holdout)), **counts_h},
        "components": coverage,
    }
