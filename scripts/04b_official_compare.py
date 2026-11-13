"""阶段入口 04b：与官方 Polymer Benchmark 2026 的分数对齐（C-a-4 的证据生成）。
Stage entry 04b: align our classical arms with the official Polymer Benchmark 2026 scores.

中文：
    用途：C-a 要回答"我们比官方差多少、差在哪"。本脚本做两组事，全部可重跑：
          【PolyBench26】① `fetch` —— 从**钉死的 commit** 下载三份官方汇总表与两份任务的划分索引，
                              逐份算 sha256 并登记到 `data/manifests/external_official_sources.csv`；
                        ② `verify-splits` —— 把官方索引 npz 与本地 `splits/polybench26/...` 逐数组比对，
                              回答"是不是同一份划分"（这是能不能比数的前提）；
                        ③ `compare` —— 用我们的预测 parquet 重算逐折 RMSE，和官方 RDKit_RF 逐折并排，
                              落 `results/aggregate/official_baseline_compare.csv` 与
                              `official_baseline_summary.json`。
          【PolyBench】④ `polybench-fetch` —— 从平台公开接口 `/api/com_result_data` 取回 **39 任务 ×
                              30 模型**的排行榜（每格一个 Test R²），登记 sha256，并核 6 个任务的
                              训练/测试行数；
                        ⑤ `polybench-compare` —— 每任务取"平台最佳格"与"最佳树集成格"，与我们的 A2 并排，
                              落 `results/aggregate/polybench_official_compare.csv` 与
                              `polybench_official_summary.json`。
    为什么不用 API：`raw.githubusercontent.com/<repo>/<commit>/<path>` 直接给字节，没有 60 次/小时的限额；
          API 取到的同一文件经 base64 解码后 sha256 与之逐位相同（2026-11-13 实测）。
    两个口径（写进产物，不许事后口头补）：
          · `scaling` —— 同一份 seed_42 五折划分，但官方只用了训练池的前 N 行（Tg N=3000，
            density N=1000）；我们用的是**整个** train_pool（5893 / 1440）。训练量不同，必须先说。
          · `legacy`  —— `published_results/` 的整表五折 CV，官方 README 明写"不是 shared-split
            结果，不要与 scaling_results 混用"。只作参照，不作主对比。
    模型差异（实现层，不是协议层）：官方 RDKit_RF = 200+ 描述符 + RandomForest(100 树) + 中位数填充；
          我们 A2 = 205 描述符 + 2048 位指纹 + XGBoost。比的是"同一划分下的实现差距"。
    依赖：numpy / pyarrow（读预测与特征）；下载用标准库 urllib。
    对外：命令行。产物只写 `results/aggregate/` 与 `data/manifests/`，不碰 `raw_runs/`。

English:
    Purpose: three reproducible steps. `fetch` downloads the official summary tables and the two
    tasks' split indices from a pinned commit and records their sha256; `verify-splits` compares the
    official index arrays with our local `splits/` copies array-by-array; `compare` recomputes
    per-fold RMSE from our prediction parquets and puts it next to the official RDKit_RF per-fold
    numbers.
    Two protocols are written into the artefacts: `scaling` shares the exact seed_42 folds but the
    official run subsampled the training pool (Tg 3000 / density 1000 rows) while we used the whole
    pool (5893 / 1440); `legacy` is the full-dataset 5-fold CV table that its own README says must
    not be treated as shared-split results.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
import sys
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

import numpy as np  # noqa: E402
import pyarrow.parquet as pq  # noqa: E402

LOGGER = logging.getLogger("chainworth.scripts.04b")

RAW_DIR = REPO_ROOT / "data" / "raw" / "external"
MANIFEST_PATH = REPO_ROOT / "data" / "manifests" / "external_official_sources.csv"
OUT_CSV = REPO_ROOT / "results" / "aggregate" / "official_baseline_compare.csv"
OUT_JSON = REPO_ROOT / "results" / "aggregate" / "official_baseline_summary.json"

# ---- PolyBench 平台（39 任务 × 30 模型的官方排行榜）----
# PolyBench platform (the official 39-task x 30-model leaderboard).
POLYBENCH_BASE = "http://202.98.16.11"           # 论文注明的主入口；镜像域名 polybench.ciac.jl.cn 同源
POLYBENCH_PAPER = "10.1021/polymscitech.6c00094"
POLYBENCH_ENDPOINTS = {
    "leaderboard": "/api/com_result_data",       # 每 (dataset, descriptor, algorithm) 一行，含 Test R²/MSE/MAE
    "r2_bands": "/api/com_test_r2",              # 每任务的 R² 三档计数（>0.8 / 0.5–0.8 / <0.5）
}
POLYBENCH_MANIFEST = REPO_ROOT / "data" / "manifests" / "external_polybench_sources.csv"
POLYBENCH_SPLIT_SIZES = RAW_DIR / "polybench_split_sizes.json"
POLYBENCH_CSV = REPO_ROOT / "results" / "aggregate" / "polybench_official_compare.csv"
POLYBENCH_JSON = REPO_ROOT / "results" / "aggregate" / "polybench_official_summary.json"
POLYBENCH_INDEX = REPO_ROOT / "results" / "aggregate" / "a1_a2_index.csv"
# 官方排行榜里哪些算法算"树集成"（A2 的家族）。Which leaderboard algorithms count as tree ensembles.
TREE_MODELS = ("XGBoost", "RF", "CatBoost", "GBDT", "HistGBDT", "ExtraTrees", "AdaBoost", "DecisionTree")

REPO_SLUG = "rlearsch/PolymerBenchmark2026"
PINNED_COMMIT = "83655bc25901a8cabe3a1e5474b7ec26f01b919f"
RAW_BASE = f"https://raw.githubusercontent.com/{REPO_SLUG}/{PINNED_COMMIT}"

# 官方路径 -> 本地文件名。Official path -> local file name.
FILES = {
    "scaling_detailed": "scaling_results/all/scaling_results_detailed.csv",
    "scaling_aggregate": "scaling_results/all/scaling_results_aggregate.csv",
    "legacy_published": "published_results/full_dataset_5fold_cv_rmse.csv",
}

# 本地任务 -> (官方划分目录, 目标名)。Local task -> (official split dir, quantity).
TASKS = {
    "pb26_polymetrix_tg": ("Datasets/scaling_indices/PolyMetriX/Tg/Tg/seed_42", "Tg"),
    "pb26_homopolymer_density": (
        "Datasets/scaling_indices/MD_5000/density/homopolymer_density/seed_42",
        "density",
    ),
}


def _download(url: str, dest: Path) -> tuple[int, str]:
    """下载一份文件并返回 (字节数, sha256)。Download one file; return (n_bytes, sha256)."""
    req = urllib.request.Request(url, headers={"User-Agent": "chainworth/07"})
    with urllib.request.urlopen(req, timeout=120) as resp:
        payload = resp.read()
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(payload)
    return len(payload), hashlib.sha256(payload).hexdigest()


def fetch(force: bool = False) -> list[dict[str, Any]]:
    """取回官方文件并登记来源；已存在且哈希不变则跳过。Fetch and record provenance."""
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    targets: list[tuple[str, str, str]] = [(name, path, Path(path).name) for name, path in FILES.items()]
    for task, (split_dir, _) in TASKS.items():
        for fold in range(5):
            name = f"{task}__fold_{fold:02d}"
            targets.append((name, f"{split_dir}/fold_{fold:02d}.npz", f"{Path(split_dir).parent.name}__fold_{fold:02d}.npz"))

    for name, remote, local_name in targets:
        dest = RAW_DIR / local_name
        if dest.exists() and not force:
            payload = dest.read_bytes()
            n_bytes, digest = len(payload), hashlib.sha256(payload).hexdigest()
            LOGGER.info("skip %s (exists, %d bytes)", local_name, n_bytes)
        else:
            url = f"{RAW_BASE}/{remote}"
            n_bytes, digest = _download(url, dest)
            LOGGER.info("fetched %s (%d bytes)", local_name, n_bytes)
        rows.append(
            {
                "name": name,
                "repo": REPO_SLUG,
                "commit": PINNED_COMMIT,
                "remote_path": remote,
                "local_path": str(dest.relative_to(REPO_ROOT)).replace("\\", "/"),
                "n_bytes": n_bytes,
                "sha256": digest,
                "retrieved_at_utc": stamp,
            }
        )

    MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    with MANIFEST_PATH.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    LOGGER.info("wrote %s (%d rows)", MANIFEST_PATH.relative_to(REPO_ROOT), len(rows))
    return rows


def verify_splits() -> dict[str, Any]:
    """官方索引与本地 splits 逐数组比对。Compare official index arrays with local splits."""
    report: dict[str, Any] = {
        "pinned_commit": PINNED_COMMIT,
        "checked_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "tasks": {},
    }
    for task, (split_dir, _) in TASKS.items():
        local_dir = REPO_ROOT / "splits" / "polybench26" / task / "seed_42"
        per_fold: list[dict[str, Any]] = []
        all_equal = True
        for fold in range(5):
            official = np.load(RAW_DIR / f"{Path(split_dir).parent.name}__fold_{fold:02d}.npz", allow_pickle=True)
            local = np.load(local_dir / f"fold_{fold:02d}.npz", allow_pickle=True)
            keys: dict[str, Any] = {}
            for key in official.files:
                if key not in local.files:
                    keys[key] = {"equal": False, "reason": "missing_locally"}
                    all_equal = False
                    continue
                a, b = np.asarray(official[key]), np.asarray(local[key])
                same = a.shape == b.shape and bool(np.array_equal(a, b))
                keys[key] = {"equal": same, "n": int(a.size)}
                all_equal = all_equal and same
            per_fold.append({"fold": f"fold_{fold:02d}", "arrays": keys})
        report["tasks"][task] = {"official_dir": split_dir, "all_arrays_equal": all_equal, "folds": per_fold}
    return report


def _rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def compare() -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """按折重算我们的 RMSE，并与官方 RDKit_RF 并排。Recompute per-fold RMSE and join the official rows."""
    detailed = list(csv.DictReader((RAW_DIR / "scaling_results_detailed.csv").open(encoding="utf-8")))
    rows_out: list[dict[str, Any]] = []
    summary: dict[str, Any] = {"pinned_commit": PINNED_COMMIT, "tasks": {}}

    for task, (_, quantity) in TASKS.items():
        pack = np.load(REPO_ROOT / "data" / "processed" / "features" / f"{task}.npz", allow_pickle=True)
        y_all = np.asarray(pack["y"], dtype=float)
        pos = {int(r): i for i, r in enumerate(pack["row_id"])}

        official = [
            r
            for r in detailed
            if r["model"] == "RDKit_RF"
            and r["quantity"] == quantity
            and r["polymer_type"] == "homopolymer"
            and r["split"] == "test"
        ]
        if not official:
            raise RuntimeError(f"official rows not found for {task} ({quantity})")
        train_size = max(int(r["train_size"]) for r in official)
        off_by_fold = {int(r["fold"]): float(r["rmse"]) for r in official if int(r["train_size"]) == train_size}
        n_test_official = {int(r["fold"]): int(r["n_samples"]) for r in official if int(r["train_size"]) == train_size}

        train_sizes: dict[int, int] = {}
        for fold in range(5):
            split = np.load(
                REPO_ROOT / "splits" / "polybench26" / task / "seed_42" / f"fold_{fold:02d}.npz", allow_pickle=True
            )
            train_sizes[fold] = int(np.asarray(split["train_pool"]).size)

        ours_by_fold: dict[int, list[float]] = {}
        n_test_ours = 0
        for fold in range(5):
            seed_rmse: list[float] = []
            for seed in range(3):
                path = (
                    REPO_ROOT
                    / "results"
                    / "aggregate"
                    / "predictions"
                    / f"{task}__a2_conventional__fold_{fold:02d}__seed{seed}.parquet"
                )
                table = pq.read_table(path).to_pydict()
                idx = [pos[int(r)] for r in table["row_id"]]
                n_test_ours = len(idx)
                seed_rmse.append(_rmse(y_all[idx], np.asarray(table["y_pred"], dtype=float)))
            ours_by_fold[fold] = seed_rmse

        # 官方 agg 表给的是“折间”均值与总体标准差；用作 2σ 带的定义。
        # The aggregate table reports the mean and population SD across folds; that defines the 2-sigma band.
        folds = sorted(off_by_fold)
        off_mean = float(np.mean([off_by_fold[f] for f in folds]))
        off_sd = float(np.std([off_by_fold[f] for f in folds], ddof=0))
        our_fold_means = {f: float(np.mean(ours_by_fold[f])) for f in folds}
        our_mean = float(np.mean(list(our_fold_means.values())))
        our_sd = float(np.std(list(our_fold_means.values()), ddof=0))

        for fold in folds:
            seeds = ours_by_fold[fold]
            rows_out.append(
                {
                    "task": task,
                    "quantity": quantity,
                    "fold": f"fold_{fold:02d}",
                    "n_test": n_test_ours,
                    "official_model": "RDKit_RF",
                    "official_protocol": "scaling_seed42_5fold",
                    "official_train_size": train_size,
                    "official_n_test": n_test_official[fold],
                    "official_rmse": f"{off_by_fold[fold]:.12g}",
                    "ours_model": "xgboost",
                    "ours_train_size": train_sizes[fold],
                    "ours_rmse_seed0": f"{seeds[0]:.12g}",
                    "ours_rmse_seed1": f"{seeds[1]:.12g}",
                    "ours_rmse_seed2": f"{seeds[2]:.12g}",
                    "ours_rmse_mean": f"{np.mean(seeds):.12g}",
                    "ours_over_official": f"{np.mean(seeds) / off_by_fold[fold]:.6f}",
                }
            )

        summary["tasks"][task] = {
            "quantity": quantity,
            "official_protocol": "scaling_seed42_5fold",
            "official_train_size": train_size,
            "official_rmse_mean_across_folds": off_mean,
            "official_rmse_sd_across_folds": off_sd,
            "two_sigma_band": [off_mean - 2 * off_sd, off_mean + 2 * off_sd],
            "ours_train_size": rows_out[-1]["ours_train_size"],
            "ours_rmse_mean_across_folds": our_mean,
            "ours_rmse_sd_across_folds": our_sd,
            "ours_inside_two_sigma": bool(off_mean - 2 * off_sd <= our_mean <= off_mean + 2 * off_sd),
            "n_folds": len(folds),
        }

    # 次级参照：legacy 整表五折 CV（官方 README 明写不可与 scaling 混用）。
    # Secondary reference: the legacy full-dataset 5-fold CV table (its README forbids mixing it with scaling).
    legacy_rows = list(csv.DictReader((RAW_DIR / "full_dataset_5fold_cv_rmse.csv").open(encoding="utf-8")))
    for task, (_, quantity) in TASKS.items():
        for r in legacy_rows:
            if r["property"] == quantity and r["model"] == "RDKit_RF" and r["polymer_architecture"] == "homopolymer":
                summary["tasks"][task]["legacy_reference"] = {
                    "protocol": r["protocol"],
                    "metric": r["metric"],
                    "dataset_size": int(r["dataset_size"]),
                    "rmse_mean": float(r["rmse_mean"]),
                    "rmse_std": float(r["rmse_std"]),
                    "comparable": False,
                    "why": "legacy full-dataset 5-fold CV; its README says it is not shared-split results",
                }

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with OUT_CSV.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows_out[0].keys()))
        writer.writeheader()
        writer.writerows(rows_out)
    OUT_JSON.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    LOGGER.info("wrote %s (%d rows) and %s", OUT_CSV.relative_to(REPO_ROOT), len(rows_out), OUT_JSON.relative_to(REPO_ROOT))
    return rows_out, summary


# ---- 3. PolyBench：平台排行榜 / The platform leaderboard ----


def _polybench_tasks() -> dict[str, str]:
    """本地 6 个主表任务 → 平台的数据集名（从 configs 读，不在这里重抄一份）。
    The six main-table tasks mapped to the platform's dataset names, read from configs.
    """
    a2 = yaml.safe_load((REPO_ROOT / "configs" / "arms" / "a2_conventional.yaml").read_text(encoding="utf-8"))
    mapping: dict[str, str] = {}
    for task in a2["runs"]["polybench"]["tasks"]:
        cfg = yaml.safe_load((REPO_ROOT / "configs" / "tasks" / f"{task}.yaml").read_text(encoding="utf-8"))
        mapping[str(task)] = str(cfg["dataset_name"])
    return mapping


def _api_json(path: str, params: dict[str, Any] | None = None, *, timeout: int = 180) -> Any:
    """GET 一个平台接口并解析 JSON；失败时把 URL 与状态码一起报出来。
    GET a platform endpoint and parse JSON; on failure report the URL and status together.
    """
    url = POLYBENCH_BASE + path
    if params:
        url = f"{url}?{urllib.parse.urlencode(params)}"
    req = urllib.request.Request(url, headers={"User-Agent": "chainworth/04b"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except Exception as exc:                              # pragma: no cover - 网络/平台不可达
        raise RuntimeError(f"PolyBench 接口取数失败 {url}: {type(exc).__name__}: {exc}") from exc


def polybench_fetch(force: bool = False) -> list[dict[str, Any]]:
    """取回平台排行榜与 R² 分档表，并核 6 个任务的训练/测试行数。
    Fetch the leaderboard and the R²-band table, and check the six tasks' train/test sizes.
    """
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    manifest: list[dict[str, Any]] = []

    payloads: dict[str, Any] = {}
    for name, endpoint in POLYBENCH_ENDPOINTS.items():
        dest = RAW_DIR / f"polybench_{name}.json"
        if dest.exists() and not force:
            body = dest.read_bytes()
            LOGGER.info("skip %s (exists, %d bytes)", dest.name, len(body))
        else:
            req = urllib.request.Request(POLYBENCH_BASE + endpoint, headers={"User-Agent": "chainworth/04b"})
            with urllib.request.urlopen(req, timeout=300) as resp:
                body = resp.read()
            dest.write_bytes(body)
            LOGGER.info("fetched %s (%d bytes)", dest.name, len(body))
        payloads[name] = json.loads(body.decode("utf-8"))
        manifest.append(
            {
                "name": name,
                "endpoint": POLYBENCH_BASE + endpoint,
                "local_path": str(dest.relative_to(REPO_ROOT)).replace("\\", "/"),
                "n_bytes": len(body),
                "sha256": hashlib.sha256(body).hexdigest(),
                "retrieved_at_utc": stamp,
                "paper_doi": POLYBENCH_PAPER,
            }
        )

    # 划分大小核对：n_samples 是**数据集级**属性，任取一个能查到的格子即可。
    # 排行榜与预测接口的模型名不完全一致（排行榜写 HistGBDT，接口写 HistGB），所以按候选表逐个试，
    # 第一个 200 的组合即为该任务的探针；全部失败只记错误，不让整轮取数挂掉。
    # Split-size check: n_samples is a dataset-level property, so any working cell answers it. The
    # leaderboard and the predictions API spell some model names differently (HistGBDT vs HistGB),
    # hence the candidate list; a total failure is recorded instead of aborting the whole fetch.
    sizes: dict[str, Any] = {}
    probes = [("XGBoost", "Mordred"), ("RF", "Mordred"), ("XGBoost", "RDKit_512"), ("RF", "RDKit_512")]
    for task, dataset in _polybench_tasks().items():
        rows = [r for r in payloads["leaderboard"] if r["Dataset"] == dataset]
        if not rows:
            sizes[task] = {"dataset": dataset, "found": False, "error": "dataset absent from leaderboard"}
            LOGGER.warning("%s: %r not in leaderboard", task, dataset)
            continue
        entry: dict[str, Any] = {
            "dataset": dataset,
            "found": True,
            "n_leaderboard_rows": len(rows),
            "split_sizes_from": "GET /api/predictions/data (n_samples)",
        }
        for algorithm, descriptor in probes:
            query = {"algorithm": algorithm, "descriptor": descriptor, "dataset": dataset}
            try:
                counts = {s: int(_api_json("/api/predictions/data", {**query, "split": s})["n_samples"])
                          for s in ("train", "test")}
            except RuntimeError as exc:
                entry.setdefault("probe_failures", []).append(
                    f"{algorithm}+{descriptor}: {str(exc).splitlines()[0]}"
                )
                continue
            entry.update({"probe_algorithm": algorithm, "probe_descriptor": descriptor,
                          "n_train": counts["train"], "n_test": counts["test"]})
            break
        sizes[task] = entry
        if "n_test" in entry:
            LOGGER.info("%s: train=%d test=%d", task, entry["n_train"], entry["n_test"])
        else:
            LOGGER.warning("%s: no probe combination answered; see probe_failures", task)
    POLYBENCH_SPLIT_SIZES.parent.mkdir(parents=True, exist_ok=True)
    POLYBENCH_SPLIT_SIZES.write_text(json.dumps(sizes, ensure_ascii=False, indent=2), encoding="utf-8", newline="\n")

    POLYBENCH_MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    with POLYBENCH_MANIFEST.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(manifest[0].keys()))
        writer.writeheader()
        writer.writerows(manifest)
    LOGGER.info("wrote %s (%d rows)", POLYBENCH_MANIFEST.relative_to(REPO_ROOT), len(manifest))
    return manifest


def polybench_compare() -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """把平台排行榜与我们的 A1/A2 并排：每个任务一行，官方取"最佳单元格"与"最佳树集成"。
    Put the leaderboard next to our A1/A2: one row per task, the official side being the best cell
    and the best tree ensemble.
    """
    # 官方的标签尺度在 SI Table S27 的 `target_scale` 列（raw / ln）。R² 对平移不变、对
    # **非线性变换不不变**，所以 ln 尺度上的官方 R² 不能与我们在原始尺度上的 R² 相减。
    # The official label scale is SI Table S27's `target_scale` column (raw / ln). R² is invariant
    # to shifts but not to nonlinear transforms, so an ln-space official R² cannot be subtracted
    # from our raw-space R².
    from chainworth.io.table_s27 import ROWS as S27_ROWS

    target_scale = {row.task_id: row.target_scale for row in S27_ROWS}

    leaderboard = json.loads((RAW_DIR / "polybench_leaderboard.json").read_text(encoding="utf-8"))
    bands = {b["Dataset"]: b for b in json.loads((RAW_DIR / "polybench_r2_bands.json").read_text(encoding="utf-8"))}
    sizes = json.loads(POLYBENCH_SPLIT_SIZES.read_text(encoding="utf-8"))
    tasks = _polybench_tasks()
    ours: dict[str, dict[str, list[float]]] = {}
    n_test_ours: dict[str, int] = {}
    with POLYBENCH_INDEX.open(encoding="utf-8", newline="") as fh:
        for row in csv.DictReader(fh):
            if row["task"] not in tasks:
                continue
            ours.setdefault(row["task"], {}).setdefault(row["arm"], []).append(float(row["final_r2"]))
            n_test_ours[row["task"]] = int(row["n_test"])

    rows_out: list[dict[str, Any]] = []
    summary: dict[str, Any] = {"paper_doi": POLYBENCH_PAPER, "tasks": {}}
    for task, dataset in tasks.items():
        cells = [r for r in leaderboard if r["Dataset"] == dataset]
        if not cells:
            raise RuntimeError(f"排行榜里没有 {dataset!r}（{task}）：平台改了数据集名？")
        best = max(cells, key=lambda r: float(r["Test R²"]))
        trees = [r for r in cells if r["Algorithm"] in TREE_MODELS]
        best_tree = max(trees, key=lambda r: float(r["Test R²"]))
        tree_mordred = [r for r in trees if r["Descriptor"] == "Mordred"]
        best_tree_mordred = max(tree_mordred, key=lambda r: float(r["Test R²"])) if tree_mordred else None
        xgb_mordred = next(
            (r for r in cells if r["Algorithm"] == "XGBoost" and r["Descriptor"] == "Mordred"), None
        )
        a2 = ours.get(task, {}).get("a2_conventional", [])
        a1 = ours.get(task, {}).get("a1_floor", [])
        size = sizes.get(task, {})
        scale = target_scale.get(dataset, "unknown")
        comparable = scale == "raw"
        # 平台另发一张分档计数表；用它对账"287 格是否都拿到了"——只核行数不够，核分布才够。
        # The platform also publishes band counts; reconciling them proves all 287 cells arrived
        # intact (a row count alone would not).
        r2s = [float(r["Test R²"]) for r in cells]
        recomputed = {
            "gt0.8": sum(1 for v in r2s if v > 0.8),
            "0.5to0.8": sum(1 for v in r2s if 0.5 <= v <= 0.8),
            "lt0.5": sum(1 for v in r2s if v < 0.5),
        }
        band = bands.get(dataset)
        band_ok = bool(band) and (
            recomputed["gt0.8"] == int(band["Test R² > 0.8"])
            and recomputed["0.5to0.8"] == int(band["0.5 <= Test R² < 0.8"])
            and recomputed["lt0.5"] == int(band["Test R² < 0.5"])
        )
        rows_out.append(
            {
                "task": task,
                "dataset": dataset,
                "official_target_scale": scale,
                "ours_target_scale": "raw",
                "r2_comparable": comparable,
                "band_counts_match": band_ok,
                "official_best_r2": f"{float(best['Test R²']):.6f}",
                "official_best_descriptor": best["Descriptor"],
                "official_best_algorithm": best["Algorithm"],
                "official_best_tree_r2": f"{float(best_tree['Test R²']):.6f}",
                "official_best_tree_descriptor": best_tree["Descriptor"],
                "official_best_tree_algorithm": best_tree["Algorithm"],
                "official_xgboost_mordred_r2": (
                    f"{float(xgb_mordred['Test R²']):.6f}" if xgb_mordred else ""
                ),
                "official_best_tree_mordred_r2": (
                    f"{float(best_tree_mordred['Test R²']):.6f}" if best_tree_mordred else ""
                ),
                "official_n_train": size.get("n_train", ""),
                "official_n_test": size.get("n_test", ""),
                "ours_n_test": n_test_ours.get(task, ""),
                "split_size_match": bool(size.get("n_test") == n_test_ours.get(task)),
                "ours_a2_r2_mean": f"{float(np.mean(a2)):.6f}" if a2 else "",
                "ours_a2_r2_sd": f"{float(np.std(a2, ddof=0)):.6f}" if a2 else "",
                "ours_a1_r2_mean": f"{float(np.mean(a1)):.6f}" if a1 else "",
                "ours_minus_best_tree": (
                    f"{float(np.mean(a2)) - float(best_tree['Test R²']):+.6f}"
                    if a2 and comparable else ""
                ),
            }
        )

        summary["tasks"][task] = {
            "dataset": dataset,
            "official_target_scale": scale,
            "ours_target_scale": "raw",
            "r2_comparable": comparable,
            "n_leaderboard_cells": len(cells),
            "band_check": {
                "match": band_ok,
                "official": band,
                "recomputed_from_leaderboard": recomputed,
            },
            "official_best": {
                "r2": float(best["Test R²"]),
                "descriptor": best["Descriptor"],
                "algorithm": best["Algorithm"],
            },
            "official_best_tree": {
                "r2": float(best_tree["Test R²"]),
                "descriptor": best_tree["Descriptor"],
                "algorithm": best_tree["Algorithm"],
            },
            "official_xgboost_mordred_r2": float(xgb_mordred["Test R²"]) if xgb_mordred else None,
            "split_size_check": size,
            "ours_n_test": n_test_ours.get(task),
            "split_size_match": bool(size.get("n_test") == n_test_ours.get(task)),
            "ours_a2_r2": {
                "mean": float(np.mean(a2)) if a2 else None,
                "sd": float(np.std(a2, ddof=0)) if a2 else None,
                "n_runs": len(a2),
            },
            "ours_a1_r2": {
                "mean": float(np.mean(a1)) if a1 else None,
                "sd": float(np.std(a1, ddof=0)) if a1 else None,
                "n_runs": len(a1),
            },
        }

    summary["source"] = {
        "version": {
            "platform": POLYBENCH_BASE,
            "endpoint": POLYBENCH_ENDPOINTS["leaderboard"],
            "paper_doi": POLYBENCH_PAPER,
            "retrieved_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "sha256": hashlib.sha256((RAW_DIR / "polybench_leaderboard.json").read_bytes()).hexdigest(),
        },
        "models": "平台排行榜的 30 个基线模型 × 7 类重复单元编码（每格一个 Test R²）",
        "split": "官方固定 70/30（平台 test-download 接口，唯一一份划分）；本地快照 splits/polybench/<task>/official_70_30.json",
        "metric": "Test R²（平台 /api/com_result_data 列名）",
    }
    summary["caveats"] = [
        "平台的评分代码未发布：Test R² 按 sklearn r2_score 的同一公式理解，未逐行核对。",
        "行身份没法像 PB26 那样逐数组核：平台 /api/predictions/data 只回位置序号 index（0..n-1），不回原始行号；"
        "能核的是测试集大小（6/6 与本地相等，见 split_size_check）与划分来源（同一 test-download 接口）。",
        "平台模型的特征实现与超参由平台固定（Mordred 等编码）；我们的 A2 = 205 RDKit 描述符 + 2048 位 Morgan + "
        "30 次 Murcko 分组 CV 选参，是同一家族的不同实现，不是同一条管线。",
        "Tg/Tm 我们按 D-15 把 ℃ 换成 K 训练；R² 对标签平移不变，所以 R² 可比，RMSE 不可直接比。",
        "标签尺度：SI Table S27 的 target_scale 列写明官方对 Gas permeability / Tensile modulus / "
        "Dielectric loss factor / IWLT 用 ln(y)。我们的 A2 在原始空间训练（Q-10 待全组裁决），"
        "所以这四行的官方 R² 与我们的 R² **不是同一个尺度**，r2_comparable=False，"
        "ours_minus_best_tree 留空；只有 Tg/Tm 两行可以相减。",
    ]
    summary["n_comparable_tasks"] = sum(1 for t in summary["tasks"].values() if t["r2_comparable"])

    POLYBENCH_CSV.parent.mkdir(parents=True, exist_ok=True)
    with POLYBENCH_CSV.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows_out[0].keys()), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows_out)
    POLYBENCH_JSON.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n")
    LOGGER.info("wrote %s and %s", POLYBENCH_CSV.relative_to(REPO_ROOT), POLYBENCH_JSON.relative_to(REPO_ROOT))
    return rows_out, summary


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Align C-a classical arms with official PB26 scores.")
    parser.add_argument(
        "action",
        choices=[
            "fetch",
            "verify-splits",
            "compare",
            "all",
            "polybench-fetch",
            "polybench-compare",
            "polybench",
        ],
    )
    parser.add_argument("--force", action="store_true", help="re-download even if the file exists")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING, format="%(message)s")

    if args.action in ("fetch", "all"):
        fetch(force=args.force)
    if args.action in ("verify-splits", "all"):
        report = verify_splits()
        out = REPO_ROOT / "results" / "aggregate" / "official_split_check.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        ok = all(v["all_arrays_equal"] for v in report["tasks"].values())
        print(f"split identity: {'ALL ARRAYS EQUAL' if ok else 'MISMATCH'}")
        for task, info in report["tasks"].items():
            print(f"  {task}: all_arrays_equal={info['all_arrays_equal']}")
    if args.action in ("compare", "all"):
        _, summary = compare()
        for task, info in summary["tasks"].items():
            print(
                f"{task}: official {info['official_rmse_mean_across_folds']:.6g}"
                f"±{info['official_rmse_sd_across_folds']:.3g} (train {info['official_train_size']})"
                f" vs ours {info['ours_rmse_mean_across_folds']:.6g}"
                f" (train {info['ours_train_size']}); inside 2sigma={info['ours_inside_two_sigma']}"
            )
    if args.action in ("polybench-fetch", "polybench"):
        polybench_fetch(force=args.force)
    if args.action in ("polybench-compare", "polybench"):
        rows, summary = polybench_compare()
        for row in rows:
            print(
                f"{row['task']:<24} [{row['official_target_scale']}] "
                f"official best {row['official_best_r2']} "
                f"({row['official_best_algorithm']}/{row['official_best_descriptor']})"
                f" | best tree {row['official_best_tree_r2']} "
                f"({row['official_best_tree_algorithm']}/{row['official_best_tree_descriptor']})"
                f" | ours A2 {row['ours_a2_r2_mean']}"
                f" | n_test {row['ours_n_test']}={row['official_n_test']} "
                f"match={row['split_size_match']}"
            )
        matched = sum(1 for info in summary["tasks"].values() if info["split_size_match"])
        print(f"split size match: {matched}/{len(summary['tasks'])} tasks")
        print(f"R2 comparable to ours (official target_scale=raw): "
              f"{summary['n_comparable_tasks']}/{len(summary['tasks'])} tasks")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
