#!/usr/bin/env python
"""派生报表：三张"从已有产物再算一次"的表，不改任何 run、不改任何记录。

Derived report tables: three tables recomputed from existing artifacts. No run is touched, no
record is rewritten.

为什么要单独一个脚本
--------------------
这三张表都有同一个性质：**它们不是实验产物，而是对实验产物的再表述**。
- `spearman.csv`：重尾任务的 R² 由少数极值主导，排序还灵不灵要看 Spearman。事后从
  `predictions/*.parquet` 算，**不写进 96 个 JSON**——写进去要改记录，而记录是协议身份。
- `cost_summary.csv`：`gpu_hours.csv` 里现在同时有 CPU 与 GPU 两种行（922 行），整表求和
  会得到 1.293 h，而真实 GPU 只有 1.034 h（D-37 的同一个错，只是维度从 arm 换成了 device）。
  这里按 **(device, arm) 两个键**分组报，并单独并入 OOD 阶段的墙钟（它不写 run 记录）。
- `n_bins_covered.csv`：低 n 档的差距里混着"这几十个样本没覆盖整个性质区间"，不写出来，
  H4 的过零点会被读成"经典方法不行"。

用法 / Usage
------------
    python scripts/05b_report_tables.py spearman
    python scripts/05b_report_tables.py cost
    python scripts/05b_report_tables.py bins
    python scripts/05b_report_tables.py all
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

import numpy as np  # noqa: E402

AGGREGATE = REPO_ROOT / "results" / "aggregate"
PREDICTIONS = AGGREGATE / "predictions"
MAIN_TABLE = AGGREGATE / "main_table.csv"
GPU_HOURS = AGGREGATE / "gpu_hours.csv"
OOD_AUDIT = AGGREGATE / "arch_ood_audit.json"
OPENPOLY_N_INDEX = AGGREGATE / "openpoly_cpu_n_index.csv"

SPEARMAN_TABLE = AGGREGATE / "spearman.csv"
COST_TABLE = AGGREGATE / "cost_summary.csv"
BINS_TABLE = AGGREGATE / "n_bins_covered.csv"


def _write(path: Path, columns: Sequence[str], rows: Sequence[dict[str, Any]]) -> Path:
    """写 CSV（UTF-8、LF、不留 CRLF）。／ Write a CSV with UTF-8 and LF endings."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(columns), lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column, "") for column in columns})
    return path


# ---- 1. Spearman / Rank correlation ----


def spearman_table() -> Path:
    """按主表的 32 格，逐格算 Spearman 的均值/极值（每格先逐 run 算，再跨 run 汇总）。
    Per main-table cell, report the mean/min/max Spearman computed run by run.

    中文：
        用途：R² 在重尾目标上会被少数极值主导（`pb_t2_permeability` 的 A2 输出甚至是常数，
              R² 为负）。Spearman 只看排序，回答的是"模型把好材料排对了吗"。
              逐 run 算完再平均，**不回填进记录**（记录是冻结的协议身份，D-48）。
        输入：`main_table.csv` 的 `run_ids`。输出：`spearman.csv`。
        数据来源：`predictions/<run_id>.parquet`。副作用：写一张表。异常：缺预测则跳过该 run。
    English:
        R2 on heavy-tailed targets is dominated by a few extremes (permeability's A2 output is even
        constant, so its R2 is negative). Spearman answers "did the model rank materials correctly".
        Computed per run and averaged per cell; never written back into the frozen records.
    """
    from scipy.stats import spearmanr

    from chainworth.arms import conventional as C

    with MAIN_TABLE.open("r", encoding="utf-8", newline="") as handle:
        cells = list(csv.DictReader(handle))
    # 预测文件只有 `row_id, y_pred` 两列（`_schema.json` 的契约），真值要从任务数据取：
    # `row_id` 就是任务行位置，`build_task_data()` 的 `y` 与它同坐标系（行序不重排）。
    # Predictions carry only row_id and y_pred; the truth comes from the task data, whose y shares
    # the same coordinate system (row order is never re-sorted).
    truth: dict[str, Any] = {}
    rows: list[dict[str, Any]] = []
    for cell in cells:
        task = str(cell["task"])
        if task not in truth:
            truth[task] = C.build_task_data(task).y
        labels = truth[task]
        run_ids = [item for item in str(cell.get("run_ids", "")).split(";") if item]
        values: list[float] = []
        n_points = 0
        for run_id in run_ids:
            path = PREDICTIONS / f"{run_id}.parquet"
            if not path.is_file():
                continue
            import pandas as pd

            frame = pd.read_parquet(path)
            index = frame["row_id"].to_numpy(dtype=np.int64)
            y_true = np.asarray(labels, dtype=float)[index]
            y_pred = frame["y_pred"].to_numpy(dtype=float)
            # 常数预测的 Spearman 无定义（分母为 0）：记为 NaN 而不是 0，免得看起来"排序很差"。
            # A constant prediction has an undefined Spearman (zero denominator): NaN, not 0.
            if np.ptp(y_pred) == 0 or np.ptp(y_true) == 0:
                values.append(float("nan"))
            else:
                rho = float(spearmanr(y_true, y_pred).statistic)
                values.append(rho)
            n_points = max(n_points, len(frame))
        finite = [value for value in values if not np.isnan(value)]
        rows.append(
            {
                "task": cell["task"],
                "arm": cell["arm"],
                "n_runs": len(values),
                "n_undefined": len(values) - len(finite),
                "n_test": n_points,
                "spearman_mean": f"{float(np.mean(finite)):.6f}" if finite else "",
                "spearman_min": f"{float(np.min(finite)):.6f}" if finite else "",
                "spearman_max": f"{float(np.max(finite)):.6f}" if finite else "",
                "note": (
                    "逐 run 算再平均；undefined = 预测或真值为常数（Spearman 分母为 0），"
                    "记 NaN 不记 0（D-64）"
                ),
            }
        )
    return _write(
        SPEARMAN_TABLE,
        ("task", "arm", "n_runs", "n_undefined", "n_test", "spearman_mean", "spearman_min",
         "spearman_max", "note"),
        rows,
    )


# ---- 2. 成本总账 / Cost ledger ----


def cost_table() -> Path:
    """按 (device, arm) 两个键汇总墙钟，并单独并入 OOD 阶段。
    Aggregate wall clock by (device, arm) and add the OOD stage separately.

    中文：
        用途：把 D-37 的"按 arm 筛"升级成"按 **device 与 arm 两个键**筛"。`gpu_hours.csv` 现在
              同时装 CPU 与 GPU 行：整表求和 1.293 h，而 `device == cuda` 只有 1.034 h。
              OOD 不写 run 记录（一次 run 一行装不下），墙钟在 `arch_ood_audit.json` 里，
              这里作为独立 scope 并进来，否则 D 报总成本时会把最重的一步整块漏掉（D-64）。
        输入：`gpu_hours.csv`；`arch_ood_audit.json`（可缺）。输出：`cost_summary.csv`。
        数据来源：两张产物。副作用：写一张表。异常：无（缺文件就少一行）。
    English:
        Upgrades D-37's "filter by arm" to "filter by device **and** arm". gpu_hours.csv now holds
        both CPU and GPU rows (whole-table sum 1.293 h vs 1.034 h of real GPU). The OOD stage
        writes no run record, so its wall clock comes from arch_ood_audit.json as its own scope.
    """
    by_key: dict[tuple[str, str], list[float]] = defaultdict(list)
    if GPU_HOURS.is_file():
        with GPU_HOURS.open("r", encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle):
                device = str(row.get("device", "")).strip() or "unknown"
                arm = str(row.get("arm", "")).strip() or "unknown"
                by_key[(device, arm)].append(float(row.get("wall_clock_s") or 0.0))
    rows: list[dict[str, Any]] = []
    for (device, arm), seconds in sorted(by_key.items()):
        rows.append(
            {
                "scope": "runs",
                "device": device,
                "arm": arm,
                "n_rows": len(seconds),
                "wall_clock_s": f"{sum(seconds):.3f}",
                "hours": f"{sum(seconds) / 3600.0:.6f}",
                "note": "来自 gpu_hours.csv，按 device + arm 分组（D-64）",
            }
        )
    if OOD_AUDIT.is_file():
        payload = json.loads(OOD_AUDIT.read_text(encoding="utf-8"))
        seconds = float(payload.get("wall_clock_s_total") or 0.0)
        rows.append(
            {
                "scope": "ood",
                "device": "cpu",
                "arm": str(payload.get("arm") or "a2_conventional"),
                "n_rows": int(payload.get("n_runs_timed") or 0),
                "wall_clock_s": f"{seconds:.3f}",
                "hours": f"{seconds / 3600.0:.6f}",
                "note": "OOD 三段式（重拟合+组分预测+官方合成），不写 run 记录（D-60）",
            }
        )
    else:
        rows.append(
            {
                "scope": "ood",
                "device": "cpu",
                "arm": "a2_conventional",
                "n_rows": 0,
                "wall_clock_s": "",
                "hours": "",
                "note": "缺 arch_ood_audit.json：需在物化机上跑一次 arch --ood（D-60）",
            }
        )
    return _write(
        COST_TABLE,
        ("scope", "device", "arm", "n_rows", "wall_clock_s", "hours", "note"),
        rows,
    )


# ---- 3. 箱覆盖 / Bin coverage ----


def bins_table() -> Path:
    """OpenPoly 每个 n 档的箱覆盖分布：多少个格子的 `n_bins_covered` 不到 10。
    Bin coverage per n level: how many cells cover fewer than ten bins.

    中文：
        用途：H4 的过零点落在 n=30，而 n=30 恰好是箱覆盖最差的一档（13/15 不足 10 箱，最低 7）。
              这条不写进交付文档，结论会被读成"经典方法在低 n 端不行"，而真实原因里混着
              "这几十个样本没覆盖整个性质区间"（C-b 验收 §6-1）。
        输入：`openpoly_cpu_n_index.csv`。输出：`n_bins_covered.csv`。
        数据来源：该索引表。副作用：写一张表。异常：无（缺表则报错退出）。
    English:
        H4's crossing sits at n=30, which is also the worst-covered tier (13/15 under ten bins,
        minimum seven). Without this, "classical arms fail at low n" conflates coverage with skill.
    """
    if not OPENPOLY_N_INDEX.is_file():
        raise SystemExit(f"缺 {OPENPOLY_N_INDEX.relative_to(REPO_ROOT).as_posix()}")
    by_level: dict[tuple[str, int], list[int]] = defaultdict(list)
    with OPENPOLY_N_INDEX.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            # `n_level` 在饱和档是字符串 "all"：按 (档名, 实际训练行数) 分组，按行数排序，
            # 这样 "all" 落在最后而不是被 int() 炸掉。
            # The saturated tier is the string "all": group by (label, actual n_train) and sort by
            # the row count so "all" lands last instead of blowing up int().
            by_level[(str(row["n_level"]), int(row["n_train"]))].append(int(row["n_bins_covered"]))
    rows: list[dict[str, Any]] = []
    for (level, n_train) in sorted(by_level, key=lambda key: (key[1], key[0])):
        covered = by_level[(level, n_train)]
        rows.append(
            {
                "n_level": level,
                "n_train": n_train,
                "n_cells": len(covered),
                "n_under_10_bins": sum(1 for value in covered if value < 10),
                "n_under_8_bins": sum(1 for value in covered if value < 8),
                "min_bins": min(covered),
                "median_bins": f"{float(np.median(covered)):.1f}",
                "max_bins": max(covered),
                "note": "分母是 10 个官方分箱；低 n 档的差距里混着覆盖不足（C-b 验收 §6-1）",
            }
        )
    return _write(
        BINS_TABLE,
        ("n_level", "n_train", "n_cells", "n_under_10_bins", "n_under_8_bins", "min_bins",
         "median_bins", "max_bins", "note"),
        rows,
    )


def main(argv: Sequence[str] | None = None) -> int:
    """命令行入口。／ Command-line entry point."""
    parser = argparse.ArgumentParser(description="派生报表：Spearman / 成本总账 / 箱覆盖")
    parser.add_argument("command", choices=("spearman", "cost", "bins", "all"))
    args = parser.parse_args(argv)
    jobs = {
        "spearman": ("Spearman", spearman_table),
        "cost": ("成本总账", cost_table),
        "bins": ("箱覆盖", bins_table),
    }
    chosen = list(jobs) if args.command == "all" else [args.command]
    for name in chosen:
        label, function = jobs[name]
        path = function()
        print(f"{label}: {path.relative_to(REPO_ROOT).as_posix()}")
    return 0


if __name__ == "__main__":  # pragma: no cover - 命令行入口
    raise SystemExit(main())
