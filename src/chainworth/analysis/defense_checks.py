"""答辩补算：审阅问题清单里"必须现算才有答案"的那几个数。
Defence checks: the numbers a reviewer can only settle by recomputation.

中文：
    用途：外部审阅（2026-11-18）提出的一批质疑中，有 7 个不能靠改措辞回答，只能现算：
      ① **12/19 是不是"稳定格"计数**——若把分母 < 0.05 的行剔掉，负号格子会变成多少？
      ② **n = 6 的 Spearman 到底有多不确定**——精确置换 p 值（720 种排列）与按任务重采样的 ρ 区间。
      ③ **预注册说的"十倍"与实跑的 74 倍是不是同一件事**——300 → 3,000 那一段的斜率是多少？
      ④ **κ 的负号是否只活在尾巴上**——首十倍段与尾段的斜率分开报。
      ⑤ **A3 与 A4 在共同任务上是否同号**——4 处翻号里有多少发生在 |Λ| < 0.05 的格子里。
      ⑥ **n* 是插值还是拟合**——在 log n 上线性插值的过零点，以及它落在哪个设计区间里。
      ⑦ **噪声地板怎么算的**——两种零信息条件取较大者（此前的表在 modulus 上取了较小者）。
    纪律：
      - 只读 `results/aggregate/` 的既有表，**不重算任何 Λ**，也不改任何 `raw_runs/`；
      - 每个数带 `source` 字段，指回它是从哪张表的哪一列算出来的；
      - 算不出来就写 `None` + `reason`，不填 0（D-73 同型纪律）。
    输出：`results/aggregate/defense_checks.json`、`defense_cells.csv`。

English:
    Purpose: seven of the review questions cannot be answered by wording, only by recomputation:
    (1) is "12 of 19" a stable-cell count; (2) exact permutation p for Spearman at n = 6 and a
    task-level bootstrap of rho; (3) does the pre-registered "tenfold" agree with the executed
    74x span; (4) is kappa's negative sign confined to the tail; (5) do A3 and A4 share signs on
    their common tasks, and how many flips sit on near-zero cells; (6) is n-star an interpolation and
    inside which design bracket; (7) how the zero-information floor is defined. Everything reads
    the aggregate tables only - no Lambda is recomputed, no run record is touched, and anything
    undefined is written as null with a reason rather than zero.
"""

from __future__ import annotations

import csv
import itertools
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
LAMBDA_PATH = REPO_ROOT / "results" / "aggregate" / "lambda_ci.csv"
ORDERING_PATH = REPO_ROOT / "results" / "aggregate" / "h1_ordering.csv"
KAPPA_CELLS_PATH = REPO_ROOT / "results" / "aggregate" / "kappa_cells.csv"
FLOOR_PATH = REPO_ROOT / "results" / "aggregate" / "null_calibration_summary.json"
CELLS_OUT = REPO_ROOT / "results" / "aggregate" / "defense_cells.csv"
SUMMARY_OUT = REPO_ROOT / "results" / "aggregate" / "defense_checks.json"

DENOMINATOR_FLOOR = 0.05
N_BOOT = 2000
BOOT_SEED = 20261118


# --------------------------------------------------------------------------- 基础统计
def average_ranks(values: Sequence[float]) -> list[float]:
    """平均秩（并列取中位秩）。Tied values share their mid-rank."""
    order = sorted(range(len(values)), key=lambda i: values[i])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and values[order[j + 1]] == values[order[i]]:
            j += 1
        mid = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            ranks[order[k]] = mid
        i = j + 1
    return ranks


def spearman_rho(x: Sequence[float], y: Sequence[float]) -> float:
    """Spearman ρ = 平均秩上的 Pearson 相关（并列用中位秩，与 `h1_ordering.csv` 同口径）。

    Spearman rho as Pearson correlation of mid-ranks; this is the convention that reproduces
    the published `spearman_rho` column, which plain `1 - 6Σd²/n(n²-1)` does not when tiers tie.
    """
    if len(x) != len(y) or len(x) < 2:
        raise ValueError("rho needs two equal-length vectors of length >= 2")
    rx = np.asarray(average_ranks(x), dtype=float)
    ry = np.asarray(average_ranks(y), dtype=float)
    rx -= rx.mean()
    ry -= ry.mean()
    denom = math.sqrt(float((rx**2).sum()) * float((ry**2).sum()))
    if denom == 0.0:
        return 0.0
    return float((rx * ry).sum() / denom)


def permutation_p(rho_obs: float, y: Sequence[float], *, n_perm: int | None = None,
                  seed: int = BOOT_SEED) -> dict[str, Any]:
    """双侧置换 p 值。n ≤ 7 时穷举全部排列（6! = 720），否则随机抽 `n_perm` 个。

    Two-sided permutation p. Exhaustive for n <= 7 (6! = 720), Monte Carlo otherwise.
    """
    n = len(y)
    total = math.factorial(n)
    exhaustive = total <= 5040
    if exhaustive:
        perms: Iterable[tuple[int, ...]] = itertools.permutations(range(n))
        n_used = total
    else:
        if n_perm is None:
            n_perm = 20000
        rng = np.random.default_rng(seed)
        perms = (tuple(rng.permutation(n).tolist()) for _ in range(n_perm))
        n_used = n_perm
    base = list(y)
    hits = 0
    for perm in perms:
        if abs(spearman_rho(base, [base[i] for i in perm])) >= abs(rho_obs) - 1e-12:
            hits += 1
    return {"p_two_sided": hits / n_used, "n_permutations": n_used,
            "method": "exhaustive" if exhaustive else "monte_carlo"}


def task_bootstrap_rho(x: Sequence[float], y: Sequence[float], *, n_boot: int = N_BOOT,
                       seed: int = BOOT_SEED) -> dict[str, float]:
    """按**任务**重采样（单位是任务，不是行）。Resample tasks, not rows."""
    rng = np.random.default_rng(seed)
    n = len(x)
    draws: list[float] = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        xs = [x[i] for i in idx]
        ys = [y[i] for i in idx]
        if len(set(xs)) < 2 or len(set(ys)) < 2:  # 全部抽到同一个任务 → ρ 无定义
            continue
        draws.append(spearman_rho(xs, ys))
    draws.sort()
    if not draws:
        return {"lo": float("nan"), "hi": float("nan"), "n_valid": 0}
    lo = draws[int(0.025 * (len(draws) - 1))]
    hi = draws[int(0.975 * (len(draws) - 1))]
    return {"lo": lo, "hi": hi, "n_valid": len(draws)}


# --------------------------------------------------------------------------- 读表
def _float(value: Any) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return None if math.isnan(out) else out


@dataclass(frozen=True)
class Cell:
    """一个 (task, arm, fold, seed) 的 Λ 读数。One paired reading."""

    task: str
    arm: str
    fold: str
    seed: str
    r2_unit: float
    lambda_: float | None
    delta_r2: float | None
    status: str


def read_lambda(path: Path = LAMBDA_PATH) -> list[Cell]:
    out: list[Cell] = []
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            lam = _float(row.get("lambda"))
            unit = _float(row.get("r2_unit"))
            if unit is None:
                continue
            out.append(Cell(row["task"], row["arm"], row.get("fold", ""),
                            row.get("seed", ""), unit,
                            lam if row.get("status") == "ok" else None,
                            _float(row.get("delta_r2")),
                            row.get("status", "")))
    return out


def cell_summary(cells: Sequence[Cell]) -> list[dict[str, Any]]:
    """19 个 (task, arm) 格的均值；同时给出该格分母最小值与可报行数。

    Per-cell means over reportable rows, with the smallest denominator in the cell.
    """
    buckets: dict[tuple[str, str], list[Cell]] = {}
    for cell in cells:
        buckets.setdefault((cell.task, cell.arm), []).append(cell)
    rows: list[dict[str, Any]] = []
    for (task, arm), group in sorted(buckets.items()):
        reportable = [c for c in group
                      if c.lambda_ is not None and not math.isnan(c.lambda_)]
        stable = [c for c in reportable if 1.0 - c.r2_unit >= DENOMINATOR_FLOOR]
        lam_all = [c.lambda_ for c in reportable]
        lam_stable = [c.lambda_ for c in stable]
        rows.append({
            "task": task,
            "arm": arm,
            "n_rows": len(group),
            "n_reportable": len(reportable),
            "n_stable": len(stable),
            "min_denominator": min(1.0 - c.r2_unit for c in group),
            "lambda_mean": float(np.mean(lam_all)) if lam_all else None,
            "lambda_mean_stable": float(np.mean(lam_stable)) if lam_stable else None,
        })
    return rows


def sign_summary(rows: Sequence[dict[str, Any]], key: str = "lambda_mean") -> dict[str, Any]:
    values = [r[key] for r in rows if r[key] is not None]
    return {
        "n_cells": len(values),
        "n_negative": sum(1 for v in values if v < 0),
        "median": float(np.median(values)) if values else None,
        "min": min(values) if values else None,
        "max": max(values) if values else None,
    }


def arm_agreement(rows: Sequence[dict[str, Any]], near_zero: float = 0.05) -> dict[str, Any]:
    """两条链臂共有的任务上，Λ 的符号是否一致；翻号是否发生在"两边都≈0"的格子里。

    On tasks both chain arms cover: do the signs agree, and do flips sit on near-zero cells?
    """
    by_task: dict[str, dict[str, float]] = {}
    for row in rows:
        if row["lambda_mean"] is None:
            continue
        by_task.setdefault(row["task"], {})[row["arm"]] = row["lambda_mean"]
    shared = {t: v for t, v in by_task.items() if {"a3_graph", "a4_polybert"} <= set(v)}
    flips = []
    for task, arms in sorted(shared.items()):
        a3, a4 = arms["a3_graph"], arms["a4_polybert"]
        if a3 * a4 < 0:
            flips.append({"task": task, "a3": a3, "a4": a4, "max_abs": max(abs(a3), abs(a4))})
    return {
        "n_shared_tasks": len(shared),
        "n_sign_flips": len(flips),
        "flips": flips,
        "n_flips_near_zero": sum(1 for f in flips if f["max_abs"] < near_zero),
        "near_zero": near_zero,
    }


# --------------------------------------------------------------------------- 规模曲线
def read_kappa_cells(path: Path = KAPPA_CELLS_PATH) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            rows.append({
                "dataset": row["dataset"],
                "fold": row["fold"],
                "n_train": int(float(row["n_train"])),
                "lambda": float(row["lambda"]),
                "delta_r2": float(row["delta_r2"]),
                "denominator": float(row["denominator"]),
            })
    return rows


def level_curve(rows: Sequence[dict[str, Any]], dataset: str) -> list[dict[str, float]]:
    """逐档位取折均值。Per-level means over folds."""
    buckets: dict[int, list[dict[str, Any]]] = {}
    for row in rows:
        if row["dataset"] != dataset:
            continue
        buckets.setdefault(row["n_train"], []).append(row)
    out = []
    for n in sorted(buckets):
        group = buckets[n]
        out.append({
            "n_train": float(n),
            "n_folds": len(group),
            "lambda": float(np.mean([g["lambda"] for g in group])),
            "delta_r2": float(np.mean([g["delta_r2"] for g in group])),
            "denominator": float(np.mean([g["denominator"] for g in group])),
        })
    return out


def segment_slope(curve: Sequence[dict[str, float]], n_from: float, n_to: float,
                  metric: str = "lambda") -> float | None:
    """段内斜率 = Δvalue / Δlog10(n)，单位是"每十倍数据"。Slope per decade."""
    by_n = {round(float(p["n_train"]), 6): p[metric] for p in curve}
    key_from, key_to = round(n_from, 6), round(n_to, 6)
    if key_from not in by_n or key_to not in by_n:
        return None
    dx = math.log10(n_to / n_from)
    if dx == 0:
        return None
    return (by_n[key_to] - by_n[key_from]) / dx


def log_crossing(n1: float, v1: float, n2: float, v2: float) -> dict[str, Any]:
    """在 log10(n) 上线性插值求过零点；两端同号则返回 None（不外推，D-5）。

    Zero crossing by linear interpolation in log10(n); None when the bracket does not change sign.
    """
    if v1 == v2 or v1 * v2 > 0:
        return {"n_star": None, "method": "no_crossing_in_window",
                "bracket": [n1, n2]}
    frac = (0.0 - v1) / (v2 - v1)
    n_star = 10 ** (math.log10(n1) + frac * (math.log10(n2) - math.log10(n1)))
    return {"n_star": n_star, "method": "log_linear_interpolation", "bracket": [n1, n2]}


# --------------------------------------------------------------------------- 地板
def floor_table(summary: dict[str, Any], *, rule: str = "max_abs") -> dict[str, Any]:
    """零信息地板：两种条件（3 列常量 / 3 列置换）的 |ΔR²| 取较大者。

    `rule="max_abs"` 是本轮采用的口径：地板问的是"不带信息的列能让 R² 移动多少"，
    所以取幅度、不分正负。此前报告在 `pb_t2_modulus` 上取了较小者（0.0015 而非 0.0195），
    与其余七个任务的口径不一致（D-81）。
    """
    rows = []
    for task, entry in sorted(summary["tasks"].items()):
        deltas = [abs(c["delta_r2_vs_baseline_mean"]) for c in entry["conditions"].values()
                  if "delta_r2_vs_baseline_mean" in c]
        rows.append({"task": task, "floor": max(deltas) if deltas else None,
                     "n_pairs": entry["n_folds"]})
    values = [r["floor"] for r in rows if r["floor"] is not None]
    return {"rule": rule, "rows": rows,
            "median": float(np.median(values)) if values else None}


def chain_gain_by_task(cells: Sequence[Cell], arm: str = "a3_graph") -> dict[str, float]:
    """逐任务的 A3 增益 |ΔR²|（可报行均值）。Per-task absolute chain gain."""
    buckets: dict[str, list[float]] = {}
    for cell in cells:
        if cell.arm != arm or cell.delta_r2 is None:
            continue
        buckets.setdefault(cell.task, []).append(cell.delta_r2)
    return {task: abs(float(np.mean(values))) for task, values in buckets.items()}


def floor_clearance(floors: dict[str, Any], chain_gain: dict[str, float]) -> dict[str, Any]:
    """逐任务比：A3 的 |ΔR²| 是否大于该任务自己的地板。Per-task clearance."""
    out = []
    for row in floors["rows"]:
        chain = chain_gain.get(row["task"])
        if row["floor"] is None or chain is None:
            continue
        out.append({"task": row["task"], "floor": row["floor"], "chain": chain,
                    "ratio": (chain / row["floor"]) if row["floor"] > 0 else None,
                    "clears": chain > row["floor"]})
    return {"rows": out, "n_clears": sum(1 for r in out if r["clears"]), "n_tasks": len(out)}


# --------------------------------------------------------------------------- 主入口
def build() -> dict[str, Any]:
    cells = read_lambda()
    per_cell = cell_summary(cells)

    ordering = list(csv.DictReader(ORDERING_PATH.open(newline="", encoding="utf-8")))
    points = [r for r in ordering if r["scope"] == "point" and r["arm"] == "a3_graph"]
    tiers = [int(r["tier"]) for r in points]
    lam_a3 = [float(r["lambda"]) for r in points]
    rho_a3 = spearman_rho(tiers, lam_a3)
    rho_a4 = spearman_rho(tiers, [float(r["lambda"]) for r in
                                  [p for p in ordering if p["scope"] == "point"
                                   and p["arm"] == "a4_polybert"]])

    curve = level_curve(read_kappa_cells(), "md5000_alternating_cp")
    ten = {"n_from": 300, "n_to": 3000,
           "lambda_slope": segment_slope(curve, 300, 3000, "lambda"),
           "delta_r2_slope": segment_slope(curve, 300, 3000, "delta_r2"),
           "lambda_ratio": next(p["lambda"] for p in curve if p["n_train"] == 3000)
           / next(p["lambda"] for p in curve if p["n_train"] == 300) if curve else None}
    tail = {"n_from": 3000, "n_to": 22242,
            "lambda_slope": segment_slope(curve, 3000, 22242, "lambda"),
            "delta_r2_slope": segment_slope(curve, 3000, 22242, "delta_r2")}
    full = {"n_from": 300, "n_to": 22242,
            "lambda_slope": segment_slope(curve, 300, 22242, "lambda"),
            "delta_r2_slope": segment_slope(curve, 300, 22242, "delta_r2")}

    crossing = {}
    for prop, v30, v60 in (("tg", 0.344, -0.037), ("dielectric", 0.435, -0.007)):
        crossing[prop] = log_crossing(30, v30, 60, v60)

    floors = floor_table(json.loads(FLOOR_PATH.read_text(encoding="utf-8")))
    clearance = floor_clearance(floors, chain_gain_by_task(cells))

    summary = {
        "generated_by": "scripts/06k_defense_checks.py",
        "claims": {
            "lambda_cells_all": sign_summary(per_cell),
            "lambda_cells_stable": sign_summary(per_cell, key="lambda_mean_stable"),
            "n_cells_without_stable_rows": sum(1 for r in per_cell
                                               if r["lambda_mean_stable"] is None),
        },
        "h1": {
            "rho_a3": rho_a3, "rho_a4": rho_a4,
            "permutation_a3": permutation_p(rho_a3, tiers),
            "permutation_a4": permutation_p(rho_a4, tiers),
            "bootstrap_a3": task_bootstrap_rho(tiers, lam_a3),
            "bootstrap_a4": task_bootstrap_rho(
                tiers, [float(p["lambda"]) for p in ordering
                        if p["scope"] == "point" and p["arm"] == "a4_polybert"]),
        },
        "scale": {"curve": curve, "tenfold": ten, "tail": tail, "full_range": full},
        "arm_agreement": arm_agreement(per_cell),
        "crossing": crossing,
        "floor": {"table": floors, "clearance": clearance,
                  "ratio_at_last_level": 0.006607 / floors["median"] if floors["median"] else None},
        "sources": {
            "lambda": str(LAMBDA_PATH.relative_to(REPO_ROOT)).replace("\\", "/"),
            "ordering": str(ORDERING_PATH.relative_to(REPO_ROOT)).replace("\\", "/"),
            "kappa_cells": str(KAPPA_CELLS_PATH.relative_to(REPO_ROOT)).replace("\\", "/"),
            "floor": str(FLOOR_PATH.relative_to(REPO_ROOT)).replace("\\", "/"),
        },
    }
    return {"summary": summary, "cells": per_cell}


def write_all(payload: dict[str, Any] | None = None) -> tuple[Path, Path]:
    payload = payload or build()
    cells = payload["cells"]
    fields = ["task", "arm", "n_rows", "n_reportable", "n_stable", "min_denominator",
              "lambda_mean", "lambda_mean_stable"]
    with CELLS_OUT.open("w", newline="\n", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in cells:
            writer.writerow({k: ("" if row[k] is None else row[k]) for k in fields})
    SUMMARY_OUT.write_text(json.dumps(payload["summary"], indent=2, ensure_ascii=False) + "\n",
                           encoding="utf-8", newline="\n")
    return CELLS_OUT, SUMMARY_OUT


def report(payload: dict[str, Any] | None = None) -> str:
    payload = payload or build()
    s = payload["summary"]
    lines = []
    allc, sta = s["claims"]["lambda_cells_all"], s["claims"]["lambda_cells_stable"]
    lines.append(f"λ 格: {allc['n_negative']}/{allc['n_cells']} 为负，中位 {allc['median']:.4f}")
    lines.append(f"稳定格: {sta['n_negative']}/{sta['n_cells']} 为负，中位 {sta['median']:.4f}"
                 f"（无稳定行的格 {s['claims']['n_cells_without_stable_rows']}）")
    h = s["h1"]
    lines.append(f"H1 ρ: A3 {h['rho_a3']:+.4f} p={h['permutation_a3']['p_two_sided']:.4f} "
                 f"boot[{h['bootstrap_a3']['lo']:+.3f},{h['bootstrap_a3']['hi']:+.3f}]")
    lines.append(f"       A4 {h['rho_a4']:+.4f} p={h['permutation_a4']['p_two_sided']:.4f} "
                 f"boot[{h['bootstrap_a4']['lo']:+.3f},{h['bootstrap_a4']['hi']:+.3f}]")
    t = s["scale"]["tenfold"]
    lines.append(f"十倍段 300→3000: κ段={t['lambda_slope']:+.4f}/dex, "
                 f"ΔR²段={t['delta_r2_slope']:+.4f}/dex, Λ比={t['lambda_ratio']:.3f}")
    tl = s["scale"]["tail"]
    lines.append(f"尾段 3000→22242: κ段={tl['lambda_slope']:+.4f}/dex, ΔR²段={tl['delta_r2_slope']:+.4f}/dex")
    a = s["arm_agreement"]
    lines.append(f"两臂共有 {a['n_shared_tasks']} 任务，翻号 {a['n_sign_flips']} 处，"
                 f"其中 {a['n_flips_near_zero']} 处两侧 |Λ|<{a['near_zero']}")
    for prop, c in s["crossing"].items():
        lines.append(f"n* {prop}: {c['n_star']:.1f}（{c['method']}，区间 {c['bracket']}）"
                     if c["n_star"] else f"n* {prop}: 不过零")
    f = s["floor"]
    lines.append(f"地板: 中位 {f['table']['median']:.4f}，{f['clearance']['n_clears']}"
                 f"/{f['clearance']['n_tasks']} 过；末档 ΔR²/中位地板 = {f['ratio_at_last_level']:.1f}×")
    return "\n".join(lines)
