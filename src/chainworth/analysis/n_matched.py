"""同 n 下的 tier1 vs tier2：H1 的混淆到底遮住了什么。
tier-1 versus tier-2 at the same n: what the tier/x confound was hiding.

中文：
    用途：分工 D-4 的 H1 敏感性。PolyBench 的层号与样本量完全反序（t3 228/279 < t2 711/1034 <
          t1 3796/8055，ρ(tier, log n) = −0.956），所以"ρ ≈ 0"有两个互斥的解释：
          甲、层号与 Λ 真的无关；乙、层号效应与样本量效应互相抵消。
          把 t1 的训练集砍到 t2 的量级、在**同一批测试集**上重训，就能分开它们——
          若同一 n 下 t1 与 t2 的 Λ 差不多，甲成立；若 t1 明显更高，则是乙。
    三条口径：
      ① **只配对 (task, level, seed)，不跨层借数。** 每个 (任务, 档位) 的 Λ 只用它自己的
         A2/A3 两条读数，两条必须来自同一档位——这正是 `n_matched.csv` 的 `subset_sha256`
         存在的理由（同一批训练行，才能相减）。
      ② **tier 由任务名推，不读记录里的 `tier` 字段。** run 记录里那列是 `None`（实测 963 条
         全是 `None`），照它分组会把 6 个任务全塞进一个层。
      ③ **A4 按设计不可评。** `polybert.train_one()` 明确拒绝 n 档（`ArmError`：n 档只属于
         B-5 的 OpenPoly 扫描），所以匹配口径只有 A3 一条链臂；不绕过那个闸门，如实写
         `not_evaluable_by_design`。
    数据来源：`results/aggregate/n_matched.csv`（06i 产出）+ `results/aggregate/lambda_ci.csv`（全量对照）。
    副作用：写两张表（见 `write_all`）。异常：`NMatchedError`。
"""

from __future__ import annotations

import csv
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
MATCHED_PATH = REPO_ROOT / "results" / "aggregate" / "n_matched.csv"
LAMBDA_PATH = REPO_ROOT / "results" / "aggregate" / "lambda_ci.csv"
CELLS_PATH = REPO_ROOT / "results" / "aggregate" / "n_matched_lambda.csv"
SUMMARY_PATH = REPO_ROOT / "results" / "aggregate" / "n_matched_summary.csv"

N_BOOT = 2000
BOOT_SEED = 20261118
# 层号从任务名取：pb_t1_* / pb_t2_* / pb_t3_*（记录里的 tier 字段全是 None）
TIER_OF = {"pb_t1_tg": 1, "pb_t1_tm": 1, "pb_t2_modulus": 2, "pb_t2_permeability": 2}

CELL_COLUMNS = [
    "task",
    "tier",
    "n_level",
    "n_unit",
    "n_chain",
    "n_seeds",
    "r2_unit",
    "r2_chain",
    "lambda",
    "lambda_min",
    "lambda_max",
    "subset_sha256",
]
SUMMARY_COLUMNS = [
    "n_level",
    "n_tier1",
    "n_tier2",
    "lambda_tier1",
    "lambda_tier2",
    "delta",
    "delta_lo",
    "delta_hi",
    "n_points_tier1",
    "n_points_tier2",
    "verdict",
    "note",
]


class NMatchedError(RuntimeError):
    """匹配表缺件、或某一档上没有可配对的臂。 / Missing table, or a level has no paired arms."""


@dataclass(frozen=True)
class Reading:
    """一条 (任务, 臂, 档位, 种子) 的 R²。 / One (task, arm, level, seed) R^2."""

    task: str
    arm: str
    level: str
    seed: int
    r2: float
    n_train: int
    subset_sha256: str


def load_readings(path: Path = MATCHED_PATH) -> list[Reading]:
    """读 `n_matched.csv`，只留 r2 有限的行。 / Read `n_matched.csv`, keep finite R^2."""
    if not path.exists():
        raise NMatchedError(f"缺 {path}：先跑 scripts/06i_n_matched.py")
    out: list[Reading] = []
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        missing = {"task", "arm", "n_level", "seed", "r2", "n_train"} - set(reader.fieldnames or [])
        if missing:
            raise NMatchedError(f"{path} 缺列 {sorted(missing)}")
        for row in reader:
            try:
                r2 = float(row["r2"])
            except (TypeError, ValueError):
                continue
            if not np.isfinite(r2):
                continue
            out.append(
                Reading(
                    task=row["task"],
                    arm=row["arm"],
                    level=str(row["n_level"]),
                    seed=int(float(row["seed"])),
                    r2=r2,
                    n_train=int(float(row["n_train"])),
                    subset_sha256=row.get("subset_sha256", ""),
                )
            )
    if not out:
        raise NMatchedError(f"{path} 里没有有限 R² 的行")
    return out


def _lambda(r2_unit: float, r2_chain: float) -> float:
    """Λ = (R²_chain − R²_unit) / (1 − R²_unit)；分母 ≤ 0 时返回 nan（不裁剪）。
    Lambda; returns nan when the denominator is not positive (never clipped).
    """
    denominator = 1.0 - r2_unit
    if denominator <= 0 or not np.isfinite(denominator):
        return float("nan")
    return (r2_chain - r2_unit) / denominator


def cell_rows(readings: Sequence[Reading]) -> list[dict[str, Any]]:
    """按 (任务, 档位) 汇总：同一 (折, 种子) 上两条臂相减，再对种子取均值。
    Aggregate per (task, level): subtract the two arms on the same (fold, seed), then average seeds.
    """
    paired: dict[tuple[str, str, int], dict[str, float]] = defaultdict(dict)
    meta: dict[tuple[str, str], dict[str, Any]] = defaultdict(dict)
    for reading in readings:
        paired[(reading.task, reading.level, reading.seed)][reading.arm] = reading.r2
        meta[(reading.task, reading.level)][f"n_{reading.arm}"] = reading.n_train
        meta[(reading.task, reading.level)]["subset_sha256"] = reading.subset_sha256

    per_cell: dict[tuple[str, str], list[float]] = defaultdict(list)
    units: dict[tuple[str, str], list[float]] = defaultdict(list)
    chains: dict[tuple[str, str], list[float]] = defaultdict(list)
    for (task, level, _seed), arms in paired.items():
        if "a2_conventional" not in arms or "a3_graph" not in arms:
            continue
        value = _lambda(arms["a2_conventional"], arms["a3_graph"])
        if not np.isfinite(value):
            continue
        per_cell[(task, level)].append(value)
        units[(task, level)].append(arms["a2_conventional"])
        chains[(task, level)].append(arms["a3_graph"])

    rows: list[dict[str, Any]] = []
    for (task, level), values in sorted(per_cell.items(), key=lambda kv: (kv[0][1], kv[0][0])):
        info = meta.get((task, level), {})
        rows.append(
            {
                "task": task,
                "tier": TIER_OF.get(task, ""),
                "n_level": level,
                # 两臂的训练行数**分开记**：A3 建图会丢行（modulus 723→651、permeability 497→447），
                # 只写一个 n 会把"同一批训练行"这句话说过头。
                "n_unit": info.get("n_a2_conventional", ""),
                "n_chain": info.get("n_a3_graph", ""),
                "n_seeds": len(values),
                "r2_unit": round(float(np.mean(units[(task, level)])), 6),
                "r2_chain": round(float(np.mean(chains[(task, level)])), 6),
                "lambda": round(float(np.mean(values)), 6),
                "lambda_min": round(float(np.min(values)), 6),
                "lambda_max": round(float(np.max(values)), 6),
                "subset_sha256": info.get("subset_sha256", ""),
            }
        )
    if not rows:
        raise NMatchedError("没有可配对的 (任务, 档位)：A2 与 A3 必须落在同一档位上")
    return rows


def _resample(values: Sequence[float], rng: np.random.Generator, n_boot: int) -> list[float]:
    """对一组 Λ 做有放回重采样，返回均值序列。 / Bootstrap the mean of a set of Lambdas."""
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        return []
    draws = rng.integers(0, arr.size, size=(n_boot, arr.size))
    return [float(np.mean(arr[draw])) for draw in draws]


def summary_rows(
    cells: Sequence[dict[str, Any]],
    *,
    n_boot: int = N_BOOT,
    seed: int = BOOT_SEED,
) -> list[dict[str, Any]]:
    """逐档位算 Δ = mean Λ(tier1) − mean Λ(tier2)，带种子级区间。
    Per level: delta = mean Lambda(tier 1) - mean Lambda(tier 2), with a seed-level interval.

    中文：tier2 在每个档位上**只有一个任务**（711 对 permeability、1034 对 modulus），
          所以 Δ 的不确定度主要来自 tier1 的两个任务 × 3 种子与 tier2 的 3 种子。
          区间是对**两组的全部 (任务, 种子) 读数**分别重采样得到的，不做配对（配对只在同任务内成立）。
    """
    rng = np.random.default_rng(seed)
    by_level: dict[str, dict[int, list[float]]] = defaultdict(lambda: defaultdict(list))
    for cell in cells:
        tier = cell.get("tier")
        if tier == "":
            continue
        by_level[str(cell["n_level"])][int(tier)].append(float(cell["lambda"]))

    rows: list[dict[str, Any]] = []
    for level in sorted(by_level, key=lambda lv: int(lv) if str(lv).isdigit() else 10**9):
        groups = by_level[level]
        if 1 not in groups or 2 not in groups:
            rows.append(
                {
                    "n_level": level,
                    "n_tier1": len(groups.get(1, [])) * 3,
                    "n_tier2": "",
                    "lambda_tier1": round(float(np.mean(groups[1])), 6) if 1 in groups else "",
                    "lambda_tier2": "",
                    "delta": "",
                    "delta_lo": "",
                    "delta_hi": "",
                    "n_points_tier1": len(groups.get(1, [])),
                    "n_points_tier2": 0,
                    "verdict": "no_tier2_at_this_level",
                    "note": "这一档上没有 tier2 的任务，Δ 无定义",
                }
            )
            continue
        t1, t2 = groups[1], groups[2]
        # 每格一个读数（已对种子取均值），重采样单位是"格"，所以要按任务数还原
        t1_cells = [float(c["lambda"]) for c in cells if c.get("tier") == 1 and str(c["n_level"]) == level]
        t2_cells = [float(c["lambda"]) for c in cells if c.get("tier") == 2 and str(c["n_level"]) == level]
        draw1 = _resample(t1_cells, rng, n_boot)
        draw2 = _resample(t2_cells, rng, n_boot)
        deltas = [a - b for a, b in zip(draw1, draw2, strict=False)]
        delta = float(np.mean(t1_cells) - np.mean(t2_cells))
        rows.append(
            {
                "n_level": level,
                "n_tier1": len(t1_cells),
                "n_tier2": len(t2_cells),
                "lambda_tier1": round(float(np.mean(t1_cells)), 6),
                "lambda_tier2": round(float(np.mean(t2_cells)), 6),
                "delta": round(delta, 6),
                "delta_lo": round(float(np.percentile(deltas, 2.5)), 6) if deltas else "",
                "delta_hi": round(float(np.percentile(deltas, 97.5)), 6) if deltas else "",
                "n_points_tier1": len(t1_cells),
                "n_points_tier2": len(t2_cells),
                "verdict": "tier1_lower" if delta < 0 else "tier1_higher_or_equal",
                "note": "",
            }
        )
    return rows


def full_data_lookup(path: Path = LAMBDA_PATH) -> dict[str, float]:
    """全量口径的 Λ（按任务），供对照。 / Full-data Lambda per task, as a contrast."""
    if not path.exists():
        return {}
    out: dict[str, float] = {}
    with path.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            if row.get("arm") != "a3_graph" or row.get("task") not in TIER_OF:
                continue
            try:
                value = float(row["lambda"])
            except (TypeError, ValueError):
                continue
            if np.isfinite(value):
                out.setdefault(row["task"], value)
    return out


def write_all(
    *,
    matched_path: Path = MATCHED_PATH,
    cells_path: Path = CELLS_PATH,
    summary_path: Path = SUMMARY_PATH,
    n_boot: int = N_BOOT,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """算完写两张表，返回 (逐格行, 逐档位行)。 / Compute and write both tables."""
    readings = load_readings(matched_path)
    cells = cell_rows(readings)
    summary = summary_rows(cells, n_boot=n_boot)
    for path, columns, rows in (
        (cells_path, CELL_COLUMNS, cells),
        (summary_path, SUMMARY_COLUMNS, summary),
    ):
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=columns)
            writer.writeheader()
            for row in rows:
                writer.writerow({key: row.get(key, "") for key in columns})
    return cells, summary


def explain(cells: Sequence[dict[str, Any]], summary: Sequence[dict[str, Any]]) -> str:
    """压成一段人读文字。 / Render as readable text."""
    lines = ["同 n 下的 tier1 vs tier2（A3 链臂，测试集逐行不变）", ""]
    lines.append(f"  {'任务':<24}{'层':>3}{'档位':>7}{'A2 n':>7}{'A3 n':>7}{'Λ':>9}{'种子范围':>20}")
    for cell in cells:
        lines.append(
            f"  {cell['task']:<24}{cell['tier']:>3}{cell['n_level']:>7}"
            f"{cell['n_unit']:>7}{cell['n_chain']:>7}"
            f"{cell['lambda']:>+9.4f}   [{cell['lambda_min']:+.3f}, {cell['lambda_max']:+.3f}]"
        )
    lines.append("")
    for row in summary:
        if row["delta"] == "":
            lines.append(f"  n={row['n_level']}: {row['verdict']}（{row['note']}）")
            continue
        lines.append(
            f"  n={row['n_level']}: tier1 {row['lambda_tier1']:+.4f} vs tier2 {row['lambda_tier2']:+.4f}"
            f"  Δ={row['delta']:+.4f} [{row['delta_lo']}, {row['delta_hi']}]  {row['verdict']}"
        )
    return "\n".join(lines)


__all__ = [
    "BOOT_SEED",
    "CELLS_PATH",
    "CELL_COLUMNS",
    "MATCHED_PATH",
    "N_BOOT",
    "NMatchedError",
    "Reading",
    "SUMMARY_COLUMNS",
    "SUMMARY_PATH",
    "TIER_OF",
    "cell_rows",
    "explain",
    "full_data_lookup",
    "load_readings",
    "summary_rows",
    "write_all",
]
