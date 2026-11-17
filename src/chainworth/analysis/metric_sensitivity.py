"""Λ 与 ΔR² 两把尺子：同一个 scaling 曲线，比值口径与绝对口径各说什么（H2′）。
Two rulers on one curve: what the ratio metric (Lambda) and the absolute metric (dR^2) each say.

中文：
    用途：分工 D-6 的 H2′。H2 用 Λ = ΔR²/(1−R²_unit) 判"数据能不能买到链信息"，结论是
          "κ 区间整段 < 0 但 Λ 没减半"。但 Λ 的分母随 n 单调缩小（1−R²_unit 从 0.135 掉到
          0.034），于是**同一个 ΔR² 会被分母放大**：交替集 300 → 1,000 时分子只降 5%，
          分母降 45%，Λ 反而从 0.295 升到 0.503。驼峰是比值造的，不是数据造的。
          本模块把两把尺子并排放在同一张表上，让 H2′ 可以被检验，而不是被论证。
    三条口径（每一条都能悄悄改结论）：
      ① **两把尺子用同一批折、同一批档位。** 行级取 `kappa_cells.csv` 的同一份数据，
         不重新算 Λ、不重新配对——否则两个比值之差里会混进"算了两次"的差别。
      ② **比值只在首档为正时才有意义。** 首档 ≤ 0 时写 `undefined`，不写 1.0、不写 0
         （Λ 会为负，负值的"减半"没有定义）。这是 D-73 同型纪律：宁可留空。
      ③ **区间按折重采样**（B = 2000，单位是折不是行）。同一折的所有档位在同一次抽样里
         一起被选中，因为同一折的测试集在各档位上逐行相同——这也是 `kappa.py` 用配对
         重采样的同一条理由。5 折的重采样只有 5 个不同的多重集，区间会偏窄，报告的
         `n_folds` 列就是要让人看见这一点。
    数据来源：`results/aggregate/kappa_cells.csv`（40 行 = 2 数据集 × 档位 × 5 折）。
    配置来源：无。副作用：写两张表（见 `write_all`）。异常：`MetricSensitivityError`。

English:
    Purpose: division D, H2'. H2 tested "can money buy chain information" with
    Lambda = dR^2 / (1 - R^2_unit). Its denominator shrinks monotonically with n (0.135 -> 0.034),
    so the **same** dR^2 gets inflated as n grows: from 300 to 1,000 rows the numerator falls 5%
    while the denominator falls 45%, pushing Lambda from 0.295 up to 0.503. The hump is an
    artifact of the ratio, not a property of the data. This module puts both rulers in one table
    so H2' can be tested rather than argued. Conventions: (1) both rulers reuse the exact rows of
    `kappa_cells.csv` - nothing is recomputed or re-paired; (2) a ratio is reported only when the
    first level is positive, otherwise `undefined` (Lambda can be negative and "halved" is then
    undefined); (3) intervals come from a **fold-level** bootstrap (B = 2000) that draws all tiers
    of a fold together, for the same reason `kappa.py` uses paired resampling.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
CELLS_PATH = REPO_ROOT / "results" / "aggregate" / "kappa_cells.csv"
LEVELS_PATH = REPO_ROOT / "results" / "aggregate" / "metric_sensitivity_levels.csv"
SUMMARY_PATH = REPO_ROOT / "results" / "aggregate" / "metric_sensitivity.csv"

N_BOOT = 2000
BOOT_SEED = 20261118
HALVED = 0.5

METRICS: tuple[str, ...] = ("lambda", "delta_r2")
METRIC_LABEL = {
    "lambda": "Λ = ΔR²/(1−R²_unit)",
    "delta_r2": "ΔR² = R²_chain − R²_unit",
}

LEVEL_COLUMNS = [
    "dataset",
    "metric",
    "level",
    "n_train_median",
    "n_folds",
    "value_mean",
    "value_lo",
    "value_hi",
]
SUMMARY_COLUMNS = [
    "dataset",
    "metric",
    "level_first",
    "level_last",
    "n_first",
    "n_last",
    "value_first",
    "value_last",
    "ratio",
    "ratio_lo",
    "ratio_hi",
    "decay_per_dex",
    "halved",
    "verdict",
    "note",
]


class MetricSensitivityError(RuntimeError):
    """两把尺子的输入不齐、或某个档位没有可用折。 / Inputs missing or a level has no usable fold."""


@dataclass(frozen=True)
class Cell:
    """一个 (数据集, 折, 档位) 的读数。 / One (dataset, fold, level) reading."""

    dataset: str
    fold: str
    level: str
    n_train: float
    lambda_: float
    delta_r2: float

    def value(self, metric: str) -> float:
        """按名字取值。 / Value by metric name."""
        if metric == "lambda":
            return self.lambda_
        if metric == "delta_r2":
            return self.delta_r2
        raise MetricSensitivityError(f"未知指标 {metric!r}；只认 {METRICS}")


def load_cells(path: Path = CELLS_PATH) -> list[Cell]:
    """读 `kappa_cells.csv`，只留两列都非 nan 的行。
    Read `kappa_cells.csv`, keeping rows where both rulers are defined.

    中文：输入：path。输出：Cell 列表。异常：文件缺、必需列缺。
          `unstable` 的行**保留**（分母小是事实，不是脏数据），但 nan 行必须丢——
          否则 `np.mean` 会让一个 nan 污染整格（图 1 踩过）。
    """
    if not path.exists():
        raise MetricSensitivityError(f"缺 {path}")
    cells: list[Cell] = []
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        missing = {"dataset", "fold", "tier", "n_train", "lambda", "delta_r2"} - set(
            reader.fieldnames or []
        )
        if missing:
            raise MetricSensitivityError(f"{path} 缺列 {sorted(missing)}")
        for row in reader:
            try:
                lam = float(row["lambda"])
                dr = float(row["delta_r2"])
            except (TypeError, ValueError):
                continue
            if not (np.isfinite(lam) and np.isfinite(dr)):
                continue
            cells.append(
                Cell(
                    dataset=row["dataset"],
                    fold=row["fold"],
                    level=str(row["tier"]),
                    n_train=float(row["n_train"]),
                    lambda_=lam,
                    delta_r2=dr,
                )
            )
    if not cells:
        raise MetricSensitivityError(f"{path} 里没有可用的行")
    return cells


def _level_order(cells: Iterable[Cell]) -> list[str]:
    """按档位的中位训练行数升序排，不按字符串排（"10000" < "300"）。
    Order levels by median n_train, not by string ("10000" < "300" lexicographically).
    """
    seen: dict[str, list[float]] = {}
    for cell in cells:
        seen.setdefault(cell.level, []).append(cell.n_train)
    return sorted(seen, key=lambda lv: float(np.median(seen[lv])))


def _draw(cells: Sequence[Cell], indices: Sequence[int]) -> dict[tuple[str, str], float]:
    """按给出的折下标重采样，返回 (档位, 指标) → 均值。 / Resample folds, return mean per (level, metric)."""
    picked = [cells[i] for i in indices]
    out: dict[tuple[str, str], float] = {}
    levels = {cell.level for cell in picked}
    for level in levels:
        for metric in METRICS:
            values = [cell.value(metric) for cell in picked if cell.level == level]
            if values:
                out[(level, metric)] = float(np.mean(values))
    return out


def _ratio(first: float, last: float) -> float | None:
    """首末比；首档 ≤ 0 时为 None（负值的"减半"没有定义）。 / Ratio, None when the first level is <= 0."""
    if not np.isfinite(first) or not np.isfinite(last) or first <= 0:
        return None
    return last / first


def level_and_ratio_rows(
    cells: Sequence[Cell],
    *,
    n_boot: int = N_BOOT,
    seed: int = BOOT_SEED,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """出逐档位行与首末比行。 / Build the per-level rows and the first-to-last ratio rows."""
    rng = np.random.default_rng(seed)
    level_rows: list[dict[str, Any]] = []
    ratio_rows: list[dict[str, Any]] = []
    for dataset in sorted({cell.dataset for cell in cells}):
        subset = [cell for cell in cells if cell.dataset == dataset]
        order = _level_order(subset)
        if len(order) < 2:
            raise MetricSensitivityError(f"{dataset} 只有 {len(order)} 个档位，算不出首末比")
        folds = sorted({cell.fold for cell in subset})
        # 折 -> 该折各档位读数的全局行下标
        by_fold: dict[str, list[int]] = {fold: [] for fold in folds}
        for index, cell in enumerate(subset):
            by_fold[cell.fold].append(index)
        draws = rng.integers(0, len(folds), size=(n_boot, len(folds)))

        for metric in METRICS:
            boot_values: dict[str, list[float]] = {level: [] for level in order}
            boot_ratio: list[float] = []
            for row in draws:
                indices: list[int] = []
                for slot in row:
                    indices.extend(by_fold[folds[int(slot)]])
                sampled = _draw(subset, indices)
                for level in order:
                    if (level, metric) in sampled:
                        boot_values[level].append(sampled[(level, metric)])
                first = sampled.get((order[0], metric))
                last = sampled.get((order[-1], metric))
                if first is None or last is None:
                    continue
                value = _ratio(first, last)
                if value is not None:
                    boot_ratio.append(value)

            for level in order:
                values = [cell.value(metric) for cell in subset if cell.level == level]
                if not values:
                    continue
                n_train = [cell.n_train for cell in subset if cell.level == level]
                level_rows.append(
                    {
                        "dataset": dataset,
                        "metric": metric,
                        "level": level,
                        "n_train_median": int(round(float(np.median(n_train)))),
                        "n_folds": len(values),
                        "value_mean": round(float(np.mean(values)), 6),
                        "value_lo": round(float(np.percentile(boot_values[level], 2.5)), 6)
                        if boot_values[level]
                        else "",
                        "value_hi": round(float(np.percentile(boot_values[level], 97.5)), 6)
                        if boot_values[level]
                        else "",
                    }
                )

            first_values = [cell.value(metric) for cell in subset if cell.level == order[0]]
            last_values = [cell.value(metric) for cell in subset if cell.level == order[-1]]
            first_mean = float(np.mean(first_values))
            last_mean = float(np.mean(last_values))
            ratio = _ratio(first_mean, last_mean)
            n_first = float(np.median([c.n_train for c in subset if c.level == order[0]]))
            n_last = float(np.median([c.n_train for c in subset if c.level == order[-1]]))
            decay = ""
            if ratio is not None and ratio > 0 and n_last > n_first:
                # 必须是 log10 除 log10：用 ln 除 log10 会把每十倍的变化率放大 2.3026 倍
                decay = round(float(np.log10(ratio) / np.log10(n_last / n_first)), 4)
            ratio_rows.append(
                {
                    "dataset": dataset,
                    "metric": metric,
                    "level_first": order[0],
                    "level_last": order[-1],
                    "n_first": int(round(n_first)),
                    "n_last": int(round(n_last)),
                    "value_first": round(first_mean, 6),
                    "value_last": round(last_mean, 6),
                    "ratio": round(ratio, 4) if ratio is not None else "",
                    "ratio_lo": round(float(np.percentile(boot_ratio, 2.5)), 4) if boot_ratio else "",
                    "ratio_hi": round(float(np.percentile(boot_ratio, 97.5)), 4) if boot_ratio else "",
                    "decay_per_dex": decay,
                    "halved": str(ratio < HALVED).lower() if ratio is not None else "",
                    "verdict": "undefined"
                    if ratio is None
                    else ("h2p_halved" if ratio < HALVED else "h2p_not_halved"),
                    "note": ""
                    if ratio is not None
                    else f"首档 {order[0]} 的均值为 {first_mean:.4f} ≤ 0，比值无定义",
                }
            )
    return level_rows, ratio_rows


def write_all(
    *,
    cells_path: Path = CELLS_PATH,
    levels_path: Path = LEVELS_PATH,
    summary_path: Path = SUMMARY_PATH,
    n_boot: int = N_BOOT,
    seed: int = BOOT_SEED,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """算完写两张表，返回 (逐档位行, 首末比行)。 / Compute and write both tables."""
    cells = load_cells(cells_path)
    level_rows, ratio_rows = level_and_ratio_rows(cells, n_boot=n_boot, seed=seed)
    for path, columns, rows in (
        (levels_path, LEVEL_COLUMNS, level_rows),
        (summary_path, SUMMARY_COLUMNS, ratio_rows),
    ):
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=columns)
            writer.writeheader()
            for row in rows:
                writer.writerow({key: row.get(key, "") for key in columns})
    return level_rows, ratio_rows


def explain(level_rows: Sequence[dict[str, Any]], ratio_rows: Sequence[dict[str, Any]]) -> str:
    """把两张表压成一段人读的文字。 / Render both tables as readable text."""
    lines = ["H2′ 两把尺子（同一批折、同一批档位）"]
    for row in ratio_rows:
        ratio = row["ratio"]
        if ratio == "":
            lines.append(f"  {row['dataset']:26s} {row['metric']:9s} 比值无定义：{row['note']}")
            continue
        interval = (
            f"[{row['ratio_lo']}, {row['ratio_hi']}]"
            if row["ratio_lo"] != "" and row["ratio_hi"] != ""
            else "[未算]"
        )
        lines.append(
            f"  {row['dataset']:26s} {row['metric']:9s} "
            f"{row['value_first']:+.4f} → {row['value_last']:+.4f}  "
            f"比值 {ratio:.3f} {interval}  {row['verdict']}"
        )
    return "\n".join(lines)


__all__ = [
    "BOOT_SEED",
    "CELLS_PATH",
    "Cell",
    "HALVED",
    "LEVEL_COLUMNS",
    "LEVELS_PATH",
    "METRICS",
    "METRIC_LABEL",
    "MetricSensitivityError",
    "N_BOOT",
    "SUMMARY_COLUMNS",
    "SUMMARY_PATH",
    "explain",
    "level_and_ratio_rows",
    "load_cells",
    "write_all",
]
