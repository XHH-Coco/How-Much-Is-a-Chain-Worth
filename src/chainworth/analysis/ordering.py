"""H1 的分层排序检验：Λ 的排序与 PolyBench 三层划分是否一致（分工 D-4）。
Does Lambda order itself the way PolyBench's three-tier division says it should?

中文：
    用途：分工 D-4。提案 §H1 的原句是"Λ 的排序与 PolyBench 自身的三层划分一致"，
          证伪条件是"Spearman ρ < 0.5，或出现跨 ≥2 层的倒序"。
    三条口径，每条都能悄悄把结论推向一边：
      ① **只在 PolyBench 内部排序。** tier 标签只有 PolyBench 发布，跨基准合并排序是被 R1
         禁止的（§2.1）——合并会让 ρ 变成"两个不同协议的乘积"，没有意义。
      ② **6 个点各自是 3 个种子的算术平均，不是挑最好的那个种子。** 挑种子等于把选择噪声
         算成信号；区间也只是三个 λ_lo/λ_hi 的均值，**不是**合并后的 95% 区间——
         合并区间要重做 bootstrap，这里不假装有。
      ③ **`undecidable` 用"任一种子被标就整体标"**（保守）。D-2 验收标准③把区间宽 > 0.3 的点
         标 `undecidable`，这些点上 Λ 的排序没有分辨力；把它们默认算进主判据，同时**另外报一版
         只含可判点的 ρ**——两版一起看才知道结论是不是被几个噪声点撑起来的。
    数据来源：`results/aggregate/lambda_ci.csv`（每 (task, arm, seed) 一行）与
              `configs/tasks/*.yaml`（tier / source_kind / n_labels）。
    配置来源：无。副作用：`write_tables()` 写 `h1_ordering.csv`。
    异常：`OrderingError` —— 任务数不足、臂不在表里、tier 缺失。

English:
    Purpose: division D, task D-4. Checks proposal H1 ("Lambda's ordering agrees with PolyBench's
    own three tiers"; falsified when Spearman rho < 0.5 or a >=2-tier inversion appears).
    Three conventions: (1) rank inside PolyBench only - the tier labels are published by nobody
    else, and R1 forbids cross-benchmark pooling; (2) each task's point is the mean over its three
    seeds, never the best seed, and the interval is the mean of the per-seed bounds rather than a
    re-bootstrapped one; (3) a task is `undecidable` if any seed was flagged, and a second rho over
    only the decidable points is reported alongside so a conclusion cannot hide behind noise.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from chainworth.io.tasks import load_all_task_configs

REPO_ROOT = Path(__file__).resolve().parents[3]
LAMBDA_PATH = REPO_ROOT / "results" / "aggregate" / "lambda_ci.csv"
ORDERING_PATH = REPO_ROOT / "results" / "aggregate" / "h1_ordering.csv"

BENCHMARK = "polybench"
CHAIN_ARMS = ("a3_graph", "a4_polybert")
# 提案 §H1 的证伪门槛。写进常量是因为它会出现在三处（判据、表、图），
# 三处各写一遍就会有一天不一致。
# The proposal's threshold, kept in one place because it appears in the verdict, the table and
# the figure.
RHO_THRESHOLD = 0.5
# 判"跨层倒序"的最小层距。相邻层倒过去只是噪声（每层只有 2 个任务）；
# 跨两层倒过去就与"三层划分一致"正面冲突。
# Minimum tier distance for an inversion to count: adjacent swaps are noise with two tasks per
# tier, a two-tier inversion contradicts the claim itself.
MIN_INVERSION_TIER_GAP = 2


class OrderingError(ValueError):
    """H1 排不了（任务不全、臂不在表里、tier 缺失）。/ H1 cannot be evaluated."""


@dataclass(frozen=True)
class TaskPoint:
    """一个任务上的一条臂：tier、Λ、区间、是否可判、样本量。／ One (task, arm) point."""

    task: str
    tier: int
    lam: float
    lam_lo: float
    lam_hi: float
    width: float
    undecidable: bool
    n_labels: int
    source_kind: str
    n_seeds: int


# ---- 1. 任务与点 / Tasks and points ----


def load_tasks(configs: Mapping[str, Mapping[str, Any]] | None = None) -> dict[str, dict[str, Any]]:
    """PolyBench 主表任务（`role=main`）：tier / 来源 / 样本量。／ Main PolyBench tasks."""
    source = configs if configs is not None else load_all_task_configs()
    tasks: dict[str, dict[str, Any]] = {}
    for task_id, config in source.items():
        if str(config.get("benchmark")) != BENCHMARK or str(config.get("role")) != "main":
            continue
        if config.get("tier") is None:
            raise OrderingError(f"{task_id} 没有 tier，H1 排不了")
        tasks[task_id] = {
            "tier": int(config["tier"]),
            "source_kind": str(config.get("source_kind", "")),
            "n_labels": int(config.get("n_labels") or 0),
            "property": str(config.get("property", "")),
        }
    if len(tasks) < 3:
        raise OrderingError(f"PolyBench 主表只有 {len(tasks)} 个任务，凑不出三层")
    return tasks


def _read_lambda_rows(path: Path | None = None) -> list[dict[str, str]]:
    target = path or LAMBDA_PATH
    if not target.exists():
        raise OrderingError(f"缺 {target}：先跑 scripts/06c_lambda.py")
    import csv

    with target.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def load_points(
    arm: str,
    *,
    tasks: Mapping[str, Mapping[str, Any]] | None = None,
    rows: Sequence[Mapping[str, str]] | None = None,
) -> list[TaskPoint]:
    """一条链臂在全部 PolyBench 主表任务上的点，按 tier 再按 task 排序。
    The chosen arm's points, one per task, sorted by tier then task id.

    中文：输入：臂名（`a3_graph` / `a4_polybert`）、可选的 tasks 与 rows（测试用）。
          输出：`TaskPoint` 列表。口径：λ 取种子算术平均；`undecidable` 取"任一种子为真"。
          异常：`OrderingError` —— 某个任务在这条臂上没有行。
    """
    task_info = dict(tasks) if tasks is not None else load_tasks()
    source = list(rows) if rows is not None else _read_lambda_rows()
    points: list[TaskPoint] = []
    for task_id, info in task_info.items():
        sub = [row for row in source if row.get("task") == task_id and row.get("arm") == arm]
        if not sub:
            raise OrderingError(f"{task_id} 上没有 {arm} 的 λ 行（先跑 06c）")
        lam = float(np.mean([float(row["lambda"]) for row in sub]))
        lo = float(np.mean([float(row["lambda_lo"]) for row in sub]))
        hi = float(np.mean([float(row["lambda_hi"]) for row in sub]))
        width = float(np.mean([float(row["lambda_width"]) for row in sub]))
        undecidable = any(str(row.get("undecidable", "")).lower() == "true" for row in sub)
        points.append(
            TaskPoint(
                task=task_id,
                tier=int(info["tier"]),
                lam=lam,
                lam_lo=lo,
                lam_hi=hi,
                width=width,
                undecidable=undecidable,
                n_labels=int(info["n_labels"]),
                source_kind=str(info["source_kind"]),
                n_seeds=len(sub),
            )
        )
    points.sort(key=lambda point: (point.tier, point.task))
    return points


# ---- 2. 秩与相关 / Ranks and correlation ----


def average_ranks(values: Iterable[float]) -> np.ndarray:
    """平均秩（并列取平均），1 起。／ Average ranks, 1-based, ties averaged.

    中文：手写而不是调 scipy：6 个点的公式一眼能核，而 `scipy.stats.rankdata` 的默认行为
          （`method="average"`）要另外写一条测试才能确认——这里不需要那层不确定。
    """
    array = np.asarray(list(values), dtype=float)
    if array.size == 0:
        raise OrderingError("空数组没有秩")
    order = array.argsort(kind="stable")
    ranks = np.empty(array.size, dtype=float)
    ranks[order] = np.arange(1, array.size + 1, dtype=float)
    for value in np.unique(array):
        hit = array == value
        if hit.sum() > 1:
            ranks[hit] = float(ranks[hit].mean())
    return ranks


def spearman(x: Iterable[float], y: Iterable[float]) -> float:
    """Spearman ρ = 两列平均秩的 Pearson 相关。／ Pearson correlation of the average ranks.

    中文：任一侧无变异（全是同一个 tier）时返回 `nan`——那时"排序"这个词本身没有意义，
          返回 0 会让它读成"不相关"，而实际是"不可评"。调用方必须自己判 `nan`。
    """
    first = average_ranks(x)
    second = average_ranks(y)
    if first.std() == 0.0 or second.std() == 0.0:
        return float("nan")
    return float(np.corrcoef(first, second)[0, 1])


def cross_layer_inversions(points: Sequence[TaskPoint]) -> list[dict[str, Any]]:
    """跨 ≥2 层的倒序：低层的 Λ 反而更大。／ Inversions across at least two tiers.

    中文：判据是 `tier_i + 2 <= tier_j and lam_i > lam_j`——即"层低两层以上，Λ 却更高"。
          相邻层的倒序不列：每层只有 2 个任务，相邻对调本来就常见，把它也算作倒序会让
          "跨 ≥2 层"这个限定失去意义。
    """
    found: list[dict[str, Any]] = []
    ordered = sorted(points, key=lambda point: (point.tier, point.task))
    for i, low in enumerate(ordered):
        for high in ordered[i + 1 :]:
            if high.tier - low.tier >= MIN_INVERSION_TIER_GAP and low.lam > high.lam:
                found.append(
                    {
                        "low_task": low.task,
                        "low_tier": low.tier,
                        "low_lambda": low.lam,
                        "high_task": high.task,
                        "high_tier": high.tier,
                        "high_lambda": high.lam,
                        "gap": high.tier - low.tier,
                        "undecidable": bool(low.undecidable or high.undecidable),
                    }
                )
    return found


def verdict(points: Sequence[TaskPoint]) -> dict[str, Any]:
    """按提案 §H1 的两个证伪条件判。／ Apply H1's two falsification conditions.

    中文：输入 points（一条臂的全部点）。输出 dict：`rho`、`n_points`、`inversions`、
          `result`（`H1_refuted` / `H1_not_refuted` / `not_evaluable`）、`reasons`。
          **任一**条件成立即证伪：ρ < 0.5，或出现跨 ≥2 层倒序。两条件各自落一个 reason，
          不合并成一句——读者要能看出是哪一条先触发的。
          这是提案的原句，**不许换一种排序方式再试**（D-4 验收标准②）。
    """
    rho = spearman([p.tier for p in points], [p.lam for p in points])
    inversions = cross_layer_inversions(points)
    reasons: list[str] = []
    if not np.isfinite(rho):
        return {
            "rho": rho,
            "n_points": len(points),
            "inversions": inversions,
            "result": "not_evaluable",
            "reasons": ["所有点挤在同一个 tier，排序无定义"],
        }
    if rho < RHO_THRESHOLD:
        reasons.append(f"Spearman ρ = {rho:+.4f} < {RHO_THRESHOLD}")
    if inversions:
        reasons.append(f"{len(inversions)} 处跨 ≥2 层倒序")
    return {
        "rho": rho,
        "n_points": len(points),
        "inversions": inversions,
        "result": "H1_refuted" if reasons else "H1_not_refuted",
        "reasons": reasons,
    }


def decidable_subset(points: Sequence[TaskPoint]) -> list[TaskPoint]:
    """只留 `undecidable=False` 的点（敏感性分析用，**不是**主判据）。
    Points with a usable interval - a sensitivity view, never the headline.
    """
    return [point for point in points if not point.undecidable]


# ---- 3. 落表 / Write the table ----


ORDERING_COLUMNS = (
    "arm",
    "scope",
    "task",
    "tier",
    "property",
    "source_kind",
    "n_labels",
    "n_seeds",
    "lambda",
    "lambda_lo",
    "lambda_hi",
    "lambda_width",
    "undecidable",
    "rank_tier",
    "rank_lambda",
    "spearman_rho",
    "n_points",
    "n_cross_layer_inversions",
    "result",
    "note",
)


def ordering_rows(
    arms: Sequence[str] = CHAIN_ARMS,
    *,
    tasks: Mapping[str, Mapping[str, Any]] | None = None,
    rows: Sequence[Mapping[str, str]] | None = None,
) -> list[dict[str, Any]]:
    """`h1_ordering.csv` 的行：逐点 + 每臂 3 行汇总（主判据、可判子集、全点重复）。"""
    task_info = dict(tasks) if tasks is not None else load_tasks()
    source = list(rows) if rows is not None else _read_lambda_rows()
    out: list[dict[str, Any]] = []
    for arm in arms:
        points = load_points(arm, tasks=task_info, rows=source)
        rank_tier = average_ranks([point.tier for point in points])
        rank_lambda = average_ranks([point.lam for point in points])
        for rank_t, rank_l, point in zip(rank_tier, rank_lambda, points):
            out.append(
                {
                    "arm": arm,
                    "scope": "point",
                    "task": point.task,
                    "tier": point.tier,
                    "property": task_info[point.task]["property"],
                    "source_kind": point.source_kind,
                    "n_labels": point.n_labels,
                    "n_seeds": point.n_seeds,
                    "lambda": f"{point.lam:.6f}",
                    "lambda_lo": f"{point.lam_lo:.6f}",
                    "lambda_hi": f"{point.lam_hi:.6f}",
                    "lambda_width": f"{point.width:.6f}",
                    "undecidable": str(point.undecidable).lower(),
                    "rank_tier": f"{rank_t:.1f}",
                    "rank_lambda": f"{rank_l:.1f}",
                    "spearman_rho": "",
                    "n_points": len(points),
                    "n_cross_layer_inversions": "",
                    "result": "",
                    "note": task_info[point.task]["property"],
                }
            )
        # 三行汇总：主判据（全部点）、可判子集、以及"跨层倒序是不是只由不可判点造成"。
        for scope, subset, note in (
            ("summary_all", points, "主判据：提案 §H1 的原口径（全部 6 个点）"),
            ("summary_decidable", decidable_subset(points), "敏感性：只留 undecidable=false 的点"),
        ):
            if not subset:
                out.append(
                    {
                        "arm": arm,
                        "scope": scope,
                        "task": "",
                        "tier": "",
                        "property": "",
                        "source_kind": "",
                        "n_labels": "",
                        "n_seeds": "",
                        "lambda": "",
                        "lambda_lo": "",
                        "lambda_hi": "",
                        "lambda_width": "",
                        "undecidable": "",
                        "rank_tier": "",
                        "rank_lambda": "",
                        "spearman_rho": "",
                        "n_points": 0,
                        "n_cross_layer_inversions": "",
                        "result": "not_evaluable",
                        "note": "没有可判点",
                    }
                )
                continue
            found = verdict(subset)
            out.append(
                {
                    "arm": arm,
                    "scope": scope,
                    "task": "",
                    "tier": "",
                    "property": "",
                    "source_kind": "",
                    "n_labels": "",
                    "n_seeds": "",
                    "lambda": "",
                    "lambda_lo": "",
                    "lambda_hi": "",
                    "lambda_width": "",
                    "undecidable": "",
                    "rank_tier": "",
                    "rank_lambda": "",
                    "spearman_rho": f"{found['rho']:+.6f}" if np.isfinite(found["rho"]) else "nan",
                    "n_points": found["n_points"],
                    "n_cross_layer_inversions": len(found["inversions"]),
                    "result": found["result"],
                    "note": note + ("；" + "；".join(found["reasons"]) if found["reasons"] else ""),
                }
            )
    return out


def write_table(
    path: Path | None = None,
    *,
    arms: Sequence[str] = CHAIN_ARMS,
    rows: Sequence[Mapping[str, str]] | None = None,
    tasks: Mapping[str, Mapping[str, Any]] | None = None,
) -> Path:
    """写 `h1_ordering.csv`（UTF-8 / LF）。／ Write the ordering table."""
    import csv

    target = path or ORDERING_PATH
    target.parent.mkdir(parents=True, exist_ok=True)
    records = ordering_rows(arms, tasks=tasks, rows=rows)
    with target.open("w", encoding="utf-8", newline="\n") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(ORDERING_COLUMNS), lineterminator="\n")
        writer.writeheader()
        for record in records:
            writer.writerow({column: record.get(column, "") for column in ORDERING_COLUMNS})
    return target


def explain(rows: Sequence[Mapping[str, Any]]) -> str:
    """把汇总行排成人读的几行。／ Render the summary rows."""
    lines = []
    for arm in CHAIN_ARMS:
        for scope, label in (("summary_all", "全部点"), ("summary_decidable", "只含可判点")):
            hit = next(
                (row for row in rows if row["arm"] == arm and row["scope"] == scope), None
            )
            if hit is None:
                continue
            lines.append(
                f"  {arm:<14} {label:<10} n={hit['n_points']:<2} ρ={hit['spearman_rho']:<9} "
                f"跨层倒序 {hit['n_cross_layer_inversions']:<2} → {hit['result']}"
            )
    return "\n".join(lines)
