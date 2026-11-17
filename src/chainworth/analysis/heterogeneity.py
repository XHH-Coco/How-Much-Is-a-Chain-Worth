"""来源异质性与单点杠杆：H1 的排序是不是某一个任务撑起来的（分工 D-4 的后半）。
Is H1's ordering an artefact of one source, or of one task?

中文：
    用途：分工 D-4 的"来源异质性审计"。提案 §2.2 要求按 label source（experimental / DFT / MD）
          分层重算排序，并做 leave-one-source-out。
    **本仓库的实测前提与分工书的预设不同，必须写清楚**：PolyBench 主表这 6 个任务在配置里
          的 `source_kind` **全部是 `experimental`**（`configs/tasks/pb_*.yaml`，11 个任务无一例外）。
          也就是说"跨来源"这一维在这份数据上**没有变异**，τ 与 leave-one-source-out 都无从谈起——
          这不是"跳过"，而是"这一维没有数据可分层"。
    因此本模块做两件事：
      ① 如实报 `n_sources`；只有 1 个来源时 `kendall_tau` 与 `leave_one_source_out` 返回
         `not_evaluable`，不硬造一个数。
      ② 补一个**在这份数据上有内容**的替代分析：`leave_one_task_out`。6 个点去掉一个重算 ρ——
         因为 H1 的 ρ 只有 6 个点，任何一个点都可能单独决定符号，而这比"来源"更能解释
         "ρ = 0 是真不相关，还是被某一个极端任务拉平了"。
    数据来源：与 `analysis.ordering` 同一套（`lambda_ci.csv` + 任务配置）。
    配置来源：无。副作用：`write_table()` 写 `h1_heterogeneity.csv`。
    异常：`OrderingError`（由 `ordering` 抛出）。

English:
    Purpose: division D, the source-heterogeneity audit. The proposal asks for a per-label-source
    ranking plus a leave-one-source-out check. **The actual data cannot support that**: all six
    PolyBench main tasks carry `source_kind: experimental`, so the "source" dimension has no
    variation here. This module therefore reports `n_sources` honestly and marks the cross-source
    Kendall tau as `not_evaluable` instead of inventing a number, then adds the analysis that *does*
    have content on six points: leave-one-task-out.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from chainworth.analysis import ordering as OR

REPO_ROOT = Path(__file__).resolve().parents[3]
HETEROGENEITY_PATH = REPO_ROOT / "results" / "aggregate" / "h1_heterogeneity.csv"

# 提案 §2.2 与 D-4 验收标准③：τ ≥ 0.6 才允许说"分层一致"。
TAU_THRESHOLD = 0.6

HETEROGENEITY_COLUMNS = (
    "arm",
    "analysis",
    "group",
    "n_points",
    "n_sources",
    "spearman_rho",
    "kendall_tau",
    "result",
    "note",
)


def kendall_tau(x: Iterable[float], y: Iterable[float]) -> float:
    """Kendall τ-b（带并列修正）。／ Kendall tau-b with tie correction.

    中文：用 τ-b 而不是 τ-a，因为 tier 一定大量并列（每层 2 个任务）；τ-a 在并列时
          分母还是 n(n−1)/2，会把 τ 系统性地压低，看起来像"排序不一致"——这是并列数据上
          最容易犯的错。全侧无变异时返回 `nan`。
    """
    first = np.asarray(list(x), dtype=float)
    second = np.asarray(list(y), dtype=float)
    if first.size != second.size:
        raise OR.OrderingError("两列长度不同")
    n = first.size
    if n < 3:
        return float("nan")
    concordant = discordant = 0
    for i in range(n):
        for j in range(i + 1, n):
            dx = first[i] - first[j]
            dy = second[i] - second[j]
            if dx == 0 or dy == 0:
                continue
            if dx * dy > 0:
                concordant += 1
            else:
                discordant += 1

    def tie_term(values: np.ndarray) -> float:
        total = 0.0
        for value in np.unique(values):
            count = float((values == value).sum())
            total += count * (count - 1) / 2.0
        return total

    n0 = n * (n - 1) / 2.0
    denominator = np.sqrt((n0 - tie_term(first)) * (n0 - tie_term(second)))
    if denominator == 0.0:
        return float("nan")
    return float((concordant - discordant) / denominator)


def source_groups(points: Sequence[OR.TaskPoint]) -> dict[str, list[OR.TaskPoint]]:
    """按 `source_kind` 分组。／ Group points by label source."""
    groups: dict[str, list[OR.TaskPoint]] = {}
    for point in points:
        groups.setdefault(point.source_kind or "(未登记)", []).append(point)
    return dict(sorted(groups.items()))


def by_source_rows(arm: str, points: Sequence[OR.TaskPoint]) -> list[dict[str, Any]]:
    """按来源分层：每组一个 ρ（组内 tier vs Λ）与一组 τ（组间排序一致性）。

    中文：**单来源时 τ 不可算**——τ 衡量的是"两组各自的排序是否一致"，只有一组时这个比较
          的对象不存在。此时只报组内 ρ，`kendall_tau` 留 `nan`，`result` 写
          `not_evaluable_single_source`。如果哪天任务配置里出现 DFT / MD 来源，这段代码
          不用改就会给出 τ。
    """
    groups = source_groups(points)
    rows: list[dict[str, Any]] = []
    for name, members in groups.items():
        rows.append(
            {
                "arm": arm,
                "analysis": "by_source",
                "group": name,
                "n_points": len(members),
                "n_sources": len(groups),
                "spearman_rho": (
                    f"{OR.spearman([p.tier for p in members], [p.lam for p in members]):+.6f}"
                    if len(members) >= 3
                    else "nan"
                ),
                "kendall_tau": "nan",
                "result": "not_evaluable_single_source" if len(groups) == 1 else "per_source_only",
                "note": (
                    f"全部 {len(points)} 个点的 source_kind 都是 `{name}`，跨来源 τ 无定义"
                    if len(groups) == 1
                    else f"{name} 组内 {len(members)} 个点"
                ),
            }
        )
    if len(groups) == 1:
        return rows
    # 两个来源以上：先按组内排序，再比较各组的排序（τ 在"每组的 Λ 秩"上算）。
    names = list(groups)
    for i, left in enumerate(names):
        for right in names[i + 1 :]:
            common_tiers = sorted(
                {p.tier for p in groups[left]} & {p.tier for p in groups[right]}
            )
            left_mean = [
                float(np.mean([p.lam for p in groups[left] if p.tier == tier]))
                for tier in common_tiers
            ]
            right_mean = [
                float(np.mean([p.lam for p in groups[right] if p.tier == tier]))
                for tier in common_tiers
            ]
            tau = kendall_tau(left_mean, right_mean)
            rows.append(
                {
                    "arm": arm,
                    "analysis": "by_source_pair",
                    "group": f"{left}|{right}",
                    "n_points": len(common_tiers) * 2,
                    "n_sources": len(groups),
                    "spearman_rho": "",
                    "kendall_tau": f"{tau:+.6f}" if np.isfinite(tau) else "nan",
                    "result": (
                        "sources_agree" if np.isfinite(tau) and tau >= TAU_THRESHOLD
                        else "sources_disagree"
                    ),
                    "note": f"在 {len(common_tiers)} 个共有 tier 上比较两组的 Λ 排序",
                }
            )
    return rows


def leave_one_source_out_rows(arm: str, points: Sequence[OR.TaskPoint]) -> list[dict[str, Any]]:
    """去掉一个来源重算 ρ。／ Leave-one-source-out.

    中文：单来源时去掉它就一个点都不剩——这类"检查"不许产出数字，只落一行
          `not_evaluable`。把"无法检查"写成"检查通过"是审计里最常见的假绿。
    """
    groups = source_groups(points)
    if len(groups) < 2:
        return [
            {
                "arm": arm,
                "analysis": "leave_one_source_out",
                "group": "(全部)",
                "n_points": len(points),
                "n_sources": len(groups),
                "spearman_rho": "nan",
                "kendall_tau": "nan",
                "result": "not_evaluable_single_source",
                "note": f"只有一个来源（{next(iter(groups))}），去掉即空集",
            }
        ]
    rows: list[dict[str, Any]] = []
    for dropped in groups:
        kept = [point for point in points if (point.source_kind or "(未登记)") != dropped]
        rho = OR.spearman([p.tier for p in kept], [p.lam for p in kept])
        rows.append(
            {
                "arm": arm,
                "analysis": "leave_one_source_out",
                "group": f"drop:{dropped}",
                "n_points": len(kept),
                "n_sources": len(groups),
                "spearman_rho": f"{rho:+.6f}" if np.isfinite(rho) else "nan",
                "kendall_tau": "",
                "result": "ok",
                "note": f"去掉 `{dropped}` 后剩 {len(kept)} 个点",
            }
        )
    return rows


def leave_one_task_out_rows(arm: str, points: Sequence[OR.TaskPoint]) -> list[dict[str, Any]]:
    """去掉一个任务重算 ρ：6 个点时，看 ρ 是不是一个点决定的。
    Leave-one-task-out: with six points, one task can decide the sign.

    中文：这是本模块在**当前数据上唯一有内容**的稳健性检查。判据是"结论是否翻转"
          （ρ 从 < 0.5 变成 ≥ 0.5，或反过来），以及"剩 5 个点时是否仍出现跨层倒序"。
          不是在找"最好的剔除方式"——所以 6 行全报，不挑。
    """
    base = OR.verdict(points)
    rows: list[dict[str, Any]] = []
    for drop in points:
        kept = [point for point in points if point.task != drop.task]
        if len(kept) < 3:
            continue
        found = OR.verdict(kept)
        flipped = base["result"] != found["result"]
        rows.append(
            {
                "arm": arm,
                "analysis": "leave_one_task_out",
                "group": f"drop:{drop.task}",
                "n_points": len(kept),
                "n_sources": len(source_groups(points)),
                "spearman_rho": f"{found['rho']:+.6f}" if np.isfinite(found["rho"]) else "nan",
                "kendall_tau": "",
                "result": "verdict_flips" if flipped else "verdict_holds",
                "note": (
                    f"去掉 tier{drop.tier} 的 {drop.task}（n_labels={drop.n_labels}）后 "
                    f"ρ={found['rho']:+.4f}、跨层倒序 {len(found['inversions'])} 处 → {found['result']}"
                ),
            }
        )
    return rows


def confound_rows(arm: str, points: Sequence[OR.TaskPoint]) -> list[dict[str, Any]]:
    """tier 与样本量的混淆：ρ(tier, log₁₀ n_labels)。／ The tier/sample-size confound.

    中文：**这是本模块最重要的一行，也是 H1 被证伪的机制。** 实测 6 个任务的 tier 与
          `n_labels` 完全反序：tier1 = 8,055 / 3,796 行，tier2 = 1,034 / 711，tier3 = 279 / 228。
          于是 tier 同时是两件事的代理：①"需要链信息"（H1 的主张），②"数据量少 30 倍"
          （对任何模型都是坏消息，对参数化 GNN 尤其坏）。
          两者在 Λ 上方向相反，H1 的检验因此不是"干净对照"——这必须与 ρ 同时报，
          否则"ρ ≈ 0"会被读成"分层与 Λ 无关"，而实际可能是"两个效应互相抵消"。
    English: On these six tasks the tier and the label count are perfectly anti-ordered
    (tier1 8,055/3,796 rows vs tier3 279/228). The tier therefore proxies both "needs chain
    information" and "has thirty times less data", which pull Lambda in opposite directions. The
    confound has to travel with rho, or "rho near zero" will be read as "tier is irrelevant".
    """
    rho = OR.spearman([p.tier for p in points], [np.log10(max(p.n_labels, 1)) for p in points])
    return [
        {
            "arm": arm,
            "analysis": "confound",
            "group": "tier_vs_log10_n_labels",
            "n_points": len(points),
            "n_sources": len(source_groups(points)),
            "spearman_rho": f"{rho:+.6f}" if np.isfinite(rho) else "nan",
            "kendall_tau": "",
            "result": "confounded" if np.isfinite(rho) and rho <= -0.5 else "no_strong_confound",
            "note": (
                "tier 与样本量反序（tier1 8,055/3,796 行 vs tier3 279/228）：tier 同时代理"
                "「需要链信息」与「数据少 30 倍」，两者对 Λ 方向相反——H1 不是干净对照"
                if np.isfinite(rho) and rho <= -0.5
                else "tier 与样本量没有强反序关系"
            ),
        }
    ]


def write_table(
    path: Path | None = None,
    *,
    arms: Sequence[str] = OR.CHAIN_ARMS,
    rows: Sequence[Mapping[str, str]] | None = None,
    tasks: Mapping[str, Mapping[str, Any]] | None = None,
) -> Path:
    """写 `h1_heterogeneity.csv`（UTF-8 / LF）。／ Write the heterogeneity table."""
    target = path or HETEROGENEITY_PATH
    target.parent.mkdir(parents=True, exist_ok=True)
    source = list(rows) if rows is not None else OR._read_lambda_rows()
    task_info = dict(tasks) if tasks is not None else OR.load_tasks()
    records: list[dict[str, Any]] = []
    for arm in arms:
        points = OR.load_points(arm, tasks=task_info, rows=source)
        records.extend(by_source_rows(arm, points))
        records.extend(leave_one_source_out_rows(arm, points))
        records.extend(confound_rows(arm, points))
        records.extend(leave_one_task_out_rows(arm, points))
    with target.open("w", encoding="utf-8", newline="\n") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(HETEROGENEITY_COLUMNS), lineterminator="\n")
        writer.writeheader()
        for record in records:
            writer.writerow({column: record.get(column, "") for column in HETEROGENEITY_COLUMNS})
    return target
