"""图 5：交叉点诊断——数据量涨到多少，链臂（图）才追上/超过经典臂（H4）。／ Figure 5: crossover.

中文：
    用途：分工 D-5 的第 5 张图的 D-6 落版。数据来自 OpenPoly 的 n 扫描：同一 (性质, 折, 种子) 上
          图臂减经典臂的 R² 差，对着 log10(n_train) 画，看它在哪个 n 过零。
    四条必须守住的口径（D-5 验收标准）：
      ① **只有三个性质 → 三个 n\*，是描述性的，不做显著性检验。** 三个数的 Spearman 是装饰，
         不许当证据（验收标准①）。
      ② **观测窗上界要说出来。** OpenPoly 的最大档是 `all`（各性质行数不同），
         若某性质在窗内不过零，就报"窗上界"并**停止外推**，不画虚线延长线（验收标准②）。
      ③ **每个 panel 标出实际 n 值**（30 / 60 / 120 / 240 / all），不标 n 的图不许进汇报材料
         （验收标准③）。
      ④ 差值用**配对**方式算：同一个 (折, 种子) 的两条臂相减，不是均值相减——
         均值相减会把折间差异当成信号。
    数据来源：`openpoly_cpu.csv`（经典臂 A2）与 `openpoly_gpu_index.csv`（图臂 A3）。
"""

from __future__ import annotations

from typing import Any

import numpy as np

from chainworth.figures import common as C

PROPERTY_LABELS = {
    "op_tg": ("Tg（玻璃化转变）", C.COLOR_A3),
    "op_tm": ("Tm（熔点）", C.COLOR_A4),
    "op_dielectric_constant_total": ("介电常数", "#3a7a4a"),
}
LEVEL_ORDER = ("30", "60", "120", "240", "all")


def _paired(unit_rows, graph_rows) -> dict[str, list[tuple[float, float, int]]]:
    """按 (property, 折, 种子) 配对，返回每性质的 (log10 n, ΔR², n_train) 列表。
    Pair the two arms on (property, fold, seed) and return per-property points.

    中文：配对是这里唯一能用的算法。两臂的 n_train 在各档位相同（同一子集指纹），
          所以按 (折, 种子) 对上就是同一批训练行上的一次对比。
    """
    index = {}
    for row in unit_rows:
        key = (row["property"], row["fold"], row["seed"], row["n_level"])
        index[key] = float(row["r2"])
    paired: dict[str, list[tuple[float, float, int]]] = {}
    for row in graph_rows:
        key = (row["property"], row["fold"], row["seed"], row["n_level"])
        if key not in index:
            continue
        n_train = int(row["n_train"])
        delta = float(row["r2"]) - index[key]
        paired.setdefault(row["property"], []).append((float(np.log10(n_train)), delta, n_train))
    return paired


def _crossover(points: list[tuple[float, float, int]]) -> tuple[float | None, str]:
    """用均值曲线做线性插值找过零点；窗内不过零时返回 None 与"窗上界"说明。
    Linear interpolation on the mean curve for the zero crossing, or None with a bound note.

    中文：只用**均值曲线**插值，不拿每个点各自求交点再平均——点的正负交错时那样会得到
          一个不存在的交点。过零落在两档之间时报插值位置；全窗同号时报窗的端点。
    """
    by_tier: dict[float, list[float]] = {}
    for x, delta, _ in points:
        by_tier.setdefault(x, []).append(delta)
    xs = sorted(by_tier)
    means = [float(np.mean(by_tier[x])) for x in xs]
    for i in range(len(xs) - 1):
        if means[i] == 0:
            return xs[i], "窗内过零"
        if means[i] * means[i + 1] < 0:
            t = -means[i] / (means[i + 1] - means[i])
            return xs[i] + t * (xs[i + 1] - xs[i]), "窗内过零（插值）"
    if means[-1] > 0:
        return None, "全窗为正：最小 n 处已超过经典臂，观测窗下界外"
    return None, f"窗上界 {int(10 ** xs[-1]):,} 行处仍为负，不外推"


def build() -> tuple[Any, list[str]]:
    """画图 5。／ Build figure 5."""
    C.use_style()
    import matplotlib.pyplot as plt

    unit_rows = C.read_csv(C.AGGREGATE / "openpoly_cpu.csv")
    graph_rows = C.read_csv(C.AGGREGATE / "openpoly_gpu_index.csv")
    paired = _paired(unit_rows, graph_rows)
    properties = [name for name in PROPERTY_LABELS if name in paired]
    if not properties:
        raise C.FigureError("openpoly 两张表没有可配对的 (性质, 折, 种子)")

    fig, axes = plt.subplots(1, len(properties), figsize=(4.1 * len(properties), 3.9), squeeze=False)
    axes = axes[0]
    lines: list[str] = []

    for ax, name in zip(axes, properties):
        points = paired[name]
        by_tier: dict[float, list[float]] = {}
        sizes: dict[float, int] = {}
        for x, delta, n_train in points:
            by_tier.setdefault(x, []).append(delta)
            sizes[x] = n_train
        xs = sorted(by_tier)
        means = [float(np.mean(by_tier[x])) for x in xs]
        # 误差棒：组内标准差 / sqrt(组内点数)。这是"这一档均值有多准"，不是预测区间。
        errs = [
            float(np.std(by_tier[x], ddof=1) / np.sqrt(len(by_tier[x]))) if len(by_tier[x]) > 1 else 0.0
            for x in xs
        ]
        _, color = PROPERTY_LABELS[name]
        # 散点：每个 (折, 种子) 一个点，让读者看见离散度。
        for x in xs:
            ax.scatter(
                [x] * len(by_tier[x]), by_tier[x], s=9, color=color, alpha=0.35, linewidths=0, zorder=2,
            )
        ax.errorbar(
            xs, means, yerr=errs, fmt="o-", color=color, linewidth=1.8, markersize=5,
            markeredgecolor="white", markeredgewidth=0.6, capsize=3, zorder=3,
        )
        ax.axhline(0.0, color=C.COLOR_REFERENCE, linewidth=1.0, linestyle="--", zorder=1)

        crossing, note = _crossover(points)
        if crossing is not None:
            ax.axvline(crossing, color=color, linewidth=1.0, linestyle=":", alpha=0.8, zorder=1)
            ax.text(
                crossing, ax.get_ylim()[1], f"  n* ≈ {int(10 ** crossing):,}",
                rotation=90, va="top", ha="left", fontsize=7, color=color,
            )
        ax.text(
            0.03, 0.04, note, transform=ax.transAxes, fontsize=6.8, color="#444444",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="#dddddd", alpha=0.9),
        )
        # ③ 每个 panel 标实际 n 值。
        ax.set_xticks(xs)
        ax.set_xticklabels([f"{sizes[x]:,}" for x in xs], rotation=35, ha="right")
        ax.set_xlabel("n_train（实际行数）")
        ax.set_ylabel("R²(图臂) − R²(经典臂)")
        ax.set_title(PROPERTY_LABELS[name][0], fontsize=9.5)
        ticks = {f"{sizes[x]:,}" for x in xs}
        lines.append(
            f"- `{name}`：档位 {sorted(ticks, key=lambda s: int(s.replace(',', '')))}；{note}"
        )

    fig.suptitle(
        "图 5　交叉点诊断（OpenPoly n 扫描；点 = 单个 (折, 种子)，误差棒 = 组内标准误）\n"
        "三个性质 → 三个 n*，描述性结论，不做显著性检验",
        fontsize=9.8, y=1.02,
    )
    fig.tight_layout()

    caption = [
        "# 图 5　交叉点诊断（H4）",
        "",
        "**这张图在说什么**：在 OpenPoly 的 n 扫描上，图臂（A3 MPNN）减经典臂（A2 描述符+指纹+XGBoost）的 R² 差",
        "随训练数据量怎么变，以及在哪个 n 过零。过零意味着「数据涨到这个量，换架构就不值了」。",
        "",
        "**四个口径**",
        "",
        "- 差值**按 (性质, 折, 种子) 配对相减**，不是两条均值曲线相减——后者会把折间差异当成信号。",
        "- 点是一个个 (折, 种子)，误差棒是组内标准误（说明「这一档的均值有多准」，不是预测区间）。",
        "- **只有三个性质 → 三个 n\\***：**描述性，不做显著性检验**（D-5 验收标准①）。",
        "- 观测窗不过零就报**窗上界**并停止外推，不画延长线（验收标准②）。",
        "",
        "**实测**",
        "",
    ] + lines + [
        "",
        "**必须一起说的限制**",
        "",
        "- OpenPoly 的最大档不是固定的 443 行，而是各性质自己的 `all`（见上面各 panel 的刻度）；",
        "  三个性质的窗不一样宽，n\\* 之间**不能直接比较**。",
        "- 经典臂与图臂都只跑了 seed 0/1/2，三档种子；折是自建 5 折（不是官方划分），",
        "  所以这张图回答的是「我们的协议下」的交叉点，不是「领域里」的交叉点。",
        "",
        f"**数据来源**：`results/aggregate/openpoly_cpu.csv`（{len(unit_rows)} 行）与 "
        f"`openpoly_gpu_index.csv`（{len(graph_rows)} 行），配对后 {sum(len(v) for v in paired.values())} 个差。",
    ]
    return fig, caption
