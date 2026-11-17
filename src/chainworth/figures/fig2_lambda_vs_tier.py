"""图 2：Λ vs 分层（PolyBench 三层，6 个任务）——H1 的证据图。／ Figure 2: Lambda against tier.

中文：
    用途：分工 D-6 的第 2 张图，"Λ 与分层的关系"这一句的证据（D-7 结论②）。
    三个必须在图上出现的读法：
      ① **每个点都要画出来，不只画 ρ。** 6 个点的 Spearman 是"装饰性"的统计量，
         读者要看得出是哪两个点把 ρ 拉下来的。
      ② **介电那一对跨不了基准。** 提案把"介电常数在第 1 层、介电损耗在第 3 层"称作
         对 H1 最锋利的对照，但介电常数在本仓库里只有 OpenPoly 版本（`op_dielectric_constant_total`），
         而 OpenPoly 与 PolyBench 是两个基准——R1 禁止把它们连成一对。所以那一对**不画连线**，
         只在注释里点名。
      ③ **tier 与样本量反序**（ρ = −0.956）。不写这一句，"ρ ≈ 0"会被读成"分层与 Λ 无关"，
         而真实图景是"tier 越高数据越少，两个效应互相抵消"。
    数据来源：`h1_ordering.csv`（逐点行）；**不重算 ρ**，直接从汇总行读。
"""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np

from chainworth.figures import common as C

ARMS = (("a3_graph", "A3 图网络", C.COLOR_A3, "o"), ("a4_polybert", "A4 冻结编码器", C.COLOR_A4, "s"))


def build() -> tuple[Any, list[str]]:
    """画图 2。／ Build figure 2."""
    C.use_style()
    import matplotlib.pyplot as plt

    rows = C.read_csv(C.AGGREGATE / "h1_ordering.csv")
    points = [row for row in rows if row.get("scope") == "point"]
    summaries = [row for row in rows if str(row.get("scope", "")).startswith("summary")]
    if not points:
        raise C.FigureError("h1_ordering.csv 里没有逐点行：先跑 scripts/06e_h1.py")

    fig, ax = plt.subplots(figsize=(6.2, 4.4))
    # tier 只有三个值，两臂的点会叠在一起：按臂水平错开 ±0.06，比随机抖动更可读
    # （随机抖动每次重画都不一样，会破坏"一条命令重画逐值一致"）。
    # Two arms at three tiers would overlap; a fixed horizontal offset is used instead of jitter,
    # because random jitter changes on every redraw and breaks the "redraw reproduces the figure".
    offsets = {"a3_graph": -0.07, "a4_polybert": +0.07}
    for arm, label, color, marker in ARMS:
        mine = sorted(
            [row for row in points if row["arm"] == arm], key=lambda row: (int(row["tier"]), row["task"])
        )
        if not mine:
            continue
        x = np.array([int(row["tier"]) + offsets[arm] for row in mine], dtype=float)
        y = np.array([float(row["lambda"]) for row in mine])
        lo = np.array([float(row["lambda_lo"]) for row in mine])
        hi = np.array([float(row["lambda_hi"]) for row in mine])
        ax.errorbar(
            x, y, yerr=[y - lo, hi - y], fmt=marker, color=color, label=label,
            capsize=3, markersize=6, linewidth=1.2, markeredgecolor="white", markeredgewidth=0.6,
            zorder=3,
        )
        for xi, yi, row in zip(x, y, mine):
            # 标签用任务短名，放上方；只对四个极端点标名字，否则六个标签会互相压住。
            if arm == "a3_graph" and row["task"] in (
                "pb_t1_tm", "pb_t3_dielectric_loss", "pb_t3_iwlt", "pb_t1_tg"
            ):
                ax.annotate(
                    row["task"].replace("pb_", ""), xy=(xi, yi), xytext=(4, 6),
                    textcoords="offset points", fontsize=7, color=color,
                )

    ax.axhline(0.0, color=C.COLOR_REFERENCE, linewidth=1.0, linestyle="--", zorder=1)
    ax.text(
        0.995, 0.02, "虚线 = Λ 0：链臂不比单元臂好", fontsize=7.2, color=C.COLOR_REFERENCE,
        va="bottom", ha="right", transform=ax.transAxes,
    )
    ax.set_xticks([1, 2, 3])
    ax.set_xticklabels(["tier 1\n（最高层）", "tier 2", "tier 3\n（最低层）"])
    ax.set_xlim(0.6, 3.4)
    ax.set_xlabel("PolyBench 自身的三层划分（层号越大 = 越依赖链的完整性）")
    ax.set_ylabel("Λ（链臂相对单元臂收回的剩余方差）")

    # 三个 verdict 写进图里，不放进 caption：读者看图就该知道结论是哪一档。
    texts = []
    for arm, label, color, _ in ARMS:
        row = next(
            (r for r in summaries if r["arm"] == arm and r["scope"] == "summary_all"), None
        )
        if row is None:
            continue
        verdict = row["result"].replace("H1_", "H1 ")
        texts.append(f"{label}：ρ = {row['spearman_rho']}，跨层倒序 {row['n_cross_layer_inversions']} 处 → {verdict}")
    ax.text(
        0.02, 0.03, "\n".join(texts), transform=ax.transAxes, fontsize=7.5, va="bottom",
        bbox=dict(boxstyle="round,pad=0.35", facecolor="white", edgecolor="#cccccc", alpha=0.92),
    )
    ax.text(
        0.98, 0.97,
        "注意：tier 与样本量反序（8,055 / 3,796 → 1,034 / 711 → 279 / 228）\n"
        "tier 越高数据越少，两个效应在 Λ 上方向相反",
        transform=ax.transAxes, fontsize=7.2, va="top", ha="right", color="#7a3b00",
        bbox=dict(boxstyle="round,pad=0.35", facecolor="#fff7ec", edgecolor="#e0c9a6"),
    )
    ax.set_title("图 2　Λ 与 PolyBench 的三层划分（6 个任务，误差棒为 95% 区间）", pad=10)
    ax.legend(loc="upper left", framealpha=0.9)

    caption = [
        "# 图 2　Λ vs 分层（H1）",
        "",
        "**这张图在说什么**：提案 H1 断言「Λ 的排序与 PolyBench 自身的三层划分一致」，",
        "证伪条件是「Spearman ρ < 0.5，或出现跨 ≥2 层的倒序」。按原口径（全部 6 个点），**两臂都是证伪**。",
        "",
        "**怎么读**",
        "",
        "- 每个点是一个任务、三种子的算术平均；误差棒是三种子区间均值的上下界（不是合并后的 95% 区间）。",
        "- 两臂在横轴上固定错开 ±0.07，**不是随机抖动**——抖动每次重画都变，会破坏「一条命令重画逐值一致」。",
        "- 6 个点的 Spearman 是装饰性的统计量；真正要看的是**哪两个点**把 ρ 拉下来。",
        "",
        "**两个必须一起说的读数**",
        "",
        "1. **证伪由 tier 3 的介电损耗单点驱动。** 全部 4 处跨层倒序（两臂各 2 处）**每一处都含 `pb_t3_dielectric_loss`**；",
        "   去掉这一个任务，两臂的 ρ 变成 +0.527 / +0.949，判定翻成 `H1_not_refuted`。",
        "   那个点的区间宽 2.19 / 1.27（`undecidable`），样本量只有 279 行，链臂 R² 是负的。",
        "   **事后剔除来「救」结论是不允许的**（D-73 同型错误），所以这里只报它、不用它改判据。",
        "2. **tier 与样本量完全反序**（ρ = −0.956）：tier 同时代理「需要链信息」与「数据少 30 倍」。",
        "   两者对 Λ 方向相反，H1 在这份数据上**不是干净对照**。",
        "",
        "**介电那个赌注为什么没画**：提案说「介电常数在第 1 层、介电损耗在第 3 层」是对 H1 最锋利的对照，",
        "但介电常数在本仓库只有 OpenPoly 版本，与 PolyBench 是两个基准——R1 禁止连成一对，图里也不连。",
        "",
        f"**数据来源**：`results/aggregate/h1_ordering.csv`（派生表，来自 `lambda_ci.csv`，{C.run_id_range(rows, 'task')} 的逐点行）。",
        "ρ 与判定直接读该表的汇总行，**本图不重算任何统计量**。",
    ]
    return fig, caption
