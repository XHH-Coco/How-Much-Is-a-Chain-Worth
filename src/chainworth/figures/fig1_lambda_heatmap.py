"""图 1：Λ 热图（任务 × 链臂，每格带 95% 区间）。／ Figure 1: the Lambda heatmap.

中文：
    用途：分工 D-6 的第 1 张图，也是"Λ 有多大"这一句的证据（D-7 结论①）。
    读法上的两个决定：
      ① **行按基准分组，不跨基准比较颜色深浅。** R1（§2.1）禁止跨基准合并排序：
         PolyBench 是固定 70/30、PolyBench26 是官方 5 折、OpenPoly 是自建 5 折，
         三者的 Λ 可比但不可混排。分组线画出来，就是为了让"哪几行能比"一眼可见。
      ② **缺失格是灰 `n/a`，不是 0。** 一个没跑的任务画成 0 会被读成"测出来链信息没用"，
         这是最省事也最严重的一类假数据（D-6 验收标准④）。
    数据来源：`results/aggregate/lambda_ci.csv`（每 (task, arm, seed) 一行）。
    每格的值是三个种子的算术平均；区间是三个种子 `lambda_lo`/`lambda_hi` 的均值，
    **不是**合并后的 95% 区间——合并要重做 bootstrap，这里不假装有。
"""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np

from chainworth.figures import common as C

# 行顺序固定：先基准分组，再按 tier / 性质名。换顺序会改变"哪几行挨着"，
# 而读者的第一反应是"挨着的可以比"，所以这个列表是口径的一部分。
# Fixed row order: grouped by benchmark, then by tier. Changing it changes what looks comparable.
ROWS = (
    ("polybench", "pb_t1_tg", "Tg（tier 1）"),
    ("polybench", "pb_t1_tm", "Tm（tier 1）"),
    ("polybench", "pb_t2_modulus", "模量（tier 2）"),
    ("polybench", "pb_t2_permeability", "渗透率（tier 2）"),
    ("polybench", "pb_t3_dielectric_loss", "介电损耗（tier 3）"),
    ("polybench", "pb_t3_iwlt", "IWLT（tier 3）"),
    ("polybench26", "pb26_polymetrix_tg", "PolyMetriX Tg"),
    ("polybench26", "pb26_homopolymer_density", "均聚物密度"),
    ("openpoly", "op_tg", "Tg（OpenPoly）"),
    ("openpoly", "op_tm", "Tm（OpenPoly）"),
    ("openpoly", "op_dielectric_constant_total", "介电常数（OpenPoly）"),
)
ARMS = (("a3_graph", "A3 图网络", C.COLOR_A3), ("a4_polybert", "A4 冻结编码器", C.COLOR_A4))


def _cell(rows: Sequence[dict[str, str]], task: str, arm: str) -> dict[str, Any] | None:
    """一个格子的均值；**只用 status=ok 的行**，一行都没有就返回 `None`（→ 画 n/a）。
    One cell, averaged over the usable rows only; None when nothing is usable.

    中文：这条修正来自一个真 bug。`pb26_homopolymer_density` 的 5 折里有 3 折 A2 的 R² ≥ 0.95，
          按 D-1 验收标准③这些折**不输出 Λ**（分母趋 0，比值无意义），`lambda` 列写 nan。
          若直接 `np.mean`，一个 nan 会把整格变成 nan，于是这个任务被画成"没跑"——
          而它其实有 2 折 × 3 种子 = 6 个可用的行，Λ 分别是 +0.78 / −0.31。
          **一格里有几个折可算，必须跟着数一起报**，否则读者无法知道这个均值站在几折上。
    English: A real bug fix. Three of this task's five folds have R^2_unit >= 0.95, where D-1
    suppresses Lambda entirely (the denominator is near zero), leaving `nan` in the column. A plain
    `mean` turned the whole cell into nan and the task rendered as "not run" - while it actually has
    six usable rows. How many folds a cell averages over travels with the number.
    """
    sub = [row for row in rows if row["task"] == task and row["arm"] == arm]
    if not sub:
        return None
    usable = [
        row
        for row in sub
        if str(row.get("status", "")) == "ok" and str(row.get("lambda", "")) not in ("", "nan")
    ]
    if not usable:
        return {"missing": True, "n_usable": 0, "n_total": len(sub), "reason": str(sub[0].get("status", ""))}
    return {
        "missing": False,
        "lam": float(np.mean([float(row["lambda"]) for row in usable])),
        "lo": float(np.mean([float(row["lambda_lo"]) for row in usable])),
        "hi": float(np.mean([float(row["lambda_hi"]) for row in usable])),
        "width": float(np.mean([float(row["lambda_width"]) for row in usable])),
        "undecidable": any(str(row.get("undecidable", "")).lower() == "true" for row in usable),
        "n_usable": len(usable),
        "n_total": len(sub),
    }


def build() -> tuple[Any, list[str]]:
    """画图 1。／ Build figure 1."""
    C.use_style()
    import matplotlib.pyplot as plt
    from matplotlib.colors import TwoSlopeNorm

    rows = C.read_csv(C.AGGREGATE / "lambda_ci.csv")
    grid = [[_cell(rows, task, arm) for arm, _, _ in ARMS] for _, task, _ in ROWS]
    values = np.array(
        [
            [
                np.nan if (cell is None or cell["missing"]) else cell["lam"]
                for cell in line
            ]
            for line in grid
        ],
        dtype=float,
    )
    finite = values[np.isfinite(values)]
    # 以 0 为中心的双向色标：Λ 为负是"链信息有害"，那是结果，不是缺失，必须能看出来。
    # Diverging scale centred on zero: a negative Lambda is a result, not a gap.
    limit = float(np.nanmax(np.abs(finite))) if finite.size else 1.0
    norm = TwoSlopeNorm(vmin=-limit, vcenter=0.0, vmax=limit)

    fig, ax = plt.subplots(figsize=(5.8, 5.6))
    fig.subplots_adjust(left=0.38, right=0.98, top=0.86, bottom=0.10)
    for i, (_, _, label) in enumerate(ROWS):
        for j, (arm, _, _) in enumerate(ARMS):
            cell = grid[i][j]
            if cell is None:
                C.mark_missing(ax, j, i, "n/a")
                continue
            if cell["missing"]:
                # 有行但没有一行可算（全部折的分母都不稳）：写 n/a 并说明原因，
                # 不许画成 0，也不许静默留白。
                C.mark_missing(ax, j, i, "n/a")
                ax.text(
                    j, i + 0.3, f"{cell['n_total']} 行全不可算", ha="center", va="center",
                    color="#888888", fontsize=6.5,
                )
                continue
            ax.add_patch(
                plt.Rectangle(
                    (j - 0.5, i - 0.5), 1.0, 1.0,
                    facecolor=plt.cm.RdBu_r(norm(cell["lam"])), edgecolor="white", linewidth=1.2,
                )
            )
            text_color = "white" if abs(cell["lam"]) > 0.55 * limit else "#111111"
            star = "†" if cell["undecidable"] else ""
            ax.text(
                j, i - 0.16,
                f"{cell['lam']:+.3f}{star}",
                ha="center", va="center", color=text_color, fontsize=8.5, fontweight="semibold",
            )
            ax.text(
                j, i + 0.10,
                f"[{cell['lo']:+.2f}, {cell['hi']:+.2f}]",
                ha="center", va="center", color=text_color, fontsize=7,
            )
            if cell["n_usable"] < cell["n_total"]:
                ax.text(
                    j, i + 0.36, f"{cell['n_usable']}/{cell['n_total']} 行可算",
                    ha="center", va="center", color=text_color, fontsize=6, style="italic",
                )

    ax.set_xlim(-0.5, len(ARMS) - 0.5)
    ax.set_ylim(len(ROWS) - 0.5, -0.5)
    ax.set_xticks(range(len(ARMS)))
    ax.set_xticklabels([label for _, label, _ in ARMS])
    ax.set_yticks(range(len(ROWS)))
    ax.set_yticklabels([label for _, _, label in ROWS])
    ax.grid(False)
    for spine in ax.spines.values():
        spine.set_visible(False)

    # 基准分组线：画出"哪几行能比"（R1）。没有它，读者会横着比 PolyBench 与 OpenPoly。
    bounds = [i for i in range(1, len(ROWS)) if ROWS[i][0] != ROWS[i - 1][0]]
    for bound in bounds:
        ax.axhline(bound - 0.5, color="#333333", linewidth=1.4)
    # 分组名放在 y 轴外左侧：图内右侧会被色标压住（第一版就是那样，文字与色标叠在一起）。
    # Group labels sit outside the left spine; inside-right they collided with the colour bar.
    for group, y0, y1 in (
        ("PolyBench", 0, 5),
        ("PolyBench26", 6, 7),
        ("OpenPoly", 8, 10),
    ):
        ax.annotate(
            group, xy=(-0.70, (y0 + y1) / 2), xycoords="data",
            ha="right", va="center", rotation=90, fontsize=8, color="#333333",
            annotation_clip=False,
        )

    ax.set_title("图 1　Λ = 链臂相对单元臂收回的剩余方差\n（每格：均值与 95% 区间；† = 区间宽 > 0.3，排序上不可判）", pad=12)
    fig.colorbar(
        plt.cm.ScalarMappable(norm=norm, cmap="RdBu_r"), ax=ax, shrink=0.72, pad=0.02,
        label="Λ（负值 = 链信息有害）",
    )

    caption = [
        "# 图 1　Λ 热图（任务 × 链臂）",
        "",
        "**这张图在说什么**：每个任务上，链臂（A3 图网络 / A4 冻结编码器）相对单元臂（A2 描述符 + 指纹 + XGBoost）",
        "收回了多少单元臂没解释掉的方差。Λ = (R²_链 − R²_单元) / (1 − R²_单元)。",
        "",
        "**怎么读**",
        "",
        "- 数值是**三种子的算术平均**；方括号是三种子区间的均值，不是合并后的 95% 区间。",
        "- `†` = 该格至少一个种子的自举区间宽 > 0.3（D-2 验收标准③），这些格参与排序没有分辨力。",
        "- **横向不可跨基准比较**：三条分组线之间没有可比性（R1，提案 §2.1）。",
        "- **灰色 `n/a` 是没跑，不是 0**。本仓库没有任何任务把它画成 0。",
        "",
        "**这一版最该注意的两行**",
        "",
        "- `pb_t3_dielectric_loss`：A3 的 Λ = −1.479、A4 = −0.905，两者都 `†`。这个任务的样本量只有 279 行，",
        "  链臂的 R² 是负的（比常数预测还差）。它不是「链信息有害」，而是「这个任务上 GNN 学不动」。",
        "- `pb_t2_modulus` / `pb_t2_permeability`：两个任务的 R²_单元 只有 0.031 与 −0.002，",
        "  Λ 的分母接近 1，Λ 退化成 ΔR²。这两格要读作「单元臂本身没学到东西」。",
        "",
        f"**数据来源**：`results/aggregate/lambda_ci.csv`（{C.run_id_range(rows, 'unit_run')}）。",
    ]
    return fig, caption
