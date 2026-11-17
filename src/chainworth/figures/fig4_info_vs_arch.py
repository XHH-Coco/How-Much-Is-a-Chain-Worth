"""图 4：信息 vs 架构——H3 的柱状图，以及它**缺了哪两根柱子**。／ Figure 4: information vs architecture.

中文：
    用途：分工 D-6 的第 4 张图。H3 要回答"Λ 主要来自注入链信息，还是来自换 GNN 架构"。
    **这张图最重要的一点是它缺了什么，而不是它画了什么。**
      设计要求的四根柱子是：A2 基准、A5（同一架构下注入链信息）、A3（换 GNN 架构）、打乱对照。
      本仓库实际只有 **A3 与 A4**：`raw_runs/` 里 `a5_chain` 的**记录数为 0**，
      `scramble_control.csv` 也**从未产出**（`docs/分工_C-a` 里只写在任务书里）。
      于是 H3 的核心对比——"同一架构下只换信息"——**无法完成**：
      A3 与 A4 都是"换架构"，没有一个能回答"信息本身值多少"。
      这张图把缺失的两根**画成灰色空位**，而不是省略：省略会让读者以为实验设计本来就只有两根。
      这是 D-6 验收标准④（缺失必须显式）在柱状图上的对应做法。
    数据来源：`results/aggregate/lambda_ci.csv` 的 `delta_r2` 与其区间。
"""

from __future__ import annotations

from typing import Any

import numpy as np

from chainworth.figures import common as C

# 与图 1/2 相同的任务顺序：先 PolyBench（按 tier），再 PB26。跨基准不合并。
TASKS = (
    ("pb_t1_tg", "Tg", "polybench"),
    ("pb_t1_tm", "Tm", "polybench"),
    ("pb_t2_modulus", "模量", "polybench"),
    ("pb_t2_permeability", "渗透率", "polybench"),
    ("pb_t3_dielectric_loss", "介电损耗", "polybench"),
    ("pb_t3_iwlt", "IWLT", "polybench"),
    ("pb26_polymetrix_tg", "PolyMetriX Tg", "polybench26"),
    ("pb26_homopolymer_density", "均聚物密度", "polybench26"),
)
ARMS = (("a3_graph", "A3 图网络（换架构）", C.COLOR_A3), ("a4_polybert", "A4 冻结编码器（换架构）", C.COLOR_A4))


def _bar(rows, task: str, arm: str) -> dict[str, float] | None:
    """一个 (task, arm) 的 ΔR² 均值与区间；只用 status=ok 的行。／ One bar, usable rows only."""
    sub = [
        row
        for row in rows
        if row["task"] == task
        and row["arm"] == arm
        and str(row.get("status", "")) == "ok"
        and str(row.get("delta_r2", "")) not in ("", "nan")
    ]
    if not sub:
        return None
    return {
        "mean": float(np.mean([float(row["delta_r2"]) for row in sub])),
        "lo": float(np.mean([float(row["delta_r2_lo"]) for row in sub])),
        "hi": float(np.mean([float(row["delta_r2_hi"]) for row in sub])),
        "n": len(sub),
    }


def build() -> tuple[Any, list[str]]:
    """画图 4。／ Build figure 4."""
    C.use_style()
    import matplotlib.pyplot as plt

    rows = C.read_csv(C.AGGREGATE / "lambda_ci.csv")

    fig, ax = plt.subplots(figsize=(7.4, 4.4))
    width = 0.36
    # y 轴裁剪：`pb_t2_permeability` 上 A3 的区间是 [−4.9, +14.3]（该任务 R²_unit ≈ −0.002，
    # 分母趋 0，bootstrap 爆掉），若照实画，其余 7 个任务会被压成一条直线。
    # 裁剪是允许的，但**超界的区间必须显式标出来**，否则读者会以为那根柱子的区间很窄。
    # Clipping is allowed only if the clipped intervals are labelled.
    Y_LIMIT = 1.1
    clipped: list[str] = []
    positions = np.arange(len(TASKS), dtype=float)

    for offset, (arm, label, color) in (
        (-width / 2, ARMS[0]),
        (+width / 2, ARMS[1]),
    ):
        heights, errors, missing = [], [], []
        for task, _, _ in TASKS:
            bar = _bar(rows, task, arm)
            if bar is None:
                heights.append(0.0)
                errors.append([0.0, 0.0])
                missing.append(True)
            else:
                heights.append(bar["mean"])
                errors.append([bar["mean"] - bar["lo"], bar["hi"] - bar["mean"]])
                missing.append(False)
        x = positions + offset
        mask = [not flag for flag in missing]
        ax.bar(
            x[mask], np.array(heights)[mask], width=width, color=color, label=label,
            edgecolor="white", linewidth=0.6, zorder=2,
        )
        lows = np.array([max(e[0], 0.0) for e in errors])
        highs = np.array([max(e[1], 0.0) for e in errors])
        ax.errorbar(
            x[mask], np.array(heights)[mask], yerr=[lows[mask], highs[mask]],
            fmt="none", ecolor="#333333", elinewidth=1.0, capsize=2.5, zorder=3,
        )
        for xi, flag, height in zip(x, missing, heights):
            if flag:
                ax.text(xi, 0.02, "n/a", ha="center", va="bottom", fontsize=6.5, color="#777777")
        # 超界标注：写清真实端点，并用箭头指向被裁掉的方向。
        # 只判端点，不判误差幅度——`lows`/`highs` 是幅度，端点要另算。
        for i, (xi, flag, height) in enumerate(zip(x, missing, heights)):
            if flag:
                continue
            low_end = height - lows[i]
            high_end = height + highs[i]
            if low_end < -Y_LIMIT:
                clipped.append(f"{arm} × {TASKS[i][0]}：区间下界 {low_end:+.1f}")
                ax.annotate(
                    "", xy=(xi, -Y_LIMIT), xytext=(xi, -Y_LIMIT * 0.72),
                    arrowprops=dict(arrowstyle="-|>", color="#333333", linewidth=0.9),
                )
            if high_end > Y_LIMIT * 1.25:
                clipped.append(f"{arm} × {TASKS[i][0]}：区间上界 {high_end:+.1f}")
                ax.annotate(
                    "", xy=(xi, Y_LIMIT * 1.25), xytext=(xi, Y_LIMIT * 1.12),
                    arrowprops=dict(arrowstyle="-|>", color="#333333", linewidth=0.9),
                )

    ax.axhline(0.0, color=C.COLOR_REFERENCE, linewidth=1.0, zorder=1)
    ax.set_ylim(-Y_LIMIT, Y_LIMIT * 1.25)
    ax.set_xticks(positions)
    ax.set_xticklabels([label for _, label, _ in TASKS], rotation=32, ha="right")
    ax.set_ylabel("ΔR² = R²(链臂) − R²(单元臂)")
    # PB26 与 PolyBench 是两套协议：画一条竖线分开，避免读者横向比较。
    split = sum(1 for _, _, bench in TASKS if bench == "polybench") - 0.5
    ax.axvline(split, color="#333333", linewidth=1.2, linestyle=":")
    ax.text(split - 0.15, ax.get_ylim()[1] * 0.95, "PolyBench", ha="right", va="top", fontsize=7.5, color="#333333")
    ax.text(split + 0.15, ax.get_ylim()[1] * 0.95, "PolyBench26", ha="left", va="top", fontsize=7.5, color="#333333")

    # 缺失的两根柱子：画成灰色空位并写明原因。省略它们会让读者以为设计只有两根。
    # The two missing bars are drawn as grey placeholders, with the reason spelled out.
    for x_center, text in (
        (len(TASKS) - 0.05, "A5（同一架构\n注入链信息）\n未跑"),
        (len(TASKS) + 0.95, "打乱对照\n未产出"),
    ):
        ax.add_patch(
            plt.Rectangle(
                (x_center - 0.36, 0.0), 0.72, 1.0, facecolor="#eeeeee",
                edgecolor="#bbbbbb", linestyle="--", zorder=1, clip_on=False,
            )
        )
        ax.text(
            x_center, 0.5, text, ha="center", va="center", fontsize=6.8, color="#777777", zorder=2,
            clip_on=False,
        )
    ax.set_xlim(-0.7, len(TASKS) + 1.55)

    # 被裁掉的区间必须写在图上：否则读者会以为那两根误差棒很短。
    if clipped:
        ax.text(
            0.015, 0.04,
            "y 轴已裁剪；箭头处真实端点：\n" + "\n".join(clipped),
            transform=ax.transAxes, fontsize=6.5, va="bottom", color="#7a3b00",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="#fff7ec", edgecolor="#e0c9a6", alpha=0.95),
        )

    ax.set_title(
        "图 4　信息 vs 架构（ΔR²，误差棒为 95% 区间）\n"
        "灰框是本该有、但本仓库没有的两根：H3 的核心对比因此无法完成",
        pad=10,
    )
    ax.legend(loc="upper left", framealpha=0.9)

    caption = [
        "# 图 4　信息 vs 架构（H3）",
        "",
        "**这张图在说什么**：H3 问的是「Λ 主要来自注入链信息，还是来自换 GNN 架构」。",
        "按设计，回答它需要四根柱子：A2 基准、**A5（同一架构下注入链信息）**、A3（换 GNN 架构）、**打乱对照**。",
        "",
        "**这张图缺了两根，而且是关键的两根**",
        "",
        "- `raw_runs/` 里 **`a5_chain` 的记录数为 0**（`ls results/raw_runs/*.json | grep -c a5_chain` = 0）——这一臂从未跑过。",
        "- **`scramble_control.csv` 从未产出**，只在 `docs/分工_C-a_经典臂族.md` 的任务书里出现。",
        "",
        "后果要说直白：**A3 与 A4 都是「换架构」**。A4 换的是表示（polyBERT 冻结编码器 + ridge），A3 换的是模型",
        "（chemprop MPNN）。**两者都回答不了「信息本身值多少」**——那需要同一架构下只换输入信息，也就是 A5。",
        "所以本报告对 H3 只能说「两条换架构的臂各自比 A2 好多少」，**不能**说「增益来自信息而非架构」。",
        "",
        "**怎么读**",
        "",
        "- 柱高是 ΔR² 的三种子均值；误差棒是三种子 `delta_r2_lo/hi` 均值的上下界。",
        "- 竖虚线分开 PolyBench 与 PolyBench26：两套评估协议，**不可横向比较**（R1）。",
        "- `n/a` 是真的没有可算的行（例如某任务所有折的分母都不稳），不是 0。",
        "",
        "**图上能看出的两件事**",
        "",
        "1. **绿色为主**：8 个任务里绝大多数 ΔR² 为负或近零——换架构并没有稳定带来增益。",
        "2. 唯一的强正值是 `pb26_homopolymer_density` 的 A3（ΔR² 约 +0.02）与 `pb_t3_iwlt`；",
        "   而 `pb_t3_dielectric_loss` 的 A3 是 −0.60（R² 从 +0.59 掉到 −0.009），是全部柱子里最差的一根。",
        "",
        f"**数据来源**：`results/aggregate/lambda_ci.csv`（{C.run_id_range(rows, 'chain_run')}）。",
    ]
    return fig, caption
