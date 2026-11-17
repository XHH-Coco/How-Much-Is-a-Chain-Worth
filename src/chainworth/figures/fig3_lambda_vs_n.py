"""图 3：Λ 随训练数据量的变化（κ 与它的区间）——首页唯一一张图。／ Figure 3: Lambda vs log10 n.

中文：
    用途：分工 D-6 的第 3 张图，"数据能不能买到链信息"这一句的全部证据（D-7 结论③）。
    三条画法上的决定：
      ① **每条细线是一折**，粗线是五折均值。只画均值会把"五折同号"这个事实藏起来，
         而那正是 κ 为负的唯一支撑。
      ② **分母不稳的档位要标出来**（阴影带）。`1 − R²_unit < 0.05` 的地方 Λ 被放大成噪声，
         负号主要来自那一段；把它画成普通区段，读者会以为整条曲线都同样可信。
      ③ **均聚集不画斜率**，只画两点连线并标 `no slope`（D-74）。它的三个档位里顶上两点
         只隔 0.158 dex，拟合出来的线"看起来很确定"，这正是要防的。
    数据来源：`kappa_cells.csv`（40 个逐格结果）与 `kappa.csv`（每数据集一行）。
    **不重算 Λ、不重算 κ**：图只读表。
"""

from __future__ import annotations

from typing import Any

import numpy as np

from chainworth.figures import common as C

LABELS = {
    "md5000_alternating_cp": "交替共聚（MD_5000 alternating）",
    "md5000_homopolymer_cp": "均聚物（MD_5000 homopolymer）",
}


def build() -> tuple[Any, list[str]]:
    """画图 3。／ Build figure 3."""
    C.use_style()
    import matplotlib.pyplot as plt

    cells = C.read_csv(C.AGGREGATE / "kappa_cells.csv")
    summary = C.read_csv(C.AGGREGATE / "kappa.csv")
    if not cells or not summary:
        raise C.FigureError("缺 kappa_cells.csv / kappa.csv：先跑 scripts/06d_kappa.py")

    datasets = [row["dataset"] for row in summary]
    fig, axes = plt.subplots(1, len(datasets), figsize=(6.4 * min(len(datasets), 2), 4.6), squeeze=False)
    axes = axes[0]

    for ax, dataset in zip(axes, datasets):
        mine = [row for row in cells if row["dataset"] == dataset]
        if not mine:
            raise C.FigureError(f"{dataset} 在 kappa_cells.csv 里没有格子")
        row = next(item for item in summary if item["dataset"] == dataset)
        tiers = sorted({int(item["n_train"]) for item in mine})
        x_all = np.log10(tiers)

        # ① 逐折细线：五折同号这件事只有画出来才看得见。
        for fold in sorted({item["fold"] for item in mine}):
            line = sorted(
                [item for item in mine if item["fold"] == fold], key=lambda item: int(item["n_train"])
            )
            ax.plot(
                np.log10([int(item["n_train"]) for item in line]),
                [float(item["lambda"]) for item in line],
                color="#b9b9b9", linewidth=1.0, marker=".", markersize=4, zorder=2,
            )
        mean = [float(np.mean([float(item["lambda"]) for item in mine if int(item["n_train"]) == tier])) for tier in tiers]
        ax.plot(x_all, mean, color=C.COLOR_A3, linewidth=2.2, marker="o", markersize=6,
                markeredgecolor="white", markeredgewidth=0.7, label="五折均值", zorder=3)

        # ② 分母不稳的档位：阴影带。判据来自该档是否有一半以上的格子标了 unstable。
        for tier in tiers:
            block = [item for item in mine if int(item["n_train"]) == tier]
            unstable = sum(1 for item in block if str(item["unstable"]).lower() == "true")
            if unstable * 2 >= len(block):
                ax.axvspan(
                    np.log10(tier) - 0.075, np.log10(tier) + 0.075,
                    color="#f2c6c6", alpha=0.45, zorder=1,
                )
        ax.axhline(0.0, color=C.COLOR_REFERENCE, linewidth=0.9, linestyle="--", zorder=1)

        kappa = row.get("kappa", "")
        status = row.get("slope_status", "")
        # ③ 两条交代：能拟合就报 κ 与区间；不能拟合就只报连线并写明原因。
        if kappa:
            note = (
                f"κ = {float(kappa):+.4f}  [{float(row['kappa_lo']):+.4f}, {float(row['kappa_hi']):+.4f}]  "
                f"（{row['n_levels']} 档 × {row['n_folds']} 折，跨度 {float(row['span_dex']):.2f} dex）"
            )
        else:
            chord = row.get("endpoint_slope", "")
            note = (
                f"不拟合斜率（{status}）\n两点连线斜率 {float(chord):+.3f}（无不确定度）"
                if chord
                else f"不拟合斜率（{status}）"
            )
        ax.text(
            0.03, 0.97, note, transform=ax.transAxes, va="top", ha="left", fontsize=8,
            bbox=dict(boxstyle="round,pad=0.35", facecolor="white", edgecolor="#cccccc", alpha=0.92),
        )
        ax.set_title(LABELS.get(dataset, dataset), fontsize=10)
        ax.set_xlabel("log10(n_train)")
        ax.set_ylabel("Λ")
        # 档位刻度标实际行数，不是 log 值：读者要知道"all"是 22,242 还是 1,440。
        ax.set_xticks(x_all)
        ax.set_xticklabels([f"{tier:,}" for tier in tiers], rotation=30, ha="right")

    axes[0].legend(loc="lower left", framealpha=0.9)
    fig.suptitle(
        "图 3　Λ 随训练数据量的变化　|　阴影带 = 该档 1 − R²(单元) < 0.05，Λ 在此被放大",
        fontsize=10.5, y=0.99,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.95))

    alternating = next((row for row in summary if row["dataset"] == "md5000_alternating_cp"), None)
    homo = next((row for row in summary if row["dataset"] == "md5000_homopolymer_cp"), None)
    header = [
        "# 图 3　Λ vs log10 n（κ）",
        "",
        "**这张图在说什么**：数据涨 10 倍，Λ 移动多少。这是 H2（「数据无法补偿链信息」）的唯一证据，",
        "也是报告首页那张图：它把「这个缺口像缺一个坐标，还是像抽样噪声」变成一个斜率。",
        "",
        "**怎么读**",
        "",
        "- 灰细线是**每一折**，蓝粗线是五折均值。五折同号这件事只有细线看得见，而它正是 κ 为负的支撑。",
        "- 粉色阴影的档位 `1 − R²_unit < 0.05`：A2 已接近饱和，Λ 的比值被放大。**负号主要来自这一段。**",
        "- 横轴标签是**实际训练行数**，不是 log 值——「all」在交替集是 22,242 行，在均聚集只有 1,440 行。",
        "",
    ]
    if alternating is not None and alternating.get("kappa"):
        header += [
            f"**左图（交替集）**：κ = {float(alternating['kappa']):+.4f} "
            f"[{float(alternating['kappa_lo']):+.4f}, {float(alternating['kappa_hi']):+.4f}]，"
            f"Λ 从 {float(alternating['lambda_first']):+.4f} 到 {float(alternating['lambda_last']):+.4f}，"
            f"减半比 {alternating['lambda_ratio']} → `{alternating['verdict']}`。",
            "",
            "两个条件只满足一个：区间整段 < 0 **成立**，Λ 减半 **不成立**（0.650 > 0.5）。按提案的证伪条件，H2 没有被推翻。",
            "",
            "**但这句不能单独引**：Λ 是驼峰不是直线（300 档 +0.295 → 1,000 档 +0.503 → all 档 +0.192），",
            "只用分母 ≥ 0.05 的两档（300 / 1,000），斜率变成 **+0.398（正号）**。",
            "去掉最小档重拟合则得 −0.224 [−0.241, −0.205]，符号不翻转（`kappa_leverage.csv`）。",
            "四件事一起说才成立（D-73、D-74）。",
            "",
        ]
    if homo is not None:
        header += [
            f"**右图（均聚集）**：不拟合斜率（`{homo.get('slope_status')}`）——"
            f"名义 3 档，但 1,000 → 1,440 只隔 0.158 dex，实质是两点连线。",
            "两点连线斜率（`endpoint_slope`）为 "
            f"{float(homo['endpoint_slope']):+.3f}，**没有不确定度，不是 κ**。",
            "",
            "此前一版曾报 κ = +0.6949 [+0.1876, +1.3605]，那是被**同一折**（斜率 2.075）拉起来的，已作废（D-74）。",
            "",
        ]
    header += [
        "**只含测试集抽样不确定度**：区间来自按测试点配对的自举（B = 2,000）。",
        "GNN 的初始化方差（只有 seed 0）**不在**区间里；折间离散度给出的量级参照见 `docs/Kappa Table Notes.md` §5。",
        "",
        f"**数据来源**：`results/aggregate/kappa_cells.csv`（40 格）与 `kappa.csv`（{len(summary)} 行），"
        f"逐格的 run 见表内 `unit_run_id` / `chain_run_id`。",
    ]
    return fig, header
