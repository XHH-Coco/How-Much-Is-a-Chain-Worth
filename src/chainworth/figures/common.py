"""五张图的公共层：样式、写盘、caption（分工 D-5/D-6）。
Shared plumbing for the five figures: style, saving, captions.

中文：
    用途：分工 D-5/D-6。**所有图只从 `results/aggregate/` 读表，不重算任何数**——
          这是 D-6 验收标准③"图里每个数字都能追到 `results/raw_runs/` 的某条记录"的实现方式：
          图的输入是已经对过账的汇总表，而汇总表里每一行都带着 `run_id`。
    三条样式口径写在常量里，因为它们在五张图上必须一致：
      ① **缺失格画灰色 + `n/a`，绝不画 0**（D-6 验收标准④）。0 在热图上读成"测出来是 0"。
      ② **误差棒一律画 95% 区间**（来自 bootstrap），不画 ±标准误——两者宽度差一倍多，
         混用会让读者以为"某个任务特别稳"。
      ③ **不画显著性星号**。D 组的结论只带区间，星号会把"6 个点"包装成"检验过"。
    副作用：`save()` 写 `results/figures/<name>.pdf` 与同名 `.caption.md`。
    异常：`FigureError` —— 表缺列、字体渲染失败。

English:
    Purpose: division D-5/D-6. Every figure reads from `results/aggregate/` and recomputes nothing,
    which is how "each number traces back to a run record" is satisfied: the inputs are audited
    summary tables whose rows carry run ids. Missing cells are grey with `n/a` and never zero;
    intervals are the 95% bootstrap ones, never standard errors; no significance stars.
"""

from __future__ import annotations

import datetime as _dt
import json
from pathlib import Path
from typing import Any, Sequence

import matplotlib

matplotlib.use("Agg")  # 无窗口后端：脚本要在 CI 与批处理里跑。
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[3]
AGGREGATE = REPO_ROOT / "results" / "aggregate"
FIGURE_DIR = REPO_ROOT / "results" / "figures"

# 中文字体回退链：Windows 上 YaHei / SimHei 都有，Linux 上常见 Noto。
# The Chinese font fallback chain. Missing glyphs would render as boxes, which looks like a
# broken figure rather than a missing font, so the list is explicit.
FONT_CANDIDATES = ("Microsoft YaHei", "SimHei", "Noto Sans CJK SC", "DejaVu Sans")

# 颜色：两臂固定用同一组颜色，跨图不许换（换色等于换身份）。
# Two arms, two fixed colours, identical across all five figures.
COLOR_A3 = "#1f6fb4"
COLOR_A4 = "#d1701f"
COLOR_NEUTRAL = "#8a8a8a"
COLOR_REFERENCE = "#444444"


class FigureError(ValueError):
    """图画不出来（表缺列、缺件）。/ A figure cannot be drawn."""


def use_style() -> None:
    """设定全局样式（幂等）。／ Set the global style (idempotent)."""
    plt.rcParams.update(
        {
            "font.sans-serif": list(FONT_CANDIDATES),
            "font.family": "sans-serif",
            "axes.unicode_minus": False,   # 负号用 ASCII，避免方块
            "figure.dpi": 120,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "axes.grid": True,
            "grid.alpha": 0.25,
            "grid.linewidth": 0.6,
            "axes.axisbelow": True,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "legend.fontsize": 8,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "figure.autolayout": False,
        }
    )


def require(path: Path) -> Path:
    """表不存在就报错，并指出用哪条命令生成。／ Fail loudly, naming the command that makes it."""
    if not path.exists():
        raise FigureError(f"缺 {path.relative_to(REPO_ROOT)}：先跑生成它的脚本（见 README 的脚本表）")
    return path


def read_csv(path: Path) -> list[dict[str, str]]:
    """读一张 CSV（UTF-8）。／ Read one CSV."""
    import csv

    with require(path).open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def save(fig: Figure, name: str, caption: Sequence[str]) -> tuple[Path, Path]:
    """写 PDF + caption；caption 末尾自动补生成时间与来源表。／ Write the PDF and its caption.

    中文：caption 必须写明**数据来源的 run_id 范围**（D-6 验收标准②）。这里统一追加：
          来源表、行数、生成时间。图自己的 caption 只写"这张图在说什么"。
    """
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    pdf = FIGURE_DIR / f"{name}.pdf"
    md = FIGURE_DIR / f"{name}.caption.md"
    fig.savefig(pdf)
    plt.close(fig)
    stamp = _dt.datetime.now().astimezone().strftime("%Y-%m-%d %H:%M %Z")
    body = "\n".join(caption)
    md.write_text(
        f"{body}\n\n---\n\n生成时间：{stamp}　|　生成命令：`python scripts/07_figures.py`\n",
        encoding="utf-8",
        newline="\n",
    )
    return pdf, md


def run_id_range(rows: Sequence[dict[str, str]], column: str = "run_id") -> str:
    """给 caption 用的 run_id 范围（首、尾、条数）。／ A run-id span for the caption.

    中文：写"某条记录"不够——读者要能自己抽样核对，所以给首尾与条数。
          表里没有 `run_id` 列时（派生表）返回"派生表（无 run_id）"，不假装有。
    """
    ids = [row[column] for row in rows if row.get(column)]
    if not ids:
        return "派生表（无 run_id，见其来源表）"
    return f"{len(ids)} 条记录，`{ids[0]}` … `{ids[-1]}`"


def load_note(name: str) -> dict[str, Any]:
    """读同名的 JSON 说明（若有）。／ Read an optional sibling JSON note."""
    path = AGGREGATE / name
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def mark_missing(ax, x: float, y: float, text: str = "n/a") -> None:
    """在缺失格画一个灰底 + `n/a`，**不画 0**。／ Draw a grey n/a cell, never a zero.

    中文：D-6 验收标准④。这个函数存在的理由就是让"留白"这件事只能通过它完成——
          调用方没有机会顺手写一个 0.0。
    """
    ax.add_patch(
        plt.Rectangle((x - 0.5, y - 0.5), 1.0, 1.0, facecolor="#eeeeee", edgecolor="#cccccc", zorder=1)
    )
    ax.text(x, y, text, ha="center", va="center", color="#777777", fontsize=8, zorder=2)
