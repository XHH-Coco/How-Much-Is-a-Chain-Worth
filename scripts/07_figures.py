#!/usr/bin/env python
"""一条命令重画 D 组的全部五张图（分工 D-6）。

One command redraws all five figures from `results/aggregate/`.

三条口径
--------
1. **先跑对账闸门**（`provenance.require_clean()`）。D-8 的验收写着"通过率 100% 才允许出图"，
   所以这里和 06c/06d/06e 一样：对账不过，一张图都不写。
2. **只从 `results/aggregate/` 读表，不重算任何数**。图里每个数字都能追到某张汇总表的一行，
   而那一行带着 `run_id`（D-6 验收标准③）。
3. **图只出 PDF**（矢向量，能放大看坐标轴）。PNG 草稿不入库（D-6 的"不入库"清单）。

重画后应当**逐值一致**（允许字体渲染差异）：图里没有任何随机量，两臂的横向错开是固定数，
不是抖动（D-6 验收标准①）。

用法 / Usage
------------
    python scripts/07_figures.py                 # 闸门 + 五张图
    python scripts/07_figures.py --no-gate       # 跳过对账闸门（只在闸门本身坏了时用）
    python scripts/07_figures.py --only fig3     # 只重画一张
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from chainworth.analysis import provenance as PV  # noqa: E402
from chainworth.figures import common as C  # noqa: E402
from chainworth.figures import fig1_lambda_heatmap as F1  # noqa: E402
from chainworth.figures import fig2_lambda_vs_tier as F2  # noqa: E402
from chainworth.figures import fig3_lambda_vs_n as F3  # noqa: E402
from chainworth.figures import fig4_info_vs_arch as F4  # noqa: E402
from chainworth.figures import fig5_crossover as F5  # noqa: E402

FIGURES = (
    ("fig1", "fig1_lambda_heatmap", F1, "图 1　Λ 热图（任务 × 链臂）"),
    ("fig2", "fig2_lambda_vs_tier", F2, "图 2　Λ vs 分层（H1）"),
    ("fig3", "fig3_lambda_vs_n", F3, "图 3　Λ vs log10 n（κ，首页图）"),
    ("fig4", "fig4_info_vs_arch", F4, "图 4　信息 vs 架构（H3，缺两根柱）"),
    ("fig5", "fig5_crossover", F5, "图 5　交叉点诊断（H4）"),
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="重画 D 组的五张图（只读汇总表）")
    parser.add_argument("--no-gate", action="store_true", help="跳过四向对账闸门")
    parser.add_argument("--only", choices=[key for key, _, _, _ in FIGURES], help="只画一张")
    args = parser.parse_args(argv)

    if not args.no_gate:
        try:
            summary = PV.require_clean()
        except PV.ProvenanceError as error:
            print(f"对账没过，图不产出：{error}", file=sys.stderr)
            return 1
        print(f"对账闸门：{summary['n_ok']} ok / {summary['n_exempt']} exempt / {summary['n_fail']} fail")
    else:
        print("跳过对账闸门（--no-gate）：产出的图没有对账凭据")

    C.use_style()
    written: list[Path] = []
    for key, module_name, module, label in FIGURES:
        if args.only and args.only != key:
            continue
        figure, caption = module.build()
        pdf, md = C.save(figure, module_name, caption)
        written.append(pdf)
        print(f"  {label:<34} → {pdf.relative_to(REPO_ROOT).as_posix()}（caption 同名 .caption.md）")

    if not written:
        print("没有图被写出（--only 选中的项不存在）", file=sys.stderr)
        return 1
    print(f"\n共 {len(written)} 张图；caption 在 results/figures/*.caption.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
