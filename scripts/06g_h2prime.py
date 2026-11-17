"""H2′：把 Λ（比值）与 ΔR²（绝对）两把尺子并排报出来。
H2': report the ratio ruler (Lambda) and the absolute ruler (dR^2) side by side.

中文：
    用途：分工 D-6 的 H2′ 入口。H2 用 Λ 判"数据能不能买到链信息"，结论是"κ 区间整段 < 0
          但 Λ 没减半"。本脚本不改 H2，只把同一批折、同一批档位换成绝对口径再读一遍。
          两把尺子放在同一张表里，判据不合并。
    输入：`results/aggregate/kappa_cells.csv`（40 行）。
    输出：`metric_sensitivity_levels.csv`（逐档位）、`metric_sensitivity.csv`（首末比 + 折级区间）。
    副作用：写上述两张表。异常：缺件时 `MetricSensitivityError`。
    用法：
        python scripts/06g_h2prime.py
        python scripts/06g_h2prime.py --no-write   # 只打印，不落盘
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from chainworth.analysis import metric_sensitivity as MS  # noqa: E402


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="06g_h2prime.py", description="H2′：两把尺子并排报")
    parser.add_argument("--cells", default=str(MS.CELLS_PATH), help="读数表（默认 kappa_cells.csv）")
    parser.add_argument("--n-boot", type=int, default=MS.N_BOOT, help="折级重采样次数（默认 2000）")
    parser.add_argument("--no-write", action="store_true", help="只打印，不写表")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    cells = MS.load_cells(Path(args.cells))
    level_rows, ratio_rows = MS.level_and_ratio_rows(cells, n_boot=args.n_boot)

    print(f"读数 {len(cells)} 行，来自 {len({c.dataset for c in cells})} 个数据集")
    print("=" * 78)
    for dataset in sorted({row['dataset'] for row in level_rows}):
        print(f"\n[{dataset}]")
        print(f"  {'档位':>6} {'n_train':>8} {'折数':>4} {'Λ':>9} {'ΔR²':>9}")
        for level in [row for row in level_rows if row["dataset"] == dataset and row["metric"] == "lambda"]:
            dr = next(
                row
                for row in level_rows
                if row["dataset"] == dataset
                and row["metric"] == "delta_r2"
                and row["level"] == level["level"]
            )
            print(
                f"  {level['level']:>6} {level['n_train_median']:>8} {level['n_folds']:>4} "
                f"{level['value_mean']:>+9.4f} {dr['value_mean']:>+9.4f}"
            )
    print("\n" + "=" * 78)
    print(MS.explain(level_rows, ratio_rows))
    print("\n判据：比值 < 0.5 记 `h2p_halved`。**H2′ 不替代 H2**——两个口径回答的不是同一件事：")
    print("  Λ 问「剩余方差里收回多少」（分母随 n 缩，会放大分子的小变化）；")
    print("  ΔR² 问「R² 绝对涨多少」（分母恒为 1，不随 n 变）。")

    if args.no_write:
        print("\n--no-write：没有写表")
        return 0
    MS.write_all(cells_path=Path(args.cells), n_boot=args.n_boot)
    print(f"\n已写 {MS.LEVELS_PATH.relative_to(REPO_ROOT)} 与 {MS.SUMMARY_PATH.relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
