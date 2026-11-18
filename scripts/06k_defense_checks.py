"""答辩补算的入口：把审阅清单里"只能现算"的七个数一次算清并落盘。
Stage entry 06k: settle the seven numbers a review can only answer by recomputation.

中文：
    用途：2026-11-18 外部审阅提出约 50 条质疑，其中 7 条不能靠改措辞回答。本脚本只读
          `results/aggregate/` 的既有表（λ、H1 排序、κ 逐格、零信息地板），输出：
          `results/aggregate/defense_cells.csv`（19 格逐格）与 `defense_checks.json`（其余）。
    **不重算任何 Λ**：384 行、321 个配对值全部原样读取；`raw_runs/` 一个字节不碰。
    三条纪律：
      ① 算不出来写 `null` + 方法名，不填 0（D-73）；
      ② 每个数字带 `sources`，指回表名与列名；
      ③ H1 的 ρ 用**平均秩上的 Pearson**，与 `h1_ordering.csv` 同口径——用
         `1−6Σd²/n(n²−1)` 在 tier 有并列时会得到 0.043 而不是 0.000。
    用法：
        python scripts/06k_defense_checks.py            # 写两张表并打印摘要
        python scripts/06k_defense_checks.py --check    # 只打印，不写文件
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from chainworth.analysis import defense_checks as DC  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true",
                        help="只打印摘要，不写 results/aggregate/ 下的两张表")
    args = parser.parse_args(argv)

    payload = DC.build()
    print(DC.report(payload))
    if not args.check:
        cells, summary = DC.write_all(payload)
        print(f"\n写入 {cells.relative_to(REPO_ROOT)}")
        print(f"写入 {summary.relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
