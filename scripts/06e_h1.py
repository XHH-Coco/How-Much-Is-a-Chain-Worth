#!/usr/bin/env python
"""H1 的分层排序检验：Λ 的排序与 PolyBench 的三层划分是否一致（分工 D-4）。

H1's ordering test: does Lambda order itself the way PolyBench's three tiers say it should?

这个脚本做的三件事
------------------
1. **先跑对账闸门**（`provenance.require_clean()`）。ρ 会进图 2，所以和 `06c`/`06d` 一样，
   对账不过就退出码 1，一行 CSV 都不写。
2. **两条链臂各排一次**（`a3_graph` 与 `a4_polybert`）。不合并成一个"平均链臂"：
   换臂可能换排序，而 H1 说的是"Λ 的排序"，引用时必须带臂名。
3. **按提案的两个证伪条件判**（ρ < 0.5，或跨 ≥2 层倒序），并**同时报**只含可判点的敏感性版本。

为什么把敏感性写进同一张表
--------------------------
6 个点里几个的区间宽到 > 0.3（`undecidable`），剔除它们再排一次会得到不同的 ρ。
不报这一版，读者无法判断"ρ 小"是"Λ 与 tier 无关"还是"6 个点里有一半是噪声"。
但**主判据只有一个**：全部点、按提案原口径。敏感性版本不进结论，只作读数。

用法 / Usage
------------
    python scripts/06e_h1.py              # 闸门 + 两张表
    python scripts/06e_h1.py --no-gate    # 跳过对账闸门（只在闸门本身坏了时用）
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from chainworth.analysis import heterogeneity as HT  # noqa: E402
from chainworth.analysis import ordering as OR  # noqa: E402
from chainworth.analysis import provenance as PV  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="H1 的分层排序检验（PolyBench 6 任务）")
    parser.add_argument("--no-gate", action="store_true", help="跳过四向对账闸门")
    args = parser.parse_args(argv)

    if not args.no_gate:
        try:
            summary = PV.require_clean()
        except PV.ProvenanceError as error:
            print(f"对账没过，H1 表不产出：{error}", file=sys.stderr)
            return 1
        print(f"对账闸门：{summary['n_ok']} ok / {summary['n_exempt']} exempt / {summary['n_fail']} fail")
    else:
        print("跳过对账闸门（--no-gate）：产出的数没有对账凭据")

    ordering_path = OR.write_table()
    heterogeneity_path = HT.write_table()
    rows = OR.ordering_rows()

    print(f"\n已写入 {ordering_path.relative_to(REPO_ROOT).as_posix()}（{len(rows)} 行）")
    print(f"已写入 {heterogeneity_path.relative_to(REPO_ROOT).as_posix()}")

    print("\n逐点（tier / Λ / 样本量 / 是否可判）：")
    tasks = OR.load_tasks()
    for arm in OR.CHAIN_ARMS:
        print(f"  {arm}")
        for point in OR.load_points(arm, tasks=tasks):
            flag = " undecidable" if point.undecidable else ""
            print(
                f"    tier {point.tier}  Λ={point.lam:+.4f} "
                f"[{point.lam_lo:+.3f}, {point.lam_hi:+.3f}]  "
                f"n_labels={point.n_labels:>5d}  {point.task}{flag}"
            )
    print("\nH1 判据（提案 §H1：ρ < 0.5 或出现跨 ≥2 层倒序 → 证伪）：")
    print(OR.explain(rows))

    print("\n跨 ≥2 层倒序明细：")
    for arm in OR.CHAIN_ARMS:
        points = OR.load_points(arm, tasks=tasks)
        found = OR.cross_layer_inversions(points)
        if not found:
            print(f"  {arm:<14} 无")
            continue
        for item in found:
            tag = "（含不可判点）" if item["undecidable"] else ""
            print(
                f"  {arm:<14} tier{item['low_tier']} {item['low_task']} Λ={item['low_lambda']:+.4f}"
                f"  >  tier{item['high_tier']} {item['high_task']} Λ={item['high_lambda']:+.4f}{tag}"
            )

    print("\n来源异质性（提案 §2.2 要求按 experimental / DFT / MD 分层的实测结果）：")
    for arm in OR.CHAIN_ARMS:
        points = OR.load_points(arm, tasks=tasks)
        groups = HT.source_groups(points)
        print(f"  {arm:<14} n_sources={len(groups)}：{', '.join(groups)}")
    print("  → 只有一个 label source，跨来源 τ 与 leave-one-source-out 均无定义（表里写 not_evaluable）")

    print("\n混淆检查（tier 同时代理了什么）：")
    for arm in OR.CHAIN_ARMS:
        points = OR.load_points(arm, tasks=tasks)
        for row in HT.confound_rows(arm, points):
            print(f"  {arm:<14} ρ(tier, log₁₀ n_labels) = {row['spearman_rho']} → {row['result']}")
    print("  → tier 越高样本量越小（8,055 / 3,796 → 1,034 / 711 → 279 / 228），")
    print("    「需要链信息」与「数据少 30 倍」在 Λ 上方向相反，H1 不是干净对照")

    print("\n单任务杠杆（去掉一个任务重算 ρ——6 个点时这是唯一有内容的稳健性检查）：")
    for arm in OR.CHAIN_ARMS:
        points = OR.load_points(arm, tasks=tasks)
        for row in HT.leave_one_task_out_rows(arm, points):
            print(f"  {arm:<14} {row['note']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
