#!/usr/bin/env python
"""Λ 与 ΔR² 的表：从预测重算 R²，按配对 bootstrap 给区间，落 `lambda_ci.csv`。

The Lambda / dR2 table: R^2 recomputed from predictions, intervals from a paired bootstrap, written
to `lambda_ci.csv`.

这个脚本做的四件事
------------------
1. **先跑对账闸门**（`provenance.require_clean()`）。分工 D-8 的验收是"通过率 100% 才允许出图，
   出图脚本开头自动跑一次"；这个脚本虽然还不出图，但它产出的数会直接进图，所以闸门放在更前面。
   对账不过就退出码 1，一行 CSV 都不写——**宁可不产出，也不产出没对过账的数**。
2. **从预测重算 R²**，不读记录里的 `metrics`（D-1 ②）。两条实现各自独立，差 > 1e-6 会在这里
   被 `--check-records` 报出来。
3. **配对 bootstrap**（B=2000、固定种子，D-2）：同一组重采样下标同时抽两条臂，每次重算 R² 再代进 Λ。
4. **落表**：一行一个 (task, arm, fold, seed)，列见 `LAMBDA_COLUMNS`；区间宽度 > 0.3 的行标
   `undecidable`（D-2 ③）——不是删掉，是标出来，"区间太宽"本身是要报的结果。

为什么编号是 06c
----------------
`06` 是 C 的 `06_aggregate.py`、`07` 是 D 的 `07_figures.py`。Λ 表是**出图之前的派生表**，
与 `06b_audit.py`（闸门）同属"06 之后、07 之前"的一层，按 `00b`/`03b`/`04b`/`05b` 的先例取
插入式编号 `06c`（D-67 登记）。

用法 / Usage
------------
    python scripts/06c_lambda.py                 # 跑闸门 + 出表
    python scripts/06c_lambda.py --no-bootstrap  # 只算点估计，不跑 2,000 次重采样
    python scripts/06c_lambda.py --check-records # 额外核"重算 R² 与记录的差 < 1e-6"
    python scripts/06c_lambda.py --no-gate       # 跳过对账闸门（只在闸门本身坏了时用）
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Any, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

import numpy as np  # noqa: E402

from chainworth.analysis import ci as CI  # noqa: E402
from chainworth.analysis import lambda_kappa as LK  # noqa: E402
from chainworth.analysis import provenance as PV  # noqa: E402

AGGREGATE = REPO_ROOT / "results" / "aggregate"
TABLE = AGGREGATE / "lambda_ci.csv"

# 区间宽度超过它，这一格只能报"无法判定"（D-2 ③）。
# Above this interval width a cell is reported as undecidable (D-2 ③).
UNDECIDABLE_WIDTH = 0.3

COLUMNS: tuple[str, ...] = (
    "task",
    "arm",
    "fold",
    "seed",
    "unit_run",
    "chain_run",
    "n_test",
    "r2_unit",
    "r2_chain",
    "delta_r2",
    "delta_r2_lo",
    "delta_r2_hi",
    "lambda",
    "lambda_lo",
    "lambda_hi",
    "lambda_width",
    "n_boot",
    "seed_policy",
    "status",
    "reportable",
    "undecidable",
    "note",
)


def _write(path: Path, columns: Sequence[str], rows: Sequence[dict[str, Any]]) -> Path:
    """写 CSV（UTF-8、显式 LF）。/ Write a CSV with UTF-8 and explicit LF endings."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(columns), lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column, "") for column in columns})
    return path


def _check_records(rows: Sequence[dict[str, Any]], tolerance: float = 1e-6) -> list[str]:
    """核"从预测重算的 R²"与"记录里的 R²"是否一致。
    Check the recomputed R^2 against the R^2 inside the run records.

    中文：输入 rows（含 `unit_run` / `chain_run` / `r2_unit` / `r2_chain`）；输出不一致的描述列表。
          数据来源：`results/raw_runs/*.json` 的 `metrics.r2`。副作用：只读。
          判据：差 > `tolerance` 就报——"差更多说明有一边错了，先查再报"（D-1 ②）。
    English: Returns the mismatches. A difference above the tolerance means one side is wrong.
    """
    records = LK.load_records()
    problems: list[str] = []
    for row in rows:
        for side, run_key in (("unit", "unit_run"), ("chain", "chain_run")):
            run_id = str(row[run_key])
            recorded = float(records[run_id]["metrics"]["r2"])
            recomputed = float(row[f"r2_{side}"])
            if not np.isnan(recorded) and abs(recorded - recomputed) > tolerance:
                problems.append(
                    f"{run_id}：记录 r2={recorded:.12f}，重算 {recomputed:.12f}，"
                    f"差 {abs(recorded - recomputed):.3e}"
                )
    return problems


def _attach_intervals(
    rows: list[dict[str, Any]],
    draws: int,
    seed: int,
) -> list[dict[str, Any]]:
    """给每一对补 bootstrap 区间（Λ 与 ΔR² 各一组）。
    Attach bootstrap intervals to every pair - one for Lambda, one for dR2.

    中文：输入 rows / draws / seed；输出补过列的行（原地修改并返回）。
          数据来源：重新读两条臂的预测 + pack 的 `y`。副作用：只读。
          异常：`LK.LambdaError` / `CI.BootstrapError` —— 预测缺失或点数不足。
    English: Reads both arms' predictions and the pack labels again; mutates and returns the rows.
    """
    labels: dict[str, np.ndarray] = {}
    for row in rows:
        truth = LK._task_labels(str(row["task"]), labels)
        unit_ids, unit_pred = LK._prediction(str(row["unit_run"]))
        chain_ids, chain_pred = LK._prediction(str(row["chain_run"]))
        if not np.array_equal(np.sort(unit_ids), np.sort(chain_ids)):
            # 行集合不同就不做配对重采样：区间会混进"看的不是同一批点"这个额外方差。
            row.update(
                delta_r2_lo="",
                delta_r2_hi="",
                lambda_lo="",
                lambda_hi="",
                lambda_width="",
                n_boot=0,
                undecidable="",
            )
            continue
        order = np.argsort(unit_ids)
        chain_lookup = {int(row_id): value for row_id, value in zip(chain_ids, chain_pred)}
        chain_pred = np.array([chain_lookup[int(i)] for i in unit_ids[order]], dtype=np.float64)
        lam, delta = CI.bootstrap_lambda(
            truth[unit_ids[order]], unit_pred[order], chain_pred, draws=draws, seed=seed
        )
        width = lam.hi - lam.lo if np.isfinite(lam.lo) and np.isfinite(lam.hi) else float("nan")
        row.update(
            delta_r2_lo=delta.lo,
            delta_r2_hi=delta.hi,
            lambda_lo=lam.lo,
            lambda_hi=lam.hi,
            lambda_width=width,
            n_boot=lam.n_boot,
            undecidable=bool(np.isfinite(width) and width > UNDECIDABLE_WIDTH),
        )
    return rows


def main(argv: list[str] | None = None) -> int:
    """入口：闸门 → 算 Λ → 补区间 → 落表。/ Entry: gate, compute, add intervals, write."""
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--out", type=Path, default=TABLE, help="输出 CSV 路径")
    parser.add_argument("--no-bootstrap", action="store_true", help="只算点估计，不跑重采样")
    parser.add_argument("--draws", type=int, default=CI.BOOTSTRAP_DRAWS, help="重采样次数（冻结项）")
    parser.add_argument("--seed", type=int, default=CI.BOOTSTRAP_SEED, help="重采样种子（冻结项）")
    parser.add_argument("--check-records", action="store_true", help="核重算 R² 与记录的差 < 1e-6")
    parser.add_argument("--no-gate", action="store_true", help="跳过对账闸门（仅在闸门本身可疑时用）")
    args = parser.parse_args(argv)

    if not args.no_gate:
        try:
            summary = PV.require_clean()
        except PV.ProvenanceError as error:
            print(str(error), file=sys.stderr)
            print("对账未通过，未写出 lambda_ci.csv。先跑 python scripts/06b_audit.py provenance 看明细。", file=sys.stderr)
            return 1
        print(f"对账通过：{summary['n_ok']} ok / {summary['n_fail']} fail / {summary['n_exempt']} exempt")

    rows = LK.build_lambda_table()
    print(f"配对数 {len(rows)}（unit = {LK.UNIT_ARM}，chain = {' / '.join(LK.CHAIN_ARMS)}）")

    if not args.no_bootstrap:
        rows = _attach_intervals(rows, args.draws, args.seed)
        print(f"bootstrap B={args.draws} seed={args.seed} 完成")

    if args.check_records:
        problems = _check_records(rows)
        if problems:
            print(f"重算 R² 与记录不一致 {len(problems)} 条：", file=sys.stderr)
            for problem in problems[:10]:
                print(f"  {problem}", file=sys.stderr)
            return 1
        print(f"重算 R² 与记录逐条一致（容差 1e-6，{len(rows) * 2} 个值）")

    for row in rows:
        row.setdefault(
            "seed_policy", f"paired-residual-bootstrap B={args.draws} seed={args.seed}"
        )

    target = _write(args.out, COLUMNS, rows)
    # `--out` 给相对路径时 `target` 不在 `REPO_ROOT` 底下，`relative_to` 会抛异常——
    # 那是在**写盘成功之后**打印时炸的，看起来像没写出东西。按绝对路径兜底。
    shown = target.resolve().relative_to(REPO_ROOT) if target.is_absolute() else target
    print(f"已写入 {Path(shown).as_posix()}（{len(rows)} 行）")

    summary_rows = LK.summarise_by_arm(rows)
    print()
    print(f"{'task':30s} {'arm':14s} {'可报/配对':>10s} {'Λ 均值':>9s} {'Λ run-sd':>9s} {'ΔR² 均值':>10s}")
    for item in summary_rows:
        def _fmt(value: Any) -> str:
            return "   —   " if value == "" else f"{float(value):+.4f}"

        print(
            f"{item['task']:30s} {item['arm']:14s} "
            f"{str(item['n_reportable']) + '/' + str(item['n_pairs']):>10s} "
            f"{_fmt(item['lambda_mean']):>9s} {_fmt(item['lambda_sd_runs']):>9s} "
            f"{_fmt(item['delta_r2_mean']):>10s}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
