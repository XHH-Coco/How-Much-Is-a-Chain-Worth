"""阶段入口 01：把 PolyBench 原始 CSV 调成"一个任务一张表"，并留下单位判定。
Stage entry 01: harmonise the raw PolyBench CSVs into one table per task, recording unit decisions.

中文：
    用途：本项目"原始字节 -> 可建模标签"的唯一入口。跑完会得到三样东西：
          data/interim/harmonised/<task_id>_harmonised.parquet（调好的表，不入库）
          data/manifests/unit_decisions.csv（每条判定一行，入库、只可追加）
          data/manifests/harmonised_manifest.csv（每任务的行数与哈希，入库）
    依赖：data/raw/polybench/*.csv（阶段 00 交付，39 份 md5 已核）；
          src/chainworth/io/harmonise.py（全部逻辑在这里，本文件只做参数与日志）。
    配置：无 yaml——任务清单与单位依据在 harmonise.TASK_SPECS 里。
    对外：命令行。`--verify` 只读不写，把 A-3 的四条验收标准跑成检查。
    被谁用：人工（阶段运行）、CI（`--verify` 当回归）。

English:
    Purpose: the single entry from raw bytes to model-ready labels. A run produces:
    the harmonised parquet per task (not committed), unit_decisions.csv (append-only, committed)
    and harmonised_manifest.csv (counts and hashes, committed).
    Depends on: data/raw/polybench/*.csv (stage 00, all 39 md5 verified) and
    src/chainworth/io/harmonise.py, which holds all the logic; this file only parses arguments
    and logs.
    Config: no yaml - the task list and the unit rationale live in harmonise.TASK_SPECS.
    Public API: the command line; `--verify` is read-only and turns A-3's acceptance criteria
    into checks.
    Callers: humans running stage 01, and CI using `--verify` as a regression gate.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# 让 `python scripts/01_harmonise.py` 直接可用：把仓库根的 src/ 放进 sys.path。
# 不做 pip install -e（P0 冻结的环境里不该出现本仓库的 egg-info）。
# Make `python scripts/01_harmonise.py` work without installing the package: put src/ on
# sys.path. No `pip install -e`, so the frozen P0 environments stay free of this repo's egg-info.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from chainworth.io.harmonise import (  # noqa: E402  —— 必须在 sys.path 调整之后导入 / import after sys.path
    EXCLUDED_TASKS,
    MANIFEST_DIR,
    TASK_SPECS,
    build_spotcheck,
    harmonise_all,
    verify_outputs,
    write_spotcheck,
)

LOGGER = logging.getLogger("chainworth.harmonise.cli")


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """解析命令行参数，并把"哪些组合不允许"挡在运行之前。
    Parse arguments and reject invalid combinations before any work starts.

    中文：
        用途：把脚本的接口固定下来，避免"加个开关就顺手改了默认行为"。
        输入：argv —— 命令行参数（None 表示读 sys.argv）。
        输出：argparse.Namespace，含 dry_run / limit / task / verify / spotcheck / verbose。
        数据来源：命令行。
        配置来源：无。
        副作用：无。
        异常：SystemExit —— 参数不合法（argparse 自己抛）。

    English:
        Purpose: freeze the CLI so a new flag cannot silently change default behaviour.
        Args: argv - arguments (None reads sys.argv).
        Returns: Namespace with dry_run / limit / task / verify / spotcheck / verbose.
        Data source: the command line. Config: none. Side effects: none.
        Raises: SystemExit from argparse on invalid input.
    """
    parser = argparse.ArgumentParser(
        description="阶段 01：PolyBench 标签调和 / stage 01: harmonise PolyBench labels"
    )
    parser.add_argument("--task", action="append", metavar="TASK_ID",
                        help="只跑指定任务，可重复 / run only these task ids (repeatable)")
    parser.add_argument("--limit", type=int, metavar="N",
                        help="每任务只取前 N 行（冒烟；产物写 *.partial.*，且不写决策表）/ first N rows only")
    parser.add_argument("--dry-run", action="store_true",
                        help="只算不写 / compute without writing")
    parser.add_argument("--verify", action="store_true",
                        help="只读：把已有的清单/决策表/parquet 核一遍 / read-only checks")
    parser.add_argument("--spotcheck", type=int, metavar="N", nargs="?", const=2, default=None,
                        help="每任务抽 N 行回查原始 CSV（默认 2，10 任务共 20 行）/ spot-check rows per task")
    parser.add_argument("-v", "--verbose", action="store_true", help="更详细的日志 / debug logging")
    args = parser.parse_args(argv)
    if args.limit is not None and args.limit <= 0:
        parser.error("--limit 必须是正整数 / --limit must be a positive integer")
    if args.limit is not None and args.verify:
        parser.error("--verify 与 --limit 不能同时用 / --verify and --limit are mutually exclusive")
    if args.spotcheck is not None and args.spotcheck <= 0:
        parser.error("--spotcheck 必须是正整数 / --spotcheck must be a positive integer")
    return args


def main(argv: list[str] | None = None) -> int:
    """跑一次阶段 01，返回进程退出码。
    Run stage 01 once and return the process exit code.

    中文：
        用途：脚本的唯一执行函数；`--verify` 只核不写，其余路径都会在跑完后自动核一遍，
              这样"跑完即验"而不是"跑完等别人发现问题"。
        输入：argv —— 命令行参数。
        输出：0 成功；1 有验收问题或运行失败。
        数据来源：data/raw/polybench/*.csv 与 data/manifests/ 下的清单。
        配置来源：命令行参数。
        副作用：写 parquet、unit_decisions.csv、harmonised_manifest.csv（`--dry-run`/`--verify` 除外）。
        异常：不向外抛——捕获后转成日志与退出码，方便 CI 判断。

    English:
        Purpose: the single execution path; `--verify` is read-only and every other path runs the
        checks afterwards, so a run is verified at the moment it finishes.
        Args: argv - command line. Returns: 0 on success, 1 on problems or failure.
        Data source: the raw CSVs and the manifests. Config: the command line.
        Side effects: writes parquet, unit_decisions.csv and harmonised_manifest.csv, except in
        dry-run and verify modes.
        Raises: nothing escapes; failures become logs plus an exit code.
    """
    args = _parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)-7s %(message)s",
        datefmt="%H:%M:%S",
    )

    if args.verify:
        problems = verify_outputs()
        if problems:
            for problem in problems:
                LOGGER.error("✗ %s", problem)
            LOGGER.error("验收未通过：%d 条问题 / %d problems", len(problems), len(problems))
            return 1
        LOGGER.info("验收通过：%d 个任务、%d 条决策全部满足 A-3 的四条标准", len(TASK_SPECS), _count_decisions())
        return 0

    if args.spotcheck is not None:
        # 抽检只读已有产物：先跑一次全量 01，再抽 20 行回查原始 CSV。
        # The spot check only reads existing artefacts: run stage 01 first, then draw 20 rows.
        try:
            rows, problems = build_spotcheck(per_task=args.spotcheck)
        except FileNotFoundError as exc:
            LOGGER.error("抽检失败 / spot check failed: %s", exc)
            return 1
        target = MANIFEST_DIR / "harmonise_spotcheck.csv"
        write_spotcheck(rows, target)
        LOGGER.info("抽检 %d 行（每任务 %d 行）-> %s", len(rows), args.spotcheck, target)
        for problem in problems:
            LOGGER.error("✗ %s", problem)
        if problems:
            LOGGER.error("抽检未通过：%d 行对不上 / %d mismatches", len(problems), len(problems))
            return 1
        LOGGER.info("抽检全对：原始值、调和值、按规则重算值三列逐行相同")
        return 0

    try:
        outcomes = harmonise_all(task_ids=args.task, limit=args.limit, dry_run=args.dry_run)
    except (FileNotFoundError, KeyError, ValueError) as exc:
        LOGGER.error("调和失败 / harmonisation failed: %s", exc)
        return 1

    _log_summary(outcomes)
    if args.dry_run or args.limit is not None:
        # 这两种模式本来就不追求产出完整（空跑 / 限量），因此不做全量验收。
        # Neither mode aims at complete output, so the full check does not apply.
        return 0

    problems = verify_outputs()
    if problems:
        for problem in problems:
            LOGGER.error("✗ %s", problem)
        return 1
    LOGGER.info(
        "全部完成：%d 个任务调和入库；另有 %d 个任务按红线整列弃用（见 unit_decisions.csv 的 exclude_task）",
        len(outcomes),
        len(EXCLUDED_TASKS),
    )
    return 0


def _count_decisions() -> int:
    """数一数决策表有几行，只用于日志。
    Count the rows in the decision table; logging only.

    中文：
        用途：给验收通过的日志一个具体数字，方便和 git diff 对照。
        输入：无（读默认路径的决策表）。
        输出：int 行数；文件不存在时返回 0。
        数据来源：data/manifests/unit_decisions.csv。
        配置来源：无。副作用：无。异常：无。

    English:
        Purpose: give the pass message a concrete number that can be compared with a git diff.
        Args: none. Returns: row count, 0 when the file is missing.
        Data source: data/manifests/unit_decisions.csv.
        Config: none. Side effects: none. Raises: none.
    """
    from chainworth.io.harmonise import DECISIONS_PATH  # 局部导入：只在日志路径需要

    if not DECISIONS_PATH.exists():
        return 0
    with DECISIONS_PATH.open("r", encoding="utf-8", newline="") as handle:
        return max(sum(1 for _ in handle) - 1, 0)


def _log_summary(outcomes) -> None:
    """把每个任务的行数与单位打成人能读的一张表。
    Log a human-readable table of per-task row counts and units.

    中文：
        用途：跑完之后一眼能看出"哪个任务少了多少行、单位是什么"。
        输入：outcomes —— harmonise_all 的返回（TaskOutcome 序列）。
        输出：无（只写日志）。
        数据来源：outcome 内的计数。
        配置来源：无。副作用：无。异常：无。

    English:
        Purpose: after a run, see at a glance how many rows each task lost and in which unit.
        Args: outcomes - the TaskOutcome list. Returns: none (log only).
        Data source: the counters inside each outcome.
        Config: none. Side effects: none. Raises: none.
    """
    LOGGER.info("%-20s %7s %7s %6s  %-26s %s", "task_id", "rows_in", "rows_out", "dupes", "unit", "range")
    for outcome in outcomes:
        stats = outcome.label_stats
        LOGGER.info(
            "%-20s %7d %7d %6d  %-26s %.4g ~ %.4g",
            outcome.spec.task_id,
            outcome.rows_in,
            outcome.rows_out,
            outcome.n_duplicates,
            outcome.spec.target_unit,
            stats["min"],
            stats["max"],
        )


if __name__ == "__main__":
    raise SystemExit(main())
