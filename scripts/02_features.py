"""阶段入口 02：把调和表里的结构算成"每结构一次"的数值表征（A-6）。
Stage entry 02: turn the structures in the harmonised tables into per-structure numeric features.

中文：
    用途：A-6 的唯一入口。跑完得到三样东西（全部不入库，靠本脚本重建）：
          data/processed/structures/psmiles.parquet       全局结构表（规范式 + wPSMILES + 画像）
          data/processed/descriptors/<task>.parquet       每任务的结构键 + 205 描述符 + 2048 位指纹
          data/manifests/{structures_manifest.csv, features_manifest.csv,
                          structure_failures.csv, descriptor_columns.json}
    依赖：data/interim/harmonised/*.parquet（阶段 01）；全部逻辑在
          `src/chainworth/chem/{notation,descriptors,structures}.py`，本文件只做参数与日志。
    配置：无 yaml——特征空间定义在 `chem/descriptors.py`，是口径不是实验参数。
    对外：命令行。`--verify` 只读，把 A-6 的四条验收标准跑成检查（`--deep` 会重算）。
    被谁用：人工（阶段运行）、B（重建结构级产物）、CI（`--verify` 当回归）。

English:
    Purpose: A-6's only entry point. A run produces three things, none of them committed, all
    rebuildable by this script: the global structure table, one descriptor parquet per task
    (structure key, 205 descriptors, 2048 fingerprint bits) and four manifests.
    Depends on: data/interim/harmonised/*.parquet (stage 01). All logic lives in
    chainworth.chem.{notation,descriptors,structures}; this file only parses arguments and logs.
    Config: no yaml - the feature space is defined in chem/descriptors.py, and it is a definition
    rather than an experiment parameter.
    Public API: the command line; `--verify` is read-only and turns A-6's acceptance criteria into
    checks (`--deep` recomputes).
    Callers: humans running stage 02, B rebuilding the structure-level artefacts, CI using
    `--verify` as a regression gate.

给队友的一句话 ／ One line for teammates
-----------------------------------------
这两个 parquet **不在仓库里**（`data/processed/*` 被 gitignore）。拿到提交后跑一次

    python scripts/02_features.py --source all

就得到同一批文件；`--verify` 用清单里的 sha256 与重算两条路证明"你重出来的和我的一样"。

**一个例外（D-38，2026-11-13）**：`structures/psmiles.parquet` 已随仓库交付（C-a 的 A5 要读
wPSMILES 与重复单元数），清单见 `data/manifests/committed_data_manifest.csv`；
`descriptors/<task>.parquet`（34 MB）与缓存仍靠本脚本重建。

The parquet files are **not** in the repository (`data/processed/*` is gitignored). After cloning,
run `python scripts/02_features.py --source all` to rebuild the same files; `--verify` proves the
rebuild is identical via the manifest hashes and a full recomputation.

**One exception (D-38)**: `structures/psmiles.parquet` ships with the repository (A5 needs the
wPSMILES and repeat-unit columns); the descriptor tables and the cache are still rebuilt here.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# 让 `python scripts/02_features.py` 直接可用：把仓库根的 src/ 放进 sys.path（理由同 01）。
# Make `python scripts/02_features.py` work without installing the package; same reason as stage 01.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from chainworth.chem import structures as S  # noqa: E402  —— sys.path 之后导入 / import after sys.path

LOGGER = logging.getLogger("chainworth.features.cli")

SOURCES = ("all", "structures", "descriptors")


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """解析参数，并把"哪些组合不允许"挡在运行之前。
    Parse arguments and reject invalid combinations before any work starts.

    中文：
        用途：固定脚本接口。`--verify` 与 `--limit`/`--force` 互斥是刻意的：校验只读，
              而 `--limit`/`--force` 都是"写"，混在一起会让人以为校验能顺带修东西。
        输入：argv —— 命令行参数（None 读 sys.argv）。
        输出：Namespace，含 source / tasks / limit / force / verify / deep / dry_run / verbose。
        数据来源：命令行。配置来源：无。副作用：无。
        异常：SystemExit —— 参数不合法（argparse 抛）。

    English:
        Purpose: freeze the CLI. `--verify` being mutually exclusive with `--limit`/`--force` is
                 deliberate: verification is read-only while both of those write, and mixing them
                 invites the belief that verify can fix things.
        Args: argv. Returns: a Namespace. Data source: the command line.
        Side effects: none. Raises: SystemExit from argparse.
    """
    parser = argparse.ArgumentParser(
        description="阶段 02：结构表征（PSMILES / wPSMILES / 描述符 / Morgan 指纹）/ stage 02: features"
    )
    parser.add_argument("--source", choices=SOURCES, default="all",
                        help="跑哪一段：结构表 / 每任务描述符 / 全部（默认 all）")
    parser.add_argument("--tasks", action="append", metavar="TASK_ID",
                        help="只跑指定任务，可重复 / run only these task ids (repeatable)")
    parser.add_argument("--limit", type=int, metavar="N",
                        help="每任务只取前 N 行（冒烟；产物写 *.partial.*，不写正式清单）")
    parser.add_argument("--force", action="store_true",
                        help="即使产物已存在也重写 / overwrite existing artefacts")
    parser.add_argument("--verify", action="store_true",
                        help="只读：把 A-6 的验收标准跑一遍 / read-only checks")
    parser.add_argument("--deep", action="store_true",
                        help="配合 --verify：重算每个结构的描述符与指纹（约 10 秒）")
    parser.add_argument("--dry-run", action="store_true",
                        help="只算不写 / compute without writing")
    parser.add_argument("-v", "--verbose", action="store_true", help="更详细的日志 / debug logging")
    args = parser.parse_args(argv)

    if args.limit is not None and args.limit <= 0:
        parser.error("--limit 必须是正整数 / --limit must be a positive integer")
    if args.verify and (args.limit is not None or args.force or args.dry_run):
        parser.error("--verify 只读，不能与 --limit/--force/--dry-run 同用")
    if args.deep and not args.verify:
        parser.error("--deep 只能跟 --verify 一起用 / --deep requires --verify")
    return args


def main(argv: list[str] | None = None) -> int:
    """跑一次阶段 02，返回进程退出码。
    Run stage 02 once and return the process exit code.

    中文：
        用途：脚本的唯一执行函数。全量跑完会自动核一遍——"跑完即验"，而不是等 B 发现不对劲。
        输入：argv。输出：0 成功；1 有验收问题；2 运行失败（缺输入、缓存 provenance 不匹配等）。
        数据来源：data/interim/harmonised、data/processed、data/manifests。
        配置来源：命令行。副作用：写 parquet、清单、缓存（dry-run/verify 除外）。
        异常：不向外抛——捕获后转日志与退出码。

    English:
        Purpose: the single execution path. A full run verifies itself at the end, so problems
                 surface now rather than when B notices.
        Args: argv. Returns: 0 on success, 1 on failed checks, 2 on a run failure (missing input,
        cache provenance mismatch, ...).
        Data source: data/interim/harmonised, data/processed, data/manifests.
        Config: the command line. Side effects: writes parquet, manifests and the cache, except in
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
        problems = S.verify_outputs(args.tasks, deep=args.deep)
        for problem in problems:
            LOGGER.error("✗ %s", problem)
        if problems:
            LOGGER.error("验收未通过：%d 条问题 / %d problems", len(problems), len(problems))
            return 1
        LOGGER.info(
            "验收通过：结构表 + 每任务特征与清单一致（deep=%s） / verification passed (deep=%s)",
            args.deep,
            args.deep,
        )
        return 0

    want_structures = args.source in ("all", "structures")
    want_descriptors = args.source in ("all", "descriptors")
    table = None
    try:
        if want_structures:
            LOGGER.info("① 结构表：规范化 + 去重 + wPSMILES")
            summary = S.build_structure_table(args.tasks, limit=args.limit, dry_run=args.dry_run)
            # 空跑与限量跑都不会写正式结构表，下一步直接用手上这一份，别让 smoke 跑只能跑一半。
            # Neither a dry run nor a limited run writes the official structure table, so the next
            # step reuses the frame in hand.
            if args.dry_run or args.limit is not None:
                table = summary["table"]
        if want_descriptors:
            partial_table = S.STRUCTURES_PATH.with_name("psmiles.partial.parquet")
            if table is None and not S.STRUCTURES_PATH.is_file():
                if args.limit is not None and partial_table.is_file():
                    table = S.load_structure_table(partial_table)
                elif args.dry_run:
                    LOGGER.info("盘上没有结构表：先内存里建一份（dry-run 不写盘）")
                    summary = S.build_structure_table(args.tasks, limit=args.limit, dry_run=True)
                    table = summary["table"]
                else:
                    LOGGER.error(
                        "缺结构表 %s：先跑 `python scripts/02_features.py --source structures`",
                        S.STRUCTURES_PATH.relative_to(REPO_ROOT).as_posix(),
                    )
                    return 2
            LOGGER.info("② 描述符与指纹：按结构算一次，再按任务投影")
            S.build_descriptors(
                args.tasks,
                limit=args.limit,
                force=args.force,
                dry_run=args.dry_run,
                table=table,
            )
    except FileNotFoundError as error:
        LOGGER.error("缺输入 / missing input: %s", error)
        return 2
    except (KeyError, ValueError) as error:
        LOGGER.error("运行失败 / run failed: %s", error)
        return 2

    if args.dry_run or args.limit is not None:
        # 这两种模式本来就不追求产出完整（空跑 / 限量），因此不做全量验收。
        # Neither mode aims at complete output, so the full check does not apply.
        return 0

    problems = S.verify_outputs(args.tasks)
    for problem in problems:
        LOGGER.error("✗ %s", problem)
    if problems:
        return 1
    LOGGER.info(
        "全部完成：结构表与每任务特征已落盘并自检通过。重跑命令见文件头；"
        "`--verify --deep` 会从原始结构重算一遍。"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
