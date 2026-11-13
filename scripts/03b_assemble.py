"""阶段入口 03b：任务级特征装配与图构建（B-1）。
Stage entry 03b: task-level feature assembly and graph construction (B-1).

中文：
    用途：B-1 的唯一入口。跑完得到三样东西：
          data/processed/features/<task>.npz              每次**测量**一行：row_id + 标签 + 结构串
                                                          + 205 描述符（float64）+ 2048 位指纹（uint8）
          data/manifests/task_features_manifest.csv       每任务一行：行数、列数、NaN、无结构行数、
                                                          划分版本、文件哈希
          data/manifests/task_feature_unresolved.csv      拿不到特征的行逐条列 ID（没有则不存在）
          加 `--graphs` 时多一份 data/manifests/graph_manifest.csv（图的数量、维度、对拍结果）。
    依赖：`data/interim/harmonised/`（01）、`configs/tasks/`（A-4）、`splits/`（03，只读）、
          `data/processed/{structures,descriptors}`（02）。逻辑全在
          `chainworth/{io/loaders,chem/features,chem/graph}.py`，本文件只做参数与日志。
    配置：无 yaml；任务清单来自 `configs/tasks/*.yaml`（经 `loaders`）。
    对外：命令行。`--verify` 只读，把 B-1 的五条验收标准跑成检查。
    被谁用：人工（阶段运行）、C（拿 npz 跑 A2/A5）、B-3（拿图与 npz 跑 A3）、CI（`--verify` 回归）。

English:
    Purpose: B-1's only entry point. A run produces three things: one `.npz` per task (one row per
    **measurement**: row_id, label, structure string, 205 descriptors as float64 and 2048
    fingerprint bits as uint8), a per-task manifest (row counts, column counts, NaN counts, rows
    without a structure, split version, file hash) and an unresolved-rows list (absent when
    empty). With `--graphs` it also writes a graph manifest (counts, dimensions, parity result).
    Depends on: data/interim/harmonised (01), configs/tasks (A-4), splits (03, read-only) and
    data/processed/{structures,descriptors} (02). All logic lives in
    chainworth/{io/loaders,chem/features,chem/graph}; this file parses arguments and logs.
    Config: no yaml; the task list comes from configs/tasks/*.yaml through `loaders`.
    Public API: the command line; `--verify` is read-only and turns B-1's five acceptance criteria
    into checks.
    Callers: humans running stage 03b, C (A2/A5 arms read the npz), B-3 (A3 reads the graphs and
    the npz) and CI (`--verify` as a regression gate).

给队友的一句话 ／ One line for teammates
-----------------------------------------
`data/processed/features/*.npz` **不在仓库里**（`data/processed/*` 被 gitignore）。拿到提交后跑

    python scripts/03b_assemble.py

就得到同一批文件（约 1 分钟）；`--verify` 用清单哈希 + 抽 100 个结构与 02 的产物逐位比对，
证明"你重出来的和我的一样"。行序**不是**随便排的：第 k 行就是 `row_id=k`，与
`splits/polybench/*/official_70_30.json` 的 `test_index` 同一个坐标系——这是 C 与 D 都依赖的那条。

**例外（D-38，2026-11-13）**：C 组要用的 **11 份**（8 个主表任务 + 3 个 OpenPoly）已随仓库交付，
见 `data/manifests/committed_data_manifest.csv`；备用队那 4 份仍靠本脚本重建
（`--tasks pb_t1_td`）。上面那段话对那 4 份依然成立。

The `.npz` files are **not** in the repository. Run `python scripts/03b_assemble.py` to rebuild
them (about a minute); `--verify` proves the rebuild matches by manifest hashes plus a 100-structure
bit-for-bit comparison against stage 02's output. Row order is not arbitrary: row k is `row_id=k`,
the same coordinate system as `test_index` in the PolyBench split JSONs - the invariant C and D
both rely on.
"""

from __future__ import annotations

import argparse
import csv
import logging
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from chainworth.chem import features as F  # noqa: E402
from chainworth.io import loaders as L  # noqa: E402

LOGGER = logging.getLogger("chainworth.assemble.cli")

GRAPH_MANIFEST = L.MANIFEST_DIR / "graph_manifest.csv"
GRAPH_MANIFEST_COLUMNS = (
    "task_id",
    "n_unique_structures",
    "n_graphs",
    "n_parse_failed",
    "n_without_wpsmiles",
    "atom_fdim",
    "bond_fdim",
    "graph_attr_dim",
    "mean_atoms",
    "mean_bonds",
    "determinism_check",
    "chemprop_parity",
    "built_at",
)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """解析参数，并挡住相互矛盾的组合。
    Parse arguments and reject contradictory combinations before any work starts.

    中文：
        用途：固定脚本接口。`--verify` 与 `--limit` 互斥：校验只读、限量跑写 `*.partial.*`，
              混在一起会让人以为校验能顺带修东西（02 用的是同一个约定）。
        输入：argv —— 命令行参数（None 读 sys.argv）。
        输出：Namespace（tasks / limit / verify / graphs / dry_run / verbose）。
        数据来源：命令行。配置来源：无。副作用：无。异常：SystemExit —— 参数不合法。

    English:
        Purpose: freeze the CLI. `--verify` and `--limit` are mutually exclusive: verification is
        read-only while a limited run writes `*.partial.*`, and mixing them invites the belief that
        verify can fix things (stage 02 uses the same convention).
        Args: argv. Returns: a Namespace. Data source: the command line.
        Side effects: none. Raises: SystemExit from argparse.
    """
    parser = argparse.ArgumentParser(
        description="阶段 03b：任务级特征装配与图构建 / stage 03b: task assembly and graphs"
    )
    parser.add_argument("--tasks", action="append", metavar="TASK_ID",
                        help="只跑指定任务，可重复 / only these task ids (repeatable)")
    parser.add_argument("--limit", type=int, metavar="N",
                        help="每任务只取前 N 行（冒烟；产物写 *.partial.*，不写正式清单）")
    parser.add_argument("--verify", action="store_true",
                        help="只读：把 B-1 的验收标准跑一遍 / read-only checks")
    parser.add_argument("--graphs", action="store_true",
                        help="额外建图并写 graph_manifest.csv（GNN 环境里才会与 chemprop 对拍）")
    parser.add_argument("--graph-sample", type=int, default=200, metavar="N",
                        help="与 chemprop 对拍抽多少个结构（默认 200）")
    parser.add_argument("--dry-run", action="store_true",
                        help="只算不写 / compute without writing")
    parser.add_argument("-v", "--verbose", action="store_true", help="更详细的日志 / debug logging")
    args = parser.parse_args(argv)

    if args.limit is not None and args.limit <= 0:
        parser.error("--limit 必须是正整数 / --limit must be a positive integer")
    if args.verify and (args.limit is not None or args.dry_run):
        parser.error("--verify 只读，不能与 --limit/--dry-run 同用")
    if args.graph_sample <= 0:
        parser.error("--graph-sample 必须是正整数 / must be positive")
    return args


def _graph_row(task_id: str, sample: int) -> dict[str, Any]:
    """给一个任务建图并写一行图清单（含确定性自查与 chemprop 对拍）。
    Build one task's graphs and produce one manifest row (determinism self-check plus chemprop
    parity).

    中文：
        用途：图不在盘上留副本（11,494 个图约 100 MB，重算只要十几秒），留的是**清单**——
              数量、维度、失败数、以及"两次构图是否逐位相同"这一条自查。验收③要的就是这一条，
              把它写进清单比写进日志更难被忽略。
        输入：task_id —— 任务；sample —— 与 chemprop 对拍抽多少个结构。
        输出：dict（键见 `GRAPH_MANIFEST_COLUMNS`）。
        数据来源：`data/processed/features/<task>.npz` 与结构表。
        配置来源：无。副作用：只读（图只存在内存里）。异常：AssemblyError —— 缺 npz。

    English:
        Purpose: graphs are not kept on disk (11,494 graphs is ~100 MB and a dozen seconds to
        rebuild); what is kept is the **manifest** - counts, dimensions, failures and the
        "two builds are bit-identical" self-check. Criterion 3 is exactly that, and a manifest is
        harder to ignore than a log line.
        Args: task_id; sample. Returns: a dict keyed by GRAPH_MANIFEST_COLUMNS.
        Data source: data/processed/features/<task>.npz plus the structure table.
        Side effects: read-only (graphs stay in memory). Raises: AssemblyError without the npz.
    """
    import numpy as np

    from chainworth.chem import graph as G

    pack = F.load_pack(task_id)
    structures = L.load_structure_table_index()
    keys = list(dict.fromkeys(key for key in pack["psmiles"] if key))
    wpsmiles = {
        key: str(structures.at[key, "wpsmiles"])
        for key in keys
        if key in structures.index and isinstance(structures.at[key, "wpsmiles"], str)
    }
    graphs, failures = G.build_graphs(keys, wpsmiles)
    without_wpsmiles = sum(1 for value in graphs.values() if value.wpsmiles is None)

    # 确定性自查：同一结构建两次（第二次走的是另一条调用路径：从图里的规范串再建一次），
    # 两次的 edge_index 必须逐位相同。抽 20 个够用——真错的话第一个就会错。
    # Determinism: build the same structure twice (the second time from the graph's own canonical
    # string, a different call path) and require identical edge_index. Twenty is plenty; a real bug
    # shows up on the first one.
    rng = np.random.default_rng(0)
    sample_keys = [keys[index] for index in rng.choice(len(keys), size=min(20, len(keys)), replace=False)]
    mismatched = [
        key
        for key in sample_keys
        if key in graphs
        and G.build_graph(graphs[key].psmiles) is not None
        and not np.array_equal(G.build_graph(graphs[key].psmiles).edge_index, graphs[key].edge_index)
    ]
    determinism = "ok（抽 20 个，两次构图 edge_index 逐位相同）" if not mismatched else (
        f"FAIL（{len(mismatched)} 个结构的两次构图不同，例：{mismatched[0][:32]}）"
    )

    parity_keys = [graphs[key].psmiles for key in sample_keys if key in graphs]
    try:
        problems = G.verify_against_chemprop(parity_keys[:sample])
        parity = (
            f"ok（{len(parity_keys[:sample])} 个结构，原子/键特征与 edge_index 逐位相同）"
            if not problems
            else f"FAIL（{len(problems)} 条问题，例：{problems[0]}）"
        )
    except ImportError:
        parity = "skipped（当前环境没有 chemprop：图臂在 GNN 环境里跑，对拍在那里做）"

    atoms = np.asarray([value.n_atoms for value in graphs.values()], dtype=float)
    bonds = np.asarray([value.n_bonds for value in graphs.values()], dtype=float)
    LOGGER.info(
        "%s：%d 个结构 → %d 个图（解析失败 %d；无 wPSMILES %d）；平均 %d 原子 / %d 键",
        task_id,
        len(keys),
        len(graphs),
        len(failures),
        without_wpsmiles,
        int(atoms.mean()) if atoms.size else 0,
        int(bonds.mean()) if bonds.size else 0,
    )
    return {
        "task_id": task_id,
        "n_unique_structures": len(keys),
        "n_graphs": len(graphs),
        "n_parse_failed": len(failures),
        "n_without_wpsmiles": without_wpsmiles,
        "atom_fdim": G.ATOM_FEATURE_DIM,
        "bond_fdim": G.BOND_FEATURE_DIM,
        "graph_attr_dim": G.GRAPH_ATTR_DIM,
        "mean_atoms": f"{atoms.mean():.1f}" if atoms.size else "",
        "mean_bonds": f"{bonds.mean():.1f}" if bonds.size else "",
        "determinism_check": determinism,
        "chemprop_parity": parity,
        "built_at": F.now_iso(),
    }


def _write_graph_manifest(rows: list[dict[str, Any]]) -> Path:
    """写图清单（原子写、LF 行尾）。/ Write the graph manifest atomically with LF endings.

    中文：用途：证据落盘。输入：rows。输出：路径。数据来源：调用方。
          配置来源：无。副作用：建目录、写一个文件。异常：OSError。
    English: Purpose: put the evidence on disk. Args: rows. Returns: the path.
             Side effects: creates the directory and writes one file. Raises: OSError.
    """
    import os

    path = GRAPH_MANIFEST
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(GRAPH_MANIFEST_COLUMNS), lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column, "") for column in GRAPH_MANIFEST_COLUMNS})
    os.replace(temporary, path)
    return path


def main(argv: list[str] | None = None) -> int:
    """跑一次阶段 03b，返回进程退出码。
    Run stage 03b once and return the process exit code.

    中文：
        用途：脚本的唯一执行函数。全量跑完**自动核一遍**——"跑完即验"，而不是等人来问。
        输入：argv。输出：0 成功；1 验收有问题；2 运行失败（缺输入等）。
        数据来源：见模块 docstring。配置来源：命令行。
        副作用：写 npz 与两份清单（dry-run / verify 除外）。异常：不向外抛。

    English:
        Purpose: the single execution path. A full run **verifies itself** at the end, so problems
        surface now instead of when somebody asks.
        Args: argv. Returns: 0 on success, 1 when a check fails, 2 on a run failure.
        Data source: see the module docstring. Config: the command line.
        Side effects: writes npz files and two manifests except in dry-run and verify modes.
        Raises: nothing escapes.
    """
    args = _parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)-7s %(message)s",
        datefmt="%H:%M:%S",
    )

    if args.verify:
        problems = F.verify_features(args.tasks)
        for problem in problems:
            LOGGER.error("✗ %s", problem)
        if problems:
            LOGGER.error("验收未通过：%d 条问题", len(problems))
            return 1
        LOGGER.info("验收通过：五条标准全过（抽查 100 个结构与 02 的产物逐位相同）")
        return 0

    tasks = args.tasks or L.available_tasks()
    if not tasks:
        LOGGER.error("没有可装配的任务：data/interim/harmonised 下为空，先跑 01_harmonise.py")
        return 2
    partial = args.limit is not None
    previous = {row.get("task_id", ""): row for row in F.read_manifest_rows(F.MANIFEST_PATH)}

    manifest_rows: list[dict[str, Any]] = []
    unresolved_rows: list[dict[str, Any]] = []
    graph_rows: list[dict[str, Any]] = []
    try:
        for task_id in tasks:
            summary = F.assemble_task(task_id, limit=args.limit)
            unresolved_rows.extend(summary["unresolved"])
            if args.dry_run:
                LOGGER.info("%s：[dry-run] 不写盘", task_id)
                if args.graphs:
                    graph_rows.append(_graph_row(task_id, args.graph_sample))
                continue
            pack_path = F.save_pack(summary, partial=partial)
            if partial:
                LOGGER.info("%s：写 %s（限量跑，清单走 *.partial.csv，不入库）", task_id, pack_path.name)
                continue
            old = previous.get(task_id, {})
            same = old.get("pack_sha256") == F.sha256_of(pack_path)
            manifest_rows.append(
                F.manifest_row(summary, pack_path, built_at=str(old.get("built_at", "")) if same else "")
            )
            if args.graphs:
                graph_rows.append(_graph_row(task_id, args.graph_sample))
    except (L.AssemblyError, FileNotFoundError) as error:
        LOGGER.error("缺输入 / missing input: %s", error)
        return 2
    except (KeyError, ValueError) as error:
        LOGGER.error("运行失败 / run failed: %s", error)
        return 2

    if args.dry_run or partial:
        return 0

    manifest_path, unresolved_path = F.write_manifests(manifest_rows, unresolved_rows)
    LOGGER.info("清单已写：%s（%d 个任务）", manifest_path.relative_to(REPO_ROOT).as_posix(), len(manifest_rows))
    if unresolved_path is not None:
        LOGGER.warning("有 %d 行拿不到结构级特征：%s", len(unresolved_rows), unresolved_path.name)
    if args.graphs:
        path = _write_graph_manifest(graph_rows)
        LOGGER.info("图清单已写：%s（%d 个任务）", path.relative_to(REPO_ROOT).as_posix(), len(graph_rows))

    problems = F.verify_features(tasks)
    for problem in problems:
        LOGGER.error("✗ %s", problem)
    if problems:
        LOGGER.error("跑完自检未过：%d 条问题", len(problems))
        return 1
    LOGGER.info("跑完自检通过：%d 个任务", len(tasks))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
