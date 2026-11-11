"""阶段入口 03：把三个基准的官方划分做成只读快照，并证明没人重算过。
Stage entry 03: snapshot the three benchmarks' official splits and prove nobody re-cut them.

中文：
    用途：A-5 的唯一命令行入口。四件事，顺序不能换：
          ① `--source canonical`：按官方脚本物化 PolyBench26 的 canonical PSMILES 表，
             并用官方 `canonical_sha256` 验收（不过就停）；
          ② `--source polybench`：从平台接口取回每个 PolyBench 任务的官方测试集；
          ③ `--source polybench`（默认接着做）：把测试集对回调和表行号，
             写 `splits/polybench/<task>/official_70_30.json`；
          ④ `--source polybench26`：把官方 `seed_42/` 原样拷进 `splits/`，并跑官方
             `manage_scaling_indices.py verify` 自检。
          `--source openpoly` 只落一份"官方未发布划分"的结论（D-22）。
    依赖：data/raw/**（00）、data/interim/harmonised/**（01）、configs/tasks/*.yaml（A-4）；
          全部逻辑在 `src/chainworth/protocol/splits.py`，本文件只做参数与日志。
    配置：`configs/tasks/*.yaml` 的 `split_ref` / `split_protocol`，即本阶段的落盘清单。
    对外：命令行。`--verify` 只读，把 A-5 的八条验收标准跑成检查；`--deep` 连行号一起重算。
    被谁用：人工（阶段运行）、CI（`--verify` 当回归门）、D（入口审计时注入错误看它变不变红）。

English:
    Purpose: the only command-line entry for A-5. Four steps, in this order:
    (1) `--source canonical` materialises PolyBench26's canonical PSMILES tables with the
    official script and validates them against the official `canonical_sha256` (stop on
    mismatch); (2) `--source polybench` fetches each task's official test set from the platform;
    (3) the same run maps those test sets onto harmonised row indices and writes
    `splits/polybench/<task>/official_70_30.json`; (4) `--source polybench26` copies the official
    `seed_42/` directories verbatim and runs the official `manage_scaling_indices.py verify`.
    `--source openpoly` only records that the official split was never published (D-22).
    Depends on: data/raw (stage 00), data/interim/harmonised (stage 01), configs/tasks (A-4);
    all logic lives in `src/chainworth/protocol/splits.py`; this file only parses and logs.
    Config: `split_ref` / `split_protocol` of configs/tasks, which *is* the to-do list.
    Public API: the command line; `--verify` is read-only and runs A-5's eight criteria as
    checks; `--deep` also recomputes the row indices.
    Callers: humans running stage 03, CI using `--verify` as a gate, and D's entry audit, which
    injects errors and expects this to go red.
"""

from __future__ import annotations

import argparse
import logging
import subprocess
import sys
from pathlib import Path

# 让 `python scripts/03_splits.py` 直接可用：把仓库根的 src/ 放进 sys.path。
# 不做 pip install -e（冻结的环境里不该出现本仓库的 egg-info）。
# Make the script runnable without installing the package: put src/ on sys.path.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from chainworth.io.tasks import load_all_task_configs  # noqa: E402
from chainworth.protocol import splits as S  # noqa: E402

LOGGER = logging.getLogger("chainworth.splits.cli")


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """解析命令行参数，并把"哪些组合不允许"挡在运行之前。
    Parse arguments and reject invalid combinations before any work starts.

    中文：
        用途：固定脚本接口。`--verify` 与写盘互斥（只读就是只读），`--deep` 只能跟 `--verify`
              一起用，否则"深校验"会变成一个没人知道何时生效的隐藏开关。
        输入：argv —— 命令行参数（None 读 sys.argv）。
        输出：argparse.Namespace，含 source / dry_run / limit / force / verify / deep / verbose。
        数据来源：命令行。配置来源：无。副作用：无。
        异常：SystemExit —— 参数不合法。

    English:
        Purpose: freeze the CLI. `--verify` is mutually exclusive with writing (read-only means
                 read-only) and `--deep` is only meaningful together with `--verify`, otherwise it
                 becomes a hidden switch nobody knows when to use.
        Args: argv - arguments (None reads sys.argv).
        Returns: Namespace with source / dry_run / limit / force / verify / deep / verbose.
        Data source: the command line. Config: none. Side effects: none.
        Raises: SystemExit on invalid input.
    """
    parser = argparse.ArgumentParser(
        description="阶段 03：官方划分快照（A-5）/ stage 03: official split snapshots",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "典型用法 / typical use:\n"
            "  python scripts/03_splits.py --source canonical      # 先物化 canonical 表\n"
            "  python scripts/03_splits.py --source polybench      # 取测试集 + 落 10 份划分\n"
            "  python scripts/03_splits.py --source polybench26    # 快照索引 + 官方自检\n"
            "  python scripts/03_splits.py --verify --deep         # 全量复核（只读）\n"
        ),
    )
    parser.add_argument(
        "--source",
        choices=("all", "canonical", "polybench", "polybench26", "openpoly"),
        default="all",
        help="做哪一步；默认 all（canonical + polybench + polybench26 + openpoly）/ which step",
    )
    parser.add_argument("--limit", type=int, default=None, help="最多处理几个任务，调试用 / max tasks")
    parser.add_argument("--force", action="store_true", help="已存在也重取（会刷新时间戳）/ refetch anyway")
    parser.add_argument("--verify", action="store_true", help="只校验，不写盘 / verify only, read-only")
    parser.add_argument("--deep", action="store_true", help="校验时连行号一起重算（慢）/ recompute indices too")
    parser.add_argument("--dry-run", dest="dry_run", action="store_true", help="只打印计划 / plan only")
    parser.add_argument("--verbose", action="store_true", help="打 DEBUG 日志 / debug logging")
    args = parser.parse_args(argv)

    if args.verify and (args.dry_run or args.force):
        parser.error("--verify 是只读的，不能与 --dry-run / --force 同时用")
    if args.deep and not args.verify:
        parser.error("--deep 只对 --verify 有意义")
    return args


def _run_official_verify(indices_root: Path) -> list[str]:
    """跑官方 `manage_scaling_indices.py verify`，返回它报的问题。
    Run the official `manage_scaling_indices.py verify` and return the problems it reports.

    中文：
        用途：官方自检与我们自己的断言**不是**同一套（它查 manifest 与折文件是否配套，
              不查 `train_pool` 是否是补集）。两边都跑，才算"官方认可 + 我们知道它没查什么"。
              缺 canonical 源时加 `--allow-missing-sources`：官方索引本来就不保证 canonical
              表在本机，我们自己的 `--deep` 才是查那条的。
        输入：indices_root —— `splits/polybench26`。
        输出：问题字符串列表（空 = 通过）。
        数据来源：官方脚本 `data/raw/polybench26/Datasets/Dataset_construction_scripts/`。
        配置来源：无。副作用：只读（官方工具不写盘）。
        异常：无（失败以问题列表返回，方便一起打印）。

    English:
        Purpose: the official self-check and our own assertions are **not** the same set (it
                 validates manifest/fold agreement, not that `train_pool` is the complement).
                 Running both is what "officially accepted plus we know its blind spots" means.
                 `--allow-missing-sources` is used because the official indices never promised
                 the canonical tables are present locally - our `--deep` covers that.
        Args: indices_root - splits/polybench26.
        Returns: problem strings (empty means pass).
        Data source: the official script under data/raw/polybench26.
        Config: none. Side effects: read-only. Raises: none (failures come back as strings).
    """
    script = S.PB26_SCRIPTS / "manage_scaling_indices.py"
    if not script.is_file():
        return [f"官方自检脚本不在：{script}"]
    completed = subprocess.run(
        [
            sys.executable,
            str(script),
            "verify",
            "--indices-root",
            str(indices_root),
            "--allow-missing-sources",
        ],
        cwd=str(S.REPO_ROOT),
        capture_output=True,
        text=True,
    )
    output = (completed.stdout or "") + (completed.stderr or "")
    LOGGER.info("官方自检输出 / official verify output:\n%s", output.strip()[-1500:])
    if completed.returncode != 0:
        return [f"官方 manage_scaling_indices.py verify 退出码 {completed.returncode}"]
    return []


def _verify(args: argparse.Namespace) -> int:
    """跑全量校验并按结果给出退出码。
    Run the full verification and map the result to an exit code.

    中文：
        用途：`--verify` 的实现。问题一条条打印，最后一行给结论；退出码 0 = 全过。
        输入：args —— 含 deep。
        输出：0 或 1。
        数据来源：`splits/**`、`configs/tasks/**`、`data/manifests/splits_manifest.csv`。
        配置来源：命令行 `--deep`。副作用：只读。异常：无。

    English:
        Purpose: the body of `--verify`: print every problem, then a verdict; exit 0 means pass.
        Args: args - carries deep. Returns: 0 or 1.
        Data source: splits/, configs/tasks/, splits_manifest.csv.
        Config: the `--deep` flag. Side effects: read-only. Raises: none.
    """
    configs = load_all_task_configs(S.CONFIGS_DIR)
    problems = S.verify_splits(configs, deep=args.deep)
    problems += _run_official_verify(S.SPLITS_ROOT / "polybench26")
    if problems:
        LOGGER.error("校验未通过 / verification FAILED（%d 条）：", len(problems))
        for problem in problems:
            LOGGER.error("  - %s", problem)
        return 1
    LOGGER.info("校验通过 / verification passed（deep=%s）", args.deep)
    return 0


def _run_canonical(args: argparse.Namespace, configs) -> None:
    """物化 canonical 表并逐项核对官方 `canonical_sha256`。
    Materialise the canonical tables and check each against the official digest.

    中文：
        用途：A-5 的第 ① 步。生成之后**必须**核对——官方索引存的是行号，行号错位不报错，
              只会算出一组好看的 R²，所以这里宁可停下。
        输入：args —— 含 dry_run / limit；configs —— 全部任务配置。
        输出：无。
        数据来源：`data/raw/polybench26/Datasets/Dataset_construction_scripts/`。
        配置来源：`configs/tasks/pb26_*.yaml` 的 `datasets[].official_indices_path`。
        副作用：写 `data/raw/polybench26/Datasets/{PSMILES,wPSMILES}/**`。
        异常：SplitError —— 哈希不对（由调用方转成退出码）。

    English:
        Purpose: step 1 of A-5. Generation **must** be followed by the digest check - the official
                 indices are row numbers, and a shifted row does not raise, it just produces a
                 nice-looking R². So this stops instead.
        Args: args - carries dry_run / limit; configs - all task configs.
        Returns: none. Data source: the official construction scripts under data/raw.
        Config: `datasets[].official_indices_path`. Side effects: writes under data/raw.
        Raises: SplitError on a digest mismatch (turned into an exit code by the caller).
    """
    if args.limit is not None:
        LOGGER.warning("--limit 对 canonical 物化无意义（官方脚本一次生成全部），已忽略")
    expected = S._canonical_sha256_map(configs)
    # 跳过条件在模块里（`materialize_canonical_tables()`）：给 expected 且不 force，全对就跳过。
    # The skip rule lives in the module: pass `expected`, and without `--force` an all-valid set
    # is left untouched.
    produced = S.materialize_canonical_tables(
        dry_run=args.dry_run, force=args.force, expected=expected
    )
    if args.dry_run:
        return
    for source, digest in sorted(expected.items()):
        actual = produced.get(source)
        if actual is None:
            raise S.SplitError(f"官方要求 {source}，但没有生成出这一份")
        if actual != digest:
            raise S.SplitError(f"{source} 的 sha256 与官方 canonical_sha256 不等：{actual} != {digest}")
        LOGGER.info("  canonical 通过 / matched：%s", source)


def main(argv: list[str] | None = None) -> int:
    """命令行主流程：按 `--source` 依次执行，任一失败即以非零退出码停下。
    The CLI flow: run stages in `--source` order and stop with a non-zero code on any failure.

    中文：
        用途：把"取数 / 落盘 / 校验"三件事按固定顺序串起来，日志里每一步都能对上分工书的验收项。
        输入：argv —— 命令行参数。
        输出：退出码：0 成功；1 校验不过；2 运行失败。
        数据来源：见各子函数。配置来源：命令行。
        副作用：按 `--source` 写 `data/raw/**` 与 `splits/**`。
        异常：无（内部捕获并转成退出码）。

    English:
        Purpose: chain fetch / write / verify in a fixed order so every log line maps onto an
                 acceptance item of the assignment sheet.
        Args: argv. Returns: 0 ok, 1 verification failed, 2 runtime failure.
        Data source: see the helpers. Config: the command line.
        Side effects: writes data/raw and splits per `--source`.
        Raises: none (converted into exit codes).
    """
    args = _parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)-7s %(message)s",
        datefmt="%H:%M:%S",
    )

    configs = load_all_task_configs(S.CONFIGS_DIR)
    if args.verify:
        return _verify(args)

    wanted = ("canonical", "polybench", "polybench26", "openpoly") if args.source == "all" else (args.source,)
    LOGGER.info("=" * 62)
    LOGGER.info("本次步骤 / requested steps：%s", list(wanted))
    try:
        if "canonical" in wanted:
            _run_canonical(args, configs)
        if "polybench" in wanted:
            S.fetch_polybench_test_sets(configs, limit=args.limit, force=args.force, dry_run=args.dry_run)
            S.build_polybench_splits(configs, limit=args.limit, dry_run=args.dry_run)
        if "polybench26" in wanted:
            S.snapshot_polybench26(configs, limit=args.limit, dry_run=args.dry_run)
            if not args.dry_run and args.limit is None:
                official = _run_official_verify(S.SPLITS_ROOT / "polybench26")
                if official:
                    raise S.SplitError("；".join(official))
        if "openpoly" in wanted:
            S.write_openpoly_blocked_note(dry_run=args.dry_run)
    except (S.SplitError, FileNotFoundError) as exc:
        LOGGER.error("失败 / failed：%s", exc)
        return 2

    LOGGER.info("=" * 62)
    LOGGER.info("完成：步骤 %s / done", list(wanted))
    if not args.dry_run:
        LOGGER.info("下一步：`python scripts/03_splits.py --verify --deep` 复核，再把 18 份 YAML 的 split_status 改成 present")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
