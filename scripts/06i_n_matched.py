"""n 匹配实验：把 tier1 的训练集砍到 tier2 的大小，再比一次 Λ（H1 的混淆能不能解开）。
n-matched probe: cut tier-1's training set down to tier-2's size, then compare Lambda again.

中文：
    用途：分工 D-4 / H1。PolyBench 的 tier 与样本量完全反序（t3 228/279 < t2 711/1034 <
          t1 3796/8055），于是"层号"与"数据量"分不开——ρ(tier, log n) = −0.956。本脚本把
          t1 的两个任务下采样到 t2 的两个量级（711 / 1034），在**同一批测试集**上重训，
          得到"同 n 下 tier1 vs tier2"的可比读数。
    **三条纪律**（每条都可能悄悄改结论）：
      ① **只动训练行，测试集逐行不变。** 用 `graph.n_level_rows()` 的 `shuffle_prefix`：
         固定种子打乱官方 train_index 再取前缀。不能取前 n 行——官方 train_index 升序，
         前缀就是 CSV 头几行（D-35）。
      ② **不写 `results/raw_runs/`。** `run_id` 只有 `task__arm__fold__seed`，不含 n 维度；
         同名会**静默覆盖**主表记录（D-34 / D-66）。故一律写 `n_matched.csv`。
      ③ **臂配置一个字节都不改。** 冻结配置里没有 `subset` 段，改动会让 `config_hash` 变，
         538 条 A3 / 328 条 A2 记录当场作废。这里在内存里给 `ArmConfig.raw` 补一个 `subset`，
         并把 `arm_config_sha256`（冻结文件字节哈希）与 `subset_seed` / `subset_sha256`
         **一起**写进 CSV——身份由 `subset_sha256` 承担，与 `kappa.py` 的口径一致。
    输入：任务（默认 `pb_t1_tg` / `pb_t1_tm`）、档位（默认 **497 / 723**）、种子（默认 0/1/2）。
          **档位是训练行数，不是总行数**：tier2 的 `pb_t2_permeability` 训练 497 行、
          `pb_t2_modulus` 训练 723 行（总行数 711 / 1034 是含测试集的，用它匹配会把
          "同 n"对错位）。tier2 两个任务用 `--no-subset` 跑全量，档位标签自然等于它们自己的
          `n_train`，于是配对自动成立。
    输出：`results/aggregate/n_matched.csv`（逐 run 一行，按 (task, arm, n_level, fold, seed) 去重）。
    副作用：训练、写 CSV。异常：缺 pack / 缺划分。
    用法：
        python scripts/06i_n_matched.py --arm a2_conventional                      # t1 下采样到 497/723
        python scripts/06i_n_matched.py --arm a2_conventional --tasks pb_t2_permeability pb_t2_modulus --no-subset
        D:/Miniconda3/envs/DSC6020_Project_GNN/python.exe scripts/06i_n_matched.py --arm a3_graph
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import sys
import time
from dataclasses import replace
from pathlib import Path
from typing import Any, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

OUT_PATH = REPO_ROOT / "results" / "aggregate" / "n_matched.csv"
ARM_CONFIGS = {
    "a1_floor": REPO_ROOT / "configs" / "arms" / "a1_floor.yaml",
    "a2_conventional": REPO_ROOT / "configs" / "arms" / "a2_conventional.yaml",
    "a3_graph": REPO_ROOT / "configs" / "arms" / "a3_graph.yaml",
    "a4_polybert": REPO_ROOT / "configs" / "arms" / "a4_polybert.yaml",
}
BEST_PARAMS = REPO_ROOT / "results" / "aggregate" / "a2_best_params.csv"
SUBSET_SEED = 20261118
DEFAULT_TASKS = ("pb_t1_tg", "pb_t1_tm")
DEFAULT_LEVELS = ("497", "723")
DEFAULT_SEEDS = (0, 1, 2)
COLUMNS = [
    "task",
    "arm",
    "n_level",
    "n_train",
    "n_test",
    "fold",
    "seed",
    "r2",
    "mae",
    "rmse",
    "wall_clock_s",
    "subset_sha256",
    "subset_seed",
    "arm_config_sha256",
    "note",
]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _module(arm: str):
    """按臂名懒加载模块（A3 要 chemprop，只在 GNN 环境里可导入）。
    Import the arm module lazily: A3 needs chemprop and only imports inside the GNN env.
    """
    if arm == "a1_floor":
        from chainworth.arms import floor as module

        return module
    if arm == "a2_conventional":
        from chainworth.arms import conventional as module

        return module
    if arm == "a3_graph":
        from chainworth.arms import graph as module

        return module
    if arm == "a4_polybert":
        from chainworth.arms import polybert as module

        return module
    raise SystemExit(f"不认识的臂 {arm!r}：只认 {sorted(ARM_CONFIGS)}")


def _read_existing(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _write(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in COLUMNS})


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="06i_n_matched.py", description="n 匹配实验：同 n 下比较 tier1 与 tier2（H1 的混淆）"
    )
    parser.add_argument("--arm", required=True, choices=tuple(ARM_CONFIGS), help="跑哪条臂")
    parser.add_argument("--tasks", nargs="+", default=list(DEFAULT_TASKS), metavar="TASK")
    parser.add_argument("--levels", nargs="+", default=list(DEFAULT_LEVELS), metavar="N")
    parser.add_argument("--seeds", nargs="+", type=int, default=list(DEFAULT_SEEDS))
    parser.add_argument(
        "--no-subset",
        action="store_true",
        help="不下采样（tier2 的任务用它：它们的训练行数就是匹配档本身）；"
        "档位标签取该任务的 n_train，故与 --levels 互斥",
    )
    parser.add_argument("--out", default=str(OUT_PATH), help="输出 CSV（默认 results/aggregate/n_matched.csv）")
    parser.add_argument("--device", default="auto", choices=("auto", "cuda", "cpu"))
    parser.add_argument("--dry-run", action="store_true", help="只打印网格")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    module = _module(args.arm)
    from chainworth.arms import graph as G  # 只用 load_arm_config / replace，导入不触发 chemprop

    frozen = G.load_arm_config(ARM_CONFIGS[args.arm])
    # 内存里补 subset：文件一个字节不动，故 arm_config_sha256 仍是冻结文件的哈希
    config = replace(
        frozen,
        raw={**frozen.raw, "subset": {"rule": "shuffle_prefix", "seed": SUBSET_SEED}},
    )
    config_sha = _sha256(ARM_CONFIGS[args.arm])

    grid = (
        [(t, None, s) for t in args.tasks for s in args.seeds]
        if args.no_subset
        else [(t, lv, s) for t in args.tasks for lv in args.levels for s in args.seeds]
    )
    mode = "不下采样（档位 = 该任务自己的 n_train）" if args.no_subset else "下采样到指定训练行数"
    print(f"臂 {args.arm}：{len(grid)} 条 run；{mode}")
    print(f"冻结配置 {ARM_CONFIGS[args.arm].name} sha256:{config_sha[:12]}…（未改动）；subset.seed={SUBSET_SEED}")
    if args.dry_run:
        for task, level, seed in grid:
            print(f"  {task} n={level or 'all'} seed={seed}")
        return 0

    out_path = Path(args.out)
    rows: list[dict[str, Any]] = [
        {k: v for k, v in row.items()} for row in _read_existing(out_path)
    ]
    done = {(r["task"], r["arm"], r["n_level"], r["fold"], str(r["seed"])) for r in rows}

    failures = 0
    added = 0
    for task, level, seed in grid:
        data = module.build_task_data(task)
        fold = data.split.folds[0].name
        # `--no-subset`：档位标签取该任务自己的训练行数（tier2 的任务用它，它们的 n_train 就是匹配档）
        n_request = None if args.no_subset else level
        n_label = str(int(data.split.folds[0].train_index.size)) if args.no_subset else str(level)
        key = (task, args.arm, n_label, fold, str(seed))
        if key in done:
            print(f"  跳过（已有） {task} n={n_label} seed={seed}")
            continue
        started = time.time()
        try:
            if args.arm == "a2_conventional":
                outcome = module.train_one(
                    data,
                    config,
                    fold=fold,
                    seed=seed,
                    n_level=n_request,
                    best_params_path=BEST_PARAMS,
                )
            elif args.arm == "a3_graph":
                outcome = module.train_one(
                    data, config, fold=fold, seed=seed, n_level=n_request, device=args.device
                )
            else:
                outcome = module.train_one(data, config, fold=fold, seed=seed, n_level=n_request)
        except Exception as error:  # noqa: BLE001 - 逐条记录，不因一条失败整批退出
            failures += 1
            print(f"  [失败] {task} n={n_label} seed={seed}: {type(error).__name__}: {error}")
            continue
        record = outcome.record
        metrics = record.get("metrics", {})
        rows.append(
            {
                "task": task,
                "arm": args.arm,
                "n_level": n_label,
                "n_train": record.get("n_train", ""),
                "n_test": record.get("n_test", ""),
                "fold": fold,
                "seed": seed,
                "r2": f"{metrics.get('r2', float('nan')):.10f}",
                "mae": f"{metrics.get('mae', float('nan')):.10f}",
                "rmse": f"{metrics.get('rmse', float('nan')):.10f}",
                "wall_clock_s": f"{time.time() - started:.1f}",
                "subset_sha256": record.get("n_subset_sha256", ""),
                "subset_seed": SUBSET_SEED,
                "arm_config_sha256": config_sha,
                "note": (
                    "全量训练（tier2 的匹配档 = 它自己的 n_train）；不写 raw_runs（D-34/D-66）"
                    if args.no_subset
                    else "train 下采样，test 逐行不变；不写 raw_runs（D-34/D-66）"
                ),
            }
        )
        done.add(key)
        added += 1
        print(
            f"  {task} n={n_label} seed={seed}  R²={metrics.get('r2', float('nan')):.4f}  "
            f"n_train={record.get('n_train')}  {time.time() - started:.1f}s"
        )
        _write(out_path, rows)

    _write(out_path, rows)
    print(f"\n已写 {out_path.relative_to(REPO_ROOT)}：共 {len(rows)} 行，本轮新增 {added} 行")
    if failures:
        print(f"有 {failures} 条失败，见上")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
