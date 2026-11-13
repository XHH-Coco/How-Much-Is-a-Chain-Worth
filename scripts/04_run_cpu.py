"""阶段入口 04：经典臂（A1 地板 / A2 经典）的调参与批量运行。
Stage entry 04: hyper-parameter search and batch runs for the classical arms (A1 floor, A2).

中文：
    用途：C-a 的入口。五件事：
          ① `search`  —— 在训练行内做 Murcko 分组 CV，把每个任务的冻结超参写进
                        `results/aggregate/a2_best_params.csv`（≤30 trials）；
          ② `run`     —— 按配置网格跑 A1/A2，逐 run 落记录、预测与墙钟，并刷新
                        `results/aggregate/a1_a2_index.csv`（给 C-c 的唯一入口）；
          ③ `index`   —— 只按磁盘上现有的记录重建索引，不训练；
          ④ `sensitivity` —— RF / kNN 的学习器敏感性，另存一张表，**不进主表、不写 raw_runs**；
          ⑤ `control` —— PB26 受控对照：训练量 / 特征 / 模型三项各自单独改，用来拆开与官方的差。
    怎么用：
          python scripts/04_run_cpu.py search --tasks pb_t1_tg
          python scripts/04_run_cpu.py run --arm a2_conventional --tasks pb_t1_tg
          python scripts/04_run_cpu.py run --arm a1_floor --tasks pb_t1_tg
          python scripts/04_run_cpu.py run --arm a2_conventional --dry-run
          python scripts/04_run_cpu.py run --arm a2_conventional --tasks pb_t3_iwlt --limit 1200 --seeds 0
                        # 冒烟：只写 results/aggregate/partial/ 下的 *.partial.*，绝不进 raw_runs
          python scripts/04_run_cpu.py repeatability --task pb_t1_tg --seed 0
                        # 同 seed 连跑两遍，写 a2_repeatability.json（C-a 自检的验收④）
          python scripts/04_run_cpu.py sensitivity --learners rf knn
                        # 学习器敏感性 → learner_sensitivity.csv（不进主表）
          python scripts/04_run_cpu.py control --variants baseline train_size no_fingerprints learner_rf official_recipe
                        # PB26 受控对照 → pb26_controlled_control.csv
    依赖：`environment/core.yml` 的同版本包（numpy / pandas / pyarrow / scikit-learn / xgboost /
          rdkit / PyYAML）。本机用 `DSC6020/env_core` 的 venv（D-39）。
    配置：`configs/arms/a1_floor.yaml`、`configs/arms/a2_conventional.yaml`、
          `configs/experiments/main_table.yaml`（expected_runs 的唯一出处）。
    对外：命令行。记录只能由训练代码落盘，**不许手写**（红线 4）。
    被谁用：C-a（每天跑）、C-b（取超参表）、C-c（取索引）。

English:
    Purpose: C-a's entry point. `search` runs Murcko-grouped CV inside the training rows and writes
    the frozen per-task hyper-parameters; `run` executes the configured grid, writing records,
    predictions and wall clocks, and refreshes the index C-c reads; `index` rebuilds that index
    from what is already on disk.
    Smoke runs (`--limit`) write only `results/aggregate/partial/*.partial.*`, never `raw_runs/`.
    Depends on the pinned core package set; records are written by the training code only.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
from dataclasses import replace
from datetime import datetime
from pathlib import Path
from typing import Any, Sequence

import yaml

# 让 `python scripts/04_run_cpu.py` 直接可用（同 05 的做法：不 pip install -e）。
# Runnable without installing the package, same as entry 05.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

import numpy as np  # noqa: E402

from chainworth.arms import conventional as C  # noqa: E402
from chainworth.arms import floor as FL  # noqa: E402
from chainworth.arms import graph as G  # noqa: E402

LOGGER = logging.getLogger("chainworth.scripts.04")

ARM_MODULES = {"a1_floor": FL, "a2_conventional": C}
ARM_CONFIGS = {
    "a1_floor": REPO_ROOT / "configs" / "arms" / "a1_floor.yaml",
    "a2_conventional": REPO_ROOT / "configs" / "arms" / "a2_conventional.yaml",
}
PARTIAL_ROOT = REPO_ROOT / "results" / "aggregate" / "partial"
SENS_CONFIG = REPO_ROOT / "configs" / "experiments" / "learner_sensitivity.yaml"
SENS_TABLE = REPO_ROOT / "results" / "aggregate" / "learner_sensitivity.csv"
SENS_PARAMS = REPO_ROOT / "results" / "aggregate" / "learner_sensitivity_params.csv"
CONTROL_CONFIG = REPO_ROOT / "configs" / "experiments" / "pb26_control.yaml"
CONTROL_TABLE = REPO_ROOT / "results" / "aggregate" / "pb26_controlled_control.csv"
CONTROL_SUMMARY = REPO_ROOT / "results" / "aggregate" / "pb26_controlled_control_summary.json"
NULL_CONFIG = REPO_ROOT / "configs" / "experiments" / "null_calibration.yaml"
NULL_TABLE = REPO_ROOT / "results" / "aggregate" / "null_calibration.csv"
NULL_SUMMARY = REPO_ROOT / "results" / "aggregate" / "null_calibration_summary.json"

SENS_COLUMNS = (
    "task",
    "learner",
    "fold",
    "seed",
    "n_train",
    "n_test",
    "r2",
    "mae",
    "rmse",
    "wall_clock_s",
    "cv_r2",
    "search_trials",
    "hyperparams",
)
CONTROL_COLUMNS = (
    "task",
    "variant",
    "changes",
    "learner",
    "features",
    "n_features",
    "n_train",
    "fold",
    "seed",
    "r2",
    "mae",
    "rmse",
    "wall_clock_s",
    "hyperparams",
)
NULL_COLUMNS = (
    "task",
    "condition",
    "action",
    "learner",
    "n_features",
    "added_columns",
    "source",
    "n_train",
    "n_test",
    "fold",
    "seed",
    "r2",
    "mae",
    "rmse",
    "wall_clock_s",
    "hyperparams",
)


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """命令行参数。／ Command-line arguments."""
    parser = argparse.ArgumentParser(
        prog="04_run_cpu.py",
        description="C-a：A1/A2 经典臂的超参搜索与批量运行（CPU）",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    search = sub.add_parser("search", help="Murcko 分组 CV 选超参，写 a2_best_params.csv")
    search.add_argument("--tasks", nargs="+", metavar="TASK", required=True)
    search.add_argument("--learner", default="xgboost", choices=("xgboost", "rf", "knn"))
    search.add_argument("--config", default=None)
    search.add_argument("--dry-run", action="store_true", help="只算不写 CSV")

    run = sub.add_parser("run", help="按网格跑 A1/A2，落记录、预测、墙钟与索引")
    run.add_argument("--arm", default="a2_conventional", choices=tuple(ARM_MODULES))
    run.add_argument("--config", default=None)
    run.add_argument("--tasks", nargs="+", metavar="TASK")
    run.add_argument("--seeds", nargs="+", type=int, metavar="S")
    run.add_argument("--limit", type=int, default=None, help="冒烟：只取前 N 行，产物写 *.partial.*")
    run.add_argument("--dry-run", action="store_true", help="只打印网格，不训练不写文件")

    index = sub.add_parser("index", help="按磁盘上的 A1/A2 记录重建 a1_a2_index.csv")
    index.add_argument("--dry-run", action="store_true")

    repeat = sub.add_parser("repeatability", help="同 seed 连跑两遍，写 a2_repeatability.json")
    repeat.add_argument("--config", default=None)
    repeat.add_argument("--task", default="pb_t1_tg")
    repeat.add_argument("--fold", default="official_70_30")
    repeat.add_argument("--seed", type=int, default=0)
    repeat.add_argument("--out", default=None)

    sens = sub.add_parser(
        "sensitivity", help="RF/kNN 学习器敏感性：另存 CSV，不进主表、不写 raw_runs"
    )
    sens.add_argument("--tasks", nargs="+", metavar="TASK")
    sens.add_argument("--learners", nargs="+", default=["rf", "knn"], choices=("rf", "knn"))
    sens.add_argument("--seeds", nargs="+", type=int, metavar="S")
    sens.add_argument("--dry-run", action="store_true", help="只打印计划，不训练不写文件")

    ctl = sub.add_parser(
        "control", help="PB26 受控对照：训练量 / 特征 / 模型各自单改，拆开与官方的差"
    )
    ctl.add_argument("--tasks", nargs="+", metavar="TASK")
    ctl.add_argument("--variants", nargs="+", metavar="V")
    ctl.add_argument("--seeds", nargs="+", type=int, metavar="S")
    ctl.add_argument("--dry-run", action="store_true", help="只打印计划，不训练不写文件")

    nul = sub.add_parser(
        "nullcal", help="零信息对照：多 3 列常数 / 置换列能带来多少 R²（A5 地板）"
    )
    nul.add_argument("--tasks", nargs="+", metavar="TASK")
    nul.add_argument("--conditions", nargs="+", metavar="C")
    nul.add_argument("--seeds", nargs="+", type=int, metavar="S")
    nul.add_argument("--dry-run", action="store_true", help="只打印计划，不训练不写文件")

    return parser.parse_args(argv)


def _load(arm: str, config_path: str | None) -> tuple[Any, G.ArmConfig]:
    """取臂模块与它冻结的配置。／ Fetch the arm module and its frozen config."""
    module = ARM_MODULES[arm]
    path = Path(config_path) if config_path else ARM_CONFIGS[arm]
    return module, G.load_arm_config(path)


def _records_on_disk(arms: Sequence[str], *, root: Path | None = None) -> list[dict[str, Any]]:
    """把 results/raw_runs 里属于这些臂的 ok 记录读回来（按 run_id 排序）。
    Read back the ok records belonging to these arms, sorted by run_id.
    """
    directory = Path(root) if root is not None else G.RAW_RUNS_DIR
    if not directory.is_dir():
        return []
    wanted = set(arms)
    records: list[dict[str, Any]] = []
    for path in sorted(directory.glob("*.json")):
        if path.name == "_schema.json":
            continue
        try:
            record = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            LOGGER.warning("跳过无法解析的记录 %s", path.name)
            continue
        if record.get("status") == "ok" and record.get("arm") in wanted:
            records.append(record)
    return records


def _run_smoke(
    module: Any, config: G.ArmConfig, tasks: Sequence[str], seeds: Sequence[int], limit: int
) -> list[dict[str, Any]]:
    """冒烟：直接调 `build_task_data` + `train_one`，产物写 `partial/` 且文件名带 `.partial`。
    Smoke path: call the builder and trainer directly; every artefact lands in `partial/` with a
    `.partial` marker in its name.

    中文：用途：`graph.run_grid` 的文件名是 `<run_id>.json`，没有地方放 `.partial` 后缀，
          所以冒烟不走它——宁可多十行代码，也不让调试数混进 `results/raw_runs/`。
    English: `run_grid` names files `<run_id>.json` with nowhere to put `.partial`, so the smoke path
    does not go through it: ten extra lines beat letting debug numbers into `raw_runs/`.
    """
    import numpy as np
    import pyarrow as pa
    import pyarrow.parquet as pq

    PARTIAL_ROOT.mkdir(parents=True, exist_ok=True)
    written: list[dict[str, Any]] = []
    for task in tasks:
        data = module.build_task_data(task, limit=limit)
        for fold in data.split.fold_names:
            for seed in seeds:
                outcome = module.train_one(data, config, fold=fold, seed=seed)
                record = dict(outcome.record)
                run_id = record["run_id"]
                pred_path = PARTIAL_ROOT / f"{run_id}.partial.parquet"
                pq.write_table(
                    pa.table(
                        {
                            "row_id": pa.array(np.asarray(outcome.row_id, dtype=np.int64)),
                            "y_pred": pa.array(np.asarray(outcome.y_pred, dtype=np.float64)),
                        }
                    ),
                    pred_path,
                )
                record["predictions"] = str(pred_path.relative_to(REPO_ROOT).as_posix())
                record["notes"] = (record.get("notes", "") + "; PARTIAL smoke run, not for the table").strip("; ")
                path = PARTIAL_ROOT / f"{run_id}.partial.json"
                path.write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n")
                written.append(record)
                print(f"{path.name} → r2={record['metrics']['r2']:.4f} n_train={record['n_train']} n_test={record['n_test']}")
    return written


def _task_grid(raw: dict[str, Any]) -> list[tuple[str, list[str]]]:
    """从 A2 配置的 `runs` 取 (task, folds)，顺序 = yaml 的书写顺序。
    Read (task, folds) from A2's `runs`, in the order the YAML writes them.

    中文：用途：敏感性表与受控对照都**不自己写任务清单**，一律抄 A2 的网格，
          否则主表与附加表的任务/折会对不上。
    English: neither extra table writes its own task list; both copy A2's grid so their
    tasks and folds line up with the main table.
    """
    grid: list[tuple[str, list[str]]] = []
    for group in (raw.get("runs") or {}).values():
        folds = [str(name) for name in (group.get("folds") or [])]
        for task in group.get("tasks") or []:
            if not folds:
                raise SystemExit(f"{task} 在 A2 配置里没有 folds：无法排网格")
            grid.append((str(task), folds))
    return grid


def _write_rows(path: Path, columns: Sequence[str], rows: Sequence[dict[str, Any]]) -> Path:
    """整表覆盖写 CSV（这两张附加表每次全量重算，不做按行合并）。
    Rewrite the whole CSV: both extra tables are recomputed in full, never merged row-wise.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(columns), lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    return path


def _sensitivity(
    tasks: Sequence[str] | None, learners: Sequence[str], seeds: Sequence[int], *, dry_run: bool
) -> int:
    """RF / kNN 的学习器敏感性：每任务每学习器一块，逐折逐种子训练，写独立 CSV。
    Learner sensitivity: one block per (task, learner), one fit per fold and seed, into its own CSV.

    中文：
        用途：回答"A2 的结论是不是 XGBoost 特有的"。RF 不调参（固定成官方 RDKit_RF 的 100 树），
              kNN 走 8 点网格搜索；两者的折、种子、划分与主表**逐条相同**，所以可以直接并排读。
              **不写 `results/raw_runs/`**：`_schema.json` 的 arm 枚举里没有"敏感性"，写进去会让
              C-c 的 144 条对不上；这里只落 `learner_sensitivity*.csv`。
        输入：tasks（None = A2 网格里的全部）；learners；seeds（None = 敏感性配置的 seeds）；
              dry_run —— 只打印计划，不训练。
        输出：写 CSV，返回 0。副作用：训练（内存中）、写两张 CSV。
    English:
        Answers "is A2's conclusion XGBoost-specific?". RF is fixed at the official recipe's 100
        trees (no tuning, and the asymmetry is stated in the config); kNN runs the 8-point grid.
        Folds, seeds and splits are identical to the main table. Nothing is written to
        `raw_runs/`, whose arm enum has no "sensitivity" value.
    """
    raw = yaml.safe_load(SENS_CONFIG.read_text(encoding="utf-8"))
    specs = raw.get("learners") or {}
    seeds = list(seeds) if seeds else [int(seed) for seed in raw.get("seeds") or [0, 1, 2]]
    a2_raw = yaml.safe_load(ARM_CONFIGS["a2_conventional"].read_text(encoding="utf-8"))
    grid = dict(_task_grid(a2_raw))
    chosen = list(tasks) if tasks else list(grid)
    unknown = [task for task in chosen if task not in grid]
    if unknown:
        raise SystemExit(f"A2 配置的 runs 里没有这些任务：{unknown}")
    for learner in learners:
        if learner not in specs:
            raise SystemExit(f"{SENS_CONFIG.name} 里没有学习器 {learner!r}：只认 {sorted(specs)}")

    a2_config = G.load_arm_config(ARM_CONFIGS["a2_conventional"])
    table = Path(str(raw.get("table") or SENS_TABLE.relative_to(REPO_ROOT)))
    table = table if table.is_absolute() else REPO_ROOT / table
    params_table = Path(str(raw.get("params_table") or SENS_PARAMS.relative_to(REPO_ROOT)))
    params_table = params_table if params_table.is_absolute() else REPO_ROOT / params_table

    print(
        f"sensitivity：{len(chosen)} 个任务 × {len(learners)} 个学习器 × "
        f"{sum(len(grid[t]) for t in chosen)} 个折 × {len(seeds)} 个种子 = "
        f"{sum(len(grid[t]) * len(seeds) for t in chosen) * len(learners)} 次评估 fit"
        "（不含 kNN 的 8 点搜索）"
    )
    if dry_run:
        for task, learner in ((task, learner) for task in chosen for learner in learners):
            print(f"  {task} [{learner}] folds={grid[task]} seeds={seeds} -> {params_table.name}")
        print(f"dry-run：没有训练，也没有写 {table.name}")
        return 0

    rows: list[dict[str, Any]] = []
    summary: dict[tuple[str, str], list[float]] = {}
    for task in chosen:
        data = C.build_task_data(task)
        for learner in learners:
            spec = specs[learner]
            if spec.get("search"):
                print(f"=== search {task} [{learner}] ===", flush=True)
                best = C.search_task(
                    data,
                    a2_config,
                    learner=learner,
                    space=spec.get("space") or {},
                    out_path=params_table,
                    log=lambda message: print(message, flush=True),
                )
                params = dict(best["params"])
                cv_score, trials = float(best["cv_score"]), int(best["trials"])
                print(
                    f"{task} [{learner}] cv_r2={cv_score:.4f} trials={trials} "
                    f"params={json.dumps(params, sort_keys=True)}",
                    flush=True,
                )
            else:
                params = dict(spec.get("fixed") or {})
                if not params:
                    raise SystemExit(f"{SENS_CONFIG.name} 的 {learner} 既没有 search 也没有 fixed")
                cv_score, trials = float("nan"), 0
                print(f"=== {task} [{learner}] fixed {json.dumps(params, sort_keys=True)} ===", flush=True)
            for fold in grid[task]:
                for seed in seeds:
                    outcome = C.train_one(
                        data,
                        a2_config,
                        fold=fold,
                        seed=seed,
                        learner_override=learner,
                        params_override=params,
                    )
                    metrics = outcome.record["metrics"]
                    rows.append(
                        {
                            "task": task,
                            "learner": learner,
                            "fold": fold,
                            "seed": seed,
                            "n_train": outcome.record["n_train"],
                            "n_test": outcome.record["n_test"],
                            "r2": f"{metrics['r2']:.10f}",
                            "mae": f"{metrics['mae']:.10f}",
                            "rmse": f"{metrics['rmse']:.10f}",
                            "wall_clock_s": f"{outcome.record['wall_clock_s']:.3f}",
                            "cv_r2": "" if np.isnan(cv_score) else f"{cv_score:.10f}",
                            "search_trials": trials,
                            "hyperparams": json.dumps(params, sort_keys=True),
                        }
                    )
                    summary.setdefault((task, learner), []).append(float(metrics["r2"]))
                    print(
                        f"  {task} [{learner}] {fold} seed{seed}: r2={metrics['r2']:.4f} "
                        f"n_train={outcome.record['n_train']} wall={outcome.record['wall_clock_s']:.1f}s",
                        flush=True,
                    )
    _write_rows(table, SENS_COLUMNS, rows)
    print(f"写出 {len(rows)} 行 → {table.relative_to(REPO_ROOT).as_posix()}")
    for (task, learner), values in summary.items():
        print(f"  {task} [{learner}] mean r2 = {float(np.mean(values)):.4f} (n={len(values)})")
    return 0


def _control_view(
    data: C.TaskData, spec: dict[str, Any], *, n_descriptors: int, train_rows: int | None
) -> C.TaskData:
    """把一个变体"只改一项"落到 `TaskData` 上：描述符子集 + 训练行前缀。
    Apply one variant to the data: the descriptor-only columns and/or the training-row prefix.

    中文：
        用途：受控对照的全部机关都在这里，所以它只做两件事，且都按 `changes` 里写的做：
              ① `features` —— 只留前 `n_descriptors` 列（205 个描述符），丢掉 2,048 位指纹；
              ② `train_rows` —— 每折的 `train_index` 截成前 `train_rows` 个。
              官方 `create_scaling_splits.py` 写的是 `train = train_pool[:train_size]`，而我们的
              `train_index` 与官方 `train_pool` 逐位相同（`official_split_check.json` 10/10 折），
              所以"前 N 个"就是官方那 N 行——这里没有抽样、没有猜测。
              `test_index` 一行都不动：对照要的是"同一个测试集上换个条件"。
        输入：data；spec（变体定义）；n_descriptors；train_rows（None = 不截）。
        输出：新的 `TaskData`（原对象不动）。副作用：无。
        异常：SystemExit —— `changes` 与 `features` 自相矛盾，或训练池不够 N 行。
    English:
        The whole control lives here: (1) keep the first `n_descriptors` columns, dropping the
        fingerprint bits; (2) truncate each fold's `train_index` to its first `train_rows` entries.
        The official source builds its training subset as `train_pool[:train_size]` and our
        `train_index` equals the official `train_pool` on all 10 folds, so the prefix *is* their
        subset. `test_index` is untouched: the point is the same test set under one changed factor.
    """
    changes = set(spec.get("changes") or [])
    for key in changes:
        if key not in {"train_rows", "features", "learner"}:
            raise SystemExit(f"变体里有不认识的 change {key!r}：只认 train_rows / features / learner")
    if "features" in changes and spec.get("features") != "descriptors_only":
        raise SystemExit(f"`changes` 含 features，但 features={spec.get('features')!r}：口径矛盾")

    X = data.X[:, :n_descriptors] if "features" in changes else data.X
    folds = []
    for fold in data.split.folds:
        if "train_rows" in changes:
            if train_rows is None:
                raise SystemExit(f"{data.task_id} 缺 official_train_size：截断没有行数")
            if train_rows > int(fold.train_index.size):
                raise SystemExit(
                    f"{data.task_id}/{fold.name} 训练池只有 {fold.train_index.size} 行，"
                    f"不够官方口径的 {train_rows} 行"
                )
            fold = replace(fold, train_index=np.asarray(fold.train_index[:train_rows]))
        folds.append(fold)
    return replace(data, X=X, split=replace(data.split, folds=tuple(folds)))


def _control(
    tasks: Sequence[str] | None,
    variants: Sequence[str] | None,
    seeds: Sequence[int],
    *,
    dry_run: bool,
) -> int:
    """PB26 受控对照：五项变体各改一项，拆开"我们与官方"的差。
    PB26 controlled control: five variants, each changing one factor, to decompose the official gap.

    中文：
        用途：回答"我们的 Tg RMSE 比官方低，是因为训练行多、特征多、还是模型不同"。
              五个变体都跑在同一份 seed_42 五折与同一批测试行上：`baseline` 什么都不改；
              `train_size` / `no_fingerprints` / `learner_rf` 各改一项；`official_recipe` 三项
              一起改成官方口径，用来验算整条对比链能不能复现官方 RMSE。
              写 `pb26_controlled_control.csv` 与 `_summary.json`，**不写 `raw_runs/`**（同敏感性表）。
        输入：tasks（None = 配置里 `official_train_size` 的任务）；variants（None = 全部）；
              seeds；dry_run。输出：写 CSV + JSON，返回 0。副作用：训练、写两个文件。
    English:
        Answers "is our lower Tg RMSE due to more training rows, more features, or a different
        model?". Every variant sits on the same seed_42 folds and the same test rows: `baseline`
        changes nothing, three variants change one factor each, and `official_recipe` changes all
        three to check that the chain reproduces the official RMSE.
    """
    raw = yaml.safe_load(CONTROL_CONFIG.read_text(encoding="utf-8"))
    sens = yaml.safe_load(SENS_CONFIG.read_text(encoding="utf-8"))
    rf_fixed = dict(((sens.get("learners") or {}).get("rf") or {}).get("fixed") or {})
    all_variants = raw.get("variants") or {}
    official_size = {str(k): int(v) for k, v in (raw.get("official_train_size") or {}).items()}
    seeds = list(seeds) if seeds else [int(seed) for seed in raw.get("seeds") or [0]]
    chosen = list(tasks) if tasks else list(official_size)
    variants = list(variants) if variants else list(all_variants)

    a2_raw = yaml.safe_load(ARM_CONFIGS["a2_conventional"].read_text(encoding="utf-8"))
    n_descriptors = int(a2_raw["input"]["columns"]["descriptors"])
    n_bits = int(a2_raw["input"]["columns"]["fingerprint_bits"])
    a2_config = G.load_arm_config(ARM_CONFIGS["a2_conventional"])
    grid = dict(_task_grid(a2_raw))

    for task in chosen:
        if task not in grid:
            raise SystemExit(f"A2 配置的 runs 里没有 {task}")
        if task not in official_size:
            raise SystemExit(f"{CONTROL_CONFIG.name} 的 official_train_size 里没有 {task}")
    for name in variants:
        if name not in all_variants:
            raise SystemExit(f"{CONTROL_CONFIG.name} 里没有变体 {name!r}：只认 {sorted(all_variants)}")

    table = REPO_ROOT / str(raw.get("table") or CONTROL_TABLE.relative_to(REPO_ROOT))
    summary_path = REPO_ROOT / str(raw.get("summary") or CONTROL_SUMMARY.relative_to(REPO_ROOT))
    print(
        f"control：{len(chosen)} 个任务 × {len(variants)} 个变体 × "
        f"{sum(len(grid[t]) for t in chosen)} 个折 × {len(seeds)} 个种子 = "
        f"{sum(len(grid[t]) for t in chosen) * len(seeds) * len(variants)} 次 fit"
    )
    if dry_run:
        for task in chosen:
            for name in variants:
                spec = all_variants[name]
                print(
                    f"  {task} [{name}] learner={spec.get('learner')} "
                    f"features={spec.get('features')} changes={spec.get('changes')} "
                    f"train_rows={official_size[task] if 'train_rows' in (spec.get('changes') or []) else 'full'}"
                )
        print(f"dry-run：没有训练，也没有写 {table.name}")
        return 0

    rows: list[dict[str, Any]] = []
    for task in chosen:
        data = C.build_task_data(task)
        if data.X.shape[1] != n_descriptors + n_bits:
            raise SystemExit(
                f"{task} 的输入是 {data.X.shape[1]} 列，配置写的是 {n_descriptors}+{n_bits}："
                "特征包换过，先核 task_features_manifest"
            )
        for name in variants:
            spec = all_variants[name]
            view = _control_view(
                data,
                spec,
                n_descriptors=n_descriptors,
                train_rows=official_size[task],
            )
            learner = str(spec.get("learner") or "xgboost")
            params = rf_fixed if learner == "rf" else None
            n_features = int(view.X.shape[1])
            for fold in grid[task]:
                for seed in seeds:
                    outcome = C.train_one(
                        view,
                        a2_config,
                        fold=fold,
                        seed=seed,
                        learner_override=learner,
                        params_override=params,
                    )
                    metrics = outcome.record["metrics"]
                    rows.append(
                        {
                            "task": task,
                            "variant": name,
                            "changes": "+".join(spec.get("changes") or []) or "none",
                            "learner": learner,
                            "features": str(spec.get("features") or "all"),
                            "n_features": n_features,
                            "n_train": outcome.record["n_train"],
                            "fold": fold,
                            "seed": seed,
                            "r2": f"{metrics['r2']:.10f}",
                            "mae": f"{metrics['mae']:.10f}",
                            "rmse": f"{metrics['rmse']:.10f}",
                            "wall_clock_s": f"{outcome.record['wall_clock_s']:.3f}",
                            "hyperparams": json.dumps(outcome.record["hyperparams"], sort_keys=True),
                        }
                    )
                    print(
                        f"  {task} [{name}] {fold} seed{seed}: rmse={metrics['rmse']:.4f} "
                        f"n_train={outcome.record['n_train']} n_features={n_features} "
                        f"wall={outcome.record['wall_clock_s']:.1f}s",
                        flush=True,
                    )
    _write_rows(table, CONTROL_COLUMNS, rows)
    print(f"写出 {len(rows)} 行 → {table.relative_to(REPO_ROOT).as_posix()}")

    summary = _control_summary(rows, raw)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n"
    )
    print(f"写出汇总 → {summary_path.relative_to(REPO_ROOT).as_posix()}")
    for task, payload in summary["tasks"].items():
        print(f"  {task}: 官方 rmse={payload['official_rmse_mean_across_folds']:.4f} @ {payload['official_train_size']} 行")
        for name, item in payload["variants"].items():
            print(
                f"    {name:<16} rmse={item['rmse_mean']:.4f} r2={item['r2_mean']:.4f} "
                f"n_train={item['n_train']} Δ vs baseline={item['delta_rmse_vs_baseline']:+.4f}"
            )
        if "official_recipe_residual" in payload:
            print(f"    official_recipe 残差 = {payload['official_recipe_residual']:+.4f}")
    return 0


def _control_summary(rows: Sequence[dict[str, Any]], control: dict[str, Any]) -> dict[str, Any]:
    """把逐折的对照行收成"每个变体一个均值 + 与 baseline 的差 + 官方残差"。
    Collapse the per-fold rows into one mean per variant, its delta against baseline, and the
    residual against the official number.
    """
    official_path = REPO_ROOT / str(control["source"]["official_rmse_from"])
    official = json.loads(official_path.read_text(encoding="utf-8"))
    grouped: dict[str, dict[str, list[dict[str, Any]]]] = {}
    for row in rows:
        grouped.setdefault(row["task"], {}).setdefault(row["variant"], []).append(row)

    tasks: dict[str, Any] = {}
    for task, variants in grouped.items():
        reference = official["tasks"][task]
        base_rows = variants.get("baseline") or []
        base_rmse = float(np.mean([float(row["rmse"]) for row in base_rows])) if base_rows else float("nan")
        payload: dict[str, Any] = {
            "quantity": reference.get("quantity"),
            "official_protocol": reference.get("official_protocol"),
            "official_train_size": int(reference["official_train_size"]),
            "official_rmse_mean_across_folds": float(reference["official_rmse_mean_across_folds"]),
            "official_rmse_sd_across_folds": float(reference["official_rmse_sd_across_folds"]),
            "official_rmse_from": control["source"]["official_rmse_from"],
            "baseline_rmse_mean": base_rmse,
            "variants": {},
        }
        for name, items in variants.items():
            rmse = np.asarray([float(row["rmse"]) for row in items], dtype=np.float64)
            r2 = np.asarray([float(row["r2"]) for row in items], dtype=np.float64)
            sd = float(np.std(rmse, ddof=1)) if rmse.size > 1 else 0.0
            payload["variants"][name] = {
                "learner": items[0]["learner"],
                "features": items[0]["features"],
                "changes": items[0]["changes"],
                "n_features": int(items[0]["n_features"]),
                "n_train": int(items[0]["n_train"]),
                "n_folds": int(rmse.size),
                "rmse_mean": float(np.mean(rmse)),
                "rmse_sd": sd,
                "r2_mean": float(np.mean(r2)),
                "delta_rmse_vs_baseline": float(np.mean(rmse) - base_rmse),
            }
        recipe = payload["variants"].get("official_recipe")
        if recipe is not None:
            payload["official_recipe_residual"] = float(
                recipe["rmse_mean"] - payload["official_rmse_mean_across_folds"]
            )
        tasks[task] = payload
    return {
        "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "pinned_commit": official.get("pinned_commit"),
        "source": control["source"],
        "seeds": sorted({int(row["seed"]) for row in rows}),
        "tasks": tasks,
    }


def _null_calibration_view(
    data: C.TaskData,
    spec: dict[str, Any],
    *,
    n_descriptors: int,
    seed: int,
    fold_index: int,
) -> tuple[C.TaskData, str, int]:
    """给一个零信息条件追加 3 列，返回新的 `TaskData`、列的来源说明与追加列数。
    Append the condition's columns and return the new `TaskData`, a provenance string and the count.

    中文：
        用途：回答"A5 比 A2 高的那点分，有多少只是因为多了 3 列"。三个条件：
              · `none`     —— 原样返回，用作配对零点（应与 A2 的 raw_runs 记录逐位相同）；
              · `constants`—— 追加 3 个常数列（0 / 1 / −1）：只加维度，不加任何变化；
              · `shuffle`  —— 从描述符块里挑**训练行方差最大**的 3 列（只在第 0 折的训练行上排序，
                              不看 test），各自用固定种子整列置换。边际分布与真特征一致，
                              但与标签的互信息被打散，因此是"零信息地板"。
        为什么置换全部行而不是只置换训练行：这里要量的就是"零信息"本身；A5 真正的打乱对照
        （只置换 train）等 A5 口径定下来再跑，两者不混。
        输入：data；spec；n_descriptors（描述符列数，用来把挑选限制在描述符块）；
              seed；fold_index（决定置换种子，使每折独立且可复现）。
        输出：(view, source, n_added)。副作用：无（原 `TaskData` 不动）。
        异常：SystemExit —— `action` 不认识，或可用描述符列不足 3 个。
    English:
        `constants` adds three constant columns (dimension only); `shuffle` takes the three
        highest-variance descriptor columns (ranked on fold 0's training rows only), copies them
        and permutes each column with a fixed seed. Both keep the official folds and the test rows
        untouched, so every delta is paired against `baseline` on the same rows.
    """
    action = str(spec.get("action") or "none")
    if action == "none":
        return data, "", 0
    if action == "constants":
        values = list(spec.get("values") or [])
        if not values:
            raise SystemExit(f"{NULL_CONFIG.name} 的 constants 条件缺 values")
        extra = np.tile(np.asarray(values, dtype=data.X.dtype), (data.n_rows, 1))
        return replace(data, X=np.hstack([data.X, extra])), "constants", len(values)
    if action != "shuffle":
        raise SystemExit(f"不认识的零信息 action {action!r}：只认 none / constants / shuffle")

    n_cols = int(spec.get("n_columns") or 3)
    first_train = np.asarray(data.split.folds[0].train_index)
    block = data.X[first_train, :n_descriptors].astype(np.float64)
    variance = np.nanvar(block, axis=0)
    usable = [int(i) for i in np.argsort(variance)[::-1] if np.isfinite(variance[i]) and variance[i] > 0]
    if len(usable) < n_cols:
        raise SystemExit(f"{data.task_id} 只有 {len(usable)} 个非常数描述符列，不够挑 {n_cols} 列")
    picked = usable[:n_cols]

    salt = int(spec.get("rng_salt") or 0)
    rng = np.random.default_rng([salt, int(fold_index), int(seed)])
    extra = np.empty((data.n_rows, n_cols), dtype=data.X.dtype)
    for j, column in enumerate(picked):
        extra[:, j] = data.X[rng.permutation(data.n_rows), column]
    source = "descriptors:" + ",".join(str(i) for i in picked)
    return replace(data, X=np.hstack([data.X, extra])), source, n_cols


def _nullcal(
    tasks: Sequence[str] | None,
    conditions: Sequence[str] | None,
    seeds: Sequence[int],
    *,
    dry_run: bool,
) -> int:
    """零信息对照：A2 同网格上，多 3 列常数 / 置换列各让 R² 动多少。
    Null calibration: how much R2 moves when three constant or permuted columns are appended.

    中文：
        用途：A5 被数据卡住（0 条记录）时，给 D 的图 4 一个"3 列零信息"的地板。
              三个条件跑在同一份官方折、同一批测试行、同一组 A2 超参上，逐折配对求差。
              写 `null_calibration.csv` 与 `_summary.json`，**不写 `raw_runs/`**（D-44 / D-45）。
        输入：tasks（None = 配置里的两个任务）；conditions（None = 全部）；seeds；dry_run。
        输出：写 CSV + JSON，返回 0。副作用：训练、写两个文件。
    English:
        Same grid as A2, same official folds and test rows; only the extra columns differ.
        Writes an auxiliary table under `results/aggregate/`, never into `results/raw_runs/`.
    """
    raw = yaml.safe_load(NULL_CONFIG.read_text(encoding="utf-8"))
    all_conditions = raw.get("conditions") or {}
    seeds = list(seeds) if seeds else [int(seed) for seed in raw.get("seeds") or [0]]
    chosen = list(tasks) if tasks else [str(task) for task in raw.get("tasks") or []]
    conditions = list(conditions) if conditions else list(all_conditions)

    a2_raw = yaml.safe_load(ARM_CONFIGS["a2_conventional"].read_text(encoding="utf-8"))
    n_descriptors = int(a2_raw["input"]["columns"]["descriptors"])
    n_bits = int(a2_raw["input"]["columns"]["fingerprint_bits"])
    a2_config = G.load_arm_config(ARM_CONFIGS["a2_conventional"])
    grid = dict(_task_grid(a2_raw))

    for task in chosen:
        if task not in grid:
            raise SystemExit(f"A2 配置的 runs 里没有 {task}")
    for name in conditions:
        if name not in all_conditions:
            raise SystemExit(
                f"{NULL_CONFIG.name} 里没有条件 {name!r}：只认 {sorted(all_conditions)}"
            )
    if "baseline" not in conditions:
        raise SystemExit("零信息对照必须有 baseline：它是配对差的零点")

    table = REPO_ROOT / str(raw.get("table") or NULL_TABLE.relative_to(REPO_ROOT))
    summary_path = REPO_ROOT / str(raw.get("summary") or NULL_SUMMARY.relative_to(REPO_ROOT))
    n_units = sum(len(grid[t]) for t in chosen)
    print(
        f"nullcal：{len(chosen)} 个任务 × {n_units} 个折 × {len(conditions)} 个条件 × "
        f"{len(seeds)} 个种子 = {n_units * len(conditions) * len(seeds)} 次 fit"
    )
    if dry_run:
        for task in chosen:
            for name in conditions:
                spec = all_conditions[name]
                print(
                    f"  {task} [{name}] action={spec.get('action')} "
                    f"folds={len(grid[task])} seeds={seeds}"
                )
        print(f"dry-run：没有训练，也没有写 {table.name}")
        return 0

    rows: list[dict[str, Any]] = []
    for task in chosen:
        data = C.build_task_data(task)
        if data.X.shape[1] != n_descriptors + n_bits:
            raise SystemExit(
                f"{task} 的输入是 {data.X.shape[1]} 列，配置写的是 {n_descriptors}+{n_bits}："
                "特征包换过，先核 task_features_manifest"
            )
        for name in conditions:
            spec = all_conditions[name]
            for fold_index, fold in enumerate(grid[task]):
                for seed in seeds:
                    view, source, n_added = _null_calibration_view(
                        data,
                        spec,
                        n_descriptors=n_descriptors,
                        seed=seed,
                        fold_index=fold_index,
                    )
                    outcome = C.train_one(view, a2_config, fold=fold, seed=seed)
                    metrics = outcome.record["metrics"]
                    rows.append(
                        {
                            "task": task,
                            "condition": name,
                            "action": str(spec.get("action") or "none"),
                            "learner": outcome.record["learner"],
                            "n_features": int(view.X.shape[1]),
                            "added_columns": n_added,
                            "source": source,
                            "n_train": outcome.record["n_train"],
                            "n_test": outcome.record["n_test"],
                            "fold": fold,
                            "seed": seed,
                            "r2": f"{metrics['r2']:.10f}",
                            "mae": f"{metrics['mae']:.10f}",
                            "rmse": f"{metrics['rmse']:.10f}",
                            "wall_clock_s": f"{outcome.record['wall_clock_s']:.3f}",
                            "hyperparams": json.dumps(outcome.record["hyperparams"], sort_keys=True),
                        }
                    )
                    print(
                        f"  {task} [{name}] {fold} seed{seed}: r2={metrics['r2']:.4f} "
                        f"n_features={view.X.shape[1]} {source} "
                        f"wall={outcome.record['wall_clock_s']:.1f}s",
                        flush=True,
                    )
    _write_rows(table, NULL_COLUMNS, rows)
    print(f"写出 {len(rows)} 行 → {table.relative_to(REPO_ROOT).as_posix()}")

    summary = _nullcal_summary(rows, raw, n_descriptors=n_descriptors, n_bits=n_bits)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n"
    )
    print(f"写出汇总 → {summary_path.relative_to(REPO_ROOT).as_posix()}")
    for task, payload in summary["tasks"].items():
        print(f"  {task}: baseline r2={payload['baseline_r2_mean']:.4f}")
        for name, item in payload["conditions"].items():
            if name == "baseline":
                continue
            print(
                f"    {name:<10} r2={item['r2_mean']:.4f} "
                f"Δ={item['delta_r2_vs_baseline_mean']:+.4f} "
                f"max|Δ|={item['max_abs_paired_delta']:.4f}"
            )
    for name, item in summary["overall"].items():
        print(
            f"  总体 [{name}] 平均 |dR2|={item['mean_abs_delta']:.4f} "
            f"最大 |dR2|={item['max_abs_delta']:.4f}"
        )
    return 0


def _nullcal_summary(
    rows: Sequence[dict[str, Any]],
    config: dict[str, Any],
    *,
    n_descriptors: int,
    n_bits: int,
) -> dict[str, Any]:
    """把逐折的零信息行收成"每任务每条件一个均值 + 与 baseline 的配对差"。
    Collapse the per-fold rows into one mean per condition plus paired deltas against baseline.
    """
    tasks: dict[str, dict[str, Any]] = {}
    all_deltas: dict[str, list[float]] = {}
    for task in dict.fromkeys(row["task"] for row in rows):
        by_condition: dict[str, dict[tuple[str, int], float]] = {}
        for row in rows:
            if row["task"] != task:
                continue
            key = (row["fold"], int(row["seed"]))
            by_condition.setdefault(row["condition"], {})[key] = float(row["r2"])
        base = by_condition["baseline"]
        baseline_r2_mean = float(np.mean(list(base.values())))
        payload: dict[str, Any] = {
            "n_folds": len({fold for fold, _ in base}),
            "baseline_r2_mean": baseline_r2_mean,
            "conditions": {},
        }
        for name, values in by_condition.items():
            if name == "baseline":
                payload["conditions"][name] = {"r2_mean": baseline_r2_mean, "n": len(values)}
                continue
            shared = sorted(set(base) & set(values))
            deltas = [values[key] - base[key] for key in shared]
            all_deltas.setdefault(name, []).extend(deltas)
            payload["conditions"][name] = {
                "r2_mean": float(np.mean(list(values.values()))),
                "delta_r2_vs_baseline_mean": float(np.mean(deltas)) if deltas else None,
                "max_abs_paired_delta": float(np.max(np.abs(deltas))) if deltas else None,
                "n_pairs": len(shared),
            }
        tasks[task] = payload

    overall: dict[str, Any] = {}
    for name, deltas in all_deltas.items():
        if not deltas:
            continue
        overall[name] = {
            "units": len(deltas),
            "mean_delta": float(np.mean(deltas)),
            "mean_abs_delta": float(np.mean(np.abs(deltas))),
            "max_abs_delta": float(np.max(np.abs(deltas))),
            "p95_abs_delta": float(np.quantile(np.abs(deltas), 0.95)),
        }
    return {
        "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "config": "configs/experiments/null_calibration.yaml",
        "a2_config_hash": G.config_hash(G.load_arm_config(ARM_CONFIGS["a2_conventional"])),
        "seeds": sorted({int(row["seed"]) for row in rows}),
        "n_features_base": n_descriptors + n_bits,
        "tasks": tasks,
        "overall": overall,
        "note": (
            "零信息地板：3 列常量与 3 列置换描述符。baseline 应与 raw_runs 里同任务同折 "
            "seed0 的 A2 记录逐位相同（差异只来自 CSV 十位小数）。不进主表、不写 raw_runs。"
        ),
    }


def main(argv: Sequence[str] | None = None) -> int:
    """命令行入口。／ Command-line entry point."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    args = _parse_args(argv)

    if args.command == "search":
        _, config = _load("a2_conventional", args.config)
        if G.KNOWN_ARMS and str(config.raw.get("arm")) != "a2_conventional":
            raise SystemExit(f"search 只写 A2 的超参表，收到 arm={config.raw.get('arm')!r}")
        rows = []
        for task in args.tasks:
            print(f"=== search {task} [{args.learner}] ===", flush=True)
            data = C.build_task_data(task)
            best = C.search_task(
                data,
                config,
                learner=args.learner,
                dry_run=args.dry_run,
                log=lambda message: print(message),
            )
            rows.append((task, best))
            print(
                f"{task} [{args.learner}] cv_r2={best['cv_score']:.4f} "
                f"trials={best['trials']} folds={best['cv_folds']} "
                f"rows={best['n_search_rows']} params={json.dumps(best['params'], sort_keys=True)}"
            )
        if args.dry_run:
            print("dry-run：没有写 a2_best_params.csv")
        return 0

    if args.command == "repeatability":
        _, config = _load("a2_conventional", args.config)
        data = C.build_task_data(args.task)
        first = C.train_one(data, config, fold=args.fold, seed=args.seed)
        second = C.train_one(data, config, fold=args.fold, seed=args.seed)
        delta = np.abs(first.y_pred - second.y_pred)
        payload = {
            "run_id": first.record["run_id"],
            "repeats": 2,
            "max_abs_prediction_delta": float(delta.max()),
            "median_abs_prediction_delta": float(np.median(delta)),
            "p99_abs_prediction_delta": float(np.quantile(delta, 0.99)),
            "n_test": int(first.row_id.size),
            "per_repeat_metrics": [first.record["metrics"], second.record["metrics"]],
            "threshold": 1e-9,
            "passed": bool(float(delta.max()) < 1e-9),
            "device": first.record["device"],
            "config_hash": first.record["config_hash"],
            "checked_at": datetime.now().astimezone().isoformat(timespec="seconds"),
            "note": (
                "同配置、同折、同种子连跑两遍；经典模型是确定性的，预测应逐位相同。"
                "这里比验收④的 r2 差 < 1e-9 更严：直接比预测向量。"
            ),
        }
        out = Path(args.out) if args.out else REPO_ROOT / "results" / "aggregate" / "a2_repeatability.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(
            f"{payload['run_id']}: max|delta|={payload['max_abs_prediction_delta']:.3e} "
            f"passed={payload['passed']} -> {out.relative_to(REPO_ROOT)}"
        )
        return 0 if payload["passed"] else 1

    if args.command == "run":
        module, config = _load(args.arm, args.config)
        if args.dry_run:
            keys = G.enumerate_runs(config, tasks=args.tasks, seeds=args.seeds)
            print(f"{len(keys)} 条 run（arm={config.raw.get('arm')}）")
            for key in keys:
                print(key.run_id())
            return 0
        if args.limit is not None:
            seeds = args.seeds or list(config.raw["seeds"])
            written = _run_smoke(module, config, args.tasks or [], seeds, args.limit)
            print(f"冒烟完成 {len(written)} 条，产物在 {PARTIAL_ROOT.relative_to(REPO_ROOT)}/（*.partial.*）")
            return 0
        records = module.run_arm(config, tasks=args.tasks, seeds=args.seeds)
        if records:
            path = C.write_index(records)
            print(f"写出 {len(records)} 条记录；索引已刷新：{path}")
        else:
            print("没有新记录（全部命中 resume）")
        return 0

    if args.command == "index":
        records = _records_on_disk(("a1_floor", "a2_conventional"))
        if args.dry_run:
            print(f"磁盘上 A1/A2 的 ok 记录 {len(records)} 条")
            return 0
        path = C.write_index(records)
        print(f"索引已重建：{path}（{len(records)} 条）")
        return 0

    if args.command == "sensitivity":
        return _sensitivity(args.tasks, args.learners, args.seeds or [], dry_run=args.dry_run)

    if args.command == "control":
        return _control(args.tasks, args.variants, args.seeds or [], dry_run=args.dry_run)

    if args.command == "nullcal":
        return _nullcal(args.tasks, args.conditions, args.seeds or [], dry_run=args.dry_run)

    raise SystemExit(f"未知子命令 {args.command!r}")


if __name__ == "__main__":                             # pragma: no cover - 命令行入口
    raise SystemExit(main())
