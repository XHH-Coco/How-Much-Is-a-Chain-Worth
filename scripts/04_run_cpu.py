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
          rdkit / PyYAML）。本机用 `DSC6020/env_core` 的 venv（D-40）。
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
# C-b-2：OpenPoly 规模扫描的经典臂一侧（225 条）。配置里带 `n_levels`，网格比 A2 多一个维度。
# C-b-2: the classical side of the OpenPoly scale scan (225 runs). The config carries `n_levels`,
# which is the one dimension A2's own grid does not have.
OPENPOLY_CLASSIC_CONFIG = REPO_ROOT / "configs" / "experiments" / "openpoly_classic.yaml"
# 对账靶子：B-5 的 GNN 一侧索引（**只读**）。两边的 `subset_sha256` 必须逐档相等，
# 否则 H4 的交叉点不是"臂的差"而是"子集的差"（分工_C-b 红线 2）。
# The reconciliation target: B-5's GNN-side index, read-only. The two arms' `subset_sha256` values
# must match level by level, or H4's crossing point differences the subset, not the arm.
OPENPOLY_GPU_N_INDEX = REPO_ROOT / "results" / "aggregate" / "openpoly_n_index.csv"
# C-b-1：PolyBench26 的数据量扫描（H2 的 κ）。档位是官方 `train_sizes` 的前缀。
# C-b-1: PolyBench26's data-size scan (H2's kappa); a tier is a prefix of the official train_sizes.
SCALING_CONFIG = REPO_ROOT / "configs" / "experiments" / "scaling.yaml"
# C-c-1/C-c-2：架构迁移 15 runs（D-49 单臂）+ OOD 三段式。冻结后其字节哈希进每条记录的 config_hash。
# C-c-1/C-c-2: the architecture transfer 15 runs (D-49 single arm) + OOD three stages.
ARCH_CONFIG = REPO_ROOT / "configs" / "experiments" / "arch_transfer.yaml"

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

    # C-b-2：OpenPoly 规模扫描的经典臂（225 条 = 3 性质 × 5 档 × 5 折 × 3 种子）。
    # 复用同一个入口而不新开编号：与 B-5 复用 05 是同一条理由（D-33 ①），且 06 是留给
    # C 的汇总脚本的（D-43），另开一个 06 会把那一格占掉。
    # C-b-2's classical OpenPoly scan reuses this entry instead of taking a new number: the same
    # reason B-5 reused 05 (D-33), and 06 is reserved for C's aggregation script (D-43).
    opc = sub.add_parser(
        "openpoly", help="C-b-2：OpenPoly 经典臂的 n 档扫描（225 条）+ subset_sha256 对账"
    )
    opc.add_argument("--config", default=None)
    opc.add_argument("--tasks", nargs="+", metavar="TASK")
    opc.add_argument("--seeds", nargs="+", type=int, metavar="S")
    opc.add_argument("--n-levels", nargs="+", metavar="N", help="只跑这些档（30 60 120 240 all）")
    opc.add_argument(
        "--index-only", action="store_true", help="不训练：只按磁盘上的记录刷新两份 CSV 并做对账"
    )
    opc.add_argument("--dry-run", action="store_true", help="只打印网格，不训练不写文件")

    # C-b-1：PolyBench26 的数据量扫描（H2 的 κ）。60–80 fits = 8 档 × 5 折 × 2 学习器。
    # C-b-1: PolyBench26's data-size scan (H2's kappa): 80 fits = 8 tiers x 5 folds x 2 learners.
    scl = sub.add_parser(
        "scaling", help="C-b-1：数据量扫描（双数据集的 pack、档位前缀、60–80 fits）"
    )
    scl.add_argument("--config", default=None)
    scl.add_argument("--datasets", nargs="+", metavar="DATASET")
    scl.add_argument("--tiers", nargs="+", metavar="N")
    scl.add_argument("--folds", nargs="+", metavar="FOLD")
    scl.add_argument("--learners", nargs="+", metavar="L", help="xgboost / rf")
    scl.add_argument(
        "--build", action="store_true", help="先造（或核）两份特征 pack，再跑拟合"
    )
    scl.add_argument(
        "--build-only", action="store_true", help="只造特征 pack，不训练"
    )
    scl.add_argument("--force", action="store_true", help="配合 --build：pack 已存在也重造")
    scl.add_argument(
        "--workers",
        type=int,
        default=0,
        metavar="N",
        help="造 pack 时的进程数（0 = 核数−1，上限 8；RDKit 不在线程间并行）",
    )
    scl.add_argument(
        "--table-only", action="store_true", help="不训练：只按磁盘上的记录重写 scaling_cpu.csv"
    )
    scl.add_argument(
        "--search",
        action="store_true",
        help="先按预注册协议搜一次 pb26_scaling_cp 的超参（写进 a2_best_params.csv 的一行），再跑",
    )
    scl.add_argument(
        "--search-dataset",
        default="md5000_alternating_cp",
        metavar="DATASET",
        help="用哪个数据集的 fold_00 训练行选参（默认交替 Cp：池子最大、代表性强）",
    )
    scl.add_argument("--dry-run", action="store_true", help="只打印网格，不训练不写文件")

    arch = sub.add_parser(
        "arch", help="C-c-1/C-c-2：架构迁移 15 runs（a2 单臂）+ OOD 三段式（分工_C-c）"
    )
    arch.add_argument("--config", default=None, help="实验配置（默认 configs/experiments/arch_transfer.yaml）")
    arch.add_argument(
        "--setup", action="store_true",
        help="划分落位 splits/polybench26/arch/ + 造 ID 特征（6,478 × 2,253）",
    )
    arch.add_argument("--verify", action="store_true", help="只读复核：物化行数、组分覆盖、特征哈希")
    arch.add_argument(
        "--search", action="store_true",
        help="按 D-51 的预注册协议搜一次 pb26_arch_transfer_ea 的超参（写 a2_best_params.csv 第 13 行）",
    )
    arch.add_argument("--ood", action="store_true", help="OOD 三段式 + 五折集成点值")
    arch.add_argument("--table-only", action="store_true", help="不训练：按磁盘记录与 OOD 指标重写 arch_transfer.csv")
    arch.add_argument("--folds", nargs="+", metavar="FOLD", help="白名单（默认全部 5 折）")
    arch.add_argument("--seeds", nargs="+", type=int, metavar="SEED", help="白名单（默认 0 1 2）")
    arch.add_argument("--no-resume", action="store_true", help="已有的 ok 记录/OOD 产物也重跑")
    arch.add_argument(
        "--workers",
        type=int,
        default=0,
        metavar="N",
        help="造 ID 特征时的进程数（0 = 核数−1，上限 8）",
    )
    arch.add_argument("--dry-run", action="store_true", help="只打印网格，不训练不写文件")

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
              C-c 的 96 条对不上；这里只落 `learner_sensitivity*.csv`。
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
              写 `null_calibration.csv` 与 `_summary.json`，**不写 `raw_runs/`**（D-45 / D-46）。
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


def _openpoly_subset_map(path: Path) -> dict[tuple[str, str, str], str]:
    """读一份 n 索引 CSV，取 (task, fold, 档位) → `subset_sha256`。
    Read one n-index CSV into a (task, fold, level) -> subset_sha256 map.

    中文：用途：对账只认"哪一批训练行"这一个问题，所以只取这三列加哈希。
          输入：path —— CSV 路径。输出：字典；文件不存在返回空字典。
          数据来源：`results/aggregate/*_n_index.csv`。副作用：只读。异常：无（解析失败的行跳过）。
    English: Reconciliation only asks "which training rows", so only these three columns and the
    digest are read. Returns an empty map when the file is absent.
    """
    table: dict[tuple[str, str, str], str] = {}
    if not path.is_file():
        return table
    with path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            key = (str(row.get("task", "")), str(row.get("fold", "")), str(row.get("n_level", "")))
            table[key] = str(row.get("subset_sha256", ""))
    return table


def _openpoly(
    config_path: str | None,
    tasks: Sequence[str] | None,
    seeds: Sequence[int],
    n_levels: Sequence[str] | None,
    *,
    dry_run: bool,
    index_only: bool,
) -> int:
    """C-b-2：跑 OpenPoly 经典臂的 225 条 n 档网格，刷新两份 CSV，并与 B-5 逐档对账。
    C-b-2: run the 225-run OpenPoly classical n-grid, refresh the two CSVs, reconcile against B-5.

    中文：
        用途：把"3 性质 × 5 档 × 5 折 × 3 种子 = 225 条"跑成记录与两张表，然后回答一个
              只有它能回答的问题——**我和 B 的 GNN 是不是落在同一批训练行上**。
              对账不靠"大概一样"：逐 (task, fold, 档位) 比 `subset_sha256`，一条不同就返回 1。
        输入：tasks / seeds / n_levels —— 白名单（None = 配置里的全部）；dry_run —— 只打印网格；
              index_only —— 不训练，只按磁盘上的记录刷新 CSV 并做对账。
        输出：0 成功且对账全对；1 对账有差异（**停**，别拿这批数去算 H4）。
        数据来源：`data/processed/features/op_*.npz`、`splits/openpoly/**`、`a2_best_params.csv`。
        配置来源：`configs/experiments/openpoly_classic.yaml`。
        副作用：训练、写记录/预测/墙钟、写两份 CSV。
    English:
        Runs the 225-run grid into records plus two tables, then answers the one question only this
        arm can answer: are B's GNN and this classical arm on **the same training rows**? Not "roughly
        the same" - the check compares `subset_sha256` per (task, fold, level) and returns 1 on any
        mismatch, because H4's crossing point would then measure the subset, not the arm.
    """
    config = G.load_arm_config(Path(config_path) if config_path else OPENPOLY_CLASSIC_CONFIG)
    if not config.raw.get("n_levels"):
        raise SystemExit(f"{config.path.name} 没有 `n_levels`：OpenPoly 扫描必须有这一维")
    keys = G.enumerate_runs(config, tasks=tasks, seeds=seeds, n_levels=n_levels)
    levels = sorted({str(key.n_level) for key in keys}, key=lambda text: (text != "all", text))
    print(
        f"openpoly：{len(keys)} 条 = {len({key.task for key in keys})} 性质 × {len(levels)} 档 "
        f"({', '.join(levels)}) × {len({key.fold for key in keys})} 折 × "
        f"{len({key.seed for key in keys})} 种子；配置 {config.path.name}"
    )
    if dry_run:
        for key in keys:
            print(f"  {key.run_id()}")
        print("dry-run：没有训练，也没有写 CSV")
        return 0

    if not index_only:
        records = C.run_arm(config, tasks=tasks, seeds=seeds, n_levels=n_levels)
        print(f"本次写出 {len(records)} 条记录（其余命中 resume）")

    summary = G.write_n_index(config)
    print(
        f"索引已刷新：{summary['n_runs']} 条 run → "
        f"{Path(summary['path']).relative_to(REPO_ROOT).as_posix()}；"
        f"子集 → {Path(summary['n_index_path']).relative_to(REPO_ROOT).as_posix()}"
    )

    ours = _openpoly_subset_map(Path(summary["n_index_path"]))
    theirs = _openpoly_subset_map(OPENPOLY_GPU_N_INDEX)
    if not theirs:
        print(f"对账跳过：{OPENPOLY_GPU_N_INDEX.name} 不在（B-5 的索引未入库）")
        return 0
    shared = sorted(set(ours) & set(theirs))
    missing = sorted(set(theirs) - set(ours))
    mismatched = [key for key in shared if ours[key] != theirs[key]]
    print(
        f"subset_sha256 对账：共有 {len(shared)} 个 (task, fold, 档位)，"
        f"一致 {len(shared) - len(mismatched)}，不一致 {len(mismatched)}，"
        f"B 有而我缺 {len(missing)}"
    )
    for key in mismatched[:10]:
        print(f"  ✗ {key}：本侧 {ours[key]} vs B 侧 {theirs[key]}")
    for key in missing[:10]:
        print(f"  … {key} 还没有跑（B 侧哈希 {theirs[key][:21]}…）")
    if mismatched:
        print("对账失败：训练行不是同一批，H4 的交叉点不能用这批数——先停下查子集规则")
        return 1
    print("对账通过：五档 × 五折 × 三性质的训练行与 B-5 的 GNN 逐档相同")
    return 0


# ---- 6. 数据量扫描的编排 / Orchestration of the scaling scan ----
def _scaling_search(config_path: str | None, dataset_id: str) -> int:
    """按 A2 的预注册协议给 `pb26_scaling_cp` 选一次超参，写进 `a2_best_params.csv` 的一行。
    Run the pre-registered A2 search once for pb26_scaling_cp and append one row.

    中文：
        用途：`a2_best_params.csv` 是按设计只覆盖 8 个主表任务（分工_C-a），而 `train_one()` 缺行
              会**报错而不是回退默认值**。于是这里用**同一条协议**（Murcko 分组 5 折、只在第一折
              训练行内、候选 1,944、固定 `search_seed=20261113`、≤30 trials）补一行，之后冻结给
              两个数据集、所有档位、所有折与种子（D-51）。
              用哪个数据集的池子选参必须写死：默认交替 Cp 的 `fold_00`（22,242 行，比均聚的 1,440
              行更能代表"这套特征上什么超参合适"）；**选了之后不许按档位重搜**（档位是训练集大小，
              不是超参）。
        输入：config_path；dataset_id —— 用哪个数据集选参。
        输出：0 成功。数据来源：该数据集的 pack 与官方五折。副作用：写 `a2_best_params.csv` 一行。
    English:
        a2_best_params.csv covers the eight main-table tasks by design, and train_one() raises (rather
        than falling back to defaults) when a row is missing. So this fills in one row with the *same*
        protocol, then freezes it for both datasets, every tier, fold and seed (D-51). Which pool does
        the choosing is pinned: the alternating dataset's fold_00 (22,242 rows). No per-tier re-search
        - a tier is a training-set size, not a hyper-parameter.
    """
    from chainworth.experiments import scaling as S

    # 搜索用的是 **A2 的冻结配置**（A2 的 `search` 段），不是 scaling.yaml：
    # 空间、trials、search_seed 只在一处定义，实验配置里再抄一份就有第二个口径（D-51）。
    # The search runs under **A2's frozen config** (its `search` section), not scaling.yaml:
    # space, trials and search_seed are defined in exactly one place, and copying them into the
    # experiment config would create a second convention (D-51).
    search_config = G.load_arm_config(ARM_CONFIGS["a2_conventional"])
    config = G.load_arm_config(Path(config_path) if config_path else SCALING_CONFIG)
    data = S.build_task_data(dataset_id)
    print(
        f"=== search pb26_scaling_cp [xgboost] on {dataset_id}/{data.split.fold_names[0]} "
        f"（{int(data.split.folds[0].train_index.size)} 行训练池；协议来自 "
        f"{search_config.path.name}，产出记录用 {config.path.name}） ===",
        flush=True,
    )
    best = C.search_task(
        data, search_config, learner="xgboost", log=lambda message: print(message, flush=True)
    )
    print(
        f"pb26_scaling_cp [xgboost] cv_r2={best['cv_score']:.4f} trials={best['trials']} "
        f"folds={best['cv_folds']} rows={best['n_search_rows']} "
        f"params={json.dumps(best['params'], sort_keys=True)}"
    )
    print("已写进 results/aggregate/a2_best_params.csv（按 (task, learner) 覆盖，8 行老记录未动）")
    return 0


def _scaling(
    config_path: str | None,
    datasets: Sequence[str] | None,
    tiers_wanted: Sequence[str] | None,
    folds: Sequence[str] | None,
    learners: Sequence[str] | None,
    *,
    build: bool,
    build_only: bool,
    force: bool,
    table_only: bool,
    dry_run: bool,
    workers: int = 0,
    search: bool = False,
    search_dataset: str = "md5000_alternating_cp",
) -> int:
    """C-b-1：造两份 pack → 80 fits → 写 `scaling_cpu.csv`。
    C-b-1: build the two packs, run the 80 fits, write scaling_cpu.csv.

    中文：
        用途：把"数据变多、误差会不会变小"变成一张能算 κ 的表。三件事按顺序做，缺一步都不许往下：
              ① `--build`：造/核两个数据集的特征 pack（`canonical_sha256` 逐位核对）；
              ② 拟合：数据集 × 官方档位 × 官方五折 × 学习器；XGBoost 落 run 记录，
                 RF 只进表（理由见 `experiments/scaling.py` 的 `run_scaling()`）；
              ③ 写表：**按 (dataset, tier, fold, learner, seed) 合并**，所以分几次跑也不会丢行。
        输入：config_path；datasets / tiers_wanted / folds / learners —— 白名单；
              build / build_only / force / table_only / dry_run。
        输出：0 成功。数据来源：pack + `splits/`。配置来源：`configs/experiments/scaling.yaml`。
        副作用：写 npz、记录、预测、`gpu_hours.csv`、`scaling_cpu.csv`。
    English:
        Turns "does more data shrink the error" into a table kappa can be fitted on. Three steps in
        order, none skippable: build/verify the two packs (canonical hash checked bit for bit), run
        the grid (XGBoost writes records, the forest only feeds the table), then write the table -
        **merged by (dataset, tier, fold, learner, seed)** so partial runs never lose rows.
    """
    from chainworth.experiments import scaling as S

    config = G.load_arm_config(Path(config_path) if config_path else SCALING_CONFIG)
    chosen_datasets = list(datasets) if datasets else S.dataset_ids()

    if search:
        if dry_run:
            print(f"[dry-run] 会先跑 search（数据集 {search_dataset}，30 trials × 5 折）")
        else:
            _scaling_search(config_path, search_dataset)

    if build or build_only:
        for dataset_id in chosen_datasets:
            info = S.build_pack(dataset_id, force=force, workers=workers)
            print(
                f"pack {dataset_id}：{info['n_rows']} 行、唯一结构 {info.get('n_unique_structures', '?')}、"
                f"pack_sha256={str(info['pack_sha256'])[:12]}…（{'重造' if info.get('rebuilt') else '沿用'}）"
            )
        if build_only:
            return 0

    chosen_learners = list(learners) if learners else [
        str(value) for value in (config.raw.get("learners") or ["xgboost", "rf"])
    ]
    plan: list[tuple[str, str]] = []
    for dataset_id in chosen_datasets:
        available = S.tiers(dataset_id)
        if tiers_wanted:
            unknown = sorted({str(value).lower() for value in tiers_wanted} - set(available))
            if unknown:
                raise SystemExit(f"{dataset_id} 没有这些档位 {unknown}：官方文件里的是 {available}")
        for level in available:
            if tiers_wanted and level not in {str(value).lower() for value in tiers_wanted}:
                continue
            plan.append((dataset_id, level))
    n_folds = len(folds) if folds else len(
        (config.raw.get("runs") or {}).get("polybench26", {}).get("folds", [])
    )
    expected = len(plan) * n_folds * len(chosen_learners)
    print(
        f"scaling：{len(chosen_datasets)} 个数据集 × {len(plan)} 个(数据集,档位) × {n_folds} 折 × "
        f"{len(chosen_learners)} 学习器（{', '.join(chosen_learners)}）= {expected} fits；"
        f"配置 {config.path.name}（expected_fits={config.raw.get('expected_fits', '未写')}）"
    )
    if dry_run:
        for dataset_id, level in plan:
            print(f"  {dataset_id} tier={level}")
        print("dry-run：没有训练，也没有写表")
        return 0

    if table_only:
        rows = _scaling_rows_from_disk(chosen_datasets, chosen_learners)
    else:
        rows = S.run_scaling(
            config,
            datasets=chosen_datasets,
            tiers_wanted=tiers_wanted,
            folds=folds,
            learners=chosen_learners,
        )
    merged: dict[tuple[str, str, str, str, str], dict[str, Any]] = {}
    for row in S.read_table():
        merged[_scaling_key(row)] = row
    for row in rows:
        merged[_scaling_key(row)] = row
    path = S.write_table(list(merged.values()))
    counts: dict[tuple[str, str], int] = {}
    for row in merged.values():
        key = (str(row.get("learner", "")), str(row.get("dataset", "")))
        counts[key] = counts.get(key, 0) + 1
    print(f"写出 {len(merged)} 行 → {path.relative_to(REPO_ROOT).as_posix()}")
    for (learner, dataset_id), count in sorted(counts.items()):
        print(f"  {dataset_id:<26} [{learner:<7}] {count} 行")
    per_learner = {}
    for row in merged.values():
        metrics = row.get("r2")
        if metrics in ("", None):
            continue
        per_learner.setdefault(str(row["learner"]), []).append(float(metrics))
    for learner, values in sorted(per_learner.items()):
        print(
            f"  {learner}：{len(values)} 行，R² 中位 {float(np.median(values)):+.4f}，"
            f"最好 {max(values):+.4f}，最差 {min(values):+.4f}"
        )
    return 0


def _scaling_key(row: Mapping[str, Any]) -> tuple[str, str, str, str, str]:
    """`scaling_cpu.csv` 的合并键。／ The merge key of one scaling_cpu.csv row."""
    return (
        str(row.get("dataset", "")),
        str(row.get("tier", "")),
        str(row.get("fold", "")),
        str(row.get("learner", "")),
        str(row.get("seed", "")),
    )


def _scaling_rows_from_disk(
    datasets: Sequence[str], learners: Sequence[str]
) -> list[dict[str, Any]]:
    """按磁盘上已有的 `raw_runs` 记录重建 XGBoost 那一侧的表行（`--table-only`）。
    Rebuild the XGBoost rows of the table from the records already on disk (`--table-only`).

    中文：用途：`--table-only` 不训练，所以只可能重建**有 run 记录**的那一半（XGBoost）。
          RF 的行没有记录可依，重写表时会被保留在旧表里（合并键相同），不会被抹掉。
    English: only the half with run records can be rebuilt (XGBoost). The forest's rows have no
    records, but the merge keeps them in place.
    """
    from chainworth.experiments import scaling as S

    wanted = set(datasets)
    rows: list[dict[str, Any]] = []
    for path in sorted(G.RAW_RUNS_DIR.glob("*.json")):
        if path.name == "_schema.json":
            continue
        try:
            record = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:                    # pragma: no cover - 半截文件
            continue
        if record.get("task") != S.TASK_ID or record.get("n_level") is None:
            continue
        dataset_id = str(record.get("dataset_id") or "")
        if dataset_id not in wanted or record.get("status") != "ok":
            continue
        if "xgboost" not in learners:
            continue
        rows.append(S._row_from_record(record, dataset_id, "xgboost"))
    return rows


# ---- 7. 架构迁移的编排 / Orchestration of the architecture transfer ----
def _arch(
    config_path: str | None,
    *,
    setup: bool,
    verify: bool,
    search: bool,
    ood: bool,
    table_only: bool,
    dry_run: bool,
    resume: bool,
    folds: Sequence[str] | None,
    seeds: Sequence[int] | None,
    workers: int,
) -> int:
    """C-c-1/C-c-2：setup → search → 15 runs → OOD 三段式 → 集成点值 → arch_transfer.csv。
    C-c-1/C-c-2: setup -> search -> 15 runs -> the OOD three stages -> ensemble -> the table.

    中文：
        用途：按分工_C-c 的顺序把六件事编排在一个子命令里，每件都可单独跑：
              ① `--setup`：划分落位 `splits/polybench26/arch/seed_42/`（polybench26 格式 +
                 PROVENANCE）+ 造 ID 特征（6,478 × 2,253，与组分表同一套列）；
              ② `--search`：D-51 预注册协议搜一次超参（A2 冻结配置的 search 段），写
                 `a2_best_params.csv` 第 13 行——**在 15 runs 之前**，`train_one()` 缺行会报错；
              ③ 15 runs：复用 `C.train_one()`，超参来自第②步，`config_hash` = 本配置字节哈希，
                 所以**本文件必须冻结成 v1 之后才能跑**；
              ④ `--ood`：覆盖断言 → 逐组分预测（重拟合须与落盘 ID 预测逐位一致）→
                 官方 `summarize_weighted_predictions.py` 合成 → 每种子五折集成点值；
              ⑤ `arch_transfer.csv`：一 run 一行、ID 两档分开列、OOD 逐折留档；
              ⑥ 排序核对：block > random > alternating > homopolymer 是否成立，照实打印。
        输入：各白名单与开关。输出：0 成功。数据来源：generated/ + ID/组分特征 + splits/。
        副作用：见各步。异常：任一验收不过即抛。
    English:
        The six C-c steps in one subcommand, each runnable alone. The 15 runs must come after the
        search (train_one() raises on a missing row) and after this config is frozen to v1, because
        its byte hash is every record's config_hash.
    """
    from chainworth.experiments import arch as A

    config = G.load_arm_config(Path(config_path) if config_path else ARCH_CONFIG)
    if setup:
        if verify:
            problems = A.verify_arch_setup()
            print(
                f"复核：{'通过' if problems['ok'] else '不过'}——"
                f"canonical {problems['canonical']['rows']}、holdout {problems['holdout']['rows']}、"
                f"组分覆盖 {problems['components']['complete']}"
            )
            for problem in problems["problems"]:
                print(f"  ✗ {problem}")
            return 0 if problems["ok"] else 1
        splits = A.materialize_splits()
        print(
            f"划分落位：{A.ARCH_SPLIT_MANIFEST.relative_to(REPO_ROOT).as_posix()} "
            f"（train {splits['train_rows']} / val {splits['val_rows']} / test 648×5）"
        )
        features = A.build_id_features(workers=workers)
        print(
            f"ID 特征：{features['n_rows']} 行 × {features['n_descriptors'] + features['n_fingerprint_bits']} 列，"
            f"NaN 单元 {features['n_nan_cells']}，sha256={features['features_sha256'][:12]}…"
        )
        return 0

    if search:
        if dry_run:
            print("[dry-run] 会按 D-51 协议在 fold_00 训练行内搜 30 trials（Murcko 分组 5 折）")
            return 0
        search_config = G.load_arm_config(ARM_CONFIGS["a2_conventional"])
        data = A.load_id_data()
        print(
            f"=== search {A.TASK_ID} [xgboost] on {data.split.fold_names[0]} "
            f"（{int(data.split.folds[0].train_index.size)} 行训练池；协议来自 "
            f"{search_config.path.name}，产出记录用 {config.path.name}） ===",
            flush=True,
        )
        best = C.search_task(
            data, search_config, learner="xgboost", log=lambda message: print(message, flush=True)
        )
        print(
            f"{A.TASK_ID} [xgboost] cv_r2={best['cv_score']:.4f} trials={best['trials']} "
            f"folds={best['cv_folds']} rows={best['n_search_rows']} "
            f"params={json.dumps(best['params'], sort_keys=True)}"
        )
        print("已写进 results/aggregate/a2_best_params.csv（按 (task, learner) 覆盖，12 行老记录未动）")
        return 0

    if verify:
        problems = A.verify_arch_setup()
        print(
            f"复核：{'通过' if problems['ok'] else '不过'}——"
            f"canonical {problems['canonical']['rows']}、holdout {problems['holdout']['rows']}、"
            f"组分覆盖 {problems['components']['complete']}"
        )
        for problem in problems["problems"]:
            print(f"  ✗ {problem}")
        return 0 if problems["ok"] else 1

    if table_only:
        records = _arch_records_on_disk()
        ood_by_run = _arch_ood_from_disk()
        path = A.write_arch_table(records, ood_by_run=ood_by_run)
        print(f"arch_transfer.csv 已重写：{len(records)} 行 → {path.relative_to(REPO_ROOT).as_posix()}")
        _arch_print_ordering(records, ood_by_run)
        return 0

    if dry_run:
        fold_names = [entry["name"] for entry in A._split_fold_names()]
        seeds_wanted = [int(s) for s in (seeds or config.raw.get("seeds") or [0])]
        for fold in fold_names:
            for seed in seeds_wanted:
                run_id = G.RunKey(
                    task=A.TASK_ID, benchmark="polybench26", fold=fold, seed=seed, arm=A.ARM
                ).run_id()
                print(run_id)
        print(f"dry-run：{len(fold_names) * len(seeds_wanted)} 条 run，没有训练")
        return 0

    records = A.run_arch(config, folds=folds, seeds=seeds, resume=resume)
    if not records:
        print("没有新记录（全部命中 resume）")
    else:
        print(f"写出/命中 {len(records)} 条 run 记录")

    ood_summaries: list[dict[str, Any]] = []
    if ood:
        ood_summaries = A.run_ood(config, folds=folds, seeds=seeds, resume=resume)
        ensemble = A.ensemble_ood(config, seeds=seeds)
        print("五折集成点值（D-09：不做 t 检验，不给均值 ± SE）：")
        for seed_key, payload in ensemble["seeds"].items():
            points = payload["ensemble_point_values"]
            print(
                f"  seed {seed_key}：random {points['random']['rmse']:.4f} / "
                f"block {points['block']['rmse']:.4f} / combined {points['combined']['rmse']:.4f}"
            )

    ood_by_run = _arch_ood_from_disk() if ood_summaries else None
    path = A.write_arch_table(records, ood_by_run=ood_by_run)
    print(f"arch_transfer.csv：{len(records)} 行 → {path.relative_to(REPO_ROOT).as_posix()}")
    _arch_print_ordering(records, ood_by_run or {})
    return 0


def _arch_records_on_disk() -> list[dict[str, Any]]:
    """磁盘上 arch 的 15 条记录（按 run_id 排序）。／ The arch records on disk, sorted by run_id."""
    from chainworth.experiments import arch as A

    records: list[dict[str, Any]] = []
    for path in sorted(G.RAW_RUNS_DIR.glob("*.json")):
        if path.name == "_schema.json":
            continue
        try:
            record = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:                    # pragma: no cover - 半截文件
            continue
        if record.get("task") == A.TASK_ID and record.get("status") == "ok":
            records.append(record)
    return records


def _arch_ood_from_disk() -> dict[str, dict[str, Any]]:
    """磁盘上逐 run 的 OOD 指标（metrics json → dict）。／ Per-run OOD metrics read back."""
    from chainworth.experiments import arch as A

    result: dict[str, dict[str, Any]] = {}
    for path in sorted(A.OOD_DIR.glob("*_metrics.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        run_id = path.name[: -len("_metrics.json")]
        result[run_id] = {"metrics": {entry["architecture"]: entry for entry in payload["metrics"]}}
    return result


def _arch_print_ordering(
    records: Sequence[Mapping[str, Any]], ood_by_run: Mapping[str, Mapping[str, Any]]
) -> None:
    """报告实际排序：block > random > alternating > homopolymer 是否成立（照实写，不重调）。
    Report the actual ordering; state it plainly instead of re-tuning when it does not hold.
    """
    import numpy as np

    from chainworth.experiments import arch as A

    id_alt = [r["id_tier_metrics"]["alternating"]["rmse"] for r in records if r.get("id_tier_metrics")]
    id_homo = [r["id_tier_metrics"]["homopolymer"]["rmse"] for r in records if r.get("id_tier_metrics")]
    ood_random = [
        v["metrics"]["random"]["rmse"] for v in ood_by_run.values() if "random" in v.get("metrics", {})
    ]
    ood_block = [
        v["metrics"]["block"]["rmse"] for v in ood_by_run.values() if "block" in v.get("metrics", {})
    ]
    alt = float(np.mean(id_alt)) if id_alt else float("nan")
    homo = float(np.mean(id_homo)) if id_homo else float("nan")
    random_ = float(np.mean(ood_random)) if ood_random else float("nan")
    block = float(np.mean(ood_block)) if ood_block else float("nan")
    print(
        f"实际排序（mean RMSE；OOD 为逐折、ID 为 15 run 均值，跨域只作陈述不作判决）："
        f"block {block:.4f} / random {random_:.4f} / alternating {alt:.4f} / homopolymer {homo:.4f}"
    )
    if not np.isfinite([alt, homo, random_, block]).all():
        print("  （OOD 尚未合成，排序判决待 arch --ood 之后再说）")
        return
    holds = bool(block > random_ > alt > homo)
    print(
        "  → " + ("成立" if holds else "不成立——按分工_C-c 验收⑤照实登记 DECISIONS.md，不重调超参")
    )


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
        # 只收**没有 n 档**的记录：`a1_a2_index.csv` 是 C-c 填主表三列的唯一入口（96 条），
        # C-b 的 scaling / OpenPoly 记录（带 `n_level`）另有两张表，混进来会让主表多出 285 行。
        # Only records **without** an n level: a1_a2_index.csv is C-c's single entry for the main
        # table's three columns (96 rows); C-b's scaling and OpenPoly records carry `n_level` and
        # have their own two tables, and mixing them in would add 285 rows to the main table.
        records = [
            record
            for record in _records_on_disk(("a1_floor", "a2_conventional"))
            if record.get("n_level") is None
        ]
        if args.dry_run:
            print(f"磁盘上 A1/A2 的 ok 记录 {len(records)} 条（不含 n 档记录）")
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

    if args.command == "openpoly":
        return _openpoly(
            args.config,
            args.tasks,
            args.seeds or [],
            args.n_levels,
            dry_run=args.dry_run,
            index_only=args.index_only,
        )

    if args.command == "scaling":
        return _scaling(
            args.config,
            args.datasets,
            args.tiers,
            args.folds,
            args.learners,
            build=args.build,
            build_only=args.build_only,
            force=args.force,
            table_only=args.table_only,
            dry_run=args.dry_run,
            workers=args.workers,
            search=args.search,
            search_dataset=args.search_dataset,
        )

    if args.command == "arch":
        return _arch(
            args.config,
            setup=args.setup,
            verify=args.verify,
            search=args.search,
            ood=args.ood,
            table_only=args.table_only,
            dry_run=args.dry_run,
            resume=not args.no_resume,
            folds=args.folds,
            seeds=args.seeds,
            workers=args.workers,
        )

    raise SystemExit(f"未知子命令 {args.command!r}")


if __name__ == "__main__":                             # pragma: no cover - 命令行入口
    raise SystemExit(main())
