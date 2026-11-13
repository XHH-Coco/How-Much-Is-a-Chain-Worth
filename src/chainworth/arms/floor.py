"""A1 地板：什么都不学，用训练集均值猜——没有它，后面所有提升都没有参照。
A1 floor: learn nothing and predict the training mean; without it no improvement has a reference.

中文：
    用途：主表的第一列。它回答"瞎猜有多差"，也是 Λ 的分母里那个"重复单元都解释不了"的基线的
          下限（A2 才是真正的 unit 臂；A1 只证明指标尺度没有异常）。
    输入：只要标签与官方划分，**完全不读特征**——所以它也是特征流水线出问题时的对照：
          A1 的数随特征变化，说明有东西漏进了标签。
    学习器：`dummy_mean`（主表）。kNN 地板需要"在什么空间里量距离"这个定义，而 A1 的输入是空矩阵，
          距离恒为 0，跑出来与均值等价——是否要给它一个定义，是 C 组要一起定的事（见 yaml 的
          `open_questions`），本模块不擅自决定。
    依赖：numpy / pandas / pyarrow / PyYAML。
    配置：`configs/arms/a1_floor.yaml`。
    对外：`FloorData`、`build_task_data()`、`train_one()`、`run_arm()`。
    被谁用：`scripts/04_run_cpu.py`、`tests/test_arms_classical.py`。

English:
    Purpose: the first column of the main table - how bad guessing is. It reads labels and the
    official split only, never the features, so it doubles as a control: if the A1 number moves
    when features change, something leaked into the labels.
    Learner: `dummy_mean` for the main table. A kNN floor needs a definition of "distance in what
    space" while A1's input is empty, where every distance is zero and the result equals the mean;
    whether to give it one is a team decision recorded in the yaml's `open_questions`.
    Depends on: numpy, pandas, pyarrow, PyYAML. Config: configs/arms/a1_floor.yaml.
    Public API: FloorData, build_task_data, train_one, run_arm.
"""

from __future__ import annotations

import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from chainworth.arms import graph as G
from chainworth.arms import conventional as C
from chainworth.io import loaders as L

ARM = "a1_floor"
ENV_FINGERPRINT = "results/env_fingerprints/env_core.json"
DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[3] / "configs" / "arms" / "a1_floor.yaml"


@dataclass
class FloorData:
    """地板臂要的东西：任务、标签、行号、官方划分。／ What the floor needs: task, labels, rows, split."""
    task_id: str
    benchmark: str
    y: np.ndarray
    row_id: np.ndarray
    split: L.SplitSpec
    limit: int | None = None


def build_task_data(task_id: str, *, limit: int | None = None) -> FloorData:
    """只读标签与划分，不读特征矩阵。／ Read labels and the split; the feature matrix is not read.

    中文：输入：task_id；limit —— 冒烟截断。输出：`FloorData`。
          数据来源：`data/processed/features/<task>.npz`（只取 `y` / `row_id`）与 `splits/`。
          副作用：只读。异常：G.ArmError —— 缺产物、标签非有限、`limit` 太小。
    English: Args: task_id, limit. Returns: FloorData. Data: the pack's y and row_id, plus splits/.
    """
    from chainworth.chem import features as F

    pack_path = L.feature_pack_path(task_id)
    if not pack_path.is_file():
        raise G.ArmError(f"缺装配产物 {pack_path}：先跑 `python scripts/03b_assemble.py`（B-1）")
    pack = F.load_pack(task_id)
    y = np.asarray(pack["y"], dtype=np.float64)
    bad = ~np.isfinite(y)
    if bad.any():
        raise G.ArmError(f"{task_id} 有 {int(bad.sum())} 行标签不是有限值：地板也建不了")
    row_id = np.asarray(pack["row_id"], dtype=np.int64)
    split = L.load_split(task_id)
    if limit is not None and limit < y.size:
        split = L.limit_split(split, limit)
        y, row_id = y[:limit], row_id[:limit]
    else:
        limit = None
    return FloorData(
        task_id=task_id,
        benchmark=str(pack["meta"].get("benchmark", split.benchmark)),
        y=y,
        row_id=row_id,
        split=split,
        limit=limit,
    )


def train_one(
    data: FloorData,
    config: G.ArmConfig,
    *,
    fold: str,
    seed: int,
    device: str | None = None,
    max_epochs: int | None = None,
    patience: int | None = None,
    n_level: str | None = None,
    **_: Any,
) -> C.TaskOutcome:
    """用训练集均值预测测试集；三种子的数**完全相同**（均值没有随机性），记录里如实反映。
    Predict the training mean; the three seeds give identical numbers and the record says so.

    中文：输入：data；config；fold；seed。输出：`C.TaskOutcome`。
          数据来源：`FloorData`。副作用：无（不训练）。异常：G.ArmError —— 折名不存在或集合为空。
    English: Args: data, config, fold, seed. Returns: an outcome with the training mean as its
    prediction. Raises on an unknown fold or an empty train/test set.
    """
    by_name = {spec.name: spec for spec in data.split.folds}
    if fold not in by_name:
        raise G.ArmError(f"{data.task_id} 没有折 {fold!r}：官方折名是 {sorted(by_name)}")
    spec = by_name[fold]
    train_index = np.asarray(spec.train_index)
    test_index = np.asarray(spec.test_index)
    if train_index.size == 0 or test_index.size == 0:
        raise G.ArmError(f"{data.task_id}/{fold} 的训练集或测试集为空：划分有问题，先别跑")

    started = datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")
    clock = time.perf_counter()
    mean = float(np.mean(data.y[train_index]))
    y_pred = np.full(test_index.size, mean, dtype=np.float64)
    wall = time.perf_counter() - clock
    finished = datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")

    split_version, _ = G.fold_version(data.split, fold)
    learner = str(config.section("learner").get("main", "dummy_mean"))
    run_id = G.RunKey(
        task=data.task_id, benchmark=data.benchmark, fold=fold, seed=seed, arm=ARM
    ).run_id()
    record: dict[str, Any] = {
        "run_id": run_id,
        "status": "ok",
        "benchmark": data.benchmark,
        "task": data.task_id,
        "arm": ARM,
        "learner": learner,
        "seed": seed,
        "split_version": split_version,
        "split_fold": fold,
        "config_hash": G.config_hash(config),
        "n_train": int(train_index.size),
        "n_test": int(test_index.size),
        "metrics": G.metrics(data.y[test_index], y_pred),
        "wall_clock_s": wall,
        "device": "cpu",
        "env_fingerprint": ENV_FINGERPRINT,
        "predictions": f"results/aggregate/predictions/{run_id}.parquet",
        "train_mean": mean,
        "started_at": started,
        "finished_at": finished,
        "notes": (
            "floor arm: constant training-mean prediction; no random source, seeds identical; "
            f"interpreter Python {sys.version.split()[0]} (D-39)"
        ),
    }
    return C.TaskOutcome(record=record, row_id=data.row_id[test_index], y_pred=y_pred)


def run_arm(
    config: G.ArmConfig,
    *,
    tasks: Sequence[str] | None = None,
    seeds: Sequence[int] | None = None,
) -> list[dict[str, Any]]:
    """按配置网格跑一个任务子集，复用 `graph.run_grid` 的 resume 与落盘。
    Run a task subset on the grid, reusing `graph.run_grid`'s resume and writers.
    """
    return G.run_grid(config, tasks=tasks, seeds=seeds, builder=build_task_data, trainer=train_one)
