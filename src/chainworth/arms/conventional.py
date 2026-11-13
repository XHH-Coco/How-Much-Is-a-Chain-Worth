"""A2 经典臂：205 个 RDKit 描述符 + 2048 位指纹，XGBoost（主表）/ 随机森林（敏感性）。
A2 conventional arm: 205 RDKit descriptors plus 2048 fingerprint bits, XGBoost (main table) or a
random forest (sensitivity).

中文：
    用途：A2 是全队的"标准秤"——只给**重复单元**的描述符，不碰链的信息。它同时是 A5 的母版：
          A5 = A2 的输入 + 链列、学习器与超参**完全不变**，所以本模块的 `train_one()` 也被
          `arms/chain_info.py` 复用，只换输入矩阵。
    输入从哪来：`data/processed/features/<task>.npz`（`chem.features.load_pack()`），
          先用 `data/manifests/task_features_manifest.csv` 里的 `pack_sha256` 核一遍再读；
          核不过就停（分工_C §7 常见坑：用了别人改过的特征，全表作废）。
    划分从哪来：`splits/` 的官方索引，只读。任何 `train_test_split` 都不许出现在评估路径上。
    调参：Murcko 骨架分组的 5 折 CV，**只在训练行内**，预算 ≤30 trials（写在本模块读的 yaml 里）。
    依赖：numpy / pandas / pyarrow / scikit-learn / xgboost / rdkit / PyYAML
          （`environment/core.yml`；本机用同版本的 venv 跑，见 docs/DECISIONS 的 D-40）。
    配置：`configs/arms/a2_conventional.yaml`（冻结后它的字节哈希就是 `config_hash`）。
    对外：`TaskData`、`build_task_data()`、`murcko_groups()`、`search_task()`、`train_one()`、
          `run_arm()`、`write_index()`、`TaskOutcome`。
    被谁用：`scripts/04_run_cpu.py`、`tests/test_arms_classical.py`。

English:
    Purpose: A2 is the team's standard scale - repeat-unit descriptors only, no chain information -
    and it is also A5's template: A5 adds chain columns while keeping the learner and its
    hyper-parameters identical, so `train_one()` here is reused by `arms/chain_info.py` with a
    different input matrix.
    Data: `data/processed/features/<task>.npz` through `chem.features.load_pack()`, after checking
    `pack_sha256` against `data/manifests/task_features_manifest.csv`.
    Splits: the read-only official indices under `splits/`; no `train_test_split` on the evaluation
    path. Tuning: Murcko-grouped 5-fold CV **inside the training rows only**, at most 30 trials.
    Depends on: numpy, pandas, pyarrow, scikit-learn, xgboost, rdkit, PyYAML.
    Config: `configs/arms/a2_conventional.yaml`; its byte hash is `config_hash` once frozen.
    Public API: TaskData, build_task_data, murcko_groups, search_task, train_one, run_arm,
    write_index, TaskOutcome. Callers: scripts/04_run_cpu.py, tests/test_arms_classical.py.

三条口径 ／ Three conventions
-----------------------------
1. **NaN 不填全局值。** 205 个描述符里有缺失（`pb_t1_tg` 有 3.9% 的格子）。XGBoost 原生接受
   NaN，直接交给它；随机森林 / kNN 走"只在训练行上 fit 的中位数填充"。
   XGBoost takes NaN natively; the forest and kNN get a median imputer fitted on training rows only.
2. **确定性三条腿。** `n_jobs=1`、`random_state=seed`、`tree_method="hist"`；同 seed 两遍的
   `r2` 必须逐位相同（差 < 1e-9）。`n_jobs` 是随机源这件事分工_C §2 已点名。
   Determinism: one job, a fixed seed, the hist tree method; same seed twice must be bit-identical.
3. **主表一臂一个学习器。** A2 主表只放 XGBoost；随机森林与 kNN 的数是"学习器敏感性"，
   另存一张表、不进主表（分工_C §2 的口径框，96 条 = 16 单元 × 2 臂 × 3 种子；A5 撤除见 D-48）。
   One learner per arm in the main table: A2 contributes XGBoost only; the forest and kNN feed a
   separate sensitivity table.
"""

from __future__ import annotations

import csv
import hashlib
import itertools
import json
import logging
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from chainworth.arms import graph as G
from chainworth.io import loaders as L

LOGGER = logging.getLogger("chainworth.arms.conventional")

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CONFIG_PATH = REPO_ROOT / "configs" / "arms" / "a2_conventional.yaml"
FEATURES_MANIFEST = REPO_ROOT / "data" / "manifests" / "task_features_manifest.csv"
BEST_PARAMS_PATH = REPO_ROOT / "results" / "aggregate" / "a2_best_params.csv"
INDEX_PATH = REPO_ROOT / "results" / "aggregate" / "a1_a2_index.csv"
ENV_FINGERPRINT = "results/env_fingerprints/env_core.json"

ARM = "a2_conventional"
BEST_PARAMS_COLUMNS = ("task", "learner", "best_params", "cv_score", "trials")
INDEX_COLUMNS = (
    "run_id",
    "task",
    "arm",
    "learner",
    "fold",
    "seed",
    "final_r2",
    "n_train",
    "n_test",
)
# 搜索空间里每个维度的固定顺序：试次编号 = 笛卡尔积里的行号，顺序写死才能复算"第 7 次试的是什么"。
# A fixed dimension order: a trial number is a row in the Cartesian product, so "what was trial 7"
# is reproducible only if the order is pinned here.
SEARCH_ORDER = (
    "n_estimators",
    "max_depth",
    "learning_rate",
    "subsample",
    "colsample_bytree",
    "min_child_weight",
    "reg_lambda",
)


# ---- 1. 任务数据 / Task data ----


@dataclass
class TaskData:
    """一个任务的输入矩阵、标签、结构串与官方划分。
    One task's input matrix, labels, psmiles strings and official split.

    中文：
        字段：`task_id`、`benchmark`、`X`（float32，列 = 205 描述符 + 2048 指纹）、`y`、`row_id`、
              `psmiles`、`split`（`SplitSpec`）、`pack_sha256`（核过的装配产物哈希）、
              `limit`（冒烟截断，None = 全量）。
        数据来源：`data/processed/features/` 与 `splits/`。副作用：只读。
        English: X is float32 with 205 descriptor columns followed by 2048 fingerprint bits; row
        order is the task row position, never re-sorted.
    """

    task_id: str
    benchmark: str
    X: np.ndarray
    y: np.ndarray
    row_id: np.ndarray
    psmiles: np.ndarray
    split: L.SplitSpec
    pack_sha256: str
    limit: int | None = None

    @property
    def n_rows(self) -> int:
        """行数。／ Number of rows."""
        return int(self.X.shape[0])


def _sha256_of_file(path: Path) -> str:
    """文件字节的 sha256（分块读）。／ sha256 of the file bytes, read in chunks."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def manifest_pack_sha256(task_id: str, *, path: Path | None = None) -> str:
    """从装配清单里取 `pack_sha256`；取不到就报错，不猜。
    Read `pack_sha256` from the assembly manifest; a missing row raises instead of guessing.

    中文：输入：task_id；path —— 清单路径（默认 `data/manifests/task_features_manifest.csv`）。
          输出：64 位十六进制。数据来源：清单 CSV。副作用：只读。异常：G.ArmError —— 没有这一行。
    English: Args: task_id, path. Returns: the 64-hex digest. Raises when the task has no row.
    """
    target = Path(path) if path is not None else FEATURES_MANIFEST
    if not target.is_file():
        raise G.ArmError(f"缺装配清单 {target}：先跑 `python scripts/03b_assemble.py`（B-1）")
    with target.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            if row.get("task_id") == task_id:
                return str(row.get("pack_sha256", ""))
    raise G.ArmError(f"装配清单里没有 {task_id}：B-1 没为它产出特征包")


def build_task_data(task_id: str, *, limit: int | None = None) -> TaskData:
    """读一个任务的输入矩阵：核哈希 → 读装配产物 → 拼 `[描述符 | 指纹]` → 取官方划分。
    Build one task's input: verify the pack hash, load it, concatenate descriptors and bits, then
    take the official split.

    中文：
        用途：A2/A5 共用的唯一数据入口。**行序不重排**：`row_id` 就是任务行位置，划分里的
              `train_index` / `test_index` 是同一坐标系，按行序切。
        输入：task_id；limit —— 冒烟只取前 N 行（调用方负责把产物写成 `*.partial.*`）。
        输出：`TaskData`。数据来源：`data/processed/features/<task>.npz`、`splits/`。
        副作用：只读。异常：G.ArmError —— 哈希不符、缺产物、标签有非有限值。
    English:
        The single data entry point for A2 and A5. Row order is never re-sorted: `row_id` is the
        task row position and the split indices live in that same coordinate system.
        Raises: ArmError on a hash mismatch, a missing pack, or non-finite labels.
    """
    from chainworth.chem import features as F

    pack_path = L.feature_pack_path(task_id)
    if not pack_path.is_file():
        raise G.ArmError(
            f"缺装配产物 {pack_path}：先跑 `python scripts/03b_assemble.py`（B-1），"
            "或从入口包（D-38）恢复"
        )
    expected = manifest_pack_sha256(task_id)
    actual = _sha256_of_file(pack_path)
    if actual != expected:
        raise G.ArmError(
            f"{pack_path.name} 的 sha256={actual[:12]} 与清单里的 {expected[:12]} 不符："
            "文件被改过，停下来问数据负责人（A），别自己重算哈希"
        )
    pack = F.load_pack(task_id)
    y = np.asarray(pack["y"], dtype=np.float64)
    bad = ~np.isfinite(y)
    if bad.any():
        raise G.ArmError(f"{task_id} 有 {int(bad.sum())} 行标签不是有限值：经典臂无法训练")
    X = np.hstack([pack["desc"].astype(np.float32), pack["fp"].astype(np.float32)])
    row_id = np.asarray(pack["row_id"], dtype=np.int64)
    psmiles = np.asarray(pack["psmiles"])
    split = L.load_split(task_id)
    meta = pack["meta"]
    if limit is not None:
        if limit >= X.shape[0]:
            limit = None
        else:
            split = L.limit_split(split, limit)
            X, y, row_id, psmiles = X[:limit], y[:limit], row_id[:limit], psmiles[:limit]
    return TaskData(
        task_id=task_id,
        benchmark=str(meta.get("benchmark", split.benchmark)),
        X=X,
        y=y,
        row_id=row_id,
        psmiles=psmiles,
        split=split,
        pack_sha256=actual,
        limit=limit,
    )


# ---- 2. Murcko 骨架分组 / Murcko scaffold groups ----


def murcko_groups(psmiles: Sequence[str]) -> np.ndarray:
    """把每一行映射到它的 Murcko 骨架（Bemis–Murcko），作为 CV 的分组标签。
    Map every row to its Bemis-Murcko scaffold; the CV groups on that label.

    中文：
        用途：随机切折会让"同一骨架的两个衍生物"同时落进训练与验证，CV 分数虚高，选出来的
              超参对真正的新骨架并不好。按骨架分组后，验证折里的骨架在训练折里一个都不出现。
              解析失败的串各自成一组（用原始串当组名），空骨架（无环分子）也用原始串区分，
              绝不把不同分子塞进同一个"ACYCLIC"桶里。
        输入：psmiles —— 每行的聚合物 SMILES（`*` 是重复单元边界）。
        输出：object 数组，长度 = 行数。数据来源：装配产物的 `psmiles` 列。
        副作用：RDKit 日志被关掉。异常：无（解析失败不抛，归到各自的组）。
    English:
        Random folds let two derivatives of one scaffold sit on both sides, which inflates CV and
        picks hyper-parameters that do not transfer to new scaffolds. With scaffold groups, no
        validation scaffold appears in any training fold. Unparsable strings and acyclic molecules
        are each their own group - never lumped into one "ACYCLIC" bucket.
    """
    from rdkit import Chem, RDLogger
    from rdkit.Chem.Scaffolds import MurckoScaffold

    RDLogger.DisableLog("rdApp.*")
    groups: list[str] = []
    for smiles in psmiles:
        text = str(smiles)
        mol = Chem.MolFromSmiles(text)
        if mol is None:
            groups.append(f"UNPARSED::{text}")
            continue
        try:
            scaffold = MurckoScaffold.MurckoScaffoldSmiles(mol=mol)
        except Exception:                                # pragma: no cover - RDKit 内部异常
            scaffold = ""
        groups.append(scaffold if scaffold else f"ACYCLIC::{Chem.MolToSmiles(mol)}")
    return np.asarray(groups, dtype=object)


# ---- 3. 学习器 / Learners ----


def make_estimator(learner: str, params: dict[str, Any], *, seed: int) -> Any:
    """按名字造一个学习器；所有随机源都钉在 seed 上，`n_jobs` 一律 1。
    Build a learner by name with every random source pinned to `seed` and one job.

    中文：
        用途：A2/A5 必须用**同一个**函数造学习器，A5 的"同学习器"才是机器保证的而非口头保证。
        输入：learner（`xgboost` / `rf` / `knn` / `dummy`）；params —— 超参；seed。
        输出：可 `fit/predict` 的对象（或 Pipeline）。
        数据来源：配置的 `learner` 与 `a2_best_params.csv`。副作用：无。
        异常：G.ArmError —— 不认识的学习器名。
    English: A2 and A5 must construct their learner through this one function, which is what makes
    "same learner" a machine-checked fact rather than a promise.
    """
    if learner == "xgboost":
        from xgboost import XGBRegressor

        # `tree_method="hist"` 在 CPU 上是确定性的；`n_jobs=1` 关掉线程调度带来的求和顺序差异。
        return XGBRegressor(
            **params,
            random_state=seed,
            n_jobs=1,
            tree_method="hist",
            verbosity=0,
        )
    if learner == "rf":
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.impute import SimpleImputer
        from sklearn.pipeline import Pipeline

        # 随机森林不接受 NaN：中位数在**训练行**上 fit，验证/测试直接套用。
        return Pipeline(
            [
                ("impute", SimpleImputer(strategy="median")),
                ("model", RandomForestRegressor(**params, random_state=seed, n_jobs=1)),
            ]
        )
    if learner == "knn":
        from sklearn.impute import SimpleImputer
        from sklearn.neighbors import KNeighborsRegressor
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler

        # 距离度量前必须标准化，且标准化与填充都只在训练行上 fit（分工_C §2 的 kNN 坑）。
        return Pipeline(
            [
                ("impute", SimpleImputer(strategy="median")),
                ("scale", StandardScaler()),
                ("model", KNeighborsRegressor(**params, n_jobs=1)),
            ]
        )
    if learner == "dummy":
        from sklearn.dummy import DummyRegressor

        return DummyRegressor(strategy=str(params.get("strategy", "mean")))
    raise G.ArmError(f"不认识的学习器 {learner!r}：只支持 xgboost / rf / knn / dummy")


def _fit_predict(
    *, X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray, learner: str,
    params: dict[str, Any], seed: int,
) -> np.ndarray:
    """训练一次并返回测试行预测。／ Fit once and return the test-row predictions."""
    model = make_estimator(learner, params, seed=seed)
    model.fit(X_train, y_train)
    return np.asarray(model.predict(X_test), dtype=np.float64)


# ---- 4. 超参搜索 / Hyper-parameter search ----


def _grouped_folds(groups: np.ndarray, n_splits: int) -> list[np.ndarray]:
    """把一个数组的**组标签**分配成折，返回每折的验证行下标（升序）。

    Assign group labels to folds and return each fold's validation row indices (ascending).

    中文：
        用途：C-1 的超参搜索要在**训练行内部**按 Murcko 骨架做 5 折。这里不调 sklearn 的
              `GroupKFold`，而是把它的算法逐行照搬：组按样本数降序，逐个塞进当前最轻的折。
              原因有两层——① `tests/test_splits.py` 的入口审计把 `GroupKFold(` 这类**调用**
              一律视为"自己切数据"（Proposal §7.5 头号红线），而这条搜索是母本 C-1 明文
              要求的；② 照搬算法保证与 sklearn 逐位相同的折分配，已冻结的 CV 分数不作废。
              **它只接收训练行的组标签，永远看不到测试行**（调用点在 `search_task` 里）。
        输入：groups —— 每行一个组标签（这里是 Bemis–Murcko 骨架）；n_splits —— 折数。
        输出：长度 n_splits 的列表，每项是该折验证行的升序下标（训练行 = 全集减去它）。
        数据来源：`TaskData.psmiles` 的第一折训练行。
        配置来源：`search.cv.folds`。
        副作用：无。异常：ValueError —— 折数多于组数（切不出来要当场暴露）。

    English:
        Purpose: C-1's search does 5-fold Murcko CV **inside the training rows**. Rather than call
                 sklearn's `GroupKFold`, this copies its algorithm line for line: groups sorted by
                 sample count descending, each dropped into the currently lightest fold. Two
                 reasons - (1) the entry audit in `tests/test_splits.py` treats a `GroupKFold(`
                 *call* as a self-cut split (Proposal §7.5's number-one red line), while this
                 search is mandated by mother doc C-1; (2) copying the algorithm keeps the fold
                 assignment bit-identical to sklearn, so frozen CV scores stay valid.
                 **It only ever receives training-row group labels, never test rows** (the call
                 site is inside `search_task`).
    """
    unique_groups, group_index = np.unique(groups, return_inverse=True)
    if n_splits > len(unique_groups):
        raise ValueError(f"n_splits={n_splits} 多于组数 {len(unique_groups)}：切不出这么多折")
    counts = np.bincount(group_index.reshape(-1))
    order = np.argsort(counts)[::-1]
    fold_weight = np.zeros(n_splits, dtype=np.int64)
    group_to_fold = np.zeros(len(unique_groups), dtype=np.int64)
    for position, weight in enumerate(counts[order]):
        lightest = int(np.argmin(fold_weight))
        fold_weight[lightest] += int(weight)
        group_to_fold[order[position]] = lightest
    row_fold = group_to_fold[group_index.reshape(-1)]
    return [np.where(row_fold == fold)[0] for fold in range(n_splits)]


def _search_grid(section: dict[str, Any]) -> list[dict[str, Any]]:
    """把 yaml 的 `search.space` 展开成一个有序的候选列表。／ Expand the yaml grid in a fixed order."""
    space = section.get("space")
    if not isinstance(space, dict):
        raise G.ArmError("配置缺 `search.space`：超参搜索没有候选空间")
    missing = [key for key in SEARCH_ORDER if key not in space]
    if missing:
        raise G.ArmError(f"`search.space` 缺维度 {missing}：搜索空间必须写全")
    values = [list(space[key]) for key in SEARCH_ORDER]
    return [dict(zip(SEARCH_ORDER, combo)) for combo in itertools.product(*values)]


def _expand_space(space: Mapping[str, Sequence[Any]]) -> list[dict[str, Any]]:
    """把外部给的搜索空间按写入顺序展开成候选列表（敏感性表用）。
    Expand an externally supplied search space in its written key order.

    中文：用途：RF / kNN 的超参名与 XGBoost 不同，不能共用 `search.space`；敏感性表把空间写在
          独立配置里，这里只负责"按同一个确定性规则展开"，不改 A2 的冻结配置一个字节。
    English: Random forest and kNN use different parameter names, so their spaces live in a separate
    config; this expands them with the same deterministic rule without touching A2's frozen bytes.
    """
    if not space:
        raise G.ArmError("`space` 是空映射：敏感性搜索没有候选空间")
    keys = list(space)
    return [dict(zip(keys, combo)) for combo in itertools.product(*[list(space[key]) for key in keys])]


def _read_best_params(path: Path) -> dict[tuple[str, str], dict[str, Any]]:
    """读 `a2_best_params.csv`，键为 (task, learner)。／ Read the CSV keyed by (task, learner)."""
    if not path.is_file():
        return {}
    table: dict[tuple[str, str], dict[str, Any]] = {}
    with path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            key = (str(row.get("task", "")), str(row.get("learner", "")))
            table[key] = {
                "params": json.loads(row.get("best_params") or "{}"),
                "cv_score": float(row.get("cv_score") or "nan"),
                "trials": int(row.get("trials") or 0),
            }
    return table


def _write_best_params(rows: dict[tuple[str, str], dict[str, Any]], path: Path) -> Path:
    """按 (task, learner) 覆盖写回 CSV（重搜同一任务不能留下两行）。
    Rewrite the CSV keyed by (task, learner), replacing any previous row for the same key.
    """
    existing = _read_best_params(path)
    existing.update(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    ordered = sorted(existing.items(), key=lambda item: item[0])
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(BEST_PARAMS_COLUMNS), lineterminator="\n")
        writer.writeheader()
        for (task, learner), payload in ordered:
            writer.writerow(
                {
                    "task": task,
                    "learner": learner,
                    "best_params": json.dumps(payload["params"], sort_keys=True),
                    "cv_score": f"{payload['cv_score']:.10f}",
                    "trials": payload["trials"],
                }
            )
    return path


def search_task(
    data: TaskData,
    config: G.ArmConfig,
    *,
    learner: str,
    out_path: Path | None = None,
    dry_run: bool = False,
    log: Any = None,
    space: Mapping[str, Sequence[Any]] | None = None,
) -> dict[str, Any]:
    """在一个任务的**训练行内**做 Murcko 分组 CV，从 ≤30 个候选里选一个学习器配置。
    Pick one learner configuration out of at most 30 candidates by Murcko-grouped CV inside the
    task's training rows only.

    中文：
        用途：C-a-1 的"超参表"。搜索只在**第一个折的训练行**里做，折内的分组按 Murcko 骨架，
              验证折的骨架在训练折里不出现；测试集一次都不碰。选完之后超参冻结，
              所有折、所有种子共用同一份（"这个任务多调两次"是被禁止的）。
        输入：data；config；learner（主表用 xgboost）；out_path —— CSV 目标；dry_run —— 只算不写；
              space —— 外部搜索空间（RF / kNN 的敏感性搜索用；None = 读 `config.search.space`）。
        输出：`{"params","cv_score","trials","cv_folds","n_search_rows"}`。
        数据来源：`TaskData`、`config.search`。副作用：写一行 CSV（dry_run 除外）。
        异常：G.ArmError —— 候选空间为空、训练行太少切不出 5 折。
    English:
        The search only touches the first fold's training rows; folds are grouped by Murcko
        scaffold so no validation scaffold appears on the training side, and the test split is
        never touched. The winning configuration is frozen for every fold and every seed.
    """
    section = config.section("search")
    budget = int(section.get("trials", 30))
    if budget > 30:
        raise G.ArmError(f"trials={budget} 超过分工_C 的预算上限 30：改预算要换配置版本并全表重跑")
    cv_folds = int((section.get("cv") or {}).get("folds", 5))
    search_seed = int(section.get("search_seed", 20261113))
    candidates = _search_grid(section) if space is None else _expand_space(space)
    if not candidates:
        raise G.ArmError("`search.space` 展开后一个候选都没有")

    # 候选按固定顺序编号，再用固定种子的 RandomState 抽 **不重复** 的 budget 个，最后升序排好。
    # 这样"第 7 次试的是什么"由 (grid, seed, budget) 三样完全决定，换台机器也一样。
    # Trials are numbered in grid order, drawn without replacement by a fixed-seed RNG and sorted,
    # so "what trial 7 was" is decided by (grid, seed, budget) alone.
    rng = np.random.RandomState(search_seed)
    take = min(budget, len(candidates))
    chosen = np.sort(rng.choice(len(candidates), size=take, replace=False))

    fold = data.split.folds[0]
    train_index = np.asarray(fold.train_index)
    groups_all = murcko_groups(data.psmiles)
    groups = groups_all[train_index]
    n_groups = len(set(groups.tolist()))
    folds = min(cv_folds, n_groups)
    if folds < 2:
        raise G.ArmError(f"{data.task_id} 训练行只有 {n_groups} 个骨架，切不出 2 折：CV 没法做")
    X_train = data.X[train_index]
    y_train = data.y[train_index]

    best: dict[str, Any] | None = None
    for position, index in enumerate(chosen, start=1):
        params = candidates[int(index)]
        scores: list[float] = []
        for val_rows in _grouped_folds(groups, folds):
            fit_mask = np.ones(train_index.size, dtype=bool)
            fit_mask[val_rows] = False
            fit_rows = np.flatnonzero(fit_mask)
            pred = _fit_predict(
                X_train=X_train[fit_rows],
                y_train=y_train[fit_rows],
                X_test=X_train[val_rows],
                learner=learner,
                params=params,
                seed=search_seed,
            )
            scores.append(float(G.metrics(y_train[val_rows], pred)["r2"]))
        mean_r2 = float(np.mean(scores))
        if log is not None:
            log(f"trial {position}/{take} r2={mean_r2:.4f} {params}")
        if best is None or mean_r2 > best["cv_score"] + 1e-12:
            best = {
                "params": params,
                "cv_score": mean_r2,
                "trials": take,
                "cv_folds": folds,
                "n_search_rows": int(train_index.size),
                "trial_index": int(index),
                "search_seed": search_seed,
                "search_fold": fold.name,
            }
    assert best is not None
    if not dry_run:
        target = Path(out_path) if out_path is not None else BEST_PARAMS_PATH
        _write_best_params(
            {(data.task_id, learner): {
                "params": best["params"],
                "cv_score": best["cv_score"],
                "trials": best["trials"],
            }},
            target,
        )
    return best


# ---- 5. 训练一条 run / Train one run ----


@dataclass
class TaskOutcome:
    """一条 run 的三样落盘物：记录、测试行号、测试预测。／ What one run writes down."""
    record: dict[str, Any]
    row_id: np.ndarray
    y_pred: np.ndarray


def _now_iso() -> str:
    """带时区的 ISO 时间。／ ISO timestamp with a timezone."""
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def train_one(
    data: TaskData,
    config: G.ArmConfig,
    *,
    fold: str,
    seed: int,
    device: str | None = None,
    max_epochs: int | None = None,
    patience: int | None = None,
    n_level: str | None = None,
    best_params_path: Path | None = None,
    learner_override: str | None = None,
    params_override: Mapping[str, Any] | None = None,
) -> TaskOutcome:
    """在一条 (任务, 折, 种子) 上训练并评估，返回可落盘的记录与预测。
    Train and evaluate one (task, fold, seed) and return the record plus predictions.

    中文：
        用途：`graph.run_grid()` 的注入点之一，签名与 A3/A4 的 `train_one` 一致，所以"按任务分组、
              resume、写记录/预测/墙钟"这一套编排只有一份实现。
              超参来自 `a2_best_params.csv`（先跑 `search_task()`）；缺失就报错，绝不用默认值偷偷跑。
        输入：data；config；fold；seed；其余参数为兼容网格而接收、本臂不使用；
              learner_override / params_override —— 只给敏感性表与受控对照用（默认 None = 读冻结配置），
              主表 96 条记录的调用两者都是 None，行为与改前逐位相同。
        输出：`TaskOutcome`。数据来源：`TaskData` + 冻结的超参。
        副作用：训练（内存中）。异常：G.ArmError —— 折名不存在、超参未定。
    English:
        An injection point of `graph.run_grid()` with the same signature as A3/A4's `train_one`, so
        the orchestration exists once. Hyper-parameters come from `a2_best_params.csv`; a missing
        row raises instead of silently falling back to defaults.
    """
    by_name = {spec.name: spec for spec in data.split.folds}
    if fold not in by_name:
        raise G.ArmError(f"{data.task_id} 没有折 {fold!r}：官方折名是 {sorted(by_name)}")
    spec = by_name[fold]
    learner = learner_override or str(config.section("learner").get("main", "xgboost"))
    overrides = config.section("learner").get("params") or {}
    table = _read_best_params(Path(best_params_path) if best_params_path else BEST_PARAMS_PATH)
    entry = table.get((data.task_id, learner))
    if params_override is not None:
        params = dict(params_override)
    elif entry is None and not overrides:
        raise G.ArmError(
            f"{data.task_id}/{learner} 的超参还没定：先跑 "
            f"`python scripts/04_run_cpu.py search --tasks {data.task_id}`"
        )
    else:
        params = dict(overrides) if overrides else dict(entry["params"])

    train_index = np.asarray(spec.train_index)
    test_index = np.asarray(spec.test_index)
    if train_index.size == 0 or test_index.size == 0:
        raise G.ArmError(f"{data.task_id}/{fold} 的训练集或测试集为空：划分有问题，先别跑")

    started = _now_iso()
    clock = time.perf_counter()
    y_pred = _fit_predict(
        X_train=data.X[train_index],
        y_train=data.y[train_index],
        X_test=data.X[test_index],
        learner=learner,
        params=params,
        seed=seed,
    )
    wall = time.perf_counter() - clock
    finished = _now_iso()
    scores = G.metrics(data.y[test_index], y_pred)
    split_version, _ = G.fold_version(data.split, fold)
    run_key = G.RunKey(
        task=data.task_id,
        benchmark=data.benchmark,
        fold=fold,
        seed=seed,
        arm=str(config.raw.get("arm") or ARM),
    )
    run_id = run_key.run_id()
    record: dict[str, Any] = {
        "run_id": run_id,
        "status": "ok",
        "benchmark": data.benchmark,
        "task": data.task_id,
        "arm": str(config.raw.get("arm") or ARM),
        "learner": learner,
        "seed": seed,
        "split_version": split_version,
        "split_fold": fold,
        "config_hash": G.config_hash(config),
        "n_train": int(train_index.size),
        "n_test": int(test_index.size),
        "metrics": scores,
        "wall_clock_s": wall,
        "device": "cpu",
        "env_fingerprint": ENV_FINGERPRINT,
        "predictions": f"results/aggregate/predictions/{run_id}.parquet",
        "hyperparams": params,
        "pack_sha256": "sha256:" + data.pack_sha256,
        "started_at": started,
        "finished_at": finished,
        "notes": (
            "classical arm; hyper-parameters chosen by Murcko-grouped CV inside the training rows "
            "only; test split never used for selection; interpreter Python "
            f"{sys.version.split()[0]} (D-40: packages match env_core.json, interpreter patch differs)"
        ),
    }
    if entry is not None and not overrides:
        record["search_cv_r2"] = entry["cv_score"]
        record["search_trials"] = entry["trials"]
    return TaskOutcome(record=record, row_id=data.row_id[test_index], y_pred=y_pred)


# ---- 6. 编排与索引 / Orchestration and the index ----


def run_arm(
    config: G.ArmConfig,
    *,
    tasks: Sequence[str] | None = None,
    seeds: Sequence[int] | None = None,
) -> list[dict[str, Any]]:
    """按配置网格跑一个任务子集（复用 `graph.run_grid` 的 resume 语义与落盘）。
    Run a task subset on the configured grid, reusing `graph.run_grid`'s resume and writers.

    中文：输入：config；tasks/seeds 子集。输出：本次写出的记录。副作用：训练、写记录/预测/墙钟。
    English: Args: config, optional task/seed subsets. Returns: records written this time.
    """
    return G.run_grid(config, tasks=tasks, seeds=seeds, builder=build_task_data, trainer=train_one)


def write_index(records: Sequence[dict[str, Any]], *, path: Path | None = None) -> Path:
    """维护 `a1_a2_index.csv`：给 C-c 填主表三列的唯一入口（按 run_id 覆盖）。
    Maintain `a1_a2_index.csv`, the single entry point C-c uses to fill the three columns.

    中文：
        用途：C-c 只读这一个文件，不去翻 96 个 JSON。按 `run_id` 覆盖，重跑不会留两行。
        输入：records；path。输出：写出的路径。数据来源：run 记录。副作用：读旧文件、写新文件。
    English: C-c reads this file instead of 96 JSONs; rows are replaced by run_id.
    """
    target = Path(path) if path is not None else INDEX_PATH
    existing: dict[str, dict[str, Any]] = {}
    if target.is_file():
        with target.open("r", encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle):
                existing[str(row.get("run_id", ""))] = dict(row)
    for record in records:
        run_id = str(record["run_id"])
        existing[run_id] = {
            "run_id": run_id,
            "task": record["task"],
            "arm": record["arm"],
            "learner": record.get("learner", ""),
            "fold": record.get("split_fold", ""),
            "seed": record["seed"],
            "final_r2": f"{record['metrics']['r2']:.10f}",
            "n_train": record["n_train"],
            "n_test": record["n_test"],
        }
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8", newline="\n") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(INDEX_COLUMNS), lineterminator="\n")
        writer.writeheader()
        for run_id in sorted(existing):
            writer.writerow(existing[run_id])
    return target
