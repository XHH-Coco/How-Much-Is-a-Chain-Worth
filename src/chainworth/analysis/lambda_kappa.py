"""Λ：换用能看见链的模型之后，"原本解释不了的那部分方差"里捞回来几成。
Lambda: after switching to a chain-aware model, how much of the previously unexplained variance
comes back.

中文：
    用途：分工 D-1 的核心量。定义（Proposal §4.1）：

        Λ = (R²_chain − R²_unit) / (1 − R²_unit)

    `unit` 臂 = **A2 经典臂**（只给重复单元描述符），`chain` 臂 = **A3 图网络 / A4 polyBERT**。
    分母是"还剩多少可捞"，所以 Λ 不带单位，八个任务可以横着比。

    三条硬规矩（分工 D-1 的验收，逐条落在类型上）：
      ① **不 clip**：负 Λ 照原样报。截成 0，图上会显得"所有任务都至少没变差"，那是篡改；
      ② **分母不稳定就不报 Λ**：`R²_unit ≥ 0.95` 时分母趋 0，Λ 能跳到 −3 或 0.8。此时
         `value` 一律给 **NaN**、`reportable=False`，只留 `delta_r2` 与 `diagnostic`（算出来的那个
         数，仅供人工看了一眼）。**用类型挡住"顺手画上去"**，而不是靠文档提醒；
      ③ **分子分母同时为 0**（`R²_unit = R²_chain = 1`）返回 NaN 并记一条 warning，不许崩。

    为什么 `value` 与 `diagnostic` 分开：只留一个字段时，下游拿到的 NaN 与"算不出来"混在一起；
          分成两个，"该报的数"与"我算了一下但没资格报"就分得开。这是 A5 撤除那一轮（D-48）学到的：
          **让不该用的值在类型上就取不到，比在文档里写'不要用'可靠**。

    数据来源：`results/raw_runs/*.json` + `results/aggregate/predictions/*.parquet` +
              `data/processed/features/<task>.npz` 的 `y`（标签的唯一来源，不从记录里抄）。
    配置来源：本模块的 `UNIT_ARM` / `CHAIN_ARMS` / `MAIN_FOLD_LIMIT`。
    副作用：`build_lambda_table()` 只读，返回行；写盘由脚本负责。
    异常：`LambdaError` —— 配对失败（同一个 (task, fold, seed) 上只有一条臂）。
    对外：`lambda_from_r2()` / `pairing_key()` / `collect_pairs()` / `build_lambda_table()`。
    被谁用：`scripts/06c_lambda.py`、`chainworth.analysis.ci`、`tests/test_lambda.py`。

English:
    Purpose: the central quantity of D-1 (Proposal §4.1). Three hard rules, enforced by the types:
    (1) never clip - a negative Lambda is reported as is; (2) when `R2_unit >= 0.95` the denominator
    collapses, so `value` is NaN and `reportable` is False, leaving only `delta_r2` and
    `diagnostic`; (3) `R2_unit == R2_chain == 1` yields NaN plus a warning instead of a crash.
    Splitting `value` from `diagnostic` keeps "the number to report" apart from "I computed it but
    may not report it" - the D-48 lesson: make an unusable value unreachable by type, not by prose.
    Data: run records, prediction parquet files, and the label inside the task pack (never copied
    from the records).
"""

from __future__ import annotations

import json
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from chainworth.metrics import regression as R

REPO_ROOT = Path(__file__).resolve().parents[3]
RAW_RUNS_DIR = REPO_ROOT / "results" / "raw_runs"
PREDICTIONS_DIR = REPO_ROOT / "results" / "aggregate" / "predictions"
FEATURES_DIR = REPO_ROOT / "data" / "processed" / "features"

# `unit` 臂与 `chain` 臂的名字。A2 是"只给重复单元"，A3/A4 是两条链侧的实现。
# The unit arm and the chain arms: A2 sees repeat units only; A3/A4 are the two chain-side arms.
UNIT_ARM = "a2_conventional"
CHAIN_ARMS = ("a3_graph", "a4_polybert")

# `R²_unit` 到了这个值，分母趋 0，Λ 不再可报（D-1 ③ 的阈值）。
# Above this `R2_unit` the denominator collapses and Lambda is not reportable.
DENOMINATOR_THRESHOLD = 0.95

# OpenPoly 的折索引 0–4 是主表、5–9 是敏感性（D-36）。不为这两批分别建表，只取主表，
# 否则同一个 (task, level, seed) 会出现两条臂各两份、Λ 被算两遍。
# OpenPoly folds 0-4 are the main table and 5-9 the sensitivity batch (D-36); only 0-4 here.
MAIN_FOLD_LIMIT = 5

# 机制实验（D-69 的 `pb26_scaling_cp`）不进 Λ 主表。理由不是"算不出来"，是**会与主表混口径**：
# 它的 40 个格是 8 个 (数据集, 档位) × 5 折，同一条折的**同一批测试行**要出现 4 次（4 个档位），
# 而 Λ 主表的每一行都对应互不重复的一份测试点。混进来会让"321 对"变成"361 行里有 40 行重复计数"，
# 读数的人会以为多了 40 份证据。机制实验的 Λ 逐格写在 `kappa_cells.csv` 里（见
# `docs/Kappa Table Notes.md` §5），那里才配得上它的设计。
# The mechanism experiment is kept out of the Lambda main table: its forty cells repeat the same
# test rows across four tiers, so mixing it in would turn "321 pairs" into "40 double counts".
SCALING_TASK = "pb26_scaling_cp"

STATUS_OK = "ok"
STATUS_UNSTABLE = "denominator_unstable"
STATUS_UNDEFINED = "undefined"

_FOLD_PATTERN = r"seed(\d+)$"


class LambdaError(RuntimeError):
    """Λ 算不了（配对缺一条臂、标签取不到）。/ Lambda cannot be formed."""


@dataclass(frozen=True)
class LambdaResult:
    """一个 (task, fold, seed) 上的 Λ 及其状态。/ Lambda on one unit, with its status.

    中文：
        字段：`value` —— **可报的 Λ**，不可报时是 NaN；`diagnostic` —— 算术结果，仅供诊断；
              `delta_r2` —— 分子，任何状态下都可报；`status` —— `ok` / `denominator_unstable` /
              `undefined`；`reportable` —— 是否允许进图、进结论；`reason` —— 人话。
        数据来源：调用方给的四个数（两个 R²）。副作用：无。
    English:
        `value` is the reportable Lambda (NaN when not reportable); `diagnostic` is the raw
        arithmetic; `delta_r2` is the numerator and is always reportable; `reportable` gates the
        figures and the conclusions.
    """

    value: float
    diagnostic: float
    delta_r2: float
    status: str
    reportable: bool
    reason: str


def lambda_from_r2(
    r2_chain: float,
    r2_unit: float,
    *,
    threshold: float = DENOMINATOR_THRESHOLD,
) -> LambdaResult:
    """按 Λ = (R²_chain − R²_unit) / (1 − R²_unit) 算，并按三条硬规矩定状态。
    Compute Lambda and classify it under the three hard rules.

    中文：
        用途：D-1 ③④ 的唯一实现。**不 clip**、分母趋 0 时不给可报值、分子分母同为 0 时不崩。
        输入：r2_chain / r2_unit（两个 R²，可以是 NaN）；threshold（默认 0.95）。
        输出：`LambdaResult`。数据来源：调用方。副作用：无（可能记一条 warning）。
        异常：无 —— 所有病态输入都变成带状态的结果，而不是异常。
    English:
        The single implementation of D-1 ③④. Never clips; refuses a reportable value when the
        denominator collapses; survives the 0/0 case. All pathological inputs become a status, not
        an exception.
    """
    delta = float(r2_chain) - float(r2_unit)

    if not np.isfinite(r2_chain) or not np.isfinite(r2_unit):
        return LambdaResult(
            value=float("nan"),
            diagnostic=float("nan"),
            delta_r2=float("nan"),
            status=STATUS_UNDEFINED,
            reportable=False,
            reason="某一侧 R² 是 NaN（测试集标签全同，或该臂没有可用预测）",
        )

    denominator = 1.0 - float(r2_unit)
    if denominator == 0.0:
        # R²_unit 恰为 1：分母为 0。分子也为 0 就是 0/0。
        if delta == 0.0:
            reason = "R²_unit = R²_chain = 1：分子分母同时为 0，Λ 无定义（记 warning，不崩）"
        else:
            reason = "R²_unit = 1：分母为 0，Λ 无定义（分子不为 0，说明链臂还差了一截）"
        warnings.warn(reason, RuntimeWarning, stacklevel=2)
        return LambdaResult(
            value=float("nan"),
            diagnostic=float("inf") if delta > 0 else float("-inf"),
            delta_r2=delta,
            status=STATUS_UNDEFINED,
            reportable=False,
            reason=reason,
        )

    diagnostic = delta / denominator
    if denominator <= (1.0 - threshold):
        return LambdaResult(
            value=float("nan"),
            diagnostic=diagnostic,
            delta_r2=delta,
            status=STATUS_UNSTABLE,
            reportable=False,
            reason=(
                f"R²_unit = {r2_unit:.4f} ≥ {threshold}：分母只剩 {denominator:.4f}，"
                f"Λ 会被放大到不可解释。只报 ΔR² = {delta:+.4f}"
            ),
        )

    return LambdaResult(
        value=diagnostic,
        diagnostic=diagnostic,
        delta_r2=delta,
        status=STATUS_OK,
        reportable=True,
        reason="",
    )


# --------------------------------------------------------------------------- #
# 配对：同一个 (task, fold, seed) 上取两条臂 / Pairing unit and chain on the same unit
# --------------------------------------------------------------------------- #
def pairing_key(run_id: str) -> tuple[str, str, str]:
    """把 run_id 里的臂名挖掉，得到跨臂的配对键。
    Remove the arm segment from a run_id to get the cross-arm pairing key.

    中文：
        run_id 的形状是 `task__arm__<中间段>__seed`（`<中间段>` 是折名或 `seed0_n30`）。
        配对键 = (task, 中间段, seed)。**不按字符串前缀猜**，按 `__` 切四段——
        臂名里带下划线（`a2_conventional`），用 `split("__")` 才切得对。
        输入：run_id。输出：三元组。副作用：无。异常：`LambdaError` —— 段数不是 4。
    English:
        The run_id shape is `task__arm__<middle>__seed`; the key is (task, middle, seed). Split on
        the double underscore, not on single underscores, because arm names contain them.
    """
    parts = run_id.split("__")
    if len(parts) != 4:
        raise LambdaError(f"run_id 不是四段：{run_id}")
    return parts[0], parts[2], parts[3]


def _fold_index(middle: str) -> int | None:
    """从中间段里取折索引（`seed0_n30` → 0）；取不到返回 None。
    The fold index inside the middle segment, or None when it is a named fold.
    """
    import re

    match = re.search(_FOLD_PATTERN, middle)
    return int(match.group(1)) if match else None


def load_records(directory: Path | None = None) -> dict[str, dict[str, Any]]:
    """读 run 记录，返回 {run_id: record}。/ Read the run records keyed by run_id."""
    base = directory or RAW_RUNS_DIR
    records: dict[str, dict[str, Any]] = {}
    for path in sorted(base.glob("*.json")):
        if path.name == "_schema.json":
            continue
        record = json.loads(path.read_text(encoding="utf-8"))
        records[record["run_id"]] = record
    return records


def _task_labels(
    task_id: str,
    cache: dict[str, np.ndarray],
    dataset_id: str | None = None,
) -> np.ndarray:
    """取该任务整表的标签（从 pack 里读，不从记录里抄）。
    The task's labels straight from the feature pack, never copied from a record.

    中文：
        输入：task_id / cache / dataset_id（双数据集任务才有，如 `pb26_scaling_cp`）。
        输出：float64 数组。数据来源：`data/processed/features/<task>.npz`；
              双数据集任务落成 `<task>_<dataset>.npz`（D-54：一个任务两个数据集）。
        副作用：只读 + 填缓存。异常：`LambdaError` —— 两份候选 pack 都不存在。
    English:
        Labels come from the pack; a dual-dataset task (D-54) stores `<task>_<dataset>.npz`, so the
        dataset id is part of the lookup. A label cache keeps the 321-pair table fast.
    """
    key = f"{task_id}::{dataset_id}" if dataset_id else task_id
    if key in cache:
        return cache[key]
    candidates = []
    if dataset_id:
        candidates.append(FEATURES_DIR / f"{task_id}_{dataset_id}.npz")
    candidates.append(FEATURES_DIR / f"{task_id}.npz")
    path = next((item for item in candidates if item.is_file()), None)
    if path is None:
        raise LambdaError(
            f"缺特征 pack：{' / '.join(str(p) for p in candidates)}"
            "（重建：python scripts/03b_assemble.py）"
        )
    cache[key] = np.asarray(np.load(path, allow_pickle=True)["y"], dtype=np.float64)
    return cache[key]


def _prediction(run_id: str, directory: Path | None = None) -> tuple[np.ndarray, np.ndarray]:
    """读一份预测，返回 (row_id, y_pred)。/ Read one prediction file."""
    import pandas as pd

    path = (directory or PREDICTIONS_DIR) / f"{run_id}.parquet"
    if not path.is_file():
        raise LambdaError(f"缺预测文件：{path}")
    frame = pd.read_parquet(path)
    return frame["row_id"].to_numpy(), frame["y_pred"].to_numpy(dtype=np.float64)


def collect_pairs(
    records: Mapping[str, Mapping[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    """把 (unit, chain) 配成对，并丢掉不满足配对条件的组合。
    Pair unit with chain and drop the combinations that cannot be paired.

    中文：
        用途：Λ 的前提是"同一个 (task, fold, seed) 上有两条臂"。这里只做配对，不算指标。
        输入：records（缺省读盘）。输出：dict 列表，含 `task` / `arm` / `middle` / `seed` /
              `unit_run` / `chain_run` / `dataset_id`。
        为什么带 `dataset_id`：双数据集任务（D-54，如 `pb26_scaling_cp`）的特征 pack 落成
              `<task>_<dataset>.npz`，少了这个字段就取不到标签。折名里虽然有数据集前缀，
              但**从折名反推数据集**是猜字符串；记录的 `dataset_id` 才是权威来源。
        数据来源：`results/raw_runs/`。副作用：只读。异常：`LambdaError` —— run_id 形状不符。
        两条过滤：① `a1_floor` 不是 unit 臂（它是地板，不是"只给重复单元"的模型），不进配对；
                  ② OpenPoly 的折索引 5–9 是敏感性（D-36），不进主表；
                  ③ `pb26_scaling_cp` 是机制实验，整任务不进主表（理由见 `SCALING_TASK`）。
    English:
        Pairing only, no metrics. `a1_floor` is excluded (a floor is not a unit-feature model),
        OpenPoly folds 5-9 are the sensitivity batch (D-36), and the mechanism experiment stays out
        of the main table because its cells share test rows across tiers.
    """
    source = records if records is not None else load_records()
    by_key: dict[tuple[str, str, str], dict[str, str]] = {}
    for run_id, record in source.items():
        arm = str(record.get("arm", ""))
        if arm not in (UNIT_ARM, *CHAIN_ARMS):
            continue
        task, middle, seed = pairing_key(run_id)
        if task == SCALING_TASK:
            continue
        fold = _fold_index(middle)
        if fold is not None and fold >= MAIN_FOLD_LIMIT:
            continue
        by_key.setdefault((task, middle, seed), {})[arm] = run_id

    pairs: list[dict[str, Any]] = []
    for (task, middle, seed), arms in sorted(by_key.items()):
        if UNIT_ARM not in arms:
            continue
        for arm in CHAIN_ARMS:
            if arm not in arms:
                continue
            chain_id = arms[arm]
            pairs.append(
                {
                    "task": task,
                    "arm": arm,
                    "middle": middle,
                    "seed": seed,
                    "unit_run": arms[UNIT_ARM],
                    "chain_run": chain_id,
                    # 双数据集任务取链臂记录上的 `dataset_id`（两条臂同数据集，取一条即可）。
                    "dataset_id": source[chain_id].get("dataset_id"),
                }
            )
    return pairs


def build_lambda_table(
    records: Mapping[str, Mapping[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    """逐对算 Λ，返回可直接写 CSV 的行（不含区间；区间由 `analysis.ci` 补）。
    Compute Lambda for every pair; the intervals are added by `analysis.ci`.

    中文：
        输入：records（缺省读盘）。输出：dict 列表，列见 `LAMBDA_COLUMNS`。
        数据来源：run 记录 + 预测 parquet + pack 的 `y`。副作用：只读。
        异常：`LambdaError` —— 配对成立但取不到预测/标签。
        自检：两条臂的**测试集行集合**必须相同（同折同种子），不同则记进 `note` 并仍算——
              因为 Λ 的前提正是"同一批测试点"，不同的话那个数不能横着比。
    English:
        Per-pair Lambda, ready for CSV. The two arms must share the same test row set; a mismatch is
        recorded in `note` because Lambda's premise is exactly "the same test points".
    """
    pairs = collect_pairs(records)
    labels: dict[str, np.ndarray] = {}
    rows: list[dict[str, Any]] = []
    for pair in pairs:
        truth = _task_labels(pair["task"], labels, pair.get("dataset_id"))
        unit_ids, unit_pred = _prediction(pair["unit_run"])
        chain_ids, chain_pred = _prediction(pair["chain_run"])
        same_rows = np.array_equal(np.sort(unit_ids), np.sort(chain_ids))

        r2_unit = R.scores_from_predictions(truth, unit_ids, unit_ids, unit_pred)["r2"]
        r2_chain = R.scores_from_predictions(truth, chain_ids, chain_ids, chain_pred)["r2"]
        result = lambda_from_r2(r2_chain, r2_unit)

        rows.append(
            {
                "task": pair["task"],
                "arm": pair["arm"],
                "fold": pair["middle"],
                "seed": pair["seed"],
                "unit_run": pair["unit_run"],
                "chain_run": pair["chain_run"],
                "n_test": int(unit_ids.size),
                "r2_unit": r2_unit,
                "r2_chain": r2_chain,
                "delta_r2": result.delta_r2,
                "lambda": result.value,
                "lambda_diagnostic": result.diagnostic,
                "status": result.status,
                "reportable": result.reportable,
                "note": "" if same_rows else "两条臂的测试集行集合不同，Λ 不可横比",
            }
        )
    return rows


LAMBDA_COLUMNS: tuple[str, ...] = (
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
    "lambda",
    "lambda_diagnostic",
    "status",
    "reportable",
    "note",
)


def summarise_by_arm(rows: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """按 (task, arm) 汇总：可报条数、Λ 的均值与 run 间 sd、ΔR² 的均值与 sd。
    Per (task, arm): how many rows are reportable, plus mean and run-to-run sd of Lambda and dR2.

    中文：
        用途：图 1 的一个格子要一个数。**报的是"run 间 sd"，不是"seed 间 sd"**——折与种子都变了，
              混在一起写会让人以为那是种子方差（D-36 ⑤ 的同一个坑）。
        输入：`build_lambda_table()` 的行。输出：dict 列表。
        副作用：无。异常：无。备注：一个可报行都没有时，`lambda_mean` 给空字符串而不是 0——
              "没算出来"与"算出来是 0"必须分得开（主表缺格的同一条规矩）。
    English:
        One number per cell of figure 1. The spread reported is the **run-to-run** sd (folds and
        seeds both vary), not a seed-only sd. When nothing is reportable, the mean is an empty
        string, never 0: "not computable" and "computed as zero" must stay distinguishable.
    """
    grouped: dict[tuple[str, str], list[Mapping[str, Any]]] = {}
    for row in rows:
        grouped.setdefault((str(row["task"]), str(row["arm"])), []).append(row)

    summary: list[dict[str, Any]] = []
    for (task, arm), group in sorted(grouped.items()):
        reportable = [row for row in group if row["reportable"]]
        lambdas = [float(row["lambda"]) for row in reportable]
        deltas = [float(row["delta_r2"]) for row in reportable]
        summary.append(
            {
                "task": task,
                "arm": arm,
                "n_pairs": len(group),
                "n_reportable": len(reportable),
                "lambda_mean": float(np.mean(lambdas)) if lambdas else "",
                "lambda_sd_runs": float(np.std(lambdas, ddof=1)) if len(lambdas) > 1 else "",
                "delta_r2_mean": float(np.mean(deltas)) if deltas else "",
                "delta_r2_sd_runs": float(np.std(deltas, ddof=1)) if len(deltas) > 1 else "",
            }
        )
    return summary
