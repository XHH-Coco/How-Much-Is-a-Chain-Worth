"""指标的手算对照：三个公式各自用"能心算的数"核一遍，再和盘上全部记录逐条对拍。
Hand-computed checks for the three formulas, then a row-by-row comparison against every record.

（条数不写死：记录只增不减，写死的数每次加批都会假红，而假的红会训练人忽略红——D-72。）

中文：
    用途：分工 D-1 的验收 ①②。两层：
          ① **手算对照**：`[1,2,3]` 对 `[1,2,4]` 的 R² 是 0.5、MAE 是 1/3、RMSE 是 √(1/3)——
             这些数在纸上算得出来，所以"实现写错了"与"数据变了"能分开；
          ② **与记录对拍**：盘上每条 run 记录的 `metrics.r2/mae/rmse` 与这里从预测重算的值，
             差必须 < 1e-6。这是"不要只信记录"那条要求的机器版。
    为什么两条都要：只有手算会漏"预测与记录对不上"，只有对拍会把"两边用了同一个错公式"
          当成通过。
    数据来源：手写数组 + `results/raw_runs/` + `results/aggregate/predictions/` + pack 的 `y`。
    副作用：只读。异常：无（缺数据走 skip）。
    被谁用：CI、D 的交接自检。

English:
    Two layers: hand-computed cases (so "the implementation is wrong" is distinguishable from "the
    data changed"), and a row-by-row comparison against the records (the machine version of "do not
    trust the record"). Both are needed: hand cases alone miss a prediction/record mismatch, while
    the comparison alone would pass if both sides shared the same wrong formula.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest

from chainworth.analysis import lambda_kappa as LK
from chainworth.metrics import regression as R

REPO_ROOT = Path(__file__).resolve().parents[1]
RAW_RUNS = REPO_ROOT / "results" / "raw_runs"
PREDICTIONS = REPO_ROOT / "results" / "aggregate" / "predictions"

# 标签不在任何 pack 里的任务：`pb26_arch_transfer_ea` 的标签与特征来自 `generated/` 的官方
# 物化产物（D-57 不入库），不在 `data/processed/features/` 下。这 15 条的对拍由
# `tests/test_arch_transfer.py` 负责（它读 arch 自己的 CSV），不在这里重复。
# Tasks whose labels live outside every pack: the arch task's labels come from the materialised
# `generated/` artifacts (D-57, not committed), and `tests/test_arch_transfer.py` already covers
# those fifteen records.
LABELS_OUTSIDE_PACKS = ("pb26_arch_transfer_ea",)


# --------------------------------------------------------------------------- #
# ① 手算对照 / Hand-computed cases
# --------------------------------------------------------------------------- #
def test_hand_computed_case_is_exact() -> None:
    """`[1,2,3]` 对 `[1,2,4]`：R² = 0.5、MAE = 1/3、RMSE = √(1/3)，逐位相等。
    The worked example: R^2 = 0.5, MAE = 1/3, RMSE = sqrt(1/3), bit for bit.

    中文：SS_res = 1（只有第三点差 1），SS_tot = 2（均值 2，离差 −1/0/+1），所以 R² = 1 − 1/2。
    English: SS_res = 1 and SS_tot = 2, so R^2 = 1 - 1/2.
    """
    truth = [1.0, 2.0, 3.0]
    pred = [1.0, 2.0, 4.0]
    assert R.r2(truth, pred) == pytest.approx(0.5, rel=0, abs=1e-15)
    assert R.mae(truth, pred) == pytest.approx(1.0 / 3.0, rel=0, abs=1e-15)
    assert R.rmse(truth, pred) == pytest.approx(math.sqrt(1.0 / 3.0), rel=0, abs=1e-15)


def test_perfect_prediction_gives_one_and_zero_error() -> None:
    """完全预测：R² = 1、MAE = RMSE = 0。/ A perfect prediction scores 1 and zero error."""
    truth = [0.0, 5.0, -3.0, 2.5]
    scores = R.all_metrics(truth, truth)
    assert scores["r2"] == 1.0
    assert scores["mae"] == 0.0
    assert scores["rmse"] == 0.0


def test_constant_label_gives_nan_not_zero() -> None:
    """测试集标签全相同 → R² 是 NaN，不是 0（0 会被读成"和均值一样差"）。
    A constant test label yields NaN, not 0, which would read as "as bad as the mean".
    """
    assert math.isnan(R.r2([2.0, 2.0, 2.0], [1.0, 2.0, 3.0]))
    assert math.isnan(R.all_metrics([2.0, 2.0, 2.0], [1.0, 2.0, 3.0])["r2"])
    # MAE / RMSE 仍然有定义，不许跟着变 NaN。
    assert R.mae([2.0, 2.0, 2.0], [1.0, 2.0, 3.0]) == pytest.approx(2.0 / 3.0)


def test_all_metrics_matches_the_individual_functions() -> None:
    """`all_metrics` 与三个单函数逐位相同（同一份口径，不许两条路各算一套）。
    `all_metrics` agrees with the single functions bit for bit.
    """
    rng = np.random.default_rng(7)
    truth = rng.normal(300.0, 40.0, 250)
    pred = truth + rng.normal(0.0, 5.0, 250)
    scores = R.all_metrics(truth, pred)
    assert scores["r2"] == R.r2(truth, pred)
    assert scores["mae"] == R.mae(truth, pred)
    assert scores["rmse"] == R.rmse(truth, pred)


def test_non_finite_values_are_dropped_not_poisoning_the_score() -> None:
    """一条 NaN 预测不许把整个 R² 变成 NaN：滤掉再算（与 `graph.metrics()` 同序）。
    One NaN prediction must not poison the whole score; non-finite rows are dropped first.
    """
    truth = [1.0, 2.0, 3.0, 4.0]
    pred = [1.0, float("nan"), 3.0, 4.0]
    assert R.r2(truth, pred) == 1.0
    assert R.mae(truth, pred) == 0.0


def test_length_mismatch_raises() -> None:
    """长度不等 → `MetricError`，不许悄悄截断（截断会让"算了几个点"变成一个谜）。
    A length mismatch raises; silent truncation would make "how many points" a mystery.
    """
    with pytest.raises(R.MetricError):
        R.r2([1.0, 2.0], [1.0])


def test_too_few_finite_points_raises() -> None:
    """有限点少于 2 个 → `MetricError`（一个点的 R² 是 0/0）。
    Fewer than two finite points raises: one point gives 0/0.
    """
    with pytest.raises(R.MetricError):
        R.r2([1.0, float("nan")], [1.0, 2.0])


# --------------------------------------------------------------------------- #
# ② 与盘上全部记录对拍 / Compare against every record on disk
# --------------------------------------------------------------------------- #
def test_recomputed_scores_match_every_run_record() -> None:
    """盘上每条记录：从预测重算的 R²/MAE/RMSE 与记录里的值，差 < 1e-6。
    Every record: recomputed scores agree with the recorded ones within 1e-6.

    中文：
        用途：D-1 ② 的原句——"差更多说明有一边错了，先查再报"。这里**先查**：谁差就点名。
        取数方式：按 `row_id` 回连（`scores_from_predictions`），不按位置。
        跳过：预测或 pack 缺失（干净 clone 上会出现；那是缺件，不是记错）；
              `pb26_arch_transfer_ea` 的 15 条显式排除——它的标签不在 pack 里，由
              `tests/test_arch_transfer.py` 覆盖。排除是**登记过**的名单，不是"取不到的就算了"。
    English:
        D-1 ② verbatim. Joined by `row_id`, never by position. Missing predictions or packs skip -
        that is a missing file, not a wrong record. The fifteen arch records are excluded through a
        registered list because their labels are not in a pack; `tests/test_arch_transfer.py`
        covers them.
    """
    records = LK.load_records()
    labels: dict[str, np.ndarray] = {}
    mismatches: list[str] = []
    checked = 0
    excluded = 0

    for run_id, record in sorted(records.items()):
        task = str(record.get("task", ""))
        if task in LABELS_OUTSIDE_PACKS:
            excluded += 1
            continue
        try:
            truth = LK._task_labels(task, labels, record.get("dataset_id"))
            row_id, y_pred = LK._prediction(run_id)
        except LK.LambdaError:
            continue
        recomputed = R.scores_from_predictions(truth, row_id, row_id, y_pred)
        for key in ("r2", "mae", "rmse"):
            recorded = float(record["metrics"][key])
            if np.isnan(recorded) and np.isnan(recomputed[key]):
                continue
            if abs(recorded - recomputed[key]) > 1e-6:
                mismatches.append(
                    f"{run_id} 的 {key}：记录 {recorded:.12f}，重算 {recomputed[key]:.12f}"
                )
        checked += 1

    if checked == 0:
        pytest.skip("没有可核的 run（缺预测或 pack）")
    assert not mismatches, f"重算与记录不一致 {len(mismatches)} 条：\n" + "\n".join(mismatches[:10])
    # 全部记录 − 15（arch，登记排除）应全部核到；少一条都说明有一类记录悄悄没进对拍。
    # 下限只防"几乎全漏"，不当作精确条数——精确条数只该有一处出处（`raw_runs/` 现场数）。
    assert checked >= 900, f"只核到 {checked} 条（排除 {excluded} 条），不像在核全量"
    assert excluded == 15, f"arch 的登记排除数变成 {excluded} 条，名单可能过期"


def test_row_join_rejects_out_of_range_ids() -> None:
    """`row_id` 越界 → 报错。按位置对齐会静默错位，按 id 对齐会当场炸。
    An out-of-range id raises; positional alignment would misalign silently.
    """
    truth = np.arange(5.0)
    with pytest.raises(R.MetricError):
        R.scores_from_predictions(truth, np.array([0, 1, 99]), np.array([0, 1, 99]), np.zeros(3))


def test_row_join_rejects_duplicate_ids() -> None:
    """`row_id` 重复 → 报错（bootstrap 会把同一个点重复计）。
    Duplicate ids raise, because the bootstrap would count one point twice.
    """
    truth = np.arange(5.0)
    with pytest.raises(R.MetricError):
        R.scores_from_predictions(truth, np.array([0, 0, 1]), np.array([0, 0, 1]), np.zeros(3))
