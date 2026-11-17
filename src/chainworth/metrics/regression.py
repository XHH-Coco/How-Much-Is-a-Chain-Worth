"""三个回归指标，独立实现一遍——用来和 run 记录里那份对拍，不是复用它。
Three regression metrics, implemented independently - for cross-checking the copy inside the run
records, not for reusing it.

中文：
    用途：分工 D-1 的 ①②。**"不要只信 run 记录里的数"** 是这一项的原文，所以这里刻意不 import
          `chainworth.arms.graph.metrics()`：两份实现各自写一遍，差超过 1e-6 才说明有一边错了。
          如果共用一份实现，这条对拍就退化成"函数调用自己"，永远相等、永远无信息。
    口径（与 `graph.metrics()` 必须一致，否则对拍没有意义）：
          - **R² 用 `1 − SS_res / SS_tot`，分母是测试集自身的总体均值**（不是训练均值）；
          - `SS_tot = 0`（测试集标签全相同）时 R² 是 **NaN**，不是 0——0 会被读成"和均值一样差"，
            NaN 会被读成"这一条不可用"，后者才是真的；
          - 先滤掉非有限值（预测里的 NaN / inf）再算，与 `graph.metrics()` 同序。
    实现上的差异（这是"独立"的实质）：这里用 `np.average` 与显式中心化，不用 `**2` 之外的花样；
          `graph.metrics()` 用 `np.sum(residual**2)`。两条路在数值上等价，但**写法不同**，
          所以一边写错时另一边不会跟着错。
    数据来源：调用方给的数组。配置来源：无。副作用：无。
    对外：`r2` / `mae` / `rmse` / `all_metrics` / `scores_from_predictions`。
    被谁用：`chainworth.analysis.lambda_kappa`、`chainworth.analysis.ci`、`tests/test_metrics.py`。

English:
    Purpose: D-1 ①②. "Do not trust the numbers inside the run records" is the requirement, so this
    module deliberately does **not** import `chainworth.arms.graph.metrics()`; two independent
    implementations, and a difference above 1e-6 means one of them is wrong. Sharing one
    implementation would turn the cross-check into a function calling itself.
    Definitions (must match `graph.metrics()` or the comparison is meaningless): R^2 is
    `1 - SS_res / SS_tot` with the **test set's own** population mean; `SS_tot == 0` gives NaN, not
    0; non-finite values are filtered first.
    No data source, no config, no side effects.
"""

from __future__ import annotations

from typing import Iterable

import numpy as np

# 非有限值滤除后少于这么多点，指标就不该被当成结论。D 的图里禁止出现以少于此数支撑的点。
# Below this many finite points a metric must not be treated as a result.
MIN_POINTS = 2


class MetricError(ValueError):
    """指标算不了（空输入、长度不等、有限点太少）。/ The metric cannot be computed."""


def _as_pair(y_true: Iterable[float], y_pred: Iterable[float]) -> tuple[np.ndarray, np.ndarray]:
    """转成 float64 数组、检查长度、滤掉非有限值。/ Cast, check the length, drop non-finite rows.

    中文：输入两条等长序列；输出两个 1-D float64 数组（同长、全有限）。
          副作用：无。异常：`MetricError` —— 长度不等、滤完不足 `MIN_POINTS` 个点。
    English: Args: two equal-length sequences. Returns two finite float64 arrays. Raises
    `MetricError` on a length mismatch or too few finite points.
    """
    truth = np.asarray(list(y_true), dtype=np.float64).ravel()
    pred = np.asarray(list(y_pred), dtype=np.float64).ravel()
    if truth.size != pred.size:
        raise MetricError(f"长度不等：y_true {truth.size} vs y_pred {pred.size}")
    finite = np.isfinite(truth) & np.isfinite(pred)
    truth, pred = truth[finite], pred[finite]
    if truth.size < MIN_POINTS:
        raise MetricError(f"有限点只有 {truth.size} 个，不够算指标（下限 {MIN_POINTS}）")
    return truth, pred


def r2(y_true: Iterable[float], y_pred: Iterable[float]) -> float:
    """`1 − SS_res / SS_tot`，分母用测试集自身均值；`SS_tot = 0` 返回 NaN。
    `1 - SS_res / SS_tot` with the test set's own mean; NaN when `SS_tot` is zero.

    中文：输入真实值与预测值；输出 float。副作用：无。异常：`MetricError`。
    English: Args: truth, prediction. Returns a float. Raises `MetricError`.
    """
    truth, pred = _as_pair(y_true, y_pred)
    centred = truth - np.average(truth)
    ss_tot = float(np.dot(centred, centred))
    if ss_tot == 0.0:
        # 测试集标签全相同：R² 没有定义。给 NaN，不给 0（见模块头）。
        return float("nan")
    residual = truth - pred
    ss_res = float(np.dot(residual, residual))
    return 1.0 - ss_res / ss_tot


def mae(y_true: Iterable[float], y_pred: Iterable[float]) -> float:
    """平均绝对误差。/ Mean absolute error.

    中文：输入真实值与预测值；输出 float。副作用：无。异常：`MetricError`。
    """
    truth, pred = _as_pair(y_true, y_pred)
    return float(np.average(np.abs(truth - pred)))


def rmse(y_true: Iterable[float], y_pred: Iterable[float]) -> float:
    """均方根误差。/ Root mean squared error.

    中文：输入真实值与预测值；输出 float。副作用：无。异常：`MetricError`。
    """
    truth, pred = _as_pair(y_true, y_pred)
    residual = truth - pred
    return float(np.sqrt(np.average(residual * residual)))


def all_metrics(y_true: Iterable[float], y_pred: Iterable[float]) -> dict[str, float]:
    """一次算三个，键名与 run 记录一致（`r2` / `mae` / `rmse`）。
    All three at once, keyed exactly like the run records.

    中文：输入真实值与预测值；输出 dict。副作用：无。异常：`MetricError`。
          只做一次长度检查与一次滤值，比连调三次省一半。
    English: One length check and one filter instead of three.
    """
    truth, pred = _as_pair(y_true, y_pred)
    centred = truth - np.average(truth)
    ss_tot = float(np.dot(centred, centred))
    residual = truth - pred
    ss_res = float(np.dot(residual, residual))
    return {
        "r2": float("nan") if ss_tot == 0.0 else 1.0 - ss_res / ss_tot,
        "mae": float(np.average(np.abs(residual))),
        "rmse": float(np.sqrt(np.average(residual * residual))),
    }


def scores_from_predictions(
    y_true: np.ndarray,
    row_id: np.ndarray,
    predicted_row_id: np.ndarray,
    y_pred: np.ndarray,
) -> dict[str, float]:
    """按 `row_id` 把预测对回标签，再算指标——**不是按位置**。
    Join predictions back to labels **by `row_id`**, not by position, then score.

    中文：
        用途：这是 D 唯一允许的取数方式。预测 parquet 里的 `row_id` 是该任务的行号（0..n-1），
              而记录里的指标是用 `data.y[test_index]` 算的；按位置对齐在"折的测试集不是整表"
              时会**静默错位**（B-4 踩过行号对齐的坑）。这里先建索引再取，错位会当场变成
              "row_id 找不到"，而不是得到一个看起来正常的分数。
        输入：y_true（整表标签，按行号索引）、row_id（预测文件里的 row_id）、
              predicted_row_id（同上，显式分开只为让调用方看清"哪个是哪个"）、y_pred。
        输出：dict（`r2` / `mae` / `rmse`）。数据来源：调用方。
        副作用：无。异常：`MetricError` —— row_id 越界或重复。
    English:
        The only allowed way to score predictions: join on `row_id`, because scoring by position
        silently misaligns whenever the fold's test set is not the whole table (the B-4 row-order
        trap). An out-of-range or duplicate id raises instead of producing a plausible score.
    """
    del predicted_row_id  # 同一个东西，显式参数只为让调用点读得懂
    truth = np.asarray(y_true, dtype=np.float64)
    ids = np.asarray(row_id, dtype=np.int64)
    if ids.size != np.asarray(y_pred).size:
        raise MetricError(f"row_id {ids.size} 个，预测 {np.asarray(y_pred).size} 个，不等长")
    if np.unique(ids).size != ids.size:
        raise MetricError("row_id 有重复：重采样会把同一个点重复计")
    if ids.min() < 0 or ids.max() >= truth.size:
        raise MetricError(f"row_id 越界：范围 [{ids.min()}, {ids.max()}]，表长 {truth.size}")
    return all_metrics(truth[ids], np.asarray(y_pred, dtype=np.float64))
