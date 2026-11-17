"""Λ 与 ΔR² 的区间：对测试集残差做 percentile bootstrap，每次重采样都重算 R²。
Intervals for Lambda and dR2: a percentile bootstrap over test-set residuals, recomputing R^2 on
every resample.

中文：
    用途：分工 D-2。三条规矩决定了这里不能图省事：
      ① **重采样测试残差，不是重采样种子。** 种子只给"run 间方差"，那是另一件事（单独报）。
         只重采样种子，区间会窄到把噪声当信号（分工 D §7 的坑表）。
      ② **每次重采样都重算 R²，再代进 Λ。** Λ 是比值，分布是偏的；对 Λ 直接做正态近似
         （`λ ± 1.96·se`）会在分母小的时候给出越界区间——而"分母小"正是最需要区间的区域。
      ③ **两条臂共用同一组重采样下标。** Λ 是配对量：同一个测试点在两条臂上的残差要一起被抽中，
         否则抽出来的 Λ 分布里混着"两条臂看的不是同一批点"这个额外方差，区间会虚宽。
    随机性：`np.random.default_rng(seed)`；同一 seed 两次调用**逐位相同**（有测试守着）。
    数据来源：调用方给的三条等长数组（真实值、unit 臂预测、chain 臂预测）。
    配置来源：`BOOTSTRAP_DRAWS` / `BOOTSTRAP_SEED` / `INTERVAL`（B=2000、seed 固定，分工 D §5 的冻结项）。
    副作用：无（纯函数）。异常：`BootstrapError` —— 长度不等或点数太少。
    对外：`bootstrap_lambda()` / `percentile_interval()`。
    被谁用：`scripts/06c_lambda.py`、`tests/test_bootstrap_repro.py`。

English:
    Purpose: D-2. Three rules forbid shortcuts: (1) resample **test residuals**, not seeds - seed
    variance is a different quantity and is reported separately; (2) recompute R^2 inside every
    resample, because Lambda is a ratio and a normal approximation breaks exactly where the
    denominator is small, i.e. where the interval matters most; (3) the two arms share the same
    resample indices, because Lambda is paired - drawing them independently adds the "not the same
    points" variance and inflates the interval.
    Reproducibility: a fixed `default_rng(seed)` gives bit-identical results across calls.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

# 冻结项（分工 D §5）：B 与随机种子只能整体换版本，不能"这个点多重采几次"。
# Frozen (division D section 5): B and the seed change only as a whole version.
BOOTSTRAP_DRAWS = 2000
BOOTSTRAP_SEED = 20261117
INTERVAL = 0.95
# 一次矩阵运算最多处理这么多重采样，控制内存（B x n 的 float64）。
# Resamples per matrix block, to bound memory (B x n float64).
CHUNK = 200


class BootstrapError(ValueError):
    """重采样做不了（长度不等、点数太少）。/ The bootstrap cannot run."""


@dataclass(frozen=True)
class BootstrapResult:
    """点估计 + 区间。/ Point estimate plus interval.

    中文：字段 `estimate` / `lo` / `hi` / `n_boot` / `n_points` / `seed`。
          退化时（所有重采样都算不出）`lo` / `hi` 为 NaN，`n_boot` 记实际成功的次数。
    English: `n_boot` counts the resamples that produced a finite value; a degenerate case leaves
    `lo`/`hi` as NaN rather than inventing a range.
    """

    estimate: float
    lo: float
    hi: float
    n_boot: int
    n_points: int
    seed: int


def percentile_interval(values: Sequence[float], level: float = INTERVAL) -> tuple[float, float]:
    """percentile 区间（忽略 NaN）。/ Percentile interval, NaN dropped.

    中文：输入样本与置信水平；输出 (lo, hi)。全 NaN 时返回 (nan, nan)。
    English: Args: samples, level. Returns (lo, hi); all-NaN returns (nan, nan).
    """
    sample = np.asarray([value for value in values if np.isfinite(value)], dtype=np.float64)
    if sample.size == 0:
        return float("nan"), float("nan")
    tail = (1.0 - level) / 2.0
    return (
        float(np.quantile(sample, tail)),
        float(np.quantile(sample, 1.0 - tail)),
    )


def _r2_rows(truth: np.ndarray, pred: np.ndarray) -> np.ndarray:
    """按行算 R²：`truth`/`pred` 是 (B, n)，返回 (B,)。
    Row-wise R^2 for (B, n) matrices; returns (B,).

    中文：与 `metrics.regression.r2` 同一口径（分母用每行自己的均值）；`SS_tot = 0` 的行给 NaN。
    English: same definition as `metrics.regression.r2`, with each row's own mean; a zero
    denominator yields NaN for that row.
    """
    residual = truth - pred
    ss_res = np.einsum("ij,ij->i", residual, residual)
    centred = truth - truth.mean(axis=1, keepdims=True)
    ss_tot = np.einsum("ij,ij->i", centred, centred)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(ss_tot == 0.0, np.nan, 1.0 - ss_res / ss_tot)


def bootstrap_lambda(
    y_true: np.ndarray,
    y_pred_unit: np.ndarray,
    y_pred_chain: np.ndarray,
    *,
    draws: int = BOOTSTRAP_DRAWS,
    seed: int = BOOTSTRAP_SEED,
    level: float = INTERVAL,
) -> tuple[BootstrapResult, BootstrapResult]:
    """对 (真实值, unit 预测, chain 预测) 做配对 bootstrap，返回 (Λ 结果, ΔR² 结果)。
    Paired bootstrap over (truth, unit prediction, chain prediction); returns (Lambda, dR2).

    中文：
        输入：三条等长 1-D 数组；draws / seed / level（冻结项，改就是换版本）。
        输出：两个 `BootstrapResult`。数据来源：调用方（来自预测 parquet 与 pack 的 `y`）。
        副作用：无。异常：`BootstrapError` —— 长度不等或点数 < 3。
        实现要点：**同一组下标同时抽两条臂**（见模块头 ③）；每块 `CHUNK` 次重采样一起算，
                   内存与点数无关；`SS_tot = 0` 的重采样被跳过而不是当 0。
    English:
        Args: three equal-length 1-D arrays plus the frozen draws/seed/level. Returns the Lambda
        and dR2 results. The same indices drive both arms (module header ③); each block of CHUNK
        resamples is computed at once so memory does not scale with the point count; resamples with
        a zero denominator are dropped rather than treated as zero.
    """
    truth = np.asarray(y_true, dtype=np.float64)
    unit = np.asarray(y_pred_unit, dtype=np.float64)
    chain = np.asarray(y_pred_chain, dtype=np.float64)
    if not (truth.size == unit.size == chain.size):
        raise BootstrapError(
            f"三条数组不等长：{truth.size} / {unit.size} / {chain.size}"
        )
    if truth.size < 3:
        raise BootstrapError(f"只有 {truth.size} 个测试点，bootstrap 没有意义")

    rng = np.random.default_rng(seed)
    n = truth.size
    lambdas: list[np.ndarray] = []
    deltas: list[np.ndarray] = []
    done = 0
    while done < draws:
        size = min(CHUNK, draws - done)
        indices = rng.integers(0, n, size=(size, n))
        r2_unit = _r2_rows(truth[indices], unit[indices])
        r2_chain = _r2_rows(truth[indices], chain[indices])
        delta = r2_chain - r2_unit
        with np.errstate(divide="ignore", invalid="ignore"):
            lam = delta / (1.0 - r2_unit)
        lambdas.append(lam)
        deltas.append(delta)
        done += size

    lambda_sample = np.concatenate(lambdas)
    delta_sample = np.concatenate(deltas)

    r2_unit_point = float(_r2_rows(truth[None, :], unit[None, :])[0])
    r2_chain_point = float(_r2_rows(truth[None, :], chain[None, :])[0])
    delta_point = r2_chain_point - r2_unit_point
    with np.errstate(divide="ignore", invalid="ignore"):
        lambda_point = float(delta_point / (1.0 - r2_unit_point))

    def _pack(sample: np.ndarray, estimate: float) -> BootstrapResult:
        finite = sample[np.isfinite(sample)]
        lo, hi = percentile_interval(finite, level)
        return BootstrapResult(
            estimate=estimate, lo=lo, hi=hi, n_boot=int(finite.size), n_points=n, seed=seed
        )

    return _pack(lambda_sample, lambda_point), _pack(delta_sample, delta_point)
