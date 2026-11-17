"""bootstrap 的三件事：同种子逐位可复现、重采样的是残差不是种子、配对抽下标。
Bootstrap: bit-identical under a fixed seed, resampling residuals rather than seeds, and paired
indices across the two arms.

中文：
    用途：分工 D-2 的验收 ①。这一项最容易被糊弄过去——"跑两次差不多"听上去就够了，
          但"差不多"没法区分"实现稳定"与"我碰巧抽到相近的数"。
    数据来源：合成数组（这一项不需要真数据）。副作用：无。

English:
    Purpose: D-2 ①. This is the item most easily faked - "two runs look close" sounds sufficient,
    but "close" cannot tell a stable implementation from a lucky pair of draws.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from chainworth.analysis import ci as CI


def _sample(n: int = 400, noise: float = 25.0, seed: int = 3) -> tuple[np.ndarray, ...]:
    """造一组"链臂略差于单元臂"的合成预测。/ Synthetic predictions with a slightly worse chain
    arm.
    """
    rng = np.random.default_rng(seed)
    truth = rng.normal(300.0, 60.0, n)
    unit = truth + rng.normal(0.0, 40.0, n)
    chain = unit + rng.normal(0.0, noise, n)
    return truth, unit, chain


def test_same_seed_is_bit_identical() -> None:
    """同一个种子跑两次，点估计与区间**逐位相同**（不是"约等于"）。
    Two runs with the same seed agree bit for bit, including the interval ends.
    """
    truth, unit, chain = _sample()
    first, first_delta = CI.bootstrap_lambda(truth, unit, chain)
    second, second_delta = CI.bootstrap_lambda(truth, unit, chain)
    assert first == second
    assert first_delta == second_delta


def test_different_seed_moves_the_interval_but_not_the_point_estimate() -> None:
    """换种子只动区间的端点，点估计一个字节都不许动。
    A different seed moves the interval ends, never the point estimate.

    中文：点估计是从真实预测直接算的，与重采样无关；它要是变了，说明实现把两种东西混在一起了。
    English: the point estimate is computed directly from the predictions; if it moved, the
    implementation is mixing two different things.
    """
    truth, unit, chain = _sample()
    base, _ = CI.bootstrap_lambda(truth, unit, chain, seed=1)
    other, _ = CI.bootstrap_lambda(truth, unit, chain, seed=2)
    assert base.estimate == other.estimate
    assert (base.lo, base.hi) != (other.lo, other.hi)


def test_interval_brackets_the_point_estimate() -> None:
    """区间必须包住点估计——包不住说明两边算的不是同一个量。
    The interval must bracket the point estimate; otherwise the two sides compute different things.
    """
    truth, unit, chain = _sample()
    lam, delta = CI.bootstrap_lambda(truth, unit, chain)
    assert lam.lo <= lam.estimate <= lam.hi
    assert delta.lo <= delta.estimate <= delta.hi


def test_identical_arms_give_exactly_zero() -> None:
    """两条臂预测完全相同时 Λ 恒为 0，区间退化成 0——这是"配对抽下标"的直接证据。
    With identical predictions Lambda is exactly zero and the interval collapses to zero; that is
    the direct evidence that both arms share the same resample indices.
    """
    truth, unit, _ = _sample()
    lam, delta = CI.bootstrap_lambda(truth, unit, unit)
    assert lam.estimate == 0.0
    assert lam.lo == 0.0 and lam.hi == 0.0
    assert delta.estimate == 0.0


def test_independent_resampling_would_widen_the_interval() -> None:
    """反证：若两条臂各抽各的下标，同一份数据给出的区间会更宽。
    Counter-check: drawing indices independently for each arm widens the interval on the same data.

    中文：
        这条测的是**方法本身**，不是实现的输出：手工跑一趟"独立重采样"，比较区间宽度。
        如果两种抽法给出几乎一样的宽度，说明配对与否不影响区间——那么模块头 ③ 那条论证
        就是空的，应该删掉而不是留着当装饰。
    English:
        This tests the *method*, not the output: run an independent-index bootstrap by hand and
        compare widths. If the widths were the same, the pairing argument in the module header
        would be empty and should be deleted rather than kept as decoration.
    """
    truth, unit, chain = _sample(n=300, noise=45.0, seed=11)
    paired, _ = CI.bootstrap_lambda(truth, unit, chain, draws=600, seed=5)

    rng = np.random.default_rng(5)
    n = truth.size
    values = []
    for _ in range(600):
        left = rng.integers(0, n, n)
        right = rng.integers(0, n, n)
        # 两条臂各用各的下标（这是"错"的做法，只为量出差多少）。
        r2_unit = CI._r2_rows(truth[left][None, :], unit[left][None, :])[0]
        r2_chain = CI._r2_rows(truth[right][None, :], chain[right][None, :])[0]
        with np.errstate(divide="ignore", invalid="ignore"):
            values.append((r2_chain - r2_unit) / (1.0 - r2_unit))
    lo, hi = CI.percentile_interval(values)
    independent_width = hi - lo
    paired_width = paired.hi - paired.lo

    assert paired_width > 0 and independent_width > 0
    assert independent_width > paired_width, (
        f"独立抽样的区间（{independent_width:.4f}）没有比配对抽样（{paired_width:.4f}）更宽——"
        "配对抽下标这条论证需要重写"
    )


def test_constant_labels_do_not_crash() -> None:
    """标签全相同（`SS_tot = 0`）时每一次重采样都算不出 R²，结果是 NaN 区间而不是异常。
    Constant labels make every resample undefined; the result is a NaN interval, not an exception.
    """
    truth = np.full(50, 7.0)
    unit = np.linspace(0.0, 10.0, 50)
    chain = unit + 0.5
    lam, delta = CI.bootstrap_lambda(truth, unit, chain, draws=50)
    assert math.isnan(lam.lo) and math.isnan(lam.hi)
    assert lam.n_boot == 0


def test_too_few_points_raises() -> None:
    """两个点做不了 bootstrap（`SS_tot` 可能为 0，重采样全是同一对点）。
    Two points cannot support a bootstrap.
    """
    with pytest.raises(CI.BootstrapError):
        CI.bootstrap_lambda(np.array([1.0, 2.0]), np.array([1.0, 2.0]), np.array([1.5, 2.5]))


def test_length_mismatch_raises() -> None:
    """三条数组不等长 → `BootstrapError`，不许截到最短的那种"宽容"。
    Unequal lengths raise; truncating to the shortest would hide a misalignment.
    """
    with pytest.raises(CI.BootstrapError):
        CI.bootstrap_lambda(np.arange(10.0), np.arange(9.0), np.arange(10.0))
