"""H2′ 双指标模块的测试：先合成曲线，再接真仓库。
Tests for the H2' dual-metric module: synthetic curves first, the real repo second.

中文：为什么要先合成——`kappa.bootstrap_kappa()` 那条教训（D-71）：点估计能跑通、区间路径
      取错形状，而真数据未齐时那条分支根本不执行，"没报错"被当成"对的"。本文件的合成用例
      都不依赖真数据，且**故意把比值造得已知**。
"""

from __future__ import annotations

import math

import pytest

from chainworth.analysis import metric_sensitivity as MS


def _cells(dataset: str, levels: dict[str, float], *, folds: int = 5, metric: str = "lambda"):
    """造一份 (数据集, 折, 档位) 读数；同一档位所有折取同一个值。
    Build one dataset's readings; every fold of a level carries the same value.
    """
    out: list[MS.Cell] = []
    for level, value in levels.items():
        for fold in range(folds):
            lam = value if metric == "lambda" else 0.0
            dr = value if metric == "delta_r2" else 0.0
            out.append(
                MS.Cell(
                    dataset=dataset,
                    fold=f"fold_{fold:02d}",
                    level=level,
                    n_train=float(level) if level.isdigit() else 1e5,
                    lambda_=lam,
                    delta_r2=dr,
                )
            )
    return out


def test_a_flat_curve_reads_ratio_one():
    """平坦曲线的首末比必须是 1.0（不是 0，也不是空）。"""
    cells = _cells("flat", {"300": 0.3, "1000": 0.3, "3000": 0.3})
    _, ratios = MS.level_and_ratio_rows(cells, n_boot=200)
    lam = next(r for r in ratios if r["metric"] == "lambda")
    assert lam["ratio"] == pytest.approx(1.0)
    assert lam["verdict"] == "h2p_not_halved"
    assert lam["halved"] == "false"


def test_a_quartering_curve_is_called_halved():
    """0.04 → 0.01 是四分之一，必须判 halved，且比值正好 0.25。"""
    cells = _cells("quarter", {"300": 0.04, "1000": 0.01}, metric="delta_r2")
    _, ratios = MS.level_and_ratio_rows(cells, n_boot=200)
    dr = next(r for r in ratios if r["metric"] == "delta_r2")
    assert dr["ratio"] == pytest.approx(0.25)
    assert dr["verdict"] == "h2p_halved"
    assert dr["halved"] == "true"


def test_ratio_is_exactly_half_is_not_halved():
    """边界：比值正好 0.5 **不算**减半（判据是严格小于）。"""
    cells = _cells("edge", {"300": 0.4, "1000": 0.2}, metric="delta_r2")
    _, ratios = MS.level_and_ratio_rows(cells, n_boot=200)
    dr = next(r for r in ratios if r["metric"] == "delta_r2")
    assert dr["ratio"] == pytest.approx(0.5)
    assert dr["verdict"] == "h2p_not_halved"


def test_a_negative_first_level_writes_undefined_not_zero():
    """首档为负时比值无定义：写空 + `undefined`，不写 0、不写 1。"""
    cells = _cells("neg", {"300": -0.2, "1000": 0.1})
    _, ratios = MS.level_and_ratio_rows(cells, n_boot=200)
    lam = next(r for r in ratios if r["metric"] == "lambda")
    assert lam["ratio"] == ""
    assert lam["verdict"] == "undefined"
    assert "≤ 0" in lam["note"]


def test_levels_are_ordered_by_rows_not_by_string():
    """档位顺序按行数排：'10000' 不能排到 '300' 前面（字符串序会排错）。"""
    cells = _cells("order", {"300": 0.3, "1000": 0.2, "10000": 0.1})
    _, ratios = MS.level_and_ratio_rows(cells, n_boot=50)
    lam = next(r for r in ratios if r["metric"] == "lambda")
    assert lam["level_first"] == "300"
    assert lam["level_last"] == "10000"
    # 表里比值保留 4 位小数，故用绝对容差而不是 1e-6
    assert lam["ratio"] == pytest.approx(0.1 / 0.3, abs=1e-4)


def test_folds_are_averaged_before_the_ratio():
    """比值用**折均值**，不是把 10 个点各自相除再平均。"""
    cells = [
        MS.Cell("mix", "fold_00", "300", 300.0, 0.40, 0.0),
        MS.Cell("mix", "fold_01", "300", 300.0, 0.20, 0.0),
        MS.Cell("mix", "fold_00", "1000", 1000.0, 0.10, 0.0),
        MS.Cell("mix", "fold_01", "1000", 1000.0, 0.50, 0.0),
    ]
    _, ratios = MS.level_and_ratio_rows(cells, n_boot=50)
    lam = next(r for r in ratios if r["metric"] == "lambda")
    assert lam["value_first"] == pytest.approx(0.30)
    assert lam["value_last"] == pytest.approx(0.30)
    assert lam["ratio"] == pytest.approx(1.0)


def test_decay_per_dex_matches_a_known_power_law():
    """种一条 y ∝ n^(−0.5)：每 10 倍数据比值应是 10^−0.5，decay_per_dex 应是 −0.5。"""
    cells = _cells(
        "power",
        {"100": 1.0, "1000": 10**-0.5, "10000": 10**-1.0},
        metric="delta_r2",
    )
    _, ratios = MS.level_and_ratio_rows(cells, n_boot=50)
    dr = next(r for r in ratios if r["metric"] == "delta_r2")
    assert dr["ratio"] == pytest.approx(0.1, rel=1e-6)
    assert dr["decay_per_dex"] == pytest.approx(-0.5, abs=1e-6)


def test_nan_rows_are_dropped_rather_than_poisoning_the_cell(tmp_path):
    """一行 nan 不能污染整格：丢掉它，档位均值只用剩下的折。"""
    path = tmp_path / "cells.csv"
    path.write_text(
        "\n".join(
            [
                "dataset,fold,tier,n_train,lambda,delta_r2,denominator",
                "d,fold_00,300,300,0.20,0.02,0.10",
                "d,fold_01,300,300,nan,nan,0.10",
                "d,fold_00,1000,1000,0.05,0.005,0.05",
                "d,fold_01,1000,1000,0.05,0.005,0.05",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    cells = MS.load_cells(path)
    assert len(cells) == 3, "nan 行必须被丢掉"
    _, ratios = MS.level_and_ratio_rows(cells, n_boot=50)
    lam = next(r for r in ratios if r["metric"] == "lambda")
    assert lam["value_first"] == pytest.approx(0.20), "300 档只有 1 个可用折"
    assert lam["ratio"] == pytest.approx(0.25)


def test_a_missing_file_raises_instead_of_returning_empty(tmp_path):
    """缺文件要抛，不返回空列表（空列表会被下游当成"没有发现"）。"""
    with pytest.raises(MS.MetricSensitivityError):
        MS.load_cells(tmp_path / "nope.csv")


def test_real_repo_reproduces_the_two_rulers():
    """真仓库：Λ 比值 0.650（未减半）、ΔR² 比值 0.167（已减半）——这就是 H2′ 的全部内容。"""
    cells = MS.load_cells()
    _, ratios = MS.level_and_ratio_rows(cells, n_boot=400)
    alt = {r["metric"]: r for r in ratios if r["dataset"] == "md5000_alternating_cp"}
    assert alt["lambda"]["ratio"] == pytest.approx(0.650, abs=0.01)
    assert alt["delta_r2"]["ratio"] == pytest.approx(0.167, abs=0.01)
    assert alt["lambda"]["verdict"] == "h2p_not_halved"
    assert alt["delta_r2"]["verdict"] == "h2p_halved"
    # 两把尺子看同一条曲线，方向相反——这就是 H2′ 要并列报的理由
    assert alt["lambda"]["ratio"] > 0.5 > alt["delta_r2"]["ratio"]


def test_real_repo_interval_brackets_the_point_estimate():
    """区间要包住点估计；5 折重采样得到的区间必然窄，但不得空。"""
    cells = MS.load_cells()
    _, ratios = MS.level_and_ratio_rows(cells, n_boot=400)
    for row in ratios:
        if row["ratio"] == "":
            continue
        lo, hi = float(row["ratio_lo"]), float(row["ratio_hi"])
        assert math.isfinite(lo) and math.isfinite(hi)
        assert lo <= float(row["ratio"]) <= hi
