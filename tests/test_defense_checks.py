"""答辩补算的测试：合成用例守住公式，再接真仓库守住口径。
Tests for the defence checks: synthetic cases guard the formulas, one repo case guards the convention.

中文：为什么要一条"接真仓库"的用例——`defense_checks.spearman_rho()` 若改用
      `1−6Σd²/n(n²−1)`，在 tier 有并列时会给出 0.043，而 `h1_ordering.csv` 里是 0.000。
      两者都"能跑、不报错"，只有逐位比对能发现差在哪儿（D-71 的同型教训）。
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from chainworth.analysis import defense_checks as DC

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_spearman_reproduces_the_published_rho():
    """模块的 ρ 必须与 `h1_ordering.csv` 的汇总行逐位一致（并列用平均秩）。"""
    rows = list(csv.DictReader((REPO_ROOT / "results/aggregate/h1_ordering.csv")
                               .open(newline="", encoding="utf-8")))
    for arm, expected in (("a3_graph", 0.000), ("a4_polybert", 0.239046)):
        points = [r for r in rows if r["scope"] == "point" and r["arm"] == arm]
        rho = DC.spearman_rho([int(r["tier"]) for r in points],
                              [float(r["lambda"]) for r in points])
        assert rho == pytest.approx(expected, abs=1e-4)


def test_permutation_p_brackets_the_two_extremes():
    """完全单调 → p 很小；完全无关联 → p 接近 1。这是 p 值的定义，不是调参。"""
    tiers = [1, 1, 2, 2, 3, 3]
    perfect = [1, 2, 3, 4, 5, 6]
    p_perfect = DC.permutation_p(DC.spearman_rho(tiers, perfect), tiers)
    assert p_perfect["method"] == "exhaustive"
    assert p_perfect["p_two_sided"] <= 0.05
    p_null = DC.permutation_p(0.0, tiers)
    assert p_null["p_two_sided"] == pytest.approx(1.0)


def test_segment_slope_is_per_decade():
    """跨 10 倍数据、值从 0.4 降到 0.2 → 斜率 −0.2 / dex（不是 −0.2 / 行）。"""
    curve = [{"n_train": 300.0, "lambda": 0.4, "delta_r2": 0.04},
             {"n_train": 3000.0, "lambda": 0.2, "delta_r2": 0.02}]
    assert DC.segment_slope(curve, 300, 3000) == pytest.approx(-0.2)
    assert DC.segment_slope(curve, 300, 3000, "delta_r2") == pytest.approx(-0.02)


def test_slope_outside_the_curve_is_none_not_zero():
    """档位不在曲线上 → None。填 0 会把"没这个点"伪装成"斜率为零"。"""
    curve = [{"n_train": 300.0, "lambda": 0.4}]
    assert DC.segment_slope(curve, 300, 3000) is None


def test_crossing_is_interpolated_in_log_space():
    """log 空间插值：30 处 +1、120 处 −1 → 过零点在几何中点 60，不是算术中点 75。"""
    out = DC.log_crossing(30, 1.0, 120, -1.0)
    assert out["n_star"] == pytest.approx(60.0, rel=1e-9)
    assert out["method"] == "log_linear_interpolation"


def test_no_sign_change_is_reported_as_no_crossing():
    """两端同号 → n\\* 为 None（D-5：窗内不过零就不外推）。"""
    out = DC.log_crossing(30, -0.09, 120, -0.38)
    assert out["n_star"] is None
    assert out["method"] == "no_crossing_in_window"


def test_sign_flips_separate_near_zero_from_real_disagreement():
    """4 处翻号里 2 处两侧都 |Λ|<0.05：近零格翻号不是"两臂矛盾"。"""
    rows = [
        {"task": "a", "arm": "a3_graph", "lambda_mean": -0.004},
        {"task": "a", "arm": "a4_polybert", "lambda_mean": 0.008},
        {"task": "b", "arm": "a3_graph", "lambda_mean": 0.638},
        {"task": "b", "arm": "a4_polybert", "lambda_mean": -0.532},
    ]
    out = DC.arm_agreement(rows, near_zero=0.05)
    assert out["n_shared_tasks"] == 2
    assert out["n_sign_flips"] == 2
    assert out["n_flips_near_zero"] == 1


def test_floor_takes_the_larger_of_the_two_conditions():
    """地板 = 两种零信息条件 |ΔR²| 的较大者；取较小者会把 0.0195 报成 0.0015。"""
    summary = {"tasks": {"t": {"n_folds": 1, "conditions": {
        "constant3": {"delta_r2_vs_baseline_mean": -0.0195},
        "shuffled3": {"delta_r2_vs_baseline_mean": 0.0015},
    }}}}
    out = DC.floor_table(summary)
    assert out["rows"][0]["floor"] == pytest.approx(0.0195)


def test_unstable_rows_are_dropped_from_the_stable_recount():
    """分母 < 0.05 的行不进"稳定格"重算；整格没有稳定行时写 None，不写 0。"""
    cells = [
        DC.Cell("t", "a3_graph", "f", "s0", 0.99, 0.8, 0.1, "ok"),
        DC.Cell("t", "a3_graph", "f", "s1", 0.98, float("nan"), 0.1, "ok"),
        DC.Cell("u", "a3_graph", "f", "s0", 0.90, -0.2, 0.1, "ok"),
    ]
    rows = {r["task"]: r for r in DC.cell_summary(cells)}
    assert rows["t"]["n_reportable"] == 1
    assert rows["t"]["n_stable"] == 0
    assert rows["t"]["lambda_mean_stable"] is None
    assert rows["u"]["lambda_mean_stable"] == pytest.approx(-0.2)


def test_written_summary_carries_its_sources():
    """落盘 JSON 必须带 sources，且每个数指回表名——这是"可复核"的最低条件。"""
    payload = DC.build()
    assert "sources" in payload["summary"]
    assert payload["summary"]["sources"]["lambda"].endswith("lambda_ci.csv")
    json.dumps(payload["summary"])  # 必须可序列化，不靠 numpy 标量
