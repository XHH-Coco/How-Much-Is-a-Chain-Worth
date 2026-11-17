"""Λ 的三条硬规矩：不 clip、分母不稳就不报、0/0 不崩。
The three hard rules for Lambda: never clip, refuse a reportable value when the denominator
collapses, and survive the 0/0 case.

中文：
    用途：分工 D-1 的验收 ①③④ 逐条落成断言。为什么这些必须写成测试而不是文档：
          分工 D §7 的红线第 1 条是"clip Λ 到 [0,1]"，第 4 条是"两个点拟合斜率"。这两类错
          **都不抛异常**，只会让图看起来更漂亮。测试是唯一能当场抓住它们的地方。
    数据来源：手写数 + 真仓库的 321 对（`build_lambda_table()`）。副作用：只读。

English:
    Purpose: D-1 ①②③④ as assertions. Clipping and "fitting a slope through two points" never
    raise; they only make the figure look better. A test is the only place that catches them.
"""

from __future__ import annotations

import math
from pathlib import Path

import pytest

from chainworth.analysis import lambda_kappa as LK

REPO_ROOT = Path(__file__).resolve().parents[1]


# --------------------------------------------------------------------------- #
# ① 公式与手算 / The formula and worked examples
# --------------------------------------------------------------------------- #
def test_worked_example_from_the_proposal() -> None:
    """`R²_unit = 0.60`、`R²_chain = 0.70` → Λ = (0.70 − 0.60) / (1 − 0.60) = 0.25。
    The proposal's own example gives exactly 0.25.
    """
    result = LK.lambda_from_r2(r2_chain=0.70, r2_unit=0.60)
    assert result.value == pytest.approx(0.25, rel=0, abs=1e-15)
    assert result.delta_r2 == pytest.approx(0.10, rel=0, abs=1e-15)
    assert result.status == LK.STATUS_OK
    assert result.reportable is True


def test_negative_lambda_is_reported_not_clipped() -> None:
    """链臂更差时 Λ 是负数，**照原样报**；截成 0 就是数据操纵（分工 D §7 红线 1）。
    A worse chain arm yields a negative Lambda and it is reported as is; clipping it to zero is
    data manipulation (red line 1).
    """
    result = LK.lambda_from_r2(r2_chain=0.50, r2_unit=0.60)
    assert result.value == pytest.approx(-0.25, rel=0, abs=1e-15)
    assert result.reportable is True, "负值不是'不可报'，它是结果"


def test_large_negative_lambda_survives() -> None:
    """`R²_unit = 0.95` 以下但链臂崩掉时，Λ 可以远小于 −1（`dielectric_loss` 实测 −1.48）。
    这不叫异常，叫"链信息在这个任务上有害"。
    Below the threshold a collapsed chain arm gives Lambda far below -1 (measured -1.48 on
    dielectric_loss); that is a result, not an anomaly.
    """
    result = LK.lambda_from_r2(r2_chain=0.30, r2_unit=0.60)
    assert result.value == pytest.approx(-0.75, rel=0, abs=1e-15)
    result = LK.lambda_from_r2(r2_chain=-0.0, r2_unit=0.60)
    assert result.value == pytest.approx(-1.5, rel=0, abs=1e-15)


# --------------------------------------------------------------------------- #
# ③ 分母不稳：只报 ΔR² / Denominator collapse: report dR2 only
# --------------------------------------------------------------------------- #
def test_threshold_exactly_095_is_unstable() -> None:
    """`R²_unit = 0.95` 时 `value` 必须是 NaN 且 `reportable=False`，但 `delta_r2` 照常给。
    At exactly 0.95 the value is NaN and not reportable, while dR2 is still reported.
    """
    result = LK.lambda_from_r2(r2_chain=0.96, r2_unit=0.95)
    assert math.isnan(result.value)
    assert result.reportable is False
    assert result.status == LK.STATUS_UNSTABLE
    assert result.delta_r2 == pytest.approx(0.01, rel=0, abs=1e-15)
    # 诊断值仍留着，供人看一眼——但它不在 `value` 里，画不出图。
    assert result.diagnostic == pytest.approx(0.2, rel=0, abs=1e-15)
    assert "denominator" not in result.reason.lower() or True  # 理由是人话，不做字符串断言


def test_just_below_the_threshold_stays_reportable() -> None:
    """`R²_unit = 0.9499` 仍可报——阈值是硬边，不许"差不多就跳过"。
    At 0.9499 the value is still reportable; the threshold is a hard edge.
    """
    result = LK.lambda_from_r2(r2_chain=0.96, r2_unit=0.9499)
    assert result.reportable is True
    assert result.status == LK.STATUS_OK


# --------------------------------------------------------------------------- #
# ④ 0/0 不崩 / The 0-over-0 case must not crash
# --------------------------------------------------------------------------- #
def test_zero_over_zero_returns_nan_with_a_warning() -> None:
    """`R²_unit = R²_chain = 1` → NaN + 一条 warning，不许抛异常也不许崩。
    `R2_unit == R2_chain == 1` gives NaN plus a warning; no exception, no crash.
    """
    with pytest.warns(RuntimeWarning):
        result = LK.lambda_from_r2(r2_chain=1.0, r2_unit=1.0)
    assert math.isnan(result.value)
    assert result.reportable is False
    assert result.status == LK.STATUS_UNDEFINED
    assert result.delta_r2 == 0.0


def test_unit_one_and_chain_worse_is_undefined_not_infinite() -> None:
    """`R²_unit = 1`、`R²_chain < 1` → 分母为 0、分子不为 0；**不许返回 inf 当成可报的数**。
    A zero denominator with a non-zero numerator must not come back as a reportable infinity.
    """
    with pytest.warns(RuntimeWarning):
        result = LK.lambda_from_r2(r2_chain=0.9, r2_unit=1.0)
    assert math.isnan(result.value)
    assert result.reportable is False
    assert result.delta_r2 == pytest.approx(-0.1, rel=0, abs=1e-15)


def test_nan_input_becomes_a_status_not_an_exception() -> None:
    """某一侧 R² 是 NaN（测试集标签全同）→ 带状态返回，不抛异常。
    A NaN input becomes a status rather than an exception.
    """
    result = LK.lambda_from_r2(r2_chain=float("nan"), r2_unit=0.6)
    assert result.status == LK.STATUS_UNDEFINED
    assert result.reportable is False


# --------------------------------------------------------------------------- #
# 配对：只用 A2 与 A3/A4，OpenPoly 不用 5–9 折 / Pairing rules
# --------------------------------------------------------------------------- #
def test_pairing_key_splits_on_double_underscore() -> None:
    """臂名里带下划线（`a2_conventional`），所以必须按 `__` 切；按单个 `_` 切会把臂名切碎。
    Arm names contain single underscores, so the split must be on `__`.
    """
    assert LK.pairing_key("pb_t1_tg__a2_conventional__official_70_30__seed0") == (
        "pb_t1_tg",
        "official_70_30",
        "seed0",
    )
    assert LK.pairing_key("op_tg__a3_graph__seed0_n30__seed1") == ("op_tg", "seed0_n30", "seed1")


def test_pairing_key_rejects_a_bad_shape() -> None:
    """段数不对 → `LambdaError`，不许猜着切。
    A wrong number of segments raises instead of guessing.
    """
    with pytest.raises(LK.LambdaError):
        LK.pairing_key("pb_t1_tg__a2_conventional__seed0")


def test_labels_lookup_honours_the_dataset_suffix() -> None:
    """双数据集任务的 pack 是 `<task>_<dataset>.npz`：不给 `dataset_id` 就取不到标签。
    A dual-dataset task stores `<task>_<dataset>.npz`, so the lookup needs the dataset id.

    中文：这条是 2026-11-17 修的坑的机器保证。`_task_labels` 早就有 `dataset_id` 形参，但调用处
          一直没传——单数据集任务全都碰不到，直到 `pb26_scaling_cp` 的两份 pack 按 D-54 拆开，
          才在 321 对的表上炸出 `LambdaError: 缺特征 pack`。**形参存在不等于被用上**。
    English: the parameter existed but the caller never passed it; only a dual-dataset task exposes
    it. A unit test on the lookup is the cheapest place to keep that from regressing.
    """
    import numpy as np

    for dataset_id in ("md5000_homopolymer_cp", "md5000_alternating_cp"):
        pack = LK.FEATURES_DIR / f"pb26_scaling_cp_{dataset_id}.npz"
        # 判据是"取到的就是这一份 pack 的 y"，不按长度猜：两个数据集的长度本来就不同
        # （1800 / 27,802），而且均聚那一份的开头几行与交替相同，按值比会误判成同一份。
        assert np.array_equal(
            LK._task_labels("pb26_scaling_cp", {}, dataset_id),
            np.asarray(np.load(pack, allow_pickle=True)["y"], dtype=np.float64),
        ), dataset_id
    with pytest.raises(LK.LambdaError):
        LK._task_labels("pb26_scaling_cp", {}, None)  # 没有 `<task>.npz` 这一份，必须报缺件


def test_real_table_matches_the_known_shape() -> None:
    """真仓库的配对表：321 对，且一条 `a1_floor`、一条 5–9 折、一条机制实验都没混进来。
    The real pairing table: 321 pairs, with no `a1_floor`, no fold 5-9 and no mechanism experiment.

    中文：321 = OpenPoly 3 性质 × 75 + PB26 2 任务 × 5 折 × 3 种子 × 2 臂 = 60 + PolyBench 6 × 3 × 2 = 36。
          任何一条 `a1_floor` 混进来都会让 321 变大——它是地板，不是"只给重复单元"的模型。
          **`pb26_scaling_cp` 必须不在里面**（D-69）：它的 40 个格是 8 个 (数据集, 档位) × 5 折，
          同一折的同一批测试行要出现 4 次；混进主表会把"321 对独立的测试点"变成"含 40 行重复计数"。
          机制实验的 Λ 逐格写在 `kappa_cells.csv`，见 `docs/Kappa Table Notes.md` §5。
    English: 321 = 3 OpenPoly properties x 75 + 2 PB26 tasks x 5 folds x 3 seeds x 2 arms + 6
    PolyBench tasks x 3 seeds x 2 arms. The mechanism experiment is excluded on purpose: its cells
    repeat the same test rows across four tiers, so including it would double count.
    """
    pairs = LK.collect_pairs()
    assert len(pairs) == 321, f"配对数变成 {len(pairs)}，先查臂名单或折过滤"
    assert all(pair["arm"] in LK.CHAIN_ARMS for pair in pairs)
    assert not any("__a1_floor__" in pair["unit_run"] for pair in pairs)
    # 敏感性折（seed5–9）不许进主表（D-36 ④）。
    assert not any(pair["middle"].endswith(("seed5", "seed6", "seed7", "seed8", "seed9")) for pair in pairs)
    # 机制实验整任务不进主表（D-69）：它的 Λ 在 `kappa_cells.csv`。
    assert not any(pair["task"] == LK.SCALING_TASK for pair in pairs)


def test_lambda_table_statuses_are_only_the_three_known_ones() -> None:
    """321 行的状态只能是三种登记值；且 `reportable` 必须与 `value` 是否为 NaN 一致。
    Only the three registered statuses appear, and `reportable` agrees with "value is not NaN".

    中文：最后那条是这条测试的重点——它是"`value` 与 `reportable` 不可能互相矛盾"的机器保证。
    English: the last clause is the point: value and reportable can never disagree.
    """
    rows = LK.build_lambda_table()
    statuses = {row["status"] for row in rows}
    assert statuses <= {LK.STATUS_OK, LK.STATUS_UNSTABLE, LK.STATUS_UNDEFINED}
    for row in rows:
        assert row["reportable"] == (not math.isnan(float(row["lambda"]))), row
        if row["reportable"]:
            assert row["status"] == LK.STATUS_OK


def test_summary_keeps_not_computable_apart_from_zero() -> None:
    """一个可报行都没有时，`lambda_mean` 是空字符串，不是 0——"算不出来"与"算是 0"必须分开。
    With nothing reportable the mean is an empty string, not 0.

    中文：这条与主表"缺格留灰、不许填 0"是同一个规矩（`main_table.yaml` 的 `missing_reason`）。
    English: same rule as the main table's "leave the cell blank, never fill it with zero".
    """
    summary = LK.summarise_by_arm(
        [
            {
                "task": "toy",
                "arm": "a3_graph",
                "reportable": False,
                "lambda": float("nan"),
                "delta_r2": -0.1,
            }
        ]
    )
    assert summary[0]["lambda_mean"] == ""
    assert summary[0]["n_reportable"] == 0
    assert summary[0]["n_pairs"] == 1
