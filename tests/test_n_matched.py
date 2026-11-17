"""n 匹配模块的测试：先合成配对，再接真表。
Tests for the n-matched module: synthetic pairings first, the real table second.

中文：合成用例不依赖真数据，且**故意把 Λ 造得已知**——`kappa.bootstrap_kappa()` 的教训
      （D-71）：区间/配对路径最容易在真数据未齐时"没报错但算错"。
"""

from __future__ import annotations

import csv

import numpy as np
import pytest

from chainworth.analysis import n_matched as NM


def _reading(task: str, arm: str, level: str, seed: int, r2: float, n: int = 711) -> NM.Reading:
    return NM.Reading(
        task=task, arm=arm, level=level, seed=seed, r2=r2, n_train=n, subset_sha256="deadbeef"
    )


def test_lambda_matches_the_definition():
    """Λ = (0.8 − 0.5)/(1 − 0.5) = 0.6。"""
    assert NM._lambda(0.5, 0.8) == pytest.approx(0.6)


def test_lambda_is_not_clipped_and_nan_when_denominator_dies():
    """分母 ≤ 0 时返回 nan（不裁剪、不写 0）：R²_unit = 1 时 Λ 无定义。"""
    assert np.isnan(NM._lambda(1.0, 0.9))
    assert np.isnan(NM._lambda(1.2, 0.9))
    # 分子为负要如实返回负值，不能裁剪成 0
    assert NM._lambda(0.5, 0.3) == pytest.approx(-0.4)


def test_cells_pair_the_two_arms_on_the_same_seed():
    """两条臂必须按 (任务, 档位, 种子) 配对相减，不能各自取均值再减。"""
    readings = [
        _reading("pb_t1_tg", "a2_conventional", "711", 0, 0.50),
        _reading("pb_t1_tg", "a3_graph", "711", 0, 0.80),  # Λ = 0.6
        _reading("pb_t1_tg", "a2_conventional", "711", 1, 0.60),
        _reading("pb_t1_tg", "a3_graph", "711", 1, 0.60),  # Λ = 0.0
    ]
    cells = NM.cell_rows(readings)
    assert len(cells) == 1
    assert cells[0]["n_seeds"] == 2
    assert cells[0]["lambda"] == pytest.approx(0.3)  # 逐种子算再平均 = (0.6 + 0.0)/2
    assert cells[0]["lambda_min"] == pytest.approx(0.0)
    assert cells[0]["lambda_max"] == pytest.approx(0.6)


def test_a_cell_without_the_chain_arm_is_dropped_not_faked():
    """只有 A2 没有 A3 的格必须丢掉，不能拿 A2 当链臂。"""
    readings = [_reading("pb_t1_tg", "a2_conventional", "711", 0, 0.5)]
    with pytest.raises(NM.NMatchedError):
        NM.cell_rows(readings)


def test_tier_comes_from_the_task_name_because_records_say_none():
    """记录里的 tier 全是 None，层号必须由任务名推。"""
    readings = [
        _reading("pb_t1_tg", "a2_conventional", "711", 0, 0.5),
        _reading("pb_t1_tg", "a3_graph", "711", 0, 0.7),
        _reading("pb_t2_modulus", "a2_conventional", "711", 0, 0.5),
        _reading("pb_t2_modulus", "a3_graph", "711", 0, 0.6),
    ]
    cells = {row["task"]: row for row in NM.cell_rows(readings)}
    assert cells["pb_t1_tg"]["tier"] == 1
    assert cells["pb_t2_modulus"]["tier"] == 2


def test_summary_delta_is_tier1_minus_tier2():
    """Δ = mean Λ(tier1) − mean Λ(tier2)；tier1 更低时判 tier1_lower。"""
    readings = [
        _reading("pb_t1_tg", "a2_conventional", "711", 0, 0.5),
        _reading("pb_t1_tg", "a3_graph", "711", 0, 0.6),  # Λ = 0.2
        _reading("pb_t1_tm", "a2_conventional", "711", 0, 0.5),
        _reading("pb_t1_tm", "a3_graph", "711", 0, 0.6),  # Λ = 0.2
        _reading("pb_t2_permeability", "a2_conventional", "711", 0, 0.5),
        _reading("pb_t2_permeability", "a3_graph", "711", 0, 0.9),  # Λ = 0.8
    ]
    cells = NM.cell_rows(readings)
    summary = NM.summary_rows(cells, n_boot=200)
    row = next(r for r in summary if r["n_level"] == "711")
    assert row["lambda_tier1"] == pytest.approx(0.2)
    assert row["lambda_tier2"] == pytest.approx(0.8)
    assert row["delta"] == pytest.approx(-0.6)
    assert row["verdict"] == "tier1_lower"


def test_a_level_without_tier2_says_so_instead_of_dropping_the_row():
    """只有 tier1 的档位要留一行并写 `no_tier2_at_this_level`，不静默消失。"""
    readings = [
        _reading("pb_t1_tg", "a2_conventional", "1034", 0, 0.5),
        _reading("pb_t1_tg", "a3_graph", "1034", 0, 0.7),
    ]
    summary = NM.summary_rows(NM.cell_rows(readings), n_boot=50)
    row = next(r for r in summary if r["n_level"] == "1034")
    assert row["delta"] == ""
    assert row["verdict"] == "no_tier2_at_this_level"


def test_interval_brackets_the_delta():
    """区间要包住点估计；两格重采样时区间可以很宽，但不能是空。"""
    readings = [
        _reading("pb_t1_tg", "a2_conventional", "711", 0, 0.50),
        _reading("pb_t1_tg", "a3_graph", "711", 0, 0.80),
        _reading("pb_t1_tm", "a2_conventional", "711", 0, 0.60),
        _reading("pb_t1_tm", "a3_graph", "711", 0, 0.62),
        _reading("pb_t2_permeability", "a2_conventional", "711", 0, 0.50),
        _reading("pb_t2_permeability", "a3_graph", "711", 0, 0.70),
    ]
    row = NM.summary_rows(NM.cell_rows(readings), n_boot=400)[0]
    assert float(row["delta_lo"]) <= float(row["delta"]) <= float(row["delta_hi"])


def test_a_missing_table_raises_instead_of_returning_empty(tmp_path):
    """缺表要抛，不返回空列表。"""
    with pytest.raises(NM.NMatchedError):
        NM.load_readings(tmp_path / "nope.csv")


def test_real_table_has_four_tasks_at_two_levels():
    """真表：4 个任务、2 个匹配档（t2 的任务各占一个档），每格 3 个种子；A4 不在里面。"""
    readings = NM.load_readings()
    cells = NM.cell_rows(readings)
    assert {c["task"] for c in cells} == {
        "pb_t1_tg",
        "pb_t1_tm",
        "pb_t2_modulus",
        "pb_t2_permeability",
    }
    assert {c["n_level"] for c in cells} == {"497", "723"}
    assert all(c["n_seeds"] == 3 for c in cells)
    assert {r.arm for r in readings} == {"a2_conventional", "a3_graph"}
    # tier1 的两臂(train 行)必须逐位相同；tier2 的 A3 建图会丢行，所以只能要求 ≤
    for cell in cells:
        if cell["tier"] == 1:
            assert cell["n_unit"] == cell["n_chain"] == int(cell["n_level"])
        else:
            assert cell["n_chain"] <= cell["n_unit"]
    # 同一格同一臂的三个种子必须落在同一批训练行上（哈希是"该臂自己的行坐标"，两臂之间不可比）
    per_cell_arm: dict[tuple[str, str, str], set[str]] = {}
    for reading in readings:
        per_cell_arm.setdefault((reading.task, reading.level, reading.arm), set()).add(
            reading.subset_sha256
        )
    assert all(len(v) == 1 for v in per_cell_arm.values()), "同格同臂的种子间换了训练行"
    # tier1 的两臂都做了下采样，必须都留下哈希；tier2 是 `--no-subset`，为空是预期的
    for (task, _level, _arm), values in per_cell_arm.items():
        if task.startswith("pb_t1_"):
            assert all(v.startswith("sha256:") for v in values), "下采样却没记 subset 哈希"


def test_real_table_delta_stays_negative_after_matching():
    """匹配后 Δ = Λ(tier1) − Λ(tier2) 仍为负，且两个档的区间都排除 0。

    这条是本轮的关键读数：混淆能解释一部分层间差（全量 −0.100 → 497 档 −0.033），
    但**不能**把它解释干净（723 档仍 −0.087）。
    """
    cells = NM.cell_rows(NM.load_readings())
    summary = NM.summary_rows(cells, n_boot=400)
    for row in summary:
        assert row["verdict"] == "tier1_lower", row
        assert float(row["delta_hi"]) < 0, "区间跨过 0 就不能说方向稳"


def test_real_table_summary_is_written_and_readable(tmp_path):
    """真表能落盘再读回，列名齐全（防止 schema 漂移）。"""
    cells_path = tmp_path / "cells.csv"
    summary_path = tmp_path / "summary.csv"
    cells, summary = NM.write_all(cells_path=cells_path, summary_path=summary_path, n_boot=200)
    assert cells and summary
    with cells_path.open(encoding="utf-8", newline="") as handle:
        assert csv.DictReader(handle).fieldnames == NM.CELL_COLUMNS
    with summary_path.open(encoding="utf-8", newline="") as handle:
        assert csv.DictReader(handle).fieldnames == NM.SUMMARY_COLUMNS
