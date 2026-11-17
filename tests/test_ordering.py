"""H1 排序检验的四条口径落成断言：同基准内、按种子平均、并列要用 τ-b、单点要说清。
The four conventions behind the H1 ordering test, as assertions.

中文：
    用途：分工 D-4 的验收。这几类错都不会抛异常，只会让 ρ 更好看或更难看得毫无根据：
          ① 把 PolyBench 与 PolyBench26 的点合并排序（R1 禁止，ρ 变成两个协议的乘积）；
          ② 挑三个种子里最好的那个（选择噪声算成信号）；
          ③ 用 τ-a 而不是 τ-b（tier 每层 2 个任务，并列会把 τ 系统性压低）；
          ④ 看到 ρ 小就说"分层无关"，而不去看是不是一个任务撑起来的。
    数据来源：合成点（不依赖 `lambda_ci.csv`）+ 真仓库的一层形状检查。
    副作用：只读。

English:
    Purpose: division D-4's acceptance. None of these mistakes raises; each makes rho look better
    or worse for no reason. The synthetic points keep the logic testable without the real table,
    and one real-repo test pins the shape that the analysis actually produced.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from chainworth.analysis import heterogeneity as HT
from chainworth.analysis import ordering as OR


def _point(task: str, tier: int, lam: float, *, n_labels: int = 1000, undecidable: bool = False) -> OR.TaskPoint:
    return OR.TaskPoint(
        task=task,
        tier=tier,
        lam=lam,
        lam_lo=lam - 0.1,
        lam_hi=lam + 0.1,
        width=0.2,
        undecidable=undecidable,
        n_labels=n_labels,
        source_kind="experimental",
        n_seeds=3,
    )


# --------------------------------------------------------------------------- #
# ① 秩与相关
# --------------------------------------------------------------------------- #
def test_average_ranks_share_ties() -> None:
    """并列取平均秩：`[1, 1, 2]` → `[1.5, 1.5, 3]`，不是 `[1, 2, 3]`。
    Ties get the average rank, not an arbitrary order.

    中文：tier 每层有 2 个任务，并列是常态。按出现顺序给秩（1, 2）会让 ρ 依赖任务名的
          字典序——换个 task_id 排序就换一个 ρ，而数据一个字没变。
    """
    ranks = OR.average_ranks([1, 1, 2])
    assert list(ranks) == [1.5, 1.5, 3.0]
    assert list(OR.average_ranks([2, 1, 1])) == [3.0, 1.5, 1.5]


def test_spearman_is_symmetric_and_undefined_without_variation() -> None:
    """完美正/负相关、以及"没有变异"时必须给 `nan` 而不是 0。
    Perfect correlations, and nan (not zero) when one side has no variation.

    中文：0 会被读成"不相关"（一个结论），`nan` 会被读成"不可评"（另一个结论）。
          这两件事在报告里的分量完全不同，不能混。
    """
    assert OR.spearman([1, 2, 3], [10, 20, 30]) == pytest.approx(1.0)
    assert OR.spearman([1, 2, 3], [30, 20, 10]) == pytest.approx(-1.0)
    assert math.isnan(OR.spearman([1, 1, 1], [1, 2, 3]))


def test_kendall_tau_b_corrects_for_ties() -> None:
    """带并列修正的 τ-b：全并列时返回 `nan`，部分并列时低于 1 但不是 0。
    tau-b with tie correction.

    中文：τ-a 的公式在并列时仍除以 n(n−1)/2，于是"完全一致"的两列也可能算出 0.6，
          读者会以为"来源之间只是大致一致"——这是并列数据上最容易犯的错。
    """
    assert math.isnan(HT.kendall_tau([1, 1, 1], [1, 2, 3]))
    # x 有并列、y 单调：τ-b < 1，但仍显著为正。
    tau = HT.kendall_tau([1, 1, 2, 3], [1, 2, 3, 4])
    assert 0.5 < tau < 1.0
    assert HT.kendall_tau([1, 2, 3], [3, 2, 1]) == pytest.approx(-1.0)


# --------------------------------------------------------------------------- #
# ② 跨层倒序：只有跨 ≥2 层才算
# --------------------------------------------------------------------------- #
def test_only_inversions_across_two_tiers_count() -> None:
    """相邻层倒过来不算（那只是噪声），跨两层才算（那与"三层划分一致"正面冲突）。
    An adjacent swap is noise; a two-tier inversion contradicts the claim itself.

    中文：每层只有 2 个任务，相邻对调几乎必然出现。若把相邻倒序也算作证伪，
          H1 在任何数据上都会被判死——判据就失去分辨力了。
    """
    adjacent = [
        _point("a", 1, -0.1),
        _point("b", 1, 0.2),
        _point("c", 2, 0.1),   # 比 tier1 的 b 低：相邻倒序
        _point("d", 2, 0.3),
    ]
    assert OR.cross_layer_inversions(adjacent) == []
    wide = adjacent + [_point("e", 3, -0.5)]     # tier3 掉到 0 以下：跨两层倒序
    found = OR.cross_layer_inversions(wide)
    assert found, "tier1 的 Λ 比 tier3 高，必须被点出来"
    assert all(item["gap"] >= 2 for item in found)
    assert {item["high_task"] for item in found} == {"e"}


# --------------------------------------------------------------------------- #
# ③ 判据：任一条件成立即证伪
# --------------------------------------------------------------------------- #
def test_a_clean_monotone_ordering_is_not_refuted() -> None:
    """三层严格递增 → `H1_not_refuted`，ρ = 1。／ A clean monotone ordering passes."""
    points = [
        _point("a", 1, 0.00),
        _point("b", 1, 0.10),
        _point("c", 2, 0.20),
        _point("d", 2, 0.30),
        _point("e", 3, 0.40),
        _point("f", 3, 0.50),
    ]
    found = OR.verdict(points)
    # tier 只有 3 个不同值（每层 2 个任务），秩里必然有并列，所以**完美单调也到不了 1.0**：
    # 实测 0.956。这不影响判据（门槛 0.5 离得很远），但写 1.0 会让这条测试假红。
    assert found["rho"] == pytest.approx(0.956, abs=0.001)
    assert found["inversions"] == []
    assert found["result"] == "H1_not_refuted"
    assert found["reasons"] == []


def test_both_conditions_are_named_separately() -> None:
    """触发时两条 reason 都要出现：不能只写"ρ < 0.5"就把倒序吞掉。
    When both fire, both are named - one must not swallow the other.

    中文：这里有一条**数学事实**要说清，否则会误以为两个条件是独立的：三层、每层 2 个点时
          一共 15 个配对，而 tier1 与 tier3 之间就有 4 对——4 对逆序必然把 ρ 拉到 0.5 以下
          （13/15 = 0.867 是最乐观的情形，对应"只有 1 对逆序"，但那在 2 个点一层时做不到）。
          所以在**这份数据的结构下**，"ρ < 0.5"与"出现跨层倒序"几乎等价；
          两个条件仍各自记账，是因为口径要能在层次更多、每层点数不等的数据上分开生效。
    """
    points = [
        _point("a", 1, 0.90),   # tier1 极高
        _point("b", 1, 0.80),
        _point("c", 2, 0.30),
        _point("d", 2, 0.20),
        _point("e", 3, 0.10),
        _point("f", 3, 0.05),
    ]
    found = OR.verdict(points)
    assert found["result"] == "H1_refuted"
    assert found["rho"] < 0.0, "Λ 随 tier 反着走"
    assert found["inversions"], "tier1 的两点都高过 tier3"
    assert len(found["reasons"]) == 2, f"两条都该出现，实际 {found['reasons']}"
    assert any("ρ" in reason for reason in found["reasons"])
    assert any("倒序" in reason for reason in found["reasons"])


def test_a_single_tier_is_not_evaluable() -> None:
    """所有点都在同一层 → `not_evaluable`，不是 `H1_not_refuted`。
    One tier only -> not evaluable, never "not refuted".

    中文：把"没排过"写成"没被推翻"是这类检验里最省事也最严重的假绿。
    """
    found = OR.verdict([_point("a", 1, 0.1), _point("b", 1, 0.2), _point("c", 1, 0.3)])
    assert found["result"] == "not_evaluable"
    assert math.isnan(found["rho"])


# --------------------------------------------------------------------------- #
# ④ 真仓库的形状（这几条一红，说明数据或判据变了）
# --------------------------------------------------------------------------- #
def test_real_polybench_has_six_main_tasks_in_three_tiers() -> None:
    """真仓库：6 个主表任务、恰好三层、每层 2 个。／ Six main tasks, three tiers, two each."""
    tasks = OR.load_tasks()
    tiers = [info["tier"] for info in tasks.values()]
    assert len(tasks) == 6, f"主表任务数变成 {len(tasks)}"
    assert sorted(set(tiers)) == [1, 2, 3]
    assert all(tiers.count(tier) == 2 for tier in (1, 2, 3)), "每层应当恰好 2 个任务"


def test_real_points_are_sorted_and_carry_their_seeds() -> None:
    """点按 tier 排好、每种子的 λ 都参与了平均。／ Points sorted by tier, all seeds averaged."""
    tasks = OR.load_tasks()
    points = OR.load_points("a3_graph", tasks=tasks)
    assert [p.tier for p in points] == sorted(p.tier for p in points)
    assert all(p.n_seeds == 3 for p in points), "PolyBench 是 3 个种子"


def test_real_single_source_marks_tau_not_evaluable() -> None:
    """真仓库只有一个 label source → τ 与 leave-one-source-out 都必须写 `not_evaluable`。
    With one source, tau and leave-one-source-out must say not_evaluable.

    中文：这条守的是"不许把做不了的检查写成通过"。哪天任务配置里出现 DFT / MD，
          这条会红，提醒去看 τ 是不是真算出来了。
    """
    tasks = OR.load_tasks()
    points = OR.load_points("a3_graph", tasks=tasks)
    groups = HT.source_groups(points)
    assert list(groups) == ["experimental"], f"来源变成 {list(groups)}"
    by_source = HT.by_source_rows("a3_graph", points)
    assert len(by_source) == 1
    assert by_source[0]["result"] == "not_evaluable_single_source"
    assert by_source[0]["kendall_tau"] == "nan"
    loos = HT.leave_one_source_out_rows("a3_graph", points)
    assert len(loos) == 1 and loos[0]["result"] == "not_evaluable_single_source"
    assert loos[0]["spearman_rho"] == "nan"


def test_real_tier_is_anti_ordered_with_sample_size() -> None:
    """真仓库：tier 与样本量反序（ρ ≈ −0.96）——这是 H1 被证伪的机制，必须随 ρ 一起报。
    The tier is anti-ordered with the label count; that is the mechanism behind the verdict.

    中文：如果不把这个混淆写进表，读者会把"ρ ≈ 0"读成"分层与 Λ 无关"，而真实图景是
          "tier 越高样本越少，两个效应互相抵消"。这两种解释在报告里分量完全不同。
    """
    tasks = OR.load_tasks()
    points = OR.load_points("a3_graph", tasks=tasks)
    row = HT.confound_rows("a3_graph", points)[0]
    assert row["result"] == "confounded"
    assert float(row["spearman_rho"]) <= -0.9
    # 逐层检查：tier 高的那两个任务样本量必须最小。
    for tier in (1, 2, 3):
        counts = [p.n_labels for p in points if p.tier == tier]
        if tier == 1:
            assert min(counts) > 3000
        if tier == 3:
            assert max(counts) < 300
