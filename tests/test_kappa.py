"""κ 的四条口径逐条落成断言：同批行、按折算、配对重采样、分母不稳要显形。
The four kappa conventions as assertions: same rows, per-fold slopes, paired resampling, and a
visibly flagged unstable denominator.

中文：
    用途：分工 D-3 的验收。`analysis/kappa.py` 里每一条口径都能**悄悄**改变结论，而不抛异常——
          把两臂的格子对错、把所有点堆起来做一次 OLS、重采样时各档位各抽各的下标、把分母过小的
          格子静默丢掉，这四类错都只会让数字更好看。所以它们必须写成测试。
    数据来源：**合成曲线**（不依赖 40 条真记录，也就不依赖链臂跑完）+ 两张假 CSV（测对齐与报错）。
    副作用：只读（真仓库那张表的形状由 `test_scaling_chain.py` 守）。

English:
    Purpose: division D-3's acceptance. Each convention in `analysis/kappa.py` can silently change
    the answer without raising, so each one gets an assertion. The synthetic curves mean these tests
    do not wait for the real forty records; the real table's shape is guarded elsewhere.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest

from chainworth.analysis import ci
from chainworth.analysis import kappa as KP


# --------------------------------------------------------------------------- #
# 合成一条曲线：R² 由"残差能量 / 总平方和"精确给到，Λ 就能种成任意形状
# A synthetic curve: R² is set exactly through the residual energy, so Lambda can be planted
# --------------------------------------------------------------------------- #
def _prediction_with_r2(
    truth: np.ndarray, r2: float, rng: np.random.Generator
) -> np.ndarray:
    """造一个 R² 恰为 `r2` 的预测：残差与 `truth` 正交、均值为 0。／ A prediction with exact R²."""
    n = truth.size
    sst = float(truth @ truth)  # truth 已去均值
    residual = rng.normal(size=n)
    residual -= residual.mean()
    residual -= truth * float(residual @ truth) / sst
    residual *= math.sqrt(max(1.0 - r2, 0.0) * sst / float(residual @ residual))
    return truth + residual


def _bundle(
    dataset: str,
    fold: str,
    tiers: list[int],
    lambda_of,
    *,
    r2_unit: float = 0.50,
    n_rows: int = 2000,
    seed: int = 0,
) -> KP.FoldBundle:
    """按 `lambda_of(n_train)` 种一条曲线，返回一折 bundle。／ Plant one fold's curve."""
    rng = np.random.default_rng(seed)
    truth = rng.normal(size=n_rows)
    truth -= truth.mean()
    levels: list[tuple[int, np.ndarray, np.ndarray]] = []
    for n_train in tiers:
        lam = float(lambda_of(n_train))
        r2_chain = (1.0 - r2_unit) * lam + r2_unit
        levels.append(
            (
                n_train,
                _prediction_with_r2(truth, r2_unit, rng),
                _prediction_with_r2(truth, r2_chain, rng),
            )
        )
    return KP.FoldBundle(
        dataset=dataset,
        fold=fold,
        truth=truth,
        row_id=np.arange(n_rows, dtype=np.int64),
        levels=levels,
        tiers=[f"n{int(item[0])}" for item in levels],
    )


# --------------------------------------------------------------------------- #
# ① 斜率：Λ 被种成线性 → κ 必须精确还原那个斜率
# --------------------------------------------------------------------------- #
def test_kappa_recovers_a_planted_slope() -> None:
    """Λ = 0.40 − 0.15·log₁₀n → κ 必须等于 **−0.15**（不是"接近"，是到浮点精度）。
    A planted linear Lambda in log10 n means kappa equals the slope to floating-point precision.

    中文：这条是整套算法的"零点校准"。若哪天有人把 x 换成 n、把 log 底数改了、或把斜率的分子
          分母写反，恢复出来的数就不对，而图仍然画得出来。
    English: The zero-point calibration for the whole chain: n instead of log10 n, a different log
    base, or an inverted slope would all still draw a figure - only this test notices.
    """
    lam = lambda n: 0.40 - 0.15 * math.log10(n)
    bundles = {("d", "fold_00"): _bundle("d", "fold_00", [300, 1000, 3000, 10000], lam)}
    point = KP.point_estimate(bundles)["d"]
    assert point["kappa"] == pytest.approx(-0.15, rel=0, abs=1e-10)
    assert point["kappa_pooled"] == pytest.approx(-0.15, rel=0, abs=1e-10)
    # 两端 Λ 也要对：最小档位 300、最大档位 10000。
    assert point["lambda_first"] == pytest.approx(lam(300), rel=0, abs=1e-10)
    assert point["lambda_last"] == pytest.approx(lam(10000), rel=0, abs=1e-10)
    assert point["n_levels"] == 4 and point["n_folds"] == 1


def test_slopes_are_averaged_per_fold_not_pooled() -> None:
    """两折斜率差很大（−0.20 vs −0.05）时，κ 是**两折的算术平均**，不是把所有点堆起来 OLS。
    With very different per-fold slopes, kappa is the mean of the two, not the pooled OLS.

    中文：这里的差别不是"点多的一折赢"，而是"**x 跨度大的那一折赢**"——池化 OLS 的权重是
          Σ(x−x̄)²，折 B 的两个档位相隔 4 个数量级，一进去就把折 A 的四个档位（跨度 0.13 个
          数量级）压成噪声。实测：池化 ≈ −0.05（几乎就是折 B 自己），按折平均 = −0.125。
          这个差**不抛异常**，只让结论更"显著"——正是分工 D §7 红线第 4 条要防的东西。
    English: pooling lets the fold with the wider x-span speak for the whole curve: with fold B
    spanning four decades, the pooled OLS collapses onto fold B's own slope (~-0.05) while the
    per-fold mean stays at -0.125. Nothing raises; the result only looks stronger.
    """
    steep = lambda n: 0.40 - 0.20 * math.log10(n)
    flat = lambda n: 0.10 - 0.05 * math.log10(n)
    bundles = {
        # 折 A：斜率陡，但四个档位挤在一起。
        ("d", "fold_00"): _bundle("d", "fold_00", [900, 1000, 1100, 1200], steep, seed=1),
        # 折 B：斜率平，但两个档位相隔四个数量级。
        ("d", "fold_01"): _bundle("d", "fold_01", [10, 100000], flat, seed=2),
    }
    point = KP.point_estimate(bundles)["d"]
    assert point["per_fold"][0] == pytest.approx(-0.20, rel=0, abs=1e-9)
    assert point["per_fold"][1] == pytest.approx(-0.05, rel=0, abs=1e-9)
    assert point["kappa"] == pytest.approx(-0.125, rel=0, abs=1e-9)
    # 池化值必须落在折 B 上，**明显不等于**按折平均——否则"对照量"形同虚设。
    assert point["kappa_pooled"] == pytest.approx(-0.05, abs=0.01), point["kappa_pooled"]
    assert abs(point["kappa_pooled"] - point["kappa"]) > 0.05
    assert point["n_levels"] == 4, "n_levels 报的是最多的那一折，不是总和"


# --------------------------------------------------------------------------- #
# ② 区间：配对重采样，且同种子必须逐位可复现
# --------------------------------------------------------------------------- #
def test_interval_brackets_the_point_estimate_and_is_reproducible() -> None:
    """区间必须把点估计包在里面；同一个 seed 跑两次必须完全一样（冻结项，换就是换版本）。
    The interval brackets the estimate, and the same seed reproduces it bit for bit.

    中文：κ 的区间与 Λ 的区间共用 `ci.BOOTSTRAP_DRAWS` / `ci.BOOTSTRAP_SEED`（D-2）。如果哪里
          混进了未播种的随机源，第二次跑出来的区间就会变，而表里 `seed` 那一列看不出来。
    """
    lam = lambda n: 0.40 - 0.15 * math.log10(n)
    bundles = {
        ("d", "fold_00"): _bundle("d", "fold_00", [300, 1000, 3000, 10000], lam, n_rows=8000, seed=3),
        ("d", "fold_01"): _bundle("d", "fold_01", [300, 1000, 3000, 10000], lam, n_rows=8000, seed=4),
    }
    first = KP.bootstrap_kappa(bundles, draws=200, seed=20261117)["d"]
    second = KP.bootstrap_kappa(bundles, draws=200, seed=20261117)["d"]
    assert (first.lo, first.hi) == (second.lo, second.hi)
    assert first.n_boot == 200 and first.seed == 20261117
    assert first.lo <= first.estimate <= first.hi
    # 种的是直线，重采样只会让曲线轻微弯；区间应当把真值包住，且不到 0.1 宽
    # （端点的 Λ 抖动会经分母 1 − 0.5 放大两倍，所以这条曲线本来就不该给出极窄的区间）。
    assert first.hi - first.lo < 0.10, (first.lo, first.hi)
    assert first.lo <= -0.15 <= first.hi
    # 换个种子，区间应"差不多但不完全相同"——相同就说明种子没被用上。
    other = KP.bootstrap_kappa(bundles, draws=200, seed=7)["d"]
    assert (other.lo, other.hi) != (first.lo, first.hi)
    assert other.lo == pytest.approx(first.lo, abs=0.02)


# --------------------------------------------------------------------------- #
# ③ 汇总：H2 的两个条件分开报，Λ_first ≤ 0 时"减半"写 undefined
# --------------------------------------------------------------------------- #
def _point(*, kappa: float, first: float, last: float, span_dex: float = 2.0, min_gap_dex: float = 0.5) -> dict:
    return {
        "kappa": kappa,
        "kappa_pooled": kappa,
        "per_fold": [kappa],
        "lambda_first": first,
        "lambda_last": last,
        "n_levels": 4,
        "n_folds": 1,
        "span_dex": span_dex,
        "min_gap_dex": min_gap_dex,
    }


def _interval(lo: float, hi: float) -> ci.BootstrapResult:
    return ci.BootstrapResult(estimate=(lo + hi) / 2, lo=lo, hi=hi, n_boot=2000, n_points=1, seed=1)


def test_h2_needs_both_conditions() -> None:
    """区间整段 < 0 **且** Λ 减半 → `H2_refuted`；只满足一条时 verdict 必须说清是哪一条。
    Both conditions -> refuted; one alone -> the verdict names which one failed.

    中文：只报"显著为负"就宣布 H2 被推翻，是这份分工最容易被审的一处：κ < 0 也可能只是
          "链有用但补不回来"，那与 H2"数据涨上去就能补回来"并不矛盾。两个条件各自落一列。
    """
    refuted = KP.summary_rows({"d": _point(kappa=-0.15, first=0.40, last=0.10)}, {"d": _interval(-0.2, -0.1)})[0]
    assert refuted["h2_sign_negative"] == "true"
    assert refuted["h2_lambda_halved"] == "true"
    assert refuted["verdict"] == "H2_refuted"
    assert float(refuted["lambda_ratio"]) == pytest.approx(0.25, abs=1e-9)

    not_halved = KP.summary_rows({"d": _point(kappa=-0.02, first=0.40, last=0.30)}, {"d": _interval(-0.03, -0.01)})[0]
    assert not_halved["h2_sign_negative"] == "true"
    assert not_halved["h2_lambda_halved"] == "false"
    assert not_halved["verdict"] == "kappa_negative_but_lambda_not_halved"

    not_negative = KP.summary_rows({"d": _point(kappa=0.01, first=0.10, last=0.02)}, {"d": _interval(-0.01, 0.03)})[0]
    assert not_negative["h2_sign_negative"] == "false"
    assert not_negative["verdict"] == "H2_not_refuted"


def test_negative_lambda_first_makes_the_ratio_undefined_not_fake() -> None:
    """Λ_first ≤ 0 时"掉到一半"没有意义 → `lambda_ratio` 写 `undefined`、`h2_lambda_halved=false`。
    With a non-positive Lambda_first the "halved" condition is undefined, and the table says so.

    中文：负的 Λ 除以一半会得到比 1 大的比值（−0.2 / −0.4 = 0.5），若照算就会"恰好"满足减半，
          把一个没有意义的比较变成一条证据。这是最阴的一类错误：它只在某些数据上触发。
    """
    row = KP.summary_rows({"d": _point(kappa=-0.15, first=-0.20, last=-0.40)}, {"d": _interval(-0.2, -0.1)})[0]
    assert row["lambda_ratio"] == "undefined"
    assert row["h2_lambda_halved"] == "false"
    assert row["verdict"] == "kappa_negative_but_lambda_not_halved", "负数不能凑出'减半'"


def test_no_interval_leaves_the_bounds_empty_not_zero() -> None:
    """没跑重采样时 `kappa_lo`/`kappa_hi` 必须是空串（不是 0）——0 会被读成真实的边界。
    Without resampling the bounds are empty strings, never zeros: a zero reads as a real bound.
    """
    row = KP.summary_rows({"d": _point(kappa=-0.15, first=0.40, last=0.10)}, {})[0]
    assert row["kappa_lo"] == "" and row["kappa_hi"] == ""
    assert row["n_boot"] == 0
    assert row["verdict"] == "H2_not_refuted", "没有区间就不能说'区间整段 < 0'"


# --------------------------------------------------------------------------- #
# ④ 该不该报斜率：两个点永远能连一条直线（D-3 验收标准 ④）
# --------------------------------------------------------------------------- #
def test_a_slope_is_not_reported_when_the_top_two_tiers_nearly_coincide() -> None:
    """均聚 MD_5000 的形状（300 / 1000 / 1440）**不许**报 κ——它实质是两点连线。
    The homopolymer shape (300 / 1000 / 1440) must not report a slope: it is a two-point line.

    中文：三档看着像"有 3 个点"，但顶上一档只比中间那个大 0.158 个数量级——那是同一个点在
          重复测。OLS 会给它一个斜率，还会把杠杆压在那个点上，于是"κ = +0.69"读起来像个数，
          其实是一条没有不确定度的线。分工 D-3 验收标准 ④ 就是防这个。
    English: three tiers look like three points, but the top one sits 0.158 dex above the middle
    one - the same point measured twice. OLS will happily fit it and lean on it, so "kappa = +0.69"
    reads like a number when it is a line with no uncertainty.
    """
    lam = lambda n: 0.05 + 0.25 * math.log10(n)
    bundles = {("d", "fold_00"): _bundle("d", "fold_00", [300, 1000, 1440], lam, seed=11)}
    point = KP.point_estimate(bundles)["d"]
    assert point["span_dex"] == pytest.approx(math.log10(1440 / 300), abs=1e-9)
    assert point["min_gap_dex"] == pytest.approx(math.log10(1440 / 1000), abs=1e-9)
    row = KP.summary_rows({"d": point}, {"d": _interval(0.40, 0.80)})[0]
    assert row["slope_status"] == "no_slope_overlapping_tiers"
    # 三个 κ 列全空：一个都不能留给读者当数看。
    assert row["kappa"] == "" and row["kappa_lo"] == "" and row["kappa_hi"] == ""
    assert row["kappa_pooled"] == ""
    assert row["verdict"] == "no_slope"
    assert row["h2_sign_negative"] == "false", "没报斜率就不许说'区间整段 < 0'"
    # 两端连线斜率仍可算，但它在**单独一列**，且等于两端之差除以跨度。
    assert float(row["endpoint_slope"]) == pytest.approx(0.25, abs=1e-9)


def test_two_tiers_alone_are_never_a_slope() -> None:
    """只有两档、跨度不到半个数量级 → `no_slope_span`。／ Two close tiers -> no slope."""
    lam = lambda n: 0.30 - 0.10 * math.log10(n)
    bundles = {("d", "fold_00"): _bundle("d", "fold_00", [300, 443], lam, seed=12)}
    row = KP.summary_rows({"d": KP.point_estimate(bundles)["d"]}, {"d": _interval(0.1, 0.4)})[0]
    assert row["slope_status"] == "no_slope_span"
    assert row["verdict"] == "no_slope"
    assert row["kappa"] == ""


def test_real_datasets_get_the_right_slope_verdict() -> None:
    """真仓库：交替集可拟合，均聚不可拟合——这就是 D-3 验收标准 ④ 落地的样子。
    On the real tables the alternating set is fittable and the homopolymer set is not.

    中文：这条把上面的合成判据钉到真数据上。若哪天档位定义变了（比如均聚补到 5000 行），
          这条会红，正好提醒去重看 κ 该不该报。
    """
    points = KP.point_estimate(KP.load_bundles(KP.load_cells()))
    alt = points["md5000_alternating_cp"]
    homo = points["md5000_homopolymer_cp"]
    assert alt["span_dex"] >= KP.MIN_SPAN_DEX and alt["min_gap_dex"] >= KP.MIN_GAP_DEX
    assert homo["span_dex"] >= KP.MIN_SPAN_DEX, "均聚的跨度本身够，卡它的是顶上一档"
    assert homo["min_gap_dex"] < KP.MIN_GAP_DEX, "顶上一档与中间那档近乎重合"


# --------------------------------------------------------------------------- #
# ② 杠杆检查：去掉最小档，符号会不会翻（D-3 验收标准 ②）
# --------------------------------------------------------------------------- #
def test_leverage_check_flags_a_sign_flip() -> None:
    """最小档位单独决定符号时，杠杆检查必须点出来（`sign_flipped=true`）。
    When the smallest tier alone decides the sign, the leverage check says so.

    中文：种的 Λ 是 [0.90, 0.10, 0.20, 0.30]——最小档高、之后跳低再缓升。全档 OLS 给负斜率，
          去掉 300 那档就变正。分工 D-3 验收标准 ② 要的正是这一句："κ 的符号依赖最小档"。
          判据只看符号：大小一定会变，变多少不是发现。
    English: Lambda is planted as [0.90, 0.10, 0.20, 0.30]: a high smallest tier, a drop, then a
    slow rise. The full fit slopes down, the trimmed fit slopes up.
    """
    table = {300: 0.90, 1000: 0.10, 3000: 0.20, 10000: 0.30}
    bundles = {
        ("d", "fold_00"): _bundle("d", "fold_00", [300, 1000, 3000, 10000], lambda n: table[n], seed=13)
    }
    rows = KP.leverage_rows(bundles, draws=50)
    assert len(rows) == 1
    row = rows[0]
    assert row["dropped_tier"] == "300"
    assert row["n_levels_before"] == 4 and row["n_levels_after"] == 3
    assert row["slope_status_after"] == "ok"
    assert float(row["kappa_base"]) < 0.0 < float(row["kappa_dropped"])
    assert row["sign_flipped"] == "true"
    assert "翻转" in row["note"]


def test_leverage_check_does_not_call_a_magnitude_change_a_flip() -> None:
    """同一符号、只是幅度变了 → `sign_flipped=false`，且 delta 照样报出来。
    Same sign, different magnitude -> no flip, but the magnitude change is still reported.

    中文：κ 去掉最小档后几乎一定变（最小档往往最不稳），所以"变了"不是发现，"翻号"才是。
          把幅度变化也标成翻转，会让这条检查失去分辨力。
    """
    lam = lambda n: 0.60 - 0.40 * math.log10(n)
    bundles = {
        ("d", "fold_00"): _bundle("d", "fold_00", [300, 1000, 3000, 10000], lam, seed=14)
    }
    row = KP.leverage_rows(bundles, draws=50)[0]
    assert row["sign_flipped"] == "false"
    assert float(row["kappa_base"]) < 0 and float(row["kappa_dropped"]) < 0
    assert row["delta"] != "" and row["note"] == "符号不翻转"


def test_leverage_check_says_so_when_the_trimmed_curve_is_no_longer_fittable() -> None:
    """删掉一档后不够拟合门槛 → 不比较、`kappa_dropped` 留空，并写明原因。
    If the trimmed curve no longer passes the gates, leave the number out and say why.

    中文：均聚（300/1000/1440）删掉 300 只剩 1000/1440，跨度 0.158。此时给出一个"删档后 κ"
          等于凭空多一个数。
    """
    lam = lambda n: 0.10 + 0.30 * math.log10(n)
    bundles = {("d", "fold_00"): _bundle("d", "fold_00", [300, 1000, 1440], lam, seed=15)}
    row = KP.leverage_rows(bundles, draws=50)[0]
    assert row["slope_status_after"] == "no_slope_span"
    assert row["kappa_dropped"] == "" and row["delta"] == ""
    assert row["sign_flipped"] == "false"
    assert "无法比较" in row["note"]


# --------------------------------------------------------------------------- #
# ④ 对齐：两张表逐格核，缺格 / 行数不同 / 指纹不同 / 分母 ≤ 0 都必须报错
# --------------------------------------------------------------------------- #
UNIT_HEADER = "arm,learner,dataset,tier,fold,seed,n_train,subset_sha256,r2,run_id\n"
CHAIN_HEADER = "arm,dataset,tier,fold,seed,n_train,subset_sha256,r2,run_id\n"


def _write_pair(tmp_path: Path, *, n_train: str = "1000", digest: str = "abc123", r2_unit: str = "0.40"):
    unit = tmp_path / "scaling_cpu.csv"
    chain = tmp_path / "scaling_chain.csv"
    unit.write_text(
        UNIT_HEADER
        + f"a2_conventional,xgboost,ds,n300,f0,0,{n_train},{digest},{r2_unit},u1\n",
        encoding="utf-8",
        newline="\n",
    )
    chain.write_text(
        CHAIN_HEADER + f"a3_graph,ds,n300,f0,0,{n_train},{digest},0.55,c1\n",
        encoding="utf-8",
        newline="\n",
    )
    return unit, chain


def test_a_clean_pair_becomes_a_cell_with_lambda_of_the_definition(tmp_path: Path) -> None:
    """对齐成功时 Λ = (0.55 − 0.40) / (1 − 0.40) = 0.25，分母 0.60 不算不稳。
    A clean pair yields Lambda = 0.25 with the denominator at 0.60, which is stable.
    """
    unit, chain = _write_pair(tmp_path)
    cells = KP.load_cells(unit, chain)
    assert len(cells) == 1
    cell = cells[0]
    assert cell.lam == pytest.approx(0.25, rel=0, abs=1e-12)
    assert cell.delta_r2 == pytest.approx(0.15, rel=0, abs=1e-12)
    assert cell.unstable is False
    assert (cell.unit_run_id, cell.chain_run_id) == ("u1", "c1")


def test_a_small_denominator_is_flagged_not_dropped(tmp_path: Path) -> None:
    """`R²_unit = 0.97` → 分母 0.03 < 0.05 → `unstable=True`，但格子照样留着。
    `R2_unit = 0.97` flags the cell as unstable and keeps it: never silently dropped.
    """
    unit, chain = _write_pair(tmp_path, r2_unit="0.97")
    cells = KP.load_cells(unit, chain)
    assert len(cells) == 1
    assert cells[0].unstable is True


@pytest.mark.parametrize(
    "column, value, needle",
    [
        (6, "other", "指纹不同"),  # subset_sha256
        (5, "3000", "训练行数不同"),  # n_train
    ],
)
def test_a_mismatched_pair_raises_instead_of_aligning(
    tmp_path: Path, column: int, value: str, needle: str
) -> None:
    """行数或指纹对不上 → 报错停下，不许"尽力对齐"（那会把协议差异算进 κ）。
    A row-count or digest mismatch raises instead of aligning as best we can.
    """
    unit, chain = _write_pair(tmp_path)
    # 只改链臂那一份的数据行，模拟"两条臂其实看的不是同一批行"。表头必须留着——
    # 少了它，`DictReader` 会拿第一行当表头，报的是 KeyError 而不是这里要验的判据。
    header, row = chain.read_text(encoding="utf-8").strip().splitlines()
    parts = row.split(",")
    parts[column] = value
    chain.write_text(f"{header}\n{','.join(parts)}\n", encoding="utf-8", newline="\n")
    with pytest.raises(KP.KappaError, match=needle):
        KP.load_cells(unit, chain)


def test_a_missing_chain_cell_raises_instead_of_shrinking(tmp_path: Path) -> None:
    """链臂少一格 → 报错并点名差几格；内连接会静默变小，那是"少了一部分数据却照样出图"。
    A missing chain cell raises and names the count; an inner join would silently shrink.
    """
    unit, _ = _write_pair(tmp_path)
    chain = tmp_path / "scaling_chain.csv"
    chain.write_text(CHAIN_HEADER, encoding="utf-8", newline="\n")
    with pytest.raises(KP.KappaError, match="链臂缺 1 格"):
        KP.load_cells(unit, chain)


def test_the_random_forest_rows_are_not_a_unit_arm(tmp_path: Path) -> None:
    """`scaling_cpu.csv` 里的 `learner=rf` 行按 D-53 不写 run_id，也**不是**单元臂：不许当 A2 用。
    The `learner=rf` rows are a sensitivity learner, not the unit arm - they must not be paired.

    中文：如果它们被当成 A2，同样一个 (数据集, 档位, 折, 种子) 会被 rf 那份静默覆盖，
          κ 就不再是"XGBoost 对 chemprop"的曲线。
    """
    unit, chain = _write_pair(tmp_path)
    unit.write_text(
        UNIT_HEADER
        + f"a2_conventional,rf,ds,n300,f0,0,1000,abc123,0.20,\n"
        + f"a2_conventional,xgboost,ds,n300,f0,0,1000,abc123,0.40,u1\n",
        encoding="utf-8",
        newline="\n",
    )
    cells = KP.load_cells(unit, chain)
    assert len(cells) == 1
    assert cells[0].r2_unit == pytest.approx(0.40)
    assert cells[0].unit_run_id == "u1"


# --------------------------------------------------------------------------- #
# ⑤ 落盘：列必须写全、顺序按常量、行尾必须是 LF
# --------------------------------------------------------------------------- #
def test_cell_rows_carry_every_registered_column() -> None:
    """`cell_rows()` 算出来的键必须与 `CELL_COLUMNS` 完全一致——多算一个字段而不写出，
    等于"算了但没人看得到"；少一个字段会被 `DictWriter` 静默填空串。
    The computed keys must equal the registered columns exactly: an extra field is computed and then
    thrown away, a missing one is silently written as an empty cell.
    """
    cell = KP.Cell(
        dataset="ds",
        fold="f0",
        tier="n300",
        n_train=300,
        r2_unit=0.4,
        r2_chain=0.55,
        unit_run_id="u1",
        chain_run_id="c1",
        subset_sha256="abc123",
        delta_r2=0.15,
        lam=0.25,
        unstable=False,
    )
    row = KP.cell_rows([cell])[0]
    assert set(row) == set(KP.CELL_COLUMNS)
    # 分母也落一列：读表的人不该自己去减。
    assert float(row["denominator"]) == pytest.approx(0.6, abs=1e-10)
    assert row["unstable"] == "false"


def test_written_tables_use_lf_not_crlf(tmp_path: Path) -> None:
    """两张表的行尾必须是 LF。Windows 上 `open()` 默认会把 `\\n` 变成 `\\r\\n`，
    而仓库规则是 `* text=auto eol=lf`——CRLF 会让 `git add` 警告、让 diff 变成整文件重写。
    Written with LF: on Windows the default newline translation would turn every row into CRLF.
    """
    cells_path = tmp_path / "cells.csv"
    summary_path = tmp_path / "summary.csv"
    KP._write(KP.cell_rows([]), KP.CELL_COLUMNS, cells_path)
    KP._write(
        KP.summary_rows({"d": _point(kappa=-0.15, first=0.40, last=0.10)}, {}),
        KP.SUMMARY_COLUMNS,
        summary_path,
    )
    for path in (cells_path, summary_path):
        raw = path.read_bytes()
        assert b"\r" not in raw, path.name
        assert raw.splitlines()[0].decode("utf-8") == ",".join(
            KP.CELL_COLUMNS if path is cells_path else KP.SUMMARY_COLUMNS
        )
