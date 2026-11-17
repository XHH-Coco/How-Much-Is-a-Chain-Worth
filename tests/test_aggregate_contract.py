"""图与表的契约：打乱数据后图**必须**变，且重画必须逐值一致（分工 D-10）。

The figure/table contract: shuffle the data and the figure must change; redraw and it must not.

中文：
    用途：分工 D-10 的验收。这一项是**自查**——出图的人和审图的人是同一个，独立性天然不足，
          所以补偿办法只有一个：**让测试能抓错**。
      ① **打乱一个任务的数，图必须变。** 若图里那个数字是硬编码的，打乱后图不会动，测试就红。
         这是本文件存在的全部理由（D-10 验收标准①）。
      ② **重画逐值一致。** 图里不许有随机量（两臂的横向错开是固定数，不是抖动），
         否则"一条命令重画"这句话不成立（D-6 验收标准①）。
      ③ **缺失格必须是 `n/a`，不许是 0。** 热图上一格 0 会被读成"测出来链信息没用"。
    做法：把 `common.AGGREGATE` 指到临时目录，喂一份改过的表，再比图里提取出来的文本与坐标。
          不去改真表——测试永远不许写 `results/`。
    副作用：只在 `tmp_path` 里写文件。

English:
    Purpose: division D-10's acceptance. This item is self-audited, so the only compensation is
    that the test must be able to catch an error: (1) shuffle one task's numbers and the figure must
    move - if the number were hardcoded it would not; (2) a redraw is value-identical, because
    nothing in the figures is random; (3) a missing cell reads `n/a` and never 0.
"""

from __future__ import annotations

import csv
import shutil
from pathlib import Path

import pytest

from chainworth.figures import common as C
from chainworth.figures import fig1_lambda_heatmap as F1
from chainworth.figures import fig2_lambda_vs_tier as F2
from chainworth.figures import fig3_lambda_vs_n as F3

REAL_AGGREGATE = Path(__file__).resolve().parents[1] / "results" / "aggregate"


def _texts(figure) -> list[str]:
    """图里所有可见文本（按坐标排序，便于稳定比较）。／ Every visible string, in a stable order."""
    collected = []
    for ax in figure.axes:
        for text in ax.texts:
            payload = text.get_text()
            if payload:
                collected.append(payload)
    return collected


def _numbers(figure) -> list[str]:
    """图里的数值文本（含数字的那些）。／ Only the strings that carry a number."""
    return [item for item in _texts(figure) if any(ch.isdigit() for ch in item)]


@pytest.fixture()
def sandbox(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """把 `common.AGGREGATE` 指到临时目录，复制三张最小的输入表。／ Point the reader at a temp copy."""
    for name in ("lambda_ci.csv", "h1_ordering.csv", "kappa.csv", "kappa_cells.csv"):
        source = REAL_AGGREGATE / name
        if source.exists():
            shutil.copyfile(source, tmp_path / name)
    monkeypatch.setattr(C, "AGGREGATE", tmp_path)
    return tmp_path


def test_redrawing_is_value_identical() -> None:
    """同一份表重画两次，图里的数值文本逐个相同。／ Two redraws give identical numbers.

    中文：这条守的是"图里没有随机量"。第一版图 2 用随机抖动错开两臂，每次重画位置都变——
          那种图没法当证据，因为读者拿到的那张和别人拿到的那张不一样。
    """
    first = _numbers(F1.build()[0])
    second = _numbers(F1.build()[0])
    assert first == second

    first = _numbers(F2.build()[0])
    second = _numbers(F2.build()[0])
    assert first == second

    first = _numbers(F3.build()[0])
    second = _numbers(F3.build()[0])
    assert first == second


def test_shuffling_one_task_moves_the_heatmap(sandbox: Path) -> None:
    """把一个任务的 Λ 改掉，图 1 里的值必须跟着变——否则说明图里的数是硬编码的。
    Change one task's Lambda and the heatmap must move; otherwise the number is hardcoded.

    中文：取 `pb_t1_tg` 的 A3 行，把 λ 全部改成 +0.999。图里必须出现 +0.999，
          且原来那个 +0.00x 不再出现。这条如果不红，说明这张图根本没在读数。
    """
    before = _numbers(F1.build()[0])
    assert not any("+0.999" in item for item in before), "基线里不该有 +0.999，否则这条测试没意义"

    path = sandbox / "lambda_ci.csv"
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    columns = list(rows[0])
    for row in rows:
        if row["task"] == "pb_t1_tg" and row["arm"] == "a3_graph":
            row["lambda"] = "0.999"
            row["lambda_lo"] = "0.998"
            row["lambda_hi"] = "1.0"
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)

    after = _numbers(F1.build()[0])
    assert any("+0.999" in item for item in after), "改了表图却没变，说明这张图是硬编码的"


def test_shuffling_the_lambda_table_moves_the_tier_panel(sandbox: Path) -> None:
    """图 2 也不能是硬编码的：改表里的 λ 与 ρ，图里的数必须变。
    The tier panel must read its numbers from the table too.

    中文：两个细节值得记下来。
      ① 图 2 **只对 4 个极端任务标名字**，所以 λ 本身不出现在文本里——比文本比不出来。
         要看的是它读的 `spearman_rho`（来自汇总行）有没有跟着变。
      ② 汇总行与逐点行**必须一起改**：真实流程里 `h1_ordering.csv` 是 `06e` 一次生成的，
         两者天然一致；如果只改逐点行而汇总行不动，那是"表自己就自相矛盾"，
         不是图的问题。所以这条测试同时改两处，模拟一次真正的重算。
    """
    before = _numbers(F2.build()[0])
    path = sandbox / "h1_ordering.csv"
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    columns = list(rows[0])
    points = 0
    summaries = 0
    for row in rows:
        if row.get("scope") == "point":
            row["lambda"] = "0.777"
            # 区间必须一起改：只改点不改区间会让 `hi − y` 变负，误差棒画不出来——
            # 那报的是 ValueError，不是"图没变"，会把这条测试的意图掩盖掉。
            row["lambda_lo"] = "0.650"
            row["lambda_hi"] = "0.900"
            points += 1
        elif str(row.get("scope", "")).startswith("summary"):
            row["spearman_rho"] = "+0.777000"
            summaries += 1
    assert points and summaries, "h1_ordering.csv 缺逐点行或汇总行"
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)

    after = _numbers(F2.build()[0])
    assert not any("+0.777" in item for item in before), "基线里不该有 +0.777"
    assert any("+0.777" in item for item in after), "改了表图却没变，说明这张图是硬编码的"


def test_missing_cells_read_na_never_zero() -> None:
    """热图里 OpenPoly 的 A4 三格是 `n/a`（A4 没跑 OpenPoly），且那些格子里没有 0。
    Missing cells are n/a, and no zero is written in their place.

    中文：这是 D-6 验收标准④。一个没跑的任务画成 0，会被读成"测出来链信息没用"——
          比留白更糟，因为它看起来像结论。
    """
    texts = _texts(F1.build()[0])
    assert texts.count("n/a") >= 3, f"应当至少有 3 个 n/a（A4 的 OpenPoly 三格），实际 {texts.count('n/a')}"
    # 热图里所有单元格的中心文本都不是 "0" 或 "+0.000"（真值为 0 的格不存在于这批数据里）。
    assert "+0.000" not in texts, "不该有 +0.000 的格子：那会被读成测出来的 0"


def test_a_cell_averaging_partial_folds_says_how_many() -> None:
    """一格只平均了部分折时，图上必须写"k/n 行可算"。／ Partial coverage must be visible.

    中文：`pb26_homopolymer_density` 的 5 折里有 3 折 A2 的 R² ≥ 0.95，按 D-1 不输出 Λ。
          若不写"6/15 行可算"，读者会以为均值站在全部 15 行上——那是把覆盖率藏起来，
          比第一个版本直接画成 nan 更难发现。
    """
    texts = _texts(F1.build()[0])
    partial = [item for item in texts if "行可算" in item]
    assert partial, "没有任何格报可用行数，说明 _cell 又把 nan 平均进去了"
    assert any(item.startswith("6/") or item.startswith("3/") for item in partial), partial


# --------------------------------------------------------------------------- #
# D-10 ①：指标只有一份实现？实测是**两份**，所以必须逐位对齐
# --------------------------------------------------------------------------- #
def test_the_two_r2_implementations_agree_bit_for_bit() -> None:
    """bootstrap 用的 `ci._r2_rows` 与常规用的 `metrics.r2_score` 必须在同一输入上逐位相同。
    The vectorised R2 used by the bootstrap and the scalar one must agree bit for bit.

    中文：D-10 验收标准①要求"指标只有一份实现"。实测本仓库有**两份**：
          `analysis/ci.py:_r2_rows`（按行向量化，重采样用）与 `metrics/regression.py:81`（单次算）。
          两份实现本身不是错——重采样那 2,000 次必须向量化——但**它们必须逐位一致**，
          否则 bootstrap 的中心会与点估计差一点点，而这一点点在 Λ 的分母趋 0 时会被放大。
          这条测试就是"D-10 ①"在这份代码上的具体形态：不要求合并成一份，要求**可对拍**。
    """
    import numpy as np

    from chainworth.analysis import ci
    from chainworth.metrics import regression as R

    rng = np.random.default_rng(20261117)
    for n in (17, 300, 4096):
        truth = rng.normal(size=n)
        truth -= truth.mean()
        pred = truth + rng.normal(scale=0.4, size=n)
        scalar = R.r2(truth, pred)
        vector = float(ci._r2_rows(truth[None, :], pred[None, :])[0])
        assert scalar == pytest.approx(vector, rel=0, abs=1e-12), (n, scalar, vector)


def test_a_constant_label_is_nan_in_both_implementations() -> None:
    """全常数标签在两份实现里都必须是 `nan`（不是 0，也不是异常）。
    A constant label is nan in both, never zero and never an exception.

    中文：0 会被读成"模型完全不准"，`nan` 才是"这一格没有定义"。bootstrap 里若把这种
          重采样当 0 参与平均，区间会被人为压窄。
    """
    import math

    import numpy as np

    from chainworth.analysis import ci
    from chainworth.metrics import regression as R

    truth = np.full(50, 3.0)
    pred = np.full(50, 3.1)
    assert math.isnan(R.r2(truth, pred))
    assert math.isnan(float(ci._r2_rows(truth[None, :], pred[None, :])[0]))


def test_only_properties_that_change_sign_get_an_n_star() -> None:
    """图 5 的 n\\* 只能来自真正过零的性质：Tm 全窗为负，必须返回 None。
    Only properties that actually change sign get an n*; Tm must return None.

    中文：这一条是被一次**文字错误**逼出来的——报告 §5.5 曾给 Tm 报了一个 n\\* ≈ 50，
          而图 5 的实现（`_crossover` 只报第一个过零点、全窗同号就报窗上界不外推）
          与它对这张表的实测都是"全窗为负"。图的代码是对的，错的是正文。
          所以把这张表钉成断言：**3 个性质里只有 2 个有 n\\*，Tm 没有**。
          它守的不是"代码算得对"（代码本来就对），而是"正文别再说错"。
    """
    from chainworth.figures import fig5_crossover as F5

    def _rows(name: str) -> list[dict[str, str]]:
        with (Path("results/aggregate") / name).open(encoding="utf-8", newline="") as fh:
            return list(csv.DictReader(fh))

    unit = _rows("openpoly_cpu.csv")
    graph = _rows("openpoly_gpu_index.csv")
    paired = F5._paired(unit, graph)
    assert set(paired) == {"op_tg", "op_tm", "op_dielectric_constant_total"}

    got: dict[str, float | None] = {}
    for prop, points in paired.items():
        crossing, _note = F5._crossover(points)
        got[prop] = None if crossing is None else round(10 ** crossing)

    assert got["op_tm"] is None, "Tm 全窗为负，不该有 n*"
    assert got["op_tg"] == 56
    assert got["op_dielectric_constant_total"] == 59
    assert len([v for v in got.values() if v is not None]) == 2
