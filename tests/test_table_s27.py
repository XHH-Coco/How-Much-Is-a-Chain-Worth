"""A-8 的断言测试：39 行齐、Tier 14/15/10、合计 42,169、与三处来源对得上、改坏了能抓到。
Assertions for A-8: 39 rows, tiers 14/15/10, 42,169 rows total, agreement with three independent
sources, and tampering is caught.

中文：
    用途：把分工书 A-8 的三条验收标准里能自动化的部分变成 `pytest`。五类：
          ① **口径**：39 行；Tier = 14 / 15 / 10；`n_rows` 之和 42,169；页脚出处是 S52/S53；
             几个带下标的标签（σy、ε′′、ΔHf）拼写被锁住——重构时不至于悄悄换成 ASCII；
          ② **正向**：CSV 与 `build_frame()` 逐格相同；`verify()` 返回空；39 个 `task_id`
             全部能在 A-2 下载清单里找到，且**逐条** `n_rows` 相等；
          ③ **三处对拍**：与 A-2 清单（条数）、与 Proposal 附录 A（23 个已命名 tier）、
             与 `configs/tasks/*.yaml`（A-4 冻结的 tier）。第三处是 A-8 真正的问题——
             主表 6 个任务若与 SI 不符，§3.1 的"以主文为准、替换为备用任务"才需要触发；
          ④ **分歧被锁死**：S27 与附录 A 的分歧**恰好一条**且是 `σy (yield strength)`
             （S27 记 3、附录 A 记 2）。这条测试的作用是：如果哪天冒出第二条分歧，
             说明有人改了其中一处，得重新走 A-8 的流程，而不是当没看见；
          ⑤ **能抓住真错**：改一个 n、改一个 tier、删一行、换成不存在的 task_id、
             制造一条没有说明的分歧、让主表 tier 与配置不符——六种改法逐个必须被抓；
             另有一组回查 PDF 原文（需要 SI 原件 + pypdf，缺任一则跳过）。
    依赖：`data/external/table_s27.csv`（A-8 产物，**不入库**）→ 不在时整组跳过；
          `data/manifests/polybench_download_manifest.csv` 与 `configs/tasks/*.yaml`（入库）。
    配置：无。对外：pytest 用例。被谁用：CI、D 的图表脚注、任何改 `io/table_s27.py` 的人。

English:
    Purpose: the automatable half of A-8's three acceptance criteria. Five groups: (1) convention -
    39 rows, tiers 14/15/10, n_rows summing to 42,169, sources on pages S52/S53, and the Greek
    subscript labels frozen so a refactor cannot silently ASCII-fy them; (2) positive - the CSV
    equals `build_frame()` cell by cell, `verify()` is empty, and all 39 task_ids exist in the A-2
    download manifest with matching row counts; (3) cross-checks against three independent sources
    (A-2 counts, Proposal Appendix A, and the frozen `configs/tasks/*.yaml`); (4) the known
    disagreement is pinned as exactly one row (`σy`: S27 says 3, Appendix A says 2) so a second one
    cannot appear unnoticed; (5) real-failure detection - six tampering modes must each be caught,
    plus an optional pass that re-reads the source PDF.
    Depends on: the A-8 CSV (not committed) -> the group skips when absent; the A-2 manifest and
    task configs (committed). Config: none. Public API: pytest cases.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from chainworth.io import table_s27

OUTPUT = table_s27.OUTPUT_PATH
DOWNLOAD_MANIFEST = table_s27.DOWNLOAD_MANIFEST
PROPOSAL = table_s27.PROPOSAL_MD
PDF = Path("data/raw/polybench_si/ps6c00094_si_001.pdf")

pytestmark = pytest.mark.skipif(
    not OUTPUT.is_file(),
    reason="缺 A-8 转录产物（data/external/ 不入库，先跑 `python -m chainworth.io.table_s27 --write`）",
)


@pytest.fixture(scope="module")
def frame() -> pd.DataFrame:
    """读一遍转录表，供多个用例共用。 / Read the CSV once for all cases."""
    return pd.read_csv(OUTPUT, dtype={"dummy_mae": str, "dummy_rmse": str, "appendix_a_tier": str})


def _scratch(name: str = "_a8_tampered_scratch.csv") -> Path:
    """篡改测试用的临时文件，放在产物同目录、用完即删（不碰系统临时目录）。"""
    return OUTPUT.with_name(name)


# --------------------------------------------------------------------------- #
# ① 口径：不依赖产物，直接查常量
# --------------------------------------------------------------------------- #
def test_expected_targets_are_self_consistent() -> None:
    """三个靶子彼此不矛盾：Tier 计数加起来 = 行数 = 39。"""
    assert sum(table_s27.EXPECTED_TIER_COUNTS.values()) == table_s27.EXPECTED_ROWS == 39
    assert table_s27.EXPECTED_TOTAL_ROWS == 42_169


def test_rows_constant_has_39_entries_with_right_tier_counts() -> None:
    """模块里的转录常量本身：39 行、Tier 14/15/10。"""
    assert len(table_s27.ROWS) == 39
    counts: dict[int, int] = {}
    for row in table_s27.ROWS:
        counts[row.tier] = counts.get(row.tier, 0) + 1
    assert counts == {1: 14, 2: 15, 3: 10}


def test_row_counts_sum_to_the_published_total() -> None:
    """39 个 n 加起来必须是 42,169——这也是 S53 页脚那句话。"""
    assert sum(row.n_rows for row in table_s27.ROWS) == 42_169


def test_greek_subscript_labels_are_not_asciified() -> None:
    """带下标的标签保留下标记号：σy / ε′′ / ΔHf 是 S27 的原样写法，不是 ascii 近似。

    这不是洁癖：`ε′`（介电常数，Tier 1）与 `ε′′`（介电损耗，Tier 3）只差一个 prime，
    而两者一个在 Tier 1 一个在 Tier 3。把 prime 丢了，两行就分不清了。
    """
    labels = {row.s27_label for row in table_s27.ROWS}
    for must in ("σy (yield strength)", "ε′′ (dielectric loss)", "ε′(dielectric constant)"):
        assert must in labels, f"标签 {must!r} 不见了（可能被 ascii 化了）"
    sigma_y = next(row for row in table_s27.ROWS if row.s27_label.startswith("σy"))
    eps_loss = next(row for row in table_s27.ROWS if row.s27_label.startswith("ε′′"))
    eps_const = next(row for row in table_s27.ROWS if row.s27_label.startswith("ε′("))
    assert (eps_const.tier, eps_loss.tier) == (1, 3), "介电常数应在 Tier 1、介电损耗应在 Tier 3"


def test_every_row_records_its_page() -> None:
    """每行都标出处页码，且只有 S52 / S53 两页（表跨页）。"""
    pages = {row.source_page for row in table_s27.ROWS}
    assert pages == {"S52", "S53"}
    assert sum(1 for row in table_s27.ROWS if row.source_page == "S52") == 19
    assert sum(1 for row in table_s27.ROWS if row.source_page == "S53") == 20


# --------------------------------------------------------------------------- #
# ② 正向：产物与三处来源对得上
# --------------------------------------------------------------------------- #
def test_columns_are_frozen(frame: pd.DataFrame) -> None:
    """列序冻结：D 的脚本按位置取列，插一列就全错。"""
    assert list(frame.columns) == table_s27.COLUMNS


def test_csv_equals_build_frame(frame: pd.DataFrame) -> None:
    """磁盘上的 CSV 必须等于 `build_frame()` 重算的结果。

    这条挡的是"CSV 被手改过、而模块没跟"：那种状态下 `--verify` 通过、`--write` 又改回去，
    两份东西悄悄分叉。比较时把 appedix 列的空值统一，避免 NaN 与 "" 的假差异。
    """
    fresh = table_s27.build_frame()
    left = frame.fillna("").astype(str)
    right = fresh.fillna("").astype(str)
    pd.testing.assert_frame_equal(left, right, check_dtype=False)


def test_task_ids_are_unique(frame: pd.DataFrame) -> None:
    assert frame["task_id"].is_unique


def test_every_n_matches_the_official_download_manifest(frame: pd.DataFrame) -> None:
    """逐条核对：S27 的 N 列 == A-2 下载清单的 `n_rows`。

    这是防手抄错最有力的一条：把 8,055 抄成 8,058，这条立刻红。
    """
    official = dict(zip(*[pd.read_csv(DOWNLOAD_MANIFEST)[c] for c in ("task_id", "n_rows")]))
    mismatched = {
        row.task_id: (int(row.n_rows), int(official[row.task_id]))
        for row in frame.itertuples()
        if int(official[row.task_id]) != int(row.n_rows)
    }
    assert not mismatched, f"这些任务的条数与官方清单不符：{mismatched}"


def test_total_equals_the_download_manifest_sum(frame: pd.DataFrame) -> None:
    """两条路径同一个数：S27 自己求和，与 A-2 清单求和。"""
    assert int(frame["n_rows"].sum()) == int(pd.read_csv(DOWNLOAD_MANIFEST)["n_rows"].sum()) == 42_169


def test_verify_passes(frame: pd.DataFrame) -> None:
    """`verify()` 必须返回空列表（含 configs 对拍那一层）。"""
    assert table_s27.verify(OUTPUT) == []


def test_appendix_a_parses_23_named_tiers() -> None:
    """Proposal 附录 A 解析出 39 行，其中 23 个已命名 tier（8 + 8 + 7）。"""
    appendix = table_s27._appendix_a_tiers()
    assert len(appendix) == 39
    named = [tier for tier in appendix.values() if tier is not None]
    assert len(named) == 23
    assert sorted(named).count(1) == 8 and sorted(named).count(2) == 8 and sorted(named).count(3) == 7


def test_main_table_six_all_agree_with_s27(frame: pd.DataFrame) -> None:
    """主表 6 个 PolyBench 任务的 tier 与 S27 完全一致 —— 验收③的替换条款**未触发**。

    这是 A-8 最要紧的一条结论：S27 与主文如果在这里打架，主表就得换人。
    实测一致，所以 A-4 冻结的 8 任务主表一个字都不用改。
    """
    import yaml

    s27_tier = dict(zip(frame["task_id"], frame["tier"]))
    main_rows = []
    for path in sorted(Path("configs/tasks").glob("*.yaml")):
        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        if data.get("role") != "main" or data.get("dataset_name") not in s27_tier:
            continue
        main_rows.append((data["dataset_name"], int(data["tier"]), int(s27_tier[data["dataset_name"]])))
    assert len(main_rows) == 6, f"主表 PolyBench 任务应为 6 个，实际 {len(main_rows)}"
    disagreed = [name for name, cfg, s27 in main_rows if cfg != s27]
    assert not disagreed, f"主表任务与 S27 不一致：{disagreed}"


def test_the_only_disagreement_is_sigma_y(frame: pd.DataFrame) -> None:
    """S27 与附录 A 的分歧**恰好一条**，且是 `σy (yield strength)`（S27=3、附录 A=2）。

    出现第二条就说明有人动了其中一处：要么 S27 转录错了，要么 Proposal 改了却没重跑 A-8。
    两种情况都该停下来重新走一遍，而不是当没看见。
    """
    appendix = table_s27._appendix_a_tiers()
    disagreements = [
        (row.task_id, int(row.tier), appendix[row.task_id])
        for row in frame.itertuples()
        if appendix.get(row.task_id) is not None and int(row.tier) != int(appendix[row.task_id])
    ]
    assert disagreements == [("Tensile stress strength at yield", 3, 2)], disagreements


def test_sigma_y_is_not_on_the_main_table(frame: pd.DataFrame) -> None:
    """σy 不是主表任务——所以它那个 tier 分歧不影响任何实验。"""
    import yaml

    names = set()
    for path in sorted(Path("configs/tasks").glob("*.yaml")):
        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        names.add(data.get("dataset_name"))
    assert "Tensile stress strength at yield" not in names


def test_sigma_y_shift_is_the_only_way_to_get_14_15_10(frame: pd.DataFrame) -> None:
    """把 σy 改回附录 A 的 2，Tier 计数就变成 14/16/9 —— 与主文 14/15/10 矛盾。

    这条不是重复上面那条断言，而是给出"为什么 S27 的 3 不是笔误"的证据：
    附录 A 的 16 个 `pending S27` 补上后，Tier 2 本会是 16、Tier 3 本会是 9
    （附录 A 已命名的 8+8+7，加上 pending 的 6+8+2）。**只有**把 σy 从 2 挪到 3，
    三个数才恰好落回主文的 14/15/10。一个手抄错误不会同时让三个数字对上，
    所以 S27 的 σy=3 是自洽的，附录 A 的 2 才是那个把 "tensile properties" 一概归 2 的推断。
    """
    appendix = table_s27._appendix_a_tiers()
    pending_s27 = [row.task_id for row in frame.itertuples() if appendix.get(row.task_id) is None]
    assert len(pending_s27) == 16

    # 按附录 A 的口径（σy = 2）把 39 行补齐，看 Tier 计数
    # 注意 `dict.get(k, d)` 在"键存在但值是 None"（pending S27）时返回 None 而不是 d，
    # 所以要显式判断，否则 16 个 pending 会整体变成 None 一档。
    naive: dict[int, int] = {}
    for row in frame.itertuples():
        a_tier = appendix.get(row.task_id)
        tier = int(row.tier) if a_tier is None else int(a_tier)
        naive[tier] = naive.get(tier, 0) + 1
    assert naive == {1: 14, 2: 16, 3: 9}, f"按附录 A 口径应是 14/16/9，实际 {naive}"

    # 按 S27 口径（σy = 3）才是主文那三个数
    actual = frame["tier"].value_counts().to_dict()
    assert actual == {1: 14, 2: 15, 3: 10}


# --------------------------------------------------------------------------- #
# ③ 能抓住真错：六种改法逐个必须被抓
# --------------------------------------------------------------------------- #
def _tampered(mutate) -> list[str]:
    """把产物复制一份、交给 `mutate` 改坏、跑 `verify()`、删掉。"""
    scratch = _scratch()
    try:
        frame = pd.read_csv(OUTPUT, dtype={"dummy_mae": str, "dummy_rmse": str, "appendix_a_tier": str})
        mutate(frame)
        frame.to_csv(scratch, index=False, lineterminator="\n")
        return table_s27.verify(scratch)
    finally:
        scratch.unlink(missing_ok=True)


def test_catches_one_changed_n() -> None:
    """把 Tg 的 8,055 改成 8,058 —— 与 A-2 清单对不上，必须报。"""

    def mutate(frame: pd.DataFrame) -> None:
        frame.loc[frame["task_id"] == "Glass transition temperature", "n_rows"] = 8058

    problems = _tampered(mutate)
    assert any("Glass transition temperature" in problem for problem in problems), problems


def test_catches_one_changed_tier() -> None:
    """把 Tg 从 Tier 1 挪到 Tier 3 —— Tier 计数变成 13/15/11，必须报。"""

    def mutate(frame: pd.DataFrame) -> None:
        frame.loc[frame["task_id"] == "Glass transition temperature", "tier"] = 3

    problems = _tampered(mutate)
    assert any("Tier 计数" in problem for problem in problems), problems


def test_catches_a_dropped_row() -> None:
    """删掉一行 —— 39 → 38，行数与总和两条都该报。"""

    def mutate(frame: pd.DataFrame) -> None:
        frame.drop(index=frame.index[-1], inplace=True)

    problems = _tampered(mutate)
    assert any("行数" in problem for problem in problems), problems
    assert any("n_rows 之和" in problem for problem in problems), problems


def test_catches_an_unknown_task_id() -> None:
    """把一个 task_id 换成 A-2 清单里没有的名字 —— 这条是"映射失效"的哨兵。"""

    def mutate(frame: pd.DataFrame) -> None:
        frame.loc[frame.index[0], "task_id"] = "Glass transition temperature (typo)"

    problems = _tampered(mutate)
    assert any("找不到" in problem for problem in problems), problems


def test_catches_a_silent_disagreement_with_appendix_a() -> None:
    """把某行的 `appendix_a_tier` 改掉而附录 A 没变 —— 必须报"两处说法不一致"。"""

    def mutate(frame: pd.DataFrame) -> None:
        mask = frame["task_id"] == "Glass transition temperature"
        frame.loc[mask, "appendix_a_tier"] = 2

    problems = _tampered(mutate)
    assert any("附录 A 现为" in problem for problem in problems), problems


def test_catches_a_disagreement_without_a_note() -> None:
    """把 tier 改到与附录 A 不同、又不写 `tier_note` —— 必须报"没有说明"。"""

    def mutate(frame: pd.DataFrame) -> None:
        mask = frame["task_id"] == "Glass transition temperature"
        frame.loc[mask, "tier"] = 3
        frame.loc[mask, "tier_note"] = ""

    problems = _tampered(mutate)
    assert any("没有在 tier_note 里说明" in problem for problem in problems), problems


def test_catches_a_main_table_tier_conflict_with_configs() -> None:
    """把主表的 Tensile modulus 从 Tier 2 改成 Tier 3 —— 与冻结的 config 冲突，必须报。

    这一条直接对应用户要的"验收③"：主表与 SI 打架时，脚本必须叫，而不是静静通过。
    """

    def mutate(frame: pd.DataFrame) -> None:
        frame.loc[frame["task_id"] == "Tensile modulus", "tier"] = 3

    problems = _tampered(mutate)
    assert any("配置冻结的 tier" in problem for problem in problems), problems


def test_missing_file_gives_an_actionable_message() -> None:
    """产物不在时，提示里要给出那条能照做的命令，而不是崩。"""
    problems = table_s27.verify(OUTPUT.with_name("_a8_never_created.csv"))
    assert problems and "--write" in problems[0]


# --------------------------------------------------------------------------- #
# ④ 可选：回查 PDF 原文
# --------------------------------------------------------------------------- #
needs_pdf = pytest.mark.skipif(
    not PDF.is_file(),
    reason="SI 原件不在（156 MB，不入库）；A-8 的 PDF 回查跳过",
)


@needs_pdf
def test_pdf_reread_finds_all_39_rows() -> None:
    """39 行逐行回查 PDF：每行的 N / MAE / RMSE 三个数字必须出现在同一文本行里。"""
    pytest.importorskip("pypdf")
    problems = table_s27.verify(OUTPUT, pdf=PDF)
    assert problems == [], problems


@needs_pdf
def test_pdf_reread_catches_a_fabricated_mae() -> None:
    """把一个 MAE 改成 999.9 —— A-2 清单抓不到（它只管条数），但 PDF 回查必须抓。"""

    def mutate(frame: pd.DataFrame) -> None:
        frame.loc[frame["task_id"] == "Glass transition temperature", "dummy_mae"] = "999.9"

    pytest.importorskip("pypdf")
    scratch = _scratch("_a8_tampered_mae.csv")
    try:
        frame = pd.read_csv(OUTPUT, dtype={"dummy_mae": str, "dummy_rmse": str, "appendix_a_tier": str})
        mutate(frame)
        frame.to_csv(scratch, index=False, lineterminator="\n")
        # 不查 PDF 时抓不到 —— 这正是文档里写明的局限
        assert table_s27.verify(scratch) == [], "这条篡改本来就不该被浅层校验发现"
        problems = table_s27.verify(scratch, pdf=PDF)
        assert any("找不到同时含" in problem for problem in problems), problems
    finally:
        scratch.unlink(missing_ok=True)
