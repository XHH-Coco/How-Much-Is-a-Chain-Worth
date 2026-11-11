"""A-7 的断言测试：五个数对得上、数字能复算、改坏了能抓到、重复结构这个事实被锁住。
Assertions for A-7: the five numbers match, they are recomputable, tampering is caught, and the
duplicate-structure fact stays visible.

中文：
    用途：把分工书 A-7 的两条验收标准里能自动化的部分变成 `pytest`。四类：
          ① **口径**：五个靶子（741 行 / 3,985 标注 / Tg 443 / 介电 295 / Tm 168）逐个对上；
             28 个性质列一个不多一个不少；"不要用 counts 文件"这条警告有常数可指；
          ② **正向**：清单与"从原始 CSV 重算一遍"逐格相同，`verify()` 返回空；
          ③ **能抓住真错**：把清单里的某一个数改掉，`verify()` 必须报出来（否则清单就是
             一张没人核的纸条）；把原始表截短，计数器必须拒绝而不是给出一个"看起来正常"的数；
          ④ **锁住发现**：OpenPoly 的同一结构被多条文献记录重复登记、且标签互相矛盾——
             Tg 443 行只有 304 个结构、Tm 168 行只有 83 个、介电 295 行没有重复。这不是
             我们造的数，是数据本身的性质；把它写成用例，上游换版时测试会先叫。
    依赖：`data/raw/openpoly/**`（A-2 下载，不入库）+ `data/manifests/openpoly_property_counts.csv`
          （A-7 产物，入库）。原始 CSV 不在时整组跳过。
    配置：无。对外：pytest 用例。被谁用：CI、C/D（分母出处）、任何改 `io/openpoly.py` 的人。

English:
    Purpose: the automatable half of A-7's two acceptance criteria. Four groups: (1) convention -
    the five targets (741 rows / 3,985 annotations / Tg 443 / dielectric 295 / Tm 168) match one by
    one, and the 28 property columns are exactly the header minus the 4 identifier columns;
    (2) positive - the manifest equals a fresh recount cell by cell and `verify()` returns empty;
    (3) real-failure detection - change one number in the manifest and `verify()` must report it,
    and truncating the raw table must make the counter refuse rather than emit a plausible number;
    (4) locked finding - OpenPoly registers the same structure in several literature entries with
    contradictory labels: Tg 443 rows hold only 304 structures, Tm 168 rows only 83, dielectric 295
    rows none. That is a property of the data, not a number we invented; as a test it shouts when
    upstream changes.
    Depends on: data/raw/openpoly (A-2, not committed) and the A-7 manifest (committed).
    Config: none. Public API: pytest cases.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import pandas as pd
import pytest

from chainworth.io import openpoly

RAW = Path("data/raw/openpoly/data/final_polymer_properties_fromliterature.csv")
MANIFEST = Path("data/manifests/openpoly_property_counts.csv")
DOWNLOAD_MANIFEST = Path("data/manifests/openpoly_download_manifest.csv")

pytestmark = pytest.mark.skipif(
    not RAW.is_file() or not MANIFEST.is_file(),
    reason="缺 OpenPoly 原始 CSV 或 A-7 清单（原始数据不入库，先跑 00_download.py）",
)


@pytest.fixture(scope="module")
def frame() -> pd.DataFrame:
    """读一遍清单，供多个用例共用。 / Read the manifest once for all cases."""
    return pd.read_csv(MANIFEST)


# --------------------------------------------------------------------------- #
# ① 口径：五个靶子与列集合
# --------------------------------------------------------------------------- #


def test_columns_are_exactly_as_declared(frame: pd.DataFrame) -> None:
    """列名与顺序固定；前三个必须是分工书点名的列。"""
    assert list(frame.columns) == list(openpoly.COLUMNS)
    assert list(frame.columns[:3]) == [
        "property",
        "n_nonnull",
        "usable_after_harmonisation",
    ]


def test_property_rows_cover_every_column_of_the_raw_table(frame: pd.DataFrame) -> None:
    """28 个性质列一个不多一个不少，顺序与原始表头一致。"""
    raw = pd.read_csv(RAW)
    expected = openpoly.property_columns(raw)
    assert len(expected) == 28
    assert list(frame["property"]) == expected


def test_five_acceptance_targets(frame: pd.DataFrame) -> None:
    """验收①：741 行 / 3,985 标注 / Tg 443 / 介电 295 / Tm 168，逐个对上。"""
    assert set(frame["n_rows"]) == {741}
    assert int(frame["n_nonnull"].sum()) == 3_985

    def nonnull(name: str) -> int:
        return int(frame.loc[frame["property"] == name, "n_nonnull"].iloc[0])

    assert nonnull("Tg (K)") == 443
    assert nonnull("Dielectric Constant Total") == 295
    assert nonnull("Tm (K)") == 168


def test_forbidden_counts_file_is_named_without_underscore() -> None:
    """那个不许用的文件真名是 `..._counts_fromliterature.csv`（from 后面没有下划线）。

    Proposal 与分工书都多写了一个下划线；写错名字的人会以为文件不存在，进而自己造一个数。
    The forbidden file has no underscore after "from"; both documents misspell it, which is how
    someone ends up inventing the number instead of finding the file.
    """
    assert openpoly.FORBIDDEN_FILE == "final_polymer_property_counts_fromliterature.csv"
    assert "from_literature" not in openpoly.FORBIDDEN_FILE
    assert openpoly.FORBIDDEN_FILE_TG == 58_755


def test_raw_file_is_the_one_a2_downloaded() -> None:
    """原始 CSV 的 sha256 必须等于 A-2 下载清单里登记的那一行。"""
    digest = hashlib.sha256(RAW.read_bytes()).hexdigest()
    registered = pd.read_csv(DOWNLOAD_MANIFEST)
    row = registered.loc[
        registered["task_id"] == "data/final_polymer_properties_fromliterature.csv"
    ]
    assert len(row) == 1
    assert row["sha256"].iloc[0] == digest


# --------------------------------------------------------------------------- #
# ② 正向：清单 == 重算
# --------------------------------------------------------------------------- #


def test_manifest_equals_a_fresh_recount(frame: pd.DataFrame) -> None:
    """从原始 CSV 重算一遍，逐格相等。"""
    fresh = openpoly.build_frame(RAW)
    pd.testing.assert_frame_equal(frame, fresh, check_dtype=False)


def test_verify_returns_nothing_on_the_real_artefacts() -> None:
    """`--verify` 在真实产物上必须是空列表。"""
    assert openpoly.verify(RAW, MANIFEST) == []


def test_dielectric_three_columns_share_one_sample_set(frame: pd.DataFrame) -> None:
    """三个介电列的非空行完全相同（295），所以它们是同一批样本的三列，不是三份数据。"""
    counts = {
        name: int(frame.loc[frame["property"] == name, "n_nonnull"].iloc[0])
        for name in (
            "Dielectric Constant Electronic",
            "Dielectric Constant Ionic",
            "Dielectric Constant Total",
        )
    }
    assert set(counts.values()) == {295}


# --------------------------------------------------------------------------- #
# ④ 锁住发现：重复结构 + 标签矛盾
# --------------------------------------------------------------------------- #


def test_duplicate_structures_are_recorded_not_silently_dropped(frame: pd.DataFrame) -> None:
    """Tg 443 行只对应 304 个结构；Tm 168 行只对应 83 个；介电 295 行无重复。

    这三个数是 D-26 的依据：OpenPoly 是文献汇编，同一聚合物被多篇文献以不同数值重复登记。
    These three numbers justify D-26: OpenPoly is a literature compilation, so the same polymer
    is registered several times with different values.
    """

    def row(name: str) -> pd.Series:
        return frame.loc[frame["property"] == name].iloc[0]

    tg = row("Tg (K)")
    assert (tg["n_nonnull"], tg["n_unique_structures"]) == (443, 304)
    assert (tg["n_duplicate_rows"], tg["n_conflicting_groups"]) == (139, 57)

    tm = row("Tm (K)")
    assert (tm["n_nonnull"], tm["n_unique_structures"]) == (168, 83)
    assert (tm["n_duplicate_rows"], tm["n_conflicting_groups"]) == (85, 33)

    dielectric = row("Dielectric Constant Total")
    assert (dielectric["n_nonnull"], dielectric["n_unique_structures"]) == (295, 295)
    assert dielectric["n_duplicate_rows"] == 0


def test_duplicate_groups_and_rows_are_both_counted(frame: pd.DataFrame) -> None:
    """逐列重算"重复组"与"冲突组"，必须与清单相等（28 列全核）。

    "行"与"组"是两个单位：2 个结构各重复 3 次 = 6 行重复但只有 2 组。混用会把 139 说成 57。
    Rows and groups are different units; conflating them turns 139 into 57.
    """
    raw = pd.read_csv(RAW)
    canonical = openpoly._canonical_psmiles(raw["PSMILES"])

    for name in frame["property"]:
        label = pd.to_numeric(raw[name], errors="coerce")
        present = pd.DataFrame({"structure": canonical, "label": label}).dropna()
        grouped = present.groupby("structure")["label"]
        sizes = grouped.size()
        duplicate_groups = sizes[sizes > 1]
        spread = grouped.agg(lambda s: s.max() - s.min())[duplicate_groups.index]

        row = frame.loc[frame["property"] == name].iloc[0]
        assert int(row["n_duplicate_groups"]) == len(duplicate_groups), f"{name} 重复组数不符"
        assert int(row["n_conflicting_groups"]) == int(
            (spread > openpoly.CONFLICT_TOLERANCE).sum()
        ), f"{name} 冲突组数不符"


def test_key_properties_have_no_pure_redundant_duplicate(frame: pd.DataFrame) -> None:
    """三个点名性质里，**每一组重复都在互相矛盾**——没有一组是可以白删的冗余。

    Tg 57/57 组冲突、Tm 33/33 组冲突、介电 0 组。所以对这三个性质，"去重"从来不是清理，
    而是在几个互相矛盾的文献值里挑一个（D-26）。其余性质里有 9 组纯冗余，但那不是 A-7 的口径。
    In the three singled-out properties every duplicate group disagrees; dedupe there is never
    cleanup but picking one of several contradictory literature values (D-26).
    """
    for name in openpoly.KEY_PROPERTIES:
        row = frame.loc[frame["property"] == name].iloc[0]
        assert row["n_conflicting_groups"] == row["n_duplicate_groups"], (
            f"{name}：{row['n_duplicate_groups']} 组重复中有 "
            f"{int(row['n_duplicate_groups']) - int(row['n_conflicting_groups'])} 组标签一致，"
            "'去重=挑值'这一条要重新论证"
        )


def test_usable_after_harmonisation_equals_nonnull_because_nothing_ran(frame: pd.DataFrame) -> None:
    """A-3 未对 OpenPoly 执行调和，所以两列相等是**声明过的状态**，不是漏填。"""
    assert set(frame["harmonisation_status"]) == {openpoly.HARMONISATION_STATUS}
    assert (frame["usable_after_harmonisation"] == frame["n_nonnull"]).all()


# --------------------------------------------------------------------------- #
# ③ 能抓住真错
# --------------------------------------------------------------------------- #


def test_verify_catches_one_changed_number() -> None:
    """把清单里 Tg 的 443 改成 444，`verify()` 必须报出来。

    用的临时文件写在清单同目录、用完立刻删掉：不碰系统临时目录（pytest 收尾会整目录删，
    在某些沙箱里会弹权限框），也不留垃圾。
    The scratch file lives next to the manifest and is removed immediately: no system temp
    directory (pytest's bulk cleanup trips some sandboxes) and no leftovers.
    """
    scratch = MANIFEST.with_name("_a7_tampered_scratch.csv")
    try:
        frame = pd.read_csv(MANIFEST)
        frame.loc[frame["property"] == "Tg (K)", "n_nonnull"] = 444
        frame.to_csv(scratch, index=False, lineterminator="\n")

        problems = openpoly.verify(RAW, scratch)
        assert any("n_nonnull" in problem for problem in problems)
    finally:
        scratch.unlink(missing_ok=True)


def test_verify_catches_a_missing_manifest() -> None:
    """清单不在时要给人一条能照做的提示，而不是崩。"""
    problems = openpoly.verify(RAW, MANIFEST.with_name("_a7_never_created.csv"))
    assert problems and "--write" in problems[0]


def test_counter_refuses_a_truncated_source_table() -> None:
    """原始表被截短时必须当场拒绝，不能给出一个"看起来正常"的数。"""
    raw = pd.read_csv(RAW).head(100)
    with pytest.raises(ValueError, match="行数"):
        openpoly.count_properties(raw)


def test_counter_refuses_a_missing_identifier_column() -> None:
    """缺 PSMILES 列时不能静默把每一行都当成"同一结构"。"""
    raw = pd.read_csv(RAW).drop(columns=["PSMILES"])
    with pytest.raises(ValueError, match="标识列"):
        openpoly.count_properties(raw)
