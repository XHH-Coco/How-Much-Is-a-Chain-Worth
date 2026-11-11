"""A-6 的断言测试：结构级表征算对了、只算一次、没把标签带进来、改坏了能抓到。
Assertions for A-6: the structure-level features are right, computed once, label-free, and a
tampered file is caught.

中文：
    用途：把分工书 A-6 的四条验收标准里能自动化的部分变成 `pytest`。四类：
          ① **口径**：wPSMILES 与 PolyBench26 官方实现逐字相同；描述符列冻结在 205 列；
             星号数不是 2 时**返回 None 而不是硬拼一个假串**；
          ② **正向**：`verify_outputs()` 在真实产物上返回空列表（deep 用 `CHAINWORTH_SLOW=1` 开）；
          ③ **边界**：特征文件里不许出现标签/`row_id` 列——那是 B 的装配工作，混进来他的验收
             就没法独立成立；
          ④ **能抓住真错**：改一个描述符数值，浅层校验（哈希）在清单被同步改掉后**看不见**，
             `_deep_check_frame()` 必须看见。第 ④ 类才是这条流水线的价值所在——只断言
             "没报错"的测试，拦不住"文件和清单一起被改"。
    依赖：`data/processed/**` 与 `data/manifests/**`（A-6 产物，**不入库**，由
          `scripts/02_features.py` 重建）。缺产物时相关用例跳过，纯函数用例照跑。
    配置：环境变量 `CHAINWORTH_SLOW=1` 打开 2 分钟的 deep 全量重算。
    对外：pytest 用例。被谁用：CI、B（重建后自检）、任何改了 `chem/` 的人。

English:
    Purpose: the automatable half of A-6's four acceptance criteria. Four groups: (1) convention -
    wPSMILES byte-equal to PolyBench26's official implementation, descriptor columns frozen at 205,
    and an asterisk count other than two yields None instead of a plausible-looking fake;
    (2) positive - `verify_outputs()` returns an empty list on the real artefacts (deep runs under
    `CHAINWORTH_SLOW=1`); (3) boundary - no label or `row_id` columns, because assembly is B's job
    and their criteria cannot stand independently otherwise; (4) real-failure detection - change one
    descriptor value and the shallow pass (hash) is blind once the manifest is updated to match,
    while `_deep_check_frame()` must see it. Group 4 is where this pipeline's value lives: a test
    that only asserts "no exception" cannot catch "file and manifest edited together".
    Depends on: data/processed and data/manifests (A-6 artefacts, **not committed**, rebuilt by
    scripts/02_features.py). Artefact-dependent cases skip when absent; pure cases always run.
    Config: `CHAINWORTH_SLOW=1` enables the two-minute deep recomputation.
    Public API: pytest cases. Callers: CI, B (self-check after rebuilding), anyone touching chem/.
"""

from __future__ import annotations

import json
import os
import sqlite3

import numpy as np
import pandas as pd
import pytest

from chainworth.chem import descriptors as D
from chainworth.chem import notation as N
from chainworth.chem import structures as S

HAS_ARTIFACTS = S.STRUCTURES_PATH.is_file() and S.FEATURES_MANIFEST.is_file()

needs_artifacts = pytest.mark.skipif(
    not HAS_ARTIFACTS,
    reason="缺 A-6 产物（data/processed 不入库）：先跑 `python scripts/02_features.py --source all`",
)

needs_slow = pytest.mark.skipif(
    os.environ.get("CHAINWORTH_SLOW") != "1",
    reason="deep 全量重算约 2 分钟：设 CHAINWORTH_SLOW=1 才跑",
)

# 期望的特征文件列序：两个键 + 205 描述符 + 2048 位指纹。
# The expected feature-file column order: two keys, 205 descriptors and 2048 fingerprint bits.
EXPECTED_FEATURE_COLUMNS = (
    ["structure_id", "psmiles_canonical"]
    + D.descriptor_columns()
    + [f"{S.FINGERPRINT_PREFIX}{index:04d}" for index in range(D.FINGERPRINT_BITS)]
)


# ---- ① 口径：与官方实现逐字对拍 / Convention: byte-equal with the official implementation ----


@pytest.mark.parametrize(
    "psmiles",
    [
        "*C*",
        "*CC*",
        "*CC(C)*",
        "*CC(*)C",
        "*CC(c1ccccc1)*",
        "*O*",
        "*N=C*",
        "*C(=O)*",
        "*C(F)(F)*",
        "*CC(O)CC*",
    ],
)
def test_wpsmiles_matches_the_official_implementation(psmiles: str) -> None:
    """我们写的转换必须与 PolyBench26 官方 `PSMILES_to_wPSMILES()` 逐字节相同。
    Our conversion must be byte-identical to PolyBench26's official `PSMILES_to_wPSMILES()`.

    中文：
        用途：wPSMILES 是 B 建图的输入。差一个字符（比如后缀少一段、连接点编号反了），
              图就建错，而且**不会报错**——训练照跑，只是学的是另一个分子。
              这里不是"看起来像"的比较，是字符串相等的比较。
        输入：psmiles —— 恰好两个星号的 PSMILES（参数化）。
        输出：无（断言）。
        数据来源：`notation.official_reference_wpsmiles()`（官方实现逐字副本）。
        配置来源：无。副作用：无。异常：AssertionError —— 两边不等。

    English:
        Purpose: wPSMILES feeds B's graph builder. One wrong character (a missing suffix segment,
                 swapped attachment numbers) builds the wrong graph and **nothing raises** - training
                 runs fine on a different molecule. This is string equality, not resemblance.
        Args: psmiles - a two-asterisk PSMILES (parametrised). Returns: none (assertions).
        Data source: `notation.official_reference_wpsmiles()`, a verbatim copy of the official code.
        Side effects: none. Raises: AssertionError when the two differ.
    """
    assert N.to_wpsmiles(psmiles) == N.official_reference_wpsmiles(psmiles)


@pytest.mark.parametrize("psmiles", ["", "*", "*C", "*CC", "*CC*C*", "*C(*)C*", "*C*C*C*"])
def test_wpsmiles_refuses_an_irregular_asterisk_count(psmiles: str) -> None:
    """星号数不是 2 时必须返回 None，不许硬拼一个"看起来像 wPSMILES"的串。
    A count other than two must yield None, never a plausible-looking wPSMILES.

    中文：
        用途：官方实现用 `split('*', 2)`，多出来的星号会**留在串里**、后缀照拼——产出一个
              语法上像、化学上错的东西。实测 PolyBench 有 116 个唯一结构是这种（3/4/6 个星号，
              支化或嵌段）。这条测试把"我们改成 None"这个决定钉住：将来有人"顺手兼容一下"，
              测试立刻变红。
        输入：psmiles —— 星号数不为 2 的样本。输出：无。数据来源：无。
        配置来源：`notation.HOMOPOLYMER_STAR_COUNT`。副作用：无。
        异常：AssertionError —— 居然返回了字符串。

    English:
        Purpose: the official implementation splits with maxsplit=2, so extras stay in the string
                 and the suffix is appended anyway - syntactically plausible, chemically wrong. 116
                 unique PolyBench structures are like this (3/4/6 asterisks, branched or block).
                 This test pins our "return None" decision so a future "just make it work" change
                 goes red immediately.
        Args: psmiles samples. Returns: none. Data source: none.
        Raises: AssertionError if a string comes back.
    """
    assert N.to_wpsmiles(psmiles) is None
    assert N.count_stars(psmiles) != N.HOMOPOLYMER_STAR_COUNT


def test_canonical_form_merges_equivalent_spellings() -> None:
    """同一种分子的两种写法必须规范成同一个键，且规范化是幂等的。
    Two spellings of one molecule must canonicalise to one key, and the map must be idempotent.

    中文：
        用途：去重键依赖这件事。`*CC(C)*` 与 `*CC(*)C` 是同一个图（三个碳 + 两个连接点），
              只有规范形认得出；不幂等则"规范两次"会漂移，结构表就出现重复行。
        输入：无。输出：无。数据来源：无。配置来源：无。副作用：无。
        异常：AssertionError —— 两种写法给出不同键，或规范两次不一致。
    English:
        Purpose: the dedupe key depends on this. `*CC(C)*` and `*CC(*)C` are one graph (three
                 carbons, two attachment points) and only the canonical form sees it; a
                 non-idempotent map drifts and the structure table grows duplicate rows.
        Args: none. Returns: none. Side effects: none.
        Raises: AssertionError on differing keys or a non-idempotent map.
    """
    left = N.canonical_psmiles("*CC(C)*")
    right = N.canonical_psmiles("*CC(*)C")
    assert left is not None and left == right
    assert N.canonical_psmiles(left) == left


def test_descriptor_columns_are_the_official_205() -> None:
    """描述符列冻结在 205 列，官方那 5 个必须不在里面，且列序稳定。
    The descriptor columns are frozen at 205, the official five are absent, and the order is stable.

    中文：
        用途：205 = 210 − 5。少丢一个，就有 4 列对每个含 `*` 的分子恒为 NaN；多丢一个，
              就与 PolyBench26 的 `RDKit_RF` 基线对不上话（D-25）。
        输入：无。输出：无。数据来源：RDKit 描述符注册表。
        配置来源：`descriptors.DROPPED_DESCRIPTORS`。副作用：无。
        异常：AssertionError —— 列数不是 205、或该丢的没丢、或两次调用列序不同。

    English:
        Purpose: 205 = 210 - 5. Dropping fewer leaves four columns constantly NaN for every
                 `*`-containing molecule; dropping more breaks comparability with PolyBench26's
                 `RDKit_RF` baseline (D-25).
        Args: none. Returns: none. Data source: RDKit's descriptor registry.
        Config: `descriptors.DROPPED_DESCRIPTORS`. Side effects: none.
        Raises: AssertionError on a wrong count, a surviving dropped name, or an unstable order.
    """
    columns = D.descriptor_columns()
    assert len(columns) == 205, f"描述符列数应为 205，实际 {len(columns)}"
    for dropped in D.DROPPED_DESCRIPTORS:
        assert dropped not in columns
    assert "MolWt" in columns and "TPSA" in columns
    assert D.descriptor_columns() == columns, "两次调用给出了不同的列序"


def test_fingerprint_is_a_fixed_width_bitmap() -> None:
    """指纹是定长 2048 位、可重算；不同分子给出不同位图。
    The fingerprint is a fixed-width 2048-bit bitmap, reproducible, and differs between molecules.

    中文：
        用途：钉住两件事。① 存的是**定长位图**（256 字节 = 2048/8），不是 RDKit 的稀疏编码——
              第一版用了 `fp.ToBinary()`，它给的是 Daylight 编码，置位少时只有 18 字节，
              读回来直接把"18 字节"当成"位图长度不对"；② 同一分子两次算出同一位图，
              否则缓存与 deep 校验都会自相矛盾。
        输入：无。输出：无。数据来源：`D.compute()`。
        配置来源：`D.FINGERPRINT_BITS`、`D.FINGERPRINT_RADIUS`。副作用：无。
        异常：AssertionError —— 长度不对、不可重算、或两个分子给出相同位图。

    English:
        Purpose: pins two things. (1) What is stored is a **fixed-width bitmap** (256 bytes =
                 2048/8), not RDKit's sparse encoding - the first version used `fp.ToBinary()`,
                 which returns the Daylight encoding (18 bytes when few bits are set) and read back
                 as "wrong bitmap length". (2) One molecule twice gives the same bitmap, otherwise
                 the cache and the deep check contradict each other.
        Args: none. Returns: none. Data source: `D.compute()`.
        Config: `D.FINGERPRINT_BITS`, `D.FINGERPRINT_RADIUS`. Side effects: none.
        Raises: AssertionError on a wrong length, a non-reproducible result, or a collision.
    """
    first = D.compute("*CC(*)C")
    second = D.compute("*CC(*)C")
    assert first is not None and second is not None
    assert len(first["fingerprint"]) == D.FINGERPRINT_BITS // 8
    assert first["fingerprint"] == second["fingerprint"]
    bits = D.fingerprint_array(first["fingerprint"])
    assert bits.shape == (D.FINGERPRINT_BITS,)
    assert bits.sum() > 0, "指纹一位都没置上"
    other = D.compute("*CC(c1ccccc1)*")
    assert other is not None
    assert D.fingerprint_array(other["fingerprint"]).sum() > bits.sum(), "加了苯环，置位数却没变多"


def test_describe_reports_why_a_wpsmiles_is_missing() -> None:
    """`describe()` 必须给出可读原因，而不是一个空字段。
    `describe()` must give a readable reason rather than an empty field.

    中文：
        用途：失败清单的 `reason` 列直接从它来。空字段会让"为什么没有 wPSMILES"这个问题
              在半年后变成一个考古工作。
        输入：无。输出：无。数据来源：`N.describe()`。配置来源：无。副作用：无。
        异常：AssertionError —— 该有原因的地方是空的。

    English:
        Purpose: the failure manifest's `reason` column comes straight from here. An empty field
                 turns "why is there no wPSMILES?" into an archaeology project six months later.
        Args: none. Returns: none. Data source: `N.describe()`. Side effects: none.
        Raises: AssertionError when a reason that should exist is empty.
    """
    good = N.describe("*CC*")
    assert good["wpsmiles"] and good["wpsmiles_reason"] == ""
    bad = N.describe("*CCC(*)*")
    assert bad["wpsmiles"] is None and "3" in str(bad["wpsmiles_reason"])
    broken = N.describe("这不是 SMILES")
    assert broken["psmiles_canonical"] is None and broken["wpsmiles_reason"] == "parse_failed"


# ---- 缓存：算一次，而且不许串味 / Cache: compute once, never across feature spaces ----


def test_cache_serves_the_second_pass_from_disk(tmp_path) -> None:
    """第二次取同一批结构必须全部命中，且结果与第一次逐位相同。
    The second pass over the same structures must hit the cache and match bit for bit.

    中文：
        用途：A-6 的验收④是"同一结构重复计算结果一致（缓存命中）"。这里把"命中"变成可
              断言的事实（`stats["hits"]`），而不是看日志里的数字。
        输入：tmp_path —— 临时缓存文件（不污染 data/processed）。
        输出：无。数据来源：临时 sqlite + RDKit。配置来源：`D.provenance()`。
        副作用：写临时文件。异常：AssertionError —— 没命中、或两次结果不同。

    English:
        Purpose: criterion 4 is "the same structure computed again gives the same result (cache
                 hit)". This turns "hit" into an assertable fact (`stats["hits"]`) instead of a
                 number in a log line.
        Args: tmp_path - a temporary cache so data/processed is untouched. Returns: none.
        Data source: a temporary sqlite plus RDKit. Config: `D.provenance()`.
        Side effects: writes a temporary file. Raises: AssertionError on a miss or a difference.
    """
    values = ["*CC(*)C", "*CC(c1ccccc1)*", "*CC(*)C"]
    with D.DescriptorCache(tmp_path / "cache.sqlite3") as cache:
        first, stats_first = cache.records(values)
        second, stats_second = cache.records(values)
        stored = cache.stored_keys()
    assert stats_first["misses"] == 2 and stats_first["hits"] == 0
    assert stats_second["misses"] == 0 and stats_second["hits"] == 2
    assert stored == {"*CC(*)C", "*CC(c1ccccc1)*"}, "缓存应只存去重后的结构"
    for left, right in zip(first, second):
        assert left["fingerprint"] == right["fingerprint"]
        assert left["descriptors"] == right["descriptors"]
    # 重复的输入位置拿到同一条记录（同一个结构不会算两次）。
    # The duplicated input position gets the same record: one structure is never computed twice.
    assert first[0]["fingerprint"] == first[2]["fingerprint"]


def test_cache_stores_256_byte_bitmaps(tmp_path) -> None:
    """缓存里的指纹必须是 256 字节的位图——这是"存的是位图不是稀疏编码"的落盘证据。
    The cache must hold 256-byte bitmaps: proof that what lands on disk is a bitmap.

    中文：
        用途：`fp.ToBinary()` 那个坑绕过去了没有，看磁盘最直接：18 字节 = 又踩回去了。
        输入：tmp_path。输出：无。数据来源：临时 sqlite。
        配置来源：`D.FINGERPRINT_BITS`。副作用：写临时文件。
        异常：AssertionError —— 长度不是 256。

    English:
        Purpose: whether the `fp.ToBinary()` trap was really avoided is most visible on disk:
                 18 bytes means it was stepped in again.
        Args: tmp_path. Returns: none. Data source: a temporary sqlite.
        Side effects: writes a temporary file. Raises: AssertionError on a length other than 256.
    """
    path = tmp_path / "cache.sqlite3"
    with D.DescriptorCache(path) as cache:
        cache.records(["*CC(*)*", "*C*"])
    connection = sqlite3.connect(path)
    lengths = [row[0] for row in connection.execute("SELECT length(fingerprint) FROM structures")]
    connection.close()
    assert lengths and set(lengths) == {D.FINGERPRINT_BITS // 8}


def test_cache_refuses_a_foreign_provenance(tmp_path) -> None:
    """RDKit 版本一变，老缓存必须拒绝打开，而不是"尽力兼容"。
    When the RDKit version changes the old cache must refuse to open, not "do its best".

    中文：
        用途：描述符数值随版本漂移（`Chi`、`BCUT` 尤其）。把两套特征混进同一个实验，
              模型无法解释，而且**不会报错**——所以宁可在打开时炸掉。
        输入：tmp_path。输出：无。数据来源：临时 sqlite。配置来源：`D.provenance()`。
        副作用：写临时文件。异常：AssertionError —— 居然打开了。

    English:
        Purpose: descriptor values drift across versions (`Chi` and `BCUT` especially). Mixing two
                 feature spaces in one experiment is unexplainable **and silent**, so failing on
                 open is the lesser evil.
        Args: tmp_path. Returns: none. Data source: a temporary sqlite.
        Side effects: writes a temporary file. Raises: AssertionError if it opens.
    """
    path = tmp_path / "cache.sqlite3"
    with D.DescriptorCache(path) as cache:
        cache.records(["*C*"])
    connection = sqlite3.connect(path)
    payload = json.loads(
        connection.execute("SELECT value FROM metadata WHERE key='provenance'").fetchone()[0]
    )
    payload["rdkit_version"] = "1999.09.1"
    connection.execute(
        "UPDATE metadata SET value=? WHERE key='provenance'", (json.dumps(payload),)
    )
    connection.commit()
    connection.close()
    with pytest.raises(ValueError, match="provenance"):
        D.DescriptorCache(path)


# ---- ② 正向：真实产物自洽 / Positive: the real artefacts hold together ----


@needs_artifacts
def test_structure_table_is_one_row_per_canonical() -> None:
    """结构表一个规范式一行，`structure_id` 就是行号，`n_rows` 合计对得上清单。
    One row per canonical form, `structure_id` is the row number, and `n_rows` sums correctly.

    中文：
        用途：去重没去干净会表现为"同一个结构两行、两张 id"，B 按 id join 就会只命中一半，
              而且不报错。`n_rows` 合计与清单的对照则防"结构表被截断"。
        输入：无。输出：无。数据来源：`psmiles.parquet` + `structures_manifest.csv`。
        配置来源：无。副作用：只读。异常：AssertionError —— 有重复键或合计不符。

    English:
        Purpose: incomplete deduplication shows up as one structure with two ids, so B's join hits
                 half the rows without raising. The `n_rows` sum guards against a truncated table.
        Args: none. Returns: none. Data source: the parquet plus its manifest.
        Side effects: read-only. Raises: AssertionError on a duplicate key or a wrong sum.
    """
    table = S.load_structure_table()
    assert table.index.is_unique, "结构表里有重复的规范式"
    assert list(table["structure_id"]) == list(range(len(table))), "structure_id 不是连续行号"
    record = S._read_csv_rows(S.STRUCTURES_MANIFEST)[0]
    expected = int(record["n_structures"])
    assert len(table) == expected
    assert int(table["n_rows"].sum()) + int(record["n_rows_without_structure"]) == int(
        record["n_rows_total"]
    )


@needs_artifacts
def test_wpsmiles_exists_exactly_when_two_asterisks() -> None:
    """结构表里"有 wPSMILES"与"星号数为 2"必须完全等价。
    "Has a wPSMILES" must be exactly equivalent to "has two asterisks".

    中文：
        用途：这是 `to_wpsmiles()` 的语义在**全量数据**上的断言。少一个就会让 B 拿到 None
              却以为有图；多一个就是硬拼出来的假串。
        输入：无。输出：无。数据来源：`psmiles.parquet`。
        配置来源：`N.HOMOPOLYMER_STAR_COUNT`。副作用：只读。
        异常：AssertionError —— 两者不等价。

    English:
        Purpose: the semantics of `to_wpsmiles()` asserted over the **whole dataset**. One short
                 and B gets None while believing there is a graph; one over is a fake string.
        Args: none. Returns: none. Data source: the parquet.
        Side effects: read-only. Raises: AssertionError when the two disagree.
    """
    table = S.load_structure_table()
    has = table["wpsmiles"].notna()
    two_stars = table["n_stars"] == N.HOMOPOLYMER_STAR_COUNT
    assert (has == two_stars).all(), (
        f"{int((has != two_stars).sum())} 行的 wPSMILES 与星号数不一致"
    )
    two_star_texts = table.loc[has, "wpsmiles"]
    assert two_star_texts.str.endswith(N.WPSMILES_SUFFIX).all(), "wPSMILES 后缀不统一"


@needs_artifacts
def test_feature_files_carry_no_labels() -> None:
    """每份特征文件都不许出现标签或 `row_id` 列——装配是 B 的事。
    No feature file may carry labels or a `row_id`: assembly is B's job.

    中文：
        用途：B-1 的验收标准①是"行序由 split 文件决定"。如果 A 先把标签和行号塞进特征文件，
              这条标准就变成"再检查一遍 A 有没有搞错"，而不是独立验收。
        输入：无。输出：无。数据来源：`data/processed/descriptors/*.parquet`。
        配置来源：`S.FORBIDDEN_FEATURE_COLUMNS`。副作用：只读。
        异常：AssertionError —— 出现了禁列。

    English:
        Purpose: B-1's criterion 1 is "row order is decided by the split file". If A prefilled
                 labels and row ids, that criterion degrades into "check whether A got it right"
                 rather than an independent check.
        Args: none. Returns: none. Data source: the descriptor parquets.
        Config: `S.FORBIDDEN_FEATURE_COLUMNS`. Side effects: read-only.
        Raises: AssertionError on a forbidden column.
    """
    for task_id in S.available_tasks():
        path = S.DESCRIPTORS_DIR / f"{task_id}.parquet"
        if not path.is_file():
            continue
        frame = pd.read_parquet(path)
        leaked = [name for name in S.FORBIDDEN_FEATURE_COLUMNS if name in frame.columns]
        assert not leaked, f"{task_id} 的特征文件里出现了 {leaked}"


@needs_artifacts
def test_feature_files_have_the_frozen_shape() -> None:
    """每份特征文件的列序、行数、指纹位数都要与清单和特征字典一致。
    Column order, row count and bit count must agree with the manifest and the feature dictionary.

    中文：
        用途：列序变了模型就不能复用（同名不同序的矩阵乘出来是垃圾且不报错），所以这里
              比的是**完整列序**，不是集合。
        输入：无。输出：无。数据来源：特征文件 + `features_manifest.csv` + `descriptor_columns.json`。
        配置来源：无。副作用：只读。异常：AssertionError —— 任一不符。

    English:
        Purpose: a changed column order invalidates every trained model (same names, different
                 order, garbage out, no error), so the comparison is the **full order**, not a set.
        Args: none. Returns: none. Data source: the feature files, their manifest and the
        dictionary. Side effects: read-only. Raises: AssertionError on any mismatch.
    """
    frozen = json.loads(S.COLUMNS_PATH.read_text(encoding="utf-8"))
    assert frozen["n_descriptors"] == 205
    assert frozen["fingerprint"]["n_bits"] == D.FINGERPRINT_BITS
    records = {row["task_id"]: row for row in S._read_csv_rows(S.FEATURES_MANIFEST)}
    for task_id in S.available_tasks():
        path = S.DESCRIPTORS_DIR / f"{task_id}.parquet"
        if not path.is_file():
            continue
        frame = pd.read_parquet(path)
        assert list(frame.columns) == EXPECTED_FEATURE_COLUMNS, f"{task_id} 的列序与特征字典不符"
        assert len(frame) == int(records[task_id]["n_unique_structures"])


@needs_artifacts
def test_failure_manifest_lists_real_failures_only() -> None:
    """失败清单只许记真的失败：星号数真不对，或者真的解析不了。
    The failure manifest may only list real failures: a wrong star count or a real parse failure.

    中文：
        用途：这张清单是"有记录的丢失"的凭据。如果它记了一条其实没问题的结构，B/C 会
              白排一个缺口；如果它漏记，`_verify_coverage()` 会报"缺失且没记"——两头都堵住。
        输入：无。输出：无。数据来源：`structure_failures.csv`。
        配置来源：无。副作用：只读。异常：AssertionError —— 有假的失败记录。

    English:
        Purpose: this manifest is the evidence of "recorded losses". A false entry makes B/C plan
                 around a gap that is not there; a missing one trips `_verify_coverage()`. Both
                 ends are closed.
        Args: none. Returns: none. Data source: the failure CSV.
        Side effects: read-only. Raises: AssertionError on a bogus failure row.
    """
    rows = S._read_csv_rows(S.FAILURES_PATH)
    assert rows, "失败清单是空的——实测有 116 个结构不是两个星号，这不正常"
    for row in rows:
        reason = row["reason"]
        if reason.startswith("star_count_"):
            claimed = int(reason.split("_")[2])
            assert N.count_stars(row["psmiles_canonical"]) == claimed
            assert claimed != N.HOMOPOLYMER_STAR_COUNT
        elif reason == "parse_failed":
            assert N.canonical_psmiles(row["smiles_raw"]) is None
        elif reason == "canonical_mismatch":
            # 这条只在 A-3 与 A-6 的规范化结果不一致时出现；真出现了是要停下来查的那种。
            # Appears only when A-3 and A-6 disagree; if it ever appears, stop and investigate.
            pytest.fail(f"出现了 canonical_mismatch：{row['psmiles_canonical'][:40]}…")
        else:
            pytest.fail(f"不认识的原因：{reason}")


@needs_artifacts
def test_verify_outputs_passes_on_the_real_artefacts() -> None:
    """浅层校验在真实产物上必须返回空列表。 / The shallow check must return an empty list."""
    problems = S.verify_outputs()
    assert problems == [], "验收未通过：\n" + "\n".join(problems)


@needs_artifacts
@needs_slow
def test_deep_verify_passes_on_the_real_artefacts() -> None:
    """`--verify --deep`：从原始结构重算 11,494 个结构，结果必须逐位相同（约 2 分钟）。
    Deep: recompute all 11,494 structures from their source and require bit-equality (~2 min).
    """
    problems = S.verify_outputs(deep=True)
    assert problems == [], "深度校验未通过：\n" + "\n".join(problems)


# ---- ④ 能抓住真错 / Real-failure detection ----


@needs_artifacts
def test_a_stale_structure_id_forces_a_rewrite(tmp_path) -> None:
    """结构表一变（任务集变了），旧特征文件里的 `structure_id` 就跟着失配，必须判为陈旧。
    When the structure table changes, an old file's `structure_id` no longer matches and must be
    judged stale.

    中文：
        用途：这是实测踩到的真坑——先跑了 1 个任务、再跑 10 个任务，前者的 id 全部平移。
              如果脚本把"文件存在"当成"文件正确"，join 出来的特征是错位的，而且没有人会报错。
        输入：tmp_path —— 一份伪造的特征文件。
        输出：无。数据来源：伪造的小表（不碰真产物）。
        配置来源：无。副作用：写临时文件。
        异常：AssertionError —— 陈旧文件被当成最新。

    English:
        Purpose: this is a trap actually hit in practice - running one task and then ten shifts
                 every earlier id. If the script treats "the file exists" as "the file is right",
                 the join is misaligned and nothing raises.
        Args: tmp_path - a synthetic feature file. Returns: none.
        Data source: a synthetic table, so the real artefacts are untouched.
        Side effects: writes a temporary file. Raises: AssertionError when a stale file passes.
    """
    table = pd.DataFrame(
        {
            "psmiles_canonical": ["*C*", "*CC*"],
            "structure_id": [0, 1],
            "wpsmiles": ["a", "b"],
        }
    ).set_index("psmiles_canonical", drop=False)
    path = tmp_path / "task.parquet"
    pd.DataFrame({"psmiles_canonical": ["*C*", "*CC*"], "structure_id": [0, 1]}).to_parquet(
        path, index=False
    )
    stale, reason = S._descriptor_file_is_stale(path, table, ["*C*", "*CC*"])
    assert not stale and reason == ""
    shifted = pd.DataFrame({"psmiles_canonical": ["*C*", "*CC*"], "structure_id": [5, 6]})
    shifted.to_parquet(path, index=False)
    stale, reason = S._descriptor_file_is_stale(path, table, ["*C*", "*CC*"])
    assert stale and "structure_id" in reason


@needs_artifacts
def test_hash_check_catches_a_tampered_file(tmp_path) -> None:
    """文件被改而清单没改：sha256 当场发现。
    A file changed while the manifest did not: sha256 catches it immediately.

    中文：
        用途：这是浅层校验**能**做到的那一半，先把这半边钉住；下一支测试证明它的另一半是盲的。
        输入：tmp_path。输出：无。数据来源：真实特征文件的副本。
        配置来源：无。副作用：写临时文件。
        异常：AssertionError —— 哈希不符却没报。

    English:
        Purpose: the half the shallow pass **can** do. Pinning it first makes the next test's claim
                 ("the other half is blind") meaningful.
        Args: tmp_path. Returns: none. Data source: a copy of a real feature file.
        Side effects: writes a temporary file. Raises: AssertionError when a hash mismatch is silent.
    """
    task_id, frame, columns = _small_real_frame()
    path = tmp_path / "copy.parquet"
    frame.to_parquet(path, index=False, compression="snappy")
    record = {
        "descriptor_sha256": "0" * 64,
        "n_unique_structures": str(len(frame)),
        "n_nan_cells": str(S._count_nan(frame, columns)),
    }
    problems: list[str] = []
    S._check_feature_file(task_id, path, record, EXPECTED_FEATURE_COLUMNS, problems)
    assert any("sha256" in problem for problem in problems), problems


@needs_artifacts
def test_shallow_check_is_blind_to_a_consistent_edit_but_deep_is_not(tmp_path) -> None:
    """改一个描述符值、并把清单里的哈希一起改掉：浅层看不见，deep 必须看见。
    Change one descriptor value and update the manifest hash to match: the shallow pass is blind,
    the deep pass must see it.

    中文：
        用途：**记录一条已知局限，而不是假装它不存在**。`--verify` 比对的是"文件 vs 清单"，
              两者一起改就一致了；只有从原始结构重算（`--verify --deep`）才绕得开清单。
              所以这条测试断言两件相反的事：浅层没问题（证明局限真实存在）、
              deep 有问题（证明兜底有效）。哪天浅层也能抓到了，说明实现变了，该改这段说明。
        输入：tmp_path。输出：无。数据来源：真实特征文件的副本（改一个 MolWt）。
        配置来源：无。副作用：写临时文件。
        异常：AssertionError —— 浅层抓到了（局限已不存在）或 deep 没抓到（兜底失效）。

    English:
        Purpose: **record a known limitation instead of pretending it is not there**. `--verify`
                 compares file against manifest, so editing both makes them agree; only a
                 recomputation from the source structure (`--verify --deep`) bypasses the manifest.
                 The test therefore asserts two opposite things: the shallow pass finds nothing
                 (the limitation is real) and the deep pass finds something (the fallback works).
                 If the shallow pass ever catches it, the implementation changed and this paragraph
                 must change with it.
        Args: tmp_path. Returns: none. Data source: a copy of a real feature file with one `MolWt`
        edited. Side effects: writes a temporary file.
        Raises: AssertionError if the shallow pass catches it or the deep pass misses it.
    """
    task_id, frame, columns = _small_real_frame()
    position = frame.columns.get_loc("MolWt")
    frame = frame.copy()
    frame.iloc[0, position] = float(frame.iloc[0, position]) + 1.0
    path = tmp_path / "tampered.parquet"
    frame.to_parquet(path, index=False, compression="snappy")
    record = {
        # 关键：清单里的哈希**跟着改成新文件的**，模拟"文件和清单一起被改"。
        # The point: the manifest hash is updated to the tampered file, simulating a coordinated
        # edit of file and manifest.
        "descriptor_sha256": S._sha256_of(path),
        "n_unique_structures": str(len(frame)),
        "n_nan_cells": str(S._count_nan(frame, columns)),
    }
    shallow: list[str] = []
    S._check_feature_file(task_id, path, record, EXPECTED_FEATURE_COLUMNS, shallow)
    assert shallow == [], f"浅层居然抓到了——局限已不存在，请改掉这条测试的说明：{shallow}"

    deep: list[str] = []
    S._deep_check_frame(task_id, frame, columns, deep)
    assert deep and "不一致" in deep[0], deep


def _small_real_frame() -> tuple[str, pd.DataFrame, list[str]]:
    """取一个**最小**的真实特征文件用于篡改测试，省掉 8055 行的读盘时间。
    Take the smallest real feature file for the tamper tests, avoiding an 8,055-row read.

    中文：
        用途：篡改测试只需要一份真实结构的特征；用最小的任务（`pb_t3_iwlt`，228 行）
              可以把两支测试的总耗时压到几秒。缺产物时跳过——这些测试不能靠伪造数据，
              因为它们要验证的正是"真实产物能不能被抓到"。
        输入：无。
        输出：(task_id, frame, descriptor_columns)。
        数据来源：`data/processed/descriptors/pb_t3_iwlt.parquet`。
        配置来源：无。副作用：只读。异常：pytest.skip —— 缺这个文件。

    English:
        Purpose: the tamper tests only need one real structure set; the smallest task (228 rows)
                 keeps both tests to a few seconds. Skips when absent - these tests cannot use
                 synthetic data because what they check is whether the real artefacts can be
                 caught.
        Args: none. Returns: (task_id, frame, descriptor_columns).
        Data source: data/processed/descriptors/pb_t3_iwlt.parquet.
        Side effects: read-only. Raises: pytest.skip when the file is missing.
    """
    task_id = "pb_t3_iwlt"
    path = S.DESCRIPTORS_DIR / f"{task_id}.parquet"
    if not path.is_file():
        pytest.skip(f"缺 {task_id} 的特征文件，无法做篡改测试")
    frame = pd.read_parquet(path)
    return task_id, frame, D.descriptor_columns()


@needs_artifacts
def test_shallow_check_catches_a_leaked_label_column(tmp_path) -> None:
    """有人把标签塞进特征文件：浅层校验必须点名这一列。
    If someone slips labels into a feature file, the shallow pass must name the column.

    中文：
        用途：B 的验收独立性靠"特征文件里没有标签"这条边界。这里不测 B，测的是**我们会不会
              在自己这边先把边界破了**。
        输入：tmp_path。输出：无。数据来源：真实特征文件的副本 + 一列 `label`。
        配置来源：`S.FORBIDDEN_FEATURE_COLUMNS`。副作用：写临时文件。
        异常：AssertionError —— 加了标签列却没报。

    English:
        Purpose: B's independence rests on "no labels in the feature file". This does not test B;
                 it tests whether we break that boundary on our own side first.
        Args: tmp_path. Returns: none. Data source: a copy plus one `label` column.
        Side effects: writes a temporary file. Raises: AssertionError when the leak is silent.
    """
    task_id, frame, columns = _small_real_frame()
    frame = frame.copy()
    frame["label"] = np.arange(len(frame), dtype=float)
    path = tmp_path / "leaky.parquet"
    frame.to_parquet(path, index=False, compression="snappy")
    record = {
        "descriptor_sha256": S._sha256_of(path),
        "n_unique_structures": str(len(frame)),
        "n_nan_cells": str(S._count_nan(frame, columns)),
    }
    problems: list[str] = []
    S._check_feature_file(task_id, path, record, EXPECTED_FEATURE_COLUMNS, problems)
    assert any("标签" in problem for problem in problems), problems
