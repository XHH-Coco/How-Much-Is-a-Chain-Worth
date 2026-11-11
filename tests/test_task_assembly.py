"""B-1 的断言测试：行序由划分决定、特征逐位来自 A-6、构图可重复且与 chemprop 一致。
Assertions for B-1: row order comes from the split, features are bit-identical to A-6's, and graphs
are reproducible and agree with chemprop.

中文：
    用途：把分工书 B-1 的五条验收标准变成 `pytest`。分四类：
          ① **钳子（口径）**：`row_id` = 行位置、`X` 的第 k 行就是 `row_id=k`、索引与
             `splits/**/test_index` 同一坐标系；改动一行必须被抓到。
          ② **正向**：`verify_features()` 在真实产物上返回空列表；抽 100 个结构与 A-6 逐位相等。
          ③ **边界**：`--limit` 裁划分后仍保持同一坐标系且太小时报错；PolyBench26/OpenPoly
             的任务装配必须**明确拒绝**并说清缺什么，不能静默给出空矩阵；解析失败的结构
             必须计数并列 ID。
          ④ **能抓住真错**：把 `.npz` 里一个数值改掉、并把清单哈希一起改掉，`verify_features()`
             必须报出来。第 ④ 类才是有价值的那一半——只断言"没报错"的测试拦不住"文件被改过"。
    依赖：`data/processed/features/*.npz` 与 `data/manifests/task_features_manifest.csv`
          （B-1 产物，**不入库**，由 `scripts/03b_assemble.py` 重建）。缺产物时相关用例跳过，
          纯函数与图相关用例照跑（图只需要 rdkit）。
    配置：无环境变量；chemprop 相关用例在没装 chemprop 的环境里自动跳过。
    对外：pytest 用例。被谁用：CI、任何改了 `chem/` 或 `io/loaders.py` 的人、D 的入口审计。

English:
    Purpose: turn B-1's five acceptance criteria into pytest cases, in four groups: (1) the tongs -
    `row_id` is the row position, row k of `X` is `row_id=k`, the index matches
    `splits/**/test_index`, and a single modified value must be caught; (2) the positive path -
    `verify_features()` returns an empty list on the real artefacts and a 100-structure sample is
    bit-identical to A-6's table; (3) the boundaries - `--limit` trims the split while keeping the
    coordinate system and refuses when it is too small, PolyBench26/OpenPoly tasks must be refused
    with a reason rather than silently yielding an empty matrix, and unparsable structures must be
    counted and listed; (4) real-failure detection - change one number inside the `.npz` and update
    the manifest hash to match, and `verify_features()` must still report it. Group 4 is the half
    that matters: a test that only asserts "no exception" cannot catch "the file was edited".
    Depends on: data/processed/features/*.npz and data/manifests/task_features_manifest.csv (B-1
    artefacts, **not committed**, rebuilt by scripts/03b_assemble.py). Artefact-dependent cases skip
    when absent; pure and graph cases always run (graphs need only rdkit).
    Config: none; chemprop-dependent cases skip automatically when chemprop is not installed.
    Public API: pytest cases. Callers: CI, anyone touching chem/ or io/loaders.py, and D's entry
    audit.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from chainworth.chem import features as F
from chainworth.chem import graph as G
from chainworth.io import loaders as L

HAS_PACKS = L.FEATURES_DIR.is_dir() and any(L.FEATURES_DIR.glob("*.npz")) and F.MANIFEST_PATH.is_file()
HAS_STRUCTURES = L.STRUCTURES_PATH.is_file()

needs_packs = pytest.mark.skipif(
    not HAS_PACKS,
    reason="缺 B-1 产物（data/processed 不入库）：先跑 `python scripts/03b_assemble.py`",
)
needs_structures = pytest.mark.skipif(
    not HAS_STRUCTURES,
    reason="缺 A-6 结构表：先跑 `python scripts/02_features.py --source all`",
)


# --------------------------------------------------------------------------- #
# ① 行序：row_id 是行位置，且与划分文件同一坐标系 / row order comes from the split
# --------------------------------------------------------------------------- #
def test_row_id_is_the_harmonised_row_position() -> None:
    """行表的第一行 `row_id=0`，末行 `row_id=n-1`，连续无跳号——"位置"就是"第几行"。
    The row table starts at row_id 0, ends at n-1 and never skips: position equals row number.

    中文：
        用途：这条看着是废话，但它是整个装配的地基：只要有人把 `row_id` 改成 `source_row`，
              索引就会从 0..n-1 变成"原始 CSV 的行号"，而两者在多数任务里**都不一样**
              （A-3 删过行）。那时测试集会悄悄换一批样品，且没有任何异常。
    English:
        Purpose: trivial-looking, but it is the foundation: if somebody swaps `row_id` for
        `source_row`, the index stops being 0..n-1 and becomes the original CSV's line numbers,
        which differ on most tasks (A-3 dropped rows). The test set would quietly change and
        nothing would raise.
    """
    task_id = "pb_t1_tg"
    rows = L.load_rows(task_id)
    assert list(rows.index[:3]) == [0, 1, 2]
    assert int(rows.index[-1]) == len(rows) - 1
    assert rows.index.is_unique and rows.index.is_monotonic_increasing


@needs_packs
def test_split_index_lands_on_the_same_row_id() -> None:
    """验收①：`X` 的第 k 行就是 `row_id=k`，取 `test_index` 取到的正是划分文件说的那些行。
    Criterion 1: row k of X is row_id=k, so indexing by `test_index` selects exactly those rows.

    中文：
        用途：分工书原文是"断言 `X.index` 与 `splits/…/test.json` 的索引逐位相同"。装配产物的
              索引就是 `row_id`，所以这里核两件事：`row_id` 连续，且 `row_id[test_index]` 逐位
              等于 `test_index`。第二件才是真正的断言——第一件只保证没跳号。
    English:
        Purpose: division B literally says "assert `X.index` is identical, position by position, to
        the indices in splits/.../test.json". The assembly's index *is* `row_id`, so two things are
        checked: `row_id` is contiguous, and `row_id[test_index]` equals `test_index` position by
        position. The second is the real assertion; the first only says nothing was skipped.
    """
    for task_id in _sample_tasks(3):
        pack = F.load_pack(task_id)
        split = L.load_split(task_id)
        assert np.array_equal(pack["row_id"], np.arange(len(pack["row_id"])))
        for fold in split.folds:
            assert pack["row_id"][fold.test_index].tolist() == fold.test_index.tolist()
            assert pack["row_id"][fold.train_index].tolist() == fold.train_index.tolist()


@needs_packs
def test_row_count_equals_harmonised_rows() -> None:
    """验收①的另一半：装配行数 = 调和表可用标签行数，**一行都没被删**。
    The other half of criterion 1: the assembly's row count equals the harmonised label count and
    no row was dropped.

    中文：
        用途：最危险的"改进"是在装配时顺手 `dropna()`——它会让行数少几十行、指标更好看，
              而位置索引全部前移。这条把它钉住。
    English:
        Purpose: the most dangerous "improvement" is a casual `dropna()` during assembly: a few
        dozen rows vanish, metrics look better, and every positional index shifts.
    """
    for task_id in _sample_tasks(3):
        pack = F.load_pack(task_id)
        rows = L.load_rows(task_id)
        assert len(pack["row_id"]) == len(rows)
        assert pack["psmiles"].tolist() == rows["psmiles_canonical"].fillna("").astype(str).tolist()


@needs_packs
def test_labels_come_from_the_configured_target_column() -> None:
    """标签按配置的 `target_column` 取，不是写死 `label`。
    Labels come from the configured `target_column`, not a hard-coded `label`.

    中文：
        用途：PolyBench26 的列名是 `Tg_K` / `density`，写死 `label` 在那两类任务上会取到
              不存在的列（或更糟：取到同名的别的列）。这条把"标签必须与调和表逐位相等"变成断言，
              换列时不会静默通过。
    English:
        Purpose: PolyBench26 calls the column `Tg_K` or `density`; a hard-coded `label` would either
        fail or, worse, pick a same-named but different column there. Asserting bit-equality with
        the harmonised table makes a swapped column fail loudly.
    """
    for task_id in _sample_tasks(2):
        pack = F.load_pack(task_id)
        rows = L.load_rows(task_id)
        assert np.array_equal(pack["y"], rows["label"].to_numpy(dtype="float64"), equal_nan=True)


# --------------------------------------------------------------------------- #
# ② 特征逐位来自 A-6 / features are bit-identical to A-6's output
# --------------------------------------------------------------------------- #
@needs_packs
def test_features_are_bit_identical_to_the_structure_level_table() -> None:
    """验收②：随机抽 100 个结构，205 个描述符与 2048 位指纹与 A-6 的表**逐位相等**。
    Criterion 2: a random 100 structures are **bit-identical** to A-6's table on all 2,253 columns.

    中文：
        用途：用 `np.array_equal` 而不是 `allclose`。`allclose` 会放过 float32 的截断——而"截断"正是
              下游最难查的那类偏差：模型照跑、指标差一点、谁也说不清差在哪。
    English:
        Purpose: `np.array_equal`, not `allclose`. `allclose` forgives float32 truncation, which is
        the hardest kind of drift to trace: the model runs, the metrics move slightly, nobody can
        say why.
    """
    task_id = _sample_tasks(1)[0]
    pack = F.load_pack(task_id)
    table = L.load_descriptor_table(task_id)
    keys = [key for key in pack["psmiles"] if key]
    rng = np.random.default_rng(7)
    picks = rng.choice(len(keys), size=100, replace=False)
    positions = np.flatnonzero(pack["psmiles"] != "")
    chosen = positions[np.sort(picks)]
    source = table.reindex([pack["psmiles"][position] for position in chosen])
    columns = list(pack["desc_columns"])
    bits = list(pack["fp_columns"])
    assert np.array_equal(source[columns].to_numpy(dtype="float64"), pack["desc"][chosen], equal_nan=True)
    assert np.array_equal(source[bits].to_numpy(dtype="uint8"), pack["fp"][chosen])


@needs_packs
def test_repeated_structure_gets_one_feature_vector() -> None:
    """同一结构出现在多行时必须给同一组特征（行序不同、特征相同）。
    One structure appearing on several rows must get one feature vector.

    中文：
        用途：任务内通常零重复（A-3 已按结构去重），但 OpenPoly 有 57 组重复标签（D-26）。
              装配若按行取数而不是按键 join，这些行会各拿一份"恰好不同"的特征，而不会报错。
    English:
        Purpose: tasks usually have no internal duplicates (A-3 deduplicates by structure), but
        OpenPoly has 57 duplicated label groups (D-26). If assembly read rows instead of joining on
        the key, those rows would each get a slightly different vector - with no error.
    """
    for task_id in _sample_tasks(3):
        pack = F.load_pack(task_id)
        counts = pd.Series(pack["psmiles"]).value_counts()
        for key, count in counts.items():
            if not key or count < 2:
                continue
            index = np.flatnonzero(pack["psmiles"] == key)
            assert np.array_equal(pack["desc"][index], pack["desc"][index][:1], equal_nan=True)


# --------------------------------------------------------------------------- #
# ③ 边界：拒绝、裁切、失败计数 / boundaries: refusals, trimming, failure counts
# --------------------------------------------------------------------------- #
def test_polybench26_and_openpoly_assembly_is_refused_with_a_reason() -> None:
    """PolyBench26 / OpenPoly 的任务装配必须**明确拒绝**，并说清缺的是调和表。
    PolyBench26 / OpenPoly assembly must be **refused** with the missing artefact named.

    中文：
        用途：这两类任务的标签在 `data/raw/` 的原始 CSV 里、结构也不在 A-6 的表里。静默返回空矩阵
              会让 B-3 在半夜拿到一堆 NaN 才开始查；所以它们必须当场失败，且消息指向 A-3 没有为
              它们产出调和表这件事。
    English:
        Purpose: those tasks' labels live in raw CSVs and their structures are absent from A-6's
        table. Silently returning an empty matrix means B-3 discovers a wall of NaN at midnight;
        failing here, naming the missing harmonised table, is the honest outcome.
    """
    for task_id in ("pb26_polymetrix_tg", "op_tg"):
        with pytest.raises(L.AssemblyError) as error:
            L.load_rows(task_id)
        assert "调和表" in str(error.value)


def test_limit_split_keeps_the_coordinate_system_and_refuses_when_too_small() -> None:
    """`--limit` 裁划分：索引仍是行位置、仍然 `< n_rows`；太小则报错而不是给出空折。
    `--limit` trims the split: indices stay row positions and stay below n_rows; too small raises
    instead of yielding an empty fold.

    中文：
        用途：限量跑是"冒烟"，不是"另一套划分"。裁完若不保持同一坐标系，冒烟跑会给出与正式跑
              不同口径的产物，而两者看起来一模一样。太小时报错，是因为空训练集跑出来的东西
              没有意义，不如当场停。
    English:
        Purpose: a limited run is a smoke test, not a second split. If the coordinate system did not
        survive trimming, the smoke artefact would use a different convention while looking
        identical. Too-small raises because training on nothing produces nothing useful.
    """
    spec = L.load_split("pb_t1_tg")
    trimmed = L.limit_split(spec, 500)
    assert trimmed.n_total == 500
    assert L.check_split_matches_rows(trimmed, 500) == []
    for fold in trimmed.folds:
        assert fold.train_index.max() < 500 and fold.test_index.max() < 500
    # 只留 1 行时训练侧为空（pb_t1_tg 的第 0 行是测试行），必须报错而不是给出空折。
    # With a single row the training side is empty (row 0 is a test row in pb_t1_tg), which must
    # raise rather than hand back an empty fold.
    with pytest.raises(L.AssemblyError):
        L.limit_split(spec, 1)


def test_check_split_matches_rows_catches_overlap_and_out_of_range() -> None:
    """划分自查必须报出"同一行两边都有"和"索引越界"这两种错。
    The split self-check must report both "one row on both sides" and "index out of range".

    中文：
        用途：这两种错都不会让任何库报错，只会让测试集混进训练数据。构造两个坏划分，
              确认它们被抓到——否则"自查通过"这句话没有意义。
    English:
        Purpose: neither mistake raises anywhere; they just leak test rows into training. Two broken
        splits are constructed so that "the check passed" means something.
    """
    good = L.FoldSpec(
        name="x",
        train_index=np.asarray([0, 1, 2]),
        val_index=np.zeros((0,), np.int64),
        test_index=np.asarray([3, 4]),
    )
    spec = L.SplitSpec("t", "polybench", "p", "ref", "sha", 5, "", (good,))
    assert L.check_split_matches_rows(spec, 5) == []

    overlapping = L.FoldSpec(
        name="x",
        train_index=np.asarray([0, 1, 2, 3]),
        val_index=np.zeros((0,), np.int64),
        test_index=np.asarray([3, 4]),
    )
    problems = L.check_split_matches_rows(
        L.SplitSpec("t", "polybench", "p", "ref", "sha", 5, "", (overlapping,)), 5
    )
    assert any("索引重复" in problem for problem in problems)

    out_of_range = L.FoldSpec(
        name="x",
        train_index=np.asarray([0, 1, 2]),
        val_index=np.zeros((0,), np.int64),
        test_index=np.asarray([3, 9]),
    )
    problems = L.check_split_matches_rows(
        L.SplitSpec("t", "polybench", "p", "ref", "sha", 5, "", (out_of_range,)), 5
    )
    assert any("越界" in problem for problem in problems)


# --------------------------------------------------------------------------- #
# ③ 图：确定性、维度、失败计数，以及与 chemprop 对拍
# --------------------------------------------------------------------------- #
def test_two_spellings_of_one_chain_give_one_graph() -> None:
    """验收③：同一根链的两种写法，`edge_index` 与原子顺序**完全相同**。
    Criterion 3: two spellings of one chain yield an identical `edge_index` and atom order.

    中文：
        用途：这是"原子排序稳定，不许依赖字典序"的直接检验。`*CC(*)C` 与 `*CC(C)*` 在 RDKit 里
              是两套原子编号——不先规范化，两行其实是同一结构的样品就会拿到两张不同的图，
              而下游只看到"两个样本"。
    English:
        Purpose: the direct test of "stable atom order, never dictionary order". `*CC(*)C` and
        `*CC(C)*` have two different RDKit atom numberings; without canonicalising first, two rows
        that are the same structure get two different graphs while downstream sees "two samples".
    """
    first = G.build_graph("*CC(*)C")
    second = G.build_graph("*CC(C)*")
    assert first is not None and second is not None
    assert np.array_equal(first.edge_index, second.edge_index)
    assert np.array_equal(first.atom, second.atom)
    assert first.psmiles == second.psmiles


def test_graph_shape_and_the_asterisk_pad_bit() -> None:
    """图维度是 72/14；`*`（原子序数 0）落在原子序数块的 unknown pad 位上，不落到第 0 位。
    Dimensions are 72/14, and `*` (atomic number 0) lands on the unknown pad of the atomic-number
    block rather than bit 0.

    中文：
        用途：pad 位不是凑数。`*` 若被塞进第 0 位（氢的位置），模型会把每个重复单元的两个连接点
              都当成氢——特征矩阵照常是 (n,72)，形状完全正确，只有化学含义错了。
    English:
        Purpose: the pad is not filler. If `*` were pushed into bit 0 (hydrogen's slot), the model
        would read both connection points as hydrogens: the matrix would still be (n, 72) and the
        shape perfectly correct - only the chemistry wrong.
    """
    graph = G.build_graph("*C*")
    assert graph is not None
    assert graph.atom.shape == (3, G.ATOM_FEATURE_DIM)
    assert graph.bond.shape == (2 * 2, G.BOND_FEATURE_DIM)
    assert graph.n_stars == 2
    pad = len(tuple(range(1, 37)) + (53,))
    star_rows = [index for index in range(graph.atom.shape[0]) if graph.atom[index, pad] == 1.0]
    assert len(star_rows) == 2
    # 第 0 位是**氢原子**的原子序数位。隐式氢不是图上的原子，所以 `*C*` 的三个原子都不占这位——
    # 这条断言的作用是证明"没有把隐式氢偷偷塞进图里"（塞进去会让图的节点数随分子变，处处错位）。
    # Bit 0 is the atomic-number slot for a **hydrogen atom**. Implicit Hs are not graph atoms, so
    # none of `*C*`'s three atoms takes that bit - the assertion proves implicit Hs were not smuggled
    # into the graph, which would change the node count per molecule and misalign everything.
    assert graph.atom[:, 0].sum() == 0
    carbon_bit = 5          # 原子序数 6 是取值表里的第 6 个（下标 5）/ atomic number 6 is index 5
    assert graph.atom[:, carbon_bit].sum() == 1


def test_graph_attributes_carry_the_repeat_unit_mass() -> None:
    """图级属性 5 维，质量项 = 重复单元质量/100（`*` 不计质量）。
    The graph has five attributes and its mass term is repeat-unit mass / 100 (`*` weighs nothing).

    中文：
        用途：`repeat_unit_mass` 是"一个重复单元多重"，A5 那类链信息要用它。用 `*C*` 验：
              只有一个碳，所以质量必须是 CH2 的 14.027 而不是 14.027 + 0（`*` 贡献 0）。
    English:
        Purpose: `repeat_unit_mass` is "how heavy one repeat unit is", the quantity A5-style chain
        information needs. `*C*` has a single carbon, so the mass must be CH2's 14.027 - the `*`
        atoms contribute nothing.
    """
    graph = G.build_graph("*C*")
    assert graph is not None
    assert graph.graph_attr.shape == (G.GRAPH_ATTR_DIM,)
    assert graph.repeat_unit_mass == pytest.approx(14.027, abs=1e-3)
    assert graph.graph_attr[3] == pytest.approx(graph.repeat_unit_mass / 100, abs=1e-6)
    assert graph.graph_attr[4] == 0.0        # 没传 wPSMILES / no wPSMILES given
    marked = G.build_graph("*C*", "wPSMILES+[#0:1]-[#0:2]")
    assert marked is not None and marked.graph_attr[4] == 1.0


def test_unparsable_structures_are_counted_not_dropped() -> None:
    """验收⑤：解析失败的结构**计数并列出 ID**，不静默丢弃。
    Criterion 5: unparsable structures are **counted and listed**, never silently dropped.

    中文：
        用途：`build_graphs()` 的失败清单是这条标准的落地。空串与乱码各给一个原因，
              调用方据此记账——"图少了 3 个"与"图少了 3 个，因为这两个串解析不了"是两件事。
    English:
        Purpose: `build_graphs()`'s failure list is where this criterion lands. An empty string and
        garbage get distinct reasons; "three graphs are missing" and "three graphs are missing
        because these two strings do not parse" are different statements.
    """
    graphs, failures = G.build_graphs(["*C*", "", "not-a-smiles", "*CC(*)"])
    assert "*C*" in graphs
    reasons = dict((text, reason) for text, reason in failures)
    assert reasons.get("") == "empty_string"
    assert "not-a-smiles" in reasons


def test_graph_matches_chemprop_bit_for_bit() -> None:
    """验收③的官方对拍：与 chemprop 的 `SimpleMoleculeMolGraphFeaturizer` 逐位相同。
    The official parity check for criterion 3: bit-identical to chemprop's featurizer.

    中文：
        用途：分工书要求"用官方脚本对拍一次，不要凭记忆写"。这条是那一次，且每次跑都会重做。
              没装 chemprop 的环境（core）自动跳过——图臂本来就不在那里跑。
    English:
        Purpose: division B requires "check once against the official script, do not write it from
        memory". This is that check, repeated on every run. It skips in an environment without
        chemprop (the core env), where the graph arm does not run anyway.
    """
    pytest.importorskip("chemprop", reason="core 环境没有 chemprop：对拍在 GNN 环境里做")
    samples = ["*C*", "*CC(*)C", "*CC(CC)*", "*CC(C)(C)*", "*[Si](C)(C)O*", "*CC(*)c1ccc([*])cc1"]
    if HAS_STRUCTURES:
        table = L.load_structure_table_index()
        keys = table.index.tolist()
        rng = np.random.default_rng(2026)
        samples += [keys[index] for index in rng.choice(len(keys), size=50, replace=False)]
    assert G.verify_against_chemprop(samples) == []


# --------------------------------------------------------------------------- #
# ④ 能抓住真错：改一个数、并把清单哈希一起改掉 / real-failure detection
# --------------------------------------------------------------------------- #
@needs_packs
def test_verify_passes_on_the_real_artefacts() -> None:
    """验收②③④⑤的汇总：真实产物上 `verify_features()` 返回空列表。
    The summary of criteria 2-5: `verify_features()` returns an empty list on the real artefacts.

    中文：
        用途：这条只证明"现在的产物自洽"。它**不能**证明有人改了产物还会被发现——那是下一条。
    English:
        Purpose: proves the current artefacts are self-consistent. It does **not** prove a modified
        artefact would be caught - that is the next test.
    """
    assert F.verify_features() == []


@needs_packs
def test_verify_catches_one_modified_value_even_with_a_rewritten_hash(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """把一个数值改掉、并把清单哈希一起改掉：`verify_features()` 必须报"整矩阵不逐位相等"。
    Change one value, rewrite the manifest hash to match: `verify_features()` must still report that
    the matrix is not bit-identical.

    中文：
        用途：这条模拟"有人为了让指标好看，手工调了一个特征值，并顺手更新了清单"。浅层哈希查不出
              来（哈希本来就是对的），唯一的兜底是"整矩阵与 A-6 的原始值逐位比"——而 A-6 的表
              本身还有自己的 `--verify --deep`，两条独立的链一起错才会漏。
              做法：把产物复制到临时目录、改一个 NaN 之外的值、重算哈希，再把清单指过去。
    English:
        Purpose: simulates "somebody nudged one feature value to make the metrics nicer and updated
        the manifest". A hash check cannot see it (the hash is genuinely correct); the only net is
        the whole-matrix comparison against A-6's raw values - and A-6's table has its own
        `--verify --deep`, so both chains would have to fail together for this to slip through.
        Method: copy the artefact into a temp directory, change one non-NaN value, recompute the
        hash, and point the manifest at it.
    """
    task_id = _sample_tasks(1)[0]
    source = L.feature_pack_path(task_id)
    features_dir = tmp_path / "data" / "processed" / "features"
    features_dir.mkdir(parents=True)
    target = features_dir / source.name
    pack = F.load_pack(task_id)
    desc = pack["desc"].copy()
    row, column = np.argwhere(~np.isnan(desc))[0]
    desc[row, column] += 1.0
    np.savez_compressed(
        target,
        row_id=pack["row_id"],
        y=pack["y"],
        psmiles=pack["psmiles"],
        desc=desc,
        fp=pack["fp"],
        desc_columns=pack["desc_columns"],
        fp_columns=pack["fp_columns"],
        meta=np.asarray(json.dumps(pack["meta"], ensure_ascii=False)),
    )
    manifest_dir = tmp_path / "data" / "manifests"
    manifest_dir.mkdir(parents=True)
    rows = list(csv.DictReader(F.MANIFEST_PATH.open("r", encoding="utf-8", newline="")))
    for item in rows:
        if item["task_id"] == task_id:
            item["pack_sha256"] = F.sha256_of(target)
            item["pack_file"] = f"data/processed/features/{source.name}"
    with (manifest_dir / "task_features_manifest.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=F.MANIFEST_COLUMNS, lineterminator="\n")
        writer.writeheader()
        for item in rows:
            writer.writerow(item)

    monkeypatch.setattr(F, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(F, "FEATURES_DIR", features_dir)
    monkeypatch.setattr(F, "MANIFEST_PATH", manifest_dir / "task_features_manifest.csv")
    monkeypatch.setattr(F, "UNRESOLVED_PATH", manifest_dir / "task_feature_unresolved.csv")
    monkeypatch.setattr(L, "FEATURES_DIR", features_dir)

    problems = F.verify_features([task_id])
    assert any("逐位相等" in problem for problem in problems), problems


# --------------------------------------------------------------------------- #
# 标准化：只能在 train 上 fit / standardisation happens on train only
# --------------------------------------------------------------------------- #
def test_standardizer_uses_train_statistics_only() -> None:
    """标准化器的均值来自训练集：变换后训练集均值≈0，而"测试集"均值明显不为 0。
    The scaler's means come from train: after transforming, train's mean is ~0 while the "test"
    block's mean is visibly not.

    中文：
        用途：验收④的落地检查。构造一个分布明显不同的"测试集"——只要有人用全量统计量，两块的均值
              都会是 0，这条就会失败。
    English:
        Purpose: the concrete check for criterion 4. A deliberately shifted "test" block is built;
        if anyone fitted on the full data, both blocks would have mean 0 and this fails.
    """
    rng = np.random.default_rng(11)
    train = rng.normal(size=(200, 2253))
    test = rng.normal(loc=5.0, size=(50, 2253))
    params = F.standardize_fit(train)
    transformed_train = F.apply_standardizer(params, train)
    transformed_test = F.apply_standardizer(params, test)
    assert np.allclose(transformed_train[:, 0], (train[:, 0] - train[:, 0].mean()) / train[:, 0].std())
    assert abs(transformed_test[:, 0].mean()) > 1.0
    assert int(params["n_train"]) == 200


def test_standardizer_leaves_the_fingerprint_block_alone() -> None:
    """指纹块不缩放：0/1 位被标准化之后就不再是"有没有这个子结构"。
    The fingerprint block is untouched: standardised bits stop meaning "this substructure is
    present or not".

    中文：
        用途：205 个描述符之后的 2048 列是二进制位。缩放它们会得到"−0.03 表示有"这种数，
              模型照训，但那 2048 列再也不可解释。
    English:
        Purpose: the 2048 columns after the 205 descriptors are binary bits. Scaling them yields
        values like "-0.03 means present": the model still trains, but those 2048 columns stop being
        interpretable.
    """
    rng = np.random.default_rng(13)
    train = rng.integers(0, 2, size=(100, 2253)).astype(np.float64)
    params = F.standardize_fit(train)
    transformed = F.apply_standardizer(params, train)
    n_descriptors = len(F.descriptor_columns())
    assert np.array_equal(transformed[:, n_descriptors:], train[:, n_descriptors:].astype(np.float32))


# --------------------------------------------------------------------------- #
# 数据目录里不该出现的东西 / what must not show up in the data directory
# --------------------------------------------------------------------------- #
@needs_packs
def test_packs_carry_no_label_column_inside_the_matrix() -> None:
    """特征矩阵里没有标签列：标签单独存成 `y`，`X` 只有 205 + 2048 列。
    No label column inside the matrix: the label lives in `y`, and `X` has exactly 205 + 2048
    columns.

    中文：
        用途：把标签塞进 `X` 是最省事、也最容易被下游当成特征的泄漏方式，而且不报错。
              这里核列名与列数两件事，任何"顺手加一列"都会当场变红。
    English:
        Purpose: putting the label into `X` is the laziest and most invisible leak, and nothing
        raises. Both the names and the width are checked, so any "just one more column" goes red.
    """
    pack = F.load_pack(_sample_tasks(1)[0])
    names = set(pack["desc_columns"]) | set(pack["fp_columns"])
    forbidden = {"label", "label_raw", "y", "target", "target_value", "row_id"}
    assert not (names & forbidden)
    assert pack["desc"].shape[1] == 205 and pack["fp"].shape[1] == 2048
    assert len(pack["row_id"]) == pack["desc"].shape[0] == pack["y"].shape[0]


def _sample_tasks(count: int) -> list[str]:
    """从盘上有产物的任务里取前 `count` 个（按任务名排序，结果稳定）。
    Take the first `count` tasks with artefacts on disk, in sorted order so results are stable.

    中文：用途：让"抽查几个任务"这类用例有一个确定的选择，而不是依赖目录遍历顺序。
          输入：count。输出：任务 ID 列表。数据来源：`data/processed/features/*.npz`。
          配置来源：无。副作用：无。异常：无。
    English: Purpose: give "spot-check a few tasks" cases a deterministic selection instead of
             directory iteration order. Args: count. Returns: task ids.
    """
    on_disk = sorted(path.stem for path in L.FEATURES_DIR.glob("*.npz"))
    return on_disk[:count]
