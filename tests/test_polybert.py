"""B-2 的验收测试：分片、合并、行序、索引、产物校验。

中文：
    用途：把 B-2 的三条验收标准落成可执行的断言，并盯住两个"不报错的错"——
          ① 裸 `*` 喂进 polyBERT 会变成 `<unk>`（嵌入照样算得出来，只是废的）；
          ② 行序按 DataFrame 走会静默换样品。
    依赖：numpy / pyarrow / pytest；**不需要 torch 与 transformers**——模型相关的那条测试
          在没有 torch 的环境里自动跳过，所以本文件在 DSC6020_Project_Core 里也能全绿。
    数据：需要 `data/processed/structures/psmiles.parquet`（A-6 的产物）；缺了自动 skip。

English:
    Purpose: turn B-2's three acceptance criteria into assertions, and guard the two failures that
    raise nothing - a bare `*` becoming `<unk>` (the embedding is still produced, just useless) and a
    row order that follows the DataFrame (silently swaps samples).
    Depends on numpy/pyarrow/pytest only; model-dependent checks skip when torch is absent, so this
    file is green in DSC6020_Project_Core too.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from chainworth.chem import polybert as P


@pytest.fixture(scope="module")
def structures():
    """真的把 A-6 的结构表读一遍（缺表就跳过）。/ Read A-6's table for real; skip if absent."""
    if not P.STRUCTURES_PATH.exists():
        pytest.skip("缺 data/processed/structures/psmiles.parquet——先跑 scripts/02_features.py")
    return P.load_structures()


@pytest.fixture()
def fake_artifacts(tmp_path, structures):
    """造一份 8 行的"合法"试跑产物，供校验器测试。/ Build an 8-row legitimate partial artifact."""
    limit = 8
    index = P.load_structures(limit=limit)
    import dataclasses

    index = dataclasses.replace(
        index,
        n_tokens=np.full(limit, 12, dtype=np.int32),
        n_unk=np.zeros(limit, dtype=np.int32),
    )
    vectors = np.zeros((limit, P.EMBED_DIM), dtype=np.float32)
    vectors[:, 0] = np.arange(limit)
    P.save_embeddings(tmp_path / P.EMBEDDINGS_NAME, vectors)
    P.write_index(tmp_path / P.INDEX_NAME, index)
    manifest = P.build_manifest(
        index,
        vectors,
        device="cpu",
        wall_clock_s=0.1,
        n_truncated=0,
        shard_size=4,
        is_partial=True,
    )
    P.write_manifest(tmp_path / P.MANIFEST_NAME, manifest)
    return tmp_path, index


def test_bare_star_becomes_bracket_form():
    """裸 `*` 必须变成 `[*]`——这是词表里唯一认的连接点写法。
    A bare `*` must become `[*]`: the only attachment-point spelling in the vocabulary.
    """
    assert P.to_polybert_input("*CC(*)C") == "[*]CC([*])C"
    assert P.to_polybert_input("*C*") == "[*]C[*]"
    assert "*" not in P.to_polybert_input("*CO*").replace("[*]", "")
    assert P.to_polybert_input(" *CO* ") == "[*]CO[*]"       # 两端空白要顺手去掉


def test_bracket_form_is_rejected_not_doubled():
    """已经带括号的输入要报错，不能悄悄产出 `[[*]]`。
    Already-bracketed input must raise instead of silently becoming `[[*]]`.
    """
    with pytest.raises(P.PolyBertError):
        P.to_polybert_input("[*]CC[*]")
    with pytest.raises(P.PolyBertError):
        P.to_polybert_input("   ")
    with pytest.raises(P.PolyBertError):
        P.to_polybert_input("CCO")


def test_order_is_canonical_sort_not_source_row(structures):
    """行序必须是规范式排序，且 A-6 的行号只作参照。
    Row order must be the canonical sort; A-6's row number is reference only.
    """
    assert list(structures.canonical) == sorted(structures.canonical)
    assert len(set(structures.canonical)) == len(structures)
    # 排序后行序与 A-6 的行号**不是**同一个序列（否则这条测试白写）
    assert not np.array_equal(structures.structure_id, np.arange(len(structures)))


def test_every_structure_has_an_input_string(structures):
    """每条结构都能转成输入串，且不含裸 `*`。/ Every structure converts and keeps no bare `*`."""
    assert len(structures.inputs) == len(structures.canonical)
    assert all("[*]" in text for text in structures.inputs)
    assert all("*" not in text.replace("[*]", "") for text in structures.inputs)


def test_plan_shards_covers_every_row_exactly_once():
    """分片必须无缝、不重叠、不越界，最后一片可以短。
    Shards must be seamless, non-overlapping and in range; the last one may be short.
    """
    assert P.plan_shards(0, 10) == []
    assert P.plan_shards(5, 10) == [(0, 5)]
    assert P.plan_shards(10, 10) == [(0, 10)]
    ranges = P.plan_shards(11_494, 1024)
    assert ranges[0] == (0, 1024)
    assert ranges[-1] == (11_264, 11_494)
    assert all(stop - start == 1024 for start, stop in ranges[:-1])
    covered = [row for start, stop in ranges for row in range(start, stop)]
    assert covered == list(range(11_494))
    with pytest.raises(P.PolyBertError):
        P.plan_shards(10, 0)


def test_shard_roundtrip_and_shape_check(tmp_path):
    """分片能原样读回；向量行数与 ID 数不符时必须炸。
    A shard reads back unchanged; mismatched row counts must raise.
    """
    path = P.shard_path(tmp_path, 0)
    P.save_shard(
        path,
        {
            "row_start": 0,
            "row_stop": 3,
            "structure_id": np.array([10, 11, 12]),
            "n_tokens": np.array([5, 6, 7], dtype=np.int32),
            "n_unk": np.zeros(3, dtype=np.int32),
            "vectors": np.ones((3, P.EMBED_DIM), dtype=np.float32),
        },
    )
    payload = P.read_shard(path)
    assert payload["vectors"].shape == (3, P.EMBED_DIM)
    assert payload["structure_id"].tolist() == [10, 11, 12]
    assert not path.with_suffix(".partial.npz").exists()

    bad = P.shard_path(tmp_path, 1)
    np.savez(
        bad,
        row_start=np.int64(3),
        row_stop=np.int64(5),
        structure_id=np.array([13, 14]),
        n_tokens=np.array([1, 1], dtype=np.int32),
        n_unk=np.zeros(2, dtype=np.int32),
        vectors=np.ones((3, P.EMBED_DIM), dtype=np.float32),
    )
    with pytest.raises(P.PolyBertError):
        P.read_shard(bad)


def test_merge_trusts_row_numbers(tmp_path):
    """合并只信行号：缺片、重叠、总数不对都要报错，绝不给半份矩阵。
    Merging trusts row numbers: a gap, an overlap or a short total must raise.
    """
    def make(number: int, start: int, stop: int, fill: float) -> Path:
        path = P.shard_path(tmp_path, number)
        P.save_shard(
            path,
            {
                "row_start": start,
                "row_stop": stop,
                "structure_id": np.arange(start, stop),
                "n_tokens": np.full(stop - start, 4, dtype=np.int32),
                "n_unk": np.zeros(stop - start, dtype=np.int32),
                "vectors": np.full((stop - start, P.EMBED_DIM), fill, dtype=np.float32),
            },
        )
        return path

    first, second = make(0, 0, 4, 1.0), make(1, 4, 8, 2.0)
    merged = P.merge_shards([second, first], 8)          # 故意倒序传入，行序必须仍按 row_start
    assert merged["vectors"].shape == (8, P.EMBED_DIM)
    assert merged["vectors"][0, 0] == 1.0 and merged["vectors"][-1, 0] == 2.0
    assert merged["structure_id"].tolist() == list(range(8))

    with pytest.raises(P.PolyBertError):                 # 少一片
        P.merge_shards([first], 8)
    with pytest.raises(P.PolyBertError):                 # 多出一行
        P.merge_shards([first, second], 9)


def test_shard_matches_detects_reordered_table(tmp_path):
    """续跑判据要比 ID，不只是"文件在"：换行序后旧片必须被判为不可复用。
    The resume predicate compares ids, not mere existence.
    """
    path = P.shard_path(tmp_path, 0)
    P.save_shard(
        path,
        {
            "row_start": 0,
            "row_stop": 2,
            "structure_id": np.array([7, 8]),
            "n_tokens": np.array([3, 3], dtype=np.int32),
            "n_unk": np.zeros(2, dtype=np.int32),
            "vectors": np.zeros((2, P.EMBED_DIM), dtype=np.float32),
        },
    )
    assert P.shard_matches(path, np.array([7, 8]))
    assert not P.shard_matches(path, np.array([8, 7]))
    assert not P.shard_matches(path, np.array([7, 8, 9]))
    assert not P.shard_matches(P.shard_path(tmp_path, 5), np.array([7, 8]))


def test_index_roundtrip(tmp_path, structures):
    """索引能读回，row 列是 0..n-1，规范式与输入串一一对应。
    The index reads back, `row` is 0..n-1, canonical and input line up one to one.
    """
    limit = 5
    import dataclasses

    index = dataclasses.replace(structures, n_tokens=np.arange(limit, dtype=np.int32), n_unk=np.zeros(limit, dtype=np.int32))
    index = dataclasses.replace(
        index,
        canonical=structures.canonical[:limit],
        inputs=structures.inputs[:limit],
        structure_id=structures.structure_id[:limit],
    )
    path = tmp_path / P.INDEX_NAME
    P.write_index(path, index)
    back = P.read_index(path)
    assert back["row"] == list(range(limit))
    assert back["psmiles_canonical"] == list(index.canonical)
    assert back["polybert_input"] == list(index.inputs)
    assert back["n_unk"] == [0] * limit


def test_verify_passes_on_a_legitimate_partial(fake_artifacts):
    """合法的试跑产物必须过校验（否则校验器只会误报，没人会信它）。
    A legitimate partial artifact must pass, otherwise the checker only cries wolf.
    """
    directory, _ = fake_artifacts
    assert P.verify_outputs(directory) == []


def test_verify_catches_reordered_rows(fake_artifacts):
    """把索引的顺序打乱一位，校验器必须抓到——这正是"静默换样品"的样子。
    Swapping two index rows must be caught - that is what a silent sample swap looks like.
    """
    directory, index = fake_artifacts
    swapped = list(index.canonical)
    swapped[2], swapped[3] = swapped[3], swapped[2]
    import dataclasses

    broken = dataclasses.replace(
        index,
        canonical=tuple(swapped),
        inputs=tuple(P.to_polybert_input(text) for text in swapped),
    )
    P.write_index(directory / P.INDEX_NAME, broken)
    problems = P.verify_outputs(directory)
    assert any("行序" in item for item in problems)


def test_verify_catches_nan_and_unknown_tokens(fake_artifacts):
    """NaN 与未知 token 都要被抓到；未知 token 是"输入形态错了"的报警器。
    NaN and unknown tokens must both be caught; unknown tokens are the input-form alarm.
    """
    directory, index = fake_artifacts
    vectors = np.load(directory / P.EMBEDDINGS_NAME)
    vectors[1, 3] = np.nan
    P.save_embeddings(directory / P.EMBEDDINGS_NAME, vectors)
    problems = P.verify_outputs(directory)
    assert any("NaN" in item for item in problems)

    vectors[1, 3] = 0.0
    P.save_embeddings(directory / P.EMBEDDINGS_NAME, vectors)
    index_n_unk = index.n_unk.copy()
    index_n_unk[4] = 2
    import dataclasses

    P.write_index(directory / P.INDEX_NAME, dataclasses.replace(index, n_unk=index_n_unk))
    problems = P.verify_outputs(directory)
    assert any("未知 token" in item for item in problems)


def test_verify_catches_wrong_dimension(fake_artifacts):
    """维度不对（比如误存成 1024 维）必须报出来，而不是让下游静默少列。
    A wrong dimension must be reported instead of letting downstream read fewer columns.
    """
    directory, index = fake_artifacts
    P.save_embeddings(directory / P.EMBEDDINGS_NAME, np.zeros((len(index), P.EMBED_DIM - 1), dtype=np.float32))
    problems = P.verify_outputs(directory)
    assert any("形状" in item for item in problems)


def test_manifest_csv_appends_one_row(tmp_path, fake_artifacts):
    """清单 CSV 只写一次表头，之后逐次追加；列名固定。/ Header once, then append; fixed columns."""
    directory, _ = fake_artifacts
    payload = json.loads((directory / P.MANIFEST_NAME).read_text(encoding="utf-8"))
    path = tmp_path / "polybert_manifest.csv"
    P.append_manifest_csv(path, payload)
    P.append_manifest_csv(path, payload)
    lines = path.read_text(encoding="utf-8").strip().split("\n")
    assert len(lines) == 3
    assert lines[0] == ",".join(P.MANIFEST_COLUMNS)
    assert str(payload["n_structures"]) in lines[1]


def test_model_is_reachable_or_skipped():
    """模型的重量级检查放在最后：没有 torch 就跳过，有就跑一次前向确认形状。
    The heavyweight model check comes last: skip without torch, otherwise run one forward pass.
    """
    pytest.importorskip("torch", reason="核心环境没有 torch，模型检查跳过")
    pytest.importorskip("transformers", reason="GNN 环境有 torch 但没有 transformers，同样跳过")
    tokenizer, model, device = P.load_model()
    vectors, tokens, unknown = P.embed_shard(tokenizer, model, device, ["[*]CC[*]", "[*]COC[*]"], batch_size=2)
    assert vectors.shape == (2, P.EMBED_DIM)
    assert int(unknown.sum()) == 0
    assert (tokens > 0).all() and (tokens <= P.MAX_TOKENS).all()
