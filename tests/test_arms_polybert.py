"""B-4 的断言测试：嵌入按规范式对齐、ridge 是闭式解、alpha 只在训练内部选、记录合规。
Assertions for B-4: embeddings join on canonical strings, ridge is closed-form, alpha is chosen
inside the training data only, and the records validate.

中文：
    用途：把分工_B §2 B-4 的三条验收标准变成 `pytest`，分三类：
          ① **纯函数（core 环境也跑）**：
             - 配置哈希 = 配置文件的字节哈希（改一个空格就换一条臂）；
             - 网格 48 条，`arm` 全是 `a4_polybert`，与 A3 **逐条同名同折同种子**（否则主表两列
               不能相减）；
             - `run_id` 符合 schema 的 pattern，且第二段是臂名；
             - 嵌入内容哈希与 B-2 的 manifest **同一算法**（`sha256_of_array`，不是文件哈希）；
             - 嵌入式对齐：抽一个任务，每行的嵌入必须等于"它的规范式那一行"——**按行号取会错**，
               这条用例就是把"错法"钉死；
             - ridge 在合成数据上还原一个已知的线性函数；
             - 同输入两次求解逐位相同（闭式解没有随机性，这条是主表 `std=0` 的依据）；
             - 标准化只用训练行（改验证行不改变 mean/scale）；
             - `record_problems()` 收 A4 记录、拒 `arm` 与 `run_id` 不匹配的记录。
          ② **微调记录的纯函数（core 环境，不需要 transformers）**：`build_finetune_record()`
             的产出必须与已交付的 48 条**逐键同构、键序相同**；cuda 必带 `gpu` 块且两个确定性
             开关为 true；预测条数不匹配当场报错。这一组补的是 B-4 §8 第 9 条——原来"微调记录
             结构"只由一条需要 transformers 的用例守着，而那个环境没有 pytest，等于从未进过
             任何一次 `pytest tests`。
          ③ **需要数据（core 环境，装配产物在）**：冒烟跑一条 (a)，断言预测行数 = 测试集行数、
             `row_id` 落在官方 `test_index` 内、`device` 与真实设备一致。
          ④ **需要 transformers（polybert 环境才跑）**：微调冒烟 1 轮，断言预测行数与记录齐全。
    依赖：`configs/arms/a4_polybert.yaml`（入库）、`data/processed/features/*.npz` 与
          `data/processed/polybert/*`（不入库）。缺产物时相关用例跳过。
    配置：无环境变量。跑 ①③ 用对应环境。
    对外：pytest 用例。被谁用：CI、任何改 `arms/polybert.py` 或 `a4_polybert.yaml` 的人。

English:
    Purpose: turn B-4's three criteria into pytest cases in three groups: (1) pure functions that also
    run in the core environment - the config hash equals the file's bytes, the grid holds 48 runs all
    named `a4_polybert` and matching A3 run for run, `run_id` follows the schema and carries the arm
    name, the embedding content hash uses B-2's recipe (not the file hash), the join really goes
    through canonical strings (row-order lookup is the wrong answer this case pins down), ridge
    recovers a known linear function, two identical solves are bit-identical (the basis for `std=0`
    in the main table), standardisation uses training rows only, and `record_problems()` accepts an
    A4 record while rejecting a mismatched arm/run_id; (2) the fine-tune record's pure builder - its
    output must be key-for-key identical to a delivered record, in the same key order, with a gpu
    block on cuda whose determinism switches are both true, and a raised error on a count mismatch
    (this closes B-4 §8 item 9: the shape used to be covered only by a transformers-dependent test
    in an environment without pytest); (3) cases needing data - one (a) smoke run with prediction
    count, `row_id` membership and `device` asserted; (4) cases needing transformers - a one-epoch
    fine-tune smoke.
    Depends on: configs/arms/a4_polybert.yaml (committed) plus the non-committed data artefacts; the
    affected cases skip when those are absent. Run group (4) in the polyBERT environment.
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

import numpy as np
import pytest

from chainworth.arms import graph as A
from chainworth.arms import polybert as P
from chainworth.io import loaders as L

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = REPO_ROOT / "configs" / "arms" / "a4_polybert.yaml"
A3_CONFIG_PATH = REPO_ROOT / "configs" / "arms" / "a3_graph.yaml"
EMBEDDINGS = REPO_ROOT / "data" / "processed" / "polybert" / "embeddings.npy"

HAS_PACKS = L.FEATURES_DIR.is_dir() and any(L.FEATURES_DIR.glob("*.npz"))
HAS_EMBEDDINGS = EMBEDDINGS.is_file()

needs_packs = pytest.mark.skipif(
    not HAS_PACKS, reason="缺 B-1 装配产物（data/processed 不入库）：先跑 `python scripts/03b_assemble.py`"
)
needs_embeddings = pytest.mark.skipif(
    not HAS_EMBEDDINGS,
    reason="缺 B-2 嵌入（data/processed 不入库）：先跑 `python scripts/03c_polybert.py`",
)


def _has_torch() -> bool:
    try:
        import torch  # noqa: F401
    except ImportError:
        return False
    return True


needs_torch = pytest.mark.skipif(not _has_torch(), reason="core 环境没有 torch：这条在 GNN/polybert 环境跑")


def _has_transformers() -> bool:
    try:
        import transformers  # noqa: F401
    except ImportError:
        return False
    return True


needs_transformers = pytest.mark.skipif(
    not _has_transformers(), reason="要 transformers：在 DSC6020_Project_Polybert 环境跑"
)


# ---- 1. 配置与网格 / Config and grid ----


def test_config_hash_is_the_file_bytes() -> None:
    """配置的哈希必须是磁盘上那份字节的哈希：改一个空格就换一条臂。
    The config hash must be the bytes on disk: changing a space changes the arm.
    """
    config = P.load_arm_config(CONFIG_PATH)
    expected = "sha256:" + hashlib.sha256(CONFIG_PATH.read_bytes()).hexdigest()
    assert P.config_hash(config) == expected


def test_grid_holds_48_runs_for_a4() -> None:
    """A4 的网格与 A3 一样是 48 条（6 x 3 + 2 x 5 x 3），只是臂名不同。
    A4's grid is 48 like A3's (6 x 3 + 2 x 5 x 3); only the arm name differs.
    """
    config = P.load_arm_config(CONFIG_PATH)
    keys = P.enumerate_runs(config)
    assert len(keys) == 48
    assert len({key.run_id() for key in keys}) == 48
    assert {key.arm for key in keys} == {"a4_polybert"}


def test_grid_mirrors_a3_task_fold_seed() -> None:
    """除了臂名，A4 的 48 条必须与 A3 **逐条**对上：同任务、同折、同种子。
    Apart from the arm name, A4's 48 runs must match A3's one for one.
    """
    a3 = A.enumerate_runs(A.load_arm_config(A3_CONFIG_PATH))
    a4 = P.enumerate_runs(P.load_arm_config(CONFIG_PATH))
    strip = lambda keys: sorted((key.task, key.fold, key.seed) for key in keys)  # noqa: E731
    assert strip(a3) == strip(a4)


def test_run_id_matches_the_schema_pattern_and_carries_the_arm() -> None:
    """`run_id` 的形状由 `results/raw_runs/_schema.json` 钉死，第二段必须是臂名。
    The run_id shape is pinned by the schema, and its second segment must be the arm name.
    """
    schema = json.loads((REPO_ROOT / "results" / "raw_runs" / "_schema.json").read_text(encoding="utf-8"))
    pattern = re.compile(schema["properties"]["run_id"]["pattern"])
    run_id = P.run_id_for("pb_t1_tg", "official_70_30", 0, arm="a4_polybert")
    assert pattern.search(run_id), run_id
    assert run_id == "pb_t1_tg__a4_polybert__official_70_30__seed0"
    assert "a4_polybert" in schema["properties"]["arm"]["enum"]


def test_every_known_arm_is_in_the_schema() -> None:
    """`KNOWN_ARMS` 是 schema 枚举的"绊线"，两者必须一致——A4 靠它而不是靠 `if arm ==`。
    KNOWN_ARMS is a tripwire for the schema enum; A4 relies on it instead of an `if arm == ...`.
    """
    schema = json.loads((REPO_ROOT / "results" / "raw_runs" / "_schema.json").read_text(encoding="utf-8"))
    assert tuple(schema["properties"]["arm"]["enum"]) == A.KNOWN_ARMS


# ---- 2. 嵌入表：对齐与哈希 / The embedding table: join and hash ----


@needs_embeddings
def test_embedding_hash_uses_the_manifest_recipe() -> None:
    """嵌入哈希必须与 B-2 的 manifest 同一算法（数组内容哈希），不是 .npy 的字节哈希。
    The embedding hash must use B-2's recipe (array content), not the .npy file hash.

    中文：这条不是吹毛求疵——两种哈希都"对"，但混用会得到一个永远对不上的数，于是"核验通过"
          变成永恒的红灯。实测 .npy 字节哈希 `203950a3…`、manifest 的 `2b79ccae…`。
    """
    manifest = json.loads(
        (REPO_ROOT / "data" / "processed" / "polybert" / "manifest.json").read_text(encoding="utf-8")
    )
    table = P.load_embedding_table()
    assert table.sha256 == manifest["embeddings_sha256"]
    assert table.vectors.shape == (int(manifest["n_structures"]), int(manifest["embed_dim"]))


@needs_embeddings
@needs_packs
def test_embeddings_join_on_canonical_string_not_row_number() -> None:
    """每行的嵌入必须是"它的规范式那一行"，不是"同一个行号那一行"。
    A row's embedding must be the one for its canonical string, not the one at the same index.

    中文：用途：把错法钉死。B-1 的 pack 按 `row_id` 排，B-2 的嵌入按规范式升序排，两者行序无关；
          如果谁改成"按行号拼"，本用例在第一条就是红——而按行号拼**不会报错**，只会让 R² 变成
          噪声（这种错最难查，所以必须有一条断言拦着）。
    English: pins down the wrong answer. A row-order join raises nothing and merely turns R^2 into
    noise, which is the hardest kind of bug to notice, so an assertion has to stand in the way.
    """
    table = P.load_embedding_table()
    data = P.build_task_data("pb_t3_iwlt", use_cache=False)
    from chainworth.chem.notation import canonicalise_many

    canonical, _ = canonicalise_many(data.psmiles)
    checked = 0
    for position in range(data.n_rows):
        row = table.row_of_or_none(canonical[position])
        if row is None:
            continue
        assert np.array_equal(
            np.asarray(data.emb[position], dtype=np.float32),
            np.asarray(table.vectors[row], dtype=np.float32),
        ), f"第 {position} 行的嵌入不是它规范式那一行"
        checked += 1
    assert checked == data.n_rows


@needs_embeddings
@needs_packs
def test_no_row_is_missing_an_embedding() -> None:
    """交付时 8 个任务的嵌入覆盖必须是 100%：缺一行就少一个测试点，D 的 bootstrap 会对不齐。
    Coverage must be 100% for all eight tasks: one missing row is one missing test point.
    """
    for task in ("pb_t3_iwlt", "pb_t1_tm"):
        data = P.build_task_data(task, use_cache=False)
        assert data.n_missing == 0, f"{task} 缺 {data.n_missing} 行嵌入"


# ---- 3. 线性头：闭式解、无随机性、不泄漏 / The head: closed form, no randomness, no leakage ----


def _synthetic(n: int = 400, d: int = 12, seed: int = 7) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """造一组"真值是特征的线性函数"的数据，供 ridge 还原。
    Build data whose truth is a linear function of the features, for ridge to recover.
    """
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, d))
    truth = np.linspace(-2.0, 2.0, d)
    y = X @ truth + 0.01 * rng.normal(size=n)
    return X, y, truth


def test_ridge_recovers_a_known_linear_function() -> None:
    """小 alpha 下 ridge 必须还原出那条直线（相对误差 < 2%）。
    With a small alpha, ridge must recover the line (relative error below 2%).
    """
    X, y, truth = _synthetic()
    X_train, y_train = X[:300], y[:300]
    X_val, y_val = X[300:], y[300:]
    head = P.fit_head(X_train, y_train, X_val, y_val, [0.001, 0.01, 0.1], device="cpu")
    recovered = head["weights"] / head["x_scale"] * head["y_scale"]
    assert np.allclose(recovered, truth, rtol=0.02, atol=0.02), (recovered, truth)


def test_ridge_is_bit_identical_across_calls() -> None:
    """同一输入解两次必须逐位相同：这是主表里 A4 三个种子完全一致的**唯一**依据。
    Two identical solves must be bit-identical - the only basis for A4's three identical seeds.
    """
    X, y, _ = _synthetic()
    first = P.fit_head(X[:300], y[:300], X[300:], y[300:], [1.0], device="cpu")
    second = P.fit_head(X[:300], y[:300], X[300:], y[300:], [1.0], device="cpu")
    assert np.array_equal(first["weights"], second["weights"])


def test_alpha_is_chosen_on_validation_only() -> None:
    """选 alpha 的判据是验证集 RMSE；候选里必须选到真正最小的那个。
    Alpha is chosen by validation RMSE; the smallest must be picked.
    """
    X, y, _ = _synthetic()
    head = P.fit_head(X[:300], y[:300], X[300:], y[300:], [1e-6, 1e-2, 1e6], device="cpu")
    best = min(head["alpha_scores"], key=lambda item: item[1])
    assert head["alpha"] == best[0]
    # 1e6 的正则会把权重压到接近 0，验证 RMSE 必然比 1e-6 差——否则说明选择根本没生效。
    # A 1e6 penalty crushes the weights, so its validation RMSE must be worse than 1e-6's.
    scores = dict(head["alpha_scores"])
    assert scores[1e6] > scores[1e-6]


def test_standardisation_ignores_validation_rows() -> None:
    """标准化只能用训练行：把验证行整体平移，`x_mean`/`x_scale` 必须一点都不变。
    Standardisation must use training rows only: shifting the validation block must not move them.
    """
    X, y, _ = _synthetic()
    X_train, y_train = X[:300], y[:300]
    X_val, y_val = X[300:], y[300:]
    first = P.fit_head(X_train, y_train, X_val, y_val, [1.0], device="cpu")
    second = P.fit_head(X_train, y_train, X_val + 1000.0, y_val, [1.0], device="cpu")
    assert np.array_equal(first["x_mean"], second["x_mean"])
    assert np.array_equal(first["x_scale"], second["x_scale"])


# ---- 4. 记录校验 / Record validation ----


def _good_record() -> dict:
    """一条合格的 A4 记录草稿（字段与 `train_one` 写出来的一致）。
    A valid A4 record draft, with the same fields `train_one` writes.
    """
    run_id = P.run_id_for("pb_t1_tg", "official_70_30", 0, arm="a4_polybert")
    return {
        "run_id": run_id,
        "status": "ok",
        "benchmark": "polybench",
        "task": "pb_t1_tg",
        "arm": "a4_polybert",
        "seed": 0,
        "split_version": "fcbe58cc7d46@splits/polybench/pb_t1_tg/official_70_30.json",
        "split_fold": "official_70_30",
        "config_hash": "sha256:" + "0" * 64,
        "n_train": 10,
        "n_test": 5,
        "metrics": {"r2": 0.5, "mae": 1.0, "rmse": 1.2},
        "wall_clock_s": 1.0,
        "device": "cpu",
        "env_fingerprint": "results/env_fingerprints/env_gnn.json",
        "predictions": f"results/aggregate/predictions/{run_id}.parquet",
    }


def test_a4_record_passes_validation() -> None:
    """A4 记录必须过 `record_problems()`——臂名换了，校验器不能因此放行任何东西。
    An A4 record must pass `record_problems()`: a different arm does not loosen the checker.
    """
    assert P.record_problems(_good_record()) == []


@pytest.mark.parametrize(
    "mutate, keyword",
    [
        (lambda row: row.update(arm="a3_graph"), "不一致"),
        (lambda row: row.update(arm="a4_polyberts"), "枚举"),
        (lambda row: row.update(run_id=row["run_id"].replace("a4_polybert", "a4")), "不一致"),
        (lambda row: row.update(predictions="results/aggregate/predictions/x.parquet"), "predictions"),
    ],
)
def test_bad_a4_records_are_caught(mutate, keyword) -> None:
    """改臂名、改 run_id、改预测路径都必须被抓——尤其是"臂名与 run_id 不同步"这一种。
    Changing the arm, the run_id or the predictions path must be caught.
    """
    record = _good_record()
    mutate(record)
    problems = P.record_problems(record)
    assert problems, f"这条变异没被抓：{record}"
    assert any(keyword in item for item in problems), problems


@pytest.mark.parametrize(
    "deterministic, ok",
    [(True, True), (False, False)],
)
def test_cuda_record_demands_both_determinism_switches(deterministic, ok) -> None:
    """`device=cuda` 的记录必须两个确定性开关都为 true（A4 也一样，不是"线性头可以例外"）。
    A cuda record needs both determinism switches true; the linear head is no exception.
    """
    record = _good_record()
    record.update(
        device="cuda",
        gpu={
            "name": "NVIDIA GeForce RTX 4070",
            "driver": "591.86",
            "torch_cuda": "12.1",
            "cudnn_deterministic": True,
            "deterministic_algorithms": deterministic,
        },
    )
    problems = P.record_problems(record)
    assert bool(not problems) is ok, problems


# ---- 5. 端到端（要数据） / End to end (needs data) ----


@needs_packs
@needs_embeddings
def test_train_one_lines_up_with_the_split(tmp_path) -> None:
    """跑一条 (a)，断言：预测条数 = 测试集条数、`row_id` 落在官方 `test_index` 内。
    Run one (a) and assert the prediction count and that every `row_id` comes from `test_index`.
    """
    config = P.load_arm_config(CONFIG_PATH)
    data = P.build_task_data("pb_t3_iwlt")
    outcome = P.train_one(data, config, fold="official_70_30", seed=0, device="cpu")
    fold = data.fold("official_70_30")
    expected_rows = {
        int(index) for index in np.asarray(fold.test_index) if data.has_embedding[int(index)]
    }
    assert outcome.y_pred.size == len(expected_rows)
    assert set(int(value) for value in outcome.row_id) == expected_rows
    assert outcome.record["arm"] == "a4_polybert"
    assert P.record_problems(outcome.record) == []
    assert outcome.record["alpha"] in [float(item) for item in config.section("head")["alpha_grid"]]


@needs_packs
@needs_embeddings
def test_same_seed_twice_gives_the_same_predictions() -> None:
    """同 seed 两次预测必须**逐位**相同（验收④）。线性头没有随机性，做不到就是有 bug。
    Two runs of the same seed must be bit-identical (criterion 4). A linear head has no randomness;
    anything else is a bug.
    """
    config = P.load_arm_config(CONFIG_PATH)
    data = P.build_task_data("pb_t3_iwlt")
    first = P.train_one(data, config, fold="official_70_30", seed=0, device="cpu")
    second = P.train_one(data, config, fold="official_70_30", seed=0, device="cpu")
    assert np.array_equal(first.y_pred, second.y_pred)


@needs_packs
@needs_embeddings
def test_a4_and_a3_share_the_same_split_version() -> None:
    """A4 与 A3 对同一任务必须报同一个 `split_version`：主表两列相减的前提。
    A4 and A3 must report the same split_version for a task: the precondition for subtracting columns.
    """
    a4 = P.build_task_data("pb_t3_iwlt").split
    a3 = A.load_split_for_tests("pb_t3_iwlt") if hasattr(A, "load_split_for_tests") else L.load_split("pb_t3_iwlt")
    version_a4, _ = P.fold_version(a4, "official_70_30")
    version_a3, _ = P.fold_version(a3, "official_70_30")
    assert version_a4 == version_a3


# ---- 6. 微调记录的结构（纯函数，不需要 transformers） / Fine-tune record shape ----
#
# 这一节是 B-4 §8 第 9 条那个缺口的补法。原来"微调记录结构与主表一致"只有一条用例守着，
# 而那条要 transformers；装了 transformers 的环境又没有 pytest，于是它从未进过任何一次
# `pytest tests`。现在记录由一个不 import torch 的纯函数拼（`build_finetune_record`），
# 结构就能在 core 环境里被断言：改字段、改顺序、漏 `hyperparameters` 的键，这里当场红。
# This section closes the gap recorded as B-4 §8 item 9: the shape used to be covered only by a
# transformers-dependent smoke test living in an environment without pytest, so it never ran in
# `pytest tests`. The record is now built by a pure function, so its shape is asserted here.

FINETUNE_RUNS_DIR = REPO_ROOT / "results" / "aggregate" / "a4_finetune" / "raw_runs"
HAS_FINETUNE_RUNS = FINETUNE_RUNS_DIR.is_dir() and any(FINETUNE_RUNS_DIR.glob("*.json"))

needs_finetune_runs = pytest.mark.skipif(
    not HAS_FINETUNE_RUNS,
    reason="缺已交付的微调记录（results/aggregate/a4_finetune 不入库）：先跑 `--finetune`",
)

# 参照物是一条**真实交付**的记录。断言的是"结构"，不是"数值"：n_train/n_val/n_test 与
# val_source 照抄那条 JSON，只为让记录自洽，不参与任何结论。
_DELIVERED_FINETUNE = FINETUNE_RUNS_DIR / "pb_t3_iwlt__a4_polybert__official_70_30__seed0.json"


def _fake_gpu_block(device: str, *, cudnn_deterministic: bool) -> dict:
    """替身：core 环境没有 torch，`A.gpu_block("cuda")` 会去 import 它。只伪造结构。
    A stand-in: core has no torch, so the real block cannot be produced here.
    """
    return {
        "model": "fake",
        "driver": "0",
        "torch_cuda": "0",
        "cudnn_deterministic": bool(cudnn_deterministic),
        "deterministic_algorithms": True,
        "capability": "8.9",
        "memory_total_mib": 0,
    }


def _build_sample_record(**overrides: object) -> dict:
    """用一组自洽的标量拼一条记录；`overrides` 用来构造反例。
    Build one record from self-consistent scalars; `overrides` builds the negative cases.
    """
    config = P.load_arm_config(CONFIG_PATH)
    rng = np.random.default_rng(0)
    y_true = rng.normal(size=69)
    y_pred = y_true + rng.normal(scale=0.1, size=69)
    kwargs: dict[str, object] = dict(
        config=config,
        benchmark="polybench",
        task_id="pb_t3_iwlt",
        fold="official_70_30",
        seed=0,
        split=L.load_split("pb_t3_iwlt"),
        n_train=143,
        n_test=69,
        n_val=16,
        y_true=y_true,
        y_pred=y_pred,
        wall_clock_s=10.0,
        device="cpu",
        hidden=600,
        epochs=30,
        epochs_run=14,
        best_epoch=9,
        best_val_rmse=40.15989251293401,
        val_source="inner_10%_of_train",
        fine_cfg=config.section("finetune"),
        batch_size=32,
        patience=5,
        freeze=False,
        max_tokens=512,
        deterministic=True,
    )
    kwargs.update(overrides)
    return P.build_finetune_record(**kwargs)  # type: ignore[arg-type]


@needs_finetune_runs
def test_finetune_record_matches_the_delivered_schema() -> None:
    """纯函数产出的记录必须与已交付的 48 条**逐键同构**，键序也一样。
    The pure builder must match a delivered record key for key, in the same order.
    """
    delivered = json.loads(_DELIVERED_FINETUNE.read_text(encoding="utf-8"))
    record = _build_sample_record()

    # cpu 不写 gpu 块（schema 只在 device=cuda 时要求它）；其余键一个不多、一个不少。
    assert set(record) == set(delivered) - {"gpu"}
    # 键序也是契约：重跑同一条 run 必须写出逐字节相同的 JSON，重排就算改格式。
    # `gpu` 是 cuda 才有的条件块，比较顺序时两边都摘掉它。
    assert [key for key in record if key != "gpu"] == [key for key in delivered if key != "gpu"]
    # C 的汇总脚本要按顶层读这四个超参（B-4 §8 第 7 条），漏掉就是它取到空值的原因。
    assert {"lr", "batch_size", "max_epochs", "patience"} <= set(record["hyperparameters"])
    assert set(record["hyperparameters"]) == set(delivered["hyperparameters"])
    assert set(record["metrics"]) == set(delivered["metrics"])
    assert record["arm"] == "a4_polybert"
    assert record["mode"] == "finetune"
    assert record["env_fingerprint"] == "results/env_fingerprints/env_polybert.json"
    assert record["n_test"] == 69
    assert record["predictions"] == f"results/aggregate/predictions/{record['run_id']}.parquet"
    assert P.record_problems(record) == []


def test_finetune_record_on_cuda_carries_the_gpu_block(monkeypatch: pytest.MonkeyPatch) -> None:
    """device=cuda 的记录必须带 gpu 块，且块里两个确定性开关都为 true；关掉一个必须被校验抓住。
    A cuda record must carry the gpu block with both determinism switches on, and turning one off
    must be caught - otherwise "must be true" is only a comment.
    """
    monkeypatch.setattr(P.A, "gpu_block", _fake_gpu_block)
    record = _build_sample_record(device="cuda")
    assert "gpu" in record
    assert record["gpu"]["cudnn_deterministic"] is True
    assert record["gpu"]["deterministic_algorithms"] is True
    assert P.record_problems(record) == []

    record["gpu"]["cudnn_deterministic"] = False
    assert P.record_problems(record) != []


def test_finetune_record_rejects_a_prediction_count_mismatch() -> None:
    """预测条数 ≠ 真值条数必须当场报错，而不是落一条 `n_test` 对不上的记录。
    A count mismatch must raise instead of landing a record whose `n_test` disagrees.
    """
    with pytest.raises(A.ArmError, match="预测"):
        _build_sample_record(
            y_true=np.zeros(10, dtype=np.float64),
            y_pred=np.zeros(9, dtype=np.float64),
            n_test=9,
        )


# ---- 7. 微调冒烟（要 transformers） / Fine-tune smoke (needs transformers) ----


@needs_transformers
@needs_packs
def test_finetune_smoke_writes_a_consistent_record() -> None:
    """微调 1 轮：记录合格、预测条数 = 测试集条数、`mode=finetune`、环境指纹是 polybert 那个。
    One fine-tune epoch: the record validates, counts line up, mode says finetune, and the env
    fingerprint is the polyBERT one.
    """
    config = P.load_arm_config(CONFIG_PATH)
    data = P.build_task_data_finetune("pb_t3_iwlt")
    outcome = P.train_one_finetune(
        data, config, fold="official_70_30", seed=0, device="cuda", max_epochs=1, patience=0
    )
    assert outcome.record["mode"] == "finetune"
    assert outcome.record["env_fingerprint"] == "results/env_fingerprints/env_polybert.json"
    assert outcome.y_pred.size == outcome.y_true.size == outcome.record["n_test"]
    assert P.record_problems(outcome.record) == []
