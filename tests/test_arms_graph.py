"""B-3 的断言测试：网格是 48 条、记录合规、早停留出固定、指标与预测对得上、同种子可复现。
Assertions for B-3: the grid holds 48 runs, records validate, the early-stopping hold-out is fixed,
metrics match the predictions, and the same seed reproduces.

中文：
    用途：把分工_B §2 B-3 的五条验收标准变成 `pytest`，分三类：
          ① **纯函数（core 环境也跑）**：配置哈希 = 文件字节哈希、网格 48 条且不重名、
             `--folds` 打错必须报错、`run_id` 符合 schema 的 pattern、早停留出"与模型种子无关、
             只落在 train 内、升序不重叠"、`metrics()` 的口径与边界（全对 → r2=1；常数真值 → NaN
             而不是 0）、`record_problems()` 能逐条抓出坏记录。
          ② **需要训练（GNN 环境才跑）**：冒烟一条 run，断言预测行数 = 测试集行数、`row_id`
             落在官方 `test_index` 内、记录里 `device` 与真实设备一致、`split_version` 的哈希
             等于折文件的哈希；同 seed 连跑两次预测最大差 < 1e-3（验收④）。
          ③ **闸门**：`device=cuda` 的记录必须带 `gpu` 块、缺 `predictions` 的 `ok` 记录必须
             被判不合格——**这一条是给"手写记录"准备的**，它必须在有人伪造时响。
    依赖：`configs/arms/a3_graph.yaml`（入库）、`data/processed/features/*.npz`（不入库）。
          缺装配产物时训练类用例跳过；没装 torch 的 core 环境里训练类用例整组跳过。
    配置：无环境变量。跑训练类用例请用 `DSC6020_Project_GNN` 环境。
    对外：pytest 用例。被谁用：CI、任何改 `arms/graph.py` 或 `a3_graph.yaml` 的人、D 的入口审计。

English:
    Purpose: turn B-3's five criteria into pytest cases in three groups: (1) pure functions that also
    run in the core environment - config hash equals the file's byte hash, the grid holds 48 uniquely
    named runs, a mistyped `--folds` raises, `run_id` matches the schema pattern, the
    early-stopping hold-out is independent of the model seed / inside train only / sorted and
    disjoint, `metrics()` edge cases (perfect predictions give r2=1, constant truth gives NaN rather
    than 0), and `record_problems()` catches every broken field; (2) cases that need training (GNN
    environment) - a smoke run whose prediction count equals the test size, whose `row_id`s all come
    from the official `test_index`, whose `device` matches reality, whose `split_version` equals the
    fold file's digest, plus a same-seed repeat under 1e-3 (criterion 4); (3) gates - a cuda record
    must carry its `gpu` block and an `ok` record without predictions must be rejected, which is the
    case written for hand-made records.
    Depends on: configs/arms/a3_graph.yaml (committed) and data/processed/features/*.npz (not
    committed); training cases skip without the artefacts, or without torch in the core environment.
    Config: none. Run the training cases in DSC6020_Project_GNN.
    Public API: pytest cases. Callers: CI, anyone touching arms/graph.py or a3_graph.yaml, D's audit.
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

import numpy as np
import pytest

from chainworth.arms import graph as A
from chainworth.io import loaders as L

HAS_PACKS = L.FEATURES_DIR.is_dir() and any(L.FEATURES_DIR.glob("*.npz"))

needs_packs = pytest.mark.skipif(
    not HAS_PACKS,
    reason="缺 B-1 装配产物（data/processed 不入库）：先跑 `python scripts/03b_assemble.py`",
)

# 最小任务的折名与任务名：训练类用例只碰 228 行的那一个，CPU 上也能在几十秒内跑完。
# The smallest task/fold: training cases touch only the 228-row one, quick even on CPU.
SMALL_TASK = "pb_t3_iwlt"
SMALL_FOLD = "official_70_30"


def _good_record() -> dict:
    """造一条**合规**的记录（训练脚本产出的形状），供"改坏它"的用例当基线。
    Build one **valid** record (the shape the trainer produces) as the baseline for the mutation
    cases.

    中文：用途：坏记录用例必须从一条好记录出发，否则"它报错了"可能只是因为基线本身就错。
          输入：无。输出：dict。数据来源：无（照 `train_one()` 的字段抄）。副作用：无。异常：无。
    English: mutation cases must start from a valid baseline, or "it raised" may just mean the
    baseline was wrong. Args: none. Returns: a dict mirroring `train_one()`'s fields.
    """
    run_id = "pb_t3_iwlt__a3_graph__official_70_30__seed0"
    return {
        "run_id": run_id,
        "status": "ok",
        "benchmark": "polybench",
        "task": "pb_t3_iwlt",
        "arm": "a3_graph",
        "seed": 0,
        "split_version": "6a9d1d91dc0a@splits/polybench/pb_t3_iwlt/official_70_30.json",
        "split_fold": "official_70_30",
        "config_hash": "sha256:" + "a" * 64,
        "n_train": 143,
        "n_test": 69,
        "metrics": {"r2": 0.5, "mae": 10.0, "rmse": 12.0},
        "wall_clock_s": 1.0,
        "device": "cpu",
        "env_fingerprint": "results/env_fingerprints/env_gnn.json",
        "predictions": f"results/aggregate/predictions/{run_id}.parquet",
    }


def _gpu_block(*, cudnn_deterministic: bool, deterministic_algorithms: bool) -> dict:
    """造 `device=cuda` 记录里的 `gpu` 块（只保留校验器会读的那几个键）。
    Build the record's `gpu` block, keeping only the keys the validator reads.

    中文：用途：两个确定性开关的守卫要单独测，这里只把"开关状态"参数化，其余字段留成常量。
          输入：两个开关的取值。输出：dict。数据来源：无。副作用：无。异常：无。
    English: the two determinism flags need their own cases, so only their values vary here.
    """
    return {
        "model": "NVIDIA GeForce RTX 4070",
        "driver": "591.86",
        "torch_cuda": "12.1",
        "cudnn_deterministic": cudnn_deterministic,
        "deterministic_algorithms": deterministic_algorithms,
    }


# ---- 1. 配置与网格 / Config and grid ----


def test_config_hash_is_the_file_bytes() -> None:
    """`config_hash` 必须是文件字节的 sha256——手写版本号在文件被换时不会变。
    `config_hash` must be the file's byte digest; a hand-written version does not change on a swap.

    中文：
        含义：这条是在守"48 条 run 的记录真的指向同一份配置"。做法不是去相信 `ArmConfig.sha256`，
              而是**自己重新读一遍文件**算一次，两边对上才算通过。
        输入：无。输出：无（断言）。副作用：只读。异常：AssertionError —— 哈希与文件不符。
    English:
        This guards "all 48 records really point at one config". Instead of trusting
        `ArmConfig.sha256`, the test re-reads the file and recomputes the digest.
    """
    config = A.load_arm_config()
    expected = hashlib.sha256(Path(config.path).read_bytes()).hexdigest()
    assert config.sha256 == expected
    assert config.hash == f"sha256:{expected}"


def test_grid_holds_48_uniquely_named_runs() -> None:
    """网格 = 6 × 1 × 3 + 2 × 5 × 3 = 48，且 `run_id` 两两不同。
    The grid is 6 x 1 x 3 + 2 x 5 x 3 = 48 and every `run_id` is distinct.

    中文：
        含义：分工_B 的"48 条"是主表的输入端，多一条少一条都要能当场发现。
              同时断言 PB26 的 30 条按五折展开（不是被当成单折跑了一遍算 6 条）。
        输入：无。输出：无（断言）。副作用：只读。异常：AssertionError。
    English:
        The 48 runs feed the main table; one missing or extra must fail here. It also asserts the
        PB26 half expands into five folds rather than being run once and counted as six.
    """
    config = A.load_arm_config()
    keys = A.enumerate_runs(config)
    assert len(keys) == 48
    ids = [key.run_id() for key in keys]
    assert len(set(ids)) == 48
    # 结构：PolyBench 18 条（6 任务 × 3 种子），PB26 30 条（2 任务 × 5 折 × 3 种子）。
    assert sum(1 for key in keys if key.benchmark == "polybench") == 18
    assert sum(1 for key in keys if key.benchmark == "polybench26") == 30
    assert len({key.fold for key in keys if key.benchmark == "polybench26"}) == 5


def test_grid_rejects_unknown_fold_and_task() -> None:
    """`--folds` / `--tasks` 打错必须报错，不能静默跑 0 条或跳过。
    A mistyped `--folds`/`--tasks` must raise instead of silently running zero or skipping.

    中文：
        含义：静默跳过是"看起来跑完了"的最短路径——输出目录空空如也但退出码是 0。
        输入：无。输出：无（断言）。副作用：只读。异常：无（用 pytest.raises 抓）。
    English: a silent skip is the shortest path to "looks finished" - an empty output directory and
    exit code 0.
    """
    config = A.load_arm_config()
    with pytest.raises(A.ArmError):
        A.enumerate_runs(config, folds=["fold_99"])
    with pytest.raises(A.ArmError):
        A.enumerate_runs(config, tasks=["pb_not_a_task"])


def test_run_id_matches_the_schema_pattern() -> None:
    """`run_id` 必须匹配 `results/raw_runs/_schema.json` 里的 pattern。
    `run_id` must match the pattern in `results/raw_runs/_schema.json`.

    中文：
        含义：pattern 是 `^[a-z0-9_]+__[a-z0-9_]+__[a-z0-9_]+__seed[0-9]+$`。这里**从 schema 文件
              读 pattern**再编译，而不是在测试里抄一份——抄一份会在 schema 改版时悄悄失同步。
        输入：无。输出：无（断言）。副作用：只读。异常：AssertionError。
    English:
        The pattern is read **from the schema file** and compiled here rather than copied, because a
        copy silently goes stale when the schema changes.
    """
    schema = json.loads((A.RAW_RUNS_DIR / "_schema.json").read_text(encoding="utf-8"))
    pattern = re.compile(schema["properties"]["run_id"]["pattern"])
    config = A.load_arm_config()
    for key in A.enumerate_runs(config):
        assert pattern.fullmatch(key.run_id()), key.run_id()
    # 反例：大写折名会被拼出非法 id，必须在拼的时候就报错（而不是写文件时才发现）。
    with pytest.raises(A.ArmError):
        A.run_id_for(SMALL_TASK, "Fold-00", 0)


# ---- 2. 早停留出 / The early-stopping hold-out ----


def test_carve_validation_is_fixed_sorted_and_inside_train() -> None:
    """留出：同一固定种子 → 同一份；落在 train 内；升序、互斥、并集 = 原 train。
    The hold-out is fixed by its own seed, lives inside train, is sorted, disjoint, and covers it.

    中文：
        含义：这条守的是"早停的验证集不是测试集、也不随模型种子变"。
              三个断言：
              ① 两次调用结果逐位相同（固定种子）；
              ② 并集等于原训练索引、交集为空（只是切，不是删）；
              ③ 与"模型种子"无关——这里没有模型种子的入参，函数签名本身就是证据。
        输入：无。输出：无（断言）。副作用：无。异常：AssertionError。
    English:
        Guarding "the early-stopping set is not the test set and does not move with the model seed":
        two calls agree bit for bit, the union equals the original train indices with an empty
        intersection, and no model seed is even an argument.
    """
    base = np.arange(0, 1000, 3, dtype=np.int64)
    train_a, val_a = A.carve_validation(base, fraction=0.1, seed=20261121)
    train_b, val_b = A.carve_validation(base, fraction=0.1, seed=20261121)
    assert np.array_equal(train_a, train_b) and np.array_equal(val_a, val_b)
    assert np.array_equal(np.sort(train_a), train_a)
    assert np.array_equal(np.sort(val_a), val_a)
    assert len(val_a) == 33                                  # round(334 × 0.1) = 33
    assert np.intersect1d(train_a, val_a).size == 0
    assert np.array_equal(np.sort(np.concatenate([train_a, val_a])), base)
    with pytest.raises(A.ArmError):
        A.carve_validation(base, fraction=0.0, seed=1)


# ---- 3. 指标 / Metrics ----


def test_metrics_are_exact_on_known_inputs() -> None:
    """指标口径：全对 → r2=1/mae=0/rmse=0；常数真值 → r2 是 NaN 而不是 0。
    Exact metrics: perfect predictions give r2=1/mae=0/rmse=0; constant truth gives NaN, not 0.

    中文：
        含义：NaN 与 0 的区别是"不可用"与"和均值一样差"。测试集只有一行时就会走到这条，
              写成 0 会让下游把一条不可用的 run 当成"正常但很差"。
        输入：无。输出：无（断言）。副作用：无。异常：AssertionError。
    English: NaN means "not available", 0 means "as bad as the mean"; writing 0 would turn an
    unusable run into "normal but poor".
    """
    exact = A.metrics([1.0, 2.0, 3.0], [1.0, 2.0, 3.0])
    assert exact["r2"] == 1.0 and exact["mae"] == 0.0 and exact["rmse"] == 0.0
    flat = A.metrics([5.0, 5.0], [4.0, 6.0])
    assert np.isnan(flat["r2"])
    assert flat["rmse"] == pytest.approx(1.0)
    with pytest.raises(A.ArmError):
        A.metrics([1.0, 2.0], [1.0])


# ---- 4. 记录合规 / Record validation ----


def test_good_record_passes_validation() -> None:
    """基线记录必须合格——否则下面所有"改坏它"的用例都失去意义。
    The baseline must pass, or every mutation case below proves nothing.

    中文：输入：无。输出：无（断言）。副作用：无。异常：AssertionError。
    English: Args: none. Returns: none (asserts).
    """
    assert A.record_problems(_good_record()) == []


def test_cuda_record_with_both_determinism_flags_passes() -> None:
    """两个开关都为 true 的 cuda 记录必须合格——守卫要抓的是"关掉"，不是"用了 GPU"。
    A cuda record with both flags set must pass: the guard is against turning determinism off,
    not against using the GPU.

    中文：
        含义：上一条用例证明"关掉会被抓"，这一条证明它不会误伤。两条一起才说明守卫的边界在
              "开关的取值"上。若只留"关掉会被抓"，把守卫写成"cuda 一律报错"也照样能过。
        输入：无。输出：无（断言）。副作用：无。异常：AssertionError。
    English: the previous case proves "off is caught"; this one proves it does not misfire. Together
    they pin the boundary at the flag values, not at "cuda is suspicious".
    """
    record = _good_record()
    record["device"] = "cuda"
    record["gpu"] = _gpu_block(cudnn_deterministic=True, deterministic_algorithms=True)
    assert A.record_problems(record) == []


@pytest.mark.parametrize(
    "mutation, expected",
    [
        (lambda row: row.pop("split_version"), "split_version"),
        (lambda row: row.update(run_id="wrong__a3_graph__official_70_30__seed0"), "run_id"),
        (lambda row: row.update(config_hash="deadbeef"), "config_hash"),
        (lambda row: row.update(split_version="v1"), "split_version"),
        (lambda row: row.update(predictions="results/raw_runs/x.parquet"), "predictions"),
        (lambda row: row.update(device="cuda"), "gpu"),
        (
            lambda row: row.update(
                device="cuda",
                gpu=_gpu_block(cudnn_deterministic=True, deterministic_algorithms=False),
            ),
            "deterministic_algorithms",
        ),
        (
            lambda row: row.update(
                device="cuda",
                gpu=_gpu_block(cudnn_deterministic=False, deterministic_algorithms=True),
            ),
            "cudnn_deterministic",
        ),
        (lambda row: row["metrics"].pop("rmse"), "metrics"),
        (lambda row: row.update(env_fingerprint="results/env.json"), "env_fingerprint"),
    ],
)
def test_bad_records_are_caught(mutation, expected) -> None:
    """逐条改坏一份好记录，校验器必须报出对应的那一项。
    Break one field at a time and the validator must name it.

    中文：
        含义：这组用例是给"手写记录"准备的闸门。红线 4 说记录只能由训练脚本落盘，
              而唯一能自动发现"有人手写了一条"的办法就是**字段与形状校验**。
        输入：mutation —— 把记录改坏的函数；expected —— 报错里必须出现的字样。
        输出：无（断言）。副作用：无。异常：AssertionError。
    English:
        This group is the gate against hand-written records. Red line 4 says only the trainer writes
        them, and field/shape validation is the only automatic way to notice that someone did not.
    """
    record = _good_record()
    mutation(record)
    problems = A.record_problems(record)
    assert any(expected in item for item in problems), problems


def test_record_rejects_nan_metrics_by_serialising_to_null() -> None:
    """NaN 指标必须落成 JSON 的 `null`，不能写成裸 `NaN`（那不是合法 JSON）。
    A NaN metric must become JSON `null`, never a bare `NaN` (which is not valid JSON).

    中文：
        含义：`json.dumps` 默认写 `NaN`，Python 读得回来、`jq` 与别的语言读不回来。这里检查
              `_json_safe()` 的递归替换，并用 `json.loads(..., parse_constant=...)` 反过来证明
              落盘的内容是合法 JSON。
        输入：无。输出：无（断言）。副作用：无。异常：AssertionError。
    English: `json.dumps` writes a bare `NaN` by default; Python reads it back and `jq` does not.
    This checks the recursive replacement and proves the payload is valid JSON.
    """
    payload = {"metrics": {"r2": float("nan"), "mae": 1.0}, "nested": [float("inf"), 2]}
    safe = A._json_safe(payload)                              # noqa: SLF001 - 私有函数是本用例的对象
    assert safe["metrics"]["r2"] is None and safe["nested"][0] is None
    text = json.dumps(safe)
    assert "NaN" not in text and "Infinity" not in text
    json.loads(text, parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)))


def test_fold_version_points_at_the_fold_file() -> None:
    """`split_version` 的路径段必须是这一折自己的文件，哈希段等于该文件的 sha256 前 12 位。
    The path segment must be the fold's own file and the digest its first 12 hex characters.

    中文：
        含义：PolyBench 一折一个 JSON、PB26 一折一个 NPZ——只写"整份划分"的哈希，
              折文件被换过时记录不会有任何变化。
        输入：无。输出：无（断言）。副作用：只读。异常：AssertionError。
    English: PolyBench has one JSON per task and PB26 one NPZ per fold; recording only the whole
    split's digest would not change when a fold file is swapped.
    """
    spec = L.load_split(SMALL_TASK)
    version, digest = A.fold_version(spec, SMALL_FOLD)
    head, _, path = version.partition("@")
    assert len(head) == 12 and path.startswith("splits/")
    assert digest is not None
    actual = hashlib.sha256((A.REPO_ROOT / path).read_bytes()).hexdigest()
    assert digest == actual and head == actual[:12]


# ---- 5. 需要训练的用例（GNN 环境） / Cases that need training (GNN environment) ----


def _tiny_run(*, max_epochs: int = 2, patience: int = 0):
    """在最小的任务上跑一条极短的 run（CPU、2 轮），返回 `TrainOutcome`。
    Run one very short run (CPU, 2 epochs) on the smallest task and return the outcome.

    中文：用途：多个用例共用同一段准备。`monkeypatch` 保证不改动仓库里的任何文件。
          输入：max_epochs / patience。输出：`TrainOutcome`。副作用：训练。异常：ArmError。
    English: shared setup for several cases; nothing in the repo is modified.
    """
    config = A.load_arm_config()
    data = A.build_task_data(SMALL_TASK)
    return A.train_one(
        data,
        config,
        fold=SMALL_FOLD,
        seed=0,
        device="cpu",
        max_epochs=max_epochs,
        patience=patience,
    )


@needs_packs
def test_train_one_produces_predictions_that_line_up_with_the_split() -> None:
    """冒烟一条 run：预测行数 = 测试集行数；`row_id` 全部落在官方 `test_index` 里。
    Smoke one run: prediction count equals the test size and every `row_id` comes from the official
    `test_index`.

    中文：
        含义：验收①要求"记录里的数与划分对得上"。这里不比对指标，只比对**身份**——
              预测文件里的 `row_id` 必须是官方测试集的行位置，不是 0..n-1 的序号。
        输入：无。输出：无（断言）。副作用：训练一次（CPU 数秒）。异常：AssertionError。
    English:
        Criterion 1 says the record's numbers line up with the split. This compares identity, not
        metrics: the parquet's `row_id` must be official test row positions, never a 0..n-1 ordinal.
    """
    pytest.importorskip("chemprop", reason="core 环境没有 chemprop：训练类用例在 GNN 环境里跑")
    outcome = _tiny_run()
    spec = L.load_split(SMALL_TASK)
    test_index = set(int(i) for i in spec.folds[0].test_index)
    assert outcome.row_id.size == outcome.y_pred.size == len(spec.folds[0].test_index)
    assert set(int(i) for i in outcome.row_id) <= test_index
    record = outcome.record
    assert record["n_test"] == outcome.row_id.size
    assert A.record_problems(record) == []
    assert record["device"] == "cpu"
    assert record["val_source"].startswith("inner_"), record["val_source"]


@needs_packs
def test_predictions_are_written_with_the_contract_columns(tmp_path: Path) -> None:
    """预测文件恰好两列 `row_id`(int64) / `y_pred`(float64)——D 的三个脚本都按这两列读。
    The predictions file has exactly two columns, row_id (int64) and y_pred (float64).

    中文：
        含义：多一列不会报错，只会让 D 的 `read_parquet` 之后那次 `df[["row_id","y_pred"]]` 悄悄
              丢掉信息；少一列会当场崩。所以列名与 dtype 都钉死。
        输入：tmp_path —— pytest 的临时目录。输出：无（断言）。副作用：写一个临时 parquet。
        异常：AssertionError。
    English: an extra column silently loses information at D's end; a missing one crashes. Names and
    dtypes are pinned here.
    """
    pandas = pytest.importorskip("pandas")
    pytest.importorskip("pyarrow", reason="写 parquet 需要 pyarrow（gnn.yml 已钉 17.0.0）")
    pytest.importorskip("chemprop", reason="core 环境没有 chemprop")
    outcome = _tiny_run()
    path = A.write_predictions(outcome, directory=tmp_path)
    frame = pandas.read_parquet(path)
    assert list(frame.columns) == ["row_id", "y_pred"]
    assert str(frame["row_id"].dtype) == "int64"
    assert str(frame["y_pred"].dtype) == "float64"
    assert len(frame) == outcome.row_id.size


@needs_packs
def test_same_seed_reproduces_predictions_within_1e_3() -> None:
    """同配置、同折、同种子跑两次，预测最大绝对差 < 1e-3（验收④）。
    Two runs with the same config, fold and seed differ by less than 1e-3 (criterion 4).

    中文：
        含义：分工书明写"不许只写'大致一致'"。这里把"大致"变成一个数，并把它与阈值放在一个
              断言里——阈值改了不会失同步。
              只跑 2 轮就够：这条查的是"确定性"，不是"收敛"。
        输入：无。输出：无（断言）。副作用：训练两次（CPU 十几秒）。异常：AssertionError。
    English:
        The division note forbids writing "roughly the same"; this turns "roughly" into one number
        next to its threshold. Two epochs suffice: this checks determinism, not convergence.
    """
    pytest.importorskip("chemprop", reason="core 环境没有 chemprop")
    first = _tiny_run()
    second = _tiny_run()
    delta = float(np.max(np.abs(first.y_pred - second.y_pred)))
    assert delta < 1e-3, f"同 seed 两次预测差 {delta:.3e}，超过 1e-3"
