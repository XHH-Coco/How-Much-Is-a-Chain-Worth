"""B-5 · OpenPoly 规模扫描的守门测试（不需要 GPU / torch / rdkit）。
Guard tests for B-5's OpenPoly scale scan (no GPU, torch or rdkit needed).

中文：
    用途：B-5 的数字要能进 H4，靠的是四件**不报错但会让结论错**的事都不发生：
          ① 超参被"顺手"重调过——那会把"信息 vs 架构"退化成"信息 vs 我调的超参"；
          ② n 档不嵌套——五档就会变成五次独立的子采样，曲线不再读作"数据变多"；
          ③ 子集依赖了模型种子——那 n 档就不只是"训练集大小"；
          ④ 截断顺手碰到了测试集——R² 的分子分母一起变，H4 失去解释力。
          这四条都在这里变成断言。跑它要 core 环境（`pytest tests/test_openpoly_scan.py`），
          GPU 侧的 `record_problems()` 由 `results/raw_runs/` 的实跑记录覆盖。
    English:
        Four things break B-5 silently: a retuned hyper-parameter, non-nested levels, a subset that
        depends on the model seed, and a truncation that touches the test set. Each becomes an
        assertion here; the GPU side is covered by `record_problems()` over the real records.
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

import numpy as np
import pytest
import yaml

from chainworth.arms import graph as G
from chainworth.io import loaders as L

REPO = Path(__file__).resolve().parents[1]
SCAN = REPO / "configs" / "experiments" / "openpoly_scan.yaml"
A3 = REPO / "configs" / "arms" / "a3_graph.yaml"
TASKS = ("op_tg", "op_dielectric_constant_total", "op_tm")
LEVELS = ("30", "60", "120", "240", "all")

# schema 的 run_id 形状（results/raw_runs/_schema.json）。
# The run_id shape the schema pins.
RUN_ID_RE = re.compile(r"^[a-z0-9_]+__[a-z0-9_]+__[a-z0-9_]+__seed[0-9]+$")


@pytest.fixture(scope="module")
def config() -> G.ArmConfig:
    """B-5 的冻结配置。/ B-5's frozen config."""
    return G.load_arm_config(SCAN)


@pytest.fixture(scope="module")
def grid(config: G.ArmConfig) -> list[G.RunKey]:
    """225 条的完整网格。/ The full 225-run grid."""
    return G.enumerate_runs(config)


# ---- 1. 配置：哈希、网格、超参没被重调 / Config: hash, grid, untouched hyper-parameters ----


def test_config_hash_is_the_file_bytes(config: G.ArmConfig) -> None:
    """`config_hash` 必须是磁盘上这份文件的字节哈希，不是解析后重排的 YAML。
    The hash must be the bytes on disk, not the re-dumped YAML.

    为什么单独立一条：`yaml.safe_load` + `safe_dump` 会把注释与键序抹掉，那样写出来的哈希
    与"文件被换掉"无关——改一个空格哈希不变，等于没有身份。
    Re-dumping YAML erases comments and key order, so the hash would no longer track "the file was
    swapped": change one space and the hash would not move.
    """
    digest = hashlib.sha256(SCAN.read_bytes()).hexdigest()
    assert config.sha256 == digest
    assert G.config_hash(config) == f"sha256:{digest}"


def test_grid_holds_225_runs(config: G.ArmConfig, grid: list[G.RunKey]) -> None:
    """225 = 3 性质 × 5 折 × 3 种子 × 5 档，且每个组合恰好一次。
    225 = 3 properties x 5 folds x 3 seeds x 5 levels, each combination exactly once.
    """
    assert len(grid) == 225
    combos = {(key.task, key.fold, key.seed, key.n_level) for key in grid}
    assert len(combos) == 225, "有重复组合：网格展开错了，run_id 会互相覆盖"
    assert {key.task for key in grid} == set(TASKS)
    assert {key.fold for key in grid} == {f"seed{k}" for k in range(5)}
    assert {key.seed for key in grid} == {0, 1, 2}
    assert {key.n_level for key in grid} == set(LEVELS)
    assert {key.arm for key in grid} == {"a3_graph"}


def test_architecture_and_training_match_a3_key_by_key(config: G.ArmConfig) -> None:
    """本实验的 `architecture` / `training` 必须与 `a3_graph.yaml` **逐键相同**。
    The architecture and training sections must match a3_graph.yaml key for key.

    为什么用断言而不是"我看过了"：分工_B §2 B-5 的原话是"复用 B-3 的配置，只改训练集大小，
    别的都不动"。少一个键、多一个键、或者把 `max_epochs` 从 100 改成 30，都会让 225 条与 48 条
    不在同一把尺子上，而**任何一种都不会报错**。C-3 验收④对经典臂用的是同款做法。
    Why an assertion: B-5 says "reuse B-3's config, change only the training-set size". A missing
    key, an extra key, or max_epochs 100 -> 30 would put the 225 and the 48 on different rulers, and
    none of them raises. C-3 uses the same machine check for the classical arm.
    """
    with A3.open("r", encoding="utf-8") as handle:
        reference = yaml.safe_load(handle)
    for section in ("architecture", "training", "device"):
        mine = config.raw[section]
        theirs = reference[section]
        assert set(mine) == set(theirs), (
            f"`{section}` 的键集合与 a3_graph.yaml 不同：多 {sorted(set(mine) - set(theirs))}、"
            f"少 {sorted(set(theirs) - set(mine))}"
        )
        for key in theirs:
            assert mine[key] == theirs[key], (
                f"`{section}.{key}` 被改过：本实验 {mine[key]!r}，a3_graph.yaml {theirs[key]!r}——"
                "B-5 只许改训练集大小"
            )


def test_levels_are_the_frozen_grid(config: G.ArmConfig) -> None:
    """档位是 A 冻结的（30/60/120/240/all），本实验不得增删。
    The levels are frozen by A; this experiment may not add or drop any.
    """
    assert config.raw["n_levels"] == [30, 60, 120, 240, "all"]
    assert config.raw["subset"]["seed"] != config.raw["training"]["val_split_seed"], (
        "子集种子与验证留出种子用了同一个数：两者是两条独立的随机流，同数会让人以为是同一条"
    )


def test_every_run_id_matches_the_schema(grid: list[G.RunKey]) -> None:
    """225 条 run_id 全部合法，且**互不相同**。/ All 225 ids are legal and pairwise distinct."""
    ids = [key.run_id() for key in grid]
    for run_id in ids:
        assert RUN_ID_RE.match(run_id), f"{run_id} 不符合 schema 的三段式"
    assert len(set(ids)) == 225


def test_a_run_without_a_level_keeps_the_old_run_id() -> None:
    """没有 n 档时 run_id 与 B-3 完全同形——已入库的 96 条记录不能因为这次改动换名字。
    Without an n-level the run_id is byte-identical to B-3's, so the 96 records on disk keep
    their names.
    """
    assert G.run_id_for("op_tg", "seed0", 0) == "op_tg__a3_graph__seed0__seed0"
    assert G.run_id_for("op_tg", "seed0", 0, n_level="all") == "op_tg__a3_graph__seed0_nall__seed0"


# ---- 2. n 档子集：嵌套、无偏、与模型种子无关 ----


def _train_rows(task: str, fold: str) -> np.ndarray:
    """该折的官方训练行（升序）。/ That fold's official training rows (ascending)."""
    payload = json.loads(
        (REPO / "splits" / "openpoly" / task / f"rule_replicated_{fold}.json").read_text(
            encoding="utf-8"
        )
    )
    return np.asarray(payload["train_index"], dtype=np.int64)


@pytest.mark.parametrize("task", TASKS)
def test_levels_nest(task: str) -> None:
    """n=30 ⊂ n=60 ⊂ n=120 ⊂ n=240 ⊂ all：五档只差"加进去的行"，不换样品。
    The levels nest, so they differ by added rows rather than by swapped samples.

    这一条是 H4 能否读成"数据变多"的前提。若五档各抽各的，曲线在两个原因之间摇摆
    （样本量 + 抽样噪声），κ 就没有可解释性。
    """
    rows = _train_rows(task, "seed0")
    previous: set[int] = set()
    for level in LEVELS:
        picked, info = G.n_level_rows(rows, level, seed=20261122)
        current = set(picked.tolist())
        assert previous <= current, f"{task} 的 n={level} 不是上一档的超集"
        assert len(picked) == info["n_train_requested"] or info["n_level_saturated"]
        previous = current


@pytest.mark.parametrize("task", TASKS)
def test_subset_never_invents_rows(task: str) -> None:
    """子集只能是官方训练行的子集——不许出现测试集/验证集里的行。
    A subset may only contain official training rows, never a validation or test row.

    如果这里混进一行测试数据，那一条 run 的 R² 就是泄漏出来的，而它**看起来只是"效果好"**。
    A single leaked row would make that run's R² look merely "good".
    """
    rows = _train_rows(task, "seed0")
    payload = json.loads(
        (REPO / "splits" / "openpoly" / task / "rule_replicated_seed0.json").read_text(
            encoding="utf-8"
        )
    )
    held_out = set(payload["val_index"]) | set(payload["test_index"])
    for level in LEVELS:
        picked, _ = G.n_level_rows(rows, level, seed=20261122)
        assert set(picked.tolist()) <= set(rows.tolist())
        assert not (set(picked.tolist()) & held_out), f"{task}/n={level} 里混进了验证或测试行"


@pytest.mark.parametrize("task", TASKS)
def test_subset_is_deterministic(task: str) -> None:
    """同一个种子调两次给同一批行（换进程也一样，因为只有 numpy 一个随机源）。
    The same seed twice gives the same rows, in any process, because numpy is the only RNG.
    """
    rows = _train_rows(task, "seed0")
    first, info_a = G.n_level_rows(rows, "120", seed=20261122)
    second, info_b = G.n_level_rows(rows, "120", seed=20261122)
    assert np.array_equal(first, second)
    assert info_a["n_subset_sha256"] == info_b["n_subset_sha256"]


def test_subset_does_not_depend_on_the_model_seed() -> None:
    """子集只由 `subset.seed` 决定，与模型种子无关——否则 n 档就不只是"训练集大小"。
    The subset depends only on `subset.seed`, never on the model seed; otherwise a level would be
    more than a size.

    三个模型种子本该只改初始化与批序。若训练行也跟着变，五档曲线的每个点都混了"换了样品"。
    """
    rows = _train_rows("op_tg", "seed0")
    digests = {
        G.n_level_rows(rows, "60", seed=20261122)[1]["n_subset_sha256"] for _ in (0, 1, 2)
    }
    assert len(digests) == 1


@pytest.mark.parametrize("task,expected_pool", [("op_tg", 355), ("op_dielectric_constant_total", 235), ("op_tm", 134)])
def test_saturation_is_reported_not_hidden(task: str, expected_pool: int) -> None:
    """240 档在池子不足时**如实标饱和**，不假装取到了 240。
    A 240 level over a small pool is reported as saturated, never as 240.

    `op_tm` 的训练行只有 134 行、`op_dielectric` 只有 235 行，两者的 240 档都取不满。
    这一条如果藏起来，D 会把 240 画在一个 134 行的点上，H4 的横轴就是假的。
    Tm has 134 training rows and dielectric 235, so neither fills a 240 level. Hiding that would put
    a 240 on a 134-row point and make H4's x axis fiction.
    """
    rows = _train_rows(task, "seed0")
    assert rows.size == expected_pool
    picked, info = G.n_level_rows(rows, "240", seed=20261122)
    if expected_pool >= 240:
        assert picked.size == 240 and info["n_level_saturated"] is False
    else:
        assert picked.size == expected_pool and info["n_level_saturated"] is True


def test_bad_level_is_rejected() -> None:
    """档位写成 `0` / `-5` / `随便` 一律报错，不静默当成 all。
    A level of 0, -5 or nonsense raises instead of silently becoming "all".
    """
    rows = np.arange(100, dtype=np.int64)
    for bad in ("0", "-5", "abc"):
        with pytest.raises(G.ArmError):
            G.n_level_rows(rows, bad, seed=1)


def test_unknown_level_filter_lists_the_known_ones(config: G.ArmConfig) -> None:
    """`--n-levels 500` 必须报错并列出可用档位，而不是跑出一个空网格。
    An unknown level filter must raise and name the available levels.
    """
    with pytest.raises(G.ArmError, match="n-levels"):
        G.enumerate_runs(config, n_levels=["500"])


# ---- 3. 记录校验：n 档的三条自洽关系 / Record checks: three consistency rules ----


def _record(**overrides: object) -> dict:
    """一条最小可用的 n 档记录（其余字段照抄实跑记录的形状）。
    A minimally valid n-level record; the other fields follow the shape of a real one.
    """
    record = {
        "run_id": "op_tm__a3_graph__seed0_n30__seed0",
        "status": "ok",
        "benchmark": "openpoly",
        "task": "op_tm",
        "arm": "a3_graph",
        "seed": 0,
        "split_version": "f78db0276d37@splits/openpoly/op_tm/rule_replicated_seed0.json",
        "split_fold": "seed0",
        "config_hash": "sha256:" + "a" * 64,
        "n_train": 30,
        "n_train_pool": 134,
        "n_train_requested": 30,
        "n_level": "30",
        "n_level_saturated": False,
        "n_subset_sha256": "sha256:" + "b" * 64,
        "n_test": 17,
        "metrics": {"r2": 0.2, "mae": 50.0, "rmse": 60.0},
        "wall_clock_s": 1.0,
        "device": "cpu",
        "env_fingerprint": "results/env_fingerprints/env_gnn.json",
        "predictions": "results/aggregate/predictions/op_tm__a3_graph__seed0_n30__seed0.parquet",
    }
    record.update(overrides)
    return record


def test_a_valid_n_record_passes() -> None:
    """基准记录必须过。/ The baseline record passes."""
    assert G.record_problems(_record()) == []


@pytest.mark.parametrize(
    "overrides,needle",
    [
        # 档位名与 run_id 不一致：文件名说 n30、字段说 n60。
        # The level disagrees with the run_id: the file says n30, the field says n60.
        ({"run_id": "op_tm__a3_graph__seed0_n60__seed0"}, "run_id"),
        # 实际行数不是 min(请求, 池子)：说明截断与记录不是一回事。
        # n_train is not min(requested, pool): the trimming and the record disagree.
        ({"n_train": 999}, "n_train"),
        # 池子只有 134 行却报 240 行、又不标饱和：D 会把它当成真的 240。
        # A 240 over a 134-row pool reported as unsaturated: D reads it as a real 240.
        ({"n_train": 240, "n_train_requested": 240}, "n_train"),
        # 标了饱和但请求值没顶到池子：反过来造假。
        # Marked saturated while the request never reached the pool: the reverse lie.
        ({"n_level_saturated": True}, "n_level_saturated"),
        # 没有子集哈希：这一档用了哪些行无法核对。
        # No subset hash: which rows this level used cannot be checked.
        ({"n_subset_sha256": ""}, "n_subset_sha256"),
    ],
)
def test_bad_n_records_are_caught(overrides: dict, needle: str) -> None:
    """五类 n 档不自洽都必须被抓住。/ All five inconsistency classes must be caught."""
    problems = G.record_problems(_record(**overrides))
    assert problems, f"这条记录应当不合格：{overrides}"
    assert any(needle in problem for problem in problems), problems


def test_a_record_without_a_level_is_unaffected() -> None:
    """没有 n 档的（B-3/B-4 的 96 条）不受新校验影响。/ Records without a level are untouched."""
    record = _record()
    for key in ("n_level", "n_train_pool", "n_train_requested", "n_level_saturated", "n_subset_sha256"):
        record.pop(key)
    record["run_id"] = "op_tm__a3_graph__seed0__seed0"
    record["predictions"] = "results/aggregate/predictions/op_tm__a3_graph__seed0__seed0.parquet"
    assert G.record_problems(record) == []


# ---- 4. 只截训练集：验证与测试在任何档位上逐行相同 ----


@pytest.mark.parametrize("task", TASKS)
@pytest.mark.parametrize("fold", ["seed0", "seed4"])
def test_holdouts_are_identical_across_levels(task: str, fold: str) -> None:
    """同一折的 val/test 索引与档位无关——这是 H4 的前提，也是本任务最容易破的一条。
    The val/test indices of a fold do not depend on the level: H4's premise, and the easiest thing
    to break here.

    截断只发生在 `train_rows`（`n_level_rows` 的入参）。这个测试拿真实划分再确认一次：
    五个档位的验证集与测试集逐行相同，且 n=30 时训练集与测试集没有交集。
    Trimming happens only on `train_rows`. This re-checks against the real splits: the holdouts are
    row-identical at all five levels, and at n=30 the training set still misses the test set.
    """
    payload = json.loads(
        (REPO / "splits" / "openpoly" / task / f"rule_replicated_{fold}.json").read_text(
            encoding="utf-8"
        )
    )
    val = np.asarray(payload["val_index"], dtype=np.int64)
    test = np.asarray(payload["test_index"], dtype=np.int64)
    train = np.asarray(payload["train_index"], dtype=np.int64)
    sizes = []
    for level in LEVELS:
        picked, _ = G.n_level_rows(train, level, seed=20261122)
        sizes.append(picked.size)
        assert not (set(picked.tolist()) & set(val.tolist()))
        assert not (set(picked.tolist()) & set(test.tolist()))
    # 档位大小必须单调不减，且 all 档等于池子大小。
    # Level sizes are monotone and "all" equals the pool.
    assert sizes == sorted(sizes)
    assert sizes[-1] == train.size


def test_task_configs_agree_on_the_row_counts() -> None:
    """三个任务配置的 `n_labels` 与划分文件的行数必须一致（443 / 295 / 168）。
    The three task configs' n_labels must equal the split files' row counts.

    这三个任务是**没有去重**的（Tg 443 行只对应 304 个结构，D-32）：去重会让折索引错位。
    所以这里比的是行数，不是结构数。
    These three tasks are **not** deduplicated (Tg's 443 rows hold 304 structures, D-32) because
    deduplication would misalign the fold indices, so this compares rows, not structures.
    """
    for task in TASKS:
        task_cfg = L.load_task_config(task)
        payload = json.loads(
            (REPO / "splits" / "openpoly" / task / "rule_replicated_seed0.json").read_text(
                encoding="utf-8"
            )
        )
        total = int(payload["n_train"]) + int(payload["n_val"]) + int(payload["n_test"])
        assert total == int(task_cfg["n_labels"]) == int(payload["n_total"])
        assert total <= 443, "B-5 验收②：一律 n ≤ 443，不许外推"
