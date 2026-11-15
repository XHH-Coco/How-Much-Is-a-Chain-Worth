"""C-b-2 · OpenPoly 经典臂 225 条的守门测试（不需要 GPU / torch / rdkit）。
Guard tests for C-b-2's 225-run classical OpenPoly scan (no GPU, torch or rdkit needed).

中文：
    用途：这一批数要能进 H4，靠的是四件**不报错但会让结论错**的事都不发生：
          ① 与 B 的 GNN 用了不同的训练行——那交叉点的差就不是"臂的差"而是"子集的差"（红线 2）；
          ② 借了别的任务的超参、或偷偷重搜——κ 与主表就不是同一个世界（红线 3）；
          ③ 240 档被池子卡住却没记 `n_level_saturated`——D 会把它当成"曲线右端抖了一下"；
          ④ 只存指标不存预测——D 的重采样与过零点无从下手。
          四条都在这里变成断言；数据不在的机器上自动跳过（与 `test_openpoly_scan.py` 同一套约定）。

English:
    Four things break H4 silently: a different training-row subset than B's GNN (then the crossing
    point differences the subset, not the arm), borrowed or quietly retuned hyper-parameters, an
    unflagged saturated tier, and metrics without predictions. Each becomes an assertion here, and
    the whole module skips on a machine without the artefacts.
"""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import pytest
import yaml

from chainworth.arms import graph as G

REPO = Path(__file__).resolve().parents[1]
CONFIG = REPO / "configs" / "experiments" / "openpoly_classic.yaml"
A2_CONFIG = REPO / "configs" / "arms" / "a2_conventional.yaml"
TABLE = REPO / "results" / "aggregate" / "openpoly_cpu.csv"
N_INDEX = REPO / "results" / "aggregate" / "openpoly_cpu_n_index.csv"
GPU_N_INDEX = REPO / "results" / "aggregate" / "openpoly_n_index.csv"
RAW_RUNS = REPO / "results" / "raw_runs"
TASKS = ("op_tg", "op_dielectric_constant_total", "op_tm")
LEVELS = ("30", "60", "120", "240", "all")
# A2 冻结配置的字节哈希（README §九 登记的那一行）：C-b 只准读它，不准改。
# The byte hash of A2's frozen config as registered in README section 9: C-b may read it, not edit it.
A2_SHA256 = "7d29768afc09624e962b75eef6d9fe3293fe408392f9a5ce943ec390c4b22d7f"


@pytest.fixture(scope="module")
def config() -> G.ArmConfig:
    """C-b-2 的冻结实验配置。/ C-b-2's frozen experiment config."""
    return G.load_arm_config(CONFIG)


@pytest.fixture(scope="module")
def grid(config: G.ArmConfig) -> list[G.RunKey]:
    """225 条的完整网格。/ The full 225-run grid."""
    return G.enumerate_runs(config)


def _records() -> dict[str, dict]:
    """磁盘上属于这批的记录（arm=a2_conventional 且有 n 档）。
    The records on disk that belong to this batch (A2 arm, carrying a level).
    """
    out: dict[str, dict] = {}
    for path in sorted(RAW_RUNS.glob("*.json")):
        if path.name == "_schema.json":
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("arm") == "a2_conventional" and payload.get("n_level") is not None:
            out[str(payload["run_id"])] = payload
    return out


def _n_index_map(path: Path) -> dict[tuple[str, str, str], str]:
    """(task, fold, 档位) → subset_sha256。/ (task, fold, level) -> subset_sha256."""
    if not path.is_file():
        return {}
    with path.open("r", encoding="utf-8", newline="") as handle:
        return {
            (row["task"], row["fold"], row["n_level"]): row["subset_sha256"]
            for row in csv.DictReader(handle)
        }


# ---- 1. 配置：身份、网格、没有重调超参 / Config: identity, grid, untouched hyper-parameters ----


def test_config_hash_is_the_file_bytes(config: G.ArmConfig) -> None:
    """`config_hash` 必须是磁盘上这份文件的字节哈希。
    The hash must be the bytes on disk, not a re-dumped YAML.
    """
    digest = hashlib.sha256(CONFIG.read_bytes()).hexdigest()
    assert config.sha256 == digest
    assert G.config_hash(config) == f"sha256:{digest}"


def test_grid_holds_225_runs(grid: list[G.RunKey]) -> None:
    """225 = 3 性质 × 5 折 × 3 种子 × 5 档，且每个组合恰好一次。
    225 = 3 properties x 5 folds x 3 seeds x 5 levels, each combination exactly once.
    """
    assert len(grid) == 225
    assert len({(key.task, key.fold, key.seed, key.n_level) for key in grid}) == 225
    assert {key.task for key in grid} == set(TASKS)
    assert {key.fold for key in grid} == {f"seed{k}" for k in range(5)}
    assert {key.n_level for key in grid} == set(LEVELS)
    assert {key.arm for key in grid} == {"a2_conventional"}


def test_expected_fits_matches_the_grid(config: G.ArmConfig, grid: list[G.RunKey]) -> None:
    """配置里写死的 `expected_fits` 必须等于网格实际条数（对账只引这一个数）。
    The config's frozen expected_fits must equal the grid size - reconciliation quotes only it.
    """
    assert int(config.raw["expected_fits"]) == len(grid)


def test_a2_frozen_config_is_untouched() -> None:
    """A2 的冻结配置一个字节都没被 C-b 改过（改了 = 96 条主表记录当场作废）。
    C-b did not touch A2's frozen config: changing it would void all 96 main-table records.
    """
    assert hashlib.sha256(A2_CONFIG.read_bytes()).hexdigest() == A2_SHA256


def test_hyperparameters_are_not_copied_into_the_experiment_config(config: G.ArmConfig) -> None:
    """实验配置里 `learner.params` 必须是空的：超参只从 `a2_best_params.csv` 取，不许另起一套。
    `learner.params` must be empty: hyper-parameters come only from a2_best_params.csv.
    """
    learner = config.section("learner")
    assert learner.get("main") == "xgboost"
    assert not learner.get("params"), "实验配置里出现了一套自己的超参：红线 3"


def test_subset_seed_matches_the_gnn_side(config: G.ArmConfig) -> None:
    """子集种子必须与 B-5 的 `openpoly_scan.yaml` 逐字相同，否则两边必然落在不同训练行上。
    The subset seed must match B-5's scan config character for character; otherwise the two arms
    cannot be on the same training rows.
    """
    theirs = yaml.safe_load((REPO / "configs" / "experiments" / "openpoly_scan.yaml").read_text(encoding="utf-8"))
    assert config.section("subset")["seed"] == theirs["subset"]["seed"]
    assert config.section("subset")["rule"] == theirs["subset"]["rule"]
    assert list(config.raw["n_levels"]) == list(theirs["n_levels"])


# ---- 2. 实跑记录：字段、预测、饱和档（数据不在就跳过） / Records: fields, predictions, saturation ----


def test_records_carry_the_level_fields(config: G.ArmConfig, grid: list[G.RunKey]) -> None:
    """225 条都在、都是 ok、都带 n 档的六个数（`subset_sha256` 是 H4 的凭据）。
    All 225 present, ok, and carrying the six level fields (subset_sha256 is H4's evidence).
    """
    records = _records()
    if not records:
        pytest.skip("results/raw_runs 里还没有 C-b-2 的记录")
    missing = [key.run_id() for key in grid if key.run_id() not in records]
    assert not missing, f"缺 {len(missing)} 条记录，例：{missing[:3]}"
    expected_hash = G.config_hash(config)
    for key in grid:
        record = records[key.run_id()]
        assert record["status"] == "ok"
        assert record["config_hash"] == expected_hash
        assert record["device"] == "cpu"
        assert record["n_level"] == key.n_level
        assert str(record["n_subset_sha256"]).startswith("sha256:"), "缺子集身份，H4 无法核对"
        assert int(record["n_train"]) <= int(record["n_train_pool"])
        assert record["split_fold"] == key.fold


def test_every_record_has_predictions_of_the_right_length(grid: list[G.RunKey]) -> None:
    """每条 run 的预测文件都在，行数 = 该折 test 长度（D 的重采样要用）。
    Every run's prediction file exists and has exactly the fold's test length.
    """
    records = _records()
    if not records:
        pytest.skip("results/raw_runs 里还没有 C-b-2 的记录")
    for key in grid:
        record = records[key.run_id()]
        path = REPO / str(record["predictions"])
        assert path.is_file(), f"缺预测 {path.name}"
        import pyarrow.parquet as pq

        assert pq.read_table(path).num_rows == int(record["n_test"])


def test_subsets_match_the_gnn_side_level_by_level() -> None:
    """逐 (task, fold, 档位) 比 `subset_sha256`：与 B-5 的 GNN 必须是同一批训练行（红线 2）。
    Compare subset_sha256 per (task, fold, level): the rows must be B-5's GNN rows (red line 2).
    """
    ours = _n_index_map(N_INDEX)
    theirs = _n_index_map(GPU_N_INDEX)
    if not ours or not theirs:
        pytest.skip("缺一侧的 n 索引：先跑 openpoly 子命令")
    shared = sorted(set(ours) & set(theirs))
    assert len(shared) >= 75, f"共有键只有 {len(shared)} 个（应为 3 性质 × 5 折 × 5 档 = 75）"
    mismatched = {key: (ours[key], theirs[key]) for key in shared if ours[key] != theirs[key]}
    assert not mismatched, f"{len(mismatched)} 个档位的训练行与 GNN 不一致，例：{list(mismatched)[:2]}"


def test_saturated_levels_score_identically_to_all(grid: list[G.RunKey]) -> None:
    """240 档在两个性质上取不满：它的 R² 必须与 `all` 档逐位相同（D-35 的自检）。
    The 240 tier cannot be filled on two properties, so its R² must be bit-identical to `all`.
    """
    records = _records()
    if not records:
        pytest.skip("results/raw_runs 里还没有 C-b-2 的记录")
    saturated = [record for record in records.values() if record.get("n_level_saturated")]
    assert saturated, "没有任何饱和档：op_dielectric 的池子 235、op_tm 的 134，240 档必然取不满"
    by_key = {
        (record["task"], record["split_fold"], record["seed"], record["n_level"]): record
        for record in records.values()
    }
    for record in saturated:
        assert record["n_level"] == "240"
        twin = by_key[(record["task"], record["split_fold"], record["seed"], "all")]
        assert int(record["n_train"]) == int(twin["n_train"])
        assert record["n_subset_sha256"] == twin["n_subset_sha256"]
        assert record["metrics"]["r2"] == twin["metrics"]["r2"], "同一条数据同一个种子应当逐位相同"


# ---- 3. 表：覆盖与一致性 / The table: coverage and agreement ----


def test_table_covers_the_grid_and_matches_the_records(grid: list[G.RunKey]) -> None:
    """`openpoly_cpu.csv` 一 fit 一行、225 行，且每个数都能在记录里找到同一份。
    openpoly_cpu.csv has one row per fit (225) and every number is traceable to a record.
    """
    if not TABLE.is_file():
        pytest.skip("还没有 openpoly_cpu.csv")
    with TABLE.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == len(grid) == 225
    assert len({row["run_id"] for row in rows}) == 225, "run_id 有重复：表被合并坏了"
    records = _records()
    if not records:
        pytest.skip("results/raw_runs 里还没有 C-b-2 的记录")
    for row in rows:
        record = records[row["run_id"]]
        assert row["n_level"] == str(record["n_level"])
        assert row["subset_sha256"] == record["n_subset_sha256"]
        # 表里的指标是 `graph.write_n_index()` 按 6 位小数写的（与 B-5 的索引同一口径），
        # 所以容差取 5e-7；超过它就说明表与记录不是同一批数。
        # The index writer rounds to six decimals (the same convention as B-5's index), so the
        # tolerance is 5e-7; anything larger means the table and the records disagree.
        assert abs(float(row["r2"]) - float(record["metrics"]["r2"])) < 5e-7
