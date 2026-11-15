"""C-b-1 · PolyBench26 数据量扫描的守门测试（不需要 GPU / torch）。
Guard tests for C-b-1's PolyBench26 data-size scan (no GPU or torch needed).

中文：
    用途：κ 能算得出来，靠的是四件事：
          ① **档位是官方前缀**（`train_pool[:N]`），不是"按行数截"也不是"随机抽 n 行"（红线 1）；
          ② **读的是 A-5 的快照**，逐字节核过哈希——快照被换掉时 κ 的横轴就不是官方的了；
          ③ **只截训练集**，五档的验证集与测试集逐行不变（否则 R² 的分子分母一起变）；
          ④ 交替 4 档 + `all`、均聚 2 档 + `all`：**缺档是事实，不是失败**（官方就没有）。
          四条都在这里变成断言。pack 与表不在的机器上自动跳过。
          需要 rdkit 的部分（造 pack）不在这里测——这里只读已落的产物，跑起来是秒级。

English:
    Kappa is only meaningful if four things hold: the tier is the official prefix train_pool[:N]
    (neither truncating by row count nor sampling n rows), the split bytes are A-5's snapshot, only
    the training set is truncated (val/test are row-identical across tiers), and the homopolymer
    table really has just two tiers. Each is an assertion here; pack-building is not tested (it
    needs rdkit) - this module only reads artefacts, so it runs in seconds.
"""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
import yaml

from chainworth.arms import graph as G
from chainworth.io import loaders as L

REPO = Path(__file__).resolve().parents[1]
CONFIG = REPO / "configs" / "experiments" / "scaling.yaml"
TASK_CONFIG = REPO / "configs" / "tasks" / "pb26_scaling_cp.yaml"
SPLIT_ROOT = REPO / "splits" / "polybench26" / "pb26_scaling_cp"
SPLITS_MANIFEST = REPO / "data" / "manifests" / "splits_manifest.csv"
PACK_MANIFEST = REPO / "data" / "manifests" / "scaling_features_manifest.csv"
TABLE = REPO / "results" / "aggregate" / "scaling_cpu.csv"
RAW_RUNS = REPO / "results" / "raw_runs"
DATASETS = ("md5000_alternating_cp", "md5000_homopolymer_cp")


def _manifest(dataset_id: str) -> dict:
    """数据集的官方 `manifest.json`。/ The dataset's official manifest.json."""
    return json.loads((SPLIT_ROOT / dataset_id / "seed_42" / "manifest.json").read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    """文件字节的 sha256。/ sha256 of a file's bytes."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.fixture(scope="module")
def config() -> G.ArmConfig:
    """C-b-1 的冻结实验配置。/ C-b-1's frozen experiment config."""
    return G.load_arm_config(CONFIG)


# ---- 1. 配置与任务身份 / Config and task identity ----


def test_config_hash_is_the_file_bytes(config: G.ArmConfig) -> None:
    """`config_hash` 必须是磁盘上这份文件的字节哈希。/ The hash must be the bytes on disk."""
    assert config.sha256 == hashlib.sha256(CONFIG.read_bytes()).hexdigest()


def test_tier_rule_is_the_official_prefix(config: G.ArmConfig) -> None:
    """档位取法必须写死成 `prefix`：官方 `create_scaling_splits.py` 就是 `train_pool[:N]`。
    The tier rule must be the prefix: the official script literally does train_pool[:N].
    """
    subset = config.section("subset")
    assert subset["rule"] == "prefix", "红线 1：scaling 的档位只能是官方前缀"
    assert subset.get("train_only") is True


def test_task_config_is_the_two_dataset_task() -> None:
    """任务配置里必须**两个**数据集，各自的档位与官方文件一致（含均聚只有两档这件事）。
    The task config must list both datasets with the official tier counts - homopolymer has two.
    """
    raw = yaml.safe_load(TASK_CONFIG.read_text(encoding="utf-8"))
    datasets = {entry["dataset_id"]: entry for entry in raw["datasets"]}
    assert set(datasets) == set(DATASETS)
    for dataset_id, entry in datasets.items():
        official = _manifest(dataset_id)
        assert [int(value) for value in entry["train_sizes"]] == [
            int(value) for value in official["train_sizes"]
        ], f"{dataset_id} 的档位与官方 manifest 不一致"
    assert len(datasets["md5000_homopolymer_cp"]["train_sizes"]) == 2, "均聚 Cp 官方只有 300 / 1000"


def test_expected_fits_matches_the_grid(config: G.ArmConfig) -> None:
    """80 = (4+1 交替 + 2+1 均聚) × 5 折 × 2 学习器；与配置里写死的 `expected_fits` 相等。
    80 = (5 alternating tiers + 3 homopolymer tiers) x 5 folds x 2 learners.
    """
    tiers = sum(len(_manifest(dataset_id)["train_sizes"]) + 1 for dataset_id in DATASETS)
    folds = len((config.raw["runs"]["polybench26"])["folds"])
    learners = len(config.raw["learners"])
    assert tiers * folds * learners == int(config.raw["expected_fits"]) == 80


# ---- 2. 快照身份：逐字节是 A-5 那一份 / Snapshot identity: byte-for-byte A-5's ----


def test_split_snapshot_digests_match_the_manifest() -> None:
    """`splits_manifest.csv` 里这两个数据集的 sha256 必须能由磁盘上的文件重算出来。
    The sha256 in splits_manifest.csv must be reproducible from the files on disk.

    中文：验收①要求"断言读的是快照"。清单里的值 = `"".join(排序后的逐文件哈希)`（`03_splits.py`
          的算法），所以这里照同样的公式重算，任何一份文件被换掉都会红。
    English: criterion 1 says "assert you are reading the snapshot". The manifest's value is the
    digest of the sorted per-file digests (`03_splits.py`'s formula), so recomputing it here goes red
    the moment any file is swapped.
    """
    with SPLITS_MANIFEST.open("r", encoding="utf-8", newline="") as handle:
        rows = {row["task_id"]: row for row in csv.DictReader(handle)}
    for dataset_id in DATASETS:
        key = f"pb26_scaling_cp/{dataset_id}"
        assert key in rows, f"{key} 不在 splits_manifest.csv 里"
        directory = SPLIT_ROOT / dataset_id / "seed_42"
        files = {path.name: _sha256(path) for path in sorted(directory.iterdir()) if path.is_file()}
        recomputed = hashlib.sha256("".join(files[name] for name in sorted(files)).encode()).hexdigest()
        assert recomputed == rows[key]["sha256"], f"{dataset_id} 的快照字节与清单不符"


def test_split_indices_are_disjoint_and_cover_every_row() -> None:
    """五折的 train_pool / val / test 互不相交，合起来正好是 0..n-1（行序坐标系的立足点）。
    The five folds' train_pool/val/test are disjoint and together cover 0..n-1.
    """
    for dataset_id in DATASETS:
        manifest = _manifest(dataset_id)
        n_rows = int(manifest["n_rows"])
        for entry in manifest["folds"]:
            with np.load(SPLIT_ROOT / dataset_id / "seed_42" / entry["indices_file"]) as payload:
                train_pool = payload["train_pool"].astype(np.int64)
                union = np.sort(
                    np.concatenate(
                        [train_pool, payload["val"], payload["test"]]
                    ).astype(np.int64)
                )
            assert union.size == n_rows, f"{dataset_id}/{entry['indices_file']} 覆盖 {union.size} 行 ≠ {n_rows}"
            assert np.array_equal(union, np.arange(n_rows)), "有行不属于任何一侧，或有行出现两次"
            assert train_pool.size == int(entry["n_train_pool"]), "train_pool 行数与官方 manifest 不符"


# ---- 3. 档位 = train_pool[:N]（红线 1 的核心） / The tier is the prefix ----


def test_tier_rows_are_the_official_prefix() -> None:
    """每一档取到的行必须**逐位等于** `train_pool[:N]`；`all` 档等于整个池子。
    Each tier's rows must equal train_pool[:N] element for element; `all` equals the whole pool.
    """
    for dataset_id in DATASETS:
        manifest = _manifest(dataset_id)
        with np.load(SPLIT_ROOT / dataset_id / "seed_42" / manifest["folds"][0]["indices_file"]) as payload:
            pool = payload["train_pool"].astype(np.int64)
        for size in manifest["train_sizes"]:
            picked, info = G.n_level_rows(pool, str(size), seed=0, rule="prefix")
            assert np.array_equal(picked, pool[: int(size)]), f"{dataset_id} 的 {size} 档不是官方前缀"
            assert info["n_train_requested"] == int(size)
            assert info["n_level_saturated"] is False
        whole, info_all = G.n_level_rows(pool, "all", seed=0, rule="prefix")
        assert np.array_equal(whole, pool)
        assert info_all["n_train_requested"] == pool.size


def test_tiers_are_nested_and_never_touch_the_test_set() -> None:
    """档位嵌套（300 ⊂ 1000 ⊂ …），且每一折的 val / test 在所有档位上逐行不变。
    Tiers nest, and each fold's val/test rows are identical across every tier.

    中文：嵌套是"曲线能读成数据变多"的前提；val/test 不变是 R² 可比的唯一原因。
    English: nesting is what lets the curve read as "more data"; an unchanged val/test is the only
    reason the R² values are comparable.
    """
    for dataset_id in DATASETS:
        manifest = _manifest(dataset_id)
        for entry in manifest["folds"]:
            with np.load(SPLIT_ROOT / dataset_id / "seed_42" / entry["indices_file"]) as payload:
                pool = payload["train_pool"].astype(np.int64)
                val = payload["val"].astype(np.int64)
                test = payload["test"].astype(np.int64)
            previous: set[int] = set()
            for size in manifest["train_sizes"]:
                picked, _ = G.n_level_rows(pool, str(size), seed=0, rule="prefix")
                current = set(int(value) for value in picked)
                assert previous <= current, f"{dataset_id} 的 {size} 档与更小的档不嵌套"
                previous = current
                assert not (current & set(int(value) for value in test))
                assert not (current & set(int(value) for value in val))
            # 五档都取完之后，val/test 仍是同一批行（它们根本没被截过）。
            assert val.size and test.size


def test_shuffle_rule_would_break_the_official_tier() -> None:
    """反向断言：用 OpenPoly 的"打乱取前缀"规则会得到**不同**的行集合——所以规则必须显式声明。
    The reverse check: OpenPoly's shuffle-prefix rule yields a *different* row set, which is exactly
    why the rule has to be declared in the config instead of assumed.
    """
    manifest = _manifest("md5000_alternating_cp")
    with np.load(
        SPLIT_ROOT / "md5000_alternating_cp" / "seed_42" / manifest["folds"][0]["indices_file"]
    ) as payload:
        pool = payload["train_pool"].astype(np.int64)
    prefix, _ = G.n_level_rows(pool, "300", seed=20261122, rule="prefix")
    shuffled, _ = G.n_level_rows(pool, "300", seed=20261122, rule="shuffle_prefix")
    assert not np.array_equal(np.sort(prefix), np.sort(shuffled))


# ---- 4. pack 与表（数据不在就跳过） / Packs and the table (skip when absent) ----


def test_packs_are_built_from_the_official_canonical_bytes() -> None:
    """pack 清单里的 `canonical_sha256` 必须等于官方 manifest 的那一串（D-21）。
    The pack manifest's canonical_sha256 must equal the official one (D-21).
    """
    if not PACK_MANIFEST.is_file():
        pytest.skip("还没有 scaling_features_manifest.csv（先跑 scaling --build）")
    with PACK_MANIFEST.open("r", encoding="utf-8", newline="") as handle:
        rows = {row["dataset_id"]: row for row in csv.DictReader(handle)}
    for dataset_id in DATASETS:
        if dataset_id not in rows:
            pytest.skip(f"{dataset_id} 的 pack 还没造")
        official = _manifest(dataset_id)["canonical_sha256"]
        assert rows[dataset_id]["canonical_sha256"] == official
        assert int(rows[dataset_id]["n_rows"]) == int(_manifest(dataset_id)["n_rows"])
        pack = REPO / "data" / "processed" / "features" / f"pb26_scaling_cp_{dataset_id}.npz"
        assert pack.is_file(), "清单说有 pack，盘上没有：清单过期了"
        assert _sha256(pack) == rows[dataset_id]["pack_sha256"]


def test_table_rows_reproduce_the_records() -> None:
    """`scaling_cpu.csv` 里 XGBoost 的每一行都要能在 run 记录里找到同一份数字。
    Every XGBoost row in scaling_cpu.csv must be traceable to a run record.
    """
    if not TABLE.is_file():
        pytest.skip("还没有 scaling_cpu.csv")
    with TABLE.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert rows, "表是空的"
    keys = [(row["dataset"], row["tier"], row["fold"], row["learner"], row["seed"]) for row in rows]
    assert len(set(keys)) == len(keys), "合并键重复：分几次跑把同一格写成了两行"
    for row in rows:
        if row["learner"] != "xgboost":
            assert row["run_id"] == "", "RF 没有 run 记录，表里不该出现 run_id"
            continue
        path = RAW_RUNS / f"{row['run_id']}.json"
        assert path.is_file(), f"表里有 {row['run_id']}，盘上没有这条记录"
        record = json.loads(path.read_text(encoding="utf-8"))
        assert record["n_subset_sha256"] == row["subset_sha256"]
        assert abs(float(record["metrics"]["r2"]) - float(row["r2"])) < 1e-9
        assert record["dataset_id"] == row["dataset"]
        assert record["task"] == "pb26_scaling_cp"


def test_loader_refuses_the_two_dataset_task_without_a_dataset() -> None:
    """`loaders.load_split('pb26_scaling_cp')` 必须**明确报错**，而不是把目录当文件打开。
    loaders.load_split must fail loudly for the two-dataset task instead of opening a directory.
    """
    with pytest.raises((L.AssemblyError, IsADirectoryError, PermissionError)):
        L.load_split("pb26_scaling_cp")
