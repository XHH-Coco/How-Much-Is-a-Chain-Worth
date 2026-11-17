"""D-69：链臂 scaling 的守门测试——配置不许漂、两臂必须落在同一批行、表不许混。

Gate tests for D-69's chain arm: the config may not drift, both arms must sit on the same training
rows, and the two tables may not be mixed.

中文：这组测试盯的是**四条会静默错的地方**：
  ① `scaling_chain.yaml` 的超参与 `a3_graph.yaml` 漂了 → 那就不叫"A3 口径"了；
  ② 有人往 `scaling.yaml` 里加东西 → 40 条 A2 记录的 `config_hash` 当场作废（不报错，只是被
     `index` 剔除）。所以这里现算那份文件的哈希，与记录里的比；
  ③ 两臂的训练行指纹不同 → Λ 里混进协议差异；
  ④ 链臂记录被写进 `scaling_cpu.csv` → "一行一臂"的读法失效（D-34/D-66 的同款坑）。
English: four ways this could go silently wrong: the hyper-parameters drifting away from A3's
frozen file; someone editing `scaling.yaml` (which would invalidate 40 A2 records without an
error); the two arms landing on different training rows; and chain records leaking into A2's table.
"""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
CHAIN_CONFIG = REPO_ROOT / "configs" / "experiments" / "scaling_chain.yaml"
UNIT_CONFIG = REPO_ROOT / "configs" / "experiments" / "scaling.yaml"
ARM_CONFIG = REPO_ROOT / "configs" / "arms" / "a3_graph.yaml"
TASK_CONFIG = REPO_ROOT / "configs" / "tasks" / "pb26_scaling_cp.yaml"
RAW_RUNS = REPO_ROOT / "results" / "raw_runs"
UNIT_TABLE = REPO_ROOT / "results" / "aggregate" / "scaling_cpu.csv"
CHAIN_TABLE = REPO_ROOT / "results" / "aggregate" / "scaling_chain.csv"
SPLITS = REPO_ROOT / "splits" / "polybench26" / "pb26_scaling_cp"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load(path: Path) -> dict:
    return yaml.safe_load(path.read_bytes())


def _records(arm: str) -> list[dict]:
    """该臂在 pb26_scaling_cp 上的全部记录。／ Every record of one arm on this task."""
    records = []
    for path in sorted(RAW_RUNS.glob(f"pb26_scaling_cp__{arm}__*.json")):
        records.append(json.loads(path.read_text(encoding="utf-8")))
    return records


def test_hyperparameters_match_the_frozen_a3_arm() -> None:
    """链臂配置的架构/训练/设备三节必须与 `a3_graph.yaml` **逐键相同**。
    Architecture, training and device must match `a3_graph.yaml` key by key.

    中文：抄一份超参是必要的（`train_one()` 只读一份配置），但抄写会漂。这条把漂移变成红灯：
          改 `a3_graph.yaml` 而不改这里，或者反过来，都会失败。
    English: copying is necessary (train_one reads one config) but copying drifts; this turns the
    drift into a red test in either direction.
    """
    chain = _load(CHAIN_CONFIG)
    arm = _load(ARM_CONFIG)
    for section in ("architecture", "training", "device"):
        assert chain[section] == arm[section], (
            f"{section} 与 a3_graph.yaml 不一致："
            f"{set(chain[section].items()) ^ set(arm[section].items())}"
        )


def test_identity_fields_of_the_chain_config() -> None:
    """臂名、臂枚举、档位取法、种子、期望条数四样写死在配置里。/ The four identity fields."""
    chain = _load(CHAIN_CONFIG)
    assert chain["arm"] == "a3_graph"
    assert chain["subset"]["rule"] == "prefix", "官方档位是 train_pool[:N] 前缀，不是打乱后取前缀"
    assert chain["subset"]["seed"] == 0
    assert chain["seeds"] == [0], "单元臂只有 seed 0，链臂跑多一档就不是配对比较"
    assert chain["expected_fits"] == 40, "8 个(数据集,档位) × 5 折 = 40"


def test_task_config_declares_the_chain_arm() -> None:
    """任务身份文件必须把 a3_graph 写进 `arms`，且版本已升。/ The task config must declare it."""
    task = _load(TASK_CONFIG)
    assert "a2_conventional" in task["arms"]
    assert "a3_graph" in task["arms"], "κ 的纵轴需要链臂：arms 里没有它，记录就落在没声明的组合上"
    header = TASK_CONFIG.read_text(encoding="utf-8").splitlines()[0]
    assert "FROZEN v2" in header, "加臂必须升版本，否则改前改后分不出来"


def test_unit_config_is_untouched_by_this_decision() -> None:
    """`scaling.yaml` 的字节哈希必须仍等于那 40 条 A2 记录的 `config_hash`。
    scaling.yaml's digest must still equal the config_hash of the 40 A2 records.

    中文：这是本次改动最容易踩坏的地方——往 `scaling.yaml` 里加一节链臂，"看起来"很自然，
          但那 40 条记录会被 `index` 判成旧协议并**静默剔除**，κ 的分母当场消失。
          所以这条测试不查"文件有没有变"，而是查"记录的哈希是否仍指向这一份文件的字节"。
    English: the trap here is adding a chain section to scaling.yaml, which would silently drop the
    40 unit-arm records. So this asserts the records' hash still points at those exact bytes - not
    merely that the file "looks unchanged".
    """
    digest = _sha256(UNIT_CONFIG)
    records = _records("a2_conventional")
    assert records, "一条 A2 scaling 记录都没有：先跑 `04_run_cpu.py scaling`"
    mismatched = [r["run_id"] for r in records if r["config_hash"] != f"sha256:{digest}"]
    assert not mismatched, (
        f"{len(mismatched)} 条 A2 记录的 config_hash 不再指向 scaling.yaml 的字节"
        f"（例：{mismatched[:2]}）。改这份文件 = 40 条记录作废"
    )


def test_chain_records_carry_the_chain_config_hash() -> None:
    """链臂记录的 `config_hash` 指向 `scaling_chain.yaml`，且 `arm` 正确。
    Chain records must point at the chain config's digest.
    """
    digest = _sha256(CHAIN_CONFIG)
    records = _records("a3_graph")
    if not records:
        pytest.skip("链臂还没跑：先跑 `python scripts/04_run_cpu.py scaling --arm a3_graph`")
    problems = [
        (r["run_id"], r.get("config_hash"), r.get("arm"))
        for r in records
        if r.get("config_hash") != f"sha256:{digest}" or r.get("arm") != "a3_graph"
    ]
    assert not problems, f"{len(problems)} 条链臂记录的 config_hash/arm 不对：{problems[:3]}"


def test_both_arms_sit_on_the_same_training_rows() -> None:
    """同一 (数据集,档位,折,种子) 上，两臂的 `n_subset_sha256` 必须逐位相同。
    Both arms must share the same training-row digest cell by cell.

    中文：这是 Λ 可相减的前提。官方前缀规则由 `graph.n_level_rows(rule="prefix")` 一份实现执行，
          所以这条同时验证"链臂没有偷偷换成打乱取前缀"。
    English: the precondition for subtracting the arms; it also proves the chain arm did not
    silently switch to shuffle-then-prefix.
    """
    unit = {r["run_id"]: r for r in _records("a2_conventional")}
    chain = _records("a3_graph")
    if not chain:
        pytest.skip("链臂还没跑")
    compared = 0
    problems = []
    for record in chain:
        # run_id 形状：<task>__<arm>__<fold>__seed<N>，而 run_id 里折段已含档位（_n300）。
        # 两条臂的记录按 (split_fold, n_level, seed) 配对，不按 run_id（臂名不同）。
        key = (record.get("split_fold"), str(record.get("n_level")), int(record["seed"]))
        match = [
            r
            for r in unit.values()
            if (r.get("split_fold"), str(r.get("n_level")), int(r["seed"])) == key
        ]
        assert len(match) == 1, f"{key} 在单元臂里有 {len(match)} 条记录"
        compared += 1
        if match[0]["n_subset_sha256"] != record["n_subset_sha256"]:
            problems.append((key, match[0]["n_subset_sha256"][-12:], record["n_subset_sha256"][-12:]))
    assert compared >= 1
    assert not problems, f"{len(problems)} 格两臂训练行不同：{problems[:3]}"


def test_tier_is_the_official_prefix() -> None:
    """档位的训练行必须等于官方 `train_pool[:N]`（按指纹核，不按行数猜）。
    The tier's rows must be the official prefix, checked by digest rather than by row count.
    """
    unit = [r for r in _records("a2_conventional")]
    if not unit:
        pytest.skip("单元臂还没跑")
    checked = 0
    for record in unit:
        dataset = record.get("dataset_id")
        level = str(record.get("n_level"))
        fold = record.get("split_fold")
        if not dataset or not fold:
            continue
        base = fold.split("_fold_")[0]
        directory = SPLITS / base / "seed_42"
        fold_file = directory / f"fold_{fold.split('_')[-1]}.npz"
        if not fold_file.is_file():
            pytest.skip(f"缺折文件 {fold_file.name}：A-5 的快照不完整")
        import numpy as np

        with np.load(fold_file) as payload:
            pool = payload["train_pool"].astype(np.int64)
        used = pool.size if level == "all" else min(int(level), pool.size)
        expected = hashlib.sha256(np.sort(pool[:used]).astype("<i8").tobytes()).hexdigest()
        assert record["n_subset_sha256"] == f"sha256:{expected}", (
            f"{dataset}/{fold}/n={level}：训练行不是官方前缀"
        )
        checked += 1
    assert checked >= 8, f"只核到 {checked} 格，不像在核全量"


def test_chain_records_are_not_in_the_unit_table() -> None:
    """`scaling_cpu.csv` 里不许出现 a3_graph 的行（两臂分表，D-69）。
    A2's table must not contain chain-arm rows.
    """
    with UNIT_TABLE.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    arms = {row.get("arm") for row in rows}
    assert "a3_graph" not in arms, "链臂混进了 A2 的表：两张表必须分开，否则'一行一臂'失效"
    assert arms == {"a2_conventional"}


def test_chain_table_shape_matches_the_grid() -> None:
    """链臂表 40 行、8 个(数据集,档位)、每格 5 折。／ The chain table has 40 rows, 5 folds per cell."""
    if not CHAIN_TABLE.is_file():
        pytest.skip("链臂表还没生成")
    with CHAIN_TABLE.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 40, f"应该是 8 个(数据集,档位) × 5 折 = 40 行，现在是 {len(rows)}"
    per_cell: dict[tuple[str, str], int] = {}
    for row in rows:
        key = (row["dataset"], row["tier"])
        per_cell[key] = per_cell.get(key, 0) + 1
    assert len(per_cell) == 8, f"档位数不对：{sorted(per_cell)}"
    assert set(per_cell.values()) == {5}, f"每格必须 5 折：{per_cell}"
    assert all(row["arm"] == "a3_graph" for row in rows)
    assert all(row["run_id"] for row in rows), "链臂每一行都必须有 run 记录"
