"""C-c-1/C-c-2 · 架构迁移（arch_transfer）的守门测试（不需要 GPU / torch）。
Guard tests for C-c-1/C-c-2's architecture-transfer experiment (no GPU or torch needed).

中文：
    用途：arch 的 15 条记录能不能信，靠四件事：
          ① **臂口径是 D-49**：a2_conventional 单臂、15 runs、三种子 0/1/2，全程 CPU；
          ② **超参不另调**：`learner.params` 留空 = 从 `a2_best_params.csv` 取
             (pb26_arch_transfer_ea, xgboost) 那一行，配置里不许内联第二份超参表；
          ③ **OOD 只走官方三段式**：覆盖断言 → 每组分预测一次 → 官方
             summarize_weighted_predictions.py 合成；五折只报集成点值、不做 t 检验（D-09）；
          ④ **划分是本地物化产物**：`splits/polybench26/arch/seed_42/` 由 `arch --setup`
             从 `generated/shared_splits/` 物化（不入库），任务配置 v3（D-49 冻结）保持
             `split_ref: splits/polybench26/arch/`（目录）+ `split_status: deferred`——
             那是 A-5 校验器的契约（D-57），谁都不许改；读划分走 C-c 自己的
             `A._load_arch_split()`（口径与 `_polybench26_split` 逐行相同，`L.load_split()`
             不新增分支）；train_rows 按物化产物 [5182, 5182, 5182, 5183, 5183] 核对
             （C-b 原写折 3 为 5,182，联合核对已改正）。
    物化产物不在的机器上自动跳过对应用例（只读已落盘的产物，跑起来是秒级）。

    English:
    The 15 arch records are only trustworthy if four things hold: the arm scope is D-49 (single
    a2_conventional arm, seeds 0/1/2, CPU only), the hyper-parameters are not re-searched (an
    empty params block means the row is read from a2_best_params.csv), the OOD numbers come only
    from the official three-stage protocol with fold-level ensemble point values and no t-test,
    and the split is the locally materialised fold npz set whose train_rows match the official
    materialisation [5182, 5182, 5182, 5183, 5183]. Cases skip when artefacts are absent.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
import yaml

from chainworth.arms import graph as G
from chainworth.experiments import arch as A

REPO = Path(__file__).resolve().parents[1]
CONFIG = REPO / "configs" / "experiments" / "arch_transfer.yaml"
TASK_CONFIG = REPO / "configs" / "tasks" / "pb26_arch_transfer_ea.yaml"
SPLIT_DIR = REPO / "splits" / "polybench26" / "arch" / "seed_42"
SPLIT_MANIFEST = SPLIT_DIR / "manifest.json"
RAW_RUNS = REPO / "results" / "raw_runs"

TASK_ID = "pb26_arch_transfer_ea"
EXPECTED_TRAIN_ROWS = [5182, 5182, 5182, 5183, 5183]
EXPECTED_VAL_ROWS = [648, 648, 648, 647, 647]


def _sha256(path: Path) -> str:
    """文件字节的 sha256。/ sha256 of a file's bytes."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.fixture(scope="module")
def config() -> G.ArmConfig:
    """C-c 的冻结实验配置。/ C-c's frozen experiment config."""
    return G.load_arm_config(CONFIG)


@pytest.fixture(scope="module")
def task_config() -> dict:
    """任务配置 v3（D-49 冻结，SHA256SUMS v5）。/ The task config v3 (frozen by D-49)."""
    return yaml.safe_load(TASK_CONFIG.read_text(encoding="utf-8"))


# ---- 1. 配置与臂口径 / Config and arm scope ----


def test_config_hash_is_the_file_bytes(config: G.ArmConfig) -> None:
    """`config_hash` 必须是磁盘上这份文件的字节哈希。/ The hash must be the bytes on disk."""
    assert config.sha256 == _sha256(CONFIG)


def test_grid_is_single_arm_15_runs(config: G.ArmConfig) -> None:
    """D-49：a2_conventional 单臂、5 折 × 3 种子 = 15 runs。/ D-49: one arm, 15 runs."""
    assert config.raw["arm"] == "a2_conventional"
    assert config.raw["expected_runs"] == 15
    assert config.raw["seeds"] == [0, 1, 2]
    runs = config.raw["runs"]["polybench26"]
    assert runs["folds"] == ["fold_00", "fold_01", "fold_02", "fold_03", "fold_04"]
    assert runs["tasks"] == [TASK_ID]
    assert config.raw["device"]["prefer"] == "cpu"


def test_hyperparameters_are_not_inlined(config: G.ArmConfig) -> None:
    """超参不许在实验配置里内联第二份：留空 = 读 a2_best_params.csv（D-51）。
    No second hyper-parameter table inlined: an empty params block means the row is read
    from a2_best_params.csv.
    """
    learner = config.raw["learner"]
    assert learner["main"] == "xgboost"
    assert learner["params"] == {}


# ---- 2. eval 段协议 / The eval protocol ----


def test_eval_protocol_is_d49(config: G.ArmConfig) -> None:
    """OOD 报告口径：只报集成点值、不做 t 检验；ID 两档分开报。
    OOD reporting: ensemble point values only, no t-test; the two ID tiers reported separately.
    """
    eval_block = config.raw["eval"]
    assert eval_block["arms"] == ["a2_conventional"]
    ood = eval_block["ood"]
    assert ood["protocol"] == "official_three_stage"
    assert ood["coverage_assertion"] == "required"
    assert ood["component_prediction"] == "once_per_psmiles"
    assert ood["synthesis"] == "summarize_weighted_predictions.py", "合成只许走官方脚本（验收②）"
    assert ood["reporting"]["no_t_test"] is True, "五折 OOD 不得做 t 检验（D-09）"
    assert ood["reporting"]["ensemble_point_values"] is True
    assert eval_block["id_tiers"]["alternating"] == 611
    assert eval_block["id_tiers"]["homopolymer"] == 37
    assert eval_block["expected_ordering"] == "block > random > alternating > homopolymer"


def test_official_floors_recorded(config: G.ArmConfig) -> None:
    """官方地板按 C-b 的复现值写死（四舍五入到千分位）。/ The reproduced floors are frozen."""
    floors = config.raw["eval"]["ood"]["official_floors"]
    assert floors == {"random": 0.536, "block": 0.581, "combined": 0.559}


# ---- 3. 任务配置与划分 / Task config and the split ----


def test_task_config_split_contract_is_a5_deferred(task_config: dict) -> None:
    """任务配置 v3 的 split_ref 保持目录形态 + deferred——A-5 校验器的契约（D-57）。
    Task config v3 keeps the directory-form split_ref and deferred: the A-5 verifier's contract.
    """
    assert task_config["task_id"] == TASK_ID
    assert task_config["split_ref"] == "splits/polybench26/arch/"
    assert task_config["split_status"] == "deferred"
    assert task_config["arms"] == ["a2_conventional"]
    split_ref_path = REPO / task_config["split_ref"]
    assert split_ref_path.is_dir() and any(split_ref_path.iterdir()), (
        "deferred 的 split_ref 必须是非空目录（A-5 校验器契约）：本地物化后由 arch --setup 落位"
    )


def test_train_rows_in_config_match_materialisation(config: G.ArmConfig) -> None:
    """配置里的 train/val 行数必须与 2026-11-16 物化产物一致（C-b 原写折 3 为 5,182，已改正）。
    The config's per-fold row counts must match the materialised folds (fold 3 was corrected).
    """
    folds = config.raw["data"]["folds"]
    assert folds["train_rows"] == EXPECTED_TRAIN_ROWS
    assert folds["val_rows"] == EXPECTED_VAL_ROWS
    assert folds["test_rows"] == [648] * 5


@pytest.mark.skipif(not SPLIT_MANIFEST.exists(), reason="本地未物化：先跑 scripts/04_run_cpu.py arch --setup")
def test_materialised_manifest_row_counts() -> None:
    """物化 manifest 的每折行数与配置一致。/ The materialised manifest matches the config."""
    manifest = json.loads(SPLIT_MANIFEST.read_text(encoding="utf-8"))
    assert manifest["task_id"] == TASK_ID
    train = [entry["n_train_pool"] for entry in manifest["folds"]]
    val = [entry["n_val"] for entry in manifest["folds"]]
    assert train == EXPECTED_TRAIN_ROWS
    assert val == EXPECTED_VAL_ROWS


@pytest.mark.skipif(not SPLIT_MANIFEST.exists(), reason="本地未物化：先跑 scripts/04_run_cpu.py arch --setup")
def test_load_split_resolves_via_arch_loader() -> None:
    """`A._load_arch_split()` 读物化产物，折数与每折行数正确（D-57：不走 L.load_split）。
    A._load_arch_split() resolves the materialised split with no L.load_split() branch.
    """
    split = A._load_arch_split()
    assert [name for name in split.fold_names] == [
        "fold_00",
        "fold_01",
        "fold_02",
        "fold_03",
        "fold_04",
    ]
    for fold, n_train, n_val in zip(split.folds, EXPECTED_TRAIN_ROWS, EXPECTED_VAL_ROWS):
        assert int(fold.train_index.size) == n_train
        assert int(fold.val_index.size) == n_val
        assert int(fold.test_index.size) == 648


# ---- 4. 已落盘的 15 条记录 / The 15 records on disk ----


def _arch_records() -> list[dict]:
    """磁盘上 arch 的 ok 记录。/ The arch records on disk."""
    records = []
    for path in sorted(RAW_RUNS.glob("*.json")):
        if path.name == "_schema.json":
            continue
        try:
            record = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:  # pragma: no cover - 半截文件
            continue
        if record.get("task") == TASK_ID and record.get("status") == "ok":
            records.append(record)
    return records


@pytest.mark.skipif(not SPLIT_MANIFEST.exists(), reason="本地未物化：先跑 scripts/04_run_cpu.py arch --setup")
def test_records_share_the_frozen_config_hash(config: G.ArmConfig) -> None:
    """15 条记录的 config_hash 逐条等于 arch_transfer.yaml 的字节哈希。
    Every arch record's config_hash equals this file's byte hash.
    """
    records = _arch_records()
    assert len(records) == 15
    for record in records:
        assert record["config_hash"] == f"sha256:{config.sha256}"


@pytest.mark.skipif(not SPLIT_MANIFEST.exists(), reason="本地未物化：先跑 scripts/04_run_cpu.py arch --setup")
def test_records_carry_id_tier_metrics() -> None:
    """每条记录都带 ID 两档分开的指标与条数（验收①：不许合并成一个 ID）。
    Each record carries the two ID tiers separately (never merged), with counts alongside.
    """
    for record in _arch_records():
        tiers = record["id_tier_metrics"]
        counts = record["id_tier_n"]
        assert set(tiers) == {"alternating", "homopolymer"}
        assert counts == {"alternating": 611, "homopolymer": 37}
        for tier in ("alternating", "homopolymer"):
            for metric in ("rmse", "mae", "r2"):
                assert metric in tiers[tier]
