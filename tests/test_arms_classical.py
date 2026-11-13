"""C-a 的断言测试：主表口径、A5 与 A2 逐键相同、确定性、记录合规、预测行数。
Assertions for C-a: the main-table convention, A5 being key-for-key identical to A2, determinism,
record validity, and prediction row counts.

中文：
    用途：把分工_C-a 的验收标准变成 `pytest`，四组：
          ① **纯函数（不需要数据）**：
             - `main_table.yaml` 的 `expected_runs` = 16 单元 × 2 臂 × 3 种子 = 96，且分臂 48 各一
               （A5 撤除，见 D-48）；
             - A5 的 `learner` / `search` / `seeds` / `runs` 与 A2 **逐键相同**（C-a-3 验收①：
               "同学习器同超参"是机器判的，不是肉眼比对的）；
             - A1/A2 的网格各 48 条，臂名与 run_id 形状符合 `_schema.json`。
          ② **需要装配产物（core 环境，缺则跳过）**：
             - A2 同 seed 跑两次，`r2` 差 < 1e-9 且预测逐位相同（C-a-1 验收④）；
             - A1 三个种子的预测完全相同（均值没有随机性）；
             - 记录过 `results/raw_runs/_schema.json`（jsonschema，不是肉眼）；
             - 预测行数 = 该折 `test_index` 的长度，`row_id` 逐位等于官方测试行。
    依赖：`configs/arms/*.yaml`、`configs/experiments/main_table.yaml`（入库）；
          `data/processed/features/*.npz`（不入库，缺则跳过第二组）。
    配置：无环境变量。跑法：`pytest tests/test_arms_classical.py`（core 环境）。
    被谁用：CI、任何改 `arms/conventional.py` / `arms/floor.py` / 三份臂配置的人。

English:
    Purpose: turn division C-a's acceptance criteria into pytest cases in two groups: (1) pure
    functions - the main table is 16 x 2 x 3 = 96 with 48 per arm (A5 withdrawn, D-48), A5's
    learner/search/seeds/runs are key-for-key identical to A2, and both grids hold 48 runs with
    schema-shaped ids; (2) with
    the assembly pack present (skipped otherwise) - A2 is bit-identical across two same-seed runs,
    A1's three seeds agree exactly, records validate against the frozen JSON schema, and prediction
    lengths and row ids match the official test index.
    Depends on: the committed configs; the gitignored feature packs (tests skip without them).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import yaml

from chainworth.arms import conventional as C
from chainworth.arms import floor as FL
from chainworth.arms import graph as G
from chainworth.io import loaders as L

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIGS = REPO_ROOT / "configs" / "arms"
SMALL_TASK = "pb_t3_iwlt"          # 228 行：测试里唯一允许真跑的任务，秒级


def _config(name: str) -> dict:
    """读一份臂配置（yaml.safe_load）。／ Read an arm config."""
    return yaml.safe_load((CONFIGS / f"{name}.yaml").read_text(encoding="utf-8"))


def _data_available(task: str) -> bool:
    """装配产物在不在。／ Whether the assembly artefact exists."""
    return L.feature_pack_path(task).is_file()


# ---- ① 纯函数 / Pure functions ----


def test_main_table_expected_runs_is_96() -> None:
    """主表口径：16 单元 × 2 臂 × 3 种子 = 96，且两臂各 48；A5 只作为"不可构造"的排除项出现。
    The main table is 16 units x 2 arms x 3 seeds = 96, with 48 per arm; A5 appears only as an
    excluded, not-constructible arm (D-48).
    """
    payload = yaml.safe_load(
        (REPO_ROOT / "configs" / "experiments" / "main_table.yaml").read_text(encoding="utf-8")
    )
    units = payload["units"]
    n_units = units["polybench"]["subtotal"] + units["polybench26"]["subtotal"]
    assert n_units == units["total"] == 16
    arms = payload["arms"]
    seeds = payload["seeds"]
    assert sorted(arms) == ["a1_floor", "a2_conventional"] and len(seeds) == 3
    assert "a5_chain" not in arms
    excluded = payload["excluded_arms"]["a5_chain"]
    assert excluded["in_main_table"] is False and excluded["runs"] == 0
    # 措辞是口径的一部分：A5 是"没运行过"，不是"增益为零"。
    assert excluded["missing_reason"] == "not_constructible_data_unavailable"
    expected = payload["expected_runs"]
    assert expected["per_arm"] == n_units * len(seeds) == 48
    assert expected["total"] == n_units * len(arms) * len(seeds) == 96
    assert sum(expected["breakdown"].values()) == expected["total"]


# 8 个 BCUT2D 描述符在每一个 pack 上都是 100% NaN（D-48）：RDKit 对重复单元边界的
# `*` 哑原子算 BCUT2D 直接返回 NaN。名单一变就红——这是换特征版本时的保险。
BCUT2D_ALL_NAN = {
    "BCUT2D_MWHI",
    "BCUT2D_MWLOW",
    "BCUT2D_CHGHI",
    "BCUT2D_CHGLO",
    "BCUT2D_LOGPHI",
    "BCUT2D_LOGPLOW",
    "BCUT2D_MRHI",
    "BCUT2D_MRLOW",
}


def test_bcut2d_columns_are_all_nan_in_every_pack() -> None:
    """每个 pack 里"整列全 NaN"的描述符必须恰好是这 8 个 BCUT2D（D-48）。
    In every pack, the set of all-NaN descriptor columns must be exactly these eight.
    """
    packs = sorted((REPO_ROOT / "data" / "processed" / "features").glob("*.npz"))
    if not packs:
        pytest.skip("缺装配产物：data/processed/features/ 下一个 pack 都没有")
    for path in packs:
        payload = np.load(path, allow_pickle=True)
        desc = payload["desc"]
        columns = [str(c) for c in payload["desc_columns"]]
        assert desc.shape[1] == 205, f"{path.name}: 描述符列数 {desc.shape[1]}，应为 205"
        all_nan = {
            columns[i] for i in range(len(columns)) if np.isnan(desc[:, i]).all()
        }
        assert all_nan == BCUT2D_ALL_NAN, (
            f"{path.name}: 全 NaN 列是 {sorted(all_nan)}，与 D-48 的名单不符"
        )


def test_a5_reuses_a2_learner_and_search() -> None:
    """A5 与 A2 的 learner / search / seeds / runs 逐键相同（A5 只许改输入）。
    A5's learner, search, seeds and runs are key-for-key identical to A2's.
    """
    a2 = _config("a2_conventional")
    a5 = _config("a5_chain")
    assert a5["learner"] == a2["learner"]
    assert a5["search"] == a2["search"]
    assert a5["seeds"] == a2["seeds"]
    assert a5["runs"] == a2["runs"]
    assert a5["derivation"]["base"] == "configs/arms/a2_conventional.yaml"


@pytest.mark.parametrize("name,arm", [("a1_floor", "a1_floor"), ("a2_conventional", "a2_conventional")])
def test_grid_is_48_and_ids_follow_schema(name: str, arm: str) -> None:
    """两个臂各 48 条，臂名正确，run_id 符合 schema 的三段 + seed 形状。
    Each arm holds 48 runs with the right arm name and a schema-shaped run id.
    """
    config = G.load_arm_config(CONFIGS / f"{name}.yaml")
    keys = G.enumerate_runs(config)
    assert len(keys) == 48
    assert {key.arm for key in keys} == {arm}
    pattern = __import__("re").compile(r"^[a-z0-9_]+__[a-z0-9_]+__[a-z0-9_]+__seed[0-9]+$")
    for key in keys:
        assert pattern.match(key.run_id())


def test_a2_and_a1_configs_are_frozen_with_a_byte_hash() -> None:
    """三份配置都能被 A3 的加载器读完，哈希 = 文件字节哈希（改一个空格就换一条臂）。
    All three configs load through A3's loader and their hash is the file's byte hash.
    """
    for name in ("a1_floor", "a2_conventional", "a5_chain"):
        config = G.load_arm_config(CONFIGS / f"{name}.yaml")
        payload = (CONFIGS / f"{name}.yaml").read_bytes()
        assert config.sha256 == __import__("hashlib").sha256(payload).hexdigest()


# ---- ② 需要装配产物 / Needs the assembly pack ----


@pytest.fixture(scope="module")
def small_data() -> C.TaskData:
    """小任务的 TaskData（228 行）；缺产物直接跳过。／ The small task's data, or skip."""
    if not _data_available(SMALL_TASK):
        pytest.skip(f"缺装配产物 {L.feature_pack_path(SMALL_TASK)}")
    return C.build_task_data(SMALL_TASK)


@pytest.fixture(scope="module")
def a2_config() -> G.ArmConfig:
    """A2 的冻结配置。／ A2's frozen config."""
    return G.load_arm_config(CONFIGS / "a2_conventional.yaml")


def _has_params(task: str) -> bool:
    """超参表里有没有这个任务。／ Whether the task already has frozen hyper-parameters."""
    return (task, "xgboost") in C._read_best_params(C.BEST_PARAMS_PATH)


def test_a2_is_bit_identical_for_the_same_seed(small_data: C.TaskData, a2_config: G.ArmConfig) -> None:
    """同 seed 跑两次：r2 差 < 1e-9，且预测逐位相同（C-a-1 验收④）。
    Same seed twice: r2 within 1e-9 and predictions bit-identical.
    """
    if not _has_params(SMALL_TASK):
        pytest.skip(f"{SMALL_TASK} 还没有超参：先跑 04_run_cpu.py search")
    first = C.train_one(small_data, a2_config, fold="official_70_30", seed=0)
    second = C.train_one(small_data, a2_config, fold="official_70_30", seed=0)
    assert first.record["metrics"]["r2"] == pytest.approx(second.record["metrics"]["r2"], abs=1e-9)
    assert first.record["n_test"] == second.record["n_test"]
    # 逐位相同比"差 < 1e-9"更严：确定性模型不该有任何抖动。
    assert np.array_equal(first.y_pred, second.y_pred)


def test_floor_seeds_are_identical(small_data: C.TaskData) -> None:
    """A1 的三个种子给出完全相同的预测（训练均值没有随机性）。
    A1's three seeds produce exactly the same predictions: the training mean has no randomness.
    """
    config = G.load_arm_config(CONFIGS / "a1_floor.yaml")
    data = FL.build_task_data(SMALL_TASK)
    outcomes = [FL.train_one(data, config, fold="official_70_30", seed=seed) for seed in (0, 1, 2)]
    base = outcomes[0].y_pred
    for outcome in outcomes[1:]:
        assert np.array_equal(base, outcome.y_pred)
        assert outcome.record["metrics"] == outcomes[0].record["metrics"]


def test_record_validates_against_the_schema(
    small_data: C.TaskData, a2_config: G.ArmConfig, tmp_path: Path
) -> None:
    """记录必须过 `results/raw_runs/_schema.json`（jsonschema，不是肉眼），且能落盘。
    The record must pass the frozen JSON schema (jsonschema, not eyeballing) and write to disk.
    """
    import jsonschema

    if not _has_params(SMALL_TASK):
        pytest.skip(f"{SMALL_TASK} 还没有超参：先跑 04_run_cpu.py search")
    outcome = C.train_one(small_data, a2_config, fold="official_70_30", seed=0)
    schema = json.loads((REPO_ROOT / "results" / "raw_runs" / "_schema.json").read_text(encoding="utf-8"))
    jsonschema.validate(outcome.record, schema)
    path = G.write_run_record(outcome.record, directory=tmp_path)
    assert path.name == f"{outcome.record['run_id']}.json"
    assert json.loads(path.read_text(encoding="utf-8"))["run_id"] == outcome.record["run_id"]


def test_predictions_cover_the_official_test_rows(
    small_data: C.TaskData, a2_config: G.ArmConfig
) -> None:
    """预测行数 = 该折 `test_index` 的长度，`row_id` 逐位等于官方测试行（验收①②）。
    The prediction length equals the fold's test_index length and the row ids match it exactly.
    """
    if not _has_params(SMALL_TASK):
        pytest.skip(f"{SMALL_TASK} 还没有超参：先跑 04_run_cpu.py search")
    outcome = C.train_one(small_data, a2_config, fold="official_70_30", seed=0)
    spec = small_data.split.folds[0]
    assert outcome.row_id.size == spec.test_index.size
    assert np.array_equal(outcome.row_id, np.asarray(spec.test_index))


def test_search_folds_stay_inside_the_training_rows(small_data: C.TaskData) -> None:
    """搜索用的分组折只覆盖第一折训练行：互斥、无遗漏、每个骨架只落一折、测试行永不进入。
    The search folds cover the first fold's training rows only: disjoint, exhaustive, one fold
    per Murcko scaffold, and the official test rows never enter.
    """
    fold = small_data.split.folds[0]
    train_index = np.asarray(fold.train_index)
    groups = C.murcko_groups(small_data.psmiles)[train_index]
    folds = C._grouped_folds(groups, min(5, len(set(groups.tolist()))))

    stacked = np.concatenate(folds)
    assert np.array_equal(np.sort(stacked), np.arange(train_index.size)), "折没有铺满训练行且不重复"
    seen: set[str] = set()
    for val in folds:
        labels = set(groups[val].tolist())
        assert not (labels & seen), "同一个骨架跨了两折：验证折会看到训练侧见过的骨架"
        seen |= labels
    assert seen == set(groups.tolist())

    # 官方测试行与训练行不相交，且 `_grouped_folds` 的输入只有训练行的组标签。
    assert not (set(train_index.tolist()) & set(np.asarray(fold.test_index).tolist()))


# ---- ③ 附加表（敏感性 / 受控对照）的纯函数 / Pure helpers of the two extra tables ----


def _entry04():
    """把 `scripts/04_run_cpu.py` 当模块加载（文件名以数字开头，不能用 `import`）。
    Load the entry script as a module: its name starts with a digit, so plain `import` fails.
    """
    import importlib.util

    path = REPO_ROOT / "scripts" / "04_run_cpu.py"
    spec = importlib.util.spec_from_file_location("entry04_run_cpu", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_sensitivity_and_control_copy_the_a2_grid() -> None:
    """两张附加表都抄 A2 的网格：8 个任务，6 个一折、2 个五折，任务不重不漏。
    Both extra tables copy A2's grid: 8 tasks, 6 with one fold and 2 with five, no duplicates.
    """
    grid = _entry04()._task_grid(_config("a2_conventional"))
    names = [task for task, _ in grid]
    assert len(names) == len(set(names)) == 8
    by_task = dict(grid)
    assert sum(1 for folds in by_task.values() if len(folds) == 1) == 6
    assert by_task["pb26_polymetrix_tg"] == [f"fold_0{i}" for i in range(5)]
    assert by_task["pb26_homopolymer_density"] == [f"fold_0{i}" for i in range(5)]
    assert by_task["pb_t1_tg"] == ["official_70_30"]


def test_control_variants_change_one_factor_each() -> None:
    """每个变体只改 `changes` 里写的那几项，且 `features` 与 `changes` 不许互相矛盾。
    Every variant changes only what `changes` names, and `features` never contradicts `changes`.
    """
    control = yaml.safe_load(
        (REPO_ROOT / "configs" / "experiments" / "pb26_control.yaml").read_text(encoding="utf-8")
    )
    variants = control["variants"]
    assert variants["baseline"]["changes"] == []
    assert set(variants["official_recipe"]["changes"]) == {"train_rows", "features", "learner"}
    for name, spec in variants.items():
        changes = set(spec.get("changes") or [])
        assert changes <= {"train_rows", "features", "learner"}, name
        assert ("features" in changes) == (spec["features"] == "descriptors_only"), name
        assert len(changes) <= 1 or name == "official_recipe", f"{name} 一次改了多项"
    assert set(control["official_train_size"]) == {"pb26_polymetrix_tg", "pb26_homopolymer_density"}


def test_control_view_truncates_the_training_prefix() -> None:
    """受控变体在数据上的两处改动：训练行取前 N 行、特征只留描述符；测试行一行不动。
    The two data changes: training rows become the first N, features keep the descriptors only;
    the test rows are untouched.
    """
    if not _data_available(SMALL_TASK):
        pytest.skip(f"缺装配产物 {L.feature_pack_path(SMALL_TASK)}")
    entry = _entry04()
    data = C.build_task_data(SMALL_TASK)
    fold = data.split.folds[0]
    view = entry._control_view(
        data,
        {"changes": ["train_rows", "features"], "features": "descriptors_only"},
        n_descriptors=205,
        train_rows=100,
    )
    assert view.X.shape[1] == 205
    # `equal_nan=True`：描述符里有缺失，NaN != NaN 会让逐位比较假失败。
    assert np.array_equal(view.X, data.X[:, :205], equal_nan=True)
    assert view.split.folds[0].train_index.size == 100
    assert np.array_equal(view.split.folds[0].train_index, np.asarray(fold.train_index)[:100])
    assert np.array_equal(view.split.folds[0].test_index, np.asarray(fold.test_index))
    # 原对象不被改动：受控对照的每次训练都必须从同一份原始数据出发。
    assert data.X.shape[1] == 2253
    assert np.asarray(fold.train_index).size > 100


def test_control_view_rejects_contradictory_variants() -> None:
    """`changes` 说改特征、`features` 却说全量：当场报错，不猜。"""
    if not _data_available(SMALL_TASK):
        pytest.skip(f"缺装配产物 {L.feature_pack_path(SMALL_TASK)}")
    entry = _entry04()
    data = C.build_task_data(SMALL_TASK)
    with pytest.raises(SystemExit):
        entry._control_view(data, {"changes": ["features"], "features": "all"}, n_descriptors=205, train_rows=None)
    with pytest.raises(SystemExit):
        entry._control_view(data, {"changes": ["train_rows"], "features": "all"}, n_descriptors=205, train_rows=None)


# ---- 附加表：受控对照与学习器敏感性（读 CSV，不重训）----
# Auxiliary tables: the controlled control and the learner sensitivity (CSV reads, no retraining).

CONTROL_CSV = REPO_ROOT / "results" / "aggregate" / "pb26_controlled_control.csv"
SENS_CSV = REPO_ROOT / "results" / "aggregate" / "learner_sensitivity.csv"


def test_control_baseline_reproduces_a2_seed0() -> None:
    """`baseline` 不减任何一项，所以它必须与 A2 的 seed-0 记录逐折相等——这是对照链的自证。
    `baseline` changes nothing, so each fold must equal A2's seed-0 record: the control's self-proof.
    """
    if not CONTROL_CSV.is_file():
        pytest.skip("还没有 pb26_controlled_control.csv")
    import csv

    baseline = {}
    with CONTROL_CSV.open(encoding="utf-8", newline="") as fh:
        for row in csv.DictReader(fh):
            if row["variant"] == "baseline":
                baseline[(row["task"], row["fold"], int(row["seed"]))] = float(row["rmse"])
    assert len(baseline) == 10, "两个任务 × 5 折 × seed0"

    checked = 0
    for (task, fold, seed), rmse in baseline.items():
        path = REPO_ROOT / "results" / "raw_runs" / f"{task}__a2_conventional__{fold}__seed{seed}.json"
        record = json.loads(path.read_text(encoding="utf-8"))
        assert record["split_fold"] == fold
        assert abs(rmse - float(record["metrics"]["rmse"])) < 1e-9, f"{task}/{fold} 与 A2 记录不一致"
        checked += 1
    assert checked == 10


def test_control_official_recipe_reproduces_the_official_rmse() -> None:
    """三项全改 = 官方配方；其折间均值必须落在官方 RMSE 的 1% 以内（否则对比链有问题）。
    The full recipe must land within 1% of the official RMSE, or the comparison chain is broken.
    """
    summary_path = REPO_ROOT / "results" / "aggregate" / "pb26_controlled_control_summary.json"
    if not summary_path.is_file():
        pytest.skip("还没有 pb26_controlled_control_summary.json")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    for task, info in summary["tasks"].items():
        official = info["official_rmse_mean_across_folds"]
        residual = abs(info["official_recipe_residual"])
        assert residual / official < 0.01, f"{task}: 残差 {residual:.6g} 超过 1%"
        assert info["variants"]["official_recipe"]["n_features"] == 205
        assert info["variants"]["official_recipe"]["n_train"] == info["official_train_size"]


def test_sensitivity_table_covers_the_main_grid() -> None:
    """敏感性表必须与主表同一网格：8 任务 × 2 学习器 × 3 种子 = 96 行（PB26 每种子 5 折）。
    Same grid as the main table: 8 tasks x 2 learners x 3 seeds = 96 rows.
    """
    if not SENS_CSV.is_file():
        pytest.skip("还没有 learner_sensitivity.csv")
    import csv

    with SENS_CSV.open(encoding="utf-8", newline="") as fh:
        rows = list(csv.DictReader(fh))
    assert len(rows) == 96, f"期望 96 行，实际 {len(rows)}"
    a2 = yaml.safe_load((CONFIGS / "a2_conventional.yaml").read_text(encoding="utf-8"))
    poly, pb26 = a2["runs"]["polybench"], a2["runs"]["polybench26"]
    units = len(poly["tasks"]) * len(poly["folds"]) + len(pb26["tasks"]) * len(pb26["folds"])
    assert units == 16, "6 个 PolyBench 单折 + 2 个 PB26 五折"
    for learner in ("rf", "knn"):
        cells = [r for r in rows if r["learner"] == learner]
        assert len(cells) == units * 3, learner
        assert all(r["hyperparams"] for r in cells)
        assert all(float(r["wall_clock_s"]) >= 0.0 for r in cells)
    per_poly = [r for r in rows if r["task"] in poly["tasks"]]
    per_pb26 = [r for r in rows if r["task"] in pb26["tasks"]]
    assert len(per_poly) == 6 * 1 * 3 * 2
    assert len(per_pb26) == 2 * 5 * 3 * 2


NULL_CSV = REPO_ROOT / "results" / "aggregate" / "null_calibration.csv"
NULL_JSON = REPO_ROOT / "results" / "aggregate" / "null_calibration_summary.json"


def test_null_calibration_baseline_reproduces_a2_seed0() -> None:
    """零信息对照的 `baseline` 必须与 A2 的 seed-0 记录逐折相等——否则地板不是同一把尺。
    The null calibration's baseline must equal A2's seed-0 record per fold, or the floor is
    measured with a different ruler.
    """
    if not NULL_CSV.is_file():
        pytest.skip("还没有 null_calibration.csv")
    import csv

    baseline = {}
    with NULL_CSV.open(encoding="utf-8", newline="") as fh:
        for row in csv.DictReader(fh):
            if row["condition"] == "baseline":
                baseline[(row["task"], row["fold"], int(row["seed"]))] = float(row["r2"])
    assert len(baseline) == 6, "pb_t1_tg 单折 + pb26_polymetrix_tg 五折"
    for (task, fold, seed), r2 in baseline.items():
        path = REPO_ROOT / "results" / "raw_runs" / f"{task}__a2_conventional__{fold}__seed{seed}.json"
        record = json.loads(path.read_text(encoding="utf-8"))
        assert abs(r2 - float(record["metrics"]["r2"])) < 1e-9, f"{task}/{fold} 与 A2 记录不一致"


def test_null_calibration_covers_six_units_and_three_conditions() -> None:
    """6 单元 × 3 条件 = 18 行；列数 2253 / 2256；置换列必须能追到具体描述符下标。
    6 units x 3 conditions = 18 rows; 2253 / 2256 columns; shuffled columns name their indices.
    """
    if not NULL_CSV.is_file():
        pytest.skip("还没有 null_calibration.csv")
    import csv

    with NULL_CSV.open(encoding="utf-8", newline="") as fh:
        rows = list(csv.DictReader(fh))
    assert len(rows) == 18, f"期望 18 行，实际 {len(rows)}"
    base = [r for r in rows if r["condition"] == "baseline"]
    const = [r for r in rows if r["condition"] == "constant3"]
    shuffled = [r for r in rows if r["condition"] == "shuffled3"]
    for cells in (base, const, shuffled):
        assert len(cells) == 6
        assert len({(r["task"], r["fold"]) for r in cells}) == 6
    assert all(int(r["n_features"]) == 2253 and int(r["added_columns"]) == 0 for r in base)
    assert all(int(r["n_features"]) == 2256 and int(r["added_columns"]) == 3 for r in const)
    assert all(int(r["n_features"]) == 2256 and int(r["added_columns"]) == 3 for r in shuffled)
    for row in shuffled:
        assert row["source"].startswith("descriptors:")
        picked = [int(i) for i in row["source"].split(":", 1)[1].split(",")]
        assert len(picked) == 3 and all(0 <= i < 205 for i in picked)

    summary = json.loads(NULL_JSON.read_text(encoding="utf-8"))
    assert summary["n_features_base"] == 2253
    for name in ("constant3", "shuffled3"):
        assert summary["overall"][name]["units"] == 6
        assert summary["overall"][name]["max_abs_delta"] > 0.0


def test_structure_table_still_has_no_chain_length() -> None:
    """D-47 的机器闸门：A5 被卡住的原因是结构表里没有链长。
    如果哪天 A 补了重复次数 / 可用权重，这条会红——那正是 A5 可以解冻的信号。
    D-47 as a gate: the structure table has no chain length. If A ever adds a repeat count or
    usable weights, this test turns red - that is the signal that A5 can be unblocked.
    """
    import pandas as pd

    path = REPO_ROOT / "data" / "processed" / "structures" / "psmiles.parquet"
    if not path.is_file():
        pytest.skip("没有 psmiles.parquet（D-38 的入口包）")
    frame = pd.read_parquet(path)
    assert len(frame) == 18152
    assert "repeat" not in " ".join(frame.columns).lower()
    wpsmiles = frame["wpsmiles"].dropna()
    assert len(wpsmiles) == 18036, "116 个支化结构没有 wpsmiles"
    fields = wpsmiles.str.split("|")
    assert (fields.str[1] == "1").all(), "重复字段 |1| 若出现别的值，说明有了链长"
    assert fields.str[2].nunique() == 1, "权重段若不再唯一，说明有了可用权重"
