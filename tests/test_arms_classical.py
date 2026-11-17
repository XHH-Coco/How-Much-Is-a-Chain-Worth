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

import csv
import json
import re
import subprocess
import sys
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


def test_main_table_cell_is_task_and_arm_not_fold() -> None:
    """一格 = (任务, 臂)，折与种子都平均掉；PB26 格 15 条、PolyBench 格 3 条（D-61）。

    The cell is (task, arm): folds and seeds are both averaged in - 15 runs for PB26, 3 for
    PolyBench (D-61).

    中文：
        用途：`main_table.yaml` 的 `cell` 段以前写"一格 = (任务,折,臂) 三个种子"，看起来自洽，
              但与产物不符。这条测试把**声明**和**产物**钉在一起：`runs_per_cell` 必须与
              `main_table.csv` 里 `n_runs` 的实际分布一致（24 格 × 3 + 8 格 × 15）。
              D 按这一段算误差棒，写错就是把折间方差整块漏掉。
    """
    payload = yaml.safe_load(
        (REPO_ROOT / "configs" / "experiments" / "main_table.yaml").read_text(encoding="utf-8")
    )
    cell = payload["cell"]
    assert payload["version"] >= "3.0"
    assert cell["aggregation"] == "mean_over_folds_and_seeds"
    assert cell["spread"] == "sample_std_over_cell_runs"
    assert cell["runs_per_cell"] == {"polybench": 3, "polybench26": 15}
    assert cell["cell_count"] == 32
    table = REPO_ROOT / "results" / "aggregate" / "main_table.csv"
    if not table.is_file():
        pytest.skip("还没有 main_table.csv")
    with table.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == cell["cell_count"]
    seen: dict[str, set[int]] = {}
    for row in rows:
        seen.setdefault(row["benchmark"], set()).add(int(row["n_runs"]))
    assert seen == {
        "polybench": {cell["runs_per_cell"]["polybench"]},
        "polybench26": {cell["runs_per_cell"]["polybench26"]},
    }, f"产物里的 n_runs 分布与 cell 段不符：{seen}"


def test_index_scope_is_the_main_table_not_every_a2_record() -> None:
    """`index --dry-run` 必须报主表口径的 96 条，而不是盘上全部 A2 记录（D-61）。

    `index --dry-run` must report the main-table 96, not every A2 record on disk (D-61).

    中文：
        用途：arch 的 15 条 A2 记录既没有 `n_level`、臂名又正好是 `a2_conventional`，
              旧过滤会把它们并进索引（96 → 111）且不报错。这里用命令行跑一遍，断言它
              打印的就是 96——把白名单钉在行为上，而不是钉在实现细节上。
    """
    script = REPO_ROOT / "scripts" / "04_run_cpu.py"
    if not script.is_file():
        pytest.skip("找不到 04_run_cpu.py")
    completed = subprocess.run(
        [sys.executable, str(script), "index", "--dry-run"],
        capture_output=True,
        text=True,
        encoding="utf-8",
        cwd=str(REPO_ROOT),
    )
    assert completed.returncode == 0, completed.stderr[-800:]
    expected = yaml.safe_load(
        (REPO_ROOT / "configs" / "experiments" / "main_table.yaml").read_text(encoding="utf-8")
    )["expected_runs"]["total"]
    assert f"{expected} 条" in completed.stdout, completed.stdout


def test_main_table_diff_class_names_what_differs() -> None:
    """差因分类不许再出现笼统的 `protocol`：只允许 `implementation` / `scale_only` / `unexplained`（D-63）。

    The difference class may no longer be the blanket "protocol": only implementation, scale_only
    and unexplained are allowed (D-63).

    中文：
        用途：`protocol` 把"换尺子就差 2%"和"换尺子仍差 38%、甚至方向反转"混成一类，读者会读错。
              这条测试锁住三件事：① 枚举里没有 `protocol`；② 每个非空标签都落在三个值里；
              ③ 标了 `scale_only` / `unexplained` 的行必须在 `notes` 里写出同尺度复跑的数，
              否则标签就是无凭据的断言。
    """
    table = REPO_ROOT / "results" / "aggregate" / "main_table.csv"
    if not table.is_file():
        pytest.skip("还没有 main_table.csv")
    allowed = {"implementation", "scale_only", "unexplained"}
    with table.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        label = (row["leaderboard_diff_class"] or "").strip()
        assert label in allowed | {""}, f"{row['task']}/{row['arm']}: 分类 {label!r} 不在枚举内"
        if label in {"scale_only", "unexplained"}:
            assert "ln" in row["notes"], (
                f"{row['task']}/{row['arm']}: 标了 {label} 但 notes 里没有同尺度复跑的数"
            )


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
    """记录必须过 `results/raw_runs/_schema.json`（机器校验，不是肉眼），且能落盘。
    The record must pass the frozen JSON schema (machine-checked, not eyeballing) and write to disk.
    """
    if not _has_params(SMALL_TASK):
        pytest.skip(f"{SMALL_TASK} 还没有超参：先跑 04_run_cpu.py search")
    outcome = C.train_one(small_data, a2_config, fold="official_70_30", seed=0)
    schema = json.loads((REPO_ROOT / "results" / "raw_runs" / "_schema.json").read_text(encoding="utf-8"))
    # 三套冻结环境（P0）里都没有 jsonschema（C-c 的审计套件为此用内置 draft-07 子集校验器，
    # 且 test_schema_uses_only_supported_keywords 保证 _schema.json 只用到被支持的关键词），
    # 所以这里复用同一校验器，而不是在测试里 import jsonschema（那样只有装了它的 venv 能跑）。
    # The frozen P0 environments ship no jsonschema; the audit suite's built-in draft-07 subset
    # validator is reused instead (test_schema_uses_only_supported_keywords proves the schema
    # only uses supported keywords).
    from test_run_records import _schema_problems

    problems = _schema_problems(outcome.record, schema, path="$")
    assert not problems, f"记录不过 _schema.json：{problems}"
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
    assert len({task for task, _fold, _seed in baseline}) == 8, "D-79：地板覆盖全部 8 个主表任务"
    per_task: dict[str, set[str]] = {}
    for task, fold, _seed in baseline:
        per_task.setdefault(task, set()).add(fold)
    # 折数由任务的官方划分决定：PolyBench 6 个任务各 1 折，PB26 的 2 个各 5 折
    assert sum(len(v) for v in per_task.values()) == len(baseline) == 16, (
        f"单元数应等于逐任务折数之和；实际 {len(baseline)} 行、{per_task}"
    )
    for (task, fold, seed), r2 in baseline.items():
        path = REPO_ROOT / "results" / "raw_runs" / f"{task}__a2_conventional__{fold}__seed{seed}.json"
        record = json.loads(path.read_text(encoding="utf-8"))
        assert abs(r2 - float(record["metrics"]["r2"])) < 1e-9, f"{task}/{fold} 与 A2 记录不一致"


def test_null_calibration_covers_all_main_tasks_and_three_conditions() -> None:
    """8 个主表任务 × 各自折数 × 3 条件 = 48 行（D-79）；列数 2253 / 2256；置换列必须能追到下标。
    Eight main-table tasks x their folds x 3 conditions = 48 rows (D-79); 2253 / 2256 columns.
    """
    if not NULL_CSV.is_file():
        pytest.skip("还没有 null_calibration.csv")
    import csv

    with NULL_CSV.open(encoding="utf-8", newline="") as fh:
        rows = list(csv.DictReader(fh))
    base = [r for r in rows if r["condition"] == "baseline"]
    const = [r for r in rows if r["condition"] == "constant3"]
    shuffled = [r for r in rows if r["condition"] == "shuffled3"]
    assert len({r["task"] for r in rows}) == 8, "D-79：地板覆盖全部 8 个主表任务"
    assert len(base) == 16, f"单元数 = 6×1 + 2×5；实际 {len(base)}"
    # 三个条件必须落在**同一批** (task, fold) 上，否则地板与零点不可比
    keys = {(r["task"], r["fold"]) for r in base}
    for cells in (const, shuffled):
        assert {(r["task"], r["fold"]) for r in cells} == keys
    assert len(rows) == 3 * len(base), f"期望 {3 * len(base)} 行，实际 {len(rows)}"
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
        assert summary["overall"][name]["units"] == len(base) * 1, "汇总里的单元数必须与表里一致"
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


# ---- D-66：Q-10 的 ln 尺度敏感性 / Q-10's ln-scale sensitivity ----
#
# 这四条守的是同一件事：`main_table.csv` 里标 `scale_only` / `unexplained` 的那四行，其 `notes`
# 里的同尺度数**必须能在一张可复算的表里找回**。D-63 当时只有一句"临时跑的、未入账"，
# 四个数（0.3413 / 0.6226 / 0.1737 / 0.4640）在仓库里没有任何东西能重算出来，
# 而守门测试只断言 `"ln" in notes`——把 0.3413 改成 0.9999 也照样绿。
# All four guard the same thing: the ln values quoted in main_table.csv's notes must be findable in
# a recomputable table. D-63 had only "ran it once, not recorded", so nothing in the repo could
# reproduce those numbers, and the old gate merely asserted `"ln" in notes`.

LOG_TARGET_TABLE = REPO_ROOT / "results" / "aggregate" / "log_target_sensitivity.csv"
OFFICIAL_COMPARE_TABLE = REPO_ROOT / "results" / "aggregate" / "polybench_official_compare.csv"
LN_TASKS = {
    "pb_t2_modulus",
    "pb_t2_permeability",
    "pb_t3_dielectric_loss",
    "pb_t3_iwlt",
}
LOG_TARGET_COLUMNS = (
    "task",
    "arm",
    "label_transform",
    "official_target_scale",
    "fold",
    "seed",
    "n_train",
    "n_test",
    "r2",
    "mae",
    "rmse",
    "wall_clock_s",
    "official_best_tree_r2",
    "delta_vs_official",
    "official_best_tree_algorithm",
    "official_best_tree_descriptor",
    "note",
)


def test_label_values_is_natural_log_or_nothing() -> None:
    """`label_values` 只认 `raw` 与 `ln`，且 `ln` 必须拒收非正标签。
    label_values accepts raw and ln only, and ln must reject non-positive labels.

    中文：这个仓库被 log10 坑过一次（PolyBench 的 `property_log` 是自然对数，D-16 差点当 log10 用），
          所以枚举里不许有第三个值；`ln(0) = −inf` 会让 R² 静默变 NaN，宁可当场抛。
    """
    values = np.array([1.0, np.e, np.e**2], dtype=np.float64)
    assert np.allclose(C.label_values(values, "raw"), values)
    assert np.allclose(C.label_values(values, "ln"), [0.0, 1.0, 2.0]), "ln 必须是自然对数"
    assert C.label_values(values, "ln")[1] == pytest.approx(1.0)
    with pytest.raises(G.ArmError):
        C.label_values(np.array([0.0, 1.0]), "ln")
    with pytest.raises(G.ArmError):
        C.label_values(np.array([-1.0]), "ln")
    with pytest.raises(G.ArmError):
        C.label_values(values, "log10")


def _ln_rows() -> list[dict[str, str]]:
    if not LOG_TARGET_TABLE.is_file():
        pytest.skip(
            f"还没有 {LOG_TARGET_TABLE.name}：跑 `python scripts/04_run_cpu.py log-target`"
        )
    with LOG_TARGET_TABLE.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _ln_mean(task: str, arm: str) -> float:
    values = [
        float(row["r2"]) for row in _ln_rows() if row["task"] == task and row["arm"] == arm
    ]
    assert values, f"{task}/{arm} 在 {LOG_TARGET_TABLE.name} 里没有行"
    return float(np.mean(values))


def test_log_target_table_shape() -> None:
    """24 行 = 4 任务 × 2 臂 × 3 种子；列齐全；尺度和官方尺度都写死在行里。
    Twenty-four rows (4 tasks x 2 arms x 3 seeds), every column present, both scales recorded.
    """
    rows = _ln_rows()
    assert list(rows[0].keys()) == list(LOG_TARGET_COLUMNS)
    assert len(rows) == len(LN_TASKS) * 2 * 3, f"应为 24 行，实际 {len(rows)}"
    assert {row["task"] for row in rows} == LN_TASKS
    assert {row["arm"] for row in rows} == {"a1_floor", "a2_conventional"}
    assert {row["label_transform"] for row in rows} == {"ln"}, "这张表只放 ln 尺度的行"
    assert {row["official_target_scale"] for row in rows} == {"ln"}, (
        "官方尺度必须从对照表读成 ln；若这里变成 raw，说明官方改了口径"
    )
    for task in LN_TASKS:
        seeds = sorted(
            int(row["seed"]) for row in rows
            if row["task"] == task and row["arm"] == "a2_conventional"
        )
        assert seeds == [0, 1, 2], f"{task} 的种子不是 0/1/2：{seeds}"


def test_log_target_official_reference_comes_from_the_compare_table() -> None:
    """表里的官方数必须与 `polybench_official_compare.csv` 逐位相同——不许手抄。
    The official values must equal the comparison table's, i.e. nothing is hand-copied.
    """
    if not OFFICIAL_COMPARE_TABLE.is_file():
        pytest.skip("没有 polybench_official_compare.csv（04b 的产物）")
    with OFFICIAL_COMPARE_TABLE.open("r", encoding="utf-8", newline="") as handle:
        reference = {
            str(row["task"]): float(row["official_best_tree_r2"])
            for row in csv.DictReader(handle)
            if (row.get("official_target_scale") or "").strip() == "ln"
            and (row.get("ours_target_scale") or "").strip() == "raw"
        }
    assert set(reference) == LN_TASKS, f"官方对照表里的 ln 任务变了：{sorted(reference)}"
    for row in _ln_rows():
        expected = reference[row["task"]]
        assert float(row["official_best_tree_r2"]) == pytest.approx(expected, abs=1e-9), (
            f"{row['task']} 的官方数 {row['official_best_tree_r2']} 与对照表的 {expected} 不符"
        )
        assert float(row["delta_vs_official"]) == pytest.approx(
            float(row["r2"]) - expected, abs=1e-9
        )


def test_log_target_notes_match_the_table() -> None:
    """`main_table.csv` 的 `notes` 里写的 ln 值必须等于本表同任务的 3 种子均值（4 位小数）。
    The ln values quoted in main_table.csv's notes must equal this table's three-seed mean.

    中文：这是把 D-63 那条"notes 里得有 ln 字样"的弱断言换成强断言——数错了、或者有人只改了
          notes 没重跑，这条都会红。措辞（"差在协议"之类）不受约束，只有数字被锁。
    English: replaces D-63's weak "notes must mention ln" with a numeric check, so a stale or
    hand-edited number turns red. Only the digits are locked, not the wording.
    """
    table = REPO_ROOT / "results" / "aggregate" / "main_table.csv"
    if not table.is_file():
        pytest.skip("还没有 main_table.csv")
    with table.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    checked = 0
    for row in rows:
        if (row["leaderboard_diff_class"] or "").strip() not in {"scale_only", "unexplained"}:
            continue
        match = re.search(r"ln R²=([0-9]+\.[0-9]+)", row["notes"] or "")
        assert match, f"{row['task']}/{row['arm']}: 标了差因分类但 notes 里没有 `ln R²=` 形式的数"
        quoted = float(match.group(1))
        expected = _ln_mean(row["task"], "a2_conventional")
        assert quoted == pytest.approx(expected, abs=5e-5), (
            f"{row['task']}: notes 写 {quoted}，log_target_sensitivity.csv 的均值是 {expected:.4f}"
        )
        checked += 1
    assert checked == 4, f"应有 4 行标了 scale_only/unexplained，实际 {checked} 行"


def test_raw_runs_never_hold_a_non_raw_transform() -> None:
    """`results/raw_runs/` 里不许出现 `label_transform` 非 `raw` 的记录。
    No record under results/raw_runs/ may carry a non-raw label_transform.

    中文：这是唯一能挡住"ln 结果覆盖主表"的机器判据。run_id 只有 `task__arm__fold__seed`，
          不含尺度；一旦写进去就是**同名覆盖**，而 `config_hash` / `split_version` /
          `pack_sha256` / `n_train` / `n_test` 全都不变——schema 校验、审计、主表重建没有一处会红
          （D-34 ③ 的微调踩的是同一个坑，所以它也被赶出了 `raw_runs/`）。
    English: the only machine check that can stop an ln result from overwriting the main table. The
    run_id has no scale segment, and an overwrite leaves every identity field unchanged, so nothing
    else in the repo would notice (the same trap D-34 section 3 hit with fine-tuning).
    """
    root = REPO_ROOT / "results" / "raw_runs"
    if not root.is_dir():
        pytest.skip("没有 results/raw_runs/")
    offenders: list[str] = []
    for path in sorted(root.glob("*.json")):
        if path.name.startswith("_"):
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        transform = payload.get("label_transform")
        if transform not in (None, "raw"):
            offenders.append(f"{path.name}={transform!r}")
    assert not offenders, (
        f"raw_runs/ 里有 {len(offenders)} 条非 raw 尺度的记录（会静默覆盖主表同名记录）："
        f"{offenders[:3]}；ln 的产物只能落 results/aggregate/log_target_sensitivity.csv"
    )


def test_log_target_floor_is_a_floor() -> None:
    """A1 在 ln 空间也必须是"瞎猜"：R² 贴近 0（绝对值 < 0.05），否则地板就不是地板。
    A1 must still be guessing on the ln scale: |R²| < 0.05, or it is not a floor.

    中文：主表里 A1 的 raw 值有 −0.0329（iwlt）这种"比均值还差"的数，读者会以为地板坏了。
          换到 ln 空间之后四个任务都在 −0.0002…−0.0171，这条把"地板在同尺度下依然贴地"锁住，
          也顺带证明 A2 的 0.17–0.62 不是被分母抬起来的。
    """
    for task in sorted(LN_TASKS):
        value = _ln_mean(task, "a1_floor")
        assert abs(value) < 0.05, f"{task} 的 ln 地板 R²={value:.4f}，不像瞎猜的地板"
        floor_r2 = value
        model_r2 = _ln_mean(task, "a2_conventional")
        assert model_r2 > floor_r2, f"{task}: ln 空间 A2 ({model_r2:.4f}) 不高于地板 ({floor_r2:.4f})"
