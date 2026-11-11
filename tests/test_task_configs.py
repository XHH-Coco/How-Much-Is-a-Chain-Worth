"""任务清单（`configs/tasks/*.yaml`）的回归测试：schema、结构、与上游清单的一致性。
Regression tests for the frozen task list: schema, structure, and agreement with upstream manifests.

中文：
    用途：A-4 的验收标准②要求"字段缺失 / 任务名拼错当场报错"。本文件把这句话变成 13 条断言，
          并且**故意把配置改坏**（在 tmp 目录里改副本，绝不碰仓库里的冻结物），确认校验器真的
          会红——否则"有校验"只是一句好话。
    依赖：src/chainworth/io/tasks.py、configs/tasks/*.yaml、data/manifests/*.csv；
          PolyBench26 官方 manifest.json 若不在本地，相关断言自动 skip（别人克隆后也能跑）。
    配置：无（测试用的坏配置在 tmp 里现造）。
    对外：pytest 收集。
    被谁用：CI、A 的交接自检、D 的入口审计（注入错误时测试必须红）。

English:
    Purpose: acceptance criterion 2 says a missing field or a misspelled dataset name must fail
    loudly. This file turns that into 13 assertions AND deliberately breaks copies of the configs
    in a tmp directory (never the frozen files) to prove the validator really goes red.
    Depends on: src/chainworth/io/tasks.py, configs/tasks/*.yaml, data/manifests/*.csv, and the
    official PolyBench26 manifest.json files (those assertions skip when the data is absent, so a
    fresh clone still passes).
    Config: none - the deliberately broken configs are built in tmp.
    Public API: none (collected by pytest).
    Callers: CI, A's hand-over self-check, D's entry audit (injecting an error must turn it red).
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest
import yaml

from chainworth.io import tasks as taskmod
from chainworth.io.tasks import (
    CONFIGS_DIR,
    RAW_POLYBENCH26,
    TaskConfigError,
    load_all_task_configs,
    load_task_config,
    render_snapshot,
    verify_all,
)


# ---- 工具：把整目录拷到 tmp 再改一份 / Copy the directory to tmp, then break one file ----


def _copy_configs(tmp_path: Path) -> Path:
    """把 `configs/tasks` 整目录拷进 tmp，返回新目录。

    Copy the whole `configs/tasks` directory into tmp and return it.

    中文：
        用途：让"故意改坏配置"不影响仓库里的冻结物——冻结物被动过等于该阶段重做（README §9）。
        输入：tmp_path —— pytest 提供的临时目录。
        输出：Path，tmp 里的配置目录。
        数据来源：configs/tasks。
        配置来源：无。副作用：写 tmp。异常：无。

    English:
        Purpose: break a copy instead of the frozen file, because touching a frozen artefact
        means redoing the stage.
        Args: tmp_path from pytest. Returns: the tmp directory.
        Data source: configs/tasks. Config: none. Side effects: writes to tmp. Raises: none.
    """
    target = tmp_path / "tasks"
    shutil.copytree(CONFIGS_DIR, target)
    return target


def _mutate(directory: Path, task_id: str, **changes: object) -> Path:
    """改一份配置里的若干字段（值传 `None` 表示删掉该键）。

    Change a few fields of one config (passing None deletes the key).

    中文：
        用途：造出"作者漏写字段"与"作者写错值"两类坏样本。
        输入：directory —— 配置目录；task_id —— 改哪一份；changes —— 字段名=新值。
        输出：被改文件的路径。
        数据来源：磁盘上的 yaml。配置来源：无。副作用：覆盖文件。异常：无。

    English:
        Purpose: build both failure classes - a forgotten field and a wrong value.
        Args: directory, task_id, changes (field=new value; None deletes).
        Returns: the mutated path. Data source: the yaml on disk.
        Side effects: overwrites the file. Raises: none.
    """
    path = directory / f"{task_id}.yaml"
    text = path.read_text(encoding="utf-8")
    header, _, body = text.partition("\n")
    data = yaml.safe_load(body)
    for key, value in changes.items():
        if value is None:
            data.pop(key, None)
        else:
            data[key] = value
    path.write_text(header + "\n" + yaml.safe_dump(data, allow_unicode=True, sort_keys=False), encoding="utf-8", newline="\n")
    return path


# ---- 1. 正向：18 份配置全部合法 / Happy path: all 18 configs are valid ----


def test_all_configs_pass_verification() -> None:
    """整份清单通过校验，且角色计数与 Proposal 表 A–E 一致。 / Whole list passes; role counts match the proposal."""
    report = verify_all()
    assert report["errors"] == [], f"任务配置有问题：{report['errors']}"
    assert report["n_tasks"] == 18
    assert report["counts"] == {"main": 8, "reserve": 5, "boundary": 3, "mechanism": 2}


def test_main_table_is_two_per_tier_plus_polybench26() -> None:
    """验收标准①：8 个主表任务 = 每层 2 个（6）+ PolyBench26 2 个。 / Two per tier plus two PolyBench26."""
    configs = load_all_task_configs()
    main = {task_id: config for task_id, config in configs.items() if config["role"] == "main"}
    per_tier = {tier: [tid for tid, cfg in main.items() if cfg["tier"] == tier] for tier in (1, 2, 3)}
    assert [len(per_tier[tier]) for tier in (1, 2, 3)] == [2, 2, 2]
    assert len([tid for tid, cfg in main.items() if cfg["benchmark"] == "polybench26"]) == 2
    assert len(main) == 8


def test_reserve_list_is_complete_and_marked() -> None:
    """验收标准④：5 个备用任务都在配置里、标 `role: reserve`，且不可用的那个写了原因。 / Five reserves, one unavailable with a reason."""
    configs = load_all_task_configs()
    reserves = {tid: cfg for tid, cfg in configs.items() if cfg["role"] == "reserve"}
    assert len(reserves) == 5
    unavailable = {tid: cfg for tid, cfg in reserves.items() if cfg["status"] == "unavailable"}
    assert set(unavailable) == {"pb_t1_bandgap_chain"}
    assert "单位" in unavailable["pb_t1_bandgap_chain"]["status_note"]


# ---- 2. 正向：数字能追到上游 / Happy path: numbers trace to upstream ----


def test_polybench_counts_match_harmonised_manifest() -> None:
    """PolyBench 6 个主表任务的 n 与 A-3 调和清单逐项相等。 / PolyBench n equals A-3's manifest."""
    configs = load_all_task_configs()
    expected = {"pb_t1_tg": 8055, "pb_t1_tm": 3796, "pb_t2_modulus": 1034,
                "pb_t2_permeability": 711, "pb_t3_dielectric_loss": 279, "pb_t3_iwlt": 228}
    for task_id, n_labels in expected.items():
        config = configs[task_id]
        assert config["n_labels"] == n_labels
        assert config["target_unit"] in ("K", "GPa", "cm3(STP)*cm/(cm2*s*Pa)", "无量纲", "hour")


def test_polybench26_numbers_match_official_manifest() -> None:
    """PolyBench26 两个主表任务的 n / 档位 / 目标列与官方 manifest.json 逐项相等。 / Equal to the official manifest."""
    official_dir = RAW_POLYBENCH26 / "Datasets" / "scaling_indices" / "MD_5000" / "density" / "homopolymer_density" / "seed_42"
    if not (official_dir / "manifest.json").is_file():
        pytest.skip("PolyBench26 原始数据不在本机（data/raw 不入库），跳过官方 manifest 比对")
    official = json.loads((official_dir / "manifest.json").read_text(encoding="utf-8"))
    config = load_all_task_configs()["pb26_homopolymer_density"]
    entry = config["datasets"][0]
    assert config["n_labels"] == official["n_rows"] == entry["n_rows"]
    assert entry["train_sizes"] == official["train_sizes"] == [300, 1000]
    assert config["target_column"] == official["target_column"] == "density"


def test_polymetrix_tg_is_already_kelvin() -> None:
    """PolyMetriX Tg 的源列就叫 `Tg_K`：这条断言守住 D-18，防止有人再加 273.15。 / The source column is Tg_K."""
    config = load_all_task_configs()["pb26_polymetrix_tg"]
    assert config["target_column"] == "Tg_K"
    assert config["target_unit"] == "K"
    assert "不得再加 273.15" in config["unit_evidence"]


# ---- 3. 负向：坏配置必须当场报错 / Negative: a broken config must fail loudly ----


def test_misspelled_dataset_name_fails(tmp_path: Path) -> None:
    """任务名拼错（少一个 t）必须报错，且错误信息指出是名字问题。 / A misspelled name must fail."""
    directory = _copy_configs(tmp_path)
    _mutate(directory, "pb_t1_tg", dataset_name="Glass transition temperture")
    with pytest.raises(TaskConfigError) as excinfo:
        load_task_config(directory / "pb_t1_tg.yaml")
    assert "39 行" in str(excinfo.value)


def test_missing_field_fails(tmp_path: Path) -> None:
    """删掉必填字段（`unit_evidence`）必须报"缺必填字段"，而不是当空字符串放过。 / A deleted field must fail."""
    directory = _copy_configs(tmp_path)
    _mutate(directory, "pb_t1_tg", unit_evidence=None)
    with pytest.raises(TaskConfigError) as excinfo:
        load_task_config(directory / "pb_t1_tg.yaml")
    assert "缺必填字段" in str(excinfo.value)


def test_unknown_field_fails(tmp_path: Path) -> None:
    """拼错的字段名（`tierr`）必须报未知字段，否则作者以为写了、下游读不到。 / A typo field must fail."""
    directory = _copy_configs(tmp_path)
    _mutate(directory, "pb_t1_tg", tierr=1)
    with pytest.raises(TaskConfigError) as excinfo:
        load_task_config(directory / "pb_t1_tg.yaml")
    assert "未知字段" in str(excinfo.value)


def test_wrong_n_against_upstream_fails(tmp_path: Path) -> None:
    """把 n 改小 1，必须与 harmonised_manifest.csv 对不上而报错。 / Off-by-one n must fail."""
    directory = _copy_configs(tmp_path)
    _mutate(directory, "pb_t1_tg", n_labels=8054)
    report = verify_all(directory)
    assert any("n_labels" in problem and "harmonised_manifest" in problem for problem in report["errors"])


def test_kfold_on_polybench_fails(tmp_path: Path) -> None:
    """PolyBench 上写 k 折必须报错——官方协议只有固定 70/30（红线）。 / k-fold on PolyBench is a red line."""
    directory = _copy_configs(tmp_path)
    _mutate(directory, "pb_t1_tg", eval_unit="5 kfold x 3 seeds")
    with pytest.raises(TaskConfigError) as excinfo:
        load_task_config(directory / "pb_t1_tg.yaml")
    assert "70/30" in str(excinfo.value)


def test_split_present_but_missing_fails(tmp_path: Path) -> None:
    """声明划分已到位但文件不存在，必须报错（A-5 交付后这一条才会变成合法）。 / Declared split must exist."""
    directory = _copy_configs(tmp_path)
    _mutate(directory, "pb_t1_tg", split_status="present")
    with pytest.raises(TaskConfigError) as excinfo:
        load_task_config(directory / "pb_t1_tg.yaml")
    assert "文件不存在" in str(excinfo.value)


def test_unverified_unit_with_a_value_fails(tmp_path: Path) -> None:
    """单位没证实却写了一个单位值，必须报错——那正是"猜"。 / An unverified unit may not carry a value."""
    directory = _copy_configs(tmp_path)
    _mutate(directory, "pb26_homopolymer_density", target_unit="g/cm3")
    with pytest.raises(TaskConfigError) as excinfo:
        load_task_config(directory / "pb26_homopolymer_density.yaml")
    assert "必须为 null" in str(excinfo.value)


# ---- 4. 快照与配置不许漂移 / The snapshot must not drift from the configs ----


def test_snapshot_doc_matches_configs() -> None:
    """`docs/Task List Snapshot D0.md` 里的清单表必须与配置逐行相同（防止手抄漂移）。 / No drift."""
    snapshot = taskmod.REPO_ROOT / "docs" / "Task List Snapshot D0.md"
    if not snapshot.is_file():
        pytest.fail("缺少 docs/Task List Snapshot D0.md——A-4 的产出之一")
    doc_lines = {line.strip() for line in snapshot.read_text(encoding="utf-8").splitlines()}
    generated = [line.strip() for line in render_snapshot(load_all_task_configs()).splitlines()]
    missing = [line for line in generated if line not in doc_lines]
    assert not missing, f"快照表与配置不一致（{len(missing)} 行），请重跑 `--snapshot` 后替换：{missing[:3]}"


def test_cli_verify_returns_zero(capsys: pytest.CaptureFixture[str]) -> None:
    """命令行 `--verify` 必须返回 0（CI 闸门）。 / The CLI must exit 0."""
    assert taskmod.main(["--verify"]) == 0
    assert "[OK]" in capsys.readouterr().out
