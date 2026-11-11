"""阶段 01（标签调和）的回归测试：合成样例守逻辑，真实数据守不变式。
Regression tests for stage 01: synthetic data guards the logic, real data guards the invariants.

中文：
    用途：把 A-3 的四条验收标准与三条"不许猜"红线变成 pytest 断言，让后面任何一次改动
          都能立刻看出"是否悄悄改了行数、单位或决策覆盖"。
    依赖：src/chainworth/io/harmonise.py、data/raw/polybench/*.csv（真实数据缺失时自动 skip，
          因为 `data/raw/` 不入库——这是 README 的硬规则 1，不是测试的缺陷）。
    配置：无。
    对外：无（pytest 收集）。
    被谁用：人工 `pytest tests/`、CI。

English:
    Purpose: turn A-3's four acceptance criteria into assertions so any later edit immediately
    shows whether rows, units or decision coverage quietly changed.
    Depends on: src/chainworth/io/harmonise.py and data/raw/polybench/*.csv (skipped when the raw
    data is absent, because data/raw/ is never committed - README hard rule 1).
    Config: none. Public API: none (pytest collects it). Callers: humans and CI running pytest.
"""

from __future__ import annotations

import csv

import pytest

from chainworth.io.harmonise import (
    DECISION_COLUMNS,
    EXCLUDED_TASKS,
    HARMONISED_MANIFEST_PATH,
    RAW_DIR,
    TASK_SPECS,
    TaskSpec,
    _identity,
    _to_kelvin,
    append_unit_decisions,
    build_decisions,
    harmonise_task,
    verify_outputs,
)

# 真实数据在不在：`data/raw/` 被 gitignore，队友克隆后没有这份数据，测试要 skip 而不是失败。
# Is the real data present? data/raw/ is gitignored, so a fresh clone skips rather than fails.
HAS_RAW_DATA = RAW_DIR.exists() and any(RAW_DIR.glob("*.csv"))
requires_raw_data = pytest.mark.skipif(
    not HAS_RAW_DATA, reason="data/raw/polybench/ 不在（不入库）；跳过真实数据断言 / raw data absent"
)


# ---- 1. 合成样例：逻辑本身对不对 / Synthetic sample: is the logic right ----


def _write_fake_task(directory, rows: list[tuple[str, str, str]]) -> TaskSpec:
    """在临时目录里造一个"像 PolyBench 一样"的小 CSV，并返回配套的配方。
    Create a small PolyBench-shaped CSV in a temp directory and return its recipe.

    中文：
        用途：让去重、剔除、换算三条逻辑可以在 4 行数据上被单独验证，不依赖真实数据是否就位。
        输入：directory —— 临时目录；rows —— (聚合物名, SMILES, 标签) 三元组列表。
        输出：TaskSpec，dataset_name 与写出的文件名一致，标签列叫 `Tg(℃)`，换算为 ℃→K。
        数据来源：本函数自己写出的临时 CSV（不进仓库）。
        配置来源：无。副作用：写一个临时 CSV。
        异常：OSError —— 临时目录不可写。

    English:
        Purpose: let dedupe, dropping and conversion be verified on four rows, independent of
        whether the real data is present.
        Args: directory - temp dir; rows - (name, SMILES, label) triples.
        Returns: a TaskSpec whose dataset matches the file written, with a `Tg(℃)` label column
        converted to kelvin.
        Data source: the temp CSV written here (never committed).
        Config: none. Side effects: writes one temp CSV. Raises: OSError if unwritable.
    """
    path = directory / "Fake task.csv"
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(["polymer_name", "SMILES", "Tg(℃)"])
        writer.writerows(rows)
    return TaskSpec(
        task_id="fake_t1_tg",
        dataset_name="Fake task",
        role="main",
        label_column="Tg(℃)",
        label_raw_unit="℃",
        target_unit="K",
        rule="T[K] = T[℃] + 273.15",
        convert=_to_kelvin,
        rationale="测试用配方",
    )


def test_harmonise_dedupes_equivalent_smiles(tmp_path, monkeypatch):
    """同一分子写成两种 SMILES，必须只留一行——去重键是规范形式，不是原始字符串。
    One molecule written two ways must collapse to one row; the key is canonical, not the raw string.

    中文：
        用途：守住"去重键 = 规范 SMILES"这条决策；原始字符串去重会漏掉写法不同的同一分子。
        输入：tmp_path、monkeypatch（由 pytest 注入）。
        输出：无（断言）。
        数据来源：本测试临时写出的 CSV。
        配置来源：无。副作用：写临时文件。异常：AssertionError —— 去重失效。

    English:
        Purpose: guard the "dedupe on canonical SMILES" decision; string-based dedupe would let
        one molecule in two spellings survive twice.
        Args: tmp_path, monkeypatch (injected by pytest). Returns: none (assertions).
        Data source: the temp CSV written here. Config: none.
        Side effects: temp file. Raises: AssertionError when dedupe fails.
    """
    from chainworth.io import harmonise

    monkeypatch.setattr(harmonise, "RAW_DIR", tmp_path)
    spec = _write_fake_task(
        tmp_path,
        [
            ("poly A", "CCO", "100"),
            ("poly A again", "OCC", "120"),      # 与 CCO 是同一分子 / same molecule as CCO
            ("poly bad", "not-a-smiles", "50"),  # RDKit 解析失败 / unparsable
            ("poly empty", "CCC", ""),           # 标签缺失 / missing label
        ],
    )
    frame, outcome = harmonise_task(spec)

    assert outcome.rows_in == 4
    assert outcome.rows_out == 1
    assert outcome.n_duplicates == 1
    assert outcome.n_unparsable_smiles == 1
    assert outcome.n_invalid_label == 1
    assert outcome.rows_removed == 3
    assert list(frame["smiles_canonical"]) == ["CCO"]
    assert frame["label"].iloc[0] == pytest.approx(100 + 273.15)


def test_decisions_cover_every_row_removed(tmp_path, monkeypatch):
    """决策表必须自己把账平掉：n_rows_affected 之和 == 调和前后的行数差。
    The decision table must balance: the sum of n_rows_affected equals rows in minus rows out.

    中文：
        用途：守住验收标准③；否则"删了几行"和"记了几行"会各说各话。
        输入：tmp_path、monkeypatch。
        输出：无（断言）。
        数据来源：本测试临时写出的 CSV。
        配置来源：无。副作用：写临时文件。
        异常：AssertionError —— 账不平。

    English:
        Purpose: guard acceptance criterion 3; otherwise "rows deleted" and "rows recorded" drift apart.
        Args: tmp_path, monkeypatch. Returns: none (assertions).
        Data source: the temp CSV. Config: none. Side effects: temp file.
        Raises: AssertionError when the books do not balance.
    """
    from chainworth.io import harmonise

    monkeypatch.setattr(harmonise, "RAW_DIR", tmp_path)
    spec = _write_fake_task(
        tmp_path,
        [
            ("poly A", "CCO", "100"),
            ("poly A again", "OCC", "120"),
            ("poly bad", "not-a-smiles", "50"),
            ("poly empty", "CCC", ""),
        ],
    )
    _, outcome = harmonise_task(spec)
    decisions = build_decisions(outcome, decided_at="2026-11-12T00:00:00+08:00")

    assert sum(int(row["n_rows_affected"]) for row in decisions) == outcome.rows_removed
    kinds = [row["decision_kind"] for row in decisions]
    assert kinds.count("label_unit") == 1
    assert all(str(row["evidence"]).strip() for row in decisions)


def test_unit_decision_records_how_many_values_changed():
    """℃→K 的决策要写明"改写了多少行"，identity 则写 0——两种情形都不能含糊。
    A degC-to-K decision must state how many values changed; an identity decision states zero.

    中文：
        用途：守住 n_values_converted 的语义，免得"换算过"和"没换算"在表里长得一样。
        输入：无。
        输出：无（断言）。
        数据来源：TASK_SPECS 注册表（不读数据文件）。
        配置来源：无。副作用：无。异常：AssertionError —— 语义丢了。

    English:
        Purpose: guard the meaning of n_values_converted so "converted" and "not converted" cannot
        look identical in the table.
        Args: none. Returns: none (assertions).
        Data source: the TASK_SPECS registry (no data file read).
        Config: none. Side effects: none. Raises: AssertionError if the distinction is lost.
    """
    identity_tasks = [spec for spec in TASK_SPECS if spec.convert is _identity]
    kelvin_tasks = [spec for spec in TASK_SPECS if spec.convert is _to_kelvin]
    assert identity_tasks and kelvin_tasks, "注册表里应同时有 identity 与 ℃→K 两类任务"
    assert all(spec.target_unit == spec.label_raw_unit for spec in identity_tasks)
    assert all(spec.target_unit == "K" for spec in kelvin_tasks)


def test_append_only_decisions_reject_rewrites(tmp_path):
    """同一 (task, kind, property) 的历史决策不许被改写——这是 README §4.2 的冻结规则。
    A historical decision may not be rewritten; this is README 4.2's freeze rule.

    中文：
        用途：守住"只能追加"；改口径必须新增一条，不能覆盖。
        输入：tmp_path。
        输出：无（断言）。
        数据来源：本测试临时写出的决策表。
        配置来源：无。副作用：写临时文件。
        异常：AssertionError —— 覆盖成功（那就是错的）。

    English:
        Purpose: guard append-only: changing a rule needs a new row, never an edit.
        Args: tmp_path. Returns: none (assertions).
        Data source: the temp decision table. Config: none. Side effects: temp file.
        Raises: AssertionError if an overwrite is allowed.
    """
    path = tmp_path / "unit_decisions.csv"
    base = {column: "" for column in DECISION_COLUMNS}
    base.update(
        {
            "task_id": "fake_t1_tg",
            "decision_kind": "label_unit",
            "property": "Tg(℃)",
            "raw_unit": "℃",
            "target_unit": "K",
            "rule": "T[K] = T[℃] + 273.15",
            "n_rows_affected": "0",
            "n_values_converted": "4",
            "evidence": "测试",
            "decided_by": "test",
            "decided_at": "2026-11-12T00:00:00+08:00",
        }
    )
    added, reused = append_unit_decisions([base], path)
    assert (added, reused) == (1, 0)

    # 内容一模一样：视为重复，不新增 / identical content: treated as a repeat
    added, reused = append_unit_decisions([base], path)
    assert (added, reused) == (0, 1)

    # 改了 evidence：必须报错，而不是悄悄覆盖 / changed evidence: must fail, not overwrite
    changed = dict(base, evidence="换了个说法")
    with pytest.raises(ValueError):
        append_unit_decisions([changed], path)


# ---- 2. 真实数据：不变式与验收 / Real data: invariants and acceptance ----


@requires_raw_data
@pytest.mark.parametrize("spec", TASK_SPECS, ids=[spec.task_id for spec in TASK_SPECS])
def test_each_task_label_is_recomputable(spec):
    """每个任务的 label 必须等于 convert(label_raw)，且规范 SMILES 任务内唯一。
    Each task's label must equal convert(label_raw) and canonical SMILES must be unique per task.

    中文：
        用途：守住验收标准④的前半段——"表里的数不是写进去的，是按规则算出来的"。
        输入：spec —— 参数化注入的任务配方。
        输出：无（断言）。
        数据来源：data/raw/polybench/<数据集名>.csv。
        配置来源：spec。
        副作用：无（harmonise_task 不写盘）。
        异常：AssertionError —— 换算与数据脱节，或有重复键漏网。

    English:
        Purpose: guard the first half of acceptance criterion 4 - the stored number is computed
        by the rule, not typed in.
        Args: spec - parametrised recipe. Returns: none (assertions).
        Data source: data/raw/polybench/<dataset>.csv. Config: the spec.
        Side effects: none. Raises: AssertionError on a drifted rule or a surviving duplicate.
    """
    frame, outcome = harmonise_task(spec)
    assert len(frame) == outcome.rows_out
    assert frame["smiles_canonical"].is_unique
    assert set(frame["label_unit"]) == {spec.target_unit}
    for raw_value, stored in zip(frame["label_raw"], frame["label"]):
        assert stored == pytest.approx(spec.convert(float(raw_value)))


@requires_raw_data
def test_main_table_tasks_clear_the_200_label_gate():
    """主表 6 个任务调和后可用标签都必须 >= 200（分工书验收标准②）。
    All six main-table tasks must keep at least 200 usable labels after harmonisation.

    中文：
        用途：守住"低于 200 触发备用任务替换"的闸门；最紧的是等温失重 228。
        输入：无。
        输出：无（断言）。
        数据来源：data/raw/polybench/*.csv（现算，不读产出的 parquet）。
        配置来源：MAIN_TABLE_MIN_LABELS。
        副作用：无。异常：AssertionError —— 有任务跌破闸门。

    English:
        Purpose: guard the "below 200, swap in a reserve" gate; the tightest is IWLT at 228.
        Args: none. Returns: none (assertions).
        Data source: the raw CSVs (computed here, not read from the parquet outputs).
        Config: MAIN_TABLE_MIN_LABELS. Side effects: none.
        Raises: AssertionError when a task falls below the gate.
    """
    from chainworth.protocol.constants import MAIN_TABLE_MIN_LABELS

    for spec in TASK_SPECS:
        if spec.role != "main":
            continue
        _, outcome = harmonise_task(spec)
        assert outcome.rows_out >= MAIN_TABLE_MIN_LABELS, f"{spec.task_id} 只有 {outcome.rows_out} 条"


def test_excluded_tasks_record_a_unit_reason():
    """被整列弃用的任务，reason 必须写清"没有单位证据"，不能只写一句"质量差"。
    A wholesale-excluded task must state "no unit evidence", not just "poor quality".

    中文：
        用途：守住红线——排除也要有可核的理由。
        输入：无。
        输出：无（断言）。
        数据来源：EXCLUDED_TASKS 注册表。
        配置来源：无。副作用：无。异常：AssertionError —— 理由不具体。

    English:
        Purpose: guard the red line - even an exclusion needs a checkable reason.
        Args: none. Returns: none (assertions).
        Data source: the EXCLUDED_TASKS registry. Config: none. Side effects: none.
        Raises: AssertionError when the reason is vague.
    """
    assert EXCLUDED_TASKS, "至少应有一个按红线弃用的任务（bandgap chain_Egc）"
    for excluded in EXCLUDED_TASKS:
        assert "单位" in excluded.reason
        assert len(excluded.reason) > 40


@pytest.mark.skipif(
    not HARMONISED_MANIFEST_PATH.exists(), reason="还没跑过 scripts/01_harmonise.py；跳过产出验收"
)
def test_outputs_pass_acceptance_checks():
    """跑过阶段 01 之后，`verify_outputs()` 必须没有问题——这就是 CI 的回归闸门。
    Once stage 01 has run, verify_outputs() must be clean; this is the CI regression gate.

    中文：
        用途：把四条验收标准一次性跑完（决策覆盖、200 闸门、行数不变式、换算一致、哈希）。
        输入：无。
        输出：无（断言）。
        数据来源：data/manifests/ 下的清单与决策表、data/interim/harmonised/*.parquet。
        配置来源：无。副作用：无（只读）。异常：AssertionError —— 列出全部问题。

    English:
        Purpose: run all four acceptance criteria at once (decision coverage, the 200 gate, the
        row-count invariant, conversion consistency, hashes).
        Args: none. Returns: none (assertions).
        Data source: the manifests, the decision table and the parquet outputs.
        Config: none. Side effects: none. Raises: AssertionError listing every problem.
    """
    problems = verify_outputs()
    assert not problems, "\n".join(problems)
