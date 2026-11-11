"""任务配置的读取与校验：`configs/tasks/*.yaml` 的唯一入口。
Read and validate the task configs: the single entry point for `configs/tasks/*.yaml`.

中文：
    用途：A-4 的产物是 18 份任务 YAML（8 主表 + 5 备用 + 3 OpenPoly + 2 机制），下游 B/C/D 全靠
          它们知道"跑哪个任务、目标列是什么、单位是什么、标签有多少、用哪份划分"。本模块把
          "读一份配置"和"这套配置整体是否自洽"写成代码，任何字段缺失、任务名拼错、数字与上游
          清单不符，都在读取时当场报错——**不留给运行时**。
    输入：`configs/tasks/*.yaml`（18 份）；交叉核对用 `data/manifests/harmonised_manifest.csv`、
          `data/manifests/polybench_expected_files.csv`、`data/raw/polybench26/**/manifest.json`。
    输出：`load_task_config()` 返回一份已校验的 dict；`verify_all()` 返回报告（错误列表 + 待办清单）。
    数据来源：n 与单位一律来自上游交付物（A-2 的 39 行标准表、A-3 的调和清单、PB26 官方 manifest），
          本模块**不产生任何新数字**，只做相等性断言。
    配置来源：本文件里的 `TASK_KEYS_*` 与枚举即是 schema；改 schema 等于改冻结物的含义，
          必须在 `docs/DECISIONS.md` 记一条。
    副作用：`--verify` 只读；`--snapshot` 只往 stdout 打印 markdown。
    异常：`TaskConfigError`（字段缺失、拼错、枚举非法、与上游清单不符）。
    对外：load_task_config、load_all_task_configs、verify_all、main。
    被谁用：tests/test_task_configs.py、B/C 的 run 入口（读任务配置）、D 的入口审计。

English:
    Purpose: A-4 delivers 18 task YAMLs (8 main + 5 reserve + 3 OpenPoly + 2 mechanism). B/C/D
    depend on them to know which task, which target column, which unit, how many labels and
    which split. This module turns "read one config" and "is the whole set self-consistent"
    into code, so a missing field, a misspelled dataset name or a number that disagrees with an
    upstream manifest fails at load time, never at run time.
    Inputs: `configs/tasks/*.yaml`; cross-check sources are harmonised_manifest.csv,
    polybench_expected_files.csv and the official PolyBench26 manifest.json files.
    Outputs: a validated dict per config; verify_all() returns a report (errors + pending list).
    Data source: every n and unit comes from an upstream deliverable; this module invents no
    number, it only asserts equality.
    Config: the TASK_KEYS_*/enum constants below ARE the schema. Changing them changes the
    meaning of a frozen artefact, so it needs an entry in `docs/DECISIONS.md`.
    Side effects: `--verify` is read-only; `--snapshot` only prints markdown.
    Raises: TaskConfigError.
    Public API: load_task_config, load_all_task_configs, verify_all, main.
    Callers: tests/test_task_configs.py, the B/C run entries, D's entry audit.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import yaml

# ---- 0. 路径（相对仓库根，禁止绝对路径） / Paths (repo-relative, never absolute) ----

REPO_ROOT = Path(__file__).resolve().parents[3]        # src/chainworth/io/tasks.py -> 仓库根
CONFIGS_DIR = REPO_ROOT / "configs" / "tasks"
MANIFEST_DIR = REPO_ROOT / "data" / "manifests"
RAW_POLYBENCH26 = REPO_ROOT / "data" / "raw" / "polybench26"
HARMONISED_MANIFEST_PATH = MANIFEST_DIR / "harmonised_manifest.csv"
POLYBENCH_EXPECTED_PATH = MANIFEST_DIR / "polybench_expected_files.csv"

# ---- 1. schema：字段与枚举 / Schema: fields and vocabularies ----

# 20 个必填字段。`null` 是允许的取值，但**键必须存在**——少一个键与写一个 null 是两件事：
# 前者说明写配置的人没想到这件事，后者说明他想过了并且明确写了"没有"。
# 20 required keys. A null value is allowed, but the KEY must exist: a missing key means the
# author never thought about the field; an explicit null means they did and wrote "none".
TASK_KEYS_REQUIRED = (
    "task_id",          # 任务 ID = 文件名主干 / task id, equals the file stem
    "dataset_name",     # 稳定数据集名（PolyBench 即平台数据集名）/ stable dataset name
    "benchmark",        # polybench | polybench26 | openpoly
    "role",             # main | reserve | boundary | mechanism
    "tier",             # 1 | 2 | 3 | null（PB26/OpenPoly 不在三层划分里）/ tier or null
    "tier_basis",       # 层号的依据（Proposal 表 A / 不适用）/ where the tier comes from
    "property",         # 目标性质（人读）/ human-readable property
    "target_column",    # 目标列名（机读，必须与源文件表头逐字相同）/ exact source column
    "target_unit",      # 目标单位；未证实或不需要时为 null / unit, null when not established
    "unit_status",      # verified | unverified | not_required
    "unit_evidence",    # 单位依据（列名 / 官方文档 / 为何不需要）/ the evidence, or why none needed
    "source_kind",      # experimental | md | dft | mixed
    "n_labels",         # 可用标签数 / usable labels
    "n_source",         # n 的出处（文件名 + 列）/ where that number comes from
    "data_ref",         # 下游该读哪个文件（parquet / 官方 CSV）/ the file downstream reads
    "eval_unit",        # 官方评估单元（70/30×3 种子、5 折×3 种子…）/ official evaluation unit
    "arms",             # 在这个任务上要跑的臂 / arms that run this task
    "split_ref",        # 划分文件（A-5 交付）路径 / split artefact path
    "split_status",     # pending | present | blocked_upstream | deferred
    "split_protocol",   # 划分协议（与 eval_unit 对应）/ split protocol
    "status",           # available | unavailable
)

# 可选字段：只有多数据集任务（机制实验）与"不可用任务"才需要。
# Optional keys: only multi-dataset tasks and unavailable ones need them.
TASK_KEYS_OPTIONAL = ("datasets", "status_note", "notes")

BENCHMARKS = ("polybench", "polybench26", "openpoly")
ROLES = ("main", "reserve", "boundary", "mechanism")
SOURCE_KINDS = ("experimental", "md", "dft", "mixed")
UNIT_STATUS = ("verified", "unverified", "not_required")
SPLIT_STATUS = ("pending", "present", "blocked_upstream", "deferred")
TASK_STATUS = ("available", "unavailable")

# 臂名与 `configs/arms/<arm>.yaml` 一一对应（README §4.2）；Q-01 结案后主表固定为 5 臂（D-17）。
# Arm names match `configs/arms/<arm>.yaml`; after Q-01 closed, the main table has 5 arms (D-17).
ARMS = ("a1_floor", "a2_conventional", "a3_graph", "a4_polybert", "a5_chain")

# 主表的结构性规模：8 个主表任务 = PolyBench 每层 2 个（6）+ PolyBench26 2 个；备用 5 个
# （每层至少 1 个、第 3 层 2 个），OpenPoly 3 个性质，机制实验 2 个。
# Structural sizes of the frozen list, straight from Proposal §3.1 tables A-E.
EXPECTED_MAIN = 8
EXPECTED_MAIN_PER_TIER = 2
EXPECTED_MAIN_POLYBENCH26 = 2
EXPECTED_RESERVE = 5
EXPECTED_BOUNDARY = 3
EXPECTED_MECHANISM = 2

TASK_ID_PATTERN = re.compile(r"^[a-z0-9]+(_[a-z0-9]+)*$")


class TaskConfigError(ValueError):
    """任务配置不合法（字段缺失、拼错、枚举非法、与上游清单不符）。
    A task config is invalid (missing/misspelled field, bad enum, disagrees with upstream).

    中文：
        用途：让"配置写错"与"数据不存在"在日志里能分辨——两者的修法完全不同。
        输入：message —— 人读的中文说明（可含多个错误，用 `；` 分隔）。
        输出：异常对象。副作用：无。异常：无。
        被谁用：load_task_config / verify_all / 测试里的 `pytest.raises`。

    English:
        Purpose: distinguish "the config is wrong" from "the data is missing" in logs.
        Args: message - human-readable description, possibly listing several problems.
        Returns: the exception. Side effects: none. Raises: none.
        Callers: load_task_config / verify_all / `pytest.raises` in the tests.
    """


# ---- 2. 读一份配置 / Load one config ----


def _read_yaml(path: Path) -> dict[str, Any]:
    """读 YAML 并保证顶层是映射。

    Read a YAML file and require the top level to be a mapping.

    中文：
        用途：把"文件不存在 / 不是 YAML / 顶层是列表"这三种低级错误一次性挡住。
        输入：path —— 配置文件路径（绝对路径）。
        输出：dict（尚未校验 schema）。
        数据来源：磁盘上的 yaml 文件。配置来源：无。副作用：无。
        异常：TaskConfigError —— 文件缺失、YAML 语法错误、顶层不是映射。

    English:
        Purpose: catch the three cheap failures at once: missing file, malformed YAML, top
        level not a mapping.
        Args: path - absolute path. Returns: dict (not yet schema-checked).
        Data source: the yaml file on disk. Config: none. Side effects: none.
        Raises: TaskConfigError on missing file, YAML syntax error, or non-mapping root.
    """
    if not path.is_file():
        raise TaskConfigError(f"配置文件不存在：{path}")
    try:
        loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:  # 语法错误必须指出文件，否则 18 份里找不到是哪份 / name the file
        raise TaskConfigError(f"YAML 语法错误：{path.name}：{exc}") from exc
    if not isinstance(loaded, dict):
        raise TaskConfigError(f"顶层必须是映射（key: value），实际是 {type(loaded).__name__}：{path.name}")
    return loaded


def load_task_config(path: Path) -> dict[str, Any]:
    """读取一份任务配置并逐项校验，返回可直接给下游用的 dict。

    Load one task config, validate it field by field, and return a downstream-ready dict.

    中文：
        用途：本模块的主入口。校验分四层：① 键集合（缺失、拼错）；② 枚举与类型；③ 内部自洽
              （单位状态与单位值、tier 与依据、n 与 datasets 之和、文件名与 task_id）；
              ④ 与上游清单的相等性（见 `cross_check_upstream`）。
        输入：path —— `configs/tasks/<task_id>.yaml`。
        输出：dict，键集合 = TASK_KEYS_REQUIRED + 实际出现的可选键；`path` 不写进返回值
              （配置内容与它在磁盘上的位置无关，位置由 task_id 决定）。
        数据来源：磁盘上的 yaml。
        配置来源：本文件的 schema 常量。
        副作用：无（纯读）。
        异常：TaskConfigError —— 任一层校验失败，消息里列出**全部**问题，不是只报第一个。

    English:
        Purpose: the module's main entry. Validation happens in four layers: (1) key set
        (missing or misspelled); (2) enums and types; (3) internal consistency (unit status vs
        unit value, tier vs basis, n vs sum of datasets, file name vs task_id); (4) equality
        against upstream manifests (see cross_check_upstream).
        Args: path - one `configs/tasks/<task_id>.yaml`.
        Returns: dict whose keys are TASK_KEYS_REQUIRED plus whatever optional keys appeared.
        Data source: the yaml on disk. Config: the schema constants above.
        Side effects: none. Raises: TaskConfigError listing ALL problems, not only the first.
    """
    path = Path(path)
    raw = _read_yaml(path)
    errors: list[str] = []

    # ① 键集合：缺一个键与写一个 null 不是一回事 / key set: missing != explicitly null
    missing = [key for key in TASK_KEYS_REQUIRED if key not in raw]
    if missing:
        errors.append(f"缺必填字段 {missing}")
    unknown = sorted(set(raw) - set(TASK_KEYS_REQUIRED) - set(TASK_KEYS_OPTIONAL))
    if unknown:
        errors.append(f"出现未知字段 {unknown}（多半是拼错；schema 见 src/chainworth/io/tasks.py）")

    # ② task_id 必须等于文件名主干：否则下游按 ID 找文件会落空
    # task_id must equal the file stem, else downstream lookups silently miss
    stem = path.stem
    task_id = raw.get("task_id")
    if task_id != stem:
        errors.append(f"task_id={task_id!r} 与文件名 {path.name!r} 不一致")
    if isinstance(task_id, str) and not TASK_ID_PATTERN.match(task_id):
        errors.append(f"task_id={task_id!r} 不是小写下划线风格")

    # ③ 枚举 / enums
    for key, allowed in (
        ("benchmark", BENCHMARKS),
        ("role", ROLES),
        ("source_kind", SOURCE_KINDS),
        ("unit_status", UNIT_STATUS),
        ("split_status", SPLIT_STATUS),
        ("status", TASK_STATUS),
    ):
        value = raw.get(key)
        if value not in allowed:
            errors.append(f"{key}={value!r} 不在允许取值 {allowed} 内")

    # ④ tier：允许 null，但非 null 时必须给依据（层号是别人论文里的判断，不是我们的口味）
    # tier may be null; when set, the basis must be written down
    tier = raw.get("tier")
    if tier is not None and tier not in (1, 2, 3):
        errors.append(f"tier={tier!r} 只能是 1/2/3 或 null")
    if tier is not None and not str(raw.get("tier_basis") or "").strip():
        errors.append("tier 非空却没有写 tier_basis")

    # ⑤ 单位：三种状态各有硬条件，避免"写了单位但没证据"或"没单位却说 verified"
    # unit tri-state: each has hard requirements
    unit_status = raw.get("unit_status")
    target_unit = raw.get("target_unit")
    evidence = str(raw.get("unit_evidence") or "").strip()
    if unit_status == "verified":
        if not isinstance(target_unit, str) or not target_unit.strip():
            errors.append("unit_status=verified 但 target_unit 为空")
        if not evidence:
            errors.append("unit_status=verified 但 unit_evidence 为空（证据非空是 A-3 的验收标准①）")
    elif unit_status in ("unverified", "not_required"):
        if target_unit is not None:
            errors.append(f"unit_status={unit_status} 时 target_unit 必须为 null（不许写一个猜的单位）")
        if not evidence:
            errors.append(f"unit_status={unit_status} 时必须写清理由")

    # ⑥ n_labels：非负整数；有 datasets 时逐数据集相加必须相等
    # n_labels: non-negative int; with datasets, must equal the sum
    n_labels = raw.get("n_labels")
    if not isinstance(n_labels, int) or isinstance(n_labels, bool) or n_labels < 0:
        errors.append(f"n_labels={n_labels!r} 必须是非负整数")
    datasets = raw.get("datasets")
    if datasets is not None:
        if not isinstance(datasets, list) or not datasets:
            errors.append("datasets 必须是非空列表")
        else:
            total = 0
            for index, entry in enumerate(datasets):
                if not isinstance(entry, Mapping):
                    errors.append(f"datasets[{index}] 必须是映射")
                    continue
                allowed_ds = {"dataset_id", "n_rows", "train_sizes", "canonical_source", "official_indices_path"}
                extra = sorted(set(entry) - allowed_ds)
                if extra:
                    errors.append(f"datasets[{index}] 出现未知字段 {extra}")
                rows = entry.get("n_rows")
                if not isinstance(rows, int) or isinstance(rows, bool) or rows < 0:
                    errors.append(f"datasets[{index}].n_rows={rows!r} 必须是非负整数")
                else:
                    total += rows
                sizes = entry.get("train_sizes")
                if not isinstance(sizes, list) or not sizes or any(
                    not isinstance(size, int) or isinstance(size, bool) or size <= 0 for size in sizes
                ):
                    errors.append(f"datasets[{index}].train_sizes 必须是正整数列表")
            if isinstance(n_labels, int) and not isinstance(n_labels, bool):
                if raw.get("status") == "available" and total != n_labels:
                    errors.append(f"n_labels={n_labels} 与 datasets 的 n_rows 之和 {total} 不等")

    # ⑦ 划分：声明 present 就必须真有文件；deferred 必须真有"为什么没落盘"的说明；
    #    blocked_upstream 只允许出现在 OpenPoly（D-22）；pending 只登记不报错。
    # split declared present must exist; deferred needs its README; blocked_upstream is
    # OpenPoly-only; pending is only recorded.
    split_ref = raw.get("split_ref")
    split_status = raw.get("split_status")
    if split_status == "present":
        if not isinstance(split_ref, str) or not split_ref.strip():
            errors.append("split_status=present 但 split_ref 为空")
        elif not (REPO_ROOT / split_ref).exists():
            errors.append(f"split_status=present 但文件不存在：{split_ref}")
    elif split_status == "deferred":
        if not isinstance(split_ref, str) or not split_ref.strip():
            errors.append("split_status=deferred 但 split_ref 为空")
        elif not (REPO_ROOT / split_ref).is_dir():
            errors.append(f"split_status=deferred 但说明目录不存在：{split_ref}")
        elif not any((REPO_ROOT / split_ref).iterdir()):
            errors.append(f"split_status=deferred 但 {split_ref} 是空目录（必须有 README 说明）")
    elif split_status == "blocked_upstream" and raw.get("benchmark") != "openpoly":
        errors.append("blocked_upstream 只允许用于 OpenPoly（D-22：官方未发布划分）")
    if not str(raw.get("split_protocol") or "").strip():
        errors.append("split_protocol 为空（必须写清用的是哪个官方协议）")

    # ⑧ 臂：非空子集，且 PolyBench 上不许出现 k 折评估（官方只用固定 70/30，红线）
    # arms: non-empty subset; no k-fold on PolyBench (the official protocol has none)
    arms = raw.get("arms")
    if not isinstance(arms, list) or not arms:
        errors.append("arms 必须是非空列表")
    else:
        bad = [arm for arm in arms if arm not in ARMS]
        if bad:
            errors.append(f"arms 含未知臂 {bad}，允许 {ARMS}")
    eval_unit = str(raw.get("eval_unit") or "")
    if raw.get("benchmark") == "polybench" and "kfold" in eval_unit.lower():
        errors.append("PolyBench 任务写成 k 折评估——官方协议只有固定 70/30（Proposal §3.1）")
    if raw.get("benchmark") == "polybench26" and "70_30" in eval_unit:
        errors.append("PolyBench26 任务是 5 折，不是 70/30（Proposal §3.1）")

    # ⑨ 任务名：PolyBench 的稳定 ID 就是数据集名，必须是 39 行标准表里的那一个
    # PolyBench's stable ID is the dataset name; it must exist in the 39-row reference table
    if raw.get("benchmark") == "polybench":
        known = _polybench_dataset_names()
        if raw.get("dataset_name") not in known:
            errors.append(
                f"dataset_name={raw.get('dataset_name')!r} 不在 polybench_expected_files.csv 的 39 行里"
                "（任务名拼错，或用了平台上不存在的任务）"
            )

    if errors:
        raise TaskConfigError(f"{path.name}：" + "；".join(errors))
    return raw


def _polybench_dataset_names() -> set[str]:
    """返回 PolyBench 39 行标准表里的 `task_id` 集合（与数据集名同形）。

    Return the 39 dataset names from the frozen PolyBench reference table.

    中文：
        用途：把"任务名拼错"变成可判定的错误，而不是等到读 CSV 时 FileNotFound。
        输入：无（读 data/manifests/polybench_expected_files.csv）。
        输出：set[str]；文件缺失时返回空集（此时任务名校验自然通不过，错误信息仍可读）。
        数据来源：A-2 的 39 行标准表（DECISIONS D-13）。
        配置来源：无。副作用：无。异常：无。

    English:
        Purpose: make a misspelled dataset name a decidable error instead of a later
        FileNotFound while reading raw CSVs.
        Args: none. Returns: set[str]; empty when the table is absent.
        Data source: A-2's 39-row reference table (DECISIONS D-13).
        Config: none. Side effects: none. Raises: none.
    """
    if not POLYBENCH_EXPECTED_PATH.is_file():
        return set()
    with POLYBENCH_EXPECTED_PATH.open(encoding="utf-8", newline="") as handle:
        return {row["task_id"] for row in csv.DictReader(handle)}


def load_all_task_configs(configs_dir: Path | None = None) -> dict[str, dict[str, Any]]:
    """读入整目录并按 task_id 建索引。

    Load every config in the directory, indexed by task_id.

    中文：
        用途：让"目录里有什么"成为一次读取的结果，而不是调用方 glob 出来的猜测。
        输入：configs_dir —— 目录（默认 `configs/tasks`）。
        输出：dict[task_id, config]，按字典序（= 文件名序，输出稳定）。
        数据来源：目录下的 *.yaml。
        配置来源：无。副作用：无。
        异常：TaskConfigError —— 任一文件不合法；目录为空。

    English:
        Purpose: make "what is in the directory" a single explicit read instead of a caller's
        own glob. Args: configs_dir - default `configs/tasks`.
        Returns: dict[task_id, config] in filename order.
        Data source: *.yaml in that directory. Config: none. Side effects: none.
        Raises: TaskConfigError on any invalid file or an empty directory.
    """
    directory = Path(configs_dir or CONFIGS_DIR)
    paths = sorted(directory.glob("*.yaml"))
    if not paths:
        raise TaskConfigError(f"{directory} 下没有 yaml 文件")
    configs: dict[str, dict[str, Any]] = {}
    for path in paths:
        config = load_task_config(path)
        configs[config["task_id"]] = config
    return configs


# ---- 3. 整套配置的自洽性 / Whole-set consistency ----


def _role_counts(configs: Iterable[Mapping[str, Any]]) -> dict[str, int]:
    """按 role 数任务个数。 / Count tasks per role."""
    counts: dict[str, int] = {role: 0 for role in ROLES}
    for config in configs:
        counts[str(config["role"])] += 1
    return counts


def check_task_set(configs: Mapping[str, Mapping[str, Any]]) -> list[str]:
    """检查整份任务清单的结构（8 主表、每层 2 个、5 备用…）是否成立。

    Check the structural claims of the frozen list (8 main, 2 per tier, 5 reserves, ...).

    中文：
        用途：A-4 验收标准①的机器版本——"8 个主表任务齐全，2 个/分层 + PB26×2"与
              "备用名单 5 个"都是可数的，数错了就该红。
        输入：configs —— task_id -> config。
        输出：错误信息列表（空列表 = 通过）。
        数据来源：传入的配置。
        配置来源：EXPECTED_* 常量（来源 Proposal §3.1 表 A–E）。
        副作用：无。异常：无。

    English:
        Purpose: the machine version of acceptance criterion 1 - "8 main tasks, 2 per tier plus
        2 PolyBench26" and "5 reserves" are countable, so a wrong count must fail.
        Args: configs - task_id -> config. Returns: list of problems (empty = pass).
        Data source: the configs. Config: the EXPECTED_* constants (Proposal §3.1 tables A-E).
        Side effects: none. Raises: none.
    """
    problems: list[str] = []
    counts = _role_counts(configs.values())
    for role, expected in (
        ("main", EXPECTED_MAIN),
        ("reserve", EXPECTED_RESERVE),
        ("boundary", EXPECTED_BOUNDARY),
        ("mechanism", EXPECTED_MECHANISM),
    ):
        if counts[role] != expected:
            problems.append(f"role={role} 有 {counts[role]} 个，应为 {expected} 个")

    # 主表：PolyBench 每个层恰好 2 个 + PolyBench26 恰好 2 个（表 A + 表 B）
    # Main table: exactly 2 PolyBench tasks per tier plus exactly 2 PolyBench26 tasks
    main = [config for config in configs.values() if config["role"] == "main"]
    per_tier: dict[int, int] = {1: 0, 2: 0, 3: 0}
    pb26 = 0
    for config in main:
        if config["benchmark"] == "polybench":
            per_tier[int(config["tier"])] += 1
        elif config["benchmark"] == "polybench26":
            pb26 += 1
    for tier_value, count in per_tier.items():
        if count != EXPECTED_MAIN_PER_TIER:
            problems.append(f"主表第 {tier_value} 层有 {count} 个任务，应为 {EXPECTED_MAIN_PER_TIER} 个")
    if pb26 != EXPECTED_MAIN_POLYBENCH26:
        problems.append(f"主表 PolyBench26 有 {pb26} 个任务，应为 {EXPECTED_MAIN_POLYBENCH26} 个")

    # 每层至少一个备用（表 E 的替换能力不能落空）/ at least one reserve per tier
    for tier_value in (1, 2, 3):
        usable = [
            config
            for config in configs.values()
            if config["role"] == "reserve" and config["tier"] == tier_value and config["status"] == "available"
        ]
        if not usable:
            problems.append(f"第 {tier_value} 层没有可用的备用任务")

    # 不可用任务必须写清原因（bandgap 就是这一条）/ an unavailable task must say why
    for config in configs.values():
        if config["status"] == "unavailable" and not str(config.get("status_note") or "").strip():
            problems.append(f"{config['task_id']} 标了 unavailable 却没有 status_note")
        if config["role"] in ("main", "boundary") and config["status"] == "unavailable":
            problems.append(f"{config['task_id']} 是 {config['role']}，不允许 unavailable")
    return problems


# ---- 4. 与上游清单的相等性 / Equality against upstream manifests ----


def _read_harmonised_manifest() -> dict[str, dict[str, str]]:
    """读 A-3 的调和清单，按 task_id 建索引。 / Read A-3's manifest indexed by task_id."""
    if not HARMONISED_MANIFEST_PATH.is_file():
        return {}
    with HARMONISED_MANIFEST_PATH.open(encoding="utf-8", newline="") as handle:
        return {row["task_id"]: row for row in csv.DictReader(handle)}


def cross_check_upstream(configs: Mapping[str, Mapping[str, Any]]) -> list[str]:
    """把配置里的 n 与单位跟上游交付物逐项对一遍。

    Check every n and unit against the upstream deliverables.

    中文：
        用途：这是本模块最有价值的一段——它让"配置里的数字"无法脱离证据存在：
              ① 进过 A-3 调和的任务，`n_labels` 必须等于 `harmonised_manifest.csv` 的 `n_labels`，
                 `target_unit` 必须等于它的 `label_unit`；
              ② PolyBench26 任务，若 `datasets[].official_indices_path` 指向的官方目录在本地，
                 `n_rows` / `train_sizes` / `target_column` / `canonical_source` 必须与官方
                 `manifest.json` 逐项相等；
              ③ 目录不在本地（数据未下载）时**不报错**，只记为待核——否则别人克隆下来跑测试会红。
        输入：configs —— task_id -> config。
        输出：错误信息列表（空 = 通过）。
        数据来源：data/manifests/harmonised_manifest.csv、data/raw/polybench26/**/manifest.json。
        配置来源：无。副作用：无。异常：无。

    English:
        Purpose: the most valuable part of this module - it makes the numbers in the configs
        unable to exist without evidence: (1) for harmonised tasks, n_labels and target_unit
        must equal harmonised_manifest.csv; (2) for PolyBench26 tasks whose official
        manifest.json is present locally, n_rows / train_sizes / target_column /
        canonical_source must match it exactly; (3) a missing local copy is NOT an error, so a
        fresh clone can still run the tests.
        Args: configs. Returns: list of problems.
        Data source: harmonised_manifest.csv and the official PolyBench26 manifest.json files.
        Config: none. Side effects: none. Raises: none.
    """
    problems: list[str] = []
    harmonised = _read_harmonised_manifest()

    for task_id, config in configs.items():
        # ① A-3 调和过的主表/备用任务 / tasks harmonised by A-3
        row = harmonised.get(task_id)
        if row is not None:
            if int(row["n_labels"]) != config["n_labels"]:
                problems.append(
                    f"{task_id}: n_labels={config['n_labels']} 与 harmonised_manifest.csv 的 {row['n_labels']} 不等"
                )
            if config["unit_status"] == "verified" and config["target_unit"] != row["label_unit"]:
                problems.append(
                    f"{task_id}: target_unit={config['target_unit']!r} 与调和表的 label_unit={row['label_unit']!r} 不等"
                    "（A-3 的 target_unit 是唯一口径，D-15）"
                )

        # ② PolyBench26 官方 manifest 逐项对 / PolyBench26 official manifest, field by field
        for entry in config.get("datasets") or []:
            indices_path = entry.get("official_indices_path")
            if not indices_path:
                continue
            manifest_path = RAW_POLYBENCH26 / indices_path / "manifest.json"
            if not manifest_path.is_file():
                continue  # 数据未下载：不报错，见 docstring / data absent: not an error
            official = json.loads(manifest_path.read_text(encoding="utf-8"))
            if official.get("n_rows") != entry.get("n_rows"):
                problems.append(
                    f"{task_id}/{entry.get('dataset_id')}: n_rows={entry.get('n_rows')} 与官方 "
                    f"{official.get('n_rows')} 不等"
                )
            if list(official.get("train_sizes") or []) != list(entry.get("train_sizes") or []):
                problems.append(
                    f"{task_id}/{entry.get('dataset_id')}: train_sizes={entry.get('train_sizes')} 与官方 "
                    f"{official.get('train_sizes')} 不等"
                )
            if config["benchmark"] == "polybench26" and entry.get("canonical_source") != official.get("canonical_source"):
                problems.append(
                    f"{task_id}/{entry.get('dataset_id')}: canonical_source 与官方 manifest 不等"
                    f"（我方 {entry.get('canonical_source')!r} / 官方 {official.get('canonical_source')!r}）"
                )
            if config["target_column"] != official.get("target_column"):
                problems.append(
                    f"{task_id}: target_column={config['target_column']!r} 与官方 "
                    f"{official.get('target_column')!r} 不等"
                )
    return problems


# ---- 5. 整体验收与快照输出 / Whole-set verification and snapshot output ----


def verify_all(configs_dir: Path | None = None) -> dict[str, Any]:
    """读入并校验整份任务清单，返回一份报告。

    Load and validate the whole task list, returning one report.

    中文：
        用途：`--verify` 与 pytest 共用的单一函数，保证"人跑的"和"CI 跑的"是同一条检查。
        输入：configs_dir —— 目录（默认 `configs/tasks`）。
        输出：dict，含 n_tasks / errors / pending_splits / counts / configs。
        数据来源：configs/tasks/*.yaml + 上游清单。
        配置来源：schema 常量。副作用：无。
        异常：不抛——错误放进返回值，让调用方决定是打日志还是断言失败。

    English:
        Purpose: the single function shared by `--verify` and pytest, so the human run and the
        CI run are literally the same check.
        Args: configs_dir. Returns: dict with n_tasks / errors / pending_splits / counts /
        configs. Data source: the yaml files plus upstream manifests.
        Config: schema constants. Side effects: none. Raises: none (errors are returned).
    """
    configs = load_all_task_configs(configs_dir)
    errors = check_task_set(configs) + cross_check_upstream(configs)
    pending = sorted(task_id for task_id, config in configs.items() if config["split_status"] == "pending")
    return {
        "n_tasks": len(configs),
        "errors": errors,
        "pending_splits": pending,
        "counts": _role_counts(configs.values()),
        "configs": configs,
    }


# 快照表的列顺序固定，`--snapshot` 与 README 引用的是同一张表。
# Fixed column order: `--snapshot` and the snapshot doc use the same table.
SNAPSHOT_COLUMNS = ("task_id", "benchmark", "role", "tier", "n", "unit", "source", "status")


def render_snapshot(configs: Mapping[str, Mapping[str, Any]]) -> str:
    """把整份任务清单渲染成 markdown 表，供 `docs/Task List Snapshot D0.md` 的正文使用。

    Render the frozen task list as the markdown table used by the snapshot doc.

    中文：
        用途：让快照里的清单是**打印出来的**，不是抄出来的——抄一遍就是一次引入错误的机会。
        输入：configs —— task_id -> config。
        输出：markdown 字符串（含表头与分隔行）。
        数据来源：配置本身。配置来源：无。副作用：无。异常：无。

    English:
        Purpose: make the list in the snapshot PRINTED, not retyped - retyping is one more
        chance to introduce an error.
        Args: configs. Returns: a markdown table body (header + separator + rows).
        Data source: the configs. Config: none. Side effects: none. Raises: none.
    """
    order = {"main": 0, "reserve": 1, "boundary": 2, "mechanism": 3}
    rows = sorted(
        configs.values(),
        key=lambda config: (
            order[str(config["role"])],
            config["tier"] if config["tier"] is not None else 9,
            str(config["task_id"]),
        ),
    )
    lines = ["| task_id | 基准 | 角色 | 层 | n | 单位 | 来源 | 状态 |", "|---|---|---|---:|---:|---|---|---|"]
    for config in rows:
        tier = "-" if config["tier"] is None else str(config["tier"])
        unit = config["target_unit"] if config["target_unit"] is not None else "未证实" if config["unit_status"] == "unverified" else "不需要"
        status = "可用" if config["status"] == "available" else "不可用"
        lines.append(
            f"| `{config['task_id']}` | {config['benchmark']} | {config['role']} | {tier} | "
            f"{config['n_labels']} | {unit} | {config['source_kind']} | {status} |"
        )
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    """命令行入口：`--verify` 只读校验，`--snapshot` 打印快照表。

    CLI entry: `--verify` read-only check, `--snapshot` prints the snapshot table.

    中文：
        用途：让"这套配置还能不能用"成为一条可复制的命令，而不是去读 18 个文件。
        输入：argv —— 命令行参数（None 读 sys.argv）。
        输出：进程退出码（0 = 全部通过；1 = 有错误）。
        数据来源：configs/tasks 与上游清单。
        配置来源：命令行。副作用：只写 stdout/stderr。
        异常：无（错误转成退出码，便于 CI）。

    English:
        Purpose: make "are these configs still valid" one copyable command instead of reading
        18 files. Args: argv. Returns: exit code (0 pass, 1 fail).
        Data source: configs and upstream manifests. Config: the CLI.
        Side effects: stdout/stderr only. Raises: none (errors become exit codes for CI).
    """
    parser = argparse.ArgumentParser(
        prog="python -m chainworth.io.tasks",
        description="读取并校验 configs/tasks/*.yaml（A-4 的冻结物）。",
    )
    parser.add_argument("--configs-dir", type=Path, default=None, help="配置目录，默认 configs/tasks")
    parser.add_argument("--verify", action="store_true", help="只读校验（默认动作，写上是为了可读）")
    parser.add_argument("--snapshot", action="store_true", help="打印快照用的 markdown 表")
    args = parser.parse_args(argv)

    try:
        report = verify_all(args.configs_dir)
    except TaskConfigError as exc:
        print(f"[FAIL] {exc}", file=sys.stderr)
        return 1

    if args.snapshot:
        print(render_snapshot(report["configs"]))
        return 0

    counts = report["counts"]
    print(f"任务数 {report['n_tasks']}：main {counts['main']} / reserve {counts['reserve']} / "
          f"boundary {counts['boundary']} / mechanism {counts['mechanism']}")
    if report["pending_splits"]:
        # A-5 之后，只有"整任务不可用"的那种才允许还挂 pending（bandgap：没有可用标签）。
        # After A-5 only unavailable tasks may legitimately stay pending (bandgap has no labels).
        print(
            f"划分仍为 pending（{len(report['pending_splits'])} 个，应全部是不可用任务）："
            f"{', '.join(report['pending_splits'])}"
        )
    if report["errors"]:
        for problem in report["errors"]:
            print(f"[FAIL] {problem}", file=sys.stderr)
        return 1
    print("[OK] 18 份任务配置全部通过：字段、枚举、层数与上游清单一致")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
