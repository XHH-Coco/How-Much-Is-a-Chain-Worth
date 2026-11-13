"""运行记录审计：全队每一条 run 记录的守门测试（C-c-4 的套件部分）。
Run-record audit: the guard tests every team run record must pass (C-c-4's suite half).

中文：
    用途：母本 C-6 说"run 记录错了不会抛异常，只会让结论对不上账"。这套测试把
          `results/raw_runs/` 的每一条记录过一遍第 0 条 + 六条断言，并用手写的四处
          坏样本证明"注入错误一定变红"（验收①）。跑套件的人是 C-b，写套件的人是 C-c，
          分开是为了防止为了跑绿而放松断言。
    依赖：`results/raw_runs/_schema.json`（字段契约唯一出处）、`results/raw_runs/*.json`、
          `results/aggregate/predictions/*.parquet`、`splits/**`（只读，判预测行数）、
          `configs/tasks/*.yaml`（经 `chainworth.io.loaders.load_split` 解析折）。
    配置：无（断言清单来自 `_schema.json` 与 `docs/分工_C-c_迁移主跑与主表审计.md` §2 C-c-4）。
    对外：`audit_run_records()`（供注入测试与未来的 CI 复用）。
    被谁用：C-b 每批收尾时跑、C-c 定稿时跑；D 的出口审计会引用它的结论。

    jsonschema 说明：任务书写"用 jsonschema 校验 _schema.json"，但三套冻结环境（P0）里都
    没有 jsonschema，且现有代码（`chainworth.arms.graph.record_problems`）刻意不引入该依赖。
    经用户 2026-11-13 拍板：本模块内置一个 draft-07 **子集**校验器，字段清单与约束全部
    读自 `_schema.json`（不手写），子集范围由 `SUPPORTED_KEYWORDS` 声明，并有测试守着
    "schema 只能使用受支持关键字"。

English:
    Purpose: C-6 says a bad run record raises nothing and only makes the numbers unreconcilable.
    These tests push every record under results/raw_runs/ through rule 0 plus six assertions, and
    four hand-built bad samples prove injected defects always turn red (criterion 1). C-b runs the
    suite, C-c wrote it, so nobody is grading their own homework.
    Depends on: results/raw_runs/_schema.json (the only field contract), results/raw_runs/*.json,
    results/aggregate/predictions/*.parquet, splits/** (read-only, for row counts) and
    configs/tasks/*.yaml (parsed through chainworth.io.loaders.load_split).
    Config: none (the assertion list comes from _schema.json and division C-c section 2, C-c-4).
    Public API: audit_run_records() (reused by the injection tests and any future CI).
    Callers: C-b at every batch close, C-c at finalisation; D's exit audit cites its verdict.

    On jsonschema: the task brief says "validate against _schema.json with jsonschema", but none
    of the three frozen P0 environments has jsonschema and the existing code deliberately avoids
    the dependency. Per the user's decision on 2026-11-13 this module ships a draft-07 **subset**
    validator; the field list and constraints are read from _schema.json (never hand-written),
    the subset is declared in SUPPORTED_KEYWORDS, and a test pins the schema to those keywords.
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any, Callable

import pytest

from chainworth.io import loaders as L

# ---- 常量区：仓库根与三个被审计的目录 / Constants: repo root and the audited locations ----

REPO = Path(__file__).resolve().parents[1]
RAW_RUNS = REPO / "results" / "raw_runs"
PREDICTIONS = REPO / "results" / "aggregate" / "predictions"
SCHEMA_PATH = RAW_RUNS / "_schema.json"

# `_schema.json` 不是记录，统计与扫描都要排除（任务书 §7 常见坑最后一条）。
# _schema.json is not a record; every count and scan must exclude it (brief, section 7, last pitfall).
_SCHEMA_FILENAME = "_schema.json"

# ① 的字段清单来自任务书 §2 C-c-4 验收①：基础七项 + status=ok 的附加六项。
# ①'s field list comes from the brief, section 2 C-c-4, criterion 1: seven base fields
# plus six extra fields for status=ok.
REQUIRED_BASE_FIELDS = (
    "run_id",
    "task",
    "arm",
    "seed",
    "split_version",
    "config_hash",
    "env_fingerprint",
)
REQUIRED_OK_FIELDS = (
    "metrics",
    "wall_clock_s",
    "device",
    "predictions",
    "n_train",
    "n_test",
)

# ⑥：device 与环境指纹的对应（任务书 §2 C-c-4 断言⑥）。
# ⑥: device to env fingerprint mapping (brief, section 2 C-c-4, assertion 6).
DEVICE_ENV_MAP = {
    "cuda": ("env_gnn.json", "env_polybert.json"),
    "cpu": ("env_core.json",),
}

# `*.partial.*` 一律不许进 results/raw_runs/（任务书 §2 C-c-4 断言⑤；用户组规第三条）。
# Anything named *.partial.* must never land in results/raw_runs/ (assertion 5; team rule 3).
PARTIAL_MARKER = ".partial."

# ---- 1. draft-07 子集校验器 / The draft-07 subset validator ----

# 只支持这些关键字；测试 `test_schema_uses_only_supported_keywords` 守着 schema 不越界。
# Only these keywords are supported; a test pins _schema.json to this set.
SUPPORTED_KEYWORDS = {
    "$schema",
    "$id",
    "title",
    "description",
    "type",
    "required",
    "properties",
    "pattern",
    "enum",
    "minimum",
    "const",
    "allOf",
    "if",
    "then",
    "additionalProperties",
}

_TYPE_CHECKERS: dict[str, Callable[[Any], bool]] = {
    "object": lambda v: isinstance(v, dict),
    "string": lambda v: isinstance(v, str),
    "integer": lambda v: isinstance(v, int) and not isinstance(v, bool),
    "number": lambda v: isinstance(v, (int, float)) and not isinstance(v, bool),
    "boolean": lambda v: isinstance(v, bool),
    "array": lambda v: isinstance(v, list),
}


def _schema_problems(instance: Any, schema: dict[str, Any], path: str) -> list[str]:
    """按 `_schema.json` 的 draft-07 子集校验一个值，收集问题（不抛错）。
    Validate one value against the draft-07 subset of _schema.json, collecting problems.

    中文：
        用途：第 0 条的实现。只实现 `_schema.json` 现在用到的关键字；每个问题带 JSON 路径，
              让人能直接定位到出错的字段。`pattern` 按 JSON Schema 语义用**非锚定**匹配
              （`re.search`），因为官方正则不要求从头到尾。
        输入：instance —— 要校验的值；schema —— 子 schema；path —— 当前 JSON 路径。
        输出：问题字符串列表；空列表表示合格。
        数据来源：调用方。配置来源：`_schema.json`（本函数只解释它）。副作用：无。
        异常：无（未知关键字直接忽略，越界由 `test_schema_uses_only_supported_keywords` 拦）。

    English:
        Purpose: implements rule 0. Only the keywords _schema.json actually uses are supported;
        each problem carries the JSON path. `pattern` follows JSON Schema semantics with an
        unanchored search, since the official regex is not implicitly anchored.
        Args: instance; schema; path. Returns: problem strings; empty means valid.
        Data source: the caller. Config: _schema.json (this function only interprets it).
        Side effects: none. Raises: none (unknown keywords are ignored; the keyword test guards).
    """
    problems: list[str] = []
    expected = schema.get("type")
    if expected is not None:
        checker = _TYPE_CHECKERS.get(expected)
        if checker is None or not checker(instance):
            problems.append(f"{path}: 类型应为 {expected}，实际 {type(instance).__name__}")
            return problems
    if isinstance(instance, dict):
        for key in schema.get("required", []):
            if key not in instance:
                problems.append(f"{path}: 缺必填字段 `{key}`")
        for key, sub in (schema.get("properties") or {}).items():
            if key in instance:
                problems.extend(_schema_problems(instance[key], sub, f"{path}.{key}"))
    if isinstance(instance, str):
        pattern = schema.get("pattern")
        if pattern is not None and re.search(pattern, instance) is None:
            problems.append(f"{path}: {instance!r} 不匹配 pattern {pattern!r}")
    enum = schema.get("enum")
    if enum is not None and instance not in enum:
        problems.append(f"{path}: {instance!r} 不在枚举 {enum} 里")
    minimum = schema.get("minimum")
    if minimum is not None and isinstance(instance, (int, float)) and instance < minimum:
        problems.append(f"{path}: {instance} < minimum {minimum}")
    const = schema.get("const")
    if "const" in schema and instance != const:
        problems.append(f"{path}: 应为 {const!r}，实际 {instance!r}")
    for sub in schema.get("allOf", []):
        problems.extend(_schema_problems(instance, sub, path))
    if_schema = schema.get("if")
    if if_schema is not None and not _schema_problems(instance, if_schema, path):
        problems.extend(_schema_problems(instance, schema.get("then", {}), path))
    return problems


# ---- 2. 记录读取与六条断言 / Record loading and the six assertions ----


def _load_records(raw_dir: Path) -> tuple[dict[str, Any], list[str]]:
    """读一个 raw_runs 目录：每条 JSON 一行键（按路径排序），顺带报告解码失败。
    Read one raw_runs directory into path -> record (sorted), reporting decode failures.

    中文：
        用途：审计的唯一入口，`_schema.json` 在这里被排除。文件名排序让"同 run_id 两条"
              的报告顺序稳定。JSON 解码失败本身就是审计问题（记录损坏 = 对不上账）。
        输入：raw_dir —— 形如 results/raw_runs 的目录。输出：(路径→记录, 问题列表)。
        数据来源：raw_dir 下的 *.json。配置来源：无。副作用：只读。
        异常：无（解码失败收集为问题，不中断）。

    English:
        Purpose: the audit's single entry point; _schema.json is excluded here. Sorting by path
        keeps duplicate reports stable, and a JSON decode failure is itself a finding.
        Args: raw_dir. Returns: (path -> record, problems). Data source: *.json under raw_dir.
        Side effects: read-only. Raises: none.
    """
    records: dict[str, Any] = {}
    problems: list[str] = []
    for path in sorted(raw_dir.glob("*.json")):
        if path.name == _SCHEMA_FILENAME:
            continue
        try:
            records[path.name] = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError, OSError) as exc:
            problems.append(f"{path.name}: JSON 读不回来（{exc}）")
    return records, problems


def _audit_required_fields(record: dict[str, Any], name: str) -> list[str]:
    """断言①：字段齐全（任务书 §2 C-c-4 的七项 + ok 六项）。
    Assertion 1: field completeness (the brief's seven base fields plus six for ok).

    中文：
        用途：与第 0 条（schema）**互相独立**——这条的清单来自任务书原文，即使有人改松了
              `_schema.json` 的 `required`，这条仍会拦。输入：record；name —— 文件名。
        输出：问题列表。数据来源：调用方。配置来源：任务书 §2 C-c-4。副作用：无。异常：无。

    English:
        Purpose: deliberately independent of rule 0 - the list comes verbatim from the brief, so
        loosening _schema.json's `required` does not weaken this. Args: record; name.
        Returns: problems. Data source: the caller. Config: brief section 2 C-c-4. Side effects:
        none. Raises: none.
    """
    problems: list[str] = []
    for field in REQUIRED_BASE_FIELDS:
        if field not in record:
            problems.append(f"{name}: ① 缺必填字段 `{field}`（任务书 §2 C-c-4）")
    if record.get("status") == "ok":
        for field in REQUIRED_OK_FIELDS:
            if field not in record:
                problems.append(f"{name}: ① status=ok 缺字段 `{field}`（任务书 §2 C-c-4）")
    return problems


def _split_spec_cache() -> dict[str, Any]:
    """按任务缓存 `load_split` 的结果（模块内只算一次）。
    Cache load_split per task (computed once per module).
    """
    cache: dict[str, Any] = {}
    return cache


def _resolve_fold(
    record: dict[str, Any], name: str, cache: dict[str, Any]
) -> tuple[Any, list[str]]:
    """把一条记录解析到它那一折：split_version 指向的文件必须在，折名必须找得到。
    Resolve one record to its fold: the split_version file must exist and the fold must resolve.

    中文：
        用途：断言③的地基。`split_version` 的前 12 位必须是**文件字节**的 sha256 前缀——
              手写版本号在文件被换时不会变（任务书 §2 C-c-4 与 `_schema.json` 的契约）。
              指向目录时（scaling 等多文件划分）跳过字节校验，只查存在。
        输入：record；name —— 文件名；cache —— 任务级 `load_split` 缓存。
        输出：(FoldSpec, 问题列表)。数据来源：`splits/**`。配置来源：`configs/tasks/`。
        副作用：无。异常：`load_split` 抛 AssemblyError 时收集为问题。

    English:
        Purpose: assertion 3's foundation. The first 12 hex chars of split_version must be the
        sha256 prefix of the file bytes - a hand-written version does not change when the file is
        swapped. Directory references (multi-file splits such as scaling) get an existence check
        only. Args: record; name; cache. Returns: (FoldSpec, problems). Data source: splits/**.
        Config: configs/tasks/. Side effects: none. Raises: none (AssemblyError becomes a problem).
    """
    problems: list[str] = []
    version = str(record.get("split_version", ""))
    parts = version.split("@", 1)
    if len(parts) != 2:
        return None, [f"{name}: split_version {version!r} 不是 `<12 位哈希>@splits/...`"]
    prefix, rel = parts
    split_path = REPO / rel
    if not split_path.exists():
        return None, [f"{name}: split_version 指向的 {rel} 不存在"]
    if split_path.is_file():
        digest = hashlib.sha256(split_path.read_bytes()).hexdigest()[:12]
        if prefix != digest:
            return None, [
                f"{name}: split_version 哈希前缀 {prefix} ≠ 文件字节 {digest}（{rel} 被换过？）"
            ]
    task = str(record.get("task", ""))
    try:
        spec = cache.setdefault(task, L.load_split(task))
    except Exception as exc:  # load_split 的 AssemblyError 分支，统一收集
        return None, [f"{name}: 读划分失败（{exc}）"]
    fold_name = str(record.get("split_fold", ""))
    for fold in spec.folds:
        if fold.name == fold_name:
            return fold, problems
    return None, [f"{name}: 折 {fold_name!r} 不在 {task} 的 {spec.fold_names} 里"]


def _audit_predictions(record: dict[str, Any], name: str, pred_dir: Path) -> list[str]:
    """断言②：status=ok 的 run 必须有预测文件（路径与 run_id 一致）。
    Assertion 2: every ok run has its predictions file, path consistent with the run_id.

    中文：
        用途：没有预测文件 D 算不出 bootstrap 区间；路径必须是 `predictions/<run_id>.parquet`，
              任何其他路径（改名、挪目录）都算问题。输入：record；name；pred_dir。
        输出：问题列表。数据来源：pred_dir。配置来源：`_schema.json` 的 `predictions` 契约。
        副作用：只读。异常：无（读取失败收集为问题）。

    English:
        Purpose: without predictions D cannot bootstrap, and the path must be
        predictions/<run_id>.parquet - any rename or move is a finding. Args: record; name;
        pred_dir. Returns: problems. Data source: pred_dir. Config: the predictions contract in
        _schema.json. Side effects: read-only. Raises: none.
    """
    problems: list[str] = []
    if record.get("status") != "ok":
        return problems
    run_id = str(record.get("run_id", ""))
    expected = f"results/aggregate/predictions/{run_id}.parquet"
    if str(record.get("predictions", "")) != expected:
        problems.append(f"{name}: predictions 字段应为 {expected}，实际 {record.get('predictions')!r}")
        return problems
    if not (pred_dir / f"{run_id}.parquet").is_file():
        problems.append(f"{name}: 缺预测文件 {expected}")
    return problems


def _audit_prediction_rows(
    record: dict[str, Any], name: str, pred_dir: Path, cache: dict[str, Any]
) -> list[str]:
    """断言③：预测行数 = 该折 test 长度，且 parquet 是 `row_id, y_pred` 两列。
    Assertion 3: prediction rows equal that fold's test length; the parquet has row_id, y_pred.

    中文：
        用途：行数对不上 = 预测来自错误的划分或错误的折（Proposal §7.5 第 (i) 类失败）。
              两列契约来自 `_schema.json` 的 `predictions` 字段描述。非 ok 记录没有预测，跳过。
        输入：record；name；pred_dir；cache —— 折缓存。输出：问题列表。
        数据来源：pred_dir 与 `splits/**`。配置来源：`_schema.json`。副作用：只读。
        异常：无（parquet 读失败收集为问题）。

    English:
        Purpose: a row mismatch means predictions from the wrong split or fold (Proposal section
        7.5, failure mode (i)); the two-column contract comes from _schema.json. Non-ok records
        have no predictions and are skipped. Args: record; name; pred_dir; cache. Returns:
        problems. Data source: pred_dir and splits/**. Config: _schema.json. Side effects:
        read-only. Raises: none.
    """
    problems: list[str] = []
    if record.get("status") != "ok":
        return problems
    import pandas as pd

    run_id = str(record.get("run_id", ""))
    fold, fold_problems = _resolve_fold(record, name, cache)
    problems.extend(fold_problems)
    if fold is None:
        return problems
    path = pred_dir / f"{run_id}.parquet"
    if not path.is_file():
        return problems  # ② 已报；这里只查行数与列
    try:
        frame = pd.read_parquet(path)
    except Exception as exc:  # 损坏的 parquet 是审计问题，不是测试崩溃
        problems.append(f"{name}: 预测 parquet 读不回来（{exc}）")
        return problems
    if list(frame.columns) != ["row_id", "y_pred"]:
        problems.append(f"{name}: 预测 parquet 列应为 [row_id, y_pred]，实际 {frame.columns.tolist()}")
    expected = int(len(fold.test_index))
    if len(frame) != expected:
        problems.append(
            f"{name}: 预测 {len(frame)} 行 ≠ 折 {fold.name} 的 test {expected} 行"
        )
    return problems


def _audit_unique_ids(filenames: dict[str, Any]) -> list[str]:
    """断言④：同一 run_id 不出现两次，且文件名必须等于 run_id。
    Assertion 4: no run_id appears twice, and each file name equals its run_id.

    中文：
        用途：`ls results/raw_runs/*.json` 要对账（任务书 §7），文件名 ≠ run_id 会让计数
              与点回都错位；重复 run_id = 两份记录争一个预测文件。输入：filenames ——
              文件名 → 记录。输出：问题列表。数据来源：调用方。配置来源：`_schema.json`
              的 run_id 描述。副作用：无。异常：无。

    English:
        Purpose: reconciliation counts *.json by name, so a mismatched name breaks both the count
        and every click-back; a duplicate run_id means two records fighting over one predictions
        file. Args: filenames (name -> record). Returns: problems. Data source: the caller.
        Config: the run_id description in _schema.json. Side effects: none. Raises: none.
    """
    problems: list[str] = []
    seen: dict[str, str] = {}
    for name, record in filenames.items():
        run_id = str(record.get("run_id", ""))
        if run_id in seen:
            problems.append(f"{name}: run_id {run_id!r} 与 {seen[run_id]} 重复（④）")
        else:
            seen[run_id] = name
        if run_id and name != f"{run_id}.json":
            problems.append(f"{name}: 文件名应等于 {run_id}.json（_schema.json 的 run_id 契约）")
    return problems


def _audit_partials(raw_dir: Path) -> list[str]:
    """断言⑤：`*.partial.*` 一律不许出现在 raw_runs 里。
    Assertion 5: nothing named *.partial.* may sit in raw_runs.

    中文：
        用途：`--limit` 试跑的产物只许叫 `*.partial.*`（组规第三条），混进全量目录一次，
              主表里就有调试数。输入：raw_dir。输出：问题列表。数据来源：raw_dir。
        配置来源：无。副作用：只读。异常：无。

    English:
        Purpose: --limit trial artefacts must stay named *.partial.* (team rule 3); one leak into
        the full-run directory puts debug numbers into the main table. Args: raw_dir. Returns:
        problems. Data source: raw_dir. Config: none. Side effects: read-only. Raises: none.
    """
    problems: list[str] = []
    for path in sorted(raw_dir.iterdir()):
        if PARTIAL_MARKER in path.name:
            problems.append(f"{path.name}: `*.partial.*` 不许进 {raw_dir.name}（⑤）")
    return problems


def _audit_device_env(record: dict[str, Any], name: str) -> list[str]:
    """断言⑥：device 与环境指纹对得上（cuda↔gnn/polybert，cpu↔core）。
    Assertion 6: the device matches the environment fingerprint.

    中文：
        用途：device 是训练脚本自动填的（红线 5）；两者对不上 = 要么记录被手改，要么
              脚本在错的环境里跑。非 ok 记录没有 device，跳过。输入：record；name。
        输出：问题列表。数据来源：调用方。配置来源：任务书 §2 C-c-4 断言⑥。副作用：无。
        异常：无。

    English:
        Purpose: device is written by the training script (red line 5); a mismatch means either a
        hand-edited record or a script that ran in the wrong environment. Non-ok records carry no
        device and are skipped. Args: record; name. Returns: problems. Data source: the caller.
        Config: brief section 2 C-c-4, assertion 6. Side effects: none. Raises: none.
    """
    problems: list[str] = []
    if record.get("status") != "ok":
        return problems
    device = str(record.get("device", ""))
    if device not in DEVICE_ENV_MAP:
        problems.append(f"{name}: device={device!r} 不在 cuda/cpu 里（⑥）")
        return problems
    env = str(record.get("env_fingerprint", ""))
    allowed = DEVICE_ENV_MAP[device]
    if not any(env.endswith(f"/{file}") for file in allowed):
        problems.append(f"{name}: device={device} 但 env_fingerprint={env}（应为 {allowed}）")
    return problems


# ---- 3. 组合审计（第 0 条 + ①–⑥）/ The composed audit (rule 0 + 1-6) ----


def audit_run_records(
    raw_dir: Path = RAW_RUNS, pred_dir: Path = PREDICTIONS
) -> list[str]:
    """对一整套记录跑第 0 条 + 六条断言，返回问题列表（空 = 全部合格）。
    Run rule 0 plus the six assertions over a full record set; empty list means all pass.

    中文：
        用途：注入测试与真实数据的共用入口。顺序：读记录 → 第 0 条（schema）→ ① 字段 →
              ② 预测文件 → ③ 行数与折 → ④ 唯一性 → ⑤ partial → ⑥ device/env。
        输入：raw_dir；pred_dir。输出：问题字符串列表。数据来源：两目录 + `splits/**`。
        配置来源：`_schema.json` 与任务书 §2 C-c-4。副作用：只读。异常：无（全部收集）。

    English:
        Purpose: the shared entry point for the injection tests and the real data. Order: load,
        rule 0 (schema), 1 fields, 2 predictions file, 3 rows vs fold, 4 uniqueness, 5 partials,
        6 device/env. Args: raw_dir; pred_dir. Returns: problem strings. Data source: both
        directories plus splits/**. Config: _schema.json and the brief. Side effects: read-only.
        Raises: none (everything is collected).
    """
    problems: list[str] = []
    records, load_problems = _load_records(raw_dir)
    problems.extend(load_problems)
    schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    cache = _split_spec_cache()
    for name, record in records.items():
        problems.extend(_schema_problems(record, schema, name))
        problems.extend(_audit_required_fields(record, name))
        problems.extend(_audit_predictions(record, name, pred_dir))
        problems.extend(_audit_prediction_rows(record, name, pred_dir, cache))
        problems.extend(_audit_device_env(record, name))
    problems.extend(_audit_unique_ids(records))
    problems.extend(_audit_partials(raw_dir))
    return problems


# ---- 4. 真实数据上的八条测试 / Eight tests over the real records ----

_REAL_AUDIT_CACHE: dict[str, list[str]] = {}


@pytest.fixture(scope="module")
def real_audit() -> list[str]:
    """对仓库真实记录跑一次完整审计（模块级，只跑一遍）。
    Run the full audit on the real records once per module.
    """
    key = "result"
    if key not in _REAL_AUDIT_CACHE:
        _REAL_AUDIT_CACHE[key] = audit_run_records()
    return _REAL_AUDIT_CACHE[key]


@pytest.fixture(scope="module")
def real_records() -> dict[str, Any]:
    """仓库真实记录（文件名 → 记录）。
    The real records (file name -> record).
    """
    records, _ = _load_records(RAW_RUNS)
    return records


def test_schema_file_parses() -> None:
    """`_schema.json` 本身必须能解析且有 required/properties（契约的根）。
    The schema file itself must parse and carry required/properties.
    """
    schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    assert isinstance(schema.get("required"), list) and schema["required"]
    assert isinstance(schema.get("properties"), dict) and schema["properties"]


def test_schema_uses_only_supported_keywords() -> None:
    """`_schema.json` 只能使用子集校验器支持的关键字，否则校验会静默漏检。
    The schema may only use the supported keywords; anything else would be silently unchecked.
    """
    schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))

    def walk(node: Any, path: str, out: list[str], *, in_properties: bool = False) -> None:
        if isinstance(node, dict):
            for key, value in node.items():
                # `properties` 块里的键是实例字段名（run_id、status…），不是 schema 关键字。
                # Keys inside a `properties` block are instance field names, not schema keywords.
                if not in_properties and key not in SUPPORTED_KEYWORDS:
                    out.append(f"{path}.{key}")
                walk(value, f"{path}.{key}", out, in_properties=key == "properties")
        elif isinstance(node, list):
            for index, item in enumerate(node):
                walk(item, f"{path}[{index}]", out)

    unknown: list[str] = []
    walk(schema, "$", unknown)
    assert not unknown, f"_schema.json 用了不支持的关键字：{unknown}"


def test_0_all_records_pass_the_schema(real_records: dict[str, Any]) -> None:
    """第 0 条：每条记录过 `_schema.json`（字段、类型、pattern、if/then 一次验全）。
    Rule 0: every record passes _schema.json (fields, types, patterns and if/then in one pass).
    """
    schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    problems = [
        p
        for name, record in real_records.items()
        for p in _schema_problems(record, schema, name)
    ]
    assert not problems, "\n".join(problems[:20])


def test_1_required_fields_present(real_records: dict[str, Any]) -> None:
    """断言①：七项基础字段 + ok 六项附加字段齐全。
    Assertion 1: the seven base fields plus six ok-extra fields are present.
    """
    problems = [
        p for name, r in real_records.items() for p in _audit_required_fields(r, name)
    ]
    assert not problems, "\n".join(problems[:20])


def test_2_every_ok_run_has_predictions(real_records: dict[str, Any]) -> None:
    """断言②：每个 status=ok 的 run 有 `predictions/<run_id>.parquet`。
    Assertion 2: every ok run has predictions/<run_id>.parquet.
    """
    problems = [
        p for name, r in real_records.items() for p in _audit_predictions(r, name, PREDICTIONS)
    ]
    assert not problems, "\n".join(problems[:20])


def test_3_prediction_rows_match_split_test_length(real_records: dict[str, Any]) -> None:
    """断言③：预测行数 = 该折 test 长度，列 = [row_id, y_pred]。
    Assertion 3: prediction rows equal the fold's test length; columns are row_id, y_pred.
    """
    cache = _split_spec_cache()
    problems = [
        p
        for name, r in real_records.items()
        for p in _audit_prediction_rows(r, name, PREDICTIONS, cache)
    ]
    assert not problems, "\n".join(problems[:20])


def test_4_run_ids_unique_and_filenames_match(real_records: dict[str, Any]) -> None:
    """断言④：run_id 全局唯一，文件名 = `<run_id>.json`。
    Assertion 4: run_ids are globally unique and file names equal <run_id>.json.
    """
    problems = _audit_unique_ids(real_records)
    assert not problems, "\n".join(problems[:20])


def test_5_no_partial_artifacts(real_audit: list[str]) -> None:
    """断言⑤：results/raw_runs/ 里没有 `*.partial.*`。
    Assertion 5: no *.partial.* sits in results/raw_runs/.
    """
    partials = [p for p in real_audit if "partial" in p]
    assert not partials, "\n".join(partials[:20])


def test_6_device_matches_env_fingerprint(real_records: dict[str, Any]) -> None:
    """断言⑥：cuda↔env_gnn/env_polybert，cpu↔env_core。
    Assertion 6: cuda pairs with env_gnn/env_polybert, cpu with env_core.
    """
    problems = [
        p for name, r in real_records.items() for p in _audit_device_env(r, name)
    ]
    assert not problems, "\n".join(problems[:20])


def test_composed_audit_is_clean_on_real_records(real_audit: list[str]) -> None:
    """组合审计在真实记录上必须零问题（六条 + 第 0 条一次跑完）。
    The composed audit must be clean on the real records.
    """
    assert not real_audit, "\n".join(real_audit[:30])


# ---- 5. 四处注入必须变红 / Four injected defects must turn red ----

# 注入样本都从这条真实记录派生：改动一处，其余全部保持真实，保证"红"来自那处注入。
# Every injection derives from this real record; one change per sample keeps the red honest.
_SAMPLE_NAME = "op_dielectric_constant_total__a3_graph__seed0_n120__seed0"
_SAMPLE_PRED = "op_dielectric_constant_total__a3_graph__seed0_n120__seed0.parquet"


def _sample_record() -> dict[str, Any]:
    """读真实样本记录（注入测试的地基）。
    Load the real sample record used by the injection tests.
    """
    return json.loads((RAW_RUNS / f"{_SAMPLE_NAME}.json").read_text(encoding="utf-8"))


def _write_world(tmp_path: Path, record: dict[str, Any], *, filename: str | None = None,
                 extra: list[tuple[str, dict[str, Any]]] | None = None) -> tuple[Path, Path]:
    """在 tmp 里搭一个最小的审计世界：一条记录 + 一份真实预测，返回 (raw_dir, pred_dir)。
    Build a minimal audit world in tmp: one record plus its real predictions; return both dirs.
    """
    raw_dir = tmp_path / "raw_runs"
    pred_dir = tmp_path / "predictions"
    raw_dir.mkdir()
    pred_dir.mkdir()
    name = filename if filename is not None else f"{record['run_id']}.json"
    (raw_dir / name).write_text(json.dumps(record, ensure_ascii=False), encoding="utf-8")
    (pred_dir / _SAMPLE_PRED).write_bytes(
        (PREDICTIONS / _SAMPLE_PRED).read_bytes()
    )
    for extra_name, extra_record in extra or []:
        (raw_dir / extra_name).write_text(
            json.dumps(extra_record, ensure_ascii=False), encoding="utf-8"
        )
    return raw_dir, pred_dir


def test_injection_a_removed_split_version_turns_red(tmp_path: Path) -> None:
    """注入 (a)：删掉一条记录的 `split_version`，套件必须红。
    Injection (a): drop split_version from one record; the suite must turn red.
    """
    record = _sample_record()
    del record["split_version"]
    raw_dir, pred_dir = _write_world(tmp_path, record)
    problems = audit_run_records(raw_dir, pred_dir)
    assert any("split_version" in p for p in problems), problems


def test_injection_b_renamed_partial_turns_red(tmp_path: Path) -> None:
    """注入 (b)：把一份 `--limit` 试跑产物放进 raw_runs（`*.partial.*`），套件必须红。
    Injection (b): a --limit trial artefact named *.partial.* lands in raw_runs; suite must go red.
    """
    record = _sample_record()
    partial_record = dict(record)
    partial_record["run_id"] = "op_dielectric_constant_total__a3_graph__seed9_n120__seed9"
    partial_record["status"] = "blocked_no_gpu"
    partial_record["requested_at"] = "2026-11-13T12:00:00+08:00"
    partial_record["command"] = "python -m chainworth.train --limit 200"
    raw_dir, pred_dir = _write_world(
        tmp_path,
        record,
        extra=[(f"{partial_record['run_id']}.partial.limit200.json", partial_record)],
    )
    problems = audit_run_records(raw_dir, pred_dir)
    assert any(PARTIAL_MARKER in p for p in problems), problems


def test_injection_c_device_env_mismatch_turns_red(tmp_path: Path) -> None:
    """注入 (c)：device=cuda 但 env_fingerprint 指向 env_core.json，套件必须红。
    Injection (c): device=cuda with env_fingerprint pointing at env_core.json; suite must go red.
    """
    record = _sample_record()
    record["env_fingerprint"] = "results/env_fingerprints/env_core.json"
    raw_dir, pred_dir = _write_world(tmp_path, record)
    problems = audit_run_records(raw_dir, pred_dir)
    assert any("env_fingerprint" in p and "device" in p for p in problems), problems


def test_injection_d_duplicate_run_id_turns_red(tmp_path: Path) -> None:
    """注入 (d)：复制一份记录让同一 run_id 出现两次，套件必须红。
    Injection (d): duplicate a record so one run_id appears twice; the suite must go red.
    """
    record = _sample_record()
    raw_dir, pred_dir = _write_world(
        tmp_path,
        record,
        extra=[(f"{record['run_id']}_dup.json", dict(record))],
    )
    problems = audit_run_records(raw_dir, pred_dir)
    assert any("重复" in p for p in problems), problems
