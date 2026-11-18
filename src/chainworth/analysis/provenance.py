"""四向对账：每条数字都要能追到一条 run 记录、一份预测、一个真实划分文件。
Four-way provenance: every number must trace to a run record, a prediction file, and a real
split file.

中文：
    用途：分工 D-8 的机器版——把"我查过了"换成"脚本能点出 run_id"。四向：
          ① **划分**：记录的 `split_version`（`<sha256前12位>@<相对路径>`）里的路径必须存在，
             且文件字节的 sha256 前 12 位必须与记录里写的一致。手写版本号换文件时不会变，
             字节哈希会变——这是 A-5 立的规矩。
          ② **指纹**：`env_fingerprint` 指向的 `results/env_fingerprints/*.json` 必须存在。
          ③ **预测**：`predictions` 指向的 parquet 必须存在，且**行数 = 记录的 `n_test`**、
             列恰好 `row_id,y_pred`、`row_id` 不重复。行数对不上意味着"算 R² 的那批点"
             与"落盘的那批点"不是同一批，而这件事不抛异常。
          ④ **汇总表**：`results/aggregate/*.csv` 里出现的每个 `run_id` 都必须在
             `results/raw_runs/` 里存在。反过来，汇总目录里出现一张**没登记**的表也判红——
             新表静默出现，等于新数字静默获得"已对账"的身份。

    **三态而不是两态**（这条是 C-c 验收的教训，D-62）：`ok` / `fail` / `exempt`。
    `exempt` 必须带**登记过的理由**，两种：
      - `MATERIALISATION_GAP`：划分文件按 D-57 不入库（`splits/polybench26/arch/`），
        本机没物化时**是缺件，不是记录错**。此时 `exempt` 并打印重建命令，
        而不许判 `fail`——判错会让人以为 15 条记录坏了。
      - `DERIVED_TABLE`：诊断/派生表（RF 的 40 行、零信息对照、ln 敏感性……）按设计
        不进 `raw_runs/`，它们的数字由同目录 CSV 自证。
    判据：`exempt` 不计入通过率，但**必须出现在表里**；`fail` 一条就说明这批数字不能出图。

    数据来源：`results/raw_runs/*.json`（B/C 写盘）、`results/aggregate/predictions/*.parquet`、
              `splits/`、`results/env_fingerprints/`、`results/aggregate/*.csv`。
    配置来源：本模块的 `RUN_ID_TABLES` / `DERIVED_TABLES` / `MATERIALISED_NOT_COMMITTED`
              三张登记表——**新增汇总表必须在这里登记**，否则 `aggregate_table_registered` 判红。
    副作用：`write_table()` 写 `results/aggregate/provenance_check.csv`（显式 LF）。
    异常：`ProvenanceError` —— `require_clean()` 在仍有 `fail` 时抛出。
    对外：`collect()` / `summarise()` / `write_table()` / `require_clean()` / `render()`。
    被谁用：`scripts/06_audit.py provenance`、`tests/test_provenance.py`、
            以及**每个出图脚本的开头**（分工 D-8 的验收：通过率 100% 才允许出图）。

English:
    Purpose: the machine version of D-8 - replace "I checked it" with "the script can point at the
    run_id". Four directions: (1) the path inside `split_version` exists and the file's sha256
    prefix matches; (2) `env_fingerprint` points at an existing file; (3) every prediction parquet
    exists with rows == `n_test`, exactly two columns, unique `row_id`; (4) every `run_id` named in
    `results/aggregate/*.csv` exists under `results/raw_runs/`, and any *unregistered* table in the
    aggregate directory is a failure too.
    Three statuses, not two (the lesson from the C-c acceptance, D-62): `ok` / `fail` / `exempt`,
    where `exempt` must carry a **registered** reason - a materialisation gap (splits not
    committed per D-57) or a derived table (diagnostics that by design never enter `raw_runs/`).
    Exempt rows stay in the table; one `fail` means the numbers may not be plotted.
    Side effects: writes `results/aggregate/provenance_check.csv` with explicit LF line endings.
    Raises: `ProvenanceError` from `require_clean()` while any `fail` remains.
"""

from __future__ import annotations

import csv
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

REPO_ROOT = Path(__file__).resolve().parents[3]
RAW_RUNS_DIR = REPO_ROOT / "results" / "raw_runs"
PREDICTIONS_DIR = REPO_ROOT / "results" / "aggregate" / "predictions"
AGGREGATE_DIR = REPO_ROOT / "results" / "aggregate"
FINGERPRINT_DIR = REPO_ROOT / "results" / "env_fingerprints"
TABLE_PATH = AGGREGATE_DIR / "provenance_check.csv"

SCHEMA_NAME = "_schema.json"

# ---- 1. 登记表：新表不登记就红 / Registries: an unregistered table fails ----

# 有 `run_id` 列的汇总表 → (列名, 分隔符)。分隔符为 None 表示一个单元格只有一个 run_id；
# `";"` 表示一格多个（`main_table.csv` 的 `run_ids`）。
# Aggregate tables carrying run ids: (column, separator). None = one id per cell, ";" = many.
RUN_ID_TABLES: Mapping[str, tuple[str, str | None]] = {
    "a1_a2_index.csv": ("run_id", None),
    "arch_transfer.csv": ("run_id", None),
    "audit_backlog.csv": ("run_id", None),
    "gpu_hours.csv": ("run_id", None),
    "main_table.csv": ("run_ids", ";"),
    "openpoly_cpu.csv": ("run_id", None),
    "openpoly_gpu_index.csv": ("run_id", None),
    "scaling_chain.csv": ("run_id", None),
    "scaling_cpu.csv": ("run_id", None),
}

# 一行要引用**多个** run 的表 → 列名元组。`lambda_ci.csv` 一行是一对（unit 臂 + chain 臂），
# 两条 run 都必须反查得到——把它当"派生表"豁免掉，就等于放过这张表最重要的身份约束。
# Tables whose rows reference **several** runs: every named column is resolved. Treating
# `lambda_ci.csv` as a derived table would waive the one identity constraint that matters there.
MULTI_RUN_ID_TABLES: Mapping[str, tuple[str, ...]] = {
    "lambda_ci.csv": ("unit_run", "chain_run"),
    # D-69：κ 的每个单元格都是一对（单元臂 + 链臂）。两列都反查，理由同上——豁免这张表
    # 就等于放过"这一格的两个数到底来自哪两条 run"。
    # Every kappa cell is a pair (unit arm + chain arm); both columns are resolved.
    "kappa_cells.csv": ("unit_run_id", "chain_run_id"),
}

# 没有 `run_id` 可反查的派生/诊断表 → 豁免理由。理由必须写成"为什么它不需要 run_id"，
# 不是"先跳过"。空理由不算登记。
# Derived/diagnostic tables with no run id to resolve -> the reason they need none. An empty
# reason is not a registration.
DERIVED_TABLES: Mapping[str, str] = {
    "a2_best_params.csv": "超参表：由 search_task() 产出，一行一个 (task, learner)，不是某一条 run 的成绩",
    "a4_finetune_vs_frozen.csv": "D-34 ③：微调诊断，按设计不进 raw_runs；冻结那列可由 a4 记录核对",
    "arch_floor_compare.csv": "官方地板对照：值来自官方发布文件，不是我们的 run",
    "cost_summary.csv": "按 scope 汇总的工时视图，逐 run 明细在 gpu_hours.csv（那张已登记）",
    "learner_sensitivity.csv": "D-44：RF/kNN 学习器敏感性，按设计不写 raw_runs",
    "learner_sensitivity_params.csv": "同上：敏感性表选出的超参",
    "kappa.csv": "D-69：按数据集汇总的 κ 表，一行一个数据集；逐格的两条 run_id 在 kappa_cells.csv（那张已按多列登记）",
    "kappa_leverage.csv": "D-73：D-3 验收标准②的杠杆检查（去掉最小档重拟合），派生自 kappa_cells.csv，不含新的 run",
    "h1_ordering.csv": "D-74：H1 的分层排序（逐点 + 每臂两行汇总），派生自 lambda_ci.csv 与任务配置，不含新的 run",
    "h1_heterogeneity.csv": "D-74：来源异质性、tier-样本量混淆与单任务杠杆，同样派生自 lambda_ci.csv",
    "metric_sensitivity.csv": "D-77：H2′ 的两把尺子（Λ 与 ΔR²）首末比 + 折级区间，派生自 kappa_cells.csv，不含新的 run",
    "metric_sensitivity_levels.csv": "D-77：同上，逐档位均值；两把尺子用同一批折与同一批档位",
    "n_matched.csv": "D-78：n 匹配重训（t1 下采样到 t2 量级），按设计不写 raw_runs（run_id 不含 n 维度，D-34）",
    "n_matched_lambda.csv": "D-78：n 匹配的逐格 Λ，派生自 n_matched.csv，不含新的 run",
    "n_matched_summary.csv": "D-78：同 n 下 tier1 vs tier2 的 Δ 与区间，派生自 n_matched_lambda.csv",
    "log_target_sensitivity.csv": "D-66：ln 尺度敏感性，按设计不写 raw_runs、不进主表",
    "n_bins_covered.csv": "由 openpoly_n_index.csv 按 n 档聚合的覆盖度视图，无单条 run",
    "null_calibration.csv": "D-46：零信息对照，按设计不写 raw_runs",
    "official_baseline_compare.csv": "官方基线对照：一侧是官方数字，另一侧按 task 汇总",
    "openpoly_cpu_n_index.csv": "n 档索引：一行一个 (task, fold, level)，由 pack 与划分派生",
    "openpoly_n_index.csv": "同上（GPU 侧的 n 档索引）",
    "pb26_controlled_control.csv": "D-45：受控对照，按设计不写 raw_runs",
    "polybench_official_compare.csv": "同 official_baseline_compare：官方数字 vs 按任务汇总",
    "spearman.csv": "D-48 ⑦：由 predictions 逐份重算、按 (task, arm) 汇总，run_id 集合即主表 run_ids",
    "defense_cells.csv": "D-80：答辩补算的逐格视图（可报行 / 稳定行 / 最小分母），派生自 lambda_ci.csv，不含新的 run",
    "defense_checks.json": "D-80：置换 p、按任务重采样、十倍段斜率、两臂翻号、n* 插值、地板口径；只读 aggregate 表，不含新的 run",
}

# 按 D-57 物化但不入库的划分前缀。缺件 → exempt（并给重建命令）；**存在但对不上 → fail**。
# Split prefixes materialised but not committed (D-57): missing -> exempt with a rebuild hint;
# present but mismatched -> fail.
MATERIALISED_NOT_COMMITTED: tuple[str, ...] = ("splits/polybench26/arch/",)
REBUILD_HINT = "python scripts/04_run_cpu.py arch --setup"

STATUS_OK = "ok"
STATUS_FAIL = "fail"
STATUS_EXEMPT = "exempt"

REASON_MATERIALISATION = "MATERIALISATION_GAP"
REASON_DERIVED = "DERIVED_TABLE"
REASON_RF_LEARNER = "RF_NOT_A_CONTRACT_ARM"

EXEMPT_REASONS: Mapping[str, str] = {
    REASON_MATERIALISATION: (
        "D-57：该划分按设计不入库，本机未物化；缺件不等于记录错。重建：" + REBUILD_HINT
    ),
    REASON_DERIVED: "诊断/派生表，按设计不写 raw_runs；理由见 DERIVED_TABLES 对应行",
    REASON_RF_LEARNER: "D-53：RF 只进 scaling_cpu.csv、不写 raw_runs（_schema 的 arm 枚举里没有它）",
}


class ProvenanceError(RuntimeError):
    """对账仍有 `fail` 时抛出——数字不许出图。
    Raised while any `fail` remains; the numbers may not be plotted.
    """


@dataclass(frozen=True)
class Row:
    """对账表的一行 / one row of the provenance table."""

    check: str
    run_id: str
    status: str
    detail: str


# ---- 2. 读盘 / Reading the tree ----


def _sha256_prefix(path: Path, length: int = 12) -> str:
    """文件字节 sha256 的前 `length` 位（与 `split_version` 的写法一致）。
    The first `length` hex chars of the file's sha256, matching how `split_version` is written.

    中文：输入 path / length；输出小写十六进制。数据来源：磁盘。副作用：只读。
    English: Args: path, length. Returns lowercase hex. Data: disk. Side effects: read-only.
    """
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()[:length]


def load_records(directory: Path | None = None) -> list[dict[str, Any]]:
    """读全部 run 记录（跳过 `_schema.json`），按 run_id 排序。
    Read every run record (skipping `_schema.json`), sorted by run_id.

    中文：输入 directory（缺省 `results/raw_runs/`，测试可替换）；输出 dict 列表。
          副作用：只读。异常：FileNotFoundError —— 目录不存在；json.JSONDecodeError —— 记录损坏。
    English: Args: directory (defaults to results/raw_runs). Returns the records sorted by run_id;
    raises when the directory is missing or a record is not valid JSON.
    """
    base = directory or RAW_RUNS_DIR
    if not base.is_dir():
        raise FileNotFoundError(f"没有 run 记录目录：{base}")
    records: list[dict[str, Any]] = []
    for path in sorted(base.glob("*.json")):
        if path.name == SCHEMA_NAME:
            continue
        records.append(json.loads(path.read_text(encoding="utf-8")))
    return sorted(records, key=lambda record: record["run_id"])


def _is_materialisation_path(relative: str) -> bool:
    """这条路径是否落在"物化但未入库"的前缀下。/ Is the path under a materialised-but-uncommitted
    prefix?
    """
    return any(relative.startswith(prefix) for prefix in MATERIALISED_NOT_COMMITTED)


def _check_split(record: Mapping[str, Any], root: Path = REPO_ROOT) -> Row:
    """① 划分：路径存在 + 字节哈希前缀一致。/ Split: the path exists and the digest matches.

    中文：输入 record / root（仓库根，测试可替换）；输出 Row。副作用：只读。
    English: Args: record, root (replaceable in tests). Returns one row.
    """
    run_id = record["run_id"]
    version = str(record.get("split_version", ""))
    if "@" not in version:
        return Row(
            "split_version_shape",
            run_id,
            STATUS_FAIL,
            f"split_version 不是 `<sha256前12位>@<路径>`：{version!r}",
        )
    digest, relative = version.split("@", 1)
    path = root / relative
    if not path.is_file():
        if _is_materialisation_path(relative):
            return Row(
                "split_file_present",
                run_id,
                STATUS_EXEMPT,
                f"{REASON_MATERIALISATION}；缺 {relative}；重建：{REBUILD_HINT}",
            )
        return Row("split_file_present", run_id, STATUS_FAIL, f"划分文件不存在：{relative}")
    actual = _sha256_prefix(path, len(digest))
    if actual != digest:
        return Row(
            "split_digest_matches",
            run_id,
            STATUS_FAIL,
            f"{relative} 的字节哈希是 {actual}，记录写的是 {digest}（文件被换过或记录写错）",
        )
    return Row("split_digest_matches", run_id, STATUS_OK, relative)


def _check_fingerprint(record: Mapping[str, Any], root: Path = REPO_ROOT) -> Row:
    """② 指纹：`env_fingerprint` 指向的文件必须存在。/ Fingerprint file must exist."""
    run_id = record["run_id"]
    relative = str(record.get("env_fingerprint", ""))
    if not relative:
        return Row("env_fingerprint_present", run_id, STATUS_FAIL, "记录里没有 env_fingerprint")
    path = root / relative
    if not path.is_file():
        return Row("env_fingerprint_present", run_id, STATUS_FAIL, f"指纹文件不存在：{relative}")
    return Row("env_fingerprint_present", run_id, STATUS_OK, relative)


def _check_prediction(record: Mapping[str, Any], root: Path = REPO_ROOT) -> list[Row]:
    """③ 预测：文件存在、列恰好两项、行数 = `n_test`、`row_id` 不重复。
    Prediction: exists, exactly two columns, rows == n_test, unique row_id.

    中文：这里用 pyarrow 只读 schema 与列长，不把上千份全量装进内存再比——
          分批读、逐份核，内存占用与文件数无关。
    English: reads one file at a time (schema + length), so memory does not scale with the count.
    """
    import pyarrow.parquet as pq

    run_id = record["run_id"]
    declared = str(record.get("predictions", ""))
    path = (root / declared) if declared else (root / "results" / "aggregate" / "predictions" / f"{run_id}.parquet")
    if not path.is_file():
        return [
            Row(
                "prediction_present",
                run_id,
                STATUS_FAIL,
                f"预测文件不存在：{declared or path.relative_to(root).as_posix()}",
            )
        ]

    table = pq.read_table(path)
    names = list(table.column_names)
    n_rows = table.num_rows
    expected = int(record.get("n_test", -1))

    rows = [Row("prediction_present", run_id, STATUS_OK, path.relative_to(root).as_posix())]
    if names != ["row_id", "y_pred"]:
        rows.append(
            Row("prediction_columns_contract", run_id, STATUS_FAIL, f"列是 {names}，应为 ['row_id','y_pred']")
        )
    else:
        row_ids = table.column("row_id").to_pylist()
        if len(set(row_ids)) != len(row_ids):
            rows.append(
                Row("prediction_row_ids_unique", run_id, STATUS_FAIL, "row_id 有重复，重采样会重复计点")
            )
        else:
            rows.append(Row("prediction_row_ids_unique", run_id, STATUS_OK, f"{n_rows} 个 row_id 互不重复"))

    if n_rows != expected:
        rows.append(
            Row(
                "prediction_rows_equal_n_test",
                run_id,
                STATUS_FAIL,
                f"预测 {n_rows} 行，记录的 n_test 是 {expected}（算 R² 的点与落盘的点不是同一批）",
            )
        )
    else:
        rows.append(Row("prediction_rows_equal_n_test", run_id, STATUS_OK, f"{n_rows} 行"))
    return rows


def check_runs(
    records: Sequence[Mapping[str, Any]] | None = None,
    root: Path = REPO_ROOT,
) -> list[Row]:
    """逐条 run 跑 ①②③。/ Run ①②③ for every record.

    中文：输入 records（缺省读盘）/ root（仓库根，测试可替换）。输出 Row 列表。副作用：只读。
    English: Args: records (read from disk by default), root (replaceable in tests). Returns rows.
    """
    if records is None:
        records = load_records()
    rows: list[Row] = []
    for record in records:
        rows.append(_check_split(record, root))
        rows.append(_check_fingerprint(record, root))
        rows.extend(_check_prediction(record, root))
    return rows


# ---- 3. 汇总表反查 / Aggregate tables ----

def _read_csv(path: Path) -> list[dict[str, str]]:
    """读一张 CSV（显式 utf-8，避免 Windows 默认代码页）。/ Read a CSV as utf-8 rows."""
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _check_multi_run_ids(
    name: str,
    data: Sequence[Mapping[str, str]],
    known: set[str],
) -> list[Row]:
    """一行引用多条 run 的表：每一列都要能反查。/ Every named column must resolve.

    中文：输入表名 / 行 / 已知 run_id 集合；输出 Row 列表。
          数据来源：`results/aggregate/<name>`。副作用：只读。
          逐行逐列判，不漏（`lambda_ci.csv` 有 321 行 × 2 列 = 642 个引用）。
    English: Skips nothing - `lambda_ci.csv` carries 321 rows x 2 columns of references.
    """
    columns = MULTI_RUN_ID_TABLES.get(name, ())
    rows: list[Row] = []
    seen: set[str] = set()
    failed = False
    for line_no, line in enumerate(data, start=2):
        for column in columns:
            run_id = (line.get(column) or "").strip()
            if not run_id:
                failed = True
                rows.append(
                    Row(
                        "aggregate_run_id_resolves",
                        f"{name}:{line_no}",
                        STATUS_FAIL,
                        f"{column} 为空",
                    )
                )
                continue
            seen.add(run_id)
            if run_id not in known:
                failed = True
                rows.append(
                    Row(
                        "aggregate_run_id_resolves",
                        run_id,
                        STATUS_FAIL,
                        f"{name}:{line_no} 的 {column} 引用了不存在的 run 记录",
                    )
                )
    if not failed:
        rows.append(
            Row(
                "aggregate_run_id_resolves",
                "",
                STATUS_OK,
                f"{name}：{len(seen)} 个 run_id 全部可反查（{len(data)} 行 × {len(columns)} 列）",
            )
        )
    return rows


def check_aggregates(
    known_run_ids: Iterable[str],
    aggregate_dir: Path | None = None,
) -> list[Row]:
    """④ 汇总表：每个 run_id 可反查；每张表都已登记。
    Aggregate tables: every run id resolves, and every table is registered.

    中文：
        用途：这是"结论必须对应一条运行记录"那条硬规则的机器版。两个方向都查：
              - **表里写的** run_id 必须存在（`aggregate_run_id_resolves`）；
              - **目录里有的**表必须登记过（`aggregate_table_registered`），否则新表静默获得
                "已对账"身份。
        输入：known_run_ids；aggregate_dir（缺省 `results/aggregate`）。
        输出：Row 列表。数据来源：`results/aggregate/*.csv`。副作用：只读。
        特例：`scaling_cpu.csv` 里 `learner=rf` 的行按 D-53 不写 run_id → `exempt`；
              同一张表里 `learner=xgboost` 的 `run_id` 为空则是 `fail`（不能把"忘了写"混进豁免）。
    English:
        Both directions: run ids named in a table must exist, and every table in the directory must
        be registered. `scaling_cpu.csv` rows with `learner=rf` are exempt per D-53, but an empty
        run_id on an xgboost row is a failure - "forgot to write it" must not hide inside the
        exemption.
    """
    directory = aggregate_dir or AGGREGATE_DIR
    known = set(known_run_ids)
    rows: list[Row] = []

    if not directory.is_dir():
        return [Row("aggregate_table_registered", "", STATUS_FAIL, f"没有汇总目录：{directory}")]

    for path in sorted(directory.glob("*.csv")):
        name = path.name
        if name == TABLE_PATH.name:
            # 本表自身不许参与反查：它记录的是检查结果，不是数据。
            continue
        if name in DERIVED_TABLES:
            rows.append(
                Row(
                    "aggregate_table_registered",
                    "",
                    STATUS_EXEMPT,
                    f"{REASON_DERIVED}：{name} —— {DERIVED_TABLES[name]}",
                )
            )
            continue
        if name in MULTI_RUN_ID_TABLES:
            rows.append(
                Row(
                    "aggregate_table_registered",
                    "",
                    STATUS_OK,
                    f"{name}：已登记，逐行核 {' + '.join(MULTI_RUN_ID_TABLES[name])}",
                )
            )
            rows.extend(_check_multi_run_ids(name, _read_csv(path), known))
            continue
        if name not in RUN_ID_TABLES:
            rows.append(
                Row(
                    "aggregate_table_registered",
                    "",
                    STATUS_FAIL,
                    f"{name} 未登记：新表要么有 run_id 列（进 RUN_ID_TABLES），"
                    "要么是派生表（进 DERIVED_TABLES 并写理由）",
                )
            )
            continue

        column, separator = RUN_ID_TABLES[name]
        rows.append(
            Row("aggregate_table_registered", "", STATUS_OK, f"{name}：已登记，列 {column}")
        )
        data = _read_csv(path)
        if not data:
            # 空表是**查过了、没有行要反查**，不是"跳过了"。`audit_backlog.csv` 现在就是这样：
            # 表头在、零条待办，含义是"没有未闭合的退回"。判成 exempt 会让人以为它没被检查。
            # An empty table was checked and has nothing to resolve - not skipped.
            rows.append(
                Row("aggregate_run_id_resolves", "", STATUS_OK, f"{name}：空表（只有表头），无行可反查")
            )
            continue
        if column not in data[0]:
            rows.append(
                Row("aggregate_run_id_resolves", "", STATUS_FAIL, f"{name} 缺列 {column}，无法反查")
            )
            continue

        seen: set[str] = set()
        table_failed = False
        for line_no, line in enumerate(data, start=2):
            cell = (line.get(column) or "").strip()
            if not cell:
                learner = (line.get("learner") or "").strip().lower()
                if name == "scaling_cpu.csv" and learner == "rf":
                    rows.append(
                        Row(
                            "aggregate_run_id_resolves",
                            f"{name}:{line_no}",
                            STATUS_EXEMPT,
                            f"{REASON_RF_LEARNER}（task={line.get('task')} tier={line.get('tier')}）",
                        )
                    )
                else:
                    table_failed = True
                    rows.append(
                        Row(
                            "aggregate_run_id_resolves",
                            f"{name}:{line_no}",
                            STATUS_FAIL,
                            f"{column} 为空，且不是登记过的豁免（learner={learner or '空'}）",
                        )
                    )
                continue
            ids = [part.strip() for part in cell.split(separator)] if separator else [cell]
            for run_id in ids:
                if not run_id or run_id in seen:
                    continue
                seen.add(run_id)
                if run_id not in known:
                    table_failed = True
                    rows.append(
                        Row(
                            "aggregate_run_id_resolves",
                            run_id,
                            STATUS_FAIL,
                            f"{name}:{line_no} 引用了不存在的 run 记录",
                        )
                    )
        if not table_failed:
            rows.append(
                Row(
                    "aggregate_run_id_resolves",
                    "",
                    STATUS_OK,
                    f"{name}：{len(seen)} 个 run_id 全部可反查",
                )
            )
    return rows


# ---- 4. 汇总、落盘、闸门 / Summary, writing, gate ----


def collect(
    aggregate_dir: Path | None = None,
    root: Path = REPO_ROOT,
) -> list[Row]:
    """跑全部四向检查，返回按 (check, run_id) 排序的行。/ Run every check, return sorted rows.

    中文：输入 aggregate_dir / root（测试可替换）；输出 Row 列表。副作用：只读。
    English: Args: aggregate_dir, root (both replaceable in tests). Returns the sorted rows.
    """
    records = load_records(root / "results" / "raw_runs")
    rows = check_runs(records, root)
    rows.extend(check_aggregates([record["run_id"] for record in records], aggregate_dir))
    return sorted(rows, key=lambda row: (row.check, row.run_id or ""))


def summarise(rows: Sequence[Row]) -> dict[str, Any]:
    """把行汇总成"通过率 + 按 check 分诊"的字典。/ Turn the rows into a pass rate and per-check
    triage.

    中文：输入 rows；输出 dict：`n_ok` / `n_fail` / `n_exempt` / `pass_rate` / `by_check` /
          `failures`。通过率 = ok / (ok + fail)，**exempt 不进分母**——缺件不是通过，也不是失败。
    English: Returns counts, `pass_rate = ok / (ok + fail)` (exempt stays out of the denominator),
    a per-check breakdown and the failing rows.
    """
    by_check: dict[str, dict[str, int]] = {}
    failures: list[Row] = []
    for row in rows:
        bucket = by_check.setdefault(row.check, {STATUS_OK: 0, STATUS_FAIL: 0, STATUS_EXEMPT: 0})
        bucket[row.status] += 1
        if row.status == STATUS_FAIL:
            failures.append(row)
    n_ok = sum(row.status == STATUS_OK for row in rows)
    n_fail = sum(row.status == STATUS_FAIL for row in rows)
    n_exempt = sum(row.status == STATUS_EXEMPT for row in rows)
    total = n_ok + n_fail
    return {
        "n_rows": len(rows),
        "n_ok": n_ok,
        "n_fail": n_fail,
        "n_exempt": n_exempt,
        "pass_rate": (n_ok / total) if total else 1.0,
        "by_check": by_check,
        "failures": failures,
    }


def write_table(rows: Sequence[Row], path: Path | None = None) -> Path:
    """把对账表写成 CSV（`check,run_id,status,detail`，显式 LF）。
    Write the table as `check,run_id,status,detail` with explicit LF endings.

    中文：输入 rows / path；输出写入的路径。副作用：写盘。
          行尾必须显式 `newline="\\n"`——Windows 默认 `os.linesep` 会写成 CRLF，`git add` 会报警
          （B-3 的 `a3_repeatability.json` 踩过同一个坑）。
    English: Explicit `newline="\\n"`; the Windows default would write CRLF and trip `git add`.
    """
    target = path or TABLE_PATH
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8", newline="\n") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(["check", "run_id", "status", "detail"])
        for row in rows:
            writer.writerow([row.check, row.run_id, row.status, row.detail])
    return target


def render(summary: Mapping[str, Any], limit: int = 20) -> str:
    """把汇总渲染成给人看的分诊报告。/ Render the summary as a human-readable triage report.

    中文：输入 summary / limit；输出字符串。副作用：无。异常：无。
    English: Args: summary, limit. Returns the report text. No side effects.
    """
    lines = [
        f"对账 {summary['n_rows']} 行：ok {summary['n_ok']} / fail {summary['n_fail']} / "
        f"exempt {summary['n_exempt']}，通过率 {summary['pass_rate'] * 100:.2f}%",
        "",
        f"{'check':32s} {'ok':>6s} {'fail':>6s} {'exempt':>7s}",
    ]
    for check in sorted(summary["by_check"]):
        bucket = summary["by_check"][check]
        lines.append(
            f"{check:32s} {bucket['ok']:>6d} {bucket['fail']:>6d} {bucket['exempt']:>7d}"
        )
    if summary["failures"]:
        lines.append("")
        lines.append(f"失败明细（最多 {limit} 条）：")
        for row in summary["failures"][:limit]:
            lines.append(f"  [{row.check}] {row.run_id}  {row.detail}")
        if len(summary["failures"]) > limit:
            lines.append(f"  …还有 {len(summary['failures']) - limit} 条，见 CSV")
    return "\n".join(lines)


def require_clean(path: Path | None = None, root: Path = REPO_ROOT) -> dict[str, Any]:
    """出图前的闸门：重跑对账，仍有 `fail` 就抛 `ProvenanceError`。
    The pre-plot gate: re-run the audit and raise while any `fail` remains.

    中文：
        用途：分工 D-8 的验收③——"通过率 100% 才允许出图，出图脚本开头自动跑一次"。
              这条不能用"我记得跑过了"实现：对账表可能是三天前的，而 `raw_runs/` 每天都在变。
        输入：path —— 对账表落盘位置（缺省 `results/aggregate/provenance_check.csv`）。
        输出：summary 字典（调用方可以打日志）。
        数据来源：`results/raw_runs/` + `results/aggregate/` 全量重跑。
        副作用：写盘（对账表）。异常：`ProvenanceError` —— 存在 `fail`。
    English:
        Implements D-8 acceptance ③: the figures may only be drawn when the pass rate is 100%, and
        the check must run at the top of the plotting script rather than be remembered. Rewrites
        the table, returns the summary, raises `ProvenanceError` while any `fail` remains.
    """
    rows = collect(aggregate_dir=root / "results" / "aggregate", root=root)
    write_table(rows, path)
    summary = summarise(rows)
    if summary["n_fail"]:
        raise ProvenanceError(
            "对账未通过，数字不许出图：\n" + render(summary, limit=10)
        )
    return summary
