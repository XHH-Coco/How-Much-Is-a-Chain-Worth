"""把 PolyBench 原始 CSV 调成"一个任务一张表"的中间产物，并逐条留下单位判定。
Harmonise the raw PolyBench CSVs into one table per task, with every unit decision recorded.

中文：
    用途：本模块是"原始 CSV -> 可建模的标签"的唯一入口。它做四件事，顺序固定：
          ① 按 (task_id, RDKit 规范 SMILES) 去重；② 把标签换到目标单位；
          ③ 落 parquet（含原始值、原始单位、行号，便于回查）；
          ④ 把每一条判定追加进 `data/manifests/unit_decisions.csv`。
    依赖：data/raw/polybench/<数据集名>.csv（A-2 交付，39 份 md5 已核）；
          data/manifests/polybench_expected_files.csv（A-2 的 39 行标准表）。
    配置：无 yaml——本阶段的"配置"就是本文件里的 `TASK_SPECS` 注册表（A-4 的
          configs/tasks/*.yaml 反过来引用这里产出的 unit_decisions.csv，不要循环依赖）。
    对外：TASK_SPECS、EXCLUDED_TASKS、harmonise_all()、verify_outputs()。
    被谁用：scripts/01_harmonise.py（阶段入口 01）、tests/test_harmonise.py。

English:
    Purpose: the single entry point from raw CSVs to model-ready labels. Four steps in a
    fixed order: (1) dedupe on (task_id, RDKit canonical SMILES); (2) convert the label to
    its target unit; (3) write parquet carrying the raw value, raw unit and source row so any
    row can be traced back; (4) append one decision row per judgement into
    `data/manifests/unit_decisions.csv`.
    Depends on: data/raw/polybench/<dataset>.csv (delivered by A-2, all 39 md5 verified) and
    data/manifests/polybench_expected_files.csv (A-2's 39-row reference table).
    Config: no yaml - the "config" at this stage is the `TASK_SPECS` registry below. A-4's
    configs/tasks/*.yaml reference the produced unit_decisions.csv, never the other way round.
    Public API: TASK_SPECS, EXCLUDED_TASKS, harmonise_all(), verify_outputs().
    Callers: scripts/01_harmonise.py (stage entry 01), tests/test_harmonise.py.
"""

from __future__ import annotations

import csv
import hashlib
import logging
import math
import os
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Callable, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger

from chainworth.protocol.constants import (
    BARRER_IN_SI_UNITS,
    CELSIUS_TO_KELVIN,
    MAIN_TABLE_MIN_LABELS,
)

# RDKit 对每个非法 SMILES 都会往 stderr 打一行解析失败；我们已把它计进 n_unparsable，
# 不需要它再刷屏。 / RDKit prints one line per bad SMILES; we count them ourselves.
RDLogger.DisableLog("rdApp.*")

LOGGER = logging.getLogger("chainworth.harmonise")

# ---- 0. 路径（相对仓库根，禁止绝对路径） / Paths (repo-relative, never absolute) ----

REPO_ROOT = Path(__file__).resolve().parents[3]          # src/chainworth/io/harmonise.py -> 仓库根
RAW_DIR = REPO_ROOT / "data" / "raw" / "polybench"
INTERIM_DIR = REPO_ROOT / "data" / "interim" / "harmonised"
MANIFEST_DIR = REPO_ROOT / "data" / "manifests"
DECISIONS_PATH = MANIFEST_DIR / "unit_decisions.csv"
HARMONISED_MANIFEST_PATH = MANIFEST_DIR / "harmonised_manifest.csv"

# ---- 1. 决策表的列与枚举 / Decision-table columns and vocabulary ----

# 前 8 列是分工书规定的，其后两列是为了让"行数不变式"可核对而加的：
#   n_rows_affected   —— 该条判定**删掉**的行数；纯换算不删行，记 0。
#   n_values_converted —— 该条判定**改写了数值**的行数（℃→K 就是 n_rows_out，identity 是 0）。
# 这样 sum(n_rows_affected) 恰好等于 调和前 - 调和后，而"换算覆盖了多少值"也不会丢。
# The first 8 columns are mandated by the task sheet; the next two exist so the row-count
# invariant is checkable: n_rows_affected counts rows REMOVED (0 for a pure conversion),
# n_values_converted counts rows whose stored value changed.
DECISION_COLUMNS = (
    "task_id",
    "decision_kind",
    "property",
    "raw_unit",
    "target_unit",
    "rule",
    "n_rows_affected",
    "n_values_converted",
    "evidence",
    "decided_by",
    "decided_at",
)

DECISION_KIND_ORDER = (
    "label_unit",
    "dedupe",
    "column_select",
    "unparsable_smiles",
    "invalid_label",
    "exclude_task",
)

DECIDED_BY = "A（仓库所有者代行）"

# ---- 2. 数据类型 / Data types ----


@dataclass(frozen=True)
class TaskSpec:
    """一个任务的调和配方：标签是哪一列、单位怎么变、依据是什么。
    The harmonisation recipe for one task: which column is the label, how the unit changes, and why.

    中文：
        用途：把"这个任务怎么算对"写成数据，而不是散在 if/else 里；新增任务只加一条。
        输入：字段见下方，其中 convert 是纯函数（标量进、标量出），rule 是它的人读描述。
        输出：不适用（这是声明式配置对象）。
        数据来源：label_column 必须是 data/raw/polybench/<dataset_name>.csv 的表头之一
                  （匹配前会 strip，个别表头尾部带空格，见 TSSAB）。
        配置来源：无 yaml，本注册表就是配置。
        副作用：无。
        异常：不在此处抛；列名写错会在 harmonise_task 的列校验里报 KeyError。

    English:
        Purpose: express "what is correct for this task" as data instead of if/else branches;
        adding a task is one more entry.
        Args: see the fields; `convert` is a pure scalar->scalar function and `rule` is its
        human-readable description.
        Returns: n/a (declarative object).
        Data source: label_column must exist in data/raw/polybench/<dataset_name>.csv
        (headers are stripped before matching; some trailing spaces, see TSSAB).
        Config: none - this registry is the config.
        Side effects: none.
        Raises: nothing here; a wrong column name surfaces as KeyError in harmonise_task.
    """

    task_id: str
    dataset_name: str
    role: str
    label_column: str
    label_raw_unit: str
    target_unit: str
    rule: str
    convert: Callable[[float], float]
    rationale: str
    # (原始列名, 调和后列名)；这些是"测试条件"，不是标签，单独保留以免丢失实验语境。
    # (raw column, harmonised column); test conditions, not labels - kept so the context survives.
    condition_columns: tuple[tuple[str, str], ...] = ()


@dataclass
class TaskOutcome:
    """一次调和留下的全部数字，供清单、决策表与验收共用同一份事实。
    Every number a harmonisation run produces, so manifest, decisions and checks share one source.

    中文：
        用途：把"进了多少行、出了多少行、按什么理由少了"打包，避免三处各算一遍算出三个数。
        输入：由 harmonise_task 填。
        输出：不适用。
        数据来源：data/raw/polybench/<dataset_name>.csv 与本次写出的 parquet。
        配置来源：无。
        副作用：无（纯数据容器）。
        异常：无。

    English:
        Purpose: bundle rows-in / rows-out / why-rows-were-dropped so manifest, decisions and
        acceptance checks cannot disagree.
        Args: filled by harmonise_task. Returns: n/a.
        Data source: the raw CSV and the parquet written in the same run.
        Config: none. Side effects: none. Raises: none.
    """

    spec: TaskSpec
    rows_in: int = 0
    rows_out: int = 0
    n_unparsable_smiles: int = 0
    n_invalid_label: int = 0
    n_duplicates: int = 0
    n_values_converted: int = 0
    source_sha256: str = ""
    parquet_path: Path | None = None
    parquet_sha256: str = ""
    label_stats: dict[str, float] = field(default_factory=dict)
    log_residual_max: float | None = None
    has_property_log: bool = False
    condition_columns_used: tuple[tuple[str, str], ...] = ()

    @property
    def rows_removed(self) -> int:
        """调和前后的行数差（= 决策表 n_rows_affected 之和应等于的数）。
        Row-count difference between raw and harmonised; the decision table must sum to this.

        中文：用途：给出不变式的左边；输入：无；输出：int，>=0；数据来源：本对象的三个计数；
              配置来源：无；副作用：无；异常：无。
        English: Purpose: the left-hand side of the invariant. Args: none. Returns: int >= 0.
        Data source: the three counters above. Config: none. Side effects: none. Raises: none.
        """
        return self.rows_in - self.rows_out


@dataclass(frozen=True)
class ExcludedTask:
    """按红线整列弃用的任务：不产出 parquet，但必须在决策表里留下理由。
    A task dropped wholesale under the "never guess" rule: no parquet, but the reason must be recorded.

    中文：
        用途：让"我们这个任务没用"与"我们忘了这个任务"在决策表里长得不一样。
        输入：dataset_name（原始文件名）、task_id（备用）、property（性质名）、reason（依据）。
        输出：不适用。数据来源：该 CSV 的表头（缺单位）与数值量级。
        配置来源：无。副作用：无。异常：无。

    English:
        Purpose: make "we deliberately excluded this task" distinguishable from "we forgot it".
        Args: dataset_name, task_id, property, reason.
        Returns: n/a. Data source: the CSV header (no unit declared) and value magnitudes.
        Config: none. Side effects: none. Raises: none.
    """

    task_id: str
    dataset_name: str
    property: str
    reason: str


# ---- 3. 换算函数 / Conversion functions ----


def _to_kelvin(value: float) -> float:
    """摄氏换开尔文：T[K] = T[℃] + 273.15。
    Celsius to kelvin: T[K] = T[degC] + 273.15.

    中文：
        用途：把热学任务从发布方的 ℃ 换到项目统一的目标单位 K。
        输入：value —— 摄氏温度，来自表头声明为 ℃ 的列（如 Tg(℃)）。
        输出：float，开尔文；NaN 原样返回，避免把缺失值变成 273.15。
        数据来源：data/raw/polybench/Glass transition temperature.csv 等。
        配置来源：偏移量取 chainworth.protocol.constants.CELSIUS_TO_KELVIN。
        副作用：无（纯函数）。异常：无。

    English:
        Purpose: move thermal tasks from the publisher's degC to the project's target unit K.
        Args: value - a temperature in degC from a column whose header declares degC.
        Returns: float in kelvin; NaN passes through so missing values never become 273.15.
        Data source: the PolyBench thermal CSVs. Config: CELSIUS_TO_KELVIN.
        Side effects: none. Raises: none.
    """
    if value != value:            # NaN 检测：与自身不相等 / NaN check without importing math.isnan
        return value
    return value + CELSIUS_TO_KELVIN


def _identity(value: float) -> float:
    """不换算：发布单位就是目标单位，只做一次显式声明。
    Identity: the published unit already is the target unit; this only makes that explicit.

    中文：
        用途：让"我们看过这一列、决定不动"也有一条决策行，而不是靠沉默表示同意。
        输入：value —— 原始标签值。输出：同一个 float。
        数据来源：对应的 PolyBench CSV。配置来源：无。副作用：无。异常：无。

    English:
        Purpose: give "we inspected this column and chose not to convert" a decision row,
        instead of signalling agreement through silence.
        Args: value - the raw label. Returns: the same float.
        Data source: the corresponding PolyBench CSV. Config: none.
        Side effects: none. Raises: none.
    """
    return value


# ---- 4. 任务注册表 / Task registry ----

# `IWLT(â„ƒ)` 不是"某个奇怪的列名"，而是 `℃` 的 UTF-8 字节被按 CP1252 解码的结果（可逆）。
# 这里按原样重建它，而不是在源码里手打三个怪字符——手打容易在编辑器/复制粘贴中再次走样。
# `IWLT(â„ƒ)` is `℃`'s UTF-8 bytes decoded as CP1252 (a reversible artefact). Rebuild it here
# instead of typing the three odd characters by hand, which breaks on copy/paste again.
MOJIBAKE_CELSIUS = "℃".encode("utf-8").decode("cp1252")
IWLT_CONDITION_COLUMN = f"IWLT({MOJIBAKE_CELSIUS})"

TASK_SPECS: tuple[TaskSpec, ...] = (
    # ---- 主表（Proposal 表 A，6 个） / Main table (Proposal Table A, six tasks) ----
    TaskSpec(
        task_id="pb_t1_tg",
        dataset_name="Glass transition temperature",
        role="main",
        label_column="Tg(℃)",
        label_raw_unit="℃",
        target_unit="K",
        rule="T[K] = T[℃] + 273.15",
        convert=_to_kelvin,
        rationale=(
            "列名直接声明 `Tg(℃)`，这是最硬的一手证据；数值 −123 ~ 495 与高分子 Tg 的摄氏区间吻合"
            "（PDMS ≈ −125 ℃、PBI ≈ 430 ℃），若按 K 读则中位数 138 K 低于任何聚合物的 Tg，"
            "与列名和物理同时冲突。发布方另有 `property_log` 列且逐行等于 ln(Tg)，说明数据未做过单位变换"
        ),
    ),
    TaskSpec(
        task_id="pb_t1_tm",
        dataset_name="Melting temperature",
        role="main",
        label_column="Tm(℃)",
        label_raw_unit="℃",
        target_unit="K",
        rule="T[K] = T[℃] + 273.15",
        convert=_to_kelvin,
        rationale=(
            "列名声明 `Tm(℃)`；数值 −54 ~ 489、中位 185 与已知熔点区间一致"
            "（PE ≈ 130 ℃、PTFE ≈ 327 ℃、芳香族聚酰亚胺 400 ℃ 以上）。若把同一列当 K 读，"
            "中位只有 185 K（约 −88 ℃），低于任何半结晶聚合物的熔点——列名与物理同时否掉这种读法"
        ),
    ),
    TaskSpec(
        task_id="pb_t2_modulus",
        dataset_name="Tensile modulus",
        role="main",
        label_column="Tensile modulus(GPa)",
        label_raw_unit="GPa",
        target_unit="GPa",
        rule="identity：按发布单位 GPa 原样保留，不换算",
        convert=_identity,
        rationale=(
            "列名声明 `Tensile modulus(GPa)`；数值 0.16 ~ 48，log10 直方为 97 行 [0.1,1)、920 行 [1,10)、"
            "17 行 [10,100)。若其中混入 MPa，会在 160 ~ 48000 处出现另一簇，实测没有；"
            "0.16 ~ 48 GPa 也正好覆盖橡胶增韧体系到高取向纤维的量级，故不换算"
        ),
    ),
    TaskSpec(
        task_id="pb_t2_permeability",
        dataset_name="Gas permeability coefficient",
        role="main",
        label_column="P[cm3(STP)cm/(cm2*s*Pa)]",
        label_raw_unit="cm3(STP)*cm/(cm2*s*Pa)",
        target_unit="cm3(STP)*cm/(cm2*s*Pa)",
        rule="identity：按发布单位 cm3(STP)*cm/(cm2*s*Pa) 原样保留，不换算",
        convert=_identity,
        rationale=(
            "列名给出了完整量纲 `P[cm3(STP)cm/(cm2*s*Pa)]`，不存在读数歧义；"
            "换算成 Barrer（1 Barrer = 7.5006e-14 同量纲单位）后中位数约 2.7 Barrer、"
            "99 分位约 1.3e3 Barrer、最大约 5.1e4 Barrer，落在高分子气体渗透的文献区间内"
            "（阻隔型 1e-3 Barrer 到超透 PTMSP 的万级），故数值与单位自洽，不换算"
        ),
    ),
    TaskSpec(
        task_id="pb_t3_dielectric_loss",
        dataset_name="Dielectric loss factor",
        role="main",
        label_column="DLF",
        label_raw_unit="无量纲",
        target_unit="无量纲",
        rule="identity：无量纲量，按发布值原样保留",
        convert=_identity,
        rationale=(
            "列名 `DLF` 未标单位，但介电损耗因子按定义 tan δ = ε''/ε' 是无量纲量，"
            "这是物理定义而非猜测；数值 0.0013 ~ 4、中位 0.1 与聚合物 tan δ 的常见区间 1e-3 ~ 1 相符；"
            "`property_log` 逐行等于 ln(DLF)，说明发布方未做变换"
        ),
    ),
    TaskSpec(
        task_id="pb_t3_iwlt",
        dataset_name="Isothermal weight loss time",
        role="main",
        label_column="IWLT(hour)",
        label_raw_unit="hour",
        target_unit="hour",
        rule="identity：按发布单位 hour 原样保留，不换算",
        convert=_identity,
        rationale=(
            "标签列 `IWLT(hour)` 明确声明小时；数值 0.18 ~ 2500 h 与等温失重寿命的时域量级相符。"
            "同表另有一列 `IWLT(â„ƒ)`——它是 `℃` 的 UTF-8 字节被按 CP1252 解码的可逆乱码，原文为 `IWLT(℃)`，"
            "取值 100 ~ 675，属**测试条件温度**；本任务只把它作为条件列保存，不用它给标签定单位"
        ),
        condition_columns=((IWLT_CONDITION_COLUMN, "condition_temperature_celsius"),),
    ),
    # ---- 预备队（Proposal 表 E，只做单位能定死的 4 个） / Reserves (Table E, the four with a decidable unit) ----
    TaskSpec(
        task_id="pb_t1_td",
        dataset_name="Thermal decomposition temperature",
        role="reserve",
        label_column="Td(℃)",
        label_raw_unit="℃",
        target_unit="K",
        rule="T[K] = T[℃] + 273.15",
        convert=_to_kelvin,
        rationale=(
            "列名声明 `Td(℃)`；数值 159 ~ 690 ℃ 与热分解温度区间一致（PE ≈ 400 ℃、PTFE ≈ 500 ℃、"
            "芳香族聚酰亚胺 ≈ 550 ~ 600 ℃）。同表 `Thermal decomposition weight loss` 取值 0 ~ 36，"
            "是“在多少百分比失重下读数”的测试条件，不是标签，故另存为条件列"
        ),
        condition_columns=(("Thermal decomposition weight loss", "condition_weight_loss_percent"),),
    ),
    TaskSpec(
        task_id="pb_t2_tssab",
        dataset_name="Tensile stress strength at break",
        role="reserve",
        label_column="TSSAB(GPa)",
        label_raw_unit="GPa",
        target_unit="GPa",
        rule="identity：按发布单位 GPa 原样保留，不换算",
        convert=_identity,
        rationale=(
            "列名是 `TSSAB(GPa) `（尾部有一个空格，匹配前 strip）；数值 0.0117 ~ 0.193 GPa"
            "即 11.7 ~ 193 MPa，与聚合物断裂强度的文献区间一致；若按 MPa 读则只有 0.01 ~ 0.2 MPa，"
            "比任何工程塑料低两个数量级，故单位只能是 GPa"
        ),
    ),
    TaskSpec(
        task_id="pb_t3_dlt",
        dataset_name="Dielectric loss tangent",
        role="reserve",
        label_column="DLT",
        label_raw_unit="无量纲",
        target_unit="无量纲",
        rule="identity：无量纲量，按发布值原样保留",
        convert=_identity,
        rationale=(
            "列名 `DLT` 未标单位，但介电损耗角正切按定义无量纲；数值 0.00045 ~ 0.455、"
            "中位 0.0215 与聚合物 tan δ 常见区间相符；`property_log` 逐行等于 ln(DLT)，说明未做变换"
        ),
    ),
    TaskSpec(
        task_id="pb_t3_dmp_sm",
        dataset_name="Dynamic mechanical properties storage modulus",
        role="reserve",
        label_column="DMPSM(GPa)",
        label_raw_unit="GPa",
        target_unit="GPa",
        rule="identity：按发布单位 GPa 原样保留，不换算",
        convert=_identity,
        rationale=(
            "列名声明 `DMPSM(GPa)`；数值 0.033 ~ 64.6 GPa 覆盖橡胶态（10^-2 GPa）到高结晶/刚性链"
            "（数十 GPa）的 DMA 储能模量全区间，与 GPa 口径自洽；`property_log` 逐行等于 ln(DMPSM)"
        ),
    ),
)

EXCLUDED_TASKS: tuple[ExcludedTask, ...] = (
    ExcludedTask(
        task_id="pb_t1_bandgap_chain",
        dataset_name="bandgap chain_Egc",
        property="bandgap_chain",
        reason=(
            "表头只写 `bandgap_chain`，**没有声明单位**。数值 0.0205 ~ 9.863 只在 eV 口径下"
            "才与聚合物带隙的文献区间（约 1 ~ 9 eV）相容，但“只有一种解释说得通”不等于证据。"
            "按分工书红线（不许猜），在该数据集的论文/元数据给出单位之前，本任务不进入调和产出；"
            "A-4 若要把它当预备队，须先补来源证据并新增一条决策"
        ),
    ),
)


# ---- 5. 读原始表 / Read the raw table ----


def _read_raw_table(path: Path) -> pd.DataFrame:
    """读一份原始 CSV，全部按字符串进内存，表头去首尾空格。
    Read one raw CSV entirely as strings, with header whitespace stripped.

    中文：
        用途：调和阶段绝不能"读进来就自动变成 float"——那会静默丢掉无法解析的行，
              我们要求无法解析的行**被计数**而不是被吞掉。
        输入：path —— data/raw/polybench/ 下的原始 CSV（A-2 交付，字节未改）。
        输出：DataFrame，dtype 全为 object，行序即原文件行序。
        数据来源：data/raw/polybench/<数据集名>.csv。
        配置来源：无。
        副作用：无（只读）。异常：FileNotFoundError —— 文件不存在（数据未下载）。

    English:
        Purpose: never let pandas silently coerce to float here - unparsable rows must be
        counted, not swallowed.
        Args: path - a raw CSV under data/raw/polybench/ (delivered by A-2, bytes untouched).
        Returns: an all-object DataFrame preserving the file's row order.
        Data source: data/raw/polybench/<dataset>.csv. Config: none.
        Side effects: none (read-only). Raises: FileNotFoundError when the file is absent.
    """
    frame = pd.read_csv(path, dtype=str, keep_default_na=False, na_filter=False, encoding="utf-8")
    frame.columns = [str(column).strip() for column in frame.columns]
    return frame


def _sha256_of(path: Path) -> str:
    """算文件的 sha256（分块读，避免把 66 MiB 的 jsonl 之类整块读进内存）。
    Hash a file in chunks so large files never need to fit in memory.

    中文：
        用途：给"这份 parquet / 这份原始 CSV"一个可核对的指纹。
        输入：path —— 任意文件路径。输出：64 位十六进制字符串。
        数据来源：本地文件。配置来源：无。副作用：无（只读）。
        异常：FileNotFoundError —— 文件不存在。

    English:
        Purpose: give "this parquet / this raw CSV" a checkable fingerprint.
        Args: path - any file. Returns: 64-char hex digest.
        Data source: the local file. Config: none. Side effects: none.
        Raises: FileNotFoundError when absent.
    """
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonicalise_smiles(values: Iterable[str]) -> tuple[list[str | None], int]:
    """把 SMILES 统一成 RDKit 规范形式；解析不了的位置返回 None 并计数。
    Canonicalise SMILES with RDKit; unparsable entries become None and are counted.

    中文：
        用途：去重键不能用原始字符串——同一分子可以写成 `C1=CC=CC=C1` 或 `c1ccccc1`。
              规范形式也是 A-5/P2 建图的输入格式，早统一一次比后面各自统一好。
        输入：values —— SMILES 字符串序列（已 strip）。
        输出：(规范 SMILES 列表, 解析失败个数)；失败位置为 None，长度与输入一致。
        数据来源：data/raw/polybench/*.csv 的 SMILES 列。
        配置来源：无（RDKit 默认参数；不做去盐、不做标准化，因为那会改变分子身份）。
        副作用：无（不写盘；已关闭 RDKit 的日志刷屏）。
        异常：无——非法 SMILES 不抛，计数后由调用方决定怎么办。

    English:
        Purpose: the dedupe key cannot be the raw string - one molecule can be written as
        `C1=CC=CC=C1` or `c1ccccc1`. Canonical form is also what P2 builds graphs from.
        Args: values - stripped SMILES strings.
        Returns: (canonical list, number of parse failures); failures are None, list length
        matches the input.
        Data source: the SMILES column of data/raw/polybench/*.csv.
        Config: none (RDKit defaults; no salt stripping, no standardisation - both would change
        molecular identity). Side effects: none. Raises: none.
    """
    canonical: list[str | None] = []
    failures = 0
    for text in values:
        molecule = Chem.MolFromSmiles(text)
        if molecule is None:
            canonical.append(None)
            failures += 1
        else:
            canonical.append(Chem.MolToSmiles(molecule))
    return canonical, failures


def _numeric(series: pd.Series) -> pd.Series:
    """把字符串列转成 float，无法解析的位置给 NaN（调用方必须数它，不许静默 dropna）。
    Convert a string column to float, NaN where unparsable (callers must count, never silently drop).

    中文：
        用途：统一数值解析口径，避免各任务各写一次 `pd.to_numeric(..., errors=...)`。
        输入：series —— 原始字符串列。输出：float64 Series，同索引。
        数据来源：data/raw/polybench/*.csv 的标签列。
        配置来源：无。副作用：无。异常：无。

    English:
        Purpose: one numeric-parsing policy instead of one per task.
        Args: series - a raw string column. Returns: float64 with the same index.
        Data source: label columns of the PolyBench CSVs. Config: none.
        Side effects: none. Raises: none.
    """
    return pd.to_numeric(series.astype(str).str.strip(), errors="coerce")


# ---- 6. 单个任务的调和 / Harmonise one task ----


def harmonise_task(spec: TaskSpec, *, limit: int | None = None) -> tuple[pd.DataFrame, TaskOutcome]:
    """按配方把一个任务的原始 CSV 调成 parquet 表，并把过程数字记进 TaskOutcome。
    Turn one task's raw CSV into the parquet table and record every intermediate count.

    中文：
        用途：本模块唯一改写数据的地方；顺序是"先剔不可解析、再剔无标签、最后去重"，
              这样每一步删掉的行数都能单独归因，加起来正好等于行数差。
        输入：spec —— 该任务的配方（见 TASK_SPECS）；
              limit —— 只取前 N 行，仅用于 --limit 冒烟跑；None 表示全量。
        输出：(DataFrame, TaskOutcome)。DataFrame 列见 HARMONISED_COLUMNS + 条件列。
        数据来源：data/raw/polybench/<spec.dataset_name>.csv（A-2，md5 已核）。
        配置来源：spec 本身。
        副作用：无（不写盘；写盘在 write_parquet）。
        异常：FileNotFoundError —— 原始 CSV 不在；KeyError —— label_column 或 SMILES 列
              在表头里找不到（多半是数据集换版）；ValueError —— 去重后一行不剩。

    English:
        Purpose: the only place in this module that rewrites data. Order matters: drop
        unparsable SMILES, then missing labels, then duplicates - each count is attributable
        and the three sum exactly to the row-count difference.
        Args: spec - the recipe; limit - take only the first N rows (smoke runs).
        Returns: (DataFrame, TaskOutcome); columns are HARMONISED_COLUMNS plus conditions.
        Data source: data/raw/polybench/<spec.dataset_name>.csv.
        Config: the spec itself. Side effects: none (writing happens in write_parquet).
        Raises: FileNotFoundError when the CSV is missing; KeyError when a declared column is
        absent (usually a dataset version change); ValueError when nothing survives.
    """
    source = RAW_DIR / f"{spec.dataset_name}.csv"
    frame = _read_raw_table(source)
    if limit is not None:
        frame = frame.head(limit)
    outcome = TaskOutcome(spec=spec, rows_in=len(frame), source_sha256=_sha256_of(source))

    # ---- 1. 列校验：宁可当场炸，也不要在缺列时算出一张"看起来正常"的表 ----
    # ---- 1. Column check: fail loudly rather than produce a plausible-looking table ----
    for needed in ("SMILES", spec.label_column):
        if needed not in frame.columns:
            raise KeyError(
                f"{spec.dataset_name} 缺少列 {needed!r}；现有表头 {list(frame.columns)}。"
                f"数据集换版要记 docs/DECISIONS.md / column {needed!r} missing, dataset may have changed"
            )

    frame = frame.assign(
        source_row=range(len(frame)),
        smiles_raw=frame["SMILES"].astype(str).str.strip(),
    )

    # ---- 2. SMILES 规范化和不可解析行剔除 / Canonicalise SMILES, drop unparsable ----
    canonical, n_unparsable = _canonicalise_smiles(frame["smiles_raw"])
    frame = frame.assign(smiles_canonical=canonical)
    outcome.n_unparsable_smiles = n_unparsable
    frame = frame[frame["smiles_canonical"].notna()].copy()

    # ---- 3. 标签解析与缺失剔除 / Parse labels, drop missing ----
    frame = frame.assign(label_raw=_numeric(frame[spec.label_column]))
    outcome.n_invalid_label = int(frame["label_raw"].isna().sum())
    frame = frame[frame["label_raw"].notna()].copy()

    # ---- 4. 去重：键是 (task_id, 规范 SMILES)，只在任务内部去 ----
    # ---- 4. Dedupe on (task_id, canonical SMILES), within the task only ----
    before_dedupe = len(frame)
    frame = frame.drop_duplicates(subset=["smiles_canonical"], keep="first").copy()
    outcome.n_duplicates = before_dedupe - len(frame)

    # ---- 5. 单位换算（只改数值，不改行数） / Unit conversion (values only, never rows) ----
    frame["label"] = [spec.convert(float(value)) for value in frame["label_raw"]]
    outcome.n_values_converted = int((frame["label"] != frame["label_raw"]).sum())

    # ---- 6. 组装输出列：每个数字都带得回原表 / Assemble output: every number traces back ----
    output = pd.DataFrame(
        {
            "task_id": spec.task_id,
            "dataset_name": spec.dataset_name,
            "source_file": f"data/raw/polybench/{spec.dataset_name}.csv",
            "source_row": frame["source_row"].astype("int32"),
            "polymer_name": frame["polymer_name"].astype(str) if "polymer_name" in frame.columns else "",
            "smiles_raw": frame["smiles_raw"],
            "smiles_canonical": frame["smiles_canonical"],
            "label": frame["label"].astype("float64"),
            "label_unit": spec.target_unit,
            "label_raw": frame["label_raw"].astype("float64"),
            "label_raw_unit": spec.label_raw_unit,
        }
    )
    for raw_column, new_column in spec.condition_columns:
        if raw_column in frame.columns:
            output[new_column] = _numeric(frame[raw_column]).astype("float64")
        else:
            output[new_column] = pd.NA

    if output.empty:
        raise ValueError(
            f"{spec.task_id} 调和后一行不剩 / nothing survived harmonisation for {spec.task_id}"
        )

    output = output.reset_index(drop=True)
    outcome.rows_out = len(output)
    outcome.label_stats = {
        "min": float(output["label"].min()),
        "median": float(output["label"].median()),
        "max": float(output["label"].max()),
    }

    outcome.condition_columns_used = tuple(
        (raw_column, new_column)
        for raw_column, new_column in spec.condition_columns
        if raw_column in frame.columns
    )

    # ---- 7. 旁证：property_log 是否逐行等于 ln(label) ----（只读，不进输出）
    # ---- 7. Side evidence: does property_log equal ln(label) row by row? (read-only) ----
    if "property_log" in frame.columns:
        outcome.has_property_log = True
        published_log = _numeric(frame["property_log"])
        residual = (published_log - frame["label"].map(_safe_log)).abs()
        outcome.log_residual_max = float(residual.max()) if not residual.empty else None

    return output, outcome


def _safe_log(value: float) -> float:
    """自然对数，非正值返回 NaN（避免 ln(0) 的 -inf 把最大残差算成 inf）。
    Natural log; non-positive input yields NaN so ln(0) cannot turn a max-residual into infinity.

    中文：
        用途：只用于核对 `property_log` 与标签是否一致。
        输入：value —— 正数标签。输出：ln(value) 或 NaN。
        数据来源：调用处传入的标签列。配置来源：无。副作用：无。异常：无。

    English:
        Purpose: only used to check `property_log` against the label.
        Args: value - a positive label. Returns: ln(value) or NaN.
        Data source: the label column at the call site. Config: none.
        Side effects: none. Raises: none.
    """
    if value != value or value <= 0:
        return float("nan")
    return math.log(value)


# ---- 8. 决策行 / Decision rows ----


def _decision_row(
    *,
    task_id: str,
    decision_kind: str,
    property_name: str,
    raw_unit: str,
    target_unit: str,
    rule: str,
    n_rows_affected: int,
    n_values_converted: int,
    evidence: str,
    decided_at: str,
) -> dict[str, object]:
    """组一行决策；列名与顺序必须与 DECISION_COLUMNS 完全一致。
    Build one decision row; names and order must match DECISION_COLUMNS exactly.

    中文：
        用途：把"写一行决策"这件事收成一个口，避免四个调用点各漏一列。
        输入：见参数，其中 evidence 是这条判定的依据（不许留空，验收标准①）。
        输出：dict，键为 DECISION_COLUMNS。
        数据来源：调用方传入（来自 TaskOutcome 与 spec）。
        配置来源：DECIDED_BY 常量。
        副作用：无。异常：ValueError —— evidence 为空。

    English:
        Purpose: one construction point for decision rows so no call site can drop a column.
        Args: as named; `evidence` is the justification and may not be empty (acceptance #1).
        Returns: dict keyed by DECISION_COLUMNS.
        Data source: the caller (derived from TaskOutcome and the spec).
        Config: the DECIDED_BY constant. Side effects: none.
        Raises: ValueError when evidence is empty.
    """
    if not evidence.strip():
        raise ValueError(f"{task_id}/{decision_kind} 的 evidence 不能为空 / evidence must not be empty")
    return {
        "task_id": task_id,
        "decision_kind": decision_kind,
        "property": property_name,
        "raw_unit": raw_unit,
        "target_unit": target_unit,
        "rule": rule,
        "n_rows_affected": n_rows_affected,
        "n_values_converted": n_values_converted,
        "evidence": evidence,
        "decided_by": DECIDED_BY,
        "decided_at": decided_at,
    }


def build_decisions(outcome: TaskOutcome, *, decided_at: str) -> list[dict[str, object]]:
    """把一个任务的调和过程翻译成若干条决策行。
    Translate one task's harmonisation into decision rows.

    中文：
        用途：让决策表能独立解释"为什么这张表长这样"——单位、去重、弃列、剔除各一行。
        输入：outcome —— harmonise_task 的返回；
              decided_at —— 本次判定时间（重跑时沿用旧值，见 append_unit_decisions）。
        输出：dict 列表，可直接写进 unit_decisions.csv。
        数据来源：outcome 里的计数与 spec 里的依据文本。
        配置来源：无。
        副作用：无。
        异常：ValueError —— 某条 evidence 为空（由 _decision_row 抛）。

    English:
        Purpose: let the decision table explain on its own why each table looks the way it
        does - one row for the unit, one for dedupe, one per dropped column, one per exclusion.
        Args: outcome - the TaskOutcome; decided_at - the timestamp (reused on re-runs).
        Returns: list of dicts ready for unit_decisions.csv.
        Data source: the counters in `outcome` and the rationale in the spec.
        Config: none. Side effects: none.
        Raises: ValueError when an evidence string is empty.
    """
    spec = outcome.spec
    stats = outcome.label_stats
    rows: list[dict[str, object]] = []

    # ---- 1. 单位判定（每个任务恰好一条） / The unit decision (exactly one per task) ----
    conversion_note = (
        f"已按 {spec.rule} 换算，{outcome.n_values_converted} 行数值改变；"
        if outcome.n_values_converted
        else "未做数值换算；"
    )
    log_note = ""
    if outcome.has_property_log and outcome.log_residual_max is not None:
        # property_log 是**自然对数**，不是 log10——这一点在 A-2 验收时差点被当成 log10 用。
        # property_log is the NATURAL log, not log10 - a trap worth repeating in the evidence.
        log_note = (
            f"另核 `property_log`：与 ln(label) 的最大绝对残差 {outcome.log_residual_max:.1e}，"
            "即它是标签的自然对数（不是 log10，也不能当特征用）。"
        )
    unit_evidence = (
        f"{spec.rationale}。实测：调和后 n={outcome.rows_out}，"
        f"{stats['min']:.4g} ~ {stats['max']:.4g} {spec.target_unit}，中位 {stats['median']:.4g}；"
        f"原始 {outcome.rows_in} 行、无法解析标签 {outcome.n_invalid_label} 行；{conversion_note}{log_note}"
    )
    rows.append(
        _decision_row(
            task_id=spec.task_id,
            decision_kind="label_unit",
            property_name=spec.label_column,
            raw_unit=spec.label_raw_unit,
            target_unit=spec.target_unit,
            rule=spec.rule,
            n_rows_affected=0,
            n_values_converted=outcome.n_values_converted,
            evidence=unit_evidence,
            decided_at=decided_at,
        )
    )

    # ---- 2. 去重（每个任务一条，删 0 行也要留痕） / Dedupe (one per task, even when 0 rows go) ----
    rows.append(
        _decision_row(
            task_id=spec.task_id,
            decision_kind="dedupe",
            property_name="smiles_canonical",
            raw_unit="",
            target_unit="",
            rule="键 = (task_id, RDKit 规范 SMILES)，仅任务内部去重，保留首次出现",
            n_rows_affected=outcome.n_duplicates,
            n_values_converted=0,
            evidence=(
                f"原始 {outcome.rows_in} 行，规范后唯一键 {outcome.rows_out} 个，删 {outcome.n_duplicates} 行。"
                "跨任务重复不删——同一结构被测了不同性质是合法数据（DECISIONS D-14）；"
                "全库 42,169 行只有 15,728 个唯一 SMILES，若全局去重会删掉 26,441 条合法样本。"
            ),
            decided_at=decided_at,
        )
    )

    # ---- 3. 弃列：property_log 与标签同源，留着只会让人误用 ----
    # ---- 3. Dropped columns: property_log duplicates the label; keeping it invites misuse ----
    if outcome.has_property_log:
        residual = outcome.log_residual_max
        rows.append(
            _decision_row(
                task_id=spec.task_id,
                decision_kind="column_select",
                property_name="property_log",
                raw_unit="自然对数",
                target_unit="（弃用）",
                rule="弃用：与 label 一一对应，不进入调和表",
                n_rows_affected=0,
                n_values_converted=0,
                evidence=(
                    "逐行核过 `property_log` = ln(label)"
                    + (f"，最大绝对残差 {residual:.1e}" if residual is not None else "")
                    + "；它与标签是同一信息，留在表里会被下游误当特征或重复计数。"
                    "需要对数目标时由训练侧现算，不在数据层固化。"
                ),
                decided_at=decided_at,
            )
        )

    # ---- 4. 条件列：测试温度 / 失重百分比，是语境不是标签 ----
    # ---- 4. Condition columns: test temperature / weight-loss level - context, not labels ----
    for raw_column, new_column in outcome.condition_columns_used:
        mojibake_note = (
            "该列名是 `℃` 的 UTF-8 字节被按 CP1252 解码的结果（可逆），原文应为 `IWLT(℃)`；"
            if "â" in raw_column
            else ""
        )
        rows.append(
            _decision_row(
                task_id=spec.task_id,
                decision_kind="column_select",
                property_name=raw_column,
                raw_unit=spec.label_raw_unit if "temperature" not in new_column else "℃",
                target_unit=new_column,
                rule=f"另存为条件列 {new_column}，不换算、不参与标签",
                n_rows_affected=0,
                n_values_converted=0,
                evidence=(
                    f"{mojibake_note}该列记录的是测量条件（在什么温度/失重水平下读出的标签），"
                    "不是标签本身。保留它是为了让“这个寿命是在 230 ℃ 测的”这类语境不丢失，"
                    "A-4 写任务配置时若要按条件分层可以直接用。"
                ),
                decided_at=decided_at,
            )
        )

    # ---- 5. 真正删行的两种意外，只有发生时才写，写就必须给数 ----
    # ---- 5. The two accidental drop paths; written only when non-zero, and always with counts ----
    if outcome.n_unparsable_smiles:
        rows.append(
            _decision_row(
                task_id=spec.task_id,
                decision_kind="unparsable_smiles",
                property_name="SMILES",
                raw_unit="",
                target_unit="（剔除）",
                rule="RDKit 无法解析的行整行剔除",
                n_rows_affected=outcome.n_unparsable_smiles,
                n_values_converted=0,
                evidence=(
                    f"{outcome.n_unparsable_smiles} 行 SMILES 无法被 RDKit 解析，整行剔除；"
                    "不猜结构、不补空结构——没有结构的样本在 P2 也建不出图。"
                ),
                decided_at=decided_at,
            )
        )
    if outcome.n_invalid_label:
        rows.append(
            _decision_row(
                task_id=spec.task_id,
                decision_kind="invalid_label",
                property_name=spec.label_column,
                raw_unit=spec.label_raw_unit,
                target_unit=spec.target_unit,
                rule="标签无法解析为数值的行整行剔除",
                n_rows_affected=outcome.n_invalid_label,
                n_values_converted=0,
                evidence=(
                    f"{outcome.n_invalid_label} 行标签不是数值，整行剔除；"
                    "不插补、不用均值填——本项目的标签只允许来自发布方原文。"
                ),
                decided_at=decided_at,
            )
        )
    return rows


def build_exclusion_decisions(*, decided_at: str) -> list[dict[str, object]]:
    """把按红线整列弃用的任务写成决策行（含被弃掉的行数，供不变式核对）。
    Turn the wholesale-excluded tasks into decision rows, including the rows they cost.

    中文：
        用途：让"这个任务我们主动不要"与"我们忘了它"在表里可区分；行数也要入账。
        输入：decided_at —— 判定时间。
        输出：dict 列表。
        数据来源：EXCLUDED_TASKS；行数读自 data/raw/polybench/<数据集名>.csv。
        配置来源：无。副作用：无（只读）。
        异常：FileNotFoundError —— 该 CSV 不在（数据未下载）。

    English:
        Purpose: distinguish "we deliberately dropped this task" from "we forgot it", with the
        row cost accounted for.
        Args: decided_at - timestamp. Returns: list of dicts.
        Data source: EXCLUDED_TASKS; row counts read from the raw CSV.
        Config: none. Side effects: none (read-only).
        Raises: FileNotFoundError when the CSV is absent.
    """
    rows: list[dict[str, object]] = []
    for excluded in EXCLUDED_TASKS:
        source = RAW_DIR / f"{excluded.dataset_name}.csv"
        n_rows = len(_read_raw_table(source)) if source.exists() else 0
        rows.append(
            _decision_row(
                task_id=excluded.task_id,
                decision_kind="exclude_task",
                property_name=excluded.property,
                raw_unit="未声明",
                target_unit="（弃用）",
                rule="整任务弃用：单位无证据，按“不许猜”红线不进入调和产出",
                n_rows_affected=n_rows,
                n_values_converted=0,
                evidence=f"{excluded.reason}。原始 {n_rows} 行全部不进产出。",
                decided_at=decided_at,
            )
        )
    return rows


# ---- 9. 写盘 / Writing ----


def _write_csv_atomic(rows: Sequence[Mapping[str, object]], path: Path, columns: Sequence[str]) -> None:
    """原子写 CSV：先写同目录临时文件再替换，避免中断留下半份清单。
    Write a CSV atomically: temp file in the same directory, then replace.

    中文：
        用途：决策表只能追加、不能残缺；写一半的 CSV 比没有更糟。
        输入：rows —— 行字典序列；path —— 目标路径；columns —— 列顺序。
        输出：无。
        数据来源：调用方传入。配置来源：无。
        副作用：写一个文件（并删掉临时文件）；建父目录。
        异常：KeyError —— 某行缺少 columns 里的键。

    English:
        Purpose: an append-only table must never be left half-written.
        Args: rows - row dicts; path - target; columns - column order.
        Returns: none. Data source: the caller. Config: none.
        Side effects: writes one file, creates the parent directory.
        Raises: KeyError when a row lacks a column.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    # lineterminator="\n"：.gitattributes 要求 *.csv 为 LF，csv 模块默认写 CRLF（A-2 踩过）。
    # lineterminator="\n" because .gitattributes pins *.csv to LF and csv defaults to CRLF.
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(columns), lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row[column] for column in columns})
    os.replace(temporary, path)


def append_unit_decisions(
    new_rows: Sequence[Mapping[str, object]], path: Path = DECISIONS_PATH
) -> tuple[int, int]:
    """把新判定追加进 unit_decisions.csv；同一 (task_id, decision_kind, property) 不许改，只许留。
    Append new decisions; an existing (task_id, decision_kind, property) may never be rewritten.

    中文：
        用途：README §4.2 的冻结规则要求这张表"只能追加、不能覆盖"。这里用键去重：
              已存在的行原样保留（包括 decided_at），只有内容完全相同时才认为重复；
              内容不同则报错——改口径要新增一条决策，而不是改历史。
        输入：new_rows —— 待写入的决策行；path —— 目标 CSV。
        输出：(新增行数, 复用行数)。
        数据来源：data/manifests/unit_decisions.csv（若存在）。
        配置来源：无。
        副作用：重写整份 CSV（按 DECISION_KIND_ORDER 排序，便于 diff）。
        异常：ValueError —— 同一键的新旧内容不一致。

    English:
        Purpose: README 4.2 freezes this table as append-only. Existing rows (including
        decided_at) are preserved; a changed decision must be a new row, not an edit.
        Args: new_rows - rows to write; path - target CSV.
        Returns: (rows added, rows reused).
        Data source: the existing unit_decisions.csv when present.
        Config: none. Side effects: rewrites the file, sorted for readable diffs.
        Raises: ValueError when a key reappears with different content.
    """
    existing: list[dict[str, object]] = []
    if path.exists():
        with path.open("r", encoding="utf-8", newline="") as handle:
            existing = list(csv.DictReader(handle))

    task_order = {spec.task_id: index for index, spec in enumerate(TASK_SPECS)}
    task_order.update({item.task_id: len(task_order) + index for index, item in enumerate(EXCLUDED_TASKS)})

    def sort_key(row: Mapping[str, object]) -> tuple[int, int, str]:
        kind = str(row["decision_kind"])
        return (
            task_order.get(str(row["task_id"]), 999),
            DECISION_KIND_ORDER.index(kind) if kind in DECISION_KIND_ORDER else 99,
            str(row["property"]),
        )

    def comparable(row: Mapping[str, object]) -> dict[str, object]:
        return {key: str(value) for key, value in row.items() if key != "decided_at"}

    index = {(str(row["task_id"]), str(row["decision_kind"]), str(row["property"])): row for row in existing}
    added = 0
    reused = 0
    for row in new_rows:
        key = (str(row["task_id"]), str(row["decision_kind"]), str(row["property"]))
        if key in index:
            if comparable(index[key]) != comparable(row):
                raise ValueError(
                    f"决策 {key} 已存在且内容不同。unit_decisions.csv 只能追加："
                    "改口径请新增一条决策并说明理由，不要覆盖历史 / "
                    f"decision {key} exists with different content; append a new row instead of editing history"
                )
            reused += 1
            continue
        index[key] = row
        added += 1

    merged = sorted(index.values(), key=sort_key)
    _write_csv_atomic(merged, path, DECISION_COLUMNS)
    return added, reused


def write_parquet(frame: pd.DataFrame, path: Path) -> str:
    """把调和表写成 parquet（snappy），返回写入字节的 sha256。
    Write the harmonised table as snappy parquet and return the sha256 of the bytes written.

    中文：
        用途：中间产物的落盘口；哈希是"别人重建出来的是不是同一份"的唯一凭据。
        输入：frame —— 调和表；path —— 目标 .parquet 路径。
        输出：64 位十六进制 sha256（对文件字节计算，不对 DataFrame 计算）。
        数据来源：harmonise_task 的输出。
        配置来源：无（压缩算法固定 snappy：三套环境都自带 pyarrow，snappy 是默认可用项）。
        副作用：建目录并写文件。
        异常：OSError —— 无写权限；pyarrow 缺失时抛 ImportError（P0 已保证存在）。

    English:
        Purpose: the single write point for interim tables; the hash is the only evidence that
        someone else's rebuild is byte-identical.
        Args: frame - the harmonised table; path - target .parquet.
        Returns: sha256 of the file bytes (not of the DataFrame).
        Data source: harmonise_task output.
        Config: none (snappy is fixed and available in all three environments).
        Side effects: creates the directory and writes the file.
        Raises: OSError on permission problems; ImportError if pyarrow is missing.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(path, index=False, compression="snappy")
    return _sha256_of(path)


HARMONISED_MANIFEST_COLUMNS = (
    "task_id",
    "dataset_name",
    "role",
    "source_file",
    "source_sha256",
    "rows_in",
    "rows_out",
    "n_labels",
    "label_raw_unit",
    "label_unit",
    "parquet_file",
    "parquet_sha256",
    "built_at",
)


def write_harmonised_manifest(outcomes: Sequence[TaskOutcome], path: Path) -> None:
    """写"调和清单"：一份任务一行的数字与哈希，谁都能照着核。
    Write the harmonised manifest: one row per task with counts and hashes anyone can check.

    中文：
        用途：parquet 本身不入库（data/interim/* 被 gitignore），入库的是这份清单——
              行数、单位、源文件哈希、产物哈希。重跑时若源与产物哈希都没变，沿用旧的 built_at，
              否则每次跑都会把时间戳刷成新的，git 里看起来像有人手改过（A-2 踩过同一个坑）。
        输入：outcomes —— 本次所有任务的结果；path —— 目标 CSV。
        输出：无。
        数据来源：TaskOutcome 与已存在的 harmonised_manifest.csv。
        配置来源：无。
        副作用：写一份 CSV。
        异常：KeyError —— 某行缺列（由 _write_csv_atomic 抛）。

    English:
        Purpose: the parquet files themselves are gitignored; this manifest is the committed
        evidence - rows, units, source hash, artefact hash. On a re-run whose hashes are
        unchanged the old `built_at` is reused, otherwise every run would dirty the file.
        Args: outcomes - all task outcomes; path - target CSV.
        Returns: none. Data source: TaskOutcome plus any existing manifest.
        Config: none. Side effects: writes one CSV. Raises: KeyError on a missing column.
    """
    previous: dict[str, dict[str, str]] = {}
    if path.exists():
        with path.open("r", encoding="utf-8", newline="") as handle:
            previous = {row["task_id"]: row for row in csv.DictReader(handle)}

    now = datetime.now().astimezone().isoformat(timespec="seconds")
    rows: list[dict[str, object]] = []
    for outcome in outcomes:
        spec = outcome.spec
        parquet_relative = (
            str(outcome.parquet_path.relative_to(REPO_ROOT)).replace("\\", "/")
            if outcome.parquet_path is not None
            else ""
        )
        built_at = now
        old = previous.get(spec.task_id)
        if old is not None and old.get("parquet_sha256") == outcome.parquet_sha256 and old.get(
            "source_sha256"
        ) == outcome.source_sha256:
            built_at = old["built_at"]
        rows.append(
            {
                "task_id": spec.task_id,
                "dataset_name": spec.dataset_name,
                "role": spec.role,
                "source_file": f"data/raw/polybench/{spec.dataset_name}.csv",
                "source_sha256": outcome.source_sha256,
                "rows_in": outcome.rows_in,
                "rows_out": outcome.rows_out,
                "n_labels": outcome.rows_out,
                "label_raw_unit": spec.label_raw_unit,
                "label_unit": spec.target_unit,
                "parquet_file": parquet_relative,
                "parquet_sha256": outcome.parquet_sha256,
                "built_at": built_at,
            }
        )
    _write_csv_atomic(rows, path, HARMONISED_MANIFEST_COLUMNS)


# ---- 10. 编排 / Orchestration ----


def select_specs(task_ids: Sequence[str] | None = None) -> tuple[TaskSpec, ...]:
    """按任务 ID 选配方；不给就全选（6 个主表 + 4 个预备队）。
    Pick recipes by task id; with no argument, all of them (6 main + 4 reserve).

    中文：
        用途：`--task` 的落地口；写错 ID 当场报错并列出可选值，避免"以为跑了其实没跑"。
        输入：task_ids —— 任务 ID 序列，None 表示全部。
        输出：TaskSpec 元组，顺序与 TASK_SPECS 一致。
        数据来源：本模块的 TASK_SPECS 注册表。
        配置来源：无。副作用：无。
        异常：ValueError —— 出现未知 ID（消息里列出全部合法 ID）。

    English:
        Purpose: the `--task` implementation; an unknown id fails loudly and lists the valid ones
        so a typo cannot look like a completed run.
        Args: task_ids - ids, or None for all.
        Returns: tuple of TaskSpec in registry order.
        Data source: the TASK_SPECS registry. Config: none. Side effects: none.
        Raises: ValueError on an unknown id, with the valid ids in the message.
    """
    if not task_ids:
        return TASK_SPECS
    known = {spec.task_id for spec in TASK_SPECS}
    unknown = [task_id for task_id in task_ids if task_id not in known]
    if unknown:
        raise ValueError(f"未知任务 ID {unknown}；可选：{sorted(known)} / unknown task ids")
    wanted = set(task_ids)
    return tuple(spec for spec in TASK_SPECS if spec.task_id in wanted)


def harmonise_all(
    *,
    task_ids: Sequence[str] | None = None,
    limit: int | None = None,
    dry_run: bool = False,
) -> list[TaskOutcome]:
    """跑一遍调和：逐任务出 parquet，最后统一追加决策行、写清单。
    Run harmonisation: one parquet per task, then append decisions and write the manifest.

    中文：
        用途：脚本 01 的唯一入口，保证"数据 -> 决策 -> 清单"三步在同一次运行里完成，
              不会出现"parquet 更新了但决策表没跟上"的半成品。
        输入：task_ids —— 只跑这些任务（None = 全部）；
              limit —— 每个任务只取前 N 行，仅用于冒烟；此时产物写 *.partial.* 且**不写决策表**
                        （决策里的行数会是错的，宁可没有也不要写错）；
              dry_run —— 只算不写。
        输出：TaskOutcome 列表（按注册表顺序）。
        数据来源：data/raw/polybench/<数据集名>.csv。
        配置来源：TASK_SPECS。
        副作用：写 data/interim/harmonised/*.parquet、data/manifests/unit_decisions.csv、
                data/manifests/harmonised_manifest.csv。
        异常：FileNotFoundError —— 原始数据缺失（先跑 00_download.py）；
              ValueError —— 决策表出现内容冲突（见 append_unit_decisions）。

    English:
        Purpose: the only entry of stage 01; keeps data, decisions and manifest inside one run so
        there is never a parquet that the decision table has not caught up with.
        Args: task_ids - subset (None = all); limit - first N rows per task for smoke runs, which
        writes *.partial.* artefacts and SKIPS the decision table (its row counts would be wrong);
        dry_run - compute without writing.
        Returns: list of TaskOutcome in registry order.
        Data source: the raw PolyBench CSVs. Config: TASK_SPECS.
        Side effects: writes parquet files, unit_decisions.csv and harmonised_manifest.csv.
        Raises: FileNotFoundError when raw data is missing (run 00_download.py first);
        ValueError on a conflicting decision row.
    """
    specs = select_specs(task_ids)
    decided_at = datetime.now().astimezone().isoformat(timespec="seconds")
    outcomes: list[TaskOutcome] = []
    for spec in specs:
        LOGGER.info("调和 %s <- %s.csv", spec.task_id, spec.dataset_name)
        frame, outcome = harmonise_task(spec, limit=limit)
        LOGGER.info(
            "  %d 行 -> %d 行（重复 %d、不可解析 SMILES %d、无标签 %d）；%s：%.4g ~ %.4g %s",
            outcome.rows_in,
            outcome.rows_out,
            outcome.n_duplicates,
            outcome.n_unparsable_smiles,
            outcome.n_invalid_label,
            spec.label_column,
            outcome.label_stats["min"],
            outcome.label_stats["max"],
            spec.target_unit,
        )
        if not dry_run:
            suffix = ".partial" if limit is not None else ""
            path = INTERIM_DIR / f"{spec.task_id}_harmonised{suffix}.parquet"
            outcome.parquet_path = path
            outcome.parquet_sha256 = write_parquet(frame, path)
        outcomes.append(outcome)

    if dry_run:
        LOGGER.info("dry-run：未写任何文件 / dry run, nothing written")
        return outcomes

    if limit is not None:
        # 限量跑的决策行数会失真（rows_in 只有 N 行），所以宁可不写决策表。
        # A limited run would record wrong counts, so the decision table stays untouched.
        LOGGER.warning("--limit 模式下不写 unit_decisions.csv（行数会失真）/ decisions skipped under --limit")
        manifest_path = HARMONISED_MANIFEST_PATH.with_name("harmonised_manifest.partial.csv")
    else:
        decisions: list[dict[str, object]] = []
        for outcome in outcomes:
            decisions.extend(build_decisions(outcome, decided_at=decided_at))
        if task_ids is None:
            decisions.extend(build_exclusion_decisions(decided_at=decided_at))
        added, reused = append_unit_decisions(decisions)
        LOGGER.info("决策表：新增 %d 行、沿用 %d 行 -> %s", added, reused, DECISIONS_PATH)
        manifest_path = HARMONISED_MANIFEST_PATH

    write_harmonised_manifest(outcomes, manifest_path)
    LOGGER.info("调和清单 -> %s", manifest_path)
    return outcomes


# ---- 11. 验收 / Checks ----


def verify_outputs(
    *,
    manifest_path: Path = HARMONISED_MANIFEST_PATH,
    decisions_path: Path = DECISIONS_PATH,
) -> list[str]:
    """把分工书 A-3 的四条验收标准做成可执行的检查，返回问题清单（空 = 全过）。
    Turn A-3's four acceptance criteria into executable checks; return the problem list (empty = pass).

    中文：
        用途：验收不靠人眼扫表。检查项：
              ① 每个任务恰好一条 label_unit 决策、全部决策的 evidence 非空；
              ② 主表任务可用标签 >= MAIN_TABLE_MIN_LABELS；
              ③ 每个任务 rows_in - rows_out == sum(n_rows_affected)（决策表自己的账要平）；
              ④ parquet 里 label 必须等于 convert(label_raw)，且 smiles_canonical 任务内唯一；
              另加：parquet 的 sha256 与清单一致（别人重建的能否对上）。
        输入：manifest_path —— 调和清单；decisions_path —— 决策表。
        输出：字符串列表；空列表表示全过，否则每条说明哪个任务哪一项不满足。
        数据来源：data/manifests/harmonised_manifest.csv、unit_decisions.csv、
                  data/interim/harmonised/*.parquet、data/raw/polybench/*.csv。
        配置来源：TASK_SPECS、MAIN_TABLE_MIN_LABELS。
        副作用：无（只读）。
        异常：无——文件缺失也走"问题清单"，由调用方决定退出码。

    English:
        Purpose: acceptance should not depend on a human scanning a table. Checks:
        (1) exactly one label_unit decision per task and every evidence string non-empty;
        (2) main-table tasks have at least MAIN_TABLE_MIN_LABELS labels;
        (3) rows_in - rows_out equals the sum of n_rows_affected for each task;
        (4) label equals convert(label_raw) row by row, and smiles_canonical is unique per task;
        plus parquet sha256 matching the manifest.
        Args: manifest and decision table paths.
        Returns: list of problems; empty means everything passed.
        Data source: the manifest, the decision table, the parquet files and the raw CSVs.
        Config: TASK_SPECS, MAIN_TABLE_MIN_LABELS. Side effects: none.
        Raises: none - missing files become problems, and the caller decides the exit code.
    """
    problems: list[str] = []
    if not manifest_path.exists():
        return [f"没有调和清单 {manifest_path}；先跑 scripts/01_harmonise.py / manifest missing"]
    if not decisions_path.exists():
        return [f"没有决策表 {decisions_path}；先跑 scripts/01_harmonise.py / decisions missing"]

    with manifest_path.open("r", encoding="utf-8", newline="") as handle:
        manifest = {row["task_id"]: row for row in csv.DictReader(handle)}
    with decisions_path.open("r", encoding="utf-8", newline="") as handle:
        decisions = list(csv.DictReader(handle))

    # ---- 1. 决策表：每个任务恰好一条单位判定；evidence 一律非空 ----
    # ---- 1. Decisions: exactly one unit row per task; no empty evidence ----
    for row in decisions:
        if not str(row.get("evidence", "")).strip():
            problems.append(f"决策 {row.get('task_id')}/{row.get('decision_kind')} 的 evidence 为空")
    for spec in TASK_SPECS:
        unit_rows = [
            row for row in decisions if row["task_id"] == spec.task_id and row["decision_kind"] == "label_unit"
        ]
        if len(unit_rows) != 1:
            problems.append(f"{spec.task_id}: label_unit 决策有 {len(unit_rows)} 条，应为 1 条")
    for excluded in EXCLUDED_TASKS:
        excluded_rows = [
            row for row in decisions if row["task_id"] == excluded.task_id and row["decision_kind"] == "exclude_task"
        ]
        if len(excluded_rows) != 1:
            problems.append(f"{excluded.task_id}: exclude_task 决策有 {len(excluded_rows)} 条，应为 1 条")

    # ---- 2/3/4. 逐任务核表 / Per-task table checks ----
    for spec in TASK_SPECS:
        record = manifest.get(spec.task_id)
        if record is None:
            problems.append(f"{spec.task_id}: 调和清单里没有这一行")
            continue
        parquet_path = REPO_ROOT / record["parquet_file"]
        if not parquet_path.exists():
            problems.append(f"{spec.task_id}: 找不到 {record['parquet_file']}（parquet 不入库，换机器要重跑 01）")
            continue
        if _sha256_of(parquet_path) != record["parquet_sha256"]:
            problems.append(f"{spec.task_id}: parquet 的 sha256 与清单不符（有人改过表或换了环境）")
        frame = pd.read_parquet(parquet_path)

        rows_in, rows_out = int(record["rows_in"]), int(record["rows_out"])
        if len(frame) != rows_out:
            problems.append(f"{spec.task_id}: 表里 {len(frame)} 行，清单写 {rows_out} 行")
        if spec.role == "main" and rows_out < MAIN_TABLE_MIN_LABELS:
            problems.append(
                f"{spec.task_id}: 可用标签 {rows_out} < {MAIN_TABLE_MIN_LABELS}（应触发备用任务替换）"
            )

        affected = sum(
            int(row["n_rows_affected"])
            for row in decisions
            if row["task_id"] == spec.task_id
        )
        if rows_in - rows_out != affected:
            problems.append(
                f"{spec.task_id}: 行数差 {rows_in - rows_out} != 决策表 n_rows_affected 之和 {affected}"
            )

        expected = [spec.convert(float(value)) for value in frame["label_raw"]]
        if not all(
            (left == right) or (left != left and right != right) for left, right in zip(expected, frame["label"])
        ):
            problems.append(f"{spec.task_id}: label 与 convert(label_raw) 不一致（换算规则与表里数据脱节）")
        if not frame["smiles_canonical"].is_unique:
            problems.append(f"{spec.task_id}: smiles_canonical 任务内有重复，去重没生效")
        if set(frame["label_unit"]) != {spec.target_unit}:
            problems.append(f"{spec.task_id}: label_unit 列不是 {spec.target_unit}")

    return problems


# ---- 12. 抽检 / Spot check ----


def build_spotcheck(
    *,
    per_task: int = 2,
    seed: int = 0,
    manifest_path: Path = HARMONISED_MANIFEST_PATH,
) -> tuple[list[dict[str, object]], list[str]]:
    """从每个调和表里抽若干行，回到原始 CSV 的同一行重新读一遍，逐项比对。
    Sample rows from each harmonised table, re-read the same row from the raw CSV, compare field by field.

    中文：
        用途：验收标准④要求"抽 20 行人工回查原始 CSV"。人工回查要有可查的对象，所以先把
              "原始值 / 调和值 / 按规则重算值"三列并排写出来——人只需要看它们是否相同，
              不需要自己去 CSV 里找第 3,881 行。
        输入：per_task —— 每个任务抽几行（默认 2，10 个任务合计 20）；
              seed —— 抽样随机种子，固定 0，保证"同一份数据抽同样的行"，便于复现与复核；
              manifest_path —— 调和清单（用来定位每个任务的 parquet）。
        输出：(抽检行列表, 问题列表)；问题列表空表示 20 行全对。
        数据来源：data/manifests/harmonised_manifest.csv、
                  data/interim/harmonised/*.parquet、data/raw/polybench/*.csv。
        配置来源：无（抽样参数由命令行给）。
        副作用：无（不写盘；写 CSV 由调用方做）。
        异常：FileNotFoundError —— 清单或 parquet 缺失（先跑一次全量 01）。

    English:
        Purpose: acceptance criterion 4 asks for 20 rows checked by hand against the raw CSV.
        A hand check needs something to look at, so this writes raw value / harmonised value /
        recomputed value side by side: the reviewer compares three numbers instead of hunting
        for row 3,881 in a CSV.
        Args: per_task - rows per task (2 x 10 tasks = 20); seed - fixed sampling seed so the same
        data yields the same rows; manifest_path - used to locate each parquet.
        Returns: (spot-check rows, problems); an empty problem list means all 20 matched.
        Data source: the harmonised manifest, the parquet files and the raw CSVs.
        Config: none (sampling comes from the command line). Side effects: none.
        Raises: FileNotFoundError when the manifest or a parquet is missing.
    """
    if not manifest_path.exists():
        raise FileNotFoundError(f"没有调和清单 {manifest_path}；先跑一次全量 01 / run stage 01 first")
    with manifest_path.open("r", encoding="utf-8", newline="") as handle:
        manifest = {row["task_id"]: row for row in csv.DictReader(handle)}

    rng = np.random.default_rng(seed)
    rows: list[dict[str, object]] = []
    problems: list[str] = []
    for spec in TASK_SPECS:
        record = manifest.get(spec.task_id)
        if record is None:
            problems.append(f"{spec.task_id}: 清单里没有这一行")
            continue
        frame = pd.read_parquet(REPO_ROOT / record["parquet_file"])
        raw_frame = _read_raw_table(RAW_DIR / f"{spec.dataset_name}.csv")
        take = min(per_task, len(frame))
        positions = rng.choice(len(frame), size=take, replace=False)
        for position in sorted(int(value) for value in positions):
            row = frame.iloc[position]
            source_row = int(row["source_row"])
            raw_row = raw_frame.iloc[source_row]
            raw_label = float(str(raw_row[spec.label_column]).strip())
            recomputed = spec.convert(raw_label)
            checks = {
                "smiles": str(row["smiles_raw"]) == str(raw_row["SMILES"]).strip(),
                "label_raw": float(row["label_raw"]) == raw_label,
                "conversion": float(row["label"]) == recomputed,
                "name": str(row["polymer_name"]) == str(raw_row.get("polymer_name", "")).strip()
                if "polymer_name" in raw_frame.columns
                else True,
            }
            all_match = all(checks.values())
            rows.append(
                {
                    "task_id": spec.task_id,
                    "dataset_name": spec.dataset_name,
                    "source_row": source_row,
                    "smiles_in_csv": str(raw_row["SMILES"]).strip(),
                    "smiles_in_parquet": str(row["smiles_raw"]),
                    "label_raw_in_csv": raw_label,
                    "label_raw_in_parquet": float(row["label_raw"]),
                    "label_recomputed": recomputed,
                    "label_in_parquet": float(row["label"]),
                    "unit": spec.target_unit,
                    "all_match": all_match,
                }
            )
            if not all_match:
                problems.append(
                    f"{spec.task_id} 第 {source_row} 行对不上：{ {k: v for k, v in checks.items() if not v} }"
                )
    return rows, problems


SPOTCHECK_COLUMNS = (
    "task_id",
    "dataset_name",
    "source_row",
    "smiles_in_csv",
    "smiles_in_parquet",
    "label_raw_in_csv",
    "label_raw_in_parquet",
    "label_recomputed",
    "label_in_parquet",
    "unit",
    "all_match",
)


def write_spotcheck(rows: Sequence[Mapping[str, object]], path: Path) -> None:
    """把抽检结果写成 CSV（入库：这是"抽过、且全对"的凭据）。
    Write the spot-check result as a CSV - the committed evidence that the sample was drawn and matched.

    中文：
        用途：让"我们抽了 20 行"留下可核的痕迹，而不是只在日志里说一句"全对"。
        输入：rows —— build_spotcheck 的行；path —— 目标 CSV。
        输出：无。数据来源：build_spotcheck。配置来源：无。
        副作用：写一份 CSV。异常：KeyError —— 某行缺列。

    English:
        Purpose: leave checkable evidence that the sample was drawn, rather than a log line
        claiming everything matched.
        Args: rows - build_spotcheck output; path - target CSV. Returns: none.
        Data source: build_spotcheck. Config: none. Side effects: writes one CSV.
        Raises: KeyError on a missing column.
    """
    _write_csv_atomic(rows, path, SPOTCHECK_COLUMNS)
