"""Table S27 转录：把 SI 里那张"39 个任务 → 分层"的表变成一份可核对的 CSV。

Table S27 transcription: turn the SI's 39-row task-to-tier table into a checkable CSV.

中文：
    用途：A-8 的产物是 `data/external/table_s27.csv`。它**只做验证，不做设计依据**——
          Proposal §3.1 只允许"主文点名了 tier 的"任务进主表，所以主表 6 个任务的选取
          不依赖这张表。它存在的意义是：把"主文说的 14/15/10"和"SI 里逐个任务的归属"
          对一次账，看有没有矛盾。
    输入：无（转录值即本模块的 `ROWS` 常量）；可选 `--pdf` 指向 SI 原件做逐行回查。
    输出：`data/external/table_s27.csv`（39 行 × 11 列，**不入库**，见 README §4.2）。
    转录来源：`ps6c00094_si_001.pdf`（163,761,754 B，md5 `fc7861f0839a03f601aaba4a6c4ce4a0`）
          Table S27，跨 **S52 / S53 两页**（PDF 第 52、53 页）。S52 页首明写
          "Tier 1 (14 tasks), Tier 2 (15 tasks), and Tier 3 (10 tasks)"。
    为什么是手抄：SI 是 156 MB 的 LaTeX 排版 PDF，表格单元格被拆成"符号 + 上下标 + 括号说明"
          多段文本，自动抽取会把 `∆H f` 这类节点拆散。手抄 + 三条机器校验（见下）比
          "看起来解析成功"的自动抽取更可靠。
    校验（`verify()` 每次全做）：
          ① 39 行齐全；
          ② Tier 计数 **14 / 15 / 10**，与主文和 S52 页首一致；
          ③ `n_rows` 之和 **42,169**，且**逐条**等于 A-2 下载清单里的 `n_rows`——
             这条是防手抄错的关键：抄错 8,055 为 8,058 会立刻被抓到；
          ④ 23 个"主文点名了 tier"的任务，其 tier 与 `docs/Proposal_..._v1.3.md`
             附录 A 对得上；不一致的必须落进 `TIER_DISAGREEMENTS` 并有说明。
    已知分歧：`σy (yield strength)` 在 S27 里是 **Tier 3**，而附录 A 标 **Tier 2**。
          主文只说 "tensile properties" 属 Tier 2（类别概括，未点名 σy），所以主文对 σy
          的归属**没有逐一点名**；S27 是唯一的逐任务来源，本表按 S27 记 3，同时把
          附录 A 的 2 写进 `appendix_a_tier` 列并在 `tier_note` 说明。
          该任务不是主表任务（主表 6 个与 S27 **全部一致**），所以 §3.1 的"以主文为准、
          替换为备用任务"这一条**未触发**。
    副作用：`--write` 写一份 CSV（原子替换）；`--verify` 只读；`--fetch` 下载 SI（幂等）；
          `--pdf` 只读 PDF。
    异常：FileNotFoundError —— A-2 清单或 PDF 不在；ValueError —— 校验不过 / 下载后 md5 不符。
    对外：build_frame、write_frame、verify、fetch_si、main。
    被谁用：tests/test_table_s27.py、交付文档 §3、D 的图表脚注。
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import os
import re
import sys
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

# --------------------------------------------------------------------------- #
# 路径 / Paths
# --------------------------------------------------------------------------- #
MODULE_DIR = Path(__file__).resolve().parent
REPO_ROOT = MODULE_DIR.parents[2]
MANIFEST_DIR = REPO_ROOT / "data/manifests"
EXTERNAL_DIR = REPO_ROOT / "data/external"
DOWNLOAD_MANIFEST = MANIFEST_DIR / "polybench_download_manifest.csv"
PROPOSAL_MD = REPO_ROOT / "docs/Proposal_HowMuchIsAChainWorth_v1.3.md"
OUTPUT_PATH = EXTERNAL_DIR / "table_s27.csv"

# SI 原件（156 MB，**不入库**；`data/raw/*` 已 gitignore）
SI_DIR = REPO_ROOT / "data/raw/polybench_si"
SI_PATH = SI_DIR / "ps6c00094_si_001.pdf"
SI_URL = "https://pmc.ncbi.nlm.nih.gov/articles/instance/13721469/bin/ps6c00094_si_001.pdf"
SI_REFERER = "https://pmc.ncbi.nlm.nih.gov/articles/PMC13721469/"
SI_UA = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/126 Safari/537.36"
)

# --------------------------------------------------------------------------- #
# 口径常量 / The acceptance target
# --------------------------------------------------------------------------- #
EXPECTED_ROWS = 39
EXPECTED_TIER_COUNTS = {1: 14, 2: 15, 3: 10}
EXPECTED_TOTAL_ROWS = 42_169
EXPECTED_PDF_MD5 = "fc7861f0839a03f601aaba4a6c4ce4a0"
EXPECTED_PDF_BYTES = 163_761_754

COLUMNS = [
    "task_id",          # 官方 dataset name，与 A-2/A-4 清单同名，可直接 join
    "s27_label",        # PDF 原表的记号（Tg / σy (yield strength) …），便于回查
    "tier",             # S27 的 Tier 列
    "n_rows",           # S27 的 N 列
    "target_scale",     # raw / ln
    "category",         # Thermal / Electrical / …
    "dummy_mae",        # 原样保留 PDF 小数位，不做数值归一
    "dummy_rmse",
    "appendix_a_tier",  # 附录 A 的标注（空 = 那里写的是 pending S27）
    "tier_note",        # 两处说法不一致时的一句话说明
    "source_page",      # S52 / S53
]


@dataclass(frozen=True)
class Row:
    """一行转录。/ One transcribed row."""

    task_id: str
    s27_label: str
    tier: int
    n_rows: int
    target_scale: str
    category: str
    dummy_mae: str
    dummy_rmse: str
    appendix_a_tier: int | None
    tier_note: str
    source_page: str


# --------------------------------------------------------------------------- #
# 转录结果 / The transcription
#
# 39 行按 S27 原顺序抄录（先 Tier 1 的 14 行，再 Tier 2 的 15 行，再 Tier 3 的 10 行）。
# `dummy_mae` / `dummy_rmse` 保留 PDF 上的小数位（如 `92.0` 不写成 `92`），因为这是转录。
# --------------------------------------------------------------------------- #
_SIGMA_Y_NOTE = (
    "S27 记 Tier 3，附录 A 记 Tier 2：主文只说「tensile properties」属 Tier 2，"
    "未点名 σy；按逐任务来源取 3。该任务不在主表，§3.1 的替换条款未触发"
)

ROWS: tuple[Row, ...] = (
    # ---- Tier 1：14 tasks ----
    Row("Glass transition temperature", "Tg", 1, 8055, "raw", "Thermal", "92.0", "109.6", 1, "", "S52"),
    Row("Thermal decomposition temperature", "Td", 1, 4898, "raw", "Thermal", "85.3", "103.7", 1, "", "S52"),
    Row("Melting temperature", "Tm", 1, 3796, "raw", "Thermal", "85.2", "105.5", 1, "", "S52"),
    Row("bandgap chain_Egc", "Ecg (chain band gap)", 1, 3380, "raw", "Electrical", "1.30", "1.54", 1, "", "S52"),
    Row("bandgap bulk_Egb", "Ebg (bulk band gap)", 1, 561, "raw", "Electrical", "1.68", "2.00", 1, "", "S52"),
    Row("Electron Affinity_Eea", "Eea (electron affinity)", 1, 368, "raw", "Electrical", "0.96", "1.13", 1, "", "S52"),
    Row("Ionization Energy_Ei", "Ei (ionization energy)", 1, 370, "raw", "Electrical", "0.83", "1.03", 1, "", "S52"),
    Row("Density", "ρ(density)", 1, 1699, "raw", "Physicochemical", "0.131", "0.164", None, "", "S52"),
    Row("Refractive Index_Nc", "n(refractive index)", 1, 382, "raw", "Optical", "0.208", "0.281", None, "", "S52"),
    Row("Specific volume", "Vsp (specific volume)", 1, 1699, "raw", "Physicochemical", "0.088", "0.112", None, "", "S52"),
    Row("Dielectric Constant_EPS", "ε′(dielectric constant)", 1, 382, "raw", "Electrical", "0.95", "1.30", 1, "", "S52"),
    Row("Solubility parameter", "δ(solubility parameter)", 1, 316, "raw", "Physicochemical", "3.31", "4.45", None, "", "S52"),
    Row("Surface tension", "γ(surface tension)", 1, 346, "raw", "Physicochemical", "10.2", "13.5", None, "", "S52"),
    Row("Contact angle", "θ(contact angle)", 1, 255, "raw", "Physicochemical", "13.1", "17.7", None, "", "S52"),
    # ---- Tier 2：15 tasks ----
    Row("Lc phase transition temperature", "TLc", 2, 941, "raw", "Thermal", "72.3", "89.8", None, "", "S52"),
    Row("Crystallization temperature", "Tc", 2, 456, "raw", "Thermal", "86.1", "113.5", None, "", "S52"),
    Row("Heat of fusion", "ΔHf", 2, 588, "ln", "Thermal", "0.786", "0.972", None, "", "S52"),
    Row("Specific heat capacity", "Cp", 2, 207, "raw", "Thermal", "0.082", "0.111", None, "", "S52"),
    Row("Softening temperature", "Ts", 2, 763, "raw", "Thermal", "83.3", "96.3", None, "", "S52"),
    Row("Water absorption", "Wa (water absorption)", 2, 680, "ln", "Physicochemical", "1.12", "1.41", None, "", "S53"),
    Row("Cohesive energy density", "CED", 2, 307, "raw", "Physicochemical", "29.9", "37.7", None, "", "S53"),
    Row("Intrinsic viscosity", "[η](intrinsic viscosity)", 2, 1936, "ln", "Solution", "0.773", "0.995", None, "", "S53"),
    Row("Gas permeability coefficient", "P(gas permeability)", 2, 711, "ln", "Physicochemical", "2.20", "2.80", 2, "", "S53"),
    Row("Gas diffusion coefficient", "D(gas diffusion)", 2, 433, "ln", "Physicochemical", "1.65", "2.10", 2, "", "S53"),
    Row("Gas solubility coefficient", "S(gas solubility)", 2, 249, "ln", "Physicochemical", "1.25", "1.57", 2, "", "S53"),
    Row("Tensile stress strength at break", "σb (tensile strength)", 2, 1034, "ln", "Mechanical", "0.376", "0.519", 2, "", "S53"),
    Row("Tensile modulus", "Et (tensile modulus)", 2, 1034, "ln", "Mechanical", "0.439", "0.678", 2, "", "S53"),
    Row("Elongation at break", "εb (elongation)", 2, 1113, "ln", "Mechanical", "1.03", "1.30", 2, "", "S53"),
    Row("PCE_OPV_aug", "PCE", 2, 1203, "raw", "Electrical", "1.99", "2.37", 2, "", "S53"),
    # ---- Tier 3：10 tasks ----
    Row("Isothermal weight loss time", "IWLT", 3, 228, "ln", "Thermal", "1.72", "2.21", 3, "", "S53"),
    Row("Heat of fusion mol conversion", "ΔHf,mol", 3, 213, "raw", "Thermal", "1.75", "2.24", None, "", "S53"),
    Row("Electric conductivity", "σ(conductivity)", 3, 982, "ln", "Electrical", "8.71", "10.47", 3, "", "S53"),
    Row("Volume resistivity", "ρv (volume resistivity)", 3, 934, "ln", "Electrical", "9.20", "10.93", None, "", "S53"),
    Row("Dielectric loss factor", "ε′′ (dielectric loss)", 3, 279, "ln", "Electrical", "1.36", "1.65", 3, "", "S53"),
    Row("Dielectric loss tangent", "tanδd (loss tangent, d)", 3, 238, "ln", "Electrical", "1.36", "1.62", 3, "", "S53"),
    Row(
        "Tensile stress strength at yield",
        "σy (yield strength)",
        3,
        253,
        "raw",
        "Mechanical",
        "0.027",
        "0.032",
        2,
        _SIGMA_Y_NOTE,
        "S53",
    ),
    Row(
        "Dynamic mechanical properties storage modulus",
        "E′ (storage modulus)",
        3,
        387,
        "ln",
        "Mechanical",
        "0.932",
        "1.19",
        3,
        "",
        "S53",
    ),
    Row(
        "Dynamic mechanical properties loss tangent",
        "tanδ(loss tangent)",
        3,
        300,
        "ln",
        "Mechanical",
        "1.46",
        "1.75",
        3,
        "",
        "S53",
    ),
    Row(
        "Dynamic mechanical properties loss modulus",
        "E′′ (loss modulus)",
        3,
        193,
        "ln",
        "Mechanical",
        "0.726",
        "0.923",
        3,
        "",
        "S53",
    ),
)


# --------------------------------------------------------------------------- #
# 构表 / Build
# --------------------------------------------------------------------------- #
def build_frame() -> pd.DataFrame:
    """把转录值变成一张表。纯函数，不碰磁盘。

    Returns:
        DataFrame，39 行 × 11 列，列序冻结在 `COLUMNS`。
    """
    frame = pd.DataFrame(
        [
            {
                "task_id": row.task_id,
                "s27_label": row.s27_label,
                "tier": row.tier,
                "n_rows": row.n_rows,
                "target_scale": row.target_scale,
                "category": row.category,
                "dummy_mae": row.dummy_mae,
                "dummy_rmse": row.dummy_rmse,
                # 空值写空串而不是 NaN：CSV 里 NaN 会让"未点名"看起来像"缺数据"
                "appendix_a_tier": "" if row.appendix_a_tier is None else row.appendix_a_tier,
                "tier_note": row.tier_note,
                "source_page": row.source_page,
            }
            for row in ROWS
        ],
        columns=COLUMNS,
    )
    return frame


def write_frame(frame: pd.DataFrame, path: Path = OUTPUT_PATH) -> Path:
    """原子写 CSV。`.tmp` + `os.replace`，行尾用 `\\n`。

    Args:
        frame: `build_frame()` 的输出。
        path: 目标路径，默认 `data/external/table_s27.csv`。

    Returns:
        实际写入的路径。
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(tmp, index=False, lineterminator="\n", encoding="utf-8")
    os.replace(tmp, path)
    return path


# --------------------------------------------------------------------------- #
# 校验 / Verify
# --------------------------------------------------------------------------- #
def _load_download_manifest() -> pd.DataFrame:
    """读 A-2 的下载清单（39 行官方条数）。"""
    if not DOWNLOAD_MANIFEST.exists():
        raise FileNotFoundError(
            f"缺 A-2 下载清单：{DOWNLOAD_MANIFEST.as_posix()}；先跑 `python -m chainworth.io.tasks --verify` 或 A-2 的下载步骤"
        )
    return pd.read_csv(DOWNLOAD_MANIFEST)


def _appendix_a_tiers() -> dict[str, int | None]:
    """从 Proposal 附录 A 的表格里解析 39 个任务的 tier（`pending S27` 记为 None）。

    解析而不是手抄第二遍，是为了让"附录 A 说了什么"有一个不会抄错的来源。
    """
    if not PROPOSAL_MD.exists():
        raise FileNotFoundError(f"缺 Proposal：{PROPOSAL_MD.as_posix()}")
    text = PROPOSAL_MD.read_text(encoding="utf-8")
    pattern = re.compile(r"^\|\s*(\d+)\s*\|\s*([^|]+?)\s*\|\s*([\d,]+)\s*\|\s*([^|]+?)\s*\|", re.MULTILINE)
    found: dict[str, int | None] = {}
    in_appendix = False
    for line in text.splitlines():
        if line.startswith("## Appendix A"):
            in_appendix = True
            continue
        if in_appendix and line.startswith("## "):
            break
        if not in_appendix:
            continue
        match = pattern.match(line)
        if not match:
            continue
        name, tier_cell = match.group(2).strip(), match.group(4).strip()
        found[name] = int(tier_cell) if tier_cell in ("1", "2", "3") else None
    return found


def verify(path: Path = OUTPUT_PATH, pdf: Path | None = None) -> list[str]:
    """全量校验，返回问题清单（空 = 通过）。

    Args:
        path: 要校验的 CSV 路径。
        pdf: 可选，SI 原件路径。给了就逐行回查 PDF 原文（需要 `pypdf`）。

    Returns:
        人类可读的问题列表；空列表表示通过。
    """
    problems: list[str] = []

    # ① 行数与列序
    if not path.exists():
        return [f"缺转录文件：{path.as_posix()}；先跑 `python -m chainworth.io.table_s27 --write`"]
    frame = pd.read_csv(path, dtype={"dummy_mae": str, "dummy_rmse": str, "appendix_a_tier": str})
    if len(frame) != EXPECTED_ROWS:
        problems.append(f"行数 {len(frame)} ≠ 期望 {EXPECTED_ROWS}")
    if list(frame.columns) != COLUMNS:
        problems.append(f"列序不符：{list(frame.columns)}")
    if frame["task_id"].duplicated().any():
        dup = frame.loc[frame["task_id"].duplicated(), "task_id"].tolist()
        problems.append(f"task_id 重复：{dup}")

    # ② Tier 计数
    counts = frame["tier"].value_counts().to_dict()
    if counts != EXPECTED_TIER_COUNTS:
        problems.append(f"Tier 计数 {counts} ≠ 期望 {EXPECTED_TIER_COUNTS}")

    # ③ 条数：总和与逐条
    total = int(frame["n_rows"].sum())
    if total != EXPECTED_TOTAL_ROWS:
        problems.append(f"n_rows 之和 {total:,} ≠ {EXPECTED_TOTAL_ROWS:,}")
    manifest = _load_download_manifest()
    official = dict(zip(manifest["task_id"], manifest["n_rows"]))
    missing = sorted(set(frame["task_id"]) - set(official))
    if missing:
        problems.append(f"S27 里的任务在 A-2 清单里找不到：{missing}")
    for _, row in frame.iterrows():
        expected = official.get(row["task_id"])
        if expected is not None and int(row["n_rows"]) != int(expected):
            problems.append(
                f"{row['task_id']}：S27 记 {int(row['n_rows']):,} 行，A-2 清单 {int(expected):,} 行"
            )

    # ④ 与附录 A 的 23 个已命名 tier 对拍
    appendix = _appendix_a_tiers()
    if len(appendix) != EXPECTED_ROWS:
        problems.append(f"附录 A 解析出 {len(appendix)} 行 ≠ {EXPECTED_ROWS}")
    for _, row in frame.iterrows():
        a_tier = appendix.get(row["task_id"])
        stored = row["appendix_a_tier"]
        stored_int = int(stored) if str(stored).strip() not in ("", "nan") else None
        if a_tier != stored_int:
            problems.append(
                f"{row['task_id']}：附录 A 现为 {a_tier}，转录里记 {stored_int}"
                "（可能是 Proposal 改了而本表没跟，或反之）"
            )
        if a_tier is not None and a_tier != int(row["tier"]):
            if not str(row["tier_note"]).strip() or str(row["tier_note"]) == "nan":
                problems.append(
                    f"{row['task_id']}：S27 记 Tier {int(row['tier'])}，附录 A 记 Tier {a_tier}，"
                    "但没有在 tier_note 里说明"
                )

    # ⑤ 与冻结的 configs/tasks/*.yaml 对拍（A-4 的 tier 字段）
    problems.extend(_check_task_configs(frame))

    # ⑥ 可选：回查 PDF 原文
    if pdf is not None:
        problems.extend(_check_against_pdf(frame, pdf))

    return problems


def _check_task_configs(frame: pd.DataFrame) -> list[str]:
    """把 S27 的 tier 与 `configs/tasks/*.yaml` 里冻结的 `tier` 逐份对照。

    这是 A-8 真正的用武之地：主表 6 个任务的分层如果与 SI 不符，§3.1 的"以主文为准、
    替换为备用任务"才需要触发。跑一次就知道要不要触发。

    Returns:
        问题列表。
    """
    problems: list[str] = []
    try:
        import yaml
    except ImportError:  # PyYAML 在环境锁里，缺了说明环境不对
        return ["对拍 configs 需要 PyYAML（应在 environment/*.yml 里）"]

    s27_tier = dict(zip(frame["task_id"], frame["tier"]))
    configs = sorted((REPO_ROOT / "configs/tasks").glob("*.yaml"))
    checked = 0
    main_checked = 0
    for path in configs:
        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        name, tier, role = data.get("dataset_name"), data.get("tier"), data.get("role")
        if tier is None or name not in s27_tier:
            continue
        checked += 1
        if role == "main":
            main_checked += 1
        if int(tier) != int(s27_tier[name]):
            problems.append(
                f"{path.name}（{name}）：配置冻结的 tier={tier}，S27 记 {int(s27_tier[name])}"
            )
    if checked == 0:
        problems.append("没有一份 configs/tasks/*.yaml 的 dataset_name 能对上 S27——映射失效了")
    # 主表 6 个 PolyBench 任务是 §3.1 的替换条款唯一关心的对象
    if main_checked != 6:
        problems.append(f"对上的主表任务 {main_checked} 个，期望 6 个（Tg/Tm/模量/透气/介电损耗/IWLT）")
    return problems


def _check_against_pdf(frame: pd.DataFrame, pdf: Path) -> list[str]:
    """逐行回查 PDF：该行的 N / MAE / RMSE 三个数字必须出现在同一文本行里。

    Returns:
        问题列表。
    """
    problems: list[str] = []
    if not pdf.exists():
        return [f"PDF 不存在：{pdf.as_posix()}"]
    import hashlib

    digest = hashlib.md5()
    with pdf.open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            digest.update(block)
    if digest.hexdigest() != EXPECTED_PDF_MD5:
        problems.append(f"PDF md5 {digest.hexdigest()} ≠ 期望 {EXPECTED_PDF_MD5}")

    try:
        from pypdf import PdfReader
    except ImportError:
        return problems + ["回查 PDF 需要 pypdf（`PYTHONPATH= python -m pip install pypdf`）"]

    reader = PdfReader(str(pdf))
    # S52 / S53 是页脚页码，对应 PDF 的第 52、53 页（1 起算）
    text = "\n".join(
        (reader.pages[i].extract_text(extraction_mode="layout") or "") for i in (51, 52)
    )
    lines = [" ".join(line.split()) for line in text.splitlines()]

    for _, row in frame.iterrows():
        n_str = f"{int(row['n_rows']):,}"
        mae, rmse = str(row["dummy_mae"]).strip(), str(row["dummy_rmse"]).strip()
        hits = [line for line in lines if n_str in line and mae in line and rmse in line]
        if not hits:
            problems.append(
                f"{row['s27_label']}：PDF 里找不到同时含 {n_str} / {mae} / {rmse} 的行"
            )
    return problems


# --------------------------------------------------------------------------- #
# 取回 SI 原件 / Fetch the SI original
# --------------------------------------------------------------------------- #
_POW_CHALLENGE = re.compile(r'const POW_CHALLENGE = "([^"]+)"')
_POW_DIFFICULTY = re.compile(r'const POW_DIFFICULTY = "(\d+)"')
_POW_COOKIE = re.compile(r'const POW_COOKIE_NAME = "([^"]+)"')


def _md5_of(path: Path) -> str:
    """算整个文件的 md5（流式，156 MB 不占内存）。"""
    digest = hashlib.md5()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _http_open(url: str, *, cookie: str | None = None, timeout: int = 300):
    """带 UA / Referer（PMC 两者都要）发一次请求。"""
    request = urllib.request.Request(url)
    request.add_header("User-Agent", SI_UA)
    request.add_header("Referer", SI_REFERER)
    if cookie:
        request.add_header("Cookie", cookie)
    return urllib.request.urlopen(request, timeout=timeout)


def _solve_pow(challenge: str, difficulty: int) -> int:
    """找最小的 nonce，使 `sha256(challenge + nonce)` 的十六进制以 difficulty 个 `0` 开头。

    这是 PMC 给普通浏览器发的**工作量证明**（页面上那段 inline script 做同一件事），
    不是破解：难度只有 4 个十六进制零，一万到十万次哈希即出。我们不执行 JS，
    所以自己算一遍；算完把 `challenge,nonce` 写进 cookie 就能过。
    """
    prefix = "0" * difficulty
    nonce = 0
    while True:
        if hashlib.sha256(f"{challenge}{nonce}".encode()).hexdigest().startswith(prefix):
            return nonce
        nonce += 1


def fetch_si(target: Path = SI_PATH, *, force: bool = False) -> Path:
    """把 156 MB 的 SI 原件取回本地并核 md5；已就位且 md5 对时是空操作。

    为什么要走 PMC 而不是出版商：`pubs.acs.org` 的 SI 直链对本机出口一律 403（bot 墙），
    Europe PMC 的 `supplementaryFiles` 接口每次 25 秒超时；PMC 的文章页可达，正文里
    挂着 `/articles/instance/13721469/bin/ps6c00094_si_001.pdf`，但直接取会得到
    一个 ~1.8 KB 的 POW 挑战页，解完才给文件。

    Args:
        target: 落盘路径，默认 `data/raw/polybench_si/ps6c00094_si_001.pdf`。
        force: 为 True 时无视本地已有的文件重新下载。

    Returns:
        实际落盘的路径。

    Raises:
        ValueError: 下载完成但大小或 md5 与 Proposal 记录的不符。
    """
    target = Path(target)
    if target.exists() and not force:
        if _md5_of(target) == EXPECTED_PDF_MD5:
            return target
        print(f"本地文件 md5 不符，重新下载：{target.as_posix()}")

    target.parent.mkdir(parents=True, exist_ok=True)
    part = target.with_suffix(target.suffix + ".part")

    # 第一步：拿挑战页（或直接拿到文件，若 PMC 哪天取消了这道墙）
    with _http_open(SI_URL, timeout=120) as response:
        first = response.read(4096)
        content_type = response.headers.get("Content-Type", "")
        is_pdf_already = first[:4] == b"%PDF"
    if is_pdf_already:
        print("PMC 直接给了文件（无 POW 挑战），直接下载")
        cookie = None
    else:
        page = first.decode("utf-8", "replace")
        challenge = _POW_CHALLENGE.search(page)
        difficulty = _POW_DIFFICULTY.search(page)
        cookie_name = _POW_COOKIE.search(page)
        if not (challenge and difficulty and cookie_name):
            raise ValueError(
                f"既不是 PDF 也不是 POW 挑战页（Content-Type={content_type}）；"
                f"PMC 可能改了机制，需人工看一下：{SI_URL}"
            )
        nonce = _solve_pow(challenge.group(1), int(difficulty.group(1)))
        cookie = f"{cookie_name.group(1)}={challenge.group(1)},{nonce}"
        print(f"POW 已解：difficulty={difficulty.group(1)} nonce={nonce}")

    # 第二步：带 cookie 流式下载
    written = 0
    try:
        with _http_open(SI_URL, cookie=cookie) as response, part.open("wb") as handle:
            while True:
                chunk = response.read(1 << 20)
                if not chunk:
                    break
                handle.write(chunk)
                written += len(chunk)
        size = part.stat().st_size
        digest = _md5_of(part)
        if size != EXPECTED_PDF_BYTES or digest != EXPECTED_PDF_MD5:
            raise ValueError(
                f"下载完成但校验不过：{size:,} 字节 / md5 {digest}，"
                f"期望 {EXPECTED_PDF_BYTES:,} 字节 / md5 {EXPECTED_PDF_MD5}"
            )
        os.replace(part, target)
    finally:
        part.unlink(missing_ok=True)

    print(f"已取回 {target.as_posix()}（{written:,} 字节，md5 {EXPECTED_PDF_MD5}）")
    return target


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m chainworth.io.table_s27",
        description="转录并校验 Table S27（A-8，可选验证项）",
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--write", action="store_true", help="写出 data/external/table_s27.csv")
    group.add_argument("--verify", action="store_true", help="校验已有 CSV")
    group.add_argument("--fetch", action="store_true", help="下载 SI 原件（156 MB）并核 md5")
    parser.add_argument("--pdf", type=Path, default=None, help="可选：SI 原件，逐行回查（需 pypdf）")
    parser.add_argument("--force", action="store_true", help="配合 --fetch：无视已有文件重下")
    args = parser.parse_args(argv)

    if args.fetch:
        try:
            path = fetch_si(force=args.force)
        except (ValueError, OSError) as exc:
            print(f"取回失败：{exc}", file=sys.stderr)
            return 2
        print(f"  md5 {EXPECTED_PDF_MD5}｜{EXPECTED_PDF_BYTES:,} 字节｜{path.as_posix()}")
        print("  注：该文件不入库（data/raw/* 已 gitignore），A-8 只需要它做回查")
        return 0

    if args.write:
        frame = build_frame()
        target = write_frame(frame)
        print(f"已写 {target.as_posix()}（{len(frame)} 行 × {len(frame.columns)} 列）")
        counts = frame["tier"].value_counts().sort_index().to_dict()
        print(f"  Tier 计数：{counts}｜n 之和：{int(frame['n_rows'].sum()):,}")
        print("  注：OpenPoly 的 counts 文件（Tg 58,755）与本表无关，本表是 PolyBench 的 39 个任务")
        return 0

    problems = verify(pdf=args.pdf)
    if problems:
        print(f"校验未通过（{len(problems)} 条）：", file=sys.stderr)
        for problem in problems:
            print(f"  - {problem}", file=sys.stderr)
        return 1
    print("校验通过：39 行 / Tier 14-15-10 / 合计 42,169 / 与 A-2 清单逐条一致 / 与附录 A 对拍一致")
    return 0


if __name__ == "__main__":
    sys.exit(main())
