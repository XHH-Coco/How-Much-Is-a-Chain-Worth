"""OpenPoly 三性质计数复核：把"741 / 3,985 / 443 / 295 / 168"变成一条可复跑的命令。
Recount the OpenPoly properties: turn "741 / 3,985 / 443 / 295 / 168" into a re-runnable command.

中文：
    用途：A-7 的唯一产物是 `data/manifests/openpoly_property_counts.csv`，它是 C 的 OpenPoly 经典臂
          与 D 的 H4 的**分母**。分母错了，两边的数一起错。所以这五个数不能是"我数过了"，
          必须是"任何人跑一遍都得到同一个数"——本模块就是那一遍。
    输入：`data/raw/openpoly/data/final_polymer_properties_fromliterature.csv`（A-2 下载，741 行 × 32 列）。
    输出：`data/manifests/openpoly_property_counts.csv`（28 个性质列 × 10 列）。
    数据来源：发布方 CSV 本体，**不读任何摘要、不读论文表格、不读 `final_polymer_property_counts_
          fromliterature.csv`**——最后那个文件报 Tg 58,755，是"匹配结构之前的文献记录数"，
          比训练表高约 100 倍，用它会把可用样本夸大两个数量级。
    配置来源：`IDENTIFIER_COLUMNS` / `KEY_PROPERTIES` / `EXPECTED_*` 是口径本身，改它们等于改
          A-7 的验收靶子，必须在 `docs/DECISIONS.md` 记一条。
    副作用：`--verify` 只读；`--write` 写一份 CSV（原子替换）。
    异常：FileNotFoundError —— 原始 CSV 不在；ValueError —— 列缺失或靶子不符。
    对外：count_properties、build_frame、write_frame、verify、main。
    被谁用：tests/test_openpoly_counts.py、C 与 D 的分母、交付文档 §3。

English:
    Purpose: A-7's only artefact is `openpoly_property_counts.csv`, the denominator behind C's
    classical OpenPoly arm and D's H4. A wrong denominator breaks both. So the five numbers must
    not be "I counted them" but "anyone who runs this gets the same numbers" - this module is
    that run.
    Inputs: the publisher's CSV (741 rows x 32 columns), downloaded by A-2.
    Outputs: `data/manifests/openpoly_property_counts.csv` (28 property rows x 10 columns).
    Data source: the published CSV itself. Never the summary table, never the paper, and never
    `final_polymer_property_counts_fromliterature.csv` - that one reports Tg 58,755, i.e.
    literature records *before* structure matching, which overstates usable samples ~100x.
    Config: the IDENTIFIER_COLUMNS/KEY_PROPERTIES/EXPECTED_* constants ARE the acceptance
    target; changing them needs a `docs/DECISIONS.md` entry.
    Side effects: `--verify` read-only; `--write` writes one CSV atomically.
    Raises: FileNotFoundError, ValueError.
    Public API: count_properties, build_frame, write_frame, verify, main.
    Callers: tests/test_openpoly_counts.py, the C/D denominators, the delivery doc.
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
from pathlib import Path

import pandas as pd

# --------------------------------------------------------------------------- #
# 口径常量 / The definition of the count
# --------------------------------------------------------------------------- #

RAW_PATH = Path("data/raw/openpoly/data/final_polymer_properties_fromliterature.csv")
OUT_PATH = Path("data/manifests/openpoly_property_counts.csv")

#: 不是性质、不能计入"结构-性质标注"的列 / Identifier columns, excluded from annotation counts
IDENTIFIER_COLUMNS: tuple[str, ...] = ("Name", "PSMILES", "PSMILES_2", "PSMILES_4")

#: 分工书 §A-7 点名要核的三列 / The three columns A-7 singles out
KEY_PROPERTIES: tuple[str, ...] = ("Tg (K)", "Dielectric Constant Total", "Tm (K)")

#: 验收靶子（分工书 §A-7 验收①，与 Proposal §2.3 一致）/ Acceptance targets
EXPECTED_N_ROWS = 741
EXPECTED_N_ANNOTATIONS = 3_985
EXPECTED_NONNULL: dict[str, int] = {
    "Tg (K)": 443,
    "Dielectric Constant Total": 295,
    "Tm (K)": 168,
}

#: 表头固定顺序：前三个是分工书点名的列，其余是审计列 / Column order: spec columns first
COLUMNS: tuple[str, ...] = (
    "property",
    "n_nonnull",
    "usable_after_harmonisation",
    "n_rows",
    "harmonisation_status",
    "n_unique_structures",
    "n_duplicate_rows",
    "n_duplicate_groups",
    "n_conflicting_groups",
    "note",
)

#: A-3 从未对 OpenPoly 跑过调和，故"调和后可用"等于非空数 / A-3 never harmonised OpenPoly
HARMONISATION_STATUS = "not_run"

#: 判定"同一结构的两个标签是否矛盾"的容差 / Tolerance for callling two labels contradictory
CONFLICT_TOLERANCE = 1e-9

#: 那个不许用的文件（注意：真实文件名 from 后面**没有**下划线）/ The forbidden counts file
FORBIDDEN_FILE = "final_polymer_property_counts_fromliterature.csv"
FORBIDDEN_FILE_TG = 58_755


# --------------------------------------------------------------------------- #
# 计数
# --------------------------------------------------------------------------- #


def property_columns(frame: pd.DataFrame) -> list[str]:
    """性质列 = 表头里除去 4 个标识列的列（本版 32 - 4 = 28）。

    Purpose: derive the property columns from the header instead of hard-coding 28 names,
    so a publisher rename shows up as a changed count rather than a silent zero.
    Args: frame - the raw table. Returns: property column names, in file order.
    """
    return [name for name in frame.columns if name not in IDENTIFIER_COLUMNS]


def _canonical_psmiles(values: pd.Series) -> pd.Series:
    """PSMILES → RDKit 规范式；解析失败给 None（不静默保留原始串）。

    Purpose: two rows can spell the same polymer differently (`*OC(C)C(=O)O*` vs a reordered
    equivalent). Counting "unique structures" on raw strings overstates variety, so we
    canonicalise exactly like A-3/A-6 do.
    Args: values - raw PSMILES column. Returns: canonical strings with None where unparsable.
    """
    from rdkit import Chem, RDLogger

    RDLogger.DisableLog("rdApp.*")  # 741 个分子会刷屏，但失败会计数进 n_duplicate_rows 的反面
    out: list[str | None] = []
    for value in values:
        if value is None or (isinstance(value, float) and pd.isna(value)):
            out.append(None)
            continue
        molecule = Chem.MolFromSmiles(str(value).strip())
        out.append(Chem.MolToSmiles(molecule) if molecule is not None else None)
    return pd.Series(out, index=values.index, dtype="object")


def count_properties(frame: pd.DataFrame) -> pd.DataFrame:
    """逐列数非空，并对有点名的三列算"重复结构与标签矛盾"。

    Purpose: one row per property column. `n_nonnull` is what the acceptance targets check;
    the duplicate columns record a fact that changes how C/D may use these samples: the same
    polymer appears several times with *different* literature values.
    Args: frame - the raw OpenPoly table.
    Returns: a DataFrame with COLUMNS, in file column order.
    Raises: ValueError when a declared column is missing or the row count is not 741.
    """
    missing = [name for name in IDENTIFIER_COLUMNS if name not in frame.columns]
    if missing:
        raise ValueError(f"OpenPoly CSV 缺少标识列 {missing}；现有表头 {list(frame.columns)}")

    n_rows = len(frame)
    if n_rows != EXPECTED_N_ROWS:
        raise ValueError(
            f"行数 {n_rows} ≠ 期望 {EXPECTED_N_ROWS}；数据集换版要记 docs/DECISIONS.md"
        )

    canonical = _canonical_psmiles(frame["PSMILES"])
    rows: list[dict[str, object]] = []

    for name in property_columns(frame):
        # 非空口径：能解析成数才算有标注（与 notna() 在本数据集上完全等价，已用测试锁定）
        label = pd.to_numeric(frame[name], errors="coerce")
        present = pd.DataFrame({"structure": canonical, "label": label}).dropna()

        n_nonnull = len(present)
        unique = int(present["structure"].nunique())
        n_duplicate_rows = n_nonnull - unique

        # "行"与"组"必须分开数：2 个结构各重复 3 次是 6 行重复 / 2 组。
        # Rows and groups are different units: two structures repeated three times each is six
        # duplicated rows but only two groups.
        n_groups = 0
        n_conflicting = 0
        n_pure_redundant = 0
        if n_duplicate_rows:
            sizes = present.groupby("structure")["label"].size()
            n_groups = int((sizes > 1).sum())
            spread = present.groupby("structure")["label"].agg(lambda s: s.max() - s.min())
            n_conflicting = int((spread[sizes > 1] > CONFLICT_TOLERANCE).sum())
            n_pure_redundant = n_groups - n_conflicting

        if n_duplicate_rows:
            redundancy = (
                f"（其中 {n_pure_redundant} 组是重复登记了同一个数，可白删）"
                if n_pure_redundant
                else "（每一组都在互相矛盾）"
            )
            note = (
                f"{n_groups} 组重复结构、{n_duplicate_rows} 行冗余，{n_conflicting} 组标签不一致"
                f"{redundancy}；若按 A-3 的 (task, 规范 SMILES) 去重，n 会从 {n_nonnull} 降到 "
                f"{unique}。本表**不静默去重**，理由见 D-26"
            )
        else:
            note = "无重复结构" if n_nonnull else "整列为空"

        rows.append(
            {
                "property": name,
                "n_nonnull": n_nonnull,
                # A-3 未对 OpenPoly 执行调和 → 没有任何行被删 → 字面上的"调和后可用"等于非空数。
                # A-3 never ran on OpenPoly, so nothing was dropped and usable == non-null.
                "usable_after_harmonisation": n_nonnull,
                "n_rows": n_rows,
                "harmonisation_status": HARMONISATION_STATUS,
                "n_unique_structures": unique,
                "n_duplicate_rows": n_duplicate_rows,
                "n_duplicate_groups": n_groups,
                "n_conflicting_groups": n_conflicting,
                "note": note,
            }
        )

    return pd.DataFrame(rows, columns=list(COLUMNS))


def build_frame(raw_path: Path = RAW_PATH) -> pd.DataFrame:
    """读原始 CSV 并计数。 / Read the raw CSV and count.

    Args: raw_path - the published table. Returns: the counts frame.
    Raises: FileNotFoundError when the CSV is absent (A-2 not run).
    """
    if not raw_path.is_file():
        raise FileNotFoundError(
            f"原始 CSV 不在：{raw_path.as_posix()}；先跑 python scripts/00_download.py --source openpoly"
        )
    frame = pd.read_csv(raw_path)
    return count_properties(frame)


# --------------------------------------------------------------------------- #
# 写盘与校验
# --------------------------------------------------------------------------- #


def write_frame(frame: pd.DataFrame, path: Path = OUT_PATH) -> Path:
    """原子写 CSV（`.tmp` + `os.replace`），行尾固定 LF。 / Atomic write, LF newlines.

    Args: frame - the counts; path - destination. Returns: the written path.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False, lineterminator="\n", quoting=csv.QUOTE_MINIMAL)
    os.replace(temporary, path)
    return path


def verify(raw_path: Path = RAW_PATH, out_path: Path = OUT_PATH) -> list[str]:
    """把"清单与原始数据是否仍然一致"变成一串问题。 / Compare the manifest against a fresh count.

    Purpose: a manifest that nobody re-checks rots silently. This recomputes everything from
    the raw CSV and reports every difference; it never repairs.
    Args: raw_path - published table; out_path - the committed manifest.
    Returns: list of human-readable problems (empty means pass).
    """
    problems: list[str] = []
    if not out_path.is_file():
        return [f"清单不在：{out_path.as_posix()}；跑 python -m chainworth.io.openpoly --write"]

    fresh = build_frame(raw_path)
    stored = pd.read_csv(out_path)

    if list(stored.columns) != list(COLUMNS):
        problems.append(f"列不符：期望 {list(COLUMNS)}，实际 {list(stored.columns)}")
        return problems

    if len(stored) != len(fresh):
        problems.append(f"行数不符：清单 {len(stored)} 行，重算 {len(fresh)} 行")
        return problems

    if list(stored["property"]) != list(fresh["property"]):
        problems.append("性质列的顺序或名字与原始 CSV 不一致")

    for column in COLUMNS[1:-1]:  # 除 property 与 note 外，全部是数字
        left = stored[column].to_numpy()
        right = fresh[column].to_numpy()
        bad = [i for i in range(len(left)) if left[i] != right[i]]
        if bad:
            detail = ", ".join(
                f"{fresh['property'][i]} 清单={left[i]} 重算={right[i]}" for i in bad[:5]
            )
            problems.append(f"{column} 不符：{detail}")

    if stored["note"].fillna("").tolist() != fresh["note"].fillna("").tolist():
        problems.append("note 列与重算结果不一致（多半是数字变了但说明没跟着改）")

    # 验收靶子：五个数必须逐个对上，否则 C/D 的分母就错了
    total = int(fresh["n_nonnull"].sum())
    if total != EXPECTED_N_ANNOTATIONS:
        problems.append(f"标注总数 {total} ≠ 期望 {EXPECTED_N_ANNOTATIONS}")
    for name, expected in EXPECTED_NONNULL.items():
        got = int(fresh.loc[fresh["property"] == name, "n_nonnull"].iloc[0])
        if got != expected:
            problems.append(f"{name} 非空数 {got} ≠ 期望 {expected}")

    return problems


def main(argv: list[str] | None = None) -> int:
    """CLI：`--write` 重算并落盘，`--verify` 只读比对（默认）。 / CLI: write or verify.

    中文：
        用途：让"这五个数还能不能用"成为一条可复制的命令。
        输入：argv。输出：进程退出码（0 通过 / 1 不过）。
        副作用：`--write` 写一份 CSV；其余只写 stdout/stderr。
    """
    parser = argparse.ArgumentParser(
        prog="python -m chainworth.io.openpoly",
        description="复核 OpenPoly 的属性计数（A-7）。",
    )
    parser.add_argument("--raw", type=Path, default=RAW_PATH, help="原始 CSV 路径")
    parser.add_argument("--out", type=Path, default=OUT_PATH, help="清单路径")
    parser.add_argument("--write", action="store_true", help="重算并写清单（默认只校验）")
    parser.add_argument("--verify", action="store_true", help="只读校验（默认动作）")
    args = parser.parse_args(argv)

    if args.write:
        frame = build_frame(args.raw)
        path = write_frame(frame, args.out)
        total = int(frame["n_nonnull"].sum())
        print(
            f"[OK] 已写 {path.as_posix()}：{len(frame)} 个性质列，标注合计 {total}，"
            f"行数 {int(frame['n_rows'].iloc[0])}"
        )
        for name in KEY_PROPERTIES:
            row = frame.loc[frame["property"] == name].iloc[0]
            print(
                f"     {name}: 非空 {row['n_nonnull']} / 唯一结构 {row['n_unique_structures']} "
                f"/ 重复 {row['n_duplicate_rows']} 行 {row['n_duplicate_groups']} 组 "
                f"/ 其中冲突 {row['n_conflicting_groups']} 组"
            )
        print(
            f"[注意] 不要用 {FORBIDDEN_FILE}：它报 Tg {FORBIDDEN_FILE_TG:,}，"
            "那是匹配结构之前的文献记录数，比可用样本高两个数量级"
        )
        return 0

    problems = verify(args.raw, args.out)
    if problems:
        for problem in problems:
            print(f"[FAIL] {problem}", file=sys.stderr)
        return 1
    print("[OK] 清单与原始 CSV 重算结果一致（28 列非空数 / 标注合计 / 三个点名性质）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
