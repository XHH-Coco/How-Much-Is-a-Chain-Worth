"""阶段入口 00b：入库数据的"入口包"（C 组真读的那 19 MB / the 19 MB team C actually reads）。
Stage entry 00b: the committed data package.

中文：
    用途：这是 `data/` 里**唯一进 git** 的那一小撮文件，给的是"C 组开工必需、缺了就卡"的东西：
          data/processed/features/<task>.npz           11 份（8 个主表任务 + 3 个 OpenPoly 性质）
          data/processed/structures/psmiles.parquet    结构层画像（wPSMILES / 重复单元）
          data/raw/polybench26/.../MD_5000/Cp/*.csv     8 份 MD_5000 Cp 的 canonical 表（scaling 的原料）
    为什么破例：新组员要自己重下 653 MB，得先撞 figshare 的 403（`SOURCES.md` 坑十）与 PMC 的
          工作量证明（坑二十五）；而 C 组真正读的只有 19 MB，且全部是 CC BY 4.0（可再分发，须署名）。
          判断标准仍是 README 硬规则 1 的那一句：**别人克隆后拿不到它，复现就卡住**。
          决策记录见 `docs/DECISIONS.md` D-38，清单见 `data/manifests/committed_data_manifest.csv`。
    依赖：无（只读盘）。配置：包的内容写在本文件 `PACKAGE` 里——**加文件就改这里再 `--write`**。
    对外：`--write` 重算清单（sha256 逐份）、`--verify` 只读复核（内容 + 是否真被 git 跟踪）、
          `--list` 只打印计划。
    被谁用：C 组（开工前 `--verify` 一次）、CI/审计（`tests/test_data_package.py`）、仓库所有者（改包）。

English:
    Purpose: the only slice of `data/` that lives in git - what team C cannot start without. Eleven
    task-level feature packs (8 main-table tasks + 3 OpenPoly properties), the structure-level table
    (wPSMILES / repeat units) and the eight canonical MD_5000 Cp tables that the scaling arm is built
    from.
    Why an exception: a teammate would otherwise re-download 653 MB through figshare's 403 wall and
    PMC's proof-of-work challenge, while team C only reads 19 MB of CC BY 4.0 material (redistributable
    with attribution). The test is still README hard rule 1: if a fresh clone cannot get it and
    reproduction stalls, it belongs here. Rationale: docs/DECISIONS.md D-38; inventory:
    data/manifests/committed_data_manifest.csv.
    Depends on: nothing (disk only). Config: the package is the `PACKAGE` list below - change it here
    and re-run `--write`.
    Public API: `--write` rewrites the manifest, `--verify` re-hashes read-only and also checks that
    every file is really tracked by git, `--list` prints the plan.
    Callers: team C (verify once before starting), the audit test (tests/test_data_package.py), the
    repository owner (when the package changes).
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import logging
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
MANIFEST_PATH = REPO_ROOT / "data" / "manifests" / "committed_data_manifest.csv"
MANIFEST_COLUMNS = ("path", "bytes", "sha256", "role", "who", "license")

LOGGER = logging.getLogger("chainworth.data_package")

# 许可：三组文件全部落在 CC BY 4.0（OPoly26 / OMersBench / PolyBench 平台）。
# 不含 GTRC 的 polyVERSE EA/IP——架构迁移那部分**不入库**，C-b 从上游自取（见 C-b 文档 §C-b-3）。
LICENSE = "CC BY 4.0 (OPoly26 / OMersBench / PolyBench)"

# 8 个主表任务（D-17）+ 3 个 OpenPoly 性质。备用队那 4 份（pb_t1_td / pb_t2_tssab / pb_t3_dlt /
# pb_t3_dmp_sm）C 组不读，留在本地：`python scripts/03b_assemble.py --tasks <id>` 可按需重建。
MAIN_TASKS = (
    "pb_t1_tg",
    "pb_t1_tm",
    "pb_t2_modulus",
    "pb_t2_permeability",
    "pb_t3_dielectric_loss",
    "pb_t3_iwlt",
    "pb26_polymetrix_tg",
    "pb26_homopolymer_density",
)
OPENPOLY_TASKS = ("op_tg", "op_tm", "op_dielectric_constant_total")

PACKAGE: list[dict[str, str]] = [
    *[
        {
            "path": f"data/processed/features/{task}.npz",
            "role": "任务级特征矩阵（每次测量一行：row_id + y + psmiles + 205 描述符 + 2048 位指纹）",
            "who": "C-a（A1/A2/A5）、C-b（OpenPoly 经典臂）",
            "license": LICENSE,
        }
        for task in (*MAIN_TASKS, *OPENPOLY_TASKS)
    ],
    {
        "path": "data/processed/structures/psmiles.parquet",
        "role": "结构层画像：18,152 个结构的 canonical PSMILES + wPSMILES + 画像（A5 链信息的来源）",
        "who": "C-a（A5）、C-b（造新任务特征时的结构表）",
        "license": LICENSE,
    },
    *[
        {
            "path": name,
            "role": "MD_5000 Cp 的 canonical PSMILES + 标签（scaling 的原料；档位文件与官方 train_sizes 对应）",
            "who": "C-b（C-b-1 造 pb26_scaling_cp 的两份特征包）",
            "license": LICENSE,
        }
        for name in (
            "alternating_Cp.csv",
            "alternating_300_Cp.csv",
            "alternating_3000_Cp.csv",
            "alternating_10000_Cp.csv",
            "random_Cp.csv",
            "homopolymer_Cp.csv",
            "homopolymer_300_Cp.csv",
            "homopolymer_1000_Cp.csv",
        )
    ],
]
CP_DIR = "data/raw/polybench26/Datasets/PSMILES/MD_5000/Cp/"


def resolve(entry: dict[str, str]) -> Path:
    """把包条目变成绝对路径：只写了文件名的那几条落在 Cp 目录下。
    Turn one entry into an absolute path; bare file names live under the Cp directory.

    数据来源：本文件的 `PACKAGE` 与 `CP_DIR`。副作用：无。
    Data source: `PACKAGE` and `CP_DIR` here. Side effects: none.
    """
    raw = entry["path"]
    return REPO_ROOT / (raw if "/" in raw else CP_DIR + raw)


def sha256_of(path: Path) -> str:
    """分块算 sha256（大文件不进内存）。/ Chunked sha256; large files never enter memory."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def repo_relative(path: Path) -> str:
    """/ 仓库相对路径，**正斜杠**（A-5 踩过反斜杠的坑，别再用 str()）。"""
    return path.resolve().relative_to(REPO_ROOT).as_posix()


def tracked_by_git(relative: str) -> bool:
    """该路径是否真的被 git 跟踪——"文件在盘上"与"文件在库里"是两件事。
    Whether git really tracks the path; "on disk" and "in the repository" are different things.

    数据来源：`git ls-files`。副作用：起一个子进程；git 不可用时返回 False 而不是抛。
    Data source: `git ls-files`. Side effect: one subprocess; returns False when git is unavailable.
    """
    try:
        done = subprocess.run(
            ["git", "ls-files", "--error-unmatch", relative],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            check=False,
        )
    except OSError:  # git 没装：调用方要能分辨"没跟踪"与"查不了"
        return False
    return done.returncode == 0


def build_rows() -> list[dict[str, str]]:
    """逐条算字节数与 sha256，缺文件当场抛错（清单不许有洞）。
    Hash every entry; a missing file raises at once - the inventory may not have holes.
    """
    rows: list[dict[str, str]] = []
    for entry in PACKAGE:
        path = resolve(entry)
        if not path.is_file():
            raise FileNotFoundError(f"入口包缺文件：{repo_relative(path) if path.exists() else path}")
        rows.append(
            {
                "path": repo_relative(path),
                "bytes": str(path.stat().st_size),
                "sha256": sha256_of(path),
                "role": entry["role"],
                "who": entry["who"],
                "license": entry["license"],
            }
        )
    return sorted(rows, key=lambda row: row["path"])


def write_manifest(rows: list[dict[str, str]]) -> Path:
    """写清单，显式 `newline=""` 让 CSV 只带 LF（Windows 下默认会写成 CRLF）。
    Write the manifest with explicit newline handling so the file stays LF on Windows.
    """
    MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    with MANIFEST_PATH.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=MANIFEST_COLUMNS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    return MANIFEST_PATH


def read_manifest() -> list[dict[str, str]]:
    """/ 读清单；不存在返回空列表（首次 `--write` 之前就是这样）。"""
    if not MANIFEST_PATH.is_file():
        return []
    with MANIFEST_PATH.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def verify() -> list[str]:
    """只读复核，返回问题列表（空 = 通过）。三条：内容对上清单、清单没有多余项、真被 git 跟踪。
    Read-only check; returns problems (empty = pass): content matches, no extras, really tracked.
    """
    problems: list[str] = []
    recorded = {row["path"]: row for row in read_manifest()}
    if not recorded:
        return [f"清单不存在或为空：{repo_relative(MANIFEST_PATH)}（先跑 --write）"]

    for row in build_rows():
        path = row["path"]
        old = recorded.pop(path, None)
        if old is None:
            problems.append(f"清单里没有这一项：{path}")
            continue
        if old["bytes"] != row["bytes"] or old["sha256"] != row["sha256"]:
            problems.append(
                f"{path} 与清单不符：清单 {old['bytes']} B / {old['sha256'][:12]}…，"
                f"实际 {row['bytes']} B / {row['sha256'][:12]}…"
            )
        if not tracked_by_git(path):
            problems.append(f"{path} 在盘上但**没有进 git**（是不是忘了 --write 后 git add？）")

    for stale in recorded:
        problems.append(f"清单里有、盘上没有（或已从 PACKAGE 移除）：{stale}")
    return problems


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="入库数据的入口包 / committed data package")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--write", action="store_true", help="重算并写清单")
    group.add_argument("--verify", action="store_true", help="只读复核（内容 + git 跟踪）")
    group.add_argument("--list", action="store_true", help="只打印包里有哪几项")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    if args.list:
        for entry in PACKAGE:
            print(f"{repo_relative(resolve(entry))}\t{entry['who']}")
        return 0

    if args.write:
        rows = build_rows()
        path = write_manifest(rows)
        total = sum(int(row["bytes"]) for row in rows)
        LOGGER.info("写入 %s：%d 项，合计 %.1f MiB", repo_relative(path), len(rows), total / 2**20)
        return 0

    problems = verify()
    if problems:
        for problem in problems:
            LOGGER.error("✗ %s", problem)
        LOGGER.error("入口包复核失败：%d 条", len(problems))
        return 1
    LOGGER.info("入口包复核通过：%d 项，逐份字节数与 sha256 与清单相同，且全部被 git 跟踪", len(PACKAGE))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
