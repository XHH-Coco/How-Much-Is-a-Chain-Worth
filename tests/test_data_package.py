"""入口包总闸：`data/` 里进了 git 的每一份文件都必须"是决定要入的那一份、且内容没变"。
Package gate: every file of `data/` that lives in git must be one we decided to ship, unchanged.

中文：
    用途：`data/` 默认全部 gitignore（README 硬规则 1），2026-11-13 起开了**唯一的例外**：
          C 组开工真读的那 20 份（D-38，清单 `data/manifests/committed_data_manifest.csv`）。
          例外一旦开了，风险就从"组员下不到数据"变成"有人顺手把 100 MB 的缓存也 add 进来"——
          而这类错误**不会报错**，只会在下一次 clone 时变成 500 MB 的仓库。所以三条闸门：
          ① **清单与盘上一致**：逐份核字节数与 sha256，改一个字节就红（`00b_data_package.py --verify` 的等价物）；
          ② **清单里的每一项真的被 git 跟踪**：文件在盘上 ≠ 文件在库里，漏了 `git add` 的入库等于没入库；
          ③ **data/ 下没有清单之外的东西被跟踪**：只允许包里的 20 份、`data/manifests/` 下的清单、
              以及各目录的 `.gitkeep`；另外任何被跟踪的文件都不许超过 20 MB（GitHub 硬限 100 MB，
              留 5 倍余量，因为一旦推上去就再也删不掉历史）。
    依赖：`data/processed/**`、`data/raw/polybench26/**`（本机有）、`git` 命令（没有就跳过 ②③）。
    配置：包的内容在 `scripts/00b_data_package.py` 的 `PACKAGE`，本文件只读它的输出。
    对外：pytest 用例。被谁用：CI、C 组开工前自检、仓库所有者改包后的回归。

English:
    Purpose: `data/` is gitignored wholesale (README hard rule 1) with exactly one exception since
    2026-11-13: the 20 files team C needs to start (D-38, inventory in
    data/manifests/committed_data_manifest.csv). Once an exception exists the risk flips from
    "teammates cannot get the data" to "someone adds the 100 MB cache by accident" - and that
    failure is silent until the next clone is half a gigabyte. Three gates:
    (1) the inventory matches the disk, byte counts and sha256 alike - one changed byte is red
        (the same check as `00b_data_package.py --verify`);
    (2) every listed file is really tracked by git - on disk is not the same as in the repository;
    (3) nothing else under `data/` is tracked: only the package, the manifests and `.gitkeep`
        files. No tracked file may exceed 20 MB either (GitHub blocks at 100 MB and history is
        forever), which leaves a 5x margin.
    Depends on: data/processed and data/raw/polybench26 (present locally), the `git` command
    (tests 2 and 3 skip when git is unavailable).
    Config: the package itself lives in `PACKAGE` in scripts/00b_data_package.py; this file only
    reads its output.
    Public API: pytest cases. Callers: CI, team C's pre-flight check, the repository owner after
    changing the package.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
MANIFEST_PATH = REPO_ROOT / "data" / "manifests" / "committed_data_manifest.csv"

# 单个被跟踪文件的上限：GitHub 100 MB 硬限留 5 倍余量。超过就说明"顺手 add 了不该 add 的东西"。
# Per-file ceiling: GitHub hard-blocks at 100 MB; 20 MB leaves a 5x margin on purpose.
MAX_TRACKED_BYTES = 20 * 1024 * 1024


def git_lines(*args: str) -> list[str]:
    """跑一条 git 命令并返回非空行；git 不可用时让调用方的用例 skip。
    Run one git command and return its non-empty lines; callers skip when git is unavailable.

    数据来源：当前工作区。副作用：起一个子进程（只读命令）。
    Data source: the working tree. Side effect: one read-only subprocess.
    """
    try:
        done = subprocess.run(
            ["git", *args], cwd=REPO_ROOT, capture_output=True, text=True, check=False
        )
    except OSError as error:  # 没装 git：调用方 skip，而不是把环境问题报成数据问题
        pytest.skip(f"没有 git，跳过：{error}")
    if done.returncode != 0:
        pytest.skip(f"git {' '.join(args)} 不可用：{done.stderr.strip()[:120]}")
    return [line for line in done.stdout.splitlines() if line.strip()]


@pytest.mark.skipif(not MANIFEST_PATH.is_file(), reason="入口包清单不存在（先跑 04 --write）")
def test_manifest_matches_disk() -> None:
    """① 清单与盘上逐份对上：字节数 + sha256。
    ① The inventory matches the disk file by file: byte count and sha256.
    """
    rows = git_lines_manifest()
    mismatched: list[str] = []
    missing: list[str] = []
    for row in rows:
        path = REPO_ROOT / row["path"]
        if not path.is_file():
            missing.append(row["path"])
            continue
        if path.stat().st_size != int(row["bytes"]):
            mismatched.append(f"{row['path']}：清单 {row['bytes']} B，实际 {path.stat().st_size} B")
            continue
        actual = _sha256(path)
        if actual != row["sha256"]:
            mismatched.append(f"{row['path']}：清单 {row['sha256'][:12]}…，实际 {actual[:12]}…")
    assert not missing, f"入口包里这几份不见了：{missing}"
    assert not mismatched, "入口包内容与清单不符（改过文件却没重新 --write？）：" + "；".join(mismatched)


def test_package_is_tracked_by_git() -> None:
    """② 清单里的每一项都真的被 git 跟踪；清单本身也在库里的。
    ② Every listed file is really tracked, and so is the inventory itself.
    """
    tracked = set(git_lines("ls-files", "data"))
    rows = git_lines_manifest() if MANIFEST_PATH.is_file() else []
    if not rows:
        pytest.skip("入口包清单不存在（先跑 04 --write）")
    untracked = [row["path"] for row in rows if row["path"] not in tracked]
    assert not untracked, f"这些文件在盘上但没进 git（忘 git add？）：{untracked}"
    assert MANIFEST_PATH.relative_to(REPO_ROOT).as_posix() in tracked, "清单自己都没入库"


def test_nothing_else_under_data_is_tracked() -> None:
    """③ data/ 下只允许：包里的文件、data/manifests/ 下的清单、各目录的 .gitkeep。
    ③ Under data/, only the package, the manifests and .gitkeep files may be tracked.
    """
    tracked = git_lines("ls-files", "data")
    rows = git_lines_manifest() if MANIFEST_PATH.is_file() else []
    allowed = {row["path"] for row in rows}
    extra: list[str] = []
    for path in tracked:
        if path in allowed:
            continue
        if path.startswith("data/manifests/") or path.endswith(".gitkeep"):
            continue
        extra.append(path)
    assert not extra, f"data/ 下混进了不该入库的文件（{len(extra)} 个）：{extra[:8]}"

    oversized = [
        f"{path}（{(REPO_ROOT / path).stat().st_size / 2**20:.1f} MiB）"
        for path in tracked
        if (REPO_ROOT / path).is_file() and (REPO_ROOT / path).stat().st_size > MAX_TRACKED_BYTES
    ]
    assert not oversized, f"这些被跟踪的文件超过 20 MiB，推不上去（GitHub 硬限 100 MB）：{oversized}"


def git_lines_manifest() -> list[dict[str, str]]:
    """/ 读清单（CSV）；空文件返回空列表。"""
    import csv

    if not MANIFEST_PATH.is_file():
        return []
    with MANIFEST_PATH.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _sha256(path: Path) -> str:
    """/ 分块 sha256；与 `00b_data_package.py` 的算法逐位相同（同一份清单两处核）。"""
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()
