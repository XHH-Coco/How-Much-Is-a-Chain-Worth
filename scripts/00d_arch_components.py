"""阶段入口 00d：架构迁移的组分特征表（C-b-3 第⑤步）。
Stage entry 00d: the task4 component feature table (C-b-3 step 5).

中文：
    用途：把官方 `generated/shared_splits/ood_components.csv` 里的每个唯一组分算一次
          205 描述符 + 2048 位指纹，落成 C-c 跑 OOD 三段式时要用的三样东西：
              `data/processed/arch/component_features.parquet`（一行一个组分）
              `data/processed/arch/component_index.csv`（`psmiles → feature_row`，含 `feature_sha256`）
              `data/processed/arch/component_failures.csv`（算不出来的组分，**不静默丢**）
          并当场断言"holdout 里每个唯一 psmiles 在索引里有且只有一行"（分工_C-b 验收③）。
    插入式编号：`00c` 是 task4 取件、`00d` 是它的组分特征；不占用 README §4.2.1 里 06（C 的汇总）
          与 07（D 的画图）两格，理由同 D-43 / D-50。
    依赖：`generated/shared_splits/`（官方 `create_task4_shared_splits.py` 物化的产物，不入库）、
          `chem/descriptors.py`（A-6 的唯一计算口）与 `experiments/scaling` 的缓存/并行层。
    许何：VIPEA / polyVERSE 属 GTRC General Public Use License（D-08）；本脚本的产物是派生数据，
          不入库，只把哈希写进 `data/manifests/arch_component_manifest.json`。
    对外：命令行。`--verify` 只读，重算覆盖与逐行指纹。
    被谁用：C-c（跑 30 runs 的 OOD 三段式）、D（复核组分覆盖）。

English:
    Computes one 205-descriptor + 2048-bit row per unique official OOD component and writes the
    feature table, the `psmiles -> feature_row` index (with per-row digests) and the failure list,
    asserting that every unique holdout psmiles appears in the index exactly once (criterion 3).
    Insertion number 00c/00d: fetching and its component features; 06 and 07 stay reserved.
    Depends on the official materialisation under generated/shared_splits (not committed) and A-6's
    descriptor module. GTRC licence (D-08); the artefacts are derived data and stay uncommitted.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from chainworth.experiments import arch as A  # noqa: E402

LOGGER = logging.getLogger("chainworth.arch.components")


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """命令行参数。／ Command-line arguments."""
    parser = argparse.ArgumentParser(
        prog="00d_arch_components.py",
        description="C-b-3：task4 的组分特征表（一行一个组分）+ 索引 + 失败清单",
    )
    parser.add_argument(
        "--workers", type=int, default=0, metavar="N", help="并行进程数（0 = 核数−1，上限 8）"
    )
    parser.add_argument("--verify", action="store_true", help="只读：重核覆盖与逐行指纹")
    parser.add_argument("--dry-run", action="store_true", help="只打印计划，不算不写")
    parser.add_argument("-v", "--verbose", action="store_true")
    return parser.parse_args(argv)


def _verify() -> int:
    """只读校验：覆盖完备 + 索引里的行号都能在特征表里取到 + 逐行指纹复算一致。
    Read-only checks: complete coverage, every index row resolvable, per-row digests reproduce.
    """
    import hashlib

    import numpy as np
    import pandas as pd

    from chainworth.chem import descriptors as D
    from chainworth.chem import features as F

    problems: list[str] = []
    if not A.FEATURES_PATH.is_file() or not A.INDEX_PATH.is_file():
        LOGGER.error("缺产物：先跑一次不带 --verify 的 00d")
        return 1
    frame = pd.read_parquet(A.FEATURES_PATH)
    index = pd.read_csv(A.INDEX_PATH)
    columns = D.descriptor_columns()
    bits = F.fingerprint_columns()
    missing_columns = [name for name in [*columns, *bits] if name not in frame.columns]
    if missing_columns:
        problems.append(f"特征表缺列 {missing_columns[:4]}…")
    coverage = A.assert_coverage_complete()
    if not coverage["complete"]:
        problems.append(
            f"覆盖不完备：缺 {coverage['n_missing']} 个组分、重复 {coverage['n_duplicated']} 个"
            f"（例：{coverage['missing'][:3]}）"
        )
    matrix = frame[[*columns, *bits]].to_numpy(dtype=np.float64)
    for row in index.itertuples(index=False):
        position = int(row.feature_row)
        if position < 0 or position >= len(frame):
            problems.append(f"{row.psmiles[:20]}… 的 feature_row={position} 越界")
            continue
        digest = hashlib.sha256(np.ascontiguousarray(matrix[position]).tobytes()).hexdigest()
        if digest != str(row.feature_sha256):
            problems.append(f"{row.psmiles[:20]}… 的 feature_sha256 与特征表对不上")
    for problem in problems:
        LOGGER.error("✗ %s", problem)
    if problems:
        return 1
    LOGGER.info(
        "验收通过：%d 个组分、holdout 唯一组分 %d 个全部有且只有一行；逐行指纹 %d 条全对",
        len(frame),
        coverage["n_holdout_components"],
        len(index),
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    """跑一次 00d，返回退出码。／ Run 00d once and return the exit code."""
    args = _parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)-7s %(message)s",
        datefmt="%H:%M:%S",
    )
    if args.verify:
        return _verify()
    if args.dry_run:
        components = A.read_component_list()
        LOGGER.info(
            "[dry-run] 组分 %d 个（去重后 %d 个）→ %s",
            len(components),
            len(set(components)),
            A.FEATURES_PATH.relative_to(REPO_ROOT).as_posix(),
        )
        return 0
    try:
        summary = A.build_component_features(workers=args.workers)
    except FileNotFoundError as error:
        LOGGER.error("缺上游产物：%s", error)
        return 2
    coverage = summary["coverage"]
    if not coverage["complete"]:
        LOGGER.error(
            "覆盖不完备：缺 %d、重复 %d（明细见 %s）",
            coverage["n_missing"],
            coverage["n_duplicated"],
            A.FAILURES_PATH.relative_to(REPO_ROOT).as_posix(),
        )
        return 1
    LOGGER.info(
        "产物：%s（%d 行）；索引 %s；失败 %d",
        summary["features_path"],
        summary["n_feature_rows"],
        summary["index_path"],
        summary["n_failed"],
    )
    print(json.dumps(coverage, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
