"""把三个基准的**官方划分**做成只读快照，并校验它们没被谁重算过。
Turn the three benchmarks' **official splits** into read-only snapshots and prove nobody re-cut them.

中文：
    用途：A-5 的唯一实现。三件事：① PolyBench 的官方 70/30 测试集从平台接口取回，
          用 RDKit 规范化 SMILES 对上调和表的行号，落成 `splits/polybench/<task>/official_70_30.json`；
          ② PolyBench26 先把官方**不发布**的 canonical PSMILES 表按官方脚本本地物化、
          用 `manifest.json` 的 `canonical_sha256` 对上，再把 `seed_42/` 原样拷进 `splits/`；
          ③ OpenPoly：官方**只发布了规则**（8:1:1 + 值域 10 等宽箱分层），没有发布划分结果与
          随机种子，所以按规则复现 R 份、公开种子、报区间（D-24，改判 D-22 的"完全拿不到"）。
    依赖：`data/interim/harmonised/*.parquet`（A-3）、`data/raw/polybench26/**`（A-2 + A-5 追加）、
          `data/raw/openpoly/data/*.csv`（A-2）、`configs/tasks/*.yaml`（A-4，`split_ref` 就是本模块的落盘清单）。
    配置：`configs/tasks/` 的 `split_ref` / `split_protocol` / `data_ref`；常量在下方常量区。
    对外：`fetch_polybench_test_sets()`、`build_polybench_splits()`、`materialize_canonical_tables()`、
          `snapshot_polybench26()`、`build_openpoly_splits()`、`stratified_split()`、`verify_splits()`。
    被谁用：`scripts/03_splits.py`（阶段入口）、`tests/test_splits.py`（断言）。

English:
    Purpose: the single implementation of A-5. Three jobs: (1) fetch PolyBench's official 70/30
    test set from the platform API, map it onto harmonised rows via RDKit canonical SMILES and
    write `splits/polybench/<task>/official_70_30.json`; (2) materialise PolyBench26's
    **unpublished** canonical PSMILES tables with the official script, check them against
    `canonical_sha256` in the official `manifest.json`, then copy `seed_42/` verbatim into
    `splits/`; (3) OpenPoly published only the **rule** (8:1:1 over ten equal-width value bins)
    and neither the partition nor its seed, so we reproduce R splits under that rule, publish the
    seeds, and report a range (D-24, superseding D-22's "entirely unobtainable").
    Depends on: `data/interim/harmonised/*.parquet` (A-3), `data/raw/polybench26/**` (A-2 plus
    the A-5 additions), `data/raw/openpoly/data/*.csv` (A-2), `configs/tasks/*.yaml` (A-4; its
    `split_ref` is this module's to-do list).
    Config: the `split_ref` / `split_protocol` / `data_ref` keys of configs/tasks plus the
    constants below.
    Public API: `fetch_polybench_test_sets()`, `build_polybench_splits()`,
    `materialize_canonical_tables()`, `snapshot_polybench26()`, `build_openpoly_splits()`,
    `stratified_split()`, `verify_splits()`.
    Callers: `scripts/03_splits.py` (stage entry) and `tests/test_splits.py` (assertions).
"""

from __future__ import annotations

import csv
import hashlib
import json
import logging
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger

from chainworth.io.tasks import load_all_task_configs

LOGGER = logging.getLogger("chainworth.splits")
RDLogger.DisableLog("rdApp.*")

# ---- 常量：路径与口径 / Constants: paths and thresholds ----

# 仓库根：本文件在 `src/chainworth/protocol/`，往上三层就是仓库根。
# Repo root: this file lives in `src/chainworth/protocol/`, three levels down.
REPO_ROOT = Path(__file__).resolve().parents[3]
RAW_ROOT = REPO_ROOT / "data" / "raw"
INTERIM_ROOT = REPO_ROOT / "data" / "interim" / "harmonised"
MANIFEST_ROOT = REPO_ROOT / "data" / "manifests"
SPLITS_ROOT = REPO_ROOT / "splits"
CONFIGS_DIR = REPO_ROOT / "configs" / "tasks"

# PolyBench 平台的官方测试集接口。**只有这一份划分，没有 seed 维度**：实测 `&seed=0/1/2`
# 返回的字节完全相同（md5 一致），所以 `eval_unit` 里的 "×3" 只能是模型初始化种子，
# 不能是"三个划分"（D-20）。
# The platform's official test-set endpoint. There is exactly one split and no seed axis:
# `&seed=0/1/2` returned byte-identical payloads, so "x3" in `eval_unit` can only mean model
# seeds, never three different splits (D-20).
POLYBENCH_API_BASE = "http://202.98.16.11"
POLYBENCH_TEST_ENDPOINT = f"{POLYBENCH_API_BASE}/api/dataset/test-download"
POLYBENCH_TEST_DIR = RAW_ROOT / "polybench_official_test"
POLYBENCH_TEST_MANIFEST = MANIFEST_ROOT / "polybench_official_test_manifest.csv"

# 官方口径 70/30；容忍 ±0.5 个百分点（任务小时整数化会让 30% 落成 29.8% 之类）。
# Official 70/30 with a +/-0.5pp tolerance (small tasks round to e.g. 29.8%).
EXPECTED_TEST_RATIO = 0.30
TEST_RATIO_TOLERANCE = 0.005

# 官方 PolyBench26 生成脚本的位置（A-5 追加下载进 `data/raw/`，见 D-21）。
# Where the official PolyBench26 generation scripts live (fetched into data/raw by A-5, D-21).
PB26_ROOT = RAW_ROOT / "polybench26"
PB26_SCRIPTS = PB26_ROOT / "Datasets" / "Dataset_construction_scripts"
PB26_CANONICAL_ROOT = PB26_ROOT / "Datasets"

# 快照清单：一条划分产物一行，供 `verify_splits()` 与审计对照。
# Snapshot manifest: one row per split artefact, read by `verify_splits()` and by the audit.
SPLITS_MANIFEST = MANIFEST_ROOT / "splits_manifest.csv"
SPLITS_MANIFEST_COLUMNS = (
    "task_id",
    "benchmark",
    "protocol",
    "artifact",
    "sha256",
    "bytes",
    "n_train",
    "n_val",
    "n_test",
    "created_at",
)

# OpenPoly：官方**只发布了划分规则，没有发布划分结果**（D-24，改判 D-22）。
# The official rule is published; the resulting partition is not (D-24, superseding D-22).
#
# 中文：论文 METHODS → Model Training Protocols 原文三句，构成本模块的实现依据：
#   ① "the dataset was randomly split into training, validation, and test sets at an 8:1:1 ratio."
#   ② "for each property, its numerical value range was divided into 10 equal-width bins, and
#       samples were proportionally drawn from each bin to construct the training, validation,
#       and test subsets."
#   ③ "All models were trained, tuned, and tested on the same data partitions."
# 缺口：全文没有 seed / random_state，所以**任何复现都是同规则的一次新抽样**，不可能与作者那份
# 逐位相同。我们因此产出 R 份规则合法的划分，用它报区间而不是报单点。
# English: the three sentences above are the implementation basis. The gap is the missing seed:
# any reproduction is a fresh draw under the same rule, never the authors' exact partition.
# We therefore emit R rule-conforming splits and report a range instead of a point.
OPENPOLY_ROOT = RAW_ROOT / "openpoly"
OPENPOLY_DATA = OPENPOLY_ROOT / "data" / "final_polymer_properties_fromliterature.csv"
OPENPOLY_SPLIT_ROOT = SPLITS_ROOT / "openpoly"
OPENPOLY_PROTOCOL = "replicated_8_1_1_stratified10"
OPENPOLY_PROTOCOL_SOURCE = (
    "Wang, J.-F. et al. OpenPoly. Chin. J. Polym. Sci. 2025, 43, 1749-1760, "
    "DOI 10.1007/s10118-025-3402-y, METHODS -> Model Training Protocols"
)
OPENPOLY_RATIOS = {"train": 0.80, "val": 0.10, "test": 0.10}
OPENPOLY_BINS = 10
# 10 个种子：0–4 是预注册的 5 折（对应 `eval_unit` 的 "x 5 folds"），5–9 是留作敏感性分析的备用。
# Ten seeds: 0-4 are the pre-registered five folds (the "x 5 folds" in `eval_unit`); 5-9 are
# reserved for the split-sensitivity analysis.
OPENPOLY_SEEDS = (0, 1, 2, 3, 4, 5, 6, 7, 8, 9)
OPENPOLY_PRIMARY_FOLDS = (0, 1, 2, 3, 4)
# 整数化之后总比例会偏 1–2 个百分点（168 条时为 10.1%），故留 2 pp 余量。
# Integer rounding moves the total shares by 1-2pp (10.1% at n=168), hence the 2pp tolerance.
OPENPOLY_RATIO_TOLERANCE = 0.02
# 每个值域箱内部的 train/val/test 份额允许偏离 0.8/0.1/0.1 的上限（小箱本来就会全部落进 train）。
# Allowed per-bin deviation from 0.8/0.1/0.1 (tiny bins legitimately fall entirely into train).
OPENPOLY_BIN_TOLERANCE = 0.25
OPENPOLY_PROTOCOL_NOTE = OPENPOLY_SPLIT_ROOT / "PROTOCOL.json"
OPENPOLY_MANIFEST = MANIFEST_ROOT / "openpoly_split_manifest.csv"
OPENPOLY_MANIFEST_COLUMNS = (
    "task_id",
    "protocol",
    "seed",
    "fold_role",
    "artifact",
    "sha256",
    "bytes",
    "n_total",
    "n_train",
    "n_val",
    "n_test",
    "n_psmiles_shared",
    "created_at",
)


class SplitError(RuntimeError):
    """划分快照不成立（来源对不上、比例不对、结构重复、官方索引与本地表不一致）。
    A split snapshot is invalid (source mismatch, wrong ratio, duplicate structure, index
    disagreement with the local table).

    中文：
        用途：让"划分这件事错了"与"网络不通"在日志里能分辨——前者必须停下，后者可以重试。
        输入：message —— 人读说明。
        输出：异常对象。副作用：无。异常：无。
        被谁用：本模块与 `verify_splits()` 的调用方。

    English:
        Purpose: separate "the split is wrong" (stop) from "the network is down" (retry).
        Args: message - human-readable description.
        Returns: the exception. Side effects: none. Raises: none.
        Callers: this module and whoever calls `verify_splits()`.
    """


# ---- 1. 摘要与网络 / Digests and HTTP ----


def sha256_of(path: Path) -> str:
    """算文件的 sha256（按字节，不做任何归一）。
    sha256 of a file, byte for byte, no normalisation.

    中文：
        用途：`splits/` 是冻结物，需要一个"改一个字节就变"的指纹。
        输入：path —— 文件路径。
        输出：64 位十六进制字符串。
        数据来源：磁盘上的字节。
        配置来源：无。
        副作用：只读。
        异常：OSError —— 文件不存在或无读权限。

    English:
        Purpose: `splits/` is frozen; it needs a fingerprint that changes on any edit.
        Args: path - file path. Returns: 64-char hex digest.
        Data source: bytes on disk. Config: none. Side effects: read-only.
        Raises: OSError when the file is missing or unreadable.
    """
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def lf_normalized_sha256(path: Path) -> str:
    """先 CRLF→LF 再算 sha256，用于跨平台比对官方写下的 `canonical_sha256`。
    Normalise CRLF to LF before hashing, so the official `canonical_sha256` can be matched.

    中文：
        用途：官方脚本在 Windows 上跑时，pandas 的 `to_csv` 默认用 `os.linesep`，落盘是 CRLF；
              官方 `canonical_sha256` 是在 LF 下算的。只差行尾、内容逐字节相同——归一后才相等
              （D-21）。这不是"放宽标准"，是把行尾这个与内容无关的传输细节剥掉。
        输入：path —— 文件路径。
        输出：64 位十六进制字符串（等价于该文件在 Linux 上落盘的 sha256）。
        数据来源：磁盘上的字节。配置来源：无。副作用：只读。
        异常：OSError —— 文件不存在。

    English:
        Purpose: on Windows, pandas' `to_csv` writes CRLF via `os.linesep`, while the official
                 `canonical_sha256` was computed on LF. Content is byte-identical, only the line
                 ending differs, so normalise before comparing (D-21).
        Args: path - file path. Returns: hex digest equal to the Linux-written file's sha256.
        Data source: bytes on disk. Config: none. Side effects: read-only.
        Raises: OSError when the file is missing.
    """
    return hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def _http_get(url: str, *, retries: int = 3, timeout: int = 60) -> bytes:
    """取一个 URL 的字节，失败按 2/4/8 秒退避重试。
    Fetch a URL's bytes, retrying with 2/4/8 s backoff.

    中文：
        用途：A-2 的经验是这台机器上 TLS/代理会偶发断连（`00_download.py` 三条路径轮换），
              而官方测试集单份只有几百 KB，重试三次足够，不必引入依赖。
        输入：url —— 完整地址；retries —— 重试次数；timeout —— 单次超时秒数。
        输出：响应体原始字节（不解析、不改行尾）。
        数据来源：`POLYBENCH_TEST_ENDPOINT` 等官方地址。
        配置来源：调用方传入的 url。
        副作用：出网。
        异常：SplitError —— 重试用尽仍失败（消息里带最后一次的真实原因）。

    English:
        Purpose: TLS/proxy hiccups are known on this host; each test set is a few hundred KB, so
                 three retries suffice and no extra dependency is needed.
        Args: url - full address; retries; timeout in seconds.
        Returns: raw response body (unparsed, line endings untouched).
        Data source: official endpoints.
        Config: the caller's url. Side effects: network egress.
        Raises: SplitError after the last retry, carrying the real cause.
    """
    last: Exception | None = None
    for attempt in range(retries + 1):
        try:
            request = urllib.request.Request(
                url, headers={"User-Agent": "chainworth-splits/0.1 (+A-5 snapshot)"}
            )
            with urllib.request.urlopen(request, timeout=timeout) as response:
                if response.status != 200:
                    raise SplitError(f"{url} 返回 {response.status}")
                return response.read()
        except (urllib.error.URLError, TimeoutError, ConnectionError, SplitError) as exc:
            last = exc
            if attempt < retries:
                delay = 2 ** (attempt + 1)
                LOGGER.warning("第 %d/%d 轮重试（等 %ds）：%s", attempt + 1, retries, delay, url)
                time.sleep(delay)
    raise SplitError(f"下载失败：{url}（最后一次：{type(last).__name__}: {last}）")


def _now_iso() -> str:
    """取带时区的本地时间戳，写法与 A-2 清单的 `downloaded_at` 保持一致。
    Return a timezone-aware local ISO timestamp, matching A-2's `downloaded_at` format.

    中文：
        用途：每一份快照都要回答"这是哪一分钟落下来的"。
        输入：无。输出：ISO 8601 字符串（含 UTC 偏移）。
        数据来源：系统时钟。配置来源：无。副作用：无。异常：无。

    English:
        Purpose: every snapshot must answer "when exactly was this taken".
        Args: none. Returns: ISO 8601 string with offset.
        Data source: system clock. Config: none. Side effects: none. Raises: none.
    """
    # `datetime.isoformat()` 给的是 `+08:00`，与 A-2 清单里的写法一致；`time.strftime("%z")`
    # 会写成 `+0800`，两种写法混在同一份仓库里会让"时间戳能否直接比较"变成一次人工判断。
    # `datetime.isoformat()` yields `+08:00`, matching the A-2 manifests; `%z` would yield
    # `+0800`, and mixing the two formats makes timestamp comparison a manual judgement call.
    return datetime.now().astimezone().isoformat(timespec="seconds")

# ---- 2. PolyBench：官方测试集 -> 行号 / PolyBench: official test set -> row indices ----


def canonical_smiles(smiles: str) -> str | None:
    """把 SMILES 规范化成 RDKit 的 canonical 形式，失败返回 None（并记录一次）。
    Canonicalise a SMILES with RDKit; return None on failure.

    中文：
        用途：平台给的测试集与 A-3 的调和表用的不是同一套写法（平台是发布方原始串，
              A-3 存的是 `smiles_canonical`）。不先规范化，同一根链会被当成两行。
        输入：smiles —— 原始 SMILES 字符串。
        输出：canonical SMILES，或 None（RDKit 解析失败）。
        数据来源：调用方传入的字符串。
        配置来源：无。
        副作用：无（RDKit 的日志已在模块级关掉，避免 2,417 条里刷满 warning）。
        异常：无。

    English:
        Purpose: the platform ships raw SMILES while A-3 stores `smiles_canonical`; without
                 canonicalisation the same chain looks like two different rows.
        Args: smiles - raw SMILES. Returns: canonical SMILES, or None when RDKit fails.
        Data source: the caller's string. Config: none.
        Side effects: none (RDKit logging is silenced module-wide).
        Raises: none.
    """
    if not isinstance(smiles, str) or not smiles.strip():
        return None
    molecule = Chem.MolFromSmiles(smiles)
    if molecule is None:
        return None
    return Chem.MolToSmiles(molecule)


def fetch_polybench_test_sets(
    configs: Mapping[str, Mapping[str, Any]],
    *,
    limit: int | None = None,
    force: bool = False,
    dry_run: bool = False,
) -> list[dict[str, Any]]:
    """从平台接口取回每个 PolyBench 任务的官方测试集，落盘并写清单。
    Fetch each PolyBench task's official test set, write it to disk and record a manifest.

    中文：
        用途：官方 70/30 的"30"只有这一个来源——平台前端那个 *Download Test Set* 按钮背后的
              接口。它只给 `polymer_name,SMILES` 两列（不给标签），但这正是我们要的：
              **哪一部分行是测试行**由官方说了算，我们只负责把 SMILES 对回行号。
        输入：configs —— task_id -> 配置（只处理 `benchmark == "polybench"` 且 `status == "available"`）；
              limit —— 最多取几个任务（调试用，清单会写 `.partial`）；force —— 已存在也重下；
              dry_run —— 只打印计划。
        输出：清单行列表，每行为 dict（列见 `POLYBENCH_TEST_MANIFEST`）。
        数据来源：`POLYBENCH_TEST_ENDPOINT`，`file_name` 取自配置的 `dataset_name` + `.csv`。
        配置来源：`configs/tasks/*.yaml` 的 `dataset_name`；上游是 figshare 的 39 个文件名。
        副作用：写 `data/raw/polybench_official_test/*.csv` 与清单 CSV。
        异常：SplitError —— 接口返回空、返回的不是两列、或行数为 0。

    English:
        Purpose: the "30" of 70/30 has exactly one source - the endpoint behind the platform's
                 *Download Test Set* button. It returns only `polymer_name,SMILES` (no labels),
                 which is exactly what is needed: the official side decides which rows are test
                 rows, we only map SMILES back to row indices.
        Args: configs - task_id -> config (only `benchmark == "polybench"` and available);
              limit - max tasks, for debugging (writes a `.partial` manifest); force - refetch;
              dry_run - plan only.
        Returns: manifest rows (columns as in `POLYBENCH_TEST_MANIFEST`).
        Data source: `POLYBENCH_TEST_ENDPOINT`, with `file_name` from the config's `dataset_name`.
        Config: `dataset_name` of configs/tasks/*.yaml, upstream the figshare file names.
        Side effects: writes the CSVs under data/raw and the manifest CSV.
        Raises: SplitError on empty payload, unexpected columns, or zero rows.
    """
    targets = [
        (task_id, config)
        for task_id, config in sorted(configs.items())
        if config["benchmark"] == "polybench" and config["status"] == "available"
    ]
    skipped = sorted(
        task_id
        for task_id, config in configs.items()
        if config["benchmark"] == "polybench" and config["status"] != "available"
    )
    if skipped:
        LOGGER.info("跳过不可用任务（无标签可取划分）/ skipping unavailable tasks: %s", skipped)
    if limit is not None:
        targets = targets[:limit]

    manifest_path = (
        POLYBENCH_TEST_MANIFEST
        if limit is None
        else POLYBENCH_TEST_MANIFEST.with_name(POLYBENCH_TEST_MANIFEST.stem + ".partial.csv")
    )
    previous = _read_manifest(manifest_path)

    rows: list[dict[str, Any]] = []
    for task_id, config in targets:
        file_name = f"{config['dataset_name']}.csv"
        url = f"{POLYBENCH_TEST_ENDPOINT}?file_name={urllib.parse.quote(file_name)}"
        target = POLYBENCH_TEST_DIR / file_name
        if dry_run:
            LOGGER.info("  dry-run：将取 %s -> %s / would fetch", url, target)
            continue
        if target.is_file() and not force:
            payload = target.read_bytes()
            LOGGER.info("  已在本机，直接核 / already local: %s", target.name)
        else:
            payload = _http_get(url)
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(payload)
            LOGGER.info("  取回 %s（%d B）/ fetched", file_name, len(payload))

        frame = pd.read_csv(target, encoding="utf-8-sig")
        if list(frame.columns) != ["polymer_name", "SMILES"]:
            raise SplitError(f"{file_name} 的列不是 polymer_name,SMILES：{list(frame.columns)}")
        if frame.empty:
            raise SplitError(f"{file_name} 的官方测试集为空")

        digest = hashlib.sha256(payload).hexdigest()
        row = previous.get(task_id, {})
        rows.append(
            {
                "task_id": task_id,
                "dataset_name": config["dataset_name"],
                "n_rows": len(frame),
                "n_total": config["n_labels"],
                "test_ratio": round(len(frame) / config["n_labels"], 6),
                "sha256": digest,
                # 若服务器哪天换了行尾，raw 摘要会变而内容不变；两列都记，便于分辨。
                # If the server ever changes line endings the raw digest moves but content does
                # not; keeping both columns makes that distinguishable.
                "sha256_lf": hashlib.sha256(payload.replace(b"\r\n", b"\n")).hexdigest(),
                "source_url": url,
                "fetched_at": (
                    row["fetched_at"] if row.get("sha256") == digest and row.get("fetched_at") else _now_iso()
                ),
                "bytes": len(payload),
            }
        )

    if not dry_run:
        _write_manifest(
            rows,
            manifest_path,
            (
                "task_id",
                "dataset_name",
                "n_rows",
                "n_total",
                "test_ratio",
                "sha256",
                "sha256_lf",
                "source_url",
                "fetched_at",
                "bytes",
            ),
        )
        LOGGER.info("已写清单 / manifest written: %s（%d 行）", manifest_path, len(rows))
    return rows


def build_polybench_split(config: Mapping[str, Any], test_csv: Path) -> dict[str, Any]:
    """把官方测试集对到调和表行号，返回一份可落盘的划分对象。
    Map the official test set onto harmonised row indices and return a serialisable split.

    中文：
        用途：这是整条 A-5 里最容易错的一步。索引存的是**行号**，行号只在 A-3 那张表的
              特定顺序下指向正确分子；所以这里三条都要成立才继续：① 调和表的 `source_row`
              单调；② 每个测试 SMILES 都能规范化且**唯一**命中一行；③ 命中数等于测试集行数。
              任何一条不成立就抛异常，不写文件——半份划分比没有划分危险。
        输入：config —— 一个 PolyBench 任务配置；test_csv —— 官方测试集 CSV 路径。
        输出：dict（schema 见模块级说明），含 `test_index` / `train_index` 两个行号列表。
        数据来源：`config["data_ref"]` 的 parquet；`test_csv` 的官方字节。
        配置来源：`config["n_labels"]`、`config["task_id"]`、`config["split_protocol"]`。
        副作用：只读。
        异常：SplitError —— 行序不单调、SMILES 解析失败、命中 0 行或多行、比例越界。

    English:
        Purpose: the easiest step in A-5 to get silently wrong. The indices are **row numbers**
                 and only mean the right molecules in A-3's exact row order, so three things
                 must hold first: (1) `source_row` is monotonic; (2) every test SMILES
                 canonicalises and hits exactly one row; (3) the hit count equals the official
                 test-set size. Any failure aborts without writing - a half split is worse than
                 none.
        Args: config - one PolyBench task config; test_csv - the official test-set CSV.
        Returns: dict (schema described at module level) with `test_index` / `train_index`.
        Data source: the parquet at `config["data_ref"]` and the official CSV bytes.
        Config: `n_labels`, `task_id`, `split_protocol` of the config.
        Side effects: read-only.
        Raises: SplitError on non-monotonic rows, unparsable SMILES, zero/ambiguous matches,
                or a ratio outside tolerance.
    """
    task_id = str(config["task_id"])
    table = pd.read_parquet(REPO_ROOT / str(config["data_ref"]))
    if "source_row" not in table.columns or "smiles_canonical" not in table.columns:
        raise SplitError(f"{task_id} 的调和表缺少 source_row / smiles_canonical 列")
    if not table["source_row"].is_monotonic_increasing:
        raise SplitError(f"{task_id} 的调和表不是按 source_row 升序，行号不可信")
    if len(table) != int(config["n_labels"]):
        raise SplitError(f"{task_id} 调和表 {len(table)} 行，配置写 {config['n_labels']} 行")

    official = pd.read_csv(test_csv, encoding="utf-8-sig")
    canonical = [canonical_smiles(value) for value in official["SMILES"]]
    unparsable = [str(official["SMILES"].iloc[i]) for i, value in enumerate(canonical) if value is None]
    if unparsable:
        raise SplitError(
            f"{task_id}：官方测试集里有 {len(unparsable)} 个 SMILES 解析失败，例：{unparsable[:3]}"
        )

    positions: dict[str, list[int]] = {}
    for position, smiles in enumerate(table["smiles_canonical"].tolist()):
        positions.setdefault(smiles, []).append(position)

    missing = [value for value in canonical if value not in positions]
    if missing:
        raise SplitError(
            f"{task_id}：官方测试集有 {len(missing)} 个结构在调和表里找不到，例：{missing[:3]}"
        )
    ambiguous = [value for value in dict.fromkeys(canonical) if len(positions[value]) > 1]
    if ambiguous:
        raise SplitError(
            f"{task_id}：{len(ambiguous)} 个结构在调和表里出现多行，行号会有歧义，例：{ambiguous[:3]}"
        )

    test_index = sorted(positions[value][0] for value in canonical)
    test_set = set(test_index)
    train_index = [position for position in range(len(table)) if position not in test_set]

    ratio = len(test_index) / len(table)
    if abs(ratio - EXPECTED_TEST_RATIO) > TEST_RATIO_TOLERANCE:
        raise SplitError(
            f"{task_id}：官方测试集占 {ratio:.6f}，超出 {EXPECTED_TEST_RATIO}±{TEST_RATIO_TOLERANCE}"
            f"（{len(test_index)}/{len(table)}）"
        )
    if len(set(test_index)) != len(test_index) or len(test_index) + len(train_index) != len(table):
        raise SplitError(f"{task_id}：划分不互斥或没覆盖全量")

    payload = test_csv.read_bytes()
    return {
        "schema_version": 1,
        "task_id": task_id,
        "benchmark": "polybench",
        "protocol": str(config["split_protocol"]),
        "n_total": int(len(table)),
        "n_train": len(train_index),
        "n_test": len(test_index),
        "test_ratio": round(ratio, 6),
        "row_order": f"{config['data_ref']}#row_position（按 source_row 升序）",
        "match_rule": "rdkit_canonical_smiles（唯一命中；不按名字、不按顺序猜）",
        "source": {
            "url": f"{POLYBENCH_TEST_ENDPOINT}?file_name={urllib.parse.quote(test_csv.name)}",
            "sha256": hashlib.sha256(payload).hexdigest(),
            "bytes": len(payload),
            "n_rows": int(len(official)),
            "fetched_at": _now_iso(),
            "note": "官方只发布测试集的 SMILES（不含标签）；训练集是它在全量里的补集，本文件不重切",
        },
        "test_index": test_index,
        "train_index": train_index,
    }


def build_polybench_splits(
    configs: Mapping[str, Mapping[str, Any]],
    *,
    limit: int | None = None,
    dry_run: bool = False,
) -> list[str]:
    """为每个可用的 PolyBench 任务生成 `splits/polybench/<task>/official_70_30.json`。
    Write `splits/polybench/<task>/official_70_30.json` for every available PolyBench task.

    中文：
        用途：把 `fetch_polybench_test_sets()` 落下的 CSV 逐份翻成行号 JSON，并顺手把
              每一份的 sha256 记进 `splits_manifest.csv`。
        输入：configs —— task_id -> 配置；limit —— 最多几个任务；dry_run —— 只打印。
        输出：写成功的 task_id 列表。
        数据来源：`data/raw/polybench_official_test/<数据集名>.csv`、调和表 parquet。
        配置来源：`config["split_ref"]`（落盘路径）。
        副作用：写 `splits/**` 与 `data/manifests/splits_manifest.csv` 的行。
        异常：SplitError —— 见 `build_polybench_split()`；FileNotFoundError —— 测试集没取回。

    English:
        Purpose: turn the CSVs from `fetch_polybench_test_sets()` into row-index JSONs and record
                 each artefact's sha256 in `splits_manifest.csv`.
        Args: configs; limit - max tasks; dry_run - print only.
        Returns: the task_ids written.
        Data source: the test CSVs under data/raw plus the harmonised parquet.
        Config: `split_ref` decides the destination path.
        Side effects: writes under `splits/` and appends rows to the snapshot manifest.
        Raises: SplitError from `build_polybench_split()`; FileNotFoundError when a test CSV is
                missing.
    """
    done: list[str] = []
    targets = [
        (task_id, config)
        for task_id, config in sorted(configs.items())
        if config["benchmark"] == "polybench" and config["status"] == "available"
    ]
    if limit is not None:
        targets = targets[:limit]

    rows: list[dict[str, Any]] = []
    for task_id, config in targets:
        test_csv = POLYBENCH_TEST_DIR / f"{config['dataset_name']}.csv"
        if dry_run:
            LOGGER.info("  dry-run：将由 %s 生成 %s / would build", test_csv.name, config["split_ref"])
            continue
        if not test_csv.is_file():
            raise FileNotFoundError(
                f"{task_id} 的官方测试集不在：{test_csv}；先跑 `--source polybench` 取回"
            )
        split = build_polybench_split(config, test_csv)
        destination = REPO_ROOT / str(config["split_ref"])
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(
            json.dumps(split, ensure_ascii=False, indent=1) + "\n", encoding="utf-8", newline="\n"
        )
        rows.append(
            {
                "task_id": task_id,
                "benchmark": "polybench",
                "protocol": split["protocol"],
                "artifact": str(config["split_ref"]),
                "sha256": sha256_of(destination),
                "bytes": destination.stat().st_size,
                "n_train": split["n_train"],
                "n_test": split["n_test"],
                "created_at": _now_iso(),
            }
        )
        LOGGER.info("  %s：train %d / test %d（%.2f%%）/ written", task_id, split["n_train"], split["n_test"], 100 * split["test_ratio"])
        done.append(task_id)
    if rows:
        _append_manifest_rows(rows)
    return done


# ---- 3. 清单读写 / Manifest helpers ----


def _read_manifest_rows(path: Path) -> list[dict[str, str]]:
    """把一份清单读成"行列表"；文件不存在或读坏就当空。
    Read a manifest as a list of rows; missing or unreadable reads as empty.

    中文：
        用途：`_read_manifest()` 按 `task_id` 建键，一条任务一份产物时够用；但 OpenPoly 一条任务
              有 R 份产物（每种子一份），按 task_id 建键会把前面的行覆盖掉，清单静默缩水。
              凡"一条任务多行"的清单必须走这个函数。
        输入：path —— 清单路径。
        输出：行字典列表（原样字符串字段）。数据来源：磁盘上的 CSV。
        配置来源：无。副作用：只读。异常：无。

    English:
        Purpose: `_read_manifest()` keys by `task_id`, which is fine while a task has exactly one
                 artefact. OpenPoly has R artefacts per task (one per seed), so keying by task_id
                 silently drops rows. Any multi-row-per-task manifest must go through here.
        Args: path. Returns: list of raw string-field dicts. Data source: the CSV on disk.
        Raises: none.
    """
    if not path.is_file():
        return []
    try:
        with path.open("r", encoding="utf-8", newline="") as handle:
            return [dict(row) for row in csv.DictReader(handle) if row.get("task_id")]
    except (OSError, csv.Error):
        return []


def _read_manifest(path: Path) -> dict[str, dict[str, str]]:
    """把一份清单读成 `task_id -> 行` 的字典；文件不存在就当空。
    Read a manifest into a `task_id -> row` dict; a missing file reads as empty.

    中文：
        用途：重跑时判断"这一份上次是不是同一份字节"，从而沿用旧的 `fetched_at`——每次重跑都
              刷新时间戳会让冻结物在 git 里显得被人动过，而实际上没变。
              **只适用于一条任务一行**的清单；一条任务多行请用 `_read_manifest_rows()`。
        输入：path —— 清单路径。
        输出：dict；键为 task_id，值为该行原样字符串字段。
        数据来源：磁盘上的 CSV。配置来源：无。副作用：只读。
        异常：无（读坏了按空处理，宁可变一次时间戳也不中断）。

    English:
        Purpose: decide whether a re-run produced the same bytes so the old `fetched_at` can be
                 kept; refreshing timestamps every run makes frozen files look hand-edited.
                 **One row per task only**; use `_read_manifest_rows()` otherwise.
        Args: path. Returns: dict of raw string fields keyed by task_id.
        Data source: the CSV on disk. Config: none. Side effects: read-only.
        Raises: none (an unreadable manifest reads as empty).
    """
    if not path.is_file():
        return {}
    try:
        with path.open("r", encoding="utf-8", newline="") as handle:
            return {row["task_id"]: row for row in csv.DictReader(handle) if row.get("task_id")}
    except (OSError, KeyError, csv.Error):
        return {}


def _write_manifest(rows: Sequence[Mapping[str, Any]], path: Path, columns: Sequence[str]) -> None:
    """按固定列序写一份清单（覆盖同名文件）。
    Write a manifest with a fixed column order, overwriting the file.

    中文：
        用途：清单只能由脚本重生成，不许手改（分工_A §7 红线 5）；列序固定，别人才能逐列对。
        输入：rows —— 本次的行；path —— 目标；columns —— 列序。
        输出：无。数据来源：调用方给的行。配置来源：无。
        副作用：覆盖 path，必要时建父目录。
        异常：OSError —— 无写权限。

    English:
        Purpose: manifests are regenerated by scripts only (red line 5), and a fixed column
                 order is what makes them comparable.
        Args: rows; path; columns. Returns: none.
        Data source: the caller's rows. Config: none.
        Side effects: overwrites path, creating parents. Raises: OSError without write access.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        # `lineterminator="\n"`：与 `.gitattributes` 的 `*.csv text eol=lf` 一致，
        # 否则每次本机重跑都会把文件改回 CRLF，git 里看着像被人动过。
        # Matches `*.csv text eol=lf`; otherwise every local re-run rewrites CRLF and looks
        # like a hand edit in git.
        writer = csv.DictWriter(handle, fieldnames=list(columns), lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column, "") for column in columns})


def _append_manifest_rows(rows: Sequence[Mapping[str, Any]]) -> None:
    """把若干行并进 `splits_manifest.csv`：同 task_id + artifact 的旧行被替换，其余保留。
    Merge rows into `splits_manifest.csv`, replacing rows with the same task_id + artifact.

    中文：
        用途：A-5 会分别生成 PolyBench（10 份）、PolyBench26（4 份划分 × 5 折）两类产物，
              分两次跑；清单要能累积，而不是第二次把第一次抹掉。
        输入：rows —— 新行（列见 `SPLITS_MANIFEST_COLUMNS`）。
        输出：无。
        数据来源：现有清单 + 本次行。
        配置来源：无。
        副作用：覆盖 `data/manifests/splits_manifest.csv`。
        异常：OSError —— 无写权限。

    English:
        Purpose: A-5 writes PolyBench and PolyBench26 artefacts in separate runs; the manifest
                 must accumulate instead of the second run wiping the first.
        Args: rows - new rows. Returns: none.
        Data source: the existing manifest plus this run's rows.
        Config: none. Side effects: overwrites the snapshot manifest.
        Raises: OSError without write access.
    """
    merged: dict[tuple[str, str], dict[str, Any]] = {}
    dropped: list[str] = []
    for row in _read_manifest_rows(SPLITS_MANIFEST):
        # 只登记盘上真有的东西：某次试验性运行写下的路径，在目录改了形状之后会变成幽灵行，
        # 留着它等于让清单描述一个不存在的产物。
        # Only record what is actually on disk: a path written by an exploratory run becomes a
        # ghost row once the layout changes, and a ghost row describes an artefact that is gone.
        if not (REPO_ROOT / str(row["artifact"])).exists():
            dropped.append(str(row["artifact"]))
            continue
        merged[(row["task_id"], row["artifact"])] = dict(row)
    if dropped:
        LOGGER.warning("清单里删掉 %d 行已不存在的产物 / pruned %d stale rows: %s", len(dropped), len(dropped), dropped)
    for row in rows:
        merged[(str(row["task_id"]), str(row["artifact"]))] = dict(row)
    ordered = [merged[key] for key in sorted(merged)]
    _write_manifest(ordered, SPLITS_MANIFEST, SPLITS_MANIFEST_COLUMNS)


# ---- 4. PolyBench26：canonical 表物化 + 索引原样快照 ----


def _as_seed_dir(path: Path) -> Path:
    """把"官方索引目录"统一成 `.../seed_42`：配置可能已经带了这一层，也可能没带。
    Normalise an official index directory to `.../seed_42`: the config may or may not include it.

    中文：
        用途：`official_indices_path` 的写法在不同配置里不完全一致（有的写到 `seed_42`，
              有的只写到实验目录）。少写或多写一层都会让"找不到目录"变成一次手工排查，所以
              在这里一次收敛。
        输入：path —— 配置给出的目录。
        输出：若最后一级已经是 `seed_42` 就原样返回，否则补上一层。
        数据来源：`configs/tasks/pb26_*.yaml` 的 `datasets[].official_indices_path`。
        配置来源：同上。副作用：无。异常：无。

    English:
        Purpose: `official_indices_path` is not written identically everywhere (some include
                 `seed_42`, some stop at the experiment directory). Converge that here instead of
                 turning a directory miss into manual debugging.
        Args: path - the configured directory. Returns: the same path, or with `seed_42` appended.
        Data source: `datasets[].official_indices_path` of configs/tasks/pb26_*.yaml.
        Config: same. Side effects: none. Raises: none.
    """
    return path if path.name == "seed_42" else path / "seed_42"


def _official_manifest_for(config: Mapping[str, Any], dataset: Mapping[str, Any] | None = None) -> Path:
    """给出一份官方 `manifest.json` 的本地路径（多数据集任务按 `dataset_id` 找）。
    Return the local path of the official `manifest.json` (per dataset for multi-dataset tasks).

    中文：
        用途：`canonical_sha256`、`train_sizes`、折数这些权威数字都在官方 manifest 里，
              本模块从不自己算它们，只读。
        输入：config —— 任务配置；dataset —— 机制实验的某个数据集条目（可空）。
        输出：Path，指向 `data/raw/polybench26/Datasets/scaling_indices/...` 下的 manifest。
        数据来源：配置的 `datasets[].official_indices_path` 或 `split_ref`。
        配置来源：`configs/tasks/pb26_*.yaml`。
        副作用：无。异常：SplitError —— 配置里没有可用的索引路径。

    English:
        Purpose: `canonical_sha256`, `train_sizes` and fold counts live in the official manifest;
                 this module reads them and never recomputes them.
        Args: config; dataset - one entry of a mechanism task's `datasets` (optional).
        Returns: path to the official manifest under data/raw.
        Data source: `datasets[].official_indices_path` or `split_ref`.
        Config: configs/tasks/pb26_*.yaml. Side effects: none.
        Raises: SplitError when the config carries no usable index path.
    """
    if dataset is not None:
        relative = dataset.get("official_indices_path")
        if not relative:
            raise SplitError(f"{config['task_id']}/{dataset.get('dataset_id')} 缺 official_indices_path")
        return PB26_ROOT / str(relative) / "manifest.json"
    split_ref = Path(str(config["split_ref"]))
    if split_ref.suffix == ".json":
        return PB26_ROOT / "Datasets" / split_ref.relative_to("splits/polybench26").parent / "seed_42" / "manifest.json"
    raise SplitError(f"{config['task_id']} 的 split_ref 不是可解析的 manifest 路径：{split_ref}")


def materialize_canonical_tables(
    *,
    dry_run: bool = False,
    force: bool = False,
    expected: Mapping[str, str] | None = None,
    timeout: int = 1800,
) -> dict[str, str]:
    """按官方脚本把 canonical PSMILES 表生成到 `data/raw/polybench26/Datasets/PSMILES/`。
    Generate the canonical PSMILES tables with the official script into data/raw.

    中文：
        用途：官方**不发布** canonical 表（只发布 compact 源 + 生成脚本，SOURCES.md【坑六】），
              而官方索引存的是行号——行号只在这张表的特定顺序下才指向正确分子。所以这一步
              不是"顺手重算一份"，而是**把官方脚本在本地重放一遍**，再用它自己写下的
              `canonical_sha256` 验收。验收不过就停，后面的划分一律不写。
        输入：dry_run —— 只打印将执行什么；force —— 已在位且哈希全对也重生成；
              expected —— 官方 `canonical_source -> sha256`（给了就用来判断能否跳过）；
              timeout —— 子进程超时秒数（官方脚本约 2.5 分钟）。
        输出：dict，官方 `canonical_source` -> 实际 sha256（已做 CRLF→LF 归一）。
        数据来源：`data/raw/polybench26/Datasets/Dataset_construction_scripts/files/` 下的
                  compact 源（A-2 已下）；脚本本体来自同一仓库的固定 commit（A-5 追加）。
        配置来源：无（路径即官方目录结构）。
        副作用：写 `data/raw/polybench26/Datasets/{PSMILES,wPSMILES}/**`；**不改**已发布的
                  PolyMetriX 文件（官方脚本不碰那个目录）。
        异常：SplitError —— 脚本缺失、子进程失败、或生成的表与 `canonical_sha256` 不等。

    English:
        Purpose: the canonical tables are **not published** (only compact sources plus the
                 generation script), yet the official indices are row numbers that only mean the
                 right molecules in that table's order. So this replays the official script
                 locally and validates with the `canonical_sha256` the official side wrote down.
                 If validation fails everything downstream stops.
        Args: dry_run - print the command only; force - regenerate even when valid; expected -
              the official `canonical_source -> sha256` map, used to decide whether to skip;
              timeout - subprocess timeout (the script takes about 2.5 minutes).
        Returns: dict mapping the official `canonical_source` to the actual sha256 (LF-normalised).
        Data source: the compact sources under Dataset_construction_scripts/files (fetched in
                     A-2) and the script itself from the same pinned commit (added in A-5).
        Config: none (the official directory layout is the config).
        Side effects: writes under data/raw/polybench26/Datasets/{PSMILES,wPSMILES}; it does
                     **not** touch the published PolyMetriX file.
        Raises: SplitError when the script is missing, the subprocess fails, or a generated table
                disagrees with `canonical_sha256`.
    """
    script = PB26_SCRIPTS / "OMers_convert_jsonl.py"
    if not script.is_file():
        raise SplitError(
            f"缺官方生成脚本：{script}；先跑 `python scripts/00_download.py --source polybench26`"
        )
    if dry_run:
        LOGGER.info("  dry-run：将在 %s 下跑 %s / would run the official script", PB26_SCRIPTS, script.name)
        return {}

    # 幂等：官方脚本一次要跑两分半。若每张表都已在位且 LF 摘要对得上，就不必再生成——
    # 重跑只会让 mtime 变化，内容一模一样。
    # Idempotent: the official script takes ~2.5 minutes; if every table is already valid, skip.
    if expected and not force and all(
        (PB26_CANONICAL_ROOT / source).is_file()
        and lf_normalized_sha256(PB26_CANONICAL_ROOT / source) == digest
        for source, digest in expected.items()
    ):
        LOGGER.info("canonical 表已就位且哈希全对，跳过生成（要重生成加 --force）/ already valid, skipped")
        return {source: lf_normalized_sha256(PB26_CANONICAL_ROOT / source) for source in expected}

    LOGGER.info("按官方脚本物化 canonical 表（约 2–3 分钟）/ materialising canonical tables")
    completed = subprocess.run(
        [sys.executable, script.name],
        cwd=str(PB26_SCRIPTS),
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    if completed.returncode != 0:
        raise SplitError(
            f"官方生成脚本失败（exit {completed.returncode}）：{completed.stderr[-800:]}"
        )

    # pandas 在 Windows 上按 `os.linesep` 写行尾（CRLF），官方 `canonical_sha256` 与官方
    # `manage_scaling_indices.py verify` 都只认 LF——后者直接用 `sha256sum` 比字节，不会替我们
    # 归一。所以**把生成物统一成 LF 写回**：这些文件是我们刚生成的、不是下载来的，
    # data/raw 里"原样"的承诺针对下载字节，不针对本地生成物（D-21）。
    # pandas writes CRLF on Windows while both the official digest and the official
    # `manage_scaling_indices.py verify` assume LF (the latter hashes bytes and will not
    # normalise for us). So the generated files are rewritten with LF: they are locally
    # produced, and the "raw stays raw" promise covers downloaded bytes, not generated ones
    # (D-21).
    produced: dict[str, str] = {}
    normalised = 0
    for path in sorted((PB26_CANONICAL_ROOT / "PSMILES").rglob("*.csv")):
        payload = path.read_bytes()
        if b"\r\n" in payload:
            path.write_bytes(payload.replace(b"\r\n", b"\n"))
            normalised += 1
        # `.as_posix()`：官方 manifest 里写的是正斜杠；Windows 的 `str(Path)` 给反斜杠，
        # 直接比会得到"官方要求的这一份没生成"这种假失败。
        # `.as_posix()`: the official manifest uses forward slashes; `str(Path)` on Windows
        # yields backslashes and would fake a "this table was not generated" failure.
        produced[path.relative_to(PB26_CANONICAL_ROOT).as_posix()] = lf_normalized_sha256(path)
    if normalised:
        LOGGER.info("  行尾归一（CRLF→LF）%d 张 / line endings normalised", normalised)
    LOGGER.info("  生成 %d 张 canonical 表 / %d tables written", len(produced), len(produced))
    return produced


def _canonical_sha256_map(configs: Mapping[str, Mapping[str, Any]]) -> dict[str, str]:
    """收集本书所有 PolyBench26 任务声称的 `canonical_source -> canonical_sha256` 映射。
    Collect every `canonical_source -> canonical_sha256` pair claimed by the PolyBench26 configs.

    中文：
        用途：把"官方说这张表的 sha256 应该是多少"集中成一张表，供生成后逐项核对。
        输入：configs —— 全部任务配置。
        输出：dict；键为 `PSMILES/...` 形式的相对路径，值为官方 sha256。
        数据来源：`data/raw/polybench26/Datasets/scaling_indices/**/manifest.json`。
        配置来源：各配置的 `datasets[].official_indices_path` 或 `split_ref`。
        副作用：只读。异常：SplitError —— 官方 manifest 缺失或字段缺失。

    English:
        Purpose: gather "what sha256 the official side says this table has" into one map so the
                 generated tables can be checked one by one.
        Args: configs. Returns: dict keyed by the `PSMILES/...` relative path.
        Data source: the official manifest.json files under data/raw.
        Config: `datasets[].official_indices_path` or `split_ref`.
        Side effects: read-only. Raises: SplitError on a missing manifest or field.
    """
    pairs: dict[str, str] = {}
    for task_id, config in configs.items():
        if config["benchmark"] != "polybench26" or config["status"] != "available":
            continue
        # `deferred` 的任务（架构迁移）官方只发物化脚本、不发索引，没有 manifest 可读，
        # 因此也没有 `canonical_sha256` 可比——跳过它不是放宽标准，是那一步本来不在这。
        # 判据用结构而不是 `split_status`：本函数在状态改成 deferred 之前就会被调用。
        # `deferred` tasks (architecture transfer) ship only a materialisation script, no
        # indices and no manifest, hence no `canonical_sha256` to compare - skipping them is not
        # a relaxation, the artefact simply does not exist upstream. The test is structural
        # (no datasets + a directory split_ref) because this runs before the status is flipped.
        if not config.get("datasets") and Path(str(config["split_ref"])).suffix != ".json":
            continue
        entries: list[Mapping[str, Any] | None] = list(config.get("datasets") or [None])
        for entry in entries:
            manifest_path = _official_manifest_for(config, entry)
            if not manifest_path.is_file():
                raise SplitError(f"{task_id}：官方 manifest 不在 {manifest_path}")
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            source = manifest.get("canonical_source")
            digest = manifest.get("canonical_sha256")
            if not source or not digest:
                raise SplitError(f"{task_id}：官方 manifest 缺 canonical_source/canonical_sha256")
            pairs[str(source)] = str(digest)
    return pairs


def _snapshot_paths(
    config: Mapping[str, Any], entry: Mapping[str, Any] | None
) -> tuple[Path, Path, str]:
    """给出一次快照的 (目标目录, 官方目录, 标签)；目标目录是"装 seed_42 的那个目录"。
    Return (destination, official directory, label) for one snapshot; the destination is the
    directory that *contains* seed_42.

    中文：
        用途：`snapshot_polybench26()` 与 `_verify_polybench26()` 必须用同一套路径推导，
              否则"写进去的地方"和"查的地方"会悄悄变成两处。规则只有两条：
              主表任务的 `split_ref` 写到 `.../seed_42/manifest.json`（目标目录是它上两级），
              机制实验任务的 `split_ref` 是一个目录（目标目录是它 + `dataset_id`）。
        输入：config —— 任务配置；entry —— `datasets[]` 的一条（主表传它的那条即可）。
        输出：(destination, official_dir, label)：
              destination —— `splits/polybench26/**` 下装 `seed_42` 的目录；
              official_dir —— `data/raw/.../<实验>/seed_42`；
              label —— 人读标签，`task_id` 或 `task_id/dataset_id`。
        数据来源：`split_ref` 与 `datasets[].official_indices_path`。
        配置来源：`configs/tasks/pb26_*.yaml`。
        副作用：无。异常：SplitError —— 配置形状不认识（既不是 seed_42/manifest.json 也没 datasets）。

    English:
        Purpose: `snapshot_polybench26()` and `_verify_polybench26()` must derive paths the same
                 way, otherwise "where we wrote" and "where we check" drift apart. Two rules
                 only: a main task's `split_ref` points at `.../seed_42/manifest.json` (the
                 destination is two levels up), a mechanism task's `split_ref` is a directory
                 (destination is that directory plus `dataset_id`).
        Args: config; entry - one `datasets[]` item (its own item for main tasks).
        Returns: (destination, official_dir, label) as described above.
        Data source: `split_ref` and `datasets[].official_indices_path`.
        Config: configs/tasks/pb26_*.yaml. Side effects: none.
        Raises: SplitError on an unrecognised config shape.
    """
    task_id = str(config["task_id"])
    split_ref = Path(str(config["split_ref"]))
    datasets = list(config.get("datasets") or [])

    # 先看 `split_ref` 的形状，再看有没有 datasets：主表任务两者都有（datasets 只有一条），
    # 若先看 datasets 就会给主表多拼一层 `dataset_id`。
    # Check the shape of `split_ref` first: main tasks have both (a single datasets entry), and
    # branching on datasets first would append a spurious `dataset_id` level.
    if split_ref.name == "manifest.json" and split_ref.parent.name == "seed_42":
        official = entry["official_indices_path"] if entry else (
            datasets[0]["official_indices_path"] if datasets else None
        )
        if not official:
            raise SplitError(f"{task_id}：split_ref 指向 seed_42/manifest.json，但没有 official_indices_path")
        return (
            REPO_ROOT / split_ref.parents[1],
            _as_seed_dir(PB26_ROOT / str(official)),
            task_id,
        )

    if entry is None:
        raise SplitError(f"{task_id}：split_ref={split_ref} 是目录，但没有 datasets 条目说明是哪个数据集")
    return (
        REPO_ROOT / split_ref / str(entry["dataset_id"]),
        _as_seed_dir(PB26_ROOT / str(entry["official_indices_path"])),
        f"{task_id}/{entry['dataset_id']}",
    )


def snapshot_polybench26(
    configs: Mapping[str, Mapping[str, Any]],
    *,
    limit: int | None = None,
    dry_run: bool = False,
) -> list[str]:
    """把官方 `seed_42/` 目录**原样**拷进 `splits/polybench26/`，并写来源旁注。
    Copy the official `seed_42/` directories verbatim into `splits/polybench26/` with provenance.

    中文：
        用途：官方索引是可执行证据，不是可再生成物——拷进仓库、**不改一个字节**，任何人
              `sha256sum` 一比就知道有没有人动过。每份旁边写一个 `PROVENANCE.json`（在
              `seed_42/` 之外，保持拷贝目录纯净）记 commit、源文件 sha256、canonical sha256。
        输入：configs；limit —— 最多几个任务；dry_run —— 只打印。
        输出：写成功的 "task_id/dataset_id" 列表。
        数据来源：`data/raw/polybench26/Datasets/scaling_indices/**/seed_42/`。
        配置来源：`config["split_ref"]` 与 `datasets[].dataset_id`。
        副作用：写 `splits/polybench26/**`（含 PROVENANCE.json）与 `splits_manifest.csv`。
        异常：SplitError —— 源目录缺失、拷贝后 sha256 与源不一致。

    English:
        Purpose: the official indices are executable evidence, not something to regenerate - copy
                 them in without touching a byte so any `sha256sum` comparison reveals edits.
                 A `PROVENANCE.json` sits beside (never inside) `seed_42/` with the commit, the
                 source digests and the canonical digest.
        Args: configs; limit; dry_run.
        Returns: the "task_id/dataset_id" keys written.
        Data source: the official seed_42 directories under data/raw.
        Config: `split_ref` and `datasets[].dataset_id`.
        Side effects: writes under splits/polybench26 plus the snapshot manifest.
        Raises: SplitError when a source directory is missing or a copy's digest differs.
    """
    done: list[str] = []
    rows: list[dict[str, Any]] = []
    tasks = [
        (task_id, config)
        for task_id, config in sorted(configs.items())
        if config["benchmark"] == "polybench26"
        and config["status"] == "available"
        # 架构迁移（task4）官方只发物化脚本，没有可快照的索引目录；判据与
        # `_canonical_sha256_map()` 相同，用结构而不是状态。
        # Architecture transfer ships only a materialisation script, nothing to snapshot; the
        # test matches `_canonical_sha256_map()` and is structural rather than status-based.
        and not (not config.get("datasets") and Path(str(config["split_ref"])).suffix != ".json")
    ]
    if limit is not None:
        tasks = tasks[:limit]

    for task_id, config in tasks:
        entries: list[Mapping[str, Any] | None] = list(config.get("datasets") or [None])
        for entry in entries:
            destination, source_dir, label = _snapshot_paths(config, entry)

            if dry_run:
                LOGGER.info("  dry-run：%s -> %s / would copy", source_dir, destination / "seed_42")
                continue
            if not source_dir.is_dir():
                raise SplitError(f"{label}：官方索引目录不在 {source_dir}")

            target_dir = destination / "seed_42"
            if target_dir.exists():
                shutil.rmtree(target_dir)
            shutil.copytree(source_dir, target_dir)

            manifest = json.loads((source_dir / "manifest.json").read_text(encoding="utf-8"))
            files: dict[str, str] = {}
            for path in sorted(target_dir.iterdir()):
                digest = sha256_of(path)
                if digest != sha256_of(source_dir / path.name):
                    raise SplitError(f"{label}：{path.name} 拷贝后哈希变了（这不该发生）")
                files[path.name] = digest

            provenance = {
                "task_id": task_id,
                "dataset_id": entry["dataset_id"] if entry else None,
                "benchmark": "polybench26",
                "protocol": str(config["split_protocol"]),
                "copied_at": _now_iso(),
                "source_repo": "https://github.com/rlearsch/PolymerBenchmark2026",
                "source_commit": "83655bc25901a8cabe3a1e5474b7ec26f01b919f",
                "source_dir": str(source_dir.relative_to(REPO_ROOT)),
                "canonical_source": manifest.get("canonical_source"),
                "canonical_sha256": manifest.get("canonical_sha256"),
                "seed": manifest.get("seed"),
                "num_folds": manifest.get("num_folds"),
                "train_sizes": manifest.get("train_sizes"),
                "n_rows": manifest.get("n_rows"),
                "files": files,
                "note": "原样拷贝，不改字节；校验命令见 splits/polybench26/README.md",
            }
            (destination / "PROVENANCE.json").write_text(
                json.dumps(provenance, ensure_ascii=False, indent=1) + "\n",
                encoding="utf-8",
                newline="\n",
            )
            total_test = sum(fold["n_test"] for fold in manifest["folds"])
            rows.append(
                {
                    "task_id": label,
                    "benchmark": "polybench26",
                    "protocol": str(config["split_protocol"]),
                    # `.as_posix()`：清单是给人看、给脚本比的；Windows 的反斜杠会让同一份
                    # 产物在 Linux 上变成另一条路径。
                    # `.as_posix()`: backslashes on Windows would make the same artefact a
                    # different path string on Linux.
                    "artifact": (destination / "seed_42").relative_to(REPO_ROOT).as_posix(),
                    "sha256": hashlib.sha256(
                        "".join(files[name] for name in sorted(files)).encode()
                    ).hexdigest(),
                    "bytes": sum((target_dir / name).stat().st_size for name in files),
                    "n_train": sum(fold["n_train_pool"] for fold in manifest["folds"]),
                    "n_test": total_test,
                    "created_at": _now_iso(),
                }
            )
            LOGGER.info("  %s：%d 折、test %d / copied", label, manifest["num_folds"], total_test)
            done.append(label)
    if rows:
        _append_manifest_rows(rows)
    return done


def _round_half_up(value: float) -> int:
    """四舍五入到整数，且 `x.5` 一律进位（Python 内置 `round` 是银行家舍入，会给出反直觉的 0）。
    Round half up; Python's built-in `round` uses banker's rounding and yields surprising zeros.

    中文：
        用途：分箱后每箱要抽 `0.1 * n_b` 条进验证/测试。`round(0.5) == 0` 会让一个 5 条的箱
              一条都不抽，分配变得不可预期；这里统一成"四舍五入，.5 进位"。
        输入：value —— 非负数。
        输出：int。
        数据来源：调用方。配置来源：无。副作用：无。异常：无。

    English:
        Purpose: allocate `0.1 * n_b` rows per bin; banker's rounding would silently allocate zero
                 from a five-row bin. Always round halves up.
        Args: value. Returns: int. Data source: caller. Config: none. Raises: none.
    """
    return int(np.floor(float(value) + 0.5))


def equal_width_edges(values: np.ndarray, bins: int = OPENPOLY_BINS) -> np.ndarray:
    """按数值域的等宽切分给出箱边界（论文口径："value range divided into 10 equal-width bins"）。
    Equal-width bin edges over the value range, as the paper specifies.

    中文：
        用途：分层抽样必须先有箱。官方说的是**值域等宽**，不是等频，所以用 `linspace(min, max)`
              而不是分位数——分位数会变成等频分层，那是另一种抽样，不能冒充官方规则。
        输入：values —— 该性质的非空标签（一维）；bins —— 箱数。
        输出：长度 bins+1 的边界数组，升序。
        数据来源：`data/raw/openpoly/data/final_polymer_properties_fromliterature.csv`。
        配置来源：`OPENPOLY_BINS`。副作用：无。
        异常：SplitError —— 标签全相同（值域为零宽，无法等宽切分）。

    English:
        Purpose: stratification needs bins, and the paper says equal **width**, not equal frequency;
                 hence linspace over [min, max] rather than quantiles. Quantile binning is a
                 different sampler and must not be passed off as the official rule.
        Args: values (1-D non-null labels); bins. Returns: ascending edges of length bins+1.
        Data source: the published OpenPoly CSV. Config: OPENPOLY_BINS. Raises: SplitError when
                the value range is degenerate.
    """
    low, high = float(np.min(values)), float(np.max(values))
    if not np.isfinite(low) or not np.isfinite(high) or high <= low:
        raise SplitError(f"值域为零宽或非有限（min={low}, max={high}），无法按等宽分箱")
    return np.linspace(low, high, bins + 1)


def _apportion(sizes: Sequence[int], total: int, *, caps: Sequence[int]) -> list[int]:
    """最大余数法：把 `total` 条按 `sizes` 的比例摊到各箱，且每箱不超过 `caps`。
    Largest-remainder apportionment: spread `total` across bins in proportion to `sizes`,
    never exceeding each bin's `caps`.

    中文：
        用途：分层抽样要同时满足两件事——总比例是 8:1:1，每箱内部也按比例。逐箱各自四舍五入
              做不到：误差会累积（168 条时训练集被压到 77.4%，读起来像另一种协议）。
              这里用最大余数法先保证总数精确，再按比例分摊；配额超过某箱容量的部分，
              退回给还有余量的箱继续分。
        输入：sizes —— 每箱条数；total —— 要摊出去的总条数；caps —— 每箱最多能拿几条。
        输出：长度与 sizes 相同的整数列表，合计等于 `total`（容量不足时尽力而为）。
        数据来源：调用方。配置来源：无。副作用：无。异常：无。

    English:
        Purpose: stratified sampling must satisfy two things at once - an 8:1:1 global ratio and
                 proportionality inside each bin. Rounding bin by bin cannot do both; the error
                 accumulates (a 77.4% train share on the 168-row task reads like another
                 protocol). Largest remainder guarantees the exact total first, then distributes in
                 proportion; whatever exceeds a bin's capacity is reoffered to bins with room.
        Args: sizes; total; caps. Returns: int allocations summing to `total` where capacity
        allows. Data source: caller. Raises: none.
    """
    sizes = list(sizes)
    caps = list(caps)
    alloc = [0] * len(sizes)
    remaining = int(total)
    while remaining > 0:
        open_bins = [index for index in range(len(sizes)) if alloc[index] < caps[index]]
        if not open_bins:
            break
        weight = sum(sizes[index] for index in open_bins) or 1
        ideal = {index: remaining * sizes[index] / weight for index in open_bins}
        base = {index: int(np.floor(ideal[index])) for index in open_bins}
        for index in open_bins:
            alloc[index] += base[index]
        remaining -= sum(base.values())
        # 余数从大到小补齐；并列时按箱号，保证可复现。
        # Hand out the leftovers by descending remainder, ties broken by bin index for stability.
        for index in sorted(open_bins, key=lambda i: (-(ideal[i] - base[i]), i)):
            if remaining <= 0:
                break
            if alloc[index] < caps[index]:
                alloc[index] += 1
                remaining -= 1
    return alloc


def _bin_members(edges: np.ndarray, values: np.ndarray) -> list[np.ndarray]:
    """把每个样本归到箱里，返回每箱的**绝对行号**（末箱闭区间）。
    Assign every sample to a bin and return each bin's absolute row numbers (last bin closed).

    中文：
        用途：`np.digitize` 的左闭右开语义会让最大值落到最后一箱之外，必须夹到最后一箱；
              否则最大值那一条永远进不了测试集，而极端值恰恰是官方分层要保住的东西。
        输入：edges —— 箱边界；values —— 标签。
        输出：长度 bins 的列表，元素是该箱内样本的行号数组。
        数据来源：同 `equal_width_edges()`。配置来源：无。副作用：无。异常：无。

    English:
        Purpose: `np.digitize` is left-closed/right-open, so the maximum falls outside the last
                 bin; clip it in. Otherwise the extreme value can never reach test, and extremes
                 are exactly what the official stratification is meant to preserve.
        Args: edges; values. Returns: list of index arrays, one per bin. Raises: none.
    """
    bins = len(edges) - 1
    which = np.clip(np.digitize(values, edges[1:-1], right=False), 0, bins - 1)
    return [np.flatnonzero(which == index) for index in range(bins)]


def stratified_split(
    values: np.ndarray,
    *,
    seed: int,
    bins: int = OPENPOLY_BINS,
    ratios: Mapping[str, float] = OPENPOLY_RATIOS,
) -> dict[str, np.ndarray]:
    """按官方规则做一次 8:1:1 分层抽样，返回三份互斥且覆盖全量的行号。
    One 8:1:1 stratified draw under the official rule; returns three disjoint, exhaustive sets.

    中文：
        用途：这是 D-24 的核心。规则三句：8:1:1、值域 10 等宽箱、箱内按比例抽。实现只在
              "怎么取整、怎么打乱"上做了两个可核查的选择（见下），其余照抄。
              ① 先用最大余数法定出验证集与测试集的总条数（`round_half_up(0.1 * N)`），再按
                 每箱大小摊到箱里（`_apportion()`），余下的都进训练。单箱容量不足时（会掏空
                 训练集）配额退回给别的箱。
              ② 箱内用 `np.random.default_rng(seed)` 打乱后**按位置切**，不排序取值——
                 排序取值会变成"每箱取最小/最大的几条"，那是另一种抽样。
        输入：values —— 非空标签（一维）；seed —— 划分种子；bins —— 箱数；ratios —— 三份比例。
        输出：`{"train": 行号, "val": 行号, "test": 行号}`，各已升序，互斥且并集为 `range(n)`。
        数据来源：调用方传入的标签列。配置来源：`OPENPOLY_*` 常量。
        副作用：无。异常：SplitError —— 值域退化，或三份不互斥/没覆盖全量。
        可复现性：同一 seed 在同一份输入上逐位相同（PCG64 跨平台稳定）。

    English:
        Purpose: the heart of D-24. The rule has three clauses (8:1:1, ten equal-width bins of the
                 value range, proportional draw per bin). Only two implementation choices are
                 ours, and both are checkable: (1) the val and test totals are fixed first
                 (`round_half_up(0.1*N)`) and then apportioned across bins by size
                 (`_apportion()`), with the remainder going to train; a bin too small to give up
                 rows passes its quota back to the others.
                 (2) within a bin the order is a `default_rng(seed)` permutation, sliced by
                 position - not sorted by value, which would be a different sampler.
        Args: values; seed; bins; ratios.
        Returns: dict of ascending index arrays, disjoint and exhaustive over `range(n)`.
        Data source: the labels handed in. Config: the OPENPOLY_* constants.
        Raises: SplitError on a degenerate range or a non-partition result.
        Reproducibility: identical for the same seed on the same input (PCG64 is stable).
    """
    values = np.asarray(values, dtype=float)
    edges = equal_width_edges(values, bins)
    rng = np.random.default_rng(int(seed))
    members = _bin_members(edges, values)
    sizes = [len(group) for group in members]

    # 两级最大余数法：先定"验证集共几条、测试集共几条"，再按每箱大小把它们摊到箱里。
    # 逐箱各自四舍五入会累积误差——168 条时训练集被压到 77.4%，看着像另一种协议。
    # Two-level largest remainder: fix the val/test totals first, then apportion them across bins
    # in proportion to bin size. Rounding each bin independently accumulates error and pushed the
    # 168-row task to a 77.4% train share, which reads like a different protocol.
    n_val_total = _round_half_up(len(values) * ratios["val"])
    n_test_total = _round_half_up(len(values) * ratios["test"])
    per_bin_val = _apportion(sizes, n_val_total, caps=[size - 1 for size in sizes])
    per_bin_test = _apportion(
        sizes, n_test_total, caps=[size - 1 - used for size, used in zip(sizes, per_bin_val)]
    )

    parts: dict[str, list[int]] = {"train": [], "val": [], "test": []}
    for group, n_val, n_test in zip(members, per_bin_val, per_bin_test):
        order = rng.permutation(group)
        parts["val"].extend(order[:n_val].tolist())
        parts["test"].extend(order[n_val : n_val + n_test].tolist())
        parts["train"].extend(order[n_val + n_test :].tolist())

    result = {name: np.sort(np.asarray(indices, dtype=np.int64)) for name, indices in parts.items()}
    total = sum(len(indices) for indices in result.values())
    union = np.union1d(np.union1d(result["train"], result["val"]), result["test"])
    if total != len(values) or len(union) != len(values):
        raise SplitError(f"分层抽样不是全量划分：三段合计 {total}，并集 {len(union)}，总样本 {len(values)}")
    return result


def _openpoly_label_rows(config: Mapping[str, Any]) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """取出某性质的"非空标签"子表，返回（子表, 标签, 原始数据行号）。
    Return (subset, labels, original data-row numbers) for one property's non-null rows.

    中文：
        用途：OpenPoly 是 741 行 × 32 列的一张宽表，每个性质的样本量不同（Tg 443 / 介电总 295 /
              Tm 168）。划分只在**该性质非空的行**上做，所以必须先把子表切出来，并把
              "子表位置 → 发布表的第几行"记下来，否则划分无法回溯到原始数据。
        输入：config —— 一个 OpenPoly 任务配置。
        输出：(subset, values, source_row)；source_row 是 0 基的数据行号（不含表头）。
        数据来源：`config["data_ref"]` 指向的发布 CSV。
        配置来源：`target_column`、`n_labels`。副作用：只读。
        异常：SplitError —— 列不存在、非空条数与配置不符、含非有限值。

    English:
        Purpose: OpenPoly is one 741x32 wide table and each property has its own sample size
                 (Tg 443 / dielectric total 295 / Tm 168). The split only covers the property's
                 non-null rows, so the subset and the mapping back to published row numbers must
                 be materialised or the split cannot be traced to the data.
        Args: config. Returns: (subset, values, source_row) with 0-based data-row numbers.
        Data source: the published CSV at `data_ref`. Config: `target_column`, `n_labels`.
        Raises: SplitError on a missing column, a count mismatch, or non-finite values.
    """
    task_id = str(config["task_id"])
    column = str(config["target_column"])
    table = pd.read_csv(REPO_ROOT / str(config["data_ref"]), encoding="utf-8-sig")
    if column not in table.columns:
        raise SplitError(f"{task_id}：发布表里没有列 {column!r}")
    subset = table.loc[table[column].notna()].copy()
    values = subset[column].to_numpy(dtype=float)
    if len(values) != int(config["n_labels"]):
        raise SplitError(
            f"{task_id}：{column} 的非空条数是 {len(values)}，配置写 {config['n_labels']} 条"
        )
    if not np.isfinite(values).all():
        raise SplitError(f"{task_id}：{column} 含 inf/nan，不能用于分箱")
    source_row = subset.index.to_numpy(dtype=np.int64)
    return subset, values, source_row


def build_openpoly_split(config: Mapping[str, Any], *, seed: int) -> dict[str, Any]:
    """按官方规则复现一份划分，返回可落盘的对象（不含时间戳，保证同 seed 幂等）。
    Reproduce one split under the published rule; returns a timestamp-free, idempotent payload.

    中文：
        用途：这是"复现"而不是"自创"的关键——所有取值都由 `stratified_split()` 决定，
              本函数只负责把它和"它凭什么这么切"的元数据绑在一起，让任何人拿到文件就能复核：
              规则引文、种子、箱边界、每箱条数、原始数据行号、结构泄漏计数。
        输入：config —— OpenPoly 任务配置；seed —— 划分种子。
        输出：dict，含 `train_index` / `val_index` / `test_index` 与 `source_row`（0 基数据行号）。
        数据来源：`config["data_ref"]` 的发布 CSV。
        配置来源：`target_column`、`n_labels`、`split_protocol`。
        副作用：只读。
        异常：SplitError —— 见 `_openpoly_label_rows()` 与 `stratified_split()`。

    English:
        Purpose: this is a *reproduction*, not an invention: every index comes from
                 `stratified_split()`, and this function only binds the result to the metadata
                 that lets anyone re-check it - the rule quotation, the seed, the bin edges, the
                 per-bin counts, the original data-row numbers, and the structural-leakage count.
        Args: config; seed. Returns: dict with the three index arrays plus `source_row`.
        Data source: the published CSV at `data_ref`. Config: `target_column`, `n_labels`.
        Side effects: read-only. Raises: SplitError (see the two helpers above).
    """
    task_id = str(config["task_id"])
    subset, values, source_row = _openpoly_label_rows(config)
    parts = stratified_split(values, seed=seed)
    edges = equal_width_edges(values, OPENPOLY_BINS)
    per_bin = [len(members) for members in _bin_members(edges, values)]

    # 结构泄漏：同一 PSMILES 在 741 行里可能重复（656 个不同结构），若同结构跨进训练与测试，
    # 那是结构级泄漏。官方规则没有去重，我们照规则切、但把这件事**量出来记账**（D-24）。
    # Structural leakage: the 741 rows hold only 656 distinct PSMILES, so one structure can land
    # in both train and test. The official rule does not group, so we follow it and *measure* it.
    structures = subset["PSMILES"].astype(str).to_numpy()
    seen: dict[str, set[str]] = {}
    for name, indices in parts.items():
        for structure in structures[indices]:
            seen.setdefault(structure, set()).add(name)
    shared = [structure for structure, names in seen.items() if len(names) > 1]

    ratios = {name: round(len(indices) / len(values), 6) for name, indices in parts.items()}
    return {
        "schema_version": 1,
        "task_id": task_id,
        "benchmark": "openpoly",
        "protocol": OPENPOLY_PROTOCOL,
        "protocol_source": OPENPOLY_PROTOCOL_SOURCE,
        "status": "rule_replicated",
        "seed": int(seed),
        "fold_role": "primary" if seed in OPENPOLY_PRIMARY_FOLDS else "reserve",
        "target_column": str(config["target_column"]),
        "target_unit": str(config.get("target_unit", "")),
        "n_total": int(len(values)),
        "n_train": int(len(parts["train"])),
        "n_val": int(len(parts["val"])),
        "n_test": int(len(parts["test"])),
        "ratios": ratios,
        "ratios_nominal": dict(OPENPOLY_RATIOS),
        "bins": OPENPOLY_BINS,
        "bin_edges": [float(edge) for edge in edges],
        "bin_counts": [int(count) for count in per_bin],
        "row_order": f"{config['data_ref']}#0 基数据行号，只取 {config['target_column']} 非空的行，保持发布顺序",
        "rule": {
            "ratio": "8:1:1",
            "stratification": "值域 10 个等宽箱，箱内按比例抽（论文原文见 protocol_source）",
            "seed_published_by_authors": None,
            "seed_used_here": int(seed),
            "note": "官方未公布随机种子；本文件是同一规则下的一次新抽样，不是作者那一份划分",
        },
        "source": {
            "data_ref": str(config["data_ref"]),
            "sha256": sha256_of(REPO_ROOT / str(config["data_ref"])),
            "n_rows_raw": int(len(pd.read_csv(REPO_ROOT / str(config["data_ref"]), encoding="utf-8-sig"))),
            "n_labels": int(len(values)),
        },
        "leakage": {
            "n_psmiles_shared_across_splits": len(shared),
            "examples": sorted(shared)[:5],
            "note": "官方规则不按结构分组；这里是量出来的既有事实，不是我们引入的偏差",
        },
        "source_row": [int(value) for value in source_row],
        "train_index": [int(value) for value in parts["train"]],
        "val_index": [int(value) for value in parts["val"]],
        "test_index": [int(value) for value in parts["test"]],
    }


def write_openpoly_protocol_note(*, dry_run: bool = False) -> dict[str, Any]:
    """写下 OpenPoly 的**规则复现**说明：规则是什么、谁说的、缺什么、我们怎么做。
    Record the OpenPoly **rule reproduction** card: what the rule is, who said it, what is
    missing, and what we do instead.

    中文：
        用途：A-5 的验收 ⑤ 原本判"做不到"（D-22），因为当时以为连规则都没发布。复核后发现
              规则在论文正文里写得很清楚（8:1:1 + 值域 10 等宽箱分层），缺的只是**那一份划分
              与随机种子**。所以状态改为 `rule_replicated`（D-24），本文件把这个改判的依据、
              查询范围（GitHub 全树 / Zenodo / 期刊补充材料三处）与验收靶子一次写清。
        输入：dry_run —— 只打印。
        输出：写进文件的 dict。
        数据来源：论文正文、GitHub 全树、Zenodo 记录、官方 `results/*.csv`。
        配置来源：`OPENPOLY_*` 常量。副作用：写 `splits/openpoly/PROTOCOL.json`。
        异常：OSError —— 无写权限。

    English:
        Purpose: A-5 criterion 5 was originally judged impossible (D-22) on the belief that even
                 the *rule* was unpublished. Re-checking showed the rule is stated plainly in the
                 paper (8:1:1 plus ten equal-width value bins); what is missing is only the
                 partition itself and its seed. The status therefore becomes `rule_replicated`
                 (D-24), and this file records the change, the search surface (GitHub tree,
                 Zenodo, journal supplement) and the acceptance targets in one place.
        Args: dry_run. Returns: the dict written. Side effects: writes PROTOCOL.json.
        Raises: OSError without write access.
    """
    note = {
        "benchmark": "openpoly",
        "status": "rule_replicated",
        "decision": "D-24",
        "supersedes": {
            "decision": "D-22",
            "status_then": "blocked_upstream",
            "why_changed": (
                "D-22 的判断建立在\"官方连划分规则都没发布\"上。复核论文正文后这一条不成立："
                "METHODS 明确写了 8:1:1 + 值域 10 等宽箱分层抽样。缺的是那一份划分结果与随机种子。"
            ),
        },
        "rule": {
            "quotes": [
                "the dataset was randomly split into training, validation, and test sets at an 8:1:1 ratio.",
                "for each property, its numerical value range was divided into 10 equal-width bins, "
                "and samples were proportionally drawn from each bin to construct the training, "
                "validation, and test subsets.",
                "All models were trained, tuned, and tested on the same data partitions.",
            ],
            "source": OPENPOLY_PROTOCOL_SOURCE,
            "seed_published": False,
        },
        "what_is_missing": [
            "experiment_dataset/{train,val,test}_<prop>_data_list.pt（notebook 只读它们，仓库里没有）",
            "生成这些 .pt 的脚本（4 个 notebook 只 import train_test_split，没有一处调用它切分）",
            "随机种子 / random_state（论文全文未提；notebook 里的 random_state=42 是模型种子）",
            "期刊补充材料与数据可用性声明（经查：两者都不存在）",
        ],
        "search_log": {
            "github_tree": "ccaa669，递归全树 39 条目（26 文件 + 13 目录），无 experiment_dataset/，无 .gitignore；"
            "__pycache__ 的 .pyc 都被提交 → 作者是整目录上传，缺的目录是真的没传",
            "zenodo": "10.5281/zenodo.15551637 只有 1 个文件 trained_model.zip（460 MB，8 个模型的权重）",
            "journal": "DOI 10.1007/s10118-025-3402-y 无 Supplementary Information，也无 data availability 章节",
        },
        "what_we_do": {
            "approach": "按公布的规则复现 R 份划分，公开种子，报区间而不是报单点",
            "seeds": list(OPENPOLY_SEEDS),
            "primary_folds": list(OPENPOLY_PRIMARY_FOLDS),
            "reserve_seeds": [seed for seed in OPENPOLY_SEEDS if seed not in OPENPOLY_PRIMARY_FOLDS],
            "fold_mapping": (
                "seed 0–4 就是 eval_unit 里的 5 折（每一折是一次独立的 8:1:1 抽样，不是互斥分块）；"
                "seed 5–9 留作划分敏感性分析。总拟合次数 3 性质 × 5 档 × 5 折 × 3 种子 = 225，不变"
            ),
            "why_not_a_red_line": (
                "红线禁的是\"不读官方划分、自己随手切\"。这里切法是论文正文逐字给出的，种子公开，"
                "每份划分带箱边界与每箱条数，任何人可用同一命令逐位复现；并且我们如实声明"
                "\"这不是作者那一份划分\"（Proposal §7.5 要防的是隐去这件事，不是禁止复现已公布的规则）"
            ),
        },
        "acceptance_target": {
            "why": "复现对不对，不能靠自觉，要靠能否跑出官方公布的成绩",
            "source": "results/final_model_test_performance_summary.csv（官方仓库）",
            "protocol": "Morgan 指纹 r=2 / 2048 维 + XGBoost，模型 random_state=42",
            "expected_test_r2": {
                "op_tg": 0.904375715,
                "op_dielectric_constant_total": 0.86,
                "op_tm": 0.807551835,
            },
            "pass_rule": (
                "R 份划分各跑一遍，官方值应落在 R 个结果的 min–max 区间内（或偏离 ≤ 0.05）；"
                "落不进去说明我们读错了规则，必须回到本文件重新核对，而不是调参凑数"
            ),
        },
        "known_limitations": [
            "与论文不是同一份划分，绝对 R² 不可逐位对比；H4 的臂间对比（同一划分内）不受影响",
            "8:1:1 下 Tm 的测试集只有约 17 条，单份划分的 R² 方差很大——这正是要报 R 份区间的原因",
            "741 行只含 656 个不同 PSMILES，官方规则不按结构分组，跨集结构泄漏由 leakage 字段记账",
        ],
        "affected_tasks": ["op_tg", "op_dielectric_constant_total", "op_tm"],
        "generated_at": _now_iso(),
    }
    if not dry_run:
        OPENPOLY_PROTOCOL_NOTE.parent.mkdir(parents=True, exist_ok=True)
        OPENPOLY_PROTOCOL_NOTE.write_text(
            json.dumps(note, ensure_ascii=False, indent=1) + "\n", encoding="utf-8", newline="\n"
        )
        LOGGER.info("已写协议说明 / protocol note written: %s", OPENPOLY_PROTOCOL_NOTE)
    return note


def build_openpoly_splits(
    configs: Mapping[str, Mapping[str, Any]],
    *,
    seeds: Sequence[int] = OPENPOLY_SEEDS,
    limit: int | None = None,
    dry_run: bool = False,
) -> list[str]:
    """为每个 OpenPoly 边界任务生成 R 份规则复现划分，并写两份清单。
    Emit R rule-conforming splits per OpenPoly boundary task and update both manifests.

    中文：
        用途：切分本身由 `build_openpoly_split()` 完全决定；本函数只管落盘、算 sha256、
              维护 `data/manifests/openpoly_split_manifest.csv`（每份一行，含 fold_role）与
              `data/manifests/splits_manifest.csv`（全局一行一份产物）。
              幂等：划分文件不含时间戳，同 seed 重跑字节相同，旧 `created_at` 沿用。
        输入：configs —— task_id -> 配置；seeds —— 要生成的种子；limit —— 最多几个任务；
              dry_run —— 只打印。
        输出：写成功的 task_id 列表。
        数据来源：发布 CSV（`data_ref`）。配置来源：`split_ref`（目标目录）。
        副作用：写 `splits/openpoly/<task>/`、两份清单、`PROTOCOL.json`。
        异常：SplitError —— 规则不成立；OSError —— 无写权限。

    English:
        Purpose: the partition itself is decided entirely by `build_openpoly_split()`; this
                 function only persists it, hashes it, and maintains the per-seed manifest plus
                 the global snapshot manifest. Timestamp-free files make re-runs byte-identical,
                 so the old `created_at` is carried over.
        Args: configs; seeds; limit; dry_run. Returns: task ids written.
        Data source: the published CSV. Config: `split_ref` is the destination directory.
        Side effects: writes splits/openpoly/**, both manifests and PROTOCOL.json.
        Raises: SplitError on a broken rule; OSError without write access.
    """
    targets = [
        (task_id, config)
        for task_id, config in sorted(configs.items())
        if config["benchmark"] == "openpoly" and config["status"] == "available"
    ]
    if limit is not None:
        targets = targets[:limit]

    write_openpoly_protocol_note(dry_run=dry_run)

    previous = {row["artifact"]: row for row in _read_manifest_rows(OPENPOLY_MANIFEST)}
    existing_global = {(row["task_id"], row["artifact"]): row for row in _read_manifest_rows(SPLITS_MANIFEST)}
    done: list[str] = []
    rows: list[dict[str, Any]] = []
    global_rows: list[dict[str, Any]] = []
    for task_id, config in targets:
        directory = REPO_ROOT / str(config["split_ref"])
        if dry_run:
            LOGGER.info("  dry-run：将在 %s 下生成 %d 份规则复现划分 / would build", config["split_ref"], len(seeds))
            continue
        directory.mkdir(parents=True, exist_ok=True)
        for seed in seeds:
            split = build_openpoly_split(config, seed=seed)
            destination = directory / f"rule_replicated_seed{seed}.json"
            destination.write_text(
                json.dumps(split, ensure_ascii=False, indent=1) + "\n", encoding="utf-8", newline="\n"
            )
            artifact = destination.relative_to(REPO_ROOT).as_posix()
            digest = sha256_of(destination)
            # 同一个 sha256 就是同一份产物：沿用旧时间戳，免得每次重跑都像有人改过冻结物。
            # Same digest means the same artefact: keep the old timestamp so re-runs do not look
            # like hand edits.
            old = previous.get(artifact)
            created_at = old["created_at"] if old and old.get("sha256") == digest else _now_iso()
            rows.append(
                {
                    "task_id": task_id,
                    "protocol": split["protocol"],
                    "seed": seed,
                    "fold_role": split["fold_role"],
                    "artifact": artifact,
                    "sha256": digest,
                    "bytes": destination.stat().st_size,
                    "n_total": split["n_total"],
                    "n_train": split["n_train"],
                    "n_val": split["n_val"],
                    "n_test": split["n_test"],
                    "n_psmiles_shared": split["leakage"]["n_psmiles_shared_across_splits"],
                    "created_at": created_at,
                }
            )
            old_global = existing_global.get((task_id, artifact))
            global_rows.append(
                {
                    "task_id": task_id,
                    "benchmark": "openpoly",
                    "protocol": split["protocol"],
                    "artifact": artifact,
                    "sha256": digest,
                    "bytes": destination.stat().st_size,
                    "n_train": split["n_train"],
                    "n_val": split["n_val"],
                    "n_test": split["n_test"],
                    "created_at": old_global["created_at"] if old_global else created_at,
                }
            )
            LOGGER.info(
                "  %s · seed %d：train %d / val %d / test %d（%s）/ written",
                task_id,
                seed,
                split["n_train"],
                split["n_val"],
                split["n_test"],
                split["fold_role"],
            )
        done.append(task_id)

    if not dry_run and rows:
        _write_manifest(rows, OPENPOLY_MANIFEST, OPENPOLY_MANIFEST_COLUMNS)
        _append_manifest_rows(global_rows)
    return done

# ---- 5. 校验：把验收标准写成断言 / Verification: acceptance criteria as assertions ----


def nested_prefix(pool: Sequence[int], n_train: int) -> list[int]:
    """取"前 n 个"作为训练集——这是官方档位的定义，也是 B/C 共用的不变量。
    Take the first `n_train` entries as the training set - the official tier rule and a shared
    invariant for B and C.

    中文：
        用途：官方 scaling 档位不是一个新抽的样本，而是 `train_pool` 的**前缀**。这条不写在
              任何文件里，但一旦有人改成随机抽样，档位之间就不再嵌套，κ 曲线会凭空变形。
              所以把它做成唯一的取数函数，让"前缀"这件事只有一种实现。
        输入：pool —— 官方 `train_pool` 数组；n_train —— 档位（300/1000/3000/10000）。
        输出：前 `n_train` 个元素的新列表。
        数据来源：`splits/polybench26/**/seed_42/fold_*.npz` 的 `train_pool`。
        配置来源：官方 manifest 的 `train_sizes`。
        副作用：无。异常：SplitError —— n_train 超过池子长度（说明档位与索引不配套）。

    English:
        Purpose: an official tier is not a fresh sample but a **prefix** of `train_pool`. Nobody
                 writes that down, yet switching to random sampling would break nesting and bend
                 the kappa curve. Making it the only accessor keeps one implementation of "prefix".
        Args: pool - the official `train_pool` array; n_train - the tier size.
        Returns: a new list with the first `n_train` entries.
        Data source: `train_pool` of the official fold_*.npz files.
        Config: `train_sizes` in the official manifest.
        Side effects: none. Raises: SplitError when n_train exceeds the pool length.
    """
    if n_train > len(pool):
        raise SplitError(f"档位 {n_train} 超过 train_pool 长度 {len(pool)}；档位与索引不配套")
    return list(pool[:n_train])


def load_fold(directory: Path, fold: int) -> dict[str, np.ndarray]:
    """读一折的 `test / val / train_pool` 三个数组。
    Load one fold's `test`, `val` and `train_pool` arrays.

    中文：
        用途：三处（校验、训练、分析）都要读折文件，格式只在这里定义一次。
        输入：directory —— 含 `fold_XX.npz` 的目录；fold —— 折号。
        输出：dict，键为 `test` / `val` / `train_pool`，值为 int64 数组。
        数据来源：`splits/polybench26/**/seed_42/fold_XX.npz`。
        配置来源：无。副作用：只读。
        异常：FileNotFoundError —— 折文件不在；SplitError —— 缺键。

    English:
        Purpose: validation, training and analysis all read fold files; define the shape once.
        Args: directory; fold index. Returns: dict of int64 arrays keyed test/val/train_pool.
        Data source: the official fold_XX.npz files. Config: none. Side effects: read-only.
        Raises: FileNotFoundError when the fold is missing; SplitError on a missing key.
    """
    path = directory / f"fold_{fold:02d}.npz"
    if not path.is_file():
        raise FileNotFoundError(f"折文件不在：{path}")
    with np.load(path) as payload:
        missing = {"test", "val", "train_pool"} - set(payload.keys())
        if missing:
            raise SplitError(f"{path.name} 缺数组：{sorted(missing)}")
        return {key: payload[key].astype(np.int64) for key in ("test", "val", "train_pool")}


def _verify_polybench(config: Mapping[str, Any], problems: list[str], deep: bool) -> None:
    """校验一份 PolyBench 划分 JSON（互斥、覆盖、比例、指纹、必要时重算）。
    Validate one PolyBench split JSON (disjoint, covers all, ratio, digest, optional recompute).

    中文：
        用途：`verify_splits()` 的 PolyBench 分支。`deep=True` 时会拿官方 CSV 重算一遍行号，
              代价是每个任务加载一次 parquet + 几千次 RDKit 规范化，所以默认关闭，由
              `--deep` 或交付前的验收手动开。
        输入：config —— 任务配置；problems —— 收集问题的列表（原地追加）；deep —— 是否重算。
        输出：无（问题写进 problems）。
        数据来源：`splits/polybench/<task>/official_70_30.json`、`splits_manifest.csv`。
        配置来源：`config["split_ref"]` / `n_labels`。
        副作用：只读。异常：无（问题收集而非抛出）。

    English:
        Purpose: the PolyBench branch of `verify_splits()`. With `deep=True` the row indices are
                 recomputed from the official CSV, which costs one parquet load and thousands of
                 RDKit canonicalisations per task, so it is opt-in.
        Args: config; problems - list appended to in place; deep - recompute or not.
        Returns: none. Data source: the split JSON plus the snapshot manifest.
        Config: `split_ref` / `n_labels`. Side effects: read-only. Raises: none (collects).
    """
    task_id = str(config["task_id"])
    path = REPO_ROOT / str(config["split_ref"])
    if not path.is_file():
        problems.append(f"{task_id}：split_status=present 但 {config['split_ref']} 不存在")
        return
    split = json.loads(path.read_text(encoding="utf-8"))
    if split.get("schema_version") != 1:
        problems.append(f"{task_id}：schema_version={split.get('schema_version')}，应为 1")
    if split.get("task_id") != task_id:
        problems.append(f"{task_id}：文件里的 task_id 是 {split.get('task_id')}")
    if split.get("protocol") != "official_70_30":
        problems.append(f"{task_id}：protocol={split.get('protocol')}，PolyBench 只允许 official_70_30")
    if "kfold" in str(split.get("protocol", "")).lower():
        problems.append(f"{task_id}：PolyBench 不得出现 k 折评估")

    test_index = list(split.get("test_index") or [])
    train_index = list(split.get("train_index") or [])
    n_total = int(config["n_labels"])
    if split.get("n_total") != n_total:
        problems.append(f"{task_id}：n_total={split.get('n_total')}，配置写 {n_total}")
    if len(test_index) != split.get("n_test"):
        problems.append(f"{task_id}：test_index 长度与 n_test 不符")
    if len(train_index) != split.get("n_train"):
        problems.append(f"{task_id}：train_index 长度与 n_train 不符")
    if len(test_index) + len(train_index) != n_total:
        problems.append(f"{task_id}：train+test={len(test_index) + len(train_index)}，应为 {n_total}")
    if len(set(test_index) & set(train_index)) != 0:
        problems.append(f"{task_id}：train 与 test 有交集")
    if len(set(test_index)) != len(test_index) or len(set(train_index)) != len(train_index):
        problems.append(f"{task_id}：索引里有重复行号")
    if test_index and (min(test_index) < 0 or max(test_index) >= n_total):
        problems.append(f"{task_id}：测试索引越界")
    if test_index != sorted(test_index):
        problems.append(f"{task_id}：测试索引没按行号升序（不同实现会算出不同顺序）")
    ratio = len(test_index) / n_total if n_total else 0.0
    if abs(ratio - EXPECTED_TEST_RATIO) > TEST_RATIO_TOLERANCE:
        problems.append(f"{task_id}：测试比例 {ratio:.4f}，超出 {EXPECTED_TEST_RATIO}±{TEST_RATIO_TOLERANCE}")

    recorded = _read_manifest(SPLITS_MANIFEST).get(task_id)
    if not recorded:
        problems.append(f"{task_id}：splits_manifest.csv 里没有这一行")
    elif recorded.get("sha256") != sha256_of(path):
        problems.append(f"{task_id}：划分文件被改过（sha256 与 splits_manifest.csv 不符）")

    if deep:
        test_csv = POLYBENCH_TEST_DIR / f"{config['dataset_name']}.csv"
        if not test_csv.is_file():
            problems.append(f"{task_id}：官方测试集不在 {test_csv}，无法重算")
        else:
            rebuilt = build_polybench_split(config, test_csv)
            if rebuilt["test_index"] != test_index:
                problems.append(f"{task_id}：按官方 CSV 重算的测试索引与落盘的不一致")


def _verify_polybench26(config: Mapping[str, Any], problems: list[str], deep: bool) -> None:
    """校验一份 PolyBench26 快照（折结构、互斥、覆盖、前缀、字节一致、canonical 哈希）。
    Validate one PolyBench26 snapshot (folds, disjointness, coverage, prefix, bytes, digest).

    中文：
        用途：`verify_splits()` 的 PolyBench26 分支。这里查的是官方 `verify` 子命令**不查**
              的那部分：`train_pool` 是否真的是补集、五折是否合起来刚好覆盖全量、拷贝有没有
              被改过、canonical 表的 sha256 是否等于官方写的 `canonical_sha256`。
        输入：config —— 任务配置；problems —— 问题列表；deep —— 是否重算 canonical 表哈希
              （只读比对，不重新生成）。
        输出：无。
        数据来源：`splits/polybench26/**`、`data/raw/polybench26/**`。
        配置来源：配置的 `split_ref` / `datasets[].official_indices_path`。
        副作用：只读。异常：无。

    English:
        Purpose: the PolyBench26 branch. It checks what the official `verify` subcommand does
                 **not**: that `train_pool` really is the complement, that the five folds cover
                 every row exactly once, that the copies were not edited, and that the canonical
                 table's digest equals the official `canonical_sha256`.
        Args: config; problems; deep - verify canonical digests (read-only, no regeneration).
        Returns: none. Data source: splits/polybench26 and data/raw/polybench26.
        Config: `split_ref` / `datasets[].official_indices_path`.
        Side effects: read-only. Raises: none.
    """
    entries: list[Mapping[str, Any] | None] = list(config.get("datasets") or [None])
    for entry in entries:
        directory, official_dir, label = _snapshot_paths(config, entry)

        manifest_path = directory / "seed_42" / "manifest.json"
        if not manifest_path.is_file():
            problems.append(f"{label}：{manifest_path.relative_to(REPO_ROOT)} 不存在")
            continue
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        n_rows = int(manifest["n_rows"])
        num_folds = int(manifest["num_folds"])
        if num_folds != 5:
            problems.append(f"{label}：{num_folds} 折，官方是 5 折")
        if len(manifest["folds"]) != num_folds:
            problems.append(f"{label}：folds 列表长度与 num_folds 不符")

        for fold_meta in manifest["folds"]:
            fold = int(fold_meta["fold"])
            try:
                arrays = load_fold(directory / "seed_42", fold)
            except (FileNotFoundError, SplitError) as exc:
                problems.append(f"{label}：{exc}")
                continue
            test, val, pool = arrays["test"], arrays["val"], arrays["train_pool"]
            for name, array, expected in (
                ("test", test, fold_meta["n_test"]),
                ("val", val, fold_meta["n_val"]),
                ("train_pool", pool, fold_meta["n_train_pool"]),
            ):
                if len(array) != expected:
                    problems.append(f"{label} 第 {fold} 折：{name} 长 {len(array)}，manifest 写 {expected}")
            merged = np.concatenate([test, val, pool])
            if len(np.unique(merged)) != n_rows:
                problems.append(
                    f"{label} 第 {fold} 折：test+val+train_pool 去重后 {len(np.unique(merged))}，应为 {n_rows}"
                )
            if merged.min(initial=0) < 0 or merged.max(initial=-1) >= n_rows:
                problems.append(f"{label} 第 {fold} 折：行号越界（0..{n_rows - 1}）")
            if len(np.intersect1d(test, val)) or len(np.intersect1d(test, pool)) or len(np.intersect1d(val, pool)):
                problems.append(f"{label} 第 {fold} 折：test/val/train_pool 有交集")
            for tier in manifest["train_sizes"]:
                try:
                    prefix = nested_prefix(pool.tolist(), int(tier))
                except SplitError as exc:
                    problems.append(f"{label} 第 {fold} 折：{exc}")
                    continue
                if prefix != pool.tolist()[: int(tier)]:
                    problems.append(f"{label} 第 {fold} 折：档位 {tier} 不是 train_pool 的前缀")

        # 字节一致：仓库里的拷贝必须与 data/raw 下那份逐字节相同（A-5 的核心承诺）
        # Byte identity: the copy in the repo must equal the raw one byte for byte.
        if official_dir.is_dir():
            for source in sorted(official_dir.iterdir()):
                target = directory / "seed_42" / source.name
                if not target.is_file():
                    problems.append(f"{label}：splits 里缺 {source.name}")
                elif sha256_of(target) != sha256_of(source):
                    problems.append(f"{label}：{source.name} 与 data/raw 下的官方副本不一致")

        provenance = directory / "PROVENANCE.json"
        if not provenance.is_file():
            problems.append(f"{label}：缺 PROVENANCE.json（来源无处可追）")
        else:
            meta = json.loads(provenance.read_text(encoding="utf-8"))
            if meta.get("source_commit") != "83655bc25901a8cabe3a1e5474b7ec26f01b919f":
                problems.append(f"{label}：PROVENANCE 的 commit 不是冻结的那个")

        if deep:
            canonical = PB26_CANONICAL_ROOT / str(manifest["canonical_source"])
            if not canonical.is_file():
                problems.append(f"{label}：canonical 表不在 {canonical}（先跑 --source canonical）")
            elif lf_normalized_sha256(canonical) != manifest["canonical_sha256"]:
                problems.append(f"{label}：canonical 表 sha256 与官方 canonical_sha256 不等")


def _check_openpoly_split(
    payload: Mapping[str, Any],
    problems: list[str],
    *,
    task_id: str,
    seed: int,
    deep: bool,
    config: Mapping[str, Any] | None = None,
) -> None:
    """逐份检查一份 OpenPoly 规则复现划分是否真的遵守了论文那三句规则。
    Check one reproduced OpenPoly split actually obeys the paper's three clauses.

    中文：
        用途：复现规则最容易出的错不是"切不出来"，而是"切出来了但不是那个规则"——比如用了等频
              分箱、或者三份不互斥、或者换了种子却复用了旧文件。所以这里逐条验：协议名、
              三份比例、互斥与全量覆盖、箱数与箱内条数合计、原始行号可回溯；
              `deep=True` 时**用同一种子重算一遍**，与文件里的行号逐位比对。
        输入：payload —— 划分文件内容；problems —— 问题列表；task_id / seed —— 期望值；
              deep —— 是否重算；config —— 给出时用于重算。
        输出：无（问题写进 problems）。数据来源：划分 JSON、发布 CSV。
        配置来源：`OPENPOLY_*` 常量。副作用：只读。异常：无（收集而非抛出）。

    English:
        Purpose: the failure mode of a reproduced rule is not "cannot split" but "split under a
                 different rule" - equal-frequency bins, overlapping subsets, or a stale file left
                 behind under a new seed. So each file is checked for protocol name, the three
                 shares, disjointness and coverage, bin count and per-bin totals, and traceable
                 source rows; with `deep=True` it is recomputed from the seed and compared
                 index-by-index.
        Args: payload; problems; task_id; seed; deep; config (needed for recomputation).
        Returns: none (problems are appended). Raises: none.
    """
    label = f"{task_id} · seed {seed}"
    if payload.get("protocol") != OPENPOLY_PROTOCOL:
        problems.append(f"{label}：protocol 是 {payload.get('protocol')!r}，应为 {OPENPOLY_PROTOCOL!r}")
    if payload.get("task_id") != task_id:
        problems.append(f"{label}：文件里的 task_id 是 {payload.get('task_id')!r}")
    if int(payload.get("seed", -1)) != seed:
        problems.append(f"{label}：文件里的 seed 是 {payload.get('seed')!r}，与文件名不符")

    counts = {name: int(payload.get(f"n_{name}", -1)) for name in ("total", "train", "val", "test")}
    if counts["total"] <= 0:
        problems.append(f"{label}：n_total 非法（{counts['total']}）")
        return
    for name in ("train", "val", "test"):
        if counts[name] <= 0:
            problems.append(f"{label}：{name} 为空（n_{name}={counts[name]}）")
        share = counts[name] / counts["total"]
        if abs(share - OPENPOLY_RATIOS[name]) > OPENPOLY_RATIO_TOLERANCE:
            problems.append(
                f"{label}：{name} 占 {share:.4f}，超出 {OPENPOLY_RATIOS[name]}±{OPENPOLY_RATIO_TOLERANCE}"
            )

    train = payload.get("train_index") or []
    val = payload.get("val_index") or []
    test = payload.get("test_index") or []
    total_parts = len(train) + len(val) + len(test)
    if total_parts != counts["total"]:
        problems.append(f"{label}：三段合计 {total_parts}，n_total 写 {counts['total']}")
    union = set(train) | set(val) | set(test)
    if len(union) != total_parts:
        problems.append(f"{label}：三段有重叠（并集 {len(union)} < 合计 {total_parts}）")
    if union and (min(union) != 0 or max(union) != counts["total"] - 1):
        problems.append(f"{label}：行号没有覆盖 0..{counts['total'] - 1}（实际 {min(union)}..{max(union)}）")

    bin_counts = payload.get("bin_counts") or []
    if len(bin_counts) != OPENPOLY_BINS:
        problems.append(f"{label}：bin_counts 有 {len(bin_counts)} 项，应为 {OPENPOLY_BINS} 箱")
    elif sum(int(value) for value in bin_counts) != counts["total"]:
        problems.append(
            f"{label}：每箱条数合计 {sum(int(v) for v in bin_counts)} ≠ n_total {counts['total']}"
        )

    source_row = payload.get("source_row") or []
    if len(source_row) != counts["total"]:
        problems.append(f"{label}：source_row 有 {len(source_row)} 项，应为 {counts['total']}")
    elif any(int(b) <= int(a) for a, b in zip(source_row, source_row[1:])):
        problems.append(f"{label}：source_row 不是严格递增，回溯到原始行会错位")

    if deep and config is not None:
        values = _openpoly_label_rows(config)[1]
        again = stratified_split(values, seed=seed)
        for name in ("train", "val", "test"):
            if [int(v) for v in again[name]] != [int(v) for v in payload.get(f"{name}_index") or []]:
                problems.append(f"{label}：用 seed {seed} 重算，{name} 的行号与文件不一致（文件可能被改过）")


def _verify_openpoly(
    configs: Mapping[str, Mapping[str, Any]], problems: list[str], deep: bool = False
) -> None:
    """校验 OpenPoly 的三条边界任务：规则说明在、R 份划分在、每一份都守规则、清单对得上。
    Verify the three OpenPoly boundary tasks: the rule card exists, R splits exist, each obeys the
    rule, and the manifests agree.

    中文：
        用途：这条校验要同时防两件事。① **退化回 D-22 之前**：有人把划分删了、或把状态改回
              `blocked_upstream` 却没人发现。② **悄悄换成自己切的**：目录里冒出
              `*_official.json`（声称"官方那份"），或某个种子下的文件被手工改了几个行号——
              后者只有 `deep=True` 的重算才抓得到。
        输入：configs —— 全部任务配置；problems —— 问题列表；deep —— 是否按种子重算。
        输出：无。数据来源：`splits/openpoly/**`、两份清单、发布 CSV。
        配置来源：三个 OpenPoly 任务的 `split_ref` 与 `split_status`。
        副作用：只读。异常：无（收集而非抛出）。

    English:
        Purpose: this guards two things at once. (1) A regression to the pre-D-22 state - splits
                 deleted or status quietly reverted. (2) A silent swap for a hand-made partition:
                 an `*_official.json` appearing in the directory, or a few indices hand-edited in
                 one seed's file - the latter is only caught by the `deep` recomputation.
        Args: configs; problems; deep. Returns: none.
        Side effects: read-only. Raises: none.
    """
    if not OPENPOLY_PROTOCOL_NOTE.is_file():
        problems.append("splits/openpoly/PROTOCOL.json 不存在（D-24 的规则说明必须有据可查）")
    else:
        note = json.loads(OPENPOLY_PROTOCOL_NOTE.read_text(encoding="utf-8"))
        if note.get("status") != "rule_replicated":
            problems.append(f"splits/openpoly 的状态是 {note.get('status')!r}，应为 'rule_replicated'")
        if not note.get("rule", {}).get("quotes"):
            problems.append("PROTOCOL.json 没有引论文原文（规则必须可追溯到期刊正文）")

    recorded = {row["artifact"]: row for row in _read_manifest_rows(OPENPOLY_MANIFEST)}
    for task_id, config in sorted(configs.items()):
        if config["benchmark"] != "openpoly":
            continue
        directory = REPO_ROOT / str(config["split_ref"])
        if not directory.is_dir():
            problems.append(f"{task_id}：划分目录不存在（{config['split_ref']}）")
            continue
        official = sorted(path.name for path in directory.glob("*_official*.json"))
        if official:
            problems.append(
                f"{task_id}：目录里出现了 {official}——OpenPoly 没有官方划分文件，不许冒充（D-24）"
            )
        for seed in OPENPOLY_SEEDS:
            path = directory / f"rule_replicated_seed{seed}.json"
            if not path.is_file():
                problems.append(f"{task_id}：缺少 seed {seed} 的划分（{path.name}）")
                continue
            payload = json.loads(path.read_text(encoding="utf-8"))
            _check_openpoly_split(payload, problems, task_id=task_id, seed=seed, deep=deep, config=config)
            artifact = path.relative_to(REPO_ROOT).as_posix()
            row = recorded.get(artifact)
            if row is None:
                problems.append(f"{task_id}：{artifact} 不在 openpoly_split_manifest.csv 里")
            elif row.get("sha256") != sha256_of(path):
                problems.append(f"{task_id}：{artifact} 的 sha256 与清单不符（文件被改过）")
            elif row.get("fold_role") != ("primary" if seed in OPENPOLY_PRIMARY_FOLDS else "reserve"):
                problems.append(f"{task_id}：{artifact} 的 fold_role 是 {row.get('fold_role')!r}")

def verify_splits(
    configs: Mapping[str, Mapping[str, Any]] | None = None, *, deep: bool = False
) -> list[str]:
    """把 A-5 的验收标准跑一遍，返回问题清单（空 = 全过）。
    Run the A-5 acceptance criteria and return the list of problems (empty means pass).

    中文：
        用途：`scripts/03_splits.py --verify` 与 `tests/test_splits.py` 共用的唯一实现，
              保证"人跑的"和"CI 跑的"是同一条检查。
        输入：configs —— 任务配置（默认现读 `configs/tasks/`）；deep —— 是否重算（见
              `_verify_polybench()`）。
        输出：问题字符串列表；空列表即通过。
        数据来源：`configs/tasks/*.yaml`、`splits/**`、`data/manifests/splits_manifest.csv`。
        配置来源：`split_status` 决定哪些任务"应当有划分"。
        副作用：只读。异常：无（问题收集而非抛出）。

    English:
        Purpose: the single implementation shared by `--verify` and pytest, so the human run and
                 the CI run are literally the same check.
        Args: configs - task configs (defaults to reading configs/tasks); deep - recompute as in
              `_verify_polybench()`.
        Returns: list of problem strings; empty means pass.
        Data source: configs/tasks, splits/, data/manifests/splits_manifest.csv.
        Config: `split_status` decides which tasks must have a split.
        Side effects: read-only. Raises: none (collects instead of raising).
    """
    # 用宽松加载：划分缺失时要"报成一条问题"，而不是在加载阶段抛异常——
    # 校验器的价值在于一次列全所有问题，抛第一个就等于让人反复重跑。
    # Lenient load: a missing split must surface as a problem line, not as a loader exception;
    # a verifier that raises on the first issue forces repeated re-runs.
    configs = configs or load_all_task_configs(CONFIGS_DIR, allow_missing_split=True)
    problems: list[str] = []
    for task_id in sorted(configs):
        config = configs[task_id]
        status = config["split_status"]
        if status == "present":
            if config["benchmark"] == "polybench":
                _verify_polybench(config, problems, deep)
            elif config["benchmark"] == "polybench26":
                _verify_polybench26(config, problems, deep)
        elif status == "pending":
            # 唯一允许还挂 pending 的是"整任务不可用"的那种（bandgap：无标签可取划分）。
            # 其它人挂 pending 就是漏做。
            # The only legitimate pending is an unavailable task (bandgap has no labels to split);
            # anything else is simply unfinished work.
            if config["status"] != "unavailable":
                problems.append(f"{task_id}：split_status 还是 pending（A-5 之后不应再有）")
        elif status == "deferred":
            directory = REPO_ROOT / str(config["split_ref"])
            if not directory.is_dir() or not any(directory.iterdir()):
                problems.append(f"{task_id}：deferred 但没有说明文件目录 {config['split_ref']}")
        elif status == "blocked_upstream":
            if config["benchmark"] != "openpoly":
                problems.append(f"{task_id}：只有 OpenPoly 允许 blocked_upstream，{config['benchmark']} 不行")
        elif status == "rule_replicated":
            # 逐份检查在 `_verify_openpoly()` 里统一做（目录、种子、比例、互斥、清单、重算）。
            # The per-file checks all live in `_verify_openpoly()`.
            if config["benchmark"] != "openpoly":
                problems.append(f"{task_id}：只有 OpenPoly 允许 rule_replicated（D-24）")
        else:
            problems.append(f"{task_id}：未知 split_status={status}")
    _verify_openpoly(configs, problems, deep)
    return problems


# ---- 6. 入口审计：扫"自己切数据"的调用 / Entry audit: scan for self-cut splits ----

# 这些名字出现在 `src/` 的**调用**里就等于自己切数据（Proposal §7.5 头号失败模式）。
# 只匹配调用、不匹配注释或 docstring——文档里写"不许用 train_test_split"是正常的，
# 真调用它才是问题。D 的入口审计会往 `src/` 里注入一次真调用，看这条扫不扫得到。
# These names appearing as *calls* in `src/` mean a self-cut split (Proposal §7.5's number-one
# failure mode). Calls only, never comments or docstrings: prose saying "do not call
# train_test_split" is fine, calling it is not. D's entry audit injects a real call and expects
# this scan to find it.
FORBIDDEN_SPLIT_CALLS = ("train_test_split", "KFold", "StratifiedKFold", "GroupKFold", "ShuffleSplit")


def find_self_cut_calls(root: Path | None = None) -> list[str]:
    """扫 `src/` 里有没有"自己切数据"的调用，返回 `文件:行:调用名` 列表。
    Scan `src/` for self-cut split calls and return `file:line:name` strings.

    中文：
        用途：把"评估只能读官方划分"这条红线做成可执行扫描。用 AST 而不是字符串匹配，
              因为①注释与 docstring 里出现这些词是合法的（甚至是被鼓励的）；②字符串匹配会
              漏掉 `from sklearn.model_selection import train_test_split as tts` 这种别名调用，
              也对 `KFold(` 与 `KFold (` 反应不一。
        输入：root —— 扫描根，默认 `src/`。
        输出：问题字符串列表；空列表 = 干净。
        数据来源：`src/**/*.py` 的源码。
        配置来源：`FORBIDDEN_SPLIT_CALLS`。
        副作用：只读。异常：SyntaxError —— 被扫文件不是合法 Python（应当直接暴露，不吞掉）。

    English:
        Purpose: make "evaluation reads the official split only" an executable scan. AST rather
                 than string matching, because (1) those words legitimately appear in comments
                 and docstrings (indeed they should); (2) string matching misses aliased imports
                 such as `from sklearn.model_selection import train_test_split as tts`.
        Args: root - scan root, defaults to `src/`.
        Returns: problem strings; empty means clean.
        Data source: sources under `src/**/*.py`.
        Config: `FORBIDDEN_SPLIT_CALLS`.
        Side effects: read-only. Raises: SyntaxError for a file that is not valid Python (must
                surface, not be swallowed).
    """
    import ast

    scan_root = root or (REPO_ROOT / "src")
    offenders: list[str] = []
    for path in sorted(scan_root.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            function = node.func
            name = (
                function.id
                if isinstance(function, ast.Name)
                else function.attr
                if isinstance(function, ast.Attribute)
                else None
            )
            if name in FORBIDDEN_SPLIT_CALLS:
                # 扫描根可能是临时目录（测试会注入一个假文件），此时没有仓库相对路径可打。
                # The scan root may be a temp directory (a test injects a fake file), in which
                # case there is no repo-relative path to print.
                try:
                    shown = path.relative_to(REPO_ROOT)
                except ValueError:
                    shown = path
                offenders.append(f"{shown}:{node.lineno}:{name}")
    return offenders
