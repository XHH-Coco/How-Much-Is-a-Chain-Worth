"""C-b-1：PolyBench26 的数据量扫描（scaling）——三层寻址、两份特征 pack、档位前缀与批量拟合。
C-b-1: PolyBench26's data-size scan (scaling) - three-level addressing, two feature packs, tier
prefixes and the batch fits.

中文：
    用途：回答"数据变多，误差会不会变小、变多快"（H2 的判据 κ）。三件事：
          ① **三层寻址**：`pb26_scaling_cp` 是**一个任务两个数据集**（交替 Cp / 均聚 Cp），
             每个数据集各有 `seed_42` 五折与自己的 `train_sizes`。仓库原来的
             `loaders.load_split('pb26_scaling_cp')` 会把**目录**当文件打开（`IsADirectoryError`），
             所以这里把"任务 → 数据集 → 档位"这一层补上，**不改**任何已冻结的划分文件；
          ② **两份特征 pack**：`data/processed/features/pb26_scaling_cp_<dataset>.npz`，每次测量
             一行（row_id + Cp + 规范 PSMILES + 205 描述符 + 2048 位指纹），造完先核
             `manifest.json` 的 `canonical_sha256` 与官方逐位相同（D-21）；
          ③ **档位 = `train_pool[:N]` 的前缀**（官方 `create_scaling_splits.py` 的定义），
             **不是**"自己按行数截"、也**不是**"随机抽 n 行"——后两者是分工_C-b 红线 1。
    数据来源：`configs/tasks/pb26_scaling_cp.yaml` 的 `datasets` 块（A-4 冻结，只读）、
              `data/raw/polybench26/Datasets/PSMILES/MD_5000/Cp/*.csv`（D-38 入库，8 份）、
              `splits/polybench26/pb26_scaling_cp/<dataset>/seed_42/`（A-5 快照，只读）。
    复用而非重写：描述符与指纹走 `chem.descriptors.compute()`（A-6 的唯一计算口），
              pack 的落盘格式走 `chem.features.save_pack()`（B-1 的格式），
              记录与预测的落盘走 `arms.graph.write_run_record()/write_predictions()`。
    配置：`configs/experiments/scaling.yaml`（档位、折、臂、`expected_fits` 的唯一出处）。
    对外：`dataset_ids()`、`build_pack()`、`load_pack()`、`load_dataset_split()`、
          `build_task_data()`、`run_scaling()`、`write_table()`。
    被谁用：`scripts/04_run_cpu.py` 的 `scaling` 子命令、`tests/test_scaling_scan.py`。

English:
    Purpose: answer "does more data lower the error, and how fast" (H2's kappa). Three jobs:
    (1) **three-level addressing** - `pb26_scaling_cp` is one task with **two datasets**, each with
    its own `seed_42` five folds and its own `train_sizes`; the stock
    `loaders.load_split('pb26_scaling_cp')` opens a *directory* as a file, so this module adds the
    task -> dataset -> tier level without touching any frozen split file;
    (2) **two feature packs** under `data/processed/features/`, one row per measurement, with the
    official `canonical_sha256` verified against `manifest.json` (D-21);
    (3) **a tier is the prefix `train_pool[:N]`** exactly as the official
    `create_scaling_splits.py` builds it - truncating by row count or sampling n rows is red line 1.
    Reuse over rewrite: descriptors and fingerprints come from `chem.descriptors.compute()` (A-6's
    single computation point), the pack format from `chem.features.save_pack()` (B-1's format), and
    records/predictions from `arms.graph`'s writers.
    Data: the frozen `datasets` block of `configs/tasks/pb26_scaling_cp.yaml`, the eight committed
    Cp CSVs (D-38) and the read-only A-5 split snapshots.
    Config: `configs/experiments/scaling.yaml` - the only place tiers, folds, arms and
    `expected_fits` are defined.
"""

from __future__ import annotations

import csv
import hashlib
import json
import logging
import math
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from chainworth.arms import conventional as C
from chainworth.arms import graph as G
from chainworth.io import loaders as L

LOGGER = logging.getLogger("chainworth.experiments.scaling")

REPO_ROOT = Path(__file__).resolve().parents[3]
TASK_ID = "pb26_scaling_cp"
RAW_ROOT = REPO_ROOT / "data" / "raw" / "polybench26" / "Datasets"
FEATURES_DIR = REPO_ROOT / "data" / "processed" / "features"
MANIFEST_PATH = REPO_ROOT / "data" / "manifests" / "scaling_features_manifest.csv"
UNRESOLVED_PATH = REPO_ROOT / "data" / "manifests" / "scaling_feature_unresolved.csv"
TABLE_PATH = REPO_ROOT / "results" / "aggregate" / "scaling_cpu.csv"

MANIFEST_COLUMNS = (
    "task_id",
    "dataset_id",
    "n_rows",
    "n_label_rows",
    "n_unique_structures",
    "n_unparsable_smiles",
    "n_descriptors",
    "n_fingerprint_bits",
    "n_nan_cells",
    "canonical_sha256",
    "canonical_source",
    "pack_sha256",
    "built_at",
)
UNRESOLVED_COLUMNS = ("task_id", "dataset_id", "row_id", "psmiles_raw", "reason")
# 表列（分工_C-b C-b-1 点名的那些都在，另加四个核对用的列）。
# Table columns: the ones 分工_C-b C-b-1 names, plus four columns needed to check the claim.
TABLE_COLUMNS = (
    "task",
    "dataset",
    "tier",
    "n_train",
    "n_train_pool",
    "n_level_saturated",
    "n_selection_rule",
    "subset_sha256",
    "fold",
    "arm",
    "learner",
    "seed",
    "r2",
    "mae",
    "rmse",
    "wall_clock_s",
    "run_id",
    "note",
)


# ---- 1. 三层寻址：任务 → 数据集 → 档位 / Three-level addressing: task -> dataset -> tier ----


def task_config() -> dict[str, Any]:
    """读冻结的任务配置（`configs/tasks/pb26_scaling_cp.yaml`）。
    Read the frozen task config.

    中文：用途：数据集清单、行数、档位、官方索引路径都在这份文件里，**不在这里重复写常量**——
          写死一份就等于有了第二个口径。输入：无。输出：dict。副作用：只读。异常：TaskConfigError。
    English: the dataset list, row counts, tiers and official index paths all live in that file;
    duplicating them as constants would create a second convention. Returns the parsed config.
    """
    return L.load_task_config(TASK_ID)


def dataset_ids(task_id: str = TASK_ID) -> list[str]:
    """列出这个任务的数据集 id（顺序 = 配置里的书写顺序）。
    List the task's dataset ids, in config order.

    中文：输入：task_id。输出：字符串列表。数据来源：任务配置的 `datasets`。副作用：只读。
    English: Args: task_id. Returns: the dataset ids. Data source: the config's `datasets` block.
    """
    config = task_config() if task_id == TASK_ID else L.load_task_config(task_id)
    entries = config.get("datasets") or []
    if not entries:
        raise L.AssemblyError(f"{task_id} 的配置里没有 `datasets`：它不是双数据集任务")
    return [str(entry["dataset_id"]) for entry in entries]


def dataset_entry(dataset_id: str, task_id: str = TASK_ID) -> dict[str, Any]:
    """取一个数据集的配置块（`canonical_source` / `train_sizes` / `n_rows` 都在里面）。
    Return one dataset's config block (canonical_source / train_sizes / n_rows).

    中文：输入：dataset_id。输出：dict。异常：AssemblyError —— 没有这个数据集。
    English: Args: dataset_id. Returns: the block. Raises AssemblyError when unknown.
    """
    config = task_config() if task_id == TASK_ID else L.load_task_config(task_id)
    for entry in config.get("datasets") or []:
        if str(entry.get("dataset_id")) == dataset_id:
            return dict(entry)
    raise L.AssemblyError(
        f"{task_id} 没有数据集 {dataset_id!r}：配置里的是 {[e.get('dataset_id') for e in config.get('datasets') or []]}"
    )


def raw_csv_path(dataset_id: str, task_id: str = TASK_ID) -> Path:
    """数据集的 canonical 原始表（`PSMILES/MD_5000/Cp/*.csv`）。
    The dataset's canonical raw table.

    中文：输入：dataset_id。输出：Path。数据来源：`data/raw/polybench26/Datasets/<canonical_source>`。
    English: Args: dataset_id. Returns: the path under data/raw/polybench26/Datasets.
    """
    entry = dataset_entry(dataset_id, task_id)
    path = RAW_ROOT / str(entry["canonical_source"])
    if not path.is_file():
        raise L.AssemblyError(
            f"缺原始表 {path.relative_to(REPO_ROOT).as_posix()}：D-38 的入口包应含这 8 份 Cp 表"
        )
    return path


def split_dir(dataset_id: str, task_id: str = TASK_ID) -> Path:
    """数据集的划分目录（`splits/polybench26/pb26_scaling_cp/<dataset>/seed_42/`）。
    The dataset's split directory.

    中文：输入：dataset_id。输出：Path。数据来源：任务配置的 `split_ref` + 数据集 id。
    English: built from the task config's `split_ref` plus the dataset id.
    """
    config = task_config() if task_id == TASK_ID else L.load_task_config(task_id)
    directory = REPO_ROOT / str(config["split_ref"]) / dataset_id / "seed_42"
    if not directory.is_dir():
        raise L.AssemblyError(
            f"缺划分目录 {directory.relative_to(REPO_ROOT).as_posix()}：先跑 scripts/03_splits.py（A-5）"
        )
    return directory


def split_manifest(dataset_id: str, task_id: str = TASK_ID) -> dict[str, Any]:
    """读数据集的 `manifest.json`（档位、折、行数、canonical 哈希的唯一来源）。
    Read the dataset's manifest.json - the single source of tiers, folds, row counts and hash.

    中文：输入：dataset_id。输出：dict。数据来源：`split_dir()/manifest.json`。副作用：只读。
    English: Args: dataset_id. Returns: the parsed manifest. Data source: split_dir()/manifest.json.
    """
    return json.loads((split_dir(dataset_id, task_id) / "manifest.json").read_text(encoding="utf-8"))


def tiers(dataset_id: str, task_id: str = TASK_ID) -> list[str]:
    """档位清单 = 官方 `train_sizes` 加上 `all`（全池）。顺序照官方文件。
    The tier list: the official train_sizes plus `all` (the whole pool), in official order.

    中文：
        用途：**档位来自官方文件，不在这里挑**。均聚 Cp 只有 300 / 1000 两档（官方 README 说
              明），硬凑五档等于造出官方没有的档位，κ 曲线会失真——缺档是事实，不是失败。
        输入：dataset_id。输出：字符串列表（如 `["300","1000","3000","10000","all"]`）。
        数据来源：`manifest.json` 的 `train_sizes`。副作用：只读。
    English:
        Purpose: the tiers come from the official file, never from a choice made here. The
        homopolymer table has only 300 / 1000; padding it to five tiers would invent tiers the
        official protocol does not have and distort kappa. Returns e.g. 300/1000/3000/10000/all.
    """
    manifest = split_manifest(dataset_id, task_id)
    sizes = [int(value) for value in manifest.get("train_sizes") or []]
    if not sizes:
        raise L.AssemblyError(f"{dataset_id} 的 manifest.json 没有 `train_sizes`：档位无处可取")
    return [str(size) for size in sizes] + ["all"]


# ---- 2. 两份特征 pack / The two feature packs ----


def pack_name(dataset_id: str, task_id: str = TASK_ID) -> str:
    """pack 的"任务名"：`<task>_<dataset>`（`data/processed/features/` 下的文件名之一）。
    The pack's task-like name: `<task>_<dataset>`.

    中文：用途：`chem.features.save_pack()`/`load_pack()` 都按"任务名"寻址，所以 pack 用这个
          合成的名字；而 run 记录里的 `task` 仍是 `pb26_scaling_cp`（配置里的 task_id），
          数据集另记在 `dataset_id` 字段——两者不可混。
    English: save_pack/load_pack address packs by a task-like name, hence this composite; the run
    record's `task` stays `pb26_scaling_cp` (the config's task_id) and the dataset is its own field.
    """
    return f"{task_id}_{dataset_id}"


def _sha256_of_file(path: Path) -> str:
    """文件字节的 sha256（分块读）。／ sha256 of a file's bytes, read in chunks."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _now_iso() -> str:
    """带时区的 ISO 时间。／ ISO timestamp with a timezone."""
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def _canonicalise(values: Sequence[str]) -> list[str]:
    """把原始 SMILES 规范化（失败的位置给空串）。／ Canonicalise raw SMILES; failures become "".

    中文：用途：缓存的键与 pack 的「结构键」都必须是规范式，所以先规范化一次。
          输入：values。输出：等长列表。副作用：关掉 RDKit 日志。异常：无。
    English: both the cache key and the pack's structure key must be canonical, so canonicalise once.
    """
    from rdkit import Chem, RDLogger

    RDLogger.DisableLog("rdApp.*")
    out: list[str] = []
    for text in values:
        molecule = Chem.MolFromSmiles(str(text))
        out.append("" if molecule is None else Chem.MolToSmiles(molecule))
    return out


def _compute_one(value: str) -> tuple[str, dict[str, Any] | None]:
    """工作进程的入口：算一个结构的表征（必须能在子进程里导入，故放在模块层）。
    Worker entry point: compute one structure (module level so a child process can import it).
    """
    from chainworth.chem import descriptors as D

    return value, D.compute(value)


def _features_from_cache(
    keys: Sequence[str], *, workers: int, log_prefix: str = ""
) -> dict[str, dict[str, Any] | None]:
    """先查 A-6 的描述符缓存，未命中的并行算，算完写回缓存；返回规范式 → 记录。
    Look A-6's descriptor cache up first, compute misses in parallel, write them back.

    中文：
        用途：29,602 个结构串行要半小时，所以这里并行；而并行进程之间不共享内存，于是用
              **A-6 的 sqlite 缓存**（`descriptors.DescriptorCache`）当持久层——provenance 不符
              会直接抛，不会把两套描述符混在一起。缓存也是"跑一半断了不用从头来"的凭据。
        输入：keys —— 规范 PSMILES（可含重复与空串）；workers —— 进程数；log_prefix —— 日志前缀。
        输出：dict（键 = 规范式，值 = `descriptors.compute()` 形状的记录；解析失败为 None）。
        数据来源：`data/processed/structures/.descriptor_cache.sqlite3`。副作用：写缓存。
        异常：ValueError —— 缓存 provenance 与当前 RDKit 不一致（按消息删缓存重跑）。
    English:
        Serial computation of 29,602 structures takes half an hour, hence the process pool; because
        processes share no memory, A-6's sqlite cache (`descriptors.DescriptorCache`) is the
        persistence layer, and its provenance check refuses to mix two descriptor spaces. The cache
        is also what makes a half-finished run resumable.
        Returns: canonical form -> a `descriptors.compute()`-shaped record (None when unparsable).
    """
    from chainworth.chem import descriptors as D

    unique = [key for key in dict.fromkeys(keys) if key]
    cache_path = REPO_ROOT / "data" / "processed" / "structures" / ".descriptor_cache.sqlite3"
    with D.DescriptorCache(cache_path) as cache:
        stored = cache.stored_keys()
        hits = [key for key in unique if key in stored]
        missing = [key for key in unique if key not in stored]
        LOGGER.info(
            "%s：缓存命中 %d 个结构，待算 %d 个（缓存 %s）",
            log_prefix or "scaling",
            len(hits),
            len(missing),
            cache_path.name,
        )
        out: dict[str, dict[str, Any] | None] = {key: None for key in unique}
        cached_records, _stats = cache.records(hits) if hits else ([], {})
        for key, record in zip(hits, cached_records):
            out[key] = record
        if missing:
            fresh = _parallel_compute(missing, workers=workers, log_prefix=log_prefix)
            rows: list[tuple[str, str, bytes]] = []
            for key, payload in fresh.items():
                if payload is None:
                    continue
                stored_payload = {
                    column: (None if not math.isfinite(number) else number)
                    for column, number in payload["descriptors"].items()
                }
                out[key] = {
                    "psmiles_canonical": payload["psmiles_canonical"],
                    "descriptors": stored_payload,
                    "fingerprint": payload["fingerprint"],
                }
                rows.append(
                    (key, json.dumps(stored_payload, sort_keys=True), payload["fingerprint"])
                )
            if rows:
                cache.connection.executemany(
                    "INSERT OR REPLACE INTO structures(psmiles, descriptors, fingerprint) "
                    "VALUES (?, ?, ?)",
                    rows,
                )
                cache.connection.commit()
    return out


def _parallel_compute(
    values: Sequence[str], *, workers: int, log_prefix: str = ""
) -> dict[str, dict[str, Any] | None]:
    """用进程池算一批结构的表征。／ Compute a batch of structures with a process pool.

    中文：用途：RDKit 不在线程间并行，所以只能开进程；`imap_unordered` + 分块让进度可打印。
          输入：values；workers（≤ 1 时退回串行，便于调试）。输出：dict。
          副作用：无（缓存由调用方写）。异常：无（解析失败记 None）。
    English: RDKit does not parallelise across threads, so this uses processes; unordered imap with
    chunks keeps progress printable. workers <= 1 falls back to serial for debugging.
    """
    import multiprocessing as mp

    from chainworth.chem import descriptors as D

    clock = time.perf_counter()
    if workers <= 1:
        out = {value: D.compute(value) for value in values}
        LOGGER.info("%s：串行算完 %d 个结构（%.0f 秒）", log_prefix, len(values), time.perf_counter() - clock)
        return out
    out: dict[str, dict[str, Any] | None] = {}
    with mp.Pool(processes=workers) as pool:
        for index, (value, payload) in enumerate(
            pool.imap_unordered(_compute_one, values, chunksize=25), start=1
        ):
            out[value] = payload
            if index % 2000 == 0:
                LOGGER.info(
                    "%s：%d/%d 个结构已算（%.0f 秒）",
                    log_prefix,
                    index,
                    len(values),
                    time.perf_counter() - clock,
                )
    LOGGER.info("%s：并行算完 %d 个结构（%.0f 秒）", log_prefix, len(values), time.perf_counter() - clock)
    return out


def read_manifest_rows(path: Path | None = None) -> list[dict[str, str]]:
    """读 pack 清单（不存在就返回空表）。／ Read the pack manifest; an absent file is empty."""
    target = Path(path) if path is not None else MANIFEST_PATH
    if not target.is_file():
        return []
    with target.open("r", encoding="utf-8", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def write_manifest_rows(rows: Sequence[Mapping[str, Any]], path: Path | None = None) -> Path:
    """按 (task_id, dataset_id) 合并写回 pack 清单（重跑不留两行）。
    Merge rows into the pack manifest keyed by (task_id, dataset_id); a re-run never duplicates.
    """
    target = Path(path) if path is not None else MANIFEST_PATH
    merged: dict[tuple[str, str], dict[str, Any]] = {}
    for row in read_manifest_rows(target):
        merged[(str(row.get("task_id", "")), str(row.get("dataset_id", "")))] = dict(row)
    for row in rows:
        merged[(str(row["task_id"]), str(row["dataset_id"]))] = {
            column: row.get(column, "") for column in MANIFEST_COLUMNS
        }
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8", newline="\n") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(MANIFEST_COLUMNS), lineterminator="\n")
        writer.writeheader()
        for key in sorted(merged):
            writer.writerow(merged[key])
    return target


def manifest_pack_sha256(dataset_id: str, task_id: str = TASK_ID) -> str:
    """从 pack 清单里取这个数据集的 `pack_sha256`；取不到就报错，不猜。
    Read the dataset's `pack_sha256` from the manifest; a missing row raises instead of guessing.
    """
    for row in read_manifest_rows():
        if row.get("task_id") == task_id and row.get("dataset_id") == dataset_id:
            return str(row.get("pack_sha256", ""))
    raise L.AssemblyError(
        f"pack 清单里没有 {task_id}/{dataset_id}：先跑 `python scripts/04_run_cpu.py scaling --build`"
    )


def build_pack(
    dataset_id: str, *, task_id: str = TASK_ID, force: bool = False, workers: int = 0
) -> dict[str, Any]:
    """给一个数据集造特征 pack（不存在或 `force` 时才算），返回摘要。
    Build one dataset's feature pack (only when absent or forced) and return a summary.

    中文：
        用途：`data/processed/features/<task>_<dataset>.npz`。步骤与理由：
              ① 读原始表（**发布顺序**，行号就是划分索引的坐标系，绝不重排）；
              ② 核 `canonical_sha256`：与官方 `manifest.json` 逐位相同才继续。不同就停——
                 那说明手上的表不是官方那一份，"同划分"的前提当场失效（D-21）；
              ③ 每个唯一结构**只算一次** 205 描述符 + 2048 位指纹（复用
                 `chem.descriptors.compute()`，A-6 的唯一计算口）；
              ④ 解析失败的行走 **NaN 特征但保留行**：删行会让后面全部前移，而划分索引不会跟着动。
        输入：dataset_id；task_id；force —— 已存在也重造。
        输出：摘要 dict（行数、唯一结构数、NaN 单元数、pack 哈希……）。
        数据来源：原始 Cp 表 + `splits/.../manifest.json`（只读）。
        副作用：写 npz 与 `data/manifests/scaling_features_manifest.csv`。
        异常：AssemblyError —— 缺原始表、哈希不符、结构全部解析失败。
    English:
        Writes `data/processed/features/<task>_<dataset>.npz`: read the raw table in published order
        (the row number is the split's coordinate system - never re-sorted), verify the official
        `canonical_sha256` and stop on a mismatch (the table then is not the official one and the
        same-split premise is void, D-21), compute 205 descriptors plus 2048 fingerprint bits once
        per unique structure through `chem.descriptors.compute()`, and keep unparsable rows with NaN
        features instead of dropping them (dropping shifts every later row while the split does not).
    """
    from chainworth.chem import descriptors as D
    from chainworth.chem import features as F

    entry = dataset_entry(dataset_id, task_id)
    manifest = split_manifest(dataset_id, task_id)
    raw_path = raw_csv_path(dataset_id, task_id)
    name = pack_name(dataset_id, task_id)
    target = FEATURES_DIR / f"{name}.npz"
    if workers <= 0:
        # 默认用"核数 − 1"，上限 8：再多人也多不过内存带宽，而每个进程要装一份 RDKit。
        # Default is (cores - 1) capped at 8: beyond that it is memory-bandwidth bound, and every
        # process loads its own copy of RDKit.
        workers = max(1, min(8, (os.cpu_count() or 2) - 1))

    # ① 原始表按发布顺序读入；`canonical_sha256` 逐位核对（D-21）。
    # Read the raw table in published order and verify the official canonical hash (D-21).
    canonical_sha = _sha256_of_file(raw_path)
    expected = str(manifest.get("canonical_sha256") or "")
    if expected and canonical_sha != expected:
        raise L.AssemblyError(
            f"{raw_path.name} 的 sha256={canonical_sha[:12]} 与官方 manifest 的 {expected[:12]} 不符："
            "手上的表不是官方那一份，同划分的前提不成立（D-21），停下来问数据负责人（A）"
        )
    if not force and target.is_file():
        recorded = manifest_pack_sha256(dataset_id, task_id)
        if recorded and recorded == _sha256_of_file(target):
            LOGGER.info("%s：pack 已在（哈希与清单一致），跳过重造", name)
            # 缓存命中也要把**清单里那一行**原样返回：调用方要打印行数/唯一结构数，
            # 返回一个只有三个键的字典会让 `info["n_rows"]` 当场 KeyError。
            # A cache hit still returns the **manifest row** verbatim: callers print row and
            # structure counts, and a three-key dict would raise KeyError on `info["n_rows"]`.
            row = next(
                (
                    item
                    for item in read_manifest_rows()
                    if item.get("task_id") == task_id and item.get("dataset_id") == dataset_id
                ),
                {},
            )
            return {
                **row,
                "task_id": task_id,
                "dataset_id": dataset_id,
                "pack_path": target,
                "pack_sha256": recorded,
                "canonical_sha256": canonical_sha,
                "rebuilt": False,
            }

    import pandas as pd

    frame = pd.read_csv(raw_path)
    if len(frame.columns) != 2:
        raise L.AssemblyError(f"{raw_path.name} 有 {len(frame.columns)} 列：期望 2 列（smiles, Cp）")
    smiles_column, label_column = list(frame.columns)
    psmiles_raw = [str(value) for value in frame[smiles_column]]
    labels = pd.to_numeric(frame[label_column], errors="coerce").to_numpy(dtype=np.float64)

    columns = D.descriptor_columns()
    bits_columns = F.fingerprint_columns()
    n_rows = len(psmiles_raw)
    desc = np.full((n_rows, len(columns)), np.nan, dtype=np.float64)
    fp = np.zeros((n_rows, len(bits_columns)), dtype=np.uint8)
    canonical = ["" for _ in range(n_rows)]
    unresolved: list[dict[str, Any]] = []

    # ③ 唯一结构只算一次，且**能复用就复用**：先规范化（约 0.45 ms/个），再查 A-6 的
    #    描述符缓存（`data/processed/structures/.descriptor_cache.sqlite3`，键是规范式、
    #    provenance 不符会直接抛），只有未命中的才真算——并行算，因为 Cp 的 29,602 行是
    #    29,602 个**不同**结构（交替 27,802 / 均聚 1,800，实测零重复），单个约 58 ms，
    #    串行要半小时。
    # Compute each unique structure once, reusing A-6's descriptor cache when possible:
    # canonicalise first (~0.45 ms each), look the canonical form up in the cache (provenance
    # checked), and only compute the misses - in parallel, because the Cp rows are 29,602
    # *distinct* structures (zero duplicates measured) at ~58 ms each, i.e. half an hour serially.
    clock = time.perf_counter()
    unique_raw = list(dict.fromkeys(psmiles_raw))
    keys = _canonicalise(unique_raw)
    LOGGER.info(
        "%s：%d 行 / %d 个不同结构，规范化用 %.0f 秒",
        name,
        n_rows,
        len(unique_raw),
        time.perf_counter() - clock,
    )
    records = _features_from_cache(keys, workers=workers, log_prefix=name)
    features: dict[str, dict[str, Any] | None] = {
        raw: records.get(key) if key else None for raw, key in zip(unique_raw, keys)
    }

    for position, raw in enumerate(psmiles_raw):
        payload = features.get(raw)
        if payload is None or payload.get("psmiles_canonical") is None:
            unresolved.append(
                {
                    "task_id": task_id,
                    "dataset_id": dataset_id,
                    "row_id": position,
                    "psmiles_raw": raw,
                    "reason": "unparsable_smiles",
                }
            )
            continue
        canonical[position] = str(payload["psmiles_canonical"])
        values = payload["descriptors"]
        desc[position] = [
            math.nan if values.get(column) is None else float(values[column]) for column in columns
        ]
        fp[position] = D.fingerprint_array(payload["fingerprint"])

    label_ok = np.isfinite(labels)
    if not label_ok.any():
        raise L.AssemblyError(f"{dataset_id} 的标签全部非有限值：pack 没有意义")

    summary: dict[str, Any] = {
        "task_id": name,
        "benchmark": "polybench26",
        "protocol": "official_scaling_indices_5fold_seed42",
        "split_ref": f"splits/polybench26/{task_id}/{dataset_id}/seed_42/manifest.json",
        "split_sha256": _sha256_of_file(split_dir(dataset_id, task_id) / "manifest.json"),
        "row_order_basis": (
            f"{raw_path.relative_to(REPO_ROOT).as_posix()}#row_position（官方 canonical 表的发布顺序，"
            "与 scaling_indices 的 fold_XX.npz 同一坐标系）"
        ),
        "row_id": np.arange(n_rows, dtype=np.int64),
        "y": labels,
        "psmiles": np.asarray(canonical, dtype=str),
        "desc": desc,
        "fp": fp,
        "desc_columns": columns,
        "fp_columns": bits_columns,
    }
    pack_path = F.save_pack(summary)
    pack_sha = _sha256_of_file(pack_path)

    row = {
        "task_id": task_id,
        "dataset_id": dataset_id,
        "n_rows": n_rows,
        "n_label_rows": int(label_ok.sum()),
        "n_unique_structures": len({value for value in canonical if value}),
        "n_unparsable_smiles": len(unresolved),
        "n_descriptors": len(columns),
        "n_fingerprint_bits": len(bits_columns),
        "n_nan_cells": int(np.isnan(desc).sum()),
        "canonical_sha256": canonical_sha,
        "canonical_source": str(entry["canonical_source"]),
        "pack_sha256": pack_sha,
        "built_at": _now_iso(),
    }
    write_manifest_rows([row])
    if unresolved:
        UNRESOLVED_PATH.parent.mkdir(parents=True, exist_ok=True)
        with UNRESOLVED_PATH.open("w", encoding="utf-8", newline="\n") as handle:
            writer = csv.DictWriter(
                handle, fieldnames=list(UNRESOLVED_COLUMNS), lineterminator="\n"
            )
            writer.writeheader()
            for item in unresolved:
                writer.writerow(item)
        LOGGER.warning("%s：%d 行的结构解析失败（特征留 NaN，行未删除）", name, len(unresolved))
    LOGGER.info(
        "%s：%d 行 × %d 列（%d 描述符 + %d 位）；唯一结构 %d；NaN 单元 %d；pack=%s…",
        name,
        n_rows,
        len(columns) + len(bits_columns),
        len(columns),
        len(bits_columns),
        row["n_unique_structures"],
        row["n_nan_cells"],
        pack_sha[:12],
    )
    return {**row, "pack_path": pack_path, "rebuilt": True}


def load_pack(dataset_id: str, *, task_id: str = TASK_ID) -> dict[str, Any]:
    """读回一个数据集的 pack（核哈希、核行号连续），返回 `chem.features.load_pack()` 的结构。
    Load one dataset's pack after checking its hash and contiguous row numbers.

    中文：输入：dataset_id。输出：dict（`row_id`/`y`/`psmiles`/`desc`/`fp`/`meta`…）。
          数据来源：`data/processed/features/<task>_<dataset>.npz`。副作用：只读。
          异常：ArmError —— 哈希与清单不符（文件被改过或清单没更新）。
    English: Args: dataset_id. Returns the same dict shape as features.load_pack(). Raises ArmError
    when the file's digest disagrees with the manifest - the file was edited or the manifest is stale.
    """
    from chainworth.chem import features as F

    path = FEATURES_DIR / f"{pack_name(dataset_id, task_id)}.npz"
    if not path.is_file():
        raise G.ArmError(
            f"缺 pack {path.relative_to(REPO_ROOT).as_posix()}：先跑 "
            "`python scripts/04_run_cpu.py scaling --build`"
        )
    expected = manifest_pack_sha256(dataset_id, task_id)
    actual = _sha256_of_file(path)
    if actual != expected:
        raise G.ArmError(
            f"{path.name} 的 sha256={actual[:12]} 与清单 {expected[:12]} 不符：文件被改过或清单过期"
        )
    return F.load_pack(pack_name(dataset_id, task_id))


# ---- 3. 划分与任务数据 / Splits and task data ----


def load_dataset_split(dataset_id: str, *, task_id: str = TASK_ID) -> L.SplitSpec:
    """把一个数据集的 `seed_42` 五折读成 `SplitSpec`（含 `train_pool`）。
    Read one dataset's seed_42 five folds into a SplitSpec, `train_pool` included.

    中文：
        用途：补上"任务 → 数据集"这一层。折名沿用官方的 `fold_00`…`fold_04`；`train_index` 就
              **是**官方 `train_pool`（档位是它的前缀），`train_pool` 也一并存进 `FoldSpec`。
              `n_total` 取 `manifest.json` 的 `n_rows`，这样 `check_split_matches_rows()` 能核
              "划分与行表配套"——行表就是原始表的行数。
        输入：dataset_id；task_id。输出：`SplitSpec`。
        数据来源：`splits/polybench26/pb26_scaling_cp/<dataset>/seed_42/`（只读）。
        副作用：只读。异常：AssemblyError —— 缺目录或某折 npz 缺键。
    English:
        Adds the task -> dataset level. Fold names stay the official fold_00..fold_04;
        `train_index` **is** the official `train_pool` (a tier is its prefix) and `train_pool` keeps
        its own copy, so `check_split_matches_rows()` can confirm the split fits the row table.
        Data source: the read-only A-5 snapshot; raises when the directory or a fold key is missing.
    """
    directory = split_dir(dataset_id, task_id)
    manifest_path = directory / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    folds: list[L.FoldSpec] = []
    for entry in manifest.get("folds") or []:
        fold_path = directory / str(entry["indices_file"])
        if not fold_path.is_file():
            raise L.AssemblyError(f"缺折文件 {fold_path.name}：A-5 的快照不完整")
        with np.load(fold_path) as payload:
            missing = {"train_pool", "val", "test"} - set(payload.files)
            if missing:
                raise L.AssemblyError(f"{fold_path.name} 缺数组 {sorted(missing)}：格式不对，别猜着读")
            train_pool = payload["train_pool"].astype(np.int64)
            folds.append(
                L.FoldSpec(
                    # 折名带数据集前缀：这个任务有**两个**数据集，裸的 `fold_00` 会让两条记录
                    # 拼出同一个 `run_id`（schema 的三段式没有第四个位置），而 `record_problems()`
                    # 会按 task/fold/seed 重算 run_id 并对不上——那一步是硬校验，绕不过去。
                    # Prefixed fold names: the task has two datasets, and a bare `fold_00` would let
                    # two records collide on one run_id (the schema has no fourth segment).
                    # `record_problems()` recomputes the id from task/fold/seed and refuses a mismatch.
                    name=f"{dataset_id}_fold_{int(entry['fold']):02d}",
                    train_index=train_pool,
                    val_index=payload["val"].astype(np.int64),
                    test_index=payload["test"].astype(np.int64),
                    train_pool=train_pool,
                )
            )
    if not folds:
        raise L.AssemblyError(f"{manifest_path} 里没有 folds")
    return L.SplitSpec(
        task_id=task_id,
        benchmark="polybench26",
        protocol="official_scaling_indices_5fold_seed42",
        ref=manifest_path.relative_to(REPO_ROOT).as_posix(),
        sha256=_sha256_of_file(manifest_path),
        n_total=int(manifest.get("n_rows") or 0),
        row_order=(
            f"{str(manifest.get('canonical_source'))}#row_position（官方 canonical 表的发布顺序）"
        ),
        folds=tuple(folds),
    )


def fold_split_version(dataset_id: str, fold_name: str, *, task_id: str = TASK_ID) -> str:
    """一折的 `split_version`：**这一折那个 npz 的字节哈希**（形如 `<12 位>@splits/...`）。
    A fold's `split_version`: the byte digest of that fold's own npz.

    中文：
        用途：`graph.fold_version()` 会拿折名去拼文件名，而这里的折名带数据集前缀
         （`md5000_alternating_cp_fold_00`），拼不出 `fold_00.npz`；所以由本模块先算好，
         通过 `extra_record` 交给 `train_one()` 落进记录。**哈希对象仍是那一折的文件**，
         不是整份 manifest——换掉一份 npz，`split_version` 必须跟着变。
        输入：dataset_id；fold_name（带前缀的折名）；task_id。
        输出：`<sha256 前 12 位>@<仓库相对路径>`。数据来源：`splits/.../seed_42/fold_XX.npz`。
        副作用：只读。异常：AssemblyError —— 折名里没有 `fold_XX` 段或文件不在。
    English:
        `graph.fold_version()` builds the file name from the fold name, and these names carry a
        dataset prefix, so the digest is computed here and handed to `train_one()` through
        `extra_record`. The hash is still over **that fold's file**, not the whole manifest.
    """
    parts = fold_name.split("_")
    if len(parts) < 2 or not parts[-2].startswith("fold"):
        raise L.AssemblyError(f"折名 {fold_name!r} 里找不到 `fold_XX` 段：无法定位折文件")
    suffix = "_".join(parts[-2:])          # `md5000_alternating_cp_fold_00` → `fold_00`
    path = split_dir(dataset_id, task_id) / f"{suffix}.npz"
    if not path.is_file():
        raise L.AssemblyError(f"缺折文件 {path.name}：A-5 的快照不完整")
    digest = _sha256_of_file(path)
    return f"{digest[:12]}@{path.relative_to(REPO_ROOT).as_posix()}"


def build_task_data(dataset_id: str, *, task_id: str = TASK_ID, limit: int | None = None) -> C.TaskData:
    """把一个数据集读成经典臂要的 `TaskData`（描述符 + 指纹 + 官方五折）。
    Build one dataset into the classical arm's TaskData (descriptors + bits + official folds).

    中文：
        用途：与 `conventional.build_task_data()` 同形状，但**不查** B-1 的 `task_features_manifest`
             （那是 11 份主表 pack 的清单），改用 C-b 自己的 `scaling_features_manifest.csv`。
             行序不重排：`row_id` 是原始表的行位置，划分索引在同一坐标系。
        输入：dataset_id；task_id；limit —— 冒烟只取前 N 行（冒烟不写正式产物）。
        输出：`TaskData`（`task_id` 仍是配置里的 `pb26_scaling_cp`）。
        数据来源：pack + `splits/`。副作用：只读。
        异常：ArmError —— pack 哈希不符；AssemblyError —— 划分与行数不配套。
    English:
        Same shape as conventional.build_task_data(), but it checks C-b's own pack manifest instead
        of B-1's (which lists the eleven main-table packs). Row order is never re-sorted: `row_id` is
        the raw table's row position and the split indices live in that same coordinate system.
    """
    pack = load_pack(dataset_id, task_id=task_id)
    split = load_dataset_split(dataset_id, task_id=task_id)
    X = np.hstack([pack["desc"].astype(np.float32), pack["fp"].astype(np.float32)])
    y = np.asarray(pack["y"], dtype=np.float64)
    problems = L.check_split_matches_rows(split, int(X.shape[0]))
    if problems:
        raise L.AssemblyError("；".join(problems))
    row_id = np.asarray(pack["row_id"], dtype=np.int64)
    psmiles = np.asarray(pack["psmiles"])
    if limit is not None and limit < X.shape[0]:
        split = L.limit_split(split, limit)
        X, y, row_id, psmiles = X[:limit], y[:limit], row_id[:limit], psmiles[:limit]
    return C.TaskData(
        task_id=task_id,
        benchmark="polybench26",
        X=X,
        y=y,
        row_id=row_id,
        psmiles=psmiles,
        split=split,
        pack_sha256=manifest_pack_sha256(dataset_id, task_id),
        limit=limit,
    )


# ---- 4. 批量拟合与表 / Batch fits and the table ----


def rf_params(config: Any) -> dict[str, Any]:
    """随机森林的超参：抄学习器敏感性表里那份固定配方（官方 RDKit_RF 的 100 树）。
    The forest's hyper-parameters: the fixed recipe from the sensitivity config (RDKit_RF's 100).

    中文：
        用途：A2 的冻结超参表按设计只有 XGBoost 行（分工_C-a §2：主表一臂一个学习器），所以
              scaling 的 RF 一侧**不能**从 `a2_best_params.csv` 取；它抄的是同一份
              `learner_sensitivity.yaml` 的 `rf.fixed`，口径与 C-a §10 一致，没有新调任何数。
        输入：config —— 实验配置（用于定位敏感性配置；当前固定在仓库里）。
        输出：超参 dict。数据来源：`configs/experiments/learner_sensitivity.yaml`。副作用：只读。
    English:
        A2's frozen table has XGBoost rows only by design (one learner per arm), so the RF side of
        the scaling table cannot read it; it copies the same `rf.fixed` block that C-a's sensitivity
        table used. No new number is tuned.
    """
    import yaml

    path = REPO_ROOT / "configs" / "experiments" / "learner_sensitivity.yaml"
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    fixed = dict(((raw.get("learners") or {}).get("rf") or {}).get("fixed") or {})
    if not fixed:
        raise G.ArmError(f"{path.name} 里没有 `learners.rf.fixed`：RF 一侧没有可抄的配方")
    del config
    return fixed


def run_scaling(
    config: G.ArmConfig,
    *,
    datasets: Sequence[str] | None = None,
    tiers_wanted: Sequence[str] | None = None,
    folds: Sequence[str] | None = None,
    learners: Sequence[str] | None = None,
    resume: bool = True,
) -> list[dict[str, Any]]:
    """按网格跑 scaling：数据集 × 档位 × 折 × 学习器，写记录/预测，返回逐 fit 的表行。
    Run the scaling grid (dataset x tier x fold x learner), write records/predictions, return rows.

    中文：
        用途：`scaling_cpu.csv` 的唯一生产者。两条口径写在实现里：
              ① **XGBoost 是正式臂**：每条都落 `results/raw_runs/<run_id>.json` 与预测
                 （`arm=a2_conventional`、`dataset_id`、`n_level=<档位>`、`subset_sha256`）；
              ② **RF 只进辅助表**：`_schema.json` 的 arm 枚举里没有"森林"，写进 `raw_runs/`
                 会让 C-c 的条数对账多出 40 条（D-45 的同一条道理，C-a 的敏感性表也是这么处理的）。
              档位取法 = 官方前缀（`config` 的 `subset.rule: prefix`）。
        输入：config；datasets / tiers_wanted / folds / learners —— 白名单；resume —— 跳过已有 ok 记录。
        输出：逐 fit 的行（`TABLE_COLUMNS`），同时写记录、预测与 `gpu_hours.csv`。
        数据来源：两份 pack + `splits/`；配置来源：`configs/experiments/scaling.yaml`。
        副作用：训练、写盘。异常：G.ArmError —— 折名或档位不在官方文件里。
    English:
        The only producer of `scaling_cpu.csv`. Two conventions are encoded here: (1) **XGBoost is
        the arm** - every fit writes a record and a parquet (`arm=a2_conventional`, `dataset_id`,
        `n_level`, `subset_sha256`); (2) **the forest feeds the auxiliary table only** - the schema's
        arm enum has no "forest" value and extra rows would break C-c's reconciliation (same reason
        as D-45, which is how C-a's sensitivity table was handled). The tier rule is the official
        prefix (`subset.rule: prefix`).
    """
    chosen_datasets = list(datasets) if datasets else dataset_ids()
    wanted_tiers = {str(value).lower() for value in tiers_wanted} if tiers_wanted else None
    wanted_folds = set(folds) if folds else None
    chosen_learners = list(learners) if learners else ["xgboost", "rf"]
    arm_name = str(config.raw.get("arm") or C.ARM)
    forest = rf_params(config) if "rf" in chosen_learners else {}
    seeds = [int(seed) for seed in config.raw.get("seeds") or [0]]

    rows: list[dict[str, Any]] = []
    for dataset_id in chosen_datasets:
        data = build_task_data(dataset_id)
        available = tiers(dataset_id)
        level_list = [level for level in available if wanted_tiers is None or level in wanted_tiers]
        if wanted_tiers is not None:
            unknown = sorted(wanted_tiers - set(available))
            if unknown:
                raise G.ArmError(f"{dataset_id} 没有这些档位 {unknown}：官方文件里的是 {available}")
        for level in level_list:
            for fold in data.split.fold_names:
                if wanted_folds is not None and fold not in wanted_folds:
                    continue
                for seed in seeds:
                    # 折名本身已带数据集前缀（见 `load_dataset_split()`），所以 run_id 直接用折名。
                    # The fold name already carries the dataset (see load_dataset_split()).
                    run_id = G.RunKey(
                        task=TASK_ID,
                        benchmark="polybench26",
                        fold=fold,
                        seed=seed,
                        arm=arm_name,
                        n_level=level,
                    ).run_id()
                    for learner in chosen_learners:
                        if learner == "xgboost" and _already_done(run_id) and resume:
                            existing = json.loads(
                                (G.RAW_RUNS_DIR / f"{run_id}.json").read_text(encoding="utf-8")
                            )
                            rows.append(_row_from_record(existing, dataset_id, learner))
                            continue
                        outcome = C.train_one(
                            data,
                            config,
                            fold=fold,
                            seed=seed,
                            n_level=level,
                            # 折名已带数据集前缀（`md5000_alternating_cp_fold_00`），所以
                            # run_id 的折段由折名本身承担，不再需要 run_id_fold 覆盖。
                            # The fold name already carries the dataset, so the run_id's fold
                            # segment comes from the name; no run_id_fold override is needed.
                            extra_record={
                                "dataset_id": dataset_id,
                                "fold_index": fold.split("_")[-1],
                                # `split_version` 由本模块按"这一折自己的 npz"算（理由见
                                # `fold_split_version()`）；`graph.fold_version()` 拼不出带前缀的文件名。
                                # split_version is computed here over the fold's own npz; the generic
                                # helper cannot build a prefixed file name.
                                "split_version": fold_split_version(dataset_id, fold),
                            },
                            learner_override=None if learner == "xgboost" else learner,
                            params_override=forest if learner == "rf" else None,
                        )
                        record = outcome.record
                        if learner == "xgboost":
                            G.write_predictions(outcome)
                            G.write_run_record(record)
                            G.append_gpu_hours([record])
                        rows.append(_row_from_record(record, dataset_id, learner))
                        LOGGER.info(
                            "%s [%s] %s/%s n=%s：r2=%.4f n_train=%d %.1f 秒",
                            run_id,
                            learner,
                            dataset_id,
                            fold,
                            level,
                            record["metrics"]["r2"],
                            record["n_train"],
                            record["wall_clock_s"],
                        )
    return rows


def _already_done(run_id: str) -> bool:
    """磁盘上是否已有这条 run 的 `status=ok` 记录。／ Is there already an ok record for this run?"""
    path = G.RAW_RUNS_DIR / f"{run_id}.json"
    if not path.is_file():
        return False
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:                       # pragma: no cover - 半截文件
        return False
    return payload.get("status") == "ok"


def _row_from_record(record: Mapping[str, Any], dataset_id: str, learner: str) -> dict[str, Any]:
    """把一条记录折成 `scaling_cpu.csv` 的一行。／ Fold one record into a scaling_cpu.csv row.

    中文：用途：表列与记录字段一一对应，**不另算任何指标**（R²/RMSE 一律抄记录）。
          `note` 说明这条是否有 run 记录：RF 的 40 行没有，XGBoost 的 40 行有。
    English: columns map one to one onto the record and no metric is recomputed; `note` says whether
    the row has a run record (the forest's 40 do not, XGBoost's 40 do).
    """
    metrics = record.get("metrics") or {}
    return {
        "task": record.get("task", TASK_ID),
        "dataset": dataset_id,
        "tier": record.get("n_level", ""),
        "n_train": record.get("n_train", ""),
        "n_train_pool": record.get("n_train_pool", ""),
        "n_level_saturated": str(bool(record.get("n_level_saturated"))).lower(),
        "n_selection_rule": record.get("n_selection_rule", ""),
        "subset_sha256": record.get("n_subset_sha256", ""),
        "fold": record.get("split_fold", ""),
        "arm": record.get("arm", ""),
        "learner": learner,
        "seed": record.get("seed", ""),
        "r2": f"{float(metrics['r2']):.10f}" if metrics else "",
        "mae": f"{float(metrics['mae']):.10f}" if metrics else "",
        "rmse": f"{float(metrics['rmse']):.10f}" if metrics else "",
        "wall_clock_s": f"{float(record.get('wall_clock_s') or 0.0):.3f}",
        "run_id": str(record.get("run_id", "")) if learner == "xgboost" else "",
        "note": (
            "run 记录 + 预测已落盘"
            if learner == "xgboost"
            else "RF 只进本表、不写 raw_runs（arm 枚举里没有它，D-45 的同款处理）"
        ),
    }


def _tier_sort_key(tier: Any) -> tuple[int, int, str]:
    """档位排序键：数字档按大小、`all` 永远最后（表格便于人读）。
    Tier sort key: numeric tiers by size, `all` always last, so the table reads in order.
    """
    text = str(tier)
    if text == "all":
        return (1, 0, text)
    try:
        return (0, int(text), text)
    except ValueError:                                  # pragma: no cover - 只在配置写错时
        return (2, 0, text)


def write_table(rows: Sequence[Mapping[str, Any]], *, path: Path | None = None) -> Path:
    """写 `scaling_cpu.csv`（整表覆盖，UTF-8 / LF，一 fit 一行）。
    Write scaling_cpu.csv in full (UTF-8 / LF, one row per fit).
    """
    target = Path(path) if path is not None else TABLE_PATH
    target.parent.mkdir(parents=True, exist_ok=True)
    ordered = sorted(
        rows,
        key=lambda row: (
            str(row.get("dataset", "")),
            _tier_sort_key(row.get("tier", "")),
            str(row.get("fold", "")),
            str(row.get("learner", "")),
        ),
    )
    with target.open("w", encoding="utf-8", newline="\n") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(TABLE_COLUMNS), lineterminator="\n")
        writer.writeheader()
        for row in ordered:
            writer.writerow({column: row.get(column, "") for column in TABLE_COLUMNS})
    return target


def read_table(*, path: Path | None = None) -> list[dict[str, str]]:
    """读回 `scaling_cpu.csv`（不存在则空表）。／ Read scaling_cpu.csv back; absent is empty."""
    target = Path(path) if path is not None else TABLE_PATH
    if not target.is_file():
        return []
    with target.open("r", encoding="utf-8", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]
