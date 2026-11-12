"""结构级 polyBERT 嵌入：把每个**结构**压成一个 600 维定长向量，分片落盘、断点续跑。
Structure-level polyBERT embeddings: one 600-dim vector per **structure**, sharded to disk and
resumable.

中文：
    用途：B-2 的主体。A-6 给的 `data/processed/structures/psmiles.parquet` 里是 11,494 个结构，
          本模块把它们逐个送进 polyBERT，得到 `embeddings.npy`（11,494 × 600，float32）+ 
          `index.parquet`（行号 ↔ 结构 ID ↔ 输入串）。三件事在这里钉死：

          ① **输入串的形态**：polyBERT 的词表里 `[*]` 是**一个 token**，裸 `*` 不在词表——直接
             喂 RDKit 规范式 `*C*` 会得到两个 `<unk>`，**不报错**，只是嵌入是废的。所以一律先
             `to_polybert_input()` 把 `*` 换成 `[*]`，并把 `n_unk` 逐条记进索引；实测 11,494 条
             转换后未知 token 数为 **0**（这是"输入形态对了"的凭据，不是运气）。
          ② **行序**：按 `psmiles_canonical` 字典序排，**不按 A-6 的行号**。A-6 的 `structure_id`
             是全局行号，会随上游版本平移（D-25）；按规范式排序，换一版结构表也不会把第 7 行
             换成另一个分子。`order_sha256` 记下这个顺序，可复核。
          ③ **分片与续跑**：`shards/shard_00000.npz` 一片存一段连续行；断了重跑只补缺的片。
             合并时校验"片号连续、区间无缝、总行数对得上"，缺一片就报错，绝不给出半份矩阵。

    依赖：numpy、pyarrow（**不用 pandas**——polybert 环境里没有 pandas，`environment/polybert.yml`
          不含它，这是有意的"只有权重"的精简环境）；torch / transformers 惰性导入，只有真正
          算的时候才要求存在。
    配置：无 yaml。模型版本、维度、分片大小都在本模块的常量里（改它就是改口径，记 D-31）。
    对外：`to_polybert_input()`、`load_structures()`、`plan_shards()`、`embed_shard()`、
          `merge_shards()`、`verify_outputs()`、`MODEL_*` / `EMBED_DIM` 常量。
    被谁用：`scripts/03c_polybert.py`（阶段入口）、`tests/test_polybert.py`（断言）、
            B-4 的 A4 臂（读 `embeddings.npy` + `index.parquet`）。

English:
    Purpose: the body of B-2. A-6's psmiles.parquet holds 11,494 structures; this module pushes each
    through polyBERT to get embeddings.npy (11,494 x 600, float32) plus index.parquet mapping row ->
    structure id -> input string. Three things are pinned here: (1) **input form** - polyBERT's
    vocabulary contains `[*]` as a single token and does not contain a bare `*`, so feeding RDKit's
    canonical `*C*` yields two `<unk>` tokens with no error and a silently useless embedding;
    `to_polybert_input()` rewrites `*` to `[*]` and the index records `n_unk` per structure (all
    11,494 come out with zero unknown tokens, which is evidence the form is right); (2) **row
    order** - sorted by `psmiles_canonical`, not by A-6's row number, because `structure_id` is a
    global position that shifts across upstream versions (D-25); `order_sha256` records the order;
    (3) **sharding and resume** - shards/shard_NNNNN.npz holds a contiguous row range, a rerun only
    fills the missing shards, and the merge refuses to hand back a partial matrix.
    Depends on: numpy, pyarrow (no pandas - the polybert env deliberately has none), and torch /
    transformers lazily, only when actually computing.
    Config: no yaml. Model revision, dimension and shard size live in module constants; changing
    them changes the definition, so they are recorded in D-31.
    Public API: `to_polybert_input()`, `load_structures()`, `plan_shards()`, `embed_shard()`,
    `merge_shards()`, `verify_outputs()` and the MODEL_* / EMBED_DIM constants.
    Callers: scripts/03c_polybert.py, tests/test_polybert.py and B-4's A4 arm.
"""

from __future__ import annotations

import csv
import hashlib
import json
import logging
import os
import time
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

LOGGER = logging.getLogger("chainworth.polybert")

REPO_ROOT = Path(__file__).resolve().parents[3]      # src/chainworth/chem/polybert.py -> 仓库根
PROCESSED_DIR = REPO_ROOT / "data" / "processed"
STRUCTURES_PATH = PROCESSED_DIR / "structures" / "psmiles.parquet"
POLYBERT_DIR = PROCESSED_DIR / "polybert"
EMBEDDINGS_NAME = "embeddings.npy"
INDEX_NAME = "index.parquet"
MANIFEST_NAME = "manifest.json"
SHARDS_NAME = "shards"
EMBEDDINGS_PATH = POLYBERT_DIR / EMBEDDINGS_NAME
INDEX_PATH = POLYBERT_DIR / INDEX_NAME
MANIFEST_PATH = POLYBERT_DIR / MANIFEST_NAME
SHARD_DIR = POLYBERT_DIR / SHARDS_NAME

MANIFEST_DIR = REPO_ROOT / "data" / "manifests"
MANIFEST_CSV = MANIFEST_DIR / "polybert_manifest.csv"

# ── 模型口径 ────────────────────────────────────────────────────────────────────────────────
# 官方仓库是 `kuelumbus/polyBERT`（论文 Data availability 里写的那个）。2026-11-12 实测：
# 该仓库公开接口返回 `X-Error-Code: RepoNotFound`，作者名下公开模型数为 0——官方权重已经拿不
# 到了，不是网络问题。因此改用**三方互相校对过**的副本（D-31）：
#   ① `xushijie/polyBERT` 与 `HAYDERphd/polyBERT` 的 `model.safetensors` **逐字节相同**
#      （sha256 `21cfab4a…`，100,800,352 B）；
#   ② `cooperzvegintzov/polyBERT` 只放 `pytorch_model.bin`，转出来的 197 个张量与上面
#      **逐张量相等、零处不同**（25,194,056 个参数）。
# 三方独立上传、两种格式互证，配 `config.json` 里 `_name_or_path: polyBERT_sentence_transformers/`、
# `hidden_size: 600`、`vocab_size: 269`、max_position 512，与论文描述一致。
# 换回官方权重只需 `--model <repo> --revision <sha>`，再跑一次 `--verify` 比对即可。
MODEL_REPO = "xushijie/polyBERT"
MODEL_REVISION = "e7dce434fb3eff37905dc114008660e5479ca9a8"
MODEL_WEIGHTS_SHA256 = "21cfab4a3b0599852fb8ef529a67e41632246b9256ed834938ba9b46cb24148f"
MODEL_WEIGHTS_BYTES = 100800352
MODEL_N_PARAMS = 25194056
MODEL_CACHE_DIR = Path.home() / ".cache" / "huggingface" / "chainworth" / "polybert_600" / MODEL_REVISION[:8]

EMBED_DIM = 600          # 模型 `hidden_size`，也是池化后的维度 / hidden_size == pooled dim
MAX_TOKENS = 512         # 相对位置编码上限；本结构集最长 309，实际截断 0 条
DEFAULT_SHARD_SIZE = 1024
DEFAULT_BATCH_SIZE = 64

INPUT_NOTE = "psmiles_canonical with every bare `*` rewritten to `[*]`"

SHARD_KEYS = ("row_start", "row_stop", "structure_id", "n_tokens", "n_unk", "vectors")

INDEX_SCHEMA = pa.schema(
    [
        ("row", pa.int32()),
        ("structure_id", pa.int64()),
        ("psmiles_canonical", pa.string()),
        ("polybert_input", pa.string()),
        ("n_tokens", pa.int32()),
        ("n_unk", pa.int32()),
    ]
)

MANIFEST_COLUMNS = (
    "n_structures",
    "embed_dim",
    "model_repo",
    "model_revision",
    "weights_sha256",
    "n_unk_tokens",
    "n_truncated",
    "order_sha256",
    "embeddings_sha256",
    "device",
    "wall_clock_s",
    "created_at",
)


class PolyBertError(RuntimeError):
    """B-2 输入/产物不满足口径时抛这个，不返回半份结果。/ Raised on a broken B-2 input or artifact."""


@dataclass(frozen=True)
class StructureIndex:
    """排好序的结构清单（本模块的唯一行序来源）。/ The sorted structure list; the only row order."""

    structure_id: np.ndarray      # int64, A-6 的全局行号，仅作参照 / A-6's global row, reference only
    canonical: tuple[str, ...]    # 规范式，排序键 / canonical form, the sort key
    inputs: tuple[str, ...]       # 真正喂模型的串 / what actually goes into the model
    n_tokens: np.ndarray          # int32, 分词后的 token 数（含首尾特殊 token）/ token count
    n_unk: np.ndarray             # int32, 该条里的未知 token 数 / unknown-token count

    def __len__(self) -> int:
        return len(self.canonical)

    def order_sha256(self) -> str:
        """对行序做指纹：把所有规范式按行序连起来算 sha256。/ Fingerprint the row order itself.

        中文：用途：证明"这一版嵌入的第 k 行是哪个结构"没被动过。
              输入：无。输出：64 位十六进制。数据来源：self.canonical。副作用：无。
        English: Purpose: prove row k of this embedding still belongs to the same structure.
        """
        digest = hashlib.sha256()
        for text in self.canonical:
            digest.update(text.encode("utf-8"))
            digest.update(b"\n")
        return digest.hexdigest()


def to_polybert_input(psmiles: str) -> str:
    """把 RDKit 规范式改写成 polyBERT 认的 PSMILES 写法（`*` → `[*]`）。
    Rewrite RDKit's canonical SMILES into the PSMILES form polyBERT was trained on (`*` -> `[*]`).

    中文：用途：**这一步错了不会报错**——裸 `*` 在 polyBERT 词表里不存在，会被换成 `<unk>`，
          嵌入照样算得出来、照样是 600 维，只是数值没有意义。所以这里显式拒绝"已经带括号"的
          输入（说明调用方重复转换了），而不是悄悄产出一串 `[[*]]`。
          输入：psmiles —— A-6 的 `psmiles_canonical`，形如 `*CC(*)C`。
          输出：形如 `[*]CC([*])C` 的字符串。数据来源：调用方。
          配置来源：无（词表口径来自模型本身）。副作用：无。
          异常：空串或已含 `[*]` 时抛 `PolyBertError`。
    English: Purpose: **getting this wrong does not raise** - a bare `*` is not in polyBERT's
             vocabulary and becomes `<unk>`, so the embedding is still produced and still 600-dim,
             just meaningless. Hence an explicit rejection of already-bracketed input (the caller
             converted twice) instead of silently emitting `[[*]]`.
             Args: psmiles - A-6's psmiles_canonical, e.g. `*CC(*)C`. Returns: `[*]CC([*])C`.
             Side effects: none. Raises: PolyBertError on an empty string or existing `[*]`.
    """
    text = str(psmiles).strip()
    if not text:
        raise PolyBertError("空的 PSMILES 串，无法嵌入 / empty PSMILES string")
    if "[*]" in text:
        raise PolyBertError(f"输入已含 `[*]`，疑似重复转换：{text!r} / already bracketed, double conversion?")
    if "*" not in text:
        raise PolyBertError(f"输入里没有连接点 `*`，不像重复单元：{text!r} / no attachment point")
    return text.replace("*", "[*]")


def load_structures(path: Path | None = None, limit: int | None = None) -> StructureIndex:
    """读 A-6 的结构表，按规范式排序，给出真正喂模型的串。
    Read A-6's structure table, sort by canonical form, and produce the strings to feed the model.

    中文：用途：本模块的入口。**不用 pandas**（polybert 环境没有 pandas），只取两列。
          输入：path —— 结构表路径，默认 `data/processed/structures/psmiles.parquet`；
                limit —— 只取排序后的前 N 条（冒烟用，产物会落到 `_partial_<N>/`）。
          输出：`StructureIndex`。数据来源：A-6 的结构表（11,494 行，`psmiles_canonical` 唯一）。
          配置来源：无。副作用：无（分词统计要模型，放在 `token_stats()` 里）。
          异常：表里规范式重复、或出现空串时抛 `PolyBertError`。
    English: Purpose: the module's entry point. **No pandas** (the polybert env has none); reads two
             columns only. Args: path, limit. Returns: StructureIndex. Source: A-6's table (11,494
             rows, unique canonical forms). Side effects: none - tokenisation lives in token_stats().
             Raises: PolyBertError on duplicate or empty canonical forms.
    """
    target = Path(path) if path is not None else STRUCTURES_PATH
    if not target.exists():
        raise PolyBertError(f"找不到结构表 {target}——先跑 `python scripts/02_features.py` / missing table")
    table = pq.read_table(target, columns=["structure_id", "psmiles_canonical"])
    ids = np.asarray(table.column("structure_id").to_numpy(), dtype=np.int64)
    raw = [str(item) for item in table.column("psmiles_canonical").to_pylist()]

    if len(set(raw)) != len(raw):
        repeated = len(raw) - len(set(raw))
        raise PolyBertError(f"结构表里有 {repeated} 个重复的规范式——按结构嵌入的前提不成立")

    order = sorted(range(len(raw)), key=lambda i: raw[i])
    if limit is not None:
        order = order[: int(limit)]

    canonical = tuple(raw[i] for i in order)
    inputs = tuple(to_polybert_input(text) for text in canonical)
    return StructureIndex(
        structure_id=ids[order],
        canonical=canonical,
        inputs=inputs,
        n_tokens=np.zeros(len(canonical), dtype=np.int32),
        n_unk=np.zeros(len(canonical), dtype=np.int32),
    )


def plan_shards(n_rows: int, shard_size: int = DEFAULT_SHARD_SIZE) -> list[tuple[int, int]]:
    """把 [0, n_rows) 切成若干左闭右开的连续区间，最后一片可以短。
    Slice [0, n_rows) into contiguous half-open ranges; the last one may be short.

    中文：用途：分片是"断点续跑"的单位，所以区间必须**无缝且不重叠**——否则合并时补不齐。
          输入：n_rows —— 总行数；shard_size —— 每片行数（必须为正）。
          输出：`[(start, stop), ...]`。数据来源：调用方。配置来源：`DEFAULT_SHARD_SIZE`。
          副作用：无。异常：shard_size ≤ 0 或 n_rows < 0 时抛 `PolyBertError`。
    English: Purpose: a shard is the unit of resume, so ranges must be seamless and non-overlapping.
             Args: n_rows, shard_size. Returns: list of (start, stop). Raises: PolyBertError.
    """
    if shard_size <= 0:
        raise PolyBertError(f"shard_size 必须为正，收到 {shard_size}")
    if n_rows < 0:
        raise PolyBertError(f"n_rows 不能为负，收到 {n_rows}")
    return [(start, min(start + shard_size, n_rows)) for start in range(0, n_rows, shard_size)]


def shard_path(directory: Path, index: int) -> Path:
    """第 index 片的路径（5 位补零，按文件名排序即按行序）。
    Path of shard `index` (zero-padded to five digits, so lexical order == row order).

    中文：用途：名字里带片号，续跑时"哪些片已有"就是一次 `glob`。
          输入：directory、index。输出：Path。数据来源：调用方。副作用：无。
    """
    return Path(directory) / f"shard_{index:05d}.npz"


def save_shard(path: Path, payload: dict[str, Any]) -> None:
    """原子写一片：先落 `.partial.npz` 再改名，断电不会留下半个文件。
    Write one shard atomically: `.partial.npz` first, then rename, so a crash leaves no half file.

    中文：输入：path、payload（见 `SHARD_KEYS`）。输出：无。副作用：写磁盘。异常：键不全时抛。
    """
    missing = [key for key in SHARD_KEYS if key not in payload]
    if missing:
        raise PolyBertError(f"分片缺少字段 {missing}")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".partial.npz")
    np.savez(
        temporary,
        row_start=np.int64(payload["row_start"]),
        row_stop=np.int64(payload["row_stop"]),
        structure_id=np.asarray(payload["structure_id"], dtype=np.int64),
        n_tokens=np.asarray(payload["n_tokens"], dtype=np.int32),
        n_unk=np.asarray(payload["n_unk"], dtype=np.int32),
        vectors=np.asarray(payload["vectors"], dtype=np.float32),
    )
    os.replace(temporary, path)


def read_shard(path: Path) -> dict[str, np.ndarray]:
    """读一片并做形状自检（向量行数必须等于 ID 数）。
    Read one shard and check its internal shape agreement.

    中文：输入：path。输出：dict（`SHARD_KEYS` 同名键）。数据来源：磁盘。
          副作用：无。异常：向量行数与 ID 数不符时抛 `PolyBertError`。
    """
    with np.load(Path(path)) as handle:
        payload = {key: handle[key] for key in SHARD_KEYS}
    vectors = payload["vectors"]
    if vectors.shape[0] != payload["structure_id"].shape[0]:
        raise PolyBertError(f"{path}：向量 {vectors.shape[0]} 行 ≠ 结构 ID {payload['structure_id'].shape[0]} 个")
    if vectors.ndim != 2 or vectors.shape[1] != EMBED_DIM:
        raise PolyBertError(f"{path}：向量形状 {vectors.shape}，期望 (n, {EMBED_DIM})")
    return payload


def merge_shards(paths: Sequence[Path], n_expected: int) -> dict[str, np.ndarray]:
    """把分片按行序拼成整份结果，并证明"一片不缺、一行不重"。
    Concatenate shards in row order and prove that no shard and no row is missing.

    中文：用途：合并是唯一能造出"半份矩阵"的地方，所以这里只信行号不信文件名：
          每片自报 `[row_start, row_stop)`，必须从 0 开始首尾相接、最后一片恰好到 n_expected。
          输入：paths —— 分片路径（顺序无关，按 `row_start` 排）；n_expected —— 期望总行数。
          输出：dict，键为 `structure_id` / `n_tokens` / `n_unk` / `vectors`。
          数据来源：分片文件。配置来源：`EMBED_DIM`。副作用：无。
          异常：缺片/重叠/多行少行/维度不符时抛 `PolyBertError`（宁可炸，不给半份）。
    English: Purpose: merging is the only place that could produce a partial matrix, so it trusts
             row numbers rather than file names: every shard declares [row_start, row_stop), the
             ranges must start at 0, touch end to end, and stop exactly at n_expected.
             Args: paths, n_expected. Returns: dict of concatenated arrays. Raises: PolyBertError.
    """
    if not paths:
        raise PolyBertError("没有分片可合并 / no shards to merge")
    payloads = [read_shard(path) for path in paths]
    payloads.sort(key=lambda item: int(item["row_start"]))

    expected_start = 0
    for payload in payloads:
        start, stop = int(payload["row_start"]), int(payload["row_stop"])
        if start != expected_start:
            raise PolyBertError(f"分片不连续：上一片到 {expected_start}，下一片从 {start} 开始（缺片或重叠）")
        if stop - start != payload["structure_id"].shape[0]:
            raise PolyBertError(f"分片 [{start}, {stop}) 的行数与内容不符：{payload['structure_id'].shape[0]} 行")
        expected_start = stop
    if expected_start != n_expected:
        raise PolyBertError(f"分片合计 {expected_start} 行 ≠ 期望 {n_expected} 行")

    return {
        "structure_id": np.concatenate([item["structure_id"] for item in payloads]),
        "n_tokens": np.concatenate([item["n_tokens"] for item in payloads]),
        "n_unk": np.concatenate([item["n_unk"] for item in payloads]),
        "vectors": np.concatenate([item["vectors"] for item in payloads], axis=0),
    }


def shard_matches(path: Path, expected_ids: np.ndarray) -> bool:
    """分片存在、能读、且结构 ID 与本版行序逐位相同，才算"可以复用"。
    A shard may be reused only if it exists, reads cleanly and its ids match this row order.

    中文：用途：续跑的唯一判据。只看文件是否存在是不够的——换了结构表顺序还沿用旧片，就会把
          "甲的行"贴到"乙的结构"上，而且合并时的行号检查**通不过**（所以这里必须自己比 ID）。
          输入：path、expected_ids。输出：bool。数据来源：磁盘分片 + 当前结构表。
          副作用：无。异常：无（读不出来就当不可复用）。
    English: Purpose: the only resume predicate. File existence is not enough - reusing an old shard
             after the order changed would staple row A's vector onto structure B.
    """
    try:
        payload = read_shard(path)
    except (PolyBertError, OSError, ValueError):
        return False
    ids = np.asarray(payload["structure_id"], dtype=np.int64)
    expected = np.asarray(expected_ids, dtype=np.int64)
    return ids.shape == expected.shape and np.array_equal(ids, expected)


def count_unknown(input_ids: Sequence[int], unk_id: int) -> int:
    """数一条里的未知 token。/ Count unknown tokens in one tokenised structure.

    中文：用途：`<unk>` 是"输入形态可能错了"的唯一报警器（裸 `*` 就是这个后果）。
          输入：input_ids、unk_id。输出：int。数据来源：分词器。副作用：无。
    """
    return sum(1 for token in input_ids if int(token) == int(unk_id))


def sha256_of_file(path: Path) -> str:
    """文件的 sha256（分块读，权重文件也不占内存）。/ sha256 of a file, read in chunks."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_of_array(array: np.ndarray) -> str:
    """数组的 sha256（按 C 序取字节，与 dtype 有关必须固定）。/ sha256 of an array's bytes."""
    contiguous = np.ascontiguousarray(array)
    return hashlib.sha256(contiguous.tobytes()).hexdigest()


def now_iso() -> str:
    """当前本地时间（带时区），与 B-1 清单同一个口径。/ Local time with offset, matching B-1."""
    return datetime.now().astimezone().isoformat(timespec="seconds")


def write_index(path: Path, index: StructureIndex) -> None:
    """写 `index.parquet`：行号 ↔ 结构 ID ↔ 输入串 ↔ token 统计。
    Write index.parquet: row -> structure id -> input string -> token statistics.

    中文：输入：path、index。输出：无。副作用：写磁盘（原子替换）。异常：列数不符时由 pa 抛。
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    table = pa.Table.from_arrays(
        [
            pa.array(np.arange(len(index), dtype=np.int32), type=pa.int32()),
            pa.array(np.asarray(index.structure_id, dtype=np.int64), type=pa.int64()),
            pa.array(list(index.canonical), type=pa.string()),
            pa.array(list(index.inputs), type=pa.string()),
            pa.array(np.asarray(index.n_tokens, dtype=np.int32), type=pa.int32()),
            pa.array(np.asarray(index.n_unk, dtype=np.int32), type=pa.int32()),
        ],
        schema=INDEX_SCHEMA,
    )
    temporary = path.with_suffix(".partial.parquet")
    pq.write_table(table, temporary, compression="zstd")
    os.replace(temporary, path)


def read_index(path: Path | None = None) -> dict[str, list[Any]]:
    """读 `index.parquet` 成普通字典（不引 pandas）。/ Read index.parquet into plain dicts."""
    table = pq.read_table(Path(path) if path is not None else INDEX_PATH)
    return {name: table.column(name).to_pylist() for name in table.column_names}


def save_embeddings(path: Path, vectors: np.ndarray) -> None:
    """原子写 `embeddings.npy`（float32，形状 (n, 600)）。/ Write embeddings.npy atomically."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".partial.npy")
    np.save(temporary, np.ascontiguousarray(vectors, dtype=np.float32))
    os.replace(temporary, path)


# ── 模型侧（惰性导入 torch / transformers，核心环境不该因此 import 失败）──────────────────────
def load_model(repo: str = MODEL_REPO, revision: str = MODEL_REVISION, local_dir: Path | None = None):
    """加载分词器与模型，返回 `(tokenizer, model, device)`，并核对权重 sha256。
    Load tokenizer and model, return (tokenizer, model, device) after checking the weights' sha256.

    中文：用途：权重来源是研究口径的一部分，所以**加载时核哈希**——换了一份权重却忘了记，
          会被这一步拦住。缓存目录由 `snapshot_download` 填（HF_ENDPOINT 可指向镜像）。
          输入：repo、revision、local_dir（默认 `MODEL_CACHE_DIR`）。
          输出：三元组，模型已在 device 上并 `eval()`。
          数据来源：HuggingFace 缓存。配置来源：本模块 `MODEL_*` 常量。
          副作用：首次会下载约 97 MB 权重（写缓存目录）。异常：哈希不符 / 维度不符时抛。
    English: Purpose: where the weights came from is part of the research definition, so the hash is
             checked **at load time** - swapping weights without recording it gets caught here.
             Args: repo, revision, local_dir. Returns: (tokenizer, model, device), model in eval().
             Side effects: first call downloads ~97 MB into the cache dir. Raises: on hash or dim
             mismatch.
    """
    import torch
    from transformers import AutoModel, AutoTokenizer

    target = Path(local_dir) if local_dir is not None else MODEL_CACHE_DIR
    weights = target / "model.safetensors"
    if not weights.exists():
        os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
        from huggingface_hub import snapshot_download

        LOGGER.info("下载权重 %s@%s -> %s", repo, revision[:8], target)
        snapshot_download(repo_id=repo, revision=revision, local_dir=str(target))

    digest = sha256_of_file(weights)
    if digest != MODEL_WEIGHTS_SHA256:
        raise PolyBertError(
            f"权重哈希不符：{digest} ≠ {MODEL_WEIGHTS_SHA256}——换模型就改常量并记一条决策"
        )

    tokenizer = AutoTokenizer.from_pretrained(str(target))
    model = AutoModel.from_pretrained(str(target)).eval()
    if int(model.config.hidden_size) != EMBED_DIM:
        raise PolyBertError(f"模型 hidden_size={model.config.hidden_size}，与常量 {EMBED_DIM} 不符")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    return tokenizer, model, device


def token_stats(tokenizer, texts: Sequence[str]) -> tuple[np.ndarray, np.ndarray]:
    """先只分词，统计每条 token 数与未知 token 数（不占显存，用来提前报警）。
    Tokenise only, counting tokens and unknowns per string - no GPU needed, so problems surface early.

    中文：输入：tokenizer、texts。输出：`(n_tokens, n_unk)` 两个 int32 数组。
          数据来源：分词器词表。副作用：无。
    """
    encoded = tokenizer(list(texts), add_special_tokens=True, truncation=False)["input_ids"]
    lengths = np.array([len(item) for item in encoded], dtype=np.int32)
    unk_id = int(tokenizer.unk_token_id)
    unknowns = np.array([count_unknown(item, unk_id) for item in encoded], dtype=np.int32)
    return lengths, unknowns


def embed_shard(
    tokenizer,
    model,
    device: str,
    texts: Sequence[str],
    batch_size: int = DEFAULT_BATCH_SIZE,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """对一段文本做前向 + 注意力掩码均值池化，返回 (向量, token 数, 未知数)。
    Forward one block of strings with attention-mask mean pooling.

    中文：用途：池化照官方 README 的写法（乘掩码求和，再除以掩码和，`clamp(min=1e-9)`），
          与 `SentenceTransformer` 的 `pooling_mode_mean_tokens` 等价——A4 臂要和论文里的
          polyBERT 指纹可比，所以不能自创池化方式。
          输入：tokenizer、model、device、texts、batch_size。
          输出：`(vectors float32 [n, 600], n_tokens int32 [n], n_unk int32 [n])`。
          数据来源：调用方给的 texts（顺序即输出顺序）。配置来源：`MAX_TOKENS`。
          副作用：占用显存（`inference_mode`，不留梯度）。异常：维度不符时抛 `PolyBertError`。
    English: Purpose: pooling follows the official README recipe (masked sum over masked sum,
             clamped at 1e-9), equivalent to SentenceTransformer's pooling_mode_mean_tokens - the A4
             arm must stay comparable to the published polyBERT fingerprints, so no home-made
             pooling. Args: tokenizer, model, device, texts, batch_size. Returns: (vectors,
             n_tokens, n_unk). Side effects: uses GPU memory, no gradients kept.
    """
    import torch

    unk_id = int(tokenizer.unk_token_id)
    chunks: list[np.ndarray] = []
    token_counts: list[np.ndarray] = []
    unknown_counts: list[np.ndarray] = []
    with torch.inference_mode():
        for start in range(0, len(texts), batch_size):
            block = list(texts[start : start + batch_size])
            batch = tokenizer(
                block,
                padding=True,
                truncation=True,
                max_length=MAX_TOKENS,
                return_tensors="pt",
            )
            ids = batch["input_ids"]
            token_counts.append(
                np.array([int(row.sum()) for row in batch["attention_mask"]], dtype=np.int32)
            )
            unknown_counts.append(
                np.array(
                    [count_unknown(row.tolist(), unk_id) for row in ids],
                    dtype=np.int32,
                )
            )
            batch = {key: value.to(device) for key, value in batch.items()}
            hidden = model(**batch)[0]
            mask = batch["attention_mask"].unsqueeze(-1).expand(hidden.size()).float()
            pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)
            chunks.append(pooled.to("cpu", dtype=torch.float32).numpy())

    vectors = np.concatenate(chunks, axis=0) if chunks else np.zeros((0, EMBED_DIM), dtype=np.float32)
    if vectors.shape != (len(texts), EMBED_DIM):
        raise PolyBertError(f"池化输出形状 {vectors.shape}，期望 ({len(texts)}, {EMBED_DIM})")
    return vectors, np.concatenate(token_counts), np.concatenate(unknown_counts)


def verify_outputs(
    directory: Path | None = None,
    recompute: int = 0,
    tokenizer=None,
    model=None,
    device: str = "cpu",
) -> list[str]:
    """逐条核对产物，返回问题清单（空列表 = 全过）。
    Check the artifacts one by one; an empty list means everything passed.

    中文：用途：五条硬检查——① 行数对；② 行序与规范式排序逐行相同（顺序错了会静默换样品）；
          ③ 没有 NaN/Inf；④ 维度是 600；⑤ 未知 token 数为 0。加上可选的"重算若干行比对"。
          输入：directory（默认 `POLYBERT_DIR`）、recompute（>0 时重算这么多行逐行比）、
                tokenizer/model/device（重算要用）。
          输出：`list[str]` 问题描述。数据来源：磁盘产物 + 结构表。
          配置来源：`EMBED_DIM`、`MODEL_*`。副作用：`recompute>0` 时占显存。
    English: Purpose: five hard checks - row count, row order equal to the canonical sort (a wrong
             order silently swaps samples), no NaN/Inf, dimension 600, zero unknown tokens; plus an
             optional recompute-and-compare of a few rows. Returns: list of problem strings.
    """
    root = Path(directory) if directory is not None else POLYBERT_DIR
    problems: list[str] = []
    embeddings_file = root / EMBEDDINGS_NAME
    index_file = root / INDEX_NAME
    manifest_file = root / MANIFEST_NAME
    for path in (embeddings_file, index_file, manifest_file):
        if not path.exists():
            return [f"缺少产物 {path}——先跑 `python scripts/03c_polybert.py`"]

    vectors = np.load(embeddings_file)
    index = read_index(index_file)
    manifest = json.loads(manifest_file.read_text(encoding="utf-8"))

    # 试跑产物（`_partial_<N>/`）只覆盖排序后的前 N 个结构，比对基准也要跟着裁；
    # 否则会把"本来就只跑了 64 条"误报成"少了 11,430 条"。
    # A limited artifact covers the first N structures in sort order, so the reference must be
    # trimmed the same way; otherwise a 64-row smoke run looks like "11,430 rows missing".
    is_partial = bool(manifest.get("is_partial", False))
    declared = manifest.get("n_structures")
    reference = load_structures(limit=int(declared) if is_partial and declared else None)
    n_rows = len(index["row"])
    if n_rows != len(reference):
        problems.append(f"索引 {n_rows} 行 ≠ 结构表 {len(reference)} 个结构")
    if vectors.shape[0] != n_rows:
        problems.append(f"embeddings {vectors.shape[0]} 行 ≠ 索引 {n_rows} 行")
    if vectors.ndim != 2 or vectors.shape[1] != EMBED_DIM:
        problems.append(f"embeddings 形状 {vectors.shape}，期望 (n, {EMBED_DIM})")
        return problems

    if list(index["psmiles_canonical"]) != list(reference.canonical):
        stored = list(index["psmiles_canonical"])
        first = next(
            (i for i, (left, right) in enumerate(zip(stored, reference.canonical)) if left != right),
            min(len(stored), len(reference.canonical)),
        )
        problems.append(f"行序与规范式排序不同（首个不同在第 {first} 行：索引 {stored[first] if first < len(stored) else '越界'}）")
    if list(index["row"]) != list(range(n_rows)):
        problems.append("index.parquet 的 row 列不是 0..n-1 连续整数")
    if len(set(index["psmiles_canonical"])) != n_rows:
        problems.append("索引里有重复的规范式")

    if not np.isfinite(vectors).all():
        problems.append(f"embeddings 里有 {int((~np.isfinite(vectors)).sum())} 个 NaN/Inf")
    unk_total = int(np.asarray(index["n_unk"], dtype=np.int64).sum())
    if unk_total:
        with_unk = int((np.asarray(index["n_unk"]) > 0).sum())
        problems.append(f"{with_unk} 条结构含未知 token（合计 {unk_total} 个）——输入形态可能不对")
    truncated = int((np.asarray(index["n_tokens"]) >= MAX_TOKENS).sum())
    if truncated and int(manifest.get("n_truncated", 0)) != truncated:
        problems.append(f"索引里有 {truncated} 条达到 {MAX_TOKENS} token，但清单记的是 {manifest.get('n_truncated')}")

    if str(manifest.get("order_sha256", "")) != reference.order_sha256() and n_rows == len(reference):
        problems.append("清单里的 order_sha256 与当前结构表的排序不一致（结构表变过？）")
    if int(manifest.get("n_structures", -1)) != n_rows:
        problems.append(f"清单记 n_structures={manifest.get('n_structures')}，实际 {n_rows} 行")

    if recompute > 0:
        if tokenizer is None or model is None:
            problems.append("要求重算，但没给 tokenizer/model")
        else:
            rng = np.random.default_rng(0)
            picks = np.sort(rng.choice(n_rows, size=min(recompute, n_rows), replace=False))
            again, _, _ = embed_shard(tokenizer, model, device, [index["polybert_input"][i] for i in picks])
            drift = np.abs(again - vectors[picks]).max()
            if drift > 1e-4:
                problems.append(f"重算 {len(picks)} 行，最大偏差 {drift:.3e} > 1e-4——落盘的向量不可复现")
    return problems


def build_manifest(
    index: StructureIndex,
    vectors: np.ndarray,
    *,
    device: str,
    wall_clock_s: float,
    n_truncated: int,
    shard_size: int,
    is_partial: bool,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """组装 `manifest.json` 的内容（维度、覆盖数、模型版本、两份哈希）。
    Build manifest.json: dimension, coverage, model version and both hashes.

    中文：输入：index、vectors 与若干运行信息。输出：dict。数据来源：本次运行。
          配置来源：`MODEL_*` / `EMBED_DIM`。副作用：无。
    """
    payload: dict[str, Any] = {
        "n_structures": int(len(index)),
        "embed_dim": int(vectors.shape[1]),
        "model_repo": MODEL_REPO,
        "model_revision": MODEL_REVISION,
        "weights_sha256": MODEL_WEIGHTS_SHA256,
        "model_weights_sha256": MODEL_WEIGHTS_SHA256,
        "model_n_params": MODEL_N_PARAMS,
        "input_form": INPUT_NOTE,
        "max_tokens": MAX_TOKENS,
        "n_truncated": int(n_truncated),
        "n_unk_tokens": int(np.asarray(index.n_unk, dtype=np.int64).sum()),
        "order_basis": "psmiles_canonical ascending",
        "order_sha256": index.order_sha256(),
        "embeddings_sha256": sha256_of_array(vectors),
        "dtype": str(vectors.dtype),
        "device": device,
        "shard_size": int(shard_size),
        "is_partial": bool(is_partial),
        "created_at": now_iso(),
        "wall_clock_s": round(float(wall_clock_s), 2),
    }
    if extra:
        payload.update(extra)
    return payload


def write_manifest(path: Path, payload: dict[str, Any]) -> None:
    """原子写 `manifest.json`（键排序，便于 diff）。/ Write manifest.json atomically, keys sorted."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".partial.json")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def append_manifest_csv(path: Path, payload: dict[str, Any]) -> None:
    """往 `data/manifests/polybert_manifest.csv` 追加一行（表头不存在时先写）。
    Append one row to the manifest CSV, writing the header first if needed.

    中文：输入：path、payload。输出：无。副作用：追加写（入库的小文件）。
          异常：payload 缺列时抛 `PolyBertError`。
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    row = {name: payload.get(name, "") for name in MANIFEST_COLUMNS}
    exists = path.exists() and path.stat().st_size > 0
    with path.open("a", encoding="utf-8", newline="") as handle:
        # lineterminator 显式给 "\n"：Windows 下默认是 "\r\n"，会让清单在 `git add` 时报
        # "CRLF will be replaced by LF"（仓库规则 `* text=auto eol=lf`）。
        # Explicit "\n": the Windows default is "\r\n", which trips the repo's eol=lf rule.
        writer = csv.DictWriter(handle, fieldnames=list(MANIFEST_COLUMNS), lineterminator="\n")
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def resolve_output_dir(limit: int | None) -> Path:
    """按是否 `--limit` 决定产物落哪：试跑进 `_partial_<N>/`，绝不与全量产物同名。
    Decide where artifacts go: a limited run lands in `_partial_<N>/`, never under the full name.

    中文：用途：B-2 的验收写着"不许把 `--limit 200` 的试跑结果当成全量交上来"，所以从**路径**
          上就分开——不看清单也知道这是试跑。
          输入：limit。输出：Path。数据来源：调用方。副作用：无。
    """
    if limit is None:
        return POLYBERT_DIR
    return POLYBERT_DIR / f"_partial_{int(limit)}"


def timer() -> float:
    """开始计时的墙钟读数（秒）。/ Wall-clock reading to start a timing block."""
    return time.perf_counter()
