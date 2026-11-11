"""结构级数值表征：RDKit 描述符矩阵 + Morgan 指纹，带 provenance 校验的本地缓存。
Structure-level numeric representation: RDKit descriptors plus a Morgan fingerprint, behind a
provenance-checked local cache.

中文：
    用途：A-6 的第三件产物。一个结构算一次 205 个描述符与 2048 位指纹，全项目复用。
    依赖：rdkit（2024.03.5，`environment/core.yml` 锁定）、numpy、标准库 sqlite3。
    配置：`DROPPED_DESCRIPTORS`、`FINGERPRINT_RADIUS`、`FINGERPRINT_BITS` 三个常量——
          它们决定**特征空间的定义**，改任何一个都要记 DECISIONS 并重出全部产物。
    对外：`descriptor_columns()`、`compute()`、`DescriptorCache`、`fingerprint_array()`、
          `DROPPED_DESCRIPTORS`、`FINGERPRINT_RADIUS`、`FINGERPRINT_BITS`、`ALGORITHM_VERSION`。
    被谁用：`src/chainworth/chem/structures.py`、`scripts/02_features.py`、`tests/test_features.py`。

English:
    Purpose: A-6's third deliverable - 205 descriptors and a 2048-bit fingerprint per structure,
    computed once and reused across the project.
    Depends on: rdkit (2024.03.5, pinned in environment/core.yml), numpy, stdlib sqlite3.
    Config: `DROPPED_DESCRIPTORS`, `FINGERPRINT_RADIUS` and `FINGERPRINT_BITS` define the
    feature space itself; changing any of them means a DECISIONS entry and full regeneration.
    Public API: `descriptor_columns()`, `compute()`, `DescriptorCache`, `fingerprint_array()`,
    `DROPPED_DESCRIPTORS`, `FINGERPRINT_RADIUS`, `FINGERPRINT_BITS`, `ALGORITHM_VERSION`.
    Callers: src/chainworth/chem/structures.py, scripts/02_features.py, tests/test_features.py.

为什么是这 5 个描述符被丢掉 ／ Why exactly these five descriptors are dropped
-----------------------------------------------------------------------------
PolyBench26 官方 `Datasets/Dataset_construction_scripts/rdkit_descriptor_cache.py` 的
`DROP_COLUMNS` 是这五个名字，一个不多一个不少：

    MaxPartialCharge, MinPartialCharge, MaxAbsPartialCharge, MinAbsPartialCharge, Ipc

前四个要算部分电荷，而部分电荷需要 Gasteiger 参数；`*`（原子序数 0 的 wildcard）没有参数，
于是这四个描述符对**每一个**含 `*` 的分子都是 NaN——实测 210 个描述符里有 4 个恒定 NaN，
留着只会让 205 变成"205 里混着 4 列全空"。`Ipc` 是走路径数的信息量描述符，对聚合物重复单元
数值可以达到 1e30 量级（官方注释称其"病态"），XGBoost 与标准化都会被打爆。
我们不重新论证这件事：**跟着官方丢，才能在 PolyBench26 的 `RDKit_RF` 基线上对话**。
这个取舍记在 docs/DECISIONS.md D-25。

复核方式（不必下载整包，直接看原始文件）：
    https://raw.githubusercontent.com/rlearsch/PolymerBenchmark2026/83655bc/Datasets/Dataset_construction_scripts/rdkit_descriptor_cache.py

与官方实现的**唯一**差别：官方的缓存键是**原始** PSMILES 字符串（它有意不改化学身份约定），
我们的键是**规范化**后的字符串——因为 A-6 的任务是"每个结构算一次"，而同一分子在 PolyBench
里可以有多种写法。两者算出的数值相同，差别只在命中率。记在 D-25。

PolyBench26's official `rdkit_descriptor_cache.py` drops exactly these five names. The first four
need partial charges, which need Gasteiger parameters; a `*` wildcard (atomic number 0) has none,
so those four are NaN for every `*`-containing molecule - 4 of the 210 descriptors are constantly
empty and keeping them would only mean "205 columns of which 4 are always blank". `Ipc` is a
path-count information index that reaches ~1e30 for repeat units and wrecks both XGBoost and
standardisation. We do not re-litigate this: dropping the same five is what lets our numbers be
compared with PolyBench26's own `RDKit_RF` baseline. Recorded as DECISIONS D-25.

Verification path (no need to fetch the whole tarball):
    https://raw.githubusercontent.com/rlearsch/PolymerBenchmark2026/83655bc/Datasets/Dataset_construction_scripts/rdkit_descriptor_cache.py

The one difference from the official implementation: its cache key is the *raw* PSMILES string (it
deliberately preserves the chemical-identity convention), whereas ours is the *canonical* form,
because A-6's job is "compute each structure once" and one molecule has several spellings here.
The numbers are identical; only the hit rate differs. Recorded in D-25.

关于指纹参数 ／ About the fingerprint parameters
-----------------------------------------------
OpenPoly 官方 XGBoost 用的 `data.morgan_fp` 是打包在**未发布**的 `.pt` 里的，
维度与半径在公开代码里查不到（notebook 只有 `data.morgan_fp.numpy()`）。
所以这两个数不是"照抄官方"，而是**我们自己的选择**，取值理由：
（a）RDKit 2022 之后 `GetMorganGenerator()` 的默认值就是 radius=2 / fpSize=2048；
（b）PolyBench 与 OpenPoly 两条臂用同一组参数，臂间比较才干净。
这条也记在 D-25——不许在论文里写成"按官方设置"。

OpenPoly's official XGBoost reads `data.morgan_fp` out of an unpublished `.pt`; neither the radius
nor the bit count is discoverable from the public code (the notebooks only call
`data.morgan_fp.numpy()`). These two numbers are therefore *our* choice, not a copy: (a) they are
the defaults of RDKit's post-2022 `GetMorganGenerator()`, and (b) using one setting across the
PolyBench and OpenPoly arms keeps the comparison clean. Recorded as D-25 - never write "official
settings" in the paper.
"""

from __future__ import annotations

import json
import math
import sqlite3
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

import numpy as np
from rdkit import Chem, RDLogger, rdBase
from rdkit.Chem import rdFingerprintGenerator
from rdkit.Chem.Descriptors import CalcMolDescriptors
from rdkit.DataStructs import ConvertToNumpyArray

RDLogger.DisableLog("rdApp.*")

# 缓存/产物的算法版本：改了描述符列、指纹参数或清洗规则就要 +1，否则老缓存会被当成新结果。
# Algorithm version: bump it whenever the columns, the fingerprint or the cleaning rule change,
# otherwise a stale cache passes itself off as a fresh result.
ALGORITHM_VERSION = 1

# 官方 `rdkit_descriptor_cache.DROP_COLUMNS`，逐字照抄，顺序也照抄（顺序进 manifest，便于对拍）。
# The official DROP_COLUMNS, copied verbatim including order (the order goes into the manifest so
# two implementations can be compared field by field).
DROPPED_DESCRIPTORS: tuple[str, ...] = (
    "MaxPartialCharge",
    "MinPartialCharge",
    "MaxAbsPartialCharge",
    "MinAbsPartialCharge",
    "Ipc",
)

# Morgan 指纹参数：RDKit 新版生成器的默认值；理由见模块 docstring 末段与 D-25。
# Morgan settings: RDKit's post-2022 defaults; rationale in the module docstring and D-25.
FINGERPRINT_RADIUS = 2
FINGERPRINT_BITS = 2048

# 指纹类型：二进制位（0/1）。官方 XGBoost 把 `morgan_fp` 拍平成向量当特征，二进制是同一口径；
# 用计数型会让指纹变成稀疏整数矩阵，和"位"语义混淆。
# Fingerprint kind: binary bits. The official XGBoost flattens `morgan_fp` into a feature vector,
# which is the same convention; a count fingerprint would be a sparse integer matrix instead.
FINGERPRINT_KIND = "binary"

_GENERATOR = rdFingerprintGenerator.GetMorganGenerator(
    radius=FINGERPRINT_RADIUS, fpSize=FINGERPRINT_BITS
)

_COLUMN_CACHE: list[str] | None = None


def descriptor_columns() -> list[str]:
    """返回冻结的描述符列名与列序（205 列，已剔除官方那 5 个）。
    Return the frozen descriptor column names and order (205 columns, official five removed).

    中文：
        用途：列序必须**与分子无关**。`CalcMolDescriptors` 对每个分子返回的字典键序是固定的，
              但仍拿一个参照分子（甲烷）跑一遍来取序，好处是空环境也能得到同一个列序，
              不必先有数据。结果缓存，避免 11,494 次重复构造。
        输入：无。
        输出：列名列表（长度见 `len(descriptor_columns())`，当前 205）。
        数据来源：RDKit 的描述符注册表。配置来源：`DROPPED_DESCRIPTORS`。
        副作用：首次调用写入模块级缓存。异常：无。

    English:
        Purpose: the column order must not depend on the molecule. `CalcMolDescriptors` returns a
                 fixed key order for every molecule, but running one reference molecule (methane)
                 makes that explicit and works even with no data on disk. Cached, so 11,494 calls
                 do not rebuild it.
        Args: none. Returns: the column names (205 today).
        Data source: RDKit's descriptor registry. Config: `DROPPED_DESCRIPTORS`.
        Side effects: fills a module-level cache on first call. Raises: none.
    """
    global _COLUMN_CACHE
    if _COLUMN_CACHE is None:
        reference = Chem.MolFromSmiles("C")
        values = CalcMolDescriptors(reference, missingVal=np.nan, silent=True)
        _COLUMN_CACHE = [name for name in values if name not in DROPPED_DESCRIPTORS]
    return list(_COLUMN_CACHE)


def fingerprint_array(payload: bytes) -> np.ndarray:
    """把 256 字节的位图展开成 2048 个 uint8（0/1），供 DataFrame 直接用。
    Expand the 256-byte bitmap into 2048 uint8 values (0/1) for direct DataFrame use.

    中文：
        用途：parquet 里存的是 2048 个独立的 `morgan_XXXX` 列，缓存里存的是紧凑位图；
              这里是两者之间唯一的转换点，避免各处自己 `unpackbits` 出偏差。
        输入：payload —— 256 字节（= 2048/8）。
        输出：shape (2048,) 的 np.uint8 数组。
        数据来源：`DescriptorCache` 的 BLOB 列。配置来源：`FINGERPRINT_BITS`。
        副作用：无。异常：ValueError —— 字节数不是 256（说明缓存被写坏或参数不同）。

    English:
        Purpose: parquet stores 2048 separate `morgan_XXXX` columns while the cache stores a
                 compact bitmap; this is the single conversion point between them.
        Args: payload - 256 bytes (= 2048/8). Returns: a (2048,) uint8 array.
        Data source: the cache's BLOB column. Config: `FINGERPRINT_BITS`.
        Side effects: none. Raises: ValueError when the byte length is not 256 (corrupt cache or a
        different bit count).
    """
    expected = FINGERPRINT_BITS // 8
    if len(payload) != expected:
        raise ValueError(f"指纹位图应为 {expected} 字节，实际 {len(payload)} 字节")
    return np.unpackbits(np.frombuffer(payload, dtype=np.uint8))


def _fingerprint_bits(molecule: Chem.Mol) -> np.ndarray:
    """把一个分子算成 2048 个 0/1（uint8），而不是 RDKit 的稀疏编码。
    Turn a molecule into 2048 0/1 values (uint8), not RDKit's sparse encoding.

    中文：
        用途：**别用 `fp.ToBinary()`**。它返回的是 Daylight 稀疏编码（置位少的时候只有十几个
              字节），不是定长位图——按 256 字节去读会得到"指纹只有 18 字节"这种莫名其妙的
              错误，而更糟的是：有人可能把它当定长向量直接塞进模型，形状随分子而变。
              `ConvertToNumpyArray` 是 RDKit 官方给的定长展开方式。
        输入：molecule —— 已解析的 RDKit 分子。
        输出：shape (FINGERPRINT_BITS,) 的 np.uint8 数组（0/1）。
        数据来源：`_GENERATOR`。配置来源：`FINGERPRINT_RADIUS`、`FINGERPRINT_BITS`。
        副作用：无。异常：无（RDKit 会给一个全零向量而不是抛）。

    English:
        Purpose: do not use `fp.ToBinary()`. It returns the Daylight sparse encoding (a dozen bytes
                 when few bits are set), not a fixed-width bitmap, so reading it as 256 bytes fails
                 with a baffling "fingerprint is 18 bytes" - and worse, someone could feed it to a
                 model as a fixed vector whose length depends on the molecule.
                 `ConvertToNumpyArray` is RDKit's official fixed-width expansion.
        Args: molecule. Returns: a (FINGERPRINT_BITS,) uint8 array of 0/1.
        Data source: `_GENERATOR`. Config: `FINGERPRINT_RADIUS`, `FINGERPRINT_BITS`.
        Side effects: none. Raises: none (RDKit yields a zero vector rather than raising).
    """
    bits = np.zeros((FINGERPRINT_BITS,), dtype=np.uint8)
    ConvertToNumpyArray(_GENERATOR.GetFingerprint(molecule), bits)
    return bits


def compute(psmiles: str) -> dict[str, Any] | None:
    """算一个结构的全部数值表征；解析失败返回 None。
    Compute every numeric representation of one structure; None when it cannot be parsed.

    中文：
        用途：整条特征管线的唯一计算口。**一次解析、一次描述符、一次指纹**，
              不在这里读文件、不在这里做缓存（缓存是 `DescriptorCache` 的职责）。
              `*` 原样留在图里——这与官方 `PSMILES_to_RDKit_descriptors.py` 一致，
              理由见模块 docstring 与 D-25。
        输入：psmiles —— 任意 PSMILES（内部会再规范化一次，防止调用方传了非规范串）。
        输出：`{"psmiles_canonical", "descriptors": {列名: float}, "fingerprint": bytes,
              "n_nan": int}`；解析失败返回 None。
        数据来源：调用方。配置来源：`DROPPED_DESCRIPTORS`、`FINGERPRINT_*`。
        副作用：无。异常：无——RDKit 解析失败走 None 分支。

    English:
        Purpose: the single computation point of the feature pipeline: parse once, descriptors
                 once, fingerprint once. No file I/O, no caching here (that is
                 `DescriptorCache`'s job). The asterisks stay in the graph, matching the official
                 `PSMILES_to_RDKit_descriptors.py`; see the module docstring and D-25.
        Args: psmiles - any PSMILES (canonicalised again here so a caller cannot slip in a
              non-canonical string). Returns: a dict with `psmiles_canonical`, `descriptors`,
              `fingerprint` (bytes) and `n_nan`; None when unparsable.
        Data source: the caller. Config: the constants above.
        Side effects: none. Raises: none - an RDKit parse failure takes the None branch.
    """
    molecule = Chem.MolFromSmiles(psmiles)
    if molecule is None:
        return None
    canonical = Chem.MolToSmiles(molecule)
    values = CalcMolDescriptors(molecule, missingVal=np.nan, silent=True)
    columns = descriptor_columns()
    descriptors: dict[str, float] = {}
    n_nan = 0
    for name in columns:
        number = values.get(name, math.nan)
        number = float(number) if number is not None else math.nan
        if not math.isfinite(number):
            number = math.nan
            n_nan += 1
        descriptors[name] = number
    payload = np.packbits(_fingerprint_bits(molecule)).tobytes()
    return {
        "psmiles_canonical": canonical,
        "descriptors": descriptors,
        "fingerprint": payload,
        "n_nan": n_nan,
    }


def provenance() -> dict[str, Any]:
    """缓存与产物必须匹配的"环境指纹"；不一致就说明特征空间变了。
    The environment fingerprint a cache must match; a mismatch means the feature space moved.

    中文：
        用途：描述符数值会随 RDKit 版本漂移（`Chi`、`BCUT` 这类尤其）。把版本、列名、列序、
              指纹参数一起写进 provenance，缓存与清单都拿它自检——**版本变了宁可报错重建，
              也不许把两套特征混进同一个实验**。
        输入：无。
        输出：dict（可直接 json.dumps）。
        数据来源：当前进程里的 RDKit。配置来源：本模块常量。
        副作用：无。异常：无。

    English:
        Purpose: descriptor values drift across RDKit versions (notably `Chi`/`BCUT`). Version,
                 column names, column order and fingerprint settings all go into the provenance,
                 and both the cache and the manifest check it: a version change must fail and
                 rebuild rather than silently mix two feature spaces in one experiment.
        Args: none. Returns: a JSON-serialisable dict.
        Data source: RDKit in this process. Config: this module's constants.
        Side effects: none. Raises: none.
    """
    return {
        "algorithm_version": ALGORITHM_VERSION,
        "rdkit_version": rdBase.rdkitVersion,
        "descriptor_columns": descriptor_columns(),
        "dropped_descriptors": list(DROPPED_DESCRIPTORS),
        "fingerprint": {
            "kind": FINGERPRINT_KIND,
            "radius": FINGERPRINT_RADIUS,
            "n_bits": FINGERPRINT_BITS,
            # 存储编码也要进 provenance：`packbits` 前那版缓存存的是 RDKit 的稀疏编码
            # （置位少时只有 18 字节），读回来会被当成"位图长度不对"。改动这个字段等于让所有
            # 老缓存失效——这正是想要的行为。
            # The storage encoding belongs in the provenance too: the pre-`packbits` cache held
            # RDKit's sparse encoding (18 bytes when few bits are set) and read back as "wrong
            # bitmap length". Bumping this field invalidates every old cache, which is the point.
            "encoding": "packbits",
        },
    }


class DescriptorCache:
    """一个只有"同 provenance 才能打开"的 SQLite 缓存；键是规范 PSMILES。
    A SQLite cache that only opens under matching provenance; keyed by canonical PSMILES.

    中文：
        用途：11,494 个结构算一次约 42 秒——不算贵，但整条流水线（A-6 之后 B/C 还要重跑）
              会把它乘以很多倍。缓存让"第二个任务里出现的同一个结构"变成一次查表。
              **provenance 不匹配直接抛**，不做"尽力而为"的兼容：混用两套描述符得出的
              模型无法解释，而且不会报错。
        输入：path —— sqlite 文件路径（放 `data/processed/structures/.descriptor_cache.sqlite3`，
              `data/processed/*` 已被 gitignore，缓存永不入库）。
        输出：实例；`records()` 返回按输入顺序的记录列表与命中统计。
        数据来源：本机磁盘。配置来源：`provenance()`。
        副作用：建库、写库。异常：ValueError —— provenance 不匹配（提示删缓存重建）。

    English:
        Purpose: 11,494 structures take about 42 s - not expensive once, but the pipeline (and B/C
                 re-running it later) multiplies that. The cache turns "the same structure seen
                 again in another task" into a lookup. A provenance mismatch raises immediately;
                 there is no best-effort compatibility, because mixing two descriptor spaces
                 produces a model nobody can explain - and it would not raise on its own.
        Args: path - the sqlite file (under data/processed/structures/, which is gitignored and
              therefore never committed).
        Returns: an instance whose `records()` gives input-ordered records plus hit/miss counts.
        Data source: local disk. Config: `provenance()`.
        Side effects: creates and writes the database.
        Raises: ValueError on a provenance mismatch (the message says to delete and rebuild).
    """

    def __init__(self, path: Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.connection = sqlite3.connect(self.path)
        self._initialise_or_validate()
        self.hits = 0
        self.misses = 0

    def __enter__(self) -> "DescriptorCache":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def close(self) -> None:
        """关连接；重复调用无副作用。
        Close the connection; calling twice is harmless.

        中文：用途：让 `with` 与显式关闭都能用。输入：无。输出：无。
              数据来源：无。配置来源：无。副作用：关闭句柄。异常：无。
        English: Purpose: support both `with` and an explicit close. Args: none. Returns: none.
        Data source: none. Config: none. Side effects: closes the handle. Raises: none.
        """
        self.connection.close()

    def _initialise_or_validate(self) -> None:
        """建表并校验 provenance；不匹配就抛。
        Create the tables and validate provenance; raise on a mismatch.

        中文：
            用途：缓存是最容易被"顺手清一下"的东西，也是最容易把旧结果当新结果的地方。
                  把 provenance 写进 metadata 表，打开时逐字段比对。
            输入：无。输出：无。数据来源：sqlite 文件。配置来源：`provenance()`。
            副作用：建表/写 metadata。异常：ValueError —— 不一致。
        English:
            Purpose: a cache is the easiest thing to clear casually and the easiest place for a
                     stale result to pass as fresh. Provenance is stored in a metadata table and
                     compared field by field on open.
            Args: none. Returns: none. Data source: the sqlite file. Config: `provenance()`.
            Side effects: creates tables and writes metadata. Raises: ValueError on mismatch.
        """
        self.connection.execute(
            "CREATE TABLE IF NOT EXISTS metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL)"
        )
        self.connection.execute(
            "CREATE TABLE IF NOT EXISTS structures ("
            "psmiles TEXT PRIMARY KEY, descriptors TEXT NOT NULL, fingerprint BLOB NOT NULL)"
        )
        expected = provenance()
        row = self.connection.execute(
            "SELECT value FROM metadata WHERE key = 'provenance'"
        ).fetchone()
        if row is None:
            self.connection.execute(
                "INSERT INTO metadata(key, value) VALUES ('provenance', ?)",
                (json.dumps(expected, sort_keys=True),),
            )
            self.connection.commit()
            return
        actual = json.loads(row[0])
        if actual != expected:
            changed = sorted(
                key for key in expected if actual.get(key) != expected.get(key)
            )
            raise ValueError(
                f"描述符缓存的 provenance 与当前环境不一致：{self.path}；"
                f"变化字段={changed}。删掉这个缓存重跑即可（它不会入库，删了不损失证据）。"
            )

    def records(self, values: Sequence[str]) -> tuple[list[dict[str, Any]], dict[str, int]]:
        """按输入顺序返回每个结构的表征记录，未命中的一次算完写回。
        Return one record per input in input order, computing misses once and writing them back.

        中文：
            用途：编排层的唯一取数口。重复结构（同一个规范式出现两次）这里只算一次。
            输入：values —— 规范 PSMILES 序列（可含重复）。
            输出：(记录列表, {"unique", "hits", "misses", "unparsable"})；
                  记录与 `compute()` 同结构，解析失败的位置是 `{"psmiles_canonical": None}`。
            数据来源：sqlite 缓存 + RDKit。配置来源：`provenance()`。
            副作用：把新算的结果写进缓存。异常：无（解析失败计数，不抛）。

        English:
            Purpose: the orchestration layer's only read path; a structure seen twice is computed
                     once.
            Args: values - canonical PSMILES strings, duplicates allowed.
            Returns: (records, {"unique", "hits", "misses", "unparsable"}); each record mirrors
            `compute()`, and an unparsable position is `{"psmiles_canonical": None}`.
            Data source: the sqlite cache plus RDKit. Config: `provenance()`.
            Side effects: writes newly computed rows. Raises: none (failures are counted).
        """
        unique = list(dict.fromkeys(values))
        cached: dict[str, dict[str, Any]] = {}
        for start in range(0, len(unique), 900):
            batch = unique[start : start + 900]
            placeholders = ",".join("?" for _ in batch)
            rows = self.connection.execute(
                "SELECT psmiles, descriptors, fingerprint FROM structures "
                f"WHERE psmiles IN ({placeholders})",
                batch,
            ).fetchall()
            for psmiles, descriptors, fingerprint in rows:
                cached[psmiles] = {
                    "psmiles_canonical": psmiles,
                    "descriptors": json.loads(descriptors),
                    "fingerprint": fingerprint,
                }

        hits = len(cached)
        missing = [value for value in unique if value not in cached]
        unparsable = 0
        fresh: list[tuple[str, str, bytes]] = []
        for value in missing:
            result = compute(value)
            if result is None:
                unparsable += 1
                cached[value] = {"psmiles_canonical": None}
                continue
            # 非有限值存成 JSON null，读回来再还原成 NaN：合法 JSON，且不把 NaN 当作数值写入。
            # Non-finite values become JSON null and are restored as NaN: valid JSON, and no NaN
            # pretends to be a number in the file.
            payload = {
                name: (None if not math.isfinite(number) else number)
                for name, number in result["descriptors"].items()
            }
            cached[value] = {
                "psmiles_canonical": result["psmiles_canonical"],
                "descriptors": payload,
                "fingerprint": result["fingerprint"],
            }
            fresh.append((value, json.dumps(payload, sort_keys=True), result["fingerprint"]))
        if fresh:
            self.connection.executemany(
                "INSERT OR REPLACE INTO structures(psmiles, descriptors, fingerprint) "
                "VALUES (?, ?, ?)",
                fresh,
            )
            self.connection.commit()

        self.hits += hits
        self.misses += len(missing) - unparsable
        records = [cached[value] for value in values]
        for record in records:
            if record.get("descriptors") is not None:
                record["descriptors"] = {
                    name: (math.nan if number is None else number)
                    for name, number in record["descriptors"].items()
                }
        stats = {
            "unique": len(unique),
            "hits": hits,
            "misses": len(missing) - unparsable,
            "unparsable": unparsable,
        }
        return records, stats

    def stored_keys(self) -> set[str]:
        """返回缓存里已存的全部键，供 `--verify` 判断"要不要重算"。
        Return every key already stored, so `--verify` can decide what to recompute.

        中文：用途：只读自检口。输入：无。输出：键集合。
              数据来源：sqlite。配置来源：无。副作用：无。异常：无。
        English: Purpose: a read-only introspection point. Args: none. Returns: the key set.
        Data source: sqlite. Config: none. Side effects: none. Raises: none.
        """
        return {
            row[0] for row in self.connection.execute("SELECT psmiles FROM structures").fetchall()
        }


def nan_count(records: Iterable[dict[str, Any]]) -> int:
    """数一组记录里有多少个 NaN 单元，写进清单（B 的特征装配要报缺失数）。
    Count NaN cells across records, for the manifest (B reports missing counts too).

    中文：
        用途：205 列里有些列对某些分子天然是 NaN（不只是被丢掉的 4 个电荷列，
              例如 `NumAmideBonds` 之外的环相关量在极端结构上也会缺）。这个数字进清单，
              下游就能判断"NaN 是我们造成的还是本来就有的"。
        输入：records —— `DescriptorCache.records()` 的输出。
        输出：int 个 NaN 单元。
        数据来源：调用方。配置来源：无。副作用：无。异常：无。

    English:
        Purpose: some of the 205 columns are naturally NaN for some molecules (not only the four
                 dropped charge columns). Recording the count lets downstream tell "NaN we caused"
                 from "NaN that was always there".
        Args: records. Returns: the number of NaN cells.
        Data source: the caller. Config: none. Side effects: none. Raises: none.
    """
    total = 0
    for record in records:
        descriptors = record.get("descriptors") or {}
        total += sum(1 for number in descriptors.values() if not math.isfinite(number))
    return total
