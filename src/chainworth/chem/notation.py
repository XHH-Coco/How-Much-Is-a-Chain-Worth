"""聚合物记法转换：PSMILES 规范化与 wPSMILES 生成（纯函数，不读写任何文件）。
Polymer notation: canonicalise PSMILES and derive wPSMILES. Pure functions, no I/O.

中文：
    用途：A-6 的第一、二件产物。把"同一分子的一万种写法"压成一个可做去重键的规范形式，
          再按 PolyBench26 官方的 wPSMILES 约定生成带连接点的字符串（B 建图要用）。
    依赖：rdkit（`environment/core.yml` 已锁 2024.03.5）。无 pandas、无文件、无网络。
    配置：无——这里的规则是**记法定义**，不是实验参数；改它等于改化学口径，必须记 DECISIONS。
    对外：`canonical_psmiles()`、`to_wpsmiles()`、`count_stars()`、`canonicalise_many()`、
          `WPSMILES_SUFFIX`、`HOMOPOLYMER_STAR_COUNT`。
    被谁用：`src/chainworth/chem/structures.py`（编排）、`scripts/02_features.py`（入口）、
            `tests/test_features.py`（断言）。

English:
    Purpose: A-6's first two deliverables. Collapse the many spellings of one molecule into a
    single canonical form usable as a dedupe key, then derive the wPSMILES string (attachment
    points plus connectivity suffix) using PolyBench26's own convention, which stage B needs
    when it builds graphs.
    Depends on: rdkit (pinned 2024.03.5 by environment/core.yml). No pandas, no files, no network.
    Config: none - these are *notation definitions*, not experiment parameters. Changing one
    changes chemical identity, so it must be recorded in docs/DECISIONS.md.
    Public API: `canonical_psmiles()`, `to_wpsmiles()`, `count_stars()`, `canonicalise_many()`,
    `WPSMILES_SUFFIX`, `HOMOPOLYMER_STAR_COUNT`.
    Callers: src/chainworth/chem/structures.py (orchestration), scripts/02_features.py (entry),
    tests/test_features.py (assertions).

关于 PSMILES 里的 `*` ／ On the asterisks in a PSMILES
------------------------------------------------------
`*` 在 PolyBench 的表里表示**重复单元两端的连接键**（RDKit 里是原子序数为 0 的 wildcard 原子）。
它不是一个可以随便丢掉的装饰：丢掉它分子就从"聚合物重复单元"变成"一个小分子"，
所有描述符随之改变。所以本模块**永远保留** `*`，并且描述符也在保留 `*` 的图上计算
（与 PolyBench26 官方 `PSMILES_to_RDKit_descriptors.py` 完全一致；见 DECISIONS D-25）。

`*` marks the two open bonds of the repeat unit (in RDKit it is an atomic-number-0 wildcard).
It is not decoration: dropping it turns a repeat unit into a small molecule and every descriptor
changes. This module therefore never removes it, and descriptors are computed on the graph with
the wildcards in place, exactly as PolyBench26's official `PSMILES_to_RDKit_descriptors.py` does
(see DECISIONS D-25).
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence

from rdkit import Chem, RDLogger

# RDKit 的解析器很爱往 stderr 打 warning；A-3 已经关过一次，这里同样关掉，
# 但**失败仍然计数**——日志静音不等于问题静音，计数在调用方。
# RDKit's parser is chatty on stderr. A-3 silenced it and so does this module, but failures are
# still counted: a muted log is not a muted problem; the counting lives in the caller.
RDLogger.DisableLog("rdApp.*")

# 重复单元两端各一个连接点，这是"均聚物重复单元"的定义。
# 少于 2 个 = 开链片段；多于 2 个 = 支化/交联/嵌段，wPSMILES 的单重复单元写法表达不了。
# Two open bonds, one per end, is the definition of a homopolymer repeat unit. Fewer than two is
# an open fragment; more than two is branching/crosslinking/block structure, which the
# single-repeat-unit wPSMILES form cannot express.
HOMOPOLYMER_STAR_COUNT = 2

# 官方 `polymer_conversions.PSMILES_to_wPSMILES()` 的后缀，逐字照抄：
# `f"{replaced_smi}|1|<1-2:1:1<1-1:0:0<2-2:0:0"`
# 含义：1 个重复单元；1-2 之间成键、权重 1:1；1-1 与 2-2 是"不与自己相连"（权重 0）。
# The official suffix, copied verbatim from `polymer_conversions.PSMILES_to_wPSMILES()`:
# one repeat unit; bonds 1-2 with weight 1:1; 1-1 and 2-2 declare "not bonded to itself" (weight 0).
WPSMILES_SUFFIX = "|1|<1-2:1:1<1-1:0:0<2-2:0:0"

# 官方脚本先试 `[*]` 再退到 `*`；我们的输入来自 RDKit 的 canonical SMILES，只会出现裸 `*`，
# 因此只用裸 `*` 做切分，并在测试里与官方实现逐字对拍（tests/test_features.py）。
# The official script tries `[*]` then falls back to `*`. Our input is RDKit canonical SMILES,
# which only ever contains bare `*`, so splitting on `*` suffices; a test compares our output
# with a verbatim copy of the official implementation.


def count_stars(psmiles: str) -> int:
    """数一个 PSMILES 里有几个连接点（`*` 的个数）。
    Count the attachment points (asterisks) in a PSMILES.

    中文：
        用途：判断这个结构能不能写成 wPSMILES。**必须数，不许 `str.count` 之外做假设**——
              实测 PolyBench 的 20,660 行里有 199 个唯一结构不是 2 个星号（3/4/6 个），
              它们是支化或嵌段结构，不是数据错误。
        输入：psmiles —— 任意 SMILES/PSMILES 字符串。
        输出：整数个数（0 表示普通小分子）。
        数据来源：调用方。配置来源：无。副作用：无。异常：无。

    English:
        Purpose: decide whether a wPSMILES can be written at all. It must be counted rather than
                 assumed: 199 of PolyBench's 11,494 unique structures carry 3, 4 or 6 asterisks
                 (branched or block structures), and they are not data errors.
        Args: psmiles - any SMILES/PSMILES string. Returns: the count (0 for a plain molecule).
        Data source: the caller. Config: none. Side effects: none. Raises: none.
    """
    return psmiles.count("*")


def canonical_psmiles(smiles: str) -> str | None:
    """把一条 SMILES/PSMILES 规范成 RDKit 的 canonical 形式；解析不了返回 None。
    Canonicalise one SMILES/PSMILES with RDKit; return None when it cannot be parsed.

    中文：
        用途：去重键不能用原始字符串——`*CC(C)*` 与 `*CC(*)C` 是同一个分子（三个碳、两个
              连接点），只有规范形式才认得出。同一份规范形式也是 B 建图的输入。
              与 A-3 的 `harmonise._canonicalise_smiles()` 用同一套 RDKit 默认参数
              （不做去盐、不做标准化，因为两者都会改变分子身份），A-6 会**逐行复核**两边一致。
        输入：smiles —— 已 strip 的字符串。
        输出：canonical SMILES；字符串为空或解析失败返回 None。
        数据来源：`data/interim/harmonised/*.parquet` 的 `smiles_raw`。
        配置来源：无（RDKit 默认参数）。副作用：无（RDKit 日志已在模块级关闭）。
        异常：无——非法输入不抛，返回 None 由调用方计数。

    English:
        Purpose: the dedupe key cannot be the raw string - `*CC(C)*` and `*CC(*)C` are the same
                 molecule (three carbons, two attachment points) and only the canonical form sees
                 that. The same canonical form is what B builds graphs from. Parameters match
                 A-3's `harmonise._canonicalise_smiles()` exactly (no salt stripping, no
                 standardisation: both would change molecular identity); A-6 re-checks row by row
                 that the two agree.
        Args: smiles - a stripped string. Returns: canonical SMILES, or None if empty/unparsable.
        Data source: the `smiles_raw` column of data/interim/harmonised/*.parquet.
        Config: none (RDKit defaults). Side effects: none (logging disabled module-wide).
        Raises: none - invalid input yields None and the caller counts it.
    """
    if not smiles or not smiles.strip():
        return None
    molecule = Chem.MolFromSmiles(smiles.strip())
    if molecule is None:
        return None
    return Chem.MolToSmiles(molecule)


def canonicalise_many(values: Iterable[str]) -> tuple[list[str | None], int]:
    """批量规范化，返回 (结果列表, 失败个数)；失败位置为 None，长度与输入一致。
    Canonicalise in bulk: (results, failure count); failures are None, length preserved.

    中文：
        用途：给编排层一个"不抛异常、只计数"的批处理口，避免一条坏 SMILES 中断整个阶段。
        输入：values —— 字符串序列。
        输出：(list[str | None], int)。
        数据来源：调用方。配置来源：无。副作用：无。异常：无。

    English:
        Purpose: a bulk entry that counts instead of raising, so one bad SMILES cannot abort the
                 whole stage.
        Args: values. Returns: (list, failure count). Data source: the caller.
        Config: none. Side effects: none. Raises: none.
    """
    results: list[str | None] = []
    failures = 0
    for value in values:
        canonical = canonical_psmiles(value)
        if canonical is None:
            failures += 1
        results.append(canonical)
    return results, failures


def to_wpsmiles(psmiles: str) -> str | None:
    """把 PSMILES 写成 wPSMILES；星号数不是 2 时返回 None（**不猜**）。
    Convert a PSMILES to wPSMILES; return None unless it has exactly two asterisks.

    中文：
        用途：复现 PolyBench26 官方的 `PSMILES_to_wPSMILES()`：
              把两个 `*` 换成 `[*:1]` 与 `[*:2]`（按出现顺序），再拼上连接关系后缀。
              官方实现用 `split('*', 2)`，**星号多于两个时会把多出来的留在字符串里**，
              仍然照拼后缀——产出一个语法上像 wPSMILES、化学上是错的东西。
              这里改成显式返回 None：宁可少一个字段，也不产出一个会被下游当真值的假串。
        输入：psmiles —— 已规范化的 PSMILES（`*` 为裸星号）。
        输出：wPSMILES 字符串；星号数 != 2 时返回 None。
        数据来源：`canonical_psmiles()` 的输出。
        配置来源：`HOMOPOLYMER_STAR_COUNT` 与 `WPSMILES_SUFFIX`。
        副作用：无。异常：无。

    English:
        Purpose: reproduce PolyBench26's official `PSMILES_to_wPSMILES()`: replace the two
                 asterisks with `[*:1]` and `[*:2]` in order of appearance, then append the
                 connectivity suffix. The official implementation splits with maxsplit=2, so a
                 string with more than two asterisks keeps the extras and still gets the suffix -
                 a string that looks like a wPSMILES but is chemically wrong. This returns None
                 instead: losing a field beats emitting a fake that downstream code will trust.
        Args: psmiles - a canonical PSMILES (bare asterisks).
        Returns: the wPSMILES string, or None when the asterisk count is not two.
        Data source: the output of `canonical_psmiles()`.
        Config: `HOMOPOLYMER_STAR_COUNT`, `WPSMILES_SUFFIX`.
        Side effects: none. Raises: none.
    """
    if count_stars(psmiles) != HOMOPOLYMER_STAR_COUNT:
        return None
    head, middle, tail = psmiles.split("*")
    return f"{head}[*:1]{middle}[*:2]{tail}{WPSMILES_SUFFIX}"


def describe(psmiles: str) -> dict[str, object]:
    """给出一个结构的记法画像，供结构表与失败清单共用。
    Return the notation profile of one structure, shared by the structure table and the failures.

    中文：
        用途：把"这条结构能不能进建模"的判断集中在一处，避免结构表和失败清单各算一遍、
              各算错一处。字段含义：
              `psmiles_canonical`（规范形式，去重键）、`n_stars`（连接点数）、
              `wpsmiles`（星号数为 2 时才有）、`wpsmiles_reason`（没有时的原因，便于写清单）。
        输入：psmiles —— 原始或已规范化的 PSMILES 字符串。
        输出：dict；解析失败时 `psmiles_canonical` 为 None、`wpsmiles` 为 None。
        数据来源：调用方。配置来源：无。副作用：无。异常：无。

    English:
        Purpose: keep "can this structure enter modelling?" in one place, so the structure table
                 and the failure list cannot disagree. Fields: `psmiles_canonical` (dedupe key),
                 `n_stars`, `wpsmiles` (only when the count is two) and `wpsmiles_reason`.
        Args: psmiles - a raw or canonical PSMILES. Returns: a dict; for an unparsable input both
        `psmiles_canonical` and `wpsmiles` are None.
        Data source: the caller. Config: none. Side effects: none. Raises: none.
    """
    canonical = canonical_psmiles(psmiles)
    if canonical is None:
        return {
            "psmiles_canonical": None,
            "n_stars": None,
            "wpsmiles": None,
            "wpsmiles_reason": "parse_failed",
        }
    n_stars = count_stars(canonical)
    if n_stars != HOMOPOLYMER_STAR_COUNT:
        return {
            "psmiles_canonical": canonical,
            "n_stars": n_stars,
            "wpsmiles": None,
            "wpsmiles_reason": f"star_count_{n_stars}_not_{HOMOPOLYMER_STAR_COUNT}",
        }
    return {
        "psmiles_canonical": canonical,
        "n_stars": n_stars,
        "wpsmiles": to_wpsmiles(canonical),
        "wpsmiles_reason": "",
    }


def official_reference_wpsmiles(psmiles: str) -> str:
    """官方实现的逐字副本，**仅供测试对拍**，不要在业务代码里调用。
    A verbatim copy of the official implementation, for differential tests only.

    中文：
        用途：`tests/test_features.py` 拿它当参照物，证明 `to_wpsmiles()` 不是"看起来像"而是
              逐字节相同。放在源码里而不是测试里，是因为它带有必须追溯的来源注释。
              来源：PolyBench26 仓库 `Datasets/Dataset_construction_scripts/polymer_conversions.py`
              的 `replace_astrix()` 与 `PSMILES_to_wPSMILES()`（commit 见 splits/polybench26/
              PROVENANCE.json 同批下载记录；`scripts/00_download.py` 白名单）。
        输入：psmiles —— 恰好两个星号的 PSMILES。
        输出：官方写法得到的字符串。
        数据来源：无。配置来源：无。副作用：无。异常：星号数不是 2 时 AssertionError（故意的）。

    English:
        Purpose: the differential test's reference object, proving `to_wpsmiles()` is byte-equal
                 rather than merely similar. It lives in source (not tests) because it carries a
                 provenance note that has to travel with it.
                 Source: PolyBench26's `Datasets/Dataset_construction_scripts/polymer_conversions.py`,
                 functions `replace_astrix()` and `PSMILES_to_wPSMILES()`.
        Args: psmiles - a PSMILES with exactly two asterisks. Returns: the official spelling.
        Data source: none. Config: none. Side effects: none.
        Raises: AssertionError when the asterisk count is not two (deliberate).
    """
    assert psmiles.count("*") == 2, "官方实现只在两个星号时有定义"
    parts = psmiles.split("*", 2)
    replaced = f"{parts[0]}[*:1]{parts[1]}[*:2]{parts[2]}"
    return f"{replaced}|1|<1-2:1:1<1-1:0:0<2-2:0:0"


def canonical_sort_key(psmiles: Sequence[str]) -> list[str]:
    """把一组结构按"确定性"排序（用于结构表的行序与 structure_id）。
    Sort structures deterministically, for the structure table's row order and ids.

    中文：
        用途：`structure_id` 是行号，行号只有排序稳定才有意义。用 (长度, 字符串) 排序是为了
              让相邻的行在体量上相近，肉眼抽查时好读；但**真正保证稳定的是"总序"**——
              Python 的字符串比较是全序，不依赖 locale。
        输入：psmiles —— 结构字符串序列。
        输出：排序后的新列表（不修改入参）。
        数据来源：调用方。配置来源：无。副作用：无。异常：无。

    English:
        Purpose: `structure_id` is a row number, and a row number only means something if the
                 order is stable. Sorting by (length, string) keeps neighbours similar in size,
                 which helps eyeballing; what actually guarantees stability is that Python's
                 string comparison is a total order independent of locale.
        Args: psmiles. Returns: a new sorted list (the argument is untouched).
        Data source: the caller. Config: none. Side effects: none. Raises: none.
    """
    return sorted(psmiles, key=lambda value: (len(value), value))
