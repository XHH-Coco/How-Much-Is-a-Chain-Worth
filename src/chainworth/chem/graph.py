"""分子图构建：把 PSMILES 变成一个确定性的原子/键张量组，与 chemprop v2 的默认特征器逐位对齐。
Molecular graph construction: turn a PSMILES into a deterministic set of atom/bond tensors that
matches chemprop v2's default featurizer bit for bit.

中文：
    用途：B-1 的第二件产物，也是 A3 图臂的唯一建图入口。三件事：
          ① **确定性**：先 `MolToSmiles(MolFromSmiles(x))` 再建图，所以"同一根链的两种写法"得到
             完全相同的原子顺序与 `edge_index`。不靠字典序、不靠集合遍历顺序——验收③查的就是这条。
          ② **与官方同口径**：原子特征 72 维、键特征 14 维，逐位复现 chemprop 2.1.0 的
             `SimpleMoleculeMolGraphFeaturizer`（`MultiHotAtomFeaturizer.v2` + `MultiHotBondFeaturizer`）。
             这不是"凭记忆写"：`verify_against_chemprop()` 在 GNN 环境里拿官方 featurizer 逐位对拍，
             `tests/test_task_assembly.py` 每次跑都会调它。官方源码位置见下方"对拍依据"。
          ③ **不依赖 torch**：本模块只产出 numpy 张量；`to_pyg()` 里才 `import torch`（惰性）。
             这样 core 环境（没有 torch）也能 import 本模块、跑装配与测试。
    依赖：rdkit（2024.03.5）。torch/torch_geometric 只在 `to_pyg()` 内部导入。
    配置：`ATOM_FEATURE_DIM` / `BOND_FEATURE_DIM` / `GRAPH_ATTR_DIM` 三个常量——它们决定图的形状，
          改任何一个都等于换了一套 A3 的输入，必须记 DECISIONS 并重跑全部 GPU 实验。
    对外：`build_graph()`、`build_graphs()`、`to_pyg()`、`verify_against_chemprop()`、
          `GRAPH_ATTR_NAMES`、`ATOM_FEATURE_DIM`、`BOND_FEATURE_DIM`。
    被谁用：`src/chainworth/arms/graph.py`（B-3）、`scripts/03b_assemble.py --graphs`、
            `tests/test_task_assembly.py`。

English:
    Purpose: B-1's second artefact and the only graph-building entry point for the A3 arm. Three
    jobs: (1) **determinism** - build from `MolToSmiles(MolFromSmiles(x))`, so two spellings of one
    chain yield exactly the same atom order and `edge_index`; no dictionary order and no set
    iteration order is involved, which is what criterion 3 checks; (2) **the official convention** -
    72 atom features and 14 bond features, reproducing chemprop 2.1.0's
    `SimpleMoleculeMolGraphFeaturizer` (`MultiHotAtomFeaturizer.v2` + `MultiHotBondFeaturizer`)
    bit for bit. This is not written from memory: `verify_against_chemprop()` compares against the
    official featurizer in the GNN environment and `tests/test_task_assembly.py` calls it on every
    run; (3) **no torch dependency** - this module produces numpy tensors only and `to_pyg()`
    imports torch lazily, so the core environment (no torch) can still import it and run assembly.
    Depends on: rdkit (2024.03.5). Config: `ATOM_FEATURE_DIM` / `BOND_FEATURE_DIM` /
    `GRAPH_ATTR_DIM` define the graph's shape; changing any of them swaps A3's input and requires a
    DECISIONS entry plus a full GPU re-run.
    Public API: `build_graph()`, `build_graphs()`, `to_pyg()`, `verify_against_chemprop()`,
    `GRAPH_ATTR_NAMES` and the dimension constants.
    Callers: src/chainworth/arms/graph.py (B-3), scripts/03b_assemble.py --graphs,
    tests/test_task_assembly.py.

对拍依据 ／ Where the parity target lives
----------------------------------------
chemp 2.1.0 安装目录下三个文件就是口径来源（本机路径
`D:/Miniconda3/envs/DSC6020_Project_GNN/Lib/site-packages/chemprop/`）：

- `featurizers/atom.py`：`MultiHotAtomFeaturizer.v2()` = 原子序数 1–36 + 53、度 0–5、形式电荷
  [-1,-2,1,2,0]、手性 0–3、氢数 0–4、杂化 [S,SP,SP2,SP2D,SP3,SP3D,SP3D2]，每项多一位 unknown pad；
  末尾 aromatic（1 位）与 0.01×质量（1 位）。合计 38+7+6+5+6+8+1+1 = **72**。
- `featurizers/bond.py`：`MultiHotBondFeaturizer` = null(1) + 键型(4) + 共轭(1) + 环内(1) +
  立体化学(6+1 pad) = **14**。
- `featurizers/molgraph/molecule.py`：`edge_index` 按 `GetBonds()` 顺序双向展开，
  `rev_edge_index = arange(2E).reshape(-1,2)[:, ::-1].ravel()`。

The three files under chemprop 2.1.0's install directory are the source of this convention:
`atom.py` (v2 multi-hot: element 1-36 + 53, degree 0-5, formal charge [-1,-2,1,2,0], chirality
0-3, num H 0-4, hybridisation [S,SP,SP2,SP2D,SP3,SP3D,SP3D2], each with an unknown pad, then
aromatic and 0.01*mass; 38+7+6+5+6+8+1+1 = 72), `bond.py` (null, bond type, conjugated, in ring,
stereo with pad = 14) and `molgraph/molecule.py` (edge_index doubles each bond in `GetBonds()`
order; rev_edge_index is `arange(2E).reshape(-1,2)[:, ::-1].ravel()`).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Sequence

import numpy as np
from rdkit import Chem, RDLogger
from rdkit.Chem import Descriptors
from rdkit.Chem.rdchem import BondType, HybridizationType

RDLogger.DisableLog("rdApp.*")

# 特征维度：写进清单，改了就等于换了 A3 的输入。
# Feature dimensions: recorded in the manifest; a change means A3's input changed.
ATOM_FEATURE_DIM = 72
BOND_FEATURE_DIM = 14
GRAPH_ATTR_DIM = 5

# 图级属性的名字与顺序（与 `graph_attr` 一一对应）。`has_wpsmiles` 是 0/1 而不是布尔——
# 下游会把它当一列数字拼进图级特征，布尔在这里没有优势、只会让 dtype 变得难预测。
# Graph-level attribute names, in order. `has_wpsmiles` is 0/1 rather than a bool because
# downstream concatenates it as a numeric column; a bool buys nothing and makes dtype unpredictable.
GRAPH_ATTR_NAMES = (
    "n_atoms",
    "n_bonds",
    "n_stars",
    "repeat_unit_mass_over_100",
    "has_wpsmiles",
)

# 与 chemprop v2 逐字相同的六个子特征的取值表（顺序即位的顺序，改顺序等于换特征空间）。
# The six sub-feature vocabularies, identical to chemprop v2 (the order is the bit order, so
# reordering equals a different feature space).
_ATOMIC_NUMS: tuple[int, ...] = tuple(range(1, 37)) + (53,)
_DEGREES: tuple[int, ...] = tuple(range(6))
_FORMAL_CHARGES: tuple[int, ...] = (-1, -2, 1, 2, 0)
_CHIRAL_TAGS: tuple[int, ...] = tuple(range(4))
_NUM_HS: tuple[int, ...] = tuple(range(5))
_HYBRIDIZATIONS: tuple[HybridizationType, ...] = (
    HybridizationType.S,
    HybridizationType.SP,
    HybridizationType.SP2,
    HybridizationType.SP2D,
    HybridizationType.SP3,
    HybridizationType.SP3D,
    HybridizationType.SP3D2,
)
_BOND_TYPES: tuple[BondType, ...] = (
    BondType.SINGLE,
    BondType.DOUBLE,
    BondType.TRIPLE,
    BondType.AROMATIC,
)
_STEREOS: tuple[int, ...] = tuple(range(6))

# 各子块的起始位置（一次算好，避免每个原子重复累加）。
# Where each sub-block starts (computed once instead of re-accumulated per atom).
_ATOM_SLICES = (
    (0, len(_ATOMIC_NUMS)),
    (len(_ATOMIC_NUMS) + 1, len(_DEGREES)),
    (len(_ATOMIC_NUMS) + 1 + len(_DEGREES) + 1, len(_FORMAL_CHARGES)),
    (len(_ATOMIC_NUMS) + 1 + len(_DEGREES) + 1 + len(_FORMAL_CHARGES) + 1, len(_CHIRAL_TAGS)),
    (
        len(_ATOMIC_NUMS)
        + 1
        + len(_DEGREES)
        + 1
        + len(_FORMAL_CHARGES)
        + 1
        + len(_CHIRAL_TAGS)
        + 1,
        len(_NUM_HS),
    ),
    (
        len(_ATOMIC_NUMS)
        + 1
        + len(_DEGREES)
        + 1
        + len(_FORMAL_CHARGES)
        + 1
        + len(_CHIRAL_TAGS)
        + 1
        + len(_NUM_HS)
        + 1,
        len(_HYBRIDIZATIONS),
    ),
)
_AROMATIC_INDEX = sum(size + 1 for _, size in _ATOM_SLICES)
_MASS_INDEX = _AROMATIC_INDEX + 1


@dataclass
class GraphTensors:
    """一个分子的图：原子/键特征 + 边索引 + 图级属性。
    One molecule's graph: atom/bond features, edge indices and graph-level attributes.

    中文：
        用途：取代 PyG 的 `Data` 作为**模块间**的交换格式——它是纯 numpy，core 环境能造、能测。
              PyG 的 `Data` 由 `to_pyg()` 现转，转换代码只有一处，B-3 不许自己拼。
        字段：`atom` (n,72) float32、`bond` (2E,14) float32、`edge_index` (2,2E) int64、
              `rev_edge_index` (2E,) int64、`n_atoms`、`n_bonds`、`n_stars`、`repeat_unit_mass`、
              `psmiles`（规范式）、`wpsmiles`（可能为 None）、`graph_attr` (5,) float32。
        数据来源：`build_graph()`。配置来源：本模块的维度常量。副作用：无。

    English:
        Purpose: replaces PyG's `Data` as the **inter-module** exchange format, because it is pure
        numpy and can therefore be built and tested in the core environment. `to_pyg()` converts
        on demand, and it is the only conversion - B-3 must not assemble one itself.
        Fields: atom (n,72), bond (2E,14), edge_index (2,2E), rev_edge_index, n_atoms, n_bonds,
        n_stars, repeat_unit_mass, psmiles (canonical), wpsmiles (may be None), graph_attr (5,).
        Data source: `build_graph()`. Config: this module's dimension constants.
    """

    atom: np.ndarray
    bond: np.ndarray
    edge_index: np.ndarray
    rev_edge_index: np.ndarray
    n_atoms: int
    n_bonds: int
    n_stars: int
    repeat_unit_mass: float
    psmiles: str
    wpsmiles: str | None = None
    graph_attr: np.ndarray = field(default_factory=lambda: np.zeros((GRAPH_ATTR_DIM,), np.float32))


def canonical_mol(psmiles: str) -> Chem.Mol | None:
    """把任意写法规范化后再解析，得到**原子顺序稳定**的分子；解析失败返回 None。
    Re-canonicalise any spelling before parsing, giving a molecule with a **stable atom order**.

    中文：
        用途：这是验收③的地基。RDKit 的原子顺序由输入串决定，而同一根链有多种合法写法
              （`*CC(*)C` 与 `*CC(C)*`），直接解析会得到两套原子编号——`edge_index` 于是不同，
              而下游没有任何检查会发现"这两行其实是同一个结构"。
              先 `MolToSmiles` 再 `MolFromSmiles`，两种写法就都变成同一个串、同一套编号。
              **不用 `Chem.MolFromSmiles` 的默认 sanitize 之外的任何处理**：`*` 原样保留（与 A-6 的
              描述符口径一致，见 D-25），不封端、不补氢。
        输入：psmiles —— 任意 PSMILES 写法。
        输出：RDKit Mol，或 None（解析失败 / 空分子）。
        数据来源：调用方。配置来源：无。副作用：无。异常：无。

    English:
        Purpose: the foundation of criterion 3. RDKit's atom order follows the input string, and one
        chain has several legal spellings (`*CC(*)C` vs `*CC(C)*`); parsing them directly yields two
        atom numberings, hence two different `edge_index` values - and nothing downstream would
        notice that the two rows are the same structure. A `MolToSmiles` round trip makes both
        spellings one string with one numbering. Nothing beyond RDKit's default sanitisation
        happens: `*` stays as-is (A-6's descriptor convention, D-25), no capping, no implicit-H
        filling.
        Args: psmiles. Returns: a Mol, or None when unparsable or empty.
        Data source: the caller. Side effects: none. Raises: none.
    """
    text = "" if psmiles is None else str(psmiles).strip()
    if not text:
        return None
    molecule = Chem.MolFromSmiles(text)
    if molecule is None or molecule.GetNumAtoms() == 0:
        return None
    canonical = Chem.MolToSmiles(molecule)
    return Chem.MolFromSmiles(canonical)


def atom_features(molecule: Chem.Mol) -> np.ndarray:
    """72 维多热原子特征（逐位复现 chemprop v2）。/ 72-dim multi-hot atom features (chemprop v2).

    中文：
        用途：**每一项都带 unknown pad**（除 aromatic 与 mass）。pad 不是"凑数"：`*` 的原子序数是 0，
              不在 1–36∪{53} 里，它必须落在一个合法位上，否则整个特征会错位一列。
        输入：molecule —— 已解析的分子。
        输出：shape (n_atoms, 72) 的 float32。
        数据来源：RDKit 原子属性。配置来源：本模块的取值表与 `ATOM_FEATURE_DIM`。
        副作用：无。异常：无。

    English:
        Purpose: every sub-feature has an unknown pad except aromaticity and mass. The pad is not
        filler: `*` has atomic number 0, which is not in 1-36 ∪ {53}, so it must land on a legal bit
        or the whole vector shifts by one column.
        Args: molecule. Returns: an (n_atoms, 72) float32 array.
        Data source: RDKit atom properties. Config: this module's vocabularies.
    """
    n_atoms = molecule.GetNumAtoms()
    features = np.zeros((n_atoms, ATOM_FEATURE_DIM), dtype=np.float32)
    atomic_index = {value: index for index, value in enumerate(_ATOMIC_NUMS)}
    charge_index = {value: index for index, value in enumerate(_FORMAL_CHARGES)}
    hybridization_index = {value: index for index, value in enumerate(_HYBRIDIZATIONS)}
    for atom in molecule.GetAtoms():
        row = features[atom.GetIdx()]
        values = (
            atomic_index.get(atom.GetAtomicNum(), len(_ATOMIC_NUMS)),
            _DEGREES.index(atom.GetTotalDegree())
            if atom.GetTotalDegree() in _DEGREES
            else len(_DEGREES),
            charge_index.get(atom.GetFormalCharge(), len(_FORMAL_CHARGES)),
            int(atom.GetChiralTag())
            if int(atom.GetChiralTag()) in _CHIRAL_TAGS
            else len(_CHIRAL_TAGS),
            int(atom.GetTotalNumHs()) if int(atom.GetTotalNumHs()) in _NUM_HS else len(_NUM_HS),
            hybridization_index.get(atom.GetHybridization(), len(_HYBRIDIZATIONS)),
        )
        for (start, _), value in zip(_ATOM_SLICES, values):
            row[start + int(value)] = 1.0
        row[_AROMATIC_INDEX] = float(atom.GetIsAromatic())
        row[_MASS_INDEX] = 0.01 * atom.GetMass()
    return features


def bond_features(molecule: Chem.Mol) -> np.ndarray:
    """14 维多热键特征，**每条键写两遍**（正反各一行），与 `edge_index` 的顺序一一对应。
    14-dim multi-hot bond features, **written twice per bond** (forward and reverse), matching the
    `edge_index` order.

    中文：
        用途：chemprop 的图是无向图用双向边表示，所以 `bond` 的行数必须是 `2 × n_bonds`；
              写成 `n_bonds` 行再让调用方去复制，是这类代码最常见的错位来源。
        输入：molecule —— 已解析的分子。
        输出：shape (2 × n_bonds, 14) 的 float32。
        数据来源：RDKit 键属性。配置来源：本模块的取值表与 `BOND_FEATURE_DIM`。
        副作用：无。异常：无。

    English:
        Purpose: chemprop represents an undirected graph with bidirectional edges, so `bond` must
        have 2 × n_bonds rows; writing n_bonds rows and letting the caller duplicate them is the
        most common source of misalignment in code like this.
        Args: molecule. Returns: a (2 × n_bonds, 14) float32 array.
        Data source: RDKit bond properties. Config: this module's vocabularies.
    """
    n_bonds = molecule.GetNumBonds()
    features = np.zeros((2 * n_bonds, BOND_FEATURE_DIM), dtype=np.float32)
    for bond in molecule.GetBonds():
        row = np.zeros((BOND_FEATURE_DIM,), dtype=np.float32)
        row[0] = 0.0                                  # 不是"空键"：null 位只对 pad 置 1
        bond_type = bond.GetBondType()
        row[1 + (_BOND_TYPES.index(bond_type) if bond_type in _BOND_TYPES else len(_BOND_TYPES))] = 1.0
        row[5] = float(bond.GetIsConjugated())
        row[6] = float(bond.IsInRing())
        stereo = int(bond.GetStereo())
        row[7 + (stereo if stereo in _STEREOS else len(_STEREOS))] = 1.0
        # 双向边：`edge_index` 是 [u,v] 然后 [v,u]，特征也按同一顺序写两行。
        # Bidirectional: edge_index holds [u,v] then [v,u], and the features follow the same order.
        features[2 * bond.GetIdx()] = row
        features[2 * bond.GetIdx() + 1] = row
    return features


def build_graph(psmiles: str, wpsmiles: str | None = None) -> GraphTensors | None:
    """把一个 PSMILES 建成图；解析失败返回 None（调用方负责记账，不许静默丢弃）。
    Build one PSMILES into a graph; None when it cannot be parsed (the caller records it).

    中文：
        用途：A3 臂的唯一建图口。**确定性**来自 `canonical_mol()`：两种写法 → 同一个规范串 →
              同一套原子编号 → 同一个 `edge_index`。
              图级属性里的 `repeat_unit_mass` 是**重复单元质量**：`*` 的原子序数为 0、质量按 0 算，
              所以这个数就是"一个重复单元多重"，正是 A5 那类链信息要用的量；除以 100 是为了让它和
              原子特征里的质量项同量级，别在拼接时把尺度差三个数量级的两列放在一起。
              **没有"每边一个 wPSMILES 权重"**：官方 polymer_chemprop 的图里没有这一项，硬加一个
              就是"我调过的架构"，而红线 2 明令 A3 必须是官方标准配置。wPSMILES 在这里只作为
              "有没有"（`has_wpsmiles`）与字符串本体保留，权重怎么进模型由 A5 决定（那才是它的活）。
        输入：psmiles —— 任意 PSMILES 写法；wpsmiles —— 该结构的 wPSMILES（没有就传 None）。
        输出：`GraphTensors`，或 None。
        数据来源：调用方（结构表 / 调和表）。配置来源：本模块维度常量。
        副作用：无。异常：无。

    English:
        Purpose: the only graph-building entry for the A3 arm. **Determinism** comes from
        `canonical_mol()`: two spellings -> one canonical string -> one atom numbering -> one
        `edge_index`. The graph-level `repeat_unit_mass` is the **repeat unit's** mass (`*` has
        atomic number 0 and contributes 0 mass), i.e. "how heavy one repeat unit is", the quantity
        A5-style chain information needs; dividing by 100 keeps it on the same order as the mass
        term in the atom features instead of mixing columns three orders of magnitude apart.
        There is deliberately **no per-edge wPSMILES weight**: the official polymer_chemprop graph
        has no such term, and adding one would make A3 "the architecture I tuned", which red line 2
        forbids. wPSMILES appears here only as presence (`has_wpsmiles`) and as the string itself;
        how a weight enters a model is A5's decision, not A3's.
        Args: psmiles; wpsmiles (None when absent). Returns: GraphTensors or None.
        Data source: the caller (structure table / harmonised table). Config: the dimension
        constants above. Side effects: none. Raises: none.
    """
    molecule = canonical_mol(psmiles)
    if molecule is None:
        return None
    n_atoms = molecule.GetNumAtoms()
    n_bonds = molecule.GetNumBonds()
    edge_index = np.zeros((2, 2 * n_bonds), dtype=np.int64)
    for bond in molecule.GetBonds():
        begin, end = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        position = 2 * bond.GetIdx()
        edge_index[0, position], edge_index[1, position] = begin, end
        edge_index[0, position + 1], edge_index[1, position + 1] = end, begin
    rev_edge_index = np.arange(2 * n_bonds, dtype=np.int64).reshape(-1, 2)[:, ::-1].ravel()
    n_stars = sum(1 for atom in molecule.GetAtoms() if atom.GetAtomicNum() == 0)
    mass = float(Descriptors.MolWt(molecule))
    attr = np.asarray(
        [n_atoms, n_bonds, n_stars, mass / 100.0, 0.0 if wpsmiles is None else 1.0],
        dtype=np.float32,
    )
    return GraphTensors(
        atom=atom_features(molecule),
        bond=bond_features(molecule),
        edge_index=edge_index,
        rev_edge_index=rev_edge_index,
        n_atoms=n_atoms,
        n_bonds=n_bonds,
        n_stars=n_stars,
        repeat_unit_mass=mass,
        psmiles=Chem.MolToSmiles(molecule),
        wpsmiles=wpsmiles,
        graph_attr=attr,
    )


def to_pyg(graph: GraphTensors) -> Any:
    """把 `GraphTensors` 转成 PyG 的 `Data`（**唯一**的转换点，torch 在这里才被导入）。
    Convert `GraphTensors` into a PyG `Data` object - the **only** conversion point, where torch
    gets imported.

    中文：
        用途：核心代码只认 numpy，训练代码只认 PyG；转换只有这一处，B-3 不许自己拼 `Data`——
              两处拼法一旦不一致（比如一处忘了 `rev_edge_index` 或用错了 dtype），
              模型能训、指标能出，但没有可比性。
        输入：graph —— `build_graph()` 的输出。
        输出：`torch_geometric.data.Data`，字段 `x` / `edge_index` / `edge_attr` /
              `rev_edge_index` / `graph_attr`。
        数据来源：调用方。配置来源：无。副作用：无（新对象）。
        异常：ImportError —— 当前环境没装 torch / torch_geometric（core 环境就是这样，属预期）。

    English:
        Purpose: core code speaks numpy, training code speaks PyG, and conversion happens here only.
        B-3 must not build a `Data` itself: two assembly routines that disagree (one forgetting
        `rev_edge_index`, another using the wrong dtype) still train and still produce metrics that
        are not comparable.
        Args: graph. Returns: a torch_geometric Data with x / edge_index / edge_attr /
        rev_edge_index / graph_attr. Data source: the caller. Side effects: none (a new object).
        Raises: ImportError when torch or torch_geometric is absent (expected in the core env).
    """
    import torch
    from torch_geometric.data import Data

    return Data(
        x=torch.from_numpy(np.ascontiguousarray(graph.atom)),
        edge_index=torch.from_numpy(np.ascontiguousarray(graph.edge_index)),
        edge_attr=torch.from_numpy(np.ascontiguousarray(graph.bond)),
        rev_edge_index=torch.from_numpy(np.ascontiguousarray(graph.rev_edge_index)),
        graph_attr=torch.from_numpy(np.ascontiguousarray(graph.graph_attr)),
    )


def build_graphs(
    psmiles: Iterable[str], wpsmiles: dict[str, str] | None = None
) -> tuple[dict[str, GraphTensors], list[tuple[str, str]]]:
    """批量建图，返回（图字典, 失败清单）；失败逐条给原因，不静默丢弃。
    Build graphs in bulk; return (graphs, failures) with a reason per failure, never a silent drop.

    中文：
        用途：A3 要按结构的**唯一集合**建一次图（20,660 行里只有 11,494 个结构，重复的是行不是图）。
              失败清单是分工_B §2 B-1 验收⑤的落地：解析不了的结构必须**计数并列 ID**。
              没有 wPSMILES 的结构并不失败——图照样能建（`*` 留在图里），只是 `has_wpsmiles=0`。
        输入：psmiles —— 结构串（可重复，内部去重）；wpsmiles —— 规范式 → wPSMILES 的映射。
        输出：(dict[规范式, GraphTensors], [(原始串, 原因)])。
        数据来源：调用方。配置来源：无。副作用：无。异常：无。

    English:
        Purpose: A3 builds one graph per **unique** structure (20,660 rows hold 11,494 structures;
        the repetition is in rows, not graphs). The failure list implements criterion 5 of B-1:
        unparsable structures must be **counted and listed**. A structure without a wPSMILES is not
        a failure - the graph still builds with `*` in place, only `has_wpsmiles=0`.
        Args: psmiles (duplicates allowed, de-duplicated inside); wpsmiles (canonical -> wPSMILES).
        Returns: (dict keyed by canonical form, [(raw, reason)]).
        Data source: the caller. Side effects: none. Raises: none.
    """
    mapping = wpsmiles or {}
    graphs: dict[str, GraphTensors] = {}
    failures: list[tuple[str, str]] = []
    for text in dict.fromkeys(str(value) for value in psmiles):
        if not text:
            failures.append(("", "empty_string"))
            continue
        graph = build_graph(text, mapping.get(text))
        if graph is None:
            failures.append((text, "parse_failed"))
            continue
        graphs[graph.psmiles] = graph
    return graphs, failures


def verify_against_chemprop(psmiles: Sequence[str]) -> list[str]:
    """拿 chemprop 官方 featurizer 逐位对拍（只在装了 chemprop 的环境里能跑）。
    Compare bit for bit against chemprop's official featurizer (needs chemprop installed).

    中文：
        用途：分工_B §2 B-1 明写"复现 chemprop 的 polymer 模式图构建，**用官方脚本对拍一次**，
              不要凭记忆写"。这就是那一次对拍，而且它被写成了测试，所以每次跑都会重做一遍。
              对三样东西：原子特征矩阵、键特征矩阵、`edge_index`。`rev_edge_index` 一并核，
              因为它错位不会让训练报错、只会让消息传递的反向边收错。
        输入：psmiles —— 要核的结构串（测试里用真实数据抽 200 个）。
        输出：问题字符串列表；空列表表示逐位相同。
        数据来源：`chemprop.featurizers.SimpleMoleculeMolGraphFeaturizer`。
        配置来源：chemprop 自身默认参数（atom v2 / bond 默认）。副作用：无。
        异常：ImportError —— 没装 chemprop（core 环境），调用方应据此跳过。

    English:
        Purpose: division B, section 2, B-1 says "reproduce chemprop's polymer-mode graph
        construction and **check it once against the official script**, do not write it from
        memory". This is that check, written as a test so it re-runs every time. Three things are
        compared: the atom matrix, the bond matrix and `edge_index`; `rev_edge_index` is checked
        too, because a mismatch there breaks reverse-edge aggregation without failing the training.
        Args: psmiles (the tests sample 200 from real data). Returns: problem strings; empty means
        bit-identical. Data source: chemprop's own featurizer.
        Side effects: none. Raises: ImportError when chemprop is absent (the core env), which the
        caller should treat as "skip".
    """
    from chemprop.featurizers import SimpleMoleculeMolGraphFeaturizer

    official = SimpleMoleculeMolGraphFeaturizer()
    problems: list[str] = []
    if official.atom_fdim != ATOM_FEATURE_DIM or official.bond_fdim != BOND_FEATURE_DIM:
        problems.append(
            f"chemprop 的维度是 {official.atom_fdim}/{official.bond_fdim}，"
            f"本模块写的是 {ATOM_FEATURE_DIM}/{BOND_FEATURE_DIM}——版本换了，先核口径"
        )
        return problems
    for text in psmiles:
        # 对拍对象是**同一个分子**，不是同一个字符串：官方 featurizer 直接吃输入串，原子顺序随写法走；
        # 我们先把串规范化，所以必须把**规范化后的分子**交给官方 featurizer，否则对拍会把
        # "我们的规范化"当成错误报出来（而规范化正是验收③要求的）。
        # The comparison target is the **same molecule**, not the same string: the official
        # featurizer eats the input string, so its atom order follows the spelling, while we
        # canonicalise first. Handing the **canonical molecule** to the official featurizer is
        # therefore right; otherwise this check would report our canonicalisation as a bug - the
        # very thing criterion 3 asks for.
        molecule = canonical_mol(str(text))
        if molecule is None:
            continue
        expected = official(molecule)
        ours = build_graph(str(text))
        if ours is None:
            problems.append(f"{str(text)[:40]}…：官方能解析，我们的 build_graph 返回 None")
            continue
        if not np.array_equal(expected.V, ours.atom):
            problems.append(f"{str(text)[:40]}…：原子特征与 chemprop 不一致")
        if not np.array_equal(expected.E, ours.bond):
            problems.append(f"{str(text)[:40]}…：键特征与 chemprop 不一致")
        if not np.array_equal(expected.edge_index, ours.edge_index):
            problems.append(f"{str(text)[:40]}…：edge_index 与 chemprop 不一致")
        if not np.array_equal(expected.rev_edge_index, ours.rev_edge_index):
            problems.append(f"{str(text)[:40]}…：rev_edge_index 与 chemprop 不一致")
    return problems
