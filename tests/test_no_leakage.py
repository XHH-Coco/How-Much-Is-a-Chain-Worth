"""泄漏总闸：特征里不许藏标签，代码里不许拿全量数据 fit，划分里不许瞒着跨集结构。
Leakage gate: no labels hidden in features, no fitting on full data, no unreported cross-split
structures.

中文：
    用途：分工书 §3 的交付物清单与 §8 自检、A-5 步骤 ④ 都点名了 `tests/test_no_leakage.py`，
          但它管的是**三件互相独立的事**，缺一件都会让后面的 R² 虚高而查不出原因：
          ① **特征侧**：`data/processed/descriptors/*.parquet` 与 `structures/psmiles.parquet`
             里不许出现 `label` / `y` / `target` / `row_id` 这类列。A-6 的 `--verify` 也查这一条，
             这里独立再查一遍——两条独立的路同时出错才会漏。
          ② **代码侧**：`src/chainworth/**` 里任何 `fit` / `fit_transform` / `partial_fit`
             调用，作用对象必须能看出是训练集（名字里带 `train` / `_tr` / `tr_`），
             否则就算"在全量上 fit"——这是 Proposal §7.5 的头号失败模式，而且它不抛异常。
             目前 `src/` 里一个 `fit` 都没有（A 只做数据，不做模型），所以这条现在是**闸门**：
             它拦的是 B、C 以后写进来的代码，而不是追查已有的错。
          ③ **划分侧**：OpenPoly 的三份性质划分不按结构分组，同一根链可能同时落在训练与测试里。
             这是既有事实（D-24），我们照规则切、但必须**量出来记账**。这里按 RDKit 规范式
             重算一遍，与文件里记的数逐个比对；口径必须是规范式——同一根链可以有两种写法，
             只比原始串会把 Tg 的 28 个结构报成 15 个（D-28）。
    依赖：`data/processed/**`（A-6，不入库；缺了就跳过 ①）、`src/chainworth/**`（只读扫描）、
          `splits/openpoly/**`（A-5）、`data/raw/openpoly/**`（A-2，不入库；缺了就跳过 ③）。
    配置：`configs/tasks/*.yaml` 里的 `data_ref` / `target_column` 由划分文件自带，无需另读。
    对外：pytest 用例。被谁用：CI、D 的入口审计（他会往 `src/` 注入一次全量 fit，看这里红不红）。

English:
    Purpose: the division-A deliverable list, §8 self-check and A-5 step 4 all name
    `tests/test_no_leakage.py`. It guards three independent things; missing any one of them
    inflates R^2 with no symptom:
    (1) feature side - the descriptor/structures parquet must not carry `label` / `y` / `target` /
        `row_id` columns (A-6's `--verify` checks this too; two independent paths, because only
        both failing at once lets it through);
    (2) code side - every `fit` / `fit_transform` / `partial_fit` call must be visibly restricted
        to training data. There are none in `src/` today (A ships data, not models), so this is a
        gate for B and C rather than an audit of existing code;
    (3) split side - OpenPoly splits do not group by structure, so one chain can land in both
        train and test. That is a published-rule consequence (D-24): follow the rule, but measure
        and record it. Recomputed here on RDKit canonical SMILES and compared with the recorded
        numbers; the basis must be canonical - the same chain has two spellings, and comparing raw
        strings under-reports Tg's 28 shared structures as 15 (D-28).
    Depends on: data/processed (A-6, not committed - test 1 skips), src/chainworth (read-only
    scan), splits/openpoly (A-5), data/raw/openpoly (A-2, not committed - test 3 skips).
    Public API: pytest cases. Callers: CI, D's entry audit.
"""

from __future__ import annotations

import ast
import json
import shutil
from functools import lru_cache
from pathlib import Path
from typing import Iterable

import pandas as pd
import pytest

from chainworth.protocol import splits as S

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src/chainworth"
DESCRIPTOR_DIR = REPO_ROOT / "data/processed/descriptors"
STRUCTURE_TABLE = REPO_ROOT / "data/processed/structures/psmiles.parquet"
OPENPOLY_SPLITS = REPO_ROOT / "splits/openpoly"

# 特征文件里出现即判红的列名。`row_id` 也在内：它会把"第几行"带进特征，等于把任务行序泄漏给模型。
# Column names that fail the gate outright. `row_id` is included: it carries row order into the
# feature matrix, i.e. leaks the task's row order to the model.
FORBIDDEN_FEATURE_COLUMNS = {"label", "label_raw", "y", "target", "target_value", "row_id"}

# 被当成"拟合"的方法名；判定规则：调用表达式里必须出现训练集标记，否则视为全量 fit。
# Method names treated as fitting; the call expression must mention a training marker.
FIT_METHODS = {"fit", "fit_transform", "partial_fit"}
TRAIN_MARKERS = ("train", "_tr", "tr_")
# 允许显式豁免：同一行行尾写 `# leak-ok: <理由>`。豁免必须写理由，空豁免不算。
# Explicit opt-out: `# leak-ok: <reason>` on the same line. A reason is mandatory.
OPT_OUT_TOKEN = "leak-ok:"


# --------------------------------------------------------------------------- #
# ② 代码侧：src/ 里不许拿全量数据 fit / no full-data fit under src/
# --------------------------------------------------------------------------- #
def find_full_data_fits(path: Path) -> list[tuple[int, str]]:
    """扫一个 .py 文件，返回（行号, 调用源码）里那些"看不出用了训练集"的 fit 调用。
    Scan one .py file and return (line, call source) for fits not visibly restricted to train.

    中文：
        用途：把"有没有在全量上 fit"这件不抛异常的事变成可自动判定的规则。
        输入：path —— 一个 Python 源文件。
        输出：[(行号, 调用表达式)]；空列表表示这个文件干净。
        数据来源：文件本身。配置来源：模块级 FIT_METHODS / TRAIN_MARKERS / OPT_OUT_TOKEN。
        副作用：只读。异常：SyntaxError —— 文件本身语法错时照常抛出（那是真错，不该吞）。
    """
    source = path.read_text(encoding="utf-8")
    lines = source.splitlines()
    offenders: list[tuple[int, str]] = []
    for node in ast.walk(ast.parse(source)):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        name = func.attr if isinstance(func, ast.Attribute) else getattr(func, "id", None)
        if name not in FIT_METHODS:
            continue
        call = ast.unparse(node)
        if any(marker in call.lower() for marker in TRAIN_MARKERS):
            continue
        line = lines[node.lineno - 1] if node.lineno - 1 < len(lines) else ""
        if OPT_OUT_TOKEN in line:
            continue
        offenders.append((node.lineno, call))
    return offenders


def scan_tree(directory: Path) -> list[tuple[Path, int, str]]:
    """扫整棵目录树，返回（文件, 行号, 调用）。/ Scan a tree; return (file, line, call)."""
    found: list[tuple[Path, int, str]] = []
    for path in sorted(directory.rglob("*.py")):
        for line, call in find_full_data_fits(path):
            found.append((path.relative_to(REPO_ROOT), line, call))
    return found


def test_src_fits_nothing_on_full_data() -> None:
    """`src/chainworth/**` 里没有一处"看不出用了训练集"的 fit 调用。

    现在 source 里一个 fit 都没有（A 只交付数据、不训练模型），所以这条断言是 **0 == 0**。
    它的价值在于**以后**：B 的标准化、C 的调参一旦写进 `src/`，忘了限定 train 就会在这里红。
    No fit call in `src/chainworth/**` is left without a visible training marker. Today there are
    none at all; the assertion's value is as a gate for the code B and C will add.
    """
    offenders = scan_tree(SRC_DIR)
    assert offenders == [], "发现疑似全量 fit：\n" + "\n".join(
        f"  {path}:{line}  {call}" for path, line, call in offenders
    )


def test_scanner_catches_a_full_data_fit(tmp_path: Path) -> None:
    """往临时文件里写一句 `scaler.fit(X_all)`，扫描器必须报出来——否则上面那条 0 == 0 毫无意义。
    A hand-written `scaler.fit(X_all)` must be reported; otherwise the zero above proves nothing.
    """
    target = tmp_path / "bad_module.py"
    target.write_text(
        "def run(X_all):\n"
        "    from sklearn.preprocessing import StandardScaler\n"
        "    scaler = StandardScaler()\n"
        "    scaler.fit(X_all)\n"
        "    return scaler\n",
        encoding="utf-8",
    )
    offenders = find_full_data_fits(target)
    assert len(offenders) == 1
    assert offenders[0][0] == 4
    assert "fit" in offenders[0][1]


def test_scanner_accepts_train_only_fits_and_explicit_opt_out(tmp_path: Path) -> None:
    """限定在训练集上的 fit 不报；写了 `# leak-ok: 理由` 的也不报。
    Train-restricted fits pass, and so does an explicit `# leak-ok: reason` opt-out.
    """
    good = tmp_path / "good_module.py"
    good.write_text(
        "def run(train, X_train, X_test, scaler):\n"
        "    scaler.fit(X_train)\n"
        "    scaler.fit_transform(X_train)\n"
        "    train_scaler.fit(train)\n"
        "    scaler.fit(X_test)  # leak-ok: 推理前只为对齐列顺序，不参与训练\n"
        "    return scaler\n",
        encoding="utf-8",
    )
    assert find_full_data_fits(good) == []


def test_partial_fit_counts_as_fit(tmp_path: Path) -> None:
    """`partial_fit` 也是拟合：增量训练同样只能喂训练集。
    `partial_fit` is fitting too: incremental training may only see train data.
    """
    target = tmp_path / "partial.py"
    target.write_text(
        "def run(model, X_all):\n    model.partial_fit(X_all)\n",
        encoding="utf-8",
    )
    offenders = find_full_data_fits(target)
    assert len(offenders) == 1 and "partial_fit" in offenders[0][1]


# --------------------------------------------------------------------------- #
# ① 特征侧：特征文件里不许有标签列 / no label columns inside feature files
# --------------------------------------------------------------------------- #
def _feature_files() -> list[Path]:
    files = []
    if DESCRIPTOR_DIR.exists():
        files.extend(sorted(DESCRIPTOR_DIR.glob("*.parquet")))
    if STRUCTURE_TABLE.exists():
        files.append(STRUCTURE_TABLE)
    return files


def test_feature_files_carry_no_label_columns() -> None:
    """A-6 的 10 份特征表与结构表，列名里不许出现标签或行号（只读 parquet 的 schema，不读数据）。
    The ten feature tables and the structure table must carry no label or row-order column
    (reads the parquet schema only, not the data).
    """
    files = _feature_files()
    if not files:
        pytest.skip("data/processed 未生成（不入库）；先跑 python scripts/02_features.py")

    import pyarrow.parquet as pq

    problems = []
    for path in files:
        columns = set(pq.read_schema(path).names)
        bad = columns & FORBIDDEN_FEATURE_COLUMNS
        if bad:
            problems.append(f"{path.name}: {sorted(bad)}")
    assert problems == [], "特征文件里出现标签/行号列：\n" + "\n".join(problems)


def test_feature_tables_expose_at_least_the_contracted_columns() -> None:
    """列名黑名单靠不住时兜一条正向：特征表必须同时有 `structure_id` 与 `psmiles_canonical`，
    且至少有一个 `morgan_` 指纹列——这样"文件被换成一个空壳"也会红。
    A positive companion to the blacklist: every feature table must have `structure_id`,
    `psmiles_canonical` and at least one `morgan_` column, so an empty stub also fails.
    """
    files = sorted(DESCRIPTOR_DIR.glob("*.parquet")) if DESCRIPTOR_DIR.exists() else []
    if not files:
        pytest.skip("data/processed/descriptors 未生成（不入库）")

    import pyarrow.parquet as pq

    for path in files:
        columns = list(pq.read_schema(path).names)
        assert "structure_id" in columns, f"{path.name} 缺 structure_id"
        assert "psmiles_canonical" in columns, f"{path.name} 缺 psmiles_canonical"
        assert any(c.startswith("morgan_") for c in columns), f"{path.name} 缺指纹列"


# --------------------------------------------------------------------------- #
# ③ 划分侧：OpenPoly 的跨集结构必须量出来、且按规范式记账
# --------------------------------------------------------------------------- #
@lru_cache(maxsize=8)
def _structure_sequences(data_ref: str, target_column: str) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """读一次发布表，返回（原始串序列, 规范式序列），顺序与划分文件的索引一致。
    Read the published table once; return (raw sequence, canonical sequence) in split-index order.

    中文：
        用途：划分里的 `train_index` / `val_index` / `test_index` 索引的是"该性质非空行"的位置，
              所以序列也必须按同样的顺序取，否则数出来的不是同一批结构。
        输入：data_ref —— 相对仓库根的 CSV 路径；target_column —— 目标性质列名。
        输出：(raw, canonical) 两个等长元组；解析失败的串保留为 `<unparsed>...` 以免被合并成一个。
        副作用：只读（并用 lru_cache 缓存，避免 30 份划分重复解析）。
    """
    table = pd.read_csv(REPO_ROOT / data_ref, encoding="utf-8-sig")
    subset = table.loc[table[target_column].notna()]
    raw = tuple(subset["PSMILES"].astype(str).tolist())
    canonical = tuple((S.canonical_smiles(smiles) or f"<unparsed>{smiles}") for smiles in raw)
    return raw, canonical


def count_shared_across_splits(sequence: Iterable[str], parts: dict[str, list[int]]) -> int:
    """统计"出现在 ≥2 个划分里"的结构个数。/ Count structures present in two or more splits."""
    seen: dict[str, set[str]] = {}
    for name, indices in parts.items():
        for index in indices:
            seen.setdefault(str(sequence[index]), set()).add(name)
    return sum(1 for names in seen.values() if len(names) > 1)


def check_split_leakage(path: Path) -> list[str]:
    """核对一份 OpenPoly 划分的 leakage 字段与重算结果；返回问题列表（空=通过）。
    Check one split's leakage block against a recomputation; return problems (empty = pass).
    """
    split = json.loads(path.read_text(encoding="utf-8"))
    problems: list[str] = []
    leakage = split.get("leakage") or {}
    for key in ("n_psmiles_shared_across_splits", "n_raw_strings_shared_across_splits"):
        if key not in leakage:
            problems.append(f"{path.name}：leakage 缺 {key}")
    if problems:
        return problems
    if leakage.get("structure_basis") != "rdkit_canonical_smiles":
        problems.append(
            f"{path.name}：structure_basis={leakage.get('structure_basis')!r}，"
            "应为 'rdkit_canonical_smiles'（按原始串统计会把泄漏报低，见 D-28）"
        )

    raw, canonical = _structure_sequences(split["source"]["data_ref"], split["target_column"])
    parts = {
        "train": split["train_index"],
        "val": split["val_index"],
        "test": split["test_index"],
    }
    if len(set(parts["train"]) | set(parts["val"]) | set(parts["test"])) != split["n_total"]:
        problems.append(f"{path.name}：三个集合的并集不等于 n_total={split['n_total']}")

    expected_canonical = count_shared_across_splits(canonical, parts)
    expected_raw = count_shared_across_splits(raw, parts)
    if leakage["n_psmiles_shared_across_splits"] != expected_canonical:
        problems.append(
            f"{path.name}：规范式口径记 {leakage['n_psmiles_shared_across_splits']}，"
            f"重算 {expected_canonical}"
        )
    if leakage["n_raw_strings_shared_across_splits"] != expected_raw:
        problems.append(
            f"{path.name}：原始串口径记 {leakage['n_raw_strings_shared_across_splits']}，"
            f"重算 {expected_raw}"
        )
    return problems


def _openpoly_split_files() -> list[Path]:
    return sorted(OPENPOLY_SPLITS.glob("*/rule_replicated_seed*.json"))


def test_openpoly_leakage_matches_recomputation() -> None:
    """30 份 OpenPoly 划分的泄漏数与规范式重算逐个相等。
    All 30 OpenPoly splits report the counts a canonical recomputation produces.
    """
    files = _openpoly_split_files()
    if not files:
        pytest.skip("splits/openpoly 没有 rule_replicated 文件")
    if not (REPO_ROOT / "data/raw/openpoly").exists():
        pytest.skip("data/raw/openpoly 不在（不入库）")

    problems: list[str] = []
    for path in files:
        problems.extend(check_split_leakage(path))
    assert problems == [], "泄漏记账与重算不一致：\n" + "\n".join(problems)


def test_leakage_basis_is_canonical_and_strictly_larger_for_tg_seed0() -> None:
    """Tg 的 seed 0 上，规范式口径必须严格大于原始串口径（28 > 15）。

    这条是 D-28 的**回归锁**：谁把统计改回按原始串做，数字会掉回 15，这里立刻红。
    而且它证明"同一根链有两种写法"不是理论问题——就发生在这批数据里。
    On Tg seed 0 the canonical count must strictly exceed the raw-string count (28 > 15). This is
    the regression lock for D-28: reverting to raw strings drops the number back to 15 and turns
    this red. It also shows the two-spellings problem is not theoretical in this dataset.
    """
    path = OPENPOLY_SPLITS / "op_tg/rule_replicated_seed0.json"
    if not path.exists() or not (REPO_ROOT / "data/raw/openpoly").exists():
        pytest.skip("需要 splits/openpoly/op_tg 与 data/raw/openpoly")
    leakage = json.loads(path.read_text(encoding="utf-8"))["leakage"]
    assert leakage["n_psmiles_shared_across_splits"] > leakage["n_raw_strings_shared_across_splits"]
    assert leakage["n_raw_strings_shared_across_splits"] == 15
    assert leakage["n_psmiles_shared_across_splits"] == 28


def test_leakage_check_catches_a_tampered_count(tmp_path: Path) -> None:
    """把记的数改掉一个，`check_split_leakage()` 必须报出来——否则上面那条"全等"不能证明检查有效。
    Bump one recorded count and the checker must complain; otherwise the equality above proves
    nothing about the checker itself.
    """
    source = OPENPOLY_SPLITS / "op_tg/rule_replicated_seed0.json"
    if not source.exists() or not (REPO_ROOT / "data/raw/openpoly").exists():
        pytest.skip("需要 splits/openpoly/op_tg 与 data/raw/openpoly")

    tampered = tmp_path / "tampered.json"
    shutil.copy(source, tampered)
    payload = json.loads(tampered.read_text(encoding="utf-8"))
    payload["leakage"]["n_psmiles_shared_across_splits"] += 1
    tampered.write_text(json.dumps(payload, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")

    problems = check_split_leakage(tampered)
    assert any("规范式口径记" in problem for problem in problems)


def test_every_openpoly_split_declares_its_leakage() -> None:
    """每份划分都得把泄漏写出来，不许有文件"忘了记"——沉默就是遮掩。
    Every split must declare its leakage; a file that simply omits it is a silent omission.
    """
    files = _openpoly_split_files()
    if not files:
        pytest.skip("splits/openpoly 没有 rule_replicated 文件")
    missing = [p.name for p in files if "leakage" not in json.loads(p.read_text(encoding="utf-8"))]
    assert missing == [], f"这些划分没有 leakage 字段：{missing}"
