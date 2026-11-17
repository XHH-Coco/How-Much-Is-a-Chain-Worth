"""入口审计：数据怎么进来的——单位换算有没有证据、特征里有没有藏全量统计量。
Entry audit: how the data got in - are unit conversions evidenced, and does a feature column ever
carry a full-data statistic.

中文：
    用途：分工书 D-9 的机器版。入口侧要查两件事，两件都**不抛异常**，只会让整张表失去意义：
          ① **单位的证据链**：`data/manifests/unit_decisions.csv` 每一行的 `evidence` 必须非空
             （"官方没说，我们猜的"就是没有证据），且**限定的换算必须能逐行复算**——
             `label = label_raw + 273.15` 要在这 8,000 多行上字节级成立，不是抽查两行看着像；
          ② **特征里的全量统计量**：把"全量标签均值"当成一列加进特征，R² 会变好而**任何现有
             检查都不会红**——`test_no_leakage.py` 只查列名（`y` / `target` / `row_id`），
             一个叫 `y_mean` 的常数列不在名单里。所以这里换一个判据：**任何一列若对全体样本
             取值相同，它的值不许等于标签的均值/中位数/标准差/最小值/最大值**。这条不依赖列名，
             只看"这一列是不是标签的一个统计量"。

    为什么判据落在**特征 pack** 而不是 `descriptors/` 目录：
          模型真正吃的是 `data/processed/features/*.npz`，而 pack 自带 `y`——判据可以自洽，
          干净 clone 上也能跑；往 `descriptors/*.parquet` 里注入的列**必然流进 pack**（pack 由
          那批描述符拼成），所以查 pack 覆盖了两个注入点。反过来，只查 `descriptors/` 目录的
          测试在干净 clone 上是跳过的（那份 34 MB 不入库），等于闸门在别人机器上不跑。

    能抓住什么、抓不住什么（老实写）：
          抓得住——某个描述符列被整列换成常数且等于标签的某个统计量（D-9 ② 的注入形态）。
          抓不住——某一列**不是**常数但与标签高度相关（比如 MolWt 与 Tg 的相关系数 0.8）。
              那不是泄漏：合法描述符本来就可能强相关，任何阈值都会变成"凭手感退单"。
              这条缺口记在 `docs/Audit Log.md` 入口侧章节，不假装已经解决。

    数据来源：`data/manifests/unit_decisions.csv`、`data/interim/harmonised/*.parquet`（A-3）、
              `data/raw/polybench/*.csv`（A-2，缺了就跳过回查那条）、
              `data/processed/features/*.npz`（D-38 入库的 11 份 + C-b 的 2 份 scaling）。
    副作用：只读。异常：无（缺件走 skip，不假装通过）。
    被谁用：CI、`docs/Audit Log.md` 入口侧章节、D 的交接自检。

English:
    Purpose: the machine version of D-9. Two entry-side things that never raise but invalidate the
    whole table: (1) every row of `unit_decisions.csv` must carry evidence, and a declared
    conversion must be re-computable row by row (`label = label_raw + 273.15`, not just "looks
    right on two rows"); (2) a feature column holding the full-data label mean would improve R^2
    with no existing gate turning red - `test_no_leakage.py` only matches column *names*. So the
    criterion here is name-free: any column that is constant across all rows must not equal a
    statistic (mean / median / std / min / max) of the label.
    Scope: the criterion lives on the feature **packs**, not the descriptor directory - the packs
    are what the model consumes, they carry `y` inline, and anything injected into the descriptors
    flows into the packs anyway; the descriptor directory is not committed, so a test reading it
    would silently skip on a clean clone.
    Honest limits: it catches the constant-column injection; it does not catch a non-constant
    column that merely correlates strongly with the label, because a legitimate descriptor may do
    that and any threshold would be arbitrary. That gap is recorded in `docs/Audit Log.md`.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
UNIT_DECISIONS = REPO_ROOT / "data" / "manifests" / "unit_decisions.csv"
HARMONISED_DIR = REPO_ROOT / "data" / "interim" / "harmonised"
FEATURES_DIR = REPO_ROOT / "data" / "processed" / "features"

# 摄氏 → 开尔文的换算，与 `unit_decisions.csv` 的 `rule` 逐字对应。
# Celsius to kelvin, matching the `rule` column of `unit_decisions.csv` verbatim.
KELVIN_OFFSET = 273.15

# 判据用到的五个标签统计量。名字进失败信息，方便一眼看出撞的是哪一个。
# The five label statistics the criterion compares against; the names go into the failure message.
LABEL_STATISTICS = ("mean", "median", "std", "min", "max")


# --------------------------------------------------------------------------- #
# ① 单位的证据链 / The evidence chain behind each unit decision
# --------------------------------------------------------------------------- #
def _label_unit_rows() -> pd.DataFrame:
    """取 `decision_kind == label_unit` 的那些行。/ The `label_unit` rows of the manifest."""
    frame = pd.read_csv(UNIT_DECISIONS)
    return frame[frame["decision_kind"] == "label_unit"].copy()


def test_every_unit_decision_carries_evidence() -> None:
    """43 行里没有一行是"没有证据地猜"：`evidence` 非空、`rule` 非空、`decided_at` 可解析。
    No row was decided without evidence: `evidence` and `rule` are non-empty and `decided_at`
    parses.

    中文：判据是**长度下限 40 字符**而不是"非空"——一个"。"也算非空。这条不追求措辞质量，
          只要求"写得出为什么"，因为 A-4 的规矩是"猜就是错"（D-16）。
    English: the bar is 40 characters, not merely "non-empty", because a lone full stop is
    non-empty too.
    """
    frame = pd.read_csv(UNIT_DECISIONS)

    assert len(frame) == 43, f"单位决策表行数变了：{len(frame)}"
    blank = frame[frame["evidence"].fillna("").str.len() < 40]
    assert blank.empty, "evidence 太短或为空：\n" + blank[["task_id", "decision_kind"]].to_string()

    blank_rule = frame[frame["rule"].fillna("").str.strip() == ""]
    assert blank_rule.empty, "rule 为空：\n" + blank_rule[["task_id"]].to_string()

    parsed = pd.to_datetime(frame["decided_at"], errors="coerce")
    assert parsed.notna().all(), "decided_at 有解析不了的值"


def test_celsius_conversions_are_exact_on_every_row() -> None:
    """声明了 `+273.15` 的任务，`label − label_raw` 必须**逐行**等于 273.15；identity 的逐行相等。
    For the tasks declaring `+273.15`, `label - label_raw` is exactly 273.15 on every row; identity
    tasks are equal on every row.

    中文：
        用途：D-9 ③ 的"不许只看抽检的两行"。抽检能发现"某一行抄错了"，但发现不了"某个任务
              整体用了另一个偏移"——这条能。
        输入：`data/interim/harmonised/*.parquet`（A-3，不入库）。
        输出：无（断言）。数据来源：调和表自带的 `label_raw` 与 `label`。
        副作用：只读。异常：无。缺件时 skip 并打印重建命令（`scripts/01_harmonise.py`）。
    English:
        Row-by-row, not two rows. A spot check finds a miscopied row but not a task-wide wrong
        offset; this finds both. Skips with the rebuild command when the harmonised tables are
        absent (they are not committed).
    """
    if not HARMONISED_DIR.is_dir():
        pytest.skip("缺 data/interim/harmonised/，重建：python scripts/01_harmonise.py")

    rules = _label_unit_rows().set_index("task_id")["rule"].to_dict()
    converted = [task for task, rule in rules.items() if "+ 273.15" in str(rule)]
    identity = [task for task, rule in rules.items() if str(rule).startswith("identity")]
    assert converted, "没有任何声明 +273.15 的任务，这条测试会变成空转"

    checked = 0
    for task_id in converted:
        path = HARMONISED_DIR / f"{task_id}_harmonised.parquet"
        if not path.is_file():
            continue
        frame = pd.read_parquet(path, columns=["label", "label_raw"])
        delta = frame["label"] - frame["label_raw"]
        assert np.allclose(delta, KELVIN_OFFSET, rtol=0, atol=1e-9), (
            f"{task_id}：label − label_raw 不是处处等于 273.15，"
            f"最小 {delta.min():.6f}、最大 {delta.max():.6f}"
        )
        checked += len(frame)

    for task_id in identity:
        path = HARMONISED_DIR / f"{task_id}_harmonised.parquet"
        if not path.is_file():
            continue
        frame = pd.read_parquet(path, columns=["label", "label_raw"])
        assert np.allclose(frame["label"], frame["label_raw"], rtol=0, atol=1e-12), (
            f"{task_id} 声明 identity，但 label 与 label_raw 不等"
        )
        checked += len(frame)

    assert checked > 10_000, f"只核到 {checked} 行，不像在核全表"


def test_twenty_rows_match_the_raw_csv() -> None:
    """抽 20 行回查原始 CSV 的单元格：`label_raw` 必须等于 `source_file` 第 `source_row` 行的那一格。
    Twenty rows traced back to the raw CSV cell: `label_raw` equals that cell of `source_file`.

    中文：
        用途：上面那条只能证明"调和表内部自洽"，证明不了"`label_raw` 真是从原始表抄来的"。
              这一条才是回查。抽法是**按任务分层**再补满 20 行：先给 15 个任务各取中间一行，
              剩下的名额按全表等距补——按行数比例抽会让 `pb_t1_tg`（8,055 行）独占一半名额，
              而 `op_tm`（168 行）一行都抽不到，那就不叫"覆盖全部任务的回查"。
        输入：`data/manifests/unit_decisions.csv` 的 `property`（原始列名，去掉尾随空格）、
              调和表的 `source_file` / `source_row` / `label_raw`、`data/raw/polybench/*.csv`。
        输出：无（断言）。副作用：只读。缺件时 skip。
    English:
        The previous test proves internal consistency; this one traces the number back to the raw
        cell. The sample is stratified by task first (one middle row per task), then topped up
        evenly across the whole table - sampling proportionally would give the 8,055-row task half
        the slots and leave the 168-row task with none.
    """
    if not HARMONISED_DIR.is_dir():
        pytest.skip("缺 data/interim/harmonised/，重建：python scripts/01_harmonise.py")

    column_of = _label_unit_rows().set_index("task_id")["property"].to_dict()
    by_task: dict[str, list[tuple[str, str, int, float]]] = {}
    for task_id, column in column_of.items():
        path = HARMONISED_DIR / f"{task_id}_harmonised.parquet"
        if not path.is_file():
            continue
        frame = pd.read_parquet(path, columns=["source_file", "source_row", "label_raw"])
        by_task[task_id] = [
            (task_id, str(source_file), int(source_row), float(label_raw))
            for source_file, source_row, label_raw in frame.itertuples(index=False)
        ]

    if not by_task:
        pytest.skip("没有可回查的调和行")

    sample: list[tuple[str, str, int, float]] = [rows[len(rows) // 2] for _, rows in sorted(by_task.items())][:20]
    everything = sorted(row for rows in by_task.values() for row in rows)
    step = max(1, len(everything) // 20)
    for row in everything[::step]:
        if len(sample) >= 20:
            break
        if row not in sample:
            sample.append(row)
    assert len(sample) == 20, f"只取到 {len(sample)} 行样本"

    cache: dict[str, pd.DataFrame] = {}
    for task_id, source_file, source_row, label_raw in sample:
        path = REPO_ROOT / source_file
        if not path.is_file():
            pytest.skip(f"缺原始表 {source_file}，重建：python scripts/00_download.py")
        if source_file not in cache:
            cache[source_file] = pd.read_csv(path)
        raw = cache[source_file]
        # 原始表头有尾随空格（实测 `TSSAB(GPa) ` 带一个空格），而 `unit_decisions.property`
        # 记的是 trim 后的名字——按 strip 后匹配，否则 KeyError 会把"名字差一个空格"
        # 报成"这行数据不对"。
        # Raw headers carry trailing spaces (measured: `TSSAB(GPa) `), while the manifest records
        # the trimmed name; match on stripped names so a stray space is not reported as bad data.
        column = next(
            (name for name in raw.columns if str(name).strip() == column_of[task_id]),
            None,
        )
        assert column is not None, (
            f"{source_file} 里找不到列 {column_of[task_id]!r}，实际列：{[str(c) for c in raw.columns]}"
        )
        cell = float(raw.iloc[source_row][column])
        assert cell == pytest.approx(label_raw, rel=0, abs=1e-9), (
            f"{task_id} 第 {source_row} 行：原始表 {column_of[task_id]} = {cell}，"
            f"调和表 label_raw = {label_raw}"
        )


# --------------------------------------------------------------------------- #
# ② 特征里不许藏全量统计量 / No feature column may hold a full-data statistic
# --------------------------------------------------------------------------- #
def label_statistic_offenders(
    features: np.ndarray,
    labels: np.ndarray,
    columns: list[str] | None = None,
) -> list[tuple[str, str, float]]:
    """返回（列名, 撞上的统计量, 值）——空列表表示这批判据干净。
    Return (column, statistic, value) for offending columns; empty means clean.

    中文：
        用途：把"特征里藏了全量统计量"变成可判定的规则。判据只有两步：
              ① 这一列在全体样本上取值相同（`nanstd == 0`，全 NaN 列跳过）；
              ② 那个常数值等于标签的某个统计量（容差 1e-9 相对）。
              两步都成立才判红——"常数"本身是合法的（指纹位、单值描述符都会是常数）。
        输入：features（n × d，含 NaN）、labels（n）、columns（可选列名）。
        输出：[(列名, 统计量, 值)]。数据来源：调用方给的矩阵。
        副作用：无。异常：无。
    English:
        Two conditions, both required: the column is constant across all rows (all-NaN columns
        skipped), and that constant equals a label statistic within 1e-9 relative tolerance.
        Constancy alone is legitimate (fingerprint bits, single-valued descriptors are constant).
    """
    features = np.asarray(features, dtype=float)
    labels = np.asarray(labels, dtype=float)
    names = columns or [f"col{i}" for i in range(features.shape[1])]

    stats = {
        "mean": float(np.mean(labels)),
        "median": float(np.median(labels)),
        "std": float(np.std(labels)),
        "min": float(np.min(labels)),
        "max": float(np.max(labels)),
    }

    # 常数判定用**极差**而不是方差：整列同值时 `np.nanstd` 仍会给出 6e-13 量级的舍入残差
    # （实测：`np.full(64, 396.33)` 的 nanstd = 6.25e-13），用 `std == 0` 会漏掉真正的常数列。
    # Constancy is judged on the **range**, not the variance: with 64 identical values `np.nanstd`
    # still returns ~6e-13, so `std == 0` would miss a genuinely constant column.
    is_nan = np.isnan(features)
    with np.errstate(invalid="ignore"):
        low = np.min(np.where(is_nan, np.inf, features), axis=0)
        high = np.max(np.where(is_nan, -np.inf, features), axis=0)
    has_value = np.isfinite(low) & np.isfinite(high)
    scale = np.maximum(1.0, np.abs(np.where(has_value, high, 0.0)))
    constant = has_value & ((high - low) <= 1e-9 * scale)

    offenders: list[tuple[str, str, float]] = []
    for index in np.nonzero(constant)[0]:
        value = float(high[index])
        for name, target in stats.items():
            # 纯相对容差，且统计量为 0 时跳过："标签均值恰好是 0"的判据没有信息量——
            # 任何恒为 0 的指纹位都会撞上它（`pb_t2_permeability` 的均值就在 1e-9 量级）。
            # Relative tolerance only, and a zero statistic is skipped: "the label mean happens to
            # be zero" carries no information because every constant-zero column matches it.
            if target == 0.0:
                continue
            if abs(value - target) <= 1e-9 * abs(target):
                offenders.append((names[index], name, value))
                break
    return offenders


def test_no_feature_column_encodes_a_label_statistic() -> None:
    """全部入包任务的 2,253 列里，没有一列是"标签的某个统计量被当成常数"。
    Across every packed task's 2,253 columns, none is a label statistic frozen into a constant.

    中文：输入 `data/processed/features/*.npz`（pack 自带 `y`）。
          输出：无（断言）。副作用：只读。
          跳过条件：一份 pack 都没有（`scripts/02_features.py` / `03b_assemble.py` 未跑）。
    English: reads the packs (they carry `y` inline); skips only when no pack exists at all.
    """
    packs = sorted(FEATURES_DIR.glob("*.npz")) if FEATURES_DIR.is_dir() else []
    if not packs:
        pytest.skip("缺 data/processed/features/*.npz，重建：python scripts/03b_assemble.py")

    problems: list[str] = []
    for path in packs:
        data = np.load(path, allow_pickle=True)
        features = np.hstack([data["desc"], data["fp"].astype(float)])
        columns = [str(name) for name in data["desc_columns"]] + [
            f"fp:{name}" for name in data["fp_columns"]
        ]
        for column, statistic, value in label_statistic_offenders(features, data["y"], columns):
            problems.append(f"{path.name} 的 {column} = {value!r}，恰好是标签的 {statistic}")

    assert not problems, "特征里出现了标签统计量：\n" + "\n".join(problems)


def test_detector_catches_an_injected_full_data_mean() -> None:
    """往矩阵里加一列"全量标签均值"，检测器必须报出来——否则上面那条 0 != 0 证明不了什么。
    Add a constant full-data label mean column and the detector must report it; otherwise the empty
    result above proves nothing.

    中文：这就是 D-9 ② 的注入实验，写成**每次 CI 都跑**的形态，而不是一次性手工验证。
    English: this is D-9's injection ② turned into a test that runs on every CI pass.
    """
    rng = np.random.default_rng(20261117)
    labels = rng.normal(400.0, 50.0, size=64)
    features = rng.normal(size=(64, 5))
    features[:, 2] = 0.0  # 一个合法的常数列（指纹位那种），不许报

    assert label_statistic_offenders(features, labels, list("abcde")) == []

    injected = np.hstack([features, np.full((64, 1), labels.mean())])
    offenders = label_statistic_offenders(injected, labels, list("abcdef"))
    assert offenders == [("f", "mean", float(labels.mean()))]


def test_detector_ignores_constants_that_are_not_label_statistics() -> None:
    """常数但不是标签统计量 → 不报。判据不许把"所有指纹位"都退单。
    A constant that is not a label statistic passes; otherwise every fingerprint bit is rejected.
    """
    labels = np.linspace(100.0, 500.0, 32)
    features = np.column_stack([np.zeros(32), np.ones(32), np.full(32, 7.42665278)])
    assert label_statistic_offenders(features, labels, ["zero", "one", "desc"]) == []


def test_detector_skips_all_nan_columns() -> None:
    """全 NaN 列（8 个 `BCUT2D_*` 就是全 NaN）不许被当成常数 0 去撞统计量。
    All-NaN columns (the eight BCUT2D ones) must not be read as a constant zero.
    """
    labels = np.zeros(16)
    features = np.full((16, 1), np.nan)
    assert label_statistic_offenders(features, labels, ["bcut2d"]) == []
