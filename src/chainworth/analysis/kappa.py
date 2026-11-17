"""κ = dΛ/dlog₁₀n：数据涨 10 倍，Λ 会不会移动——H2 的判据。
kappa = dLambda/dlog10 n - how far Lambda moves per decade of training data (H2's criterion).

中文：
    用途：分工 D-3。把两条臂的 scaling 表接起来，逐 (数据集, 折, 档位) 求 Λ，再对 log₁₀(n_train)
          求斜率。四条口径写在这里，因为每一条都能悄悄改变结论：
      ① **两臂必须落在同一批训练行与同一批测试点上。** 训练行的凭据是记录里的
         `n_subset_sha256`（A2 与 A3 必须逐位相同），测试点的凭据是预测 parquet 的 `row_id`
         集合（同一折所有档位必须相同）。任一不等就**报错停下**，不"尽力对齐"——那会把
         协议差异算进 κ。
      ② **斜率先按折算，再对折取平均**，不是把所有点堆起来做一次 OLS。同一条曲线上的点彼此
         相关（同一测试集、嵌套的训练集），按折算再平均等于"每折一票"，不会被点数多的折压死。
         `pooled` 那个数只作对照，不进结论。
      ③ **区间来自按测试行配对重采样**（`analysis.ci` 的同一套）：同一个测试点在两条臂上的
         残差一起被抽中，抽中的下标**同时用于该折的所有档位**——因为同一折的测试集在各档位上
         逐行相同，这正是官方 scaling 设计的可读性前提。
      ④ **Λ 的分母（1 − R²_unit）不稳时要显形。** 分母 < `MIN_DENOMINATOR` 的单元格照样算，
         但标 `unstable=True`，且在汇总里报"有几个档位是这种"——不静默丢。
    数据来源：`results/aggregate/scaling_cpu.csv`（A2）、`results/aggregate/scaling_chain.csv`
          （A3）、两份预测 parquet、两份特征 pack 的 `y`。
    配置来源：无（档位来自官方 manifest，已经写在两张表里）。副作用：写三张表（见 `write_all`）。
    异常：`KappaError` —— 两臂对不上、预测缺件、点数太少。

English:
    Purpose: division D, task D-3. Joins the two arms' scaling tables, forms Lambda per
    (dataset, fold, tier) and takes the slope against log10(n_train). Four conventions, each of
    which can quietly change the answer: (1) both arms must sit on the **same training rows**
    (proved by `n_subset_sha256`) and the **same test rows** (proved by the prediction `row_id`
    sets) - any mismatch raises instead of "aligning as best we can"; (2) the slope is fitted
    **per fold and then averaged**, not pooled, so one fold with many points cannot dominate; a
    pooled number is reported alongside as a contrast only; (3) the interval is a **paired
    bootstrap over test rows**, with one draw reused across every tier of that fold - the official
    scaling design keeps the test set row-identical at every tier, which is what makes that legal;
    (4) an unstable denominator `1 - R^2_unit` is flagged (`unstable=True`) and counted, never
    silently dropped.
"""

from __future__ import annotations

import csv
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from chainworth.analysis import ci

REPO_ROOT = Path(__file__).resolve().parents[3]
UNIT_TABLE = REPO_ROOT / "results" / "aggregate" / "scaling_cpu.csv"
CHAIN_TABLE = REPO_ROOT / "results" / "aggregate" / "scaling_chain.csv"
PREDICTIONS_DIR = REPO_ROOT / "results" / "aggregate" / "predictions"
FEATURES_DIR = REPO_ROOT / "data" / "processed" / "features"
CELLS_PATH = REPO_ROOT / "results" / "aggregate" / "kappa_cells.csv"
SUMMARY_PATH = REPO_ROOT / "results" / "aggregate" / "kappa.csv"
TASK_ID = "pb26_scaling_cp"

# 冻结项：与 D-2 的 Λ 区间同一套 B / seed（换就是换版本）。
# Frozen: the same B and seed as D-2's Lambda intervals.
BOOTSTRAP_DRAWS = ci.BOOTSTRAP_DRAWS
BOOTSTRAP_SEED = ci.BOOTSTRAP_SEED
INTERVAL = ci.INTERVAL
# 分母小于它就算"不稳"：R²_unit > 0.95 时 Λ 的比值会被放大成噪声。
# A denominator below this is called unstable: above R^2_unit = 0.95 the ratio amplifies noise.
MIN_DENOMINATOR = 0.05

# 不报斜率的两个门槛（D-3 验收标准 ④）。**两个都必须满足**才拟合：
#   ① 档位跨度 ≥ 半个数量级 —— 否则"每十倍数据移多少"这句话本身没有可分辨的量程；
#   ② 相邻档位之间至少隔 0.25 个数量级 —— 否则"多一档"其实是同一个点在重复测，
#      既不给斜率添信息，又会让 OLS 的杠杆落在那个点上。
# 判据不是"档位数 ≥ 3"：均聚 MD_5000 有 3 档（300 / 1000 / 1440），但顶上两点只差 0.158 dex，
# 实质仍是两点连线。
# Two gates for reporting a slope at all (D-3 criterion 4); both must pass. The criterion is NOT
# "at least three tiers": homopolymer MD_5000 has three (300 / 1000 / 1440) yet the top two sit
# 0.158 dex apart, which is still a two-point line wearing a third point as decoration.
MIN_SPAN_DEX = 0.5
MIN_GAP_DEX = 0.25


class KappaError(ValueError):
    """κ 算不了（两臂对不上、缺件、点数太少）。/ kappa cannot be computed."""


@dataclass(frozen=True)
class Cell:
    """一个 (数据集, 折, 档位) 上的两臂结果。/ One (dataset, fold, tier) cell of both arms.

    中文：字段依次是数据集、折名、档位、实际训练行数、两臂的 R² 与 run_id、训练行指纹、
          `lambda`（= ΔR² /(1 − R²_unit)）、`delta_r2`、`unstable`（分母过小）。
    English: dataset, fold, tier, actual training rows, both arms' R² and run ids, the training-row
    digest, Lambda, dR2, and the unstable-denominator flag.
    """

    dataset: str
    fold: str
    tier: str
    n_train: int
    r2_unit: float
    r2_chain: float
    unit_run_id: str
    chain_run_id: str
    subset_sha256: str
    delta_r2: float
    lam: float
    unstable: bool


@dataclass
class FoldBundle:
    """一折的全部档位：真实值、两臂预测、档位大小。／ One fold: truth, both arms' predictions, tiers."""

    dataset: str
    fold: str
    truth: np.ndarray
    row_id: np.ndarray
    levels: list[tuple[int, np.ndarray, np.ndarray]] = field(default_factory=list)
    tiers: list[str] = field(default_factory=list)


# ---- 1. 读表并对齐 / Read the tables and align them ----


def _read_csv(path: Path) -> list[dict[str, str]]:
    """读一张 CSV（UTF-8，兼容 CRLF）。／ Read one CSV (UTF-8, CRLF tolerated)."""
    if not path.is_file():
        raise KappaError(f"缺表 {path}：先跑两条臂的 scaling 扫描")
    with path.open(encoding="utf-8", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def _float(value: Any, what: str) -> float:
    """把表里的字符串转成浮点；空值直接报错（不拿 0 顶）。／ Parse a float, never default to 0."""
    text = str(value).strip()
    if not text:
        raise KappaError(f"{what} 是空的：这条记录不完整，不能用 0 顶替")
    return float(text)


def load_cells(
    unit_table: Path | None = None, chain_table: Path | None = None
) -> list[Cell]:
    """把两张 scaling 表按 (数据集, 档位, 折, 种子) 接成单元格，并核三件事。
    Join the two scaling tables on (dataset, tier, fold, seed) and check three things.

    中文：
        用途：κ 的输入。三件事**任一不成立就报错**：
              ① 训练行数相同（`n_train` 逐格相等）；
              ② 训练行指纹相同（`n_subset_sha256` 逐格相等）——这是"两臂看的是同一批行"的唯一凭据；
              ③ 两臂不是同一条记录（`arm` 分别是 a2_conventional / a3_graph）。
        输入：两张表路径（默认仓库里的两份）。输出：`Cell` 列表（按数据集、折、档位排序）。
        异常：`KappaError` —— 缺格、行数或指纹不等。
    English:
        Three checks, any of which raises: the same training-row count, the same training-row
        digest (`n_subset_sha256` - the only proof both arms saw the same rows), and two distinct
        arms. Missing cells also raise instead of producing an inner join that silently shrinks.
    """
    unit_rows = _read_csv(unit_table or UNIT_TABLE)
    chain_rows = _read_csv(chain_table or CHAIN_TABLE)

    unit: dict[tuple[str, str, str, str], dict[str, str]] = {}
    for row in unit_rows:
        if str(row.get("arm", "")) != "a2_conventional":
            continue
        if str(row.get("learner", "")) != "xgboost":
            continue
        key = (str(row["dataset"]), str(row["tier"]), str(row["fold"]), str(row["seed"]))
        unit[key] = row

    chain: dict[tuple[str, str, str, str], dict[str, str]] = {}
    for row in chain_rows:
        if str(row.get("arm", "")) != "a3_graph":
            continue
        key = (str(row["dataset"]), str(row["tier"]), str(row["fold"]), str(row["seed"]))
        chain[key] = row

    missing_unit = sorted(set(chain) - set(unit))
    missing_chain = sorted(set(unit) - set(chain))
    if missing_chain:
        raise KappaError(
            f"链臂缺 {len(missing_chain)} 格（例：{missing_chain[:3]}）：κ 需要两支都齐"
        )
    if missing_unit:
        raise KappaError(
            f"单元臂缺 {len(missing_unit)} 格（例：{missing_unit[:3]}）：这些格子没法算 Λ"
        )

    cells: list[Cell] = []
    for key in sorted(unit):
        u = unit[key]
        c = chain[key]
        if int(u["n_train"]) != int(c["n_train"]):
            raise KappaError(
                f"{key}：两臂训练行数不同（{u['n_train']} vs {c['n_train']}）——不是同一个档位"
            )
        if str(u["subset_sha256"]).strip() != str(c["subset_sha256"]).strip():
            raise KappaError(
                f"{key}：两臂训练行指纹不同（{u['subset_sha256'][:20]} vs "
                f"{c['subset_sha256'][:20]}）——不是同一批行，Λ 会被协议差异污染"
            )
        r2_unit = _float(u["r2"], f"{key} 的单元臂 R²")
        r2_chain = _float(c["r2"], f"{key} 的链臂 R²")
        delta = r2_chain - r2_unit
        denominator = 1.0 - r2_unit
        if denominator <= 0.0:
            raise KappaError(f"{key}：1 − R²_unit = {denominator:.4f} ≤ 0，Λ 无定义")
        cells.append(
            Cell(
                dataset=str(u["dataset"]),
                fold=str(u["fold"]),
                tier=str(u["tier"]),
                n_train=int(u["n_train"]),
                r2_unit=r2_unit,
                r2_chain=r2_chain,
                unit_run_id=str(u["run_id"]),
                chain_run_id=str(c["run_id"]),
                subset_sha256=str(u["subset_sha256"]),
                delta_r2=delta,
                lam=delta / denominator,
                unstable=denominator < MIN_DENOMINATOR,
            )
        )
    return cells


def _pack_labels(dataset: str) -> np.ndarray:
    """从 pack 里读该数据集的标签（按行位置）。／ Labels from the pack, by row position."""
    path = FEATURES_DIR / f"{TASK_ID}_{dataset}.npz"
    if not path.is_file():
        raise KappaError(f"缺 pack {path.name}：先跑 `scaling --build`")
    return np.asarray(np.load(path, allow_pickle=True)["y"], dtype=np.float64)


def _prediction(run_id: str) -> tuple[np.ndarray, np.ndarray]:
    """读一份预测 parquet，返回 (row_id, y_pred)，按 row_id 升序。／ Read one prediction file."""
    import pyarrow.parquet as pq

    path = PREDICTIONS_DIR / f"{run_id}.parquet"
    if not path.is_file():
        raise KappaError(f"缺预测 {path.name}：这条 run 没有预测就算不了区间")
    table = pq.read_table(path, columns=["row_id", "y_pred"])
    row_id = np.asarray(table.column("row_id").to_pylist(), dtype=np.int64)
    y_pred = np.asarray(table.column("y_pred").to_pylist(), dtype=np.float64)
    order = np.argsort(row_id, kind="stable")
    return row_id[order], y_pred[order]


def load_bundles(cells: Sequence[Cell]) -> dict[tuple[str, str], FoldBundle]:
    """按 (数据集, 折) 把各档位的预测对齐成同一套测试行。
    Align every tier's predictions onto one test-row order, per (dataset, fold).

    中文：
        用途：bootstrap 要在"同一批测试点"上重算 R²，所以先把一折里各档位的预测按 `row_id`
              排成同一顺序，并核**各档位的 row_id 集合逐位相同**（官方 scaling 的前提）。
        输入：`cells`。输出：{(dataset, fold): FoldBundle}，每个 bundle 里 `levels` 按 n_train 升序。
        数据来源：两份预测 parquet + pack 的 `y`。副作用：只读。
        异常：`KappaError` —— 档位之间测试集不同（说明划分被换过）。
    """
    bundles: dict[tuple[str, str], FoldBundle] = {}
    labels_cache: dict[str, np.ndarray] = {}
    for cell in cells:
        key = (cell.dataset, cell.fold)
        if cell.dataset not in labels_cache:
            labels_cache[cell.dataset] = _pack_labels(cell.dataset)
        unit_row, unit_pred = _prediction(cell.unit_run_id)
        chain_row, chain_pred = _prediction(cell.chain_run_id)
        if not np.array_equal(unit_row, chain_row):
            raise KappaError(
                f"{cell.dataset}/{cell.fold}/{cell.tier}：两臂的测试行不同（{unit_row.size} vs "
                f"{chain_row.size}）——配对 bootstrap 的前提不成立"
            )
        bundle = bundles.get(key)
        if bundle is None:
            bundle = FoldBundle(
                dataset=cell.dataset,
                fold=cell.fold,
                truth=labels_cache[cell.dataset][unit_row],
                row_id=unit_row,
            )
            bundles[key] = bundle
        elif not np.array_equal(bundle.row_id, unit_row):
            raise KappaError(
                f"{cell.dataset}/{cell.fold}：档位 {cell.tier} 的测试集与其它档位不同："
                "官方 scaling 的测试集在所有档位上必须逐行相同"
            )
        bundle.levels.append((cell.n_train, unit_pred, chain_pred))
        bundle.tiers.append(cell.tier)
    for bundle in bundles.values():
        order = np.argsort([item[0] for item in bundle.levels])
        bundle.levels = [bundle.levels[index] for index in order]
        bundle.tiers = [bundle.tiers[index] for index in order]
    return bundles


# ---- 2. Λ 与 κ / Lambda and kappa ----


def _lambda_rows(truth: np.ndarray, unit: np.ndarray, chain: np.ndarray) -> np.ndarray:
    """按行算 Λ（`truth`/`unit`/`chain` 是 (B, n)）。／ Row-wise Lambda for (B, n) inputs."""
    r2_unit = ci._r2_rows(truth, unit)
    r2_chain = ci._r2_rows(truth, chain)
    with np.errstate(divide="ignore", invalid="ignore"):
        return (r2_chain - r2_unit) / (1.0 - r2_unit)


def _slope_rows(x: np.ndarray, values: np.ndarray) -> np.ndarray:
    """按行算 OLS 斜率：`x` 是 (p,)，`values` 是 (B, p)，返回 (B,)。
    Row-wise OLS slope of `values` on `x`; x is (p,) and values is (B, p).
    """
    centre = x - x.mean()
    denominator = float(np.einsum("i,i->", centre, centre))
    if denominator <= 0.0:                              # pragma: no cover - 档位只有一个
        raise KappaError("档位只有一个，斜率无定义")
    return (values - values.mean(axis=1, keepdims=True)) @ centre / denominator


def point_estimate(bundles: Mapping[tuple[str, str], FoldBundle]) -> dict[str, dict[str, Any]]:
    """每个数据集：逐折斜率、按折平均的 κ、池化对照、两端 Λ。／ Per-dataset kappa and diagnostics.

    中文：输入 bundles；输出 {dataset: {...}}。副作用：无。
          字段：`kappa`（按折平均）、`kappa_pooled`（所有点堆一次 OLS，只作对照）、
          `per_fold`（每折斜率）、`lambda_first` / `lambda_last`（最小/最大档位的 Λ 均值）、
          `n_levels` / `n_folds`，以及两个**决定"这条斜率该不该报"**的诊断量：
          `span_dex`（档位跨度，log₁₀ 最大/最小）与 `min_gap_dex`（相邻档位的最小间隔）。
    English: Returns per-dataset kappa (mean of per-fold slopes), the pooled contrast, the per-fold
    slopes, the two end-point Lambdas, and two diagnostics that decide whether a slope should be
    reported at all: `span_dex` (log10 max/min) and `min_gap_dex` (smallest gap between tiers).
    """
    out: dict[str, dict[str, Any]] = {}
    for dataset in sorted({key[0] for key in bundles}):
        folds = [key for key in sorted(bundles) if key[0] == dataset]
        slopes: list[float] = []
        all_x: list[float] = []
        all_lam: list[float] = []
        firsts: list[float] = []
        lasts: list[float] = []
        n_levels = 0
        span_dex = 0.0
        min_gap_dex = math.inf
        for key in folds:
            bundle = bundles[key]
            x = np.log10([float(item[0]) for item in bundle.levels])
            # 档位跨度与最小相邻间隔：**两个点永远能连一条直线**，但那条线的斜率没有不确定度
            # 可言。D-3 的验收标准 ④ 要求这种情形不许拟合成一条"看起来很确定"的线。
            # Span and smallest gap: two points always define a line, but that line has no
            # uncertainty. D-3's criterion 4 forbids reporting such a slope as if it were one.
            span_dex = max(span_dex, float(x.max() - x.min()))
            if x.size > 1:
                min_gap_dex = min(min_gap_dex, float(np.min(np.diff(np.sort(x)))))
            lam = _lambda_rows(
                bundle.truth[None, :],
                np.vstack([item[1] for item in bundle.levels]),
                np.vstack([item[2] for item in bundle.levels]),
            )
            if not np.all(np.isfinite(lam)):
                raise KappaError(f"{key}：Λ 里出现非有限值，这条曲线不能用")
            slopes.append(float(_slope_rows(x, lam[None, :])[0]))
            all_x.extend(x.tolist())
            all_lam.extend(lam.tolist())
            firsts.append(float(lam[0]))
            lasts.append(float(lam[-1]))
            n_levels = max(n_levels, len(bundle.levels))
        pooled = float(
            _slope_rows(np.asarray(all_x), np.asarray(all_lam)[None, :])[0]
        )
        out[dataset] = {
            "kappa": float(np.mean(slopes)),
            "kappa_pooled": pooled,
            "per_fold": slopes,
            "lambda_first": float(np.mean(firsts)),
            "lambda_last": float(np.mean(lasts)),
            "n_levels": n_levels,
            "n_folds": len(folds),
            "span_dex": span_dex,
            "min_gap_dex": 0.0 if not np.isfinite(min_gap_dex) else min_gap_dex,
        }
    return out


def bootstrap_kappa(
    bundles: Mapping[tuple[str, str], FoldBundle],
    *,
    draws: int = BOOTSTRAP_DRAWS,
    seed: int = BOOTSTRAP_SEED,
    level: float = INTERVAL,
) -> dict[str, ci.BootstrapResult]:
    """κ 的配对 bootstrap 区间（同一折的所有档位共用一组下标）。
    Paired bootstrap interval for kappa; one draw is reused across every tier of a fold.

    中文：
        输入：bundles；draws / seed / level（冻结项）。
        输出：{dataset: BootstrapResult}（`estimate` 是点估计，`lo`/`hi` 是 percentile 区间）。
        副作用：无。异常：`KappaError` —— 某折只有不到 2 个档位。
        实现：每块 `ci.CHUNK` 次重采样一起算。对每一折抽 (B, n) 的下标矩阵，**同一下标矩阵
              用于该折所有档位与两条臂**，于是每个重采样得到一条完整曲线，除以 log₁₀n 的斜率
              就是该折的 β*；κ* = 各折 β* 的均值。这样"曲线形状"的不确定度与"折间差异"同时进来。
    English:
        For each fold one (B, n) index matrix drives every tier and both arms, so each resample
        yields a whole curve; its slope is that fold's beta*, and kappa* is the mean over folds.
        Both the curve's shape uncertainty and the between-fold spread enter the interval.
    """
    results: dict[str, ci.BootstrapResult] = {}
    for dataset in sorted({key[0] for key in bundles}):
        folds = [bundles[key] for key in sorted(bundles) if key[0] == dataset]
        for bundle in folds:
            if len(bundle.levels) < 2:
                raise KappaError(f"{bundle.fold} 只有 {len(bundle.levels)} 个档位：斜率算不出来")
        sample: list[np.ndarray] = []
        done = 0
        rng = np.random.default_rng(seed)
        while done < draws:
            size = min(ci.CHUNK, draws - done)
            per_fold: list[np.ndarray] = []
            for bundle in folds:
                n = bundle.truth.size
                indices = rng.integers(0, n, size=(size, n))
                truth = bundle.truth[indices]
                x = np.log10([float(item[0]) for item in bundle.levels])
                # 逐档位算，而不是 `vstack(levels)[:, indices]`：后者在二维下标下会返回
                # (p, B, n) 的三维数组，`_lambda_rows` 的 `ij,ij->i` 当场报"维数过多"。
                # 逐档位的 (B, n) 才是配对重采样要的形状——**同一下标矩阵用于该折所有档位**。
                # One tier at a time: `vstack(levels)[:, indices]` is 3-D under a 2-D index, which
                # `_lambda_rows` cannot take. Per-tier (B, n) with the *same* index matrix is the
                # shape the paired bootstrap needs.
                lam = np.empty((size, len(bundle.levels)), dtype=np.float64)
                for column, (_, unit_pred, chain_pred) in enumerate(bundle.levels):
                    lam[:, column] = _lambda_rows(
                        truth, unit_pred[indices], chain_pred[indices]
                    )
                per_fold.append(_slope_rows(x, lam))
            sample.append(np.mean(np.vstack(per_fold), axis=0))
            done += size
        draws_all = np.concatenate(sample)
        finite = draws_all[np.isfinite(draws_all)]
        point = point_estimate({key: bundles[key] for key in bundles if key[0] == dataset})[dataset]
        lo, hi = ci.percentile_interval(finite, level)
        results[dataset] = ci.BootstrapResult(
            estimate=float(point["kappa"]), lo=lo, hi=hi, n_boot=int(finite.size), n_points=len(folds), seed=seed
        )
    return results


# ---- 3. 落表 / Writing ----


def cell_rows(cells: Sequence[Cell]) -> list[dict[str, Any]]:
    """`kappa_cells.csv` 的行（一 (数据集, 折, 档位) 一行）。／ Rows of kappa_cells.csv."""
    rows: list[dict[str, Any]] = []
    for cell in cells:
        rows.append(
            {
                "dataset": cell.dataset,
                "fold": cell.fold,
                "tier": cell.tier,
                "n_train": cell.n_train,
                "r2_unit": f"{cell.r2_unit:.10f}",
                "r2_chain": f"{cell.r2_chain:.10f}",
                "delta_r2": f"{cell.delta_r2:.10f}",
                "lambda": f"{cell.lam:.10f}",
                "denominator": f"{1.0 - cell.r2_unit:.10f}",
                "unstable": str(cell.unstable).lower(),
                "subset_sha256": cell.subset_sha256,
                "unit_run_id": cell.unit_run_id,
                "chain_run_id": cell.chain_run_id,
            }
        )
    return rows


def summary_rows(
    points: Mapping[str, Mapping[str, Any]],
    intervals: Mapping[str, ci.BootstrapResult],
) -> list[dict[str, Any]]:
    """`kappa.csv` 的行（一数据集一行，含 H2 判据的两个条件）。／ Rows of kappa.csv."""
    rows: list[dict[str, Any]] = []
    for dataset, point in points.items():
        interval = intervals.get(dataset)
        kappa = float(point["kappa"])
        lo = float(interval.lo) if interval is not None else math.nan
        hi = float(interval.hi) if interval is not None else math.nan
        first = float(point["lambda_first"])
        last = float(point["lambda_last"])
        span = float(point["span_dex"])
        gap = float(point["min_gap_dex"])
        # 先判"这条斜率该不该报"（D-3 验收标准 ④），再谈符号与减半。顺序不能反：
        # 一旦先把 κ 报出来，读者就已经当它是数了，后面那行"仅供参考"没人看。
        # Decide whether a slope is reportable at all before discussing its sign. Order matters:
        # once kappa is printed the reader has already treated it as a number.
        if span < MIN_SPAN_DEX:
            slope_status = "no_slope_span"
        elif gap < MIN_GAP_DEX:
            slope_status = "no_slope_overlapping_tiers"
        else:
            slope_status = "ok"
        reportable = slope_status == "ok"
        # 两点连线斜率：无论中间几个点，只用两端算。"不拟合"时它是唯一还能报的斜率，
        # 但**不是 κ**——它没有不确定度，所以进单独一列，不进 `kappa`。
        # The end-point chord slope. When no slope is reported this is the only slope left, but it
        # is not kappa: it carries no uncertainty, so it lives in its own column.
        endpoint_slope = (last - first) / span if span > 0.0 else math.nan
        # H2 要两个条件同时成立才算被推翻：κ 的 95% 区间整段 < 0，**且** 最大档位的 Λ
        # 掉到最小档位的一半以下。Λ 本身可能为负，"一半"对负数没有意义，所以那一半只在
        # Λ_first > 0 时可判；其余情况如实写 `undefined`，不硬套。
        # H2 is refuted only if both hold: the whole 95% interval below zero, and Lambda at the
        # largest tier below half of the smallest. "Half" is meaningless when Lambda is negative,
        # so that condition is only evaluated when Lambda_first > 0.
        sign_negative = bool(reportable and np.isfinite(hi) and hi < 0.0)
        if first > 0.0:
            ratio = last / first
            halved = bool(reportable and ratio < 0.5)
            ratio_text = f"{ratio:.4f}"
        else:
            halved = False
            ratio_text = "undefined"
        if not reportable:
            verdict = "no_slope"
        elif sign_negative and halved:
            verdict = "H2_refuted"
        elif sign_negative:
            verdict = "kappa_negative_but_lambda_not_halved"
        else:
            verdict = "H2_not_refuted"
        rows.append(
            {
                "dataset": dataset,
                "n_levels": int(point["n_levels"]),
                "n_folds": int(point["n_folds"]),
                "span_dex": f"{span:.4f}",
                "min_gap_dex": f"{gap:.4f}",
                "slope_status": slope_status,
                "kappa": f"{kappa:.6f}" if reportable else "",
                "kappa_lo": "" if not (reportable and np.isfinite(lo)) else f"{lo:.6f}",
                "kappa_hi": "" if not (reportable and np.isfinite(hi)) else f"{hi:.6f}",
                "kappa_pooled": f"{float(point['kappa_pooled']):.6f}" if reportable else "",
                "endpoint_slope": "" if not np.isfinite(endpoint_slope) else f"{endpoint_slope:.6f}",
                "per_fold_kappa": ";".join(f"{value:.6f}" for value in point["per_fold"]),
                "lambda_first": f"{first:.6f}",
                "lambda_last": f"{last:.6f}",
                "lambda_ratio": ratio_text,
                "n_boot": 0 if interval is None else interval.n_boot,
                "interval_level": INTERVAL,
                "seed": BOOTSTRAP_SEED,
                "h2_sign_negative": str(sign_negative).lower(),
                "h2_lambda_halved": str(halved).lower(),
                "verdict": verdict,
            }
        )
    return rows


SUMMARY_COLUMNS = (
    "dataset",
    "n_levels",
    "n_folds",
    "span_dex",
    "min_gap_dex",
    "slope_status",
    "kappa",
    "kappa_lo",
    "kappa_hi",
    "kappa_pooled",
    "endpoint_slope",
    "per_fold_kappa",
    "lambda_first",
    "lambda_last",
    "lambda_ratio",
    "n_boot",
    "interval_level",
    "seed",
    "h2_sign_negative",
    "h2_lambda_halved",
    "verdict",
)

CELL_COLUMNS = (
    "dataset",
    "fold",
    "tier",
    "n_train",
    "r2_unit",
    "r2_chain",
    "delta_r2",
    "lambda",
    "denominator",
    "unstable",
    "subset_sha256",
    "unit_run_id",
    "chain_run_id",
)

# 杠杆检查（D-3 验收标准 ②）：去掉 n 最小的一档重拟合，看 κ 的符号会不会翻。
# Leverage check (D-3 criterion 2): drop the smallest tier, refit, and see whether kappa flips.
LEVERAGE_PATH = REPO_ROOT / "results" / "aggregate" / "kappa_leverage.csv"
LEVERAGE_COLUMNS = (
    "dataset",
    "dropped_tier",
    "n_levels_before",
    "n_levels_after",
    "span_dex_after",
    "slope_status_after",
    "kappa_base",
    "kappa_dropped",
    "delta",
    "lo_dropped",
    "hi_dropped",
    "sign_flipped",
    "verdict_dropped",
    "note",
)


def _drop_smallest_tier(
    bundles: Mapping[tuple[str, str], FoldBundle],
) -> tuple[dict[tuple[str, str], FoldBundle], dict[str, int]]:
    """每折去掉 n 最小的一档；返回（新 bundles, 各数据集被去掉的档位 n）。／ Drop the smallest tier."""
    trimmed: dict[tuple[str, str], FoldBundle] = {}
    dropped_n: dict[str, int] = {}
    for key, bundle in bundles.items():
        levels = list(bundle.levels)
        tiers = list(bundle.tiers)
        if len(levels) < 2:                      # pragma: no cover - 只有一档的折
            trimmed[key] = bundle
            continue
        dropped_n[bundle.dataset] = levels[0][0]
        trimmed[key] = FoldBundle(
            dataset=bundle.dataset,
            fold=bundle.fold,
            truth=bundle.truth,
            row_id=bundle.row_id,
            levels=levels[1:],
            tiers=tiers[1:],
        )
    return trimmed, dropped_n


def leverage_rows(
    bundles: Mapping[tuple[str, str], FoldBundle],
    points: Mapping[str, Mapping[str, Any]] | None = None,
    *,
    draws: int = BOOTSTRAP_DRAWS,
    level: float = INTERVAL,
) -> list[dict[str, Any]]:
    """D-3 验收标准 ②：去掉最小档重拟合，κ 是否翻转。／ Leverage check rows.

    中文：输入 bundles 与（可选的）原点估计；输出 `kappa_leverage.csv` 的行。
          为什么必须做：κ 是"曲线的整体倾斜"，而最小档位的分母（1 − R²_unit）最大、Λ 最不稳，
          所以它常常单独决定符号。不报这一行，读者无法判断"κ < 0"是不是一个点的功劳。
          翻转与否按**符号**判，不按大小判——大小会变是必然的，那不是发现。
    English: Refit without the smallest tier and report whether the sign flips. Symbols, not
    magnitudes: a magnitude shift is expected, a sign flip is a finding.
    """
    base = dict(points) if points is not None else point_estimate(bundles)
    trimmed, dropped_n = _drop_smallest_tier(bundles)
    after = point_estimate(trimmed)
    intervals = bootstrap_kappa(trimmed, draws=draws, level=level)
    rows: list[dict[str, Any]] = []
    for dataset in sorted(base):
        before = base[dataset]
        if dataset not in after:                 # pragma: no cover - 每折都只剩一档
            continue
        now = after[dataset]
        span = float(now["span_dex"])
        gap = float(now["min_gap_dex"])
        if span < MIN_SPAN_DEX:
            status = "no_slope_span"
        elif gap < MIN_GAP_DEX:
            status = "no_slope_overlapping_tiers"
        else:
            status = "ok"
        reportable_before = float(before["span_dex"]) >= MIN_SPAN_DEX and float(before["min_gap_dex"]) >= MIN_GAP_DEX
        reportable_after = status == "ok"
        interval = intervals.get(dataset)
        lo = float(interval.lo) if interval is not None else math.nan
        hi = float(interval.hi) if interval is not None else math.nan
        if reportable_before and reportable_after:
            kappa_dropped: float | str = round(float(now["kappa"]), 6)
            delta: float | str = round(float(now["kappa"]) - float(before["kappa"]), 6)
            flipped = bool(np.sign(float(now["kappa"])) != np.sign(float(before["kappa"])))
            note = "符号翻转：结论依赖最小档" if flipped else "符号不翻转"
        else:
            kappa_dropped = ""
            delta = ""
            flipped = False
            note = "删掉最小档后不再满足拟合门槛（D-3 ④），无法比较"
        rows.append(
            {
                "dataset": dataset,
                "dropped_tier": str(dropped_n.get(dataset, "")),
                "n_levels_before": int(before["n_levels"]),
                "n_levels_after": int(now["n_levels"]),
                "span_dex_after": f"{span:.4f}",
                "slope_status_after": status,
                "kappa_base": f"{float(before['kappa']):.6f}" if reportable_before else "",
                "kappa_dropped": "" if kappa_dropped == "" else f"{float(kappa_dropped):.6f}",
                "delta": "" if delta == "" else f"{float(delta):+.6f}",
                "lo_dropped": "" if not (reportable_after and np.isfinite(lo)) else f"{lo:.6f}",
                "hi_dropped": "" if not (reportable_after and np.isfinite(hi)) else f"{hi:.6f}",
                "sign_flipped": str(flipped).lower(),
                "verdict_dropped": "no_slope" if not reportable_after else (
                    "kappa_negative" if np.isfinite(hi) and hi < 0.0 else "not_significant_negative"
                ),
                "note": note,
            }
        )
    return rows


def _write(rows: Sequence[Mapping[str, Any]], columns: Sequence[str], path: Path) -> Path:
    """写一张 CSV（UTF-8 / LF）。／ Write one CSV (UTF-8 / LF)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(columns), lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column, "") for column in columns})
    return path


def write_all(
    *,
    bootstrap: bool = True,
    draws: int = BOOTSTRAP_DRAWS,
    cells_path: Path | None = None,
    summary_path: Path | None = None,
    leverage_path: Path | None = None,
) -> dict[str, Any]:
    """一条龙：读表 → 对齐 → 点估计 → （可选）区间 → 杠杆检查 → 写三张表。
    End to end: read, align, point estimate, optional interval, leverage check, write three tables.

    中文：输入：bootstrap（默认 True）、draws（重采样次数）、三张表路径。输出：汇总 dict。
          副作用：写 `kappa_cells.csv`、`kappa.csv`、`kappa_leverage.csv`。异常：`KappaError`。
    English: Args: bootstrap flag, draw count and the three output paths. Returns a summary dict;
    writes the cells, summary and leverage tables.
    """
    cells = load_cells()
    bundles = load_bundles(cells)
    points = point_estimate(bundles)
    intervals = bootstrap_kappa(bundles, draws=draws) if bootstrap else {}
    leverage = (
        leverage_rows(bundles, points, draws=draws) if bootstrap else []
    )
    _write(cell_rows(cells), CELL_COLUMNS, cells_path or CELLS_PATH)
    _write(summary_rows(points, intervals), SUMMARY_COLUMNS, summary_path or SUMMARY_PATH)
    if leverage:
        _write(leverage, LEVERAGE_COLUMNS, leverage_path or LEVERAGE_PATH)
    return {
        "cells": len(cells),
        "datasets": sorted(points),
        "points": points,
        "intervals": intervals,
        "leverage": leverage,
        "cells_path": str((cells_path or CELLS_PATH).relative_to(REPO_ROOT)),
        "summary_path": str((summary_path or SUMMARY_PATH).relative_to(REPO_ROOT)),
        "leverage_path": str((leverage_path or LEVERAGE_PATH).relative_to(REPO_ROOT)) if leverage else "",
    }


def explain(summary: Mapping[str, Any]) -> str:
    """把汇总 dict 排成人读的几行。／ Render the summary dict as a few readable lines."""
    lines = [
        f"κ 表：{summary['cells']} 个单元格 → {summary['cells_path']}；汇总 {summary['summary_path']}"
    ]
    if summary.get("leverage_path"):
        lines.append(f"杠杆检查（去掉最小档重拟合）→ {summary['leverage_path']}")
    for dataset in summary["datasets"]:
        point = summary["points"][dataset]
        interval = summary["intervals"].get(dataset)
        span = float(point["span_dex"])
        gap = float(point["min_gap_dex"])
        if span < MIN_SPAN_DEX:
            lines.append(
                f"  {dataset:<26} 不拟合：跨度只有 {span:.3f} 个数量级（门槛 {MIN_SPAN_DEX}）"
            )
            continue
        if gap < MIN_GAP_DEX:
            lines.append(
                f"  {dataset:<26} 不拟合：相邻档位最小间隔 {gap:.3f}（门槛 {MIN_GAP_DEX}），"
                f"顶上两点近乎重合"
            )
            continue
        span_text = (
            f"[{interval.lo:+.4f}, {interval.hi:+.4f}]"
            if interval is not None and np.isfinite(interval.lo)
            else "（未算区间）"
        )
        lines.append(
            f"  {dataset:<26} κ={point['kappa']:+.4f} {span_text}  "
            f"Λ: {point['lambda_first']:+.4f} → {point['lambda_last']:+.4f}  "
            f"{point['n_levels']} 档 × {point['n_folds']} 折 跨度 {span:.2f} dex"
        )
    return "\n".join(lines)
