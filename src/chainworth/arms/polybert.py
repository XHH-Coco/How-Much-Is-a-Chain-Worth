"""A4 polyBERT 迁移臂 —— 冻结嵌入 + ridge 头（主表），以及微调（对照）。
A4 polyBERT transfer arm - frozen embedding + ridge head (main table), plus fine-tuning (contrast).

中文：
    这个模块回答的问题：**一个已经读过约一千万条聚合物串的编码器，能不能直接迁移到这几个
    下游任务上？** 它与 A3 看的是同一份结构信息（PSMILES），但架构与训练方式完全不同：
    A3 从零学图，A4 用一个预训练好的 Transformer 当"特征提取器"（(a)）或当"初始化"（(b)）。

    三种口径，改任何一个都等于换了一条臂：

    ① **对齐用规范式，不用行号。** B-1 的 `features/<task>.npz` 按 `row_id` 排，B-2 的
       `embeddings.npy` 按 `psmiles_canonical` 升序排。两份文件的行序来源不同，按行号拼会
       得到"张冠李戴"的矩阵，而且不会报错、只会让 R² 变成噪声。本模块一律：
       `psmiles` → `canonicalise_many` → 查 `index.parquet` 的 `psmiles_canonical` → 取嵌入行。

    ② **(a) 的选择只用训练内部那份留出。** ridge 的正则强度 alpha 从配置里写死的候选网格中，
       按**训练内部 10% 留出**的 RMSE 取最小——与 A3 的早停用同一份留出（同一个
       `val_split_seed`），所以两条臂在"哪份样本上做选择"这件事上完全对齐。测试集一次不碰。

    ③ **线性头没有随机性，三个种子必然给出同一个数。** 不装作有方差：记录里三个种子逐位
       相同，`r2_std` 在汇总时会是 0。这不是"结果特别稳"，这是"闭式解没有掷骰子"——写进文档，
       免得读者把 0 当成好消息。

English:
    The question: can an encoder that already read ~10^7 polymer strings transfer to these
    downstream tasks? Same input as A3 (PSMILES), different architecture and training: A3 learns a
    graph from scratch, A4 uses a pretrained Transformer as a feature extractor (a) or an
    initialisation (b).

    Three conventions, each of which changes the arm if broken:

    (1) **Join on the canonical string, never on row order.** B-1's pack is ordered by `row_id`,
        B-2's embeddings are ordered by `psmiles_canonical`; joining them by position silently
        pairs structure i with label j - no error, just noise. This module always goes
        `psmiles` -> `canonicalise_many` -> `index.parquet` -> embedding row.

    (2) **(a) selects alpha on the training-internal hold-out only.** Same held-out split as A3's
        early stopping (same `val_split_seed`), so both arms make their choices on the same rows.
        The test split is never touched.

    (3) **A linear head has no randomness, so the three seeds give the same number.** The records
        say so instead of manufacturing variance: a standard deviation of 0 here means "closed form,
        no dice", not "remarkably stable".

被谁用：`scripts/06_run_a4.py`（(a) 与 (b) 的入口）、`tests/test_arms_polybert.py`。
Callers: `scripts/06_run_a4.py` and the test module.

数据来源 / Data sources:
    `data/processed/features/<task>.npz`（标签与 PSMILES，B-1）
    `data/processed/polybert/{embeddings.npy,index.parquet}`（B-2，冻结）
    `splits/`（A 的划分，只读）
    `configs/arms/a4_polybert.yaml`（本臂的冻结配置）
"""

from __future__ import annotations

import json
import logging
import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from chainworth.arms import graph as A
from chainworth.chem import features as F
from chainworth.io import loaders as L

LOGGER = logging.getLogger("chainworth.arms.polybert")

REPO_ROOT = A.REPO_ROOT
ARM = "a4_polybert"
DEFAULT_CONFIG_PATH = REPO_ROOT / "configs" / "arms" / "a4_polybert.yaml"

# 嵌入缓存的键：同一进程里 PB26 的 15 条 run 要共用同一份 600 维矩阵。
# Key for the in-process cache: PB26's 15 runs share one 600-dim matrix per task.
_DATA_CACHE: dict[str, "EmbeddingData"] = {}


# ---- 1. 嵌入表：规范式 → 行号 / The embedding table: canonical string -> row ----

@dataclass(frozen=True)
class EmbeddingTable:
    """B-2 的嵌入，按规范式建索引；`row_of` 是唯一允许的对齐入口。
    B-2's embeddings indexed by canonical string; `row_of` is the only allowed join.

    中文：
        字段：`vectors`（n×600 float32）、`row_of`（规范式 → 行号）、`sha256`（**数组内容**的哈希，
              与 B-2 的 manifest 同口径，不是 .npy 的字节哈希）、`source`（路径，写进日志）。
        为什么不在模块加载时就建索引：A4 的 48 条 run 里有一半用不到 PB26 的嵌入，
              而建索引要读 43 MB 并做 18,152 次字典插入——首条 run 建一次就够。
        English: `vectors` (n x 600), `row_of` (canonical -> row), `sha256` (a *content* hash,
        same recipe as B-2's manifest - not the .npy file hash), `source` (path, for the log).
    """

    vectors: np.ndarray
    row_of: dict[str, int]
    sha256: str
    source: str

    def row_of_or_none(self, canonical: str | None) -> int | None:
        """取行号；`None`（结构解析失败）或不在表里都返回 None，由调用方计数。
        Look up a row; None (unparseable) and absent both return None, counted by the caller.
        """
        if not canonical:
            return None
        return self.row_of.get(str(canonical))


def load_embedding_table(
    embeddings: str | Path | None = None, index: str | Path | None = None
) -> EmbeddingTable:
    """读 B-2 的 `embeddings.npy` + `index.parquet`，按 `psmiles_canonical` 建索引。
    Load B-2's embeddings and index, keyed by `psmiles_canonical`.

    中文：
        用途：这是 A4 唯一的表示入口。两件事要说清：
              ① 用 `mmap_mode="r"` 读 `.npy`：43 MB 的矩阵不必整个进内存，切片取 5,000 行很快；
              ② 用 pyarrow 读 parquet 而**不用 pandas**——`DSC6020_Project_Polybert` 环境里没有
                 pandas（`lock/polybert.pip.txt`），而这条臂就跑在那个环境里。
        输入：embeddings / index —— 路径（默认取 `configs/arms/a4_polybert.yaml` 的
              `representation` 段，见 `table_from_config`）。
        输出：`EmbeddingTable`。
        数据来源：`data/processed/polybert/`。副作用：读盘（最多 43 MB 映射）。异常：ArmError。
    English:
        The arm's only representation entry point. Two notes: (1) `.npy` is opened with
        `mmap_mode="r"` so 43 MB need not be resident; (2) the parquet is read with pyarrow, not
        pandas, because the polyBERT environment has no pandas.
    """
    vectors_path = Path(embeddings) if embeddings is not None else None
    index_path = Path(index) if index is not None else None
    if vectors_path is None or index_path is None:
        # 没给路径就读配置里的（配置文件是这条臂的冻结契约，路径写在那儿）。
        # Without explicit paths, read them from the frozen config.
        config = A.load_arm_config(DEFAULT_CONFIG_PATH)
        section = config.section("representation")
        vectors_path = vectors_path or (REPO_ROOT / str(section["embeddings"]))
        index_path = index_path or (REPO_ROOT / str(section["index"]))

    if not vectors_path.is_file():
        raise A.ArmError(f"缺嵌入 {vectors_path}：B-2 的产物不在，A4 跑不了")
    if not index_path.is_file():
        raise A.ArmError(f"缺嵌入索引 {index_path}：B-2 的产物不在，A4 跑不了")

    matrix = np.load(vectors_path, mmap_mode="r")
    if matrix.ndim != 2:
        raise A.ArmError(f"{vectors_path.name} 的维度是 {matrix.shape}，不是二维矩阵")

    import pyarrow.parquet as pq

    table = pq.read_table(index_path, columns=["psmiles_canonical"])
    canonical = [str(item) for item in table.column("psmiles_canonical").to_pylist()]
    if len(canonical) != matrix.shape[0]:
        raise A.ArmError(
            f"嵌入 {matrix.shape[0]} 行 ≠ 索引 {len(canonical)} 行：B-2 的两份产物不同步"
        )
    duplicates = len(canonical) - len(set(canonical))
    if duplicates:
        raise A.ArmError(f"索引里有 {duplicates} 个重复的规范式：对齐会指向两行，不能继续")

    row_of = {text: position for position, text in enumerate(canonical)}
    # 哈希用 B-2 的 `sha256_of_array`，不是为了省事：只有用**同一个算法**，本臂记录里的
    # 指纹才能和 `data/processed/polybert/manifest.json` 里的 `embeddings_sha256` 直接对账。
    # 换成"文件字节哈希"会得到一个永远对不上的数（实测 .npy 的字节哈希是 203950a3…，
    # 而 manifest 记的是数组内容哈希 2b79ccae…——两者都对，但混用就变成假警报）。
    # Use B-2's `sha256_of_array`, not the file hash: only the same recipe lets this arm's fingerprint
    # be compared with the manifest's `embeddings_sha256`. Mixing the two produces a false alarm.
    from chainworth.chem import polybert as PB

    content_sha = PB.sha256_of_array(np.asarray(matrix))
    return EmbeddingTable(
        vectors=matrix,
        row_of=row_of,
        sha256=content_sha,
        source=A._repo_relative(vectors_path),
    )


# ---- 2. 任务数据：标签 + 嵌入，按行号对齐 / Task data: labels + embeddings, in row order ----

@dataclass
class EmbeddingData:
    """一个任务的标签、嵌入与划分，行号同一套坐标系（0..n-1）。
    One task's labels, embeddings and split, all in one row numbering (0..n-1).

    中文：
        字段与 `graph.TaskData` 同名（`task_id` / `benchmark` / `n_rows` / `y` / `row_id` / `fold()`），
        因为 `run_grid()` 的编排不分臂——它只调 `data.fold(name)` 与 `data.y[...]`。
        差别只在 `mgs`（图）换成了 `emb`（嵌入），失败计数从 `n_graph_failed` 换成
        `n_rows_without_embedding`。
        English: field names mirror `graph.TaskData` because `run_grid()` is arm-agnostic; the only
        difference is `emb` instead of `mgs`, and a missing-embedding count instead of a graph one.
    """

    task_id: str
    benchmark: str
    n_rows: int
    split: L.SplitSpec
    y: np.ndarray
    row_id: np.ndarray
    emb: np.ndarray
    psmiles: np.ndarray
    has_embedding: np.ndarray
    n_missing: int
    missing: list[str]
    built_in_s: float
    embed_sha256: str = ""

    def fold(self, name: str) -> Any:
        """按名字取一折；名字不在就报错并列出可用的折（与 `graph.TaskData.fold` 同语义）。
        Fetch a fold by name, raising with the available names (same semantics as TaskData.fold).
        """
        for fold in self.split.folds:
            if fold.name == name:
                return fold
        raise A.ArmError(
            f"{self.task_id} 没有折 {name!r}；可用的折：{list(self.split.fold_names)}"
        )


def build_task_data(
    task_id: str, *, limit: int | None = None, use_cache: bool = True, need_embedding: bool = True
) -> EmbeddingData:
    """读装配产物与划分，把每行的嵌入按其规范式对齐到行号上。
    Read the pack and the split, and line the embeddings up with the rows via canonical strings.

    中文：
        用途：A4 的 `build_task_data`，与 A3 的同名函数一一对应（`run_grid` 用注入的方式选它）。
        三条约束，与 A3 完全一致：
              ① **行序不重排**：`row_id` 就是 `pack["row_id"]`，与 `splits` 的 `test_index`
                 同一坐标系；这里只做切片（`--limit`），从不排序或过滤；
              ② **缺嵌入不删行**：`has_embedding=False` 的行留在表里，标签留着，训练与预测时
                 跳过并记进记录——删行会让后面全部前移，测试集悄悄换样品；
              ③ **对齐看规范式**：同一结构出现多次（OpenPoly）共享同一行嵌入，这是对的；
                 但**标签是逐行的**，所以嵌入按行取、标签按行取，两者都按 `row_id` 索引。
        为什么有 `need_embedding=False`：规范化要 RDKit，而 `DSC6020_Project_Polybert` 环境
              （微调要在那儿跑，因为它有 transformers）**没有 rdkit**。微调模式只拿 `psmiles`
              原文去 tokenize，用不到冻结向量，所以走 `False` 分支——不查嵌入、不算规范式，
              代价是"缺嵌入"这件事在那条路径上测不出来（实测冻结臂覆盖 100%，见 §5，所以
              两条路径的样本集在本次交付里完全相同；汇总脚本会核对这一点）。
        输入：task_id；limit；use_cache；need_embedding —— 是否查询冻结嵌入（(a) 要，(b) 不要）。
        输出：`EmbeddingData`。
        数据来源：`data/processed/features/`、`data/processed/polybert/`、`splits/`。
        副作用：读盘。异常：ArmError —— 缺产物或一行都对不上。
    English:
        A4's counterpart of A3's `build_task_data`, selected by injection in `run_grid`. The three
        constraints match A3: row order is never rebuilt; a row without an embedding stays in the
        table and is skipped; joins go through the canonical string, while labels stay per row.
        `need_embedding=False` exists because canonicalisation needs RDKit and the polyBERT
        environment (where fine-tuning must run, since it has transformers) has none. The fine-tune
        path only needs the raw PSMILES, so it skips the embedding lookup; the cost is that missing
        embeddings cannot be counted on that path (measured coverage is 100%, and the summary script
        checks that the two paths saw the same rows).
    """
    key = f"{task_id}@{limit}@{int(need_embedding)}"
    if use_cache and key in _DATA_CACHE:
        return _DATA_CACHE[key]

    started = time.perf_counter()
    pack = F.load_pack(task_id)
    split = L.load_split(task_id)
    n_rows = int(len(pack["row_id"]))
    if limit is not None:
        n_rows = min(int(limit), n_rows)
        split = L.limit_split(split, n_rows)
        pack = {
            name: (value[:n_rows] if isinstance(value, np.ndarray) and value.ndim and value.shape[0] == len(pack["row_id"]) else value)
            for name, value in pack.items()
        }
    problems = L.check_split_matches_rows(split, n_rows)
    if problems:
        raise A.ArmError("；".join(problems))

    raw = np.asarray(pack["psmiles"]).astype(str)
    if need_embedding:
        table = load_embedding_table()
        from chainworth.chem.notation import canonicalise_many

        canonical, _n_bad = canonicalise_many(raw)
        dim = int(table.vectors.shape[1])
        emb = np.zeros((n_rows, dim), dtype=np.float32)
        has = np.zeros(n_rows, dtype=bool)
        missing: list[str] = []
        for position in range(n_rows):
            row = table.row_of_or_none(canonical[position])
            if row is None:
                if len(missing) < 50:
                    missing.append(f"第 {position} 行：{raw[position][:32]}")
                continue
            emb[position] = np.asarray(table.vectors[row], dtype=np.float32)
            has[position] = True
        n_missing = int((~has).sum())
        if n_missing == n_rows:
            raise A.ArmError(f"{task_id} 的 {n_rows} 行一行都对不上嵌入：B-2 与 B-1 的结构集不同")
        embed_sha = table.sha256
    else:
        # 微调路径：不碰嵌入，也不碰 RDKit。`has_embedding` 全 True 是"这条路径不筛行"的意思，
        # 不是"确认每行都有嵌入"——记录里 `n_rows_without_embedding` 因此写 None 而不是 0。
        # Fine-tune path: no embeddings, no RDKit. All-True here means "this path filters nothing",
        # not "every row has an embedding", so the record writes None rather than 0.
        emb = np.zeros((n_rows, 0), dtype=np.float32)
        has = np.ones(n_rows, dtype=bool)
        missing = []
        n_missing = 0
        embed_sha = ""

    built_in_s = time.perf_counter() - started
    LOGGER.info(
        "%s：%d 行 / 缺嵌入 %s；对齐 %.1f 秒",
        task_id,
        n_rows,
        f"{n_missing} 行（{100.0 * n_missing / max(n_rows, 1):.2f}%）" if need_embedding else "不检查（微调路径）",
        built_in_s,
    )
    data = EmbeddingData(
        task_id=task_id,
        benchmark=split.benchmark,
        n_rows=n_rows,
        split=split,
        y=np.asarray(pack["y"], dtype=np.float64),
        row_id=np.asarray(pack["row_id"], dtype=np.int64),
        emb=emb,
        psmiles=raw,
        has_embedding=has,
        n_missing=n_missing,
        missing=missing,
        built_in_s=built_in_s,
        embed_sha256=embed_sha,
    )
    if use_cache:
        _DATA_CACHE[key] = data
    return data


# ---- 3. 线性头：标准化 + ridge，闭式解 / The linear head: standardise, then solve ridge ----

def _standardise(train: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """算出**训练行**的均值与标准差（标准差为 0 的列置 1，避免除零）。
    Mean and standard deviation of the **training rows** (constant columns get scale 1).

    中文：输入：train（训练行）。输出：`(mean, scale)`，两者都是 float64。
          数据来源：调用方。副作用：无。异常：无。
          为什么要单独一个函数：验收④要求"不许用全量统计量"，把它写成一个只吃 `X_train` 的
          函数，`tests/test_no_leakage.py` 的扫描器就能按调用表达式判红黑。
    English: Args: training rows only. Returns: (mean, scale) in float64. Split out so the
    no-leakage scanner can judge the call expression instead of trusting a comment.
    """
    matrix = np.asarray(train, dtype=np.float64)
    mean = matrix.mean(axis=0)
    scale = matrix.std(axis=0, ddof=1) if matrix.shape[0] > 1 else np.ones(matrix.shape[1])
    scale = np.where(scale > 0.0, scale, 1.0)
    return mean, scale


def _ridge_solve(
    X: np.ndarray, y: np.ndarray, alpha: float, *, device: str = "cpu"
) -> np.ndarray:
    """解 `(X'X + aI) w = X'y`，返回 600 维权重（X、y 均已标准化、已中心化）。
    Solve `(X'X + aI) w = X'y` and return the 600-dim weights (X and y already standardised).

    中文：
        用途：闭式解，不引 sklearn（`DSC6020_Project_Polybert` 里没有它，也不该为一行公式加）。
              用 `torch.linalg.solve` 而不是 `np.linalg.solve`：`device="cuda"` 时求解真的在卡上
              完成，于是记录里的 `device` 与 A3 同一口径（见文档"为什么线性头也上 GPU"）。
              `alpha > 0` 保证正定，所以不会走最小二乘的伪逆分支。
        输入：X（n×d，已标准化）、y（n，已标准化）、alpha > 0、device。
        输出：`w`，shape (d,)，float64。
        数据来源：调用方。配置来源：`head.alpha_grid`。副作用：占用一点显存。
        异常：ArmError —— 求解失败（alpha ≤ 0 或矩阵病态）。
    English:
        A closed form, no sklearn (the polyBERT environment has none and one formula does not
        justify adding it). `torch.linalg.solve` instead of numpy so that `device="cuda"` really
        runs on the card, keeping the record's `device` on the same footing as A3. `alpha > 0`
        keeps the system positive definite, so the pseudo-inverse branch is never taken.
    """
    matrix = np.asarray(X, dtype=np.float64)
    target = np.asarray(y, dtype=np.float64)
    if alpha <= 0.0:
        raise A.ArmError(f"alpha={alpha}：ridge 的正则强度必须是正数")
    try:
        import torch
    except ImportError:
        # 没有 torch 就退回 numpy：公式完全相同，只是求解发生在 CPU 上。
        # 这条退路是给 `DSC6020_Project_Core`（无 torch，但有 rdkit 与 pandas）留的，
        # 万一 polyBERT 环境不可用，冻结臂仍能跑完——记录里的 `device` 会是 cpu，如实写。
        # Without torch, fall back to numpy: identical formula, solved on the CPU. This keeps the
        # frozen arm runnable in the core environment, and the record then says device=cpu.
        gram = matrix.T @ matrix + float(alpha) * np.eye(matrix.shape[1])
        return np.linalg.solve(gram, matrix.T @ target).astype(np.float64)
    xt = torch.as_tensor(matrix, dtype=torch.float64, device=device)
    yt = torch.as_tensor(target, dtype=torch.float64, device=device)
    gram = xt.transpose(0, 1) @ xt
    eye = torch.eye(gram.shape[0], dtype=torch.float64, device=device)
    try:
        weights = torch.linalg.solve(gram + float(alpha) * eye, xt.transpose(0, 1) @ yt)
    except RuntimeError as error:                       # pragma: no cover - 只在病态时触发
        raise A.ArmError(f"ridge 求解失败（alpha={alpha}）：{error}") from error
    return weights.detach().cpu().numpy().astype(np.float64)


def fit_head(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    alphas: Sequence[float],
    *,
    device: str = "cpu",
) -> dict[str, Any]:
    """在候选 alpha 上训练并在验证集上挑一个；返回权重与"选了哪个、为什么"。
    Fit on each candidate alpha, pick one on the validation set, and return the choice.

    中文：
        用途：(a) 的全部"学习"就在这一个函数里。三条约束：
              ① 特征与标签都用**训练行**的均值/标准差标准化（验证行只被变换，不参与统计）；
              ② 选 alpha 的判据是**验证集原单位 RMSE**——与 A3 的 `val_loss` 同一口径，
                 否则两条臂选的东西不可比；
              ③ 并列时取**更小的 alpha**（更少正则）：规则写死，不靠 `min()` 的遍历顺序。
              最终权重只在 `train` 上解（验证集只用于选 alpha，不参与拟合），与 A3 的
              "val 选 epoch、梯度只来自 train"对称。
        输入：X_train / y_train / X_val / y_val（原单位）、alphas（候选网格）、device。
        输出：dict，键 `weights`、`x_mean`、`x_scale`、`y_mean`、`y_scale`、`alpha`、
              `alpha_scores`（list of (alpha, val_rmse)）、`val_rmse`。
        数据来源：调用方。配置来源：`head.alpha_grid`。
        副作用：占显存。异常：ArmError —— 空数据或 alpha 网格为空。
    English:
        All of (a)'s learning is here. Three constraints: (1) feature and target statistics come
        from training rows only; (2) alpha is chosen by **validation RMSE in original units**, the
        same quantity A3's early stopping monitors; (3) ties go to the smaller alpha, a written rule
        rather than `min()`'s traversal order. The final weights are solved on `train` alone -
        validation picks alpha but never contributes gradients, mirroring A3's early stopping.
    """
    if len(alphas) == 0:
        raise A.ArmError("alpha_grid 是空的：配置里必须写死候选正则强度")
    x_train = np.asarray(X_train, dtype=np.float64)
    x_val = np.asarray(X_val, dtype=np.float64)
    if x_train.size == 0 or x_val.size == 0:
        raise A.ArmError("训练集或验证集为空：这条 run 不能跑")

    x_mean, x_scale = _standardise(x_train)
    z_train = (x_train - x_mean) / x_scale
    z_val = (x_val - x_mean) / x_scale

    y_train = np.asarray(y_train, dtype=np.float64)
    y_mean, y_scale = A.target_scaler(y_train)
    t_train = (y_train - float(y_mean)) / float(y_scale)

    scores: list[tuple[float, float]] = []
    best: tuple[float, float, np.ndarray] | None = None
    for alpha in alphas:
        weights = _ridge_solve(z_train, t_train, float(alpha), device=device)
        predicted = (z_val @ weights) * float(y_scale) + float(y_mean)
        rmse = float(math.sqrt(float(np.mean((np.asarray(y_val, dtype=np.float64) - predicted) ** 2))))
        scores.append((float(alpha), rmse))
        if best is None or rmse < best[1] - 1e-12:
            best = (float(alpha), rmse, weights)
    assert best is not None
    chosen_alpha, val_rmse, chosen_weights = best
    return {
        "weights": chosen_weights,
        "x_mean": x_mean,
        "x_scale": x_scale,
        "y_mean": float(y_mean),
        "y_scale": float(y_scale),
        "alpha": chosen_alpha,
        "alpha_scores": scores,
        "val_rmse": val_rmse,
    }


def apply_head(head: dict[str, Any], X: np.ndarray) -> np.ndarray:
    """把训练好的线性头应用到特征上，返回原单位的预测。
    Apply a fitted head to features and return predictions in original units.
    """
    matrix = np.asarray(X, dtype=np.float64)
    z = (matrix - head["x_mean"]) / head["x_scale"]
    return (z @ head["weights"]) * float(head["y_scale"]) + float(head["y_mean"])


# ---- 4. 一条 run：(a) 冻结 + ridge / One run: (a) frozen + ridge ----

def train_one(
    data: EmbeddingData,
    config: A.ArmConfig,
    *,
    fold: str,
    seed: int,
    device: str | None = None,
    max_epochs: int | None = None,
    patience: int | None = None,
    n_level: str | None = None,
) -> A.TrainOutcome:
    """跑一条 (a) 的 run：冻结嵌入 → ridge 头 → 在官方测试集上打分。
    Run one (a) run: frozen embeddings, a ridge head, scored once on the official test split.

    中文：
        用途：A4 主表的一条。步骤与 A3 的 `train_one` 同构，只是把"训练一个 MPNN"换成
              "解一个线性方程组"：
              ① 取折（官方索引）→ 需要时切出训练内部 10% 当验证（**与 A3 同一份留出**）；
              ② 标准化（只用训练行）→ 在候选 alpha 上选一个；
              ③ 用选中的 alpha 在训练行上解出权重；
              ④ 在**官方测试集**上预测一次，算 r2/mae/rmse；
              ⑤ 记录草稿：`run_id`、`split_version`、`config_hash`、`device`、`gpu` 块、
                 实测墙钟、选中的 `alpha`、缺嵌入行数、验证集来源。
              `seed` 不参与任何计算（闭式解无随机性），但**照实记进 run_id 与记录**：
              A4 的 48 条里有 16 条是同一个数的三个副本，这是设计而不是意外。
        输入：data；config；fold；seed；device（覆盖）；max_epochs/patience（兼容签名，未用）。
        输出：`TrainOutcome`。
        数据来源：`EmbeddingData`。配置来源：`config`。
        副作用：占显存、写日志。异常：ArmError —— 折不配套或没有可用行。
    English:
        One row of A4's main table. Structurally the same as A3's `train_one`, with "train an MPNN"
        replaced by "solve a linear system": take the fold (carving the same inner hold-out when the
        benchmark ships none), standardise on training rows, pick alpha on validation, solve on
        train, predict once on the official test split, then draft the record. `seed` enters no
        computation - a closed form has no randomness - and is still recorded verbatim, because 16
        of A4's 48 runs are three copies of one number by design.
    """
    try:
        import torch
    except ImportError:                                 # Core 环境没有 torch：走 numpy 那条
        torch = None

    head_cfg = config.section("head")
    device_cfg = config.section("device")
    if torch is None:
        chosen = "cpu"
    else:
        chosen = device or A.resolve_device(str(device_cfg.get("prefer", "cuda")))
        if chosen == "cuda" and not torch.cuda.is_available():
            chosen = "cpu"
        # 与 A3 同一套确定性开关（D-33 ⑦）。线性头不用 scatter_reduce，所以第二个开关在这里
        # 不是"必需"，但**记录里两个开关必须都为 true**——那是 `record_problems()` 对所有
        # `device=cuda` 记录的要求，也是"同 seed 两次逐位相同"能被机械核对的前提。多设一次无害。
        # Same determinism switches as A3 (D-33 item 7). The linear head never uses scatter_reduce, so
        # the second switch is not strictly needed here, but `record_problems()` requires both on every
        # cuda record - that is what makes "bit-identical on re-run" machine-checkable.
        deterministic = bool(device_cfg.get("cudnn_deterministic", True))
        torch.backends.cudnn.deterministic = deterministic
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(deterministic)

    val_cfg = config.section("validation")
    # A4 的网格没有 n 档：B-5 的规模扫描只做 GNN 那 225 条（分工_B §2 B-5），polyBERT 的
    # 嵌入是**按结构预算好的**，截训练行并不省任何编码时间，做出来也回答不了 H4 的问题。
    # 这条守卫存在的意义是：万一哪天有人把 `n_levels` 抄进 A4 的配置，会当场报错而不是
    # 悄悄跑出 225 条"名字带 n 档、实际全量"的记录。
    # A4 has no n dimension: B-5 scans only the GNN arm, and polyBERT's embeddings are precomputed
    # per structure, so trimming training rows saves no encoding time. The guard is here so that a
    # copy-pasted `n_levels` fails loudly instead of producing 225 records that claim a level.
    if n_level is not None:
        raise A.ArmError(
            "A4 的网格没有 n 档：n 档只属于 B-5 的 OpenPoly 扫描（分工_B §2 B-5），"
            "要跑规模扫描请用 --arm a3_graph --config configs/experiments/openpoly_scan.yaml"
        )
    spec_fold = data.fold(fold)
    train_index = np.asarray(spec_fold.train_index, dtype=np.int64)
    val_index = np.asarray(spec_fold.val_index, dtype=np.int64)
    test_index = np.asarray(spec_fold.test_index, dtype=np.int64)
    val_source = "official"
    if val_index.size == 0:
        train_index, val_index = A.carve_validation(
            train_index,
            fraction=float(val_cfg["val_fraction_if_missing"]),
            seed=int(val_cfg["val_split_seed"]),
        )
        val_source = f"inner_{float(val_cfg['val_fraction_if_missing']):.0%}_of_train"
    if test_index.size == 0:
        raise A.ArmError(f"{data.task_id}/{fold} 的测试集为空：这条 run 不该存在")

    def usable(index: np.ndarray) -> np.ndarray:
        """只留有嵌入的行；缺嵌入的行在记录里计数，不静默丢掉（同 A3 的建图失败口径）。
        Keep rows that have an embedding; the rest are counted in the record, never dropped
        silently (same convention as A3's graph failures).
        """
        return np.asarray([i for i in index if data.has_embedding[int(i)]], dtype=np.int64)

    train_rows = usable(train_index)
    val_rows = usable(val_index)
    test_rows = usable(test_index)
    if train_rows.size == 0 or val_rows.size == 0 or test_rows.size == 0:
        raise A.ArmError(
            f"{data.task_id}/{fold}：去掉缺嵌入的行后 train/val/test = "
            f"{train_rows.size}/{val_rows.size}/{test_rows.size}，有空的，这条 run 不能跑"
        )

    # 种子：线性头本身没有随机性，但求解器与 GPU 上的算子在某些情况下会读 RNG 状态；
    # 显式设一次，保证"同 seed 两次逐位相同"这条验收在任何路径上都成立。
    # The linear head has no randomness, but solvers and some GPU kernels can read RNG state; setting
    # the seed keeps "same seed, bit-identical" true on every path.
    if torch is not None:
        torch.manual_seed(int(seed))

    alphas = [float(item) for item in (head_cfg.get("alpha_grid") or [])]
    started = time.perf_counter()
    head = fit_head(
        data.emb[train_rows],
        data.y[train_rows],
        data.emb[val_rows],
        data.y[val_rows],
        alphas,
        device=chosen,
    )
    y_pred = apply_head(head, data.emb[test_rows])
    wall = time.perf_counter() - started

    y_true = np.asarray(data.y[test_rows], dtype=np.float64)
    run_id = A.run_id_for(data.task_id, fold, seed, arm=ARM)
    split_version, fold_sha = A.fold_version(data.split, fold)
    record: dict[str, Any] = {
        "run_id": run_id,
        "status": "ok",
        "benchmark": data.benchmark,
        "task": data.task_id,
        "arm": ARM,
        "seed": int(seed),
        "split_version": split_version,
        "split_fold": fold,
        "config_hash": A.config_hash(config),
        "n_train": int(train_rows.size),
        "n_test": int(test_rows.size),
        "n_val": int(val_rows.size),
        "metrics": A._json_safe(A.metrics(y_true, y_pred)),
        "wall_clock_s": round(wall, 3),
        "device": chosen,
        "env_fingerprint": str(config.section("records")["env_fingerprint"]),
        "predictions": f"results/aggregate/predictions/{run_id}.parquet",
        "epochs_run": 0,
        "early_stopped": False,
        "best_epoch": None,
        "best_val_rmse": A._json_safe(head["val_rmse"]),
        "val_loss_normalized": None,
        "val_loss_unit": "rmse_original（原单位；线性头没有标准化单位的损失曲线）",
        "val_source": val_source,
        "n_rows_without_embedding": int(data.n_missing),
        "rows_without_embedding_ids": [text for text in data.missing[:20]],
        "target_scaling": "train_zscore",
        "mode": "frozen_ridge",
        "alpha": float(head["alpha"]),
        "alpha_grid": alphas,
        "alpha_scores": [[float(a), A._json_safe(score)] for a, score in head["alpha_scores"]],
        "n_features": int(data.emb.shape[1]),
        "representation": {
            "source": "B-2 polyBERT (frozen)",
            "pooling": str(config.section("representation").get("pooling")),
            "join_key": str(config.section("representation").get("join_key")),
            "embedding_content_sha256": data.embed_sha256,
        },
        "hyperparameters": {
            "head": str(head_cfg.get("kind")),
            "solver": str(head_cfg.get("solver")),
            "feature_scaling": str(head_cfg.get("feature_scaling")),
            "target_scaling": str(head_cfg.get("target_scaling")),
            "intercept": bool(head_cfg.get("intercept", False)),
        },
        "notes": (
            "polyBERT 嵌入冻结（B-2 产物）+ ridge 线性头（闭式解）；"
            "alpha 在训练内部留出上选，测试集只用于打分；"
            "线性头无随机性，三个种子的数逐位相同（这是设计，不是稳定性的证据）"
        ),
    }
    gpu = A.gpu_block(chosen, cudnn_deterministic=bool(device_cfg.get("cudnn_deterministic", True)))
    if gpu is not None:
        record["gpu"] = gpu
    if fold_sha:
        record["split_fold_sha256"] = fold_sha
    if data.n_missing:
        LOGGER.warning(
            "%s/%s：%d 行没有嵌入（前几条：%s），已从 train/val/test 中跳过并记进记录",
            data.task_id,
            fold,
            data.n_missing,
            data.missing[:3],
        )

    return A.TrainOutcome(
        record=record,
        row_id=np.asarray(data.row_id[test_rows], dtype=np.int64),
        y_true=y_true,
        y_pred=np.asarray(y_pred, dtype=np.float64),
        history=[],
    )


# ---- 5. 微调（对照，不进主表）/ Fine-tuning (a contrast, kept out of the main table) ----

def _pool_hidden(hidden: Any, mask: Any) -> Any:
    """掩码均值池化，与 `chem.polybert.embed_shard` 逐式相同。
    Masked mean pooling, the same expression as `chem.polybert.embed_shard`.

    中文：用途：微调时的前向必须与 B-2 生成冻结嵌入时的池化**完全一致**，否则"冻结 vs 微调"
          的对比里混进了池化方式的差异，差值就不再只归因于"训不训权重"。
    输入：hidden（B×T×600）、mask（B×T）。输出：B×600。副作用：无。
    English: the forward pass must pool exactly as B-2 did, otherwise the frozen-vs-finetuned
    contrast also carries a pooling difference and no longer isolates "trained or not".
    """
    import torch

    weights = mask.unsqueeze(-1).expand(hidden.size()).float()
    return (hidden * weights).sum(dim=1) / weights.sum(dim=1).clamp(min=1e-9)


def _tokenise(tokenizer: Any, texts: Sequence[str], max_tokens: int) -> Any:
    """把一批 PSMILES 编成张量（padding + truncation）。／Encode a batch of PSMILES.
    """
    return tokenizer(
        list(texts),
        padding=True,
        truncation=True,
        max_length=int(max_tokens),
        return_tensors="pt",
    )


def build_finetune_record(
    *,
    config: A.ArmConfig,
    benchmark: str,
    task_id: str,
    fold: str,
    seed: int,
    split: Any,
    n_train: int,
    n_test: int,
    n_val: int,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    wall_clock_s: float,
    device: str,
    hidden: int,
    epochs: int,
    epochs_run: int,
    best_epoch: int,
    best_val_rmse: float,
    val_source: str,
    fine_cfg: Any,
    batch_size: int,
    patience: int,
    freeze: bool,
    max_tokens: int,
    deterministic: bool,
) -> dict[str, Any]:
    """把一条微调 run 的结果拼成记录（**纯函数，不 import torch**）。
    Assemble one fine-tune record — a pure function that never imports torch.

    中文：
        用途：记录的结构是 `results/raw_runs/_schema.json` 的冻结契约，也是 C 的汇总脚本与
              D 的指标表的输入。把它从 `train_one_finetune` 里拆出来，是为了让"结构对不对"
              能在**没有 transformers/torch 的环境里**被 pytest 检查。拆之前，唯一的检查长在
              `test_finetune_smoke_writes_a_consistent_record` 上，而那条要 transformers，
              装了 transformers 的环境又没有 pytest——所以它从未进过任何一次 `pytest tests`
              （B-4 交付文档 §8 第 9 条，那是一处"绿着但什么都没查"的缺口）。
              拆之后：结构由 `tests/test_arms_polybert.py` 的纯函数用例守（core 环境即可），
              端到端一致性仍由那条冒烟用例在 polybert 环境守。
        输入：config；benchmark/task_id/fold/seed/split；n_train/n_test/n_val；y_true/y_pred；
              wall_clock_s/device；hidden/epochs/epochs_run/best_epoch/best_val_rmse/val_source；
              fine_cfg/batch_size/patience/freeze/max_tokens/deterministic。
        输出：记录 dict。**键的顺序也是契约**：重跑同一条 run 必须写出逐字节相同的 JSON，
              所以下面这个字面量的顺序不要为了"好读"去重排。
        数据来源：调用方（`train_one_finetune` 的训练结果）。
        配置来源：`config`（`config_hash` 与 `records.env_fingerprint_finetune`）。
        副作用：无。异常：ArmError —— 预测条数 ≠ 真值条数。
    English:
        The record shape is a frozen contract (it feeds C's summary script and D's metric table).
        Pulling it out of `train_one_finetune` lets pytest check the shape **without torch or
        transformers**; before the split, the only check needed transformers and lived in an
        environment with no pytest, so it never ran in any `pytest tests` (B-4 doc, §8 item 9).
        Returns the record dict. **Key order is part of the contract** — do not reorder the literal
        below for readability, or a re-run stops being byte-identical. Raises ArmError when the
        prediction count disagrees with the truth count.
    """
    if y_pred.size != y_true.size:
        raise A.ArmError(
            f"{task_id}/{fold}/seed{seed}：预测 {y_pred.size} 条 ≠ 测试集 {y_true.size} 条"
        )
    run_id = A.run_id_for(task_id, fold, seed, arm=ARM)
    split_version, fold_sha = A.fold_version(split, fold)
    record: dict[str, Any] = {
        "run_id": run_id,
        "status": "ok",
        "benchmark": benchmark,
        "task": task_id,
        "arm": ARM,
        "seed": int(seed),
        "split_version": split_version,
        "split_fold": fold,
        "config_hash": A.config_hash(config),
        "n_train": int(n_train),
        "n_test": int(n_test),
        "n_val": int(n_val),
        "metrics": A._json_safe(A.metrics(y_true, y_pred)),
        "wall_clock_s": round(float(wall_clock_s), 3),
        "device": device,
        "env_fingerprint": str(config.section("records")["env_fingerprint_finetune"]),
        "predictions": f"results/aggregate/predictions/{run_id}.parquet",
        "epochs_run": int(epochs_run),
        "early_stopped": int(epochs_run) < int(epochs),
        "best_epoch": int(best_epoch),
        "best_val_rmse": A._json_safe(best_val_rmse),
        "val_loss_normalized": None,
        "val_loss_unit": "rmse_original（已乘回 scale）",
        "val_source": val_source,
        # 微调路径不查嵌入（无 rdkit），所以这里写 None 而不是 0：0 会被读成"确认没有缺行"。
        # None reads as "not measured", whereas 0 would claim "confirmed no missing rows".
        "n_rows_without_embedding": None,
        "rows_without_embedding_ids": [],
        "target_scaling": "train_zscore",
        "mode": "finetune",
        "n_features": int(hidden),
        "hyperparameters": {
            "lr": float(fine_cfg.get("lr", 2e-5)),
            "batch_size": int(batch_size),
            "max_epochs": int(epochs),
            "patience": int(patience),
            "weight_decay": float(fine_cfg.get("weight_decay", 0.01)),
            "freeze_backbone": bool(freeze),
            "max_tokens": int(max_tokens),
        },
        "notes": (
            "polyBERT 微调（对照，不进主表）；与冻结臂同折、同留出、同测试集，"
            "所以两者之差只归因于'训不训权重'"
        ),
    }
    gpu = A.gpu_block(device, cudnn_deterministic=deterministic)
    if gpu is not None:
        record["gpu"] = gpu
    if fold_sha:
        record["split_fold_sha256"] = fold_sha
    return record


def train_one_finetune(
    data: EmbeddingData,
    config: A.ArmConfig,
    *,
    fold: str,
    seed: int,
    device: str | None = None,
    max_epochs: int | None = None,
    patience: int | None = None,
    n_level: str | None = None,
) -> A.TrainOutcome:
    """跑一条 (b) 的 run：在任务上微调 polyBERT 的最后几层 + 一个线性输出头。
    Run one (b) run: fine-tune polyBERT plus a linear output head on the task.

    中文：
        用途：**对照**，不是主表。它回答"如果让编码器在本任务上也学一学，会比冻结好多少"。
              与 (a) 的差别只有一处：权重会更新。其余（同一折、同一份内部留出、同一测试集、
              同一套指标）全部对齐，所以两者的差可以归因于"训不训权重"。
        输入：data；config；fold；seed；device；max_epochs/patience（覆盖）。
        输出：`TrainOutcome`（`arm` 仍是 `a4_polybert`，`mode` 字段写 `finetune`）。
        数据来源：`EmbeddingData`（这里只用它的 `psmiles` / `y` / 划分，**不用 `emb`**——
              微调要重新前向，冻结向量派不上用场）。
        配置来源：`finetune` 段。副作用：训练、占显存。
        异常：ArmError —— 缺 transformers/torch 或数据不足。
    English:
        A contrast, not the main table: it answers "how much better if the encoder also learns on
        this task". Only one thing differs from (a) - the weights move. Fold, inner hold-out, test
        split and metrics are identical, so the gap is attributable to training the encoder.
    """
    try:
        import torch
        from torch.utils.data import DataLoader, Dataset
    except ImportError as error:                        # pragma: no cover - 环境问题
        raise A.ArmError(f"微调要 torch：{error}") from error
    from chainworth.chem import polybert as PB

    # 签名里收 `n_level` 只为与 `train_one()` 对齐：`05_run_gpu.py --repeat-check` 把 n 档
    # 一起传下去（B-5 修过的一个真 bug：不传就检查了另一条 run），它对两条臂用同一个调用形状。
    # A4 的网格没有 n 档，所以这里值恒为 None；万一有人把 `n_levels` 抄进 A4 的配置，下面
    # 这条守卫当场报错，而不是悄悄跑出"名字带档位、实际全量"的记录。
    # `n_level` exists so both trainers share one call shape for `--repeat-check`; it is always
    # None for A4, and the guard below makes a copy-pasted `n_levels` fail loudly.
    if n_level is not None:
        raise A.ArmError(
            "A4 的网格没有 n 档：n 档只属于 B-5 的 OpenPoly 扫描（分工_B §2 B-5），"
            "要跑规模扫描请用 --arm a3_graph --config configs/experiments/openpoly_scan.yaml"
        )
    fine_cfg = config.section("finetune")
    device_cfg = config.section("device")
    val_cfg = config.section("validation")
    chosen = device or A.resolve_device(str(device_cfg.get("prefer", "cuda")))
    if chosen == "cuda" and not torch.cuda.is_available():
        chosen = "cpu"
    deterministic = bool(device_cfg.get("cudnn_deterministic", True))
    torch.backends.cudnn.deterministic = deterministic
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(deterministic)

    spec_fold = data.fold(fold)
    train_index = np.asarray(spec_fold.train_index, dtype=np.int64)
    val_index = np.asarray(spec_fold.val_index, dtype=np.int64)
    test_index = np.asarray(spec_fold.test_index, dtype=np.int64)
    val_source = "official"
    if val_index.size == 0:
        train_index, val_index = A.carve_validation(
            train_index,
            fraction=float(val_cfg["val_fraction_if_missing"]),
            seed=int(val_cfg["val_split_seed"]),
        )
        val_source = f"inner_{float(val_cfg['val_fraction_if_missing']):.0%}_of_train"

    def usable(index: np.ndarray) -> np.ndarray:
        return np.asarray([i for i in index if data.has_embedding[int(i)]], dtype=np.int64)

    train_rows, val_rows, test_rows = usable(train_index), usable(val_index), usable(test_index)
    if train_rows.size == 0 or val_rows.size == 0 or test_rows.size == 0:
        raise A.ArmError(f"{data.task_id}/{fold}：微调的 train/val/test 有空集，这条 run 不能跑")

    torch.manual_seed(int(seed))
    np.random.seed(int(seed))
    tokenizer, backbone, _auto = PB.load_model()
    backbone = backbone.to(chosen)
    max_tokens = int(fine_cfg.get("max_tokens", 512))
    freeze = bool(fine_cfg.get("freeze_backbone", False))
    for parameter in backbone.parameters():
        parameter.requires_grad_(not freeze)

    hidden = int(backbone.config.hidden_size)
    head = torch.nn.Linear(hidden, 1).to(chosen)

    y_mean, y_scale = A.target_scaler(data.y[train_rows])

    class _Rows(Dataset):
        """按行号取 (PSMILES, 标准化后的标签)。／Rows as (PSMILES, standardised label)."""

        def __init__(self, rows: np.ndarray) -> None:
            self.rows = np.asarray(rows, dtype=np.int64)

        def __len__(self) -> int:
            return int(self.rows.size)

        def __getitem__(self, position: int) -> tuple[str, float]:
            row = int(self.rows[position])
            label = (float(data.y[row]) - float(y_mean)) / float(y_scale)
            return str(data.psmiles[row]), label

    def _collate(batch: list[tuple[str, float]]) -> tuple[Any, Any]:
        texts = [item[0] for item in batch]
        labels = torch.tensor([item[1] for item in batch], dtype=torch.float32).reshape(-1, 1)
        encoded = _tokenise(tokenizer, texts, max_tokens)
        return encoded, labels

    batch_size = int(fine_cfg.get("batch_size", 32))
    train_loader = DataLoader(_Rows(train_rows), batch_size=batch_size, shuffle=True, collate_fn=_collate)
    val_loader = DataLoader(_Rows(val_rows), batch_size=batch_size, shuffle=False, collate_fn=_collate)
    test_loader = DataLoader(_Rows(test_rows), batch_size=batch_size, shuffle=False, collate_fn=_collate)

    def _forward(encoded: dict[str, Any]) -> Any:
        batch = {key: value.to(chosen) for key, value in encoded.items()}
        output = backbone(**batch)[0]
        return head(_pool_hidden(output, batch["attention_mask"]))

    parameters = list(head.parameters()) + [p for p in backbone.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(
        parameters,
        lr=float(fine_cfg.get("lr", 2e-5)),
        weight_decay=float(fine_cfg.get("weight_decay", 0.01)),
    )
    epochs = int(max_epochs if max_epochs is not None else fine_cfg.get("max_epochs", 30))
    wait = int(patience if patience is not None else fine_cfg.get("patience", 5))
    criterion = torch.nn.MSELoss()

    def _evaluate(loader: Any) -> float:
        backbone.eval()
        head.eval()
        total, count = 0.0, 0
        with torch.no_grad():
            for encoded, labels in loader:
                predicted = _forward(encoded)
                total += float(torch.sum((predicted.cpu() - labels) ** 2))
                count += int(labels.numel())
        return math.sqrt(total / max(count, 1)) * float(y_scale)

    history: list[dict[str, float]] = []
    best_state: dict[str, Any] | None = None
    best_val = float("inf")
    best_epoch: int | None = None
    stale = 0
    started = time.perf_counter()
    for epoch in range(epochs):
        backbone.train()
        head.train()
        running, seen = 0.0, 0
        for encoded, labels in train_loader:
            optimizer.zero_grad(set_to_none=True)
            predicted = _forward(encoded)
            loss = criterion(predicted, labels.to(chosen))
            loss.backward()
            optimizer.step()
            running += float(loss.detach().cpu()) * int(labels.numel())
            seen += int(labels.numel())
        val_rmse = _evaluate(val_loader)
        history.append(
            {
                "epoch": float(epoch),
                "train_loss": math.sqrt(running / max(seen, 1)) * float(y_scale),
                "val_loss": float(val_rmse),
            }
        )
        if val_rmse < best_val - 1e-9:
            best_val = val_rmse
            best_epoch = epoch + 1
            best_state = {
                "backbone": {key: value.detach().clone() for key, value in backbone.state_dict().items()},
                "head": {key: value.detach().clone() for key, value in head.state_dict().items()},
            }
            stale = 0
        else:
            stale += 1
            if wait > 0 and stale >= wait:
                break
    wall = time.perf_counter() - started

    if best_state is not None:
        backbone.load_state_dict(best_state["backbone"])
        head.load_state_dict(best_state["head"])
    backbone.eval()
    head.eval()
    collected: list[float] = []
    with torch.no_grad():
        for encoded, _labels in test_loader:
            predicted = _forward(encoded)
            collected.extend(predicted.detach().cpu().numpy().reshape(-1).tolist())
    y_pred = np.asarray(collected, dtype=np.float64) * float(y_scale) + float(y_mean)
    y_true = np.asarray(data.y[test_rows], dtype=np.float64)
    if y_pred.size != y_true.size:
        raise A.ArmError(
            f"{data.task_id}/{fold}/seed{seed}：预测 {y_pred.size} 条 ≠ 测试集 {y_true.size} 条"
        )

    epochs_run = len(history)
    # 记录由纯函数拼（`build_finetune_record`）：本函数要 torch，而记录结构必须有能在
    # core 环境里跑的用例守着——见该函数的 docstring。
    # The record is assembled by a pure function so its shape can be tested without torch.
    record = build_finetune_record(
        config=config,
        benchmark=data.benchmark,
        task_id=data.task_id,
        fold=fold,
        seed=seed,
        split=data.split,
        n_train=int(train_rows.size),
        n_test=int(test_rows.size),
        n_val=int(val_rows.size),
        y_true=y_true,
        y_pred=y_pred,
        wall_clock_s=wall,
        device=chosen,
        hidden=hidden,
        epochs=epochs,
        epochs_run=epochs_run,
        best_epoch=best_epoch,
        best_val_rmse=best_val,
        val_source=val_source,
        fine_cfg=fine_cfg,
        batch_size=batch_size,
        patience=wait,
        freeze=freeze,
        max_tokens=max_tokens,
        deterministic=deterministic,
    )

    # 一个进程连跑多条 run 时，CUDA 缓存分配器不把空闲块还给驱动，单轮会变慢：实测同一条
    # pb26_polymetrix_tg 在"已经跑过 18 个其他任务"的进程里是 37–39 秒/轮，在刚启动的进程里是
    # 27–31 秒/轮。所以这里显式摘掉引用再释放缓存，让下一条 run 从干净状态开始。
    # 只释放显存，不动任何数值——empty_cache() 不参与计算，同一个 seed 的 R² 逐位不变。
    # **不要**把墙钟差异算到这条上：300 秒与 1,163 秒的差别几乎全来自 epochs_run 是 11 还是 30
    # （同一进程内 s/ep 是同一个数），不是显存换页。2026-11-12 第一版归因写错了，据此勘误。
    # A process that trains run after run never hands free blocks back to the driver, and the
    # per-epoch cost rises: 37-39 s/epoch inside a process that had already run 18 other tasks
    # versus 27-31 s/epoch in a fresh one (measured 2026-11-12). Dropping the references and
    # emptying the cache makes the next run start clean; no number changes. Do NOT blame wall
    # clock on this: 300 s vs 1,163 s is about epochs_run (11 or 30), not about paging - the
    # seconds-per-epoch figure is flat within one process.
    del _forward, _evaluate, optimizer, parameters, backbone, head
    if chosen == "cuda":
        torch.cuda.empty_cache()

    return A.TrainOutcome(
        record=record,
        row_id=np.asarray(data.row_id[test_rows], dtype=np.int64),
        y_true=y_true,
        y_pred=y_pred,
        history=history,
    )


# ---- 7. 两种模式的对照表 / The frozen-vs-finetuned comparison ----

def write_comparison(
    *,
    frozen_dir: str | Path | None = None,
    finetune_dir: str | Path | None = None,
    target: str | Path | None = None,
) -> dict[str, Any]:
    """把 (a) 与 (b) 同任务同折同种子的结果并成一张逐条对照表，写 CSV；返回汇总。
    Join (a) and (b) run by run (same task/fold/seed) into one table, write the CSV, return a summary.

    中文：
        用途：分工_B §2 B-4 验收②——"(a) 与 (b) 若都跑了，同一 split/seed 上比较，差异写进主表
              备注"。这里把"差异"变成一个可核对的文件：`results/aggregate/a4_finetune_vs_frozen.csv`。
              每一行是一个 (task, fold, seed)，两列各一个 R²，一列是差值。
              只有两边的 run 都在时才对这一行——缺一边就跳过并计数，绝不拿 0 去填。
        输入：frozen_dir —— A4 冻结臂的记录目录（默认 `results/raw_runs/`，按 `arm` 过滤）；
              finetune_dir —— 微调记录目录（默认 `results/aggregate/a4_finetune/raw_runs/`）；
              target —— CSV 路径（默认 `results/aggregate/a4_finetune_vs_frozen.csv`）。
        输出：dict，键 `rows`（逐条）、`by_task`（每任务的均值）、`n_matched`、`n_missing_frozen`、
              `n_missing_finetune`、`path`。
        数据来源：两个记录目录。副作用：写一个 CSV（UTF-8 / LF）。异常：ArmError —— 两边都空。
    English:
        Criterion 2 asks for the (a)-vs-(b) gap on the same split and seed; this turns "the gap" into
        a file anyone can check. One row per (task, fold, seed), one R^2 per column and their
        difference. A row is only emitted when both sides exist - a missing side is counted, never
        filled with 0.
    """
    frozen_root = Path(frozen_dir) if frozen_dir is not None else A.RAW_RUNS_DIR
    fine_root = (
        Path(finetune_dir)
        if finetune_dir is not None
        else (REPO_ROOT / "results" / "aggregate" / "a4_finetune" / "raw_runs")
    )
    out_path = (
        Path(target)
        if target is not None
        else (REPO_ROOT / "results" / "aggregate" / "a4_finetune_vs_frozen.csv")
    )

    def _load(directory: Path, want: str) -> dict[tuple[str, str, int], dict[str, Any]]:
        pool: dict[tuple[str, str, int], dict[str, Any]] = {}
        if not directory.is_dir():
            return pool
        for path in sorted(directory.glob("*.json")):
            if path.name.startswith("_"):
                continue
            payload = json.loads(path.read_text(encoding="utf-8"))
            if payload.get("arm") != "a4_polybert":
                continue
            if str(payload.get("mode")) != want:
                continue
            if payload.get("status") != "ok":
                continue
            pool[(str(payload["task"]), str(payload["split_fold"]), int(payload["seed"]))] = payload
        return pool

    frozen = _load(frozen_root, "frozen_ridge")
    finetuned = _load(fine_root, "finetune")
    if not frozen and not finetuned:
        raise A.ArmError(f"对照表没法生成：{frozen_root} 与 {fine_root} 里都没有 a4_polybert 的记录")

    matched = sorted(set(frozen) & set(finetuned))
    rows: list[dict[str, Any]] = []
    for key in matched:
        task, fold, seed = key
        left, right = frozen[key], finetuned[key]
        rows.append(
            {
                "task": task,
                "benchmark": left["benchmark"],
                "split_fold": fold,
                "seed": seed,
                "r2_frozen": round(float(left["metrics"]["r2"]), 6),
                "r2_finetune": round(float(right["metrics"]["r2"]), 6),
                "delta_r2": round(float(right["metrics"]["r2"]) - float(left["metrics"]["r2"]), 6),
                "rmse_frozen": round(float(left["metrics"]["rmse"]), 6),
                "rmse_finetune": round(float(right["metrics"]["rmse"]), 6),
                "mae_frozen": round(float(left["metrics"]["mae"]), 6),
                "mae_finetune": round(float(right["metrics"]["mae"]), 6),
                "epochs_finetune": int(right.get("epochs_run") or 0),
                "wall_frozen_s": float(left["wall_clock_s"]),
                "wall_finetune_s": float(right["wall_clock_s"]),
            }
        )

    by_task: dict[str, dict[str, float]] = {}
    for row in rows:
        bucket = by_task.setdefault(
            row["task"], {"n": 0.0, "r2_frozen": 0.0, "r2_finetune": 0.0, "delta_r2": 0.0}
        )
        bucket["n"] += 1.0
        bucket["r2_frozen"] += row["r2_frozen"]
        bucket["r2_finetune"] += row["r2_finetune"]
        bucket["delta_r2"] += row["delta_r2"]
    for bucket in by_task.values():
        for name in ("r2_frozen", "r2_finetune", "delta_r2"):
            bucket[name] = round(bucket[name] / max(bucket["n"], 1.0), 6)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    columns = (
        "task",
        "benchmark",
        "split_fold",
        "seed",
        "r2_frozen",
        "r2_finetune",
        "delta_r2",
        "rmse_frozen",
        "rmse_finetune",
        "mae_frozen",
        "mae_finetune",
        "epochs_finetune",
        "wall_frozen_s",
        "wall_finetune_s",
    )
    import csv

    with out_path.open("w", encoding="utf-8", newline="\n") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(columns), lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    return {
        "rows": rows,
        "by_task": by_task,
        "n_matched": len(rows),
        "n_missing_frozen": len(set(finetuned) - set(frozen)),
        "n_missing_finetune": len(set(frozen) - set(finetuned)),
        "path": A._repo_relative(out_path),
    }


# ---- 8. 与 A3 同名的公开 API / Public API with the same names as A3 ----
#
# `scripts/05_run_gpu.py` 按 `--arm` 换模块，然后调同名函数；这里把 A3 的通用实现转出来，
# 只把 `load_arm_config` 的默认路径换成 A4 的配置。**不复制实现**——`enumerate_runs` 里那套
# "白名单里出现配置没有的任务就报错"的规则，两条臂必须是同一份，否则一边静默跳过、一边报错，
# 对账时会得到两个不同的"跑完了"。
# Entry 05 swaps modules by `--arm` and calls same-named functions; these re-export A3's generic
# implementation, changing only the default config path. Nothing is copied: the whitelist rule in
# `enumerate_runs` must be one implementation, or the two arms disagree about "finished".

ArmConfig = A.ArmConfig
ArmError = A.ArmError
RunKey = A.RunKey
TrainOutcome = A.TrainOutcome
enumerate_runs = A.enumerate_runs
run_id_for = A.run_id_for
carve_validation = A.carve_validation
metrics = A.metrics
config_hash = A.config_hash
fold_version = A.fold_version
target_scaler = A.target_scaler
gpu_block = A.gpu_block
resolve_device = A.resolve_device
write_run_record = A.write_run_record
write_predictions = A.write_predictions
append_gpu_hours = A.append_gpu_hours
record_problems = A.record_problems


def load_arm_config(path: str | Path | None = None) -> A.ArmConfig:
    """读 A4 的臂配置（默认 `configs/arms/a4_polybert.yaml`）。
    Load A4's arm config (defaults to configs/arms/a4_polybert.yaml).

    中文：输入：path（None 时用 A4 默认路径）。输出：`ArmConfig`。副作用：只读。
          异常：ArmError —— 文件缺失或不合法。校验规则与 A3 共用（按 arm 分派）。
    English: Args: path. Returns: an ArmConfig. Raises: ArmError on a missing or bad file. The
    validation rules are shared with A3 and dispatched by arm.
    """
    return A.load_arm_config(path if path is not None else DEFAULT_CONFIG_PATH)

# ---- 6. 整轮编排：复用 A3 的 run_grid / Reuse A3's run_grid ----

def build_task_data_finetune(
    task_id: str, *, limit: int | None = None, use_cache: bool = True
) -> EmbeddingData:
    """微调路径的数据准备：只读 pack 与划分，**不查嵌入、不碰 RDKit**。
    Data prep for the fine-tune path: read the pack and split only, no embeddings, no RDKit.

    中文：
        用途：`DSC6020_Project_Polybert` 环境有 transformers 但没有 rdkit，而规范化必须用 RDKit
              （见 `build_task_data` 的 `need_embedding` 说明）。微调只拿 `psmiles` 原文去
              tokenize，冻结向量派不上用场，所以这条路径整段跳过嵌入。
              签名与 `build_task_data` 相同，所以 `run_grid` 只换一个函数引用。
        输入：task_id；limit；use_cache。输出：`EmbeddingData`（`emb` 是 0 列）。
        副作用：读盘。异常：ArmError —— 缺 pack。
    English:
        The polyBERT environment has transformers but no rdkit, and canonicalisation needs RDKit, so
        this path skips embeddings entirely - fine-tuning tokenises the raw PSMILES. Same signature
        as `build_task_data`, so `run_grid` only swaps a function reference.
    """
    return build_task_data(task_id, limit=limit, use_cache=use_cache, need_embedding=False)


def run_grid(
    config: A.ArmConfig,
    *,
    tasks: Sequence[str] | None = None,
    folds: Sequence[str] | None = None,
    seeds: Sequence[int] | None = None,
    limit: int | None = None,
    device: str | None = None,
    max_epochs: int | None = None,
    patience: int | None = None,
    resume: bool = True,
    out_root: Path | None = None,
    finetune: bool = False,
) -> list[dict[str, Any]]:
    """按网格逐条跑 A4，落记录与预测；`finetune=True` 时改用微调那条。
    Run A4's grid writing records and predictions; `finetune=True` switches to the fine-tune path.

    中文：
        用途：**不重写编排**。按任务分组、`resume` 的判据、"failed 不算跑过"、写记录/预测/
        墙钟的次序，全部来自 `graph.run_grid()`；本函数只做两件事——把 `build_task_data`
        与 `train_one` 换成 A4 的，以及在微调模式下换掉 trainer。
        这样一来，"哪些算跑完了"这件事两条臂只有一份实现。
        输入：同 `graph.run_grid()`；`finetune` —— 走 (b)。
        输出：本次写出的记录列表。
        副作用：训练、写文件。异常：同 `graph.run_grid()`。
    English:
        The orchestration is **not** rewritten: task grouping, the resume rule, "failed is not
        done", and the write order all come from `graph.run_grid()`. This function only swaps in
        A4's builder and trainer, so "what counts as finished" has exactly one implementation.
    """
    trainer = train_one_finetune if finetune else train_one
    builder = build_task_data_finetune if finetune else build_task_data
    return A.run_grid(
        config,
        tasks=tasks,
        folds=folds,
        seeds=seeds,
        limit=limit,
        device=device,
        max_epochs=max_epochs,
        patience=patience,
        resume=resume,
        out_root=out_root,
        builder=builder,
        trainer=trainer,
    )
