"""阶段入口 03c：polyBERT 结构级嵌入（B-2）。
Stage entry 03c: polyBERT structure-level embeddings (B-2).

中文：
    用途：B-2 的唯一入口。跑完得到三样东西：
          data/processed/polybert/embeddings.npy     11,494 × 600，float32，行序按规范式排序
          data/processed/polybert/index.parquet      行号 ↔ 结构 ID ↔ 输入串 ↔ token 统计
          data/processed/polybert/manifest.json      维度、覆盖数、模型版本与权重哈希、两份哈希
          data/manifests/polybert_manifest.csv       上面那份清单的一行（入库，便于看历史）
          中间产物 `polybert/shards/shard_NNNNN.npz` 保留：断了重跑只补缺的片。
    依赖：`data/processed/structures/psmiles.parquet`（A-6，02 的产物）；torch / transformers
          只在 `DSC6020_Project_Polybert` 环境里有。逻辑全在 `chainworth/chem/polybert.py`，
          本文件只做参数、日志与"跳过哪些片"的判断。
    配置：无 yaml。模型版本、维度、分片大小是 `polybert.py` 里的常量（改就是改口径，见 D-31）。
    对外：命令行。`--verify` 只读，把 B-2 的三条验收标准跑成检查；`--limit` 的产物落到
          `_partial_<N>/`，从路径上就与全量分开。
    被谁用：人工（阶段运行）、B-4 的 A4 臂（读 embeddings + index）、CI（`--verify` 回归）。

English:
    Purpose: B-2's only entry point. A run produces embeddings.npy (11,494 x 600 float32, rows in
    canonical-form order), index.parquet (row -> structure id -> input string -> token stats),
    manifest.json (dimension, coverage, model revision, weight hash, artifact hashes) and one row in
    data/manifests/polybert_manifest.csv. The shard files are kept so a rerun only fills the gaps.
    Depends on A-6's structure table; torch/transformers exist only in DSC6020_Project_Polybert. All
    logic lives in chainworth/chem/polybert.py - this file parses arguments, logs and decides which
    shards to skip.
    Config: no yaml; the model revision, dimension and shard size are module constants (changing
    them changes the definition, see D-31).
    Public API: the command line. `--verify` is read-only and turns B-2's three acceptance criteria
    into checks; `--limit` writes into `_partial_<N>/` so a smoke run is separated by path.
    Callers: humans running stage 03c, B-4's A4 arm and CI.

给队友的一句话 ／ One line for teammates
-----------------------------------------
`data/processed/polybert/` **不在仓库里**。拿到提交后在本机跑

    python scripts/03c_polybert.py          # 约 1 分钟（RTX 4070）

就得到同一批嵌入；`--verify --recompute 50` 会重算 50 行并逐行比对，证明"你重出来的和我的一样"。
注意**不要**用 `--limit` 的产物冒充全量：它落在 `_partial_<N>/` 里，路径就写着不是全量。
"""

from __future__ import annotations

import argparse
import dataclasses
import logging
import sys
from pathlib import Path

import numpy as np

from chainworth.chem import polybert as P

LOGGER = logging.getLogger("chainworth.scripts.03c")


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """命令行参数。/ Command-line arguments.

    中文：输入：argv（None 时读 sys.argv）。输出：Namespace。数据来源：命令行。
          副作用：无。异常：参数不合法时由 argparse 抛 SystemExit。
    English: Args: argv. Returns: Namespace. Raises: SystemExit from argparse.
    """
    parser = argparse.ArgumentParser(
        prog="03c_polybert.py",
        description="B-2：polyBERT 结构级嵌入（600 维，分片可续跑）",
    )
    parser.add_argument("--limit", type=int, metavar="N", help="只嵌入排序后的前 N 个结构（试跑，落 _partial_N/）")
    parser.add_argument("--shard-size", type=int, default=P.DEFAULT_SHARD_SIZE, metavar="N", help="每片行数")
    parser.add_argument("--batch-size", type=int, default=P.DEFAULT_BATCH_SIZE, metavar="N", help="前向批大小")
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto", help="设备；auto = 有卡就用卡")
    parser.add_argument("--model", default=P.MODEL_REPO, help="HuggingFace 仓库（换官方权重时改这里）")
    parser.add_argument("--revision", default=P.MODEL_REVISION, help="仓库 commit sha（锁版本）")
    parser.add_argument("--merge-only", action="store_true", help="不跑模型，只把已有分片合并成产物")
    parser.add_argument("--verify", action="store_true", help="只读校验已有产物（含行序与未知 token）")
    parser.add_argument("--recompute", type=int, default=0, metavar="N", help="配合 --verify：重算 N 行逐行比对")
    parser.add_argument("--dry-run", action="store_true", help="只打印计划，不加载模型、不写文件")
    parser.add_argument("-v", "--verbose", action="store_true", help="更详细的日志 / debug logging")
    return parser.parse_args(argv)


def _pick_device(requested: str) -> str:
    """把 `auto` 解析成实际设备，并把"要卡却没卡"当成错误而不是悄悄降级。
    Resolve `auto` into a real device, and treat "asked for cuda but no cuda" as an error.

    中文：用途：交付纪律要求"不许写死 GPU"，也不许悄悄把 CPU 结果标成 cuda——所以显式要卡
          而不可用时直接失败。
          输入：requested（auto/cuda/cpu）。输出："cuda" 或 "cpu"。数据来源：torch。
          副作用：无。异常：要 cuda 但没有时抛 `P.PolyBertError`。
    English: Purpose: no hard-coded GPU, and no silently CPU-labelled-as-cuda either - an explicit
             request for cuda with no cuda fails loudly.
    """
    import torch

    if requested == "cpu":
        return "cpu"
    if torch.cuda.is_available():
        return "cuda"
    if requested == "cuda":
        raise P.PolyBertError("指定了 --device cuda，但 torch.cuda.is_available() 为 False")
    LOGGER.warning("没有可用 GPU，退回 CPU（会明显变慢，产物清单会如实记 cpu）")
    return "cpu"


def _run(args: argparse.Namespace) -> int:
    """一次完整运行：分词统计 → 分片 → 合并 → 清单 → 轻量自检。
    One full run: token stats -> shards -> merge -> manifest -> light self-check.
    """
    started = P.timer()
    index = P.load_structures(limit=args.limit)
    out_dir = P.resolve_output_dir(args.limit)
    shard_dir = out_dir / P.SHARDS_NAME
    ranges = P.plan_shards(len(index), args.shard_size)
    LOGGER.info(
        "结构 %d 个，分 %d 片（每片 %d），产物目录 %s",
        len(index),
        len(ranges),
        args.shard_size,
        out_dir.relative_to(P.REPO_ROOT),
    )
    if args.limit is not None:
        LOGGER.warning("--limit %d：产物写进 %s，**不是全量**", args.limit, out_dir.name)

    tokenizer, model, device = P.load_model(repo=args.model, revision=args.revision)
    if args.device != "auto":
        device = _pick_device(args.device)
        model.to(device)

    lengths, unknowns = P.token_stats(tokenizer, index.inputs)
    index = dataclasses.replace(index, n_tokens=lengths, n_unk=unknowns)
    LOGGER.info(
        "分词：最长 %d token，达上限 %d 条，未知 token %d 个",
        int(lengths.max()) if lengths.size else 0,
        int((lengths >= P.MAX_TOKENS).sum()),
        int(unknowns.sum()),
    )
    if int(unknowns.sum()) > 0:
        LOGGER.warning("有 %d 条含未知 token——先看输入串形态（裸 * 会变成 <unk>）", int((unknowns > 0).sum()))

    reused = 0
    for number, (start, stop) in enumerate(ranges):
        path = P.shard_path(shard_dir, number)
        expected = index.structure_id[start:stop]
        if P.shard_matches(path, expected):
            reused += 1
            continue
        vectors, tokens, unknown = P.embed_shard(
            tokenizer, model, device, index.inputs[start:stop], batch_size=args.batch_size
        )
        P.save_shard(
            path,
            {
                "row_start": start,
                "row_stop": stop,
                "structure_id": expected,
                "n_tokens": tokens,
                "n_unk": unknown,
                "vectors": vectors,
            },
        )
        LOGGER.info("  片 %d/%d 完成 [%d, %d)", number + 1, len(ranges), start, stop)
    if reused:
        LOGGER.info("复用已有分片 %d/%d（续跑）", reused, len(ranges))

    return _merge_and_finish(args, index, out_dir, shard_dir, device, started)


def _merge_and_finish(
    args: argparse.Namespace,
    index: P.StructureIndex,
    out_dir: Path,
    shard_dir: Path,
    device: str,
    started: float,
) -> int:
    """合并分片、写三份产物、追加清单行，然后跑一遍轻量自检。
    Merge the shards, write the three artifacts, append the manifest row, then self-check.
    """
    paths = sorted(shard_dir.glob("shard_*.npz"))
    merged = P.merge_shards(paths, len(index))
    vectors = merged["vectors"]
    if not np.array_equal(merged["structure_id"], index.structure_id):
        raise P.PolyBertError("合并出来的结构 ID 与当前行序不同——分片来自别的版本，删掉 shards/ 重跑")

    P.save_embeddings(out_dir / P.EMBEDDINGS_NAME, vectors)
    index = dataclasses.replace(
        index,
        n_tokens=merged["n_tokens"].astype(np.int32),
        n_unk=merged["n_unk"].astype(np.int32),
    )
    P.write_index(out_dir / P.INDEX_NAME, index)

    manifest = P.build_manifest(
        index,
        vectors,
        device=device,
        wall_clock_s=P.timer() - started,
        n_truncated=int((index.n_tokens >= P.MAX_TOKENS).sum()),
        shard_size=args.shard_size,
        is_partial=args.limit is not None,
        extra={"index_sha256": P.sha256_of_file(out_dir / P.INDEX_NAME)},
    )
    P.write_manifest(out_dir / P.MANIFEST_NAME, manifest)
    if args.limit is None:
        P.append_manifest_csv(P.MANIFEST_CSV, manifest)

    LOGGER.info(
        "写入 %s：%s，%d 维，%.1f 秒，device=%s",
        out_dir.relative_to(P.REPO_ROOT),
        f"{vectors.shape[0]} 行",
        vectors.shape[1],
        manifest["wall_clock_s"],
        device,
    )
    problems = P.verify_outputs(out_dir)
    if problems:
        for item in problems:
            LOGGER.error("自检未过：%s", item)
        return 1
    LOGGER.info("自检通过：行数、行序、维度、NaN、未知 token 五项都对得上")
    return 0


def _dry_run(args: argparse.Namespace) -> int:
    """只打印计划（不加载模型、不写文件），用来在开跑前确认路径与规模。
    Print the plan only - no model, no writes - to confirm paths and sizes before running.
    """
    index = P.load_structures(limit=args.limit)
    out_dir = P.resolve_output_dir(args.limit)
    ranges = P.plan_shards(len(index), args.shard_size)
    LOGGER.info("会读：%s", P.STRUCTURES_PATH.relative_to(P.REPO_ROOT))
    LOGGER.info("会写：%s/{%s, %s, %s}", out_dir.relative_to(P.REPO_ROOT), P.EMBEDDINGS_NAME, P.INDEX_NAME, P.MANIFEST_NAME)
    LOGGER.info("结构 %d 个 × %d 维 = %.1f MiB（float32）", len(index), P.EMBED_DIM, len(index) * P.EMBED_DIM * 4 / 2**20)
    LOGGER.info("分 %d 片，每片最多 %d 行；已存在的有效分片会被跳过", len(ranges), args.shard_size)
    LOGGER.info("模型 %s@%s（权重 sha256 %s…）", args.model, args.revision[:8], P.MODEL_WEIGHTS_SHA256[:8])
    LOGGER.info("首行结构：%s → %s", index.canonical[0], index.inputs[0])
    return 0


def _verify(args: argparse.Namespace) -> int:
    """只读校验。`--recompute N` 会真的再跑一遍模型比对数值。
    Read-only verification; `--recompute N` actually re-runs the model and compares values.
    """
    out_dir = P.resolve_output_dir(args.limit)
    tokenizer = model = None
    device = "cpu"
    if args.recompute > 0:
        tokenizer, model, device = P.load_model(repo=args.model, revision=args.revision)
        if args.device != "auto":
            device = _pick_device(args.device)
            model.to(device)
    problems = P.verify_outputs(out_dir, recompute=args.recompute, tokenizer=tokenizer, model=model, device=device)
    if problems:
        for item in problems:
            LOGGER.error("未过：%s", item)
        return 1
    LOGGER.info("全部通过：%s", out_dir.relative_to(P.REPO_ROOT))
    return 0


def main(argv: list[str] | None = None) -> int:
    """入口。/ Entry point."""
    args = _parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(message)s",
        stream=sys.stdout,
    )
    try:
        if args.verify:
            return _verify(args)
        if args.dry_run:
            return _dry_run(args)
        if args.merge_only:
            index = P.load_structures(limit=args.limit)
            out_dir = P.resolve_output_dir(args.limit)
            return _merge_and_finish(args, index, out_dir, out_dir / P.SHARDS_NAME, "unknown", P.timer())
        return _run(args)
    except P.PolyBertError as error:
        LOGGER.error("%s", error)
        return 1
    except FileNotFoundError as error:
        LOGGER.error("缺文件：%s", error)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
