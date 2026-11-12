"""阶段入口 05：GPU 臂批量运行（A3 图神经网络主扫 / A4 polyBERT 迁移臂；B-3 / B-4 / B-5 共用）。
Stage entry 05: batch GPU arm runs (A3 graph sweep / A4 polyBERT transfer arm; shared by B-3/B-4/B-5).

中文：
    用途：B 的 **05_run_gpu.py**——README §4.2.1 预先把这个编号留给"GNN / polyBERT 臂的 GPU 批量"。
          `--arm` 选臂，两个臂**共用一个编排**（`graph.run_grid` 的分组、resume、写盘次序），
          所以"哪些算跑完了"只有一份实现，不会两条臂各说各话。
          跑完得到：
          configs/arms/<arm>.yaml                      冻结的臂配置（本脚本只读，`config_hash` 是它的字节哈希）
          results/raw_runs/<run_id>.json               一条 run 一份记录（入库）
          results/aggregate/predictions/<run_id>.parquet  测试集预测 row_id + y_pred（入库）
          results/aggregate/gpu_hours.csv              每条 run 的实测墙钟（两臂共用一张表，按 run_id 覆盖）
    怎么用（按需要挑一条）：
          python scripts/05_run_gpu.py --dry-run                    # 只打印 48 条网格，不训练
          python scripts/05_run_gpu.py --limit 600 --tasks pb_t3_iwlt --seeds 0 --max-epochs 3
                                                                    # 冒烟：两分钟内跑通
          python scripts/05_run_gpu.py --require-gpu                # A3 正式主扫：48 条，GPU
          python scripts/05_run_gpu.py --arm a4_polybert --require-gpu
                                                                    # A4 主表：48 条（冻结 + ridge）
          python scripts/05_run_gpu.py --arm a4_polybert --finetune --require-gpu
                                                                    # A4 对照：48 条微调，写到独立目录
          python scripts/05_run_gpu.py --config configs/experiments/openpoly_scan.yaml --require-gpu
                                                                    # B-5：OpenPoly 225 条（3 性质 × 5 档 × 5 折 × 3 种子）
          python scripts/05_run_gpu.py --config configs/experiments/openpoly_scan.yaml \
              --tasks op_tm --n-levels 30 --seeds 0 --require-gpu  # B-5 单条：只跑一档一串
    依赖：A3 要 `environment/gnn.yml`；A4 要 `environment/polybert.yml`（那个环境**没有 pandas /
          rdkit**，所以 `arms/graph.py` 与 `chem/features.py` 的顶层依赖已被裁掉，见 D-34）。
    配置：`configs/arms/<arm>.yaml`（B-5 用 `configs/experiments/openpoly_scan.yaml`，它同时是臂
          配置与实验配置，所以 `config_hash` 的语义与 B-3/B-4 一样是单文件字节哈希，见 D-35 ⑥）。
    对外：命令行。所有记录由训练代码落盘，**不许手写**（红线 4）。
    被谁用：人工（D1 夜里跑）、B-5（同一脚本 + `openpoly_scan.yaml`）、CI（`--dry-run`）。

English:
    Purpose: entry 05 for both GPU arms. `--arm` selects the arm; both share one orchestration
    (`graph.run_grid`), so "what counts as finished" has exactly one implementation. Produces one
    record per run, one predictions parquet per run, and the shared wall-clock table.
    Usage: `--dry-run` prints the grid; `--limit/--max-epochs` is a smoke run; `--require-gpu` is the
    real sweep; `--arm a4_polybert` switches to the transfer arm; `--finetune` runs the contrast
    into its own directory instead of the frozen arm's records.
    Depends on gnn.yml (A3) or polybert.yml (A4). Records are written by the training code only.
    Callers: humans, B-5, CI (`--dry-run`).

一条命令一个结论 ／ One command, one conclusion
-----------------------------------------------
`--dry-run` 的数只有一种解释：**网格对不对**。它不会训练、不会写文件，所以 CI 里可以放心跑。
`--repeat-check 2` 的数也只有一种解释：**同一份配置、同一个种子跑两次，预测差多少**——验收④要的
就是"< 1e-3"，不是"大致一致"。A4 的线性头是闭式解，这个数必然是 0；跑它是为了证明"必然"，
不是为了让数字好看。微调臂不一样，它有随机初始化、批次顺序与 dropout，这个数可能不为 0，
所以 `--finetune --repeat-check 2` 是真检查（结果另写 `a4_finetune_repeatability.json`）。
`--dry-run` answers exactly one question (is the grid right) and writes nothing, so CI may run it.
`--repeat-check 2` answers exactly one question (how far apart are two runs of the same seed).
For A4's closed-form head the answer is necessarily 0; running it proves the "necessarily".
The finetuned arm is a real test instead: random init, batch order and dropout could make it
nonzero, and `--finetune --repeat-check 2` writes `a4_finetune_repeatability.json`.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

# 让 `python scripts/05_run_gpu.py` 直接可用：把仓库根的 src/ 放进 sys.path。
# 不做 pip install -e（P0 冻结的环境里不该出现本仓库的 egg-info）。
# Make the script runnable without installing the package: put src/ on sys.path. No
# `pip install -e`, so the frozen P0 environments stay free of this repo's egg-info.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from chainworth.arms import graph as A  # noqa: E402  —— sys.path 之后导入 / import after sys.path

LOGGER = logging.getLogger("chainworth.scripts.05")

# 两个臂各一份模块；它们导出同名 API（`load_arm_config` / `enumerate_runs` / `run_grid` /
# `build_task_data` / `train_one`），所以下面的编排代码对两者是同一段。
# One module per arm; both export the same API, so the orchestration below is shared.
ARM_MODULES = {"a3_graph": "graph", "a4_polybert": "polybert"}
ARM_CONFIGS = {
    "a3_graph": "configs/arms/a3_graph.yaml",
    "a4_polybert": "configs/arms/a4_polybert.yaml",
}


def _load_arm(arm_name: str):
    """按名字取臂模块（默认 A3）。／Fetch the arm module by name (A3 by default).

    中文：输入：arm_name。输出：模块。数据来源：`chainworth.arms`。副作用：import。
          异常：SystemExit —— 名字不在 `ARM_MODULES` 里。
    English: Args: arm_name. Returns: the module. Raises SystemExit on an unknown name.
    """
    module_name = ARM_MODULES.get(arm_name)
    if module_name is None:
        raise SystemExit(f"--arm 只能是 {sorted(ARM_MODULES)}，收到 {arm_name!r}")
    if module_name == "polybert":
        from chainworth.arms import polybert as module
        return module
    return A


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """命令行参数。/ Command-line arguments.

    中文：输入：argv（None 时读 sys.argv）。输出：Namespace。数据来源：命令行。
          副作用：无。异常：参数不合法时由 argparse 抛 SystemExit。
    English: Args: argv. Returns: Namespace. Raises: SystemExit from argparse.
    """
    parser = argparse.ArgumentParser(
        prog="05_run_gpu.py",
        description="B-3/B-4：GPU 臂批量运行（A3 chemprop MPNN / A4 polyBERT 迁移），逐 run 落记录与预测",
    )
    parser.add_argument(
        "--arm",
        choices=tuple(ARM_MODULES),
        default="a3_graph",
        help="跑哪条臂（默认 a3_graph；a4_polybert 是 polyBERT 迁移臂）",
    )
    parser.add_argument(
        "--config",
        default=None,
        help="臂配置路径（默认按 --arm 取 configs/arms/<arm>.yaml）",
    )
    parser.add_argument(
        "--finetune",
        action="store_true",
        help="A4 专用：跑微调对照而不是冻结+ridge。产物写到 results/aggregate/a4_finetune/，"
             "不写 results/raw_runs/（那是冻结臂的位置）",
    )
    parser.add_argument("--tasks", nargs="+", metavar="TASK", help="只跑这些任务（默认跑配置里的全部）")
    parser.add_argument("--folds", nargs="+", metavar="FOLD", help="只跑这些折（默认跑配置里的全部）")
    parser.add_argument("--seeds", nargs="+", type=int, metavar="S", help="只跑这些种子（默认 {0,1,2}）")
    parser.add_argument(
        "--extra-folds",
        nargs="+",
        metavar="FOLD",
        help="配置网格之外的折（B-5 的 5–9 折敏感性分析；只对 openpoly 有意义）。"
             "**不能把这些折写进配置**：配置的字节哈希就是 225 条的 config_hash，改它等于换协议，"
             "`write_n_index()` 会把已交付的 225 条当成旧协议剔除。折是数据，不是超参",
    )
    parser.add_argument(
        "--n-levels",
        nargs="+",
        metavar="LEVEL",
        help="只跑这些 n 档（30 60 120 240 all；只有带 `n_levels` 的配置如 openpoly_scan.yaml 有意义）。"
             "档位是 A 冻结的，改档位 = 换协议",
    )
    parser.add_argument("--limit", type=int, metavar="N", help="每个任务只取前 N 行（冒烟；不进正式产物）")
    parser.add_argument("--max-epochs", type=int, metavar="N", help="覆盖 max_epochs（冒烟用）")
    parser.add_argument("--patience", type=int, metavar="N", help="覆盖早停耐心（0 = 不早停，冒烟用）")
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto", help="设备；auto = 有卡就用卡")
    parser.add_argument(
        "--require-gpu",
        action="store_true",
        help="没有可用 GPU 就报错退出（正式主扫用：B-3 是 GPU 必需项，不许悄悄跑成 CPU）",
    )
    parser.add_argument("--out-root", metavar="DIR", help="把记录/预测写到别处（冒烟用；默认写仓库 results/）")
    parser.add_argument("--no-resume", action="store_true", help="不跳过已有记录，全部重跑")
    parser.add_argument("--repeat-check", type=int, metavar="N", default=0,
                        help="可复现性检查：把选中的第一条 run 连跑 N 次，比较预测最大差（验收④）")
    parser.add_argument(
        "--compare",
        action="store_true",
        help="A4 专用：把冻结臂与微调的记录并成 results/aggregate/a4_finetune_vs_frozen.csv，"
             "不训练、不写记录（验收②的凭据）",
    )
    parser.add_argument("--dry-run", action="store_true", help="只打印网格，不训练、不写文件")
    parser.add_argument("-v", "--verbose", action="store_true", help="更详细的日志 / debug logging")
    return parser.parse_args(argv)


def _describe_grid(keys: list[A.RunKey]) -> str:
    """把网格压成一张短表（任务 × 折数 × 种子数），供 `--dry-run` 打印。
    Compress the grid into a short table (task x folds x seeds) for `--dry-run`.

    中文：输入：RunKey 列表。输出：多行字符串。数据来源：调用方。副作用：无。异常：无。
    English: Args: keys. Returns: a multi-line string. Side effects: none.
    """
    grouped: dict[str, set[str]] = {}
    seeds: dict[str, set[int]] = {}
    for key in keys:
        grouped.setdefault(key.task, set()).add(key.fold)
        seeds.setdefault(key.task, set()).add(key.seed)
    lines = [f"{'任务':<32}{'基准':<14}{'折数':>4}{'种子数':>6}{'run 数':>7}"]
    total = 0
    for task in grouped:
        benchmark = next(key.benchmark for key in keys if key.task == task)
        count = len(grouped[task]) * len(seeds[task])
        total += count
        lines.append(f"{task:<32}{benchmark:<14}{len(grouped[task]):>4}{len(seeds[task]):>6}{count:>7}")
    lines.append(f"{'合计':<32}{'':<14}{'':>4}{'':>6}{total:>7}")
    return "\n".join(lines)


def _summary(records: list[dict]) -> str:
    """把本次写出的记录汇总成一段（条数、指标中位数、总墙钟、GPU 小时）。
    Summarise the records written in this call (count, median metric, wall clock, GPU hours).

    中文：输入：记录列表。输出：字符串。数据来源：调用方。副作用：无。异常：无（空列表给一句话）。
    English: Args: records. Returns: a string; an empty list still returns one sentence.
    """
    if not records:
        return "本次没有新写的记录（全部被 resume 跳过）。"
    r2 = [row["metrics"]["r2"] for row in records if row["metrics"].get("r2") is not None]
    seconds = sum(float(row["wall_clock_s"]) for row in records)
    cuda = sum(1 for row in records if row["device"] == "cuda")
    return (
        f"本次写出 {len(records)} 条记录：r2 中位数 "
        f"{np.median(r2):.4f}（{len(r2)} 条有值）；总墙钟 {seconds / 60:.1f} 分钟；"
        f"device=cuda {cuda} 条 / cpu {len(records) - cuda} 条。"
    )


def _repeat_check(module: Any, config: A.ArmConfig, keys: list[A.RunKey], *, device: str | None,
                  limit: int | None, max_epochs: int | None, patience: int | None,
                  repeats: int, out_root: Path,
                  trainer: Any = None, builder: Any = None, mode: str | None = None) -> int:
    """同一 seed 连跑 N 次，比较测试集预测的最大差（验收④：< 1e-3）。
    Run the same seed N times and compare test predictions (criterion 4: below 1e-3).

    中文：
        用途：分工_B §2 B-3 验收④"同 seed 两次差异 < 1e-3（GPU 非确定性允许，但要开
              cudnn.deterministic 并记录；不许只写'大致一致'）"。这里把"大致一致"变成一个数：
              同一份配置、同一折、同一种子跑 N 次，取预测向量的最大绝对差。
              结果写 `results/aggregate/<arm>[_<模式>]_repeatability.json`（入库，是证据不是日志）。
              A4 的线性头是闭式解，这个数必然是 0——跑它是为了**证明**"必然"，不是为了让数字好看。
              微调臂不一样：它有随机初始化、批次顺序、dropout，这个数**可能**不为 0，所以必须实测。
        输入：module —— 臂模块（提供 `build_task_data` / `train_one`）；config；keys —— 取第一条；
              device/limit/max_epochs/patience —— 冒烟覆盖；repeats —— 次数（≥2）；
              out_root —— 记录目录（检查模式不进正式产物）；
              trainer/builder —— 覆盖训练与取数入口（微调臂传 `train_one_finetune` /
              `build_task_data_finetune`；默认走冻结臂那两个）；
              mode —— 写进 JSON 的模式标签（None = 冻结臂）。
        输出：退出码（0 = 通过）。
        数据来源：`TaskData` / `EmbeddingData`。配置来源：`config`。副作用：训练 N 次、写一份 JSON。
        异常：ArmError —— 次数 < 2 或两次预测长度不一致。
    English:
        Criterion 4 says "two runs of the same seed differ by < 1e-3 ... do not just write
        'roughly the same'". This turns that into one number: same config, same fold, same seed, N
        runs, maximum absolute difference between prediction vectors. The result goes to
        `results/aggregate/<arm>[_<mode>]_repeatability.json` - evidence, not a log. For A4's
        closed-form head the answer is necessarily 0; running it is how the "necessarily" gets
        proven. The finetuned arm is different: random init, batch order and dropout mean the
        number *could* be nonzero, so it has to be measured rather than argued.
        Returns: exit code (0 = pass). Raises: ArmError when repeats < 2 or lengths disagree.
    """
    if repeats < 2:
        raise A.ArmError("--repeat-check 至少 2 次")
    trainer = trainer or module.train_one
    builder = builder or module.build_task_data
    key = keys[0]
    data = builder(key.task, limit=limit)
    predictions = []
    metrics: list[dict[str, Any]] = []
    for attempt in range(repeats):
        started = time.perf_counter()
        outcome = trainer(
            data,
            config,
            fold=key.fold,
            seed=key.seed,
            device=device,
            max_epochs=max_epochs,
            patience=patience,
            # n 档必须跟着传：不传就等于检查了**另一条 run**——训练行没截断、行序也没打乱，
            # 实测会把 op_tm/seed0/n30 的 R² 从 0.2658 报成 0.0922（那是 nall 档的行集合），
            # 而两次之间照样逐位相同，于是"差为 0、通过"看起来完全正常。这正是 B-5 最危险的
            # 一类错误：检查本身在跑，只是查的不是这条。
            # The n-level has to travel with it: without it this checks a **different run** - rows
            # untrimmed and unshuffled - which measured 0.0922 instead of 0.2658 for op_tm/n30 while
            # still coming out bit-identical twice, so "gap 0, passed" looks perfectly healthy.
            n_level=key.n_level,
        )
        predictions.append(outcome.y_pred)
        # 墙钟必须落进凭据：否则"这次检查花了多少卡时"只存在于控制台日志里，事后无法从产物核算。
        # 本文件第一版就是这样——2026-11-12 补做最大任务两次对比花了 0.489 GPU-h，只有文档叙述，
        # 任何脚本都读不出来。重复训练恰好是唯一"花两倍卡时买一个数字"的操作，账更不能含糊。
        # Wall clock has to land in the evidence: otherwise "what did this check cost" lives only in
        # a console log. The first version did exactly that - the 0.489 GPU-h repeat check of
        # 2026-11-12 was unreadable from any artefact.
        metrics.append({"r2": outcome.record["metrics"]["r2"],
                        "rmse": outcome.record["metrics"]["rmse"],
                        "epochs_run": outcome.record["epochs_run"],
                        "wall_clock_s": round(time.perf_counter() - started, 3)})
        LOGGER.info(
            "第 %d/%d 次：r2=%.6f rmse=%.6f %.1f 秒",
            attempt + 1,
            repeats,
            outcome.record["metrics"]["r2"],
            outcome.record["metrics"]["rmse"],
            time.perf_counter() - started,
        )
    base = np.asarray(predictions[0], dtype=np.float64)
    deltas = [
        float(np.max(np.abs(np.asarray(item, dtype=np.float64) - base)))
        for item in predictions[1:]
    ]
    worst = max(deltas) if deltas else 0.0
    # 只报"最大差"会掩盖形状：65 K 的最大差可能来自一个离群点，也可能来自整体平移，两者的
    # 处置完全不同。所以同时记中位差、99 分位、以及越过阈值的点数——三个数一起才说得清。
    # A max alone hides the shape: 65 K could be one outlier or a global shift, and the two call for
    # different handling. Median, p99 and the count over threshold are recorded alongside it.
    diff = np.abs(np.asarray(predictions[-1], dtype=np.float64) - base)
    import torch

    payload = {
        "run_id": key.run_id(),
        # `run_id` 里已经有 `_n30` 这种后缀，这里再单列一份是为了让人不看文件名也能读：
        # 一条 `--repeat-check` 的结论只对它自己那一条 run 成立，n 档写错就会张冠李戴。
        # The run_id already carries the `_n30` suffix; repeating it here is so the level is legible
        # without parsing a file name - a repeat-check result holds for exactly one run.
        "n_level": key.n_level,
        "n_train": int(outcome.record.get("n_train") or 0),
        "repeats": repeats,
        # 这条检查的总卡时 = 各次墙钟之和。它不进 `gpu_hours.csv`（那张表一次 run 一行，
        # 而这里同一条 run 被跑了 N 次），所以必须自带这一项，否则总账会少算。
        # Total GPU time for this check. It is deliberately absent from `gpu_hours.csv` (one row per
        # run, but here one run is trained N times), so the payload has to carry it.
        "wall_clock_total_s": round(sum(item["wall_clock_s"] for item in metrics), 3),
        "wall_clock_s": [item["wall_clock_s"] for item in metrics],
        "max_abs_prediction_delta": worst,
        "median_abs_prediction_delta": float(np.median(diff)),
        "p99_abs_prediction_delta": float(np.percentile(diff, 99)),
        "n_over_threshold": int((diff >= 1e-3).sum()),
        "n_test": int(diff.size),
        "deltas": deltas,
        "per_repeat_metrics": metrics,
        "threshold": 1e-3,
        "passed": worst < 1e-3,
        "cudnn_deterministic": bool(torch.backends.cudnn.deterministic),
        "deterministic_algorithms": bool(torch.are_deterministic_algorithms_enabled()),
        "device": A.resolve_device(device or "cuda"),
        "checked_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "note": (
            "同配置、同折、同种子重复训练；预测是原单位，差值就是验收④要的那个数。"
            "两个确定性开关都必须为 true：只钉 cudnn 时本臂实测中位差 0.56 K、超过阈值的点占 99.96%，"
            "因为 chemprop 的图聚合是 CUDA 上的 scatter_reduce 原子加；两条都钉则逐位相同"
            if key.arm != "a4_polybert"
            else (
                "同配置、同折、同种子重复训练；预测是原单位，差值就是验收④要的那个数。"
                + (
                    "微调臂有随机初始化、批次顺序与 dropout，差**可能**不为 0，所以必须实测。"
                    "实测为 0 说明两个确定性开关（cudnn.deterministic + use_deterministic_algorithms）"
                    "在这条路径上也是够的：两次的轮数、best_epoch、R² 与每个预测值都逐位相同。"
                    if mode == "finetune"
                    else "A4 的线性头是闭式解（(X'X+aI)w = X'y），没有随机初始化、没有批次顺序、没有原子加，"
                    "所以差**必然**是 0——跑这一遍是为了把这个「必然」变成一条可核对的记录，不是为了让数字好看。"
                    "两个确定性开关仍置 true：记录里 cuda 的 run 一旦为 false 会被 record_problems() 判不合格，"
                    "而这两条记录与 A3 用同一套校验。"
                )
            )
        ),
    }
    if mode:
        payload["mode"] = mode
    # 文件名按臂区分：A3 与 A4 的可复现性证据是两份文件，不能互相覆盖。
    # 有 `experiment` 的配置（B-5）再用实验名打头：它的网格与 A3 的 48 条不同一条 run，
    # 若沿用 `a3_repeatability.json` 就会把 B-3 的凭据覆盖掉——而覆盖之后文件名照样合法，
    # 只有内容悄悄变成另一个网格的。
    # File name per arm: A3 and A4 keep separate evidence files, never overwriting each other.
    # A config with an `experiment` (B-5) prefixes the experiment name too: its grid shares no run
    # with A3's 48, and reusing `a3_repeatability.json` would overwrite B-3's evidence - the file
    # would still look legal, only its content would quietly belong to another grid.
    experiment = str(config.raw.get("experiment") or "")
    if experiment:
        stem = experiment
    else:
        arm_name = str(config.raw.get("arm", "a3_graph"))
        stem = "a3" if arm_name == "a3_graph" else "a4"
        if mode:
            stem = f"{stem}_{mode}"
    target = out_root / f"{stem}_repeatability.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    # `newline="\n"` 不能省：`Path.write_text` 在 Windows 下会把 `\n` 翻成 `\r\n`，
    # 而 `.gitattributes` 钉的是 `* text=auto eol=lf`——写出来是 CRLF，入库变 LF，
    # 于是每次 `git add` 都警告、工作区与 blob 长期不一致。
    # Do not drop `newline="\n"`: `Path.write_text` turns `\n` into `\r\n` on Windows while
    # `.gitattributes` pins `eol=lf`, so the working copy and the blob would never agree.
    target.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n"
    )
    LOGGER.info("可复现性：最大预测差 %.3e（阈值 1e-3）→ %s", worst, "通过" if worst < 1e-3 else "未通过")
    return 0 if worst < 1e-3 else 2


def main(argv: list[str] | None = None) -> int:
    """入口：解析参数 → 展开网格 → 训练（或只打印）→ 总结。
    Entry: parse arguments, expand the grid, train (or just print), summarise.

    中文：输入：argv。输出：退出码（0 成功 / 2 可选检查未通过 / 1 致命错误）。
          数据来源：配置与装配产物。副作用：训练、写记录/预测/墙钟。异常：ArmError 转为退出码 1。
    English: Args: argv. Returns: 0 ok, 2 a check failed, 1 fatal.
    """
    args = _parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )
    module = _load_arm(args.arm)
    if args.finetune and args.arm != "a4_polybert":
        LOGGER.error("--finetune 只对 --arm a4_polybert 有意义：A3 是端到端训练的，没有冻结/微调之分")
        return 1
    extra_kwargs: dict[str, Any] = {}
    if args.extra_folds:
        if args.arm != "a3_graph":
            LOGGER.error(
                "--extra-folds 只对 --arm a3_graph 有意义：它是给 B-5 的 OpenPoly 敏感性分析用的"
                "（跑配置网格之外的划分）。A4 的任务是 PolyBench/PB26，没有第二套折文件。"
            )
            return 1
        # 只给 `--extra-folds` 时，意图显然是"就跑这几折"；否则会和配置里的 5 折并集，变成 450 条。
        # Naming only `--extra-folds` means "run exactly these"; otherwise it unions with the config's
        # five folds and the grid silently doubles to 450 runs.
        if not args.folds:
            args.folds = list(args.extra_folds)
        extra_kwargs["extra_folds"] = args.extra_folds
    config_path = args.config or str(A.REPO_ROOT / ARM_CONFIGS[args.arm])
    try:
        config = module.load_arm_config(config_path)
        keys = module.enumerate_runs(
            config,
            tasks=args.tasks,
            folds=args.folds,
            seeds=args.seeds,
            n_levels=args.n_levels,
            **extra_kwargs,
        )
    except A.ArmError as error:
        LOGGER.error("%s", error)
        return 1

    device = None if args.device == "auto" else args.device
    if args.require_gpu:
        resolved = A.resolve_device("cuda")
        if resolved != "cuda":
            LOGGER.error(
                "本机没有可用的 CUDA 设备，而 --require-gpu 要求必须用卡。"
                "B-3 是 GPU 必需项：宁可停下，也不要悄悄跑成 CPU（分工_B §7 常见坑）。"
            )
            return 1
        device = "cuda"

    LOGGER.info("配置 %s（%s）", config.path.name, config.hash)
    LOGGER.info("网格 %d 条：\n%s", len(keys), _describe_grid(keys))
    if args.limit is not None or args.max_epochs is not None:
        LOGGER.warning(
            "冒烟模式（--limit %s / --max-epochs %s）：产物不得当作正式结果",
            args.limit,
            args.max_epochs,
        )
    if args.dry_run:
        for key in keys:
            print(key.run_id())
        return 0

    if args.compare:
        if args.arm != "a4_polybert":
            LOGGER.error("--compare 只对 --arm a4_polybert 有意义：A3 只有一种训练模式")
            return 1
        try:
            report = module.write_comparison()
        except A.ArmError as error:
            LOGGER.error("%s", error)
            return 1
        print(
            f"对照表 {report['path']}：{report['n_matched']} 条配对；"
            f"只有微调 {report['n_missing_frozen']} 条 / 只有冻结 {report['n_missing_finetune']} 条"
        )
        for task, bucket in report["by_task"].items():
            print(
                f"  {task:<30} 冻结 {bucket['r2_frozen']:.4f}  微调 {bucket['r2_finetune']:.4f}  "
                f"差 {bucket['delta_r2']:+.4f}  ({int(bucket['n'])} 条)"
            )
        return 0

    out_root = Path(args.out_root) if args.out_root else A.REPO_ROOT / "results" / "aggregate"
    if args.repeat_check:
        try:
            return _repeat_check(
                module,
                config,
                keys,
                device=device,
                limit=args.limit,
                max_epochs=args.max_epochs,
                patience=args.patience,
                repeats=args.repeat_check,
                out_root=out_root,
                trainer=(module.train_one_finetune if args.finetune else module.train_one),
                builder=(module.build_task_data_finetune if args.finetune else module.build_task_data),
                mode=("finetune" if args.finetune else None),
            )
        except A.ArmError as error:
            LOGGER.error("%s", error)
            return 1

    if args.finetune:
        # 微调是**对照**，不能写 `results/raw_runs/`：那里是冻结臂的 48 条，同名 run 会被覆盖，
        # 于是主表与对照混成一份，谁也说不清哪条是哪条（D-34）。
        # The fine-tune is a contrast and must not write into `results/raw_runs/`: those are the
        # frozen arm's 48 records, and a same-named run would overwrite one of them (D-34).
        record_root = Path(args.out_root) if args.out_root else (
            A.REPO_ROOT / "results" / "aggregate" / "a4_finetune"
        )
        LOGGER.info("微调对照模式：产物写到 %s（不入 raw_runs）", A._repo_relative(record_root))
    else:
        record_root = Path(args.out_root) if args.out_root else None
    try:
        records = module.run_grid(
            config,
            tasks=args.tasks,
            folds=args.folds,
            seeds=args.seeds,
            n_levels=args.n_levels,
            limit=args.limit,
            device=device,
            max_epochs=args.max_epochs,
            patience=args.patience,
            resume=not args.no_resume,
            out_root=record_root,
            **extra_kwargs,
            **({"finetune": True} if args.finetune else {}),
        )
    except A.ArmError as error:
        LOGGER.error("%s", error)
        return 1
    # 有 n 档的配置（B-5）跑完顺手刷一份索引：D 的 κ 与 H4 过零点要用它读 n，不必自己去翻 225
    # 个 JSON。刷的是**全量记录的现状**，不是"这次写了哪些"——所以 `resume` 跳过已有记录时，
    # 索引依然完整。
    # A config with n-levels (B-5) also refreshes an index: D reads n from it instead of parsing 225
    # JSON files. It is built from every record on disk, not just the ones written this time, so a
    # resumed run still leaves a complete index.
    if config.raw.get("n_levels"):
        try:
            info = module.write_n_index(config) if hasattr(module, "write_n_index") else None
        except A.ArmError as error:
            LOGGER.error("%s", error)
            return 1
        if info:
            print(
                f"索引 {A._repo_relative(info['path'])}：{info['n_runs']} 条"
                f"（{info['n_tasks']} 性质 × {info['n_levels']} 档 × {info['n_folds']} 折"
                f" × {info['n_seeds']} 种子）"
            )
    print(_summary(records))
    return 0


if __name__ == "__main__":
    sys.exit(main())
