# 运行请求单（Run Requests）

> **用途**：没有 GPU 的组员把需要代跑的实验写成请求，由有卡的人（仓库所有者或分工 B）代跑，结果回传到 `results/`。
> **铁律**：这个文件里**只有请求和回传，没有结果**。结果只能由运行产生——不许预填、不许估算、不许"我猜大概是 0.8x"。

---

## 一条请求的合格标准（三条，缺一条不算交接）

1. **代码能在一小片上跑通**：入口必须有 `--limit` / `--dry-run`，CPU 上 2 分钟内跑完一片并打印形状与指标。没通过 smoke 的代码不要提交请求，那只会让代跑的人替你调错。
2. **不写死 GPU**：`device = "cuda" if torch.cuda.is_available() else "cpu"`，CPU 下能跑完（慢无所谓）。
3. **请求单里写清五件事**：命令、预计耗时、依赖的 `config_hash`、期望产出路径、提交时间。

---

## 请求

字段固定（与 `docs/分工_B_表示与GPU臂.md` §4 一致）。`run_id 规则` 用 `task__arm__seed{...}` 的形式，方便代跑人对着生成。

| run_id 规则 | 命令 | 预计 GPU-h | 依赖的 config_hash | 期望产出路径 | 提交时间 |
|---|---|---|---|---|---|
| （示例，勿当真实请求）`pb_t1_tg__a3_graph__seed{0,1,2}` | `python -m chainworth.train --config configs/arms/a3_graph.yaml --task pb_t1_tg --seeds 0 1 2` | 0.2 | `sha256:1a9d…` | `results/raw_runs/` | 11-13 20:00 |

> 示例行保留在表里，新增请求往下追加；**不要删示例行的"（示例，勿当真实请求）"标记**。
>
> **本机有卡，B-1 / B-2 / B-3 / B-4 / B-5 都是自跑，不经这张表。** 这张表为空**不代表没跑过**——B-3 的 48 条、B-2 的 18,152 个结构嵌入、B-4 的 48 条冻结臂 + 48 条微调、B-5 的 225 条 OpenPoly 扫描，都在本机 `NVIDIA GeForce RTX 4070`（12 GB）上跑完，凭据在各自的交付文档（`docs/交付_P2_B3_A3图神经网络主扫.md` §5、`docs/交付_P2_B4_A4polyBERT迁移臂.md` §5、`docs/交付_P2_B5_OpenPoly规模扫描.md` §5）与 `results/` 里。表里没有请求行，是因为没有"没卡的人"要请人代跑；等 C 的 04 批量需要 GPU 时才往这里加。
>
> **B-5 的实测值得单记一笔**：225 条只用 **0.1712 GPU-h**，而 Proposal §7.4 预测 4–11 GPU-h——**低 23–64 倍**。原因不是省了工作（单条 0.8–8.2 秒是真的跑了 225 次），而是预测偏保守：训练行最多 355、183/225 早停在中位 46 轮、图按结构去重后只有 304 个。**这笔差额应当从 P2 的 6 pH 里退回去**，别让它挂在账上——否则后面 C/D 会按虚高的 GPU 预算排期。
>
> **B-4 有一处与本表约定的落盘位置不同，先说清楚**：微调 48 条的 `run_id` 与冻结 48 条**完全相同**（同任务/同折/同种子），写进 `results/raw_runs/` 就是静默覆盖前一份。所以微调产物落 `results/aggregate/a4_finetune/`（内含 `raw_runs/` 与 `predictions/` 两个同名子目录结构），由 `--compare` 并成 `results/aggregate/a4_finetune_vs_frozen.csv`。查微调记录时要到那个目录找，别在 `results/raw_runs/` 里翻。

---

## 回传（由代跑人填，请求人不要改）

| 对应请求行 | 实际 run_id | device | 实际耗时 | 落盘路径 |
|---|---|---|---|---|
| — | — | — | — | — |

回传时要同时满足：

- `results/raw_runs/<run_id>.json` 与 `results/aggregate/predictions/<run_id>.parquet` **两份都在**，缺预测下游算不出区间；
- 失败的 run 也要写一行，`落盘路径` 填 `failed: <原因>`，不许静默跳过；
- 与请求里 `run_id 规则` 对不上时，在 `实际 run_id` 里说明为什么改了规则。
