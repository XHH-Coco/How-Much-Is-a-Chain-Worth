# 分工 B · 整体验收记录（B-1 → B-5，2026-11-13）

> **这份文件是"交接前自检"的凭据**，不是总结陈词。它记三件事：**怎么验的**（可重跑的命令）、
> **结果是什么**（原样输出）、**发现了什么并改了什么**（含改前改后的数字）。
> 读取顺序建议：§1 → §5（问题）→ §4（逐项）。
> 与本文件冲突时，以 `README.md` 第九节冻结登记与 `docs/DECISIONS.md` 为准。

- 验收人：仓库所有者（代分工 B 执行整体验收）
- 验收时刻：2026-11-13 03:05–03:40（+08:00）
- 验收对象：`docs/分工_B_表示与GPU臂.md` §2 的 B-1 / B-1.5 / B-2 / B-3 / B-4 / B-5、§3 交付物清单、§8 交接前自检
- 起点提交：`3fd28f6`（B-4/B-5 三件补做，跟踪文件 1,418）

---

## 1. 一句话结论

**六项任务的验收标准全部通过，546 条 run 记录逐条过 `record_problems()` 零不合格，
546 份测试集预测逐份对上行数，三份配置的字节哈希与冻结登记表逐字相同。**

发现 **6 处需要改的**，已全部当场改掉：其中 **1 处是会误导决策的记账错误**（B-4 的"3.38 GPU-h /
超预算 11%"把 A3 的 0.696 算进了 B-4 的账；实际 **3.17 = 91%，仍在预算内**），
其余 5 处是文档过期或清单笔误（README 产物登记停在 B-3、分工书三处"273 条"、
`test_splits_identity.py` 这个文件从来不存在、B-2 的清单写"1 行"实为 2 行、`trials ≤30` 对 B 不适用）。

**不改结论、但交接时必须说清的是四条**：`--repeat-check` 的两份旧凭据没有墙钟字段（脚本已补，存量未动）；
`polybert_manifest.csv` 无版本列、靠"最后一行"约定；B-4 的 `max_epochs=30` 让微调数字是**下界**；
GPU 总账分散在三处，相加才是 4.21 GPU-h。

---

## 2. 验收方法

**不看文档怎么写，跑命令、数文件、比哈希。** 每条验收标准都要能落到一条命令或一次逐行比对；
落不到的写进 §6。

| 层面 | 怎么验 | 次数 |
|:--|:--|--:|
| 记录完整性 | 遍历 `results/raw_runs/*.json`，查 15 个必需字段非空、`device`、`config_hash`、`split_version` | 546 条 |
| 记录自检 | 对每条调 `record_problems()`（A3/B-5 用 `graph`，A4 用 `polybert`） | 546 条 |
| 预测对齐 | 逐份读 parquet：行数 = `n_test`、`row_id` 无重复、列名 = `row_id,y_pred` | 546 份 |
| 装配 | `03b_assemble.py --verify`（含抽查 100 个结构与 02 产物逐位比对） | 15 任务 |
| 嵌入 | `embeddings.npy` 形状 vs `index.parquet` 行数 vs 结构表行数；`n_unk_tokens` | 18,152 |
| 配置冻结 | 三份配置的字节 sha256 vs 记录里的 `config_hash` vs README 第九节登记 | 3 份 |
| 网格自洽 | 饱和档 R² 与 `all` 档逐位相同；`n_train` 上界 ≤ 443；两批 `config_hash` 相同 | 60 + 450 |
| 索引 | `openpoly_gpu_index.csv` 与记录逐行比 `subset_sha256`；`n_index` 三元组唯一性 | 450 + 150 |
| 工时 | `gpu_hours.csv` 的 `run_id` 集合 vs 磁盘集合（双向差集）；微调表单独求和 | 546 + 48 |
| 测试 | `pytest tests`（core 环境） | 280 项 |
| 纪律 | `git status`、`git ls-files` 查权重/日志/缓存、`.gitignore` 覆盖 | 全仓 |

---

## 3. 结果一览

| 项 | 验收条目数 | 通过 | 关键数字 |
|:--|--:|--:|:--|
| B-1 表示装配 | 5 | **5** | 15 任务 / 30,733 行 / 2,253 列（205 描述符 + 2,048 位指纹）；图 **30,509 个、解析失败 0**；`--verify` 五条全过 |
| B-1.5 补行表扩表征 | 4 | **4** | 15 张调和表；443 / 295 / 168 与 `n_labels`、A-7 复核三处一致；PB26 7,367 / 1,800 与官方 `n_rows` 相等 |
| B-2 polyBERT 嵌入 | 3 | **3** | **18,152 × 600** float32；`n_unk_tokens = 0`；权重 sha256 `21cfab4a…`；GPU 25.36 秒 |
| B-3 A3 主扫 | 5 | **5** | **48 条**、0.6961 GPU-h、中位 R² **0.8899**、早停 36/48、同 seed 两次差 **0.0** |
| B-4 A4 迁移臂 | 3 | **3** | 冻结 **48 条**（6.0 秒、0.0017 GPU-h、中位 R² 0.8408）+ 微调 **48 条**（2.6835 GPU-h、中位 0.8956）；自身 **3.17 GPU-h = 91%** |
| B-5 OpenPoly 扫描 | 4 | **4** | **450 条**（225 主表 + 225 敏感性）、0.1712 + 0.1645 = **0.3357 GPU-h**；饱和 60 条与 `all` 逐位相同；`n_train` 上界 **355 ≤ 443** |

**run 记录 546 条** = 48（A3）+ 48（A4 冻结）+ 225（B-5 主表）+ 225（B-5 敏感性）；
微调另有 48 条，按 D-34 落 `results/aggregate/a4_finetune/`，**刻意不进 `raw_runs/`**。

**测试基线**：`pytest tests` = **280 passed / 7 skipped**（53 秒）。7 个 skip 是要 GPU / 大模型 /
`CHAINWORTH_SLOW=1` 的那几项；本轮新增的 3 条测试分层用例**在 core 环境真跑**，不是 skip。

**B 组 GPU 总账**：0.6961 + 0.0017 + 2.6835 + 0.489 + 0.1712 + 0.1645 = **≈ 4.21 GPU-h**，
占 Proposal §7.4 给 P2 的 6 GPU-h 的 **70%**。另有两轮 `pb_t3_iwlt` 复现性（约 0.02）未计入——
它们和最大任务那 0.489 一样不进任何工时表。账分三处，见 §6-4。

---

## 4. 逐项凭据

### B-1 表示装配（5 条全过）

| 验收标准 | 实测 | 结论 |
|:--|:--|:--|
| 行数 = 可用标签数，行序由 split 决定 | `task_features_manifest.csv` 15 行、合计 30,733 行；`row_order_basis` 字段声明来源 | 通过 |
| 抽 100 个结构与描述符矩阵逐位相等 | `03b_assemble.py --verify` 原话："验收通过：五条标准全过（抽查 100 个结构与 02 的产物逐位相同）" | 通过 |
| 同结构两次构图 `edge_index` 相同 | `graph_manifest.csv` 记 `n_graphs = 30,509`、`n_parse_failed = 0`；构图走固定原子序 | 通过 |
| 不含全量统计量 | 标准化只在 train 上 fit（`tests/test_task_assembly.py` 守） | 通过 |
| 解析失败计数并列出 ID | 失败 0；清单有 `n_rows_without_structure` 列 | 通过 |

`npz` 内容实测（`pb_t1_tg`）：`row_id / y / psmiles / desc(8055,205) / fp(8055,2048)`，
列数 205 + 2,048 = **2,253**，与 B-1 交付时的口径一致。

### B-1.5 补五张表（4 条全过）

443 / 295 / 168 三个分母与 `configs/tasks/op_*.yaml` 的 `n_labels`、A-7 的
`openpoly_property_counts.csv` **三处独立一致**；PB26 两个任务 7,367 / 1,800 与官方
`manifest.json` 的 `n_rows` 相等。结构表 18,152、特征 15 份 / 30,733 行、polyBERT 18,152 × 600，
与交付文档 §1 逐项对得上。A-3 四条、A-6 深层、B-1 五条、B-2 五检全过（本轮重跑了 B-1 那条）。

### B-2 polyBERT 嵌入（3 条全过）

| 验收标准 | 实测 |
|:--|:--|
| 行数 = 结构数、顺序可校验 | `embeddings.npy` **(18,152, 600)**、`index.parquet` 18,152 行、结构表 18,152 行，三者相等；`order_sha256` 在清单里 |
| ndim 与模型一致并记录 | `embed_dim = 600`；`model_repo = xushijie/polyBERT`、`revision = e7dce434…`、权重 sha256 `21cfab4a…` |
| 断网可续跑 | 12 个分片落盘，`is_partial` 字段区分 |

清单里 `n_unk_tokens = 0`、`n_truncated = 0`——D-31 那条「裸 `*` 会静默变 `<unk>`」的坑没有复发。

### B-3 A3 主扫（5 条全过）

48 条一条不缺、全部 `device=cuda`、`config_hash` 单一（`f9c2264e…`，与 README 第九节第 8 行逐字相同）、
16 个 `split_version`、预测行数逐条对上、48 条逐条过 `record_problems()`。合计 **2,506.0 秒 = 0.6961 GPU-h**，
中位 R² **0.8899**，早停 36/48。

第 ④ 条（同 seed 两次差 < 1e-3）实测 **0.0**，2,417 个点全等（`a3_repeatability.json`）——
比分工书要求的更强，代价是墙钟 ×1.92（D-33 ⑦）。

### B-4 A4 迁移臂（3 条全过）

- (a) 冻结 48 条：6.0 秒 = 0.0017 GPU-h，中位 R² **0.8408**，`mode=frozen_ridge`。
- (b) 微调 48 条：9,660.6 秒 = **2.6835 GPU-h**，中位 R² **0.8956**，与 (a) 同任务/同折/同种子/同验证留出。
- 验收②：`a4_finetune_vs_frozen.csv` 48 条配对、0 缺边。
- 验收③（若只跑 (a) 必须写明 triage (ii)）：两条都跑了，**不触发**降级。
- 复现性：(a) 差 0.0（闭式解必然）、(b) 差 0.0（`pb_t3_iwlt` 同进程 + 跨进程、`pb26_polymetrix_tg` 最大任务两次）。

### B-5 OpenPoly 扫描（4 条全过）

| 验收标准 | 实测 |
|:--|:--|
| ① 225 条齐全 | 主表 225（`split_fold ∈ seed0–4`）、敏感性 225（`seed5–9`），**0 条 `blocked_no_gpu`** |
| ② 一律 n ≤ 443 | `n_train` 上界 **355**；`op_tg` 池子 355、`op_dielectric` 235、`op_tm` 134，**未外推** |
| ③ 每 run 落预测 | 450 份，与记录一一对应，行数逐条等于 `n_test` |
| ④ 墙钟写进索引、超 1.5× 报 | `openpoly_gpu_index.csv` 450 行含 `wall_clock_s`；单条 0.4–8.2 秒，远低于 1.5× 的 4.5 分钟，**未触发 triage** |

**饱和档自检**：`n_level_saturated=true` 共 **60 条**（2 性质 × 10 折 × 3 种子），
与同 (task, fold, seed) 的 `all` 档 R² **逐位相同，0 条不等**。这条相等本身就是
"子集构造只动了行集合"的证明。

**索引一致性**：`openpoly_gpu_index.csv` 450 行的 `subset_sha256` 与记录里的 `n_subset_sha256`
**逐行相同，0 条不一致**；`openpoly_n_index.csv` 150 行、150 个唯一三元组（3 性质 × 10 折 × 5 档）。
**这是 C-3 唯一的对账依据**，见 §8。

**两批同哈希**：450 条的 `config_hash` 全为 `9328cfdf…`，配置一个字节没动（D-36 要的效果）。

---

## 5. 发现并修掉的问题

### 5.1 【记账错误，会误导决策】B-4 的 3.38 GPU-h 里混进了 A3 的 0.696

**现象**：B-4 交付文档写"查总花销要两处相加 = 3.38 GPU-h"，§9.1 又写"实付 3.38 → 3.87，
**超 3.5 预算 11%**"。

**原因**：那"两处"之一是 `results/aggregate/gpu_hours.csv` 的**整表求和**（96 行 = 0.698），
而这张表是 B 组共用的——96 行 = A3 的 48 条（**0.6961**）+ A4 冻结的 48 条（**0.0017**）。
把整表当作 B-4 支出，等于让同一次 A3 运行在 B-3 与 B-4 两处各记一遍。

**改前 / 改后**：

| | 改前 | 改后 |
|:--|--:|--:|
| B-4 自身花销 | 3.38 GPU-h | **3.17 GPU-h**（0.0017 + 2.6835 + 0.489） |
| 占 3.5 预算 | "97%" | **91%** |
| 补做后 | 3.87，"超 11%" | **3.17，"仍在预算内，余 0.33"** |

**影响**：不改变任何实验数字，但改变"下一步还能不能花钱"的判断。已改 §1 产物表注、§9-430 行、
§9.1 末尾，并在 §9.2 新增一张四行对照表说明每笔归谁。**B-3 的 0.6961 不动**。

### 5.2 【文档过期】README 的 `results/aggregate/` 只登记到 B-3

该行原话只写"**B-3 已填**：48 份预测、`gpu_hours.csv`、`a3_repeatability.json`"，
B-4 的 `a4_finetune/`、`a4_finetune_vs_frozen.csv`、两份 repeatability，以及 B-5 的 450 份预测、
两份索引全都没登记。已补全（README 第四节 `results/aggregate/` 行）。

### 5.3 【文档过期】分工书三处"273 条"

`§3 交付物清单`、`§4 交接后验收`、`§8 交接前自检` 都写 273 = 48 + 225。
这个数只算了 A3 与 B-5 主表，**漏了 A4 的 96 条（48 冻结 + 48 微调）与 B-5 敏感性的 225 条**。
按它验收会对不上账。三处均改为 **546**（并注明微调的 48 条在 `a4_finetune/`）。

### 5.4 【清单笔误】`test_splits_identity.py` 这个文件从来不存在

§8 要求"`pytest tests/` 全绿，含 `test_splits_identity.py`"。`tests/` 里没有这个文件，
但等价断言在 `tests/test_splits.py` 里：`test_polybench26_copies_are_byte_identical`、
`test_openpoly_splits_are_reproducible`（"同一种子在发布表上重算，行号必须逐位相同"）、
`test_canonical_tables_match_official_digest`、`test_tampered_split_is_detected`。
已把清单改成真实文件名。

### 5.5 【文档过期】B-2 的 `polybert_manifest.csv` 写"1 行"，实为 2 行

B-1.5 扩表征后重跑了一次嵌入，`append_manifest_csv()` 追加了第二行：
v1 = 11,494 结构（08:08）、v2 = 18,152 结构（08:39）。清单**无版本列**，只能靠 `created_at` 分辨。
已注明"取最后一行"。好消息是**没有任何脚本读这个 csv**（只有写），所以是"人会读错"而不是"程序会读错"。

### 5.6 【自检项不适用】`configs/arms/*.yaml` 的"预算表 ≤30 trials"

B 的两条臂都没有 trials 表：A3 按红线 2 不许调参，A4 的 ridge 头是闭式解（`alpha` 走固定网格，
一次解完 9 个候选，不做搜索）。已在 §8 该条注明不适用，并指出等价的自检是
"`openpoly_scan.yaml` 与 `a3_graph.yaml` 逐训练键相同"（由 `tests/test_openpoly_scan.py` 守着）。

### 5.7 【代码补强】`--repeat-check` 不记墙钟

`_repeat_check()` 里算了 `perf_counter()`，但只写进日志、没进 payload。后果是
2026-11-12 补做最大任务两次对比花的 **0.489 GPU-h 只存在于文档叙述里**，任何脚本都读不出来。
已给 payload 加 `wall_clock_total_s` 与逐次 `wall_clock_s`（`scripts/05_run_gpu.py`）。
**存量两份凭据不改**（记录只能由脚本落盘，手改等于伪造）；下次跑就自带这一项。

---

## 6. 没改、但交接时必须知道的四条

1. **`polybert_manifest.csv` 无版本列**——加列要迁移旧行，收益不大；用"`created_at` 最新一行"这条约定代替。
2. **B-4 的 `max_epochs=30` 让微调是下界**——两条跑满上限（第 29、25 轮 best 仍在降）。
   这是预注册的预算旋钮（Proposal §7.1 降级梯第 (i) 条），不是待调超参；偏差方向单一，只会让微调显得更差。
3. **`gpu_hours.csv` 里 A3 与 A4 冻结混在一张表**（96 行 = 0.698）。这张表按 `run_id` 一行一条，
   **不能整表求和当作某一项的花销**——5.1 那个错误就是这么来的。按 `arm` 列分组才是正确读法。
4. **GPU 总账分三处**：`gpu_hours.csv`（1.0335，546 条）、`a4_finetune/gpu_hours.csv`（2.6835，48 条）、
   两份 `*repeatability*.json`（最大任务 0.489；`pb_t3_iwlt` 那两轮不足 0.02）。
   相加 **≈ 4.21 GPU-h**——重复训练不在这两张表里，所以只加表会少算 0.489。

---

## 7. 复核方式（一条命令一条结论）

```bash
# 1) 记录数：546 条（另 1 份 _schema.json）
ls results/raw_runs/*.json | wc -l

# 2) 预测数：546 份，与记录一一对应
ls results/aggregate/predictions/*.parquet | wc -l

# 3) 逐条自检 + 预测行数（应输出「0 不合格 / 0 缺行 / 0 重复」）
PYTHONPATH=src python - <<'PY'
import json,glob,os,collections,pandas as pd
from chainworth.arms import graph as G, polybert as P
recs=[json.load(open(p,encoding="utf-8")) for p in glob.glob("results/raw_runs/*.json") if not p.endswith("_schema.json")]
bad=[r["run_id"] for r in recs if (P.record_problems if r["arm"]=="a4_polybert" else G.record_problems)(r)]
rows=[(r["run_id"],len(pd.read_parquet(r["predictions"])),r["n_test"]) for r in recs]
print(len(recs),"条 | 自检不合格",len(bad),"| 行数不符",sum(1 for _,a,b in rows if a!=b))
PY

# 4) 三份配置哈希（应分别等于 f9c2264e… / ca4261c3… / 9328cfdf…）
sha256sum configs/arms/a3_graph.yaml configs/arms/a4_polybert.yaml configs/experiments/openpoly_scan.yaml

# 5) 饱和档：60 条，与 all 档 R² 逐位相同
PYTHONPATH=src python -c "
import json,glob
d={(r['task'],r['split_fold'],r['seed'],str(r['n_level'])):r for r in (json.load(open(p,encoding='utf-8')) for p in glob.glob('results/raw_runs/op_*__a3_graph__*.json'))}
sat=[k for k,r in d.items() if r.get('n_level_saturated')]
print('饱和',len(sat),'| 不等',sum(1 for k in sat if d[(k[0],k[1],k[2],'all')]['metrics']['r2']!=d[k]['metrics']['r2']))"

# 6) 索引：450 行 / 150 行
wc -l results/aggregate/openpoly_gpu_index.csv results/aggregate/openpoly_n_index.csv

# 7) 工时：546 行 = 1.0335 GPU-h；微调 48 行 = 2.6835
PYTHONPATH=src python -c "
import csv
for p in ['results/aggregate/gpu_hours.csv','results/aggregate/a4_finetune/gpu_hours.csv']:
    r=list(csv.DictReader(open(p,encoding='utf-8')))
    print(p, len(r), round(sum(float(x['gpu_hours']) for x in r),4))"

# 8) 测试（Windows 下临时目录必须指定到仓库内，否则沙箱会拦批量删除）
PYTHONPATH= python -m pytest tests --basetemp=.pytest_tmp_verify2 -p no:cacheprovider -q
```

---

## 8. 交接状态

**交给 C（主表 / 经典臂）**

- A3 的 48 条 + A4 冻结的 48 条，`run_id` 里带 `__a3_graph__` / `__a4_polybert__`；
  A4 只取 `mode=frozen_ridge` 的 48 条，**不要取 `a4_finetune/` 里的**。
- B-5 的 450 条按 `fold` 分两批：`seed0–4` 是主表、`seed5–9` 是敏感性，**不许合并成"每格 30 条"**（D-36 ④）。
- **C-3 的 225 fits 必须用 `openpoly_n_index.csv` 的 `subset_sha256` 核对**，按 `(task, fold, level)`
  三元组取；若也跑 5–9，**不能拿 0–4 的哈希去核**——不同折的 `train_index` 不同，子集哈希必然不同。

**交给 D（κ / H4 / 图）**

- `predictions/<run_id>.parquet` 两列 `row_id, y_pred`，行序与 `splits/` 的 `test_index` 同一坐标系；
  546 份齐全（含 B-5 的 450 份）。
- **κ 的横轴必须用 `n_train` 而不是档位名义值**（240 档在 `op_dielectric` 只有 235 行、`op_tm` 只有 134 行）。
- 低 n 端要拿 `n_bins_covered` 判别"覆盖不足"；`op_tm` 在 n=30 时 R² 为负是**数据事实**，不是坏点。
- **形状不可用、水平可用**：`op_tm` 的"先升后降"已被 5–9 折推翻（10 折合并降幅 0.008 < 折间 sd 0.12）；
  三个性质**水平**的排序（介电 > Tg > Tm）两批都成立。

**交给仓库所有者（汇报）**

- 可复现性：A3 与 A4 都是**同 seed 两次逐位相同（0.0）**，不是"小于 1e-3"。
- 权重来源：`kuelumbus/polyBERT` 已 404，用的是三方互校副本（D-31），论文 Data availability 必须写。
- B 组 GPU 总账 **≈ 4.21 GPU-h / 6 GPU-h 预算 = 70%**；其中 B-5 的 450 条只占 0.3357，
  比 Proposal 预估的 4–11 GPU-h 低 12–33 倍，**这笔差额应从 P2 的 6 GPU-h 里退回**。

**未完成、也不属于 B 的**：C-3 的经典臂 225 fits、D 的 κ 与 H4 判据、H4 的消融实验
（GNN vs XGBoost 的差距只做了归因推理，未做分解——见 B-5 文档 §6 与 §9-7）。
