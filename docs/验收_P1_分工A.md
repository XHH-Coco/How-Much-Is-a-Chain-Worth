# 分工 A · 整体验收记录（A-1 → A-8，2026-11-12）

> **这份文件是"交接前自检"的凭据**，不是总结陈词。它记三件事：**怎么验的**（可重跑的命令）、**结果是什么**（原样输出）、
> **发现了什么并改了什么**（含改前改后的数字）。读取顺序建议：§1 → §5（问题）→ §4（逐项）。
> 与本文件冲突时，以 `README.md` 第九节冻结登记与 `docs/DECISIONS.md` 为准。

- 验收人：仓库所有者（代分工 A 执行整体验收）
- 验收时刻：2026-11-12 07:00–07:30（+08:00）
- 验收对象：`docs/分工_A_数据与合规.md` §2 的 A-1 至 A-8、§3 交付物清单、§8 交接前自检
- 起点提交：`3942209`（A-8 交付，跟踪文件 191）

---

## 1. 一句话结论

**八项任务的验收标准全部通过，交付物清单里点名而仓库里没有的东西只有一处（`tests/test_no_leakage.py`），已补齐；
另外发现一处会误导 C/D 的统计口径错误（OpenPoly 跨集结构数按原始串统计，把 Tg 的 28 个报成了 15 个），已改正并加了回归锁。**

不改结论、只改证据的是两条文档条目：README 冻结登记第 2、3 行的"未冻结（待 P1）"早已过期（还指向不存在的
`data/manifests/tasks_*.csv` 与 `*.sha256`）。

---

## 2. 验收方法

**不看文档怎么写，跑命令、数文件、比哈希。** 每条验收标准都要能落到一条命令或一次逐行比对；落不到的写进 §6。

| 层面 | 怎么验 | 次数 |
|:--|:--|--:|
| 环境 | `sha256sum -c lock/SHA256SUMS`、三份指纹与 `lock/*.pip.txt` 逐包比对、三个环境实跑 `import` | 3 环境 |
| 下载 | `00_download.py --verify-only`（不出网，逐份核官方 md5/sha256）+ 与 Proposal 附录 A 逐行对拍 | 39 + 69 + 5 行 |
| 调和 | `01_harmonise.py --verify` + 30 条决策逐条查 `evidence` + 逐任务行数差核对 | 10 任务 |
| 配置 | `python -m chainworth.io.tasks --verify` + `sha256sum -c configs/tasks/SHA256SUMS` | 18 份 |
| 划分 | `03_splits.py --verify --deep`（含官方 `manage_scaling_indices.py verify`） | 44 份 |
| 特征 | `02_features.py --verify`（浅层；深层 2 分 32 秒，当日 A-6 已跑过） | 10 份 |
| OpenPoly 计数 | `python -m chainworth.io.openpoly --verify` | 28 列 |
| S27 转录 | `python -m chainworth.io.table_s27 --verify --pdf ...`（含逐行回查原文） | 39 行 |
| 测试 | `pytest tests` | 160 项 |
| 纪律 | `git ls-files data/raw`、`.gitignore`、CRLF 字节检查、文档相对链接扫描 | 全仓 |

---

## 3. 结果一览

| 项 | 验收条目数 | 通过 | 关键数字 |
|:--|--:|--:|:--|
| A-1 三套环境 | 4 | **4** | `lock/SHA256SUMS` 6/6 OK；三份指纹 pip 锁 0 条版本不符；core 8 个 / gnn 3 个 / polybert 1 个核心包 `import` 全通 |
| A-2 三基准下载 | 6 | **6** | 39 行 / 42,169 与附录 A 逐行相等；39 份 CSV 逐份对上官方 md5；`data/raw` 只有 `.gitkeep` 在库里 |
| A-3 标签调和 | 4 | **4** | 10 任务 / 30 条决策 / 20,660 行；`evidence` 全非空；主表最小 n = 228 ≥ 200 |
| A-4 任务清单冻结 | 4 | **4** | 18 份配置（main 8 / reserve 5 / boundary 3 / mechanism 2）；`SHA256SUMS` 18/18 OK |
| A-5 官方划分对齐 | 8 | **8** | 44 份划分；`--verify --deep` 通过；官方自检 `Verified 4 compact scaling-index packages` |
| A-6 结构级表征 | 4 | **4** | 11,494 结构 / 10 份 × 2,255 列 / 116 条失败记账；`--verify` 通过 |
| A-7 OpenPoly 复核 | 2 | **2** | 741 / 3,985 / 443 / 295 / 168 全对；28 列 × 10 列清单可重算 |
| A-8 S27 转录 | 3 | **3** | 39 行 / 14-15-10 / 42,169；含 PDF 逐行回查通过 |

测试基线：**159 passed, 1 skipped**（起点 149 passed，新增 `tests/test_no_leakage.py` 10 项）。
唯一的 skip 是 `test_features.py` 里要 `CHAINWORTH_SLOW=1` 的深层重算——当日已单独跑过，2 分 32 秒通过。

---

## 4. 逐项凭据

### A-1 环境

```
core      | python 3.10.14 | lock.conda/ pip sha256 一致 | pip 锁 35 条，版本不符 0 条
gnn       | python 3.11.18 | 一致                        | pip 锁 70 条，版本不符 0 条
polybert  | python 3.10.14 | 一致                        | pip 锁 29 条，版本不符 0 条
import: rdkit 2024.03.5 / xgboost 2.0.3 / pysmiles / yaml 6.0.3 / pyarrow 17.0.0
        torch 2.3.1+cu121 / torch_geometric 2.5.3 / chemprop 2.1.0 / transformers 4.41.2
```

指纹里多出 `packaging` / `setuptools` / `wheel` 三个包，是 conda 层装进来的、pip 锁里本来就没有——属
"平台/工具链包"，不构成版本不符。

### A-2 下载

- `polybench_download_manifest.csv` 39 行，`n_rows` 合计 **42,169**，与附录 A 解析出的 39 行**名字与条数逐个相等**。
- `polybench26` 69 行 / 90,357；`openpoly` 5 行 / 741。三份清单 `sha256`、`source_url`、`version` 均 100% 非空。
- 验收 ⑥ 的实现方式：`source_url` 全部形如 `https://ndownloader.figshare.com/files/<id>`，该**形态**登记在
  `SOURCES.md` §2.1，逐文件 ID 在 `polybench_expected_files.csv`。即"能追到登记处"，但不是逐条字面出现在 `SOURCES.md`
  里——见 §6 第 1 条。

### A-3 调和

- 10 任务（main 6 + reserve 4）、30 条决策、20,660 行；`rows_in == rows_out` 逐任务成立。
- 唯一"删行"是 `bandgap chain_Egc`：3,380 行整列弃用，记在 `exclude_task` 一行里，没有凭空消失。
- 主表最紧的 `pb_t3_iwlt` n = **228**，仍高于 200 的下限。

### A-4 配置

```
任务数 18：main 8 / reserve 5 / boundary 3 / mechanism 2
划分仍为 pending（1 个，应全部是不可用任务）：pb_t1_bandgap_chain
[OK] 18 份任务配置全部通过：字段、枚举、层数与上游清单一致
configs/tasks && sha256sum -c SHA256SUMS → 18 个 OK
```

### A-5 划分

```
Verified 4 compact scaling-index packages.
校验通过 / verification passed（deep=True）
```

44 份划分 = PolyBench 10 + PolyBench26 4（每个任务一份 `manifest.json` + 5 个 `fold_0x.npz`）+ OpenPoly 30（3 任务 × 10 种子）。`test_splits.py` 41 个用例全绿，
其中"注入一次 `train_test_split` 必红"与"一行交换只有 deep 抓得到"是两条真正能抓错的。

### A-6 特征

`02_features.py --verify` 通过：结构表 11,494 行，10 份特征与清单一致（sha256 / 列序 / 行数 / NaN / 无标签列）。
另按验收 ① 逐任务对账：10 个任务的 `n_rows` **逐一等于** `harmonised_manifest.csv` 的 `rows_out`（合计 20,660 = 20,660，丢失 0 行），且任务内 `n_unique_structures == n_rows`——重复只发生在任务之间。

### A-7 OpenPoly 计数

`openpoly --verify` 通过：28 列非空数、标注合计 3,985、三个点名性质（443 / 295 / 168）与清单一致。

### A-8 S27 转录

```
校验通过：39 行 / Tier 14-15-10 / 合计 42,169 / 与 A-2 清单逐条一致 / 与附录 A 对拍一致
```

含 `--pdf` 时会把 39 行的三个数字逐行回查 SI 原文（需 `pypdf` + 156 MB 原件，缺则跳过）。

---

## 5. 发现并修掉的问题

### 5.1 【交付物缺失】`tests/test_no_leakage.py` 不存在

分工书 §3 的入库清单、§8 自检第 4 条、A-5 步骤 ④ 都点名了这个文件，`分工_B`、`分工_D` 也各自引用它；
仓库里没有，`test_splits.py` 里也没有任何泄漏相关断言（搜 `leak` / `泄漏` / `fit(` 均为空）。

**已补齐**，10 项，管三件互相独立的事：

| 组 | 断言 | 为什么必须独立 |
|:--|:--|:--|
| 特征侧 | 10 份 `descriptors/*.parquet` 与 `structures/psmiles.parquet` 不含 `label` / `y` / `target` / `row_id`；且必须同时有 `structure_id` + `psmiles_canonical` + 至少一个 `morgan_` 列 | A-6 的 `--verify` 也查，但两条独立的路同时出错才会漏；正向断言防"文件被换成空壳" |
| 代码侧 | `src/chainworth/**` 里任何 `fit` / `fit_transform` / `partial_fit`，调用表达式必须能看出训练集（`train` / `_tr` / `tr_`），否则红；允许 `# leak-ok: 理由` 显式豁免 | 现在 `src/` 里一个 `fit` 都没有，所以这条**是闸门不是审计**：拦的是 B、C 以后写进来的代码。附"注入一次 `scaler.fit(X_all)` 必红"的自测，否则 0 == 0 没有意义 |
| 划分侧 | 30 份 OpenPoly 划分的泄漏数与规范式重算逐个相等；`structure_basis` 必须是 `rdkit_canonical_smiles`；Tg seed 0 的规范式数必须**严格大于**原始串数 | 用 5.2 的教训反向锁死：谁把统计改回按原始串，数字掉回 15，这条立刻红 |

### 5.2 【统计口径错误】OpenPoly 跨集结构数按原始串统计，把 Tg 的 28 个报成 15 个

`splits/openpoly/*.json` 的 `leakage.n_psmiles_shared_across_splits` 在 `build_openpoly_split()` 里是拿
`subset["PSMILES"]`（发布方**原始串**）分组的。但 A-7 刚证明：741 行只有 **656 个原始串、585 个规范结构**——
62 行是"换了种写法的同一根链"。只比字符串，这部分重复全部漏掉。

| 任务（seed 0） | 原记录（原始串） | 重算（RDKit 规范式） |
|:--|--:|--:|
| `op_tg` | 15 | **28** |
| `op_tm` | 14 | **17** |
| `op_dielectric_constant_total` | 0 | 0 |

10 个种子全线低估：Tg 规范式 24–36 vs 记录 13–21；Tm 14–21 vs 13–18。

**已修正**：`leakage` 改为三键——`structure_basis`（声明口径）、`n_psmiles_shared_across_splits`（规范式，主口径）、
`n_raw_strings_shared_across_splits`（原始串，只作对照）。30 份划分重生成，
**划分成员一字节未动**（`train_index` / `val_index` / `test_index` / `source_row` 逐份 sha256 全等，差异只在 `leakage` 块）。
登记为 **D-28**；D-24 的原话保留并加勘误标注（历史记录不删，只标"当时按哪个口径"）。

为什么这条重要：Q-06 要 C/D 决定"保留全部行 + 按结构分组切分"还是"去重到 304/83"，而这个数是那个决策的
**紧迫性证据**——把 28 说成 15，等于让"跨集泄漏"看起来只有实际的一半。

### 5.3 【文档过期】README 冻结登记第 2、3 行仍写"未冻结（待 P1）"

- 第 2 行"原始下载文件"：验证方式写的是 `data/manifests/*.sha256`，**仓库里没有任何 `.sha256` 文件**；
  真正的凭据是三份下载清单的 `sha256` 列 + `00_download.py --verify-only`。
- 第 3 行"任务清单与条数"：位置写的是 `data/manifests/tasks_*.csv`，**不存在**；实际是
  `polybench_download_manifest.csv` / `polybench_expected_files.csv` / `openpoly_property_counts.csv`。
- 两行状态都还停在 P1 前。**已按 A-2 / A-7 的实际凭据重写**，状态改为"已交（2026-11-12）"。

### 5.4 【渲染错误】A-8 文档里一行被当成链接

`docs/交付_P1_A8_S27转录.md` 的 Tier 1 表里，η 记号写成「方括号 + 圆括号」的形式，Markdown 会把它读成「链接文字 + 链接目标」，渲染出一个指向 `intrinsic viscosity` 的链接（点开 404）。已把记号改成 `η（intrinsic viscosity）`（全角括号，不加方括号）。

---

## 6. 没改、但需要知道的四条

1. **A-2 验收 ⑥ 是"形态匹配"而不是"逐条字面匹配"**：39 个 `source_url` 各自带不同的 figshare file id，
   不可能逐条抄进 `SOURCES.md`；现在的做法是"形态登记在 SOURCES.md §2.1 + 逐文件 ID 在 `expected_files.csv`"。
   建议将来加一条机器检查：断言 `source_url` 的前缀属于 `SOURCES.md` 里登记过的形态，且 file id 能在
   `polybench_expected_files.csv` 里找到。**这次没加**——它属于 A-2 的范围，加进本次验收会混淆"谁负责"。
2. **`pypdf` 不在环境锁里**：A-8 的 `--verify --pdf` 依赖它，属"有能力就跑、没能力就跳过"。若有人报
   "PDF 回查失败"，先看是不是没装 `pypdf`，不要以为 SI 坏了。`environment/*.yml` 与 `lock/*` 是冻结物，未动。
3. **`scripts/_scratch/` 里有 32 个一次性脚本与中间数据**（含 `op.tgz`、`patch_*.py`）。已全部被 `.gitignore` 挡住，
   `git status` 看不到它们；不影响仓库，但会占本地空间，可随时 `rm -rf scripts/_scratch/*`。
4. **A-5 的 `n_psmiles_shared` 列在 `openpoly_split_manifest.csv` 里现在记的是规范式口径**（D-28）。
   若有人拿 2026-11-12 早先的副本对账，会看到 15/14 而不是 28/17——那不是数据被改坏，是口径换了。

---

## 7. 复核方式（一条命令一条结论）

```bash
# 环境
cd lock && sha256sum -c SHA256SUMS                 # 6 个 OK
# 下载 / 调和 / 配置 / 划分 / 特征 / 计数 / 转录
PYTHONPATH=src python scripts/00_download.py --verify-only
PYTHONPATH=src python scripts/01_harmonise.py --verify
PYTHONPATH=src python -m chainworth.io.tasks --verify
PYTHONPATH=src python scripts/03_splits.py --verify --deep
PYTHONPATH=src python scripts/02_features.py --verify
PYTHONPATH=src python -m chainworth.io.openpoly --verify
PYTHONPATH=src python -m chainworth.io.table_s27 --verify --pdf data/raw/polybench_si/ps6c00094_si_001.pdf
# 冻结配置
cd configs/tasks && sha256sum -c SHA256SUMS        # 18 个 OK
# 测试（Windows 下临时目录必须指定到仓库内，否则沙箱会拦批量删除）
python -m pytest tests -q -p no:cacheprovider --basetemp=.pytest_tmp_check
```

预期：全部退出码 0；`pytest` 输出 `159 passed, 1 skipped`。

---

## 8. 交接状态

| 项 | 状态 |
|:--|:--|
| A-1 → A-8 八项任务 | **全部交付并验收通过** |
| 分工书 §3 交付物清单 | **齐**（缺的 `test_no_leakage.py` 已补，§5.1） |
| 分工书 §8 自检 12 条 | **逐条勾选**，每条写了怎么验的 |
| 阻挡 B / C / D 的事项 | **无**。B 可直接跑 `scripts/02_features.py` 重建结构产物；C/D 的 OpenPoly 分母是 443 / 295 / 168 **行**（对应 304 / 295 / 83 个结构，D-26） |
| 留给 C / D 的待决 | **Q-06**（保留全部行 + 分组切分，还是去重到 304/83）、**Q-07**（S27 补出的 16 个任务要不要进主表） |
