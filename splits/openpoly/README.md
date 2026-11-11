# `splits/openpoly/` — 按论文公布的规则复现的划分（**不是官方那一份**）

`status: rule_replicated` · 决策 `D-24`（改判 `D-22`）· 建立 `2026-11-12`

---

## 0. 一句话

**官方发布了"怎么切"的规则，没有发布"切好的那份"。** 所以我们照规则切 10 份、把种子公开、报区间——而不是假装自己拿到了作者那份。

打个比方：菜谱公布了（8 份面 1 份馅 1 份汤，按每锅的大小分），但**厨师分好的那一桌没端上来**，也没写他用的随机顺序。照着菜谱再分一桌，能吃到味道一致的东西；但你不能说"这就是他那一桌"。

---

## 1. 规则原文（一手来源）

> Wang, J.-F.; Sun, Y.-B.; Chen, Q.-T.; Ji, F.-F.; Song, Y.-Y.; Ruan, M.-Y.; Wang, Y.
> *OpenPoly: A Polymer Database Empowering Benchmarking and Multi-property Predictions.*
> **Chin. J. Polym. Sci. 2025, 43, 1749–1760.** DOI `10.1007/s10118-025-3402-y`
> 位置：METHODS → **Model Training Protocols**

三句原文：

1. "the dataset was **randomly split** into training, validation, and test sets at an **8:1:1** ratio."
2. "for each property, its numerical value range was divided into **10 equal-width bins**, and samples were **proportionally drawn** from each bin to construct the training, validation, and test subsets."
3. "All models were trained, tuned, and tested on the **same** data partitions."

**缺的那一句**：全文没有 `seed` / `random_state`。所以任何复现都是"同一规则下的一次新抽样"，不可能与作者那份逐位相同。

---

## 2. 怎么切（我们的实现，`stratified_split()`）

| 步骤 | 做法 |
|:--|:--|
| ① 取子表 | 只取 `target_column` 非空的行，保持发布顺序；`source_row` 记下每行是原始 CSV 的第几行（0 基） |
| ② 分箱 | 对该性质的**值域**做 `linspace(min, max, 11)` → 10 个**等宽**箱（不是等频！分位数分箱是另一种抽样） |
| ③ 定总量 | `n_val = round_half_up(0.1 N)`、`n_test = round_half_up(0.1 N)` |
| ④ 摊到箱里 | 两级**最大余数法**（`_apportion()`）：先按箱大小把总量摊到各箱，单箱容量不足（会掏空训练集）的配额退回给别的箱 |
| ⑤ 箱内取 | `np.random.default_rng(seed)` 打乱后**按位置切**——不按数值排序取，那会变成"每箱取最小/最大的几条" |

为什么不用"逐箱各自四舍五入"：误差会累积。168 条的 Tm 试过，训练集被压到 **77.4%**，读起来像另一种协议。改成两级最大余数法之后是 79.76 / 10.12 / 10.12。

---

## 3. 产物

```
splits/openpoly/
├── PROTOCOL.json                 # 规则引文、种子、查询范围、验收靶子、已知局限（机器可读）
├── README.md                     # 本文件（人读）
├── op_tg/                        # 443 条
│   ├── rule_replicated_seed0.json
│   └── ... seed9.json
├── op_dielectric_constant_total/ # 295 条
└── op_tm/                        # 168 条
```

每份 JSON 的内容：`train_index` / `val_index` / `test_index`（对 `source_row` 取下标）、`source_row`（原始行号）、`bin_edges`（11 个边界）、`bin_counts`（每箱条数）、`seed`、`fold_role`、`ratios`、`leakage`。

`data/manifests/openpoly_split_manifest.csv`：一份产物一行（30 行），带 `sha256` 与 `fold_role`。
`data/manifests/splits_manifest.csv`：全局清单，也登记这 30 份（总行数 14 → 44）。

**没有时间戳字段**：同 seed 重跑字节相同，所以清单里的 `created_at` 沿用旧值——重跑不该看起来像有人动过冻结物。

---

## 4. 十份怎么用（与 `eval_unit` 的对应）

| 种子 | 角色 | 用途 |
|:--|:--|:--|
| **0–4** | `primary` | 就是配置里 `eval_unit` 的 **5 折**。措辞已改为 `5 replicated folds (split seeds 0-4)`——它们**不是互斥分块**，是 5 次独立的 8:1:1 抽样（repeated holdout） |
| 5–9 | `reserve` | 划分敏感性分析：同一套 n 档与臂，报 `n*` 的区间而不报单点 |

**总拟合次数不变**：3 性质 × 5 个 n 档 × 5 折 × 3 模型种子 = **225**。

---

## 5. 复现

```bash
# 重新生成 30 份（幂等：同 seed 字节相同）
PYTHONPATH=src python scripts/03_splits.py --source openpoly

# 校验：规则一致性、互斥、比例、清单 sha256；--deep 还会按种子重算一遍
PYTHONPATH=src python scripts/03_splits.py --verify --deep
```

`--deep` 不是可选项。一行索引的手工改动（把 1 条从测试集挪进训练集）在浅层校验下**完全看不出来**——三份仍互斥、仍覆盖全量、比例只动 0.2pp。只有按种子重算才能抓到。`tests/test_splits.py::test_openpoly_shallow_check_cannot_see_a_one_row_swap` 把这条局限写成了测试。

---

## 6. 为什么这不是"自己切数据"（Proposal §7.5 红线）

Proposal §7.5 禁的是「**不读官方划分、自己随手切**」，防的是"没人报错但结果全错"。它的对立面是**隐去来源**，不是禁止复现已公布的规则。这里四条同时成立：

1. 切法**逐字来自期刊正文**，不是我们发明的；
2. 种子**全部公开**（0–9），每份划分带 `bin_edges` 与 `bin_counts`，任何人可逐位复现；
3. 状态名就叫 `rule_replicated`，不叫 `present`；目录里若出现 `*_official*.json` 测试直接变红；
4. 配置 `notes`、本文档、`PROTOCOL.json`、论文限局一节**四处都写明"不是作者那一份"**。

---

## 7. 怎么验收复现对不对（不要靠自觉）

官方仓库发布了逐性质成绩 `results/final_model_test_performance_summary.csv`，XGBoost 那三行正好是我们 H4 的三个性质：

| 性质 | 可用标签 n | 8:1:1 下测试集 | 官方 XGBoost 测试 R² |
|:--|--:|--:|--:|
| `op_tg` | 443 | 44 | **0.904375715** |
| `op_dielectric_constant_total` | 295 | 30 | **0.86** |
| `op_tm` | 168 | 17 | **0.807551835** |

特征与模型也固定：Morgan 指纹 `radius=2` / `2048` 维，`XGBRegressor(random_state=42)`。

**验收规则**：10 份划分各跑一遍，官方值应落在 10 个结果的 **min–max 区间内**（或偏离 ≤ 0.05）。落不进去 = 我们把规则读错了，回头核对 `PROTOCOL.json`，**不许调参凑数**。

---

## 8. 已知局限（写在明面上）

1. **不是作者那份划分**，绝对 R² 不可与论文逐位对比。H4 比的是同一划分内两臂之差（Λ），划分是共同项、不是混淆项——但"我们的 XGBoost 比论文的强/弱"这类话不能说。
2. **Tm 的测试集只有 17 条**。单份划分的 R² 方差很大；这正是要报 10 份区间而不是单点的原因。
3. **结构泄漏**：741 行只有 **656 个不同 PSMILES**。官方规则不按结构分组，所以同一结构可能同时落进训练与测试——实测 Tg **15 个**、Tm **14 个**、介电总 **0 个**。这是既有事实，不是我们引入的偏差；每份文件的 `leakage` 字段如实记账，别遮掩。
4. **`n = all` 与 8:1:1 的口径**：`eval_unit` 的 `all` 指该性质的全部可用标签，而固定测试集占了 10%，所以训练池实际是 90%（训练 80% + 验证 10%）。C/D 在算 Λ 的分母时要与文件的 `n_train`/`n_test` 对齐，不要用 `n_labels`。

---

## 9. 变更记录

| 日期 | 变更 |
|:--|:--|
| 2026-11-12 | 建立。按论文规则复现 30 份（3 性质 × 10 种子），`status: rule_replicated`；删除 `NOT_AVAILABLE.json`（其结论由 `PROTOCOL.json` 的 `supersedes` 承接，D-24） |
