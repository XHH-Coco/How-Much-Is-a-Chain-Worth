# splits/polybench —— 官方 70/30 测试集的**行号快照**（只读）

> 官方只发"哪一批 SMILES 是测试集"，不发索引文件。所以这里的 JSON 是**我们做的翻译**：
> 把官方测试集的 SMILES 用 RDKit 规范化，再用它去调和表里找行号。
> 翻译的过程写死在一个函数里（`build_polybench_split()`），**没有第二处实现**。

## 1. 官方那一半是什么

| 项 | 值 |
|:--|:--|
| 接口 | `http://202.98.16.11/api/dataset/test-download?file_name=<数据集名>.csv` |
| 返回 | 两列 `polymer_name,SMILES`，**没有标签**——这正是"测试集成员名单" |
| 规模 | 每个任务约 30%（10 个任务实测 30.01% ~ 30.26%，见 `data/manifests/polybench_official_test_manifest.csv`） |
| 有没有多个种子 | **没有**。`&seed=0/1/2` 返回的字节完全相同（md5 一致）。所以 `eval_unit` 里的"×3"是**模型初始化种子**，不是三份划分（D-20） |
| 训练集 | 官方没单独发 → 取测试集在全量里的**补集**，本仓库不重切 |

取回的文件留在 `data/raw/polybench_official_test/<数据集名>.csv`（不入库），摘要与条数登记在
`data/manifests/polybench_official_test_manifest.csv`（入库）。

## 2. 我们的那一半是什么

`splits/polybench/<task_id>/official_70_30.json`，字段：

| 字段 | 含义 |
|:--|:--|
| `test_index` / `train_index` | **行号**（0 起，指向 `data/interim/harmonised/<task_id>_harmonised.parquet` 的**行位置**） |
| `row_order` | 说明行号按什么排：A-3 那张表按 `source_row` 升序，行位置即原始 CSV 的数据行序 |
| `match_rule` | `rdkit_canonical_smiles`：唯一命中。不按名字、不按顺序猜 |
| `source.sha256` / `source.bytes` / `source.n_rows` | 官方 CSV 的摘要、字节数、行数 |
| `n_train` / `n_test` / `n_total` / `test_ratio` | 四个数必须自洽，`verify_splits()` 逐条查 |
| `protocol` | 恒为 `official_70_30`（PolyBench 上不允许出现 k 折） |

## 3. 怎么校验

```bash
PYTHONPATH=src python scripts/03_splits.py --verify --deep
```

`--deep` 会拿官方 CSV **重算一遍**行号并与落盘的逐个比对——这是唯一能发现"表被重排过"的检查，
默认关闭只因它要跑几千次 RDKit。

## 4. 已落盘 10 个

6 个主表（`pb_t1_tg` / `pb_t1_tm` / `pb_t2_modulus` / `pb_t2_permeability` / `pb_t3_dielectric_loss` /
`pb_t3_iwlt`）+ 4 个预注册备用（`pb_t1_td` / `pb_t2_tssab` / `pb_t3_dlt` / `pb_t3_dmp_sm`）。

`pb_t1_bandgap_chain` **没有**划分文件，因为该任务在 A-3 就按红线整列弃用了（0 条可用标签，D-16）；
没有数据，就没有"哪一行进测试集"可言。

## 5. 规矩

1. **没人可以重新 `train_test_split`**：`tests/test_splits.py` 会扫 `src/` 里有没有这类调用（入口审计 D 也会注入一次看它红不红）。
2. **行号只在当前版本的表上有效**：A-3 的表重建（哪怕只是重排）之后，`--verify --deep` 必须重跑；不跑就等于拿旧行号指新表。
3. **不许把 `train_index` 当"剩下的随便切"**：它就是补集，写进 JSON 是为了让"有没有漏行/重行"能一眼查。
