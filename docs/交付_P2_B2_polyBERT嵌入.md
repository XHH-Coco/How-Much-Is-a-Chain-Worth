# B-2 交付：polyBERT 结构级嵌入

> **任务**：`docs/分工_B_表示与GPU臂.md` §2 · B-2（2.0 pH）
> **交付日期**：2026-11-12
> **一句话**：11,494 个结构 × 600 维，**17.5 秒**跑完（RTX 4070），行序按规范式排序、与 A-6 的行号解耦；权重来源是**三方互校的副本**（官方仓库已不可公开取得），这件事记在 **D-31**。
>
> **【B-1.5 增量 · 2026-11-12】本记录的数字是"11,494 结构"那一版；现已扩到 18,152 结构 × 600 维（25.4 秒，18 个分片）。** 结构表扩容后 `structure_id` 整体平移，旧分片因 ID 对不上而全部重算——这正是"分片复用要逐位比结构 ID"那条设计的预期行为，别把它当成浪费。权重哈希未变（`21cfab4a…`），未知 token 仍为 0。凭据见 [`交付_P2_B15_补行表与扩表征.md`](交付_P2_B15_补行表与扩表征.md)。

---

## 1. 产物

| 路径 | 内容 | 体积 |
|:--|:--|--:|
| `data/processed/polybert/embeddings.npy` | **11,494 × 600**，float32，行序 = `psmiles_canonical` 升序 | 26.3 MiB |
| `data/processed/polybert/index.parquet` | `row / structure_id / psmiles_canonical / polybert_input / n_tokens / n_unk` | 376 KiB |
| `data/processed/polybert/manifest.json` | 维度、覆盖数、模型 revision、权重 sha256、两份产物哈希 | 1 KiB |
| `data/processed/polybert/shards/shard_00000..00011.npz` | 12 片中间产物，**保留**：断了重跑只补缺的片 | 26 MiB |
| `data/manifests/polybert_manifest.csv` | 上面那份清单的一行（**入库**，便于看历史）。**现 2 行**：v1 = 11,494 结构（B-2 初版，08:08）、v2 = 18,152 结构（B-1.5 扩表征后重嵌入，08:39）。追加式写入、**无版本列**，读的人必须按 `created_at` 取最新一行，别取第一行（2026-11-13 验收时补的说明；只有 `append_manifest_csv()` 写它，没有任何脚本读它，所以是"人会读错"而不是"程序会读错"） | 2 行 |

代码：`src/chainworth/chem/polybert.py`（逻辑）+ `scripts/03c_polybert.py`（入口）+ `tests/test_polybert.py`（15 项）。
**`data/processed/polybert/` 不入库**（`data/processed/*` 被 gitignore），重建命令见 §6。

## 2. 三个口径（改这三个就是改定义，见 D-31）

**① 行序 = 规范式排序，不是 A-6 的行号。** A-6 的 `structure_id` 是**全局行号**，会随上游版本平移（D-25）；按 `psmiles_canonical` 字典序排，换一版结构表也不会把第 7 行换成另一个分子。清单里的 `order_sha256 = a8c9fcea…` 就是这个顺序的指纹，`--verify` 每次重算它。

**② 输入串 = `psmiles_canonical` 把 `*` 换成 `[*]`。** 这条是本任务最容易踩的坑：

> polyBERT 的词表里 `[*]` 是**一个 token**，而裸 `*` **不在词表里**。直接喂 RDKit 规范式 `*C*`，两个 `*` 会变成 `<unk>`——**不报错、不告警，照样返回 600 维向量**，只是数值没有意义。
>
> ```
> '*CC*'      -> ['▁', '<unk>', 'C', 'C', '<unk>']
> '[*]CC[*]'  -> ['▁', '[*]', 'C', 'C', '[*]']
> ```
>
> 所以 `to_polybert_input()` 做这个替换，并**拒绝已经带 `[*]` 的输入**（说明调用方重复转换了，会产出 `[[*]]`）。转换后全量 11,494 条的未知 token 数是 **0**——这是"输入形态对了"的凭据，不是运气。`n_unk` 逐条记进 `index.parquet`，>0 会被 `--verify` 当成问题报出来。

**③ 嵌入以「结构」为单位，不是以「行」为单位。** 结构表 11,494 行就是 11,494 个向量；20,660 行「测量」要靠 B-1 的 `psmiles_canonical` 键回去 join。这与 B-1 的分工一致：B-1 负责"每行对上结构"，B-2 负责"每个结构一个向量"。

## 3. 三条验收标准的凭据

| 标准 | 怎么验的 | 结果 |
|:--|:--|:--|
| ① 嵌入行数 = 输入结构数，顺序对齐可校验 | `--verify` 逐行比 `index.psmiles_canonical` 与"当前结构表按规范式排序"的结果；`order_sha256` 双向核对；测试 `test_order_is_canonical_sort_not_source_row` 另断言"排序后的行序 ≠ A-6 行号序列"（否则这条测试白写） | **11,494 = 11,494**，行序逐行相同 |
| ② ndim 与模型一致并记录 | 加载时断言 `model.config.hidden_size == 600`；清单记 `embed_dim=600`、`model_revision`、`weights_sha256`；测试另查维度不符能被抓 | **600**，`(11494, 600)` float32 |
| ③ 网络断了可续跑（分片落盘） | 12 片 × 1024 行，片内自报 `[row_start, row_stop)`；合并只信行号不信文件名，缺片/重叠/多行少行一律抛错；实测"再跑一次"日志为 `复用已有分片 2/2（续跑）` | 续跑生效，合并不给半份 |

另加三条自查（不在分工书里，但 B-2 的坑都在这）：

- **重算可复现**：`--verify --recompute 50` 真的再跑一遍模型比数值，最大偏差 **< 1e-4**（GPU fp32，同一设备同一批）。
- **输入形态**：`n_unk_tokens = 0`，`n_truncated = 0`（最长 309 token，上限 512）。
- **语义没坏**：最近邻抽查是化学近邻，不是随机邻居——`*CO*` → `*CCO*`（0.906）、`*CCCO*`（0.871）；`*CS*` → `*CCS*`（0.939）；`*CC(*)C` → `*CC(*)CC`（0.925）。随机两行余弦约 0.603（BERT 均值池化共有的公共分量），近邻明显高于它。

## 4. 实际数字

| 项 | 值 |
|:--|:--|
| 结构数 / 维度 / dtype | 11,494 / 600 / float32 |
| 墙钟（GPU，含分词与合并） | **17.49 s**（`device=cuda`） |
| 分片 | 12 × 1024 行（末片 230） |
| 分词长度 | max 309，p99 162，达上限 **0** 条 |
| 未知 token | **0** 个（全量） |
| 向量统计 | 范数 10.82 / 15.05 / 21.85（min/中位/max），均值 −0.0016，标准差 0.6247 |
| `embeddings_sha256` | `f88ea290bf764445bd344b6cd173c1dbbc584862fabfe4340c4c7ee975a44cfc` |
| `index_sha256` | `036aa24dfddbeef8b2a6afa2c07b759d2ee6d94a65b1fa7d9803ce96319e0f7d` |
| 测试 | `test_polybert.py` **14 passed + 1 skipped**；全仓 **191 passed / 3 skipped**（原 177/2） |

## 5. 权重来源：官方仓库拿不到，用三方互校的副本（**这一节请务必看**）

分工书写的是"从 HuggingFace 取权重（缓存约 7.3 GB）"。实测两件事都和它不一样：

**（1）官方仓库 `kuelumbus/polyBERT` 2026-11-12 已不可公开取得。** 证据链：

```
GET /api/models/kuelumbus/polyBERT   -> 404  X-Error-Code: RepoNotFound
GET /api/models?author=kuelumbus     -> 0 个公开模型
对照：/api/models/google/bert_uncased_L-2_H-128_A-2 -> 200；author=google -> 3 条
```

`RepoNotFound` 不是"要登录"（gated 会返回 401 `GatedRepo`），也不是网络问题（同一出口访问别的仓库正常）。

**（2）权重只有 97 MB，不是 7.3 GB。** 25,194,056 个参数、`hidden_size 600`、12 层、`vocab_size 269`、`max_position_embeddings 512`——是 DeBERTa-v2 的小尺寸变体，不是 xlarge。`environment/polybert.yml` 里"~7 GB 要能单独删掉"那句是估大了（yml 已冻结，不改，仅在此记录）。

**替代方案与互校。** 镜像上有三份独立上传的副本，用两种格式互相印证：

| 仓库 | 文件 | 证据 |
|:--|:--|:--|
| `xushijie/polyBERT` | `model.safetensors` | sha256 `21cfab4a…`，100,800,352 B |
| `HAYDERphd/polyBERT` | `model.safetensors` | **与上一条逐字节相同**（同 sha256） |
| `cooperzvegintzov/polyBERT` | `pytorch_model.bin` | 转出 197 个张量，与上者**逐张量相等、零处不同** |

三方独立上传、两种序列化格式互证同一组参数；`config.json` 的 `_name_or_path` 写着 `polyBERT_sentence_transformers/`，与论文描述（600 维指纹、DebertaV2Model、均值池化）一致；语义近邻抽查也是化学近邻。**采用它，并把来源写死**：`model_repo`、`model_revision`（commit sha）、`weights_sha256` 全部进清单，加载时**核哈希**——换了一份权重忘记录，加载就失败。

**未消除的风险（必须写进论文的可复现性说明）**：副本来自第三方，无法与官方原件逐字节对证。缓解手段是"换回官方只需一条命令"：拿到官方权重后

```
python scripts/03c_polybert.py --model kuelumbus/polyBERT --revision <sha> \
    --verify --recompute 200
```

若两份嵌入的最大偏差 < 1e-4，说明副本与官方等价，D-31 可结案；若偏差大，A4 的结论要按官方权重重跑。

## 6. 重建与验收

```bash
# 全量（RTX 4070 上约 20 秒；CPU 也能跑，慢一个数量级）
PYTHONPATH=src D:/Miniconda3/envs/DSC6020_Project_Polybert/python.exe scripts/03c_polybert.py

# 只校验（不加载模型，1 秒）
PYTHONPATH=src D:/Miniconda3/envs/DSC6020_Project_Polybert/python.exe scripts/03c_polybert.py --verify

# 校验 + 重算 50 行比对数值（会占卡几秒）
... scripts/03c_polybert.py --verify --recompute 50

# 试跑（产物落 _partial_<N>/，从路径上就不是全量）
... scripts/03c_polybert.py --limit 200

# 只合并已有分片（比如模型那边断了、片都齐了）
... scripts/03c_polybert.py --merge-only
```

**环境说明**：`DSC6020_Project_Polybert` 里**没有 pandas、没有 pytest**（有意精简，只放 torch/transformers/pyarrow）。所以 `polybert.py` 读 parquet 用 **pyarrow**，测试里的模型那条在核心环境与 GNN 环境都自动 skip——模型路径的自动化覆盖由 `--verify --recompute` 承担（本次已跑过 50 行）。

## 7. 没做到的

1. **只覆盖了 11,494 个 PolyBench 结构。** PolyBench26 与 OpenPoly 的结构不在 A-6 结构表里（它们的标签也不在 A-3 调和表里），所以这批嵌入**没有**它们。补法与 B-1 那条未做项同路：先扩结构表，再重跑 `03c`（分片机制保证只算新增的片）。
2. **没跑 A4 臂**（B-4）。它有前置：`embeddings.npy` 就绪（现在有了）+ 行表对齐任务（PB26/OpenPoly 仍未扩）。
3. **官方权重仍未取得。** 双向比对（§5 的 `--recompute` 方案）要等拿到官方仓库权限才能做。
4. **`--limit` 的产物不删。** 它们落在 `_partial_<N>/`，`du` 会算进 `data/processed/polybert/`；发布前若看到可疑体积，先查这个目录。
