# splits/polybench26 —— 官方划分索引的**原样拷贝**（只读）

> 这里的东西**不是我们生成的**，是从 PolyBench26 仓库里原样拷过来的官方索引。
> 判据只有一条：**一个字节都不能改**。要改就是改冻结物，得先记 `docs/DECISIONS.md`。

## 1. 从哪来

| 项 | 值 |
|:--|:--|
| 仓库 | <https://github.com/rlearsch/PolymerBenchmark2026> |
| commit | `83655bc25901a8cabe3a1e5474b7ec26f01b919f`（2026-11-12 的 `main` HEAD） |
| 路径 | `Datasets/scaling_indices/<实验>/<性质>/<数据集>/seed_42/` |
| 拷贝方式 | `scripts/03_splits.py --source polybench26`（`shutil.copytree`，逐文件核 sha256） |
| 拷了什么 | 主表 2 个 + 机制实验 Cp 的 2 个数据集，共 **4 份 × (manifest.json + fold_00..04.npz)** |

拷贝进来的每一份旁边有一份 `PROVENANCE.json`（放在 `seed_42/` **外面**，保证拷贝目录纯净），记：commit、源目录、每个文件的 sha256、`canonical_source`、`canonical_sha256`、折数与档位。

## 2. 目录

```
splits/polybench26/
├── pb26_polymetrix_tg/          主表第 7 行（PolyMetriX Tg，7,367 行，档位 300/1000/3000）
│   ├── PROVENANCE.json
│   └── seed_42/{manifest.json, fold_00..04.npz}
├── pb26_homopolymer_density/    主表第 8 行（MD_5000 均聚物密度，1,800 行，档位 300/1000）
├── pb26_scaling_cp/             机制实验（H2 / κ）的两个数据集
│   ├── md5000_alternating_cp/   27,802 行，档位 300/1000/3000/10000
│   └── md5000_homopolymer_cp/   1,800 行，档位 300/1000
└── arch/                        留空：官方只发物化脚本，没有可拷的索引（见 arch/README.md）
```

## 3. 怎么校验（两条，都要跑）

```bash
# ① 官方自检：manifest 与折文件是否配套（官方工具不查 train_pool 是否为补集）
python data/raw/polybench26/Datasets/Dataset_construction_scripts/manage_scaling_indices.py \
    verify --indices-root splits/polybench26 --allow-missing-sources
# 期望：Verified 4 compact scaling-index packages.

# ② 我们自己的断言：互斥、覆盖全量、前缀、字节一致、canonical 哈希
PYTHONPATH=src python -m pytest tests/test_splits.py -q
```

`--allow-missing-sources` 不是放水：官方索引本来就不保证 canonical 表在本机。canonical 表由
`python scripts/03_splits.py --source canonical` 按官方脚本物化，并用官方 `canonical_sha256` 验收。

## 4. 四组 sha256（本文件不抄具体值，用命令取，抄一次就会过期）

```bash
# 拷贝过来的每一份文件
cd splits/polybench26 && find . -name "*.npz" -o -name "manifest.json" | sort | xargs sha256sum
# 与源目录逐字节相同（应无输出）
diff -r splits/polybench26/pb26_polymetrix_tg/seed_42 \
        data/raw/polybench26/Datasets/scaling_indices/PolyMetriX/Tg/Tg/seed_42
# canonical 表是否等于官方写下的摘要
PYTHONPATH=src python scripts/03_splits.py --source canonical   # 会逐份打印 "canonical 通过"
```

## 5. 规矩

1. **不许用 `fold_00..04.npz` 重新抽样**：档位是 `train_pool` 的**前缀**（`nested_prefix()` 是唯一取数入口），换一种取法 κ 曲线会变形且不报错。
2. **不许在这里新增文件**：`seed_42/` 里多一个文件，`diff -r` 就不干净了；说明写在 `PROVENANCE.json` 或本文件。
3. **不许把 `data/raw/` 下的源目录删了再拷**：`diff -r` 与 `PROVENANCE.json` 的 sha256 是唯一的"没被改过"证据。
