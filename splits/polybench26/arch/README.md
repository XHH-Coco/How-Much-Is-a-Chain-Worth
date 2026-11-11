# splits/polybench26/arch —— 留空，且**故意**留空

`pb26_arch_transfer_ea`（架构迁移，H3 的架构子句）的划分**不在仓库里**，原因只有一条：
**官方没有发布它**。

| 项 | 事实 |
|:--|:--|
| 官方发布物 | `Datasets/Dataset_construction_scripts/create_task4_shared_splits.py`（**只发脚本**） |
| 官方没发什么 | 划分索引文件、canonical 表——两者都要本地跑脚本才出现（见 `data/manifests/SOURCES.md`【坑七】） |
| 本仓库怎么办 | 任务配置里 `split_status: deferred`，`split_ref` 指向本目录；**谁要用谁按官方脚本本地物化** |
| 物化后放哪 | `splits/polybench26/arch/`（本地生成、不入库），并在 `PROVENANCE.json` 里记脚本 sha256 与 commit |
| 为什么现在不做 | 它是机制实验（H3 架构子句），不在主表；主表 8 个任务的划分已经齐了。P4 之前谁先用它谁先物化 |

> 这条不是"漏做"，是**刻意的口径**：`deferred` 与 `present` 的区别在于"官方有没有可拷的东西"。
> 把本地生成的索引伪装成官方发布物，是比缺一份文件更严重的问题——它会让人以为那是官方背书的。
