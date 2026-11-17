# 图 4　信息 vs 架构（H3）

**这张图在说什么**：H3 问的是「Λ 主要来自注入链信息，还是来自换 GNN 架构」。
按设计，回答它需要四根柱子：A2 基准、**A5（同一架构下注入链信息）**、A3（换 GNN 架构）、**打乱对照**。

**这张图缺了两根，而且是关键的两根**

- `raw_runs/` 里 **`a5_chain` 的记录数为 0**（`ls results/raw_runs/*.json | grep -c a5_chain` = 0）——这一臂从未跑过。
- **`scramble_control.csv` 从未产出**，只在 `docs/分工_C-a_经典臂族.md` 的任务书里出现。

后果要说直白：**A3 与 A4 都是「换架构」**。A4 换的是表示（polyBERT 冻结编码器 + ridge），A3 换的是模型
（chemprop MPNN）。**两者都回答不了「信息本身值多少」**——那需要同一架构下只换输入信息，也就是 A5。
所以本报告对 H3 只能说「两条换架构的臂各自比 A2 好多少」，**不能**说「增益来自信息而非架构」。

**怎么读**

- 柱高是 ΔR² 的三种子均值；误差棒是三种子 `delta_r2_lo/hi` 均值的上下界。
- 竖虚线分开 PolyBench 与 PolyBench26：两套评估协议，**不可横向比较**（R1）。
- `n/a` 是真的没有可算的行（例如某任务所有折的分母都不稳），不是 0。

**图上能看出的两件事**

1. **绿色为主**：8 个任务里绝大多数 ΔR² 为负或近零——换架构并没有稳定带来增益。
2. 唯一的强正值是 `pb26_homopolymer_density` 的 A3（ΔR² 约 +0.02）与 `pb_t3_iwlt`；
   而 `pb_t3_dielectric_loss` 的 A3 是 −0.60（R² 从 +0.59 掉到 −0.009），是全部柱子里最差的一根。

**数据来源**：`results/aggregate/lambda_ci.csv`（321 条记录，`op_dielectric_constant_total__a3_graph__seed0_n120__seed0` … `pb_t3_iwlt__a4_polybert__official_70_30__seed2`）。

---

生成时间：2026-11-18 00:16 中国标准时间　|　生成命令：`python scripts/07_figures.py`
