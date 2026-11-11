"""项目级常量：物理换算系数与协议门槛，每个常量都带来源注释。
Project-wide constants: physical conversion factors and protocol thresholds, each with its source.

中文：
    用途：把"会在多个阶段被引用的数字"集中到一处，避免同一条换算在两个脚本里写出两种精度。
    依赖：无（只依赖标准库）。
    配置：无——这里放的是**不随实验配置变化**的量；实验参数一律走 `configs/**`。
    对外：CELSIUS_TO_KELVIN、BARRER_IN_SI_UNITS、MAIN_TABLE_MIN_LABELS、POLYBENCH_TOTAL_ROWS。
    被谁用：src/chainworth/io/harmonise.py、scripts/01_harmonise.py、tests/test_harmonise.py。

English:
    Purpose: keep the few numbers that several stages depend on in one place, so the same
    conversion cannot end up with two different precisions in two scripts.
    Depends on: nothing beyond the standard library.
    Config: none - only quantities that do NOT vary with an experiment live here; every
    experiment parameter belongs in `configs/**`.
    Public API: CELSIUS_TO_KELVIN, BARRER_IN_SI_UNITS, MAIN_TABLE_MIN_LABELS,
    POLYBENCH_TOTAL_ROWS.
    Callers: src/chainworth/io/harmonise.py, scripts/01_harmonise.py, tests/test_harmonise.py.
"""

from __future__ import annotations

# ---- 1. 物理换算 / Physical conversions ----

# 摄氏 → 开尔文的偏移。这不是"约等于"，是 SI 定义：0 ℃ := 273.15 K，精度到这里为止。
# 来源：SI 定义（CGPM 1948/1954）；用 273.15 而不是 273 是因为 0.15 K 在 Tg 上已经是可测量差异。
# Celsius -> kelvin offset. Not an approximation but the SI definition: 0 degC := 273.15 K.
# Source: SI definition; 273 is wrong by 0.15 K, which is a measurable difference for Tg.
CELSIUS_TO_KELVIN = 273.15

# 1 Barrer 换算成 PolyBench 渗透系数列所用的量纲 cm3(STP)*cm/(cm2*s*Pa)。
# 推导：1 Barrer = 1e-10 cm3(STP)*cm/(cm2*s*cmHg)；1 cmHg = 1333.22 Pa
#       ⇒ 1e-10 / 1333.22 = 7.5006e-14。只用它做"数值量级是否物理可信"的旁证，不写回数据。
# 来源：Barrer 的标准定义 + cmHg=1333.22 Pa（0 ℃ 下的汞柱）。
# 1 Barrer expressed in the units of PolyBench's permeability column. Used ONLY to sanity-check
# magnitudes; never written back into the data.
BARRER_IN_SI_UNITS = 7.5006e-14

# ---- 2. 协议门槛 / Protocol thresholds ----

# 主表任务的可用标签下限：低于 200 触发备用任务替换。
# 来源：docs/分工_A_数据与合规.md A-3 验收标准②；Proposal 表 E 的准入说明。
# Minimum usable labels per main-table task; below this the reserve task takes over.
MAIN_TABLE_MIN_LABELS = 200

# PolyBench 39 个 CSV 的记录总数（= 附录 A 的 39 行相加）。
# 来源：data/manifests/polybench_download_manifest.csv 的 n_rows 列（本地数出来的，见 DECISIONS D-14）。
# Total records across PolyBench's 39 CSVs, recounted locally (DECISIONS D-14).
POLYBENCH_TOTAL_ROWS = 42169
