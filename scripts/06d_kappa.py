#!/usr/bin/env python
"""H2 的判据：κ = dΛ/dlog₁₀n，落 `kappa_cells.csv` 与 `kappa.csv`。

H2's criterion: kappa = dLambda/dlog10 n, written to `kappa_cells.csv` and `kappa.csv`.

这个脚本做的四件事
------------------
1. **先跑对账闸门**（`provenance.require_clean()`）。κ 的数会直接进图，所以和 `06c` 一样，
   对账不过就退出码 1，一行 CSV 都不写。
2. **接两支臂**：`scaling_cpu.csv`（A2 单元臂）与 `scaling_chain.csv`（A3 链臂），按
   (数据集, 档位, 折, 种子) 对齐；训练行指纹（`n_subset_sha256`）必须逐格相同，否则直接报错。
3. **按折算斜率再平均**（`analysis.kappa`）：同一条曲线上的点彼此相关，池化 OLS 会让点数多的
   折压死别的折；池化值只作对照。
4. **配对 bootstrap 区间**（B=2000、固定种子；同一折的所有档位共用一组重采样下标），
   并按 H2 的两个条件给判据：区间整段 < 0，**且** 最大档位的 Λ 掉到最小档位的一半以下。

为什么不与 `06c` 合成一个脚本
----------------------------
两者回答不同的问题：`06c` 是"同一档位上链臂比单元臂强多少"（Λ，跨任务），这里是"档位变大时
Λ 怎么动"（κ，单一任务的曲线）。混在一起会让"Λ 表没跑完就不能算 κ"这类依赖变得不可见。

用法 / Usage
------------
    python scripts/06d_kappa.py                  # 闸门 + 点估计 + 2,000 次区间
    python scripts/06d_kappa.py --no-bootstrap   # 只算点估计（几秒）
    python scripts/06d_kappa.py --no-gate        # 跳过对账闸门（只在闸门本身坏了时用）
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from chainworth.analysis import kappa as KP  # noqa: E402
from chainworth.analysis import provenance as PV  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="κ = dΛ/dlog10 n（H2 的判据）")
    parser.add_argument("--no-bootstrap", action="store_true", help="只算点估计，不跑重采样")
    parser.add_argument("--no-gate", action="store_true", help="跳过四向对账闸门")
    parser.add_argument("--draws", type=int, default=KP.BOOTSTRAP_DRAWS, help="重采样次数（默认 2000）")
    args = parser.parse_args(argv)

    if not args.no_gate:
        try:
            summary = PV.require_clean()
        except PV.ProvenanceError as error:
            print(f"对账没过，κ 不产出：{error}", file=sys.stderr)
            return 1
        print(f"对账闸门：{summary['n_ok']} ok / {summary['n_exempt']} exempt / {summary['n_fail']} fail")
    else:
        print("跳过对账闸门（--no-gate）：产出的数没有对账凭据")

    result: dict[str, Any] = KP.write_all(
        bootstrap=not args.no_bootstrap, draws=args.draws
    )
    print(KP.explain(result))

    if not args.no_bootstrap:
        print("H2 判据（每个条件单独报，别只引一句）：")
        for dataset in result["datasets"]:
            point = result["points"][dataset]
            span = float(point["span_dex"])
            gap = float(point["min_gap_dex"])
            # 先看这条斜率该不该报（D-3 验收标准④）。不报的时候也不许静默跳过——
            # 打印一行说明，否则"表里少了一行"会被读成"这个数据集没跑"。
            if span < KP.MIN_SPAN_DEX or gap < KP.MIN_GAP_DEX:
                reason = (
                    f"跨度只有 {span:.3f} 个数量级（门槛 {KP.MIN_SPAN_DEX}）"
                    if span < KP.MIN_SPAN_DEX
                    else f"相邻档位最小间隔 {gap:.3f}（门槛 {KP.MIN_GAP_DEX}），顶上两点近乎重合"
                )
                print(f"  {dataset:<26} **不报 κ**：{reason}")
                continue
            interval = result["intervals"][dataset]
            sign = "是" if (interval.lo == interval.lo and interval.hi < 0.0) else "否"
            first = float(point["lambda_first"])
            last = float(point["lambda_last"])
            if first > 0:
                halved = "是" if last / first < 0.5 else "否"
                ratio = f"{last / first:.3f}"
            else:
                halved = "不可判（Λ_first ≤ 0，'一半'对负数没有意义）"
                ratio = "undefined"
            print(
                f"  {dataset:<26} κ 的 95% 区间整段 < 0：{sign}"
                f"（[{interval.lo:+.4f}, {interval.hi:+.4f}]）；"
                f"Λ 减半：{halved}（比值 {ratio}）"
            )
        print()
        print("杠杆检查（去掉最小档重拟合，D-3 验收标准②）：")
        for row in result["leverage"]:
            if row["kappa_dropped"] == "":
                print(f"  {row['dataset']:<26} 删掉 n={row['dropped_tier']} → {row['note']}")
            else:
                print(
                    f"  {row['dataset']:<26} 删掉 n={row['dropped_tier']}："
                    f"κ {float(row['kappa_base']):+.4f} → {float(row['kappa_dropped']):+.4f}"
                    f"（Δ {row['delta']}）；符号翻转：{row['sign_flipped']}"
                )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
