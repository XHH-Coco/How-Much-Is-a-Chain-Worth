#!/usr/bin/env python
"""审计入口：对账、入口/出口两类检查。数字出图前的闸门就在这里。

The audit entry point: provenance, entry-side and exit-side checks. This is the gate the numbers
must pass before they may be plotted.

为什么是 06b 而不是 06
--------------------
README 第 4.2.1 节的编号表里，`06` 已经被 C 的 `06_aggregate.py`（主表汇总）占了，`07` 是 D 的
`07_figures.py`（出图）。审计既不是"C 的汇总"也不是"图"，它是**图之前的一道闸门**，所以按
`00b` / `03b` / `04b` / `05b` 的先例取插入式编号 **06b**，不挤占 06 / 07。这条按 README 的规矩
"谁接手别人的编号，先改这张表并记一条"登记在 D-67。

`make provenance` 在哪
----------------------
仓库里没有 Makefile（也不该为一条命令引入一个构建系统）。分工书 D-8 写的 `make provenance`
在这里的等价物是：

    python scripts/06b_audit.py provenance

它做四件事，与 D-8 的验收逐条对应：
  ① 跑四向对账（`chainworth.analysis.provenance.collect()`）并重写
     `results/aggregate/provenance_check.csv`；
  ② 控制台打印按 check 分诊的报告（缺件 / 失败 / 豁免分开列，不混成一个红）；
  ③ `--strict`（默认开）在仍有 `fail` 时以退出码 1 结束——CI 与出图脚本靠这个；
  ④ `--expect-clean-only` 供出图脚本调用：只判不改（不重写表），避免画图时把别人的表覆盖掉。

用法 / Usage
------------
    python scripts/06b_audit.py provenance
    python scripts/06b_audit.py provenance --no-write
    python scripts/06b_audit.py provenance --json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from chainworth.analysis import provenance as P  # noqa: E402


def _cmd_provenance(args: argparse.Namespace) -> int:
    """跑四向对账，打印分诊报告，按需落盘。
    Run the four-way audit, print the triage report and write the table when asked.

    中文：
        输入：args（`--no-write` / `--json` / `--limit`）。
        输出：退出码（0 = 无 fail；1 = 有 fail）。
        数据来源：`results/raw_runs/`、`results/aggregate/`、`splits/`、`results/env_fingerprints/`。
        副作用：写 `results/aggregate/provenance_check.csv`（除非 `--no-write`）。
    English:
        Args: `--no-write` skips the table, `--json` prints the summary as JSON, `--limit` caps the
        failure list. Returns 0 when nothing failed, 1 otherwise. Writes the CSV unless told not to.
    """
    rows = P.collect()
    summary = P.summarise(rows)

    if args.no_write:
        # 只判不改：出图脚本用这条，避免在画图时重写别人刚生成的对账表。
        pass
    else:
        target = P.write_table(rows)
        print(f"已写入 {target.relative_to(REPO_ROOT).as_posix()}")

    if args.json:
        payload = {
            key: summary[key]
            for key in ("n_rows", "n_ok", "n_fail", "n_exempt", "pass_rate")
        }
        payload["by_check"] = summary["by_check"]
        payload["failures"] = [
            {"check": row.check, "run_id": row.run_id, "detail": row.detail}
            for row in summary["failures"]
        ]
        print(json.dumps(payload, ensure_ascii=False, indent=2))
    else:
        print(P.render(summary, limit=args.limit))

    return 1 if summary["n_fail"] else 0


def build_parser() -> argparse.ArgumentParser:
    """构造 CLI。/ Build the CLI."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    prove = sub.add_parser("provenance", help="四向对账：划分 / 指纹 / 预测 / 汇总表反查")
    prove.add_argument("--no-write", action="store_true", help="只判不改，不重写对账表")
    prove.add_argument("--json", action="store_true", help="以 JSON 打印汇总（给 CI 与出图脚本读）")
    prove.add_argument("--limit", type=int, default=20, help="失败明细最多打印几条（默认 20）")
    prove.set_defaults(func=_cmd_provenance)
    return parser


def main(argv: list[str] | None = None) -> int:
    """解析参数并分发。/ Parse the arguments and dispatch."""
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
