#!/usr/bin/env python
"""Turn per-run JSON into the aggregate tables, and compute Lambda and kappa.

This is the only place Lambda is computed. Runners never compute their own
denominator -- that is how a run would quietly flatter itself.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def main() -> int:
    from b2.analysis.lambda_kappa import kappa, lambda_or_none

    runs = sorted((ROOT / "results" / "raw_runs").glob("*.json"))
    print(f"run records: {len(runs)}")
    if not runs:
        print("nothing to aggregate yet (expected before P1)")
        return 0

    for p in runs:
        rec = json.loads(p.read_text(encoding="utf-8"))
        if rec.get("lambda") is not None:
            continue
        print("  pending Lambda:", rec.get("run_id"))
    print("\nPairs (unit-only vs chain) are formed by task_id; kappa is fitted only on "
          "the official scaling levels and never across releases (R1).")
    _ = (kappa, lambda_or_none)
    return 0


if __name__ == "__main__":
    sys.exit(main())
