#!/usr/bin/env python
"""Repository self-check. Stdlib only, so it runs before any conda install.

Exit code is non-zero if any check fails; the message names the file and the fix.
"""

from __future__ import annotations

import csv
import json
import re
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

FAILS = []
OKS = []


def ok(msg):
    OKS.append(msg)


def fail(msg):
    FAILS.append(msg)


def check(cond, good, bad):
    (ok if cond else fail)(good if cond else bad)


def main():
    # 1. layout
    needed = ["README.md", ".gitignore", "pyproject.toml",
              "environment/pins.json", "environment/verify_env.py",
              "docs/PROTOCOL.md", "docs/RUNBOOK.md", "docs/DECISIONS.md",
              "splits/SCHEMA.json", "data/manifests/polybench_tasks.csv"]
    for rel in needed:
        check((ROOT / rel).exists(), f"present {rel}", f"MISSING {rel}")

    # 2. every config parses and validates
    from b2.config import all_arms, all_experiments, all_tasks, validate_task
    try:
        tasks = all_tasks()
        arms = all_arms()
        exps = all_experiments()
    except Exception as e:  # noqa: BLE001
        fail(f"configs unreadable: {e}")
        tasks = arms = exps = []

    for t in tasks:
        try:
            validate_task(t)
        except Exception as e:  # noqa: BLE001
            fail(f"task {t.get('task_id')}: {e}")
    check(len(tasks) == 11, "11 task configs", f"expected 11 task configs, found {len(tasks)}")
    check(len(arms) == 5, "5 arms", f"expected 5 arms, found {len(arms)}")
    check(len(exps) == 4, "4 experiments", f"expected 4 experiments, found {len(exps)}")

    # 3. PolyBench manifest arithmetic
    p = ROOT / "data/manifests/polybench_tasks.csv"
    if p.exists():
        with open(p, encoding="utf-8", newline="") as f:
            m = list(csv.DictReader(f))
        check(len(m) == 39, "39 PolyBench tasks", f"manifest has {len(m)} rows")
        total = sum(int(r["n"]) for r in m)
        check(total == 42169, "records sum to 42,169", f"records sum to {total}, expected 42,169")
        c = Counter(r["tier"] for r in m)
        check(c["1"] == 8 and c["2"] == 8 and c["3"] == 7 and c["pending S27"] == 16,
              "tier split 8/8/7 + 16 pending",
              f"tier split is {dict(c)}")

    # 4. environment pins are internally consistent
    pins = ROOT / "environment/pins.json"
    if pins.exists():
        data = json.loads(pins.read_text(encoding="utf-8"))
        envs = data.get("environments", {})
        check(len(envs) == 3, "three environments pinned", f"{len(envs)} environments pinned")
        for name, spec in envs.items():
            check(spec.get("python", "").startswith("3.10"),
                  f"{name} on 3.10.x", f"{name} pins python {spec.get('python')}")

    # 5. .gitignore must not swallow protocol artefacts
    gi = ROOT / ".gitignore"
    if gi.exists():
        active = [ln.strip() for ln in gi.read_text(encoding="utf-8").splitlines()
                  if ln.strip() and not ln.strip().startswith("#")]
        for bad in ("*.json", "*.csv"):
            check(bad not in active, f"{bad} absent from .gitignore",
                  f"{bad} in .gitignore would drop protocol files")

    # 6. no accidental large blobs
    big = [str(f.relative_to(ROOT)) for f in ROOT.rglob("*")
           if f.is_file() and ".git" not in f.parts and f.stat().st_size > 1_000_000]
    check(not big, "no file over 1 MB", f"large files tracked: {big}")

    # 7. the structure section in README.md still describes this repository
    import importlib.util
    spec = importlib.util.spec_from_file_location("gen_tree", ROOT / "scripts/gen_tree.py")
    gen = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(gen)
    problems = gen.check()
    check(not problems,
          "README tree matches the working tree",
          "README tree is stale -- run `python scripts/gen_tree.py`"
          + (": " + "; ".join(problems[:2]) if problems else ""))

    # report
    for line in OKS:
        print("  ok   " + line)
    for line in FAILS:
        print("  FAIL " + line)
    print(f"\n{len(OKS)} passed, {len(FAILS)} failed")
    return 1 if FAILS else 0


if __name__ == "__main__":
    sys.exit(main())
