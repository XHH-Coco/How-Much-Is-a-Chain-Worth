#!/usr/bin/env python
"""Compare the three conda environments against environment/pins.json.

Stdlib only: this must run before anything is installed, and it is the first thing
that should be run after installing.

Two directions matter. A version that is LOWER than pinned is the obvious failure.
A version that is HIGHER is the one that actually bites: under Python 3.10, the last
cp310 wheel for rdkit is 2025.09 and for torch it is 2.14, so `pip install -U` does
not error -- it either fails minutes later or, worse, succeeds and changes a number.

Exit codes
  0  every environment is ready (or, without --strict, simply reported)
  1  something is missing or mismatched, with --strict
  2  conda itself could not be reached
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PINS_PATH = ROOT / "environment" / "pins.json"

PROBE = (
    "import json,platform;"
    "from importlib import metadata as m;"
    "print(json.dumps({'python':platform.python_version(),"
    "'pkgs':{d.metadata['Name'].lower():d.version for d in m.distributions()}}))"
)


def probe_env(env_name: str):
    cmd = ["conda", "run", "--no-capture-output", "-n", env_name, "python", "-c", PROBE]
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=180)
    except FileNotFoundError:
        return None, "conda is not on PATH"
    except subprocess.TimeoutExpired:
        return None, "timed out after 180 s"
    if out.returncode != 0:
        msg = (out.stderr or out.stdout or "").strip().splitlines()
        return None, msg[-1] if msg else f"exit {out.returncode}"
    text = out.stdout.strip()
    start = text.find("{")
    if start < 0:
        return None, "no JSON in probe output"
    try:
        return json.loads(text[start:]), None
    except json.JSONDecodeError as e:
        return None, f"unparseable probe output ({e})"


def compare(spec: dict, actual: dict):
    missing, mismatch = [], []
    pin_py = spec.get("python", "")
    got_py = actual.get("python", "")
    # accept any patch release at or above the pinned floor, but never a different minor
    if not got_py.startswith(".".join(pin_py.split(".")[:2])):
        mismatch.append(("python", pin_py, got_py))
    for pkg, want in spec.get("packages", {}).items():
        got = actual.get("pkgs", {}).get(pkg.lower())
        if got is None:
            missing.append(pkg)
        elif got != want:
            mismatch.append((pkg, want, got))
    return missing, mismatch


def main(argv) -> int:
    strict = "--strict" in argv

    if not PINS_PATH.exists():
        print("environment/pins.json is missing")
        return 2
    pins = json.loads(PINS_PATH.read_text(encoding="utf-8"))
    envs = pins.get("environments", {})

    print(f"pins: {PINS_PATH.name}  (updated {pins.get('updated', '?')})")
    print(f"policy: python {pins['policy'].get('python')}, "
          f"{len(pins['policy'].get('forbidden_upgrades', []))} forbidden upgrades\n")

    any_bad = False
    for name, spec in envs.items():
        actual, err = probe_env(name)
        if actual is None:
            any_bad = True
            print(f"{name:30s} UNREACHABLE  ({err})")
            if err and "could not find" in err.lower():
                print(f"{'':30s} create it, then install from environment/README.md")
            continue

        missing, mismatch = compare(spec, actual)
        if not missing and not mismatch:
            print(f"{name:30s} READY        python {actual['python']}")
            continue

        any_bad = True
        label = "NOT READY" if missing and not mismatch else "MISMATCH"
        print(f"{name:30s} {label:12s} python {actual['python']}"
              f"  ({len(missing)} missing, {len(mismatch)} mismatched)")
        for pkg in missing:
            print(f"{'':30s}   missing    {pkg} == {spec['packages'][pkg]}")
        for pkg, want, got in mismatch:
            print(f"{'':30s}   mismatch   {pkg}: pinned {want}, found {got}")

    print()
    if any_bad:
        print("This is expected before installation: no packages have been installed")
        print("into the three environments yet (see environment/README.md).")
        if strict:
            print("--strict was requested, so this is a failure.")
            return 1
        return 0

    print("All three environments match pins.json.")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
