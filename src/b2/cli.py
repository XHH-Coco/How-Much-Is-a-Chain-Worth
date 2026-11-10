"""Command line entry point: python -m b2.cli <command>."""

from __future__ import annotations

import argparse
import json
import platform
import sys
from pathlib import Path

from b2 import __version__, config
from b2.analysis.lambda_kappa import lambda_or_none


def cmd_check(args) -> int:
    rc = 0
    for path in sorted((config.REPO_ROOT / "configs" / "tasks").glob("*.json")):
        try:
            config.validate_task(config.load_json(path))
        except Exception as e:  # noqa: BLE001 - report, do not crash the whole sweep
            print("FAIL", path.name, e)
            rc = 1
    print("task configs:", "ok" if rc == 0 else "see above")
    return rc


def cmd_lambda(args) -> int:
    lam = lambda_or_none(args.r2_unit, args.r2_chain)
    print(json.dumps({"lambda": lam, "r2_unit": args.r2_unit, "r2_chain": args.r2_chain}, indent=2))
    return 0 if lam is not None else 0


def cmd_fingerprint(args) -> int:
    import hashlib
    import importlib.metadata as md

    try:
        torch = md.version("torch")
    except md.PackageNotFoundError:
        torch = None
    fp = {
        "python": platform.python_version(),
        "implementation": platform.python_implementation(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "b2_version": __version__,
        "torch": torch,
        "note": "GPU model and driver version are appended by the runner via nvidia-smi; "
                "conda cannot pin the driver, so we record it instead.",
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(fp, indent=2) + "\n", encoding="utf-8")
    print(out)
    return 0


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="b2", description="How Much Is a Chain Worth?")
    p.add_argument("--version", action="version", version=__version__)
    sub = p.add_subparsers(dest="cmd", required=True)

    sub.add_parser("check", help="validate every task config").set_defaults(func=cmd_check)

    pl = sub.add_parser("lambda", help="compute Lambda from two R2 values")
    pl.add_argument("--r2-unit", type=float, required=True)
    pl.add_argument("--r2-chain", type=float, required=True)
    pl.set_defaults(func=cmd_lambda)

    pf = sub.add_parser("fingerprint", help="write an environment fingerprint")
    pf.add_argument("--out", required=True)
    pf.set_defaults(func=cmd_fingerprint)

    args = p.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
