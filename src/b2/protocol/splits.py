"""Split files: load, validate, and refuse to guess.

A split file is an index file, not data. Three kinds coexist (see splits/README.md)
and the only thing that makes them comparable is that each one is frozen and
digest-checked.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

REQUIRED = ("task_id", "benchmark", "kind", "n_total", "source", "folds")


class SplitError(ValueError):
    pass


def load_split(path) -> Dict:
    path = Path(path)
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def validate_split(split: Dict) -> None:
    missing = [k for k in REQUIRED if k not in split]
    if missing:
        raise SplitError(f"split missing {missing}")
    if split["kind"] not in ("fixed_70_30", "official_5fold", "nested"):
        raise SplitError(f"unknown kind {split['kind']}")
    folds = split["folds"]
    if not isinstance(folds, list) or not folds:
        raise SplitError("folds must be a non-empty list")
    if split["kind"] == "fixed_70_30" and len(folds) != 1:
        raise SplitError("PolyBench has exactly one official split, not %d" % len(folds))
    seen_total = 0
    for i, fold in enumerate(folds):
        for key in ("train", "test"):
            if key not in fold:
                raise SplitError(f"fold {i} missing {key}")
        tr, te = set(fold["train"]), set(fold["test"])
        if tr & te:
            raise SplitError(f"fold {i}: %d indices leak across train and test" % len(tr & te))
        hi = max(max(tr, default=-1), max(te, default=-1))
        if hi >= split["n_total"]:
            raise SplitError(f"fold {i}: index {hi} outside n_total={split['n_total']}")
        seen_total = max(seen_total, len(tr) + len(te))


def fold_sizes(split: Dict) -> List[Dict]:
    return [{"fold": i, "train": len(f["train"]), "test": len(f["test"])}
            for i, f in enumerate(split["folds"])]
