"""Config loading and validation. Stdlib only, on purpose (see ADR-001)."""

from __future__ import annotations

import io
import json
from pathlib import Path
from typing import Any, Dict

REPO_ROOT = Path(__file__).resolve().parents[2]

TASK_REQUIRED = (
    "task_id", "dataset_name", "benchmark", "label_source",
    "target_unit", "split", "metrics", "arms",
)
SPLIT_KINDS = ("fixed_70_30", "official_5fold", "nested")
BENCHMARKS = ("polybench", "polybench26", "openpoly")
METRIC_KEYS = ("r2", "mae", "rmse")


class ConfigError(ValueError):
    pass


def load_json(path):
    path = Path(path)
    if not path.is_absolute():
        path = REPO_ROOT / path
    with io_open(path) as f:
        try:
            return json.load(f)
        except json.JSONDecodeError as e:
            raise ConfigError(f"{path}: not valid JSON ({e})") from e


def io_open(path):
    return io.open(path, encoding="utf-8")


def load_task(task_id):
    return load_json(REPO_ROOT / "configs" / "tasks" / (task_id + ".json"))


def load_arm(arm_id):
    return load_json(REPO_ROOT / "configs" / "arms" / (arm_id + ".json"))


def load_experiment(exp_id):
    return load_json(REPO_ROOT / "configs" / "experiments" / (exp_id + ".json"))


def all_tasks():
    return [load_json(p) for p in sorted((REPO_ROOT / "configs" / "tasks").glob("*.json"))]


def all_arms():
    return [load_json(p) for p in sorted((REPO_ROOT / "configs" / "arms").glob("*.json"))]


def all_experiments():
    return [load_json(p) for p in sorted((REPO_ROOT / "configs" / "experiments").glob("*.json"))]


def validate_task(cfg: Dict[str, Any]) -> None:
    """Raise ConfigError on anything that would make a result unattributable."""
    missing = [k for k in TASK_REQUIRED if k not in cfg]
    if missing:
        raise ConfigError(f"{cfg.get('task_id', '?')}: missing {missing}")
    if cfg["benchmark"] not in BENCHMARKS:
        raise ConfigError(f"{cfg['task_id']}: unknown benchmark {cfg['benchmark']}")
    split = cfg["split"]
    if "kind" not in split or "path" not in split:
        raise ConfigError(f"{cfg['task_id']}: split needs kind and path")
    if split["kind"] not in SPLIT_KINDS:
        raise ConfigError(f"{cfg['task_id']}: unknown split kind {split['kind']}")
    if list(cfg["metrics"]) != list(METRIC_KEYS):
        raise ConfigError(f"{cfg['task_id']}: metrics must be exactly {list(METRIC_KEYS)}")
    if not cfg["arms"]:
        raise ConfigError(f"{cfg['task_id']}: no arms")
    if cfg["benchmark"] == "polybench26" and split["kind"] != "official_5fold":
        raise ConfigError(f"{cfg['task_id']}: PolyBench26 must use the official 5-fold files")
    if cfg["benchmark"] == "polybench" and split["kind"] != "fixed_70_30":
        raise ConfigError(
            f"{cfg['task_id']}: PolyBench has ONE fixed 70/30 split; "
            "k-fold does not exist under the official protocol"
        )
    if cfg["benchmark"] == "polybench" and cfg.get("tier") is None:
        raise ConfigError(f"{cfg['task_id']}: main-table PolyBench tasks need a tier")


def four_tuple(cfg: Dict[str, Any], split_version: str = "v1") -> str:
    """The tag every reported Lambda value must carry (R3)."""
    return "|".join([cfg["benchmark"], cfg["task_id"], split_version, cfg["target_unit"]])
