"""Digests. Every claim in this repo is checked against one of these."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def canonical_json_sha256(obj) -> str:
    """Hash a config the way a reviewer would reproduce it: sorted keys, no spaces."""
    blob = json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return sha256_bytes(blob.encode("utf-8"))


def read_sha256_manifest(path) -> dict:
    """Parse `sha256.txt`: <digest>  <relpath>, '#' comments allowed."""
    out = {}
    p = Path(path)
    if not p.exists():
        return out
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split(None, 1)
        if len(parts) != 2:
            continue
        out[parts[1]] = parts[0]
    return out
