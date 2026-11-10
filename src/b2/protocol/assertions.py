"""Assertions that fail loudly rather than quietly producing a nicer number."""

from __future__ import annotations

from pathlib import Path

from b2.io.hash import read_sha256_manifest, sha256_file


class FrozenArtifactChanged(RuntimeError):
    pass


def assert_digest(path, expected: str) -> None:
    actual = sha256_file(path)
    if actual != expected:
        raise FrozenArtifactChanged(
            "%s changed.\n  expected %s\n  actual   %s\n"
            "A split that moves is a different experiment. Restore the file; do not "
            "update the digest to match." % (Path(path).name, expected, actual)
        )


def assert_matches_manifest(root, manifest_path) -> int:
    """Verify every entry in sha256.txt. Returns the number checked."""
    root = Path(root)
    manifest = read_sha256_manifest(manifest_path)
    checked = 0
    for rel, digest in manifest.items():
        target = root / rel
        if not target.exists():
            raise FileNotFoundError(f"manifest lists {rel} but it is not on disk")
        assert_digest(target, digest)
        checked += 1
    return checked


def assert_manifest_frozen(manifest_lock) -> bool:
    with open(manifest_lock, encoding="utf-8") as f:
        import json
        return bool(json.load(f).get("frozen", False))
