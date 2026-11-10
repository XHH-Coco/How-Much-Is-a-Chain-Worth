#!/usr/bin/env python
"""Redraw the file tree inside README.md. Stdlib only.

    python scripts/gen_tree.py          # rewrite the block between the markers
    python scripts/gen_tree.py --check   # exit 1 if README is out of date

Why a generator instead of a hand-maintained tree: a structure section that is
edited by hand goes stale on the first rename, and a stale structure section is
worse than none -- it teaches the reader a layout that does not exist. Here the
file list comes from `git ls-files --cached --others --exclude-standard`, so it
shows exactly what a reviewer gets on clone: nothing ignored by .gitignore.

Per-path annotations live in docs/tree_notes.json, keyed by path (directories end
with "/"). A note whose path no longer exists is an error, not a silent orphan --
that is how a rename gets caught.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Tuple

ROOT = Path(__file__).resolve().parents[1]
README = ROOT / "README.md"
NOTES = ROOT / "docs" / "tree_notes.json"
BEGIN = "<!-- TREE:BEGIN -->"
END = "<!-- TREE:END -->"

SKIP_PARTS = {".git", "__pycache__", ".mypy_cache", ".pytest_cache"}


# --------------------------------------------------------------------------- files


def git_files() -> List[str]:
    """Tracked files plus untracked-but-not-ignored ones, i.e. what a clone shows."""
    try:
        out = subprocess.run(
            ["git", "ls-files", "--cached", "--others", "--exclude-standard"],
            cwd=str(ROOT),
            capture_output=True,
            text=True,
            check=True,
        ).stdout
    except Exception:  # git absent, or not a repository yet
        return walk_files()
    return sorted({ln.strip() for ln in out.splitlines() if ln.strip()})


def walk_files() -> List[str]:
    files = []
    for p in ROOT.rglob("*"):
        if p.is_file() and not SKIP_PARTS & set(p.parts):
            files.append(p.relative_to(ROOT).as_posix())
    return sorted(files)


# --------------------------------------------------------------------------- tree


class Node:
    __slots__ = ("dirs", "files")

    def __init__(self) -> None:
        self.dirs: Dict[str, "Node"] = {}
        self.files: List[str] = []


def build(paths: List[str]) -> Node:
    root = Node()
    for rel in paths:
        parts = rel.split("/")
        cur = root
        for part in parts[:-1]:
            cur = cur.dirs.setdefault(part, Node())
        cur.files.append(parts[-1])
    return root


def render(paths: List[str]) -> List[Tuple[str, str]]:
    """Return (drawn_line, full_path) pairs, root line first."""
    root = build(paths)
    out: List[Tuple[str, str]] = [(ROOT.name + "/", "")]

    def rec(node: Node, prefix: str, path: str) -> None:
        entries = [(True, d) for d in sorted(node.dirs, key=str.lower)]
        entries += [(False, f) for f in sorted(node.files, key=str.lower)]
        for i, (is_dir, name) in enumerate(entries):
            last = i == len(entries) - 1
            drawn = prefix + ("└─ " if last else "├─ ") + name + ("/" if is_dir else "")
            full = f"{path}{name}" + ("/" if is_dir else "")
            out.append((drawn, full))
            if is_dir:
                rec(node.dirs[name], prefix + ("   " if last else "│  "), full)

    rec(root, "", "")
    return out


def load_notes() -> Dict[str, str]:
    if not NOTES.exists():
        return {}
    raw = json.loads(NOTES.read_text(encoding="utf-8"))
    return {k: v for k, v in raw.items() if not k.startswith("_")}


def annotated_block() -> Tuple[str, List[str]]:
    """Return (fenced tree block, list of stale note keys)."""
    pairs = render(git_files())
    notes = load_notes()
    lines, full_paths = zip(*pairs) if pairs else ((), ())

    stale = sorted(k for k in notes if k not in set(full_paths))
    width = max((len(drawn) for drawn, full in pairs if full in notes), default=0)
    body = []
    for drawn, full in pairs:
        note = notes.get(full)
        body.append(f"{drawn.ljust(width)}  ← {note}" if note else drawn)

    block = "```\n" + "\n".join(body) + "\n```"
    return block, stale


# --------------------------------------------------------------------------- readme


def split_readme(text: str) -> Tuple[str, str, str]:
    i = text.index(BEGIN)
    j = text.index(END)
    return text[: i + len(BEGIN)], text[i + len(BEGIN) : j], text[j:]


def current_block() -> str:
    text = README.read_text(encoding="utf-8")
    _, mid, _ = split_readme(text)
    return mid.strip("\n")


def check() -> List[str]:
    problems: List[str] = []
    text = README.read_text(encoding="utf-8")
    if BEGIN not in text or END not in text:
        return [f"README.md has no {BEGIN} / {END} markers"]
    block, stale = annotated_block()
    if current_block() != block:
        problems.append("README.md tree block does not match the working tree "
                        "(fix: python scripts/gen_tree.py)")
    if stale:
        problems.append("docs/tree_notes.json annotates paths that no longer exist: "
                        + ", ".join(stale))
    return problems


def write() -> int:
    block, stale = annotated_block()
    head, _, tail = split_readme(README.read_text(encoding="utf-8"))
    README.write_text(f"{head}\n{block}\n{tail}", encoding="utf-8", newline="\n")
    print(f"rewrote README tree: {len(git_files())} files")
    for key in stale:
        print(f"  STALE note for {key}", file=sys.stderr)
    return 1 if stale else 0


def main(argv: List[str]) -> int:
    if "--check" in argv:
        problems = check()
        for p in problems:
            print("  FAIL " + p)
        print("tree in sync" if not problems else "tree OUT OF DATE")
        return 1 if problems else 0
    return write()


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
