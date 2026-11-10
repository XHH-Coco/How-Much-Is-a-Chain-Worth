"""The structure section in README.md must match the working tree.

`scripts/gen_tree.py` is the generator; this test is the tripwire. It runs the
same code path as `python scripts/gen_tree.py --check`, so `make test` and CI
both refuse a README that describes a layout the repository no longer has.
"""

import importlib.util
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
GEN = ROOT / "scripts" / "gen_tree.py"


def load_gen():
    spec = importlib.util.spec_from_file_location("gen_tree", GEN)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class TestReadmeTree(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.gen = load_gen()
        cls.text = (ROOT / "README.md").read_text(encoding="utf-8")

    def test_markers_present(self):
        self.assertIn(self.gen.BEGIN, self.text)
        self.assertIn(self.gen.END, self.text)

    def test_block_matches_working_tree(self):
        problems = self.gen.check()
        self.assertEqual(problems, [], "run `python scripts/gen_tree.py` to fix")

    def test_notes_have_no_orphans(self):
        _, stale = self.gen.annotated_block()
        self.assertEqual(stale, [], "docs/tree_notes.json points at paths that are gone")

    def test_every_note_appears_in_the_block(self):
        notes = self.gen.load_notes()
        block, _ = self.gen.annotated_block()
        for key, note in notes.items():
            with self.subTest(path=key):
                self.assertIn(note, block, f"note for {key} was dropped while rendering")


if __name__ == "__main__":
    unittest.main()
