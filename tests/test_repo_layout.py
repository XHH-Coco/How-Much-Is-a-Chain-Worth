import re
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

REQUIRED_FILES = [
    "README.md", ".gitignore", "pyproject.toml", "LICENSE",
    "environment/pins.json", "environment/verify_env.py",
    "environment/b2-core.yml", "environment/b2-gnn.yml", "environment/b2-polybert.yml",
    "data/manifests/polybench_tasks.csv", "data/manifests/candidate_tasks.csv",
    "splits/README.md", "splits/SCHEMA.json",
    "docs/PROTOCOL.md", "docs/DECISIONS.md", "docs/RUNBOOK.md",
    "src/b2/config.py", "src/b2/analysis/lambda_kappa.py",
]

REQUIRED_DIRS = ["configs/tasks", "configs/arms", "configs/experiments",
                 "results/raw_runs", "tests/fixtures/tiny"]


class TestLayout(unittest.TestCase):
    def test_required_files_exist(self):
        for rel in REQUIRED_FILES:
            with self.subTest(file=rel):
                self.assertTrue((ROOT / rel).exists(), rel)

    def test_required_dirs_exist(self):
        for rel in REQUIRED_DIRS:
            with self.subTest(dir=rel):
                self.assertTrue((ROOT / rel).is_dir(), rel)

    def test_gitignore_has_no_blanket_json_or_csv_rule(self):
        text = (ROOT / ".gitignore").read_text(encoding="utf-8")
        active = [ln.strip() for ln in text.splitlines()
                  if ln.strip() and not ln.strip().startswith("#")]
        for bad in ("*.json", "*.csv", "*.txt", "*.md"):
            self.assertNotIn(bad, active, f"{bad} would drop protocol artefacts")

    def test_gitignore_ignores_data_but_not_manifests(self):
        text = (ROOT / ".gitignore").read_text(encoding="utf-8")
        self.assertIn("data/raw/", text)
        self.assertIn("!data/manifests/", text)

    def test_no_stray_python_bytecode(self):
        self.assertEqual(list(ROOT.rglob("*.pyc")), [])


if __name__ == "__main__":
    unittest.main()
