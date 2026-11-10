import csv
import unittest
from collections import Counter
from pathlib import Path

MANIFESTS = Path(__file__).resolve().parents[1] / "data" / "manifests"


def rows(name):
    with open(MANIFESTS / name, encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


class TestPolyBenchManifest(unittest.TestCase):
    def setUp(self):
        self.rows = rows("polybench_tasks.csv")

    def test_thirty_nine_tasks(self):
        self.assertEqual(len(self.rows), 39)

    def test_records_sum_to_published_total(self):
        self.assertEqual(sum(int(r["n"]) for r in self.rows), 42169)

    def test_tier_counts_match_the_main_text(self):
        c = Counter(r["tier"] for r in self.rows)
        self.assertEqual(c["1"], 8)
        self.assertEqual(c["2"], 8)
        self.assertEqual(c["3"], 7)
        self.assertEqual(c["pending S27"], 16)
        self.assertEqual(sum(c.values()), 39)

    def test_six_eligible_main_tasks(self):
        main = [r for r in self.rows if r["eligible_for_main"] == "yes"]
        self.assertEqual(len(main), 6)

    def test_pending_tasks_are_never_marked_eligible(self):
        for r in self.rows:
            if r["tier"] == "pending S27":
                self.assertEqual(r["eligible_for_main"], "no", r["task_id"])


class TestCandidateManifest(unittest.TestCase):
    def test_sixteen_rows(self):
        self.assertEqual(len(rows("candidate_tasks.csv")), 16)

    def test_three_openpoly_properties(self):
        op = [r for r in rows("candidate_tasks.csv") if r["benchmark"] == "openpoly"]
        self.assertEqual(len(op), 3)
        self.assertEqual({r["n"] for r in op}, {"443", "295", "168"})

    def test_reserve_list_is_five(self):
        res = [r for r in rows("candidate_tasks.csv") if r["role_in_design"].startswith("reserve")]
        self.assertEqual(len(res), 5)

    def test_family_mismatch_is_declared(self):
        flag = [r for r in rows("candidate_tasks.csv") if "mismatch" in r["role_in_design"]]
        self.assertEqual(len(flag), 1)


if __name__ == "__main__":
    unittest.main()
