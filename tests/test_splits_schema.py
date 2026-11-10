import json
import unittest
from pathlib import Path

from b2.protocol.splits import SplitError, fold_sizes, load_split, validate_split

FIXTURE = Path(__file__).resolve().parent / "fixtures" / "tiny" / "split.json"


class TestSplitSchema(unittest.TestCase):
    def test_fixture_is_valid(self):
        split = load_split(FIXTURE)
        validate_split(split)
        self.assertEqual(fold_sizes(split), [{"fold": 0, "train": 8, "test": 4}])

    def test_leak_is_rejected(self):
        split = load_split(FIXTURE)
        split["folds"][0]["test"] = [7, 8, 9, 10]
        with self.assertRaises(SplitError):
            validate_split(split)

    def test_index_beyond_n_total_is_rejected(self):
        split = load_split(FIXTURE)
        split["folds"][0]["test"] = [8, 9, 10, 99]
        with self.assertRaises(SplitError):
            validate_split(split)

    def test_polybench_cannot_have_two_folds(self):
        split = load_split(FIXTURE)
        split["folds"] = split["folds"] * 2
        with self.assertRaises(SplitError):
            validate_split(split)

    def test_missing_key_is_rejected(self):
        split = load_split(FIXTURE)
        del split["n_total"]
        with self.assertRaises(SplitError):
            validate_split(split)

    def test_official_five_fold_is_allowed(self):
        base = load_split(FIXTURE)
        five = dict(base)
        five["kind"] = "official_5fold"
        five["folds"] = [{"train": [0, 1, 2, 3, 4, 5], "test": [6, 7, 8, 9]} for _ in range(5)]
        validate_split(five)


if __name__ == "__main__":
    unittest.main()
