import unittest

from b2.config import (
    ConfigError,
    all_arms,
    all_experiments,
    all_tasks,
    four_tuple,
    validate_task,
)


class TestTaskConfigs(unittest.TestCase):
    def test_counts(self):
        self.assertEqual(len(all_tasks()), 11)
        self.assertEqual(len(all_arms()), 5)
        self.assertEqual(len(all_experiments()), 4)

    def test_every_task_validates(self):
        for cfg in all_tasks():
            with self.subTest(task=cfg["task_id"]):
                validate_task(cfg)

    def test_main_table_is_eight_tasks(self):
        main = [t for t in all_tasks() if t["benchmark"] in ("polybench", "polybench26")]
        self.assertEqual(len(main), 8)

    def test_two_per_tier(self):
        tiers = [t["tier"] for t in all_tasks() if t["benchmark"] == "polybench"]
        self.assertEqual(sorted(tiers), [1, 1, 2, 2, 3, 3])

    def test_openpoly_is_scan_only(self):
        op = [t for t in all_tasks() if t["benchmark"] == "openpoly"]
        self.assertEqual(len(op), 3)
        for t in op:
            self.assertEqual(t["split"]["kind"], "nested")

    def test_polybench_must_not_use_kfold(self):
        cfg = next(t for t in all_tasks() if t["benchmark"] == "polybench")
        cfg = dict(cfg)
        cfg["split"] = {"kind": "official_5fold", "path": "x"}
        with self.assertRaises(ConfigError):
            validate_task(cfg)

    def test_polybench26_must_use_official_folds(self):
        cfg = next(t for t in all_tasks() if t["benchmark"] == "polybench26")
        broken = dict(cfg)
        broken["split"] = {"kind": "fixed_70_30", "path": "x"}
        with self.assertRaises(ConfigError):
            validate_task(broken)

    def test_missing_field_is_caught(self):
        cfg = dict(all_tasks()[0])
        del cfg["target_unit"]
        with self.assertRaises(ConfigError):
            validate_task(cfg)

    def test_four_tuple_format(self):
        cfg = all_tasks()[0]
        tag = four_tuple(cfg)
        self.assertEqual(tag.split("|")[0], cfg["benchmark"])
        self.assertEqual(tag.split("|")[1], cfg["task_id"])
        self.assertEqual(len(tag.split("|")), 4)


if __name__ == "__main__":
    unittest.main()
