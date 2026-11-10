import unittest

from b2.metrics.bootstrap import bootstrap_ci, ci_excludes_zero, mean, percentile


class TestPercentile(unittest.TestCase):
    def test_median_of_odd_sample(self):
        self.assertAlmostEqual(percentile([1, 2, 3], 0.5), 2.0)

    def test_interpolates(self):
        self.assertAlmostEqual(percentile([0.0, 10.0], 0.25), 2.5)

    def test_single_value(self):
        self.assertAlmostEqual(percentile([7.0], 0.9), 7.0)


class TestBootstrap(unittest.TestCase):
    def test_reproducible_with_seed(self):
        sample = [0.1, 0.2, 0.3, 0.4, 0.5]
        a = bootstrap_ci(sample, n_resamples=500, seed=7)
        b = bootstrap_ci(sample, n_resamples=500, seed=7)
        self.assertEqual(a, b)

    def test_interval_brackets_the_point_estimate(self):
        sample = [x / 100 for x in range(100)]
        ci = bootstrap_ci(sample, n_resamples=2000, seed=1)
        self.assertLessEqual(ci["lo"], ci["point"])
        self.assertGreaterEqual(ci["hi"], ci["point"])

    def test_level_is_respected(self):
        sample = [float(x) for x in range(50)]
        wide = bootstrap_ci(sample, n_resamples=1000, level=0.99, seed=3)
        narrow = bootstrap_ci(sample, n_resamples=1000, level=0.80, seed=3)
        self.assertLess(wide["lo"], narrow["lo"])
        self.assertGreater(wide["hi"], narrow["hi"])

    def test_excludes_zero(self):
        self.assertTrue(ci_excludes_zero({"lo": 0.1, "hi": 0.4}))
        self.assertFalse(ci_excludes_zero({"lo": -0.1, "hi": 0.4}))

    def test_empty_sample_raises(self):
        with self.assertRaises(ValueError):
            bootstrap_ci([])

    def test_mean_helper(self):
        self.assertAlmostEqual(mean([1, 2, 3]), 2.0)


if __name__ == "__main__":
    unittest.main()
