import unittest

from b2.analysis.lambda_kappa import (
    DenominatorUnstable,
    kappa,
    kappa_significant,
    lambda_or_none,
    lambda_recovery,
)


class TestLambda(unittest.TestCase):
    def test_known_value(self):
        # unit model explains 0.50, chain model 0.65 -> (0.65-0.50)/(1-0.50) = 0.30
        self.assertAlmostEqual(lambda_recovery(0.50, 0.65), 0.30, places=12)

    def test_zero_baseline_is_just_the_gain(self):
        self.assertAlmostEqual(lambda_recovery(0.0, 0.2), 0.2, places=12)

    def test_nothing_recovered(self):
        self.assertAlmostEqual(lambda_recovery(0.7, 0.7), 0.0, places=12)

    def test_chain_worse_than_unit_is_negative(self):
        self.assertLess(lambda_recovery(0.6, 0.5), 0.0)

    def test_degenerate_denominator_raises(self):
        with self.assertRaises(DenominatorUnstable):
            lambda_recovery(0.99999, 0.999995)

    def test_batch_version_returns_none_instead_of_raising(self):
        self.assertIsNone(lambda_or_none(0.99999, 0.999995))

    def test_r2_unit_out_of_range(self):
        with self.assertRaises(ValueError):
            lambda_recovery(1.0, 1.0)


class TestKappa(unittest.TestCase):
    def test_exact_decline(self):
        # Lambda falls 0.4 -> 0.2 per decade: slope must be -0.1
        fit = kappa([100, 1000, 10000], [0.4, 0.3, 0.2])
        self.assertAlmostEqual(fit["slope"], -0.1, places=12)
        self.assertEqual(fit["n_points"], 3)
        self.assertEqual(fit["dropped"], 0)

    def test_flat_curve_is_kappa_zero(self):
        fit = kappa([300, 1000, 3000, 10000], [0.2, 0.2, 0.2, 0.2])
        self.assertAlmostEqual(fit["slope"], 0.0, places=12)
        self.assertEqual(fit["claim"], "kappa ~ 0")

    def test_none_points_are_dropped_not_zeroed(self):
        fit = kappa([100, 1000, 10000], [0.4, None, 0.2])
        self.assertEqual(fit["dropped"], 1)
        self.assertEqual(fit["n_points"], 2)

    def test_needs_two_points(self):
        with self.assertRaises(ValueError):
            kappa([100], [0.4])

    def test_length_mismatch(self):
        with self.assertRaises(ValueError):
            kappa([100, 1000], [0.4])

    def test_only_negative_direction_counts_as_h2_falsification(self):
        self.assertTrue(kappa_significant({}, {"lo": -0.3, "hi": -0.05}))
        self.assertFalse(kappa_significant({}, {"lo": -0.3, "hi": 0.05}))
        # a positive kappa is not support for H2, and must not be read as such
        self.assertFalse(kappa_significant({}, {"lo": 0.02, "hi": 0.3}))


if __name__ == "__main__":
    unittest.main()
