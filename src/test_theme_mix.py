"""Offline checks for the theme-mix comparisons (synthetic labels, no data files)."""

import unittest

import numpy as np
import pandas as pd

from src import theme_mix


class MeasureTests(unittest.TestCase):
    def test_tvd_identical_and_disjoint(self):
        self.assertAlmostEqual(theme_mix.tvd([0.2, 0.3, 0.5], [0.2, 0.3, 0.5]), 0.0)
        self.assertAlmostEqual(theme_mix.tvd([1, 0, 0], [0, 0.5, 0.5]), 1.0)
        self.assertAlmostEqual(theme_mix.tvd([0.5, 0.5], [0.4, 0.6]), 0.1)

    def test_evenness_bounds(self):
        self.assertAlmostEqual(theme_mix.evenness([1, 0, 0, 0]), 0.0)
        self.assertAlmostEqual(theme_mix.evenness([0.25] * 4), 1.0)
        self.assertLess(theme_mix.evenness([0.7, 0.1, 0.1, 0.1]), 1.0)

    def test_wilson_known_value(self):
        # reference values: statsmodels proportion_confint(..., method="wilson")
        lo, hi = theme_mix.wilson(30, 100)
        self.assertAlmostEqual(float(lo), 0.218949, places=5)
        self.assertAlmostEqual(float(hi), 0.395849, places=5)
        lo, hi = theme_mix.wilson(0, 50)
        self.assertAlmostEqual(float(lo), 0.0)
        self.assertAlmostEqual(float(hi), 0.071348, places=5)

    def test_mix_counts_absent_categories(self):
        np.testing.assert_allclose(theme_mix.mix([0, 0, 2, 2], 4), [0.5, 0, 0.5, 0])


class WindowTests(unittest.TestCase):
    def test_three_year_windows(self):
        labels = theme_mix.window_of([2001, 2003, 2004, 2022, 2024])
        self.assertEqual(list(labels), ["2001–03", "2001–03", "2004–06", "2022–24", "2022–24"])
        self.assertEqual(len(set(theme_mix.window_of(range(2001, 2025)))), 8)

    def test_shares_sum_to_one_per_window(self):
        rng = np.random.default_rng(0)
        labels, years = rng.integers(0, 5, 600), rng.integers(2001, 2025, 600)
        table = theme_mix.shares(labels, theme_mix.window_of(years))
        np.testing.assert_allclose(table.groupby("window")["share"].sum(), 1.0)
        self.assertTrue(((table["lo"] <= table["share"]) & (table["share"] <= table["hi"])).all())


class TestTests(unittest.TestCase):
    def setUp(self):
        self.rng = np.random.default_rng(1)

    def test_permutation_no_change(self):
        labels = self.rng.integers(0, 7, 600)
        groups = np.repeat(["a", "b"], 300)
        result = theme_mix.permutation_test(labels, groups, "a", "b", n_perm=500, seed=2)
        self.assertGreater(result["p"], 0.05)
        self.assertGreater(result["noise"], 0)  # TVD is above 0 even with no real change
        self.assertEqual((result["n_a"], result["n_b"]), (300, 300))

    def test_permutation_large_change(self):
        labels = np.concatenate([self.rng.choice(7, 300, p=[.4, .1, .1, .1, .1, .1, .1]),
                                 self.rng.choice(7, 300, p=[.1, .1, .1, .1, .1, .1, .4])])
        groups = np.repeat(["a", "b"], 300)
        result = theme_mix.permutation_test(labels, groups, "a", "b", n_perm=500, seed=2)
        self.assertLess(result["p"], 0.01)
        self.assertGreater(result["above_noise"], 0.2)

    def test_string_labels_and_other_windows_ignored(self):
        labels = np.array(["x", "y"] * 150 + ["z"] * 100)
        groups = np.array(["a"] * 150 + ["b"] * 150 + ["c"] * 100)
        result = theme_mix.permutation_test(labels, groups, "a", "b", n_perm=100)
        self.assertEqual(result["n_a"] + result["n_b"], 300)
        diff = theme_mix.bootstrap_diff(labels, groups, "a", "b", n_boot=200)
        self.assertEqual(list(diff["theme"]), ["x", "y", "z"])
        np.testing.assert_allclose(diff["change"], 0.0)

    def test_bootstrap_interval_covers_change(self):
        labels = np.concatenate([np.repeat([0, 1], [200, 100]), np.repeat([0, 1], [100, 200])])
        groups = np.repeat(["a", "b"], 300)
        diff = theme_mix.bootstrap_diff(labels, groups, "a", "b", n_boot=500)
        np.testing.assert_allclose(diff["change"], [-1 / 3, 1 / 3])
        self.assertTrue(((diff["lo"] < diff["change"]) & (diff["change"] < diff["hi"])).all())

    def test_detection_rate(self):
        even = [1 / 7] * 7
        self.assertLess(theme_mix.detection_rate(even, even, 300, 300, n_sim=200, n_perm=199), 0.15)
        shifted = [.4, .1, .1, .1, .1, .1, .1]
        self.assertGreater(theme_mix.detection_rate(shifted, shifted[::-1], 300, 300, n_sim=50, n_perm=199), 0.9)

    def test_chi_square_df(self):
        labels = self.rng.integers(0, 7, 2400)
        groups = theme_mix.window_of(np.repeat(np.arange(2001, 2025), 100))
        self.assertEqual(theme_mix.chi_square(labels, groups)["df"], 42)

    def test_evenness_by_group(self):
        labels = np.array([0] * 100 + [0, 1, 2, 3] * 25)
        groups = np.repeat(["a", "b"], 100)
        table = theme_mix.evenness_by_group(labels, groups, n_boot=100)
        self.assertAlmostEqual(table.loc[0, "evenness"], 0.0)
        self.assertAlmostEqual(table.loc[1, "evenness"], 1.0)


class EligibilityTests(unittest.TestCase):
    def test_ke_eligible(self):
        pool = pd.DataFrame({"title": ["A", "", None, "D", "E"],
                             "abstract": ["one", "two", "three", "Same ", "same"]})
        self.assertEqual(theme_mix.ke_eligible(pool).tolist(), [True, False, False, False, False])


if __name__ == "__main__":
    unittest.main()
