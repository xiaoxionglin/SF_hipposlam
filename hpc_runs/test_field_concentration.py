"""Scientific invariants for saved-map spatial concentration."""

import unittest

import numpy as np

from hpc_runs.intrmotiv_study.field_concentration import calculate_field_concentration


class ConcentrationTests(unittest.TestCase):
    def test_uniform_and_single_peak_have_expected_endpoints(self):
        maps = np.ones((3, 3, 2))
        maps[:, :, 1] = 0
        maps[1, 1, 1] = 2
        result = calculate_field_concentration(maps, np.ones((3, 3)))
        np.testing.assert_allclose(result["effective_area_bins"], [9, 1])
        np.testing.assert_allclose(result["concentration"], [0, 1], atol=1e-14)
        np.testing.assert_allclose(result["entropy_concentration"], [0, 1], atol=1e-14)
        np.testing.assert_allclose(result["mass80_area_fraction"], [0.8, 0.8 / 9])

    def test_activity_gain_cannot_change_shape(self):
        maps = np.arange(18, dtype=float).reshape(3, 3, 2)
        first = calculate_field_concentration(maps, np.ones((3, 3)))
        scaled = calculate_field_concentration(maps * np.array([1e-100, 1e100]), np.ones((3, 3)))
        for key in first:
            np.testing.assert_allclose(first[key], scaled[key], equal_nan=True)

    def test_silent_units_are_undefined_not_maximally_concentrated(self):
        result = calculate_field_concentration(np.zeros((3, 3, 2)), np.ones((3, 3)))
        self.assertTrue(np.isnan(result["concentration"]).all())
        self.assertTrue(np.isnan(result["effective_area_bins"]).all())

    def test_unknown_bins_and_visit_counts_do_not_supply_zero_activity(self):
        maps = np.array([1.0, 1.0, np.nan, 100.0]).reshape(2, 2, 1)
        result = calculate_field_concentration(maps, np.array([[1, 200], [0, 0]]))
        self.assertEqual(result["supported_bins"], 2)
        self.assertAlmostEqual(result["concentration"][0], 0)
        self.assertAlmostEqual(result["effective_area_bins"][0], 2)

    def test_geometric_fragmentation_is_distinct_from_concentration(self):
        compact = np.zeros((5, 5, 1))
        compact[1:3, 1:3, 0] = 1
        fragmented = np.zeros_like(compact)
        fragmented[::4, ::4, 0] = 1
        first = calculate_field_concentration(compact, np.ones((5, 5)))
        second = calculate_field_concentration(fragmented, np.ones((5, 5)))
        for key in first:
            np.testing.assert_allclose(first[key], second[key], equal_nan=True)

    def test_insufficient_support_and_invalid_rates(self):
        result = calculate_field_concentration(np.ones((2, 2, 1)), np.array([[1, 0], [0, 0]]))
        self.assertTrue(np.isnan(result["concentration"][0]))
        self.assertEqual(result["effective_area_bins"][0], 1)
        with self.assertRaises(ValueError):
            calculate_field_concentration(-np.ones((2, 2, 1)), np.ones((2, 2)))
        with self.assertRaises(ValueError):
            calculate_field_concentration(np.ones((2, 2, 1)), np.ones((2, 2)), minimum_bin_observations=0)


if __name__ == "__main__":
    unittest.main()
