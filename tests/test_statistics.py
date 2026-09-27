import unittest

from llm_systems_lab.statistics import bootstrap_mean_ci, relative_change


class StatisticsTests(unittest.TestCase):
    def test_bootstrap_is_deterministic_and_contains_mean(self):
        result = bootstrap_mean_ci([1, 2, 3, 4], samples=500, seed=7)
        self.assertEqual(result, bootstrap_mean_ci([1, 2, 3, 4], samples=500, seed=7))
        self.assertLessEqual(result[1], result[0])
        self.assertGreaterEqual(result[2], result[0])

    def test_relative_change(self):
        self.assertAlmostEqual(relative_change(12, 10), 0.2)


if __name__ == "__main__":
    unittest.main()
