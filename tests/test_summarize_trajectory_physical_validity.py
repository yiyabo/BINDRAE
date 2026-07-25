import unittest

from scripts.summarize_trajectory_physical_validity import ensemble_method


class SummarizeTrajectoryPhysicalValidityTest(unittest.TestCase):
    def test_ensemble_method_averages_seeds_by_system(self):
        members = [
            [
                {"sample_id": "a", "method": "model", "metric": 1.0},
                {"sample_id": "a", "method": "cubic_ref", "metric": 5.0},
            ],
            [
                {"sample_id": "a", "method": "model", "metric": 3.0},
                {"sample_id": "a", "method": "cubic_ref", "metric": 5.0},
            ],
        ]
        model = ensemble_method(members, "model")
        baseline = ensemble_method(members, "cubic_ref")
        self.assertAlmostEqual(model["a"]["metric"], 2.0)
        self.assertAlmostEqual(baseline["a"]["metric"], 5.0)


if __name__ == "__main__":
    unittest.main()
