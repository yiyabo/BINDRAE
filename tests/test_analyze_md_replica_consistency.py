import unittest

import numpy as np

from scripts.analyze_md_replica_consistency import (
    deterministic_explained_energy,
    system_id_from_sample_id,
    weighted_pair_metrics,
)


class MdReplicaConsistencyTest(unittest.TestCase):
    def test_extracts_base_system_from_replica_id(self):
        self.assertEqual(
            system_id_from_sample_id("6x19-R-UK1-501__silver_r01"),
            "6x19-R-UK1-501",
        )
        with self.assertRaises(ValueError):
            system_id_from_sample_id("6x19-R-UK1-501")

    def test_identical_replicas_are_fully_deterministic(self):
        values = np.array([[1.0, -2.0], [1.0, -2.0]])
        weights = np.ones_like(values)
        result = deterministic_explained_energy(values, weights)
        self.assertAlmostEqual(result["explained_energy"], 1.0)
        self.assertAlmostEqual(result["variance_to_energy"], 0.0)

        pair = weighted_pair_metrics(values[0], values[1], weights[0], weights[1])
        self.assertAlmostEqual(pair["cosine"], 1.0)
        self.assertAlmostEqual(pair["relative_rmse"], 0.0)

    def test_opposite_replicas_cancel_for_deterministic_target(self):
        values = np.array([[1.0, -2.0], [-1.0, 2.0]])
        weights = np.ones_like(values)
        result = deterministic_explained_energy(values, weights)
        self.assertAlmostEqual(result["explained_energy"], 0.0)
        self.assertAlmostEqual(result["variance_to_energy"], 1.0)

        pair = weighted_pair_metrics(values[0], values[1], weights[0], weights[1])
        self.assertAlmostEqual(pair["cosine"], -1.0)
        self.assertAlmostEqual(pair["relative_rmse"], 2.0)

    def test_single_replica_cells_do_not_inflate_explained_energy(self):
        values = np.array([[1.0, 100.0], [1.0, -100.0]])
        weights = np.array([[1.0, 1.0], [1.0, 0.0]])
        result = deterministic_explained_energy(values, weights)
        self.assertEqual(result["support"], 1)
        self.assertAlmostEqual(result["explained_energy"], 1.0)

    def test_orthogonal_replicas_retain_half_the_energy_in_the_mean(self):
        values = np.array([[1.0, 0.0], [0.0, 1.0]])
        weights = np.ones_like(values)
        result = deterministic_explained_energy(values, weights)
        self.assertAlmostEqual(result["explained_energy"], 0.5)
        self.assertAlmostEqual(result["variance_to_energy"], 0.5)


if __name__ == "__main__":
    unittest.main()
