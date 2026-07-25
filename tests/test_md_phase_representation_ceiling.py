import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from scripts.analyze_md_phase_representation_ceiling import (
    identity_phase,
    load_model_eval_ensemble,
    weighted_global_phase,
)


class MdPhaseRepresentationCeilingTest(unittest.TestCase):
    def test_weighted_global_phase_is_endpoint_fixed_and_monotone(self):
        times = np.linspace(0.0, 1.0, 5)
        tau = np.array(
            [
                [0.0, 0.0],
                [0.4, 0.1],
                [0.7, 0.3],
                [0.9, 0.7],
                [1.0, 1.0],
            ]
        )
        confidence = np.ones_like(tau)
        result = weighted_global_phase(
            tau,
            confidence,
            np.ones(2, dtype=bool),
            np.ones(2, dtype=bool),
            times,
            0.05,
        )
        np.testing.assert_allclose(result[:, 0], result[:, 1])
        self.assertEqual(result[0, 0], 0.0)
        self.assertEqual(result[-1, 0], 1.0)
        self.assertTrue(np.all(np.diff(result[:, 0]) >= 0.0))
        self.assertAlmostEqual(result[2, 0], 0.5)

    def test_identity_phase_repeats_the_time_grid(self):
        times = np.array([0.0, 0.5, 1.0])
        result = identity_phase(times, 3)
        self.assertEqual(result.shape, (3, 3))
        np.testing.assert_allclose(result[:, 2], times)

    def test_model_eval_ensemble_averages_matching_replicas(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            paths = []
            for seed, value in ((7, 1.0), (42, 3.0)):
                path = Path(tmp_dir) / f"seed_{seed}.json"
                path.write_text(
                    json.dumps(
                        {
                            "records": [
                                {
                                    "sample_id": "system",
                                    "reference_id": "replica",
                                    "reference_path": "/cache/replica_1.npz",
                                    "metrics": {"error": value},
                                },
                                {
                                    "sample_id": "system",
                                    "reference_id": "replica",
                                    "reference_path": "/cache/replica_2.npz",
                                    "metrics": {"error": value + 2.0},
                                },
                            ]
                        }
                    )
                )
                paths.append(str(path))
            records = load_model_eval_ensemble(paths)
        self.assertEqual(len(records), 2)
        self.assertAlmostEqual(records[0]["metrics"]["error"], 2.0)
        self.assertAlmostEqual(records[1]["metrics"]["error"], 4.0)


if __name__ == "__main__":
    unittest.main()
