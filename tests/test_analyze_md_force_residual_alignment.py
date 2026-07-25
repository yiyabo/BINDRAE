import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from scripts.analyze_md_force_residual_alignment import (
    ChannelSums,
    channel_metrics,
    fitted_nonnegative_amplitude,
    load_source_records,
    project_dual_force_normal,
    select_cache_frame_indices,
)


class MdForceResidualAlignmentTest(unittest.TestCase):
    def test_dual_force_projection_is_orthogonal_to_tangent(self):
        force = np.asarray([[2.0, 1.0, -1.0], [1.0, 3.0, 2.0]])
        tangent = np.asarray([[1.0, 0.0, 0.0], [1.0, 1.0, 0.0]])
        projected = project_dual_force_normal(force, tangent)
        np.testing.assert_allclose(np.sum(projected * tangent, axis=-1), 0.0, atol=1e-10)

    def test_zero_tangent_keeps_force(self):
        force = np.asarray([[2.0, 1.0, -1.0]])
        projected = project_dual_force_normal(force, np.zeros_like(force))
        np.testing.assert_allclose(projected, force)

    def test_training_amplitude_recovers_directional_target_scale(self):
        sums = ChannelSums()
        for target in (np.asarray([2.0, 0.0]), np.asarray([0.0, 2.0])):
            sums.add(target, target)
        amplitude = fitted_nonnegative_amplitude([sums])
        self.assertAlmostEqual(amplitude, 2.0)
        metrics = channel_metrics(sums, amplitude)
        self.assertAlmostEqual(metrics["relative_mse_reduction"], 1.0)
        self.assertAlmostEqual(metrics["mean_cosine"], 1.0)
        self.assertAlmostEqual(metrics["positive_fraction"], 1.0)

    def test_opposite_force_is_not_given_a_negative_fitted_scale(self):
        sums = ChannelSums()
        sums.add(np.asarray([1.0, 0.0]), np.asarray([-1.0, 0.0]))
        self.assertEqual(fitted_nonnegative_amplitude([sums]), 0.0)
        self.assertAlmostEqual(channel_metrics(sums, 0.0)["relative_mse_reduction"], 0.0)

    def test_fitted_scale_balances_systems_not_residue_time_points(self):
        small = ChannelSums(count=1, direction_dot=2.0)
        large = ChannelSums(count=9, direction_dot=0.0)
        self.assertAlmostEqual(fitted_nonnegative_amplitude([small, large]), 1.0)

    def test_frame_selection_excludes_exact_endpoints(self):
        selected = select_cache_frame_indices(201, 9)
        self.assertEqual(len(selected), 9)
        self.assertGreaterEqual(int(selected.min()), 1)
        self.assertLessEqual(int(selected.max()), 199)
        self.assertTrue(np.all(np.diff(selected) > 0))

    def test_source_manifest_resolves_original_replica_matrix(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            collection = root / "collection"
            target_dir = collection / "targets" / "sys" / "replica_00"
            target_dir.mkdir(parents=True)
            matrix = collection / "replica_matrix.jsonl"
            matrix.write_text(json.dumps({"sample_id": "sys__silver_r00", "pull_dir": "pull"}) + "\n")
            manifest = root / "manifest.jsonl"
            manifest.write_text(
                json.dumps(
                    {
                        "sample_id": "sys__silver_r00",
                        "source_dir": str(target_dir),
                    }
                )
                + "\n"
            )
            resolved = load_source_records([manifest], {"sys__silver_r00"})
            self.assertEqual(resolved["sys__silver_r00"]["pull_dir"], "pull")


if __name__ == "__main__":
    unittest.main()
