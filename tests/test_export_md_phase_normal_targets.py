import unittest

import numpy as np

from scripts.export_md_phase_normal_targets import (
    _extract_endpoint_arrays,
    contact_annotations,
    infer_monotone_phase,
    monotone_path_indices,
    smoothstep,
    temporal_circular_average,
    temporal_moving_average,
)


class MdPhaseNormalTargetTest(unittest.TestCase):
    def test_single_chain_endpoint_alias_uses_residue_number(self):
        atoms = {
            "N": np.array([0.0, 0.0, 0.0]),
            "CA": np.array([1.0, 0.0, 0.0]),
            "C": np.array([1.0, 1.0, 0.0]),
        }
        residues = [{"key": ("D", 2, ""), "resname": "GLY", "atoms": atoms}]
        arrays = _extract_endpoint_arrays(
            residues, [("A", 2, "")], ignore_chain=True
        )
        np.testing.assert_array_equal(arrays["Ca"], [[1.0, 0.0, 0.0]])

    def test_multichain_endpoint_alias_remains_strict(self):
        atoms = {
            "N": np.array([0.0, 0.0, 0.0]),
            "CA": np.array([1.0, 0.0, 0.0]),
            "C": np.array([1.0, 1.0, 0.0]),
        }
        residues = [{"key": ("D", 2, ""), "resname": "GLY", "atoms": atoms}]
        with self.assertRaises(KeyError):
            _extract_endpoint_arrays(residues, [("A", 2, "")])

    def test_temporal_smoothing_preserves_length_and_wrapped_angles(self):
        values = np.array([0.0, 0.0, 9.0, 0.0, 0.0])[:, None]
        smoothed = temporal_moving_average(values, 3)
        self.assertEqual(smoothed.shape, values.shape)
        self.assertAlmostEqual(float(smoothed[2, 0]), 3.0)
        angles = np.array([3.10, -3.10, 3.12])[:, None]
        circular = temporal_circular_average(angles, 3)
        self.assertTrue(np.all(np.abs(circular[:, 0]) > 3.0))

    def test_monotone_path_enforces_endpoints_and_order(self):
        cost = np.full((4, 5), 10.0)
        cost[0, 0] = 0.0
        cost[1, 3] = 0.0
        cost[2, 1] = 0.0
        cost[2, 3] = 0.2
        cost[3, 4] = 0.0
        path = monotone_path_indices(cost)
        np.testing.assert_array_equal(path, np.array([0, 3, 3, 4]))

    def test_phase_projection_recovers_delayed_progress(self):
        progress = np.linspace(0.0, 1.0, 7)
        tau_grid = np.linspace(0.0, 1.0, 101)
        true_tau = progress**1.6
        observed_translation = np.zeros((len(progress), 3))
        observed_translation[:, 0] = smoothstep(true_tau) * 4.0
        bridge_translation = np.zeros((len(tau_grid), 3))
        bridge_translation[:, 0] = smoothstep(tau_grid) * 4.0
        identity_rotation = np.broadcast_to(np.eye(3), (len(progress), 3, 3)).copy()
        bridge_rotation = np.broadcast_to(np.eye(3), (len(tau_grid), 3, 3)).copy()
        result = infer_monotone_phase(
            identity_rotation,
            observed_translation,
            np.zeros((len(progress), 4)),
            bridge_rotation,
            bridge_translation,
            np.zeros((len(tau_grid), 4)),
            np.zeros(4, dtype=bool),
            progress,
            tau_grid,
            rotation_scale=1.0,
            translation_scale=1.0,
            chi_scale=1.0,
            identity_prior_weight=0.0,
        )
        np.testing.assert_allclose(result["tau"], true_tau, atol=0.02)
        self.assertTrue(np.all(np.diff(result["tau"]) >= 0.0))

    def test_contact_annotations_separate_endpoint_and_transient_events(self):
        progress = np.linspace(0.0, 1.0, 5)
        distances = np.array(
            [
                [7.0, 3.5, 7.0],
                [6.0, 4.0, 4.0],
                [4.0, 5.0, 3.5],
                [3.5, 6.0, 4.0],
                [3.0, 7.0, 7.0],
            ]
        )
        labels = contact_annotations(
            distances,
            progress,
            contact_distance_angstrom=4.5,
            pocket_distance_angstrom=6.0,
        )
        np.testing.assert_array_equal(labels["formed_contact_mask"], [True, False, False])
        np.testing.assert_array_equal(labels["release_mask"], [False, True, False])
        np.testing.assert_array_equal(labels["transient_contact_mask"], [False, False, True])
        self.assertAlmostEqual(float(labels["contact_event_progress"][0]), 0.5)


if __name__ == "__main__":
    unittest.main()
