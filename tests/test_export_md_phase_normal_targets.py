import importlib.util
import tempfile
import unittest
from pathlib import Path

import numpy as np

from scripts.export_md_phase_normal_targets import (
    apply_phase_target_mode,
    _extract_endpoint_arrays,
    _load_canonical_residue_axis,
    _scatter_residue_axis,
    _triple_sequence_alignment,
    contact_annotations,
    endpoint_envelope,
    infer_monotone_phase,
    monotone_path_indices,
    smoothstep,
    temporal_circular_average,
    temporal_moving_average,
)


class MdPhaseNormalTargetTest(unittest.TestCase):
    def test_sin2_envelope_is_c1_at_endpoints(self):
        epsilon = 1e-5
        values = np.asarray([0.0, epsilon, 0.5, 1.0 - epsilon, 1.0])
        sin2 = endpoint_envelope(values)
        poly = endpoint_envelope(values, "poly")
        self.assertEqual(float(sin2[0]), 0.0)
        self.assertEqual(float(sin2[-1]), 0.0)
        self.assertAlmostEqual(float(sin2[2]), 1.0)
        self.assertLess(float(sin2[1] / epsilon), 1e-3)
        self.assertGreater(float(poly[1] / epsilon), 3.9)

    def test_identity_phase_target_uses_synchronous_reference(self):
        progress = np.asarray([0.0, 0.4, 1.0])
        tau = np.asarray([[0.0, 0.0], [0.2, 0.7], [1.0, 1.0]])
        confidence = np.full_like(tau, 0.25)
        projection_cost = np.full_like(tau, 3.0)
        identity_cost = np.full_like(tau, 5.0)
        violations = np.asarray([2, 1], dtype=np.int32)

        result = apply_phase_target_mode(
            "identity",
            progress,
            tau,
            confidence,
            projection_cost,
            identity_cost,
            violations,
        )

        np.testing.assert_allclose(
            result[0], np.broadcast_to(progress[:, None], tau.shape)
        )
        np.testing.assert_array_equal(result[1], np.ones_like(confidence))
        np.testing.assert_array_equal(result[2], identity_cost)
        np.testing.assert_array_equal(result[3], np.zeros_like(violations))

    def test_canonical_axis_comes_from_stage2_apo_torsion_cache(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            data_dir = Path(tmpdir)
            sample_dir = data_dir / "samples" / "sample-a"
            sample_dir.mkdir(parents=True)
            np.savez_compressed(
                sample_dir / "torsion_apo.npz",
                residue_keys=np.array(["A|6|", "A|7|", "A|301|"]),
                residue_names=np.array(["ALA", "GLY", "ARG"]),
                residue_alignment_version=np.array(
                    "canonical_residue_key_v2_esm_compatible"
                ),
            )
            keys, names = _load_canonical_residue_axis(
                data_dir, "sample-a__silver_r05"
            )
            self.assertEqual(keys, [("A", 6, ""), ("A", 7, ""), ("A", 301, "")])
            self.assertEqual(names, ["ALA", "GLY", "ARG"])

    def test_sequence_alignment_handles_numbering_offsets_and_masks_mutations(self):
        if importlib.util.find_spec("Bio") is None:
            self.skipTest("BioPython is unavailable in the local test environment")
        aligned = _triple_sequence_alignment(
            ["ALA", "PHE", "GLY", "SER"],
            ["ALA", "LEU", "GLY", "SER"],
            ["ALA", "LEU", "GLY", "SER"],
        )
        self.assertEqual(aligned, [(0, 0, 0), (2, 2, 2), (3, 3, 3)])

    def test_residue_scatter_preserves_canonical_axis(self):
        compact = np.array([[1.0, 2.0], [3.0, 4.0]])
        scattered = _scatter_residue_axis(
            compact, [0, 2], 4, axis=1, fill_value=-1.0
        )
        np.testing.assert_array_equal(
            scattered,
            np.array([[1.0, -1.0, 2.0, -1.0], [3.0, -1.0, 4.0, -1.0]]),
        )

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
