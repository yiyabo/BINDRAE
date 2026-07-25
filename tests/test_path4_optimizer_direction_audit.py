import argparse
import unittest
from pathlib import Path

import numpy as np

from scripts.diagnose_path4_openmm_optimizer_direction import (
    _named_array_sha256,
    atomic_force_diagnostics,
    classify_directional_trials,
    classify_local_derivative,
    compare_numeric_arrays,
    directional_derivative_agreement,
    exact_all_atom_force_derivative,
    parse_step_multipliers,
    predicted_directional_derivatives,
    select_central_difference_plateau,
    validate_unclipped_trial_bound,
)


class Path4OptimizerDirectionAuditTest(unittest.TestCase):
    def test_gpu33_launcher_canonicalizes_prepared_system_path(self):
        launcher = (
            Path(__file__).resolve().parents[1]
            / "scripts"
            / "run_path4_gate0_direction_audit_gpu33.sh"
        ).read_text()
        self.assertIn('if [[ "$PREPARED_SYSTEMS_DIR" != /* ]]', launcher)
        self.assertIn('PREPARED_SYSTEMS_DIR="$ROOT/$PREPARED_SYSTEMS_DIR"', launcher)

    def test_classifies_force_direction_descent(self):
        self.assertEqual(
            classify_directional_trials(10.0, [9.0, 9.5], [11.0, 10.5], tolerance=1e-3),
            "force_direction_descends",
        )

    def test_classifies_reversed_and_neither_direction(self):
        self.assertEqual(
            classify_directional_trials(10.0, [11.0, 10.5], [9.0, 9.5], tolerance=1e-3),
            "opposite_direction_descends",
        )
        self.assertEqual(
            classify_directional_trials(
                10.0, [10.1, 10.2], [10.3, 10.4], tolerance=1e-3
            ),
            "neither_direction_descends",
        )

    def test_local_classification_uses_stable_derivative_not_any_large_step(self):
        self.assertEqual(
            classify_directional_trials(
                10.0, [9.0, 10.01, 10.005], [11.0, 9.99, 9.995], tolerance=1e-4
            ),
            "both_directions_descend",
        )
        plateau = select_central_difference_plateau(
            [1.0, 0.5, 0.25],
            [9.0, 10.01, 10.005],
            [11.0, 9.99, 9.995],
            window_size=2,
            rtol=0.1,
            atol_kj_mol=1.0e-6,
        )
        derivative = plateau["selected"]["selected_derivative_kj_mol"]
        self.assertEqual(
            classify_local_derivative(derivative, tolerance_kj_mol=1.0e-6),
            "opposite_direction_descends",
        )

    def test_central_difference_plateau_selects_smallest_stable_window(self):
        scales = [1.0, 0.5, 0.25, 0.125]
        derivatives = [50.0, -10.0, -10.2, -9.9]
        plus = [scale * derivative for scale, derivative in zip(scales, derivatives)]
        minus = [-value for value in plus]
        result = select_central_difference_plateau(
            scales,
            plus,
            minus,
            window_size=3,
            rtol=0.05,
            atol_kj_mol=0.0,
        )
        self.assertTrue(result["stable"])
        self.assertEqual(result["selected"]["step_multipliers"], scales[1:])
        self.assertAlmostEqual(result["selected"]["selected_derivative_kj_mol"], -10.0)

    def test_central_difference_plateau_can_reject_all_windows(self):
        result = select_central_difference_plateau(
            [1.0, 0.5, 0.25],
            [10.0, -5.0, 2.5],
            [-10.0, 5.0, -2.5],
            window_size=2,
            rtol=0.01,
            atol_kj_mol=0.0,
        )
        self.assertFalse(result["stable"])
        self.assertIsNone(result["selected"])
        self.assertEqual(
            classify_local_derivative(None, tolerance_kj_mol=1.0e-3),
            "no_stable_central_difference_plateau",
        )

    def test_step_multipliers_are_positive_and_decreasing(self):
        self.assertEqual(parse_step_multipliers("1,0.5,0.25"), (1.0, 0.5, 0.25))
        with self.assertRaises(argparse.ArgumentTypeError):
            parse_step_multipliers("0.5,1")
        with self.assertRaises(argparse.ArgumentTypeError):
            parse_step_multipliers("1,0")

    def test_predicted_derivative_uses_force_and_penalty_signs(self):
        forces = np.zeros((3, 1, 3), dtype=np.float64)
        forces[1, 0, 0] = 10.0
        weights = np.asarray([0.0, 1.0, 0.0])
        correction = np.zeros_like(forces)
        correction[1, 0, 0] = 0.1
        direction = np.zeros_like(forces)
        direction[1, 0, 0] = 0.2

        result = predicted_directional_derivatives(
            residue_forces_kj_mol_nm=forces,
            frame_weights=weights,
            correction_angstrom=correction,
            direction_correction_angstrom=direction,
            node_mask=np.asarray([True]),
            magnitude_penalty_kj_mol_a2=5.0,
            temporal_penalty_kj_mol_a2=0.0,
        )

        self.assertAlmostEqual(result["softmax_tail"], -0.2)
        self.assertAlmostEqual(result["magnitude_penalty"], 0.2)
        self.assertAlmostEqual(result["total"], 0.0)

    def test_exact_all_atom_jvp_uses_injected_coordinates_and_force_units(self):
        forces = np.zeros((3, 2, 3), dtype=np.float64)
        forces[1, 0, 0] = 20.0
        weights = np.asarray([0.0, 0.25, 0.0])
        minus = np.zeros_like(forces)
        plus = np.zeros_like(forces)
        plus[1, 0, 0] = 0.4

        result = exact_all_atom_force_derivative(
            atomic_forces_kj_mol_nm=forces,
            frame_weights=weights,
            plus_positions_angstrom=plus,
            minus_positions_angstrom=minus,
            epsilon=0.1,
        )

        self.assertAlmostEqual(result["coordinate_jvp_max_angstrom"], 2.0)
        self.assertAlmostEqual(result["softmax_tail"], -1.0)

    def test_force_saturation_is_reported_by_frame(self):
        forces = np.zeros((3, 2, 3), dtype=np.float64)
        forces[1, 1] = [6.0e8, 8.0e8, 0.0]
        forces[2, 0] = [1.1e9, 0.0, 0.0]

        result = atomic_force_diagnostics(forces, saturation_threshold_kj_mol_nm=1.0e9)

        self.assertEqual(result["saturated_frame_indices"], [1, 2])
        self.assertFalse(result["force_direction_trustworthy"])
        self.assertEqual(result["maximum_force_frame_index"], 2)

    def test_trial_bound_stays_well_below_clipping(self):
        result = validate_unclipped_trial_bound(
            seed_correction_max_angstrom=0.05,
            direction_correction_max_angstrom=0.05,
            largest_step_multiplier=1.0,
            maximum_translation_angstrom=0.75,
        )
        self.assertAlmostEqual(result["upper_bound_to_maximum_ratio"], 2.0 / 15.0)
        with self.assertRaises(ValueError):
            validate_unclipped_trial_bound(
                seed_correction_max_angstrom=0.4,
                direction_correction_max_angstrom=0.1,
                largest_step_multiplier=1.0,
                maximum_translation_angstrom=0.75,
            )

    def test_directional_derivative_gate_requires_signal_and_agreement(self):
        agreeing = directional_derivative_agreement(
            -100.0,
            -95.0,
            rtol=0.1,
            atol_kj_mol=0.01,
            minimum_informative_magnitude_kj_mol=1.0e-6,
        )
        disagreeing = directional_derivative_agreement(
            -100.0,
            20.0,
            rtol=0.1,
            atol_kj_mol=0.01,
            minimum_informative_magnitude_kj_mol=1.0e-6,
        )
        uninformative = directional_derivative_agreement(
            0.0,
            0.0,
            rtol=0.1,
            atol_kj_mol=0.01,
            minimum_informative_magnitude_kj_mol=1.0e-6,
        )
        self.assertTrue(agreeing["agrees"])
        self.assertFalse(disagreeing["agrees"])
        self.assertFalse(uninformative["informative"])
        self.assertFalse(uninformative["agrees"])

    def test_replay_comparison_and_array_hash_are_deterministic(self):
        left = np.asarray([1.0, 2.0], dtype=np.float64)
        close = compare_numeric_arrays(left, left + 1.0e-6, rtol=1.0e-5, atol=1.0e-8)
        far = compare_numeric_arrays(left, left + 1.0e-2, rtol=1.0e-5, atol=1.0e-8)
        self.assertTrue(close["allclose"])
        self.assertFalse(far["allclose"])
        first = _named_array_sha256([("value", left)])
        second = _named_array_sha256([("value", left.copy())])
        changed = _named_array_sha256([("value", left.astype(np.float32))])
        self.assertEqual(first, second)
        self.assertNotEqual(first, changed)


if __name__ == "__main__":
    unittest.main()
