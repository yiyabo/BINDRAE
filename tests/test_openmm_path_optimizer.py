import unittest

import numpy as np

from src.data.openmm_path_optimizer import (
    coefficient_force_direction,
    correction_from_coefficients,
    finite_difference_path_tangent,
    normalize_coefficient_step,
    project_translation_normal,
    sine_time_basis,
    softmax_tail_objective,
)


class OpenMMPathOptimizerGeometryTest(unittest.TestCase):
    def test_basis_and_correction_preserve_endpoints_and_normality(self):
        times = np.linspace(0.0, 1.0, 5)
        basis = sine_time_basis(times, 2)
        ca = np.zeros((5, 2, 3), dtype=np.float64)
        ca[:, :, 0] = times[:, None]
        tangent = finite_difference_path_tangent(
            ca, times, np.asarray([True, True])
        )
        coefficients = np.ones((2, 2, 3), dtype=np.float64)

        correction = correction_from_coefficients(
            coefficients,
            basis,
            tangent,
            np.asarray([True, True]),
            max_residue_translation_angstrom=0.4,
        )

        np.testing.assert_array_equal(correction[[0, -1]], 0.0)
        parallel = np.sum(correction * tangent, axis=-1)
        np.testing.assert_allclose(parallel, 0.0, atol=1e-10)
        self.assertLessEqual(np.linalg.norm(correction, axis=-1).max(), 0.4 + 1e-12)

    def test_force_direction_and_step_normalization(self):
        times = np.linspace(0.0, 1.0, 5)
        basis = sine_time_basis(times, 2)
        tangent = np.zeros((5, 3, 3), dtype=np.float64)
        tangent[..., 0] = 1.0
        forces = np.zeros_like(tangent)
        forces[2, :, 1] = [1.0, 2.0, 3.0]
        weights = np.zeros((5,), dtype=np.float64)
        weights[2] = 1.0
        nodes = np.asarray([True, True, True])

        direction = coefficient_force_direction(
            forces,
            weights,
            basis,
            tangent,
            nodes,
            np.asarray([True, True]),
            chain_smoothing_steps=1,
        )
        normalized, raw_max = normalize_coefficient_step(
            direction, basis, tangent, nodes, 0.05
        )
        induced = correction_from_coefficients(
            normalized, basis, tangent, nodes
        )

        self.assertGreater(raw_max, 0.0)
        self.assertAlmostEqual(np.linalg.norm(induced, axis=-1).max(), 0.05)
        np.testing.assert_allclose(induced[..., 0], 0.0, atol=1e-12)

    def test_softmax_tail_weights_focus_on_high_excess(self):
        objective, weights, excess = softmax_tail_objective(
            [0.0, 2.0, 8.0, 3.0, 0.0],
            np.linspace(0.0, 1.0, 5),
            scale_kj_mol=8.0,
            beta=8.0,
        )

        self.assertGreater(objective, 0.0)
        self.assertEqual(int(np.argmax(weights)), 2)
        self.assertAlmostEqual(float(weights.sum()), 1.0)
        self.assertEqual(float(excess[2]), 8.0)

    def test_zero_tangent_retains_full_vector(self):
        vectors = np.ones((3, 2, 3), dtype=np.float64)
        projected = project_translation_normal(
            vectors, np.zeros_like(vectors), np.asarray([True, False])
        )
        np.testing.assert_array_equal(projected[:, 0], 1.0)
        np.testing.assert_array_equal(projected[:, 1], 0.0)


if __name__ == "__main__":
    unittest.main()
