import unittest

import torch

from scripts.run_controlled_manifold_benchmark import (
    build_path,
    make_dataset,
    path_metrics,
)


class ControlledManifoldBenchmarkTest(unittest.TestCase):
    def test_hidden_route_pairs_have_identical_inputs_and_opposite_targets(self):
        dataset = make_dataset(3, 5, seed=11)
        for pair in range(3):
            left = 2 * pair
            right = left + 1
            torch.testing.assert_close(
                dataset.features_hidden[left], dataset.features_hidden[right]
            )
            torch.testing.assert_close(
                dataset.residual_coefficient[left],
                -dataset.residual_coefficient[right],
            )
            self.assertFalse(
                torch.equal(
                    dataset.features_observed[left],
                    dataset.features_observed[right],
                )
            )

    def test_oracle_path_is_endpoint_exact_and_metric_normal(self):
        dataset = make_dataset(2, 4, seed=13)
        path = build_path(
            dataset,
            dataset.phase_coefficient,
            dataset.residual_coefficient,
            n_steps=12,
        )
        torch.testing.assert_close(path["translation"][:, 0], dataset.translation_apo)
        torch.testing.assert_close(
            path["translation"][:, -1],
            dataset.translation_apo + dataset.translation_delta,
        )
        self.assertLess(
            float(
                (dataset.translation_delta * dataset.translation_normal)
                .sum(dim=-1)
                .abs()
                .max()
            ),
            1e-5,
        )
        self.assertLess(
            float(
                (dataset.rotation_delta * dataset.rotation_normal)
                .sum(dim=-1)
                .abs()
                .max()
            ),
            1e-5,
        )
        self.assertLess(
            float(
                (dataset.chi_delta * dataset.chi_normal)
                .sum(dim=-1)
                .abs()
                .max()
            ),
            1e-5,
        )

    def test_oracle_mode_has_zero_path_error(self):
        dataset = make_dataset(2, 4, seed=17)
        metrics = path_metrics(
            dataset,
            dataset.phase_coefficient,
            dataset.residual_coefficient,
            n_steps=12,
        )
        self.assertLess(metrics["product_path_rmse"], 1e-6)
        self.assertLess(metrics["endpoint_max_error"], 1e-5)

    def test_phase_coefficients_shift_midpoint_events_and_remain_monotone(self):
        dataset = make_dataset(2, 4, seed=19)
        path = build_path(
            dataset,
            dataset.phase_coefficient,
            torch.zeros_like(dataset.residual_coefficient),
            n_steps=40,
        )
        self.assertTrue(bool((torch.diff(path["tau"], dim=1) > 0.0).all()))
        midpoint = path["tau"][:, 20]
        expected = 0.5 + 0.25 * dataset.phase_coefficient
        torch.testing.assert_close(midpoint, expected)


if __name__ == "__main__":
    unittest.main()
