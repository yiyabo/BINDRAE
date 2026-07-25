import unittest

import numpy as np

from scripts.evaluate_md_phase_normal_oracle_ladder import (
    bootstrap_mean_ci,
    decision_hint,
    select_route_oracle,
    select_train_medoid,
    weighted_consensus_prediction,
    weighted_prediction_metrics,
)


class MdPhaseNormalOracleLadderTest(unittest.TestCase):
    def test_consensus_requires_configured_replica_support(self):
        values = np.array(
            [
                [[[2.0, 4.0]]],
                [[[4.0, 8.0]]],
            ]
        )
        weights = np.array(
            [
                [[[1.0, 1.0]]],
                [[[1.0, 0.0]]],
            ]
        )
        prediction, available = weighted_consensus_prediction(
            values,
            weights,
            min_support_count=2,
            min_support_fraction=0.5,
        )
        np.testing.assert_allclose(prediction, [[[3.0, 0.0]]])
        np.testing.assert_array_equal(available, [[[True, False]]])

    def test_missing_prediction_support_is_filled_with_zero_not_dropped(self):
        target = np.array([1.0, 2.0])
        prediction = np.array([1.0, 100.0])
        weights = np.ones(2)
        metrics = weighted_prediction_metrics(
            prediction,
            target,
            weights,
            available=np.array([True, False]),
        )
        self.assertAlmostEqual(metrics["mse"], 2.0)
        self.assertAlmostEqual(metrics["available_weight_fraction"], 0.5)
        self.assertAlmostEqual(metrics["relative_mse_reduction_vs_zero"], 0.2)

    def test_medoid_selection_does_not_use_heldout_target(self):
        values = [
            np.array([0.0]),
            np.array([0.2]),
            np.array([3.0]),
        ]
        weights = [np.ones(1) for _ in values]
        medoid_index, scores = select_train_medoid(values, weights)
        self.assertEqual(medoid_index, 1)
        self.assertLess(scores[medoid_index], scores[2])

        oracle_index, _ = select_route_oracle(
            values,
            weights,
            heldout_value=np.array([3.1]),
            heldout_weight=np.ones(1),
        )
        self.assertEqual(oracle_index, 2)

    def test_route_oracle_can_recover_opposing_route_while_consensus_cancels(self):
        candidates = np.array([[[[1.0]]], [[[-1.0]]]])
        weights = np.ones_like(candidates)
        consensus, available = weighted_consensus_prediction(
            candidates,
            weights,
            min_support_count=2,
            min_support_fraction=0.5,
        )
        target = np.array([[[1.0]]])
        target_weight = np.ones_like(target)
        consensus_metrics = weighted_prediction_metrics(
            consensus,
            target,
            target_weight,
            available=available,
        )
        oracle_index, _ = select_route_oracle(
            list(candidates),
            list(weights),
            target,
            target_weight,
        )
        oracle_metrics = weighted_prediction_metrics(
            candidates[oracle_index],
            target,
            target_weight,
            available=weights[oracle_index] > 0.0,
        )
        self.assertAlmostEqual(consensus_metrics["mse"], 1.0)
        self.assertAlmostEqual(oracle_metrics["mse"], 0.0)

    def test_decision_hint_separates_deterministic_and_route_evidence(self):
        comparisons = {
            "loo_consensus_vs_zero": {
                "ci95_low": -0.01,
                "relative_macro_mse_reduction": 0.01,
            },
            "route_oracle_vs_loo_consensus": {
                "ci95_low": 0.1,
                "relative_macro_mse_reduction": 0.2,
            },
        }
        decision = decision_hint(comparisons, 0.05)
        self.assertEqual(decision["direction"], "stochastic_path_latent_candidate")
        self.assertFalse(decision["deterministic_consensus_supported"])
        self.assertTrue(decision["route_oracle_supported"])

    def test_bootstrap_reports_higher_is_better_gain(self):
        result = bootstrap_mean_ci([0.1, 0.2, 0.3], samples=1000, seed=7)
        self.assertAlmostEqual(result["mean"], 0.2)
        self.assertGreater(result["ci95_low"], 0.0)
        self.assertEqual(result["systems_candidate_better_fraction"], 1.0)


if __name__ == "__main__":
    unittest.main()
