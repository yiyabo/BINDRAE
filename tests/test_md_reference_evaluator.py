import math
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch

import scripts.evaluate_stage2_md_reference_paths as evaluator

from scripts.evaluate_stage2_md_reference_paths import (
    aggregate_records,
    distance_event_times,
    interpolate_time_series,
    monotone_crossing_times,
    ordering_accuracy,
    spearman_correlation,
)


class MdReferenceEvaluatorTest(unittest.TestCase):
    def test_physical_normal_path_reuses_learned_tau(self):
        sentinel = [torch.tensor([1.0])]
        args = SimpleNamespace(path_parameterization="phase_physical_normal_v1")
        config = SimpleNamespace(
            phase_residual_tau_mode="learned",
            phase_residual_bridge_mode="cartesian_backbone",
            time_warp_logit_scale=1.0,
            time_warp_rate_eps=1e-3,
            time_warp_rate_clip=10.0,
        )
        batch = SimpleNamespace(node_mask=torch.ones(1, 2, dtype=torch.bool))
        with patch.object(
            evaluator.base,
            "phase_residual_tau_values",
            return_value=sentinel,
        ) as tau_values:
            result = evaluator.predicted_tau_values(
                args,
                config,
                object(),
                batch,
                object(),
                object(),
                4,
                None,
                None,
            )
        self.assertIs(result, sentinel)
        tau_values.assert_called_once()

    def test_time_interpolation_preserves_endpoints_and_midpoint(self):
        source_t = np.array([0.0, 0.5, 1.0])
        values = np.array([[0.0, 0.0], [1.0, 2.0], [2.0, 4.0]])
        target_t = np.array([0.0, 0.25, 0.75, 1.0])
        result = interpolate_time_series(source_t, values, target_t)
        expected = np.array([[0.0, 0.0], [0.5, 1.0], [1.5, 3.0], [2.0, 4.0]])
        np.testing.assert_allclose(result, expected)

    def test_phase_crossing_and_order_metrics(self):
        times = np.array([0.0, 0.5, 1.0])
        tau = np.array(
            [
                [0.0, 0.0, 0.0],
                [0.8, 0.5, 0.2],
                [1.0, 1.0, 1.0],
            ]
        )
        crossing = monotone_crossing_times(tau, times)
        self.assertLess(crossing[0], crossing[1])
        self.assertLess(crossing[1], crossing[2])
        self.assertAlmostEqual(spearman_correlation(crossing, crossing), 1.0)
        self.assertAlmostEqual(ordering_accuracy(crossing, crossing, 0.01), 1.0)

    def test_contact_event_crossings_are_direction_aware(self):
        times = np.array([0.0, 0.5, 1.0])
        distances = np.array(
            [
                [6.0, 3.0],
                [4.0, 5.0],
                [3.0, 6.0],
            ]
        )
        formed = distance_event_times(distances, times, 4.5, "formed")
        released = distance_event_times(distances, times, 4.5, "released")
        self.assertAlmostEqual(formed[0], 0.375)
        self.assertTrue(math.isnan(formed[1]))
        self.assertTrue(math.isnan(released[0]))
        self.assertAlmostEqual(released[1], 0.375)

    def test_phase_dynamics_detects_pause_and_backtracking(self):
        times = np.linspace(0.0, 1.0, 5)
        target = np.array(
            [
                [0.0, 0.0],
                [0.25, 0.0],
                [0.50, 0.50],
                [0.75, 0.75],
                [1.00, 1.00],
            ]
        )
        predicted = np.array(
            [
                [0.0, 0.0],
                [0.4, 0.0],
                [0.3, 0.5],
                [0.7, 0.75],
                [1.0, 1.0],
            ]
        )
        arrays = {
            "tau": target,
            "phase_confidence": np.ones_like(target),
            "node_mask": np.ones(2, dtype=bool),
            "active_mask": np.ones(2, dtype=bool),
        }
        metrics = evaluator.phase_dynamics_metrics(
            predicted,
            arrays,
            times,
            min_confidence=0.05,
            pause_rate_threshold=0.25,
            backtrack_rate_threshold=0.05,
        )
        self.assertGreater(metrics["phase_pred_backtrack_interval_fraction"], 0.0)
        self.assertEqual(metrics["phase_target_backtrack_interval_fraction"], 0.0)
        self.assertAlmostEqual(
            metrics["phase_pred_nonmonotone_residue_fraction"], 0.5
        )
        self.assertGreater(metrics["phase_target_pause_interval_fraction"], 0.0)
        self.assertEqual(metrics["phase_endpoint_max_error"], 0.0)

    def test_cummax_phase_postprocess_removes_backtracking_and_keeps_endpoints(self):
        tau = np.array(
            [
                [0.0, 0.0],
                [0.4, 0.2],
                [0.3, 0.6],
                [0.8, 0.7],
                [1.0, 1.0],
            ]
        )
        result = evaluator.postprocess_phase_tau(tau, "cummax")
        self.assertTrue(np.all(np.diff(result, axis=0) >= 0.0))
        np.testing.assert_array_equal(result[0], np.zeros(2))
        np.testing.assert_array_equal(result[-1], np.ones(2))
        self.assertAlmostEqual(result[2, 0], 0.4)

    def test_system_macro_does_not_overweight_replica_count(self):
        records = [
            {
                "sample_id": "system_a",
                "metrics": {
                    "error": 0.0,
                    "contact_event_count": 1,
                    "contact_event_matched": 1,
                },
            },
            {
                "sample_id": "system_a",
                "metrics": {
                    "error": 2.0,
                    "contact_event_count": 1,
                    "contact_event_matched": 1,
                },
            },
            {
                "sample_id": "system_b",
                "metrics": {
                    "error": 5.0,
                    "contact_event_count": 2,
                    "contact_event_matched": 0,
                },
            },
        ]
        summary = aggregate_records(records)
        self.assertAlmostEqual(summary["replica_macro"]["error"], 7.0 / 3.0)
        self.assertAlmostEqual(summary["system_macro"]["error"], 3.0)
        self.assertAlmostEqual(summary["pooled"]["contact_event_coverage"], 0.5)

    def test_aggregate_records_tolerates_sparse_optional_metrics(self):
        records = [
            {
                "sample_id": "system_a",
                "metrics": {
                    "always": 1.0,
                    "optional": 3.0,
                    "contact_event_count": 0,
                    "contact_event_matched": 0,
                },
            },
            {
                "sample_id": "system_b",
                "metrics": {
                    "always": 2.0,
                    "contact_event_count": 0,
                    "contact_event_matched": 0,
                },
            },
        ]
        summary = aggregate_records(records)
        self.assertAlmostEqual(summary["replica_macro"]["always"], 1.5)
        self.assertAlmostEqual(summary["replica_macro"]["optional"], 3.0)


if __name__ == "__main__":
    unittest.main()
