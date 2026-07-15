import math
import unittest

import numpy as np

from scripts.evaluate_stage2_md_reference_paths import (
    aggregate_records,
    distance_event_times,
    interpolate_time_series,
    monotone_crossing_times,
    ordering_accuracy,
    spearman_correlation,
)


class MdReferenceEvaluatorTest(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
