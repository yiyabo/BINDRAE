import unittest

import numpy as np

from scripts.analyze_md_consensus_residual_structure import (
    endpoint_envelope,
    low_rank_metrics,
    parse_ranks,
)


class MdConsensusResidualStructureTest(unittest.TestCase):
    def test_rank_one_path_is_identified(self):
        temporal = np.array([0.0, 1.0, 2.0, 1.0])
        spatial = np.array([[1.0, 2.0], [-1.0, 0.5]])
        values = temporal[:, None, None] * spatial[None, :, :]
        weights = np.ones_like(values)
        metrics = low_rank_metrics(
            values,
            weights,
            np.ones(temporal.size),
            ranks=(1, 2, 4),
        )
        self.assertAlmostEqual(metrics["rank_1_explained_energy"], 1.0)
        self.assertAlmostEqual(metrics["effective_rank"], 1.0)
        self.assertEqual(metrics["rank_90"], 1)

    def test_two_independent_modes_need_rank_two(self):
        values = np.zeros((2, 2, 1), dtype=float)
        values[0, 0, 0] = 1.0
        values[1, 1, 0] = 1.0
        metrics = low_rank_metrics(
            values,
            np.ones_like(values),
            np.ones(2),
            ranks=(1, 2),
        )
        self.assertAlmostEqual(metrics["rank_1_explained_energy"], 0.5)
        self.assertAlmostEqual(metrics["rank_2_explained_energy"], 1.0)
        self.assertAlmostEqual(metrics["effective_rank"], 2.0)

    def test_zero_weight_components_do_not_add_rank(self):
        values = np.array([[[1.0, 100.0]], [[2.0, -100.0]]])
        weights = np.array([[[1.0, 0.0]], [[1.0, 0.0]]])
        metrics = low_rank_metrics(values, weights, np.ones(2), ranks=(1,))
        self.assertAlmostEqual(metrics["rank_1_explained_energy"], 1.0)
        self.assertEqual(metrics["active_residues"], 1)

    def test_endpoint_envelopes_are_zero_at_boundaries(self):
        t_values = np.array([0.0, 0.5, 1.0])
        np.testing.assert_allclose(endpoint_envelope(t_values, "sin2"), [0.0, 1.0, 0.0])
        np.testing.assert_allclose(endpoint_envelope(t_values, "poly"), [0.0, 1.0, 0.0])

    def test_rank_parser_deduplicates_and_sorts(self):
        self.assertEqual(parse_ranks("8,1,4,4"), (1, 4, 8))
        with self.assertRaises(ValueError):
            parse_ranks("0,2")


if __name__ == "__main__":
    unittest.main()
