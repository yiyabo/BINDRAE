from __future__ import annotations

import unittest

from src.data.residue_alignment import (
    align_residue_names,
    triple_exact_residue_alignment,
)


class ResidueAlignmentTest(unittest.TestCase):
    def test_global_alignment_retains_exact_matches_across_an_insertion(self):
        alignment = align_residue_names(
            ["ALA", "GLY", "SER", "THR"],
            ["ALA", "GLY", "ASP", "SER", "THR"],
        )

        self.assertEqual(alignment.exact_pairs, ((0, 0), (1, 1), (2, 3), (3, 4)))
        self.assertEqual(alignment.sequence_identity, 1.0)
        self.assertEqual(alignment.reference_mapping_fraction, 1.0)
        self.assertAlmostEqual(alignment.query_mapping_fraction, 0.8)
        self.assertAlmostEqual(alignment.symmetric_mapping_fraction, 0.8)

    def test_aliases_match_prepared_topology_residue_names(self):
        alignment = align_residue_names(["HIS", "MET"], ["HIE", "MSE"])

        self.assertEqual(alignment.exact_pairs, ((0, 0), (1, 1)))
        self.assertEqual(alignment.symmetric_mapping_fraction, 1.0)

    def test_triple_alignment_uses_one_pivot_correspondence(self):
        triples = triple_exact_residue_alignment(
            ["ALA", "GLY", "SER", "THR"],
            ["ALA", "GLY", "ASP", "SER", "THR"],
            ["ALA", "GLY", "ASP", "SER", "THR"],
        )

        self.assertEqual(triples, ((0, 0, 0), (1, 1, 1), (2, 3, 3), (3, 4, 4)))


if __name__ == "__main__":
    unittest.main()
