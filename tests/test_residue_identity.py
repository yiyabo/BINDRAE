import unittest

import numpy as np

from src.data.residue_identity import (
    is_standard_residue,
    residue_identity_hash,
    residue_keys_from_array,
    residue_keys_to_array,
    scatter_by_residue_keys,
)


class ResidueIdentityTest(unittest.TestCase):
    def test_standard_amino_acid_hetatm_matches_esm_contract(self):
        class FakeResidue:
            def get_id(self):
                return ("H_LEU", 1301, " ")

            def get_resname(self):
                return "LEU"

        self.assertTrue(is_standard_residue(FakeResidue()))

    def test_round_trip_preserves_insertion_codes(self):
        keys = [("A", 59, ""), ("A", 60, "A"), ("A", 60, "B"), ("A", 61, "")]
        self.assertEqual(residue_keys_from_array(residue_keys_to_array(keys)), keys)
        self.assertEqual(residue_identity_hash(keys), residue_identity_hash(list(keys)))

    def test_middle_missing_residues_keep_apo_positions(self):
        target = [("A", 399, ""), ("A", 400, ""), ("A", 401, ""), ("A", 402, "")]
        source = [("B", 401, ""), ("B", 402, "")]
        values = np.asarray([10.0, 20.0], dtype=np.float32)

        aligned, present = scatter_by_residue_keys(
            values,
            source,
            target,
            label="holo test",
        )

        np.testing.assert_array_equal(aligned, np.asarray([0.0, 0.0, 10.0, 20.0]))
        np.testing.assert_array_equal(present, np.asarray([False, False, True, True]))

    def test_insertion_codes_do_not_collide(self):
        keys = [("A", 60, ""), ("A", 60, "A"), ("A", 60, "B")]
        values = np.asarray([1, 2, 3], dtype=np.int64)
        aligned, present = scatter_by_residue_keys(values, keys, keys, label="insertions")
        np.testing.assert_array_equal(aligned, values)
        self.assertTrue(present.all())

    def test_duplicate_key_is_rejected(self):
        keys = [("A", 60, "A"), ("A", 60, "A")]
        with self.assertRaisesRegex(ValueError, "Duplicate residue identity"):
            scatter_by_residue_keys(
                np.asarray([1, 2]),
                keys,
                [("A", 60, "A")],
                label="duplicate",
            )

    def test_multichain_alignment_does_not_alias_chain_ids(self):
        target = [("A", 1, ""), ("B", 1, "")]
        source = [("C", 1, "")]
        aligned, present = scatter_by_residue_keys(
            np.asarray([5.0]),
            source,
            target,
            label="multichain",
        )
        np.testing.assert_array_equal(aligned, np.zeros(2))
        self.assertFalse(present.any())


if __name__ == "__main__":
    unittest.main()
