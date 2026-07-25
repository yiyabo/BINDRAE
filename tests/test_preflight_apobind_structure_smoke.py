from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path

import numpy as np


SCRIPT_PATH = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "preflight_apobind_structure_smoke.py"
)
SPEC = importlib.util.spec_from_file_location(
    "preflight_apobind_structure_smoke", SCRIPT_PATH
)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def residue(letter: str, author_id: int) -> dict:
    return {
        "one_letter": letter,
        "author_sequence_id": author_id,
        "heavy_xyz": np.asarray([[float(author_id), 0.0, 0.0]]),
    }


class PreflightApobindStructureSmokeTest(unittest.TestCase):
    def test_site_signature_resolves_author_numbering(self):
        residues = [residue("A", 10), residue("G", 20), residue("S", 30)]

        result = MODULE.resolve_binding_site_signature(
            residues, ["10", "30"], ["A", "S"]
        )

        self.assertTrue(result["eligible"])
        self.assertEqual(result["scheme"], "author_sequence_id")
        self.assertEqual(result["chain_indices"], [0, 2])
        self.assertEqual(result["signature_match_fraction"], 1.0)

    def test_site_signature_accepts_equivalent_author_and_one_based_mapping(self):
        residues = [residue("A", 1), residue("G", 2), residue("S", 3)]

        result = MODULE.resolve_binding_site_signature(
            residues, ["1", "3"], ["A", "S"]
        )

        self.assertTrue(result["eligible"])
        self.assertEqual(result["chain_indices"], [0, 2])
        self.assertIn("author_sequence_id", result["equivalent_schemes"])
        self.assertIn("ordinal_one_based", result["equivalent_schemes"])

    def test_site_signature_skips_paired_apobind_alignment_gaps(self):
        residues = [residue("A", 10), residue("G", 20), residue("S", 30)]

        result = MODULE.resolve_binding_site_signature(
            residues, ["10", "-", "30"], ["A", "-", "S"]
        )

        self.assertTrue(result["eligible"])
        self.assertEqual(result["chain_indices"], [0, 2])
        self.assertEqual(result["source_gap_count"], 1)
        self.assertEqual(result["coverage"], 1.0)

    def test_site_signature_rejects_one_sided_apobind_gap(self):
        residues = [residue("A", 10), residue("S", 30)]

        result = MODULE.resolve_binding_site_signature(
            residues, ["10", "-"], ["A", "S"]
        )

        self.assertFalse(result["eligible"])
        self.assertEqual(result["reason"], "inconsistent_binding_site_gap")

    def test_site_signature_rejects_ambiguous_numbering(self):
        residues = [
            residue("A", 10),
            residue("A", 11),
            residue("A", 1),
            residue("A", 2),
        ]

        result = MODULE.resolve_binding_site_signature(residues, ["1"], ["A"])

        self.assertFalse(result["eligible"])
        self.assertEqual(result["reason"], "ambiguous_binding_site_index_semantics")

    def test_unique_site_ligand_is_not_selected_by_distance_when_ambiguous(self):
        candidates = [
            {
                "chain_id": "A",
                "resname": "L1",
                "author_sequence_id": 1,
                "minimum_site_distance_angstrom": 1.0,
                "contact_site_residues": 4,
                "plausible_site_ligand": True,
            },
            {
                "chain_id": "A",
                "resname": "L2",
                "author_sequence_id": 2,
                "minimum_site_distance_angstrom": 2.0,
                "contact_site_residues": 2,
                "plausible_site_ligand": True,
            },
        ]

        result = MODULE.choose_unique_site_ligand(candidates)

        self.assertFalse(result["eligible"])
        self.assertEqual(result["reason"], "ambiguous_multiple_site_ligands")
        self.assertEqual(len(result["plausible_candidates"]), 2)

    @unittest.skipUnless(importlib.util.find_spec("Bio"), "Biopython not installed")
    def test_mmcif_component_type_distinguishes_polymer_from_nonpolymer(self):
        from Bio.PDB.Atom import Atom
        from Bio.PDB.Chain import Chain
        from Bio.PDB.Model import Model
        from Bio.PDB.Residue import Residue

        model = Model(0)
        chain = Chain("B")
        model.add(chain)
        modified = Residue(("H_TPO", 10, " "), "TPO", " ")
        modified.add(
            Atom(
                "CA",
                np.asarray([0.0, 0.0, 0.0]),
                1.0,
                1.0,
                " ",
                " CA ",
                1,
                element="C",
            )
        )
        chain.add(modified)
        nonpolymer = Residue(("H_SAH", 11, " "), "SAH", " ")
        nonpolymer.add(
            Atom(
                "C1",
                np.asarray([0.0, 0.0, 0.5]),
                1.0,
                1.0,
                " ",
                " C1 ",
                2,
                element="C",
            )
        )
        chain.add(nonpolymer)
        site = [{"heavy_xyz": np.asarray([[0.0, 0.0, 1.0]])}]

        candidates = MODULE.enumerate_hetero_candidates(
            model,
            site,
            component_types={
                "TPO": "L-peptide linking",
                "SAH": "non-polymer",
            },
            min_heavy_atoms=1,
            max_heavy_atoms=120,
            site_distance=4.5,
        )

        self.assertEqual(len(candidates), 1)
        self.assertEqual(candidates[0]["resname"], "SAH")
        self.assertEqual(candidates[0]["chem_comp_type"], "non-polymer")

    def test_site_mapping_fraction_uses_endpoint_alignment(self):
        fraction = MODULE._site_mapping_fraction(
            [1, 2, 3], [5, 6, 9], [(1, 5), (2, 6), (3, 7)]
        )
        self.assertAlmostEqual(fraction, 2 / 3)


if __name__ == "__main__":
    unittest.main()
