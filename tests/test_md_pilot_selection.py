from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from src.data.md_pilot_selection import (
    candidate_to_transition_record,
    describe_ligand,
    kabsch_align,
    read_sample_ids,
    screen_sample,
    select_diverse_candidates,
)
from src.data.md_transition_manifest import validate_transition_record


class MDPilotSelectionTest(unittest.TestCase):
    @staticmethod
    def _write_ca_pdb(path: Path, residue_names: list[str]) -> None:
        lines = []
        for index, residue_name in enumerate(residue_names, start=1):
            x = float(index)
            y = float((index % 3) * 0.7)
            z = float((index % 5) * 0.4)
            lines.append(
                f"ATOM  {index:5d}  CA  {residue_name:>3s} A{index:4d}    "
                f"{x:8.3f}{y:8.3f}{z:8.3f}  1.00 20.00           C\n"
            )
        path.write_text("".join(lines) + "END\n")

    def test_kabsch_removes_rigid_transform(self):
        points = np.asarray(
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 3.0]]
        )
        rotation = np.asarray([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
        target = points @ rotation + np.asarray([4.0, -2.0, 7.0])
        aligned, rmsd = kabsch_align(points, target)
        self.assertLess(rmsd, 1e-10)
        np.testing.assert_allclose(aligned, target, atol=1e-10)

    def test_manifest_sampling_is_seeded_and_unique(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "samples.txt"
            path.write_text("a\nb\na\nc\nd\n")
            first = read_sample_ids(path, scan_limit=3, seed=9)
            second = read_sample_ids(path, scan_limit=3, seed=9)
        self.assertEqual(first, second)
        self.assertEqual(len(first), 3)
        self.assertEqual(len(set(first)), 3)

    def test_screen_maps_a_small_endpoint_residue_count_difference(self):
        reference_names = [
            "ALA", "GLY", "SER", "THR", "VAL", "LEU", "ILE", "ASN", "GLN", "ASP",
            "GLU", "LYS", "ARG", "HIS", "PHE", "TYR", "TRP", "CYS", "MET", "PRO",
        ]
        query_names = [*reference_names[:10], "ALA", *reference_names[10:]]
        with tempfile.TemporaryDirectory() as temporary:
            data_dir = Path(temporary)
            sample_dir = data_dir / "samples" / "mapped-A-LIG-1"
            sample_dir.mkdir(parents=True)
            self._write_ca_pdb(sample_dir / "apo.pdb", reference_names)
            self._write_ca_pdb(sample_dir / "holo.pdb", query_names)
            (sample_dir / "ligand.sdf").write_text("test\n")
            (sample_dir / "meta.json").write_text(
                json.dumps(
                    {
                        "apo_pdb": "1abc",
                        "holo_pdb": "2def",
                        "ligand_resname": "LIG",
                    }
                )
            )
            config = {
                "data_dir": str(data_dir),
                "excluded_resnames": [],
                "min_residues": 10,
                "max_residues": 500,
                "min_sequence_identity": 0.95,
                "min_residue_mapping_fraction": 0.95,
                "min_heavy_atoms": 1,
                "max_heavy_atoms": 100,
                "max_abs_charge": 2,
                "pocket_radius": 100.0,
                "contact_radius": 8.0,
                "min_pocket_residues": 1,
                "moving_threshold": 1.0,
                "min_global_rmsd": 0.0,
                "min_pocket_rmsd": 0.0,
                "min_max_displacement": 0.0,
                "max_global_rmsd": 100.0,
                "max_pocket_rmsd": 100.0,
                "max_max_displacement": 100.0,
            }
            ligand = {
                "eligible": True,
                "reason": "ok",
                "heavy_atoms": 10,
                "formal_charge": 0,
                "contains_metal": False,
                "fragment_count": 1,
                "organic_copy_count": 1,
                "extra_nonorganic_fragments": 0,
                "canonical_smiles": "CC",
                "inchikey": "TEST",
                "ligand_xyz": np.asarray([[1.0, 0.0, 0.0]]),
            }
            with patch("src.data.md_pilot_selection.describe_ligand", return_value=ligand):
                result = screen_sample("mapped-A-LIG-1", config)

        self.assertTrue(result["eligible"])
        self.assertEqual(result["mapped_residues"], 20)
        self.assertAlmostEqual(result["residue_mapping_fraction"], 20 / 21)
        self.assertEqual(result["reason"], "ok")

    def test_selection_balances_categories_and_deduplicates_endpoints(self):
        rows = [
            {
                "sample_id": "a1",
                "eligible": True,
                "motion_score": 0.9,
                "motion_category": "domain_motion",
                "apo_pdb": "1aaa",
                "holo_pdb": "2aaa",
                "ligand_inchikey": "scaffold-a",
            },
            {
                "sample_id": "a2",
                "eligible": True,
                "motion_score": 0.8,
                "motion_category": "domain_motion",
                "apo_pdb": "1aaa",
                "holo_pdb": "2aaa",
                "ligand_inchikey": "scaffold-b",
            },
            {
                "sample_id": "b1",
                "eligible": True,
                "motion_score": 0.7,
                "motion_category": "contact_switch",
                "apo_pdb": "1bbb",
                "holo_pdb": "2bbb",
                "ligand_inchikey": "scaffold-b",
            },
            {
                "sample_id": "c1",
                "eligible": True,
                "motion_score": 0.6,
                "motion_category": "local_pocket",
                "apo_pdb": "1ccc",
                "holo_pdb": "2ccc",
                "ligand_inchikey": "scaffold-c",
            },
        ]
        selected = select_diverse_candidates(rows, 3)
        self.assertEqual({row["sample_id"] for row in selected}, {"a1", "b1", "c1"})
        self.assertEqual([row["selection_rank"] for row in selected], [1, 2, 3])

    def test_selected_candidate_converts_to_valid_metadata_record(self):
        row = {
            "sample_id": "1abc-A-LIG-1",
            "sample_dir": "processed_data/triplets/samples/1abc-A-LIG-1",
            "query_pdb": "1abc",
            "apo_pdb": "2abc",
            "holo_pdb": "3abc",
            "apo_chain": "A",
            "holo_chain": "B",
            "ligand_resname": "LIG",
            "ligand_inchikey": "ABCDEFGHIJKLMN-ABCDEFGHIJ-A",
            "ligand_canonical_smiles": "CCO",
            "sequence_identity": 1.0,
        }
        record = candidate_to_transition_record(row)
        errors = [issue for issue in validate_transition_record(record) if issue.severity == "error"]
        self.assertEqual(errors, [])
        self.assertFalse(record["usage"]["phase_supervision"])
        self.assertFalse(record["evidence"]["contains_endpoint_transition"])

    def test_duplicate_identical_ligands_are_one_chemical_component(self):
        try:
            from rdkit import Chem
        except ImportError:
            self.skipTest("RDKit is only required in the BINDRAE-MD environment")
        molecule = Chem.MolFromSmiles("CCO.CCO")
        molecule = Chem.AddHs(molecule)
        from rdkit.Chem import AllChem

        self.assertEqual(AllChem.EmbedMolecule(molecule, randomSeed=7), 0)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "duplicate.sdf"
            writer = Chem.SDWriter(str(path))
            writer.write(molecule)
            writer.close()
            result = describe_ligand(path, min_heavy_atoms=2, max_heavy_atoms=20, max_abs_charge=2)
        self.assertTrue(result["eligible"])
        self.assertEqual(result["organic_copy_count"], 2)
        self.assertEqual(result["heavy_atoms"], 3)


if __name__ == "__main__":
    unittest.main()
