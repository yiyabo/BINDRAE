from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

try:
    from rdkit import Chem

    RDKIT_AVAILABLE = True
except ImportError:  # pragma: no cover - exercised only on hosts without RDKit
    RDKIT_AVAILABLE = False

from scripts.repair_triplet_ligand_extraction import (
    COORDINATE_AGREEMENT_TOLERANCE_ANGSTROM,
    iter_sample_dirs,
    load_meta,
    resolve_query_pdb,
)

if RDKIT_AVAILABLE:
    from scripts.repair_triplet_ligand_extraction import (
        conformer_positions,
        subset_molecule,
    )


class ResolveQueryPdbTest(unittest.TestCase):
    def test_a_lowercase_pdb_file_is_found(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "2hdr.pdb").write_text("")

            self.assertEqual(resolve_query_pdb(root, "2HDR"), root / "2hdr.pdb")

    def test_the_ent_naming_is_also_accepted(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "pdb1abc.ent").write_text("")

            self.assertEqual(resolve_query_pdb(root, "1abc"), root / "pdb1abc.ent")

    def test_a_missing_entry_returns_none_rather_than_a_guess(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.assertIsNone(resolve_query_pdb(Path(tmp), "9zzz"))


class IterSampleDirsTest(unittest.TestCase):
    def test_only_directories_holding_a_ligand_are_yielded(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "with-ligand").mkdir()
            (root / "with-ligand" / "ligand.sdf").write_text("")
            (root / "without-ligand").mkdir()
            (root / "stray-file.txt").write_text("")

            found = sorted(p.name for p in iter_sample_dirs(root))

            self.assertEqual(found, ["with-ligand"])


class LoadMetaTest(unittest.TestCase):
    def test_a_missing_or_corrupt_meta_reads_as_empty(self):
        """A broken meta must reject the sample, not crash the pool."""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.assertEqual(load_meta(root), {})
            (root / "meta.json").write_text("{ not json")
            self.assertEqual(load_meta(root), {})

    def test_a_valid_meta_is_returned(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "meta.json").write_text(json.dumps({"query_pdb": "2hdr"}))

            self.assertEqual(load_meta(root)["query_pdb"], "2hdr")


@unittest.skipUnless(RDKIT_AVAILABLE, "RDKit not installed")
class SubsetMoleculeTest(unittest.TestCase):
    def _two_copies(self):
        """Two disconnected ethanols, the second translated 40 A away."""
        mol = Chem.MolFromSmiles("CCO.CCO")
        mol = Chem.AddHs(mol, addCoords=False)
        mol = Chem.RemoveHs(mol)
        editable = Chem.RWMol(mol)
        conformer = Chem.Conformer(editable.GetNumAtoms())
        for index in range(editable.GetNumAtoms()):
            shift = 0.0 if index < 3 else 40.0
            conformer.SetAtomPosition(index, (float(index) + shift, 0.0, 0.0))
        editable.AddConformer(conformer, assignId=True)
        return editable.GetMol()

    def test_removing_atoms_keeps_the_conformer_aligned(self):
        """The repair slices ligand_coords.npy with the same indices, so the
        conformer must follow atom removal or the two files desynchronise."""
        mol = self._two_copies()
        before = conformer_positions(mol)

        subset = subset_molecule(mol, (0, 1, 2))

        self.assertEqual(subset.GetNumAtoms(), 3)
        np.testing.assert_allclose(conformer_positions(subset), before[:3], atol=1e-9)

    def test_the_second_copy_can_be_the_one_kept(self):
        mol = self._two_copies()
        before = conformer_positions(mol)

        subset = subset_molecule(mol, (3, 4, 5))

        np.testing.assert_allclose(conformer_positions(subset), before[3:], atol=1e-9)

    def test_atom_order_and_elements_are_preserved(self):
        mol = self._two_copies()
        expected = [mol.GetAtomWithIdx(i).GetSymbol() for i in (0, 1, 2)]

        subset = subset_molecule(mol, (0, 1, 2))

        self.assertEqual(
            [a.GetSymbol() for a in subset.GetAtoms()], expected
        )

    def test_bond_orders_survive_the_subset(self):
        """Part of the corpus already carries CCD-reconstructed bond orders;
        subsetting must not silently discard them."""
        mol = Chem.MolFromSmiles("C=CO.C=CO")
        editable = Chem.RWMol(mol)
        conformer = Chem.Conformer(editable.GetNumAtoms())
        for index in range(editable.GetNumAtoms()):
            conformer.SetAtomPosition(index, (float(index), 0.0, 0.0))
        editable.AddConformer(conformer, assignId=True)
        mol = editable.GetMol()

        subset = subset_molecule(mol, (0, 1, 2))

        orders = sorted(str(b.GetBondType()) for b in subset.GetBonds())
        self.assertIn("DOUBLE", orders)

    def test_keeping_every_atom_is_a_no_op(self):
        mol = self._two_copies()

        subset = subset_molecule(mol, tuple(range(mol.GetNumAtoms())))

        self.assertEqual(subset.GetNumAtoms(), mol.GetNumAtoms())
        np.testing.assert_allclose(
            conformer_positions(subset), conformer_positions(mol), atol=1e-12
        )


class ToleranceTest(unittest.TestCase):
    def test_the_coordinate_gate_is_looser_than_float32_but_still_strict(self):
        """float32 round-trip on ~100 A coordinates lands near 1e-5 A."""
        self.assertGreater(COORDINATE_AGREEMENT_TOLERANCE_ANGSTROM, 1e-4)
        self.assertLess(COORDINATE_AGREEMENT_TOLERANCE_ANGSTROM, 1e-2)


if __name__ == "__main__":
    unittest.main()
