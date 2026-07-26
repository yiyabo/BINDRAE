from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

try:
    from rdkit import Chem
    from rdkit.Chem import AllChem

    RDKIT_AVAILABLE = True
except ImportError:  # pragma: no cover - exercised only on hosts without RDKit
    RDKIT_AVAILABLE = False

from src.data.ligand_bond_orders import (
    LigandBondOrderError,
    describe_ligand,
    find_sentinel_hydrides,
    reconstruct_bond_orders,
    reconstruct_ligand_sdf,
    unexpected_hydrides,
)


#: Methyl phosphate: the minimal molecule that reproduces the production defect.
#: Written as ``CO[P](=O)([O-])O`` the phosphorus has valence 5; stripped of bond
#: orders and formal charges it has valence 4, which RDKit completes with a
#: hydride.
PHOSPHATE_SMILES = "CO[P](=O)([O-])O"
PHOSPHATE_RESNAME = "MPO"


def _template_from_smiles(smiles: str) -> "Chem.Mol":
    molecule = Chem.MolFromSmiles(smiles)
    assert molecule is not None
    AllChem.Compute2DCoords(molecule)
    return molecule


def _degrade_to_connectivity_only(molecule: "Chem.Mol") -> "Chem.Mol":
    """Emulate ``MolFromPDBBlock``: keep the graph, drop orders and charges."""

    flat = Chem.RWMol(Chem.RemoveHs(Chem.Mol(molecule)))
    for bond in flat.GetBonds():
        bond.SetBondType(Chem.BondType.SINGLE)
        bond.SetIsAromatic(False)
    for atom in flat.GetAtoms():
        atom.SetFormalCharge(0)
        atom.SetIsAromatic(False)
        atom.SetNoImplicit(False)
        atom.SetNumExplicitHs(0)
    degraded = flat.GetMol()
    Chem.SanitizeMol(degraded)
    return degraded


def _write_sdf(molecule: "Chem.Mol", path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    writer = Chem.SDWriter(str(path))
    writer.write(molecule)
    writer.close()
    return path


@unittest.skipUnless(RDKIT_AVAILABLE, "RDKit is not installed in this environment")
class LigandBondOrderReconstructionTest(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        self.ccd_dir = self.root / "ccd"
        self.template = _template_from_smiles(PHOSPHATE_SMILES)
        _write_sdf(self.template, self.ccd_dir / f"{PHOSPHATE_RESNAME}.sdf")
        self.degraded = _degrade_to_connectivity_only(self.template)

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def test_connectivity_only_ligand_reproduces_the_phosphorus_hydride(self):
        """Guards the premise: without this defect the repair would be pointless."""

        summary = describe_ligand(self.degraded)

        self.assertEqual(summary["formal_charge"], 0)
        self.assertEqual(set(summary["bond_orders"]), {"SINGLE"})
        self.assertTrue(summary["sentinel_hydrides"])
        self.assertEqual(summary["sentinel_hydrides"][0]["element"], "P")

    def test_reconstruction_restores_orders_charge_and_removes_hydride(self):
        reconstructed, diagnostics = reconstruct_bond_orders(
            self.degraded, Chem.RemoveHs(self.template)
        )

        reference = describe_ligand(Chem.RemoveHs(self.template))
        summary = describe_ligand(reconstructed)
        self.assertEqual(summary["bond_orders"], reference["bond_orders"])
        self.assertEqual(summary["formal_charge"], reference["formal_charge"])
        self.assertEqual(summary["canonical_smiles"], reference["canonical_smiles"])
        self.assertEqual(summary["sentinel_hydrides"], [])
        self.assertGreater(diagnostics["bond_orders_changed"], 0)
        self.assertFalse(diagnostics["partial_observation"])

    def test_reconstruction_preserves_atom_order_and_coordinates(self):
        before_elements = [atom.GetSymbol() for atom in self.degraded.GetAtoms()]
        before_xyz = np.asarray(
            self.degraded.GetConformer().GetPositions(), dtype=np.float64
        )

        reconstructed, _ = reconstruct_bond_orders(
            self.degraded, Chem.RemoveHs(self.template)
        )

        after_elements = [atom.GetSymbol() for atom in reconstructed.GetAtoms()]
        after_xyz = np.asarray(
            reconstructed.GetConformer().GetPositions(), dtype=np.float64
        )
        self.assertEqual(after_elements, before_elements)
        np.testing.assert_allclose(after_xyz, before_xyz, atol=0.0)

    def test_sdf_entry_point_writes_a_strictly_parsable_ligand(self):
        input_sdf = _write_sdf(self.degraded, self.root / "in" / "ligand.sdf")
        output_sdf = self.root / "out" / "ligand.sdf"

        record = reconstruct_ligand_sdf(
            input_sdf=input_sdf,
            resname=PHOSPHATE_RESNAME,
            output_sdf=output_sdf,
            ccd_dir=self.ccd_dir,
        )

        self.assertTrue(output_sdf.is_file())
        self.assertTrue(record["before"]["sentinel_hydrides"])
        self.assertEqual(record["after"]["sentinel_hydrides"], [])
        self.assertEqual(record["roundtrip"]["sentinel_hydrides"], [])
        self.assertEqual(record["after"]["formal_charge"], -1)
        self.assertTrue(record["after"]["formal_charge_matches_template"])
        self.assertEqual(len(record["output_sdf_sha256"]), 64)

        reparsed = Chem.SDMolSupplier(str(output_sdf), removeHs=False, sanitize=True)
        molecule = next(value for value in reparsed if value is not None)
        self.assertEqual(find_sentinel_hydrides(Chem.AddHs(molecule)), [])

    def test_unobserved_phosphate_oxygen_is_rejected_with_its_own_reason(self):
        editable = Chem.RWMol(self.degraded)
        victim = next(
            atom.GetIdx()
            for atom in self.degraded.GetAtoms()
            if atom.GetDegree() == 1
            and any(neighbor.GetSymbol() == "P" for neighbor in atom.GetNeighbors())
        )
        editable.RemoveAtom(int(victim))
        partial = editable.GetMol()
        Chem.SanitizeMol(partial)

        with self.assertRaises(LigandBondOrderError) as caught:
            reconstruct_bond_orders(partial, Chem.RemoveHs(self.template))

        self.assertEqual(caught.exception.reason, "unobserved_neighbor_on_sentinel_element")

    def test_bond_absent_from_the_component_is_rejected(self):
        editable = Chem.RWMol(self.degraded)
        carbon = next(
            atom.GetIdx()
            for atom in self.degraded.GetAtoms()
            if atom.GetSymbol() == "C"
        )
        terminal_oxygen = next(
            atom.GetIdx()
            for atom in self.degraded.GetAtoms()
            if atom.GetSymbol() == "O"
            and atom.GetDegree() == 1
            and self.degraded.GetBondBetweenAtoms(int(carbon), atom.GetIdx()) is None
        )
        editable.AddBond(int(carbon), int(terminal_oxygen), Chem.BondType.SINGLE)
        spurious = editable.GetMol()
        Chem.SanitizeMol(spurious)

        with self.assertRaises(LigandBondOrderError) as caught:
            reconstruct_bond_orders(spurious, Chem.RemoveHs(self.template))

        self.assertIn(
            caught.exception.reason,
            {"substructure_match_failed", "observed_bond_absent_in_template"},
        )

    def test_multiple_copies_of_the_component_are_repaired_per_fragment(self):
        # extract_ligand_from_pdb() collects every same-resname HETATM residue in
        # the chain, so glycans and repeated sugars arrive as one SDF holding N
        # disconnected copies of the component.
        offset = Chem.Mol(self.degraded)
        conformer = offset.GetConformer()
        for index in range(offset.GetNumAtoms()):
            position = conformer.GetAtomPosition(index)
            conformer.SetAtomPosition(index, (position.x + 50.0, position.y, position.z))
        merged = Chem.CombineMols(self.degraded, offset)
        Chem.SanitizeMol(merged)
        self.assertEqual(merged.GetNumAtoms(), 2 * self.degraded.GetNumAtoms())

        reconstructed, diagnostics = reconstruct_bond_orders(
            merged, Chem.RemoveHs(self.template)
        )

        self.assertEqual(diagnostics["component_copies"], 2)
        self.assertEqual(diagnostics["observed_heavy_atoms"], 2 * self.degraded.GetNumAtoms())
        # The fraction is per copy, so a doubled component still reads 1.0.
        self.assertAlmostEqual(diagnostics["observed_heavy_atom_fraction"], 1.0)
        self.assertFalse(diagnostics["partial_observation"])
        summary = describe_ligand(reconstructed)
        self.assertEqual(summary["sentinel_hydrides"], [])
        self.assertEqual(summary["formal_charge"], -2)  # two anionic copies

    def test_covalently_linked_copies_need_residue_groups_not_fragments(self):
        # An oligosaccharide is N components joined by glycosidic bonds: one
        # connected fragment N times the component size.  Splitting on
        # connectivity cannot separate them; the PDB residue numbers can.
        offset = Chem.Mol(self.degraded)
        conformer = offset.GetConformer()
        for index in range(offset.GetNumAtoms()):
            position = conformer.GetAtomPosition(index)
            conformer.SetAtomPosition(index, (position.x + 10.0, position.y, position.z))
        size = self.degraded.GetNumAtoms()
        merged = Chem.RWMol(Chem.CombineMols(self.degraded, offset))
        carbon = next(
            atom.GetIdx() for atom in merged.GetAtoms()
            if atom.GetSymbol() == "C" and atom.GetIdx() < size
        )
        # A glycosidic bond joins a hydroxyl oxygen, not a carbonyl or an anion,
        # so pick the O the component declares as single-bonded and neutral.
        reference = Chem.RemoveHs(self.template)
        hydroxyl = next(
            atom.GetIdx() for atom in reference.GetAtoms()
            if atom.GetSymbol() == "O"
            and atom.GetDegree() == 1
            and atom.GetFormalCharge() == 0
            and atom.GetBonds()[0].GetBondType() == Chem.BondType.SINGLE
        )
        bridging_oxygen = size + hydroxyl
        merged.AddBond(int(carbon), int(bridging_oxygen), Chem.BondType.SINGLE)
        linked = merged.GetMol()
        Chem.SanitizeMol(linked)
        self.assertEqual(len(Chem.GetMolFrags(linked)), 1)  # one fragment, two copies

        with self.assertRaises(LigandBondOrderError) as caught:
            reconstruct_bond_orders(linked, Chem.RemoveHs(self.template))
        self.assertEqual(caught.exception.reason, "substructure_match_failed")

        groups = [tuple(range(size)), tuple(range(size, 2 * size))]
        reconstructed, diagnostics = reconstruct_bond_orders(
            linked, Chem.RemoveHs(self.template), atom_groups=groups
        )

        self.assertEqual(diagnostics["component_copies"], 2)
        self.assertEqual(describe_ligand(reconstructed)["sentinel_hydrides"], [])
        # The glycosidic bond joins two components and is left as observed.
        self.assertIsNotNone(
            reconstructed.GetBondBetweenAtoms(int(carbon), int(bridging_oxygen))
        )

    def test_coverage_threshold_rejects_a_small_ligand_inside_a_large_component(self):
        # G3H (10 heavy atoms) is a subgraph of NAD (44), so partial matching
        # accepts it. Only a per-copy coverage floor can rule that out.
        big = _template_from_smiles("CO[P](=O)([O-])OCCOP(=O)([O-])OCC")
        _write_sdf(big, self.ccd_dir / "BIG.sdf")
        small_sdf = _write_sdf(self.degraded, self.root / "in" / "ligand.sdf")

        with self.assertRaises(LigandBondOrderError) as caught:
            reconstruct_ligand_sdf(
                input_sdf=small_sdf,
                resname="BIG",
                output_sdf=self.root / "out" / "ligand.sdf",
                ccd_dir=self.ccd_dir,
                min_observed_heavy_atom_fraction=0.80,
            )

        self.assertIn(
            caught.exception.reason,
            {
                "observed_heavy_atom_fraction_below_threshold",
                "unobserved_neighbor_on_sentinel_element",
                "substructure_match_failed",
            },
        )

    def test_a_thiol_the_component_declares_is_not_treated_as_a_defect(self):
        # Glutathione's cysteine legitimately carries S-H.  An absolute element
        # blacklist rejects it; the template-relative check must not.
        thiol_template = _template_from_smiles("SCC(N)C(=O)O")
        _write_sdf(thiol_template, self.ccd_dir / "THI.sdf")
        degraded = _degrade_to_connectivity_only(thiol_template)
        self.assertTrue(describe_ligand(degraded)["sentinel_hydrides"])  # S-H is present

        reconstructed, diagnostics = reconstruct_bond_orders(
            degraded, Chem.RemoveHs(thiol_template)
        )

        summary = describe_ligand(reconstructed)
        self.assertTrue(summary["sentinel_hydrides"])  # still there, and correct
        self.assertEqual(
            unexpected_hydrides(reconstructed, diagnostics["expected_hydrogen_counts"]), []
        )

    def test_extra_hydrogen_beyond_the_component_is_still_rejected(self):
        expected = [0] * self.degraded.GetNumAtoms()

        offenders = unexpected_hydrides(self.degraded, expected)

        self.assertTrue(offenders)
        self.assertEqual(offenders[0]["element"], "P")
        self.assertGreater(offenders[0]["hydrogens"], offenders[0]["template_hydrogens"])

    def test_missing_template_is_rejected_when_downloads_are_disabled(self):
        input_sdf = _write_sdf(self.degraded, self.root / "in" / "ligand.sdf")

        with self.assertRaises(LigandBondOrderError) as caught:
            reconstruct_ligand_sdf(
                input_sdf=input_sdf,
                resname="ZZZ",
                output_sdf=self.root / "out" / "ligand.sdf",
                ccd_dir=self.ccd_dir,
                allow_download=False,
            )

        self.assertEqual(caught.exception.reason, "ccd_template_missing")

    def test_heavy_atom_fraction_threshold_is_enforced(self):
        input_sdf = _write_sdf(self.degraded, self.root / "in" / "ligand.sdf")

        with self.assertRaises(LigandBondOrderError) as caught:
            reconstruct_ligand_sdf(
                input_sdf=input_sdf,
                resname=PHOSPHATE_RESNAME,
                output_sdf=self.root / "out" / "ligand.sdf",
                ccd_dir=self.ccd_dir,
                min_observed_heavy_atom_fraction=1.5,
            )

        self.assertEqual(
            caught.exception.reason, "observed_heavy_atom_fraction_below_threshold"
        )


if __name__ == "__main__":
    unittest.main()
