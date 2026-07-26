from __future__ import annotations

import importlib.util
import json
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

SCRIPT_PATH = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "repair_triplet_ligand_bond_orders.py"
)
SPEC = importlib.util.spec_from_file_location(
    "repair_triplet_ligand_bond_orders", SCRIPT_PATH
)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


#: Graph-identical pair: same heavy atoms, same bonds, different charge state.
#: This is the shape of the real NAI/NAD collision, where the sample id and the
#: deposited HETATM disagree and the connectivity cannot arbitrate.
NEUTRAL_SMILES = "CO[P](=O)(O)O"
ANIONIC_SMILES = "CO[P](=O)([O-])O"


def _mol(smiles: str) -> "Chem.Mol":
    molecule = Chem.MolFromSmiles(smiles)
    assert molecule is not None
    AllChem.Compute2DCoords(molecule)
    return molecule


def _degrade(molecule: "Chem.Mol") -> "Chem.Mol":
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
class RepairTripletLigandBondOrdersTest(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        self.ccd_dir = self.root / "ccd"
        _write_sdf(_mol(NEUTRAL_SMILES), self.ccd_dir / "NEU.sdf")
        _write_sdf(_mol(ANIONIC_SMILES), self.ccd_dir / "ANI.sdf")

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def _make_sample(self, sample_id: str, ccd_resname: str, *, meta_resname=None,
                     holo_resnames=()) -> Path:
        source = Chem.MolFromMolFile(
            str(self.ccd_dir / f"{ccd_resname}.sdf"), removeHs=False, sanitize=True
        )
        degraded = _degrade(source)
        sample_dir = self.root / "triplets" / sample_id
        _write_sdf(degraded, sample_dir / "ligand.sdf")
        xyz = np.asarray(degraded.GetConformer().GetPositions(), dtype=np.float32)
        np.save(sample_dir / "ligand_coords.npy", xyz)
        (sample_dir / "meta.json").write_text(
            json.dumps({"ligand_resname": meta_resname or ccd_resname}), encoding="utf-8"
        )
        if holo_resnames:
            lines = []
            for index, resname in enumerate(holo_resnames):
                x, y, z = xyz[index % len(xyz)]
                lines.append(
                    f"HETATM{index + 1:5d}  C1  {resname:>3s} A 401    "
                    f"{x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00           C"
                )
            (sample_dir / "holo.pdb").write_text("\n".join(lines) + "\nEND\n", encoding="utf-8")
        return sample_dir

    def _repair(self, sample_dir: Path, *, apply_changes: bool = True) -> dict:
        return MODULE.repair_sample(
            sample_dir,
            ccd_dir=self.ccd_dir,
            apply_changes=apply_changes,
            allow_download=False,
            missing_bond_policy="restore",
            min_observed_heavy_atom_fraction=0.0,
            keep_backup=True,
        )

    def test_repair_rewrites_sdf_and_keeps_a_legacy_copy(self):
        sample_dir = self._make_sample("1abc-A-ANI-401", "ANI")
        original = (sample_dir / "ligand.sdf").read_bytes()

        row = self._repair(sample_dir)

        self.assertEqual(row["status"], "repaired")
        self.assertEqual(row["resolved_resname"], "ANI")
        self.assertTrue(row["defect_present_before"])
        self.assertEqual(row["after"]["sentinel_hydrides"], [])
        self.assertEqual(row["after"]["formal_charge"], -1)
        self.assertNotEqual((sample_dir / "ligand.sdf").read_bytes(), original)
        legacy = sample_dir / MODULE.LEGACY_SDF_NAME
        self.assertTrue(legacy.is_file())
        self.assertEqual(legacy.read_bytes(), original)

    def test_dry_run_leaves_every_file_untouched(self):
        sample_dir = self._make_sample("1abc-A-ANI-401", "ANI")
        original = (sample_dir / "ligand.sdf").read_bytes()

        row = self._repair(sample_dir, apply_changes=False)

        self.assertEqual(row["status"], "would_repair")
        self.assertEqual((sample_dir / "ligand.sdf").read_bytes(), original)
        self.assertFalse((sample_dir / MODULE.LEGACY_SDF_NAME).exists())
        self.assertFalse(list(sample_dir.glob("*.tmp*.sdf")))

    def test_repair_verifies_against_frozen_ligand_coords(self):
        sample_dir = self._make_sample("1abc-A-ANI-401", "ANI")

        row = self._repair(sample_dir)

        self.assertTrue(row["coords_check"]["checked"])
        self.assertLess(row["coords_check"]["max_abs_deviation_angstrom"], 1e-3)

    def test_corrupt_ligand_coords_blocks_the_write(self):
        sample_dir = self._make_sample("1abc-A-ANI-401", "ANI")
        original = (sample_dir / "ligand.sdf").read_bytes()
        np.save(sample_dir / "ligand_coords.npy", np.zeros((3, 3), dtype=np.float32))

        row = self._repair(sample_dir)

        self.assertEqual(row["status"], "rejected")
        self.assertEqual(row["reason"], "ligand_coords_shape_mismatch")
        self.assertEqual((sample_dir / "ligand.sdf").read_bytes(), original)

    def test_holo_hetatm_does_not_compete_with_the_name_on_the_sample(self):
        # AHoJ pairs a query entry with a different holo entry, and the two
        # routinely hold related-but-distinct components (1t26 has NAI, its holo
        # 1t2d has NAD).  holo.pdb therefore contributes the holo component's
        # name, which must not be read as a chemistry ambiguity about the
        # ligand that was extracted from the query structure.
        sample_dir = self._make_sample(
            "1abc-A-NEU-401", "NEU", meta_resname="NEU", holo_resnames=("ANI",)
        )

        row = self._repair(sample_dir)

        self.assertEqual(row["status"], "repaired")
        self.assertEqual(row["resolved_resname"], "NEU")
        self.assertEqual(row["resolved_tier"], 0)
        self.assertFalse(row["resname_ambiguous"])

    def test_two_names_in_the_same_tier_are_rejected_not_guessed(self):
        # Genuine ambiguity: meta and the sample id disagree, both reconstruct,
        # and they imply different formal charges.
        sample_dir = self._make_sample(
            "1abc-A-ANI-401", "ANI", meta_resname="NEU"
        )
        original = (sample_dir / "ligand.sdf").read_bytes()

        row = self._repair(sample_dir)

        self.assertEqual(row["status"], "rejected")
        self.assertEqual(row["reason"], "resname_chemistry_ambiguous")
        self.assertTrue(row["resname_ambiguous"])
        self.assertEqual(
            {entry["formal_charge"] for entry in row["accepted_resnames"]}, {0, -1}
        )
        self.assertEqual((sample_dir / "ligand.sdf").read_bytes(), original)
        self.assertFalse(list(sample_dir.glob("*.tmp*.sdf")))

    def test_holo_hetatm_still_rescues_a_stale_meta_entry(self):
        sample_dir = self._make_sample(
            "9zzz-Q-QQQ-401", "ANI", meta_resname="QQQ", holo_resnames=("ANI",)
        )

        row = self._repair(sample_dir)

        self.assertEqual(row["status"], "repaired")
        self.assertEqual(row["resolved_resname"], "ANI")
        self.assertEqual(row["resolved_tier"], 1)

    def test_unknown_resname_lands_in_the_reject_ledger(self):
        # No candidate resolves: meta and the sample id both name a component
        # that has no CCD entry, and there is no holo.pdb to fall back to.
        sample_dir = self._make_sample("1abc-A-QQQ-401", "ANI", meta_resname="QQQ")

        row = self._repair(sample_dir)

        self.assertEqual(row["status"], "rejected")
        self.assertEqual(row["reason"], "ccd_template_missing")

    def test_sample_id_resname_rescues_a_stale_meta_entry(self):
        sample_dir = self._make_sample("1abc-A-ANI-401", "ANI", meta_resname="QQQ")

        row = self._repair(sample_dir)

        self.assertEqual(row["status"], "repaired")
        self.assertEqual(row["resolved_resname"], "ANI")
        self.assertEqual(
            [attempt["reason"] for attempt in row["failed_resname_attempts"]],
            ["ccd_template_missing"],
        )

    def test_resname_candidates_prefers_meta_then_sample_id_then_holo(self):
        sample_dir = self._make_sample(
            "1abc-A-NEU-401", "ANI", meta_resname="NEU", holo_resnames=("ANI", "GOL", "HOH")
        )

        candidates = MODULE.resname_candidates(sample_dir)

        self.assertEqual(candidates[0], "NEU")
        self.assertIn("ANI", candidates)
        self.assertNotIn("HOH", candidates)


if __name__ == "__main__":
    unittest.main()
