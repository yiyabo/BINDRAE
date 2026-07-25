from __future__ import annotations

import importlib.util
import tempfile
import unittest
from pathlib import Path

import numpy as np


SCRIPT_PATH = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "export_apobind_structure_triplets.py"
)
SPEC = importlib.util.spec_from_file_location(
    "export_apobind_structure_triplets", SCRIPT_PATH
)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class ExportApobindStructureTripletsTest(unittest.TestCase):
    def test_kabsch_transform_recovers_row_vector_rigid_transform(self):
        mobile = np.asarray(
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 3.0]]
        )
        expected_rotation = np.asarray(
            [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]
        )
        expected_translation = np.asarray([4.0, -2.0, 1.5])
        target = mobile @ expected_rotation + expected_translation

        rotation, translation, aligned, rmsd = MODULE.kabsch_transform(mobile, target)

        np.testing.assert_allclose(rotation, expected_rotation, atol=1e-12)
        np.testing.assert_allclose(translation, expected_translation, atol=1e-12)
        np.testing.assert_allclose(aligned, target, atol=1e-12)
        self.assertLess(rmsd, 1e-12)

    def test_sample_id_is_stable_and_source_specific(self):
        record = {
            "source_rows": [12],
            "metadata": {"source_index": "10"},
            "endpoints": [{"pdb_id": "1abc"}, {"pdb_id": "2def"}],
        }
        ligand = {"resname": "LIG"}

        self.assertEqual(
            MODULE.sample_id_for(record, ligand),
            "apobind_10_1ABC_2DEF_LIG",
        )

    @unittest.skipUnless(
        importlib.util.find_spec("rdkit") and importlib.util.find_spec("Bio"),
        "Biopython and RDKit not installed",
    )
    def test_write_protein_chain_resolves_selected_altloc(self):
        from Bio.PDB import Chain, Model, Residue
        from Bio.PDB.Atom import Atom, DisorderedAtom

        model = Model.Model(0)
        chain = Chain.Chain("A")
        residue = Residue.Residue((" ", 1, " "), "ALA", " ")
        disordered_ca = DisorderedAtom("CA")
        disordered_ca.disordered_add(
            Atom(
                "CA",
                np.asarray([1.0, 2.0, 3.0]),
                1.0,
                0.2,
                "A",
                " CA ",
                1,
                element="C",
            )
        )
        disordered_ca.disordered_add(
            Atom(
                "CA",
                np.asarray([4.0, 5.0, 6.0]),
                1.0,
                0.8,
                "B",
                " CA ",
                2,
                element="C",
            )
        )
        residue.add(disordered_ca)
        chain.add(residue)
        model.add(chain)

        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "protein.pdb"
            MODULE.write_protein_chain(model, "A", output)
            ca_lines = [
                line
                for line in output.read_text().splitlines()
                if line.startswith("ATOM") and line[12:16].strip() == "CA"
            ]

        self.assertEqual(len(ca_lines), 1)
        self.assertEqual(ca_lines[0][16], " ")
        np.testing.assert_allclose(
            [
                float(ca_lines[0][30:38]),
                float(ca_lines[0][38:46]),
                float(ca_lines[0][46:54]),
            ],
            [4.0, 5.0, 6.0],
        )

    @unittest.skipUnless(
        importlib.util.find_spec("rdkit"),
        "RDKit not installed",
    )
    def test_component_hydrogen_filter_removes_disconnected_hydrogen(self):
        from rdkit import Chem

        molecule = Chem.MolFromSmiles("CC.[H-]")
        rows = [
            {"atom_id": "C1", "element": "C", "ordinal": 1},
            {"atom_id": "C2", "element": "C", "ordinal": 2},
            {"atom_id": "H1", "element": "H", "ordinal": 3},
        ]

        heavy_molecule = MODULE.remove_component_hydrogens(molecule, rows)

        self.assertEqual(heavy_molecule.GetNumAtoms(), 2)
        self.assertEqual(Chem.MolToSmiles(heavy_molecule), "CC")

    @unittest.skipUnless(
        importlib.util.find_spec("rdkit"),
        "RDKit not installed",
    )
    def test_component_hydrogen_filter_removes_stereo_defining_hydrogen(self):
        from rdkit import Chem

        parser = Chem.SmilesParserParams()
        parser.removeHs = False
        molecule = Chem.MolFromSmiles("[H]/N=C(/C)N", parser)
        rows = [
            {
                "atom_id": f"A{index}",
                "element": atom.GetSymbol(),
                "ordinal": index,
            }
            for index, atom in enumerate(molecule.GetAtoms(), start=1)
        ]

        heavy_molecule = MODULE.remove_component_hydrogens(molecule, rows)

        self.assertEqual(heavy_molecule.GetNumAtoms(), 4)
        self.assertFalse(any(atom.GetAtomicNum() == 1 for atom in heavy_molecule.GetAtoms()))

    @unittest.skipUnless(
        importlib.util.find_spec("rdkit") and importlib.util.find_spec("Bio"),
        "Biopython and RDKit not installed",
    )
    def test_observed_ligand_sdf_uses_ccd_bonds_and_observed_coordinates(self):
        from Bio.PDB.Atom import Atom
        from Bio.PDB.Residue import Residue
        from rdkit import Chem
        from rdkit.Chem import AllChem

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            ccd = root / "LIG.sdf"
            template = Chem.MolFromSmiles("C=O")
            template = Chem.AddHs(template)
            AllChem.EmbedMolecule(template, randomSeed=7)
            writer = Chem.SDWriter(str(ccd))
            writer.write(template)
            writer.close()
            rows = [
                {"atom_id": "C1", "element": "C", "ordinal": 1},
                {"atom_id": "O1", "element": "O", "ordinal": 2},
                {"atom_id": "H1", "element": "H", "ordinal": 3},
                {"atom_id": "H2", "element": "H", "ordinal": 4},
            ]
            residue = Residue(("H_LIG", 1, " "), "LIG", " ")
            coordinates = {
                "C1": np.asarray([1.0, 2.0, 3.0]),
                "O1": np.asarray([2.2, 2.0, 3.0]),
            }
            for serial, (name, coordinate) in enumerate(coordinates.items(), start=1):
                residue.add(
                    Atom(
                        name,
                        coordinate,
                        1.0,
                        1.0,
                        " ",
                        f" {name:<3}",
                        serial,
                        element=name[0],
                    )
                )

            output_sdf = root / "observed.sdf"
            output_coords = root / "coords.npy"
            result = MODULE.build_observed_ligand_sdf(
                ccd_sdf=ccd,
                component_rows=rows,
                ligand_residue=residue,
                output_sdf=output_sdf,
                output_coords=output_coords,
            )

            molecule = Chem.SDMolSupplier(str(output_sdf), removeHs=False)[0]
            self.assertEqual(molecule.GetNumAtoms(), 2)
            self.assertEqual(molecule.GetBondBetweenAtoms(0, 1).GetBondTypeAsDouble(), 2.0)
            np.testing.assert_allclose(
                molecule.GetConformer().GetPositions(),
                np.asarray([coordinates["C1"], coordinates["O1"]]),
                atol=1e-3,
            )
            self.assertEqual(result["heavy_atoms"], 2)


if __name__ == "__main__":
    unittest.main()
