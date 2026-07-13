from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path

import numpy as np


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "audit_md_atomistic_path.py"
SPEC = importlib.util.spec_from_file_location("audit_md_atomistic_path", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class AuditMDAtomisticPathTest(unittest.TestCase):
    def test_relative_bond_deviation(self):
        lengths = np.asarray([[1.0, 2.0], [1.1, 1.8]])
        reference = np.asarray([1.0, 2.0])
        self.assertAlmostEqual(MODULE.relative_bond_deviation(lengths, reference), 0.1)

    def test_summary_passes_clean_path(self):
        result = MODULE.summarize_path(
            severe_clash_counts=[0, 0],
            peptide_lengths_angstrom=np.asarray([[1.33], [1.34]]),
            heavy_bond_lengths_angstrom=np.asarray([[1.5], [1.52]]),
            reference_heavy_bonds_angstrom=np.asarray([1.5]),
            ligand_protein_minima_angstrom=[1.8, 1.7],
            temperatures_k=[299.0, 301.0],
            potentials_kj_mol=[-100.0, -101.0],
            max_severe_clashes_per_frame=0,
            max_peptide_bond_a=1.7,
            max_heavy_bond_relative_deviation=0.25,
            min_ligand_protein_distance_a=1.0,
        )
        self.assertTrue(result["passed"])


if __name__ == "__main__":
    unittest.main()
