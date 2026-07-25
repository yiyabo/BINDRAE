from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path


SCRIPT_PATH = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "build_apobind_preparation_rescue.py"
)
SPEC = importlib.util.spec_from_file_location(
    "build_apobind_preparation_rescue", SCRIPT_PATH
)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def _row(index: int, category: str = "contact_switch") -> dict:
    return {
        "sample_id": f"sample_{index}",
        "eligible": True,
        "motion_category": category,
        "pilot_score": index / 10.0,
        "n_residues": 100 + index * 20,
        "ligand_heavy_atoms": 10 + index,
        "ca_aligned_rmsd": 0.5 + index * 0.2,
        "pocket_ca_rmsd": 0.4 + index * 0.3,
        "max_ca_displacement": 2.0 + index,
        "contact_changes": index,
        "ligand_scaffold": f"scaffold_{index}",
        "endpoint_pdb_ids": [f"A{index}", f"H{index}"],
    }


class BuildApobindPreparationRescueTest(unittest.TestCase):
    def test_replacement_is_same_category_and_deterministic(self):
        rows = [_row(index) for index in range(6)]
        prior = ["sample_0", "sample_1", "sample_2"]

        first, ranking = MODULE.choose_replacement(
            rows,
            prior_sample_ids=prior,
            failed_sample_id="sample_1",
        )
        second, _ = MODULE.choose_replacement(
            list(reversed(rows)),
            prior_sample_ids=prior,
            failed_sample_id="sample_1",
        )

        self.assertEqual(first["sample_id"], second["sample_id"])
        self.assertNotIn(first["sample_id"], prior)
        self.assertEqual(first["motion_category"], "contact_switch")
        self.assertEqual(ranking[0], first)

    def test_replacement_rejects_failed_sample_outside_prior_panel(self):
        with self.assertRaisesRegex(ValueError, "absent from the prior panel"):
            MODULE.choose_replacement(
                [_row(0), _row(1)],
                prior_sample_ids=["sample_0"],
                failed_sample_id="sample_1",
            )


if __name__ == "__main__":
    unittest.main()
