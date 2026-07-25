from __future__ import annotations

import importlib.util
import unittest
from collections import Counter
from pathlib import Path


SCRIPT_PATH = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "select_apobind_preparation_panel.py"
)
SPEC = importlib.util.spec_from_file_location(
    "select_apobind_preparation_panel", SCRIPT_PATH
)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def _row(index: int, category: str) -> dict:
    return {
        "sample_id": f"sample_{index}",
        "eligible": True,
        "motion_category": category,
        "pilot_score": index / 10.0,
        "n_residues": 100 + 10 * index,
        "ligand_heavy_atoms": 10 + index,
        "ca_aligned_rmsd": 0.3 + 0.1 * index,
        "pocket_ca_rmsd": 0.4 + 0.2 * index,
        "max_ca_displacement": 1.0 + index,
        "contact_changes": index,
        "ligand_scaffold": f"scaffold_{index}",
        "endpoint_pdb_ids": [f"A{index}", f"H{index}"],
    }


class SelectApobindPreparationPanelTest(unittest.TestCase):
    def test_selection_balances_categories_and_is_deterministic(self):
        rows = [
            *[_row(index, "contact_switch") for index in range(6)],
            *[_row(index, "moderate_motion") for index in range(6, 10)],
        ]

        first, quotas = MODULE.select_panel(rows, 6)
        second, second_quotas = MODULE.select_panel(list(reversed(rows)), 6)

        self.assertEqual(quotas, {"contact_switch": 3, "moderate_motion": 3})
        self.assertEqual(second_quotas, quotas)
        self.assertEqual(
            [row["sample_id"] for row in first],
            [row["sample_id"] for row in second],
        )
        self.assertEqual(
            Counter(row["motion_category"] for row in first),
            Counter({"contact_switch": 3, "moderate_motion": 3}),
        )
        self.assertEqual(len({row["ligand_scaffold"] for row in first}), 6)

    def test_selection_rejects_missing_features(self):
        rows = [_row(0, "a"), _row(1, "b")]
        rows[0]["pocket_ca_rmsd"] = None

        with self.assertRaisesRegex(ValueError, "missing selection feature"):
            MODULE.select_panel(rows, 2)


if __name__ == "__main__":
    unittest.main()
