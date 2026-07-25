from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "select_md_mapping_smoke_panel.py"
SPEC = importlib.util.spec_from_file_location("select_md_mapping_smoke_panel", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def candidate(category: str, index: int, *, mapping: float = 0.96, delta: int = 2) -> dict:
    name = f"{category}_{index}"
    return {
        "transition_id": name,
        "endpoints": {"holo_structure_path": f"samples/{name}/holo.pdb"},
        "screening": {
            "apo_n_residues": 100 + delta,
            "holo_n_residues": 100,
            "residue_mapping_fraction": mapping,
            "motion_category": category,
            "selection_rank": index,
        },
    }


class MappingSmokePanelTest(unittest.TestCase):
    def test_balances_categories_and_is_deterministic(self):
        rows = [
            candidate(category, index, mapping=0.95 + index * 0.001, delta=index + 1)
            for category in ("domain_motion", "local_pocket", "moderate_motion", "contact_switch")
            for index in range(8)
        ]
        first, quotas = MODULE.build_panel(rows, panel_size=8, minimum_mapping_fraction=0.95)
        second, _ = MODULE.build_panel(rows, panel_size=8, minimum_mapping_fraction=0.95)
        self.assertEqual(quotas, {key: 2 for key in sorted(quotas)})
        self.assertEqual([MODULE.sample_id(row) for row in first], [MODULE.sample_id(row) for row in second])
        self.assertEqual(
            {row["screening"]["motion_category"] for row in first},
            {"domain_motion", "local_pocket", "moderate_motion", "contact_switch"},
        )

    def test_excludes_equal_count_and_under_mapping_rows(self):
        rows = [candidate("a", index) for index in range(4)]
        rows[0]["screening"]["apo_n_residues"] = rows[0]["screening"]["holo_n_residues"]
        rows[1]["screening"]["residue_mapping_fraction"] = 0.94
        panel, _ = MODULE.build_panel(rows, panel_size=2, minimum_mapping_fraction=0.95)
        self.assertEqual(len(panel), 2)
        self.assertNotIn(MODULE.sample_id(rows[0]), {MODULE.sample_id(row) for row in panel})
        self.assertNotIn(MODULE.sample_id(rows[1]), {MODULE.sample_id(row) for row in panel})


if __name__ == "__main__":
    unittest.main()
