from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path


SCRIPT_PATH = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "build_apobind_replica_pilot.py"
)
SPEC = importlib.util.spec_from_file_location("build_apobind_replica_pilot", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def candidate(name: str) -> dict:
    return {
        "transition_id": f"apobind:{name}:pilot",
        "endpoints": {
            "apo_structure_path": f"samples/{name}/apo.pdb",
            "holo_structure_path": f"samples/{name}/holo.pdb",
        },
    }


def prepared(name: str, index: int) -> dict:
    return {"sample_id": name, "panel_index": index}


def dynamics(name: str, index: int, *, npt_status: str = "npt_smoke_passed") -> dict:
    return {
        "sample_id": name,
        "panel_index": index,
        "nvt_exit_code": 0,
        "nvt_status": "nvt_smoke_passed",
        "npt_exit_code": 0,
        "npt_status": npt_status,
    }


class BuildApobindReplicaPilotTest(unittest.TestCase):
    def test_orders_complete_contract_by_frozen_panel(self):
        names = ["system_b", "system_a"]
        rows = MODULE.ordered_contract_rows(
            [candidate("system_a"), candidate("system_b")],
            [prepared(name, index) for index, name in enumerate(names)],
            [dynamics(name, index) for index, name in enumerate(names)],
            expected_systems=2,
        )

        self.assertEqual([row[1]["sample_id"] for row in rows], names)
        self.assertEqual([row[0]["sample_id"] for row in rows], names)

    def test_rejects_a_failed_npt_system(self):
        with self.assertRaisesRegex(ValueError, "NPT did not pass"):
            MODULE.ordered_contract_rows(
                [candidate("system_a")],
                [prepared("system_a", 0)],
                [dynamics("system_a", 0, npt_status="npt_smoke_failed")],
                expected_systems=1,
            )

    def test_rejects_dynamics_system_mismatch(self):
        with self.assertRaisesRegex(ValueError, "do not match"):
            MODULE.ordered_contract_rows(
                [candidate("system_a")],
                [prepared("system_a", 0)],
                [dynamics("system_b", 0)],
                expected_systems=1,
            )


if __name__ == "__main__":
    unittest.main()
