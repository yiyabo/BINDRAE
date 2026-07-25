from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


SCRIPT_PATH = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "freeze_apobind_prepared_panel.py"
)
SPEC = importlib.util.spec_from_file_location("freeze_apobind_prepared_panel", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def candidate(root: Path, name: str) -> dict:
    sample = root / "samples" / name
    sample.mkdir(parents=True)
    for filename in ("apo.pdb", "holo.pdb", "ligand.sdf"):
        (sample / filename).write_text("candidate\n")
    return {
        "transition_id": f"apobind:{name}:pilot",
        "endpoints": {
            "apo_structure_path": str(sample / "apo.pdb"),
            "holo_structure_path": str(sample / "holo.pdb"),
        },
    }


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))


def preparation(root: Path, name: str, maximum: float = 100.0) -> str:
    system = root / "prepared" / name
    system.mkdir(parents=True)
    outputs = {}
    for key, filename in {
        "minimized_pdb": "minimized_solvated.pdb",
        "state_xml": "minimized_state.xml",
        "system_xml": "system.xml",
        "unsolvated_pdb": "repaired_complex_unsolvated.pdb",
    }.items():
        path = system / filename
        path.write_text("prepared\n")
        outputs[key] = str(path)
    report = system / "preparation_report.json"
    report.write_text(
        json.dumps(
            {
                "status": "minimized_ready_for_dynamics",
                "minimization": {
                    "ready_for_dynamics_smoke": True,
                    "maximum_residue_net_force_kj_mol_nm": maximum,
                    "max_residue_net_force_threshold_kj_mol_nm": 500.0,
                },
                "outputs": outputs,
            }
        )
        + "\n"
    )
    return str(report)


def state_row(name: str, report: str, ready: bool = True) -> dict:
    return {
        "sample_id": name,
        "exit_code": 0,
        "status": "minimized_ready_for_dynamics" if ready else "minimized_incomplete",
        "ready_for_dynamics_smoke": ready,
        "preparation_report": report,
    }


class FreezeApobindPreparedPanelTest(unittest.TestCase):
    def build_fixture(self, root: Path, retry_ready: bool) -> argparse.Namespace:
        panel_records = [candidate(root, name) for name in ("original", "failed", "retry")]
        rescue_records = [panel_records[2], candidate(root, "replacement")]
        panel_manifest = root / "panel.jsonl"
        rescue_manifest = root / "rescue.jsonl"
        write_jsonl(panel_manifest, panel_records)
        write_jsonl(rescue_manifest, rescue_records)
        original_report = preparation(root, "original")
        retry_report = preparation(root, "retry")
        replacement_report = preparation(root, "replacement")
        panel_state = root / "panel_state.json"
        panel_state.write_text(
            json.dumps(
                {
                    "schema_version": MODULE.PANEL_STATE_SCHEMA,
                    "candidate_manifest_sha256": sha256(panel_manifest),
                    "systems": [
                        state_row("original", original_report),
                        state_row("failed", str(root / "missing.json"), False),
                        state_row("retry", str(root / "old_retry.json"), False),
                    ],
                }
            )
            + "\n"
        )
        rescue_plan = root / "rescue_plan.json"
        rescue_plan.write_text(
            json.dumps(
                {
                    "schema_version": MODULE.RESCUE_PLAN_SCHEMA,
                    "roles": [
                        {"sample_id": "retry", "role": "longer_minimization_rescue"},
                        {
                            "sample_id": "replacement",
                            "role": "forcefield_unsupported_replacement",
                            "replaces_sample_id": "failed",
                        },
                    ],
                }
            )
            + "\n"
        )
        rescue_state = root / "rescue_state.json"
        rescue_state.write_text(
            json.dumps(
                {
                    "schema_version": MODULE.RESCUE_STATE_SCHEMA,
                    "manifest_sha256": sha256(rescue_manifest),
                    "plan_sha256": sha256(rescue_plan),
                    "systems": [
                        state_row("retry", retry_report, retry_ready),
                        state_row("replacement", replacement_report),
                    ],
                }
            )
            + "\n"
        )
        return argparse.Namespace(
            panel_manifest=panel_manifest,
            panel_state=panel_state,
            rescue_manifest=rescue_manifest,
            rescue_plan=rescue_plan,
            rescue_state=rescue_state,
            output_dir=root / "frozen",
            base_dir=root,
            expected_count=3,
            residue_force_threshold=500.0,
        )

    def test_freezes_original_retry_and_replacement_in_slot_order(self):
        with tempfile.TemporaryDirectory() as temporary:
            args = self.build_fixture(Path(temporary), retry_ready=True)
            state = MODULE.run(args)

            self.assertEqual(state["status"], "ready_for_dynamics_smoke")
            self.assertEqual(
                [row["sample_id"] for row in state["systems"]],
                ["original", "replacement", "retry"],
            )
            self.assertEqual(state["counts"]["prepared_ready"], 3)
            self.assertEqual(state["counts"]["forcefield_replacements"], 1)
            self.assertEqual(state["counts"]["longer_minimization_rescues"], 1)

    def test_failed_retry_keeps_panel_incomplete(self):
        with tempfile.TemporaryDirectory() as temporary:
            args = self.build_fixture(Path(temporary), retry_ready=False)
            state = MODULE.run(args)

            self.assertEqual(state["status"], "incomplete")
            self.assertEqual(state["counts"]["prepared_ready"], 2)
            self.assertEqual(state["counts"]["unresolved"], 1)
            self.assertEqual(state["unresolved_slots"][0]["reason"], "rescue_not_ready")


if __name__ == "__main__":
    unittest.main()
