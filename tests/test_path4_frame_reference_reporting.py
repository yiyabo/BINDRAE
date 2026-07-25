import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from scripts.build_path4_openmm_frame_reference import (
    FrameReferenceBuildError,
    _rejection_report,
    _top_force_atom_diagnostics,
    _validate_state_for_stage,
)
from scripts.optimize_path4_openmm_gate0 import _minimization_reporter_diagnostics
from scripts.run_path4_gate0_dev_panel import _record_failure
from src.data.openmm_gate0 import TopologyAtomRecord


class Path4FrameReferenceReportingTest(unittest.TestCase):
    def test_top_force_atom_reports_scope_and_nearest_nonbonded_atom(self):
        atom_records = [
            TopologyAtomRecord(0, 0, "A", 1, "", "ALA", "CA", "C"),
            TopologyAtomRecord(1, 0, "A", 1, "", "ALA", "HA", "H"),
            TopologyAtomRecord(2, 1, "L", 2, "", "LIG", "C1", "C"),
            TopologyAtomRecord(3, 2, "A", 3, "", "GLY", "CA", "C"),
        ]
        diagnostics = _top_force_atom_diagnostics(
            np.asarray(
                [
                    [0.0, 0.0, 0.0],
                    [1.0, 0.0, 0.0],
                    [0.2, 0.0, 0.0],
                    [5.0, 0.0, 0.0],
                ]
            ),
            np.asarray(
                [
                    [100.0, 0.0, 0.0],
                    [1.0, 0.0, 0.0],
                    [-90.0, 0.0, 0.0],
                    [2.0, 0.0, 0.0],
                ]
            ),
            atom_records,
            [{1}, {0}, set(), set()],
            protein_atom_count=2,
            mapped_atom_indices={0},
            top_k=2,
        )

        maximum = diagnostics[0]
        self.assertEqual(maximum["topology_atom_index"], 0)
        self.assertEqual(maximum["scope"], "mapped_protein")
        nearest = maximum["nearest_not_directly_bonded_atom"]
        self.assertEqual(nearest["topology_atom_index"], 2)
        self.assertEqual(nearest["scope"], "environment")
        self.assertAlmostEqual(nearest["distance_angstrom"], 0.2)

    def test_minimization_reporter_diagnostics_do_not_infer_termination(self):
        self.assertEqual(
            _minimization_reporter_diagnostics(None),
            {
                "reporter_available": False,
                "reporter_callback_count": None,
                "last_reported_iteration_index": None,
            },
        )
        self.assertEqual(
            _minimization_reporter_diagnostics(
                SimpleNamespace(report_calls=4, last_iteration=3)
            ),
            {
                "reporter_available": True,
                "reporter_callback_count": 4,
                "last_reported_iteration_index": 3,
            },
        )

    def test_prepared_failure_keeps_measured_force_and_stage(self):
        with self.assertRaises(FrameReferenceBuildError) as raised:
            _validate_state_for_stage(
                10.0,
                np.asarray([[0.0, 0.0, 101.0]]),
                1,
                maximum_atomic_force_kj_mol_nm=100.0,
                state_label="Prepared OpenMM reference topology",
                failure_stage="prepared_topology_preflight",
                rejection_type="prepared_topology_physical_failure",
            )

        error = raised.exception
        self.assertEqual(error.failure_stage, "prepared_topology_preflight")
        self.assertEqual(
            error.rejection_type, "prepared_topology_physical_failure"
        )
        self.assertEqual(
            error.failure_context["atomic_force_max_kj_mol_nm"], 101.0
        )
        self.assertIn("Prepared OpenMM reference topology", error.rejection_reason)

    def test_frame_failure_reports_exact_frame_and_partial_diagnostics(self):
        with self.assertRaises(FrameReferenceBuildError) as raised:
            _validate_state_for_stage(
                20.0,
                np.asarray([[0.0, 0.0, 250.0]]),
                1,
                maximum_atomic_force_kj_mol_nm=100.0,
                state_label="All-atom frame reference 7",
                failure_stage="frame_reference_preflight",
                rejection_type="frame_reference_physical_failure",
                context={
                    "frame_index": 7,
                    "time": 0.35,
                    "mapped_heavy_rms_angstrom": 0.02,
                },
            )

        error = raised.exception
        failed_record = {**error.failure_context, "accepted": False}
        error.partial_report = {
            "status": "running",
            "frame_preflight_completed": 1,
            "frame_preflight": [
                {"frame_index": 0, "atomic_force_max_kj_mol_nm": 50.0},
                failed_record,
            ],
        }
        report = _rejection_report(error, wall_seconds=1.25)

        self.assertEqual(report["status"], "rejected")
        self.assertEqual(report["failure_stage"], "frame_reference_preflight")
        self.assertEqual(report["failed_frame_index"], 7)
        self.assertEqual(report["failed_frame_time"], 0.35)
        self.assertEqual(report["frame_preflight_completed"], 1)
        self.assertEqual(
            report["frame_reference_atomic_force_max_observed_kj_mol_nm"],
            250.0,
        )

    def test_panel_state_uses_reported_frame_failure_stage(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            report_path = root / "report.json"
            log_path = root / "reference.log"
            state_path = root / "panel_state.json"
            report_path.write_text(
                json.dumps(
                    {
                        "status": "rejected",
                        "failure_stage": "frame_reference_preflight",
                        "rejection_type": "frame_reference_physical_failure",
                        "rejection_reason": "force threshold",
                        "failed_frame_index": 9,
                        "failed_frame_time": 0.45,
                        "frame_preflight_completed": 9,
                        "failure_context": {"atomic_force_max_kj_mol_nm": 1.5e6},
                    }
                )
            )
            log_path.write_text("failed\n")
            state = {
                "systems": {
                    "sample": {
                        "status": "running",
                        "completed_stages": [],
                    }
                }
            }

            _record_failure(
                state,
                state_path,
                "sample",
                status="rejected_reference_preflight",
                stage="reference_preflight",
                log_path=log_path,
                report_path=report_path,
            )

            record = state["systems"]["sample"]
            self.assertEqual(record["runner_failure_stage"], "reference_preflight")
            self.assertEqual(record["failure_stage"], "frame_reference_preflight")
            self.assertEqual(record["failed_frame_index"], 9)
            self.assertEqual(record["failed_frame_time"], 0.45)
            self.assertEqual(record["frame_preflight_completed"], 9)
            self.assertEqual(
                record["failure_context"]["atomic_force_max_kj_mol_nm"], 1.5e6
            )


if __name__ == "__main__":
    unittest.main()
