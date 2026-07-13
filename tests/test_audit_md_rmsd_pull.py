from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "audit_md_rmsd_pull.py"
SPEC = importlib.util.spec_from_file_location("audit_md_rmsd_pull", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class AuditMDRMSDPullTest(unittest.TestCase):
    def test_spearman_detects_decreasing_path(self):
        correlation = MODULE.spearman_correlation([0, 1, 2, 3], [4, 3, 2, 1])
        self.assertAlmostEqual(correlation, -1.0)

    def test_build_audit_keeps_phase_supervision_disabled(self):
        rows = []
        for index in range(12):
            rows.append(
                {
                    "stage": "rmsd_pull",
                    "progress": index / 11,
                    "apo_ca_rmsd_angstrom": 1.5 - index * 0.08,
                    "holo_ca_rmsd_angstrom": 0.4 + index * 0.08,
                    "ligand_heavy_rmsd_angstrom": 1.0,
                    "temperature_k": 300.0,
                    "potential_kj_mol": -100.0,
                    "kinetic_kj_mol": 20.0,
                }
            )
        rows.extend(
            {
                "stage": "endpoint_hold",
                "progress": 1.0,
                "apo_ca_rmsd_angstrom": 0.5,
                "holo_ca_rmsd_angstrom": 1.2,
                "ligand_heavy_rmsd_angstrom": 1.0,
                "temperature_k": 300.0,
                "potential_kj_mol": -100.0,
                "kinetic_kj_mol": 20.0,
            }
            for _ in range(4)
        )
        audit = MODULE.build_audit(
            {"status": "rmsd_pull_smoke_passed", "transition_id": "test"},
            rows,
            max_ligand_rmsd_a=5.0,
            min_progress_correlation_magnitude=0.7,
            min_endpoint_hold_occupancy=0.6,
        )
        self.assertEqual(audit["status"], "path_metrics_passed")
        self.assertTrue(audit["passed"])
        self.assertTrue(audit["usage"]["geometry_supervision_candidate"])
        self.assertFalse(audit["usage"]["phase_supervision"])


if __name__ == "__main__":
    unittest.main()
