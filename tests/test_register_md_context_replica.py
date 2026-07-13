from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "register_md_context_replica.py"
SPEC = importlib.util.spec_from_file_location("register_md_context_replica", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class RegisterMDContextReplicaTest(unittest.TestCase):
    def test_infer_frame_interval_uses_median(self):
        rows = [{"time_ps": 0.0}, {"time_ps": 0.2}, {"time_ps": 0.4}, {"time_ps": 1.0}]
        self.assertAlmostEqual(MODULE.infer_frame_interval_ps(rows), 0.2)

    def test_context_record_never_enables_phase_supervision(self):
        candidate = {
            "transition_id": "ahoj:sample:pilot",
            "status": "metadata_verified",
            "evidence": {},
            "trajectory": {},
            "usage": {},
            "quality": {"transition_verified": False, "notes": []},
        }
        report = {"status": "npt_smoke_passed", "final": {"temperature_k": 300.0}}
        metrics = [{"time_ps": 0.0}, {"time_ps": 0.2}, {"time_ps": 0.4}]
        record = MODULE.build_context_record(
            candidate,
            npt_dir=Path("replica"),
            npt_report=report,
            metrics=metrics,
            replica_index=2,
        )
        self.assertEqual(record["evidence"]["tier"], "context_equilibrium")
        self.assertFalse(record["evidence"]["contains_endpoint_transition"])
        self.assertFalse(record["usage"]["phase_supervision"])
        self.assertEqual(record["transition_id"], "ahoj:sample:context-npt-2")


if __name__ == "__main__":
    unittest.main()
