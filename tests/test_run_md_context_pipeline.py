from __future__ import annotations

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "run_md_context_pipeline.py"
SPEC = importlib.util.spec_from_file_location("run_md_context_pipeline", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class RunMDContextPipelineTest(unittest.TestCase):
    def test_passed_reads_pretty_printed_report(self):
        with tempfile.TemporaryDirectory() as temporary:
            report = Path(temporary) / "report.json"
            report.write_text(
                json.dumps(
                    {"status": "minimized_ready_for_dynamics", "nested": {"value": 1}},
                    indent=2,
                )
                + "\n"
            )
            self.assertTrue(MODULE.passed(report, "minimized_ready_for_dynamics"))
            self.assertFalse(MODULE.passed(report, "nvt_smoke_passed"))


if __name__ == "__main__":
    unittest.main()
