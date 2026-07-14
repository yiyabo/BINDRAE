from __future__ import annotations

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "collect_md_context_results.py"
SPEC = importlib.util.spec_from_file_location("collect_md_context_results", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class CollectMDContextResultsTest(unittest.TestCase):
    def test_collects_passed_and_keeps_failed_outcome(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            rows = []
            for index, status in enumerate(("completed", "failed")):
                system = root / f"system-{index}"
                setup = system / "setup"
                setup.mkdir(parents=True)
                (system / "pipeline_status.json").write_text(
                    json.dumps({"status": status, "failed_stage": "npt" if index else None})
                )
                context = system / "context.jsonl"
                if index == 0:
                    context.write_text(json.dumps({"status": "prepared"}))
                rows.append(
                    {
                        "matrix_index": index,
                        "system_sample_id": f"sample-{index}",
                        "transition_id": f"transition-{index}",
                        "setup_dir": str(setup),
                        "context_record": str(context),
                    }
                )
            matrix = root / "matrix.jsonl"
            matrix.write_text("".join(json.dumps(row) + "\n" for row in rows))
            with patch.object(MODULE, "validate_transition_record", return_value=[]):
                contexts, summary = MODULE.collect(matrix)
            self.assertEqual(len(contexts), 1)
            self.assertEqual(summary["admitted_systems"], 1)
            self.assertEqual(summary["outcome_counts"], {"admitted": 1, "npt": 1})


if __name__ == "__main__":
    unittest.main()
