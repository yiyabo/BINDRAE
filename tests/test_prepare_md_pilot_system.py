from __future__ import annotations

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "prepare_md_pilot_system.py"
SPEC = importlib.util.spec_from_file_location("prepare_md_pilot_system", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class PrepareMDPilotSystemTest(unittest.TestCase):
    def test_load_candidate_by_index_and_transition_id(self):
        records = [
            {"transition_id": "a", "value": 1},
            {"transition_id": "b", "value": 2},
        ]
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "manifest.jsonl"
            path.write_text("".join(json.dumps(record) + "\n" for record in records))
            by_index = MODULE.load_candidate(path, candidate_index=1, transition_id=None)
            by_id = MODULE.load_candidate(path, candidate_index=0, transition_id="a")
        self.assertEqual(by_index["value"], 2)
        self.assertEqual(by_id["value"], 1)

    def test_load_candidate_rejects_ambiguous_transition_id(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "manifest.jsonl"
            path.write_text('{"transition_id":"a"}\n{"transition_id":"a"}\n')
            with self.assertRaises(ValueError):
                MODULE.load_candidate(path, candidate_index=0, transition_id="a")


if __name__ == "__main__":
    unittest.main()
