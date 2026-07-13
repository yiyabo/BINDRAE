from __future__ import annotations

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "merge_md_transition_manifests.py"
SPEC = importlib.util.spec_from_file_location("merge_md_transition_manifests", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class MergeMDTransitionManifestsTest(unittest.TestCase):
    def test_merge_sorts_records(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            first = root / "first.jsonl"
            second = root / "second.jsonl"
            first.write_text(json.dumps({"transition_id": "b"}) + "\n")
            second.write_text(json.dumps({"transition_id": "a"}) + "\n")
            records = MODULE.merge_records([first, second])
        self.assertEqual([record["transition_id"] for record in records], ["a", "b"])

    def test_merge_rejects_duplicate_ids(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            first = root / "first.jsonl"
            second = root / "second.jsonl"
            first.write_text(json.dumps({"transition_id": "same"}) + "\n")
            second.write_text(json.dumps({"transition_id": "same"}) + "\n")
            with self.assertRaises(ValueError):
                MODULE.merge_records([first, second])


if __name__ == "__main__":
    unittest.main()
