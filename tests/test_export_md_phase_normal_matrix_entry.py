import json
import tempfile
import unittest
from pathlib import Path

from scripts.export_md_phase_normal_matrix_entry import (
    find_replica_record,
    infer_replica_matrix,
)


class ExportMdPhaseNormalMatrixEntryTest(unittest.TestCase):
    def test_matrix_is_inferred_before_target_variant(self):
        root = Path("processed_data/md_transition/silver_collection")
        source = root / "targets_mappingfix_v2/system/replica_07"
        self.assertEqual(infer_replica_matrix(source), root / "replica_matrix.jsonl")

    def test_missing_target_component_fails(self):
        with self.assertRaises(ValueError):
            infer_replica_matrix(Path("processed_data/md_transition/cache/system"))

    def test_explicit_matrix_supports_detached_target_tree(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            matrix_path = Path(temporary_dir) / "replica_matrix.jsonl"
            expected = {"sample_id": "sample__silver_r00", "replica_index": 0}
            matrix_path.write_text(json.dumps(expected) + "\n")
            collection_record = {
                "sample_id": expected["sample_id"],
                "source_dir": "processed_data/md_transition/phase_block_targets/x",
            }

            actual = find_replica_record(collection_record, [matrix_path])

            self.assertEqual(actual, expected)


if __name__ == "__main__":
    unittest.main()
