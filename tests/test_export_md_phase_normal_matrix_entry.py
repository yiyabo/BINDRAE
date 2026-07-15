import unittest
from pathlib import Path

from scripts.export_md_phase_normal_matrix_entry import infer_replica_matrix


class ExportMdPhaseNormalMatrixEntryTest(unittest.TestCase):
    def test_matrix_is_inferred_before_target_variant(self):
        root = Path("processed_data/md_transition/silver_collection")
        source = root / "targets_mappingfix_v2/system/replica_07"
        self.assertEqual(infer_replica_matrix(source), root / "replica_matrix.jsonl")

    def test_missing_target_component_fails(self):
        with self.assertRaises(ValueError):
            infer_replica_matrix(Path("processed_data/md_transition/cache/system"))


if __name__ == "__main__":
    unittest.main()
