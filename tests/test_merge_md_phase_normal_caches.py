import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from scripts.merge_md_phase_normal_caches import (
    deduplicate_records,
    merge_collections,
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_collection(directory: Path, sample_id: str, transition_id: str) -> None:
    directory.mkdir(parents=True)
    target = directory / f"{sample_id}.npz"
    np.savez_compressed(
        target,
        schema_version=np.array("md_phase_normal_v1"),
        sample_id=np.array(sample_id),
        transition_id=np.array(transition_id),
    )
    record = {
        "sample_id": sample_id,
        "transition_id": transition_id,
        "relative_path": target.name,
        "sha256": _sha256(target),
        "n_frames": 3,
        "n_residues": 2,
        "valid_residual_points": 1,
        "audit_metrics": {
            "active_interior_points": 2,
            "confident_phase_points": 1,
            "residual_candidate_points": 2,
        },
    }
    (directory / "manifest.jsonl").write_text(json.dumps(record) + "\n")
    (directory / "summary.json").write_text(
        json.dumps(
            {
                "schema_version": "md_phase_normal_cache_collection_v1",
                "samples": 1,
            }
        )
    )


class MergeMdPhaseNormalCachesTest(unittest.TestCase):
    def test_merge_writes_deduplicated_summary(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            first = root / "first"
            second = root / "second"
            output = root / "merged"
            _write_collection(first, "system-a__silver_r00", "a:r00")
            _write_collection(second, "system-b__silver_r01", "b:r01")
            summary = merge_collections([first, second], output)
            self.assertEqual(summary["samples"], 2)
            self.assertEqual(summary["base_systems"], 2)
            self.assertEqual(summary["duplicate_records_removed"], 0)
            self.assertEqual(summary["phase_supervision_density"], 0.5)
            self.assertEqual(len(list(output.glob("*.npz"))), 2)

    def test_exact_duplicate_is_removed(self):
        record = {
            "sample_id": "sample",
            "transition_id": "transition",
            "sha256": "hash",
            "source_path": "one",
        }
        records, count = deduplicate_records([record, {**record, "source_path": "two"}])
        self.assertEqual(len(records), 1)
        self.assertEqual(count, 1)

    def test_conflicting_duplicate_fails(self):
        first = {
            "sample_id": "sample",
            "transition_id": "transition-a",
            "sha256": "hash-a",
            "source_path": "one",
        }
        second = {
            "sample_id": "sample",
            "transition_id": "transition-b",
            "sha256": "hash-b",
            "source_path": "two",
        }
        with self.assertRaisesRegex(ValueError, "Conflicting duplicate"):
            deduplicate_records([first, second])


if __name__ == "__main__":
    unittest.main()
