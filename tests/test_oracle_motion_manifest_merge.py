import contextlib
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from scripts import build_oracle_motion_remainder


def _write_minimal_cache(path: Path, sample_id: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        path,
        sample_id=np.asarray(sample_id),
        n_residues=np.asarray(3, dtype=np.int32),
    )


class OracleMotionManifestMergeTest(unittest.TestCase):
    def test_merge_uses_all_sources_and_writes_resolvable_relative_paths(self):
        with tempfile.TemporaryDirectory() as raw_tmp:
            tmp_path = Path(raw_tmp)
            shard0 = tmp_path / "shard0"
            shard1 = tmp_path / "shard1"
            _write_minimal_cache(shard0 / "sample-a.npz", "sample-a")
            _write_minimal_cache(shard1 / "sample-b.npz", "sample-b")

            candidates = tmp_path / "candidates.txt"
            candidates.write_text("sample-a\nsample-b\nsample-c\n")
            output_dir = tmp_path / "merged"
            argv = [
                "build_oracle_motion_remainder.py",
                "--candidate_list",
                str(candidates),
                "--partial_cache_dir",
                str(shard0),
                "--extra_cache_dir",
                str(shard1),
                "--remainder_out",
                str(output_dir / "remainder.txt"),
                "--rejects_out",
                str(output_dir / "rejects.jsonl"),
                "--merged_manifest_out",
                str(output_dir / "manifest.json"),
                "--skip_ligand_check",
                "--fast_manifest",
            ]
            stdout = io.StringIO()
            with mock.patch.object(sys, "argv", argv), contextlib.redirect_stdout(stdout):
                build_oracle_motion_remainder.main()

            summary = json.loads(stdout.getvalue())
            self.assertEqual(summary["completed_unique"], 2)
            self.assertEqual(summary["remainder"], 1)
            self.assertEqual(
                (output_dir / "remainder.txt").read_text().splitlines(),
                ["sample-c"],
            )

            manifest = json.loads((output_dir / "manifest.json").read_text())
            self.assertEqual(manifest["num_samples"], 2)
            self.assertEqual(
                {record["sample_id"] for record in manifest["records"]},
                {"sample-a", "sample-b"},
            )
            for record in manifest["records"]:
                relative_path = output_dir / record["relative_path"]
                self.assertEqual(relative_path.resolve(), Path(record["path"]).resolve())
                self.assertTrue(relative_path.exists())


if __name__ == "__main__":
    unittest.main()
