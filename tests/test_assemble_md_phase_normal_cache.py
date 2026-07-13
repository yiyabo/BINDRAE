import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from scripts.assemble_md_phase_normal_cache import inspect_target


class AssembleMdPhaseNormalCacheTest(unittest.TestCase):
    def test_inspect_target_requires_passed_audit_and_schema(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            directory = Path(tmpdir)
            np.savez_compressed(
                directory / "sample.npz",
                schema_version=np.array("md_phase_normal_v1"),
                sample_id=np.array("sample"),
                transition_id=np.array("source:sample:replica"),
                n_residues=np.array(2, dtype=np.int32),
                t_values=np.array([0.0, 0.5, 1.0], dtype=np.float32),
                residual_valid_mask=np.array(
                    [[False, False], [True, False], [False, False]]
                ),
            )
            (directory / "target_audit.json").write_text(
                json.dumps(
                    {
                        "status": "md_phase_normal_targets_passed",
                        "passed": True,
                        "metrics": {"phase_supervision_density": 0.1},
                    }
                )
            )
            record = inspect_target(directory)
            self.assertEqual(record["sample_id"], "sample")
            self.assertEqual(record["n_frames"], 3)
            self.assertEqual(record["valid_residual_points"], 1)


if __name__ == "__main__":
    unittest.main()
