from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path

import numpy as np


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "run_md_global_rmsd_pull.py"
SPEC = importlib.util.spec_from_file_location("run_md_global_rmsd_pull", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class MDGlobalRMSDPullTest(unittest.TestCase):
    def test_kabsch_rmsd_removes_rigid_transform(self):
        points = np.asarray(
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 3.0]]
        )
        rotation = np.asarray([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
        transformed = points @ rotation + np.asarray([4.0, -2.0, 7.0])
        self.assertLess(MODULE.kabsch_rmsd(points, transformed), 1e-10)

    def test_platform_properties_limit_cpu_threads(self):
        self.assertEqual(MODULE.platform_properties("CPU", 1), {"Threads": "1"})
        self.assertEqual(MODULE.platform_properties("CPU", 0), {})
        self.assertEqual(
            MODULE.platform_properties("CUDA", 0), {"Precision": "mixed"}
        )
        with self.assertRaisesRegex(ValueError, "cpu_threads must be >= 0"):
            MODULE.platform_properties("CPU", -1)


if __name__ == "__main__":
    unittest.main()
