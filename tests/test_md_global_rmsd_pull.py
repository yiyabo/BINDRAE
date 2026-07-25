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

    def test_ca_mapping_handles_one_holo_insertion(self):
        mapping = MODULE.match_endpoint_topology_ca(
            ["ALA", "GLY", "SER", "THR"],
            ["ALA", "GLY", "ASP", "SER", "THR"],
            ["ALA", "GLY", "ASP", "SER", "THR"],
            [10, 20, 30, 40, 50],
            min_mapping_fraction=0.8,
        )

        np.testing.assert_array_equal(mapping["apo_indices"], [0, 1, 2, 3])
        np.testing.assert_array_equal(mapping["holo_indices"], [0, 1, 3, 4])
        np.testing.assert_array_equal(mapping["topology_ca_indices"], [10, 20, 40, 50])
        self.assertEqual(mapping["mapped_residues"], 4)
        self.assertAlmostEqual(mapping["mapping_fraction"], 0.8)

    def test_ca_mapping_enforces_frozen_coverage(self):
        with self.assertRaisesRegex(ValueError, "below 0.9500"):
            MODULE.match_endpoint_topology_ca(
                ["ALA", "GLY", "SER", "THR"],
                ["ALA", "GLY", "ASP", "SER", "THR"],
                ["ALA", "GLY", "ASP", "SER", "THR"],
                [10, 20, 30, 40, 50],
                min_mapping_fraction=0.95,
            )


if __name__ == "__main__":
    unittest.main()
