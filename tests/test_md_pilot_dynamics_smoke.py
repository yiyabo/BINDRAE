from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path

import numpy as np


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "run_md_pilot_dynamics_smoke.py"
SPEC = importlib.util.spec_from_file_location("run_md_pilot_dynamics_smoke", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class MDPilotDynamicsSmokeTest(unittest.TestCase):
    def test_parse_temperatures(self):
        self.assertEqual(MODULE.parse_temperatures("50, 100,300"), [50.0, 100.0, 300.0])
        with self.assertRaises(ValueError):
            MODULE.parse_temperatures("0,300")

    def test_kabsch_transform_recovers_rigid_motion(self):
        points = np.asarray(
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 3.0]]
        )
        rotation = np.asarray([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
        target = points @ rotation + np.asarray([4.0, -2.0, 7.0])
        recovered_rotation, translation = MODULE.kabsch_transform(points, target)
        np.testing.assert_allclose(points @ recovered_rotation + translation, target, atol=1e-10)

    def test_closest_periodic_image_removes_lattice_translation(self):
        box = np.diag([50.0, 60.0, 70.0])
        reference = np.asarray([[1.0, 2.0, 3.0], [2.0, 2.0, 3.0]])
        wrapped = reference + box[0] - box[2]
        corrected = MODULE.closest_periodic_image(wrapped, reference, box)
        np.testing.assert_allclose(corrected, reference, atol=1e-10)


if __name__ == "__main__":
    unittest.main()
