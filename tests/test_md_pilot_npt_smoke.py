from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "run_md_pilot_npt_smoke.py"
SPEC = importlib.util.spec_from_file_location("run_md_pilot_npt_smoke", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class MDPilotNPTSmokeTest(unittest.TestCase):
    def test_coefficient_of_variation(self):
        self.assertEqual(MODULE.coefficient_of_variation([2.0]), 0.0)
        self.assertAlmostEqual(MODULE.coefficient_of_variation([1.0, 1.0, 1.0]), 0.0)
        self.assertGreater(MODULE.coefficient_of_variation([1.0, 2.0, 3.0]), 0.0)

    def test_density_conversion(self):
        self.assertAlmostEqual(MODULE.density_g_ml(1000.0, 1.0), 1.66053906660)
        with self.assertRaises(ValueError):
            MODULE.density_g_ml(1000.0, 0.0)


if __name__ == "__main__":
    unittest.main()
