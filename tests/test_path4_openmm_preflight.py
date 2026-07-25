import unittest

from scripts.preflight_path4_openmm_gate0 import classify_failure


class Path4OpenmmPreflightTest(unittest.TestCase):
    def test_classifies_scientific_rejections_separately_from_engineering_failure(self):
        self.assertEqual(
            classify_failure(
                "candidate_topology_mapping",
                ValueError("Residue identity mismatch for A:1"),
            ),
            ("rejected", "residue_identity_mismatch"),
        )
        self.assertEqual(
            classify_failure(
                "reference_topology_state", ValueError("physical preflight failed")
            ),
            ("rejected", "reference_topology_physical_failure"),
        )
        self.assertEqual(
            classify_failure("implicit_system", RuntimeError("tool failure")),
            ("failed", "implicit_system_failure"),
        )


if __name__ == "__main__":
    unittest.main()
