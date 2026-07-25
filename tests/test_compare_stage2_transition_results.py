import json
import tempfile
import unittest
from pathlib import Path

from scripts.compare_stage2_transition_results import compare_files


class CompareStage2TransitionResultsTest(unittest.TestCase):
    def test_paired_system_comparison_uses_lower_is_better_direction(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            baseline = root / "baseline.json"
            candidate = root / "candidate.json"
            baseline.write_text(
                json.dumps(
                    {
                        "per_sample_metrics": [
                            {
                                "sample_id": "a",
                                "metrics": {"all_pocket/ligand_clash_severity": 2.0},
                            },
                            {
                                "sample_id": "b",
                                "metrics": {"all_pocket/ligand_clash_severity": 4.0},
                            },
                        ]
                    }
                )
            )
            candidate.write_text(
                json.dumps(
                    {
                        "per_sample_metrics": [
                            {
                                "sample_id": "a",
                                "metrics": {"all_pocket/ligand_clash_severity": 1.0},
                            },
                            {
                                "sample_id": "b",
                                "metrics": {"all_pocket/ligand_clash_severity": 3.0},
                            },
                        ]
                    }
                )
            )

            result = compare_files(
                baseline,
                candidate,
                ["all_pocket/ligand_clash_severity"],
                bootstrap_samples=1000,
                bootstrap_seed=7,
            )

        comparison = result["comparisons"]["all_pocket/ligand_clash_severity"]
        self.assertEqual(comparison["systems"], 2)
        self.assertAlmostEqual(comparison["improvement_mean"], 1.0)
        self.assertAlmostEqual(comparison["candidate_win_fraction"], 1.0)


if __name__ == "__main__":
    unittest.main()
