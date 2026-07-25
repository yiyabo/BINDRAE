import json
import tempfile
import unittest
from pathlib import Path

from scripts.compare_stage2_md_reference_results import compare_files


def write_result(path: Path, values):
    records = []
    for sample_id, reference_id, metric_value in values:
        records.append(
            {
                "sample_id": sample_id,
                "reference_id": reference_id,
                "metrics": {
                    "md_path_product_rmse": metric_value,
                    "phase_order_pair_accuracy": 1.0 - metric_value,
                },
            }
        )
    path.write_text(json.dumps({"records": records}))


class CompareStage2MdReferenceResultsTest(unittest.TestCase):
    def test_system_macro_pairing_and_directions(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            baseline = root / "baseline.json"
            candidate = root / "candidate.json"
            write_result(
                baseline,
                [
                    ("a", "r1", 2.0),
                    ("a", "r2", 4.0),
                    ("b", "r1", 1.0),
                ],
            )
            write_result(
                candidate,
                [
                    ("a", "r1", 1.0),
                    ("a", "r2", 3.0),
                    ("b", "r1", 1.5),
                ],
            )
            result = compare_files(
                baseline,
                candidate,
                ("md_path_product_rmse", "phase_order_pair_accuracy"),
                bootstrap_samples=100,
                bootstrap_seed=1,
            )

        rmse = result["comparisons"]["md_path_product_rmse"]
        self.assertEqual(rmse["systems"], 2)
        self.assertAlmostEqual(rmse["baseline_system_macro"], 2.0)
        self.assertAlmostEqual(rmse["candidate_system_macro"], 1.75)
        self.assertAlmostEqual(rmse["improvement_mean"], 0.25)
        self.assertEqual(rmse["direction"], "lower")

        accuracy = result["comparisons"]["phase_order_pair_accuracy"]
        self.assertAlmostEqual(accuracy["improvement_mean"], 0.25)
        self.assertEqual(accuracy["direction"], "higher")

    def test_misaligned_references_fail(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            baseline = root / "baseline.json"
            candidate = root / "candidate.json"
            write_result(baseline, [("a", "r1", 1.0)])
            write_result(candidate, [("a", "r2", 1.0)])
            with self.assertRaisesRegex(ValueError, "not aligned"):
                compare_files(
                    baseline,
                    candidate,
                    ("md_path_product_rmse",),
                    bootstrap_samples=10,
                    bootstrap_seed=1,
                )

    def test_noninferiority_margin_distinguishes_small_and_large_regressions(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            baseline = root / "baseline.json"
            small = root / "small.json"
            large = root / "large.json"
            write_result(baseline, [("a", "r1", 1.0)])
            write_result(small, [("a", "r1", 1.005)])
            write_result(large, [("a", "r1", 1.02)])
            small_result = compare_files(
                baseline,
                small,
                ("md_path_product_rmse",),
                bootstrap_samples=10,
                bootstrap_seed=1,
                noninferiority_margin_percent=1.0,
            )
            large_result = compare_files(
                baseline,
                large,
                ("md_path_product_rmse",),
                bootstrap_samples=10,
                bootstrap_seed=1,
                noninferiority_margin_percent=1.0,
            )

        self.assertTrue(
            small_result["comparisons"]["md_path_product_rmse"][
                "noninferiority_passes"
            ]
        )
        self.assertFalse(
            large_result["comparisons"]["md_path_product_rmse"][
                "noninferiority_passes"
            ]
        )


if __name__ == "__main__":
    unittest.main()
