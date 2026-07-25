import unittest

from scripts.summarize_path4_gate0_pairs import METRICS, summarize_pairs
from src.data.openmm_gate0 import (
    OPENMM_GATE0_SCORE_SCHEMA_VERSION,
    assess_relaxed_frame_validity,
    relaxed_frame_validity_contract,
    summarize_energy_profile,
)


def _score(sample_id, scale, *, invalid_frames=()):
    times = [0.0, 0.25, 0.5, 0.75, 1.0]
    relaxed_energies = [9.0, 11.5 + 80.0 * scale, 14.0 + 160.0 * scale,
                        16.5 + 40.0 * scale, 19.0]
    invalid_frames = set(invalid_frames)
    validity_contract = relaxed_frame_validity_contract(500.0)
    frames = []
    for frame_index, (time, relaxed_energy) in enumerate(
        zip(times, relaxed_energies)
    ):
        residue_force = 800.0 if frame_index in invalid_frames else 100.0 * scale
        validity = assess_relaxed_frame_validity(
            residue_force,
            maximum_residue_net_force_kj_mol_nm=500.0,
        )
        frames.append(
            {
                "frame_index": frame_index,
                "time": time,
                "raw_potential_kj_mol": 10.0 + time * 10.0,
                "relaxed_potential_kj_mol": relaxed_energy,
                "protein_residue_net_force_max_kj_mol_nm": residue_force,
                "protein_ligand_severe_clash_pairs": 0,
                "protein_internal_severe_clash_pairs": 0,
                "relaxed_valid": validity["valid"],
                "relaxed_invalid_reasons": validity["invalid_reasons"],
                "relaxed_validity": validity,
                "minimization": {
                    "reporter_available": True,
                    "reporter_callback_count": 10,
                    "last_reported_iteration_index": 9,
                    "termination": "returned_from_local_energy_minimizer",
                    "termination_reason_available": False,
                    "maximum_iterations": 250,
                    "tolerance_kj_mol_nm": 25.0,
                    "unrestrained_potential_change_kj_mol": -1.0,
                },
            }
        )
    interior = frames[1:-1]
    invalid_reason_counts = {}
    for frame in interior:
        for reason in frame["relaxed_invalid_reasons"]:
            invalid_reason_counts[reason] = invalid_reason_counts.get(reason, 0) + 1
    relaxed_profile = summarize_energy_profile(
        times,
        relaxed_energies,
        interior_valid_mask=[frame["relaxed_valid"] for frame in frames],
    )
    return {
        "schema_version": OPENMM_GATE0_SCORE_SCHEMA_VERSION,
        "status": "completed",
        "sample_id": sample_id,
        "implicit_system_contract": {"prepared_system": "same"},
        "contract": {
            "forcefield": "same",
            "frame_initialization": "reference_cache",
            "frame_reference_cache_sha256": "0" * 64,
            "relaxed_frame_validity": validity_contract,
        },
        "raw_energy_profile": {
            "endpoint_apo_energy_kj_mol": 10.0,
            "endpoint_holo_energy_kj_mol": 20.0,
            "interior_frames_total": 3,
            "interior_frames_included": 3,
            "interior_frames_excluded": 0,
            "interior_positive_excess_p95_kj_mol": 100.0 * scale,
            "interior_positive_excess_max_kj_mol": 200.0 * scale,
        },
        "relaxed_energy_profile": relaxed_profile,
        "relaxed_path": {
            "interior_frames": len(interior),
            "valid_interior_frames": sum(
                bool(frame["relaxed_valid"]) for frame in interior
            ),
            "invalid_interior_frames": sum(
                not bool(frame["relaxed_valid"]) for frame in interior
            ),
            "invalid_frame_fraction": sum(
                not bool(frame["relaxed_valid"]) for frame in interior
            ) / len(interior),
            "invalid_reason_counts": invalid_reason_counts,
            "invalid_or_severe_clash_frame_fraction": sum(
                not bool(frame["relaxed_valid"]) for frame in interior
            ) / len(interior),
            "severe_clash_frame_fraction": 0.0,
            "severe_clash_pairs_max": 0,
            "restraint_target_rms_p95_angstrom": 0.1 * scale,
            "protein_residue_net_force_p95_over_frames_kj_mol_nm": 10.0 * scale,
        },
        "frames": frames,
    }


class Path4Gate0PairSummaryTest(unittest.TestCase):
    def test_summarizes_lower_is_better_paired_metrics(self):
        path3 = {"a": _score("a", 1.0), "b": _score("b", 2.0)}
        candidate = {"a": _score("a", 0.5), "b": _score("b", 1.0)}
        optimizer = {
            "a": {
                "status": "completed",
                "sample_id": "a",
                "selected_improvement": True,
                "fallback_to_path3": False,
                "energy_force_calls": 10,
                "wall_seconds": 2.0,
                "correction": {},
            }
        }
        summary = summarize_pairs(
            path3,
            candidate,
            optimizer,
            bootstrap_resamples=100,
            seed=7,
        )
        self.assertEqual(summary["counts"]["paired_systems"], 2)
        self.assertEqual(set(summary["metrics"]), set(METRICS))
        metric = summary["metrics"]["raw_excess_p95_kj_mol"]
        self.assertEqual(metric["improvement_fraction"], 1.0)
        self.assertAlmostEqual(metric["mean_relative_improvement"], 0.5)
        self.assertEqual(summary["optimizer"]["total_energy_force_calls"], 10)
        self.assertEqual(
            summary["pair_contract"]["relaxed_energy_frame_policy"],
            "paired_valid_intersection",
        )

    def test_relaxed_metrics_use_paired_valid_frame_intersection(self):
        path3 = _score("a", 1.0, invalid_frames={1})
        candidate = _score("a", 0.5)

        summary = summarize_pairs(
            {"a": path3},
            {"a": candidate},
            {},
            bootstrap_resamples=10,
            seed=7,
        )

        row = summary["systems"][0]
        self.assertEqual(
            row["relaxed_frame_pairing"]["paired_valid_interior_frames"], 2
        )
        self.assertEqual(
            row["relaxed_frame_pairing"]["path3_invalid_interior_frames"], 1
        )
        self.assertEqual(
            row["relaxed_frame_pairing"]["candidate_invalid_interior_frames"], 0
        )
        invalid = row["metrics"]["relaxed_invalid_frame_fraction"]
        self.assertAlmostEqual(invalid["path3"], 1.0 / 3.0)
        self.assertEqual(invalid["candidate"], 0.0)
        self.assertTrue(invalid["available"])
        relaxed = row["metrics"]["relaxed_excess_max_kj_mol"]
        self.assertTrue(relaxed["available"])
        self.assertEqual(relaxed["path3"], 160.0)
        self.assertEqual(relaxed["candidate"], 80.0)

    def test_requires_at_least_one_pair(self):
        with self.assertRaisesRegex(ValueError, "No Path-3/Path-4"):
            summarize_pairs(
                {"a": _score("a", 1.0)},
                {"b": _score("b", 1.0)},
                {},
                bootstrap_resamples=10,
                seed=7,
            )

    def test_rejects_mismatched_scorer_contract(self):
        path3 = _score("a", 1.0)
        candidate = _score("a", 0.5)
        candidate["contract"]["forcefield"] = "different"

        with self.assertRaisesRegex(ValueError, "scorer contract mismatch"):
            summarize_pairs(
                {"a": path3},
                {"a": candidate},
                {},
                bootstrap_resamples=10,
                seed=7,
            )

    def test_rejects_endpoint_energy_mismatch(self):
        path3 = _score("a", 1.0)
        candidate = _score("a", 0.5)
        candidate["raw_energy_profile"]["endpoint_apo_energy_kj_mol"] += 2.0
        candidate["frames"][0]["raw_potential_kj_mol"] += 2.0

        with self.assertRaisesRegex(ValueError, "endpoint energy mismatch"):
            summarize_pairs(
                {"a": path3},
                {"a": candidate},
                {},
                bootstrap_resamples=10,
                seed=7,
                maximum_endpoint_energy_difference_kj_mol=1.0,
            )

    def test_rejects_mismatched_frame_grid(self):
        path3 = _score("a", 1.0)
        candidate = _score("a", 0.5)
        candidate["frames"][2]["time"] = 0.6

        with self.assertRaisesRegex(ValueError, "paired time-grid mismatch"):
            summarize_pairs(
                {"a": path3},
                {"a": candidate},
                {},
                bootstrap_resamples=10,
                seed=7,
            )


if __name__ == "__main__":
    unittest.main()
