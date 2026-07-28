import json
import tempfile
import unittest
from pathlib import Path

from scripts.consolidate_md_replica_corpus import (
    sha256,
    validate_consensus_summary,
    validate_finalization_summary,
    write_immutable_json,
)


class ConsolidateMdReplicaCorpusTest(unittest.TestCase):
    def finalization_summary(self):
        return {
            "schema_version": "bindrae_md_replica_finalization_v1",
            "planned_replicas": 4,
            "passed_targets": 2,
            "outcome_counts": {"failed_pull": 2, "target_passed": 2},
            "per_system": {
                "system_a": {"target_passed": 2},
                "system_b": {"failed_pull": 2},
            },
        }

    def test_finalization_requires_complete_consistent_outcomes(self):
        audit = validate_finalization_summary(
            self.finalization_summary(), min_replicas=2
        )
        self.assertEqual(audit["eligible_systems"], ["system_a"])
        self.assertEqual(audit["passed_targets"], 2)

        incomplete = self.finalization_summary()
        incomplete["outcome_counts"] = {
            "incomplete_missing": 2,
            "target_passed": 2,
        }
        incomplete["per_system"]["system_b"] = {"incomplete_missing": 2}
        with self.assertRaisesRegex(ValueError, "incomplete outcomes"):
            validate_finalization_summary(incomplete, min_replicas=2)

        inconsistent = self.finalization_summary()
        inconsistent["per_system"]["system_b"] = {"failed_path_audit": 2}
        with self.assertRaisesRegex(ValueError, "do not match outcome_counts"):
            validate_finalization_summary(inconsistent, min_replicas=2)

    def test_existing_consensus_is_hash_and_membership_checked(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            input_cache = root / "phase_normal_cache"
            output_cache = root / "consensus_cache"
            input_cache.mkdir()
            output_cache.mkdir()
            target = output_cache / "system_a.npz"
            target.write_bytes(b"consensus")
            (output_cache / "manifest.jsonl").write_text(
                json.dumps(
                    {
                        "sample_id": "system_a",
                        "relative_path": target.name,
                        "sha256": sha256(target),
                    }
                )
                + "\n"
            )
            summary_path = output_cache / "summary.json"
            summary_path.write_text(
                json.dumps(
                    {
                        "schema_version": "md_phase_normal_consensus_v1",
                        "input_cache": str(input_cache),
                        "output_cache": str(output_cache),
                        "min_replicas": 2,
                        "min_support_fraction": 0.5,
                        "consensus_systems": 1,
                    }
                )
            )
            validate_consensus_summary(
                summary_path,
                input_cache=input_cache,
                output_cache=output_cache,
                expected_system_ids={"system_a"},
                min_replicas=2,
                min_support_fraction=0.5,
            )
            with self.assertRaisesRegex(ValueError, "eligible systems"):
                validate_consensus_summary(
                    summary_path,
                    input_cache=input_cache,
                    output_cache=output_cache,
                    expected_system_ids={"system_b"},
                    min_replicas=2,
                    min_support_fraction=0.5,
                )
            target.write_bytes(b"corrupt")
            with self.assertRaisesRegex(ValueError, "hash mismatch"):
                validate_consensus_summary(
                    summary_path,
                    input_cache=input_cache,
                    output_cache=output_cache,
                    expected_system_ids={"system_a"},
                    min_replicas=2,
                    min_support_fraction=0.5,
                )

    def test_state_output_is_immutable_and_idempotent(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "state.json"
            self.assertEqual(
                write_immutable_json(path, {"status": "complete"}), "written"
            )
            self.assertEqual(
                write_immutable_json(path, {"status": "complete"}), "reused"
            )
            with self.assertRaisesRegex(FileExistsError, "different state"):
                write_immutable_json(path, {"status": "changed"})


if __name__ == "__main__":
    unittest.main()
