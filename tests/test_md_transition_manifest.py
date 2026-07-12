import json
import tempfile
import unittest
from pathlib import Path

from src.data.md_transition_manifest import (
    SCHEMA_VERSION,
    audit_transition_manifest,
    heldout_benchmark_eligible,
    load_transition_manifest,
    phase_supervision_eligible,
    validate_transition_record,
)


def make_record(**overrides):
    record = {
        "schema_version": SCHEMA_VERSION,
        "transition_id": "source:system:replica-0",
        "ensemble_id": "source:system",
        "status": "candidate",
        "source": {
            "name": "source",
            "record_url": "https://example.org/record",
            "license": "CC-BY-4.0",
        },
        "evidence": {
            "tier": "bronze_modeled_path",
            "contains_endpoint_transition": True,
            "biased_sampling": True,
            "physical_time_interpretable": False,
        },
        "protein": {
            "uniprot_id": None,
            "chain_ids": ["A"],
            "sequence_sha256": None,
        },
        "ligand": {"comp_id": None, "inchikey": None},
        "endpoints": {
            "apo_pdb_id": "1ABC",
            "holo_pdb_id": "2ABC",
            "apo_structure_path": None,
            "holo_structure_path": None,
        },
        "trajectory": {
            "topology_path": None,
            "coordinate_paths": [],
            "n_frames": None,
            "frame_interval_ps": None,
        },
        "usage": {
            "phase_supervision": False,
            "heldout_benchmark": False,
            "kinetics_claims": False,
        },
        "split": {
            "name": "unassigned",
            "family_group": None,
            "ligand_scaffold_group": None,
        },
        "quality": {
            "endpoint_mapping_verified": False,
            "residue_mapping_fraction": None,
            "transition_verified": False,
            "notes": [],
        },
    }
    record.update(overrides)
    return record


class MDTransitionManifestTest(unittest.TestCase):
    def test_metadata_candidate_is_valid_without_local_files(self):
        issues = validate_transition_record(make_record())
        self.assertEqual(issues, [])

    def test_prepared_phase_record_passes_file_and_usage_gates(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            (root / "trajectory.xtc").write_bytes(b"trajectory")
            record = make_record(
                status="prepared",
                trajectory={
                    "topology_path": None,
                    "coordinate_paths": ["trajectory.xtc"],
                    "n_frames": 100,
                    "frame_interval_ps": 10.0,
                },
                usage={
                    "phase_supervision": True,
                    "heldout_benchmark": False,
                    "kinetics_claims": False,
                },
                split={
                    "name": "train",
                    "family_group": "family:1",
                    "ligand_scaffold_group": "scaffold:1",
                },
                quality={
                    "endpoint_mapping_verified": True,
                    "residue_mapping_fraction": 0.98,
                    "transition_verified": True,
                    "notes": [],
                },
            )
            issues = validate_transition_record(
                record,
                base_dir=root,
                check_files=True,
            )
            self.assertEqual(issues, [])
            self.assertTrue(phase_supervision_eligible(record))
            self.assertFalse(heldout_benchmark_eligible(record))

    def test_modeled_path_cannot_enter_headline_benchmark(self):
        record = make_record(
            status="prepared",
            trajectory={
                "topology_path": None,
                "coordinate_paths": ["trajectory.xtc"],
                "n_frames": 100,
                "frame_interval_ps": None,
            },
            usage={
                "phase_supervision": False,
                "heldout_benchmark": True,
                "kinetics_claims": False,
            },
            split={
                "name": "test",
                "family_group": "family:1",
                "ligand_scaffold_group": "scaffold:1",
            },
            quality={
                "endpoint_mapping_verified": True,
                "residue_mapping_fraction": 1.0,
                "transition_verified": True,
                "notes": [],
            },
        )
        codes = {issue.code for issue in validate_transition_record(record)}
        self.assertIn("benchmark_tier", codes)
        self.assertFalse(heldout_benchmark_eligible(record))

    def test_loader_preserves_parse_errors_and_audit_finds_duplicates(self):
        first = make_record()
        second = make_record()
        with tempfile.TemporaryDirectory() as tmpdir:
            manifest = Path(tmpdir) / "manifest.jsonl"
            with manifest.open("w", encoding="utf-8") as handle:
                handle.write(json.dumps(first) + "\n")
                handle.write("{not-json}\n")
                handle.write(json.dumps(second) + "\n")

            records, parse_issues = load_transition_manifest(manifest)
            issues, summary = audit_transition_manifest(
                records,
                initial_issues=parse_issues,
            )

        codes = {issue.code for issue in issues}
        self.assertEqual(len(records), 2)
        self.assertIn("invalid_json", codes)
        self.assertIn("duplicate_transition_id", codes)
        self.assertEqual(summary["num_records"], 2)
        self.assertEqual(summary["tier_counts"], {"bronze_modeled_path": 2})
        self.assertEqual(summary["num_errors"], 2)

    def test_audit_rejects_ensemble_and_family_test_leakage(self):
        train_record = make_record(
            transition_id="source:system:train",
            split={
                "name": "train",
                "family_group": "family:shared",
                "ligand_scaffold_group": "scaffold:train",
            },
        )
        test_record = make_record(
            transition_id="source:system:test",
            split={
                "name": "test",
                "family_group": "family:shared",
                "ligand_scaffold_group": "scaffold:test",
            },
        )
        issues, summary = audit_transition_manifest([train_record, test_record])
        codes = {issue.code for issue in issues}
        self.assertIn("ensemble_id_split_leakage", codes)
        self.assertIn("family_group_split_leakage", codes)
        self.assertEqual(summary["num_errors"], 2)


if __name__ == "__main__":
    unittest.main()
