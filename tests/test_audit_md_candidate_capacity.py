from __future__ import annotations

import argparse
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


SCRIPT_PATH = (
    Path(__file__).resolve().parents[1] / "scripts" / "audit_md_candidate_capacity.py"
)
SPEC = importlib.util.spec_from_file_location("audit_md_candidate_capacity", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def candidate(
    root: Path,
    sample_id: str,
    apo_pdb: str,
    holo_pdb: str,
    *,
    apo_residues: int,
    holo_residues: int,
    mapping_fraction: float = 1.0,
    selection_rank: int = 1,
) -> dict:
    sample_dir = root / "samples" / sample_id
    return {
        "transition_id": f"ahoj:{sample_id}:pilot",
        "endpoints": {
            "apo_pdb_id": apo_pdb,
            "holo_pdb_id": holo_pdb,
            "apo_structure_path": str(sample_dir / "apo.pdb"),
            "holo_structure_path": str(sample_dir / "holo.pdb"),
        },
        "screening": {
            "apo_n_residues": apo_residues,
            "holo_n_residues": holo_residues,
            "residue_mapping_fraction": mapping_fraction,
            "selection_rank": selection_rank,
        },
    }


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))


class AuditMDCandidateCapacityTest(unittest.TestCase):
    def test_filters_holdout_groups_and_attempted_endpoint_pairs(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            rows = [
                candidate(
                    root,
                    "sample-a",
                    "1aaa",
                    "2aaa",
                    apo_residues=100,
                    holo_residues=96,
                    mapping_fraction=0.96,
                    selection_rank=4,
                ),
                candidate(
                    root,
                    "sample-b",
                    "3bbb",
                    "4bbb",
                    apo_residues=120,
                    holo_residues=120,
                    selection_rank=2,
                ),
                candidate(
                    root,
                    "sample-c",
                    "5ccc",
                    "6ccc",
                    apo_residues=140,
                    holo_residues=138,
                    selection_rank=3,
                ),
                candidate(
                    root,
                    "sample-d",
                    "7ddd",
                    "8ddd",
                    apo_residues=160,
                    holo_residues=160,
                    selection_rank=1,
                ),
            ]
            candidate_manifest = root / "candidates.jsonl"
            write_jsonl(candidate_manifest, rows)
            clean_list = root / "clean.txt"
            clean_list.write_text("sample-a\nsample-b\nsample-d\n")
            leakage_report = root / "leakage.json"
            leakage_report.write_text(
                json.dumps(
                    {
                        "counts": {"retained": 3},
                        "excluded": [
                            {
                                "sample_id": "sample-c",
                                "reasons": ["holdout_protein_family"],
                            }
                        ],
                    }
                )
            )
            attempted = root / "attempted.jsonl"
            write_jsonl(
                attempted,
                [
                    candidate(
                        root,
                        "historical-b",
                        "3BBB",
                        "4BBB",
                        apo_residues=120,
                        holo_residues=120,
                    )
                ],
            )
            output_dir = root / "audit"
            args = argparse.Namespace(
                candidate_manifest=candidate_manifest,
                leakage_clean_sample_list=clean_list,
                leakage_report=leakage_report,
                attempted_candidate_manifest=[attempted],
                output_dir=output_dir,
                minimum_planning_pool=2,
                mismatch_smoke_count=1,
            )

            summary = MODULE.audit_capacity(args)

            self.assertEqual(summary["counts"]["novel_candidates"], 2)
            self.assertEqual(summary["counts"]["planning_pool_shortfall"], 0)
            self.assertEqual(summary["counts"]["mismatched_novel_candidates"], 1)
            self.assertTrue(summary["gate"]["minimum_planning_pool_passed"])
            self.assertEqual(
                summary["exclusion_reason_counts"],
                {
                    "holdout_protein_family": 1,
                    "previously_attempted_endpoint_pair": 1,
                },
            )
            self.assertEqual(
                (output_dir / "novel_sample_ids.txt").read_text().splitlines(),
                ["sample-a", "sample-d"],
            )
            self.assertEqual(
                (output_dir / "mismatched_residue_smoke_sample_ids.txt")
                .read_text()
                .splitlines(),
                ["sample-a"],
            )

    def test_rejects_duplicate_endpoint_pairs(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            rows = [
                candidate(
                    root,
                    "sample-a",
                    "1aaa",
                    "2aaa",
                    apo_residues=100,
                    holo_residues=100,
                ),
                candidate(
                    root,
                    "sample-b",
                    "1AAA",
                    "2AAA",
                    apo_residues=100,
                    holo_residues=100,
                ),
            ]
            with self.assertRaisesRegex(ValueError, "repeats endpoint pairs"):
                MODULE._validate_unique_candidates(rows)


if __name__ == "__main__":
    unittest.main()
