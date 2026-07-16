from __future__ import annotations

import argparse
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "build_md_context_matrix.py"
SPEC = importlib.util.spec_from_file_location("build_md_context_matrix", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def candidate(
    root: Path,
    name: str,
    *,
    apo_pdb_id: str | None = None,
    holo_pdb_id: str | None = None,
) -> dict:
    sample = root / "samples" / name
    sample.mkdir(parents=True)
    for filename in ("apo.pdb", "holo.pdb", "ligand.sdf"):
        (sample / filename).write_text("test\n")
    return {
        "transition_id": f"ahoj:{name}:pilot",
        "endpoints": {
            "apo_pdb_id": apo_pdb_id or f"APO_{name}",
            "holo_pdb_id": holo_pdb_id or f"HOLO_{name}",
            "apo_structure_path": str(sample / "apo.pdb"),
            "holo_structure_path": str(sample / "holo.pdb"),
        },
    }


class BuildMDContextMatrixTest(unittest.TestCase):
    def test_excludes_existing_contexts_and_preserves_candidate_index(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            candidates = [candidate(root, "one-A-LIG-1"), candidate(root, "two-A-LIG-1")]
            existing = {
                **candidates[0],
                "transition_id": "ahoj:one-A-LIG-1:context-npt-0",
            }
            candidate_manifest = root / "candidates.jsonl"
            context_manifest = root / "contexts.jsonl"
            candidate_manifest.write_text(
                "".join(json.dumps(record) + "\n" for record in candidates)
            )
            context_manifest.write_text(json.dumps(existing) + "\n")
            output_dir = root / "matrix"
            args = argparse.Namespace(
                candidate_manifest=candidate_manifest,
                existing_context_manifest=context_manifest,
                exclude_sample_list=None,
                output_dir=output_dir,
                seed_base=1000,
                protocol_tag="test_protocol",
            )
            summary = MODULE.build_matrix(args)
            row = json.loads((output_dir / "context_matrix.jsonl").read_text())
            self.assertEqual(summary["planned_systems"], 1)
            self.assertEqual(row["system_sample_id"], "two-A-LIG-1")
            self.assertEqual(row["candidate_index"], 1)
            self.assertEqual(row["seed"], 1001)
            self.assertEqual(row["protocol"]["setup"]["max_minimization_iterations"], 1000)

    def test_excludes_sample_list_without_existing_context_manifest(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            candidates = [candidate(root, "one-A-LIG-1"), candidate(root, "two-A-LIG-1")]
            candidate_manifest = root / "candidates.jsonl"
            candidate_manifest.write_text(
                "".join(json.dumps(record) + "\n" for record in candidates)
            )
            excluded = root / "excluded.txt"
            excluded.write_text("one-A-LIG-1\nabsent-A-LIG-1\n")
            output_dir = root / "matrix"
            args = argparse.Namespace(
                candidate_manifest=candidate_manifest,
                existing_context_manifest=None,
                exclude_sample_list=excluded,
                output_dir=output_dir,
                seed_base=2000,
                protocol_tag="test_protocol",
            )

            summary = MODULE.build_matrix(args)
            row = json.loads((output_dir / "context_matrix.jsonl").read_text())

            self.assertEqual(summary["excluded_requested_systems"], 2)
            self.assertEqual(summary["excluded_systems"], 1)
            self.assertEqual(summary["excluded_sample_ids"], ["one-A-LIG-1"])
            self.assertEqual(summary["excluded_not_in_candidates"], ["absent-A-LIG-1"])
            self.assertEqual(row["system_sample_id"], "two-A-LIG-1")
            self.assertEqual(row["candidate_index"], 1)
            self.assertEqual(row["seed"], 2001)

    def test_excludes_endpoint_pair_with_a_different_sample_id(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            prior = candidate(
                root,
                "old-A-LIG-1",
                apo_pdb_id="1ABC",
                holo_pdb_id="2DEF",
            )
            duplicate_pair = candidate(
                root,
                "new-B-ALT-2",
                apo_pdb_id="1abc",
                holo_pdb_id="2def",
            )
            independent = candidate(
                root,
                "new-C-LIG-3",
                apo_pdb_id="3GHI",
                holo_pdb_id="4JKL",
            )
            candidate_manifest = root / "candidates.jsonl"
            candidate_manifest.write_text(
                json.dumps(duplicate_pair) + "\n" + json.dumps(independent) + "\n"
            )
            prior_manifest = root / "prior.jsonl"
            prior_manifest.write_text(json.dumps(prior) + "\n")
            output_dir = root / "matrix"
            args = argparse.Namespace(
                candidate_manifest=candidate_manifest,
                existing_context_manifest=None,
                exclude_sample_list=None,
                exclude_candidate_manifest=[prior_manifest],
                output_dir=output_dir,
                seed_base=3000,
                protocol_tag="test_protocol",
            )

            summary = MODULE.build_matrix(args)
            row = json.loads((output_dir / "context_matrix.jsonl").read_text())

            self.assertEqual(summary["excluded_endpoint_pairs_requested"], 1)
            self.assertEqual(summary["excluded_endpoint_pair_systems"], 1)
            self.assertEqual(
                summary["excluded_endpoint_pair_sample_ids"], ["new-B-ALT-2"]
            )
            self.assertEqual(summary["planned_systems"], 1)
            self.assertEqual(row["system_sample_id"], "new-C-LIG-3")
            self.assertEqual(row["candidate_index"], 1)
            self.assertEqual(row["seed"], 3001)


if __name__ == "__main__":
    unittest.main()
