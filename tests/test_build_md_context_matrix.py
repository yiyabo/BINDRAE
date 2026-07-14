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


def candidate(root: Path, name: str) -> dict:
    sample = root / "samples" / name
    sample.mkdir(parents=True)
    for filename in ("apo.pdb", "holo.pdb", "ligand.sdf"):
        (sample / filename).write_text("test\n")
    return {
        "transition_id": f"ahoj:{name}:pilot",
        "endpoints": {
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


if __name__ == "__main__":
    unittest.main()
