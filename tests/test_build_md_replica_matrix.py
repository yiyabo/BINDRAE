from __future__ import annotations

import argparse
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "build_md_replica_matrix.py"
SPEC = importlib.util.spec_from_file_location("build_md_replica_matrix", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class BuildMDReplicaMatrixTest(unittest.TestCase):
    def test_builds_unique_replica_ids_and_seeds(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            sample = root / "samples" / "protein-A-LIG-1"
            sample.mkdir(parents=True)
            (sample / "apo.pdb").write_text("END\n")
            (sample / "holo.pdb").write_text("END\n")
            system_dir = root / "setup"
            npt_dir = root / "npt"
            system_dir.mkdir()
            npt_dir.mkdir()
            (system_dir / "preparation_report.json").write_text("{}\n")
            (npt_dir / "npt_report.json").write_text(
                json.dumps({"status": "npt_smoke_passed", "system_dir": str(system_dir)})
            )
            candidate = {
                "transition_id": "ahoj:protein-A-LIG-1:pilot",
                "endpoints": {
                    "apo_structure_path": str(sample / "apo.pdb"),
                    "holo_structure_path": str(sample / "holo.pdb"),
                },
            }
            context = {
                **candidate,
                "trajectory": {"topology_path": str(npt_dir / "final_npt.pdb")},
            }
            candidate_manifest = root / "candidates.jsonl"
            context_manifest = root / "contexts.jsonl"
            candidate_manifest.write_text(json.dumps(candidate) + "\n")
            context_manifest.write_text(json.dumps(context) + "\n")
            output_dir = root / "matrix"
            args = argparse.Namespace(
                candidate_manifest=candidate_manifest,
                context_manifest=context_manifest,
                output_dir=output_dir,
                replica_start=1,
                replica_stop=4,
                seed_base=1000,
                protocol_tag="test_protocol",
                pre_equilibration_steps=500,
                pulling_steps=10000,
                endpoint_hold_steps=2000,
                report_interval=100,
                rmsd_k_kj_mol_nm2=200000.0,
                final_target_rmsd_nm=0.025,
                min_mapping_fraction=0.95,
            )
            summary = MODULE.build_matrix(args)
            rows = [
                json.loads(line)
                for line in (output_dir / "replica_matrix.jsonl").read_text().splitlines()
            ]
            self.assertEqual(summary["tasks"], 4)
            self.assertEqual([row["seed"] for row in rows], [1001, 1002, 1003, 1004])
            self.assertEqual(len({row["transition_id"] for row in rows}), 4)
            self.assertTrue(all(row["protocol"]["resample_initial_velocities"] for row in rows))
            self.assertTrue(
                all(row["protocol"]["min_mapping_fraction"] == 0.95 for row in rows)
            )


if __name__ == "__main__":
    unittest.main()
