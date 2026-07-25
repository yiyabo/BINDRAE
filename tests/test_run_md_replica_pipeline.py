from __future__ import annotations

import argparse
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np


SCRIPT_PATH = (
    Path(__file__).resolve().parents[1] / "scripts" / "run_md_replica_pipeline.py"
)
SPEC = importlib.util.spec_from_file_location("run_md_replica_pipeline", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def write_canonical_cache(data_dir: Path, sample_id: str = "sample-a") -> Path:
    sample_dir = data_dir / "samples" / sample_id
    sample_dir.mkdir(parents=True)
    cache = sample_dir / "torsion_apo.npz"
    np.savez_compressed(
        cache,
        residue_keys=np.asarray(["A|1|", "A|2|"]),
        residue_names=np.asarray(["ALA", "GLY"]),
        residue_alignment_version=np.asarray(
            "canonical_residue_key_v2_esm_compatible"
        ),
    )
    return cache


def write_matrix(root: Path) -> Path:
    row = {
        "matrix_index": 0,
        "sample_id": "sample-a__silver_r01",
        "system_sample_id": "sample-a",
        "transition_id": "apobind:sample-a:silver-pull-r01",
        "replica_index": 1,
        "seed": 123,
        "candidate_manifest": str(root / "candidates.jsonl"),
        "npt_dir": str(root / "npt"),
        "preparation_report": str(root / "preparation_report.json"),
        "pull_dir": str(root / "pull"),
        "target_dir": str(root / "target"),
        "protocol": {
            "pre_equilibration_steps": 500,
            "pulling_steps": 10000,
            "endpoint_hold_steps": 2000,
            "report_interval": 100,
            "rmsd_k_kj_mol_nm2": 200000.0,
            "final_target_rmsd_nm": 0.025,
            "min_mapping_fraction": 0.95,
            "resample_initial_velocities": True,
        },
    }
    matrix = root / "matrix.jsonl"
    matrix.write_text(json.dumps(row) + "\n")
    return matrix


def args_for(matrix: Path, data_dir: Path) -> argparse.Namespace:
    return argparse.Namespace(
        matrix=matrix,
        index=0,
        platform="CPU",
        cpu_threads=0,
        residual_envelope="sin2",
        normal_projection_mode="product",
        canonical_data_dir=data_dir,
        force=False,
    )


class RunMdReplicaPipelineTest(unittest.TestCase):
    def test_preflight_binds_replica_to_canonical_cache(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            data_dir = root / "data"
            cache = write_canonical_cache(data_dir)
            record = {
                "sample_id": "sample-a__silver_r01",
                "system_sample_id": "sample-a",
            }
            result = MODULE.preflight_canonical_cache(data_dir, record)
            self.assertEqual(result["cache"], str(cache.resolve()))
            self.assertEqual(result["n_residues"], 2)

            record["system_sample_id"] = "sample-b"
            with self.assertRaisesRegex(ValueError, "does not belong"):
                MODULE.preflight_canonical_cache(data_dir, record)

    def test_target_export_command_receives_explicit_canonical_data_dir(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            data_dir = root / "data"
            write_canonical_cache(data_dir)
            matrix = write_matrix(root)
            calls = []

            def capture(**kwargs):
                calls.append(kwargs)

            with mock.patch.object(MODULE, "run_command", side_effect=capture):
                state = MODULE.run_pipeline(args_for(matrix, data_dir))

            self.assertEqual(state["schema_version"], "bindrae_md_replica_pipeline_v2")
            self.assertEqual(state["status"], "completed")
            self.assertEqual(len(calls), 4)
            target_command = calls[-1]["command"]
            data_index = target_command.index("--data-dir")
            self.assertEqual(target_command[data_index + 1], str(data_dir.resolve()))
            self.assertEqual(
                state["stages"]["canonical_cache_preflight"]["status"], "passed"
            )

    def test_missing_cache_is_recorded_before_any_stage_runs(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            data_dir = root / "data"
            matrix = write_matrix(root)
            with self.assertRaises(FileNotFoundError):
                MODULE.run_pipeline(args_for(matrix, data_dir))

            status = json.loads((root / "pull" / "pipeline_status.json").read_text())
            self.assertEqual(status["failed_stage"], "canonical_cache_preflight")
            self.assertEqual(status["stages"]["canonical_cache_preflight"]["status"], "failed")


if __name__ == "__main__":
    unittest.main()
