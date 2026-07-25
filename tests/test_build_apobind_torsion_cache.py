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
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "build_apobind_torsion_cache.py"
)
SPEC = importlib.util.spec_from_file_location("build_apobind_torsion_cache", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def torsion_payload(
    *, chain: str = "A", names: tuple[str, ...] = ("ALA", "GLY")
) -> dict[str, np.ndarray | int]:
    n_residues = len(names)
    return {
        "phi": np.zeros(n_residues, dtype=np.float32),
        "psi": np.zeros(n_residues, dtype=np.float32),
        "omega": np.full(n_residues, np.pi, dtype=np.float32),
        "chi": np.zeros((n_residues, 4), dtype=np.float32),
        "bb_mask": np.ones(n_residues, dtype=bool),
        "chi_mask": np.zeros((n_residues, 4), dtype=bool),
        "omega_cis_trans": np.zeros(n_residues, dtype=np.int8),
        "n_residues": n_residues,
        "residue_keys": np.asarray(
            [f"{chain}|{index}|" for index in range(1, n_residues + 1)]
        ),
        "residue_names": np.asarray(names),
        "sequence_str": np.asarray("AG"[:n_residues]),
        "residue_alignment_version": np.asarray(
            "canonical_residue_key_v2_esm_compatible"
        ),
    }


def matrix_row(index: int, replica_index: int, system_id: str = "apobind_test") -> dict:
    return {
        "schema_version": "bindrae_md_replica_matrix_v1",
        "matrix_index": index,
        "system_sample_id": system_id,
        "sample_id": f"{system_id}__silver_r{replica_index:02d}",
        "replica_index": replica_index,
        "protocol": {"min_mapping_fraction": 0.95},
    }


class BuildApobindTorsionCacheTest(unittest.TestCase):
    def test_validates_current_torsion_schema(self):
        keys, names = MODULE.validate_torsion_payload(
            torsion_payload(), label="fixture"
        )
        self.assertEqual(keys, [("A", 1, ""), ("A", 2, "")])
        self.assertEqual(names, ["ALA", "GLY"])

    def test_rejects_stale_alignment_version(self):
        payload = torsion_payload()
        payload["residue_alignment_version"] = np.asarray("legacy")
        with self.assertRaisesRegex(ValueError, "residue_alignment_version"):
            MODULE.validate_torsion_payload(payload, label="fixture")

    def test_matrix_contract_freezes_replica_identity_and_mapping_gate(self):
        rows = [matrix_row(0, 1), matrix_row(1, 2)]
        systems = MODULE.ordered_matrix_systems(
            rows,
            expected_systems=1,
            expected_replicas_per_system=2,
            min_pair_mapping_fraction=0.95,
        )
        self.assertEqual(systems, ["apobind_test"])

        rows[1]["protocol"]["min_mapping_fraction"] = 0.90
        with self.assertRaisesRegex(ValueError, "frozen value"):
            MODULE.ordered_matrix_systems(
                rows,
                expected_systems=1,
                expected_replicas_per_system=2,
                min_pair_mapping_fraction=0.95,
            )

    def test_run_builds_both_endpoint_caches_and_hash_audit(self):
        class FakeExtractor:
            def extract(self, path: Path):
                chain = "A" if path.name == "apo.pdb" else "B"
                return torsion_payload(chain=chain)

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            data_dir = root / "apobind"
            sample_dir = data_dir / "samples" / "apobind_test"
            sample_dir.mkdir(parents=True)
            (sample_dir / "apo.pdb").write_text("APO\n")
            (sample_dir / "holo.pdb").write_text("HOLO\n")
            matrix = root / "matrix.jsonl"
            matrix.write_text(
                "".join(
                    json.dumps(row) + "\n"
                    for row in (matrix_row(0, 1), matrix_row(1, 2))
                )
            )
            report_path = root / "audit.json"
            args = argparse.Namespace(
                matrix=matrix,
                data_dir=data_dir,
                output_report=report_path,
                expected_systems=1,
                expected_replicas_per_system=2,
                min_pair_mapping_fraction=0.95,
            )

            with mock.patch.object(MODULE, "TorsionExtractor", FakeExtractor):
                report = MODULE.run(args)

            self.assertEqual(report["counts"]["created"], 2)
            self.assertEqual(report["counts"]["endpoint_caches"], 2)
            self.assertEqual(
                report["systems"][0]["pair_identity"]["symmetric_mapping_fraction"],
                1.0,
            )
            self.assertTrue((sample_dir / "torsion_apo.npz").is_file())
            self.assertTrue((sample_dir / "torsion_holo.npz").is_file())
            self.assertEqual(json.loads(report_path.read_text())["status"], "complete")

    def test_existing_different_cache_is_not_overwritten(self):
        class FakeExtractor:
            def extract(self, path: Path):
                return torsion_payload()

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            pdb_path = root / "apo.pdb"
            cache_path = root / "torsion_apo.npz"
            pdb_path.write_text("APO\n")
            stale = torsion_payload()
            stale["phi"] = np.ones(2, dtype=np.float32)
            np.savez_compressed(cache_path, **stale)

            with self.assertRaisesRegex(FileExistsError, "differs"):
                MODULE.build_endpoint_cache(
                    extractor=FakeExtractor(),
                    pdb_path=pdb_path,
                    cache_path=cache_path,
                )
            with np.load(cache_path, allow_pickle=False) as data:
                np.testing.assert_array_equal(data["phi"], np.ones(2))


if __name__ == "__main__":
    unittest.main()
