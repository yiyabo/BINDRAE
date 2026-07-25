from __future__ import annotations

import argparse
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


SCRIPT_PATH = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "select_apobind_structure_smoke.py"
)
SPEC = importlib.util.spec_from_file_location(
    "select_apobind_structure_smoke", SCRIPT_PATH
)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def make_record(
    index: int,
    *,
    apo_pdb: str | None = None,
    holo_pdb: str | None = None,
    rmsd: float = 1.0,
    resolution: float = 2.0,
    site_size: int = 20,
    identity: float = 0.99,
    coverage: float = 0.99,
    apo_chains: list[str] | None = None,
) -> dict:
    apo = apo_pdb or f"A{index:03d}"[-4:]
    holo = holo_pdb or f"H{index:03d}"[-4:]
    indices = [str(value) for value in range(1, site_size + 1)]
    residues = ["A"] * site_size
    return {
        "schema_version": "bindrae_external_endpoint_index_audit_v1",
        "record_key": f"apobind:{index}:{apo}:{holo}",
        "source": "apobind",
        "source_rows": [index + 2],
        "reference_pair_match": {"match_level": "none"},
        "endpoints": [
            {"role": "apo", "pdb_id": apo, "chains": apo_chains or ["A"]},
            {"role": "holo", "pdb_id": holo, "chains": ["A"]},
        ],
        "metadata": {
            "sequence_identity": identity,
            "sequence_coverage": coverage,
            "backbone_rmsd": rmsd,
            "tmscore": 0.9,
            "apo_resolution": resolution,
            "binding_site": {
                "apo_indices": indices,
                "holo_indices": indices,
                "apo_residues": residues,
                "holo_residues": residues,
            },
        },
    }


class SelectApobindStructureSmokeTest(unittest.TestCase):
    def test_strict_proxy_excludes_invalid_values_instead_of_clipping(self):
        valid = make_record(1)
        self.assertEqual(MODULE.strict_proxy_rejection_reasons(valid), [])

        invalid = make_record(2, identity=1.01, apo_chains=["A", "B"])
        reasons = MODULE.strict_proxy_rejection_reasons(invalid)
        self.assertIn("sequence_identity_outside_0_95_to_1", reasons)
        self.assertIn("not_single_chain_endpoints", reasons)

    def test_pair_deduplication_keeps_deterministic_representative(self):
        second = make_record(2, apo_pdb="1AAA", holo_pdb="2BBB")
        first = make_record(1, apo_pdb="2BBB", holo_pdb="1AAA")
        records, duplicate_count = MODULE.deduplicate_pairs([second, first])

        self.assertEqual(duplicate_count, 1)
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0]["record_key"], first["record_key"])

    def test_balanced_selection_is_deterministic_and_avoids_shared_pdbs(self):
        records = []
        for index in range(64):
            records.append(
                make_record(
                    index,
                    rmsd=0.4 + (index % 4) * 1.0 + index * 1e-4,
                    resolution=1.2 + ((index // 4) % 4) * 0.6 + index * 1e-4,
                    site_size=8 + ((index // 16) % 4) * 12 + index % 3,
                )
            )

        selected_a, edges_a = MODULE.select_balanced_records(
            records, count=32, seed="test"
        )
        selected_b, edges_b = MODULE.select_balanced_records(
            list(reversed(records)), count=32, seed="test"
        )

        self.assertEqual(edges_a, edges_b)
        self.assertEqual(
            [record["record_key"] for record in selected_a],
            [record["record_key"] for record in selected_b],
        )
        pdb_ids = [
            pdb
            for record in selected_a
            for pdb in MODULE.endpoint_ids(record)
        ]
        self.assertEqual(len(pdb_ids), len(set(pdb_ids)))
        self.assertFalse(
            any(
                record["smoke_selection"]["used_shared_endpoint_fallback"]
                for record in selected_a
            )
        )
        for field in MODULE.STRATUM_FIELDS:
            occupied = {
                record["smoke_selection"]["stratum"][field]
                for record in selected_a
            }
            self.assertEqual(occupied, {0, 1, 2, 3})

    def test_shared_endpoint_fallback_is_explicit(self):
        records = [
            make_record(index, apo_pdb="1AAA", holo_pdb=f"H{index:03d}"[-4:])
            for index in range(4)
        ]
        selected, _ = MODULE.select_balanced_records(records, count=3, seed="test")

        self.assertEqual(len(selected), 3)
        self.assertFalse(
            selected[0]["smoke_selection"]["used_shared_endpoint_fallback"]
        )
        self.assertTrue(
            selected[1]["smoke_selection"]["used_shared_endpoint_fallback"]
        )
        self.assertEqual(
            selected[1]["smoke_selection"]["shared_endpoint_pdb_ids"], ["1AAA"]
        )

    def test_run_writes_machine_readable_selection_artifacts(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            index_path = root / "index.jsonl"
            records = [make_record(index) for index in range(8)]
            index_path.write_text(
                "".join(json.dumps(record) + "\n" for record in records)
            )
            output_dir = root / "output"
            args = argparse.Namespace(
                index_jsonl=index_path,
                output_dir=output_dir,
                count=4,
                seed="test",
            )

            report = MODULE.run(args)

            self.assertEqual(report["counts"]["selected_systems"], 4)
            self.assertEqual(
                report["counts"]["selected_unique_endpoint_pdbs"], 8
            )
            self.assertTrue((output_dir / "selected_records.jsonl").exists())
            self.assertTrue((output_dir / "selected_records.csv").exists())
            saved_report = json.loads(
                (output_dir / "selection_report.json").read_text()
            )
            self.assertEqual(saved_report["schema_version"], MODULE.SCHEMA_VERSION)


if __name__ == "__main__":
    unittest.main()
