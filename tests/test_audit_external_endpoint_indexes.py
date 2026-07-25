from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


SCRIPT_PATH = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "audit_external_endpoint_indexes.py"
)
SPEC = importlib.util.spec_from_file_location(
    "audit_external_endpoint_indexes", SCRIPT_PATH
)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def write_csv(path: Path, fieldnames: list[str], rows: list[dict], *, delimiter=","):
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter=delimiter)
        writer.writeheader()
        writer.writerows(rows)


class AuditExternalEndpointIndexesTest(unittest.TestCase):
    def test_pscdb_aggregates_components_but_not_distinct_source_records(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "pscdb.csv"
            fieldnames = [
                "PSCID",
                "FreeID",
                "BoundID",
                "Ligands",
                "Component No",
                "Type of motion",
                "Ligand binding",
                "RMSD",
                "Fixed segment",
                "Moving segment",
            ]
            write_csv(
                path,
                fieldnames,
                [
                    {
                        "PSCID": "CD.1",
                        "FreeID": "1aaa_AB",
                        "BoundID": "2bbb_AA",
                        "Ligands": "LIG",
                        "Component No": "1",
                        "Type of motion": "Domain",
                    },
                    {
                        "PSCID": "CD.1",
                        "FreeID": "1aaa_AB",
                        "BoundID": "2bbb_AA",
                        "Ligands": "LIG",
                        "Component No": "2",
                        "Type of motion": "Local",
                    },
                    {
                        "PSCID": "CD.2",
                        "FreeID": "1aaa_AB",
                        "BoundID": "2bbb_AA",
                        "Ligands": "OTHER",
                        "Component No": "1",
                        "Type of motion": "Local",
                    },
                ],
            )

            records = MODULE.load_pscdb(path)
            summary = MODULE._source_summary(records)

            self.assertEqual(len(records), 2)
            self.assertEqual(records[0]["source_rows"], [2, 3])
            self.assertEqual(len(records[0]["metadata"]["components"]), 2)
            self.assertEqual(records[0]["endpoints"][0]["chains"], ["A", "B"])
            self.assertEqual(records[0]["endpoints"][1]["chains"], ["A"])
            self.assertNotEqual(
                records[0]["site_identity"]["record_identity_key"],
                records[1]["site_identity"]["record_identity_key"],
            )
            self.assertEqual(summary["unique_directed_pdb_pairs"], 1)
            self.assertEqual(summary["records_beyond_unique_directed_pairs"], 1)

    def test_reference_matching_reports_forward_reverse_and_pair_only(self):
        records = [
            {
                "record_key": "source:forward",
                "source": "source",
                "source_rows": [2],
                "endpoints": [
                    {"pdb_id": "1AAA"},
                    {"pdb_id": "2BBB"},
                ],
                "site_identity": {
                    "exact_ligand_site_available": False,
                    "granularity": "pdb_pair_only",
                },
            },
            {
                "record_key": "source:reverse",
                "source": "source",
                "source_rows": [3],
                "endpoints": [
                    {"pdb_id": "4DDD"},
                    {"pdb_id": "3CCC"},
                ],
                "site_identity": {
                    "exact_ligand_site_available": False,
                    "granularity": "pdb_pair_only",
                },
            },
        ]
        reference = [
            {"reference_id": "r1", "pair": ("1AAA", "2BBB")},
            {"reference_id": "r2", "pair": ("3CCC", "4DDD")},
        ]

        report = MODULE.match_reference_pairs(records, reference)

        self.assertEqual(report["overall"]["directed_pdb_pair_matches"], 1)
        self.assertEqual(report["overall"]["reverse_pdb_pair_matches"], 1)
        self.assertEqual(report["overall"]["any_direction_pdb_pair_matches"], 2)
        self.assertEqual(
            report["overall"]["unique_undirected_pdb_pair_matches"], 2
        )
        self.assertIsNone(report["overall"]["exact_ligand_site_matches"])
        self.assertIsNone(report["overall"]["net_new_systems"])
        self.assertEqual(
            records[1]["reference_pair_match"]["match_level"], "pdb_pair_only"
        )

    def test_codnas_uses_maximum_tertiary_pair_without_moving_ligand_labels(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "codnas.csv"
            fieldnames = [
                "Cluster ID",
                "PDB_ID_query",
                "PDB_ID_target",
                "Query_Chain_ID",
                "Target_Chain_ID",
                "Query_ligands",
                "Target_ligands",
                "Biological_Assembly_query",
                "Biological_Assembly_target",
                "maxRMSD_T",
                "Group",
                "num_of_conformers",
                "conformers_ids",
            ]
            write_csv(
                path,
                fieldnames,
                [
                    {
                        "Cluster ID": "cluster-1",
                        "PDB_ID_query": "1aaa",
                        "PDB_ID_target": "2bbb",
                        "Query_Chain_ID": "3ccc_A",
                        "Target_Chain_ID": "4ddd_B",
                        "Query_ligands": "no/data",
                        "Target_ligands": "LIG ligand",
                        "Biological_Assembly_query": "1",
                        "Biological_Assembly_target": "2",
                        "maxRMSD_T": "4.2",
                        "Group": "TD",
                        "num_of_conformers": "3",
                        "conformers_ids": "1aaa|3ccc|4ddd",
                    }
                ],
                delimiter=";",
            )

            record = MODULE.load_codnas_q(path)[0]

            self.assertEqual(MODULE.pair_key(record), ("3CCC", "4DDD"))
            self.assertEqual(record["endpoints"][0]["chains"], ["A"])
            self.assertFalse(record["apo_holo_orientation_available"])
            representative = record["metadata"]["representative_alignment_pair"]
            self.assertEqual(representative["query_pdb_id"], "1AAA")
            self.assertEqual(representative["target_pdb_id"], "2BBB")
            self.assertEqual(representative["target_ligands_raw"], "LIG ligand")
            self.assertEqual(record["site_identity"]["labels"], [])

    def test_end_to_end_audit_preserves_duplicate_pairs(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            apobind = root / "apobind.csv"
            write_csv(
                apobind,
                ["", "holo_id", "holo_chains", "apo_id", "apo_chains"],
                [
                    {
                        "": "0",
                        "holo_id": "2bbb",
                        "holo_chains": "A",
                        "apo_id": "1aaa",
                        "apo_chains": "A",
                    },
                    {
                        "": "1",
                        "holo_id": "2bbb",
                        "holo_chains": "B",
                        "apo_id": "1aaa",
                        "apo_chains": "B",
                    },
                ],
            )
            output_dir = root / "audit"
            args = argparse.Namespace(
                pscdb_csv=None,
                apobind_csv=apobind,
                codnas_q_csv=None,
                reference_manifest=None,
                reference_endpoint_csv=None,
                output_dir=output_dir,
            )

            report = MODULE.run_audit(args)

            self.assertEqual(report["counts"]["index_records"], 2)
            self.assertEqual(
                report["cross_source_pdb_pair_summary"][
                    "unique_undirected_pdb_pairs"
                ],
                1,
            )
            self.assertEqual(
                report["source_summaries"]["apobind"][
                    "unique_directed_pdb_pairs"
                ],
                1,
            )
            index_rows = [
                json.loads(line)
                for line in (output_dir / "external_endpoint_index.jsonl")
                .read_text()
                .splitlines()
            ]
            self.assertEqual(len(index_rows), 2)
            self.assertNotEqual(
                index_rows[0]["site_identity"]["record_identity_key"],
                index_rows[1]["site_identity"]["record_identity_key"],
            )

    def test_reference_endpoint_csv_counts_missing_endpoint_rows(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "endpoint_index.csv"
            write_csv(
                path,
                ["sample_id", "apo_pdb", "holo_pdb", "ligand_resname"],
                [
                    {
                        "sample_id": "site-a",
                        "apo_pdb": "1aaa",
                        "holo_pdb": "2bbb",
                        "ligand_resname": "LIG",
                    },
                    {
                        "sample_id": "missing",
                        "apo_pdb": "",
                        "holo_pdb": "",
                        "ligand_resname": "",
                    },
                ],
            )

            records, counts = MODULE.load_reference_endpoint_csv(path)

            self.assertEqual(len(records), 1)
            self.assertEqual(records[0]["pair"], ("1AAA", "2BBB"))
            self.assertEqual(records[0]["ligand_comp_id"], "LIG")
            self.assertEqual(
                counts,
                {
                    "source_rows": 2,
                    "valid_endpoint_rows": 1,
                    "missing_endpoint_rows": 1,
                },
            )

    def test_apobind_metadata_funnel_excludes_out_of_range_identity(self):
        def record(key: str, identity: float) -> dict:
            return {
                "record_key": key,
                "source": "apobind",
                "endpoints": [
                    {"pdb_id": "1AAA", "chains": ["A"]},
                    {"pdb_id": "2BBB", "chains": ["A"]},
                ],
                "reference_pair_match": {"match_level": "none"},
                "metadata": {
                    "sequence_identity": identity,
                    "sequence_coverage": 0.98,
                    "backbone_rmsd": 1.2,
                    "tmscore": 0.9,
                    "apo_resolution": 2.0,
                    "binding_site": {
                        "apo_indices": ["1"],
                        "holo_indices": ["1"],
                    },
                },
            }

        report = MODULE._apobind_metadata_funnel(
            [record("valid", 0.98), record("invalid", 1.01)]
        )

        self.assertIsNotNone(report)
        self.assertEqual(report["proxy_records"], 1)
        self.assertEqual(report["source_quality_flags"]["sequence_identity_above_1"], 1)
        self.assertFalse(report["eligible_net_new_systems"])


if __name__ == "__main__":
    unittest.main()
