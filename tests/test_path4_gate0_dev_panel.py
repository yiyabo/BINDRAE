import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from scripts.run_path4_gate0_dev_panel import (
    OPTIMIZER_CONTRACT,
    PANEL_SCHEMA_VERSION,
    REFERENCE_CONTRACT,
    SCORER_CONTRACT,
    _scientific_contract,
    _validate_resume_scientific_contract,
    _write_valid_primary_samples,
    build_selection_manifest,
    read_sample_ids,
    select_diverse_rows,
)


class Path4Gate0DevPanelTest(unittest.TestCase):
    def test_reads_sample_ids_from_prior_panel_manifest(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "selection_manifest.json"
            path.write_text(
                json.dumps(
                    {
                        "systems": [
                            {"sample_id": "a-A-LIG-1", "role": "primary"},
                            {"sample_id": "b-A-LIG-2", "role": "sentinel"},
                        ]
                    }
                )
            )

            self.assertEqual(
                read_sample_ids(path), ["a-A-LIG-1", "b-A-LIG-2"]
            )

    def test_valid_primary_list_contains_only_accepted_primary_systems(self):
        with tempfile.TemporaryDirectory() as temporary:
            state = {
                "systems": {
                    "accepted": {"role": "primary", "status": "preflight_completed"},
                    "rejected": {"role": "primary", "status": "rejected_preflight"},
                    "sentinel": {"role": "sentinel", "status": "completed"},
                }
            }
            path = _write_valid_primary_samples(Path(temporary), state)

            self.assertEqual(path.read_text(), "accepted\n")

    def test_frozen_contract_matches_completed_one_system_run(self):
        self.assertEqual(OPTIMIZER_CONTRACT["time_basis_rank"], 2)
        self.assertEqual(OPTIMIZER_CONTRACT["num_starts"], 3)
        self.assertEqual(OPTIMIZER_CONTRACT["iterations"], 4)
        self.assertEqual(OPTIMIZER_CONTRACT["route_seed_scale_angstrom"], 0.05)
        self.assertEqual(OPTIMIZER_CONTRACT["max_residue_translation_angstrom"], 0.75)
        self.assertEqual(OPTIMIZER_CONTRACT["step_size_angstrom"], 0.05)
        self.assertEqual(OPTIMIZER_CONTRACT["line_search_steps"], 5)
        self.assertEqual(
            REFERENCE_CONTRACT["maximum_reference_atomic_force_kj_mol_nm"],
            1.0e6,
        )
        self.assertEqual(REFERENCE_CONTRACT["reference_relaxation_iterations"], 250)
        self.assertEqual(
            SCORER_CONTRACT["maximum_relaxed_residue_net_force_kj_mol_nm"],
            500.0,
        )
        self.assertEqual(SCORER_CONTRACT["severe_clash_distance_angstrom"], 1.5)

    def test_resume_rejects_reference_contract_drift(self):
        manifest = {"scientific_contract": _scientific_contract()}
        _validate_resume_scientific_contract(manifest)

        stale = json.loads(json.dumps(manifest))
        stale["scientific_contract"]["reference"][
            "reference_relaxation_iterations"
        ] = 25
        with self.assertRaisesRegex(ValueError, "reference"):
            _validate_resume_scientific_contract(stale)

    def test_diverse_selection_is_deterministic_and_unique(self):
        rows = [
            {
                "sample_id": f"{index:04d}-A-L{index % 4}-1",
                "pdb_id": f"{index:04d}",
                "ligand_code": f"L{index % 4}",
                "n_residues": 80 + index * 10,
            }
            for index in range(20)
        ]
        first = select_diverse_rows(rows, 7)
        second = select_diverse_rows(list(reversed(rows)), 7)
        self.assertEqual(
            [row["sample_id"] for row in first],
            [row["sample_id"] for row in second],
        )
        self.assertEqual(len({row["sample_id"] for row in first}), 7)
        self.assertEqual(first[0]["n_residues"], 80)
        self.assertEqual(first[-1]["n_residues"], 270)

    def test_manifest_excludes_holdouts_and_keeps_nonheadline_sentinel(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            phase = root / "phase"
            systems = root / "systems"
            phase.mkdir()
            ids = [
                "sent-A-SEN-1",
                "0001-A-L01-1",
                "0002-A-L02-1",
                "0003-A-L03-1",
                "0004-A-L04-1",
                "0005-A-L05-1",
            ]
            for index, sample_id in enumerate(ids):
                np.savez_compressed(
                    phase / f"{sample_id}.npz",
                    sample_id=np.array(sample_id),
                    n_residues=np.array(80 + 20 * index),
                    t_values=np.linspace(0.05, 0.95, 19),
                )
                report = systems / sample_id / "setup" / "preparation_report.json"
                report.parent.mkdir(parents=True)
                report.write_text(
                    json.dumps({"protein": {"prepared_protein_atoms": 1000 + index}})
                )
            train = root / "train.txt"
            excluded = root / "postselection.txt"
            train.write_text("\n".join(ids) + "\n")
            excluded.write_text("sent-A-SEN-1\n0003-A-L03-1\n")

            manifest = build_selection_manifest(
                train_samples_file=train,
                exclude_samples_files=[excluded],
                phase_cache_dir=phase,
                prepared_systems_dir=systems,
                panel_size=4,
                sentinels=["sent-A-SEN-1"],
                minimum_residues=60,
                maximum_residues=400,
            )

            self.assertEqual(manifest["schema_version"], PANEL_SCHEMA_VERSION)
            self.assertTrue(manifest["exclusion_audit"]["passed"])
            systems_by_id = {
                row["sample_id"]: row for row in manifest["systems"]
            }
            self.assertEqual(systems_by_id["sent-A-SEN-1"]["role"], "sentinel")
            self.assertTrue(
                systems_by_id["sent-A-SEN-1"]["excluded_from_primary_summary"]
            )
            self.assertNotIn("0003-A-L03-1", systems_by_id)
            primary = [
                row for row in manifest["systems"] if row["role"] == "primary"
            ]
            self.assertEqual(len(primary), 3)


if __name__ == "__main__":
    unittest.main()
