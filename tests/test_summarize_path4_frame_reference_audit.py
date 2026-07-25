import json
import tempfile
import unittest
from pathlib import Path

from scripts.summarize_path4_frame_reference_audit import summarize_reports


class SummarizePath4FrameReferenceAuditTest(unittest.TestCase):
    def test_flags_frozen_threshold_and_uses_holo_to_apo_order(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "sample" / "report.json"
            path.parent.mkdir()
            path.write_text(
                json.dumps(
                    {
                        "status": "completed",
                        "sample_id": "sample-A-LIG-1",
                        "prepared_topology_preflight": {
                            "atomic_force_max_kj_mol_nm": 1234.0
                        },
                        "generation_contract": {
                            "reference_relaxation_iterations": 25
                        },
                        "frame_preflight": [
                            {
                                "frame_index": 0,
                                "time": 0.0,
                                "atomic_force_max_kj_mol_nm": 100.0,
                            },
                            {
                                "frame_index": 1,
                                "time": 0.5,
                                "atomic_force_max_kj_mol_nm": 1.1e6,
                            },
                            {
                                "frame_index": 2,
                                "time": 1.0,
                                "atomic_force_max_kj_mol_nm": 2.0e6,
                                "atom_force_diagnostics": {
                                    "top_force_atoms": [
                                        {
                                            "topology_atom_index": 10,
                                            "scope": "hidden_protein",
                                            "atom_name": "H",
                                        }
                                    ],
                                    "force_components": {
                                        "HarmonicBondForce": {
                                            "force_on_total_max_atom_kj_mol_nm": 10.0
                                        },
                                        "NonbondedForce": {
                                            "force_on_total_max_atom_kj_mol_nm": 1.9e6,
                                            "atomic_force_max_kj_mol_nm": 1.9e6,
                                        },
                                    },
                                },
                            },
                        ],
                        "relaxation": {
                            "frame_diagnostics": [
                                {
                                    "frame_index": 0,
                                    "traversal_order": 2,
                                    "mapped_path_step_rms_angstrom": 0.4,
                                    "hidden_atom_injection_rms_angstrom": 0.3,
                                    "hidden_atom_relaxation_rms_angstrom": 0.2,
                                    "hidden_protein_atom_injection_rms_angstrom": 0.3,
                                    "hidden_protein_atom_relaxation_rms_angstrom": 0.2,
                                    "environment_atom_relaxation_rms_angstrom": 0.1,
                                    "mapped_heavy_rms_after_relaxation_angstrom": 0.1,
                                },
                                {
                                    "frame_index": 1,
                                    "traversal_order": 1,
                                    "mapped_path_step_rms_angstrom": 0.5,
                                    "hidden_atom_injection_rms_angstrom": 0.4,
                                    "hidden_atom_relaxation_rms_angstrom": 0.3,
                                    "hidden_protein_atom_injection_rms_angstrom": 0.4,
                                    "hidden_protein_atom_relaxation_rms_angstrom": 0.3,
                                    "environment_atom_relaxation_rms_angstrom": 0.2,
                                    "mapped_heavy_rms_after_relaxation_angstrom": 0.2,
                                },
                                {
                                    "frame_index": 2,
                                    "traversal_order": 0,
                                    "mapped_path_step_rms_angstrom": 0.6,
                                    "hidden_atom_injection_rms_angstrom": 0.5,
                                    "hidden_atom_relaxation_rms_angstrom": 0.4,
                                    "hidden_protein_atom_injection_rms_angstrom": 0.5,
                                    "hidden_protein_atom_relaxation_rms_angstrom": 0.4,
                                    "environment_atom_relaxation_rms_angstrom": 0.3,
                                    "mapped_heavy_rms_after_relaxation_angstrom": 0.3,
                                },
                            ]
                        },
                    }
                )
            )

            summary = summarize_reports([path], scientific_threshold=1.0e6)

            self.assertEqual(summary["reports"], 1)
            self.assertEqual(summary["systems_with_frames_above_frozen_threshold"], 1)
            row = summary["rows"][0]
            self.assertEqual(row["frame_indices_above_frozen_threshold"], [1, 2])
            self.assertEqual(row["maximum_frame"]["frame_index"], 2)
            self.assertEqual(
                row["maximum_force_atom"]["topology_atom_index"], 10
            )
            self.assertEqual(
                row["dominant_force_component_on_max_atom"]["label"],
                "NonbondedForce",
            )
            self.assertEqual(
                row["first_above_threshold_in_holo_to_apo_traversal"][
                    "frame_index"
                ],
                2,
            )
            self.assertIsNotNone(
                row["correlations"][
                    "log10_force_vs_hidden_protein_atom_injection_rms_angstrom"
                ]
            )

    def test_compares_iteration_caps_without_changing_frozen_threshold(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            paths = []
            for iterations, forces in (
                (25, [100.0, 2.0e6, 3.0e6]),
                (250, [100.0, 2.5e6, 5.0e5]),
            ):
                path = root / f"iterations_{iterations}" / "sample" / "report.json"
                path.parent.mkdir(parents=True)
                path.write_text(
                    json.dumps(
                        {
                            "status": "completed",
                            "sample_id": "sample-A-LIG-1",
                            "generation_contract": {
                                "reference_relaxation_iterations": iterations
                            },
                            "frame_preflight": [
                                {
                                    "frame_index": index,
                                    "time": index / 2.0,
                                    "atomic_force_max_kj_mol_nm": force,
                                }
                                for index, force in enumerate(forces)
                            ],
                        }
                    )
                )
                paths.append(path)

            summary = summarize_reports(paths, scientific_threshold=1.0e6)

            comparison = summary["iteration_comparisons"][0]
            self.assertEqual(comparison["baseline_iterations"], 25)
            self.assertEqual(comparison["candidate_iterations"], 250)
            self.assertEqual(comparison["baseline_frames_above_threshold"], 2)
            self.assertEqual(comparison["candidate_frames_above_threshold"], 1)
            self.assertEqual(comparison["resolved_frame_indices"], [2])
            self.assertEqual(comparison["newly_above_frame_indices"], [])
            self.assertAlmostEqual(
                comparison["shared_problem_frame_force_ratios"]["1"], 1.25
            )


if __name__ == "__main__":
    unittest.main()
