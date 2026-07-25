import tempfile
import unittest
from pathlib import Path

import numpy as np

from scripts.evaluate_path4_openmm_gate0 import (
    _assign_diagnostic_force_groups,
    _frame_initial_positions,
    _make_minimization_reporter,
    _minimization_audit,
    _system_generator_options,
)
from src.data.residue_identity import residue_keys_to_array
from src.stage1.data.residue_constants import (
    restype_1to3,
    restype_name_to_atom14_names,
    restype_order,
)
from src.data.openmm_gate0 import (
    PATH_CANDIDATE_LEGACY_SCHEMA_VERSIONS,
    PATH_CANDIDATE_SCHEMA_VERSION,
    PathCandidate,
    TopologyAtomRecord,
    assess_relaxed_frame_validity,
    align_candidate_to_topology,
    build_candidate_topology_mapping,
    build_frame_reference_cache_payload,
    inject_candidate_frame,
    kabsch_row_transform,
    load_frame_reference_cache,
    load_path_candidate,
    reconstruct_peptide_carbonyl_oxygen,
    summarize_energy_profile,
    validate_reference_topology_state,
)


def _candidate(tmp_path: Path) -> PathCandidate:
    residue_letters = ["A", "G", "S"]
    aatype = np.asarray([restype_order[value] for value in residue_letters])
    positions = np.zeros((3, 3, 14, 3), dtype=np.float32)
    mask = np.zeros((3, 3, 14), dtype=np.bool_)
    for frame_index, progress in enumerate((0.0, 0.5, 1.0)):
        for residue_index, letter in enumerate(residue_letters):
            residue_name = restype_1to3[letter]
            for atom_index, atom_name in enumerate(
                restype_name_to_atom14_names[residue_name]
            ):
                if not atom_name:
                    continue
                mask[frame_index, residue_index, atom_index] = True
                positions[frame_index, residue_index, atom_index] = np.asarray(
                    [
                        4.0 * residue_index + atom_index * 0.1 + progress,
                        atom_index * 0.2,
                        progress * (residue_index + 1),
                    ]
                )
    residue_keys = (("A", 1, ""), ("A", 2, ""), ("A", 3, ""))
    path = tmp_path / "candidate.npz"
    np.savez_compressed(
        path,
        schema_version=np.array(PATH_CANDIDATE_SCHEMA_VERSION),
        sample_id=np.array("sample"),
        candidate_label=np.array("path3"),
        path_parameterization=np.array("phase_block_orthogonal_residual_v2"),
        times=np.asarray([0.0, 0.5, 1.0], dtype=np.float32),
        atom14_pos_angstrom=positions,
        atom14_mask=mask,
        aatype=aatype,
        node_mask=np.ones(3, dtype=np.bool_),
        residue_keys=residue_keys_to_array(residue_keys),
        residue_identity_hash=np.array("test-hash"),
        rigid_rotation_matrix=np.broadcast_to(
            np.eye(3, dtype=np.float32), (3, 3, 3, 3)
        ).copy(),
        rigid_translation_angstrom=positions[:, :, 1],
        chi_radians=np.zeros((3, 3, 4), dtype=np.float32),
    )
    return load_path_candidate(path)


def _topology(candidate: PathCandidate):
    records = []
    atom_index = 0
    for residue_index, aatype in enumerate(candidate.aatype):
        residue_name = restype_1to3[
            next(key for key, value in restype_order.items() if value == int(aatype))
        ]
        for atom_name in restype_name_to_atom14_names[residue_name]:
            if not atom_name:
                continue
            records.append(
                TopologyAtomRecord(
                    index=atom_index,
                    residue_index=residue_index,
                    chain_id="X",
                    residue_number=residue_index + 1,
                    insertion_code="",
                    residue_name=residue_name,
                    atom_name=atom_name,
                    element=atom_name[0],
                )
            )
            atom_index += 1
        records.append(
            TopologyAtomRecord(
                index=atom_index,
                residue_index=residue_index,
                chain_id="X",
                residue_number=residue_index + 1,
                insertion_code="",
                residue_name=residue_name,
                atom_name="H",
                element="H",
            )
        )
        atom_index += 1
    return records


class OpenMMGate0Test(unittest.TestCase):
    def test_candidate_v3_exposes_product_state_and_v2_remains_loadable(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            candidate = _candidate(root)
            self.assertTrue(candidate.has_product_state)
            self.assertEqual(candidate.rigid_rotation_matrix.shape, (3, 3, 3, 3))

            with np.load(candidate.source_path, allow_pickle=False) as loaded:
                payload = {
                    name: np.asarray(loaded[name]).copy()
                    for name in loaded.files
                    if name
                    not in {
                        "rigid_rotation_matrix",
                        "rigid_translation_angstrom",
                        "chi_radians",
                    }
                }
            payload["schema_version"] = np.array(
                PATH_CANDIDATE_LEGACY_SCHEMA_VERSIONS[0]
            )
            legacy_path = root / "legacy.npz"
            np.savez_compressed(legacy_path, **payload)
            legacy = load_path_candidate(legacy_path)

            self.assertFalse(legacy.has_product_state)

    def test_frame_initialization_defaults_to_common_prepared_reference(self):
        prepared = np.zeros((3, 3), dtype=np.float64)
        previous = np.ones((3, 3), dtype=np.float64)
        cached = np.full((3, 3), 2.0, dtype=np.float64)

        independent = _frame_initial_positions(
            prepared, previous, "prepared_reference"
        )
        legacy = _frame_initial_positions(prepared, previous, "previous_relaxed")
        frozen = _frame_initial_positions(
            prepared, previous, "reference_cache", cached
        )

        np.testing.assert_array_equal(independent, prepared)
        np.testing.assert_array_equal(legacy, previous)
        np.testing.assert_array_equal(frozen, cached)
        self.assertIsNot(independent, prepared)

    def test_frame_reference_cache_round_trip_and_contract_check(self):
        with tempfile.TemporaryDirectory() as tmp:
            candidate = _candidate(Path(tmp))
            positions = np.zeros((candidate.n_frames, 12, 3), dtype=np.float64)
            system_contract = {"forcefield": "test", "input_hash": "abc"}
            payload = build_frame_reference_cache_payload(
                candidate,
                positions,
                source_candidate_sha256="0" * 64,
                system_contract=system_contract,
                generation_contract={"mode": "path3_preconditioned"},
                frame_preflight=[
                    {"frame_index": index, "atomic_force_max_kj_mol_nm": 10.0}
                    for index in range(candidate.n_frames)
                ],
            )
            cache_path = Path(tmp) / "reference.npz"
            np.savez_compressed(cache_path, **payload)

            loaded = load_frame_reference_cache(
                cache_path,
                candidate,
                system_contract=system_contract,
                topology_atom_count=12,
            )
            np.testing.assert_array_equal(loaded.all_atom_pos_angstrom, positions)
            self.assertEqual(loaded.source_candidate_sha256, "0" * 64)

            with self.assertRaisesRegex(ValueError, "implicit-system contract"):
                load_frame_reference_cache(
                    cache_path,
                    candidate,
                    system_contract={"forcefield": "different"},
                    topology_atom_count=12,
                )

    def test_reconstructs_peptide_carbonyl_oxygen_in_plane(self):
        positions = np.zeros((2, 3, 14, 3), dtype=np.float32)
        mask = np.ones((2, 3, 14), dtype=np.bool_)
        positions[:, 0, 1] = [-1.0, 0.0, 0.0]
        positions[:, 0, 2] = [0.0, 0.0, 0.0]
        positions[:, 0, 3] = [9.0, 9.0, 9.0]
        positions[:, 1, 0] = [0.5, np.sqrt(3.0) / 2.0, 0.0]

        rebuilt, count = reconstruct_peptide_carbonyl_oxygen(
            positions,
            mask,
            np.asarray([True, True, True]),
            np.asarray([True, False]),
        )

        self.assertEqual(count, 2)
        bond = np.linalg.norm(rebuilt[:, 0, 3] - rebuilt[:, 0, 2], axis=-1)
        np.testing.assert_allclose(bond, 1.231, atol=1e-6)
        np.testing.assert_allclose(rebuilt[:, 0, 3, 2], 0.0, atol=1e-7)
        np.testing.assert_array_equal(rebuilt[:, 1:, 3], positions[:, 1:, 3])

    def test_diagnostic_force_groups_are_unique_and_stable(self):
        class HarmonicBondForce:
            def setForceGroup(self, group):
                self.group = group

        class FakeSystem:
            def __init__(self):
                self.forces = [HarmonicBondForce(), HarmonicBondForce()]

            def getNumForces(self):
                return len(self.forces)

            def getForce(self, index):
                return self.forces[index]

        system = FakeSystem()
        labels = _assign_diagnostic_force_groups(system)

        self.assertEqual(labels, [(0, "HarmonicBondForce"), (1, "HarmonicBondForce_1")])
        self.assertEqual([force.group for force in system.forces], [0, 1])

    def test_minimization_audit_does_not_invent_a_termination_reason(self):
        class ReporterBase:
            pass

        class FakeOpenMM:
            MinimizationReporter = ReporterBase

        reporter = _make_minimization_reporter(FakeOpenMM)
        reporter.report(7, None, None, {})
        reporter.report(0, None, None, {})
        audit = _minimization_audit(
            reporter,
            maximum_iterations=250,
            tolerance_kj_mol_nm=25.0,
            unrestrained_potential_change_kj_mol=-10.0,
        )

        self.assertEqual(audit["reporter_callback_count"], 2)
        self.assertEqual(audit["last_reported_iteration_index"], 0)
        self.assertEqual(
            audit["termination"], "returned_from_local_energy_minimizer"
        )
        self.assertFalse(audit["termination_reason_available"])

    def test_nonbonded_method_is_scoped_to_nonperiodic_options(self):
        class FakeApp:
            HBonds = object()
            NoCutoff = object()

        options = _system_generator_options(FakeApp)

        self.assertNotIn("nonbondedMethod", options["forcefield_kwargs"])
        self.assertIs(
            options["nonperiodic_forcefield_kwargs"]["nonbondedMethod"],
            FakeApp.NoCutoff,
        )

    def test_candidate_load_mapping_alignment_and_injection(self):
        with tempfile.TemporaryDirectory() as tmp:
            candidate = _candidate(Path(tmp))
        records = _topology(candidate)
        mapping = build_candidate_topology_mapping(candidate, records)
        self.assertAlmostEqual(mapping.mapped_residue_fraction, 1.0)
        self.assertAlmostEqual(mapping.mapped_atom_fraction, 1.0)
        self.assertTrue(mapping.ignored_chain_labels)

        topology_positions = np.zeros((len(records), 3), dtype=np.float64)
        for candidate_residue, atom14_index, topology_index in zip(
            mapping.candidate_residue_indices,
            mapping.candidate_atom14_indices,
            mapping.topology_atom_indices,
        ):
            source = candidate.atom14_pos_angstrom[
                -1, candidate_residue, atom14_index
            ]
            topology_positions[topology_index] = np.asarray(
                [-source[1] + 3.0, source[0] - 2.0, source[2] + 1.0]
            )
        for atom in records:
            if atom.element == "H":
                topology_positions[atom.index] = topology_positions[atom.index - 1] + 0.2

        aligned, diagnostics = align_candidate_to_topology(
            candidate, mapping, topology_positions
        )
        self.assertLess(diagnostics["holo_ca_alignment_rms_angstrom"], 1e-6)
        injected = inject_candidate_frame(topology_positions, aligned[1], mapping)
        self.assertTrue(
            np.allclose(
                injected[mapping.topology_atom_indices],
                aligned[
                    1,
                    mapping.candidate_residue_indices,
                    mapping.candidate_atom14_indices,
                ],
                atol=1e-6,
            )
        )
        self.assertTrue(np.isfinite(injected).all())

    def test_kabsch_row_transform_recovers_rigid_motion(self):
        source = np.asarray(
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]
        )
        rotation = np.asarray(
            [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]
        )
        translation = np.asarray([3.0, -2.0, 1.0])
        target = source @ rotation + translation
        observed_rotation, observed_translation = kabsch_row_transform(source, target)
        self.assertTrue(np.allclose(observed_rotation, rotation, atol=1e-7))
        self.assertTrue(np.allclose(observed_translation, translation, atol=1e-7))

    def test_energy_profile_uses_endpoint_baseline(self):
        summary = summarize_energy_profile(
            [0.0, 0.25, 0.5, 0.75, 1.0],
            [10.0, 15.0, 20.0, 17.0, 14.0],
        )
        self.assertAlmostEqual(summary["interior_excess_max_kj_mol"], 8.0)
        self.assertGreater(summary["interior_positive_excess_p95_kj_mol"], 0.0)

    def test_energy_profile_can_exclude_invalid_interior_frames(self):
        summary = summarize_energy_profile(
            [0.0, 0.25, 0.5, 0.75, 1.0],
            [0.0, 1000.0, 20.0, 30.0, 0.0],
            interior_valid_mask=[True, False, True, True, True],
        )

        self.assertEqual(summary["interior_frames_total"], 3)
        self.assertEqual(summary["interior_frames_included"], 2)
        self.assertEqual(summary["interior_excess_max_kj_mol"], 30.0)

    def test_relaxed_frame_validity_uses_residue_net_force(self):
        accepted = assess_relaxed_frame_validity(
            500.0,
            maximum_residue_net_force_kj_mol_nm=500.0,
        )
        rejected = assess_relaxed_frame_validity(
            500.1,
            maximum_residue_net_force_kj_mol_nm=500.0,
        )

        self.assertTrue(accepted["valid"])
        self.assertFalse(rejected["valid"])
        self.assertEqual(
            rejected["invalid_reasons"],
            ["protein_residue_net_force_above_threshold"],
        )

    def test_reference_topology_preflight_accepts_finite_moderate_forces(self):
        diagnostics = validate_reference_topology_state(
            232.3,
            np.asarray([[3.0, 4.0, 0.0], [0.0, 12.0, 5.0], [0.0, 0.0, 2.0]]),
            2,
            maximum_atomic_force_kj_mol_nm=100.0,
        )

        self.assertEqual(diagnostics["atomic_force_max_kj_mol_nm"], 13.0)
        self.assertEqual(diagnostics["protein_atomic_force_max_kj_mol_nm"], 13.0)

    def test_reference_topology_preflight_rejects_pathological_force(self):
        with self.assertRaisesRegex(ValueError, "physical preflight"):
            validate_reference_topology_state(
                1.0,
                np.asarray([[0.0, 0.0, 101.0]]),
                1,
                maximum_atomic_force_kj_mol_nm=100.0,
            )


if __name__ == "__main__":
    unittest.main()
