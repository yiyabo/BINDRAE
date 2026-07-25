#!/usr/bin/env python3
"""Optimize a Path-3 candidate with a non-learned OpenMM normal route search."""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.evaluate_path4_openmm_gate0 import (  # noqa: E402
    _add_target_restraint,
    _load_json,
    _make_minimization_reporter,
    _platform,
    _state_values,
    _update_target_restraint,
    ensure_implicit_system,
)
from src.data.openmm_gate0 import (  # noqa: E402
    apply_row_transform,
    build_candidate_topology_mapping,
    inject_candidate_frame,
    kabsch_row_transform,
    load_path_candidate,
    reconstruct_peptide_carbonyl_oxygen,
    summarize_energy_profile,
    topology_atom_records,
    validate_reference_topology_state,
)
from src.data.openmm_path_optimizer import (  # noqa: E402
    coefficient_force_direction,
    correction_from_coefficients,
    finite_difference_path_tangent,
    infer_peptide_bond_mask,
    normalize_coefficient_step,
    sine_time_basis,
    smooth_chain_coefficients,
    softmax_tail_objective,
)


OPTIMIZER_SCHEMA_VERSION = "bindrae_path4_openmm_optimizer_v2"


def _minimization_reporter_diagnostics(reporter) -> Dict[str, Any]:
    if reporter is None:
        return {
            "reporter_available": False,
            "reporter_callback_count": None,
            "last_reported_iteration_index": None,
        }
    last_iteration = getattr(reporter, "last_iteration", None)
    return {
        "reporter_available": True,
        "reporter_callback_count": int(getattr(reporter, "report_calls", 0)),
        "last_reported_iteration_index": (
            None if last_iteration is None else int(last_iteration)
        ),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--preparation-report", type=Path, required=True)
    parser.add_argument("--implicit-cache-dir", type=Path, required=True)
    parser.add_argument("--output-candidate", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--candidate-label", default="openmm_multistart_path4")
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--platform", choices=["CPU", "CUDA", "OpenCL"], default="CPU")
    parser.add_argument("--device-index", default="0")
    parser.add_argument("--cpu-threads", type=int, default=4)
    parser.add_argument("--time-basis-rank", type=int, default=4)
    parser.add_argument("--num-starts", type=int, default=5)
    parser.add_argument("--iterations", type=int, default=12)
    parser.add_argument("--seed", type=int, default=20260720)
    parser.add_argument("--route-seed-scale-angstrom", type=float, default=0.10)
    parser.add_argument("--max-residue-translation-angstrom", type=float, default=1.0)
    parser.add_argument("--step-size-angstrom", type=float, default=0.10)
    parser.add_argument("--line-search-steps", type=int, default=7)
    parser.add_argument("--line-search-shrink", type=float, default=0.5)
    parser.add_argument("--chain-smoothing-steps", type=int, default=2)
    parser.add_argument("--softmax-beta", type=float, default=8.0)
    parser.add_argument("--tail-scale-kj-mol", type=float, default=0.0)
    parser.add_argument("--magnitude-penalty-kj-mol-a2", type=float, default=500.0)
    parser.add_argument("--temporal-penalty-kj-mol-a2", type=float, default=100.0)
    parser.add_argument("--acceptance-tolerance-kj-mol", type=float, default=1e-3)
    parser.add_argument("--reference-relaxation-iterations", type=int, default=25)
    parser.add_argument(
        "--reference-restraint-k-kj-mol-nm2", type=float, default=100000.0
    )
    parser.add_argument(
        "--reference-minimization-tolerance-kj-mol-nm", type=float, default=500.0
    )
    parser.add_argument("--minimum-residue-mapping", type=float, default=0.98)
    parser.add_argument("--minimum-atom-mapping", type=float, default=0.95)
    parser.add_argument(
        "--maximum-reference-atomic-force-kj-mol-nm",
        type=float,
        default=1.0e6,
    )
    parser.add_argument("--force-rebuild-system", action="store_true")
    return parser.parse_args()


def _validate_args(args: argparse.Namespace) -> None:
    positive = (
        "cpu_threads",
        "time_basis_rank",
        "num_starts",
        "iterations",
        "line_search_steps",
        "softmax_beta",
        "max_residue_translation_angstrom",
        "step_size_angstrom",
        "reference_relaxation_iterations",
        "reference_restraint_k_kj_mol_nm2",
        "reference_minimization_tolerance_kj_mol_nm",
        "maximum_reference_atomic_force_kj_mol_nm",
    )
    for name in positive:
        if float(getattr(args, name)) <= 0.0:
            raise ValueError(f"--{name.replace('_', '-')} must be positive")
    if args.route_seed_scale_angstrom < 0.0:
        raise ValueError("--route-seed-scale-angstrom must be non-negative")
    if args.chain_smoothing_steps < 0:
        raise ValueError("--chain-smoothing-steps must be non-negative")
    if not 0.0 < args.line_search_shrink < 1.0:
        raise ValueError("--line-search-shrink must be in (0, 1)")
    if args.magnitude_penalty_kj_mol_a2 < 0.0:
        raise ValueError("--magnitude-penalty-kj-mol-a2 must be non-negative")
    if args.temporal_penalty_kj_mol_a2 < 0.0:
        raise ValueError("--temporal-penalty-kj-mol-a2 must be non-negative")


def _candidate_payload(path: Path) -> Dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as loaded:
        return {name: np.asarray(loaded[name]).copy() for name in loaded.files}


def _translated_product_state(candidate, correction_angstrom: np.ndarray) -> np.ndarray:
    """Apply the residue-global optimizer correction to frame translations."""
    if not candidate.has_product_state:
        raise ValueError(
            "Path candidate lacks v3 frame/chi state required for Product-RMSE "
            "evaluation"
        )
    correction = np.asarray(correction_angstrom, dtype=np.float64)
    translation = np.asarray(candidate.rigid_translation_angstrom, dtype=np.float64)
    if correction.shape != translation.shape:
        raise ValueError(
            f"Product-state correction shape mismatch: {correction.shape} != "
            f"{translation.shape}"
        )
    updated = translation + correction
    if not np.array_equal(updated[[0, -1]], translation[[0, -1]]):
        raise RuntimeError("Product-state endpoint translations changed")
    return updated.astype(np.float32)


def _aligned_candidate_path(candidate, mapping, topology_positions):
    source_ca = candidate.atom14_pos_angstrom[
        -1, mapping.mapped_ca_candidate_residue_indices, 1
    ]
    target_ca = topology_positions[mapping.mapped_ca_topology_indices]
    rotation, translation = kabsch_row_transform(source_ca, target_ca)
    aligned = apply_row_transform(
        candidate.atom14_pos_angstrom.reshape(-1, 3), rotation, translation
    ).reshape(candidate.atom14_pos_angstrom.shape)
    error = np.linalg.norm(
        aligned[-1, mapping.mapped_ca_candidate_residue_indices, 1] - target_ca,
        axis=-1,
    )
    diagnostics = {
        "holo_ca_alignment_rms_angstrom": float(np.sqrt(np.mean(error * error))),
        "holo_ca_alignment_max_angstrom": float(error.max()),
        "alignment_rotation_det": float(np.linalg.det(rotation)),
    }
    return aligned, rotation, translation, diagnostics


class OpenMMPathEvaluator:
    def __init__(
        self,
        context,
        reference_positions_angstrom: np.ndarray,
        mapping,
        n_frames: int,
        n_residues: int,
    ) -> None:
        self.context = context
        self.reference_positions = np.asarray(
            reference_positions_angstrom, dtype=np.float64
        )
        self.mapping = mapping
        self.n_frames = int(n_frames)
        self.n_residues = int(n_residues)
        if self.reference_positions.ndim != 3:
            raise ValueError("reference positions must have shape [T, A, 3]")
        if self.reference_positions.shape[0] != self.n_frames:
            raise ValueError("reference positions do not match the path frame axis")
        self.energy_force_calls = 0

    def evaluate(
        self, path_atom14_angstrom: np.ndarray, *, get_forces: bool
    ) -> Tuple[np.ndarray, np.ndarray | None]:
        from openmm import unit

        path = np.asarray(path_atom14_angstrom, dtype=np.float64)
        if path.shape[:2] != (self.n_frames, self.n_residues):
            raise ValueError("Candidate path shape changed during OpenMM optimization")
        energies = np.zeros((self.n_frames,), dtype=np.float64)
        residue_forces = (
            np.zeros((self.n_frames, self.n_residues, 3), dtype=np.float64)
            if get_forces
            else None
        )
        for frame_index in range(self.n_frames):
            injected = inject_candidate_frame(
                self.reference_positions[frame_index],
                path[frame_index],
                self.mapping,
            )
            self.context.setPositions(
                unit.Quantity(injected / 10.0, unit.nanometer)
            )
            state = self.context.getState(getEnergy=True, getForces=get_forces)
            energies[frame_index] = float(
                state.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)
            )
            if get_forces:
                atomic_forces = np.asarray(
                    state.getForces(asNumpy=True).value_in_unit(
                        unit.kilojoule_per_mole / unit.nanometer
                    ),
                    dtype=np.float64,
                )
                for residue_index, atom_indices in enumerate(
                    self.mapping.residue_all_atom_indices
                ):
                    if atom_indices.size:
                        residue_forces[frame_index, residue_index] = atomic_forces[
                            atom_indices
                        ].sum(axis=0)
            self.energy_force_calls += 1
        if not np.isfinite(energies).all():
            raise RuntimeError("OpenMM optimizer observed a non-finite energy")
        if residue_forces is not None and not np.isfinite(residue_forces).all():
            raise RuntimeError("OpenMM optimizer observed a non-finite force")
        return energies, residue_forces


def _precondition_reference_positions(
    context,
    restraint,
    topology_positions_angstrom: np.ndarray,
    base_path_atom14_angstrom: np.ndarray,
    mapping,
    args: argparse.Namespace,
    *,
    protein_atom_count: int | None = None,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Relax hydrogens/unmapped atoms while keeping base heavy atoms fixed."""
    import openmm as openmm_module
    from openmm import LocalEnergyMinimizer, unit

    topology_positions = np.asarray(topology_positions_angstrom, dtype=np.float64)
    references = np.zeros(
        (base_path_atom14_angstrom.shape[0], topology_positions.shape[0], 3),
        dtype=np.float64,
    )
    current_positions = topology_positions.copy()
    started = time.perf_counter()
    maximum_heavy_rms = 0.0
    mapped_atom_mask = np.zeros(topology_positions.shape[0], dtype=np.bool_)
    mapped_atom_mask[mapping.topology_atom_indices] = True
    hidden_atom_indices = np.flatnonzero(~mapped_atom_mask)
    if protein_atom_count is None:
        hidden_protein_atom_indices = hidden_atom_indices
        environment_atom_indices = np.empty(0, dtype=np.int64)
    else:
        protein_count = int(protein_atom_count)
        if not 0 < protein_count <= topology_positions.shape[0]:
            raise ValueError("protein_atom_count is outside the topology atom axis")
        hidden_protein_atom_indices = np.flatnonzero(
            (~mapped_atom_mask) & (np.arange(topology_positions.shape[0]) < protein_count)
        )
        environment_atom_indices = np.arange(
            protein_count, topology_positions.shape[0], dtype=np.int64
        )

    def displacement_rms(
        indices: np.ndarray, left: np.ndarray, right: np.ndarray
    ) -> float:
        if not indices.size:
            return 0.0
        return float(
            np.sqrt(np.mean(np.sum((left[indices] - right[indices]) ** 2, axis=-1)))
        )

    frame_diagnostics: List[Dict[str, Any]] = []
    traversal_order = 0
    for frame_index in reversed(range(base_path_atom14_angstrom.shape[0])):
        target = base_path_atom14_angstrom[frame_index]
        injected = inject_candidate_frame(current_positions, target, mapping)
        hidden_injection_rms = displacement_rms(
            hidden_atom_indices, injected, current_positions
        )
        hidden_protein_injection_rms = displacement_rms(
            hidden_protein_atom_indices, injected, current_positions
        )
        environment_injection_rms = displacement_rms(
            environment_atom_indices, injected, current_positions
        )
        context.setPositions(unit.Quantity(injected / 10.0, unit.nanometer))
        targets_nm = target[
            mapping.candidate_residue_indices,
            mapping.candidate_atom14_indices,
        ] / 10.0
        _update_target_restraint(restraint, context, targets_nm)
        context.setParameter(
            "gate0_k", float(args.reference_restraint_k_kj_mol_nm2)
        )
        minimization_reporter = _make_minimization_reporter(openmm_module)
        minimization_args = (
            context,
            float(args.reference_minimization_tolerance_kj_mol_nm)
            * unit.kilojoule_per_mole
            / unit.nanometer,
            int(args.reference_relaxation_iterations),
        )
        if minimization_reporter is None:
            LocalEnergyMinimizer.minimize(*minimization_args)
        else:
            LocalEnergyMinimizer.minimize(*minimization_args, minimization_reporter)
        state = context.getState(getPositions=True)
        relaxed = np.asarray(
            state.getPositions(asNumpy=True).value_in_unit(unit.angstrom),
            dtype=np.float64,
        )
        heavy_rms = float(
            np.sqrt(
                np.mean(
                    np.sum(
                        (
                            relaxed[mapping.topology_atom_indices]
                            - target[
                                mapping.candidate_residue_indices,
                                mapping.candidate_atom14_indices,
                            ]
                        )
                        ** 2,
                        axis=-1,
                    )
                )
            )
        )
        maximum_heavy_rms = max(maximum_heavy_rms, heavy_rms)
        hidden_relaxation_rms = displacement_rms(
            hidden_atom_indices, relaxed, injected
        )
        hidden_protein_relaxation_rms = displacement_rms(
            hidden_protein_atom_indices, relaxed, injected
        )
        environment_relaxation_rms = displacement_rms(
            environment_atom_indices, relaxed, injected
        )
        mapped_target = target[
            mapping.candidate_residue_indices,
            mapping.candidate_atom14_indices,
        ]
        if frame_index + 1 < base_path_atom14_angstrom.shape[0]:
            source_target = base_path_atom14_angstrom[
                frame_index + 1,
                mapping.candidate_residue_indices,
                mapping.candidate_atom14_indices,
            ]
            mapped_path_step_rms = float(
                np.sqrt(np.mean(np.sum((mapped_target - source_target) ** 2, axis=-1)))
            )
            carried_from_frame_index: int | None = frame_index + 1
        else:
            mapped_path_step_rms = 0.0
            carried_from_frame_index = None
        frame_diagnostics.append(
            {
                "frame_index": int(frame_index),
                "traversal_order": int(traversal_order),
                "carried_from_frame_index": carried_from_frame_index,
                "mapped_path_step_rms_angstrom": mapped_path_step_rms,
                "mapped_heavy_rms_after_relaxation_angstrom": heavy_rms,
                "hidden_atom_injection_rms_angstrom": hidden_injection_rms,
                "hidden_atom_relaxation_rms_angstrom": hidden_relaxation_rms,
                "hidden_protein_atom_injection_rms_angstrom": (
                    hidden_protein_injection_rms
                ),
                "hidden_protein_atom_relaxation_rms_angstrom": (
                    hidden_protein_relaxation_rms
                ),
                "environment_atom_injection_rms_angstrom": (
                    environment_injection_rms
                ),
                "environment_atom_relaxation_rms_angstrom": (
                    environment_relaxation_rms
                ),
                "minimization": {
                    **_minimization_reporter_diagnostics(minimization_reporter),
                    "maximum_iterations": int(args.reference_relaxation_iterations),
                    "tolerance_kj_mol_nm": float(
                        args.reference_minimization_tolerance_kj_mol_nm
                    ),
                    "termination_reason_available": False,
                },
            }
        )
        references[frame_index] = relaxed
        current_positions = relaxed
        traversal_order += 1
    context.setParameter("gate0_k", 0.0)
    return references, {
        "wall_seconds": float(time.perf_counter() - started),
        "maximum_mapped_heavy_rms_angstrom": maximum_heavy_rms,
        "iterations_per_frame": int(args.reference_relaxation_iterations),
        "restraint_k_kj_mol_nm2": float(
            args.reference_restraint_k_kj_mol_nm2
        ),
        "minimization_tolerance_kj_mol_nm": float(
            args.reference_minimization_tolerance_kj_mol_nm
        ),
        "mapped_atom_count": int(mapping.topology_atom_indices.size),
        "hidden_atom_count": int(hidden_atom_indices.size),
        "hidden_protein_atom_count": int(hidden_protein_atom_indices.size),
        "environment_atom_count": int(environment_atom_indices.size),
        "frame_diagnostics": sorted(
            frame_diagnostics, key=lambda record: int(record["frame_index"])
        ),
    }


def _corrected_path(
    base_path: np.ndarray,
    coefficients: np.ndarray,
    basis: np.ndarray,
    tangent: np.ndarray,
    node_mask: np.ndarray,
    peptide_bond_mask: np.ndarray,
    atom14_mask: np.ndarray,
    maximum_translation: float,
) -> Tuple[np.ndarray, np.ndarray]:
    correction = correction_from_coefficients(
        coefficients,
        basis,
        tangent,
        node_mask,
        max_residue_translation_angstrom=maximum_translation,
    )
    path = base_path + correction[:, :, None, :]
    path, _ = reconstruct_peptide_carbonyl_oxygen(
        path,
        atom14_mask,
        node_mask,
        peptide_bond_mask,
    )
    return path, correction


def _objective(
    energies: np.ndarray,
    correction: np.ndarray,
    times: np.ndarray,
    node_mask: np.ndarray,
    args: argparse.Namespace,
    tail_scale: float,
) -> Tuple[float, Dict[str, Any], np.ndarray]:
    tail, frame_weights, excess = softmax_tail_objective(
        energies,
        times,
        scale_kj_mol=tail_scale,
        beta=float(args.softmax_beta),
    )
    active = correction[1:-1, node_mask]
    magnitude = float(np.mean(np.sum(active * active, axis=-1)))
    temporal_delta = np.diff(correction, axis=0)[:, node_mask]
    temporal = float(np.mean(np.sum(temporal_delta * temporal_delta, axis=-1)))
    magnitude_term = float(args.magnitude_penalty_kj_mol_a2) * magnitude
    temporal_term = float(args.temporal_penalty_kj_mol_a2) * temporal
    total = tail + magnitude_term + temporal_term
    return (
        float(total),
        {
            "total": float(total),
            "softmax_tail": float(tail),
            "magnitude_penalty": float(magnitude_term),
            "temporal_penalty": float(temporal_term),
            "correction_rms_angstrom": float(math.sqrt(max(magnitude, 0.0))),
            "excess_p95_kj_mol": float(np.quantile(excess[1:-1], 0.95)),
            "excess_max_kj_mol": float(excess[1:-1].max()),
        },
        frame_weights,
    )


def _seed_coefficients(
    args: argparse.Namespace,
    basis: np.ndarray,
    tangent: np.ndarray,
    node_mask: np.ndarray,
    peptide_bond_mask: np.ndarray,
) -> List[np.ndarray]:
    shape = (basis.shape[1], tangent.shape[1], 3)
    seeds = [np.zeros(shape, dtype=np.float64)]
    rng = np.random.default_rng(int(args.seed))
    while len(seeds) < int(args.num_starts):
        raw = smooth_chain_coefficients(
            rng.normal(size=shape),
            node_mask,
            peptide_bond_mask,
            int(args.chain_smoothing_steps),
        )
        if args.route_seed_scale_angstrom > 0.0:
            scaled, maximum = normalize_coefficient_step(
                raw,
                basis,
                tangent,
                node_mask,
                float(args.route_seed_scale_angstrom),
            )
            if maximum <= 1e-12:
                continue
        else:
            scaled = np.zeros_like(raw)
        seeds.append(scaled)
        if len(seeds) < int(args.num_starts):
            seeds.append(-scaled)
    return seeds[: int(args.num_starts)]


def _run_start(
    start_index: int,
    initial_coefficients: np.ndarray,
    evaluator: OpenMMPathEvaluator,
    base_path: np.ndarray,
    basis: np.ndarray,
    tangent: np.ndarray,
    candidate,
    peptide_bond_mask: np.ndarray,
    args: argparse.Namespace,
    tail_scale: float,
) -> Dict[str, Any]:
    coefficients = np.asarray(initial_coefficients, dtype=np.float64).copy()
    path, correction = _corrected_path(
        base_path,
        coefficients,
        basis,
        tangent,
        candidate.node_mask,
        peptide_bond_mask,
        candidate.atom14_mask,
        float(args.max_residue_translation_angstrom),
    )
    energies, forces = evaluator.evaluate(path, get_forces=True)
    objective, terms, weights = _objective(
        energies,
        correction,
        candidate.times,
        candidate.node_mask,
        args,
        tail_scale,
    )
    best = {
        "objective": objective,
        "terms": terms,
        "coefficients": coefficients.copy(),
        "path": path.copy(),
        "correction": correction.copy(),
        "energies": energies.copy(),
    }
    history: List[Dict[str, Any]] = [
        {"iteration": 0, "accepted": True, "line_search_step": None, **terms}
    ]
    stop_reason = "iteration_budget"
    for iteration in range(1, int(args.iterations) + 1):
        direction = coefficient_force_direction(
            forces,
            weights,
            basis,
            tangent,
            candidate.node_mask,
            peptide_bond_mask,
            chain_smoothing_steps=int(args.chain_smoothing_steps),
        )
        direction, raw_direction_max = normalize_coefficient_step(
            direction,
            basis,
            tangent,
            candidate.node_mask,
            float(args.step_size_angstrom),
        )
        if raw_direction_max <= 1e-12:
            stop_reason = "zero_projected_force"
            break
        accepted = None
        for line_search_step in range(int(args.line_search_steps)):
            multiplier = float(args.line_search_shrink) ** line_search_step
            trial_coefficients = coefficients + multiplier * direction
            trial_path, trial_correction = _corrected_path(
                base_path,
                trial_coefficients,
                basis,
                tangent,
                candidate.node_mask,
                peptide_bond_mask,
                candidate.atom14_mask,
                float(args.max_residue_translation_angstrom),
            )
            trial_energies, _ = evaluator.evaluate(trial_path, get_forces=False)
            trial_objective, trial_terms, trial_weights = _objective(
                trial_energies,
                trial_correction,
                candidate.times,
                candidate.node_mask,
                args,
                tail_scale,
            )
            if trial_objective < objective - float(args.acceptance_tolerance_kj_mol):
                accepted = (
                    line_search_step,
                    trial_coefficients,
                    trial_path,
                    trial_correction,
                    trial_energies,
                    trial_objective,
                    trial_terms,
                    trial_weights,
                )
                break
        if accepted is None:
            history.append(
                {
                    "iteration": iteration,
                    "accepted": False,
                    "line_search_step": None,
                    "total": objective,
                }
            )
            stop_reason = "line_search_rejected"
            break
        (
            line_search_step,
            coefficients,
            path,
            correction,
            energies,
            objective,
            terms,
            weights,
        ) = accepted
        _, forces = evaluator.evaluate(path, get_forces=True)
        history.append(
            {
                "iteration": iteration,
                "accepted": True,
                "line_search_step": int(line_search_step),
                **terms,
            }
        )
        if objective < float(best["objective"]):
            best = {
                "objective": objective,
                "terms": terms,
                "coefficients": coefficients.copy(),
                "path": path.copy(),
                "correction": correction.copy(),
                "energies": energies.copy(),
            }
    return {
        "start_index": int(start_index),
        "stop_reason": stop_reason,
        "history": history,
        "best": best,
    }


def main() -> None:
    args = parse_args()
    _validate_args(args)
    for path in (args.output_candidate, args.report):
        if path.exists():
            raise FileExistsError(f"Refusing to overwrite Path-4 output: {path}")

    preparation = _load_json(args.preparation_report)
    system_path, topology_path, system_contract = ensure_implicit_system(
        cache_dir=args.implicit_cache_dir,
        preparation=preparation,
        preparation_report=args.preparation_report,
        project_root=args.project_root,
        force_rebuild=bool(args.force_rebuild_system),
    )

    from openmm import Context, VerletIntegrator, XmlSerializer, unit
    from openmm import app

    candidate = load_path_candidate(args.candidate)
    payload = _candidate_payload(args.candidate)
    pdb = app.PDBFile(str(topology_path))
    topology_positions = np.asarray(
        pdb.positions.value_in_unit(unit.angstrom), dtype=np.float64
    )
    atom_records = topology_atom_records(pdb.topology)
    mapping = build_candidate_topology_mapping(
        candidate,
        atom_records,
        minimum_residue_fraction=float(args.minimum_residue_mapping),
        minimum_atom_fraction=float(args.minimum_atom_mapping),
    )
    base_path, rotation, translation, alignment = _aligned_candidate_path(
        candidate, mapping, topology_positions
    )
    peptide_bond_mask = infer_peptide_bond_mask(
        candidate.residue_keys,
        candidate.node_mask,
        candidate.atom14_pos_angstrom,
    )
    basis = sine_time_basis(candidate.times, int(args.time_basis_rank))
    tangent = finite_difference_path_tangent(
        base_path[:, :, 1], candidate.times, candidate.node_mask
    )

    system = XmlSerializer.deserialize(system_path.read_text())
    restraint = _add_target_restraint(
        system,
        mapping.topology_atom_indices,
        base_path[
            -1,
            mapping.candidate_residue_indices,
            mapping.candidate_atom14_indices,
        ]
        / 10.0,
        float(args.reference_restraint_k_kj_mol_nm2),
    )
    integrator = VerletIntegrator(0.001 * unit.picoseconds)
    platform, properties = _platform(args)
    context = Context(system, integrator, platform, properties)
    context.setParameter("gate0_k", 0.0)
    context.setPositions(unit.Quantity(topology_positions / 10.0, unit.nanometer))
    reference_energy, reference_forces, _ = _state_values(
        context, get_positions=False
    )
    protein_atoms = int((preparation.get("protein") or {})["prepared_protein_atoms"])
    reference_topology = validate_reference_topology_state(
        reference_energy,
        reference_forces,
        protein_atoms,
        maximum_atomic_force_kj_mol_nm=float(
            args.maximum_reference_atomic_force_kj_mol_nm
        ),
    )
    reference_positions, reference_diagnostics = _precondition_reference_positions(
        context,
        restraint,
        topology_positions,
        base_path,
        mapping,
        args,
        protein_atom_count=protein_atoms,
    )
    print(
        json.dumps(
            {"stage": "reference_preconditioning", **reference_diagnostics},
            sort_keys=True,
        ),
        flush=True,
    )
    evaluator = OpenMMPathEvaluator(
        context,
        reference_positions,
        mapping,
        candidate.n_frames,
        candidate.n_residues,
    )

    started = time.perf_counter()
    base_energies, _ = evaluator.evaluate(base_path, get_forces=False)
    base_profile = summarize_energy_profile(candidate.times, base_energies)
    tail_scale = float(args.tail_scale_kj_mol)
    if tail_scale <= 0.0:
        positive = np.maximum(
            np.asarray(
                [
                    base_profile["interior_positive_excess_p95_kj_mol"],
                    base_profile["interior_positive_excess_mean_kj_mol"],
                ]
            ),
            0.0,
        )
        tail_scale = max(float(positive.max()), 1000.0)
    print(
        json.dumps(
            {
                "stage": "base_objective",
                "tail_scale_kj_mol": tail_scale,
                "energy_profile": base_profile,
            },
            sort_keys=True,
        ),
        flush=True,
    )

    starts = []
    global_best = None
    for start_index, seed in enumerate(
        _seed_coefficients(
            args, basis, tangent, candidate.node_mask, peptide_bond_mask
        )
    ):
        result = _run_start(
            start_index,
            seed,
            evaluator,
            base_path,
            basis,
            tangent,
            candidate,
            peptide_bond_mask,
            args,
            tail_scale,
        )
        starts.append(result)
        print(
            json.dumps(
                {
                    "stage": "optimizer_start_complete",
                    "start_index": result["start_index"],
                    "stop_reason": result["stop_reason"],
                    "best_objective": result["best"]["terms"],
                    "energy_force_calls": evaluator.energy_force_calls,
                },
                sort_keys=True,
            ),
            flush=True,
        )
        if global_best is None or result["best"]["objective"] < global_best["objective"]:
            global_best = result["best"]

    zero_correction = np.zeros(
        (candidate.n_frames, candidate.n_residues, 3), dtype=np.float64
    )
    base_objective, base_terms, _ = _objective(
        base_energies,
        zero_correction,
        candidate.times,
        candidate.node_mask,
        args,
        tail_scale,
    )
    improved = bool(
        global_best is not None
        and float(global_best["objective"])
        < base_objective - float(args.acceptance_tolerance_kj_mol)
    )
    selected = global_best if improved else {
        "objective": base_objective,
        "terms": base_terms,
        "path": base_path,
        "correction": zero_correction,
        "energies": base_energies,
    }

    selected_path_aligned = np.asarray(selected["path"], dtype=np.float64)
    selected_original = apply_row_transform(
        (selected_path_aligned.reshape(-1, 3) - translation),
        rotation.T,
        np.zeros((3,), dtype=np.float64),
    ).reshape(selected_path_aligned.shape)
    selected_original[[0, -1]] = candidate.atom14_pos_angstrom[[0, -1]]
    correction_original = np.asarray(selected["correction"], dtype=np.float64) @ rotation.T
    correction_original[[0, -1]] = 0.0
    endpoint_error = float(
        np.max(
            np.abs(
                selected_original[[0, -1]]
                - candidate.atom14_pos_angstrom[[0, -1]]
            )
        )
    )
    if endpoint_error > 1e-7:
        raise RuntimeError(f"Optimizer endpoint contract failed: {endpoint_error}")

    final_profile = summarize_energy_profile(
        candidate.times, np.asarray(selected["energies"], dtype=np.float64)
    )
    correction_norm = np.linalg.norm(correction_original, axis=-1)
    tangent_norm = np.linalg.norm(tangent, axis=-1)
    valid_normal = (
        candidate.node_mask[None, :]
        & (tangent_norm > 1e-6)
        & (correction_norm > 1e-8)
    )
    normal_cos = np.zeros_like(correction_norm)
    normal_cos[valid_normal] = np.abs(
        np.sum(selected["correction"] * tangent, axis=-1)[valid_normal]
        / (correction_norm[valid_normal] * tangent_norm[valid_normal])
    )

    report = {
        "schema_version": OPTIMIZER_SCHEMA_VERSION,
        "status": "completed",
        "sample_id": candidate.sample_id,
        "input_candidate": str(args.candidate),
        "output_candidate": str(args.output_candidate),
        "selected_improvement": improved,
        "fallback_to_path3": not improved,
        "platform": args.platform,
        "device_index": str(args.device_index),
        "system_contract": system_contract,
        "reference_topology": reference_topology,
        "alignment": alignment,
        "mapping": {
            "mapped_residue_fraction": mapping.mapped_residue_fraction,
            "mapped_atom_fraction": mapping.mapped_atom_fraction,
            "mapped_atoms": int(mapping.topology_atom_indices.size),
            "peptide_bonds": int(peptide_bond_mask.sum()),
        },
        "optimizer": {
            "time_basis_rank": int(args.time_basis_rank),
            "num_starts": int(args.num_starts),
            "iterations": int(args.iterations),
            "seed": int(args.seed),
            "route_seed_scale_angstrom": float(args.route_seed_scale_angstrom),
            "max_residue_translation_angstrom": float(
                args.max_residue_translation_angstrom
            ),
            "step_size_angstrom": float(args.step_size_angstrom),
            "line_search_steps": int(args.line_search_steps),
            "line_search_shrink": float(args.line_search_shrink),
            "chain_smoothing_steps": int(args.chain_smoothing_steps),
            "softmax_beta": float(args.softmax_beta),
            "tail_scale_kj_mol": float(tail_scale),
            "magnitude_penalty_kj_mol_a2": float(
                args.magnitude_penalty_kj_mol_a2
            ),
            "temporal_penalty_kj_mol_a2": float(
                args.temporal_penalty_kj_mol_a2
            ),
        },
        "reference_preconditioning": reference_diagnostics,
        "base_objective": base_terms,
        "selected_objective": selected["terms"],
        "base_energy_profile": base_profile,
        "selected_energy_profile": final_profile,
        "correction": {
            "rms_angstrom": float(
                np.sqrt(
                    np.mean(
                        correction_norm[1:-1, candidate.node_mask] ** 2
                    )
                )
            ),
            "max_angstrom": float(correction_norm.max()),
            "normal_parallel_cos_abs_max": float(
                normal_cos[valid_normal].max() if valid_normal.any() else 0.0
            ),
            "endpoint_max_error_angstrom": endpoint_error,
        },
        "energy_force_calls": int(evaluator.energy_force_calls),
        "wall_seconds": float(time.perf_counter() - started),
        "starts": [
            {
                "start_index": result["start_index"],
                "stop_reason": result["stop_reason"],
                "best_objective": result["best"]["terms"],
                "history": result["history"],
            }
            for result in starts
        ],
    }

    payload["candidate_label"] = np.array(str(args.candidate_label))
    payload["path_parameterization"] = np.array(
        "openmm_projected_normal_multistart_v2"
    )
    payload["atom14_pos_angstrom"] = selected_original.astype(np.float32)
    payload["openmm_residue_translation_correction_angstrom"] = (
        correction_original.astype(np.float32)
    )
    if candidate.has_product_state:
        payload["rigid_translation_angstrom"] = _translated_product_state(
            candidate, correction_original
        )
    payload["correction_diagnostics_json"] = np.array(
        json.dumps(
            {
                "source": OPTIMIZER_SCHEMA_VERSION,
                "report": str(args.report),
                "selected_improvement": improved,
                "fallback_to_path3": not improved,
                "base_objective": base_terms,
                "selected_objective": selected["terms"],
            },
            sort_keys=True,
        )
    )
    args.output_candidate.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output_candidate, **payload)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(
        json.dumps(
            {
                "sample_id": candidate.sample_id,
                "selected_improvement": improved,
                "base_objective": base_objective,
                "selected_objective": float(selected["objective"]),
                "base_excess_p95_kj_mol": base_profile[
                    "interior_positive_excess_p95_kj_mol"
                ],
                "selected_excess_p95_kj_mol": final_profile[
                    "interior_positive_excess_p95_kj_mol"
                ],
                "energy_force_calls": evaluator.energy_force_calls,
                "wall_seconds": report["wall_seconds"],
                "output_candidate": str(args.output_candidate),
            },
            indent=2,
            sort_keys=True,
        )
    )

    del context
    del integrator


if __name__ == "__main__":
    main()
