#!/usr/bin/env python3
"""Audit whether the Path-4 OpenMM force direction is locally downhill."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, Mapping, Sequence

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.evaluate_path4_openmm_gate0 import (  # noqa: E402
    _load_json,
    _platform,
    ensure_implicit_system,
)
from scripts.optimize_path4_openmm_gate0 import (  # noqa: E402
    _aligned_candidate_path,
    _objective,
    _seed_coefficients,
)
from src.data.openmm_gate0 import (  # noqa: E402
    build_candidate_topology_mapping,
    inject_candidate_frame,
    load_frame_reference_cache,
    load_path_candidate,
    reconstruct_peptide_carbonyl_oxygen,
    topology_atom_records,
)
from src.data.openmm_path_optimizer import (  # noqa: E402
    coefficient_force_direction,
    correction_from_coefficients,
    finite_difference_path_tangent,
    infer_peptide_bond_mask,
    normalize_coefficient_step,
    sine_time_basis,
)


SCHEMA_VERSION = "bindrae_path4_optimizer_direction_audit_v2"
VARIANTS = {
    "current": (True, None),
    "no_chain_smoothing": (True, 0),
    "no_carbonyl_rebuild": (False, None),
    "no_chain_smoothing_or_carbonyl_rebuild": (False, 0),
}


@dataclass(frozen=True)
class PathEvaluation:
    energies_kj_mol: np.ndarray
    residue_forces_kj_mol_nm: np.ndarray | None
    atomic_forces_kj_mol_nm: np.ndarray | None
    injected_positions_angstrom: np.ndarray


class DirectionAuditEvaluator:
    """Evaluate a path while retaining the exact all-atom injection Jacobian."""

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
        expected_prefix = (self.n_frames,)
        if (
            self.reference_positions.ndim != 3
            or self.reference_positions.shape[:1] != expected_prefix
            or self.reference_positions.shape[-1] != 3
        ):
            raise ValueError("reference positions must have shape [T, A, 3]")
        self.energy_force_calls = 0

    def inject_path(self, path_atom14_angstrom: np.ndarray) -> np.ndarray:
        path = np.asarray(path_atom14_angstrom, dtype=np.float64)
        if path.shape[:2] != (self.n_frames, self.n_residues):
            raise ValueError("Candidate path shape changed during direction audit")
        return np.stack(
            [
                inject_candidate_frame(
                    self.reference_positions[frame_index],
                    path[frame_index],
                    self.mapping,
                )
                for frame_index in range(self.n_frames)
            ],
            axis=0,
        )

    def evaluate(
        self,
        path_atom14_angstrom: np.ndarray,
        *,
        get_forces: bool,
        frame_order: Sequence[int] | None = None,
    ) -> PathEvaluation:
        from openmm import unit

        injected = self.inject_path(path_atom14_angstrom)
        order = (
            tuple(range(self.n_frames))
            if frame_order is None
            else tuple(int(value) for value in frame_order)
        )
        if sorted(order) != list(range(self.n_frames)):
            raise ValueError("frame_order must be a permutation of all frame indices")

        energies = np.zeros((self.n_frames,), dtype=np.float64)
        atomic_forces = (
            np.zeros_like(injected, dtype=np.float64) if get_forces else None
        )
        residue_forces = (
            np.zeros((self.n_frames, self.n_residues, 3), dtype=np.float64)
            if get_forces
            else None
        )
        for frame_index in order:
            self.context.setPositions(
                unit.Quantity(injected[frame_index] / 10.0, unit.nanometer)
            )
            state = self.context.getState(getEnergy=True, getForces=get_forces)
            energies[frame_index] = float(
                state.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)
            )
            if get_forces:
                frame_forces = np.asarray(
                    state.getForces(asNumpy=True).value_in_unit(
                        unit.kilojoule_per_mole / unit.nanometer
                    ),
                    dtype=np.float64,
                )
                atomic_forces[frame_index] = frame_forces
                for residue_index, atom_indices in enumerate(
                    self.mapping.residue_all_atom_indices
                ):
                    if atom_indices.size:
                        residue_forces[frame_index, residue_index] = frame_forces[
                            atom_indices
                        ].sum(axis=0)
            self.energy_force_calls += 1

        if not np.isfinite(energies).all():
            raise RuntimeError("Direction audit observed a non-finite energy")
        if atomic_forces is not None and not np.isfinite(atomic_forces).all():
            raise RuntimeError("Direction audit observed a non-finite atomic force")
        return PathEvaluation(
            energies_kj_mol=energies,
            residue_forces_kj_mol_nm=residue_forces,
            atomic_forces_kj_mol_nm=atomic_forces,
            injected_positions_angstrom=injected,
        )


def parse_step_multipliers(value: str) -> tuple[float, ...]:
    values = tuple(float(item.strip()) for item in value.split(",") if item.strip())
    if not values or any(not np.isfinite(item) or item <= 0.0 for item in values):
        raise argparse.ArgumentTypeError(
            "step multipliers must be a comma-separated list of positive values"
        )
    if any(left <= right for left, right in zip(values, values[1:])):
        raise argparse.ArgumentTypeError("step multipliers must be strictly decreasing")
    return values


def classify_directional_trials(
    base_value: float,
    plus_values: Sequence[float],
    minus_values: Sequence[float],
    *,
    tolerance: float,
) -> str:
    if len(plus_values) != len(minus_values) or not plus_values:
        raise ValueError("plus and minus trials must have equal non-zero lengths")
    plus_descends = any(value < base_value - tolerance for value in plus_values)
    minus_descends = any(value < base_value - tolerance for value in minus_values)
    if plus_descends and minus_descends:
        return "both_directions_descend"
    if plus_descends:
        return "force_direction_descends"
    if minus_descends:
        return "opposite_direction_descends"
    return "neither_direction_descends"


def classify_local_derivative(
    derivative_kj_mol: float | None, *, tolerance_kj_mol: float
) -> str:
    if derivative_kj_mol is None:
        return "no_stable_central_difference_plateau"
    if derivative_kj_mol < -float(tolerance_kj_mol):
        return "force_direction_descends"
    if derivative_kj_mol > float(tolerance_kj_mol):
        return "opposite_direction_descends"
    return "locally_flat_within_tolerance"


def select_central_difference_plateau(
    step_multipliers: Sequence[float],
    plus_values: Sequence[float],
    minus_values: Sequence[float],
    *,
    window_size: int,
    rtol: float,
    atol_kj_mol: float,
) -> Dict[str, Any]:
    scales = np.asarray(step_multipliers, dtype=np.float64)
    plus = np.asarray(plus_values, dtype=np.float64)
    minus = np.asarray(minus_values, dtype=np.float64)
    if scales.ndim != 1 or scales.size == 0:
        raise ValueError("central-difference plateau requires at least one scale")
    if plus.shape != scales.shape or minus.shape != scales.shape:
        raise ValueError("central-difference values must match the scale axis")
    if np.any(scales <= 0.0) or np.any(np.diff(scales) >= 0.0):
        raise ValueError("central-difference scales must be positive and decreasing")
    if not 2 <= int(window_size) <= scales.size:
        raise ValueError("central-difference plateau window is outside the scale axis")
    if rtol < 0.0 or atol_kj_mol < 0.0:
        raise ValueError("central-difference plateau tolerances must be non-negative")

    derivatives = (plus - minus) / (2.0 * scales)
    scale_records = []
    for index, (scale, derivative) in enumerate(zip(scales, derivatives)):
        record: Dict[str, Any] = {
            "step_multiplier": float(scale),
            "central_derivative_kj_mol": float(derivative),
        }
        if index:
            previous = float(derivatives[index - 1])
            difference = abs(float(derivative) - previous)
            record["previous_scale_absolute_change_kj_mol"] = difference
            record["previous_scale_relative_change"] = float(
                difference / max(abs(float(derivative)), abs(previous), 1.0e-300)
            )
        scale_records.append(record)

    selected = None
    for start in reversed(range(0, scales.size - int(window_size) + 1)):
        stop = start + int(window_size)
        window = derivatives[start:stop]
        center = float(np.median(window))
        scale = float(np.max(np.abs(window)))
        maximum_deviation = float(np.max(np.abs(window - center)))
        tolerance = float(atol_kj_mol) + float(rtol) * scale
        if maximum_deviation <= tolerance:
            selected = {
                "start_index": int(start),
                "stop_index_exclusive": int(stop),
                "step_multipliers": scales[start:stop].tolist(),
                "derivatives_kj_mol": window.tolist(),
                "selected_derivative_kj_mol": center,
                "maximum_deviation_kj_mol": maximum_deviation,
                "tolerance_kj_mol": tolerance,
            }
            break
    return {
        "stable": bool(selected is not None),
        "window_size": int(window_size),
        "rtol": float(rtol),
        "atol_kj_mol": float(atol_kj_mol),
        "selected": selected,
        "scales": scale_records,
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _named_array_sha256(arrays: Sequence[tuple[str, np.ndarray]]) -> str:
    digest = hashlib.sha256()
    for name, value in arrays:
        array = np.ascontiguousarray(value)
        digest.update(name.encode("utf-8"))
        digest.update(str(array.dtype).encode("ascii"))
        digest.update(
            json.dumps(list(array.shape), separators=(",", ":")).encode("ascii")
        )
        digest.update(array.tobytes())
    return digest.hexdigest()


def _mapping_payload(mapping) -> Dict[str, Any]:
    named_arrays = [
        ("candidate_residue_indices", mapping.candidate_residue_indices),
        ("candidate_atom14_indices", mapping.candidate_atom14_indices),
        ("topology_atom_indices", mapping.topology_atom_indices),
        ("topology_residue_indices", mapping.topology_residue_indices),
        ("mapped_ca_topology_indices", mapping.mapped_ca_topology_indices),
        (
            "mapped_ca_candidate_residue_indices",
            mapping.mapped_ca_candidate_residue_indices,
        ),
    ]
    for residue_index, values in enumerate(mapping.residue_all_atom_indices):
        named_arrays.append((f"residue_all_atom_indices/{residue_index}", values))
    for residue_index, values in enumerate(mapping.residue_mapped_topology_indices):
        named_arrays.append(
            (f"residue_mapped_topology_indices/{residue_index}", values)
        )
    for residue_index, values in enumerate(mapping.residue_mapped_atom14_indices):
        named_arrays.append((f"residue_mapped_atom14_indices/{residue_index}", values))
    return {
        "sha256": _named_array_sha256(named_arrays),
        "candidate_residue_indices": mapping.candidate_residue_indices.tolist(),
        "candidate_atom14_indices": mapping.candidate_atom14_indices.tolist(),
        "topology_atom_indices": mapping.topology_atom_indices.tolist(),
        "topology_residue_indices": mapping.topology_residue_indices.tolist(),
        "mapped_ca_topology_indices": mapping.mapped_ca_topology_indices.tolist(),
        "mapped_ca_candidate_residue_indices": (
            mapping.mapped_ca_candidate_residue_indices.tolist()
        ),
        "mapped_residue_fraction": float(mapping.mapped_residue_fraction),
        "mapped_atom_fraction": float(mapping.mapped_atom_fraction),
        "ignored_chain_labels": bool(mapping.ignored_chain_labels),
    }


def compare_numeric_arrays(
    left: np.ndarray,
    right: np.ndarray,
    *,
    rtol: float,
    atol: float,
) -> Dict[str, Any]:
    left_array = np.asarray(left, dtype=np.float64)
    right_array = np.asarray(right, dtype=np.float64)
    if left_array.shape != right_array.shape:
        raise ValueError(
            f"Cannot compare arrays with shapes {left_array.shape} and {right_array.shape}"
        )
    difference = np.abs(left_array - right_array)
    return {
        "allclose": bool(np.allclose(left_array, right_array, rtol=rtol, atol=atol)),
        "rtol": float(rtol),
        "atol": float(atol),
        "max_absolute_difference": float(difference.max(initial=0.0)),
        "rms_difference": float(np.sqrt(np.mean(difference * difference))),
    }


def atomic_force_diagnostics(
    atomic_forces_kj_mol_nm: np.ndarray,
    *,
    saturation_threshold_kj_mol_nm: float,
) -> Dict[str, Any]:
    forces = np.asarray(atomic_forces_kj_mol_nm, dtype=np.float64)
    if forces.ndim != 3 or forces.shape[-1] != 3:
        raise ValueError("atomic forces must have shape [T, A, 3]")
    if saturation_threshold_kj_mol_nm <= 0.0:
        raise ValueError("force saturation threshold must be positive")
    norms = np.linalg.norm(forces, axis=-1)
    per_frame = norms.max(axis=1)
    saturated = np.flatnonzero(per_frame >= float(saturation_threshold_kj_mol_nm))
    maximum_flat = int(np.argmax(norms))
    maximum_frame, maximum_atom = np.unravel_index(maximum_flat, norms.shape)
    return {
        "atomic_force_max_kj_mol_nm": float(norms[maximum_frame, maximum_atom]),
        "atomic_force_abs_component_max_kj_mol_nm": float(np.abs(forces).max()),
        "maximum_force_frame_index": int(maximum_frame),
        "maximum_force_topology_atom_index": int(maximum_atom),
        "per_frame_atomic_force_max_kj_mol_nm": per_frame.tolist(),
        "saturation_threshold_kj_mol_nm": float(saturation_threshold_kj_mol_nm),
        "saturated_frame_indices": saturated.tolist(),
        "force_direction_trustworthy": bool(saturated.size == 0),
    }


def exact_all_atom_force_derivative(
    *,
    atomic_forces_kj_mol_nm: np.ndarray,
    frame_weights: np.ndarray,
    plus_positions_angstrom: np.ndarray,
    minus_positions_angstrom: np.ndarray,
    epsilon: float,
) -> Dict[str, float]:
    forces = np.asarray(atomic_forces_kj_mol_nm, dtype=np.float64)
    weights = np.asarray(frame_weights, dtype=np.float64)
    plus = np.asarray(plus_positions_angstrom, dtype=np.float64)
    minus = np.asarray(minus_positions_angstrom, dtype=np.float64)
    if forces.shape != plus.shape or forces.shape != minus.shape:
        raise ValueError(
            "forces and injected positions must have equal [T, A, 3] shapes"
        )
    if weights.shape != (forces.shape[0],):
        raise ValueError("frame weights do not match all-atom force frames")
    if epsilon <= 0.0:
        raise ValueError("JVP epsilon must be positive")
    coordinate_jvp = (plus - minus) / (2.0 * float(epsilon))
    derivative = -float(np.sum(forces * coordinate_jvp * weights[:, None, None]) / 10.0)
    return {
        "softmax_tail": derivative,
        "coordinate_jvp_max_angstrom": float(
            np.linalg.norm(coordinate_jvp, axis=-1).max()
        ),
        "coordinate_jvp_rms_angstrom": float(
            np.sqrt(np.mean(np.sum(coordinate_jvp * coordinate_jvp, axis=-1)))
        ),
    }


def directional_derivative_agreement(
    observed: float,
    predicted: float,
    *,
    rtol: float,
    atol_kj_mol: float,
    minimum_informative_magnitude_kj_mol: float,
) -> Dict[str, Any]:
    values = (
        observed,
        predicted,
        rtol,
        atol_kj_mol,
        minimum_informative_magnitude_kj_mol,
    )
    if any(not np.isfinite(value) for value in values):
        raise ValueError("directional derivative agreement inputs must be finite")
    if rtol < 0.0 or atol_kj_mol < 0.0 or minimum_informative_magnitude_kj_mol < 0.0:
        raise ValueError(
            "directional derivative agreement tolerances must be non-negative"
        )
    scale = max(abs(float(observed)), abs(float(predicted)))
    absolute_error = abs(float(observed) - float(predicted))
    tolerance = float(atol_kj_mol) + float(rtol) * scale
    informative = scale >= float(minimum_informative_magnitude_kj_mol)
    return {
        "observed_central_difference_kj_mol": float(observed),
        "predicted_force_dot_jvp_kj_mol": float(predicted),
        "absolute_error_kj_mol": absolute_error,
        "relative_error": float(absolute_error / max(scale, 1.0e-300)),
        "rtol": float(rtol),
        "atol_kj_mol": float(atol_kj_mol),
        "minimum_informative_magnitude_kj_mol": float(
            minimum_informative_magnitude_kj_mol
        ),
        "informative": bool(informative),
        "agrees": bool(informative and absolute_error <= tolerance),
    }


def validate_unclipped_trial_bound(
    *,
    seed_correction_max_angstrom: float,
    direction_correction_max_angstrom: float,
    largest_step_multiplier: float,
    maximum_translation_angstrom: float,
    safety_fraction: float = 0.5,
) -> Dict[str, float]:
    values = (
        seed_correction_max_angstrom,
        direction_correction_max_angstrom,
        largest_step_multiplier,
        maximum_translation_angstrom,
        safety_fraction,
    )
    if any(not np.isfinite(value) or value < 0.0 for value in values):
        raise ValueError("clipping-bound inputs must be finite and non-negative")
    if maximum_translation_angstrom <= 0.0 or not 0.0 < safety_fraction < 1.0:
        raise ValueError(
            "clipping bound requires positive maximum and fractional safety"
        )
    upper_bound = float(seed_correction_max_angstrom) + float(
        largest_step_multiplier
    ) * float(direction_correction_max_angstrom)
    ratio = upper_bound / float(maximum_translation_angstrom)
    if ratio > float(safety_fraction):
        raise ValueError(
            "Direction audit could enter the translation clipping regime: "
            f"upper_bound={upper_bound:.6g} A, maximum={maximum_translation_angstrom:.6g} A"
        )
    return {
        "seed_correction_max_angstrom": float(seed_correction_max_angstrom),
        "direction_correction_max_angstrom": float(direction_correction_max_angstrom),
        "largest_step_multiplier": float(largest_step_multiplier),
        "trial_translation_upper_bound_angstrom": upper_bound,
        "maximum_translation_angstrom": float(maximum_translation_angstrom),
        "upper_bound_to_maximum_ratio": ratio,
        "required_safety_fraction": float(safety_fraction),
    }


def _write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(dict(payload), indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def _path_from_coefficients(
    base_path: np.ndarray,
    coefficients: np.ndarray,
    basis: np.ndarray,
    tangent: np.ndarray,
    candidate,
    peptide_bond_mask: np.ndarray,
    *,
    maximum_translation: float,
    rebuild_carbonyl: bool,
) -> tuple[np.ndarray, np.ndarray]:
    correction = correction_from_coefficients(
        coefficients,
        basis,
        tangent,
        candidate.node_mask,
        max_residue_translation_angstrom=maximum_translation,
    )
    path = base_path + correction[:, :, None, :]
    if rebuild_carbonyl:
        path, _ = reconstruct_peptide_carbonyl_oxygen(
            path,
            candidate.atom14_mask,
            candidate.node_mask,
            peptide_bond_mask,
        )
    return path, correction


def _terms_subset(terms: Mapping[str, Any]) -> Dict[str, float]:
    return {
        key: float(terms[key])
        for key in (
            "total",
            "softmax_tail",
            "magnitude_penalty",
            "temporal_penalty",
            "correction_rms_angstrom",
            "excess_p95_kj_mol",
            "excess_max_kj_mol",
        )
    }


def predicted_directional_derivatives(
    *,
    residue_forces_kj_mol_nm: np.ndarray,
    frame_weights: np.ndarray,
    correction_angstrom: np.ndarray,
    direction_correction_angstrom: np.ndarray,
    node_mask: np.ndarray,
    magnitude_penalty_kj_mol_a2: float,
    temporal_penalty_kj_mol_a2: float,
) -> Dict[str, float]:
    nodes = np.asarray(node_mask, dtype=np.bool_)
    weighted_force = (
        np.asarray(residue_forces_kj_mol_nm, dtype=np.float64)
        * np.asarray(frame_weights, dtype=np.float64)[:, None, None]
    )
    direction_correction = np.asarray(direction_correction_angstrom, dtype=np.float64)
    force_derivative = -float(
        np.sum(weighted_force[:, nodes] * direction_correction[:, nodes]) / 10.0
    )

    correction = np.asarray(correction_angstrom, dtype=np.float64)
    active_correction = correction[1:-1, nodes]
    active_direction = direction_correction[1:-1, nodes]
    magnitude_derivative = (
        float(magnitude_penalty_kj_mol_a2)
        * 2.0
        * float(np.mean(np.sum(active_correction * active_direction, axis=-1)))
    )
    temporal_correction = np.diff(correction, axis=0)[:, nodes]
    temporal_direction = np.diff(direction_correction, axis=0)[:, nodes]
    temporal_derivative = (
        float(temporal_penalty_kj_mol_a2)
        * 2.0
        * float(np.mean(np.sum(temporal_correction * temporal_direction, axis=-1)))
    )
    return {
        "softmax_tail": force_derivative,
        "magnitude_penalty": magnitude_derivative,
        "temporal_penalty": temporal_derivative,
        "total": force_derivative + magnitude_derivative + temporal_derivative,
    }


def _audit_start(
    *,
    start_index: int,
    seed: np.ndarray,
    evaluator: DirectionAuditEvaluator,
    base_path: np.ndarray,
    basis: np.ndarray,
    tangent: np.ndarray,
    candidate,
    peptide_bond_mask: np.ndarray,
    settings: SimpleNamespace,
    tail_scale: float,
    step_multipliers: Sequence[float],
    rebuild_carbonyl: bool,
    chain_smoothing_steps: int,
    force_saturation_threshold_kj_mol_nm: float,
    derivative_agreement_rtol: float,
    derivative_agreement_atol_kj_mol: float,
    minimum_informative_derivative_kj_mol: float,
    central_difference_window_size: int,
) -> Dict[str, Any]:
    path, correction = _path_from_coefficients(
        base_path,
        seed,
        basis,
        tangent,
        candidate,
        peptide_bond_mask,
        maximum_translation=float(settings.max_residue_translation_angstrom),
        rebuild_carbonyl=rebuild_carbonyl,
    )
    evaluation = evaluator.evaluate(path, get_forces=True)
    energies = evaluation.energies_kj_mol
    forces = evaluation.residue_forces_kj_mol_nm
    atomic_forces = evaluation.atomic_forces_kj_mol_nm
    if forces is None or atomic_forces is None:
        raise RuntimeError("Direction audit requires residue and atomic forces")
    objective, terms, weights = _objective(
        energies,
        correction,
        candidate.times,
        candidate.node_mask,
        settings,
        tail_scale,
    )
    direction = coefficient_force_direction(
        forces,
        weights,
        basis,
        tangent,
        candidate.node_mask,
        peptide_bond_mask,
        chain_smoothing_steps=chain_smoothing_steps,
    )
    direction, raw_direction_max = normalize_coefficient_step(
        direction,
        basis,
        tangent,
        candidate.node_mask,
        float(settings.step_size_angstrom),
    )
    direction_correction = correction_from_coefficients(
        direction,
        basis,
        tangent,
        candidate.node_mask,
    )
    normalized_direction_max = float(
        np.linalg.norm(direction_correction, axis=-1).max()
    )
    direction_nonzero = bool(
        raw_direction_max > 1.0e-12 and normalized_direction_max > 1.0e-12
    )
    clipping_contract = validate_unclipped_trial_bound(
        seed_correction_max_angstrom=float(np.linalg.norm(correction, axis=-1).max()),
        direction_correction_max_angstrom=normalized_direction_max,
        largest_step_multiplier=float(step_multipliers[0]),
        maximum_translation_angstrom=float(settings.max_residue_translation_angstrom),
    )
    residue_derivatives = predicted_directional_derivatives(
        residue_forces_kj_mol_nm=forces,
        frame_weights=weights,
        correction_angstrom=correction,
        direction_correction_angstrom=direction_correction,
        node_mask=candidate.node_mask,
        magnitude_penalty_kj_mol_a2=float(settings.magnitude_penalty_kj_mol_a2),
        temporal_penalty_kj_mol_a2=float(settings.temporal_penalty_kj_mol_a2),
    )
    trials = []
    jvp_plus_positions = None
    jvp_minus_positions = None
    for multiplier in step_multipliers:
        record: Dict[str, Any] = {"step_multiplier": float(multiplier)}
        for label, sign in (("plus", 1.0), ("minus", -1.0)):
            trial_path, trial_correction = _path_from_coefficients(
                base_path,
                seed + sign * float(multiplier) * direction,
                basis,
                tangent,
                candidate,
                peptide_bond_mask,
                maximum_translation=float(settings.max_residue_translation_angstrom),
                rebuild_carbonyl=rebuild_carbonyl,
            )
            trial_evaluation = evaluator.evaluate(trial_path, get_forces=False)
            trial_energies = trial_evaluation.energies_kj_mol
            _, trial_terms, _ = _objective(
                trial_energies,
                trial_correction,
                candidate.times,
                candidate.node_mask,
                settings,
                tail_scale,
            )
            record[label] = _terms_subset(trial_terms)
            record[label]["total_delta"] = float(trial_terms["total"] - objective)
            record[label]["softmax_tail_delta"] = float(
                trial_terms["softmax_tail"] - terms["softmax_tail"]
            )
            if multiplier == step_multipliers[-1]:
                if label == "plus":
                    jvp_plus_positions = trial_evaluation.injected_positions_angstrom
                else:
                    jvp_minus_positions = trial_evaluation.injected_positions_angstrom
        trials.append(record)

    plus_total = [record["plus"]["total"] for record in trials]
    minus_total = [record["minus"]["total"] for record in trials]
    plus_tail = [record["plus"]["softmax_tail"] for record in trials]
    minus_tail = [record["minus"]["softmax_tail"] for record in trials]
    smallest = float(step_multipliers[-1])
    if jvp_plus_positions is None or jvp_minus_positions is None:
        raise RuntimeError("Direction audit did not retain the coordinate-JVP trials")
    exact_derivatives = exact_all_atom_force_derivative(
        atomic_forces_kj_mol_nm=atomic_forces,
        frame_weights=weights,
        plus_positions_angstrom=jvp_plus_positions,
        minus_positions_angstrom=jvp_minus_positions,
        epsilon=smallest,
    )
    exact_derivatives["magnitude_penalty"] = residue_derivatives["magnitude_penalty"]
    exact_derivatives["temporal_penalty"] = residue_derivatives["temporal_penalty"]
    exact_derivatives["total"] = (
        exact_derivatives["softmax_tail"]
        + exact_derivatives["magnitude_penalty"]
        + exact_derivatives["temporal_penalty"]
    )
    central_total = float((plus_total[-1] - minus_total[-1]) / (2.0 * smallest))
    central_tail = float((plus_tail[-1] - minus_tail[-1]) / (2.0 * smallest))
    total_plateau = select_central_difference_plateau(
        step_multipliers,
        plus_total,
        minus_total,
        window_size=central_difference_window_size,
        rtol=derivative_agreement_rtol,
        atol_kj_mol=derivative_agreement_atol_kj_mol,
    )
    tail_plateau = select_central_difference_plateau(
        step_multipliers,
        plus_tail,
        minus_tail,
        window_size=central_difference_window_size,
        rtol=derivative_agreement_rtol,
        atol_kj_mol=derivative_agreement_atol_kj_mol,
    )
    force_diagnostics = atomic_force_diagnostics(
        atomic_forces,
        saturation_threshold_kj_mol_nm=force_saturation_threshold_kj_mol_nm,
    )
    selected_total_derivative = (
        None
        if total_plateau["selected"] is None
        else float(total_plateau["selected"]["selected_derivative_kj_mol"])
    )
    selected_tail_derivative = (
        None
        if tail_plateau["selected"] is None
        else float(tail_plateau["selected"]["selected_derivative_kj_mol"])
    )
    if selected_total_derivative is None:
        total_derivative_gate = {
            "agrees": False,
            "informative": False,
            "reason": "no_stable_central_difference_plateau",
        }
    else:
        total_derivative_gate = directional_derivative_agreement(
            selected_total_derivative,
            exact_derivatives["total"],
            rtol=derivative_agreement_rtol,
            atol_kj_mol=derivative_agreement_atol_kj_mol,
            minimum_informative_magnitude_kj_mol=(
                minimum_informative_derivative_kj_mol
            ),
        )
    if selected_tail_derivative is None:
        tail_derivative_gate = {
            "agrees": False,
            "informative": False,
            "reason": "no_stable_central_difference_plateau",
        }
    else:
        tail_derivative_gate = directional_derivative_agreement(
            selected_tail_derivative,
            exact_derivatives["softmax_tail"],
            rtol=derivative_agreement_rtol,
            atol_kj_mol=derivative_agreement_atol_kj_mol,
            minimum_informative_magnitude_kj_mol=(
                minimum_informative_derivative_kj_mol
            ),
        )
    direction_conclusion_eligible = bool(
        direction_nonzero
        and force_diagnostics["force_direction_trustworthy"]
        and total_derivative_gate["agrees"]
        and tail_derivative_gate["agrees"]
    )
    return {
        "start_index": int(start_index),
        "initial": _terms_subset(terms),
        "raw_direction_induced_max_angstrom": float(raw_direction_max),
        "normalized_full_step_max_angstrom": normalized_direction_max,
        "direction_nonzero": direction_nonzero,
        "clipping_contract": clipping_contract,
        "force_diagnostics": force_diagnostics,
        "directional_derivative_gate": {
            "total": total_derivative_gate,
            "softmax_tail": tail_derivative_gate,
        },
        "direction_conclusion_eligible": direction_conclusion_eligible,
        "predicted_directional_derivative": {
            "residue_translation_pullback": residue_derivatives,
            "exact_all_atom_coordinate_jvp": exact_derivatives,
        },
        "classification_total": classify_local_derivative(
            selected_total_derivative,
            tolerance_kj_mol=derivative_agreement_atol_kj_mol,
        ),
        "classification_softmax_tail": classify_local_derivative(
            selected_tail_derivative,
            tolerance_kj_mol=derivative_agreement_atol_kj_mol,
        ),
        "any_scale_classification_total": classify_directional_trials(
            objective,
            plus_total,
            minus_total,
            tolerance=float(settings.acceptance_tolerance_kj_mol),
        ),
        "any_scale_classification_softmax_tail": classify_directional_trials(
            float(terms["softmax_tail"]),
            plus_tail,
            minus_tail,
            tolerance=float(settings.acceptance_tolerance_kj_mol),
        ),
        "central_difference_plateau_total": total_plateau,
        "central_difference_plateau_softmax_tail": tail_plateau,
        "smallest_scale_central_derivative_total": central_total,
        "smallest_scale_central_derivative_softmax_tail": central_tail,
        "selected_plateau_to_residue_pullback_total_derivative_ratio": (
            None
            if selected_total_derivative is None
            or abs(residue_derivatives["total"]) <= 1e-12
            else float(selected_total_derivative / residue_derivatives["total"])
        ),
        "selected_plateau_to_exact_jvp_total_derivative_ratio": (
            None
            if selected_total_derivative is None
            or abs(exact_derivatives["total"]) <= 1e-12
            else float(selected_total_derivative / exact_derivatives["total"])
        ),
        "selected_plateau_to_exact_jvp_tail_derivative_ratio": (
            None
            if selected_tail_derivative is None
            or abs(exact_derivatives["softmax_tail"]) <= 1e-12
            else float(selected_tail_derivative / exact_derivatives["softmax_tail"])
        ),
        "trials": trials,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--preparation-report", type=Path, required=True)
    parser.add_argument("--implicit-cache-dir", type=Path, required=True)
    parser.add_argument("--frame-reference-cache", type=Path, required=True)
    parser.add_argument("--source-optimizer-report", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--platform", choices=["CPU", "CUDA", "OpenCL"], default="CPU")
    parser.add_argument("--device-index", default="0")
    parser.add_argument("--cpu-threads", type=int, default=4)
    parser.add_argument("--minimum-residue-mapping", type=float, default=0.98)
    parser.add_argument("--minimum-atom-mapping", type=float, default=0.95)
    parser.add_argument("--acceptance-tolerance-kj-mol", type=float, default=1e-3)
    parser.add_argument(
        "--force-saturation-threshold-kj-mol-nm", type=float, default=1.0e9
    )
    parser.add_argument("--replay-rtol", type=float, default=1.0e-8)
    parser.add_argument("--replay-atol", type=float, default=1.0e-3)
    parser.add_argument("--derivative-agreement-rtol", type=float, default=0.1)
    parser.add_argument(
        "--derivative-agreement-atol-kj-mol", type=float, default=1.0e-2
    )
    parser.add_argument(
        "--minimum-informative-derivative-kj-mol", type=float, default=1.0e-6
    )
    parser.add_argument("--central-difference-window-size", type=int, default=3)
    parser.add_argument(
        "--step-multipliers",
        type=parse_step_multipliers,
        default=parse_step_multipliers(
            "1,0.5,0.25,0.125,0.0625,0.03125,0.015625,0.0078125,0.00390625,0.001953125"
        ),
    )
    parser.add_argument(
        "--variants",
        default=",".join(VARIANTS),
        help="Comma-separated variants; non-current variants audit only start 0.",
    )
    parser.add_argument("--force-rebuild-system", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.report.exists():
        raise FileExistsError(f"Refusing to overwrite direction audit: {args.report}")
    if args.force_saturation_threshold_kj_mol_nm <= 0.0:
        raise ValueError("--force-saturation-threshold-kj-mol-nm must be positive")
    if args.replay_rtol < 0.0 or args.replay_atol < 0.0:
        raise ValueError("Replay tolerances must be non-negative")
    if (
        args.derivative_agreement_rtol < 0.0
        or args.derivative_agreement_atol_kj_mol < 0.0
        or args.minimum_informative_derivative_kj_mol < 0.0
    ):
        raise ValueError("Directional derivative tolerances must be non-negative")
    if not 2 <= args.central_difference_window_size <= len(args.step_multipliers):
        raise ValueError("Central-difference window is outside the scale grid")
    variant_names = tuple(
        item.strip() for item in args.variants.split(",") if item.strip()
    )
    unknown = sorted(set(variant_names) - set(VARIANTS))
    if not variant_names or unknown or "current" not in variant_names:
        raise ValueError(
            f"Invalid direction-audit variants: {unknown or variant_names}"
        )

    source_report = _load_json(args.source_optimizer_report)
    if source_report.get("status") != "completed":
        raise ValueError("Source optimizer report is not completed")
    optimizer = dict(source_report["optimizer"])
    optimizer["acceptance_tolerance_kj_mol"] = float(args.acceptance_tolerance_kj_mol)
    settings = SimpleNamespace(**optimizer)

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
    if candidate.sample_id != str(source_report.get("sample_id")):
        raise ValueError("Candidate and source optimizer report sample IDs differ")
    pdb = app.PDBFile(str(topology_path))
    topology_positions = np.asarray(
        pdb.positions.value_in_unit(unit.angstrom), dtype=np.float64
    )
    mapping = build_candidate_topology_mapping(
        candidate,
        topology_atom_records(pdb.topology),
        minimum_residue_fraction=float(args.minimum_residue_mapping),
        minimum_atom_fraction=float(args.minimum_atom_mapping),
    )
    source_base_path, rotation, translation, alignment = _aligned_candidate_path(
        candidate, mapping, topology_positions
    )
    peptide_bond_mask = infer_peptide_bond_mask(
        candidate.residue_keys,
        candidate.node_mask,
        candidate.atom14_pos_angstrom,
    )
    basis = sine_time_basis(candidate.times, int(settings.time_basis_rank))
    tangent = finite_difference_path_tangent(
        source_base_path[:, :, 1], candidate.times, candidate.node_mask
    )
    canonical_base_path, rebuilt_carbonyl_count = reconstruct_peptide_carbonyl_oxygen(
        source_base_path,
        candidate.atom14_mask,
        candidate.node_mask,
        peptide_bond_mask,
    )
    active_atom_mask = candidate.atom14_mask & candidate.node_mask[None, :, None]
    canonical_delta = np.linalg.norm(canonical_base_path - source_base_path, axis=-1)[
        active_atom_mask
    ]
    reference_cache = load_frame_reference_cache(
        args.frame_reference_cache,
        candidate,
        system_contract=system_contract,
        topology_atom_count=topology_positions.shape[0],
    )
    candidate_sha256 = _sha256(args.candidate)
    if candidate_sha256 != reference_cache.source_candidate_sha256:
        raise ValueError(
            "Exact-cache direction audit requires the cache source candidate: "
            f"{candidate_sha256} != {reference_cache.source_candidate_sha256}"
        )

    system_xml = system_path.read_text()

    def build_evaluator():
        system = XmlSerializer.deserialize(system_xml)
        integrator = VerletIntegrator(0.001 * unit.picoseconds)
        platform, properties = _platform(args)
        context = Context(system, integrator, platform, properties)
        evaluator = DirectionAuditEvaluator(
            context,
            reference_cache.all_atom_pos_angstrom,
            mapping,
            candidate.n_frames,
            candidate.n_residues,
        )
        return evaluator, context, integrator

    evaluator, context, integrator = build_evaluator()
    seeds = _seed_coefficients(
        settings, basis, tangent, candidate.node_mask, peptide_bond_mask
    )
    tail_scale = float(settings.tail_scale_kj_mol)
    started = time.perf_counter()

    source_ascending = evaluator.evaluate(source_base_path, get_forces=True)
    source_descending = evaluator.evaluate(
        source_base_path,
        get_forces=True,
        frame_order=tuple(reversed(range(candidate.n_frames))),
    )
    fresh_evaluator, fresh_context, fresh_integrator = build_evaluator()
    source_fresh = fresh_evaluator.evaluate(source_base_path, get_forces=True)
    source_energy_order_comparison = compare_numeric_arrays(
        source_ascending.energies_kj_mol,
        source_descending.energies_kj_mol,
        rtol=float(args.replay_rtol),
        atol=float(args.replay_atol),
    )
    source_energy_fresh_comparison = compare_numeric_arrays(
        source_ascending.energies_kj_mol,
        source_fresh.energies_kj_mol,
        rtol=float(args.replay_rtol),
        atol=float(args.replay_atol),
    )
    source_force_order_comparison = compare_numeric_arrays(
        source_ascending.atomic_forces_kj_mol_nm,
        source_descending.atomic_forces_kj_mol_nm,
        rtol=float(args.replay_rtol),
        atol=float(args.replay_atol),
    )
    source_force_fresh_comparison = compare_numeric_arrays(
        source_ascending.atomic_forces_kj_mol_nm,
        source_fresh.atomic_forces_kj_mol_nm,
        rtol=float(args.replay_rtol),
        atol=float(args.replay_atol),
    )
    replay_comparisons = (
        source_energy_order_comparison,
        source_energy_fresh_comparison,
        source_force_order_comparison,
        source_force_fresh_comparison,
    )
    if not all(record["allclose"] for record in replay_comparisons):
        raise RuntimeError(
            "Exact candidate/cache replay depends on traversal order or Context state"
        )
    fresh_calls = int(fresh_evaluator.energy_force_calls)
    del fresh_evaluator
    del fresh_context
    del fresh_integrator

    if source_ascending.atomic_forces_kj_mol_nm is None:
        raise RuntimeError("Source replay did not return atomic forces")
    source_force_diagnostics = atomic_force_diagnostics(
        source_ascending.atomic_forces_kj_mol_nm,
        saturation_threshold_kj_mol_nm=float(args.force_saturation_threshold_kj_mol_nm),
    )
    cache_preflight_energies = np.asarray(
        [record["potential_kj_mol"] for record in reference_cache.frame_preflight],
        dtype=np.float64,
    )
    cache_preflight_force_max = np.asarray(
        [
            record["atomic_force_max_kj_mol_nm"]
            for record in reference_cache.frame_preflight
        ],
        dtype=np.float64,
    )
    source_reinjected_force_max = np.asarray(
        source_force_diagnostics["per_frame_atomic_force_max_kj_mol_nm"],
        dtype=np.float64,
    )
    zero_correction = np.zeros(
        (candidate.n_frames, candidate.n_residues, 3), dtype=np.float64
    )
    _, source_reinjected_terms, _ = _objective(
        source_ascending.energies_kj_mol,
        zero_correction,
        candidate.times,
        candidate.node_mask,
        settings,
        tail_scale,
    )
    _, cache_preflight_terms, _ = _objective(
        cache_preflight_energies,
        zero_correction,
        candidate.times,
        candidate.node_mask,
        settings,
        tail_scale,
    )
    variants = []
    for variant_name in variant_names:
        rebuild_carbonyl, smoothing_override = VARIANTS[variant_name]
        smoothing = (
            int(settings.chain_smoothing_steps)
            if smoothing_override is None
            else int(smoothing_override)
        )
        selected_seeds = seeds if variant_name == "current" else seeds[:1]
        variants.append(
            {
                "name": variant_name,
                "rebuild_carbonyl": bool(rebuild_carbonyl),
                "chain_smoothing_steps": smoothing,
                "starts": [
                    _audit_start(
                        start_index=start_index,
                        seed=seed,
                        evaluator=evaluator,
                        base_path=canonical_base_path,
                        basis=basis,
                        tangent=tangent,
                        candidate=candidate,
                        peptide_bond_mask=peptide_bond_mask,
                        settings=settings,
                        tail_scale=tail_scale,
                        step_multipliers=args.step_multipliers,
                        rebuild_carbonyl=bool(rebuild_carbonyl),
                        chain_smoothing_steps=smoothing,
                        force_saturation_threshold_kj_mol_nm=float(
                            args.force_saturation_threshold_kj_mol_nm
                        ),
                        derivative_agreement_rtol=float(args.derivative_agreement_rtol),
                        derivative_agreement_atol_kj_mol=float(
                            args.derivative_agreement_atol_kj_mol
                        ),
                        minimum_informative_derivative_kj_mol=float(
                            args.minimum_informative_derivative_kj_mol
                        ),
                        central_difference_window_size=int(
                            args.central_difference_window_size
                        ),
                    )
                    for start_index, seed in enumerate(selected_seeds)
                ],
            }
        )

    primary_calls = int(evaluator.energy_force_calls)
    del context
    del integrator
    current_zero = variants[variant_names.index("current")]["starts"][0]
    source_base = dict(source_report["base_energy_profile"])
    mapping_report = _mapping_payload(mapping)
    reinjection_energy_comparison = compare_numeric_arrays(
        cache_preflight_energies,
        source_ascending.energies_kj_mol,
        rtol=float(args.replay_rtol),
        atol=float(args.replay_atol),
    )
    reinjection_force_comparison = compare_numeric_arrays(
        cache_preflight_force_max,
        source_reinjected_force_max,
        rtol=float(args.replay_rtol),
        atol=float(args.replay_atol),
    )
    payload = {
        "schema_version": SCHEMA_VERSION,
        "status": "completed",
        "sample_id": candidate.sample_id,
        "candidate": str(args.candidate),
        "candidate_sha256": candidate_sha256,
        "source_optimizer_report": str(args.source_optimizer_report),
        "source_optimizer_report_sha256": _sha256(args.source_optimizer_report),
        "frame_reference_cache": str(args.frame_reference_cache),
        "frame_reference_cache_sha256": _sha256(args.frame_reference_cache),
        "frame_reference_source_candidate_sha256": (
            reference_cache.source_candidate_sha256
        ),
        "platform": args.platform,
        "device_index": str(args.device_index),
        "alignment": {
            **alignment,
            "rotation": np.asarray(rotation, dtype=np.float64).tolist(),
            "translation_angstrom": np.asarray(translation, dtype=np.float64).tolist(),
        },
        "mapping": mapping_report,
        "optimizer_contract": optimizer,
        "audit_contract": {
            "step_multipliers": list(args.step_multipliers),
            "variants": list(variant_names),
            "non_current_variants_audit_only_zero_start": True,
            "canonical_base_rebuilds_carbonyl_before_all_variants": True,
            "frozen_carbonyl_variant_uses_canonical_rebuilt_oxygen": True,
            "force_saturation_threshold_kj_mol_nm": float(
                args.force_saturation_threshold_kj_mol_nm
            ),
            "replay_rtol": float(args.replay_rtol),
            "replay_atol": float(args.replay_atol),
            "derivative_agreement_rtol": float(args.derivative_agreement_rtol),
            "derivative_agreement_atol_kj_mol": float(
                args.derivative_agreement_atol_kj_mol
            ),
            "minimum_informative_derivative_kj_mol": float(
                args.minimum_informative_derivative_kj_mol
            ),
            "central_difference_window_size": int(args.central_difference_window_size),
        },
        "canonical_base": {
            "rebuilt_carbonyl_count": int(rebuilt_carbonyl_count),
            "source_to_canonical_coordinate_max_angstrom": float(
                canonical_delta.max(initial=0.0)
            ),
            "source_to_canonical_coordinate_rms_angstrom": float(
                np.sqrt(np.mean(canonical_delta * canonical_delta))
            ),
        },
        "source_cache_replay": {
            "ascending_energy_kj_mol": (source_ascending.energies_kj_mol.tolist()),
            "ascending_atomic_force_max_kj_mol_nm": (
                source_reinjected_force_max.tolist()
            ),
            "cache_preflight_energy_kj_mol": cache_preflight_energies.tolist(),
            "cache_preflight_atomic_force_max_kj_mol_nm": (
                cache_preflight_force_max.tolist()
            ),
            "ascending_vs_descending_energy": source_energy_order_comparison,
            "ascending_vs_fresh_context_energy": source_energy_fresh_comparison,
            "ascending_vs_descending_atomic_force": (source_force_order_comparison),
            "ascending_vs_fresh_context_atomic_force": (source_force_fresh_comparison),
            "cache_preflight_vs_reinjected_energy": (reinjection_energy_comparison),
            "cache_preflight_vs_reinjected_force_max": (reinjection_force_comparison),
            "cache_preflight_vs_reinjection_expected_nonidentity": True,
            "replay_reproducibility_contract_applies_only_to": (
                "the_same_reinjected_candidate_and_frozen_cache_background"
            ),
            "reinjected_source_force_diagnostics": source_force_diagnostics,
            "cache_preflight_objective": _terms_subset(cache_preflight_terms),
            "reinjected_source_objective": _terms_subset(source_reinjected_terms),
        },
        "base_profile_comparison": {
            "historical_optimizer_unrebuilt_private_reference_p95_kj_mol": float(
                source_base["interior_excess_p95_kj_mol"]
            ),
            "exact_cache_reinjected_source_p95_kj_mol": float(
                source_reinjected_terms["excess_p95_kj_mol"]
            ),
            "exact_cache_canonical_rebuilt_zero_start_p95_kj_mol": float(
                current_zero["initial"]["excess_p95_kj_mol"]
            ),
            "differences_are_not_attributable_to_cache_alone": True,
        },
        "energy_force_calls": {
            "primary_context": primary_calls,
            "fresh_replay_context": fresh_calls,
            "total": primary_calls + fresh_calls,
        },
        "wall_seconds": float(time.perf_counter() - started),
        "variants": variants,
    }
    _write_json_atomic(args.report, payload)
    print(
        json.dumps(
            {
                "status": "completed",
                "sample_id": candidate.sample_id,
                "report": str(args.report),
                "energy_force_calls": payload["energy_force_calls"],
                "wall_seconds": payload["wall_seconds"],
                "force_direction_trustworthy": current_zero["force_diagnostics"][
                    "force_direction_trustworthy"
                ],
                "direction_conclusion_eligible": current_zero[
                    "direction_conclusion_eligible"
                ],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
