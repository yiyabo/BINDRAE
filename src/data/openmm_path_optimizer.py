"""Pure NumPy geometry for the non-learned OpenMM Path-4 optimizer."""

from __future__ import annotations

import math
from typing import Sequence, Tuple

import numpy as np

from src.data.residue_identity import ResidueKey


def sine_time_basis(times: Sequence[float], rank: int) -> np.ndarray:
    times_array = np.asarray(times, dtype=np.float64)
    if times_array.ndim != 1 or times_array.size < 3:
        raise ValueError("times must be a one-dimensional grid with at least 3 frames")
    if rank <= 0:
        raise ValueError("rank must be positive")
    if not np.isclose(times_array[0], 0.0) or not np.isclose(times_array[-1], 1.0):
        raise ValueError("times must contain exact 0 and 1 endpoints")
    if np.any(np.diff(times_array) <= 0.0):
        raise ValueError("times must be strictly increasing")
    frequencies = np.arange(1, int(rank) + 1, dtype=np.float64)
    basis = np.sin(math.pi * times_array[:, None] * frequencies[None, :])
    basis[[0, -1]] = 0.0
    return basis


def finite_difference_path_tangent(
    ca_path_angstrom: np.ndarray,
    times: Sequence[float],
    node_mask: np.ndarray,
) -> np.ndarray:
    ca = np.asarray(ca_path_angstrom, dtype=np.float64)
    times_array = np.asarray(times, dtype=np.float64)
    nodes = np.asarray(node_mask, dtype=np.bool_)
    if ca.ndim != 3 or ca.shape[-1] != 3:
        raise ValueError("ca_path_angstrom must have shape [T, N, 3]")
    if times_array.shape != (ca.shape[0],):
        raise ValueError("times do not match the path frame axis")
    if nodes.shape != (ca.shape[1],):
        raise ValueError("node_mask does not match the path residue axis")
    if np.any(np.diff(times_array) <= 0.0):
        raise ValueError("times must be strictly increasing")
    tangent = np.zeros_like(ca)
    tangent[0] = (ca[1] - ca[0]) / (times_array[1] - times_array[0])
    tangent[-1] = (ca[-1] - ca[-2]) / (times_array[-1] - times_array[-2])
    denominator = (times_array[2:] - times_array[:-2])[:, None, None]
    tangent[1:-1] = (ca[2:] - ca[:-2]) / denominator
    tangent *= nodes[None, :, None]
    return tangent


def project_translation_normal(
    vectors: np.ndarray,
    tangent: np.ndarray,
    node_mask: np.ndarray,
    *,
    minimum_tangent_norm: float = 1e-6,
) -> np.ndarray:
    value = np.asarray(vectors, dtype=np.float64)
    tangent_value = np.asarray(tangent, dtype=np.float64)
    nodes = np.asarray(node_mask, dtype=np.bool_)
    if value.shape != tangent_value.shape or value.ndim != 3 or value.shape[-1] != 3:
        raise ValueError("vectors and tangent must have equal [T, N, 3] shapes")
    if nodes.shape != (value.shape[1],):
        raise ValueError("node_mask does not match the residue axis")
    norm = np.linalg.norm(tangent_value, axis=-1, keepdims=True)
    active = (norm > float(minimum_tangent_norm)) & nodes[None, :, None]
    unit = np.divide(
        tangent_value,
        np.maximum(norm, float(minimum_tangent_norm)),
        out=np.zeros_like(tangent_value),
        where=active,
    )
    parallel = np.sum(value * unit, axis=-1, keepdims=True) * unit
    projected = np.where(active, value - parallel, value)
    return projected * nodes[None, :, None]


def smooth_chain_coefficients(
    coefficients: np.ndarray,
    node_mask: np.ndarray,
    peptide_bond_mask: np.ndarray,
    steps: int,
) -> np.ndarray:
    value = np.asarray(coefficients, dtype=np.float64).copy()
    nodes = np.asarray(node_mask, dtype=np.bool_)
    peptide = np.asarray(peptide_bond_mask, dtype=np.bool_)
    if value.ndim != 3 or value.shape[-1] != 3:
        raise ValueError("coefficients must have shape [K, N, 3]")
    if nodes.shape != (value.shape[1],):
        raise ValueError("node_mask does not match coefficients")
    if peptide.shape != (max(value.shape[1] - 1, 0),):
        raise ValueError("peptide_bond_mask does not match coefficients")
    if steps < 0:
        raise ValueError("steps must be non-negative")
    value *= nodes[None, :, None]
    for _ in range(int(steps)):
        total = value.copy()
        count = np.ones((1, value.shape[1], 1), dtype=np.float64)
        if peptide.size:
            left = peptide[None, :, None]
            total[:, 1:] += value[:, :-1] * left
            count[:, 1:] += left
            total[:, :-1] += value[:, 1:] * left
            count[:, :-1] += left
        value = total / count
        value *= nodes[None, :, None]
    return value


def correction_from_coefficients(
    coefficients: np.ndarray,
    basis: np.ndarray,
    tangent: np.ndarray,
    node_mask: np.ndarray,
    *,
    max_residue_translation_angstrom: float = 0.0,
) -> np.ndarray:
    coefficients_array = np.asarray(coefficients, dtype=np.float64)
    basis_array = np.asarray(basis, dtype=np.float64)
    if coefficients_array.ndim != 3 or coefficients_array.shape[-1] != 3:
        raise ValueError("coefficients must have shape [K, N, 3]")
    if basis_array.shape != (tangent.shape[0], coefficients_array.shape[0]):
        raise ValueError("basis does not match coefficient rank or path frames")
    raw = np.einsum("tk,knd->tnd", basis_array, coefficients_array)
    correction = project_translation_normal(raw, tangent, node_mask)
    correction[[0, -1]] = 0.0
    if max_residue_translation_angstrom > 0.0:
        norm = np.linalg.norm(correction, axis=-1, keepdims=True)
        scale = np.minimum(
            1.0,
            float(max_residue_translation_angstrom) / np.maximum(norm, 1e-12),
        )
        correction *= scale
    return correction


def coefficient_force_direction(
    residue_forces_kj_mol_nm: np.ndarray,
    frame_weights: np.ndarray,
    basis: np.ndarray,
    tangent: np.ndarray,
    node_mask: np.ndarray,
    peptide_bond_mask: np.ndarray,
    *,
    chain_smoothing_steps: int,
) -> np.ndarray:
    forces = np.asarray(residue_forces_kj_mol_nm, dtype=np.float64)
    weights = np.asarray(frame_weights, dtype=np.float64)
    if forces.shape != tangent.shape:
        raise ValueError("residue forces and tangent must have equal shapes")
    if weights.shape != (forces.shape[0],):
        raise ValueError("frame_weights do not match force frames")
    if not np.isfinite(forces).all() or not np.isfinite(weights).all():
        raise ValueError("force direction inputs must be finite")
    normal_force = project_translation_normal(forces, tangent, node_mask)
    # OpenMM forces are kJ/mol/nm. Dividing by 10 gives the downhill energy
    # direction in kJ/mol/A before the caller normalizes the proposed step.
    weighted_force = normal_force * weights[:, None, None] / 10.0
    direction = np.einsum("tk,tnd->knd", basis, weighted_force)
    return smooth_chain_coefficients(
        direction,
        node_mask,
        peptide_bond_mask,
        chain_smoothing_steps,
    )


def normalize_coefficient_step(
    direction: np.ndarray,
    basis: np.ndarray,
    tangent: np.ndarray,
    node_mask: np.ndarray,
    target_max_step_angstrom: float,
) -> Tuple[np.ndarray, float]:
    if target_max_step_angstrom <= 0.0:
        raise ValueError("target_max_step_angstrom must be positive")
    induced = correction_from_coefficients(
        direction, basis, tangent, node_mask
    )
    maximum = float(np.linalg.norm(induced, axis=-1).max())
    if maximum <= 1e-12:
        return np.zeros_like(direction, dtype=np.float64), maximum
    return (
        np.asarray(direction, dtype=np.float64)
        * (float(target_max_step_angstrom) / maximum),
        maximum,
    )


def softmax_tail_objective(
    energies_kj_mol: Sequence[float],
    times: Sequence[float],
    *,
    scale_kj_mol: float,
    beta: float,
) -> Tuple[float, np.ndarray, np.ndarray]:
    energies = np.asarray(energies_kj_mol, dtype=np.float64)
    times_array = np.asarray(times, dtype=np.float64)
    if energies.shape != times_array.shape or energies.ndim != 1:
        raise ValueError("energies and times must have equal one-dimensional shapes")
    if energies.size < 3:
        raise ValueError("tail objective requires at least one interior frame")
    if scale_kj_mol <= 0.0 or beta <= 0.0:
        raise ValueError("scale_kj_mol and beta must be positive")
    baseline = (1.0 - times_array) * energies[0] + times_array * energies[-1]
    excess = energies - baseline
    logits = float(beta) * excess[1:-1] / float(scale_kj_mol)
    maximum = float(logits.max())
    exponential = np.exp(logits - maximum)
    interior_weights = exponential / exponential.sum()
    objective = float(scale_kj_mol) * (
        maximum + math.log(float(exponential.mean()))
    ) / float(beta)
    weights = np.zeros_like(energies)
    weights[1:-1] = interior_weights
    return objective, weights, excess


def infer_peptide_bond_mask(
    residue_keys: Sequence[ResidueKey],
    node_mask: np.ndarray,
    atom14_pos_angstrom: np.ndarray,
    *,
    maximum_endpoint_cn_distance_angstrom: float = 2.0,
) -> np.ndarray:
    nodes = np.asarray(node_mask, dtype=np.bool_)
    positions = np.asarray(atom14_pos_angstrom, dtype=np.float64)
    if positions.ndim != 4 or positions.shape[-2:] != (14, 3):
        raise ValueError("atom14_pos_angstrom must have shape [T, N, 14, 3]")
    if len(residue_keys) != positions.shape[1] or nodes.shape != (positions.shape[1],):
        raise ValueError("residue identity does not match the path axis")
    if positions.shape[1] < 2:
        return np.zeros((0,), dtype=np.bool_)
    same_chain = np.asarray(
        [left[0] == right[0] for left, right in zip(residue_keys[:-1], residue_keys[1:])],
        dtype=np.bool_,
    )
    endpoint_cn = np.linalg.norm(
        positions[[0, -1], :-1, 2] - positions[[0, -1], 1:, 0], axis=-1
    )
    return (
        nodes[:-1]
        & nodes[1:]
        & same_chain
        & np.all(endpoint_cn <= float(maximum_endpoint_cn_distance_angstrom), axis=0)
    )
