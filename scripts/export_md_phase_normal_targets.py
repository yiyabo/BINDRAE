#!/usr/bin/env python3
"""Export monotone phase and bridge-normal targets from an audited MD path."""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import math
import re
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


SCHEMA_VERSION = "md_phase_normal_v1"
STANDARD_RESIDUES = {
    "ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY", "HIS", "ILE",
    "LEU", "LYS", "MET", "PHE", "PRO", "SER", "THR", "TRP", "TYR", "VAL",
}
RESIDUE_ALIASES = {
    "ASH": "ASP", "CYM": "CYS", "CYX": "CYS", "GLH": "GLU",
    "HID": "HIS", "HIE": "HIS", "HIP": "HIS", "LYN": "LYS",
}
CHI_ATOMS: Mapping[str, Sequence[Sequence[str]]] = {
    "ALA": (),
    "ARG": (("N", "CA", "CB", "CG"), ("CA", "CB", "CG", "CD"),
            ("CB", "CG", "CD", "NE"), ("CG", "CD", "NE", "CZ")),
    "ASN": (("N", "CA", "CB", "CG"), ("CA", "CB", "CG", "OD1")),
    "ASP": (("N", "CA", "CB", "CG"), ("CA", "CB", "CG", "OD1")),
    "CYS": (("N", "CA", "CB", "SG"),),
    "GLN": (("N", "CA", "CB", "CG"), ("CA", "CB", "CG", "CD"),
            ("CB", "CG", "CD", "OE1")),
    "GLU": (("N", "CA", "CB", "CG"), ("CA", "CB", "CG", "CD"),
            ("CB", "CG", "CD", "OE1")),
    "GLY": (),
    "HIS": (("N", "CA", "CB", "CG"), ("CA", "CB", "CG", "ND1")),
    "ILE": (("N", "CA", "CB", "CG1"), ("CA", "CB", "CG1", "CD1")),
    "LEU": (("N", "CA", "CB", "CG"), ("CA", "CB", "CG", "CD1")),
    "LYS": (("N", "CA", "CB", "CG"), ("CA", "CB", "CG", "CD"),
            ("CB", "CG", "CD", "CE"), ("CG", "CD", "CE", "NZ")),
    "MET": (("N", "CA", "CB", "CG"), ("CA", "CB", "CG", "SD"),
            ("CB", "CG", "SD", "CE")),
    "PHE": (("N", "CA", "CB", "CG"), ("CA", "CB", "CG", "CD1")),
    "PRO": (("N", "CA", "CB", "CG"), ("CA", "CB", "CG", "CD")),
    "SER": (("N", "CA", "CB", "OG"),),
    "THR": (("N", "CA", "CB", "OG1"),),
    "TRP": (("N", "CA", "CB", "CG"), ("CA", "CB", "CG", "CD1")),
    "TYR": (("N", "CA", "CB", "CG"), ("CA", "CB", "CG", "CD1")),
    "VAL": (("N", "CA", "CB", "CG1"),),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pull-dir", type=Path, required=True)
    parser.add_argument("--candidate-manifest", type=Path, required=True)
    parser.add_argument("--transition-id", required=True)
    parser.add_argument("--preparation-report", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--sample-id", default=None)
    parser.add_argument("--phase-grid-size", type=int, default=201)
    parser.add_argument("--smoothing-window", type=int, default=21)
    parser.add_argument("--identity-prior-weight", type=float, default=0.02)
    parser.add_argument("--rotation-scale-rad", type=float, default=1.0)
    parser.add_argument("--translation-scale-a", type=float, default=1.0)
    parser.add_argument("--chi-scale-rad", type=float, default=1.0)
    parser.add_argument("--min-endpoint-motion-norm", type=float, default=0.5)
    parser.add_argument("--min-phase-confidence", type=float, default=0.05)
    parser.add_argument("--min-supervision-density", type=float, default=0.05)
    parser.add_argument("--min-residual-envelope", type=float, default=0.15)
    parser.add_argument("--max-normal-residual-norm", type=float, default=5.0)
    parser.add_argument("--max-mean-reconstruction-translation-a", type=float, default=0.75)
    parser.add_argument("--max-mean-reconstruction-rotation-rad", type=float, default=0.50)
    parser.add_argument("--max-mean-reconstruction-chi-rad", type=float, default=0.75)
    parser.add_argument("--min-mapping-fraction", type=float, default=0.95)
    parser.add_argument("--contact-distance-a", type=float, default=4.5)
    parser.add_argument("--pocket-distance-a", type=float, default=6.0)
    return parser.parse_args()


def canonical_resname(name: str) -> str:
    name = str(name).strip().upper()
    return RESIDUE_ALIASES.get(name, name)


def wrap_to_pi(value: np.ndarray) -> np.ndarray:
    return (np.asarray(value) + np.pi) % (2.0 * np.pi) - np.pi


def smoothstep(value: np.ndarray) -> np.ndarray:
    value = np.asarray(value, dtype=np.float64)
    return 3.0 * value * value - 2.0 * value * value * value


def endpoint_envelope(value: np.ndarray) -> np.ndarray:
    value = np.asarray(value, dtype=np.float64)
    envelope = 4.0 * value * (1.0 - value)
    return np.where((value > 0.0) & (value < 1.0), envelope, 0.0)


def temporal_moving_average(values: np.ndarray, window: int) -> np.ndarray:
    """Apply an endpoint-replicated moving average along the first axis."""
    values = np.asarray(values, dtype=np.float64)
    window = int(window)
    if window <= 1:
        return values.copy()
    if window % 2 == 0:
        raise ValueError("smoothing window must be odd")
    radius = window // 2
    padded = np.pad(
        values,
        [(radius, radius)] + [(0, 0)] * (values.ndim - 1),
        mode="edge",
    )
    cumulative = np.cumsum(padded, axis=0, dtype=np.float64)
    cumulative = np.concatenate([np.zeros_like(cumulative[:1]), cumulative], axis=0)
    return (cumulative[window:] - cumulative[:-window]) / float(window)


def temporal_circular_average(angles: np.ndarray, window: int) -> np.ndarray:
    sine = temporal_moving_average(np.sin(angles), window)
    cosine = temporal_moving_average(np.cos(angles), window)
    return np.arctan2(sine, cosine)


def kabsch_transform(mobile: np.ndarray, target: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    mobile = np.asarray(mobile, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    mobile_center = mobile.mean(axis=0)
    target_center = target.mean(axis=0)
    covariance = (mobile - mobile_center).T @ (target - target_center)
    left, _, right_t = np.linalg.svd(covariance)
    rotation = left @ right_t
    if np.linalg.det(rotation) < 0.0:
        left[:, -1] *= -1.0
        rotation = left @ right_t
    translation = target_center - mobile_center @ rotation
    return rotation, translation


def apply_transform(points: np.ndarray, rotation: np.ndarray, translation: np.ndarray) -> np.ndarray:
    return np.asarray(points) @ rotation + translation


def backbone_frames(
    n_coord: np.ndarray, ca_coord: np.ndarray, c_coord: np.ndarray, eps: float = 1e-8
) -> Tuple[np.ndarray, np.ndarray]:
    e1 = c_coord - ca_coord
    e1 = e1 / np.maximum(np.linalg.norm(e1, axis=-1, keepdims=True), eps)
    direction_n = n_coord - ca_coord
    e2 = direction_n - np.sum(direction_n * e1, axis=-1, keepdims=True) * e1
    e2 = e2 / np.maximum(np.linalg.norm(e2, axis=-1, keepdims=True), eps)
    e3 = np.cross(e1, e2)
    e3 = e3 / np.maximum(np.linalg.norm(e3, axis=-1, keepdims=True), eps)
    return np.stack([e1, e2, e3], axis=-1), ca_coord


def dihedral(points: np.ndarray) -> np.ndarray:
    points = np.asarray(points, dtype=np.float64)
    b0 = -(points[..., 1, :] - points[..., 0, :])
    b1 = points[..., 2, :] - points[..., 1, :]
    b2 = points[..., 3, :] - points[..., 2, :]
    b1 = b1 / np.maximum(np.linalg.norm(b1, axis=-1, keepdims=True), 1e-8)
    v = b0 - np.sum(b0 * b1, axis=-1, keepdims=True) * b1
    w = b2 - np.sum(b2 * b1, axis=-1, keepdims=True) * b1
    x = np.sum(v * w, axis=-1)
    y = np.sum(np.cross(b1, v) * w, axis=-1)
    return np.arctan2(y, x)


def parse_pdb_residues(path: Path) -> List[Dict[str, Any]]:
    residues: Dict[Tuple[str, int, str], Dict[str, Any]] = {}
    order: List[Tuple[str, int, str]] = []
    in_first_model = True
    with path.open() as handle:
        for line in handle:
            record = line[:6].strip()
            if record == "ENDMDL":
                break
            if record == "MODEL":
                in_first_model = line[10:14].strip() in {"", "1"}
                continue
            if not in_first_model or record != "ATOM":
                continue
            altloc = line[16].strip()
            if altloc not in {"", "A"}:
                continue
            atom_name = line[12:16].strip()
            resname = canonical_resname(line[17:20])
            chain = line[21].strip()
            resseq = int(line[22:26])
            insertion = line[26].strip()
            key = (chain, resseq, insertion)
            if key not in residues:
                residues[key] = {
                    "key": key,
                    "resname": resname,
                    "atoms": {},
                }
                order.append(key)
            residues[key]["atoms"].setdefault(
                atom_name,
                np.array(
                    [float(line[30:38]), float(line[38:46]), float(line[46:54])],
                    dtype=np.float64,
                ),
            )
    return [residues[key] for key in order if residues[key]["resname"] in STANDARD_RESIDUES]


def residue_chi_from_atoms(residue: Mapping[str, Any]) -> Tuple[np.ndarray, np.ndarray]:
    values = np.zeros(4, dtype=np.float64)
    mask = np.zeros(4, dtype=bool)
    atoms = residue["atoms"]
    for index, names in enumerate(CHI_ATOMS.get(str(residue["resname"]), ())):
        if all(name in atoms for name in names):
            values[index] = float(dihedral(np.stack([atoms[name] for name in names], axis=0)))
            mask[index] = True
    return values, mask


def monotone_path_indices(cost: np.ndarray) -> np.ndarray:
    """Find a minimum-cost nondecreasing grid path in O(TG)."""
    cost = np.asarray(cost, dtype=np.float64)
    if cost.ndim != 2 or cost.shape[0] < 2 or cost.shape[1] < 2:
        raise ValueError("cost must have shape [T>=2, G>=2]")
    n_time, n_grid = cost.shape
    previous = cost[0].copy()
    back = np.zeros((n_time, n_grid), dtype=np.int32)
    for time_index in range(1, n_time):
        prefix_cost = np.empty(n_grid, dtype=np.float64)
        prefix_index = np.empty(n_grid, dtype=np.int32)
        best_cost = math.inf
        best_index = 0
        for grid_index in range(n_grid):
            if previous[grid_index] < best_cost:
                best_cost = previous[grid_index]
                best_index = grid_index
            prefix_cost[grid_index] = best_cost
            prefix_index[grid_index] = best_index
        previous = cost[time_index] + prefix_cost
        back[time_index] = prefix_index
    if not np.isfinite(previous[-1]):
        raise ValueError("No finite monotone endpoint-to-endpoint phase path")
    path = np.empty(n_time, dtype=np.int32)
    path[-1] = n_grid - 1
    for time_index in range(n_time - 1, 0, -1):
        path[time_index - 1] = back[time_index, path[time_index]]
    return path


def _rotation_distance_matrix(observed: np.ndarray, bridge: np.ndarray) -> np.ndarray:
    trace = np.einsum("gji,tji->tg", bridge, observed)
    cosine = np.clip((trace - 1.0) * 0.5, -1.0, 1.0)
    return np.arccos(cosine)


def infer_monotone_phase(
    observed_rotation: np.ndarray,
    observed_translation: np.ndarray,
    observed_chi: np.ndarray,
    bridge_rotation: np.ndarray,
    bridge_translation: np.ndarray,
    bridge_chi: np.ndarray,
    chi_mask: np.ndarray,
    progress: np.ndarray,
    tau_grid: np.ndarray,
    *,
    rotation_scale: float,
    translation_scale: float,
    chi_scale: float,
    identity_prior_weight: float,
) -> Dict[str, np.ndarray]:
    rotation_cost = (_rotation_distance_matrix(observed_rotation, bridge_rotation) / rotation_scale) ** 2
    translation_cost = (
        np.linalg.norm(
            observed_translation[:, None, :] - bridge_translation[None, :, :], axis=-1
        ) / translation_scale
    ) ** 2
    if bool(np.any(chi_mask)):
        chi_delta = wrap_to_pi(observed_chi[:, None, :] - bridge_chi[None, :, :])
        chi_cost = np.sum((chi_delta / chi_scale) ** 2 * chi_mask[None, None, :], axis=-1)
        chi_cost = chi_cost / max(int(chi_mask.sum()), 1)
    else:
        chi_cost = np.zeros_like(rotation_cost)
    physical_cost = rotation_cost + translation_cost + chi_cost
    endpoint_motion = float(
        rotation_cost[-1, 0] + translation_cost[-1, 0] + chi_cost[-1, 0]
    )
    regularized = physical_cost + identity_prior_weight * max(endpoint_motion, 1.0) * (
        tau_grid[None, :] - progress[:, None]
    ) ** 2
    regularized[0, 1:] = np.inf
    regularized[-1, :-1] = np.inf
    indices = monotone_path_indices(regularized)
    selected_cost = physical_cost[np.arange(len(progress)), indices]
    identity_indices = np.abs(tau_grid[None, :] - progress[:, None]).argmin(axis=1)
    identity_cost = physical_cost[np.arange(len(progress)), identity_indices]
    confidence = np.zeros(len(progress), dtype=np.float64)
    exclusion = max(1, int(round(0.05 * (len(tau_grid) - 1))))
    for time_index, selected in enumerate(indices):
        alternative = physical_cost[time_index].copy()
        lower = max(0, selected - exclusion)
        upper = min(len(tau_grid), selected + exclusion + 1)
        alternative[lower:upper] = np.inf
        second = float(np.min(alternative))
        best = float(selected_cost[time_index])
        if np.isfinite(second):
            uniqueness = max(0.0, second - best) / max(second, 1e-8)
            fit = math.exp(-best / max(endpoint_motion, 1.0))
            confidence[time_index] = min(1.0, uniqueness * fit)
    confidence[[0, -1]] = 1.0
    raw_indices = physical_cost.argmin(axis=1)
    return {
        "tau": tau_grid[indices],
        "confidence": confidence,
        "projection_cost": selected_cost,
        "identity_cost": identity_cost,
        "endpoint_motion_norm": np.array(math.sqrt(max(endpoint_motion, 0.0))),
        "raw_monotonicity_violations": np.array(int(np.sum(np.diff(raw_indices) < 0))),
    }


def contact_annotations(
    distances_angstrom: np.ndarray,
    progress: np.ndarray,
    *,
    contact_distance_angstrom: float,
    pocket_distance_angstrom: float,
) -> Dict[str, np.ndarray]:
    distances = np.asarray(distances_angstrom, dtype=np.float64)
    apo = distances[0]
    holo = distances[-1]
    apo_contact = apo <= contact_distance_angstrom
    holo_contact = holo <= contact_distance_angstrom
    formed = ~apo_contact & holo_contact
    released = apo_contact & ~holo_contact
    approach = (holo + 1.0 < apo) | formed
    pocket = np.minimum(apo, holo) <= pocket_distance_angstrom
    transient = (~apo_contact & ~holo_contact) & np.any(
        distances <= contact_distance_angstrom, axis=0
    )
    event_progress = np.full(distances.shape[1], np.nan, dtype=np.float64)
    for residue_index in np.flatnonzero(formed):
        crossing = np.flatnonzero(distances[:, residue_index] <= contact_distance_angstrom)
        if crossing.size:
            event_progress[residue_index] = progress[int(crossing[0])]
    for residue_index in np.flatnonzero(released):
        crossing = np.flatnonzero(distances[:, residue_index] > contact_distance_angstrom)
        if crossing.size:
            event_progress[residue_index] = progress[int(crossing[0])]
    return {
        "pocket_mask": pocket,
        "approach_mask": approach,
        "formed_contact_mask": formed,
        "release_mask": released,
        "transient_contact_mask": transient,
        "contact_event_progress": event_progress,
    }


def _load_candidate(path: Path, transition_id: str) -> Dict[str, Any]:
    matches = []
    with path.open() as handle:
        for line in handle:
            if line.strip():
                record = json.loads(line)
                if record.get("transition_id") == transition_id:
                    matches.append(record)
    if len(matches) != 1:
        raise ValueError(f"Expected one {transition_id!r} record in {path}, found {len(matches)}")
    return matches[0]


def _load_metric_rows(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open(newline="") as handle:
        for raw in csv.DictReader(handle):
            row = dict(raw)
            for key in ("progress", "temperature_k", "potential_kj_mol"):
                row[key] = float(row[key])
            rows.append(row)
    return rows


def _topology_residue_key(residue: Any) -> Tuple[str, int, str]:
    chain_id = str(getattr(residue.chain, "chain_id", "") or "").strip()
    return chain_id, int(residue.resSeq), ""


def _safe_sample_id(sample_id: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(sample_id))


def _trajectory_chi(
    trajectory_xyz_angstrom: np.ndarray,
    topology_residues: Sequence[Any],
) -> Tuple[np.ndarray, np.ndarray]:
    n_frames = trajectory_xyz_angstrom.shape[0]
    values = np.zeros((n_frames, len(topology_residues), 4), dtype=np.float64)
    mask = np.zeros((len(topology_residues), 4), dtype=bool)
    for residue_index, residue in enumerate(topology_residues):
        atom_indices = {atom.name: atom.index for atom in residue.atoms}
        resname = canonical_resname(residue.name)
        for chi_index, names in enumerate(CHI_ATOMS.get(resname, ())):
            if all(name in atom_indices for name in names):
                indices = [atom_indices[name] for name in names]
                values[:, residue_index, chi_index] = dihedral(
                    trajectory_xyz_angstrom[:, indices, :]
                )
                mask[residue_index, chi_index] = True
    return values, mask


def _residue_ligand_distances(
    trajectory: Any,
    topology_residues: Sequence[Any],
    ligand_heavy_indices: Sequence[int],
) -> np.ndarray:
    import mdtraj as md

    output = np.full((trajectory.n_frames, len(topology_residues)), np.inf, dtype=np.float64)
    for residue_index, residue in enumerate(topology_residues):
        heavy = [
            atom.index for atom in residue.atoms
            if atom.element is None or atom.element.symbol.upper() != "H"
        ]
        pairs = np.asarray(
            [(protein_atom, ligand_atom) for protein_atom in heavy for ligand_atom in ligand_heavy_indices],
            dtype=np.int32,
        )
        if pairs.size:
            output[:, residue_index] = md.compute_distances(
                trajectory, pairs, periodic=True
            ).min(axis=1) * 10.0
    return output


def _extract_endpoint_arrays(
    residues: Sequence[Mapping[str, Any]], keys: Sequence[Tuple[str, int, str]]
) -> Dict[str, np.ndarray]:
    by_key = {tuple(residue["key"]): residue for residue in residues}
    n_coord, ca_coord, c_coord = [], [], []
    chi, chi_mask = [], []
    names = []
    for key in keys:
        residue = by_key[key]
        atoms = residue["atoms"]
        if not all(name in atoms for name in ("N", "CA", "C")):
            raise ValueError(f"Endpoint residue {key} lacks N/CA/C")
        n_coord.append(atoms["N"])
        ca_coord.append(atoms["CA"])
        c_coord.append(atoms["C"])
        chi_value, mask = residue_chi_from_atoms(residue)
        chi.append(chi_value)
        chi_mask.append(mask)
        names.append(residue["resname"])
    return {
        "N": np.stack(n_coord),
        "Ca": np.stack(ca_coord),
        "C": np.stack(c_coord),
        "chi": np.stack(chi),
        "chi_mask": np.stack(chi_mask),
        "resnames": np.asarray(names),
    }


def _normal_decomposition(
    *,
    observed_rotation: np.ndarray,
    observed_translation: np.ndarray,
    observed_chi: np.ndarray,
    endpoint_n: np.ndarray,
    endpoint_ca: np.ndarray,
    endpoint_c: np.ndarray,
    endpoint_chi: np.ndarray,
    tau: np.ndarray,
    progress: np.ndarray,
    node_mask: np.ndarray,
    chi_mask: np.ndarray,
    confidence: np.ndarray,
    args: argparse.Namespace,
) -> Dict[str, np.ndarray]:
    import torch

    def load_geometry_module(name: str, relative_path: str) -> Any:
        path = PROJECT_ROOT / relative_path
        spec = importlib.util.spec_from_file_location(name, path)
        if spec is None or spec.loader is None:
            raise ImportError(f"Cannot load geometry module from {path}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    phase_module = load_geometry_module(
        "bindrae_md_phase_residual", "src/stage2/modules/phase_residual.py"
    )
    se3_module = load_geometry_module("bindrae_md_se3", "src/stage2/modules/se3.py")
    project_product_tangent_normal = phase_module.project_product_tangent_normal
    rigid_compose = se3_module.rigid_compose
    rigid_inverse = se3_module.rigid_inverse
    se3_exp = se3_module.se3_exp
    se3_log = se3_module.se3_log

    torch.set_grad_enabled(False)
    dtype = torch.float64
    tau_t = torch.as_tensor(tau, dtype=dtype)
    gamma = 3.0 * tau_t.square() - 2.0 * tau_t.pow(3)
    gamma_coord = gamma.unsqueeze(-1)
    n0, n1 = (torch.as_tensor(endpoint_n[index], dtype=dtype) for index in (0, 1))
    ca0, ca1 = (torch.as_tensor(endpoint_ca[index], dtype=dtype) for index in (0, 1))
    c0, c1 = (torch.as_tensor(endpoint_c[index], dtype=dtype) for index in (0, 1))
    n_bridge = (1.0 - gamma_coord) * n0 + gamma_coord * n1
    ca_bridge = (1.0 - gamma_coord) * ca0 + gamma_coord * ca1
    c_bridge = (1.0 - gamma_coord) * c0 + gamma_coord * c1

    def torch_frames(n_value: Any, ca_value: Any, c_value: Any) -> Tuple[Any, Any]:
        e1 = c_value - ca_value
        e1 = e1 / torch.linalg.norm(e1, dim=-1, keepdim=True).clamp_min(1e-8)
        u = n_value - ca_value
        e2 = u - (u * e1).sum(dim=-1, keepdim=True) * e1
        e2 = e2 / torch.linalg.norm(e2, dim=-1, keepdim=True).clamp_min(1e-8)
        e3 = torch.cross(e1, e2, dim=-1)
        return torch.stack([e1, e2, e3], dim=-1), ca_value

    bridge_rotation, bridge_translation = torch_frames(n_bridge, ca_bridge, c_bridge)
    observed_rotation_t = torch.as_tensor(observed_rotation, dtype=dtype)
    observed_translation_t = torch.as_tensor(observed_translation, dtype=dtype)
    inverse_rotation, inverse_translation = rigid_inverse(bridge_rotation, bridge_translation)
    delta_rotation, delta_translation = rigid_compose(
        inverse_rotation, inverse_translation, observed_rotation_t, observed_translation_t
    )
    raw_rigid = se3_log(delta_rotation, delta_translation)

    endpoint_chi_t = torch.as_tensor(endpoint_chi, dtype=dtype)
    endpoint_delta_chi = torch.remainder(
        endpoint_chi_t[1] - endpoint_chi_t[0] + torch.pi, 2.0 * torch.pi
    ) - torch.pi
    bridge_chi = torch.remainder(
        endpoint_chi_t[0] + gamma_coord * endpoint_delta_chi + torch.pi,
        2.0 * torch.pi,
    ) - torch.pi
    observed_chi_t = torch.as_tensor(observed_chi, dtype=dtype)
    raw_chi = torch.remainder(observed_chi_t - bridge_chi + torch.pi, 2.0 * torch.pi) - torch.pi

    epsilon = 1e-3
    tau_low = (tau_t - epsilon).clamp(0.0, 1.0)
    tau_high = (tau_t + epsilon).clamp(0.0, 1.0)
    low_gamma = (3.0 * tau_low.square() - 2.0 * tau_low.pow(3)).unsqueeze(-1)
    high_gamma = (3.0 * tau_high.square() - 2.0 * tau_high.pow(3)).unsqueeze(-1)
    low_rotation, low_translation = torch_frames(
        (1.0 - low_gamma) * n0 + low_gamma * n1,
        (1.0 - low_gamma) * ca0 + low_gamma * ca1,
        (1.0 - low_gamma) * c0 + low_gamma * c1,
    )
    high_rotation, high_translation = torch_frames(
        (1.0 - high_gamma) * n0 + high_gamma * n1,
        (1.0 - high_gamma) * ca0 + high_gamma * ca1,
        (1.0 - high_gamma) * c0 + high_gamma * c1,
    )
    low_inverse_rotation, low_inverse_translation = rigid_inverse(low_rotation, low_translation)
    tangent_rotation, tangent_translation = rigid_compose(
        low_inverse_rotation, low_inverse_translation, high_rotation, high_translation
    )
    denominator = (tau_high - tau_low).unsqueeze(-1).clamp_min(1e-6)
    bridge_tangent_rigid = se3_log(tangent_rotation, tangent_translation) / denominator
    bridge_tangent_chi = endpoint_delta_chi.unsqueeze(0).expand_as(raw_chi)

    node_t = torch.as_tensor(node_mask, dtype=torch.bool).unsqueeze(0).expand(tau.shape)
    chi_mask_t = torch.as_tensor(chi_mask, dtype=torch.bool).unsqueeze(0).expand(raw_chi.shape)
    projection = project_product_tangent_normal(
        raw_rigid,
        raw_chi,
        bridge_tangent_rigid,
        bridge_tangent_chi,
        node_mask=node_t,
        chi_mask=chi_mask_t,
        rotation_scale=args.rotation_scale_rad,
        translation_scale=args.translation_scale_a,
        chi_scale=args.chi_scale_rad,
        min_tangent_norm=args.min_endpoint_motion_norm,
        max_metric_norm=0.0,
    )
    projected_rigid = projection["projected_rigid"]
    projected_chi = projection["projected_chi"]
    residual_rotation, residual_translation = se3_exp(projected_rigid)
    reconstructed_rotation, reconstructed_translation = rigid_compose(
        bridge_rotation, bridge_translation, residual_rotation, residual_translation
    )
    reconstructed_inverse_rotation, reconstructed_inverse_translation = rigid_inverse(
        reconstructed_rotation, reconstructed_translation
    )
    error_rotation, error_translation = rigid_compose(
        reconstructed_inverse_rotation,
        reconstructed_inverse_translation,
        observed_rotation_t,
        observed_translation_t,
    )
    reconstruction_rigid = se3_log(error_rotation, error_translation)
    reconstruction_chi = torch.remainder(
        bridge_chi + projected_chi - observed_chi_t + torch.pi, 2.0 * torch.pi
    ) - torch.pi

    envelope = torch.as_tensor(endpoint_envelope(progress), dtype=dtype).view(-1, 1)
    confidence_t = torch.as_tensor(confidence, dtype=dtype)
    residual_norm = projection["projected_residual_metric_norm"]
    valid = (
        projection["active_mask"]
        & (confidence_t >= args.min_phase_confidence)
        & (envelope >= args.min_residual_envelope)
        & (residual_norm <= args.max_normal_residual_norm)
    )
    safe_envelope = envelope.clamp_min(args.min_residual_envelope).unsqueeze(-1)
    head_rigid = torch.where(
        valid.unsqueeze(-1), projected_rigid / safe_envelope, torch.zeros_like(projected_rigid)
    )
    head_chi = torch.where(
        valid.unsqueeze(-1), projected_chi / safe_envelope, torch.zeros_like(projected_chi)
    )
    residual_confidence = torch.where(valid, confidence_t, torch.zeros_like(confidence_t))
    return {
        "residual_rot": head_rigid[..., :3].cpu().numpy(),
        "residual_trans": head_rigid[..., 3:].cpu().numpy(),
        "residual_chi": head_chi.cpu().numpy(),
        "residual_valid_mask": valid.cpu().numpy(),
        "residual_confidence": residual_confidence.cpu().numpy(),
        "normal_residual_metric_norm": residual_norm.cpu().numpy(),
        "raw_parallel_cos_abs": projection["raw_parallel_cos_abs"].cpu().numpy(),
        "projected_parallel_cos_abs": projection["projected_parallel_cos_abs"].cpu().numpy(),
        "reconstruction_rotation_error_rad": torch.linalg.norm(
            reconstruction_rigid[..., :3], dim=-1
        ).cpu().numpy(),
        "reconstruction_translation_error_a": torch.linalg.norm(
            reconstruction_rigid[..., 3:], dim=-1
        ).cpu().numpy(),
        "reconstruction_chi_error_rad": reconstruction_chi.abs().cpu().numpy(),
    }


def export_targets(args: argparse.Namespace) -> Dict[str, Any]:
    import mdtraj as md

    required = [
        args.pull_dir / "rmsd_pull.dcd",
        args.pull_dir / "final_pulled.pdb",
        args.pull_dir / "rmsd_pull_metrics.csv",
        args.pull_dir / "rmsd_pull_report.json",
        args.pull_dir / "path_metrics_audit.json",
        args.pull_dir / "atomistic_path_audit.json",
        args.candidate_manifest,
        args.preparation_report,
    ]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing audited path inputs: {missing}")
    path_audit = json.loads((args.pull_dir / "path_metrics_audit.json").read_text())
    atomistic_audit = json.loads((args.pull_dir / "atomistic_path_audit.json").read_text())
    path_audit_passed = bool(
        path_audit.get("passed", path_audit.get("status") == "path_metrics_passed")
    )
    atomistic_audit_passed = bool(
        atomistic_audit.get(
            "passed", atomistic_audit.get("status") == "atomistic_path_passed"
        )
    )
    if not path_audit_passed or not atomistic_audit_passed:
        raise ValueError("Path and atomistic audits must both pass before target export")
    candidate = _load_candidate(args.candidate_manifest, args.transition_id)
    endpoints = candidate["endpoints"]
    apo_residues = parse_pdb_residues(Path(endpoints["apo_structure_path"]))
    holo_residues = parse_pdb_residues(Path(endpoints["holo_structure_path"]))
    apo_by_key = {tuple(residue["key"]): residue for residue in apo_residues}
    holo_by_key = {tuple(residue["key"]): residue for residue in holo_residues}

    preparation = json.loads(args.preparation_report.read_text())
    protein_atoms = int(preparation["protein"]["prepared_protein_atoms"])
    solute_atoms = int(preparation["system"]["pre_solvent_atoms"])
    metrics = _load_metric_rows(args.pull_dir / "rmsd_pull_metrics.csv")
    trajectory = md.load(
        str(args.pull_dir / "rmsd_pull.dcd"), top=str(args.pull_dir / "final_pulled.pdb")
    )
    if trajectory.n_frames != len(metrics):
        raise ValueError(
            f"DCD/metrics mismatch: {trajectory.n_frames} frames vs {len(metrics)} rows"
        )

    topology_residues = []
    keys = []
    for residue in trajectory.topology.residues:
        atom_indices = [atom.index for atom in residue.atoms]
        if not atom_indices or min(atom_indices) >= protein_atoms:
            continue
        key = _topology_residue_key(residue)
        names = {atom.name for atom in residue.atoms}
        if key in apo_by_key and key in holo_by_key and {"N", "CA", "C"} <= names:
            topology_residues.append(residue)
            keys.append(key)
    mapping_fraction = len(keys) / max(len(holo_residues), 1)
    if mapping_fraction < args.min_mapping_fraction:
        raise ValueError(
            f"Mapped only {len(keys)}/{len(holo_residues)} residues ({mapping_fraction:.3f})"
        )
    apo = _extract_endpoint_arrays(apo_residues, keys)
    holo = _extract_endpoint_arrays(holo_residues, keys)
    n_residues = len(keys)

    atom_maps = [{atom.name: atom.index for atom in residue.atoms} for residue in topology_residues]
    n_indices = np.asarray([atoms["N"] for atoms in atom_maps], dtype=np.int32)
    ca_indices = np.asarray([atoms["CA"] for atoms in atom_maps], dtype=np.int32)
    c_indices = np.asarray([atoms["C"] for atoms in atom_maps], dtype=np.int32)
    xyz_angstrom = trajectory.xyz.astype(np.float64) * 10.0

    endpoint_rotation, endpoint_translation = kabsch_transform(
        holo["Ca"], xyz_angstrom[0, ca_indices]
    )
    for endpoint in (apo, holo):
        for name in ("N", "Ca", "C"):
            endpoint[name] = apply_transform(endpoint[name], endpoint_rotation, endpoint_translation)
    endpoint_displacement = np.linalg.norm(holo["Ca"] - apo["Ca"], axis=-1)
    stable_cutoff = float(np.quantile(endpoint_displacement, 0.50))
    stable_mask = endpoint_displacement <= stable_cutoff
    if int(stable_mask.sum()) < 20:
        stable_mask[np.argsort(endpoint_displacement)[: min(20, n_residues)]] = True

    pull_indices = [index for index, row in enumerate(metrics) if row["stage"] == "rmsd_pull"]
    pull_indices.reverse()
    pull_progress = np.asarray([1.0 - float(metrics[index]["progress"]) for index in pull_indices])
    keep = (pull_progress > 1e-8) & (pull_progress < 1.0 - 1e-8)
    pull_indices = [index for index, selected in zip(pull_indices, keep) if selected]
    pull_progress = pull_progress[keep]
    if len(pull_indices) < 3:
        raise ValueError("Fewer than three interior RMSD-pull frames remain")

    interior_n, interior_ca, interior_c = [], [], []
    for frame_index in pull_indices:
        frame_ca = xyz_angstrom[frame_index, ca_indices]
        rotation, translation = kabsch_transform(
            frame_ca[stable_mask], holo["Ca"][stable_mask]
        )
        interior_n.append(apply_transform(xyz_angstrom[frame_index, n_indices], rotation, translation))
        interior_ca.append(apply_transform(frame_ca, rotation, translation))
        interior_c.append(apply_transform(xyz_angstrom[frame_index, c_indices], rotation, translation))
    observed_n = np.concatenate(
        [
            apo["N"][None],
            temporal_moving_average(np.stack(interior_n), args.smoothing_window),
            holo["N"][None],
        ],
        axis=0,
    )
    observed_ca = np.concatenate(
        [
            apo["Ca"][None],
            temporal_moving_average(np.stack(interior_ca), args.smoothing_window),
            holo["Ca"][None],
        ],
        axis=0,
    )
    observed_c = np.concatenate(
        [
            apo["C"][None],
            temporal_moving_average(np.stack(interior_c), args.smoothing_window),
            holo["C"][None],
        ],
        axis=0,
    )
    progress = np.concatenate([[0.0], pull_progress, [1.0]]).astype(np.float64)
    if np.any(np.diff(progress) <= 0.0):
        raise ValueError("Reversed apo-to-holo progress is not strictly increasing")
    observed_rotation, observed_translation = backbone_frames(
        observed_n, observed_ca, observed_c
    )

    trajectory_chi, topology_chi_mask = _trajectory_chi(xyz_angstrom, topology_residues)
    interior_chi = temporal_circular_average(
        trajectory_chi[pull_indices], args.smoothing_window
    )
    chi_mask = apo["chi_mask"] & holo["chi_mask"] & topology_chi_mask
    observed_chi = np.concatenate(
        [apo["chi"][None], interior_chi, holo["chi"][None]], axis=0
    )

    tau_grid = np.linspace(0.0, 1.0, args.phase_grid_size, dtype=np.float64)
    tau_target = np.zeros((len(progress), n_residues), dtype=np.float64)
    phase_confidence = np.zeros_like(tau_target)
    projection_cost = np.zeros_like(tau_target)
    identity_cost = np.zeros_like(tau_target)
    endpoint_motion_norm = np.zeros(n_residues, dtype=np.float64)
    raw_monotonicity_violations = np.zeros(n_residues, dtype=np.int32)
    gamma_grid = smoothstep(tau_grid)
    endpoint_n = np.stack([apo["N"], holo["N"]])
    endpoint_ca = np.stack([apo["Ca"], holo["Ca"]])
    endpoint_c = np.stack([apo["C"], holo["C"]])
    endpoint_chi = np.stack([apo["chi"], holo["chi"]])
    for residue_index in range(n_residues):
        bridge_n = (
            (1.0 - gamma_grid[:, None]) * apo["N"][residue_index]
            + gamma_grid[:, None] * holo["N"][residue_index]
        )
        bridge_ca = (
            (1.0 - gamma_grid[:, None]) * apo["Ca"][residue_index]
            + gamma_grid[:, None] * holo["Ca"][residue_index]
        )
        bridge_c = (
            (1.0 - gamma_grid[:, None]) * apo["C"][residue_index]
            + gamma_grid[:, None] * holo["C"][residue_index]
        )
        bridge_rotation, bridge_translation = backbone_frames(bridge_n, bridge_ca, bridge_c)
        delta_chi = wrap_to_pi(holo["chi"][residue_index] - apo["chi"][residue_index])
        bridge_chi = wrap_to_pi(
            apo["chi"][residue_index][None] + gamma_grid[:, None] * delta_chi[None]
        )
        inferred = infer_monotone_phase(
            observed_rotation[:, residue_index],
            observed_translation[:, residue_index],
            observed_chi[:, residue_index],
            bridge_rotation,
            bridge_translation,
            bridge_chi,
            chi_mask[residue_index],
            progress,
            tau_grid,
            rotation_scale=args.rotation_scale_rad,
            translation_scale=args.translation_scale_a,
            chi_scale=args.chi_scale_rad,
            identity_prior_weight=args.identity_prior_weight,
        )
        tau_target[:, residue_index] = inferred["tau"]
        phase_confidence[:, residue_index] = inferred["confidence"]
        projection_cost[:, residue_index] = inferred["projection_cost"]
        identity_cost[:, residue_index] = inferred["identity_cost"]
        endpoint_motion_norm[residue_index] = inferred["endpoint_motion_norm"]
        raw_monotonicity_violations[residue_index] = inferred["raw_monotonicity_violations"]

    active_mask = endpoint_motion_norm >= args.min_endpoint_motion_norm
    phase_confidence[:, ~active_mask] = 0.0
    phase_confidence[0, active_mask] = 1.0
    phase_confidence[-1, active_mask] = 1.0
    decomposition = _normal_decomposition(
        observed_rotation=observed_rotation,
        observed_translation=observed_translation,
        observed_chi=observed_chi,
        endpoint_n=endpoint_n,
        endpoint_ca=endpoint_ca,
        endpoint_c=endpoint_c,
        endpoint_chi=endpoint_chi,
        tau=tau_target,
        progress=progress,
        node_mask=np.ones(n_residues, dtype=bool),
        chi_mask=chi_mask,
        confidence=phase_confidence,
        args=args,
    )

    ligand_heavy_indices = [
        atom.index for atom in trajectory.topology.atoms
        if protein_atoms <= atom.index < solute_atoms
        and (atom.element is None or atom.element.symbol.upper() != "H")
    ]
    physical_path = trajectory[pull_indices]
    physical_distances = _residue_ligand_distances(
        physical_path, topology_residues, ligand_heavy_indices
    )
    distances = np.concatenate(
        [physical_distances[:1], physical_distances, physical_distances[-1:]], axis=0
    )
    contacts = contact_annotations(
        distances,
        progress,
        contact_distance_angstrom=args.contact_distance_a,
        pocket_distance_angstrom=args.pocket_distance_a,
    )

    interior = (progress > 0.0) & (progress < 1.0)
    active_points = interior[:, None] & active_mask[None, :]
    confident_points = active_points & (phase_confidence >= args.min_phase_confidence)
    residual_valid = decomposition["residual_valid_mask"]
    chi_error = decomposition["reconstruction_chi_error_rad"]
    chi_error_mask = chi_mask[None] & residual_valid[..., None]

    def masked_mean(values: np.ndarray, mask: np.ndarray) -> float:
        selected = np.asarray(values)[np.asarray(mask, dtype=bool)]
        return float(selected.mean()) if selected.size else math.inf

    mean_translation_error = masked_mean(
        decomposition["reconstruction_translation_error_a"], residual_valid
    )
    mean_rotation_error = masked_mean(
        decomposition["reconstruction_rotation_error_rad"], residual_valid
    )
    mean_chi_error = masked_mean(chi_error, chi_error_mask)
    max_projected_parallel_cos = masked_mean(
        decomposition["projected_parallel_cos_abs"], residual_valid
    )
    median_confidence = (
        float(np.median(phase_confidence[confident_points]))
        if np.any(confident_points) else 0.0
    )
    tau_identity_mae = masked_mean(
        np.abs(tau_target - progress[:, None]), active_points
    )
    contact_event_count = int(
        contacts["formed_contact_mask"].sum() + contacts["release_mask"].sum()
    )
    active_point_count = int(active_points.sum())
    confident_phase_density = int(confident_points.sum()) / max(active_point_count, 1)
    residual_candidate_points = active_points & (
        endpoint_envelope(progress)[:, None] >= args.min_residual_envelope
    )
    residual_candidate_count = int(residual_candidate_points.sum())
    residual_supervision_density = int(residual_valid.sum()) / max(
        residual_candidate_count, 1
    )
    checks = {
        "upstream_path_audits": path_audit_passed and atomistic_audit_passed,
        "residue_mapping": mapping_fraction >= args.min_mapping_fraction,
        "monotone_tau": bool(np.all(np.diff(tau_target, axis=0) >= -1e-8)),
        "active_phase_support": int(active_mask.sum()) >= 5,
        "phase_confidence": median_confidence >= args.min_phase_confidence,
        "residual_support": int(residual_valid.sum()) >= 10,
        "phase_supervision_density": (
            confident_phase_density >= args.min_supervision_density
        ),
        "residual_supervision_density": (
            residual_supervision_density >= args.min_supervision_density
        ),
        "normal_projection": max_projected_parallel_cos <= 1e-4,
        "translation_reconstruction": mean_translation_error <= args.max_mean_reconstruction_translation_a,
        "rotation_reconstruction": mean_rotation_error <= args.max_mean_reconstruction_rotation_rad,
        "chi_reconstruction": mean_chi_error <= args.max_mean_reconstruction_chi_rad,
    }
    passed = all(checks.values())
    sample_id = args.sample_id or Path(endpoints["holo_structure_path"]).parent.name
    args.output_dir.mkdir(parents=True, exist_ok=True)
    cache_path = args.output_dir / f"{_safe_sample_id(sample_id)}.npz"
    np.savez_compressed(
        cache_path,
        schema_version=np.array(SCHEMA_VERSION),
        source=np.array("silver_global_ca_rmsd_pull"),
        sample_id=np.array(sample_id),
        transition_id=np.array(args.transition_id),
        evidence_tier=np.array("silver_enhanced_sampling"),
        bridge_mode=np.array("cartesian_backbone"),
        residual_envelope=np.array("poly"),
        rotation_metric_scale=np.array(args.rotation_scale_rad, dtype=np.float32),
        translation_metric_scale=np.array(args.translation_scale_a, dtype=np.float32),
        chi_metric_scale=np.array(args.chi_scale_rad, dtype=np.float32),
        n_residues=np.array(n_residues, dtype=np.int32),
        t_values=progress.astype(np.float32),
        tau_target=tau_target.astype(np.float32),
        phase_confidence=phase_confidence.astype(np.float32),
        projection_cost=projection_cost.astype(np.float32),
        identity_cost=identity_cost.astype(np.float32),
        node_mask=np.ones(n_residues, dtype=bool),
        chi_mask=chi_mask,
        w_res=contacts["pocket_mask"].astype(np.float32),
        pocket_mask=contacts["pocket_mask"],
        active_mask=active_mask,
        motion_active=active_mask,
        approach_mask=contacts["approach_mask"],
        formed_contact_mask=contacts["formed_contact_mask"],
        release_mask=contacts["release_mask"],
        transient_contact_mask=contacts["transient_contact_mask"],
        contact_event_progress=contacts["contact_event_progress"].astype(np.float32),
        contact_distance_angstrom=distances.astype(np.float32),
        endpoint_motion_metric_norm=endpoint_motion_norm.astype(np.float32),
        residual_rot=decomposition["residual_rot"].astype(np.float32),
        residual_trans=decomposition["residual_trans"].astype(np.float32),
        residual_chi=decomposition["residual_chi"].astype(np.float32),
        residual_valid_mask=residual_valid,
        residual_confidence=decomposition["residual_confidence"].astype(np.float32),
        normal_residual_metric_norm=decomposition["normal_residual_metric_norm"].astype(np.float32),
        raw_parallel_cos_abs=decomposition["raw_parallel_cos_abs"].astype(np.float32),
        projected_parallel_cos_abs=decomposition["projected_parallel_cos_abs"].astype(np.float32),
        raw_monotonicity_violations=raw_monotonicity_violations,
        residue_chain=np.asarray([key[0] for key in keys]),
        residue_number=np.asarray([key[1] for key in keys], dtype=np.int32),
        residue_name=apo["resnames"],
    )
    audit = {
        "schema_version": SCHEMA_VERSION,
        "status": "md_phase_normal_targets_passed" if passed else "md_phase_normal_targets_failed",
        "passed": passed,
        "checks": checks,
        "metrics": {
            "frames": len(progress),
            "residues": n_residues,
            "mapping_fraction": mapping_fraction,
            "active_residues": int(active_mask.sum()),
            "confident_phase_points": int(confident_points.sum()),
            "valid_residual_points": int(residual_valid.sum()),
            "active_interior_points": active_point_count,
            "residual_candidate_points": residual_candidate_count,
            "phase_supervision_density": confident_phase_density,
            "residual_supervision_density": residual_supervision_density,
            "smoothing_window_frames": args.smoothing_window,
            "median_phase_confidence": median_confidence,
            "tau_identity_mae": tau_identity_mae,
            "mean_reconstruction_translation_angstrom": mean_translation_error,
            "mean_reconstruction_rotation_rad": mean_rotation_error,
            "mean_reconstruction_chi_rad": mean_chi_error,
            "mean_projected_parallel_cos_abs": max_projected_parallel_cos,
            "formed_contact_residues": int(contacts["formed_contact_mask"].sum()),
            "released_contact_residues": int(contacts["release_mask"].sum()),
            "transient_contact_residues": int(contacts["transient_contact_mask"].sum()),
            "contact_event_residues": contact_event_count,
        },
        "usage": {
            "phase_supervision": passed,
            "geometry_supervision": passed,
            "heldout_benchmark": False,
            "kinetics_claims": False,
            "evidence_weight": "low_silver_pilot",
        },
        "outputs": {"cache": str(cache_path)},
    }
    audit_path = args.output_dir / "target_audit.json"
    audit_path.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n")
    (args.output_dir / "manifest.jsonl").write_text(
        json.dumps(
            {
                "sample_id": sample_id,
                "transition_id": args.transition_id,
                "schema_version": SCHEMA_VERSION,
                "relative_path": cache_path.name,
                "status": audit["status"],
                "phase_supervision": passed,
            },
            sort_keys=True,
        ) + "\n"
    )
    return audit


def main() -> None:
    args = parse_args()
    if args.phase_grid_size < 21 or args.phase_grid_size % 2 == 0:
        raise ValueError("phase-grid-size must be an odd integer >= 21")
    if args.smoothing_window < 1 or args.smoothing_window % 2 == 0:
        raise ValueError("smoothing-window must be an odd integer >= 1")
    if not (0.0 < args.min_supervision_density <= 1.0):
        raise ValueError("min-supervision-density must be in (0, 1]")
    for name in ("rotation_scale_rad", "translation_scale_a", "chi_scale_rad"):
        if float(getattr(args, name)) <= 0.0:
            raise ValueError(f"{name} must be > 0")
    result = export_targets(args)
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)
    if not result["passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
