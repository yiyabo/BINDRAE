#!/usr/bin/env python3
"""Measure whether unbiased atomistic forces identify Path-4 normal residuals.

This is an information-upper-bound diagnostic, not a deployable model result.
Forces are evaluated with the unmodified NPT OpenMM system at accepted MD
replica frames, then aggregated into residue-local rigid-body generalized
forces and projected away from the endpoint-bridge tangent.  A single residual
amplitude is fitted on training systems and evaluated on disjoint validation
systems.  The frozen test split must never be passed to this script.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import sys
from collections import defaultdict
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Sequence

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.analyze_md_replica_consistency import system_id_from_sample_id  # noqa: E402
from scripts.export_md_phase_normal_targets import (  # noqa: E402
    _exact_sequence_alignment_pairs,
    _extract_residue_arrays,
    _load_candidate,
    _load_metric_rows,
    _residue_lookup,
    _single_chain_keys,
    _topology_residue_key,
    _triple_sequence_alignment,
    apply_transform,
    backbone_frames,
    canonical_resname,
    endpoint_envelope,
    kabsch_transform,
    parse_pdb_residues,
    smoothstep,
)


CHANNEL_SLICES = {
    "rigid": slice(0, 6),
    "rotation": slice(0, 3),
    "translation": slice(3, 6),
}


@dataclass
class ChannelSums:
    count: int = 0
    target_sq: float = 0.0
    direction_dot: float = 0.0
    cosine_sum: float = 0.0
    positive_count: int = 0

    def add(self, target: np.ndarray, direction: np.ndarray) -> None:
        target_norm = float(np.linalg.norm(target))
        direction_norm = float(np.linalg.norm(direction))
        if target_norm <= 1e-10 or direction_norm <= 1e-10:
            return
        unit_direction = direction / direction_norm
        dot = float(np.dot(unit_direction, target))
        self.count += 1
        self.target_sq += target_norm * target_norm
        self.direction_dot += dot
        self.cosine_sum += dot / target_norm
        self.positive_count += int(dot > 0.0)

    def merge(self, other: "ChannelSums") -> None:
        self.count += other.count
        self.target_sq += other.target_sq
        self.direction_dot += other.direction_dot
        self.cosine_sum += other.cosine_sum
        self.positive_count += other.positive_count


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument(
        "--source-manifest",
        type=Path,
        action="append",
        required=True,
        help=(
            "Manifest whose source_dir points into an original replica collection. "
            "Repeat for each immutable source lane."
        ),
    )
    parser.add_argument("--fit-system-list", type=Path, required=True)
    parser.add_argument("--eval-system-list", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-fit-systems", type=int, default=0)
    parser.add_argument("--max-eval-systems", type=int, default=0)
    parser.add_argument("--max-replicas-per-system", type=int, default=1)
    parser.add_argument("--frames-per-replica", type=int, default=9)
    parser.add_argument(
        "--force-window",
        type=int,
        default=1,
        help="Odd number of neighboring accepted pull frames averaged per target frame.",
    )
    parser.add_argument("--min-residual-confidence", type=float, default=0.0)
    parser.add_argument("--platform", choices=("CPU", "CUDA", "OpenCL"), default="CUDA")
    parser.add_argument("--device-index", default="0")
    parser.add_argument("--cpu-threads", type=int, default=8)
    parser.add_argument("--bootstrap-samples", type=int, default=10000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260719)
    parser.add_argument("--random-seed", type=int, default=20260719)
    parser.add_argument(
        "--minimum-practical-relative-reduction",
        type=float,
        default=0.02,
        help="Minimum validation system-macro residual-MSE reduction for a positive gate.",
    )
    return parser.parse_args()


def read_jsonl(path: Path) -> list[Dict[str, Any]]:
    records = []
    with path.open() as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {path}:{line_number}") from exc
    return records


def load_systems(path: Path) -> list[str]:
    systems = [line.strip() for line in path.read_text().splitlines() if line.strip()]
    if not systems:
        raise ValueError(f"System list is empty: {path}")
    if len(systems) != len(set(systems)):
        raise ValueError(f"System list contains duplicates: {path}")
    return systems


def select_systems(systems: Sequence[str], maximum: int, seed: int) -> list[str]:
    systems = sorted(systems)
    if maximum <= 0 or maximum >= len(systems):
        return systems
    rng = np.random.default_rng(seed)
    indices = np.sort(rng.choice(len(systems), size=maximum, replace=False))
    return [systems[int(index)] for index in indices]


def select_cache_frame_indices(n_frames: int, count: int) -> np.ndarray:
    if n_frames < 3:
        raise ValueError("At least one interior frame is required")
    interior = np.arange(1, n_frames - 1, dtype=np.int64)
    if count <= 0 or count >= interior.size:
        return interior
    positions = np.linspace(0, interior.size - 1, count)
    return np.unique(interior[np.rint(positions).astype(np.int64)])


def project_dual_force_normal(
    force_dual: np.ndarray, tangent: np.ndarray, eps: float = 1e-12
) -> np.ndarray:
    """Project a normalized dual force away from a normalized bridge tangent."""
    force_dual = np.asarray(force_dual, dtype=np.float64)
    tangent = np.asarray(tangent, dtype=np.float64)
    tangent_sq = np.sum(tangent * tangent, axis=-1, keepdims=True)
    coefficient = np.sum(force_dual * tangent, axis=-1, keepdims=True) / np.maximum(
        tangent_sq, eps
    )
    projected = force_dual - coefficient * tangent
    return np.where(tangent_sq > eps, projected, force_dual)


def fitted_nonnegative_amplitude(sums: Iterable[ChannelSums]) -> float:
    system_optima = [
        item.direction_dot / item.count
        for item in sums
        if item.count > 0
    ]
    if not system_optima:
        return 0.0
    return max(0.0, float(np.mean(system_optima)))


def channel_metrics(sums: ChannelSums, amplitude: float) -> Dict[str, float | int]:
    if sums.count == 0 or sums.target_sq <= 0.0:
        return {
            "points": 0,
            "mean_cosine": math.nan,
            "positive_fraction": math.nan,
            "baseline_mse": math.nan,
            "force_direction_mse": math.nan,
            "relative_mse_reduction": math.nan,
        }
    predicted_sse = (
        sums.target_sq
        - 2.0 * amplitude * sums.direction_dot
        + amplitude * amplitude * sums.count
    )
    return {
        "points": sums.count,
        "mean_cosine": sums.cosine_sum / sums.count,
        "positive_fraction": sums.positive_count / sums.count,
        "baseline_mse": sums.target_sq / sums.count,
        "force_direction_mse": predicted_sse / sums.count,
        "relative_mse_reduction": (sums.target_sq - predicted_sse)
        / sums.target_sq,
    }


def bootstrap_mean_interval(
    values: Sequence[float], samples: int, seed: int
) -> Dict[str, float | int]:
    array = np.asarray([value for value in values if math.isfinite(value)], dtype=np.float64)
    if array.size == 0:
        return {"systems": 0, "mean": math.nan, "ci_low": math.nan, "ci_high": math.nan}
    rng = np.random.default_rng(seed)
    draws = rng.choice(array, size=(samples, array.size), replace=True).mean(axis=1)
    return {
        "systems": int(array.size),
        "mean": float(array.mean()),
        "ci_low": float(np.quantile(draws, 0.025)),
        "ci_high": float(np.quantile(draws, 0.975)),
    }


def _source_collection_root(source_dir: Path) -> Path:
    try:
        index = source_dir.parts.index("targets")
    except ValueError as exc:
        raise ValueError(f"source_dir does not point into an original targets tree: {source_dir}") from exc
    return Path(*source_dir.parts[:index])


def load_source_records(
    manifests: Sequence[Path], selected_samples: set[str]
) -> Dict[str, Dict[str, Any]]:
    source_dirs: Dict[str, Path] = {}
    for manifest in manifests:
        for record in read_jsonl(manifest):
            sample_id = str(record.get("sample_id", ""))
            if sample_id not in selected_samples:
                continue
            source_dir = Path(str(record.get("source_dir", "")))
            if not source_dir.parts:
                raise ValueError(f"{manifest} lacks source_dir for {sample_id}")
            if sample_id in source_dirs and source_dirs[sample_id] != source_dir:
                raise ValueError(f"Conflicting source_dir records for {sample_id}")
            source_dirs[sample_id] = source_dir

    missing = sorted(selected_samples - source_dirs.keys())
    if missing:
        raise ValueError(f"Source manifests do not cover {len(missing)} samples; first={missing[:5]}")

    matrix_cache: Dict[Path, Dict[str, Dict[str, Any]]] = {}
    output: Dict[str, Dict[str, Any]] = {}
    for sample_id, source_dir in source_dirs.items():
        root = _source_collection_root(source_dir)
        matrix_path = root / "replica_matrix.jsonl"
        if matrix_path not in matrix_cache:
            if not matrix_path.is_file():
                raise FileNotFoundError(f"Missing source replica matrix: {matrix_path}")
            indexed = {}
            for record in read_jsonl(matrix_path):
                current = str(record.get("sample_id", ""))
                if current:
                    indexed[current] = record
            matrix_cache[matrix_path] = indexed
        try:
            output[sample_id] = matrix_cache[matrix_path][sample_id]
        except KeyError as exc:
            raise ValueError(f"{sample_id} is absent from {matrix_path}") from exc
    return output


def load_cache_records(cache_dir: Path) -> Dict[str, Dict[str, Any]]:
    manifest = cache_dir / "manifest.jsonl"
    if not manifest.is_file():
        raise FileNotFoundError(f"Missing immutable cache manifest: {manifest}")
    output = {}
    for record in read_jsonl(manifest):
        sample_id = str(record.get("sample_id", ""))
        relative_path = Path(str(record.get("relative_path", "")))
        target_path = cache_dir / relative_path
        if not sample_id or not target_path.is_file():
            raise ValueError(f"Invalid cache record for {sample_id!r}: {target_path}")
        if sample_id in output:
            raise ValueError(f"Duplicate cache sample: {sample_id}")
        copied = dict(record)
        copied["target_path"] = target_path
        copied["system_id"] = system_id_from_sample_id(sample_id)
        output[sample_id] = copied
    return output


def _build_residue_mapping(
    target: Mapping[str, np.ndarray],
    trajectory: Any,
    protein_atoms: int,
    apo_path: Path,
    holo_path: Path,
) -> Dict[str, Any]:
    canonical_names = [canonical_resname(value) for value in target["residue_name"].tolist()]
    canonical_keys = [
        (str(chain), int(number), "")
        for chain, number in zip(target["residue_chain"], target["residue_number"])
    ]
    apo_residues = parse_pdb_residues(apo_path)
    holo_residues = parse_pdb_residues(holo_path)
    candidate_topology_residues = []
    for residue in trajectory.topology.residues:
        atom_indices = [atom.index for atom in residue.atoms]
        if not atom_indices or min(atom_indices) >= protein_atoms:
            continue
        if {"N", "CA", "C"} <= {atom.name for atom in residue.atoms}:
            candidate_topology_residues.append(residue)

    apo_keys = [tuple(residue["key"]) for residue in apo_residues]
    holo_keys = [tuple(residue["key"]) for residue in holo_residues]
    topology_keys = [_topology_residue_key(residue) for residue in candidate_topology_residues]
    topology_names = [canonical_resname(residue.name) for residue in candidate_topology_residues]
    ignore_chain = (
        _single_chain_keys(apo_keys)
        and _single_chain_keys(holo_keys)
        and _single_chain_keys(topology_keys)
    )
    if ignore_chain:
        aligned = _triple_sequence_alignment(
            [residue["resname"] for residue in apo_residues],
            [residue["resname"] for residue in holo_residues],
            topology_names,
        )
        canonical_to_apo = _exact_sequence_alignment_pairs(
            canonical_names, [residue["resname"] for residue in apo_residues]
        )
        canonical_by_apo = {apo_index: canonical for canonical, apo_index in canonical_to_apo}
        aligned = [entry for entry in aligned if entry[0] in canonical_by_apo]
        canonical_indices = [canonical_by_apo[apo_index] for apo_index, _, _ in aligned]
        selected_apo = [apo_residues[apo_index] for apo_index, _, _ in aligned]
        selected_holo = [holo_residues[holo_index] for _, holo_index, _ in aligned]
        topology_residues = [candidate_topology_residues[topology_index] for _, _, topology_index in aligned]
    else:
        apo_by_key = _residue_lookup(apo_residues, label="apo", ignore_chain=False)
        holo_by_key = _residue_lookup(holo_residues, label="holo", ignore_chain=False)
        canonical_by_key = {key: index for index, key in enumerate(canonical_keys)}
        canonical_indices, selected_apo, selected_holo, topology_residues = [], [], [], []
        for residue, key, name in zip(
            candidate_topology_residues, topology_keys, topology_names
        ):
            apo_residue = apo_by_key.get(key)
            holo_residue = holo_by_key.get(key)
            canonical_index = canonical_by_key.get(key)
            if apo_residue is None or holo_residue is None or canonical_index is None:
                continue
            if apo_residue["resname"] != holo_residue["resname"] or apo_residue["resname"] != name:
                continue
            canonical_indices.append(canonical_index)
            selected_apo.append(apo_residue)
            selected_holo.append(holo_residue)
            topology_residues.append(residue)

    canonical_indices_array = np.asarray(canonical_indices, dtype=np.int64)
    expected = np.flatnonzero(np.asarray(target["node_mask"], dtype=bool))
    if not np.array_equal(canonical_indices_array, expected):
        raise ValueError(
            "Force diagnostic residue mapping does not reproduce the frozen target axis: "
            f"mapped={len(canonical_indices_array)}, expected={len(expected)}"
        )
    return {
        "canonical_indices": canonical_indices_array,
        "apo": _extract_residue_arrays(selected_apo),
        "holo": _extract_residue_arrays(selected_holo),
        "topology_residues": topology_residues,
    }


def _bridge_geometry(
    apo: Mapping[str, np.ndarray],
    holo: Mapping[str, np.ndarray],
    tau: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    gamma = smoothstep(np.asarray(tau, dtype=np.float64))[..., None]
    n_coord = (1.0 - gamma) * apo["N"] + gamma * holo["N"]
    ca_coord = (1.0 - gamma) * apo["Ca"] + gamma * holo["Ca"]
    c_coord = (1.0 - gamma) * apo["C"] + gamma * holo["C"]
    rotation, translation = backbone_frames(n_coord, ca_coord, c_coord)
    return rotation, translation, ca_coord


@lru_cache(maxsize=1)
def _load_se3_log() -> Any:
    path = PROJECT_ROOT / "src/stage2/modules/se3.py"
    spec = importlib.util.spec_from_file_location("bindrae_force_diag_se3", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load Stage-2 SE(3) utilities from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.se3_log


def _bridge_tangent(
    apo: Mapping[str, np.ndarray],
    holo: Mapping[str, np.ndarray],
    tau: np.ndarray,
) -> np.ndarray:
    import torch

    epsilon = 1e-3
    tau_low = np.clip(tau - epsilon, 0.0, 1.0)
    tau_high = np.clip(tau + epsilon, 0.0, 1.0)
    low_rotation, low_translation, _ = _bridge_geometry(apo, holo, tau_low)
    high_rotation, high_translation, _ = _bridge_geometry(apo, holo, tau_high)
    relative_rotation = np.swapaxes(low_rotation, -1, -2) @ high_rotation
    relative_translation = np.einsum(
        "nij,nj->ni", np.swapaxes(low_rotation, -1, -2), high_translation - low_translation
    )
    with torch.no_grad():
        tangent = _load_se3_log()(
            torch.as_tensor(relative_rotation, dtype=torch.float64),
            torch.as_tensor(relative_translation, dtype=torch.float64),
        ).cpu().numpy()
    denominator = np.maximum(tau_high - tau_low, 1e-6)[:, None]
    return tangent / denominator


def _force_platform(args: argparse.Namespace) -> tuple[Any, Dict[str, str]]:
    from openmm import Platform

    platform = Platform.getPlatformByName(args.platform)
    properties: Dict[str, str] = {}
    if args.platform in {"CUDA", "OpenCL"}:
        properties["Precision"] = "mixed"
        properties["DeviceIndex"] = str(args.device_index)
    elif args.platform == "CPU":
        properties["Threads"] = str(args.cpu_threads)
    return platform, properties


def _make_force_context(system_path: Path, args: argparse.Namespace) -> tuple[Any, Any]:
    from openmm import Context, VerletIntegrator, XmlSerializer, unit

    system = XmlSerializer.deserialize(system_path.read_text())
    integrator = VerletIntegrator(0.001 * unit.picoseconds)
    platform, properties = _force_platform(args)
    return Context(system, integrator, platform, properties), integrator


def _context_force(
    context: Any, xyz_nm: np.ndarray, box_nm: np.ndarray | None
) -> tuple[np.ndarray, float]:
    from openmm import Vec3, unit

    if box_nm is not None and np.isfinite(box_nm).all():
        vectors = [Vec3(*row) * unit.nanometer for row in box_nm]
        context.setPeriodicBoxVectors(*vectors)
    context.setPositions(unit.Quantity(np.asarray(xyz_nm, dtype=np.float64), unit.nanometer))
    state = context.getState(getForces=True, getEnergy=True)
    forces = np.asarray(
        state.getForces(asNumpy=True).value_in_unit(
            unit.kilojoule_per_mole / unit.nanometer
        ),
        dtype=np.float64,
    )
    energy = float(state.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole))
    if not np.isfinite(forces).all() or not math.isfinite(energy):
        raise RuntimeError("OpenMM returned non-finite force or potential energy")
    return forces, energy


def _aggregate_residue_force(
    positions_a: np.ndarray,
    forces_kj_mol_nm: np.ndarray,
    residues: Sequence[Any],
    alignment_rotation: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    aligned_positions = positions_a @ alignment_rotation
    aligned_force_a = (forces_kj_mol_nm / 10.0) @ alignment_rotation
    net_forces, torques = [], []
    for residue in residues:
        atoms = list(residue.atoms)
        heavy = [
            atom.index
            for atom in atoms
            if atom.element is None or atom.element.symbol.upper() != "H"
        ]
        atom_by_name = {atom.name: atom.index for atom in atoms}
        if "CA" not in atom_by_name or not heavy:
            raise ValueError(f"Residue {residue} lacks CA or heavy atoms")
        center = aligned_positions[atom_by_name["CA"]]
        residue_force = aligned_force_a[heavy]
        offsets = aligned_positions[heavy] - center
        net_forces.append(residue_force.sum(axis=0))
        torques.append(np.cross(offsets, residue_force).sum(axis=0))
    return np.stack(net_forces), np.stack(torques)


def _evaluate_replica(
    cache_record: Mapping[str, Any],
    source_record: Mapping[str, Any],
    args: argparse.Namespace,
) -> Dict[str, ChannelSums]:
    import mdtraj as md

    target_path = Path(cache_record["target_path"])
    pull_dir = PROJECT_ROOT / Path(str(source_record["pull_dir"]))
    npt_dir = PROJECT_ROOT / Path(str(source_record["npt_dir"]))
    candidate_manifest = PROJECT_ROOT / Path(str(source_record["candidate_manifest"]))
    preparation_report = PROJECT_ROOT / Path(str(source_record["preparation_report"]))
    required = [
        pull_dir / "rmsd_pull.dcd",
        pull_dir / "final_pulled.pdb",
        pull_dir / "rmsd_pull_metrics.csv",
        npt_dir / "npt_system.xml",
        candidate_manifest,
        preparation_report,
    ]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing force diagnostic inputs: {missing}")

    with np.load(target_path, allow_pickle=False) as loaded:
        target = {key: np.asarray(loaded[key]) for key in loaded.files}
    trajectory = md.load(
        str(pull_dir / "rmsd_pull.dcd"), top=str(pull_dir / "final_pulled.pdb")
    )
    metrics = _load_metric_rows(pull_dir / "rmsd_pull_metrics.csv")
    if trajectory.n_frames != len(metrics):
        raise ValueError("DCD and pull metrics have different frame counts")
    preparation = json.loads(preparation_report.read_text())
    protein_atoms = int(preparation["protein"]["prepared_protein_atoms"])
    candidate = _load_candidate(candidate_manifest, str(source_record["transition_id"]))
    endpoints = candidate["endpoints"]
    mapping = _build_residue_mapping(
        target,
        trajectory,
        protein_atoms,
        PROJECT_ROOT / Path(str(endpoints["apo_structure_path"])),
        PROJECT_ROOT / Path(str(endpoints["holo_structure_path"])),
    )
    canonical_indices = mapping["canonical_indices"]
    apo, holo = mapping["apo"], mapping["holo"]
    topology_residues = mapping["topology_residues"]
    atom_maps = [{atom.name: atom.index for atom in residue.atoms} for residue in topology_residues]
    ca_indices = np.asarray([atoms["CA"] for atoms in atom_maps], dtype=np.int64)
    xyz_a = np.asarray(trajectory.xyz, dtype=np.float64) * 10.0
    endpoint_rotation, endpoint_translation = kabsch_transform(holo["Ca"], xyz_a[0, ca_indices])
    for endpoint in (apo, holo):
        for name in ("N", "Ca", "C"):
            endpoint[name] = apply_transform(
                endpoint[name], endpoint_rotation, endpoint_translation
            )
    endpoint_displacement = np.linalg.norm(holo["Ca"] - apo["Ca"], axis=-1)
    stable_cutoff = float(np.quantile(endpoint_displacement, 0.50))
    stable_mask = endpoint_displacement <= stable_cutoff
    if int(stable_mask.sum()) < 20:
        stable_mask[np.argsort(endpoint_displacement)[: min(20, len(stable_mask))]] = True

    pull_indices = [index for index, row in enumerate(metrics) if row["stage"] == "rmsd_pull"]
    pull_indices.reverse()
    pull_progress = np.asarray([1.0 - float(metrics[index]["progress"]) for index in pull_indices])
    keep = (pull_progress > 1e-8) & (pull_progress < 1.0 - 1e-8)
    pull_indices = [index for index, selected in zip(pull_indices, keep) if selected]
    if len(pull_indices) != len(target["t_values"]) - 2:
        raise ValueError(
            f"Target/trajectory interior mismatch: {len(target['t_values']) - 2} vs {len(pull_indices)}"
        )

    selected_cache_indices = select_cache_frame_indices(
        len(target["t_values"]), args.frames_per_replica
    )
    half_window = args.force_window // 2
    context, integrator = _make_force_context(npt_dir / "npt_system.xml", args)
    frame_force_cache: Dict[int, tuple[np.ndarray, np.ndarray]] = {}
    output = {name: ChannelSums() for name in CHANNEL_SLICES}
    try:
        for cache_index in selected_cache_indices:
            center_interior = int(cache_index - 1)
            neighbor_interior = range(
                max(0, center_interior - half_window),
                min(len(pull_indices), center_interior + half_window + 1),
            )
            net_values, torque_values = [], []
            for interior_index in neighbor_interior:
                dcd_index = int(pull_indices[interior_index])
                if dcd_index not in frame_force_cache:
                    box = None
                    if trajectory.unitcell_vectors is not None:
                        box = np.asarray(trajectory.unitcell_vectors[dcd_index], dtype=np.float64)
                    forces, _ = _context_force(
                        context, trajectory.xyz[dcd_index], box
                    )
                    frame_ca = xyz_a[dcd_index, ca_indices]
                    alignment_rotation, _ = kabsch_transform(
                        frame_ca[stable_mask], holo["Ca"][stable_mask]
                    )
                    frame_force_cache[dcd_index] = _aggregate_residue_force(
                        xyz_a[dcd_index],
                        forces,
                        topology_residues,
                        alignment_rotation,
                    )
                net_force, torque = frame_force_cache[dcd_index]
                net_values.append(net_force)
                torque_values.append(torque)
            net_force_world = np.mean(net_values, axis=0)
            torque_world = np.mean(torque_values, axis=0)

            tau = np.asarray(target["tau_target"][cache_index, canonical_indices], dtype=np.float64)
            bridge_rotation, _, _ = _bridge_geometry(apo, holo, tau)
            tangent = _bridge_tangent(apo, holo, tau)
            net_force_local = np.einsum("ni,nij->nj", net_force_world, bridge_rotation)
            torque_local = np.einsum("ni,nij->nj", torque_world, bridge_rotation)

            rotation_scale = float(target["rotation_metric_scale"])
            translation_scale = float(target["translation_metric_scale"])
            scales = np.asarray(
                [rotation_scale] * 3 + [translation_scale] * 3,
                dtype=np.float64,
            )
            dual_force = np.concatenate([torque_local, net_force_local], axis=-1) * scales
            tangent_normalized = tangent / scales
            projected_force = project_dual_force_normal(dual_force, tangent_normalized)
            envelope = float(
                endpoint_envelope(
                    np.asarray([target["t_values"][cache_index]], dtype=np.float64),
                    str(target["residual_envelope"]),
                )[0]
            )
            target_rigid = envelope * np.concatenate(
                [
                    target["residual_rot"][cache_index, canonical_indices],
                    target["residual_trans"][cache_index, canonical_indices],
                ],
                axis=-1,
            )
            target_normalized = target_rigid / scales
            valid = np.asarray(
                target["residual_valid_mask"][cache_index, canonical_indices], dtype=bool
            )
            valid &= np.asarray(
                target["residual_confidence"][cache_index, canonical_indices]
                >= args.min_residual_confidence,
                dtype=bool,
            )
            for residue_index in np.flatnonzero(valid):
                for name, channel_slice in CHANNEL_SLICES.items():
                    output[name].add(
                        target_normalized[residue_index, channel_slice],
                        projected_force[residue_index, channel_slice],
                    )
    finally:
        del context
        del integrator
    return output


def _run_split(
    name: str,
    systems: Sequence[str],
    grouped_cache: Mapping[str, Sequence[Mapping[str, Any]]],
    source_records: Mapping[str, Mapping[str, Any]],
    args: argparse.Namespace,
) -> Dict[str, Dict[str, ChannelSums]]:
    output: Dict[str, Dict[str, ChannelSums]] = {}
    for system_index, system_id in enumerate(systems, start=1):
        records = sorted(
            grouped_cache.get(system_id, []), key=lambda record: str(record["sample_id"])
        )[: args.max_replicas_per_system]
        if not records:
            raise ValueError(f"No frozen cache replicas for {system_id}")
        system_sums = {channel: ChannelSums() for channel in CHANNEL_SLICES}
        for record in records:
            sample_id = str(record["sample_id"])
            replica_sums = _evaluate_replica(record, source_records[sample_id], args)
            for channel in CHANNEL_SLICES:
                system_sums[channel].merge(replica_sums[channel])
        output[system_id] = system_sums
        print(
            f"[{name}] {system_index}/{len(systems)} {system_id}: "
            f"rigid_points={system_sums['rigid'].count}",
            flush=True,
        )
    return output


def main() -> None:
    args = parse_args()
    if args.force_window <= 0 or args.force_window % 2 == 0:
        raise ValueError("force-window must be a positive odd integer")
    if args.max_replicas_per_system <= 0:
        raise ValueError("max-replicas-per-system must be positive")
    fit_systems = select_systems(
        load_systems(args.fit_system_list), args.max_fit_systems, args.random_seed
    )
    eval_systems = select_systems(
        load_systems(args.eval_system_list), args.max_eval_systems, args.random_seed + 1
    )
    overlap = set(fit_systems) & set(eval_systems)
    if overlap:
        raise ValueError(f"Fit/eval system overlap is forbidden: {sorted(overlap)[:5]}")

    cache_records = load_cache_records(args.cache_dir)
    grouped_cache: Dict[str, list[Dict[str, Any]]] = defaultdict(list)
    for record in cache_records.values():
        grouped_cache[str(record["system_id"])].append(record)
    selected_samples = set()
    for system_id in [*fit_systems, *eval_systems]:
        records = sorted(
            grouped_cache.get(system_id, []), key=lambda record: str(record["sample_id"])
        )[: args.max_replicas_per_system]
        selected_samples.update(str(record["sample_id"]) for record in records)
    source_records = load_source_records(args.source_manifest, selected_samples)

    fit = _run_split("fit", fit_systems, grouped_cache, source_records, args)
    evaluation = _run_split("eval", eval_systems, grouped_cache, source_records, args)
    result: Dict[str, Any] = {
        "schema_version": "bindrae_md_force_residual_alignment_v1",
        "diagnostic_scope": "same-replica actual-state unbiased-force upper bound",
        "deployable": False,
        "test_split_touched": False,
        "settings": {
            "fit_systems": len(fit_systems),
            "eval_systems": len(eval_systems),
            "max_replicas_per_system": args.max_replicas_per_system,
            "frames_per_replica": args.frames_per_replica,
            "force_window": args.force_window,
            "platform": args.platform,
        },
        "channels": {},
    }
    for channel_index, channel in enumerate(CHANNEL_SLICES):
        amplitude = fitted_nonnegative_amplitude(
            fit[system_id][channel] for system_id in fit_systems
        )
        fit_total = ChannelSums()
        eval_total = ChannelSums()
        for system_id in fit_systems:
            fit_total.merge(fit[system_id][channel])
        per_system = {}
        for system_id in eval_systems:
            current = evaluation[system_id][channel]
            eval_total.merge(current)
            per_system[system_id] = channel_metrics(current, amplitude)
        reductions = [
            float(metrics["relative_mse_reduction"])
            for metrics in per_system.values()
        ]
        cosines = [float(metrics["mean_cosine"]) for metrics in per_system.values()]
        positive_fractions = [
            float(metrics["positive_fraction"]) for metrics in per_system.values()
        ]
        result["channels"][channel] = {
            "fit_amplitude": amplitude,
            "fit_micro": channel_metrics(fit_total, amplitude),
            "eval_micro": channel_metrics(eval_total, amplitude),
            "eval_system_macro_relative_mse_reduction": bootstrap_mean_interval(
                reductions,
                args.bootstrap_samples,
                args.bootstrap_seed + channel_index,
            ),
            "eval_system_macro_mean_cosine": bootstrap_mean_interval(
                cosines,
                args.bootstrap_samples,
                args.bootstrap_seed + 10 + channel_index,
            ),
            "eval_system_macro_positive_fraction": bootstrap_mean_interval(
                positive_fractions,
                args.bootstrap_samples,
                args.bootstrap_seed + 20 + channel_index,
            ),
            "eval_per_system": per_system,
        }

    rigid = result["channels"]["rigid"]
    macro = rigid["eval_system_macro_relative_mse_reduction"]
    cosine_macro = rigid["eval_system_macro_mean_cosine"]
    positive_macro = rigid["eval_system_macro_positive_fraction"]
    result["decision"] = {
        "force_signal_supported": bool(
            macro["ci_low"] > 0.0
            and macro["mean"] >= args.minimum_practical_relative_reduction
            and cosine_macro["ci_low"] > 0.0
            and positive_macro["mean"] > 0.5
        ),
        "next_step_if_supported": (
            "Evaluate inference-available bridge-state atom14 energy gradients before "
            "building a force-conditioned Path-4 model."
        ),
        "next_step_if_not_supported": (
            "Do not condition Path-4 on instantaneous atomistic force; move to a "
            "path-optimized/string pseudo-target or stochastic route model."
        ),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_path = args.output_dir / "force_residual_alignment.json"
    output_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
