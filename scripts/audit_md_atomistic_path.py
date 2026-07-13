#!/usr/bin/env python3
"""Audit atomistic geometry and severe clashes in a generated MD path."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pull-dir", type=Path, required=True)
    parser.add_argument("--preparation-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--severe-clash-distance-a", type=float, default=1.0)
    parser.add_argument("--max-severe-clashes-per-frame", type=int, default=0)
    parser.add_argument("--max-peptide-bond-a", type=float, default=1.70)
    parser.add_argument("--max-heavy-bond-relative-deviation", type=float, default=0.25)
    parser.add_argument("--min-ligand-protein-distance-a", type=float, default=1.0)
    return parser.parse_args()


def load_metric_rows(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open(newline="") as handle:
        for raw in csv.DictReader(handle):
            row: Dict[str, Any] = dict(raw)
            for key in (
                "step",
                "time_ps",
                "progress",
                "target_rmsd_nm",
                "cv_apo_rmsd_nm",
                "apo_ca_rmsd_angstrom",
                "holo_ca_rmsd_angstrom",
                "ligand_heavy_rmsd_angstrom",
                "temperature_k",
                "potential_kj_mol",
                "kinetic_kj_mol",
            ):
                row[key] = float(row[key])
            rows.append(row)
    return rows


def relative_bond_deviation(
    bond_lengths_angstrom: np.ndarray, reference_lengths_angstrom: np.ndarray
) -> float:
    if bond_lengths_angstrom.size == 0:
        return 0.0
    denominator = np.maximum(reference_lengths_angstrom[None, :], 1e-6)
    return float(np.max(np.abs(bond_lengths_angstrom - reference_lengths_angstrom) / denominator))


def summarize_path(
    *,
    severe_clash_counts: Sequence[int],
    peptide_lengths_angstrom: np.ndarray,
    heavy_bond_lengths_angstrom: np.ndarray,
    reference_heavy_bonds_angstrom: np.ndarray,
    ligand_protein_minima_angstrom: Sequence[float],
    temperatures_k: Sequence[float],
    potentials_kj_mol: Sequence[float],
    max_severe_clashes_per_frame: int,
    max_peptide_bond_a: float,
    max_heavy_bond_relative_deviation: float,
    min_ligand_protein_distance_a: float,
) -> Dict[str, Any]:
    max_clashes = int(max(severe_clash_counts, default=0))
    max_peptide = (
        float(np.max(peptide_lengths_angstrom)) if peptide_lengths_angstrom.size else None
    )
    bond_deviation = relative_bond_deviation(
        heavy_bond_lengths_angstrom, reference_heavy_bonds_angstrom
    )
    min_ligand_protein = float(min(ligand_protein_minima_angstrom))
    finite = all(
        math.isfinite(value)
        for values in (temperatures_k, potentials_kj_mol, ligand_protein_minima_angstrom)
        for value in values
    )
    checks = {
        "finite_trajectory": finite,
        "severe_clashes": max_clashes <= max_severe_clashes_per_frame,
        "peptide_geometry": max_peptide is not None and max_peptide <= max_peptide_bond_a,
        "heavy_bond_geometry": bond_deviation <= max_heavy_bond_relative_deviation,
        "ligand_protein_separation": min_ligand_protein >= min_ligand_protein_distance_a,
        "temperature_stable": min(temperatures_k) >= 250.0 and max(temperatures_k) <= 350.0,
    }
    passed = all(checks.values())
    return {
        "status": "atomistic_path_passed" if passed else "atomistic_path_failed",
        "checks": checks,
        "metrics": {
            "audited_frames": len(severe_clash_counts),
            "max_severe_clashes_per_frame": max_clashes,
            "max_peptide_bond_angstrom": max_peptide,
            "max_heavy_bond_relative_deviation": bond_deviation,
            "min_ligand_protein_distance_angstrom": min_ligand_protein,
            "temperature_range_k": [min(temperatures_k), max(temperatures_k)],
            "potential_energy_range_kj_mol": [
                min(potentials_kj_mol),
                max(potentials_kj_mol),
            ],
        },
        "passed": passed,
    }


def main() -> None:
    args = parse_args()
    import mdtraj as md
    from scipy.spatial import cKDTree

    dcd_path = args.pull_dir / "rmsd_pull.dcd"
    topology_path = args.pull_dir / "final_pulled.pdb"
    metrics_path = args.pull_dir / "rmsd_pull_metrics.csv"
    required = [dcd_path, topology_path, metrics_path, args.preparation_report]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Incomplete atomistic path inputs: {missing}")

    preparation = json.loads(args.preparation_report.read_text())
    protein_atoms = int(preparation["protein"]["prepared_protein_atoms"])
    solute_atoms = int(preparation["system"]["pre_solvent_atoms"])
    metrics = load_metric_rows(metrics_path)
    trajectory = md.load(str(dcd_path), top=str(topology_path))
    if trajectory.n_frames != len(metrics):
        raise ValueError(
            f"DCD/metrics frame mismatch: trajectory={trajectory.n_frames}, metrics={len(metrics)}"
        )
    path_indices = [
        index
        for index, row in enumerate(metrics)
        if row["stage"] in {"rmsd_pull", "endpoint_hold"}
    ]
    if len(path_indices) < 10:
        raise ValueError(f"Expected at least ten path frames, found {len(path_indices)}")
    path = trajectory.slice(path_indices, copy=True)
    path_metrics = [metrics[index] for index in path_indices]

    atoms = list(path.topology.atoms)
    heavy = [
        atom.index
        for atom in atoms[:solute_atoms]
        if atom.element is not None and atom.element.symbol != "H"
    ]
    protein_heavy = [index for index in heavy if index < protein_atoms]
    ligand_heavy = [index for index in heavy if protein_atoms <= index < solute_atoms]
    if not ligand_heavy:
        raise ValueError("No ligand heavy atoms found")
    residue_indices = np.asarray([atoms[index].residue.index for index in range(len(atoms))])
    bonded_pairs = {
        tuple(sorted((left.index, right.index))) for left, right in path.topology.bonds
    }

    heavy_bond_pairs: List[Tuple[int, int]] = []
    peptide_pairs: List[Tuple[int, int]] = []
    for left, right in path.topology.bonds:
        if left.index >= solute_atoms or right.index >= solute_atoms:
            continue
        if left.element is not None and right.element is not None:
            if left.element.symbol != "H" and right.element.symbol != "H":
                heavy_bond_pairs.append((left.index, right.index))
        names = {(left.name, right.name), (right.name, left.name)}
        if ("C", "N") in names and left.residue.index != right.residue.index:
            peptide_pairs.append((left.index, right.index))

    heavy_bond_lengths = (
        md.compute_distances(path, np.asarray(heavy_bond_pairs), periodic=True) * 10.0
        if heavy_bond_pairs
        else np.zeros((path.n_frames, 0), dtype=np.float64)
    )
    peptide_lengths = (
        md.compute_distances(path, np.asarray(peptide_pairs), periodic=True) * 10.0
        if peptide_pairs
        else np.zeros((path.n_frames, 0), dtype=np.float64)
    )
    reference_heavy_bonds = heavy_bond_lengths[0].copy()
    ligand_protein_pairs = np.asarray(
        [(ligand, protein) for ligand in ligand_heavy for protein in protein_heavy],
        dtype=np.int64,
    )
    ligand_protein_distances = (
        md.compute_distances(path, ligand_protein_pairs, periodic=True) * 10.0
    )

    severe_clash_counts: List[int] = []
    ligand_protein_minima = ligand_protein_distances.min(axis=1).astype(float).tolist()
    for frame_index, xyz_nm in enumerate(path.xyz):
        xyz_angstrom = xyz_nm * 10.0
        tree = cKDTree(xyz_angstrom[heavy])
        count = 0
        for local_left, local_right in tree.query_pairs(args.severe_clash_distance_a):
            left = heavy[local_left]
            right = heavy[local_right]
            if residue_indices[left] == residue_indices[right]:
                continue
            if tuple(sorted((left, right))) in bonded_pairs:
                continue
            if (left < protein_atoms <= right < solute_atoms) or (
                right < protein_atoms <= left < solute_atoms
            ):
                continue
            count += 1
        count += int(
            np.sum(ligand_protein_distances[frame_index] < args.severe_clash_distance_a)
        )
        severe_clash_counts.append(count)

    result = summarize_path(
        severe_clash_counts=severe_clash_counts,
        peptide_lengths_angstrom=peptide_lengths,
        heavy_bond_lengths_angstrom=heavy_bond_lengths,
        reference_heavy_bonds_angstrom=reference_heavy_bonds,
        ligand_protein_minima_angstrom=ligand_protein_minima,
        temperatures_k=[float(row["temperature_k"]) for row in path_metrics],
        potentials_kj_mol=[float(row["potential_kj_mol"]) for row in path_metrics],
        max_severe_clashes_per_frame=args.max_severe_clashes_per_frame,
        max_peptide_bond_a=args.max_peptide_bond_a,
        max_heavy_bond_relative_deviation=args.max_heavy_bond_relative_deviation,
        min_ligand_protein_distance_a=args.min_ligand_protein_distance_a,
    )
    result["pull_dir"] = str(args.pull_dir)
    result["usage"] = {
        "geometry_supervision_candidate": bool(result["passed"]),
        "phase_supervision": False,
        "heldout_benchmark": False,
        "kinetics_claims": False,
        "blocking_reason": "Per-residue phase/event-order and manifold-projection audits remain.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)
    if not result["passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
