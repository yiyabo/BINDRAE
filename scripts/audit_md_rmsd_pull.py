#!/usr/bin/env python3
"""Audit global-RMSD pulling output before any supervision is enabled."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pull-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-ligand-rmsd-a", type=float, default=5.0)
    parser.add_argument("--min-progress-correlation-magnitude", type=float, default=0.70)
    parser.add_argument("--min-endpoint-hold-occupancy", type=float, default=0.60)
    return parser.parse_args()


def load_metrics(path: Path) -> List[Dict[str, Any]]:
    numeric = {
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
    }
    rows: List[Dict[str, Any]] = []
    with path.open(newline="") as handle:
        for raw in csv.DictReader(handle):
            row: Dict[str, Any] = dict(raw)
            for key in numeric:
                row[key] = float(row[key])
            rows.append(row)
    if len(rows) < 3:
        raise ValueError(f"Pull metrics require at least three rows, found {len(rows)}")
    return rows


def rank_values(values: Iterable[float]) -> np.ndarray:
    array = np.asarray(list(values), dtype=np.float64)
    order = np.argsort(array, kind="mergesort")
    ranks = np.empty(len(array), dtype=np.float64)
    ranks[order] = np.arange(len(array), dtype=np.float64)
    return ranks


def spearman_correlation(left: Iterable[float], right: Iterable[float]) -> float:
    left_rank = rank_values(left)
    right_rank = rank_values(right)
    if len(left_rank) < 2 or left_rank.std() == 0.0 or right_rank.std() == 0.0:
        return 0.0
    return float(np.corrcoef(left_rank, right_rank)[0, 1])


def build_audit(
    report: Mapping[str, Any],
    rows: List[Mapping[str, Any]],
    *,
    max_ligand_rmsd_a: float,
    min_progress_correlation_magnitude: float,
    min_endpoint_hold_occupancy: float,
) -> Dict[str, Any]:
    pull_rows = [row for row in rows if row["stage"] == "rmsd_pull"]
    hold_rows = [row for row in rows if row["stage"] == "endpoint_hold"]
    if len(pull_rows) < 10 or len(hold_rows) < 3:
        raise ValueError(
            f"Insufficient pull/hold frames: pull={len(pull_rows)}, hold={len(hold_rows)}"
        )
    correlation = spearman_correlation(
        (float(row["progress"]) for row in pull_rows),
        (float(row["apo_ca_rmsd_angstrom"]) for row in pull_rows),
    )
    all_path_rows = pull_rows + hold_rows
    crossing_indices = [
        index
        for index, row in enumerate(all_path_rows)
        if float(row["apo_ca_rmsd_angstrom"])
        < float(row["holo_ca_rmsd_angstrom"])
    ]
    hold_occupancy = float(
        np.mean(
            [
                float(row["apo_ca_rmsd_angstrom"])
                < float(row["holo_ca_rmsd_angstrom"])
                for row in hold_rows
            ]
        )
    )
    temperatures = [float(row["temperature_k"]) for row in all_path_rows]
    ligand_rmsds = [float(row["ligand_heavy_rmsd_angstrom"]) for row in all_path_rows]
    energies = [
        float(row["potential_kj_mol"]) + float(row["kinetic_kj_mol"])
        for row in all_path_rows
    ]
    checks = {
        "pull_report_passed": report.get("status") == "rmsd_pull_smoke_passed",
        "enough_frames": len(pull_rows) >= 10 and len(hold_rows) >= 3,
        "progress_correlation": correlation <= -min_progress_correlation_magnitude,
        "crossed_endpoint_bisector": bool(crossing_indices),
        "endpoint_hold_occupancy": hold_occupancy >= min_endpoint_hold_occupancy,
        "ligand_stable": max(ligand_rmsds) <= max_ligand_rmsd_a,
        "temperature_stable": min(temperatures) >= 250.0 and max(temperatures) <= 350.0,
        "finite_energy": all(math.isfinite(value) for value in energies),
    }
    geometry_candidate = all(checks.values())
    return {
        "status": "path_metrics_passed" if geometry_candidate else "path_metrics_failed",
        "passed": geometry_candidate,
        "transition_id": report.get("transition_id"),
        "evidence_tier": "silver_enhanced_sampling",
        "biased_sampling": True,
        "metrics": {
            "pull_frames": len(pull_rows),
            "hold_frames": len(hold_rows),
            "spearman_progress_vs_apo_rmsd": correlation,
            "first_crossing_path_index": crossing_indices[0] if crossing_indices else None,
            "endpoint_hold_apo_basin_occupancy": hold_occupancy,
            "max_ligand_heavy_rmsd_angstrom": max(ligand_rmsds),
            "temperature_range_k": [min(temperatures), max(temperatures)],
        },
        "checks": checks,
        "usage": {
            "geometry_supervision_candidate": geometry_candidate,
            "phase_supervision": False,
            "heldout_benchmark": False,
            "kinetics_claims": False,
            "blocking_reason": (
                "Per-residue event-order, manifold projection, and independent physical-validity "
                "audits are still required before phase supervision."
            ),
        },
    }


def main() -> None:
    args = parse_args()
    report_path = args.pull_dir / "rmsd_pull_report.json"
    metrics_path = args.pull_dir / "rmsd_pull_metrics.csv"
    if not report_path.is_file() or not metrics_path.is_file():
        raise FileNotFoundError(f"Incomplete RMSD-pull output: {args.pull_dir}")
    report = json.loads(report_path.read_text())
    rows = load_metrics(metrics_path)
    audit = build_audit(
        report,
        rows,
        max_ligand_rmsd_a=args.max_ligand_rmsd_a,
        min_progress_correlation_magnitude=args.min_progress_correlation_magnitude,
        min_endpoint_hold_occupancy=args.min_endpoint_hold_occupancy,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n")
    print(json.dumps(audit, indent=2, sort_keys=True), flush=True)
    if audit["status"] != "path_metrics_passed":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
