#!/usr/bin/env python3
"""Summarize diagnostic Path-4 all-atom frame-reference rebuilds."""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Sequence


SCHEMA_VERSION = "bindrae_path4_frame_reference_audit_summary_v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reports-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--scientific-frame-threshold-kj-mol-nm", type=float, default=1.0e6
    )
    return parser.parse_args()


def _pearson(left: Sequence[float], right: Sequence[float]) -> float | None:
    if len(left) != len(right) or len(left) < 3:
        return None
    left_mean = sum(left) / len(left)
    right_mean = sum(right) / len(right)
    left_centered = [value - left_mean for value in left]
    right_centered = [value - right_mean for value in right]
    left_norm = math.sqrt(sum(value * value for value in left_centered))
    right_norm = math.sqrt(sum(value * value for value in right_centered))
    if left_norm == 0.0 or right_norm == 0.0:
        return None
    return sum(
        left_value * right_value
        for left_value, right_value in zip(left_centered, right_centered)
    ) / (left_norm * right_norm)


def _finite_pairs(
    records: Iterable[Mapping[str, Any]], left_key: str, right_key: str
) -> tuple[list[float], list[float]]:
    left: list[float] = []
    right: list[float] = []
    for record in records:
        try:
            left_value = float(record[left_key])
            right_value = float(record[right_key])
        except (KeyError, TypeError, ValueError):
            continue
        if math.isfinite(left_value) and math.isfinite(right_value):
            left.append(left_value)
            right.append(right_value)
    return left, right


def summarize_report(
    report: Mapping[str, Any], report_path: Path, *, scientific_threshold: float
) -> Dict[str, Any]:
    frame_preflight = [
        dict(record)
        for record in report.get("frame_preflight", [])
        if isinstance(record, Mapping)
    ]
    relaxation = report.get("relaxation")
    relaxation_frames = (
        relaxation.get("frame_diagnostics", [])
        if isinstance(relaxation, Mapping)
        else []
    )
    relaxation_by_frame = {
        int(record["frame_index"]): dict(record)
        for record in relaxation_frames
        if isinstance(record, Mapping) and "frame_index" in record
    }
    joined_frames = []
    for frame in frame_preflight:
        if "frame_index" not in frame:
            continue
        frame_index = int(frame["frame_index"])
        joined_frames.append({**relaxation_by_frame.get(frame_index, {}), **frame})

    force_frames = [
        frame
        for frame in joined_frames
        if isinstance(frame.get("atomic_force_max_kj_mol_nm"), (int, float))
        and math.isfinite(float(frame["atomic_force_max_kj_mol_nm"]))
    ]
    above = [
        frame
        for frame in force_frames
        if float(frame["atomic_force_max_kj_mol_nm"]) > scientific_threshold
    ]
    maximum = (
        max(force_frames, key=lambda frame: float(frame["atomic_force_max_kj_mol_nm"]))
        if force_frames
        else None
    )
    maximum_atom_diagnostics = (
        maximum.get("atom_force_diagnostics")
        if isinstance(maximum, Mapping)
        else None
    )
    maximum_top_atoms = (
        maximum_atom_diagnostics.get("top_force_atoms", [])
        if isinstance(maximum_atom_diagnostics, Mapping)
        else []
    )
    maximum_force_components = (
        maximum_atom_diagnostics.get("force_components", {})
        if isinstance(maximum_atom_diagnostics, Mapping)
        else {}
    )
    dominant_component = None
    if isinstance(maximum_force_components, Mapping) and maximum_force_components:
        dominant_label, dominant_record = max(
            maximum_force_components.items(),
            key=lambda item: float(
                item[1].get("force_on_total_max_atom_kj_mol_nm", 0.0)
            ),
        )
        dominant_component = {
            "label": dominant_label,
            **dict(dominant_record),
        }
    first_in_traversal = (
        min(above, key=lambda frame: int(frame.get("traversal_order", 10**9)))
        if above
        else None
    )

    correlations: Dict[str, float | None] = {}
    log_force_records = []
    for frame in force_frames:
        force = float(frame["atomic_force_max_kj_mol_nm"])
        if force > 0.0:
            log_force_records.append({**frame, "log10_atomic_force": math.log10(force)})
    for metric in (
        "traversal_order",
        "mapped_path_step_rms_angstrom",
        "hidden_atom_injection_rms_angstrom",
        "hidden_atom_relaxation_rms_angstrom",
        "hidden_protein_atom_injection_rms_angstrom",
        "hidden_protein_atom_relaxation_rms_angstrom",
        "environment_atom_relaxation_rms_angstrom",
        "mapped_heavy_rms_after_relaxation_angstrom",
    ):
        force_values, metric_values = _finite_pairs(
            log_force_records, "log10_atomic_force", metric
        )
        correlations[f"log10_force_vs_{metric}"] = _pearson(
            force_values, metric_values
        )

    prepared = report.get("prepared_topology_preflight")
    prepared_force = (
        prepared.get("atomic_force_max_kj_mol_nm")
        if isinstance(prepared, Mapping)
        else None
    )
    generation = report.get("generation_contract")
    iterations = (
        generation.get("reference_relaxation_iterations")
        if isinstance(generation, Mapping)
        else None
    )
    return {
        "sample_id": report.get("sample_id"),
        "report": str(report_path),
        "status": report.get("status"),
        "failure_stage": report.get("failure_stage"),
        "rejection_type": report.get("rejection_type"),
        "reference_relaxation_iterations": iterations,
        "prepared_atomic_force_max_kj_mol_nm": prepared_force,
        "frames_observed": len(force_frames),
        "frames_above_frozen_threshold": len(above),
        "frame_indices_above_frozen_threshold": [
            int(frame["frame_index"]) for frame in above
        ],
        "maximum_frame": (
            {
                "frame_index": int(maximum["frame_index"]),
                "time": maximum.get("time"),
                "atomic_force_max_kj_mol_nm": float(
                    maximum["atomic_force_max_kj_mol_nm"]
                ),
            }
            if maximum is not None
            else None
        ),
        "maximum_force_atom": (
            dict(maximum_top_atoms[0]) if maximum_top_atoms else None
        ),
        "dominant_force_component_on_max_atom": dominant_component,
        "first_above_threshold_in_holo_to_apo_traversal": (
            {
                "frame_index": int(first_in_traversal["frame_index"]),
                "time": first_in_traversal.get("time"),
                "traversal_order": first_in_traversal.get("traversal_order"),
                "atomic_force_max_kj_mol_nm": float(
                    first_in_traversal["atomic_force_max_kj_mol_nm"]
                ),
            }
            if first_in_traversal is not None
            else None
        ),
        "correlations": correlations,
        "frames": joined_frames,
    }


def summarize_reports(
    report_paths: Sequence[Path], *, scientific_threshold: float
) -> Dict[str, Any]:
    if not math.isfinite(scientific_threshold) or scientific_threshold <= 0.0:
        raise ValueError("scientific_threshold must be positive and finite")
    rows = []
    for path in sorted(report_paths):
        report = json.loads(path.read_text())
        if not isinstance(report, Mapping):
            raise ValueError(f"Frame-reference report is not an object: {path}")
        rows.append(
            summarize_report(
                report, path, scientific_threshold=scientific_threshold
            )
        )
    status_counts = Counter(str(row["status"]) for row in rows)
    rows_by_sample: Dict[str, list[Dict[str, Any]]] = {}
    for row in rows:
        rows_by_sample.setdefault(str(row.get("sample_id")), []).append(row)
    iteration_comparisons = []
    for sample_id, sample_rows in sorted(rows_by_sample.items()):
        comparable = [
            row
            for row in sample_rows
            if row.get("reference_relaxation_iterations") is not None
        ]
        comparable.sort(key=lambda row: int(row["reference_relaxation_iterations"]))
        if len(comparable) < 2:
            continue
        baseline = comparable[0]
        baseline_forces = {
            int(frame["frame_index"]): float(frame["atomic_force_max_kj_mol_nm"])
            for frame in baseline["frames"]
            if "atomic_force_max_kj_mol_nm" in frame
        }
        baseline_above = {
            index
            for index, force in baseline_forces.items()
            if force > scientific_threshold
        }
        for candidate in comparable[1:]:
            candidate_forces = {
                int(frame["frame_index"]): float(
                    frame["atomic_force_max_kj_mol_nm"]
                )
                for frame in candidate["frames"]
                if "atomic_force_max_kj_mol_nm" in frame
            }
            candidate_above = {
                index
                for index, force in candidate_forces.items()
                if force > scientific_threshold
            }
            shared_problem_frames = sorted(baseline_above & set(candidate_forces))
            iteration_comparisons.append(
                {
                    "sample_id": sample_id,
                    "baseline_iterations": int(
                        baseline["reference_relaxation_iterations"]
                    ),
                    "candidate_iterations": int(
                        candidate["reference_relaxation_iterations"]
                    ),
                    "baseline_frames_above_threshold": len(baseline_above),
                    "candidate_frames_above_threshold": len(candidate_above),
                    "resolved_frame_indices": sorted(baseline_above - candidate_above),
                    "newly_above_frame_indices": sorted(candidate_above - baseline_above),
                    "shared_problem_frame_force_ratios": {
                        str(index): float(
                            candidate_forces[index] / baseline_forces[index]
                        )
                        for index in shared_problem_frames
                        if baseline_forces[index] > 0.0
                    },
                    "baseline_maximum_force_kj_mol_nm": (
                        baseline["maximum_frame"]["atomic_force_max_kj_mol_nm"]
                        if baseline["maximum_frame"] is not None
                        else None
                    ),
                    "candidate_maximum_force_kj_mol_nm": (
                        candidate["maximum_frame"]["atomic_force_max_kj_mol_nm"]
                        if candidate["maximum_frame"] is not None
                        else None
                    ),
                }
            )
    return {
        "schema_version": SCHEMA_VERSION,
        "scientific_frame_threshold_kj_mol_nm": float(scientific_threshold),
        "reports": len(rows),
        "status_counts": dict(sorted(status_counts.items())),
        "systems_with_frames_above_frozen_threshold": sum(
            int(row["frames_above_frozen_threshold"] > 0) for row in rows
        ),
        "iteration_comparisons": iteration_comparisons,
        "rows": rows,
    }


def main() -> None:
    args = parse_args()
    reports = sorted(args.reports_root.rglob("report.json"))
    if not reports:
        raise FileNotFoundError(
            f"No frame-reference report.json files under {args.reports_root}"
        )
    summary = summarize_reports(
        reports,
        scientific_threshold=float(
            args.scientific_frame_threshold_kj_mol_nm
        ),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    printable = {key: value for key, value in summary.items() if key != "rows"}
    print(json.dumps(printable, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
