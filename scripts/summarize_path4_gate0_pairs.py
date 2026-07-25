#!/usr/bin/env python3
"""Summarize paired Path-3 and Path-4 OpenMM Gate-0 score reports."""

from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Sequence, Tuple

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.openmm_gate0 import (
    PATH4_GATE0_PAIRED_SUMMARY_SCHEMA_VERSION,
    summarize_energy_profile,
    validate_openmm_gate0_score_report,
)


REPORT_METRICS: Mapping[str, Tuple[str, ...]] = {
    "raw_excess_p95_kj_mol": (
        "raw_energy_profile",
        "interior_positive_excess_p95_kj_mol",
    ),
    "raw_excess_max_kj_mol": (
        "raw_energy_profile",
        "interior_positive_excess_max_kj_mol",
    ),
    "relaxed_invalid_frame_fraction": (
        "relaxed_path",
        "invalid_frame_fraction",
    ),
    "relaxed_invalid_or_severe_clash_frame_fraction": (
        "relaxed_path",
        "invalid_or_severe_clash_frame_fraction",
    ),
    "severe_clash_frame_fraction": (
        "relaxed_path",
        "severe_clash_frame_fraction",
    ),
    "severe_clash_pairs_max": ("relaxed_path", "severe_clash_pairs_max"),
    "restraint_target_rms_p95_angstrom": (
        "relaxed_path",
        "restraint_target_rms_p95_angstrom",
    ),
    "protein_residue_force_p95_kj_mol_nm": (
        "relaxed_path",
        "protein_residue_net_force_p95_over_frames_kj_mol_nm",
    ),
}
PAIRED_RELAXED_METRICS: Mapping[str, Tuple[str, ...]] = {
    "relaxed_excess_p95_kj_mol": (
        "interior_positive_excess_p95_kj_mol",
    ),
    "relaxed_excess_max_kj_mol": (
        "interior_positive_excess_max_kj_mol",
    ),
}
METRICS: Mapping[str, Tuple[str, ...]] = {
    **REPORT_METRICS,
    **PAIRED_RELAXED_METRICS,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--path3-glob", action="append", required=True)
    parser.add_argument("--candidate-glob", action="append", required=True)
    parser.add_argument("--optimizer-glob", action="append", default=[])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap-resamples", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20260721)
    parser.add_argument(
        "--maximum-endpoint-energy-difference-kj-mol", type=float, default=1.0
    )
    return parser.parse_args()


def _load_json(path: Path) -> Dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"Expected one JSON object in {path}")
    return value


def _discover(patterns: Iterable[str]) -> Sequence[Path]:
    return sorted(
        {
            Path(match).resolve()
            for pattern in patterns
            for match in glob.glob(pattern, recursive=True)
        }
    )


def _index_reports(
    paths: Sequence[Path], label: str, *, require_completed: bool = True
) -> Dict[str, Dict[str, Any]]:
    reports: Dict[str, Dict[str, Any]] = {}
    for path in paths:
        report = _load_json(path)
        if require_completed and report.get("status") != "completed":
            raise ValueError(f"{label} report is not completed: {path}")
        sample_id = str(report.get("sample_id") or "")
        if not sample_id:
            raise ValueError(f"{label} report misses sample_id: {path}")
        if sample_id in reports:
            raise ValueError(f"Duplicate {label} report for {sample_id}")
        report["_report_path"] = str(path)
        reports[sample_id] = report
    return reports


def _nested_float(report: Mapping[str, Any], path: Sequence[str]) -> float:
    value: Any = report
    for key in path:
        if not isinstance(value, Mapping) or key not in value:
            raise ValueError(f"Score report misses metric path {'/'.join(path)}")
        value = value[key]
    result = float(value)
    if not np.isfinite(result):
        raise ValueError(f"Score report has non-finite metric {'/'.join(path)}")
    return result


def _optional_nested_float(
    report: Mapping[str, Any], path: Sequence[str]
) -> float | None:
    value: Any = report
    for key in path:
        if not isinstance(value, Mapping) or key not in value:
            raise ValueError(f"Score report misses metric path {'/'.join(path)}")
        value = value[key]
    if value is None:
        return None
    result = float(value)
    if not np.isfinite(result):
        raise ValueError(f"Score report has non-finite metric {'/'.join(path)}")
    return result


def _bootstrap_mean_ci(
    values: np.ndarray, resamples: int, rng: np.random.Generator
) -> Tuple[float, float]:
    if values.ndim != 1 or values.size == 0:
        raise ValueError("Paired bootstrap requires a non-empty vector")
    if resamples <= 0:
        raise ValueError("bootstrap_resamples must be positive")
    indices = rng.integers(0, values.size, size=(resamples, values.size))
    means = values[indices].mean(axis=1)
    low, high = np.quantile(means, [0.025, 0.975])
    return float(low), float(high)


def _validate_pair_contract(
    sample_id: str,
    path3_report: Mapping[str, Any],
    candidate_report: Mapping[str, Any],
    *,
    maximum_endpoint_energy_difference_kj_mol: float,
) -> Dict[str, float]:
    if maximum_endpoint_energy_difference_kj_mol <= 0.0:
        raise ValueError("maximum endpoint energy difference must be positive")
    for field in ("schema_version", "implicit_system_contract", "contract"):
        if path3_report.get(field) != candidate_report.get(field):
            raise ValueError(f"{sample_id} paired scorer contract mismatch: {field}")

    differences: Dict[str, float] = {}
    for profile in ("raw_energy_profile", "relaxed_energy_profile"):
        for endpoint in (
            "endpoint_apo_energy_kj_mol",
            "endpoint_holo_energy_kj_mol",
        ):
            metric_path = (profile, endpoint)
            path3_value = _nested_float(path3_report, metric_path)
            candidate_value = _nested_float(candidate_report, metric_path)
            difference = abs(path3_value - candidate_value)
            key = f"{profile}/{endpoint}"
            differences[key] = float(difference)
            if difference > float(maximum_endpoint_energy_difference_kj_mol):
                raise ValueError(
                    f"{sample_id} paired endpoint energy mismatch for {key}: "
                    f"{difference:.6g} > "
                    f"{float(maximum_endpoint_energy_difference_kj_mol):.6g} kJ/mol"
                )
    return differences


def _paired_relaxed_profiles(
    sample_id: str,
    path3_report: Mapping[str, Any],
    candidate_report: Mapping[str, Any],
) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
    path3_frames = path3_report["frames"]
    candidate_frames = candidate_report["frames"]
    if len(path3_frames) != len(candidate_frames):
        raise ValueError(f"{sample_id} paired frame-count mismatch")

    times: list[float] = []
    for frame_index, (path3_frame, candidate_frame) in enumerate(
        zip(path3_frames, candidate_frames)
    ):
        if int(path3_frame["frame_index"]) != int(candidate_frame["frame_index"]):
            raise ValueError(f"{sample_id} paired frame-index mismatch")
        path3_time = float(path3_frame["time"])
        candidate_time = float(candidate_frame["time"])
        if not np.isclose(path3_time, candidate_time, atol=1e-8, rtol=0.0):
            raise ValueError(
                f"{sample_id} paired time-grid mismatch at frame {frame_index}"
            )
        times.append(path3_time)

    endpoint_indices = (0, len(path3_frames) - 1)
    endpoints_valid = all(
        bool(path3_frames[index]["relaxed_valid"])
        and bool(candidate_frames[index]["relaxed_valid"])
        for index in endpoint_indices
    )
    paired_valid = np.asarray(
        [
            bool(path3_frame["relaxed_valid"])
            and bool(candidate_frame["relaxed_valid"])
            for path3_frame, candidate_frame in zip(
                path3_frames, candidate_frames
            )
        ],
        dtype=np.bool_,
    )
    if not endpoints_valid:
        paired_valid[1:-1] = False

    path3_profile = summarize_energy_profile(
        times,
        [float(frame["relaxed_potential_kj_mol"]) for frame in path3_frames],
        interior_valid_mask=paired_valid,
    )
    candidate_profile = summarize_energy_profile(
        times,
        [float(frame["relaxed_potential_kj_mol"]) for frame in candidate_frames],
        interior_valid_mask=paired_valid,
    )
    path3_invalid = np.asarray(
        [not bool(frame["relaxed_valid"]) for frame in path3_frames[1:-1]],
        dtype=np.bool_,
    )
    candidate_invalid = np.asarray(
        [not bool(frame["relaxed_valid"]) for frame in candidate_frames[1:-1]],
        dtype=np.bool_,
    )
    paired_interior_valid = paired_valid[1:-1]
    contract = {
        "endpoint_frames_valid": bool(endpoints_valid),
        "interior_frames": int(paired_interior_valid.size),
        "paired_valid_interior_frames": int(paired_interior_valid.sum()),
        "paired_invalid_interior_frames": int((~paired_interior_valid).sum()),
        "path3_invalid_interior_frames": int(path3_invalid.sum()),
        "candidate_invalid_interior_frames": int(candidate_invalid.sum()),
        "paired_valid_times": [
            float(time)
            for time, valid in zip(times[1:-1], paired_interior_valid)
            if bool(valid)
        ],
    }
    return path3_profile, candidate_profile, contract


def summarize_pairs(
    path3_reports: Mapping[str, Mapping[str, Any]],
    candidate_reports: Mapping[str, Mapping[str, Any]],
    optimizer_reports: Mapping[str, Mapping[str, Any]],
    *,
    bootstrap_resamples: int,
    seed: int,
    maximum_endpoint_energy_difference_kj_mol: float = 1.0,
) -> Dict[str, Any]:
    paired_ids = sorted(set(path3_reports) & set(candidate_reports))
    if not paired_ids:
        raise ValueError("No Path-3/Path-4 score pairs were found")
    rng = np.random.default_rng(int(seed))
    collected = {
        name: {
            "path3": [],
            "candidate": [],
            "improvement": [],
            "relative": [],
            "unavailable": [],
        }
        for name in METRICS
    }
    rows = []
    for sample_id in paired_ids:
        base_report = path3_reports[sample_id]
        candidate_report = candidate_reports[sample_id]
        validate_openmm_gate0_score_report(
            base_report,
            label=f"{sample_id} Path-3 score report",
            require_reference_cache=True,
        )
        validate_openmm_gate0_score_report(
            candidate_report,
            label=f"{sample_id} Path-4 score report",
            require_reference_cache=True,
        )
        path3_relaxed, candidate_relaxed, relaxed_pairing = (
            _paired_relaxed_profiles(sample_id, base_report, candidate_report)
        )
        row: Dict[str, Any] = {
            "sample_id": sample_id,
            "path3_report": base_report.get("_report_path"),
            "candidate_report": candidate_report.get("_report_path"),
            "endpoint_energy_abs_differences_kj_mol": _validate_pair_contract(
                sample_id,
                base_report,
                candidate_report,
                maximum_endpoint_energy_difference_kj_mol=float(
                    maximum_endpoint_energy_difference_kj_mol
                ),
            ),
            "relaxed_frame_pairing": relaxed_pairing,
            "metrics": {},
        }
        for name, metric_path in METRICS.items():
            if name in PAIRED_RELAXED_METRICS:
                base = _optional_nested_float(path3_relaxed, metric_path)
                candidate = _optional_nested_float(candidate_relaxed, metric_path)
            else:
                base = _optional_nested_float(base_report, metric_path)
                candidate = _optional_nested_float(candidate_report, metric_path)
            if base is None or candidate is None:
                row["metrics"][name] = {
                    "available": False,
                    "path3": base,
                    "candidate": candidate,
                    "improvement": None,
                    "relative_improvement": None,
                }
                collected[name]["unavailable"].append(sample_id)
                continue
            improvement = base - candidate
            relative = improvement / base if base > 1e-12 else None
            row["metrics"][name] = {
                "available": True,
                "path3": base,
                "candidate": candidate,
                "improvement": improvement,
                "relative_improvement": relative,
            }
            collected[name]["path3"].append(base)
            collected[name]["candidate"].append(candidate)
            collected[name]["improvement"].append(improvement)
            if relative is not None:
                collected[name]["relative"].append(relative)
        optimizer = optimizer_reports.get(sample_id)
        if optimizer is not None:
            correction = optimizer.get("correction") or {}
            row["optimizer"] = {
                "report": optimizer.get("_report_path"),
                "selected_improvement": bool(optimizer.get("selected_improvement")),
                "fallback_to_path3": bool(optimizer.get("fallback_to_path3")),
                "energy_force_calls": int(optimizer.get("energy_force_calls", 0)),
                "wall_seconds": float(optimizer.get("wall_seconds", 0.0)),
                "correction_rms_angstrom": float(correction.get("rms_angstrom", 0.0)),
                "correction_max_angstrom": float(correction.get("max_angstrom", 0.0)),
                "normal_parallel_cos_abs_max": float(
                    correction.get("normal_parallel_cos_abs_max", 0.0)
                ),
                "endpoint_max_error_angstrom": float(
                    correction.get("endpoint_max_error_angstrom", 0.0)
                ),
            }
        rows.append(row)

    summaries = {}
    for name, values in collected.items():
        base = np.asarray(values["path3"], dtype=np.float64)
        candidate = np.asarray(values["candidate"], dtype=np.float64)
        improvement = np.asarray(values["improvement"], dtype=np.float64)
        relative = np.asarray(values["relative"], dtype=np.float64)
        if improvement.size == 0:
            summaries[name] = {
                "paired_systems_with_metric": 0,
                "unavailable_systems": list(values["unavailable"]),
                "path3_mean": None,
                "candidate_mean": None,
                "mean_improvement": None,
                "median_improvement": None,
                "mean_improvement_ci95": None,
                "improvement_fraction": None,
                "mean_relative_improvement": None,
                "relative_improvement_at_least_10pct_fraction": None,
            }
            continue
        ci_low, ci_high = _bootstrap_mean_ci(
            improvement, int(bootstrap_resamples), rng
        )
        summaries[name] = {
            "paired_systems_with_metric": int(improvement.size),
            "unavailable_systems": list(values["unavailable"]),
            "path3_mean": float(base.mean()),
            "candidate_mean": float(candidate.mean()),
            "mean_improvement": float(improvement.mean()),
            "median_improvement": float(np.median(improvement)),
            "mean_improvement_ci95": [ci_low, ci_high],
            "improvement_fraction": float(np.mean(improvement > 0.0)),
            "mean_relative_improvement": (
                float(relative.mean()) if relative.size else None
            ),
            "relative_improvement_at_least_10pct_fraction": (
                float(np.mean(relative >= 0.10)) if relative.size else None
            ),
        }

    optimizer_rows = [row["optimizer"] for row in rows if "optimizer" in row]
    return {
        "schema_version": PATH4_GATE0_PAIRED_SUMMARY_SCHEMA_VERSION,
        "counts": {
            "path3_reports": len(path3_reports),
            "candidate_reports": len(candidate_reports),
            "paired_systems": len(paired_ids),
            "optimizer_reports": len(optimizer_reports),
            "path3_only": sorted(set(path3_reports) - set(candidate_reports)),
            "candidate_only": sorted(set(candidate_reports) - set(path3_reports)),
        },
        "metrics": summaries,
        "optimizer": {
            "paired_reports": len(optimizer_rows),
            "selected_improvement_fraction": (
                float(np.mean([row["selected_improvement"] for row in optimizer_rows]))
                if optimizer_rows
                else None
            ),
            "fallback_fraction": (
                float(np.mean([row["fallback_to_path3"] for row in optimizer_rows]))
                if optimizer_rows
                else None
            ),
            "total_energy_force_calls": int(
                sum(row["energy_force_calls"] for row in optimizer_rows)
            ),
            "total_wall_seconds": float(
                sum(row["wall_seconds"] for row in optimizer_rows)
            ),
        },
        "bootstrap": {
            "resamples": int(bootstrap_resamples),
            "seed": int(seed),
            "unit": "system",
            "paired": True,
        },
        "pair_contract": {
            "required_equal_fields": [
                "schema_version",
                "implicit_system_contract",
                "contract",
            ],
            "maximum_endpoint_energy_difference_kj_mol": float(
                maximum_endpoint_energy_difference_kj_mol
            ),
            "relaxed_energy_frame_policy": "paired_valid_intersection",
            "relaxed_invalid_frames_are_excluded_from_energy_profiles": True,
        },
        "systems": rows,
    }


def main() -> None:
    args = parse_args()
    path3 = _index_reports(_discover(args.path3_glob), "Path-3")
    candidate = _index_reports(_discover(args.candidate_glob), "Path-4")
    optimizer = _index_reports(_discover(args.optimizer_glob), "optimizer")
    summary = summarize_pairs(
        path3,
        candidate,
        optimizer,
        bootstrap_resamples=int(args.bootstrap_resamples),
        seed=int(args.seed),
        maximum_endpoint_energy_difference_kj_mol=float(
            args.maximum_endpoint_energy_difference_kj_mol
        ),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    printable = dict(summary)
    printable.pop("systems", None)
    print(json.dumps(printable, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
