#!/usr/bin/env python3
"""Paired system-level comparison for Stage-2 transition evaluations."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Dict, Mapping, Sequence, Tuple

import numpy as np


METRIC_DIRECTIONS = {
    "all_pocket/ligand_clash_severity": "lower",
    "all_pocket/ligand_clash_proxy": "lower",
    "all_pocket/path_mae_dist": "lower",
    "all_pocket/path_mae_dist_interior": "lower",
    "all_pocket/path_length_total": "lower",
    "all_pocket/endpoint_abs_dist": "lower",
    "all_pocket/direction_acc": "higher",
}

DEFAULT_METRICS = (
    "all_pocket/ligand_clash_severity",
    "all_pocket/ligand_clash_proxy",
    "all_pocket/path_mae_dist",
    "all_pocket/path_length_total",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--baseline-name", default="baseline")
    parser.add_argument("--candidate-name", default="candidate")
    parser.add_argument("--metrics", default=",".join(DEFAULT_METRICS))
    parser.add_argument("--bootstrap-samples", type=int, default=10000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260719)
    return parser.parse_args()


def load_records(path: Path) -> Sequence[Mapping[str, object]]:
    payload = json.loads(path.read_text())
    records = payload.get("per_sample_metrics")
    if not isinstance(records, list) or not records:
        raise ValueError(
            f"No per_sample_metrics found in {path}; rerun with "
            "--include_per_sample_metrics"
        )
    return records


def finite_metric(record: Mapping[str, object], metric: str) -> float | None:
    metrics = record.get("metrics")
    if not isinstance(metrics, Mapping):
        return None
    value = metrics.get(metric)
    if not isinstance(value, (int, float)):
        return None
    value = float(value)
    return value if math.isfinite(value) else None


def aligned_metric_pairs(
    baseline_records: Sequence[Mapping[str, object]],
    candidate_records: Sequence[Mapping[str, object]],
    metric: str,
) -> Sequence[Tuple[str, float, float]]:
    baseline = {str(record["sample_id"]): record for record in baseline_records}
    candidate = {str(record["sample_id"]): record for record in candidate_records}
    if len(baseline) != len(baseline_records) or len(candidate) != len(candidate_records):
        raise ValueError("Duplicate sample_id records")
    if baseline.keys() != candidate.keys():
        raise ValueError("Transition evaluation sample sets are not aligned")
    pairs = []
    for sample_id in sorted(baseline):
        baseline_value = finite_metric(baseline[sample_id], metric)
        candidate_value = finite_metric(candidate[sample_id], metric)
        if baseline_value is not None and candidate_value is not None:
            pairs.append((sample_id, baseline_value, candidate_value))
    return pairs


def bootstrap_ci(values: np.ndarray, samples: int, seed: int) -> Tuple[float, float]:
    if samples <= 0:
        raise ValueError("bootstrap_samples must be positive")
    if values.size == 1:
        return float(values[0]), float(values[0])
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, values.size, size=(samples, values.size))
    means = values[indices].mean(axis=1)
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def compare_metric(
    pairs: Sequence[Tuple[str, float, float]],
    direction: str,
    *,
    bootstrap_samples: int,
    bootstrap_seed: int,
) -> Dict[str, object]:
    if not pairs:
        raise ValueError("No finite aligned systems for metric")
    baseline = np.asarray([item[1] for item in pairs], dtype=np.float64)
    candidate = np.asarray([item[2] for item in pairs], dtype=np.float64)
    improvement = candidate - baseline if direction == "higher" else baseline - candidate
    ci_low, ci_high = bootstrap_ci(improvement, bootstrap_samples, bootstrap_seed)
    baseline_mean = float(baseline.mean())
    improvement_mean = float(improvement.mean())
    return {
        "direction": direction,
        "systems": len(pairs),
        "baseline_system_macro": baseline_mean,
        "candidate_system_macro": float(candidate.mean()),
        "improvement_mean": improvement_mean,
        "relative_improvement_percent": (
            100.0 * improvement_mean / abs(baseline_mean)
            if abs(baseline_mean) > 1e-12
            else None
        ),
        "improvement_ci95_low": ci_low,
        "improvement_ci95_high": ci_high,
        "candidate_win_fraction": float(np.mean(improvement > 1e-12)),
        "tie_fraction": float(np.mean(np.abs(improvement) <= 1e-12)),
    }


def compare_files(
    baseline_path: Path,
    candidate_path: Path,
    metrics: Sequence[str],
    *,
    bootstrap_samples: int,
    bootstrap_seed: int,
) -> Dict[str, object]:
    baseline = load_records(baseline_path)
    candidate = load_records(candidate_path)
    comparisons = {}
    for index, metric in enumerate(metrics):
        if metric not in METRIC_DIRECTIONS:
            raise ValueError(f"Unknown metric direction: {metric}")
        comparisons[metric] = compare_metric(
            aligned_metric_pairs(baseline, candidate, metric),
            METRIC_DIRECTIONS[metric],
            bootstrap_samples=bootstrap_samples,
            bootstrap_seed=bootstrap_seed + index,
        )
    return {
        "schema_version": "stage2_transition_paired_comparison_v1",
        "baseline": str(baseline_path),
        "candidate": str(candidate_path),
        "aligned_systems": len(baseline),
        "comparisons": comparisons,
    }


def main() -> None:
    args = parse_args()
    metrics = tuple(item.strip() for item in args.metrics.split(",") if item.strip())
    result = compare_files(
        args.baseline,
        args.candidate,
        metrics,
        bootstrap_samples=args.bootstrap_samples,
        bootstrap_seed=args.bootstrap_seed,
    )
    result["baseline_name"] = args.baseline_name
    result["candidate_name"] = args.candidate_name
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")

    print("metric\tdirection\tbaseline\tcandidate\timprovement\tci95\twins")
    for metric, comparison in result["comparisons"].items():
        print(
            f"{metric}\t{comparison['direction']}\t"
            f"{comparison['baseline_system_macro']:.6f}\t"
            f"{comparison['candidate_system_macro']:.6f}\t"
            f"{comparison['improvement_mean']:+.6f}\t"
            f"[{comparison['improvement_ci95_low']:+.6f},"
            f"{comparison['improvement_ci95_high']:+.6f}]\t"
            f"{comparison['candidate_win_fraction']:.3f}"
        )
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
