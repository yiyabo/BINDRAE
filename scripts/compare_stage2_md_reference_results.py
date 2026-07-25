#!/usr/bin/env python3
"""Paired system-level comparison for Stage-2 MD-reference evaluations."""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np


METRIC_DIRECTIONS = {
    "md_path_product_rmse": "lower",
    "md_path_translation_mae_a": "lower",
    "md_path_rotation_mae_rad": "lower",
    "md_path_chi_mae_rad": "lower",
    "phase_midpoint_mae": "lower",
    "phase_tau_mae": "lower",
    "contact_event_timing_mae": "lower",
    "phase_order_pair_accuracy": "higher",
    "phase_order_spearman": "higher",
    "contact_event_coverage": "higher",
    "contact_event_matched": "higher",
    "contact_event_order_spearman": "higher",
    "transient_contact_recall": "higher",
}

DEFAULT_METRICS = (
    "md_path_product_rmse",
    "md_path_translation_mae_a",
    "md_path_rotation_mae_rad",
    "md_path_chi_mae_rad",
    "phase_tau_mae",
    "phase_order_pair_accuracy",
    "contact_event_coverage",
    "transient_contact_recall",
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
    parser.add_argument(
        "--noninferiority-margin-percent",
        type=float,
        default=None,
        help="optional practical-equivalence margin relative to the baseline mean",
    )
    return parser.parse_args()


def load_records(path: Path) -> List[Mapping[str, object]]:
    payload = json.loads(path.read_text())
    records = payload.get("records")
    if not isinstance(records, list) or not records:
        raise ValueError(f"No records found in {path}")
    return records


def record_key(record: Mapping[str, object]) -> Tuple[str, str]:
    return str(record["sample_id"]), str(record["reference_id"])


def aligned_records(
    baseline: Sequence[Mapping[str, object]],
    candidate: Sequence[Mapping[str, object]],
) -> List[Tuple[Mapping[str, object], Mapping[str, object]]]:
    baseline_map = {record_key(record): record for record in baseline}
    candidate_map = {record_key(record): record for record in candidate}
    if len(baseline_map) != len(baseline) or len(candidate_map) != len(candidate):
        raise ValueError("Duplicate (sample_id, reference_id) records")
    if baseline_map.keys() != candidate_map.keys():
        missing_candidate = sorted(baseline_map.keys() - candidate_map.keys())
        missing_baseline = sorted(candidate_map.keys() - baseline_map.keys())
        raise ValueError(
            "Evaluation records are not aligned: "
            f"missing_candidate={missing_candidate[:3]} "
            f"missing_baseline={missing_baseline[:3]}"
        )
    return [(baseline_map[key], candidate_map[key]) for key in sorted(baseline_map)]


def finite_metric(record: Mapping[str, object], metric: str) -> float | None:
    metrics = record.get("metrics")
    if not isinstance(metrics, Mapping):
        return None
    value = metrics.get(metric)
    if not isinstance(value, (int, float)):
        return None
    value = float(value)
    return value if math.isfinite(value) else None


def system_metric_pairs(
    records: Iterable[Tuple[Mapping[str, object], Mapping[str, object]]],
    metric: str,
) -> List[Tuple[str, float, float]]:
    grouped: Dict[str, List[Tuple[float, float]]] = defaultdict(list)
    for baseline, candidate in records:
        baseline_value = finite_metric(baseline, metric)
        candidate_value = finite_metric(candidate, metric)
        if baseline_value is None or candidate_value is None:
            continue
        grouped[str(baseline["sample_id"])].append(
            (baseline_value, candidate_value)
        )
    output = []
    for sample_id, values in sorted(grouped.items()):
        array = np.asarray(values, dtype=np.float64)
        output.append((sample_id, float(array[:, 0].mean()), float(array[:, 1].mean())))
    return output


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
    noninferiority_margin_percent: float | None = None,
) -> Dict[str, object]:
    if not pairs:
        raise ValueError("No finite aligned systems for metric")
    baseline = np.asarray([item[1] for item in pairs], dtype=np.float64)
    candidate = np.asarray([item[2] for item in pairs], dtype=np.float64)
    improvement = candidate - baseline if direction == "higher" else baseline - candidate
    ci_low, ci_high = bootstrap_ci(improvement, bootstrap_samples, bootstrap_seed)
    baseline_mean = float(baseline.mean())
    improvement_mean = float(improvement.mean())
    relative = (
        100.0 * improvement_mean / abs(baseline_mean)
        if abs(baseline_mean) > 1e-12
        else None
    )
    result = {
        "direction": direction,
        "systems": len(pairs),
        "baseline_system_macro": baseline_mean,
        "candidate_system_macro": float(candidate.mean()),
        "improvement_mean": improvement_mean,
        "relative_improvement_percent": relative,
        "improvement_ci95_low": ci_low,
        "improvement_ci95_high": ci_high,
        "candidate_win_fraction": float(np.mean(improvement > 1e-12)),
        "tie_fraction": float(np.mean(np.abs(improvement) <= 1e-12)),
    }
    if noninferiority_margin_percent is not None:
        if noninferiority_margin_percent < 0.0:
            raise ValueError("noninferiority_margin_percent must be nonnegative")
        margin = abs(baseline_mean) * noninferiority_margin_percent / 100.0
        result.update(
            {
                "noninferiority_margin_percent": noninferiority_margin_percent,
                "noninferiority_margin_absolute": margin,
                "noninferiority_passes": bool(ci_low >= -margin),
            }
        )
    return result


def compare_files(
    baseline_path: Path,
    candidate_path: Path,
    metrics: Sequence[str],
    *,
    bootstrap_samples: int,
    bootstrap_seed: int,
    noninferiority_margin_percent: float | None = None,
) -> Dict[str, object]:
    aligned = aligned_records(load_records(baseline_path), load_records(candidate_path))
    comparisons = {}
    for index, metric in enumerate(metrics):
        if metric not in METRIC_DIRECTIONS:
            raise ValueError(f"Unknown metric direction: {metric}")
        pairs = system_metric_pairs(aligned, metric)
        comparisons[metric] = compare_metric(
            pairs,
            METRIC_DIRECTIONS[metric],
            bootstrap_samples=bootstrap_samples,
            bootstrap_seed=bootstrap_seed + index,
            noninferiority_margin_percent=noninferiority_margin_percent,
        )
    return {
        "schema_version": "stage2_md_reference_paired_comparison_v1",
        "baseline": str(baseline_path),
        "candidate": str(candidate_path),
        "aligned_replicas": len(aligned),
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
        noninferiority_margin_percent=args.noninferiority_margin_percent,
    )
    result["baseline_name"] = args.baseline_name
    result["candidate_name"] = args.candidate_name
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")

    print("metric\tdirection\tbaseline\tcandidate\timprovement\tci95\twins\tnoninferior")
    for metric, comparison in result["comparisons"].items():
        print(
            f"{metric}\t{comparison['direction']}\t"
            f"{comparison['baseline_system_macro']:.6f}\t"
            f"{comparison['candidate_system_macro']:.6f}\t"
            f"{comparison['improvement_mean']:+.6f}\t"
            f"[{comparison['improvement_ci95_low']:+.6f},"
            f"{comparison['improvement_ci95_high']:+.6f}]\t"
            f"{comparison['candidate_win_fraction']:.3f}\t"
            f"{comparison.get('noninferiority_passes', 'NA')}"
        )
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
