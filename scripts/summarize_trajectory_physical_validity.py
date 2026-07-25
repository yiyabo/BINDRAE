#!/usr/bin/env python3
"""Aggregate multi-seed per-system trajectory physical-validity metrics."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence

import numpy as np

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from scripts.summarize_md_reference_benchmark import paired_bootstrap


DEFAULT_METRICS = (
    "peptide_bond_mae_a",
    "peptide_bond_max_error_a",
    "peptide_bond_violation_frac",
    "peptide_angle_mae_rad",
    "peptide_angle_violation_frac",
    "peptide_omega_planarity_mae_rad",
    "peptide_omega_violation_frac",
    "all_atom_clash_penalty",
    "ligand_clash_atom_fraction",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", action="append", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--metric", action="append", default=[])
    parser.add_argument("--model_method", default="model")
    parser.add_argument("--baseline_method", default="cubic_ref")
    parser.add_argument("--bootstrap_samples", type=int, default=100_000)
    parser.add_argument("--seed", type=int, default=20260720)
    return parser.parse_args()


def load_rows(path: Path) -> List[Dict[str, object]]:
    rows = []
    with path.open() as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def finite_mean(values: Iterable[object]) -> float:
    finite = [
        float(value)
        for value in values
        if isinstance(value, (int, float)) and math.isfinite(float(value))
    ]
    return float(np.mean(finite)) if finite else math.nan


def ensemble_method(
    members: Sequence[Sequence[Mapping[str, object]]], method: str
) -> Dict[str, Dict[str, float]]:
    member_maps = []
    for rows in members:
        member_maps.append(
            {
                str(row["sample_id"]): row
                for row in rows
                if str(row.get("method")) == method
            }
        )
    shared = set.intersection(*(set(member) for member in member_maps))
    result = {}
    for sample_id in sorted(shared):
        metric_names = set.intersection(
            *(set(member[sample_id]) for member in member_maps)
        )
        result[sample_id] = {
            metric: finite_mean(member[sample_id].get(metric) for member in member_maps)
            for metric in metric_names
            if metric not in {"sample_id", "method", "n_residues"}
        }
    return result


def metric_summary(
    records: Mapping[str, Mapping[str, float]], metrics: Sequence[str]
) -> Dict[str, float]:
    return {
        metric: finite_mean(record.get(metric) for record in records.values())
        for metric in metrics
    }


def metric_distributions(
    records: Mapping[str, Mapping[str, float]], metrics: Sequence[str]
) -> Dict[str, Dict[str, float]]:
    result = {}
    for metric in metrics:
        values = np.asarray(
            [
                float(record[metric])
                for record in records.values()
                if math.isfinite(record.get(metric, math.nan))
            ],
            dtype=np.float64,
        )
        if values.size:
            result[metric] = {
                "mean": float(values.mean()),
                "p50": float(np.quantile(values, 0.50)),
                "p90": float(np.quantile(values, 0.90)),
                "p95": float(np.quantile(values, 0.95)),
                "max": float(values.max()),
            }
    return result


def json_safe(value):
    if isinstance(value, dict):
        return {key: json_safe(item) for key, item in value.items()}
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def main() -> None:
    args = parse_args()
    members = [load_rows(Path(path)) for path in args.input]
    model = ensemble_method(members, args.model_method)
    baseline = ensemble_method(members, args.baseline_method)
    metrics = tuple(args.metric or DEFAULT_METRICS)
    comparisons = {}
    rng = np.random.default_rng(args.seed)
    for metric in metrics:
        shared = sorted(
            sample_id
            for sample_id in set(model) & set(baseline)
            if math.isfinite(model[sample_id].get(metric, math.nan))
            and math.isfinite(baseline[sample_id].get(metric, math.nan))
        )
        if not shared:
            continue
        comparisons[metric] = paired_bootstrap(
            np.asarray([model[sample_id][metric] for sample_id in shared]),
            np.asarray([baseline[sample_id][metric] for sample_id in shared]),
            higher_is_better=False,
            samples=args.bootstrap_samples,
            rng=rng,
        )
    result = {
        "schema_version": "trajectory_physical_validity_summary_v1",
        "inputs": args.input,
        "model_method": args.model_method,
        "baseline_method": args.baseline_method,
        "systems": len(set(model) & set(baseline)),
        "model_summary": metric_summary(model, metrics),
        "baseline_summary": metric_summary(baseline, metrics),
        "model_distributions": metric_distributions(model, metrics),
        "baseline_distributions": metric_distributions(baseline, metrics),
        "paired_comparisons": comparisons,
        "model_system_metrics": model,
        "baseline_system_metrics": baseline,
    }
    text = json.dumps(json_safe(result), indent=2, sort_keys=True, allow_nan=False)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
