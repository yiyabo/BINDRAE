#!/usr/bin/env python3
"""Summarize matched MD-reference evaluations with paired system bootstrap."""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Mapping

import numpy as np


DEFAULT_METRICS = (
    "md_path_translation_mae_a",
    "md_path_product_rmse",
    "phase_tau_mae",
    "phase_order_pair_accuracy",
    "ca_endpoint_mae_a",
    "ca_bond_deviation_a",
)
HIGHER_IS_BETTER = {"phase_order_pair_accuracy", "phase_order_spearman"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--result",
        action="append",
        default=[],
        help="Named evaluation as LABEL=PATH; repeat for each method.",
    )
    parser.add_argument(
        "--ensemble",
        action="append",
        default=[],
        help="Named mean ensemble as LABEL=MEMBER1,MEMBER2,...",
    )
    parser.add_argument("--primary", required=True)
    parser.add_argument("--metric", action="append", default=[])
    parser.add_argument("--bootstrap_samples", type=int, default=100_000)
    parser.add_argument("--seed", type=int, default=20260720)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def parse_assignment(value: str) -> tuple[str, str]:
    if "=" not in value:
        raise ValueError(f"Expected LABEL=VALUE, got {value!r}")
    label, assigned = value.split("=", 1)
    label = label.strip()
    assigned = assigned.strip()
    if not label or not assigned:
        raise ValueError(f"Expected non-empty LABEL=VALUE, got {value!r}")
    return label, assigned


def finite_mean(values: Iterable[float]) -> float:
    values = np.asarray(list(values), dtype=np.float64)
    values = values[np.isfinite(values)]
    return float(values.mean()) if values.size else math.nan


def load_system_metrics(path: Path) -> Dict[str, Dict[str, float]]:
    payload = json.loads(path.read_text())
    by_system: Dict[str, Dict[str, List[float]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for record in payload.get("records", []):
        sample_id = str(record["sample_id"])
        for metric, value in record.get("metrics", {}).items():
            if isinstance(value, (int, float)) and math.isfinite(float(value)):
                by_system[sample_id][metric].append(float(value))
    return {
        sample_id: {
            metric: finite_mean(values) for metric, values in metric_values.items()
        }
        for sample_id, metric_values in by_system.items()
    }


def ensemble_metrics(
    members: Iterable[Mapping[str, Mapping[str, float]]],
) -> Dict[str, Dict[str, float]]:
    members = list(members)
    common_systems = set.intersection(*(set(member) for member in members))
    result: Dict[str, Dict[str, float]] = {}
    for sample_id in sorted(common_systems):
        metric_names = set.intersection(
            *(set(member[sample_id]) for member in members)
        )
        result[sample_id] = {
            metric: finite_mean(member[sample_id][metric] for member in members)
            for metric in metric_names
        }
    return result


def paired_bootstrap(
    primary: np.ndarray,
    comparator: np.ndarray,
    *,
    higher_is_better: bool,
    samples: int,
    rng: np.random.Generator,
) -> Dict[str, float]:
    if primary.shape != comparator.shape or primary.ndim != 1:
        raise ValueError("Paired bootstrap requires equally sized vectors")
    improvement = primary - comparator if higher_is_better else comparator - primary
    n = improvement.size
    if n == 0:
        raise ValueError("Paired bootstrap requires at least one system")
    chunk_size = max(1, min(samples, 10_000))
    means = []
    remaining = samples
    while remaining:
        current = min(chunk_size, remaining)
        indices = rng.integers(0, n, size=(current, n))
        means.append(improvement[indices].mean(axis=1))
        remaining -= current
    bootstrap = np.concatenate(means)
    return {
        "paired_systems": int(n),
        "primary_mean": float(primary.mean()),
        "comparator_mean": float(comparator.mean()),
        "improvement": float(improvement.mean()),
        "ci95_low": float(np.quantile(bootstrap, 0.025)),
        "ci95_high": float(np.quantile(bootstrap, 0.975)),
        "win_rate": float(np.mean(improvement > 0.0)),
    }


def json_safe(value):
    if isinstance(value, dict):
        return {key: json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [json_safe(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def main() -> None:
    args = parse_args()
    methods: Dict[str, Dict[str, Dict[str, float]]] = {}
    sources = {}
    for assignment in args.result:
        label, raw_path = parse_assignment(assignment)
        if label in methods:
            raise ValueError(f"Duplicate method label: {label}")
        path = Path(raw_path)
        methods[label] = load_system_metrics(path)
        sources[label] = str(path)
    for assignment in args.ensemble:
        label, raw_members = parse_assignment(assignment)
        member_names = [name.strip() for name in raw_members.split(",") if name.strip()]
        if len(member_names) < 2:
            raise ValueError(f"Ensemble {label} requires at least two members")
        missing = [name for name in member_names if name not in methods]
        if missing:
            raise ValueError(f"Ensemble {label} has unknown members: {missing}")
        methods[label] = ensemble_metrics(methods[name] for name in member_names)
        sources[label] = {"members": member_names, "aggregation": "system_mean"}
    if args.primary not in methods:
        raise ValueError(f"Unknown primary method: {args.primary}")

    metrics = tuple(args.metric or DEFAULT_METRICS)
    summaries = {}
    for label, system_metrics in methods.items():
        summaries[label] = {
            "systems": len(system_metrics),
            "metrics": {
                metric: finite_mean(
                    values[metric]
                    for values in system_metrics.values()
                    if metric in values
                )
                for metric in metrics
            },
        }

    rng = np.random.default_rng(args.seed)
    comparisons = {}
    primary = methods[args.primary]
    for label, comparator in methods.items():
        if label == args.primary:
            continue
        metric_results = {}
        for metric in metrics:
            common = sorted(
                sample_id
                for sample_id in set(primary) & set(comparator)
                if metric in primary[sample_id] and metric in comparator[sample_id]
            )
            if not common:
                continue
            primary_values = np.asarray(
                [primary[sample_id][metric] for sample_id in common], dtype=np.float64
            )
            comparator_values = np.asarray(
                [comparator[sample_id][metric] for sample_id in common], dtype=np.float64
            )
            metric_results[metric] = paired_bootstrap(
                primary_values,
                comparator_values,
                higher_is_better=metric in HIGHER_IS_BETTER,
                samples=args.bootstrap_samples,
                rng=rng,
            )
        comparisons[label] = metric_results

    result = {
        "schema_version": "md_reference_benchmark_summary_v1",
        "primary": args.primary,
        "metrics": list(metrics),
        "bootstrap_samples": args.bootstrap_samples,
        "seed": args.seed,
        "sources": sources,
        "summaries": summaries,
        "paired_comparisons": comparisons,
    }
    text = json.dumps(json_safe(result), indent=2, sort_keys=True, allow_nan=False)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
