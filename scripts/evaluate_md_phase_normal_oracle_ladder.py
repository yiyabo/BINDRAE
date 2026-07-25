#!/usr/bin/env python3
"""Diagnose whether replicated MD normal residuals are predictable or multimodal.

This is a target-space oracle ladder, not a model benchmark. For every held-out
replica of an endpoint system it compares:

1. zero residual (warp-only reference),
2. a leave-one-replica-out deterministic consensus,
3. a training-only medoid route, and
4. a held-out-aware route oracle.

The medoid is a held-out-free, same-system cross-replica upper bound: its
selection never uses the held-out replica, but it still requires MD replicas of
that endpoint system and is therefore not deployable from endpoints alone. The
route oracle is deliberately leaky and may only motivate a later stochastic
path model; it must not be reported as prediction performance.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence, Tuple

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.analyze_md_replica_consistency import (  # noqa: E402
    METRIC_GROUPS,
    ReplicaTargets,
    load_replica,
    system_id_from_sample_id,
    validate_replica_group,
)


METHODS = ("zero", "loo_consensus", "train_medoid", "route_oracle")
REPRESENTATION_AUDIT_FIELDS = (
    "mean_reconstruction_translation_angstrom",
    "mean_reconstruction_rotation_rad",
    "mean_reconstruction_chi_rad",
    "mean_projected_parallel_cos_abs",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--system-list",
        type=Path,
        default=None,
        help="Optional newline-delimited system IDs. Use validation systems only.",
    )
    parser.add_argument("--min-replicas", type=int, default=3)
    parser.add_argument("--min-t", type=float, default=0.05)
    parser.add_argument("--max-t", type=float, default=0.95)
    parser.add_argument("--min-residual-confidence", type=float, default=0.0)
    parser.add_argument("--min-consensus-support-count", type=int, default=2)
    parser.add_argument("--min-consensus-support-fraction", type=float, default=0.5)
    parser.add_argument("--bootstrap-samples", type=int, default=10000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260719)
    parser.add_argument(
        "--minimum-practical-relative-reduction",
        type=float,
        default=0.05,
        help="Minimum macro MSE reduction used only for the decision hint.",
    )
    return parser.parse_args()


def load_system_filter(path: Optional[Path]) -> Optional[set[str]]:
    if path is None:
        return None
    systems = {line.strip() for line in path.read_text().splitlines() if line.strip()}
    if not systems:
        raise ValueError(f"System list is empty: {path}")
    return systems


def load_manifest(cache_dir: Path) -> list[Dict[str, Any]]:
    manifest_path = cache_dir / "manifest.jsonl"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Missing immutable cache manifest: {manifest_path}")
    records: list[Dict[str, Any]] = []
    seen: set[str] = set()
    for line_number, line in enumerate(manifest_path.read_text().splitlines(), start=1):
        if not line.strip():
            continue
        record = json.loads(line)
        sample_id = str(record.get("sample_id", ""))
        relative_path = str(record.get("relative_path", ""))
        if not sample_id or not relative_path:
            raise ValueError(
                f"{manifest_path}:{line_number} lacks sample_id or relative_path"
            )
        if sample_id in seen:
            raise ValueError(f"Duplicate sample_id in manifest: {sample_id}")
        seen.add(sample_id)
        target_path = cache_dir / relative_path
        if not target_path.is_file():
            raise FileNotFoundError(f"Missing target declared by manifest: {target_path}")
        copied = dict(record)
        copied["target_path"] = target_path
        copied["system_id"] = system_id_from_sample_id(sample_id)
        records.append(copied)
    if not records:
        raise ValueError(f"No records in {manifest_path}")
    return records


def summarize_values(values: Iterable[float]) -> Dict[str, float]:
    array = np.asarray([value for value in values if math.isfinite(value)], dtype=float)
    if array.size == 0:
        return {"count": 0}
    return {
        "count": int(array.size),
        "mean": float(array.mean()),
        "std": float(array.std()),
        "min": float(array.min()),
        "p10": float(np.quantile(array, 0.10)),
        "p50": float(np.quantile(array, 0.50)),
        "p90": float(np.quantile(array, 0.90)),
        "p95": float(np.quantile(array, 0.95)),
        "max": float(array.max()),
    }


def representation_audit_summary(
    manifest_records: Sequence[Mapping[str, Any]],
) -> Dict[str, Any]:
    metrics: Dict[str, Dict[str, float]] = {}
    for field in REPRESENTATION_AUDIT_FIELDS:
        metrics[field] = summarize_values(
            float(record["audit_metrics"][field])
            for record in manifest_records
            if field in record.get("audit_metrics", {})
        )
    return {
        "replicas": len(manifest_records),
        "metrics": metrics,
        "interpretation": {
            "lower_is_better": list(REPRESENTATION_AUDIT_FIELDS),
            "scope": (
                "Exporter self-reconstruction against each source MD path. "
                "This tests representation fidelity, not held-out prediction."
            ),
        },
    }


def residual_weights(
    replica: ReplicaTargets,
    group: str,
    interior: np.ndarray,
    min_confidence: float,
) -> np.ndarray:
    weights = np.asarray(replica.metric_weights[group], dtype=np.float64)
    return weights * (
        interior[:, None, None]
        & (replica.residual_confidence[..., None] >= min_confidence)
    )


def weighted_prediction_metrics(
    prediction: np.ndarray,
    target: np.ndarray,
    target_weight: np.ndarray,
    *,
    available: Optional[np.ndarray] = None,
    eps: float = 1e-12,
) -> Dict[str, float]:
    """Score a predictor on every held-out target cell, filling missing cells with 0."""
    prediction = np.asarray(prediction, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    target_weight = np.maximum(np.asarray(target_weight, dtype=np.float64), 0.0)
    if prediction.shape != target.shape or target_weight.shape != target.shape:
        raise ValueError("prediction, target, and target_weight must share a shape")
    if available is None:
        available = np.ones(target.shape, dtype=bool)
    available = np.broadcast_to(np.asarray(available, dtype=bool), target.shape)
    effective_prediction = np.where(available, prediction, 0.0)
    weight_sum = float(target_weight.sum())
    support = int(np.count_nonzero(target_weight > 0.0))
    if weight_sum <= eps:
        return {"support": support, "weight_sum": weight_sum}

    squared_error = float(
        np.sum(target_weight * np.square(effective_prediction - target))
    )
    absolute_error = float(
        np.sum(target_weight * np.abs(effective_prediction - target))
    )
    target_energy = float(np.sum(target_weight * np.square(target)))
    prediction_energy = float(
        np.sum(target_weight * np.square(effective_prediction))
    )
    dot = float(np.sum(target_weight * effective_prediction * target))
    mse = squared_error / weight_sum
    target_mse = target_energy / weight_sum
    result = {
        "support": support,
        "weight_sum": weight_sum,
        "mae": absolute_error / weight_sum,
        "mse": mse,
        "rmse": math.sqrt(max(mse, 0.0)),
        "target_rms": math.sqrt(max(target_mse, 0.0)),
        "available_weight_fraction": float(
            np.sum(target_weight * available) / weight_sum
        ),
    }
    if target_mse > eps:
        result["relative_rmse"] = math.sqrt(max(mse / target_mse, 0.0))
        result["relative_mse_reduction_vs_zero"] = 1.0 - mse / target_mse
    norm_product = math.sqrt(max(target_energy * prediction_energy, 0.0))
    if norm_product > eps:
        result["cosine"] = max(-1.0, min(1.0, dot / norm_product))
    return result


def weighted_consensus_prediction(
    values: np.ndarray,
    weights: np.ndarray,
    *,
    min_support_count: int,
    min_support_fraction: float,
    eps: float = 1e-12,
) -> Tuple[np.ndarray, np.ndarray]:
    values = np.asarray(values, dtype=np.float64)
    weights = np.maximum(np.asarray(weights, dtype=np.float64), 0.0)
    if values.shape != weights.shape or values.ndim < 2:
        raise ValueError("Expected replica-first values and weights with equal shapes")
    if values.shape[0] < 1:
        raise ValueError("Consensus requires at least one source replica")
    if min_support_count < 1:
        raise ValueError("min_support_count must be positive")
    if not 0.0 < min_support_fraction <= 1.0:
        raise ValueError("min_support_fraction must be in (0, 1]")

    support = np.count_nonzero(weights > 0.0, axis=0)
    required = max(
        min_support_count,
        math.ceil(values.shape[0] * min_support_fraction),
    )
    weight_sum = weights.sum(axis=0)
    consensus = np.divide(
        np.sum(values * weights, axis=0),
        weight_sum,
        out=np.zeros_like(values[0], dtype=np.float64),
        where=weight_sum > eps,
    )
    available = (support >= required) & (weight_sum > eps)
    return np.where(available, consensus, 0.0), available


def select_train_medoid(
    values: Sequence[np.ndarray],
    weights: Sequence[np.ndarray],
) -> Tuple[int, list[float]]:
    """Select one route using only its fit to the other training replicas."""
    if len(values) != len(weights) or len(values) < 2:
        raise ValueError("Medoid selection requires at least two training replicas")
    scores: list[float] = []
    for candidate_index, (candidate, candidate_weight) in enumerate(
        zip(values, weights)
    ):
        candidate_available = candidate_weight > 0.0
        candidate_scores = []
        for target_index, (target, target_weight) in enumerate(zip(values, weights)):
            if target_index == candidate_index:
                continue
            metrics = weighted_prediction_metrics(
                candidate,
                target,
                target_weight,
                available=candidate_available,
            )
            if "mse" in metrics:
                candidate_scores.append(metrics["mse"])
        scores.append(float(np.mean(candidate_scores)) if candidate_scores else math.inf)
    return int(np.argmin(scores)), scores


def select_route_oracle(
    values: Sequence[np.ndarray],
    weights: Sequence[np.ndarray],
    heldout_value: np.ndarray,
    heldout_weight: np.ndarray,
) -> Tuple[int, list[float]]:
    """Select the closest route using the held-out target (diagnostic leakage)."""
    if len(values) != len(weights) or not values:
        raise ValueError("Route oracle requires at least one candidate")
    scores = []
    for candidate, candidate_weight in zip(values, weights):
        metrics = weighted_prediction_metrics(
            candidate,
            heldout_value,
            heldout_weight,
            available=candidate_weight > 0.0,
        )
        scores.append(float(metrics.get("mse", math.inf)))
    return int(np.argmin(scores)), scores


def prefix_metrics(prefix: str, metrics: Mapping[str, float]) -> Dict[str, float]:
    return {f"{prefix}_{key}": value for key, value in metrics.items()}


def evaluate_holdout(
    replicas: Sequence[ReplicaTargets],
    heldout_index: int,
    *,
    min_t: float,
    max_t: float,
    min_residual_confidence: float,
    min_consensus_support_count: int,
    min_consensus_support_fraction: float,
) -> Dict[str, Any]:
    validate_replica_group(replicas)
    heldout = replicas[heldout_index]
    training = [
        replica for index, replica in enumerate(replicas) if index != heldout_index
    ]
    interior = (heldout.t_values >= min_t) & (heldout.t_values <= max_t)
    if not interior.any():
        raise ValueError(f"No interior frames for {heldout.system_id}")

    group_values: Dict[str, list[np.ndarray]] = {}
    group_weights: Dict[str, list[np.ndarray]] = {}
    for group in METRIC_GROUPS:
        group_values[group] = [
            np.asarray(replica.metric_values[group], dtype=np.float64)
            for replica in training
        ]
        group_weights[group] = [
            residual_weights(
                replica,
                group,
                interior,
                min_residual_confidence,
            )
            for replica in training
        ]

    heldout_combined_weight = residual_weights(
        heldout,
        "combined",
        interior,
        min_residual_confidence,
    )
    medoid_index, medoid_scores = select_train_medoid(
        group_values["combined"], group_weights["combined"]
    )
    oracle_index, oracle_scores = select_route_oracle(
        group_values["combined"],
        group_weights["combined"],
        np.asarray(heldout.metric_values["combined"], dtype=np.float64),
        heldout_combined_weight,
    )
    record: Dict[str, Any] = {
        "system_id": heldout.system_id,
        "heldout_sample_id": heldout.sample_id,
        "replicas": len(replicas),
        "training_replicas": len(training),
        "training_sample_ids": [replica.sample_id for replica in training],
        "train_medoid_sample_id": training[medoid_index].sample_id,
        "route_oracle_sample_id": training[oracle_index].sample_id,
        "train_medoid_selection_mse": medoid_scores[medoid_index],
        "route_oracle_selection_mse": oracle_scores[oracle_index],
    }

    for group in METRIC_GROUPS:
        target = np.asarray(heldout.metric_values[group], dtype=np.float64)
        target_weight = residual_weights(
            heldout,
            group,
            interior,
            min_residual_confidence,
        )
        zero = np.zeros_like(target)
        record.update(
            prefix_metrics(
                f"{group}_zero",
                weighted_prediction_metrics(zero, target, target_weight),
            )
        )

        stacked_values = np.stack(group_values[group], axis=0)
        stacked_weights = np.stack(group_weights[group], axis=0)
        consensus, consensus_available = weighted_consensus_prediction(
            stacked_values,
            stacked_weights,
            min_support_count=min_consensus_support_count,
            min_support_fraction=min_consensus_support_fraction,
        )
        record.update(
            prefix_metrics(
                f"{group}_loo_consensus",
                weighted_prediction_metrics(
                    consensus,
                    target,
                    target_weight,
                    available=consensus_available,
                ),
            )
        )

        for method, selected_index in (
            ("train_medoid", medoid_index),
            ("route_oracle", oracle_index),
        ):
            selected_value = group_values[group][selected_index]
            selected_weight = group_weights[group][selected_index]
            record.update(
                prefix_metrics(
                    f"{group}_{method}",
                    weighted_prediction_metrics(
                        selected_value,
                        target,
                        target_weight,
                        available=selected_weight > 0.0,
                    ),
                )
            )
    return record


def aggregate_system_holdouts(
    system_id: str,
    holdouts: Sequence[Mapping[str, Any]],
) -> Dict[str, Any]:
    record: Dict[str, Any] = {
        "system_id": system_id,
        "replicas": len(holdouts),
        "heldout_sample_ids": [str(item["heldout_sample_id"]) for item in holdouts],
    }
    for group in METRIC_GROUPS:
        for method in METHODS:
            for metric in (
                "mae",
                "mse",
                "rmse",
                "relative_rmse",
                "relative_mse_reduction_vs_zero",
                "available_weight_fraction",
                "cosine",
            ):
                key = f"{group}_{method}_{metric}"
                values = [float(item[key]) for item in holdouts if key in item]
                if values:
                    record[key] = float(np.mean(values))

        zero_mse = record.get(f"{group}_zero_mse")
        if zero_mse is None:
            continue
        for method in METHODS[1:]:
            method_mse = record.get(f"{group}_{method}_mse")
            if method_mse is None:
                continue
            record[f"{group}_{method}_mse_gain_vs_zero"] = zero_mse - method_mse
            if zero_mse > 1e-12:
                record[
                    f"{group}_{method}_relative_mse_reduction_vs_zero_macro"
                ] = 1.0 - method_mse / zero_mse
        consensus_mse = record.get(f"{group}_loo_consensus_mse")
        oracle_mse = record.get(f"{group}_route_oracle_mse")
        if consensus_mse is not None and oracle_mse is not None:
            record[f"{group}_route_oracle_mse_gain_vs_loo_consensus"] = (
                consensus_mse - oracle_mse
            )
            if consensus_mse > 1e-12:
                record[
                    f"{group}_route_oracle_relative_mse_reduction_vs_loo_consensus"
                ] = 1.0 - oracle_mse / consensus_mse
    return record


def bootstrap_mean_ci(
    values: Sequence[float],
    *,
    samples: int,
    seed: int,
) -> Dict[str, float]:
    array = np.asarray([value for value in values if math.isfinite(value)], dtype=float)
    if array.size == 0:
        return {"count": 0}
    if samples < 1:
        raise ValueError("bootstrap samples must be positive")
    rng = np.random.default_rng(seed)
    if array.size == 1:
        bootstrap = np.repeat(array, samples)
    else:
        indices = rng.integers(0, array.size, size=(samples, array.size))
        bootstrap = array[indices].mean(axis=1)
    return {
        "count": int(array.size),
        "mean": float(array.mean()),
        "median": float(np.median(array)),
        "ci95_low": float(np.quantile(bootstrap, 0.025)),
        "ci95_high": float(np.quantile(bootstrap, 0.975)),
        "systems_candidate_better_fraction": float(np.mean(array > 0.0)),
    }


def comparison_summary(
    system_records: Sequence[Mapping[str, Any]],
    *,
    group: str,
    candidate: str,
    baseline: str,
    bootstrap_samples: int,
    bootstrap_seed: int,
) -> Dict[str, Any]:
    candidate_key = f"{group}_{candidate}_mse"
    baseline_key = f"{group}_{baseline}_mse"
    paired = [
        (float(record[baseline_key]), float(record[candidate_key]))
        for record in system_records
        if baseline_key in record and candidate_key in record
    ]
    result: Dict[str, Any] = bootstrap_mean_ci(
        [baseline_mse - candidate_mse for baseline_mse, candidate_mse in paired],
        samples=bootstrap_samples,
        seed=bootstrap_seed,
    )
    if paired:
        baseline_macro = float(np.mean([pair[0] for pair in paired]))
        candidate_macro = float(np.mean([pair[1] for pair in paired]))
        result.update(
            {
                "baseline_macro_mse": baseline_macro,
                "candidate_macro_mse": candidate_macro,
                "relative_macro_mse_reduction": (
                    1.0 - candidate_macro / baseline_macro
                    if baseline_macro > 1e-12
                    else 0.0
                ),
                "comparison": f"{candidate}_vs_{baseline}",
                "metric": "baseline_mse_minus_candidate_mse",
                "higher_is_better": True,
            }
        )
    return result


def decision_hint(
    comparisons: Mapping[str, Mapping[str, Any]],
    minimum_practical_relative_reduction: float,
) -> Dict[str, Any]:
    deterministic = comparisons["loo_consensus_vs_zero"]
    route = comparisons["route_oracle_vs_loo_consensus"]

    def supported(result: Mapping[str, Any]) -> bool:
        return (
            float(result.get("ci95_low", -math.inf)) > 0.0
            and float(result.get("relative_macro_mse_reduction", -math.inf))
            >= minimum_practical_relative_reduction
        )

    deterministic_supported = supported(deterministic)
    route_supported = supported(route)
    if deterministic_supported and route_supported:
        direction = "shared_deterministic_target_plus_possible_route_variation"
    elif deterministic_supported:
        direction = "shared_deterministic_target_exists"
    elif route_supported:
        direction = "stochastic_path_latent_candidate"
    else:
        direction = "stop_residual_or_revisit_silver_targets"
    return {
        "direction": direction,
        "deterministic_consensus_supported": deterministic_supported,
        "route_oracle_supported": route_supported,
        "minimum_practical_relative_reduction": (
            minimum_practical_relative_reduction
        ),
        "warning": (
            "The LOO consensus and medoid both use other MD replicas of the same "
            "endpoint system, so they test target identifiability rather than "
            "endpoint-conditioned predictability. The route oracle additionally "
            "sees the held-out target. A positive route result only licenses a "
            "stochastic probe and does not distinguish modes from noise by itself."
        ),
    }


def write_jsonl(path: Path, records: Iterable[Mapping[str, Any]]) -> None:
    path.write_text(
        "".join(json.dumps(record, sort_keys=True) + "\n" for record in records)
    )


def run_oracle_ladder(
    cache_dir: Path,
    output_dir: Path,
    *,
    system_filter: Optional[set[str]] = None,
    min_replicas: int = 3,
    min_t: float = 0.05,
    max_t: float = 0.95,
    min_residual_confidence: float = 0.0,
    min_consensus_support_count: int = 2,
    min_consensus_support_fraction: float = 0.5,
    bootstrap_samples: int = 10000,
    bootstrap_seed: int = 20260719,
    minimum_practical_relative_reduction: float = 0.05,
) -> Dict[str, Any]:
    if min_replicas < 3:
        raise ValueError("LOO consensus requires min_replicas >= 3")
    if not 0.0 <= min_t < max_t <= 1.0:
        raise ValueError("Expected 0 <= min_t < max_t <= 1")
    manifest_records = load_manifest(cache_dir)
    if system_filter is not None:
        available_systems = {str(record["system_id"]) for record in manifest_records}
        missing = sorted(system_filter - available_systems)
        if missing:
            raise ValueError(f"Requested systems absent from cache: {missing}")
        manifest_records = [
            record
            for record in manifest_records
            if str(record["system_id"]) in system_filter
        ]

    grouped: Dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for record in manifest_records:
        grouped[str(record["system_id"])].append(record)
    eligible = {
        system_id: records
        for system_id, records in grouped.items()
        if len(records) >= min_replicas
    }
    if not eligible:
        raise ValueError(f"No systems have at least {min_replicas} replicas")

    holdout_records: list[Dict[str, Any]] = []
    system_records: list[Dict[str, Any]] = []
    for system_index, system_id in enumerate(sorted(eligible), start=1):
        records = sorted(eligible[system_id], key=lambda record: record["sample_id"])
        replicas = [
            load_replica(str(record["sample_id"]), Path(record["target_path"]))
            for record in records
        ]
        validate_replica_group(replicas)
        system_holdouts = [
            evaluate_holdout(
                replicas,
                heldout_index,
                min_t=min_t,
                max_t=max_t,
                min_residual_confidence=min_residual_confidence,
                min_consensus_support_count=min_consensus_support_count,
                min_consensus_support_fraction=min_consensus_support_fraction,
            )
            for heldout_index in range(len(replicas))
        ]
        holdout_records.extend(system_holdouts)
        system_records.append(aggregate_system_holdouts(system_id, system_holdouts))
        if system_index % 10 == 0 or system_index == len(eligible):
            print(f"Analyzed {system_index}/{len(eligible)} systems", flush=True)

    comparisons: Dict[str, Dict[str, Dict[str, Any]]] = {}
    for group_index, group in enumerate(METRIC_GROUPS):
        comparisons[group] = {}
        for comparison_index, (candidate, baseline) in enumerate(
            (
                ("loo_consensus", "zero"),
                ("train_medoid", "zero"),
                ("route_oracle", "zero"),
                ("route_oracle", "loo_consensus"),
            )
        ):
            name = f"{candidate}_vs_{baseline}"
            comparisons[group][name] = comparison_summary(
                system_records,
                group=group,
                candidate=candidate,
                baseline=baseline,
                bootstrap_samples=bootstrap_samples,
                bootstrap_seed=(
                    bootstrap_seed + group_index * 100 + comparison_index
                ),
            )

    combined_decision = decision_hint(
        comparisons["combined"], minimum_practical_relative_reduction
    )
    report: Dict[str, Any] = {
        "schema_version": "md_phase_normal_oracle_ladder_v1",
        "cache_dir": str(cache_dir),
        "systems_selected": len(grouped),
        "systems_analyzed": len(system_records),
        "systems_below_min_replicas": len(grouped) - len(eligible),
        "replicas_selected": len(manifest_records),
        "replicas_analyzed": len(holdout_records),
        "min_replicas": min_replicas,
        "interior_t_range": [min_t, max_t],
        "min_residual_confidence": min_residual_confidence,
        "consensus_support": {
            "minimum_count": min_consensus_support_count,
            "minimum_fraction": min_consensus_support_fraction,
        },
        "representation_self_reconstruction": representation_audit_summary(
            manifest_records
        ),
        "comparisons": comparisons,
        "combined_decision_hint": combined_decision,
        "interpretation": {
            "lower_is_better": ["*_mae", "*_mse", "*_rmse", "*_relative_rmse"],
            "higher_is_better": [
                "*_cosine",
                "*_available_weight_fraction",
                "*_relative_mse_reduction*",
                "baseline_mse_minus_candidate_mse",
            ],
            "zero": "No off-bridge correction; target-space warp-only reference.",
            "loo_consensus": (
                "Weighted deterministic mean of the other replicas. Missing "
                "training support is conservatively filled with zero."
            ),
            "train_medoid": (
                "One route selected only by its average fit to the other training "
                "replicas. It avoids held-out leakage but still requires same-system "
                "MD replicas, so it is not endpoint-only model performance."
            ),
            "route_oracle": (
                "Best remaining route selected after seeing the held-out target. "
                "Diagnostic upper bound only, never prediction performance."
            ),
            "macro_unit": "endpoint system",
            "test_split_policy": (
                "Run on the frozen validation system list. Do not use the untouched "
                "test systems for architecture selection."
            ),
        },
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(output_dir / "holdouts.jsonl", holdout_records)
    write_jsonl(output_dir / "systems.jsonl", system_records)
    (output_dir / "summary.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n"
    )
    return report


def main() -> None:
    args = parse_args()
    report = run_oracle_ladder(
        args.cache_dir,
        args.output_dir,
        system_filter=load_system_filter(args.system_list),
        min_replicas=args.min_replicas,
        min_t=args.min_t,
        max_t=args.max_t,
        min_residual_confidence=args.min_residual_confidence,
        min_consensus_support_count=args.min_consensus_support_count,
        min_consensus_support_fraction=args.min_consensus_support_fraction,
        bootstrap_samples=args.bootstrap_samples,
        bootstrap_seed=args.bootstrap_seed,
        minimum_practical_relative_reduction=(
            args.minimum_practical_relative_reduction
        ),
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
