"""Offline path critic for Stage-2 validation metrics.

This module deliberately scores existing ``metrics.jsonl`` records instead of
touching the training loop. The score is a rank-normalized weighted aggregate:
lower critic scores are better, and each metric is compared only against the
records supplied in the same evaluation call.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, MutableMapping, Sequence


@dataclass(frozen=True)
class MetricSpec:
    """Definition of one validation metric used by the critic."""

    key: str
    direction: str
    weight: float
    label: str

    def __post_init__(self) -> None:
        if self.direction not in {"min", "max"}:
            raise ValueError(f"Unsupported metric direction: {self.direction}")
        if self.weight <= 0:
            raise ValueError(f"Metric weight must be positive: {self.key}")


DEFAULT_METRIC_SPECS: Sequence[MetricSpec] = (
    MetricSpec("val_end", "min", 3.0, "endpoint"),
    MetricSpec("val_clash", "min", 1.0, "clash"),
    MetricSpec("val_pep", "min", 0.8, "peptide"),
    MetricSpec("val_smooth", "min", 0.8, "smoothness"),
    MetricSpec("val_contact", "min", 1.0, "contact_path_loss"),
    MetricSpec("val_contact_score_holo_gap_abs", "min", 1.8, "contact_holo_gap"),
    MetricSpec("val_contact_score_direction_acc", "max", 0.8, "contact_direction"),
    MetricSpec("val_interaction_prior_final", "min", 0.8, "prior_final"),
)


def is_finite_number(value: object) -> bool:
    return isinstance(value, (int, float)) and math.isfinite(float(value))


def resolve_metric_paths(paths: Iterable[str | Path]) -> List[Path]:
    """Resolve files or directories into sorted ``metrics.jsonl`` paths."""

    metric_paths: List[Path] = []
    for raw_path in paths:
        path = Path(raw_path)
        if path.is_dir():
            metric_paths.extend(sorted(path.rglob("metrics.jsonl")))
        elif path.is_file():
            metric_paths.append(path)
        else:
            raise FileNotFoundError(f"Metrics path does not exist: {path}")
    return sorted(dict.fromkeys(metric_paths))


def load_metric_records(paths: Iterable[str | Path]) -> List[Dict[str, object]]:
    """Load Stage-2 JSONL metric records and attach source metadata."""

    records: List[Dict[str, object]] = []
    for path in resolve_metric_paths(paths):
        run_id = path.parent.name
        with path.open("r", encoding="utf-8") as handle:
            for line_no, line in enumerate(handle, start=1):
                text = line.strip()
                if not text:
                    continue
                try:
                    record = json.loads(text)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"Invalid JSON in {path}:{line_no}: {exc}") from exc
                if not isinstance(record, dict):
                    raise ValueError(f"Metric record must be a JSON object: {path}:{line_no}")
                record = dict(record)
                record["_metrics_path"] = str(path)
                record["_metrics_line"] = line_no
                record["_run_id"] = str(record.get("run_id") or run_id)
                records.append(record)
    return records


def _average_rank_positions(values: List[tuple[int, float]]) -> Dict[int, float]:
    """Return average 0-based rank positions for sorted numeric values."""

    indexed = sorted(values, key=lambda item: item[1])
    ranks: Dict[int, float] = {}
    start = 0
    while start < len(indexed):
        end = start + 1
        while end < len(indexed) and indexed[end][1] == indexed[start][1]:
            end += 1
        avg_rank = (start + end - 1) / 2.0
        for idx in range(start, end):
            ranks[indexed[idx][0]] = avg_rank
        start = end
    return ranks


def _rank_normalized_scores(
    records: Sequence[Mapping[str, object]],
    metric: MetricSpec,
) -> Dict[int, float]:
    values = [
        (idx, float(record[metric.key]))
        for idx, record in enumerate(records)
        if is_finite_number(record.get(metric.key))
    ]
    if not values:
        return {}
    if len(values) == 1:
        return {values[0][0]: 0.5}

    ranks = _average_rank_positions(values)
    denom = max(len(values) - 1, 1)
    normalized: Dict[int, float] = {}
    for idx, rank in ranks.items():
        frac = rank / denom
        normalized[idx] = frac if metric.direction == "min" else 1.0 - frac
    return normalized


def score_metric_records(
    records: Sequence[Mapping[str, object]],
    metric_specs: Sequence[MetricSpec] = DEFAULT_METRIC_SPECS,
) -> List[Dict[str, object]]:
    """Score records with rank-normalized weighted metrics.

    The returned list is sorted by ``path_critic_score`` ascending. Missing
    metrics are skipped for that record and reported in ``missing_metrics``.
    """

    if not records:
        return []

    per_metric_scores = {
        metric.key: _rank_normalized_scores(records, metric)
        for metric in metric_specs
    }

    scored: List[Dict[str, object]] = []
    for idx, record in enumerate(records):
        weighted_sum = 0.0
        total_weight = 0.0
        used: List[str] = []
        missing: List[str] = []
        components: Dict[str, float] = {}

        for metric in metric_specs:
            metric_scores = per_metric_scores[metric.key]
            if idx not in metric_scores:
                missing.append(metric.key)
                continue
            value = metric_scores[idx]
            components[metric.key] = value
            weighted_sum += metric.weight * value
            total_weight += metric.weight
            used.append(metric.key)

        output: MutableMapping[str, object] = dict(record)
        output["path_critic_score"] = weighted_sum / total_weight if total_weight > 0 else None
        output["path_critic_weight"] = total_weight
        output["path_critic_components"] = components
        output["metrics_used"] = used
        output["missing_metrics"] = missing
        scored.append(dict(output))

    return sorted(
        scored,
        key=lambda item: (
            float("inf")
            if item.get("path_critic_score") is None
            else float(item["path_critic_score"]),
            str(item.get("_run_id", "")),
            int(item.get("epoch", 10**9)) if is_finite_number(item.get("epoch")) else 10**9,
        ),
    )


def best_record_per_run(scored_records: Sequence[Mapping[str, object]]) -> List[Dict[str, object]]:
    """Keep the best-scoring record for each run."""

    best: Dict[str, Mapping[str, object]] = {}
    for record in scored_records:
        run_id = str(record.get("_run_id", "unknown"))
        score = record.get("path_critic_score")
        if score is None:
            continue
        if run_id not in best or float(score) < float(best[run_id]["path_critic_score"]):
            best[run_id] = record
    return [dict(item) for item in sorted(best.values(), key=lambda row: float(row["path_critic_score"]))]
