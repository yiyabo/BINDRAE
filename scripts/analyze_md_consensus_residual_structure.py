#!/usr/bin/env python3
"""Measure low-rank and spatial-localization structure in MD consensus residuals.

The analysis is performed on the endpoint-envelope-applied correction in metric
coordinates. It diagnoses whether a graph-coupled low-rank residual decoder is
a defensible inductive bias; it does not measure endpoint-conditioned model
performance.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence

import numpy as np


METRIC_GROUPS = ("rotation", "translation", "rigid", "chi", "combined")
BIOLOGICAL_MASKS = (
    "active_mask",
    "formed_contact_mask",
    "release_mask",
    "transient_contact_mask",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--system-list", type=Path, default=None)
    parser.add_argument("--min-t", type=float, default=0.05)
    parser.add_argument("--max-t", type=float, default=0.95)
    parser.add_argument("--ranks", default="1,2,4,8,16")
    return parser.parse_args()


def load_system_filter(path: Optional[Path]) -> Optional[set[str]]:
    if path is None:
        return None
    systems = {line.strip() for line in path.read_text().splitlines() if line.strip()}
    if not systems:
        raise ValueError(f"System list is empty: {path}")
    return systems


def parse_ranks(value: str) -> tuple[int, ...]:
    ranks = tuple(sorted({int(part.strip()) for part in value.split(",") if part.strip()}))
    if not ranks or ranks[0] < 1:
        raise ValueError("ranks must contain positive integers")
    return ranks


def load_manifest(cache_dir: Path) -> list[Dict[str, Any]]:
    path = cache_dir / "manifest.jsonl"
    if not path.is_file():
        raise FileNotFoundError(f"Missing immutable cache manifest: {path}")
    records = []
    seen = set()
    for line_number, line in enumerate(path.read_text().splitlines(), start=1):
        if not line.strip():
            continue
        record = json.loads(line)
        sample_id = str(record.get("sample_id", ""))
        relative_path = str(record.get("relative_path", ""))
        if not sample_id or not relative_path:
            raise ValueError(f"{path}:{line_number} is incomplete")
        if sample_id in seen:
            raise ValueError(f"Duplicate sample_id in manifest: {sample_id}")
        seen.add(sample_id)
        target_path = cache_dir / relative_path
        if not target_path.is_file():
            raise FileNotFoundError(target_path)
        copied = dict(record)
        copied["target_path"] = target_path
        records.append(copied)
    if not records:
        raise ValueError(f"No records in {path}")
    return records


def endpoint_envelope(t_values: np.ndarray, kind: str) -> np.ndarray:
    if kind == "sin2":
        envelope = np.sin(np.pi * t_values) ** 2
    elif kind == "poly":
        envelope = 4.0 * t_values * (1.0 - t_values)
    else:
        raise ValueError(f"Unsupported residual envelope: {kind!r}")
    return np.where((t_values > 0.0) & (t_values < 1.0), envelope, 0.0)


def metric_arrays(data: Mapping[str, np.ndarray]) -> Dict[str, np.ndarray]:
    rotation_scale = float(np.asarray(data["rotation_metric_scale"]).item())
    translation_scale = float(np.asarray(data["translation_metric_scale"]).item())
    chi_scale = float(np.asarray(data["chi_metric_scale"]).item())
    rotation = np.asarray(data["residual_rot"], dtype=np.float64) / rotation_scale
    translation = (
        np.asarray(data["residual_trans"], dtype=np.float64) / translation_scale
    )
    chi = np.asarray(data["residual_chi"], dtype=np.float64) / chi_scale
    rigid = np.concatenate([rotation, translation], axis=-1)
    combined = np.concatenate([rigid, chi], axis=-1)
    return {
        "rotation": rotation,
        "translation": translation,
        "rigid": rigid,
        "chi": chi,
        "combined": combined,
    }


def metric_component_masks(data: Mapping[str, np.ndarray]) -> Dict[str, np.ndarray]:
    node_mask = np.asarray(data["node_mask"], dtype=bool)
    chi_mask = np.asarray(data["chi_mask"], dtype=bool) & node_mask[:, None]
    rotation = np.broadcast_to(node_mask[:, None], (node_mask.size, 3))
    translation = np.broadcast_to(node_mask[:, None], (node_mask.size, 3))
    rigid = np.concatenate([rotation, translation], axis=-1)
    return {
        "rotation": rotation,
        "translation": translation,
        "rigid": rigid,
        "chi": chi_mask,
        "combined": np.concatenate([rigid, chi_mask], axis=-1),
    }


def low_rank_metrics(
    values: np.ndarray,
    weights: np.ndarray,
    envelope: np.ndarray,
    ranks: Sequence[int],
    *,
    eps: float = 1e-12,
) -> Dict[str, float]:
    """Return weighted SVD energy and localization diagnostics."""
    values = np.asarray(values, dtype=np.float64)
    weights = np.maximum(np.asarray(weights, dtype=np.float64), 0.0)
    envelope = np.asarray(envelope, dtype=np.float64)
    if values.shape != weights.shape or values.ndim != 3:
        raise ValueError("values and weights must have shape [time, residue, component]")
    if envelope.shape != (values.shape[0],):
        raise ValueError("envelope must match the time dimension")

    applied = values * envelope[:, None, None]
    weighted = applied * np.sqrt(weights)
    matrix = weighted.reshape(values.shape[0], -1)
    singular_values = np.linalg.svd(matrix, compute_uv=False)
    energy = np.square(singular_values)
    total_energy = float(energy.sum())
    result: Dict[str, float] = {
        "weighted_energy": total_energy,
        "time_points": int(values.shape[0]),
        "residues": int(values.shape[1]),
        "components": int(values.shape[2]),
    }
    if total_energy <= eps:
        for rank in ranks:
            result[f"rank_{rank}_explained_energy"] = 0.0
        result.update(
            effective_rank=0.0,
            rank_90=0,
            rank_95=0,
            effective_residue_count=0.0,
            effective_residue_fraction=0.0,
            active_residues=0,
        )
        return result

    cumulative = np.cumsum(energy) / total_energy
    for rank in ranks:
        result[f"rank_{rank}_explained_energy"] = float(
            cumulative[min(rank, cumulative.size) - 1]
        )
    normalized_energy = energy / total_energy
    positive = normalized_energy > 0.0
    result["effective_rank"] = float(
        np.exp(-np.sum(normalized_energy[positive] * np.log(normalized_energy[positive])))
    )
    result["rank_90"] = int(np.searchsorted(cumulative, 0.90) + 1)
    result["rank_95"] = int(np.searchsorted(cumulative, 0.95) + 1)

    residue_energy = np.square(weighted).sum(axis=(0, 2))
    active = residue_energy > eps
    active_count = int(np.count_nonzero(active))
    effective_count = float(
        np.square(residue_energy.sum()) / max(float(np.square(residue_energy).sum()), eps)
    )
    result["active_residues"] = active_count
    result["effective_residue_count"] = effective_count
    result["effective_residue_fraction"] = (
        effective_count / active_count if active_count else 0.0
    )

    time_energy = np.square(weighted).sum(axis=(1, 2))
    active_times = time_energy > eps
    active_time_count = int(np.count_nonzero(active_times))
    effective_time_count = float(
        np.square(time_energy.sum()) / max(float(np.square(time_energy).sum()), eps)
    )
    result["active_time_points"] = active_time_count
    result["effective_time_count"] = effective_time_count
    result["effective_time_fraction"] = (
        effective_time_count / active_time_count if active_time_count else 0.0
    )
    return result


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
        "max": float(array.max()),
    }


def analyze_target(
    sample_id: str,
    target_path: Path,
    *,
    min_t: float,
    max_t: float,
    ranks: Sequence[int],
) -> Dict[str, Any]:
    with np.load(target_path, allow_pickle=False) as source:
        data = {key: np.asarray(source[key]).copy() for key in source.files}
    if str(data["schema_version"].item()) != "md_phase_normal_v1":
        raise ValueError(f"{target_path} has an unsupported schema")
    if str(data["sample_id"].item()) != sample_id:
        raise ValueError(f"Manifest/cache sample mismatch for {sample_id}")

    t_values = np.asarray(data["t_values"], dtype=np.float64)
    interior = (t_values >= min_t) & (t_values <= max_t)
    if not interior.any():
        raise ValueError(f"No interior frames for {sample_id}")
    envelope_kind = str(data["residual_envelope"].item())
    envelope = endpoint_envelope(t_values[interior], envelope_kind)
    residual_valid = np.asarray(data["residual_valid_mask"], dtype=bool)[interior]
    residual_confidence = np.clip(
        np.asarray(data["residual_confidence"], dtype=np.float64)[interior],
        0.0,
        1.0,
    )
    base_weight = residual_confidence * residual_valid
    values = metric_arrays(data)
    component_masks = metric_component_masks(data)

    record: Dict[str, Any] = {
        "sample_id": sample_id,
        "target_path": str(target_path),
        "residual_envelope": envelope_kind,
    }
    combined_residue_energy = None
    for group in METRIC_GROUPS:
        group_values = values[group][interior]
        group_weight = base_weight[..., None] * component_masks[group][None, ...]
        metrics = low_rank_metrics(group_values, group_weight, envelope, ranks)
        record.update({f"{group}_{key}": value for key, value in metrics.items()})
        if group == "combined":
            applied = group_values * envelope[:, None, None]
            combined_residue_energy = np.sum(
                np.square(applied) * group_weight,
                axis=(0, 2),
            )

    if combined_residue_energy is not None:
        total = float(combined_residue_energy.sum())
        for mask_name in BIOLOGICAL_MASKS:
            if mask_name not in data:
                continue
            mask = np.asarray(data[mask_name], dtype=bool)
            if mask.shape != combined_residue_energy.shape:
                continue
            record[f"combined_energy_fraction_{mask_name}"] = (
                float(combined_residue_energy[mask].sum() / total)
                if total > 1e-12
                else 0.0
            )
            record[f"residue_fraction_{mask_name}"] = float(mask.mean())
    return record


def write_jsonl(path: Path, records: Iterable[Mapping[str, Any]]) -> None:
    path.write_text(
        "".join(json.dumps(record, sort_keys=True) + "\n" for record in records)
    )


def run_analysis(
    cache_dir: Path,
    output_dir: Path,
    *,
    system_filter: Optional[set[str]] = None,
    min_t: float = 0.05,
    max_t: float = 0.95,
    ranks: Sequence[int] = (1, 2, 4, 8, 16),
) -> Dict[str, Any]:
    if not 0.0 <= min_t < max_t <= 1.0:
        raise ValueError("Expected 0 <= min_t < max_t <= 1")
    manifest_records = load_manifest(cache_dir)
    available = {str(record["sample_id"]) for record in manifest_records}
    if system_filter is not None:
        missing = sorted(system_filter - available)
        if missing:
            raise ValueError(f"Requested systems absent from cache: {missing}")
        manifest_records = [
            record
            for record in manifest_records
            if str(record["sample_id"]) in system_filter
        ]
    records = []
    for index, manifest_record in enumerate(manifest_records, start=1):
        records.append(
            analyze_target(
                str(manifest_record["sample_id"]),
                Path(manifest_record["target_path"]),
                min_t=min_t,
                max_t=max_t,
                ranks=ranks,
            )
        )
        if index % 25 == 0 or index == len(manifest_records):
            print(f"Analyzed {index}/{len(manifest_records)} targets", flush=True)

    summary_metrics: Dict[str, Dict[str, Dict[str, float]]] = {}
    for group in METRIC_GROUPS:
        keys = [
            *(f"rank_{rank}_explained_energy" for rank in ranks),
            "effective_rank",
            "rank_90",
            "rank_95",
            "effective_residue_fraction",
            "effective_time_fraction",
        ]
        summary_metrics[group] = {
            key: summarize_values(
                float(record[f"{group}_{key}"])
                for record in records
                if f"{group}_{key}" in record
            )
            for key in keys
        }

    combined_rank4 = summary_metrics["combined"].get(
        "rank_4_explained_energy", {"p10": 0.0, "p50": 0.0}
    )
    low_rank_supported = (
        float(combined_rank4.get("p50", 0.0)) >= 0.85
        and float(combined_rank4.get("p10", 0.0)) >= 0.75
    )
    report: Dict[str, Any] = {
        "schema_version": "md_consensus_residual_structure_v1",
        "cache_dir": str(cache_dir),
        "systems_analyzed": len(records),
        "interior_t_range": [min_t, max_t],
        "ranks": list(ranks),
        "metric_summaries": summary_metrics,
        "low_rank_decoder_hint": {
            "supported": low_rank_supported,
            "criterion": "combined rank-4 median >= 0.85 and p10 >= 0.75",
            "recommended_rank": 4 if low_rank_supported else None,
            "warning": (
                "Low-rank target structure supports an inductive bias but does "
                "not prove that endpoint features can predict its coefficients."
            ),
        },
        "interpretation": {
            "higher_is_better": ["rank_*_explained_energy"],
            "lower_is_more_compressible": ["effective_rank", "rank_90", "rank_95"],
            "weighted_matrix": (
                "endpoint-envelope-applied metric residual multiplied by the "
                "square root of consensus supervision confidence"
            ),
            "scope": (
                "Target structure only. No validation/test model prediction is "
                "used, and this is not a path benchmark."
            ),
        },
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(output_dir / "systems.jsonl", records)
    (output_dir / "summary.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n"
    )
    return report


def main() -> None:
    args = parse_args()
    report = run_analysis(
        args.cache_dir,
        args.output_dir,
        system_filter=load_system_filter(args.system_list),
        min_t=args.min_t,
        max_t=args.max_t,
        ranks=parse_ranks(args.ranks),
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
