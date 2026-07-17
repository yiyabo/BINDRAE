#!/usr/bin/env python3
"""Measure how reproducible phase-normal targets are across MD replicas.

The deterministic explained-energy ratio is the weighted MSE upper bound for a
single target conditioned only on the shared endpoint system. A low ratio means
that replica-specific residuals cancel and motivates an explicit path latent;
a high ratio points instead to an optimization, masking, or capacity problem.
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np


REPLICA_SUFFIX = re.compile(r"^(?P<system>.+)__silver_r\d+$")
METRIC_GROUPS = ("rotation", "translation", "rigid", "chi", "combined")


@dataclass(frozen=True)
class ReplicaTargets:
    sample_id: str
    system_id: str
    path: Path
    t_values: np.ndarray
    node_mask: np.ndarray
    chi_mask: np.ndarray
    tau: np.ndarray
    phase_confidence: np.ndarray
    residual_valid: np.ndarray
    residual_confidence: np.ndarray
    metric_values: Mapping[str, np.ndarray]
    metric_weights: Mapping[str, np.ndarray]
    identity_hash: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--system-list",
        type=Path,
        default=None,
        help="Optional newline-delimited base system IDs to include.",
    )
    parser.add_argument("--min-t", type=float, default=0.05)
    parser.add_argument("--max-t", type=float, default=0.95)
    parser.add_argument("--min-phase-confidence", type=float, default=0.05)
    parser.add_argument("--min-residual-confidence", type=float, default=0.0)
    return parser.parse_args()


def system_id_from_sample_id(sample_id: str) -> str:
    match = REPLICA_SUFFIX.fullmatch(sample_id)
    if match is None:
        raise ValueError(
            f"Sample ID {sample_id!r} does not end in '__silver_r<digits>'"
        )
    return match.group("system")


def _load_system_filter(path: Optional[Path]) -> Optional[set[str]]:
    if path is None:
        return None
    values = {line.strip() for line in path.read_text().splitlines() if line.strip()}
    if not values:
        raise ValueError(f"System list is empty: {path}")
    return values


def _load_manifest_paths(cache_dir: Path) -> List[Tuple[str, Path]]:
    manifest_path = cache_dir / "manifest.jsonl"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Missing immutable cache manifest: {manifest_path}")
    rows: List[Tuple[str, Path]] = []
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
        path = cache_dir / relative_path
        if not path.is_file():
            raise FileNotFoundError(f"Missing target declared by manifest: {path}")
        rows.append((sample_id, path))
    if not rows:
        raise ValueError(f"No target records in {manifest_path}")
    return rows


def _scalar(data: Any, key: str, default: Any = None) -> Any:
    if key not in data.files:
        if default is not None:
            return default
        raise KeyError(key)
    return data[key].item()


def _finite(name: str, values: np.ndarray, path: Path) -> np.ndarray:
    values = np.asarray(values)
    if not np.isfinite(values).all():
        raise ValueError(f"{path} contains non-finite {name}")
    return values


def load_replica(expected_sample_id: str, path: Path) -> ReplicaTargets:
    with np.load(path, allow_pickle=False) as data:
        schema = str(_scalar(data, "schema_version"))
        if schema != "md_phase_normal_v1":
            raise ValueError(f"{path} schema_version={schema!r}")
        sample_id = str(_scalar(data, "sample_id"))
        if sample_id != expected_sample_id:
            raise ValueError(
                f"Manifest/cache sample mismatch: {expected_sample_id!r} != {sample_id!r}"
            )
        system_id = system_id_from_sample_id(sample_id)
        t_values = _finite("t_values", data["t_values"], path).astype(np.float64)
        node_mask = np.asarray(data["node_mask"], dtype=bool)
        chi_mask = np.asarray(data["chi_mask"], dtype=bool)
        tau = _finite("tau_target", data["tau_target"], path).astype(np.float64)
        phase_confidence = _finite(
            "phase_confidence", data["phase_confidence"], path
        ).astype(np.float64)
        residual_valid = np.asarray(data["residual_valid_mask"], dtype=bool)
        residual_confidence = _finite(
            "residual_confidence", data["residual_confidence"], path
        ).astype(np.float64)
        residual_rot = _finite("residual_rot", data["residual_rot"], path).astype(
            np.float64
        )
        residual_trans = _finite("residual_trans", data["residual_trans"], path).astype(
            np.float64
        )
        residual_chi = _finite("residual_chi", data["residual_chi"], path).astype(
            np.float64
        )
        rotation_scale = float(_scalar(data, "rotation_metric_scale", 1.0))
        translation_scale = float(_scalar(data, "translation_metric_scale", 1.0))
        chi_scale = float(_scalar(data, "chi_metric_scale", 1.0))
        identity_hash = str(_scalar(data, "residue_identity_hash", ""))

    if t_values.ndim != 1 or t_values.size < 3:
        raise ValueError(f"{path} has invalid t_values shape {t_values.shape}")
    expected_point_shape = (t_values.size, node_mask.size)
    if (
        tau.shape != expected_point_shape
        or phase_confidence.shape != expected_point_shape
    ):
        raise ValueError(f"{path} has invalid phase target shapes")
    if residual_valid.shape != expected_point_shape:
        raise ValueError(f"{path} has invalid residual_valid_mask shape")
    if residual_confidence.shape != expected_point_shape:
        raise ValueError(f"{path} has invalid residual_confidence shape")
    if chi_mask.shape != (node_mask.size, 4):
        raise ValueError(f"{path} has invalid chi_mask shape {chi_mask.shape}")
    if residual_rot.shape != (*expected_point_shape, 3):
        raise ValueError(f"{path} has invalid residual_rot shape")
    if residual_trans.shape != (*expected_point_shape, 3):
        raise ValueError(f"{path} has invalid residual_trans shape")
    if residual_chi.shape != (*expected_point_shape, 4):
        raise ValueError(f"{path} has invalid residual_chi shape")
    for name, scale in (
        ("rotation", rotation_scale),
        ("translation", translation_scale),
        ("chi", chi_scale),
    ):
        if not math.isfinite(scale) or scale <= 0.0:
            raise ValueError(f"{path} has invalid {name} metric scale {scale}")

    rotation = residual_rot / rotation_scale
    translation = residual_trans / translation_scale
    rigid = np.concatenate([rotation, translation], axis=-1)
    chi = residual_chi / chi_scale
    node_component = np.broadcast_to(
        node_mask[None, :, None], (*expected_point_shape, 1)
    )
    rigid_component_mask = np.broadcast_to(
        node_component, (*expected_point_shape, rigid.shape[-1])
    )
    rotation_component_mask = np.broadcast_to(node_component, rotation.shape)
    translation_component_mask = np.broadcast_to(node_component, translation.shape)
    chi_component_mask = (
        np.broadcast_to(chi_mask[None, :, :], chi.shape) & node_component
    )
    base_weight = np.clip(residual_confidence, 0.0, 1.0)
    base_weight = base_weight * residual_valid * node_mask[None, :]
    rigid_weight = base_weight[..., None] * rigid_component_mask
    rotation_weight = base_weight[..., None] * rotation_component_mask
    translation_weight = base_weight[..., None] * translation_component_mask
    chi_weight = base_weight[..., None] * chi_component_mask
    combined = np.concatenate([rigid, chi], axis=-1)
    combined_weight = np.concatenate([rigid_weight, chi_weight], axis=-1)
    return ReplicaTargets(
        sample_id=sample_id,
        system_id=system_id,
        path=path,
        t_values=t_values,
        node_mask=node_mask,
        chi_mask=chi_mask,
        tau=tau,
        phase_confidence=np.clip(phase_confidence, 0.0, 1.0),
        residual_valid=residual_valid,
        residual_confidence=np.clip(residual_confidence, 0.0, 1.0),
        metric_values={
            "rotation": rotation,
            "translation": translation,
            "rigid": rigid,
            "chi": chi,
            "combined": combined,
        },
        metric_weights={
            "rotation": rotation_weight,
            "translation": translation_weight,
            "rigid": rigid_weight,
            "chi": chi_weight,
            "combined": combined_weight,
        },
        identity_hash=identity_hash,
    )


def validate_replica_group(replicas: Sequence[ReplicaTargets]) -> None:
    if len(replicas) < 2:
        raise ValueError("Replica consistency requires at least two replicas")
    reference = replicas[0]
    for replica in replicas[1:]:
        if replica.system_id != reference.system_id:
            raise ValueError("Mixed systems in one replica group")
        if replica.t_values.shape != reference.t_values.shape or not np.allclose(
            replica.t_values, reference.t_values, rtol=0.0, atol=1e-7
        ):
            raise ValueError(f"Mismatched t grid in {replica.path}")
        if not np.array_equal(replica.node_mask, reference.node_mask):
            raise ValueError(f"Mismatched node mask in {replica.path}")
        if not np.array_equal(replica.chi_mask, reference.chi_mask):
            raise ValueError(f"Mismatched chi mask in {replica.path}")
        if reference.identity_hash and replica.identity_hash != reference.identity_hash:
            raise ValueError(f"Mismatched residue identity hash in {replica.path}")


def weighted_pair_metrics(
    first: np.ndarray,
    second: np.ndarray,
    first_weight: np.ndarray,
    second_weight: np.ndarray,
    *,
    eps: float = 1e-12,
) -> Dict[str, float]:
    if first.shape != second.shape or first.shape != first_weight.shape:
        raise ValueError("Pair values and weights must share a shape")
    if second_weight.shape != first.shape:
        raise ValueError("Pair values and weights must share a shape")
    joint_weight = np.sqrt(
        np.maximum(first_weight, 0.0) * np.maximum(second_weight, 0.0)
    )
    weight_sum = float(joint_weight.sum())
    support = int(np.count_nonzero(joint_weight > 0.0))
    if weight_sum <= eps:
        return {"support": support, "weight_sum": weight_sum}
    dot = float(np.sum(joint_weight * first * second))
    first_energy = float(np.sum(joint_weight * np.square(first)))
    second_energy = float(np.sum(joint_weight * np.square(second)))
    squared_error = float(np.sum(joint_weight * np.square(first - second)))
    rmse = math.sqrt(max(squared_error / weight_sum, 0.0))
    target_rms = math.sqrt(max(0.5 * (first_energy + second_energy) / weight_sum, 0.0))
    result = {
        "support": support,
        "weight_sum": weight_sum,
        "rmse": rmse,
        "target_rms": target_rms,
    }
    norm_product = math.sqrt(max(first_energy * second_energy, 0.0))
    if norm_product > eps:
        result["cosine"] = max(-1.0, min(1.0, dot / norm_product))
    if target_rms > eps:
        result["relative_rmse"] = rmse / target_rms
    return result


def deterministic_explained_energy(
    values: np.ndarray,
    weights: np.ndarray,
    *,
    eps: float = 1e-12,
) -> Dict[str, float]:
    """Return weighted signal fraction recoverable by one deterministic target."""
    if values.shape != weights.shape or values.ndim < 2:
        raise ValueError("Expected replica-first values and weights with equal shapes")
    valid_replicas = np.count_nonzero(weights > 0.0, axis=0)
    shared = valid_replicas >= 2
    shared_weights = np.where(shared[None, ...], np.maximum(weights, 0.0), 0.0)
    weight_per_cell = shared_weights.sum(axis=0)
    total_weight = float(weight_per_cell.sum())
    support = int(np.count_nonzero(shared))
    if total_weight <= eps:
        return {"support": support, "weight_sum": total_weight}
    mean = np.divide(
        (shared_weights * values).sum(axis=0),
        weight_per_cell,
        out=np.zeros_like(weight_per_cell, dtype=np.float64),
        where=weight_per_cell > 0.0,
    )
    total_energy = float(np.sum(shared_weights * np.square(values)))
    unexplained_energy = float(
        np.sum(shared_weights * np.square(values - mean[None, ...]))
    )
    result = {
        "support": support,
        "weight_sum": total_weight,
        "target_rms": math.sqrt(max(total_energy / total_weight, 0.0)),
    }
    if total_energy > eps:
        unexplained_ratio = max(0.0, min(1.0, unexplained_energy / total_energy))
        result["explained_energy"] = 1.0 - unexplained_ratio
        result["variance_to_energy"] = unexplained_ratio
    return result


def _phase_weights(
    replica: ReplicaTargets,
    interior: np.ndarray,
    min_confidence: float,
) -> np.ndarray:
    confidence = replica.phase_confidence
    return confidence * (
        interior[:, None] & replica.node_mask[None, :] & (confidence >= min_confidence)
    )


def _residual_weights(
    replica: ReplicaTargets,
    group: str,
    interior: np.ndarray,
    min_confidence: float,
) -> np.ndarray:
    weights = replica.metric_weights[group]
    return weights * (
        interior[:, None, None]
        & (replica.residual_confidence[..., None] >= min_confidence)
    )


def _prefix_metrics(prefix: str, metrics: Mapping[str, float]) -> Dict[str, float]:
    return {f"{prefix}_{name}": value for name, value in metrics.items()}


def _mean_present(records: Sequence[Mapping[str, Any]], key: str) -> Optional[float]:
    values = [float(record[key]) for record in records if key in record]
    return float(np.mean(values)) if values else None


def analyze_system(
    replicas: Sequence[ReplicaTargets],
    *,
    min_t: float,
    max_t: float,
    min_phase_confidence: float,
    min_residual_confidence: float,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    validate_replica_group(replicas)
    replicas = sorted(replicas, key=lambda replica: replica.sample_id)
    interior = (replicas[0].t_values >= min_t) & (replicas[0].t_values <= max_t)
    if not interior.any():
        raise ValueError(f"No t-grid points remain for {replicas[0].system_id}")

    phase_values = np.stack(
        [replica.tau - replica.t_values[:, None] for replica in replicas], axis=0
    )
    phase_weights = np.stack(
        [
            _phase_weights(replica, interior, min_phase_confidence)
            for replica in replicas
        ],
        axis=0,
    )
    system_record: Dict[str, Any] = {
        "system_id": replicas[0].system_id,
        "replicas": len(replicas),
        "sample_ids": [replica.sample_id for replica in replicas],
    }
    system_record.update(
        _prefix_metrics(
            "phase_offset_deterministic",
            deterministic_explained_energy(phase_values, phase_weights),
        )
    )

    residual_values: Dict[str, np.ndarray] = {}
    residual_weights: Dict[str, np.ndarray] = {}
    for group in METRIC_GROUPS:
        residual_values[group] = np.stack(
            [replica.metric_values[group] for replica in replicas], axis=0
        )
        residual_weights[group] = np.stack(
            [
                _residual_weights(replica, group, interior, min_residual_confidence)
                for replica in replicas
            ],
            axis=0,
        )
        system_record.update(
            _prefix_metrics(
                f"residual_{group}_deterministic",
                deterministic_explained_energy(
                    residual_values[group], residual_weights[group]
                ),
            )
        )

    pair_records: List[Dict[str, Any]] = []
    for first_index, second_index in itertools.combinations(range(len(replicas)), 2):
        first = replicas[first_index]
        second = replicas[second_index]
        pair: Dict[str, Any] = {
            "system_id": first.system_id,
            "sample_id_a": first.sample_id,
            "sample_id_b": second.sample_id,
        }
        phase_pair = weighted_pair_metrics(
            phase_values[first_index],
            phase_values[second_index],
            phase_weights[first_index],
            phase_weights[second_index],
        )
        pair.update(_prefix_metrics("phase_offset", phase_pair))
        raw_tau_pair = weighted_pair_metrics(
            first.tau,
            second.tau,
            phase_weights[first_index],
            phase_weights[second_index],
        )
        pair["phase_tau_mae"] = raw_tau_pair.get("rmse", 0.0)
        joint_phase_weight = np.sqrt(
            phase_weights[first_index] * phase_weights[second_index]
        )
        phase_weight_sum = float(joint_phase_weight.sum())
        if phase_weight_sum > 0.0:
            pair["phase_tau_mae"] = float(
                np.sum(joint_phase_weight * np.abs(first.tau - second.tau))
                / phase_weight_sum
            )
        for group in METRIC_GROUPS:
            pair.update(
                _prefix_metrics(
                    f"residual_{group}",
                    weighted_pair_metrics(
                        residual_values[group][first_index],
                        residual_values[group][second_index],
                        residual_weights[group][first_index],
                        residual_weights[group][second_index],
                    ),
                )
            )
        pair_records.append(pair)

    pair_mean_keys = (
        "phase_tau_mae",
        "phase_offset_cosine",
        "phase_offset_relative_rmse",
        "residual_rotation_cosine",
        "residual_rotation_relative_rmse",
        "residual_translation_cosine",
        "residual_translation_relative_rmse",
        "residual_rigid_cosine",
        "residual_rigid_relative_rmse",
        "residual_chi_cosine",
        "residual_chi_relative_rmse",
        "residual_combined_cosine",
        "residual_combined_relative_rmse",
    )
    for key in pair_mean_keys:
        value = _mean_present(pair_records, key)
        if value is not None:
            system_record[f"pair_mean_{key}"] = value
    return system_record, pair_records


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


def summarize_records(
    records: Sequence[Mapping[str, Any]], keys: Sequence[str]
) -> Dict[str, Dict[str, float]]:
    return {
        key: summarize_values(float(record[key]) for record in records if key in record)
        for key in keys
    }


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
    min_phase_confidence: float = 0.05,
    min_residual_confidence: float = 0.0,
) -> Dict[str, Any]:
    if not 0.0 <= min_t < max_t <= 1.0:
        raise ValueError("Expected 0 <= min_t < max_t <= 1")
    grouped_paths: Dict[str, List[Tuple[str, Path]]] = defaultdict(list)
    for sample_id, path in _load_manifest_paths(cache_dir):
        system_id = system_id_from_sample_id(sample_id)
        if system_filter is None or system_id in system_filter:
            grouped_paths[system_id].append((sample_id, path))
    if system_filter is not None:
        missing = sorted(system_filter - set(grouped_paths))
        if missing:
            raise ValueError(f"Requested systems absent from cache: {missing}")
    eligible = {
        system_id: paths
        for system_id, paths in grouped_paths.items()
        if len(paths) >= 2
    }
    if not eligible:
        raise ValueError("No systems have at least two replicas")

    system_records: List[Dict[str, Any]] = []
    pair_records: List[Dict[str, Any]] = []
    for index, system_id in enumerate(sorted(eligible), start=1):
        replicas = [
            load_replica(sample_id, path) for sample_id, path in eligible[system_id]
        ]
        system_record, system_pairs = analyze_system(
            replicas,
            min_t=min_t,
            max_t=max_t,
            min_phase_confidence=min_phase_confidence,
            min_residual_confidence=min_residual_confidence,
        )
        system_records.append(system_record)
        pair_records.extend(system_pairs)
        if index % 25 == 0 or index == len(eligible):
            print(f"Analyzed {index}/{len(eligible)} systems", flush=True)

    system_keys = (
        "phase_offset_deterministic_explained_energy",
        "phase_offset_deterministic_variance_to_energy",
        "pair_mean_phase_tau_mae",
        "pair_mean_phase_offset_cosine",
        "pair_mean_phase_offset_relative_rmse",
        "residual_rotation_deterministic_explained_energy",
        "residual_rotation_deterministic_variance_to_energy",
        "residual_translation_deterministic_explained_energy",
        "residual_translation_deterministic_variance_to_energy",
        "residual_rigid_deterministic_explained_energy",
        "residual_rigid_deterministic_variance_to_energy",
        "residual_chi_deterministic_explained_energy",
        "residual_chi_deterministic_variance_to_energy",
        "residual_combined_deterministic_explained_energy",
        "residual_combined_deterministic_variance_to_energy",
        "pair_mean_residual_rotation_cosine",
        "pair_mean_residual_rotation_relative_rmse",
        "pair_mean_residual_translation_cosine",
        "pair_mean_residual_translation_relative_rmse",
        "pair_mean_residual_rigid_cosine",
        "pair_mean_residual_rigid_relative_rmse",
        "pair_mean_residual_chi_cosine",
        "pair_mean_residual_chi_relative_rmse",
        "pair_mean_residual_combined_cosine",
        "pair_mean_residual_combined_relative_rmse",
    )
    pair_keys = (
        "phase_tau_mae",
        "phase_offset_cosine",
        "phase_offset_relative_rmse",
        "residual_rotation_cosine",
        "residual_rotation_relative_rmse",
        "residual_translation_cosine",
        "residual_translation_relative_rmse",
        "residual_rigid_cosine",
        "residual_rigid_relative_rmse",
        "residual_chi_cosine",
        "residual_chi_relative_rmse",
        "residual_combined_cosine",
        "residual_combined_relative_rmse",
    )
    report: Dict[str, Any] = {
        "schema_version": "md_replica_consistency_v1",
        "cache_dir": str(cache_dir),
        "systems_selected": len(grouped_paths),
        "systems_analyzed": len(system_records),
        "systems_single_replica_skipped": len(grouped_paths) - len(eligible),
        "replicas_analyzed": int(sum(record["replicas"] for record in system_records)),
        "replica_pairs_analyzed": len(pair_records),
        "interior_t_range": [min_t, max_t],
        "min_phase_confidence": min_phase_confidence,
        "min_residual_confidence": min_residual_confidence,
        "interpretation": {
            "higher_is_better": [
                "*_cosine",
                "*_deterministic_explained_energy",
            ],
            "lower_is_better": [
                "phase_tau_mae",
                "*_relative_rmse",
                "*_deterministic_variance_to_energy",
            ],
            "deterministic_explained_energy": (
                "Fraction of weighted target energy retained by the per-cell "
                "replica mean; this is an oracle upper bound for a deterministic "
                "endpoint-conditioned target under weighted MSE."
            ),
        },
        "system_macro": summarize_records(system_records, system_keys),
        "pair_macro": summarize_records(pair_records, pair_keys),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(output_dir / "systems.jsonl", system_records)
    write_jsonl(output_dir / "pairs.jsonl", pair_records)
    (output_dir / "summary.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n"
    )
    return report


def main() -> None:
    args = parse_args()
    report = run_analysis(
        args.cache_dir,
        args.output_dir,
        system_filter=_load_system_filter(args.system_list),
        min_t=args.min_t,
        max_t=args.max_t,
        min_phase_confidence=args.min_phase_confidence,
        min_residual_confidence=args.min_residual_confidence,
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
