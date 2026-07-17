#!/usr/bin/env python3
"""Build deterministic consensus targets from replicated MD silver paths.

The output keeps only the endpoint-conditioned component that is reproducible
across replicas. Replica disagreement is folded into the target confidence;
the remaining variation is intentionally left for a future stochastic latent.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Sequence, Tuple

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.analyze_md_replica_consistency import system_id_from_sample_id


STATIC_KEYS = (
    "schema_version",
    "phase_target_mode",
    "normal_projection_mode",
    "bridge_mode",
    "residual_envelope",
    "rotation_metric_scale",
    "translation_metric_scale",
    "chi_metric_scale",
    "n_residues",
    "mapped_residues",
    "mapping_fraction",
    "mapping_method",
    "residue_alignment_version",
    "residue_identity_hash",
    "t_values",
    "node_mask",
    "chi_mask",
    "active_mask",
    "motion_active",
    "endpoint_motion_metric_norm",
    "residue_chain",
    "residue_number",
    "residue_name",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-cache", type=Path, required=True)
    parser.add_argument("--output-cache", type=Path, required=True)
    parser.add_argument("--min-replicas", type=int, default=2)
    parser.add_argument("--min-support-fraction", type=float, default=0.5)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def _load_manifest(cache_dir: Path) -> list[Tuple[str, Path]]:
    manifest = cache_dir / "manifest.jsonl"
    if not manifest.is_file():
        raise FileNotFoundError(f"Missing cache manifest: {manifest}")
    rows = []
    seen = set()
    for line_number, line in enumerate(manifest.read_text().splitlines(), start=1):
        if not line.strip():
            continue
        record = json.loads(line)
        sample_id = str(record.get("sample_id", ""))
        relative_path = str(record.get("relative_path", ""))
        if not sample_id or not relative_path:
            raise ValueError(f"{manifest}:{line_number} is incomplete")
        if sample_id in seen:
            raise ValueError(f"Duplicate sample ID in manifest: {sample_id}")
        seen.add(sample_id)
        path = cache_dir / relative_path
        if not path.is_file():
            raise FileNotFoundError(path)
        rows.append((sample_id, path))
    if not rows:
        raise ValueError(f"No cache entries in {manifest}")
    return rows


def _load_payload(path: Path) -> Dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as data:
        return {key: np.asarray(data[key]).copy() for key in data.files}


def _validate_group(
    system_id: str,
    sample_ids: Sequence[str],
    payloads: Sequence[Mapping[str, np.ndarray]],
) -> None:
    reference = payloads[0]
    if str(reference["schema_version"].item()) != "md_phase_normal_v1":
        raise ValueError(f"{system_id} has an unsupported schema")
    for sample_id, payload in zip(sample_ids, payloads):
        if str(payload["sample_id"].item()) != sample_id:
            raise ValueError(f"Cache sample mismatch for {sample_id}")
        for key in STATIC_KEYS:
            if key not in payload or key not in reference:
                raise KeyError(f"{system_id} is missing static key {key!r}")
            if not np.array_equal(payload[key], reference[key]):
                raise ValueError(f"{system_id} replicas disagree on {key!r}")


def weighted_vector_consensus(
    values: np.ndarray,
    weights: np.ndarray,
    *,
    component_mask: np.ndarray | None = None,
    min_support_fraction: float = 0.5,
    min_support_count: int = 2,
    eps: float = 1e-12,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return mean, confidence, valid mask, and explained-energy agreement."""
    values = np.asarray(values, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    if values.ndim < 2 or values.shape[:-1] != weights.shape:
        raise ValueError("values must have shape [replica, ..., component]")
    if not 0.0 < min_support_fraction <= 1.0:
        raise ValueError("min_support_fraction must be in (0, 1]")

    weights = np.clip(weights, 0.0, 1.0)
    support = np.count_nonzero(weights > 0.0, axis=0)
    required = max(min_support_count, math.ceil(values.shape[0] * min_support_fraction))
    weight_sum = weights.sum(axis=0)
    safe_weight = np.maximum(weight_sum, eps)
    mean = (values * weights[..., None]).sum(axis=0) / safe_weight[..., None]

    if component_mask is None:
        component_mask = np.ones(values.shape[1:], dtype=bool)
    component_mask = np.asarray(component_mask, dtype=bool)
    if component_mask.shape != values.shape[1:]:
        component_mask = np.broadcast_to(component_mask, values.shape[1:])
    masked_values = np.where(component_mask[None, ...], values, 0.0)
    masked_mean = np.where(component_mask, mean, 0.0)
    replica_energy = np.square(masked_values).sum(axis=-1)
    mean_replica_energy = (
        replica_energy * weights
    ).sum(axis=0) / safe_weight
    consensus_energy = np.square(masked_mean).sum(axis=-1)
    agreement = np.divide(
        consensus_energy,
        mean_replica_energy,
        out=np.ones_like(consensus_energy),
        where=mean_replica_energy > eps,
    )
    agreement = np.clip(agreement, 0.0, 1.0)

    valid = (support >= required) & (weight_sum > eps)
    source_confidence = weight_sum / np.maximum(support, 1)
    support_fraction = support / float(values.shape[0])
    confidence = np.clip(source_confidence * support_fraction * agreement, 0.0, 1.0)
    confidence = np.where(valid, confidence, 0.0)
    mean = np.where(valid[..., None], mean, 0.0)
    return mean, confidence, valid, agreement


def _weighted_scalar_mean(values: np.ndarray, weights: np.ndarray) -> np.ndarray:
    finite = np.isfinite(values)
    effective_weight = np.where(finite, weights, 0.0)
    safe_values = np.where(finite, values, 0.0)
    weight_sum = effective_weight.sum(axis=0)
    return np.divide(
        (safe_values * effective_weight).sum(axis=0),
        weight_sum,
        out=np.zeros_like(values[0], dtype=np.float64),
        where=weight_sum > 1e-12,
    )


def _endpoint_envelope(t_values: np.ndarray, kind: str) -> np.ndarray:
    if kind == "sin2":
        values = np.sin(np.pi * t_values) ** 2
    elif kind == "poly":
        values = 4.0 * t_values * (1.0 - t_values)
    else:
        raise ValueError(f"Unsupported residual envelope: {kind!r}")
    return np.where((t_values > 0.0) & (t_values < 1.0), values, 0.0)


def build_consensus_payload(
    system_id: str,
    sample_ids: Sequence[str],
    payloads: Sequence[Mapping[str, np.ndarray]],
    *,
    min_support_fraction: float,
    min_support_count: int,
) -> Tuple[Dict[str, np.ndarray], Dict[str, float]]:
    _validate_group(system_id, sample_ids, payloads)
    reference = payloads[0]
    n_replica = len(payloads)
    t_values = np.asarray(reference["t_values"], dtype=np.float64)
    node_mask = np.asarray(reference["node_mask"], dtype=bool)
    chi_mask = np.asarray(reference["chi_mask"], dtype=bool)

    phase_conf = np.stack(
        [np.asarray(p["phase_confidence"], dtype=np.float64) for p in payloads]
    )
    phase_weight = phase_conf * node_mask[None, None, :]
    tau = np.stack([np.asarray(p["tau_target"], dtype=np.float64) for p in payloads])
    phase_offset = tau - t_values[None, :, None]
    mean_offset, consensus_phase_conf, _, phase_agreement = weighted_vector_consensus(
        phase_offset[..., None],
        phase_weight,
        min_support_fraction=min_support_fraction,
        min_support_count=min_support_count,
    )
    consensus_tau = np.clip(t_values[:, None] + mean_offset[..., 0], 0.0, 1.0)
    consensus_tau = np.maximum.accumulate(consensus_tau, axis=0)

    residual_conf = np.stack(
        [np.asarray(p["residual_confidence"], dtype=np.float64) for p in payloads]
    )
    residual_valid = np.stack(
        [np.asarray(p["residual_valid_mask"], dtype=bool) for p in payloads]
    )
    residual_weight = residual_conf * residual_valid * node_mask[None, None, :]
    rotation_scale = float(reference["rotation_metric_scale"].item())
    translation_scale = float(reference["translation_metric_scale"].item())
    chi_scale = float(reference["chi_metric_scale"].item())
    rot = np.stack([np.asarray(p["residual_rot"], dtype=np.float64) for p in payloads])
    trans = np.stack(
        [np.asarray(p["residual_trans"], dtype=np.float64) for p in payloads]
    )
    chi = np.stack([np.asarray(p["residual_chi"], dtype=np.float64) for p in payloads])
    metric_values = np.concatenate(
        [rot / rotation_scale, trans / translation_scale, chi / chi_scale], axis=-1
    )
    metric_mask = np.concatenate(
        [
            np.ones((*rot.shape[1:-1], 6), dtype=bool),
            np.broadcast_to(chi_mask[None, :, :], chi.shape[1:]),
        ],
        axis=-1,
    )
    mean_metric, consensus_residual_conf, consensus_valid, residual_agreement = (
        weighted_vector_consensus(
            metric_values,
            residual_weight,
            component_mask=metric_mask,
            min_support_fraction=min_support_fraction,
            min_support_count=min_support_count,
        )
    )
    mean_rot = mean_metric[..., :3] * rotation_scale
    mean_trans = mean_metric[..., 3:6] * translation_scale
    mean_chi = mean_metric[..., 6:] * chi_scale

    result = {key: np.asarray(value).copy() for key, value in reference.items()}
    envelope = _endpoint_envelope(
        t_values, str(reference["residual_envelope"].item())
    )
    result.update(
        source=np.array("silver_replica_consensus"),
        sample_id=np.array(system_id),
        transition_id=np.array(f"ahoj:{system_id}:silver-consensus-r{n_replica:02d}"),
        evidence_tier=np.array("silver_replica_consensus"),
        tau_target=consensus_tau.astype(np.float32),
        phase_confidence=consensus_phase_conf.astype(np.float32),
        residual_rot=mean_rot.astype(np.float32),
        residual_trans=mean_trans.astype(np.float32),
        residual_chi=mean_chi.astype(np.float32),
        residual_valid_mask=consensus_valid,
        residual_confidence=consensus_residual_conf.astype(np.float32),
        normal_residual_metric_norm=(
            np.linalg.norm(np.where(metric_mask, mean_metric, 0.0), axis=-1)
            * envelope[:, None]
        ).astype(np.float32),
        raw_monotonicity_violations=np.max(
            np.stack([p["raw_monotonicity_violations"] for p in payloads]), axis=0
        ).astype(np.int32),
    )

    result["w_res"] = np.mean(
        np.stack([p["w_res"] for p in payloads]).astype(np.float64), axis=0
    ).astype(np.float32)
    for key in (
        "pocket_mask",
        "approach_mask",
        "formed_contact_mask",
        "release_mask",
        "transient_contact_mask",
    ):
        frequency = np.mean(
            np.stack([p[key] for p in payloads]).astype(np.float64), axis=0
        )
        result[key] = frequency >= min_support_fraction

    phase_stack = np.stack([p["phase_confidence"] for p in payloads]).astype(np.float64)
    residual_stack = np.stack([p["residual_confidence"] for p in payloads]).astype(
        np.float64
    )
    for key, weights in (
        ("projection_cost", phase_stack),
        ("identity_cost", phase_stack),
        ("contact_distance_angstrom", phase_stack),
        ("raw_parallel_cos_abs", residual_stack),
        ("projected_parallel_cos_abs", residual_stack),
    ):
        values = np.stack([np.asarray(p[key], dtype=np.float64) for p in payloads])
        result[key] = _weighted_scalar_mean(values, weights).astype(np.float32)
    event_progress = np.stack(
        [p["contact_event_progress"] for p in payloads]
    ).astype(np.float64)
    event_finite = np.isfinite(event_progress)
    result["contact_event_progress"] = np.divide(
        np.where(event_finite, event_progress, 0.0).sum(axis=0),
        event_finite.sum(axis=0),
        out=np.zeros_like(event_progress[0]),
        where=event_finite.sum(axis=0) > 0,
    ).astype(np.float32)

    diagnostics = {
        "replicas": float(n_replica),
        "phase_agreement_mean": float(
            phase_agreement[node_mask[None, :].repeat(t_values.size, axis=0)].mean()
        ),
        "residual_agreement_mean": float(
            residual_agreement[consensus_valid].mean()
            if np.any(consensus_valid)
            else 0.0
        ),
        "residual_valid_fraction": float(consensus_valid.mean()),
        "residual_confidence_mean": float(
            consensus_residual_conf[consensus_valid].mean()
            if np.any(consensus_valid)
            else 0.0
        ),
    }
    return result, diagnostics


def _atomic_savez(path: Path, payload: Mapping[str, np.ndarray]) -> None:
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **payload)
    os.replace(temporary, path)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def run(args: argparse.Namespace) -> Dict[str, Any]:
    if args.min_replicas < 2:
        raise ValueError("min_replicas must be at least 2")
    if args.output_cache.exists() and any(args.output_cache.iterdir()):
        if not args.overwrite:
            raise FileExistsError(f"Output cache is not empty: {args.output_cache}")
    args.output_cache.mkdir(parents=True, exist_ok=True)

    grouped: Dict[str, list[Tuple[str, Path]]] = defaultdict(list)
    for sample_id, path in _load_manifest(args.input_cache):
        grouped[system_id_from_sample_id(sample_id)].append((sample_id, path))

    manifest_rows = []
    system_rows = []
    skipped = []
    for system_id in sorted(grouped):
        records = sorted(grouped[system_id])
        if len(records) < args.min_replicas:
            skipped.append({"system_id": system_id, "replicas": len(records)})
            continue
        sample_ids = [sample_id for sample_id, _ in records]
        payloads = [_load_payload(path) for _, path in records]
        payload, diagnostics = build_consensus_payload(
            system_id,
            sample_ids,
            payloads,
            min_support_fraction=args.min_support_fraction,
            min_support_count=args.min_replicas,
        )
        output_path = args.output_cache / f"{system_id}.npz"
        _atomic_savez(output_path, payload)
        manifest_rows.append(
            {
                "sample_id": system_id,
                "relative_path": output_path.name,
                "sha256": _sha256(output_path),
                "replica_ids": sample_ids,
                "n_replicas": len(sample_ids),
            }
        )
        system_rows.append({"system_id": system_id, **diagnostics})

    manifest_path = args.output_cache / "manifest.jsonl"
    manifest_path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in manifest_rows)
    )
    summary = {
        "schema_version": "md_phase_normal_consensus_v1",
        "input_cache": str(args.input_cache),
        "output_cache": str(args.output_cache),
        "input_systems": len(grouped),
        "consensus_systems": len(system_rows),
        "skipped_systems": skipped,
        "min_replicas": args.min_replicas,
        "min_support_fraction": args.min_support_fraction,
        "mean_phase_agreement": float(
            np.mean([row["phase_agreement_mean"] for row in system_rows])
        ),
        "mean_residual_agreement": float(
            np.mean([row["residual_agreement_mean"] for row in system_rows])
        ),
    }
    _write_json(args.output_cache / "summary.json", summary)
    (args.output_cache / "systems.jsonl").write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in system_rows)
    )
    return summary


def main() -> None:
    summary = run(parse_args())
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
