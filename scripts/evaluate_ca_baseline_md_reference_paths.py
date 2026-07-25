#!/usr/bin/env python3
"""Evaluate external C-alpha path baselines against held-out MD references.

The primary metric is the same frame-origin translation error reported by
``evaluate_stage2_md_reference_paths.py``. Rotation, chi, and atomistic contact
metrics are intentionally omitted because CA-only methods do not predict them.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import numpy as np
import torch
from tqdm import tqdm


PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import scripts.evaluate_ebdims2_ca_paths as ca_eval  # noqa: E402
import scripts.evaluate_stage2_md_reference_paths as md_eval  # noqa: E402
import scripts.evaluate_stage2_transition_paths as stage2_eval  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate CA-only baseline paths against held-out MD references"
    )
    parser.add_argument("--data_dir", default="processed_data/triplets")
    parser.add_argument("--split", default="train")
    parser.add_argument("--valid_samples_file", required=True)
    parser.add_argument("--run_manifest", required=True)
    parser.add_argument("--md_reference_cache_dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--per_sample_output", default=None)
    parser.add_argument("--method", required=True)
    parser.add_argument("--n_path_steps", type=int, default=20)
    parser.add_argument("--min_frames", type=int, default=4)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--md_min_phase_confidence", type=float, default=0.05)
    parser.add_argument("--md_event_tie_tolerance", type=float, default=0.02)
    parser.add_argument(
        "--status_allow",
        default="success,partial_success,timeout_partial,success_too_few_frames,skipped_existing",
    )
    parser.add_argument(
        "--frame_offset",
        choices=["apo_mass_com", "none"],
        default="none",
    )
    parser.add_argument(
        "--append_holo_endpoint",
        action="store_true",
        help="Diagnostic only: append the exact holo endpoint to the method output.",
    )
    parser.add_argument("--trust_prechecked_samples", action="store_true")
    parser.add_argument(
        "--allow_missing_systems",
        action="store_true",
        help="Evaluate successful systems but record missing baseline outputs explicitly.",
    )
    return parser.parse_args()


def read_manifest(path: Path, allowed: set[str]) -> Dict[str, Dict[str, object]]:
    rows: Dict[str, Dict[str, object]] = {}
    for row in ca_eval.read_jsonl(path):
        if str(row.get("status")) not in allowed:
            continue
        sample_id = str(row["sample_id"])
        if sample_id in rows:
            raise ValueError(f"Duplicate sample_id in run manifest: {sample_id}")
        rows[sample_id] = row
    if not rows:
        raise ValueError(f"No allowed baseline records in {path}")
    return rows


def interpolate_path(path: np.ndarray, target_times: np.ndarray) -> np.ndarray:
    path = np.asarray(path, dtype=np.float64)
    if path.ndim != 3 or path.shape[-1] != 3 or path.shape[0] < 2:
        raise ValueError(f"Expected path [time,residue,3], got {path.shape}")
    source_times = np.linspace(0.0, 1.0, path.shape[0], dtype=np.float64)
    return md_eval.interpolate_time_series(source_times, path, target_times)


def inverse_smoothstep(values: np.ndarray, iterations: int = 32) -> np.ndarray:
    """Invert 3*t^2-2*t^3 on [0, 1] by vectorized bisection."""
    values = np.clip(np.asarray(values, dtype=np.float64), 0.0, 1.0)
    lower = np.zeros_like(values)
    upper = np.ones_like(values)
    for _ in range(iterations):
        middle = 0.5 * (lower + upper)
        mapped = 3.0 * middle * middle - 2.0 * middle * middle * middle
        lower = np.where(mapped < values, middle, lower)
        upper = np.where(mapped >= values, middle, upper)
    return 0.5 * (lower + upper)


def projected_tau_from_ca(
    path: np.ndarray,
    apo_ca: np.ndarray,
    holo_ca: np.ndarray,
    eps: float = 1.0e-8,
) -> np.ndarray:
    """Recover bridge phase from each residue's endpoint displacement."""
    displacement = holo_ca - apo_ca
    denominator = np.sum(displacement * displacement, axis=-1)
    numerator = np.sum(
        (path - apo_ca[None, ...]) * displacement[None, ...], axis=-1
    )
    progress = np.divide(
        numerator,
        denominator[None, ...],
        out=np.zeros_like(numerator),
        where=denominator[None, ...] > eps,
    )
    tau = inverse_smoothstep(progress)
    tau[:, denominator <= eps] = np.linspace(0.0, 1.0, path.shape[0])[:, None]
    tau[0] = 0.0
    tau[-1] = 1.0
    return tau


def load_ca_path(
    row: Dict[str, object],
    apo_ca: np.ndarray,
    holo_ca: np.ndarray,
    frame_offset: str,
    append_holo_endpoint: bool,
) -> np.ndarray:
    run_dir = Path(str(row["run_dir"]))
    frames = ca_eval.list_frames(run_dir)
    apo_chain = str(row.get("apo_chain") or "")
    if not frames:
        raise ValueError(f"No path frames under {run_dir}")

    offset = np.zeros(3, dtype=np.float64)
    if frame_offset == "apo_mass_com":
        parsed_apo, _, masses = ca_eval.parse_ca_records(Path(str(row["apo_pdb"])), apo_chain)
        if parsed_apo.shape != apo_ca.shape or masses.shape[0] != apo_ca.shape[0]:
            raise ValueError(
                f"Apo CA mapping mismatch for {row['sample_id']}: "
                f"parsed={parsed_apo.shape}, expected={apo_ca.shape}"
            )
        offset = (parsed_apo * masses[:, None]).sum(axis=0) / masses.sum()

    path = [apo_ca]
    for frame in frames:
        frame_ca, _, _ = ca_eval.parse_ca_records(frame, apo_chain)
        if frame_ca.shape != apo_ca.shape:
            raise ValueError(
                f"Frame CA mapping mismatch for {row['sample_id']} in {frame}: "
                f"parsed={frame_ca.shape}, expected={apo_ca.shape}"
            )
        path.append(frame_ca + offset)
    if append_holo_endpoint and not np.allclose(path[-1], holo_ca, atol=1.0e-6):
        path.append(holo_ca)
    return np.stack(path, axis=0)


def ca_path_metrics(
    predicted_ca: np.ndarray,
    target_rigids,
    arrays: Dict[str, np.ndarray],
    target_times: np.ndarray,
    apo_ca: np.ndarray,
    holo_ca: np.ndarray,
    peptide_bond_mask: np.ndarray,
) -> Dict[str, float]:
    target_ca = np.stack(
        [rigid.get_trans()[0].detach().cpu().numpy() for rigid in target_rigids]
    )
    errors = np.linalg.norm(predicted_ca - target_ca, axis=-1)
    valid = arrays["residual_valid"] & arrays["node_mask"][None, :]
    weights = arrays["residual_confidence"] * valid.astype(np.float64)
    interior = np.zeros(predicted_ca.shape[0], dtype=bool)
    interior[1:-1] = True
    weights = weights * interior[:, None]

    endpoint_error = np.linalg.norm(predicted_ca[-1] - holo_ca, axis=-1)
    endpoint_weights = arrays["node_mask"].astype(np.float64)
    step = np.linalg.norm(np.diff(predicted_ca, axis=0), axis=-1)
    acceleration = np.linalg.norm(
        predicted_ca[2:] - 2.0 * predicted_ca[1:-1] + predicted_ca[:-2], axis=-1
    )

    bond_deviation = math.nan
    bond_mask = np.asarray(peptide_bond_mask, dtype=bool)
    if bond_mask.size and np.any(bond_mask):
        predicted_bonds = np.linalg.norm(
            predicted_ca[:, 1:] - predicted_ca[:, :-1], axis=-1
        )
        apo_bonds = np.linalg.norm(apo_ca[1:] - apo_ca[:-1], axis=-1)
        holo_bonds = np.linalg.norm(holo_ca[1:] - holo_ca[:-1], axis=-1)
        target_bonds = (
            (1.0 - target_times[:, None]) * apo_bonds[None, :]
            + target_times[:, None] * holo_bonds[None, :]
        )
        bond_deviation = float(
            np.mean(np.abs(predicted_bonds[:, bond_mask] - target_bonds[:, bond_mask]))
        )

    return {
        "md_path_translation_mae_a": md_eval.weighted_mean(errors, weights),
        "md_path_weight": float(weights.sum()),
        "ca_endpoint_mae_a": md_eval.weighted_mean(
            endpoint_error, endpoint_weights
        ),
        "ca_mean_step_a": float(np.mean(step)),
        "ca_max_final_step_a": float(np.max(step[-1])),
        "ca_acceleration_a": float(np.mean(acceleration)),
        "ca_bond_deviation_a": bond_deviation,
    }


def finite_mean(values: Iterable[float]) -> float:
    values = np.asarray(list(values), dtype=np.float64)
    values = values[np.isfinite(values)]
    return float(values.mean()) if values.size else math.nan


def aggregate(records: Sequence[Dict[str, object]]) -> Dict[str, object]:
    metric_names = sorted(
        {
            key
            for record in records
            for key, value in record["metrics"].items()
            if isinstance(value, (int, float)) and not key.endswith("_count")
        }
    )
    replica_macro = {
        key: finite_mean(float(record["metrics"][key]) for record in records)
        for key in metric_names
    }
    by_system: Dict[str, List[Dict[str, object]]] = defaultdict(list)
    for record in records:
        by_system[str(record["sample_id"])].append(record)
    system_macro = {
        key: finite_mean(
            finite_mean(float(record["metrics"][key]) for record in system_records)
            for system_records in by_system.values()
        )
        for key in metric_names
    }
    return {"replica_macro": replica_macro, "system_macro": system_macro}


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
    if args.n_path_steps < 2:
        raise ValueError("n_path_steps must be at least 2")
    allowed = {value.strip() for value in args.status_allow.split(",") if value.strip()}
    manifest = read_manifest(Path(args.run_manifest), allowed)
    loader = stage2_eval.create_stage2_dataloader(
        args.data_dir,
        split=args.split,
        batch_size=1,
        shuffle=False,
        num_workers=args.num_workers,
        valid_samples_file=args.valid_samples_file,
        trust_prechecked_samples=args.trust_prechecked_samples,
        stage1v2_posterior_feature_mode="none",
        esm_num_layers=1,
    )
    target_times = np.linspace(0.0, 1.0, args.n_path_steps + 1)
    md_cache_dir = Path(args.md_reference_cache_dir)
    records: List[Dict[str, object]] = []
    missing_manifest: List[str] = []

    with torch.no_grad():
        for batch in tqdm(loader, desc=f"{args.method} MD-reference eval", ncols=120):
            sample_id = str(batch.pdb_ids[0])
            row = manifest.get(sample_id)
            if row is None:
                missing_manifest.append(sample_id)
                continue
            frames = ca_eval.list_frames(Path(str(row["run_dir"])))
            if len(frames) < args.min_frames:
                raise ValueError(
                    f"{sample_id} has {len(frames)} frames, below min_frames={args.min_frames}"
                )
            expected_n = int(batch.n_residues[0])
            apo_ca = batch.Ca_apo[0, :expected_n].numpy().astype(np.float64)
            holo_ca = batch.Ca_holo[0, :expected_n].numpy().astype(np.float64)
            raw_path = load_ca_path(
                row,
                apo_ca,
                holo_ca,
                args.frame_offset,
                args.append_holo_endpoint,
            )
            predicted_ca = interpolate_path(raw_path, target_times)
            predicted_tau = projected_tau_from_ca(predicted_ca, apo_ca, holo_ca)
            rigids_apo = stage2_eval.build_rigids_from_backbone(
                batch.N_apo, batch.Ca_apo, batch.C_apo, batch.node_mask
            )
            rigids_holo = stage2_eval.build_rigids_from_backbone(
                batch.N_holo, batch.Ca_holo, batch.C_holo, batch.node_mask
            )
            reference_paths = md_eval.replica_paths(md_cache_dir, sample_id)
            if not reference_paths:
                raise FileNotFoundError(
                    f"No MD reference replicas for {sample_id} under {md_cache_dir}"
                )
            for reference_path in reference_paths:
                with np.load(reference_path, allow_pickle=False) as data:
                    if str(data["schema_version"].item()) != "md_phase_normal_v1":
                        raise ValueError(f"Unsupported MD cache schema in {reference_path}")
                    if int(data["n_residues"].item()) != expected_n:
                        raise ValueError(f"Residue count mismatch for {reference_path}")
                    expected_hash = str(batch.residue_identity_hashes[0])
                    cached_hash = (
                        str(data["residue_identity_hash"].item())
                        if "residue_identity_hash" in data
                        else ""
                    )
                    if cached_hash != expected_hash:
                        raise ValueError(
                            f"Residue identity mismatch for {reference_path}: "
                            f"cache={cached_hash!r}, batch={expected_hash!r}"
                        )
                    target_rigids, _, arrays = md_eval.reconstruct_md_reference(
                        batch, rigids_apo, rigids_holo, data, target_times
                    )
                    metrics = ca_path_metrics(
                        predicted_ca,
                        target_rigids,
                        arrays,
                        target_times,
                        apo_ca,
                        holo_ca,
                        batch.peptide_bond_mask[0, : max(expected_n - 1, 0)].numpy(),
                    )
                    metrics.update(
                        md_eval.phase_metrics(
                            predicted_tau,
                            arrays,
                            target_times,
                            args.md_min_phase_confidence,
                            args.md_event_tie_tolerance,
                        )
                    )
                    records.append(
                        {
                            "sample_id": sample_id,
                            "reference_id": str(data["sample_id"].item()),
                            "reference_path": str(reference_path),
                            "baseline_run_dir": str(row["run_dir"]),
                            "metrics": metrics,
                        }
                    )

    if missing_manifest and not args.allow_missing_systems:
        raise ValueError(
            f"Baseline manifest misses {len(missing_manifest)} requested systems: "
            f"{missing_manifest[:8]}"
        )
    if not records:
        raise ValueError("No CA baseline records were evaluated")
    result = {
        "schema_version": "ca_baseline_md_reference_eval_v1",
        "method": args.method,
        "run_manifest": args.run_manifest,
        "md_reference_cache_dir": args.md_reference_cache_dir,
        "split": args.split,
        "valid_samples_file": args.valid_samples_file,
        "n_path_steps": args.n_path_steps,
        "frame_offset": args.frame_offset,
        "append_holo_endpoint": args.append_holo_endpoint,
        "requested_systems": len(loader.dataset),
        "systems": len({str(record["sample_id"]) for record in records}),
        "system_coverage": (
            len({str(record["sample_id"]) for record in records}) / len(loader.dataset)
        ),
        "missing_systems": missing_manifest,
        "replicas": len(records),
        "aggregate": aggregate(records),
        "records": records,
    }
    text = json.dumps(json_safe(result), indent=2, sort_keys=True, allow_nan=False)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(text + "\n", encoding="utf-8")
    print(text)
    if args.per_sample_output:
        per_sample = Path(args.per_sample_output)
        per_sample.parent.mkdir(parents=True, exist_ok=True)
        with per_sample.open("w", encoding="utf-8") as handle:
            for record in records:
                handle.write(json.dumps(json_safe(record), sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
