#!/usr/bin/env python3
"""Measure how much held-out MD path error a phase-only bridge can explain."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np
import torch
from tqdm import tqdm

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

import scripts.evaluate_stage2_md_reference_paths as md_eval  # noqa: E402
import scripts.evaluate_stage2_transition_paths as base  # noqa: E402
from scripts.summarize_md_reference_benchmark import paired_bootstrap  # noqa: E402
from src.stage2.datasets import create_stage2_dataloader  # noqa: E402


METHODS = ("synchronous", "oracle_global_phase", "oracle_residue_phase")
LOWER_IS_BETTER = (
    "md_path_translation_mae_a",
    "md_path_rotation_mae_rad",
    "md_path_chi_mae_rad",
    "md_path_product_rmse",
    "phase_tau_mae",
)
HIGHER_IS_BETTER = ("phase_order_pair_accuracy", "phase_order_spearman")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data_dir", default="processed_data/triplets")
    parser.add_argument("--split", default="train", choices=["train", "val", "test"])
    parser.add_argument("--valid_samples_file", required=True)
    parser.add_argument("--md_reference_cache_dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--n_path_steps", type=int, default=20)
    parser.add_argument("--max_batches", type=int, default=None)
    parser.add_argument("--md_min_phase_confidence", type=float, default=0.05)
    parser.add_argument("--md_event_tie_tolerance", type=float, default=0.02)
    parser.add_argument("--md_pause_rate_threshold", type=float, default=0.25)
    parser.add_argument("--md_backtrack_rate_threshold", type=float, default=0.05)
    parser.add_argument("--bootstrap_samples", type=int, default=100_000)
    parser.add_argument("--seed", type=int, default=20260720)
    parser.add_argument(
        "--model_eval",
        action="append",
        default=[],
        help="Optional strict MD evaluation JSON; repeat to average seeds.",
    )
    parser.add_argument("--trust_prechecked_samples", action="store_true")
    return parser.parse_args()


def weighted_global_phase(
    target_tau: np.ndarray,
    phase_confidence: np.ndarray,
    node_mask: np.ndarray,
    active_mask: np.ndarray,
    target_times: np.ndarray,
    min_confidence: float,
) -> np.ndarray:
    """Return the least-squares global phase under fixed residue weights."""
    target_tau = np.asarray(target_tau, dtype=np.float64)
    phase_confidence = np.asarray(phase_confidence, dtype=np.float64)
    target_times = np.asarray(target_times, dtype=np.float64)
    if target_tau.shape != phase_confidence.shape:
        raise ValueError("target tau and phase confidence must share shape")
    if target_tau.shape[0] != target_times.size:
        raise ValueError("target time grid does not match tau")
    residue_confidence = np.max(phase_confidence[1:-1], axis=0)
    residue_mask = (
        np.asarray(node_mask, dtype=bool)
        & np.asarray(active_mask, dtype=bool)
        & (residue_confidence >= float(min_confidence))
    )
    residue_weight = residue_confidence * residue_mask.astype(np.float64)
    if not np.any(residue_weight > 0.0):
        global_values = target_times.copy()
    else:
        global_values = np.sum(target_tau * residue_weight[None], axis=1) / np.sum(
            residue_weight
        )
    global_values = np.maximum.accumulate(np.clip(global_values, 0.0, 1.0))
    global_values[0] = 0.0
    global_values[-1] = 1.0
    return np.broadcast_to(global_values[:, None], target_tau.shape).copy()


def identity_phase(target_times: np.ndarray, n_residues: int) -> np.ndarray:
    return np.broadcast_to(
        np.asarray(target_times, dtype=np.float64)[:, None],
        (len(target_times), int(n_residues)),
    ).copy()


def bridge_path_from_tau(
    batch,
    rigids_apo,
    rigids_holo,
    tau_values: np.ndarray,
    bridge_mode: str,
) -> Tuple[List, List[torch.Tensor]]:
    tau_values = np.asarray(tau_values, dtype=np.float64)
    if tau_values.ndim != 2 or tau_values.shape[1] != int(batch.n_residues[0]):
        raise ValueError("tau residue axis does not match the production batch")
    rigids: List = []
    chi: List[torch.Tensor] = []
    for index, tau_value in enumerate(tau_values):
        if index == 0:
            rigids.append(rigids_apo)
            chi.append(batch.torsion_apo[..., 3:7])
            continue
        if index == len(tau_values) - 1:
            rigids.append(rigids_holo)
            chi.append(batch.torsion_holo[..., 3:7])
            continue
        tau_tensor = torch.from_numpy(tau_value).to(
            device=batch.node_mask.device, dtype=torch.float32
        ).unsqueeze(0)
        tau_tensor = torch.where(
            batch.node_mask.bool(),
            tau_tensor,
            torch.full_like(tau_tensor, index / (len(tau_values) - 1)),
        )
        rigid_t, chi_t = base.phase_interpolate_endpoints_tensor(
            batch, rigids_apo, rigids_holo, tau_tensor, bridge_mode
        )
        rigids.append(rigid_t)
        chi.append(chi_t)
    return rigids, chi


def validate_reference_cache(batch, data, path: Path) -> None:
    if str(data["schema_version"].item()) != "md_phase_normal_v1":
        raise ValueError(f"Unsupported MD cache schema in {path}")
    if str(data["phase_target_mode"].item()) != "inferred":
        raise ValueError(f"MD benchmark requires inferred phase targets: {path}")
    expected_n = int(batch.n_residues[0])
    if int(data["n_residues"].item()) != expected_n:
        raise ValueError(f"Residue count mismatch for {path}")
    expected_hash = str(batch.residue_identity_hashes[0])
    cached_hash = str(data["residue_identity_hash"].item())
    if cached_hash != expected_hash:
        raise ValueError(
            f"Residue identity mismatch for {path}: "
            f"cache={cached_hash!r}, batch={expected_hash!r}"
        )


def method_metrics(
    tau_values: np.ndarray,
    predicted_rigids,
    predicted_chi,
    target_rigids,
    target_chi,
    arrays: Dict[str, np.ndarray],
    data,
    target_times: np.ndarray,
    args: argparse.Namespace,
) -> Dict[str, float]:
    metrics = md_eval.path_error_metrics(
        predicted_rigids,
        predicted_chi,
        target_rigids,
        target_chi,
        arrays,
        data,
    )
    metrics.update(
        md_eval.phase_metrics(
            tau_values,
            arrays,
            target_times,
            args.md_min_phase_confidence,
            args.md_event_tie_tolerance,
        )
    )
    metrics.update(
        md_eval.phase_dynamics_metrics(
            tau_values,
            arrays,
            target_times,
            args.md_min_phase_confidence,
            args.md_pause_rate_threshold,
            args.md_backtrack_rate_threshold,
        )
    )
    metrics["contact_event_count"] = 0
    metrics["contact_event_matched"] = 0
    return metrics


def load_model_eval_ensemble(paths: Sequence[str]) -> List[Dict[str, object]]:
    if not paths:
        return []
    members: List[Dict[Tuple[str, str], Dict[str, object]]] = []
    for raw_path in paths:
        payload = json.loads(Path(raw_path).read_text())
        member = {}
        for record in payload.get("records", []):
            reference_key = (
                Path(str(record["reference_path"])).name
                if record.get("reference_path")
                else str(record["reference_id"])
            )
            key = (str(record["sample_id"]), reference_key)
            member[key] = record
        members.append(member)
    common = set.intersection(*(set(member) for member in members))
    records = []
    for key in sorted(common):
        metric_names = set.intersection(
            *(set(member[key]["metrics"]) for member in members)
        )
        metrics = {}
        for metric in metric_names:
            values = [member[key]["metrics"][metric] for member in members]
            finite = [
                float(value)
                for value in values
                if isinstance(value, (int, float)) and math.isfinite(float(value))
            ]
            if finite:
                metrics[metric] = float(np.mean(finite))
        records.append(
            {
                "sample_id": key[0],
                "reference_id": str(members[0][key].get("reference_id", key[1])),
                "reference_path": str(members[0][key].get("reference_path", "")),
                "metrics": metrics,
            }
        )
    return records


def system_metric_map(
    records: Iterable[Mapping[str, object]], metric: str
) -> Dict[str, float]:
    grouped: Dict[str, List[float]] = {}
    for record in records:
        value = record["metrics"].get(metric)
        if isinstance(value, (int, float)) and math.isfinite(float(value)):
            grouped.setdefault(str(record["sample_id"]), []).append(float(value))
    return {sample_id: float(np.mean(values)) for sample_id, values in grouped.items()}


def compare_methods(
    methods: Mapping[str, Sequence[Mapping[str, object]]],
    primary: str,
    comparator: str,
    metrics: Sequence[str],
    bootstrap_samples: int,
    rng: np.random.Generator,
) -> Dict[str, object]:
    result = {}
    for metric in metrics:
        primary_values = system_metric_map(methods[primary], metric)
        comparator_values = system_metric_map(methods[comparator], metric)
        shared = sorted(set(primary_values) & set(comparator_values))
        if not shared:
            continue
        result[metric] = paired_bootstrap(
            np.asarray([primary_values[key] for key in shared]),
            np.asarray([comparator_values[key] for key in shared]),
            higher_is_better=metric in HIGHER_IS_BETTER,
            samples=int(bootstrap_samples),
            rng=rng,
        )
    return result


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
    device = torch.device(
        args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu"
    )
    loader = create_stage2_dataloader(
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
    cache_dir = Path(args.md_reference_cache_dir)
    target_times = np.linspace(0.0, 1.0, args.n_path_steps + 1, dtype=np.float64)
    method_records: Dict[str, List[Dict[str, object]]] = {
        method: [] for method in METHODS
    }
    systems = 0
    with torch.no_grad():
        for batch_index, batch in enumerate(
            tqdm(loader, desc="MD phase ceiling", ncols=120)
        ):
            if args.max_batches is not None and batch_index >= args.max_batches:
                break
            batch = base.batch_to_device(batch, device)
            sample_id = str(batch.pdb_ids[0])
            paths = md_eval.replica_paths(cache_dir, sample_id)
            if not paths:
                raise FileNotFoundError(f"No MD references for {sample_id}")
            rigids_apo = base.build_rigids_from_backbone(
                batch.N_apo, batch.Ca_apo, batch.C_apo, batch.node_mask
            )
            rigids_holo = base.build_rigids_from_backbone(
                batch.N_holo, batch.Ca_holo, batch.C_holo, batch.node_mask
            )
            for path in paths:
                with np.load(path, allow_pickle=False) as data:
                    validate_reference_cache(batch, data, path)
                    target_rigids, target_chi, arrays = md_eval.reconstruct_md_reference(
                        batch, rigids_apo, rigids_holo, data, target_times
                    )
                    tau_by_method = {
                        "synchronous": identity_phase(
                            target_times, int(batch.n_residues[0])
                        ),
                        "oracle_global_phase": weighted_global_phase(
                            arrays["tau"],
                            arrays["phase_confidence"],
                            arrays["node_mask"],
                            arrays["active_mask"],
                            target_times,
                            args.md_min_phase_confidence,
                        ),
                        "oracle_residue_phase": arrays["tau"].copy(),
                    }
                    bridge_mode = str(data["bridge_mode"].item())
                    for method, tau_values in tau_by_method.items():
                        predicted_rigids, predicted_chi = bridge_path_from_tau(
                            batch,
                            rigids_apo,
                            rigids_holo,
                            tau_values,
                            bridge_mode,
                        )
                        metrics = method_metrics(
                            tau_values,
                            predicted_rigids,
                            predicted_chi,
                            target_rigids,
                            target_chi,
                            arrays,
                            data,
                            target_times,
                            args,
                        )
                        method_records[method].append(
                            {
                                "sample_id": sample_id,
                                "reference_id": str(data["sample_id"].item()),
                                "reference_path": str(path),
                                "metrics": metrics,
                            }
                        )
            systems += 1

    model_records = load_model_eval_ensemble(args.model_eval)
    if model_records:
        oracle_keys = {
            (str(record["sample_id"]), Path(str(record["reference_path"])).name)
            for record in method_records["synchronous"]
        }
        model_records = [
            record
            for record in model_records
            if (
                str(record["sample_id"]),
                Path(str(record.get("reference_path", ""))).name,
            )
            in oracle_keys
        ]
        method_records["model_mean"] = model_records
    aggregates = {
        method: md_eval.aggregate_records(records)
        for method, records in method_records.items()
    }
    comparisons = {}
    rng = np.random.default_rng(args.seed)
    metrics = LOWER_IS_BETTER + HIGHER_IS_BETTER
    for primary, comparator in (
        ("oracle_global_phase", "synchronous"),
        ("oracle_residue_phase", "oracle_global_phase"),
        ("oracle_residue_phase", "synchronous"),
        ("model_mean", "synchronous"),
        ("model_mean", "oracle_residue_phase"),
    ):
        if primary not in method_records or comparator not in method_records:
            continue
        comparisons[f"{primary}_vs_{comparator}"] = compare_methods(
            method_records,
            primary,
            comparator,
            metrics,
            args.bootstrap_samples,
            rng,
        )

    result = {
        "schema_version": "md_phase_representation_ceiling_v1",
        "valid_samples_file": args.valid_samples_file,
        "md_reference_cache_dir": str(cache_dir),
        "systems": systems,
        "replicas": len(method_records["synchronous"]),
        "n_path_steps": args.n_path_steps,
        "model_eval_members": args.model_eval,
        "method_aggregates": aggregates,
        "paired_comparisons": comparisons,
        "method_records": method_records,
    }
    text = json.dumps(json_safe(result), indent=2, sort_keys=True, allow_nan=False)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
