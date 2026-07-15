#!/usr/bin/env python3
"""Evaluate Stage-2 paths against held-out audited MD phase-normal targets."""

from __future__ import annotations

import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
from tqdm import tqdm

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

import scripts.evaluate_stage2_transition_paths as base  # noqa: E402
from src.stage2.modules import (  # noqa: E402
    endpoint_zero_envelope,
    rigid_compose,
    se3_exp,
    so3_log,
    wrap_to_pi,
)


def parse_args():
    parser = base.build_arg_parser()
    parser.description = "Evaluate Stage-2 paths against held-out MD references"
    parser.add_argument(
        "--md_reference_cache_dir",
        required=True,
        help="audited inferred-phase md_phase_normal_v1 cache",
    )
    parser.add_argument(
        "--md_min_phase_confidence",
        type=float,
        default=0.05,
    )
    parser.add_argument(
        "--md_event_tie_tolerance",
        type=float,
        default=0.02,
    )
    args = parser.parse_args()
    if args.batch_size != 1:
        parser.error("MD-reference evaluation currently requires --batch_size 1")
    if not args.output:
        parser.error("--output is required for MD-reference evaluation")
    if args.md_min_phase_confidence < 0.0:
        parser.error("--md_min_phase_confidence must be >= 0")
    return args


def interpolate_time_series(
    source_t: np.ndarray,
    values: np.ndarray,
    target_t: np.ndarray,
) -> np.ndarray:
    """Linearly interpolate an array whose first axis is time."""
    source_t = np.asarray(source_t, dtype=np.float64)
    target_t = np.asarray(target_t, dtype=np.float64)
    values = np.asarray(values)
    if source_t.ndim != 1 or values.shape[0] != source_t.size:
        raise ValueError("time-series shape does not match source_t")
    if source_t.size < 2 or np.any(np.diff(source_t) <= 0.0):
        raise ValueError("source_t must be strictly increasing")
    if np.any(target_t < source_t[0] - 1e-8) or np.any(
        target_t > source_t[-1] + 1e-8
    ):
        raise ValueError("target times fall outside the MD reference grid")

    upper = np.searchsorted(source_t, target_t, side="right")
    upper = np.clip(upper, 1, source_t.size - 1)
    lower = upper - 1
    denom = source_t[upper] - source_t[lower]
    alpha = ((target_t - source_t[lower]) / denom).reshape(
        (-1,) + (1,) * (values.ndim - 1)
    )
    result = (1.0 - alpha) * values[lower] + alpha * values[upper]
    result[target_t <= source_t[0] + 1e-12] = values[0]
    result[target_t >= source_t[-1] - 1e-12] = values[-1]
    return result


def nearest_time_indices(source_t: np.ndarray, target_t: np.ndarray) -> np.ndarray:
    source_t = np.asarray(source_t, dtype=np.float64)
    target_t = np.asarray(target_t, dtype=np.float64)
    distance = np.abs(source_t[:, None] - target_t[None, :])
    return distance.argmin(axis=0)


def weighted_mean(values: np.ndarray, weights: np.ndarray) -> float:
    values = np.asarray(values, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    valid = np.isfinite(values) & np.isfinite(weights) & (weights > 0.0)
    if not np.any(valid):
        return math.nan
    return float(np.sum(values[valid] * weights[valid]) / np.sum(weights[valid]))


def average_ranks(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(values.size, dtype=np.float64)
    start = 0
    while start < values.size:
        end = start + 1
        while end < values.size and values[order[end]] == values[order[start]]:
            end += 1
        ranks[order[start:end]] = 0.5 * (start + end - 1) + 1.0
        start = end
    return ranks


def spearman_correlation(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    valid = np.isfinite(x) & np.isfinite(y)
    if int(valid.sum()) < 2:
        return math.nan
    rx = average_ranks(x[valid])
    ry = average_ranks(y[valid])
    if np.std(rx) <= 1e-12 or np.std(ry) <= 1e-12:
        return math.nan
    return float(np.corrcoef(rx, ry)[0, 1])


def ordering_accuracy(
    predicted: np.ndarray,
    target: np.ndarray,
    tie_tolerance: float,
) -> float:
    predicted = np.asarray(predicted, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    valid = np.isfinite(predicted) & np.isfinite(target)
    predicted = predicted[valid]
    target = target[valid]
    correct = 0
    total = 0
    for i in range(target.size):
        for j in range(i + 1, target.size):
            if abs(target[i] - target[j]) <= tie_tolerance:
                continue
            total += 1
            correct += int(
                np.sign(predicted[i] - predicted[j])
                == np.sign(target[i] - target[j])
            )
    return float(correct / total) if total else math.nan


def monotone_crossing_times(
    values: np.ndarray,
    times: np.ndarray,
    threshold: float = 0.5,
) -> np.ndarray:
    """Find the first upward threshold crossing for each residue."""
    values = np.asarray(values, dtype=np.float64)
    times = np.asarray(times, dtype=np.float64)
    if values.ndim != 2 or values.shape[0] != times.size:
        raise ValueError("values must have shape [time, residue]")
    result = np.full(values.shape[1], np.nan, dtype=np.float64)
    for residue in range(values.shape[1]):
        series = values[:, residue]
        indices = np.flatnonzero(series >= threshold)
        if not indices.size:
            continue
        index = int(indices[0])
        if index == 0:
            result[residue] = times[0]
            continue
        lo, hi = float(series[index - 1]), float(series[index])
        if abs(hi - lo) <= 1e-12:
            result[residue] = times[index]
        else:
            fraction = np.clip((threshold - lo) / (hi - lo), 0.0, 1.0)
            result[residue] = times[index - 1] + fraction * (
                times[index] - times[index - 1]
            )
    return result


def distance_event_times(
    distances: np.ndarray,
    times: np.ndarray,
    threshold: float,
    direction: str,
) -> np.ndarray:
    """Find first contact formation or release crossing per residue."""
    distances = np.asarray(distances, dtype=np.float64)
    times = np.asarray(times, dtype=np.float64)
    if direction not in {"formed", "released"}:
        raise ValueError(f"unsupported event direction {direction!r}")
    result = np.full(distances.shape[1], np.nan, dtype=np.float64)
    for residue in range(distances.shape[1]):
        series = distances[:, residue]
        for index in range(1, times.size):
            lo, hi = float(series[index - 1]), float(series[index])
            crossed = (
                lo > threshold and hi <= threshold
                if direction == "formed"
                else lo <= threshold and hi > threshold
            )
            if not crossed:
                continue
            if abs(hi - lo) <= 1e-12:
                result[residue] = times[index]
            else:
                fraction = np.clip((threshold - lo) / (hi - lo), 0.0, 1.0)
                result[residue] = times[index - 1] + fraction * (
                    times[index] - times[index - 1]
                )
            break
    return result


def resolve_value(args, config, name: str, default):
    value = getattr(args, name, None)
    return value if value is not None else getattr(config, name, default)


def safe_sample_id(sample_id: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "_.-" else "_" for ch in sample_id)


def replica_paths(cache_dir: Path, sample_id: str) -> List[Path]:
    safe_id = safe_sample_id(sample_id)
    direct = cache_dir / f"{safe_id}.npz"
    replicas = sorted(cache_dir.glob(f"{safe_id}__silver_r*.npz"))
    if direct.is_file():
        replicas.insert(0, direct)
    return replicas


def predicted_tau_values(
    args,
    config,
    model,
    batch,
    rigids_apo,
    rigids_holo,
    n_steps: int,
    interaction_prior,
    gate_context,
) -> List[torch.Tensor]:
    path_mode = base.resolve_path_parameterization(args, config)
    if path_mode != "phase_orthogonal_residual_v1":
        bsz, n_res = batch.node_mask.shape
        return [
            torch.full(
                (bsz, n_res),
                k / n_steps,
                dtype=torch.float32,
                device=batch.node_mask.device,
            )
            for k in range(n_steps + 1)
        ]
    return base.phase_residual_tau_values(
        model,
        batch,
        rigids_apo,
        rigids_holo,
        n_steps,
        interaction_prior,
        gate_context,
        tau_mode=str(resolve_value(args, config, "phase_residual_tau_mode", "learned")),
        bridge_mode=str(
            resolve_value(args, config, "phase_residual_bridge_mode", "se3_geodesic")
        ),
        logit_scale=float(resolve_value(args, config, "time_warp_logit_scale", 1.0)),
        rate_eps=float(resolve_value(args, config, "time_warp_rate_eps", 1e-3)),
        rate_clip=float(resolve_value(args, config, "time_warp_rate_clip", 10.0)),
    )


def reconstruct_md_reference(
    batch,
    rigids_apo,
    rigids_holo,
    data,
    target_times: np.ndarray,
) -> Tuple[List, List[torch.Tensor], Dict[str, np.ndarray]]:
    source_t = np.asarray(data["t_values"], dtype=np.float64)
    tau = interpolate_time_series(source_t, data["tau_target"], target_times)
    residual_rot = interpolate_time_series(
        source_t, data["residual_rot"], target_times
    )
    residual_trans = interpolate_time_series(
        source_t, data["residual_trans"], target_times
    )
    residual_chi = interpolate_time_series(
        source_t, data["residual_chi"], target_times
    )
    phase_confidence = interpolate_time_series(
        source_t, data["phase_confidence"], target_times
    )
    residual_confidence = interpolate_time_series(
        source_t, data["residual_confidence"], target_times
    )
    nearest = nearest_time_indices(source_t, target_times)
    residual_valid = np.asarray(data["residual_valid_mask"])[nearest].astype(bool)
    node_mask = np.asarray(data["node_mask"]).astype(bool)
    chi_mask = np.asarray(data["chi_mask"]).astype(bool)
    active_mask = np.asarray(data["active_mask"]).astype(bool)
    bridge_mode = str(data["bridge_mode"].item())
    envelope_kind = str(data["residual_envelope"].item())
    device = batch.node_mask.device

    target_rigids = []
    target_chi = []
    for index, t_value in enumerate(target_times):
        if index == 0:
            target_rigids.append(rigids_apo)
            target_chi.append(batch.torsion_apo[..., 3:7])
            continue
        if index == len(target_times) - 1:
            target_rigids.append(rigids_holo)
            target_chi.append(batch.torsion_holo[..., 3:7])
            continue
        tau_t = torch.from_numpy(tau[index]).to(device=device, dtype=torch.float32)
        tau_t = tau_t.unsqueeze(0)
        tau_t = torch.where(
            batch.node_mask.bool(),
            tau_t,
            torch.full_like(tau_t, float(t_value)),
        )
        bridge_rigid, bridge_chi = base.phase_interpolate_endpoints_tensor(
            batch, rigids_apo, rigids_holo, tau_t, bridge_mode
        )
        rigid_residual = torch.from_numpy(
            np.concatenate([residual_rot[index], residual_trans[index]], axis=-1)
        ).to(device=device, dtype=torch.float32).unsqueeze(0)
        chi_residual = torch.from_numpy(residual_chi[index]).to(
            device=device, dtype=torch.float32
        ).unsqueeze(0)
        envelope = endpoint_zero_envelope(
            torch.tensor([float(t_value)], device=device), kind=envelope_kind
        ).view(1, 1, 1)
        bridge_rotation, bridge_translation = base.rigid_to_rt(bridge_rigid)
        residual_rotation, residual_translation = se3_exp(rigid_residual * envelope)
        target_rotation, target_translation = rigid_compose(
            bridge_rotation,
            bridge_translation,
            residual_rotation,
            residual_translation,
        )
        target_rigids.append(base.rt_to_rigid(target_rotation, target_translation))
        target_chi.append(wrap_to_pi(bridge_chi + chi_residual * envelope))

    arrays = {
        "tau": tau,
        "phase_confidence": phase_confidence,
        "residual_confidence": residual_confidence,
        "residual_valid": residual_valid,
        "node_mask": node_mask,
        "chi_mask": chi_mask,
        "active_mask": active_mask,
    }
    return target_rigids, target_chi, arrays


def path_error_metrics(
    predicted_rigids,
    predicted_chi,
    target_rigids,
    target_chi,
    arrays: Dict[str, np.ndarray],
    data,
) -> Dict[str, float]:
    rotation_errors = []
    translation_errors = []
    chi_errors = []
    product_errors_sq = []
    residue_weights = []
    rotation_scale = float(data["rotation_metric_scale"].item())
    translation_scale = float(data["translation_metric_scale"].item())
    chi_scale = float(data["chi_metric_scale"].item())

    for index in range(1, len(predicted_rigids) - 1):
        pred_rotation, pred_translation = base.rigid_to_rt(predicted_rigids[index])
        target_rotation, target_translation = base.rigid_to_rt(target_rigids[index])
        relative_rotation = target_rotation.transpose(-2, -1) @ pred_rotation
        rotation_error = torch.linalg.vector_norm(so3_log(relative_rotation), dim=-1)[0]
        translation_error = torch.linalg.vector_norm(
            pred_translation - target_translation, dim=-1
        )[0]
        chi_error_all = wrap_to_pi(predicted_chi[index] - target_chi[index]).abs()[0]

        valid = arrays["residual_valid"][index] & arrays["node_mask"]
        weight = arrays["residual_confidence"][index] * valid.astype(np.float64)
        chi_mask = arrays["chi_mask"] & valid[:, None]
        chi_count = np.maximum(chi_mask.sum(axis=-1), 1)
        chi_error_np = chi_error_all.detach().cpu().numpy()
        chi_mean = (chi_error_np * chi_mask).sum(axis=-1) / chi_count
        chi_sq = ((chi_error_np / chi_scale) ** 2 * chi_mask).sum(axis=-1) / chi_count
        product_sq = (
            (rotation_error.detach().cpu().numpy() / rotation_scale) ** 2
            + (translation_error.detach().cpu().numpy() / translation_scale) ** 2
            + chi_sq
        )

        rotation_errors.append(rotation_error.detach().cpu().numpy())
        translation_errors.append(translation_error.detach().cpu().numpy())
        chi_errors.append(chi_mean)
        product_errors_sq.append(product_sq)
        residue_weights.append(weight)

    weights = np.stack(residue_weights)
    product_mse = weighted_mean(np.stack(product_errors_sq), weights)
    return {
        "md_path_rotation_mae_rad": weighted_mean(
            np.stack(rotation_errors), weights
        ),
        "md_path_translation_mae_a": weighted_mean(
            np.stack(translation_errors), weights
        ),
        "md_path_chi_mae_rad": weighted_mean(np.stack(chi_errors), weights),
        "md_path_product_rmse": (
            math.sqrt(product_mse) if math.isfinite(product_mse) else math.nan
        ),
        "md_path_weight": float(weights.sum()),
    }


def phase_metrics(
    predicted_tau: np.ndarray,
    arrays: Dict[str, np.ndarray],
    target_times: np.ndarray,
    min_confidence: float,
    tie_tolerance: float,
) -> Dict[str, float]:
    target_tau = arrays["tau"]
    confidence = arrays["phase_confidence"]
    interior = np.zeros(target_tau.shape[0], dtype=bool)
    interior[1:-1] = True
    mask = (
        interior[:, None]
        & arrays["node_mask"][None]
        & arrays["active_mask"][None]
        & (confidence >= min_confidence)
    )
    weights = confidence * mask.astype(np.float64)
    tau_mae = weighted_mean(np.abs(predicted_tau - target_tau), weights)

    predicted_midpoint = monotone_crossing_times(predicted_tau, target_times)
    target_midpoint = monotone_crossing_times(target_tau, target_times)
    residue_confidence = np.max(confidence[1:-1], axis=0)
    valid = (
        arrays["node_mask"]
        & arrays["active_mask"]
        & (residue_confidence >= min_confidence)
        & np.isfinite(predicted_midpoint)
        & np.isfinite(target_midpoint)
    )
    midpoint_weights = residue_confidence * valid.astype(np.float64)
    return {
        "phase_tau_mae": tau_mae,
        "phase_midpoint_mae": weighted_mean(
            np.abs(predicted_midpoint - target_midpoint), midpoint_weights
        ),
        "phase_order_spearman": spearman_correlation(
            predicted_midpoint[valid], target_midpoint[valid]
        ),
        "phase_order_pair_accuracy": ordering_accuracy(
            predicted_midpoint[valid], target_midpoint[valid], tie_tolerance
        ),
        "phase_residue_count": int(valid.sum()),
        "phase_weight": float(weights.sum()),
    }


def predicted_ligand_distances(batch, fk_module, rigids, chi, times) -> np.ndarray:
    distances = []
    for rigid_t, chi_t, t_value in zip(rigids, chi, times):
        atom14 = base.torsions_to_atom14(
            fk_module,
            base.interpolate_backbone_torsions(batch, float(t_value)),
            chi_t,
            rigid_t,
            batch.aatype,
        )
        distance = base.min_sc_ligand_dist(
            atom14["atom14_pos"].float(),
            atom14["atom14_mask"].bool(),
            batch.lig_points.float(),
            batch.lig_mask.bool(),
            batch.node_mask.bool(),
        )
        distances.append(distance[0].detach().cpu().numpy())
    return np.stack(distances)


def contact_event_metrics(
    predicted_distances: np.ndarray,
    target_times: np.ndarray,
    data,
    contact_dist: float,
) -> Dict[str, float]:
    formed = np.asarray(data["formed_contact_mask"]).astype(bool)
    released = np.asarray(data["release_mask"]).astype(bool)
    transient = np.asarray(data["transient_contact_mask"]).astype(bool)
    target_progress = np.asarray(data["contact_event_progress"], dtype=np.float64)
    formed_time = distance_event_times(
        predicted_distances, target_times, contact_dist, "formed"
    )
    released_time = distance_event_times(
        predicted_distances, target_times, contact_dist, "released"
    )
    predicted_time = np.where(formed, formed_time, released_time)
    event_mask = formed | released
    matched = event_mask & np.isfinite(predicted_time) & np.isfinite(target_progress)
    target_count = int(event_mask.sum())
    matched_count = int(matched.sum())
    transient_predicted = np.min(predicted_distances, axis=0) <= contact_dist
    return {
        "contact_event_coverage": (
            float(matched_count / target_count) if target_count else math.nan
        ),
        "contact_event_timing_mae": (
            float(np.mean(np.abs(predicted_time[matched] - target_progress[matched])))
            if matched_count
            else math.nan
        ),
        "contact_event_order_spearman": spearman_correlation(
            predicted_time[matched], target_progress[matched]
        ),
        "transient_contact_recall": (
            float(np.mean(transient_predicted[transient]))
            if np.any(transient)
            else math.nan
        ),
        "contact_event_count": target_count,
        "contact_event_matched": matched_count,
        "transient_contact_count": int(transient.sum()),
    }


def finite_mean(values: Iterable[float]) -> float:
    values = np.asarray(list(values), dtype=np.float64)
    values = values[np.isfinite(values)]
    return float(values.mean()) if values.size else math.nan


def json_safe(value):
    if isinstance(value, dict):
        return {key: json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [json_safe(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def aggregate_records(records: Sequence[Dict[str, object]]) -> Dict[str, object]:
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
    system_values: Dict[str, List[float]] = defaultdict(list)
    for system_records in by_system.values():
        for key in metric_names:
            value = finite_mean(
                float(record["metrics"][key]) for record in system_records
            )
            if math.isfinite(value):
                system_values[key].append(value)
    system_macro = {
        key: finite_mean(system_values.get(key, [])) for key in metric_names
    }
    event_total = sum(int(record["metrics"]["contact_event_count"]) for record in records)
    event_matched = sum(
        int(record["metrics"]["contact_event_matched"]) for record in records
    )
    return {
        "replica_macro": replica_macro,
        "system_macro": system_macro,
        "pooled": {
            "contact_event_count": event_total,
            "contact_event_matched": event_matched,
            "contact_event_coverage": (
                float(event_matched / event_total) if event_total else math.nan
            ),
        },
    }


def main() -> None:
    args = parse_args()
    device = torch.device(
        args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu"
    )
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    config = checkpoint.get("config", SimpleNamespace())
    args.config = config
    model_config, interaction_settings, stage1v2_settings = (
        base.build_model_config_for_checkpoint(args, config, args.split)
    )
    integration_clips = base.resolve_integration_clips(args, config)
    model = base.TorsionFlowNet(model_config).to(device)
    base.load_model_state_allow_timewarp_head(model, checkpoint["model_state_dict"])
    model.eval()
    fk_module = base.create_openfold_fk().to(device)
    fk_module.eval()

    n_steps = int(args.n_integration_steps or getattr(config, "n_integration_steps", 5))
    if n_steps < 2:
        raise ValueError("MD-reference evaluation requires at least two path steps")
    loader = base.create_stage2_dataloader(
        args.data_dir,
        split=args.split,
        batch_size=1,
        shuffle=False,
        num_workers=args.num_workers,
        require_nma=bool(getattr(config, "use_nma", False)),
        valid_samples_file=args.valid_samples_file,
        trust_prechecked_samples=args.trust_prechecked_samples,
        stage1v2_posterior_cache_dir=stage1v2_settings["cache_dir"],
        stage1v2_posterior_feature_mode=stage1v2_settings["mode"],
        stage1v2_posterior_feature_names=stage1v2_settings["names_raw"],
        esm_num_layers=int(getattr(config, "esm_num_layers", 1)),
    )
    md_cache_dir = Path(args.md_reference_cache_dir)
    target_times = np.linspace(0.0, 1.0, n_steps + 1, dtype=np.float64)
    records: List[Dict[str, object]] = []
    systems = 0

    with torch.no_grad():
        for batch_index, batch in enumerate(
            tqdm(loader, desc="MD-reference eval", ncols=120)
        ):
            if args.max_batches is not None and batch_index >= args.max_batches:
                break
            batch = base.batch_to_device(batch, device)
            sample_id = str(batch.pdb_ids[0])
            paths = replica_paths(md_cache_dir, sample_id)
            if not paths:
                raise FileNotFoundError(
                    f"No MD reference replicas for {sample_id} under {md_cache_dir}"
                )
            rigids_apo = base.build_rigids_from_backbone(
                batch.N_apo, batch.Ca_apo, batch.C_apo, batch.node_mask
            )
            rigids_holo = base.build_rigids_from_backbone(
                batch.N_holo, batch.Ca_holo, batch.C_holo, batch.node_mask
            )
            interaction_prior_feature = base.compute_interaction_prior_feature(
                args, config, batch, fk_module, rigids_apo, rigids_holo
            )
            prior_features = base.combined_prior_features(
                args,
                config,
                batch,
                interaction_prior_feature,
                stage1v2_settings,
                interaction_settings,
            )
            gate_context = base.esm_gate_context(config, batch, stage1v2_settings)
            predicted_rigids, predicted_chi, times, _ = base.construct_path(
                args,
                config,
                model,
                batch,
                rigids_apo,
                rigids_holo,
                n_steps,
                prior_features,
                gate_context,
                integration_clips,
                fk_module=fk_module,
            )
            if not np.allclose(times, target_times, atol=1e-8):
                raise ValueError("predicted path grid does not match the MD evaluation grid")
            tau_values = predicted_tau_values(
                args,
                config,
                model,
                batch,
                rigids_apo,
                rigids_holo,
                n_steps,
                prior_features,
                gate_context,
            )
            predicted_tau = torch.stack(tau_values, dim=0)[:, 0].cpu().numpy()
            predicted_distances = predicted_ligand_distances(
                batch,
                fk_module,
                predicted_rigids,
                predicted_chi,
                times,
            )

            for path in paths:
                with np.load(path, allow_pickle=False) as data:
                    if str(data["schema_version"].item()) != "md_phase_normal_v1":
                        raise ValueError(f"Unsupported MD cache schema in {path}")
                    phase_target_mode = (
                        str(data["phase_target_mode"].item())
                        if "phase_target_mode" in data
                        else "inferred"
                    )
                    if phase_target_mode != "inferred":
                        raise ValueError(f"MD benchmark requires inferred phase targets: {path}")
                    expected_n = int(batch.n_residues[0])
                    if int(data["n_residues"].item()) != expected_n:
                        raise ValueError(f"Residue count mismatch for {path}")
                    target_rigids, target_chi, arrays = reconstruct_md_reference(
                        batch,
                        rigids_apo,
                        rigids_holo,
                        data,
                        target_times,
                    )
                    metrics = path_error_metrics(
                        predicted_rigids,
                        predicted_chi,
                        target_rigids,
                        target_chi,
                        arrays,
                        data,
                    )
                    metrics.update(
                        phase_metrics(
                            predicted_tau,
                            arrays,
                            target_times,
                            args.md_min_phase_confidence,
                            args.md_event_tie_tolerance,
                        )
                    )
                    metrics.update(
                        contact_event_metrics(
                            predicted_distances,
                            target_times,
                            data,
                            args.contact_dist,
                        )
                    )
                    records.append(
                        {
                            "sample_id": sample_id,
                            "reference_id": str(data["sample_id"].item()),
                            "reference_path": str(path),
                            "metrics": metrics,
                        }
                    )
            systems += 1

    aggregate = aggregate_records(records)
    result = {
        "schema_version": "stage2_md_reference_eval_v1",
        "checkpoint": args.checkpoint,
        "path_parameterization": base.resolve_path_parameterization(args, config),
        "md_reference_cache_dir": str(md_cache_dir),
        "split": args.split,
        "valid_samples_file": args.valid_samples_file,
        "n_integration_steps": n_steps,
        "systems": systems,
        "replicas": len(records),
        "md_min_phase_confidence": args.md_min_phase_confidence,
        "aggregate": aggregate,
        "records": records,
    }
    text = json.dumps(json_safe(result), indent=2, sort_keys=True, allow_nan=False)
    print(text)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
