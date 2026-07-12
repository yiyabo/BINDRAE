#!/usr/bin/env python3
"""Evaluate endpoint-exact bridge time-warp headroom for Stage-2 paths.

This evaluator answers a narrow question before training another model:

    Can a monotone progress warp along the apo-holo bridge beat pure_bridge?

It compares pure_bridge with oracle global/group/residue time-warp paths.  It
can also project a free-flow teacher path onto the bridge and evaluate the
resulting endpoint-exact progress path.  All evaluated paths remain exact at
apo/holo because they only sample the analytic bridge at tau(0)=0 and tau(1)=1.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import torch
from tqdm import tqdm

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))
flash_ipa_path = project_root / "vendor" / "flash_ipa" / "src"
if flash_ipa_path.exists():
    sys.path.insert(0, str(flash_ipa_path))

import scripts.evaluate_stage2_transition_paths as base  # noqa: E402
from src.stage1.models.fk_openfold import create_openfold_fk  # noqa: E402
from src.stage2.datasets import create_stage2_dataloader  # noqa: E402
from src.stage2.models import TorsionFlowNet  # noqa: E402
from src.stage2.modules import rigid_compose, rigid_inverse, se3_exp, se3_log, wrap_to_pi  # noqa: E402
from flash_ipa.rigid import Rigid, Rotation  # noqa: E402


METHODS = (
    "pure_bridge",
    "oracle_global",
    "oracle_group",
    "oracle_residue",
    "freeflow_projected",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, help="Stage-2 checkpoint used for config and optional free-flow")
    parser.add_argument("--free_flow_checkpoint", default=None, help="Optional free-flow teacher checkpoint")
    parser.add_argument("--data_dir", default="processed_data/triplets")
    parser.add_argument("--split", default="val", choices=["train", "val", "test"])
    parser.add_argument("--valid_samples_file", default=None)
    parser.add_argument("--trust_prechecked_samples", action="store_true")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max_batches", type=int, default=None)
    parser.add_argument("--n_path_steps", type=int, default=16)
    parser.add_argument("--n_tau_grid", type=int, default=33)
    parser.add_argument("--tau_transition_weight", type=float, default=0.0)
    parser.add_argument("--methods", default="pure_bridge,oracle_global,oracle_group,oracle_residue")
    parser.add_argument(
        "--reference_bridge_mode",
        default="se3_geodesic",
        choices=["se3_geodesic", "cartesian_backbone"],
    )

    parser.add_argument("--free_flow_path_parameterization", default="flow",
                        choices=["flow", "projected_flow", "boundary_residual_v1", "boundary_residual"])
    parser.add_argument("--n_free_flow_steps", type=int, default=None)
    parser.add_argument("--integration_chi_clip", type=float, default=None)
    parser.add_argument("--integration_rot_clip", type=float, default=None)
    parser.add_argument("--integration_trans_clip", type=float, default=None)

    parser.add_argument("--interaction_prior_ckpt", default=None)
    parser.add_argument(
        "--interaction_prior_feature_mode",
        default=None,
        choices=["none", "prior", "prior_shuffled", "zero", "oracle_contact"],
    )
    parser.add_argument("--interaction_prior_temperature", type=float, default=None)
    parser.add_argument("--interaction_prior_feature_scale", type=float, default=None)
    parser.add_argument(
        "--stage1v2_posterior_feature_mode",
        default=None,
        choices=[
            "none",
            "zero",
            "student",
            "student_shuffled",
            "oracle_holo_truth",
            "external_teacher_cached",
            "oracle_motion",
            "oracle_motion_residue_shuffled",
            "oracle_motion_sample_shuffled",
        ],
    )
    parser.add_argument("--stage1v2_posterior_cache_dir", default=None)
    parser.add_argument("--stage1v2_posterior_feature_names", default=None)
    parser.add_argument("--stage1v2_posterior_feature_scale", type=float, default=None)

    parser.add_argument("--active_delta", type=float, default=0.75)
    parser.add_argument("--contact_dist", type=float, default=4.5)
    parser.add_argument("--path_dist_cap", type=float, default=20.0)
    parser.add_argument("--ligand_clash_dist", type=float, default=2.2)
    parser.add_argument("--pocket_threshold", type=float, default=0.3)
    parser.add_argument(
        "--phase_teacher_cache_dir",
        default=None,
        help=(
            "Optional output directory for phase_teacher_v1 pseudo-label caches. "
            "Requires freeflow_projected and stores projected tau plus confidence."
        ),
    )
    parser.add_argument("--phase_teacher_skip_existing", action="store_true")
    parser.add_argument("--output", default=None)
    return parser.parse_args()


def parse_methods(raw: str) -> Tuple[str, ...]:
    normalized = str(raw)
    for sep in (";", ":", "+"):
        normalized = normalized.replace(sep, ",")
    methods = tuple(item.strip() for item in normalized.split(",") if item.strip())
    unknown = sorted(set(methods) - set(METHODS))
    if unknown:
        raise ValueError(f"Unsupported methods: {unknown}; allowed={METHODS}")
    if not methods:
        raise ValueError("at least one method is required")
    return methods


def rigid_from_rt(R: torch.Tensor, t: torch.Tensor) -> Rigid:
    return Rigid(rots=Rotation(rot_mats=R), trans=t)


def bridge_state_at_tau(
    batch,
    rigids_apo: Rigid,
    rigids_holo: Rigid,
    tau: torch.Tensor,
    bridge_mode: str = "se3_geodesic",
) -> Tuple[Rigid, torch.Tensor]:
    return base.phase_interpolate_endpoints_tensor(
        batch,
        rigids_apo,
        rigids_holo,
        tau,
        bridge_mode,
    )


def build_tau_grid_path(
    batch,
    rigids_apo: Rigid,
    rigids_holo: Rigid,
    tau_seq: torch.Tensor,
    bridge_mode: str = "se3_geodesic",
) -> Tuple[List[Rigid], List[torch.Tensor], List[float]]:
    rigids_list: List[Rigid] = []
    chi_list: List[torch.Tensor] = []
    t_list: List[float] = []
    steps = tau_seq.shape[0] - 1
    for k in range(tau_seq.shape[0]):
        rigids_t, chi_t = bridge_state_at_tau(
            batch, rigids_apo, rigids_holo, tau_seq[k], bridge_mode
        )
        rigids_list.append(rigids_t)
        chi_list.append(chi_t)
        t_list.append(k / max(steps, 1))
    return rigids_list, chi_list, t_list


def compute_bridge_distance_grid(
    fk_module,
    batch,
    rigids_apo: Rigid,
    rigids_holo: Rigid,
    tau_grid: torch.Tensor,
    bridge_mode: str = "se3_geodesic",
) -> Tuple[torch.Tensor, List[Rigid], List[torch.Tensor]]:
    dists: List[torch.Tensor] = []
    rigids_grid: List[Rigid] = []
    chi_grid: List[torch.Tensor] = []
    for tau_value in tau_grid:
        tau = batch.w_res.new_full(batch.w_res.shape, float(tau_value))
        rigids_t, chi_t = bridge_state_at_tau(
            batch, rigids_apo, rigids_holo, tau, bridge_mode
        )
        atom14 = base.torsions_to_atom14(
            fk_module,
            batch.torsion_apo[..., :3],
            chi_t,
            rigids_t,
            batch.aatype,
        )
        d_t = base.min_sc_ligand_dist(
            atom14["atom14_pos"].float(),
            atom14["atom14_mask"].bool(),
            batch.lig_points.float(),
            batch.lig_mask.bool(),
            batch.node_mask.bool(),
        )
        dists.append(d_t)
        rigids_grid.append(rigids_t)
        chi_grid.append(chi_t)
    return torch.stack(dists, dim=0), rigids_grid, chi_grid


def monotone_dp(
    cost: torch.Tensor,
    transition_weight: float = 0.0,
) -> torch.Tensor:
    """Find monotone grid indices minimizing cost.

    Args:
        cost: [T, G, ...] cost tensor. Caller should set impossible states to
            a large value; endpoints are usually forced to grid 0 and G-1.

    Returns:
        Long tensor [T, ...] with nondecreasing grid indices.
    """
    if cost.ndim < 2:
        raise ValueError("cost must have shape [T, G, ...]")
    T, G = int(cost.shape[0]), int(cost.shape[1])
    if T < 2 or G < 2:
        raise ValueError(f"monotone_dp requires T>=2 and G>=2, got T={T}, G={G}")

    trailing_shape = cost.shape[2:]
    flat_cost = cost.reshape(T, G, -1)
    dp = flat_cost[0]
    backptrs: List[torch.Tensor] = []
    if float(transition_weight) > 0.0:
        grid_float = torch.arange(G, device=cost.device, dtype=cost.dtype)
        previous = grid_float[:, None]
        current = grid_float[None, :]
        expected_step = (G - 1) / max(T - 1, 1)
        transition = float(transition_weight) * (
            current - previous - expected_step
        ).square()
        transition = transition.masked_fill(previous > current, float("inf"))
    for t_idx in range(1, T):
        if float(transition_weight) > 0.0:
            candidate = dp[:, None, :] + transition[:, :, None]
            best_value, best_index = candidate.min(dim=0)
            dp = flat_cost[t_idx] + best_value
            backptrs.append(best_index)
        else:
            prefix_val, prefix_arg = torch.cummin(dp, dim=0)
            dp = flat_cost[t_idx] + prefix_val
            backptrs.append(prefix_arg)

    last_idx = dp.argmin(dim=0)
    path = [last_idx]
    for backptr in reversed(backptrs):
        gather_idx = path[-1].unsqueeze(0)
        prev_idx = torch.gather(backptr, dim=0, index=gather_idx).squeeze(0)
        path.append(prev_idx)
    path.reverse()
    out = torch.stack(path, dim=0).reshape(T, *trailing_shape)
    if out.shape[0] != T:
        raise RuntimeError("failed to reconstruct monotone path")
    return out.long()


def force_endpoint_costs(cost: torch.Tensor) -> torch.Tensor:
    cost = cost.clone()
    large = cost.new_tensor(1.0e8)
    cost[0, 1:] = large
    cost[-1, :-1] = large
    return cost


def identity_tau_seq(batch, n_steps: int) -> torch.Tensor:
    t = torch.linspace(0.0, 1.0, int(n_steps) + 1, device=batch.w_res.device)
    return t[:, None, None].expand(-1, batch.w_res.shape[0], batch.w_res.shape[1]).clone()


def nearest_grid_indices(tau_seq: torch.Tensor, tau_grid: torch.Tensor) -> torch.Tensor:
    diff = (tau_seq.unsqueeze(1) - tau_grid.view(1, -1, 1, 1)).abs()
    return diff.argmin(dim=1).long()


def path_cost_against_distance_schedule(
    d_bridge_grid: torch.Tensor,
    d_apo: torch.Tensor,
    d_holo: torch.Tensor,
    n_steps: int,
    cap: float,
) -> torch.Tensor:
    t_values = torch.linspace(0.0, 1.0, int(n_steps) + 1, device=d_apo.device)
    target = d_apo.unsqueeze(0) + t_values[:, None, None] * (d_holo - d_apo).unsqueeze(0)
    cost = (
        d_bridge_grid.unsqueeze(0).clamp(max=float(cap))
        - target[:, None].clamp(max=float(cap))
    ).abs()
    return cost


def masked_mean_cost(cost: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Reduce [T,G,B,N] cost to [T,G,B] using a residue mask [B,N]."""
    mask_f = mask.float()
    denom = mask_f.sum(dim=-1).clamp(min=1.0)
    return (cost * mask_f[None, None]).sum(dim=-1) / denom[None, None]


def build_oracle_global_tau(cost: torch.Tensor, mask: torch.Tensor, tau_grid: torch.Tensor) -> torch.Tensor:
    group_cost = masked_mean_cost(cost, mask)
    idx = monotone_dp(force_endpoint_costs(group_cost))
    tau = tau_grid[idx]
    return tau[:, :, None].expand(-1, -1, mask.shape[1]).clone()


def build_group_masks(
    finite: torch.Tensor,
    d_apo: torch.Tensor,
    d_holo: torch.Tensor,
    active_delta: float,
    contact_dist: float,
) -> List[Tuple[str, torch.Tensor]]:
    delta = d_holo - d_apo
    active = finite & (delta.abs() >= float(active_delta))
    formed = active & (d_apo > float(contact_dist)) & (d_holo <= float(contact_dist))
    released = active & (d_apo <= float(contact_dist)) & (d_holo > float(contact_dist))
    stable_contact = finite & (d_apo <= float(contact_dist)) & (d_holo <= float(contact_dist))
    stable_noncontact = finite & (d_apo > float(contact_dist)) & (d_holo > float(contact_dist))
    used = formed | released | stable_contact | stable_noncontact
    other = finite & ~used
    return [
        ("formed_contact", formed),
        ("released_contact", released),
        ("stable_contact", stable_contact),
        ("stable_noncontact", stable_noncontact),
        ("other_pocket", other),
    ]


def build_oracle_group_tau(
    cost: torch.Tensor,
    finite: torch.Tensor,
    group_masks: List[Tuple[str, torch.Tensor]],
    tau_grid: torch.Tensor,
    n_steps: int,
) -> Tuple[torch.Tensor, Dict[str, int]]:
    tau = identity_tau_seq(SimpleNamespace(w_res=finite.float()), n_steps)
    counts: Dict[str, int] = {}
    for name, mask in group_masks:
        counts[name] = int(mask.sum().item())
        if not mask.any():
            continue
        group_tau = build_oracle_global_tau(cost, mask, tau_grid)
        tau = torch.where(mask[None], group_tau, tau)
    tau = torch.where(finite[None], tau, identity_tau_seq(SimpleNamespace(w_res=finite.float()), n_steps))
    return tau, counts


def build_oracle_residue_tau(
    cost: torch.Tensor,
    finite: torch.Tensor,
    tau_grid: torch.Tensor,
    n_steps: int,
) -> torch.Tensor:
    idx = monotone_dp(force_endpoint_costs(cost))
    tau = tau_grid[idx]
    identity = identity_tau_seq(SimpleNamespace(w_res=finite.float()), n_steps)
    return torch.where(finite[None], tau, identity)


def state_projection_cost_to_grid(
    grid_rigids: List[Rigid],
    grid_chi: List[torch.Tensor],
    teacher_rigids: List[Rigid],
    teacher_chi: List[torch.Tensor],
    chi_mask: torch.Tensor,
) -> torch.Tensor:
    rows: List[torch.Tensor] = []
    for teach_rigid, teach_chi in zip(teacher_rigids, teacher_chi):
        per_grid: List[torch.Tensor] = []
        for grid_rigid, grid_chi_t in zip(grid_rigids, grid_chi):
            total_gap, _, _, _ = base.state_step_motion(
                grid_rigid,
                grid_chi_t,
                teach_rigid,
                teach_chi,
                chi_mask,
            )
            per_grid.append(total_gap)
        rows.append(torch.stack(per_grid, dim=0))
    return torch.stack(rows, dim=0)


def load_model_for_checkpoint(args, ckpt_path: str, device: torch.device, split: str):
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    config = ckpt.get("config", SimpleNamespace())
    args.config = config
    model_config, interaction_settings, stage1v2_settings = base.build_model_config_for_checkpoint(
        args, config, split
    )
    model = TorsionFlowNet(model_config).to(device)
    base.load_model_state_allow_timewarp_head(model, ckpt["model_state_dict"])
    model.eval()
    model._bindrae_config = config
    return ckpt, config, model, interaction_settings, stage1v2_settings


def build_free_flow_projection(
    args,
    model,
    fk_module,
    batch,
    rigids_apo: Rigid,
    rigids_holo: Rigid,
    grid_rigids: List[Rigid],
    grid_chi: List[torch.Tensor],
    n_steps: int,
    tau_grid: torch.Tensor,
    interaction_settings: Dict[str, object],
    stage1v2_settings: Dict[str, object],
    integration_clips: Dict[str, float],
    finite: torch.Tensor,
) -> Dict[str, torch.Tensor]:
    original_mode = getattr(args, "path_parameterization", "checkpoint")
    args.path_parameterization = args.free_flow_path_parameterization
    try:
        interaction_prior_feature = base.compute_interaction_prior_feature(
            args, args.config, batch, fk_module, rigids_apo, rigids_holo
        )
        prior_features = base.combined_prior_features(
            args,
            args.config,
            batch,
            interaction_prior_feature,
            stage1v2_settings,
            interaction_settings,
        )
        gate_context = base.esm_gate_context(args.config, batch, stage1v2_settings)
        teacher_rigids, teacher_chi, _, _ = base.construct_path(
            args,
            args.config,
            model,
            batch,
            rigids_apo,
            rigids_holo,
            n_steps,
            prior_features,
            gate_context,
            integration_clips,
        )
    finally:
        args.path_parameterization = original_mode

    proj_cost = state_projection_cost_to_grid(
        grid_rigids,
        grid_chi,
        teacher_rigids,
        teacher_chi,
        batch.chi_mask,
    )
    idx = monotone_dp(
        force_endpoint_costs(proj_cost),
        transition_weight=float(args.tau_transition_weight),
    )
    tau = tau_grid[idx]
    identity = identity_tau_seq(batch, n_steps)
    tau = torch.where(finite[None], tau, identity)

    selected_cost = torch.gather(proj_cost, dim=1, index=idx.unsqueeze(1)).squeeze(1)
    identity_idx = torch.linspace(
        0,
        int(tau_grid.numel()) - 1,
        steps=n_steps + 1,
        device=tau_grid.device,
    ).round().long()
    identity_idx = identity_idx.view(-1, 1, 1).expand_as(idx)
    identity_cost = torch.gather(
        proj_cost,
        dim=1,
        index=identity_idx.unsqueeze(1),
    ).squeeze(1)
    relative_gain = (
        (identity_cost - selected_cost).clamp(min=0.0)
        / identity_cost.clamp(min=1e-6)
    ).clamp(max=1.0)
    phase_offset = (tau - identity).abs()
    confidence = relative_gain * (
        phase_offset / max(1.0 / max(n_steps, 1), 1e-6)
    ).clamp(max=1.0)
    confidence = confidence * finite.unsqueeze(0).float()
    confidence[0] = 0.0
    confidence[-1] = 0.0
    return {
        "tau": tau,
        "confidence": confidence,
        "projection_cost": selected_cost,
        "identity_cost": identity_cost,
        "teacher_rigids": teacher_rigids,
        "teacher_chi": teacher_chi,
    }


def build_free_flow_projected_tau(
    args,
    model,
    fk_module,
    batch,
    rigids_apo: Rigid,
    rigids_holo: Rigid,
    grid_rigids: List[Rigid],
    grid_chi: List[torch.Tensor],
    n_steps: int,
    tau_grid: torch.Tensor,
    interaction_settings: Dict[str, object],
    stage1v2_settings: Dict[str, object],
    integration_clips: Dict[str, float],
    finite: torch.Tensor,
) -> torch.Tensor:
    return build_free_flow_projection(
        args,
        model,
        fk_module,
        batch,
        rigids_apo,
        rigids_holo,
        grid_rigids,
        grid_chi,
        n_steps,
        tau_grid,
        interaction_settings,
        stage1v2_settings,
        integration_clips,
        finite,
    )["tau"]


def class_masks(args, batch, d_apo: torch.Tensor, d_holo: torch.Tensor) -> Dict[str, torch.Tensor]:
    delta = d_holo - d_apo
    node = batch.node_mask.bool()
    pocket = node & (batch.w_res > float(args.pocket_threshold))
    finite = pocket & torch.isfinite(d_apo) & torch.isfinite(d_holo) & (d_apo < 49.0) & (d_holo < 49.0)
    active = finite & (delta.abs() >= float(args.active_delta))
    formed = active & (d_apo > float(args.contact_dist)) & (d_holo <= float(args.contact_dist))
    released = active & (d_apo <= float(args.contact_dist)) & (d_holo > float(args.contact_dist))
    stable_contact = finite & (d_apo <= float(args.contact_dist)) & (d_holo <= float(args.contact_dist))
    stable_noncontact = finite & (d_apo > float(args.contact_dist)) & (d_holo > float(args.contact_dist))
    return {
        "all_pocket": finite,
        "active": active,
        "approach": active & (delta < 0),
        "release": active & (delta > 0),
        "formed_contact": formed,
        "released_contact": released,
        "stable_contact": stable_contact,
        "stable_noncontact": stable_noncontact,
    }


def evaluate_path(
    args,
    fk_module,
    batch,
    rigids_apo: Rigid,
    rigids_holo: Rigid,
    path: Tuple[List[Rigid], List[torch.Tensor], List[float]],
    bridge_path: Tuple[List[Rigid], List[torch.Tensor], List[float]],
    d_apo: torch.Tensor,
    d_holo: torch.Tensor,
) -> Tuple[Dict[str, base.RunningStats], Dict[str, int]]:
    rigids_list, chi_list, t_list = path
    bridge_rigids, bridge_chi, bridge_t = bridge_path
    path_dists: List[torch.Tensor] = []
    for rigids_t, chi_t in zip(rigids_list, chi_list):
        atom14_t = base.torsions_to_atom14(
            fk_module, batch.torsion_apo[..., :3], chi_t, rigids_t, batch.aatype
        )
        path_dists.append(
            base.min_sc_ligand_dist(
                atom14_t["atom14_pos"].float(),
                atom14_t["atom14_mask"].bool(),
                batch.lig_points.float(),
                batch.lig_mask.bool(),
                batch.node_mask.bool(),
            )
        )

    d_final = path_dists[-1]
    delta_target = d_holo - d_apo
    delta_actual = d_final - d_apo
    classes = class_masks(args, batch, d_apo, d_holo)
    cap = float(args.path_dist_cap)
    endpoint_abs = (d_final.clamp(max=cap) - d_holo.clamp(max=cap)).abs()
    improvement = (d_apo - d_holo).abs() - (d_final - d_holo).abs()

    path_mae = d_final.new_zeros(d_final.shape)
    for d_t, t_val in zip(path_dists, t_list):
        target_t = d_apo + float(t_val) * delta_target
        path_mae = path_mae + (d_t.clamp(max=cap) - target_t.clamp(max=cap)).abs()
    path_mae = path_mae / max(len(path_dists), 1)

    path_dist_stack = torch.stack(path_dists, dim=0)
    path_min_ligand_dist = path_dist_stack.min(dim=0).values
    clash_severity = torch.relu(
        path_min_ligand_dist.new_tensor(float(args.ligand_clash_dist)) - path_min_ligand_dist
    )
    clash_proxy = (clash_severity > 0).float()

    chi_holo = batch.torsion_holo[..., 3:7]
    gap_total = []
    step_total = []
    step_chi = []
    step_trans = []
    for rigids_t, chi_t in zip(rigids_list, chi_list):
        total_gap, _, _, _ = base.state_gap_to_holo(
            rigids_t, chi_t, rigids_holo, chi_holo, batch.chi_mask
        )
        gap_total.append(total_gap)
    for idx in range(1, len(rigids_list)):
        total_step, trans_step, _, chi_step = base.state_step_motion(
            rigids_list[idx - 1],
            chi_list[idx - 1],
            rigids_list[idx],
            chi_list[idx],
            batch.chi_mask,
        )
        step_total.append(total_step)
        step_trans.append(trans_step)
        step_chi.append(chi_step)

    if step_total:
        step_total_stack = torch.stack(step_total, dim=0)
        path_length_total = step_total_stack.sum(dim=0)
        path_action_total = (step_total_stack ** 2).sum(dim=0)
        velocity_spike = base.velocity_spike_ratio(step_total)
        late_fraction_10 = base.late_fraction(step_total, t_list, 0.9)
    else:
        path_length_total = d_final.new_zeros(d_final.shape)
        path_action_total = d_final.new_zeros(d_final.shape)
        velocity_spike = d_final.new_zeros(d_final.shape)
        late_fraction_10 = d_final.new_zeros(d_final.shape)

    bridge_dev = []
    for rigids_t, chi_t, rigids_b, chi_b, t_val in zip(
        rigids_list, chi_list, bridge_rigids, bridge_chi, bridge_t
    ):
        if not (1e-6 < float(t_val) < 1.0 - 1e-6):
            continue
        total_dev, _, _, _ = base.state_step_motion(
            rigids_b, chi_b, rigids_t, chi_t, batch.chi_mask
        )
        bridge_dev.append(total_dev)
    if bridge_dev:
        bridge_dev_mean = torch.stack(bridge_dev, dim=0).mean(dim=0)
        bridge_dev_max = torch.stack(bridge_dev, dim=0).max(dim=0).values
    else:
        bridge_dev_mean = d_final.new_zeros(d_final.shape)
        bridge_dev_max = d_final.new_zeros(d_final.shape)

    progress_auc = base.progress_auc_from_gaps(gap_total)
    stats = defaultdict(base.RunningStats)
    counts: Dict[str, int] = {}
    for name, mask in classes.items():
        counts[name] = int(mask.sum().item())
        stats[f"{name}/endpoint_abs_dist"].add(endpoint_abs, mask)
        stats[f"{name}/path_mae_dist"].add(path_mae, mask)
        stats[f"{name}/improvement_to_holo"].add(improvement, mask)
        base.add_direction(stats, f"{name}/direction_acc", delta_actual, delta_target, mask)
        stats[f"{name}/path_min_ligand_dist"].add(path_min_ligand_dist, mask)
        stats[f"{name}/ligand_clash_proxy"].add(clash_proxy, mask)
        stats[f"{name}/ligand_clash_severity"].add(clash_severity, mask)
        stats[f"{name}/path_length_total"].add(path_length_total, mask)
        stats[f"{name}/path_action_total"].add(path_action_total, mask)
        stats[f"{name}/velocity_spike_ratio"].add(velocity_spike, mask)
        stats[f"{name}/late_motion_fraction_10"].add(late_fraction_10, mask)
        stats[f"{name}/bridge_dev_total_mean"].add(bridge_dev_mean, mask)
        stats[f"{name}/bridge_dev_total_max"].add(bridge_dev_max, mask)
        stats[f"{name}/progress_auc_total"].add(progress_auc, mask)
    return stats, counts


def merge_stats(total_stats, batch_stats):
    for key, stat in batch_stats.items():
        total_stats[key].sum += stat.sum
        total_stats[key].count += stat.count


def summarize(stats: Dict[str, base.RunningStats]) -> Dict[str, Optional[float]]:
    return {key: stat.mean for key, stat in sorted(stats.items())}


def metrics_delta(metrics: Dict[str, Optional[float]], reference: Dict[str, Optional[float]]) -> Dict[str, float]:
    out = {}
    for key, value in metrics.items():
        ref = reference.get(key)
        if value is None or ref is None:
            continue
        out[key] = float(value) - float(ref)
    return out


def _safe_sample_id(sample_id: str) -> str:
    return str(sample_id).replace("/", "_").replace("\\", "_")


def export_phase_teacher_batch(
    args: argparse.Namespace,
    batch,
    projection: Dict[str, torch.Tensor],
    class_map: Dict[str, torch.Tensor],
    output_dir: Path,
) -> List[Dict[str, object]]:
    """Write explicit pseudo-teacher phase targets without changing legacy caches."""
    tau = projection["tau"].detach().float().cpu()
    confidence = projection["confidence"].detach().float().cpu()
    projection_cost = projection["projection_cost"].detach().float().cpu()
    identity_cost = projection["identity_cost"].detach().float().cpu()
    t_values = np.linspace(0.0, 1.0, tau.shape[0], dtype=np.float32)
    records: List[Dict[str, object]] = []

    for b, sample_id in enumerate(batch.pdb_ids):
        n_res = int(batch.n_residues[b])
        output_path = output_dir / f"{_safe_sample_id(sample_id)}.npz"
        if args.phase_teacher_skip_existing and output_path.is_file():
            records.append({
                "sample_id": str(sample_id),
                "relative_path": output_path.name,
                "n_residues": n_res,
                "status": "exists",
            })
            continue

        masks = {
            name: class_map[name][b, :n_res].detach().bool().cpu().numpy()
            for name in (
                "all_pocket",
                "active",
                "approach",
                "formed_contact",
                "release",
            )
        }
        confidence_np = confidence[:, b, :n_res].numpy().astype(np.float32)
        np.savez_compressed(
            output_path,
            schema_version=np.array("phase_teacher_v1"),
            source=np.array("free_flow_projected_pseudo_teacher"),
            sample_id=np.array(str(sample_id)),
            n_residues=np.array(n_res, dtype=np.int32),
            teacher_checkpoint=np.array(str(args.free_flow_checkpoint or args.checkpoint)),
            reference_bridge_mode=np.array(str(args.reference_bridge_mode)),
            tau_transition_weight=np.array(float(args.tau_transition_weight), dtype=np.float32),
            t_values=t_values,
            tau_target=tau[:, b, :n_res].numpy().astype(np.float32),
            phase_confidence=confidence_np,
            projection_cost=projection_cost[:, b, :n_res].numpy().astype(np.float32),
            identity_cost=identity_cost[:, b, :n_res].numpy().astype(np.float32),
            node_mask=batch.node_mask[b, :n_res].detach().bool().cpu().numpy(),
            w_res=batch.w_res[b, :n_res].detach().float().cpu().numpy().astype(np.float32),
            pocket_mask=masks["all_pocket"],
            active_mask=masks["active"],
            approach_mask=masks["approach"],
            formed_contact_mask=masks["formed_contact"],
            release_mask=masks["release"],
        )
        contact_event = masks["approach"] | masks["formed_contact"]
        weighted = confidence_np[:, contact_event]
        records.append({
            "sample_id": str(sample_id),
            "relative_path": output_path.name,
            "n_residues": n_res,
            "contact_event_residues": int(contact_event.sum()),
            "mean_contact_event_confidence": (
                float(weighted.mean()) if weighted.size else 0.0
            ),
            "status": "written",
        })
    return records


def main() -> None:
    args = parse_args()
    methods = parse_methods(args.methods)
    if "freeflow_projected" in methods and not (args.free_flow_checkpoint or args.checkpoint):
        raise ValueError("freeflow_projected requires --checkpoint or --free_flow_checkpoint")
    if int(args.n_path_steps) <= 0:
        raise ValueError("--n_path_steps must be > 0")
    if int(args.n_tau_grid) < 3:
        raise ValueError("--n_tau_grid must be >= 3")
    if float(args.tau_transition_weight) < 0.0:
        raise ValueError("--tau_transition_weight must be non-negative")
    if args.phase_teacher_cache_dir and "freeflow_projected" not in methods:
        raise ValueError("--phase_teacher_cache_dir requires freeflow_projected in --methods")

    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    ckpt, config, model, interaction_settings, stage1v2_settings = load_model_for_checkpoint(
        args, args.checkpoint, device, args.split
    )
    del ckpt
    model.eval()
    integration_clips = base.resolve_integration_clips(args, config)

    free_model = model
    free_config = config
    free_interaction_settings = interaction_settings
    free_stage1v2_settings = stage1v2_settings
    if "freeflow_projected" in methods and args.free_flow_checkpoint:
        _, free_config, free_model, free_interaction_settings, free_stage1v2_settings = load_model_for_checkpoint(
            args, args.free_flow_checkpoint, device, args.split
        )
        args.config = free_config
        integration_clips = base.resolve_integration_clips(args, free_config)

    # Restore the primary config for dataloader construction.
    args.config = config
    fk_module = create_openfold_fk().to(device)
    fk_module.eval()
    loader = create_stage2_dataloader(
        args.data_dir,
        split=args.split,
        batch_size=args.batch_size,
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

    tau_grid = torch.linspace(0.0, 1.0, int(args.n_tau_grid), device=device)
    method_stats = {method: defaultdict(base.RunningStats) for method in methods}
    method_counts = {method: defaultdict(int) for method in methods}
    group_counts_total = defaultdict(int)
    total_batches = 0
    total_samples = 0
    phase_teacher_records: List[Dict[str, object]] = []
    phase_teacher_dir = None
    if args.phase_teacher_cache_dir:
        phase_teacher_dir = Path(args.phase_teacher_cache_dir)
        phase_teacher_dir.mkdir(parents=True, exist_ok=True)

    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(loader, desc="Bridge time-warp", ncols=120)):
            if args.max_batches is not None and batch_idx >= int(args.max_batches):
                break
            batch = base.batch_to_device(batch, device)
            rigids_apo = base.build_rigids_from_backbone(batch.N_apo, batch.Ca_apo, batch.C_apo, batch.node_mask)
            rigids_holo = base.build_rigids_from_backbone(batch.N_holo, batch.Ca_holo, batch.C_holo, batch.node_mask)
            n_steps = int(args.n_path_steps)
            bridge_path = build_tau_grid_path(
                batch,
                rigids_apo,
                rigids_holo,
                identity_tau_seq(batch, n_steps),
                args.reference_bridge_mode,
            )

            atom14_apo = base.torsions_to_atom14(
                fk_module, batch.torsion_apo[..., :3], batch.torsion_apo[..., 3:7], rigids_apo, batch.aatype
            )
            atom14_holo = base.torsions_to_atom14(
                fk_module, batch.torsion_holo[..., :3], batch.torsion_holo[..., 3:7], rigids_holo, batch.aatype
            )
            d_apo = base.min_sc_ligand_dist(
                atom14_apo["atom14_pos"].float(),
                atom14_apo["atom14_mask"].bool(),
                batch.lig_points.float(),
                batch.lig_mask.bool(),
                batch.node_mask.bool(),
            )
            d_holo = base.min_sc_ligand_dist(
                atom14_holo["atom14_pos"].float(),
                atom14_holo["atom14_mask"].bool(),
                batch.lig_points.float(),
                batch.lig_mask.bool(),
                batch.node_mask.bool(),
            )
            class_map = class_masks(args, batch, d_apo, d_holo)
            finite = class_map["all_pocket"]
            d_bridge_grid, grid_rigids, grid_chi = compute_bridge_distance_grid(
                fk_module,
                batch,
                rigids_apo,
                rigids_holo,
                tau_grid,
                args.reference_bridge_mode,
            )
            cost = path_cost_against_distance_schedule(
                d_bridge_grid, d_apo, d_holo, n_steps, float(args.path_dist_cap)
            )

            paths: Dict[str, Tuple[List[Rigid], List[torch.Tensor], List[float]]] = {}
            if "pure_bridge" in methods:
                paths["pure_bridge"] = bridge_path
            if "oracle_global" in methods:
                tau = build_oracle_global_tau(cost, finite, tau_grid)
                paths["oracle_global"] = build_tau_grid_path(
                    batch, rigids_apo, rigids_holo, tau, args.reference_bridge_mode
                )
            if "oracle_group" in methods:
                group_masks = build_group_masks(
                    finite, d_apo, d_holo, float(args.active_delta), float(args.contact_dist)
                )
                tau, group_counts = build_oracle_group_tau(cost, finite, group_masks, tau_grid, n_steps)
                for name, count in group_counts.items():
                    group_counts_total[name] += int(count)
                paths["oracle_group"] = build_tau_grid_path(
                    batch, rigids_apo, rigids_holo, tau, args.reference_bridge_mode
                )
            if "oracle_residue" in methods:
                tau = build_oracle_residue_tau(cost, finite, tau_grid, n_steps)
                paths["oracle_residue"] = build_tau_grid_path(
                    batch, rigids_apo, rigids_holo, tau, args.reference_bridge_mode
                )
            if "freeflow_projected" in methods:
                args.config = free_config
                n_free_steps = int(args.n_free_flow_steps or n_steps)
                if n_free_steps != n_steps:
                    raise ValueError("freeflow_projected currently requires n_free_flow_steps == n_path_steps")
                projection = build_free_flow_projection(
                    args,
                    free_model,
                    fk_module,
                    batch,
                    rigids_apo,
                    rigids_holo,
                    grid_rigids,
                    grid_chi,
                    n_steps,
                    tau_grid,
                    free_interaction_settings,
                    free_stage1v2_settings,
                    integration_clips,
                    finite,
                )
                tau = projection["tau"]
                args.config = config
                paths["freeflow_projected"] = build_tau_grid_path(
                    batch, rigids_apo, rigids_holo, tau, args.reference_bridge_mode
                )
                if phase_teacher_dir is not None:
                    phase_teacher_records.extend(
                        export_phase_teacher_batch(
                            args,
                            batch,
                            projection,
                            class_map,
                            phase_teacher_dir,
                        )
                    )

            for method, path in paths.items():
                stats, counts = evaluate_path(
                    args,
                    fk_module,
                    batch,
                    rigids_apo,
                    rigids_holo,
                    path,
                    bridge_path,
                    d_apo,
                    d_holo,
                )
                merge_stats(method_stats[method], stats)
                for key, value in counts.items():
                    method_counts[method][key] += int(value)

            total_batches += 1
            total_samples += int(batch.esm.shape[0])

    summaries = {}
    pure_metrics = summarize(method_stats["pure_bridge"]) if "pure_bridge" in method_stats else {}
    for method in methods:
        metrics = summarize(method_stats[method])
        summaries[method] = {
            "metrics": metrics,
            "delta_vs_pure_bridge": metrics_delta(metrics, pure_metrics) if pure_metrics else {},
            "counts": dict(method_counts[method]),
        }

    output = {
        "checkpoint": args.checkpoint,
        "free_flow_checkpoint": args.free_flow_checkpoint or args.checkpoint,
        "data_dir": args.data_dir,
        "split": args.split,
        "valid_samples_file": args.valid_samples_file,
        "methods": list(methods),
        "settings": {
            "n_path_steps": int(args.n_path_steps),
            "n_tau_grid": int(args.n_tau_grid),
            "tau_transition_weight": float(args.tau_transition_weight),
            "reference_bridge_mode": args.reference_bridge_mode,
            "active_delta": float(args.active_delta),
            "contact_dist": float(args.contact_dist),
            "path_dist_cap": float(args.path_dist_cap),
            "ligand_clash_dist": float(args.ligand_clash_dist),
            "pocket_threshold": float(args.pocket_threshold),
        },
        "batches": total_batches,
        "samples": total_samples,
        "group_counts": dict(group_counts_total),
        "phase_teacher_cache": {
            "directory": str(phase_teacher_dir) if phase_teacher_dir else None,
            "records": len(phase_teacher_records),
            "written": sum(r["status"] == "written" for r in phase_teacher_records),
        },
        "results": summaries,
    }
    if phase_teacher_dir is not None:
        manifest_path = phase_teacher_dir / "manifest.jsonl"
        manifest_path.write_text(
            "".join(json.dumps(record, sort_keys=True) + "\n" for record in phase_teacher_records)
        )
    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(output, indent=2, sort_keys=True))
        print(f"Saved: {out_path}")
    else:
        print(json.dumps(output, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
