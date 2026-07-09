#!/usr/bin/env python3
"""Benchmark Stage-2 induced-fit trajectory reliability.

This evaluator compares a trained Stage-2 path against a strong endpoint-aware
baseline: cubic interpolation in per-residue SE(3) frames and chi angles.  The
goal is not docking accuracy; it is whether generated intermediate structures
are more usable than a direct apo->holo interpolation under geometry/contact
audits.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, List, Optional, Tuple

import torch
from tqdm import tqdm

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))
flash_ipa_path = project_root / "vendor" / "flash_ipa" / "src"
if flash_ipa_path.exists():
    sys.path.insert(0, str(flash_ipa_path))

from evaluate_stage2_transition_paths import (  # noqa: E402
    batch_to_device,
    build_model_config_for_checkpoint,
    build_rigids_from_backbone,
    combined_prior_features,
    compute_interaction_prior_feature,
    integrate_path,
    load_model_state_allow_timewarp_head,
    min_sc_ligand_dist,
    rigid_to_rt,
    resolve_integration_clips,
    rt_to_rigid,
    torsions_to_atom14,
)
from flash_ipa.rigid import Rigid  # noqa: E402
from src.stage1.models.fk_openfold import create_openfold_fk  # noqa: E402
from src.stage1.modules.losses import clash_penalty  # noqa: E402
from src.stage2.datasets import create_stage2_dataloader  # noqa: E402
from src.stage2.models import TorsionFlowNet  # noqa: E402
from src.stage2.modules import (  # noqa: E402
    compute_peptide_loss,
    rigid_compose,
    rigid_inverse,
    se3_exp,
    se3_log,
    so3_log,
    wrap_to_pi,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate Stage-2 trajectory reliability")
    parser.add_argument("--checkpoint", required=True, help="Stage-2 checkpoint path")
    parser.add_argument("--data_dir", default="processed_data/triplets")
    parser.add_argument("--split", default="val", choices=["train", "val", "test"])
    parser.add_argument("--index_file", default=None)
    parser.add_argument("--valid_samples_file", default=None)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--n_integration_steps", type=int, default=None)
    parser.add_argument("--integration_chi_clip", type=float, default=None)
    parser.add_argument("--integration_rot_clip", type=float, default=None)
    parser.add_argument("--integration_trans_clip", type=float, default=None)
    parser.add_argument("--max_batches", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--interaction_prior_feature_mode",
        default=None,
        choices=["none", "prior", "prior_shuffled", "zero", "oracle_contact"],
    )
    parser.add_argument("--interaction_prior_ckpt", default=None)
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
    parser.add_argument("--pocket_threshold", type=float, default=0.3)
    parser.add_argument("--path_dist_cap", type=float, default=20.0)
    parser.add_argument("--clash_threshold", type=float, default=2.2)
    parser.add_argument("--clash_sample_size", type=int, default=512)
    parser.add_argument(
        "--include_boundary_residual",
        action="store_true",
        help=(
            "Also evaluate an endpoint-preserving path: SE(3)+chi interpolation "
            "plus a 4t(1-t)-gated residual from the model free trajectory."
        ),
    )
    parser.add_argument(
        "--include_boundary_native",
        action="store_true",
        help="Also evaluate the native boundary_residual parameterization used in training.",
    )
    parser.add_argument(
        "--boundary_residual_envelope",
        choices=["sin2", "poly"],
        default="sin2",
        help="Endpoint-zero envelope for --include_boundary_native.",
    )
    parser.add_argument("--boundary_residual_scale", type=float, default=1.0)
    parser.add_argument("--output", default=None, help="optional JSON output path")
    parser.add_argument(
        "--per_sample_output",
        default=None,
        help="optional JSONL output with one row per sample and method",
    )
    return parser.parse_args()


class RunningMean:
    def __init__(self) -> None:
        self.sum = 0.0
        self.count = 0

    def add(self, values: torch.Tensor, mask: Optional[torch.Tensor] = None) -> None:
        values = values.detach().float()
        if mask is not None:
            mask = mask.bool()
            if not bool(mask.any()):
                return
            values = values[mask]
        else:
            values = values.reshape(-1)
        if values.numel() == 0:
            return
        values = values[torch.isfinite(values)]
        if values.numel() == 0:
            return
        self.sum += float(values.sum().item())
        self.count += int(values.numel())

    def add_scalar(self, value: torch.Tensor, weight: int = 1) -> None:
        value = value.detach().float()
        if not bool(torch.isfinite(value).all()):
            return
        self.sum += float(value.item()) * int(weight)
        self.count += int(weight)

    @property
    def mean(self):
        return self.sum / self.count if self.count > 0 else None


def build_cubic_reference_path(batch, rigids_apo: Rigid, rigids_holo: Rigid, n_steps: int):
    chi0 = batch.torsion_apo[..., 3:7]
    chi1 = batch.torsion_holo[..., 3:7]
    delta_chi = wrap_to_pi(chi1 - chi0)
    R0, t0 = rigid_to_rt(rigids_apo)
    R1, t1 = rigid_to_rt(rigids_holo)
    R0_inv, t0_inv = rigid_inverse(R0, t0)
    R_delta, t_delta = rigid_compose(R0_inv, t0_inv, R1, t1)
    xi = se3_log(R_delta, t_delta)

    rigids_list = []
    chi_list = []
    t_list = []
    for k in range(int(n_steps) + 1):
        t_val = k / max(int(n_steps), 1)
        gamma = 3.0 * t_val * t_val - 2.0 * t_val * t_val * t_val
        chi_t = wrap_to_pi(chi0 + gamma * delta_chi)
        R_inc, t_inc = se3_exp(xi * gamma)
        R_t, trans_t = rigid_compose(R0, t0, R_inc, t_inc)
        rigids_list.append(rt_to_rigid(R_t, trans_t))
        chi_list.append(chi_t)
        t_list.append(t_val)
    return rigids_list, chi_list, t_list


def build_boundary_residual_path(model_path, reference_path):
    model_rigids, model_chi, model_t = model_path
    ref_rigids, ref_chi, ref_t = reference_path
    if len(model_rigids) != len(ref_rigids) or len(model_chi) != len(ref_chi):
        raise ValueError("model and reference paths must have the same number of frames")

    rigids_list = []
    chi_list = []
    t_list = []
    for model_rigid, model_chi_t, ref_rigid, ref_chi_t, t_val, ref_t_val in zip(
        model_rigids,
        model_chi,
        ref_rigids,
        ref_chi,
        model_t,
        ref_t,
    ):
        if abs(float(t_val) - float(ref_t_val)) > 1e-6:
            raise ValueError("model and reference paths must share t values")
        beta = 4.0 * float(t_val) * (1.0 - float(t_val))

        R_ref, t_ref = rigid_to_rt(ref_rigid)
        R_model, t_model = rigid_to_rt(model_rigid)
        R_ref_inv, t_ref_inv = rigid_inverse(R_ref, t_ref)
        R_delta, t_delta = rigid_compose(R_ref_inv, t_ref_inv, R_model, t_model)
        xi_delta = se3_log(R_delta, t_delta)
        R_res, t_res = se3_exp(xi_delta * beta)
        R_pinned, t_pinned = rigid_compose(R_ref, t_ref, R_res, t_res)

        d_chi = wrap_to_pi(model_chi_t - ref_chi_t)
        chi_pinned = wrap_to_pi(ref_chi_t + beta * d_chi)

        rigids_list.append(rt_to_rigid(R_pinned, t_pinned))
        chi_list.append(chi_pinned)
        t_list.append(float(t_val))
    return rigids_list, chi_list, t_list


def boundary_envelope(name: str, t_val: float) -> float:
    if name == "poly":
        return 4.0 * float(t_val) * (1.0 - float(t_val))
    return math.sin(math.pi * float(t_val)) ** 2


def build_boundary_native_path(
    args,
    model,
    fk_module,
    batch,
    reference_path,
    interaction_settings: Dict[str, object],
    stage1v2_settings: Dict[str, object],
    integration_clips: Dict[str, float],
):
    ref_rigids, ref_chi, ref_t = reference_path
    interaction_prior_feature = compute_interaction_prior_feature(
        args, args.config, batch, fk_module, ref_rigids[0], ref_rigids[-1]
    )
    prior_features = combined_prior_features(
        args,
        args.config,
        batch,
        interaction_prior_feature,
        stage1v2_settings,
        interaction_settings,
    )
    rigids_list = []
    chi_list = []
    t_list = []
    bsz = batch.esm.shape[0]
    scale = float(args.boundary_residual_scale)
    for rigids_t, chi_t_ref, t_val in zip(ref_rigids, ref_chi, ref_t):
        beta = boundary_envelope(args.boundary_residual_envelope, float(t_val))
        if beta == 0.0:
            rigids_list.append(rigids_t)
            chi_list.append(chi_t_ref)
            t_list.append(float(t_val))
            continue
        t_tensor = torch.full((bsz,), float(t_val), device=chi_t_ref.device)
        out = model(
            chi=chi_t_ref,
            rigids=rigids_t,
            esm=batch.esm,
            lig_points=batch.lig_points,
            lig_types=batch.lig_types,
            lig_mask=batch.lig_mask,
            w_res=batch.w_res,
            t=t_tensor,
            node_mask=batch.node_mask,
            nma_features=batch.nma_features,
            interaction_prior=prior_features,
            current_step=10**9,
        )
        d_chi = out["d_chi"].float().clamp(
            min=-float(integration_clips["chi"]), max=float(integration_clips["chi"])
        ) if float(integration_clips["chi"]) > 0.0 else out["d_chi"].float()
        d_rot = out["d_rigid_rot"].float().clamp(
            min=-float(integration_clips["rot"]), max=float(integration_clips["rot"])
        ) if float(integration_clips["rot"]) > 0.0 else out["d_rigid_rot"].float()
        d_trans = out["d_rigid_trans"].float().clamp(
            min=-float(integration_clips["trans"]), max=float(integration_clips["trans"])
        ) if float(integration_clips["trans"]) > 0.0 else out["d_rigid_trans"].float()
        weight = beta * scale
        R_ref, t_ref = rigid_to_rt(rigids_t)
        R_res, t_res = se3_exp(torch.cat([d_rot, d_trans], dim=-1) * weight)
        R_new, t_new = rigid_compose(R_ref, t_ref, R_res, t_res)
        rigids_list.append(rt_to_rigid(R_new, t_new))
        chi_list.append(wrap_to_pi(chi_t_ref + weight * d_chi))
        t_list.append(float(t_val))
    return rigids_list, chi_list, t_list


def terminal_projection_weight(name: str, t_val: float) -> float:
    t_scalar = min(max(float(t_val), 0.0), 1.0)
    if name == "quadratic":
        return t_scalar * t_scalar
    if name == "smoothstep":
        return 3.0 * t_scalar * t_scalar - 2.0 * t_scalar * t_scalar * t_scalar
    if name == "late_smoother":
        late_t = t_scalar**3
        return 10.0 * late_t**3 - 15.0 * late_t**4 + 6.0 * late_t**5
    return 10.0 * t_scalar**3 - 15.0 * t_scalar**4 + 6.0 * t_scalar**5


def project_terminal_path(reference_path, rigids_holo, chi_holo, schedule: str):
    rigids_list, chi_list, t_list = reference_path
    if not rigids_list or len(rigids_list) != len(chi_list):
        raise ValueError("terminal projection requires matching non-empty rigid/chi paths")

    final_R, final_t = rigid_to_rt(rigids_list[-1])
    holo_R, holo_t = rigid_to_rt(rigids_holo)
    final_R_inv, final_t_inv = rigid_inverse(final_R, final_t)
    delta_R, delta_t = rigid_compose(final_R_inv, final_t_inv, holo_R, holo_t)
    delta_xi = se3_log(delta_R, delta_t)
    delta_chi = wrap_to_pi(chi_holo - chi_list[-1])

    projected_rigids = []
    projected_chi = []
    for rigids_t, chi_t, t_val in zip(rigids_list, chi_list, t_list):
        weight = terminal_projection_weight(schedule, float(t_val))
        if weight <= 0.0:
            projected_rigids.append(rigids_t)
            projected_chi.append(chi_t)
            continue
        if weight >= 1.0:
            projected_rigids.append(rigids_holo)
            projected_chi.append(chi_holo)
            continue
        R_t, trans_t = rigid_to_rt(rigids_t)
        R_corr, t_corr = se3_exp(delta_xi * weight)
        R_proj, trans_proj = rigid_compose(R_t, trans_t, R_corr, t_corr)
        projected_rigids.append(rt_to_rigid(R_proj, trans_proj))
        projected_chi.append(wrap_to_pi(chi_t + weight * delta_chi))
    return projected_rigids, projected_chi, list(t_list)


def build_model_path(
    args,
    model,
    fk_module,
    batch,
    rigids_apo,
    rigids_holo,
    n_steps: int,
    interaction_settings: Dict[str, object],
    stage1v2_settings: Dict[str, object],
    integration_clips: Dict[str, float],
):
    interaction_prior_feature = compute_interaction_prior_feature(
        args, args.config, batch, fk_module, rigids_apo, rigids_holo
    )
    prior_features = combined_prior_features(
        args,
        args.config,
        batch,
        interaction_prior_feature,
        stage1v2_settings,
        interaction_settings,
    )
    model_path = integrate_path(
        model,
        batch,
        rigids_apo,
        batch.torsion_apo[..., 3:7],
        n_steps=n_steps,
        interaction_prior=prior_features,
        chi_clip=integration_clips["chi"],
        rot_clip=integration_clips["rot"],
        trans_clip=integration_clips["trans"],
    )
    if getattr(args.config, "path_parameterization", "flow") == "projected_flow":
        schedule = getattr(args.config, "terminal_projection_schedule", "smootherstep")
        return project_terminal_path(
            model_path,
            rigids_holo,
            batch.torsion_holo[..., 3:7],
            schedule,
        )
    return model_path


def class_masks(args, batch, d_apo: torch.Tensor, d_holo: torch.Tensor) -> Dict[str, torch.Tensor]:
    node = batch.node_mask.bool()
    pocket = node & (batch.w_res > float(args.pocket_threshold))
    finite = pocket & torch.isfinite(d_apo) & torch.isfinite(d_holo) & (d_apo < 49.0) & (d_holo < 49.0)
    delta = d_holo - d_apo
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


def add_direction(stats, key: str, actual_delta: torch.Tensor, target_delta: torch.Tensor, mask: torch.Tensor) -> None:
    ok = (actual_delta * target_delta) > 0
    stats[key].add(ok.float(), mask)


def add_path_smoothness(
    stats,
    prefix: str,
    rigids_list,
    chi_list,
    masks: Dict[str, torch.Tensor],
    chi_mask: torch.Tensor,
) -> None:
    for prev_rigid, rigid_t, prev_chi, chi_t in zip(rigids_list[:-1], rigids_list[1:], chi_list[:-1], chi_list[1:]):
        R_prev, t_prev = rigid_to_rt(prev_rigid)
        R_t, trans_t = rigid_to_rt(rigid_t)
        R_inv, t_inv = rigid_inverse(R_prev, t_prev)
        R_delta, t_delta = rigid_compose(R_inv, t_inv, R_t, trans_t)
        xi_delta = se3_log(R_delta, t_delta)
        step_frame = torch.linalg.norm(xi_delta, dim=-1)
        step_chi = torch.abs(wrap_to_pi(chi_t - prev_chi))
        for name, mask in masks.items():
            stats[prefix + f"{name}/smooth_frame_step_norm"].add(step_frame, mask)
            stats[prefix + f"{name}/smooth_chi_step_abs"].add(step_chi, mask.unsqueeze(-1) & chi_mask)


def add_geometry_scores(args, stats, prefix: str, batch, atom14_list: List[Dict]) -> None:
    for atom14 in atom14_list:
        atom14_pos = atom14["atom14_pos"].float().clamp(min=-1000.0, max=1000.0)
        atom14_mask = atom14["atom14_mask"].bool() & batch.node_mask.unsqueeze(-1)
        flat_atoms = atom14_pos.reshape(atom14_pos.shape[0], -1, 3)
        flat_mask = atom14_mask.reshape(atom14_mask.shape[0], -1)
        stats[prefix + "path/clash_penalty"].add_scalar(
            clash_penalty(
                flat_atoms,
                clash_threshold=float(args.clash_threshold),
                aatype=batch.aatype,
                atom_mask=flat_mask,
                sample_size=int(args.clash_sample_size),
            )
        )
        stats[prefix + "path/peptide_loss"].add_scalar(
            compute_peptide_loss(atom14_pos, atom14_mask, batch.node_mask)
        )


def _masked_mean_value(values: torch.Tensor, mask: torch.Tensor):
    mask = mask.bool()
    if not bool(mask.any()):
        return None
    vals = values.detach().float()[mask]
    vals = vals[torch.isfinite(vals)]
    if vals.numel() == 0:
        return None
    return float(vals.mean().item())


def _sample_id(batch, index: int) -> str:
    if hasattr(batch, "pdb_ids") and batch.pdb_ids:
        return str(batch.pdb_ids[index])
    return f"sample_{index}"


def build_per_sample_rows(
    args,
    method: str,
    batch,
    masks: Dict[str, torch.Tensor],
    endpoint_abs: torch.Tensor,
    path_mae: torch.Tensor,
    delta_actual: torch.Tensor,
    delta_target: torch.Tensor,
    frame_trans_err: torch.Tensor,
    frame_rot_err: torch.Tensor,
    chi_err: torch.Tensor,
    atom_err: torch.Tensor,
    atom_overlap: torch.Tensor,
    pred_contact: torch.Tensor,
) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    bsz = int(batch.node_mask.shape[0])
    direction_ok = (delta_actual * delta_target) > 0
    chi_valid = batch.chi_mask.bool()

    for b in range(bsz):
        row: Dict[str, object] = {
            "sample_id": _sample_id(batch, b),
            "method": method,
            "n_residues": int(batch.n_residues[b]) if hasattr(batch, "n_residues") else None,
        }
        for name, mask in masks.items():
            mb = mask[b].bool()
            row[f"{name}/count"] = int(mb.sum().item())
            row[f"{name}/endpoint_abs_dist_A"] = _masked_mean_value(endpoint_abs[b], mb)
            row[f"{name}/path_mae_dist_A"] = _masked_mean_value(path_mae[b], mb)
            row[f"{name}/direction_acc"] = _masked_mean_value(direction_ok[b].float(), mb)
            row[f"{name}/frame_trans_err_A"] = _masked_mean_value(frame_trans_err[b], mb)
            row[f"{name}/frame_rot_err_rad"] = _masked_mean_value(frame_rot_err[b], mb)
            row[f"{name}/chi_err_rad"] = _masked_mean_value(
                chi_err[b],
                mb.unsqueeze(-1) & chi_valid[b],
            )
            row[f"{name}/atom14_endpoint_err_A"] = _masked_mean_value(
                atom_err[b],
                mb.unsqueeze(-1) & atom_overlap[b],
            )
        formed = masks["formed_contact"][b].bool()
        released = masks["released_contact"][b].bool()
        stable_contact = masks["stable_contact"][b].bool()
        row["formed_contact/recall"] = _masked_mean_value(pred_contact[b].float(), formed)
        row["released_contact/release_success"] = _masked_mean_value((~pred_contact[b]).float(), released)
        row["stable_contact/retention"] = _masked_mean_value(pred_contact[b].float(), stable_contact)
        rows.append(row)
    return rows


def evaluate_path(
    args,
    stats: Dict[str, RunningMean],
    counts: Dict[str, int],
    method: str,
    fk_module,
    batch,
    rigids_apo,
    rigids_holo,
    rigids_list,
    chi_list,
    t_list,
) -> List[Dict[str, object]]:
    atom14_apo = torsions_to_atom14(
        fk_module, batch.torsion_apo[..., :3], batch.torsion_apo[..., 3:7], rigids_apo, batch.aatype
    )
    atom14_holo = torsions_to_atom14(
        fk_module, batch.torsion_holo[..., :3], batch.torsion_holo[..., 3:7], rigids_holo, batch.aatype
    )
    d_apo = min_sc_ligand_dist(
        atom14_apo["atom14_pos"].float(),
        atom14_apo["atom14_mask"].bool(),
        batch.lig_points.float(),
        batch.lig_mask.bool(),
        batch.node_mask.bool(),
    )
    d_holo = min_sc_ligand_dist(
        atom14_holo["atom14_pos"].float(),
        atom14_holo["atom14_mask"].bool(),
        batch.lig_points.float(),
        batch.lig_mask.bool(),
        batch.node_mask.bool(),
    )
    masks = class_masks(args, batch, d_apo, d_holo)
    if method == "model":
        for name, mask in masks.items():
            counts[name] += int(mask.sum().item())

    atom14_list = [
        torsions_to_atom14(fk_module, batch.torsion_apo[..., :3], chi_t, rigids_t, batch.aatype)
        for rigids_t, chi_t in zip(rigids_list, chi_list)
    ]
    path_dists = [
        min_sc_ligand_dist(
            atom14_t["atom14_pos"].float(),
            atom14_t["atom14_mask"].bool(),
            batch.lig_points.float(),
            batch.lig_mask.bool(),
            batch.node_mask.bool(),
        )
        for atom14_t in atom14_list
    ]

    prefix = f"{method}/"
    d_final = path_dists[-1]
    delta_target = d_holo - d_apo
    delta_actual = d_final - d_apo
    cap = float(args.path_dist_cap)
    endpoint_abs = (d_final.clamp(max=cap) - d_holo.clamp(max=cap)).abs()
    path_mae = d_final.new_zeros(d_final.shape)
    for d_t, t_val in zip(path_dists, t_list):
        target_t = d_apo + float(t_val) * delta_target
        path_mae = path_mae + (d_t.clamp(max=cap) - target_t.clamp(max=cap)).abs()
    path_mae = path_mae / max(len(path_dists), 1)

    R_final, t_final = rigid_to_rt(rigids_list[-1])
    R_holo, t_holo = rigid_to_rt(rigids_holo)
    frame_trans_err = torch.linalg.norm(t_final - t_holo, dim=-1)
    rot_residual = R_final.transpose(-2, -1) @ R_holo
    frame_rot_err = torch.linalg.norm(so3_log(rot_residual), dim=-1)
    chi_err = torch.abs(wrap_to_pi(chi_list[-1] - batch.torsion_holo[..., 3:7]))

    final_pos = atom14_list[-1]["atom14_pos"].float()
    holo_pos = atom14_holo["atom14_pos"].float()
    atom_overlap = (
        atom14_list[-1]["atom14_mask"].bool()
        & atom14_holo["atom14_mask"].bool()
        & batch.node_mask.unsqueeze(-1)
    )
    atom_err = torch.linalg.norm(final_pos - holo_pos, dim=-1)

    pred_contact = d_final <= float(args.contact_dist)
    for name, mask in masks.items():
        stats[prefix + f"{name}/endpoint_abs_dist_A"].add(endpoint_abs, mask)
        stats[prefix + f"{name}/path_mae_dist_A"].add(path_mae, mask)
        stats[prefix + f"{name}/frame_trans_err_A"].add(frame_trans_err, mask)
        stats[prefix + f"{name}/frame_rot_err_rad"].add(frame_rot_err, mask)
        stats[prefix + f"{name}/chi_err_rad"].add(chi_err, mask.unsqueeze(-1) & batch.chi_mask.bool())
        stats[prefix + f"{name}/atom14_endpoint_err_A"].add(atom_err, mask.unsqueeze(-1) & atom_overlap)
        add_direction(stats, prefix + f"{name}/direction_acc", delta_actual, delta_target, mask)
        if name == "formed_contact":
            stats[prefix + "formed_contact/recall"].add(pred_contact.float(), mask)
        if name == "released_contact":
            stats[prefix + "released_contact/release_success"].add((~pred_contact).float(), mask)
        if name == "stable_contact":
            stats[prefix + "stable_contact/retention"].add(pred_contact.float(), mask)

    add_path_smoothness(
        stats,
        prefix,
        rigids_list,
        chi_list,
        {k: v for k, v in masks.items() if k in {"all_pocket", "active"}},
        batch.chi_mask.bool(),
    )
    add_geometry_scores(args, stats, prefix, batch, atom14_list)
    return build_per_sample_rows(
        args,
        method,
        batch,
        masks,
        endpoint_abs,
        path_mae,
        delta_actual,
        delta_target,
        frame_trans_err,
        frame_rot_err,
        chi_err,
        atom_err,
        atom_overlap,
        pred_contact,
    )


def summarize(stats: Dict[str, RunningMean]) -> Dict[str, object]:
    return {
        "metrics": {key: stat.mean for key, stat in sorted(stats.items())},
        "metric_counts": {key: stat.count for key, stat in sorted(stats.items())},
    }


def main() -> None:
    args = parse_args()
    torch.manual_seed(int(args.seed))
    device = torch.device(args.device if args.device != "cuda" or torch.cuda.is_available() else "cpu")
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    config = ckpt.get("config", SimpleNamespace())
    args.config = config

    model_config, interaction_settings, stage1v2_settings = build_model_config_for_checkpoint(
        args, config, args.split
    )
    integration_clips = resolve_integration_clips(args, config)
    model = TorsionFlowNet(model_config).to(device)
    load_model_state_allow_timewarp_head(model, ckpt["model_state_dict"])
    model.eval()
    fk_module = create_openfold_fk().to(device)
    fk_module.eval()
    n_steps = int(args.n_integration_steps or getattr(config, "n_integration_steps", 5))

    loader = create_stage2_dataloader(
        args.data_dir,
        split=args.split,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        require_nma=bool(getattr(config, "use_nma", False)),
        index_file=args.index_file,
        valid_samples_file=args.valid_samples_file,
        stage1v2_posterior_cache_dir=stage1v2_settings["cache_dir"],
        stage1v2_posterior_feature_mode=stage1v2_settings["mode"],
        stage1v2_posterior_feature_names=stage1v2_settings["names_raw"],
        esm_num_layers=int(getattr(config, "esm_num_layers", 1)),
    )

    stats: Dict[str, RunningMean] = defaultdict(RunningMean)
    counts: Dict[str, int] = defaultdict(int)
    per_sample_rows: List[Dict[str, object]] = []
    total_batches = 0
    total_samples = 0

    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(loader, desc="Trajectory reliability", ncols=120)):
            if args.max_batches is not None and batch_idx >= int(args.max_batches):
                break
            batch = batch_to_device(batch, device)
            rigids_apo = build_rigids_from_backbone(batch.N_apo, batch.Ca_apo, batch.C_apo, batch.node_mask)
            rigids_holo = build_rigids_from_backbone(batch.N_holo, batch.Ca_holo, batch.C_holo, batch.node_mask)
            model_path = build_model_path(
                args,
                model,
                fk_module,
                batch,
                rigids_apo,
                rigids_holo,
                n_steps,
                interaction_settings,
                stage1v2_settings,
                integration_clips,
            )
            cubic_path = build_cubic_reference_path(batch, rigids_apo, rigids_holo, n_steps)
            if args.include_boundary_residual:
                boundary_residual_path = build_boundary_residual_path(model_path, cubic_path)
            if args.include_boundary_native:
                boundary_native_path = build_boundary_native_path(
                    args,
                    model,
                    fk_module,
                    batch,
                    cubic_path,
                    interaction_settings,
                    stage1v2_settings,
                    integration_clips,
                )

            per_sample_rows.extend(
                evaluate_path(args, stats, counts, "model", fk_module, batch, rigids_apo, rigids_holo, *model_path)
            )
            if args.include_boundary_residual:
                per_sample_rows.extend(
                    evaluate_path(
                        args,
                        stats,
                        counts,
                        "model_boundary_residual",
                        fk_module,
                        batch,
                        rigids_apo,
                        rigids_holo,
                        *boundary_residual_path,
                    )
                )
            if args.include_boundary_native:
                per_sample_rows.extend(
                    evaluate_path(
                        args,
                        stats,
                        counts,
                        "model_boundary_native",
                        fk_module,
                        batch,
                        rigids_apo,
                        rigids_holo,
                        *boundary_native_path,
                    )
                )
            per_sample_rows.extend(
                evaluate_path(
                    args,
                    stats,
                    counts,
                    "cubic_ref",
                    fk_module,
                    batch,
                    rigids_apo,
                    rigids_holo,
                    *cubic_path,
                )
            )
            total_batches += 1
            total_samples += int(batch.esm.shape[0])

    summary = {
        "checkpoint": args.checkpoint,
        "data_dir": args.data_dir,
        "split": args.split,
        "valid_samples_file": args.valid_samples_file,
        "samples": total_samples,
        "batches": total_batches,
        "n_integration_steps": n_steps,
        "integration_clips": integration_clips,
        "interaction_prior_feature_mode": interaction_settings["mode"],
        "stage1v2_posterior_feature_mode": stage1v2_settings["mode"],
        "stage1v2_posterior_cache_dir": stage1v2_settings["cache_dir"],
        "stage1v2_posterior_feature_names": list(stage1v2_settings["names"]),
        "stage1v2_posterior_feature_scale": stage1v2_settings["scale"],
        "include_boundary_residual": bool(args.include_boundary_residual),
        "include_boundary_native": bool(args.include_boundary_native),
        "boundary_residual_envelope": args.boundary_residual_envelope,
        "boundary_residual_scale": float(args.boundary_residual_scale),
        "counts": dict(counts),
        **summarize(stats),
    }
    text = json.dumps(summary, indent=2, sort_keys=True)
    print(text)
    if args.output:
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text + "\n", encoding="utf-8")
    if args.per_sample_output:
        out = Path(args.per_sample_output)
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w", encoding="utf-8") as handle:
            for row in per_sample_rows:
                handle.write(json.dumps(row, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
