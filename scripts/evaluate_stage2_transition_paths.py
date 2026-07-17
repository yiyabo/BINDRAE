#!/usr/bin/env python3
"""Residue-level Stage-2 ligand-distance transition evaluator.

This is an offline evaluator for induced-fit paths. It integrates a trained
Stage-2 checkpoint, decodes atom14 coordinates along the path, and scores
residue-ligand min-distance transitions instead of global contact averages.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, Iterable, List, Optional, Tuple

import torch
import torch.nn.functional as F
from tqdm import tqdm

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))
flash_ipa_path = project_root / "vendor" / "flash_ipa" / "src"
if flash_ipa_path.exists():
    sys.path.insert(0, str(flash_ipa_path))

from src.stage1.models.fk_openfold import create_openfold_fk, reorder_torsions_to_openfold  # noqa: E402
from src.stage1.models.interaction_prior import (  # noqa: E402
    load_interaction_prior,
    sidechain_atom_mask,
)
from src.stage2.datasets import create_stage2_dataloader  # noqa: E402
from src.stage2.models import TorsionFlowNet, TorsionFlowNetConfig  # noqa: E402
from src.stage2.modules.chain_internal import (  # noqa: E402
    interpolate_backbone_internal,
    project_anchored_pose_graph,
    project_peptide_frame_translations,
)
from src.stage2.modules import (  # noqa: E402
    endpoint_zero_envelope,
    project_block_tangent_normal,
    project_product_tangent_normal,
    rigid_compose,
    rigid_inverse,
    se3_exp,
    se3_log,
    wrap_to_pi,
)
from flash_ipa.rigid import Rigid, Rotation  # noqa: E402


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate residue-level Stage-2 path transitions")
    parser.add_argument("--checkpoint", required=True, help="Stage-2 checkpoint path")
    parser.add_argument("--data_dir", default="processed_data/triplets", help="Stage-2 data directory")
    parser.add_argument("--split", default="val", choices=["train", "val", "test"], help="dataset split")
    parser.add_argument("--valid_samples_file", default=None, help="sample filter file relative to data_dir")
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument(
        "--trust_prechecked_samples",
        action="store_true",
        help="Skip startup file/cache existence scans when valid sample files were prechecked",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--path_parameterization",
        default="checkpoint",
        choices=[
            "checkpoint",
            "flow",
            "projected_flow",
            "boundary_residual_v1",
            "boundary_residual",
            "pure_bridge",
            "cartesian_backbone_bridge_v1",
            "cartesian_peptide_projected_bridge_v1",
            "chain_internal_bridge_v1",
            "peptide_projected_bridge_v1",
            "pose_graph_projected_bridge_v1",
            "bridge_timewarp_v1",
            "phase_orthogonal_residual_v1",
            "phase_block_orthogonal_residual_v2",
        ],
        help="Path construction to evaluate; checkpoint uses checkpoint config",
    )
    parser.add_argument(
        "--boundary_residual_envelope",
        default=None,
        choices=["sin2", "poly"],
        help="override checkpoint endpoint-zero envelope for boundary residual paths",
    )
    parser.add_argument(
        "--boundary_residual_scale",
        type=float,
        default=None,
        help="override checkpoint scale applied to boundary residual model outputs",
    )
    parser.add_argument(
        "--terminal_projection_schedule",
        default=None,
        choices=["smoothstep", "smootherstep", "late_smoother", "quadratic"],
        help="override checkpoint correction schedule for projected_flow",
    )
    parser.add_argument("--time_warp_logit_scale", type=float, default=None)
    parser.add_argument("--time_warp_rate_eps", type=float, default=None)
    parser.add_argument("--time_warp_rate_clip", type=float, default=None)
    parser.add_argument(
        "--phase_residual_tau_mode",
        default=None,
        choices=["learned", "identity"],
    )
    parser.add_argument(
        "--phase_residual_bridge_mode",
        default=None,
        choices=["se3_geodesic", "cartesian_backbone"],
    )
    parser.add_argument(
        "--phase_residual_envelope",
        default=None,
        choices=["poly", "sin2"],
    )
    parser.add_argument("--phase_residual_scale", type=float, default=None)
    parser.add_argument("--phase_residual_rotation_metric_scale", type=float, default=None)
    parser.add_argument("--phase_residual_translation_metric_scale", type=float, default=None)
    parser.add_argument("--phase_residual_chi_metric_scale", type=float, default=None)
    parser.add_argument("--phase_residual_min_tangent_norm", type=float, default=None)
    parser.add_argument("--phase_residual_max_metric_norm", type=float, default=None)
    parser.add_argument("--peptide_projection_iterations", type=int, default=12)
    parser.add_argument("--peptide_projection_relaxation", type=float, default=0.75)
    parser.add_argument("--peptide_projection_anchor_strength", type=float, default=0.02)
    parser.add_argument("--peptide_projection_max_translation", type=float, default=2.0)
    parser.add_argument(
        "--peptide_projection_activation_loss_threshold", type=float, default=0.0
    )
    parser.add_argument("--pose_graph_iterations", type=int, default=20)
    parser.add_argument("--pose_graph_learning_rate", type=float, default=0.05)
    parser.add_argument("--pose_graph_edge_weight", type=float, default=1.0)
    parser.add_argument("--pose_graph_anchor_weight", type=float, default=0.1)
    parser.add_argument("--pose_graph_rotation_metric_scale", type=float, default=1.5)
    parser.add_argument("--pose_graph_max_rotation", type=float, default=0.5)
    parser.add_argument("--pose_graph_max_translation", type=float, default=2.0)
    parser.add_argument("--n_integration_steps", type=int, default=None)
    parser.add_argument("--integration_chi_clip", type=float, default=None)
    parser.add_argument("--integration_rot_clip", type=float, default=None)
    parser.add_argument("--integration_trans_clip", type=float, default=None)
    parser.add_argument("--max_batches", type=int, default=None)
    parser.add_argument("--interaction_prior_ckpt", default=None)
    parser.add_argument(
        "--interaction_prior_feature_mode",
        default=None,
        choices=["none", "prior", "prior_shuffled", "zero", "oracle_contact"],
        help="override checkpoint interaction prior feature mode",
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
        help="override checkpoint Stage-1-v2/oracle-motion feature mode",
    )
    parser.add_argument(
        "--stage1v2_posterior_cache_dir",
        default=None,
        help="override Stage-1-v2/oracle-motion feature cache/label directory for this split",
    )
    parser.add_argument(
        "--stage1v2_posterior_feature_names",
        default=None,
        help="override comma-separated Stage-1-v2/oracle-motion feature names",
    )
    parser.add_argument("--stage1v2_posterior_feature_scale", type=float, default=None)
    parser.add_argument("--active_delta", type=float, default=0.75, help="active switch distance delta in Angstrom")
    parser.add_argument("--contact_dist", type=float, default=4.5, help="hard contact class threshold in Angstrom")
    parser.add_argument("--path_dist_cap", type=float, default=20.0, help="cap distances for MAE statistics")
    parser.add_argument(
        "--ligand_clash_dist",
        type=float,
        default=2.2,
        help="side-chain atom to ligand distance threshold for path clash proxy in Angstrom",
    )
    parser.add_argument("--pocket_threshold", type=float, default=0.3)
    parser.add_argument("--output", default=None, help="optional JSON output path")
    return parser


def parse_args() -> argparse.Namespace:
    return build_arg_parser().parse_args()


def batch_to_device(batch, device: torch.device):
    for name in (
        "esm",
        "torsion_apo",
        "torsion_holo",
        "bb_mask",
        "chi_mask",
        "node_mask",
        "peptide_bond_mask",
        "N_apo",
        "Ca_apo",
        "C_apo",
        "N_holo",
        "Ca_holo",
        "C_holo",
        "lig_points",
        "lig_types",
        "lig_mask",
        "w_res",
        "aatype",
    ):
        setattr(batch, name, getattr(batch, name).to(device))
    if batch.stage1v2_posterior_features is not None:
        batch.stage1v2_posterior_features = batch.stage1v2_posterior_features.to(device)
    if batch.nma_features is not None:
        batch.nma_features = batch.nma_features.to(device)
    return batch


def parse_feature_names(raw: Optional[str]) -> Tuple[str, ...]:
    names = tuple(item.strip() for item in str(raw or "").split(",") if item.strip())
    if not names:
        raise ValueError("feature names cannot be empty")
    return names


def checkpoint_stage1v2_dir(config, split: str) -> Optional[str]:
    mode = getattr(config, "stage1v2_posterior_feature_mode", "none")
    if mode in {"none", "zero"}:
        return None
    if mode == "oracle_holo_truth":
        return (
            getattr(config, "stage1v2_train_label_dir", None)
            if split == "train"
            else getattr(config, "stage1v2_val_label_dir", None)
        )
    return (
        getattr(config, "stage1v2_train_cache_dir", None)
        if split == "train"
        else getattr(config, "stage1v2_val_cache_dir", None)
    )


def resolve_stage1v2_feature_settings(args, config, split: str) -> Dict[str, object]:
    mode = args.stage1v2_posterior_feature_mode
    if mode is None:
        mode = getattr(config, "stage1v2_posterior_feature_mode", "none")
    mode = str(mode or "none")
    names_raw = (
        args.stage1v2_posterior_feature_names
        if args.stage1v2_posterior_feature_names is not None
        else getattr(
            config,
            "stage1v2_posterior_feature_names",
            "contact_prob,active_prob,approach_prob,release_prob,confidence,"
            "teacher_min_dist_pred_norm,signed_delta_dist_pred_norm",
        )
    )
    names = parse_feature_names(names_raw)
    scale = (
        float(args.stage1v2_posterior_feature_scale)
        if args.stage1v2_posterior_feature_scale is not None
        else float(getattr(config, "stage1v2_posterior_feature_scale", 1.0))
    )
    cache_dir = args.stage1v2_posterior_cache_dir
    if cache_dir is None:
        cache_dir = checkpoint_stage1v2_dir(config, split)
    dim = len(names) if mode != "none" else 0
    if mode not in {"none", "zero"} and not cache_dir:
        raise ValueError(
            f"stage1v2_posterior_feature_mode={mode} requires --stage1v2_posterior_cache_dir "
            "or a cache/label dir stored in the checkpoint config"
        )
    return {
        "mode": mode,
        "names": names,
        "names_raw": ",".join(names),
        "scale": scale,
        "cache_dir": cache_dir,
        "dim": dim,
    }


def resolve_interaction_prior_feature_settings(args, config) -> Dict[str, object]:
    mode = args.interaction_prior_feature_mode or getattr(config, "interaction_prior_feature_mode", "none")
    mode = str(mode or "none")
    scale = (
        float(args.interaction_prior_feature_scale)
        if args.interaction_prior_feature_scale is not None
        else float(getattr(config, "interaction_prior_feature_scale", 1.0))
    )
    dim = 0 if mode == "none" else 1
    return {"mode": mode, "scale": scale, "dim": dim}


def combined_prior_features(
    args,
    config,
    batch,
    interaction_prior_feature: Optional[torch.Tensor],
    stage1v2_settings: Dict[str, object],
    interaction_settings: Dict[str, object],
) -> Optional[torch.Tensor]:
    features = []
    if int(interaction_settings["dim"]) > 0:
        if interaction_prior_feature is None:
            interaction_prior_feature = batch.w_res.new_zeros(batch.w_res.shape)
        if interaction_prior_feature.ndim == 2:
            interaction_prior_feature = interaction_prior_feature.unsqueeze(-1)
        if interaction_prior_feature.shape[-1] != int(interaction_settings["dim"]):
            raise ValueError(
                "interaction prior feature dim mismatch: "
                f"{interaction_prior_feature.shape[-1]} != {interaction_settings['dim']}"
            )
        features.append(interaction_prior_feature.detach().float() * float(interaction_settings["scale"]))

    if int(stage1v2_settings["dim"]) > 0:
        stage1v2_features = batch.stage1v2_posterior_features
        if stage1v2_features is None:
            stage1v2_features = batch.w_res.new_zeros((*batch.w_res.shape, int(stage1v2_settings["dim"])))
        if stage1v2_features.shape[-1] != int(stage1v2_settings["dim"]):
            raise ValueError(
                "Stage-1-v2/oracle feature dim mismatch: "
                f"{stage1v2_features.shape[-1]} != {stage1v2_settings['dim']}"
            )
        features.append(stage1v2_features.detach().float() * float(stage1v2_settings["scale"]))

    if not features:
        return None
    return torch.cat(features, dim=-1) * batch.node_mask.unsqueeze(-1).float()


def esm_gate_context(config, batch, stage1v2_settings: Dict[str, object]) -> Optional[torch.Tensor]:
    mode = str(getattr(config, "esm_gate_context_mode", "none") or "none")
    if mode == "none":
        return None
    if mode != "pocket_motion":
        raise ValueError(f"Unsupported esm_gate_context_mode={mode}")
    motion = batch.w_res.new_zeros(batch.w_res.shape)
    features = batch.stage1v2_posterior_features
    names = tuple(stage1v2_settings.get("names", ()))
    if features is not None and "motion_active" in names:
        motion = features[..., names.index("motion_active")].detach().float()
    context = torch.stack([batch.w_res.detach().float(), motion], dim=-1)
    return context * batch.node_mask.unsqueeze(-1).float()


def build_model_config_for_checkpoint(args, config, split: str) -> Tuple[TorsionFlowNetConfig, Dict[str, object], Dict[str, object]]:
    interaction_settings = resolve_interaction_prior_feature_settings(args, config)
    stage1v2_settings = resolve_stage1v2_feature_settings(args, config, split)
    prior_dim = int(interaction_settings["dim"]) + int(stage1v2_settings["dim"])
    esm_gate_context_dim = 2 if str(getattr(config, "esm_gate_context_mode", "none")) == "pocket_motion" else 0
    model_config = TorsionFlowNetConfig(
        esm_fusion_enabled=bool(getattr(config, "esm_fusion_enabled", False)),
        esm_num_layers=int(getattr(config, "esm_num_layers", 1)),
        esm_fusion_mode=str(getattr(config, "esm_fusion_mode", "sum")),
        esm_layer_dropout=float(getattr(config, "esm_layer_dropout", 0.0)),
        esm_gate_bias=float(getattr(config, "esm_gate_bias", -3.0)),
        esm_gate_context_dim=esm_gate_context_dim,
        nma_dim=int(getattr(config, "nma_dim", 0)),
        stage1_chi_feature_scale=float(getattr(config, "stage1_chi_feature_scale", 1.0)),
        interaction_prior_feature_dim=prior_dim,
        interaction_prior_feature_scale=1.0,
        repa_enabled=bool(getattr(config, "repa_enabled", False)),
        repa_dim=int(getattr(config, "repa_dim", 128)),
        repa_target_dim=int(stage1v2_settings["dim"]),
        phase_residual_enabled=(
            str(getattr(config, "path_parameterization", "flow"))
            in {
                "phase_orthogonal_residual_v1",
                "phase_block_orthogonal_residual_v2",
            }
        ),
        phase_residual_blockwise=(
            str(getattr(config, "path_parameterization", "flow"))
            == "phase_block_orthogonal_residual_v2"
        ),
        phase_residual_active_blocks=str(
            getattr(config, "phase_residual_active_blocks", "all")
        ),
    )
    return model_config, interaction_settings, stage1v2_settings


def load_model_state_allow_timewarp_head(model: TorsionFlowNet, state_dict: Dict[str, torch.Tensor]) -> None:
    result = model.load_state_dict(state_dict, strict=False)
    disallowed_missing = [
        key for key in result.missing_keys
        if not key.startswith("time_warp_head.")
    ]
    if disallowed_missing or result.unexpected_keys:
        raise RuntimeError(
            "Incompatible Stage-2 checkpoint. "
            f"missing={disallowed_missing}, unexpected={list(result.unexpected_keys)}"
        )


def resolve_integration_clips(args, config) -> Dict[str, float]:
    return {
        "chi": (
            float(args.integration_chi_clip)
            if args.integration_chi_clip is not None
            else float(getattr(config, "integration_chi_clip", 1.0))
        ),
        "rot": (
            float(args.integration_rot_clip)
            if args.integration_rot_clip is not None
            else float(getattr(config, "integration_rot_clip", 0.1))
        ),
        "trans": (
            float(args.integration_trans_clip)
            if args.integration_trans_clip is not None
            else float(getattr(config, "integration_trans_clip", 0.2))
        ),
    }


def maybe_clip(x: torch.Tensor, limit: float) -> torch.Tensor:
    limit = float(limit)
    if limit <= 0.0:
        return x
    return x.clamp(min=-limit, max=limit)


def resolve_path_parameterization(args, config) -> str:
    mode = str(args.path_parameterization or "checkpoint")
    if mode == "checkpoint":
        mode = str(getattr(config, "path_parameterization", "flow") or "flow")
    allowed = {
        "flow",
        "projected_flow",
        "boundary_residual_v1",
        "boundary_residual",
        "pure_bridge",
        "cartesian_backbone_bridge_v1",
        "cartesian_peptide_projected_bridge_v1",
        "chain_internal_bridge_v1",
        "peptide_projected_bridge_v1",
        "pose_graph_projected_bridge_v1",
        "bridge_timewarp_v1",
        "phase_orthogonal_residual_v1",
        "phase_block_orthogonal_residual_v2",
    }
    if mode not in allowed:
        raise ValueError(f"Unsupported path_parameterization={mode}")
    return mode


def boundary_residual_envelope(t_value: float, mode: str) -> float:
    t_scalar = min(max(float(t_value), 0.0), 1.0)
    if t_scalar <= 0.0 or t_scalar >= 1.0:
        return 0.0
    if mode == "poly":
        return 4.0 * t_scalar * (1.0 - t_scalar)
    return math.sin(math.pi * t_scalar) ** 2


def terminal_projection_weight(t_value: float, schedule: str) -> float:
    t_scalar = min(max(float(t_value), 0.0), 1.0)
    if schedule == "quadratic":
        return t_scalar * t_scalar
    if schedule == "smoothstep":
        return 3.0 * t_scalar * t_scalar - 2.0 * t_scalar * t_scalar * t_scalar
    if schedule == "late_smoother":
        late_t = t_scalar**3
        return 10.0 * late_t**3 - 15.0 * late_t**4 + 6.0 * late_t**5
    return 10.0 * t_scalar**3 - 15.0 * t_scalar**4 + 6.0 * t_scalar**5


def build_rigids_from_backbone(N, Ca, C, mask, eps: float = 1e-8) -> Rigid:
    e1 = C - Ca
    e1 = e1 / (torch.norm(e1, dim=-1, keepdim=True) + eps)
    u = N - Ca
    proj = (u * e1).sum(dim=-1, keepdim=True) * e1
    e2 = u - proj
    e2 = e2 / (torch.norm(e2, dim=-1, keepdim=True) + eps)
    e3 = torch.cross(e1, e2, dim=-1)
    R = torch.stack([e1, e2, e3], dim=-1)
    t = Ca
    if mask is not None:
        mask_exp = mask.unsqueeze(-1).unsqueeze(-1)
        eye = torch.eye(3, device=R.device, dtype=R.dtype).view(1, 1, 3, 3)
        R = torch.where(mask_exp, R, eye)
        t = torch.where(mask.unsqueeze(-1), t, torch.zeros_like(t))
    return Rigid(rots=Rotation(rot_mats=R), trans=t)


def rigid_to_rt(rigids: Rigid) -> Tuple[torch.Tensor, torch.Tensor]:
    return rigids.get_rots().get_rot_mats(), rigids.get_trans()


def rt_to_rigid(R: torch.Tensor, t: torch.Tensor) -> Rigid:
    return Rigid(rots=Rotation(rot_mats=R), trans=t)


def torsions_to_atom14(fk_module, phi_psi_omega, chi, rigids, aatype):
    bb_sincos = torch.stack([torch.sin(phi_psi_omega), torch.cos(phi_psi_omega)], dim=-1)
    chi_sincos = torch.stack([torch.sin(chi), torch.cos(chi)], dim=-1)
    torsions_sincos = reorder_torsions_to_openfold(torch.cat([bb_sincos, chi_sincos], dim=2))
    return fk_module(torsions_sincos, rigids, aatype)


def interpolate_backbone_torsions(batch, t_value: float) -> torch.Tensor:
    """Decode-only backbone-torsion bridge matching the endpoint frame bridge."""
    t_scalar = min(max(float(t_value), 0.0), 1.0)
    gamma = 3.0 * t_scalar * t_scalar - 2.0 * t_scalar * t_scalar * t_scalar
    torsion_apo = batch.torsion_apo[..., :3]
    torsion_holo = batch.torsion_holo[..., :3]
    return wrap_to_pi(
        torsion_apo + gamma * wrap_to_pi(torsion_holo - torsion_apo)
    )


def peptide_geometry_targets(
    n_coord: torch.Tensor,
    ca_coord: torch.Tensor,
    c_coord: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return C-N lengths and the two peptide angles for adjacent residues."""
    c_to_n = n_coord[:, 1:] - c_coord[:, :-1]
    ca_to_c = ca_coord[:, :-1] - c_coord[:, :-1]
    n_to_ca = ca_coord[:, 1:] - n_coord[:, 1:]
    direction = F.normalize(c_to_n, dim=-1)
    left_direction = F.normalize(ca_to_c, dim=-1)
    right_direction = F.normalize(n_to_ca, dim=-1)
    length = torch.linalg.norm(c_to_n, dim=-1)
    angle_cacn = torch.acos(
        (left_direction * direction).sum(dim=-1).clamp(-1.0, 1.0)
    )
    angle_cnca = torch.acos(
        (right_direction * -direction).sum(dim=-1).clamp(-1.0, 1.0)
    )
    return length, angle_cacn, angle_cnca


def interpolate_peptide_geometry_targets(batch, t_value: float):
    t_scalar = min(max(float(t_value), 0.0), 1.0)
    gamma = 3.0 * t_scalar * t_scalar - 2.0 * t_scalar * t_scalar * t_scalar
    apo_targets = peptide_geometry_targets(batch.N_apo, batch.Ca_apo, batch.C_apo)
    holo_targets = peptide_geometry_targets(batch.N_holo, batch.Ca_holo, batch.C_holo)
    return tuple(
        (1.0 - gamma) * apo_value + gamma * holo_value
        for apo_value, holo_value in zip(apo_targets, holo_targets)
    )


def interpolate_endpoints(
    batch,
    rigids_apo: Rigid,
    rigids_holo: Rigid,
    t_value: float,
) -> Tuple[Rigid, torch.Tensor]:
    t_scalar = float(t_value)
    gamma = 3.0 * t_scalar * t_scalar - 2.0 * t_scalar * t_scalar * t_scalar
    chi0 = batch.torsion_apo[..., 3:7]
    chi1 = batch.torsion_holo[..., 3:7]
    chi_t = wrap_to_pi(chi0 + gamma * wrap_to_pi(chi1 - chi0))

    R0, t0 = rigid_to_rt(rigids_apo)
    R1, t1 = rigid_to_rt(rigids_holo)
    R0_inv, t0_inv = rigid_inverse(R0, t0)
    R_delta, t_delta = rigid_compose(R0_inv, t0_inv, R1, t1)
    xi = se3_log(R_delta, t_delta)
    R_inc, t_inc = se3_exp(xi * gamma)
    R_t, trans_t = rigid_compose(R0, t0, R_inc, t_inc)
    return rt_to_rigid(R_t, trans_t), chi_t


def interpolate_endpoints_tensor(
    batch,
    rigids_apo: Rigid,
    rigids_holo: Rigid,
    tau: torch.Tensor,
) -> Tuple[Rigid, torch.Tensor]:
    if tau.ndim != 2:
        raise ValueError(f"tau must have shape [B, N], got {tuple(tau.shape)}")
    tau = tau.float().clamp(0.0, 1.0)
    gamma = (3.0 * tau * tau - 2.0 * tau * tau * tau).unsqueeze(-1)
    chi0 = batch.torsion_apo[..., 3:7]
    chi1 = batch.torsion_holo[..., 3:7]
    chi_t = wrap_to_pi(chi0 + gamma * wrap_to_pi(chi1 - chi0))

    R0, t0 = rigid_to_rt(rigids_apo)
    R1, t1 = rigid_to_rt(rigids_holo)
    R0_inv, t0_inv = rigid_inverse(R0, t0)
    R_delta, t_delta = rigid_compose(R0_inv, t0_inv, R1, t1)
    xi = se3_log(R_delta, t_delta)
    R_inc, t_inc = se3_exp(xi * gamma)
    R_t, trans_t = rigid_compose(R0, t0, R_inc, t_inc)
    return rt_to_rigid(R_t, trans_t), chi_t


def phase_interpolate_endpoints_tensor(
    batch,
    rigids_apo: Rigid,
    rigids_holo: Rigid,
    tau: torch.Tensor,
    bridge_mode: str,
) -> Tuple[Rigid, torch.Tensor]:
    if bridge_mode == "se3_geodesic":
        return interpolate_endpoints_tensor(batch, rigids_apo, rigids_holo, tau)
    if bridge_mode != "cartesian_backbone":
        raise ValueError(f"Unsupported phase residual bridge_mode={bridge_mode}")
    if tau.ndim != 2:
        raise ValueError(f"tau must have shape [B, N], got {tuple(tau.shape)}")
    tau = tau.float().clamp(0.0, 1.0)
    gamma = 3.0 * tau * tau - 2.0 * tau * tau * tau
    gamma_coord = gamma.unsqueeze(-1)
    n_coord = (1.0 - gamma_coord) * batch.N_apo + gamma_coord * batch.N_holo
    ca_coord = (1.0 - gamma_coord) * batch.Ca_apo + gamma_coord * batch.Ca_holo
    c_coord = (1.0 - gamma_coord) * batch.C_apo + gamma_coord * batch.C_holo
    rigids_t = build_rigids_from_backbone(
        n_coord, ca_coord, c_coord, batch.node_mask
    )
    chi0 = batch.torsion_apo[..., 3:7]
    chi1 = batch.torsion_holo[..., 3:7]
    chi_t = wrap_to_pi(
        chi0 + gamma_coord * wrap_to_pi(chi1 - chi0)
    )
    return rigids_t, chi_t


def phase_bridge_tangent(
    batch,
    rigids_apo: Rigid,
    rigids_holo: Rigid,
    tau: torch.Tensor,
    bridge_mode: str,
) -> torch.Tensor:
    if bridge_mode == "se3_geodesic":
        R0, trans0 = rigid_to_rt(rigids_apo)
        R1, trans1 = rigid_to_rt(rigids_holo)
        R0_inv, trans0_inv = rigid_inverse(R0, trans0)
        R_delta, trans_delta = rigid_compose(R0_inv, trans0_inv, R1, trans1)
        return se3_log(R_delta, trans_delta)

    epsilon = 1e-3
    tau_low = (tau - epsilon).clamp(0.0, 1.0)
    tau_high = (tau + epsilon).clamp(0.0, 1.0)
    low_rigids, _ = phase_interpolate_endpoints_tensor(
        batch, rigids_apo, rigids_holo, tau_low, bridge_mode
    )
    high_rigids, _ = phase_interpolate_endpoints_tensor(
        batch, rigids_apo, rigids_holo, tau_high, bridge_mode
    )
    low_rotation, low_translation = rigid_to_rt(low_rigids)
    high_rotation, high_translation = rigid_to_rt(high_rigids)
    low_inverse_rotation, low_inverse_translation = rigid_inverse(
        low_rotation, low_translation
    )
    delta_rotation, delta_translation = rigid_compose(
        low_inverse_rotation,
        low_inverse_translation,
        high_rotation,
        high_translation,
    )
    denominator = (tau_high - tau_low).unsqueeze(-1).clamp_min(1e-6)
    return se3_log(delta_rotation, delta_translation) / denominator


def min_sc_ligand_dist(
    atom14_pos: torch.Tensor,
    atom14_mask: torch.Tensor,
    lig_points: torch.Tensor,
    lig_mask: torch.Tensor,
    node_mask: torch.Tensor,
    residue_chunk: int = 64,
) -> torch.Tensor:
    bsz, n_res, n_atom, _ = atom14_pos.shape
    n_lig = lig_points.shape[1]
    out = atom14_pos.new_full((bsz, n_res), 50.0)
    if n_lig == 0:
        return out
    sc_mask = sidechain_atom_mask(atom14_mask.bool(), node_mask.bool())
    for start in range(0, n_res, max(int(residue_chunk), 1)):
        end = min(start + int(residue_chunk), n_res)
        c = end - start
        dists = torch.cdist(
            atom14_pos[:, start:end].reshape(bsz * c, n_atom, 3),
            lig_points[:, None, :, :].expand(bsz, c, n_lig, 3).reshape(bsz * c, n_lig, 3),
        ).reshape(bsz, c, n_atom, n_lig)
        valid = (
            sc_mask[:, start:end, :, None]
            & lig_mask[:, None, None, :].bool()
            & node_mask[:, start:end, None, None].bool()
        )
        dists = dists.masked_fill(~valid.expand_as(dists), 50.0)
        out[:, start:end] = dists.amin(dim=(-1, -2))
    return out


def integrate_path(
    model,
    batch,
    rigids0: Rigid,
    chi0: torch.Tensor,
    n_steps: int,
    interaction_prior: Optional[torch.Tensor] = None,
    esm_gate_context: Optional[torch.Tensor] = None,
    chi_clip: float = 1.0,
    rot_clip: float = 0.1,
    trans_clip: float = 0.2,
) -> Tuple[List[Rigid], List[torch.Tensor], List[float]]:
    dt = 1.0 / int(n_steps)
    rigids = rigids0
    chi = chi0
    rigids_list = [rigids]
    chi_list = [chi]
    t_list = [0.0]
    for k in range(int(n_steps)):
        t = torch.full((chi.shape[0],), k * dt, device=chi.device)
        t_next = torch.full((chi.shape[0],), (k + 1) * dt, device=chi.device)
        out1 = model(
            chi=chi,
            rigids=rigids,
            esm=batch.esm,
            lig_points=batch.lig_points,
            lig_types=batch.lig_types,
            lig_mask=batch.lig_mask,
            w_res=batch.w_res,
            t=t,
            node_mask=batch.node_mask,
            nma_features=batch.nma_features,
            interaction_prior=interaction_prior,
            esm_gate_context=esm_gate_context,
            current_step=10**9,
        )
        d_chi1 = maybe_clip(out1["d_chi"], chi_clip)
        d_rot1 = maybe_clip(out1["d_rigid_rot"], rot_clip)
        d_trans1 = maybe_clip(out1["d_rigid_trans"], trans_clip)
        chi_pred = wrap_to_pi(chi + dt * d_chi1)
        R_curr, t_curr = rigid_to_rt(rigids)
        R_inc1, t_inc1 = se3_exp(torch.cat([d_rot1, d_trans1], dim=-1) * dt)
        R_pred, t_pred = rigid_compose(R_curr, t_curr, R_inc1, t_inc1)
        rigids_pred = rt_to_rigid(R_pred, t_pred)
        out2 = model(
            chi=chi_pred,
            rigids=rigids_pred,
            esm=batch.esm,
            lig_points=batch.lig_points,
            lig_types=batch.lig_types,
            lig_mask=batch.lig_mask,
            w_res=batch.w_res,
            t=t_next,
            node_mask=batch.node_mask,
            nma_features=batch.nma_features,
            interaction_prior=interaction_prior,
            esm_gate_context=esm_gate_context,
            current_step=10**9,
        )
        d_chi2 = maybe_clip(out2["d_chi"], chi_clip)
        d_rot2 = maybe_clip(out2["d_rigid_rot"], rot_clip)
        d_trans2 = maybe_clip(out2["d_rigid_trans"], trans_clip)
        d_chi = 0.5 * (d_chi1 + d_chi2)
        d_rot = 0.5 * (d_rot1 + d_rot2)
        d_trans = 0.5 * (d_trans1 + d_trans2)
        chi = wrap_to_pi(chi + dt * d_chi)
        R_inc, t_inc = se3_exp(torch.cat([d_rot, d_trans], dim=-1) * dt)
        R_new, t_new = rigid_compose(R_curr, t_curr, R_inc, t_inc)
        rigids = rt_to_rigid(R_new, t_new)
        rigids_list.append(rigids)
        chi_list.append(chi)
        t_list.append((k + 1) * dt)
    return rigids_list, chi_list, t_list


def boundary_residual_path(
    model,
    batch,
    rigids_apo: Rigid,
    rigids_holo: Rigid,
    n_steps: int,
    interaction_prior: Optional[torch.Tensor] = None,
    esm_gate_context: Optional[torch.Tensor] = None,
    envelope: str = "sin2",
    residual_scale: float = 1.0,
    chi_clip: float = 1.0,
    rot_clip: float = 0.1,
    trans_clip: float = 0.2,
) -> Tuple[List[Rigid], List[torch.Tensor], List[float]]:
    rigids_list: List[Rigid] = []
    chi_list: List[torch.Tensor] = []
    t_list: List[float] = []
    n_steps = int(n_steps)
    bsz = batch.esm.shape[0]
    for k in range(n_steps + 1):
        t_val = k / max(n_steps, 1)
        interp_rigids, interp_chi = interpolate_endpoints(batch, rigids_apo, rigids_holo, t_val)
        beta = boundary_residual_envelope(t_val, envelope)
        if beta <= 0.0:
            rigids_list.append(interp_rigids)
            chi_list.append(interp_chi)
            t_list.append(t_val)
            continue

        t_tensor = torch.full((bsz,), t_val, device=interp_chi.device)
        out = model(
            chi=interp_chi,
            rigids=interp_rigids,
            esm=batch.esm,
            lig_points=batch.lig_points,
            lig_types=batch.lig_types,
            lig_mask=batch.lig_mask,
            w_res=batch.w_res,
            t=t_tensor,
            node_mask=batch.node_mask,
            nma_features=batch.nma_features,
            interaction_prior=interaction_prior,
            esm_gate_context=esm_gate_context,
            current_step=10**9,
        )
        d_chi = maybe_clip(out["d_chi"], chi_clip)
        d_rot = maybe_clip(out["d_rigid_rot"], rot_clip)
        d_trans = maybe_clip(out["d_rigid_trans"], trans_clip)
        residual_weight = beta * float(residual_scale)
        chi_t = wrap_to_pi(interp_chi + residual_weight * d_chi)
        R_ref, t_ref = rigid_to_rt(interp_rigids)
        R_res, t_res = se3_exp(torch.cat([d_rot, d_trans], dim=-1) * residual_weight)
        R_t, trans_t = rigid_compose(R_ref, t_ref, R_res, t_res)
        rigids_list.append(rt_to_rigid(R_t, trans_t))
        chi_list.append(chi_t)
        t_list.append(t_val)
    return rigids_list, chi_list, t_list


def pure_bridge_path(
    batch,
    rigids_apo: Rigid,
    rigids_holo: Rigid,
    n_steps: int,
) -> Tuple[List[Rigid], List[torch.Tensor], List[float]]:
    rigids_list: List[Rigid] = []
    chi_list: List[torch.Tensor] = []
    t_list: List[float] = []
    for k in range(int(n_steps) + 1):
        t_val = k / max(int(n_steps), 1)
        rigids_t, chi_t = interpolate_endpoints(batch, rigids_apo, rigids_holo, t_val)
        rigids_list.append(rigids_t)
        chi_list.append(chi_t)
        t_list.append(t_val)
    return rigids_list, chi_list, t_list


def bridge_timewarp_path(
    model,
    batch,
    rigids_apo: Rigid,
    rigids_holo: Rigid,
    n_steps: int,
    interaction_prior: Optional[torch.Tensor],
    esm_gate_context: Optional[torch.Tensor],
    logit_scale: float,
    rate_eps: float,
    rate_clip: float,
) -> Tuple[List[Rigid], List[torch.Tensor], List[float]]:
    n_steps = int(n_steps)
    if n_steps <= 0:
        raise ValueError("n_steps must be > 0")
    bsz, n_res = batch.node_mask.shape
    device = batch.node_mask.device
    node_mask = batch.node_mask.bool()
    node_mask_f = node_mask.float()
    rates: List[torch.Tensor] = []

    for k in range(n_steps):
        t_mid = (k + 0.5) / n_steps
        mid_rigids, mid_chi = interpolate_endpoints(batch, rigids_apo, rigids_holo, t_mid)
        t_tensor = torch.full((bsz,), t_mid, device=device)
        out = model(
            chi=mid_chi,
            rigids=mid_rigids,
            esm=batch.esm,
            lig_points=batch.lig_points,
            lig_types=batch.lig_types,
            lig_mask=batch.lig_mask,
            w_res=batch.w_res,
            t=t_tensor,
            node_mask=batch.node_mask,
            nma_features=batch.nma_features,
            interaction_prior=interaction_prior,
            esm_gate_context=esm_gate_context,
        )
        logits = out["time_warp_logits"].float().squeeze(-1) * node_mask_f
        rate = F.softplus(logits * float(logit_scale)) + float(rate_eps)
        if float(rate_clip) > 0.0:
            rate = rate.clamp(max=float(rate_clip))
        rate = torch.where(node_mask, rate, torch.ones_like(rate))
        rates.append(rate)

    rate_stack = torch.stack(rates, dim=0)
    cumulative = torch.cumsum(rate_stack, dim=0)
    total_rate = cumulative[-1].clamp(min=float(rate_eps))
    tau_values: List[torch.Tensor] = [
        torch.zeros((bsz, n_res), dtype=torch.float32, device=device)
    ]
    for k in range(n_steps):
        tau_values.append((cumulative[k] / total_rate).clamp(0.0, 1.0))

    rigids_list: List[Rigid] = [rigids_apo]
    chi_list: List[torch.Tensor] = [batch.torsion_apo[..., 3:7]]
    t_list: List[float] = [0.0]
    for k in range(1, n_steps):
        base_t = k / n_steps
        tau = torch.where(
            node_mask,
            tau_values[k],
            torch.full_like(tau_values[k], base_t),
        )
        rigids_t, chi_t = interpolate_endpoints_tensor(batch, rigids_apo, rigids_holo, tau)
        rigids_list.append(rigids_t)
        chi_list.append(chi_t)
        t_list.append(base_t)
    rigids_list.append(rigids_holo)
    chi_list.append(batch.torsion_holo[..., 3:7])
    t_list.append(1.0)
    return rigids_list, chi_list, t_list


def phase_residual_tau_values(
    model,
    batch,
    rigids_apo: Rigid,
    rigids_holo: Rigid,
    n_steps: int,
    interaction_prior: Optional[torch.Tensor],
    esm_gate_context: Optional[torch.Tensor],
    *,
    tau_mode: str,
    bridge_mode: str,
    logit_scale: float,
    rate_eps: float,
    rate_clip: float,
) -> List[torch.Tensor]:
    """Evaluate monotone per-residue phase values on a uniform time grid."""
    n_steps = int(n_steps)
    if n_steps < 2:
        raise ValueError("phase_orthogonal_residual_v1 requires n_steps >= 2")
    bsz, n_res = batch.node_mask.shape
    device = batch.node_mask.device
    node_mask = batch.node_mask.bool()
    node_mask_f = node_mask.float()

    if tau_mode == "identity":
        return [
            torch.full(
                (bsz, n_res),
                k / n_steps,
                dtype=torch.float32,
                device=device,
            )
            for k in range(n_steps + 1)
        ]
    if tau_mode != "learned":
        raise ValueError(f"Unsupported phase residual tau_mode={tau_mode}")

    rates: List[torch.Tensor] = []
    for k in range(n_steps):
        t_mid = (k + 0.5) / n_steps
        mid_tau = torch.full(
            (bsz, n_res), t_mid, dtype=torch.float32, device=device
        )
        mid_rigids, mid_chi = phase_interpolate_endpoints_tensor(
            batch, rigids_apo, rigids_holo, mid_tau, bridge_mode
        )
        out = model(
            chi=mid_chi,
            rigids=mid_rigids,
            esm=batch.esm,
            lig_points=batch.lig_points,
            lig_types=batch.lig_types,
            lig_mask=batch.lig_mask,
            w_res=batch.w_res,
            t=torch.full((bsz,), t_mid, device=device),
            node_mask=batch.node_mask,
            nma_features=batch.nma_features,
            interaction_prior=interaction_prior,
            esm_gate_context=esm_gate_context,
        )
        logits = out["time_warp_logits"].float().squeeze(-1) * node_mask_f
        rate = F.softplus(logits * float(logit_scale)) + float(rate_eps)
        if float(rate_clip) > 0.0:
            rate = rate.clamp(max=float(rate_clip))
        rates.append(torch.where(node_mask, rate, torch.ones_like(rate)))
    rate_stack = torch.stack(rates, dim=0)
    cumulative = torch.cumsum(rate_stack, dim=0)
    total_rate = cumulative[-1].clamp(min=float(rate_eps))
    tau_values = [
        torch.zeros((bsz, n_res), dtype=torch.float32, device=device)
    ]
    tau_values.extend(
        (cumulative[k] / total_rate).clamp(0.0, 1.0)
        for k in range(n_steps)
    )
    return tau_values


def phase_orthogonal_residual_path(
    model,
    batch,
    rigids_apo: Rigid,
    rigids_holo: Rigid,
    n_steps: int,
    interaction_prior: Optional[torch.Tensor],
    esm_gate_context: Optional[torch.Tensor],
    *,
    tau_mode: str,
    bridge_mode: str,
    logit_scale: float,
    rate_eps: float,
    rate_clip: float,
    envelope_kind: str,
    residual_scale: float,
    rotation_metric_scale: float,
    translation_metric_scale: float,
    chi_metric_scale: float,
    min_tangent_norm: float,
    max_metric_norm: float,
    projection_mode: str = "product",
) -> Tuple[List[Rigid], List[torch.Tensor], List[float]]:
    """Evaluate the endpoint-exact phase plus normal-residual path family."""
    n_steps = int(n_steps)
    if n_steps < 2:
        raise ValueError("phase_orthogonal_residual_v1 requires n_steps >= 2")
    bsz, _ = batch.node_mask.shape
    device = batch.node_mask.device
    node_mask = batch.node_mask.bool()
    tau_values = phase_residual_tau_values(
        model,
        batch,
        rigids_apo,
        rigids_holo,
        n_steps,
        interaction_prior,
        esm_gate_context,
        tau_mode=tau_mode,
        bridge_mode=bridge_mode,
        logit_scale=logit_scale,
        rate_eps=rate_eps,
        rate_clip=rate_clip,
    )

    bridge_tangent_chi = wrap_to_pi(
        batch.torsion_holo[..., 3:7] - batch.torsion_apo[..., 3:7]
    )

    rigids_list: List[Rigid] = [rigids_apo]
    chi_list: List[torch.Tensor] = [batch.torsion_apo[..., 3:7]]
    t_list: List[float] = [0.0]
    for k in range(1, n_steps):
        t_value = k / n_steps
        tau = torch.where(
            node_mask,
            tau_values[k],
            torch.full_like(tau_values[k], t_value),
        )
        bridge_rigids, bridge_chi = phase_interpolate_endpoints_tensor(
            batch,
            rigids_apo,
            rigids_holo,
            tau,
            bridge_mode,
        )
        bridge_tangent_rigid = phase_bridge_tangent(
            batch, rigids_apo, rigids_holo, tau, bridge_mode
        )
        out = model(
            chi=bridge_chi,
            rigids=bridge_rigids,
            esm=batch.esm,
            lig_points=batch.lig_points,
            lig_types=batch.lig_types,
            lig_mask=batch.lig_mask,
            w_res=batch.w_res,
            t=torch.full((bsz,), t_value, device=device),
            node_mask=batch.node_mask,
            nma_features=batch.nma_features,
            interaction_prior=interaction_prior,
            esm_gate_context=esm_gate_context,
        )
        residual_rigid = torch.cat(
            [out["residual_rigid_rot"], out["residual_rigid_trans"]],
            dim=-1,
        )
        projection_fn = {
            "product": project_product_tangent_normal,
            "block": project_block_tangent_normal,
        }.get(projection_mode)
        if projection_fn is None:
            raise ValueError(f"Unsupported normal projection mode={projection_mode}")
        projection = projection_fn(
            residual_rigid,
            out["residual_chi"],
            bridge_tangent_rigid,
            bridge_tangent_chi,
            node_mask=batch.node_mask,
            chi_mask=batch.chi_mask,
            rotation_scale=rotation_metric_scale,
            translation_scale=translation_metric_scale,
            chi_scale=chi_metric_scale,
            min_tangent_norm=min_tangent_norm,
            max_metric_norm=max_metric_norm,
        )
        envelope = endpoint_zero_envelope(
            torch.full((bsz,), t_value, device=device),
            kind=envelope_kind,
        ).view(bsz, 1, 1)
        weight = envelope * float(residual_scale)
        R_bridge, trans_bridge = rigid_to_rt(bridge_rigids)
        R_residual, trans_residual = se3_exp(
            projection["projected_rigid"] * weight
        )
        R_path, trans_path = rigid_compose(
            R_bridge,
            trans_bridge,
            R_residual,
            trans_residual,
        )
        rigids_list.append(rt_to_rigid(R_path, trans_path))
        chi_list.append(
            wrap_to_pi(bridge_chi + projection["projected_chi"] * weight)
        )
        t_list.append(t_value)

    rigids_list.append(rigids_holo)
    chi_list.append(batch.torsion_holo[..., 3:7])
    t_list.append(1.0)
    return rigids_list, chi_list, t_list


def chain_internal_bridge_path(
    batch,
    rigids_apo: Rigid,
    rigids_holo: Rigid,
    n_steps: int,
) -> Tuple[List[Rigid], List[torch.Tensor], List[float]]:
    """Construct an endpoint-exact, peptide-connected backbone bridge."""
    if int(n_steps) < 2:
        raise ValueError("chain_internal_bridge_v1 requires n_steps >= 2")
    rigids_list: List[Rigid] = [rigids_apo]
    chi_list: List[torch.Tensor] = [batch.torsion_apo[..., 3:7]]
    t_list: List[float] = [0.0]
    for step in range(1, int(n_steps)):
        t_value = step / int(n_steps)
        n_coord, ca_coord, c_coord = interpolate_backbone_internal(
            batch.N_apo,
            batch.Ca_apo,
            batch.C_apo,
            batch.N_holo,
            batch.Ca_holo,
            batch.C_holo,
            batch.node_mask,
            batch.peptide_bond_mask,
            batch.N_apo.new_tensor(t_value),
        )
        rigids_list.append(
            build_rigids_from_backbone(
                n_coord, ca_coord, c_coord, batch.node_mask
            )
        )
        chi_list.append(
            interpolate_endpoints(
                batch, rigids_apo, rigids_holo, t_value
            )[1]
        )
        t_list.append(t_value)
    rigids_list.append(rigids_holo)
    chi_list.append(batch.torsion_holo[..., 3:7])
    t_list.append(1.0)
    return rigids_list, chi_list, t_list


def cartesian_backbone_bridge_path(
    batch,
    rigids_apo: Rigid,
    rigids_holo: Rigid,
    n_steps: int,
) -> Tuple[List[Rigid], List[torch.Tensor], List[float]]:
    """Interpolate endpoint backbone triplets before rebuilding residue frames."""
    if int(n_steps) < 2:
        raise ValueError("cartesian_backbone_bridge_v1 requires n_steps >= 2")
    rigids_list: List[Rigid] = [rigids_apo]
    chi_list: List[torch.Tensor] = [batch.torsion_apo[..., 3:7]]
    t_list: List[float] = [0.0]
    for step in range(1, int(n_steps)):
        t_value = step / int(n_steps)
        gamma = 3.0 * t_value * t_value - 2.0 * t_value * t_value * t_value
        n_coord = (1.0 - gamma) * batch.N_apo + gamma * batch.N_holo
        ca_coord = (1.0 - gamma) * batch.Ca_apo + gamma * batch.Ca_holo
        c_coord = (1.0 - gamma) * batch.C_apo + gamma * batch.C_holo
        rigids_list.append(
            build_rigids_from_backbone(
                n_coord, ca_coord, c_coord, batch.node_mask
            )
        )
        chi_list.append(
            interpolate_endpoints(
                batch, rigids_apo, rigids_holo, t_value
            )[1]
        )
        t_list.append(t_value)
    rigids_list.append(rigids_holo)
    chi_list.append(batch.torsion_holo[..., 3:7])
    t_list.append(1.0)
    return rigids_list, chi_list, t_list


def peptide_projected_bridge_path(
    batch,
    rigids_apo: Rigid,
    rigids_holo: Rigid,
    n_steps: int,
    fk_module,
    n_iterations: int,
    relaxation: float,
    anchor_strength: float,
    max_translation: float,
    activation_loss_threshold: float,
) -> Tuple[List[Rigid], List[torch.Tensor], List[float]]:
    """Project local peptide geometry while retaining the global SE(3) bridge."""
    return project_peptide_geometry_onto_path(
        batch,
        *pure_bridge_path(batch, rigids_apo, rigids_holo, n_steps),
        fk_module=fk_module,
        n_iterations=n_iterations,
        relaxation=relaxation,
        anchor_strength=anchor_strength,
        max_translation=max_translation,
        activation_loss_threshold=activation_loss_threshold,
    )


def project_peptide_geometry_onto_path(
    batch,
    rigids_list: List[Rigid],
    chi_list: List[torch.Tensor],
    t_list: List[float],
    fk_module,
    n_iterations: int,
    relaxation: float,
    anchor_strength: float,
    max_translation: float,
    activation_loss_threshold: float,
) -> Tuple[List[Rigid], List[torch.Tensor], List[float]]:
    """Apply bounded local peptide projection to an endpoint-exact guide path."""
    projected_rigids: List[Rigid] = [rigids_list[0]]
    for rigids_t, chi_t, t_value in zip(
        rigids_list[1:-1], chi_list[1:-1], t_list[1:-1]
    ):
        atom14_t = torsions_to_atom14(
            fk_module,
            interpolate_backbone_torsions(batch, t_value),
            chi_t,
            rigids_t,
            batch.aatype,
        )
        target_length, target_cacn, target_cnca = (
            interpolate_peptide_geometry_targets(batch, t_value)
        )
        translation = project_peptide_frame_translations(
            atom14_t["atom14_pos"].float(),
            atom14_t["atom14_mask"].bool(),
            batch.node_mask.bool(),
            batch.peptide_bond_mask.bool(),
            n_iterations=n_iterations,
            relaxation=relaxation,
            anchor_strength=anchor_strength,
            bond_length=target_length,
            angle_cacn=target_cacn,
            angle_cnca=target_cnca,
            max_translation=max_translation,
            activation_loss_threshold=activation_loss_threshold,
        )
        rotation, guide_translation = rigid_to_rt(rigids_t)
        projected_rigids.append(
            rt_to_rigid(rotation, guide_translation + translation)
        )
    projected_rigids.append(rigids_list[-1])
    return projected_rigids, chi_list, t_list


def cartesian_peptide_projected_bridge_path(
    batch,
    rigids_apo: Rigid,
    rigids_holo: Rigid,
    n_steps: int,
    fk_module,
    n_iterations: int,
    relaxation: float,
    anchor_strength: float,
    max_translation: float,
    activation_loss_threshold: float,
) -> Tuple[List[Rigid], List[torch.Tensor], List[float]]:
    """Combine the Cartesian backbone guide with bounded local projection."""
    return project_peptide_geometry_onto_path(
        batch,
        *cartesian_backbone_bridge_path(
            batch, rigids_apo, rigids_holo, n_steps
        ),
        fk_module=fk_module,
        n_iterations=n_iterations,
        relaxation=relaxation,
        anchor_strength=anchor_strength,
        max_translation=max_translation,
        activation_loss_threshold=activation_loss_threshold,
    )


def pose_graph_projected_bridge_path(
    batch,
    rigids_apo: Rigid,
    rigids_holo: Rigid,
    n_steps: int,
    n_iterations: int,
    learning_rate: float,
    edge_weight: float,
    anchor_weight: float,
    rotation_metric_scale: float,
    max_rotation: float,
    max_translation: float,
) -> Tuple[List[Rigid], List[torch.Tensor], List[float]]:
    """Project adjacent frame transforms while anchored to the global bridge."""
    rigids_list, chi_list, t_list = pure_bridge_path(
        batch, rigids_apo, rigids_holo, n_steps
    )
    apo_rotation, apo_translation = rigid_to_rt(rigids_apo)
    holo_rotation, holo_translation = rigid_to_rt(rigids_holo)
    projected_rigids: List[Rigid] = [rigids_list[0]]
    for guide_rigids, t_value in zip(rigids_list[1:-1], t_list[1:-1]):
        guide_rotation, guide_translation = rigid_to_rt(guide_rigids)
        gamma = 3.0 * t_value * t_value - 2.0 * t_value * t_value * t_value
        rotation, translation, _ = project_anchored_pose_graph(
            guide_rotation,
            guide_translation,
            apo_rotation,
            apo_translation,
            holo_rotation,
            holo_translation,
            batch.node_mask,
            batch.peptide_bond_mask,
            progress=gamma,
            n_iterations=n_iterations,
            learning_rate=learning_rate,
            edge_weight=edge_weight,
            anchor_weight=anchor_weight,
            rotation_metric_scale=rotation_metric_scale,
            max_rotation=max_rotation,
            max_translation=max_translation,
        )
        projected_rigids.append(rt_to_rigid(rotation, translation))
    projected_rigids.append(rigids_list[-1])
    return projected_rigids, chi_list, t_list


def project_terminal_path(
    rigids_list: List[Rigid],
    chi_list: List[torch.Tensor],
    t_list: List[float],
    rigids_holo: Rigid,
    chi_holo: torch.Tensor,
    schedule: str,
) -> Tuple[List[Rigid], List[torch.Tensor], List[float], Dict[str, torch.Tensor]]:
    if not rigids_list or not chi_list or len(rigids_list) != len(chi_list):
        raise ValueError("terminal projection requires matching non-empty rigid/chi paths")

    final_R, final_t = rigid_to_rt(rigids_list[-1])
    holo_R, holo_t = rigid_to_rt(rigids_holo)
    final_R_inv, final_t_inv = rigid_inverse(final_R, final_t)
    delta_R, delta_t = rigid_compose(final_R_inv, final_t_inv, holo_R, holo_t)
    delta_xi = se3_log(delta_R, delta_t)
    delta_chi = wrap_to_pi(chi_holo - chi_list[-1])

    projected_rigids: List[Rigid] = []
    projected_chi: List[torch.Tensor] = []
    for rigids_t, chi_t, t_value in zip(rigids_list, chi_list, t_list):
        weight = terminal_projection_weight(float(t_value), schedule)
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

    correction = {
        "delta_xi": delta_xi,
        "delta_chi": delta_chi,
    }
    return projected_rigids, projected_chi, list(t_list), correction


def construct_path(
    args,
    config,
    model,
    batch,
    rigids_apo: Rigid,
    rigids_holo: Rigid,
    n_steps: int,
    interaction_prior: Optional[torch.Tensor],
    esm_gate_context: Optional[torch.Tensor],
    integration_clips: Dict[str, float],
    fk_module=None,
) -> Tuple[List[Rigid], List[torch.Tensor], List[float], Dict[str, torch.Tensor]]:
    path_mode = resolve_path_parameterization(args, config)
    correction: Dict[str, torch.Tensor] = {}
    if path_mode == "pure_bridge":
        return (*pure_bridge_path(batch, rigids_apo, rigids_holo, n_steps), correction)
    if path_mode == "cartesian_backbone_bridge_v1":
        return (
            *cartesian_backbone_bridge_path(
                batch, rigids_apo, rigids_holo, n_steps
            ),
            correction,
        )
    if path_mode == "cartesian_peptide_projected_bridge_v1":
        if fk_module is None:
            raise ValueError(
                "cartesian_peptide_projected_bridge_v1 requires an FK module"
            )
        return (
            *cartesian_peptide_projected_bridge_path(
                batch,
                rigids_apo,
                rigids_holo,
                n_steps,
                fk_module,
                n_iterations=int(args.peptide_projection_iterations),
                relaxation=float(args.peptide_projection_relaxation),
                anchor_strength=float(args.peptide_projection_anchor_strength),
                max_translation=float(args.peptide_projection_max_translation),
                activation_loss_threshold=float(
                    args.peptide_projection_activation_loss_threshold
                ),
            ),
            correction,
        )
    if path_mode == "chain_internal_bridge_v1":
        return (
            *chain_internal_bridge_path(batch, rigids_apo, rigids_holo, n_steps),
            correction,
        )
    if path_mode == "peptide_projected_bridge_v1":
        if fk_module is None:
            raise ValueError("peptide_projected_bridge_v1 requires an FK module")
        return (
            *peptide_projected_bridge_path(
                batch,
                rigids_apo,
                rigids_holo,
                n_steps,
                fk_module,
                n_iterations=int(args.peptide_projection_iterations),
                relaxation=float(args.peptide_projection_relaxation),
                anchor_strength=float(args.peptide_projection_anchor_strength),
                max_translation=float(args.peptide_projection_max_translation),
                activation_loss_threshold=float(
                    args.peptide_projection_activation_loss_threshold
                ),
            ),
            correction,
        )
    if path_mode == "pose_graph_projected_bridge_v1":
        return (
            *pose_graph_projected_bridge_path(
                batch,
                rigids_apo,
                rigids_holo,
                n_steps,
                n_iterations=int(args.pose_graph_iterations),
                learning_rate=float(args.pose_graph_learning_rate),
                edge_weight=float(args.pose_graph_edge_weight),
                anchor_weight=float(args.pose_graph_anchor_weight),
                rotation_metric_scale=float(args.pose_graph_rotation_metric_scale),
                max_rotation=float(args.pose_graph_max_rotation),
                max_translation=float(args.pose_graph_max_translation),
            ),
            correction,
        )
    if path_mode == "bridge_timewarp_v1":
        logit_scale = (
            float(args.time_warp_logit_scale)
            if args.time_warp_logit_scale is not None
            else float(getattr(config, "time_warp_logit_scale", 1.0))
        )
        rate_eps = (
            float(args.time_warp_rate_eps)
            if args.time_warp_rate_eps is not None
            else float(getattr(config, "time_warp_rate_eps", 1e-3))
        )
        rate_clip = (
            float(args.time_warp_rate_clip)
            if args.time_warp_rate_clip is not None
            else float(getattr(config, "time_warp_rate_clip", 10.0))
        )
        return (
            *bridge_timewarp_path(
                model,
                batch,
                rigids_apo,
                rigids_holo,
                n_steps=n_steps,
                interaction_prior=interaction_prior,
                esm_gate_context=esm_gate_context,
                logit_scale=logit_scale,
                rate_eps=rate_eps,
                rate_clip=rate_clip,
            ),
            correction,
        )
    if path_mode in {
        "phase_orthogonal_residual_v1",
        "phase_block_orthogonal_residual_v2",
    }:
        def resolve_value(arg_name: str, default):
            value = getattr(args, arg_name)
            return value if value is not None else getattr(config, arg_name, default)

        phase_path = phase_orthogonal_residual_path(
            model,
            batch,
            rigids_apo,
            rigids_holo,
            n_steps=n_steps,
            interaction_prior=interaction_prior,
            esm_gate_context=esm_gate_context,
            tau_mode=str(resolve_value("phase_residual_tau_mode", "learned")),
            bridge_mode=str(
                resolve_value("phase_residual_bridge_mode", "se3_geodesic")
            ),
            logit_scale=float(resolve_value("time_warp_logit_scale", 1.0)),
            rate_eps=float(resolve_value("time_warp_rate_eps", 1e-3)),
            rate_clip=float(resolve_value("time_warp_rate_clip", 10.0)),
            envelope_kind=str(resolve_value("phase_residual_envelope", "poly")),
            residual_scale=float(resolve_value("phase_residual_scale", 1.0)),
            rotation_metric_scale=float(
                resolve_value("phase_residual_rotation_metric_scale", 1.0)
            ),
            translation_metric_scale=float(
                resolve_value("phase_residual_translation_metric_scale", 1.0)
            ),
            chi_metric_scale=float(
                resolve_value("phase_residual_chi_metric_scale", 1.0)
            ),
            min_tangent_norm=float(
                resolve_value("phase_residual_min_tangent_norm", 1e-3)
            ),
            max_metric_norm=float(
                resolve_value("phase_residual_max_metric_norm", 0.0)
            ),
            projection_mode=(
                "block"
                if path_mode == "phase_block_orthogonal_residual_v2"
                else "product"
            ),
        )
        if bool(getattr(config, "phase_residual_peptide_retraction", False)):
            if fk_module is None:
                raise ValueError(
                    "phase residual peptide retraction requires an FK module"
                )
            phase_path = project_peptide_geometry_onto_path(
                batch,
                *phase_path,
                fk_module=fk_module,
                n_iterations=int(
                    getattr(
                        config,
                        "phase_residual_peptide_retraction_iterations",
                        8,
                    )
                ),
                relaxation=float(
                    getattr(
                        config,
                        "phase_residual_peptide_retraction_relaxation",
                        0.75,
                    )
                ),
                anchor_strength=float(
                    getattr(
                        config,
                        "phase_residual_peptide_retraction_anchor_strength",
                        0.02,
                    )
                ),
                max_translation=float(
                    getattr(
                        config,
                        "phase_residual_peptide_retraction_max_translation",
                        1.0,
                    )
                ),
                activation_loss_threshold=float(
                    getattr(
                        config,
                        "phase_residual_peptide_retraction_activation_loss_threshold",
                        0.0,
                    )
                ),
            )
        return (
            *phase_path,
            correction,
        )
    if path_mode in {"boundary_residual_v1", "boundary_residual"}:
        envelope = args.boundary_residual_envelope or getattr(config, "boundary_residual_envelope", "sin2")
        residual_scale = (
            float(args.boundary_residual_scale)
            if args.boundary_residual_scale is not None
            else float(getattr(config, "boundary_residual_scale", 1.0))
        )
        return (
            *boundary_residual_path(
                model,
                batch,
                rigids_apo,
                rigids_holo,
                n_steps=n_steps,
                interaction_prior=interaction_prior,
                esm_gate_context=esm_gate_context,
                envelope=envelope,
                residual_scale=residual_scale,
                chi_clip=integration_clips["chi"],
                rot_clip=integration_clips["rot"],
                trans_clip=integration_clips["trans"],
            ),
            correction,
        )

    rigids_list, chi_list, t_list = integrate_path(
        model,
        batch,
        rigids_apo,
        batch.torsion_apo[..., 3:7],
        n_steps=n_steps,
        interaction_prior=interaction_prior,
        esm_gate_context=esm_gate_context,
        chi_clip=integration_clips["chi"],
        rot_clip=integration_clips["rot"],
        trans_clip=integration_clips["trans"],
    )
    if path_mode == "projected_flow":
        schedule = args.terminal_projection_schedule or getattr(
            config, "terminal_projection_schedule", "smootherstep"
        )
        rigids_list, chi_list, t_list, correction = project_terminal_path(
            rigids_list,
            chi_list,
            t_list,
            rigids_holo,
            batch.torsion_holo[..., 3:7],
            schedule,
        )
    return rigids_list, chi_list, t_list, correction


def compute_interaction_prior_feature(args, config, batch, fk_module, rigids_apo, rigids_holo):
    mode = args.interaction_prior_feature_mode or getattr(config, "interaction_prior_feature_mode", "none")
    if mode == "none":
        return None
    if mode == "zero":
        return batch.w_res.new_zeros(batch.w_res.shape)
    if mode == "oracle_contact":
        atom14_holo = torsions_to_atom14(
            fk_module,
            batch.torsion_holo[..., :3],
            batch.torsion_holo[..., 3:7],
            rigids_holo,
            batch.aatype,
        )
        d_holo = min_sc_ligand_dist(
            atom14_holo["atom14_pos"].float(),
            atom14_holo["atom14_mask"].bool(),
            batch.lig_points.float(),
            batch.lig_mask.bool(),
            batch.node_mask.bool(),
        )
        contact_dist = float(getattr(config, "interaction_prior_contact_dist", args.contact_dist))
        tau = float(getattr(config, "interaction_prior_contact_tau", 0.75))
        return torch.sigmoid((contact_dist - d_holo.clamp(max=50.0)) / max(tau, 1e-6))
    ckpt_path = args.interaction_prior_ckpt or getattr(config, "interaction_prior_ckpt", None)
    if not ckpt_path:
        raise ValueError("interaction prior feature mode requires --interaction_prior_ckpt or checkpoint config")
    prior_model = compute_interaction_prior_feature._cache.get(str(ckpt_path))
    if prior_model is None:
        prior_model = load_interaction_prior(str(ckpt_path), batch.esm.device)
        compute_interaction_prior_feature._cache[str(ckpt_path)] = prior_model
    atom14_apo = torsions_to_atom14(
        fk_module,
        batch.torsion_apo[..., :3],
        batch.torsion_apo[..., 3:7],
        rigids_apo,
        batch.aatype,
    )
    with torch.no_grad():
        logits = prior_model(
            atom14_apo["atom14_pos"].float(),
            sidechain_atom_mask(atom14_apo["atom14_mask"].bool(), batch.node_mask.bool()),
            atom14_apo["atom14_pos"][:, :, 1].float(),
            batch.aatype,
            batch.lig_points.float(),
            batch.lig_types.float(),
            batch.lig_mask.bool(),
            batch.node_mask.bool(),
        )
    temp = args.interaction_prior_temperature
    if temp is None:
        temp = float(getattr(config, "interaction_prior_temperature", 1.0))
    prior = torch.sigmoid(logits / max(float(temp), 1e-6)) * batch.node_mask.float()
    if mode == "prior_shuffled":
        if prior.shape[0] > 1:
            prior = prior.roll(shifts=1, dims=0)
        else:
            prior = prior.roll(shifts=1, dims=1)
        prior = prior * batch.node_mask.float()
    return prior


compute_interaction_prior_feature._cache = {}


class RunningStats:
    def __init__(self):
        self.sum = 0.0
        self.count = 0

    def add(self, values: torch.Tensor, mask: torch.Tensor):
        mask = mask.bool()
        if mask.any():
            self.sum += float(values[mask].detach().float().sum().item())
            self.count += int(mask.sum().item())

    @property
    def mean(self):
        return self.sum / self.count if self.count > 0 else None


def add_direction(stats: Dict[str, RunningStats], name: str, actual_delta, target_delta, mask):
    ok = (actual_delta * target_delta) > 0
    stats[name].add(ok.float(), mask)


def chi_gap_mean(delta_chi: torch.Tensor, chi_mask: torch.Tensor) -> torch.Tensor:
    chi_mask_f = chi_mask.float()
    return (delta_chi.abs() * chi_mask_f).sum(dim=-1) / chi_mask_f.sum(dim=-1).clamp(min=1.0)


def state_gap_to_holo(
    rigids_t: Rigid,
    chi_t: torch.Tensor,
    rigids_holo: Rigid,
    chi_holo: torch.Tensor,
    chi_mask: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    R_t, trans_t = rigid_to_rt(rigids_t)
    R_h, trans_h = rigid_to_rt(rigids_holo)
    R_inv, trans_inv = rigid_inverse(R_t, trans_t)
    R_delta, trans_delta = rigid_compose(R_inv, trans_inv, R_h, trans_h)
    xi = se3_log(R_delta, trans_delta)
    rot_gap = torch.linalg.norm(xi[..., :3], dim=-1)
    trans_gap = torch.linalg.norm(xi[..., 3:], dim=-1)
    chi_gap = chi_gap_mean(wrap_to_pi(chi_holo - chi_t), chi_mask)
    total_gap = trans_gap + rot_gap + chi_gap
    return total_gap, trans_gap, rot_gap, chi_gap


def state_step_motion(
    rigids_prev: Rigid,
    chi_prev: torch.Tensor,
    rigids_next: Rigid,
    chi_next: torch.Tensor,
    chi_mask: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    R_prev, trans_prev = rigid_to_rt(rigids_prev)
    R_next, trans_next = rigid_to_rt(rigids_next)
    R_inv, trans_inv = rigid_inverse(R_prev, trans_prev)
    R_delta, trans_delta = rigid_compose(R_inv, trans_inv, R_next, trans_next)
    xi = se3_log(R_delta, trans_delta)
    rot_step = torch.linalg.norm(xi[..., :3], dim=-1)
    trans_step = torch.linalg.norm(xi[..., 3:], dim=-1)
    chi_step = chi_gap_mean(wrap_to_pi(chi_next - chi_prev), chi_mask)
    total_step = trans_step + rot_step + chi_step
    return total_step, trans_step, rot_step, chi_step


def late_fraction(step_values: List[torch.Tensor], t_list: List[float], cutoff: float) -> torch.Tensor:
    if not step_values:
        raise ValueError("late_fraction requires at least one step")
    total = torch.stack(step_values, dim=0).sum(dim=0)
    late_terms = [
        step
        for step, t_end in zip(step_values, t_list[1:])
        if float(t_end) > float(cutoff)
    ]
    if late_terms:
        late = torch.stack(late_terms, dim=0).sum(dim=0)
    else:
        late = torch.zeros_like(total)
    return torch.where(total > 1e-8, late / total.clamp(min=1e-8), torch.zeros_like(total))


def velocity_spike_ratio(step_values: List[torch.Tensor]) -> torch.Tensor:
    if not step_values:
        raise ValueError("velocity_spike_ratio requires at least one step")
    stacked = torch.stack(step_values, dim=0)
    total = stacked.sum(dim=0)
    mean = total / max(stacked.shape[0], 1)
    max_step = stacked.max(dim=0).values
    return torch.where(total > 1e-8, max_step / mean.clamp(min=1e-8), torch.zeros_like(total))


def progress_auc_from_gaps(gap_values: List[torch.Tensor]) -> torch.Tensor:
    if not gap_values:
        raise ValueError("progress_auc_from_gaps requires at least one path state")
    initial = gap_values[0]
    valid = initial > 1e-6
    progress = [
        torch.where(valid, 1.0 - gap / initial.clamp(min=1e-6), torch.zeros_like(gap))
        for gap in gap_values
    ]
    return torch.stack(progress, dim=0).clamp(min=-1.0, max=1.0).mean(dim=0)


def evaluate_batch(
    args,
    model,
    fk_module,
    batch,
    n_steps: int,
    interaction_settings: Dict[str, object],
    stage1v2_settings: Dict[str, object],
    integration_clips: Dict[str, float],
):
    rigids_apo = build_rigids_from_backbone(batch.N_apo, batch.Ca_apo, batch.C_apo, batch.node_mask)
    rigids_holo = build_rigids_from_backbone(batch.N_holo, batch.Ca_holo, batch.C_holo, batch.node_mask)
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
    gate_context = esm_gate_context(args.config, batch, stage1v2_settings)
    rigids_list, chi_list, t_list, correction = construct_path(
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
        fk_module=fk_module,
    )
    bridge_rigids_list, bridge_chi_list, bridge_t_list = pure_bridge_path(
        batch, rigids_apo, rigids_holo, n_steps
    )

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
    path_dists = []
    for rigids_t, chi_t, t_value in zip(rigids_list, chi_list, t_list):
        atom14_t = torsions_to_atom14(
            fk_module,
            interpolate_backbone_torsions(batch, t_value),
            chi_t,
            rigids_t,
            batch.aatype,
        )
        path_dists.append(
            min_sc_ligand_dist(
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

    node = batch.node_mask.bool()
    pocket = node & (batch.w_res > float(args.pocket_threshold))
    finite = pocket & torch.isfinite(d_apo) & torch.isfinite(d_holo) & (d_apo < 49.0) & (d_holo < 49.0)
    active = finite & (delta_target.abs() >= float(args.active_delta))
    formed = active & (d_apo > float(args.contact_dist)) & (d_holo <= float(args.contact_dist))
    released = active & (d_apo <= float(args.contact_dist)) & (d_holo > float(args.contact_dist))
    stable_contact = finite & (d_apo <= float(args.contact_dist)) & (d_holo <= float(args.contact_dist))
    stable_noncontact = finite & (d_apo > float(args.contact_dist)) & (d_holo > float(args.contact_dist))
    approach = active & (delta_target < 0)
    release = active & (delta_target > 0)

    classes = {
        "all_pocket": finite,
        "active": active,
        "approach": approach,
        "release": release,
        "formed_contact": formed,
        "released_contact": released,
        "stable_contact": stable_contact,
        "stable_noncontact": stable_noncontact,
    }
    cap = float(args.path_dist_cap)
    endpoint_abs = (d_final.clamp(max=cap) - d_holo.clamp(max=cap)).abs()
    improvement = (d_apo - d_holo).abs() - (d_final - d_holo).abs()

    path_mae = d_final.new_zeros(d_final.shape)
    path_mae_interior = d_final.new_zeros(d_final.shape)
    interior_path_frames = 0
    for d_t, t_val in zip(path_dists, t_list):
        target_t = d_apo + float(t_val) * delta_target
        frame_error = (d_t.clamp(max=cap) - target_t.clamp(max=cap)).abs()
        path_mae = path_mae + frame_error
        if 1e-6 < float(t_val) < 1.0 - 1e-6:
            path_mae_interior = path_mae_interior + frame_error
            interior_path_frames += 1
    path_mae = path_mae / max(len(path_dists), 1)
    path_mae_interior = path_mae_interior / max(interior_path_frames, 1)
    path_dist_stack = torch.stack(path_dists, dim=0)
    path_min_ligand_dist = path_dist_stack.min(dim=0).values
    ligand_clash_severity = torch.relu(
        path_min_ligand_dist.new_tensor(float(args.ligand_clash_dist)) - path_min_ligand_dist
    )
    ligand_clash_proxy = (ligand_clash_severity > 0).float()

    chi_holo = batch.torsion_holo[..., 3:7]
    gap_total = []
    gap_trans = []
    gap_rot = []
    gap_chi = []
    for rigids_t, chi_t in zip(rigids_list, chi_list):
        total_gap, trans_gap, rot_gap, chi_gap = state_gap_to_holo(
            rigids_t,
            chi_t,
            rigids_holo,
            chi_holo,
            batch.chi_mask,
        )
        gap_total.append(total_gap)
        gap_trans.append(trans_gap)
        gap_rot.append(rot_gap)
        gap_chi.append(chi_gap)

    step_total = []
    step_trans = []
    step_rot = []
    step_chi = []
    for idx in range(1, len(rigids_list)):
        total_step, trans_step, rot_step, chi_step = state_step_motion(
            rigids_list[idx - 1],
            chi_list[idx - 1],
            rigids_list[idx],
            chi_list[idx],
            batch.chi_mask,
        )
        step_total.append(total_step)
        step_trans.append(trans_step)
        step_rot.append(rot_step)
        step_chi.append(chi_step)

    if step_total:
        step_total_stack = torch.stack(step_total, dim=0)
        step_trans_stack = torch.stack(step_trans, dim=0)
        step_rot_stack = torch.stack(step_rot, dim=0)
        step_chi_stack = torch.stack(step_chi, dim=0)
        path_length_total = step_total_stack.sum(dim=0)
        path_length_trans = step_trans_stack.sum(dim=0)
        path_length_rot = step_rot_stack.sum(dim=0)
        path_length_chi = step_chi_stack.sum(dim=0)
        path_action_total = (step_total_stack ** 2).sum(dim=0)
        if step_total_stack.shape[0] > 1:
            path_curvature_proxy_total = (
                step_total_stack[1:] - step_total_stack[:-1]
            ).abs().sum(dim=0)
        else:
            path_curvature_proxy_total = d_final.new_zeros(d_final.shape)
    else:
        path_length_total = d_final.new_zeros(d_final.shape)
        path_length_trans = d_final.new_zeros(d_final.shape)
        path_length_rot = d_final.new_zeros(d_final.shape)
        path_length_chi = d_final.new_zeros(d_final.shape)
        path_action_total = d_final.new_zeros(d_final.shape)
        path_curvature_proxy_total = d_final.new_zeros(d_final.shape)

    bridge_dev_total = []
    bridge_dev_trans = []
    bridge_dev_rot = []
    bridge_dev_chi = []
    for rigids_t, chi_t, rigids_b, chi_b, t_val in zip(
        rigids_list, chi_list, bridge_rigids_list, bridge_chi_list, bridge_t_list
    ):
        if not (1e-6 < float(t_val) < 1.0 - 1e-6):
            continue
        total_dev, trans_dev, rot_dev, chi_dev = state_step_motion(
            rigids_b,
            chi_b,
            rigids_t,
            chi_t,
            batch.chi_mask,
        )
        bridge_dev_total.append(total_dev)
        bridge_dev_trans.append(trans_dev)
        bridge_dev_rot.append(rot_dev)
        bridge_dev_chi.append(chi_dev)
    if bridge_dev_total:
        bridge_dev_total_stack = torch.stack(bridge_dev_total, dim=0)
        bridge_dev_trans_stack = torch.stack(bridge_dev_trans, dim=0)
        bridge_dev_rot_stack = torch.stack(bridge_dev_rot, dim=0)
        bridge_dev_chi_stack = torch.stack(bridge_dev_chi, dim=0)
        bridge_dev_total_mean = bridge_dev_total_stack.mean(dim=0)
        bridge_dev_total_max = bridge_dev_total_stack.max(dim=0).values
        bridge_dev_trans_mean = bridge_dev_trans_stack.mean(dim=0)
        bridge_dev_rot_mean = bridge_dev_rot_stack.mean(dim=0)
        bridge_dev_chi_mean = bridge_dev_chi_stack.mean(dim=0)
    else:
        bridge_dev_total_mean = d_final.new_zeros(d_final.shape)
        bridge_dev_total_max = d_final.new_zeros(d_final.shape)
        bridge_dev_trans_mean = d_final.new_zeros(d_final.shape)
        bridge_dev_rot_mean = d_final.new_zeros(d_final.shape)
        bridge_dev_chi_mean = d_final.new_zeros(d_final.shape)

    progress_auc_total = progress_auc_from_gaps(gap_total)
    progress_auc_trans = progress_auc_from_gaps(gap_trans)
    progress_auc_rot = progress_auc_from_gaps(gap_rot)
    progress_auc_chi = progress_auc_from_gaps(gap_chi)
    late_motion_fraction_10 = late_fraction(step_total, t_list, 0.9)
    late_motion_fraction_20 = late_fraction(step_total, t_list, 0.8)
    late_motion_fraction_trans_10 = late_fraction(step_trans, t_list, 0.9)
    late_motion_fraction_chi_10 = late_fraction(step_chi, t_list, 0.9)
    velocity_spike_total = velocity_spike_ratio(step_total)

    terminal_correction_total = d_final.new_zeros(d_final.shape)
    terminal_correction_trans = d_final.new_zeros(d_final.shape)
    terminal_correction_rot = d_final.new_zeros(d_final.shape)
    terminal_correction_chi = d_final.new_zeros(d_final.shape)
    if correction:
        delta_xi = correction["delta_xi"]
        terminal_correction_trans = torch.linalg.norm(delta_xi[..., 3:], dim=-1)
        terminal_correction_rot = torch.linalg.norm(delta_xi[..., :3], dim=-1)
        terminal_correction_chi = chi_gap_mean(correction["delta_chi"], batch.chi_mask)
        terminal_correction_total = (
            terminal_correction_trans
            + terminal_correction_rot
            + terminal_correction_chi
        )

    stats = defaultdict(RunningStats)
    counts = {}
    for name, mask in classes.items():
        counts[name] = int(mask.sum().item())
        stats[f"{name}/endpoint_abs_dist"].add(endpoint_abs, mask)
        stats[f"{name}/path_mae_dist"].add(path_mae, mask)
        stats[f"{name}/path_mae_dist_interior"].add(path_mae_interior, mask)
        stats[f"{name}/improvement_to_holo"].add(improvement, mask)
        add_direction(stats, f"{name}/direction_acc", delta_actual, delta_target, mask)
        stats[f"{name}/path_min_ligand_dist"].add(path_min_ligand_dist, mask)
        stats[f"{name}/ligand_clash_proxy"].add(ligand_clash_proxy, mask)
        stats[f"{name}/ligand_clash_severity"].add(ligand_clash_severity, mask)
        stats[f"{name}/path_length_total"].add(path_length_total, mask)
        stats[f"{name}/path_length_trans"].add(path_length_trans, mask)
        stats[f"{name}/path_length_rot"].add(path_length_rot, mask)
        stats[f"{name}/path_length_chi"].add(path_length_chi, mask)
        stats[f"{name}/path_action_total"].add(path_action_total, mask)
        stats[f"{name}/path_curvature_proxy_total"].add(path_curvature_proxy_total, mask)
        stats[f"{name}/bridge_dev_total_mean"].add(bridge_dev_total_mean, mask)
        stats[f"{name}/bridge_dev_total_max"].add(bridge_dev_total_max, mask)
        stats[f"{name}/bridge_dev_trans_mean"].add(bridge_dev_trans_mean, mask)
        stats[f"{name}/bridge_dev_rot_mean"].add(bridge_dev_rot_mean, mask)
        stats[f"{name}/bridge_dev_chi_mean"].add(bridge_dev_chi_mean, mask)
        stats[f"{name}/progress_auc_total"].add(progress_auc_total, mask)
        stats[f"{name}/progress_auc_trans"].add(progress_auc_trans, mask)
        stats[f"{name}/progress_auc_rot"].add(progress_auc_rot, mask)
        stats[f"{name}/progress_auc_chi"].add(progress_auc_chi, mask)
        stats[f"{name}/late_motion_fraction_10"].add(late_motion_fraction_10, mask)
        stats[f"{name}/late_motion_fraction_20"].add(late_motion_fraction_20, mask)
        stats[f"{name}/late_motion_fraction_trans_10"].add(late_motion_fraction_trans_10, mask)
        stats[f"{name}/late_motion_fraction_chi_10"].add(late_motion_fraction_chi_10, mask)
        stats[f"{name}/velocity_spike_ratio"].add(velocity_spike_total, mask)
        stats[f"{name}/terminal_correction_total"].add(terminal_correction_total, mask)
        stats[f"{name}/terminal_correction_trans"].add(terminal_correction_trans, mask)
        stats[f"{name}/terminal_correction_rot"].add(terminal_correction_rot, mask)
        stats[f"{name}/terminal_correction_chi"].add(terminal_correction_chi, mask)
    return stats, counts


def merge_stats(total_stats, batch_stats):
    for key, stat in batch_stats.items():
        total_stats[key].sum += stat.sum
        total_stats[key].count += stat.count


def main() -> None:
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
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
    if n_steps <= 0:
        raise ValueError(f"n_integration_steps must be > 0, got {n_steps}")
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
    if len(loader.dataset) == 0:
        raise ValueError(
            "No evaluation samples remain after split/subset/cache filtering: "
            f"split={args.split!r}, valid_samples_file={args.valid_samples_file!r}"
        )

    total_stats = defaultdict(RunningStats)
    total_counts = defaultdict(int)
    total_batches = 0
    total_samples = 0
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(loader, desc="Transition eval", ncols=120)):
            if args.max_batches is not None and batch_idx >= int(args.max_batches):
                break
            batch = batch_to_device(batch, device)
            batch_stats, counts = evaluate_batch(
                args,
                model,
                fk_module,
                batch,
                n_steps,
                interaction_settings,
                stage1v2_settings,
                integration_clips,
            )
            merge_stats(total_stats, batch_stats)
            for key, value in counts.items():
                total_counts[key] += int(value)
            total_batches += 1
            total_samples += int(batch.esm.shape[0])

    summary = {
        "checkpoint": args.checkpoint,
        "data_dir": args.data_dir,
        "split": args.split,
        "valid_samples_file": args.valid_samples_file,
        "interaction_prior_feature_mode": interaction_settings["mode"],
        "stage1v2_posterior_feature_mode": stage1v2_settings["mode"],
        "stage1v2_posterior_cache_dir": stage1v2_settings["cache_dir"],
        "stage1v2_posterior_feature_names": list(stage1v2_settings["names"]),
        "path_parameterization": resolve_path_parameterization(args, config),
        "boundary_residual_envelope": (
            args.boundary_residual_envelope
            if args.boundary_residual_envelope is not None
            else getattr(config, "boundary_residual_envelope", None)
        ),
        "boundary_residual_scale": (
            args.boundary_residual_scale
            if args.boundary_residual_scale is not None
            else getattr(config, "boundary_residual_scale", None)
        ),
        "terminal_projection_schedule": (
            args.terminal_projection_schedule
            if args.terminal_projection_schedule is not None
            else getattr(config, "terminal_projection_schedule", None)
        ),
        "time_warp_logit_scale": (
            args.time_warp_logit_scale
            if args.time_warp_logit_scale is not None
            else getattr(config, "time_warp_logit_scale", None)
        ),
        "time_warp_rate_eps": (
            args.time_warp_rate_eps
            if args.time_warp_rate_eps is not None
            else getattr(config, "time_warp_rate_eps", None)
        ),
        "time_warp_rate_clip": (
            args.time_warp_rate_clip
            if args.time_warp_rate_clip is not None
            else getattr(config, "time_warp_rate_clip", None)
        ),
        "phase_residual_tau_mode": (
            args.phase_residual_tau_mode
            if args.phase_residual_tau_mode is not None
            else getattr(config, "phase_residual_tau_mode", None)
        ),
        "phase_residual_bridge_mode": (
            args.phase_residual_bridge_mode
            if args.phase_residual_bridge_mode is not None
            else getattr(config, "phase_residual_bridge_mode", "se3_geodesic")
        ),
        "phase_residual_envelope": (
            args.phase_residual_envelope
            if args.phase_residual_envelope is not None
            else getattr(config, "phase_residual_envelope", None)
        ),
        "phase_residual_scale": (
            args.phase_residual_scale
            if args.phase_residual_scale is not None
            else getattr(config, "phase_residual_scale", None)
        ),
        "phase_residual_rotation_metric_scale": (
            args.phase_residual_rotation_metric_scale
            if args.phase_residual_rotation_metric_scale is not None
            else getattr(config, "phase_residual_rotation_metric_scale", None)
        ),
        "phase_residual_translation_metric_scale": (
            args.phase_residual_translation_metric_scale
            if args.phase_residual_translation_metric_scale is not None
            else getattr(config, "phase_residual_translation_metric_scale", None)
        ),
        "phase_residual_chi_metric_scale": (
            args.phase_residual_chi_metric_scale
            if args.phase_residual_chi_metric_scale is not None
            else getattr(config, "phase_residual_chi_metric_scale", None)
        ),
        "phase_residual_max_metric_norm": (
            args.phase_residual_max_metric_norm
            if args.phase_residual_max_metric_norm is not None
            else getattr(config, "phase_residual_max_metric_norm", None)
        ),
        "phase_residual_min_tangent_norm": (
            args.phase_residual_min_tangent_norm
            if args.phase_residual_min_tangent_norm is not None
            else getattr(config, "phase_residual_min_tangent_norm", None)
        ),
        "phase_residual_peptide_retraction": bool(
            getattr(config, "phase_residual_peptide_retraction", False)
        ),
        "phase_residual_peptide_retraction_iterations": int(
            getattr(config, "phase_residual_peptide_retraction_iterations", 8)
        ),
        "phase_residual_peptide_retraction_relaxation": float(
            getattr(config, "phase_residual_peptide_retraction_relaxation", 0.75)
        ),
        "phase_residual_peptide_retraction_anchor_strength": float(
            getattr(
                config,
                "phase_residual_peptide_retraction_anchor_strength",
                0.02,
            )
        ),
        "phase_residual_peptide_retraction_max_translation": float(
            getattr(
                config,
                "phase_residual_peptide_retraction_max_translation",
                1.0,
            )
        ),
        "phase_residual_peptide_retraction_activation_loss_threshold": float(
            getattr(
                config,
                "phase_residual_peptide_retraction_activation_loss_threshold",
                0.0,
            )
        ),
        "n_integration_steps": n_steps,
        "integration_clips": integration_clips,
        "active_delta": args.active_delta,
        "contact_dist": args.contact_dist,
        "ligand_clash_dist": args.ligand_clash_dist,
        "pocket_threshold": args.pocket_threshold,
        "interaction_prior_temperature": (
            args.interaction_prior_temperature
            if args.interaction_prior_temperature is not None
            else getattr(config, "interaction_prior_temperature", None)
        ),
        "interaction_prior_feature_scale": (
            args.interaction_prior_feature_scale
            if args.interaction_prior_feature_scale is not None
            else getattr(config, "interaction_prior_feature_scale", None)
        ),
        "stage1v2_posterior_feature_scale": stage1v2_settings["scale"],
        "batches": total_batches,
        "samples": total_samples,
        "counts": dict(total_counts),
        "metrics": {key: stat.mean for key, stat in sorted(total_stats.items())},
    }
    text = json.dumps(summary, indent=2, sort_keys=True)
    print(text)
    if args.output:
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
