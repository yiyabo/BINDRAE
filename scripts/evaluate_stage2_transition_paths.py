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
from src.stage2.modules import rigid_compose, rigid_inverse, se3_exp, se3_log, wrap_to_pi  # noqa: E402
from flash_ipa.rigid import Rigid, Rotation  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate residue-level Stage-2 path transitions")
    parser.add_argument("--checkpoint", required=True, help="Stage-2 checkpoint path")
    parser.add_argument("--data_dir", default="processed_data/triplets", help="Stage-2 data directory")
    parser.add_argument("--split", default="val", choices=["train", "val", "test"], help="dataset split")
    parser.add_argument("--valid_samples_file", default=None, help="sample filter file relative to data_dir")
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--device", default="cuda")
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
    parser.add_argument("--pocket_threshold", type=float, default=0.3)
    parser.add_argument("--output", default=None, help="optional JSON output path")
    return parser.parse_args()


def batch_to_device(batch, device: torch.device):
    for name in (
        "esm",
        "torsion_apo",
        "torsion_holo",
        "bb_mask",
        "chi_mask",
        "node_mask",
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


def build_model_config_for_checkpoint(args, config, split: str) -> Tuple[TorsionFlowNetConfig, Dict[str, object], Dict[str, object]]:
    interaction_settings = resolve_interaction_prior_feature_settings(args, config)
    stage1v2_settings = resolve_stage1v2_feature_settings(args, config, split)
    prior_dim = int(interaction_settings["dim"]) + int(stage1v2_settings["dim"])
    model_config = TorsionFlowNetConfig(
        esm_fusion_enabled=bool(getattr(config, "esm_fusion_enabled", False)),
        esm_num_layers=int(getattr(config, "esm_num_layers", 1)),
        esm_fusion_mode=str(getattr(config, "esm_fusion_mode", "sum")),
        esm_layer_dropout=float(getattr(config, "esm_layer_dropout", 0.0)),
        nma_dim=int(getattr(config, "nma_dim", 0)),
        stage1_chi_feature_scale=float(getattr(config, "stage1_chi_feature_scale", 1.0)),
        interaction_prior_feature_dim=prior_dim,
        interaction_prior_feature_scale=1.0,
        repa_enabled=bool(getattr(config, "repa_enabled", False)),
        repa_dim=int(getattr(config, "repa_dim", 128)),
        repa_target_dim=int(stage1v2_settings["dim"]),
    )
    return model_config, interaction_settings, stage1v2_settings


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
    rigids_list, chi_list, t_list = integrate_path(
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
    for rigids_t, chi_t in zip(rigids_list, chi_list):
        atom14_t = torsions_to_atom14(fk_module, batch.torsion_apo[..., :3], chi_t, rigids_t, batch.aatype)
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
    for d_t, t_val in zip(path_dists, t_list):
        target_t = d_apo + float(t_val) * delta_target
        path_mae = path_mae + (d_t.clamp(max=cap) - target_t.clamp(max=cap)).abs()
    path_mae = path_mae / max(len(path_dists), 1)

    stats = defaultdict(RunningStats)
    counts = {}
    for name, mask in classes.items():
        counts[name] = int(mask.sum().item())
        stats[f"{name}/endpoint_abs_dist"].add(endpoint_abs, mask)
        stats[f"{name}/path_mae_dist"].add(path_mae, mask)
        stats[f"{name}/improvement_to_holo"].add(improvement, mask)
        add_direction(stats, f"{name}/direction_acc", delta_actual, delta_target, mask)
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
    model.load_state_dict(ckpt["model_state_dict"], strict=True)
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
        valid_samples_file=args.valid_samples_file,
        stage1v2_posterior_cache_dir=stage1v2_settings["cache_dir"],
        stage1v2_posterior_feature_mode=stage1v2_settings["mode"],
        stage1v2_posterior_feature_names=stage1v2_settings["names_raw"],
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
        "n_integration_steps": n_steps,
        "integration_clips": integration_clips,
        "active_delta": args.active_delta,
        "contact_dist": args.contact_dist,
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
