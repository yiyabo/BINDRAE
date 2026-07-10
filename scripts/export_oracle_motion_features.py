#!/usr/bin/env python3
"""Export privileged apo-to-holo motion features and audit direct oracle apply.

This script builds the first OracleMotion-UB artifact:

1. Per-residue local apo->holo motion features computed from paired Stage-2
   apo/holo structures.
2. A direct oracle-apply audit that reconstructs holo frames and chi angles
   from apo plus the oracle deltas. This must be near ceiling before using the
   features in Stage-2 training.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))
flash_ipa_path = project_root / "vendor" / "flash_ipa" / "src"
if flash_ipa_path.exists():
    sys.path.insert(0, str(flash_ipa_path))

from flash_ipa.rigid import Rigid, Rotation  # noqa: E402

from src.stage1.models.fk_openfold import create_openfold_fk, reorder_torsions_to_openfold  # noqa: E402
from src.stage1.models.interaction_prior import min_sidechain_ligand_dist, sidechain_atom_mask  # noqa: E402
from src.stage2.datasets.dataset_stage2 import (  # noqa: E402
    ApoHoloBridgeDataset,
    collate_stage2_batch,
    create_stage2_dataloader,
)
from src.stage2.modules import rigid_compose, rigid_inverse, se3_log, so3_log, wrap_to_pi  # noqa: E402


SCHEMA_VERSION = "bindrae_oracle_motion_v2_canonical_residue_keys"
FEATURE_NAMES = (
    "delta_trans_local_x_norm",
    "delta_trans_local_y_norm",
    "delta_trans_local_z_norm",
    "delta_rot_log_x_norm",
    "delta_rot_log_y_norm",
    "delta_rot_log_z_norm",
    "delta_chi1_sin",
    "delta_chi2_sin",
    "delta_chi3_sin",
    "delta_chi4_sin",
    "delta_chi1_cos",
    "delta_chi2_cos",
    "delta_chi3_cos",
    "delta_chi4_cos",
    "chi1_mask",
    "chi2_mask",
    "chi3_mask",
    "chi4_mask",
    "trans_mag_norm",
    "rot_angle_norm",
    "max_abs_delta_chi_norm",
    "motion_active",
    "contact_apo",
    "contact_holo",
    "formed_contact",
    "released_contact",
    "signed_delta_dist_norm",
    "w_res",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export oracle apo-to-holo motion features and direct oracle-apply audit"
    )
    parser.add_argument("--data_dir", default="processed_data/triplets")
    parser.add_argument("--split", choices=("train", "val", "test"), default="val")
    parser.add_argument("--valid_samples_file", default=None)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument(
        "--esm_num_layers",
        type=int,
        default=1,
        help="Require Stage-2 ESM features with this many last layers while exporting.",
    )
    parser.add_argument("--max_batches", type=int, default=0)
    parser.add_argument("--max_samples", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--contact_dist", type=float, default=4.5)
    parser.add_argument("--contact_tau", type=float, default=0.75)
    parser.add_argument("--pocket_threshold", type=float, default=0.3)
    parser.add_argument("--moving_trans_threshold", type=float, default=0.5)
    parser.add_argument("--moving_rot_threshold", type=float, default=0.25)
    parser.add_argument("--moving_chi_threshold_deg", type=float, default=30.0)
    parser.add_argument("--translation_scale", type=float, default=5.0)
    parser.add_argument("--distance_scale", type=float, default=5.0)
    parser.add_argument("--max_saved_dist", type=float, default=50.0)
    parser.add_argument("--residue_chunk", type=int, default=64)
    parser.add_argument("--skip_direct_apply", action="store_true")
    parser.add_argument(
        "--skip_bad_samples",
        action="store_true",
        help="Skip samples that fail to load and record them instead of aborting.",
    )
    parser.add_argument("--bad_samples_out", default=None)
    parser.add_argument("--log_every", type=int, default=25)
    parser.add_argument("--shard_id", type=int, default=0, help="0-indexed shard id for parallel export")
    parser.add_argument("--num_shards", type=int, default=1, help="total number of shards for parallel export")
    return parser.parse_args()


def batch_to_device(batch, device: torch.device):
    for name, value in vars(batch).items():
        if torch.is_tensor(value):
            setattr(batch, name, value.to(device))
    return batch


def truncate_batch(batch, keep: int):
    keep = int(keep)
    for name, value in vars(batch).items():
        if torch.is_tensor(value):
            setattr(batch, name, value[:keep])
        elif isinstance(value, list):
            setattr(batch, name, value[:keep])
    return batch


def safe_sample_id(sample_id: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", sample_id)


def tensor_to_np(x: torch.Tensor) -> np.ndarray:
    return x.detach().cpu().numpy()


def build_rigids_from_backbone(n_coord, ca_coord, c_coord, mask, eps: float = 1e-6) -> Rigid:
    device = ca_coord.device
    dtype = ca_coord.dtype
    default_e1 = torch.tensor([1.0, 0.0, 0.0], device=device, dtype=dtype)
    default_e2 = torch.tensor([0.0, 1.0, 0.0], device=device, dtype=dtype)
    default_e3 = torch.tensor([0.0, 0.0, 1.0], device=device, dtype=dtype)

    e1 = c_coord - ca_coord
    e1_norm = torch.norm(e1, dim=-1, keepdim=True)
    e1 = torch.where(e1_norm > eps, e1 / e1_norm.clamp_min(eps), default_e1.expand_as(e1))

    u = n_coord - ca_coord
    proj = (u * e1).sum(dim=-1, keepdim=True) * e1
    e2 = u - proj
    e2_norm = torch.norm(e2, dim=-1, keepdim=True)
    e2 = torch.where(e2_norm > eps, e2 / e2_norm.clamp_min(eps), default_e2.expand_as(e2))

    e3 = torch.cross(e1, e2, dim=-1)
    e3_norm = torch.norm(e3, dim=-1, keepdim=True)
    e3 = torch.where(e3_norm > eps, e3 / e3_norm.clamp_min(eps), default_e3.expand_as(e3))
    e2 = torch.cross(e3, e1, dim=-1)

    rot = torch.stack([e1, e2, e3], dim=-1)
    valid = mask.bool().unsqueeze(-1).unsqueeze(-1)
    eye = torch.eye(3, device=device, dtype=dtype).view(1, 1, 3, 3)
    rot = torch.where(valid, rot, eye.expand_as(rot))
    trans = torch.where(mask.bool().unsqueeze(-1), ca_coord, torch.zeros_like(ca_coord))
    return Rigid(rots=Rotation(rot_mats=rot), trans=trans)


def rigid_to_rt(rigids: Rigid) -> Tuple[torch.Tensor, torch.Tensor]:
    return rigids.get_rots().get_rot_mats(), rigids.get_trans()


def rt_to_rigid(R: torch.Tensor, t: torch.Tensor) -> Rigid:
    return Rigid(rots=Rotation(rot_mats=R), trans=t)


def torsions_to_atom14(fk_module, rigids: Rigid, aatype: torch.Tensor, torsions: torch.Tensor):
    torsions_sincos = torch.stack([torch.sin(torsions), torch.cos(torsions)], dim=-1)
    torsions_sincos = reorder_torsions_to_openfold(torsions_sincos)
    return fk_module(torsions_sincos.float(), rigids, aatype.clamp(0, 20))


def build_sidechain_distances(batch, fk_module, rigids_apo, rigids_holo, args: argparse.Namespace):
    apo_atom14 = torsions_to_atom14(fk_module, rigids_apo, batch.aatype, batch.torsion_apo)
    holo_atom14 = torsions_to_atom14(fk_module, rigids_holo, batch.aatype, batch.torsion_holo)
    apo_sc_mask = sidechain_atom_mask(apo_atom14["atom14_mask"].bool(), batch.node_mask.bool())
    holo_sc_mask = sidechain_atom_mask(holo_atom14["atom14_mask"].bool(), batch.node_mask.bool())

    d_apo = min_sidechain_ligand_dist(
        apo_atom14["atom14_pos"].float(),
        apo_sc_mask,
        batch.lig_points.float(),
        batch.lig_mask.bool(),
        batch.node_mask.bool(),
        residue_chunk=int(args.residue_chunk),
    )
    d_holo = min_sidechain_ligand_dist(
        holo_atom14["atom14_pos"].float(),
        holo_sc_mask,
        batch.lig_points.float(),
        batch.lig_mask.bool(),
        batch.node_mask.bool(),
        residue_chunk=int(args.residue_chunk),
    )
    return apo_atom14, holo_atom14, d_apo, d_holo


def build_motion_for_batch(batch, fk_module, args: argparse.Namespace) -> Dict[str, torch.Tensor]:
    rigids_apo = build_rigids_from_backbone(batch.N_apo, batch.Ca_apo, batch.C_apo, batch.node_mask)
    rigids_holo = build_rigids_from_backbone(batch.N_holo, batch.Ca_holo, batch.C_holo, batch.node_mask)
    R_apo, t_apo = rigid_to_rt(rigids_apo)
    R_holo, t_holo = rigid_to_rt(rigids_holo)

    R_apo_inv, t_apo_inv = rigid_inverse(R_apo, t_apo)
    R_delta, delta_trans_local = rigid_compose(R_apo_inv, t_apo_inv, R_holo, t_holo)
    delta_rot_log = so3_log(R_delta)
    delta_frame_log = se3_log(R_delta, delta_trans_local)

    chi_apo = batch.torsion_apo[..., 3:7]
    chi_holo = batch.torsion_holo[..., 3:7]
    chi_mask = batch.chi_mask.bool() & batch.node_mask.bool().unsqueeze(-1)
    delta_chi = wrap_to_pi(chi_holo - chi_apo)
    delta_chi_masked = torch.where(chi_mask, delta_chi, torch.zeros_like(delta_chi))
    delta_chi_sin = torch.sin(delta_chi)
    delta_chi_cos = torch.cos(delta_chi)

    valid = batch.node_mask.bool()
    trans_mag = torch.linalg.norm(delta_trans_local, dim=-1)
    rot_angle = torch.linalg.norm(delta_rot_log, dim=-1)
    max_abs_delta_chi = delta_chi.abs().amax(dim=-1)
    motion_active = valid & (
        (trans_mag >= float(args.moving_trans_threshold))
        | (rot_angle >= float(args.moving_rot_threshold))
        | (max_abs_delta_chi >= math.radians(float(args.moving_chi_threshold_deg)))
    )

    _, holo_atom14, d_apo, d_holo = build_sidechain_distances(batch, fk_module, rigids_apo, rigids_holo, args)
    dist_valid = valid & torch.isfinite(d_apo) & torch.isfinite(d_holo)
    d_apo_c = d_apo.clamp(max=float(args.max_saved_dist))
    d_holo_c = d_holo.clamp(max=float(args.max_saved_dist))
    signed_delta_dist = d_holo_c - d_apo_c
    contact_apo = dist_valid & (d_apo <= float(args.contact_dist))
    contact_holo = dist_valid & (d_holo <= float(args.contact_dist))
    formed_contact = dist_valid & (~contact_apo) & contact_holo
    released_contact = dist_valid & contact_apo & (~contact_holo)
    pocket = valid & (
        (batch.w_res >= float(args.pocket_threshold))
        | contact_apo
        | contact_holo
        | motion_active
    )
    contact_prob = torch.sigmoid((float(args.contact_dist) - d_holo_c) / max(float(args.contact_tau), 1e-6))

    scale_t = max(float(args.translation_scale), 1e-6)
    scale_d = max(float(args.distance_scale), 1e-6)
    feature_columns = [
        (delta_trans_local / scale_t).clamp(-5.0, 5.0),
        (delta_rot_log / math.pi).clamp(-1.0, 1.0),
        delta_chi_sin,
        delta_chi_cos,
        chi_mask.float(),
        (trans_mag / scale_t).clamp(max=5.0).unsqueeze(-1),
        (rot_angle / math.pi).clamp(max=1.0).unsqueeze(-1),
        (max_abs_delta_chi / math.pi).clamp(max=1.0).unsqueeze(-1),
        motion_active.float().unsqueeze(-1),
        contact_apo.float().unsqueeze(-1),
        contact_holo.float().unsqueeze(-1),
        formed_contact.float().unsqueeze(-1),
        released_contact.float().unsqueeze(-1),
        (signed_delta_dist / scale_d).clamp(-10.0, 10.0).unsqueeze(-1),
        batch.w_res.float().unsqueeze(-1),
    ]
    oracle_motion_features = torch.cat(feature_columns, dim=-1)
    oracle_motion_features = oracle_motion_features * valid.float().unsqueeze(-1)

    return {
        "rigids_apo": rigids_apo,
        "rigids_holo": rigids_holo,
        "R_apo": R_apo,
        "t_apo": t_apo,
        "R_holo": R_holo,
        "t_holo": t_holo,
        "R_delta": R_delta,
        "delta_trans_local": delta_trans_local * valid.float().unsqueeze(-1),
        "delta_rot_log": delta_rot_log * valid.float().unsqueeze(-1),
        "delta_frame_log": delta_frame_log * valid.float().unsqueeze(-1),
        "delta_chi": delta_chi,
        "delta_chi_masked": delta_chi_masked,
        "delta_chi_sin": delta_chi_sin,
        "delta_chi_cos": delta_chi_cos,
        "chi_mask": chi_mask,
        "trans_mag": trans_mag * valid.float(),
        "rot_angle": rot_angle * valid.float(),
        "max_abs_delta_chi": max_abs_delta_chi * valid.float(),
        "motion_active": motion_active,
        "valid_mask": valid,
        "dist_valid_mask": dist_valid,
        "pocket_mask": pocket,
        "apo_min_dist": d_apo_c,
        "holo_min_dist": d_holo_c,
        "signed_delta_dist": signed_delta_dist,
        "contact_prob": contact_prob,
        "contact_apo": contact_apo,
        "contact_holo": contact_holo,
        "formed_contact": formed_contact,
        "released_contact": released_contact,
        "oracle_motion_features": oracle_motion_features,
        "holo_atom14_pos": holo_atom14["atom14_pos"].float(),
        "holo_atom14_mask": holo_atom14["atom14_mask"].bool(),
    }


class RunningMean:
    def __init__(self) -> None:
        self.sum = 0.0
        self.count = 0

    def add(self, values: torch.Tensor, mask: torch.Tensor) -> None:
        mask = mask.bool()
        if mask.any():
            selected = values[mask].detach().float()
            self.sum += float(selected.sum().item())
            self.count += int(selected.numel())

    @property
    def mean(self):
        if self.count == 0:
            return None
        return self.sum / self.count


class ContactCounts:
    def __init__(self) -> None:
        self.tp = 0
        self.fp = 0
        self.fn = 0
        self.tn = 0

    def add(self, pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> None:
        pred = pred.bool() & mask.bool()
        target = target.bool() & mask.bool()
        valid = mask.bool()
        self.tp += int((pred & target & valid).sum().item())
        self.fp += int((pred & ~target & valid).sum().item())
        self.fn += int((~pred & target & valid).sum().item())
        self.tn += int((~pred & ~target & valid).sum().item())

    def summary(self) -> Dict[str, float]:
        precision = self.tp / max(self.tp + self.fp, 1)
        recall = self.tp / max(self.tp + self.fn, 1)
        f1 = 2.0 * precision * recall / max(precision + recall, 1e-12)
        acc = (self.tp + self.tn) / max(self.tp + self.fp + self.fn + self.tn, 1)
        return {
            "tp": self.tp,
            "fp": self.fp,
            "fn": self.fn,
            "tn": self.tn,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "acc": acc,
        }


def class_masks(motion: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    dist_valid = motion["dist_valid_mask"]
    return {
        "all_valid": motion["valid_mask"],
        "dist_valid": dist_valid,
        "pocket": motion["pocket_mask"],
        "motion_active": motion["motion_active"],
        "holo_contact": motion["contact_holo"],
        "apo_contact": motion["contact_apo"],
        "formed_contact": motion["formed_contact"],
        "released_contact": motion["released_contact"],
        "moving_or_contact": motion["motion_active"] | motion["contact_holo"] | motion["contact_apo"],
    }


def add_direct_apply_stats(
    stats: Dict[str, RunningMean],
    contacts: Dict[str, ContactCounts],
    batch,
    fk_module,
    motion: Dict[str, torch.Tensor],
    args: argparse.Namespace,
) -> None:
    R_oracle, t_oracle = rigid_compose(
        motion["R_apo"],
        motion["t_apo"],
        motion["R_delta"],
        motion["delta_trans_local"],
    )
    rigids_oracle = rt_to_rigid(R_oracle, t_oracle)
    chi_oracle = wrap_to_pi(batch.torsion_apo[..., 3:7] + motion["delta_chi"])
    torsion_oracle = batch.torsion_apo.clone()
    torsion_oracle[..., 3:7] = chi_oracle
    oracle_atom14 = torsions_to_atom14(fk_module, rigids_oracle, batch.aatype, torsion_oracle)
    oracle_sc_mask = sidechain_atom_mask(oracle_atom14["atom14_mask"].bool(), batch.node_mask.bool())
    holo_sc_mask = sidechain_atom_mask(motion["holo_atom14_mask"].bool(), batch.node_mask.bool())
    sc_atom_mask = oracle_sc_mask & holo_sc_mask
    d_oracle = min_sidechain_ligand_dist(
        oracle_atom14["atom14_pos"].float(),
        oracle_sc_mask,
        batch.lig_points.float(),
        batch.lig_mask.bool(),
        batch.node_mask.bool(),
        residue_chunk=int(args.residue_chunk),
    ).clamp(max=float(args.max_saved_dist))

    frame_trans_err = torch.linalg.norm(t_oracle - motion["t_holo"], dim=-1)
    rot_residual = R_oracle.transpose(-2, -1) @ motion["R_holo"]
    frame_rot_err = torch.linalg.norm(so3_log(rot_residual), dim=-1)
    chi_err = torch.abs(wrap_to_pi(chi_oracle - batch.torsion_holo[..., 3:7]))
    lig_dist_err = torch.abs(d_oracle - motion["holo_min_dist"])
    sc_atom14_pos_err = torch.linalg.norm(
        oracle_atom14["atom14_pos"].float() - motion["holo_atom14_pos"],
        dim=-1,
    )

    target_trans = torch.linalg.norm(motion["t_holo"] - motion["t_apo"], dim=-1)
    trans_recovery = (target_trans - frame_trans_err) / target_trans.clamp_min(1e-6)
    apo_rot_delta = motion["R_apo"].transpose(-2, -1) @ motion["R_holo"]
    target_rot = torch.linalg.norm(so3_log(apo_rot_delta), dim=-1)
    rot_recovery = (target_rot - frame_rot_err) / target_rot.clamp_min(1e-6)
    target_chi = torch.abs(motion["delta_chi"])
    chi_recovery = (target_chi - chi_err) / target_chi.clamp_min(1e-6)

    pred_contact = d_oracle <= float(args.contact_dist)
    masks = class_masks(motion)
    chi_mask = motion["chi_mask"]
    for name, mask in masks.items():
        stats[f"{name}/frame_trans_err_A"].add(frame_trans_err, mask)
        stats[f"{name}/frame_rot_err_rad"].add(frame_rot_err, mask)
        stats[f"{name}/trans_recovery"].add(trans_recovery, mask & (target_trans > 1e-4))
        stats[f"{name}/rot_recovery"].add(rot_recovery, mask & (target_rot > 1e-4))
        stats[f"{name}/lig_min_dist_abs_err_A"].add(lig_dist_err, mask & motion["dist_valid_mask"])
        stats[f"{name}/sidechain_atom14_pos_err_A"].add(sc_atom14_pos_err, mask.unsqueeze(-1) & sc_atom_mask)
        contacts[name].add(pred_contact, motion["contact_holo"], mask & motion["dist_valid_mask"])

        chi_class_mask = mask.unsqueeze(-1) & chi_mask
        stats[f"{name}/chi_err_rad"].add(chi_err, chi_class_mask)
        stats[f"{name}/chi_recovery"].add(chi_recovery, chi_class_mask & (target_chi > 1e-4))


def sample_counts(motion: Dict[str, torch.Tensor], index: int, n_res: int) -> Dict[str, int]:
    keys = (
        "valid_mask",
        "dist_valid_mask",
        "pocket_mask",
        "motion_active",
        "contact_apo",
        "contact_holo",
        "formed_contact",
        "released_contact",
    )
    return {key: int(motion[key][index, :n_res].sum().item()) for key in keys}


def save_sample(output_dir: Path, batch, motion: Dict[str, torch.Tensor], index: int) -> Dict:
    sample_id = batch.pdb_ids[index]
    n_res = int(batch.n_residues[index])
    filename = f"{safe_sample_id(sample_id)}.npz"
    path = output_dir / filename

    arrays = {
        "schema_version": np.array(SCHEMA_VERSION),
        "source": np.array("holo_truth"),
        "sample_id": np.array(sample_id),
        "n_residues": np.array(n_res, dtype=np.int32),
        "residue_identity_hash": np.array(batch.residue_identity_hashes[index]),
        "feature_names": np.array(FEATURE_NAMES),
        "aatype": tensor_to_np(batch.aatype[index, :n_res]).astype(np.int16),
        "node_mask": tensor_to_np(motion["valid_mask"][index, :n_res]).astype(np.bool_),
        "chi_mask": tensor_to_np(motion["chi_mask"][index, :n_res]).astype(np.bool_),
        "w_res": tensor_to_np(batch.w_res[index, :n_res]).astype(np.float32),
        "delta_trans_local": tensor_to_np(motion["delta_trans_local"][index, :n_res]).astype(np.float32),
        "delta_rot_log": tensor_to_np(motion["delta_rot_log"][index, :n_res]).astype(np.float32),
        "delta_frame_log": tensor_to_np(motion["delta_frame_log"][index, :n_res]).astype(np.float32),
        "delta_chi": tensor_to_np(motion["delta_chi"][index, :n_res]).astype(np.float32),
        "delta_chi_masked": tensor_to_np(motion["delta_chi_masked"][index, :n_res]).astype(np.float32),
        "delta_chi_sin": tensor_to_np(motion["delta_chi_sin"][index, :n_res]).astype(np.float32),
        "delta_chi_cos": tensor_to_np(motion["delta_chi_cos"][index, :n_res]).astype(np.float32),
        "trans_mag": tensor_to_np(motion["trans_mag"][index, :n_res]).astype(np.float32),
        "rot_angle": tensor_to_np(motion["rot_angle"][index, :n_res]).astype(np.float32),
        "max_abs_delta_chi": tensor_to_np(motion["max_abs_delta_chi"][index, :n_res]).astype(np.float32),
        "motion_active": tensor_to_np(motion["motion_active"][index, :n_res]).astype(np.bool_),
        "pocket_mask": tensor_to_np(motion["pocket_mask"][index, :n_res]).astype(np.bool_),
        "dist_valid_mask": tensor_to_np(motion["dist_valid_mask"][index, :n_res]).astype(np.bool_),
        "apo_min_dist": tensor_to_np(motion["apo_min_dist"][index, :n_res]).astype(np.float32),
        "holo_min_dist": tensor_to_np(motion["holo_min_dist"][index, :n_res]).astype(np.float32),
        "signed_delta_dist": tensor_to_np(motion["signed_delta_dist"][index, :n_res]).astype(np.float32),
        "contact_prob": tensor_to_np(motion["contact_prob"][index, :n_res]).astype(np.float32),
        "contact_apo": tensor_to_np(motion["contact_apo"][index, :n_res]).astype(np.bool_),
        "contact_holo": tensor_to_np(motion["contact_holo"][index, :n_res]).astype(np.bool_),
        "formed_contact": tensor_to_np(motion["formed_contact"][index, :n_res]).astype(np.bool_),
        "released_contact": tensor_to_np(motion["released_contact"][index, :n_res]).astype(np.bool_),
        "oracle_motion_features": tensor_to_np(motion["oracle_motion_features"][index, :n_res]).astype(np.float32),
    }
    np.savez_compressed(path, **arrays)
    return {
        "sample_id": sample_id,
        "path": str(path),
        "relative_path": filename,
        "n_residues": n_res,
        "counts": sample_counts(motion, index, n_res),
    }


def merge_counts(total: Dict[str, int], counts: Dict[str, int]) -> None:
    for key, value in counts.items():
        total[key] = total.get(key, 0) + int(value)


def summarize_stats(stats: Dict[str, RunningMean], contacts: Dict[str, ContactCounts]) -> Dict:
    return {
        "metrics": {key: stat.mean for key, stat in sorted(stats.items())},
        "metric_counts": {key: stat.count for key, stat in sorted(stats.items())},
        "contact_confusion": {key: value.summary() for key, value in sorted(contacts.items())},
    }


def print_progress(n_seen: int, records: List[Dict], counts: Dict[str, int]) -> None:
    if not records:
        return
    print(
        json.dumps(
            {
                "samples": n_seen,
                "last_sample": records[-1]["sample_id"],
                "counts": counts,
            },
            sort_keys=True,
        )
    )


def iter_dataset_batches(args: argparse.Namespace, *, skip_bad_samples: bool):
    """Iterate one dataset shard, optionally recording samples that fail to load."""
    dataset = ApoHoloBridgeDataset(
        args.data_dir,
        split=args.split,
        valid_samples_file=args.valid_samples_file,
        esm_num_layers=args.esm_num_layers,
    )
    batch_samples = []
    bad_records = []

    for idx in range(len(dataset)):
        if args.num_shards > 1 and idx % args.num_shards != args.shard_id:
            continue
        sample_id = dataset.samples[idx].get("id", f"sample_{idx}")
        try:
            sample = dataset[idx]
        except Exception as exc:
            if not skip_bad_samples:
                raise
            bad_records.append({"index": idx, "sample_id": sample_id, "error": str(exc)})
            print(
                json.dumps(
                    {
                        "bad_sample": sample_id,
                        "index": idx,
                        "error": str(exc),
                    },
                    ensure_ascii=False,
                    sort_keys=True,
                ),
                file=sys.stderr,
                flush=True,
            )
            continue
        batch_samples.append(sample)
        if len(batch_samples) >= int(args.batch_size):
            yield collate_stage2_batch(batch_samples), bad_records
            batch_samples = []

    if batch_samples:
        yield collate_stage2_batch(batch_samples), bad_records
    else:
        yield None, bad_records


def write_bad_samples(path_raw: Optional[str], bad_records: List[Dict]) -> None:
    if not path_raw:
        return
    path = Path(path_raw)
    if not path.is_absolute():
        path = project_root / path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(bad_records, indent=2, sort_keys=True, ensure_ascii=False) + "\n")


def main() -> None:
    args = parse_args()
    if args.num_shards < 1:
        raise ValueError(f"num_shards must be >= 1, got {args.num_shards}")
    if not 0 <= args.shard_id < args.num_shards:
        raise ValueError(
            f"shard_id must be in [0, {args.num_shards}), got {args.shard_id}"
        )
    device = torch.device(args.device if args.device != "cuda" or torch.cuda.is_available() else "cpu")
    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = project_root / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    bad_records: List[Dict] = []
    if args.num_shards > 1 or args.skip_bad_samples:
        loader_iter = iter_dataset_batches(
            args,
            skip_bad_samples=bool(args.skip_bad_samples),
        )
    else:
        loader_iter = (
            (batch, bad_records)
            for batch in create_stage2_dataloader(
                args.data_dir,
                split=args.split,
                batch_size=args.batch_size,
                shuffle=False,
                num_workers=args.num_workers,
                valid_samples_file=args.valid_samples_file,
                esm_num_layers=args.esm_num_layers,
            )
        )
    fk_module = create_openfold_fk().to(device)
    fk_module.eval()

    records: List[Dict] = []
    counts: Dict[str, int] = {}
    direct_stats: Dict[str, RunningMean] = defaultdict(RunningMean)
    direct_contacts: Dict[str, ContactCounts] = defaultdict(ContactCounts)
    n_seen = 0

    with torch.no_grad():
        for batch_idx, (batch, current_bad_records) in enumerate(loader_iter):
            bad_records = current_bad_records
            if batch is None:
                continue
            if args.max_batches > 0 and batch_idx >= args.max_batches:
                break
            if args.max_samples > 0:
                remaining = int(args.max_samples) - n_seen
                if remaining <= 0:
                    break
                if remaining < len(batch.pdb_ids):
                    batch = truncate_batch(batch, remaining)
            batch = batch_to_device(batch, device)
            motion = build_motion_for_batch(batch, fk_module, args)
            if not args.skip_direct_apply:
                add_direct_apply_stats(direct_stats, direct_contacts, batch, fk_module, motion, args)

            for i in range(len(batch.pdb_ids)):
                if args.max_samples > 0 and n_seen >= args.max_samples:
                    break
                record = save_sample(output_dir, batch, motion, i)
                records.append(record)
                merge_counts(counts, record["counts"])
                n_seen += 1
                if args.log_every > 0 and n_seen % int(args.log_every) == 0:
                    print_progress(n_seen, records, counts)
            if args.max_samples > 0 and n_seen >= args.max_samples:
                break

    manifest = {
        "schema_version": SCHEMA_VERSION,
        "source": "holo_truth",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "command": " ".join([os.path.basename(sys.argv[0]), *sys.argv[1:]]),
        "args": vars(args),
        "feature_names": list(FEATURE_NAMES),
        "feature_dim": len(FEATURE_NAMES),
        "num_samples": len(records),
        "counts": counts,
        "bad_samples": bad_records,
        "num_bad_samples": len(bad_records),
        "records": records,
    }
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    direct_summary = {
        "schema_version": SCHEMA_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "skipped": bool(args.skip_direct_apply),
        "num_samples": len(records),
        "num_bad_samples": len(bad_records),
        "counts": counts,
    }
    if not args.skip_direct_apply:
        direct_summary.update(summarize_stats(direct_stats, direct_contacts))
    direct_path = output_dir / "direct_oracle_apply_summary.json"
    direct_path.write_text(json.dumps(direct_summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_bad_samples(args.bad_samples_out, bad_records)

    print(f"Wrote {len(records)} oracle motion files")
    print(f"Bad samples skipped: {len(bad_records)}")
    print(f"Manifest: {manifest_path}")
    print(f"Direct apply summary: {direct_path}")
    print(json.dumps({"counts": counts, "direct_apply_skipped": bool(args.skip_direct_apply)}, indent=2))


if __name__ == "__main__":
    main()
