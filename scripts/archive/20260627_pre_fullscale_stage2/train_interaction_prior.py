#!/usr/bin/env python3
"""Train a Stage-1 local residue-ligand interaction prior.

This experiment deliberately avoids deterministic chi endpoint supervision.
It trains a small explicit pairwise-geometry network to predict holo
sidechain-ligand contact labels from apo sidechain geometry and known-pose
ligand tokens.  The selection metric is correct-ligand separation from
no-ligand / translated / batch-shuffled controls on true contact residues.
"""

import argparse
import json
import math
import os
import sys
from dataclasses import is_dataclass, replace
from datetime import timedelta
from functools import partial
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Subset
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))
flash_ipa_path = project_root / "vendor" / "flash_ipa" / "src"
if flash_ipa_path.exists():
    sys.path.insert(0, str(flash_ipa_path))

from flash_ipa.rigid import Rigid, Rotation

from src.stage1.datasets import ApoHoloTripletDataset, collate_stage1_batch
from src.stage1.models.interaction_prior import (
    InteractionPriorNet,
    build_pair_features as build_pair_features_impl,
    interaction_prior_config_from_args,
    min_sidechain_ligand_dist,
    sidechain_atom_mask,
)
from src.stage1.models.fk_openfold import create_openfold_fk, reorder_torsions_to_openfold
from src.stage1.modules.losses import chi1_rotamer_labels
from utils.ligand_utils import LIGAND_TYPE_DIM


AA_RESTYPE_MAP = {
    "A": 0, "R": 1, "N": 2, "D": 3, "C": 4,
    "Q": 5, "E": 6, "G": 7, "H": 8, "I": 9,
    "L": 10, "K": 11, "M": 12, "F": 13, "P": 14,
    "S": 15, "T": 16, "W": 17, "Y": 18, "V": 19,
}

CONTROL_NAMES = ("nolig", "translated", "shuffled")
SUBSET_NAMES = (
    "all",
    "target_contact",
    "apo_contact",
    "contact_gain",
    "contact_switch",
    "gain_switch",
    "pocket_switch",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train explicit local residue-ligand interaction prior")
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--val_samples_file", default=None)
    parser.add_argument("--sample_metadata_file", default=None)
    parser.add_argument("--batch_size", type=int, default=96)
    parser.add_argument("--train_max_samples", type=int, default=12000)
    parser.add_argument("--val_max_samples", type=int, default=1200)
    parser.add_argument("--subset_seed", type=int, default=20260617)
    parser.add_argument("--max_n_res", type=int, default=1600)
    parser.add_argument("--max_local_res", type=int, default=192)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--hidden_dim", type=int, default=192)
    parser.add_argument("--num_layers", type=int, default=3)
    parser.add_argument("--dropout", type=float, default=0.05)
    parser.add_argument("--num_rbf", type=int, default=24)
    parser.add_argument("--rbf_max", type=float, default=14.0)
    parser.add_argument("--type_sigma", type=float, default=4.5)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--max_epochs", type=int, default=10)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--target_contact_dist", type=float, default=4.5)
    parser.add_argument("--gain_margin", type=float, default=0.5)
    parser.add_argument("--soft_contact_tau", type=float, default=0.5)
    parser.add_argument("--contact_threshold", type=float, default=0.5)
    parser.add_argument("--pocket_threshold", type=float, default=0.2)
    parser.add_argument("--crop_mode", choices=("pocket_only", "label_enriched"), default="pocket_only")
    parser.add_argument("--local_contact_bonus", type=float, default=3.0)
    parser.add_argument("--local_switch_bonus", type=float, default=2.0)
    parser.add_argument("--local_contact_switch_bonus", type=float, default=6.0)
    parser.add_argument("--residue_chunk", type=int, default=64)
    parser.add_argument("--lambda_bce", type=float, default=1.0)
    parser.add_argument("--lambda_soft", type=float, default=0.3)
    parser.add_argument("--lambda_decoy_zero", type=float, default=0.2)
    parser.add_argument("--lambda_contrastive", type=float, default=0.8)
    parser.add_argument("--decoy_margin", type=float, default=1.0)
    parser.add_argument("--pos_weight", type=float, default=0.0)
    parser.add_argument("--save_dir", default="checkpoints/stage1/interaction_prior")
    parser.add_argument("--log_dir", default="logs/stage1/interaction_prior")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--distributed", action="store_true")
    parser.add_argument("--collect_train_metrics", action="store_true")
    return parser.parse_args()


def replace_batch(batch, **updates):
    if is_dataclass(batch):
        return replace(batch, **updates)
    clone = type("BatchClone", (), {})()
    clone.__dict__.update(getattr(batch, "__dict__", {}))
    clone.__dict__.update(updates)
    return clone


def batch_to_device(batch, device: torch.device):
    for name in (
        "N_apo",
        "Ca_apo",
        "C_apo",
        "N_holo",
        "Ca_holo",
        "C_holo",
        "node_mask",
        "lig_points",
        "lig_types",
        "lig_mask",
        "chi_holo",
        "chi_mask",
        "torsion_apo",
        "torsion_holo",
        "w_res",
    ):
        value = getattr(batch, name, None)
        if value is not None:
            setattr(batch, name, value.to(device))
    return batch


def deterministic_subset(dataset, max_samples: Optional[int], seed: int, label: str, is_main: bool):
    if max_samples is None or max_samples <= 0 or max_samples >= len(dataset):
        return dataset
    generator = torch.Generator()
    generator.manual_seed(int(seed))
    indices = torch.randperm(len(dataset), generator=generator)[: int(max_samples)].tolist()
    if is_main:
        print(f"{label} subset: {len(indices)} / {len(dataset)} samples (seed={seed})")
    return Subset(dataset, indices)


def sequences_to_aatype(sequences: List[str], max_len: int, device: torch.device) -> torch.Tensor:
    aatype = torch.zeros(len(sequences), max_len, dtype=torch.long, device=device)
    for i, seq in enumerate(sequences):
        for j, aa in enumerate(seq):
            if j >= max_len:
                break
            aatype[i, j] = AA_RESTYPE_MAP.get(aa, 20)
    return aatype


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


def gather_residue_tensor(tensor: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
    if tensor.ndim == 2:
        return tensor.gather(1, indices)
    view_shape = [indices.shape[0], indices.shape[1]] + [1] * (tensor.ndim - 2)
    expand_shape = [indices.shape[0], indices.shape[1]] + list(tensor.shape[2:])
    return tensor.gather(1, indices.view(*view_shape).expand(*expand_shape))


def select_local_batch(batch, aatype: torch.Tensor, args: argparse.Namespace):
    _, n_res = batch.node_mask.shape
    k_res = min(int(args.max_local_res), n_res)
    if k_res >= n_res:
        return batch, aatype

    score = batch.w_res.float().clone()
    if args.crop_mode == "label_enriched":
        apo_labels, apo_valid = chi1_rotamer_labels(batch.torsion_apo[:, :, 3], batch.chi_mask[:, :, 0])
        holo_labels, holo_valid = chi1_rotamer_labels(batch.chi_holo[:, :, 0], batch.chi_mask[:, :, 0])
        valid = apo_valid & holo_valid & batch.node_mask.bool()
        contact = valid & (batch.w_res > float(args.contact_threshold))
        switch = valid & (apo_labels != holo_labels)
        contact_switch = contact & switch
        score = score + float(args.local_contact_bonus) * contact.float()
        score = score + float(args.local_switch_bonus) * switch.float()
        score = score + float(args.local_contact_switch_bonus) * contact_switch.float()
        score = score + 0.01 * valid.float()
    else:
        score = score + 0.01 * batch.node_mask.float()
    score = score.masked_fill(~batch.node_mask.bool(), -1e9)
    indices = score.topk(k=k_res, dim=1).indices

    updates = {}
    for name in (
        "N_apo",
        "Ca_apo",
        "C_apo",
        "N_holo",
        "Ca_holo",
        "C_holo",
        "node_mask",
        "chi_holo",
        "chi_mask",
        "torsion_apo",
        "torsion_holo",
        "w_res",
    ):
        value = getattr(batch, name, None)
        if value is not None:
            updates[name] = gather_residue_tensor(value, indices)
    return replace_batch(batch, **updates), gather_residue_tensor(aatype, indices)


def torsions_to_atom14(fk_module, rigids: Rigid, aatype: torch.Tensor, torsions: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    torsion_sc = torch.stack([torch.sin(torsions), torch.cos(torsions)], dim=-1)
    torsion_sc = reorder_torsions_to_openfold(torsion_sc)
    result = fk_module(torsion_sc.float(), rigids, aatype.clamp(0, 20))
    return result["atom14_pos"], result["atom14_mask"].bool()


def make_translated_ligand(batch, offset: float = 100.0) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    offset_vec = torch.tensor([offset, offset, offset], device=batch.lig_points.device, dtype=batch.lig_points.dtype)
    return batch.lig_points + offset_vec.view(1, 1, 3), batch.lig_types, batch.lig_mask


def make_shuffled_ligand(batch) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    bsz = batch.lig_points.shape[0]
    if bsz < 2:
        return batch.lig_points, batch.lig_types, batch.lig_mask
    perm = torch.arange(bsz, device=batch.lig_points.device).roll(shifts=1)
    return (
        batch.lig_points.index_select(0, perm),
        batch.lig_types.index_select(0, perm),
        batch.lig_mask.index_select(0, perm),
    )


def make_nolig(batch) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    return batch.lig_points, torch.zeros_like(batch.lig_types), torch.zeros_like(batch.lig_mask)


def build_pair_features(
    apo_atom14: torch.Tensor,
    atom_mask: torch.Tensor,
    ca_apo: torch.Tensor,
    aatype: torch.Tensor,
    lig_points: torch.Tensor,
    lig_types: torch.Tensor,
    lig_mask: torch.Tensor,
    node_mask: torch.Tensor,
    args: argparse.Namespace,
) -> torch.Tensor:
    return build_pair_features_impl(
        apo_atom14,
        atom_mask,
        ca_apo,
        aatype,
        lig_points,
        lig_types,
        lig_mask,
        node_mask,
        interaction_prior_config_from_args(args),
    )


def prepare_batch(batch, fk_module, args):
    device = batch.node_mask.device
    aatype = sequences_to_aatype(batch.sequences, batch.node_mask.shape[1], device)
    batch, aatype = select_local_batch(batch, aatype, args)

    rigids_apo = build_rigids_from_backbone(batch.N_apo.float(), batch.Ca_apo.float(), batch.C_apo.float(), batch.node_mask)
    rigids_holo = build_rigids_from_backbone(batch.N_holo.float(), batch.Ca_holo.float(), batch.C_holo.float(), batch.node_mask)
    apo_atom14, apo_mask = torsions_to_atom14(fk_module, rigids_apo, aatype, batch.torsion_apo.float())
    holo_atom14, holo_mask = torsions_to_atom14(fk_module, rigids_holo, aatype, batch.torsion_holo.float())
    apo_sc_mask = sidechain_atom_mask(apo_mask, batch.node_mask)
    holo_sc_mask = sidechain_atom_mask(holo_mask, batch.node_mask)

    target_dist = min_sidechain_ligand_dist(
        holo_atom14,
        holo_sc_mask,
        batch.lig_points.float(),
        batch.lig_mask.bool(),
        batch.node_mask.bool(),
        int(args.residue_chunk),
    )
    apo_target_dist = min_sidechain_ligand_dist(
        apo_atom14,
        apo_sc_mask,
        batch.lig_points.float(),
        batch.lig_mask.bool(),
        batch.node_mask.bool(),
        int(args.residue_chunk),
    )
    target = (target_dist <= float(args.target_contact_dist)) & batch.node_mask.bool() & holo_sc_mask.any(dim=-1)
    apo_contact = (apo_target_dist <= float(args.target_contact_dist)) & batch.node_mask.bool() & apo_sc_mask.any(dim=-1)
    soft_target = torch.sigmoid((float(args.target_contact_dist) - target_dist.clamp(max=50.0)) / max(float(args.soft_contact_tau), 1e-6))
    soft_target = soft_target * batch.node_mask.float() * holo_sc_mask.any(dim=-1).float()

    apo_labels, apo_valid = chi1_rotamer_labels(batch.torsion_apo[:, :, 3], batch.chi_mask[:, :, 0])
    holo_labels, holo_valid = chi1_rotamer_labels(batch.chi_holo[:, :, 0], batch.chi_mask[:, :, 0])
    valid = apo_valid & holo_valid & batch.node_mask.bool() & holo_sc_mask.any(dim=-1)
    switch = valid & (apo_labels != holo_labels)
    pocket = valid & (batch.w_res > float(args.pocket_threshold))
    contact_gain = target & valid & (apo_target_dist > float(args.target_contact_dist) + float(args.gain_margin))
    masks = {
        "all": valid,
        "target_contact": target & valid,
        "apo_contact": apo_contact & valid,
        "contact_gain": contact_gain,
        "contact_switch": target & switch,
        "gain_switch": contact_gain & switch,
        "pocket_switch": target & pocket & switch,
    }
    return batch, aatype, apo_atom14, apo_sc_mask, target.float(), soft_target.float(), valid, masks


def masked_bce_with_logits(logits, targets, mask, pos_weight_value: float):
    if not mask.any():
        return logits.sum() * 0.0
    pos_weight = None
    if pos_weight_value and pos_weight_value > 0:
        pos_weight = torch.tensor(float(pos_weight_value), device=logits.device, dtype=logits.dtype)
    else:
        positives = (targets[mask] > 0.5).float().sum()
        negatives = mask.float().sum() - positives
        if positives > 0:
            pos_weight = (negatives / positives.clamp(min=1.0)).clamp(min=1.0, max=50.0).detach()
    loss = F.binary_cross_entropy_with_logits(logits, targets, reduction="none", pos_weight=pos_weight)
    return (loss * mask.float()).sum() / mask.float().sum().clamp(min=1.0)


def compute_losses(logits: Dict[str, torch.Tensor], target, soft_target, valid, masks, args):
    correct = logits["correct"]
    loss_bce = masked_bce_with_logits(correct, target, valid, float(args.pos_weight))
    loss_soft = F.mse_loss(torch.sigmoid(correct)[valid], soft_target[valid]) if valid.any() else correct.sum() * 0.0
    zero_target = torch.zeros_like(target)
    loss_decoy_zero = 0.5 * (
        masked_bce_with_logits(logits["nolig"], zero_target, valid, 1.0)
        + masked_bce_with_logits(logits["translated"], zero_target, valid, 1.0)
    )
    pos_mask = masks["target_contact"]
    contrast_terms = []
    for name in CONTROL_NAMES:
        if pos_mask.any():
            contrast_terms.append(F.relu(float(args.decoy_margin) - (correct[pos_mask] - logits[name][pos_mask])).mean())
    loss_contrast = torch.stack(contrast_terms).mean() if contrast_terms else correct.sum() * 0.0
    total = (
        float(args.lambda_bce) * loss_bce
        + float(args.lambda_soft) * loss_soft
        + float(args.lambda_decoy_zero) * loss_decoy_zero
        + float(args.lambda_contrastive) * loss_contrast
    )
    return {
        "loss": total,
        "bce": loss_bce,
        "soft": loss_soft,
        "decoy_zero": loss_decoy_zero,
        "contrastive": loss_contrast,
        "target_contact_pos_count": masks["target_contact"].float().sum(),
    }


def add_loss_sums(sums: Dict[str, float], losses: Dict[str, torch.Tensor], prefix: str):
    sums[f"{prefix}batch_count"] = sums.get(f"{prefix}batch_count", 0.0) + 1.0
    for key, value in losses.items():
        sums[f"{prefix}{key}_sum"] = sums.get(f"{prefix}{key}_sum", 0.0) + float(value.detach().item())


def reduce_sums(sums: Dict[str, float], device: torch.device, distributed: bool) -> Dict[str, float]:
    keys = sorted(sums)
    if not keys:
        return {}
    tensor = torch.tensor([sums[k] for k in keys], dtype=torch.float64, device=device)
    if distributed:
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    return {k: float(tensor[i].item()) for i, k in enumerate(keys)}


def average_precision(scores: torch.Tensor, labels: torch.Tensor) -> float:
    if labels.numel() == 0 or labels.sum() <= 0:
        return float("nan")
    order = torch.argsort(scores, descending=True)
    y = labels[order].float()
    precision = torch.cumsum(y, dim=0) / torch.arange(1, y.numel() + 1, device=y.device, dtype=torch.float32)
    return float((precision * y).sum().item() / y.sum().item())


def auroc(scores: torch.Tensor, labels: torch.Tensor) -> float:
    labels = labels.bool()
    n_pos = int(labels.sum().item())
    n_neg = int((~labels).sum().item())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = torch.argsort(scores)
    ranks = torch.empty_like(order, dtype=torch.float32)
    ranks[order] = torch.arange(1, scores.numel() + 1, device=scores.device, dtype=torch.float32)
    rank_sum_pos = ranks[labels].sum()
    auc = (rank_sum_pos - n_pos * (n_pos + 1) / 2.0) / max(n_pos * n_neg, 1)
    return float(auc.item())


def collect_metric_tensors(logits: Dict[str, torch.Tensor], target, valid, masks) -> Dict[str, torch.Tensor]:
    out = {}
    out["labels_all"] = target[valid].detach().float().cpu()
    for name, value in logits.items():
        out[f"{name}_all"] = value[valid].detach().float().cpu()
    for subset in SUBSET_NAMES:
        mask = masks[subset]
        out[f"{subset}_mask_count"] = torch.tensor([float(mask.float().sum().item())])
        for name, value in logits.items():
            out[f"{name}_{subset}"] = value[mask].detach().float().cpu()
    return out


def merge_metric_tensors(items: Iterable[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    buckets: Dict[str, List[torch.Tensor]] = {}
    for item in items:
        for key, value in item.items():
            buckets.setdefault(key, []).append(value)
    return {key: torch.cat(values, dim=0) if values else torch.empty(0) for key, values in buckets.items()}


def finalize_metrics(loss_sums: Dict[str, float], metric_tensors: Dict[str, torch.Tensor], prefix: str) -> Dict[str, float]:
    out: Dict[str, float] = {}
    batch_count = max(loss_sums.get(f"{prefix}batch_count", 0.0), 1.0)
    for key, value in loss_sums.items():
        if key.startswith(prefix) and key.endswith("_sum"):
            out[key[:-4]] = value / batch_count

    labels = metric_tensors.get("labels_all", torch.empty(0)).float()
    if labels.numel() > 0:
        correct_scores = metric_tensors["correct_all"].float()
        correct_probs = torch.sigmoid(correct_scores)
        pred = correct_probs >= 0.5
        lab = labels.bool()
        tp = float((pred & lab).sum().item())
        fp = float((pred & (~lab)).sum().item())
        fn = float(((~pred) & lab).sum().item())
        precision = tp / max(tp + fp, 1.0)
        recall = tp / max(tp + fn, 1.0)
        out[f"{prefix}contact_precision"] = precision
        out[f"{prefix}contact_recall"] = recall
        out[f"{prefix}contact_f1"] = 2.0 * precision * recall / max(precision + recall, 1e-8)
        out[f"{prefix}contact_ap"] = average_precision(correct_scores, labels)
        out[f"{prefix}contact_auroc"] = auroc(correct_scores, labels)

    for subset in SUBSET_NAMES:
        count = float(metric_tensors.get(f"{subset}_mask_count", torch.zeros(1)).sum().item())
        out[f"{prefix}{subset}_count"] = count
        if count <= 0:
            continue
        correct = metric_tensors.get(f"correct_{subset}", torch.empty(0)).float()
        out[f"{prefix}{subset}_correct_logit"] = float(correct.mean().item()) if correct.numel() else float("nan")
        out[f"{prefix}{subset}_correct_prob"] = float(torch.sigmoid(correct).mean().item()) if correct.numel() else float("nan")
        for name in CONTROL_NAMES:
            decoy = metric_tensors.get(f"{name}_{subset}", torch.empty(0)).float()
            if decoy.numel() == 0 or correct.numel() == 0:
                continue
            out[f"{prefix}{subset}_{name}_logit"] = float(decoy.mean().item())
            out[f"{prefix}{subset}_lift_{name}_logit"] = float((correct - decoy).mean().item())
            out[f"{prefix}{subset}_lift_{name}_prob"] = float((torch.sigmoid(correct) - torch.sigmoid(decoy)).mean().item())
    lifts = []
    for subset in ("target_contact", "contact_gain", "contact_switch", "gain_switch", "pocket_switch"):
        if out.get(f"{prefix}{subset}_count", 0.0) <= 0:
            continue
        for name in CONTROL_NAMES:
            v = out.get(f"{prefix}{subset}_lift_{name}_logit")
            if v is not None and math.isfinite(v):
                lifts.append(v)
    out[f"{prefix}selection_interaction_lift_min"] = min(lifts) if lifts else -1.0
    return out


def forward_controls(model, batch, aatype, apo_atom14, apo_sc_mask, args):
    controls = {
        "correct": (batch.lig_points, batch.lig_types, batch.lig_mask),
        "nolig": make_nolig(batch),
        "translated": make_translated_ligand(batch),
        "shuffled": make_shuffled_ligand(batch),
    }
    logits = {}
    for name, (lig_points, lig_types, lig_mask) in controls.items():
        feats = build_pair_features(
            apo_atom14,
            apo_sc_mask,
            batch.Ca_apo.float(),
            aatype,
            lig_points.float(),
            lig_types.float(),
            lig_mask.bool(),
            batch.node_mask.bool(),
            args,
        )
        logits[name] = model(feats)
    return logits


def train_epoch(model, fk_module, loader, optimizer, device, args, distributed: bool):
    model.train()
    raw_model = model.module if isinstance(model, DDP) else model
    sums: Dict[str, float] = {}
    metric_items: List[Dict[str, torch.Tensor]] = []
    iterator = tqdm(loader, desc="Training", disable=dist.is_initialized() and dist.get_rank() != 0)
    for batch in iterator:
        if batch is None:
            continue
        batch = batch_to_device(batch, device)
        batch, aatype, apo_atom14, apo_sc_mask, target, soft_target, valid, masks = prepare_batch(batch, fk_module, args)
        logits = forward_controls(model, batch, aatype, apo_atom14, apo_sc_mask, args)
        losses = compute_losses(logits, target, soft_target, valid, masks, args)
        optimizer.zero_grad(set_to_none=True)
        losses["loss"].backward()
        if args.grad_clip and args.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(raw_model.parameters(), float(args.grad_clip))
        optimizer.step()
        add_loss_sums(sums, losses, "train_")
        if args.collect_train_metrics:
            metric_items.append(collect_metric_tensors(logits, target, valid, masks))
        iterator.set_postfix({"loss": f"{losses['loss'].item():.3f}", "pos": f"{losses['target_contact_pos_count'].item():.0f}"})
    reduced = reduce_sums(sums, device, distributed)
    if distributed:
        gathered: List[List[Dict[str, torch.Tensor]]] = [None for _ in range(dist.get_world_size())]  # type: ignore[list-item]
        dist.all_gather_object(gathered, metric_items)
        metric_items = [x for sub in gathered for x in sub]
    return finalize_metrics(reduced, merge_metric_tensors(metric_items), "train_")


@torch.no_grad()
def validate(model, fk_module, loader, device, args, distributed: bool):
    model.eval()
    sums: Dict[str, float] = {}
    metric_items: List[Dict[str, torch.Tensor]] = []
    iterator = tqdm(loader, desc="Validation", disable=dist.is_initialized() and dist.get_rank() != 0)
    for batch in iterator:
        if batch is None:
            continue
        batch = batch_to_device(batch, device)
        batch, aatype, apo_atom14, apo_sc_mask, target, soft_target, valid, masks = prepare_batch(batch, fk_module, args)
        logits = forward_controls(model, batch, aatype, apo_atom14, apo_sc_mask, args)
        losses = compute_losses(logits, target, soft_target, valid, masks, args)
        add_loss_sums(sums, losses, "val_")
        metric_items.append(collect_metric_tensors(logits, target, valid, masks))
    reduced = reduce_sums(sums, device, distributed)
    if distributed:
        gathered: List[List[Dict[str, torch.Tensor]]] = [None for _ in range(dist.get_world_size())]  # type: ignore[list-item]
        dist.all_gather_object(gathered, metric_items)
        metric_items = [x for sub in gathered for x in sub]
    return finalize_metrics(reduced, merge_metric_tensors(metric_items), "val_")


def main() -> None:
    args = parse_args()
    distributed = bool(args.distributed)
    local_rank = 0
    world_size = 1
    if distributed:
        dist.init_process_group("nccl", timeout=timedelta(seconds=int(os.environ.get("DDP_TIMEOUT", "7200"))))
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        world_size = dist.get_world_size()
        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")
    else:
        device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    is_main = local_rank == 0

    if is_main:
        Path(args.save_dir).mkdir(parents=True, exist_ok=True)
        Path(args.log_dir).mkdir(parents=True, exist_ok=True)
        print(f"Device: {device}")
        print(f"World size: {world_size}")
        print(f"Args: {json.dumps(vars(args), indent=2)}")

    input_dim = 21 + int(args.num_rbf) + int(args.num_rbf) + int(LIGAND_TYPE_DIM) + 4
    model = InteractionPriorNet(input_dim, int(args.hidden_dim), int(args.num_layers), float(args.dropout)).to(device)
    fk_module = create_openfold_fk().to(device)
    fk_module.eval()
    for param in fk_module.parameters():
        param.requires_grad_(False)

    if distributed:
        model = DDP(model, device_ids=[local_rank], find_unused_parameters=False)

    train_dataset = ApoHoloTripletDataset(
        data_dir=args.data_dir,
        split="train",
        sample_metadata_file=args.sample_metadata_file,
        require_atom14=False,
    )
    train_dataset = deterministic_subset(train_dataset, args.train_max_samples, args.subset_seed, "Train", is_main)
    val_dataset = ApoHoloTripletDataset(
        data_dir=args.data_dir,
        split="val",
        valid_samples_file=args.val_samples_file,
        sample_metadata_file=args.sample_metadata_file,
        require_atom14=False,
    )
    val_dataset = deterministic_subset(val_dataset, args.val_max_samples, args.subset_seed + 1, "Val", is_main)

    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=local_rank, shuffle=True) if distributed else None
    val_sampler = DistributedSampler(val_dataset, num_replicas=world_size, rank=local_rank, shuffle=False) if distributed else None
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=train_sampler is None,
        sampler=train_sampler,
        num_workers=args.num_workers,
        collate_fn=partial(collate_stage1_batch, max_n_res=args.max_n_res),
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        sampler=val_sampler,
        num_workers=args.num_workers,
        collate_fn=partial(collate_stage1_batch, max_n_res=args.max_n_res),
        pin_memory=True,
    )
    if is_main:
        print(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")
        print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    raw_model = model.module if isinstance(model, DDP) else model
    optimizer = torch.optim.AdamW(raw_model.parameters(), lr=float(args.lr), weight_decay=float(args.weight_decay))
    best_metric = -float("inf")
    patience = 0
    metrics_path = Path(args.log_dir) / "metrics.jsonl"

    for epoch in range(int(args.max_epochs)):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        if is_main:
            print(f"\nEpoch {epoch + 1}/{args.max_epochs}")
        train_metrics = train_epoch(model, fk_module, train_loader, optimizer, device, args, distributed)
        val_metrics = validate(model, fk_module, val_loader, device, args, distributed)
        record = {"epoch": epoch, **train_metrics, **val_metrics}
        current = float(record.get("val_selection_interaction_lift_min", -1.0))
        if is_main:
            print(json.dumps(record, indent=2))
            with metrics_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(record) + "\n")
            if math.isfinite(current) and current > best_metric:
                best_metric = current
                patience = 0
                torch.save(
                    {
                        "epoch": epoch,
                        "model_state_dict": raw_model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "selection_interaction_lift_min": current,
                        "args": vars(args),
                    },
                    Path(args.save_dir) / "best_model.pt",
                )
                print(f"  New best interaction lift: {current:.6f}")
            else:
                patience += 1
                print(f"  No improvement ({patience}/{args.patience})")
        if distributed:
            stop = torch.tensor([patience >= int(args.patience)], device=device, dtype=torch.int)
            dist.broadcast(stop, src=0)
            if stop.item():
                break
        elif patience >= int(args.patience):
            break

    if is_main:
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": raw_model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "selection_interaction_lift_min": best_metric,
                "args": vars(args),
            },
            Path(args.save_dir) / "latest_model.pt",
        )
        print(f"Training complete. Best interaction lift: {best_metric:.6f}")
    if distributed:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
