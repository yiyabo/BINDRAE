#!/usr/bin/env python3
"""Train Stage-1-v2 local pocket chi1 relaxation.

This experiment is intentionally narrower than the deterministic Stage-1 head:
given apo backbone and an aligned known-pose ligand, learn whether ligand-pocket
geometry can rerank local chi1 rotamer candidates on contact/switch residues.
The scientific selection metric is correct-ligand lift over no-ligand and
decoy-ligand controls, not global chi accuracy.
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
from typing import Dict, Iterable, Optional, Tuple

import torch
import torch.distributed as dist
import torch.nn as nn
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
from src.stage1.models.adapter import ESMAdapter
from src.stage1.models.fk_openfold import create_openfold_fk
from src.stage1.models.torsion_head import GeometryCandidateScorer
from src.stage1.modules.losses import (
    candidate_decoy_contrastive_loss,
    candidate_decoy_rerank_loss,
    chi1_rotamer_labels,
    g_antiharm_loss,
    g_decoy_contrastive_loss,
    g_switch_amp_loss,
    g_switch_dir_loss,
    g_switch_rank_loss,
)


AA_RESTYPE_MAP = {
    "A": 0, "R": 1, "N": 2, "D": 3, "C": 4,
    "Q": 5, "E": 6, "G": 7, "H": 8, "I": 9,
    "L": 10, "K": 11, "M": 12, "F": 13, "P": 14,
    "S": 15, "T": 16, "W": 17, "Y": 18, "V": 19,
}

CONTROL_NAMES = ("base", "nolig", "shuffled", "translated", "scrambled")
SUBSET_NAMES = ("all", "contact", "switch", "contact_switch", "pocket_switch")


def parse_args():
    parser = argparse.ArgumentParser(description="Train local pocket chi1 relaxation with ligand-causal controls")
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--val_samples_file", default=None)
    parser.add_argument("--sample_metadata_file", default=None)
    parser.add_argument("--stage1_checkpoint", default=None)
    parser.add_argument("--batch_size", type=int, default=24)
    parser.add_argument("--train_max_samples", type=int, default=12000)
    parser.add_argument("--val_max_samples", type=int, default=1000)
    parser.add_argument("--subset_seed", type=int, default=20260617)
    parser.add_argument("--max_n_res", type=int, default=1600)
    parser.add_argument("--max_local_res", type=int, default=192)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--max_epochs", type=int, default=12)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--contact_threshold", type=float, default=0.5)
    parser.add_argument("--pocket_threshold", type=float, default=0.2)
    parser.add_argument("--local_contact_bonus", type=float, default=3.0)
    parser.add_argument("--local_switch_bonus", type=float, default=2.0)
    parser.add_argument("--local_contact_switch_bonus", type=float, default=6.0)
    parser.add_argument("--lambda_base_ce", type=float, default=0.0)
    parser.add_argument("--lambda_contact_ce", type=float, default=0.2)
    parser.add_argument("--lambda_switch_ce", type=float, default=0.6)
    parser.add_argument("--lambda_rerank", type=float, default=1.0)
    parser.add_argument("--lambda_contrastive", type=float, default=1.0)
    parser.add_argument("--lambda_g", type=float, default=0.2)
    parser.add_argument("--lambda_antiharm", type=float, default=0.05)
    parser.add_argument("--decoy_margin", type=float, default=0.2)
    parser.add_argument("--rank_margin", type=float, default=0.05)
    parser.add_argument("--g_margin", type=float, default=0.05)
    parser.add_argument("--residual_beta", type=float, default=1.0)
    parser.add_argument("--base_temperature", type=float, default=8.0)
    parser.add_argument("--reset_residual", action="store_true")
    parser.add_argument("--freeze_base", action="store_true")
    parser.add_argument("--freeze_gate", action="store_true")
    parser.add_argument("--use_typed_energy", action="store_true")
    parser.add_argument("--save_dir", default="checkpoints/stage1/pocket_relax_v2")
    parser.add_argument("--log_dir", default="logs/stage1/pocket_relax_v2")
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--distributed", action="store_true")
    return parser.parse_args()


def batch_to_device(batch, device):
    for name in (
        "esm",
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
        "atom14_holo",
        "atom14_holo_mask",
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


def replace_batch(batch, **updates):
    if is_dataclass(batch):
        return replace(batch, **updates)
    clone = type("BatchClone", (), {})()
    clone.__dict__.update(getattr(batch, "__dict__", {}))
    clone.__dict__.update(updates)
    return clone


def make_nolig_batch(batch):
    return replace_batch(
        batch,
        lig_points=torch.zeros_like(batch.lig_points),
        lig_types=torch.zeros_like(batch.lig_types),
        lig_mask=torch.zeros_like(batch.lig_mask),
    )


def make_translated_batch(batch, offset: float = 100.0):
    offset_vec = torch.tensor([offset, offset, offset], device=batch.lig_points.device, dtype=batch.lig_points.dtype)
    return replace_batch(batch, lig_points=batch.lig_points + offset_vec.view(1, 1, 3))


def make_shuffled_batch(batch):
    bsz = batch.lig_points.shape[0]
    if bsz < 2:
        return replace_batch(batch)
    perm = torch.arange(bsz, device=batch.lig_points.device).roll(shifts=1)
    return replace_batch(
        batch,
        lig_points=batch.lig_points.index_select(0, perm),
        lig_types=batch.lig_types.index_select(0, perm),
        lig_mask=batch.lig_mask.index_select(0, perm),
    )


def make_scrambled_types_batch(batch):
    lig_types = batch.lig_types.clone()
    for i in range(lig_types.shape[0]):
        valid = torch.nonzero(batch.lig_mask[i].bool(), as_tuple=False).squeeze(-1)
        if valid.numel() > 1:
            lig_types[i, valid] = batch.lig_types[i, valid.roll(shifts=1)]
    return replace_batch(batch, lig_types=lig_types)


def sequences_to_aatype(sequences, max_len: int, device: torch.device) -> torch.Tensor:
    aatype = torch.zeros(len(sequences), max_len, dtype=torch.long, device=device)
    for i, seq in enumerate(sequences):
        for j, aa in enumerate(seq):
            if j >= max_len:
                break
            aatype[i, j] = AA_RESTYPE_MAP.get(aa, 0)
    return aatype


def build_rigids_from_backbone(N, Ca, C, mask, eps: float = 1e-6) -> Rigid:
    device = Ca.device
    default_e1 = torch.tensor([1.0, 0.0, 0.0], device=device, dtype=Ca.dtype)
    default_e2 = torch.tensor([0.0, 1.0, 0.0], device=device, dtype=Ca.dtype)
    default_e3 = torch.tensor([0.0, 0.0, 1.0], device=device, dtype=Ca.dtype)

    e1 = C - Ca
    e1_norm = torch.norm(e1, dim=-1, keepdim=True)
    e1 = torch.where(e1_norm > eps, e1 / e1_norm.clamp_min(eps), default_e1.expand_as(e1))

    u = N - Ca
    proj = (u * e1).sum(dim=-1, keepdim=True) * e1
    e2 = u - proj
    e2_norm = torch.norm(e2, dim=-1, keepdim=True)
    e2 = torch.where(e2_norm > eps, e2 / e2_norm.clamp_min(eps), default_e2.expand_as(e2))

    e3 = torch.cross(e1, e2, dim=-1)
    e3_norm = torch.norm(e3, dim=-1, keepdim=True)
    e3 = torch.where(e3_norm > eps, e3 / e3_norm.clamp_min(eps), default_e3.expand_as(e3))
    e2 = torch.cross(e3, e1, dim=-1)

    R = torch.stack([e1, e2, e3], dim=-1)
    valid = mask.bool().unsqueeze(-1).unsqueeze(-1)
    eye = torch.eye(3, device=device, dtype=Ca.dtype).view(1, 1, 3, 3)
    R = torch.where(valid, R, eye.expand_as(R))
    t = torch.where(mask.bool().unsqueeze(-1), Ca, torch.zeros_like(Ca))
    return Rigid(rots=Rotation(rot_mats=R), trans=t)


def gather_residue_tensor(tensor: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
    if tensor.ndim == 2:
        return tensor.gather(1, indices)
    view_shape = [indices.shape[0], indices.shape[1]] + [1] * (tensor.ndim - 2)
    expand_shape = [indices.shape[0], indices.shape[1]] + list(tensor.shape[2:])
    return tensor.gather(1, indices.view(*view_shape).expand(*expand_shape))


def select_local_batch(batch, aatype: torch.Tensor, args):
    B, N = batch.node_mask.shape
    K = min(int(args.max_local_res), N)
    if K >= N:
        return batch, aatype

    apo_labels, apo_valid = chi1_rotamer_labels(batch.torsion_apo[:, :, 3], batch.chi_mask[:, :, 0])
    holo_labels, holo_valid = chi1_rotamer_labels(batch.chi_holo[:, :, 0], batch.chi_mask[:, :, 0])
    valid = apo_valid & holo_valid & batch.node_mask.bool()
    contact = valid & (batch.w_res > float(args.contact_threshold))
    switch = valid & (apo_labels != holo_labels)
    contact_switch = contact & switch

    score = batch.w_res.float().clone()
    score = score + float(args.local_contact_bonus) * contact.float()
    score = score + float(args.local_switch_bonus) * switch.float()
    score = score + float(args.local_contact_switch_bonus) * contact_switch.float()
    score = score + 0.01 * valid.float()
    score = score.masked_fill(~batch.node_mask.bool(), -1e9)

    indices = score.topk(k=K, dim=1).indices
    updates = {}
    for name in (
        "esm",
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
        "atom14_holo",
        "atom14_holo_mask",
    ):
        value = getattr(batch, name, None)
        if value is not None:
            updates[name] = gather_residue_tensor(value, indices)

    local_aatype = gather_residue_tensor(aatype, indices)
    return replace_batch(batch, **updates), local_aatype


def masked_cross_entropy(logits, labels, mask):
    grad_anchor = logits.sum() * 0.0
    if not mask.any():
        return grad_anchor
    return F.cross_entropy(logits[mask], labels[mask])


class PocketRelaxV2(nn.Module):
    """Local candidate chi1 scorer with explicit base prior and ligand residual."""

    def __init__(self, use_typed_energy: bool = True):
        super().__init__()
        self.esm_adapter = ESMAdapter(esm_dim=1280, output_dim=384, dropout=0.1)
        self.fk_module = create_openfold_fk()
        for param in self.fk_module.parameters():
            param.requires_grad = False
        self.scorer = GeometryCandidateScorer(
            c_hidden=64,
            num_rbf=16,
            dropout=0.0,
            use_sgeo=True,
            c_s=384,
            sgeo_proj_dim=32,
            bounded_residual=True,
            residual_max=5.0,
            residual_tau=2.0,
            gate_norm=True,
            gate_init_bias=2.0,
            use_typed_energy=bool(use_typed_energy),
            lig_type_dim=20,
            typed_pair_dim=64,
            typed_cutoff=6.0,
            typed_init_scale=0.1,
        )
        for param in self.esm_adapter.parameters():
            param.requires_grad = False

    def load_stage1_modules(self, checkpoint: Dict[str, torch.Tensor]) -> Dict[str, int]:
        state = checkpoint.get("model_state_dict", checkpoint)
        own_state = self.state_dict()
        loadable = {}
        skipped = 0
        for key, value in state.items():
            clean_key = key[len("module.") :] if key.startswith("module.") else key
            if clean_key.startswith("esm_adapter."):
                clean_key = clean_key
            elif clean_key.startswith("geometry_candidate_scorer."):
                clean_key = "scorer." + clean_key[len("geometry_candidate_scorer.") :]
            elif clean_key.startswith("scorer."):
                pass
            else:
                continue
            if clean_key in own_state and tuple(own_state[clean_key].shape) == tuple(value.shape):
                loadable[clean_key] = value
            else:
                skipped += 1
        self.load_state_dict(loadable, strict=False)
        return {"loaded": len(loadable), "skipped": skipped}

    def apply_controls(self, reset_residual: bool, freeze_base: bool, freeze_gate: bool):
        if reset_residual:
            self.scorer.reset_residual_and_gate()
        if freeze_base:
            self.scorer.freeze_base(True)
        self.scorer.freeze_gate(True)

    def forward(self, batch, aatype, base_detach: bool = True, beta: float = 1.0):
        rigids = build_rigids_from_backbone(batch.N_apo, batch.Ca_apo, batch.C_apo, batch.node_mask)
        with torch.no_grad():
            s_geo = self.esm_adapter(batch.esm)
        return self.scorer(
            self.fk_module,
            rigids,
            aatype,
            batch.torsion_apo,
            batch.lig_points,
            batch.lig_types,
            batch.lig_mask,
            batch.node_mask,
            s_geo=s_geo,
            s_lig=None,
            return_decomposition=True,
            base_detach=base_detach,
            gate_override=1.0,
            beta=beta,
        )


def subset_masks(batch, args):
    apo_labels, apo_valid = chi1_rotamer_labels(batch.torsion_apo[:, :, 3], batch.chi_mask[:, :, 0])
    holo_labels, holo_valid = chi1_rotamer_labels(batch.chi_holo[:, :, 0], batch.chi_mask[:, :, 0])
    valid = apo_valid & holo_valid & batch.node_mask.bool()
    contact = valid & (batch.w_res > float(args.contact_threshold))
    pocket = valid & (batch.w_res > float(args.pocket_threshold))
    switch = valid & (apo_labels != holo_labels)
    masks = {
        "all": valid,
        "contact": contact,
        "switch": switch,
        "contact_switch": contact & switch,
        "pocket_switch": pocket & switch,
    }
    return apo_labels, holo_labels, valid, masks


def compute_losses(outputs: Dict[str, torch.Tensor], batch, args):
    correct = outputs["correct"]
    base = outputs["base"]
    apo_labels, holo_labels, valid, masks = subset_masks(batch, args)
    contact = masks["contact"]
    contact_switch = masks["contact_switch"]

    base_ce = masked_cross_entropy(base, holo_labels, valid)
    contact_ce = masked_cross_entropy(correct, holo_labels, contact)
    switch_ce = masked_cross_entropy(correct, holo_labels, contact_switch)

    rerank_terms = []
    contrastive_terms = []
    g_decoy_terms = []
    for name in ("nolig", "shuffled", "translated", "scrambled"):
        decoy = outputs[name]
        rerank_terms.append(
            candidate_decoy_rerank_loss(
                correct,
                decoy,
                batch.torsion_apo[:, :, 3],
                batch.chi_holo[:, :, 0],
                batch.chi_mask[:, :, 0],
                contact_mask=contact,
                decoy_margin=float(args.decoy_margin),
                rank_margin=float(args.rank_margin),
            )
        )
        contrastive_terms.append(
            candidate_decoy_contrastive_loss(
                correct,
                decoy,
                batch.torsion_apo[:, :, 3],
                batch.chi_holo[:, :, 0],
                batch.chi_mask[:, :, 0],
                contact_mask=contact,
                decoy_margin=float(args.decoy_margin),
                repulsion_margin=float(args.rank_margin),
            )
        )
        g_decoy_terms.append(
            g_decoy_contrastive_loss(
                correct,
                decoy,
                base,
                batch.torsion_apo[:, :, 3],
                batch.chi_holo[:, :, 0],
                batch.chi_mask[:, :, 0],
                contact_mask=contact,
                margin=float(args.g_margin),
            )
        )

    rerank = torch.stack(rerank_terms).mean()
    contrastive = torch.stack(contrastive_terms).mean()
    g_decoy = torch.stack(g_decoy_terms).mean()
    g_dir = g_switch_dir_loss(
        correct,
        base,
        batch.torsion_apo[:, :, 3],
        batch.chi_holo[:, :, 0],
        batch.chi_mask[:, :, 0],
        contact_mask=contact,
    )
    g_amp = g_switch_amp_loss(
        correct,
        base,
        batch.torsion_apo[:, :, 3],
        batch.chi_holo[:, :, 0],
        batch.chi_mask[:, :, 0],
        contact_mask=contact,
        margin=float(args.g_margin),
    )
    g_rank = g_switch_rank_loss(
        correct,
        base,
        batch.torsion_apo[:, :, 3],
        batch.chi_holo[:, :, 0],
        batch.chi_mask[:, :, 0],
        contact_mask=contact,
        margin=float(args.g_margin),
    )
    antiharm = g_antiharm_loss(
        correct,
        base,
        batch.torsion_apo[:, :, 3],
        batch.chi_holo[:, :, 0],
        batch.chi_mask[:, :, 0],
        contact_mask=contact,
        tau=float(args.g_margin),
    )

    total = (
        float(args.lambda_base_ce) * base_ce
        + float(args.lambda_contact_ce) * contact_ce
        + float(args.lambda_switch_ce) * switch_ce
        + float(args.lambda_rerank) * rerank
        + float(args.lambda_contrastive) * contrastive
        + float(args.lambda_g) * (g_dir + g_amp + g_rank + g_decoy)
        + float(args.lambda_antiharm) * antiharm
    )
    return {
        "loss": total,
        "base_ce": base_ce,
        "contact_ce": contact_ce,
        "switch_ce": switch_ce,
        "rerank": rerank,
        "contrastive": contrastive,
        "g_dir": g_dir,
        "g_amp": g_amp,
        "g_rank": g_rank,
        "g_decoy": g_decoy,
        "antiharm": antiharm,
        "contact_switch_count": contact_switch.float().sum(),
    }


def metric_sums(outputs: Dict[str, torch.Tensor], batch, args, prefix: str = "") -> Dict[str, float]:
    _, holo_labels, _, masks = subset_masks(batch, args)
    sums: Dict[str, float] = {}
    logits_by_name = {"correct": outputs["correct"], **{name: outputs[name] for name in CONTROL_NAMES}}
    logp_by_name = {name: F.log_softmax(logits, dim=-1) for name, logits in logits_by_name.items()}

    for subset_name in SUBSET_NAMES:
        mask = masks[subset_name]
        count = float(mask.float().sum().item())
        sums[f"{prefix}{subset_name}_count"] = count
        for name, logits in logits_by_name.items():
            pred = logits.argmax(dim=-1)
            hit = ((pred == holo_labels) & mask).float().sum()
            gold = logp_by_name[name].gather(-1, holo_labels.unsqueeze(-1)).squeeze(-1)
            sums[f"{prefix}{subset_name}_{name}_hits"] = float(hit.item())
            sums[f"{prefix}{subset_name}_{name}_logp_sum"] = float((gold * mask.float()).sum().item())

        correct_pred = logits_by_name["correct"].argmax(dim=-1)
        correct_hit = (correct_pred == holo_labels).float()
        correct_gold = logp_by_name["correct"].gather(-1, holo_labels.unsqueeze(-1)).squeeze(-1)
        for control in CONTROL_NAMES:
            control_pred = logits_by_name[control].argmax(dim=-1)
            control_hit = (control_pred == holo_labels).float()
            control_gold = logp_by_name[control].gather(-1, holo_labels.unsqueeze(-1)).squeeze(-1)
            sums[f"{prefix}{subset_name}_lift_{control}_acc_sum"] = float(
                ((correct_hit - control_hit) * mask.float()).sum().item()
            )
            sums[f"{prefix}{subset_name}_lift_{control}_logp_sum"] = float(
                ((correct_gold - control_gold) * mask.float()).sum().item()
            )
    return sums


def add_loss_sums(target: Dict[str, float], losses: Dict[str, torch.Tensor], prefix: str = ""):
    target[f"{prefix}batch_count"] = target.get(f"{prefix}batch_count", 0.0) + 1.0
    for key, value in losses.items():
        target[f"{prefix}{key}_sum"] = target.get(f"{prefix}{key}_sum", 0.0) + float(value.detach().item())


def merge_sums(target: Dict[str, float], update: Dict[str, float]):
    for key, value in update.items():
        target[key] = target.get(key, 0.0) + float(value)


def reduce_sums(sums: Dict[str, float], device, distributed: bool) -> Dict[str, float]:
    keys = sorted(sums)
    if not keys:
        return {}
    tensor = torch.tensor([sums[k] for k in keys], device=device, dtype=torch.float64)
    if distributed:
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    return {k: float(tensor[i].item()) for i, k in enumerate(keys)}


def finalize_metrics(sums: Dict[str, float], prefix: str = "") -> Dict[str, float]:
    out: Dict[str, float] = {}
    batch_count = max(sums.get(f"{prefix}batch_count", 0.0), 1.0)
    for key, value in sums.items():
        if key.startswith(prefix) and key.endswith("_sum") and not any(x in key for x in ("logp_sum", "acc_sum")):
            metric = key[len(prefix) : -4]
            out[f"{prefix}{metric}"] = value / batch_count

    for subset_name in SUBSET_NAMES:
        count = sums.get(f"{prefix}{subset_name}_count", 0.0)
        out[f"{prefix}{subset_name}_count"] = count
        denom = max(count, 1.0)
        for name in ("correct", *CONTROL_NAMES):
            out[f"{prefix}{subset_name}_{name}_acc"] = sums.get(
                f"{prefix}{subset_name}_{name}_hits", 0.0
            ) / denom
            out[f"{prefix}{subset_name}_{name}_logp"] = sums.get(
                f"{prefix}{subset_name}_{name}_logp_sum", 0.0
            ) / denom
        for control in CONTROL_NAMES:
            out[f"{prefix}{subset_name}_lift_{control}_acc"] = sums.get(
                f"{prefix}{subset_name}_lift_{control}_acc_sum", 0.0
            ) / denom
            out[f"{prefix}{subset_name}_lift_{control}_logp"] = sums.get(
                f"{prefix}{subset_name}_lift_{control}_logp_sum", 0.0
            ) / denom
    return out


def selection_lift(metrics: Dict[str, float], prefix: str = "val_") -> float:
    candidates = []
    for subset in ("contact_switch", "pocket_switch", "switch"):
        if metrics.get(f"{prefix}{subset}_count", 0.0) < 1.0:
            continue
        lifts = [
            metrics.get(f"{prefix}{subset}_lift_nolig_logp", -1.0),
            metrics.get(f"{prefix}{subset}_lift_shuffled_logp", -1.0),
            metrics.get(f"{prefix}{subset}_lift_translated_logp", -1.0),
        ]
        scrambled_key = f"{prefix}{subset}_lift_scrambled_logp"
        if scrambled_key in metrics:
            lifts.append(metrics.get(scrambled_key, -1.0))
        candidates.append((subset, min(lifts)))
    if not candidates:
        return -1.0
    return candidates[0][1]


def forward_all(model, batch, aatype, args):
    raw_model = model.module if isinstance(model, DDP) else model
    correct, base, _, _, _ = model(batch, aatype, base_detach=bool(args.freeze_base), beta=float(args.residual_beta))
    base_scale = 1.0 / max(float(args.base_temperature), 1e-3)
    base_eval = base * base_scale
    correct = base_eval + (correct - base)
    with torch.no_grad():
        base_only = base_eval
    nolig, _, _, _, _ = model(make_nolig_batch(batch), aatype, base_detach=bool(args.freeze_base), beta=float(args.residual_beta))
    nolig = base_eval + (nolig - base)
    shuffled, _, _, _, _ = model(make_shuffled_batch(batch), aatype, base_detach=bool(args.freeze_base), beta=float(args.residual_beta))
    shuffled = base_eval + (shuffled - base)
    translated, _, _, _, _ = model(make_translated_batch(batch), aatype, base_detach=bool(args.freeze_base), beta=float(args.residual_beta))
    translated = base_eval + (translated - base)
    scrambled, _, _, _, _ = model(make_scrambled_types_batch(batch), aatype, base_detach=bool(args.freeze_base), beta=float(args.residual_beta))
    scrambled = base_eval + (scrambled - base)
    _ = raw_model
    return {
        "correct": correct,
        "base": base_only,
        "nolig": nolig,
        "shuffled": shuffled,
        "translated": translated,
        "scrambled": scrambled,
    }


def train_epoch(model, loader, optimizer, device, args, distributed: bool):
    model.train()
    raw_model = model.module if isinstance(model, DDP) else model
    sums: Dict[str, float] = {}
    iterator = tqdm(loader, desc="Training", disable=dist.is_initialized() and dist.get_rank() != 0)
    for batch in iterator:
        if batch is None:
            continue
        batch = batch_to_device(batch, device)
        aatype = sequences_to_aatype(batch.sequences, batch.node_mask.shape[1], device)
        batch, aatype = select_local_batch(batch, aatype, args)

        outputs = forward_all(model, batch, aatype, args)
        losses = compute_losses(outputs, batch, args)

        optimizer.zero_grad(set_to_none=True)
        losses["loss"].backward()
        if args.grad_clip and args.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_((p for p in raw_model.parameters() if p.requires_grad), args.grad_clip)
        optimizer.step()

        add_loss_sums(sums, losses, prefix="train_")
        with torch.no_grad():
            merge_sums(sums, metric_sums(outputs, batch, args, prefix="train_"))
        iterator.set_postfix(
            {
                "loss": f"{losses['loss'].item():.3f}",
                "csw": f"{losses['contact_switch_count'].item():.0f}",
            }
        )
    return finalize_metrics(reduce_sums(sums, device, distributed), prefix="train_")


@torch.no_grad()
def validate(model, loader, device, args, distributed: bool):
    model.eval()
    sums: Dict[str, float] = {}
    iterator = tqdm(loader, desc="Validation", disable=dist.is_initialized() and dist.get_rank() != 0)
    for batch in iterator:
        if batch is None:
            continue
        batch = batch_to_device(batch, device)
        aatype = sequences_to_aatype(batch.sequences, batch.node_mask.shape[1], device)
        batch, aatype = select_local_batch(batch, aatype, args)
        outputs = forward_all(model, batch, aatype, args)
        losses = compute_losses(outputs, batch, args)
        add_loss_sums(sums, losses, prefix="val_")
        merge_sums(sums, metric_sums(outputs, batch, args, prefix="val_"))
    return finalize_metrics(reduce_sums(sums, device, distributed), prefix="val_")


def main():
    args = parse_args()
    distributed = args.distributed
    local_rank = 0
    world_size = 1
    is_main = True

    if distributed:
        if not dist.is_initialized():
            dist.init_process_group("nccl", timeout=timedelta(seconds=int(os.environ.get("DDP_TIMEOUT", "7200"))))
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        world_size = dist.get_world_size()
        is_main = local_rank == 0
        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")
    else:
        device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    save_dir = Path(args.save_dir)
    log_dir = Path(args.log_dir)
    if is_main:
        save_dir.mkdir(parents=True, exist_ok=True)
        log_dir.mkdir(parents=True, exist_ok=True)
        print(f"Using device: {device}")
        print(f"World size: {world_size}")
        print(f"Args: {json.dumps(vars(args), indent=2)}")

    model = PocketRelaxV2(use_typed_energy=args.use_typed_energy).to(device)
    if args.stage1_checkpoint:
        ckpt = torch.load(args.stage1_checkpoint, map_location=device)
        report = model.load_stage1_modules(ckpt)
        if is_main:
            print(f"Loaded geometry scorer modules: {report}")
    model.apply_controls(args.reset_residual, args.freeze_base, args.freeze_gate)

    trainable = [p for p in model.parameters() if p.requires_grad]
    if is_main:
        n_trainable = sum(p.numel() for p in trainable)
        print(f"Trainable parameters: {n_trainable:,}")
    if not trainable:
        raise RuntimeError("No trainable parameters")

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

    raw_model = model.module if isinstance(model, DDP) else model
    optimizer = torch.optim.AdamW((p for p in raw_model.parameters() if p.requires_grad), lr=args.lr, weight_decay=args.weight_decay)
    best_metric = -float("inf")
    patience = 0

    for epoch in range(args.max_epochs):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        if is_main:
            print(f"\nEpoch {epoch + 1}/{args.max_epochs}")

        train_metrics = train_epoch(model, train_loader, optimizer, device, args, distributed)
        val_metrics = validate(model, val_loader, device, args, distributed)
        record = {"epoch": epoch, **train_metrics, **val_metrics}
        record["selection_contact_switch_ligand_lift_logp"] = selection_lift(record, prefix="val_")

        if is_main:
            print(json.dumps(record, indent=2))
            with open(log_dir / "metrics.jsonl", "a") as f:
                f.write(json.dumps(record) + "\n")

            raw_model = model.module if isinstance(model, DDP) else model
            current = record["selection_contact_switch_ligand_lift_logp"]
            if math.isfinite(current) and current > best_metric:
                best_metric = current
                patience = 0
                torch.save(
                    {
                        "epoch": epoch,
                        "model_state_dict": raw_model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "selection_contact_switch_ligand_lift_logp": current,
                        "args": vars(args),
                    },
                    save_dir / "best_model.pt",
                )
                print(f"  New best ligand-causal lift: {current:.6f}")
            else:
                patience += 1
                print(f"  No improvement ({patience}/{args.patience})")

        if distributed:
            stop = torch.tensor([patience >= args.patience], device=device, dtype=torch.int)
            dist.broadcast(stop, src=0)
            if stop.item():
                break
        elif patience >= args.patience:
            break

    if is_main:
        raw_model = model.module if isinstance(model, DDP) else model
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": raw_model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "selection_contact_switch_ligand_lift_logp": best_metric,
                "args": vars(args),
            },
            save_dir / "latest_model.pt",
        )
        print(f"Training complete. Best ligand-causal lift: {best_metric:.6f}")

    if distributed:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
