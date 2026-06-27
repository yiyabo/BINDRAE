#!/usr/bin/env python3
"""Audit whether ligand geometry can rank holo chi1 rotamer candidates.

This diagnostic does not train a model.  It uses the same Stage-1 triplet
loader, apo-backbone FK, and chi1 rotamer bins as the pocket-relax experiment,
then asks a narrow question:

    Do simple sidechain-ligand geometry scores make the holo chi1 bin rank
    better with the correct known-pose ligand than with no/translated/shuffled
    ligand controls?

If the answer is weak even for this oracle-style audit, more Stage-1 residual
training is unlikely to recover robust ligand-causal signal without changing
the supervision/data framing.
"""

import argparse
import csv
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
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))
flash_ipa_path = project_root / "vendor" / "flash_ipa" / "src"
if flash_ipa_path.exists():
    sys.path.insert(0, str(flash_ipa_path))

from flash_ipa.rigid import Rigid, Rotation

from src.stage1.datasets import ApoHoloTripletDataset, collate_stage1_batch
from src.stage1.models.fk_openfold import create_openfold_fk
from src.stage1.models.torsion_head import compute_candidate_geometries
from src.stage1.modules.losses import chi1_rotamer_labels


AA_RESTYPE_MAP = {
    "A": 0, "R": 1, "N": 2, "D": 3, "C": 4,
    "Q": 5, "E": 6, "G": 7, "H": 8, "I": 9,
    "L": 10, "K": 11, "M": 12, "F": 13, "P": 14,
    "S": 15, "T": 16, "W": 17, "Y": 18, "V": 19,
}

CONTROL_NAMES = ("nolig", "translated", "shuffled")
SUBSET_NAMES = ("all", "contact", "switch", "contact_switch", "pocket_switch")
SCORE_NAMES = ("shell35", "shell45", "contact", "closest", "anticlash")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit ligand rotamer-oracle geometry signal")
    parser.add_argument("--data_dir", default="processed_data/triplets")
    parser.add_argument("--split", default="val", choices=["train", "val", "test"])
    parser.add_argument("--valid_samples_file", default=None)
    parser.add_argument("--sample_metadata_file", default="sample_metadata.json")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--max_n_res", type=int, default=1600)
    parser.add_argument("--max_samples", type=int, default=0)
    parser.add_argument("--max_batches", type=int, default=0)
    parser.add_argument("--subset_seed", type=int, default=20260617)
    parser.add_argument("--max_local_res", type=int, default=192)
    parser.add_argument("--contact_threshold", type=float, default=0.5)
    parser.add_argument("--pocket_threshold", type=float, default=0.2)
    parser.add_argument("--local_contact_bonus", type=float, default=3.0)
    parser.add_argument("--local_switch_bonus", type=float, default=2.0)
    parser.add_argument("--local_contact_switch_bonus", type=float, default=6.0)
    parser.add_argument("--residue_chunk", type=int, default=64)
    parser.add_argument("--clash_dist", type=float, default=2.4)
    parser.add_argument("--contact_dist", type=float, default=4.5)
    parser.add_argument("--contact_sigma", type=float, default=0.6)
    parser.add_argument("--shell_sigma", type=float, default=0.8)
    parser.add_argument("--clash_weight", type=float, default=2.0)
    parser.add_argument("--eps", type=float, default=1e-6)
    parser.add_argument("--output_dir", default="logs/stage1_diagnostics/rotamer_oracle_signal")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--distributed", action="store_true")
    return parser.parse_args()


def batch_to_device(batch, device: torch.device):
    for name in (
        "N_apo",
        "Ca_apo",
        "C_apo",
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


def replace_batch(batch, **updates):
    if is_dataclass(batch):
        return replace(batch, **updates)
    clone = type("BatchClone", (), {})()
    clone.__dict__.update(getattr(batch, "__dict__", {}))
    clone.__dict__.update(updates)
    return clone


def sequences_to_aatype(sequences: List[str], max_len: int, device: torch.device) -> torch.Tensor:
    aatype = torch.zeros(len(sequences), max_len, dtype=torch.long, device=device)
    for i, seq in enumerate(sequences):
        for j, aa in enumerate(seq):
            if j >= max_len:
                break
            aatype[i, j] = AA_RESTYPE_MAP.get(aa, 0)
    return aatype


def build_rigids_from_backbone(
    n_coord: torch.Tensor,
    ca_coord: torch.Tensor,
    c_coord: torch.Tensor,
    mask: torch.Tensor,
    eps: float = 1e-6,
) -> Rigid:
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
    bsz, n_res = batch.node_mask.shape
    k_res = min(int(args.max_local_res), n_res)
    if k_res >= n_res:
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
    indices = score.topk(k=k_res, dim=1).indices

    updates = {}
    for name in (
        "N_apo",
        "Ca_apo",
        "C_apo",
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


def make_translated_ligand(batch, offset: float = 100.0) -> Tuple[torch.Tensor, torch.Tensor]:
    offset_vec = torch.tensor([offset, offset, offset], device=batch.lig_points.device, dtype=batch.lig_points.dtype)
    return batch.lig_points + offset_vec.view(1, 1, 3), batch.lig_mask


def make_shuffled_ligand(batch) -> Tuple[torch.Tensor, torch.Tensor]:
    bsz = batch.lig_points.shape[0]
    if bsz < 2:
        return batch.lig_points, batch.lig_mask
    perm = torch.arange(bsz, device=batch.lig_points.device).roll(shifts=1)
    return batch.lig_points.index_select(0, perm), batch.lig_mask.index_select(0, perm)


def candidate_min_dist(
    cand_atom14: torch.Tensor,
    chi1_dep_mask: torch.Tensor,
    lig_points: torch.Tensor,
    lig_mask: torch.Tensor,
    node_mask: torch.Tensor,
    residue_chunk: int,
) -> torch.Tensor:
    bsz, n_res, n_cand, n_atom, _ = cand_atom14.shape
    n_lig = lig_points.shape[1]
    out = cand_atom14.new_full((bsz, n_res, n_cand), float("inf"))
    if n_lig == 0:
        return out

    chunk = max(int(residue_chunk), 1)
    for start in range(0, n_res, chunk):
        end = min(start + chunk, n_res)
        c = end - start
        sc = cand_atom14[:, start:end]  # [B, C, 3, 14, 3]
        dists = torch.cdist(
            sc.reshape(bsz * c * n_cand, n_atom, 3),
            lig_points[:, None, :, :].expand(bsz, c * n_cand, n_lig, 3).reshape(bsz * c * n_cand, n_lig, 3),
        ).reshape(bsz, c, n_cand, n_atom, n_lig)

        atom_valid = chi1_dep_mask[:, start:end].unsqueeze(2).unsqueeze(-1)
        lig_valid = lig_mask[:, None, None, None, :].bool()
        res_valid = node_mask[:, start:end, None, None, None].bool()
        valid = atom_valid & lig_valid & res_valid
        dists = dists.masked_fill(~valid.expand_as(dists), float("inf"))
        out[:, start:end] = dists.amin(dim=(-1, -2))
    return out


def score_from_min_dist(min_dist: torch.Tensor, args: argparse.Namespace) -> Dict[str, torch.Tensor]:
    finite = torch.isfinite(min_dist) & (min_dist < 1e5)
    d = torch.where(finite, min_dist.clamp(max=50.0), min_dist.new_full((), 50.0))
    clash_penalty = torch.relu(float(args.clash_dist) - d).pow(2) / max(float(args.clash_dist) ** 2, 1e-6)
    contact = torch.sigmoid((float(args.contact_dist) - d) / max(float(args.contact_sigma), 1e-6))
    shell35 = torch.exp(-0.5 * ((d - 3.5) / max(float(args.shell_sigma), 1e-6)).pow(2))
    shell45 = torch.exp(-0.5 * ((d - 4.5) / max(float(args.shell_sigma), 1e-6)).pow(2))
    closest = 1.0 / (d + 0.5)
    valid_f = finite.to(d.dtype)

    return {
        "shell35": valid_f * shell35 - float(args.clash_weight) * clash_penalty,
        "shell45": valid_f * shell45 - float(args.clash_weight) * clash_penalty,
        "contact": valid_f * contact - float(args.clash_weight) * clash_penalty,
        "closest": valid_f * closest - float(args.clash_weight) * clash_penalty,
        "anticlash": -clash_penalty,
    }


def subset_masks(batch, chi1_dep_mask: torch.Tensor, args: argparse.Namespace):
    apo_labels, apo_valid = chi1_rotamer_labels(batch.torsion_apo[:, :, 3], batch.chi_mask[:, :, 0])
    holo_labels, holo_valid = chi1_rotamer_labels(batch.chi_holo[:, :, 0], batch.chi_mask[:, :, 0])
    dep_valid = chi1_dep_mask.any(dim=-1)
    valid = apo_valid & holo_valid & batch.node_mask.bool() & dep_valid
    contact = valid & (batch.w_res > float(args.contact_threshold))
    pocket = valid & (batch.w_res > float(args.pocket_threshold))
    switch = valid & (apo_labels != holo_labels)
    return apo_labels, holo_labels, {
        "all": valid,
        "contact": contact,
        "switch": switch,
        "contact_switch": contact & switch,
        "pocket_switch": pocket & switch,
    }


def add(stats: Dict[str, float], key: str, value: torch.Tensor | float) -> None:
    if isinstance(value, torch.Tensor):
        value = float(value.detach().double().sum().item())
    stats[key] = stats.get(key, 0.0) + float(value)


def masked_mean_tensor(value: torch.Tensor, mask: torch.Tensor) -> float:
    if not mask.any():
        return float("nan")
    return float(value[mask].detach().float().mean().item())


def accumulate_score_metrics(
    stats: Dict[str, float],
    scores_by_control: Dict[str, Dict[str, torch.Tensor]],
    apo_labels: torch.Tensor,
    holo_labels: torch.Tensor,
    masks: Dict[str, torch.Tensor],
    eps: float,
) -> None:
    label_idx = holo_labels.unsqueeze(-1)
    apo_idx = apo_labels.unsqueeze(-1)

    for subset_name, mask in masks.items():
        add(stats, f"{subset_name}_count", mask.float().sum())

    for score_name in SCORE_NAMES:
        control_cache = {}
        for control, scores in scores_by_control[score_name].items():
            gold = scores.gather(-1, label_idx).squeeze(-1)
            apo = scores.gather(-1, apo_idx).squeeze(-1)
            other = scores.masked_fill(F.one_hot(holo_labels, num_classes=3).bool(), -1e9).amax(dim=-1)
            margin = gold - other
            pred = scores.argmax(dim=-1)
            control_cache[control] = (gold, apo, margin, pred)

            for subset_name, mask in masks.items():
                prefix = f"{score_name}_{subset_name}_{control}"
                add(stats, f"{prefix}_gold_sum", (gold * mask.float()).sum())
                add(stats, f"{prefix}_apo_delta_sum", ((gold - apo) * mask.float()).sum())
                add(stats, f"{prefix}_margin_sum", (margin * mask.float()).sum())
                add(stats, f"{prefix}_strict_win", ((margin > eps) & mask).float().sum())
                add(stats, f"{prefix}_win01", ((margin > 0.1) & mask).float().sum())
                add(stats, f"{prefix}_argmax_hit", ((pred == holo_labels) & mask).float().sum())

        correct_gold, _, correct_margin, _ = control_cache["correct"]
        for control in CONTROL_NAMES:
            decoy_gold, _, decoy_margin, _ = control_cache[control]
            margin_lift = correct_margin - decoy_margin
            gold_lift = correct_gold - decoy_gold
            for subset_name, mask in masks.items():
                prefix = f"{score_name}_{subset_name}_lift_{control}"
                add(stats, f"{prefix}_margin_sum", (margin_lift * mask.float()).sum())
                add(stats, f"{prefix}_gold_sum", (gold_lift * mask.float()).sum())
                add(stats, f"{prefix}_margin_positive", ((margin_lift > eps) & mask).float().sum())
                add(stats, f"{prefix}_gold_positive", ((gold_lift > eps) & mask).float().sum())


def accumulate_distance_metrics(
    stats: Dict[str, float],
    min_dist: torch.Tensor,
    apo_labels: torch.Tensor,
    holo_labels: torch.Tensor,
    masks: Dict[str, torch.Tensor],
    args: argparse.Namespace,
) -> None:
    holo_dist = min_dist.gather(-1, holo_labels.unsqueeze(-1)).squeeze(-1)
    apo_dist = min_dist.gather(-1, apo_labels.unsqueeze(-1)).squeeze(-1)
    finite = torch.isfinite(holo_dist) & torch.isfinite(apo_dist)
    clash_dist = float(args.clash_dist)
    contact_dist = float(args.contact_dist)

    for subset_name, mask_raw in masks.items():
        mask = mask_raw & finite
        prefix = f"dist_{subset_name}"
        signed_delta = torch.where(mask, holo_dist - apo_dist, torch.zeros_like(holo_dist))
        abs_delta = torch.where(mask, (holo_dist - apo_dist).abs(), torch.zeros_like(holo_dist))
        add(stats, f"{prefix}_finite_count", mask.float().sum())
        add(stats, f"{prefix}_holo_minus_apo_sum", signed_delta.sum())
        add(stats, f"{prefix}_abs_holo_minus_apo_sum", abs_delta.sum())
        add(stats, f"{prefix}_apo_clash", ((apo_dist < clash_dist) & mask).float().sum())
        add(stats, f"{prefix}_holo_clash", ((holo_dist < clash_dist) & mask).float().sum())
        add(stats, f"{prefix}_clash_rescue", ((apo_dist < clash_dist) & (holo_dist >= clash_dist) & mask).float().sum())
        add(stats, f"{prefix}_clash_harm", ((apo_dist >= clash_dist) & (holo_dist < clash_dist) & mask).float().sum())
        add(stats, f"{prefix}_contact_gain", ((apo_dist > contact_dist) & (holo_dist <= contact_dist) & mask).float().sum())
        add(stats, f"{prefix}_contact_loss", ((apo_dist <= contact_dist) & (holo_dist > contact_dist) & mask).float().sum())


def sample_records(
    batch,
    masks: Dict[str, torch.Tensor],
    shell_scores: Dict[str, torch.Tensor],
    apo_labels: torch.Tensor,
    holo_labels: torch.Tensor,
) -> List[Dict[str, object]]:
    records = []
    one_hot_holo = F.one_hot(holo_labels, num_classes=3).bool()
    correct = shell_scores["correct"]
    translated = shell_scores["translated"]
    correct_gold = correct.gather(-1, holo_labels.unsqueeze(-1)).squeeze(-1)
    correct_margin = correct_gold - correct.masked_fill(one_hot_holo, -1e9).amax(dim=-1)
    translated_gold = translated.gather(-1, holo_labels.unsqueeze(-1)).squeeze(-1)
    translated_margin = translated_gold - translated.masked_fill(one_hot_holo, -1e9).amax(dim=-1)

    for i, sample_id in enumerate(batch.pdb_ids):
        rec: Dict[str, object] = {
            "sample_id": sample_id,
            "n_residues": int(batch.n_residues[i]),
        }
        for subset_name, mask in masks.items():
            mi = mask[i]
            rec[f"{subset_name}_count"] = int(mi.sum().item())
            rec[f"shell35_{subset_name}_correct_margin_mean"] = masked_mean_tensor(correct_margin[i], mi)
            rec[f"shell35_{subset_name}_translated_margin_mean"] = masked_mean_tensor(translated_margin[i], mi)
            rec[f"shell35_{subset_name}_margin_lift_translated_mean"] = masked_mean_tensor(
                correct_margin[i] - translated_margin[i], mi
            )
        records.append(rec)
    return records


def finalize_stats(stats: Dict[str, float], args: argparse.Namespace, n_dataset: int, n_batches: int) -> Dict[str, object]:
    summary: Dict[str, object] = {
        "data_dir": str(Path(args.data_dir).resolve()),
        "split": args.split,
        "valid_samples_file": args.valid_samples_file,
        "dataset_items_seen_by_loader": int(n_dataset),
        "batches_seen": int(n_batches),
        "geometry_controls": {
            "nolig": "all ligand masks are false; all geometry scores are zero",
            "translated": "ligand coordinates are shifted by +100A on x/y/z",
            "shuffled": "ligand coordinates/masks are rolled across samples in the batch",
            "scrambled_types": "not evaluated here because this oracle is geometry-only",
        },
        "score_definitions": {
            "shell35": "Gaussian preference around 3.5A minus clash penalty",
            "shell45": "Gaussian preference around 4.5A minus clash penalty",
            "contact": "soft contact within contact_dist minus clash penalty",
            "closest": "inverse distance preference minus clash penalty",
            "anticlash": "only penalizes distances below clash_dist",
        },
        "thresholds": {
            "max_local_res": args.max_local_res,
            "contact_threshold_w_res": args.contact_threshold,
            "pocket_threshold_w_res": args.pocket_threshold,
            "clash_dist": args.clash_dist,
            "contact_dist": args.contact_dist,
            "shell_sigma": args.shell_sigma,
            "clash_weight": args.clash_weight,
        },
        "raw_sums": stats,
    }

    for subset_name in SUBSET_NAMES:
        count = stats.get(f"{subset_name}_count", 0.0)
        summary[f"{subset_name}_count"] = count
        denom = max(count, 1.0)
        for score_name in SCORE_NAMES:
            for control in ("correct", *CONTROL_NAMES):
                prefix = f"{score_name}_{subset_name}_{control}"
                summary[f"{prefix}_gold_mean"] = stats.get(f"{prefix}_gold_sum", 0.0) / denom
                summary[f"{prefix}_apo_delta_mean"] = stats.get(f"{prefix}_apo_delta_sum", 0.0) / denom
                summary[f"{prefix}_margin_mean"] = stats.get(f"{prefix}_margin_sum", 0.0) / denom
                summary[f"{prefix}_strict_win_rate"] = stats.get(f"{prefix}_strict_win", 0.0) / denom
                summary[f"{prefix}_win01_rate"] = stats.get(f"{prefix}_win01", 0.0) / denom
                summary[f"{prefix}_argmax_acc"] = stats.get(f"{prefix}_argmax_hit", 0.0) / denom
            for control in CONTROL_NAMES:
                prefix = f"{score_name}_{subset_name}_lift_{control}"
                summary[f"{prefix}_margin_mean"] = stats.get(f"{prefix}_margin_sum", 0.0) / denom
                summary[f"{prefix}_gold_mean"] = stats.get(f"{prefix}_gold_sum", 0.0) / denom
                summary[f"{prefix}_margin_positive_rate"] = stats.get(f"{prefix}_margin_positive", 0.0) / denom
                summary[f"{prefix}_gold_positive_rate"] = stats.get(f"{prefix}_gold_positive", 0.0) / denom

        dist_count = max(stats.get(f"dist_{subset_name}_finite_count", 0.0), 1.0)
        for key in (
            "holo_minus_apo_sum",
            "abs_holo_minus_apo_sum",
        ):
            metric = key[:-4] if key.endswith("_sum") else key
            summary[f"dist_{subset_name}_{metric}_mean"] = stats.get(f"dist_{subset_name}_{key}", 0.0) / dist_count
        for key in (
            "apo_clash",
            "holo_clash",
            "clash_rescue",
            "clash_harm",
            "contact_gain",
            "contact_loss",
        ):
            summary[f"dist_{subset_name}_{key}_rate"] = stats.get(f"dist_{subset_name}_{key}", 0.0) / dist_count

    return summary


def merge_dicts(dicts: Iterable[Dict[str, float]]) -> Dict[str, float]:
    merged: Dict[str, float] = {}
    for item in dicts:
        for key, value in item.items():
            merged[key] = merged.get(key, 0.0) + float(value)
    return merged


def flatten_records(nested: Iterable[List[Dict[str, object]]]) -> List[Dict[str, object]]:
    out: List[Dict[str, object]] = []
    for records in nested:
        out.extend(records)
    return out


@torch.no_grad()
def main() -> None:
    args = parse_args()
    distributed = bool(args.distributed)
    rank = 0
    world_size = 1
    if distributed:
        dist.init_process_group("nccl", timeout=timedelta(seconds=int(os.environ.get("DDP_TIMEOUT", "7200"))))
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")
    else:
        device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    is_main = rank == 0
    output_dir = Path(args.output_dir)
    if is_main:
        output_dir.mkdir(parents=True, exist_ok=True)
        print(f"Device: {device}")
        print(f"World size: {world_size}")
        print(f"Args: {json.dumps(vars(args), indent=2)}")

    dataset = ApoHoloTripletDataset(
        data_dir=args.data_dir,
        split=args.split,
        valid_samples_file=args.valid_samples_file,
        sample_metadata_file=args.sample_metadata_file,
        require_atom14=False,
    )
    base_index_count = len(dataset)
    indices = list(range(base_index_count))
    if args.max_samples and args.max_samples > 0 and args.max_samples < len(indices):
        gen = torch.Generator()
        gen.manual_seed(int(args.subset_seed))
        indices = torch.randperm(len(indices), generator=gen)[: int(args.max_samples)].tolist()
    if distributed:
        indices = indices[rank::world_size]
    dataset = Subset(dataset, indices)

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=partial(collate_stage1_batch, max_n_res=args.max_n_res),
        pin_memory=True,
        persistent_workers=args.num_workers > 0,
    )

    fk_module = create_openfold_fk().to(device)
    fk_module.eval()

    stats: Dict[str, float] = {}
    per_sample: List[Dict[str, object]] = []
    n_batches = 0
    iterator = tqdm(loader, desc=f"Oracle audit r{rank}", disable=not is_main)
    for batch in iterator:
        if batch is None:
            continue
        n_batches += 1
        batch = batch_to_device(batch, device)
        aatype = sequences_to_aatype(batch.sequences, batch.node_mask.shape[1], device)
        batch, aatype = select_local_batch(batch, aatype, args)

        rigids = build_rigids_from_backbone(batch.N_apo.float(), batch.Ca_apo.float(), batch.C_apo.float(), batch.node_mask)
        cand_atom14, _, chi1_dep_mask = compute_candidate_geometries(
            fk_module,
            rigids,
            aatype,
            batch.torsion_apo.float(),
        )
        chi1_dep_mask = chi1_dep_mask.bool()
        apo_labels, holo_labels, masks = subset_masks(batch, chi1_dep_mask, args)

        controls = {
            "correct": (batch.lig_points, batch.lig_mask),
            "nolig": (batch.lig_points, torch.zeros_like(batch.lig_mask)),
            "translated": make_translated_ligand(batch),
            "shuffled": make_shuffled_ligand(batch),
        }

        min_dists: Dict[str, torch.Tensor] = {}
        scores_by_score: Dict[str, Dict[str, torch.Tensor]] = {name: {} for name in SCORE_NAMES}
        for control_name, (lig_points, lig_mask) in controls.items():
            md = candidate_min_dist(
                cand_atom14,
                chi1_dep_mask,
                lig_points.float(),
                lig_mask.bool(),
                batch.node_mask.bool(),
                int(args.residue_chunk),
            )
            min_dists[control_name] = md
            score_map = score_from_min_dist(md, args)
            for score_name, score in score_map.items():
                scores_by_score[score_name][control_name] = score

        accumulate_score_metrics(stats, scores_by_score, apo_labels, holo_labels, masks, float(args.eps))
        accumulate_distance_metrics(stats, min_dists["correct"], apo_labels, holo_labels, masks, args)
        per_sample.extend(sample_records(batch, masks, scores_by_score["shell35"], apo_labels, holo_labels))

        if args.max_batches and args.max_batches > 0 and n_batches >= args.max_batches:
            break

    if distributed:
        gathered_stats: List[Dict[str, float]] = [None for _ in range(world_size)]  # type: ignore[list-item]
        gathered_records: List[List[Dict[str, object]]] = [None for _ in range(world_size)]  # type: ignore[list-item]
        gathered_batches = [None for _ in range(world_size)]
        gathered_index_counts = [None for _ in range(world_size)]
        dist.all_gather_object(gathered_stats, stats)
        dist.all_gather_object(gathered_records, per_sample)
        dist.all_gather_object(gathered_batches, n_batches)
        dist.all_gather_object(gathered_index_counts, len(indices))
        if is_main:
            stats = merge_dicts(gathered_stats)
            per_sample = flatten_records(gathered_records)
            n_batches = int(sum(int(x) for x in gathered_batches if x is not None))
            base_index_count = int(sum(int(x) for x in gathered_index_counts if x is not None))
    if is_main:
        summary = finalize_stats(stats, args, base_index_count, n_batches)
        result = {
            "summary": summary,
            "per_sample": per_sample,
        }
        json_path = output_dir / "rotamer_oracle_signal.json"
        csv_path = output_dir / "rotamer_oracle_signal_per_sample.csv"
        with json_path.open("w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        fieldnames = sorted({key for row in per_sample for key in row.keys()}) if per_sample else ["sample_id"]
        with csv_path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(per_sample)

        print("=== Rotamer oracle signal audit ===")
        focus_keys = [
            "contact_switch_count",
            "pocket_switch_count",
            "shell35_contact_switch_correct_margin_mean",
            "shell35_contact_switch_correct_strict_win_rate",
            "shell35_contact_switch_lift_translated_margin_mean",
            "shell35_contact_switch_lift_translated_margin_positive_rate",
            "shell35_contact_switch_lift_shuffled_margin_mean",
            "shell35_contact_switch_lift_shuffled_margin_positive_rate",
            "contact_contact_switch_correct_margin_mean",
            "closest_contact_switch_correct_margin_mean",
            "dist_contact_switch_holo_minus_apo_mean",
            "dist_contact_switch_clash_rescue_rate",
            "dist_contact_switch_contact_gain_rate",
        ]
        for key in focus_keys:
            print(f"{key}: {summary.get(key)}")
        print(f"JSON: {json_path}")
        print(f"CSV:  {csv_path}")

    if distributed:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
