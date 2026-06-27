#!/usr/bin/env python3
"""Audit whether holo sidechains are geometrically better for the ligand than apo.

This is a data-label sanity check, not a model.  It reconstructs apo and holo
atom14 sidechains from stored torsions on the apo backbone and compares their
minimum sidechain-ligand distances on the same local subsets used by the
Stage-1 ligand-causality experiments.

If holo labels do not reduce clash or improve contact relative to apo on
contact/switch residues, then chi1 endpoint supervision is a weak causal target
for known-pose ligand conditioning.
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
from typing import Dict, Iterable, List, Tuple

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))
flash_ipa_path = project_root / "vendor" / "flash_ipa" / "src"
if flash_ipa_path.exists():
    sys.path.insert(0, str(flash_ipa_path))

from flash_ipa.rigid import Rigid, Rotation

from src.stage1.datasets import ApoHoloTripletDataset, collate_stage1_batch
from src.stage1.models.fk_openfold import create_openfold_fk, reorder_torsions_to_openfold
from src.stage1.models.torsion_head import compute_candidate_geometries
from src.stage1.modules.losses import chi1_rotamer_labels


AA_RESTYPE_MAP = {
    "A": 0, "R": 1, "N": 2, "D": 3, "C": 4,
    "Q": 5, "E": 6, "G": 7, "H": 8, "I": 9,
    "L": 10, "K": 11, "M": 12, "F": 13, "P": 14,
    "S": 15, "T": 16, "W": 17, "Y": 18, "V": 19,
}

SUBSET_NAMES = ("all", "contact", "switch", "contact_switch", "pocket_switch")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit apo-vs-holo ligand geometry in Stage-1 data")
    parser.add_argument("--data_dir", default="processed_data/triplets")
    parser.add_argument("--split", default="val", choices=["train", "val", "test"])
    parser.add_argument("--valid_samples_file", default=None)
    parser.add_argument("--sample_metadata_file", default="sample_metadata.json")
    parser.add_argument("--batch_size", type=int, default=24)
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
    parser.add_argument("--output_dir", default="logs/stage1_diagnostics/apo_holo_ligand_geometry")
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


def torsions_to_atom14(fk_module, rigids: Rigid, aatype: torch.Tensor, torsions: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    torsion_sc = torch.stack([torch.sin(torsions), torch.cos(torsions)], dim=-1)
    torsion_sc = reorder_torsions_to_openfold(torsion_sc)
    result = fk_module(torsion_sc.float(), rigids, aatype)
    return result["atom14_pos"], result["atom14_mask"].bool()


def min_sidechain_ligand_dist(
    atom14: torch.Tensor,
    sidechain_mask: torch.Tensor,
    lig_points: torch.Tensor,
    lig_mask: torch.Tensor,
    node_mask: torch.Tensor,
    residue_chunk: int,
) -> torch.Tensor:
    bsz, n_res, n_atom, _ = atom14.shape
    n_lig = lig_points.shape[1]
    out = atom14.new_full((bsz, n_res), float("inf"))
    if n_lig == 0:
        return out
    chunk = max(int(residue_chunk), 1)
    for start in range(0, n_res, chunk):
        end = min(start + chunk, n_res)
        c = end - start
        dists = torch.cdist(
            atom14[:, start:end].reshape(bsz * c, n_atom, 3),
            lig_points[:, None, :, :].expand(bsz, c, n_lig, 3).reshape(bsz * c, n_lig, 3),
        ).reshape(bsz, c, n_atom, n_lig)
        valid = (
            sidechain_mask[:, start:end, :, None]
            & lig_mask[:, None, None, :].bool()
            & node_mask[:, start:end, None, None].bool()
        )
        dists = dists.masked_fill(~valid.expand_as(dists), float("inf"))
        out[:, start:end] = dists.amin(dim=(-1, -2))
    return out


def subset_masks(batch, sidechain_mask: torch.Tensor, args: argparse.Namespace):
    apo_labels, apo_valid = chi1_rotamer_labels(batch.torsion_apo[:, :, 3], batch.chi_mask[:, :, 0])
    holo_labels, holo_valid = chi1_rotamer_labels(batch.chi_holo[:, :, 0], batch.chi_mask[:, :, 0])
    valid = apo_valid & holo_valid & batch.node_mask.bool() & sidechain_mask.any(dim=-1)
    contact = valid & (batch.w_res > float(args.contact_threshold))
    pocket = valid & (batch.w_res > float(args.pocket_threshold))
    switch = valid & (apo_labels != holo_labels)
    return {
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


def accumulate(stats: Dict[str, float], apo_dist: torch.Tensor, holo_dist: torch.Tensor, masks: Dict[str, torch.Tensor], args) -> None:
    finite = torch.isfinite(apo_dist) & torch.isfinite(holo_dist)
    delta = holo_dist - apo_dist
    clash = float(args.clash_dist)
    contact = float(args.contact_dist)
    for subset_name, mask_raw in masks.items():
        mask = mask_raw & finite
        prefix = f"{subset_name}"
        safe_delta = torch.where(mask, delta, torch.zeros_like(delta))
        add(stats, f"{prefix}_count", mask.float().sum())
        add(stats, f"{prefix}_holo_minus_apo_sum", safe_delta.sum())
        add(stats, f"{prefix}_abs_holo_minus_apo_sum", safe_delta.abs().sum())
        add(stats, f"{prefix}_holo_closer", ((delta < -1e-6) & mask).float().sum())
        add(stats, f"{prefix}_apo_closer", ((delta > 1e-6) & mask).float().sum())
        add(stats, f"{prefix}_apo_clash", ((apo_dist < clash) & mask).float().sum())
        add(stats, f"{prefix}_holo_clash", ((holo_dist < clash) & mask).float().sum())
        add(stats, f"{prefix}_clash_rescue", ((apo_dist < clash) & (holo_dist >= clash) & mask).float().sum())
        add(stats, f"{prefix}_clash_harm", ((apo_dist >= clash) & (holo_dist < clash) & mask).float().sum())
        add(stats, f"{prefix}_apo_contact", ((apo_dist <= contact) & mask).float().sum())
        add(stats, f"{prefix}_holo_contact", ((holo_dist <= contact) & mask).float().sum())
        add(stats, f"{prefix}_contact_gain", ((apo_dist > contact) & (holo_dist <= contact) & mask).float().sum())
        add(stats, f"{prefix}_contact_loss", ((apo_dist <= contact) & (holo_dist > contact) & mask).float().sum())


def sample_records(batch, apo_dist: torch.Tensor, holo_dist: torch.Tensor, masks: Dict[str, torch.Tensor]) -> List[Dict[str, object]]:
    records = []
    delta = holo_dist - apo_dist
    for i, sample_id in enumerate(batch.pdb_ids):
        rec: Dict[str, object] = {"sample_id": sample_id, "n_residues": int(batch.n_residues[i])}
        for subset_name, mask in masks.items():
            mi = mask[i] & torch.isfinite(delta[i])
            rec[f"{subset_name}_count"] = int(mi.sum().item())
            if mi.any():
                rec[f"{subset_name}_holo_minus_apo_mean"] = float(delta[i][mi].float().mean().item())
                rec[f"{subset_name}_holo_closer_rate"] = float((delta[i][mi] < -1e-6).float().mean().item())
            else:
                rec[f"{subset_name}_holo_minus_apo_mean"] = float("nan")
                rec[f"{subset_name}_holo_closer_rate"] = float("nan")
        records.append(rec)
    return records


def finalize(stats: Dict[str, float], args: argparse.Namespace, n_dataset: int, n_batches: int) -> Dict[str, object]:
    out: Dict[str, object] = {
        "data_dir": str(Path(args.data_dir).resolve()),
        "split": args.split,
        "valid_samples_file": args.valid_samples_file,
        "dataset_items_seen_by_loader": n_dataset,
        "batches_seen": n_batches,
        "thresholds": {
            "max_local_res": args.max_local_res,
            "contact_threshold_w_res": args.contact_threshold,
            "pocket_threshold_w_res": args.pocket_threshold,
            "clash_dist": args.clash_dist,
            "contact_dist": args.contact_dist,
        },
        "raw_sums": stats,
    }
    for subset_name in SUBSET_NAMES:
        count = max(stats.get(f"{subset_name}_count", 0.0), 1.0)
        out[f"{subset_name}_count"] = stats.get(f"{subset_name}_count", 0.0)
        for key in ("holo_minus_apo", "abs_holo_minus_apo"):
            out[f"{subset_name}_{key}_mean"] = stats.get(f"{subset_name}_{key}_sum", 0.0) / count
        for key in (
            "holo_closer",
            "apo_closer",
            "apo_clash",
            "holo_clash",
            "clash_rescue",
            "clash_harm",
            "apo_contact",
            "holo_contact",
            "contact_gain",
            "contact_loss",
        ):
            out[f"{subset_name}_{key}_rate"] = stats.get(f"{subset_name}_{key}", 0.0) / count
    return out


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
    iterator = tqdm(loader, desc=f"Apo/holo geom r{rank}", disable=not is_main)
    for batch in iterator:
        if batch is None:
            continue
        n_batches += 1
        batch = batch_to_device(batch, device)
        aatype = sequences_to_aatype(batch.sequences, batch.node_mask.shape[1], device)
        batch, aatype = select_local_batch(batch, aatype, args)

        rigids = build_rigids_from_backbone(batch.N_apo.float(), batch.Ca_apo.float(), batch.C_apo.float(), batch.node_mask)
        apo_atom14, atom14_mask = torsions_to_atom14(fk_module, rigids, aatype, batch.torsion_apo.float())
        holo_atom14, _ = torsions_to_atom14(fk_module, rigids, aatype, batch.torsion_holo.float())
        _, _, chi1_dep_mask = compute_candidate_geometries(fk_module, rigids, aatype, batch.torsion_apo.float())
        sidechain_mask = atom14_mask & chi1_dep_mask.bool() & batch.node_mask.bool().unsqueeze(-1)

        apo_dist = min_sidechain_ligand_dist(
            apo_atom14, sidechain_mask, batch.lig_points.float(), batch.lig_mask.bool(), batch.node_mask.bool(), args.residue_chunk
        )
        holo_dist = min_sidechain_ligand_dist(
            holo_atom14, sidechain_mask, batch.lig_points.float(), batch.lig_mask.bool(), batch.node_mask.bool(), args.residue_chunk
        )
        masks = subset_masks(batch, sidechain_mask, args)
        accumulate(stats, apo_dist, holo_dist, masks, args)
        per_sample.extend(sample_records(batch, apo_dist, holo_dist, masks))

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
        summary = finalize(stats, args, base_index_count, n_batches)
        json_path = output_dir / "apo_holo_ligand_geometry.json"
        csv_path = output_dir / "apo_holo_ligand_geometry_per_sample.csv"
        with json_path.open("w", encoding="utf-8") as f:
            json.dump({"summary": summary, "per_sample": per_sample}, f, indent=2, ensure_ascii=False)
        fieldnames = sorted({key for row in per_sample for key in row.keys()}) if per_sample else ["sample_id"]
        with csv_path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(per_sample)
        print("=== Apo-vs-holo ligand geometry audit ===")
        for key in [
            "contact_switch_count",
            "contact_switch_holo_minus_apo_mean",
            "contact_switch_holo_closer_rate",
            "contact_switch_clash_rescue_rate",
            "contact_switch_clash_harm_rate",
            "contact_switch_contact_gain_rate",
            "contact_switch_contact_loss_rate",
            "pocket_switch_holo_minus_apo_mean",
            "pocket_switch_holo_closer_rate",
        ]:
            print(f"{key}: {summary.get(key)}")
        print(f"JSON: {json_path}")
        print(f"CSV:  {csv_path}")

    if distributed:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
