#!/usr/bin/env python3
"""Train a direct ligand-causal chi probe.

This is an intentionally narrow Stage-1 experiment: known apo structure plus
known-pose ligand should predict holo chi angles better than no-ligand or decoy
ligand controls, especially on ligand-facing residues.
"""

import argparse
import json
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

from src.stage1.datasets import ApoHoloTripletDataset, collate_stage1_batch
from src.stage1.models.adapter import ESMAdapter
from src.stage1.models.ligand_condition import LigandConditioner, LigandConditionerConfig
from src.stage1.modules.losses import torsion_sincos_loss


class LigandChiProbe(nn.Module):
    """Ligand-conditioned residual chi predictor over apo chi."""

    def __init__(
        self,
        esm_dim: int = 1280,
        c_s: int = 384,
        d_lig: int = 64,
        num_heads: int = 8,
        hidden: int = 512,
        dropout: float = 0.1,
        warmup_steps: int = 80,
        residual_scale: float = 0.5,
    ):
        super().__init__()
        self.residual_scale = float(residual_scale)
        self.esm_adapter = ESMAdapter(esm_dim=esm_dim, output_dim=c_s, dropout=dropout)
        self.ligand_conditioner = LigandConditioner(
            LigandConditionerConfig(
                c_s=c_s,
                d_lig=d_lig,
                num_heads=num_heads,
                dropout=dropout,
                warmup_steps=warmup_steps,
            )
        )
        self.head = nn.Sequential(
            nn.LayerNorm(c_s + 8 + 1),
            nn.Linear(c_s + 8 + 1, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 8),
        )

    def load_stage1_modules(
        self,
        checkpoint: Dict[str, torch.Tensor],
        prefixes: Iterable[str] = ("esm_adapter", "ligand_conditioner"),
    ) -> Dict[str, int]:
        state = checkpoint.get("model_state_dict", checkpoint)
        own_state = self.state_dict()
        wanted = tuple(f"{p}." for p in prefixes)
        loadable = {}
        skipped = 0
        for key, value in state.items():
            clean_key = key[len("module.") :] if key.startswith("module.") else key
            if not clean_key.startswith(wanted):
                continue
            if clean_key in own_state and tuple(own_state[clean_key].shape) == tuple(value.shape):
                loadable[clean_key] = value
            else:
                skipped += 1
        self.load_state_dict(loadable, strict=False)
        return {"loaded": len(loadable), "skipped": skipped}

    @staticmethod
    def apo_chi_sincos(batch) -> torch.Tensor:
        apo_chi = batch.torsion_apo[:, :, 3:7]
        if apo_chi.shape[-1] < 4:
            pad = torch.zeros(*apo_chi.shape[:2], 4 - apo_chi.shape[-1], device=apo_chi.device)
            apo_chi = torch.cat([apo_chi, pad], dim=-1)
        return torch.stack([torch.sin(apo_chi), torch.cos(apo_chi)], dim=-1)

    def forward(
        self,
        batch,
        current_step: int = 0,
        gate_lambda: Optional[float] = None,
    ) -> torch.Tensor:
        s = self.esm_adapter(batch.esm)
        kwargs = {"gate_lambda": gate_lambda} if gate_lambda is not None else {"current_step": current_step}
        s_lig = self.ligand_conditioner(
            s,
            batch.lig_points,
            batch.lig_types,
            batch.node_mask.bool(),
            batch.lig_mask.bool(),
            **kwargs,
        )
        apo = self.apo_chi_sincos(batch)
        features = torch.cat([s_lig, apo.reshape(*apo.shape[:2], 8), batch.w_res.unsqueeze(-1)], dim=-1)
        delta = self.head(features).view(*apo.shape)
        pred = apo + self.residual_scale * delta
        return F.normalize(pred, p=2, dim=-1, eps=1e-8)


def parse_args():
    parser = argparse.ArgumentParser(description="Train ligand-causal chi residual probe")
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--train_samples_file", default=None)
    parser.add_argument("--val_samples_file", default=None)
    parser.add_argument("--sample_metadata_file", default=None)
    parser.add_argument("--stage1_checkpoint", default=None)
    parser.add_argument("--batch_size", type=int, default=100)
    parser.add_argument("--train_max_samples", type=int, default=12000)
    parser.add_argument("--val_max_samples", type=int, default=1000)
    parser.add_argument("--subset_seed", type=int, default=20260617)
    parser.add_argument("--max_n_res", type=int, default=1600)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--max_epochs", type=int, default=12)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--warmup_steps", type=int, default=80)
    parser.add_argument("--contact_threshold", type=float, default=0.5)
    parser.add_argument("--lambda_contact", type=float, default=1.0)
    parser.add_argument("--lambda_rank", type=float, default=0.2)
    parser.add_argument("--rank_margin", type=float, default=0.02)
    parser.add_argument("--residual_scale", type=float, default=0.5)
    parser.add_argument("--save_dir", default="checkpoints/stage1/ligand_chi_probe")
    parser.add_argument("--log_dir", default="logs/stage1/ligand_chi_probe")
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
    ):
        setattr(batch, name, getattr(batch, name).to(device))
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


def make_no_ligand_batch(batch):
    return replace_batch(
        batch,
        lig_points=torch.zeros_like(batch.lig_points),
        lig_types=torch.zeros_like(batch.lig_types),
        lig_mask=torch.zeros_like(batch.lig_mask),
    )


def make_translated_batch(batch, offset: float = 100.0):
    offset_vec = torch.tensor([offset, offset, offset], device=batch.lig_points.device, dtype=batch.lig_points.dtype)
    return replace_batch(batch, lig_points=batch.lig_points + offset_vec.view(1, 1, 3))


def make_batch_shuffled_ligand(batch):
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


def contact_weights(batch, threshold: float):
    contact = ((batch.w_res > threshold) & batch.node_mask.bool()).float()
    if contact.sum() < 1:
        contact = batch.node_mask.float()
    return contact


def chi1_accuracy(pred, target, chi_mask, residue_weights=None, threshold_deg: float = 20.0):
    pred_angle = torch.atan2(pred[:, :, 0, 0], pred[:, :, 0, 1])
    true_angle = target[:, :, 0]
    diff = torch.abs(torch.atan2(torch.sin(pred_angle - true_angle), torch.cos(pred_angle - true_angle)))
    mask = chi_mask[:, :, 0].float()
    if residue_weights is not None:
        mask = mask * residue_weights.float()
    denom = mask.sum().clamp_min(1e-8)
    return (((torch.rad2deg(diff) < threshold_deg).float() * mask).sum() / denom).item()


def loss_and_metrics(pred, batch, contact_w):
    global_loss = torsion_sincos_loss(pred, batch.chi_holo, batch.chi_mask)
    contact_loss = torsion_sincos_loss(pred, batch.chi_holo, batch.chi_mask, w_res=contact_w)
    return {
        "global_loss": global_loss,
        "contact_loss": contact_loss,
        "chi1_acc": chi1_accuracy(pred, batch.chi_holo, batch.chi_mask),
        "contact_chi1_acc": chi1_accuracy(pred, batch.chi_holo, batch.chi_mask, contact_w),
    }


def reduce_scalar(value: float, device, distributed: bool):
    tensor = torch.tensor([float(value), 1.0], device=device)
    if distributed:
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    return (tensor[0] / tensor[1].clamp_min(1.0)).item()


def average_records(records, device, distributed: bool):
    if not records:
        return {}
    keys = records[0].keys()
    local = torch.tensor([sum(float(r[k]) for r in records) for k in keys] + [len(records)], device=device)
    if distributed:
        dist.all_reduce(local, op=dist.ReduceOp.SUM)
    denom = local[-1].clamp_min(1.0)
    return {k: (local[i] / denom).item() for i, k in enumerate(keys)}


def train_epoch(model, loader, optimizer, device, args, start_step: int, distributed: bool):
    model.train()
    raw_model = model.module if isinstance(model, DDP) else model
    records = []
    global_step = int(start_step)
    iterator = tqdm(loader, desc="Training", disable=dist.is_initialized() and dist.get_rank() != 0)
    for batch in iterator:
        if batch is None:
            continue
        batch = batch_to_device(batch, device)
        contact_w = contact_weights(batch, args.contact_threshold)

        pred_correct = model(batch, current_step=global_step)
        correct = loss_and_metrics(pred_correct, batch, contact_w)

        pred_no = model(make_no_ligand_batch(batch), current_step=global_step)
        pred_shuffle = model(make_batch_shuffled_ligand(batch), current_step=global_step)
        pred_trans = model(make_translated_batch(batch), current_step=global_step)
        no_contact = loss_and_metrics(pred_no, batch, contact_w)["contact_loss"]
        shuffle_contact = loss_and_metrics(pred_shuffle, batch, contact_w)["contact_loss"]
        trans_contact = loss_and_metrics(pred_trans, batch, contact_w)["contact_loss"]

        rank_loss = (
            F.relu(args.rank_margin + correct["contact_loss"] - no_contact)
            + F.relu(args.rank_margin + correct["contact_loss"] - shuffle_contact)
            + F.relu(args.rank_margin + correct["contact_loss"] - trans_contact)
        ) / 3.0
        loss = correct["global_loss"] + args.lambda_contact * correct["contact_loss"] + args.lambda_rank * rank_loss

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if args.grad_clip and args.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(raw_model.parameters(), args.grad_clip)
        optimizer.step()
        global_step += 1

        record = {
            "loss": loss.item(),
            "global_loss": correct["global_loss"].item(),
            "contact_loss": correct["contact_loss"].item(),
            "rank_loss": rank_loss.item(),
            "chi1_acc": correct["chi1_acc"],
            "contact_chi1_acc": correct["contact_chi1_acc"],
            "lift_no_ligand": (no_contact - correct["contact_loss"]).item(),
            "lift_shuffled": (shuffle_contact - correct["contact_loss"]).item(),
            "lift_translated": (trans_contact - correct["contact_loss"]).item(),
        }
        records.append(record)
        iterator.set_postfix({"loss": f"{record['loss']:.4f}", "lift": f"{record['lift_shuffled']:.4f}"})
    return average_records(records, device, distributed), global_step


@torch.no_grad()
def validate(model, loader, device, args, distributed: bool):
    model.eval()
    records = []
    iterator = tqdm(loader, desc="Validation", disable=dist.is_initialized() and dist.get_rank() != 0)
    for batch in iterator:
        if batch is None:
            continue
        batch = batch_to_device(batch, device)
        contact_w = contact_weights(batch, args.contact_threshold)

        pred_correct = model(batch, gate_lambda=1.0)
        pred_no = model(make_no_ligand_batch(batch), gate_lambda=1.0)
        pred_shuffle = model(make_batch_shuffled_ligand(batch), gate_lambda=1.0)
        pred_trans = model(make_translated_batch(batch), gate_lambda=1.0)

        correct = loss_and_metrics(pred_correct, batch, contact_w)
        no_metrics = loss_and_metrics(pred_no, batch, contact_w)
        shuf_metrics = loss_and_metrics(pred_shuffle, batch, contact_w)
        trans_metrics = loss_and_metrics(pred_trans, batch, contact_w)

        records.append(
            {
                "val_global_loss": correct["global_loss"].item(),
                "val_contact_loss": correct["contact_loss"].item(),
                "val_chi1_acc": correct["chi1_acc"],
                "val_contact_chi1_acc": correct["contact_chi1_acc"],
                "val_no_ligand_contact_loss": no_metrics["contact_loss"].item(),
                "val_shuffled_contact_loss": shuf_metrics["contact_loss"].item(),
                "val_translated_contact_loss": trans_metrics["contact_loss"].item(),
                "val_lift_no_ligand": (no_metrics["contact_loss"] - correct["contact_loss"]).item(),
                "val_lift_shuffled": (shuf_metrics["contact_loss"] - correct["contact_loss"]).item(),
                "val_lift_translated": (trans_metrics["contact_loss"] - correct["contact_loss"]).item(),
                "val_no_ligand_sensitivity": torch.norm(pred_correct - pred_no, dim=-1).mean().item(),
                "val_shuffled_sensitivity": torch.norm(pred_correct - pred_shuffle, dim=-1).mean().item(),
                "val_translated_sensitivity": torch.norm(pred_correct - pred_trans, dim=-1).mean().item(),
            }
        )
    return average_records(records, device, distributed)


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

    model = LigandChiProbe(warmup_steps=args.warmup_steps, residual_scale=args.residual_scale).to(device)
    if args.stage1_checkpoint:
        ckpt = torch.load(args.stage1_checkpoint, map_location=device)
        report = model.load_stage1_modules(ckpt)
        if is_main:
            print(f"Loaded Stage-1 modules: {report}")

    if distributed:
        model = DDP(model, device_ids=[local_rank], find_unused_parameters=False)

    train_dataset = ApoHoloTripletDataset(
        data_dir=args.data_dir,
        split="train",
        valid_samples_file=args.train_samples_file,
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
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        sampler=val_sampler,
        num_workers=args.num_workers,
        collate_fn=partial(collate_stage1_batch, max_n_res=args.max_n_res),
    )
    if is_main:
        print(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    best_metric = -float("inf")
    patience = 0
    global_step = 0

    for epoch in range(args.max_epochs):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        if is_main:
            print(f"\nEpoch {epoch + 1}/{args.max_epochs}")

        train_metrics, global_step = train_epoch(model, train_loader, optimizer, device, args, global_step, distributed)
        val_metrics = validate(model, val_loader, device, args, distributed)
        record = {"epoch": epoch, "global_step": global_step, **train_metrics, **val_metrics}
        selection_metric = (
            record["val_lift_shuffled"]
            + record["val_lift_translated"]
            + record["val_lift_no_ligand"]
        ) / 3.0
        record["selection_ligand_lift"] = selection_metric

        if is_main:
            print(json.dumps(record, indent=2))
            with open(log_dir / "metrics.jsonl", "a") as f:
                f.write(json.dumps(record) + "\n")

            raw_model = model.module if isinstance(model, DDP) else model
            if selection_metric > best_metric:
                best_metric = selection_metric
                patience = 0
                torch.save(
                    {
                        "epoch": epoch,
                        "global_step": global_step,
                        "model_state_dict": raw_model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "selection_ligand_lift": selection_metric,
                        "args": vars(args),
                    },
                    save_dir / "best_model.pt",
                )
                print(f"  ✓ New best ligand lift: {selection_metric:.6f}")
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
                "global_step": global_step,
                "model_state_dict": raw_model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "selection_ligand_lift": best_metric,
                "args": vars(args),
            },
            save_dir / "latest_model.pt",
        )
        print(f"Training complete. Best ligand lift: {best_metric:.6f}")

    if distributed:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
