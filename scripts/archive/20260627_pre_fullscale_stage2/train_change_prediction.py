#!/usr/bin/env python3
"""
Training script for Change-Prediction RAE.

This script trains a model to predict latent space changes (delta_z = z_holo - z_apo)
from ligand information, avoiding chi1 dominance by operating in latent space.

Usage:
    python scripts/train_change_prediction.py --data_dir processed_data/triplets --max_epochs 50
"""

import os
import sys
import json
import argparse
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, Any

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data import Subset
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

try:
    import torch.multiprocessing as mp
    mp.set_sharing_strategy('file_system')
except Exception:
    pass

from src.stage1.models.change_prediction_rae import (
    ChangePredictionRAE,
    ChangePredictionRAEConfig,
)
from src.stage1.modules.losses import (
    latent_change_prediction_loss,
    torsion_sincos_loss,
)
from src.stage1.datasets import (
    create_stage1_dataloader,
    ApoHoloTripletDataset,
    collate_stage1_batch,
)
from utils.metrics import compute_chi12_accuracy
from functools import partial


def batch_to_device(batch, device):
    batch.esm = batch.esm.to(device)
    batch.N_apo = batch.N_apo.to(device)
    batch.Ca_apo = batch.Ca_apo.to(device)
    batch.C_apo = batch.C_apo.to(device)
    batch.N_holo = batch.N_holo.to(device)
    batch.Ca_holo = batch.Ca_holo.to(device)
    batch.C_holo = batch.C_holo.to(device)
    batch.node_mask = batch.node_mask.to(device)
    batch.lig_points = batch.lig_points.to(device)
    batch.lig_types = batch.lig_types.to(device)
    batch.lig_mask = batch.lig_mask.to(device)
    batch.chi_holo = batch.chi_holo.to(device)
    batch.chi_mask = batch.chi_mask.to(device)
    batch.torsion_apo = batch.torsion_apo.to(device)
    batch.torsion_holo = batch.torsion_holo.to(device)
    batch.w_res = batch.w_res.to(device)
    return batch


def parse_args():
    parser = argparse.ArgumentParser(description='Train Change-Prediction RAE')
    
    parser.add_argument('--data_dir', type=str, required=True)
    parser.add_argument('--val_samples_file', type=str, default=None)
    parser.add_argument('--sample_metadata_file', type=str, default=None)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--max_n_res', type=int, default=1600)
    parser.add_argument('--train_max_samples', type=int, default=None,
                        help='Optional deterministic random subset size for fast train-lane experiments')
    parser.add_argument('--val_max_samples', type=int, default=None,
                        help='Optional deterministic random subset size for fast validation-lane experiments')
    parser.add_argument('--subset_seed', type=int, default=20260617,
                        help='Seed used when selecting train/val subsets')
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--max_epochs', type=int, default=50)
    parser.add_argument('--grad_clip', type=float, default=1.0)
    parser.add_argument('--lambda_latent', type=float, default=1.0)
    parser.add_argument('--lambda_recon', type=float, default=0.1)
    parser.add_argument('--stage1_encoder_checkpoint', type=str, default=None,
                        help='Optional Stage-1 checkpoint whose compatible encoder/conditioner modules seed the RAE')
    parser.add_argument('--ligand_warmup_steps', type=int, default=2000,
                        help='Ligand conditioner gate warmup in optimizer steps')
    parser.add_argument('--eval_gate_step', type=int, default=100000,
                        help='current_step used during validation so ligand gate is fully open')
    parser.add_argument('--patience', type=int, default=10)
    parser.add_argument('--save_dir', type=str, default='checkpoints/stage1/change_prediction')
    parser.add_argument('--log_dir', type=str, default='logs/stage1/change_prediction')
    parser.add_argument('--num_workers', type=int, default=2)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--distributed', action='store_true',
                        help='Enable DDP training')
    
    return parser.parse_args()


def make_deterministic_subset(dataset, max_samples, seed: int, name: str, is_main_process: bool):
    if max_samples is None or max_samples <= 0 or max_samples >= len(dataset):
        return dataset

    generator = torch.Generator()
    generator.manual_seed(int(seed))
    indices = torch.randperm(len(dataset), generator=generator)[:int(max_samples)].tolist()

    if is_main_process:
        print(f"{name} subset: {len(indices)} / {len(dataset)} samples (seed={seed})")

    return Subset(dataset, indices)


def compute_chi1_accuracy(pred_chi, chi_holo, chi_mask, threshold_deg=20.0):
    pred_chi1 = pred_chi[:, :, 0, :]
    true_chi1 = chi_holo[:, :, 0]
    mask = chi_mask[:, :, 0]
    
    pred_sin = pred_chi1[:, :, 0]
    pred_cos = pred_chi1[:, :, 1]
    pred_angle = torch.atan2(pred_sin, pred_cos)
    
    angle_diff = torch.abs(pred_angle - true_chi1)
    angle_diff = torch.minimum(angle_diff, 2 * torch.pi - angle_diff)
    angle_diff_deg = torch.rad2deg(angle_diff)
    
    correct = (angle_diff_deg < threshold_deg) & mask.bool()
    accuracy = correct.float().sum() / (mask.float().sum() + 1e-8)
    
    return accuracy.item()


def train_epoch(
    model,
    dataloader,
    optimizer,
    device,
    lambda_latent,
    lambda_recon,
    grad_clip: float,
    start_step: int = 0,
):
    model.train()
    total_loss = 0.0
    total_latent_loss = 0.0
    total_recon_loss = 0.0
    total_chi_acc = 0.0
    n_batches = 0
    
    pbar = tqdm(dataloader, desc='Training')
    global_step = int(start_step)
    for batch in pbar:
        if batch is None:
            continue
        batch = batch_to_device(batch, device)
        
        optimizer.zero_grad()
        
        outputs = model(batch, current_step=global_step)
        
        latent_loss = latent_change_prediction_loss(
            outputs['delta_z_pred'],
            outputs['delta_z_true'],
            batch.node_mask,
        )
        
        recon_loss = torsion_sincos_loss(
            outputs['pred_chi'],
            batch.chi_holo,
            batch.chi_mask,
        )
        
        loss = lambda_latent * latent_loss + lambda_recon * recon_loss
        
        loss.backward()
        if grad_clip and grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        
        chi_acc = compute_chi1_accuracy(
            outputs['pred_chi'],
            batch.chi_holo,
            batch.chi_mask,
        )
        
        total_loss += loss.item()
        total_latent_loss += latent_loss.item()
        total_recon_loss += recon_loss.item()
        total_chi_acc += chi_acc
        n_batches += 1
        global_step += 1
        
        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'latent': f'{latent_loss.item():.4f}',
            'chi_acc': f'{chi_acc:.4f}',
            'step': global_step,
        })
    
    return {
        'loss': total_loss / n_batches,
        'latent_loss': total_latent_loss / n_batches,
        'recon_loss': total_recon_loss / n_batches,
        'chi1_acc': total_chi_acc / n_batches,
    }, global_step


@torch.no_grad()
def validate(model, dataloader, device, lambda_latent, lambda_recon, current_step: int):
    model.eval()
    total_loss = 0.0
    total_latent_loss = 0.0
    total_recon_loss = 0.0
    total_chi_acc = 0.0
    n_batches = 0
    
    for batch in tqdm(dataloader, desc='Validation'):
        if batch is None:
            continue
        batch = batch_to_device(batch, device)
        
        outputs = model(batch, current_step=current_step)
        
        latent_loss = latent_change_prediction_loss(
            outputs['delta_z_pred'],
            outputs['delta_z_true'],
            batch.node_mask,
        )
        
        recon_loss = torsion_sincos_loss(
            outputs['pred_chi'],
            batch.chi_holo,
            batch.chi_mask,
        )
        
        loss = lambda_latent * latent_loss + lambda_recon * recon_loss
        
        chi_acc = compute_chi1_accuracy(
            outputs['pred_chi'],
            batch.chi_holo,
            batch.chi_mask,
        )
        
        total_loss += loss.item()
        total_latent_loss += latent_loss.item()
        total_recon_loss += recon_loss.item()
        total_chi_acc += chi_acc
        n_batches += 1
    
    return {
        'loss': total_loss / n_batches,
        'latent_loss': total_latent_loss / n_batches,
        'recon_loss': total_recon_loss / n_batches,
        'chi1_acc': total_chi_acc / n_batches,
    }


def main():
    args = parse_args()
    
    distributed = args.distributed
    local_rank = 0
    world_size = 1
    is_main_process = True
    
    if distributed:
        if not dist.is_initialized():
            timeout_sec = int(os.environ.get("DDP_TIMEOUT", "7200"))
            dist.init_process_group(
                backend='nccl',
                timeout=timedelta(seconds=timeout_sec),
            )
        local_rank = int(os.environ.get('LOCAL_RANK', 0))
        world_size = dist.get_world_size()
        is_main_process = (local_rank == 0)
        torch.cuda.set_device(local_rank)
        device = torch.device(f'cuda:{local_rank}')
        if is_main_process:
            print(f"[DDP] Initialized: world_size={world_size}, local_rank={local_rank}")
    else:
        device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    
    if is_main_process:
        print(f"Using device: {device}")
    
    save_dir = Path(args.save_dir)
    log_dir = Path(args.log_dir)
    if is_main_process:
        save_dir.mkdir(parents=True, exist_ok=True)
        log_dir.mkdir(parents=True, exist_ok=True)
    
    if is_main_process:
        print("Creating model...")
    config = ChangePredictionRAEConfig(
        lambda_latent=args.lambda_latent,
        lambda_recon=args.lambda_recon,
        warmup_steps=args.ligand_warmup_steps,
    )
    model = ChangePredictionRAE(config).to(device)

    if args.stage1_encoder_checkpoint:
        if is_main_process:
            print(f"Loading compatible Stage-1 modules from {args.stage1_encoder_checkpoint}...")
        ckpt = torch.load(args.stage1_encoder_checkpoint, map_location=device)
        load_report = model.load_stage1_modules(ckpt)
        if is_main_process:
            print(f"  loaded tensors: {len(load_report['loaded'])}")
            if load_report['skipped']:
                print(f"  skipped tensors: {len(load_report['skipped'])}")
                for key, reason in load_report['skipped'][:20]:
                    print(f"    - {key}: {reason}")
                if len(load_report['skipped']) > 20:
                    print(f"    ... {len(load_report['skipped']) - 20} more")
    
    if distributed:
        model = DDP(model, device_ids=[local_rank], find_unused_parameters=False)
    
    if is_main_process:
        print("Creating dataloaders...")
    train_dataset = ApoHoloTripletDataset(
        data_dir=args.data_dir,
        split='train',
        sample_metadata_file=args.sample_metadata_file,
        require_atom14=False,
    )
    train_dataset = make_deterministic_subset(
        train_dataset,
        args.train_max_samples,
        args.subset_seed,
        'Train',
        is_main_process,
    )
    
    train_sampler = DistributedSampler(
        train_dataset,
        num_replicas=world_size,
        rank=local_rank,
        shuffle=True,
    ) if distributed else None
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        num_workers=args.num_workers,
        collate_fn=partial(collate_stage1_batch, max_n_res=args.max_n_res),
    )
    
    val_dataset = ApoHoloTripletDataset(
        data_dir=args.data_dir,
        split='val',
        valid_samples_file=args.val_samples_file,
        sample_metadata_file=args.sample_metadata_file,
        require_atom14=False,
    )
    val_dataset = make_deterministic_subset(
        val_dataset,
        args.val_max_samples,
        args.subset_seed + 1,
        'Val',
        is_main_process,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=partial(collate_stage1_batch, max_n_res=args.max_n_res),
    )
    
    if is_main_process:
        print(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")
    
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr,
    )
    
    best_val_loss = float('inf')
    patience_counter = 0
    metrics_log = []
    global_step = 0
    
    for epoch in range(args.max_epochs):
        if distributed:
            train_sampler.set_epoch(epoch)
        
        if is_main_process:
            print(f"\nEpoch {epoch+1}/{args.max_epochs}")
        
        train_metrics, global_step = train_epoch(
            model, train_loader, optimizer, device,
            args.lambda_latent, args.lambda_recon,
            args.grad_clip,
            start_step=global_step,
        )
        
        val_metrics = validate(
            model, val_loader, device,
            args.lambda_latent, args.lambda_recon,
            current_step=max(global_step, args.eval_gate_step),
        )
        
        if is_main_process:
            print(f"Train - loss: {train_metrics['loss']:.4f}, "
                  f"latent: {train_metrics['latent_loss']:.4f}, "
                  f"chi1_acc: {train_metrics['chi1_acc']:.4f}")
            print(f"Val   - loss: {val_metrics['loss']:.4f}, "
                  f"latent: {val_metrics['latent_loss']:.4f}, "
                  f"chi1_acc: {val_metrics['chi1_acc']:.4f}")
        
        record = {
            'epoch': epoch,
            'global_step': global_step,
            'train_loss': train_metrics['loss'],
            'train_latent_loss': train_metrics['latent_loss'],
            'train_recon_loss': train_metrics['recon_loss'],
            'train_chi1_acc': train_metrics['chi1_acc'],
            'val_loss': val_metrics['loss'],
            'val_latent_loss': val_metrics['latent_loss'],
            'val_recon_loss': val_metrics['recon_loss'],
            'val_chi1_acc': val_metrics['chi1_acc'],
        }
        metrics_log.append(record)
        
        if is_main_process:
            with open(log_dir / 'metrics.jsonl', 'a') as f:
                f.write(json.dumps(record) + '\n')
        
        if val_metrics['loss'] < best_val_loss:
            best_val_loss = val_metrics['loss']
            patience_counter = 0
            if is_main_process:
                model_to_save = model.module if distributed else model
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model_to_save.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_loss': val_metrics['loss'],
                    'global_step': global_step,
                    'config': config,
                }, save_dir / 'best_model.pt')
                print(f"  ✓ New best model saved (val_loss={val_metrics['loss']:.4f})")
        else:
            patience_counter += 1
            if is_main_process:
                print(f"  No improvement ({patience_counter}/{args.patience})")
        
        if patience_counter >= args.patience:
            if is_main_process:
                print(f"\nEarly stopping at epoch {epoch}")
            break
    
    if is_main_process:
        model_to_save = model.module if distributed else model
        torch.save({
            'epoch': epoch,
            'model_state_dict': model_to_save.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'val_loss': val_metrics['loss'],
            'global_step': global_step,
            'config': config,
        }, save_dir / 'latest_model.pt')
        
        print(f"\nTraining complete. Best val_loss: {best_val_loss:.4f}")
        print(f"Checkpoints saved to: {save_dir}")
        print(f"Metrics saved to: {log_dir / 'metrics.jsonl'}")


if __name__ == '__main__':
    main()
