#!/usr/bin/env python3
"""
Fast training script for Change-Prediction RAE using precomputed latents.

This script trains a base-only latent-change predictor using precomputed
z_apo and z_holo, avoiding the need to run the frozen encoder on every batch.

It does not consume ligand tokens. Use it as a base-only baseline unless the
latent cache is extended to include ligand-conditioned features.

Usage:
    python scripts/train_change_prediction_fast.py --latent_dir processed_data/latents --max_epochs 50
"""

import os
import sys
import json
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, Any

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import numpy as np

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

try:
    import torch.multiprocessing as mp
    mp.set_sharing_strategy('file_system')
except Exception:
    pass

from src.stage1.models.delta_z_predictor import DeltaZPredictor
from src.stage1.modules.losses import (
    latent_change_prediction_loss,
)


class PrecomputedLatentDataset(Dataset):
    """Dataset that loads precomputed latent representations."""
    
    def __init__(self, latent_dir: Path, split: str = 'train'):
        self.latent_dir = Path(latent_dir)
        self.split = split
        self.mode = 'monolithic'
        self._shard_cache = {}
        self._shard_cache_order = []
        self._max_cached_shards = 2

        index_file = self.latent_dir / f'{split}_index.json'
        if index_file.exists():
            self.mode = 'sharded'
            with open(index_file) as f:
                index = json.load(f)
            self.shards = index['shards']
            self.target_n_res = int(index['target_n_res'])
            self.entries = []
            for shard_idx, shard in enumerate(self.shards):
                for local_idx in range(int(shard['n_samples'])):
                    self.entries.append((shard_idx, local_idx))
            print(f"Loading {split} sharded latents from {index_file}...")
            print(f"  shards: {len(self.shards)}")
            print(f"  samples: {len(self.entries)}")
            print(f"  target_n_res: {self.target_n_res}")
            return
        
        z_apo_file = self.latent_dir / f'{split}_z_apo.npy'
        z_holo_file = self.latent_dir / f'{split}_z_holo.npy'
        node_mask_file = self.latent_dir / f'{split}_node_mask.npy'
        
        print(f"Loading {split} latents from {latent_dir}...")
        print("  [WARN] Using legacy monolithic .npy cache. If this cache was produced by the old append writer, regenerate it with scripts/precompute_latents.py.")
        self.z_apo = np.load(z_apo_file, mmap_mode='r')
        self.z_holo = np.load(z_holo_file, mmap_mode='r')
        self.node_mask = np.load(node_mask_file, mmap_mode='r')
        
        print(f"  z_apo shape: {self.z_apo.shape}")
        print(f"  z_holo shape: {self.z_holo.shape}")
        print(f"  node_mask shape: {self.node_mask.shape}")
        
        assert self.z_apo.shape == self.z_holo.shape
        assert self.z_apo.shape[:2] == self.node_mask.shape

    @staticmethod
    def _pad_residue_axis(array: np.ndarray, target_n_res: int, value=0):
        if array.shape[0] == target_n_res:
            return array
        if array.shape[0] > target_n_res:
            raise ValueError(f"latent residue length {array.shape[0]} exceeds target {target_n_res}")
        out_shape = (target_n_res,) + array.shape[1:]
        out = np.full(out_shape, value, dtype=array.dtype)
        out[:array.shape[0]] = array
        return out

    def _load_shard(self, shard_idx: int):
        if shard_idx in self._shard_cache:
            return self._shard_cache[shard_idx]
        shard_path = self.latent_dir / self.shards[shard_idx]['file']
        shard = np.load(shard_path, allow_pickle=False)
        self._shard_cache[shard_idx] = shard
        self._shard_cache_order.append(shard_idx)
        while len(self._shard_cache_order) > self._max_cached_shards:
            old_idx = self._shard_cache_order.pop(0)
            old = self._shard_cache.pop(old_idx, None)
            if old is not None:
                old.close()
        return shard
    
    def __len__(self):
        if self.mode == 'sharded':
            return len(self.entries)
        return len(self.z_apo)
    
    def __getitem__(self, idx):
        if self.mode == 'sharded':
            shard_idx, local_idx = self.entries[idx]
            shard = self._load_shard(shard_idx)
            z_apo_np = self._pad_residue_axis(shard['z_apo'][local_idx], self.target_n_res)
            z_holo_np = self._pad_residue_axis(shard['z_holo'][local_idx], self.target_n_res)
            node_mask_np = self._pad_residue_axis(shard['node_mask'][local_idx], self.target_n_res, value=False)
        else:
            z_apo_np = self.z_apo[idx]
            z_holo_np = self.z_holo[idx]
            node_mask_np = self.node_mask[idx]

        z_apo = torch.from_numpy(z_apo_np.copy())
        z_holo = torch.from_numpy(z_holo_np.copy())
        node_mask = torch.from_numpy(node_mask_np.copy())
        
        delta_z_true = z_holo - z_apo
        
        return {
            'z_apo': z_apo,
            'z_holo': z_holo,
            'delta_z_true': delta_z_true,
            'node_mask': node_mask,
        }


def parse_args():
    parser = argparse.ArgumentParser(description='Fast training for Change-Prediction RAE')
    
    parser.add_argument('--latent_dir', type=str, required=True)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--max_epochs', type=int, default=50)
    parser.add_argument('--grad_clip', type=float, default=1.0)
    parser.add_argument('--c_s', type=int, default=384)
    parser.add_argument('--delta_z_hidden', type=int, default=256)
    parser.add_argument('--delta_z_layers', type=int, default=3)
    parser.add_argument('--patience', type=int, default=10)
    parser.add_argument('--save_dir', type=str, default='checkpoints/stage1/change_prediction_fast')
    parser.add_argument('--log_dir', type=str, default='logs/stage1/change_prediction_fast')
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--allow_base_only', action='store_true',
                        help='Acknowledge that this fast path has no ligand input and is only a base-only baseline')
    
    return parser.parse_args()


def train_epoch(model, dataloader, optimizer, device):
    model.train()
    total_loss = 0.0
    n_batches = 0
    
    pbar = tqdm(dataloader, desc='Training')
    for batch in pbar:
        z_apo = batch['z_apo'].to(device)
        delta_z_true = batch['delta_z_true'].to(device)
        node_mask = batch['node_mask'].to(device)
        
        optimizer.zero_grad()
        
        delta_z_pred = model(z_apo, node_mask)
        
        loss = latent_change_prediction_loss(
            delta_z_pred,
            delta_z_true,
            node_mask,
        )
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        total_loss += loss.item()
        n_batches += 1
        
        pbar.set_postfix({'loss': f'{loss.item():.4f}'})
    
    return {
        'loss': total_loss / n_batches,
    }


@torch.no_grad()
def validate(model, dataloader, device):
    model.eval()
    total_loss = 0.0
    n_batches = 0
    
    for batch in tqdm(dataloader, desc='Validation'):
        z_apo = batch['z_apo'].to(device)
        delta_z_true = batch['delta_z_true'].to(device)
        node_mask = batch['node_mask'].to(device)
        
        delta_z_pred = model(z_apo, node_mask)
        
        loss = latent_change_prediction_loss(
            delta_z_pred,
            delta_z_true,
            node_mask,
        )
        
        total_loss += loss.item()
        n_batches += 1
    
    return {
        'loss': total_loss / n_batches,
    }


def main():
    args = parse_args()
    
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    if not args.allow_base_only:
        raise SystemExit(
            "train_change_prediction_fast.py has no ligand input. Re-run with "
            "--allow_base_only only if you intentionally want a base-only latent-change baseline."
        )
    print("[WARN] Fast change-prediction path is base-only: DeltaZPredictor receives z_apo, not ligand tokens.")
    
    save_dir = Path(args.save_dir)
    log_dir = Path(args.log_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    
    print("Creating model...")
    model = DeltaZPredictor(
        c_s=args.c_s,
        hidden=args.delta_z_hidden,
        n_layers=args.delta_z_layers,
    ).to(device)
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    print("Creating datasets...")
    train_dataset = PrecomputedLatentDataset(Path(args.latent_dir), 'train')
    val_dataset = PrecomputedLatentDataset(Path(args.latent_dir), 'val')
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    
    print(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")
    
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    
    best_val_loss = float('inf')
    patience_counter = 0
    
    for epoch in range(args.max_epochs):
        print(f"\nEpoch {epoch+1}/{args.max_epochs}")
        
        train_metrics = train_epoch(model, train_loader, optimizer, device)
        val_metrics = validate(model, val_loader, device)
        
        print(f"Train - loss: {train_metrics['loss']:.6f}")
        print(f"Val   - loss: {val_metrics['loss']:.6f}")
        
        record = {
            'epoch': epoch,
            'train_loss': train_metrics['loss'],
            'val_loss': val_metrics['loss'],
        }
        
        with open(log_dir / 'metrics.jsonl', 'a') as f:
            f.write(json.dumps(record) + '\n')
        
        if val_metrics['loss'] < best_val_loss:
            best_val_loss = val_metrics['loss']
            patience_counter = 0
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_metrics['loss'],
            }, save_dir / 'best_model.pt')
            print(f"  ✓ New best model saved (val_loss={val_metrics['loss']:.6f})")
        else:
            patience_counter += 1
            print(f"  No improvement ({patience_counter}/{args.patience})")
        
        if patience_counter >= args.patience:
            print(f"\nEarly stopping at epoch {epoch}")
            break
    
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'val_loss': val_metrics['loss'],
    }, save_dir / 'latest_model.pt')
    
    print(f"\nTraining complete. Best val_loss: {best_val_loss:.6f}")
    print(f"Checkpoints saved to: {save_dir}")
    print(f"Metrics saved to: {log_dir / 'metrics.jsonl'}")


if __name__ == '__main__':
    main()
