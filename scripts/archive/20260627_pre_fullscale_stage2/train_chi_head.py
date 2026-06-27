#!/usr/bin/env python3
"""
Train chi head using precomputed latents and trained DeltaZPredictor.

This script:
1. Loads precomputed z_apo
2. Uses trained DeltaZPredictor to predict delta_z
3. Computes z_holo_pred = z_apo + delta_z_pred
4. Trains chi head to predict chi_holo from z_holo_pred

This ensures the chi head learns to work with predicted latents, not true latents.
"""

import sys
import json
import argparse
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import numpy as np

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from src.stage1.models.delta_z_predictor import DeltaZPredictor
from src.stage1.models.torsion_head import TorsionHead
from src.stage1.modules.losses import torsion_sincos_loss


class ChiHeadDataset(Dataset):
    """Dataset for training chi head on predicted latents."""
    
    def __init__(self, latent_dir: Path, predictor_ckpt: Path, split: str = 'train'):
        self.latent_dir = Path(latent_dir)
        self.split = split
        
        z_apo_file = self.latent_dir / f'{split}_z_apo.npy'
        node_mask_file = self.latent_dir / f'{split}_node_mask.npy'
        
        print(f"Loading {split} latents from {latent_dir}...")
        self.z_apo = np.load(z_apo_file, mmap_mode='r')
        self.node_mask = np.load(node_mask_file, mmap_mode='r')
        
        print(f"  z_apo shape: {self.z_apo.shape}")
        print(f"  node_mask shape: {self.node_mask.shape}")
        
        print(f"Loading DeltaZPredictor from {predictor_ckpt}...")
        ckpt = torch.load(predictor_ckpt, map_location='cpu')
        self.predictor = DeltaZPredictor(c_s=384, hidden=256, n_layers=3)
        self.predictor.load_state_dict(ckpt['model_state_dict'])
        self.predictor.eval()
        
        print(f"Precomputing z_holo_pred for {len(self.z_apo)} samples...")
        with torch.no_grad():
            z_apo_tensor = torch.from_numpy(self.z_apo[:])
            node_mask_tensor = torch.from_numpy(self.node_mask[:])
            delta_z_pred = self.predictor(z_apo_tensor, node_mask_tensor)
            self.z_holo_pred = (z_apo_tensor + delta_z_pred).numpy()
        
        print(f"  z_holo_pred shape: {self.z_holo_pred.shape}")
        
        chi_holo_file = self.latent_dir.parent / 'triplets' / f'{split}_chi_holo.npy'
        chi_mask_file = self.latent_dir.parent / 'triplets' / f'{split}_chi_mask.npy'
        
        print(f"Loading chi supervision from {chi_holo_file.parent}...")
        self.chi_holo = np.load(chi_holo_file, mmap_mode='r')
        self.chi_mask = np.load(chi_mask_file, mmap_mode='r')
        
        print(f"  chi_holo shape: {self.chi_holo.shape}")
        print(f"  chi_mask shape: {self.chi_mask.shape}")
    
    def __len__(self):
        return len(self.z_holo_pred)
    
    def __getitem__(self, idx):
        z_holo_pred = torch.from_numpy(self.z_holo_pred[idx].copy())
        chi_holo = torch.from_numpy(self.chi_holo[idx].copy())
        chi_mask = torch.from_numpy(self.chi_mask[idx].copy())
        node_mask = torch.from_numpy(self.node_mask[idx].copy())
        
        return {
            'z_holo_pred': z_holo_pred,
            'chi_holo': chi_holo,
            'chi_mask': chi_mask,
            'node_mask': node_mask,
        }


def parse_args():
    parser = argparse.ArgumentParser(description='Train chi head on predicted latents')
    
    parser.add_argument('--latent_dir', type=str, required=True)
    parser.add_argument('--predictor_ckpt', type=str, required=True)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--max_epochs', type=int, default=50)
    parser.add_argument('--grad_clip', type=float, default=1.0)
    parser.add_argument('--c_s', type=int, default=384)
    parser.add_argument('--torsion_hidden', type=int, default=128)
    parser.add_argument('--patience', type=int, default=10)
    parser.add_argument('--save_dir', type=str, default='checkpoints/stage1/chi_head')
    parser.add_argument('--log_dir', type=str, default='logs/stage1/chi_head')
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--device', type=str, default='cuda')
    
    return parser.parse_args()


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


def train_epoch(model, dataloader, optimizer, device):
    model.train()
    total_loss = 0.0
    total_chi_acc = 0.0
    n_batches = 0
    
    pbar = tqdm(dataloader, desc='Training')
    for batch in pbar:
        z_holo_pred = batch['z_holo_pred'].to(device)
        chi_holo = batch['chi_holo'].to(device)
        chi_mask = batch['chi_mask'].to(device)
        
        optimizer.zero_grad()
        
        pred_chi = model(z_holo_pred)
        
        loss = torsion_sincos_loss(pred_chi, chi_holo, chi_mask)
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        chi_acc = compute_chi1_accuracy(pred_chi, chi_holo, chi_mask)
        
        total_loss += loss.item()
        total_chi_acc += chi_acc
        n_batches += 1
        
        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'chi1_acc': f'{chi_acc:.4f}',
        })
    
    return {
        'loss': total_loss / n_batches,
        'chi1_acc': total_chi_acc / n_batches,
    }


@torch.no_grad()
def validate(model, dataloader, device):
    model.eval()
    total_loss = 0.0
    total_chi_acc = 0.0
    n_batches = 0
    
    for batch in tqdm(dataloader, desc='Validation'):
        z_holo_pred = batch['z_holo_pred'].to(device)
        chi_holo = batch['chi_holo'].to(device)
        chi_mask = batch['chi_mask'].to(device)
        
        pred_chi = model(z_holo_pred)
        
        loss = torsion_sincos_loss(pred_chi, chi_holo, chi_mask)
        
        chi_acc = compute_chi1_accuracy(pred_chi, chi_holo, chi_mask)
        
        total_loss += loss.item()
        total_chi_acc += chi_acc
        n_batches += 1
    
    return {
        'loss': total_loss / n_batches,
        'chi1_acc': total_chi_acc / n_batches,
    }


def main():
    args = parse_args()
    
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    save_dir = Path(args.save_dir)
    log_dir = Path(args.log_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    
    print("Creating chi head model...")
    chi_head = TorsionHead(
        c_s=args.c_s,
        c_hidden=args.torsion_hidden,
        dropout=0.1,
        n_angles=4,
    ).to(device)
    
    print(f"Chi head parameters: {sum(p.numel() for p in chi_head.parameters()):,}")
    
    print("Creating datasets...")
    train_dataset = ChiHeadDataset(
        Path(args.latent_dir),
        Path(args.predictor_ckpt),
        'train',
    )
    val_dataset = ChiHeadDataset(
        Path(args.latent_dir),
        Path(args.predictor_ckpt),
        'val',
    )
    
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
    
    optimizer = torch.optim.Adam(chi_head.parameters(), lr=args.lr)
    
    best_val_loss = float('inf')
    patience_counter = 0
    
    for epoch in range(args.max_epochs):
        print(f"\nEpoch {epoch+1}/{args.max_epochs}")
        
        train_metrics = train_epoch(chi_head, train_loader, optimizer, device)
        val_metrics = validate(chi_head, val_loader, device)
        
        print(f"Train - loss: {train_metrics['loss']:.6f}, chi1_acc: {train_metrics['chi1_acc']:.4f}")
        print(f"Val   - loss: {val_metrics['loss']:.6f}, chi1_acc: {val_metrics['chi1_acc']:.4f}")
        
        record = {
            'epoch': epoch,
            'train_loss': train_metrics['loss'],
            'train_chi1_acc': train_metrics['chi1_acc'],
            'val_loss': val_metrics['loss'],
            'val_chi1_acc': val_metrics['chi1_acc'],
        }
        
        with open(log_dir / 'metrics.jsonl', 'a') as f:
            f.write(json.dumps(record) + '\n')
        
        if val_metrics['loss'] < best_val_loss:
            best_val_loss = val_metrics['loss']
            patience_counter = 0
            torch.save({
                'epoch': epoch,
                'model_state_dict': chi_head.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_metrics['loss'],
                'val_chi1_acc': val_metrics['chi1_acc'],
            }, save_dir / 'best_model.pt')
            print(f"  ✓ New best model saved (val_loss={val_metrics['loss']:.6f}, chi1_acc={val_metrics['chi1_acc']:.4f})")
        else:
            patience_counter += 1
            print(f"  No improvement ({patience_counter}/{args.patience})")
        
        if patience_counter >= args.patience:
            print(f"\nEarly stopping at epoch {epoch}")
            break
    
    torch.save({
        'epoch': epoch,
        'model_state_dict': chi_head.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'val_loss': val_metrics['loss'],
        'val_chi1_acc': val_metrics['chi1_acc'],
    }, save_dir / 'latest_model.pt')
    
    print(f"\nTraining complete. Best val_loss: {best_val_loss:.6f}")
    print(f"Checkpoints saved to: {save_dir}")
    print(f"Metrics saved to: {log_dir / 'metrics.jsonl'}")


if __name__ == '__main__':
    main()
