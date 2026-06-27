#!/usr/bin/env python3
"""
Train chi head for ChangePredictionRAE using original dataset.

This script trains the chi head to predict chi angles from predicted holo latents.
"""

import sys
import json
import argparse
from pathlib import Path
from functools import partial

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from src.stage1.models.change_prediction_rae import ChangePredictionRAE, ChangePredictionRAEConfig
from src.stage1.models.torsion_head import TorsionHead
from src.stage1.modules.losses import torsion_sincos_loss
from src.stage1.datasets import ApoHoloTripletDataset, collate_stage1_batch


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
    parser = argparse.ArgumentParser(description='Train chi head for ChangePredictionRAE')
    
    parser.add_argument('--data_dir', type=str, required=True)
    parser.add_argument('--predictor_ckpt', type=str, required=True,
                        help='Checkpoint from fast training (DeltaZPredictor)')
    parser.add_argument('--gate_lambda', type=float, default=1.0,
                        help='Force ligand conditioner gate while building predicted latents; use <0 for current_step')
    parser.add_argument('--current_step', type=int, default=100000)
    parser.add_argument('--val_samples_file', type=str, default=None)
    parser.add_argument('--sample_metadata_file', type=str, default=None)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--max_n_res', type=int, default=1600)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--max_epochs', type=int, default=50)
    parser.add_argument('--grad_clip', type=float, default=1.0)
    parser.add_argument('--c_s', type=int, default=384)
    parser.add_argument('--torsion_hidden', type=int, default=128)
    parser.add_argument('--patience', type=int, default=10)
    parser.add_argument('--save_dir', type=str, default='checkpoints/stage1/chi_head')
    parser.add_argument('--log_dir', type=str, default='logs/stage1/chi_head')
    parser.add_argument('--num_workers', type=int, default=2)
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


@torch.no_grad()
def get_predicted_latents(model, batch, gate_lambda=None, current_step: int = 100000):
    """Run encoder + DeltaZPredictor to get z_holo_pred."""
    z_apo, _ = model.encode(batch, 'apo')
    delta_z_pred = model.predict_delta_z(
        batch,
        current_step=current_step,
        gate_lambda=gate_lambda,
    )
    z_holo_pred = z_apo + delta_z_pred
    return z_holo_pred


def train_epoch(model, chi_head, dataloader, optimizer, device, gate_lambda=None, current_step: int = 100000):
    chi_head.train()
    total_loss = 0.0
    total_chi_acc = 0.0
    n_batches = 0
    
    pbar = tqdm(dataloader, desc='Training')
    for batch in pbar:
        if batch is None:
            continue
        batch = batch_to_device(batch, device)
        
        z_holo_pred = get_predicted_latents(
            model, batch, gate_lambda=gate_lambda, current_step=current_step
        )
        
        optimizer.zero_grad()
        pred_chi = chi_head(z_holo_pred)
        loss = torsion_sincos_loss(pred_chi, batch.chi_holo, batch.chi_mask)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(chi_head.parameters(), 1.0)
        optimizer.step()
        
        chi_acc = compute_chi1_accuracy(pred_chi, batch.chi_holo, batch.chi_mask)
        
        total_loss += loss.item()
        total_chi_acc += chi_acc
        n_batches += 1
        
        pbar.set_postfix({'loss': f'{loss.item():.4f}', 'chi1_acc': f'{chi_acc:.4f}'})
    
    return {'loss': total_loss / n_batches, 'chi1_acc': total_chi_acc / n_batches}


@torch.no_grad()
def validate(model, chi_head, dataloader, device, gate_lambda=None, current_step: int = 100000):
    chi_head.eval()
    total_loss = 0.0
    total_chi_acc = 0.0
    n_batches = 0
    
    for batch in tqdm(dataloader, desc='Validation'):
        if batch is None:
            continue
        batch = batch_to_device(batch, device)
        
        z_holo_pred = get_predicted_latents(
            model, batch, gate_lambda=gate_lambda, current_step=current_step
        )
        pred_chi = chi_head(z_holo_pred)
        loss = torsion_sincos_loss(pred_chi, batch.chi_holo, batch.chi_mask)
        chi_acc = compute_chi1_accuracy(pred_chi, batch.chi_holo, batch.chi_mask)
        
        total_loss += loss.item()
        total_chi_acc += chi_acc
        n_batches += 1
    
    return {'loss': total_loss / n_batches, 'chi1_acc': total_chi_acc / n_batches}


def main():
    args = parse_args()
    
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    save_dir = Path(args.save_dir)
    log_dir = Path(args.log_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    
    print("Loading ChangePredictionRAE model...")
    config = ChangePredictionRAEConfig()
    model = ChangePredictionRAE(config).to(device)
    
    print(f"Loading DeltaZPredictor from {args.predictor_ckpt}...")
    ckpt = torch.load(args.predictor_ckpt, map_location=device)
    model.delta_z_predictor.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    gate_lambda = None if args.gate_lambda < 0 else float(args.gate_lambda)
    
    print("Creating chi head...")
    chi_head = TorsionHead(
        c_s=args.c_s,
        c_hidden=args.torsion_hidden,
        dropout=0.1,
        n_angles=4,
    ).to(device)
    
    print(f"Chi head parameters: {sum(p.numel() for p in chi_head.parameters()):,}")
    
    print("Creating dataloaders...")
    train_dataset = ApoHoloTripletDataset(
        data_dir=args.data_dir,
        split='train',
        sample_metadata_file=args.sample_metadata_file,
        require_atom14=False,
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
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
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=partial(collate_stage1_batch, max_n_res=args.max_n_res),
    )
    
    print(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")
    
    optimizer = torch.optim.Adam(chi_head.parameters(), lr=args.lr)
    
    best_val_loss = float('inf')
    patience_counter = 0
    
    for epoch in range(args.max_epochs):
        print(f"\nEpoch {epoch+1}/{args.max_epochs}")
        
        train_metrics = train_epoch(
            model, chi_head, train_loader, optimizer, device,
            gate_lambda=gate_lambda, current_step=args.current_step,
        )
        val_metrics = validate(
            model, chi_head, val_loader, device,
            gate_lambda=gate_lambda, current_step=args.current_step,
        )
        
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
