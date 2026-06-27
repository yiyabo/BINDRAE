#!/usr/bin/env python3
"""
Precompute latent representations for Change-Prediction RAE.

This script runs the frozen encoder on all samples to compute z_apo and z_holo,
which are then saved to disk for fast base-only latent-change baselines.

Usage:
    python scripts/precompute_latents.py --data_dir processed_data/triplets --output_dir processed_data/latents
"""

import os
import sys
import argparse
import json
from pathlib import Path

import torch
import numpy as np
from torch.utils.data import DataLoader
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
from src.stage1.datasets import (
    ApoHoloTripletDataset,
    collate_stage1_batch,
)
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
    parser = argparse.ArgumentParser(description='Precompute latent representations')
    
    parser.add_argument('--data_dir', type=str, required=True)
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--val_samples_file', type=str, default=None)
    parser.add_argument('--sample_metadata_file', type=str, default=None)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--max_n_res', type=int, default=1600)
    parser.add_argument('--num_workers', type=int, default=2)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--stage1_encoder_checkpoint', type=str, default=None,
                        help='Optional Stage-1 checkpoint whose compatible modules seed the frozen encoder')
    
    return parser.parse_args()


@torch.no_grad()
def precompute_latents(model, dataloader, device, output_dir: Path, split: str):
    model.eval()
    
    output_dir.mkdir(parents=True, exist_ok=True)
    shard_dir = output_dir / f'{split}_shards'
    shard_dir.mkdir(parents=True, exist_ok=True)
    index_file = output_dir / f'{split}_index.json'
    
    all_pdb_ids = []
    shard_records = []
    
    for i, batch in enumerate(tqdm(dataloader, desc=f'Precomputing {split}')):
        if batch is None:
            continue
        batch = batch_to_device(batch, device)
        
        z_apo, _ = model.encode(batch, 'apo')
        z_holo, _ = model.encode(batch, 'holo')
        
        z_apo_np = z_apo.cpu().numpy()
        z_holo_np = z_holo.cpu().numpy()
        node_mask_np = batch.node_mask.cpu().numpy()
        
        shard_name = f'{split}_{len(shard_records):06d}.npz'
        shard_path = shard_dir / shard_name
        np.savez(
            shard_path,
            z_apo=z_apo_np,
            z_holo=z_holo_np,
            node_mask=node_mask_np,
            pdb_ids=np.array(batch.pdb_ids),
        )
        shard_records.append({
            'file': str(shard_path.relative_to(output_dir)),
            'n_samples': int(z_apo_np.shape[0]),
            'n_res': int(z_apo_np.shape[1]),
            'c_s': int(z_apo_np.shape[2]),
        })
        
        all_pdb_ids.extend(batch.pdb_ids)
        
        if (i + 1) % 500 == 0:
            print(f"  Processed {i + 1} batches, {len(all_pdb_ids)} samples...")
    
    target_n_res = max((record['n_res'] for record in shard_records), default=0)
    with open(index_file, 'w') as f:
        json.dump({
            'format': 'bindrae_latent_shards_v1',
            'split': split,
            'n_samples': len(all_pdb_ids),
            'target_n_res': target_n_res,
            'shards': shard_records,
            'pdb_ids': all_pdb_ids,
        }, f, indent=2)
    
    print(f"✓ Saved {len(all_pdb_ids)} samples to {output_dir}")
    print(f"  Index: {index_file.name}")
    print(f"  Shards: {shard_dir.name}/ ({len(shard_records)} files)")


def main():
    args = parse_args()
    
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    output_dir = Path(args.output_dir)
    
    print("Creating model (frozen encoder)...")
    config = ChangePredictionRAEConfig()
    model = ChangePredictionRAE(config).to(device)
    if args.stage1_encoder_checkpoint:
        print(f"Loading compatible Stage-1 modules from {args.stage1_encoder_checkpoint}...")
        ckpt = torch.load(args.stage1_encoder_checkpoint, map_location=device)
        load_report = model.load_stage1_modules(ckpt)
        print(f"  loaded tensors: {len(load_report['loaded'])}")
        if load_report['skipped']:
            print(f"  skipped tensors: {len(load_report['skipped'])}")
    model.eval()
    
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
        shuffle=False,
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
    
    print("\nPrecomputing train latents...")
    precompute_latents(model, train_loader, device, output_dir, 'train')
    
    print("\nPrecomputing val latents...")
    precompute_latents(model, val_loader, device, output_dir, 'val')
    
    print(f"\n✓ Precomputation complete. Latents saved to: {output_dir}")


if __name__ == '__main__':
    main()
