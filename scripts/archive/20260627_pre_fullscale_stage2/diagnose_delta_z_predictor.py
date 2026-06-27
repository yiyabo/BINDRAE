#!/usr/bin/env python3
"""
Diagnostic script for DeltaZPredictor ligand sensitivity.

This script evaluates whether the trained DeltaZPredictor produces different
delta_z predictions for different ligand conditions, which is necessary for
ligand-causal effects.

Metrics:
- delta_z magnitude for correct ligand
- delta_z magnitude for no ligand, scrambled ligand, translated ligand
- Sensitivity: ||delta_z_correct - delta_z_control|| / ||delta_z_correct||
"""

import sys
import json
import argparse
from pathlib import Path
from functools import partial
from dataclasses import replace

import torch
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from src.stage1.models.change_prediction_rae import ChangePredictionRAE, ChangePredictionRAEConfig
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
    parser = argparse.ArgumentParser(description='DeltaZPredictor ligand sensitivity diagnostic')
    
    parser.add_argument('--data_dir', type=str, required=True)
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Checkpoint from fast training (DeltaZPredictor) or full ChangePredictionRAE')
    parser.add_argument('--stage1_encoder_checkpoint', type=str, default=None,
                        help='Optional Stage-1 checkpoint for compatible encoder/conditioner modules')
    parser.add_argument('--gate_lambda', type=float, default=1.0,
                        help='Force ligand conditioner gate for diagnostics; use <0 to rely on current_step')
    parser.add_argument('--current_step', type=int, default=100000,
                        help='current_step used when gate_lambda < 0')
    parser.add_argument('--val_samples_file', type=str, default=None)
    parser.add_argument('--sample_metadata_file', type=str, default=None)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--max_n_res', type=int, default=1600)
    parser.add_argument('--max_batches', type=int, default=100,
                        help='Max batches to evaluate (for speed)')
    parser.add_argument('--output_json', type=str, required=True)
    parser.add_argument('--num_workers', type=int, default=2)
    parser.add_argument('--device', type=str, default='cuda')
    
    return parser.parse_args()


def make_no_ligand_batch(batch):
    return replace(
        batch,
        lig_points=torch.zeros_like(batch.lig_points),
        lig_types=torch.zeros_like(batch.lig_types),
        lig_mask=torch.zeros_like(batch.lig_mask),
    )


def make_scrambled_types_batch(batch):
    lig_types = batch.lig_types.clone()
    for i in range(batch.lig_types.shape[0]):
        valid = torch.nonzero(batch.lig_mask[i].bool(), as_tuple=False).squeeze(-1)
        if valid.numel() > 1:
            lig_types[i, valid] = batch.lig_types[i, valid.roll(shifts=1)]
    return replace(batch, lig_types=lig_types)


def make_translated_batch(batch, offset: float = 100.0):
    offset_vec = torch.tensor(
        [offset, offset, offset],
        device=batch.lig_points.device,
        dtype=batch.lig_points.dtype,
    ).view(1, 1, 3)
    return replace(batch, lig_points=batch.lig_points + offset_vec)


def make_batch_shuffled_ligand(batch):
    B = batch.lig_points.shape[0]
    if B < 2:
        return replace(batch)
    perm = torch.arange(B, device=batch.lig_points.device).roll(shifts=1)
    return replace(
        batch,
        lig_points=batch.lig_points.index_select(0, perm),
        lig_types=batch.lig_types.index_select(0, perm),
        lig_mask=batch.lig_mask.index_select(0, perm),
    )


@torch.no_grad()
def get_delta_z_pred(model, batch, gate_lambda=None, current_step: int = 100000):
    """Run ligand-conditioned DeltaZPredictor to get delta_z_pred and features."""
    return model.predict_delta_z(
        batch,
        current_step=current_step,
        gate_lambda=gate_lambda,
        return_conditioned=True,
    )


def compute_delta_z_stats(delta_z, node_mask):
    """Compute statistics of delta_z."""
    norm = torch.norm(delta_z, dim=-1)
    masked_norm = norm * node_mask
    mean_norm = masked_norm.sum() / node_mask.sum()
    return mean_norm.item()


def compute_l2_error(pred, target, mask):
    """Mean per-residue L2 error under a residue mask."""
    mask = mask.float()
    err = torch.norm(pred - target, dim=-1)
    return ((err * mask).sum() / mask.sum().clamp_min(1e-8)).item()


def compute_contact_mask(batch, threshold: float = 0.5):
    """Use Stage-1 pocket/contact weights as a ligand-facing subset."""
    contact_mask = (batch.w_res > threshold) & batch.node_mask.bool()
    if contact_mask.float().sum() < 1:
        contact_mask = batch.node_mask.bool()
    return contact_mask.float()


def main():
    args = parse_args()
    
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    print("Loading ChangePredictionRAE model...")
    config = ChangePredictionRAEConfig()
    model = ChangePredictionRAE(config).to(device)
    
    print(f"Loading DeltaZPredictor from {args.checkpoint}...")
    ckpt = torch.load(args.checkpoint, map_location=device)
    state = ckpt.get('model_state_dict', ckpt)
    loaded_full_checkpoint = False
    try:
        model.delta_z_predictor.load_state_dict(state)
        print("  loaded checkpoint as DeltaZPredictor-only state")
    except RuntimeError:
        missing, unexpected = model.load_state_dict(state, strict=False)
        print("  loaded checkpoint as partial/full ChangePredictionRAE state")
        print(f"  missing keys: {len(missing)}, unexpected keys: {len(unexpected)}")
        loaded_full_checkpoint = (len(missing) == 0 and len(unexpected) == 0)
    if args.stage1_encoder_checkpoint and not loaded_full_checkpoint:
        print(f"Loading compatible Stage-1 modules from {args.stage1_encoder_checkpoint}...")
        stage1_ckpt = torch.load(args.stage1_encoder_checkpoint, map_location=device)
        load_report = model.load_stage1_modules(
            stage1_ckpt,
            prefixes=('esm_adapter', 'edge_embedder', 'ipa_module'),
        )
        print(f"  loaded tensors: {len(load_report['loaded'])}")
        if load_report['skipped']:
            print(f"  skipped tensors: {len(load_report['skipped'])}")
    elif args.stage1_encoder_checkpoint and loaded_full_checkpoint:
        print("  full checkpoint loaded; skipping stage1_encoder_checkpoint to avoid overwriting trained modules")
    model.eval()
    gate_lambda = None if args.gate_lambda < 0 else float(args.gate_lambda)
    
    print("Creating dataloader...")
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
    
    print(f"Val samples: {len(val_dataset)}")
    
    all_stats = {
        'correct_ligand': [],
        'no_ligand': [],
        'scrambled_ligand': [],
        'translated_ligand': [],
        'batch_shuffled_ligand': [],
        'sensitivity_no_ligand': [],
        'sensitivity_scrambled': [],
        'sensitivity_translated': [],
        'sensitivity_batch_shuffled': [],
        'feature_sensitivity_no_ligand': [],
        'feature_sensitivity_scrambled': [],
        'feature_sensitivity_translated': [],
        'feature_sensitivity_batch_shuffled': [],
        'true_delta_z': [],
        'latent_error_correct': [],
        'latent_error_no_ligand': [],
        'latent_error_scrambled': [],
        'latent_error_translated': [],
        'latent_error_batch_shuffled': [],
        'contact_error_correct': [],
        'contact_error_no_ligand': [],
        'contact_error_scrambled': [],
        'contact_error_translated': [],
        'contact_error_batch_shuffled': [],
    }
    
    n_batches = 0
    for batch in tqdm(val_loader, desc='Evaluating'):
        if batch is None:
            continue
        if n_batches >= args.max_batches:
            break
        
        batch = batch_to_device(batch, device)

        with torch.no_grad():
            z_apo, _ = model.encode(batch, 'apo')
            z_holo, _ = model.encode(batch, 'holo')
            delta_z_true = z_holo - z_apo
        contact_mask = compute_contact_mask(batch)
        
        delta_z_correct, feat_correct = get_delta_z_pred(
            model, batch, gate_lambda=gate_lambda, current_step=args.current_step
        )

        batch_no_lig = make_no_ligand_batch(batch)
        delta_z_no_lig, feat_no_lig = get_delta_z_pred(
            model, batch_no_lig, gate_lambda=gate_lambda, current_step=args.current_step
        )

        batch_scrambled = make_scrambled_types_batch(batch)
        delta_z_scrambled, feat_scrambled = get_delta_z_pred(
            model, batch_scrambled, gate_lambda=gate_lambda, current_step=args.current_step
        )

        batch_translated = make_translated_batch(batch)
        delta_z_translated, feat_translated = get_delta_z_pred(
            model, batch_translated, gate_lambda=gate_lambda, current_step=args.current_step
        )

        batch_shuffled = make_batch_shuffled_ligand(batch)
        delta_z_shuffled, feat_shuffled = get_delta_z_pred(
            model, batch_shuffled, gate_lambda=gate_lambda, current_step=args.current_step
        )
        
        all_stats['correct_ligand'].append(compute_delta_z_stats(delta_z_correct, batch.node_mask))
        all_stats['no_ligand'].append(compute_delta_z_stats(delta_z_no_lig, batch.node_mask))
        all_stats['scrambled_ligand'].append(compute_delta_z_stats(delta_z_scrambled, batch.node_mask))
        all_stats['translated_ligand'].append(compute_delta_z_stats(delta_z_translated, batch.node_mask))
        all_stats['batch_shuffled_ligand'].append(compute_delta_z_stats(delta_z_shuffled, batch.node_mask))
        all_stats['true_delta_z'].append(compute_delta_z_stats(delta_z_true, batch.node_mask))
        
        diff_no = torch.norm(delta_z_correct - delta_z_no_lig, dim=-1)
        diff_scrambled = torch.norm(delta_z_correct - delta_z_scrambled, dim=-1)
        diff_translated = torch.norm(delta_z_correct - delta_z_translated, dim=-1)
        diff_shuffled = torch.norm(delta_z_correct - delta_z_shuffled, dim=-1)
        norm_correct = torch.norm(delta_z_correct, dim=-1)
        denom = (norm_correct * batch.node_mask).sum().clamp_min(1e-8)
        
        sens_no = (diff_no * batch.node_mask).sum() / denom
        sens_scrambled = (diff_scrambled * batch.node_mask).sum() / denom
        sens_translated = (diff_translated * batch.node_mask).sum() / denom
        sens_shuffled = (diff_shuffled * batch.node_mask).sum() / denom

        feat_norm = torch.norm(feat_correct, dim=-1)
        feat_denom = (feat_norm * batch.node_mask).sum().clamp_min(1e-8)
        feat_sens_no = (torch.norm(feat_correct - feat_no_lig, dim=-1) * batch.node_mask).sum() / feat_denom
        feat_sens_scrambled = (torch.norm(feat_correct - feat_scrambled, dim=-1) * batch.node_mask).sum() / feat_denom
        feat_sens_translated = (torch.norm(feat_correct - feat_translated, dim=-1) * batch.node_mask).sum() / feat_denom
        feat_sens_shuffled = (torch.norm(feat_correct - feat_shuffled, dim=-1) * batch.node_mask).sum() / feat_denom
        
        all_stats['sensitivity_no_ligand'].append(sens_no.item())
        all_stats['sensitivity_scrambled'].append(sens_scrambled.item())
        all_stats['sensitivity_translated'].append(sens_translated.item())
        all_stats['sensitivity_batch_shuffled'].append(sens_shuffled.item())
        all_stats['feature_sensitivity_no_ligand'].append(feat_sens_no.item())
        all_stats['feature_sensitivity_scrambled'].append(feat_sens_scrambled.item())
        all_stats['feature_sensitivity_translated'].append(feat_sens_translated.item())
        all_stats['feature_sensitivity_batch_shuffled'].append(feat_sens_shuffled.item())

        all_stats['latent_error_correct'].append(compute_l2_error(delta_z_correct, delta_z_true, batch.node_mask))
        all_stats['latent_error_no_ligand'].append(compute_l2_error(delta_z_no_lig, delta_z_true, batch.node_mask))
        all_stats['latent_error_scrambled'].append(compute_l2_error(delta_z_scrambled, delta_z_true, batch.node_mask))
        all_stats['latent_error_translated'].append(compute_l2_error(delta_z_translated, delta_z_true, batch.node_mask))
        all_stats['latent_error_batch_shuffled'].append(compute_l2_error(delta_z_shuffled, delta_z_true, batch.node_mask))

        all_stats['contact_error_correct'].append(compute_l2_error(delta_z_correct, delta_z_true, contact_mask))
        all_stats['contact_error_no_ligand'].append(compute_l2_error(delta_z_no_lig, delta_z_true, contact_mask))
        all_stats['contact_error_scrambled'].append(compute_l2_error(delta_z_scrambled, delta_z_true, contact_mask))
        all_stats['contact_error_translated'].append(compute_l2_error(delta_z_translated, delta_z_true, contact_mask))
        all_stats['contact_error_batch_shuffled'].append(compute_l2_error(delta_z_shuffled, delta_z_true, contact_mask))
        
        n_batches += 1
    
    results = {
        'n_batches': n_batches,
        'n_samples': n_batches * args.batch_size,
        'delta_z_magnitude': {
            'true_delta_z': float(np.mean(all_stats['true_delta_z'])),
            'correct_ligand': float(np.mean(all_stats['correct_ligand'])),
            'no_ligand': float(np.mean(all_stats['no_ligand'])),
            'scrambled_ligand': float(np.mean(all_stats['scrambled_ligand'])),
            'translated_ligand': float(np.mean(all_stats['translated_ligand'])),
            'batch_shuffled_ligand': float(np.mean(all_stats['batch_shuffled_ligand'])),
        },
        'sensitivity': {
            'vs_no_ligand': float(np.mean(all_stats['sensitivity_no_ligand'])),
            'vs_scrambled_ligand': float(np.mean(all_stats['sensitivity_scrambled'])),
            'vs_translated_ligand': float(np.mean(all_stats['sensitivity_translated'])),
            'vs_batch_shuffled_ligand': float(np.mean(all_stats['sensitivity_batch_shuffled'])),
        },
        'conditioned_feature_sensitivity': {
            'vs_no_ligand': float(np.mean(all_stats['feature_sensitivity_no_ligand'])),
            'vs_scrambled_ligand': float(np.mean(all_stats['feature_sensitivity_scrambled'])),
            'vs_translated_ligand': float(np.mean(all_stats['feature_sensitivity_translated'])),
            'vs_batch_shuffled_ligand': float(np.mean(all_stats['feature_sensitivity_batch_shuffled'])),
        },
        'latent_l2_error': {
            'correct_ligand': float(np.mean(all_stats['latent_error_correct'])),
            'no_ligand': float(np.mean(all_stats['latent_error_no_ligand'])),
            'scrambled_ligand': float(np.mean(all_stats['latent_error_scrambled'])),
            'translated_ligand': float(np.mean(all_stats['latent_error_translated'])),
            'batch_shuffled_ligand': float(np.mean(all_stats['latent_error_batch_shuffled'])),
        },
        'contact_l2_error': {
            'correct_ligand': float(np.mean(all_stats['contact_error_correct'])),
            'no_ligand': float(np.mean(all_stats['contact_error_no_ligand'])),
            'scrambled_ligand': float(np.mean(all_stats['contact_error_scrambled'])),
            'translated_ligand': float(np.mean(all_stats['contact_error_translated'])),
            'batch_shuffled_ligand': float(np.mean(all_stats['contact_error_batch_shuffled'])),
        },
        'diagnostic_gate_lambda': gate_lambda,
        'diagnostic_current_step': args.current_step,
    }

    correct_err = results['latent_l2_error']['correct_ligand']
    contact_correct_err = results['contact_l2_error']['correct_ligand']
    results['latent_error_lift_over_correct'] = {}
    results['contact_error_lift_over_correct'] = {}
    for key in ('no_ligand', 'scrambled_ligand', 'translated_ligand', 'batch_shuffled_ligand'):
        err = results['latent_l2_error'][key]
        contact_err = results['contact_l2_error'][key]
        results['latent_error_lift_over_correct'][key] = {
            'absolute': float(err - correct_err),
            'relative': float((err - correct_err) / max(correct_err, 1e-8)),
        }
        results['contact_error_lift_over_correct'][key] = {
            'absolute': float(contact_err - contact_correct_err),
            'relative': float((contact_err - contact_correct_err) / max(contact_correct_err, 1e-8)),
        }
    
    print("\n" + "="*60)
    print("DeltaZPredictor Ligand Sensitivity Diagnostic")
    print("="*60)
    print(f"Evaluating {results['n_batches']} batches ({results['n_samples']} samples)")
    print()
    print("Delta-z magnitude (mean L2 norm per residue):")
    for k, v in results['delta_z_magnitude'].items():
        print(f"  {k:25s}: {v:.6f}")
    print()
    print("Sensitivity (||delta_z_correct - delta_z_control|| / ||delta_z_correct||):")
    for k, v in results['sensitivity'].items():
        print(f"  {k:25s}: {v:.6f}")
    print()
    print("Conditioned-feature sensitivity:")
    for k, v in results['conditioned_feature_sensitivity'].items():
        print(f"  {k:25s}: {v:.6f}")
    print()
    print("Latent L2 error to true delta_z (lower is better):")
    for k, v in results['latent_l2_error'].items():
        print(f"  {k:25s}: {v:.6f}")
    print()
    print("Latent error lift over correct ligand (positive means correct ligand wins):")
    for k, v in results['latent_error_lift_over_correct'].items():
        print(f"  {k:25s}: abs={v['absolute']:.6f}, rel={v['relative']:.6f}")
    print()
    print("Contact/pocket L2 error to true delta_z (lower is better):")
    for k, v in results['contact_l2_error'].items():
        print(f"  {k:25s}: {v:.6f}")
    print()
    print("Contact/pocket error lift over correct ligand:")
    for k, v in results['contact_error_lift_over_correct'].items():
        print(f"  {k:25s}: abs={v['absolute']:.6f}, rel={v['relative']:.6f}")
    print()
    print("Interpretation:")
    print("  - Sensitivity close to 0: model ignores ligand (bad)")
    print("  - Sensitivity close to 1: model is highly sensitive to ligand (good)")
    print("  - Sensitivity > 1: control produces very different predictions (good)")
    print("  - Positive error lift: correct ligand is better than the control (good)")
    print("="*60)
    
    output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to: {output_path}")


if __name__ == '__main__':
    main()
