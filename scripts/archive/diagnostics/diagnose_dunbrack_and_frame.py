#!/usr/bin/env python
"""
Diagnostic: Dunbrack-like train-table baseline + ligand pose frame check.

1. Build (aatype, phi_bin, psi_bin) -> holo_chi1_bin majority table from training data.
   Evaluate on validation set. Also test with apo_chi1_bin as extra key.
2. Check ligand pose frame alignment: min(candidate_sc_atom, ligand_atom) distances.
"""
import sys, os, math
import numpy as np
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from src.stage1.datasets.dataset_stage1 import create_stage1_dataloader


def circular_nearest_label(chi1_rad):
    """Assign 0=g-, 1=g+, 2=t by circular nearest center."""
    centers = np.array([-np.pi / 3, np.pi / 3, np.pi])
    angle = ((chi1_rad + np.pi) % (2 * np.pi)) - np.pi
    diff = angle[..., None] - centers
    circ = np.abs(((diff + np.pi) % (2 * np.pi)) - np.pi)
    return circ.argmin(axis=-1)


def angle_to_bin_idx(angle_rad, n_bins=12):
    """Bin phi/psi into n_bins equal sectors in [-pi, pi)."""
    angle = ((angle_rad + np.pi) % (2 * np.pi)) - np.pi
    bin_idx = ((angle + np.pi) / (2 * np.pi) * n_bins).astype(int)
    return np.clip(bin_idx, 0, n_bins - 1)


def collect_data(loader, max_samples=2000):
    """Collect per-residue data from loader."""
    records = []
    n = 0
    for batch in loader:
        if batch is None:
            continue
        B, N = batch.node_mask.shape
        for i in range(B):
            mask = batch.node_mask[i].bool() & batch.chi_mask[i, :, 0].bool()
            holo_chi1 = batch.chi_holo[i, :, 0].numpy()
            apo_chi1 = batch.torsion_apo[i, :, 3].numpy()
            phi = batch.torsion_apo[i, :, 0].numpy()
            psi = batch.torsion_apo[i, :, 1].numpy()
            w = batch.w_res[i].numpy()
            m = mask.numpy()

            seq = batch.sequences[i]
            aa_map = {c: idx for idx, c in enumerate("ACDEFGHIKLMNPQRSTVWY")}
            for j in range(min(N, len(seq))):
                if not m[j]:
                    continue
                aa = aa_map.get(seq[j], 20)
                records.append({
                    'aatype': aa,
                    'phi': phi[j],
                    'psi': psi[j],
                    'holo_chi1': holo_chi1[j],
                    'apo_chi1': apo_chi1[j],
                    'w_res': w[j],
                    'seq_char': seq[j],
                })
        n += B
        if n >= max_samples:
            break
    return records


def dunbrack_baseline(train_records, val_records, use_apo_key=False, n_phi_bins=12, n_psi_bins=12):
    """
    Build majority-class table from train_records, evaluate on val_records.

    Key: (aatype, phi_bin, psi_bin[, apo_chi1_bin])
    """
    # Build table
    table = defaultdict(lambda: defaultdict(int))
    for r in train_records:
        phi_bin = angle_to_bin_idx(r['phi'], n_phi_bins)
        psi_bin = angle_to_bin_idx(r['psi'], n_psi_bins)
        holo_label = circular_nearest_label(np.array([r['holo_chi1']]))[0]

        if use_apo_key:
            apo_label = circular_nearest_label(np.array([r['apo_chi1']]))[0]
            key = (r['aatype'], phi_bin, psi_bin, apo_label)
        else:
            key = (r['aatype'], phi_bin, psi_bin)
        table[key][holo_label] += 1

    # Build majority prediction per key
    majority = {}
    for key, counts in table.items():
        majority[key] = max(counts, key=counts.get)

    # Evaluate on val
    correct = 0
    total = 0
    per_aa_correct = defaultdict(int)
    per_aa_total = defaultdict(int)

    for r in val_records:
        phi_bin = angle_to_bin_idx(r['phi'], n_phi_bins)
        psi_bin = angle_to_bin_idx(r['psi'], n_psi_bins)
        holo_label = circular_nearest_label(np.array([r['holo_chi1']]))[0]

        if use_apo_key:
            apo_label = circular_nearest_label(np.array([r['apo_chi1']]))[0]
            key = (r['aatype'], phi_bin, psi_bin, apo_label)
        else:
            key = (r['aatype'], phi_bin, psi_bin)

        pred = majority.get(key, 0)  # default to class 0 if unseen
        if pred == holo_label:
            correct += 1
        total += 1
        per_aa_correct[r['aatype']] += (1 if pred == holo_label else 0)
        per_aa_total[r['aatype']] += 1

    acc = correct / total if total > 0 else 0
    per_aa_acc = {}
    aa_names = list("ACDEFGHIKLMNPQRSTVWY") + ["UNK"]
    for aa in range(21):
        if per_aa_total[aa] > 0:
            per_aa_acc[aa_names[aa]] = per_aa_correct[aa] / per_aa_total[aa]

    return acc, per_aa_acc, total, len(table)


def ligand_frame_check(records_by_batch, loader, max_samples=200):
    """
    Check ligand pose frame alignment.

    For each residue, compute min distance from candidate sidechain atoms
    (using apo backbone FK) to ligand atoms. We approximate by checking
    Ca-apo to nearest ligand atom distance as a quick proxy.
    """
    ca_lig_dists = []
    pocket_ca_lig_dists = []
    nonpocket_ca_lig_dists = []

    n = 0
    for batch in loader:
        if batch is None:
            continue
        B, N = batch.node_mask.shape
        for i in range(B):
            mask = batch.node_mask[i].bool()
            ca = batch.Ca_apo[i].numpy()  # [N, 3]
            lig = batch.lig_points[i].numpy()  # [M, 3]
            lig_m = batch.lig_mask[i].bool().numpy()  # [M]
            w = batch.w_res[i].numpy()

            lig_valid = lig[lig_m]  # [M', 3]
            if len(lig_valid) == 0:
                continue

            for j in range(N):
                if not mask[j]:
                    continue
                dists = np.sqrt(np.sum((ca[j] - lig_valid) ** 2, axis=-1))
                min_dist = dists.min()
                ca_lig_dists.append(min_dist)

                if w[j] > 0.5:
                    pocket_ca_lig_dists.append(min_dist)
                else:
                    nonpocket_ca_lig_dists.append(min_dist)

        n += B
        if n >= max_samples:
            break

    return {
        'all': np.array(ca_lig_dists),
        'pocket': np.array(pocket_ca_lig_dists),
        'nonpocket': np.array(nonpocket_ca_lig_dists),
    }


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', default='processed_data/triplets')
    parser.add_argument('--train_samples_file', default=None)
    parser.add_argument('--val_samples_file', default=None)
    parser.add_argument('--max_train', type=int, default=2000)
    parser.add_argument('--max_val', type=int, default=2000)
    parser.add_argument('--max_n_res', type=int, default=900)
    args = parser.parse_args()

    print("=" * 60)
    print("DIAGNOSTIC: Dunbrack Baseline + Ligand Frame Check")
    print("=" * 60)

    # --- Part 1: Dunbrack-like baseline ---
    print("\n[1/2] Collecting training data...")
    train_loader = create_stage1_dataloader(
        args.data_dir, split='train', batch_size=1, shuffle=False,
        num_workers=0, max_n_res=args.max_n_res,
        valid_samples_file=args.train_samples_file,
        require_atom14=False,
    )
    train_records = collect_data(train_loader, max_samples=args.max_train)
    print(f"  Collected {len(train_records)} train residues from up to {args.max_train} samples")

    print("\n[2/2] Collecting validation data...")
    val_loader = create_stage1_dataloader(
        args.data_dir, split='val', batch_size=1, shuffle=False,
        num_workers=0, max_n_res=args.max_n_res,
        valid_samples_file=args.val_samples_file,
        require_atom14=False,
    )
    val_records = collect_data(val_loader, max_samples=args.max_val)
    print(f"  Collected {len(val_records)} val residues from up to {args.max_val} samples")

    # --- Baselines ---
    print("\n" + "=" * 60)
    print("DUNBRACK-LIKE BASELINES")
    print("=" * 60)

    # Global majority
    holo_labels = circular_nearest_label(np.array([r['holo_chi1'] for r in val_records]))
    apo_labels = circular_nearest_label(np.array([r['apo_chi1'] for r in val_records]))
    global_majority_class = np.argmax(np.bincount(holo_labels, minlength=3))
    global_majority_acc = np.mean(holo_labels == global_majority_class)
    print(f"\nGlobal majority class baseline: {global_majority_acc:.4f} ({global_majority_acc*100:.1f}%)")
    print(f"  Class distribution: g-={np.mean(holo_labels==0):.3f}, g+={np.mean(holo_labels==1):.3f}, t={np.mean(holo_labels==2):.3f}")

    # Apo carryover
    apo_acc = np.mean(apo_labels == holo_labels)
    pocket_mask = np.array([r['w_res'] for r in val_records]) > 0.5
    apo_acc_pocket = np.mean(apo_labels[pocket_mask] == holo_labels[pocket_mask]) if pocket_mask.sum() > 0 else 0
    print(f"\nApo carryover baseline:")
    print(f"  All:     {apo_acc:.4f} ({apo_acc*100:.1f}%)")
    print(f"  Pocket:  {apo_acc_pocket:.4f} ({apo_acc_pocket*100:.1f}%)")

    # Dunbrack (aatype, phi, psi)
    acc_basic, per_aa_basic, n_val, n_keys = dunbrack_baseline(
        train_records, val_records, use_apo_key=False, n_phi_bins=12, n_psi_bins=12
    )
    print(f"\nDunbrack baseline (aatype, phi_bin, psi_bin):")
    print(f"  Accuracy: {acc_basic:.4f} ({acc_basic*100:.1f}%)")
    print(f"  Unique keys in table: {n_keys}")
    print(f"  Val residues: {n_val}")

    # Dunbrack + apo chi1
    acc_apo, per_aa_apo, _, n_keys_apo = dunbrack_baseline(
        train_records, val_records, use_apo_key=True, n_phi_bins=12, n_psi_bins=12
    )
    print(f"\nDunbrack baseline (aatype, phi_bin, psi_bin, apo_chi1_bin):")
    print(f"  Accuracy: {acc_apo:.4f} ({acc_apo*100:.1f}%)")
    print(f"  Unique keys in table: {n_keys_apo}")

    # Per-residue-type comparison
    aa_names = list("ACDEFGHIKLMNPQRSTVWY") + ["UNK"]
    print(f"\nPer-residue-type accuracy:")
    print(f"  {'AA':>3s}  {'Majority':>8s}  {'Dunbrack':>8s}  {'Dun+Apo':>8s}  {'ApoCarr':>8s}")
    for aa in range(21):
        name = aa_names[aa]
        aa_mask = np.array([r['aatype'] for r in val_records]) == aa
        if aa_mask.sum() == 0:
            continue
        aa_holo = holo_labels[aa_mask]
        aa_apo = apo_labels[aa_mask]
        maj = np.argmax(np.bincount(aa_holo, minlength=3))
        maj_acc = np.mean(aa_holo == maj)
        apo_aa = np.mean(aa_apo == aa_holo)
        d_basic = per_aa_basic.get(name, 0)
        d_apo = per_aa_apo.get(name, 0)
        print(f"  {name:>3s}  {maj_acc:>8.3f}  {d_basic:>8.3f}  {d_apo:>8.3f}  {apo_aa:>8.3f}  (n={aa_mask.sum()})")

    # --- Part 2: Ligand frame check ---
    print("\n" + "=" * 60)
    print("LIGAND POSE FRAME CHECK")
    print("=" * 60)

    # Re-create loaders for frame check
    frame_loader = create_stage1_dataloader(
        args.data_dir, split='val', batch_size=1, shuffle=False,
        num_workers=0, max_n_res=args.max_n_res,
        valid_samples_file=args.val_samples_file,
        require_atom14=False,
    )
    dists = ligand_frame_check(None, frame_loader, max_samples=200)

    print(f"\nCa(Apo) to nearest ligand atom distances:")
    for subset_name, subset_dists in dists.items():
        if len(subset_dists) == 0:
            print(f"  {subset_name}: no data")
            continue
        print(f"  {subset_name}:")
        print(f"    N:        {len(subset_dists)}")
        print(f"    Mean:     {subset_dists.mean():.1f} A")
        print(f"    Median:   {np.median(subset_dists):.1f} A")
        print(f"    P(<5A):   {np.mean(subset_dists < 5):.3f}")
        print(f"    P(<8A):   {np.mean(subset_dists < 8):.3f}")
        print(f"    P(<12A):  {np.mean(subset_dists < 12):.3f}")
        print(f"    P(>20A):  {np.mean(subset_dists > 20):.3f}")
        print(f"    P(>50A):  {np.mean(subset_dists > 50):.3f}")

    # Sanity: pocket residues should be close to ligand
    if len(dists['pocket']) > 0 and len(dists['nonpocket']) > 0:
        print(f"\n  Pocket vs non-pocket comparison:")
        print(f"    Pocket median:     {np.median(dists['pocket']):.1f} A")
        print(f"    Non-pocket median: {np.median(dists['nonpocket']):.1f} A")
        if np.median(dists['pocket']) > 15:
            print(f"  WARNING: Pocket residues are far from ligand! Possible frame alignment issue.")
        elif np.median(dists['pocket']) < 8:
            print(f"  OK: Pocket residues are close to ligand. Frame alignment looks correct.")

    # --- Summary ---
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"  Global majority baseline:   {global_majority_acc*100:.1f}%")
    print(f"  Apo carryover (all):        {apo_acc*100:.1f}%")
    print(f"  Apo carryover (pocket):     {apo_acc_pocket*100:.1f}%")
    print(f"  Dunbrack (aa,phi,psi):      {acc_basic*100:.1f}%")
    print(f"  Dunbrack+apo (aa,phi,psi,chi1): {acc_apo*100:.1f}%")
    print(f"  Current model (M3 unfreeze): ~58.0%")
    if acc_basic > 0.65:
        print(f"\n  Dunbrack baseline > 65%: room for model to improve with better base prior.")
    elif acc_basic > 0.55:
        print(f"\n  Dunbrack baseline ~55-65%: task is moderately hard. Current model is in range.")
    else:
        print(f"\n  Dunbrack baseline < 55%: task is very hard with apo backbone.")


if __name__ == '__main__':
    main()
