#!/usr/bin/env python
"""Sanity check: χ1 rotamer bin labels, baselines, and coordinate frame."""
import sys, os, math, json
import numpy as np
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from src.stage1.datasets.dataset_stage1 import create_stage1_dataloader


def circular_nearest_label(chi1_rad):
    """Assign 0=g-, 1=g+, 2=t by circular nearest center."""
    centers = np.array([-np.pi/3, np.pi/3, np.pi])
    angle = ((chi1_rad + np.pi) % (2*np.pi)) - np.pi
    diff = angle[..., None] - centers
    circ = np.abs(((diff + np.pi) % (2*np.pi)) - np.pi)
    return circ.argmin(axis=-1)


def old_label(chi1_rad):
    """Old (buggy) bin assignment."""
    angle = ((chi1_rad + np.pi) % (2*np.pi)) - np.pi
    labels = np.zeros_like(angle, dtype=int)
    labels[(angle >= -np.pi/3) & (angle < np.pi/3)] = 1
    labels[angle >= np.pi/3] = 2
    return labels


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', default='processed_data/triplets')
    parser.add_argument('--val_samples_file', default=None)
    parser.add_argument('--max_samples', type=int, default=500)
    parser.add_argument('--max_n_res', type=int, default=900)
    args = parser.parse_args()

    loader = create_stage1_dataloader(
        args.data_dir, split='val', batch_size=1, shuffle=False,
        num_workers=0, max_n_res=args.max_n_res,
        valid_samples_file=args.val_samples_file,
        require_atom14=False,
    )

    all_holo_chi1 = []
    all_apo_chi1 = []
    all_chi1_mask = []
    all_aatype = []
    all_phi = []
    all_psi = []
    all_w_res = []
    n = 0

    for batch in loader:
        if batch is None:
            continue
        B, N = batch.node_mask.shape
        for i in range(B):
            mask = batch.node_mask[i].bool() & batch.chi_mask[i, :, 0].bool()
            holo = batch.chi_holo[i, :, 0].numpy()
            apo = batch.torsion_apo[i, :, 3].numpy()
            phi = batch.torsion_apo[i, :, 0].numpy()
            psi = batch.torsion_apo[i, :, 1].numpy()
            w = batch.w_res[i].numpy()
            m = mask.numpy()

            seq = batch.sequences[i]
            aa_idx = []
            aa_map = {c: i for i, c in enumerate("ACDEFGHIKLMNPQRSTVWY")}
            for c in seq:
                aa_idx.append(aa_map.get(c, 20))
            aa_idx = np.array(aa_idx[:N])

            all_holo_chi1.append(holo[m])
            all_apo_chi1.append(apo[m])
            all_chi1_mask.append(m)
            all_aatype.append(aa_idx[m])
            all_phi.append(phi[m])
            all_psi.append(psi[m])
            all_w_res.append(w[m])

        n += B
        if n >= args.max_samples:
            break

    holo = np.concatenate(all_holo_chi1)
    apo = np.concatenate(all_apo_chi1)
    aa = np.concatenate(all_aatype)
    w = np.concatenate(all_w_res)

    print(f"Total valid χ1 residues: {len(holo)}")
    print(f"Samples processed: {n}")
    print()

    # 1. Label comparison: old vs new
    new_labels = circular_nearest_label(holo)
    old_labels = old_label(holo)
    mismatch = (new_labels != old_labels).sum()
    print(f"=== BIN LABEL CHECK ===")
    print(f"Old vs New label mismatch: {mismatch} / {len(holo)} ({100*mismatch/len(holo):.1f}%)")
    print(f"New label distribution: g-={np.mean(new_labels==0):.3f}, g+={np.mean(new_labels==1):.3f}, t={np.mean(new_labels==2):.3f}")
    print(f"Old label distribution: 0={np.mean(old_labels==0):.3f}, 1={np.mean(old_labels==1):.3f}, 2={np.mean(old_labels==2):.3f}")
    print()

    # 2. Assigned center distance
    centers_deg = np.array([-60, 60, 180])
    holo_deg = np.degrees(((holo + np.pi) % (2*np.pi)) - np.pi)
    assigned_center = centers_deg[new_labels]
    diff = holo_deg - assigned_center
    circ_diff = np.abs(((diff + 180) % 360) - 180)
    print(f"=== ASSIGNED CENTER DISTANCE ===")
    print(f"Mean: {circ_diff.mean():.1f}°, Median: {np.median(circ_diff):.1f}°")
    print(f"P(<20°): {np.mean(circ_diff<20):.3f}, P(<40°): {np.mean(circ_diff<40):.3f}, P(<60°): {np.mean(circ_diff<60):.3f}")
    print()

    # 3. Apo carryover accuracy
    apo_labels = circular_nearest_label(apo)
    apo_acc_all = np.mean(apo_labels == new_labels)
    pocket_mask = w > 0.5
    apo_acc_pocket = np.mean(apo_labels[pocket_mask] == new_labels[pocket_mask]) if pocket_mask.sum() > 0 else float('nan')
    print(f"=== APO CARRYOVER ACCURACY ===")
    print(f"All residues: {apo_acc_all:.4f} ({apo_acc_all*100:.1f}%)")
    print(f"Pocket (w>0.5): {apo_acc_pocket:.4f} ({apo_acc_pocket*100:.1f}%)")
    switch_mask = apo_labels != new_labels
    print(f"Switch rate (apo!=holo): {switch_mask.mean():.4f} ({switch_mask.mean()*100:.1f}%)")
    print(f"Switch rate pocket: {switch_mask[pocket_mask].mean():.4f}" if pocket_mask.sum() > 0 else "")
    print()

    # 4. Majority class baseline
    majority_class = np.argmax(np.bincount(new_labels, minlength=3))
    majority_acc = np.mean(new_labels == majority_class)
    print(f"=== MAJORITY CLASS BASELINE ===")
    print(f"Majority class: {majority_class} (acc={majority_acc:.4f}, {majority_acc*100:.1f}%)")
    print()

    # 5. Per-residue-type majority
    aa_names = list("ACDEFGHIKLMNPQRSTVWY") + ["UNK"]
    per_aa_acc = []
    print(f"=== PER-RESIDUE-TYPE MAJORITY ===")
    for ai in range(21):
        mask_aa = aa == ai
        if mask_aa.sum() == 0:
            continue
        labels_aa = new_labels[mask_aa]
        maj = np.argmax(np.bincount(labels_aa, minlength=3))
        acc = np.mean(labels_aa == maj)
        per_aa_acc.append(acc * mask_aa.sum())
        dist = [np.mean(labels_aa==k) for k in range(3)]
        print(f"  {aa_names[ai]}: n={mask_aa.sum():5d}, majority_acc={acc:.3f}, dist=[{dist[0]:.2f},{dist[1]:.2f},{dist[2]:.2f}]")
    weighted_per_aa = sum(per_aa_acc) / len(holo)
    print(f"  Weighted per-aa majority: {weighted_per_aa:.4f} ({weighted_per_aa*100:.1f}%)")
    print()

    # 6. Holo self-accuracy check
    holo_self = circular_nearest_label(holo)
    self_acc = np.mean(holo_self == new_labels)
    print(f"=== HOLO SELF-ACCURACY (should be 100%) ===")
    print(f"Self accuracy: {self_acc:.6f}")
    print()

    # 7. Sign flip test
    flipped_labels = circular_nearest_label(-apo)
    flip_acc = np.mean(flipped_labels == new_labels)
    print(f"=== SIGN FLIP TEST ===")
    print(f"Apo carryover: {apo_acc_all:.4f}")
    print(f"-Apo (sign flip): {flip_acc:.4f}")
    if flip_acc > apo_acc_all:
        print("  WARNING: sign-flipped apo is BETTER than apo! Check angle convention.")
    print()

    # 8. Summary
    print(f"=== SUMMARY ===")
    print(f"Old/new bin mismatch: {100*mismatch/len(holo):.1f}%")
    print(f"Apo carryover: {apo_acc_all*100:.1f}%")
    print(f"Majority baseline: {majority_acc*100:.1f}%")
    print(f"Per-aa majority: {weighted_per_aa*100:.1f}%")
    print(f"Switch rate: {switch_mask.mean()*100:.1f}%")


if __name__ == '__main__':
    main()
