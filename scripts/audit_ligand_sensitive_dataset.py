#!/usr/bin/env python3
"""Audit ligand-sensitive Stage-1 validation subsets without model training.

This script answers whether the data/split contains residues that can support
ligand-causal chi1 learning: apo->holo rotamer switches, ligand-contact chi1
residues, and their intersections. It is intentionally data-only and should be
run through a Slurm wrapper for full validation scans on the cluster.
"""

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import torch

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from src.stage1.datasets.dataset_stage1 import create_stage1_dataloader
from utils.metrics import compute_residue_contact_mask, wrap_angle_diff


ROTAMER_CENTERS = np.array([-math.pi / 3.0, math.pi / 3.0, math.pi], dtype=np.float32)
AA_RESTYPE_MAP = {
    'A': 0, 'R': 1, 'N': 2, 'D': 3, 'C': 4,
    'Q': 5, 'E': 6, 'G': 7, 'H': 8, 'I': 9,
    'L': 10, 'K': 11, 'M': 12, 'F': 13, 'P': 14,
    'S': 15, 'T': 16, 'W': 17, 'Y': 18, 'V': 19,
}


def rotamer_labels(chi1: np.ndarray) -> np.ndarray:
    angle = ((chi1 + math.pi) % (2.0 * math.pi)) - math.pi
    diff = angle[..., None] - ROTAMER_CENTERS
    circ_dist = np.abs(np.arctan2(np.sin(diff), np.cos(diff)))
    return np.argmin(circ_dist, axis=-1).astype(np.int64)


def valid_ligand_coords(lig_points: torch.Tensor, lig_mask: torch.Tensor) -> np.ndarray:
    mask = lig_mask.detach().cpu().numpy().astype(bool)
    if not mask.any():
        return np.zeros((0, 3), dtype=np.float32)
    return lig_points.detach().cpu().numpy().astype(np.float32)[mask]


def ca_contact_mask(ca_coords: np.ndarray, lig_coords: np.ndarray, threshold: float) -> np.ndarray:
    if lig_coords.size == 0:
        return np.zeros((ca_coords.shape[0],), dtype=bool)
    dists = np.linalg.norm(ca_coords[:, None, :] - lig_coords[None, :, :], axis=-1)
    return np.min(dists, axis=1) <= threshold


def parse_thresholds(value: str) -> List[float]:
    thresholds = []
    for item in value.split(','):
        item = item.strip()
        if not item:
            continue
        thresholds.append(float(item))
    if not thresholds:
        raise ValueError('--contact_thresholds must contain at least one threshold')
    return sorted(set(thresholds))


def threshold_key(threshold: float) -> str:
    return f'{threshold:.1f}'.replace('.', 'p').replace('-', 'm')


def safe_div(numerator: float, denominator: float) -> float:
    if denominator <= 0:
        return float('nan')
    return float(numerator / denominator)


def add_count(stats: Dict[str, float], key: str, value: int) -> None:
    stats[key] = stats.get(key, 0.0) + float(value)


def finite_mean(values: Iterable[float]) -> float:
    arr = np.asarray([v for v in values if math.isfinite(float(v))], dtype=np.float64)
    if arr.size == 0:
        return float('nan')
    return float(arr.mean())


def sequences_to_aatype(sequences: List[str], max_len: int, device: torch.device) -> torch.Tensor:
    aatype = torch.zeros(len(sequences), max_len, dtype=torch.long, device=device)
    for i, seq in enumerate(sequences):
        for j, aa in enumerate(seq):
            if j >= max_len:
                break
            aatype[i, j] = AA_RESTYPE_MAP.get(aa, 0)
    return aatype


def load_fk_tools():
    try:
        from src.stage1.models.fk_openfold import create_openfold_fk, reorder_torsions_to_openfold
    except ModuleNotFoundError as exc:
        if exc.name == 'flash_ipa':
            raise RuntimeError(
                'FK-derived contact audit requires flash_ipa. Run this audit in the '
                'BINDRAE cluster environment or install vendor/flash_ipa locally.'
            ) from exc
        raise
    return create_openfold_fk, reorder_torsions_to_openfold


def build_rigids_from_backbone(
    n_coord: torch.Tensor,
    ca_coord: torch.Tensor,
    c_coord: torch.Tensor,
    mask: torch.Tensor,
    eps: float = 1e-6,
):
    try:
        from flash_ipa.rigid import Rigid, Rotation
    except ModuleNotFoundError as exc:
        if exc.name == 'flash_ipa':
            raise RuntimeError(
                'FK-derived contact audit requires flash_ipa. Run this audit in the '
                'BINDRAE cluster environment or install vendor/flash_ipa locally.'
            ) from exc
        raise

    device = ca_coord.device
    default_e1 = torch.tensor([1.0, 0.0, 0.0], device=device)
    default_e2 = torch.tensor([0.0, 1.0, 0.0], device=device)
    default_e3 = torch.tensor([0.0, 0.0, 1.0], device=device)

    e1 = c_coord - ca_coord
    e1_norm = torch.norm(e1, dim=-1, keepdim=True)
    e1 = torch.where(e1_norm > eps, e1 / torch.clamp(e1_norm, min=eps), default_e1.expand_as(e1))

    u = n_coord - ca_coord
    proj = (u * e1).sum(dim=-1, keepdim=True) * e1
    e2 = u - proj
    e2_norm = torch.norm(e2, dim=-1, keepdim=True)
    e2 = torch.where(e2_norm > eps, e2 / torch.clamp(e2_norm, min=eps), default_e2.expand_as(e2))

    e3 = torch.cross(e1, e2, dim=-1)
    e3_norm = torch.norm(e3, dim=-1, keepdim=True)
    e3 = torch.where(e3_norm > eps, e3 / torch.clamp(e3_norm, min=eps), default_e3.expand_as(e3))

    rot = torch.stack([e1, e2, e3], dim=-1)
    trans = ca_coord
    if mask is not None:
        mask_rot = mask.bool().unsqueeze(-1).unsqueeze(-1)
        eye = torch.eye(3, device=device).view(1, 1, 3, 3)
        rot = torch.where(mask_rot, rot, eye)
        trans = torch.where(mask.bool().unsqueeze(-1), trans, torch.zeros_like(trans))

    rot = torch.where(torch.isnan(rot), torch.eye(3, device=device).view(1, 1, 3, 3).expand_as(rot), rot)
    trans = torch.where(torch.isnan(trans), torch.zeros_like(trans), trans)
    return Rigid(rots=Rotation(rot_mats=rot), trans=trans)


def build_fk_atom14_holo(fk_module, reorder_torsions_to_openfold, batch) -> Tuple[torch.Tensor, torch.Tensor]:
    torsion_sincos = torch.stack(
        [torch.sin(batch.torsion_holo), torch.cos(batch.torsion_holo)],
        dim=-1,
    )
    torsion_sincos = reorder_torsions_to_openfold(torsion_sincos)
    backbone_rigids = build_rigids_from_backbone(
        batch.N_holo.float(),
        batch.Ca_holo.float(),
        batch.C_holo.float(),
        batch.node_mask.bool(),
    )
    _, max_len = batch.torsion_holo.shape[:2]
    aatype = sequences_to_aatype(batch.sequences, max_len, batch.torsion_holo.device)
    result = fk_module(torsion_sincos.float(), backbone_rigids, aatype)
    return result['atom14_pos'], result['atom14_mask'].bool()


def finalize_rates(stats: Dict[str, float]) -> Dict[str, float]:
    out = dict(stats)
    chi1_total = max(out.get('chi1_valid_residues', 0.0), 1.0)
    node_total = max(out.get('valid_residues', 0.0), 1.0)
    sample_total = max(out.get('samples', 0.0), 1.0)
    for prefix in ('switch', 'apo_wrong20', 'apo_distance_pocket', 'ca_contact', 'atom14_contact',
                   'fk_atom14_contact', 'ca_contact_switch', 'atom14_contact_switch',
                   'fk_atom14_contact_switch', 'pocket_switch'):
        out[f'{prefix}_fraction_of_chi1'] = out.get(prefix, 0.0) / chi1_total
    out['apo_distance_pocket_fraction_of_valid_residues'] = out.get('apo_distance_pocket', 0.0) / node_total
    out['mean_chi1_valid_per_sample'] = out.get('chi1_valid_residues', 0.0) / sample_total
    out['mean_switch_per_sample'] = out.get('switch', 0.0) / sample_total
    out['mean_ca_contact_switch_per_sample'] = out.get('ca_contact_switch', 0.0) / sample_total
    out['mean_atom14_contact_switch_per_sample'] = out.get('atom14_contact_switch', 0.0) / sample_total
    out['mean_fk_atom14_contact_switch_per_sample'] = out.get('fk_atom14_contact_switch', 0.0) / sample_total
    out['raw_atom14_available_sample_fraction'] = out.get('atom14_available', 0.0) / sample_total
    out['fk_atom14_available_sample_fraction'] = out.get('fk_atom14_available', 0.0) / sample_total
    return out


def finalize_threshold_audit(threshold_stats: Dict[str, Dict[str, float]]) -> Dict[str, Dict[str, float]]:
    finalized = {}
    for key, stat in threshold_stats.items():
        chi1_total = stat.get('chi1_valid_residues', 0.0)
        ca_contact = stat.get('ca_contact', 0.0)
        fk_contact = stat.get('fk_atom14_contact', 0.0)
        overlap = stat.get('ca_fk_overlap', 0.0)
        union = stat.get('ca_fk_union', 0.0)
        translated_fk = stat.get('translated_fk_atom14_contact', 0.0)
        translated_ca = stat.get('translated_ca_contact', 0.0)
        out = dict(stat)
        out.update({
            'ca_contact_fraction_of_chi1': safe_div(ca_contact, chi1_total),
            'fk_atom14_contact_fraction_of_chi1': safe_div(fk_contact, chi1_total),
            'ca_fk_jaccard': safe_div(overlap, union),
            'ca_overlap_fraction': safe_div(overlap, ca_contact),
            'fk_overlap_fraction': safe_div(overlap, fk_contact),
            'translated_fk_fraction_of_original_fk': safe_div(translated_fk, fk_contact),
            'translated_ca_fraction_of_original_ca': safe_div(translated_ca, ca_contact),
            'translated_fk_fraction_of_chi1': safe_div(translated_fk, chi1_total),
            'translated_ca_fraction_of_chi1': safe_div(translated_ca, chi1_total),
        })
        finalized[key] = out
    return finalized


def main() -> None:
    parser = argparse.ArgumentParser(description='Audit ligand-sensitive subsets in Stage-1 data')
    parser.add_argument('--data_dir', default='processed_data/triplets')
    parser.add_argument('--split', default='val', choices=['train', 'val', 'test'])
    parser.add_argument('--valid_samples_file', default=None)
    parser.add_argument('--sample_metadata_file', default='sample_metadata.json')
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--num_workers', type=int, default=2)
    parser.add_argument('--max_n_res', type=int, default=900)
    parser.add_argument('--max_batches', type=int, default=None)
    parser.add_argument('--ca_contact_threshold', type=float, default=8.0)
    parser.add_argument('--atom_contact_threshold', type=float, default=4.5)
    parser.add_argument('--contact_thresholds', type=str, default='3.5,4.0,4.5,5.0',
                        help='comma-separated thresholds for CA-vs-FK contact stability audit')
    parser.add_argument('--pocket_threshold', type=float, default=0.5)
    parser.add_argument('--angle_threshold_deg', type=float, default=20.0)
    parser.add_argument('--output_dir', default='logs/stage1_diagnostics/ligand_sensitive_dataset_audit')
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    loader = create_stage1_dataloader(
        args.data_dir,
        split=args.split,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        max_n_res=args.max_n_res,
        valid_samples_file=args.valid_samples_file,
        sample_metadata_file=args.sample_metadata_file,
        require_atom14=False,
    )

    stats: Dict[str, float] = {}
    per_sample: List[Dict[str, object]] = []
    n_batches = 0
    angle_threshold = float(args.angle_threshold_deg)
    thresholds = parse_thresholds(args.contact_thresholds)
    if args.atom_contact_threshold not in thresholds:
        thresholds = sorted(set(thresholds + [float(args.atom_contact_threshold)]))
    main_threshold_key = threshold_key(float(args.atom_contact_threshold))
    threshold_stats = {
        threshold_key(threshold): {
            'threshold': float(threshold),
            'samples': 0.0,
            'chi1_valid_residues': 0.0,
            'ca_contact': 0.0,
            'fk_atom14_contact': 0.0,
            'raw_atom14_contact': 0.0,
            'ca_fk_overlap': 0.0,
            'ca_fk_union': 0.0,
            'translated_ca_contact': 0.0,
            'translated_fk_atom14_contact': 0.0,
        }
        for threshold in thresholds
    }
    create_openfold_fk, reorder_torsions_to_openfold = load_fk_tools()
    fk_module = create_openfold_fk()
    fk_module.eval()

    for batch in loader:
        if batch is None:
            continue
        n_batches += 1
        batch_size = len(batch.pdb_ids)
        with torch.no_grad():
            fk_atom14_batch, fk_atom14_mask_batch = build_fk_atom14_holo(
                fk_module,
                reorder_torsions_to_openfold,
                batch,
            )
        for i in range(batch_size):
            sample_id = batch.pdb_ids[i]
            node_mask = batch.node_mask[i].numpy().astype(bool)
            chi1_valid = batch.chi_mask[i, :, 0].numpy().astype(bool) & node_mask
            apo_chi1 = batch.torsion_apo[i, :, 3].numpy()
            holo_chi1 = batch.torsion_holo[i, :, 3].numpy()
            apo_bins = rotamer_labels(apo_chi1)
            holo_bins = rotamer_labels(holo_chi1)
            switch = chi1_valid & (apo_bins != holo_bins)

            apo_err_deg = np.abs(wrap_angle_diff(apo_chi1, holo_chi1)) * 180.0 / math.pi
            apo_wrong20 = chi1_valid & (apo_err_deg >= angle_threshold)

            lig_coords = valid_ligand_coords(batch.lig_points[i], batch.lig_mask[i])
            ca_apo = batch.Ca_apo[i].numpy().astype(np.float32)
            ca_contact = chi1_valid & ca_contact_mask(ca_apo, lig_coords, args.ca_contact_threshold)

            atom14_mask = batch.atom14_holo_mask[i].numpy().astype(bool)
            atom14_available = bool((atom14_mask & node_mask[:, None]).any())
            if atom14_available:
                atom_contact = compute_residue_contact_mask(
                    batch.atom14_holo[i].numpy().astype(np.float32),
                    atom14_mask,
                    lig_coords,
                    contact_threshold=args.atom_contact_threshold,
                ) & chi1_valid
            else:
                atom_contact = np.zeros_like(chi1_valid, dtype=bool)
            fk_atom14_mask = fk_atom14_mask_batch[i].detach().cpu().numpy().astype(bool)
            fk_atom14_available = bool((fk_atom14_mask & node_mask[:, None]).any())
            if fk_atom14_available:
                fk_atom_contact = compute_residue_contact_mask(
                    fk_atom14_batch[i].detach().cpu().numpy().astype(np.float32),
                    fk_atom14_mask,
                    lig_coords,
                    contact_threshold=args.atom_contact_threshold,
                ) & chi1_valid
            else:
                fk_atom_contact = np.zeros_like(chi1_valid, dtype=bool)
            pocket = chi1_valid & (batch.w_res[i].numpy() > args.pocket_threshold)

            translated_lig_coords = lig_coords + np.array([100.0, 100.0, 100.0], dtype=np.float32)
            threshold_sample_counts: Dict[str, int] = {}
            for threshold in thresholds:
                t_key = threshold_key(threshold)
                ca_at_t = chi1_valid & ca_contact_mask(ca_apo, lig_coords, threshold)
                ca_translated = chi1_valid & ca_contact_mask(ca_apo, translated_lig_coords, threshold)
                if atom14_available:
                    raw_at_t = compute_residue_contact_mask(
                        batch.atom14_holo[i].numpy().astype(np.float32),
                        atom14_mask,
                        lig_coords,
                        contact_threshold=threshold,
                    ) & chi1_valid
                else:
                    raw_at_t = np.zeros_like(chi1_valid, dtype=bool)
                if fk_atom14_available:
                    fk_at_t = compute_residue_contact_mask(
                        fk_atom14_batch[i].detach().cpu().numpy().astype(np.float32),
                        fk_atom14_mask,
                        lig_coords,
                        contact_threshold=threshold,
                    ) & chi1_valid
                    fk_translated = compute_residue_contact_mask(
                        fk_atom14_batch[i].detach().cpu().numpy().astype(np.float32),
                        fk_atom14_mask,
                        translated_lig_coords,
                        contact_threshold=threshold,
                    ) & chi1_valid
                else:
                    fk_at_t = np.zeros_like(chi1_valid, dtype=bool)
                    fk_translated = np.zeros_like(chi1_valid, dtype=bool)

                overlap = ca_at_t & fk_at_t
                union = ca_at_t | fk_at_t
                stat = threshold_stats[t_key]
                stat['samples'] += 1.0
                stat['chi1_valid_residues'] += float(chi1_valid.sum())
                stat['ca_contact'] += float(ca_at_t.sum())
                stat['fk_atom14_contact'] += float(fk_at_t.sum())
                stat['raw_atom14_contact'] += float(raw_at_t.sum())
                stat['ca_fk_overlap'] += float(overlap.sum())
                stat['ca_fk_union'] += float(union.sum())
                stat['translated_ca_contact'] += float(ca_translated.sum())
                stat['translated_fk_atom14_contact'] += float(fk_translated.sum())
                threshold_sample_counts[f'ca_contact_t{t_key}'] = int(ca_at_t.sum())
                threshold_sample_counts[f'fk_atom14_contact_t{t_key}'] = int(fk_at_t.sum())
                threshold_sample_counts[f'ca_fk_overlap_t{t_key}'] = int(overlap.sum())
                threshold_sample_counts[f'translated_fk_atom14_contact_t{t_key}'] = int(fk_translated.sum())

            sample_counts = {
                'sample_id': sample_id,
                'n_residues': int(batch.n_residues[i]),
                'valid_residues': int(node_mask.sum()),
                'chi1_valid_residues': int(chi1_valid.sum()),
                'switch': int(switch.sum()),
                'apo_wrong20': int(apo_wrong20.sum()),
                'apo_distance_pocket': int(pocket.sum()),
                'ca_contact': int(ca_contact.sum()),
                'atom14_contact': int(atom_contact.sum()),
                'fk_atom14_contact': int(fk_atom_contact.sum()),
                'ca_contact_switch': int((ca_contact & switch).sum()),
                'atom14_contact_switch': int((atom_contact & switch).sum()),
                'fk_atom14_contact_switch': int((fk_atom_contact & switch).sum()),
                'pocket_switch': int((pocket & switch).sum()),
                'ligand_atoms': int(batch.lig_mask[i].sum().item()),
                'atom14_available': int(atom14_available),
                'fk_atom14_available': int(fk_atom14_available),
                'mean_apo_holo_chi1_error_deg': finite_mean(apo_err_deg[chi1_valid]),
            }
            sample_counts.update(threshold_sample_counts)
            per_sample.append(sample_counts)
            add_count(stats, 'samples', 1)
            for key, value in sample_counts.items():
                if isinstance(value, int):
                    add_count(stats, key, value)

        if args.max_batches is not None and n_batches >= args.max_batches:
            break

    summary = finalize_rates(stats)
    summary.update({
        'data_dir': str(Path(args.data_dir).resolve()),
        'split': args.split,
        'valid_samples_file': args.valid_samples_file,
        'n_batches': n_batches,
        'contact_threshold_audit': finalize_threshold_audit(threshold_stats),
        'decoy_semantics': {
            'translated_away': 'ligand coordinates are shifted by +100A on x/y/z; identity and intra-ligand geometry are preserved but local contact is removed',
            'scrambled_types': 'ligand coordinates and mask are preserved; ligand type rows are permuted within each ligand',
            'batch_shuffled_ligand': 'ligand coordinates/types/mask are swapped across samples in the batch; this is not a same-pocket chemical decoy',
        },
        'thresholds': {
            'ca_contact_threshold': args.ca_contact_threshold,
            'atom_contact_threshold': args.atom_contact_threshold,
            'main_contact_threshold_key': main_threshold_key,
            'contact_thresholds': thresholds,
            'pocket_threshold': args.pocket_threshold,
            'angle_threshold_deg': args.angle_threshold_deg,
            'max_n_res': args.max_n_res,
        },
    })

    json_path = out_dir / 'ligand_sensitive_dataset_audit.json'
    csv_path = out_dir / 'ligand_sensitive_dataset_audit_per_sample.csv'
    with json_path.open('w', encoding='utf-8') as f:
        json.dump({'summary': summary, 'per_sample': per_sample}, f, indent=2, ensure_ascii=False)
    fieldnames = list(per_sample[0].keys()) if per_sample else ['sample_id']
    with csv_path.open('w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(per_sample)

    print('=== Ligand-sensitive dataset audit ===')
    for key in sorted(summary):
        print(f'{key}: {summary[key]}')
    print(f'JSON: {json_path}')
    print(f'CSV:  {csv_path}')


if __name__ == '__main__':
    main()
