#!/usr/bin/env python3
"""
Build stricter Stage-1 screening sample lists.

The base valid lists only check file existence and basic numeric health. For
ligand-causality screening we also need usable apo/holo alignment, chi1
supervision, local ligand-contact switch signal, and ligand elements that are
represented by the current ligand type encoding.
"""

import argparse
import json
import math
import sys
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.stage1.datasets.dataset_stage1 import (  # noqa: E402
    _coords_valid_mask,
    _load_torsions,
    align_by_residue_ids,
    extract_backbone_coords,
)


MAPPED_LIGAND_ELEMENTS = {"C", "N", "O", "S", "P", "F", "Cl", "Br", "I"}
ROTAMER_CENTERS = np.array([-math.pi / 3.0, math.pi / 3.0, math.pi], dtype=np.float32)


def load_sample_ids(path: Path) -> List[str]:
    if path.suffix == ".json":
        data = json.loads(path.read_text())
        if isinstance(data, dict):
            raw = data.get("pdb_ids", data.get("ids"))
            if raw is None:
                raise ValueError(f"Unsupported JSON sample list: {path}")
        else:
            raw = data
    else:
        raw = [line.strip() for line in path.read_text().splitlines() if line.strip()]

    out: List[str] = []
    seen = set()
    for item in raw:
        sample_id = item if isinstance(item, str) else item.get("id")
        if sample_id and sample_id not in seen:
            out.append(sample_id)
            seen.add(sample_id)
    return out


def parse_sdf_symbols(path: Path) -> List[str]:
    try:
        lines = path.read_text(errors="ignore").splitlines()
    except OSError:
        return []
    if len(lines) < 4:
        return []
    try:
        n_atoms = int(lines[3][:3])
    except ValueError:
        return []
    symbols: List[str] = []
    for line in lines[4: 4 + n_atoms]:
        parts = line.split()
        if len(parts) >= 4:
            symbols.append(parts[3])
    return symbols


def rotamer_labels(chi1: np.ndarray) -> np.ndarray:
    angle = ((chi1 + math.pi) % (2.0 * math.pi)) - math.pi
    diff = angle[..., None] - ROTAMER_CENTERS
    circ_dist = np.abs(np.arctan2(np.sin(diff), np.cos(diff)))
    return np.argmin(circ_dist, axis=-1).astype(np.int64)


def ca_contact_mask(ca_coords: np.ndarray, lig_coords: np.ndarray, threshold: float) -> np.ndarray:
    if lig_coords.size == 0:
        return np.zeros((ca_coords.shape[0],), dtype=bool)
    dists = np.linalg.norm(ca_coords[:, None, :] - lig_coords[None, :, :], axis=-1)
    return np.min(dists, axis=1) <= threshold


def align_backbones(sample_dir: Path, n_res: int) -> Tuple[np.ndarray, np.ndarray]:
    apo = extract_backbone_coords(sample_dir / "apo.pdb")
    holo = extract_backbone_coords(sample_dir / "holo.pdb")
    n_apo, ca_apo, c_apo, _, apo_res_ids = apo
    n_holo, ca_holo, c_holo, _, holo_res_ids = holo

    if apo_res_ids is not None and holo_res_ids is not None:
        (n_apo, ca_apo, c_apo), _, node_mask = align_by_residue_ids(
            (n_apo, ca_apo, c_apo),
            apo_res_ids,
            (n_holo, ca_holo, c_holo),
            holo_res_ids,
            n_res,
        )
    else:
        n_valid = min(len(n_apo), len(n_holo), n_res)
        n_apo_out = np.zeros((n_res, 3), dtype=np.float32)
        ca_apo_out = np.zeros((n_res, 3), dtype=np.float32)
        c_apo_out = np.zeros((n_res, 3), dtype=np.float32)
        n_holo_out = np.zeros((n_res, 3), dtype=np.float32)
        ca_holo_out = np.zeros((n_res, 3), dtype=np.float32)
        c_holo_out = np.zeros((n_res, 3), dtype=np.float32)
        n_apo_out[:n_valid] = n_apo[:n_valid]
        ca_apo_out[:n_valid] = ca_apo[:n_valid]
        c_apo_out[:n_valid] = c_apo[:n_valid]
        n_holo_out[:n_valid] = n_holo[:n_valid]
        ca_holo_out[:n_valid] = ca_holo[:n_valid]
        c_holo_out[:n_valid] = c_holo[:n_valid]
        node_mask = _coords_valid_mask(n_apo_out, ca_apo_out, c_apo_out) & _coords_valid_mask(
            n_holo_out, ca_holo_out, c_holo_out
        )
        ca_apo = ca_apo_out

    return ca_apo.astype(np.float32), node_mask.astype(bool)


def evaluate_sample(args_tuple) -> Dict:
    sample_id, data_dir, cfg = args_tuple
    sample_dir = data_dir / "samples" / sample_id
    reasons: List[str] = []
    metrics: Dict[str, object] = {"sample_id": sample_id}

    try:
        esm = torch.load(sample_dir / "esm.pt", weights_only=False)["per_residue"]
        n_res = int(esm.shape[0])
        metrics["n_residues"] = n_res

        lig_coords = np.load(sample_dir / "ligand_coords.npy").astype(np.float32)
        metrics["ligand_atoms"] = int(lig_coords.shape[0])
        if not np.isfinite(lig_coords).all() or lig_coords.shape[0] == 0:
            reasons.append("bad_ligand_coords")
        if cfg["max_ligand_atoms"] > 0 and lig_coords.shape[0] > cfg["max_ligand_atoms"]:
            reasons.append("ligand_too_large")

        symbols = parse_sdf_symbols(sample_dir / "ligand.sdf")
        metrics["sdf_atoms"] = int(len(symbols))
        metrics["has_unmapped_ligand_elements"] = bool(any(s not in MAPPED_LIGAND_ELEMENTS for s in symbols))
        metrics["all_unmapped_ligand_elements"] = bool(symbols and not any(s in MAPPED_LIGAND_ELEMENTS for s in symbols))
        if not symbols:
            reasons.append("empty_sdf_symbols")
        if cfg["exclude_all_unmapped"] and metrics["all_unmapped_ligand_elements"]:
            reasons.append("all_unmapped_ligand_elements")
        if cfg["exclude_any_unmapped"] and metrics["has_unmapped_ligand_elements"]:
            reasons.append("has_unmapped_ligand_elements")

        ca_apo, node_mask = align_backbones(sample_dir, n_res)
        valid_residues = int(node_mask.sum())
        valid_fraction = float(valid_residues / max(n_res, 1))
        metrics["valid_residues"] = valid_residues
        metrics["valid_fraction"] = valid_fraction
        if valid_residues < cfg["min_valid_residues"]:
            reasons.append("too_few_valid_residues")
        if valid_fraction < cfg["min_valid_fraction"]:
            reasons.append("low_valid_fraction")

        torsion_apo = _load_torsions(sample_dir / "torsion_apo.npz", n_res)
        torsion_holo = _load_torsions(sample_dir / "torsion_holo.npz", n_res)
        chi1_valid = torsion_holo["chi_mask"][:, 0].astype(bool) & node_mask
        metrics["chi1_valid_residues"] = int(chi1_valid.sum())
        if int(chi1_valid.sum()) < cfg["min_chi1_valid"]:
            reasons.append("too_few_chi1_valid")

        apo_chi1 = torsion_apo["angles"][:, 3]
        holo_chi1 = torsion_holo["angles"][:, 3]
        switch = chi1_valid & (rotamer_labels(apo_chi1) != rotamer_labels(holo_chi1))
        ca_contact = chi1_valid & ca_contact_mask(ca_apo, lig_coords, cfg["ca_contact_threshold"])
        contact_switch = ca_contact & switch
        metrics["switch_residues"] = int(switch.sum())
        metrics["ca_contact_residues"] = int(ca_contact.sum())
        metrics["ca_contact_switch_residues"] = int(contact_switch.sum())
        if int(switch.sum()) < cfg["min_switch"]:
            reasons.append("too_few_switch_residues")
        if int(ca_contact.sum()) < cfg["min_ca_contact"]:
            reasons.append("too_few_ca_contact")
        if int(contact_switch.sum()) < cfg["min_ca_contact_switch"]:
            reasons.append("too_few_ca_contact_switch")

    except Exception as exc:
        reasons.append(f"exception:{type(exc).__name__}")
        metrics["error"] = str(exc)[:240]

    metrics["keep"] = not reasons
    metrics["reasons"] = reasons
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Create strict Stage-1 ligand-causality screening lists.")
    parser.add_argument("--data_dir", default="processed_data/triplets")
    parser.add_argument("--sample_ids_file", required=True)
    parser.add_argument("--output_prefix", required=True)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--ca_contact_threshold", type=float, default=8.0)
    parser.add_argument("--min_valid_residues", type=int, default=10)
    parser.add_argument("--min_valid_fraction", type=float, default=0.5)
    parser.add_argument("--min_chi1_valid", type=int, default=1)
    parser.add_argument("--min_switch", type=int, default=1)
    parser.add_argument("--min_ca_contact", type=int, default=1)
    parser.add_argument("--min_ca_contact_switch", type=int, default=1)
    parser.add_argument("--max_ligand_atoms", type=int, default=128,
                        help="0 disables the large-ligand filter")
    parser.add_argument("--exclude_all_unmapped", action="store_true", default=True)
    parser.add_argument("--allow_all_unmapped", dest="exclude_all_unmapped", action="store_false")
    parser.add_argument("--exclude_any_unmapped", action="store_true",
                        help="Stricter option; excludes ligands containing any unsupported element.")
    args = parser.parse_args()

    data_dir = Path(args.data_dir).resolve()
    sample_ids = load_sample_ids(Path(args.sample_ids_file))
    if args.limit is not None:
        sample_ids = sample_ids[: args.limit]

    cfg = {
        "ca_contact_threshold": args.ca_contact_threshold,
        "min_valid_residues": args.min_valid_residues,
        "min_valid_fraction": args.min_valid_fraction,
        "min_chi1_valid": args.min_chi1_valid,
        "min_switch": args.min_switch,
        "min_ca_contact": args.min_ca_contact,
        "min_ca_contact_switch": args.min_ca_contact_switch,
        "max_ligand_atoms": args.max_ligand_atoms,
        "exclude_all_unmapped": args.exclude_all_unmapped,
        "exclude_any_unmapped": args.exclude_any_unmapped,
    }

    tasks = [(sample_id, data_dir, cfg) for sample_id in sample_ids]
    rows: List[Dict] = []
    if args.workers <= 1:
        for task in tqdm(tasks, desc="Filtering samples"):
            rows.append(evaluate_sample(task))
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            futures = [pool.submit(evaluate_sample, task) for task in tasks]
            for fut in tqdm(as_completed(futures), total=len(futures), desc="Filtering samples"):
                rows.append(fut.result())

    sample_order = {sample_id: i for i, sample_id in enumerate(sample_ids)}
    rows.sort(key=lambda row: sample_order.get(row["sample_id"], len(sample_order)))
    kept = [row["sample_id"] for row in rows if row["keep"]]
    rejected = [row for row in rows if not row["keep"]]
    reason_counts = Counter(reason for row in rejected for reason in row["reasons"])

    output_prefix = Path(args.output_prefix)
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    kept_path = output_prefix.with_suffix(".txt")
    rejected_path = output_prefix.with_name(output_prefix.name + "_rejected.json")
    summary_path = output_prefix.with_name(output_prefix.name + "_summary.json")

    kept_path.write_text("\n".join(kept) + ("\n" if kept else ""))
    rejected_path.write_text(json.dumps(rejected, indent=2, ensure_ascii=False))
    summary = {
        "input_sample_ids_file": str(Path(args.sample_ids_file).resolve()),
        "output_kept_file": str(kept_path.resolve()),
        "n_input": len(sample_ids),
        "n_kept": len(kept),
        "n_rejected": len(rejected),
        "keep_fraction": len(kept) / max(len(sample_ids), 1),
        "criteria": cfg,
        "reason_counts": dict(reason_counts.most_common()),
    }
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False))

    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
