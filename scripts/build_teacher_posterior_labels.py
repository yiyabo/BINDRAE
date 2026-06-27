#!/usr/bin/env python3
"""Export per-residue teacher posterior labels for Stage-1-v2.

The first supported teacher is `holo_truth`: labels are computed from apo and
holo sidechain-ligand distances in the existing known-pose triplet data.  The
schema is intentionally teacher-agnostic so later DynamicBind/FlowDock/Boltz
adapters can write the same fields.
"""

import argparse
import json
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict

import numpy as np
import torch

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))
flash_ipa_path = project_root / "vendor" / "flash_ipa" / "src"
if flash_ipa_path.exists():
    sys.path.insert(0, str(flash_ipa_path))

from flash_ipa.rigid import Rigid, Rotation  # noqa: E402

from src.stage1.models.fk_openfold import create_openfold_fk, reorder_torsions_to_openfold  # noqa: E402
from src.stage1.models.interaction_prior import min_sidechain_ligand_dist, sidechain_atom_mask  # noqa: E402
from src.stage1.posterior_v2.schema import BOOL_FIELDS, FLOAT_FIELDS, SCHEMA_VERSION  # noqa: E402
from src.stage2.datasets.dataset_stage2 import create_stage2_dataloader  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build Stage-1-v2 teacher posterior labels")
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--split", choices=("train", "val", "test"), default="val")
    parser.add_argument("--valid_samples_file", default=None)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--teacher_source", choices=("holo_truth",), default="holo_truth")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--max_batches", type=int, default=0)
    parser.add_argument("--max_samples", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--contact_dist", type=float, default=4.5)
    parser.add_argument("--contact_tau", type=float, default=0.75)
    parser.add_argument("--active_delta", type=float, default=0.75)
    parser.add_argument("--pocket_threshold", type=float, default=0.3)
    parser.add_argument("--max_saved_dist", type=float, default=50.0)
    return parser.parse_args()


def batch_to_device(batch, device: torch.device):
    for name, value in vars(batch).items():
        if torch.is_tensor(value):
            setattr(batch, name, value.to(device))
    return batch


def build_rigids_from_backbone(n_coord, ca_coord, c_coord, mask, eps: float = 1e-6) -> Rigid:
    device = ca_coord.device
    dtype = ca_coord.dtype
    default_e1 = torch.tensor([1.0, 0.0, 0.0], device=device, dtype=dtype)
    default_e2 = torch.tensor([0.0, 1.0, 0.0], device=device, dtype=dtype)
    default_e3 = torch.tensor([0.0, 0.0, 1.0], device=device, dtype=dtype)

    e1 = c_coord - ca_coord
    e1_norm = torch.norm(e1, dim=-1, keepdim=True)
    e1 = torch.where(e1_norm > eps, e1 / e1_norm.clamp_min(eps), default_e1.expand_as(e1))

    u = n_coord - ca_coord
    proj = (u * e1).sum(dim=-1, keepdim=True) * e1
    e2 = u - proj
    e2_norm = torch.norm(e2, dim=-1, keepdim=True)
    e2 = torch.where(e2_norm > eps, e2 / e2_norm.clamp_min(eps), default_e2.expand_as(e2))

    e3 = torch.cross(e1, e2, dim=-1)
    e3_norm = torch.norm(e3, dim=-1, keepdim=True)
    e3 = torch.where(e3_norm > eps, e3 / e3_norm.clamp_min(eps), default_e3.expand_as(e3))
    e2 = torch.cross(e3, e1, dim=-1)

    rot = torch.stack([e1, e2, e3], dim=-1)
    valid = mask.bool().unsqueeze(-1).unsqueeze(-1)
    eye = torch.eye(3, device=device, dtype=dtype).view(1, 1, 3, 3)
    rot = torch.where(valid, rot, eye.expand_as(rot))
    trans = torch.where(mask.bool().unsqueeze(-1), ca_coord, torch.zeros_like(ca_coord))
    return Rigid(rots=Rotation(rot_mats=rot), trans=trans)


def torsions_to_atom14(fk_module, rigids: Rigid, aatype: torch.Tensor, torsions: torch.Tensor):
    torsions_sincos = torch.stack([torch.sin(torsions), torch.cos(torsions)], dim=-1)
    torsions_sincos = reorder_torsions_to_openfold(torsions_sincos)
    out = fk_module(torsions_sincos.float(), rigids, aatype.clamp(0, 20))
    return out["atom14_pos"], out["atom14_mask"].bool()


def safe_sample_id(sample_id: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", sample_id)


def tensor_to_np(x: torch.Tensor) -> np.ndarray:
    return x.detach().cpu().numpy()


def build_labels_for_batch(batch, fk_module, args: argparse.Namespace) -> Dict[str, torch.Tensor]:
    rigids_apo = build_rigids_from_backbone(batch.N_apo, batch.Ca_apo, batch.C_apo, batch.node_mask)
    rigids_holo = build_rigids_from_backbone(batch.N_holo, batch.Ca_holo, batch.C_holo, batch.node_mask)

    apo_atom14, apo_atom14_mask = torsions_to_atom14(fk_module, rigids_apo, batch.aatype, batch.torsion_apo)
    holo_atom14, holo_atom14_mask = torsions_to_atom14(fk_module, rigids_holo, batch.aatype, batch.torsion_holo)
    apo_sc_mask = sidechain_atom_mask(apo_atom14_mask, batch.node_mask)
    holo_sc_mask = sidechain_atom_mask(holo_atom14_mask, batch.node_mask)

    d_apo = min_sidechain_ligand_dist(
        apo_atom14,
        apo_sc_mask,
        batch.lig_points,
        batch.lig_mask,
        batch.node_mask,
        residue_chunk=64,
    )
    d_holo = min_sidechain_ligand_dist(
        holo_atom14,
        holo_sc_mask,
        batch.lig_points,
        batch.lig_mask,
        batch.node_mask,
        residue_chunk=64,
    )

    valid = batch.node_mask.bool() & torch.isfinite(d_apo) & torch.isfinite(d_holo)
    d_apo_c = d_apo.clamp(max=float(args.max_saved_dist))
    d_holo_c = d_holo.clamp(max=float(args.max_saved_dist))
    signed_delta = d_holo_c - d_apo_c

    contact_apo = valid & (d_apo <= float(args.contact_dist))
    contact_holo = valid & (d_holo <= float(args.contact_dist))
    approach = valid & ((d_apo - d_holo) > float(args.active_delta))
    release = valid & ((d_holo - d_apo) > float(args.active_delta))
    formed_contact = valid & (~contact_apo) & contact_holo
    released_contact = valid & contact_apo & (~contact_holo)
    stable_contact = valid & contact_apo & contact_holo
    stable_noncontact = valid & (~contact_apo) & (~contact_holo)
    switch = approach | release | formed_contact | released_contact
    active = switch | (torch.abs(d_holo_c - d_apo_c) > float(args.active_delta))
    pocket = valid & (
        (batch.w_res >= float(args.pocket_threshold))
        | contact_apo
        | contact_holo
        | active
    )

    contact_prob = torch.sigmoid((float(args.contact_dist) - d_holo_c) / max(float(args.contact_tau), 1e-6))
    approach_prob = approach.float()
    release_prob = release.float()
    switch_prob = switch.float()
    confidence = pocket.float()

    return {
        "valid_mask": valid,
        "pocket_mask": pocket,
        "apo_min_dist": d_apo_c,
        "teacher_min_dist": d_holo_c,
        "signed_delta_dist": signed_delta,
        "contact_prob": contact_prob,
        "approach_prob": approach_prob,
        "release_prob": release_prob,
        "switch_prob": switch_prob,
        "confidence": confidence,
        "contact_apo": contact_apo,
        "contact_teacher": contact_holo,
        "approach": approach,
        "release": release,
        "formed_contact": formed_contact,
        "released_contact": released_contact,
        "stable_contact": stable_contact,
        "stable_noncontact": stable_noncontact,
        "active": active,
    }


def class_counts(labels: Dict[str, torch.Tensor], index: int, n_res: int) -> Dict[str, int]:
    keys = (
        "valid_mask",
        "pocket_mask",
        "active",
        "approach",
        "release",
        "formed_contact",
        "released_contact",
        "stable_contact",
        "stable_noncontact",
    )
    return {key: int(labels[key][index, :n_res].sum().item()) for key in keys}


def save_sample(output_dir: Path, batch, labels: Dict[str, torch.Tensor], index: int, args: argparse.Namespace):
    sample_id = batch.pdb_ids[index]
    n_res = int(batch.n_residues[index])
    filename = f"{safe_sample_id(sample_id)}.npz"
    path = output_dir / filename

    arrays = {
        "schema_version": np.array(SCHEMA_VERSION),
        "teacher_source": np.array(args.teacher_source),
        "sample_id": np.array(sample_id),
        "n_residues": np.array(n_res, dtype=np.int32),
        "aatype": tensor_to_np(batch.aatype[index, :n_res]).astype(np.int16),
        "w_res": tensor_to_np(batch.w_res[index, :n_res]).astype(np.float32),
    }
    for key in FLOAT_FIELDS:
        if key == "w_res":
            continue
        arrays[key] = tensor_to_np(labels[key][index, :n_res]).astype(np.float32)
    for key in BOOL_FIELDS:
        arrays[key] = tensor_to_np(labels[key][index, :n_res]).astype(np.bool_)
    np.savez_compressed(path, **arrays)

    return {
        "sample_id": sample_id,
        "path": str(path),
        "n_residues": n_res,
        "counts": class_counts(labels, index, n_res),
    }


def main() -> None:
    args = parse_args()
    device = torch.device(args.device if args.device != "cuda" or torch.cuda.is_available() else "cpu")
    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = project_root / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    loader = create_stage2_dataloader(
        args.data_dir,
        split=args.split,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        valid_samples_file=args.valid_samples_file,
    )
    fk_module = create_openfold_fk().to(device)
    fk_module.eval()

    records = []
    totals: Dict[str, int] = {}
    n_seen = 0
    with torch.no_grad():
        for batch_idx, batch in enumerate(loader):
            if args.max_batches > 0 and batch_idx >= args.max_batches:
                break
            batch = batch_to_device(batch, device)
            labels = build_labels_for_batch(batch, fk_module, args)
            for i in range(len(batch.pdb_ids)):
                if args.max_samples > 0 and n_seen >= args.max_samples:
                    break
                record = save_sample(output_dir, batch, labels, i, args)
                records.append(record)
                for key, value in record["counts"].items():
                    totals[key] = totals.get(key, 0) + int(value)
                n_seen += 1
            if args.max_samples > 0 and n_seen >= args.max_samples:
                break

    manifest = {
        "schema_version": SCHEMA_VERSION,
        "teacher_source": args.teacher_source,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "command": " ".join([os.path.basename(sys.argv[0]), *sys.argv[1:]]),
        "args": vars(args),
        "num_samples": len(records),
        "counts": totals,
        "records": records,
    }
    manifest_path = output_dir / "manifest.json"
    with manifest_path.open("w") as f:
        json.dump(manifest, f, indent=2)

    print(f"Wrote {len(records)} teacher posterior label files")
    print(f"Manifest: {manifest_path}")
    print(json.dumps({"counts": totals}, indent=2))


if __name__ == "__main__":
    main()
