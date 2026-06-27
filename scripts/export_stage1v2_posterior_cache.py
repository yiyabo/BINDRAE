#!/usr/bin/env python3
"""Export Stage-1-v2 posterior student predictions as per-sample cache files."""

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
from torch.utils.data import DataLoader, Subset

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))
flash_ipa_path = project_root / "vendor" / "flash_ipa" / "src"
if flash_ipa_path.exists():
    sys.path.insert(0, str(flash_ipa_path))

from src.stage1.posterior_v2.dataset import (  # noqa: E402
    TeacherPosteriorDataset,
    collate_teacher_posterior_batch,
)
from src.stage1.posterior_v2.inference import (  # noqa: E402
    batch_to_device,
    load_stage1v2_posterior_checkpoint,
    predict_posterior,
)


CACHE_SCHEMA_VERSION = "bindrae_stage1v2_student_posterior_cache_v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export Stage-1-v2 posterior cache")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--label_dir", required=True)
    parser.add_argument("--split", default="train", choices=("train", "val", "test"))
    parser.add_argument("--valid_samples_file", default=None)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--max_samples", type=int, default=0)
    parser.add_argument("--subset_seed", type=int, default=20260622)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--include_teacher_labels", action="store_true")
    parser.add_argument("--no_amp", action="store_true")
    return parser.parse_args()


def _safe_sample_id(sample_id: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", sample_id)


def _subset(dataset, max_samples: int, seed: int):
    if max_samples <= 0 or max_samples >= len(dataset):
        return dataset
    g = torch.Generator()
    g.manual_seed(int(seed))
    indices = torch.randperm(len(dataset), generator=g)[: int(max_samples)].tolist()
    return Subset(dataset, indices)


def _tensor_slice_np(tensor: torch.Tensor, index: int, n_res: int, dtype=None) -> np.ndarray:
    arr = tensor[index, :n_res].detach().cpu().numpy()
    return arr.astype(dtype) if dtype is not None else arr


def save_cache_sample(
    output_dir: Path,
    batch,
    pred: Dict[str, torch.Tensor],
    index: int,
    checkpoint_path: Path,
    checkpoint: Dict,
    include_teacher_labels: bool,
) -> Dict:
    sample_id = batch.pdb_ids[index]
    n_res = int(batch.n_residues[index])
    path = output_dir / f"{_safe_sample_id(sample_id)}.npz"

    arrays = {
        "schema_version": np.array(CACHE_SCHEMA_VERSION),
        "sample_id": np.array(sample_id),
        "n_residues": np.array(n_res, dtype=np.int32),
        "source_checkpoint": np.array(str(checkpoint_path)),
        "checkpoint_epoch": np.array(int(checkpoint.get("epoch", -1)), dtype=np.int32),
        "teacher_source": np.array(batch.teacher_source[index]),
        "teacher_label_path": np.array(batch.teacher_label_paths[index]),
        "aatype": _tensor_slice_np(batch.aatype, index, n_res, np.int16),
        "node_mask": _tensor_slice_np(batch.node_mask, index, n_res, np.bool_),
        "w_res": _tensor_slice_np(batch.w_res, index, n_res, np.float32),
        "contact_prob": _tensor_slice_np(pred["contact_prob"], index, n_res, np.float32),
        "active_prob": _tensor_slice_np(pred["switch_prob"], index, n_res, np.float32),
        "switch_prob": _tensor_slice_np(pred["switch_prob"], index, n_res, np.float32),
        "approach_prob": _tensor_slice_np(pred["approach_prob"], index, n_res, np.float32),
        "release_prob": _tensor_slice_np(pred["release_prob"], index, n_res, np.float32),
        "confidence": _tensor_slice_np(pred["confidence"], index, n_res, np.float32),
        "teacher_min_dist_pred": _tensor_slice_np(pred["teacher_min_dist"], index, n_res, np.float32),
        "signed_delta_dist_pred": _tensor_slice_np(pred["signed_delta_dist"], index, n_res, np.float32),
        "contact_logit": _tensor_slice_np(pred["contact_logit"], index, n_res, np.float32),
        "active_logit": _tensor_slice_np(pred["switch_logit"], index, n_res, np.float32),
        "approach_logit": _tensor_slice_np(pred["approach_logit"], index, n_res, np.float32),
        "release_logit": _tensor_slice_np(pred["release_logit"], index, n_res, np.float32),
        "confidence_logit": _tensor_slice_np(pred["confidence_logit"], index, n_res, np.float32),
    }
    if "z_post" in pred:
        arrays["z_post"] = _tensor_slice_np(pred["z_post"], index, n_res, np.float32)

    if include_teacher_labels:
        for key, value in batch.teacher_float.items():
            arrays[f"teacher_{key}"] = _tensor_slice_np(value, index, n_res, np.float32)
        for key, value in batch.teacher_bool.items():
            arrays[f"teacher_{key}"] = _tensor_slice_np(value, index, n_res, np.bool_)

    np.savez_compressed(path, **arrays)
    return {
        "sample_id": sample_id,
        "path": str(path),
        "n_residues": n_res,
        "teacher_source": batch.teacher_source[index],
        "teacher_label_path": batch.teacher_label_paths[index],
    }


def main() -> None:
    args = parse_args()
    device = torch.device(args.device if args.device != "cuda" or torch.cuda.is_available() else "cpu")
    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = project_root / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_path = Path(args.checkpoint)
    model, checkpoint = load_stage1v2_posterior_checkpoint(checkpoint_path, device=device)
    dataset = TeacherPosteriorDataset(
        args.data_dir,
        args.label_dir,
        split=args.split,
        valid_samples_file=args.valid_samples_file,
    )
    dataset = _subset(dataset, args.max_samples, args.subset_seed)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_teacher_posterior_batch,
        pin_memory=True,
    )

    records = []
    total_residues = 0
    with torch.no_grad():
        for batch in loader:
            batch = batch_to_device(batch, device)
            with torch.cuda.amp.autocast(enabled=(not args.no_amp) and device.type == "cuda", dtype=torch.bfloat16):
                pred = predict_posterior(model, batch, mode="real")
            for i in range(len(batch.pdb_ids)):
                record = save_cache_sample(
                    output_dir,
                    batch,
                    pred,
                    i,
                    checkpoint_path,
                    checkpoint,
                    include_teacher_labels=bool(args.include_teacher_labels),
                )
                records.append(record)
                total_residues += int(record["n_residues"])

    manifest = {
        "schema_version": CACHE_SCHEMA_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "command": " ".join([os.path.basename(sys.argv[0]), *sys.argv[1:]]),
        "args": vars(args),
        "checkpoint": str(checkpoint_path),
        "checkpoint_epoch": checkpoint.get("epoch"),
        "checkpoint_metrics": checkpoint.get("metrics", {}),
        "split": args.split,
        "num_samples": len(records),
        "total_residues": total_residues,
        "include_teacher_labels": bool(args.include_teacher_labels),
        "records": records,
    }
    manifest_path = output_dir / "manifest.json"
    with manifest_path.open("w") as f:
        json.dump(manifest, f, indent=2)

    print(f"Wrote {len(records)} Stage-1-v2 posterior cache files")
    print(f"Manifest: {manifest_path}")
    print(json.dumps({"num_samples": len(records), "total_residues": total_residues}, indent=2))


if __name__ == "__main__":
    main()
