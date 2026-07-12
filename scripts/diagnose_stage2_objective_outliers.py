#!/usr/bin/env python3
"""Identify validation systems that dominate a Stage-2 training objective."""

import argparse
import copy
import json
import sys
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.stage2.training.trainer import Stage2Trainer


METRIC_KEYS = (
    "total_no_repa",
    "objective_pep",
    "objective_clash",
    "objective_contact",
    "pep",
    "pep_interior",
    "clash",
    "clash_interior",
    "contact",
    "phase_residual_norm_mean",
    "phase_residual_norm_max",
    "time_warp_tau_abs_mean",
    "time_warp_tau_abs_max",
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--valid_samples_file", default=None)
    parser.add_argument("--max_batches", type=int, default=None)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--top_k", type=int, default=32)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def summarize(records):
    summary = {}
    for key in METRIC_KEYS:
        values = np.asarray([record[key] for record in records], dtype=np.float64)
        finite = values[np.isfinite(values)]
        if finite.size == 0:
            continue
        summary[key] = {
            "mean": float(finite.mean()),
            "p50": float(np.percentile(finite, 50)),
            "p95": float(np.percentile(finite, 95)),
            "p99": float(np.percentile(finite, 99)),
            "max": float(finite.max()),
        }
    return summary


def main():
    args = parse_args()
    if int(args.top_k) <= 0:
        raise ValueError("--top_k must be > 0")

    device = torch.device(args.device)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    config = copy.deepcopy(checkpoint["config"])
    config.distributed = False
    config.device = str(device)
    config.batch_size = 1
    config.val_batch_size = 1
    config.num_workers = int(args.num_workers)
    config.length_bucketed_train = False
    config.resume_from = None
    config.auto_resume = False
    config.init_from_checkpoint = args.checkpoint
    config.max_epochs = 0
    config.save_dir = str(Path(args.output).parent / ".objective_outlier_tmp_ckpt")
    config.log_dir = str(Path(args.output).parent / ".objective_outlier_tmp_log")
    if args.valid_samples_file:
        config.val_samples_file = args.valid_samples_file
        config.trust_prechecked_samples = True

    trainer = Stage2Trainer(config)
    trainer.model.eval()
    records = []
    with torch.no_grad():
        for batch_idx, batch in enumerate(
            tqdm(trainer.val_loader, desc="Objective outliers", ncols=120)
        ):
            if args.max_batches is not None and batch_idx >= int(args.max_batches):
                break
            if len(batch.pdb_ids) != 1:
                raise ValueError("Objective outlier diagnostics require val_batch_size=1")
            sample_id = str(batch.pdb_ids[0])
            n_residues = int(batch.n_residues[0])
            batch = trainer._batch_to_device(batch)
            if config.val_t is None:
                t = torch.full((1,), 0.5, device=device)
            else:
                t = torch.full((1,), float(config.val_t), device=device)
            losses = trainer.compute_losses(batch, t, force_geom=True)
            trainer._check_finite_losses(losses, f"objective outlier {sample_id}")
            record = {
                "sample_id": sample_id,
                "n_residues": n_residues,
                **{key: float(losses[key].item()) for key in METRIC_KEYS},
            }
            record["objective_pep_fraction"] = (
                record["objective_pep"] / max(record["total_no_repa"], 1e-12)
            )
            records.append(record)

    records.sort(key=lambda item: item["objective_pep"], reverse=True)
    output = {
        "checkpoint": args.checkpoint,
        "valid_samples_file": config.val_samples_file,
        "samples": len(records),
        "summary": summarize(records),
        "top_objective_pep": records[: int(args.top_k)],
        "records": records,
    }
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "samples": len(records),
        "output": str(output_path),
        "top_objective_pep": records[: min(5, len(records))],
    }, indent=2))


if __name__ == "__main__":
    main()
