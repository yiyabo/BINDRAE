#!/usr/bin/env python3
"""Measure objective-gradient conflict on the Stage-2 time-warp head."""

from __future__ import annotations

import argparse
import copy
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import torch

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))
flash_ipa_path = project_root / "vendor" / "flash_ipa" / "src"
if flash_ipa_path.exists():
    sys.path.insert(0, str(flash_ipa_path))

from src.stage2.training import Stage2Trainer  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--valid_samples_file", required=True)
    parser.add_argument("--val_samples_file", required=True)
    parser.add_argument("--phase_teacher_cache_dir", required=True)
    parser.add_argument("--max_batches", type=int, default=4)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", default=None)
    return parser.parse_args()


def flatten_gradients(
    loss: torch.Tensor,
    parameters: Iterable[torch.nn.Parameter],
    *,
    retain_graph: bool,
) -> Optional[torch.Tensor]:
    params = list(parameters)
    if not loss.requires_grad:
        return None
    gradients = torch.autograd.grad(
        loss,
        params,
        retain_graph=retain_graph,
        allow_unused=True,
    )
    flat = [
        gradient.detach().float().reshape(-1)
        for gradient in gradients
        if gradient is not None
    ]
    return torch.cat(flat) if flat else None


def cosine(left: Optional[torch.Tensor], right: Optional[torch.Tensor]) -> Optional[float]:
    if left is None or right is None:
        return None
    denom = left.norm() * right.norm()
    if float(denom) <= 0.0:
        return None
    return float(torch.dot(left, right) / denom)


def mean_or_none(values: List[float]) -> Optional[float]:
    return sum(values) / len(values) if values else None


def main() -> None:
    args = parse_args()
    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    config = copy.deepcopy(payload["config"])
    config.distributed = False
    config.device = args.device
    config.batch_size = int(args.batch_size)
    config.val_batch_size = int(args.batch_size)
    config.num_workers = 0
    config.prefetch_factor = 2
    config.length_bucketed_train = False
    config.valid_samples_file = args.valid_samples_file
    config.val_samples_file = args.val_samples_file
    config.trust_prechecked_samples = True
    config.phase_teacher_cache_dir = args.phase_teacher_cache_dir
    config.phase_teacher_missing_policy = "error"
    config.phase_teacher_head_only = False
    config.init_from_checkpoint = args.checkpoint
    config.resume_from = None
    config.auto_resume = False
    config.max_epochs = 1
    config.save_dir = "/tmp/bindrae_phase_gradient_diagnostic"
    config.log_dir = "/tmp/bindrae_phase_gradient_diagnostic"

    trainer = Stage2Trainer(config)
    trainer.model.eval()
    raw_model = trainer._raw_model
    parameters = [
        parameter
        for name, parameter in raw_model.named_parameters()
        if name.startswith("time_warp_head.") and parameter.requires_grad
    ]
    if not parameters:
        raise RuntimeError("No trainable time_warp_head parameters found")

    term_keys = (
        "phase_teacher",
        "pep",
        "contact",
        "clash",
        "bg",
        "phase_residual_magnitude",
        "phase_residual_temporal_smooth",
        "phase_residual_neighbor_smooth",
    )
    norms: Dict[str, List[float]] = defaultdict(list)
    cosines: Dict[str, List[float]] = defaultdict(list)
    batches = 0

    for batch in trainer.train_loader:
        if batches >= int(args.max_batches):
            break
        batch = trainer._batch_to_device(batch)
        t = torch.full((batch.esm.shape[0],), 0.5, device=trainer.device)
        losses = trainer.compute_losses(batch, t, force_geom=True)
        gradients = {}
        for key in term_keys:
            gradient = flatten_gradients(
                losses[key],
                parameters,
                retain_graph=True,
            )
            gradients[key] = gradient
            if gradient is not None:
                norm = float(gradient.norm())
                if math.isfinite(norm):
                    norms[key].append(norm)
        phase_gradient = gradients["phase_teacher"]
        for key in term_keys[1:]:
            value = cosine(phase_gradient, gradients[key])
            if value is not None and math.isfinite(value):
                cosines[key].append(value)
        batches += 1

    output = {
        "checkpoint": args.checkpoint,
        "valid_samples_file": args.valid_samples_file,
        "phase_teacher_cache_dir": args.phase_teacher_cache_dir,
        "batches": batches,
        "time_warp_parameters": sum(parameter.numel() for parameter in parameters),
        "gradient_norm_mean": {
            key: mean_or_none(norms[key]) for key in term_keys
        },
        "cosine_with_phase_teacher_mean": {
            key: mean_or_none(cosines[key]) for key in term_keys[1:]
        },
    }
    rendered = json.dumps(output, indent=2, sort_keys=True)
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(rendered)
        print(f"Saved: {output_path}")
    print(rendered)


if __name__ == "__main__":
    main()
