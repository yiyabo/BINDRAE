#!/usr/bin/env python3
"""Diagnose Stage-2 loss components that create non-finite gradients."""

import argparse
import json
import sys
from pathlib import Path

import torch

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from src.stage2.training.config import TrainingConfig
from src.stage2.training.trainer import Stage2Trainer


LOSS_KEYS = (
    "total",
    "fm_chi",
    "fm_rigid",
    "bg",
    "smooth",
    "clash",
    "pep",
    "contact",
    "prior",
    "interaction_prior",
    "end",
)


def parse_args():
    parser = argparse.ArgumentParser(description="Stage-2 gradient component diagnostic")
    parser.add_argument("--data_dir", default="processed_data/triplets")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--valid_samples_file", default=None)
    parser.add_argument("--val_samples_file", default=None)
    parser.add_argument("--stage1_ckpt", default="checkpoints/stage1_best.pt")
    parser.add_argument("--no_stage1_prior", action="store_true")
    parser.add_argument("--stage1_prior_mode", default="stage1")
    parser.add_argument("--no_stage1_rigid_prior", action="store_true")
    parser.add_argument("--interaction_prior_ckpt", default=None)
    parser.add_argument("--w_interaction_prior", type=float, default=0.0)
    parser.add_argument("--interaction_prior_min_score", type=float, default=0.0)
    parser.add_argument("--interaction_prior_temperature", type=float, default=1.0)
    parser.add_argument("--interaction_prior_contact_dist", type=float, default=4.5)
    parser.add_argument("--interaction_prior_contact_tau", type=float, default=0.75)
    parser.add_argument("--interaction_prior_t_mid", type=float, default=0.3)
    parser.add_argument("--w_prior", type=float, default=0.0)
    parser.add_argument("--t_mid", type=float, default=0.5)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--grad_clip", type=float, default=0.3)
    parser.add_argument("--warmup_steps", type=int, default=0)
    parser.add_argument("--n_integration_steps", type=int, default=2)
    parser.add_argument("--n_geom_steps", type=int, default=3)
    parser.add_argument("--geom_loss_every_n_steps", type=int, default=1)
    parser.add_argument("--val_t", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--no_mixed_precision", action="store_true")
    parser.add_argument("--amp_dtype", default="bf16", choices=("auto", "bf16", "fp16"))
    parser.add_argument("--loss_keys", default=",".join(LOSS_KEYS))
    parser.add_argument("--output_json", default=None)
    return parser.parse_args()


def bad_grad_summary(model, limit=16):
    bad = []
    max_abs = 0.0
    max_name = None
    for name, param in model.named_parameters():
        if param.grad is None:
            continue
        grad = param.grad.detach()
        finite = torch.isfinite(grad)
        if not finite.all():
            bad.append(name)
            if len(bad) >= limit:
                break
        finite_grad = grad[finite]
        if finite_grad.numel() > 0:
            value = float(finite_grad.abs().max().item())
            if value > max_abs:
                max_abs = value
                max_name = name
    return bad, max_name, max_abs


def main():
    args = parse_args()
    config = TrainingConfig(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        valid_samples_file=args.valid_samples_file,
        val_samples_file=args.val_samples_file,
        lr=args.lr,
        max_epochs=1,
        grad_clip=args.grad_clip,
        warmup_steps=args.warmup_steps,
        seed=args.seed,
        val_t=args.val_t,
        stage1_ckpt=args.stage1_ckpt,
        use_stage1_prior=not args.no_stage1_prior,
        stage1_prior_mode=args.stage1_prior_mode,
        use_stage1_rigid_prior=not args.no_stage1_rigid_prior,
        interaction_prior_ckpt=args.interaction_prior_ckpt,
        w_interaction_prior=args.w_interaction_prior,
        interaction_prior_min_score=args.interaction_prior_min_score,
        interaction_prior_temperature=args.interaction_prior_temperature,
        interaction_prior_contact_dist=args.interaction_prior_contact_dist,
        interaction_prior_contact_tau=args.interaction_prior_contact_tau,
        interaction_prior_t_mid=args.interaction_prior_t_mid,
        w_prior=args.w_prior,
        t_mid=args.t_mid,
        n_integration_steps=args.n_integration_steps,
        n_geom_steps=args.n_geom_steps,
        geom_loss_every_n_steps=args.geom_loss_every_n_steps,
        save_dir="/tmp/stage2_grad_diag_ckpt",
        log_dir="/tmp/stage2_grad_diag_logs",
        device=args.device,
        mixed_precision=not args.no_mixed_precision,
        amp_dtype=args.amp_dtype,
        distributed=False,
    )

    trainer = Stage2Trainer(config)
    trainer.model.train()
    batch = next(iter(trainer.train_loader))
    batch = trainer._batch_to_device(batch)
    t = torch.full((batch.esm.shape[0],), float(args.val_t), device=trainer.device)
    keys = [k.strip() for k in args.loss_keys.split(",") if k.strip()]

    results = []
    for key in keys:
        trainer.optimizer.zero_grad(set_to_none=True)
        try:
            losses = trainer.compute_losses(batch, t)
            loss = losses[key]
            loss_value = float(loss.detach().float().item()) if loss.numel() == 1 else float("nan")
            if not torch.isfinite(loss).all():
                record = {"loss_key": key, "loss_value": loss_value, "status": "nonfinite_loss"}
            elif not loss.requires_grad:
                record = {"loss_key": key, "loss_value": loss_value, "status": "no_grad"}
            else:
                loss.backward()
                bad, max_name, max_abs = bad_grad_summary(trainer.model)
                record = {
                    "loss_key": key,
                    "loss_value": loss_value,
                    "status": "bad_grad" if bad else "ok",
                    "bad_grad_params": bad,
                    "max_finite_grad_param": max_name,
                    "max_finite_grad_abs": max_abs,
                }
        except Exception as exc:
            record = {"loss_key": key, "status": "exception", "error": repr(exc)}
        results.append(record)
        print(json.dumps(record, ensure_ascii=False), flush=True)

    if args.output_json:
        path = Path(args.output_json)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
