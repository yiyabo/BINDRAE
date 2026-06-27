#!/usr/bin/env python3
"""Train the Stage-1-v2 teacher-distilled posterior student."""

import argparse
import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))
flash_ipa_path = project_root / "vendor" / "flash_ipa" / "src"
if flash_ipa_path.exists():
    sys.path.insert(0, str(flash_ipa_path))

from src.stage1.posterior_v2.trainer import PosteriorV2Trainer, PosteriorV2TrainingConfig


def parse_args():
    parser = argparse.ArgumentParser(description="Train Stage-1-v2 posterior student")
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--train_label_dir", required=True)
    parser.add_argument("--val_label_dir", required=True)
    parser.add_argument("--train_split", default="train")
    parser.add_argument("--val_split", default="val")
    parser.add_argument("--train_valid_samples_file", default=None)
    parser.add_argument("--val_valid_samples_file", default=None)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--train_max_samples", type=int, default=0)
    parser.add_argument("--val_max_samples", type=int, default=0)
    parser.add_argument("--subset_seed", type=int, default=20260622)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--max_epochs", type=int, default=20)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--save_dir", default="checkpoints/stage1v2/posterior")
    parser.add_argument("--log_dir", default="logs/stage1v2/posterior")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--distributed", action="store_true")
    parser.add_argument("--c_s", type=int, default=256)
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--num_layers", type=int, default=3)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--use_latent_head", action="store_true")
    parser.add_argument("--no_amp", action="store_true")
    parser.add_argument("--contact_bce_weight", type=float, default=1.0)
    parser.add_argument("--approach_bce_weight", type=float, default=0.5)
    parser.add_argument("--release_bce_weight", type=float, default=0.5)
    parser.add_argument("--switch_bce_weight", type=float, default=1.0)
    parser.add_argument("--confidence_bce_weight", type=float, default=0.3)
    parser.add_argument("--dist_mae_weight", type=float, default=0.1)
    parser.add_argument("--delta_mae_weight", type=float, default=0.2)
    parser.add_argument("--eval_counterfactuals", default="nolig,shuffled,translated")
    parser.add_argument("--class_balanced_heads", default="switch,approach,release")
    parser.add_argument("--selection_metric", default="posterior_score")
    return parser.parse_args()


def main():
    args = parse_args()
    config = PosteriorV2TrainingConfig(
        data_dir=args.data_dir,
        train_label_dir=args.train_label_dir,
        val_label_dir=args.val_label_dir,
        train_split=args.train_split,
        val_split=args.val_split,
        train_valid_samples_file=args.train_valid_samples_file,
        val_valid_samples_file=args.val_valid_samples_file,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        train_max_samples=args.train_max_samples,
        val_max_samples=args.val_max_samples,
        subset_seed=args.subset_seed,
        lr=args.lr,
        weight_decay=args.weight_decay,
        max_epochs=args.max_epochs,
        grad_clip=args.grad_clip,
        patience=args.patience,
        save_dir=args.save_dir,
        log_dir=args.log_dir,
        device=args.device,
        distributed=args.distributed,
        c_s=args.c_s,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        dropout=args.dropout,
        use_latent_head=args.use_latent_head,
        amp=not args.no_amp,
        contact_bce_weight=args.contact_bce_weight,
        approach_bce_weight=args.approach_bce_weight,
        release_bce_weight=args.release_bce_weight,
        switch_bce_weight=args.switch_bce_weight,
        confidence_bce_weight=args.confidence_bce_weight,
        dist_mae_weight=args.dist_mae_weight,
        delta_mae_weight=args.delta_mae_weight,
        eval_counterfactuals=args.eval_counterfactuals,
        class_balanced_heads=args.class_balanced_heads,
        selection_metric=args.selection_metric,
    )
    trainer = PosteriorV2Trainer(config)
    trainer.train()


if __name__ == "__main__":
    main()
