#!/usr/bin/env python3
"""
Audit a Stage-1 checkpoint with the current validation pipeline.

This script is intended for post-hoc checkpoint analysis (E2 / E3 / E5),
not for training. It reuses Stage1Trainer.validate() so that secondary
metrics and dual-mask audit stay consistent with the current codebase.
"""

import argparse
import json
import os
import sys
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Dict

import torch

# Project root
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from src.stage1.training.config import TrainingConfig
from src.stage1.training.trainer import Stage1Trainer


FK_BUFFER_KEYS = frozenset({
    'fk_module.default_frames',
    'fk_module.restype_atom14_positions',
    'fk_module.restype_atom14_to_group',
    'fk_module.restype_atom14_mask',
})


def _config_to_dict(config_obj: Any) -> Dict[str, Any]:
    if config_obj is None:
        return {}
    if isinstance(config_obj, TrainingConfig):
        return asdict(config_obj)
    if is_dataclass(config_obj):
        return asdict(config_obj)
    if isinstance(config_obj, dict):
        return dict(config_obj)
    return {
        key: getattr(config_obj, key)
        for key in dir(config_obj)
        if not key.startswith('_') and not callable(getattr(config_obj, key))
    }


def _make_audit_config(ckpt: Dict[str, Any], args) -> TrainingConfig:
    ckpt_cfg = _config_to_dict(ckpt.get('config'))
    config = TrainingConfig(**{k: v for k, v in ckpt_cfg.items() if k in TrainingConfig.__dataclass_fields__})

    config.distributed = False
    config.resume_from = None
    config.device = args.device
    config.data_dir = args.data_dir or config.data_dir
    config.valid_samples_file = args.valid_samples_file if args.valid_samples_file is not None else config.valid_samples_file
    config.val_samples_file = args.val_samples_file if args.val_samples_file is not None else config.val_samples_file
    config.sample_metadata_file = args.sample_metadata_file if args.sample_metadata_file is not None else config.sample_metadata_file
    config.batch_size = args.batch_size or config.batch_size
    config.num_workers = args.num_workers
    config.max_n_res = args.max_n_res if args.max_n_res is not None else config.max_n_res
    config.compute_slow_metrics = args.compute_slow_metrics
    config.enable_dual_mask_audit = args.enable_dual_mask_audit
    config.selection_metric = args.selection_metric or config.selection_metric

    output_dir = Path(args.output_dir) if args.output_dir else (Path(args.checkpoint).resolve().parent / 'audit')
    output_dir.mkdir(parents=True, exist_ok=True)
    config.log_dir = str(output_dir)
    config.save_dir = str(output_dir)
    config.metrics_filename = args.metrics_filename
    config.audit_filename = args.audit_filename

    return config


def _load_checkpoint_into_trainer(trainer: Stage1Trainer, ckpt: Dict[str, Any], checkpoint_path: str):
    state_dict = ckpt.get('model_state_dict', ckpt)
    result = trainer.model.load_state_dict(state_dict, strict=False)
    unexpected = [k for k in result.unexpected_keys if k not in FK_BUFFER_KEYS]
    missing = [k for k in result.missing_keys if k not in FK_BUFFER_KEYS]
    if unexpected or missing:
        raise RuntimeError(
            f"Checkpoint mismatch for {checkpoint_path} (ignoring FK buffers): "
            f"unexpected={unexpected}, missing={missing}"
        )


def _print_results(results: Dict[str, Any]):
    print("\n=== Validation Results ===")
    for key in sorted(results.keys()):
        value = results[key]
        if isinstance(value, float):
            if value != value:  # NaN
                print(f"{key}: NaN")
            else:
                print(f"{key}: {value:.6f}")
        else:
            print(f"{key}: {value}")


def main():
    parser = argparse.ArgumentParser(description='Audit a Stage-1 checkpoint with secondary metrics and dual-mask evaluation')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to Stage-1 checkpoint (best_model.pt / latest_model.pt / epoch_xxx.pt)')
    parser.add_argument('--data_dir', type=str, default=None, help='Override data_dir from checkpoint config')
    parser.add_argument('--valid_samples_file', type=str, default=None, help='Override training valid samples file')
    parser.add_argument('--val_samples_file', type=str, default=None, help='Override validation samples file')
    parser.add_argument('--sample_metadata_file', type=str, default=None, help='Override sample metadata file')
    parser.add_argument('--batch_size', type=int, default=None, help='Override batch size for audit')
    parser.add_argument('--num_workers', type=int, default=2, help='Validation DataLoader workers')
    parser.add_argument('--max_n_res', type=int, default=None, help='Optional max_n_res override')
    parser.add_argument('--device', type=str, default='cuda', help='Audit device')
    parser.add_argument('--selection_metric', type=str, default=None, help='Optional selection metric override for display')
    parser.add_argument('--output_dir', type=str, default=None, help='Audit output directory (default: <checkpoint_dir>/audit)')
    parser.add_argument('--metrics_filename', type=str, default='audit_metrics.jsonl', help='Metrics log file written into output_dir')
    parser.add_argument('--audit_filename', type=str, default='audit_val_results.json', help='Audit summary JSON written into output_dir')
    parser.add_argument('--compute_slow_metrics', action='store_true', help='Enable slower metrics (contact/clash/iRMSD)')
    parser.add_argument('--enable_dual_mask_audit', action='store_true', help='Enable apo/holo/ligand-facing subset audit')
    args = parser.parse_args()

    ckpt = torch.load(args.checkpoint, map_location=args.device, weights_only=False)
    config = _make_audit_config(ckpt, args)

    print("\n=== Stage-1 Checkpoint Audit ===")
    print(f"checkpoint: {args.checkpoint}")
    print(f"data_dir: {config.data_dir}")
    print(f"val_samples_file: {config.val_samples_file}")
    print(f"batch_size: {config.batch_size}")
    print(f"device: {config.device}")
    print(f"compute_slow_metrics: {config.compute_slow_metrics}")
    print(f"enable_dual_mask_audit: {config.enable_dual_mask_audit}")
    print(f"output_dir: {config.log_dir}")

    trainer = Stage1Trainer(config)
    _load_checkpoint_into_trainer(trainer, ckpt, args.checkpoint)

    if 'epoch' in ckpt:
        trainer.current_epoch = int(ckpt['epoch'])
    if 'global_step' in ckpt:
        trainer.global_step = int(ckpt['global_step'])

    results = trainer.validate()
    trainer._write_audit_snapshot({
        'checkpoint': str(Path(args.checkpoint).resolve()),
        'epoch': trainer.current_epoch,
        'global_step': trainer.global_step,
        'selection_metric_name': config.selection_metric,
        'val_metrics': results,
    })

    metrics_record_path = Path(config.log_dir) / config.metrics_filename
    with metrics_record_path.open('w', encoding='utf-8') as f:
        f.write(json.dumps({
            'checkpoint': str(Path(args.checkpoint).resolve()),
            'epoch': trainer.current_epoch,
            'global_step': trainer.global_step,
            **results,
        }, ensure_ascii=False) + '\n')

    _print_results(results)
    print(f"\nAudit summary saved to: {Path(config.log_dir) / config.audit_filename}")
    print(f"Audit metrics record saved to: {metrics_record_path}")


if __name__ == '__main__':
    main()
