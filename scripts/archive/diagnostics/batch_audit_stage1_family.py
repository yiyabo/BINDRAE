#!/usr/bin/env python3
"""
Batch-audit a Stage-1 checkpoint family and rank checkpoints by selected metrics.

This is a thin wrapper around scripts/audit_stage1_checkpoint.py so that we can
retrospectively mine epoch checkpoints for Stage2-utility-aligned candidates.
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description='Batch audit a Stage-1 checkpoint family')
    parser.add_argument('--checkpoint_dir', type=str, required=True,
                        help='Directory containing best_model.pt / epoch_*.pt')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Directory to write per-checkpoint audits and summary ranking')
    parser.add_argument('--data_dir', type=str, default=None)
    parser.add_argument('--valid_samples_file', type=str, default=None)
    parser.add_argument('--val_samples_file', type=str, default=None)
    parser.add_argument('--sample_metadata_file', type=str, default=None)
    parser.add_argument('--batch_size', type=int, default=None)
    parser.add_argument('--num_workers', type=int, default=2)
    parser.add_argument('--max_n_res', type=int, default=None)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--selection_metric', type=str, default=None)
    parser.add_argument('--compute_slow_metrics', action='store_true')
    parser.add_argument('--enable_dual_mask_audit', action='store_true')
    parser.add_argument('--include_latest', action='store_true',
                        help='Also audit latest_model.pt if present')
    parser.add_argument('--top_k', type=int, default=10,
                        help='Number of top checkpoints to print by each ranking metric')
    parser.add_argument('--python_executable', type=str, default=None,
                        help='Python executable used to invoke audit_stage1_checkpoint.py')
    return parser.parse_args()


def build_audit_cmd(args, checkpoint_path: Path, output_dir: Path):
    python_exec = args.python_executable or sys.executable
    cmd = [
        python_exec,
        str(Path(__file__).resolve().parent / 'audit_stage1_checkpoint.py'),
        '--checkpoint', str(checkpoint_path),
        '--output_dir', str(output_dir),
        '--device', args.device,
        '--num_workers', str(args.num_workers),
    ]
    if args.data_dir is not None:
        cmd += ['--data_dir', args.data_dir]
    if args.valid_samples_file is not None:
        cmd += ['--valid_samples_file', args.valid_samples_file]
    if args.val_samples_file is not None:
        cmd += ['--val_samples_file', args.val_samples_file]
    if args.sample_metadata_file is not None:
        cmd += ['--sample_metadata_file', args.sample_metadata_file]
    if args.batch_size is not None:
        cmd += ['--batch_size', str(args.batch_size)]
    if args.max_n_res is not None:
        cmd += ['--max_n_res', str(args.max_n_res)]
    if args.selection_metric is not None:
        cmd += ['--selection_metric', args.selection_metric]
    if args.compute_slow_metrics:
        cmd.append('--compute_slow_metrics')
    if args.enable_dual_mask_audit:
        cmd.append('--enable_dual_mask_audit')
    return cmd


def iter_checkpoints(checkpoint_dir: Path, include_latest: bool):
    seen = set()
    for name in ['best_model.pt']:
        p = checkpoint_dir / name
        if p.exists():
            seen.add(p.name)
            yield p
    for p in sorted(checkpoint_dir.glob('epoch_*.pt')):
        seen.add(p.name)
        yield p
    if include_latest:
        p = checkpoint_dir / 'latest_model.pt'
        if p.exists() and p.name not in seen:
            yield p


def safe_float(value):
    try:
        x = float(value)
    except Exception:
        return None
    if x != x:
        return None
    return x


def main():
    args = parse_args()
    checkpoint_dir = Path(args.checkpoint_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    checkpoints = list(iter_checkpoints(checkpoint_dir, args.include_latest))
    if not checkpoints:
        raise FileNotFoundError(f'No checkpoints found under {checkpoint_dir}')

    rows = []
    for checkpoint_path in checkpoints:
        stem = checkpoint_path.stem
        ckpt_out_dir = output_dir / stem
        ckpt_out_dir.mkdir(parents=True, exist_ok=True)
        cmd = build_audit_cmd(args, checkpoint_path, ckpt_out_dir)
        print(f'\n=== Auditing {checkpoint_path.name} ===')
        subprocess.run(cmd, check=True)

        metrics_path = ckpt_out_dir / 'audit_metrics.jsonl'
        if not metrics_path.exists():
            raise FileNotFoundError(f'Missing audit output: {metrics_path}')
        lines = [line for line in metrics_path.read_text(encoding='utf-8').splitlines() if line.strip()]
        if not lines:
            raise RuntimeError(f'Empty audit output: {metrics_path}')
        row = json.loads(lines[-1])
        row['checkpoint_name'] = checkpoint_path.name
        row['checkpoint_path'] = str(checkpoint_path)
        rows.append(row)

    summary_path = output_dir / 'family_audit_summary.json'
    summary_path.write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding='utf-8')

    ranking_metrics = [
        ('ligand_facing_chi1_acc', True),
        ('ligand_facing_chi12_acc', True),
        ('ligand_facing_contact_f1', True),
        ('ligand_facing_irmsd', False),
        ('holo_pocket_chi1_acc', True),
        ('holo_pocket_irmsd', False),
        ('pocket_chi1_acc', True),
        ('pocket_irmsd', False),
    ]

    print(f'\nSaved family summary to: {summary_path}')
    for metric_name, descending in ranking_metrics:
        ranked = [row for row in rows if safe_float(row.get(metric_name)) is not None]
        if not ranked:
            continue
        ranked.sort(key=lambda row: safe_float(row.get(metric_name)), reverse=descending)
        print(f'\n=== Top {min(args.top_k, len(ranked))} by {metric_name} ({"desc" if descending else "asc"}) ===')
        for row in ranked[:args.top_k]:
            print({
                'checkpoint_name': row['checkpoint_name'],
                metric_name: row.get(metric_name),
                'pocket_chi1_acc': row.get('pocket_chi1_acc'),
                'holo_pocket_chi1_acc': row.get('holo_pocket_chi1_acc'),
                'ligand_facing_chi1_acc': row.get('ligand_facing_chi1_acc'),
            })


if __name__ == '__main__':
    main()
