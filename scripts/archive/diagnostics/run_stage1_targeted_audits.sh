#!/bin/bash
set -euo pipefail

ROOT="${1:-$(pwd)}"

cd "$ROOT"

source /mnt/inaisfs/data/home/zhaozc_criait/miniconda3/etc/profile.d/conda.sh
conda activate BINDRAE

COMMON_ARGS=(
  --data_dir processed_data/triplets
  --valid_samples_file /mnt/inaisfs/data/home/zhaozc_criait/XinxiangWang/BINDRAE/processed_data/triplets/train_valid.txt
  --val_samples_file /mnt/inaisfs/data/home/zhaozc_criait/XinxiangWang/BINDRAE/processed_data/triplets/val_valid.txt
  --sample_metadata_file sample_metadata.json
  --batch_size 2
  --num_workers 2
  --device cuda
  --compute_slow_metrics
  --enable_dual_mask_audit
)

python scripts/audit_stage1_checkpoint.py \
  --checkpoint checkpoints/stage1/opt8e2_20260411_122707/best_model.pt \
  --output_dir logs/stage1/opt8e2_20260411_122707/audit_stage2_utility_best \
  "${COMMON_ARGS[@]}"

python scripts/audit_stage1_checkpoint.py \
  --checkpoint checkpoints/stage1/opt8e2_20260411_122707/epoch_018.pt \
  --output_dir logs/stage1/opt8e2_20260411_122707/audit_stage2_utility_epoch018 \
  "${COMMON_ARGS[@]}"

python scripts/audit_stage1_checkpoint.py \
  --checkpoint checkpoints/stage1/e6b_slowramp_from_opt8e6a_20260415_024951_20260415_055419/best_model.pt \
  --output_dir logs/stage1/e6b_slowramp_from_opt8e6a_20260415_024951_20260415_055419/audit_stage2_utility_best \
  "${COMMON_ARGS[@]}"

python scripts/audit_stage1_checkpoint.py \
  --checkpoint checkpoints/stage1/e6b_slowramp_from_opt8e6a_20260415_024951_20260415_055419/epoch_010.pt \
  --output_dir logs/stage1/e6b_slowramp_from_opt8e6a_20260415_024951_20260415_055419/audit_stage2_utility_epoch010 \
  "${COMMON_ARGS[@]}"

python scripts/audit_stage1_checkpoint.py \
  --checkpoint checkpoints/stage1/e6b_slowramp_from_opt8e6a_20260415_024951_20260415_055419/epoch_011.pt \
  --output_dir logs/stage1/e6b_slowramp_from_opt8e6a_20260415_024951_20260415_055419/audit_stage2_utility_epoch011 \
  "${COMMON_ARGS[@]}"
