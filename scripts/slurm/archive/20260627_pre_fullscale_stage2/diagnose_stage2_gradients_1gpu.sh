#!/bin/bash
# Diagnose which Stage-2 loss component creates non-finite gradients.

#SBATCH --job-name=s2_grad_diag
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:A100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=60G
#SBATCH --time=01:00:00
#SBATCH --output=logs/slurm/stage2_grad_diag_%j.out
#SBATCH --error=logs/slurm/stage2_grad_diag_%j.err

set -euo pipefail
unset LD_PRELOAD
unset PROXYCHAINS_CONF_FILE
unset PROXYCHAINS_QUIET_MODE

ROOT=/mnt/inaisfs/data/home/zhaozc_criait/XinxiangWang/BINDRAE
ENV_PREFIX=/mnt/inaisfs/data/home/zhaozc_criait/miniconda3/envs/BINDRAE

export PATH=/data/soft/slurm/24.11.4/bin:$ENV_PREFIX/bin:$PATH
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=4

cd "$ROOT"
mkdir -p logs/slurm logs/stage2/grad_diag

source /mnt/inaisfs/data/home/zhaozc_criait/miniconda3/etc/profile.d/conda.sh
conda activate BINDRAE

INTERACTION_PRIOR_CKPT="${INTERACTION_PRIOR_CKPT:-checkpoints/stage1/interaction_prior_pocket_only_bs96_local192_tc4.5_20260617_112147/best_model.pt}"
W_INTERACTION_PRIOR="${W_INTERACTION_PRIOR:-0.05}"
NO_MIXED_PRECISION="${NO_MIXED_PRECISION:-0}"
AMP_DTYPE="${AMP_DTYPE:-bf16}"
TAG="stage2_grad_diag_${SLURM_JOB_ID:-manual}"

EXTRA_ARGS=()
if [[ "$NO_MIXED_PRECISION" == "1" ]]; then
  EXTRA_ARGS+=(--no_mixed_precision)
fi

echo "Stage-2 gradient diagnostic"
echo "Job ID: ${SLURM_JOB_ID:-NA}"
echo "w_interaction_prior: $W_INTERACTION_PRIOR"
echo "no_mixed_precision: $NO_MIXED_PRECISION"
echo "amp_dtype: $AMP_DTYPE"

python scripts/diagnose_stage2_gradients.py \
  --data_dir processed_data/triplets \
  --batch_size 1 \
  --num_workers 0 \
  --valid_samples_file stage2_smoke_train.txt \
  --val_samples_file stage2_smoke_val.txt \
  --no_stage1_prior \
  --w_prior 0.0 \
  --interaction_prior_ckpt "$INTERACTION_PRIOR_CKPT" \
  --w_interaction_prior "$W_INTERACTION_PRIOR" \
  --interaction_prior_min_score 0.2 \
  --interaction_prior_t_mid 0.3 \
  --n_integration_steps 2 \
  --n_geom_steps 3 \
  --geom_loss_every_n_steps 1 \
  --val_t 0.5 \
  --amp_dtype "$AMP_DTYPE" \
  --output_json "logs/stage2/grad_diag/${TAG}.json" \
  "${EXTRA_ARGS[@]}"
