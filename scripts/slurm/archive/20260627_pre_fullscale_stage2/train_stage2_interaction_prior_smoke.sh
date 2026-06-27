#!/bin/bash
# Stage-2 smoke test with explicit interaction-prior guidance.

#SBATCH --job-name=s2_iprior_smoke
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:A100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=04:00:00
#SBATCH --output=logs/slurm/stage2_iprior_smoke_%j.out
#SBATCH --error=logs/slurm/stage2_iprior_smoke_%j.err

set -euo pipefail
unset LD_PRELOAD
unset PROXYCHAINS_CONF_FILE
unset PROXYCHAINS_QUIET_MODE

ROOT=/mnt/inaisfs/data/home/zhaozc_criait/XinxiangWang/BINDRAE
ENV_PREFIX=/mnt/inaisfs/data/home/zhaozc_criait/miniconda3/envs/BINDRAE

export PATH=/data/soft/slurm/24.11.4/bin:$ENV_PREFIX/bin:$PATH
export PYTHONUNBUFFERED=1
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export OMP_NUM_THREADS=4

cd "$ROOT"
mkdir -p logs/slurm logs/stage2 checkpoints/stage2

source /mnt/inaisfs/data/home/zhaozc_criait/miniconda3/etc/profile.d/conda.sh
conda activate BINDRAE

INTERACTION_PRIOR_CKPT="${INTERACTION_PRIOR_CKPT:-checkpoints/stage1/interaction_prior_pocket_only_bs96_local192_tc4.5_20260617_112147/best_model.pt}"
W_INTERACTION_PRIOR="${W_INTERACTION_PRIOR:-0.05}"
INTERACTION_PRIOR_MIN_SCORE="${INTERACTION_PRIOR_MIN_SCORE:-0.2}"
INTERACTION_PRIOR_T_MID="${INTERACTION_PRIOR_T_MID:-0.3}"
MAX_EPOCHS="${MAX_EPOCHS:-2}"
BATCH_SIZE="${BATCH_SIZE:-1}"
VAL_T="${VAL_T:-0.5}"
TAG_PREFIX="${TAG_PREFIX:-stage2_iprior_smoke}"
NO_MIXED_PRECISION="${NO_MIXED_PRECISION:-0}"

EXTRA_ARGS=()
if [[ "$NO_MIXED_PRECISION" == "1" ]]; then
  EXTRA_ARGS+=(--no_mixed_precision)
fi

TAG="${TAG_PREFIX}_w${W_INTERACTION_PRIOR}_$(date +%Y%m%d_%H%M%S)"
SAVE_DIR="checkpoints/stage2/${TAG}"
LOG_DIR="logs/stage2/${TAG}"
TRAIN_TINY="stage2_smoke_train.txt"
VAL_TINY="stage2_smoke_val.txt"

echo "=============================================="
echo "Stage-2 interaction-prior smoke"
echo "=============================================="
echo "Job ID:              ${SLURM_JOB_ID:-NA}"
echo "Node:                ${SLURM_NODELIST:-NA}"
echo "Tag:                 $TAG"
echo "Interaction prior:   $INTERACTION_PRIOR_CKPT"
echo "w_interaction_prior: $W_INTERACTION_PRIOR"
echo "min_score:           $INTERACTION_PRIOR_MIN_SCORE"
echo "t_mid:               $INTERACTION_PRIOR_T_MID"
echo "max_epochs:          $MAX_EPOCHS"
echo "batch:               $BATCH_SIZE"
echo "no_mixed_precision:  $NO_MIXED_PRECISION"
echo "Save dir:            $SAVE_DIR"
echo "Log dir:             $LOG_DIR"
echo "Start:               $(date)"
echo "=============================================="

python scripts/train_stage2.py \
  --data_dir processed_data/triplets \
  --batch_size "$BATCH_SIZE" \
  --num_workers 0 \
  --max_epochs "$MAX_EPOCHS" \
  --lr 2e-5 \
  --grad_clip 0.3 \
  --accum_steps 1 \
  --warmup_steps 0 \
  --seed 42 \
  --val_t "$VAL_T" \
  --no_stage1_prior \
  --w_prior 0.0 \
  --interaction_prior_ckpt "$INTERACTION_PRIOR_CKPT" \
  --w_interaction_prior "$W_INTERACTION_PRIOR" \
  --interaction_prior_min_score "$INTERACTION_PRIOR_MIN_SCORE" \
  --interaction_prior_t_mid "$INTERACTION_PRIOR_T_MID" \
  --n_integration_steps 2 \
  --n_geom_steps 3 \
  --geom_loss_every_n_steps 1 \
  --valid_samples_file "$TRAIN_TINY" \
  --val_samples_file "$VAL_TINY" \
  --save_dir "$SAVE_DIR" \
  --log_dir "$LOG_DIR" \
  --device cuda \
  "${EXTRA_ARGS[@]}"

echo ""
echo "=============================================="
echo "Stage-2 interaction-prior smoke completed: $(date)"
echo "Metrics: $LOG_DIR/metrics.jsonl"
echo "=============================================="
