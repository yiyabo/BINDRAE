#!/bin/bash
#SBATCH --job-name=s1v2_post
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:A100:4
#SBATCH --cpus-per-task=8
#SBATCH --mem=220G
#SBATCH --time=08:00:00
#SBATCH --output=logs/slurm/stage1v2_posterior_%j.out
#SBATCH --error=logs/slurm/stage1v2_posterior_%j.err

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
mkdir -p logs/slurm logs/stage1v2 checkpoints/stage1v2

source /mnt/inaisfs/data/home/zhaozc_criait/miniconda3/etc/profile.d/conda.sh
conda activate BINDRAE

NPROC_PER_NODE="${NPROC_PER_NODE:-${SLURM_GPUS_ON_NODE:-4}}"
BATCH_SIZE="${BATCH_SIZE:-8}"
MAX_EPOCHS="${MAX_EPOCHS:-20}"
PATIENCE="${PATIENCE:-5}"
NUM_WORKERS="${NUM_WORKERS:-4}"
TRAIN_MAX_SAMPLES="${TRAIN_MAX_SAMPLES:-0}"
VAL_MAX_SAMPLES="${VAL_MAX_SAMPLES:-0}"
LR="${LR:-3e-4}"
CONTACT_BCE_WEIGHT="${CONTACT_BCE_WEIGHT:-1.0}"
APPROACH_BCE_WEIGHT="${APPROACH_BCE_WEIGHT:-0.5}"
RELEASE_BCE_WEIGHT="${RELEASE_BCE_WEIGHT:-0.5}"
SWITCH_BCE_WEIGHT="${SWITCH_BCE_WEIGHT:-1.0}"
CONFIDENCE_BCE_WEIGHT="${CONFIDENCE_BCE_WEIGHT:-0.3}"
DIST_MAE_WEIGHT="${DIST_MAE_WEIGHT:-0.1}"
DELTA_MAE_WEIGHT="${DELTA_MAE_WEIGHT:-0.2}"
C_S="${C_S:-256}"
HIDDEN_DIM="${HIDDEN_DIM:-256}"
NUM_LAYERS="${NUM_LAYERS:-3}"
TAG_PREFIX="${TAG_PREFIX:-stage1v2_posterior}"
DATA_DIR="${DATA_DIR:-processed_data/triplets}"
TRAIN_LABEL_DIR="${TRAIN_LABEL_DIR:-logs/stage1v2_teacher_posteriors/holo_truth_train}"
VAL_LABEL_DIR="${VAL_LABEL_DIR:-logs/stage1v2_teacher_posteriors/holo_truth_val}"
TRAIN_SPLIT="${TRAIN_SPLIT:-train}"
VAL_SPLIT="${VAL_SPLIT:-val}"
TRAIN_VALID_SAMPLES_FILE="${TRAIN_VALID_SAMPLES_FILE:-}"
VAL_VALID_SAMPLES_FILE="${VAL_VALID_SAMPLES_FILE:-}"
EVAL_COUNTERFACTUALS="${EVAL_COUNTERFACTUALS:-nolig,shuffled,translated}"
CLASS_BALANCED_HEADS="${CLASS_BALANCED_HEADS:-switch,approach,release}"
SELECTION_METRIC="${SELECTION_METRIC:-posterior_score}"

TAG="${TAG_PREFIX}_bs${BATCH_SIZE}_cs${C_S}_h${HIDDEN_DIM}_$(date +%Y%m%d_%H%M%S)"
SAVE_DIR="checkpoints/stage1v2/${TAG}"
LOG_DIR="logs/stage1v2/${TAG}"

ARGS=(
  scripts/train_stage1v2_posterior.py
  --data_dir "$DATA_DIR"
  --train_label_dir "$TRAIN_LABEL_DIR"
  --val_label_dir "$VAL_LABEL_DIR"
  --train_split "$TRAIN_SPLIT"
  --val_split "$VAL_SPLIT"
  --batch_size "$BATCH_SIZE"
  --num_workers "$NUM_WORKERS"
  --train_max_samples "$TRAIN_MAX_SAMPLES"
  --val_max_samples "$VAL_MAX_SAMPLES"
  --lr "$LR"
  --max_epochs "$MAX_EPOCHS"
  --patience "$PATIENCE"
  --contact_bce_weight "$CONTACT_BCE_WEIGHT"
  --approach_bce_weight "$APPROACH_BCE_WEIGHT"
  --release_bce_weight "$RELEASE_BCE_WEIGHT"
  --switch_bce_weight "$SWITCH_BCE_WEIGHT"
  --confidence_bce_weight "$CONFIDENCE_BCE_WEIGHT"
  --dist_mae_weight "$DIST_MAE_WEIGHT"
  --delta_mae_weight "$DELTA_MAE_WEIGHT"
  --c_s "$C_S"
  --hidden_dim "$HIDDEN_DIM"
  --num_layers "$NUM_LAYERS"
  --eval_counterfactuals "$EVAL_COUNTERFACTUALS"
  --class_balanced_heads "$CLASS_BALANCED_HEADS"
  --selection_metric "$SELECTION_METRIC"
  --save_dir "$SAVE_DIR"
  --log_dir "$LOG_DIR"
  --device cuda
  --distributed
)

if [[ -n "$TRAIN_VALID_SAMPLES_FILE" ]]; then
  ARGS+=(--train_valid_samples_file "$TRAIN_VALID_SAMPLES_FILE")
fi
if [[ -n "$VAL_VALID_SAMPLES_FILE" ]]; then
  ARGS+=(--val_valid_samples_file "$VAL_VALID_SAMPLES_FILE")
fi
if [[ "${USE_LATENT_HEAD:-0}" == "1" ]]; then
  ARGS+=(--use_latent_head)
fi
if [[ "${NO_AMP:-0}" == "1" ]]; then
  ARGS+=(--no_amp)
fi

echo "=============================================="
echo "Stage-1-v2 posterior student (multi-GPU DDP)"
echo "=============================================="
echo "Job ID:          ${SLURM_JOB_ID:-NA}"
echo "Node:            ${SLURM_NODELIST:-NA}"
echo "Tag:             $TAG"
echo "Data dir:        $DATA_DIR"
echo "Train labels:    $TRAIN_LABEL_DIR"
echo "Val labels:      $VAL_LABEL_DIR"
echo "Train/val split: $TRAIN_SPLIT / $VAL_SPLIT"
echo "Save dir:        $SAVE_DIR"
echo "Log dir:         $LOG_DIR"
echo "GPUs:            $NPROC_PER_NODE"
echo "Batch/GPU:       $BATCH_SIZE"
echo "Train max:       $TRAIN_MAX_SAMPLES"
echo "Val max:         $VAL_MAX_SAMPLES"
echo "Counterfactuals: $EVAL_COUNTERFACTUALS"
echo "Balanced heads:  $CLASS_BALANCED_HEADS"
echo "Selection metric:$SELECTION_METRIC"
echo "Loss weights:    contact=$CONTACT_BCE_WEIGHT approach=$APPROACH_BCE_WEIGHT release=$RELEASE_BCE_WEIGHT switch=$SWITCH_BCE_WEIGHT conf=$CONFIDENCE_BCE_WEIGHT dist=$DIST_MAE_WEIGHT delta=$DELTA_MAE_WEIGHT"
echo "Start:           $(date)"
echo "=============================================="

torchrun --nproc_per_node="$NPROC_PER_NODE" "${ARGS[@]}"

echo "completed: $(date)"
