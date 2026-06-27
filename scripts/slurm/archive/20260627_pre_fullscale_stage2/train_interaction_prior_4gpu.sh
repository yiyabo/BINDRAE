#!/bin/bash
#SBATCH --job-name=s1_inter_prior
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:A100:4
#SBATCH --cpus-per-task=8
#SBATCH --mem=220G
#SBATCH --time=04:00:00
#SBATCH --output=logs/slurm/stage1_interaction_prior_%j.out
#SBATCH --error=logs/slurm/stage1_interaction_prior_%j.err

set -euo pipefail
unset LD_PRELOAD
unset PROXYCHAINS_CONF_FILE
unset PROXYCHAINS_QUIET_MODE

ROOT=/mnt/inaisfs/data/home/zhaozc_criait/XinxiangWang/BINDRAE
ENV_PREFIX=/mnt/inaisfs/data/home/zhaozc_criait/miniconda3/envs/BINDRAE

export PATH=/data/soft/slurm/24.11.4/bin:$ENV_PREFIX/bin:$PATH
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=4
export DDP_TIMEOUT="${DDP_TIMEOUT:-7200}"

cd "$ROOT"
mkdir -p logs/slurm logs/stage1 checkpoints/stage1

source /mnt/inaisfs/data/home/zhaozc_criait/miniconda3/etc/profile.d/conda.sh
conda activate BINDRAE

NPROC_PER_NODE="${NPROC_PER_NODE:-4}"
BATCH_SIZE="${BATCH_SIZE:-96}"
MAX_EPOCHS="${MAX_EPOCHS:-10}"
PATIENCE="${PATIENCE:-4}"
NUM_WORKERS="${NUM_WORKERS:-4}"
TRAIN_MAX_SAMPLES="${TRAIN_MAX_SAMPLES:-12000}"
VAL_MAX_SAMPLES="${VAL_MAX_SAMPLES:-1200}"
SUBSET_SEED="${SUBSET_SEED:-20260617}"
MAX_LOCAL_RES="${MAX_LOCAL_RES:-192}"
MAX_N_RES="${MAX_N_RES:-1600}"
LR="${LR:-3e-4}"
HIDDEN_DIM="${HIDDEN_DIM:-192}"
NUM_LAYERS="${NUM_LAYERS:-3}"
RESIDUE_CHUNK="${RESIDUE_CHUNK:-64}"
LAMBDA_BCE="${LAMBDA_BCE:-1.0}"
LAMBDA_SOFT="${LAMBDA_SOFT:-0.3}"
LAMBDA_DECOY_ZERO="${LAMBDA_DECOY_ZERO:-0.2}"
LAMBDA_CONTRASTIVE="${LAMBDA_CONTRASTIVE:-0.8}"
DECOY_MARGIN="${DECOY_MARGIN:-1.0}"
TARGET_CONTACT_DIST="${TARGET_CONTACT_DIST:-4.5}"
CROP_MODE="${CROP_MODE:-pocket_only}"
GAIN_MARGIN="${GAIN_MARGIN:-0.5}"
TAG_PREFIX="${TAG_PREFIX:-interaction_prior}"

TAG="${TAG_PREFIX}_${CROP_MODE}_bs${BATCH_SIZE}_local${MAX_LOCAL_RES}_tc${TARGET_CONTACT_DIST}_$(date +%Y%m%d_%H%M%S)"
SAVE_DIR="checkpoints/stage1/${TAG}"
LOG_DIR="logs/stage1/${TAG}"

TRAIN_METRIC_ARGS=()
if [[ "${COLLECT_TRAIN_METRICS:-0}" == "1" ]]; then
  TRAIN_METRIC_ARGS+=(--collect_train_metrics)
fi

echo "=============================================="
echo "Stage-1 explicit interaction prior (4-GPU DDP)"
echo "=============================================="
echo "Job ID:          ${SLURM_JOB_ID:-NA}"
echo "Node:            ${SLURM_NODELIST:-NA}"
echo "Tag:             $TAG"
echo "Save dir:        $SAVE_DIR"
echo "Log dir:         $LOG_DIR"
echo "GPUs:            $NPROC_PER_NODE"
echo "Batch/GPU:       $BATCH_SIZE (global $((BATCH_SIZE * NPROC_PER_NODE)))"
echo "Train subset:    $TRAIN_MAX_SAMPLES"
echo "Val subset:      $VAL_MAX_SAMPLES"
echo "Max local res:   $MAX_LOCAL_RES"
echo "Max epochs:      $MAX_EPOCHS"
echo "LR:              $LR"
echo "Hidden/layers:   $HIDDEN_DIM / $NUM_LAYERS"
echo "Loss weights:    bce=$LAMBDA_BCE soft=$LAMBDA_SOFT decoy_zero=$LAMBDA_DECOY_ZERO contrastive=$LAMBDA_CONTRASTIVE"
echo "Decoy margin:    $DECOY_MARGIN"
echo "Target contact:  $TARGET_CONTACT_DIST A"
echo "Crop mode:       $CROP_MODE"
echo "Gain margin:     $GAIN_MARGIN A"
echo "Start:           $(date)"
echo "=============================================="

torchrun --nproc_per_node="$NPROC_PER_NODE" scripts/train_interaction_prior.py \
  --data_dir processed_data/triplets \
  --val_samples_file processed_data/triplets/val_valid.txt \
  --sample_metadata_file sample_metadata.json \
  --batch_size "$BATCH_SIZE" \
  --train_max_samples "$TRAIN_MAX_SAMPLES" \
  --val_max_samples "$VAL_MAX_SAMPLES" \
  --subset_seed "$SUBSET_SEED" \
  --max_n_res "$MAX_N_RES" \
  --max_local_res "$MAX_LOCAL_RES" \
  --num_workers "$NUM_WORKERS" \
  --hidden_dim "$HIDDEN_DIM" \
  --num_layers "$NUM_LAYERS" \
  --lr "$LR" \
  --max_epochs "$MAX_EPOCHS" \
  --patience "$PATIENCE" \
  --residue_chunk "$RESIDUE_CHUNK" \
  --target_contact_dist "$TARGET_CONTACT_DIST" \
  --gain_margin "$GAIN_MARGIN" \
  --crop_mode "$CROP_MODE" \
  --lambda_bce "$LAMBDA_BCE" \
  --lambda_soft "$LAMBDA_SOFT" \
  --lambda_decoy_zero "$LAMBDA_DECOY_ZERO" \
  --lambda_contrastive "$LAMBDA_CONTRASTIVE" \
  --decoy_margin "$DECOY_MARGIN" \
  --save_dir "$SAVE_DIR" \
  --log_dir "$LOG_DIR" \
  --device cuda \
  --distributed \
  "${TRAIN_METRIC_ARGS[@]}"

echo ""
echo "=============================================="
echo "Interaction prior completed: $(date)"
echo "Checkpoints: $SAVE_DIR"
echo "Metrics: $LOG_DIR/metrics.jsonl"
echo "=============================================="
