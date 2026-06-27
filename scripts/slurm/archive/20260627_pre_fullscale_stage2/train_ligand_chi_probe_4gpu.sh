#!/bin/bash
#SBATCH --job-name=s1_ligchi_probe
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:A100:4
#SBATCH --cpus-per-task=8
#SBATCH --mem=256G
#SBATCH --time=06:00:00
#SBATCH --output=logs/slurm/stage1_ligchi_probe_%j.out
#SBATCH --error=logs/slurm/stage1_ligchi_probe_%j.err

set -e
unset LD_PRELOAD
unset PROXYCHAINS_CONF_FILE
unset PROXYCHAINS_QUIET_MODE

export PATH=/mnt/inaisfs/data/home/zhaozc_criait/miniconda3/bin:$PATH
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=4

source /mnt/inaisfs/data/home/zhaozc_criait/miniconda3/etc/profile.d/conda.sh
conda activate BINDRAE

cd /mnt/inaisfs/data/home/zhaozc_criait/XinxiangWang/BINDRAE
mkdir -p logs/slurm logs/stage1 checkpoints/stage1

BATCH_SIZE="${BATCH_SIZE:-100}"
MAX_EPOCHS="${MAX_EPOCHS:-12}"
NUM_WORKERS="${NUM_WORKERS:-4}"
TRAIN_MAX_SAMPLES="${TRAIN_MAX_SAMPLES:-12000}"
VAL_MAX_SAMPLES="${VAL_MAX_SAMPLES:-1000}"
TRAIN_SAMPLES_FILE="${TRAIN_SAMPLES_FILE:-}"
VAL_SAMPLES_FILE="${VAL_SAMPLES_FILE:-/mnt/inaisfs/data/home/zhaozc_criait/XinxiangWang/BINDRAE/processed_data/triplets/val_valid.txt}"
SUBSET_SEED="${SUBSET_SEED:-20260617}"
WARMUP_STEPS="${WARMUP_STEPS:-80}"
LR="${LR:-1e-4}"
LAMBDA_CONTACT="${LAMBDA_CONTACT:-1.0}"
LAMBDA_RANK="${LAMBDA_RANK:-0.2}"
RANK_MARGIN="${RANK_MARGIN:-0.02}"
RESIDUAL_SCALE="${RESIDUAL_SCALE:-0.5}"
STAGE1_CHECKPOINT="${STAGE1_CHECKPOINT:-}"
TAG_PREFIX="${TAG_PREFIX:-ligchi_probe}"

TAG="${TAG_PREFIX}_bs${BATCH_SIZE}_$(date +%Y%m%d_%H%M%S)"
SAVE_DIR="checkpoints/stage1/${TAG}"
LOG_DIR="logs/stage1/${TAG}"

EXTRA_ARGS=()
if [[ -n "${STAGE1_CHECKPOINT:-}" ]]; then
  EXTRA_ARGS+=(--stage1_checkpoint "$STAGE1_CHECKPOINT")
fi
if [[ -n "${TRAIN_SAMPLES_FILE:-}" ]]; then
  EXTRA_ARGS+=(--train_samples_file "$TRAIN_SAMPLES_FILE")
fi

echo "=============================================="
echo "Stage-1 Ligand-Causal Chi Probe (4-GPU DDP)"
echo "=============================================="
echo "Job ID:          $SLURM_JOB_ID"
echo "Node:            $SLURM_NODELIST"
echo "GPUs:            4x A100"
echo "Tag:             $TAG"
echo "Save dir:        $SAVE_DIR"
echo "Log dir:         $LOG_DIR"
echo "Stage1 ckpt:     ${STAGE1_CHECKPOINT:-<random init>}"
echo "Batch size:      $BATCH_SIZE per GPU (global $((BATCH_SIZE * 4)))"
echo "Train subset:    $TRAIN_MAX_SAMPLES"
echo "Val subset:      $VAL_MAX_SAMPLES"
echo "Train file:      ${TRAIN_SAMPLES_FILE:-<split default>}"
echo "Val file:        $VAL_SAMPLES_FILE"
echo "Max epochs:      $MAX_EPOCHS"
echo "Warmup steps:    $WARMUP_STEPS"
echo "LR:              $LR"
echo "Lambda contact:  $LAMBDA_CONTACT"
echo "Lambda rank:     $LAMBDA_RANK"
echo "Rank margin:     $RANK_MARGIN"
echo "Residual scale:  $RESIDUAL_SCALE"
echo "Start:           $(date)"
echo "=============================================="

torchrun --nproc_per_node=4 scripts/train_ligand_chi_probe.py \
  --data_dir processed_data/triplets \
  --val_samples_file "$VAL_SAMPLES_FILE" \
  --sample_metadata_file sample_metadata.json \
  --batch_size "$BATCH_SIZE" \
  --train_max_samples "$TRAIN_MAX_SAMPLES" \
  --val_max_samples "$VAL_MAX_SAMPLES" \
  --subset_seed "$SUBSET_SEED" \
  --max_n_res 1600 \
  --lr "$LR" \
  --max_epochs "$MAX_EPOCHS" \
  --warmup_steps "$WARMUP_STEPS" \
  --lambda_contact "$LAMBDA_CONTACT" \
  --lambda_rank "$LAMBDA_RANK" \
  --rank_margin "$RANK_MARGIN" \
  --residual_scale "$RESIDUAL_SCALE" \
  --save_dir "$SAVE_DIR" \
  --log_dir "$LOG_DIR" \
  --num_workers "$NUM_WORKERS" \
  --device cuda \
  --distributed \
  "${EXTRA_ARGS[@]}"

echo ""
echo "=============================================="
echo "Ligand-chi probe completed: $(date)"
echo "Checkpoints: $SAVE_DIR"
echo "Metrics: $LOG_DIR/metrics.jsonl"
echo "=============================================="
