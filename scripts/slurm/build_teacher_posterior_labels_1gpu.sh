#!/bin/bash
#SBATCH --job-name=teacher_posterior
#SBATCH --output=logs/slurm/teacher_posterior_%j.out
#SBATCH --error=logs/slurm/teacher_posterior_%j.err
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:A100:1
#SBATCH --cpus-per-task=4
#SBATCH --time=04:00:00

set -euo pipefail

ROOT="/mnt/inaisfs/data/home/zhaozc_criait/XinxiangWang/BINDRAE"
cd "$ROOT"

source ~/miniconda3/etc/profile.d/conda.sh
conda activate BINDRAE

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"

SPLIT="${SPLIT:-val}"
DATA_DIR="${DATA_DIR:-processed_data/triplets}"
VALID_SAMPLES_FILE="${VALID_SAMPLES_FILE:-}"
TAG="${TAG:-holo_truth_${SPLIT}_$(date +%Y%m%d_%H%M%S)}"
OUTPUT_DIR="${OUTPUT_DIR:-logs/stage1v2_teacher_posteriors/${TAG}}"
BATCH_SIZE="${BATCH_SIZE:-1}"
NUM_WORKERS="${NUM_WORKERS:-2}"
MAX_BATCHES="${MAX_BATCHES:-0}"
MAX_SAMPLES="${MAX_SAMPLES:-0}"
CONTACT_DIST="${CONTACT_DIST:-4.5}"
CONTACT_TAU="${CONTACT_TAU:-0.75}"
ACTIVE_DELTA="${ACTIVE_DELTA:-0.75}"
POCKET_THRESHOLD="${POCKET_THRESHOLD:-0.3}"
DEVICE="${DEVICE:-cuda}"

mkdir -p logs/slurm "$OUTPUT_DIR"

echo "Stage-1-v2 teacher posterior label export"
echo "  split:              $SPLIT"
echo "  data_dir:           $DATA_DIR"
echo "  valid_samples_file: ${VALID_SAMPLES_FILE:-OFF}"
echo "  output_dir:         $OUTPUT_DIR"
echo "  batch_size:         $BATCH_SIZE"
echo "  max_batches:        $MAX_BATCHES"
echo "  max_samples:        $MAX_SAMPLES"
echo "  device:             $DEVICE"
echo "  started:            $(date)"

ARGS=(
  scripts/build_teacher_posterior_labels.py
  --data_dir "$DATA_DIR"
  --split "$SPLIT"
  --output_dir "$OUTPUT_DIR"
  --batch_size "$BATCH_SIZE"
  --num_workers "$NUM_WORKERS"
  --max_batches "$MAX_BATCHES"
  --max_samples "$MAX_SAMPLES"
  --device "$DEVICE"
  --contact_dist "$CONTACT_DIST"
  --contact_tau "$CONTACT_TAU"
  --active_delta "$ACTIVE_DELTA"
  --pocket_threshold "$POCKET_THRESHOLD"
)

if [[ -n "$VALID_SAMPLES_FILE" ]]; then
  ARGS+=(--valid_samples_file "$VALID_SAMPLES_FILE")
fi

python "${ARGS[@]}"

echo "completed: $(date)"
