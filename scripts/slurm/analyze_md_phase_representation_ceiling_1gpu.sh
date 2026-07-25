#!/usr/bin/env bash
# Analyze synchronous/global/residue phase representation ceilings on MD paths.

#SBATCH --job-name=md_phase_ceiling
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --gres=gpu:A100:1
#SBATCH --time=04:00:00
#SBATCH --output=logs/slurm/md_phase_ceiling_%j.out
#SBATCH --error=logs/slurm/md_phase_ceiling_%j.err

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/mnt/inaisfs/data/home/zhaozc_criait/XinxiangWang/BINDRAE}"
cd "${PROJECT_ROOT}"

source /mnt/inaisfs/data/home/zhaozc_criait/miniconda3/etc/profile.d/conda.sh
conda activate BINDRAE

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"

SAMPLE_FILE="${SAMPLE_FILE:?Set SAMPLE_FILE}"
MD_REFERENCE_CACHE_DIR="${MD_REFERENCE_CACHE_DIR:?Set MD_REFERENCE_CACHE_DIR}"
OUTPUT="${OUTPUT:?Set OUTPUT}"

MODEL_EVAL_ARGS=()
if [[ -n "${MODEL_EVALS:-}" ]]; then
  NORMALIZED_MODEL_EVALS="${MODEL_EVALS//;/,}"
  NORMALIZED_MODEL_EVALS="${NORMALIZED_MODEL_EVALS//:/,}"
  IFS=',' read -ra MODEL_EVAL_PATHS <<< "${NORMALIZED_MODEL_EVALS}"
  for path in "${MODEL_EVAL_PATHS[@]}"; do
    MODEL_EVAL_ARGS+=(--model_eval "${path}")
  done
fi

MAX_BATCH_ARGS=()
if [[ -n "${MAX_BATCHES:-}" ]]; then
  MAX_BATCH_ARGS+=(--max_batches "${MAX_BATCHES}")
fi

python scripts/analyze_md_phase_representation_ceiling.py \
  --data_dir "${DATA_DIR:-processed_data/triplets}" \
  --split "${SPLIT:-train}" \
  --valid_samples_file "${SAMPLE_FILE}" \
  --md_reference_cache_dir "${MD_REFERENCE_CACHE_DIR}" \
  --output "${OUTPUT}" \
  --device "${DEVICE:-cuda}" \
  --num_workers "${NUM_WORKERS:-4}" \
  --n_path_steps "${N_PATH_STEPS:-20}" \
  --md_min_phase_confidence "${MD_MIN_PHASE_CONFIDENCE:-0.05}" \
  --md_pause_rate_threshold "${MD_PAUSE_RATE_THRESHOLD:-0.25}" \
  --md_backtrack_rate_threshold "${MD_BACKTRACK_RATE_THRESHOLD:-0.05}" \
  --bootstrap_samples "${BOOTSTRAP_SAMPLES:-100000}" \
  --seed "${SEED:-20260720}" \
  --trust_prechecked_samples \
  "${MAX_BATCH_ARGS[@]}" \
  "${MODEL_EVAL_ARGS[@]}"
