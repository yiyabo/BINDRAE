#!/usr/bin/env bash
# Evaluate an already generated CA baseline on the frozen MD-reference subset.

#SBATCH --job-name=ca_md_eval
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=logs/slurm/ca_md_eval_%j.out
#SBATCH --error=logs/slurm/ca_md_eval_%j.err

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/mnt/inaisfs/data/home/zhaozc_criait/XinxiangWang/BINDRAE}"
cd "${PROJECT_ROOT}"

source /mnt/inaisfs/data/home/zhaozc_criait/miniconda3/etc/profile.d/conda.sh
conda activate BINDRAE

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"

RUN_MANIFEST="${RUN_MANIFEST:?Set RUN_MANIFEST}"
METHOD="${METHOD:?Set METHOD}"
OUTPUT="${OUTPUT:?Set OUTPUT}"
SAMPLE_FILE="${SAMPLE_FILE:?Set SAMPLE_FILE}"
MD_REFERENCE_CACHE_DIR="${MD_REFERENCE_CACHE_DIR:?Set MD_REFERENCE_CACHE_DIR}"
FRAME_OFFSET="${FRAME_OFFSET:-none}"

python scripts/evaluate_ca_baseline_md_reference_paths.py \
  --data_dir "${DATA_DIR:-processed_data/triplets}" \
  --split "${SPLIT:-train}" \
  --valid_samples_file "${SAMPLE_FILE}" \
  --run_manifest "${RUN_MANIFEST}" \
  --md_reference_cache_dir "${MD_REFERENCE_CACHE_DIR}" \
  --output "${OUTPUT}" \
  --method "${METHOD}" \
  --n_path_steps "${N_PATH_STEPS:-20}" \
  --min_frames "${MIN_FRAMES:-4}" \
  --num_workers "${NUM_WORKERS:-4}" \
  --frame_offset "${FRAME_OFFSET}" \
  --trust_prechecked_samples \
  ${ALLOW_MISSING_SYSTEMS:+--allow_missing_systems} \
  ${APPEND_HOLO_ENDPOINT:+--append_holo_endpoint}
