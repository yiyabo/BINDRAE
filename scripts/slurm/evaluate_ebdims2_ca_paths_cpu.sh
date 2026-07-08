#!/usr/bin/env bash
# Evaluate already generated eBDIMS2 CA paths.

#SBATCH --job-name=ebdims2_ca_eval
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=01:00:00
#SBATCH --output=logs/slurm/ebdims2_ca_eval_%j.out
#SBATCH --error=logs/slurm/ebdims2_ca_eval_%j.err

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/mnt/inaisfs/data/home/zhaozc_criait/XinxiangWang/BINDRAE}"
cd "${PROJECT_ROOT}"

source /mnt/inaisfs/data/home/zhaozc_criait/miniconda3/etc/profile.d/conda.sh
conda activate BINDRAE

RUN_MANIFEST="${RUN_MANIFEST:?Set RUN_MANIFEST to an eBDIMS2 run_manifest.jsonl}"
OUTPUT="${OUTPUT:-$(dirname "${RUN_MANIFEST}")/ca_eval_summary.json}"
PER_SAMPLE_OUTPUT="${PER_SAMPLE_OUTPUT:-$(dirname "${RUN_MANIFEST}")/ca_eval_per_sample.jsonl}"

python scripts/evaluate_ebdims2_ca_paths.py \
  --data_dir "${DATA_DIR:-processed_data/triplets}" \
  --run_manifest "${RUN_MANIFEST}" \
  --output "${OUTPUT}" \
  --per_sample_output "${PER_SAMPLE_OUTPUT}" \
  --min_frames "${MIN_FRAMES:-4}" \
  --n_path_frames "${N_PATH_FRAMES:-16}" \
  --method_prefix "${METHOD_PREFIX:-ebdims2_ca}" \
  --frame_offset "${FRAME_OFFSET:-apo_mass_com}"
