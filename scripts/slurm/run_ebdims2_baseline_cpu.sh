#!/usr/bin/env bash
# Run eBDIMS2 external path baseline on a curated BINDRAE sample subset.
# This is CPU/OpenMP only; eBDIMS2 itself hard-codes 16 OpenMP threads.

#SBATCH --job-name=ebdims2_base
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=16G
#SBATCH --time=06:00:00
#SBATCH --output=logs/slurm/ebdims2_base_%j.out
#SBATCH --error=logs/slurm/ebdims2_base_%j.err

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/mnt/inaisfs/data/home/zhaozc_criait/XinxiangWang/BINDRAE}"
cd "${PROJECT_ROOT}"

source /mnt/inaisfs/data/home/zhaozc_criait/miniconda3/etc/profile.d/conda.sh
conda activate BINDRAE

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-16}"

SAMPLE_FILE="${SAMPLE_FILE:-processed_data/triplets/ablation_subsets/stage2_oracle_motion_esm7sync20260630_fullcap1024_val_4270_seed20260630.txt}"
EBDIMS2_BIN="${EBDIMS2_BIN:-reference/baselines/eBDIMS2/eBDIMS2_stand_alone/LINUX_code/eBDIMS2}"
TAG="${TAG:-ebdims2_val_smoke_16_$(date +%Y%m%d_%H%M%S)}"
OUTPUT_DIR="${OUTPUT_DIR:-reference/baselines/eBDIMS2/runs/${TAG}}"

MAX_SAMPLES="${MAX_SAMPLES:-16}"
CANDIDATE_SCAN_LIMIT="${CANDIDATE_SCAN_LIMIT:-512}"
SELECTION="${SELECTION:-ca_rmsd_desc}"
MIN_RESIDUES="${MIN_RESIDUES:-50}"
MAX_RESIDUES="${MAX_RESIDUES:-512}"
MIN_CA_RMSD="${MIN_CA_RMSD:-0.5}"
MAX_CA_RMSD="${MAX_CA_RMSD:-1000000}"
SAVE_FREQ="${SAVE_FREQ:-25}"
CONVERGENCE="${CONVERGENCE:-99.9}"
TIMEOUT_SEC="${TIMEOUT_SEC:-1800}"
MIN_FRAMES="${MIN_FRAMES:-8}"
CLEAN="${CLEAN:-1}"
SAME_CHAIN_ONLY="${SAME_CHAIN_ONLY:-0}"
SKIP_EXISTING="${SKIP_EXISTING:-0}"

ARGS=(
  --data_dir processed_data/triplets
  --sample_file "${SAMPLE_FILE}"
  --ebdims2_bin "${EBDIMS2_BIN}"
  --output_dir "${OUTPUT_DIR}"
  --manifest_output "${OUTPUT_DIR}/run_manifest.jsonl"
  --summary_output "${OUTPUT_DIR}/summary.json"
  --max_samples "${MAX_SAMPLES}"
  --candidate_scan_limit "${CANDIDATE_SCAN_LIMIT}"
  --selection "${SELECTION}"
  --min_residues "${MIN_RESIDUES}"
  --max_residues "${MAX_RESIDUES}"
  --min_ca_rmsd "${MIN_CA_RMSD}"
  --max_ca_rmsd "${MAX_CA_RMSD}"
  --save_freq "${SAVE_FREQ}"
  --convergence "${CONVERGENCE}"
  --timeout_sec "${TIMEOUT_SEC}"
  --min_frames "${MIN_FRAMES}"
  --shard_index "${SHARD_INDEX:-0}"
  --num_shards "${NUM_SHARDS:-1}"
)

if [[ "${CLEAN}" == "1" ]]; then
  ARGS+=(--clean)
fi
if [[ "${SAME_CHAIN_ONLY}" == "1" ]]; then
  ARGS+=(--same_chain_only)
fi
if [[ "${SKIP_EXISTING}" == "1" ]]; then
  ARGS+=(--skip_existing)
fi

echo "Running eBDIMS2 baseline"
printf '  TAG=%s\n  OUTPUT_DIR=%s\n  SAMPLE_FILE=%s\n  MAX_SAMPLES=%s\n  CONVERGENCE=%s\n' \
  "${TAG}" "${OUTPUT_DIR}" "${SAMPLE_FILE}" "${MAX_SAMPLES}" "${CONVERGENCE}"

python scripts/run_ebdims2_baseline.py "${ARGS[@]}"

python scripts/evaluate_ebdims2_ca_paths.py \
  --data_dir processed_data/triplets \
  --run_manifest "${OUTPUT_DIR}/run_manifest.jsonl" \
  --output "${OUTPUT_DIR}/ca_eval_summary.json" \
  --per_sample_output "${OUTPUT_DIR}/ca_eval_per_sample.jsonl" \
  --min_frames "${MIN_FRAMES}" \
  --n_path_frames "${N_PATH_FRAMES:-16}"
