#!/usr/bin/env bash
# Run simple CA morphing baselines and evaluate ligand-pocket CA metrics.

#SBATCH --job-name=ca_morph
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=01:00:00
#SBATCH --output=logs/slurm/ca_morph_%j.out
#SBATCH --error=logs/slurm/ca_morph_%j.err

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/mnt/inaisfs/data/home/zhaozc_criait/XinxiangWang/BINDRAE}"
cd "${PROJECT_ROOT}"

source /mnt/inaisfs/data/home/zhaozc_criait/miniconda3/etc/profile.d/conda.sh
conda activate BINDRAE

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-2}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-2}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-2}"
export PYTHONUNBUFFERED=1

METHOD="${METHOD:?Set METHOD to static, linear, or smoothstep}"
TAG="${TAG:-ca_${METHOD}_clean309_$(date +%Y%m%d_%H%M%S)}"
OUTPUT_DIR="${OUTPUT_DIR:-reference/baselines/CA_morph/runs/${TAG}}"
SAMPLE_FILE="${SAMPLE_FILE:-processed_data/triplets/ablation_subsets/stage2_ebdims2_clean309_val_20260706.txt}"
RUN_MANIFEST="${RUN_MANIFEST:-${OUTPUT_DIR}/run_manifest.jsonl}"
SUMMARY_OUTPUT="${SUMMARY_OUTPUT:-${OUTPUT_DIR}/summary.json}"
EVAL_OUTPUT="${EVAL_OUTPUT:-${OUTPUT_DIR}/ca_eval_summary.json}"
PER_SAMPLE_OUTPUT="${PER_SAMPLE_OUTPUT:-${OUTPUT_DIR}/ca_eval_per_sample.jsonl}"
METHOD_PREFIX="${METHOD_PREFIX:-ca_${METHOD}}"

mkdir -p logs/slurm "${OUTPUT_DIR}"

echo "=============================================="
echo "BINDRAE simple CA morph baseline"
echo "=============================================="
echo "Job ID:       ${SLURM_JOB_ID:-NA}"
echo "Node:         ${SLURM_NODELIST:-NA}"
echo "Method:       ${METHOD}"
echo "Tag:          ${TAG}"
echo "Sample file:  ${SAMPLE_FILE}"
echo "Output dir:   ${OUTPUT_DIR}"
echo "N frames:     ${N_FRAMES:-16}"
echo "Start:        $(date)"
echo "=============================================="

python scripts/run_ca_morph_baseline.py \
  --data_dir "${DATA_DIR:-processed_data/triplets}" \
  --sample_file "${SAMPLE_FILE}" \
  --output_dir "${OUTPUT_DIR}" \
  --manifest_output "${RUN_MANIFEST}" \
  --summary_output "${SUMMARY_OUTPUT}" \
  --method "${METHOD}" \
  --max_samples "${MAX_SAMPLES:-10000}" \
  --candidate_scan_limit "${CANDIDATE_SCAN_LIMIT:-10000}" \
  --min_residues "${MIN_RESIDUES:-50}" \
  --max_residues "${MAX_RESIDUES:-512}" \
  --min_ca_rmsd "${MIN_CA_RMSD:-0.0}" \
  --max_ca_rmsd "${MAX_CA_RMSD:-1000000.0}" \
  --n_frames "${N_FRAMES:-16}" \
  ${CLEAN:+--clean} \
  ${SKIP_EXISTING:+--skip_existing}

python scripts/evaluate_ebdims2_ca_paths.py \
  --data_dir "${DATA_DIR:-processed_data/triplets}" \
  --run_manifest "${RUN_MANIFEST}" \
  --output "${EVAL_OUTPUT}" \
  --per_sample_output "${PER_SAMPLE_OUTPUT}" \
  --min_frames "${MIN_FRAMES:-4}" \
  --n_path_frames "${N_PATH_FRAMES:-16}" \
  --method_prefix "${METHOD_PREFIX}" \
  --frame_offset none

echo "Completed: $(date)"
echo "Run manifest: ${RUN_MANIFEST}"
echo "Eval output:  ${EVAL_OUTPUT}"
