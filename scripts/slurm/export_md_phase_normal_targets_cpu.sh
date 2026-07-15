#!/usr/bin/env bash
#SBATCH --job-name=md_phase_target
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=logs/slurm/md_phase_target_%j.out
#SBATCH --error=logs/slurm/md_phase_target_%j.err

set -euo pipefail

ROOT=${ROOT:-/mnt/inaisfs/data/home/zhaozc_criait/XinxiangWang/BINDRAE}
PULL_DIR=${PULL_DIR:?PULL_DIR is required}
CANDIDATE_MANIFEST=${CANDIDATE_MANIFEST:?CANDIDATE_MANIFEST is required}
TRANSITION_ID=${TRANSITION_ID:?TRANSITION_ID is required}
PREPARATION_REPORT=${PREPARATION_REPORT:?PREPARATION_REPORT is required}
OUTPUT_DIR=${OUTPUT_DIR:?OUTPUT_DIR is required}

source ~/miniconda3/etc/profile.d/conda.sh
conda activate BINDRAE-MD
cd "${ROOT}"
mkdir -p logs/slurm "${OUTPUT_DIR}"

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-16}

python scripts/export_md_phase_normal_targets.py \
  --pull-dir "${PULL_DIR}" \
  --candidate-manifest "${CANDIDATE_MANIFEST}" \
  --transition-id "${TRANSITION_ID}" \
  --preparation-report "${PREPARATION_REPORT}" \
  --output-dir "${OUTPUT_DIR}" \
  --phase-grid-size "${PHASE_GRID_SIZE:-201}" \
  --smoothing-window "${SMOOTHING_WINDOW:-21}" \
  --identity-prior-weight "${IDENTITY_PRIOR_WEIGHT:-0.02}" \
  --phase-target-mode "${PHASE_TARGET_MODE:-inferred}" \
  --residual-envelope "${RESIDUAL_ENVELOPE:-sin2}" \
  --min-endpoint-motion-norm "${MIN_ENDPOINT_MOTION_NORM:-0.5}" \
  --min-phase-confidence "${MIN_PHASE_CONFIDENCE:-0.05}" \
  --min-supervision-density "${MIN_SUPERVISION_DENSITY:-0.05}" \
  --min-residual-envelope "${MIN_RESIDUAL_ENVELOPE:-0.15}" \
  --max-normal-residual-norm "${MAX_NORMAL_RESIDUAL_NORM:-5.0}"
