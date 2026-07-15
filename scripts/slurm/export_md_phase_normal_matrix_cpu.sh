#!/usr/bin/env bash
#SBATCH --job-name=md_phase_reexport
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=logs/slurm/md_phase_reexport_%A_%a.out
#SBATCH --error=logs/slurm/md_phase_reexport_%A_%a.err

set -euo pipefail

ROOT=${ROOT:-/mnt/inaisfs/data/home/zhaozc_criait/XinxiangWang/BINDRAE}
COLLECTION_MANIFEST=${COLLECTION_MANIFEST:?COLLECTION_MANIFEST is required}
OUTPUT_ROOT=${OUTPUT_ROOT:?OUTPUT_ROOT is required}
PHASE_TARGET_MODE=${PHASE_TARGET_MODE:-identity}

source ~/miniconda3/etc/profile.d/conda.sh
conda activate BINDRAE-MD
cd "${ROOT}"
mkdir -p logs/slurm "${OUTPUT_ROOT}"

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}

python scripts/export_md_phase_normal_matrix_entry.py \
  --collection-manifest "${COLLECTION_MANIFEST}" \
  --index "${SLURM_ARRAY_TASK_ID}" \
  --output-root "${OUTPUT_ROOT}" \
  --phase-target-mode "${PHASE_TARGET_MODE}"
