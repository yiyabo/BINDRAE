#!/usr/bin/env bash
# Fixed-protocol silver path replicas. Each array element is one independent seed.

#SBATCH --job-name=mdreplica
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --output=logs/slurm/md_replica_%A_%a.out
#SBATCH --error=logs/slurm/md_replica_%A_%a.err

set -euo pipefail

ROOT="${ROOT:-/mnt/inaisfs/data/home/zhaozc_criait/XinxiangWang/BINDRAE}"
MATRIX="${MATRIX:?Set MATRIX to replica_matrix.jsonl}"
# Slurm rejects array indices at or above MaxArraySize (1001 here), so matrices
# larger than that are submitted as several arrays, each passing the offset of
# its own chunk. A single-array submission leaves the offset at 0.
ARRAY_TASK_ID="${SLURM_ARRAY_TASK_ID:?Submit this launcher as a Slurm array}"
MATRIX_INDEX=$((ARRAY_TASK_ID + ${MATRIX_INDEX_OFFSET:-0}))
PLATFORM="${PLATFORM:-CPU}"
RESIDUAL_ENVELOPE="${RESIDUAL_ENVELOPE:-sin2}"
NORMAL_PROJECTION_MODE="${NORMAL_PROJECTION_MODE:-product}"

cd "$ROOT"
mkdir -p logs/slurm

source /mnt/inaisfs/data/home/zhaozc_criait/miniconda3/etc/profile.d/conda.sh
conda activate BINDRAE-MD

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-32}"

echo "Matrix: $MATRIX"
echo "Matrix index: $MATRIX_INDEX (array task $ARRAY_TASK_ID + offset ${MATRIX_INDEX_OFFSET:-0})"
echo "Platform: $PLATFORM"
echo "Residual envelope: $RESIDUAL_ENVELOPE"
echo "Normal projection: $NORMAL_PROJECTION_MODE"

python scripts/run_md_replica_pipeline.py \
  --matrix "$MATRIX" \
  --index "$MATRIX_INDEX" \
  --platform "$PLATFORM" \
  --residual-envelope "$RESIDUAL_ENVELOPE" \
  --normal-projection-mode "$NORMAL_PROJECTION_MODE"
