#!/usr/bin/env bash
# Rebuild backbone/torsion caches with canonical residue identities.

#SBATCH --job-name=resid_cache_v1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=40
#SBATCH --mem=120G
#SBATCH --time=12:00:00
#SBATCH --output=logs/slurm/residue_identity_cache_%j.out
#SBATCH --error=logs/slurm/residue_identity_cache_%j.err

set -euo pipefail

ROOT="${ROOT:-/mnt/inaisfs/data/home/zhaozc_criait/XinxiangWang/BINDRAE}"
DATA_DIR="${DATA_DIR:-processed_data/triplets}"
SAMPLE_LIST="${SAMPLE_LIST:?Set SAMPLE_LIST to a file containing sample IDs}"
NUM_WORKERS="${NUM_WORKERS:-36}"
MAX_SAMPLES="${MAX_SAMPLES:-0}"

cd "$ROOT"
mkdir -p logs/slurm

source /mnt/inaisfs/data/home/zhaozc_criait/miniconda3/etc/profile.d/conda.sh
conda activate BINDRAE

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

echo "=============================================="
echo "Canonical residue cache rebuild"
echo "Job ID:       ${SLURM_JOB_ID:-NA}"
echo "Node:         ${SLURM_NODELIST:-NA}"
echo "Data dir:     $DATA_DIR"
echo "Sample list:  $SAMPLE_LIST"
echo "Samples:      $(wc -l < "$SAMPLE_LIST")"
echo "Workers:      $NUM_WORKERS"
echo "Max samples:  $MAX_SAMPLES"
echo "Start:        $(date)"
echo "=============================================="

python -m unittest \
  tests.test_residue_identity \
  tests.test_stage2_residue_alignment

python scripts/extract_ahojdb_torsions.py \
  --data-dir "$DATA_DIR" \
  --sample-list "$SAMPLE_LIST" \
  --num-workers "$NUM_WORKERS" \
  --max-samples "$MAX_SAMPLES"

python scripts/cache_backbone_coords.py \
  --data_dir "$DATA_DIR" \
  --split train \
  --sample-list "$SAMPLE_LIST" \
  --num-workers "$NUM_WORKERS" \
  --max-samples "$MAX_SAMPLES"

python scripts/cache_backbone_coords.py \
  --data_dir "$DATA_DIR" \
  --split val \
  --sample-list "$SAMPLE_LIST" \
  --num-workers "$NUM_WORKERS" \
  --max-samples "$MAX_SAMPLES"

echo "Completed: $(date)"
