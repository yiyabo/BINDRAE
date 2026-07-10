#!/usr/bin/env bash
# Audit canonical Stage-2 backbone/torsion caches and write a valid manifest.

#SBATCH --job-name=s2canon_audit
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=80G
#SBATCH --time=02:00:00
#SBATCH --output=logs/slurm/stage2_canonical_audit_%j.out
#SBATCH --error=logs/slurm/stage2_canonical_audit_%j.err

set -euo pipefail

ROOT="${ROOT:-/mnt/inaisfs/data/home/zhaozc_criait/XinxiangWang/BINDRAE}"
DATA_DIR="${DATA_DIR:-processed_data/triplets}"
SAMPLE_LIST="${SAMPLE_LIST:?Set SAMPLE_LIST}"
VALID_OUTPUT="${VALID_OUTPUT:?Set VALID_OUTPUT}"
INVALID_OUTPUT="${INVALID_OUTPUT:?Set INVALID_OUTPUT}"
WORKERS="${WORKERS:-20}"

cd "$ROOT"
mkdir -p logs/slurm "$(dirname "$VALID_OUTPUT")" "$(dirname "$INVALID_OUTPUT")"

source /mnt/inaisfs/data/home/zhaozc_criait/miniconda3/etc/profile.d/conda.sh
conda activate BINDRAE

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

echo "Sample list:    $SAMPLE_LIST"
echo "Requested:      $(wc -l < "$SAMPLE_LIST")"
echo "Valid output:   $VALID_OUTPUT"
echo "Invalid output: $INVALID_OUTPUT"
echo "Workers:        $WORKERS"

python scripts/audit_canonical_stage2_manifest.py \
  --data-dir "$DATA_DIR" \
  --sample-list "$SAMPLE_LIST" \
  --valid-output "$VALID_OUTPUT" \
  --invalid-output "$INVALID_OUTPUT" \
  --workers "$WORKERS"
