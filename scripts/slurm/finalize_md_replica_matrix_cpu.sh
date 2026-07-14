#!/usr/bin/env bash
# Aggregate scientific outcomes after every replica array task has terminated.

#SBATCH --job-name=mdrep_final
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=logs/slurm/md_replica_finalize_%j.out
#SBATCH --error=logs/slurm/md_replica_finalize_%j.err

set -euo pipefail

ROOT="${ROOT:-/mnt/inaisfs/data/home/zhaozc_criait/XinxiangWang/BINDRAE}"
MATRIX="${MATRIX:?Set MATRIX to replica_matrix.jsonl}"
OUTPUT_DIR="${OUTPUT_DIR:?Set OUTPUT_DIR for finalization artifacts}"

cd "$ROOT"
mkdir -p logs/slurm "$OUTPUT_DIR"
source /mnt/inaisfs/data/home/zhaozc_criait/miniconda3/etc/profile.d/conda.sh
conda activate BINDRAE-MD
export PYTHONUNBUFFERED=1

python scripts/finalize_md_replica_matrix.py \
  --matrix "$MATRIX" \
  --output-dir "$OUTPUT_DIR"
