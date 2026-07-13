#!/usr/bin/env bash
# Atomistic bond/clash audit for a generated RMSD-CV path.

#SBATCH --job-name=mdpath_audit
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G
#SBATCH --time=01:00:00
#SBATCH --output=logs/slurm/md_path_audit_%j.out
#SBATCH --error=logs/slurm/md_path_audit_%j.err

set -euo pipefail

ROOT="${ROOT:-/mnt/inaisfs/data/home/zhaozc_criait/XinxiangWang/BINDRAE}"
PULL_DIR="${PULL_DIR:?Set PULL_DIR}"
PREPARATION_REPORT="${PREPARATION_REPORT:?Set PREPARATION_REPORT}"
OUTPUT="${OUTPUT:-${PULL_DIR}/atomistic_path_audit.json}"

cd "$ROOT"
mkdir -p logs/slurm "$(dirname "$OUTPUT")"

source /mnt/inaisfs/data/home/zhaozc_criait/miniconda3/etc/profile.d/conda.sh
conda activate BINDRAE-MD

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1

python scripts/audit_md_atomistic_path.py \
  --pull-dir "$PULL_DIR" \
  --preparation-report "$PREPARATION_REPORT" \
  --output "$OUTPUT"
