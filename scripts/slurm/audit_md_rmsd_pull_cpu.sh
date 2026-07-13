#!/usr/bin/env bash
#SBATCH --job-name=mdpull_audit
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=00:15:00
#SBATCH --output=logs/slurm/md_pull_audit_%j.out
#SBATCH --error=logs/slurm/md_pull_audit_%j.err

set -euo pipefail

ROOT=${ROOT:-/mnt/inaisfs/data/home/zhaozc_criait/XinxiangWang/BINDRAE}
PULL_DIR=${PULL_DIR:?PULL_DIR is required}
OUTPUT=${OUTPUT:-${PULL_DIR}/path_metrics_audit.json}

cd "${ROOT}"
mkdir -p logs/slurm "$(dirname "${OUTPUT}")"
source ~/miniconda3/etc/profile.d/conda.sh
conda activate BINDRAE-MD
export PYTHONUNBUFFERED=1

python scripts/audit_md_rmsd_pull.py \
  --pull-dir "${PULL_DIR}" \
  --output "${OUTPUT}"
