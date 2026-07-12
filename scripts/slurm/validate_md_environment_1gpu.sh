#!/bin/bash
# Validate the isolated BINDRAE-MD environment on one allocated A100.

#SBATCH --job-name=md_env_smoke
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:A100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:10:00
#SBATCH --output=logs/slurm/md_env_smoke_%j.out
#SBATCH --error=logs/slurm/md_env_smoke_%j.err

set -euo pipefail

ROOT="/mnt/inaisfs/data/home/zhaozc_criait/XinxiangWang/BINDRAE"
MD_ENV_NAME="${MD_ENV_NAME:-BINDRAE-MD}"

unset LD_PRELOAD
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=4

source /mnt/inaisfs/data/home/zhaozc_criait/miniconda3/etc/profile.d/conda.sh
conda activate "${MD_ENV_NAME}"

cd "${ROOT}"
mkdir -p logs/slurm logs/md

echo "Job ID: ${SLURM_JOB_ID}"
echo "Node: ${SLURM_NODELIST}"
echo "Environment: ${MD_ENV_NAME}"
echo "Start: $(date --iso-8601=seconds)"
nvidia-smi --query-gpu=index,name,driver_version,memory.total --format=csv,noheader

python scripts/validate_md_environment.py \
  --require-cuda \
  --output "logs/md/environment_${SLURM_JOB_ID}.json"
python -m openmm.testInstallation

echo "Completed: $(date --iso-8601=seconds)"
