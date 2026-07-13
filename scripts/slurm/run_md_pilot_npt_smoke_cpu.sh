#!/usr/bin/env bash
# Short restrained-to-unrestrained NPT density and endpoint stability smoke.

#SBATCH --job-name=mdpilot_npt
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=logs/slurm/md_pilot_npt_%j.out
#SBATCH --error=logs/slurm/md_pilot_npt_%j.err

set -euo pipefail

ROOT="${ROOT:-/mnt/inaisfs/data/home/zhaozc_criait/XinxiangWang/BINDRAE}"
SYSTEM_DIR="${SYSTEM_DIR:?Set SYSTEM_DIR to the minimized system directory}"
NVT_DIR="${NVT_DIR:?Set NVT_DIR to a passed NVT smoke directory}"
OUTPUT_DIR="${OUTPUT_DIR:-processed_data/md_transition/pilot_npt_${SLURM_JOB_ID}}"
SEED="${SEED:-20260713}"
RESTRAINED_EQUILIBRATION_STEPS="${RESTRAINED_EQUILIBRATION_STEPS:-2500}"
UNRESTRAINED_PRODUCTION_STEPS="${UNRESTRAINED_PRODUCTION_STEPS:-5000}"

cd "$ROOT"
mkdir -p logs/slurm "$OUTPUT_DIR"

source /mnt/inaisfs/data/home/zhaozc_criait/miniconda3/etc/profile.d/conda.sh
conda activate BINDRAE-MD

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-32}"

python scripts/run_md_pilot_npt_smoke.py \
  --system-dir "$SYSTEM_DIR" \
  --nvt-dir "$NVT_DIR" \
  --output-dir "$OUTPUT_DIR" \
  --platform CPU \
  --seed "$SEED" \
  --restrained-equilibration-steps "$RESTRAINED_EQUILIBRATION_STEPS" \
  --unrestrained-production-steps "$UNRESTRAINED_PRODUCTION_STEPS"
