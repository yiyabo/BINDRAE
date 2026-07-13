#!/usr/bin/env bash
# Restrained heating plus short unrestrained NVT stability smoke.

#SBATCH --job-name=mdpilot_nvt
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=logs/slurm/md_pilot_nvt_%j.out
#SBATCH --error=logs/slurm/md_pilot_nvt_%j.err

set -euo pipefail

ROOT="${ROOT:-/mnt/inaisfs/data/home/zhaozc_criait/XinxiangWang/BINDRAE}"
INPUT_DIR="${INPUT_DIR:?Set INPUT_DIR to a prepared MD system directory}"
OUTPUT_DIR="${OUTPUT_DIR:-processed_data/md_transition/pilot_nvt_${SLURM_JOB_ID}}"
SEED="${SEED:-20260713}"
HEATING_STEPS_PER_STAGE="${HEATING_STEPS_PER_STAGE:-250}"
RESTRAINED_EQUILIBRATION_STEPS="${RESTRAINED_EQUILIBRATION_STEPS:-1000}"
UNRESTRAINED_NVT_STEPS="${UNRESTRAINED_NVT_STEPS:-1000}"

cd "$ROOT"
mkdir -p logs/slurm "$OUTPUT_DIR"

source /mnt/inaisfs/data/home/zhaozc_criait/miniconda3/etc/profile.d/conda.sh
conda activate BINDRAE-MD

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-32}"

python scripts/run_md_pilot_dynamics_smoke.py \
  --input-dir "$INPUT_DIR" \
  --output-dir "$OUTPUT_DIR" \
  --platform CPU \
  --seed "$SEED" \
  --heating-steps-per-stage "$HEATING_STEPS_PER_STAGE" \
  --restrained-equilibration-steps "$RESTRAINED_EQUILIBRATION_STEPS" \
  --unrestrained-nvt-steps "$UNRESTRAINED_NVT_STEPS"
