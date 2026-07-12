#!/bin/bash
# Diagnose sample-level Stage-2 objective tails from a fixed checkpoint.

#SBATCH --job-name=s2_obj_outlier
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:A100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
#SBATCH --time=03:00:00
#SBATCH --output=logs/slurm/stage2_objective_outliers_%j.out
#SBATCH --error=logs/slurm/stage2_objective_outliers_%j.err

set -euo pipefail
unset LD_PRELOAD

ROOT="${ROOT:-/mnt/inaisfs/data/home/zhaozc_criait/XinxiangWang/BINDRAE}"
ENV_PREFIX=/mnt/inaisfs/data/home/zhaozc_criait/miniconda3/envs/BINDRAE
export PATH=/data/soft/slurm/24.11.4/bin:$ENV_PREFIX/bin:/data/soft/slurm/24.11.4/bin:$PATH
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=4

cd "$ROOT"
mkdir -p logs/slurm logs/stage2/objective_outliers
source /mnt/inaisfs/data/home/zhaozc_criait/miniconda3/etc/profile.d/conda.sh
conda activate BINDRAE

CHECKPOINT="${CHECKPOINT:?CHECKPOINT is required}"
VALID_SAMPLES_FILE="${VALID_SAMPLES_FILE:-}"
MAX_BATCHES="${MAX_BATCHES:-}"
TOP_K="${TOP_K:-32}"
PHASE_RESIDUAL_SCALE="${PHASE_RESIDUAL_SCALE:-}"
PHASE_RESIDUAL_TAU_MODE="${PHASE_RESIDUAL_TAU_MODE:-}"
PHASE_RESIDUAL_BRIDGE_MODE="${PHASE_RESIDUAL_BRIDGE_MODE:-}"
OUTPUT="${OUTPUT:-logs/stage2/objective_outliers/$(basename "$(dirname "$CHECKPOINT")").json}"

ARGS=(
  --checkpoint "$CHECKPOINT"
  --num_workers 0
  --top_k "$TOP_K"
  --device cuda
  --output "$OUTPUT"
)
if [[ -n "$VALID_SAMPLES_FILE" ]]; then
  ARGS+=(--valid_samples_file "$VALID_SAMPLES_FILE")
fi
if [[ -n "$MAX_BATCHES" ]]; then
  ARGS+=(--max_batches "$MAX_BATCHES")
fi
if [[ -n "$PHASE_RESIDUAL_SCALE" ]]; then
  ARGS+=(--phase_residual_scale "$PHASE_RESIDUAL_SCALE")
fi
if [[ -n "$PHASE_RESIDUAL_TAU_MODE" ]]; then
  ARGS+=(--phase_residual_tau_mode "$PHASE_RESIDUAL_TAU_MODE")
fi
if [[ -n "$PHASE_RESIDUAL_BRIDGE_MODE" ]]; then
  ARGS+=(--phase_residual_bridge_mode "$PHASE_RESIDUAL_BRIDGE_MODE")
fi

python scripts/diagnose_stage2_objective_outliers.py "${ARGS[@]}"
