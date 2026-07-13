#!/usr/bin/env bash
# Prepare, solvate, and minimize one AHoJ protein-ligand MD pilot system.

#SBATCH --job-name=mdpilot_setup
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --gres=gpu:A100:1
#SBATCH --time=02:00:00
#SBATCH --output=logs/slurm/md_pilot_setup_%j.out
#SBATCH --error=logs/slurm/md_pilot_setup_%j.err

set -euo pipefail

ROOT="${ROOT:-/mnt/inaisfs/data/home/zhaozc_criait/XinxiangWang/BINDRAE}"
CANDIDATE_MANIFEST="${CANDIDATE_MANIFEST:?Set CANDIDATE_MANIFEST}"
CANDIDATE_INDEX="${CANDIDATE_INDEX:-0}"
OUTPUT_DIR="${OUTPUT_DIR:-processed_data/md_transition/pilot_setup_${SLURM_JOB_ID}}"
PLATFORM="${PLATFORM:-CUDA}"
MAX_MINIMIZATION_ITERATIONS="${MAX_MINIMIZATION_ITERATIONS:-500}"
SOLVENT_MINIMIZATION_ITERATIONS="${SOLVENT_MINIMIZATION_ITERATIONS:-1000}"
SEED="${SEED:-20260713}"

cd "$ROOT"
mkdir -p logs/slurm "$OUTPUT_DIR"

source /mnt/inaisfs/data/home/zhaozc_criait/miniconda3/etc/profile.d/conda.sh
conda activate BINDRAE-MD

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=4

echo "Candidate manifest: $CANDIDATE_MANIFEST"
echo "Candidate index:    $CANDIDATE_INDEX"
echo "Output dir:         $OUTPUT_DIR"
echo "Platform:           $PLATFORM"

python scripts/prepare_md_pilot_system.py \
  --candidate-manifest "$CANDIDATE_MANIFEST" \
  --candidate-index "$CANDIDATE_INDEX" \
  --output-dir "$OUTPUT_DIR" \
  --platform "$PLATFORM" \
  --seed "$SEED" \
  --solvent-minimization-iterations "$SOLVENT_MINIMIZATION_ITERATIONS" \
  --max-minimization-iterations "$MAX_MINIMIZATION_ITERATIONS"
