#!/usr/bin/env bash
# Biased holo-to-apo global CA-RMSD pulling smoke; not a kinetics trajectory.

#SBATCH --job-name=md_rmsdpull
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=logs/slurm/md_rmsd_pull_%j.out
#SBATCH --error=logs/slurm/md_rmsd_pull_%j.err

set -euo pipefail

ROOT="${ROOT:-/mnt/inaisfs/data/home/zhaozc_criait/XinxiangWang/BINDRAE}"
CANDIDATE_MANIFEST="${CANDIDATE_MANIFEST:?Set CANDIDATE_MANIFEST}"
TRANSITION_ID="${TRANSITION_ID:?Set TRANSITION_ID}"
NPT_DIR="${NPT_DIR:?Set NPT_DIR to a passed NPT directory}"
OUTPUT_DIR="${OUTPUT_DIR:-processed_data/md_transition/rmsd_pull_${SLURM_JOB_ID}}"
SEED="${SEED:-20260713}"
PRE_EQUILIBRATION_STEPS="${PRE_EQUILIBRATION_STEPS:-500}"
PULLING_STEPS="${PULLING_STEPS:-5000}"
ENDPOINT_HOLD_STEPS="${ENDPOINT_HOLD_STEPS:-1000}"
RMSD_K_KJ_MOL_NM2="${RMSD_K_KJ_MOL_NM2:-5000}"

cd "$ROOT"
mkdir -p logs/slurm "$OUTPUT_DIR"

source /mnt/inaisfs/data/home/zhaozc_criait/miniconda3/etc/profile.d/conda.sh
conda activate BINDRAE-MD

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-32}"

python scripts/run_md_global_rmsd_pull.py \
  --candidate-manifest "$CANDIDATE_MANIFEST" \
  --transition-id "$TRANSITION_ID" \
  --npt-dir "$NPT_DIR" \
  --output-dir "$OUTPUT_DIR" \
  --platform CPU \
  --seed "$SEED" \
  --pre-equilibration-steps "$PRE_EQUILIBRATION_STEPS" \
  --pulling-steps "$PULLING_STEPS" \
  --endpoint-hold-steps "$ENDPOINT_HOLD_STEPS" \
  --rmsd-k-kj-mol-nm2 "$RMSD_K_KJ_MOL_NM2"
