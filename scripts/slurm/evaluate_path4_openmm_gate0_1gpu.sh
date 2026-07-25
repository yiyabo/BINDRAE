#!/bin/bash
# One-candidate ff14SB/OpenFF/GBn2 Gate-0 relaxation and score.

#SBATCH --job-name=p4_omm_gate0
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:A100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=02:00:00
#SBATCH --output=logs/slurm/path4_openmm_gate0_%j.out
#SBATCH --error=logs/slurm/path4_openmm_gate0_%j.err

set -euo pipefail
unset LD_PRELOAD

ROOT="${ROOT:-/mnt/inaisfs/data/home/zhaozc_criait/XinxiangWang/BINDRAE}"
CANDIDATE="${CANDIDATE:?CANDIDATE is required}"
PREPARATION_REPORT="${PREPARATION_REPORT:?PREPARATION_REPORT is required}"
IMPLICIT_CACHE_DIR="${IMPLICIT_CACHE_DIR:?IMPLICIT_CACHE_DIR is required}"
OUTPUT="${OUTPUT:?OUTPUT is required}"
FRAME_REFERENCE_CACHE="${FRAME_REFERENCE_CACHE:?FRAME_REFERENCE_CACHE is required}"

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"

source /mnt/inaisfs/data/home/zhaozc_criait/miniconda3/etc/profile.d/conda.sh
conda activate BINDRAE-MD
cd "$ROOT"
mkdir -p logs/slurm "$(dirname "$OUTPUT")" "$IMPLICIT_CACHE_DIR"

ARGS=(
  --candidate "$CANDIDATE"
  --preparation-report "$PREPARATION_REPORT"
  --implicit-cache-dir "$IMPLICIT_CACHE_DIR"
  --output "$OUTPUT"
  --platform CUDA
  --device-index "${DEVICE_INDEX:-0}"
  --cpu-threads "${SLURM_CPUS_PER_TASK:-8}"
  --restraint-mode "${RESTRAINT_MODE:-ca}"
  --frame-initialization reference_cache
  --frame-reference-cache "$FRAME_REFERENCE_CACHE"
  --restraint-k-kj-mol-nm2 "${RESTRAINT_K_KJ_MOL_NM2:-5000}"
  --minimization-tolerance-kj-mol-nm "${MINIMIZATION_TOLERANCE_KJ_MOL_NM:-25}"
  --max-minimization-iterations "${MAX_MINIMIZATION_ITERATIONS:-250}"
  --maximum-relaxed-residue-net-force-kj-mol-nm \
    "${MAXIMUM_RELAXED_RESIDUE_NET_FORCE_KJ_MOL_NM:-500}"
  --minimum-residue-mapping "${MINIMUM_RESIDUE_MAPPING:-0.98}"
  --minimum-atom-mapping "${MINIMUM_ATOM_MAPPING:-0.95}"
  --severe-clash-distance-angstrom "${SEVERE_CLASH_DISTANCE_ANGSTROM:-1.5}"
)

if [[ "${FORCE_REBUILD_SYSTEM:-0}" == "1" ]]; then
  ARGS+=(--force-rebuild-system)
fi

python scripts/evaluate_path4_openmm_gate0.py "${ARGS[@]}"
