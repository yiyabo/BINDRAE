#!/bin/bash
# Development-only OpenMM optimizer smoke on an old training system.

#SBATCH --job-name=p4g0_omm_opt
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=logs/slurm/path4_gate0_openmm_optimizer_%j.out
#SBATCH --error=logs/slurm/path4_gate0_openmm_optimizer_%j.err

set -euo pipefail
unset LD_PRELOAD

ROOT="${ROOT:-/mnt/inaisfs/data/home/zhaozc_criait/XinxiangWang/BINDRAE}"
SAMPLE_ID="${SAMPLE_ID:-3zw1-E-FUC-3}"
INPUT_CANDIDATE="${INPUT_CANDIDATE:-logs/stage2/path4_gate0/path4_gate0_dev_3zw1-E-FUC-3_147139/candidates/path3/3zw1-E-FUC-3.npz}"
PREPARATION_REPORT="${PREPARATION_REPORT:-processed_data/md_transition/context_full500_endpointdedup_20260716_v2/systems/${SAMPLE_ID}/setup/preparation_report.json}"
IMPLICIT_CACHE_DIR="${IMPLICIT_CACHE_DIR:-processed_data/md_transition/path4_gate0_implicit/${SAMPLE_ID}_gbn2_v1}"
TAG="${TAG:-path4_gate0_openmm_optimizer_dev_${SAMPLE_ID}_${SLURM_JOB_ID:-manual}}"
OUTPUT_ROOT="${OUTPUT_ROOT:-logs/stage2/path4_gate0/$TAG}"
OUTPUT_CANDIDATE="$OUTPUT_ROOT/candidates/openmm_multistart/${SAMPLE_ID}.npz"
OPTIMIZER_REPORT="$OUTPUT_ROOT/optimizer/report.json"
FRAME_REFERENCE_CACHE="$OUTPUT_ROOT/reference/path3_all_atom_reference.npz"
FRAME_REFERENCE_REPORT="$OUTPUT_ROOT/reference/report.json"
SCORE_REPORT="$OUTPUT_ROOT/openmm/score.json"

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"

cd "$ROOT"
mkdir -p logs/slurm "$(dirname "$OUTPUT_CANDIDATE")" \
  "$(dirname "$OPTIMIZER_REPORT")" "$OUTPUT_ROOT/reference" \
  "$(dirname "$SCORE_REPORT")"

source /mnt/inaisfs/data/home/zhaozc_criait/miniconda3/etc/profile.d/conda.sh
conda activate BINDRAE-MD

python scripts/optimize_path4_openmm_gate0.py \
  --candidate "$INPUT_CANDIDATE" \
  --preparation-report "$PREPARATION_REPORT" \
  --implicit-cache-dir "$IMPLICIT_CACHE_DIR" \
  --output-candidate "$OUTPUT_CANDIDATE" \
  --report "$OPTIMIZER_REPORT" \
  --candidate-label openmm_multistart_dev_smoke \
  --platform CPU \
  --cpu-threads "${SLURM_CPUS_PER_TASK:-4}" \
  --time-basis-rank "${TIME_BASIS_RANK:-2}" \
  --num-starts "${NUM_STARTS:-3}" \
  --iterations "${ITERATIONS:-4}" \
  --route-seed-scale-angstrom "${ROUTE_SEED_SCALE_ANGSTROM:-0.05}" \
  --max-residue-translation-angstrom "${MAX_RESIDUE_TRANSLATION_ANGSTROM:-0.75}" \
  --step-size-angstrom "${STEP_SIZE_ANGSTROM:-0.05}" \
  --line-search-steps "${LINE_SEARCH_STEPS:-5}" \
  --chain-smoothing-steps "${CHAIN_SMOOTHING_STEPS:-2}" \
  --maximum-reference-atomic-force-kj-mol-nm \
    "${MAXIMUM_REFERENCE_ATOMIC_FORCE_KJ_MOL_NM:-1000000}"

python scripts/build_path4_openmm_frame_reference.py \
  --candidate "$INPUT_CANDIDATE" \
  --preparation-report "$PREPARATION_REPORT" \
  --implicit-cache-dir "$IMPLICIT_CACHE_DIR" \
  --output-cache "$FRAME_REFERENCE_CACHE" \
  --report "$FRAME_REFERENCE_REPORT" \
  --platform CPU \
  --cpu-threads "${SLURM_CPUS_PER_TASK:-4}" \
  --maximum-reference-atomic-force-kj-mol-nm \
    "${MAXIMUM_REFERENCE_ATOMIC_FORCE_KJ_MOL_NM:-1000000}" \
  --maximum-frame-reference-atomic-force-kj-mol-nm \
    "${MAXIMUM_FRAME_REFERENCE_ATOMIC_FORCE_KJ_MOL_NM:-1000000}"

python scripts/evaluate_path4_openmm_gate0.py \
  --candidate "$OUTPUT_CANDIDATE" \
  --preparation-report "$PREPARATION_REPORT" \
  --implicit-cache-dir "$IMPLICIT_CACHE_DIR" \
  --output "$SCORE_REPORT" \
  --platform CPU \
  --cpu-threads "${SLURM_CPUS_PER_TASK:-4}" \
  --restraint-mode ca \
  --frame-initialization reference_cache \
  --frame-reference-cache "$FRAME_REFERENCE_CACHE" \
  --restraint-k-kj-mol-nm2 "${RESTRAINT_K_KJ_MOL_NM2:-5000}" \
  --minimization-tolerance-kj-mol-nm "${MINIMIZATION_TOLERANCE_KJ_MOL_NM:-250}" \
  --max-minimization-iterations "${MAX_MINIMIZATION_ITERATIONS:-10}" \
  --maximum-relaxed-residue-net-force-kj-mol-nm \
    "${MAXIMUM_RELAXED_RESIDUE_NET_FORCE_KJ_MOL_NM:-500}" \
  --maximum-reference-atomic-force-kj-mol-nm \
    "${MAXIMUM_REFERENCE_ATOMIC_FORCE_KJ_MOL_NM:-1000000}" \
  --diagnostic-force-components

echo "OPENMM_OPTIMIZER_DEV_OK=$OPTIMIZER_REPORT"
echo "OPENMM_OPTIMIZER_SCORE_OK=$SCORE_REPORT"
