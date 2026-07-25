#!/bin/bash
# Matched engineering rescore of frozen Path-3 and one optimized candidate.
# This diagnoses scorer/optimizer agreement on one old training system only.

#SBATCH --job-name=p4g0_pair_score
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:A100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=01:00:00
#SBATCH --output=logs/slurm/path4_gate0_paired_rescore_%j.out
#SBATCH --error=logs/slurm/path4_gate0_paired_rescore_%j.err

set -euo pipefail
unset LD_PRELOAD

ROOT="${ROOT:-/mnt/inaisfs/data/home/zhaozc_criait/XinxiangWang/BINDRAE}"
SAMPLE_ID="${SAMPLE_ID:-3zw1-E-FUC-3}"
PATH3_CANDIDATE="${PATH3_CANDIDATE:-logs/stage2/path4_gate0/path4_gate0_dev_3zw1-E-FUC-3_147139/candidates/path3/3zw1-E-FUC-3.npz}"
OPTIMIZED_CANDIDATE="${OPTIMIZED_CANDIDATE:-logs/stage2/path4_gate0/path4_gate0_openmm_optimizer_dev_3zw1-E-FUC-3_147166/candidates/openmm_multistart/3zw1-E-FUC-3.npz}"
OPTIMIZER_REPORT="${OPTIMIZER_REPORT:-logs/stage2/path4_gate0/path4_gate0_openmm_optimizer_dev_3zw1-E-FUC-3_147166/optimizer/report.json}"
PREPARATION_REPORT="${PREPARATION_REPORT:-processed_data/md_transition/context_full500_endpointdedup_20260716_v2/systems/${SAMPLE_ID}/setup/preparation_report.json}"
IMPLICIT_CACHE_DIR="${IMPLICIT_CACHE_DIR:-processed_data/md_transition/path4_gate0_implicit/${SAMPLE_ID}_gbn2_v1}"
TAG="${TAG:-path4_gate0_paired_rescore_${SAMPLE_ID}_${SLURM_JOB_ID:-manual}}"
OUTPUT_ROOT="${OUTPUT_ROOT:-logs/stage2/path4_gate0/$TAG}"
FRAME_REFERENCE_CACHE="$OUTPUT_ROOT/reference/path3_all_atom_reference.npz"
FRAME_REFERENCE_REPORT="$OUTPUT_ROOT/reference/report.json"
PATH3_SCORE="$OUTPUT_ROOT/openmm/path3.json"
OPTIMIZED_SCORE="$OUTPUT_ROOT/openmm/optimized.json"
PAIRED_SUMMARY="$OUTPUT_ROOT/paired_summary.json"

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"

cd "$ROOT"
mkdir -p logs/slurm "$OUTPUT_ROOT/reference" "$OUTPUT_ROOT/openmm"
for required in "$PATH3_CANDIDATE" "$OPTIMIZED_CANDIDATE" \
  "$OPTIMIZER_REPORT" "$PREPARATION_REPORT"; do
  if [[ ! -f "$required" ]]; then
    echo "Missing required input: $required" >&2
    exit 1
  fi
done

source /mnt/inaisfs/data/home/zhaozc_criait/miniconda3/etc/profile.d/conda.sh
conda activate BINDRAE-MD

python scripts/build_path4_openmm_frame_reference.py \
  --candidate "$PATH3_CANDIDATE" \
  --preparation-report "$PREPARATION_REPORT" \
  --implicit-cache-dir "$IMPLICIT_CACHE_DIR" \
  --output-cache "$FRAME_REFERENCE_CACHE" \
  --report "$FRAME_REFERENCE_REPORT" \
  --platform CUDA \
  --device-index "${DEVICE_INDEX:-0}" \
  --cpu-threads "${SLURM_CPUS_PER_TASK:-8}" \
  --reference-relaxation-iterations \
    "${REFERENCE_RELAXATION_ITERATIONS:-25}" \
  --reference-restraint-k-kj-mol-nm2 \
    "${REFERENCE_RESTRAINT_K_KJ_MOL_NM2:-100000}" \
  --reference-minimization-tolerance-kj-mol-nm \
    "${REFERENCE_MINIMIZATION_TOLERANCE_KJ_MOL_NM:-500}" \
  --maximum-reference-atomic-force-kj-mol-nm \
    "${MAXIMUM_REFERENCE_ATOMIC_FORCE_KJ_MOL_NM:-1000000}" \
  --maximum-frame-reference-atomic-force-kj-mol-nm \
    "${MAXIMUM_FRAME_REFERENCE_ATOMIC_FORCE_KJ_MOL_NM:-1000000}"

COMMON_ARGS=(
  --preparation-report "$PREPARATION_REPORT"
  --implicit-cache-dir "$IMPLICIT_CACHE_DIR"
  --platform CUDA
  --device-index "${DEVICE_INDEX:-0}"
  --cpu-threads "${SLURM_CPUS_PER_TASK:-8}"
  --restraint-mode ca
  --frame-initialization reference_cache
  --frame-reference-cache "$FRAME_REFERENCE_CACHE"
  --restraint-k-kj-mol-nm2 "${RESTRAINT_K_KJ_MOL_NM2:-5000}"
  --minimization-tolerance-kj-mol-nm "${MINIMIZATION_TOLERANCE_KJ_MOL_NM:-25}"
  --max-minimization-iterations "${MAX_MINIMIZATION_ITERATIONS:-250}"
  --maximum-relaxed-residue-net-force-kj-mol-nm \
    "${MAXIMUM_RELAXED_RESIDUE_NET_FORCE_KJ_MOL_NM:-500}"
  --maximum-reference-atomic-force-kj-mol-nm \
    "${MAXIMUM_REFERENCE_ATOMIC_FORCE_KJ_MOL_NM:-1000000}"
  --diagnostic-force-components
)

python scripts/evaluate_path4_openmm_gate0.py \
  --candidate "$PATH3_CANDIDATE" \
  --output "$PATH3_SCORE" \
  "${COMMON_ARGS[@]}"

python scripts/evaluate_path4_openmm_gate0.py \
  --candidate "$OPTIMIZED_CANDIDATE" \
  --output "$OPTIMIZED_SCORE" \
  "${COMMON_ARGS[@]}"

python scripts/summarize_path4_gate0_pairs.py \
  --path3-glob "$PATH3_SCORE" \
  --candidate-glob "$OPTIMIZED_SCORE" \
  --optimizer-glob "$OPTIMIZER_REPORT" \
  --output "$PAIRED_SUMMARY" \
  --bootstrap-resamples "${BOOTSTRAP_RESAMPLES:-10000}" \
  --maximum-endpoint-energy-difference-kj-mol \
    "${MAXIMUM_ENDPOINT_ENERGY_DIFFERENCE_KJ_MOL:-1.0}"

echo "PATH4_GATE0_PAIRED_RESCORE_OK=$PAIRED_SUMMARY"
