#!/bin/bash
# End-to-end engineering smoke: Path-3 export -> GBn2 OpenMM relaxation.
# Uses one frozen train system only. It must never be reported as Gate-0 evidence.

#SBATCH --job-name=p4g0_cpu_smoke
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=logs/slurm/path4_gate0_dev_smoke_%j.out
#SBATCH --error=logs/slurm/path4_gate0_dev_smoke_%j.err

set -euo pipefail
unset LD_PRELOAD

ROOT="${ROOT:-/mnt/inaisfs/data/home/zhaozc_criait/XinxiangWang/BINDRAE}"
SAMPLE_ID="${SAMPLE_ID:-3zw1-E-FUC-3}"
CANDIDATE_MODE="${CANDIDATE_MODE:-path3}"
CANDIDATE_LABEL="${CANDIDATE_LABEL:-frozen_path3_cummax_dev_smoke}"
CHECKPOINT="${CHECKPOINT:-checkpoints/stage2/stage2_pbv2_phase_cons303_fam30scaf_e10_bs8x2_20260718_canonical_fromscratch_v2/best_model.pt}"
ORACLE_CACHE_DIR="${ORACLE_CACHE_DIR:-logs/stage2_oracle_motion/oracle_motion_mdphase_consensus303_canonical_v2e_20260718_v1/merged}"
PHASE_CACHE_DIR="${PHASE_CACHE_DIR:-logs/stage2/physical_normal_targets_cons303_rw20_20260719_v1}"
PREPARATION_REPORT="${PREPARATION_REPORT:-processed_data/md_transition/context_full500_endpointdedup_20260716_v2/systems/${SAMPLE_ID}/setup/preparation_report.json}"
TAG="${TAG:-path4_gate0_dev_${SAMPLE_ID}_${CANDIDATE_MODE}_${SLURM_JOB_ID:-manual}}"
OUTPUT_ROOT="${OUTPUT_ROOT:-logs/stage2/path4_gate0/$TAG}"
SUBSET_FILE="$OUTPUT_ROOT/${SAMPLE_ID}.txt"
EXPORT_DIR="$OUTPUT_ROOT/candidates/$CANDIDATE_MODE"
CANDIDATE="$EXPORT_DIR/${SAMPLE_ID}.npz"
IMPLICIT_CACHE_DIR="${IMPLICIT_CACHE_DIR:-processed_data/md_transition/path4_gate0_implicit/${SAMPLE_ID}_gbn2_v1}"
FRAME_REFERENCE_CACHE="$OUTPUT_ROOT/reference/path3_all_atom_reference.npz"
FRAME_REFERENCE_REPORT="$OUTPUT_ROOT/reference/report.json"
RESULT="$OUTPUT_ROOT/openmm/${CANDIDATE_MODE}.json"

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"

cd "$ROOT"
mkdir -p logs/slurm "$EXPORT_DIR" "$OUTPUT_ROOT/reference" "$(dirname "$RESULT")"
python3 -c 'import pathlib,sys; pathlib.Path(sys.argv[1]).write_text(sys.argv[2] + "\n")' "$SUBSET_FILE" "$SAMPLE_ID"

source /mnt/inaisfs/data/home/zhaozc_criait/miniconda3/etc/profile.d/conda.sh
if [[ "${SKIP_EXPORT:-0}" == "1" ]]; then
  if [[ ! -f "$CANDIDATE" ]]; then
    echo "Missing pre-exported candidate: $CANDIDATE" >&2
    exit 1
  fi
else
  conda activate BINDRAE
  python scripts/export_cached_phase_path_candidate.py \
    --checkpoint "$CHECKPOINT" \
    --data-dir processed_data/triplets \
    --split train \
    --valid-samples-file "$SUBSET_FILE" \
    --phase-cache-dir "$PHASE_CACHE_DIR" \
    --device cpu \
    --output-dir "$EXPORT_DIR" \
    --manifest "$OUTPUT_ROOT/${CANDIDATE_MODE}_manifest.json" \
    --candidate-label "$CANDIDATE_LABEL" \
    --candidate-mode "$CANDIDATE_MODE" \
    --phase-tau-postprocess cummax \
    --max-samples 1
  conda deactivate
fi
conda activate BINDRAE-MD

python scripts/build_path4_openmm_frame_reference.py \
  --candidate "$CANDIDATE" \
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
  --candidate "$CANDIDATE" \
  --preparation-report "$PREPARATION_REPORT" \
  --implicit-cache-dir "$IMPLICIT_CACHE_DIR" \
  --output "$RESULT" \
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

echo "DEV_SMOKE_OK=$RESULT"
