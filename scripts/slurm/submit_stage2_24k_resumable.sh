#!/bin/bash
# Submit/resubmit the 24k Stage-2 OracleMotion ESM7 ablation matrix with stable
# checkpoint directories. Re-running the same variant resumes from
# checkpoints/stage2/<TAG>/last_checkpoint.pt through AUTO_RESUME=1.

set -euo pipefail

ROOT="${ROOT:-/mnt/inaisfs/data/home/zhaozc_criait/XinxiangWang/BINDRAE}"
LAUNCHER="${LAUNCHER:-scripts/slurm/train_stage2_oracle_motion_ablation_4gpu.sh}"

TRAIN_CACHE="${TRAIN_CACHE:-logs/stage2_oracle_motion/oracle_motion_train24000_esm7sync_20260629}"
VAL_CACHE="${VAL_CACHE:-logs/stage2_oracle_motion/oracle_motion_val2048_esm7sync_20260629}"
SUBSET_TAG="${SUBSET_TAG:-esm7sync20260629}"
SUBSET_SEED="${SUBSET_SEED:-20260629}"

VARIANT="${1:-}"
if [[ -z "$VARIANT" ]]; then
  cat >&2 <<'USAGE'
Usage: scripts/slurm/submit_stage2_24k_resumable.sh <variant>

Variants:
  single  - single ESM/noREPA baseline, resumes the interrupted 138292 run
  gm2     - gated_residual bias=-2.0/noREPA, resumes the interrupted 138361 run
  gm15    - gated_residual bias=-1.5/noREPA, resumes the interrupted 138294 run
  grm     - gated_residual bias=-2.0 matched REPA, resumes the interrupted 138295 run
  grs     - gated_residual bias=-2.0 shuffled REPA control, resumes the interrupted 138296 run

This wrapper intentionally fixes TAG/SAVE_DIR/LOG_DIR per variant. Do not add a
timestamp when resubmitting an interrupted run; otherwise auto-resume will not
find last_checkpoint.pt.
USAGE
  exit 2
fi

COMMON_EXPORTS=(
  ALL
  "ROOT=$ROOT"
  "STAGE1V2_MODE=oracle_motion"
  "STAGE1V2_TRAIN_CACHE_DIR=$TRAIN_CACHE"
  "STAGE1V2_VAL_CACHE_DIR=$VAL_CACHE"
  "TRAIN_N=24000"
  "VAL_N=2048"
  "MAX_EPOCHS=10"
  "BATCH_SIZE=4"
  "NPROC_PER_NODE=4"
  "NUM_WORKERS=2"
  "SUBSET_TAG=$SUBSET_TAG"
  "SUBSET_SEED=$SUBSET_SEED"
  "USE_EXISTING_SUBSETS=1"
  "AUTO_RESUME=1"
  "ESM_NUM_LAYERS=7"
  "ESM_FUSION_MODE=gated_residual"
  "ESM_GATE_CONTEXT_MODE=none"
)

case "$VARIANT" in
  single)
    JOB_NAME="s2_24_single_resume"
    TAG="stage2_stage2_24k_single_norepa_esm7sync_train24000_val2048_e10_bs4x4_20260629_204455"
    EXTRA_EXPORTS=(
      "ESM_FUSION_ENABLED=0"
      "ESM_LAYER_ENTROPY_WEIGHT=0.01"
      "ESM_GATE_BIAS=-3.0"
      "REPA_ENABLED=0"
      "REPA_WEIGHT=0.0"
    )
    ;;
  gm2)
    JOB_NAME="s2_24_gm2_resume"
    TAG="stage2_stage2_24k_gm2_gated_m2_norepa_esm7sync_train24000_val2048_e10_bs4x4_20260630_054826"
    EXTRA_EXPORTS=(
      "ESM_FUSION_ENABLED=1"
      "ESM_LAYER_ENTROPY_WEIGHT=0.02"
      "ESM_GATE_BIAS=-2.0"
      "REPA_ENABLED=0"
      "REPA_WEIGHT=0.0"
    )
    ;;
  gm15)
    JOB_NAME="s2_24_gm15_resume"
    TAG="stage2_stage2_24k_gated_bm15_norepa_esm7sync_train24000_val2048_e10_bs4x4_20260630_015555"
    EXTRA_EXPORTS=(
      "ESM_FUSION_ENABLED=1"
      "ESM_LAYER_ENTROPY_WEIGHT=0.01"
      "ESM_GATE_BIAS=-1.5"
      "REPA_ENABLED=0"
      "REPA_WEIGHT=0.0"
    )
    ;;
  grm)
    JOB_NAME="s2_24_grm_resume"
    TAG="stage2_stage2_24k_gm2_repa_mcont_esm7sync_train24000_val2048_e10_bs4x4_20260630_043832"
    EXTRA_EXPORTS=(
      "ESM_FUSION_ENABLED=1"
      "ESM_LAYER_ENTROPY_WEIGHT=0.02"
      "ESM_GATE_BIAS=-2.0"
      "REPA_ENABLED=1"
      "REPA_WEIGHT=0.05"
      "REPA_TARGET_MODE=motion_context"
      "REPA_TARGET_SHUFFLE_MODE=none"
    )
    ;;
  grs)
    JOB_NAME="s2_24_grs_resume"
    TAG="stage2_stage2_24k_gm2_repa_mcont_shuf_esm7sync_train24000_val2048_e10_bs4x4_20260630_050036"
    EXTRA_EXPORTS=(
      "ESM_FUSION_ENABLED=1"
      "ESM_LAYER_ENTROPY_WEIGHT=0.02"
      "ESM_GATE_BIAS=-2.0"
      "REPA_ENABLED=1"
      "REPA_WEIGHT=0.05"
      "REPA_TARGET_MODE=motion_context"
      "REPA_TARGET_SHUFFLE_MODE=residue"
    )
    ;;
  *)
    echo "ERROR: unknown variant '$VARIANT'" >&2
    exit 2
    ;;
esac

SAVE_DIR="checkpoints/stage2/$TAG"
LOG_DIR="logs/stage2/$TAG"
EXPORTS=("${COMMON_EXPORTS[@]}" "${EXTRA_EXPORTS[@]}" "TAG=$TAG" "SAVE_DIR=$SAVE_DIR" "LOG_DIR=$LOG_DIR")
EXPORT_ARG=$(IFS=,; echo "${EXPORTS[*]}")

cd "$ROOT"
echo "Submitting resumable 24k Stage-2 variant: $VARIANT"
echo "  tag:      $TAG"
echo "  save_dir: $SAVE_DIR"
echo "  log_dir:  $LOG_DIR"
echo "  resume:   $SAVE_DIR/last_checkpoint.pt if present"

sbatch \
  --job-name="$JOB_NAME" \
  --gres=gpu:A100:4 \
  --cpus-per-task=32 \
  --mem=300G \
  --export="$EXPORT_ARG" \
  "$LAUNCHER"
