#!/usr/bin/env bash
# Submit the current-corpus Path-3 anchor and matched internal controls.

set -euo pipefail

ROOT="${ROOT:-/mnt/inaisfs/data/home/zhaozc_criait/XinxiangWang/BINDRAE}"
MODE="${MODE:-plan}"
MATRIX="${MATRIX:-candidate}"
SEEDS="${SEEDS:-7 42 137}"
GPUS="${GPUS:-4}"
PRECHECK_GPUS="${PRECHECK_GPUS:-1}"
BATCH_SIZE="${BATCH_SIZE:-4}"
TARGET_GLOBAL_BATCH="${TARGET_GLOBAL_BATCH:-16}"
MAX_EPOCHS="${MAX_EPOCHS:-40}"
EARLY_STOP_PATIENCE="${EARLY_STOP_PATIENCE:-10}"
LR="${LR:-1e-4}"
WALLTIME="${WALLTIME:-12:00:00}"
RUN_VERSION="${RUN_VERSION:-path3_final_budget_v1}"
DATE_TAG="${DATE_TAG:-$(date +%Y%m%d)}"
CONFIRM_LAUNCH="${CONFIRM_LAUNCH:-0}"
ALLOW_GLOBAL_BATCH_CHANGE="${ALLOW_GLOBAL_BATCH_CHANGE:-0}"
ALLOW_TRAIN241_ANCHOR="${ALLOW_TRAIN241_ANCHOR:-0}"

case "$MODE" in
  plan|precheck|smoke|train) ;;
  *)
    echo "ERROR: MODE must be plan, precheck, smoke, or train; got $MODE" >&2
    exit 2
    ;;
esac

case "$MATRIX" in
  candidate)
    WARP_VARIANTS=(chain_nonmonotone)
    RUN_VARIANTS="phase"
    ;;
  phase_controls)
    WARP_VARIANTS=(
      global_monotone
      residue_monotone
      residue_nonmonotone
      chain_nonmonotone
    )
    RUN_VARIANTS="phase"
    ;;
  decomposition)
    WARP_VARIANTS=(chain_nonmonotone)
    RUN_VARIANTS="residual,full"
    ;;
  *)
    echo "ERROR: MATRIX must be candidate, phase_controls, or decomposition; got $MATRIX" >&2
    exit 2
    ;;
esac

for value_name in GPUS PRECHECK_GPUS BATCH_SIZE TARGET_GLOBAL_BATCH MAX_EPOCHS EARLY_STOP_PATIENCE; do
  value="${!value_name}"
  if ! [[ "$value" =~ ^[1-9][0-9]*$ ]]; then
    echo "ERROR: $value_name must be a positive integer; got $value" >&2
    exit 2
  fi
done

read -r -a seed_values <<< "$SEEDS"
if [[ "${#seed_values[@]}" -eq 0 ]]; then
  echo "ERROR: SEEDS must contain at least one integer" >&2
  exit 2
fi
for seed in "${seed_values[@]}"; do
  if ! [[ "$seed" =~ ^[0-9]+$ ]]; then
    echo "ERROR: every seed must be a non-negative integer; got $seed" >&2
    exit 2
  fi
done

global_batch=$((GPUS * BATCH_SIZE))
if [[ "$global_batch" -ne "$TARGET_GLOBAL_BATCH" && "$ALLOW_GLOBAL_BATCH_CHANGE" != "1" ]]; then
  echo "ERROR: GPUS x BATCH_SIZE = $global_batch, expected frozen global batch $TARGET_GLOBAL_BATCH" >&2
  echo "Set ALLOW_GLOBAL_BATCH_CHANGE=1 only for an explicitly separate optimization study." >&2
  exit 2
fi

print_plan() {
  echo "Path-3 current-corpus anchor training plan"
  echo "  root:             $ROOT"
  echo "  mode:             $MODE"
  echo "  matrix:           $MATRIX"
  echo "  warp variants:    ${WARP_VARIANTS[*]}"
  echo "  trained branches: $RUN_VARIANTS"
  echo "  seeds:            ${seed_values[*]}"
  echo "  GPUs/run:         $GPUS"
  if [[ "$MODE" == "precheck" ]]; then
    echo "  precheck GPUs:    $PRECHECK_GPUS"
  fi
  echo "  batch/GPU:        $BATCH_SIZE"
  echo "  global batch:     $global_batch"
  echo "  max epochs:       $MAX_EPOCHS"
  echo "  early stop:       $EARLY_STOP_PATIENCE"
  echo "  learning rate:    $LR"
  echo "  walltime/run:     $WALLTIME"
  echo "  run version:      $RUN_VERSION"
  echo "  train241 allowed: $ALLOW_TRAIN241_ANCHOR"
}

print_plan
if [[ "$MODE" == "plan" ]]; then
  exit 0
fi

if [[ "$MODE" == "smoke" || "$MODE" == "train" ]]; then
  if [[ "$CONFIRM_LAUNCH" != "1" ]]; then
    echo "ERROR: set CONFIRM_LAUNCH=1 to submit GPU training jobs" >&2
    exit 2
  fi
fi
if [[ "$MODE" == "train" && "$ALLOW_TRAIN241_ANCHOR" != "1" ]]; then
  echo "ERROR: the Path-3 data-scale gate is open; train241 is an anchor, not the final model." >&2
  echo "Set ALLOW_TRAIN241_ANCHOR=1 only for the predeclared learning-curve anchor." >&2
  exit 2
fi

export PATH="/data/soft/slurm/24.11.4/bin:${PATH}"
cd "$ROOT"

submit_one() {
  local variant="$1"
  local seed="$2"
  local run_variants="$3"
  local epochs="$4"
  local precheck_only="$5"
  local requested_gpus="$GPUS"
  local version="$RUN_VERSION"
  if [[ "$MODE" == "precheck" ]]; then
    version="${RUN_VERSION}_precheck"
    requested_gpus="$PRECHECK_GPUS"
  elif [[ "$MODE" == "smoke" ]]; then
    version="${RUN_VERSION}_smoke"
  fi

  PHASE_WARP_VARIANT="$variant" \
  PHASE_CHAIN_RESIDUAL_SCALE=1.0 \
  PHASE_CHAIN_SMOOTHING_STEPS=1 \
  TRAIN_SEED="$seed" \
  RUN_VARIANTS="$run_variants" \
  RUN_VERSION="$version" \
  DATE_TAG="$DATE_TAG" \
  GPUS="$requested_gpus" \
  BATCH_SIZE="$BATCH_SIZE" \
  MAX_EPOCHS="$epochs" \
  EARLY_STOP_PATIENCE="$EARLY_STOP_PATIENCE" \
  LR="$LR" \
  WALLTIME="$WALLTIME" \
  PRECHECK_ONLY="$precheck_only" \
    bash scripts/slurm/submit_stage2_md_phase_normal_screen.sh
}

if [[ "$MODE" == "precheck" ]]; then
  # One branch is sufficient because the shared launcher validates both target
  # caches and the exact train/validation subsets before training starts.
  submit_one "${WARP_VARIANTS[0]}" "${seed_values[0]}" "phase" "$MAX_EPOCHS" 1
  exit 0
fi

if [[ "$MODE" == "smoke" ]]; then
  submit_one "${WARP_VARIANTS[0]}" "${seed_values[0]}" "phase" 1 0
  exit 0
fi

for seed in "${seed_values[@]}"; do
  for variant in "${WARP_VARIANTS[@]}"; do
    submit_one "$variant" "$seed" "$RUN_VARIANTS" "$MAX_EPOCHS" 0
  done
done
