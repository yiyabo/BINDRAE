#!/usr/bin/env bash
# Submit the matched Path-4 v2 block-identification screen.
#
# This screen isolates what each block contributes before any full-model run:
# phase-only, learned phase + rotation, learned phase + rotation/chi, and all
# three residual blocks. All variants use the same strict split and MD cache.

set -euo pipefail

ROOT="${ROOT:-/mnt/inaisfs/data/home/zhaozc_criait/XinxiangWang/BINDRAE}"
BLOCK_CACHE="${BLOCK_CACHE:-processed_data/md_transition/phase_block_cache_sin2_inferred1180_20260718_v2}"
ORACLE_CACHE="${ORACLE_CACHE:-logs/stage2_oracle_motion/oracle_motion_md1185_fam30_canonical_v2e_20260717_v2_merged}"
SUBSET_TAG="${SUBSET_TAG:-mdphase_silver1180_fam30scaf_blockv2}"
SUBSET_SEED="${SUBSET_SEED:-20260718}"
TRAIN_N="${TRAIN_N:-226}"
VAL_N="${VAL_N:-28}"
MAX_EPOCHS="${MAX_EPOCHS:-10}"
EARLY_STOP_PATIENCE="${EARLY_STOP_PATIENCE:-10}"
GPUS_PER_RUN="${GPUS_PER_RUN:-1}"
BATCH_SIZE="${BATCH_SIZE:-2}"
WALLTIME="${WALLTIME:-04:00:00}"
LR="${LR:-3e-4}"
SUPERVISION_WEIGHT="${SUPERVISION_WEIGHT:-1.0}"
PHASE_SUPERVISION_WEIGHT="${PHASE_SUPERVISION_WEIGHT:-${SUPERVISION_WEIGHT}}"
NORMAL_SUPERVISION_WEIGHT="${NORMAL_SUPERVISION_WEIGHT:-${SUPERVISION_WEIGHT}}"
SUPERVISION_REPLICA_MODE="${SUPERVISION_REPLICA_MODE:-cycle}"
TRAIN_SCOPE="${TRAIN_SCOPE:-heads}"
ROTATION_GATE_BIAS="${ROTATION_GATE_BIAS:--2.0}"
TRANSLATION_GATE_BIAS="${TRANSLATION_GATE_BIAS:--6.0}"
CHI_GATE_BIAS="${CHI_GATE_BIAS:--2.0}"
NORMAL_MIN_CONFIDENCE="${NORMAL_MIN_CONFIDENCE:-0.0}"
RUN_VARIANTS="${RUN_VARIANTS:-phase,rotation,rotation_chi,all}"
RUN_VERSION="${RUN_VERSION:-v1}"
DATE_TAG="${DATE_TAG:-$(date +%Y%m%d)}"

export PATH="/data/soft/slurm/24.11.4/bin:${PATH}"
cd "$ROOT"

for required in "$BLOCK_CACHE" "$ORACLE_CACHE"; do
  if [[ ! -d "$required" ]]; then
    echo "ERROR: required cache directory is missing: $required" >&2
    exit 2
  fi
done

case "$SUPERVISION_REPLICA_MODE" in
  cycle|first) ;;
  *)
    echo "ERROR: SUPERVISION_REPLICA_MODE must be cycle or first" >&2
    exit 2
    ;;
esac

case "$TRAIN_SCOPE" in
  heads|full) ;;
  *)
    echo "ERROR: TRAIN_SCOPE must be heads or full" >&2
    exit 2
    ;;
esac

variant_enabled() {
  case ",${RUN_VARIANTS}," in
    *",$1,"*) return 0 ;;
    *) return 1 ;;
  esac
}

COMMON_EXPORT="NPROC_PER_NODE=${GPUS_PER_RUN},STAGE1V2_MODE=oracle_motion,STAGE1V2_TRAIN_CACHE_DIR=${ORACLE_CACHE},STAGE1V2_VAL_CACHE_DIR=${ORACLE_CACHE},VAL_SPLIT=train,ESM_FUSION_ENABLED=1,ESM_NUM_LAYERS=7,ESM_FUSION_MODE=gated_residual,ESM_GATE_BIAS=-2.0,ESM_GATE_CONTEXT_MODE=pocket_motion,PATH_PARAMETERIZATION=phase_block_orthogonal_residual_v2,PHASE_RESIDUAL_TAU_MODE=learned,PHASE_RESIDUAL_BRIDGE_MODE=cartesian_backbone,PHASE_RESIDUAL_ENVELOPE=sin2,PHASE_RESIDUAL_ROTATION_GATE_BIAS=${ROTATION_GATE_BIAS},PHASE_RESIDUAL_TRANSLATION_GATE_BIAS=${TRANSLATION_GATE_BIAS},PHASE_RESIDUAL_CHI_GATE_BIAS=${CHI_GATE_BIAS},PHASE_RESIDUAL_ROTATION_METRIC_SCALE=1.0,PHASE_RESIDUAL_TRANSLATION_METRIC_SCALE=1.0,PHASE_RESIDUAL_CHI_METRIC_SCALE=1.0,PHASE_RESIDUAL_MIN_TANGENT_NORM=0.5,PHASE_RESIDUAL_MAX_METRIC_NORM=5.0,PHASE_TEACHER_CACHE_DIR=${BLOCK_CACHE},W_PHASE_TEACHER=${PHASE_SUPERVISION_WEIGHT},PHASE_TEACHER_MASK_MODE=active,PHASE_TEACHER_MIN_CONFIDENCE=0.05,PHASE_TEACHER_MISSING_POLICY=error,PHASE_NORMAL_CACHE_DIR=${BLOCK_CACHE},PHASE_NORMAL_MISSING_POLICY=error,PHASE_NORMAL_RESIDUAL_MIN_CONFIDENCE=${NORMAL_MIN_CONFIDENCE},PHASE_NORMAL_RESIDUAL_RIGID_WEIGHT=1.0,PHASE_NORMAL_RESIDUAL_CHI_WEIGHT=1.0,SUPERVISION_REPLICA_MODE=${SUPERVISION_REPLICA_MODE},TRAIN_N=${TRAIN_N},VAL_N=${VAL_N},SUBSET_SEED=${SUBSET_SEED},SUBSET_TAG=${SUBSET_TAG},USE_EXISTING_SUBSETS=1,TRUST_PRECHECKED_SAMPLES=0,MAX_EPOCHS=${MAX_EPOCHS},EARLY_STOP_PATIENCE=${EARLY_STOP_PATIENCE},BATCH_SIZE=${BATCH_SIZE},VAL_BATCH_SIZE=1,NUM_WORKERS=4,PREFETCH_FACTOR=4,LR=${LR},WARMUP_STEPS=0,N_INTEGRATION_STEPS=3,N_GEOM_STEPS=4,GEOM_EVERY=1,W_FM_CHI=0.0,W_FM_RIGID=0.0,W_BG=0.0,W_SMOOTH=0.0,W_CLASH=0.0,W_PEP=0.0,W_CONTACT=0.0,W_END=0.0,W_PHASE_RESIDUAL_MAGNITUDE=0.0,W_PHASE_RESIDUAL_TEMPORAL_SMOOTH=0.0,W_PHASE_RESIDUAL_NEIGHBOR_SMOOTH=0.0,REPA_ENABLED=0,AUTO_RESUME=0,CHECKPOINT_EVERY_N_EPOCHS=1,PROGRESS_LOG_EVERY=25,GPU_MONITOR_INTERVAL=30"

submit_variant() {
  local variant="$1"
  local active_blocks="$2"
  local residual_scale="$3"
  local normal_weight="$4"
  local phase_head_only="$5"
  local phase_residual_heads_only="$6"
  if [[ "$TRAIN_SCOPE" == "full" ]]; then
    phase_head_only=0
    phase_residual_heads_only=0
  fi
  local tag="stage2_pbv2_${variant}_${TRAIN_SCOPE}_train${TRAIN_N}_val${VAL_N}_e${MAX_EPOCHS}_bs${BATCH_SIZE}x${GPUS_PER_RUN}_${DATE_TAG}_${RUN_VERSION}"
  local job_id

  job_id=$(sbatch \
    --parsable \
    --job-name="pbv2_${variant}" \
    --gres="gpu:A100:${GPUS_PER_RUN}" \
    --cpus-per-task=8 \
    --mem=100G \
    --time="$WALLTIME" \
    --export="ALL,${COMMON_EXPORT},PHASE_RESIDUAL_ACTIVE_BLOCKS=${active_blocks},PHASE_RESIDUAL_SCALE=${residual_scale},W_PHASE_NORMAL_RESIDUAL=${normal_weight},PHASE_TEACHER_HEAD_ONLY=${phase_head_only},PHASE_TEACHER_RESIDUAL_HEADS_ONLY=${phase_residual_heads_only},TAG=${tag}" \
    scripts/slurm/train_stage2_oracle_motion_ablation_4gpu.sh)
  printf '%s\t%s\t%s\n' "$variant" "$job_id" "$tag"
}

printf 'variant\tjob_id\ttag\n'
if variant_enabled phase; then
  submit_variant phase rotation_chi 0.0 0.0 1 0
fi
if variant_enabled rotation; then
  submit_variant rotation rotation 1.0 "$NORMAL_SUPERVISION_WEIGHT" 0 1
fi
if variant_enabled rotation_chi; then
  submit_variant rotation_chi rotation_chi 1.0 "$NORMAL_SUPERVISION_WEIGHT" 0 1
fi
if variant_enabled all; then
  submit_variant all all 1.0 "$NORMAL_SUPERVISION_WEIGHT" 0 1
fi
