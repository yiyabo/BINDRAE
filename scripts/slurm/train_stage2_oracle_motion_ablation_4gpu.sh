#!/bin/bash
# Stage-2 ablation with OracleMotion-UB feature caches.
#
# Required:
#   STAGE1V2_TRAIN_CACHE_DIR=logs/stage2_oracle_motion/<train_export>
#   STAGE1V2_VAL_CACHE_DIR=logs/stage2_oracle_motion/<val_export>
#
# Typical controls:
#   sbatch --export=ALL,STAGE1V2_MODE=zero,STAGE1V2_TRAIN_CACHE_DIR=...,STAGE1V2_VAL_CACHE_DIR=... scripts/slurm/train_stage2_oracle_motion_ablation_4gpu.sh
#   sbatch --export=ALL,STAGE1V2_MODE=oracle_motion,STAGE1V2_TRAIN_CACHE_DIR=...,STAGE1V2_VAL_CACHE_DIR=... scripts/slurm/train_stage2_oracle_motion_ablation_4gpu.sh
#   sbatch --export=ALL,STAGE1V2_MODE=oracle_motion_residue_shuffled,STAGE1V2_TRAIN_CACHE_DIR=...,STAGE1V2_VAL_CACHE_DIR=... scripts/slurm/train_stage2_oracle_motion_ablation_4gpu.sh

#SBATCH --job-name=s2_omotion_ablate
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:A100:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=200G
#SBATCH --time=18:00:00
#SBATCH --output=logs/slurm/stage2_oracle_motion_ablation_%j.out
#SBATCH --error=logs/slurm/stage2_oracle_motion_ablation_%j.err

set -euo pipefail
unset LD_PRELOAD
unset PROXYCHAINS_CONF_FILE
unset PROXYCHAINS_QUIET_MODE

ROOT="${ROOT:-/mnt/inaisfs/data/home/zhaozc_criait/XinxiangWang/BINDRAE}"
ENV_PREFIX=/mnt/inaisfs/data/home/zhaozc_criait/miniconda3/envs/BINDRAE

export PATH=/data/soft/slurm/24.11.4/bin:$ENV_PREFIX/bin:$PATH
export PYTHONUNBUFFERED=1
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"

cd "$ROOT"
mkdir -p logs/slurm logs/stage2 checkpoints/stage2 processed_data/triplets/ablation_subsets

source /mnt/inaisfs/data/home/zhaozc_criait/miniconda3/etc/profile.d/conda.sh
conda activate BINDRAE

NPROC_PER_NODE="${NPROC_PER_NODE:-${SLURM_GPUS_ON_NODE:-2}}"
STAGE1V2_MODE="${STAGE1V2_MODE:-oracle_motion}"
STAGE1V2_FEATURES="${STAGE1V2_FEATURES:-delta_trans_local_x_norm,delta_trans_local_y_norm,delta_trans_local_z_norm,delta_rot_log_x_norm,delta_rot_log_y_norm,delta_rot_log_z_norm,delta_chi1_sin,delta_chi2_sin,delta_chi3_sin,delta_chi4_sin,delta_chi1_cos,delta_chi2_cos,delta_chi3_cos,delta_chi4_cos,chi1_mask,chi2_mask,chi3_mask,chi4_mask,trans_mag_norm,rot_angle_norm,max_abs_delta_chi_norm,motion_active,contact_apo,contact_holo,formed_contact,released_contact,signed_delta_dist_norm,w_res}"
STAGE1V2_FEATURE_SCALE="${STAGE1V2_FEATURE_SCALE:-1.0}"
STAGE1V2_TRAIN_CACHE_DIR="${STAGE1V2_TRAIN_CACHE_DIR:-}"
STAGE1V2_VAL_CACHE_DIR="${STAGE1V2_VAL_CACHE_DIR:-logs/stage2_oracle_motion/oracle_motion_val512_direct_20260623_020114}"
TRAIN_N="${TRAIN_N:-512}"
VAL_N="${VAL_N:-512}"
SUBSET_SEED="${SUBSET_SEED:-20260623}"
SUBSET_TAG="${SUBSET_TAG:-}"
MAX_EPOCHS="${MAX_EPOCHS:-3}"
BATCH_SIZE="${BATCH_SIZE:-2}"
VAL_BATCH_SIZE="${VAL_BATCH_SIZE:-$BATCH_SIZE}"
NUM_WORKERS="${NUM_WORKERS:-4}"
PREFETCH_FACTOR="${PREFETCH_FACTOR:-4}"
LENGTH_BUCKETED_TRAIN="${LENGTH_BUCKETED_TRAIN:-0}"
LENGTH_BUCKET_MULTIPLIER="${LENGTH_BUCKET_MULTIPLIER:-8}"
LENGTH_BUCKET_DROP_LAST="${LENGTH_BUCKET_DROP_LAST:-1}"
LENGTH_BUCKET_LENGTHS_FILE="${LENGTH_BUCKET_LENGTHS_FILE:-}"
LENGTH_BUCKET_RESIDUE_BUDGET="${LENGTH_BUCKET_RESIDUE_BUDGET:-}"
PROGRESS_LOG_EVERY="${PROGRESS_LOG_EVERY:-100}"
CHECKPOINT_EVERY_N_EPOCHS="${CHECKPOINT_EVERY_N_EPOCHS:-0}"
LR="${LR:-2e-5}"
WARMUP_STEPS="${WARMUP_STEPS:-0}"
EARLY_STOP_PATIENCE="${EARLY_STOP_PATIENCE:-20}"
VAL_T="${VAL_T:-0.5}"
VAL_SPLIT="${VAL_SPLIT:-val}"
SEED="${SEED:-42}"
AMP_DTYPE="${AMP_DTYPE:-bf16}"
PATH_PARAMETERIZATION="${PATH_PARAMETERIZATION:-boundary_residual_v1}"
BOUNDARY_RESIDUAL_ENVELOPE="${BOUNDARY_RESIDUAL_ENVELOPE:-sin2}"
BOUNDARY_RESIDUAL_SCALE="${BOUNDARY_RESIDUAL_SCALE:-1.0}"
TERMINAL_PROJECTION_SCHEDULE="${TERMINAL_PROJECTION_SCHEDULE:-smootherstep}"
TIME_WARP_LOGIT_SCALE="${TIME_WARP_LOGIT_SCALE:-1.0}"
TIME_WARP_RATE_EPS="${TIME_WARP_RATE_EPS:-1e-3}"
TIME_WARP_RATE_CLIP="${TIME_WARP_RATE_CLIP:-10.0}"
PHASE_RESIDUAL_TAU_MODE="${PHASE_RESIDUAL_TAU_MODE:-learned}"
PHASE_RESIDUAL_BRIDGE_MODE="${PHASE_RESIDUAL_BRIDGE_MODE:-se3_geodesic}"
PHASE_RESIDUAL_ENVELOPE="${PHASE_RESIDUAL_ENVELOPE:-poly}"
PHASE_RESIDUAL_SCALE="${PHASE_RESIDUAL_SCALE:-1.0}"
PHASE_RESIDUAL_ROTATION_METRIC_SCALE="${PHASE_RESIDUAL_ROTATION_METRIC_SCALE:-1.0}"
PHASE_RESIDUAL_TRANSLATION_METRIC_SCALE="${PHASE_RESIDUAL_TRANSLATION_METRIC_SCALE:-1.0}"
PHASE_RESIDUAL_CHI_METRIC_SCALE="${PHASE_RESIDUAL_CHI_METRIC_SCALE:-1.0}"
PHASE_RESIDUAL_MIN_TANGENT_NORM="${PHASE_RESIDUAL_MIN_TANGENT_NORM:-1e-3}"
PHASE_RESIDUAL_MAX_METRIC_NORM="${PHASE_RESIDUAL_MAX_METRIC_NORM:-0.0}"
PHASE_RESIDUAL_PEPTIDE_RETRACTION="${PHASE_RESIDUAL_PEPTIDE_RETRACTION:-0}"
PHASE_RESIDUAL_PEPTIDE_RETRACTION_ITERATIONS="${PHASE_RESIDUAL_PEPTIDE_RETRACTION_ITERATIONS:-8}"
PHASE_RESIDUAL_PEPTIDE_RETRACTION_RELAXATION="${PHASE_RESIDUAL_PEPTIDE_RETRACTION_RELAXATION:-0.75}"
PHASE_RESIDUAL_PEPTIDE_RETRACTION_ANCHOR_STRENGTH="${PHASE_RESIDUAL_PEPTIDE_RETRACTION_ANCHOR_STRENGTH:-0.02}"
PHASE_RESIDUAL_PEPTIDE_RETRACTION_MAX_TRANSLATION="${PHASE_RESIDUAL_PEPTIDE_RETRACTION_MAX_TRANSLATION:-1.0}"
PHASE_RESIDUAL_PEPTIDE_RETRACTION_ACTIVATION_LOSS_THRESHOLD="${PHASE_RESIDUAL_PEPTIDE_RETRACTION_ACTIVATION_LOSS_THRESHOLD:-0.0}"
INIT_FROM_CHECKPOINT="${INIT_FROM_CHECKPOINT:-}"
TEACHER_RESIDUAL_CACHE_DIR="${TEACHER_RESIDUAL_CACHE_DIR:-}"
W_TEACHER_RESIDUAL="${W_TEACHER_RESIDUAL:-0.0}"
TEACHER_RESIDUAL_LOSS_TYPE="${TEACHER_RESIDUAL_LOSS_TYPE:-mse}"
TEACHER_RESIDUAL_HUBER_DELTA="${TEACHER_RESIDUAL_HUBER_DELTA:-1.0}"
TEACHER_RESIDUAL_T_MIN="${TEACHER_RESIDUAL_T_MIN:-0.08}"
TEACHER_RESIDUAL_T_MAX="${TEACHER_RESIDUAL_T_MAX:-0.92}"
TEACHER_RESIDUAL_MASK_MODE="${TEACHER_RESIDUAL_MASK_MODE:-motion_active_or_pocket}"
TEACHER_RESIDUAL_CLASH_WEIGHT_THRESHOLD="${TEACHER_RESIDUAL_CLASH_WEIGHT_THRESHOLD:-1e-4}"
TEACHER_RESIDUAL_MISSING_POLICY="${TEACHER_RESIDUAL_MISSING_POLICY:-error}"
PHASE_TEACHER_CACHE_DIR="${PHASE_TEACHER_CACHE_DIR:-}"
W_PHASE_TEACHER="${W_PHASE_TEACHER:-0.0}"
PHASE_TEACHER_LOSS_TYPE="${PHASE_TEACHER_LOSS_TYPE:-huber}"
PHASE_TEACHER_HUBER_DELTA="${PHASE_TEACHER_HUBER_DELTA:-0.1}"
PHASE_TEACHER_MASK_MODE="${PHASE_TEACHER_MASK_MODE:-contact_event}"
PHASE_TEACHER_MIN_CONFIDENCE="${PHASE_TEACHER_MIN_CONFIDENCE:-0.05}"
PHASE_TEACHER_MISSING_POLICY="${PHASE_TEACHER_MISSING_POLICY:-error}"
PHASE_TEACHER_HEAD_ONLY="${PHASE_TEACHER_HEAD_ONLY:-0}"
PHASE_TEACHER_RESIDUAL_HEADS_ONLY="${PHASE_TEACHER_RESIDUAL_HEADS_ONLY:-0}"
PHASE_NORMAL_CACHE_DIR="${PHASE_NORMAL_CACHE_DIR:-}"
W_PHASE_NORMAL_RESIDUAL="${W_PHASE_NORMAL_RESIDUAL:-0.0}"
PHASE_NORMAL_RESIDUAL_LOSS_TYPE="${PHASE_NORMAL_RESIDUAL_LOSS_TYPE:-huber}"
PHASE_NORMAL_RESIDUAL_HUBER_DELTA="${PHASE_NORMAL_RESIDUAL_HUBER_DELTA:-0.25}"
PHASE_NORMAL_MISSING_POLICY="${PHASE_NORMAL_MISSING_POLICY:-error}"
ESM_FUSION_ENABLED="${ESM_FUSION_ENABLED:-0}"
ESM_NUM_LAYERS="${ESM_NUM_LAYERS:-1}"
ESM_FUSION_MODE="${ESM_FUSION_MODE:-sum}"
ESM_LAYER_DROPOUT="${ESM_LAYER_DROPOUT:-0.0}"
ESM_LAYER_ENTROPY_WEIGHT="${ESM_LAYER_ENTROPY_WEIGHT:-0.0}"
ESM_GATE_BIAS="${ESM_GATE_BIAS:--3.0}"
ESM_GATE_CONTEXT_MODE="${ESM_GATE_CONTEXT_MODE:-none}"
REPA_ENABLED="${REPA_ENABLED:-0}"
REPA_WEIGHT="${REPA_WEIGHT:-0.0}"
REPA_DIM="${REPA_DIM:-128}"
REPA_LOSS_TYPE="${REPA_LOSS_TYPE:-cosine}"
REPA_MASK_MODE="${REPA_MASK_MODE:-motion_active_or_pocket}"
REPA_TARGET_MODE="${REPA_TARGET_MODE:-full}"
REPA_TARGET_SHUFFLE_MODE="${REPA_TARGET_SHUFFLE_MODE:-none}"
RESUME_FROM="${RESUME_FROM:-}"
AUTO_RESUME="${AUTO_RESUME:-1}"
GEOM_EVERY="${GEOM_EVERY:-1}"
N_INTEGRATION_STEPS="${N_INTEGRATION_STEPS:-3}"
INTEGRATION_CHI_CLIP="${INTEGRATION_CHI_CLIP:-1.0}"
INTEGRATION_ROT_CLIP="${INTEGRATION_ROT_CLIP:-0.1}"
INTEGRATION_TRANS_CLIP="${INTEGRATION_TRANS_CLIP:-0.2}"
N_GEOM_STEPS="${N_GEOM_STEPS:-4}"
W_FM_CHI="${W_FM_CHI:-}"
W_FM_RIGID="${W_FM_RIGID:-}"
W_BG="${W_BG:-0.1}"
W_PHASE_RESIDUAL_MAGNITUDE="${W_PHASE_RESIDUAL_MAGNITUDE:-0.01}"
W_PHASE_RESIDUAL_TEMPORAL_SMOOTH="${W_PHASE_RESIDUAL_TEMPORAL_SMOOTH:-0.01}"
W_PHASE_RESIDUAL_NEIGHBOR_SMOOTH="${W_PHASE_RESIDUAL_NEIGHBOR_SMOOTH:-0.01}"
W_SMOOTH="${W_SMOOTH:-0.05}"
W_CLASH="${W_CLASH:-0.1}"
W_LIGAND_CLEARANCE="${W_LIGAND_CLEARANCE:-0.0}"
LIGAND_CLEARANCE_DIST="${LIGAND_CLEARANCE_DIST:-2.2}"
LIGAND_CLEARANCE_MASK_MODE="${LIGAND_CLEARANCE_MASK_MODE:-pocket}"
LIGAND_CLEARANCE_LOSS_MODE="${LIGAND_CLEARANCE_LOSS_MODE:-all}"
LIGAND_CLEARANCE_HARD_NEGATIVE_DIST="${LIGAND_CLEARANCE_HARD_NEGATIVE_DIST:-2.2}"
LIGAND_CLEARANCE_T_MIN="${LIGAND_CLEARANCE_T_MIN:-0.05}"
LIGAND_CLEARANCE_T_MAX="${LIGAND_CLEARANCE_T_MAX:-0.95}"
W_BRIDGE_ANCHOR="${W_BRIDGE_ANCHOR:-0.0}"
BRIDGE_ANCHOR_MASK_MODE="${BRIDGE_ANCHOR_MASK_MODE:-non_clash_node}"
BRIDGE_ANCHOR_T_MIN="${BRIDGE_ANCHOR_T_MIN:-0.05}"
BRIDGE_ANCHOR_T_MAX="${BRIDGE_ANCHOR_T_MAX:-0.95}"
W_PEP="${W_PEP:-0.1}"
W_CONTACT="${W_CONTACT:-0.1}"
W_END="${W_END:-0.1}"
TAG_SUFFIX="${TAG_SUFFIX:-omotion_${STAGE1V2_MODE}_${PATH_PARAMETERIZATION}}"
PRECHECK_ONLY="${PRECHECK_ONLY:-0}"
STRICT_CACHE_NPZ_PRECHECK="${STRICT_CACHE_NPZ_PRECHECK:-0}"
AATYPE_PRECHECK="${AATYPE_PRECHECK:-1}"
NODE_MASK_PRECHECK="${NODE_MASK_PRECHECK:-1}"
USE_EXISTING_SUBSETS="${USE_EXISTING_SUBSETS:-0}"
TRUST_PRECHECKED_SAMPLES="${TRUST_PRECHECKED_SAMPLES:-$USE_EXISTING_SUBSETS}"
PRECHECK_WORKERS="${PRECHECK_WORKERS:-8}"
GPU_MONITOR_INTERVAL="${GPU_MONITOR_INTERVAL:-30}"

case "$STAGE1V2_MODE" in
  zero|oracle_motion|oracle_motion_residue_shuffled|oracle_motion_sample_shuffled) ;;
  *)
    echo "ERROR: STAGE1V2_MODE must be one of zero, oracle_motion, oracle_motion_residue_shuffled, oracle_motion_sample_shuffled"
    exit 1
    ;;
esac

case "$ESM_FUSION_ENABLED" in
  0|1) ;;
  *)
    echo "ERROR: ESM_FUSION_ENABLED must be 0 or 1"
    exit 1
    ;;
esac
if [[ "$ESM_FUSION_ENABLED" == "1" && "$ESM_NUM_LAYERS" -lt 1 ]]; then
  echo "ERROR: ESM_NUM_LAYERS must be >= 1"
  exit 1
fi
case "$ESM_FUSION_MODE" in
  sum|mean|softmax_weighted|gated_residual) ;;
  *)
    echo "ERROR: ESM_FUSION_MODE must be one of sum, mean, softmax_weighted, gated_residual"
    exit 1
    ;;
esac
case "$PATH_PARAMETERIZATION" in
  flow|boundary_residual_v1|boundary_residual|projected_flow|bridge_timewarp_v1|phase_orthogonal_residual_v1) ;;
  *)
    echo "ERROR: unsupported PATH_PARAMETERIZATION=$PATH_PARAMETERIZATION"
    exit 1
    ;;
esac
if [[ -z "$W_FM_CHI" ]]; then
  case "$PATH_PARAMETERIZATION" in
    boundary_residual_v1|boundary_residual|bridge_timewarp_v1|phase_orthogonal_residual_v1)
      W_FM_CHI="0.1"
      ;;
    *)
      W_FM_CHI="1.0"
      ;;
  esac
fi
if [[ -z "$W_FM_RIGID" ]]; then
  case "$PATH_PARAMETERIZATION" in
    boundary_residual_v1|boundary_residual|bridge_timewarp_v1|phase_orthogonal_residual_v1)
      W_FM_RIGID="0.1"
      ;;
    *)
      W_FM_RIGID="1.0"
      ;;
  esac
fi
case "$BOUNDARY_RESIDUAL_ENVELOPE" in
  sin2|poly) ;;
  *)
    echo "ERROR: BOUNDARY_RESIDUAL_ENVELOPE must be sin2 or poly"
    exit 1
    ;;
esac
case "$TEACHER_RESIDUAL_MASK_MODE" in
  node|pocket|motion_active|motion_active_or_pocket|clash_relief|clash_relief_or_motion_active|clash_relief_or_pocket) ;;
  *)
    echo "ERROR: TEACHER_RESIDUAL_MASK_MODE must be node, pocket, motion_active, motion_active_or_pocket, clash_relief, clash_relief_or_motion_active, or clash_relief_or_pocket"
    exit 1
    ;;
esac
python - <<PY
threshold = float("$TEACHER_RESIDUAL_CLASH_WEIGHT_THRESHOLD")
if threshold < 0.0:
    raise SystemExit("ERROR: TEACHER_RESIDUAL_CLASH_WEIGHT_THRESHOLD must be >= 0")
PY
case "$TEACHER_RESIDUAL_MISSING_POLICY" in
  error|skip) ;;
  *)
    echo "ERROR: TEACHER_RESIDUAL_MISSING_POLICY must be error or skip"
    exit 1
    ;;
esac
case "$TEACHER_RESIDUAL_LOSS_TYPE" in
  mse|huber) ;;
  *)
    echo "ERROR: TEACHER_RESIDUAL_LOSS_TYPE must be mse or huber"
    exit 1
    ;;
esac
python - <<PY
delta = float("$TEACHER_RESIDUAL_HUBER_DELTA")
if delta <= 0.0:
    raise SystemExit("ERROR: TEACHER_RESIDUAL_HUBER_DELTA must be > 0")
PY
if python - <<PY
import sys
sys.exit(0 if float("$W_TEACHER_RESIDUAL") > 0.0 else 1)
PY
then
  if [[ -z "$TEACHER_RESIDUAL_CACHE_DIR" ]]; then
    echo "ERROR: W_TEACHER_RESIDUAL > 0 requires TEACHER_RESIDUAL_CACHE_DIR"
    exit 1
  fi
  case "$PATH_PARAMETERIZATION" in
    boundary_residual_v1|boundary_residual) ;;
    *)
      echo "ERROR: teacher residual distillation requires boundary_residual path mode"
      exit 1
      ;;
  esac
fi
case "$TERMINAL_PROJECTION_SCHEDULE" in
  smoothstep|smootherstep|late_smoother|quadratic) ;;
  *)
    echo "ERROR: TERMINAL_PROJECTION_SCHEDULE must be smoothstep, smootherstep, late_smoother, or quadratic"
    exit 1
    ;;
esac
python - <<PY
logit_scale = float("$TIME_WARP_LOGIT_SCALE")
rate_eps = float("$TIME_WARP_RATE_EPS")
rate_clip = float("$TIME_WARP_RATE_CLIP")
if logit_scale <= 0.0:
    raise SystemExit("ERROR: TIME_WARP_LOGIT_SCALE must be > 0")
if rate_eps <= 0.0:
    raise SystemExit("ERROR: TIME_WARP_RATE_EPS must be > 0")
if rate_clip < 0.0:
    raise SystemExit("ERROR: TIME_WARP_RATE_CLIP must be >= 0")
PY
case "$PHASE_RESIDUAL_TAU_MODE" in
  learned|identity) ;;
  *)
    echo "ERROR: PHASE_RESIDUAL_TAU_MODE must be learned or identity"
    exit 1
    ;;
esac
case "$PHASE_RESIDUAL_BRIDGE_MODE" in
  se3_geodesic|cartesian_backbone) ;;
  *)
    echo "ERROR: PHASE_RESIDUAL_BRIDGE_MODE must be se3_geodesic or cartesian_backbone"
    exit 1
    ;;
esac
case "$PHASE_RESIDUAL_ENVELOPE" in
  poly|sin2) ;;
  *)
    echo "ERROR: PHASE_RESIDUAL_ENVELOPE must be poly or sin2"
    exit 1
    ;;
esac
python - <<PY
positive = {
    "PHASE_RESIDUAL_SCALE": float("$PHASE_RESIDUAL_SCALE"),
    "PHASE_RESIDUAL_ROTATION_METRIC_SCALE": float("$PHASE_RESIDUAL_ROTATION_METRIC_SCALE"),
    "PHASE_RESIDUAL_TRANSLATION_METRIC_SCALE": float("$PHASE_RESIDUAL_TRANSLATION_METRIC_SCALE"),
    "PHASE_RESIDUAL_CHI_METRIC_SCALE": float("$PHASE_RESIDUAL_CHI_METRIC_SCALE"),
}
for name, value in positive.items():
    if value <= 0.0:
        raise SystemExit(f"ERROR: {name} must be > 0")
if float("$PHASE_RESIDUAL_MIN_TANGENT_NORM") < 0.0:
    raise SystemExit("ERROR: PHASE_RESIDUAL_MIN_TANGENT_NORM must be >= 0")
if float("$PHASE_RESIDUAL_MAX_METRIC_NORM") < 0.0:
    raise SystemExit("ERROR: PHASE_RESIDUAL_MAX_METRIC_NORM must be >= 0")
if int("$PHASE_RESIDUAL_PEPTIDE_RETRACTION_ITERATIONS") < 0:
    raise SystemExit("ERROR: PHASE_RESIDUAL_PEPTIDE_RETRACTION_ITERATIONS must be >= 0")
relaxation = float("$PHASE_RESIDUAL_PEPTIDE_RETRACTION_RELAXATION")
if not 0.0 < relaxation <= 1.0:
    raise SystemExit("ERROR: PHASE_RESIDUAL_PEPTIDE_RETRACTION_RELAXATION must be in (0, 1]")
anchor = float("$PHASE_RESIDUAL_PEPTIDE_RETRACTION_ANCHOR_STRENGTH")
if not 0.0 <= anchor < 1.0:
    raise SystemExit("ERROR: PHASE_RESIDUAL_PEPTIDE_RETRACTION_ANCHOR_STRENGTH must be in [0, 1)")
if float("$PHASE_RESIDUAL_PEPTIDE_RETRACTION_MAX_TRANSLATION") <= 0.0:
    raise SystemExit("ERROR: PHASE_RESIDUAL_PEPTIDE_RETRACTION_MAX_TRANSLATION must be > 0")
if float("$PHASE_RESIDUAL_PEPTIDE_RETRACTION_ACTIVATION_LOSS_THRESHOLD") < 0.0:
    raise SystemExit("ERROR: PHASE_RESIDUAL_PEPTIDE_RETRACTION_ACTIVATION_LOSS_THRESHOLD must be >= 0")
for name, value in {
    "W_PHASE_RESIDUAL_MAGNITUDE": float("$W_PHASE_RESIDUAL_MAGNITUDE"),
    "W_PHASE_RESIDUAL_TEMPORAL_SMOOTH": float("$W_PHASE_RESIDUAL_TEMPORAL_SMOOTH"),
    "W_PHASE_RESIDUAL_NEIGHBOR_SMOOTH": float("$W_PHASE_RESIDUAL_NEIGHBOR_SMOOTH"),
}.items():
    if value < 0.0:
        raise SystemExit(f"ERROR: {name} must be >= 0")
PY
if [[ "$PATH_PARAMETERIZATION" == "phase_orthogonal_residual_v1" && "$GEOM_EVERY" != "1" ]]; then
  echo "ERROR: phase_orthogonal_residual_v1 requires GEOM_EVERY=1"
  exit 1
fi
case "$PHASE_NORMAL_RESIDUAL_LOSS_TYPE" in
  mse|huber) ;;
  *)
    echo "ERROR: PHASE_NORMAL_RESIDUAL_LOSS_TYPE must be mse or huber"
    exit 1
    ;;
esac
case "$PHASE_NORMAL_MISSING_POLICY" in
  error|skip) ;;
  *)
    echo "ERROR: PHASE_NORMAL_MISSING_POLICY must be error or skip"
    exit 1
    ;;
esac
python - <<PY
weight = float("$W_PHASE_NORMAL_RESIDUAL")
delta = float("$PHASE_NORMAL_RESIDUAL_HUBER_DELTA")
if weight < 0.0:
    raise SystemExit("ERROR: W_PHASE_NORMAL_RESIDUAL must be >= 0")
if delta <= 0.0:
    raise SystemExit("ERROR: PHASE_NORMAL_RESIDUAL_HUBER_DELTA must be > 0")
if weight > 0.0 and not "$PHASE_NORMAL_CACHE_DIR":
    raise SystemExit("ERROR: positive MD residual weight requires PHASE_NORMAL_CACHE_DIR")
if weight > 0.0 and "$PATH_PARAMETERIZATION" != "phase_orthogonal_residual_v1":
    raise SystemExit("ERROR: MD phase-normal residual supervision requires phase_orthogonal_residual_v1")
PY
case "$LIGAND_CLEARANCE_MASK_MODE" in
  pocket|node|motion_active|pocket_or_motion_active) ;;
  *)
    echo "ERROR: LIGAND_CLEARANCE_MASK_MODE must be pocket, node, motion_active, or pocket_or_motion_active"
    exit 1
    ;;
esac
case "$LIGAND_CLEARANCE_LOSS_MODE" in
  all|hard_negative) ;;
  *)
    echo "ERROR: LIGAND_CLEARANCE_LOSS_MODE must be all or hard_negative"
    exit 1
    ;;
esac
case "$BRIDGE_ANCHOR_MASK_MODE" in
  non_clash_node|non_clash_pocket|node|pocket) ;;
  *)
    echo "ERROR: BRIDGE_ANCHOR_MASK_MODE must be non_clash_node, non_clash_pocket, node, or pocket"
    exit 1
    ;;
esac
python - <<PY
w = float("$W_LIGAND_CLEARANCE")
dist = float("$LIGAND_CLEARANCE_DIST")
hard_dist = float("$LIGAND_CLEARANCE_HARD_NEGATIVE_DIST")
t_min = float("$LIGAND_CLEARANCE_T_MIN")
t_max = float("$LIGAND_CLEARANCE_T_MAX")
anchor_w = float("$W_BRIDGE_ANCHOR")
anchor_t_min = float("$BRIDGE_ANCHOR_T_MIN")
anchor_t_max = float("$BRIDGE_ANCHOR_T_MAX")
if w < 0.0:
    raise SystemExit("ERROR: W_LIGAND_CLEARANCE must be >= 0")
if dist <= 0.0:
    raise SystemExit("ERROR: LIGAND_CLEARANCE_DIST must be > 0")
if hard_dist <= 0.0:
    raise SystemExit("ERROR: LIGAND_CLEARANCE_HARD_NEGATIVE_DIST must be > 0")
if not (0.0 <= t_min < t_max <= 1.0):
    raise SystemExit("ERROR: LIGAND_CLEARANCE_T_MIN/MAX must satisfy 0 <= min < max <= 1")
if anchor_w < 0.0:
    raise SystemExit("ERROR: W_BRIDGE_ANCHOR must be >= 0")
if not (0.0 <= anchor_t_min < anchor_t_max <= 1.0):
    raise SystemExit("ERROR: BRIDGE_ANCHOR_T_MIN/MAX must satisfy 0 <= min < max <= 1")
PY
case "$ESM_GATE_CONTEXT_MODE" in
  none|pocket_motion) ;;
  *)
    echo "ERROR: ESM_GATE_CONTEXT_MODE must be one of none, pocket_motion"
    exit 1
    ;;
esac
case "$REPA_ENABLED" in
  0|1) ;;
  *)
    echo "ERROR: REPA_ENABLED must be 0 or 1"
    exit 1
    ;;
esac
case "$REPA_LOSS_TYPE" in
  cosine|mse) ;;
  *)
    echo "ERROR: REPA_LOSS_TYPE must be one of cosine, mse"
    exit 1
    ;;
esac
case "$REPA_MASK_MODE" in
  node|pocket|motion_active|motion_active_or_pocket) ;;
  *)
    echo "ERROR: REPA_MASK_MODE must be one of node, pocket, motion_active, motion_active_or_pocket"
    exit 1
    ;;
esac
case "$REPA_TARGET_SHUFFLE_MODE" in
  none|residue) ;;
  *)
    echo "ERROR: REPA_TARGET_SHUFFLE_MODE must be one of none, residue"
    exit 1
    ;;
esac
case "$REPA_TARGET_MODE" in
  full|motion_continuous) ;;
  *)
    echo "ERROR: REPA_TARGET_MODE must be one of full, motion_continuous"
    exit 1
    ;;
esac
case "$PRECHECK_ONLY" in
  0|1) ;;
  *)
    echo "ERROR: PRECHECK_ONLY must be 0 or 1"
    exit 1
    ;;
esac
case "$LENGTH_BUCKETED_TRAIN" in
  0|1) ;;
  *)
    echo "ERROR: LENGTH_BUCKETED_TRAIN must be 0 or 1"
    exit 1
    ;;
esac
case "$LENGTH_BUCKET_DROP_LAST" in
  0|1) ;;
  *)
    echo "ERROR: LENGTH_BUCKET_DROP_LAST must be 0 or 1"
    exit 1
    ;;
esac
if [[ "$LENGTH_BUCKET_MULTIPLIER" -lt 1 ]]; then
  echo "ERROR: LENGTH_BUCKET_MULTIPLIER must be >= 1"
  exit 1
fi
case "$STRICT_CACHE_NPZ_PRECHECK" in
  0|1) ;;
  *)
    echo "ERROR: STRICT_CACHE_NPZ_PRECHECK must be 0 or 1"
    exit 1
    ;;
esac
case "$AATYPE_PRECHECK" in
  0|1) ;;
  *)
    echo "ERROR: AATYPE_PRECHECK must be 0 or 1"
    exit 1
    ;;
esac
case "$NODE_MASK_PRECHECK" in
  0|1) ;;
  *)
    echo "ERROR: NODE_MASK_PRECHECK must be 0 or 1"
    exit 1
    ;;
esac
case "$USE_EXISTING_SUBSETS" in
  0|1) ;;
  *)
    echo "ERROR: USE_EXISTING_SUBSETS must be 0 or 1"
    exit 1
    ;;
esac
case "$TRUST_PRECHECKED_SAMPLES" in
  0|1) ;;
  *)
    echo "ERROR: TRUST_PRECHECKED_SAMPLES must be 0 or 1"
    exit 1
    ;;
esac
case "$VAL_SPLIT" in
  train|val|test) ;;
  *)
    echo "ERROR: VAL_SPLIT must be one of train, val, test"
    exit 1
    ;;
esac
if [[ "$PRECHECK_WORKERS" -lt 1 ]]; then
  echo "ERROR: PRECHECK_WORKERS must be >= 1"
  exit 1
fi

if [[ -z "$STAGE1V2_TRAIN_CACHE_DIR" ]]; then
  echo "ERROR: STAGE1V2_TRAIN_CACHE_DIR is required"
  exit 1
fi
if [[ -z "$STAGE1V2_VAL_CACHE_DIR" ]]; then
  echo "ERROR: STAGE1V2_VAL_CACHE_DIR is required"
  exit 1
fi
FEATURE_DIM="$(python - <<PY
print(len("$STAGE1V2_FEATURES".split(",")))
PY
)"

if [[ -n "$SUBSET_TAG" ]]; then
  TRAIN_SUBSET_REL="ablation_subsets/stage2_oracle_motion_${SUBSET_TAG}_train_${TRAIN_N}_seed${SUBSET_SEED}.txt"
  VAL_SUBSET_REL="ablation_subsets/stage2_oracle_motion_${SUBSET_TAG}_val_${VAL_N}_seed${SUBSET_SEED}.txt"
else
  TRAIN_SUBSET_REL="ablation_subsets/stage2_oracle_motion_train_${TRAIN_N}_seed${SUBSET_SEED}.txt"
  VAL_SUBSET_REL="ablation_subsets/stage2_oracle_motion_val_${VAL_N}_seed${SUBSET_SEED}.txt"
fi
TRAIN_SUBSET="processed_data/triplets/${TRAIN_SUBSET_REL}"
VAL_SUBSET="processed_data/triplets/${VAL_SUBSET_REL}"

if [[ "$USE_EXISTING_SUBSETS" == "1" ]]; then
  if [[ ! -s "$TRAIN_SUBSET" || ! -s "$VAL_SUBSET" ]]; then
    echo "ERROR: USE_EXISTING_SUBSETS=1 but subset files are missing or empty"
    echo "  train: $TRAIN_SUBSET"
    echo "  val:   $VAL_SUBSET"
    exit 1
  fi
  TRAIN_SUBSET_COUNT="$(wc -l < "$TRAIN_SUBSET")"
  VAL_SUBSET_COUNT="$(wc -l < "$VAL_SUBSET")"
  if [[ "$TRAIN_SUBSET_COUNT" -ne "$TRAIN_N" || "$VAL_SUBSET_COUNT" -ne "$VAL_N" ]]; then
    echo "ERROR: USE_EXISTING_SUBSETS=1 but subset line counts do not match TRAIN_N/VAL_N"
    echo "  train: $TRAIN_SUBSET_COUNT != $TRAIN_N ($TRAIN_SUBSET)"
    echo "  val:   $VAL_SUBSET_COUNT != $VAL_N ($VAL_SUBSET)"
    exit 1
  fi
  echo "Using existing prechecked subsets:"
  echo "  train: $TRAIN_SUBSET ($TRAIN_SUBSET_COUNT IDs)"
  echo "  val:   $VAL_SUBSET ($VAL_SUBSET_COUNT IDs)"
else
python - <<PY
import json
import random
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import torch

from src.data.residue_identity import load_residue_keys, residue_identity_hash
from src.stage2.datasets.backbone import (
    _coords_valid_mask,
    _load_backbone_npz,
    _load_torsion_residue_keys,
    _load_torsions,
    _sequence_to_aatype,
    align_by_residue_ids,
    extract_backbone_coords,
)
from src.stage2.datasets.esm_cache import _esm_features_from_data

DATA_DIR = Path("processed_data/triplets")
STRICT_CACHE_NPZ_PRECHECK = "$STRICT_CACHE_NPZ_PRECHECK" == "1"
AATYPE_PRECHECK = "$AATYPE_PRECHECK" == "1"
NODE_MASK_PRECHECK = "$NODE_MASK_PRECHECK" == "1"
PRECHECK_WORKERS = int("$PRECHECK_WORKERS")
ESM_NUM_LAYERS = int("$ESM_NUM_LAYERS")
EXPECTED_META_CACHE = {}


def cache_path_for_record(record, manifest_path):
    raw_path = record.get("path")
    if raw_path:
        path = Path(raw_path)
        if not path.is_absolute():
            path = manifest_path.parent / path
        return path
    sample_id = record["sample_id"]
    safe_id = "".join(ch if ch.isalnum() or ch in "_.-" else "_" for ch in sample_id)
    return manifest_path.parent / f"{safe_id}.npz"


def expected_stage2_meta(sample_id):
    cached = EXPECTED_META_CACHE.get(sample_id)
    if cached is not None:
        return cached
    sample_dir = DATA_DIR / "samples" / sample_id
    esm_path = sample_dir / "esm.pt"
    if not esm_path.exists():
        raise FileNotFoundError(esm_path)
    try:
        esm_data = torch.load(esm_path, map_location="cpu", weights_only=False, mmap=True)
    except TypeError:
        esm_data = torch.load(esm_path, map_location="cpu", weights_only=False)
    esm_features = _esm_features_from_data(esm_data, esm_path, ESM_NUM_LAYERS)
    n_res = int(esm_features.shape[0])

    torsion_apo_path = DATA_DIR / "samples" / sample_id / "torsion_apo.npz"
    torsion_holo_path = DATA_DIR / "samples" / sample_id / "torsion_holo.npz"
    apo_torsion_keys = _load_torsion_residue_keys(torsion_apo_path)
    target_residue_keys = load_residue_keys(esm_data) or apo_torsion_keys
    if len(target_residue_keys) != n_res:
        raise ValueError(
            f"canonical residue count {len(target_residue_keys)} != ESM length {n_res}"
        )
    torsion_apo = _load_torsions(torsion_apo_path, target_residue_keys)
    torsion_holo = _load_torsions(torsion_holo_path, target_residue_keys)
    if torsion_apo["aatype"] is not None:
        aatype = torsion_apo["aatype"].astype(np.int64)
    else:
        sequence_str = str(esm_data.get("sequence_str", ""))
        aatype = _sequence_to_aatype(sequence_str, n_res)

    apo_backbone = sample_dir / "apo_backbone.npz"
    holo_backbone = sample_dir / "holo_backbone.npz"
    apo_pdb = sample_dir / "apo.pdb"
    holo_pdb = sample_dir / "holo.pdb"
    if apo_backbone.exists():
        N_apo, Ca_apo, C_apo, apo_res_ids = _load_backbone_npz(apo_backbone)
        if apo_res_ids is None:
            N_apo, Ca_apo, C_apo, _, apo_res_ids = extract_backbone_coords(apo_pdb)
    else:
        N_apo, Ca_apo, C_apo, _, apo_res_ids = extract_backbone_coords(apo_pdb)
    if holo_backbone.exists():
        N_holo, Ca_holo, C_holo, holo_res_ids = _load_backbone_npz(holo_backbone)
        if holo_res_ids is None:
            N_holo, Ca_holo, C_holo, _, holo_res_ids = extract_backbone_coords(holo_pdb)
    else:
        N_holo, Ca_holo, C_holo, _, holo_res_ids = extract_backbone_coords(holo_pdb)
    apo_aligned, holo_aligned, node_mask = align_by_residue_ids(
        (N_apo, Ca_apo, C_apo), apo_res_ids,
        (N_holo, Ca_holo, C_holo), holo_res_ids,
        target_residue_keys,
    )
    node_mask &= _coords_valid_mask(*apo_aligned)
    node_mask &= _coords_valid_mask(*holo_aligned)
    node_mask &= torsion_apo["residue_present"] & torsion_holo["residue_present"]
    residue_hash = residue_identity_hash(target_residue_keys)

    EXPECTED_META_CACHE[sample_id] = (n_res, aatype, node_mask, residue_hash)
    return n_res, aatype, node_mask, residue_hash


def record_matches_stage2(record, manifest_path):
    sample_id = record["sample_id"]
    cache_path = cache_path_for_record(record, manifest_path)
    if not cache_path.exists():
        return False, "missing_cache"
    try:
        n_res, expected_aatype, expected_node_mask, expected_residue_hash = expected_stage2_meta(sample_id)
        if "n_residues" not in record:
            return False, "manifest_missing_n_residues"
        if int(record["n_residues"]) != n_res:
            return False, "n_residues_mismatch"
        if not STRICT_CACHE_NPZ_PRECHECK and not AATYPE_PRECHECK and not NODE_MASK_PRECHECK:
            return True, "ok"
        with np.load(cache_path, allow_pickle=False) as data:
            if "sample_id" in data and str(np.asarray(data["sample_id"]).item()) != sample_id:
                return False, "sample_id_mismatch"
            if "n_residues" not in data:
                return False, "missing_n_residues"
            cache_n_res = int(np.asarray(data["n_residues"]).item())
            if cache_n_res != n_res:
                return False, "n_residues_mismatch"
            if "oracle_motion_features" in data and int(data["oracle_motion_features"].shape[0]) != n_res:
                return False, "feature_length_mismatch"
            if "residue_identity_hash" not in data:
                return False, "missing_residue_identity_hash"
            if str(np.asarray(data["residue_identity_hash"]).item()) != expected_residue_hash:
                return False, "residue_identity_hash_mismatch"
            if AATYPE_PRECHECK:
                if "aatype" not in data:
                    return False, "missing_aatype"
                cache_aatype = np.asarray(data["aatype"]).astype(np.int64)
                if cache_aatype.shape[0] != n_res:
                    return False, "aatype_length_mismatch"
                if not np.array_equal(cache_aatype, expected_aatype):
                    return False, "aatype_mismatch"
            if NODE_MASK_PRECHECK:
                mask_key = "node_mask" if "node_mask" in data else ("valid_mask" if "valid_mask" in data else None)
                if mask_key is not None:
                    cache_mask = np.asarray(data[mask_key]).astype(np.bool_)
                    if cache_mask.shape[0] != n_res:
                        return False, "node_mask_length_mismatch"
                    invalid_claims = cache_mask & (~np.asarray(expected_node_mask).astype(np.bool_))
                    if invalid_claims.any():
                        return False, "node_mask_mismatch"
    except Exception as exc:
        return False, f"{type(exc).__name__}"
    return True, "ok"


def write_from_manifest(manifest_path, out_path, n, seed):
    manifest_path = Path(manifest_path)
    out_path = Path(out_path)
    with manifest_path.open() as f:
        manifest = json.load(f)
    records = list(manifest["records"])
    if int(n) > len(records):
        raise SystemExit(f"requested {n} IDs from {manifest_path}, only {len(records)} available")
    rng = random.Random(int(seed))
    rng.shuffle(records)
    chosen = []
    skipped = {}
    checked = 0

    def check_record(record):
        ok, reason = record_matches_stage2(record, manifest_path)
        return record["sample_id"], ok, reason

    if PRECHECK_WORKERS > 1:
        executor = ThreadPoolExecutor(max_workers=PRECHECK_WORKERS)
        chunk_size = max(PRECHECK_WORKERS * 64, 256)
    else:
        executor = None
        chunk_size = 256
    try:
        for start in range(0, len(records), chunk_size):
            chunk = records[start:start + chunk_size]
            if executor is not None:
                results = executor.map(check_record, chunk)
            else:
                results = map(check_record, chunk)
            for sample_id, ok, reason in results:
                checked += 1
                if ok:
                    chosen.append(sample_id)
                    if len(chosen) >= int(n):
                        break
                else:
                    skipped[reason] = skipped.get(reason, 0) + 1
                if checked % 1000 == 0:
                    print(f"Checked {checked} records from {manifest_path}; chosen={len(chosen)} skipped={skipped}", flush=True)
            if len(chosen) >= int(n):
                break
    finally:
        if executor is not None:
            executor.shutdown(wait=True, cancel_futures=True)
    if len(chosen) < int(n):
        raise SystemExit(
            f"requested {n} Stage-2/cache-consistent IDs from {manifest_path}, "
            f"only found {len(chosen)}; skipped={skipped}"
        )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\\n".join(chosen) + "\\n")
    print(f"Wrote {len(chosen)} Stage-2/cache-consistent IDs to {out_path} from {manifest_path}; skipped={skipped}")

write_from_manifest("$STAGE1V2_TRAIN_CACHE_DIR/manifest.json", "$TRAIN_SUBSET", "$TRAIN_N", "$SUBSET_SEED")
write_from_manifest("$STAGE1V2_VAL_CACHE_DIR/manifest.json", "$VAL_SUBSET", "$VAL_N", int("$SUBSET_SEED") + 1)
PY
fi

if [[ "$PRECHECK_ONLY" == "1" ]]; then
  echo "=============================================="
  echo "Stage-2 + OracleMotion-UB precheck passed"
  echo "=============================================="
  echo "Mode:         $STAGE1V2_MODE"
  echo "Train cache:  $STAGE1V2_TRAIN_CACHE_DIR"
  echo "Val cache:    $STAGE1V2_VAL_CACHE_DIR"
  echo "Val split:    $VAL_SPLIT"
  echo "Train subset: $TRAIN_SUBSET"
  echo "Val subset:   $VAL_SUBSET"
  echo "TRAIN_N:      $TRAIN_N"
  echo "VAL_N:        $VAL_N"
  echo "No torchrun/training launched because PRECHECK_ONLY=1."
  exit 0
fi

TAG="${TAG:-stage2_${TAG_SUFFIX}_train${TRAIN_N}_val${VAL_N}_e${MAX_EPOCHS}_bs${BATCH_SIZE}x${NPROC_PER_NODE}_$(date +%Y%m%d_%H%M%S)}"
SAVE_DIR="${SAVE_DIR:-checkpoints/stage2/${TAG}}"
LOG_DIR="${LOG_DIR:-logs/stage2/${TAG}}"
mkdir -p "$LOG_DIR"
GPU_MONITOR_PID=""
if [[ "$GPU_MONITOR_INTERVAL" -gt 0 ]]; then
  GPU_MONITOR_LOG="$LOG_DIR/gpu_util_${SLURM_JOB_ID:-manual}.csv"
  nvidia-smi \
    --query-gpu=timestamp,index,utilization.gpu,utilization.memory,memory.used,memory.total,power.draw \
    --format=csv \
    -l "$GPU_MONITOR_INTERVAL" > "$GPU_MONITOR_LOG" 2>/dev/null &
  GPU_MONITOR_PID="$!"
  cleanup_gpu_monitor() {
    if [[ -n "$GPU_MONITOR_PID" ]]; then
      kill "$GPU_MONITOR_PID" 2>/dev/null || true
    fi
  }
  trap cleanup_gpu_monitor EXIT
fi
ESM_ARGS=()
if [[ "$ESM_FUSION_ENABLED" == "1" ]]; then
  ESM_ARGS=(
    --esm_fusion_enabled
    --esm_num_layers "$ESM_NUM_LAYERS"
    --esm_fusion_mode "$ESM_FUSION_MODE"
    --esm_layer_dropout "$ESM_LAYER_DROPOUT"
    --esm_layer_entropy_weight "$ESM_LAYER_ENTROPY_WEIGHT"
    --esm_gate_bias "$ESM_GATE_BIAS"
    --esm_gate_context_mode "$ESM_GATE_CONTEXT_MODE"
  )
fi
REPA_ARGS=()
if [[ "$REPA_ENABLED" == "1" ]]; then
  REPA_ARGS=(
    --repa_enabled
    --repa_weight "$REPA_WEIGHT"
    --repa_dim "$REPA_DIM"
    --repa_loss_type "$REPA_LOSS_TYPE"
    --repa_mask_mode "$REPA_MASK_MODE"
    --repa_target_mode "$REPA_TARGET_MODE"
    --repa_target_shuffle_mode "$REPA_TARGET_SHUFFLE_MODE"
  )
fi
RESUME_ARGS=()
if [[ -n "$RESUME_FROM" ]]; then
  RESUME_ARGS+=(--resume_from "$RESUME_FROM")
fi
if [[ -n "$INIT_FROM_CHECKPOINT" ]]; then
  RESUME_ARGS+=(--init_from_checkpoint "$INIT_FROM_CHECKPOINT")
fi
if [[ "$AUTO_RESUME" != "1" ]]; then
  RESUME_ARGS+=(--no_auto_resume)
fi
TEACHER_RESIDUAL_ARGS=(
  --w_teacher_residual "$W_TEACHER_RESIDUAL"
  --teacher_residual_loss_type "$TEACHER_RESIDUAL_LOSS_TYPE"
  --teacher_residual_huber_delta "$TEACHER_RESIDUAL_HUBER_DELTA"
  --teacher_residual_t_min "$TEACHER_RESIDUAL_T_MIN"
  --teacher_residual_t_max "$TEACHER_RESIDUAL_T_MAX"
  --teacher_residual_mask_mode "$TEACHER_RESIDUAL_MASK_MODE"
  --teacher_residual_clash_weight_threshold "$TEACHER_RESIDUAL_CLASH_WEIGHT_THRESHOLD"
  --teacher_residual_missing_policy "$TEACHER_RESIDUAL_MISSING_POLICY"
)
if [[ -n "$TEACHER_RESIDUAL_CACHE_DIR" ]]; then
  TEACHER_RESIDUAL_ARGS+=(--teacher_residual_cache_dir "$TEACHER_RESIDUAL_CACHE_DIR")
fi
PHASE_TEACHER_ARGS=(
  --w_phase_teacher "$W_PHASE_TEACHER"
  --phase_teacher_loss_type "$PHASE_TEACHER_LOSS_TYPE"
  --phase_teacher_huber_delta "$PHASE_TEACHER_HUBER_DELTA"
  --phase_teacher_mask_mode "$PHASE_TEACHER_MASK_MODE"
  --phase_teacher_min_confidence "$PHASE_TEACHER_MIN_CONFIDENCE"
  --phase_teacher_missing_policy "$PHASE_TEACHER_MISSING_POLICY"
)
if [[ -n "$PHASE_TEACHER_CACHE_DIR" ]]; then
  PHASE_TEACHER_ARGS+=(--phase_teacher_cache_dir "$PHASE_TEACHER_CACHE_DIR")
fi
if [[ "$PHASE_TEACHER_HEAD_ONLY" == "1" ]]; then
  PHASE_TEACHER_ARGS+=(--phase_teacher_head_only)
fi
if [[ "$PHASE_TEACHER_RESIDUAL_HEADS_ONLY" == "1" ]]; then
  PHASE_TEACHER_ARGS+=(--phase_teacher_residual_heads_only)
fi
PHASE_NORMAL_ARGS=(
  --w_phase_normal_residual "$W_PHASE_NORMAL_RESIDUAL"
  --phase_normal_residual_loss_type "$PHASE_NORMAL_RESIDUAL_LOSS_TYPE"
  --phase_normal_residual_huber_delta "$PHASE_NORMAL_RESIDUAL_HUBER_DELTA"
  --phase_normal_missing_policy "$PHASE_NORMAL_MISSING_POLICY"
)
if [[ -n "$PHASE_NORMAL_CACHE_DIR" ]]; then
  PHASE_NORMAL_ARGS+=(--phase_normal_cache_dir "$PHASE_NORMAL_CACHE_DIR")
fi
PHASE_RETRACTION_ARGS=(
  --phase_residual_peptide_retraction_iterations "$PHASE_RESIDUAL_PEPTIDE_RETRACTION_ITERATIONS"
  --phase_residual_peptide_retraction_relaxation "$PHASE_RESIDUAL_PEPTIDE_RETRACTION_RELAXATION"
  --phase_residual_peptide_retraction_anchor_strength "$PHASE_RESIDUAL_PEPTIDE_RETRACTION_ANCHOR_STRENGTH"
  --phase_residual_peptide_retraction_max_translation "$PHASE_RESIDUAL_PEPTIDE_RETRACTION_MAX_TRANSLATION"
  --phase_residual_peptide_retraction_activation_loss_threshold "$PHASE_RESIDUAL_PEPTIDE_RETRACTION_ACTIVATION_LOSS_THRESHOLD"
)
if [[ "$PHASE_RESIDUAL_PEPTIDE_RETRACTION" == "1" ]]; then
  PHASE_RETRACTION_ARGS+=(--phase_residual_peptide_retraction)
fi
LENGTH_BUCKET_ARGS=()
if [[ "$LENGTH_BUCKETED_TRAIN" == "1" ]]; then
  LENGTH_BUCKET_ARGS=(
    --length_bucketed_train
    --length_bucket_multiplier "$LENGTH_BUCKET_MULTIPLIER"
  )
  if [[ "$LENGTH_BUCKET_DROP_LAST" == "1" ]]; then
    LENGTH_BUCKET_ARGS+=(--length_bucket_drop_last)
  else
    LENGTH_BUCKET_ARGS+=(--no_length_bucket_drop_last)
  fi
  if [[ -n "$LENGTH_BUCKET_LENGTHS_FILE" ]]; then
    LENGTH_BUCKET_ARGS+=(--length_bucket_lengths_file "$LENGTH_BUCKET_LENGTHS_FILE")
  fi
  if [[ -n "$LENGTH_BUCKET_RESIDUE_BUDGET" ]]; then
    LENGTH_BUCKET_ARGS+=(--length_bucket_residue_budget "$LENGTH_BUCKET_RESIDUE_BUDGET")
  fi
fi
DATASET_ARGS=()
if [[ "$TRUST_PRECHECKED_SAMPLES" == "1" ]]; then
  DATASET_ARGS+=(--trust_prechecked_samples)
fi

echo "=============================================="
echo "Stage-2 + OracleMotion-UB ablation"
echo "=============================================="
echo "Job ID:          ${SLURM_JOB_ID:-NA}"
echo "Node:            ${SLURM_NODELIST:-NA}"
echo "GPUs:            $NPROC_PER_NODE"
echo "Mode:            $STAGE1V2_MODE"
echo "Feature dim:     $FEATURE_DIM"
echo "Feature scale:   $STAGE1V2_FEATURE_SCALE"
echo "Train cache:     $STAGE1V2_TRAIN_CACHE_DIR"
echo "Val cache:       $STAGE1V2_VAL_CACHE_DIR"
echo "Train subset:    $TRAIN_SUBSET"
echo "Val subset:      $VAL_SUBSET"
echo "Val split:       $VAL_SPLIT"
echo "Trust precheck:  $TRUST_PRECHECKED_SAMPLES"
echo "max_epochs:      $MAX_EPOCHS"
echo "batch/GPU:       $BATCH_SIZE"
echo "val batch/GPU:   $VAL_BATCH_SIZE"
echo "num_workers/GPU: $NUM_WORKERS"
echo "prefetch factor: $PREFETCH_FACTOR"
echo "length buckets:  $LENGTH_BUCKETED_TRAIN"
echo "bucket mult:     $LENGTH_BUCKET_MULTIPLIER"
echo "bucket drop_last:$LENGTH_BUCKET_DROP_LAST"
echo "bucket lengths:  ${LENGTH_BUCKET_LENGTHS_FILE:-OFF}"
echo "bucket res budget:${LENGTH_BUCKET_RESIDUE_BUDGET:-OFF}"
echo "progress every:  $PROGRESS_LOG_EVERY"
echo "ckpt every ep:   $CHECKPOINT_EVERY_N_EPOCHS"
echo "lr:              $LR"
echo "warmup steps:    $WARMUP_STEPS"
echo "early stop:      $EARLY_STOP_PATIENCE"
echo "path param:      $PATH_PARAMETERIZATION"
echo "boundary env:    $BOUNDARY_RESIDUAL_ENVELOPE"
echo "boundary scale:  $BOUNDARY_RESIDUAL_SCALE"
echo "projection sched:$TERMINAL_PROJECTION_SCHEDULE"
echo "timewarp logit:  $TIME_WARP_LOGIT_SCALE"
echo "timewarp eps:    $TIME_WARP_RATE_EPS"
echo "timewarp clip:   $TIME_WARP_RATE_CLIP"
echo "phase tau mode:  $PHASE_RESIDUAL_TAU_MODE"
echo "phase bridge:    $PHASE_RESIDUAL_BRIDGE_MODE"
echo "phase envelope:  $PHASE_RESIDUAL_ENVELOPE"
echo "phase scale:     $PHASE_RESIDUAL_SCALE"
echo "phase metric:    rot=$PHASE_RESIDUAL_ROTATION_METRIC_SCALE trans=$PHASE_RESIDUAL_TRANSLATION_METRIC_SCALE chi=$PHASE_RESIDUAL_CHI_METRIC_SCALE"
echo "phase min norm:  $PHASE_RESIDUAL_MIN_TANGENT_NORM"
echo "phase max norm:  ${PHASE_RESIDUAL_MAX_METRIC_NORM:-OFF}"
echo "phase pep retract:$PHASE_RESIDUAL_PEPTIDE_RETRACTION"
echo "phase pep retract config: iter=$PHASE_RESIDUAL_PEPTIDE_RETRACTION_ITERATIONS relax=$PHASE_RESIDUAL_PEPTIDE_RETRACTION_RELAXATION anchor=$PHASE_RESIDUAL_PEPTIDE_RETRACTION_ANCHOR_STRENGTH max=$PHASE_RESIDUAL_PEPTIDE_RETRACTION_MAX_TRANSLATION"
echo "init ckpt:       ${INIT_FROM_CHECKPOINT:-OFF}"
echo "teacher cache:   ${TEACHER_RESIDUAL_CACHE_DIR:-OFF}"
echo "w_teacher_resid: $W_TEACHER_RESIDUAL"
echo "teacher loss:    $TEACHER_RESIDUAL_LOSS_TYPE"
echo "teacher huber d: $TEACHER_RESIDUAL_HUBER_DELTA"
echo "teacher t range: $TEACHER_RESIDUAL_T_MIN-$TEACHER_RESIDUAL_T_MAX"
echo "teacher mask:    $TEACHER_RESIDUAL_MASK_MODE"
echo "teacher cw thr:  $TEACHER_RESIDUAL_CLASH_WEIGHT_THRESHOLD"
echo "teacher missing: $TEACHER_RESIDUAL_MISSING_POLICY"
echo "phase teacher:   ${PHASE_TEACHER_CACHE_DIR:-OFF}"
echo "w_phase_teacher: $W_PHASE_TEACHER"
echo "phase teacher mask/conf: $PHASE_TEACHER_MASK_MODE/$PHASE_TEACHER_MIN_CONFIDENCE"
echo "phase head only: $PHASE_TEACHER_HEAD_ONLY"
echo "phase/resid heads only: $PHASE_TEACHER_RESIDUAL_HEADS_ONLY"
echo "phase-normal cache: ${PHASE_NORMAL_CACHE_DIR:-OFF}"
echo "w_phase_normal_residual: $W_PHASE_NORMAL_RESIDUAL"
echo "w_fm_chi:        $W_FM_CHI"
echo "w_fm_rigid:      $W_FM_RIGID"
echo "w_bg:            $W_BG"
echo "w_phase_mag:     $W_PHASE_RESIDUAL_MAGNITUDE"
echo "w_phase_time:    $W_PHASE_RESIDUAL_TEMPORAL_SMOOTH"
echo "w_phase_neighbor:$W_PHASE_RESIDUAL_NEIGHBOR_SMOOTH"
echo "w_smooth:        $W_SMOOTH"
echo "w_clash:         $W_CLASH"
echo "w_lig_clear:     $W_LIGAND_CLEARANCE"
echo "lig clear dist:  $LIGAND_CLEARANCE_DIST"
echo "lig hard dist:   $LIGAND_CLEARANCE_HARD_NEGATIVE_DIST"
echo "lig clear mode:  $LIGAND_CLEARANCE_LOSS_MODE"
echo "lig clear mask:  $LIGAND_CLEARANCE_MASK_MODE"
echo "lig clear t:     $LIGAND_CLEARANCE_T_MIN-$LIGAND_CLEARANCE_T_MAX"
echo "w_bridge_anchor: $W_BRIDGE_ANCHOR"
echo "bridge mask:     $BRIDGE_ANCHOR_MASK_MODE"
echo "bridge t:        $BRIDGE_ANCHOR_T_MIN-$BRIDGE_ANCHOR_T_MAX"
echo "w_pep:           $W_PEP"
echo "w_contact:       $W_CONTACT"
echo "w_end:           $W_END"
echo "ESM fusion:      $ESM_FUSION_ENABLED"
echo "ESM layers/mode: $ESM_NUM_LAYERS / $ESM_FUSION_MODE"
echo "ESM layer drop:  $ESM_LAYER_DROPOUT"
echo "ESM entropy wt:  $ESM_LAYER_ENTROPY_WEIGHT"
echo "ESM gate bias:   $ESM_GATE_BIAS"
echo "ESM gate ctx:    $ESM_GATE_CONTEXT_MODE"
echo "REPA enabled:    $REPA_ENABLED"
echo "REPA weight:     $REPA_WEIGHT"
echo "REPA dim/loss:   $REPA_DIM / $REPA_LOSS_TYPE"
echo "REPA mask:       $REPA_MASK_MODE"
echo "REPA target:     $REPA_TARGET_MODE"
echo "REPA shuffle:    $REPA_TARGET_SHUFFLE_MODE"
echo "Resume from:     ${RESUME_FROM:-OFF}"
echo "Auto resume:     $AUTO_RESUME"
echo "geom_every:      $GEOM_EVERY"
echo "integration:     $N_INTEGRATION_STEPS"
echo "chi clip:        $INTEGRATION_CHI_CLIP"
echo "rot clip:        $INTEGRATION_ROT_CLIP"
echo "trans clip:      $INTEGRATION_TRANS_CLIP"
echo "geom steps:      $N_GEOM_STEPS"
echo "Save dir:        $SAVE_DIR"
echo "Log dir:         $LOG_DIR"
echo "GPU monitor:     ${GPU_MONITOR_LOG:-OFF}"
echo "Start:           $(date)"
echo "=============================================="

nvidia-smi --query-gpu=index,name,memory.total,memory.free --format=csv,noheader
python -c "import torch; print(f'PyTorch {torch.__version__}, CUDA {torch.version.cuda}, GPUs: {torch.cuda.device_count()}')"

"$ENV_PREFIX/bin/torchrun" --standalone --nproc_per_node="$NPROC_PER_NODE" scripts/train_stage2.py \
  --data_dir processed_data/triplets \
  --batch_size "$BATCH_SIZE" \
  --val_batch_size "$VAL_BATCH_SIZE" \
  --num_workers "$NUM_WORKERS" \
  --prefetch_factor "$PREFETCH_FACTOR" \
  --progress_log_every "$PROGRESS_LOG_EVERY" \
  --checkpoint_every_n_epochs "$CHECKPOINT_EVERY_N_EPOCHS" \
  --max_epochs "$MAX_EPOCHS" \
  --lr "$LR" \
  --grad_clip 0.3 \
  --accum_steps 1 \
  --warmup_steps "$WARMUP_STEPS" \
  --early_stop_patience "$EARLY_STOP_PATIENCE" \
  --seed "$SEED" \
  --val_t "$VAL_T" \
  --path_parameterization "$PATH_PARAMETERIZATION" \
  --boundary_residual_envelope "$BOUNDARY_RESIDUAL_ENVELOPE" \
  --boundary_residual_scale "$BOUNDARY_RESIDUAL_SCALE" \
  --terminal_projection_schedule "$TERMINAL_PROJECTION_SCHEDULE" \
  --time_warp_logit_scale "$TIME_WARP_LOGIT_SCALE" \
  --time_warp_rate_eps "$TIME_WARP_RATE_EPS" \
  --time_warp_rate_clip "$TIME_WARP_RATE_CLIP" \
  --phase_residual_tau_mode "$PHASE_RESIDUAL_TAU_MODE" \
  --phase_residual_bridge_mode "$PHASE_RESIDUAL_BRIDGE_MODE" \
  --phase_residual_envelope "$PHASE_RESIDUAL_ENVELOPE" \
  --phase_residual_scale "$PHASE_RESIDUAL_SCALE" \
  --phase_residual_rotation_metric_scale "$PHASE_RESIDUAL_ROTATION_METRIC_SCALE" \
  --phase_residual_translation_metric_scale "$PHASE_RESIDUAL_TRANSLATION_METRIC_SCALE" \
  --phase_residual_chi_metric_scale "$PHASE_RESIDUAL_CHI_METRIC_SCALE" \
  --phase_residual_min_tangent_norm "$PHASE_RESIDUAL_MIN_TANGENT_NORM" \
  --phase_residual_max_metric_norm "$PHASE_RESIDUAL_MAX_METRIC_NORM" \
  "${PHASE_RETRACTION_ARGS[@]}" \
  "${TEACHER_RESIDUAL_ARGS[@]}" \
  "${PHASE_TEACHER_ARGS[@]}" \
  "${PHASE_NORMAL_ARGS[@]}" \
  --val_split "$VAL_SPLIT" \
  --no_stage1_prior \
  --w_prior 0.0 \
  --interaction_prior_feature_mode none \
  --w_interaction_prior 0.0 \
  --stage1v2_posterior_feature_mode "$STAGE1V2_MODE" \
  --stage1v2_train_cache_dir "$STAGE1V2_TRAIN_CACHE_DIR" \
  --stage1v2_val_cache_dir "$STAGE1V2_VAL_CACHE_DIR" \
  --stage1v2_posterior_feature_names "$STAGE1V2_FEATURES" \
  --stage1v2_posterior_feature_scale "$STAGE1V2_FEATURE_SCALE" \
  --stage1v2_loss_weight_mode none \
  --stage1v2_loss_weight_alpha 0.0 \
  --w_stage1v2_guidance 0.0 \
  --contact_loss_mode holo_target \
  --w_fm_chi "$W_FM_CHI" \
  --w_fm_rigid "$W_FM_RIGID" \
  --w_bg "$W_BG" \
  --w_phase_residual_magnitude "$W_PHASE_RESIDUAL_MAGNITUDE" \
  --w_phase_residual_temporal_smooth "$W_PHASE_RESIDUAL_TEMPORAL_SMOOTH" \
  --w_phase_residual_neighbor_smooth "$W_PHASE_RESIDUAL_NEIGHBOR_SMOOTH" \
  --w_smooth "$W_SMOOTH" \
  --w_clash "$W_CLASH" \
  --w_ligand_clearance "$W_LIGAND_CLEARANCE" \
  --ligand_clearance_dist "$LIGAND_CLEARANCE_DIST" \
  --ligand_clearance_mask_mode "$LIGAND_CLEARANCE_MASK_MODE" \
  --ligand_clearance_loss_mode "$LIGAND_CLEARANCE_LOSS_MODE" \
  --ligand_clearance_hard_negative_dist "$LIGAND_CLEARANCE_HARD_NEGATIVE_DIST" \
  --ligand_clearance_t_min "$LIGAND_CLEARANCE_T_MIN" \
  --ligand_clearance_t_max "$LIGAND_CLEARANCE_T_MAX" \
  --w_bridge_anchor "$W_BRIDGE_ANCHOR" \
  --bridge_anchor_mask_mode "$BRIDGE_ANCHOR_MASK_MODE" \
  --bridge_anchor_t_min "$BRIDGE_ANCHOR_T_MIN" \
  --bridge_anchor_t_max "$BRIDGE_ANCHOR_T_MAX" \
  --w_pep "$W_PEP" \
  --w_contact "$W_CONTACT" \
  --w_end "$W_END" \
  --n_integration_steps "$N_INTEGRATION_STEPS" \
  --integration_chi_clip "$INTEGRATION_CHI_CLIP" \
  --integration_rot_clip "$INTEGRATION_ROT_CLIP" \
  --integration_trans_clip "$INTEGRATION_TRANS_CLIP" \
  --n_geom_steps "$N_GEOM_STEPS" \
  --geom_loss_every_n_steps "$GEOM_EVERY" \
  --valid_samples_file "$TRAIN_SUBSET_REL" \
  --val_samples_file "$VAL_SUBSET_REL" \
  "${DATASET_ARGS[@]}" \
  "${LENGTH_BUCKET_ARGS[@]}" \
  "${ESM_ARGS[@]}" \
  "${REPA_ARGS[@]}" \
  "${RESUME_ARGS[@]}" \
  --save_dir "$SAVE_DIR" \
  --log_dir "$LOG_DIR" \
  --device cuda \
  --amp_dtype "$AMP_DTYPE" \
  --distributed

echo ""
echo "=============================================="
echo "Stage-2 + OracleMotion-UB ablation completed: $(date)"
echo "Metrics: $LOG_DIR/metrics.jsonl"
echo "=============================================="
