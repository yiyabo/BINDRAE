#!/bin/bash
# Offline residue-level Stage-2 transition path evaluator.

#SBATCH --job-name=s2_trans_eval
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:A100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=80G
#SBATCH --time=04:00:00
#SBATCH --output=logs/slurm/stage2_transition_eval_%j.out
#SBATCH --error=logs/slurm/stage2_transition_eval_%j.err

set -euo pipefail
unset LD_PRELOAD
unset PROXYCHAINS_CONF_FILE
unset PROXYCHAINS_QUIET_MODE

ROOT="${ROOT:-/mnt/inaisfs/data/home/zhaozc_criait/XinxiangWang/BINDRAE}"
ENV_PREFIX=/mnt/inaisfs/data/home/zhaozc_criait/miniconda3/envs/BINDRAE

export PATH=/data/soft/slurm/24.11.4/bin:$ENV_PREFIX/bin:$PATH
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=4

cd "$ROOT"
mkdir -p logs/slurm logs/stage2/transition_eval

source /mnt/inaisfs/data/home/zhaozc_criait/miniconda3/etc/profile.d/conda.sh
conda activate BINDRAE

CHECKPOINT="${CHECKPOINT:?CHECKPOINT is required}"
TAG="${TAG:-$(basename "$(dirname "$CHECKPOINT")")}"
SPLIT="${SPLIT:-val}"
VALID_SAMPLES_FILE="${VALID_SAMPLES_FILE:-ablation_subsets/stage2_lc_pgbf_val_1200_seed20260618.txt}"
TRUST_PRECHECKED_SAMPLES="${TRUST_PRECHECKED_SAMPLES:-0}"
PATH_PARAMETERIZATION="${PATH_PARAMETERIZATION:-checkpoint}"
BOUNDARY_RESIDUAL_ENVELOPE="${BOUNDARY_RESIDUAL_ENVELOPE:-}"
BOUNDARY_RESIDUAL_SCALE="${BOUNDARY_RESIDUAL_SCALE:-}"
TERMINAL_PROJECTION_SCHEDULE="${TERMINAL_PROJECTION_SCHEDULE:-}"
TIME_WARP_LOGIT_SCALE="${TIME_WARP_LOGIT_SCALE:-}"
TIME_WARP_RATE_EPS="${TIME_WARP_RATE_EPS:-}"
TIME_WARP_RATE_CLIP="${TIME_WARP_RATE_CLIP:-}"
PHASE_RESIDUAL_TAU_MODE="${PHASE_RESIDUAL_TAU_MODE:-}"
PHASE_RESIDUAL_BRIDGE_MODE="${PHASE_RESIDUAL_BRIDGE_MODE:-}"
PHASE_RESIDUAL_ENVELOPE="${PHASE_RESIDUAL_ENVELOPE:-}"
PHASE_RESIDUAL_SCALE="${PHASE_RESIDUAL_SCALE:-}"
PHASE_RESIDUAL_MAX_METRIC_NORM="${PHASE_RESIDUAL_MAX_METRIC_NORM:-}"
PHASE_TAU_POSTPROCESS="${PHASE_TAU_POSTPROCESS:-none}"
PHYSICAL_NORMAL_ITERATIONS="${PHYSICAL_NORMAL_ITERATIONS:-8}"
PHYSICAL_NORMAL_LEARNING_RATE="${PHYSICAL_NORMAL_LEARNING_RATE:-0.05}"
PHYSICAL_NORMAL_ENVELOPE="${PHYSICAL_NORMAL_ENVELOPE:-poly}"
PHYSICAL_NORMAL_PROJECTION_MODE="${PHYSICAL_NORMAL_PROJECTION_MODE:-block}"
PHYSICAL_NORMAL_COMPONENTS="${PHYSICAL_NORMAL_COMPONENTS:-translation}"
PHYSICAL_NORMAL_MAX_METRIC_NORM="${PHYSICAL_NORMAL_MAX_METRIC_NORM:-1.0}"
PHYSICAL_NORMAL_GRADIENT_CLIP="${PHYSICAL_NORMAL_GRADIENT_CLIP:-10.0}"
PHYSICAL_NORMAL_PROTEIN_CLASH_DIST="${PHYSICAL_NORMAL_PROTEIN_CLASH_DIST:-2.0}"
PHYSICAL_NORMAL_LIGAND_CLASH_DIST="${PHYSICAL_NORMAL_LIGAND_CLASH_DIST:-2.2}"
PHYSICAL_NORMAL_MAX_CLASH_ATOMS="${PHYSICAL_NORMAL_MAX_CLASH_ATOMS:-256}"
PHYSICAL_NORMAL_WEIGHT_PEPTIDE="${PHYSICAL_NORMAL_WEIGHT_PEPTIDE:-1.0}"
PHYSICAL_NORMAL_WEIGHT_PROTEIN_CLASH="${PHYSICAL_NORMAL_WEIGHT_PROTEIN_CLASH:-1.0}"
PHYSICAL_NORMAL_WEIGHT_LIGAND_CLASH="${PHYSICAL_NORMAL_WEIGHT_LIGAND_CLASH:-1.0}"
PHYSICAL_NORMAL_WEIGHT_CONTACT_ANCHOR="${PHYSICAL_NORMAL_WEIGHT_CONTACT_ANCHOR:-0.25}"
PHYSICAL_NORMAL_WEIGHT_DISTANCE_ANCHOR="${PHYSICAL_NORMAL_WEIGHT_DISTANCE_ANCHOR:-1.0}"
PHYSICAL_NORMAL_WEIGHT_RESIDUAL="${PHYSICAL_NORMAL_WEIGHT_RESIDUAL:-0.1}"
PHYSICAL_NORMAL_WEIGHT_TEMPORAL="${PHYSICAL_NORMAL_WEIGHT_TEMPORAL:-0.2}"
PHYSICAL_NORMAL_OPTIMIZER="${PHYSICAL_NORMAL_OPTIMIZER:-adam}"
PHYSICAL_NORMAL_NUM_STARTS="${PHYSICAL_NORMAL_NUM_STARTS:-1}"
PHYSICAL_NORMAL_ROUTE_SEED_SCALE="${PHYSICAL_NORMAL_ROUTE_SEED_SCALE:-0.0}"
PHYSICAL_NORMAL_ROUTE_SEED_RANK="${PHYSICAL_NORMAL_ROUTE_SEED_RANK:-2}"
PHYSICAL_NORMAL_ROUTE_SEED_SMOOTHING_STEPS="${PHYSICAL_NORMAL_ROUTE_SEED_SMOOTHING_STEPS:-2}"
PHYSICAL_NORMAL_ROUTE_SEED="${PHYSICAL_NORMAL_ROUTE_SEED:-20260720}"
PHYSICAL_NORMAL_FRAME_AGGREGATION="${PHYSICAL_NORMAL_FRAME_AGGREGATION:-mean}"
PHYSICAL_NORMAL_FRAME_SOFTMAX_BETA="${PHYSICAL_NORMAL_FRAME_SOFTMAX_BETA:-10.0}"
PHYSICAL_NORMAL_LINE_SEARCH_STEPS="${PHYSICAL_NORMAL_LINE_SEARCH_STEPS:-8}"
PHYSICAL_NORMAL_LINE_SEARCH_SHRINK="${PHYSICAL_NORMAL_LINE_SEARCH_SHRINK:-0.5}"
PHYSICAL_NORMAL_ACCEPTANCE_TOLERANCE="${PHYSICAL_NORMAL_ACCEPTANCE_TOLERANCE:-1e-8}"
PEPTIDE_PROJECTION_ITERATIONS="${PEPTIDE_PROJECTION_ITERATIONS:-12}"
PEPTIDE_PROJECTION_RELAXATION="${PEPTIDE_PROJECTION_RELAXATION:-0.75}"
PEPTIDE_PROJECTION_ANCHOR_STRENGTH="${PEPTIDE_PROJECTION_ANCHOR_STRENGTH:-0.02}"
PEPTIDE_PROJECTION_MAX_TRANSLATION="${PEPTIDE_PROJECTION_MAX_TRANSLATION:-2.0}"
PEPTIDE_PROJECTION_ACTIVATION_LOSS_THRESHOLD="${PEPTIDE_PROJECTION_ACTIVATION_LOSS_THRESHOLD:-0.0}"
POSE_GRAPH_ITERATIONS="${POSE_GRAPH_ITERATIONS:-20}"
POSE_GRAPH_LEARNING_RATE="${POSE_GRAPH_LEARNING_RATE:-0.05}"
POSE_GRAPH_EDGE_WEIGHT="${POSE_GRAPH_EDGE_WEIGHT:-1.0}"
POSE_GRAPH_ANCHOR_WEIGHT="${POSE_GRAPH_ANCHOR_WEIGHT:-0.1}"
POSE_GRAPH_ROTATION_METRIC_SCALE="${POSE_GRAPH_ROTATION_METRIC_SCALE:-1.5}"
POSE_GRAPH_MAX_ROTATION="${POSE_GRAPH_MAX_ROTATION:-0.5}"
POSE_GRAPH_MAX_TRANSLATION="${POSE_GRAPH_MAX_TRANSLATION:-2.0}"
INTERACTION_PRIOR_FEATURE_MODE="${INTERACTION_PRIOR_FEATURE_MODE:-}"
INTERACTION_PRIOR_CKPT="${INTERACTION_PRIOR_CKPT:-}"
INTERACTION_PRIOR_TEMPERATURE="${INTERACTION_PRIOR_TEMPERATURE:-}"
INTERACTION_PRIOR_FEATURE_SCALE="${INTERACTION_PRIOR_FEATURE_SCALE:-}"
STAGE1V2_MODE="${STAGE1V2_MODE:-}"
STAGE1V2_CACHE_DIR="${STAGE1V2_CACHE_DIR:-}"
STAGE1V2_FEATURES="${STAGE1V2_FEATURES:-}"
STAGE1V2_FEATURE_SCALE="${STAGE1V2_FEATURE_SCALE:-}"
MAX_BATCHES="${MAX_BATCHES:-32}"
BATCH_SIZE="${BATCH_SIZE:-1}"
INCLUDE_PER_SAMPLE_METRICS="${INCLUDE_PER_SAMPLE_METRICS:-0}"
N_INTEGRATION_STEPS="${N_INTEGRATION_STEPS:-3}"
ACTIVE_DELTA="${ACTIVE_DELTA:-0.75}"
CONTACT_DIST="${CONTACT_DIST:-4.5}"
LIGAND_CLASH_DIST="${LIGAND_CLASH_DIST:-2.2}"
POCKET_THRESHOLD="${POCKET_THRESHOLD:-0.3}"
OUTPUT="${OUTPUT:-logs/stage2/transition_eval/${TAG}_${PATH_PARAMETERIZATION}_maxb${MAX_BATCHES}.json}"

case "$SPLIT" in
  train|val|test) ;;
  *)
    echo "ERROR: SPLIT must be one of train, val, test"
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
case "$INCLUDE_PER_SAMPLE_METRICS" in
  0|1) ;;
  *)
    echo "ERROR: INCLUDE_PER_SAMPLE_METRICS must be 0 or 1"
    exit 1
    ;;
esac
case "$PATH_PARAMETERIZATION" in
  checkpoint|flow|projected_flow|boundary_residual_v1|boundary_residual|pure_bridge|cartesian_backbone_bridge_v1|cartesian_peptide_projected_bridge_v1|chain_internal_bridge_v1|peptide_projected_bridge_v1|pose_graph_projected_bridge_v1|bridge_timewarp_v1|phase_orthogonal_residual_v1|phase_block_orthogonal_residual_v2|phase_physical_normal_v1) ;;
  *)
    echo "ERROR: unsupported PATH_PARAMETERIZATION=$PATH_PARAMETERIZATION"
    exit 1
    ;;
esac
if [[ -n "$BOUNDARY_RESIDUAL_ENVELOPE" ]]; then
  case "$BOUNDARY_RESIDUAL_ENVELOPE" in
    sin2|poly) ;;
    *)
      echo "ERROR: BOUNDARY_RESIDUAL_ENVELOPE must be sin2 or poly"
      exit 1
      ;;
  esac
fi
if [[ -n "$PHASE_RESIDUAL_TAU_MODE" ]]; then
  case "$PHASE_RESIDUAL_TAU_MODE" in
    learned|identity) ;;
    *)
      echo "ERROR: PHASE_RESIDUAL_TAU_MODE must be learned or identity"
      exit 1
      ;;
  esac
fi
if [[ -n "$PHASE_RESIDUAL_BRIDGE_MODE" ]]; then
  case "$PHASE_RESIDUAL_BRIDGE_MODE" in
    se3_geodesic|cartesian_backbone) ;;
    *)
      echo "ERROR: PHASE_RESIDUAL_BRIDGE_MODE must be se3_geodesic or cartesian_backbone"
      exit 1
      ;;
  esac
fi
if [[ -n "$PHASE_RESIDUAL_ENVELOPE" ]]; then
  case "$PHASE_RESIDUAL_ENVELOPE" in
    poly|sin2) ;;
    *)
      echo "ERROR: PHASE_RESIDUAL_ENVELOPE must be poly or sin2"
      exit 1
      ;;
  esac
fi
if [[ -n "$PHASE_RESIDUAL_MAX_METRIC_NORM" ]]; then
  python - <<PY
value = float("$PHASE_RESIDUAL_MAX_METRIC_NORM")
if value < 0.0:
    raise SystemExit("ERROR: PHASE_RESIDUAL_MAX_METRIC_NORM must be >= 0")
PY
fi
if [[ -n "$TERMINAL_PROJECTION_SCHEDULE" ]]; then
  case "$TERMINAL_PROJECTION_SCHEDULE" in
    smoothstep|smootherstep|late_smoother|quadratic) ;;
    *)
      echo "ERROR: TERMINAL_PROJECTION_SCHEDULE must be smoothstep, smootherstep, late_smoother, or quadratic"
      exit 1
      ;;
  esac
fi
if [[ -n "$TIME_WARP_LOGIT_SCALE" || -n "$TIME_WARP_RATE_EPS" || -n "$TIME_WARP_RATE_CLIP" ]]; then
  python - <<PY
logit_scale = float("${TIME_WARP_LOGIT_SCALE:-1.0}")
rate_eps = float("${TIME_WARP_RATE_EPS:-1e-3}")
rate_clip = float("${TIME_WARP_RATE_CLIP:-10.0}")
if logit_scale <= 0.0:
    raise SystemExit("ERROR: TIME_WARP_LOGIT_SCALE must be > 0")
if rate_eps <= 0.0:
    raise SystemExit("ERROR: TIME_WARP_RATE_EPS must be > 0")
if rate_clip < 0.0:
    raise SystemExit("ERROR: TIME_WARP_RATE_CLIP must be >= 0")
PY
fi

echo "=============================================="
echo "BINDRAE Stage-2 transition evaluator"
echo "=============================================="
echo "Job ID:            ${SLURM_JOB_ID:-NA}"
echo "Node:              ${SLURM_NODELIST:-NA}"
echo "Checkpoint:        $CHECKPOINT"
echo "Tag:               $TAG"
echo "Split:             $SPLIT"
echo "Valid samples:     $VALID_SAMPLES_FILE"
echo "Trust prechecked:  $TRUST_PRECHECKED_SAMPLES"
echo "Path mode:         $PATH_PARAMETERIZATION"
echo "Boundary envelope: ${BOUNDARY_RESIDUAL_ENVELOPE:-checkpoint_default}"
echo "Boundary scale:    ${BOUNDARY_RESIDUAL_SCALE:-checkpoint_default}"
echo "Projection sched:  ${TERMINAL_PROJECTION_SCHEDULE:-checkpoint_default}"
echo "Timewarp logit:    ${TIME_WARP_LOGIT_SCALE:-checkpoint_default}"
echo "Timewarp eps:      ${TIME_WARP_RATE_EPS:-checkpoint_default}"
echo "Timewarp clip:     ${TIME_WARP_RATE_CLIP:-checkpoint_default}"
echo "Phase tau mode:    ${PHASE_RESIDUAL_TAU_MODE:-checkpoint_default}"
echo "Phase bridge mode: ${PHASE_RESIDUAL_BRIDGE_MODE:-checkpoint_default}"
echo "Phase envelope:    ${PHASE_RESIDUAL_ENVELOPE:-checkpoint_default}"
echo "Phase scale:       ${PHASE_RESIDUAL_SCALE:-checkpoint_default}"
echo "Phase max norm:    ${PHASE_RESIDUAL_MAX_METRIC_NORM:-checkpoint_default}"
echo "Phase tau postproc:$PHASE_TAU_POSTPROCESS"
echo "Physical iter:     $PHYSICAL_NORMAL_ITERATIONS"
echo "Physical proj:     $PHYSICAL_NORMAL_PROJECTION_MODE"
echo "Physical comps:    $PHYSICAL_NORMAL_COMPONENTS"
echo "Physical max norm: $PHYSICAL_NORMAL_MAX_METRIC_NORM"
echo "Physical optimizer:$PHYSICAL_NORMAL_OPTIMIZER"
echo "Physical starts:   $PHYSICAL_NORMAL_NUM_STARTS"
echo "Physical seed scale:$PHYSICAL_NORMAL_ROUTE_SEED_SCALE"
echo "Physical frame agg:$PHYSICAL_NORMAL_FRAME_AGGREGATION"
echo "Peptide proj iter: $PEPTIDE_PROJECTION_ITERATIONS"
echo "Peptide proj relax:$PEPTIDE_PROJECTION_RELAXATION"
echo "Peptide proj anchor:$PEPTIDE_PROJECTION_ANCHOR_STRENGTH"
echo "Peptide proj max:  $PEPTIDE_PROJECTION_MAX_TRANSLATION"
echo "Peptide proj gate: $PEPTIDE_PROJECTION_ACTIVATION_LOSS_THRESHOLD"
echo "Pose graph iter:    $POSE_GRAPH_ITERATIONS"
echo "Pose graph lr:      $POSE_GRAPH_LEARNING_RATE"
echo "Pose graph edge:    $POSE_GRAPH_EDGE_WEIGHT"
echo "Pose graph anchor:  $POSE_GRAPH_ANCHOR_WEIGHT"
echo "Feature mode:      ${INTERACTION_PRIOR_FEATURE_MODE:-checkpoint_default}"
echo "Interaction prior: ${INTERACTION_PRIOR_CKPT:-checkpoint_default}"
echo "Prior temperature: ${INTERACTION_PRIOR_TEMPERATURE:-checkpoint_default}"
echo "Prior feat scale:  ${INTERACTION_PRIOR_FEATURE_SCALE:-checkpoint_default}"
echo "Max batches:       $MAX_BATCHES"
echo "Batch size:        $BATCH_SIZE"
echo "Per-sample stats:  $INCLUDE_PER_SAMPLE_METRICS"
echo "Integration steps: $N_INTEGRATION_STEPS"
echo "Lig clash dist:    $LIGAND_CLASH_DIST"
echo "Output:            $OUTPUT"
echo "Start:             $(date)"
echo "=============================================="

ARGS=(
  --checkpoint "$CHECKPOINT"
  --data_dir processed_data/triplets
  --split "$SPLIT"
  --valid_samples_file "$VALID_SAMPLES_FILE"
  --batch_size "$BATCH_SIZE"
  --num_workers 0
  --path_parameterization "$PATH_PARAMETERIZATION"
  --max_batches "$MAX_BATCHES"
  --n_integration_steps "$N_INTEGRATION_STEPS"
  --peptide_projection_iterations "$PEPTIDE_PROJECTION_ITERATIONS"
  --peptide_projection_relaxation "$PEPTIDE_PROJECTION_RELAXATION"
  --peptide_projection_anchor_strength "$PEPTIDE_PROJECTION_ANCHOR_STRENGTH"
  --peptide_projection_max_translation "$PEPTIDE_PROJECTION_MAX_TRANSLATION"
  --peptide_projection_activation_loss_threshold "$PEPTIDE_PROJECTION_ACTIVATION_LOSS_THRESHOLD"
  --pose_graph_iterations "$POSE_GRAPH_ITERATIONS"
  --pose_graph_learning_rate "$POSE_GRAPH_LEARNING_RATE"
  --pose_graph_edge_weight "$POSE_GRAPH_EDGE_WEIGHT"
  --pose_graph_anchor_weight "$POSE_GRAPH_ANCHOR_WEIGHT"
  --pose_graph_rotation_metric_scale "$POSE_GRAPH_ROTATION_METRIC_SCALE"
  --pose_graph_max_rotation "$POSE_GRAPH_MAX_ROTATION"
  --pose_graph_max_translation "$POSE_GRAPH_MAX_TRANSLATION"
  --physical_normal_iterations "$PHYSICAL_NORMAL_ITERATIONS"
  --physical_normal_learning_rate "$PHYSICAL_NORMAL_LEARNING_RATE"
  --physical_normal_envelope "$PHYSICAL_NORMAL_ENVELOPE"
  --physical_normal_projection_mode "$PHYSICAL_NORMAL_PROJECTION_MODE"
  --physical_normal_components "$PHYSICAL_NORMAL_COMPONENTS"
  --physical_normal_max_metric_norm "$PHYSICAL_NORMAL_MAX_METRIC_NORM"
  --physical_normal_gradient_clip "$PHYSICAL_NORMAL_GRADIENT_CLIP"
  --physical_normal_protein_clash_dist "$PHYSICAL_NORMAL_PROTEIN_CLASH_DIST"
  --physical_normal_ligand_clash_dist "$PHYSICAL_NORMAL_LIGAND_CLASH_DIST"
  --physical_normal_max_clash_atoms "$PHYSICAL_NORMAL_MAX_CLASH_ATOMS"
  --physical_normal_weight_peptide "$PHYSICAL_NORMAL_WEIGHT_PEPTIDE"
  --physical_normal_weight_protein_clash "$PHYSICAL_NORMAL_WEIGHT_PROTEIN_CLASH"
  --physical_normal_weight_ligand_clash "$PHYSICAL_NORMAL_WEIGHT_LIGAND_CLASH"
  --physical_normal_weight_contact_anchor "$PHYSICAL_NORMAL_WEIGHT_CONTACT_ANCHOR"
  --physical_normal_weight_distance_anchor "$PHYSICAL_NORMAL_WEIGHT_DISTANCE_ANCHOR"
  --physical_normal_weight_residual "$PHYSICAL_NORMAL_WEIGHT_RESIDUAL"
  --physical_normal_weight_temporal "$PHYSICAL_NORMAL_WEIGHT_TEMPORAL"
  --phase_tau_postprocess "$PHASE_TAU_POSTPROCESS"
  --physical_normal_optimizer "$PHYSICAL_NORMAL_OPTIMIZER"
  --physical_normal_num_starts "$PHYSICAL_NORMAL_NUM_STARTS"
  --physical_normal_route_seed_scale "$PHYSICAL_NORMAL_ROUTE_SEED_SCALE"
  --physical_normal_route_seed_rank "$PHYSICAL_NORMAL_ROUTE_SEED_RANK"
  --physical_normal_route_seed_smoothing_steps "$PHYSICAL_NORMAL_ROUTE_SEED_SMOOTHING_STEPS"
  --physical_normal_route_seed "$PHYSICAL_NORMAL_ROUTE_SEED"
  --physical_normal_frame_aggregation "$PHYSICAL_NORMAL_FRAME_AGGREGATION"
  --physical_normal_frame_softmax_beta "$PHYSICAL_NORMAL_FRAME_SOFTMAX_BETA"
  --physical_normal_line_search_steps "$PHYSICAL_NORMAL_LINE_SEARCH_STEPS"
  --physical_normal_line_search_shrink "$PHYSICAL_NORMAL_LINE_SEARCH_SHRINK"
  --physical_normal_acceptance_tolerance "$PHYSICAL_NORMAL_ACCEPTANCE_TOLERANCE"
  --active_delta "$ACTIVE_DELTA"
  --contact_dist "$CONTACT_DIST"
  --ligand_clash_dist "$LIGAND_CLASH_DIST"
  --pocket_threshold "$POCKET_THRESHOLD"
  --device cuda
  --output "$OUTPUT"
)

if [[ "$TRUST_PRECHECKED_SAMPLES" == "1" ]]; then
  ARGS+=(--trust_prechecked_samples)
fi
if [[ "$INCLUDE_PER_SAMPLE_METRICS" == "1" ]]; then
  ARGS+=(--include_per_sample_metrics)
fi
if [[ -n "$BOUNDARY_RESIDUAL_ENVELOPE" ]]; then
  ARGS+=(--boundary_residual_envelope "$BOUNDARY_RESIDUAL_ENVELOPE")
fi
if [[ -n "$BOUNDARY_RESIDUAL_SCALE" ]]; then
  ARGS+=(--boundary_residual_scale "$BOUNDARY_RESIDUAL_SCALE")
fi
if [[ -n "$TERMINAL_PROJECTION_SCHEDULE" ]]; then
  ARGS+=(--terminal_projection_schedule "$TERMINAL_PROJECTION_SCHEDULE")
fi
if [[ -n "$TIME_WARP_LOGIT_SCALE" ]]; then
  ARGS+=(--time_warp_logit_scale "$TIME_WARP_LOGIT_SCALE")
fi
if [[ -n "$TIME_WARP_RATE_EPS" ]]; then
  ARGS+=(--time_warp_rate_eps "$TIME_WARP_RATE_EPS")
fi
if [[ -n "$TIME_WARP_RATE_CLIP" ]]; then
  ARGS+=(--time_warp_rate_clip "$TIME_WARP_RATE_CLIP")
fi
if [[ -n "$PHASE_RESIDUAL_TAU_MODE" ]]; then
  ARGS+=(--phase_residual_tau_mode "$PHASE_RESIDUAL_TAU_MODE")
fi
if [[ -n "$PHASE_RESIDUAL_BRIDGE_MODE" ]]; then
  ARGS+=(--phase_residual_bridge_mode "$PHASE_RESIDUAL_BRIDGE_MODE")
fi
if [[ -n "$PHASE_RESIDUAL_ENVELOPE" ]]; then
  ARGS+=(--phase_residual_envelope "$PHASE_RESIDUAL_ENVELOPE")
fi
if [[ -n "$PHASE_RESIDUAL_SCALE" ]]; then
  ARGS+=(--phase_residual_scale "$PHASE_RESIDUAL_SCALE")
fi
if [[ -n "$PHASE_RESIDUAL_MAX_METRIC_NORM" ]]; then
  ARGS+=(--phase_residual_max_metric_norm "$PHASE_RESIDUAL_MAX_METRIC_NORM")
fi
if [[ -n "$INTERACTION_PRIOR_FEATURE_MODE" ]]; then
  ARGS+=(--interaction_prior_feature_mode "$INTERACTION_PRIOR_FEATURE_MODE")
fi
if [[ -n "$INTERACTION_PRIOR_CKPT" ]]; then
  ARGS+=(--interaction_prior_ckpt "$INTERACTION_PRIOR_CKPT")
fi
if [[ -n "$INTERACTION_PRIOR_TEMPERATURE" ]]; then
  ARGS+=(--interaction_prior_temperature "$INTERACTION_PRIOR_TEMPERATURE")
fi
if [[ -n "$INTERACTION_PRIOR_FEATURE_SCALE" ]]; then
  ARGS+=(--interaction_prior_feature_scale "$INTERACTION_PRIOR_FEATURE_SCALE")
fi
if [[ -n "$STAGE1V2_MODE" ]]; then
  ARGS+=(--stage1v2_posterior_feature_mode "$STAGE1V2_MODE")
fi
if [[ -n "$STAGE1V2_CACHE_DIR" ]]; then
  ARGS+=(--stage1v2_posterior_cache_dir "$STAGE1V2_CACHE_DIR")
fi
if [[ -n "$STAGE1V2_FEATURES" ]]; then
  ARGS+=(--stage1v2_posterior_feature_names "$STAGE1V2_FEATURES")
fi
if [[ -n "$STAGE1V2_FEATURE_SCALE" ]]; then
  ARGS+=(--stage1v2_posterior_feature_scale "$STAGE1V2_FEATURE_SCALE")
fi

python scripts/evaluate_stage2_transition_paths.py "${ARGS[@]}"

echo "Completed: $(date)"
echo "Output: $OUTPUT"
