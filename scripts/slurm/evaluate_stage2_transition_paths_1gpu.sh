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
case "$PATH_PARAMETERIZATION" in
  checkpoint|flow|projected_flow|boundary_residual_v1|boundary_residual|pure_bridge) ;;
  *)
    echo "ERROR: PATH_PARAMETERIZATION must be checkpoint, flow, projected_flow, boundary_residual_v1, boundary_residual, or pure_bridge"
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
if [[ -n "$TERMINAL_PROJECTION_SCHEDULE" ]]; then
  case "$TERMINAL_PROJECTION_SCHEDULE" in
    smoothstep|smootherstep|late_smoother|quadratic) ;;
    *)
      echo "ERROR: TERMINAL_PROJECTION_SCHEDULE must be smoothstep, smootherstep, late_smoother, or quadratic"
      exit 1
      ;;
  esac
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
echo "Feature mode:      ${INTERACTION_PRIOR_FEATURE_MODE:-checkpoint_default}"
echo "Interaction prior: ${INTERACTION_PRIOR_CKPT:-checkpoint_default}"
echo "Prior temperature: ${INTERACTION_PRIOR_TEMPERATURE:-checkpoint_default}"
echo "Prior feat scale:  ${INTERACTION_PRIOR_FEATURE_SCALE:-checkpoint_default}"
echo "Max batches:       $MAX_BATCHES"
echo "Batch size:        $BATCH_SIZE"
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
if [[ -n "$BOUNDARY_RESIDUAL_ENVELOPE" ]]; then
  ARGS+=(--boundary_residual_envelope "$BOUNDARY_RESIDUAL_ENVELOPE")
fi
if [[ -n "$BOUNDARY_RESIDUAL_SCALE" ]]; then
  ARGS+=(--boundary_residual_scale "$BOUNDARY_RESIDUAL_SCALE")
fi
if [[ -n "$TERMINAL_PROJECTION_SCHEDULE" ]]; then
  ARGS+=(--terminal_projection_schedule "$TERMINAL_PROJECTION_SCHEDULE")
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
