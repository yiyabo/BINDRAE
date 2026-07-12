#!/bin/bash
# Evaluate endpoint-exact bridge time-warp headroom on Stage-2 validation data.

#SBATCH --job-name=s2_timewarp
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:A100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=80G
#SBATCH --time=04:00:00
#SBATCH --output=logs/slurm/stage2_bridge_timewarp_%j.out
#SBATCH --error=logs/slurm/stage2_bridge_timewarp_%j.err

set -euo pipefail
unset LD_PRELOAD
unset PROXYCHAINS_CONF_FILE
unset PROXYCHAINS_QUIET_MODE

ROOT="${ROOT:-/mnt/inaisfs/data/home/zhaozc_criait/XinxiangWang/BINDRAE}"
ENV_PREFIX="${ENV_PREFIX:-/mnt/inaisfs/data/home/zhaozc_criait/miniconda3/envs/BINDRAE}"

export PATH=/data/soft/slurm/24.11.4/bin:$ENV_PREFIX/bin:$PATH
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"

cd "$ROOT"
mkdir -p logs/slurm logs/stage2/bridge_timewarp

source /mnt/inaisfs/data/home/zhaozc_criait/miniconda3/etc/profile.d/conda.sh
conda activate BINDRAE

CHECKPOINT="${CHECKPOINT:?CHECKPOINT is required}"
FREE_FLOW_CHECKPOINT="${FREE_FLOW_CHECKPOINT:-}"
TAG="${TAG:-$(basename "$(dirname "$CHECKPOINT")")}"
SPLIT="${SPLIT:-val}"
VALID_SAMPLES_FILE="${VALID_SAMPLES_FILE:-}"
TRUST_PRECHECKED_SAMPLES="${TRUST_PRECHECKED_SAMPLES:-0}"
BATCH_SIZE="${BATCH_SIZE:-1}"
NUM_WORKERS="${NUM_WORKERS:-0}"
MAX_BATCHES="${MAX_BATCHES:-512}"
N_PATH_STEPS="${N_PATH_STEPS:-16}"
N_TAU_GRID="${N_TAU_GRID:-33}"
TAU_TRANSITION_WEIGHT="${TAU_TRANSITION_WEIGHT:-0.0}"
# Commas are safe in this script default, but Slurm --export also uses commas.
# When passing from sbatch --export, prefer ":" or "+" separators.
METHODS="${METHODS:-pure_bridge,oracle_global,oracle_group,oracle_residue}"
REFERENCE_BRIDGE_MODE="${REFERENCE_BRIDGE_MODE:-se3_geodesic}"
FREE_FLOW_PATH_PARAMETERIZATION="${FREE_FLOW_PATH_PARAMETERIZATION:-flow}"
N_FREE_FLOW_STEPS="${N_FREE_FLOW_STEPS:-}"
INTEGRATION_CHI_CLIP="${INTEGRATION_CHI_CLIP:-}"
INTEGRATION_ROT_CLIP="${INTEGRATION_ROT_CLIP:-}"
INTEGRATION_TRANS_CLIP="${INTEGRATION_TRANS_CLIP:-}"
INTERACTION_PRIOR_FEATURE_MODE="${INTERACTION_PRIOR_FEATURE_MODE:-}"
INTERACTION_PRIOR_CKPT="${INTERACTION_PRIOR_CKPT:-}"
INTERACTION_PRIOR_TEMPERATURE="${INTERACTION_PRIOR_TEMPERATURE:-}"
INTERACTION_PRIOR_FEATURE_SCALE="${INTERACTION_PRIOR_FEATURE_SCALE:-}"
STAGE1V2_MODE="${STAGE1V2_MODE:-}"
STAGE1V2_CACHE_DIR="${STAGE1V2_CACHE_DIR:-}"
STAGE1V2_FEATURES="${STAGE1V2_FEATURES:-}"
STAGE1V2_FEATURE_SCALE="${STAGE1V2_FEATURE_SCALE:-}"
ACTIVE_DELTA="${ACTIVE_DELTA:-0.75}"
CONTACT_DIST="${CONTACT_DIST:-4.5}"
PATH_DIST_CAP="${PATH_DIST_CAP:-20.0}"
LIGAND_CLASH_DIST="${LIGAND_CLASH_DIST:-2.2}"
POCKET_THRESHOLD="${POCKET_THRESHOLD:-0.3}"
OUTPUT="${OUTPUT:-logs/stage2/bridge_timewarp/${TAG}_timewarp_maxb${MAX_BATCHES}_s${N_PATH_STEPS}_g${N_TAU_GRID}.json}"

case "$SPLIT" in
  train|val|test) ;;
  *)
    echo "ERROR: SPLIT must be train, val, or test"
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
case "$FREE_FLOW_PATH_PARAMETERIZATION" in
  flow|projected_flow|boundary_residual_v1|boundary_residual) ;;
  *)
    echo "ERROR: unsupported FREE_FLOW_PATH_PARAMETERIZATION=$FREE_FLOW_PATH_PARAMETERIZATION"
    exit 1
    ;;
esac
case "$REFERENCE_BRIDGE_MODE" in
  se3_geodesic|cartesian_backbone) ;;
  *)
    echo "ERROR: REFERENCE_BRIDGE_MODE must be se3_geodesic or cartesian_backbone"
    exit 1
    ;;
esac

echo "=============================================="
echo "BINDRAE Stage-2 bridge time-warp evaluator"
echo "=============================================="
echo "Job ID:             ${SLURM_JOB_ID:-NA}"
echo "Node:               ${SLURM_NODELIST:-NA}"
echo "Checkpoint:         $CHECKPOINT"
echo "Free-flow ckpt:     ${FREE_FLOW_CHECKPOINT:-checkpoint_default}"
echo "Tag:                $TAG"
echo "Split:              $SPLIT"
echo "Valid samples:      ${VALID_SAMPLES_FILE:-split_default}"
echo "Trust prechecked:   $TRUST_PRECHECKED_SAMPLES"
echo "Methods:            $METHODS"
echo "Reference bridge:   $REFERENCE_BRIDGE_MODE"
echo "Max batches:        $MAX_BATCHES"
echo "Batch size:         $BATCH_SIZE"
echo "Path steps:         $N_PATH_STEPS"
echo "Tau grid:           $N_TAU_GRID"
echo "Tau transition wt:  $TAU_TRANSITION_WEIGHT"
echo "Output:             $OUTPUT"
echo "Start:              $(date)"
echo "=============================================="

ARGS=(
  --checkpoint "$CHECKPOINT"
  --data_dir processed_data/triplets
  --split "$SPLIT"
  --batch_size "$BATCH_SIZE"
  --num_workers "$NUM_WORKERS"
  --device cuda
  --max_batches "$MAX_BATCHES"
  --n_path_steps "$N_PATH_STEPS"
  --n_tau_grid "$N_TAU_GRID"
  --tau_transition_weight "$TAU_TRANSITION_WEIGHT"
  --methods "$METHODS"
  --reference_bridge_mode "$REFERENCE_BRIDGE_MODE"
  --free_flow_path_parameterization "$FREE_FLOW_PATH_PARAMETERIZATION"
  --active_delta "$ACTIVE_DELTA"
  --contact_dist "$CONTACT_DIST"
  --path_dist_cap "$PATH_DIST_CAP"
  --ligand_clash_dist "$LIGAND_CLASH_DIST"
  --pocket_threshold "$POCKET_THRESHOLD"
  --output "$OUTPUT"
)

if [[ -n "$FREE_FLOW_CHECKPOINT" ]]; then
  ARGS+=(--free_flow_checkpoint "$FREE_FLOW_CHECKPOINT")
fi
if [[ -n "$VALID_SAMPLES_FILE" ]]; then
  ARGS+=(--valid_samples_file "$VALID_SAMPLES_FILE")
fi
if [[ "$TRUST_PRECHECKED_SAMPLES" == "1" ]]; then
  ARGS+=(--trust_prechecked_samples)
fi
if [[ -n "$N_FREE_FLOW_STEPS" ]]; then
  ARGS+=(--n_free_flow_steps "$N_FREE_FLOW_STEPS")
fi
if [[ -n "$INTEGRATION_CHI_CLIP" ]]; then
  ARGS+=(--integration_chi_clip "$INTEGRATION_CHI_CLIP")
fi
if [[ -n "$INTEGRATION_ROT_CLIP" ]]; then
  ARGS+=(--integration_rot_clip "$INTEGRATION_ROT_CLIP")
fi
if [[ -n "$INTEGRATION_TRANS_CLIP" ]]; then
  ARGS+=(--integration_trans_clip "$INTEGRATION_TRANS_CLIP")
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

python scripts/evaluate_stage2_bridge_timewarp.py "${ARGS[@]}"

echo "Completed: $(date)"
echo "Output: $OUTPUT"
