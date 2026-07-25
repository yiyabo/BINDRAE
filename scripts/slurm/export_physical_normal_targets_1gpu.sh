#!/bin/bash
# Export one shard of canonical Path-4 physical-normal pseudo-targets.

#SBATCH --job-name=s2_phys_targets
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:A100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=04:00:00
#SBATCH --output=logs/slurm/stage2_physical_targets_%j.out
#SBATCH --error=logs/slurm/stage2_physical_targets_%j.err

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
mkdir -p logs/slurm
source /mnt/inaisfs/data/home/zhaozc_criait/miniconda3/etc/profile.d/conda.sh
conda activate BINDRAE

CHECKPOINT="${CHECKPOINT:-}"
DATA_DIR="${DATA_DIR:-processed_data/triplets}"
SPLIT="${SPLIT:-train}"
VALID_SAMPLES_FILE="${VALID_SAMPLES_FILE:-}"
OUTPUT_DIR="${OUTPUT_DIR:-logs/stage2/physical_normal_targets_${SLURM_JOB_ID:-manual}}"
NUM_SHARDS="${NUM_SHARDS:-1}"
SHARD_INDEX="${SHARD_INDEX:-0}"
MANIFEST="${MANIFEST:-$OUTPUT_DIR/manifest_shard$(printf '%03d' "$SHARD_INDEX").json}"
NUM_WORKERS="${NUM_WORKERS:-0}"
MAX_BATCHES="${MAX_BATCHES:-}"
MAX_SAMPLES="${MAX_SAMPLES:-}"
SKIP_EXISTING="${SKIP_EXISTING:-1}"
STAGE1V2_POSTERIOR_FEATURE_MODE="${STAGE1V2_POSTERIOR_FEATURE_MODE:-oracle_motion}"
STAGE1V2_POSTERIOR_CACHE_DIR="${STAGE1V2_POSTERIOR_CACHE_DIR:-}"

N_INTEGRATION_STEPS="${N_INTEGRATION_STEPS:-20}"
PHYSICAL_NORMAL_ITERATIONS="${PHYSICAL_NORMAL_ITERATIONS:-8}"
PHYSICAL_NORMAL_LEARNING_RATE="${PHYSICAL_NORMAL_LEARNING_RATE:-0.05}"
PHYSICAL_NORMAL_MAX_METRIC_NORM="${PHYSICAL_NORMAL_MAX_METRIC_NORM:-1.0}"
PHYSICAL_NORMAL_MAX_CLASH_ATOMS="${PHYSICAL_NORMAL_MAX_CLASH_ATOMS:-256}"
PHYSICAL_NORMAL_WEIGHT_RESIDUAL="${PHYSICAL_NORMAL_WEIGHT_RESIDUAL:-20.0}"
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

if [[ -z "$CHECKPOINT" || -z "$VALID_SAMPLES_FILE" || -z "$STAGE1V2_POSTERIOR_CACHE_DIR" ]]; then
  echo "ERROR: CHECKPOINT, VALID_SAMPLES_FILE, and STAGE1V2_POSTERIOR_CACHE_DIR are required"
  exit 1
fi
case "$SKIP_EXISTING" in
  0|1) ;;
  *) echo "ERROR: SKIP_EXISTING must be 0 or 1"; exit 1 ;;
esac

ARGS=(
  --checkpoint "$CHECKPOINT"
  --data_dir "$DATA_DIR"
  --split "$SPLIT"
  --valid_samples_file "$VALID_SAMPLES_FILE"
  --output_dir "$OUTPUT_DIR"
  --manifest "$MANIFEST"
  --batch_size 1
  --num_workers "$NUM_WORKERS"
  --num_shards "$NUM_SHARDS"
  --shard_index "$SHARD_INDEX"
  --path_parameterization phase_physical_normal_v1
  --n_integration_steps "$N_INTEGRATION_STEPS"
  --physical_normal_iterations "$PHYSICAL_NORMAL_ITERATIONS"
  --physical_normal_learning_rate "$PHYSICAL_NORMAL_LEARNING_RATE"
  --physical_normal_envelope poly
  --physical_normal_projection_mode block
  --physical_normal_components translation
  --physical_normal_max_metric_norm "$PHYSICAL_NORMAL_MAX_METRIC_NORM"
  --physical_normal_max_clash_atoms "$PHYSICAL_NORMAL_MAX_CLASH_ATOMS"
  --physical_normal_weight_peptide 1.0
  --physical_normal_weight_protein_clash 1.0
  --physical_normal_weight_ligand_clash 1.0
  --physical_normal_weight_contact_anchor 0.25
  --physical_normal_weight_distance_anchor 1.0
  --physical_normal_weight_residual "$PHYSICAL_NORMAL_WEIGHT_RESIDUAL"
  --physical_normal_weight_temporal "$PHYSICAL_NORMAL_WEIGHT_TEMPORAL"
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
  --stage1v2_posterior_feature_mode "$STAGE1V2_POSTERIOR_FEATURE_MODE"
  --stage1v2_posterior_cache_dir "$STAGE1V2_POSTERIOR_CACHE_DIR"
  --trust_prechecked_samples
  --device cuda
)
if [[ "$SKIP_EXISTING" == "1" ]]; then ARGS+=(--skip_existing); fi
if [[ -n "$MAX_BATCHES" ]]; then ARGS+=(--max_batches "$MAX_BATCHES"); fi
if [[ -n "$MAX_SAMPLES" ]]; then ARGS+=(--max_samples "$MAX_SAMPLES"); fi

echo "Path-4 physical target shard $SHARD_INDEX/$NUM_SHARDS"
echo "Checkpoint: $CHECKPOINT"
echo "Subset:     $VALID_SAMPLES_FILE"
echo "Output:     $OUTPUT_DIR"
echo "Manifest:   $MANIFEST"
echo "Optimizer:  $PHYSICAL_NORMAL_OPTIMIZER"
echo "Starts:     $PHYSICAL_NORMAL_NUM_STARTS"
echo "Seed scale: $PHYSICAL_NORMAL_ROUTE_SEED_SCALE"
echo "Frame agg:  $PHYSICAL_NORMAL_FRAME_AGGREGATION"
echo "Start:      $(date)"
python scripts/export_physical_normal_targets.py "${ARGS[@]}"
echo "Done:       $(date)"
