#!/bin/bash
# Evaluate Stage-2 trajectory reliability against cubic SE(3)+chi interpolation.

#SBATCH --job-name=s2_traj_rely
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:A100:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=20G
#SBATCH --time=04:00:00
#SBATCH --output=logs/slurm/stage2_trajectory_reliability_%j.out
#SBATCH --error=logs/slurm/stage2_trajectory_reliability_%j.err

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
mkdir -p logs/slurm logs/stage2/trajectory_reliability processed_data/triplets/ablation_subsets

source /mnt/inaisfs/data/home/zhaozc_criait/miniconda3/etc/profile.d/conda.sh
conda activate BINDRAE

CHECKPOINT="${CHECKPOINT:?CHECKPOINT is required}"
TAG="${TAG:-$(basename "$(dirname "$CHECKPOINT")")_traj_reliability_$(date +%Y%m%d_%H%M%S)}"
STAGE1V2_MODE="${STAGE1V2_MODE:-}"
STAGE1V2_CACHE_DIR="${STAGE1V2_CACHE_DIR:-logs/stage2_oracle_motion/oracle_motion_val512_direct_20260623_020114}"
STAGE1V2_FEATURES="${STAGE1V2_FEATURES:-}"
STAGE1V2_FEATURE_SCALE="${STAGE1V2_FEATURE_SCALE:-}"
VALID_SAMPLES_FILE="${VALID_SAMPLES_FILE:-}"
INDEX_FILE="${INDEX_FILE:-}"
VAL_N="${VAL_N:-64}"
SUBSET_SEED="${SUBSET_SEED:-20260623}"
BATCH_SIZE="${BATCH_SIZE:-1}"
NUM_WORKERS="${NUM_WORKERS:-0}"
MAX_BATCHES="${MAX_BATCHES:-}"
N_INTEGRATION_STEPS="${N_INTEGRATION_STEPS:-12}"
INTEGRATION_CHI_CLIP="${INTEGRATION_CHI_CLIP:-}"
INTEGRATION_ROT_CLIP="${INTEGRATION_ROT_CLIP:-}"
INTEGRATION_TRANS_CLIP="${INTEGRATION_TRANS_CLIP:-}"
INCLUDE_BOUNDARY_RESIDUAL="${INCLUDE_BOUNDARY_RESIDUAL:-0}"
INCLUDE_BOUNDARY_NATIVE="${INCLUDE_BOUNDARY_NATIVE:-0}"
BOUNDARY_RESIDUAL_ENVELOPE="${BOUNDARY_RESIDUAL_ENVELOPE:-sin2}"
BOUNDARY_RESIDUAL_SCALE="${BOUNDARY_RESIDUAL_SCALE:-1.0}"
OUTPUT="${OUTPUT:-logs/stage2/trajectory_reliability/${TAG}.json}"

case "$BOUNDARY_RESIDUAL_ENVELOPE" in
  sin2|poly) ;;
  *)
    echo "ERROR: BOUNDARY_RESIDUAL_ENVELOPE must be sin2 or poly"
    exit 1
    ;;
esac

if [[ -z "$VALID_SAMPLES_FILE" && -z "$INDEX_FILE" ]]; then
  SUBSET_REL="ablation_subsets/stage2_traj_reliability_val_${VAL_N}_seed${SUBSET_SEED}.txt"
  SUBSET_PATH="processed_data/triplets/${SUBSET_REL}"
  python - <<PY
import json
import random
from pathlib import Path

manifest_path = Path("$STAGE1V2_CACHE_DIR") / "manifest.json"
out_path = Path("$SUBSET_PATH")
with manifest_path.open() as handle:
    manifest = json.load(handle)
ids = [record["sample_id"] for record in manifest["records"]]
if int("$VAL_N") > len(ids):
    raise SystemExit(f"requested {int('$VAL_N')} IDs from {manifest_path}, only {len(ids)} available")
rng = random.Random(int("$SUBSET_SEED"))
rng.shuffle(ids)
chosen = ids[:int("$VAL_N")]
out_path.parent.mkdir(parents=True, exist_ok=True)
out_path.write_text("\\n".join(chosen) + "\\n")
print(f"Wrote {len(chosen)} IDs to {out_path}")
PY
  VALID_SAMPLES_FILE="$SUBSET_REL"
fi

echo "=============================================="
echo "BINDRAE Stage-2 trajectory reliability"
echo "=============================================="
echo "Job ID:            ${SLURM_JOB_ID:-NA}"
echo "Node:              ${SLURM_NODELIST:-NA}"
echo "Checkpoint:        $CHECKPOINT"
echo "Tag:               $TAG"
echo "Index file:        ${INDEX_FILE:-split_default}"
echo "Valid samples:     $VALID_SAMPLES_FILE"
echo "Stage1v2 mode:     ${STAGE1V2_MODE:-checkpoint_default}"
echo "Stage1v2 cache:    ${STAGE1V2_CACHE_DIR:-checkpoint_default}"
echo "Stage1v2 features: ${STAGE1V2_FEATURES:-checkpoint_default}"
echo "Stage1v2 scale:    ${STAGE1V2_FEATURE_SCALE:-checkpoint_default}"
echo "Batch size:        $BATCH_SIZE"
echo "Integration steps: $N_INTEGRATION_STEPS"
echo "Chi clip:          ${INTEGRATION_CHI_CLIP:-checkpoint_default}"
echo "Rot clip:          ${INTEGRATION_ROT_CLIP:-checkpoint_default}"
echo "Trans clip:        ${INTEGRATION_TRANS_CLIP:-checkpoint_default}"
echo "Boundary residual: $INCLUDE_BOUNDARY_RESIDUAL"
echo "Boundary native:   $INCLUDE_BOUNDARY_NATIVE"
echo "Boundary env:      $BOUNDARY_RESIDUAL_ENVELOPE"
echo "Boundary scale:    $BOUNDARY_RESIDUAL_SCALE"
echo "Output:            $OUTPUT"
echo "Start:             $(date)"
echo "=============================================="

ARGS=(
  --checkpoint "$CHECKPOINT"
  --data_dir processed_data/triplets
  --split val
  --batch_size "$BATCH_SIZE"
  --num_workers "$NUM_WORKERS"
  --n_integration_steps "$N_INTEGRATION_STEPS"
  --device cuda
  --output "$OUTPUT"
)

if [[ -n "$INDEX_FILE" ]]; then
  ARGS+=(--index_file "$INDEX_FILE")
fi
if [[ -n "$VALID_SAMPLES_FILE" ]]; then
  ARGS+=(--valid_samples_file "$VALID_SAMPLES_FILE")
fi
if [[ -n "$MAX_BATCHES" ]]; then
  ARGS+=(--max_batches "$MAX_BATCHES")
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
if [[ "$INCLUDE_BOUNDARY_RESIDUAL" == "1" || "$INCLUDE_BOUNDARY_RESIDUAL" == "true" ]]; then
  ARGS+=(--include_boundary_residual)
fi
if [[ "$INCLUDE_BOUNDARY_NATIVE" == "1" || "$INCLUDE_BOUNDARY_NATIVE" == "true" ]]; then
  ARGS+=(
    --include_boundary_native
    --boundary_residual_envelope "$BOUNDARY_RESIDUAL_ENVELOPE"
    --boundary_residual_scale "$BOUNDARY_RESIDUAL_SCALE"
  )
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

python scripts/evaluate_stage2_trajectory_reliability.py "${ARGS[@]}"

echo "Completed: $(date)"
echo "Output: $OUTPUT"
