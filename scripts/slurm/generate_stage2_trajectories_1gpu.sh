#!/bin/bash
# Generate apo-to-holo Stage-2 trajectory examples from a trained checkpoint.

#SBATCH --job-name=s2_gen_traj
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:A100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=80G
#SBATCH --time=02:00:00
#SBATCH --output=logs/slurm/stage2_generate_trajectories_%j.out
#SBATCH --error=logs/slurm/stage2_generate_trajectories_%j.err

set -euo pipefail
unset LD_PRELOAD
unset PROXYCHAINS_CONF_FILE
unset PROXYCHAINS_QUIET_MODE

ROOT=/mnt/inaisfs/data/home/zhaozc_criait/XinxiangWang/BINDRAE
ENV_PREFIX=/mnt/inaisfs/data/home/zhaozc_criait/miniconda3/envs/BINDRAE

export PATH=/data/soft/slurm/24.11.4/bin:$ENV_PREFIX/bin:$PATH
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=4

cd "$ROOT"
mkdir -p logs/slurm logs/stage2/generated_paths

source /mnt/inaisfs/data/home/zhaozc_criait/miniconda3/etc/profile.d/conda.sh
conda activate BINDRAE

CHECKPOINT="${CHECKPOINT:?CHECKPOINT is required}"
TAG="${TAG:-$(basename "$(dirname "$CHECKPOINT")")}"
VALID_SAMPLES_FILE="${VALID_SAMPLES_FILE:-ablation_subsets/stage2_lc_pgbf_val_1200_seed20260618.txt}"
OUTPUT_DIR="${OUTPUT_DIR:-logs/stage2/generated_paths/${TAG}}"
INTERACTION_PRIOR_FEATURE_MODE="${INTERACTION_PRIOR_FEATURE_MODE:-}"
INTERACTION_PRIOR_CKPT="${INTERACTION_PRIOR_CKPT:-}"
INTERACTION_PRIOR_TEMPERATURE="${INTERACTION_PRIOR_TEMPERATURE:-}"
INTERACTION_PRIOR_FEATURE_SCALE="${INTERACTION_PRIOR_FEATURE_SCALE:-}"
STAGE1V2_MODE="${STAGE1V2_MODE:-}"
STAGE1V2_CACHE_DIR="${STAGE1V2_CACHE_DIR:-}"
STAGE1V2_FEATURES="${STAGE1V2_FEATURES:-}"
STAGE1V2_FEATURE_SCALE="${STAGE1V2_FEATURE_SCALE:-}"
MAX_SAMPLES="${MAX_SAMPLES:-4}"
MAX_BATCHES="${MAX_BATCHES:-}"
BATCH_SIZE="${BATCH_SIZE:-1}"
N_INTEGRATION_STEPS="${N_INTEGRATION_STEPS:-12}"
INTEGRATION_CHI_CLIP="${INTEGRATION_CHI_CLIP:-}"
INTEGRATION_ROT_CLIP="${INTEGRATION_ROT_CLIP:-}"
INTEGRATION_TRANS_CLIP="${INTEGRATION_TRANS_CLIP:-}"
SAMPLE_IDS="${SAMPLE_IDS:-}"

echo "=============================================="
echo "BINDRAE Stage-2 trajectory generation"
echo "=============================================="
echo "Job ID:            ${SLURM_JOB_ID:-NA}"
echo "Node:              ${SLURM_NODELIST:-NA}"
echo "Checkpoint:        $CHECKPOINT"
echo "Tag:               $TAG"
echo "Valid samples:     $VALID_SAMPLES_FILE"
echo "Feature mode:      ${INTERACTION_PRIOR_FEATURE_MODE:-checkpoint_default}"
echo "Interaction prior: ${INTERACTION_PRIOR_CKPT:-checkpoint_default}"
echo "Prior temperature: ${INTERACTION_PRIOR_TEMPERATURE:-checkpoint_default}"
echo "Prior feat scale:  ${INTERACTION_PRIOR_FEATURE_SCALE:-checkpoint_default}"
echo "Stage1v2 mode:     ${STAGE1V2_MODE:-checkpoint_default}"
echo "Stage1v2 cache:    ${STAGE1V2_CACHE_DIR:-checkpoint_default}"
echo "Stage1v2 features: ${STAGE1V2_FEATURES:-checkpoint_default}"
echo "Stage1v2 scale:    ${STAGE1V2_FEATURE_SCALE:-checkpoint_default}"
echo "Max samples:       $MAX_SAMPLES"
echo "Batch size:        $BATCH_SIZE"
echo "Integration steps: $N_INTEGRATION_STEPS"
echo "Chi clip:          ${INTEGRATION_CHI_CLIP:-checkpoint_default}"
echo "Rot clip:          ${INTEGRATION_ROT_CLIP:-checkpoint_default}"
echo "Trans clip:        ${INTEGRATION_TRANS_CLIP:-checkpoint_default}"
echo "Output dir:        $OUTPUT_DIR"
echo "Start:             $(date)"
echo "=============================================="

ARGS=(
  --checkpoint "$CHECKPOINT"
  --data_dir processed_data/triplets
  --split val
  --valid_samples_file "$VALID_SAMPLES_FILE"
  --batch_size "$BATCH_SIZE"
  --num_workers 0
  --max_samples "$MAX_SAMPLES"
  --n_integration_steps "$N_INTEGRATION_STEPS"
  --device cuda
  --output_dir "$OUTPUT_DIR"
)

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
if [[ -n "$SAMPLE_IDS" ]]; then
  # shellcheck disable=SC2206
  SAMPLE_ID_ARRAY=($SAMPLE_IDS)
  ARGS+=(--sample_ids "${SAMPLE_ID_ARRAY[@]}")
fi

python scripts/generate_stage2_trajectories.py "${ARGS[@]}"

echo "Completed: $(date)"
echo "Output dir: $OUTPUT_DIR"
