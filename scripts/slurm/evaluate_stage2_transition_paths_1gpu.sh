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

ROOT=/mnt/inaisfs/data/home/zhaozc_criait/XinxiangWang/BINDRAE
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
VALID_SAMPLES_FILE="${VALID_SAMPLES_FILE:-ablation_subsets/stage2_lc_pgbf_val_1200_seed20260618.txt}"
INTERACTION_PRIOR_FEATURE_MODE="${INTERACTION_PRIOR_FEATURE_MODE:-}"
INTERACTION_PRIOR_CKPT="${INTERACTION_PRIOR_CKPT:-}"
INTERACTION_PRIOR_TEMPERATURE="${INTERACTION_PRIOR_TEMPERATURE:-}"
INTERACTION_PRIOR_FEATURE_SCALE="${INTERACTION_PRIOR_FEATURE_SCALE:-}"
MAX_BATCHES="${MAX_BATCHES:-32}"
BATCH_SIZE="${BATCH_SIZE:-1}"
N_INTEGRATION_STEPS="${N_INTEGRATION_STEPS:-3}"
ACTIVE_DELTA="${ACTIVE_DELTA:-0.75}"
CONTACT_DIST="${CONTACT_DIST:-4.5}"
POCKET_THRESHOLD="${POCKET_THRESHOLD:-0.3}"
OUTPUT="${OUTPUT:-logs/stage2/transition_eval/${TAG}_maxb${MAX_BATCHES}.json}"

echo "=============================================="
echo "BINDRAE Stage-2 transition evaluator"
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
echo "Max batches:       $MAX_BATCHES"
echo "Batch size:        $BATCH_SIZE"
echo "Integration steps: $N_INTEGRATION_STEPS"
echo "Output:            $OUTPUT"
echo "Start:             $(date)"
echo "=============================================="

ARGS=(
  --checkpoint "$CHECKPOINT"
  --data_dir processed_data/triplets
  --valid_samples_file "$VALID_SAMPLES_FILE"
  --batch_size "$BATCH_SIZE"
  --num_workers 0
  --max_batches "$MAX_BATCHES"
  --n_integration_steps "$N_INTEGRATION_STEPS"
  --active_delta "$ACTIVE_DELTA"
  --contact_dist "$CONTACT_DIST"
  --pocket_threshold "$POCKET_THRESHOLD"
  --device cuda
  --output "$OUTPUT"
)

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

python scripts/evaluate_stage2_transition_paths.py "${ARGS[@]}"

echo "Completed: $(date)"
echo "Output: $OUTPUT"
