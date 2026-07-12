#!/bin/bash
# Compare existing and chain-internal Stage-2 bridges on a fixed subset.

#SBATCH --job-name=s2_chain_bridge
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:A100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
#SBATCH --time=03:00:00
#SBATCH --output=logs/slurm/stage2_chain_bridge_%j.out
#SBATCH --error=logs/slurm/stage2_chain_bridge_%j.err

set -euo pipefail
unset LD_PRELOAD

ROOT="${ROOT:-/mnt/inaisfs/data/home/zhaozc_criait/XinxiangWang/BINDRAE}"
ENV_PREFIX=/mnt/inaisfs/data/home/zhaozc_criait/miniconda3/envs/BINDRAE
export PATH=/data/soft/slurm/24.11.4/bin:$ENV_PREFIX/bin:$PATH
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=4

cd "$ROOT"
mkdir -p logs/slurm logs/stage2/chain_bridge
source /mnt/inaisfs/data/home/zhaozc_criait/miniconda3/etc/profile.d/conda.sh
conda activate BINDRAE

CHECKPOINT="${CHECKPOINT:?CHECKPOINT is required}"
VALID_SAMPLES_FILE="${VALID_SAMPLES_FILE:?VALID_SAMPLES_FILE is required}"
N_PATH_STEPS="${N_PATH_STEPS:-4}"
MAX_BATCHES="${MAX_BATCHES:-}"
PROJECTION_ITERATIONS="${PROJECTION_ITERATIONS:-12}"
PROJECTION_RELAXATION="${PROJECTION_RELAXATION:-0.75}"
PROJECTION_ANCHOR_STRENGTH="${PROJECTION_ANCHOR_STRENGTH:-0.02}"
PROJECTION_MAX_TRANSLATION="${PROJECTION_MAX_TRANSLATION:-2.0}"
PROJECTION_ACTIVATION_LOSS_THRESHOLD="${PROJECTION_ACTIVATION_LOSS_THRESHOLD:-0.0}"
POSE_GRAPH_ITERATIONS="${POSE_GRAPH_ITERATIONS:-20}"
POSE_GRAPH_LEARNING_RATE="${POSE_GRAPH_LEARNING_RATE:-0.05}"
POSE_GRAPH_EDGE_WEIGHT="${POSE_GRAPH_EDGE_WEIGHT:-1.0}"
POSE_GRAPH_ANCHOR_WEIGHT="${POSE_GRAPH_ANCHOR_WEIGHT:-0.1}"
POSE_GRAPH_ROTATION_METRIC_SCALE="${POSE_GRAPH_ROTATION_METRIC_SCALE:-1.5}"
POSE_GRAPH_MAX_ROTATION="${POSE_GRAPH_MAX_ROTATION:-0.5}"
POSE_GRAPH_MAX_TRANSLATION="${POSE_GRAPH_MAX_TRANSLATION:-2.0}"
OUTPUT="${OUTPUT:-logs/stage2/chain_bridge/chain_internal_bridge.json}"

ARGS=(
  --checkpoint "$CHECKPOINT"
  --valid_samples_file "$VALID_SAMPLES_FILE"
  --n_path_steps "$N_PATH_STEPS"
  --projection_iterations "$PROJECTION_ITERATIONS"
  --projection_relaxation "$PROJECTION_RELAXATION"
  --projection_anchor_strength "$PROJECTION_ANCHOR_STRENGTH"
  --projection_max_translation "$PROJECTION_MAX_TRANSLATION"
  --projection_activation_loss_threshold "$PROJECTION_ACTIVATION_LOSS_THRESHOLD"
  --pose_graph_iterations "$POSE_GRAPH_ITERATIONS"
  --pose_graph_learning_rate "$POSE_GRAPH_LEARNING_RATE"
  --pose_graph_edge_weight "$POSE_GRAPH_EDGE_WEIGHT"
  --pose_graph_anchor_weight "$POSE_GRAPH_ANCHOR_WEIGHT"
  --pose_graph_rotation_metric_scale "$POSE_GRAPH_ROTATION_METRIC_SCALE"
  --pose_graph_max_rotation "$POSE_GRAPH_MAX_ROTATION"
  --pose_graph_max_translation "$POSE_GRAPH_MAX_TRANSLATION"
  --device cuda
  --output "$OUTPUT"
)
if [[ -n "$MAX_BATCHES" ]]; then
  ARGS+=(--max_batches "$MAX_BATCHES")
fi

python scripts/diagnose_stage2_chain_internal_bridge.py "${ARGS[@]}"
