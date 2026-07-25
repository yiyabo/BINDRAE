#!/bin/bash
# Export frozen Stage-2 paths for independent BINDRAE-MD/OpenMM scoring.

#SBATCH --job-name=p4_path_export
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:A100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=02:00:00
#SBATCH --output=logs/slurm/path4_candidate_export_%j.out
#SBATCH --error=logs/slurm/path4_candidate_export_%j.err

set -euo pipefail
unset LD_PRELOAD

ROOT="${ROOT:-/mnt/inaisfs/data/home/zhaozc_criait/XinxiangWang/BINDRAE}"
CHECKPOINT="${CHECKPOINT:?CHECKPOINT is required}"
VALID_SAMPLES_FILE="${VALID_SAMPLES_FILE:?VALID_SAMPLES_FILE is required}"
STAGE1V2_CACHE_DIR="${STAGE1V2_CACHE_DIR:?STAGE1V2_CACHE_DIR is required}"
OUTPUT_DIR="${OUTPUT_DIR:?OUTPUT_DIR is required}"
MANIFEST="${MANIFEST:-$OUTPUT_DIR/manifest_${SLURM_JOB_ID:-manual}.json}"

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"

source /mnt/inaisfs/data/home/zhaozc_criait/miniconda3/etc/profile.d/conda.sh
conda activate BINDRAE
cd "$ROOT"
mkdir -p logs/slurm "$OUTPUT_DIR"

ARGS=(
  --checkpoint "$CHECKPOINT"
  --data_dir processed_data/triplets
  --split "${SPLIT:-train}"
  --valid_samples_file "$VALID_SAMPLES_FILE"
  --trust_prechecked_samples
  --batch_size 1
  --num_workers 0
  --path_parameterization "${PATH_PARAMETERIZATION:-phase_block_orthogonal_residual_v2}"
  --phase_residual_tau_mode "${PHASE_RESIDUAL_TAU_MODE:-learned}"
  --phase_residual_bridge_mode "${PHASE_RESIDUAL_BRIDGE_MODE:-cartesian_backbone}"
  --phase_residual_scale "${PHASE_RESIDUAL_SCALE:-0.0}"
  --phase_warp_variant "${PHASE_WARP_VARIANT:-chain_nonmonotone}"
  --phase_chain_residual_scale "${PHASE_CHAIN_RESIDUAL_SCALE:-1.0}"
  --phase_chain_smoothing_steps "${PHASE_CHAIN_SMOOTHING_STEPS:-1}"
  --phase_tau_postprocess "${PHASE_TAU_POSTPROCESS:-cummax}"
  --stage1v2_posterior_feature_mode "${STAGE1V2_MODE:-oracle_motion}"
  --stage1v2_posterior_cache_dir "$STAGE1V2_CACHE_DIR"
  --n_integration_steps "${N_INTEGRATION_STEPS:-20}"
  --device cuda
  --output_dir "$OUTPUT_DIR"
  --manifest "$MANIFEST"
  --candidate_label "${CANDIDATE_LABEL:-frozen_path3_cummax}"
  --max_samples "${MAX_SAMPLES:-1}"
  --num_shards "${NUM_SHARDS:-1}"
  --shard_index "${SHARD_INDEX:-0}"
)

if [[ "${SKIP_EXISTING:-0}" == "1" ]]; then
  ARGS+=(--skip_existing)
fi

python scripts/export_stage2_path_candidates.py "${ARGS[@]}"
