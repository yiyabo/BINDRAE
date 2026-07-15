#!/bin/bash
# Evaluate one Stage-2 checkpoint against all held-out MD replicas.

#SBATCH --job-name=s2_mdref_eval
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:A100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
#SBATCH --time=04:00:00
#SBATCH --output=logs/slurm/stage2_mdref_eval_%j.out
#SBATCH --error=logs/slurm/stage2_mdref_eval_%j.err

set -euo pipefail
unset LD_PRELOAD
unset PROXYCHAINS_CONF_FILE
unset PROXYCHAINS_QUIET_MODE

ROOT="${ROOT:-/mnt/inaisfs/data/home/zhaozc_criait/XinxiangWang/BINDRAE}"
ENV_PREFIX=/mnt/inaisfs/data/home/zhaozc_criait/miniconda3/envs/BINDRAE
export PATH=/data/soft/slurm/24.11.4/bin:$ENV_PREFIX/bin:$PATH
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"

cd "$ROOT"
mkdir -p logs/slurm logs/stage2/md_reference_eval
source /mnt/inaisfs/data/home/zhaozc_criait/miniconda3/etc/profile.d/conda.sh
conda activate BINDRAE

CHECKPOINT="${CHECKPOINT:?CHECKPOINT is required}"
MD_REFERENCE_CACHE_DIR="${MD_REFERENCE_CACHE_DIR:?MD_REFERENCE_CACHE_DIR is required}"
OUTPUT="${OUTPUT:?OUTPUT is required}"
SPLIT="${SPLIT:-train}"
VALID_SAMPLES_FILE="${VALID_SAMPLES_FILE:?VALID_SAMPLES_FILE is required}"
PATH_PARAMETERIZATION="${PATH_PARAMETERIZATION:-checkpoint}"
STAGE1V2_MODE="${STAGE1V2_MODE:-oracle_motion}"
STAGE1V2_CACHE_DIR="${STAGE1V2_CACHE_DIR:?STAGE1V2_CACHE_DIR is required}"
N_INTEGRATION_STEPS="${N_INTEGRATION_STEPS:-20}"
MAX_BATCHES="${MAX_BATCHES:-}"
MD_MIN_PHASE_CONFIDENCE="${MD_MIN_PHASE_CONFIDENCE:-0.05}"

ARGS=(
  --checkpoint "$CHECKPOINT"
  --data_dir processed_data/triplets
  --split "$SPLIT"
  --valid_samples_file "$VALID_SAMPLES_FILE"
  --trust_prechecked_samples
  --batch_size 1
  --num_workers 0
  --path_parameterization "$PATH_PARAMETERIZATION"
  --stage1v2_posterior_feature_mode "$STAGE1V2_MODE"
  --stage1v2_posterior_cache_dir "$STAGE1V2_CACHE_DIR"
  --n_integration_steps "$N_INTEGRATION_STEPS"
  --md_reference_cache_dir "$MD_REFERENCE_CACHE_DIR"
  --md_min_phase_confidence "$MD_MIN_PHASE_CONFIDENCE"
  --device cuda
  --output "$OUTPUT"
)
if [[ -n "$MAX_BATCHES" ]]; then
  ARGS+=(--max_batches "$MAX_BATCHES")
fi

echo "Checkpoint:    $CHECKPOINT"
echo "Path mode:     $PATH_PARAMETERIZATION"
echo "MD references: $MD_REFERENCE_CACHE_DIR"
echo "Samples:       $VALID_SAMPLES_FILE"
echo "Path steps:    $N_INTEGRATION_STEPS"
echo "Output:        $OUTPUT"
echo "Start:         $(date)"

python scripts/evaluate_stage2_md_reference_paths.py "${ARGS[@]}"

echo "Completed: $(date)"
echo "Output:    $OUTPUT"
