#!/bin/bash
#SBATCH --job-name=s1v2_audit
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:A100:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=120G
#SBATCH --time=02:00:00
#SBATCH --output=logs/slurm/stage1v2_audit_%j.out
#SBATCH --error=logs/slurm/stage1v2_audit_%j.err

set -euo pipefail
unset LD_PRELOAD
unset PROXYCHAINS_CONF_FILE
unset PROXYCHAINS_QUIET_MODE

ROOT=/mnt/inaisfs/data/home/zhaozc_criait/XinxiangWang/BINDRAE
ENV_PREFIX=/mnt/inaisfs/data/home/zhaozc_criait/miniconda3/envs/BINDRAE

export PATH=/data/soft/slurm/24.11.4/bin:$ENV_PREFIX/bin:$PATH
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"

cd "$ROOT"
mkdir -p logs/slurm logs/stage1v2_audits

source /mnt/inaisfs/data/home/zhaozc_criait/miniconda3/etc/profile.d/conda.sh
conda activate BINDRAE

CHECKPOINT="${CHECKPOINT:-checkpoints/stage1v2/stage1v2_posterior_train4096_lcpgbf_cb20e_3gpu_fullcf_bs16_cs384_h384_20260622_222239/best_model.pt}"
DATA_DIR="${DATA_DIR:-processed_data/triplets}"
LABEL_DIR="${LABEL_DIR:-logs/stage1v2_teacher_posteriors/holo_truth_val512_lcpgbf_20260622}"
SPLIT="${SPLIT:-val}"
VALID_SAMPLES_FILE="${VALID_SAMPLES_FILE:-}"
OUTPUT_DIR="${OUTPUT_DIR:-logs/stage1v2_audits}"
TAG="${TAG:-}"
BATCH_SIZE="${BATCH_SIZE:-16}"
NUM_WORKERS="${NUM_WORKERS:-4}"
MAX_SAMPLES="${MAX_SAMPLES:-0}"
COUNTERFACTUALS="${COUNTERFACTUALS:-nolig,shuffled,translated}"
DEVICE="${DEVICE:-cuda}"

ARGS=(
  scripts/audit_stage1v2_posterior.py
  --checkpoint "$CHECKPOINT"
  --data_dir "$DATA_DIR"
  --label_dir "$LABEL_DIR"
  --split "$SPLIT"
  --output_dir "$OUTPUT_DIR"
  --batch_size "$BATCH_SIZE"
  --num_workers "$NUM_WORKERS"
  --max_samples "$MAX_SAMPLES"
  --counterfactuals "$COUNTERFACTUALS"
  --device "$DEVICE"
)

if [[ -n "$VALID_SAMPLES_FILE" ]]; then
  ARGS+=(--valid_samples_file "$VALID_SAMPLES_FILE")
fi
if [[ -n "$TAG" ]]; then
  ARGS+=(--tag "$TAG")
fi
if [[ "${NO_AMP:-0}" == "1" ]]; then
  ARGS+=(--no_amp)
fi

echo "Stage-1-v2 posterior checkpoint audit"
echo "  checkpoint:       $CHECKPOINT"
echo "  split:            $SPLIT"
echo "  label_dir:        $LABEL_DIR"
echo "  output_dir:       $OUTPUT_DIR"
echo "  counterfactuals:  $COUNTERFACTUALS"
echo "  started:          $(date)"

python "${ARGS[@]}"

echo "completed: $(date)"
