#!/bin/bash
# Export OracleMotion-UB features and run the direct oracle-apply audit.
#
# Smoke:
#   sbatch scripts/slurm/export_oracle_motion_features_1gpu.sh
#
# Full validation subset:
#   sbatch --export=ALL,MAX_SAMPLES=0,VALID_SAMPLES_FILE=ablation_subsets/stage2_s1v2_val_512_seed20260622.txt scripts/slurm/export_oracle_motion_features_1gpu.sh

#SBATCH --job-name=oracle_motion
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:A100:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=100G
#SBATCH --time=04:00:00
#SBATCH --output=logs/slurm/oracle_motion_%j.out
#SBATCH --error=logs/slurm/oracle_motion_%j.err

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
mkdir -p logs/slurm logs/stage2_oracle_motion

source /mnt/inaisfs/data/home/zhaozc_criait/miniconda3/etc/profile.d/conda.sh
conda activate BINDRAE

SPLIT="${SPLIT:-val}"
DATA_DIR="${DATA_DIR:-processed_data/triplets}"
VALID_SAMPLES_FILE="${VALID_SAMPLES_FILE:-}"
TAG="${TAG:-oracle_motion_${SPLIT}_smoke$(date +%Y%m%d_%H%M%S)}"
OUTPUT_DIR="${OUTPUT_DIR:-logs/stage2_oracle_motion/${TAG}}"
BATCH_SIZE="${BATCH_SIZE:-1}"
NUM_WORKERS="${NUM_WORKERS:-2}"
ESM_NUM_LAYERS="${ESM_NUM_LAYERS:-1}"
MAX_BATCHES="${MAX_BATCHES:-0}"
MAX_SAMPLES="${MAX_SAMPLES:-16}"
CONTACT_DIST="${CONTACT_DIST:-4.5}"
CONTACT_TAU="${CONTACT_TAU:-0.75}"
POCKET_THRESHOLD="${POCKET_THRESHOLD:-0.3}"
MOVING_TRANS_THRESHOLD="${MOVING_TRANS_THRESHOLD:-0.5}"
MOVING_ROT_THRESHOLD="${MOVING_ROT_THRESHOLD:-0.25}"
MOVING_CHI_THRESHOLD_DEG="${MOVING_CHI_THRESHOLD_DEG:-30.0}"
TRANSLATION_SCALE="${TRANSLATION_SCALE:-5.0}"
DISTANCE_SCALE="${DISTANCE_SCALE:-5.0}"
DEVICE="${DEVICE:-cuda}"
SKIP_BAD_SAMPLES="${SKIP_BAD_SAMPLES:-0}"
BAD_SAMPLES_OUT="${BAD_SAMPLES_OUT:-}"

echo "=============================================="
echo "BINDRAE OracleMotion-UB feature export"
echo "=============================================="
echo "Job ID:             ${SLURM_JOB_ID:-NA}"
echo "Node:               ${SLURM_NODELIST:-NA}"
echo "Split:              $SPLIT"
echo "Data dir:           $DATA_DIR"
echo "Valid samples:      ${VALID_SAMPLES_FILE:-OFF}"
echo "Output dir:         $OUTPUT_DIR"
echo "Batch size:         $BATCH_SIZE"
echo "ESM num layers:     $ESM_NUM_LAYERS"
echo "Max batches:        $MAX_BATCHES"
echo "Max samples:        $MAX_SAMPLES"
echo "Skip bad samples:   $SKIP_BAD_SAMPLES"
echo "Bad samples out:    ${BAD_SAMPLES_OUT:-OFF}"
echo "Contact dist/tau:   $CONTACT_DIST / $CONTACT_TAU"
echo "Moving thresholds:  trans=$MOVING_TRANS_THRESHOLD rot=$MOVING_ROT_THRESHOLD chi_deg=$MOVING_CHI_THRESHOLD_DEG"
echo "Scales:             translation=$TRANSLATION_SCALE distance=$DISTANCE_SCALE"
echo "Start:              $(date)"
echo "=============================================="

nvidia-smi --query-gpu=index,name,memory.total,memory.free --format=csv,noheader
python -c "import torch; print(f'PyTorch {torch.__version__}, CUDA {torch.version.cuda}, GPUs: {torch.cuda.device_count()}')"

ARGS=(
  scripts/export_oracle_motion_features.py
  --data_dir "$DATA_DIR"
  --split "$SPLIT"
  --output_dir "$OUTPUT_DIR"
  --batch_size "$BATCH_SIZE"
  --num_workers "$NUM_WORKERS"
  --esm_num_layers "$ESM_NUM_LAYERS"
  --max_batches "$MAX_BATCHES"
  --max_samples "$MAX_SAMPLES"
  --device "$DEVICE"
  --contact_dist "$CONTACT_DIST"
  --contact_tau "$CONTACT_TAU"
  --pocket_threshold "$POCKET_THRESHOLD"
  --moving_trans_threshold "$MOVING_TRANS_THRESHOLD"
  --moving_rot_threshold "$MOVING_ROT_THRESHOLD"
  --moving_chi_threshold_deg "$MOVING_CHI_THRESHOLD_DEG"
  --translation_scale "$TRANSLATION_SCALE"
  --distance_scale "$DISTANCE_SCALE"
)

if [[ -n "$VALID_SAMPLES_FILE" ]]; then
  ARGS+=(--valid_samples_file "$VALID_SAMPLES_FILE")
fi
if [[ "${SKIP_DIRECT_APPLY:-0}" == "1" ]]; then
  ARGS+=(--skip_direct_apply)
fi
if [[ "$SKIP_BAD_SAMPLES" == "1" ]]; then
  ARGS+=(--skip_bad_samples)
fi
if [[ -n "$BAD_SAMPLES_OUT" ]]; then
  ARGS+=(--bad_samples_out "$BAD_SAMPLES_OUT")
fi

python "${ARGS[@]}"

echo "Completed: $(date)"
echo "Manifest: $OUTPUT_DIR/manifest.json"
echo "Direct apply summary: $OUTPUT_DIR/direct_oracle_apply_summary.json"
