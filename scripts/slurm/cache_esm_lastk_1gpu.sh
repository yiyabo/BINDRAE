#!/bin/bash
# Upgrade selected Stage-2 triplet ESM caches with last-K ESM layers.

#SBATCH --job-name=esm_lastk
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:A100:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=100G
#SBATCH --time=04:00:00
#SBATCH --output=logs/slurm/esm_lastk_%j.out
#SBATCH --error=logs/slurm/esm_lastk_%j.err

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
mkdir -p logs/slurm

source /mnt/inaisfs/data/home/zhaozc_criait/miniconda3/etc/profile.d/conda.sh
conda activate BINDRAE

DATA_DIR="${DATA_DIR:-processed_data/triplets}"
SAMPLE_LIST_DEFAULT="ablation_subsets/esm_lastk_smoke_all25.txt"
SAMPLE_LIST="${SAMPLE_LIST:-$ROOT/$SAMPLE_LIST_DEFAULT}"
LAST_K_LAYERS="${LAST_K_LAYERS:-7}"
BATCH_SIZE="${BATCH_SIZE:-1}"
MAX_SAMPLES="${MAX_SAMPLES:-0}"
FORCE="${FORCE:-0}"

echo "=============================================="
echo "BINDRAE ESM last-K cache upgrade"
echo "=============================================="
echo "Job ID:        ${SLURM_JOB_ID:-NA}"
echo "Node:          ${SLURM_NODELIST:-NA}"
echo "Data dir:      $DATA_DIR"
echo "Sample list:   $SAMPLE_LIST"
echo "Last-K layers: $LAST_K_LAYERS"
echo "Batch size:    $BATCH_SIZE"
echo "Max samples:   $MAX_SAMPLES"
echo "Force:         $FORCE"
echo "Start:         $(date)"
echo "=============================================="

nvidia-smi --query-gpu=index,name,memory.total,memory.free --format=csv,noheader
python -c "import torch; print(f'PyTorch {torch.__version__}, CUDA {torch.version.cuda}, GPUs: {torch.cuda.device_count()}')"

ARGS=(
  scripts/cache_ahojdb_esm2.py
  --data-dir "$DATA_DIR"
  --batch-size "$BATCH_SIZE"
  --last-k-layers "$LAST_K_LAYERS"
  --sample-list "$SAMPLE_LIST"
  --max-samples "$MAX_SAMPLES"
)

if [[ "$FORCE" == "1" ]]; then
  ARGS+=(--force)
fi

python "${ARGS[@]}"

echo "Completed: $(date)"
