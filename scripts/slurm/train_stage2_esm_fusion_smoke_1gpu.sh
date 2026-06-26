#!/bin/bash
# Stage-2 smoke test for RAEv2-like ESM last-K fusion.

#SBATCH --job-name=s2_esm_smoke
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:A100:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=100G
#SBATCH --time=02:00:00
#SBATCH --output=logs/slurm/s2_esm_smoke_%j.out
#SBATCH --error=logs/slurm/s2_esm_smoke_%j.err

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
mkdir -p logs/slurm logs/stage2 checkpoints/stage2

source /mnt/inaisfs/data/home/zhaozc_criait/miniconda3/etc/profile.d/conda.sh
conda activate BINDRAE

TAG="${TAG:-stage2_esm_last7_smoke_$(date +%Y%m%d_%H%M%S)}"
DATA_DIR="${DATA_DIR:-processed_data/triplets}"
TRAIN_LIST_DEFAULT="ablation_subsets/esm_lastk_smoke_train5.txt"
VAL_LIST_DEFAULT="ablation_subsets/esm_lastk_smoke_val20.txt"
TRAIN_LIST="${TRAIN_LIST:-$ROOT/$TRAIN_LIST_DEFAULT}"
VAL_LIST="${VAL_LIST:-$ROOT/$VAL_LIST_DEFAULT}"
ESM_NUM_LAYERS="${ESM_NUM_LAYERS:-7}"
ESM_FUSION_MODE="${ESM_FUSION_MODE:-sum}"
BATCH_SIZE="${BATCH_SIZE:-1}"
MAX_EPOCHS="${MAX_EPOCHS:-1}"
NUM_WORKERS="${NUM_WORKERS:-0}"

echo "=============================================="
echo "BINDRAE Stage-2 ESM last-K fusion smoke"
echo "=============================================="
echo "Job ID:          ${SLURM_JOB_ID:-NA}"
echo "Node:            ${SLURM_NODELIST:-NA}"
echo "Tag:             $TAG"
echo "Train list:      $TRAIN_LIST"
echo "Val list:        $VAL_LIST"
echo "ESM layers/mode: $ESM_NUM_LAYERS / $ESM_FUSION_MODE"
echo "Batch/epochs:    $BATCH_SIZE / $MAX_EPOCHS"
echo "Start:           $(date)"
echo "=============================================="

nvidia-smi --query-gpu=index,name,memory.total,memory.free --format=csv,noheader
python -c "import torch; print(f'PyTorch {torch.__version__}, CUDA {torch.version.cuda}, GPUs: {torch.cuda.device_count()}')"

python scripts/train_stage2.py \
  --data_dir "$DATA_DIR" \
  --valid_samples_file "$TRAIN_LIST" \
  --val_samples_file "$VAL_LIST" \
  --batch_size "$BATCH_SIZE" \
  --max_epochs "$MAX_EPOCHS" \
  --num_workers "$NUM_WORKERS" \
  --lr 2e-5 \
  --grad_clip 0.3 \
  --warmup_steps 0 \
  --no_stage1_prior \
  --w_prior 0.0 \
  --w_contact 0.1 \
  --n_integration_steps 2 \
  --n_geom_steps 1 \
  --geom_loss_every_n_steps 1 \
  --integration_chi_clip 5.0 \
  --integration_rot_clip 1.0 \
  --integration_trans_clip 5.0 \
  --esm_fusion_enabled \
  --esm_num_layers "$ESM_NUM_LAYERS" \
  --esm_fusion_mode "$ESM_FUSION_MODE" \
  --save_dir "checkpoints/stage2/${TAG}" \
  --log_dir "logs/stage2/${TAG}"

echo "Completed: $(date)"
