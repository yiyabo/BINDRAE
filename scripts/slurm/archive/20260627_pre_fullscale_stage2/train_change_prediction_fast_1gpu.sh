#!/bin/bash
#SBATCH --job-name=s1_change_fast
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:A100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=logs/slurm/stage1_change_fast_%j.out
#SBATCH --error=logs/slurm/stage1_change_fast_%j.err

set -e
unset LD_PRELOAD
unset PROXYCHAINS_CONF_FILE
unset PROXYCHAINS_QUIET_MODE

export PATH=/mnt/inaisfs/data/home/zhaozc_criait/miniconda3/bin:$PATH
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=4

source /mnt/inaisfs/data/home/zhaozc_criait/miniconda3/etc/profile.d/conda.sh
conda activate BINDRAE

cd /mnt/inaisfs/data/home/zhaozc_criait/XinxiangWang/BINDRAE
mkdir -p logs/slurm logs/stage1/change_prediction_fast checkpoints/stage1/change_prediction_fast

TAG="change_fast_1gpu_$(date +%Y%m%d_%H%M%S)"
SAVE_DIR="checkpoints/stage1/${TAG}"
LOG_DIR="logs/stage1/${TAG}"

echo "=============================================="
echo "Stage-1 Change-Prediction RAE Fast Training (base-only baseline)"
echo "=============================================="
echo "Job ID:     $SLURM_JOB_ID"
echo "Node:       $SLURM_NODELIST"
echo "Tag:        $TAG"
echo "Save dir:   $SAVE_DIR"
echo "Log dir:    $LOG_DIR"
echo "Start:      $(date)"
echo "=============================================="

python scripts/train_change_prediction_fast.py \
  --latent_dir processed_data/latents \
  --batch_size 32 \
  --lr 1e-3 \
  --max_epochs 50 \
  --grad_clip 1.0 \
  --c_s 384 \
	  --delta_z_hidden 256 \
	  --delta_z_layers 3 \
	  --patience 10 \
	  --allow_base_only \
	  --save_dir "$SAVE_DIR" \
  --log_dir "$LOG_DIR" \
  --num_workers 4 \
  --device cuda

echo ""
echo "=============================================="
echo "Training completed: $(date)"
echo "Checkpoints: $SAVE_DIR"
echo "Metrics: $LOG_DIR/metrics.jsonl"
echo "=============================================="
