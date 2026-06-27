#!/bin/bash
#SBATCH --job-name=s1_chi_head
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:A100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=logs/slurm/stage1_chi_head_%j.out
#SBATCH --error=logs/slurm/stage1_chi_head_%j.err

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
mkdir -p logs/slurm logs/stage1/chi_head checkpoints/stage1/chi_head

PREDICTOR_CKPT="checkpoints/stage1/change_fast_1gpu_20260616_164925/best_model.pt"
TAG="chi_head_1gpu_$(date +%Y%m%d_%H%M%S)"
SAVE_DIR="checkpoints/stage1/${TAG}"
LOG_DIR="logs/stage1/${TAG}"

echo "=============================================="
echo "Stage-1 Chi Head Training"
echo "=============================================="
echo "Job ID:          $SLURM_JOB_ID"
echo "Node:            $SLURM_NODELIST"
echo "Predictor CKPT:  $PREDICTOR_CKPT"
echo "Tag:             $TAG"
echo "Save dir:        $SAVE_DIR"
echo "Log dir:         $LOG_DIR"
echo "Start:           $(date)"
echo "=============================================="

python scripts/train_chi_head_simple.py \
  --data_dir processed_data/triplets \
  --predictor_ckpt "$PREDICTOR_CKPT" \
  --val_samples_file /mnt/inaisfs/data/home/zhaozc_criait/XinxiangWang/BINDRAE/processed_data/triplets/val_valid.txt \
  --sample_metadata_file sample_metadata.json \
  --batch_size 4 \
  --max_n_res 1600 \
  --lr 1e-3 \
  --max_epochs 50 \
  --grad_clip 1.0 \
  --c_s 384 \
  --torsion_hidden 128 \
  --patience 10 \
  --save_dir "$SAVE_DIR" \
  --log_dir "$LOG_DIR" \
  --num_workers 2 \
  --device cuda

echo ""
echo "=============================================="
echo "Training completed: $(date)"
echo "Checkpoints: $SAVE_DIR"
echo "Metrics: $LOG_DIR/metrics.jsonl"
echo "=============================================="
