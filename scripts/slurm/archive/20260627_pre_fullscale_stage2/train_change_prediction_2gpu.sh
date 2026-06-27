#!/bin/bash
#SBATCH --job-name=s1_change_ddp
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2
#SBATCH --gres=gpu:A100:2
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=12:00:00
#SBATCH --output=logs/slurm/stage1_change_ddp_%j.out
#SBATCH --error=logs/slurm/stage1_change_ddp_%j.err

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
mkdir -p logs/slurm logs/stage1/change_prediction_ddp checkpoints/stage1/change_prediction_ddp

TAG="change_ddp_2gpu_$(date +%Y%m%d_%H%M%S)"
SAVE_DIR="checkpoints/stage1/${TAG}"
LOG_DIR="logs/stage1/${TAG}"
EXTRA_ARGS=()
if [[ -n "${STAGE1_ENCODER_CHECKPOINT:-}" ]]; then
  EXTRA_ARGS+=(--stage1_encoder_checkpoint "$STAGE1_ENCODER_CHECKPOINT")
fi

echo "=============================================="
echo "Stage-1 Change-Prediction RAE Training (2-GPU DDP)"
echo "=============================================="
echo "Job ID:     $SLURM_JOB_ID"
echo "Node:       $SLURM_NODELIST"
echo "GPUs:       2x A100"
echo "Tag:        $TAG"
echo "Save dir:   $SAVE_DIR"
echo "Log dir:    $LOG_DIR"
echo "Stage1 encoder checkpoint: ${STAGE1_ENCODER_CHECKPOINT:-<random init>}"
echo "Start:      $(date)"
echo "=============================================="

torchrun --nproc_per_node=2 scripts/train_change_prediction.py \
  --data_dir processed_data/triplets \
  --val_samples_file /mnt/inaisfs/data/home/zhaozc_criait/XinxiangWang/BINDRAE/processed_data/triplets/val_valid.txt \
  --sample_metadata_file sample_metadata.json \
  --batch_size 4 \
  --max_n_res 1600 \
  --lr 1e-4 \
  --max_epochs 50 \
  --grad_clip 1.0 \
  --lambda_latent 1.0 \
  --lambda_recon 0.1 \
  --patience 10 \
  --save_dir "$SAVE_DIR" \
	  --log_dir "$LOG_DIR" \
	  --num_workers 2 \
	  --device cuda \
	  --distributed \
	  "${EXTRA_ARGS[@]}"

echo ""
echo "=============================================="
echo "Training completed: $(date)"
echo "Checkpoints: $SAVE_DIR"
echo "Metrics: $LOG_DIR/metrics.jsonl"
echo "=============================================="
