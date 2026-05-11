#!/bin/bash
# Stage-1 全量训练（1节点 3×A100，73k 训练样本，100 epoch）
# 预估时间：~48-72h（effective_batch=24，~3042 steps/epoch）
#
# 提交方式：
#   export PATH=/data/soft/slurm/24.11.4/bin:$PATH
#   sbatch scripts/slurm/train_stage1_3gpu.sh

#SBATCH --job-name=s1_3gpu
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:A100:3
#SBATCH --cpus-per-task=24
#SBATCH --mem=192G
#SBATCH --time=72:00:00
#SBATCH --output=logs/slurm/stage1_3gpu_%j.out
#SBATCH --error=logs/slurm/stage1_3gpu_%j.err

set -e

unset LD_PRELOAD
unset PROXYCHAINS_CONF_FILE
unset PROXYCHAINS_QUIET_MODE

export PATH=/mnt/inaisfs/data/home/zhaozc_criait/miniconda3/bin:$PATH
export PYTHONUNBUFFERED=1
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export OMP_NUM_THREADS=4
export NCCL_DEBUG=WARN

eval "$(/mnt/inaisfs/data/home/zhaozc_criait/miniconda3/bin/conda shell.bash hook)"
conda activate BINDRAE

cd /mnt/inaisfs/data/home/zhaozc_criait/XinxiangWang/BINDRAE

mkdir -p logs/slurm logs/stage1 checkpoints/stage1

CKPT_TAG="3gpu_$(date +%Y%m%d_%H%M%S)"

echo "=============================================="
echo "Stage-1 Full Training (1 node × 3 A100)"
echo "=============================================="
echo "Job ID:      $SLURM_JOB_ID"
echo "Node:        $SLURM_NODELIST"
echo "Eff. batch:  $((3 * 8))"
echo "Checkpoint:  checkpoints/stage1/$CKPT_TAG"
echo "Start:       $(date)"
echo "=============================================="

nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader
echo ""

python -c "import torch; print(f'PyTorch {torch.__version__}, CUDA {torch.version.cuda}, GPUs: {torch.cuda.device_count()}')"
echo ""

echo "Starting full training..."
echo "=============================================="

torchrun \
    --standalone \
    --nproc_per_node=3 \
    scripts/train_stage1.py \
    --model_size enhanced_ligand \
    --data_dir processed_data/triplets \
    --valid_samples_file /mnt/inaisfs/data/home/zhaozc_criait/XinxiangWang/BINDRAE/processed_data/triplets/train_valid.txt \
    --val_samples_file /mnt/inaisfs/data/home/zhaozc_criait/XinxiangWang/BINDRAE/processed_data/triplets/val_valid.txt \
    --batch_size 8 \
    --num_workers 4 \
    --lr 1e-4 \
    --max_epochs 100 \
    --grad_clip 1.0 \
    --patience 20 \
    --save_dir checkpoints/stage1/${CKPT_TAG} \
    --log_dir logs/stage1/${CKPT_TAG} \
    --distributed

echo ""
echo "=============================================="
echo "Training completed: $(date)"
echo "Checkpoint saved to: checkpoints/stage1/${CKPT_TAG}"
echo "=============================================="
