#!/bin/bash
#SBATCH -J stage2_ddp
#SBATCH -N 1
#SBATCH -n 8
#SBATCH --gres=gpu:8
#SBATCH -o logs/slurm/train_stage2_ddp_%j.out
#SBATCH -e logs/slurm/train_stage2_ddp_%j.err
#SBATCH -t 7-00:00:00
#SBATCH --mem=384G

unset LD_PRELOAD
unset PROXYCHAINS_CONF_FILE
unset PROXYCHAINS_QUIET_MODE

export PATH=/mnt/inaisfs/data/home/zhaozc_criait/miniconda3/bin:$PATH
export PYTHONUNBUFFERED=1
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export OMP_NUM_THREADS=1

eval "$(/mnt/inaisfs/data/home/zhaozc_criait/miniconda3/bin/conda shell.bash hook)"
conda activate BINDRAE

cd /mnt/inaisfs/data/home/zhaozc_criait/XinxiangWang/BINDRAE

echo "=============================================="
echo "Stage-2 Training (8 GPUs DDP)"
echo "=============================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Start time: $(date)"
echo "=============================================="

nvidia-smi --query-gpu=index,name,memory.total,memory.free --format=csv
echo ""

python -c "
import torch
print(f'PyTorch: {torch.__version__}')
print(f'CUDA: {torch.version.cuda}')
print(f'GPU count: {torch.cuda.device_count()}')
for i in range(torch.cuda.device_count()):
    print(f'  GPU {i}: {torch.cuda.get_device_name(i)}')
"

echo ""
echo "Starting DDP training with 8 GPUs..."
echo "=============================================="

torchrun --standalone --nproc_per_node=8 scripts/train_stage2.py \
    --data_dir processed_data/triplets \
    --batch_size 1 \
    --accum_steps 4 \
    --lr 2e-5 \
    --max_epochs 50 \
    --grad_clip 0.3 \
    --stage1_ckpt checkpoints/stage1/full_6gpu_bs24_20260125_234713/best_model.pt \
    --save_dir checkpoints/stage2_ddp \
    --log_dir logs/stage2_ddp \
    --device cuda \
    --distributed

echo ""
echo "=============================================="
echo "Training completed!"
echo "End time: $(date)"
echo "=============================================="
