#!/bin/bash
# Stage-1 正式训练脚本（2节点 × 8 A100 = 16 GPUs，全量数据集 82k 样本）
# 预估时间：~24h（100 epoch，effective_batch_size=64，~1281 steps/epoch）
#
# 提交方式：
#   export PATH=/data/soft/slurm/24.11.4/bin:$PATH
#   sbatch scripts/slurm/train_stage1_full.sh

#SBATCH --job-name=s1_full
#SBATCH --partition=gpu
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:A100:8
#SBATCH --cpus-per-task=32
#SBATCH --mem=384G
#SBATCH --time=48:00:00
#SBATCH --output=logs/slurm/stage1_full_%j.out
#SBATCH --error=logs/slurm/stage1_full_%j.err

set -e

unset LD_PRELOAD
unset PROXYCHAINS_CONF_FILE
unset PROXYCHAINS_QUIET_MODE

export PATH=/mnt/inaisfs/data/home/zhaozc_criait/miniconda3/bin:$PATH
export PYTHONUNBUFFERED=1
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export OMP_NUM_THREADS=4
export NCCL_DEBUG=WARN
# 多节点 NCCL 优化
export NCCL_SOCKET_IFNAME=^lo,docker
export NCCL_IB_DISABLE=0

eval "$(/mnt/inaisfs/data/home/zhaozc_criait/miniconda3/bin/conda shell.bash hook)"
conda activate BINDRAE

cd /mnt/inaisfs/data/home/zhaozc_criait/XinxiangWang/BINDRAE

mkdir -p logs/slurm logs/stage1 checkpoints/stage1

# ---- 多节点 torchrun 配置 ----
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_PORT=29500
export NNODES=$SLURM_NNODES
export NPROC_PER_NODE=8
export WORLD_SIZE=$((NNODES * NPROC_PER_NODE))

CKPT_TAG="full_${NNODES}node_$(date +%Y%m%d_%H%M%S)"

echo "=============================================="
echo "Stage-1 Full Training (${NNODES} nodes × 8 A100 = ${WORLD_SIZE} GPUs)"
echo "=============================================="
echo "Job ID:      $SLURM_JOB_ID"
echo "Nodes:       $SLURM_NODELIST"
echo "Master:      $MASTER_ADDR:$MASTER_PORT"
echo "World size:  $WORLD_SIZE"
echo "Eff. batch:  $((NPROC_PER_NODE * NNODES * 4))"
echo "Checkpoint:  checkpoints/stage1/$CKPT_TAG"
echo "Start:       $(date)"
echo "=============================================="

nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader
echo ""

python -c "import torch; print(f'PyTorch {torch.__version__}, CUDA {torch.version.cuda}, GPUs: {torch.cuda.device_count()}')"
echo ""

echo "Starting full training with ${WORLD_SIZE} GPUs..."
echo "=============================================="

# srun 启动每个节点上的 torchrun（每节点 1 个 srun task，torchrun 再 fork 8 个进程）
srun --ntasks=$NNODES --ntasks-per-node=1 bash -c "
torchrun \
    --nnodes=${NNODES} \
    --nproc_per_node=${NPROC_PER_NODE} \
    --rdzv_id=${SLURM_JOB_ID} \
    --rdzv_backend=c10d \
    --rdzv_endpoint=${MASTER_ADDR}:${MASTER_PORT} \
    scripts/train_stage1.py \
    --model_size enhanced_ligand \
    --data_dir processed_data/triplets \
    --valid_samples_file /mnt/inaisfs/data/home/zhaozc_criait/XinxiangWang/BINDRAE/processed_data/triplets/train_valid.txt \
    --val_samples_file /mnt/inaisfs/data/home/zhaozc_criait/XinxiangWang/BINDRAE/processed_data/triplets/val_valid.txt \
    --batch_size 4 \
    --num_workers 4 \
    --lr 1e-4 \
    --max_epochs 100 \
    --grad_clip 1.0 \
    --patience 20 \
    --save_dir checkpoints/stage1/${CKPT_TAG} \
    --log_dir logs/stage1/${CKPT_TAG} \
    --distributed
"

echo ""
echo "=============================================="
echo "Training completed: $(date)"
echo "Checkpoint saved to: checkpoints/stage1/${CKPT_TAG}"
echo "=============================================="
