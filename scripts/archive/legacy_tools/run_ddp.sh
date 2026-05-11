#!/bin/bash
#
# Stage-1 多卡 DDP 训练启动脚本
#
# 使用方法:
#   ./scripts/run_ddp.sh [NUM_GPUS] [BATCH_SIZE] [LR]
#
# 示例:
#   ./scripts/run_ddp.sh 2 32 2e-4      # 双卡，每卡batch=32，lr=2e-4
#   ./scripts/run_ddp.sh 4 16 4e-4      # 四卡，每卡batch=16，lr=4e-4
#

set -e

# 默认参数
NUM_GPUS=${1:-2}
BATCH_SIZE=${2:-32}
LR=${3:-2e-4}

# 数据目录
DATA_DIR="data/apo_holo_triplets"
VALID_SAMPLES="valid_samples.txt"
SAVE_DIR="outputs/stage1_ddp"

# 计算有效 batch size
EFFECTIVE_BATCH=$((BATCH_SIZE * NUM_GPUS))

echo "========================================"
echo "Stage-1 DDP Training"
echo "========================================"
echo "  GPUs: ${NUM_GPUS}"
echo "  Batch per GPU: ${BATCH_SIZE}"
echo "  Effective batch: ${EFFECTIVE_BATCH}"
echo "  Learning rate: ${LR}"
echo "  Data dir: ${DATA_DIR}"
echo "  Save dir: ${SAVE_DIR}"
echo "========================================"

# 使用 torchrun 启动分布式训练
torchrun \
    --standalone \
    --nproc_per_node=${NUM_GPUS} \
    scripts/train_stage1.py \
    --distributed \
    --data_dir ${DATA_DIR} \
    --valid_samples_file ${VALID_SAMPLES} \
    --save_dir ${SAVE_DIR} \
    --batch_size ${BATCH_SIZE} \
    --lr ${LR} \
    --max_epochs 100 \
    --num_workers 2 \
    --patience 20

echo "Training completed!"

