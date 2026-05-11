#!/bin/bash
# Stage-1 调试验证脚本（1节点 8 A100，debug 子集 2000 样本，5 epoch）
# 用途：同步代码后先跑这个，确认修复后的代码可以正常训练，再提交正式任务
#
# 提交方式：
#   export PATH=/data/soft/slurm/24.11.4/bin:$PATH
#   sbatch scripts/slurm/train_stage1_debug.sh

#SBATCH --job-name=s1_debug
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:A100:4
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=02:00:00
#SBATCH --output=logs/slurm/stage1_debug_%j.out
#SBATCH --error=logs/slurm/stage1_debug_%j.err

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

echo "=============================================="
echo "Stage-1 Debug Training (1 node, 8 A100)"
echo "=============================================="
echo "Job ID:   $SLURM_JOB_ID"
echo "Node:     $SLURM_NODELIST"
echo "Start:    $(date)"
echo "=============================================="

nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader
echo ""

python -c "import torch; print(f'PyTorch {torch.__version__}, CUDA {torch.version.cuda}, GPUs: {torch.cuda.device_count()}')"
echo ""

TAG="debug_$(date +%Y%m%d_%H%M%S)"

# 从 debug 子集 JSON 生成 valid_samples txt（dataset 需要 txt 格式）
DEBUG_TRAIN_TXT="/tmp/debug_train_${SLURM_JOB_ID}.txt"
DEBUG_VAL_TXT="/tmp/debug_val_${SLURM_JOB_ID}.txt"
python3 -c "import json; [print(x) for x in json.load(open('processed_data/triplets/subsets/debug/train.json'))]" > "$DEBUG_TRAIN_TXT"
python3 -c "import json; [print(x) for x in json.load(open('processed_data/triplets/subsets/debug/val.json'))]" > "$DEBUG_VAL_TXT"
echo "Debug train samples: $(wc -l < $DEBUG_TRAIN_TXT)"
echo "Debug val samples:   $(wc -l < $DEBUG_VAL_TXT)"
echo ""

echo "Starting debug training (debug subset ~1800 samples, 5 epochs)..."
echo "=============================================="

torchrun \
    --standalone \
    --nproc_per_node=4 \
    scripts/train_stage1.py \
    --model_size enhanced_ligand \
    --data_dir processed_data/triplets \
    --valid_samples_file "$DEBUG_TRAIN_TXT" \
    --val_samples_file "$DEBUG_VAL_TXT" \
    --batch_size 4 \
    --num_workers 4 \
    --lr 1e-4 \
    --max_epochs 5 \
    --grad_clip 1.0 \
    --save_dir checkpoints/stage1/$TAG \
    --log_dir logs/stage1/$TAG \
    --distributed

echo ""
echo "=============================================="
echo "Debug run completed: $(date)"
echo "=============================================="
