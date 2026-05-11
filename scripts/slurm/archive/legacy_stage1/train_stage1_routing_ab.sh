#!/bin/bash
#SBATCH --job-name=s1_routing_ab
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:A100:8
#SBATCH --cpus-per-task=32
#SBATCH --mem=384G
#SBATCH --time=24:00:00
#SBATCH --output=logs/slurm/stage1_routing_ab_%j.out
#SBATCH --error=logs/slurm/stage1_routing_ab_%j.err

set -e
unset LD_PRELOAD
unset PROXYCHAINS_CONF_FILE
unset PROXYCHAINS_QUIET_MODE

export PATH=/mnt/inaisfs/data/home/zhaozc_criait/miniconda3/bin:$PATH
export PYTHONUNBUFFERED=1
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export OMP_NUM_THREADS=4
export NCCL_DEBUG=WARN

source /mnt/inaisfs/data/home/zhaozc_criait/miniconda3/etc/profile.d/conda.sh
conda activate BINDRAE

cd /mnt/inaisfs/data/home/zhaozc_criait/XinxiangWang/BINDRAE
mkdir -p logs/slurm logs/stage1 checkpoints/stage1

ARM="${ARM:-adapter}"
CKPT_TAG="routing_${ARM}_$(date +%Y%m%d_%H%M%S)"

ADAPTER_FLAGS=""
if [ "$ARM" = "adapter" ]; then
    ADAPTER_FLAGS="--use_pocket_routing_adapter --pocket_routing_hidden 128 --pocket_routing_layers 2 --pocket_routing_gate_threshold 0.3 --pocket_routing_residual_scale 0.5"
fi

echo "=============================================="
echo "Stage-1 Routing A/B: ARM=${ARM}"
echo "=============================================="
echo "Job ID:      $SLURM_JOB_ID"
echo "Node:        $SLURM_NODELIST"
echo "Checkpoint:  checkpoints/stage1/$CKPT_TAG"
echo "Log dir:     logs/stage1/$CKPT_TAG"
echo "Adapter:     ${ADAPTER_FLAGS:-none}"
echo "Start:       $(date)"
echo "=============================================="

nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader
echo ""
python -c "import torch; print(f'PyTorch {torch.__version__}, CUDA {torch.version.cuda}, GPUs: {torch.cuda.device_count()}')"
echo ""

torchrun \
  --standalone \
  --nproc_per_node=8 \
  scripts/train_stage1.py \
  --model_size enhanced_ligand \
  --data_dir processed_data/triplets \
  --valid_samples_file /mnt/inaisfs/data/home/zhaozc_criait/XinxiangWang/BINDRAE/processed_data/triplets/train_valid.txt \
  --val_samples_file /mnt/inaisfs/data/home/zhaozc_criait/XinxiangWang/BINDRAE/processed_data/triplets/val_valid.txt \
  --sample_metadata_file sample_metadata.json \
  --batch_size 8 \
  --num_workers 4 \
  --max_n_res 1600 \
  --length_bucketed_sampling \
  --residue_budget 1600 \
  --lr 8e-4 \
  --warmup_steps 1000 \
  --lr_scheduler plateau \
  --plateau_factor 0.5 \
  --plateau_patience 2 \
  --min_lr_scale 0.01 \
  --pocket_warmup_steps 2000 \
  --ligand_gate_warmup_steps 2000 \
  --max_epochs 50 \
  --grad_clip 1.0 \
  --patience 15 \
  --w_fape 1.0 \
  --w_chi 1.0 \
  --w_clash 0.1 \
  --selection_metric pocket_chi1_acc \
  --save_epoch_checkpoints \
  --save_dir checkpoints/stage1/${CKPT_TAG} \
  --log_dir logs/stage1/${CKPT_TAG} \
  --distributed \
  ${ADAPTER_FLAGS}

echo ""
echo "=============================================="
echo "Training completed: $(date)"
echo "ARM: ${ARM}"
echo "Checkpoint saved to: checkpoints/stage1/$CKPT_TAG"
echo "=============================================="
