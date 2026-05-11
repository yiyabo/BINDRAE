#!/bin/bash
# First concrete shared-prefix branch for E6-style screening.
# Usage:
#   PREFIX_TAG=prefix_e6_YYYYmmdd_HHMMSS sbatch scripts/slurm/train_stage1_branch_e6b_slowramp.sh

#SBATCH --job-name=s1_e6b_sr
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:A100:8
#SBATCH --cpus-per-task=32
#SBATCH --mem=384G
#SBATCH --time=24:00:00
#SBATCH --output=logs/slurm/stage1_e6b_slowramp_%j.out
#SBATCH --error=logs/slurm/stage1_e6b_slowramp_%j.err

set -e
unset LD_PRELOAD
unset PROXYCHAINS_CONF_FILE
unset PROXYCHAINS_QUIET_MODE

if [ -z "${PREFIX_TAG:-}" ]; then
  echo "ERROR: set PREFIX_TAG before submission, e.g. PREFIX_TAG=prefix_e6_20260415_120000 sbatch $0"
  exit 1
fi

export PATH=/mnt/inaisfs/data/home/zhaozc_criait/miniconda3/bin:$PATH
export PYTHONUNBUFFERED=1
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export OMP_NUM_THREADS=4
export NCCL_DEBUG=WARN

source /mnt/inaisfs/data/home/zhaozc_criait/miniconda3/etc/profile.d/conda.sh
conda activate BINDRAE

cd /mnt/inaisfs/data/home/zhaozc_criait/XinxiangWang/BINDRAE
mkdir -p logs/slurm logs/stage1 checkpoints/stage1

PREFIX_CKPT="checkpoints/stage1/${PREFIX_TAG}/epoch_003.pt"
BRANCH_TAG="e6b_slowramp_from_${PREFIX_TAG}_$(date +%Y%m%d_%H%M%S)"

if [ ! -f "$PREFIX_CKPT" ]; then
  echo "ERROR: prefix checkpoint not found: $PREFIX_CKPT"
  exit 1
fi

echo "=============================================="
echo "Stage-1 E6 branch: ligand_facing + slow ramp"
echo "=============================================="
echo "Job ID:      $SLURM_JOB_ID"
echo "Resume from: $PREFIX_CKPT"
echo "Checkpoint:  checkpoints/stage1/$BRANCH_TAG"
echo "Log dir:     logs/stage1/$BRANCH_TAG"
echo "=============================================="

torchrun \
  --standalone \
  --nproc_per_node=8 \
  scripts/train_stage1.py \
  --model_size medium \
  --data_dir processed_data/triplets \
  --valid_samples_file /mnt/inaisfs/data/home/zhaozc_criait/XinxiangWang/BINDRAE/processed_data/triplets/train_valid.txt \
  --val_samples_file /mnt/inaisfs/data/home/zhaozc_criait/XinxiangWang/BINDRAE/processed_data/triplets/val_valid.txt \
  --sample_metadata_file sample_metadata.json \
  --batch_size 8 \
  --num_workers 4 \
  --max_n_res 1600 \
  --length_bucketed_sampling \
  --residue_budget 1600 \
  --lr 1e-4 \
  --warmup_steps 1000 \
  --lr_scheduler plateau \
  --plateau_factor 0.5 \
  --plateau_patience 2 \
  --min_lr_scale 0.01 \
  --pocket_warmup_steps 8000 \
  --ligand_gate_warmup_steps 8000 \
  --max_epochs 12 \
  --grad_clip 1.0 \
  --patience 20 \
  --w_fape 1.0 \
  --w_chi 1.0 \
  --w_clash 0.1 \
  --lambda_pchi1 0.2 \
  --pchi1_mask_mode ligand_facing \
  --pchi1_start_step 8000 \
  --pchi1_ramp_steps 8000 \
  --selection_metric pocket_chi1_acc \
  --save_epoch_checkpoints \
  --resume_from ${PREFIX_CKPT} \
  --save_dir checkpoints/stage1/${BRANCH_TAG} \
  --log_dir logs/stage1/${BRANCH_TAG} \
  --distributed
