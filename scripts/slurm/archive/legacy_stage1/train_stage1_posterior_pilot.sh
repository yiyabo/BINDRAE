#!/bin/bash
#SBATCH --job-name=s1_posterior
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:A100:4
#SBATCH --cpus-per-task=16
#SBATCH --mem=160G
#SBATCH --time=08:00:00
#SBATCH --output=logs/slurm/stage1_posterior_%j.out
#SBATCH --error=logs/slurm/stage1_posterior_%j.err

set -euo pipefail
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

ROOT=/mnt/inaisfs/data/home/zhaozc_criait/XinxiangWang/BINDRAE
cd "$ROOT"
mkdir -p logs/slurm logs/stage1 checkpoints/stage1

TAG="posterior_pilot_$(date +%Y%m%d_%H%M%S)"
TRAIN_TXT="${TRAIN_TXT:-$ROOT/processed_data/triplets/train_valid.txt}"
VAL_TXT="${VAL_TXT:-$ROOT/processed_data/triplets/val_valid.txt}"
LAMBDA_ROT="${LAMBDA_ROT:-0.5}"
LAMBDA_CONTACT="${LAMBDA_CONTACT:-0.5}"
MAX_EPOCHS="${MAX_EPOCHS:-12}"

if [ -n "${RESUME_FROM:-}" ]; then
  RESUME_FLAGS="--resume_from ${RESUME_FROM}"
else
  RESUME_FLAGS=""
fi

echo "=============================================="
echo "Stage-1 posterior/contact pilot"
echo "=============================================="
echo "Job ID:          $SLURM_JOB_ID"
echo "Node:            $SLURM_NODELIST"
echo "Tag:             $TAG"
echo "Train list:      $TRAIN_TXT"
echo "Val list:        $VAL_TXT"
echo "Lambda rotamer:  $LAMBDA_ROT"
echo "Lambda contact:  $LAMBDA_CONTACT"
echo "Resume:          ${RESUME_FROM:-none}"
echo "Start:           $(date)"
echo "=============================================="

nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader
python -c "import torch; print(f'PyTorch {torch.__version__}, CUDA {torch.version.cuda}, GPUs: {torch.cuda.device_count()}')"

torchrun \
  --standalone \
  --nproc_per_node=4 \
  scripts/train_stage1.py \
  --model_size enhanced_ligand \
  --data_dir processed_data/triplets \
  --valid_samples_file "$TRAIN_TXT" \
  --val_samples_file "$VAL_TXT" \
  --sample_metadata_file sample_metadata.json \
  --batch_size 4 \
  --num_workers 4 \
  --max_n_res 1600 \
  --length_bucketed_sampling \
  --residue_budget 1600 \
  --lr 5e-4 \
  --warmup_steps 1000 \
  --lr_scheduler plateau \
  --plateau_factor 0.5 \
  --plateau_patience 2 \
  --min_lr_scale 0.01 \
  --pocket_warmup_steps 2000 \
  --ligand_gate_warmup_steps 2000 \
  --max_epochs "$MAX_EPOCHS" \
  --grad_clip 1.0 \
  --patience 8 \
  --w_fape 1.0 \
  --w_chi 1.0 \
  --w_clash 0.1 \
  --lambda_chi1_rotamer "$LAMBDA_ROT" \
  --lambda_contact "$LAMBDA_CONTACT" \
  --selection_metric pocket_chi1_acc \
  --compute_slow_metrics \
  --enable_dual_mask_audit \
  --save_epoch_checkpoints \
  --save_dir "checkpoints/stage1/${TAG}" \
  --log_dir "logs/stage1/${TAG}" \
  --distributed \
  $RESUME_FLAGS

echo ""
echo "=============================================="
echo "Posterior pilot completed: $(date)"
echo "Checkpoint saved to: checkpoints/stage1/$TAG"
echo "=============================================="
