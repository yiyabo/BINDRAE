#!/bin/bash
#SBATCH --job-name=s1_geom_fix
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:A100:4
#SBATCH --cpus-per-task=4
#SBATCH --mem=192G
#SBATCH --time=72:00:00
#SBATCH --output=logs/slurm/stage1_geom_fixbin_%j.out
#SBATCH --error=logs/slurm/stage1_geom_fixbin_%j.err

set -euo pipefail
unset LD_PRELOAD
unset PROXYCHAINS_CONF_FILE
unset PROXYCHAINS_QUIET_MODE

ROOT=/mnt/inaisfs/data/home/zhaozc_criait/XinxiangWang/BINDRAE
ENV_PREFIX=/mnt/inaisfs/data/home/zhaozc_criait/miniconda3/envs/BINDRAE

export PATH=/data/soft/slurm/24.11.4/bin:$ENV_PREFIX/bin:$PATH
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=4
export NCCL_TIMEOUT=7200
export NCCL_DEBUG=WARN

cd "$ROOT"
mkdir -p logs/slurm logs/stage1 checkpoints/stage1

source /mnt/inaisfs/data/home/zhaozc_criait/miniconda3/etc/profile.d/conda.sh
conda activate BINDRAE

TORCHRUN_BIN="$ENV_PREFIX/bin/torchrun"

TAG="${TAG:-geom_m3_fixbin_$(date +%Y%m%d_%H%M%S)}"
TRAIN_TXT="${TRAIN_TXT:-$ROOT/processed_data/triplets/train_valid.txt}"
VAL_TXT="${VAL_TXT:-$ROOT/processed_data/triplets/val_valid.txt}"
# Resume from original backbone checkpoint (clean, no posterior tuning)
RESUME_FROM="${RESUME_FROM:-$ROOT/checkpoints/stage1/posterior_ligcond_unfreeze_run_20260427_065651/best_model.pt}"
LR="${LR:-1e-4}"
MAX_EPOCHS="${MAX_EPOCHS:-80}"

echo "=============================================="
echo "Stage-1 Geometry M3 + FIXED BIN (4-GPU)"
echo "=============================================="
echo "Job ID:     $SLURM_JOB_ID"
echo "Node:       $SLURM_NODELIST"
echo "Tag:        $TAG"
echo "Resume:     $RESUME_FROM"
echo "LR:         $LR"
echo "Max epochs: $MAX_EPOCHS"
echo "BIN FIX:    circular nearest center (g-/g+/t)"
echo "Mode:       M3 (geometry + s_geo 32d)"
echo "Start:      $(date)"
echo "=============================================="

nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader

"$TORCHRUN_BIN" \
  --nproc_per_node=4 \
  --master_port=29500 \
  scripts/train_stage1.py \
  --model_size medium \
  --data_dir processed_data/triplets \
  --valid_samples_file "$TRAIN_TXT" \
  --val_samples_file "$VAL_TXT" \
  --sample_metadata_file sample_metadata.json \
  --batch_size 2 \
  --num_workers 2 \
  --max_n_res 900 \
  --length_bucketed_sampling \
  --residue_budget 900 \
  --lr "$LR" \
  --warmup_steps 1000 \
  --lr_scheduler plateau \
  --plateau_factor 0.5 \
  --plateau_patience 5 \
  --min_lr_scale 0.01 \
  --pocket_warmup_steps 0 \
  --ligand_gate_warmup_steps 0 \
  --max_epochs "$MAX_EPOCHS" \
  --grad_clip 1.0 \
  --patience 15 \
  --w_fape 1.0 \
  --w_chi 1.0 \
  --w_clash 0.1 \
  --lambda_chi1_rotamer 0.0 \
  --lambda_contact 0.0 \
  --lambda_candidate_chi1 0.0 \
  --lambda_ligand_contrastive 0.0 \
  --lambda_geometry_chi1 1.0 \
  --geometry_scorer_lr_scale 1.0 \
  --geometry_scorer_use_sgeo \
  --geometry_scorer_sgeo_dim 32 \
  --freeze_stage1_backbone_for_posteriors \
  --unfreeze_ligand_conditioner_for_posteriors \
  --selection_metric chi1_rotamer_acc \
  --compute_slow_metrics \
  --save_epoch_checkpoints \
  --resume_from "$RESUME_FROM" \
  --save_dir "checkpoints/stage1/${TAG}" \
  --log_dir "logs/stage1/${TAG}"

echo ""
echo "=============================================="
echo "Fixed-bin training completed: $(date)"
echo "Checkpoint: checkpoints/stage1/$TAG"
echo "=============================================="
