#!/bin/bash
#SBATCH --job-name=s1_guid_2n2g
#SBATCH --partition=gpu
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:A100:2
#SBATCH --cpus-per-task=8
#SBATCH --mem=192G
#SBATCH --time=24:00:00
#SBATCH --output=logs/slurm/stage1_ligand_guidance_2n2g_%j.out
#SBATCH --error=logs/slurm/stage1_ligand_guidance_2n2g_%j.err

set -euo pipefail
unset LD_PRELOAD
unset PROXYCHAINS_CONF_FILE
unset PROXYCHAINS_QUIET_MODE

ROOT=/mnt/inaisfs/data/home/zhaozc_criait/XinxiangWang/BINDRAE
ENV_PREFIX=/mnt/inaisfs/data/home/zhaozc_criait/miniconda3/envs/BINDRAE

export PATH=/data/soft/slurm/24.11.4/bin:$ENV_PREFIX/bin:$PATH
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=4

cd "$ROOT"
mkdir -p logs/slurm logs/stage1 checkpoints/stage1 tmp

source /mnt/inaisfs/data/home/zhaozc_criait/miniconda3/etc/profile.d/conda.sh
conda activate BINDRAE

# Multi-node setup
export MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
export MASTER_PORT=29500
export WORLD_SIZE=$SLURM_NTASKS
export RANK=$SLURM_PROCID

TAG="${TAG:-ligand_guidance_2n2g_$(date +%Y%m%d_%H%M%S)}"
TRAIN_TXT_FULL="${TRAIN_TXT:-$ROOT/processed_data/triplets/train_valid.txt}"
VAL_TXT_FULL="${VAL_TXT:-$ROOT/processed_data/triplets/val_valid.txt}"
SCREEN_TRAIN_SAMPLES="${SCREEN_TRAIN_SAMPLES:-0}"
SCREEN_VAL_SAMPLES="${SCREEN_VAL_SAMPLES:-0}"
RESUME_FROM="${RESUME_FROM:-$ROOT/checkpoints/stage1/geom_baseprior_20260501_170858/best_model.pt}"
MAX_EPOCHS="${MAX_EPOCHS:-20}"
LR="${LR:-3e-4}"
LAMBDA_GEOMETRY_CHI1="${LAMBDA_GEOMETRY_CHI1:-0.2}"
LAMBDA_GUIDANCE="${LAMBDA_GUIDANCE:-2.0}"
GUIDANCE_SWITCH_MARGIN="${GUIDANCE_SWITCH_MARGIN:-0.1}"
GUIDANCE_NONSWITCH_WEIGHT="${GUIDANCE_NONSWITCH_WEIGHT:-0.1}"
SELECTION_METRIC="${SELECTION_METRIC:-ligand_guidance_switch_loss}"
BATCH_SIZE="${BATCH_SIZE:-2}"
RESIDUE_BUDGET="${RESIDUE_BUDGET:-1400}"

TRAIN_TXT_USED="$TRAIN_TXT_FULL"
VAL_TXT_USED="$VAL_TXT_FULL"
if [ "$SCREEN_TRAIN_SAMPLES" -gt 0 ]; then
  TRAIN_TXT_USED="$ROOT/tmp/${TAG}_train_valid.txt"
  shuf -n "$SCREEN_TRAIN_SAMPLES" "$TRAIN_TXT_FULL" > "$TRAIN_TXT_USED"
fi
if [ "$SCREEN_VAL_SAMPLES" -gt 0 ]; then
  VAL_TXT_USED="$ROOT/tmp/${TAG}_val_valid.txt"
  shuf -n "$SCREEN_VAL_SAMPLES" "$VAL_TXT_FULL" > "$VAL_TXT_USED"
fi

cat <<EOF
============================================================
Stage-1 ligand internal guidance FULL training (2-node × 2-GPU, RAEv2-inspired)
============================================================
Job ID:          ${SLURM_JOB_ID:-NA}
Nodes:           ${SLURM_NODELIST:-NA}
GPUs:            2 x A100 per node (4 total)
Master:          $MASTER_ADDR:$MASTER_PORT
Tag:             $TAG
Resume:          $RESUME_FROM
Train samples:   $TRAIN_TXT_USED (full dataset)
Val samples:     $VAL_TXT_USED (full dataset)
Max epochs:      $MAX_EPOCHS
LR:              $LR
Lambda chi1:     $LAMBDA_GEOMETRY_CHI1
Lambda guidance: $LAMBDA_GUIDANCE
Switch margin:   $GUIDANCE_SWITCH_MARGIN
Nonswitch wt:    $GUIDANCE_NONSWITCH_WEIGHT
Selection metric:$SELECTION_METRIC
Batch size:      $BATCH_SIZE
Residue budget:  $RESIDUE_BUDGET
Goal:            FULL training with RAEv2-inspired ligand internal guidance
Start:           $(date)
============================================================
EOF

nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader

torchrun --nnodes=2 --nproc_per_node=2 --master_addr=$MASTER_ADDR --master_port=$MASTER_PORT --node_rank=$SLURM_PROCID scripts/train_stage1.py \
  --model_size medium \
  --data_dir processed_data/triplets \
  --valid_samples_file "$TRAIN_TXT_USED" \
  --val_samples_file "$VAL_TXT_USED" \
  --sample_metadata_file sample_metadata.json \
  --batch_size "$BATCH_SIZE" \
  --num_workers 2 \
  --max_n_res 700 \
  --residue_budget "$RESIDUE_BUDGET" \
  --lr "$LR" \
  --warmup_steps 200 \
  --lr_scheduler plateau \
  --plateau_factor 0.5 \
  --plateau_patience 2 \
  --min_lr_scale 0.1 \
  --pocket_warmup_steps 0 \
  --ligand_gate_warmup_steps 0 \
  --max_epochs "$MAX_EPOCHS" \
  --grad_clip 1.0 \
  --patience 8 \
  --w_fape 1.0 \
  --w_chi 1.0 \
  --w_clash 0.1 \
  --lambda_contact 0.0 \
  --lambda_geometry_chi1 "$LAMBDA_GEOMETRY_CHI1" \
  --lambda_base_prior 0.0 \
  --lambda_ligand_guidance "$LAMBDA_GUIDANCE" \
  --ligand_guidance_switch_margin "$GUIDANCE_SWITCH_MARGIN" \
  --ligand_guidance_nonswitch_weight "$GUIDANCE_NONSWITCH_WEIGHT" \
  --geometry_scorer_lr_scale 4.0 \
  --geometry_scorer_use_sgeo \
  --geometry_scorer_sgeo_dim 32 \
  --geometry_scorer_bounded_residual \
  --geometry_scorer_residual_max 5.0 \
  --geometry_scorer_residual_tau 3.0 \
  --geometry_scorer_gate_norm \
  --geometry_scorer_gate_clamp 6.0 \
  --geometry_scorer_gate_init_bias 2.0 \
  --reset_residual_and_gate_on_resume \
  --freeze_base_mlp \
  --freeze_gate_mlp \
  --detach_base_for_residual \
  --gate_warmup_open_steps 999999999 \
  --residual_beta_warmup_steps 1000 \
  --residual_beta_min 0.1 \
  --freeze_stage1_backbone_for_posteriors \
  --ddp_find_unused_parameters \
  --selection_metric "$SELECTION_METRIC" \
  --resume_from "$RESUME_FROM" \
  --save_epoch_checkpoints \
  --save_dir "checkpoints/stage1/${TAG}" \
  --log_dir "logs/stage1/${TAG}"

echo ""
echo "============================================================"
echo "Ligand guidance FULL training completed: $(date)"
echo "Checkpoint: checkpoints/stage1/$TAG"
echo "Metrics:    logs/stage1/$TAG/metrics.jsonl"
echo "============================================================"
