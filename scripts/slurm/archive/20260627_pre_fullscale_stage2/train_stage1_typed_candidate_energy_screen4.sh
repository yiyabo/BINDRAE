#!/bin/bash
#SBATCH --job-name=s1_typed_screen4
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:A100:4
#SBATCH --cpus-per-task=8
#SBATCH --mem=240G
#SBATCH --time=08:00:00
#SBATCH --output=logs/slurm/stage1_typed_candidate_screen4_%j.out
#SBATCH --error=logs/slurm/stage1_typed_candidate_screen4_%j.err

set -euo pipefail
unset LD_PRELOAD
unset PROXYCHAINS_CONF_FILE
unset PROXYCHAINS_QUIET_MODE

ROOT=/mnt/inaisfs/data/home/zhaozc_criait/XinxiangWang/BINDRAE
ENV_PREFIX=/mnt/inaisfs/data/home/zhaozc_criait/miniconda3/envs/BINDRAE

export PATH=/data/soft/slurm/24.11.4/bin:$ENV_PREFIX/bin:$PATH
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=4
export NCCL_TIMEOUT=14400
export NCCL_DEBUG=WARN
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1

cd "$ROOT"
mkdir -p logs/slurm logs/stage1 checkpoints/stage1 tmp

source /mnt/inaisfs/data/home/zhaozc_criait/miniconda3/etc/profile.d/conda.sh
conda activate BINDRAE

TORCHRUN_BIN="$ENV_PREFIX/bin/torchrun"

TAG="${TAG:-typed_candidate_energy_screen4_$(date +%Y%m%d_%H%M%S)}"
TRAIN_TXT_FULL="${TRAIN_TXT:-$ROOT/processed_data/triplets/train_valid.txt}"
VAL_TXT_FULL="${VAL_TXT:-$ROOT/processed_data/triplets/val_valid.txt}"
SCREEN_TRAIN_SAMPLES="${SCREEN_TRAIN_SAMPLES:-6000}"
SCREEN_VAL_SAMPLES="${SCREEN_VAL_SAMPLES:-1024}"
RESUME_FROM="${RESUME_FROM:-$ROOT/checkpoints/stage1/geom_baseprior_20260501_170858/best_model.pt}"
MASTER_PORT="${MASTER_PORT:-29545}"
TYPED_DECOY_KIND="${TYPED_DECOY_KIND:-scrambled}"
MAX_EPOCHS="${MAX_EPOCHS:-6}"
LR="${LR:-3e-4}"
LAMBDA_GEOMETRY_CHI1="${LAMBDA_GEOMETRY_CHI1:-0.2}"
LAMBDA_TYPED_CANDIDATE_ENERGY="${LAMBDA_TYPED_CANDIDATE_ENERGY:-5.0}"
SELECTION_METRIC="${SELECTION_METRIC:-typed_energy_gap_contact_switch}"
NUM_WORKERS="${NUM_WORKERS:-2}"
BATCH_SIZE="${BATCH_SIZE:-1}"
RESIDUE_BUDGET="${RESIDUE_BUDGET:-700}"

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
Stage-1 typed candidate interaction energy screening (4-GPU)
============================================================
Job ID:          ${SLURM_JOB_ID:-NA}
Node:            ${SLURM_NODELIST:-NA}
GPUs:            4 x A100 on one node
Tag:             $TAG
Resume:          $RESUME_FROM
Train samples:   $TRAIN_TXT_USED
Val samples:     $VAL_TXT_USED
Max epochs:      $MAX_EPOCHS
Typed decoy:     $TYPED_DECOY_KIND
Lambda typed:    $LAMBDA_TYPED_CANDIDATE_ENERGY
Selection metric:$SELECTION_METRIC
Num workers:     $NUM_WORKERS
Batch size/rank: $BATCH_SIZE
Residue budget:  $RESIDUE_BUDGET
Goal:            get early typed-energy signal without waiting for 8-GPU queue
Start:           $(date)
============================================================
EOF

nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader

"$TORCHRUN_BIN" \
  --nproc_per_node=4 \
  --master_port="$MASTER_PORT" \
  scripts/train_stage1.py \
  --model_size medium \
  --data_dir processed_data/triplets \
  --valid_samples_file "$TRAIN_TXT_USED" \
  --val_samples_file "$VAL_TXT_USED" \
  --sample_metadata_file sample_metadata.json \
  --batch_size "$BATCH_SIZE" \
  --num_workers "$NUM_WORKERS" \
  --max_n_res 700 \
  --length_bucketed_sampling \
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
  --patience 4 \
  --w_fape 1.0 \
  --w_chi 1.0 \
  --w_clash 0.1 \
  --lambda_contact 0.0 \
  --lambda_geometry_chi1 "$LAMBDA_GEOMETRY_CHI1" \
  --lambda_base_prior 0.0 \
  --lambda_typed_candidate_energy "$LAMBDA_TYPED_CANDIDATE_ENERGY" \
  --typed_candidate_decoy_kind "$TYPED_DECOY_KIND" \
  --typed_candidate_margin 0.05 \
  --typed_candidate_noharm_weight 0.0 \
  --typed_candidate_noncontact_zero_weight 0.0 \
  --g_noncontact_threshold 8.0 \
  --geometry_scorer_lr_scale 4.0 \
  --geometry_scorer_use_sgeo \
  --geometry_scorer_sgeo_dim 32 \
  --geometry_scorer_use_typed_energy \
  --geometry_scorer_typed_pair_dim 64 \
  --geometry_scorer_typed_cutoff 6.0 \
  --geometry_scorer_typed_init_scale 1.0 \
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
echo "Typed candidate energy 4-GPU screening completed: $(date)"
echo "Checkpoint: checkpoints/stage1/$TAG"
echo "Metrics:    logs/stage1/$TAG/metrics.jsonl"
echo "============================================================"
