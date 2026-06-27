#!/bin/bash
#SBATCH --job-name=s1_typed_smoke
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:A100:4
#SBATCH --cpus-per-task=8
#SBATCH --mem=220G
#SBATCH --time=01:00:00
#SBATCH --output=logs/slurm/stage1_typed_candidate_smoke_%j.out
#SBATCH --error=logs/slurm/stage1_typed_candidate_smoke_%j.err

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
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1

cd "$ROOT"
mkdir -p logs/slurm logs/stage1 checkpoints/stage1 tmp

source /mnt/inaisfs/data/home/zhaozc_criait/miniconda3/etc/profile.d/conda.sh
conda activate BINDRAE

TORCHRUN_BIN="$ENV_PREFIX/bin/torchrun"

TAG="${TAG:-typed_candidate_energy_smoke_$(date +%Y%m%d_%H%M%S)}"
TRAIN_TXT_FULL="${TRAIN_TXT:-$ROOT/processed_data/triplets/train_valid.txt}"
VAL_TXT_FULL="${VAL_TXT:-$ROOT/processed_data/triplets/val_valid.txt}"
SMOKE_TRAIN_SAMPLES="${SMOKE_TRAIN_SAMPLES:-128}"
SMOKE_VAL_SAMPLES="${SMOKE_VAL_SAMPLES:-64}"
SMOKE_TRAIN_TXT="$ROOT/tmp/${TAG}_train_valid.txt"
SMOKE_VAL_TXT="$ROOT/tmp/${TAG}_val_valid.txt"
RESUME_FROM="${RESUME_FROM:-$ROOT/checkpoints/stage1/geom_baseprior_20260501_170858/best_model.pt}"
MASTER_PORT="${MASTER_PORT:-29543}"
TYPED_DECOY_KIND="${TYPED_DECOY_KIND:-scrambled}"

head -n "$SMOKE_TRAIN_SAMPLES" "$TRAIN_TXT_FULL" > "$SMOKE_TRAIN_TXT"
head -n "$SMOKE_VAL_SAMPLES" "$VAL_TXT_FULL" > "$SMOKE_VAL_TXT"

cat <<EOF
============================================================
Stage-1 typed candidate interaction energy smoke (4-GPU)
============================================================
Job ID:          ${SLURM_JOB_ID:-NA}
Node:            ${SLURM_NODELIST:-NA}
GPUs:            4 x A100 on one node
Tag:             $TAG
Resume:          $RESUME_FROM
Train sample n:  $SMOKE_TRAIN_SAMPLES
Val sample n:    $SMOKE_VAL_SAMPLES
Typed decoy:     $TYPED_DECOY_KIND
Goal:            verify typed energy forward/backward + strict decoy plumbing
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
  --valid_samples_file "$SMOKE_TRAIN_TXT" \
  --val_samples_file "$SMOKE_VAL_TXT" \
  --sample_metadata_file sample_metadata.json \
  --batch_size 1 \
  --num_workers 2 \
  --max_n_res 700 \
  --length_bucketed_sampling \
  --residue_budget 700 \
  --lr 3e-4 \
  --warmup_steps 20 \
  --lr_scheduler plateau \
  --plateau_factor 0.5 \
  --plateau_patience 1 \
  --min_lr_scale 0.1 \
  --pocket_warmup_steps 0 \
  --ligand_gate_warmup_steps 0 \
  --max_epochs 1 \
  --grad_clip 1.0 \
  --patience 1 \
  --w_fape 1.0 \
  --w_chi 1.0 \
  --w_clash 0.1 \
  --lambda_contact 0.0 \
  --lambda_geometry_chi1 0.2 \
  --lambda_base_prior 0.0 \
  --lambda_typed_candidate_energy 1.0 \
  --typed_candidate_decoy_kind "$TYPED_DECOY_KIND" \
  --typed_candidate_margin 0.05 \
  --typed_candidate_noharm_weight 0.1 \
  --typed_candidate_noncontact_zero_weight 0.05 \
  --g_noncontact_threshold 8.0 \
  --geometry_scorer_lr_scale 4.0 \
  --geometry_scorer_use_sgeo \
  --geometry_scorer_sgeo_dim 32 \
  --geometry_scorer_use_typed_energy \
  --geometry_scorer_typed_pair_dim 64 \
  --geometry_scorer_typed_cutoff 6.0 \
  --geometry_scorer_typed_init_scale 0.1 \
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
  --residual_beta_warmup_steps 100 \
  --residual_beta_min 0.1 \
  --freeze_stage1_backbone_for_posteriors \
  --selection_metric ligand_decoy_lift_contact_switch_rotamer_acc \
  --resume_from "$RESUME_FROM" \
  --ddp_find_unused_parameters \
  --save_dir "checkpoints/stage1/${TAG}" \
  --log_dir "logs/stage1/${TAG}"

echo ""
echo "============================================================"
echo "Typed candidate energy smoke completed: $(date)"
echo "Checkpoint: checkpoints/stage1/$TAG"
echo "Metrics:    logs/stage1/$TAG/metrics.jsonl"
echo "============================================================"
