#!/bin/bash
#SBATCH --job-name=s1_p1_lift8_screen
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --gres=gpu:A100:8
#SBATCH --cpus-per-task=4
#SBATCH --mem=320G
#SBATCH --time=12:00:00
#SBATCH --output=logs/slurm/stage1_phase1_ligcausal_screen8_%j.out
#SBATCH --error=logs/slurm/stage1_phase1_ligcausal_screen8_%j.err

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

TAG="${TAG:-phase1_residual_ligcausal_screen8_$(date +%Y%m%d_%H%M%S)}"
TRAIN_TXT="${TRAIN_TXT:-$ROOT/processed_data/triplets/train_valid.txt}"
VAL_TXT="${VAL_TXT:-$ROOT/processed_data/triplets/val_valid.txt}"
LR="${LR:-1e-5}"
MAX_EPOCHS="${MAX_EPOCHS:-8}"
BETA_WARMUP="${BETA_WARMUP:-6000}"
BETA_MIN="${BETA_MIN:-0.05}"
SELECTION_METRIC="${SELECTION_METRIC:-ligand_lift_contact_switch_rotamer_acc}"
MASTER_PORT="${MASTER_PORT:-29518}"

cat <<EOF
============================================================
Stage-1 Phase-1 residual ligand-causal SCREENING run (8-GPU)
============================================================
Job ID:           ${SLURM_JOB_ID:-NA}
Node:             ${SLURM_NODELIST:-NA}
GPUs:             8 x A100 on one node
Tag:              $TAG
Train samples:    $TRAIN_TXT
Val samples:      $VAL_TXT
LR:               $LR
Max epochs:       $MAX_EPOCHS
Selection metric: $SELECTION_METRIC

Purpose:
  Short-budget screen for ligand-causal posterior learning. Keep current
  16-GPU stable reference running; this branch tests whether stronger
  contact/switch G-vector and decoy losses produce sustained ligand lift.

Main changes versus stable 16-GPU reference:
  residual beta warmup:   16000 -> $BETA_WARMUP, beta_min=$BETA_MIN
  gate open warmup:       2000 -> 3000
  residual max/clamp:     max=3.0 -> 4.0, gate clamp=5.0, init bias=0.5
  geometry scorer LR:     1.0x -> 1.5x
  G lift/noharm:          0/0 -> 0.5/0.2
  switch dir/amp/rank:    1/0.3/0.3 -> 2.0/0.7/0.7
  g_decoy:                0.5 -> 1.5
  g_zero_noncontact:      0.2 -> 0.3

Pre-registered kill criterion:
  If epoch 3-4 validation keeps contact+switch ligand lift <= 0 and decoy
  lift remains 0, kill instead of extending this screening run.
Start: $(date)
============================================================
EOF

nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader

"$TORCHRUN_BIN" \
  --nproc_per_node=8 \
  --master_port="$MASTER_PORT" \
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
  --plateau_patience 3 \
  --min_lr_scale 0.1 \
  --pocket_warmup_steps 2000 \
  --ligand_gate_warmup_steps 2000 \
  --max_epochs "$MAX_EPOCHS" \
  --grad_clip 0.5 \
  --patience 8 \
  --w_fape 1.0 \
  --w_chi 1.0 \
  --w_clash 0.1 \
  --lambda_chi1_rotamer 0.0 \
  --lambda_contact 0.0 \
  --lambda_candidate_chi1 0.0 \
  --lambda_ligand_contrastive 0.0 \
  --lambda_geometry_chi1 0.35 \
  --lambda_base_prior 0.0 \
  --lambda_switch_bce 0.0 \
  --lambda_rescue_noharm 0.0 \
  --lambda_ligand_residual 0.0 \
  --lambda_g_lift_switch 0.5 \
  --lambda_g_noharm 0.2 \
  --lambda_g_zero_noncontact 0.3 \
  --g_lift_margin 0.3 \
  --g_noharm_margin 0.3 \
  --g_noncontact_threshold 8.0 \
  --lambda_g_switch_dir 2.0 \
  --lambda_g_switch_amp 0.7 \
  --lambda_g_switch_rank 0.7 \
  --lambda_g_antiharm 0.5 \
  --lambda_g_decoy 1.5 \
  --g_switch_temperature 1.0 \
  --g_switch_amp_margin 0.05 \
  --g_switch_rank_margin 0.05 \
  --g_antiharm_tau 0.05 \
  --g_decoy_margin 0.05 \
  --g_decoy_kind shuffled \
  --geometry_scorer_lr_scale 1.5 \
  --geometry_scorer_use_sgeo \
  --geometry_scorer_sgeo_dim 32 \
  --geometry_scorer_bounded_residual \
  --geometry_scorer_residual_max 4.0 \
  --geometry_scorer_residual_tau 3.0 \
  --geometry_scorer_gate_norm \
  --geometry_scorer_gate_clamp 5.0 \
  --geometry_scorer_gate_init_bias 0.5 \
  --detach_base_for_residual \
  --gate_warmup_open_steps 3000 \
  --residual_beta_warmup_steps "$BETA_WARMUP" \
  --residual_beta_min "$BETA_MIN" \
  --selection_metric "$SELECTION_METRIC" \
  --ddp_find_unused_parameters \
  --save_epoch_checkpoints \
  --save_dir "checkpoints/stage1/${TAG}" \
  --log_dir "logs/stage1/${TAG}"

echo ""
echo "============================================================"
echo "8-GPU ligand-causal screening completed: $(date)"
echo "Checkpoint: checkpoints/stage1/$TAG"
echo "Metrics:    logs/stage1/$TAG/metrics.jsonl"
echo "============================================================"
