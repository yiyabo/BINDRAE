#!/bin/bash
#SBATCH --job-name=s1_p1_zero_lift8
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --gres=gpu:A100:8
#SBATCH --cpus-per-task=4
#SBATCH --mem=320G
#SBATCH --time=72:00:00
#SBATCH --output=logs/slurm/stage1_phase1_fullscratch_ligcausal_8gpu_%j.out
#SBATCH --error=logs/slurm/stage1_phase1_fullscratch_ligcausal_8gpu_%j.err

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

TAG="${TAG:-phase1_residual_fullscratch_ligcausal_8gpu_$(date +%Y%m%d_%H%M%S)}"
TRAIN_TXT="${TRAIN_TXT:-$ROOT/processed_data/triplets/train_valid.txt}"
VAL_TXT="${VAL_TXT:-$ROOT/processed_data/triplets/val_valid.txt}"
LR="${LR:-5e-5}"
MAX_EPOCHS="${MAX_EPOCHS:-100}"
BETA_WARMUP="${BETA_WARMUP:-8000}"
BETA_MIN="${BETA_MIN:-0.05}"
SELECTION_METRIC="${SELECTION_METRIC:-ligand_decoy_lift_contact_switch_rotamer_acc}"
MASTER_PORT="${MASTER_PORT:-29508}"

cat <<EOF
==============================================
Stage-1 Phase-1 Residual: full from-scratch ligand-causal training (8-GPU)
==============================================
Job ID:           ${SLURM_JOB_ID:-NA}
Node:             ${SLURM_NODELIST:-NA}
GPUs:             8 x A100 on one node
Tag:              $TAG
Train samples:    $TRAIN_TXT
Val samples:      $VAL_TXT
LR:               $LR
Max epochs:       $MAX_EPOCHS
Selection metric: $SELECTION_METRIC

Scientific target:
  Same objective as the 4-GPU fullscratch run, but with single-node 8-GPU DDP
  for faster queue/runtime if an 8xA100 node becomes available.

Controls:
  resume checkpoint:     NONE (true from-scratch run)
  freeze base_mlp:       NO
  freeze trunk/backbone: NO
  bounded residual:      YES (tanh, max=5.0, tau=3.0)
  gate norm/clamp:       LayerNorm + clamp(±6)
  beta warmup:           $BETA_WARMUP steps ($BETA_MIN -> 1.0)
  decoy loss kind:       shuffled ligand
  selection:             correct-vs-decoy contact+switch lift
Start: $(date)
==============================================
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
  --plateau_patience 5 \
  --min_lr_scale 0.1 \
  --pocket_warmup_steps 2000 \
  --ligand_gate_warmup_steps 2000 \
  --max_epochs "$MAX_EPOCHS" \
  --grad_clip 1.0 \
  --patience 20 \
  --w_fape 1.0 \
  --w_chi 1.0 \
  --w_clash 0.1 \
  --lambda_chi1_rotamer 0.0 \
  --lambda_contact 0.0 \
  --lambda_candidate_chi1 0.0 \
  --lambda_ligand_contrastive 0.0 \
  --lambda_geometry_chi1 0.5 \
  --lambda_base_prior 0.5 \
  --lambda_switch_bce 0.0 \
  --lambda_rescue_noharm 0.0 \
  --lambda_g_lift_switch 0.0 \
  --lambda_g_noharm 0.0 \
  --lambda_g_zero_noncontact 0.5 \
  --g_noncontact_threshold 8.0 \
  --lambda_g_switch_dir 3.0 \
  --lambda_g_switch_amp 1.0 \
  --lambda_g_switch_rank 1.0 \
  --lambda_g_antiharm 0.5 \
  --lambda_g_decoy 2.0 \
  --g_switch_temperature 1.0 \
  --g_switch_amp_margin 0.05 \
  --g_switch_rank_margin 0.05 \
  --g_antiharm_tau 0.05 \
  --g_decoy_margin 0.05 \
  --g_decoy_kind shuffled \
  --geometry_scorer_lr_scale 1.0 \
  --geometry_scorer_use_sgeo \
  --geometry_scorer_sgeo_dim 32 \
  --geometry_scorer_bounded_residual \
  --geometry_scorer_residual_max 5.0 \
  --geometry_scorer_residual_tau 3.0 \
  --geometry_scorer_gate_norm \
  --geometry_scorer_gate_clamp 6.0 \
  --geometry_scorer_gate_init_bias 1.0 \
  --detach_base_for_residual \
  --gate_warmup_open_steps 4000 \
  --residual_beta_warmup_steps "$BETA_WARMUP" \
  --residual_beta_min "$BETA_MIN" \
  --selection_metric "$SELECTION_METRIC" \
  --ddp_find_unused_parameters \
  --save_epoch_checkpoints \
  --save_dir "checkpoints/stage1/${TAG}" \
  --log_dir "logs/stage1/${TAG}"

echo ""
echo "=============================================="
echo "8-GPU full from-scratch ligand-causal Phase-1 residual training completed: $(date)"
echo "Checkpoint: checkpoints/stage1/$TAG"
echo "Metrics:    logs/stage1/$TAG/metrics.jsonl"
echo "=============================================="
