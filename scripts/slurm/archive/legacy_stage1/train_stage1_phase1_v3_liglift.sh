#!/bin/bash
#SBATCH --job-name=s1_v3_lift
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:A100:4
#SBATCH --cpus-per-task=4
#SBATCH --mem=192G
#SBATCH --time=72:00:00
#SBATCH --output=logs/slurm/stage1_phase1_v3_liglift_%j.out
#SBATCH --error=logs/slurm/stage1_phase1_v3_liglift_%j.err

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

TAG="${TAG:-phase1_v3_liglift_$(date +%Y%m%d_%H%M%S)}"
TRAIN_TXT="${TRAIN_TXT:-$ROOT/processed_data/triplets/train_valid.txt}"
VAL_TXT="${VAL_TXT:-$ROOT/processed_data/triplets/val_valid.txt}"
RESUME_FROM="${RESUME_FROM:-$ROOT/checkpoints/stage1/geom_baseprior_20260501_170858/best_model.pt}"
LR="${LR:-3e-4}"
MAX_EPOCHS="${MAX_EPOCHS:-80}"
BETA_WARMUP="${BETA_WARMUP:-4000}"
BETA_MIN="${BETA_MIN:-0.1}"
SELECTION_METRIC="${SELECTION_METRIC:-ligand_lift_contact_switch_rotamer_acc}"

if [[ ! -f "$RESUME_FROM" ]]; then
  echo "ERROR: resume checkpoint not found: $RESUME_FROM" >&2
  exit 2
fi

cat <<EOF
==============================================
Stage-1 Phase-1 V3: Ligand-Lift Residual Training (4-GPU)
==============================================
Job ID:           ${SLURM_JOB_ID:-NA}
Tag:              $TAG
Resume:           $RESUME_FROM
LR:               $LR
Max epochs:       $MAX_EPOCHS
Selection metric: $SELECTION_METRIC

Scientific target:
  Select checkpoints by correct-ligand lift over base_prior on contact+switch residues,
  not by global rotamer accuracy. This directly tests ligand-causal residual utility.

Controls:
  reset residual+gate: YES
  freeze base_mlp:     YES
  freeze gate_mlp:     YES (gate fixed open via override)
  detach base:         YES
  beta warmup:         $BETA_WARMUP steps ($BETA_MIN -> 1.0)
  full CE/base CE:     OFF
  v2 G-vector losses:  ON
  decoy:               translated ligand
Start: $(date)
==============================================
EOF

nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader

"$TORCHRUN_BIN" \
  --nproc_per_node=4 \
  --master_port=29505 \
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
  --warmup_steps 500 \
  --lr_scheduler plateau \
  --plateau_factor 0.5 \
  --plateau_patience 5 \
  --min_lr_scale 0.1 \
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
  --lambda_geometry_chi1 0.0 \
  --lambda_base_prior 0.0 \
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
  --lambda_g_decoy 1.0 \
  --g_switch_temperature 1.0 \
  --g_switch_amp_margin 0.05 \
  --g_switch_rank_margin 0.05 \
  --g_antiharm_tau 0.05 \
  --g_decoy_margin 0.05 \
  --g_decoy_kind translated \
  --g_decoy_translation_offset 100.0 \
  --geometry_scorer_lr_scale 3.0 \
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
  --residual_beta_warmup_steps "$BETA_WARMUP" \
  --residual_beta_min "$BETA_MIN" \
  --freeze_stage1_backbone_for_posteriors \
  --selection_metric "$SELECTION_METRIC" \
  --compute_slow_metrics \
  --save_epoch_checkpoints \
  --resume_from "$RESUME_FROM" \
  --save_dir "checkpoints/stage1/${TAG}" \
  --log_dir "logs/stage1/${TAG}"

echo ""
echo "=============================================="
echo "Phase-1 v3 ligand-lift training completed: $(date)"
echo "Checkpoint: checkpoints/stage1/$TAG"
echo "=============================================="
