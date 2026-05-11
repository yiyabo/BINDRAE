#!/bin/bash
#SBATCH --job-name=s1_phase1
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:A100:4
#SBATCH --cpus-per-task=4
#SBATCH --mem=192G
#SBATCH --time=72:00:00
#SBATCH --output=logs/slurm/stage1_phase1_residual_%j.out
#SBATCH --error=logs/slurm/stage1_phase1_residual_%j.err

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

TAG="${TAG:-phase1_residual_$(date +%Y%m%d_%H%M%S)}"
TRAIN_TXT="${TRAIN_TXT:-$ROOT/processed_data/triplets/train_valid.txt}"
VAL_TXT="${VAL_TXT:-$ROOT/processed_data/triplets/val_valid.txt}"
# Resume from baseprior best (epoch 38, 62.45% rotamer_acc)
RESUME_FROM="${RESUME_FROM:-$ROOT/checkpoints/stage1/geom_baseprior_20260501_170858/best_model.pt}"
LR="${LR:-1e-4}"
MAX_EPOCHS="${MAX_EPOCHS:-50}"
WARMUP_OPEN="${WARMUP_OPEN:-2000}"
BETA_WARMUP="${BETA_WARMUP:-4000}"
BETA_MIN="${BETA_MIN:-0.1}"

echo "=============================================="
echo "Stage-1 Phase-1: Residual Re-training (4-GPU)"
echo "=============================================="
echo "Job ID:           $SLURM_JOB_ID"
echo "Node:             $SLURM_NODELIST"
echo "Tag:              $TAG"
echo "Resume from:      $RESUME_FROM"
echo "LR:               $LR"
echo "Max epochs:       $MAX_EPOCHS"
echo ""
echo "Phase-1 controls:"
echo "  reset residual+gate:  YES (zero-init residual final)"
echo "  freeze base_mlp:      YES (lock baseprior epoch-38 base)"
echo "  detach base in fwd:   YES"
echo "  gate warmup open:     $WARMUP_OPEN steps (gate=1.0 forced)"
echo "  beta warmup:          $BETA_WARMUP steps (beta: $BETA_MIN -> 1.0)"
echo "  bounded residual:     YES (tanh, max=5.0, tau=2.0)"
echo "  gate norm/clamp:      LayerNorm + clamp(±6)"
echo "  gate init bias:       +2.0 (sigmoid≈0.88 open)"
echo ""
echo "Loss weights:"
echo "  base_prior:          0.0 (frozen, no need to train)"
echo "  geometry_chi1 CE:    0.5 (still want full to match holo)"
echo "  g_lift_switch:       2.0 (G(holo) > G(apo) on switch)"
echo "  g_noharm:            1.0 (no flips on apo-correct)"
echo "  g_zero_noncontact:   0.2 (||G||^2 -> 0 far from ligand)"
echo "  switch_bce:          0.0 (replaced by g_lift)"
echo "  rescue_noharm:       0.0 (replaced by g_lift+g_noharm)"
echo "Start:           $(date)"
echo "=============================================="

nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader

"$TORCHRUN_BIN" \
  --nproc_per_node=4 \
  --master_port=29503 \
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
  --lambda_geometry_chi1 0.5 \
  --lambda_base_prior 0.0 \
  --lambda_switch_bce 0.0 \
  --lambda_rescue_noharm 0.0 \
  --lambda_g_lift_switch 2.0 \
  --lambda_g_noharm 1.0 \
  --lambda_g_zero_noncontact 0.2 \
  --g_lift_margin 0.5 \
  --g_noharm_margin 0.5 \
  --g_noncontact_threshold 8.0 \
  --geometry_scorer_lr_scale 1.0 \
  --geometry_scorer_use_sgeo \
  --geometry_scorer_sgeo_dim 32 \
  --geometry_scorer_bounded_residual \
  --geometry_scorer_residual_max 5.0 \
  --geometry_scorer_residual_tau 2.0 \
  --geometry_scorer_gate_norm \
  --geometry_scorer_gate_clamp 6.0 \
  --geometry_scorer_gate_init_bias 2.0 \
  --reset_residual_and_gate_on_resume \
  --freeze_base_mlp \
  --detach_base_for_residual \
  --gate_warmup_open_steps "$WARMUP_OPEN" \
  --residual_beta_warmup_steps "$BETA_WARMUP" \
  --residual_beta_min "$BETA_MIN" \
  --freeze_stage1_backbone_for_posteriors \
  --unfreeze_ligand_conditioner_for_posteriors \
  --unfreeze_last_ipa_blocks_for_posteriors 2 \
  --selection_metric chi1_rotamer_acc \
  --compute_slow_metrics \
  --save_epoch_checkpoints \
  --resume_from "$RESUME_FROM" \
  --save_dir "checkpoints/stage1/${TAG}" \
  --log_dir "logs/stage1/${TAG}"

echo ""
echo "=============================================="
echo "Phase-1 training completed: $(date)"
echo "Checkpoint: checkpoints/stage1/$TAG"
echo "=============================================="
