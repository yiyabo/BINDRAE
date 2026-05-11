#!/bin/bash
#SBATCH --job-name=s1_v2_smoke
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:A100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=00:30:00
#SBATCH --output=logs/slurm/stage1_v2_smoke_%j.out
#SBATCH --error=logs/slurm/stage1_v2_smoke_%j.err

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
mkdir -p logs/slurm logs/stage1 checkpoints/stage1

source /mnt/inaisfs/data/home/zhaozc_criait/miniconda3/etc/profile.d/conda.sh
conda activate BINDRAE

TAG="v2_smoke_$(date +%Y%m%d_%H%M%S)"
TRAIN_TXT="$ROOT/processed_data/triplets/train_valid.txt"
VAL_TXT="$ROOT/processed_data/triplets/val_valid.txt"
RESUME_FROM="$ROOT/checkpoints/stage1/geom_baseprior_20260501_170858/best_model.pt"

echo "=============================================="
echo "Stage-1 Phase-1 V2 SMOKE TEST (1-GPU)"
echo "Goal: verify v2 forward+backward+decoy works"
echo "Tag:        $TAG"
echo "Start:      $(date)"
echo "=============================================="

nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader

python scripts/train_stage1.py \
  --model_size medium \
  --data_dir processed_data/triplets \
  --valid_samples_file "$TRAIN_TXT" \
  --val_samples_file "$VAL_TXT" \
  --sample_metadata_file sample_metadata.json \
  --batch_size 1 \
  --num_workers 1 \
  --max_n_res 300 \
  --lr 3e-4 \
  --warmup_steps 50 \
  --lr_scheduler plateau \
  --plateau_factor 0.5 \
  --plateau_patience 2 \
  --min_lr_scale 0.1 \
  --pocket_warmup_steps 0 \
  --ligand_gate_warmup_steps 0 \
  --max_epochs 41 \
  --grad_clip 1.0 \
  --patience 5 \
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
  --residual_beta_warmup_steps 200 \
  --residual_beta_min 0.1 \
  --freeze_stage1_backbone_for_posteriors \
  --selection_metric chi1_rotamer_acc \
  --resume_from "$RESUME_FROM" \
  --save_dir "checkpoints/stage1/${TAG}" \
  --log_dir "logs/stage1/${TAG}"

echo ""
echo "=============================================="
echo "v2 Smoke completed: $(date)"
echo "If you see 'g_switch_dir/amp/rank/decoy' in losses, v2 works."
echo "=============================================="
