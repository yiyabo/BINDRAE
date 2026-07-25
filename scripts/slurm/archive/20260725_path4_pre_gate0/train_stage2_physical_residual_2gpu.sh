#!/bin/bash
# Amortize the canonical physical-normal Path-4 teacher into residual heads.

#SBATCH --job-name=s2_phys_student
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:A100:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=200G
#SBATCH --time=12:00:00
#SBATCH --output=logs/slurm/stage2_physical_student_%j.out
#SBATCH --error=logs/slurm/stage2_physical_student_%j.err

set -euo pipefail

ROOT="${ROOT:-/mnt/inaisfs/data/home/zhaozc_criait/XinxiangWang/BINDRAE}"
ANCHOR_CHECKPOINT="${ANCHOR_CHECKPOINT:-checkpoints/stage2/stage2_pbv2_phase_cons303_fam30scaf_e10_bs8x2_20260718_canonical_fromscratch_v2/best_model.pt}"
PHYSICAL_NORMAL_CACHE_DIR="${PHYSICAL_NORMAL_CACHE_DIR:-logs/stage2/physical_normal_targets_cons303_rw20_20260719_v1}"
ORACLE_CACHE_DIR="${ORACLE_CACHE_DIR:-logs/stage2_oracle_motion/oracle_motion_mdphase_consensus303_canonical_v2e_20260718_v1/merged}"
ORACLE_TRAIN_CACHE_DIR="${ORACLE_TRAIN_CACHE_DIR:-$ORACLE_CACHE_DIR}"
ORACLE_VAL_CACHE_DIR="${ORACLE_VAL_CACHE_DIR:-$ORACLE_CACHE_DIR}"

export ROOT
export NPROC_PER_NODE="${NPROC_PER_NODE:-2}"
export STAGE1V2_MODE=oracle_motion
export STAGE1V2_TRAIN_CACHE_DIR="$ORACLE_TRAIN_CACHE_DIR"
export STAGE1V2_VAL_CACHE_DIR="$ORACLE_VAL_CACHE_DIR"
export TRAIN_N="${TRAIN_N:-241}"
export VAL_N="${VAL_N:-30}"
export SUBSET_SEED="${SUBSET_SEED:-20260717}"
export SUBSET_TAG="${SUBSET_TAG:-mdphase_consensus303_fam30scaf_blockv2}"
export USE_EXISTING_SUBSETS="${USE_EXISTING_SUBSETS:-1}"
export VALIDATE_EXISTING_SUBSETS="${VALIDATE_EXISTING_SUBSETS:-0}"
export TRUST_PRECHECKED_SAMPLES="${TRUST_PRECHECKED_SAMPLES:-1}"
export VAL_SPLIT="${VAL_SPLIT:-train}"

export PATH_PARAMETERIZATION=phase_block_orthogonal_residual_v2
export PHASE_RESIDUAL_TAU_MODE=learned
export PHASE_RESIDUAL_BRIDGE_MODE=cartesian_backbone
export PHASE_RESIDUAL_ACTIVE_BLOCKS=translation
export PHASE_RESIDUAL_DECODER_MODE=independent
export PHASE_RESIDUAL_ENVELOPE=poly
export PHASE_RESIDUAL_SCALE=1.0
export PHASE_RESIDUAL_MAX_METRIC_NORM=1.0
export PHASE_RESIDUAL_TRANSLATION_GATE_BIAS="${PHASE_RESIDUAL_TRANSLATION_GATE_BIAS:--2.0}"
export PHASE_RESIDUAL_HEADS_ONLY="${PHASE_RESIDUAL_HEADS_ONLY:-1}"
export PHASE_NORMAL_CACHE_DIR="$PHYSICAL_NORMAL_CACHE_DIR"
export W_PHASE_NORMAL_RESIDUAL=1.0
export PHASE_NORMAL_RESIDUAL_LOSS_TYPE=mse
export PHASE_NORMAL_RESIDUAL_MIN_CONFIDENCE=0.0
export PHASE_NORMAL_RESIDUAL_WEIGHT_MODE="${PHASE_NORMAL_RESIDUAL_WEIGHT_MODE:-uniform}"
export PHASE_NORMAL_RESIDUAL_MAGNITUDE_SCALE="${PHASE_NORMAL_RESIDUAL_MAGNITUDE_SCALE:-0.05}"
export PHASE_NORMAL_RESIDUAL_MAGNITUDE_BOOST="${PHASE_NORMAL_RESIDUAL_MAGNITUDE_BOOST:-0.0}"
export PHASE_NORMAL_RESIDUAL_RIGID_WEIGHT=1.0
export PHASE_NORMAL_RESIDUAL_CHI_WEIGHT=0.0
export PHASE_NORMAL_MISSING_POLICY=error
export INIT_FROM_CHECKPOINT="$ANCHOR_CHECKPOINT"
export INIT_FROM_CHECKPOINT_MODE=phase_warp

export ESM_FUSION_ENABLED=1
export ESM_NUM_LAYERS=7
export ESM_FUSION_MODE=gated_residual
export ESM_GATE_BIAS=-2.0
export ESM_GATE_CONTEXT_MODE=pocket_motion
export REPA_ENABLED=0

export MAX_EPOCHS="${MAX_EPOCHS:-10}"
export BATCH_SIZE="${BATCH_SIZE:-8}"
export VAL_BATCH_SIZE="${VAL_BATCH_SIZE:-4}"
export NUM_WORKERS="${NUM_WORKERS:-4}"
export PREFETCH_FACTOR="${PREFETCH_FACTOR:-4}"
export LR="${LR:-5e-4}"
export WARMUP_STEPS="${WARMUP_STEPS:-0}"
export EARLY_STOP_PATIENCE="${EARLY_STOP_PATIENCE:-10}"
export N_INTEGRATION_STEPS=20
export GEOM_EVERY=1
export N_GEOM_STEPS=4
export LENGTH_BUCKETED_TRAIN=0
export AUTO_RESUME="${AUTO_RESUME:-0}"
export CHECKPOINT_EVERY_N_EPOCHS="${CHECKPOINT_EVERY_N_EPOCHS:-1}"

# Distill only the deterministic teacher field. Geometry is evaluated later on
# the frozen validation set and must not silently reshape the teacher target.
export W_FM_CHI=0.0
export W_FM_RIGID=0.0
export W_BG=0.0
export W_PHASE_RESIDUAL_MAGNITUDE=0.0
export W_PHASE_RESIDUAL_TEMPORAL_SMOOTH=0.0
export W_PHASE_RESIDUAL_NEIGHBOR_SMOOTH=0.0
export W_SMOOTH=0.0
export W_CLASH=0.0
export W_PEP=0.0
export W_CONTACT=0.0
export W_END=0.0

export TAG="${TAG:-stage2_path4_physdistill_cons303_e${MAX_EPOCHS}_bs${BATCH_SIZE}x${NPROC_PER_NODE}_$(date +%Y%m%d_%H%M%S)}"

cd "$ROOT"
exec bash scripts/slurm/train_stage2_oracle_motion_ablation_4gpu.sh
