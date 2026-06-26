#!/bin/bash
# Stage-2 ablation with OracleMotion-UB feature caches.
#
# Required:
#   STAGE1V2_TRAIN_CACHE_DIR=logs/stage2_oracle_motion/<train_export>
#   STAGE1V2_VAL_CACHE_DIR=logs/stage2_oracle_motion/<val_export>
#
# Typical controls:
#   sbatch --export=ALL,STAGE1V2_MODE=zero,STAGE1V2_TRAIN_CACHE_DIR=...,STAGE1V2_VAL_CACHE_DIR=... scripts/slurm/train_stage2_oracle_motion_ablation_4gpu.sh
#   sbatch --export=ALL,STAGE1V2_MODE=oracle_motion,STAGE1V2_TRAIN_CACHE_DIR=...,STAGE1V2_VAL_CACHE_DIR=... scripts/slurm/train_stage2_oracle_motion_ablation_4gpu.sh
#   sbatch --export=ALL,STAGE1V2_MODE=oracle_motion_residue_shuffled,STAGE1V2_TRAIN_CACHE_DIR=...,STAGE1V2_VAL_CACHE_DIR=... scripts/slurm/train_stage2_oracle_motion_ablation_4gpu.sh

#SBATCH --job-name=s2_omotion_ablate
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:A100:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=200G
#SBATCH --time=18:00:00
#SBATCH --output=logs/slurm/stage2_oracle_motion_ablation_%j.out
#SBATCH --error=logs/slurm/stage2_oracle_motion_ablation_%j.err

set -euo pipefail
unset LD_PRELOAD
unset PROXYCHAINS_CONF_FILE
unset PROXYCHAINS_QUIET_MODE

ROOT=/mnt/inaisfs/data/home/zhaozc_criait/XinxiangWang/BINDRAE
ENV_PREFIX=/mnt/inaisfs/data/home/zhaozc_criait/miniconda3/envs/BINDRAE

export PATH=/data/soft/slurm/24.11.4/bin:$ENV_PREFIX/bin:$PATH
export PYTHONUNBUFFERED=1
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"

cd "$ROOT"
mkdir -p logs/slurm logs/stage2 checkpoints/stage2 processed_data/triplets/ablation_subsets

source /mnt/inaisfs/data/home/zhaozc_criait/miniconda3/etc/profile.d/conda.sh
conda activate BINDRAE

NPROC_PER_NODE="${NPROC_PER_NODE:-${SLURM_GPUS_ON_NODE:-2}}"
STAGE1V2_MODE="${STAGE1V2_MODE:-oracle_motion}"
STAGE1V2_FEATURES="${STAGE1V2_FEATURES:-delta_trans_local_x_norm,delta_trans_local_y_norm,delta_trans_local_z_norm,delta_rot_log_x_norm,delta_rot_log_y_norm,delta_rot_log_z_norm,delta_chi1_sin,delta_chi2_sin,delta_chi3_sin,delta_chi4_sin,delta_chi1_cos,delta_chi2_cos,delta_chi3_cos,delta_chi4_cos,chi1_mask,chi2_mask,chi3_mask,chi4_mask,trans_mag_norm,rot_angle_norm,max_abs_delta_chi_norm,motion_active,contact_apo,contact_holo,formed_contact,released_contact,signed_delta_dist_norm,w_res}"
STAGE1V2_FEATURE_SCALE="${STAGE1V2_FEATURE_SCALE:-1.0}"
STAGE1V2_TRAIN_CACHE_DIR="${STAGE1V2_TRAIN_CACHE_DIR:-}"
STAGE1V2_VAL_CACHE_DIR="${STAGE1V2_VAL_CACHE_DIR:-logs/stage2_oracle_motion/oracle_motion_val512_direct_20260623_020114}"
TRAIN_N="${TRAIN_N:-512}"
VAL_N="${VAL_N:-512}"
SUBSET_SEED="${SUBSET_SEED:-20260623}"
MAX_EPOCHS="${MAX_EPOCHS:-3}"
BATCH_SIZE="${BATCH_SIZE:-2}"
NUM_WORKERS="${NUM_WORKERS:-2}"
LR="${LR:-2e-5}"
VAL_T="${VAL_T:-0.5}"
SEED="${SEED:-42}"
AMP_DTYPE="${AMP_DTYPE:-bf16}"
ESM_FUSION_ENABLED="${ESM_FUSION_ENABLED:-0}"
ESM_NUM_LAYERS="${ESM_NUM_LAYERS:-1}"
ESM_FUSION_MODE="${ESM_FUSION_MODE:-sum}"
ESM_LAYER_DROPOUT="${ESM_LAYER_DROPOUT:-0.0}"
REPA_ENABLED="${REPA_ENABLED:-0}"
REPA_WEIGHT="${REPA_WEIGHT:-0.0}"
REPA_DIM="${REPA_DIM:-128}"
REPA_LOSS_TYPE="${REPA_LOSS_TYPE:-cosine}"
REPA_MASK_MODE="${REPA_MASK_MODE:-motion_active_or_pocket}"
REPA_TARGET_SHUFFLE_MODE="${REPA_TARGET_SHUFFLE_MODE:-none}"
GEOM_EVERY="${GEOM_EVERY:-1}"
N_INTEGRATION_STEPS="${N_INTEGRATION_STEPS:-3}"
INTEGRATION_CHI_CLIP="${INTEGRATION_CHI_CLIP:-1.0}"
INTEGRATION_ROT_CLIP="${INTEGRATION_ROT_CLIP:-0.1}"
INTEGRATION_TRANS_CLIP="${INTEGRATION_TRANS_CLIP:-0.2}"
N_GEOM_STEPS="${N_GEOM_STEPS:-4}"
W_CONTACT="${W_CONTACT:-0.1}"
TAG_SUFFIX="${TAG_SUFFIX:-omotion_${STAGE1V2_MODE}}"

case "$STAGE1V2_MODE" in
  zero|oracle_motion|oracle_motion_residue_shuffled|oracle_motion_sample_shuffled) ;;
  *)
    echo "ERROR: STAGE1V2_MODE must be one of zero, oracle_motion, oracle_motion_residue_shuffled, oracle_motion_sample_shuffled"
    exit 1
    ;;
esac

case "$ESM_FUSION_ENABLED" in
  0|1) ;;
  *)
    echo "ERROR: ESM_FUSION_ENABLED must be 0 or 1"
    exit 1
    ;;
esac
if [[ "$ESM_FUSION_ENABLED" == "1" && "$ESM_NUM_LAYERS" -lt 1 ]]; then
  echo "ERROR: ESM_NUM_LAYERS must be >= 1"
  exit 1
fi
case "$ESM_FUSION_MODE" in
  sum|mean|softmax_weighted) ;;
  *)
    echo "ERROR: ESM_FUSION_MODE must be one of sum, mean, softmax_weighted"
    exit 1
    ;;
esac
case "$REPA_ENABLED" in
  0|1) ;;
  *)
    echo "ERROR: REPA_ENABLED must be 0 or 1"
    exit 1
    ;;
esac
case "$REPA_LOSS_TYPE" in
  cosine|mse) ;;
  *)
    echo "ERROR: REPA_LOSS_TYPE must be one of cosine, mse"
    exit 1
    ;;
esac
case "$REPA_MASK_MODE" in
  node|pocket|motion_active|motion_active_or_pocket) ;;
  *)
    echo "ERROR: REPA_MASK_MODE must be one of node, pocket, motion_active, motion_active_or_pocket"
    exit 1
    ;;
esac
case "$REPA_TARGET_SHUFFLE_MODE" in
  none|residue) ;;
  *)
    echo "ERROR: REPA_TARGET_SHUFFLE_MODE must be one of none, residue"
    exit 1
    ;;
esac

if [[ -z "$STAGE1V2_TRAIN_CACHE_DIR" ]]; then
  echo "ERROR: STAGE1V2_TRAIN_CACHE_DIR is required"
  exit 1
fi
if [[ -z "$STAGE1V2_VAL_CACHE_DIR" ]]; then
  echo "ERROR: STAGE1V2_VAL_CACHE_DIR is required"
  exit 1
fi
FEATURE_DIM="$(python - <<PY
print(len("$STAGE1V2_FEATURES".split(",")))
PY
)"

TRAIN_SUBSET_REL="ablation_subsets/stage2_oracle_motion_train_${TRAIN_N}_seed${SUBSET_SEED}.txt"
VAL_SUBSET_REL="ablation_subsets/stage2_oracle_motion_val_${VAL_N}_seed${SUBSET_SEED}.txt"
TRAIN_SUBSET="processed_data/triplets/${TRAIN_SUBSET_REL}"
VAL_SUBSET="processed_data/triplets/${VAL_SUBSET_REL}"

python - <<PY
import json
import random
from pathlib import Path

def write_from_manifest(manifest_path, out_path, n, seed):
    manifest_path = Path(manifest_path)
    out_path = Path(out_path)
    with manifest_path.open() as f:
        manifest = json.load(f)
    ids = [r["sample_id"] for r in manifest["records"]]
    if int(n) > len(ids):
        raise SystemExit(f"requested {n} IDs from {manifest_path}, only {len(ids)} available")
    rng = random.Random(int(seed))
    rng.shuffle(ids)
    chosen = ids[:int(n)]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\\n".join(chosen) + "\\n")
    print(f"Wrote {len(chosen)} IDs to {out_path} from {manifest_path}")

write_from_manifest("$STAGE1V2_TRAIN_CACHE_DIR/manifest.json", "$TRAIN_SUBSET", "$TRAIN_N", "$SUBSET_SEED")
write_from_manifest("$STAGE1V2_VAL_CACHE_DIR/manifest.json", "$VAL_SUBSET", "$VAL_N", int("$SUBSET_SEED") + 1)
PY

TAG="stage2_${TAG_SUFFIX}_train${TRAIN_N}_val${VAL_N}_e${MAX_EPOCHS}_bs${BATCH_SIZE}x${NPROC_PER_NODE}_$(date +%Y%m%d_%H%M%S)"
SAVE_DIR="checkpoints/stage2/${TAG}"
LOG_DIR="logs/stage2/${TAG}"
ESM_ARGS=()
if [[ "$ESM_FUSION_ENABLED" == "1" ]]; then
  ESM_ARGS=(
    --esm_fusion_enabled
    --esm_num_layers "$ESM_NUM_LAYERS"
    --esm_fusion_mode "$ESM_FUSION_MODE"
    --esm_layer_dropout "$ESM_LAYER_DROPOUT"
  )
fi
REPA_ARGS=()
if [[ "$REPA_ENABLED" == "1" ]]; then
  REPA_ARGS=(
    --repa_enabled
    --repa_weight "$REPA_WEIGHT"
    --repa_dim "$REPA_DIM"
    --repa_loss_type "$REPA_LOSS_TYPE"
    --repa_mask_mode "$REPA_MASK_MODE"
    --repa_target_shuffle_mode "$REPA_TARGET_SHUFFLE_MODE"
  )
fi

echo "=============================================="
echo "Stage-2 + OracleMotion-UB ablation"
echo "=============================================="
echo "Job ID:          ${SLURM_JOB_ID:-NA}"
echo "Node:            ${SLURM_NODELIST:-NA}"
echo "GPUs:            $NPROC_PER_NODE"
echo "Mode:            $STAGE1V2_MODE"
echo "Feature dim:     $FEATURE_DIM"
echo "Feature scale:   $STAGE1V2_FEATURE_SCALE"
echo "Train cache:     $STAGE1V2_TRAIN_CACHE_DIR"
echo "Val cache:       $STAGE1V2_VAL_CACHE_DIR"
echo "Train subset:    $TRAIN_SUBSET"
echo "Val subset:      $VAL_SUBSET"
echo "max_epochs:      $MAX_EPOCHS"
echo "batch/GPU:       $BATCH_SIZE"
echo "lr:              $LR"
echo "w_contact:       $W_CONTACT"
echo "ESM fusion:      $ESM_FUSION_ENABLED"
echo "ESM layers/mode: $ESM_NUM_LAYERS / $ESM_FUSION_MODE"
echo "ESM layer drop:  $ESM_LAYER_DROPOUT"
echo "REPA enabled:    $REPA_ENABLED"
echo "REPA weight:     $REPA_WEIGHT"
echo "REPA dim/loss:   $REPA_DIM / $REPA_LOSS_TYPE"
echo "REPA mask:       $REPA_MASK_MODE"
echo "REPA shuffle:    $REPA_TARGET_SHUFFLE_MODE"
echo "geom_every:      $GEOM_EVERY"
echo "integration:     $N_INTEGRATION_STEPS"
echo "chi clip:        $INTEGRATION_CHI_CLIP"
echo "rot clip:        $INTEGRATION_ROT_CLIP"
echo "trans clip:      $INTEGRATION_TRANS_CLIP"
echo "geom steps:      $N_GEOM_STEPS"
echo "Save dir:        $SAVE_DIR"
echo "Log dir:         $LOG_DIR"
echo "Start:           $(date)"
echo "=============================================="

nvidia-smi --query-gpu=index,name,memory.total,memory.free --format=csv,noheader
python -c "import torch; print(f'PyTorch {torch.__version__}, CUDA {torch.version.cuda}, GPUs: {torch.cuda.device_count()}')"

"$ENV_PREFIX/bin/torchrun" --standalone --nproc_per_node="$NPROC_PER_NODE" scripts/train_stage2.py \
  --data_dir processed_data/triplets \
  --batch_size "$BATCH_SIZE" \
  --num_workers "$NUM_WORKERS" \
  --max_epochs "$MAX_EPOCHS" \
  --lr "$LR" \
  --grad_clip 0.3 \
  --accum_steps 1 \
  --warmup_steps 0 \
  --seed "$SEED" \
  --val_t "$VAL_T" \
  --no_stage1_prior \
  --w_prior 0.0 \
  --interaction_prior_feature_mode none \
  --w_interaction_prior 0.0 \
  --stage1v2_posterior_feature_mode "$STAGE1V2_MODE" \
  --stage1v2_train_cache_dir "$STAGE1V2_TRAIN_CACHE_DIR" \
  --stage1v2_val_cache_dir "$STAGE1V2_VAL_CACHE_DIR" \
  --stage1v2_posterior_feature_names "$STAGE1V2_FEATURES" \
  --stage1v2_posterior_feature_scale "$STAGE1V2_FEATURE_SCALE" \
  --stage1v2_loss_weight_mode none \
  --stage1v2_loss_weight_alpha 0.0 \
  --w_stage1v2_guidance 0.0 \
  --contact_loss_mode holo_target \
  --w_contact "$W_CONTACT" \
  --n_integration_steps "$N_INTEGRATION_STEPS" \
  --integration_chi_clip "$INTEGRATION_CHI_CLIP" \
  --integration_rot_clip "$INTEGRATION_ROT_CLIP" \
  --integration_trans_clip "$INTEGRATION_TRANS_CLIP" \
  --n_geom_steps "$N_GEOM_STEPS" \
  --geom_loss_every_n_steps "$GEOM_EVERY" \
  --valid_samples_file "$TRAIN_SUBSET_REL" \
  --val_samples_file "$VAL_SUBSET_REL" \
  "${ESM_ARGS[@]}" \
  "${REPA_ARGS[@]}" \
  --save_dir "$SAVE_DIR" \
  --log_dir "$LOG_DIR" \
  --device cuda \
  --amp_dtype "$AMP_DTYPE" \
  --distributed

echo ""
echo "=============================================="
echo "Stage-2 + OracleMotion-UB ablation completed: $(date)"
echo "Metrics: $LOG_DIR/metrics.jsonl"
echo "=============================================="
