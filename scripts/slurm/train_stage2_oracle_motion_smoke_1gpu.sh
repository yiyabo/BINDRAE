#!/bin/bash
# Minimal Stage-2 smoke using privileged OracleMotion-UB feature caches.
#
# Requires both train and val OracleMotion cache dirs. Example:
#   sbatch --export=ALL,STAGE1V2_TRAIN_CACHE_DIR=logs/stage2_oracle_motion/oracle_motion_train512_direct_YYYYMMDD_HHMMSS,STAGE1V2_VAL_CACHE_DIR=logs/stage2_oracle_motion/oracle_motion_val512_direct_20260623_020114 scripts/slurm/train_stage2_oracle_motion_smoke_1gpu.sh
# Controls:
#   sbatch --export=ALL,STAGE1V2_MODE=zero scripts/slurm/train_stage2_oracle_motion_smoke_1gpu.sh
#   sbatch --export=ALL,STAGE1V2_MODE=oracle_motion_residue_shuffled scripts/slurm/train_stage2_oracle_motion_smoke_1gpu.sh
#   sbatch --export=ALL,STAGE1V2_MODE=oracle_motion_sample_shuffled scripts/slurm/train_stage2_oracle_motion_smoke_1gpu.sh

#SBATCH --job-name=s2_omotion_smoke
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:A100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
#SBATCH --time=04:00:00
#SBATCH --output=logs/slurm/stage2_oracle_motion_smoke_%j.out
#SBATCH --error=logs/slurm/stage2_oracle_motion_smoke_%j.err

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

STAGE1V2_MODE="${STAGE1V2_MODE:-oracle_motion}"
STAGE1V2_FEATURES="${STAGE1V2_FEATURES:-delta_trans_local_x_norm,delta_trans_local_y_norm,delta_trans_local_z_norm,delta_rot_log_x_norm,delta_rot_log_y_norm,delta_rot_log_z_norm,delta_chi1_sin,delta_chi2_sin,delta_chi3_sin,delta_chi4_sin,delta_chi1_cos,delta_chi2_cos,delta_chi3_cos,delta_chi4_cos,chi1_mask,chi2_mask,chi3_mask,chi4_mask,trans_mag_norm,rot_angle_norm,max_abs_delta_chi_norm,motion_active,contact_apo,contact_holo,formed_contact,released_contact,signed_delta_dist_norm,w_res}"
STAGE1V2_FEATURE_SCALE="${STAGE1V2_FEATURE_SCALE:-1.0}"
STAGE1V2_TRAIN_CACHE_DIR="${STAGE1V2_TRAIN_CACHE_DIR:-}"
STAGE1V2_VAL_CACHE_DIR="${STAGE1V2_VAL_CACHE_DIR:-logs/stage2_oracle_motion/oracle_motion_val512_direct_20260623_020114}"
TRAIN_N="${TRAIN_N:-8}"
VAL_N="${VAL_N:-4}"
MAX_EPOCHS="${MAX_EPOCHS:-1}"
BATCH_SIZE="${BATCH_SIZE:-1}"
NUM_WORKERS="${NUM_WORKERS:-0}"
LR="${LR:-2e-5}"
VAL_T="${VAL_T:-0.5}"
SEED="${SEED:-42}"
AMP_DTYPE="${AMP_DTYPE:-bf16}"
GEOM_EVERY="${GEOM_EVERY:-1}"
N_INTEGRATION_STEPS="${N_INTEGRATION_STEPS:-2}"
N_GEOM_STEPS="${N_GEOM_STEPS:-2}"
W_CONTACT="${W_CONTACT:-0.1}"
TAG_SUFFIX="${TAG_SUFFIX:-omotion_${STAGE1V2_MODE}}"

case "$STAGE1V2_MODE" in
  zero|oracle_motion|oracle_motion_residue_shuffled|oracle_motion_sample_shuffled) ;;
  *)
    echo "ERROR: STAGE1V2_MODE must be one of zero, oracle_motion, oracle_motion_residue_shuffled, oracle_motion_sample_shuffled"
    exit 1
    ;;
esac

if [[ -z "$STAGE1V2_TRAIN_CACHE_DIR" ]]; then
  echo "ERROR: STAGE1V2_TRAIN_CACHE_DIR is required to define the train subset"
  exit 1
fi
if [[ -z "$STAGE1V2_VAL_CACHE_DIR" ]]; then
  echo "ERROR: STAGE1V2_VAL_CACHE_DIR is required to define the val subset"
  exit 1
fi
FEATURE_DIM="$(python - <<PY
print(len("$STAGE1V2_FEATURES".split(",")))
PY
)"

TRAIN_SUBSET_REL="ablation_subsets/stage2_oracle_motion_smoke_train_${TRAIN_N}.txt"
VAL_SUBSET_REL="ablation_subsets/stage2_oracle_motion_smoke_val_${VAL_N}.txt"
TRAIN_SUBSET="processed_data/triplets/${TRAIN_SUBSET_REL}"
VAL_SUBSET="processed_data/triplets/${VAL_SUBSET_REL}"

python - <<PY
import json
from pathlib import Path

def write_from_manifest(manifest_path, out_path, n):
    manifest_path = Path(manifest_path)
    out_path = Path(out_path)
    with manifest_path.open() as f:
        manifest = json.load(f)
    ids = [r["sample_id"] for r in manifest["records"][:int(n)]]
    if int(n) > len(ids):
        raise SystemExit(f"requested {n} IDs from {manifest_path}, only {len(ids)} available")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\\n".join(ids) + "\\n")
    print(f"Wrote {len(ids)} IDs to {out_path} from {manifest_path}")

write_from_manifest("$STAGE1V2_TRAIN_CACHE_DIR/manifest.json", "$TRAIN_SUBSET", "$TRAIN_N")
write_from_manifest("$STAGE1V2_VAL_CACHE_DIR/manifest.json", "$VAL_SUBSET", "$VAL_N")
PY

TAG="stage2_${TAG_SUFFIX}_train${TRAIN_N}_val${VAL_N}_e${MAX_EPOCHS}_$(date +%Y%m%d_%H%M%S)"
SAVE_DIR="checkpoints/stage2/${TAG}"
LOG_DIR="logs/stage2/${TAG}"

echo "=============================================="
echo "Stage-2 + OracleMotion-UB smoke"
echo "=============================================="
echo "Job ID:        ${SLURM_JOB_ID:-NA}"
echo "Node:          ${SLURM_NODELIST:-NA}"
echo "Mode:          $STAGE1V2_MODE"
echo "Feature dim:   $FEATURE_DIM"
echo "Feature scale: $STAGE1V2_FEATURE_SCALE"
echo "Train cache:   ${STAGE1V2_TRAIN_CACHE_DIR:-OFF}"
echo "Val cache:     ${STAGE1V2_VAL_CACHE_DIR:-OFF}"
echo "Train subset:  $TRAIN_SUBSET"
echo "Val subset:    $VAL_SUBSET"
echo "Save dir:      $SAVE_DIR"
echo "Log dir:       $LOG_DIR"
echo "Start:         $(date)"
echo "=============================================="

nvidia-smi --query-gpu=index,name,memory.total,memory.free --format=csv,noheader
python -c "import torch; print(f'PyTorch {torch.__version__}, CUDA {torch.version.cuda}, GPUs: {torch.cuda.device_count()}')"

python scripts/train_stage2.py \
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
  --n_geom_steps "$N_GEOM_STEPS" \
  --geom_loss_every_n_steps "$GEOM_EVERY" \
  --valid_samples_file "$TRAIN_SUBSET_REL" \
  --val_samples_file "$VAL_SUBSET_REL" \
  --save_dir "$SAVE_DIR" \
  --log_dir "$LOG_DIR" \
  --device cuda \
  --amp_dtype "$AMP_DTYPE"

echo ""
echo "=============================================="
echo "Stage-2 + OracleMotion-UB smoke completed: $(date)"
echo "Metrics: $LOG_DIR/metrics.jsonl"
echo "=============================================="
