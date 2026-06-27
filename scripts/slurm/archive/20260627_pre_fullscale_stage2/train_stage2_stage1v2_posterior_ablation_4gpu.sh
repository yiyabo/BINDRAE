#!/bin/bash
# Stage-2 ablation with Stage-1-v2 posterior scalar features.
#
# Typical runs:
#   sbatch --export=ALL,STAGE1V2_MODE=none scripts/slurm/train_stage2_stage1v2_posterior_ablation_4gpu.sh
#   sbatch --export=ALL,STAGE1V2_MODE=zero scripts/slurm/train_stage2_stage1v2_posterior_ablation_4gpu.sh
#   sbatch --export=ALL,STAGE1V2_MODE=student scripts/slurm/train_stage2_stage1v2_posterior_ablation_4gpu.sh
#   sbatch --export=ALL,STAGE1V2_MODE=student_shuffled scripts/slurm/train_stage2_stage1v2_posterior_ablation_4gpu.sh
#   sbatch --export=ALL,STAGE1V2_MODE=oracle_holo_truth scripts/slurm/train_stage2_stage1v2_posterior_ablation_4gpu.sh

#SBATCH --job-name=s2_s1v2_ablate
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:A100:4
#SBATCH --cpus-per-task=16
#SBATCH --mem=180G
#SBATCH --time=18:00:00
#SBATCH --output=logs/slurm/stage2_s1v2_ablation_%j.out
#SBATCH --error=logs/slurm/stage2_s1v2_ablation_%j.err

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

NPROC_PER_NODE="${NPROC_PER_NODE:-${SLURM_GPUS_ON_NODE:-4}}"
STAGE1V2_MODE="${STAGE1V2_MODE:-student}"
STAGE1V2_FEATURES="${STAGE1V2_FEATURES:-contact_prob,active_prob,approach_prob,release_prob,confidence,teacher_min_dist_pred_norm,signed_delta_dist_pred_norm}"
STAGE1V2_FEATURE_SCALE="${STAGE1V2_FEATURE_SCALE:-1.0}"
STAGE1V2_TRAIN_CACHE_DIR="${STAGE1V2_TRAIN_CACHE_DIR:-logs/stage1v2_student_posteriors/stage1v2_best_train4096_cache_20260622_231423}"
STAGE1V2_VAL_CACHE_DIR="${STAGE1V2_VAL_CACHE_DIR:-logs/stage1v2_student_posteriors/stage1v2_best_val512_cache_20260622_231033}"
STAGE1V2_TRAIN_LABEL_DIR="${STAGE1V2_TRAIN_LABEL_DIR:-logs/stage1v2_teacher_posteriors/holo_truth_train4096_lcpgbf_20260622}"
STAGE1V2_VAL_LABEL_DIR="${STAGE1V2_VAL_LABEL_DIR:-logs/stage1v2_teacher_posteriors/holo_truth_val512_lcpgbf_20260622}"
STAGE1V2_LOSS_WEIGHT_MODE="${STAGE1V2_LOSS_WEIGHT_MODE:-none}"
STAGE1V2_LOSS_WEIGHT_ALPHA="${STAGE1V2_LOSS_WEIGHT_ALPHA:-0.0}"
W_STAGE1V2_GUIDANCE="${W_STAGE1V2_GUIDANCE:-0.0}"
STAGE1V2_GUIDANCE_FEATURE="${STAGE1V2_GUIDANCE_FEATURE:-contact_prob}"
STAGE1V2_GUIDANCE_MIN_PROB="${STAGE1V2_GUIDANCE_MIN_PROB:-0.0}"
STAGE1V2_GUIDANCE_T_MID="${STAGE1V2_GUIDANCE_T_MID:-0.3}"
TRAIN_N="${TRAIN_N:-4096}"
VAL_N="${VAL_N:-512}"
SUBSET_SEED="${SUBSET_SEED:-20260622}"
MAX_EPOCHS="${MAX_EPOCHS:-5}"
BATCH_SIZE="${BATCH_SIZE:-2}"
NUM_WORKERS="${NUM_WORKERS:-2}"
LR="${LR:-2e-5}"
VAL_T="${VAL_T:-0.5}"
SEED="${SEED:-42}"
AMP_DTYPE="${AMP_DTYPE:-bf16}"
GEOM_EVERY="${GEOM_EVERY:-1}"
N_INTEGRATION_STEPS="${N_INTEGRATION_STEPS:-3}"
N_GEOM_STEPS="${N_GEOM_STEPS:-4}"
W_CONTACT="${W_CONTACT:-1.0}"
CONTACT_LOSS_MODE="${CONTACT_LOSS_MODE:-holo_target}"
TAG_SUFFIX="${TAG_SUFFIX:-s1v2_${STAGE1V2_MODE}}"

case "$STAGE1V2_MODE" in
  none|zero|student|student_shuffled|oracle_holo_truth|external_teacher_cached) ;;
  *)
    echo "ERROR: STAGE1V2_MODE must be one of none, zero, student, student_shuffled, oracle_holo_truth, external_teacher_cached"
    exit 1
    ;;
esac

TRAIN_SUBSET_REL="ablation_subsets/stage2_s1v2_train_${TRAIN_N}_seed${SUBSET_SEED}.txt"
VAL_SUBSET_REL="ablation_subsets/stage2_s1v2_val_${VAL_N}_seed${SUBSET_SEED}.txt"
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

echo "=============================================="
echo "Stage-2 + Stage-1-v2 posterior ablation"
echo "=============================================="
echo "Job ID:          ${SLURM_JOB_ID:-NA}"
echo "Node:            ${SLURM_NODELIST:-NA}"
echo "GPUs:            $NPROC_PER_NODE"
echo "Mode:            $STAGE1V2_MODE"
echo "Features:        $STAGE1V2_FEATURES"
echo "Feature scale:   $STAGE1V2_FEATURE_SCALE"
echo "Train cache:     $STAGE1V2_TRAIN_CACHE_DIR"
echo "Val cache:       $STAGE1V2_VAL_CACHE_DIR"
echo "Train labels:    $STAGE1V2_TRAIN_LABEL_DIR"
echo "Val labels:      $STAGE1V2_VAL_LABEL_DIR"
echo "Loss weight:     $STAGE1V2_LOSS_WEIGHT_MODE alpha=$STAGE1V2_LOSS_WEIGHT_ALPHA"
echo "Guidance:        w=$W_STAGE1V2_GUIDANCE feature=$STAGE1V2_GUIDANCE_FEATURE min=$STAGE1V2_GUIDANCE_MIN_PROB t_mid=$STAGE1V2_GUIDANCE_T_MID"
echo "Train subset:    $TRAIN_SUBSET"
echo "Val subset:      $VAL_SUBSET"
echo "max_epochs:      $MAX_EPOCHS"
echo "batch/GPU:       $BATCH_SIZE"
echo "lr:              $LR"
echo "w_contact:       $W_CONTACT"
echo "geom_every:      $GEOM_EVERY"
echo "integration:     $N_INTEGRATION_STEPS"
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
  --stage1v2_train_label_dir "$STAGE1V2_TRAIN_LABEL_DIR" \
  --stage1v2_val_label_dir "$STAGE1V2_VAL_LABEL_DIR" \
  --stage1v2_posterior_feature_names "$STAGE1V2_FEATURES" \
  --stage1v2_posterior_feature_scale "$STAGE1V2_FEATURE_SCALE" \
  --stage1v2_loss_weight_mode "$STAGE1V2_LOSS_WEIGHT_MODE" \
  --stage1v2_loss_weight_alpha "$STAGE1V2_LOSS_WEIGHT_ALPHA" \
  --w_stage1v2_guidance "$W_STAGE1V2_GUIDANCE" \
  --stage1v2_guidance_feature "$STAGE1V2_GUIDANCE_FEATURE" \
  --stage1v2_guidance_min_prob "$STAGE1V2_GUIDANCE_MIN_PROB" \
  --stage1v2_guidance_t_mid "$STAGE1V2_GUIDANCE_T_MID" \
  --contact_loss_mode "$CONTACT_LOSS_MODE" \
  --w_contact "$W_CONTACT" \
  --n_integration_steps "$N_INTEGRATION_STEPS" \
  --n_geom_steps "$N_GEOM_STEPS" \
  --geom_loss_every_n_steps "$GEOM_EVERY" \
  --valid_samples_file "$TRAIN_SUBSET_REL" \
  --val_samples_file "$VAL_SUBSET_REL" \
  --save_dir "$SAVE_DIR" \
  --log_dir "$LOG_DIR" \
  --device cuda \
  --amp_dtype "$AMP_DTYPE" \
  --distributed

echo ""
echo "=============================================="
echo "Stage-2 + Stage-1-v2 posterior ablation completed: $(date)"
echo "Metrics: $LOG_DIR/metrics.jsonl"
echo "=============================================="
