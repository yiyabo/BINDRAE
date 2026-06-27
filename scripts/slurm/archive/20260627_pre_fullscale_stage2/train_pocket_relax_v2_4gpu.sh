#!/bin/bash
#SBATCH --job-name=s1_pocket_relax_v2
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:A100:4
#SBATCH --cpus-per-task=8
#SBATCH --mem=256G
#SBATCH --time=06:00:00
#SBATCH --output=logs/slurm/stage1_pocket_relax_v2_%j.out
#SBATCH --error=logs/slurm/stage1_pocket_relax_v2_%j.err

set -e
unset LD_PRELOAD
unset PROXYCHAINS_CONF_FILE
unset PROXYCHAINS_QUIET_MODE

export PATH=/mnt/inaisfs/data/home/zhaozc_criait/miniconda3/bin:$PATH
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=4
export DDP_TIMEOUT="${DDP_TIMEOUT:-7200}"

source /mnt/inaisfs/data/home/zhaozc_criait/miniconda3/etc/profile.d/conda.sh
conda activate BINDRAE

cd /mnt/inaisfs/data/home/zhaozc_criait/XinxiangWang/BINDRAE
mkdir -p logs/slurm logs/stage1 checkpoints/stage1

BATCH_SIZE="${BATCH_SIZE:-80}"
MAX_EPOCHS="${MAX_EPOCHS:-12}"
NUM_WORKERS="${NUM_WORKERS:-4}"
TRAIN_MAX_SAMPLES="${TRAIN_MAX_SAMPLES:-12000}"
VAL_MAX_SAMPLES="${VAL_MAX_SAMPLES:-1000}"
SUBSET_SEED="${SUBSET_SEED:-20260617}"
MAX_LOCAL_RES="${MAX_LOCAL_RES:-192}"
LR="${LR:-2e-4}"
LAMBDA_BASE_CE="${LAMBDA_BASE_CE:-0.0}"
LAMBDA_CONTACT_CE="${LAMBDA_CONTACT_CE:-0.2}"
LAMBDA_SWITCH_CE="${LAMBDA_SWITCH_CE:-0.6}"
LAMBDA_RERANK="${LAMBDA_RERANK:-1.0}"
LAMBDA_CONTRASTIVE="${LAMBDA_CONTRASTIVE:-1.0}"
LAMBDA_G="${LAMBDA_G:-0.2}"
LAMBDA_ANTIHARM="${LAMBDA_ANTIHARM:-0.05}"
DECOY_MARGIN="${DECOY_MARGIN:-0.2}"
RANK_MARGIN="${RANK_MARGIN:-0.05}"
G_MARGIN="${G_MARGIN:-0.05}"
RESIDUAL_BETA="${RESIDUAL_BETA:-1.0}"
BASE_TEMPERATURE="${BASE_TEMPERATURE:-8.0}"
STAGE1_CHECKPOINT="${STAGE1_CHECKPOINT:-checkpoints/stage1/phase1_residual_fullscratch_ligcausal_16gpu_stable_20260509_000057/epoch_018.pt}"
TAG_PREFIX="${TAG_PREFIX:-pocket_relax_v2}"

TAG="${TAG_PREFIX}_bs${BATCH_SIZE}_local${MAX_LOCAL_RES}_bt${BASE_TEMPERATURE}_$(date +%Y%m%d_%H%M%S)"
SAVE_DIR="checkpoints/stage1/${TAG}"
LOG_DIR="logs/stage1/${TAG}"

EXTRA_ARGS=()
if [[ -n "${STAGE1_CHECKPOINT:-}" ]]; then
  EXTRA_ARGS+=(--stage1_checkpoint "$STAGE1_CHECKPOINT")
fi
if [[ "${RESET_RESIDUAL:-1}" == "1" ]]; then
  EXTRA_ARGS+=(--reset_residual)
fi
if [[ "${FREEZE_BASE:-1}" == "1" ]]; then
  EXTRA_ARGS+=(--freeze_base)
fi
if [[ "${FREEZE_GATE:-1}" == "1" ]]; then
  EXTRA_ARGS+=(--freeze_gate)
fi
if [[ "${USE_TYPED_ENERGY:-1}" == "1" ]]; then
  EXTRA_ARGS+=(--use_typed_energy)
fi

echo "=============================================="
echo "Stage-1 Pocket Relax v2 (4-GPU DDP)"
echo "=============================================="
echo "Job ID:          $SLURM_JOB_ID"
echo "Node:            $SLURM_NODELIST"
echo "GPUs:            4x A100"
echo "Tag:             $TAG"
echo "Save dir:        $SAVE_DIR"
echo "Log dir:         $LOG_DIR"
echo "Stage1 ckpt:     ${STAGE1_CHECKPOINT:-<none>}"
echo "Batch size:      $BATCH_SIZE per GPU (global $((BATCH_SIZE * 4)))"
echo "Max local res:   $MAX_LOCAL_RES"
echo "Train subset:    $TRAIN_MAX_SAMPLES"
echo "Val subset:      $VAL_MAX_SAMPLES"
echo "Max epochs:      $MAX_EPOCHS"
echo "LR:              $LR"
echo "Loss weights:    base=$LAMBDA_BASE_CE contact=$LAMBDA_CONTACT_CE switch=$LAMBDA_SWITCH_CE rerank=$LAMBDA_RERANK contrastive=$LAMBDA_CONTRASTIVE g=$LAMBDA_G antiharm=$LAMBDA_ANTIHARM"
echo "Margins:         decoy=$DECOY_MARGIN rank=$RANK_MARGIN g=$G_MARGIN"
echo "Residual beta:   $RESIDUAL_BETA"
echo "Base temp:       $BASE_TEMPERATURE"
echo "Start:           $(date)"
echo "=============================================="

torchrun --nproc_per_node=4 scripts/train_pocket_relax_v2.py \
  --data_dir processed_data/triplets \
  --val_samples_file /mnt/inaisfs/data/home/zhaozc_criait/XinxiangWang/BINDRAE/processed_data/triplets/val_valid.txt \
  --sample_metadata_file sample_metadata.json \
  --batch_size "$BATCH_SIZE" \
  --train_max_samples "$TRAIN_MAX_SAMPLES" \
  --val_max_samples "$VAL_MAX_SAMPLES" \
  --subset_seed "$SUBSET_SEED" \
  --max_n_res 1600 \
  --max_local_res "$MAX_LOCAL_RES" \
  --lr "$LR" \
  --max_epochs "$MAX_EPOCHS" \
  --lambda_base_ce "$LAMBDA_BASE_CE" \
  --lambda_contact_ce "$LAMBDA_CONTACT_CE" \
  --lambda_switch_ce "$LAMBDA_SWITCH_CE" \
  --lambda_rerank "$LAMBDA_RERANK" \
  --lambda_contrastive "$LAMBDA_CONTRASTIVE" \
  --lambda_g "$LAMBDA_G" \
  --lambda_antiharm "$LAMBDA_ANTIHARM" \
  --decoy_margin "$DECOY_MARGIN" \
  --rank_margin "$RANK_MARGIN" \
  --g_margin "$G_MARGIN" \
  --residual_beta "$RESIDUAL_BETA" \
  --base_temperature "$BASE_TEMPERATURE" \
  --save_dir "$SAVE_DIR" \
  --log_dir "$LOG_DIR" \
  --num_workers "$NUM_WORKERS" \
  --device cuda \
  --distributed \
  "${EXTRA_ARGS[@]}"

echo ""
echo "=============================================="
echo "Pocket Relax v2 completed: $(date)"
echo "Checkpoints: $SAVE_DIR"
echo "Metrics: $LOG_DIR/metrics.jsonl"
echo "=============================================="
