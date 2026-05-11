#!/bin/bash
# Minimal Stage-2 smoke test.
# Usage examples:
#   PRIOR_MODE=none sbatch scripts/slurm/train_stage2_smoke.sh
#   PRIOR_MODE=e2   STAGE1_CKPT=/abs/path/to/e2/best_model.pt sbatch scripts/slurm/train_stage2_smoke.sh
#   PRIOR_MODE=e6b  STAGE1_CKPT=/abs/path/to/e6b/best_model.pt sbatch scripts/slurm/train_stage2_smoke.sh
#   PRIOR_MODE=oracle_holo STAGE1_PRIOR_MODE=holo_chi sbatch scripts/slurm/train_stage2_smoke.sh

#SBATCH --job-name=s2_smoke
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:A100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=logs/slurm/stage2_smoke_%j.out
#SBATCH --error=logs/slurm/stage2_smoke_%j.err

set -euo pipefail
unset LD_PRELOAD
unset PROXYCHAINS_CONF_FILE
unset PROXYCHAINS_QUIET_MODE

PRIOR_MODE="${PRIOR_MODE:-none}"
STAGE1_PRIOR_MODE="${STAGE1_PRIOR_MODE:-stage1}"
STAGE1_PRIOR_NOISE_SCALE="${STAGE1_PRIOR_NOISE_SCALE:-0.0}"
USE_STAGE1_RIGID_PRIOR="${USE_STAGE1_RIGID_PRIOR:-1}"
STAGE1_CHI_FEATURE_SCALE="${STAGE1_CHI_FEATURE_SCALE:-1.0}"
if [[ "${PRIOR_MODE}" != "none" && "${PRIOR_MODE}" != "e2" && "${PRIOR_MODE}" != "e6b" && "${PRIOR_MODE}" != "oracle_apo" && "${PRIOR_MODE}" != "oracle_holo" && "${PRIOR_MODE}" != "oracle_holo_lf" ]]; then
  echo "ERROR: PRIOR_MODE must be one of: none, e2, e6b, oracle_apo, oracle_holo, oracle_holo_lf"
  exit 1
fi

export PATH=/mnt/inaisfs/data/home/zhaozc_criait/miniconda3/bin:$PATH
export PYTHONUNBUFFERED=1
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export OMP_NUM_THREADS=4

USE_POCKET_LOCAL_PRIOR="${USE_POCKET_LOCAL_PRIOR:-0}"
PRIOR_POCKET_THRESHOLD="${PRIOR_POCKET_THRESHOLD:-0.3}"
T_MID="${T_MID:-0.0}"
W_PRIOR="${W_PRIOR:-0.1}"
MAX_EPOCHS="${MAX_EPOCHS:-2}"
VAL_T="${VAL_T:-0.5}"

source /mnt/inaisfs/data/home/zhaozc_criait/miniconda3/etc/profile.d/conda.sh
conda activate BINDRAE

cd /mnt/inaisfs/data/home/zhaozc_criait/XinxiangWang/BINDRAE
mkdir -p logs/slurm logs/stage2 checkpoints/stage2

SMOKE_TAG="stage2_smoke_${PRIOR_MODE}_$(date +%Y%m%d_%H%M%S)"
TRAIN_TINY="processed_data/triplets/stage2_smoke_train.txt"
VAL_TINY="processed_data/triplets/stage2_smoke_val.txt"

if [[ ! -f "${TRAIN_TINY}" || ! -f "${VAL_TINY}" ]]; then
  echo "ERROR: smoke subset files not found: ${TRAIN_TINY} / ${VAL_TINY}"
  exit 1
fi

NO_PRIOR_FLAG=""
STAGE1_CKPT_ARG=()
if [[ "${PRIOR_MODE}" == "none" ]]; then
  NO_PRIOR_FLAG="--no_stage1_prior"
else
  if [[ -z "${STAGE1_CKPT:-}" ]]; then
    if [[ "${STAGE1_PRIOR_MODE}" == "stage1" || "${STAGE1_PRIOR_MODE}" == "noisy_stage1_chi" ]]; then
      echo "ERROR: STAGE1_CKPT must be set when STAGE1_PRIOR_MODE=${STAGE1_PRIOR_MODE}"
      exit 1
    fi
  fi
  if [[ -n "${STAGE1_CKPT:-}" ]]; then
    STAGE1_CKPT_ARG=(--stage1_ckpt "${STAGE1_CKPT}")
  fi
fi

echo "=============================================="
echo "Stage-2 smoke test"
echo "=============================================="
echo "Job ID:        $SLURM_JOB_ID"
echo "Node:          $SLURM_NODELIST"
echo "Prior mode:    ${PRIOR_MODE}"
echo "Prior source:  ${STAGE1_PRIOR_MODE}"
echo "Prior noise:   ${STAGE1_PRIOR_NOISE_SCALE}"
echo "Rigid prior:   ${USE_STAGE1_RIGID_PRIOR}"
echo "Chi scale:     ${STAGE1_CHI_FEATURE_SCALE}"
echo "Stage1 ckpt:   ${STAGE1_CKPT:-OFF}"
echo "Pocket-local:  ${USE_POCKET_LOCAL_PRIOR}"
echo "Prior thresh:  ${PRIOR_POCKET_THRESHOLD}"
echo "t_mid:         ${T_MID}"
echo "w_prior:       ${W_PRIOR}"
echo "max_epochs:    ${MAX_EPOCHS}"
echo "val_t:         ${VAL_T}"
echo "Train subset:  ${TRAIN_TINY}"
echo "Val subset:    ${VAL_TINY}"
echo "Save dir:      checkpoints/stage2/${SMOKE_TAG}"
echo "Log dir:       logs/stage2/${SMOKE_TAG}"
echo "Start:         $(date)"
echo "=============================================="

nvidia-smi --query-gpu=index,name,memory.total,memory.free --format=csv,noheader
echo ""
python -c "import torch; print(f'PyTorch {torch.__version__}, CUDA {torch.version.cuda}, GPUs: {torch.cuda.device_count()}')"
echo ""

POCKET_LOCAL_ARGS=()
if [[ "${USE_POCKET_LOCAL_PRIOR}" == "1" ]]; then
  POCKET_LOCAL_ARGS+=(--use_pocket_local_prior --prior_pocket_threshold "${PRIOR_POCKET_THRESHOLD}")
fi

RIGID_PRIOR_ARGS=()
if [[ "${USE_STAGE1_RIGID_PRIOR}" == "0" ]]; then
  RIGID_PRIOR_ARGS+=(--no_stage1_rigid_prior)
fi

python scripts/train_stage2.py \
  --data_dir processed_data/triplets \
  --batch_size 1 \
  --num_workers 0 \
  --max_epochs "${MAX_EPOCHS}" \
  --lr 2e-5 \
  --grad_clip 0.3 \
  --accum_steps 1 \
  --warmup_steps 0 \
  --seed 42 \
  --val_t "${VAL_T}" \
  --w_prior "${W_PRIOR}" \
  --stage1_prior_mode "${STAGE1_PRIOR_MODE}" \
  --stage1_prior_noise_scale "${STAGE1_PRIOR_NOISE_SCALE}" \
  --stage1_chi_feature_scale "${STAGE1_CHI_FEATURE_SCALE}" \
  --t_mid "${T_MID}" \
  --n_integration_steps 2 \
  --n_geom_steps 2 \
  --geom_loss_every_n_steps 1000000 \
  --valid_samples_file stage2_smoke_train.txt \
  --val_samples_file stage2_smoke_val.txt \
  --save_dir checkpoints/stage2/${SMOKE_TAG} \
  --log_dir logs/stage2/${SMOKE_TAG} \
  --device cuda \
  "${POCKET_LOCAL_ARGS[@]}" \
  "${RIGID_PRIOR_ARGS[@]}" \
  ${NO_PRIOR_FLAG} \
  "${STAGE1_CKPT_ARG[@]}"

echo ""
echo "=============================================="
echo "Stage-2 smoke completed: $(date)"
echo "=============================================="
