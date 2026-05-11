#!/bin/bash
#SBATCH --job-name=s1_post_stable
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:A100:4
#SBATCH --cpus-per-task=16
#SBATCH --mem=160G
#SBATCH --output=logs/slurm/stage1_posterior_stable_%j.out
#SBATCH --error=logs/slurm/stage1_posterior_stable_%j.err

set -euo pipefail
unset LD_PRELOAD
unset PROXYCHAINS_CONF_FILE
unset PROXYCHAINS_QUIET_MODE

export PATH=/mnt/inaisfs/data/home/zhaozc_criait/miniconda3/bin:$PATH
export PYTHONUNBUFFERED=1
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export OMP_NUM_THREADS=4
export NCCL_DEBUG=WARN

source /mnt/inaisfs/data/home/zhaozc_criait/miniconda3/etc/profile.d/conda.sh
conda activate BINDRAE
ENV_PREFIX=/mnt/inaisfs/data/home/zhaozc_criait/miniconda3/envs/BINDRAE
export PATH="$ENV_PREFIX/bin:$PATH"
PYTHON_BIN="$ENV_PREFIX/bin/python"
TORCHRUN_BIN="$ENV_PREFIX/bin/torchrun"

ROOT=/mnt/inaisfs/data/home/zhaozc_criait/XinxiangWang/BINDRAE
cd "$ROOT"
mkdir -p logs/slurm logs/stage1 checkpoints/stage1

UNFREEZE_LIGAND="${UNFREEZE_LIGAND:-0}"
UNFREEZE_LAST_IPA="${UNFREEZE_LAST_IPA:-0}"
if [[ "$UNFREEZE_LIGAND" == "1" || "$UNFREEZE_LAST_IPA" != "0" ]]; then
  TAG_PREFIX="posterior_ligunfreeze"
else
  TAG_PREFIX="posterior_headonly_stable"
fi
TAG="${TAG:-${TAG_PREFIX}_$(date +%Y%m%d_%H%M%S)}"
TRAIN_TXT="${TRAIN_TXT:-$ROOT/processed_data/triplets/train_valid.txt}"
VAL_TXT="${VAL_TXT:-$ROOT/processed_data/triplets/val_valid.txt}"
RESUME_FROM="${RESUME_FROM:-$ROOT/checkpoints/stage1/opt8e2_20260411_122707/epoch_018.pt}"
LAMBDA_ROT="${LAMBDA_ROT:-0.1}"
LAMBDA_CONTACT="${LAMBDA_CONTACT:-0.1}"
LAMBDA_CANDIDATE_CHI1="${LAMBDA_CANDIDATE_CHI1:-0.0}"
LAMBDA_LIGAND_CONTRASTIVE="${LAMBDA_LIGAND_CONTRASTIVE:-0.0}"
LIGAND_CONTRASTIVE_MARGIN="${LIGAND_CONTRASTIVE_MARGIN:-0.2}"
LR="${LR:-1e-4}"
MAX_EPOCHS="${MAX_EPOCHS:-40}"
EXTRA_UNFREEZE_ARGS=()
if [[ "$UNFREEZE_LIGAND" == "1" ]]; then
  EXTRA_UNFREEZE_ARGS+=(--unfreeze_ligand_conditioner_for_posteriors)
fi
if [[ "$UNFREEZE_LAST_IPA" != "0" ]]; then
  EXTRA_UNFREEZE_ARGS+=(--unfreeze_last_ipa_blocks_for_posteriors "$UNFREEZE_LAST_IPA")
fi

echo "=============================================="
echo "Stage-1 posterior head-only stable run"
echo "=============================================="
echo "Job ID:          $SLURM_JOB_ID"
echo "Node:            $SLURM_NODELIST"
echo "Tag:             $TAG"
echo "Train list:      $TRAIN_TXT"
echo "Val list:        $VAL_TXT"
echo "Resume:          $RESUME_FROM"
echo "LR:              $LR"
echo "Lambda rotamer:  $LAMBDA_ROT"
echo "Lambda contact:  $LAMBDA_CONTACT"
echo "Lambda candidate:$LAMBDA_CANDIDATE_CHI1"
echo "Lambda contrast: $LAMBDA_LIGAND_CONTRASTIVE"
echo "Contrast margin: $LIGAND_CONTRASTIVE_MARGIN"
echo "Max epochs:      $MAX_EPOCHS"
echo "Unfreeze ligand: $UNFREEZE_LIGAND"
echo "Unfreeze IPA:    $UNFREEZE_LAST_IPA"
echo "Start:           $(date)"
echo "=============================================="

nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader
"$PYTHON_BIN" -c "import torch; print(f'PyTorch {torch.__version__}, CUDA {torch.version.cuda}, GPUs: {torch.cuda.device_count()}')"

"$TORCHRUN_BIN" \
  --standalone \
  --nproc_per_node=4 \
  scripts/train_stage1.py \
  --model_size medium \
  --data_dir processed_data/triplets \
  --valid_samples_file "$TRAIN_TXT" \
  --val_samples_file "$VAL_TXT" \
  --sample_metadata_file sample_metadata.json \
  --batch_size 4 \
  --num_workers 4 \
  --max_n_res 1600 \
  --length_bucketed_sampling \
  --residue_budget 1600 \
  --lr "$LR" \
  --warmup_steps 1000 \
  --lr_scheduler plateau \
  --plateau_factor 0.5 \
  --plateau_patience 3 \
  --min_lr_scale 0.01 \
  --pocket_warmup_steps 8000 \
  --ligand_gate_warmup_steps 8000 \
  --max_epochs "$MAX_EPOCHS" \
  --grad_clip 0.5 \
  --patience 10 \
  --w_fape 1.0 \
  --w_chi 1.0 \
  --w_clash 0.1 \
  --lambda_chi1_rotamer "$LAMBDA_ROT" \
  --lambda_contact "$LAMBDA_CONTACT" \
  --lambda_candidate_chi1 "$LAMBDA_CANDIDATE_CHI1" \
  --lambda_ligand_contrastive "$LAMBDA_LIGAND_CONTRASTIVE" \
  --ligand_contrastive_margin "$LIGAND_CONTRASTIVE_MARGIN" \
  --freeze_stage1_backbone_for_posteriors \
  "${EXTRA_UNFREEZE_ARGS[@]}" \
  --selection_metric chi1_rotamer_acc \
  --compute_slow_metrics \
  --enable_dual_mask_audit \
  --save_epoch_checkpoints \
  --resume_from "$RESUME_FROM" \
  --save_dir "checkpoints/stage1/${TAG}" \
  --log_dir "logs/stage1/${TAG}" \
  --distributed

echo ""
echo "=============================================="
echo "Posterior stable run completed: $(date)"
echo "Checkpoint saved to: checkpoints/stage1/$TAG"
echo "=============================================="
