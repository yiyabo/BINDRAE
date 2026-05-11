#!/bin/bash
#SBATCH --job-name=s1_cand_ext
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:A100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=24:00:00
#SBATCH --output=logs/slurm/stage1_candidate_ext_%j.out
#SBATCH --error=logs/slurm/stage1_candidate_ext_%j.err

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

PYTHON_BIN="$ENV_PREFIX/bin/python"

TAG="${TAG:-candidate_chi1_ext_$(date +%Y%m%d_%H%M%S)}"
TRAIN_TXT="${TRAIN_TXT:-$ROOT/processed_data/triplets/train_valid.txt}"
VAL_TXT="${VAL_TXT:-$ROOT/processed_data/triplets/val_valid.txt}"
RESUME_FROM="${RESUME_FROM:-$ROOT/checkpoints/stage1/candidate_chi1_lcon_single_smoke_20260428_113514/best_model.pt}"
LR="${LR:-3e-5}"
MAX_EPOCHS="${MAX_EPOCHS:-40}"
LAMBDA_CANDIDATE_CHI1="${LAMBDA_CANDIDATE_CHI1:-0.3}"
LAMBDA_LIGAND_CONTRASTIVE="${LAMBDA_LIGAND_CONTRASTIVE:-0.1}"
LIGAND_CONTRASTIVE_MARGIN="${LIGAND_CONTRASTIVE_MARGIN:-0.5}"
CANDIDATE_SCORER_LR_SCALE="${CANDIDATE_SCORER_LR_SCALE:-10.0}"

echo "=============================================="
echo "Stage-1 candidate-aware EXTENDED training"
echo "=============================================="
echo "Job ID:     $SLURM_JOB_ID"
echo "Node:       $SLURM_NODELIST"
echo "Tag:        $TAG"
echo "Resume:     $RESUME_FROM"
echo "LR:         $LR"
echo "Scorer LR:  ${LR} x ${CANDIDATE_SCORER_LR_SCALE}"
echo "Max epochs: $MAX_EPOCHS"
echo "Lambda cand:$LAMBDA_CANDIDATE_CHI1"
echo "Lambda con: $LAMBDA_LIGAND_CONTRASTIVE"
echo "Margin:     $LIGAND_CONTRASTIVE_MARGIN"
echo "Start:      $(date)"
echo "=============================================="

nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader
"$PYTHON_BIN" -c "import torch; print(f'PyTorch {torch.__version__}, CUDA {torch.version.cuda}, GPUs: {torch.cuda.device_count()}')"
"$PYTHON_BIN" -m py_compile \
  src/stage1/models/torsion_head.py \
  src/stage1/models/stage1_model.py \
  src/stage1/modules/losses.py \
  src/stage1/training/config.py \
  src/stage1/training/trainer.py \
  scripts/train_stage1.py

"$PYTHON_BIN" scripts/train_stage1.py \
  --model_size medium \
  --data_dir processed_data/triplets \
  --valid_samples_file "$TRAIN_TXT" \
  --val_samples_file "$VAL_TXT" \
  --sample_metadata_file sample_metadata.json \
  --batch_size 2 \
  --num_workers 0 \
  --max_n_res 900 \
  --length_bucketed_sampling \
  --residue_budget 900 \
  --lr "$LR" \
  --warmup_steps 500 \
  --lr_scheduler plateau \
  --plateau_factor 0.5 \
  --plateau_patience 3 \
  --min_lr_scale 0.01 \
  --pocket_warmup_steps 0 \
  --ligand_gate_warmup_steps 0 \
  --max_epochs "$MAX_EPOCHS" \
  --grad_clip 0.5 \
  --patience 10 \
  --w_fape 1.0 \
  --w_chi 1.0 \
  --w_clash 0.1 \
  --lambda_chi1_rotamer 0.0 \
  --lambda_contact 0.0 \
  --lambda_candidate_chi1 "$LAMBDA_CANDIDATE_CHI1" \
  --lambda_ligand_contrastive "$LAMBDA_LIGAND_CONTRASTIVE" \
  --ligand_contrastive_margin "$LIGAND_CONTRASTIVE_MARGIN" \
  --candidate_scorer_lr_scale "$CANDIDATE_SCORER_LR_SCALE" \
  --freeze_stage1_backbone_for_posteriors \
  --unfreeze_ligand_conditioner_for_posteriors \
  --selection_metric chi1_rotamer_acc \
  --compute_slow_metrics \
  --save_epoch_checkpoints \
  --resume_from "$RESUME_FROM" \
  --save_dir "checkpoints/stage1/${TAG}" \
  --log_dir "logs/stage1/${TAG}"

echo ""
echo "=============================================="
echo "Candidate extended completed: $(date)"
echo "Checkpoint saved to: checkpoints/stage1/$TAG"
echo "=============================================="
