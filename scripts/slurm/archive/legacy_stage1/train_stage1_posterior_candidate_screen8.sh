#!/bin/bash
#SBATCH --job-name=s1_postcand8
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --gres=gpu:A100:8
#SBATCH --cpus-per-task=4
#SBATCH --mem=320G
#SBATCH --time=16:00:00
#SBATCH --output=logs/slurm/stage1_posterior_candidate_screen8_%j.out
#SBATCH --error=logs/slurm/stage1_posterior_candidate_screen8_%j.err

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

TAG="${TAG:-posterior_candidate_ligcausal_screen8_$(date +%Y%m%d_%H%M%S)}"
TRAIN_TXT="${TRAIN_TXT:-$ROOT/processed_data/triplets/train_valid.txt}"
VAL_TXT="${VAL_TXT:-$ROOT/processed_data/triplets/val_valid.txt}"
RESUME_FROM="${RESUME_FROM:-$ROOT/checkpoints/stage1/candidate_chi1_lcon_single_smoke_20260428_113514/best_model.pt}"
LR="${LR:-3e-5}"
# Stage-1 branch-resume preserves the source checkpoint epoch counter.
# The default resume checkpoint is at epoch 22, so max_epochs=30 gives this
# screen roughly eight additional epochs instead of exiting immediately.
MAX_EPOCHS="${MAX_EPOCHS:-30}"
LAMBDA_CANDIDATE_CHI1="${LAMBDA_CANDIDATE_CHI1:-0.5}"
LAMBDA_LIGAND_CONTRASTIVE="${LAMBDA_LIGAND_CONTRASTIVE:-0.3}"
LIGAND_CONTRASTIVE_MARGIN="${LIGAND_CONTRASTIVE_MARGIN:-0.5}"
CANDIDATE_SCORER_LR_SCALE="${CANDIDATE_SCORER_LR_SCALE:-8.0}"
LAMBDA_CONTACT="${LAMBDA_CONTACT:-0.2}"
MASTER_PORT="${MASTER_PORT:-29520}"

cat <<EOF
============================================================
Stage-1 candidate/contact posterior ligand-causality screen
============================================================
Job ID:           ${SLURM_JOB_ID:-NA}
Node:             ${SLURM_NODELIST:-NA}
GPUs:             8 x A100 on one node
Tag:              $TAG
Resume:           $RESUME_FROM
Train samples:    $TRAIN_TXT
Val samples:      $VAL_TXT
LR:               $LR (candidate scorer x${CANDIDATE_SCORER_LR_SCALE})
Max epochs:       $MAX_EPOCHS
Candidate CE:     $LAMBDA_CANDIDATE_CHI1
Lig contrastive:  $LAMBDA_LIGAND_CONTRASTIVE
Contrast margin:  $LIGAND_CONTRASTIVE_MARGIN
Contact BCE:      $LAMBDA_CONTACT

Purpose:
  Validate the Stage-1 pivot from deterministic endpoint / G-vector residual
  training toward a ligand-conditioned pocket posterior. This run trains the
  candidate chi1 scorer and contact head while freezing the deterministic
  trunk and unfreezing ligand conditioning.

Success criterion:
  Produce a checkpoint for posterior diagnostics where correct-ligand
  candidate/contact posterior utility beats no-ligand and shuffled-ligand on
  ligand-facing / pocket residues. Training metrics alone are not final.

Kill criterion:
  If validation candidate/contact losses diverge or startup shows DDP/NaN
  instability, stop and lower LR/contrastive weight before retrying.
Start: $(date)
============================================================
EOF

nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader

"$TORCHRUN_BIN" \
  --nproc_per_node=8 \
  --master_port="$MASTER_PORT" \
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
  --plateau_patience 3 \
  --min_lr_scale 0.01 \
  --pocket_warmup_steps 0 \
  --ligand_gate_warmup_steps 0 \
  --max_epochs "$MAX_EPOCHS" \
  --grad_clip 0.5 \
  --patience 8 \
  --w_fape 1.0 \
  --w_chi 1.0 \
  --w_clash 0.1 \
  --lambda_chi1_rotamer 0.0 \
  --lambda_contact "$LAMBDA_CONTACT" \
  --lambda_candidate_chi1 "$LAMBDA_CANDIDATE_CHI1" \
  --lambda_ligand_contrastive "$LAMBDA_LIGAND_CONTRASTIVE" \
  --ligand_contrastive_margin "$LIGAND_CONTRASTIVE_MARGIN" \
  --candidate_scorer_lr_scale "$CANDIDATE_SCORER_LR_SCALE" \
  --freeze_stage1_backbone_for_posteriors \
  --unfreeze_ligand_conditioner_for_posteriors \
  --selection_metric chi1_rotamer_acc \
  --compute_slow_metrics \
  --enable_dual_mask_audit \
  --save_epoch_checkpoints \
  --resume_from "$RESUME_FROM" \
  --save_dir "checkpoints/stage1/${TAG}" \
  --log_dir "logs/stage1/${TAG}"

echo ""
echo "============================================================"
echo "Posterior candidate screen completed: $(date)"
echo "Checkpoint: checkpoints/stage1/$TAG"
echo "Metrics:    logs/stage1/$TAG/metrics.jsonl"
echo "Next: run posterior diagnostics on best_model.pt with correct/no/shuffled ligands"
echo "============================================================"
