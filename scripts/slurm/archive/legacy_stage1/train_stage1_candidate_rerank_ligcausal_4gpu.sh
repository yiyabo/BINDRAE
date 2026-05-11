#!/bin/bash
#SBATCH --job-name=s1_cand_rerank4
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:A100:4
#SBATCH --cpus-per-task=8
#SBATCH --mem=220G
#SBATCH --time=16:00:00
#SBATCH --output=logs/slurm/stage1_candidate_rerank_ligcausal_4gpu_%j.out
#SBATCH --error=logs/slurm/stage1_candidate_rerank_ligcausal_4gpu_%j.err

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
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1

cd "$ROOT"
mkdir -p logs/slurm logs/stage1 checkpoints/stage1

source /mnt/inaisfs/data/home/zhaozc_criait/miniconda3/etc/profile.d/conda.sh
conda activate BINDRAE

TORCHRUN_BIN="$ENV_PREFIX/bin/torchrun"

TAG="${TAG:-candidate_rerank_ligcausal_4gpu_$(date +%Y%m%d_%H%M%S)}"
TRAIN_TXT="${TRAIN_TXT:-$ROOT/processed_data/triplets/train_valid.txt}"
VAL_TXT="${VAL_TXT:-$ROOT/processed_data/triplets/val_valid.txt}"
RESUME_FROM="${RESUME_FROM:-$ROOT/checkpoints/stage1/posterior_candidate_ligcausal_screen4safe_20260509_184220/best_model.pt}"
LR="${LR:-1e-5}"
MAX_EPOCHS="${MAX_EPOCHS:-30}"
LAMBDA_CANDIDATE_CHI1="${LAMBDA_CANDIDATE_CHI1:-0.2}"
LAMBDA_CANDIDATE_RERANK="${LAMBDA_CANDIDATE_RERANK:-0.8}"
CANDIDATE_RERANK_MARGIN="${CANDIDATE_RERANK_MARGIN:-0.15}"
CANDIDATE_RERANK_RANK_MARGIN="${CANDIDATE_RERANK_RANK_MARGIN:-0.03}"
CANDIDATE_RERANK_DECOY_KIND="${CANDIDATE_RERANK_DECOY_KIND:-translated}"
CANDIDATE_SCORER_LR_SCALE="${CANDIDATE_SCORER_LR_SCALE:-6.0}"
LAMBDA_CONTACT="${LAMBDA_CONTACT:-0.1}"
MASTER_PORT="${MASTER_PORT:-29531}"

cat <<EOF
============================================================
Stage-1 explicit ligand-causal candidate reranking (4-GPU)
============================================================
Job ID:              ${SLURM_JOB_ID:-NA}
Node:                ${SLURM_NODELIST:-NA}
GPUs:                4 x A100 on one node
Tag:                 $TAG
Resume:              $RESUME_FROM
Train samples:       $TRAIN_TXT
Val samples:         $VAL_TXT
LR:                  $LR (candidate scorer x${CANDIDATE_SCORER_LR_SCALE})
Max epochs:          $MAX_EPOCHS
Candidate CE:        $LAMBDA_CANDIDATE_CHI1
Candidate rerank:    $LAMBDA_CANDIDATE_RERANK
Rerank margins:      decoy=$CANDIDATE_RERANK_MARGIN rank=$CANDIDATE_RERANK_RANK_MARGIN
Rerank decoy:        $CANDIDATE_RERANK_DECOY_KIND
Contact BCE:         $LAMBDA_CONTACT

Why this lane:
  Previous posterior/candidate diagnostics showed no correct-ligand lift over
  no-ligand/translated/shuffled decoys on switch rotamers. This branch trains
  the candidate posterior directly to rank the correct-ligand holo candidate
  above decoy-ligand candidates on contact switch residues.
Start: $(date)
============================================================
EOF

nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader

"$TORCHRUN_BIN" \
  --nproc_per_node=4 \
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
  --patience 6 \
  --w_fape 1.0 \
  --w_chi 1.0 \
  --w_clash 0.1 \
  --lambda_chi1_rotamer 0.0 \
  --lambda_contact "$LAMBDA_CONTACT" \
  --lambda_candidate_chi1 "$LAMBDA_CANDIDATE_CHI1" \
  --lambda_candidate_rerank "$LAMBDA_CANDIDATE_RERANK" \
  --candidate_rerank_margin "$CANDIDATE_RERANK_MARGIN" \
  --candidate_rerank_rank_margin "$CANDIDATE_RERANK_RANK_MARGIN" \
  --candidate_rerank_decoy_kind "$CANDIDATE_RERANK_DECOY_KIND" \
  --candidate_scorer_lr_scale "$CANDIDATE_SCORER_LR_SCALE" \
  --freeze_stage1_backbone_for_posteriors \
  --unfreeze_ligand_conditioner_for_posteriors \
  --selection_metric candidate_decoy_lift_contact_switch_rotamer_acc \
  --compute_slow_metrics \
  --enable_dual_mask_audit \
  --ddp_find_unused_parameters \
  --save_epoch_checkpoints \
  --resume_from "$RESUME_FROM" \
  --save_dir "checkpoints/stage1/${TAG}" \
  --log_dir "logs/stage1/${TAG}"

echo ""
echo "============================================================"
echo "Candidate reranking ligand-causal run completed: $(date)"
echo "Checkpoint: checkpoints/stage1/$TAG"
echo "Metrics:    logs/stage1/$TAG/metrics.jsonl"
echo "Next: run candidate-lift diagnostics on best_model.pt"
echo "============================================================"
