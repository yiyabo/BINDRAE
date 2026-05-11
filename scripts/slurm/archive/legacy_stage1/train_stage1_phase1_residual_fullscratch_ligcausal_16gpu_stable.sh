#!/bin/bash
#SBATCH --job-name=s1_p1_lift16_stable
#SBATCH --partition=gpu
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:A100:8
#SBATCH --cpus-per-task=32
#SBATCH --mem=384G
#SBATCH --time=72:00:00
#SBATCH --output=logs/slurm/stage1_phase1_fullscratch_ligcausal_16gpu_stable_%j.out
#SBATCH --error=logs/slurm/stage1_phase1_fullscratch_ligcausal_16gpu_stable_%j.err

set -euo pipefail
unset LD_PRELOAD
unset PROXYCHAINS_CONF_FILE
unset PROXYCHAINS_QUIET_MODE

ROOT=/mnt/inaisfs/data/home/zhaozc_criait/XinxiangWang/BINDRAE
ENV_PREFIX=/mnt/inaisfs/data/home/zhaozc_criait/miniconda3/envs/BINDRAE

export PATH=/data/soft/slurm/24.11.4/bin:$ENV_PREFIX/bin:$PATH
export PYTHONUNBUFFERED=1
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export OMP_NUM_THREADS=4
export NCCL_TIMEOUT=7200
export NCCL_DEBUG=WARN
export NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-^lo,docker}"
export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-0}"

cd "$ROOT"
mkdir -p logs/slurm logs/stage1 checkpoints/stage1

source /mnt/inaisfs/data/home/zhaozc_criait/miniconda3/etc/profile.d/conda.sh
conda activate BINDRAE

MASTER_ADDR="$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)"
MASTER_PORT="${MASTER_PORT:-29526}"
NNODES="${SLURM_NNODES:-2}"
NPROC_PER_NODE=8
WORLD_SIZE=$((NNODES * NPROC_PER_NODE))

TAG="${TAG:-phase1_residual_fullscratch_ligcausal_16gpu_stable_$(date +%Y%m%d_%H%M%S)}"
TRAIN_TXT="${TRAIN_TXT:-$ROOT/processed_data/triplets/train_valid.txt}"
VAL_TXT="${VAL_TXT:-$ROOT/processed_data/triplets/val_valid.txt}"
LR="${LR:-1e-5}"
MAX_EPOCHS="${MAX_EPOCHS:-100}"
BETA_WARMUP="${BETA_WARMUP:-16000}"
BETA_MIN="${BETA_MIN:-0.01}"
SELECTION_METRIC="${SELECTION_METRIC:-ligand_lift_contact_switch_rotamer_acc}"

cat <<EOF
==============================================
Stage-1 Phase-1 Residual: stable fullscratch ligand-causal training (16-GPU)
==============================================
Job ID:           ${SLURM_JOB_ID:-NA}
Nodes:            ${SLURM_JOB_NODELIST:-NA}
Master:           ${MASTER_ADDR}:${MASTER_PORT}
World size:       ${WORLD_SIZE} (${NNODES} nodes × ${NPROC_PER_NODE} A100)
Tag:              $TAG
LR:               $LR
Max epochs:       $MAX_EPOCHS
Selection metric: $SELECTION_METRIC

Stability changes versus previous run:
  lr:                    5e-5 -> 1e-5
  base_prior CE:          0.5 -> 0.0
  switch_dir/amp/rank:    3/1/1 -> 1/0.3/0.3
  g_decoy:                2.0 -> 0.5
  g_zero_noncontact:      0.5 -> 0.2
  beta warmup:            8000 -> 16000, beta_min=0.01
  selection:              lift over base on contact+switch (decoy still logged/trained weakly)
Start: $(date)
==============================================
EOF

nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader || true
python -c "import torch; print(f'PyTorch {torch.__version__}, CUDA {torch.version.cuda}, local GPUs: {torch.cuda.device_count()}')"

srun --ntasks="$NNODES" --ntasks-per-node=1 bash -c '
  set -euo pipefail
  cd '"$ROOT"'
  source /mnt/inaisfs/data/home/zhaozc_criait/miniconda3/etc/profile.d/conda.sh
  conda activate BINDRAE
  echo "[node ${SLURM_NODEID}] $(hostname) starting torchrun at $(date)"
  '"$ENV_PREFIX"'/bin/torchrun \
    --nnodes='"$NNODES"' \
    --nproc_per_node='"$NPROC_PER_NODE"' \
    --node_rank=${SLURM_NODEID} \
    --rdzv_id='"$SLURM_JOB_ID"' \
    --rdzv_backend=c10d \
    --rdzv_endpoint='"$MASTER_ADDR:$MASTER_PORT"' \
    scripts/train_stage1.py \
    --model_size medium \
    --data_dir processed_data/triplets \
    --valid_samples_file '"$TRAIN_TXT"' \
    --val_samples_file '"$VAL_TXT"' \
    --sample_metadata_file sample_metadata.json \
    --batch_size 2 \
    --num_workers 2 \
    --max_n_res 900 \
    --length_bucketed_sampling \
    --residue_budget 900 \
    --lr '"$LR"' \
    --warmup_steps 2000 \
    --lr_scheduler plateau \
    --plateau_factor 0.5 \
    --plateau_patience 5 \
    --min_lr_scale 0.1 \
    --pocket_warmup_steps 2000 \
    --ligand_gate_warmup_steps 2000 \
    --max_epochs '"$MAX_EPOCHS"' \
    --grad_clip 0.5 \
    --patience 20 \
    --w_fape 1.0 \
    --w_chi 1.0 \
    --w_clash 0.1 \
    --lambda_chi1_rotamer 0.0 \
    --lambda_contact 0.0 \
    --lambda_candidate_chi1 0.0 \
    --lambda_ligand_contrastive 0.0 \
    --lambda_geometry_chi1 0.5 \
    --lambda_base_prior 0.0 \
    --lambda_switch_bce 0.0 \
    --lambda_rescue_noharm 0.0 \
    --lambda_g_lift_switch 0.0 \
    --lambda_g_noharm 0.0 \
    --lambda_g_zero_noncontact 0.2 \
    --g_noncontact_threshold 8.0 \
    --lambda_g_switch_dir 1.0 \
    --lambda_g_switch_amp 0.3 \
    --lambda_g_switch_rank 0.3 \
    --lambda_g_antiharm 0.2 \
    --lambda_g_decoy 0.5 \
    --g_switch_temperature 1.0 \
    --g_switch_amp_margin 0.05 \
    --g_switch_rank_margin 0.05 \
    --g_antiharm_tau 0.05 \
    --g_decoy_margin 0.02 \
    --g_decoy_kind shuffled \
    --geometry_scorer_lr_scale 1.0 \
    --geometry_scorer_use_sgeo \
    --geometry_scorer_sgeo_dim 32 \
    --geometry_scorer_bounded_residual \
    --geometry_scorer_residual_max 3.0 \
    --geometry_scorer_residual_tau 3.0 \
    --geometry_scorer_gate_norm \
    --geometry_scorer_gate_clamp 4.0 \
    --geometry_scorer_gate_init_bias 0.0 \
    --detach_base_for_residual \
    --gate_warmup_open_steps 2000 \
    --residual_beta_warmup_steps '"$BETA_WARMUP"' \
    --residual_beta_min '"$BETA_MIN"' \
    --selection_metric '"$SELECTION_METRIC"' \
    --ddp_find_unused_parameters \
    --save_epoch_checkpoints \
    --save_dir checkpoints/stage1/'"$TAG"' \
    --log_dir logs/stage1/'"$TAG"'
'

echo ""
echo "=============================================="
echo "Stable 16-GPU fullscratch ligand-causal Phase-1 training completed: $(date)"
echo "Checkpoint: checkpoints/stage1/$TAG"
echo "Metrics:    logs/stage1/$TAG/metrics.jsonl"
echo "=============================================="
