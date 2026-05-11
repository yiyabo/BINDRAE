#!/bin/bash
#SBATCH --job-name=s1_ablation_bp
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:A100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=logs/slurm/ablation_baseprior_%j.out
#SBATCH --error=logs/slurm/ablation_baseprior_%j.err

set -e
unset LD_PRELOAD
unset PROXYCHAINS_CONF_FILE
unset PROXYCHAINS_QUIET_MODE

export PATH=/mnt/inaisfs/data/home/zhaozc_criait/miniconda3/bin:$PATH
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=4

source /mnt/inaisfs/data/home/zhaozc_criait/miniconda3/etc/profile.d/conda.sh
conda activate BINDRAE

cd /mnt/inaisfs/data/home/zhaozc_criait/XinxiangWang/BINDRAE
mkdir -p logs/slurm logs/stage1_diagnostics

CHECKPOINT="${CHECKPOINT:-checkpoints/stage1/geom_baseprior_20260501_170858/best_model.pt}"
MAX_BATCHES="${MAX_BATCHES:-}"
TAG="ablation_baseprior_$(date +%Y%m%d_%H%M%S)"
OUT_DIR="logs/stage1_diagnostics/${TAG}"

MAX_BATCH_ARGS=()
if [[ -n "$MAX_BATCHES" ]]; then
  MAX_BATCH_ARGS=(--max_batches "$MAX_BATCHES")
fi

echo "=============================================="
echo "Stage-1 Base Prior Decomposition Ablation"
echo "=============================================="
echo "Job ID:      $SLURM_JOB_ID"
echo "Node:        $SLURM_NODELIST"
echo "Checkpoint:  $CHECKPOINT"
echo "Max batches: ${MAX_BATCHES:-all}"
echo "Output dir:  $OUT_DIR"
echo "Start:       $(date)"
echo "=============================================="

python scripts/diagnose_stage1_prior.py \
  --checkpoint "$CHECKPOINT" \
  --data_dir processed_data/triplets \
  --val_samples_file processed_data/triplets/val_valid.txt \
  --sample_metadata_file sample_metadata.json \
  --batch_size 4 \
  --num_workers 2 \
  --max_n_res 900 \
  "${MAX_BATCH_ARGS[@]}" \
  --device cuda \
  --output_dir "$OUT_DIR" \
  --output_json "$OUT_DIR/ablation_baseprior.json" \
  --posterior_diagnostics \
  --decomposition

echo ""
echo "=============================================="
echo "Ablation completed: $(date)"
echo "Results: $OUT_DIR/ablation_baseprior.json"
echo "=============================================="
