#!/bin/bash
#SBATCH --job-name=s1_alpha_sweep
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:A100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=logs/slurm/forced_alpha_sweep_%j.out
#SBATCH --error=logs/slurm/forced_alpha_sweep_%j.err

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
TAG="forced_alpha_$(date +%Y%m%d_%H%M%S)"
OUT_DIR="logs/stage1_diagnostics/${TAG}"

MAX_BATCH_ARGS=()
if [[ -n "$MAX_BATCHES" ]]; then
  MAX_BATCH_ARGS=(--max_batches "$MAX_BATCHES")
fi

echo "=============================================="
echo "Forced Residual Alpha Sweep"
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
  --output_json "$OUT_DIR/forced_alpha_sweep.json" \
  --forced_alpha "0,1e-5,1e-4,3e-4,1e-3,3e-3,1e-2,3e-2,0.1,0.3,1.0"

echo ""
echo "=============================================="
echo "Sweep completed: $(date)"
echo "Results: $OUT_DIR/forced_alpha_sweep.json"
echo "=============================================="
