#!/bin/bash
#SBATCH --job-name=s1_prior_diag
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:A100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=logs/slurm/stage1_prior_diag_%j.out
#SBATCH --error=logs/slurm/stage1_prior_diag_%j.err

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

CHECKPOINT="${CHECKPOINT:-checkpoints/stage1/opt8e2_20260411_122707/epoch_018.pt}"
MAX_BATCHES="${MAX_BATCHES:-64}"
POSTERIOR_DIAGNOSTICS="${POSTERIOR_DIAGNOSTICS:-0}"
CALIBRATION_BINS="${CALIBRATION_BINS:-10}"
TAG="${TAG:-$(basename "$(dirname "$CHECKPOINT")")_$(basename "$CHECKPOINT" .pt)}"
OUT_DIR="logs/stage1_diagnostics/${TAG}_$(date +%Y%m%d_%H%M%S)"
POSTERIOR_ARGS=()
if [[ "$POSTERIOR_DIAGNOSTICS" == "1" ]]; then
  POSTERIOR_ARGS+=(--posterior_diagnostics --calibration_bins "$CALIBRATION_BINS")
fi

echo "=============================================="
echo "Stage-1 prior diagnostic"
echo "=============================================="
echo "Job ID:      $SLURM_JOB_ID"
echo "Node:        $SLURM_NODELIST"
echo "Checkpoint:  $CHECKPOINT"
echo "Max batches: $MAX_BATCHES"
echo "Posterior:   $POSTERIOR_DIAGNOSTICS"
echo "Output dir:  $OUT_DIR"
echo "Start:       $(date)"
echo "=============================================="

python scripts/diagnose_stage1_prior.py \
  --checkpoint "$CHECKPOINT" \
  --data_dir processed_data/triplets \
  --val_samples_file /mnt/inaisfs/data/home/zhaozc_criait/XinxiangWang/BINDRAE/processed_data/triplets/val_valid.txt \
  --sample_metadata_file sample_metadata.json \
  --batch_size 4 \
  --num_workers 2 \
  --max_n_res 1600 \
  --max_batches "$MAX_BATCHES" \
  --device cuda \
  --output_dir "$OUT_DIR" \
  --output_json "$OUT_DIR/stage1_prior_diagnostics.json" \
  "${POSTERIOR_ARGS[@]}"

echo ""
echo "=============================================="
echo "Diagnostic completed: $(date)"
echo "Saved to: $OUT_DIR/stage1_prior_diagnostics.json"
echo "=============================================="
