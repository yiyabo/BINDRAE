#!/bin/bash
#SBATCH --job-name=s1_postcand_diag
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:A100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=12:00:00
#SBATCH --output=logs/slurm/stage1_postcand_diag_%j.out
#SBATCH --error=logs/slurm/stage1_postcand_diag_%j.err

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
mkdir -p logs/slurm logs/stage1_diagnostics

source /mnt/inaisfs/data/home/zhaozc_criait/miniconda3/etc/profile.d/conda.sh
conda activate BINDRAE

CHECKPOINT="${CHECKPOINT:-checkpoints/stage1/posterior_candidate_ligcausal_screen4safe_20260509_184220/best_model.pt}"
TAG="${TAG:-postcand4safe_best_posterior_diag_$(date +%Y%m%d_%H%M%S)}"
OUT_DIR="${OUT_DIR:-logs/stage1_diagnostics/${TAG}}"
MAX_BATCHES="${MAX_BATCHES:-}"
CALIBRATION_BINS="${CALIBRATION_BINS:-15}"

echo "=============================================="
echo "Stage-1 posterior candidate diagnostic"
echo "=============================================="
echo "Job ID:      ${SLURM_JOB_ID:-NA}"
echo "Node:        ${SLURM_NODELIST:-NA}"
echo "Checkpoint:  $CHECKPOINT"
echo "Output dir:  $OUT_DIR"
echo "Max batches: ${MAX_BATCHES:-full validation}"
echo "Start:       $(date)"
echo "=============================================="

nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader

CMD=(
  python scripts/diagnose_stage1_prior.py
  --checkpoint "$CHECKPOINT"
  --data_dir processed_data/triplets
  --val_samples_file processed_data/triplets/val_valid.txt
  --sample_metadata_file sample_metadata.json
  --batch_size 2
  --num_workers 2
  --max_n_res 900
  --device cuda
  --output_dir "$OUT_DIR"
  --output_json "$OUT_DIR/stage1_prior_diagnostics.json"
  --posterior_diagnostics
  --calibration_bins "$CALIBRATION_BINS"
  --decomposition
  --trust_checkpoint_pickle
)

if [[ -n "$MAX_BATCHES" ]]; then
  CMD+=(--max_batches "$MAX_BATCHES")
fi

"${CMD[@]}"

echo ""
echo "=============================================="
echo "Posterior diagnostic completed: $(date)"
echo "JSON: $OUT_DIR/stage1_prior_diagnostics.json"
echo "=============================================="
