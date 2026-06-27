#!/bin/bash
#SBATCH --job-name=s1_diag_dz
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:A100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=logs/slurm/stage1_diag_dz_%j.out
#SBATCH --error=logs/slurm/stage1_diag_dz_%j.err

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

CHECKPOINT="${CHECKPOINT:-checkpoints/stage1/change_fast_1gpu_20260616_164925/best_model.pt}"
TAG="${TAG:-$(basename "$(dirname "$CHECKPOINT")")_$(basename "$CHECKPOINT" .pt)}"
OUT_DIR="logs/stage1_diagnostics/${TAG}_$(date +%Y%m%d_%H%M%S)"
GATE_LAMBDA="${GATE_LAMBDA:-1.0}"
BATCH_SIZE="${BATCH_SIZE:-4}"
MAX_BATCHES="${MAX_BATCHES:-100}"
NUM_WORKERS="${NUM_WORKERS:-2}"
EXTRA_ARGS=()
if [[ -n "${STAGE1_ENCODER_CHECKPOINT:-}" ]]; then
  EXTRA_ARGS+=(--stage1_encoder_checkpoint "$STAGE1_ENCODER_CHECKPOINT")
fi

echo "=============================================="
echo "Stage-1 DeltaZPredictor Ligand Sensitivity Diagnostic"
echo "=============================================="
echo "Job ID:      $SLURM_JOB_ID"
echo "Node:        $SLURM_NODELIST"
echo "Checkpoint:  $CHECKPOINT"
echo "Gate lambda: $GATE_LAMBDA"
echo "Batch size:  $BATCH_SIZE"
echo "Max batches: $MAX_BATCHES"
echo "Stage1 encoder checkpoint: ${STAGE1_ENCODER_CHECKPOINT:-<none>}"
echo "Output dir:  $OUT_DIR"
echo "Start:       $(date)"
echo "=============================================="

python scripts/diagnose_delta_z_predictor.py \
  --data_dir processed_data/triplets \
  --checkpoint "$CHECKPOINT" \
  --val_samples_file /mnt/inaisfs/data/home/zhaozc_criait/XinxiangWang/BINDRAE/processed_data/triplets/val_valid.txt \
  --sample_metadata_file sample_metadata.json \
  --batch_size "$BATCH_SIZE" \
  --max_n_res 1600 \
	  --max_batches "$MAX_BATCHES" \
	  --gate_lambda "$GATE_LAMBDA" \
	  --output_json "$OUT_DIR/delta_z_sensitivity.json" \
	  --num_workers "$NUM_WORKERS" \
	  --device cuda \
	  "${EXTRA_ARGS[@]}"

echo ""
echo "=============================================="
echo "Diagnostic completed: $(date)"
echo "Saved to: $OUT_DIR/delta_z_sensitivity.json"
echo "=============================================="
