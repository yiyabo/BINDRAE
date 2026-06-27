#!/bin/bash
#SBATCH --job-name=s1_rotoracle
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2
#SBATCH --gres=gpu:A100:2
#SBATCH --cpus-per-task=8
#SBATCH --mem=160G
#SBATCH --time=04:00:00
#SBATCH --output=logs/slurm/stage1_rotamer_oracle_%j.out
#SBATCH --error=logs/slurm/stage1_rotamer_oracle_%j.err

set -euo pipefail
unset LD_PRELOAD
unset PROXYCHAINS_CONF_FILE
unset PROXYCHAINS_QUIET_MODE

ROOT=/mnt/inaisfs/data/home/zhaozc_criait/XinxiangWang/BINDRAE
ENV_PREFIX=/mnt/inaisfs/data/home/zhaozc_criait/miniconda3/envs/BINDRAE

export PATH=/data/soft/slurm/24.11.4/bin:$ENV_PREFIX/bin:$PATH
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=4
export DDP_TIMEOUT="${DDP_TIMEOUT:-7200}"

cd "$ROOT"
mkdir -p logs/slurm logs/stage1_diagnostics

source /mnt/inaisfs/data/home/zhaozc_criait/miniconda3/etc/profile.d/conda.sh
conda activate BINDRAE

TAG="${TAG:-rotamer_oracle_signal_$(date +%Y%m%d_%H%M%S)}"
OUT_DIR="${OUT_DIR:-logs/stage1_diagnostics/${TAG}}"
SPLIT="${SPLIT:-val}"
VALID_SAMPLES_FILE="${VALID_SAMPLES_FILE:-processed_data/triplets/val_valid.txt}"
BATCH_SIZE="${BATCH_SIZE:-24}"
NUM_WORKERS="${NUM_WORKERS:-2}"
MAX_N_RES="${MAX_N_RES:-1600}"
MAX_LOCAL_RES="${MAX_LOCAL_RES:-192}"
MAX_SAMPLES="${MAX_SAMPLES:-0}"
MAX_BATCHES="${MAX_BATCHES:-0}"
RESIDUE_CHUNK="${RESIDUE_CHUNK:-64}"

echo "=============================================="
echo "Stage-1 rotamer oracle signal audit (2-GPU)"
echo "=============================================="
echo "Job ID:       ${SLURM_JOB_ID:-NA}"
echo "Node:         ${SLURM_NODELIST:-NA}"
echo "Split:        $SPLIT"
echo "Valid file:   $VALID_SAMPLES_FILE"
echo "Output dir:   $OUT_DIR"
echo "Batch/GPU:    $BATCH_SIZE"
echo "Max samples:  $MAX_SAMPLES"
echo "Max batches:  $MAX_BATCHES"
echo "Max local res:$MAX_LOCAL_RES"
echo "Start:        $(date)"
echo "=============================================="

torchrun --nproc_per_node=2 scripts/audit_rotamer_oracle_signal.py \
  --data_dir processed_data/triplets \
  --split "$SPLIT" \
  --valid_samples_file "$VALID_SAMPLES_FILE" \
  --sample_metadata_file sample_metadata.json \
  --batch_size "$BATCH_SIZE" \
  --num_workers "$NUM_WORKERS" \
  --max_n_res "$MAX_N_RES" \
  --max_local_res "$MAX_LOCAL_RES" \
  --max_samples "$MAX_SAMPLES" \
  --max_batches "$MAX_BATCHES" \
  --residue_chunk "$RESIDUE_CHUNK" \
  --output_dir "$OUT_DIR" \
  --device cuda \
  --distributed

echo ""
echo "=============================================="
echo "Rotamer oracle audit completed: $(date)"
echo "JSON: $OUT_DIR/rotamer_oracle_signal.json"
echo "CSV:  $OUT_DIR/rotamer_oracle_signal_per_sample.csv"
echo "=============================================="
