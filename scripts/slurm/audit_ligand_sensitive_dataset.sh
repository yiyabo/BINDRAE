#!/bin/bash
#SBATCH --job-name=s1_ligdata_audit
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:A100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=logs/slurm/stage1_ligdata_audit_%j.out
#SBATCH --error=logs/slurm/stage1_ligdata_audit_%j.err

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

TAG="${TAG:-ligand_sensitive_dataset_audit_$(date +%Y%m%d_%H%M%S)}"
OUT_DIR="${OUT_DIR:-logs/stage1_diagnostics/${TAG}}"
MAX_BATCHES="${MAX_BATCHES:-}"
SPLIT="${SPLIT:-val}"
VALID_SAMPLES_FILE="${VALID_SAMPLES_FILE:-processed_data/triplets/val_valid.txt}"
CA_CONTACT_THRESHOLD="${CA_CONTACT_THRESHOLD:-8.0}"
ATOM_CONTACT_THRESHOLD="${ATOM_CONTACT_THRESHOLD:-4.5}"
CONTACT_THRESHOLDS="${CONTACT_THRESHOLDS:-3.5,4.0,4.5,5.0}"

echo "=============================================="
echo "Stage-1 ligand-sensitive dataset audit"
echo "=============================================="
echo "Job ID:      ${SLURM_JOB_ID:-NA}"
echo "Node:        ${SLURM_NODELIST:-NA}"
echo "Split:       $SPLIT"
echo "Valid file:  $VALID_SAMPLES_FILE"
echo "Output dir:  $OUT_DIR"
echo "Max batches: ${MAX_BATCHES:-full split}"
echo "CA contact:  $CA_CONTACT_THRESHOLD"
echo "Atom contact:$ATOM_CONTACT_THRESHOLD"
echo "Audit cuts:  $CONTACT_THRESHOLDS"
echo "Start:       $(date)"
echo "=============================================="

CMD=(
  python scripts/audit_ligand_sensitive_dataset.py
  --data_dir processed_data/triplets
  --split "$SPLIT"
  --valid_samples_file "$VALID_SAMPLES_FILE"
  --sample_metadata_file sample_metadata.json
  --batch_size 8
  --num_workers 2
  --max_n_res 900
  --ca_contact_threshold "$CA_CONTACT_THRESHOLD"
  --atom_contact_threshold "$ATOM_CONTACT_THRESHOLD"
  --contact_thresholds "$CONTACT_THRESHOLDS"
  --output_dir "$OUT_DIR"
)

if [[ -n "$MAX_BATCHES" ]]; then
  CMD+=(--max_batches "$MAX_BATCHES")
fi

"${CMD[@]}"

echo ""
echo "=============================================="
echo "Dataset audit completed: $(date)"
echo "JSON: $OUT_DIR/ligand_sensitive_dataset_audit.json"
echo "CSV:  $OUT_DIR/ligand_sensitive_dataset_audit_per_sample.csv"
echo "=============================================="
