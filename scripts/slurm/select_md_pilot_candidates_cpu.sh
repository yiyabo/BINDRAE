#!/usr/bin/env bash
# Screen AHoJ endpoint pairs for a small OpenMM setup/minimization pilot.

#SBATCH --job-name=mdpilot_select
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=96G
#SBATCH --time=03:00:00
#SBATCH --output=logs/slurm/md_pilot_select_%j.out
#SBATCH --error=logs/slurm/md_pilot_select_%j.err

set -euo pipefail

ROOT="${ROOT:-/mnt/inaisfs/data/home/zhaozc_criait/XinxiangWang/BINDRAE}"
SAMPLE_LIST="${SAMPLE_LIST:?Set SAMPLE_LIST to a Stage-2 sample-ID manifest}"
OUTPUT_DIR="${OUTPUT_DIR:-processed_data/md_transition/pilot_selection_${SLURM_JOB_ID}}"
SCAN_LIMIT="${SCAN_LIMIT:-8000}"
SELECT_COUNT="${SELECT_COUNT:-16}"
SEED="${SEED:-20260713}"
WORKERS="${WORKERS:-20}"

cd "$ROOT"
mkdir -p logs/slurm "$OUTPUT_DIR"

source /mnt/inaisfs/data/home/zhaozc_criait/miniconda3/etc/profile.d/conda.sh
conda activate BINDRAE-MD

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

echo "Sample list:  $SAMPLE_LIST"
echo "Output dir:   $OUTPUT_DIR"
echo "Scan limit:   $SCAN_LIMIT"
echo "Select count: $SELECT_COUNT"
echo "Workers:      $WORKERS"

python scripts/select_md_pilot_candidates.py \
  --data-dir processed_data/triplets \
  --sample-list "$SAMPLE_LIST" \
  --output-dir "$OUTPUT_DIR" \
  --scan-limit "$SCAN_LIMIT" \
  --select-count "$SELECT_COUNT" \
  --seed "$SEED" \
  --workers "$WORKERS"

python scripts/audit_md_transition_manifest.py \
  --manifest "$OUTPUT_DIR/selected_transition_manifest.jsonl" \
  --base-dir "$ROOT" \
  --check-files \
  --summary-output "$OUTPUT_DIR/manifest_audit.json"
