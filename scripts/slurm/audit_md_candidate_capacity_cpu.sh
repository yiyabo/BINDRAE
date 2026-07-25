#!/usr/bin/env bash
# Audit novel MD-candidate capacity after frozen holdout and prior-pair exclusion.

#SBATCH --job-name=path3_cap_audit
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=logs/slurm/path3_capacity_audit_%j.out
#SBATCH --error=logs/slurm/path3_capacity_audit_%j.err

set -euo pipefail

ROOT="${ROOT:-/mnt/inaisfs/data/home/zhaozc_criait/XinxiangWang/BINDRAE}"
SELECTION_DIR="${SELECTION_DIR:?Set SELECTION_DIR to the mapping-aware selection output}"
VAL_SAMPLE_LIST="${VAL_SAMPLE_LIST:?Set VAL_SAMPLE_LIST to the frozen validation IDs}"
TEST_SAMPLE_LIST="${TEST_SAMPLE_LIST:?Set TEST_SAMPLE_LIST to the frozen test IDs}"
ATTEMPTED_CANDIDATE_MANIFESTS="${ATTEMPTED_CANDIDATE_MANIFESTS:?Set colon-separated prior candidate manifests}"
OUTPUT_DIR="${OUTPUT_DIR:-processed_data/md_transition/path3_capacity_audit_${SLURM_JOB_ID}}"
METADATA_CACHE="${METADATA_CACHE:-$OUTPUT_DIR/endpoint_metadata.jsonl}"
WORKERS="${WORKERS:-24}"
MINIMUM_PLANNING_POOL="${MINIMUM_PLANNING_POOL:-3000}"
MISMATCH_SMOKE_COUNT="${MISMATCH_SMOKE_COUNT:-4}"

cd "$ROOT"
mkdir -p logs/slurm "$OUTPUT_DIR"

source /mnt/inaisfs/data/home/zhaozc_criait/miniconda3/etc/profile.d/conda.sh
conda activate BINDRAE

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

CANDIDATE_MANIFEST="$SELECTION_DIR/selected_transition_manifest.jsonl"
CANDIDATE_SAMPLE_LIST="$SELECTION_DIR/selected_sample_ids.txt"
CLEAN_SAMPLE_LIST="$OUTPUT_DIR/holdout_leakage_clean_sample_ids.txt"
LEAKAGE_REPORT="$OUTPUT_DIR/holdout_leakage_report.json"

for path in \
  "$CANDIDATE_MANIFEST" \
  "$CANDIDATE_SAMPLE_LIST" \
  "$VAL_SAMPLE_LIST" \
  "$TEST_SAMPLE_LIST"; do
  if [[ ! -s "$path" ]]; then
    echo "Required input is missing or empty: $path" >&2
    exit 2
  fi
done

IFS=':' read -r -a ATTEMPTED_PATHS <<< "$ATTEMPTED_CANDIDATE_MANIFESTS"
if [[ "${#ATTEMPTED_PATHS[@]}" -eq 0 ]]; then
  echo "No attempted candidate manifests were provided" >&2
  exit 2
fi

AUDIT_ARGS=()
for path in "${ATTEMPTED_PATHS[@]}"; do
  if [[ ! -s "$path" ]]; then
    echo "Attempted candidate manifest is missing or empty: $path" >&2
    exit 2
  fi
  AUDIT_ARGS+=(--attempted-candidate-manifest "$path")
done

python scripts/build_stage2_leakage_clean_subset.py \
  --input-sample-list "$CANDIDATE_SAMPLE_LIST" \
  --holdout-sample-list "$VAL_SAMPLE_LIST" \
  --holdout-sample-list "$TEST_SAMPLE_LIST" \
  --samples-dir processed_data/triplets/samples \
  --output-sample-list "$CLEAN_SAMPLE_LIST" \
  --report "$LEAKAGE_REPORT" \
  --metadata-cache "$METADATA_CACHE" \
  --sequence-identity 0.30 \
  --sequence-coverage 0.80 \
  --workers "$WORKERS"

python scripts/audit_md_candidate_capacity.py \
  --candidate-manifest "$CANDIDATE_MANIFEST" \
  --leakage-clean-sample-list "$CLEAN_SAMPLE_LIST" \
  --leakage-report "$LEAKAGE_REPORT" \
  "${AUDIT_ARGS[@]}" \
  --output-dir "$OUTPUT_DIR" \
  --minimum-planning-pool "$MINIMUM_PLANNING_POOL" \
  --mismatch-smoke-count "$MISMATCH_SMOKE_COUNT"

python scripts/audit_md_transition_manifest.py \
  --manifest "$OUTPUT_DIR/novel_transition_manifest.jsonl" \
  --base-dir "$ROOT" \
  --check-files \
  --summary-output "$OUTPUT_DIR/novel_manifest_audit.json"
