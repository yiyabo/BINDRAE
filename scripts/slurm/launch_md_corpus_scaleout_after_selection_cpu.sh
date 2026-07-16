#!/usr/bin/env bash
# Launch a resumable context -> replica scale-out after candidate selection.

#SBATCH --job-name=mdscale_launch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=logs/slurm/md_scaleout_launch_%j.out
#SBATCH --error=logs/slurm/md_scaleout_launch_%j.err

set -euo pipefail

ROOT="${ROOT:-/mnt/inaisfs/data/home/zhaozc_criait/XinxiangWang/BINDRAE}"
SELECTION_DIR="${SELECTION_DIR:?Set SELECTION_DIR to the completed selection output}"
EXCLUDE_SAMPLE_LIST="${EXCLUDE_SAMPLE_LIST:?Set EXCLUDE_SAMPLE_LIST to previously attempted sample IDs}"
CONTEXT_DIR="${CONTEXT_DIR:?Set CONTEXT_DIR for the new context matrix and systems}"
REPLICA_DIR="${REPLICA_DIR:?Set REPLICA_DIR for accepted-system replicas}"
CONTEXT_MAX_CONCURRENT="${CONTEXT_MAX_CONCURRENT:-16}"
REPLICA_MAX_CONCURRENT="${REPLICA_MAX_CONCURRENT:-16}"
CONTEXT_SEED_BASE="${CONTEXT_SEED_BASE:-60719000}"
REPLICA_SEED_BASE="${REPLICA_SEED_BASE:-60720000}"

export PATH="/data/soft/slurm/24.11.4/bin:${PATH}"
cd "$ROOT"
mkdir -p logs/slurm "$CONTEXT_DIR" "$REPLICA_DIR"

source /mnt/inaisfs/data/home/zhaozc_criait/miniconda3/etc/profile.d/conda.sh
conda activate BINDRAE-MD
export PYTHONUNBUFFERED=1

CANDIDATE_MANIFEST="$SELECTION_DIR/selected_transition_manifest.jsonl"
CONTEXT_MATRIX="$CONTEXT_DIR/context_matrix.jsonl"
COLLECTION_DIR="$CONTEXT_DIR/collection"
SUBMISSION_RECORD="$CONTEXT_DIR/submission.json"

if [[ ! -s "$CANDIDATE_MANIFEST" ]]; then
  echo "Selection manifest is missing or empty: $CANDIDATE_MANIFEST" >&2
  exit 2
fi
if [[ ! -s "$EXCLUDE_SAMPLE_LIST" ]]; then
  echo "Exclusion sample list is missing or empty: $EXCLUDE_SAMPLE_LIST" >&2
  exit 2
fi
if [[ -e "$SUBMISSION_RECORD" ]]; then
  echo "Submission record already exists; refusing duplicate scale-out:" >&2
  cat "$SUBMISSION_RECORD" >&2
  exit 3
fi

python scripts/build_md_context_matrix.py \
  --candidate-manifest "$CANDIDATE_MANIFEST" \
  --exclude-sample-list "$EXCLUDE_SAMPLE_LIST" \
  --output-dir "$CONTEXT_DIR" \
  --seed-base "$CONTEXT_SEED_BASE" \
  --protocol-tag endpoint_context_fixed_v1

TASKS=$(wc -l < "$CONTEXT_MATRIX" | tr -d ' ')
if [[ "$TASKS" -le 0 ]]; then
  echo "Context matrix is empty: $CONTEXT_MATRIX" >&2
  exit 4
fi
LAST_INDEX=$((TASKS - 1))

CONTEXT_JOB=$(sbatch --parsable \
  --job-name=mdctx_scale255 \
  --array="0-${LAST_INDEX}%${CONTEXT_MAX_CONCURRENT}" \
  --export=ALL,MATRIX="$CONTEXT_MATRIX",PLATFORM=CPU \
  scripts/slurm/run_md_context_pipeline_array_cpu.sh)

CONTINUATION_JOB=$(sbatch --parsable \
  --dependency="afterany:${CONTEXT_JOB}" \
  --export=ALL,CONTEXT_MATRIX="$CONTEXT_MATRIX",CANDIDATE_MANIFEST="$CANDIDATE_MANIFEST",COLLECTION_DIR="$COLLECTION_DIR",REPLICA_OUTPUT_DIR="$REPLICA_DIR",MAX_CONCURRENT="$REPLICA_MAX_CONCURRENT",SEED_BASE="$REPLICA_SEED_BASE" \
  scripts/slurm/continue_md_pilot_after_context_cpu.sh)

python - "$SUBMISSION_RECORD" "$TASKS" "$CONTEXT_JOB" "$CONTINUATION_JOB" <<'PY'
import json
import os
import sys
from pathlib import Path

path, tasks, context_job, continuation_job = sys.argv[1:]
record = {
    "schema_version": "bindrae_md_scaleout_submission_v1",
    "launcher_job_id": os.environ.get("SLURM_JOB_ID"),
    "context_tasks": int(tasks),
    "context_job_id": context_job,
    "continuation_job_id": continuation_job,
}
target = Path(path)
temporary = target.with_suffix(target.suffix + ".tmp")
temporary.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
temporary.replace(target)
print(json.dumps(record, indent=2, sort_keys=True))
PY
