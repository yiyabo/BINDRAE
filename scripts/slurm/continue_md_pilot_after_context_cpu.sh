#!/usr/bin/env bash
# Collect passed contexts, then submit a declared range of silver replicas.

#SBATCH --job-name=mdctx_continue
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=logs/slurm/md_context_continue_%j.out
#SBATCH --error=logs/slurm/md_context_continue_%j.err

set -euo pipefail

ROOT="${ROOT:-/mnt/inaisfs/data/home/zhaozc_criait/XinxiangWang/BINDRAE}"
CONTEXT_MATRIX="${CONTEXT_MATRIX:?Set CONTEXT_MATRIX}"
CANDIDATE_MANIFEST="${CANDIDATE_MANIFEST:?Set CANDIDATE_MANIFEST}"
COLLECTION_DIR="${COLLECTION_DIR:?Set COLLECTION_DIR}"
REPLICA_OUTPUT_DIR="${REPLICA_OUTPUT_DIR:?Set REPLICA_OUTPUT_DIR}"
MAX_CONCURRENT="${MAX_CONCURRENT:-4}"
SEED_BASE="${SEED_BASE:-60716000}"
PROTOCOL_TAG="${PROTOCOL_TAG:-global_ca_rmsd_fixed_v1}"
PRE_EQUILIBRATION_STEPS="${PRE_EQUILIBRATION_STEPS:-500}"
PULLING_STEPS="${PULLING_STEPS:-10000}"
ENDPOINT_HOLD_STEPS="${ENDPOINT_HOLD_STEPS:-2000}"
REPORT_INTERVAL="${REPORT_INTERVAL:-100}"
RMSD_K_KJ_MOL_NM2="${RMSD_K_KJ_MOL_NM2:-200000}"
FINAL_TARGET_RMSD_NM="${FINAL_TARGET_RMSD_NM:-0.025}"
MIN_MAPPING_FRACTION="${MIN_MAPPING_FRACTION:-0.95}"
REPLICA_START="${REPLICA_START:-0}"
REPLICA_STOP="${REPLICA_STOP:-4}"
REPLICA_JOB_NAME="${REPLICA_JOB_NAME:-mdrep_pilot12}"

for value_name in REPLICA_START REPLICA_STOP; do
  value="${!value_name}"
  if ! [[ "$value" =~ ^[0-9]+$ ]]; then
    echo "$value_name must be a non-negative integer; got $value" >&2
    exit 2
  fi
done
if (( REPLICA_STOP < REPLICA_START )); then
  echo "REPLICA_STOP must be >= REPLICA_START" >&2
  exit 2
fi

export PATH=/data/soft/slurm/24.11.4/bin:$PATH
cd "$ROOT"
mkdir -p logs/slurm "$COLLECTION_DIR" "$REPLICA_OUTPUT_DIR"

COLLECTED_MANIFEST="$COLLECTION_DIR/passed_contexts.jsonl"
COLLECTION_SUMMARY="$COLLECTION_DIR/summary.json"
SUBMISSION_RECORD="$REPLICA_OUTPUT_DIR/submission.json"

python3 scripts/collect_md_context_results.py \
  --matrix "$CONTEXT_MATRIX" \
  --output-manifest "$COLLECTED_MANIFEST" \
  --summary "$COLLECTION_SUMMARY"

SYSTEMS=$(wc -l < "$COLLECTED_MANIFEST" | tr -d ' ')
if [[ "$SYSTEMS" -eq 0 ]]; then
  echo "No contexts passed; no replica array submitted."
  exit 0
fi
if [[ -f "$SUBMISSION_RECORD" ]]; then
  echo "Submission record already exists; refusing duplicate submission: $SUBMISSION_RECORD"
  cat "$SUBMISSION_RECORD"
  exit 0
fi

python3 scripts/build_md_replica_matrix.py \
  --candidate-manifest "$CANDIDATE_MANIFEST" \
  --context-manifest "$COLLECTED_MANIFEST" \
  --output-dir "$REPLICA_OUTPUT_DIR" \
  --replica-start "$REPLICA_START" \
  --replica-stop "$REPLICA_STOP" \
  --seed-base "$SEED_BASE" \
  --protocol-tag "$PROTOCOL_TAG" \
  --pre-equilibration-steps "$PRE_EQUILIBRATION_STEPS" \
  --pulling-steps "$PULLING_STEPS" \
  --endpoint-hold-steps "$ENDPOINT_HOLD_STEPS" \
  --report-interval "$REPORT_INTERVAL" \
  --rmsd-k-kj-mol-nm2 "$RMSD_K_KJ_MOL_NM2" \
  --final-target-rmsd-nm "$FINAL_TARGET_RMSD_NM" \
  --min-mapping-fraction "$MIN_MAPPING_FRACTION"

MATRIX="$REPLICA_OUTPUT_DIR/replica_matrix.jsonl"
TASKS=$(wc -l < "$MATRIX" | tr -d ' ')
LAST_INDEX=$((TASKS - 1))
REPLICA_JOB=$(sbatch --parsable \
  --job-name="$REPLICA_JOB_NAME" \
  --array="0-${LAST_INDEX}%${MAX_CONCURRENT}" \
  --export=ALL,MATRIX="$MATRIX",PLATFORM=CPU \
  scripts/slurm/run_md_replica_pipeline_array_cpu.sh)

FINALIZATION_DIR="$REPLICA_OUTPUT_DIR/finalization"
FINALIZE_JOB=$(sbatch --parsable \
  --dependency="afterany:${REPLICA_JOB}" \
  --export=ALL,MATRIX="$MATRIX",OUTPUT_DIR="$FINALIZATION_DIR" \
  scripts/slurm/finalize_md_replica_matrix_cpu.sh)

python3 - "$SUBMISSION_RECORD" "$SYSTEMS" "$TASKS" "$REPLICA_JOB" "$FINALIZE_JOB" "$REPLICA_START" "$REPLICA_STOP" "$MIN_MAPPING_FRACTION" <<'PY'
import json, sys
path, systems, tasks, replica_job, finalize_job, replica_start, replica_stop, mapping = sys.argv[1:]
with open(path, "w") as handle:
    json.dump(
        {
            "passed_context_systems": int(systems),
            "replica_tasks": int(tasks),
            "replica_job_id": replica_job,
            "finalize_job_id": finalize_job,
            "replica_range": [int(replica_start), int(replica_stop)],
            "min_mapping_fraction": float(mapping),
        },
        handle,
        indent=2,
        sort_keys=True,
    )
    handle.write("\n")
PY

cat "$SUBMISSION_RECORD"
