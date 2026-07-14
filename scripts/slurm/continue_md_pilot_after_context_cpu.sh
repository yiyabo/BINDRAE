#!/usr/bin/env bash
# Collect passed contexts, then submit five independent silver replicas per system.

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
  --replica-start 0 \
  --replica-stop 4 \
  --seed-base "$SEED_BASE" \
  --protocol-tag global_ca_rmsd_fixed_v1 \
  --pre-equilibration-steps 500 \
  --pulling-steps 10000 \
  --endpoint-hold-steps 2000 \
  --report-interval 100 \
  --rmsd-k-kj-mol-nm2 200000 \
  --final-target-rmsd-nm 0.025

MATRIX="$REPLICA_OUTPUT_DIR/replica_matrix.jsonl"
TASKS=$(wc -l < "$MATRIX" | tr -d ' ')
LAST_INDEX=$((TASKS - 1))
REPLICA_JOB=$(sbatch --parsable \
  --job-name=mdrep_pilot12 \
  --array="0-${LAST_INDEX}%${MAX_CONCURRENT}" \
  --export=ALL,MATRIX="$MATRIX",PLATFORM=CPU \
  scripts/slurm/run_md_replica_pipeline_array_cpu.sh)

FINALIZATION_DIR="$REPLICA_OUTPUT_DIR/finalization"
FINALIZE_JOB=$(sbatch --parsable \
  --dependency="afterany:${REPLICA_JOB}" \
  --export=ALL,MATRIX="$MATRIX",OUTPUT_DIR="$FINALIZATION_DIR" \
  scripts/slurm/finalize_md_replica_matrix_cpu.sh)

python3 - "$SUBMISSION_RECORD" "$SYSTEMS" "$TASKS" "$REPLICA_JOB" "$FINALIZE_JOB" <<'PY'
import json, sys
path, systems, tasks, replica_job, finalize_job = sys.argv[1:]
with open(path, "w") as handle:
    json.dump(
        {
            "passed_context_systems": int(systems),
            "replica_tasks": int(tasks),
            "replica_job_id": replica_job,
            "finalize_job_id": finalize_job,
        },
        handle,
        indent=2,
        sort_keys=True,
    )
    handle.write("\n")
PY

cat "$SUBMISSION_RECORD"
