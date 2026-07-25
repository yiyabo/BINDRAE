#!/usr/bin/env bash
# Recover target export only for the seven APObind replicas that passed all physical gates.

set -euo pipefail
unset LD_PRELOAD

ROOT="${ROOT:-/mnt/inaisfs/data/home/zhaozc_criait/XinxiangWang/BINDRAE}"
OUTPUT_ROOT="${OUTPUT_ROOT:-processed_data/md_transition/apobind_replica_pilot8x2_20260725_v1}"
PLAN="${PLAN:-$OUTPUT_ROOT/pilot_plan.json}"
MATRIX="${MATRIX:-$OUTPUT_ROOT/replica_matrix.jsonl}"
PILOT_STATE="${PILOT_STATE:-$OUTPUT_ROOT/pilot_state.json}"
CANONICAL_DATA_DIR="${CANONICAL_DATA_DIR:-tmp/external_dataset_audit/apobind_triplets25_v1}"
CACHE_AUDIT="${CACHE_AUDIT:-$OUTPUT_ROOT/target_recovery_torsion_cache_audit.json}"
PROVENANCE_DIR="${PROVENANCE_DIR:-$OUTPUT_ROOT/target_recovery_provenance}"
FINALIZATION_DIR="${FINALIZATION_DIR:-$OUTPUT_ROOT/target_recovery_finalization}"
RECOVERY_STATE="${RECOVERY_STATE:-$OUTPUT_ROOT/target_recovery_state.json}"
LOG_ROOT="${LOG_ROOT:-logs/stage2/apobind_replica_pilot8x2_20260725_v1_target_recovery}"
CONDA_EXE="${CONDA_EXE:-/mnt/inaisfs/data/home/zhaozc_criait/miniconda3/bin/conda}"
MD_ENV_NAME="${MD_ENV_NAME:-BINDRAE-MD}"
RECOVERY_WORKERS="${RECOVERY_WORKERS:-4}"
EXPECTED_SYSTEMS="${EXPECTED_SYSTEMS:-8}"
EXPECTED_TASKS="${EXPECTED_TASKS:-16}"
RECOVERY_INDICES=(2 3 6 7 10 11 13)

if [[ "$(hostname -s)" != "gpu33" && "${ALLOW_NON_GPU33:-0}" != "1" ]]; then
  echo "ERROR: this direct recovery launcher is restricted to gpu33" >&2
  exit 2
fi
for value in "$RECOVERY_WORKERS" "$EXPECTED_SYSTEMS" "$EXPECTED_TASKS"; do
  if ! [[ "$value" =~ ^[1-9][0-9]*$ ]]; then
    echo "ERROR: expected a positive integer, got: $value" >&2
    exit 2
  fi
done
if [[ "$RECOVERY_WORKERS" -gt "${#RECOVERY_INDICES[@]}" ]]; then
  echo "ERROR: RECOVERY_WORKERS cannot exceed ${#RECOVERY_INDICES[@]}" >&2
  exit 2
fi

cd "$ROOT"
for path in "$PLAN" "$MATRIX" "$PILOT_STATE"; do
  if [[ ! -s "$path" ]]; then
    echo "ERROR: required recovery input is missing or empty: $path" >&2
    exit 2
  fi
done
for path in "$CACHE_AUDIT" "$PROVENANCE_DIR" "$FINALIZATION_DIR" "$RECOVERY_STATE" "$LOG_ROOT"; do
  if [[ -e "$path" ]]; then
    echo "ERROR: refusing to overwrite target-recovery artifact: $path" >&2
    exit 2
  fi
done

python3 - "$MATRIX" "$PILOT_STATE" "$PROVENANCE_DIR" \
  "$EXPECTED_SYSTEMS" "$EXPECTED_TASKS" "${RECOVERY_INDICES[@]}" <<'PY'
import hashlib
import json
import shutil
import sys
from pathlib import Path

matrix_path, pilot_state_path, provenance_dir = map(Path, sys.argv[1:4])
expected_systems, expected_tasks = map(int, sys.argv[4:6])
recovery_indices = [int(value) for value in sys.argv[6:]]
expected_indices = [2, 3, 6, 7, 10, 11, 13]
if recovery_indices != expected_indices:
    raise SystemExit(f"Recovery indices changed: {recovery_indices}")

rows = [json.loads(line) for line in matrix_path.read_text().splitlines() if line.strip()]
state = json.loads(pilot_state_path.read_text())
if len(rows) != expected_tasks or len({row["system_sample_id"] for row in rows}) != expected_systems:
    raise SystemExit("Replica matrix count mismatch")
if [row.get("matrix_index") for row in rows] != list(range(expected_tasks)):
    raise SystemExit("Replica matrix indices are not contiguous")
matrix_sha = hashlib.sha256(matrix_path.read_bytes()).hexdigest()
if state.get("matrix_sha256") != matrix_sha:
    raise SystemExit("Pilot state is not bound to the current replica matrix")
if state.get("status") != "complete":
    raise SystemExit(f"Pilot state is not complete: {state.get('status')}")
if state.get("outcome_counts") != {"failed_pull": 9, "failed_target_export": 7}:
    raise SystemExit(f"Unexpected frozen pilot outcomes: {state.get('outcome_counts')}")

def passed(path: Path, status: str) -> bool:
    if not path.is_file():
        return False
    record = json.loads(path.read_text())
    return record.get("status") == status and bool(record.get("passed", True))

snapshots = []
recovery_set = set(recovery_indices)
for row in rows:
    index = row["matrix_index"]
    pull_dir = Path(row["pull_dir"])
    target_dir = Path(row["target_dir"])
    pipeline_path = pull_dir / "pipeline_status.json"
    if not pipeline_path.is_file():
        raise SystemExit(f"Missing pipeline status for index {index}: {pipeline_path}")
    pipeline = json.loads(pipeline_path.read_text())
    if index in recovery_set:
        if pipeline.get("failed_stage") != "target_export":
            raise SystemExit(f"Index {index} is not a target-export failure")
        requirements = (
            (pull_dir / "rmsd_pull_report.json", "rmsd_pull_smoke_passed"),
            (pull_dir / "path_metrics_audit.json", "path_metrics_passed"),
            (pull_dir / "atomistic_path_audit.json", "atomistic_path_passed"),
        )
        for path, status in requirements:
            if not passed(path, status):
                raise SystemExit(f"Index {index} lacks passed prerequisite {path}")
        if passed(target_dir / "target_audit.json", "md_phase_normal_targets_passed"):
            raise SystemExit(f"Index {index} already has a passed target")
    elif pipeline.get("failed_stage") != "pull":
        raise SystemExit(f"Non-recovery index {index} is not a frozen pull rejection")

    if index in recovery_set:
        snapshots.append(
            {
                "matrix_index": index,
                "source": str(pipeline_path),
                "sha256": hashlib.sha256(pipeline_path.read_bytes()).hexdigest(),
            }
        )

provenance_dir.mkdir(parents=True)
for record in snapshots:
    destination = provenance_dir / f"pipeline_status_before_{record['matrix_index']:02d}.json"
    shutil.copy2(record["source"], destination)
    if hashlib.sha256(destination.read_bytes()).hexdigest() != record["sha256"]:
        raise SystemExit(f"Pipeline-status snapshot verification failed: {destination}")
    record["snapshot"] = str(destination)
(provenance_dir / "manifest.json").write_text(
    json.dumps(
        {
            "schema_version": "bindrae_apobind_target_recovery_provenance_v1",
            "matrix": str(matrix_path),
            "matrix_sha256": matrix_sha,
            "recovery_indices": recovery_indices,
            "snapshots": snapshots,
        },
        indent=2,
        sort_keys=True,
    )
    + "\n"
)
print(json.dumps({"recovery_indices": recovery_indices, "snapshots": len(snapshots)}))
PY

"$CONDA_EXE" run --no-capture-output -n "$MD_ENV_NAME" \
  python scripts/build_apobind_torsion_cache.py \
  --matrix "$MATRIX" \
  --data-dir "$CANONICAL_DATA_DIR" \
  --output-report "$CACHE_AUDIT" \
  --expected-systems "$EXPECTED_SYSTEMS" \
  --expected-replicas-per-system 2 \
  --min-pair-mapping-fraction 0.95

mkdir -p "$LOG_ROOT"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"

echo "APObind target-export-only recovery"
echo "Host:       $(hostname -s)"
echo "Indices:    ${RECOVERY_INDICES[*]}"
echo "Workers:    $RECOVERY_WORKERS"
echo "Matrix:     $MATRIX"
echo "Data root:  $CANONICAL_DATA_DIR"
echo "Cache audit:$CACHE_AUDIT"
echo "Logs:       $LOG_ROOT"
echo "Start:      $(date --iso-8601=seconds)"

worker_pids=()
for ((worker_index=0; worker_index<RECOVERY_WORKERS; worker_index++)); do
  (
    status_path="$LOG_ROOT/status_worker_${worker_index}.tsv"
    : > "$status_path"
    for ((position=worker_index; position<${#RECOVERY_INDICES[@]}; position+=RECOVERY_WORKERS)); do
      matrix_index="${RECOVERY_INDICES[$position]}"
      task_log="$LOG_ROOT/$(printf '%02d' "$matrix_index")_target_recovery.log"
      start_time="$(date --iso-8601=seconds)"
      set +e
      "$CONDA_EXE" run --no-capture-output -n "$MD_ENV_NAME" \
        python scripts/run_md_replica_pipeline.py \
        --matrix "$MATRIX" \
        --index "$matrix_index" \
        --platform CPU \
        --canonical-data-dir "$CANONICAL_DATA_DIR" \
        --residual-envelope sin2 \
        --normal-projection-mode product \
        > "$task_log" 2>&1
      exit_code=$?
      set -e
      end_time="$(date --iso-8601=seconds)"
      printf '%s\t%s\t%s\t%s\n' \
        "$matrix_index" "$exit_code" "$start_time" "$end_time" \
        >> "$status_path"
    done
  ) &
  worker_pids+=("$!")
done
for pid in "${worker_pids[@]}"; do
  wait "$pid"
done

set +e
"$CONDA_EXE" run --no-capture-output -n "$MD_ENV_NAME" \
  python scripts/finalize_md_replica_matrix.py \
  --matrix "$MATRIX" \
  --output-dir "$FINALIZATION_DIR" \
  > "$LOG_ROOT/finalization.log" 2>&1
finalization_exit=$?
set -e

consensus_exit=125
eligible_consensus_systems=0
if [[ "$finalization_exit" -eq 0 ]]; then
  eligible_consensus_systems="$(
    python3 -c 'import json,sys; s=json.load(open(sys.argv[1])); print(sum(v.get("target_passed", 0) >= 2 for v in s["per_system"].values()))' \
      "$FINALIZATION_DIR/summary.json"
  )"
  if [[ "$eligible_consensus_systems" -gt 0 ]]; then
    set +e
    "$CONDA_EXE" run --no-capture-output -n "$MD_ENV_NAME" \
      python scripts/build_md_phase_normal_consensus_cache.py \
      --input-cache "$FINALIZATION_DIR/phase_normal_cache" \
      --output-cache "$FINALIZATION_DIR/consensus_cache" \
      --min-replicas 2 \
      --min-support-fraction 0.5 \
      > "$LOG_ROOT/consensus.log" 2>&1
    consensus_exit=$?
    set -e
  fi
fi

python3 - "$OUTPUT_ROOT" "$LOG_ROOT" "$MATRIX" "$PILOT_STATE" \
  "$CACHE_AUDIT" "$PROVENANCE_DIR" "$FINALIZATION_DIR" "$RECOVERY_STATE" \
  "$finalization_exit" "$consensus_exit" "$eligible_consensus_systems" \
  "${RECOVERY_INDICES[@]}" <<'PY'
import csv
import hashlib
import json
import sys
from pathlib import Path

(
    output_root,
    log_root,
    matrix_path,
    pilot_state_path,
    cache_audit_path,
    provenance_dir,
    finalization_dir,
    recovery_state_path,
) = map(Path, sys.argv[1:9])
finalization_exit, consensus_exit, eligible = map(int, sys.argv[9:12])
recovery_indices = [int(value) for value in sys.argv[12:]]
processes = []
for status_path in sorted(log_root.glob("status_worker_*.tsv")):
    with status_path.open() as handle:
        for index, exit_code, start, end in csv.reader(handle, delimiter="\t"):
            index = int(index)
            processes.append(
                {
                    "matrix_index": index,
                    "exit_code": int(exit_code),
                    "start": start,
                    "end": end,
                    "log": str(log_root / f"{index:02d}_target_recovery.log"),
                }
            )
processes.sort(key=lambda row: row["matrix_index"])
finalization_path = finalization_dir / "summary.json"
finalization = json.loads(finalization_path.read_text()) if finalization_path.is_file() else None
consensus_path = finalization_dir / "consensus_cache" / "summary.json"
consensus = json.loads(consensus_path.read_text()) if consensus_path.is_file() else None

complete = (
    [row["matrix_index"] for row in processes] == recovery_indices
    and all(row["exit_code"] == 0 for row in processes)
    and finalization_exit == 0
    and (finalization or {}).get("passed_targets") == len(recovery_indices)
    and (finalization or {}).get("outcome_counts")
    == {"failed_pull": 9, "target_passed": 7}
    and eligible == 3
    and consensus_exit == 0
    and (consensus or {}).get("consensus_systems") == 3
)
state = {
    "schema_version": "bindrae_apobind_target_recovery_state_v1",
    "status": "complete" if complete else "incomplete_or_failed",
    "matrix": str(matrix_path),
    "matrix_sha256": hashlib.sha256(matrix_path.read_bytes()).hexdigest(),
    "original_pilot_state": str(pilot_state_path),
    "original_pilot_state_sha256": hashlib.sha256(pilot_state_path.read_bytes()).hexdigest(),
    "canonical_torsion_cache_audit": str(cache_audit_path),
    "canonical_torsion_cache_audit_sha256": hashlib.sha256(cache_audit_path.read_bytes()).hexdigest(),
    "pre_recovery_provenance": str(provenance_dir / "manifest.json"),
    "recovery_indices": recovery_indices,
    "counts": {
        "planned_recoveries": len(recovery_indices),
        "recorded_processes": len(processes),
        "process_succeeded": sum(row["exit_code"] == 0 for row in processes),
        "passed_targets": (finalization or {}).get("passed_targets", 0),
        "consensus_eligible_systems": eligible,
        "consensus_systems": (consensus or {}).get("consensus_systems", 0),
    },
    "outcome_counts": (finalization or {}).get("outcome_counts", {}),
    "per_system": (finalization or {}).get("per_system", {}),
    "processes": processes,
    "finalization": {
        "exit_code": finalization_exit,
        "summary": str(finalization_path),
        "log": str(log_root / "finalization.log"),
    },
    "consensus": {
        "exit_code": consensus_exit,
        "summary": str(consensus_path) if consensus else None,
        "log": str(log_root / "consensus.log") if consensus_exit != 125 else None,
    },
    "claim_boundary": (
        "Recovered targets remain biased silver paths, not physical kinetics. "
        "The recovery changes only the missing canonical-axis engineering input; "
        "all frozen pull, path, atomistic, mapping, and consensus gates remain unchanged."
    ),
}
recovery_state_path.write_text(json.dumps(state, indent=2, sort_keys=True) + "\n")
print(json.dumps(state["counts"], sort_keys=True))
print(json.dumps(state["outcome_counts"], sort_keys=True))
print(state["status"])
PY

status="$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["status"])' "$RECOVERY_STATE")"
echo "Done:  $(date --iso-8601=seconds)"
echo "State: $RECOVERY_STATE"
if [[ "$status" != "complete" ]]; then
  exit 4
fi
