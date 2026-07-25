#!/usr/bin/env bash
# Direct four-GPU execution of the frozen APObind 8-system x 2-replica pilot.

set -euo pipefail
unset LD_PRELOAD

ROOT="${ROOT:-/mnt/inaisfs/data/home/zhaozc_criait/XinxiangWang/BINDRAE}"
OUTPUT_ROOT="${OUTPUT_ROOT:-processed_data/md_transition/apobind_replica_pilot8x2_20260725_v1}"
PLAN="${PLAN:-$OUTPUT_ROOT/pilot_plan.json}"
MATRIX="${MATRIX:-$OUTPUT_ROOT/replica_matrix.jsonl}"
CANONICAL_DATA_DIR="${CANONICAL_DATA_DIR:-tmp/external_dataset_audit/apobind_triplets25_v1}"
CACHE_AUDIT="${CACHE_AUDIT:-$OUTPUT_ROOT/canonical_torsion_cache_audit.json}"
LOG_ROOT="${LOG_ROOT:-logs/stage2/apobind_replica_pilot8x2_20260725_v1}"
GPU_INDICES="${GPU_INDICES:-1 2 4 7}"
MIN_FREE_GPU_MEMORY_MB="${MIN_FREE_GPU_MEMORY_MB:-20000}"
CONDA_EXE="${CONDA_EXE:-/mnt/inaisfs/data/home/zhaozc_criait/miniconda3/bin/conda}"
MD_ENV_NAME="${MD_ENV_NAME:-BINDRAE-MD}"
EXPECTED_SYSTEMS="${EXPECTED_SYSTEMS:-8}"
EXPECTED_TASKS="${EXPECTED_TASKS:-16}"

if [[ "$(hostname -s)" != "gpu33" && "${ALLOW_NON_GPU33:-0}" != "1" ]]; then
  echo "ERROR: this direct launcher is restricted to gpu33" >&2
  exit 2
fi
for value in "$MIN_FREE_GPU_MEMORY_MB" "$EXPECTED_SYSTEMS" "$EXPECTED_TASKS"; do
  if ! [[ "$value" =~ ^[1-9][0-9]*$ ]]; then
    echo "ERROR: expected a positive integer, got: $value" >&2
    exit 2
  fi
done

cd "$ROOT"
for path in "$PLAN" "$MATRIX"; do
  if [[ ! -s "$path" ]]; then
    echo "ERROR: required replica-pilot input is missing or empty: $path" >&2
    exit 2
  fi
done
if [[ -e "$OUTPUT_ROOT/pilot_state.json" || -e "$OUTPUT_ROOT/pulls" || -e "$OUTPUT_ROOT/targets" ]]; then
  echo "ERROR: replica-pilot task output already exists; use a fresh output tag" >&2
  exit 2
fi
if [[ -e "$LOG_ROOT" ]]; then
  unexpected_log="$(
    find "$LOG_ROOT" -mindepth 1 -maxdepth 1 ! -name launcher.log -print -quit
  )"
  if [[ -n "$unexpected_log" ]]; then
    echo "ERROR: replica-pilot log tag already contains task output: $unexpected_log" >&2
    exit 2
  fi
fi

python3 - "$PLAN" "$MATRIX" "$EXPECTED_SYSTEMS" "$EXPECTED_TASKS" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

plan_path, matrix_path = map(Path, sys.argv[1:3])
expected_systems, expected_tasks = map(int, sys.argv[3:5])
plan = json.loads(plan_path.read_text())
rows = [json.loads(line) for line in matrix_path.read_text().splitlines() if line.strip()]
if plan.get("schema_version") != "bindrae_apobind_replica_pilot_plan_v1":
    raise SystemExit(f"Unexpected pilot-plan schema: {plan.get('schema_version')}")
if plan.get("status") != "ready_to_launch":
    raise SystemExit(f"Pilot plan is not ready: {plan.get('status')}")
counts = plan.get("counts", {})
if counts.get("systems") != expected_systems or counts.get("tasks") != expected_tasks:
    raise SystemExit(f"Unexpected plan counts: {counts}")
if counts.get("replicas_per_system") != 2:
    raise SystemExit(f"Pilot requires exactly two replicas per system: {counts}")
actual = hashlib.sha256(matrix_path.read_bytes()).hexdigest()
if plan.get("outputs", {}).get("replica_matrix_sha256") != actual:
    raise SystemExit("Replica matrix SHA256 mismatch")
if len(rows) != expected_tasks:
    raise SystemExit(f"Replica matrix has {len(rows)} tasks, expected {expected_tasks}")
for index, row in enumerate(rows):
    if row.get("matrix_index") != index:
        raise SystemExit(f"Matrix index mismatch at row {index}")
protocol = plan.get("protocol", {})
frozen = {
    "pre_equilibration_steps": 500,
    "pulling_steps": 10000,
    "endpoint_hold_steps": 2000,
    "report_interval": 100,
    "rmsd_k_kj_mol_nm2": 200000.0,
    "final_target_rmsd_nm": 0.025,
    "min_mapping_fraction": 0.95,
    "resample_initial_velocities": True,
}
for key, expected in frozen.items():
    if protocol.get(key) != expected:
        raise SystemExit(f"Frozen protocol mismatch for {key}: {protocol.get(key)}")
print(json.dumps({"systems": expected_systems, "tasks": expected_tasks}, sort_keys=True))
PY

read -r -a gpu_values <<< "$GPU_INDICES"
if [[ "${#gpu_values[@]}" -lt 1 ]]; then
  echo "ERROR: GPU_INDICES is empty" >&2
  exit 2
fi
for gpu in "${gpu_values[@]}"; do
  if ! [[ "$gpu" =~ ^[0-9]+$ ]]; then
    echo "ERROR: invalid GPU index: $gpu" >&2
    exit 2
  fi
  free_mb="$(
    nvidia-smi --id="$gpu" --query-gpu=memory.free --format=csv,noheader,nounits \
      | tr -d ' '
  )"
  if [[ -z "$free_mb" || "$free_mb" -lt "$MIN_FREE_GPU_MEMORY_MB" ]]; then
    echo "ERROR: GPU $gpu has ${free_mb:-unknown} MiB free; need $MIN_FREE_GPU_MEMORY_MB" >&2
    exit 3
  fi
done

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

echo "APObind 8-system x 2-replica silver-path pilot"
echo "Host:      $(hostname -s)"
echo "GPUs:      ${gpu_values[*]}"
echo "Matrix:    $MATRIX"
echo "Data root: $CANONICAL_DATA_DIR"
echo "Cache audit: $CACHE_AUDIT"
echo "Tasks:     $EXPECTED_TASKS"
echo "Output:    $OUTPUT_ROOT"
echo "Logs:      $LOG_ROOT"
echo "Start:     $(date --iso-8601=seconds)"

worker_pids=()
for worker_index in "${!gpu_values[@]}"; do
  gpu="${gpu_values[$worker_index]}"
  (
    status_path="$OUTPUT_ROOT/status_worker_${worker_index}.tsv"
    : > "$status_path"
    for ((matrix_index=worker_index; matrix_index<EXPECTED_TASKS; matrix_index+=${#gpu_values[@]})); do
      task_log="$LOG_ROOT/$(printf '%02d' "$matrix_index")_replica.log"
      start_time="$(date --iso-8601=seconds)"
      set +e
      CUDA_VISIBLE_DEVICES="$gpu" "$CONDA_EXE" run --no-capture-output \
        -n "$MD_ENV_NAME" python scripts/run_md_replica_pipeline.py \
        --matrix "$MATRIX" \
        --index "$matrix_index" \
        --platform CUDA \
        --canonical-data-dir "$CANONICAL_DATA_DIR" \
        --residual-envelope sin2 \
        --normal-projection-mode product \
        > "$task_log" 2>&1
      exit_code=$?
      set -e
      end_time="$(date --iso-8601=seconds)"
      printf '%s\t%s\t%s\t%s\t%s\n' \
        "$matrix_index" "$gpu" "$exit_code" "$start_time" "$end_time" \
        >> "$status_path"
    done
  ) &
  worker_pids+=("$!")
done
for pid in "${worker_pids[@]}"; do
  wait "$pid"
done

finalization_dir="$OUTPUT_ROOT/finalization"
set +e
"$CONDA_EXE" run --no-capture-output -n "$MD_ENV_NAME" \
  python scripts/finalize_md_replica_matrix.py \
  --matrix "$MATRIX" \
  --output-dir "$finalization_dir" \
  > "$LOG_ROOT/finalization.log" 2>&1
finalization_exit=$?
set -e

consensus_exit=125
eligible_consensus_systems=0
if [[ "$finalization_exit" -eq 0 ]]; then
  eligible_consensus_systems="$(
    python3 -c 'import json,sys; s=json.load(open(sys.argv[1])); print(sum(v.get("target_passed", 0) >= 2 for v in s["per_system"].values()))' \
      "$finalization_dir/summary.json"
  )"
  if [[ "$eligible_consensus_systems" -gt 0 ]]; then
    set +e
    "$CONDA_EXE" run --no-capture-output -n "$MD_ENV_NAME" \
      python scripts/build_md_phase_normal_consensus_cache.py \
      --input-cache "$finalization_dir/phase_normal_cache" \
      --output-cache "$finalization_dir/consensus_cache" \
      --min-replicas 2 \
      --min-support-fraction 0.5 \
      > "$LOG_ROOT/consensus.log" 2>&1
    consensus_exit=$?
    set -e
  fi
fi

python3 - "$OUTPUT_ROOT" "$LOG_ROOT" "$PLAN" "$MATRIX" "$CACHE_AUDIT" \
  "$finalization_exit" "$consensus_exit" "$eligible_consensus_systems" <<'PY'
import csv
import hashlib
import json
import sys
from pathlib import Path

output_root, log_root, plan_path, matrix_path, cache_audit_path = map(Path, sys.argv[1:6])
finalization_exit, consensus_exit, eligible = map(int, sys.argv[6:9])
matrix = [json.loads(line) for line in matrix_path.read_text().splitlines() if line.strip()]
by_index = {row["matrix_index"]: row for row in matrix}
process_rows = []
for status_path in sorted(output_root.glob("status_worker_*.tsv")):
    with status_path.open() as handle:
        for values in csv.reader(handle, delimiter="\t"):
            index, gpu, exit_code, start, end = values
            index = int(index)
            task = by_index[index]
            process_rows.append(
                {
                    "matrix_index": index,
                    "system_sample_id": task["system_sample_id"],
                    "sample_id": task["sample_id"],
                    "replica_index": task["replica_index"],
                    "seed": task["seed"],
                    "physical_gpu": int(gpu),
                    "exit_code": int(exit_code),
                    "start": start,
                    "end": end,
                    "log": str(log_root / f"{index:02d}_replica.log"),
                }
            )
process_rows.sort(key=lambda row: row["matrix_index"])
finalization_path = output_root / "finalization" / "summary.json"
finalization = json.loads(finalization_path.read_text()) if finalization_path.is_file() else None
consensus_path = output_root / "finalization" / "consensus_cache" / "summary.json"
consensus = json.loads(consensus_path.read_text()) if consensus_path.is_file() else None
if finalization_exit != 0:
    status = "finalization_failed"
elif consensus_exit not in (0, 125):
    status = "consensus_failed"
else:
    status = "complete"
state = {
    "schema_version": "bindrae_apobind_replica_pilot_state_v1",
    "status": status,
    "plan": str(plan_path),
    "plan_sha256": hashlib.sha256(plan_path.read_bytes()).hexdigest(),
    "matrix": str(matrix_path),
    "matrix_sha256": hashlib.sha256(matrix_path.read_bytes()).hexdigest(),
    "canonical_torsion_cache_audit": str(cache_audit_path),
    "canonical_torsion_cache_audit_sha256": hashlib.sha256(
        cache_audit_path.read_bytes()
    ).hexdigest(),
    "counts": {
        "planned_tasks": len(matrix),
        "recorded_processes": len(process_rows),
        "process_succeeded": sum(row["exit_code"] == 0 for row in process_rows),
        "process_failed": sum(row["exit_code"] != 0 for row in process_rows),
        "passed_targets": (finalization or {}).get("passed_targets", 0),
        "consensus_eligible_systems": eligible,
        "consensus_systems": (consensus or {}).get("consensus_systems", 0),
    },
    "outcome_counts": (finalization or {}).get("outcome_counts", {}),
    "per_system": (finalization or {}).get("per_system", {}),
    "processes": process_rows,
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
        "These are biased silver-path attempts, not physical kinetics. Only passed "
        "targets in systems with at least two replicas contribute to consensus "
        "Path-3 supervision."
    ),
}
(output_root / "pilot_state.json").write_text(
    json.dumps(state, indent=2, sort_keys=True) + "\n"
)
print(json.dumps(state["counts"], sort_keys=True))
print(json.dumps(state["outcome_counts"], sort_keys=True))
PY

echo "Done:  $(date --iso-8601=seconds)"
echo "State: $OUTPUT_ROOT/pilot_state.json"
