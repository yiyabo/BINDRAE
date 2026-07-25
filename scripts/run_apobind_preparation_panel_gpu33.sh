#!/usr/bin/env bash
# Direct four-worker OpenMM preparation smoke for the leakage-clean APObind panel.

set -euo pipefail
unset LD_PRELOAD

ROOT="${ROOT:-/mnt/inaisfs/data/home/zhaozc_criait/XinxiangWang/BINDRAE}"
PANEL_DIR="${PANEL_DIR:-tmp/external_dataset_audit/apobind_triplets25_v1/preparation_panel8_v1}"
CANDIDATE_MANIFEST="${CANDIDATE_MANIFEST:-$PANEL_DIR/selected_transition_manifest.jsonl}"
SAMPLE_IDS_FILE="${SAMPLE_IDS_FILE:-$PANEL_DIR/selected_sample_ids.txt}"
OUTPUT_ROOT="${OUTPUT_ROOT:-processed_data/md_transition/apobind_preparation_panel8_20260725_v1}"
LOG_ROOT="${LOG_ROOT:-logs/stage2/apobind_preparation_panel8_20260725_v1}"
GPU_INDICES="${GPU_INDICES:-1 2 4 7}"
MIN_FREE_GPU_MEMORY_MB="${MIN_FREE_GPU_MEMORY_MB:-20000}"
CONDA_EXE="${CONDA_EXE:-/mnt/inaisfs/data/home/zhaozc_criait/miniconda3/bin/conda}"
MD_ENV_NAME="${MD_ENV_NAME:-BINDRAE-MD}"
MAX_MINIMIZATION_ITERATIONS="${MAX_MINIMIZATION_ITERATIONS:-500}"
SOLVENT_MINIMIZATION_ITERATIONS="${SOLVENT_MINIMIZATION_ITERATIONS:-1000}"
MAX_RESIDUE_NET_FORCE="${MAX_RESIDUE_NET_FORCE:-500}"

if [[ "$(hostname -s)" != "gpu33" && "${ALLOW_NON_GPU33:-0}" != "1" ]]; then
  echo "ERROR: this direct launcher is restricted to gpu33" >&2
  exit 2
fi
if ! [[ "$MIN_FREE_GPU_MEMORY_MB" =~ ^[1-9][0-9]*$ ]]; then
  echo "ERROR: MIN_FREE_GPU_MEMORY_MB must be positive" >&2
  exit 2
fi

cd "$ROOT"
for path in "$CANDIDATE_MANIFEST" "$SAMPLE_IDS_FILE"; do
  if [[ ! -s "$path" ]]; then
    echo "ERROR: required panel input is missing or empty: $path" >&2
    exit 2
  fi
done
if [[ -e "$OUTPUT_ROOT/panel_state.json" ]]; then
  echo "ERROR: panel state already exists: $OUTPUT_ROOT/panel_state.json" >&2
  exit 2
fi

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

mapfile -t sample_ids < <(sed '/^[[:space:]]*$/d' "$SAMPLE_IDS_FILE")
manifest_count="$(wc -l < "$CANDIDATE_MANIFEST" | tr -d ' ')"
if [[ "${#sample_ids[@]}" -ne "$manifest_count" ]]; then
  echo "ERROR: sample list has ${#sample_ids[@]} rows but manifest has $manifest_count" >&2
  exit 2
fi
if [[ "${#sample_ids[@]}" -lt 1 ]]; then
  echo "ERROR: preparation panel is empty" >&2
  exit 2
fi

mkdir -p "$OUTPUT_ROOT" "$LOG_ROOT"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"

echo "APObind prepared-system smoke"
echo "Host:       $(hostname -s)"
echo "GPUs:       ${gpu_values[*]}"
echo "Candidates: ${#sample_ids[@]}"
echo "Manifest:   $CANDIDATE_MANIFEST"
echo "Output:     $OUTPUT_ROOT"
echo "Logs:       $LOG_ROOT"
echo "Start:      $(date --iso-8601=seconds)"

worker_pids=()
for worker_index in "${!gpu_values[@]}"; do
  gpu="${gpu_values[$worker_index]}"
  (
    status_path="$OUTPUT_ROOT/status_worker_${worker_index}.tsv"
    : > "$status_path"
    for ((candidate_index=worker_index; candidate_index<${#sample_ids[@]}; candidate_index+=${#gpu_values[@]})); do
      sample_id="${sample_ids[$candidate_index]}"
      sample_output="$OUTPUT_ROOT/$sample_id"
      sample_log="$LOG_ROOT/$(printf '%02d' "$candidate_index")_${sample_id}.log"
      start_time="$(date --iso-8601=seconds)"
      set +e
      CUDA_VISIBLE_DEVICES="$gpu" "$CONDA_EXE" run --no-capture-output \
        -n "$MD_ENV_NAME" python scripts/prepare_md_pilot_system.py \
        --candidate-manifest "$CANDIDATE_MANIFEST" \
        --candidate-index "$candidate_index" \
        --output-dir "$sample_output" \
        --platform CUDA \
        --max-minimization-iterations "$MAX_MINIMIZATION_ITERATIONS" \
        --solvent-minimization-iterations "$SOLVENT_MINIMIZATION_ITERATIONS" \
        --max-residue-net-force-kj-mol-nm "$MAX_RESIDUE_NET_FORCE" \
        > "$sample_log" 2>&1
      exit_code=$?
      set -e
      end_time="$(date --iso-8601=seconds)"
      printf '%s\t%s\t%s\t%s\t%s\t%s\n' \
        "$candidate_index" "$sample_id" "$gpu" "$exit_code" "$start_time" "$end_time" \
        >> "$status_path"
    done
  ) &
  worker_pids+=("$!")
done

for pid in "${worker_pids[@]}"; do
  wait "$pid"
done

"$CONDA_EXE" run --no-capture-output -n "$MD_ENV_NAME" python - \
  "$OUTPUT_ROOT" "$LOG_ROOT" "$CANDIDATE_MANIFEST" <<'PY'
import csv
import hashlib
import json
import sys
from pathlib import Path

output_root = Path(sys.argv[1])
log_root = Path(sys.argv[2])
manifest = Path(sys.argv[3])
rows = []
for status_path in sorted(output_root.glob("status_worker_*.tsv")):
    with status_path.open() as handle:
        for values in csv.reader(handle, delimiter="\t"):
            index, sample_id, gpu, exit_code, start, end = values
            report_path = output_root / sample_id / "preparation_report.json"
            report = json.loads(report_path.read_text()) if report_path.is_file() else None
            log_path = log_root / f"{int(index):02d}_{sample_id}.log"
            rows.append(
                {
                    "candidate_index": int(index),
                    "sample_id": sample_id,
                    "physical_gpu": int(gpu),
                    "exit_code": int(exit_code),
                    "start": start,
                    "end": end,
                    "preparation_report": str(report_path),
                    "log": str(log_path),
                    "status": report.get("status") if report else "process_failed",
                    "ready_for_dynamics_smoke": bool(
                        report
                        and (report.get("minimization") or {}).get(
                            "ready_for_dynamics_smoke"
                        )
                    ),
                    "error_tail": (
                        []
                        if report or not log_path.is_file()
                        else log_path.read_text(errors="replace").splitlines()[-12:]
                    ),
                }
            )
rows.sort(key=lambda row: row["candidate_index"])
digest = hashlib.sha256(manifest.read_bytes()).hexdigest()
state = {
    "schema_version": "bindrae_apobind_preparation_panel_state_v1",
    "status": "complete_with_failures" if any(row["exit_code"] for row in rows) else "complete",
    "candidate_manifest": str(manifest),
    "candidate_manifest_sha256": digest,
    "counts": {
        "attempted": len(rows),
        "process_succeeded": sum(row["exit_code"] == 0 for row in rows),
        "process_failed": sum(row["exit_code"] != 0 for row in rows),
        "ready_for_dynamics_smoke": sum(row["ready_for_dynamics_smoke"] for row in rows),
        "minimized_incomplete": sum(row["status"] == "minimized_incomplete" for row in rows),
    },
    "systems": rows,
    "claim_boundary": (
        "This is a direct GPU33 prepared-system engineering smoke. A successful "
        "preparation is not an accepted MD replica or Path-3 supervision."
    ),
}
(output_root / "panel_state.json").write_text(
    json.dumps(state, indent=2, sort_keys=True) + "\n"
)
print(json.dumps(state["counts"], sort_keys=True))
PY

echo "Done:       $(date --iso-8601=seconds)"
echo "State:      $OUTPUT_ROOT/panel_state.json"
