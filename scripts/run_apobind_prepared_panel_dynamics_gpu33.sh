#!/usr/bin/env bash
# Direct NVT/NPT engineering smoke for one frozen APObind prepared panel.

set -euo pipefail
unset LD_PRELOAD

ROOT="${ROOT:-/mnt/inaisfs/data/home/zhaozc_criait/XinxiangWang/BINDRAE}"
FROZEN_DIR="${FROZEN_DIR:-tmp/external_dataset_audit/apobind_triplets25_v1/prepared_panel8_v1}"
PREPARED_PANEL="${PREPARED_PANEL:-$FROZEN_DIR/prepared_panel.jsonl}"
PREPARED_STATE="${PREPARED_STATE:-$FROZEN_DIR/prepared_panel_state.json}"
OUTPUT_ROOT="${OUTPUT_ROOT:-processed_data/md_transition/apobind_prepared_panel8_dynamics_20260725_v1}"
LOG_ROOT="${LOG_ROOT:-logs/stage2/apobind_prepared_panel8_dynamics_20260725_v1}"
GPU_INDICES="${GPU_INDICES:-1 2 4 7}"
MIN_FREE_GPU_MEMORY_MB="${MIN_FREE_GPU_MEMORY_MB:-20000}"
CONDA_EXE="${CONDA_EXE:-/mnt/inaisfs/data/home/zhaozc_criait/miniconda3/bin/conda}"
MD_ENV_NAME="${MD_ENV_NAME:-BINDRAE-MD}"
SEED_BASE="${SEED_BASE:-2026072500}"
EXPECTED_SYSTEMS="${EXPECTED_SYSTEMS:-8}"

if [[ "$(hostname -s)" != "gpu33" && "${ALLOW_NON_GPU33:-0}" != "1" ]]; then
  echo "ERROR: this direct launcher is restricted to gpu33" >&2
  exit 2
fi
for value in "$MIN_FREE_GPU_MEMORY_MB" "$SEED_BASE" "$EXPECTED_SYSTEMS"; do
  if ! [[ "$value" =~ ^[1-9][0-9]*$ ]]; then
    echo "ERROR: expected a positive integer, got: $value" >&2
    exit 2
  fi
done

cd "$ROOT"
for path in "$PREPARED_PANEL" "$PREPARED_STATE"; do
  if [[ ! -s "$path" ]]; then
    echo "ERROR: required frozen-panel input is missing or empty: $path" >&2
    exit 2
  fi
done
if [[ -e "$OUTPUT_ROOT" ]]; then
  echo "ERROR: dynamics output tag already exists; use a fresh tag" >&2
  exit 2
fi
if [[ -e "$LOG_ROOT" ]]; then
  unexpected_log="$(
    find "$LOG_ROOT" -mindepth 1 -maxdepth 1 ! -name launcher.log -print -quit
  )"
  if [[ -n "$unexpected_log" ]]; then
    echo "ERROR: dynamics log tag already contains experiment output: $unexpected_log" >&2
    exit 2
  fi
fi

mapfile -t panel_rows < <(
  python3 - "$PREPARED_STATE" "$PREPARED_PANEL" "$EXPECTED_SYSTEMS" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

state_path, panel_path = map(Path, sys.argv[1:3])
expected = int(sys.argv[3])
state = json.loads(state_path.read_text())
rows = [json.loads(line) for line in panel_path.read_text().splitlines() if line.strip()]
if state.get("schema_version") != "bindrae_apobind_prepared_panel_v1":
    raise SystemExit(f"Unexpected frozen-panel schema: {state.get('schema_version')}")
if state.get("status") != "ready_for_dynamics_smoke":
    raise SystemExit(f"Frozen panel is not ready: {state.get('status')}")
if len(rows) != expected or state.get("counts", {}).get("prepared_ready") != expected:
    raise SystemExit(f"Frozen panel must contain exactly {expected} ready systems")
observed = state.get("outputs", {}).get("prepared_panel_sha256")
actual = hashlib.sha256(panel_path.read_bytes()).hexdigest()
if observed != actual:
    raise SystemExit(f"Frozen prepared-panel SHA256 mismatch: {observed} != {actual}")
for index, row in enumerate(rows):
    if row.get("panel_index") != index:
        raise SystemExit(f"Panel index mismatch at row {index}")
    print(f"{row['sample_id']}\t{row['preparation_dir']}")
PY
)
if [[ "${#panel_rows[@]}" -ne "$EXPECTED_SYSTEMS" ]]; then
  echo "ERROR: frozen panel yielded ${#panel_rows[@]} rows, expected $EXPECTED_SYSTEMS" >&2
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

mkdir -p "$OUTPUT_ROOT" "$LOG_ROOT"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"

echo "APObind prepared-panel NVT/NPT smoke"
echo "Host:     $(hostname -s)"
echo "GPUs:     ${gpu_values[*]}"
echo "Systems:  ${#panel_rows[@]}"
echo "Panel:    $PREPARED_PANEL"
echo "Output:   $OUTPUT_ROOT"
echo "Logs:     $LOG_ROOT"
echo "Start:    $(date --iso-8601=seconds)"

worker_pids=()
for worker_index in "${!gpu_values[@]}"; do
  gpu="${gpu_values[$worker_index]}"
  (
    status_path="$OUTPUT_ROOT/status_worker_${worker_index}.tsv"
    : > "$status_path"
    for ((panel_index=worker_index; panel_index<${#panel_rows[@]}; panel_index+=${#gpu_values[@]})); do
      IFS=$'\t' read -r sample_id preparation_dir <<< "${panel_rows[$panel_index]}"
      seed=$((SEED_BASE + panel_index))
      sample_output="$OUTPUT_ROOT/$sample_id"
      nvt_dir="$sample_output/nvt"
      npt_dir="$sample_output/npt"
      nvt_log="$LOG_ROOT/$(printf '%02d' "$panel_index")_${sample_id}_nvt.log"
      npt_log="$LOG_ROOT/$(printf '%02d' "$panel_index")_${sample_id}_npt.log"
      mkdir -p "$nvt_dir" "$npt_dir"
      start_time="$(date --iso-8601=seconds)"
      set +e
      CUDA_VISIBLE_DEVICES="$gpu" "$CONDA_EXE" run --no-capture-output \
        -n "$MD_ENV_NAME" python scripts/run_md_pilot_dynamics_smoke.py \
        --input-dir "$preparation_dir" \
        --output-dir "$nvt_dir" \
        --platform CUDA \
        --seed "$seed" \
        --heating-steps-per-stage 250 \
        --restrained-equilibration-steps 1000 \
        --unrestrained-nvt-steps 1000 \
        > "$nvt_log" 2>&1
      nvt_exit=$?
      npt_exit=125
      if [[ "$nvt_exit" -eq 0 ]]; then
        CUDA_VISIBLE_DEVICES="$gpu" "$CONDA_EXE" run --no-capture-output \
          -n "$MD_ENV_NAME" python scripts/run_md_pilot_npt_smoke.py \
          --system-dir "$preparation_dir" \
          --nvt-dir "$nvt_dir" \
          --output-dir "$npt_dir" \
          --platform CUDA \
          --seed "$seed" \
          --restrained-equilibration-steps 2500 \
          --unrestrained-production-steps 5000 \
          > "$npt_log" 2>&1
        npt_exit=$?
      fi
      set -e
      end_time="$(date --iso-8601=seconds)"
      printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
        "$panel_index" "$sample_id" "$preparation_dir" "$gpu" "$seed" \
        "$nvt_exit" "$npt_exit" "$start_time" "$end_time" >> "$status_path"
    done
  ) &
  worker_pids+=("$!")
done
for pid in "${worker_pids[@]}"; do
  wait "$pid"
done

python3 - "$OUTPUT_ROOT" "$LOG_ROOT" "$PREPARED_PANEL" "$PREPARED_STATE" <<'PY'
import csv
import hashlib
import json
import sys
from pathlib import Path

output_root, log_root, panel_path, state_path = map(Path, sys.argv[1:])
systems = []
for status_path in sorted(output_root.glob("status_worker_*.tsv")):
    with status_path.open() as handle:
        for values in csv.reader(handle, delimiter="\t"):
            index, sample_id, preparation_dir, gpu, seed, nvt_exit, npt_exit, start, end = values
            prefix = f"{int(index):02d}_{sample_id}"
            nvt_report_path = output_root / sample_id / "nvt" / "dynamics_report.json"
            npt_report_path = output_root / sample_id / "npt" / "npt_report.json"
            nvt_report = json.loads(nvt_report_path.read_text()) if nvt_report_path.is_file() else None
            npt_report = json.loads(npt_report_path.read_text()) if npt_report_path.is_file() else None
            systems.append(
                {
                    "panel_index": int(index),
                    "sample_id": sample_id,
                    "preparation_dir": preparation_dir,
                    "physical_gpu": int(gpu),
                    "seed": int(seed),
                    "start": start,
                    "end": end,
                    "nvt_exit_code": int(nvt_exit),
                    "nvt_status": (nvt_report or {}).get("status", "process_failed"),
                    "nvt_report": str(nvt_report_path),
                    "nvt_log": str(log_root / f"{prefix}_nvt.log"),
                    "npt_exit_code": int(npt_exit),
                    "npt_status": (
                        (npt_report or {}).get("status", "process_failed")
                        if int(nvt_exit) == 0
                        else "not_run_after_nvt_failure"
                    ),
                    "npt_report": str(npt_report_path),
                    "npt_log": str(log_root / f"{prefix}_npt.log"),
                }
            )
systems.sort(key=lambda row: row["panel_index"])
complete = all(
    row["nvt_status"] == "nvt_smoke_passed"
    and row["npt_status"] == "npt_smoke_passed"
    for row in systems
)
state = {
    "schema_version": "bindrae_apobind_prepared_panel_dynamics_state_v1",
    "status": "complete" if complete else "complete_with_failures",
    "prepared_panel": str(panel_path),
    "prepared_panel_sha256": hashlib.sha256(panel_path.read_bytes()).hexdigest(),
    "prepared_panel_state": str(state_path),
    "prepared_panel_state_sha256": hashlib.sha256(state_path.read_bytes()).hexdigest(),
    "protocol": {
        "platform": "CUDA",
        "timestep_fs": 2.0,
        "nvt_heating_temperatures_k": [50, 100, 150, 200, 250, 300],
        "nvt_heating_steps_per_stage": 250,
        "nvt_restrained_steps": 1000,
        "nvt_unrestrained_steps": 1000,
        "npt_restrained_steps": 2500,
        "npt_unrestrained_steps": 5000,
    },
    "counts": {
        "attempted": len(systems),
        "nvt_process_succeeded": sum(row["nvt_exit_code"] == 0 for row in systems),
        "nvt_passed": sum(row["nvt_status"] == "nvt_smoke_passed" for row in systems),
        "npt_attempted": sum(row["nvt_exit_code"] == 0 for row in systems),
        "npt_process_succeeded": sum(row["npt_exit_code"] == 0 for row in systems),
        "npt_passed": sum(row["npt_status"] == "npt_smoke_passed" for row in systems),
    },
    "systems": systems,
    "claim_boundary": (
        "This is a short prepared-system NVT/NPT engineering smoke. Passing systems "
        "are not accepted independent MD replicas or Path-3 supervision."
    ),
}
(output_root / "panel_dynamics_state.json").write_text(
    json.dumps(state, indent=2, sort_keys=True) + "\n"
)
print(json.dumps(state["counts"], sort_keys=True))
PY

echo "Done:  $(date --iso-8601=seconds)"
echo "State: $OUTPUT_ROOT/panel_dynamics_state.json"
