#!/usr/bin/env bash
# Direct two-system APObind preparation rescue on private GPU33.

set -euo pipefail
unset LD_PRELOAD

ROOT="${ROOT:-/mnt/inaisfs/data/home/zhaozc_criait/XinxiangWang/BINDRAE}"
PLAN_DIR="${PLAN_DIR:-tmp/external_dataset_audit/apobind_triplets25_v1/preparation_rescue_v1}"
PLAN="${PLAN:-$PLAN_DIR/rescue_plan.json}"
MANIFEST="${MANIFEST:-$PLAN_DIR/rescue_transition_manifest.jsonl}"
OUTPUT_ROOT="${OUTPUT_ROOT:-processed_data/md_transition/apobind_preparation_rescue_20260725_v1}"
LOG_ROOT="${LOG_ROOT:-logs/stage2/apobind_preparation_rescue_20260725_v1}"
GPU_INDICES="${GPU_INDICES:-1 2}"
MIN_FREE_GPU_MEMORY_MB="${MIN_FREE_GPU_MEMORY_MB:-20000}"
CONDA_EXE="${CONDA_EXE:-/mnt/inaisfs/data/home/zhaozc_criait/miniconda3/bin/conda}"
MD_ENV_NAME="${MD_ENV_NAME:-BINDRAE-MD}"

if [[ "$(hostname -s)" != "gpu33" && "${ALLOW_NON_GPU33:-0}" != "1" ]]; then
  echo "ERROR: this direct launcher is restricted to gpu33" >&2
  exit 2
fi
cd "$ROOT"
for path in "$PLAN" "$MANIFEST"; do
  if [[ ! -s "$path" ]]; then
    echo "ERROR: required rescue input is missing or empty: $path" >&2
    exit 2
  fi
done
if [[ -e "$OUTPUT_ROOT/rescue_state.json" ]]; then
  echo "ERROR: rescue state already exists: $OUTPUT_ROOT/rescue_state.json" >&2
  exit 2
fi

read -r -a gpu_values <<< "$GPU_INDICES"
if [[ "${#gpu_values[@]}" -ne 2 ]]; then
  echo "ERROR: GPU_INDICES must contain exactly two GPUs" >&2
  exit 2
fi
for gpu in "${gpu_values[@]}"; do
  free_mb="$(nvidia-smi --id="$gpu" --query-gpu=memory.free --format=csv,noheader,nounits | tr -d ' ')"
  if [[ -z "$free_mb" || "$free_mb" -lt "$MIN_FREE_GPU_MEMORY_MB" ]]; then
    echo "ERROR: GPU $gpu has ${free_mb:-unknown} MiB free; need $MIN_FREE_GPU_MEMORY_MB" >&2
    exit 3
  fi
done

mapfile -t role_rows < <(
  python3 -c 'import json,sys; p=json.load(open(sys.argv[1])); [print("{}\t{}\t{}\t{}".format(r["sample_id"], r["role"], r["max_minimization_iterations"], r["residue_force_threshold_kj_mol_nm"])) for r in p["roles"]]' "$PLAN"
)
if [[ "${#role_rows[@]}" -ne 2 || "$(wc -l < "$MANIFEST" | tr -d ' ')" -ne 2 ]]; then
  echo "ERROR: rescue plan and manifest must each contain two systems" >&2
  exit 2
fi

mkdir -p "$OUTPUT_ROOT" "$LOG_ROOT"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"

echo "APObind bounded preparation rescue"
echo "Host:     $(hostname -s)"
echo "GPUs:     ${gpu_values[*]}"
echo "Plan:     $PLAN"
echo "Manifest: $MANIFEST"
echo "Output:   $OUTPUT_ROOT"
echo "Start:    $(date --iso-8601=seconds)"

pids=()
for candidate_index in 0 1; do
  IFS=$'\t' read -r sample_id role max_iterations threshold <<< "${role_rows[$candidate_index]}"
  gpu="${gpu_values[$candidate_index]}"
  sample_output="$OUTPUT_ROOT/$sample_id"
  sample_log="$LOG_ROOT/$(printf '%02d' "$candidate_index")_${sample_id}.log"
  status_path="$OUTPUT_ROOT/status_${candidate_index}.tsv"
  (
    start_time="$(date --iso-8601=seconds)"
    set +e
    CUDA_VISIBLE_DEVICES="$gpu" "$CONDA_EXE" run --no-capture-output \
      -n "$MD_ENV_NAME" python scripts/prepare_md_pilot_system.py \
      --candidate-manifest "$MANIFEST" \
      --candidate-index "$candidate_index" \
      --output-dir "$sample_output" \
      --platform CUDA \
      --max-minimization-iterations "$max_iterations" \
      --solvent-minimization-iterations 1000 \
      --max-residue-net-force-kj-mol-nm "$threshold" \
      > "$sample_log" 2>&1
    exit_code=$?
    set -e
    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
      "$candidate_index" "$sample_id" "$role" "$gpu" "$max_iterations" \
      "$exit_code" "$start_time" > "$status_path"
  ) &
  pids+=("$!")
done
for pid in "${pids[@]}"; do
  wait "$pid"
done

python3 - "$OUTPUT_ROOT" "$LOG_ROOT" "$PLAN" "$MANIFEST" <<'PY'
import csv
import hashlib
import json
import sys
from pathlib import Path

output_root, log_root, plan_path, manifest_path = map(Path, sys.argv[1:])
systems = []
for status_path in sorted(output_root.glob("status_*.tsv")):
    values = next(csv.reader(status_path.open(), delimiter="\t"))
    index, sample_id, role, gpu, max_iterations, exit_code, start = values
    report_path = output_root / sample_id / "preparation_report.json"
    report = json.loads(report_path.read_text()) if report_path.is_file() else None
    log_path = log_root / f"{int(index):02d}_{sample_id}.log"
    minimization = (report or {}).get("minimization") or {}
    systems.append(
        {
            "candidate_index": int(index),
            "sample_id": sample_id,
            "role": role,
            "physical_gpu": int(gpu),
            "max_minimization_iterations": int(max_iterations),
            "exit_code": int(exit_code),
            "start": start,
            "status": (report or {}).get("status", "process_failed"),
            "ready_for_dynamics_smoke": bool(
                minimization.get("ready_for_dynamics_smoke")
            ),
            "maximum_residue_net_force_kj_mol_nm": minimization.get(
                "maximum_residue_net_force_kj_mol_nm"
            ),
            "preparation_report": str(report_path),
            "log": str(log_path),
            "error_tail": (
                []
                if report or not log_path.is_file()
                else log_path.read_text(errors="replace").splitlines()[-12:]
            ),
        }
    )
systems.sort(key=lambda row: row["candidate_index"])
state = {
    "schema_version": "bindrae_apobind_preparation_rescue_state_v1",
    "status": "complete_with_failures" if any(row["exit_code"] for row in systems) else "complete",
    "plan": str(plan_path),
    "plan_sha256": hashlib.sha256(plan_path.read_bytes()).hexdigest(),
    "manifest": str(manifest_path),
    "manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
    "counts": {
        "attempted": len(systems),
        "process_succeeded": sum(row["exit_code"] == 0 for row in systems),
        "process_failed": sum(row["exit_code"] != 0 for row in systems),
        "ready_for_dynamics_smoke": sum(row["ready_for_dynamics_smoke"] for row in systems),
    },
    "systems": systems,
    "claim_boundary": (
        "This bounded rescue preserves the 500 kJ/mol/nm residue-force threshold. "
        "Ready systems remain engineering inputs, not accepted MD replicas."
    ),
}
(output_root / "rescue_state.json").write_text(
    json.dumps(state, indent=2, sort_keys=True) + "\n"
)
print(json.dumps(state["counts"], sort_keys=True))
PY

echo "Done:  $(date --iso-8601=seconds)"
echo "State: $OUTPUT_ROOT/rescue_state.json"
