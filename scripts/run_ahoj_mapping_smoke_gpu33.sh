#!/usr/bin/env bash
# Direct, resumable GPU33 smoke for 32 mapping-aware AHoJ endpoint systems.

set -euo pipefail
unset LD_PRELOAD

ROOT="${ROOT:-/mnt/inaisfs/data/home/zhaozc_criait/XinxiangWang/BINDRAE}"
CANDIDATE_MANIFEST="${CANDIDATE_MANIFEST:-processed_data/md_transition/path3_capacity_leakclean_mapping95_20260725_v1/novel_transition_manifest.jsonl}"
OUTPUT_ROOT="${OUTPUT_ROOT:-processed_data/md_transition/ahoj_mapping_smoke32x2_gpu33_20260725_v1}"
LOG_ROOT="${LOG_ROOT:-logs/stage2/ahoj_mapping_smoke32x2_gpu33_20260725_v1}"
CANONICAL_DATA_DIR="${CANONICAL_DATA_DIR:-processed_data/triplets}"
GPU_INDEX="${GPU_INDEX:-0}"
MIN_FREE_GPU_MEMORY_MB="${MIN_FREE_GPU_MEMORY_MB:-20000}"
PANEL_SIZE="${PANEL_SIZE:-32}"
CONDA_EXE="${CONDA_EXE:-/mnt/inaisfs/data/home/zhaozc_criait/miniconda3/bin/conda}"
MD_ENV_NAME="${MD_ENV_NAME:-BINDRAE-MD}"
RESUME="${RESUME:-0}"

if [[ "$(hostname -s)" != "gpu33" && "${ALLOW_NON_GPU33:-0}" != "1" ]]; then
  echo "ERROR: this direct launcher is restricted to gpu33" >&2
  exit 2
fi
for value_name in GPU_INDEX MIN_FREE_GPU_MEMORY_MB PANEL_SIZE RESUME; do
  value="${!value_name}"
  if ! [[ "$value" =~ ^[0-9]+$ ]]; then
    echo "ERROR: $value_name must be a non-negative integer: $value" >&2
    exit 2
  fi
done
if [[ "$PANEL_SIZE" -lt 1 || "$MIN_FREE_GPU_MEMORY_MB" -lt 1 ]]; then
  echo "ERROR: PANEL_SIZE and MIN_FREE_GPU_MEMORY_MB must be positive" >&2
  exit 2
fi

cd "$ROOT"
if [[ ! -s "$CANDIDATE_MANIFEST" ]]; then
  echo "ERROR: missing candidate manifest: $CANDIDATE_MANIFEST" >&2
  exit 2
fi
if [[ "$RESUME" -eq 0 && -e "$OUTPUT_ROOT" ]]; then
  echo "ERROR: output tag already exists; use a fresh tag or RESUME=1" >&2
  exit 3
fi
if [[ "$RESUME" -eq 0 && -d "$LOG_ROOT" ]]; then
  unexpected_log="$(find "$LOG_ROOT" -mindepth 1 -maxdepth 1 ! -name launcher.log -print -quit)"
  if [[ -n "$unexpected_log" ]]; then
    echo "ERROR: log tag already contains task output: $unexpected_log" >&2
    exit 3
  fi
fi
mkdir -p "$OUTPUT_ROOT" "$LOG_ROOT"

free_mb="$(nvidia-smi --id="$GPU_INDEX" --query-gpu=memory.free --format=csv,noheader,nounits | tr -d ' ')"
if [[ -z "$free_mb" || "$free_mb" -lt "$MIN_FREE_GPU_MEMORY_MB" ]]; then
  echo "ERROR: GPU $GPU_INDEX has ${free_mb:-unknown} MiB free; need $MIN_FREE_GPU_MEMORY_MB" >&2
  exit 4
fi

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
echo "AHoJ mapping-aware 32-system x 2-replica smoke"
echo "Host: $(hostname -s); GPU: $GPU_INDEX; free MiB: $free_mb"
echo "Start: $(date --iso-8601=seconds)"

"$CONDA_EXE" run --no-capture-output -n "$MD_ENV_NAME" \
  python scripts/select_md_mapping_smoke_panel.py \
  --candidate-manifest "$CANDIDATE_MANIFEST" \
  --output-dir "$OUTPUT_ROOT/panel" \
  --panel-size "$PANEL_SIZE" \
  > "$LOG_ROOT/select_panel.log" 2>&1

PANEL_MANIFEST="$OUTPUT_ROOT/panel/mapping_smoke_panel_manifest.jsonl"
CONTEXT_DIR="$OUTPUT_ROOT/context"
CONTEXT_MATRIX="$CONTEXT_DIR/context_matrix.jsonl"
"$CONDA_EXE" run --no-capture-output -n "$MD_ENV_NAME" \
  python scripts/build_md_context_matrix.py \
  --candidate-manifest "$PANEL_MANIFEST" \
  --output-dir "$CONTEXT_DIR" \
  --seed-base 2026072500 \
  --protocol-tag ahoj_mapping_smoke32_v1 \
  > "$LOG_ROOT/build_context_matrix.log" 2>&1

context_tasks="$(wc -l < "$CONTEXT_MATRIX" | tr -d ' ')"
for ((index=0; index<context_tasks; index++)); do
  task_log="$LOG_ROOT/context_$(printf '%02d' "$index").log"
  set +e
  CUDA_VISIBLE_DEVICES="$GPU_INDEX" "$CONDA_EXE" run --no-capture-output -n "$MD_ENV_NAME" \
    python scripts/run_md_context_pipeline.py \
    --matrix "$CONTEXT_MATRIX" --index "$index" --platform CUDA \
    >> "$task_log" 2>&1
  exit_code=$?
  set -e
  printf '%s\t%s\n' "$index" "$exit_code" >> "$OUTPUT_ROOT/context_exit_codes.tsv"
done

CONTEXT_COLLECTION="$CONTEXT_DIR/collection"
PASSED_CONTEXTS="$CONTEXT_COLLECTION/passed_contexts.jsonl"
"$CONDA_EXE" run --no-capture-output -n "$MD_ENV_NAME" \
  python scripts/collect_md_context_results.py \
  --matrix "$CONTEXT_MATRIX" \
  --output-manifest "$PASSED_CONTEXTS" \
  --summary "$CONTEXT_COLLECTION/summary.json" \
  > "$LOG_ROOT/collect_context.log" 2>&1

passed_contexts="$(wc -l < "$PASSED_CONTEXTS" | tr -d ' ')"
REPLICA_DIR="$OUTPUT_ROOT/replicas"
finalization_exit=125
consensus_exit=125
if [[ "$passed_contexts" -gt 0 ]]; then
  "$CONDA_EXE" run --no-capture-output -n "$MD_ENV_NAME" \
    python scripts/build_md_replica_matrix.py \
    --candidate-manifest "$PANEL_MANIFEST" \
    --context-manifest "$PASSED_CONTEXTS" \
    --output-dir "$REPLICA_DIR" \
    --replica-start 0 --replica-stop 1 \
    --seed-base 2026072600 \
    --protocol-tag ahoj_mapping_smoke32_v1 \
    --min-mapping-fraction 0.95 \
    > "$LOG_ROOT/build_replica_matrix.log" 2>&1

  MATRIX="$REPLICA_DIR/replica_matrix.jsonl"
  replica_tasks="$(wc -l < "$MATRIX" | tr -d ' ')"
  for ((index=0; index<replica_tasks; index++)); do
    task_log="$LOG_ROOT/replica_$(printf '%02d' "$index").log"
    set +e
    CUDA_VISIBLE_DEVICES="$GPU_INDEX" "$CONDA_EXE" run --no-capture-output -n "$MD_ENV_NAME" \
      python scripts/run_md_replica_pipeline.py \
      --matrix "$MATRIX" --index "$index" --platform CUDA \
      --canonical-data-dir "$CANONICAL_DATA_DIR" \
      --residual-envelope sin2 --normal-projection-mode product \
      >> "$task_log" 2>&1
    exit_code=$?
    set -e
    printf '%s\t%s\n' "$index" "$exit_code" >> "$OUTPUT_ROOT/replica_exit_codes.tsv"
  done

  FINALIZATION_DIR="$REPLICA_DIR/finalization"
  set +e
  "$CONDA_EXE" run --no-capture-output -n "$MD_ENV_NAME" \
    python scripts/finalize_md_replica_matrix.py \
    --matrix "$MATRIX" --output-dir "$FINALIZATION_DIR" \
    > "$LOG_ROOT/finalization.log" 2>&1
  finalization_exit=$?
  set -e
  if [[ "$finalization_exit" -eq 0 ]]; then
    eligible="$(python3 -c 'import json,sys; s=json.load(open(sys.argv[1])); print(sum(v.get("target_passed", 0) >= 2 for v in s["per_system"].values()))' "$FINALIZATION_DIR/summary.json")"
    if [[ "$eligible" -gt 0 ]]; then
      set +e
      "$CONDA_EXE" run --no-capture-output -n "$MD_ENV_NAME" \
        python scripts/build_md_phase_normal_consensus_cache.py \
        --input-cache "$FINALIZATION_DIR/phase_normal_cache" \
        --output-cache "$FINALIZATION_DIR/consensus_cache" \
        --min-replicas 2 --min-support-fraction 0.5 \
        > "$LOG_ROOT/consensus.log" 2>&1
      consensus_exit=$?
      set -e
    fi
  fi
fi

python3 - "$OUTPUT_ROOT" "$LOG_ROOT" "$PANEL_MANIFEST" "$CONTEXT_COLLECTION/summary.json" \
  "$finalization_exit" "$consensus_exit" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

output_root, log_root, panel_manifest, context_summary = map(Path, sys.argv[1:5])
finalization_exit, consensus_exit = map(int, sys.argv[5:7])
def load(path):
    return json.loads(path.read_text()) if path.is_file() else None
finalization = load(output_root / "replicas" / "finalization" / "summary.json")
consensus = load(output_root / "replicas" / "finalization" / "consensus_cache" / "summary.json")
state = {
    "schema_version": "bindrae_ahoj_mapping_smoke_gpu33_v1",
    "panel_manifest": str(panel_manifest),
    "panel_manifest_sha256": hashlib.sha256(panel_manifest.read_bytes()).hexdigest(),
    "context": load(context_summary),
    "finalization_exit_code": finalization_exit,
    "consensus_exit_code": consensus_exit,
    "finalization": finalization,
    "consensus": consensus,
    "logs": str(log_root),
    "claim_boundary": "A mapping-aware engineering smoke, not a training corpus or a physical-kinetics result.",
}
(output_root / "smoke_state.json").write_text(json.dumps(state, indent=2, sort_keys=True) + "\n")
print(json.dumps({"context": (state["context"] or {}).get("outcome_counts"), "replicas": (finalization or {}).get("outcome_counts"), "consensus_systems": (consensus or {}).get("consensus_systems")}, sort_keys=True))
PY

echo "Done: $(date --iso-8601=seconds)"
echo "State: $OUTPUT_ROOT/smoke_state.json"
