#!/usr/bin/env bash
# Parallel (Slurm CPU array) re-launch of the mapping-aware AHoJ 32x2 smoke.
#
# This preserves the EXACT scientific definition of the serial direct-GPU33
# launcher scripts/run_ahoj_mapping_smoke_gpu33.sh:
#   - identical panel (reused frozen manifest by default),
#   - identical context seed-base 2026072500 and protocol tag,
#   - identical replica seed-base 2026072600, 0.95 mapping gate, replica range
#     [0, 1], and the frozen pull protocol (500/10000/2000/200000/0.025).
#
# Only the execution model changes: the two serial CUDA for-loops become
# cross-node CPU Slurm arrays. Context prep is CPU-bound (the GPU sat at 0%
# utilisation in the serial run), so CPU arrays are the correct scale-out.
#
# It reuses the tested infrastructure:
#   scripts/slurm/run_md_context_pipeline_array_cpu.sh  (context array)
#   scripts/slurm/continue_md_pilot_after_context_cpu.sh (collect -> replica
#     array -> finalize dependency chain).
# The trailing consensus + smoke_state.json step is produced by the companion
# scripts/slurm/consolidate_ahoj_smoke_cpu.sh after finalization completes.

#SBATCH --job-name=ahoj_smoke_launch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=logs/slurm/ahoj_smoke_launch_%j.out
#SBATCH --error=logs/slurm/ahoj_smoke_launch_%j.err

set -euo pipefail
unset LD_PRELOAD

ROOT="${ROOT:-/mnt/inaisfs/data/home/zhaozc_criait/XinxiangWang/BINDRAE}"
CANDIDATE_MANIFEST="${CANDIDATE_MANIFEST:-processed_data/md_transition/path3_capacity_leakclean_mapping95_20260725_v1/novel_transition_manifest.jsonl}"
# Reuse the already-frozen panel from the serial run by default so the 32
# systems are byte-identical. Leave empty to re-select deterministically.
SOURCE_PANEL="${SOURCE_PANEL:-processed_data/md_transition/ahoj_mapping_smoke32x2_gpu33_20260725_v1/panel/mapping_smoke_panel_manifest.jsonl}"
OUTPUT_ROOT="${OUTPUT_ROOT:-processed_data/md_transition/ahoj_mapping_smoke32x2_parallel_20260725_v1}"
LOG_ROOT="${LOG_ROOT:-logs/stage2/ahoj_mapping_smoke32x2_parallel_20260725_v1}"
CANONICAL_DATA_DIR="${CANONICAL_DATA_DIR:-processed_data/triplets}"
PANEL_SIZE="${PANEL_SIZE:-32}"
CONTEXT_MAX_CONCURRENT="${CONTEXT_MAX_CONCURRENT:-32}"
REPLICA_MAX_CONCURRENT="${REPLICA_MAX_CONCURRENT:-32}"
CONTEXT_SEED_BASE="${CONTEXT_SEED_BASE:-2026072500}"
REPLICA_SEED_BASE="${REPLICA_SEED_BASE:-2026072600}"
PROTOCOL_TAG="${PROTOCOL_TAG:-ahoj_mapping_smoke32_v1}"
MIN_MAPPING_FRACTION="${MIN_MAPPING_FRACTION:-0.95}"
REPLICA_START="${REPLICA_START:-0}"
REPLICA_STOP="${REPLICA_STOP:-1}"
# Frozen pull protocol (identical to the serial smoke defaults).
PRE_EQUILIBRATION_STEPS="${PRE_EQUILIBRATION_STEPS:-500}"
PULLING_STEPS="${PULLING_STEPS:-10000}"
ENDPOINT_HOLD_STEPS="${ENDPOINT_HOLD_STEPS:-2000}"
REPORT_INTERVAL="${REPORT_INTERVAL:-100}"
RMSD_K_KJ_MOL_NM2="${RMSD_K_KJ_MOL_NM2:-200000}"
FINAL_TARGET_RMSD_NM="${FINAL_TARGET_RMSD_NM:-0.025}"
CONTEXT_JOB_NAME="${CONTEXT_JOB_NAME:-ahoj_ctx}"
REPLICA_JOB_NAME="${REPLICA_JOB_NAME:-ahoj_rep}"

for value_name in PANEL_SIZE CONTEXT_MAX_CONCURRENT REPLICA_MAX_CONCURRENT REPLICA_START REPLICA_STOP; do
  value="${!value_name}"
  if ! [[ "$value" =~ ^[0-9]+$ ]]; then
    echo "ERROR: $value_name must be a non-negative integer: $value" >&2
    exit 2
  fi
done
if (( REPLICA_STOP < REPLICA_START )); then
  echo "ERROR: REPLICA_STOP must be >= REPLICA_START" >&2
  exit 2
fi

export PATH="/data/soft/slurm/24.11.4/bin:${PATH}"
cd "$ROOT"

if [[ ! -s "$CANDIDATE_MANIFEST" ]]; then
  echo "ERROR: missing candidate manifest: $CANDIDATE_MANIFEST" >&2
  exit 2
fi
if [[ -e "$OUTPUT_ROOT" ]]; then
  echo "ERROR: output tag already exists; use a fresh OUTPUT_ROOT: $OUTPUT_ROOT" >&2
  exit 3
fi

mkdir -p logs/slurm "$OUTPUT_ROOT" "$LOG_ROOT"

source /mnt/inaisfs/data/home/zhaozc_criait/miniconda3/etc/profile.d/conda.sh
conda activate BINDRAE-MD
export PYTHONUNBUFFERED=1

PANEL_DIR="$OUTPUT_ROOT/panel"
PANEL_MANIFEST="$PANEL_DIR/mapping_smoke_panel_manifest.jsonl"
mkdir -p "$PANEL_DIR"
if [[ -n "$SOURCE_PANEL" && -s "$SOURCE_PANEL" ]]; then
  echo "Reusing frozen panel: $SOURCE_PANEL"
  cp "$SOURCE_PANEL" "$PANEL_MANIFEST"
else
  echo "Selecting fresh deterministic panel (size $PANEL_SIZE)"
  python scripts/select_md_mapping_smoke_panel.py \
    --candidate-manifest "$CANDIDATE_MANIFEST" \
    --output-dir "$PANEL_DIR" \
    --panel-size "$PANEL_SIZE" \
    > "$LOG_ROOT/select_panel.log" 2>&1
fi
if [[ ! -s "$PANEL_MANIFEST" ]]; then
  echo "ERROR: panel manifest missing or empty: $PANEL_MANIFEST" >&2
  exit 4
fi

CONTEXT_DIR="$OUTPUT_ROOT/context"
CONTEXT_MATRIX="$CONTEXT_DIR/context_matrix.jsonl"
python scripts/build_md_context_matrix.py \
  --candidate-manifest "$PANEL_MANIFEST" \
  --output-dir "$CONTEXT_DIR" \
  --seed-base "$CONTEXT_SEED_BASE" \
  --protocol-tag "$PROTOCOL_TAG" \
  > "$LOG_ROOT/build_context_matrix.log" 2>&1

CONTEXT_TASKS="$(wc -l < "$CONTEXT_MATRIX" | tr -d ' ')"
if [[ "$CONTEXT_TASKS" -le 0 ]]; then
  echo "ERROR: empty context matrix: $CONTEXT_MATRIX" >&2
  exit 4
fi
CONTEXT_LAST=$((CONTEXT_TASKS - 1))

CONTEXT_JOB=$(sbatch --parsable \
  --job-name="$CONTEXT_JOB_NAME" \
  --array="0-${CONTEXT_LAST}%${CONTEXT_MAX_CONCURRENT}" \
  --export=ALL,MATRIX="$CONTEXT_MATRIX",PLATFORM=CPU \
  scripts/slurm/run_md_context_pipeline_array_cpu.sh)

COLLECTION_DIR="$CONTEXT_DIR/collection"
REPLICA_DIR="$OUTPUT_ROOT/replicas"
CONTINUE_JOB=$(sbatch --parsable \
  --dependency="afterany:${CONTEXT_JOB}" \
  --export=ALL,CONTEXT_MATRIX="$CONTEXT_MATRIX",CANDIDATE_MANIFEST="$PANEL_MANIFEST",COLLECTION_DIR="$COLLECTION_DIR",REPLICA_OUTPUT_DIR="$REPLICA_DIR",MAX_CONCURRENT="$REPLICA_MAX_CONCURRENT",SEED_BASE="$REPLICA_SEED_BASE",PROTOCOL_TAG="$PROTOCOL_TAG",PRE_EQUILIBRATION_STEPS="$PRE_EQUILIBRATION_STEPS",PULLING_STEPS="$PULLING_STEPS",ENDPOINT_HOLD_STEPS="$ENDPOINT_HOLD_STEPS",REPORT_INTERVAL="$REPORT_INTERVAL",RMSD_K_KJ_MOL_NM2="$RMSD_K_KJ_MOL_NM2",FINAL_TARGET_RMSD_NM="$FINAL_TARGET_RMSD_NM",MIN_MAPPING_FRACTION="$MIN_MAPPING_FRACTION",REPLICA_START="$REPLICA_START",REPLICA_STOP="$REPLICA_STOP",REPLICA_JOB_NAME="$REPLICA_JOB_NAME" \
  scripts/slurm/continue_md_pilot_after_context_cpu.sh)

python3 - "$OUTPUT_ROOT/submission.json" "$CONTEXT_JOB" "$CONTINUE_JOB" \
  "$CONTEXT_TASKS" "$PANEL_MANIFEST" "$CONTEXT_MATRIX" "$REPLICA_DIR" <<'PY'
import json
import sys
from pathlib import Path

(
    path,
    context_job,
    continue_job,
    context_tasks,
    panel_manifest,
    context_matrix,
    replica_dir,
) = sys.argv[1:]
record = {
    "schema_version": "bindrae_ahoj_mapping_smoke_parallel_submission_v1",
    "execution_model": "slurm_cpu_arrays",
    "context_job_id": context_job,
    "continue_job_id": continue_job,
    "context_tasks": int(context_tasks),
    "panel_manifest": panel_manifest,
    "context_matrix": context_matrix,
    "replica_output_dir": replica_dir,
    "finalization_summary": str(Path(replica_dir) / "finalization" / "summary.json"),
    "note": (
        "Consensus cache and smoke_state.json are produced by "
        "scripts/slurm/consolidate_ahoj_smoke_cpu.sh after finalization."
    ),
}
Path(path).write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
print(json.dumps(record, indent=2, sort_keys=True))
PY

echo "Context array job: $CONTEXT_JOB (0-${CONTEXT_LAST}%${CONTEXT_MAX_CONCURRENT})"
echo "Continuation job:  $CONTINUE_JOB (afterany:${CONTEXT_JOB})"
echo "Output root:       $OUTPUT_ROOT"
echo "Completion:        $REPLICA_DIR/finalization/summary.json, then run consolidate_ahoj_smoke_cpu.sh"
