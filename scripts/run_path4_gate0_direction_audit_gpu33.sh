#!/usr/bin/env bash
# Direct exact-cache directional audit for the completed reserve7 development run.

set -euo pipefail
unset LD_PRELOAD

ROOT="${ROOT:-/mnt/inaisfs/data/home/zhaozc_criait/XinxiangWang/BINDRAE}"
GPU_INDEX="${GPU_INDEX:-auto}"
MIN_FREE_GPU_MEMORY_MB="${MIN_FREE_GPU_MEMORY_MB:-12000}"
SOURCE_TAG="${SOURCE_TAG:-path4_gate0_full_reserve7_ref250_gpu33_20260724_222939_v2}"
TAG="${TAG:-path4_gate0_direction_exactjvp_4sys_gpu33_$(date +%Y%m%d_%H%M%S)_v2}"
OUTPUT_ROOT="${OUTPUT_ROOT:-logs/stage2/path4_gate0/$TAG}"
SOURCE_ROOT="logs/stage2/path4_gate0/$SOURCE_TAG"
SAMPLES_FILE="${SAMPLES_FILE:-logs/stage2/path4_gate0/path4_gate0_preflight_reserve_train24_gpu33_20260722_011109_v1/valid_primary_samples.txt}"
AUDIT_SAMPLE_IDS="6hfx-A-DMU-201 2qje-D-Z8T-2 6lr4-C-CLR-301 8iy2-E-3AM-204"
PREPARED_SYSTEMS_DIR="${PREPARED_SYSTEMS_DIR:-processed_data/md_transition/context_full500_endpointdedup_20260716_v2/systems}"
CONDA_EXE="${CONDA_EXE:-/mnt/inaisfs/data/home/zhaozc_criait/miniconda3/bin/conda}"

if [[ "$PREPARED_SYSTEMS_DIR" != /* ]]; then
  PREPARED_SYSTEMS_DIR="$ROOT/$PREPARED_SYSTEMS_DIR"
fi

if [[ "$(hostname -s)" != "gpu33" && "${ALLOW_NON_GPU33:-0}" != "1" ]]; then
  echo "ERROR: this direct launcher is restricted to gpu33" >&2
  exit 2
fi

cd "$ROOT"
if [[ ! -f "$SOURCE_ROOT/panel_state.json" || ! -f "$SAMPLES_FILE" ]]; then
  echo "ERROR: reserve7 source artifacts are incomplete" >&2
  exit 2
fi
read -r -a AUDIT_SAMPLES <<< "$AUDIT_SAMPLE_IDS"
SAMPLE_COUNT="${#AUDIT_SAMPLES[@]}"
if [[ "$SAMPLE_COUNT" -ne 4 ]]; then
  echo "ERROR: direction audit requires four discriminating systems" >&2
  exit 2
fi
for SAMPLE_ID in "${AUDIT_SAMPLES[@]}"; do
  if ! grep -Fxq "$SAMPLE_ID" "$SAMPLES_FILE"; then
    echo "ERROR: audit system is absent from the frozen reserve panel: $SAMPLE_ID" >&2
    exit 2
  fi
done

if [[ "$GPU_INDEX" == "auto" ]]; then
  GPU_INDEX="$(
    nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits |
      awk -F, -v minimum="$MIN_FREE_GPU_MEMORY_MB" '
        {
          gsub(/[[:space:]]/, "", $1)
          gsub(/[[:space:]]/, "", $2)
          if (($2 + 0) >= minimum && ($2 + 0) > best) {
            selected_index = $1
            best = $2 + 0
          }
        }
        END { if (selected_index != "") print selected_index }
      '
  )"
fi
if ! [[ "$GPU_INDEX" =~ ^[0-9]+$ ]]; then
  echo "ERROR: no GPU has at least $MIN_FREE_GPU_MEMORY_MB MiB free" >&2
  exit 3
fi
GPU_FREE_MB="$(nvidia-smi --id="$GPU_INDEX" --query-gpu=memory.free --format=csv,noheader,nounits | tr -d ' ')"
if [[ -z "$GPU_FREE_MB" || "$GPU_FREE_MB" -lt "$MIN_FREE_GPU_MEMORY_MB" ]]; then
  echo "ERROR: GPU $GPU_INDEX has ${GPU_FREE_MB:-unknown} MiB free; need $MIN_FREE_GPU_MEMORY_MB MiB" >&2
  exit 3
fi

mkdir -p "$OUTPUT_ROOT/systems" logs/slurm
SELECTION_FILE="$OUTPUT_ROOT/audit_selection.tsv"
{
  printf 'sample_id\taudit_stratum\n'
  printf '%s\t%s\n' '6hfx-A-DMU-201' 'cache-sensitive raw reversal and CUDA force saturation'
  printf '%s\t%s\n' '2qje-D-Z8T-2' 'moderate cache sensitivity near raw improvement threshold'
  printf '%s\t%s\n' '6lr4-C-CLR-301' 'exact-cache control with high unsaturated force and carbonyl conflict'
  printf '%s\t%s\n' '8iy2-E-3AM-204' 'exact-cache low-force control with accepted optimizer steps'
} > "$SELECTION_FILE"
export CUDA_VISIBLE_DEVICES="$GPU_INDEX"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"

echo "Path-4 exact-cache optimizer direction audit"
echo "Host:        $(hostname -s)"
echo "Physical GPU:$GPU_INDEX (${GPU_FREE_MB} MiB free before launch)"
echo "Source:      $SOURCE_TAG"
echo "Tag:         $TAG"
echo "Systems:     $SAMPLE_COUNT"
echo "Sample IDs:  $AUDIT_SAMPLE_IDS"
echo "Selection:   $SELECTION_FILE"
echo "Start:       $(date --iso-8601=seconds)"

FAILURES=0
STATUS_FILE="$OUTPUT_ROOT/run_status.tsv"
printf 'sample_id\tstatus\treport\n' > "$STATUS_FILE"
for SAMPLE_ID in "${AUDIT_SAMPLES[@]}"; do
  SYSTEM_ROOT="$SOURCE_ROOT/systems/$SAMPLE_ID"
  REPORT="$OUTPUT_ROOT/systems/$SAMPLE_ID/direction_audit.json"
  mkdir -p "$(dirname "$REPORT")"
  echo "$(date --iso-8601=seconds) $SAMPLE_ID start"
  if "$CONDA_EXE" run --no-capture-output -n BINDRAE-MD python \
    scripts/diagnose_path4_openmm_optimizer_direction.py \
    --candidate "$SYSTEM_ROOT/candidates/path3/$SAMPLE_ID.npz" \
    --preparation-report "$PREPARED_SYSTEMS_DIR/$SAMPLE_ID/setup/preparation_report.json" \
    --implicit-cache-dir "$SOURCE_ROOT/implicit_cache/${SAMPLE_ID}_gbn2_v1" \
    --frame-reference-cache "$SYSTEM_ROOT/reference/path3_all_atom_reference.npz" \
    --source-optimizer-report "$SYSTEM_ROOT/optimizer/report.json" \
    --report "$REPORT" \
    --project-root "$ROOT" \
    --platform CUDA \
    --device-index 0 \
    --cpu-threads "$OMP_NUM_THREADS"; then
    printf '%s\tcompleted\t%s\n' "$SAMPLE_ID" "$REPORT" >> "$STATUS_FILE"
    echo "$(date --iso-8601=seconds) $SAMPLE_ID completed"
  else
    printf '%s\tfailed\t%s\n' "$SAMPLE_ID" "$REPORT" >> "$STATUS_FILE"
    echo "$(date --iso-8601=seconds) $SAMPLE_ID failed" >&2
    FAILURES=$((FAILURES + 1))
  fi
done

echo "Done:        $(date --iso-8601=seconds)"
echo "Reports:     $OUTPUT_ROOT/systems/*/direction_audit.json"
echo "Status:      $STATUS_FILE"
if [[ "$FAILURES" -ne 0 ]]; then
  echo "ERROR: $FAILURES direction audit systems failed" >&2
  exit 1
fi
