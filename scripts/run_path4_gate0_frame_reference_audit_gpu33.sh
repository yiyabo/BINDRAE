#!/usr/bin/env bash
# Diagnostic-only rebuild of the six reserve frame references rejected at 1e6.

set -euo pipefail
unset LD_PRELOAD

ROOT="${ROOT:-/mnt/inaisfs/data/home/zhaozc_criait/XinxiangWang/BINDRAE}"
SOURCE_TAG="${SOURCE_TAG:-path4_gate0_full_reserve7_gpu33_20260723_011300_v3}"
SOURCE_ROOT="${SOURCE_ROOT:-logs/stage2/path4_gate0/$SOURCE_TAG}"
TAG="${TAG:-path4_gate0_frame_reference_audit_reserve6_gpu33_$(date +%Y%m%d_%H%M%S)_v1}"
OUTPUT_ROOT="${OUTPUT_ROOT:-logs/stage2/path4_gate0/$TAG}"
PREPARED_SYSTEMS_DIR="${PREPARED_SYSTEMS_DIR:-processed_data/md_transition/context_full500_endpointdedup_20260716_v2/systems}"
CONDA_EXE="${CONDA_EXE:-/mnt/inaisfs/data/home/zhaozc_criait/miniconda3/bin/conda}"
GPU_INDEX="${GPU_INDEX:-auto}"
MIN_FREE_GPU_MEMORY_MB="${MIN_FREE_GPU_MEMORY_MB:-12000}"
OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
PREPARED_FORCE_THRESHOLD="${PREPARED_FORCE_THRESHOLD:-1000000}"
DIAGNOSTIC_FRAME_FORCE_THRESHOLD="${DIAGNOSTIC_FRAME_FORCE_THRESHOLD:-1000000000000}"
ITERATION_COUNTS="${ITERATION_COUNTS:-25}"
SAMPLE_IDS="${SAMPLE_IDS:-}"

DEFAULT_SAMPLES=(
  2e2o-A-BGC-400
  2qje-D-Z8T-2
  4wq2-B-3SU-301
  6hfx-A-DMU-201
  6lr4-C-CLR-301
  7mql-A-RIO-302
)
if [[ -n "$SAMPLE_IDS" ]]; then
  read -r -a SAMPLES <<<"$SAMPLE_IDS"
else
  SAMPLES=("${DEFAULT_SAMPLES[@]}")
fi

if [[ "$(hostname -s)" != "gpu33" && "${ALLOW_NON_GPU33:-0}" != "1" ]]; then
  echo "ERROR: this direct launcher is restricted to gpu33" >&2
  exit 2
fi
if [[ "$PREPARED_FORCE_THRESHOLD" != "1000000" ]]; then
  echo "ERROR: prepared-state threshold is frozen at 1000000 kJ/mol/nm" >&2
  exit 2
fi
for iterations in $ITERATION_COUNTS; do
  if ! [[ "$iterations" =~ ^[1-9][0-9]*$ ]]; then
    echo "ERROR: ITERATION_COUNTS must contain positive integers" >&2
    exit 2
  fi
done

cd "$ROOT"
if [[ "$PREPARED_SYSTEMS_DIR" != /* ]]; then
  PREPARED_SYSTEMS_DIR="$ROOT/$PREPARED_SYSTEMS_DIR"
fi
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
GPU_FREE_MB="$(
  nvidia-smi --id="$GPU_INDEX" --query-gpu=memory.free \
    --format=csv,noheader,nounits | tr -d ' '
)"
if [[ -z "$GPU_FREE_MB" || "$GPU_FREE_MB" -lt "$MIN_FREE_GPU_MEMORY_MB" ]]; then
  echo "ERROR: GPU $GPU_INDEX has ${GPU_FREE_MB:-unknown} MiB free; need $MIN_FREE_GPU_MEMORY_MB MiB" >&2
  exit 3
fi

export CUDA_VISIBLE_DEVICES="$GPU_INDEX"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS
mkdir -p "$OUTPUT_ROOT"

echo "Path-4 Gate-0 frame-reference audit"
echo "Host:                       $(hostname -s)"
echo "Physical GPU:               $GPU_INDEX ($GPU_FREE_MB MiB free before launch)"
echo "Source:                     $SOURCE_ROOT"
echo "Output:                     $OUTPUT_ROOT"
echo "Prepared threshold:         $PREPARED_FORCE_THRESHOLD kJ/mol/nm (frozen)"
echo "Diagnostic frame threshold: $DIAGNOSTIC_FRAME_FORCE_THRESHOLD kJ/mol/nm"
echo "Relaxation iterations:      $ITERATION_COUNTS"
echo "Samples:                    ${SAMPLES[*]}"
echo "Start:                      $(date --iso-8601=seconds)"

failed=0
for iterations in $ITERATION_COUNTS; do
  for sample_id in "${SAMPLES[@]}"; do
    safe_id="${sample_id//[^A-Za-z0-9_.-]/_}"
    candidate="$SOURCE_ROOT/systems/$safe_id/candidates/path3/$safe_id.npz"
    preparation_report="$PREPARED_SYSTEMS_DIR/$safe_id/setup/preparation_report.json"
    implicit_cache="$SOURCE_ROOT/implicit_cache/${safe_id}_gbn2_v1"
    sample_root="$OUTPUT_ROOT/iterations_${iterations}/systems/$safe_id"
    report="$sample_root/report.json"
    cache="$sample_root/path3_all_atom_reference_diagnostic.npz"
    log="$sample_root/build.log"
    mkdir -p "$sample_root"

    if [[ ! -f "$candidate" || ! -f "$preparation_report" ]]; then
      echo "ERROR: missing source input for $sample_id" | tee "$log" >&2
      failed=$((failed + 1))
      continue
    fi
    echo "[$(date --iso-8601=seconds)] $sample_id iterations=$iterations"
    if "$CONDA_EXE" run --no-capture-output -n BINDRAE-MD python \
      scripts/build_path4_openmm_frame_reference.py \
      --candidate "$candidate" \
      --preparation-report "$preparation_report" \
      --implicit-cache-dir "$implicit_cache" \
      --output-cache "$cache" \
      --report "$report" \
      --platform CUDA \
      --device-index 0 \
      --cpu-threads "$OMP_NUM_THREADS" \
      --reference-relaxation-iterations "$iterations" \
      --reference-restraint-k-kj-mol-nm2 100000 \
      --reference-minimization-tolerance-kj-mol-nm 500 \
      --minimum-residue-mapping 0.98 \
      --minimum-atom-mapping 0.95 \
      --maximum-reference-atomic-force-kj-mol-nm "$PREPARED_FORCE_THRESHOLD" \
      --maximum-frame-reference-atomic-force-kj-mol-nm \
        "$DIAGNOSTIC_FRAME_FORCE_THRESHOLD" \
      --diagnostic-atom-force-threshold-kj-mol-nm \
        "$PREPARED_FORCE_THRESHOLD" \
      --diagnostic-top-force-atoms 5 >"$log" 2>&1; then
      echo "[$(date --iso-8601=seconds)] $sample_id completed"
    else
      echo "[$(date --iso-8601=seconds)] $sample_id diagnostic build failed" >&2
      failed=$((failed + 1))
    fi
  done
done

python3 scripts/summarize_path4_frame_reference_audit.py \
  --reports-root "$OUTPUT_ROOT" \
  --output "$OUTPUT_ROOT/summary.json" \
  --scientific-frame-threshold-kj-mol-nm "$PREPARED_FORCE_THRESHOLD"

echo "Done:    $(date --iso-8601=seconds)"
echo "Summary: $OUTPUT_ROOT/summary.json"
echo "Failures:$failed"
if [[ "$failed" -ne 0 ]]; then
  exit 1
fi
