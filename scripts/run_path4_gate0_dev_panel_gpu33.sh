#!/usr/bin/env bash
# Direct, resumable Gate-0 development panel for the private gpu33 node.

set -euo pipefail
unset LD_PRELOAD

ROOT="${ROOT:-/mnt/inaisfs/data/home/zhaozc_criait/XinxiangWang/BINDRAE}"
GPU_INDEX="${GPU_INDEX:-3}"
MIN_FREE_GPU_MEMORY_MB="${MIN_FREE_GPU_MEMORY_MB:-30000}"
PANEL_SIZE="${PANEL_SIZE:-12}"
TAG="${TAG:-path4_gate0_dev_panel_train11_sentinel1_gpu33_$(date +%Y%m%d_%H%M%S)_v1}"
OUTPUT_ROOT="${OUTPUT_ROOT:-logs/stage2/path4_gate0/$TAG}"
CHECKPOINT="${CHECKPOINT:-checkpoints/stage2/stage2_pbv2_phase_cons303_fam30scaf_e10_bs8x2_20260718_canonical_fromscratch_v2/best_model.pt}"
PHASE_CACHE_DIR="${PHASE_CACHE_DIR:-logs/stage2/physical_normal_targets_cons303_rw20_20260719_v1}"
MD_REFERENCE_CACHE_DIR="${MD_REFERENCE_CACHE_DIR:-processed_data/md_transition/phase_block_cache_sin2_inferred1340_20260718_v3}"
PREPARED_SYSTEMS_DIR="${PREPARED_SYSTEMS_DIR:-processed_data/md_transition/context_full500_endpointdedup_20260716_v2/systems}"
TRAIN_SAMPLES_FILE="${TRAIN_SAMPLES_FILE:-processed_data/triplets/ablation_subsets/stage2_oracle_motion_mdphase_consensus303_fam30scaf_blockv2_train_241_seed20260717.txt}"
VAL_SAMPLES_FILE="${VAL_SAMPLES_FILE:-processed_data/triplets/ablation_subsets/stage2_oracle_motion_mdphase_consensus303_fam30scaf_blockv2_val_30_seed20260717.txt}"
TEST_SAMPLES_FILE="${TEST_SAMPLES_FILE:-processed_data/triplets/ablation_subsets/stage2_oracle_motion_mdphase_consensus303_fam30scaf_blockv2_test_32_seed20260717.txt}"
STRICT30_SAMPLES_FILE="${STRICT30_SAMPLES_FILE:-processed_data/triplets/ablation_subsets/stage2_oracle_motion_mdphase_consensus303_fam30scaf_blockv2_test_strict30_20260720.txt}"
POSTSELECTION_SAMPLES_FILE="${POSTSELECTION_SAMPLES_FILE:-processed_data/triplets/ablation_subsets/stage2_oracle_motion_mdphase_consensus303_physical_calib12_seed20260719.txt}"
CONDA_EXE="${CONDA_EXE:-/mnt/inaisfs/data/home/zhaozc_criait/miniconda3/bin/conda}"

if [[ "$(hostname -s)" != "gpu33" && "${ALLOW_NON_GPU33:-0}" != "1" ]]; then
  echo "ERROR: this direct launcher is restricted to gpu33" >&2
  exit 2
fi
if ! [[ "$GPU_INDEX" =~ ^[0-9]+$ ]]; then
  echo "ERROR: GPU_INDEX must be a non-negative integer" >&2
  exit 2
fi
GPU_FREE_MB="$(nvidia-smi --id="$GPU_INDEX" --query-gpu=memory.free --format=csv,noheader,nounits | tr -d ' ')"
if [[ -z "$GPU_FREE_MB" || "$GPU_FREE_MB" -lt "$MIN_FREE_GPU_MEMORY_MB" ]]; then
  echo "ERROR: GPU $GPU_INDEX has ${GPU_FREE_MB:-unknown} MiB free; need $MIN_FREE_GPU_MEMORY_MB MiB" >&2
  exit 3
fi

cd "$ROOT"
mkdir -p logs/slurm
export CUDA_VISIBLE_DEVICES="$GPU_INDEX"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"

ARGS=(
  --project-root "$ROOT"
  --checkpoint "$CHECKPOINT"
  --data-dir processed_data/triplets
  --train-samples-file "$TRAIN_SAMPLES_FILE"
  --exclude-samples-file "$VAL_SAMPLES_FILE"
  --exclude-samples-file "$TEST_SAMPLES_FILE"
  --exclude-samples-file "$STRICT30_SAMPLES_FILE"
  --exclude-samples-file "$POSTSELECTION_SAMPLES_FILE"
  --phase-cache-dir "$PHASE_CACHE_DIR"
  --md-reference-cache-dir "$MD_REFERENCE_CACHE_DIR"
  --prepared-systems-dir "$PREPARED_SYSTEMS_DIR"
  --output-root "$OUTPUT_ROOT"
  --panel-size "$PANEL_SIZE"
  --sentinel 3zw1-E-FUC-3
  --minimum-residues "${MINIMUM_RESIDUES:-60}"
  --maximum-residues "${MAXIMUM_RESIDUES:-400}"
  --platform CUDA
  --device-index 0
  --cpu-threads "${OMP_NUM_THREADS}"
  --execution-mode full
  --conda-executable "$CONDA_EXE"
)
if [[ "${RESUME:-0}" == "1" ]]; then
  ARGS+=(--resume)
fi

echo "Path-4 Gate-0 training-scope development panel"
echo "Host:        $(hostname -s)"
echo "Physical GPU:$GPU_INDEX (${GPU_FREE_MB} MiB free before launch)"
echo "Panel size:  $PANEL_SIZE (sentinel excluded from primary aggregate)"
echo "Tag:         $TAG"
echo "Output:      $OUTPUT_ROOT"
echo "Start:       $(date --iso-8601=seconds)"

"$CONDA_EXE" run --no-capture-output -n BINDRAE python \
  scripts/run_path4_gate0_dev_panel.py "${ARGS[@]}"

echo "Done:        $(date --iso-8601=seconds)"
echo "State:       $OUTPUT_ROOT/panel_state.json"
echo "Primary:     $OUTPUT_ROOT/aggregate/paired_summary_primary.json"
