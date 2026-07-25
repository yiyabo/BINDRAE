#!/usr/bin/env bash
# Submit parameter-matched learned-phase controls on the frozen MD split.
# The synchronous Cartesian row is analytic and is evaluated separately.

set -euo pipefail

ROOT="${ROOT:-/mnt/inaisfs/data/home/zhaozc_criait/XinxiangWang/BINDRAE}"
SEEDS="${SEEDS:-42}"
WARP_VARIANTS="${WARP_VARIANTS:-global_monotone,residue_monotone,residue_nonmonotone}"
RUN_VERSION="${RUN_VERSION:-phase_specificity_v1}"

export PATH="/data/soft/slurm/24.11.4/bin:${PATH}"
cd "$ROOT"

IFS=',' read -r -a variants <<< "$WARP_VARIANTS"
for variant in "${variants[@]}"; do
  case "$variant" in
    global_monotone|global_chain_monotone|chain_nonmonotone|residue_monotone|residue_nonmonotone) ;;
    *)
      echo "ERROR: unsupported phase specificity variant: $variant" >&2
      exit 2
      ;;
  esac
done

for seed in $SEEDS; do
  if ! [[ "$seed" =~ ^[0-9]+$ ]]; then
    echo "ERROR: seed must be a non-negative integer: $seed" >&2
    exit 2
  fi
  for variant in "${variants[@]}"; do
    echo "Submitting phase specificity control: variant=$variant seed=$seed"
    PHASE_WARP_VARIANT="$variant" \
    TRAIN_SEED="$seed" \
    RUN_VARIANTS=phase \
    RUN_VERSION="$RUN_VERSION" \
      bash scripts/slurm/submit_stage2_md_phase_normal_screen.sh
  done
done
