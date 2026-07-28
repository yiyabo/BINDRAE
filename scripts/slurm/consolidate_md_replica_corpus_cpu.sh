#!/usr/bin/env bash
# Validate finalized replicas and build an immutable system-level consensus cache.

#SBATCH --job-name=mdrep_consolidate
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=logs/slurm/md_replica_consolidate_%j.out
#SBATCH --error=logs/slurm/md_replica_consolidate_%j.err

set -euo pipefail
unset LD_PRELOAD

ROOT="${ROOT:-/mnt/inaisfs/data/home/zhaozc_criait/XinxiangWang/BINDRAE}"
REPLICA_OUTPUT_DIR="${REPLICA_OUTPUT_DIR:?Set REPLICA_OUTPUT_DIR}"
CONTEXT_SUMMARY="${CONTEXT_SUMMARY:-}"
STATE_OUTPUT="${STATE_OUTPUT:-$REPLICA_OUTPUT_DIR/corpus_state.json}"
MIN_REPLICAS="${MIN_REPLICAS:-2}"
MIN_SUPPORT_FRACTION="${MIN_SUPPORT_FRACTION:-0.5}"
REFERENCE_CONSENSUS_SYSTEMS="${REFERENCE_CONSENSUS_SYSTEMS:-}"
PRECHECK_ONLY="${PRECHECK_ONLY:-0}"

cd "$ROOT"
mkdir -p logs/slurm "$(dirname "$STATE_OUTPUT")"

source /mnt/inaisfs/data/home/zhaozc_criait/miniconda3/etc/profile.d/conda.sh
conda activate BINDRAE-MD
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"

ARGS=(
  --finalization-dir "$REPLICA_OUTPUT_DIR/finalization"
  --state-output "$STATE_OUTPUT"
  --min-replicas "$MIN_REPLICAS"
  --min-support-fraction "$MIN_SUPPORT_FRACTION"
)
if [[ -n "$CONTEXT_SUMMARY" ]]; then
  ARGS+=(--context-summary "$CONTEXT_SUMMARY")
fi
if [[ -n "$REFERENCE_CONSENSUS_SYSTEMS" ]]; then
  ARGS+=(--reference-consensus-systems "$REFERENCE_CONSENSUS_SYSTEMS")
fi
if [[ "$PRECHECK_ONLY" == "1" ]]; then
  ARGS+=(--precheck-only)
elif [[ "$PRECHECK_ONLY" != "0" ]]; then
  echo "PRECHECK_ONLY must be 0 or 1; got $PRECHECK_ONLY" >&2
  exit 2
fi

python3 scripts/consolidate_md_replica_corpus.py "${ARGS[@]}"
