#!/usr/bin/env bash
# Finalize the parallel AHoJ mapping-aware smoke: build the two-replica consensus
# cache (when eligible) and write smoke_state.json in the same schema as the
# serial direct-GPU33 launcher. Run this after the continuation's finalize job
# completes (replicas/finalization/summary.json exists).

#SBATCH --job-name=ahoj_smoke_consolidate
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=logs/slurm/ahoj_smoke_consolidate_%j.out
#SBATCH --error=logs/slurm/ahoj_smoke_consolidate_%j.err

set -euo pipefail
unset LD_PRELOAD

ROOT="${ROOT:-/mnt/inaisfs/data/home/zhaozc_criait/XinxiangWang/BINDRAE}"
OUTPUT_ROOT="${OUTPUT_ROOT:-processed_data/md_transition/ahoj_mapping_smoke32x2_parallel_20260725_v1}"
LOG_ROOT="${LOG_ROOT:-logs/stage2/ahoj_mapping_smoke32x2_parallel_20260725_v1}"
PANEL_MANIFEST="${PANEL_MANIFEST:-$OUTPUT_ROOT/panel/mapping_smoke_panel_manifest.jsonl}"

cd "$ROOT"
mkdir -p logs/slurm "$LOG_ROOT"

source /mnt/inaisfs/data/home/zhaozc_criait/miniconda3/etc/profile.d/conda.sh
conda activate BINDRAE-MD
export PYTHONUNBUFFERED=1

CONTEXT_SUMMARY="$OUTPUT_ROOT/context/collection/summary.json"
REPLICA_DIR="$OUTPUT_ROOT/replicas"
FINALIZATION_DIR="$REPLICA_DIR/finalization"

if [[ ! -s "$PANEL_MANIFEST" ]]; then
  echo "ERROR: missing panel manifest: $PANEL_MANIFEST" >&2
  exit 2
fi

finalization_exit=125
consensus_exit=125
if [[ -s "$FINALIZATION_DIR/summary.json" ]]; then
  finalization_exit=0
  eligible="$(python3 -c 'import json,sys; s=json.load(open(sys.argv[1])); print(sum(v.get("target_passed", 0) >= 2 for v in s["per_system"].values()))' "$FINALIZATION_DIR/summary.json")"
  if [[ "$eligible" -gt 0 ]]; then
    set +e
    python scripts/build_md_phase_normal_consensus_cache.py \
      --input-cache "$FINALIZATION_DIR/phase_normal_cache" \
      --output-cache "$FINALIZATION_DIR/consensus_cache" \
      --min-replicas 2 --min-support-fraction 0.5 \
      > "$LOG_ROOT/consensus.log" 2>&1
    consensus_exit=$?
    set -e
  else
    echo "No system has >=2 passed replicas; skipping consensus cache."
    consensus_exit=0
  fi
else
  echo "WARNING: finalization summary not found: $FINALIZATION_DIR/summary.json" >&2
fi

python3 - "$OUTPUT_ROOT" "$LOG_ROOT" "$PANEL_MANIFEST" "$CONTEXT_SUMMARY" \
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
    "execution_model": "slurm_cpu_arrays",
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

echo "State: $OUTPUT_ROOT/smoke_state.json"
