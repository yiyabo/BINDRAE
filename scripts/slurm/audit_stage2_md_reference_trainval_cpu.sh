#!/usr/bin/env bash
# Apply the frozen strict30 production-coordinate contract to the train241 and
# val30 consensus303 splits.
#
# The 2026-07-20 audit (scripts/audit_stage2_md_reference_subset.py) was run only
# on the 32-system test manifest and removed two systems whose production
# node_mask disagreed with the MD reference cache. The same contract has never
# been applied to the 241 training or 30 validation systems. This launcher runs
# the identical script, with the identical --require_full_node_mask flag, and
# changes no threshold, split, or cache.
#
# It is read-only with respect to every frozen artifact: all outputs go under a
# fresh TAG directory.
#
# Each split is audited against both reference caches, because they play
# different roles:
#   consensus_min2 ... the deterministic system target consumed by training;
#   inferred1340   ... the per-replica reference consumed by path evaluation.

#SBATCH --job-name=s2mdref_audit
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=logs/slurm/stage2_mdref_trainval_audit_%j.out
#SBATCH --error=logs/slurm/stage2_mdref_trainval_audit_%j.err

set -euo pipefail

ROOT="${ROOT:-/mnt/inaisfs/data/home/zhaozc_criait/XinxiangWang/BINDRAE}"
DATA_DIR="${DATA_DIR:-processed_data/triplets}"
TAG="${TAG:-trainval_contract_audit_20260726_v1}"
SUBSET_DIR="${SUBSET_DIR:-processed_data/triplets/ablation_subsets}"
PREFIX="${PREFIX:-stage2_oracle_motion_mdphase_consensus303_fam30scaf_blockv2}"
TRAIN_LIST="${TRAIN_LIST:-$SUBSET_DIR/${PREFIX}_train_241_seed20260717.txt}"
VAL_LIST="${VAL_LIST:-$SUBSET_DIR/${PREFIX}_val_30_seed20260717.txt}"
CONSENSUS_CACHE="${CONSENSUS_CACHE:-processed_data/md_transition/phase_block_cache_sin2_consensus_min2_20260718_v2}"
REPLICA_CACHE="${REPLICA_CACHE:-processed_data/md_transition/phase_block_cache_sin2_inferred1340_20260718_v3}"
OUT_DIR="${OUT_DIR:-logs/stage2/md_reference_eval/$TAG}"

cd "$ROOT"
mkdir -p logs/slurm "$OUT_DIR"

source /mnt/inaisfs/data/home/zhaozc_criait/miniconda3/etc/profile.d/conda.sh
conda activate BINDRAE

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

for path in "$TRAIN_LIST" "$VAL_LIST"; do
  if [[ ! -s "$path" ]]; then
    echo "ERROR: missing sample list: $path" >&2
    exit 2
  fi
done
for path in "$CONSENSUS_CACHE" "$REPLICA_CACHE"; do
  if [[ ! -d "$path" ]]; then
    echo "ERROR: missing reference cache: $path" >&2
    exit 2
  fi
  echo "Cache $path: $(find "$path" -maxdepth 1 -name '*.npz' | wc -l) npz files"
done

# The consensus303 split is not the same partition as the on-disk Stage-2
# splits/<split>.json index. Resolve the split whose index actually contains the
# whole manifest, and refuse to audit a silently truncated subset.
resolve_split() {
  local sample_list="$1"
  python3 - "$DATA_DIR" "$sample_list" <<'PY'
import json
import sys
from pathlib import Path

data_dir, sample_list = Path(sys.argv[1]), Path(sys.argv[2])
wanted = {line.strip() for line in sample_list.read_text().splitlines() if line.strip()}


def ids_for(path: Path, split: str):
    data = json.loads(path.read_text())
    if isinstance(data, dict):
        entries = data.get(split, [])
    else:
        entries = data
    out = set()
    for entry in entries:
        if isinstance(entry, str):
            out.add(entry)
        elif isinstance(entry, dict) and "id" in entry:
            if "split" in entry and entry["split"] != split:
                continue
            out.add(entry["id"])
    return out


for split in ("train", "val", "test", "all"):
    path = data_dir / "splits" / f"{split}.json"
    if path.exists() and wanted <= ids_for(path, split):
        print(split)
        break
else:
    index = data_dir / "index.json"
    if index.exists() and wanted <= ids_for(index, "train"):
        print("train")
    else:
        print("UNRESOLVED")
PY
}

run_audit() {
  local label="$1" sample_list="$2" cache_label="$3" cache_dir="$4"
  local split
  split="$(resolve_split "$sample_list")"
  if [[ "$split" == "UNRESOLVED" ]]; then
    echo "ERROR: no on-disk split index contains all of $sample_list" >&2
    exit 3
  fi
  local expected
  expected="$(grep -c . "$sample_list")"
  echo "=== $label / $cache_label (split=$split, expected=$expected) ==="
  python scripts/audit_stage2_md_reference_subset.py \
    --data_dir "$DATA_DIR" \
    --split "$split" \
    --sample_file "$sample_list" \
    --md_reference_cache_dir "$cache_dir" \
    --output_samples "$OUT_DIR/${label}__${cache_label}.samples.txt" \
    --output_audit "$OUT_DIR/${label}__${cache_label}.audit.json" \
    --require_full_node_mask
  python3 - "$OUT_DIR/${label}__${cache_label}.audit.json" "$expected" <<'PY'
import json
import sys
from collections import Counter

report = json.loads(open(sys.argv[1]).read())
expected = int(sys.argv[2])
passed = int(report["passed_systems"])
failed = int(report["failed_systems"])
requested = int(report["requested_systems"])
print(f"passed={passed} failed={failed} requested={requested} expected={expected}")
if requested != expected:
    print("WARNING: audited count does not match the manifest; subset was truncated")
reasons = Counter()
for row in report.get("records", []):
    for reason in row.get("reasons", []):
        reasons[reason] += 1
for reason, count in reasons.most_common():
    print(f"  {reason}: {count}")
for row in report.get("records", []):
    if not row.get("passed", True):
        print(
            f"  FAIL {row['sample_id']} "
            f"prod={row['production_valid_residues']}/{row['n_residues']} "
            f"ref={row['reference_valid_residues']} replicas={row['replicas']} "
            f"reasons={','.join(row['reasons'])}"
        )
PY
}

run_audit train241 "$TRAIN_LIST" consensus "$CONSENSUS_CACHE"
run_audit train241 "$TRAIN_LIST" replica "$REPLICA_CACHE"
run_audit val30 "$VAL_LIST" consensus "$CONSENSUS_CACHE"
run_audit val30 "$VAL_LIST" replica "$REPLICA_CACHE"

echo "Reports: $OUT_DIR"
