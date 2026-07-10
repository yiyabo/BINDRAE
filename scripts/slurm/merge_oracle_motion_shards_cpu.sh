#!/usr/bin/env bash
# Merge independently exported OracleMotion shard manifests.

#SBATCH --job-name=merge_om_shards
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=logs/slurm/merge_oracle_motion_%j.out
#SBATCH --error=logs/slurm/merge_oracle_motion_%j.err

set -euo pipefail

ROOT="${ROOT:-/mnt/inaisfs/data/home/zhaozc_criait/XinxiangWang/BINDRAE}"
CANDIDATE_LIST="${CANDIDATE_LIST:?Set CANDIDATE_LIST}"
SHARD_DIRS="${SHARD_DIRS:?Set colon-separated SHARD_DIRS}"
OUTPUT_DIR="${OUTPUT_DIR:?Set OUTPUT_DIR}"

cd "$ROOT"
mkdir -p logs/slurm "$OUTPUT_DIR"

source /mnt/inaisfs/data/home/zhaozc_criait/miniconda3/etc/profile.d/conda.sh
conda activate BINDRAE

IFS=':' read -r -a shard_dirs <<< "$SHARD_DIRS"
if [[ "${#shard_dirs[@]}" -lt 1 ]]; then
  echo "ERROR: SHARD_DIRS is empty"
  exit 1
fi

args=(
  scripts/build_oracle_motion_remainder.py
  --candidate_list "$CANDIDATE_LIST"
  --partial_cache_dir "${shard_dirs[0]}"
  --remainder_out "$OUTPUT_DIR/remainder.txt"
  --rejects_out "$OUTPUT_DIR/rejects.jsonl"
  --merged_manifest_out "$OUTPUT_DIR/manifest.json"
  --skip_ligand_check
  --fast_manifest
)
for shard_dir in "${shard_dirs[@]:1}"; do
  args+=(--extra_cache_dir "$shard_dir")
done

python "${args[@]}"

echo "Merged manifest: $OUTPUT_DIR/manifest.json"
echo "Remaining IDs:   $(wc -l < "$OUTPUT_DIR/remainder.txt")"
