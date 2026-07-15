#!/usr/bin/env bash
#SBATCH --job-name=md_phase_assemble
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=logs/slurm/md_phase_assemble_%j.out
#SBATCH --error=logs/slurm/md_phase_assemble_%j.err

set -euo pipefail

ROOT=${ROOT:-/mnt/inaisfs/data/home/zhaozc_criait/XinxiangWang/BINDRAE}
INPUT_ROOT=${INPUT_ROOT:?INPUT_ROOT is required}
OUTPUT_DIR=${OUTPUT_DIR:?OUTPUT_DIR is required}
EXPECTED_TARGETS=${EXPECTED_TARGETS:?EXPECTED_TARGETS is required}

source ~/miniconda3/etc/profile.d/conda.sh
conda activate BINDRAE-MD
cd "${ROOT}"
mkdir -p logs/slurm

mapfile -t TARGET_DIRS < <(
  find "${INPUT_ROOT}" -type f -name target_audit.json -printf '%h\n' | sort
)
if [[ "${#TARGET_DIRS[@]}" -ne "${EXPECTED_TARGETS}" ]]; then
  echo "ERROR: expected ${EXPECTED_TARGETS} target directories, found ${#TARGET_DIRS[@]}"
  exit 1
fi

ARGS=()
for directory in "${TARGET_DIRS[@]}"; do
  ARGS+=(--input-dir "${directory}")
done
python scripts/assemble_md_phase_normal_cache.py \
  "${ARGS[@]}" \
  --output-dir "${OUTPUT_DIR}"
