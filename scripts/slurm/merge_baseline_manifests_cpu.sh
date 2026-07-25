#!/usr/bin/env bash

#SBATCH --job-name=merge_baseline
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:30:00
#SBATCH --output=logs/slurm/merge_baseline_%j.out
#SBATCH --error=logs/slurm/merge_baseline_%j.err

set -euo pipefail

ROOT="${ROOT:-/mnt/inaisfs/data/home/zhaozc_criait/XinxiangWang/BINDRAE}"
cd "${ROOT}"
source /mnt/inaisfs/data/home/zhaozc_criait/miniconda3/etc/profile.d/conda.sh
conda activate BINDRAE

INPUT_MANIFESTS="${INPUT_MANIFESTS:?Set colon-separated INPUT_MANIFESTS}"
SAMPLE_FILE="${SAMPLE_FILE:?Set SAMPLE_FILE}"
OUTPUT="${OUTPUT:?Set OUTPUT}"
SUMMARY="${SUMMARY:?Set SUMMARY}"

IFS=':' read -r -a MANIFESTS <<< "${INPUT_MANIFESTS}"
ARGS=()
for manifest in "${MANIFESTS[@]}"; do
  ARGS+=(--input "${manifest}")
done
python scripts/merge_baseline_manifests.py \
  "${ARGS[@]}" \
  --sample_file "${SAMPLE_FILE}" \
  --output "${OUTPUT}" \
  --summary "${SUMMARY}"
