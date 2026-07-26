#!/usr/bin/env bash
# Rebuild ligand bond orders and formal charges from the PDB Chemical Component
# Dictionary across the AHoJ triplet corpus.
#
# prepare_ahojdb_triplets.py wrote ligand.sdf with Chem.MolFromPDBBlock, and the
# PDB format stores no bond orders, so every bond was emitted as order 1 with no
# M  CHG record. Polyphosphate ligands therefore carry a phosphorus with illegal
# valence 4; RDKit completes it to 5 with a hydride and OpenFF parameterises a
# neutral P-H species whose solvated initial energy is non-finite. That is the
# abort at prepare_md_pilot_system.py:328 behind the AHoJ smoke setup failures.
# Non-phosphate ligands do not abort but are simulated fully saturated.
#
# The repair preserves observed atom order and coordinates, and verifies each
# rewritten SDF against the frozen ligand_coords.npy. It changes no scientific
# threshold, split, or cache.
#
# TWO STEPS -- the prefetch must run where there is outbound network, which
# compute nodes do not have:
#
#   1) On the login node (network, cheap, no Python training):
#        python scripts/repair_triplet_ligand_bond_orders.py prefetch \
#          --triplet-root processed_data/triplets \
#          --ccd-dir processed_data/ccd_cache
#
#   2) Here, offline, dry run first:
#        MODE=dryrun sbatch scripts/slurm/repair_triplet_ligand_bond_orders_cpu.sh
#      then, after reading the report:
#        MODE=apply  sbatch scripts/slurm/repair_triplet_ligand_bond_orders_cpu.sh

#SBATCH --job-name=lig_bondfix
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=logs/slurm/ligand_bond_order_repair_%j.out
#SBATCH --error=logs/slurm/ligand_bond_order_repair_%j.err

set -euo pipefail

ROOT="${ROOT:-/mnt/inaisfs/data/home/zhaozc_criait/XinxiangWang/BINDRAE}"
TRIPLET_ROOT="${TRIPLET_ROOT:-processed_data/triplets}"
CCD_DIR="${CCD_DIR:-processed_data/ccd_cache}"
MODE="${MODE:-dryrun}"
TAG="${TAG:-ligand_bond_order_repair_20260726_v1}"
REPORT_DIR="${REPORT_DIR:-logs/ligand_bond_order_repair/${TAG}_${MODE}}"
SAMPLE_LIST="${SAMPLE_LIST:-}"
MISSING_BOND_POLICY="${MISSING_BOND_POLICY:-restore}"

cd "$ROOT"
mkdir -p logs/slurm "$REPORT_DIR"

source /mnt/inaisfs/data/home/zhaozc_criait/miniconda3/etc/profile.d/conda.sh
conda activate BINDRAE

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

if [[ ! -d "$TRIPLET_ROOT" ]]; then
  echo "ERROR: missing triplet root: $TRIPLET_ROOT" >&2
  exit 2
fi

# Compute nodes are offline, so a cold cache would reject every sample with
# ccd_template_missing and look like a catastrophic yield collapse. Fail loudly
# instead, and say which step was skipped.
if [[ ! -d "$CCD_DIR" ]] || [[ -z "$(find "$CCD_DIR" -maxdepth 1 -name '*.sdf' -print -quit)" ]]; then
  echo "ERROR: CCD cache $CCD_DIR is empty." >&2
  echo "       Run the prefetch step on the login node first (see header)." >&2
  exit 2
fi
echo "CCD cache $CCD_DIR: $(find "$CCD_DIR" -maxdepth 1 -name '*.sdf' | wc -l) components"
echo "Triplet root $TRIPLET_ROOT: $(find "$TRIPLET_ROOT" -maxdepth 1 -mindepth 1 -type d | wc -l) sample directories"

ARGS=(
  repair
  --triplet-root "$TRIPLET_ROOT"
  --ccd-dir "$CCD_DIR"
  --report-dir "$REPORT_DIR"
  --missing-bond-policy "$MISSING_BOND_POLICY"
)
if [[ -n "$SAMPLE_LIST" ]]; then
  if [[ ! -s "$SAMPLE_LIST" ]]; then
    echo "ERROR: SAMPLE_LIST is set but empty or missing: $SAMPLE_LIST" >&2
    exit 2
  fi
  ARGS+=(--sample-list "$SAMPLE_LIST")
fi

case "$MODE" in
  dryrun)
    echo "=== DRY RUN: no file will be modified ==="
    ;;
  apply)
    echo "=== APPLY: ligand.sdf will be rewritten, originals kept as ligand.legacy_connectivity_only.sdf ==="
    ARGS+=(--apply)
    ;;
  *)
    echo "ERROR: MODE must be 'dryrun' or 'apply', got '$MODE'" >&2
    exit 2
    ;;
esac

python scripts/repair_triplet_ligand_bond_orders.py "${ARGS[@]}"

echo
echo "Report: $REPORT_DIR/summary.json"
echo "Per-sample ledger: $REPORT_DIR/records.jsonl"
