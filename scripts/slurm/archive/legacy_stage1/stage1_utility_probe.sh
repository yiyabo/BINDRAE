#!/bin/bash
# Fast Stage-1 utility-aligned probe on a reduced validation subset.

#SBATCH --job-name=s1_ut_probe
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:A100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --output=logs/slurm/stage1_utility_probe_%j.out
#SBATCH --error=logs/slurm/stage1_utility_probe_%j.err

set -euo pipefail
unset LD_PRELOAD
unset PROXYCHAINS_CONF_FILE
unset PROXYCHAINS_QUIET_MODE

export PATH=/mnt/inaisfs/data/home/zhaozc_criait/miniconda3/bin:$PATH
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=4

source /mnt/inaisfs/data/home/zhaozc_criait/miniconda3/etc/profile.d/conda.sh
conda activate BINDRAE

ROOT=/mnt/inaisfs/data/home/zhaozc_criait/XinxiangWang/BINDRAE
cd "$ROOT"
mkdir -p logs/slurm logs/stage1

VAL_SOURCE="$ROOT/processed_data/triplets/val_valid.txt"
VAL_SUBSET="$ROOT/processed_data/triplets/val_stage1_utility_probe_256.txt"

python - <<'PY'
from pathlib import Path
src = Path('/mnt/inaisfs/data/home/zhaozc_criait/XinxiangWang/BINDRAE/processed_data/triplets/val_valid.txt')
dst = Path('/mnt/inaisfs/data/home/zhaozc_criait/XinxiangWang/BINDRAE/processed_data/triplets/val_stage1_utility_probe_256.txt')
ids = [line.strip() for line in src.read_text().splitlines() if line.strip()]
subset = ids[:256]
dst.write_text('\n'.join(subset) + '\n')
print(f'Wrote {len(subset)} IDs to {dst}')
PY

echo "=============================================="
echo "Stage-1 utility probe"
echo "=============================================="
echo "Job ID:      $SLURM_JOB_ID"
echo "Node:        $SLURM_NODELIST"
echo "Val subset:  $VAL_SUBSET"
echo "Start:       $(date)"
echo "=============================================="

run_probe() {
  local ckpt_path="$1"
  local out_dir="$2"
  echo "\n>>> Probing $(basename "$ckpt_path") -> $out_dir"
  python scripts/audit_stage1_checkpoint.py \
    --checkpoint "$ckpt_path" \
    --data_dir processed_data/triplets \
    --valid_samples_file /mnt/inaisfs/data/home/zhaozc_criait/XinxiangWang/BINDRAE/processed_data/triplets/train_valid.txt \
    --val_samples_file "$VAL_SUBSET" \
    --sample_metadata_file sample_metadata.json \
    --batch_size 4 \
    --num_workers 2 \
    --device cuda \
    --enable_dual_mask_audit \
    --output_dir "$out_dir"
}

run_probe checkpoints/stage1/opt8e2_20260411_122707/best_model.pt \
  logs/stage1/opt8e2_20260411_122707/probe_stage2_utility_best

run_probe checkpoints/stage1/opt8e2_20260411_122707/epoch_018.pt \
  logs/stage1/opt8e2_20260411_122707/probe_stage2_utility_epoch018

run_probe checkpoints/stage1/e6b_slowramp_from_opt8e6a_20260415_024951_20260415_055419/best_model.pt \
  logs/stage1/e6b_slowramp_from_opt8e6a_20260415_024951_20260415_055419/probe_stage2_utility_best

run_probe checkpoints/stage1/e6b_slowramp_from_opt8e6a_20260415_024951_20260415_055419/epoch_010.pt \
  logs/stage1/e6b_slowramp_from_opt8e6a_20260415_024951_20260415_055419/probe_stage2_utility_epoch010

python - <<'PY'
import json
from pathlib import Path
paths = [
    Path('logs/stage1/opt8e2_20260411_122707/probe_stage2_utility_best/audit_metrics.jsonl'),
    Path('logs/stage1/opt8e2_20260411_122707/probe_stage2_utility_epoch018/audit_metrics.jsonl'),
    Path('logs/stage1/e6b_slowramp_from_opt8e6a_20260415_024951_20260415_055419/probe_stage2_utility_best/audit_metrics.jsonl'),
    Path('logs/stage1/e6b_slowramp_from_opt8e6a_20260415_024951_20260415_055419/probe_stage2_utility_epoch010/audit_metrics.jsonl'),
]
rows = []
for p in paths:
    row = json.loads(p.read_text().splitlines()[-1])
    row['source'] = str(p.parent)
    rows.append(row)

def metric(x, name):
    v = x.get(name)
    return float('-inf') if v is None or v != v else float(v)

for key in ['ligand_facing_chi1_acc', 'holo_pocket_chi1_acc', 'pocket_chi1_acc']:
    rows_sorted = sorted(rows, key=lambda r: metric(r, key), reverse=True)
    print(f'\n=== Ranking by {key} ===')
    for row in rows_sorted:
        print({
            'source': row['source'],
            key: row.get(key),
            'ligand_facing_chi12_acc': row.get('ligand_facing_chi12_acc'),
            'pocket_chi1_acc': row.get('pocket_chi1_acc'),
            'holo_pocket_chi1_acc': row.get('holo_pocket_chi1_acc'),
        })
PY

echo "\n=============================================="
echo "Stage-1 utility probe completed: $(date)"
echo "=============================================="
