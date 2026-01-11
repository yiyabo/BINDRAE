#!/usr/bin/env python3
"""
Create a small training subset file from a split JSON.

Usage:
  python scripts/create_train_subset.py \
    --data_dir data/apo_holo_triplets \
    --n_samples 500 \
    --output train_samples_500.txt \
    --valid_samples_file valid_samples_5k.txt
"""

import argparse
import json
import random
from pathlib import Path


def _extract_ids(payload, split_name: str):
    if isinstance(payload, dict) and split_name in payload:
        payload = payload[split_name]
    if isinstance(payload, list):
        if not payload:
            return []
        if isinstance(payload[0], str):
            return payload
        if isinstance(payload[0], dict):
            return [x["id"] for x in payload if "id" in x]
    raise ValueError("Unsupported split format")


def main():
    parser = argparse.ArgumentParser(description="Create training subset file")
    parser.add_argument("--data_dir", type=str, required=True,
                        help="Data directory (e.g. data/apo_holo_triplets)")
    parser.add_argument("--n_samples", type=int, default=500,
                        help="Number of samples (default: 500)")
    parser.add_argument("--output", type=str, default=None,
                        help="Output file name (default: train_samples_{n}.txt)")
    parser.add_argument("--seed", type=int, default=2025,
                        help="Random seed")
    parser.add_argument("--valid_samples_file", type=str, default=None,
                        help="Optional valid samples file for filtering")
    parser.add_argument("--split", type=str, default="train",
                        help="Split name (default: train)")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    split_file = data_dir / "splits" / f"{args.split}.json"
    if not split_file.exists():
        raise FileNotFoundError(f"Split file not found: {split_file}")

    with open(split_file, "r") as f:
        split_data = json.load(f)

    split_ids = _extract_ids(split_data, args.split)
    print(f"Split '{args.split}' samples: {len(split_ids)}")

    if args.valid_samples_file:
        valid_path = Path(args.valid_samples_file)
        if not valid_path.is_absolute():
            valid_path = data_dir / valid_path
        if valid_path.exists():
            with open(valid_path, "r") as f:
                valid_ids = {line.strip() for line in f if line.strip()}
            before = len(split_ids)
            split_ids = [x for x in split_ids if x in valid_ids]
            print(f"After valid filter: {len(split_ids)} (removed {before - len(split_ids)})")
        else:
            print(f"[WARN] valid_samples_file not found: {valid_path}")

    random.seed(args.seed)
    n_samples = min(args.n_samples, len(split_ids))
    subset = random.sample(split_ids, n_samples)

    if args.output:
        out_path = Path(args.output)
        if not out_path.is_absolute():
            out_path = data_dir / out_path
    else:
        out_path = data_dir / f"train_samples_{n_samples}.txt"

    out_path.write_text("\n".join(subset) + "\n")
    print(f"Created subset: {out_path}")
    print(f"  - samples: {n_samples}")
    print(f"  - seed: {args.seed}")


if __name__ == "__main__":
    main()
