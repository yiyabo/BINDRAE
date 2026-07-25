#!/usr/bin/env python3
"""Validate an exact Stage-2 subset through the production dataset loader."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.stage2.datasets.dataset_stage2 import ApoHoloBridgeDataset  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default="processed_data/triplets")
    parser.add_argument("--split", choices=("train", "val", "test"), required=True)
    parser.add_argument("--sample-list", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--feature-mode", required=True)
    parser.add_argument("--feature-names", required=True)
    parser.add_argument("--esm-num-layers", type=int, default=1)
    parser.add_argument("--log-every", type=int, default=100)
    return parser.parse_args()


def read_ids(path: Path) -> list[str]:
    sample_ids = [line.strip() for line in path.read_text().splitlines() if line.strip()]
    if not sample_ids:
        raise ValueError(f"Sample list is empty: {path}")
    if len(sample_ids) != len(set(sample_ids)):
        raise ValueError(f"Sample list contains duplicate IDs: {path}")
    return sample_ids


def main() -> None:
    args = parse_args()
    expected_ids = read_ids(args.sample_list)
    dataset = ApoHoloBridgeDataset(
        str(args.data_dir),
        split=args.split,
        valid_samples_file=str(args.sample_list),
        stage1v2_posterior_cache_dir=str(args.cache_dir),
        stage1v2_posterior_feature_mode=args.feature_mode,
        stage1v2_posterior_feature_names=args.feature_names,
        esm_num_layers=args.esm_num_layers,
        trust_prechecked_samples=False,
    )
    actual_ids = [str(record.get("id", "")) for record in dataset.samples]
    missing = sorted(set(expected_ids) - set(actual_ids))
    unexpected = sorted(set(actual_ids) - set(expected_ids))
    if len(actual_ids) != len(expected_ids) or missing or unexpected:
        raise ValueError(
            "Production dataset filtering changed the exact subset: "
            f"expected={len(expected_ids)} actual={len(actual_ids)} "
            f"missing={missing[:8]} unexpected={unexpected[:8]}"
        )

    validated_ids = []
    for index in range(len(dataset)):
        sample = dataset[index]
        sample_id = str(dataset.samples[index]["id"])
        feature_matrix = sample.get("stage1v2_posterior_features")
        if feature_matrix is None:
            raise ValueError(f"{sample_id} did not load Stage-1-v2 features")
        validated_ids.append(sample_id)
        if args.log_every > 0 and len(validated_ids) % args.log_every == 0:
            print(f"Validated {len(validated_ids)}/{len(dataset)} samples", flush=True)

    print(
        json.dumps(
            {
                "cache_dir": str(args.cache_dir),
                "feature_mode": args.feature_mode,
                "sample_list": str(args.sample_list),
                "split": args.split,
                "validated_samples": len(validated_ids),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
