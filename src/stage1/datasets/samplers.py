"""
Stage-1 data samplers.
"""

from __future__ import annotations

from typing import List, Optional, Sequence

import torch
from torch.utils.data import Sampler


class DistributedLengthBatchSampler(Sampler[List[int]]):
    """
    Build per-rank batches from globally bucketed sample costs.

    The sampler uses the same shuffled order on every rank, packs samples into
    global steps, and yields only the local rank's indices for each step.
    When ``residue_budget`` is provided, local batch sizes may vary; this is
    intended to keep long proteins from forcing every rank to the VRAM limit.
    """

    def __init__(
        self,
        sample_costs: Sequence[int],
        batch_size: int,
        num_replicas: int,
        rank: int,
        *,
        shuffle: bool = True,
        drop_last: bool = True,
        seed: int = 0,
        bucket_size_multiplier: int = 8,
        residue_budget: Optional[int] = None,
    ) -> None:
        if batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {batch_size}")
        if num_replicas <= 0:
            raise ValueError(f"num_replicas must be positive, got {num_replicas}")
        if rank < 0 or rank >= num_replicas:
            raise ValueError(f"rank must be in [0, {num_replicas}), got {rank}")
        if bucket_size_multiplier <= 0:
            raise ValueError(
                f"bucket_size_multiplier must be positive, got {bucket_size_multiplier}"
            )
        if residue_budget is not None and residue_budget <= 0:
            raise ValueError(f"residue_budget must be positive, got {residue_budget}")

        self.sample_costs = [int(x) for x in sample_costs]
        self.batch_size = batch_size
        self.num_replicas = num_replicas
        self.rank = rank
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.seed = seed
        self.bucket_size_multiplier = bucket_size_multiplier
        self.residue_budget = residue_budget

        self.global_batch_size = self.batch_size * self.num_replicas
        self.bucket_size = max(
            self.global_batch_size * self.bucket_size_multiplier,
            self.global_batch_size,
        )
        self.epoch = 0
        self._cached_epoch: Optional[int] = None
        self._cached_batches: Optional[List[List[int]]] = None

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __len__(self) -> int:
        return len(self._get_local_batches())

    def __iter__(self):
        yield from self._get_local_batches()

    def _get_generator(self) -> torch.Generator:
        generator = torch.Generator()
        generator.manual_seed(self.seed + self.epoch)
        return generator

    def _get_local_batches(self) -> List[List[int]]:
        if self._cached_epoch != self.epoch or self._cached_batches is None:
            self._cached_batches = self._build_local_batches()
            self._cached_epoch = self.epoch
        return self._cached_batches

    def _ordered_indices(self, generator: torch.Generator) -> List[int]:
        if not self.shuffle:
            return list(range(len(self.sample_costs)))
        return torch.randperm(len(self.sample_costs), generator=generator).tolist()

    def _build_local_batches(self) -> List[List[int]]:
        generator = self._get_generator()
        indices = self._ordered_indices(generator)
        global_batches: List[List[List[int]]] = []

        for start in range(0, len(indices), self.bucket_size):
            bucket = indices[start:start + self.bucket_size]
            bucket.sort(key=lambda idx: self.sample_costs[idx], reverse=True)
            if self.residue_budget is None:
                global_batches.extend(self._pack_fixed_bucket(bucket))
            else:
                global_batches.extend(self._pack_budgeted_bucket(bucket))

        if self.shuffle and global_batches:
            order = torch.randperm(len(global_batches), generator=generator).tolist()
            global_batches = [global_batches[i] for i in order]

        return [batch[self.rank] for batch in global_batches]

    def _pack_fixed_bucket(self, bucket: List[int]) -> List[List[List[int]]]:
        batches: List[List[List[int]]] = []
        for start in range(0, len(bucket), self.global_batch_size):
            chunk = bucket[start:start + self.global_batch_size]
            if len(chunk) < self.global_batch_size and self.drop_last:
                break

            rank_batches = [[] for _ in range(self.num_replicas)]
            rank_costs = [0 for _ in range(self.num_replicas)]

            for idx in chunk:
                eligible = [
                    replica for replica in range(self.num_replicas)
                    if len(rank_batches[replica]) < self.batch_size
                ]
                if not eligible:
                    break
                replica = min(
                    eligible,
                    key=lambda replica: (rank_costs[replica], len(rank_batches[replica])),
                )
                rank_batches[replica].append(idx)
                rank_costs[replica] += self.sample_costs[idx]

            if any(len(batch) == 0 for batch in rank_batches):
                continue
            if self.drop_last and any(len(batch) != self.batch_size for batch in rank_batches):
                continue
            batches.append(rank_batches)

        return batches

    def _pack_budgeted_bucket(self, bucket: List[int]) -> List[List[List[int]]]:
        assert self.residue_budget is not None

        batches: List[List[List[int]]] = []
        remaining = list(bucket)

        while remaining:
            if self.drop_last and len(remaining) < self.num_replicas:
                break

            rank_batches = [[] for _ in range(self.num_replicas)]
            rank_costs = [0 for _ in range(self.num_replicas)]

            # Seed one sample per rank so every DDP worker gets work.
            for replica in range(self.num_replicas):
                if not remaining:
                    break
                idx = remaining.pop(0)
                rank_batches[replica].append(idx)
                rank_costs[replica] = self.sample_costs[idx]

            if any(len(batch) == 0 for batch in rank_batches):
                break

            made_progress = True
            while made_progress:
                made_progress = False
                if all(len(batch) >= self.batch_size for batch in rank_batches):
                    break

                pos = 0
                while pos < len(remaining):
                    idx = remaining[pos]
                    cost = self.sample_costs[idx]
                    eligible = [
                        replica for replica in range(self.num_replicas)
                        if len(rank_batches[replica]) < self.batch_size
                        and rank_costs[replica] + cost <= self.residue_budget
                    ]
                    if not eligible:
                        pos += 1
                        continue

                    replica = min(
                        eligible,
                        key=lambda replica: (rank_costs[replica], len(rank_batches[replica])),
                    )
                    rank_batches[replica].append(idx)
                    rank_costs[replica] += cost
                    remaining.pop(pos)
                    made_progress = True

                    if all(len(batch) >= self.batch_size for batch in rank_batches):
                        break

            batches.append(rank_batches)

        return batches
