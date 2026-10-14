"""Source-qualified datasets and exact, distributed mixture batches."""

from __future__ import annotations

import math
from bisect import bisect_right

import torch
from torch.utils.data import Dataset, Sampler


class SourceMixtureDataset(Dataset):
    def __init__(self, datasets):
        self.names = list(datasets)
        self.datasets = list(datasets.values())
        self.sizes = [len(d) for d in self.datasets]
        if not self.sizes or min(self.sizes) <= 0:
            raise ValueError("Every mixture source must be nonempty")
        self.ends = []
        for size in self.sizes:
            self.ends.append(size + (self.ends[-1] if self.ends else 0))

    def __len__(self):
        return self.ends[-1]

    def __getitem__(self, index):
        source = bisect_right(self.ends, index)
        offset = self.ends[source - 1] if source else 0
        sample = dict(self.datasets[source][index - offset])
        sample["source"] = self.names[source]
        sample["sample_id"] = f"{self.names[source]}::{sample['sample_id']}"
        return sample


class MixtureBatchSampler(Sampler):
    """Identical global draws on each rank, with disjoint per-rank slices.

    Sources cycle through shuffled permutations. Repeated exposure across cycles
    is intentional; an individual global batch never contains a duplicate.
    """

    def __init__(
        self, sizes, proportions, batch_size, *, seed=42, rank=0, world_size=1
    ):
        if len(sizes) != len(proportions) or not sizes:
            raise ValueError("Mixture sizes and proportions must match")
        if not 0 <= rank < world_size or batch_size <= 0:
            raise ValueError("Invalid mixture rank/world size/batch size")
        if any(not math.isfinite(p) or p <= 0 for p in proportions):
            raise ValueError("Source proportions must be positive and finite")
        if not math.isclose(sum(proportions), 1.0, abs_tol=1e-8):
            raise ValueError("Source proportions must sum to one")
        self.quotas = [round(batch_size * p) for p in proportions]
        if any(
            q < 1 or not math.isclose(q, batch_size * p, abs_tol=1e-8)
            for q, p in zip(self.quotas, proportions)
        ):
            raise ValueError(
                "Source proportions must produce integer per-rank batch quotas"
            )
        if any(n < q * world_size for n, q in zip(sizes, self.quotas)):
            raise ValueError("Source too small for a disjoint global batch")
        self.sizes = list(sizes)
        self.seed, self.rank, self.world_size = seed, rank, world_size
        self.epoch = 0
        self.steps = math.ceil(sizes[0] / (self.quotas[0] * world_size))
        # Lightning forwards set_epoch through batch_sampler.sampler.
        self.sampler = self

    def set_epoch(self, epoch):
        self.epoch = int(epoch)

    def __len__(self):
        return self.steps

    def __iter__(self):
        generator = torch.Generator().manual_seed(self.seed + 1_000_003 * self.epoch)
        pools = [torch.randperm(n, generator=generator).tolist() for n in self.sizes]
        positions = [0] * len(pools)
        for _ in range(self.steps):
            batch, offset = [], 0
            for i, (size, quota) in enumerate(zip(self.sizes, self.quotas)):
                count = quota * self.world_size
                # Start a new cycle rather than duplicate a sample within a batch.
                if positions[i] + count > size:
                    pools[i] = torch.randperm(size, generator=generator).tolist()
                    positions[i] = 0
                start = positions[i] + self.rank * quota
                batch.extend(offset + x for x in pools[i][start : start + quota])
                positions[i] += count
                offset += size
            yield batch
