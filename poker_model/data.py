"""Trajectory-aware loading for PokerGPT binary training artifacts.

Each item is one complete player-perspective hand trajectory.  The index file
contains the starting token offset of each trajectory; no sample is assembled
from arbitrary regions of the flat token file.
"""

from __future__ import annotations

import pickle
import struct
from functools import partial
from pathlib import Path
from typing import Callable, Iterator, Sequence

import torch
from torch.utils.data import Dataset, Sampler

from .evaluation_access import TestSplitPermit, require_split_access


BatchItem = tuple[torch.Tensor, torch.Tensor, torch.Tensor]


class PokerTrajectoryDataset(Dataset[BatchItem]):
    """Read complete trajectories from one preprocessing split.

    Expected files are ``<split>.bin`` (little-endian uint16 tokens),
    ``<split>_loss_mask.bin`` (uint8), ``<split>.idx`` (little-endian uint64
    starting offsets), and ``meta.pkl``.  A trajectory of ``n`` stored tokens
    yields ``n - 1`` next-token training positions.
    """

    def __init__(
        self,
        output_dir: str | Path,
        split: str,
        *,
        test_permit: TestSplitPermit | None = None,
    ) -> None:
        if split not in {"train", "val", "test"}:
            raise ValueError(f"unsupported dataset split: {split!r}")
        require_split_access(split, test_permit)
        self.output_dir = Path(output_dir)
        self.split = split

        with (self.output_dir / "meta.pkl").open("rb") as handle:
            meta = pickle.load(handle)
        try:
            self.pad_token_id = int(meta["pad_token_id"])
            self.block_size = int(meta["block_size"])
            self.vocab_size = int(meta["vocab_size"])
        except KeyError as exc:
            raise ValueError(f"meta.pkl is missing required key {exc.args[0]!r}") from exc
        if self.pad_token_id < 0:
            raise ValueError("pad_token_id must be non-negative")
        if self.block_size <= 0:
            raise ValueError("block_size must be positive")
        if not 0 < self.vocab_size <= 32768:
            raise ValueError("vocab_size must fit in the signed int16 artifact reader")

        token_path = self.output_dir / f"{split}.bin"
        mask_path = self.output_dir / f"{split}_loss_mask.bin"
        index_path = self.output_dir / f"{split}.idx"
        token_bytes = token_path.stat().st_size
        mask_bytes = mask_path.stat().st_size
        index_bytes = index_path.read_bytes()
        if token_bytes % 2:
            raise ValueError(f"{token_path.name} has an odd byte length")
        if len(index_bytes) % 8:
            raise ValueError(f"{index_path.name} is not aligned to uint64 entries")

        self.token_count = token_bytes // 2
        if mask_bytes != self.token_count:
            raise ValueError("token and loss-mask files contain different numbers of entries")
        self.offsets = tuple(value[0] for value in struct.iter_unpack("<Q", index_bytes))
        if self.offsets:
            if self.offsets[0] != 0:
                raise ValueError("the first trajectory offset must be zero")
            if any(left >= right for left, right in zip(self.offsets, self.offsets[1:])):
                raise ValueError("trajectory offsets must be strictly increasing")
            if self.offsets[-1] >= self.token_count:
                raise ValueError("trajectory offset is outside the token file")
        elif self.token_count:
            raise ValueError("token data exists but the trajectory index is empty")

        ends = self.offsets[1:] + (self.token_count,) if self.offsets else ()
        trajectory_lengths = []
        for number, (start, end) in enumerate(zip(self.offsets, ends)):
            length = end - start
            if length < 2:
                raise ValueError(f"trajectory {number} has fewer than two tokens")
            if length - 1 > self.block_size:
                raise ValueError(
                    f"trajectory {number} has {length - 1} training positions, "
                    f"exceeding block_size={self.block_size}"
                )
            trajectory_lengths.append(length - 1)
        self.trajectory_lengths = tuple(trajectory_lengths)

        # torch.from_file memory-maps the artifacts instead of copying the full
        # dataset into process memory. The uint16 artifact bytes are read as
        # int16 because this vocabulary is below 32768; valid token bit patterns
        # therefore have identical signed and unsigned values, while int16 has
        # broader eager-mode support across PyTorch 2.x releases. Slices are
        # converted to model dtypes in __getitem__.
        self._tokens = (
            torch.from_file(str(token_path), shared=False, size=self.token_count, dtype=torch.int16)
            if self.token_count
            else torch.empty(0, dtype=torch.int16)
        )
        self._loss_mask = (
            torch.from_file(str(mask_path), shared=False, size=self.token_count, dtype=torch.uint8)
            if self.token_count
            else torch.empty(0, dtype=torch.uint8)
        )
        self.decision_count = int(self._loss_mask.sum().item())

    def __len__(self) -> int:
        return len(self.offsets)

    def __getitem__(self, index: int) -> BatchItem:
        if index < 0:
            index += len(self)
        if index < 0 or index >= len(self):
            raise IndexError("trajectory index out of range")
        start = self.offsets[index]
        end = self.offsets[index + 1] if index + 1 < len(self) else self.token_count
        tokens = self._tokens[start:end].to(dtype=torch.long)
        loss_mask = self._loss_mask[start:end].clone()
        return tokens[:-1], tokens[1:], loss_mask[1:]

    @property
    def collate_fn(self) -> Callable[[Sequence[BatchItem]], BatchItem]:
        """A DataLoader-compatible collator bound to this dataset's pad ID."""

        return partial(collate_trajectories, pad_token_id=self.pad_token_id)


class LengthAwareBatchSampler(Sampler[list[int]]):
    """Deterministically batch trajectories after local length sorting.

    Each epoch starts from a permutation seeded by ``seed + epoch``.  Indexes
    are sorted by length only inside bounded pools, batched, and then the
    batches are shuffled.  This reduces padding without fixing a global
    shortest-to-longest curriculum.

    ``next_batch_cursor`` is advanced before a batch is yielded.  With the
    baseline's zero-worker loader, a checkpoint taken after an optimizer-step
    boundary can therefore resume at the next unconsumed batch.
    """

    def __init__(
        self,
        lengths: Sequence[int],
        batch_size: int,
        *,
        seed: int,
        pool_size: int | None = None,
        drop_last: bool = False,
    ) -> None:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if any(int(length) <= 0 for length in lengths):
            raise ValueError("all trajectory lengths must be positive")
        if pool_size is None:
            pool_size = batch_size * 100
        if pool_size < batch_size:
            raise ValueError("pool_size must be at least batch_size")
        if pool_size % batch_size:
            raise ValueError("pool_size must be a multiple of batch_size")

        self.lengths = tuple(int(length) for length in lengths)
        self.batch_size = int(batch_size)
        self.seed = int(seed)
        self.pool_size = int(pool_size)
        self.drop_last = bool(drop_last)
        self.epoch = 0
        self.next_batch_cursor = 0

    def _epoch_batches(self) -> list[list[int]]:
        generator = torch.Generator()
        generator.manual_seed(self.seed + self.epoch)
        permutation = torch.randperm(len(self.lengths), generator=generator).tolist()
        batches: list[list[int]] = []
        for pool_start in range(0, len(permutation), self.pool_size):
            pool = permutation[pool_start : pool_start + self.pool_size]
            pool.sort(key=lambda index: self.lengths[index])
            for batch_start in range(0, len(pool), self.batch_size):
                batch = pool[batch_start : batch_start + self.batch_size]
                if len(batch) == self.batch_size or not self.drop_last:
                    batches.append(batch)
        if batches:
            order = torch.randperm(len(batches), generator=generator).tolist()
            batches = [batches[index] for index in order]
        return batches

    def __iter__(self) -> Iterator[list[int]]:
        batches = self._epoch_batches()
        if self.next_batch_cursor > len(batches):
            raise ValueError("next_batch_cursor is outside the epoch")
        for batch_index in range(self.next_batch_cursor, len(batches)):
            self.next_batch_cursor = batch_index + 1
            yield batches[batch_index]

    def __len__(self) -> int:
        full, remainder = divmod(len(self.lengths), self.batch_size)
        total = full + (bool(remainder) and not self.drop_last)
        return max(0, int(total) - self.next_batch_cursor)

    def set_epoch(self, epoch: int, next_batch_cursor: int = 0) -> None:
        if epoch < 0:
            raise ValueError("epoch must be non-negative")
        if next_batch_cursor < 0:
            raise ValueError("next_batch_cursor must be non-negative")
        self.epoch = int(epoch)
        self.next_batch_cursor = int(next_batch_cursor)
        if self.next_batch_cursor > self.total_batches:
            raise ValueError("next_batch_cursor is outside the epoch")

    @property
    def total_batches(self) -> int:
        full, remainder = divmod(len(self.lengths), self.batch_size)
        return full + int(bool(remainder) and not self.drop_last)

    def state_dict(self) -> dict[str, int | bool]:
        return {
            "seed": self.seed,
            "epoch": self.epoch,
            "next_batch_cursor": self.next_batch_cursor,
            "batch_size": self.batch_size,
            "pool_size": self.pool_size,
            "drop_last": self.drop_last,
            "trajectory_count": len(self.lengths),
        }

    def load_state_dict(self, state: dict[str, int | bool]) -> None:
        expected = {
            "seed": self.seed,
            "batch_size": self.batch_size,
            "pool_size": self.pool_size,
            "drop_last": self.drop_last,
            "trajectory_count": len(self.lengths),
        }
        mismatches = {
            key: (expected_value, state.get(key))
            for key, expected_value in expected.items()
            if state.get(key) != expected_value
        }
        if mismatches:
            details = ", ".join(
                f"{key}: expected {expected_value!r}, got {actual!r}"
                for key, (expected_value, actual) in mismatches.items()
            )
            raise ValueError(f"incompatible sampler state ({details})")
        self.set_epoch(int(state["epoch"]), int(state["next_batch_cursor"]))


def collate_trajectories(
    batch: Sequence[BatchItem], *, pad_token_id: int
) -> BatchItem:
    """Right-pad complete trajectories without combining their token streams."""

    if not batch:
        raise ValueError("cannot collate an empty batch")
    max_length = max(inputs.numel() for inputs, _, _ in batch)
    batch_size = len(batch)
    inputs_out = torch.full((batch_size, max_length), pad_token_id, dtype=torch.long)
    targets_out = torch.full((batch_size, max_length), -1, dtype=torch.long)
    masks_out = torch.zeros((batch_size, max_length), dtype=torch.uint8)

    for row, (inputs, targets, loss_mask) in enumerate(batch):
        if inputs.ndim != 1 or targets.ndim != 1 or loss_mask.ndim != 1:
            raise ValueError("each trajectory tensor must be one-dimensional")
        if not (inputs.numel() == targets.numel() == loss_mask.numel()):
            raise ValueError("input, target, and loss-mask lengths must match")
        length = inputs.numel()
        inputs_out[row, :length] = inputs
        targets_out[row, :length] = targets
        masks_out[row, :length] = loss_mask
    return inputs_out, targets_out, masks_out

