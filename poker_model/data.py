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
from typing import Callable, Sequence

import torch
from torch.utils.data import Dataset


BatchItem = tuple[torch.Tensor, torch.Tensor, torch.Tensor]


class PokerTrajectoryDataset(Dataset[BatchItem]):
    """Read complete trajectories from one preprocessing split.

    Expected files are ``<split>.bin`` (little-endian uint16 tokens),
    ``<split>_loss_mask.bin`` (uint8), ``<split>.idx`` (little-endian uint64
    starting offsets), and ``meta.pkl``.  A trajectory of ``n`` stored tokens
    yields ``n - 1`` next-token training positions.
    """

    def __init__(self, output_dir: str | Path, split: str) -> None:
        self.output_dir = Path(output_dir)
        self.split = split

        with (self.output_dir / "meta.pkl").open("rb") as handle:
            meta = pickle.load(handle)
        try:
            self.pad_token_id = int(meta["pad_token_id"])
            self.block_size = int(meta["block_size"])
        except KeyError as exc:
            raise ValueError(f"meta.pkl is missing required key {exc.args[0]!r}") from exc
        if self.pad_token_id < 0:
            raise ValueError("pad_token_id must be non-negative")
        if self.block_size <= 0:
            raise ValueError("block_size must be positive")

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
        for number, (start, end) in enumerate(zip(self.offsets, ends)):
            length = end - start
            if length < 2:
                raise ValueError(f"trajectory {number} has fewer than two tokens")
            if length - 1 > self.block_size:
                raise ValueError(
                    f"trajectory {number} has {length - 1} training positions, "
                    f"exceeding block_size={self.block_size}"
                )

        # torch.from_file memory-maps the artifacts instead of copying the full
        # dataset into process memory.  Slices are converted to model dtypes in
        # __getitem__, so returned tensors do not alias the maps.
        self._tokens = (
            torch.from_file(str(token_path), shared=False, size=self.token_count, dtype=torch.uint16)
            if self.token_count
            else torch.empty(0, dtype=torch.uint16)
        )
        self._loss_mask = (
            torch.from_file(str(mask_path), shared=False, size=self.token_count, dtype=torch.uint8)
            if self.token_count
            else torch.empty(0, dtype=torch.uint8)
        )

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

