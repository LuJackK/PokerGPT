from __future__ import annotations

import pickle
import struct
import tempfile
import unittest
from pathlib import Path

try:
    import torch
except ImportError:  # Let data-pipeline-only environments keep running their tests.
    torch = None


@unittest.skipIf(torch is None, "PyTorch is not installed")
class TrajectoryDatasetTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.output = Path(self.temporary.name)
        trajectories = ([10, 11, 12, 13], [20, 21, 22])
        masks = ([0, 1, 0, 1], [0, 0, 1])
        (self.output / "train.bin").write_bytes(
            b"".join(struct.pack(f"<{len(row)}H", *row) for row in trajectories)
        )
        (self.output / "train_loss_mask.bin").write_bytes(
            b"".join(bytes(row) for row in masks)
        )
        (self.output / "train.idx").write_bytes(struct.pack("<2Q", 0, 4))
        with (self.output / "meta.pkl").open("wb") as handle:
            pickle.dump({"pad_token_id": 99, "block_size": 8, "vocab_size": 128}, handle)

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def test_items_shift_within_trajectory_boundaries(self) -> None:
        from poker_model.data import PokerTrajectoryDataset

        dataset = PokerTrajectoryDataset(self.output, "train")
        first = dataset[0]
        second = dataset[1]
        self.assertEqual(first[0].tolist(), [10, 11, 12])
        self.assertEqual(first[1].tolist(), [11, 12, 13])
        self.assertEqual(first[2].tolist(), [1, 0, 1])
        self.assertEqual(second[0].tolist(), [20, 21])
        self.assertEqual(second[1].tolist(), [21, 22])
        self.assertEqual(second[2].tolist(), [0, 1])
        self.assertNotIn(20, first[1].tolist())

    def test_collate_right_pads_each_trajectory(self) -> None:
        from poker_model.data import PokerTrajectoryDataset

        dataset = PokerTrajectoryDataset(self.output, "train")
        inputs, targets, masks = dataset.collate_fn([dataset[0], dataset[1]])
        self.assertEqual(inputs.tolist(), [[10, 11, 12], [20, 21, 99]])
        self.assertEqual(targets.tolist(), [[11, 12, 13], [21, 22, -1]])
        self.assertEqual(masks.tolist(), [[1, 0, 1], [0, 1, 0]])

    def test_oversized_trajectory_is_rejected_without_truncation(self) -> None:
        with (self.output / "meta.pkl").open("wb") as handle:
            pickle.dump({"pad_token_id": 99, "block_size": 2, "vocab_size": 128}, handle)
        from poker_model.data import PokerTrajectoryDataset

        with self.assertRaisesRegex(ValueError, "exceeding block_size"):
            PokerTrajectoryDataset(self.output, "train")

    def test_test_split_is_sealed_without_final_evaluation_permit(self) -> None:
        from poker_model.data import PokerTrajectoryDataset

        for suffix in (".bin", "_loss_mask.bin", ".idx"):
            source = self.output / f"train{suffix}"
            target = self.output / f"test{suffix}"
            target.write_bytes(source.read_bytes())
        with self.assertRaisesRegex(PermissionError, "held-out test split is sealed"):
            PokerTrajectoryDataset(self.output, "test")


if __name__ == "__main__":
    unittest.main()
