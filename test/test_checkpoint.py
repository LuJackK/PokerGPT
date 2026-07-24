from __future__ import annotations

import pickle
import random
import struct
import tempfile
import unittest
from pathlib import Path

try:
    import torch
except ImportError:
    torch = None


@unittest.skipIf(torch is None, "PyTorch is not installed")
class CheckpointTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def _write_dataset(self) -> None:
        with (self.root / "meta.pkl").open("wb") as handle:
            pickle.dump({"version": "0.8.1"}, handle)
        for number, split in enumerate(("train", "val", "test"), start=1):
            (self.root / f"{split}.bin").write_bytes(struct.pack("<2H", number, number + 1))
            (self.root / f"{split}_loss_mask.bin").write_bytes(bytes((0, 1)))
            (self.root / f"{split}.idx").write_bytes(struct.pack("<Q", 0))

    @staticmethod
    def _payload(fingerprint):
        return {
            "model": {"weight": torch.tensor([1.0])},
            "model_config": {"n_layer": 1},
            "optimizer": {"state": {}},
            "scheduler": {"last_epoch": 2},
            "scaler": None,
            "optimizer_step": 2,
            "epoch": 1,
            "next_batch_cursor": 3,
            "best_validation_loss": 0.5,
            "elapsed_seconds": 1.25,
            "counters": {"trajectories": 4, "tokens": 20, "decisions": 6},
            "rng_state": {"python": None, "torch_cpu": None, "torch_cuda": []},
            "sampler_state": {"epoch": 1, "next_batch_cursor": 3},
            "training_config": {"seed": 1337},
            "dataset_fingerprint": fingerprint,
            "preprocessing_identity": {"version": "0.8.1"},
            "environment": {"python": "test"},
        }

    def test_fingerprint_changes_with_any_artifact(self) -> None:
        from poker_model.checkpoint import dataset_fingerprint

        self._write_dataset()
        before = dataset_fingerprint(self.root)
        (self.root / "test_loss_mask.bin").write_bytes(bytes((1, 1)))
        after = dataset_fingerprint(self.root)
        self.assertNotEqual(before["digest"], after["digest"])
        self.assertEqual(len(before["files"]), 10)

    def test_atomic_round_trip_and_compatibility_checks(self) -> None:
        from poker_model.checkpoint import dataset_fingerprint, load_checkpoint, save_checkpoint

        self._write_dataset()
        fingerprint = dataset_fingerprint(self.root)
        path = self.root / "latest.pt"
        save_checkpoint(path, self._payload(fingerprint))
        loaded = load_checkpoint(
            path,
            expected_dataset_fingerprint=fingerprint,
            expected_training_config={"seed": 1337},
            expected_model_config={"n_layer": 1},
        )
        self.assertEqual(loaded["optimizer_step"], 2)
        self.assertTrue(torch.equal(loaded["model"]["weight"], torch.tensor([1.0])))
        self.assertEqual(list(self.root.glob("*.tmp")), [])

        incompatible = dict(fingerprint)
        incompatible["digest"] = "different"
        with self.assertRaisesRegex(ValueError, "dataset fingerprint"):
            load_checkpoint(path, expected_dataset_fingerprint=incompatible)

    def test_rng_state_round_trip(self) -> None:
        from poker_model.checkpoint import capture_rng_state, restore_rng_state

        random.seed(7)
        torch.manual_seed(7)
        state = capture_rng_state()
        expected = (random.random(), torch.rand(3))
        random.random()
        torch.rand(9)
        restore_rng_state(state)
        actual = (random.random(), torch.rand(3))
        self.assertEqual(expected[0], actual[0])
        self.assertTrue(torch.equal(expected[1], actual[1]))

    @unittest.skipUnless(torch is not None and torch.cuda.is_available(), "CUDA is unavailable")
    def test_rng_restore_accepts_cuda_mapped_checkpoint_tensors(self) -> None:
        from poker_model.checkpoint import capture_rng_state, restore_rng_state

        state = capture_rng_state()
        mapped = {
            "python": state["python"],
            "torch_cpu": state["torch_cpu"].cuda(),
            "torch_cuda": [cuda_state.cuda() for cuda_state in state["torch_cuda"]],
        }
        restore_rng_state(mapped)


if __name__ == "__main__":
    unittest.main()
