from __future__ import annotations

import copy
import pickle
import struct
import tempfile
import unittest
from pathlib import Path

try:
    import torch
except ImportError:
    torch = None


@unittest.skipIf(torch is None, "PyTorch is not installed")
class TrainerTests(unittest.TestCase):
    def assertNestedEqual(self, left, right, path="root") -> None:
        if torch.is_tensor(left):
            self.assertTrue(torch.equal(left, right), path)
        elif isinstance(left, dict):
            self.assertEqual(set(left), set(right), path)
            for key in left:
                self.assertNestedEqual(left[key], right[key], f"{path}.{key}")
        elif isinstance(left, (list, tuple)):
            self.assertEqual(len(left), len(right), path)
            for index, (left_item, right_item) in enumerate(zip(left, right)):
                self.assertNestedEqual(left_item, right_item, f"{path}[{index}]")
        else:
            self.assertEqual(left, right, path)

    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.data_dir = self.root / "data"
        self.data_dir.mkdir()
        self.tokens = ("ACTION_FOLD", "ACTION_PASSIVE", "ACTION_ALL_IN")
        self.meta = {
            "version": "test-v1",
            "format": "trainer-test",
            "vocab_size": 6,
            "pad_token_id": 0,
            "block_size": 4,
            "decision_tokens": list(self.tokens),
            "decision_token_ids": [3, 4, 5],
            "stoi": {
                "PAD": 0,
                "CONTEXT_A": 1,
                "CONTEXT_B": 2,
                "ACTION_FOLD": 3,
                "ACTION_PASSIVE": 4,
                "ACTION_ALL_IN": 5,
            },
            "range_representative_ratio": {},
        }
        with (self.data_dir / "meta.pkl").open("wb") as handle:
            pickle.dump(self.meta, handle)
        trajectories = ([1, 1, 3], [1, 2, 4], [2, 1, 5])
        masks = ([0, 0, 1], [0, 0, 1], [0, 0, 1])
        for split in ("train", "val", "test"):
            (self.data_dir / f"{split}.bin").write_bytes(
                b"".join(struct.pack(f"<{len(row)}H", *row) for row in trajectories)
            )
            (self.data_dir / f"{split}_loss_mask.bin").write_bytes(
                b"".join(bytes(row) for row in masks)
            )
            (self.data_dir / f"{split}.idx").write_bytes(struct.pack("<3Q", 0, 3, 6))

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def _config(self, run_id: str, max_steps: int = 4) -> dict:
        return {
            "run_id": run_id,
            "data_dir": str(self.data_dir),
            "runs_dir": str(self.root / "runs"),
            "dataset": {
                "version": "test-v1",
                "format": "trainer-test",
                "vocab_size": 6,
                "pad_token_id": 0,
                "block_size": 4,
                "decision_tokens": list(self.tokens),
                "splits": ["train", "val", "test"],
            },
            "model": {
                "n_layer": 1,
                "n_head": 1,
                "n_embd": 8,
                "dropout": 0.0,
                "bias": False,
            },
            "training": {
                "seed": 1337,
                "optimizer": "adamw",
                "schedule": "cosine",
                "label_smoothing": 0.0,
                "device": "cpu",
                "precision": "float32",
                "micro_batch_size": 3,
                "gradient_accumulation_steps": 1,
                "loader_workers": 0,
                "pool_size": 3,
                "peak_learning_rate": 0.02,
                "minimum_learning_rate": 0.002,
                "adam_betas": [0.9, 0.95],
                "adam_epsilon": 1e-8,
                "weight_decay": 0.0,
                "gradient_clip": 1.0,
                "warmup_steps": 0,
                "max_steps": max_steps,
                "validation_interval": 2,
                "checkpoint_interval": 2,
                "archive_interval": 1000,
                "log_interval": 2,
                "deterministic": True,
                "allow_tf32": False,
                "compile": False,
            },
        }

    def test_decision_weighted_accumulation_matches_combined_batch(self) -> None:
        from poker_model.trainer import normalize_gradients

        torch.manual_seed(3)
        combined = torch.nn.Linear(2, 3, bias=False)
        accumulated = copy.deepcopy(combined)
        features = torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [2.0, 1.0]])
        targets = torch.tensor([0, 1, 2, 1])

        torch.nn.functional.cross_entropy(combined(features), targets).backward()
        for indexes in (slice(0, 1), slice(1, 4)):
            count = len(features[indexes])
            loss = torch.nn.functional.cross_entropy(
                accumulated(features[indexes]), targets[indexes]
            )
            (loss * count).backward()
        normalize_gradients(accumulated.parameters(), len(features))
        self.assertTrue(torch.allclose(combined.weight.grad, accumulated.weight.grad))

    def test_validation_loss_is_weighted_by_decision_count(self) -> None:
        from poker_model.trainer import MetricAccumulator

        accumulator = MetricAccumulator(self.meta)
        logits_one = torch.zeros((1, 1, 6))
        accumulator.update(
            logits_one, torch.tensor([[3]]), torch.tensor([[1]], dtype=torch.uint8), 2.0
        )
        logits_three = torch.zeros((1, 3, 6))
        accumulator.update(
            logits_three,
            torch.tensor([[3, 4, 5]]),
            torch.tensor([[1, 1, 1]], dtype=torch.uint8),
            4.0,
        )
        self.assertAlmostEqual(accumulator.compute()["loss"], 3.5)

    def test_four_steps_match_checkpoint_resume(self) -> None:
        from poker_model.checkpoint import load_checkpoint
        from poker_model.trainer import run_training

        uninterrupted_config = self._config("uninterrupted")
        uninterrupted_config["model"]["dropout"] = 0.1
        run_training(uninterrupted_config)
        uninterrupted = load_checkpoint(self.root / "runs" / "uninterrupted" / "latest.pt")

        resumed_config = self._config("resumed")
        resumed_config["model"]["dropout"] = 0.1
        run_training(resumed_config, stop_after_steps=2)
        resume_path = self.root / "runs" / "resumed" / "latest.pt"
        run_training(resumed_config, resume_path=resume_path)
        resumed = load_checkpoint(resume_path)

        self.assertEqual(uninterrupted["optimizer_step"], resumed["optimizer_step"])
        for key in ("model", "optimizer", "scheduler", "sampler_state", "counters"):
            self.assertNestedEqual(uninterrupted[key], resumed[key], key)

    def test_one_batch_overfits(self) -> None:
        from poker_model.trainer import run_training

        config = self._config("overfit", max_steps=100)
        config["training"]["peak_learning_rate"] = 0.05
        config["training"]["minimum_learning_rate"] = 0.005
        config["training"]["validation_interval"] = 100
        config["training"]["checkpoint_interval"] = 100
        config["training"]["log_interval"] = 100
        result = run_training(config)
        metrics = result["final_validation"]
        self.assertLess(metrics["loss"], 0.1)
        self.assertGreaterEqual(metrics["decision_accuracy"], 0.99)


if __name__ == "__main__":
    unittest.main()
