from __future__ import annotations

import json
import pickle
import struct
import tempfile
import unittest
from pathlib import Path

import torch

from poker_model.checkpoint import dataset_fingerprint, save_checkpoint
from poker_model.run_tracking import (
    build_run_identity,
    canonical_json_sha256,
    write_run_identity,
)


class RunTrackingTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.run_dir = self.root / "run"
        self.data_dir = self.root / "data"
        self.run_dir.mkdir()
        self.data_dir.mkdir()
        with (self.data_dir / "meta.pkl").open("wb") as handle:
            pickle.dump({"version": "fixture"}, handle)
        for index, split in enumerate(("train", "val", "test"), start=1):
            (self.data_dir / f"{split}.bin").write_bytes(struct.pack("<H", index))
            (self.data_dir / f"{split}_loss_mask.bin").write_bytes(b"\1")
            (self.data_dir / f"{split}.idx").write_bytes(struct.pack("<Q", 0))
        self.fingerprint = dataset_fingerprint(self.data_dir)
        self.config = {
            "run_id": "fixture-seed7",
            "training": {"seed": 7},
        }
        self.environment = {"python": "fixture", "git_commit": "abc123", "git_dirty": False}
        (self.run_dir / "config.json").write_text(
            json.dumps(self.config), encoding="utf-8"
        )
        (self.run_dir / "environment.json").write_text(
            json.dumps(self.environment), encoding="utf-8"
        )
        (self.run_dir / "dataset_fingerprint.json").write_text(
            json.dumps(self.fingerprint), encoding="utf-8"
        )
        (self.run_dir / "metrics.jsonl").write_text("{}\n", encoding="utf-8")
        self.bundle = self.root / "bundle.zip"
        self.bundle.write_bytes(b"fixture bundle")
        self.checkpoint = self.run_dir / "best.pt"
        save_checkpoint(
            self.checkpoint,
            {
                "model": {},
                "model_config": {},
                "optimizer": {},
                "scheduler": {},
                "scaler": None,
                "optimizer_step": 12,
                "epoch": 1,
                "next_batch_cursor": 0,
                "best_validation_loss": 0.75,
                "elapsed_seconds": 1.0,
                "counters": {},
                "rng_state": {},
                "sampler_state": {},
                "training_config": self.config,
                "dataset_fingerprint": self.fingerprint,
                "preprocessing_identity": {},
                "environment": self.environment,
            },
        )

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def test_canonical_hash_ignores_json_key_order(self) -> None:
        self.assertEqual(
            canonical_json_sha256({"a": 1, "b": 2}),
            canonical_json_sha256({"b": 2, "a": 1}),
        )

    def test_builds_verified_write_once_identity(self) -> None:
        identity = build_run_identity(
            self.run_dir,
            checkpoint_path=self.checkpoint,
            artifact_bundle_path=self.bundle,
        )
        self.assertEqual(identity["seed"], 7)
        self.assertEqual(identity["optimizer_step"], 12)
        self.assertTrue(identity["checkpoint_environment_matches_record"])
        output = self.run_dir / "run_identity.json"
        write_run_identity(output, identity)
        with self.assertRaises(FileExistsError):
            write_run_identity(output, identity)


if __name__ == "__main__":
    unittest.main()
