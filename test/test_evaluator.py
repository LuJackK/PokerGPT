from __future__ import annotations

import json
import tempfile
import unittest
import zipfile
from pathlib import Path

import torch

from poker_model.checkpoint import dataset_fingerprint, save_checkpoint
from poker_model.evaluation_access import (
    FINAL_TEST_CONFIRMATION,
    create_test_access_receipt,
)
from poker_model.evaluation_freeze import (
    build_freeze_manifest,
    verify_freeze_manifest,
    write_freeze_manifest,
)
from poker_model.evaluator import (
    evaluate_split,
    report_markdown,
    write_json,
)
from poker_model.model import GPT, GPTConfig
from poker_model.run_tracking import file_identity
from poker_pipeline.prepare import PrepareOptions, prepare_dataset


FOUR_STREET_HAND = """\
variant = 'NT'
antes = [0, 0, 0, 0, 0, 0]
blinds_or_straddles = [50, 100, 0, 0, 0, 0]
min_bet = 100
starting_stacks = [10000, 10000, 10000, 10000, 10000, 10000]
actions = ['d dh p1 6s6h', 'd dh p2 KdAs', 'd dh p3 8h5h', 'd dh p4 ThJc',
  'd dh p5 Jh8c', 'd dh p6 Qh5s', 'p3 f', 'p4 f', 'p5 f', 'p6 f',
  'p1 cc', 'p2 cc', 'd db 9sAhTs', 'p1 cc', 'p2 cbr 200', 'p1 cc',
  'd db 2c', 'p1 cc', 'p2 cc', 'd db 3d', 'p1 cc', 'p2 cc']
"""


class EvaluatorEndToEndTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.archive = self.root / "fixture.zip"
        member = "data/pluribus/session-a/1.phh"
        with zipfile.ZipFile(self.archive, "w", zipfile.ZIP_DEFLATED) as target:
            target.writestr(member, FOUR_STREET_HAND)
        with zipfile.ZipFile(self.archive) as source:
            info = source.getinfo(member)
        self.selection = self.root / "selection.jsonl"
        row = {
            "member": member,
            "source_folder": "pluribus",
            "split": "val",
            "selected_player_counts": [6],
            "crc32": f"{info.CRC:08x}",
            "compressed_size": info.compress_size,
            "uncompressed_size": info.file_size,
        }
        self.selection.write_text(json.dumps(row) + "\n", encoding="utf-8")
        self.data_dir = self.root / "processed"
        prepare_dataset(
            self.archive,
            self.selection,
            self.data_dir,
            PrepareOptions(block_size=320, audit_samples=0),
        )
        fingerprint = dataset_fingerprint(self.data_dir)
        model_config = GPTConfig(
            block_size=320,
            vocab_size=105,
            n_layer=1,
            n_head=1,
            n_embd=8,
            dropout=0.0,
            bias=False,
        )
        torch.manual_seed(7)
        model = GPT(model_config)
        training_config = {"training": {"seed": 7}}
        environment = {"python": "fixture"}
        self.checkpoint = self.root / "best.pt"
        save_checkpoint(
            self.checkpoint,
            {
                "model": model.state_dict(),
                "model_config": {
                    "block_size": 320,
                    "vocab_size": 105,
                    "n_layer": 1,
                    "n_head": 1,
                    "n_embd": 8,
                    "dropout": 0.0,
                    "bias": False,
                },
                "optimizer": {},
                "scheduler": {},
                "scaler": None,
                "optimizer_step": 7,
                "epoch": 1,
                "next_batch_cursor": 0,
                "best_validation_loss": 1.0,
                "elapsed_seconds": 1.0,
                "counters": {},
                "rng_state": {},
                "sampler_state": {},
                "training_config": training_config,
                "dataset_fingerprint": fingerprint,
                "preprocessing_identity": {},
                "environment": environment,
            },
        )
        self.bundle = self.root / "bundle.zip"
        self.bundle.write_bytes(b"fixture bundle")
        self.config_path = self.root / "evaluator.json"
        self.config = {
            "format_version": 1,
            "evaluator_id": "fixture-evaluator-v1",
            "prediction_policy": "decision_token_argmax",
            "decision_top_k": 3,
            "mapped_action_top_k": 2,
            "batch_size": 2,
            "device": "cpu",
            "precision": "float32",
            "candidate": {
                "run_id": "fixture",
                "seed": 7,
                "optimizer_step": 7,
                "checkpoint_sha256": file_identity(self.checkpoint)["sha256"],
                "configuration_sha256": "fixture",
                "dataset_fingerprint": fingerprint["digest"],
                "artifact_bundle_sha256": file_identity(self.bundle)["sha256"],
            },
            "freeze_files": ["evaluator.json"],
        }
        self.config_path.write_text(
            json.dumps(self.config, indent=2), encoding="utf-8"
        )

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def test_complete_validation_evaluation_and_freeze_tamper_check(self) -> None:
        report = evaluate_split(
            self.config,
            split="val",
            zip_path=self.archive,
            selection_path=self.selection,
            data_dir=self.data_dir,
            checkpoint_path=self.checkpoint,
        )
        self.assertTrue(report["complete_split"])
        self.assertGreater(report["metrics"]["decisions"], 0)
        self.assertEqual(
            set(report["metrics"]["per_street"]),
            {"PREFLOP", "FLOP", "TURN", "RIVER"},
        )
        self.assertIn("action_confusion_matrix", report["metrics"])
        self.assertIn("Sizing-range error", report_markdown(report))

        validation_path = self.root / "validation.json"
        write_json(validation_path, report, exclusive=False)
        run_identity = self.root / "run_identity.json"
        run_identity.write_text("{}\n", encoding="utf-8")
        manifest = build_freeze_manifest(
            repo_root=self.root,
            config_path=self.config_path,
            validation_report_path=validation_path,
            checkpoint_path=self.checkpoint,
            artifact_bundle_path=self.bundle,
            run_identity_path=run_identity,
        )
        manifest_path = self.root / "freeze.json"
        write_freeze_manifest(manifest_path, manifest)
        verify_freeze_manifest(
            manifest_path=manifest_path,
            repo_root=self.root,
            config_path=self.config_path,
            checkpoint_path=self.checkpoint,
            artifact_bundle_path=self.bundle,
            run_identity_path=run_identity,
        )
        self.config_path.write_text("{}\n", encoding="utf-8")
        with self.assertRaises((ValueError, KeyError)):
            verify_freeze_manifest(
                manifest_path=manifest_path,
                repo_root=self.root,
                config_path=self.config_path,
                checkpoint_path=self.checkpoint,
                artifact_bundle_path=self.bundle,
                run_identity_path=run_identity,
            )

    def test_one_time_access_receipt_requires_confirmation_and_is_exclusive(self) -> None:
        receipt = self.root / "receipt.json"
        with self.assertRaises(PermissionError):
            create_test_access_receipt(
                receipt,
                freeze_manifest_sha256="frozen",
                confirmation="no",
                checkpoint_sha256="checkpoint",
            )
        permit = create_test_access_receipt(
            receipt,
            freeze_manifest_sha256="frozen",
            confirmation=FINAL_TEST_CONFIRMATION,
            checkpoint_sha256="checkpoint",
        )
        self.assertEqual(permit.split, "test")
        with self.assertRaises(FileExistsError):
            create_test_access_receipt(
                receipt,
                freeze_manifest_sha256="frozen",
                confirmation=FINAL_TEST_CONFIRMATION,
                checkpoint_sha256="checkpoint",
            )


if __name__ == "__main__":
    unittest.main()
