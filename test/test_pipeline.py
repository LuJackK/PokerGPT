from __future__ import annotations

import json
import pickle
import struct
import tempfile
import unittest
import zipfile
from decimal import Decimal
from pathlib import Path

from poker_pipeline.io_utils import read_jsonl
from poker_pipeline.manifest import ManifestOptions, build_manifest
from poker_pipeline.phh import parse_document, replay_hand
from poker_pipeline.prepare import PrepareOptions, prepare_dataset
from poker_pipeline.selection import SelectionOptions, select_dataset
from poker_pipeline.tokenizer import PokerTokenizer
from poker_pipeline.validate import validate_artifacts


HAND_1 = """\
variant = 'NT'
ante_trimming_status = true
antes = [0, 0, 0, 0, 0, 0]
blinds_or_straddles = [50, 100, 0, 0, 0, 0]
min_bet = 100
starting_stacks = [10000, 10000, 10000, 10000, 10000, 10000]
actions = ['d dh p1 6s6h', 'd dh p2 KdAs', 'd dh p3 8h5h', 'd dh p4 ThJc', 'd dh p5 Jh8c', 'd dh p6 Qh5s', 'p3 f', 'p4 f', 'p5 f', 'p6 f', 'p1 cc', 'p2 cbr 300', 'p1 cc', 'd db 9sAhTs', 'p1 cc', 'p2 cbr 480', 'p1 f']
hand = 47
players = ['redacted1', 'redacted2', 'redacted3', 'redacted4', 'redacted5', 'redacted6']
"""

HAND_2 = HAND_1.replace("hand = 47", "hand = 48").replace("6s6h", "2c2d")


class PipelineTests(unittest.TestCase):
    def setUp(self) -> None:
        test_root = Path(__file__).resolve().parent
        self.temporary = tempfile.TemporaryDirectory(dir=test_root)
        self.root = Path(self.temporary.name)
        self.archive = self.root / "fixture.zip"
        with zipfile.ZipFile(self.archive, "w", zipfile.ZIP_DEFLATED) as target:
            target.writestr("data/pluribus/session-a/47.phh", HAND_1)
            target.writestr("data/pluribus/session-b/48.phh", HAND_2)
            target.writestr("data/acpc/fixed.phh", HAND_1.replace("'NT'", "'FT'"))

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def test_manifest_and_selection(self) -> None:
        manifest = self.root / "manifest.jsonl"
        summary = build_manifest(self.archive, manifest, ManifestOptions(header_bytes=4096))
        self.assertEqual(summary["rows_added"], 3)
        rows = list(read_jsonl(manifest))
        self.assertEqual(rows[0]["variant"], "NT")
        self.assertEqual(rows[0]["player_count"], 6)
        selection = self.root / "selected.jsonl"
        selected_summary = select_dataset(
            manifest,
            selection,
            SelectionOptions(validation_fraction=0.5, split_seed="fixture"),
        )
        self.assertEqual(selected_summary["selected_rows"], 2)
        selected = list(read_jsonl(selection))
        self.assertTrue(all(row["source_folder"] == "pluribus" for row in selected))
        self.assertEqual(selected[0]["split_group"], "data/pluribus/session-a")
        self.assertEqual({row["split"] for row in selected}, {"train", "val"})

    def test_manifest_resume_repairs_partial_and_duplicate_rows(self) -> None:
        manifest = self.root / "manifest.jsonl"
        build_manifest(self.archive, manifest, ManifestOptions(header_bytes=4096))
        first = manifest.read_text(encoding="utf-8").splitlines()[0]
        with manifest.open("a", encoding="utf-8") as handle:
            handle.write(first + "\n")
            handle.write('{"member":')
        summary = build_manifest(
            self.archive, manifest, ManifestOptions(header_bytes=4096), resume=True
        )
        rows = list(read_jsonl(manifest))
        self.assertEqual(len(rows), 3)
        self.assertEqual(len({row["member"] for row in rows}), 3)
        self.assertEqual(summary["rows_preexisting"], 3)

    def test_replay_disambiguates_and_normalizes_exactly(self) -> None:
        hand = parse_document(HAND_1, "phh")[0][1]
        decisions = list(replay_hand("fixture.phh", "1", hand))
        actions = [decision.target_action for decision in decisions]
        self.assertEqual(actions[:7], ["FOLD", "FOLD", "FOLD", "FOLD", "CALL", "RAISE", "CALL"])
        raise_decision = decisions[5]
        self.assertEqual(raise_decision.target_amount, Decimal(200))
        self.assertEqual(raise_decision.target_amount_bb, Decimal(2))
        self.assertEqual(raise_decision.target_amount_pot, Decimal(1))
        flop_check = next(
            item for item in decisions if item.street == "FLOP" and item.target_action == "CHECK"
        )
        self.assertEqual(flop_check.to_call, 0)

    def test_tokenizer_masks_opponent_cards_and_future_board(self) -> None:
        hand = parse_document(HAND_1, "phh")[0][1]
        first = next(replay_hand("fixture.phh", "1", hand))
        tokenizer = PokerTokenizer()
        encoded = tokenizer.encode_decision(first)
        self.assertIn("CARD_8h", encoded.tokens)  # acting player's cards
        self.assertIn("CARD_5h", encoded.tokens)
        self.assertNotIn("CARD_6s", encoded.tokens)  # opponent private card
        self.assertNotIn("CARD_Ah", encoded.tokens)  # future flop card
        self.assertEqual(encoded.tokens.count("CARD_UNKNOWN"), 10)
        self.assertFalse(any(token.startswith("LEGAL_") for token in encoded.tokens))
        self.assertFalse(any(token.startswith("LEGAL_") for token in tokenizer.itos))
        target_positions = [i for i, bit in enumerate(encoded.loss_mask) if bit]
        self.assertEqual([encoded.tokens[i] for i in target_positions], ["ACTION_FOLD"])

    def test_end_to_end_binary_alignment_and_metadata(self) -> None:
        selection = self.root / "selected.jsonl"
        rows = [
            {
                "member": "data/pluribus/session-a/47.phh",
                "source_folder": "pluribus",
                "split": "train",
                "selected_player_counts": [6],
            },
            {
                "member": "data/pluribus/session-b/48.phh",
                "source_folder": "pluribus",
                "split": "val",
                "selected_player_counts": [6],
            },
        ]
        selection.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
        output = self.root / "processed"
        stats = prepare_dataset(
            self.archive,
            selection,
            output,
            PrepareOptions(block_size=256, audit_samples=2),
        )
        self.assertGreater(stats["writer_examples"]["train"], 0)
        self.assertGreater(stats["writer_examples"]["val"], 0)
        for split in ("train", "val"):
            token_bytes = (output / f"{split}.bin").read_bytes()
            masks = (output / f"{split}_loss_mask.bin").read_bytes()
            indexes = (output / f"{split}.idx").read_bytes()
            self.assertEqual(len(token_bytes) // 2, len(masks))
            self.assertEqual(len(indexes) // 8, stats["writer_examples"][split])
            if indexes:
                self.assertEqual(struct.unpack("<Q", indexes[:8])[0], 0)
        with (output / "meta.pkl").open("rb") as handle:
            meta = pickle.load(handle)
        self.assertEqual(meta["token_dtype"], "uint16_le")
        self.assertIn("opponent private card", meta["privacy"])
        self.assertEqual(json.loads((output / "stats.json").read_text())["parse_error_count"], 0)
        validation = validate_artifacts(output, selection)
        self.assertTrue(validation["valid"], validation["errors"])


if __name__ == "__main__":
    unittest.main()
