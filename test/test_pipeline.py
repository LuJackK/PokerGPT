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
from poker_pipeline.phh import Decision, build_hero_trajectories, parse_document, replay_hand
from poker_pipeline.prepare import PrepareOptions, prepare_dataset
from poker_pipeline.selection import SelectionOptions, select_dataset
from poker_pipeline.tokenizer import (
    CONTEXT_ONLY_RANGE_TOKENS,
    DECISION_TOKENS,
    POSITION_TOKENS,
    RANGE_TOKENS,
    SIZING_OUTPUT_TOKENS,
    PokerTokenizer,
    ratio_label,
)
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
ALL_IN_HAND = HAND_1.replace(
    "'p2 cbr 300', 'p1 cc', 'd db 9sAhTs', 'p1 cc', 'p2 cbr 480', 'p1 f'",
    "'p2 cbr 10000', 'p1 f'",
)


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

    def test_amount_bucket_names_are_non_overlapping_ranges(self) -> None:
        self.assertEqual(ratio_label(0), "ZERO")
        self.assertEqual(ratio_label(Decimal("0.5")), "0.25_TO_0.5")
        self.assertEqual(ratio_label(Decimal("1.5")), "1_TO_1.5")
        self.assertEqual(ratio_label(Decimal("2.5")), "2_TO_3")
        self.assertEqual(ratio_label(Decimal("50")), "20_TO_50")
        self.assertEqual(ratio_label(Decimal("50.01")), "50_TO_75")
        self.assertEqual(ratio_label(Decimal("75")), "50_TO_75")
        self.assertEqual(ratio_label(Decimal("75.01")), "75_TO_100")
        self.assertEqual(ratio_label(Decimal("100")), "75_TO_100")
        self.assertEqual(ratio_label(Decimal("100.01")), "100_TO_150")
        self.assertEqual(ratio_label(Decimal("150")), "100_TO_150")
        self.assertEqual(ratio_label(Decimal("150.01")), "GT_150")

    def test_tokenizer_builds_complete_relative_hero_trajectories(self) -> None:
        hand = parse_document(HAND_1, "phh")[0][1]
        decisions = list(replay_hand("fixture.phh", "1", hand))
        trajectories = build_hero_trajectories(decisions)
        tokenizer = PokerTokenizer()
        self.assertEqual(len(tokenizer.itos), 105)
        self.assertNotIn("RANGE_GT_50", tokenizer.stoi)
        self.assertIn("RANGE_50_TO_75", tokenizer.stoi)
        self.assertIn("RANGE_75_TO_100", tokenizer.stoi)
        self.assertIn("RANGE_100_TO_150", tokenizer.stoi)
        self.assertIn("RANGE_GT_150", tokenizer.stoi)
        self.assertIn("ACTION_PASSIVE", tokenizer.stoi)
        self.assertIn("ACTION_ALL_IN", tokenizer.stoi)
        self.assertNotIn("RANGE_ZERO", DECISION_TOKENS)
        self.assertEqual(
            SIZING_OUTPUT_TOKENS,
            RANGE_TOKENS[:10],
        )
        self.assertNotIn("RANGE_20_TO_50", DECISION_TOKENS)
        self.assertIn("RANGE_20_TO_50", CONTEXT_ONLY_RANGE_TOKENS)
        self.assertIn("RANGE_20_TO_50", tokenizer.stoi)
        self.assertNotIn("TABLE_SIZE", tokenizer.stoi)
        self.assertNotIn("POST_BLIND", tokenizer.stoi)
        self.assertNotIn("POST_ANTE", tokenizer.stoi)
        self.assertNotIn("POST_STRADDLE", tokenizer.stoi)
        self.assertNotIn("VALUE_BB_0.5", tokenizer.stoi)
        self.assertNotIn("VALUE_BB_1", tokenizer.stoi)
        self.assertNotIn("COUNT_2", tokenizer.stoi)
        self.assertNotIn("COUNT_0", tokenizer.stoi)
        self.assertNotIn("COUNT_6", tokenizer.stoi)
        self.assertNotIn("PLAYER_7", tokenizer.stoi)
        self.assertNotIn("<PLAYER_STATES>", tokenizer.stoi)
        self.assertNotIn("STACK_BB", tokenizer.stoi)
        self.assertIn("STACK_POT", tokenizer.stoi)

        expected_positions = {
            0: "POSITION_SMALL_BLIND",
            1: "POSITION_BIG_BLIND",
            2: "POSITION_UTG",
            3: "POSITION_HIJACK",
            4: "POSITION_CUTOFF",
            5: "POSITION_BUTTON",
        }
        for trajectory in trajectories:
            encoded_position = tokenizer.encode_trajectory(trajectory).tokens[1]
            self.assertEqual(encoded_position, expected_positions[trajectory.hero])
            self.assertIn(encoded_position, POSITION_TOKENS)

        early_fold = next(trajectory for trajectory in trajectories if trajectory.hero == 2)
        encoded = tokenizer.encode_trajectory(early_fold)
        self.assertEqual(encoded.tokens[:2], ("<BOS>", "POSITION_UTG"))
        self.assertIn("CARD_8h", encoded.tokens)  # acting player's cards
        self.assertIn("CARD_5h", encoded.tokens)
        self.assertNotIn("CARD_6s", encoded.tokens)  # opponent private card
        self.assertNotIn("CARD_Ah", encoded.tokens)  # future flop card
        self.assertNotIn("<EVENT_SEQUENCE>", encoded.tokens)
        self.assertFalse(any(token.startswith("STREET_") for token in encoded.tokens))
        self.assertFalse(any(token.startswith("HIST_STREET_") for token in encoded.tokens))
        self.assertFalse(any(token.startswith("PLAYER_REL_") for token in encoded.tokens))
        self.assertFalse(any(token.startswith("LEGAL_") for token in encoded.tokens))
        self.assertFalse(any(token.startswith("LEGAL_") for token in tokenizer.itos))
        target_positions = [i for i, bit in enumerate(encoded.loss_mask) if bit]
        self.assertEqual([encoded.tokens[i] for i in target_positions], ["ACTION_FOLD"])

        multi_decision = next(trajectory for trajectory in trajectories if trajectory.hero == 0)
        full = tokenizer.encode_trajectory(multi_decision)
        self.assertEqual(full.tokens.count("<PLAYER_1_DECISION>"), multi_decision.decision_count)
        self.assertEqual(
            full.tokens.count("PLAYER_1_HOLE_CARDS"), multi_decision.decision_count
        )
        postflop_decisions = sum(
            bool(decision.board)
            for decision in multi_decision.items
            if isinstance(decision, Decision)
        )
        self.assertEqual(full.tokens.count("CURRENT_BOARD"), postflop_decisions)
        self.assertNotIn("COUNT_0", full.tokens)
        self.assertNotIn("<PLAYER_STATES>", full.tokens)
        self.assertNotIn("STACK_BB", full.tokens)
        self.assertIn("STACK_POT", full.tokens)
        self.assertIn("BOARD_REVEAL", full.tokens)
        self.assertIn("COUNT_3", full.tokens)
        self.assertIn("CARD_9s", full.tokens)
        self.assertIn("POT_SIZE_BB", full.tokens)
        self.assertFalse(any(token.startswith("POT_SIZE_BB_") for token in full.tokens))
        supervised_tokens = [
            full.tokens[index]
            for index, bit in enumerate(full.loss_mask)
            if bit
        ]
        self.assertEqual(supervised_tokens, ["ACTION_PASSIVE", "ACTION_PASSIVE", "ACTION_PASSIVE", "ACTION_FOLD"])
        self.assertEqual(len(supervised_tokens), multi_decision.decision_count)

        aggressive = next(trajectory for trajectory in trajectories if trajectory.hero == 1)
        aggressive_encoded = tokenizer.encode_trajectory(aggressive)
        aggressive_targets = [
            aggressive_encoded.tokens[index]
            for index, bit in enumerate(aggressive_encoded.loss_mask)
            if bit
        ]
        self.assertEqual(aggressive_targets, ["RANGE_0.75_TO_1", "RANGE_0.75_TO_1"])
        self.assertEqual(len(aggressive_targets), aggressive.decision_count)

        decision_offsets = [
            index
            for index, token in enumerate(full.tokens)
            if token == "<PLAYER_1_DECISION>"
        ]
        expected_boards = [
            decision.board for decision in multi_decision.items if isinstance(decision, Decision)
        ]
        for offset, expected_board in zip(decision_offsets, expected_boards):
            self.assertEqual(full.tokens[offset + 1], "PLAYER_1_HOLE_CARDS")
            self.assertEqual(full.tokens[offset + 2 : offset + 4], ("CARD_6s", "CARD_6h"))
            if expected_board:
                self.assertEqual(full.tokens[offset + 4], "CURRENT_BOARD")
                self.assertEqual(full.tokens[offset + 5], f"COUNT_{len(expected_board)}")
                self.assertEqual(
                    full.tokens[offset + 6 : offset + 6 + len(expected_board)],
                    tuple(f"CARD_{card}" for card in expected_board),
                )
            else:
                self.assertEqual(full.tokens[offset + 4], "POT_SIZE_BB")

    def test_aggressive_all_in_is_one_supervised_token(self) -> None:
        hand = parse_document(ALL_IN_HAND, "phh")[0][1]
        decisions = list(replay_hand("all-in.phh", "1", hand))
        all_in = next(decision for decision in decisions if decision.actor == 1)
        self.assertEqual(all_in.target_action, "RAISE")
        self.assertEqual(all_in.target_amount, all_in.hero_stack)
        trajectory = next(
            item for item in build_hero_trajectories(decisions) if item.hero == 1
        )
        encoded = PokerTokenizer().encode_trajectory(trajectory)
        targets = [encoded.tokens[index] for index, bit in enumerate(encoded.loss_mask) if bit]
        self.assertEqual(targets, ["ACTION_ALL_IN"])

        folding_trajectory = next(
            item for item in build_hero_trajectories(decisions) if item.hero == 0
        )
        folding_tokens = PokerTokenizer().encode_trajectory(folding_trajectory).tokens
        all_in_status = folding_tokens.index("STATUS_ALL_IN")
        self.assertNotEqual(folding_tokens[all_in_status + 1], "STACK_POT")

    def test_encoder_enforces_fixed_game_and_sizing_output_contracts(self) -> None:
        ante_hand = parse_document(
            HAND_1.replace(
                "antes = [0, 0, 0, 0, 0, 0]",
                "antes = [10, 10, 10, 10, 10, 10]",
            ),
            "phh",
        )[0][1]
        ante_trajectory = build_hero_trajectories(
            list(replay_hand("ante.phh", "1", ante_hand))
        )[0]
        with self.assertRaisesRegex(ValueError, "no antes/straddles"):
            PokerTokenizer().encode_trajectory(ante_trajectory)

        short_stack_hand = parse_document(
            HAND_1.replace("10000", "9900"), "phh"
        )[0][1]
        short_stack_trajectory = build_hero_trajectories(
            list(replay_hand("short.phh", "1", short_stack_hand))
        )[0]
        with self.assertRaisesRegex(ValueError, "start at 100 BB"):
            PokerTokenizer().encode_trajectory(short_stack_trajectory)

        deep_raise_hand = parse_document(
            HAND_1.replace("'p2 cbr 300'", "'p2 cbr 5000'"), "phh"
        )[0][1]
        deep_raise_trajectory = next(
            trajectory
            for trajectory in build_hero_trajectories(
                list(replay_hand("deep-raise.phh", "1", deep_raise_hand))
            )
            if trajectory.hero == 1
        )
        with self.assertRaisesRegex(ValueError, "context-only"):
            PokerTokenizer().encode_trajectory(deep_raise_trajectory)

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
            PrepareOptions(block_size=320, audit_samples=2),
        )
        self.assertGreater(stats["writer_trajectories"]["train"], 0)
        self.assertGreater(stats["writer_trajectories"]["val"], 0)
        for split in ("train", "val"):
            token_bytes = (output / f"{split}.bin").read_bytes()
            masks = (output / f"{split}_loss_mask.bin").read_bytes()
            indexes = (output / f"{split}.idx").read_bytes()
            self.assertEqual(len(token_bytes) // 2, len(masks))
            self.assertEqual(len(indexes) // 8, stats["writer_trajectories"][split])
            if indexes:
                self.assertEqual(struct.unpack("<Q", indexes[:8])[0], 0)
        with (output / "meta.pkl").open("rb") as handle:
            meta = pickle.load(handle)
        self.assertEqual(meta["token_dtype"], "uint16_le")
        self.assertEqual(meta["version"], "0.8.0")
        self.assertEqual(
            meta["format"],
            "pluribus_6max_100bb_spr_position_single_decision_v5",
        )
        self.assertEqual(meta["block_size"], 320)
        self.assertEqual(meta["decision_tokens"], list(DECISION_TOKENS))
        self.assertEqual(meta["sizing_output_tokens"], list(SIZING_OUTPUT_TOKENS))
        self.assertEqual(
            meta["context_only_range_tokens"], list(CONTEXT_ONLY_RANGE_TOKENS)
        )
        self.assertEqual(meta["position_tokens"], list(POSITION_TOKENS))
        self.assertNotIn("RANGE_ZERO", meta["decision_tokens"])
        self.assertNotIn("RANGE_20_TO_50", meta["decision_tokens"])
        self.assertNotIn("RANGE_20_TO_50", meta["range_representative_ratio"])
        self.assertIn("RANGE_20_TO_50", meta["range_tokens"])
        self.assertEqual(
            meta["range_representative_source"]["RANGE_0.75_TO_1"],
            "training_split_median",
        )
        self.assertEqual(meta["range_representative_ratio"]["RANGE_0.75_TO_1"], "0.9")
        self.assertEqual(meta["privacy"], "one complete trajectory per hero; opponent private cards omitted")
        self.assertEqual(meta["game_contract"]["starting_depth_bb"], "100")
        output_stats = json.loads((output / "stats.json").read_text())
        self.assertEqual(output_stats["parse_error_count"], 0)
        self.assertEqual(
            output_stats["token_frequency"]["ACTION_FOLD"]["supervised"],
            output_stats["counts"]["decision_token:ACTION_FOLD"],
        )
        self.assertEqual(output_stats["token_frequency"]["<PAD>"]["total"], 0)
        validation = validate_artifacts(output, selection)
        self.assertTrue(validation["valid"], validation["errors"])

    def test_preparation_rejects_selection_outside_fixed_baseline_contract(self) -> None:
        selection = self.root / "selected-invalid.jsonl"
        selection.write_text(
            json.dumps(
                {
                    "member": "data/pluribus/session-a/47.phh",
                    "source_folder": "acpc",
                    "split": "train",
                    "selected_player_counts": [6],
                }
            )
            + "\n",
            encoding="utf-8",
        )
        with self.assertRaisesRegex(ValueError, "only selected Pluribus six-max"):
            prepare_dataset(self.archive, selection, self.root / "invalid-output")


if __name__ == "__main__":
    unittest.main()
