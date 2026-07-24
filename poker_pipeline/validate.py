from __future__ import annotations

import hashlib
import json
import pickle
import struct
from array import array
from collections import Counter
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any

from . import ARTIFACT_FORMAT, PREPARED_OUTPUT_NAMES, __version__
from .io_utils import read_jsonl
from .tokenizer import (
    CONTEXT_ONLY_RANGE_TOKENS,
    DECISION_TOKENS,
    POSITION_TOKENS,
    RANGE_THRESHOLDS,
    RANGE_TOKENS,
    SIZING_OUTPUT_TOKENS,
    build_vocabulary,
)

def validate_artifacts(output_dir: Path, selection_path: Path | None = None) -> dict[str, Any]:
    output_dir = Path(output_dir)
    with (output_dir / "meta.pkl").open("rb") as handle:
        meta = pickle.load(handle)
    vocab_size = int(meta["vocab_size"])
    itos = meta["itos"]
    bos_id = meta["stoi"]["<BOS>"]
    eos_id = meta["stoi"]["<EOS>"]
    decision_id = meta["stoi"]["<PLAYER_1_DECISION>"]
    block_size = int(meta["block_size"])
    expected_decision_tokens = set(DECISION_TOKENS)
    metadata_decision_tokens = set(meta.get("decision_tokens", ()))
    report: dict[str, Any] = {"splits": {}, "errors": []}

    manifest_path = output_dir / "preprocessing_manifest.json"
    if not manifest_path.exists():
        report["errors"].append("preprocessing_manifest.json is missing")
    else:
        preprocessing_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if preprocessing_manifest.get("pipeline_version") != __version__:
            report["errors"].append(
                "preprocessing manifest pipeline version does not match the pipeline"
            )
        declared_outputs = preprocessing_manifest.get("outputs", {})
        if set(declared_outputs) != set(PREPARED_OUTPUT_NAMES):
            report["errors"].append(
                "preprocessing manifest output set does not match the pipeline"
            )
        else:
            for name in PREPARED_OUTPUT_NAMES:
                path = output_dir / name
                if not path.exists():
                    report["errors"].append(f"preprocessing output is missing: {name}")
                    continue
                expected = declared_outputs[name]
                digest = hashlib.sha256(path.read_bytes()).hexdigest()
                if (
                    expected.get("bytes") != path.stat().st_size
                    or expected.get("sha256") != digest
                ):
                    report["errors"].append(
                        f"preprocessing output size/hash mismatch: {name}"
                    )
    if meta.get("version") != __version__:
        report["errors"].append("meta.pkl pipeline version does not match the pipeline")
    if meta.get("format") != ARTIFACT_FORMAT:
        report["errors"].append("meta.pkl artifact format does not match the pipeline")
    expected_vocabulary = build_vocabulary()
    if itos != expected_vocabulary or vocab_size != len(expected_vocabulary):
        report["errors"].append("meta.pkl vocabulary does not match the pipeline")
    if meta.get("stoi") != {token: index for index, token in enumerate(itos)}:
        report["errors"].append("meta.pkl stoi/itos mappings are inconsistent")
    if metadata_decision_tokens != expected_decision_tokens:
        report["errors"].append("meta.pkl decision-token set does not match the pipeline")
    if "RANGE_ZERO" in metadata_decision_tokens:
        report["errors"].append("meta.pkl permits RANGE_ZERO as a hero decision")
    if tuple(meta.get("range_tokens", ())) != RANGE_TOKENS:
        report["errors"].append("meta.pkl range-token set does not match the pipeline")
    if tuple(meta.get("sizing_output_tokens", ())) != SIZING_OUTPUT_TOKENS:
        report["errors"].append("meta.pkl sizing-output set does not match the pipeline")
    if tuple(meta.get("context_only_range_tokens", ())) != CONTEXT_ONLY_RANGE_TOKENS:
        report["errors"].append("meta.pkl context-only range set does not match the pipeline")
    if tuple(meta.get("position_tokens", ())) != POSITION_TOKENS:
        report["errors"].append("meta.pkl position-token set does not match the pipeline")
    representatives = meta.get("range_representative_ratio", {})
    if set(representatives) != set(SIZING_OUTPUT_TOKENS):
        report["errors"].append(
            "meta.pkl does not define exactly one representative for every sizing output"
        )
    else:
        for token in SIZING_OUTPUT_TOKENS:
            index = RANGE_TOKENS.index(token)
            lower = Decimal(0) if index == 0 else RANGE_THRESHOLDS[index - 1]
            try:
                representative = Decimal(str(representatives[token]))
            except (InvalidOperation, ValueError):
                report["errors"].append(
                    f"meta.pkl has an invalid representative for {token}"
                )
                continue
            upper = RANGE_THRESHOLDS[index] if index < len(RANGE_THRESHOLDS) else None
            if (
                not representative.is_finite()
                or representative <= lower
                or upper is not None and representative > upper
            ):
                report["errors"].append(
                    f"meta.pkl representative {representative} is outside {token}"
                )

    expected_game_contract = {
        "variant": "NT",
        "players": 6,
        "starting_depth_bb": "100",
        "antes": False,
        "straddles": False,
        "blinds_bb": ["0.5", "1"],
    }
    if meta.get("game_contract") != expected_game_contract:
        report["errors"].append("meta.pkl game contract does not match the baseline")

    token_counts: Counter[str] = Counter()
    context_token_counts: Counter[str] = Counter()
    supervised_token_counts: Counter[str] = Counter()
    for split in ("train", "val", "test"):
        token_path = output_dir / f"{split}.bin"
        mask_path = output_dir / f"{split}_loss_mask.bin"
        index_path = output_dir / f"{split}.idx"
        token_bytes = token_path.read_bytes()
        masks = mask_path.read_bytes()
        index_bytes = index_path.read_bytes()
        if len(token_bytes) % 2:
            report["errors"].append(f"{split}: token file has odd byte length")
        tokens = array("H")
        tokens.frombytes(token_bytes)
        offsets = [value[0] for value in struct.iter_unpack("<Q", index_bytes)]
        if len(tokens) != len(masks):
            report["errors"].append(f"{split}: token/mask length mismatch")
        if any(token >= vocab_size for token in tokens):
            report["errors"].append(f"{split}: token ID outside vocabulary")
        for token_id, supervised in zip(tokens, masks):
            if token_id >= len(itos):
                continue
            token = itos[token_id]
            token_counts[token] += 1
            if supervised:
                supervised_token_counts[token] += 1
            else:
                context_token_counts[token] += 1
        if any(mask not in (0, 1) for mask in masks):
            report["errors"].append(f"{split}: loss mask contains value other than 0/1")
        if offsets != sorted(set(offsets)):
            report["errors"].append(f"{split}: indexes are not unique and increasing")
        lengths: list[int] = []
        masked = 0
        missing_position_prefixes = 0
        invalid_decision_observations = 0
        for number, start in enumerate(offsets):
            end = offsets[number + 1] if number + 1 < len(offsets) else len(tokens)
            lengths.append(end - start)
            if end <= start or tokens[start] != bos_id or tokens[end - 1] != eos_id:
                report["errors"].append(f"{split}: invalid framing at example {number}")
                continue
            if end - start < 3 or itos[tokens[start + 1]] not in POSITION_TOKENS:
                missing_position_prefixes += 1
            if end - start > block_size:
                report["errors"].append(f"{split}: example {number} exceeds block size")
            example_masked = [index for index in range(start, end) if masks[index]]
            if not example_masked:
                report["errors"].append(f"{split}: example {number} has no supervised token")
            for index in example_masked:
                token = itos[tokens[index]]
                if token not in expected_decision_tokens:
                    report["errors"].append(
                        f"{split}: invalid supervised token {token} at token {index}"
                    )
            decision_positions = [
                index for index in range(start, end) if tokens[index] == decision_id
            ]
            decision_count = len(decision_positions)
            visible_board: list[str] = []
            for index in range(start, end):
                token = itos[tokens[index]]
                if token == "BOARD_REVEAL":
                    try:
                        reveal_count = int(
                            itos[tokens[index + 1]].removeprefix("COUNT_")
                        )
                        reveal_cards = [
                            itos[tokens[position]]
                            for position in range(index + 2, index + 2 + reveal_count)
                        ]
                    except (IndexError, ValueError):
                        continue
                    if reveal_count in {1, 3} and all(
                        card.startswith("CARD_") for card in reveal_cards
                    ):
                        visible_board.extend(reveal_cards)
                    continue
                if token != "<PLAYER_1_DECISION>":
                    continue
                valid_observation = True
                try:
                    if itos[tokens[index + 1]] != "PLAYER_1_HOLE_CARDS" or not all(
                        itos[tokens[position]].startswith("CARD_")
                        for position in (index + 2, index + 3)
                    ):
                        valid_observation = False
                    cursor = index + 4
                    if visible_board:
                        if itos[tokens[cursor]] != "CURRENT_BOARD":
                            valid_observation = False
                        board_count = int(
                            itos[tokens[cursor + 1]].removeprefix("COUNT_")
                        )
                        repeated_board = [
                            itos[tokens[position]]
                            for position in range(cursor + 2, cursor + 2 + board_count)
                        ]
                        if (
                            board_count != len(visible_board)
                            or board_count not in {3, 4, 5}
                            or repeated_board != visible_board
                        ):
                            valid_observation = False
                        cursor += 2 + board_count
                    if (
                        itos[tokens[cursor]] != "POT_SIZE_BB"
                        or not itos[tokens[cursor + 1]].startswith("RANGE_")
                        or itos[tokens[cursor + 2]] != "TO_CALL_BB"
                        or not itos[tokens[cursor + 3]].startswith("RANGE_")
                        or not itos[tokens[cursor + 4]].startswith("PLAYER_")
                    ):
                        valid_observation = False
                except (IndexError, ValueError):
                    valid_observation = False
                if not valid_observation:
                    invalid_decision_observations += 1
            if not decision_positions:
                report["errors"].append(
                    f"{split}: trajectory {number} has no decision marker"
                )
            elif any(masks[index] for index in range(start, decision_positions[0])):
                report["errors"].append(
                    f"{split}: trajectory {number} has supervision before its first decision"
                )
            for decision_number, decision_position in enumerate(decision_positions):
                next_position = (
                    decision_positions[decision_number + 1]
                    if decision_number + 1 < decision_count
                    else end
                )
                targets = [
                    index for index in range(decision_position, next_position) if masks[index]
                ]
                if len(targets) != 1:
                    report["errors"].append(
                        f"{split}: trajectory {number} decision {decision_number} has "
                        f"{len(targets)} supervised targets; expected exactly one"
                    )
            if decision_count != len(example_masked):
                report["errors"].append(
                    f"{split}: trajectory {number} has {decision_count} decision markers "
                    f"but {len(example_masked)} supervised targets"
                )
            masked += len(example_masked)
        if missing_position_prefixes:
            report["errors"].append(
                f"{split}: {missing_position_prefixes} trajectories lack one "
                "hero-position prefix token"
            )
        if invalid_decision_observations:
            report["errors"].append(
                f"{split}: {invalid_decision_observations} decision observations "
                "have invalid cards, board, pot, call, or state framing"
            )
        report["splits"][split] = {
            "trajectories": len(offsets),
            "tokens": len(tokens),
            "supervised_tokens": masked,
            "min_sequence": min(lengths, default=0),
            "max_sequence": max(lengths, default=0),
        }

    stats_path = output_dir / "stats.json"
    if not stats_path.exists():
        report["errors"].append("stats.json is missing")
    else:
        recorded_frequency = json.loads(stats_path.read_text(encoding="utf-8")).get(
            "token_frequency"
        )
        expected_frequency = {
            token: {
                "total": token_counts[token],
                "context": context_token_counts[token],
                "supervised": supervised_token_counts[token],
            }
            for token in itos
        }
        if recorded_frequency != expected_frequency:
            report["errors"].append(
                "stats.json token frequencies do not match the binary artifacts"
            )

    if selection_path is not None:
        groups: dict[str, set[str]] = {"train": set(), "val": set(), "test": set()}
        for row in read_jsonl(selection_path):
            if row.get("split") in groups:
                groups[row["split"]].add(row.get("split_group", row["member"]))
        overlaps: dict[str, list[str]] = {}
        for left, right in (("train", "val"), ("train", "test"), ("val", "test")):
            key = f"{left}-{right}"
            overlaps[key] = sorted(groups[left] & groups[right])
            if overlaps[key]:
                report["errors"].append(
                    f"{key} split group leakage: {overlaps[key][:5]}"
                )
        report["split_groups"] = {key: len(value) for key, value in groups.items()}
        report["split_group_overlap"] = {
            key: len(value) for key, value in overlaps.items()
        }
    report["valid"] = not report["errors"]
    return report


def write_validation_report(
    output_dir: Path, selection_path: Path | None = None
) -> dict[str, Any]:
    report = validate_artifacts(output_dir, selection_path)
    (Path(output_dir) / "validation_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return report
