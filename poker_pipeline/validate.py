from __future__ import annotations

import json
import pickle
import struct
from array import array
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any

from .io_utils import read_jsonl
from .tokenizer import DECISION_TOKENS, RANGE_THRESHOLDS, RANGE_TOKENS


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
    if metadata_decision_tokens != expected_decision_tokens:
        report["errors"].append("meta.pkl decision-token set does not match the pipeline")
    if "RANGE_ZERO" in metadata_decision_tokens:
        report["errors"].append("meta.pkl permits RANGE_ZERO as a hero decision")
    if tuple(meta.get("range_tokens", ())) != RANGE_TOKENS:
        report["errors"].append("meta.pkl range-token set does not match the pipeline")
    representatives = meta.get("range_representative_ratio", {})
    if set(representatives) != set(RANGE_TOKENS):
        report["errors"].append(
            "meta.pkl does not define exactly one representative for every range"
        )
    else:
        lower = Decimal(0)
        for index, token in enumerate(RANGE_TOKENS):
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
            if upper is not None:
                lower = upper

    for split in ("train", "val"):
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
        if any(mask not in (0, 1) for mask in masks):
            report["errors"].append(f"{split}: loss mask contains value other than 0/1")
        if offsets != sorted(set(offsets)):
            report["errors"].append(f"{split}: indexes are not unique and increasing")
        lengths: list[int] = []
        masked = 0
        for number, start in enumerate(offsets):
            end = offsets[number + 1] if number + 1 < len(offsets) else len(tokens)
            lengths.append(end - start)
            if end <= start or tokens[start] != bos_id or tokens[end - 1] != eos_id:
                report["errors"].append(f"{split}: invalid framing at example {number}")
                continue
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
            for index in range(start, end):
                if tokens[index] != decision_id:
                    continue
                local = [itos[tokens[position]] for position in range(index, min(end, index + 12))]
                if len(local) < 7:
                    report["errors"].append(
                        f"{split}: truncated decision observation at token {index}"
                    )
                    continue
                if local[1] != "PLAYER_1_HOLE_CARDS" or not all(
                    token.startswith("CARD_") for token in local[2:4]
                ):
                    report["errors"].append(
                        f"{split}: decision at token {index} lacks local hero cards"
                    )
                if local[4] != "CURRENT_BOARD" or not local[5].startswith("COUNT_"):
                    report["errors"].append(
                        f"{split}: decision at token {index} lacks local current board"
                    )
                    continue
                try:
                    board_count = int(local[5].removeprefix("COUNT_"))
                except ValueError:
                    report["errors"].append(
                        f"{split}: invalid current-board count at token {index + 5}"
                    )
                    continue
                board_end = index + 6 + board_count
                if board_count not in {0, 3, 4, 5} or board_end >= end:
                    report["errors"].append(
                        f"{split}: invalid current-board framing at token {index}"
                    )
                    continue
                if not all(
                    itos[tokens[position]].startswith("CARD_")
                    for position in range(index + 6, board_end)
                ) or itos[tokens[board_end]] != "POT_SIZE_BB":
                    report["errors"].append(
                        f"{split}: invalid current-board cards at token {index}"
                    )
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
        report["splits"][split] = {
            "trajectories": len(offsets),
            "tokens": len(tokens),
            "supervised_tokens": masked,
            "min_sequence": min(lengths, default=0),
            "max_sequence": max(lengths, default=0),
        }

    if selection_path is not None:
        groups: dict[str, set[str]] = {"train": set(), "val": set()}
        for row in read_jsonl(selection_path):
            if row.get("split") in groups:
                groups[row["split"]].add(row.get("split_group", row["member"]))
        overlap = sorted(groups["train"] & groups["val"])
        if overlap:
            report["errors"].append(f"split group leakage: {overlap[:5]}")
        report["split_groups"] = {key: len(value) for key, value in groups.items()}
        report["split_group_overlap"] = len(overlap)
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
