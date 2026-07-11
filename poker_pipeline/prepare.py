from __future__ import annotations

import hashlib
import json
import os
import pickle
import struct
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from . import __version__
from .io_utils import read_jsonl
from .phh import Decision, iter_archive_decisions
from .tokenizer import EncodedDecision, PokerTokenizer


@dataclass(frozen=True)
class PrepareOptions:
    block_size: int = 256
    history_limit: int = 32
    max_member_bytes: int = 64 * 1024 * 1024
    max_members: int | None = None
    audit_samples: int = 20


class SplitWriter:
    def __init__(self, output_dir: Path, split: str) -> None:
        self.split = split
        self.paths = {
            "tokens": output_dir / f"{split}.bin",
            "mask": output_dir / f"{split}_loss_mask.bin",
            "index": output_dir / f"{split}.idx",
        }
        self.temporary = {key: path.with_suffix(path.suffix + ".tmp") for key, path in self.paths.items()}
        self.handles = {key: path.open("wb") for key, path in self.temporary.items()}
        self.token_count = 0
        self.example_count = 0

    def write(self, encoded: EncodedDecision) -> None:
        self.handles["index"].write(struct.pack("<Q", self.token_count))
        self.handles["tokens"].write(struct.pack(f"<{len(encoded.ids)}H", *encoded.ids))
        self.handles["mask"].write(bytes(encoded.loss_mask))
        self.token_count += len(encoded.ids)
        self.example_count += 1

    def close(self, commit: bool = True) -> None:
        for handle in self.handles.values():
            handle.flush()
            os.fsync(handle.fileno())
            handle.close()
        if commit:
            for key, target in self.paths.items():
                self.temporary[key].replace(target)
        else:
            for path in self.temporary.values():
                path.unlink(missing_ok=True)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _encode_with_limit(
    tokenizer: PokerTokenizer, decision: Decision, options: PrepareOptions
) -> EncodedDecision:
    history_limit = min(options.history_limit, len(decision.history))
    while history_limit >= 0:
        encoded = tokenizer.encode_decision(decision, history_limit)
        if len(encoded.ids) <= options.block_size:
            return encoded
        history_limit -= 1
    raise ValueError(
        f"Decision requires more than block_size={options.block_size} tokens without history"
    )


def prepare_dataset(
    zip_path: Path,
    selection_path: Path,
    output_dir: Path,
    options: PrepareOptions = PrepareOptions(),
) -> dict[str, Any]:
    zip_path = Path(zip_path)
    selection_path = Path(selection_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    selection = list(read_jsonl(selection_path))
    if options.max_members is not None:
        selection = selection[: options.max_members]
    if not selection:
        raise ValueError("Selection is empty")
    tokenizer = PokerTokenizer()
    writers = {split: SplitWriter(output_dir, split) for split in ("train", "val")}
    counts: Counter[str] = Counter()
    sequence_lengths: list[int] = []
    errors: list[dict[str, str]] = []
    audit: list[dict[str, Any]] = []
    committed = False
    try:
        for selected, decision, error in iter_archive_decisions(
            zip_path, selection, options.max_member_bytes
        ):
            if error:
                errors.append({"member": selected["member"], "error": error})
                counts["parse_errors"] += 1
                continue
            assert decision is not None
            allowed_counts = selected.get("selected_player_counts")
            if allowed_counts and decision.player_count not in allowed_counts:
                counts["player_count_skipped"] += 1
                continue
            if len(decision.hero_cards) != 2 or any(card == "??" for card in decision.hero_cards):
                counts["unknown_hero_cards_skipped"] += 1
                continue
            split = selected.get("split")
            if split not in writers:
                errors.append({"member": selected["member"], "error": f"invalid_split:{split}"})
                counts["invalid_split"] += 1
                continue
            try:
                encoded = _encode_with_limit(tokenizer, decision, options)
            except ValueError as exc:
                errors.append({"member": selected["member"], "error": str(exc)})
                counts["oversized_examples"] += 1
                continue
            writers[split].write(encoded)
            sequence_lengths.append(len(encoded.ids))
            counts[f"split:{split}:examples"] += 1
            counts[f"action:{decision.target_action}"] += 1
            counts[f"street:{decision.street}"] += 1
            counts[f"source:{selected.get('source_folder', 'unknown')}"] += 1
            if len(audit) < options.audit_samples:
                audit.append(
                    {
                        "member": selected["member"],
                        "hand_key": decision.hand_key,
                        "actor": decision.actor + 1,
                        "split": split,
                        "tokens": encoded.tokens,
                        "loss_positions": [i for i, bit in enumerate(encoded.loss_mask) if bit],
                    }
                )
        for writer in writers.values():
            writer.close(commit=True)
        committed = True
    finally:
        if not committed:
            for writer in writers.values():
                try:
                    writer.close(commit=False)
                except ValueError:
                    pass

    error_path = output_dir / "parse_errors.jsonl"
    error_path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in errors), encoding="utf-8"
    )
    audit_path = output_dir / "audit_samples.jsonl"
    audit_path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in audit), encoding="utf-8"
    )
    meta = {
        "version": __version__,
        "vocab_size": len(tokenizer.itos),
        "stoi": tokenizer.stoi,
        "itos": tokenizer.itos,
        "token_dtype": "uint16_le",
        "loss_mask_dtype": "uint8",
        "index_dtype": "uint64_le_token_offset",
        "block_size": options.block_size,
        "normalization": {
            "chip_arithmetic": "decimal_exact",
            "amount_features": ["incremental_chips_over_big_blind", "incremental_chips_over_pot_before_action"],
            "amount_buckets": "see poker_pipeline.tokenizer.RATIO_LABELS",
        },
        "privacy": "actor hole cards only; every opponent private card encoded CARD_UNKNOWN",
        "legality": "not encoded or stored; replay validates source actions only",
        "loss": "action token plus amount buckets for BET/RAISE",
    }
    with (output_dir / "meta.pkl").open("wb") as handle:
        pickle.dump(meta, handle)
    stats = {
        "selection_rows": len(selection),
        "counts": dict(sorted(counts.items())),
        "sequence_length": {
            "min": min(sequence_lengths, default=0),
            "max": max(sequence_lengths, default=0),
            "mean": sum(sequence_lengths) / max(len(sequence_lengths), 1),
        },
        "parse_error_count": len(errors),
        "writer_examples": {split: writer.example_count for split, writer in writers.items()},
        "writer_tokens": {split: writer.token_count for split, writer in writers.items()},
    }
    (output_dir / "stats.json").write_text(
        json.dumps(stats, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    manifest = {
        "pipeline_version": __version__,
        "archive": str(zip_path.resolve()),
        "archive_bytes": zip_path.stat().st_size,
        "selection": str(selection_path.resolve()),
        "selection_sha256": _sha256(selection_path),
        "options": asdict(options),
        "outputs": {
            path.name: {"bytes": path.stat().st_size, "sha256": _sha256(path)}
            for path in sorted(output_dir.iterdir())
            if path.is_file() and path.name != "preprocessing_manifest.json"
        },
    }
    (output_dir / "preprocessing_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return stats
