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
from .phh import Decision, build_hero_trajectories, iter_archive_hand_decisions
from .tokenizer import EncodedTrajectory, PokerTokenizer


@dataclass(frozen=True)
class PrepareOptions:
    block_size: int = 256
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
        self.temporary = {
            key: path.with_suffix(path.suffix + ".tmp") for key, path in self.paths.items()
        }
        self.handles = {key: path.open("wb") for key, path in self.temporary.items()}
        self.token_count = 0
        self.trajectory_count = 0

    def write(self, encoded: EncodedTrajectory) -> None:
        self.handles["index"].write(struct.pack("<Q", self.token_count))
        self.handles["tokens"].write(struct.pack(f"<{len(encoded.ids)}H", *encoded.ids))
        self.handles["mask"].write(bytes(encoded.loss_mask))
        self.token_count += len(encoded.ids)
        self.trajectory_count += 1

    def close(self, commit: bool = True) -> None:
        for handle in self.handles.values():
            if handle.closed:
                continue
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


def _percentile(values: list[int], fraction: float) -> int:
    if not values:
        return 0
    ordered = sorted(values)
    return ordered[round((len(ordered) - 1) * fraction)]


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
    decisions_per_trajectory: list[int] = []
    errors: list[dict[str, str]] = []
    audit: list[dict[str, Any]] = []
    committed = False
    try:
        for selected, decisions, error in iter_archive_hand_decisions(
            zip_path, selection, options.max_member_bytes
        ):
            if error:
                errors.append({"member": selected["member"], "error": error})
                counts["parse_errors"] += 1
                continue
            assert decisions is not None and decisions
            counts["hands_processed"] += 1
            allowed_counts = selected.get("selected_player_counts")
            if allowed_counts and decisions[0].player_count not in allowed_counts:
                counts["player_count_skipped"] += 1
                continue
            split = selected.get("split")
            if split not in writers:
                errors.append({"member": selected["member"], "error": f"invalid_split:{split}"})
                counts["invalid_split"] += 1
                continue
            trajectories = build_hero_trajectories(decisions)
            acting_players = len({decision.actor for decision in decisions})
            counts["unknown_hero_trajectories_skipped"] += acting_players - len(trajectories)
            for trajectory in trajectories:
                encoded = tokenizer.encode_trajectory(trajectory)
                if len(encoded.ids) > options.block_size:
                    raise ValueError(
                        f"Complete trajectory is {len(encoded.ids)} tokens, exceeding "
                        f"block_size={options.block_size}: {trajectory.member}:"
                        f"{trajectory.hand_key}:p{trajectory.hero + 1}. Increase block size; "
                        "history truncation is disabled."
                    )
                writers[split].write(encoded)
                sequence_lengths.append(len(encoded.ids))
                decisions_per_trajectory.append(trajectory.decision_count)
                counts[f"split:{split}:trajectories"] += 1
                counts[f"source:{selected.get('source_folder', 'unknown')}:trajectories"] += 1
                for item in trajectory.items:
                    if not isinstance(item, Decision):
                        continue
                    counts[f"action:{item.target_action}"] += 1
                    counts[f"street:{item.street}"] += 1
                    counts["supervised_decisions"] += 1
                if len(audit) < options.audit_samples:
                    audit.append(
                        {
                            "member": selected["member"],
                            "hand_key": trajectory.hand_key,
                            "hero_source_seat": trajectory.hero + 1,
                            "split": split,
                            "decision_count": trajectory.decision_count,
                            "tokens": encoded.tokens,
                            "loss_positions": [
                                index for index, bit in enumerate(encoded.loss_mask) if bit
                            ],
                        }
                    )
        for writer in writers.values():
            writer.close(commit=True)
        committed = True
    finally:
        if not committed:
            for writer in writers.values():
                writer.close(commit=False)

    (output_dir / "parse_errors.jsonl").write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in errors), encoding="utf-8"
    )
    (output_dir / "audit_samples.jsonl").write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in audit), encoding="utf-8"
    )
    meta = {
        "version": __version__,
        "format": "complete_player_perspective_trajectories_v1",
        "vocab_size": len(tokenizer.itos),
        "stoi": tokenizer.stoi,
        "itos": tokenizer.itos,
        "token_dtype": "uint16_le",
        "loss_mask_dtype": "uint8",
        "index_dtype": "uint64_le_token_offset",
        "block_size": options.block_size,
        "pad_token_id": tokenizer.stoi["<PAD>"],
        "normalization": {
            "chip_arithmetic": "decimal_exact",
            "amount_features": [
                "incremental_chips_over_big_blind",
                "incremental_chips_over_pot_before_action",
            ],
            "amount_buckets": "shared non-overlapping RANGE_* tokens",
        },
        "privacy": "one complete trajectory per hero; opponent private cards omitted",
        "legality": "not encoded or stored; replay validates source actions only",
        "perspective": "hero is PLAYER_1; other seats numbered clockwise",
        "events": "forced posts, public actions, and board reveals are chronological",
        "observations": "public pot, call amount, status, and stack state precede each hero decision",
        "loss": "multiple hero action targets per trajectory; sizing ranges for BET/RAISE",
        "batching": "sample complete indexed trajectories; never random-crop across boundaries",
    }
    with (output_dir / "meta.pkl").open("wb") as handle:
        pickle.dump(meta, handle)
    stats = {
        "selection_rows": len(selection),
        "counts": dict(sorted(counts.items())),
        "trajectory_length": {
            "min": min(sequence_lengths, default=0),
            "median": _percentile(sequence_lengths, 0.5),
            "p95": _percentile(sequence_lengths, 0.95),
            "p99": _percentile(sequence_lengths, 0.99),
            "max": max(sequence_lengths, default=0),
            "mean": sum(sequence_lengths) / max(len(sequence_lengths), 1),
            "over_block_size": sum(length > options.block_size for length in sequence_lengths),
        },
        "decisions_per_trajectory": {
            "min": min(decisions_per_trajectory, default=0),
            "median": _percentile(decisions_per_trajectory, 0.5),
            "max": max(decisions_per_trajectory, default=0),
            "mean": sum(decisions_per_trajectory) / max(len(decisions_per_trajectory), 1),
        },
        "parse_error_count": len(errors),
        "writer_trajectories": {
            split: writer.trajectory_count for split, writer in writers.items()
        },
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
