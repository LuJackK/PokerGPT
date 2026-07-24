from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator

from .io_utils import read_jsonl, stable_fraction, write_jsonl_atomic


@dataclass(frozen=True)
class SelectionOptions:
    variants: tuple[str, ...] = ("NT",)
    player_counts: tuple[int, ...] = (6,)
    included_sources: tuple[str, ...] = ("pluribus",)
    excluded_sources: tuple[str, ...] = ("annual-computer-poker-competition",)
    max_member_bytes: int = 64 * 1024 * 1024
    validation_fraction: float = 0.1
    test_fraction: float = 0.05
    split_seed: str = "pokergpt-v081-split"


def rejection_reason(row: dict[str, Any], options: SelectionOptions) -> str | None:
    if row.get("parse_error"):
        return "manifest_parse_error"
    if row.get("variant") not in options.variants:
        return "variant"
    if row.get("betting_structure") != "no_limit":
        return "betting_structure"
    if row.get("game_type") != "texas_holdem":
        return "game_type"
    if row.get("player_count") not in options.player_counts:
        return "player_count"
    if options.included_sources and row.get("source_folder") not in options.included_sources:
        return "source_not_included"
    if row.get("source_folder") in options.excluded_sources:
        return "excluded_source"
    if int(row.get("uncompressed_size") or 0) > options.max_member_bytes:
        return "member_too_large"
    return None


def iter_selected(
    rows: Iterable[dict[str, Any]], options: SelectionOptions
) -> Iterator[dict[str, Any]]:
    if not 0 <= options.validation_fraction < 1:
        raise ValueError("validation_fraction must be in [0, 1)")
    if not 0 <= options.test_fraction < 1:
        raise ValueError("test_fraction must be in [0, 1)")
    if options.validation_fraction + options.test_fraction >= 1:
        raise ValueError("validation_fraction and test_fraction must sum to less than 1")
    accepted = [row for row in rows if rejection_reason(row, options) is None]
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in accepted:
        parts = row["member"].split("/")
        split_group = (
            "/".join(parts[:-1])
            if row.get("source_folder") == "pluribus" and len(parts) >= 4
            else row["member"]
        )
        grouped.setdefault(split_group, []).append(row)

    test_groups = _closest_group_subset(
        grouped,
        len(accepted) * options.test_fraction,
        f"{options.split_seed}:test",
    )
    validation_candidates = {
        group: rows for group, rows in grouped.items() if group not in test_groups
    }
    validation_groups = _closest_group_subset(
        validation_candidates,
        len(accepted) * options.validation_fraction,
        f"{options.split_seed}:val",
    )

    for split_group, group_rows in grouped.items():
        if split_group in test_groups:
            split = "test"
        elif split_group in validation_groups:
            split = "val"
        else:
            split = "train"
        for row in group_rows:
            selected = dict(row)
            selected["split"] = split
            selected["split_group"] = split_group
            selected["selected_player_counts"] = list(options.player_counts)
            selected["selection"] = "clean_nt_6max_v2"
            yield selected


def _closest_group_subset(
    grouped: dict[str, list[dict[str, Any]]],
    target_rows: float,
    seed: str,
) -> set[str]:
    """Choose a deterministic indivisible-group subset nearest a row target."""

    if target_rows <= 0 or not grouped:
        return set()
    ordered = sorted(grouped, key=lambda group: stable_fraction(group, seed))
    paths: dict[int, tuple[str, ...]] = {0: ()}
    for group in ordered:
        size = len(grouped[group])
        additions: dict[int, tuple[str, ...]] = {}
        for total, selected in tuple(paths.items()):
            candidate_total = total + size
            if candidate_total not in paths and candidate_total not in additions:
                additions[candidate_total] = (*selected, group)
        paths.update(additions)
    best_total = min(paths, key=lambda total: (abs(total - target_rows), total))
    return set(paths[best_total])


def select_dataset(
    manifest_path: Path, output_path: Path, options: SelectionOptions = SelectionOptions()
) -> dict[str, Any]:
    rows = list(read_jsonl(manifest_path))
    rejected = Counter()
    for row in rows:
        reason = rejection_reason(row, options)
        if reason:
            rejected[reason] += 1
    selected = list(iter_selected(rows, options))
    write_jsonl_atomic(output_path, selected)
    split_counts = Counter(row["split"] for row in selected)
    source_counts = Counter(row["source_folder"] for row in selected)
    summary = {
        "manifest": str(Path(manifest_path).resolve()),
        "output": str(Path(output_path).resolve()),
        "input_rows": len(rows),
        "selected_rows": len(selected),
        "rejected": dict(sorted(rejected.items())),
        "splits": dict(sorted(split_counts.items())),
        "sources": dict(sorted(source_counts.items())),
        "options": {
            "variants": options.variants,
            "player_counts": options.player_counts,
            "included_sources": options.included_sources,
            "excluded_sources": options.excluded_sources,
            "max_member_bytes": options.max_member_bytes,
            "validation_fraction": options.validation_fraction,
            "test_fraction": options.test_fraction,
            "split_seed": options.split_seed,
        },
    }
    summary_path = Path(output_path).with_suffix(Path(output_path).suffix + ".summary.json")
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return summary
