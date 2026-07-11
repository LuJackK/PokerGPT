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
    split_seed: str = "pokergpt-v1"


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

    target_validation = len(accepted) * options.validation_fraction
    desired_group_count = (
        max(1, round(len(grouped) * options.validation_fraction))
        if options.validation_fraction > 0 and grouped
        else 0
    )
    randomized_groups = sorted(
        grouped, key=lambda group: stable_fraction(group, options.split_seed)
    )
    validation_groups = set(randomized_groups[:desired_group_count])
    validation_rows = sum(len(grouped[group]) for group in validation_groups)
    # Preserve an approximately representative number of groups, then make
    # deterministic one-for-one swaps until no swap improves the row ratio.
    while validation_groups:
        current_error = abs(validation_rows - target_validation)
        best: tuple[float, str, str, int] | None = None
        for remove in sorted(validation_groups):
            for add in randomized_groups:
                if add in validation_groups:
                    continue
                candidate_rows = validation_rows - len(grouped[remove]) + len(grouped[add])
                candidate_error = abs(candidate_rows - target_validation)
                if candidate_error >= current_error:
                    continue
                candidate = (candidate_error, remove, add, candidate_rows)
                if best is None or candidate < best:
                    best = candidate
        if best is None:
            break
        _, remove, add, validation_rows = best
        validation_groups.remove(remove)
        validation_groups.add(add)

    for split_group, group_rows in grouped.items():
        for row in group_rows:
            selected = dict(row)
            selected["split"] = "val" if split_group in validation_groups else "train"
            selected["split_group"] = split_group
            selected["selected_player_counts"] = list(options.player_counts)
            selected["selection"] = "clean_nt_6max_v1"
            yield selected


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
            "split_seed": options.split_seed,
        },
    }
    summary_path = Path(output_path).with_suffix(Path(output_path).suffix + ".summary.json")
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return summary
