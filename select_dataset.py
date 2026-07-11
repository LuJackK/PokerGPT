from __future__ import annotations

import argparse
import json
from pathlib import Path

from poker_pipeline.selection import SelectionOptions, select_dataset


def main() -> None:
    parser = argparse.ArgumentParser(description="Select a clean NT no-limit Hold'em subset.")
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--players", type=int, nargs="+", default=[6])
    parser.add_argument("--include-source", action="append", default=None)
    parser.add_argument("--exclude-source", action="append", default=["annual-computer-poker-competition"])
    parser.add_argument("--max-member-mib", type=float, default=64.0)
    parser.add_argument("--val-fraction", type=float, default=0.1)
    parser.add_argument("--seed", default="pokergpt-v1")
    args = parser.parse_args()
    if not 0 <= args.val_fraction < 1:
        parser.error("--val-fraction must be in [0, 1)")
    options = SelectionOptions(
        player_counts=tuple(args.players),
        included_sources=tuple(args.include_source or ["pluribus"]),
        excluded_sources=tuple(dict.fromkeys(args.exclude_source)),
        max_member_bytes=int(args.max_member_mib * 1024 * 1024),
        validation_fraction=args.val_fraction,
        split_seed=args.seed,
    )
    print(json.dumps(select_dataset(args.manifest, args.output, options), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
