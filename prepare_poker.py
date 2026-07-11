from __future__ import annotations

import argparse
import json
from pathlib import Path

from poker_pipeline.prepare import PrepareOptions, prepare_dataset


def main() -> None:
    parser = argparse.ArgumentParser(description="Stream selected PHH/PHHS hands into PokerGPT binaries.")
    parser.add_argument("archive", type=Path)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--block-size", type=int, default=256)
    parser.add_argument("--history-limit", type=int, default=32)
    parser.add_argument("--max-member-mib", type=float, default=64.0)
    parser.add_argument("--max-members", type=int)
    parser.add_argument("--audit-samples", type=int, default=20)
    args = parser.parse_args()
    if args.block_size < 64:
        parser.error("--block-size must be at least 64")
    options = PrepareOptions(
        block_size=args.block_size,
        history_limit=args.history_limit,
        max_member_bytes=int(args.max_member_mib * 1024 * 1024),
        max_members=args.max_members,
        audit_samples=args.audit_samples,
    )
    print(
        json.dumps(
            prepare_dataset(args.archive, args.selection, args.output_dir, options),
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()

