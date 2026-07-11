from __future__ import annotations

import argparse
import json
from pathlib import Path

from poker_pipeline.manifest import ManifestOptions, build_manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a streaming PHH/PHHS ZIP manifest.")
    parser.add_argument("archive", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--header-bytes", type=int, default=8_192)
    parser.add_argument("--max-members", type=int)
    parser.add_argument("--member-prefix", action="append", default=[])
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.header_bytes < 1024:
        parser.error("--header-bytes must be at least 1024")
    summary = build_manifest(
        args.archive,
        args.output,
        ManifestOptions(args.header_bytes, args.max_members, tuple(args.member_prefix)),
        args.resume,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
