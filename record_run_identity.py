from __future__ import annotations

import argparse
import json
from pathlib import Path

from poker_model.run_tracking import build_run_identity, write_run_identity


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Write a one-time identity record for an exact PokerGPT checkpoint"
    )
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--artifact-bundle", type=Path, required=True)
    parser.add_argument(
        "--output",
        type=Path,
        help="default: RUN_DIR/run_identity.json",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output = args.output or args.run_dir / "run_identity.json"
    identity = build_run_identity(
        args.run_dir,
        checkpoint_path=args.checkpoint,
        artifact_bundle_path=args.artifact_bundle,
    )
    write_run_identity(output, identity)
    print(json.dumps(identity, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
