from __future__ import annotations

import argparse
import json
from pathlib import Path

from poker_pipeline.validate import write_validation_report


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate PokerGPT binary dataset artifacts.")
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--selection", type=Path)
    args = parser.parse_args()
    report = write_validation_report(args.output_dir, args.selection)
    print(json.dumps(report, indent=2, sort_keys=True))
    if not report["valid"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

