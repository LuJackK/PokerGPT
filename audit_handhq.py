from __future__ import annotations

import argparse
import json
from pathlib import Path

from poker_pipeline.handhq_audit import audit_handhq_archive


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Stream a full per-hand HandHQ eligibility and selection-bias audit."
    )
    parser.add_argument("archive", type=Path)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/processed/handhq_audit.json"),
        help="Aggregate JSON report (a Markdown report is written beside it)",
    )
    parser.add_argument("--max-members", type=int, default=None)
    parser.add_argument("--max-member-mib", type=int, default=64)
    parser.add_argument("--progress-every", type=int, default=250)
    args = parser.parse_args()
    report = audit_handhq_archive(
        args.archive,
        args.output,
        max_members=args.max_members,
        max_member_bytes=args.max_member_mib * 1024 * 1024,
        progress_every=args.progress_every,
    )
    print(json.dumps(report["eligibility"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
