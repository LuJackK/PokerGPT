"""Access controls for the untouched held-out PokerGPT test split."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping


FINAL_TEST_CONFIRMATION = "SCORE_UNTOUCHED_TEST_ONCE"


@dataclass(frozen=True)
class TestSplitPermit:
    """Capability issued only after a write-once access receipt is created."""

    split: str
    freeze_manifest_sha256: str
    receipt_path: Path


def require_split_access(split: str, permit: TestSplitPermit | None) -> None:
    if split != "test":
        return
    if permit is None or permit.split != "test":
        raise PermissionError(
            "the held-out test split is sealed; use the final-test evaluator command"
        )
    if not permit.receipt_path.is_file():
        raise PermissionError("the held-out test access receipt is missing")


def create_test_access_receipt(
    path: str | Path,
    *,
    freeze_manifest_sha256: str,
    confirmation: str,
    checkpoint_sha256: str,
) -> TestSplitPermit:
    """Irreversibly consume the one-time test access before examples are read."""

    if confirmation != FINAL_TEST_CONFIRMATION:
        raise PermissionError(
            f"final test access requires --confirmation {FINAL_TEST_CONFIRMATION}"
        )
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    payload: Mapping[str, Any] = {
        "format_version": 1,
        "status": "test_access_started",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "freeze_manifest_sha256": freeze_manifest_sha256,
        "checkpoint_sha256": checkpoint_sha256,
        "warning": (
            "This receipt is created before held-out examples are read. Its presence "
            "permanently prevents a normal second evaluation attempt."
        ),
    }
    descriptor = os.open(target, os.O_WRONLY | os.O_CREAT | os.O_EXCL)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
    except BaseException:
        target.unlink(missing_ok=True)
        raise
    return TestSplitPermit(
        split="test",
        freeze_manifest_sha256=freeze_manifest_sha256,
        receipt_path=target,
    )
