from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Iterable, Iterator


def write_jsonl_atomic(path: Path, rows: Iterable[dict[str, Any]]) -> int:
    """Write JSON Lines atomically and return the number of rows."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    count = 0
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True, ensure_ascii=True) + "\n")
            count += 1
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)
    return count


def read_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on {path}:{line_number}: {exc}") from exc


def stable_fraction(value: str, seed: str = "pokergpt-v1") -> float:
    digest = hashlib.sha256(f"{seed}\0{value}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") / 2**64

