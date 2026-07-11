from __future__ import annotations

import ast
import json
import os
import re
import zipfile
from collections import Counter
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Iterator


FIELD_PATTERNS = {
    "variant": re.compile(r"(?m)^\s*variant\s*=\s*['\"]([^'\"]+)['\"]"),
    "antes": re.compile(r"(?m)^\s*antes\s*=\s*(\[[^\n\r]*\])"),
    "blinds_or_straddles": re.compile(
        r"(?m)^\s*blinds_or_straddles\s*=\s*(\[[^\n\r]*\])"
    ),
    "starting_stacks": re.compile(r"(?m)^\s*starting_stacks\s*=\s*(\[[^\n\r]*\])"),
    "players": re.compile(r"(?m)^\s*players\s*=\s*(\[[^\n\r]*\])"),
    "min_bet": re.compile(r"(?m)^\s*min_bet\s*=\s*([0-9.eE+-]+)"),
    "small_bet": re.compile(r"(?m)^\s*small_bet\s*=\s*([0-9.eE+-]+)"),
}
SECTION_PATTERN = re.compile(r"(?m)^\s*\[([^\]]+)\]\s*$")

VARIANT_INFO = {
    "NT": ("no_limit", "texas_holdem"),
    "FT": ("fixed_limit", "texas_holdem"),
    "NS": ("no_limit", "short_deck_holdem"),
    "PO": ("pot_limit", "omaha_holdem"),
    "FO": ("fixed_limit", "omaha_holdem"),
}


@dataclass(frozen=True)
class ManifestOptions:
    header_bytes: int = 8_192
    max_members: int | None = None
    member_prefixes: tuple[str, ...] = ()


def _literal_list(text: str, field: str) -> list[Any] | None:
    match = FIELD_PATTERNS[field].search(text)
    if not match:
        return None
    try:
        value = ast.literal_eval(match.group(1))
    except (SyntaxError, ValueError):
        return None
    return value if isinstance(value, list) else None


def _source_folder(member_name: str) -> str:
    parts = PurePosixPath(member_name).parts
    if parts and parts[0].lower() == "data":
        parts = parts[1:]
    return parts[0] if len(parts) > 1 else "root"


def inspect_member(
    archive: zipfile.ZipFile, info: zipfile.ZipInfo, header_bytes: int
) -> dict[str, Any]:
    extension = PurePosixPath(info.filename).suffix.lower().lstrip(".")
    row: dict[str, Any] = {
        "member": info.filename,
        "extension": extension,
        "source_folder": _source_folder(info.filename),
        "compressed_size": info.compress_size,
        "uncompressed_size": info.file_size,
        "crc32": f"{info.CRC:08x}",
        "compression_ratio": round(info.file_size / max(info.compress_size, 1), 3),
        "variant": None,
        "betting_structure": None,
        "game_type": None,
        "player_count": None,
        "hand_count_estimate": None,
        "hand_count_exact": False,
        "header_truncated": info.file_size > header_bytes,
        "parse_error": None,
    }
    if info.file_size == 0:
        row["parse_error"] = "empty_member"
        return row
    try:
        with archive.open(info, "r") as member:
            payload = member.read(header_bytes)
        text = payload.decode("utf-8", errors="replace")
    except (OSError, RuntimeError, zipfile.BadZipFile) as exc:
        row["parse_error"] = f"header_read_error:{type(exc).__name__}:{exc}"
        return row

    variant_match = FIELD_PATTERNS["variant"].search(text)
    variant = variant_match.group(1).upper() if variant_match else None
    row["variant"] = variant
    betting, game = VARIANT_INFO.get(variant, (None, None))
    row["betting_structure"] = betting
    row["game_type"] = game

    stacks = _literal_list(text, "starting_stacks")
    players = _literal_list(text, "players")
    blinds = _literal_list(text, "blinds_or_straddles")
    row["player_count"] = len(stacks or players or blinds or []) or None

    if extension == "phh":
        row["hand_count_estimate"] = 1
        row["hand_count_exact"] = True
    elif extension == "phhs":
        sections = SECTION_PATTERN.findall(text)
        observed = len(sections)
        if info.file_size <= len(payload):
            row["hand_count_estimate"] = observed
            row["hand_count_exact"] = True
        elif observed:
            estimated = round(observed * info.file_size / max(len(payload), 1))
            row["hand_count_estimate"] = max(observed, estimated)
        else:
            row["parse_error"] = "no_phhs_section_in_header"

    if not variant and not row["parse_error"]:
        row["parse_error"] = "variant_not_found_in_header"
    return row


def iter_manifest_rows(
    zip_path: Path,
    options: ManifestOptions = ManifestOptions(),
    skip_members: set[str] | None = None,
) -> Iterator[dict[str, Any]]:
    skip_members = skip_members or set()
    emitted = 0
    with zipfile.ZipFile(zip_path) as archive:
        for info in archive.infolist():
            if info.is_dir() or not info.filename.lower().endswith((".phh", ".phhs")):
                continue
            if options.member_prefixes and not info.filename.startswith(options.member_prefixes):
                continue
            if info.filename in skip_members:
                continue
            yield inspect_member(archive, info, options.header_bytes)
            emitted += 1
            if options.max_members is not None and emitted >= options.max_members:
                break


def _clean_resumable_output(output_path: Path) -> set[str]:
    """Remove partial/duplicate rows left by an interrupted append."""
    existing: set[str] = set()
    cleaned = output_path.with_suffix(output_path.suffix + ".clean.tmp")
    with output_path.open("r", encoding="utf-8", errors="replace") as source, cleaned.open(
        "w", encoding="utf-8", newline="\n"
    ) as target:
        for line in source:
            try:
                row = json.loads(line)
                member = row["member"]
            except (json.JSONDecodeError, KeyError, TypeError):
                continue
            if member in existing:
                continue
            existing.add(member)
            target.write(json.dumps(row, sort_keys=True, ensure_ascii=True) + "\n")
        target.flush()
        os.fsync(target.fileno())
    cleaned.replace(output_path)
    return existing


def _build_manifest_unlocked(
    zip_path: Path,
    output_path: Path,
    options: ManifestOptions = ManifestOptions(),
    resume: bool = False,
) -> dict[str, Any]:
    zip_path = Path(zip_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    existing: set[str] = set()
    mode = "w"
    if resume and output_path.exists():
        mode = "a"
        existing = _clean_resumable_output(output_path)

    counts: Counter[str] = Counter()
    added = 0
    with output_path.open(mode, encoding="utf-8", newline="\n") as output:
        for row in iter_manifest_rows(zip_path, options, existing):
            output.write(json.dumps(row, sort_keys=True, ensure_ascii=True) + "\n")
            added += 1
            counts[f"variant:{row['variant'] or 'unknown'}"] += 1
            counts[f"source:{row['source_folder']}"] += 1
            if row["parse_error"]:
                counts["parse_errors"] += 1
            if added % 1000 == 0:
                output.flush()

    summary = {
        "archive": str(zip_path.resolve()),
        "output": str(output_path.resolve()),
        "rows_preexisting": len(existing),
        "rows_added": added,
        "header_bytes": options.header_bytes,
        "member_prefixes": options.member_prefixes,
        "counts_for_added_rows": dict(sorted(counts.items())),
    }
    summary_path = output_path.with_suffix(output_path.suffix + ".summary.json")
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return summary


def build_manifest(
    zip_path: Path,
    output_path: Path,
    options: ManifestOptions = ManifestOptions(),
    resume: bool = False,
) -> dict[str, Any]:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    lock_path = output_path.with_suffix(output_path.suffix + ".lock")
    try:
        lock_fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError as exc:
        raise RuntimeError(
            f"Manifest is already being written (or a stale lock exists): {lock_path}"
        ) from exc
    try:
        os.write(lock_fd, str(os.getpid()).encode("ascii"))
        return _build_manifest_unlocked(zip_path, output_path, options, resume)
    finally:
        os.close(lock_fd)
        lock_path.unlink(missing_ok=True)
