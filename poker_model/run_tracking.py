"""Immutable identity records for PokerGPT training and evaluation candidates."""

from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from .checkpoint import load_checkpoint


RUN_IDENTITY_FORMAT_VERSION = 1


def canonical_json_sha256(value: Any) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def file_identity(path: str | Path, chunk_size: int = 1024 * 1024) -> dict[str, Any]:
    target = Path(path)
    if not target.is_file():
        raise FileNotFoundError(target)
    digest = hashlib.sha256()
    with target.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return {
        "path": target.as_posix(),
        "bytes": target.stat().st_size,
        "sha256": digest.hexdigest(),
    }


def build_run_identity(
    run_dir: str | Path,
    *,
    checkpoint_path: str | Path,
    artifact_bundle_path: str | Path,
) -> dict[str, Any]:
    """Verify and identify one exact checkpoint candidate without reading test data."""

    run_dir = Path(run_dir)
    config_path = run_dir / "config.json"
    environment_path = run_dir / "environment.json"
    fingerprint_path = run_dir / "dataset_fingerprint.json"
    metrics_path = run_dir / "metrics.jsonl"
    for path in (config_path, environment_path, fingerprint_path, metrics_path):
        if not path.is_file():
            raise FileNotFoundError(f"run identity input is missing: {path}")

    config = json.loads(config_path.read_text(encoding="utf-8"))
    environment = json.loads(environment_path.read_text(encoding="utf-8"))
    dataset = json.loads(fingerprint_path.read_text(encoding="utf-8"))
    checkpoint_path = Path(checkpoint_path)
    artifact_bundle_path = Path(artifact_bundle_path)
    checkpoint = load_checkpoint(
        checkpoint_path,
        expected_dataset_fingerprint=dataset,
        expected_training_config=config,
        map_location="cpu",
    )
    seed = int(config["training"]["seed"])
    return {
        "format_version": RUN_IDENTITY_FORMAT_VERSION,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "run_id": str(config["run_id"]),
        "seed": seed,
        "optimizer_step": int(checkpoint["optimizer_step"]),
        "best_validation_loss": float(checkpoint["best_validation_loss"]),
        "configuration": {
            **file_identity(config_path),
            "canonical_sha256": canonical_json_sha256(config),
        },
        "dataset": {
            "fingerprint": dataset["digest"],
            "fingerprint_record": file_identity(fingerprint_path),
            "artifact_bundle": file_identity(artifact_bundle_path),
        },
        "checkpoint": file_identity(checkpoint_path),
        "metrics": file_identity(metrics_path),
        "training_environment": environment,
        "checkpoint_environment_matches_record": checkpoint["environment"] == environment,
        "source_control": {
            "training_git_commit": environment.get("git_commit"),
            "training_git_dirty": environment.get("git_dirty"),
            "note": (
                None
                if environment.get("git_commit")
                else "training environment did not capture a Git commit"
            ),
        },
    }


def write_run_identity(path: str | Path, identity: Mapping[str, Any]) -> Path:
    """Create an identity record once; never replace an existing record."""

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    descriptor = os.open(target, flags)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            json.dump(dict(identity), handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
    except BaseException:
        target.unlink(missing_ok=True)
        raise
    return target
