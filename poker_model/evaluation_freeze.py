"""Freeze and verify the evaluator before one-time held-out scoring."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from .evaluator import load_evaluator_config, write_json
from .run_tracking import canonical_json_sha256, file_identity


def build_freeze_manifest(
    *,
    repo_root: str | Path,
    config_path: str | Path,
    validation_report_path: str | Path,
    checkpoint_path: str | Path,
    artifact_bundle_path: str | Path,
    run_identity_path: str | Path,
) -> dict[str, Any]:
    repo_root = Path(repo_root).resolve()
    config_path = Path(config_path)
    config = load_evaluator_config(config_path)
    validation_report = json.loads(
        Path(validation_report_path).read_text(encoding="utf-8")
    )
    if validation_report.get("split") != "val" or not validation_report.get(
        "complete_split"
    ):
        raise ValueError("freeze requires a complete validation-split report")
    config_hash = canonical_json_sha256(config)
    if validation_report.get("evaluator_configuration_sha256") != config_hash:
        raise ValueError("validation report does not match evaluator configuration")
    checkpoint = file_identity(checkpoint_path)
    artifact = file_identity(artifact_bundle_path)
    if checkpoint["sha256"] != config["candidate"]["checkpoint_sha256"]:
        raise ValueError("checkpoint does not match evaluator candidate")
    if artifact["sha256"] != config["candidate"]["artifact_bundle_sha256"]:
        raise ValueError("artifact bundle does not match evaluator candidate")
    frozen_files: dict[str, Any] = {}
    for relative_name in config["freeze_files"]:
        relative = Path(relative_name)
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError(f"freeze file must be repository-relative: {relative_name}")
        frozen_files[relative.as_posix()] = file_identity(repo_root / relative)
    return {
        "format_version": 1,
        "status": "frozen_before_test",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "evaluator_id": config["evaluator_id"],
        "evaluator_configuration": {
            **file_identity(config_path),
            "canonical_sha256": config_hash,
        },
        "candidate": dict(config["candidate"]),
        "checkpoint": checkpoint,
        "artifact_bundle": artifact,
        "run_identity": file_identity(run_identity_path),
        "validation_report": file_identity(validation_report_path),
        "frozen_files": frozen_files,
        "test_accessed": False,
    }


def write_freeze_manifest(path: str | Path, manifest: Mapping[str, Any]) -> Path:
    return write_json(path, manifest, exclusive=True)


def verify_freeze_manifest(
    *,
    manifest_path: str | Path,
    repo_root: str | Path,
    config_path: str | Path,
    checkpoint_path: str | Path,
    artifact_bundle_path: str | Path,
    run_identity_path: str | Path,
) -> dict[str, Any]:
    manifest_path = Path(manifest_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("status") != "frozen_before_test":
        raise ValueError("evaluator freeze manifest has an invalid status")
    config = load_evaluator_config(config_path)
    if canonical_json_sha256(config) != manifest["evaluator_configuration"]["canonical_sha256"]:
        raise ValueError("evaluator configuration changed after freeze")
    checks = (
        ("checkpoint", checkpoint_path),
        ("artifact_bundle", artifact_bundle_path),
        ("run_identity", run_identity_path),
    )
    for label, path in checks:
        actual = file_identity(path)
        expected = manifest[label]
        if actual["bytes"] != expected["bytes"] or actual["sha256"] != expected["sha256"]:
            raise ValueError(f"{label} changed after evaluator freeze")
    repo_root = Path(repo_root).resolve()
    for relative_name, expected in manifest["frozen_files"].items():
        actual = file_identity(repo_root / relative_name)
        if actual["bytes"] != expected["bytes"] or actual["sha256"] != expected["sha256"]:
            raise ValueError(f"frozen evaluator file changed: {relative_name}")
    return manifest
