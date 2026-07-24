"""Atomic, compatibility-checked checkpoints for PokerGPT training."""

from __future__ import annotations

import hashlib
import os
import random
import tempfile
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch


CHECKPOINT_FORMAT_VERSION = 1
DEFAULT_SPLITS = ("train", "val", "test")


def _hash_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def dataset_fingerprint(
    output_dir: str | Path,
    splits: Sequence[str] = DEFAULT_SPLITS,
) -> dict[str, Any]:
    """Hash metadata and every aligned artifact for the requested splits."""

    output_dir = Path(output_dir)
    relative_paths = [Path("meta.pkl")]
    for split in splits:
        relative_paths.extend(
            (
                Path(f"{split}.bin"),
                Path(f"{split}_loss_mask.bin"),
                Path(f"{split}.idx"),
            )
        )

    files: dict[str, dict[str, Any]] = {}
    combined = hashlib.sha256()
    for relative_path in relative_paths:
        path = output_dir / relative_path
        if not path.is_file():
            raise FileNotFoundError(f"dataset fingerprint input is missing: {path}")
        file_digest = _hash_file(path)
        size = path.stat().st_size
        name = relative_path.as_posix()
        files[name] = {"bytes": size, "sha256": file_digest}
        combined.update(name.encode("utf-8"))
        combined.update(b"\0")
        combined.update(str(size).encode("ascii"))
        combined.update(b"\0")
        combined.update(file_digest.encode("ascii"))
        combined.update(b"\n")

    return {
        "format_version": 1,
        "algorithm": "sha256",
        "digest": combined.hexdigest(),
        "splits": list(splits),
        "files": files,
    }


def capture_rng_state() -> dict[str, Any]:
    """Capture Python, PyTorch CPU, and all available CUDA RNG streams."""

    return {
        "python": random.getstate(),
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
    }


def restore_rng_state(state: Mapping[str, Any]) -> None:
    missing = {"python", "torch_cpu", "torch_cuda"}.difference(state)
    if missing:
        raise ValueError(f"RNG state is missing keys: {sorted(missing)}")
    random.setstate(state["python"])
    torch.set_rng_state(state["torch_cpu"].cpu())
    cuda_states = state["torch_cuda"]
    if cuda_states:
        if not torch.cuda.is_available():
            raise RuntimeError("checkpoint contains CUDA RNG state but CUDA is unavailable")
        if len(cuda_states) != torch.cuda.device_count():
            raise RuntimeError(
                "checkpoint CUDA RNG device count does not match the current machine"
            )
        torch.cuda.set_rng_state_all([cuda_state.cpu() for cuda_state in cuda_states])


def save_checkpoint(path: str | Path, checkpoint: Mapping[str, Any]) -> None:
    """Write a checkpoint beside its destination and atomically replace it."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = dict(checkpoint)
    payload.setdefault("checkpoint_format_version", CHECKPOINT_FORMAT_VERSION)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb", prefix=f".{path.name}.", suffix=".tmp", dir=path.parent, delete=False
        ) as handle:
            temporary_path = Path(handle.name)
            torch.save(payload, handle)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, path)
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()


def load_checkpoint(
    path: str | Path,
    *,
    expected_dataset_fingerprint: Mapping[str, Any] | None = None,
    expected_training_config: Mapping[str, Any] | None = None,
    expected_model_config: Mapping[str, Any] | None = None,
    map_location: str | torch.device = "cpu",
) -> dict[str, Any]:
    """Load a trusted local checkpoint and reject incompatible resume state."""

    try:
        checkpoint = torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:  # PyTorch before the weights_only argument.
        checkpoint = torch.load(path, map_location=map_location)
    if not isinstance(checkpoint, dict):
        raise ValueError("checkpoint root must be a dictionary")
    version = checkpoint.get("checkpoint_format_version")
    if version != CHECKPOINT_FORMAT_VERSION:
        raise ValueError(
            f"unsupported checkpoint format {version!r}; expected {CHECKPOINT_FORMAT_VERSION}"
        )

    required = {
        "model",
        "model_config",
        "optimizer",
        "scheduler",
        "scaler",
        "optimizer_step",
        "epoch",
        "next_batch_cursor",
        "best_validation_loss",
        "elapsed_seconds",
        "counters",
        "rng_state",
        "sampler_state",
        "training_config",
        "dataset_fingerprint",
        "preprocessing_identity",
        "environment",
    }
    missing = required.difference(checkpoint)
    if missing:
        raise ValueError(f"checkpoint is missing required keys: {sorted(missing)}")

    expectations = (
        ("dataset fingerprint", expected_dataset_fingerprint, checkpoint["dataset_fingerprint"]),
        ("training configuration", expected_training_config, checkpoint["training_config"]),
        ("model configuration", expected_model_config, checkpoint["model_config"]),
    )
    for label, expected, actual in expectations:
        if expected is not None and dict(expected) != actual:
            raise ValueError(f"checkpoint {label} does not match the current run")
    return checkpoint
