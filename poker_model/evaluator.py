"""Replay-aligned, deterministic evaluation for the frozen PokerGPT baseline."""

from __future__ import annotations

import hashlib
import json
import os
import pickle
import platform
import zipfile
from contextlib import nullcontext
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path
from typing import Any, Mapping

import torch

from poker_pipeline.io_utils import read_jsonl
from poker_pipeline.legality import LegalityState
from poker_pipeline.phh import Decision, build_hero_trajectories, iter_archive_hand_decisions
from poker_pipeline.tokenizer import PokerTokenizer

from .checkpoint import load_checkpoint
from .data import PokerTrajectoryDataset
from .decision import action_group, decision_distribution
from .evaluation import FrozenEvaluationMetrics
from .evaluation_access import TestSplitPermit, require_split_access
from .model import GPT, GPTConfig
from .run_tracking import canonical_json_sha256, file_identity


EVALUATOR_FORMAT_VERSION = 1
STREETS = ("PREFLOP", "FLOP", "TURN", "RIVER")


@dataclass(frozen=True)
class DecisionEvaluationContext:
    street: str
    truth_token: str
    truth_ratio: Decimal | None
    legality_state: LegalityState


@dataclass(frozen=True)
class TrajectoryEvaluationContext:
    member: str
    hand_key: str
    hero: int
    decisions: tuple[DecisionEvaluationContext, ...]


def load_evaluator_config(path: str | Path) -> dict[str, Any]:
    config = json.loads(Path(path).read_text(encoding="utf-8"))
    required = {
        "format_version",
        "evaluator_id",
        "prediction_policy",
        "decision_top_k",
        "mapped_action_top_k",
        "batch_size",
        "device",
        "precision",
        "candidate",
        "freeze_files",
    }
    missing = required.difference(config)
    if missing:
        raise ValueError(f"evaluator configuration is missing keys: {sorted(missing)}")
    if int(config["format_version"]) != EVALUATOR_FORMAT_VERSION:
        raise ValueError("unsupported evaluator configuration format")
    if config["prediction_policy"] != "decision_token_argmax":
        raise ValueError("the frozen evaluator requires decision_token_argmax")
    if int(config["decision_top_k"]) <= 1:
        raise ValueError("decision_top_k must be greater than one")
    if int(config["mapped_action_top_k"]) <= 1:
        raise ValueError("mapped_action_top_k must be greater than one")
    if int(config["batch_size"]) <= 0:
        raise ValueError("batch_size must be positive")
    if config["device"] not in {"auto", "cpu", "cuda"}:
        raise ValueError("device must be auto, cpu, or cuda")
    if config["precision"] not in {"auto", "float32", "bf16", "fp16"}:
        raise ValueError("precision must be auto, float32, bf16, or fp16")
    candidate = config["candidate"]
    for key in (
        "run_id",
        "seed",
        "optimizer_step",
        "checkpoint_sha256",
        "dataset_fingerprint",
        "artifact_bundle_sha256",
    ):
        if key not in candidate:
            raise ValueError(f"candidate configuration is missing {key!r}")
    return config


def _read_meta(data_dir: Path) -> dict[str, Any]:
    with (data_dir / "meta.pkl").open("rb") as handle:
        meta = pickle.load(handle)
    if not isinstance(meta, dict):
        raise ValueError("meta.pkl root must be a dictionary")
    return meta


def _selected_rows(selection_path: Path, split: str) -> list[dict[str, Any]]:
    rows = [row for row in read_jsonl(selection_path) if row.get("split") == split]
    if not rows:
        raise ValueError(f"selection contains no {split!r} rows")
    return rows


def _verify_zip_directory(zip_path: Path, rows: list[Mapping[str, Any]]) -> None:
    """Check selected member identities without extracting the archive."""

    with zipfile.ZipFile(zip_path) as archive:
        for row in rows:
            info = archive.getinfo(str(row["member"]))
            expected_crc = str(row.get("crc32", "")).lower()
            actual_crc = f"{info.CRC:08x}"
            if expected_crc and actual_crc != expected_crc:
                raise ValueError(f"archive CRC mismatch for {row['member']}")
            if int(row.get("uncompressed_size", info.file_size)) != info.file_size:
                raise ValueError(f"archive size mismatch for {row['member']}")
            if int(row.get("compressed_size", info.compress_size)) != info.compress_size:
                raise ValueError(f"archive compressed-size mismatch for {row['member']}")


def _stored_ids(dataset: PokerTrajectoryDataset, index: int) -> tuple[int, ...]:
    inputs, targets, _ = dataset[index]
    return tuple(inputs.tolist()) + (int(targets[-1].item()),)


def build_replay_contexts(
    *,
    zip_path: str | Path,
    selection_path: str | Path,
    dataset: PokerTrajectoryDataset,
    split: str,
    meta: Mapping[str, Any],
    test_permit: TestSplitPermit | None = None,
) -> list[TrajectoryEvaluationContext]:
    """Replay one split and prove exact alignment with every prepared trajectory."""

    require_split_access(split, test_permit)
    zip_path = Path(zip_path)
    selection_path = Path(selection_path)
    rows = _selected_rows(selection_path, split)
    _verify_zip_directory(zip_path, rows)
    tokenizer = PokerTokenizer()
    if tokenizer.itos != list(meta["itos"]):
        raise ValueError("runtime tokenizer does not match prepared metadata")
    contexts: list[TrajectoryEvaluationContext] = []
    trajectory_index = 0
    for selected, decisions, error in iter_archive_hand_decisions(zip_path, rows):
        if error is not None:
            raise ValueError(f"replay failed for {selected['member']}: {error}")
        assert decisions is not None
        for trajectory in build_hero_trajectories(decisions):
            if trajectory_index >= len(dataset):
                raise ValueError("replay produced more trajectories than the binary split")
            encoded = tokenizer.encode_trajectory(trajectory)
            stored = _stored_ids(dataset, trajectory_index)
            if encoded.ids != stored:
                raise ValueError(
                    "replay/binary trajectory mismatch at "
                    f"{trajectory_index}: {trajectory.member}:{trajectory.hand_key}:"
                    f"p{trajectory.hero + 1}"
                )
            decision_contexts: list[DecisionEvaluationContext] = []
            for item in trajectory.items:
                if not isinstance(item, Decision):
                    continue
                truth_token = tokenizer.hero_decision_token(item)
                truth_ratio = (
                    item.target_amount_pot
                    if action_group(truth_token) == "AGGRESSIVE"
                    else None
                )
                decision_contexts.append(
                    DecisionEvaluationContext(
                        street=item.street,
                        truth_token=truth_token,
                        truth_ratio=truth_ratio,
                        legality_state=LegalityState(
                            pot=item.pot,
                            to_call=item.to_call,
                            current_bet=item.current_bet,
                            street_contribution=item.current_bet - item.to_call,
                            remaining_stack=item.hero_stack,
                            minimum_bet=item.minimum_bet,
                            minimum_raise_increment=item.minimum_raise_increment,
                            raise_reopened=item.raise_reopened,
                        ),
                    )
                )
            if len(decision_contexts) != int(sum(encoded.loss_mask)):
                raise ValueError("replay decision count does not match encoded loss mask")
            contexts.append(
                TrajectoryEvaluationContext(
                    member=trajectory.member,
                    hand_key=trajectory.hand_key,
                    hero=trajectory.hero,
                    decisions=tuple(decision_contexts),
                )
            )
            trajectory_index += 1
    if trajectory_index != len(dataset):
        raise ValueError(
            f"replay produced {trajectory_index} trajectories, binary split has {len(dataset)}"
        )
    return contexts


def _resolve_device_precision(config: Mapping[str, Any]) -> tuple[torch.device, str]:
    requested_device = str(config["device"])
    device_name = (
        "cuda"
        if requested_device == "auto" and torch.cuda.is_available()
        else "cpu"
        if requested_device == "auto"
        else requested_device
    )
    if device_name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    requested_precision = str(config["precision"])
    if requested_precision == "auto":
        precision = (
            "bf16"
            if device_name == "cuda" and torch.cuda.is_bf16_supported()
            else "float32"
        )
    else:
        precision = requested_precision
    if precision == "fp16" and device_name != "cuda":
        raise ValueError("FP16 evaluation requires CUDA")
    if precision == "bf16" and device_name == "cuda" and not torch.cuda.is_bf16_supported():
        raise ValueError("BF16 evaluation is unavailable on this CUDA device")
    return torch.device(device_name), precision


def _autocast(device: torch.device, precision: str):
    if precision == "float32":
        return nullcontext()
    dtype = torch.bfloat16 if precision == "bf16" else torch.float16
    return torch.autocast(device_type=device.type, dtype=dtype)


def _split_file_identities(data_dir: Path, split: str) -> dict[str, Any]:
    return {
        name: file_identity(data_dir / name)
        for name in (
            "meta.pkl",
            f"{split}.bin",
            f"{split}_loss_mask.bin",
            f"{split}.idx",
        )
    }


@torch.inference_mode()
def evaluate_split(
    config: Mapping[str, Any],
    *,
    split: str,
    zip_path: str | Path,
    selection_path: str | Path,
    data_dir: str | Path,
    checkpoint_path: str | Path,
    test_permit: TestSplitPermit | None = None,
) -> dict[str, Any]:
    """Evaluate validation or an explicitly permitted final held-out test split."""

    if split not in {"val", "test"}:
        raise ValueError("the frozen evaluator accepts only val or test")
    require_split_access(split, test_permit)
    data_dir = Path(data_dir)
    checkpoint_path = Path(checkpoint_path)
    checkpoint_identity = file_identity(checkpoint_path)
    expected_checkpoint = str(config["candidate"]["checkpoint_sha256"]).lower()
    if checkpoint_identity["sha256"] != expected_checkpoint:
        raise ValueError("checkpoint hash does not match the frozen candidate")
    checkpoint = load_checkpoint(checkpoint_path, map_location="cpu")
    if int(checkpoint["optimizer_step"]) != int(config["candidate"]["optimizer_step"]):
        raise ValueError("checkpoint step does not match the frozen candidate")
    if int(checkpoint["training_config"]["training"]["seed"]) != int(
        config["candidate"]["seed"]
    ):
        raise ValueError("checkpoint seed does not match the frozen candidate")
    if checkpoint["dataset_fingerprint"]["digest"] != config["candidate"]["dataset_fingerprint"]:
        raise ValueError("checkpoint dataset fingerprint does not match the candidate")

    meta = _read_meta(data_dir)
    dataset = PokerTrajectoryDataset(
        data_dir, split, test_permit=test_permit
    )
    contexts = build_replay_contexts(
        zip_path=zip_path,
        selection_path=selection_path,
        dataset=dataset,
        split=split,
        meta=meta,
        test_permit=test_permit,
    )
    device, precision = _resolve_device_precision(config)
    model_config = GPTConfig(**checkpoint["model_config"])
    model = GPT(model_config)
    model.load_state_dict(checkpoint["model"])
    model.to(device)
    model.eval()
    metrics = FrozenEvaluationMetrics(
        meta,
        decision_top_k=int(config["decision_top_k"]),
        action_top_k=int(config["mapped_action_top_k"]),
    )
    batch_size = int(config["batch_size"])
    evaluated_decisions = 0
    for batch_start in range(0, len(dataset), batch_size):
        batch_end = min(batch_start + batch_size, len(dataset))
        items = [dataset[index] for index in range(batch_start, batch_end)]
        inputs, targets, loss_mask = dataset.collate_fn(items)
        inputs = inputs.to(device)
        targets_device = targets.to(device)
        masks_device = loss_mask.to(device)
        with _autocast(device, precision):
            logits, _ = model(inputs, targets_device, masks_device)
        for local_index, context in enumerate(contexts[batch_start:batch_end]):
            positions = loss_mask[local_index].nonzero(as_tuple=False).flatten().tolist()
            if len(positions) != len(context.decisions):
                raise ValueError("batch loss mask does not align with replay decisions")
            for position, decision_context in zip(positions, context.decisions):
                truth_id = int(targets[local_index, position].item())
                if meta["itos"][truth_id] != decision_context.truth_token:
                    raise ValueError("binary target does not match replay truth token")
                distribution = decision_distribution(
                    logits[local_index, position], meta
                )
                metrics.update(
                    distribution,
                    truth_token=decision_context.truth_token,
                    truth_ratio=decision_context.truth_ratio,
                    street=decision_context.street,
                    legality_state=decision_context.legality_state,
                )
                evaluated_decisions += 1
    if evaluated_decisions != dataset.decision_count:
        raise ValueError("evaluated decision count does not match the split loss mask")
    result = metrics.compute()
    missing_streets = set(STREETS).difference(result["per_street"])
    if missing_streets:
        raise ValueError(f"evaluation split has no decisions for streets: {sorted(missing_streets)}")
    return {
        "report_format_version": EVALUATOR_FORMAT_VERSION,
        "evaluator_id": config["evaluator_id"],
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "split": split,
        "complete_split": True,
        "prediction_policy": config["prediction_policy"],
        "evaluator_configuration_sha256": canonical_json_sha256(config),
        "candidate": dict(config["candidate"]),
        "checkpoint": checkpoint_identity,
        "dataset_files": _split_file_identities(data_dir, split),
        "selection": file_identity(selection_path),
        "archive": {
            "path": Path(zip_path).as_posix(),
            "bytes": Path(zip_path).stat().st_size,
            "mode": "streamed selected members only; never extracted",
        },
        "environment": {
            "python": platform.python_version(),
            "pytorch": torch.__version__,
            "device": str(device),
            "precision": precision,
            "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
        },
        "metric_definitions": {
            "joint_decision_token": (
                "argmax/top-k after softmax over the fixed decision-token vocabulary"
            ),
            "mapped_action": (
                "the same raw token mapped to FOLD/CHECK/CALL/BET/RAISE from exact "
                "replay state; top-k sums token probabilities by mapped action"
            ),
            "illegal_move": (
                "raw argmax token is not executable without clamping, including "
                "minimum-size, all-in under-raise, and action-reopening semantics"
            ),
            "sizing_range_error_overall": (
                "non-exact decision token among every ground-truth aggressive "
                "decision; non-aggressive predictions count as errors"
            ),
            "sizing_range_error_conditional": (
                "non-exact decision token only where the raw prediction maps to the "
                "correct aggressive five-way action"
            ),
        },
        "metrics": result,
    }


def report_markdown(report: Mapping[str, Any]) -> str:
    metrics = report["metrics"]
    sizing = metrics["sizing"]
    lines = [
        f"# PokerGPT {report['split']} evaluation",
        "",
        f"- Evaluator: `{report['evaluator_id']}`",
        f"- Candidate checkpoint: `{report['checkpoint']['sha256']}`",
        f"- Optimizer step: {report['candidate']['optimizer_step']}",
        f"- Seed: {report['candidate']['seed']}",
        f"- Decisions: {metrics['decisions']:,}",
        "",
        "## Primary metrics",
        "",
        f"- Joint decision-token accuracy: {metrics['joint_decision_token_accuracy']:.6f}",
        (
            f"- Joint decision-token top-{metrics['decision_top_k']} accuracy: "
            f"{metrics['joint_decision_token_top_k_accuracy']:.6f}"
        ),
        f"- Mapped action accuracy: {metrics['mapped_action_accuracy']:.6f}",
        (
            f"- Mapped action top-{metrics['mapped_action_top_k']} accuracy: "
            f"{metrics['mapped_action_top_k_accuracy']:.6f}"
        ),
        f"- Illegal-move rate: {metrics['illegal_move_rate']:.6f}",
        (
            "- Sizing-range error, overall: "
            f"{sizing['range_error_rate_overall']:.6f}"
        ),
        (
            "- Sizing-range error, conditional on aggressive-action correctness: "
            f"{sizing['range_error_rate_conditional_on_aggressive_action_correct']:.6f}"
        ),
        "",
        "## Per-street results",
        "",
        "| Street | Decisions | Token accuracy | Action accuracy | Illegal rate |",
        "|---|---:|---:|---:|---:|",
    ]
    for street, values in metrics["per_street"].items():
        lines.append(
            f"| {street} | {values['decisions']:,} | "
            f"{values['joint_decision_token_accuracy']:.6f} | "
            f"{values['mapped_action_accuracy']:.6f} | "
            f"{values['illegal_move_rate']:.6f} |"
        )
    lines += [
        "",
        "## Action confusion matrix",
        "",
        "| Truth \\ Predicted | " + " | ".join(("FOLD", "CHECK", "CALL", "BET", "RAISE")) + " |",
        "|---|" + "---:|" * 5,
    ]
    matrix = metrics["action_confusion_matrix"]
    for truth in ("FOLD", "CHECK", "CALL", "BET", "RAISE"):
        lines.append(
            f"| {truth} | "
            + " | ".join(str(matrix[truth][predicted]) for predicted in (
                "FOLD", "CHECK", "CALL", "BET", "RAISE"
            ))
            + " |"
        )
    lines += [
        "",
        "This report contains aggregate results only. The held-out split was not "
        "used for evaluator development or checkpoint selection.",
        "",
    ]
    return "\n".join(lines)


def write_json(
    path: str | Path, value: Mapping[str, Any], *, exclusive: bool
) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    if exclusive:
        descriptor = os.open(target, os.O_WRONLY | os.O_CREAT | os.O_EXCL)
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            json.dump(dict(value), handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
    else:
        temporary = target.with_name(f".{target.name}.tmp")
        with temporary.open("w", encoding="utf-8", newline="\n") as handle:
            json.dump(dict(value), handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, target)
    return target


def write_markdown(path: str | Path, text: str, *, exclusive: bool) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    mode = "x" if exclusive else "w"
    with target.open(mode, encoding="utf-8", newline="\n") as handle:
        handle.write(text)
        handle.flush()
        os.fsync(handle.fileno())
    return target
