"""Reproducible supervised training orchestration for PokerGPT."""

from __future__ import annotations

import copy
import json
import math
import os
import pickle
import platform
import random
import re
import shutil
import subprocess
import sys
import time
from collections import Counter
from contextlib import nullcontext
from dataclasses import asdict
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping, Sequence

import torch
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader

from .checkpoint import (
    capture_rng_state,
    dataset_fingerprint,
    load_checkpoint,
    restore_rng_state,
    save_checkpoint,
)
from .data import BatchItem, LengthAwareBatchSampler, PokerTrajectoryDataset
from .decision import ACTION_GROUPS, action_group
from .model import GPT, GPTConfig
from .run_tracking import canonical_json_sha256


def _require(mapping: Mapping[str, Any], key: str, context: str) -> Any:
    if key not in mapping:
        raise ValueError(f"{context} is missing required key {key!r}")
    return mapping[key]


def validate_training_config(config: Mapping[str, Any]) -> None:
    """Validate the complete JSON configuration before any run state is made."""

    for key in ("run_id", "data_dir", "runs_dir", "dataset", "model", "training"):
        _require(config, key, "configuration")
    run_id = str(config["run_id"])
    if not run_id.strip():
        raise ValueError("run_id must not be empty")
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", run_id):
        raise ValueError("run_id may contain only letters, digits, dots, underscores, and hyphens")

    dataset = config["dataset"]
    for key in (
        "version",
        "format",
        "vocab_size",
        "pad_token_id",
        "block_size",
        "decision_tokens",
        "splits",
    ):
        _require(dataset, key, "dataset configuration")
    if tuple(dataset["splits"]) != ("train", "val", "test"):
        raise ValueError("dataset splits must be exactly train, val, and test")
    decision_tokens = tuple(dataset["decision_tokens"])
    if len(decision_tokens) < 3 or len(set(decision_tokens)) != len(decision_tokens):
        raise ValueError("dataset decision_tokens is incomplete")
    if not {"ACTION_FOLD", "ACTION_PASSIVE", "ACTION_ALL_IN"}.issubset(decision_tokens):
        raise ValueError("dataset decision_tokens is missing a required action")
    if "RANGE_ZERO" in decision_tokens:
        raise ValueError("RANGE_ZERO cannot be a supervised decision token")
    vocab_size = int(dataset["vocab_size"])
    pad_token_id = int(dataset["pad_token_id"])
    if vocab_size <= 0 or int(dataset["block_size"]) <= 0:
        raise ValueError("dataset vocab_size and block_size must be positive")
    if not 0 <= pad_token_id < vocab_size:
        raise ValueError("dataset pad_token_id must be inside the vocabulary")

    model = config["model"]
    for key in ("n_layer", "n_head", "n_embd", "dropout", "bias"):
        _require(model, key, "model configuration")
    if int(model["n_layer"]) <= 0 or int(model["n_head"]) <= 0 or int(model["n_embd"]) <= 0:
        raise ValueError("model dimensions must be positive")
    if int(model["n_embd"]) % int(model["n_head"]):
        raise ValueError("n_embd must be divisible by n_head")
    if not 0 <= float(model["dropout"]) < 1:
        raise ValueError("dropout must be in [0, 1)")

    training = config["training"]
    required_training = (
        "seed",
        "optimizer",
        "schedule",
        "label_smoothing",
        "device",
        "precision",
        "micro_batch_size",
        "gradient_accumulation_steps",
        "loader_workers",
        "pool_size",
        "peak_learning_rate",
        "minimum_learning_rate",
        "adam_betas",
        "adam_epsilon",
        "weight_decay",
        "gradient_clip",
        "warmup_steps",
        "max_steps",
        "validation_interval",
        "checkpoint_interval",
        "archive_interval",
        "log_interval",
        "deterministic",
        "allow_tf32",
        "compile",
    )
    for key in required_training:
        _require(training, key, "training configuration")
    for key in (
        "micro_batch_size",
        "gradient_accumulation_steps",
        "pool_size",
        "max_steps",
        "validation_interval",
        "checkpoint_interval",
        "archive_interval",
        "log_interval",
    ):
        if int(training[key]) <= 0:
            raise ValueError(f"training.{key} must be positive")
    if int(training["loader_workers"]) != 0:
        raise ValueError("loader_workers must remain 0 until prefetch state is resumable")
    if int(training["pool_size"]) < int(training["micro_batch_size"]):
        raise ValueError("pool_size must be at least micro_batch_size")
    if int(training["pool_size"]) % int(training["micro_batch_size"]):
        raise ValueError("pool_size must be a multiple of micro_batch_size")
    if int(training["warmup_steps"]) < 0:
        raise ValueError("warmup_steps must be non-negative")
    if int(training["warmup_steps"]) > int(training["max_steps"]):
        raise ValueError("warmup_steps cannot exceed max_steps")
    peak = float(training["peak_learning_rate"])
    minimum = float(training["minimum_learning_rate"])
    if peak <= 0 or minimum < 0 or minimum > peak:
        raise ValueError("learning rates must satisfy peak > 0 and 0 <= minimum <= peak")
    if float(training["adam_epsilon"]) <= 0 or float(training["gradient_clip"]) <= 0:
        raise ValueError("adam_epsilon and gradient_clip must be positive")
    if float(training["weight_decay"]) < 0:
        raise ValueError("weight_decay must be non-negative")
    betas = training["adam_betas"]
    if len(betas) != 2 or not all(0 <= float(beta) < 1 for beta in betas):
        raise ValueError("adam_betas must contain two values in [0, 1)")
    if bool(training["compile"]):
        raise ValueError("torch.compile is locked off for the first baseline")
    if str(training["optimizer"]).lower() != "adamw":
        raise ValueError("optimizer must be AdamW for the first baseline")
    if str(training["schedule"]).lower() != "cosine":
        raise ValueError("schedule must be cosine for the first baseline")
    if float(training["label_smoothing"]) != 0.0:
        raise ValueError("label_smoothing is locked to 0.0 for the first baseline")
    if str(training["device"]) not in {"auto", "cpu", "cuda"}:
        raise ValueError("device must be auto, cpu, or cuda")
    if str(training["precision"]) not in {"auto", "float32", "bf16", "fp16"}:
        raise ValueError("precision must be auto, float32, bf16, or fp16")


def resolve_training_config(config: Mapping[str, Any]) -> dict[str, Any]:
    """Resolve environment-sensitive device and precision choices."""

    validate_training_config(config)
    resolved = copy.deepcopy(dict(config))
    training = resolved["training"]
    requested_device = str(training["device"])
    device = "cuda" if requested_device == "auto" and torch.cuda.is_available() else requested_device
    if device == "auto":
        device = "cpu"
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")

    requested_precision = str(training["precision"])
    if requested_precision == "auto":
        if device == "cuda":
            precision = "bf16" if torch.cuda.is_bf16_supported() else "fp16"
        else:
            precision = "float32"
    else:
        precision = requested_precision
    if precision == "fp16" and device != "cuda":
        raise ValueError("FP16 training requires CUDA")
    if precision == "bf16" and device == "cuda" and not torch.cuda.is_bf16_supported():
        raise ValueError("BF16 was requested but this CUDA device does not support it")

    training["resolved_device"] = device
    training["resolved_precision"] = precision
    return resolved


def set_reproducibility(seed: int, *, deterministic: bool, allow_tf32: bool) -> None:
    if deterministic:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(deterministic)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.benchmark = False
    if hasattr(torch.backends, "cuda") and hasattr(torch.backends.cuda, "matmul"):
        torch.backends.cuda.matmul.allow_tf32 = allow_tf32
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.allow_tf32 = allow_tf32


def _git_output(arguments: Sequence[str]) -> str | None:
    executable = os.environ.get("POKERGPT_GIT_EXECUTABLE") or shutil.which("git")
    if not executable:
        return None
    try:
        result = subprocess.run(
            [executable, *arguments],
            capture_output=True,
            text=True,
            check=False,
            timeout=10,
        )
    except (FileNotFoundError, subprocess.SubprocessError):
        return None
    return result.stdout.strip() if result.returncode == 0 else None


def collect_environment(config: Mapping[str, Any]) -> dict[str, Any]:
    device = config["training"]["resolved_device"]
    git_status = _git_output(("status", "--porcelain"))
    return {
        "python": sys.version,
        "platform": platform.platform(),
        "pytorch": torch.__version__,
        "cpu_logical_count": os.cpu_count(),
        "torch_threads": torch.get_num_threads(),
        "cuda_runtime": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version() if torch.cuda.is_available() else None,
        "cuda_available": torch.cuda.is_available(),
        "gpu": torch.cuda.get_device_name(0) if device == "cuda" else None,
        "gpu_compute_capability": (
            list(torch.cuda.get_device_capability(0)) if device == "cuda" else None
        ),
        "gpu_total_memory_bytes": (
            torch.cuda.get_device_properties(0).total_memory if device == "cuda" else None
        ),
        "git_commit": _git_output(("rev-parse", "HEAD")),
        "git_dirty": None if git_status is None else bool(git_status),
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "allow_tf32": bool(config["training"]["allow_tf32"]),
        "resolved_device": device,
        "resolved_precision": config["training"]["resolved_precision"],
        "configuration_canonical_sha256": canonical_json_sha256(config),
    }


def _write_json_atomic(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _read_meta(data_dir: Path) -> dict[str, Any]:
    with (data_dir / "meta.pkl").open("rb") as handle:
        meta = pickle.load(handle)
    if not isinstance(meta, dict):
        raise ValueError("meta.pkl root must be a dictionary")
    return meta


def verify_metadata(meta: Mapping[str, Any], expected: Mapping[str, Any]) -> None:
    checks = {
        "version": str(expected["version"]),
        "format": str(expected["format"]),
        "vocab_size": int(expected["vocab_size"]),
        "pad_token_id": int(expected["pad_token_id"]),
        "block_size": int(expected["block_size"]),
    }
    for key, expected_value in checks.items():
        if key not in meta:
            raise ValueError(f"meta.pkl is missing required key {key!r}")
        actual = type(expected_value)(meta[key])
        if actual != expected_value:
            raise ValueError(
                f"metadata {key} mismatch: expected {expected_value!r}, got {actual!r}"
            )
    expected_tokens = tuple(expected["decision_tokens"])
    actual_tokens = tuple(meta.get("decision_tokens", ()))
    if actual_tokens != expected_tokens:
        raise ValueError("metadata decision_tokens do not match the configured contract")
    stoi = meta.get("stoi", {})
    if any(token not in stoi for token in expected_tokens):
        raise ValueError("metadata vocabulary is missing a configured decision token")
    ids = tuple(int(stoi[token]) for token in expected_tokens)
    if len(set(ids)) != len(ids) or any(not 0 <= token_id < checks["vocab_size"] for token_id in ids):
        raise ValueError("metadata decision-token IDs are invalid")
    declared_ids = meta.get("decision_token_ids")
    if declared_ids is not None and tuple(map(int, declared_ids)) != ids:
        raise ValueError("metadata decision_token_ids disagree with stoi")


def make_cosine_scheduler(
    optimizer: torch.optim.Optimizer,
    *,
    warmup_steps: int,
    max_steps: int,
    minimum_ratio: float,
) -> torch.optim.lr_scheduler.LambdaLR:
    def multiplier(step: int) -> float:
        if warmup_steps and step < warmup_steps:
            return float(step + 1) / warmup_steps
        if max_steps <= warmup_steps:
            return minimum_ratio
        progress = min(max((step - warmup_steps) / (max_steps - warmup_steps), 0.0), 1.0)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return minimum_ratio + (1.0 - minimum_ratio) * cosine

    return torch.optim.lr_scheduler.LambdaLR(optimizer, multiplier)


def normalize_gradients(
    parameters: Iterable[torch.nn.Parameter], decision_count: int
) -> None:
    """Convert accumulated summed-decision gradients to their global mean."""

    if decision_count <= 0:
        raise ValueError("decision_count must be positive")
    for parameter in parameters:
        if parameter.grad is not None:
            parameter.grad.div_(decision_count)


class MetricAccumulator:
    """Decision-weighted loss and fixed-vocabulary prediction metrics."""

    def __init__(self, meta: Mapping[str, Any]) -> None:
        self.tokens = tuple(meta["decision_tokens"])
        self.token_ids = tuple(int(meta["stoi"][token]) for token in self.tokens)
        self.local_by_vocab = torch.full((int(meta["vocab_size"]),), -1, dtype=torch.long)
        for local, token_id in enumerate(self.token_ids):
            self.local_by_vocab[token_id] = local
        self.group_by_local = torch.tensor(
            [ACTION_GROUPS.index(action_group(token)) for token in self.tokens], dtype=torch.long
        )
        representatives = meta.get("range_representative_ratio", {})
        self.representatives = tuple(
            float(representatives[token]) if token in representatives else None
            for token in self.tokens
        )
        self.loss_sum = 0.0
        self.decisions = 0
        self.trajectories = 0
        self.sequence_tokens = 0
        self.exact_correct = 0
        self.action_correct = 0
        self.action_top_k_correct = 0
        self.group_truth_counts: Counter[int] = Counter()
        self.group_prediction_counts: Counter[int] = Counter()
        self.group_correct: Counter[int] = Counter()
        self.group_confusion: Counter[tuple[int, int]] = Counter()
        self.sizing_total = 0
        self.sizing_correct = 0
        self.ratio_error_sum = 0.0
        self.ratio_error_count = 0
        self.truth_counts: Counter[int] = Counter()
        self.prediction_counts: Counter[int] = Counter()
        self.confusion: Counter[tuple[int, int]] = Counter()

    def update(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        loss_mask: torch.Tensor,
        mean_loss: float,
    ) -> None:
        selected = targets.ne(-1) & loss_mask.bool()
        count = int(selected.sum().item())
        if count == 0:
            raise ValueError("metric batch has no selected decisions")
        target_ids = targets[selected].detach().cpu()
        local_truth = self.local_by_vocab[target_ids]
        if torch.any(local_truth.lt(0)):
            raise ValueError("loss mask selected a target outside the decision vocabulary")
        decision_logits = logits[selected][:, list(self.token_ids)].detach().float().cpu()
        probabilities = torch.softmax(decision_logits, dim=-1)
        local_prediction = probabilities.argmax(dim=-1)
        group_lookup = self.group_by_local
        truth_groups = group_lookup[local_truth]
        group_probabilities = torch.zeros((count, len(ACTION_GROUPS)))
        group_probabilities.scatter_add_(
            1, group_lookup.unsqueeze(0).expand(count, -1), probabilities
        )
        predicted_groups = group_probabilities.argmax(dim=-1)
        top_k_groups = group_probabilities.topk(min(2, len(ACTION_GROUPS)), dim=-1).indices

        self.loss_sum += float(mean_loss) * count
        self.decisions += count
        self.trajectories += int(targets.shape[0])
        self.sequence_tokens += int(targets.ne(-1).sum().item())
        self.exact_correct += int(local_prediction.eq(local_truth).sum().item())
        self.action_correct += int(predicted_groups.eq(truth_groups).sum().item())
        self.action_top_k_correct += int(
            top_k_groups.eq(truth_groups.unsqueeze(1)).any(dim=1).sum().item()
        )
        for truth_group, predicted_group in zip(
            truth_groups.tolist(), predicted_groups.tolist()
        ):
            self.group_truth_counts[truth_group] += 1
            self.group_prediction_counts[predicted_group] += 1
            self.group_confusion[(truth_group, predicted_group)] += 1
            if truth_group == predicted_group:
                self.group_correct[truth_group] += 1

        for truth, prediction in zip(local_truth.tolist(), local_prediction.tolist()):
            self.truth_counts[truth] += 1
            self.prediction_counts[prediction] += 1
            self.confusion[(truth, prediction)] += 1
            truth_token = self.tokens[truth]
            if truth_token.startswith("RANGE_"):
                self.sizing_total += 1
                if truth == prediction:
                    self.sizing_correct += 1
                truth_ratio = self.representatives[truth]
                prediction_ratio = self.representatives[prediction]
                if truth_ratio is not None and prediction_ratio is not None:
                    self.ratio_error_sum += abs(prediction_ratio - truth_ratio)
                    self.ratio_error_count += 1

    def compute(self) -> dict[str, Any]:
        if not self.decisions:
            raise ValueError("cannot compute metrics without decisions")
        return {
            "loss": self.loss_sum / self.decisions,
            "decisions": self.decisions,
            "trajectories": self.trajectories,
            "tokens": self.sequence_tokens,
            "decision_accuracy": self.exact_correct / self.decisions,
            "action_accuracy": self.action_correct / self.decisions,
            "action_top_k_accuracy": self.action_top_k_correct / self.decisions,
            "per_action_group_accuracy": {
                group: (
                    self.group_correct[index] / self.group_truth_counts[index]
                    if self.group_truth_counts[index]
                    else None
                )
                for index, group in enumerate(ACTION_GROUPS)
            },
            "action_group_confusion": {
                f"{ACTION_GROUPS[truth]}->{ACTION_GROUPS[prediction]}": count
                for (truth, prediction), count in sorted(self.group_confusion.items())
            },
            "sizing_accuracy": (
                self.sizing_correct / self.sizing_total if self.sizing_total else None
            ),
            "mean_absolute_ratio_error": (
                self.ratio_error_sum / self.ratio_error_count
                if self.ratio_error_count
                else None
            ),
            "per_class_truth": {
                token: self.truth_counts[index] for index, token in enumerate(self.tokens)
            },
            "per_class_prediction": {
                token: self.prediction_counts[index] for index, token in enumerate(self.tokens)
            },
            "decision_confusion": {
                f"{self.tokens[truth]}->{self.tokens[prediction]}": count
                for (truth, prediction), count in sorted(self.confusion.items())
            },
        }

    def merge(self, other: "MetricAccumulator") -> None:
        if self.tokens != other.tokens:
            raise ValueError("cannot merge metrics from different decision vocabularies")
        for attribute in (
            "loss_sum",
            "decisions",
            "trajectories",
            "sequence_tokens",
            "exact_correct",
            "action_correct",
            "action_top_k_correct",
            "sizing_total",
            "sizing_correct",
            "ratio_error_sum",
            "ratio_error_count",
        ):
            setattr(self, attribute, getattr(self, attribute) + getattr(other, attribute))
        self.truth_counts.update(other.truth_counts)
        self.prediction_counts.update(other.prediction_counts)
        self.confusion.update(other.confusion)
        self.group_truth_counts.update(other.group_truth_counts)
        self.group_prediction_counts.update(other.group_prediction_counts)
        self.group_correct.update(other.group_correct)
        self.group_confusion.update(other.group_confusion)


class PokerTrainer:
    def __init__(
        self,
        config: Mapping[str, Any],
        run_dir: str | Path,
        *,
        environment: Mapping[str, Any],
        resume_path: str | Path | None = None,
    ) -> None:
        self.config = copy.deepcopy(dict(config))
        self.run_dir = Path(run_dir)
        self.environment = dict(environment)
        training = self.config["training"]
        self.device = torch.device(training["resolved_device"])
        self.precision = str(training["resolved_precision"])
        self.data_dir = Path(self.config["data_dir"])

        self.meta = _read_meta(self.data_dir)
        verify_metadata(self.meta, self.config["dataset"])
        training_splits = ("train", "val")
        self.fingerprint = dataset_fingerprint(self.data_dir, training_splits)
        fingerprint_path = self.run_dir / "dataset_fingerprint.json"
        if resume_path is None:
            _write_json_atomic(fingerprint_path, self.fingerprint)
        else:
            if not fingerprint_path.is_file():
                raise FileNotFoundError(
                    f"saved dataset fingerprint is missing: {fingerprint_path}"
                )
            with fingerprint_path.open("r", encoding="utf-8") as handle:
                saved_fingerprint = json.load(handle)
            if saved_fingerprint != self.fingerprint:
                raise ValueError("dataset fingerprint does not match the immutable resumed run")

        self.datasets = {
            split: PokerTrajectoryDataset(self.data_dir, split)
            for split in training_splits
        }
        if not self.datasets["train"] or not self.datasets["val"]:
            raise ValueError("train and validation datasets must both be nonempty")
        self.train_sampler = LengthAwareBatchSampler(
            self.datasets["train"].trajectory_lengths,
            int(training["micro_batch_size"]),
            seed=int(training["seed"]),
            pool_size=int(training["pool_size"]),
            drop_last=False,
        )
        loader_generator = torch.Generator().manual_seed(int(training["seed"]) + 17)
        self.train_loader = DataLoader(
            self.datasets["train"],
            batch_sampler=self.train_sampler,
            collate_fn=self.datasets["train"].collate_fn,
            num_workers=0,
            pin_memory=self.device.type == "cuda",
            generator=loader_generator,
        )
        self._train_iterator: Iterator[BatchItem] | None = None

        model_settings = self.config["model"]
        self.model_config = GPTConfig(
            block_size=int(self.meta["block_size"]),
            vocab_size=int(self.meta["vocab_size"]),
            n_layer=int(model_settings["n_layer"]),
            n_head=int(model_settings["n_head"]),
            n_embd=int(model_settings["n_embd"]),
            dropout=float(model_settings["dropout"]),
            bias=bool(model_settings["bias"]),
        )
        self.model = GPT(self.model_config).to(self.device)
        self.optimizer = self.model.configure_optimizer(
            weight_decay=float(training["weight_decay"]),
            learning_rate=float(training["peak_learning_rate"]),
            betas=tuple(map(float, training["adam_betas"])),
            eps=float(training["adam_epsilon"]),
            device_type=self.device.type,
        )
        self.scheduler = make_cosine_scheduler(
            self.optimizer,
            warmup_steps=int(training["warmup_steps"]),
            max_steps=int(training["max_steps"]),
            minimum_ratio=float(training["minimum_learning_rate"])
            / float(training["peak_learning_rate"]),
        )
        self.scaler = self._make_scaler()
        self.optimizer_step = 0
        self.best_validation_loss = math.inf
        self.counters = {"trajectories": 0, "tokens": 0, "decisions": 0}
        self.started_at = time.perf_counter()
        self.elapsed_before_resume = 0.0
        self.last_validation_step = -1
        self.last_validation_metrics: dict[str, Any] | None = None

        if resume_path is not None:
            self._resume(resume_path)
        self._print_summary()

    def _make_scaler(self) -> Any | None:
        if self.precision != "fp16":
            return None
        try:
            return torch.amp.GradScaler("cuda")
        except (AttributeError, TypeError):
            return torch.cuda.amp.GradScaler()

    def _autocast(self):
        if self.precision == "float32":
            return nullcontext()
        dtype = torch.bfloat16 if self.precision == "bf16" else torch.float16
        return torch.autocast(device_type=self.device.type, dtype=dtype)

    def _resume(self, path: str | Path) -> None:
        checkpoint = load_checkpoint(
            path,
            expected_dataset_fingerprint=self.fingerprint,
            expected_training_config=self.config,
            expected_model_config=asdict(self.model_config),
            map_location="cpu",
        )
        self.model.load_state_dict(checkpoint["model"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        self.scheduler.load_state_dict(checkpoint["scheduler"])
        saved_scaler = checkpoint["scaler"]
        if (self.scaler is None) != (saved_scaler is None):
            raise ValueError("checkpoint precision scaler does not match resolved precision")
        if self.scaler is not None:
            self.scaler.load_state_dict(saved_scaler)
        self.optimizer_step = int(checkpoint["optimizer_step"])
        self.best_validation_loss = float(checkpoint["best_validation_loss"])
        self.elapsed_before_resume = float(checkpoint["elapsed_seconds"])
        saved_counters = checkpoint["counters"]
        if set(saved_counters) != set(self.counters):
            raise ValueError("checkpoint counters do not match the trainer contract")
        self.counters = {key: int(value) for key, value in saved_counters.items()}
        self.train_sampler.load_state_dict(checkpoint["sampler_state"])
        if self.train_sampler.epoch != int(checkpoint["epoch"]):
            raise ValueError("checkpoint epoch disagrees with sampler state")
        if self.train_sampler.next_batch_cursor != int(checkpoint["next_batch_cursor"]):
            raise ValueError("checkpoint batch cursor disagrees with sampler state")
        restore_rng_state(checkpoint["rng_state"])

    def _elapsed(self) -> float:
        return self.elapsed_before_resume + (time.perf_counter() - self.started_at)

    def _print_summary(self) -> None:
        train = self.datasets["train"]
        steps_per_epoch = math.ceil(
            self.train_sampler.total_batches
            / int(self.config["training"]["gradient_accumulation_steps"])
        )
        print(
            f"device={self.device} precision={self.precision} "
            f"parameters={self.model.get_num_params(non_embedding=False):,}"
        )
        for split, dataset in self.datasets.items():
            average = dataset.decision_count / len(dataset)
            print(
                f"{split}: trajectories={len(dataset):,} tokens={dataset.token_count:,} "
                f"decisions={dataset.decision_count:,} decisions/trajectory={average:.3f}"
            )
        print(
            f"steps/epoch~={steps_per_epoch:,} "
            f"planned_updates={int(self.config['training']['max_steps']):,}"
        )

    def _next_batch(self) -> BatchItem:
        while True:
            if self._train_iterator is None:
                self._train_iterator = iter(self.train_loader)
            try:
                return next(self._train_iterator)
            except StopIteration:
                self.train_sampler.set_epoch(self.train_sampler.epoch + 1)
                self._train_iterator = iter(self.train_loader)

    def _move_batch(self, batch: BatchItem) -> BatchItem:
        return tuple(
            tensor.to(self.device, non_blocking=self.device.type == "cuda") for tensor in batch
        )  # type: ignore[return-value]

    def train_step(self) -> tuple[MetricAccumulator, dict[str, float]]:
        step_started = time.perf_counter()
        training = self.config["training"]
        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)
        accumulator = MetricAccumulator(self.meta)
        total_decisions = 0
        learning_rate = float(self.optimizer.param_groups[0]["lr"])

        for _ in range(int(training["gradient_accumulation_steps"])):
            inputs, targets, loss_mask = self._move_batch(self._next_batch())
            decisions = int((targets.ne(-1) & loss_mask.bool()).sum().item())
            if decisions <= 0:
                raise ValueError("training microbatch has no supervised decisions")
            with self._autocast():
                logits, loss = self.model(inputs, targets, loss_mask)
            assert loss is not None
            if not torch.isfinite(loss):
                raise FloatingPointError("nonfinite training loss")
            weighted_loss = loss * decisions
            if self.scaler is None:
                weighted_loss.backward()
            else:
                self.scaler.scale(weighted_loss).backward()
            accumulator.update(logits, targets, loss_mask, float(loss.detach().item()))
            total_decisions += decisions

        if self.scaler is not None:
            self.scaler.unscale_(self.optimizer)
        normalize_gradients(self.model.parameters(), total_decisions)
        gradient_norm = clip_grad_norm_(
            self.model.parameters(), float(training["gradient_clip"]), error_if_nonfinite=True
        )
        if self.scaler is None:
            self.optimizer.step()
        else:
            self.scaler.step(self.optimizer)
            self.scaler.update()
        self.scheduler.step()
        self.optimizer_step += 1

        metrics = accumulator.compute()
        for key in self.counters:
            self.counters[key] += int(metrics[key])
        return accumulator, {
            "learning_rate": learning_rate,
            "gradient_norm": float(gradient_norm.item()),
            "step_seconds": time.perf_counter() - step_started,
        }

    @torch.inference_mode()
    def validate(self) -> dict[str, Any]:
        self.model.eval()
        dataset = self.datasets["val"]
        training = self.config["training"]
        sampler = LengthAwareBatchSampler(
            dataset.trajectory_lengths,
            int(training["micro_batch_size"]),
            seed=int(training["seed"]) + 1_000_003,
            pool_size=int(training["pool_size"]),
            drop_last=False,
        )
        generator = torch.Generator().manual_seed(int(training["seed"]) + 31)
        loader = DataLoader(
            dataset,
            batch_sampler=sampler,
            collate_fn=dataset.collate_fn,
            num_workers=0,
            pin_memory=self.device.type == "cuda",
            generator=generator,
        )
        accumulator = MetricAccumulator(self.meta)
        for batch in loader:
            inputs, targets, loss_mask = self._move_batch(batch)
            with self._autocast():
                logits, loss = self.model(inputs, targets, loss_mask)
            assert loss is not None
            if not torch.isfinite(loss):
                raise FloatingPointError("nonfinite validation loss")
            accumulator.update(logits, targets, loss_mask, float(loss.item()))
        self.last_validation_step = self.optimizer_step
        self.last_validation_metrics = accumulator.compute()
        return self.last_validation_metrics

    def _checkpoint_payload(self) -> dict[str, Any]:
        return {
            "model": self.model.state_dict(),
            "model_config": asdict(self.model_config),
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict(),
            "scaler": self.scaler.state_dict() if self.scaler is not None else None,
            "optimizer_step": self.optimizer_step,
            "epoch": self.train_sampler.epoch,
            "next_batch_cursor": self.train_sampler.next_batch_cursor,
            "best_validation_loss": self.best_validation_loss,
            "elapsed_seconds": self._elapsed(),
            "counters": dict(self.counters),
            "rng_state": capture_rng_state(),
            "sampler_state": self.train_sampler.state_dict(),
            "training_config": copy.deepcopy(self.config),
            "dataset_fingerprint": self.fingerprint,
            "preprocessing_identity": {
                "version": self.meta["version"],
                "format": self.meta["format"],
                "vocab_size": self.meta["vocab_size"],
                "block_size": self.meta["block_size"],
            },
            "environment": dict(self.environment),
        }

    def save(self, name: str) -> Path:
        path = self.run_dir / name
        save_checkpoint(path, self._checkpoint_payload())
        return path

    def _log(self, split: str, metrics: Mapping[str, Any], elapsed: float) -> None:
        record = {
            "split": split,
            "optimizer_step": self.optimizer_step,
            "epoch": self.train_sampler.epoch,
            "next_batch_cursor": self.train_sampler.next_batch_cursor,
            "elapsed_seconds": elapsed,
            **metrics,
        }
        if self.device.type == "cuda":
            record["peak_cuda_memory_bytes"] = torch.cuda.max_memory_allocated(self.device)
        with (self.run_dir / "metrics.jsonl").open("a", encoding="utf-8", newline="\n") as handle:
            handle.write(json.dumps(record, sort_keys=True) + "\n")
            handle.flush()

    def _validate_and_checkpoint_best(self) -> dict[str, Any]:
        metrics = self.validate()
        self._log("val", metrics, self._elapsed())
        print(
            f"step {self.optimizer_step}: val loss={metrics['loss']:.6f} "
            f"decision_acc={metrics['decision_accuracy']:.4f}"
        )
        if float(metrics["loss"]) < self.best_validation_loss:
            self.best_validation_loss = float(metrics["loss"])
            self.save("best.pt")
        return metrics

    def train(self, *, stop_after_steps: int | None = None) -> dict[str, Any]:
        training = self.config["training"]
        target_step = int(training["max_steps"])
        if stop_after_steps is not None:
            if stop_after_steps < self.optimizer_step:
                raise ValueError("stop_after_steps precedes the resumed optimizer step")
            target_step = min(target_step, int(stop_after_steps))
        interval_accumulator = MetricAccumulator(self.meta)
        latest_step_metrics: dict[str, float] = {}
        interval_train_seconds = 0.0
        if self.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(self.device)

        while self.optimizer_step < target_step:
            step_accumulator, latest_step_metrics = self.train_step()
            interval_accumulator.merge(step_accumulator)
            interval_train_seconds += latest_step_metrics["step_seconds"]
            elapsed = self._elapsed()

            if self.optimizer_step % int(training["log_interval"]) == 0:
                duration = max(interval_train_seconds, 1e-9)
                log_metrics = interval_accumulator.compute()
                decisions = int(log_metrics["decisions"])
                tokens = int(log_metrics["tokens"])
                log_metrics.update(latest_step_metrics)
                log_metrics["decisions_per_second"] = decisions / duration
                log_metrics["tokens_per_second"] = tokens / duration
                self._log("train", log_metrics, elapsed)
                print(
                    f"step {self.optimizer_step}: train loss={log_metrics['loss']:.6f} "
                    f"lr={latest_step_metrics['learning_rate']:.3e} "
                    f"decisions/s={decisions / duration:.1f}"
                )
                interval_accumulator = MetricAccumulator(self.meta)
                interval_train_seconds = 0.0

            if self.optimizer_step % int(training["validation_interval"]) == 0:
                self._validate_and_checkpoint_best()
            if self.optimizer_step % int(training["checkpoint_interval"]) == 0:
                self.save("latest.pt")
            if self.optimizer_step % int(training["archive_interval"]) == 0:
                self.save(f"step_{self.optimizer_step:08d}.pt")

        elapsed = self._elapsed()
        if interval_accumulator.decisions:
            duration = max(interval_train_seconds, 1e-9)
            log_metrics = interval_accumulator.compute()
            decisions = int(log_metrics["decisions"])
            tokens = int(log_metrics["tokens"])
            log_metrics.update(latest_step_metrics)
            log_metrics["decisions_per_second"] = decisions / duration
            log_metrics["tokens_per_second"] = tokens / duration
            self._log("train", log_metrics, elapsed)
        if self.last_validation_step != self.optimizer_step:
            final_validation = self._validate_and_checkpoint_best()
        else:
            assert self.last_validation_metrics is not None
            final_validation = self.last_validation_metrics
        self.save("latest.pt")
        return {
            "optimizer_step": self.optimizer_step,
            "best_validation_loss": self.best_validation_loss,
            "final_validation": final_validation,
            "counters": dict(self.counters),
            "elapsed_seconds": self._elapsed(),
            "run_dir": str(self.run_dir),
        }


def prepare_run(
    config: Mapping[str, Any], *, resume_path: str | Path | None = None
) -> tuple[dict[str, Any], Path, dict[str, Any]]:
    resolved = resolve_training_config(config)
    training = resolved["training"]
    set_reproducibility(
        int(training["seed"]),
        deterministic=bool(training["deterministic"]),
        allow_tf32=bool(training["allow_tf32"]),
    )
    run_dir = Path(resolved["runs_dir"]) / str(resolved["run_id"])
    config_path = run_dir / "config.json"
    if resume_path is None:
        if run_dir.exists() and any(run_dir.iterdir()):
            raise FileExistsError(f"run directory already contains files: {run_dir}")
        run_dir.mkdir(parents=True, exist_ok=True)
        _write_json_atomic(config_path, resolved)
        _write_json_atomic(
            run_dir / "run_manifest.json",
            {
                "format_version": 1,
                "run_id": str(resolved["run_id"]),
                "seed": int(resolved["training"]["seed"]),
                "configuration_canonical_sha256": canonical_json_sha256(resolved),
            },
        )
        (run_dir / "metrics.jsonl").touch(exist_ok=False)
    else:
        resume = Path(resume_path).resolve()
        if resume.parent != run_dir.resolve():
            raise ValueError("resume checkpoint is not inside the configured run directory")
        if not config_path.is_file():
            raise FileNotFoundError(f"resolved run configuration is missing: {config_path}")
        with config_path.open("r", encoding="utf-8") as handle:
            saved_config = json.load(handle)
        if saved_config != resolved:
            raise ValueError("configuration does not match the immutable resumed run")
    environment = collect_environment(resolved)
    if resume_path is None:
        _write_json_atomic(run_dir / "environment.json", environment)
    return resolved, run_dir, environment


def run_training(
    config: Mapping[str, Any],
    *,
    resume_path: str | Path | None = None,
    stop_after_steps: int | None = None,
) -> dict[str, Any]:
    resolved, run_dir, environment = prepare_run(config, resume_path=resume_path)
    trainer = PokerTrainer(
        resolved, run_dir, environment=environment, resume_path=resume_path
    )
    return trainer.train(stop_after_steps=stop_after_steps)
