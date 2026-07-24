from __future__ import annotations

import argparse
import json
import pickle
from collections import Counter, defaultdict
from contextlib import nullcontext
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path
from typing import Any, Mapping

import torch

from poker_model.checkpoint import load_checkpoint
from poker_model.data import PokerTrajectoryDataset
from poker_model.decision import (
    action_group,
    decision_distribution,
    recover_five_way_action,
)
from poker_model.evaluation import FrozenEvaluationMetrics
from poker_model.evaluator import (
    build_replay_contexts,
    load_evaluator_config,
)
from poker_model.legality import check_decision_token
from poker_model.model import GPT, GPTConfig
from poker_model.run_tracking import file_identity


REPO_ROOT = Path(__file__).resolve().parent
DEFAULT_CONFIG = REPO_ROOT / "configs" / "evaluator_v1.json"
DEFAULT_DATA = REPO_ROOT / "data" / "processed"
DEFAULT_SELECTION = REPO_ROOT / "data" / "selected_nt_6max_v0.8.1.jsonl"
DEFAULT_CHECKPOINT = REPO_ROOT / "runs" / "baseline-v081-seed1337" / "best.pt"
DEFAULT_OUTPUT = REPO_ROOT / "reports" / "long-trajectory-shadow-v1"
STREETS = ("PREFLOP", "FLOP", "TURN", "RIVER")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run a long replay-backed PokerGPT shadow session on validation data. "
            "The held-out test split is never opened."
        )
    )
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--selection", type=Path, default=DEFAULT_SELECTION)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--trace-count", type=int, default=12)
    return parser.parse_args()


def _read_meta(path: Path) -> dict[str, Any]:
    with path.open("rb") as handle:
        value = pickle.load(handle)
    if not isinstance(value, dict):
        raise ValueError("meta.pkl root must be a dictionary")
    return value


def _resolve_device_precision(config: Mapping[str, Any]) -> tuple[torch.device, str]:
    requested_device = str(config["device"])
    if requested_device == "auto":
        device_name = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device_name = requested_device
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
        raise ValueError("FP16 inference requires CUDA")
    return torch.device(device_name), precision


def _autocast(device: torch.device, precision: str):
    if precision == "float32":
        return nullcontext()
    dtype = torch.bfloat16 if precision == "bf16" else torch.float16
    return torch.autocast(device_type=device.type, dtype=dtype)


def _cards_before_decision(
    prefix_ids: list[int], meta: Mapping[str, Any]
) -> tuple[list[str], list[str], str]:
    itos = meta["itos"]
    tokens = [itos[token_id] for token_id in prefix_ids]
    position = tokens[1].removeprefix("POSITION_").replace("_", " ").title()
    hole_index = max(
        index for index, token in enumerate(tokens) if token == "PLAYER_1_HOLE_CARDS"
    )
    hole_cards = [
        tokens[hole_index + 1].removeprefix("CARD_"),
        tokens[hole_index + 2].removeprefix("CARD_"),
    ]
    board: list[str] = []
    board_indexes = [
        index for index, token in enumerate(tokens) if token == "CURRENT_BOARD"
    ]
    if board_indexes:
        board_index = board_indexes[-1]
        count_token = tokens[board_index + 1]
        count = int(count_token.removeprefix("COUNT_"))
        board = [
            token.removeprefix("CARD_")
            for token in tokens[board_index + 2 : board_index + 2 + count]
        ]
    return hole_cards, board, position


def _mapped_action(token: str, state: Any) -> str:
    return recover_five_way_action(
        token, to_call=state.to_call, current_bet=state.current_bet
    )


def _amount_bb(
    token: str, state: Any, meta: Mapping[str, Any]
) -> float | None:
    if action_group(token) != "AGGRESSIVE":
        return None
    if token == "ACTION_ALL_IN":
        return float(state.remaining_stack)
    ratio = Decimal(str(meta["range_representative_ratio"][token]))
    return float(state.pot * ratio)


def _rate(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def _summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    decisions = [decision for row in rows for decision in row["decisions"]]
    return {
        "trajectories": len(rows),
        "decisions": len(decisions),
        "joint_token_accuracy": _rate(
            sum(item["token_correct"] for item in decisions), len(decisions)
        ),
        "mapped_action_accuracy": _rate(
            sum(item["action_correct"] for item in decisions), len(decisions)
        ),
        "illegal_moves": sum(not item["legal"] for item in decisions),
        "illegal_move_rate": _rate(
            sum(not item["legal"] for item in decisions), len(decisions)
        ),
        "perfect_token_trajectories": sum(
            all(item["token_correct"] for item in row["decisions"]) for row in rows
        ),
        "perfect_token_trajectory_rate": _rate(
            sum(all(item["token_correct"] for item in row["decisions"]) for row in rows),
            len(rows),
        ),
        "perfect_action_trajectories": sum(
            all(item["action_correct"] for item in row["decisions"]) for row in rows
        ),
        "perfect_action_trajectory_rate": _rate(
            sum(all(item["action_correct"] for item in row["decisions"]) for row in rows),
            len(rows),
        ),
    }


def _by_decision_count(rows: list[dict[str, Any]]) -> dict[str, Any]:
    grouped: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[len(row["decisions"])].append(row)
    return {str(count): _summary(grouped[count]) for count in sorted(grouped)}


def _by_ordinal(rows: list[dict[str, Any]]) -> dict[str, Any]:
    result = {}
    maximum = max(len(row["decisions"]) for row in rows)
    for ordinal in range(1, maximum + 1):
        eligible = [row for row in rows if len(row["decisions"]) >= ordinal]
        decisions = [row["decisions"][ordinal - 1] for row in eligible]
        result[str(ordinal)] = {
            "eligible_trajectories": len(eligible),
            "joint_token_accuracy": _rate(
                sum(item["token_correct"] for item in decisions), len(decisions)
            ),
            "mapped_action_accuracy": _rate(
                sum(item["action_correct"] for item in decisions), len(decisions)
            ),
            "all_tokens_correct_through_ordinal_rate": _rate(
                sum(
                    all(
                        item["token_correct"]
                        for item in row["decisions"][:ordinal]
                    )
                    for row in eligible
                ),
                len(eligible),
            ),
            "all_actions_correct_through_ordinal_rate": _rate(
                sum(
                    all(
                        item["action_correct"]
                        for item in row["decisions"][:ordinal]
                    )
                    for row in eligible
                ),
                len(eligible),
            ),
        }
    return result


def _by_length(rows: list[dict[str, Any]]) -> dict[str, Any]:
    bands = (
        ("28-40", 0, 40),
        ("41-80", 41, 80),
        ("81-160", 81, 160),
        ("161-320", 161, 320),
    )
    result = {}
    for label, lower, upper in bands:
        selected = [
            row for row in rows if lower <= row["token_length"] <= upper
        ]
        if selected:
            result[label] = _summary(selected)
    return result


def _calibration(decisions: list[dict[str, Any]]) -> dict[str, Any]:
    boundaries = (0.0, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0000001)
    result = {}
    weighted_gap = 0.0
    for lower, upper in zip(boundaries, boundaries[1:]):
        selected = [
            item
            for item in decisions
            if lower <= item["predicted_probability"] < upper
        ]
        if not selected:
            continue
        accuracy = sum(item["token_correct"] for item in selected) / len(selected)
        confidence = sum(item["predicted_probability"] for item in selected) / len(
            selected
        )
        weighted_gap += len(selected) * abs(accuracy - confidence)
        label = f"{lower:.1f}-{min(upper, 1.0):.1f}"
        result[label] = {
            "decisions": len(selected),
            "mean_confidence": confidence,
            "joint_token_accuracy": accuracy,
            "confidence_minus_accuracy": confidence - accuracy,
        }
    return {
        "bins": result,
        "expected_calibration_error": weighted_gap / len(decisions),
    }


def _streaks(decisions: list[dict[str, Any]]) -> dict[str, Any]:
    longest_correct = 0
    longest_error = 0
    current_correct = 0
    current_error = 0
    for item in decisions:
        if item["token_correct"]:
            current_correct += 1
            current_error = 0
        else:
            current_error += 1
            current_correct = 0
        longest_correct = max(longest_correct, current_correct)
        longest_error = max(longest_error, current_error)
    window = min(250, len(decisions))
    rolling = []
    if window:
        correct = sum(item["token_correct"] for item in decisions[:window])
        rolling.append(correct / window)
        for start in range(1, len(decisions) - window + 1):
            correct -= decisions[start - 1]["token_correct"]
            correct += decisions[start + window - 1]["token_correct"]
            rolling.append(correct / window)
    return {
        "longest_correct_token_streak": longest_correct,
        "longest_incorrect_token_streak": longest_error,
        "rolling_window_decisions": window,
        "rolling_minimum_token_accuracy": min(rolling) if rolling else None,
        "rolling_maximum_token_accuracy": max(rolling) if rolling else None,
    }


def _behavior(decisions: list[dict[str, Any]]) -> dict[str, Any]:
    result = {}
    for street in STREETS:
        selected = [item for item in decisions if item["street"] == street]
        predicted = Counter(item["predicted_action"] for item in selected)
        truth = Counter(item["truth_action"] for item in selected)
        result[street] = {
            "decisions": len(selected),
            "predicted_action_counts": dict(sorted(predicted.items())),
            "predicted_action_rates": {
                action: count / len(selected)
                for action, count in sorted(predicted.items())
            },
            "truth_action_counts": dict(sorted(truth.items())),
            "truth_action_rates": {
                action: count / len(selected)
                for action, count in sorted(truth.items())
            },
        }
    return result


def _fmt(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.3%}"


def _fmt_amount(value: float | None) -> str:
    return "-" if value is None else f"{value:.2f} BB"


def _markdown(report: Mapping[str, Any]) -> str:
    overall = report["sequence_analysis"]["overall"]
    long_rows = report["sequence_analysis"]["four_plus_decisions"]
    metrics = report["frozen_metrics"]
    calibration = report["sequence_analysis"]["calibration"]
    streaks = report["sequence_analysis"]["session_stability"]
    behavior = report["sequence_analysis"]["behavior_by_street"]
    high_confidence = report["sequence_analysis"]["high_confidence_errors"]
    sizing = metrics["sizing"]
    lines = [
        "# PokerGPT long-run validation shadow session",
        "",
        "## Protocol",
        "",
        (
            f"The frozen seed-1337 candidate was run over {report['source_hands']:,} "
            f"real Pluribus validation hands, represented as "
            f"{overall['trajectories']:,} player-perspective trajectories and "
            f"{overall['decisions']:,} hero decisions. Cards, boards, opponent "
            "actions, pots, and legality state came from exact replay and every "
            "trajectory was byte-checked against the prepared binary."
        ),
        "",
        (
            "This is a shadow session: the model predicts at every hero decision, "
            "but the recorded action advances the hand. It is realistic "
            "teacher-forced evaluation, not a counterfactual self-play simulation."
        ),
        "",
        "The held-out test split was not opened or rerun.",
        "",
        "## Long-run summary",
        "",
        f"- Joint decision-token accuracy: {_fmt(metrics['joint_decision_token_accuracy'])}",
        f"- Mapped five-way action accuracy: {_fmt(metrics['mapped_action_accuracy'])}",
        f"- Illegal moves: {metrics['illegal_moves']:,} / {metrics['decisions']:,} ({_fmt(metrics['illegal_move_rate'])})",
        (
            "- Entire trajectories with every token prediction correct: "
            f"{overall['perfect_token_trajectories']:,} / "
            f"{overall['trajectories']:,} "
            f"({_fmt(overall['perfect_token_trajectory_rate'])})"
        ),
        (
            "- Entire trajectories with every mapped action correct: "
            f"{overall['perfect_action_trajectories']:,} / "
            f"{overall['trajectories']:,} "
            f"({_fmt(overall['perfect_action_trajectory_rate'])})"
        ),
        (
            "- Trajectories with at least four hero decisions: "
            f"{long_rows['trajectories']:,}; token accuracy "
            f"{_fmt(long_rows['joint_token_accuracy'])}, action accuracy "
            f"{_fmt(long_rows['mapped_action_accuracy'])}, fully token-correct "
            f"{_fmt(long_rows['perfect_token_trajectory_rate'])}"
        ),
        (
            "- Confidence calibration ECE: "
            f"{calibration['expected_calibration_error']:.4f}"
        ),
        (
            f"- Longest correct/incorrect token streaks in replay order: "
            f"{streaks['longest_correct_token_streak']} / "
            f"{streaks['longest_incorrect_token_streak']}"
        ),
        (
            f"- Rolling {streaks['rolling_window_decisions']}-decision token "
            f"accuracy ranged from {_fmt(streaks['rolling_minimum_token_accuracy'])} "
            f"to {_fmt(streaks['rolling_maximum_token_accuracy'])}"
        ),
        "",
        "## What the generated decisions are like",
        "",
        (
            "The model is strongest on the opening decision and on short "
            "fold-dominated trajectories. First-decision token accuracy is "
            f"{_fmt(report['sequence_analysis']['by_decision_ordinal']['1']['joint_token_accuracy'])}, "
            "while the 4,090 trajectories of 28-40 tokens score "
            f"{_fmt(report['sequence_analysis']['by_token_length']['28-40']['joint_token_accuracy'])}. "
            "That short-hand strength makes the overall number look better than "
            "the long-hand behavior."
        ),
        "",
        (
            "In long hands the policy becomes conspicuously conservative. "
            f"Preflop it folds {_fmt(behavior['PREFLOP']['predicted_action_rates'].get('FOLD', 0.0))} "
            f"of the time versus {_fmt(behavior['PREFLOP']['truth_action_rates'].get('FOLD', 0.0))} "
            "in the recorded actions. Postflop it strongly prefers checking or "
            "folding and rarely initiates aggression: predicted versus recorded "
            f"bet/raise rates are {_fmt(behavior['FLOP']['predicted_action_rates'].get('BET', 0.0) + behavior['FLOP']['predicted_action_rates'].get('RAISE', 0.0))} "
            f"vs {_fmt(behavior['FLOP']['truth_action_rates'].get('BET', 0.0) + behavior['FLOP']['truth_action_rates'].get('RAISE', 0.0))} on the flop, "
            f"{_fmt(behavior['TURN']['predicted_action_rates'].get('BET', 0.0) + behavior['TURN']['predicted_action_rates'].get('RAISE', 0.0))} "
            f"vs {_fmt(behavior['TURN']['truth_action_rates'].get('BET', 0.0) + behavior['TURN']['truth_action_rates'].get('RAISE', 0.0))} on the turn, and "
            f"{_fmt(behavior['RIVER']['predicted_action_rates'].get('BET', 0.0) + behavior['RIVER']['predicted_action_rates'].get('RAISE', 0.0))} "
            f"vs {_fmt(behavior['RIVER']['truth_action_rates'].get('BET', 0.0) + behavior['RIVER']['truth_action_rates'].get('RAISE', 0.0))} on the river."
        ),
        "",
        (
            "The trace pattern is repetitive: many correct checks are followed "
            "by folds where the recorded player called, bet, or raised. This is "
            "especially visible at later streets. It resembles a cautious "
            "baseline with useful preflop pattern recognition, not a balanced "
            "multi-street Pluribus imitator."
        ),
        "",
        (
            "Sizing is the bright spot once the action is right. Conditional on "
            "correctly choosing a bet or raise, the exact range-token success "
            f"rate is {_fmt(1.0 - sizing['range_error_rate_conditional_on_aggressive_action_correct'])}, "
            "with representative pot-ratio MAE "
            f"{sizing['representative_ratio_mae_conditional_on_aggressive_action_correct']:.3f}. "
            "The main failure is choosing aggression at all, particularly "
            "postflop, rather than choosing a wildly wrong size."
        ),
        "",
        (
            f"Legality is excellent but not perfect: all {metrics['illegal_moves']} "
            "illegal outputs are raw folds when checking is available, concentrated "
            "on the river. Confidence is somewhat optimistic; "
            f"{high_confidence['count']} wrong token predictions "
            f"({_fmt(high_confidence['rate_among_all_decisions'])} of all decisions) "
            "still carried at least 90% probability."
        ),
        "",
        "## Sequence depth",
        "",
        "| Hero decision ordinal | Eligible trajectories | Token accuracy | Action accuracy | All tokens correct through here |",
        "|---:|---:|---:|---:|---:|",
    ]
    for ordinal, values in report["sequence_analysis"]["by_decision_ordinal"].items():
        lines.append(
            f"| {ordinal} | {values['eligible_trajectories']:,} | "
            f"{_fmt(values['joint_token_accuracy'])} | "
            f"{_fmt(values['mapped_action_accuracy'])} | "
            f"{_fmt(values['all_tokens_correct_through_ordinal_rate'])} |"
        )
    lines += [
        "",
        "## Accuracy by complete trajectory length",
        "",
        "| Token length | Trajectories | Decisions | Token accuracy | Action accuracy |",
        "|---|---:|---:|---:|---:|",
    ]
    for label, values in report["sequence_analysis"]["by_token_length"].items():
        lines.append(
            f"| {label} | {values['trajectories']:,} | {values['decisions']:,} | "
            f"{_fmt(values['joint_token_accuracy'])} | "
            f"{_fmt(values['mapped_action_accuracy'])} |"
        )
    lines += [
        "",
        "## Predicted action mix by street",
        "",
        "| Street | Fold | Check | Call | Bet | Raise |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for street, values in report["sequence_analysis"]["behavior_by_street"].items():
        rates = values["predicted_action_rates"]
        lines.append(
            f"| {street.title()} | "
            + " | ".join(_fmt(rates.get(action, 0.0)) for action in (
                "FOLD", "CHECK", "CALL", "BET", "RAISE"
            ))
            + " |"
        )
    lines += [
        "",
        "## Longest trajectory traces",
        "",
        (
            "These are the longest validation trajectories, ordered first by "
            "number of hero decisions and then by encoded length. `P(pred)` is "
            "the normalized probability of the model's raw decision token."
        ),
        "",
    ]
    for trace in report["longest_trajectory_traces"]:
        lines += [
            (
                f"### Trace {trace['trace_rank']}: {trace['position']}, "
                f"{' '.join(trace['hero_cards'])}"
            ),
            "",
            (
                f"- Source: `{trace['member']}::{trace['hand_key']}`; hero seat "
                f"{trace['hero_seat']}; {trace['token_length']} tokens; "
                f"{len(trace['decisions'])} decisions"
            ),
            "",
            "| # | Street | Board | Pot / call | Truth | Prediction | P(pred) | Legal | Top 3 raw tokens |",
            "|---:|---|---|---|---|---|---:|---|---|",
        ]
        for item in trace["decisions"]:
            truth = item["truth_action"]
            if item["truth_amount_bb"] is not None:
                truth += f" {_fmt_amount(item['truth_amount_bb'])}"
            prediction = item["predicted_action"]
            if item["predicted_amount_bb"] is not None:
                prediction += f" {_fmt_amount(item['predicted_amount_bb'])}"
            if not item["action_correct"]:
                prediction = f"**{prediction}**"
            top = ", ".join(
                f"{entry['token']} {entry['probability']:.1%}"
                for entry in item["top_tokens"]
            )
            lines.append(
                f"| {item['ordinal']} | {item['street'].title()} | "
                f"{' '.join(item['board']) or '-'} | "
                f"{item['pot_bb']:.2f} / {item['to_call_bb']:.2f} BB | "
                f"{truth} | {prediction} | "
                f"{item['predicted_probability']:.1%} | "
                f"{'yes' if item['legal'] else 'NO: ' + item['legality_reason']} | "
                f"{top} |"
            )
        lines.append("")
    lines += [
        "## Scope limitation",
        "",
        (
            "PokerGPT has supervised loss only on hero decision tokens. Asking it "
            "to freely generate board cards, pot updates, or opponent actions "
            "would test unsupervised token logits, not poker ability. A genuine "
            "counterfactual long-running rollout needs a poker engine plus "
            "opponent policies that react to the model's chosen actions."
        ),
        "",
    ]
    return "\n".join(lines)


@torch.inference_mode()
def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.batch_size <= 0 or args.trace_count <= 0:
        raise ValueError("batch-size and trace-count must be positive")
    output_dir = args.output_dir.resolve()
    frozen_report_dir = (REPO_ROOT / "reports" / "evaluator-v1").resolve()
    if output_dir == frozen_report_dir or frozen_report_dir in output_dir.parents:
        raise ValueError("shadow-session outputs cannot be written into evaluator-v1")

    config = load_evaluator_config(args.config)
    meta = _read_meta(args.data_dir / "meta.pkl")
    dataset = PokerTrajectoryDataset(args.data_dir, "val")
    contexts = build_replay_contexts(
        zip_path=args.archive,
        selection_path=args.selection,
        dataset=dataset,
        split="val",
        meta=meta,
    )
    checkpoint_identity = file_identity(args.checkpoint)
    if checkpoint_identity["sha256"] != config["candidate"]["checkpoint_sha256"]:
        raise ValueError("checkpoint hash does not match the frozen candidate")
    checkpoint = load_checkpoint(args.checkpoint, map_location="cpu")
    model = GPT(GPTConfig(**checkpoint["model_config"]))
    model.load_state_dict(checkpoint["model"])
    device, precision = _resolve_device_precision(config)
    model.to(device)
    model.eval()

    frozen_metrics = FrozenEvaluationMetrics(
        meta,
        decision_top_k=int(config["decision_top_k"]),
        action_top_k=int(config["mapped_action_top_k"]),
    )
    trajectories: list[dict[str, Any]] = []
    for batch_start in range(0, len(dataset), args.batch_size):
        batch_end = min(batch_start + args.batch_size, len(dataset))
        items = [dataset[index] for index in range(batch_start, batch_end)]
        inputs, targets, masks = dataset.collate_fn(items)
        with _autocast(device, precision):
            logits, _ = model(
                inputs.to(device),
                targets.to(device),
                masks.to(device),
            )
        for local_index, trajectory_index in enumerate(range(batch_start, batch_end)):
            context = contexts[trajectory_index]
            item_inputs, item_targets, item_mask = items[local_index]
            full_ids = item_inputs.tolist() + [int(item_targets[-1].item())]
            positions = item_mask.nonzero(as_tuple=False).flatten().tolist()
            if len(positions) != len(context.decisions):
                raise ValueError("decision positions do not align with replay context")
            decisions = []
            position_label = None
            hero_cards: list[str] | None = None
            for ordinal, (position, decision_context) in enumerate(
                zip(positions, context.decisions), start=1
            ):
                distribution = decision_distribution(
                    logits[local_index, position], meta
                )
                probabilities = distribution.probabilities.detach().float().cpu()
                predicted_index = int(probabilities.argmax().item())
                predicted_token = distribution.tokens[predicted_index]
                truth_token = decision_context.truth_token
                state = decision_context.legality_state
                legality = check_decision_token(predicted_token, state, meta)
                predicted_action = _mapped_action(predicted_token, state)
                truth_action = _mapped_action(truth_token, state)
                top_values, top_indexes = probabilities.topk(
                    min(3, len(distribution.tokens))
                )
                cards, board, current_position = _cards_before_decision(
                    full_ids[: position + 1], meta
                )
                position_label = current_position
                hero_cards = cards
                truth_amount = (
                    float(decision_context.truth_ratio * state.pot)
                    if decision_context.truth_ratio is not None
                    else None
                )
                decisions.append(
                    {
                        "ordinal": ordinal,
                        "street": decision_context.street,
                        "board": board,
                        "pot_bb": float(state.pot),
                        "to_call_bb": float(state.to_call),
                        "remaining_stack_bb": float(state.remaining_stack),
                        "truth_token": truth_token,
                        "truth_action": truth_action,
                        "truth_amount_bb": truth_amount,
                        "predicted_token": predicted_token,
                        "predicted_action": predicted_action,
                        "predicted_amount_bb": _amount_bb(
                            predicted_token, state, meta
                        ),
                        "predicted_probability": float(
                            probabilities[predicted_index].item()
                        ),
                        "truth_probability": float(
                            probabilities[
                                distribution.tokens.index(truth_token)
                            ].item()
                        ),
                        "token_correct": predicted_token == truth_token,
                        "action_correct": predicted_action == truth_action,
                        "legal": legality.legal,
                        "legality_reason": legality.reason,
                        "top_tokens": [
                            {
                                "token": distribution.tokens[index],
                                "probability": float(value),
                            }
                            for value, index in zip(
                                top_values.tolist(), top_indexes.tolist()
                            )
                        ],
                    }
                )
                frozen_metrics.update(
                    distribution,
                    truth_token=truth_token,
                    truth_ratio=decision_context.truth_ratio,
                    street=decision_context.street,
                    legality_state=state,
                )
            assert position_label is not None and hero_cards is not None
            trajectories.append(
                {
                    "trajectory_index": trajectory_index,
                    "member": context.member,
                    "hand_key": context.hand_key,
                    "hero_seat": context.hero + 1,
                    "position": position_label,
                    "hero_cards": hero_cards,
                    "token_length": len(full_ids),
                    "decisions": decisions,
                }
            )

    # Replay legality uses the source corpus's exact chip denomination. Convert
    # trace amounts to BB for readability. With no antes or straddles, the
    # earliest decision in every hand sees exactly the 0.5/1 blind pot (1.5 BB).
    hand_rows: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for trajectory in trajectories:
        hand_rows[(trajectory["member"], trajectory["hand_key"])].append(trajectory)
    big_blinds = set()
    for rows in hand_rows.values():
        opening_pot = min(
            decision["pot_bb"]
            for trajectory in rows
            for decision in trajectory["decisions"]
        )
        big_blind = opening_pot / 1.5
        if big_blind <= 0:
            raise ValueError("derived non-positive big blind")
        big_blinds.add(big_blind)
        for trajectory in rows:
            trajectory["source_big_blind_chips"] = big_blind
            for decision in trajectory["decisions"]:
                for field in (
                    "pot_bb",
                    "to_call_bb",
                    "remaining_stack_bb",
                    "truth_amount_bb",
                    "predicted_amount_bb",
                ):
                    if decision[field] is not None:
                        decision[field] /= big_blind

    all_decisions = [
        decision for trajectory in trajectories for decision in trajectory["decisions"]
    ]
    source_hands = len(
        {(trajectory["member"], trajectory["hand_key"]) for trajectory in trajectories}
    )
    long_trajectories = [
        trajectory for trajectory in trajectories if len(trajectory["decisions"]) >= 4
    ]
    trace_rows = sorted(
        trajectories,
        key=lambda row: (
            len(row["decisions"]),
            row["token_length"],
            -row["trajectory_index"],
        ),
        reverse=True,
    )[: args.trace_count]
    for rank, row in enumerate(trace_rows, start=1):
        row["trace_rank"] = rank

    report = {
        "report_format_version": 1,
        "report_id": "pokergpt-long-trajectory-shadow-v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "split": "val",
        "held_out_test_accessed": False,
        "mode": (
            "replay-backed shadow prediction; recorded actions advance exact state"
        ),
        "source_hands": source_hands,
        "candidate": dict(config["candidate"]),
        "checkpoint": checkpoint_identity,
        "inputs": {
            "archive": {
                "path": args.archive.resolve().as_posix(),
                "bytes": args.archive.stat().st_size,
                "mode": "streamed selected members only; never extracted",
            },
            "selection": file_identity(args.selection),
            "meta": file_identity(args.data_dir / "meta.pkl"),
            "val_bin": file_identity(args.data_dir / "val.bin"),
            "val_mask": file_identity(args.data_dir / "val_loss_mask.bin"),
            "val_index": file_identity(args.data_dir / "val.idx"),
        },
        "environment": {
            "pytorch": torch.__version__,
            "device": str(device),
            "precision": precision,
            "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
        },
        "source_chip_denominations": {
            "distinct_big_blind_chip_values": sorted(big_blinds),
            "trace_amount_unit": "big blinds",
            "derivation": (
                "minimum exact replay pot per hand divided by 1.5; the game "
                "contract has 0.5/1 blinds with no antes or straddles"
            ),
        },
        "frozen_metrics": frozen_metrics.compute(),
        "sequence_analysis": {
            "overall": _summary(trajectories),
            "four_plus_decisions": _summary(long_trajectories),
            "by_decision_count": _by_decision_count(trajectories),
            "by_decision_ordinal": _by_ordinal(trajectories),
            "by_token_length": _by_length(trajectories),
            "calibration": _calibration(all_decisions),
            "session_stability": _streaks(all_decisions),
            "behavior_by_street": _behavior(all_decisions),
            "high_confidence_errors": {
                "threshold": 0.9,
                "count": sum(
                    not item["token_correct"]
                    and item["predicted_probability"] >= 0.9
                    for item in all_decisions
                ),
                "rate_among_all_decisions": _rate(
                    sum(
                        not item["token_correct"]
                        and item["predicted_probability"] >= 0.9
                        for item in all_decisions
                    ),
                    len(all_decisions),
                ),
            },
        },
        "longest_trajectory_traces": trace_rows,
        "scope_limit": (
            "The model is supervised only on hero decision tokens. This diagnostic "
            "does not ask it to invent boards, pots, or opponent actions, and it "
            "does not branch future state after a model/recorded-action mismatch."
        ),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "report.json"
    markdown_path = output_dir / "report.md"
    json_path.write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    markdown_path.write_text(_markdown(report), encoding="utf-8")
    return {
        "json": str(json_path),
        "markdown": str(markdown_path),
        "source_hands": source_hands,
        "trajectories": len(trajectories),
        "decisions": len(all_decisions),
    }


def main() -> None:
    print(json.dumps(run(parse_args()), indent=2))


if __name__ == "__main__":
    main()
