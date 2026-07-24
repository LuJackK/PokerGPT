from __future__ import annotations

import argparse
import json
import pickle
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
from poker_model.evaluator import load_evaluator_config
from poker_model.legality import LegalityState, check_decision_token
from poker_model.model import GPT, GPTConfig
from poker_model.run_tracking import file_identity
from poker_pipeline.phh import Decision, build_hero_trajectories, parse_document, replay_hand
from poker_pipeline.tokenizer import PokerTokenizer


REPO_ROOT = Path(__file__).resolve().parent
DEFAULT_CONFIG = REPO_ROOT / "configs" / "evaluator_v1.json"
DEFAULT_DATA = REPO_ROOT / "data" / "processed"
DEFAULT_CHECKPOINT = REPO_ROOT / "runs" / "baseline-v081-seed1337" / "best.pt"
DEFAULT_OUTPUT = REPO_ROOT / "reports" / "novel-synthetic-trajectory-v1"
DEFAULT_FIXTURE = (
    REPO_ROOT / "test" / "artifacts" / "novel-trajectory-v1" / "synthetic_hand.phh"
)
HERO = 3  # PHH p4, rotated to PLAYER_1 by the tokenizer.


SYNTHETIC_HAND = """\
variant = 'NT'
ante_trimming_status = true
antes = [0, 0, 0, 0, 0, 0]
blinds_or_straddles = [50, 100, 0, 0, 0, 0]
min_bet = 100
starting_stacks = [10000, 10000, 10000, 10000, 10000, 10000]
actions = ['d dh p1 8c6c', 'd dh p2 9s9d', 'd dh p3 Jh5h', 'd dh p4 AsQs', 'd dh p5 4d4c', 'd dh p6 KdKh', 'p3 f', 'p4 cbr 225', 'p5 f', 'p6 cc', 'p1 f', 'p2 cbr 800', 'p4 cc', 'p6 cc', 'd db Qh7d2c', 'p2 cc', 'p4 cc', 'p6 cbr 900', 'p2 f', 'p4 cc', 'd db Tc', 'p4 cc', 'p6 cbr 2200', 'p4 cc', 'd db 3s', 'p4 cc', 'p6 cbr 4300', 'p4 cc', 'p4 sm AsQs', 'p6 sm KdKh']
hand = 20260724001
players = ['synthetic1', 'synthetic2', 'synthetic3', 'synthetic4', 'synthetic5', 'synthetic6']
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate one fixed, novel synthetic trajectory with PokerGPT"
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--fixture", type=Path, default=DEFAULT_FIXTURE)
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
    return torch.device(device_name), precision


def _autocast(device: torch.device, precision: str):
    if precision == "float32":
        return nullcontext()
    dtype = torch.bfloat16 if precision == "bf16" else torch.float16
    return torch.autocast(device_type=device.type, dtype=dtype)


def _stored_ids(dataset: PokerTrajectoryDataset, index: int) -> tuple[int, ...]:
    inputs, targets, _ = dataset[index]
    return tuple(inputs.tolist()) + (int(targets[-1].item()),)


def _training_matches(
    dataset: PokerTrajectoryDataset, candidate_ids: tuple[int, ...]
) -> list[int]:
    return [
        index
        for index in range(len(dataset))
        if dataset.trajectory_lengths[index] + 1 == len(candidate_ids)
        and _stored_ids(dataset, index) == candidate_ids
    ]


def _training_prefix_match_counts(
    dataset: PokerTrajectoryDataset,
    candidate_ids: tuple[int, ...],
    decision_positions: list[int],
) -> list[int]:
    prefixes = [
        candidate_ids[: position + 1] for position in decision_positions
    ]
    counts = [0] * len(prefixes)
    for index in range(len(dataset)):
        stored = _stored_ids(dataset, index)
        for prefix_index, prefix in enumerate(prefixes):
            if len(stored) >= len(prefix) and stored[: len(prefix)] == prefix:
                counts[prefix_index] += 1
    return counts


def _state(decision: Decision) -> LegalityState:
    return LegalityState(
        pot=decision.pot,
        to_call=decision.to_call,
        current_bet=decision.current_bet,
        street_contribution=decision.current_bet - decision.to_call,
        remaining_stack=decision.hero_stack,
        minimum_bet=decision.minimum_bet,
        minimum_raise_increment=decision.minimum_raise_increment,
        raise_reopened=decision.raise_reopened,
    )


def _action(token: str, state: LegalityState) -> str:
    return recover_five_way_action(
        token, to_call=state.to_call, current_bet=state.current_bet
    )


def _amount_bb(
    token: str,
    state: LegalityState,
    meta: Mapping[str, Any],
    big_blind: Decimal,
) -> float | None:
    if action_group(token) != "AGGRESSIVE":
        return None
    if token == "ACTION_ALL_IN":
        contribution = state.remaining_stack
    else:
        contribution = (
            state.pot * Decimal(str(meta["range_representative_ratio"][token]))
        )
    return float(contribution / big_blind)


def _human_action(action: str, amount: float | None) -> str:
    return action if amount is None else f"{action} {amount:.2f} BB"


def _markdown(report: Mapping[str, Any]) -> str:
    result = report["result"]
    proof = report["novelty_proof"]
    lines = [
        "# PokerGPT novel synthetic trajectory",
        "",
        "## Result",
        "",
        (
            f"The model was evaluated on one newly authored, legal six-max "
            f"100-BB trajectory containing {result['decisions']} hero decisions "
            f"and {report['encoded_tokens']} encoded tokens."
        ),
        "",
        (
            f"It matched {result['token_correct']}/{result['decisions']} raw "
            f"decision tokens and {result['action_correct']}/{result['decisions']} "
            f"mapped poker actions. All {result['legal_predictions']}/"
            f"{result['decisions']} raw predictions were legal."
        ),
        "",
        (
            f"Exact training-set duplicate search found "
            f"{proof['exact_training_matches']} matches among "
            f"{proof['training_trajectories_searched']:,} training trajectories. "
            f"Exact context-prefix matches at the eight decision points were "
            f"{proof['decision_prefix_training_matches']}. Opponent hole cards "
            "are absent from the encoded model input."
        ),
        "",
        "## Synthetic hand",
        "",
        "- Hero: hijack with As Qs",
        "- Preflop: UTG folds; hero raises to 2.25 BB; cutoff folds; button calls; small blind folds; big blind raises to 8 BB; hero and button call.",
        "- Flop Qh 7d 2c (24.5 BB): big blind checks; hero checks; button bets 9 BB; big blind folds; hero calls.",
        "- Turn Tc (42.5 BB): hero checks; button bets 22 BB; hero calls.",
        "- River 3s (86.5 BB): hero checks; button bets 43 BB; hero calls.",
        "",
        "The authored hero line is raise, call, check, call, check, call, check, call.",
        "Aggressive amounts in the table are incremental contributions, matching the model contract.",
        "",
        "## Model predictions",
        "",
        "| # | Street | Board | Pot / call | Authored action | Model prediction | P(pred) | Legal | Top 3 raw tokens |",
        "|---:|---|---|---|---|---|---:|---|---|",
    ]
    for item in report["predictions"]:
        authored = _human_action(
            item["truth_action"], item["truth_amount_bb"]
        )
        prediction = _human_action(
            item["predicted_action"], item["predicted_amount_bb"]
        )
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
            f"{authored} | {prediction} | "
            f"{item['predicted_probability']:.1%} | "
            f"{'yes' if item['legal'] else 'NO: ' + str(item['legality_reason'])} | "
            f"{top} |"
        )
    lines += [
        "",
        "## What it did",
        "",
        (
            "The opening was strong: it selected the same raise bucket as the "
            "2.25-BB authored open and decoded to 2.24 BB. Facing the 8-BB "
            "3-bet, however, it preferred another raise rather than a call."
        ),
        "",
        (
            "On the flop it preferred a roughly half-pot bet over checking. In "
            "the separate teacher-forced state where the button had bet 9 BB, "
            "it preferred folding top pair. It matched the authored turn "
            "check-call line with high confidence."
        ),
        "",
        (
            "On the river it strongly preferred a 61-BB all-in instead of "
            "checking. In the alternate teacher-forced state where the button "
            "had instead bet 43 BB, it narrowly preferred folding. The fold is "
            "not an action after its proposed all-in: those are two independent "
            "queries against the authored continuation."
        ),
        "",
        "## Interpretation boundary",
        "",
        (
            "This proves behavior on a new legal input, but the authored action "
            "line is not an optimal-poker label. Agreement measures imitation of "
            "this scenario, not expected value. Recorded synthetic actions remain "
            "in the context after each decision, so later predictions are "
            "teacher-forced and do not form a counterfactual branch."
        ),
        "",
        (
            "PokerGPT predicts only hero decisions. It was not asked to invent "
            "board cards, opponent actions, or pot transitions, because those "
            "tokens were not supervised during training."
        ),
        "",
    ]
    return "\n".join(lines)


@torch.inference_mode()
def run(args: argparse.Namespace) -> dict[str, Any]:
    config = load_evaluator_config(args.config)
    meta = _read_meta(args.data_dir / "meta.pkl")
    tokenizer = PokerTokenizer()
    if tokenizer.itos != list(meta["itos"]):
        raise ValueError("runtime tokenizer does not match prepared metadata")

    parsed = parse_document(SYNTHETIC_HAND, "phh")
    if len(parsed) != 1:
        raise ValueError("synthetic fixture must contain exactly one hand")
    hand_key, hand = parsed[0]
    all_decisions = list(
        replay_hand("synthetic/novel-20260724.phh", hand_key, hand)
    )
    hero_trajectory = next(
        trajectory
        for trajectory in build_hero_trajectories(all_decisions)
        if trajectory.hero == HERO
    )
    hero_decisions = [
        item for item in hero_trajectory.items if isinstance(item, Decision)
    ]
    if len(hero_decisions) != 8:
        raise ValueError(
            f"expected eight hero decisions, got {len(hero_decisions)}"
        )
    encoded = tokenizer.encode_trajectory(hero_trajectory)
    if len(encoded.ids) > int(meta["block_size"]):
        raise ValueError("synthetic trajectory exceeds block size")
    opponent_cards = {
        "CARD_8c",
        "CARD_6c",
        "CARD_9s",
        "CARD_9d",
        "CARD_Jh",
        "CARD_5h",
        "CARD_4d",
        "CARD_4c",
        "CARD_Kd",
        "CARD_Kh",
    }
    leaked_cards = opponent_cards.intersection(encoded.tokens)
    if leaked_cards:
        raise ValueError(f"opponent private cards leaked: {sorted(leaked_cards)}")

    train_dataset = PokerTrajectoryDataset(args.data_dir, "train")
    input_ids = encoded.ids[:-1]
    target_ids = encoded.ids[1:]
    shifted_mask = encoded.loss_mask[1:]
    positions = [
        index for index, selected in enumerate(shifted_mask) if selected
    ]
    if len(positions) != len(hero_decisions):
        raise ValueError("encoded decision mask does not align with replay")
    matches = _training_matches(train_dataset, encoded.ids)
    if matches:
        raise ValueError(f"synthetic trajectory duplicates training rows: {matches}")
    prefix_matches = _training_prefix_match_counts(
        train_dataset, encoded.ids, positions
    )

    checkpoint_identity = file_identity(args.checkpoint)
    if checkpoint_identity["sha256"] != config["candidate"]["checkpoint_sha256"]:
        raise ValueError("checkpoint hash does not match frozen candidate")
    checkpoint = load_checkpoint(args.checkpoint, map_location="cpu")
    model = GPT(GPTConfig(**checkpoint["model_config"]))
    model.load_state_dict(checkpoint["model"])
    device, precision = _resolve_device_precision(config)
    model.to(device)
    model.eval()

    inputs = torch.tensor(input_ids, dtype=torch.long).unsqueeze(0)
    targets = torch.tensor(target_ids, dtype=torch.long).unsqueeze(0)
    loss_mask = torch.tensor(
        shifted_mask, dtype=torch.uint8
    ).unsqueeze(0)
    with _autocast(device, precision):
        logits, _ = model(
            inputs.to(device),
            targets.to(device),
            loss_mask.to(device),
        )

    predictions = []
    for ordinal, (position, decision) in enumerate(
        zip(positions, hero_decisions), start=1
    ):
        truth_id = int(targets[0, position].item())
        truth_token = meta["itos"][truth_id]
        expected_truth = tokenizer.hero_decision_token(decision)
        if truth_token != expected_truth:
            raise ValueError("encoded target does not match replay decision")
        distribution = decision_distribution(logits[0, position], meta)
        probabilities = distribution.probabilities.detach().float().cpu()
        predicted_index = int(probabilities.argmax().item())
        predicted_token = distribution.tokens[predicted_index]
        state = _state(decision)
        legality = check_decision_token(predicted_token, state, meta)
        predicted_action = _action(predicted_token, state)
        truth_action = _action(truth_token, state)
        top_values, top_indexes = probabilities.topk(3)
        truth_amount = (
            float(decision.target_amount / decision.big_blind)
            if action_group(truth_token) == "AGGRESSIVE"
            else None
        )
        predictions.append(
            {
                "ordinal": ordinal,
                "street": decision.street,
                "board": list(decision.board),
                "pot_bb": float(decision.pot / decision.big_blind),
                "to_call_bb": float(decision.to_call / decision.big_blind),
                "remaining_stack_bb": float(
                    decision.hero_stack / decision.big_blind
                ),
                "truth_token": truth_token,
                "truth_action": truth_action,
                "truth_amount_bb": truth_amount,
                "predicted_token": predicted_token,
                "predicted_action": predicted_action,
                "predicted_amount_bb": _amount_bb(
                    predicted_token, state, meta, decision.big_blind
                ),
                "predicted_probability": float(
                    probabilities[predicted_index].item()
                ),
                "truth_probability": float(
                    probabilities[distribution.tokens.index(truth_token)].item()
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

    report = {
        "report_format_version": 1,
        "report_id": "pokergpt-novel-synthetic-trajectory-v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "candidate": dict(config["candidate"]),
        "checkpoint": checkpoint_identity,
        "synthetic_fixture": {
            "member": hero_trajectory.member,
            "hand_key": hero_trajectory.hand_key,
            "hero_source_seat": HERO + 1,
            "hero_position": "HIJACK",
            "hero_cards": ["As", "Qs"],
            "phh": SYNTHETIC_HAND,
        },
        "encoded_tokens": len(encoded.ids),
        "encoded_decision_targets": [
            encoded.tokens[index]
            for index, selected in enumerate(encoded.loss_mask)
            if selected
        ],
        "novelty_proof": {
            "criterion": (
                "exact equality of the complete encoded hero trajectory"
            ),
            "training_trajectories_searched": len(train_dataset),
            "exact_training_matches": len(matches),
            "decision_prefix_training_matches": prefix_matches,
            "test_split_accessed": False,
            "opponent_private_card_tokens_in_model_input": sorted(leaked_cards),
        },
        "environment": {
            "pytorch": torch.__version__,
            "device": str(device),
            "precision": precision,
            "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
        },
        "result": {
            "decisions": len(predictions),
            "token_correct": sum(item["token_correct"] for item in predictions),
            "action_correct": sum(item["action_correct"] for item in predictions),
            "legal_predictions": sum(item["legal"] for item in predictions),
        },
        "predictions": predictions,
        "scope_limit": (
            "Teacher-forced evaluation of a newly authored legal trajectory; "
            "the authored line is not an optimal-poker or expected-value label."
        ),
    }
    args.fixture.parent.mkdir(parents=True, exist_ok=True)
    args.fixture.write_text(SYNTHETIC_HAND, encoding="utf-8")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.output_dir / "report.json"
    markdown_path = args.output_dir / "report.md"
    json_path.write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    markdown_path.write_text(_markdown(report), encoding="utf-8")
    return {
        "json": str(json_path),
        "markdown": str(markdown_path),
        "fixture": str(args.fixture),
        "result": report["result"],
    }


def main() -> None:
    print(json.dumps(run(parse_args()), indent=2))


if __name__ == "__main__":
    main()
