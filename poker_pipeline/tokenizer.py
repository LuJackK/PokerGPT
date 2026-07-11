from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal

from .phh import BoardReveal, Decision, ForcedPost, HandEvent, HistoryAction


RANKS = "23456789TJQKA"
SUITS = "cdhs"
ACTIONS = ("FOLD", "CHECK", "CALL", "BET", "RAISE")
RATIO_LABELS = (
    "ZERO",
    "0_TO_0.25",
    "0.25_TO_0.5",
    "0.5_TO_0.75",
    "0.75_TO_1",
    "1_TO_1.5",
    "1.5_TO_2",
    "2_TO_3",
    "3_TO_5",
    "5_TO_10",
    "10_TO_20",
    "20_TO_50",
    "GT_50",
)
RATIO_THRESHOLDS = tuple(
    Decimal(value)
    for value in ("0.25", "0.5", "0.75", "1", "1.5", "2", "3", "5", "10", "20", "50")
)


def ratio_label(value: object) -> str:
    numeric = value if isinstance(value, Decimal) else Decimal(str(value))
    if numeric <= 0:
        return "ZERO"
    for threshold, label in zip(RATIO_THRESHOLDS, RATIO_LABELS[1:-1]):
        if numeric <= threshold:
            return label
    return "GT_50"


def build_vocabulary() -> list[str]:
    tokens = [
        "<PAD>",
        "<BOS>",
        "<EOS>",
        "<TARGET>",
        "<EVENT_SEQUENCE>",
        "<NO_EVENTS>",
        "PLAYER_1_HOLE_CARDS",
    ]
    tokens += [f"PLAYERS_{count}" for count in range(2, 11)]
    tokens += [f"ACTIVE_{count}" for count in range(1, 11)]
    tokens += [f"PLAYER_{number}" for number in range(1, 11)]
    tokens += ["CARD_UNKNOWN"] + [f"CARD_{rank}{suit}" for rank in RANKS for suit in SUITS]
    tokens += [f"BOARD_COUNT_{count}" for count in range(6)]
    tokens += [f"BOARD_REVEAL_{count}" for count in range(1, 6)]
    tokens += ["POST_ANTE", "POST_BLIND", "POST_STRADDLE", "BLIND_BB_0.5", "BLIND_BB_1"]
    tokens += [f"ACTION_{action}" for action in ACTIONS]
    for prefix in (
        "POT_SIZE_BB",
        "PLAYER_1_TO_CALL_BB",
        "PLAYER_1_STACK_BB",
        "EFFECTIVE_STACK_BB",
        "AMOUNT_BB",
        "AMOUNT_POT",
    ):
        tokens += [f"{prefix}_{label}" for label in RATIO_LABELS]
    return tokens


@dataclass(frozen=True)
class EncodedDecision:
    ids: tuple[int, ...]
    loss_mask: tuple[int, ...]
    tokens: tuple[str, ...]


class PokerTokenizer:
    def __init__(self) -> None:
        self.itos = build_vocabulary()
        self.stoi = {token: index for index, token in enumerate(self.itos)}
        if len(self.stoi) != len(self.itos):
            raise RuntimeError("Vocabulary contains duplicate tokens")

    def _id(self, token: str) -> int:
        try:
            return self.stoi[token]
        except KeyError as exc:
            raise ValueError(f"Token is outside the fixed vocabulary: {token}") from exc

    @staticmethod
    def _player_token(original_player: int, hero: int, player_count: int) -> str:
        relative_number = (original_player - hero) % player_count + 1
        return f"PLAYER_{relative_number}"

    @staticmethod
    def _limit_events(events: tuple[HandEvent, ...], event_limit: int) -> list[HandEvent]:
        if len(events) <= event_limit:
            return list(events)
        structural = {
            index
            for index, event in enumerate(events)
            if isinstance(event, (ForcedPost, BoardReveal))
        }
        remaining = max(0, event_limit - len(structural))
        action_indexes = [index for index in range(len(events)) if index not in structural]
        selected = structural | set(action_indexes[-remaining:] if remaining else [])
        return [event for index, event in enumerate(events) if index in selected]

    def encode_decision(self, decision: Decision, event_limit: int = 64) -> EncodedDecision:
        bb = decision.big_blind
        tokens = [
            "<BOS>",
            f"PLAYERS_{decision.player_count}",
            f"ACTIVE_{decision.active_players}",
            "PLAYER_1_HOLE_CARDS",
        ]
        hero_cards = list(decision.hero_cards[:2])
        while len(hero_cards) < 2:
            hero_cards.append("UNKNOWN")
        tokens += ["CARD_UNKNOWN" if card == "UNKNOWN" else f"CARD_{card}" for card in hero_cards]
        tokens += [f"BOARD_COUNT_{len(decision.board)}"]
        tokens += [f"CARD_{card}" for card in decision.board]
        tokens += [
            f"POT_SIZE_BB_{ratio_label(decision.pot / bb)}",
            f"PLAYER_1_TO_CALL_BB_{ratio_label(decision.to_call / bb)}",
            f"PLAYER_1_STACK_BB_{ratio_label(decision.hero_stack / bb)}",
            f"EFFECTIVE_STACK_BB_{ratio_label(decision.effective_stack / bb)}",
            "<EVENT_SEQUENCE>",
        ]
        events = self._limit_events(decision.events, event_limit)
        if not events:
            tokens.append("<NO_EVENTS>")
        for event in events:
            if isinstance(event, BoardReveal):
                tokens.append(f"BOARD_REVEAL_{event.count}")
                continue
            player = self._player_token(event.player, decision.actor, decision.player_count)
            if isinstance(event, ForcedPost):
                tokens += [player, event.kind]
                if event.kind == "POST_BLIND" and event.amount_bb == Decimal("0.5"):
                    tokens.append("BLIND_BB_0.5")
                elif event.kind == "POST_BLIND" and event.amount_bb == Decimal(1):
                    tokens.append("BLIND_BB_1")
                else:
                    tokens.append(f"AMOUNT_BB_{ratio_label(event.amount_bb)}")
                continue
            if isinstance(event, HistoryAction):
                tokens += [player, f"ACTION_{event.action}"]
                if event.amount > 0:
                    tokens += [
                        f"AMOUNT_BB_{ratio_label(event.amount_bb)}",
                        f"AMOUNT_POT_{ratio_label(event.amount_pot)}",
                    ]
                continue
            raise TypeError(f"Unsupported event type: {type(event).__name__}")
        tokens += ["<TARGET>", f"ACTION_{decision.target_action}"]
        target_indices = [len(tokens) - 1]
        if decision.target_action in {"BET", "RAISE"}:
            tokens += [
                f"AMOUNT_BB_{ratio_label(decision.target_amount_bb)}",
                f"AMOUNT_POT_{ratio_label(decision.target_amount_pot)}",
            ]
            target_indices.extend((len(tokens) - 2, len(tokens) - 1))
        tokens.append("<EOS>")
        mask = [0] * len(tokens)
        for index in target_indices:
            mask[index] = 1
        return EncodedDecision(
            tuple(self._id(token) for token in tokens), tuple(mask), tuple(tokens)
        )

    def decode(self, ids: list[int] | tuple[int, ...]) -> list[str]:
        return [self.itos[index] for index in ids]
