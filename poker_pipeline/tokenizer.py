from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal

from .phh import BoardReveal, Decision, ForcedPost, HeroTrajectory, HistoryAction


RANKS = "23456789TJQKA"
SUITS = "cdhs"
ACTIONS = ("FOLD", "CHECK", "CALL", "BET", "RAISE")
RANGE_LABELS = (
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
RANGE_THRESHOLDS = tuple(
    Decimal(value)
    for value in ("0.25", "0.5", "0.75", "1", "1.5", "2", "3", "5", "10", "20", "50")
)


def range_label(value: object) -> str:
    numeric = value if isinstance(value, Decimal) else Decimal(str(value))
    if numeric <= 0:
        return "ZERO"
    for threshold, label in zip(RANGE_THRESHOLDS, RANGE_LABELS[1:-1]):
        if numeric <= threshold:
            return label
    return "GT_50"


# Backward-compatible import name for callers that only need bucket assignment.
ratio_label = range_label


def build_vocabulary() -> list[str]:
    tokens = [
        "<PAD>",
        "<BOS>",
        "<EOS>",
        "<PLAYER_1_DECISION>",
        "<PLAYER_STATES>",
        "TABLE_SIZE",
        "PLAYER_1_HOLE_CARDS",
        "BOARD_REVEAL",
        "POT_SIZE_BB",
        "TO_CALL_BB",
        "STACK_BB",
        "AMOUNT_BB",
        "AMOUNT_POT",
        "POST_ANTE",
        "POST_BLIND",
        "POST_STRADDLE",
        "VALUE_BB_0.5",
        "VALUE_BB_1",
        "STATUS_ACTIVE",
        "STATUS_ALL_IN",
    ]
    tokens += [f"COUNT_{count}" for count in range(11)]
    tokens += [f"PLAYER_{number}" for number in range(1, 11)]
    tokens += [f"CARD_{rank}{suit}" for rank in RANKS for suit in SUITS]
    tokens += [f"ACTION_{action}" for action in ACTIONS]
    tokens += [f"RANGE_{label}" for label in RANGE_LABELS]
    return tokens


@dataclass(frozen=True)
class EncodedTrajectory:
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
    def _range(value: object) -> str:
        return f"RANGE_{range_label(value)}"

    def _append_public_event(
        self, tokens: list[str], event: ForcedPost | BoardReveal | HistoryAction, trajectory: HeroTrajectory
    ) -> None:
        if isinstance(event, BoardReveal):
            tokens += ["BOARD_REVEAL", f"COUNT_{event.count}"]
            tokens += [f"CARD_{card}" for card in event.cards]
            return
        player = self._player_token(event.player, trajectory.hero, trajectory.player_count)
        if isinstance(event, ForcedPost):
            tokens += [player, event.kind]
            if event.amount_bb == Decimal("0.5"):
                tokens.append("VALUE_BB_0.5")
            elif event.amount_bb == Decimal(1):
                tokens.append("VALUE_BB_1")
            else:
                tokens += ["AMOUNT_BB", self._range(event.amount_bb)]
            return
        if isinstance(event, HistoryAction):
            tokens += [player, f"ACTION_{event.action}"]
            if event.amount > 0:
                tokens += [
                    "AMOUNT_BB",
                    self._range(event.amount_bb),
                    "AMOUNT_POT",
                    self._range(event.amount_pot),
                ]
            return
        raise TypeError(f"Unsupported public event type: {type(event).__name__}")

    def _append_hero_decision(
        self, tokens: list[str], mask: list[int], decision: Decision, trajectory: HeroTrajectory
    ) -> None:
        tokens += [
            "<PLAYER_1_DECISION>",
            "POT_SIZE_BB",
            self._range(decision.pot / decision.big_blind),
            "TO_CALL_BB",
            self._range(decision.to_call / decision.big_blind),
            "<PLAYER_STATES>",
        ]
        mask.extend([0] * 6)
        for original_player in range(decision.player_count):
            player = self._player_token(original_player, trajectory.hero, trajectory.player_count)
            status = decision.player_statuses[original_player]
            if status == "FOLDED":
                continue
            state_tokens = [
                player,
                f"STATUS_{status}",
                "STACK_BB",
                self._range(decision.player_stacks[original_player] / decision.big_blind),
            ]
            tokens += state_tokens
            mask.extend([0] * len(state_tokens))
        tokens.append(f"ACTION_{decision.target_action}")
        mask.append(1)
        if decision.target_action in {"BET", "RAISE"}:
            tokens += [
                "AMOUNT_BB",
                self._range(decision.target_amount_bb),
                "AMOUNT_POT",
                self._range(decision.target_amount_pot),
            ]
            mask.extend((0, 1, 0, 1))

    def encode_trajectory(self, trajectory: HeroTrajectory) -> EncodedTrajectory:
        if len(trajectory.hero_cards) != 2 or any(card == "??" for card in trajectory.hero_cards):
            raise ValueError("A trajectory requires two known hero cards")
        tokens = [
            "<BOS>",
            "TABLE_SIZE",
            f"COUNT_{trajectory.player_count}",
            "PLAYER_1_HOLE_CARDS",
            *(f"CARD_{card}" for card in trajectory.hero_cards),
        ]
        mask = [0] * len(tokens)
        for item in trajectory.items:
            if isinstance(item, Decision):
                self._append_hero_decision(tokens, mask, item, trajectory)
            else:
                before = len(tokens)
                self._append_public_event(tokens, item, trajectory)
                mask.extend([0] * (len(tokens) - before))
        tokens.append("<EOS>")
        mask.append(0)
        if len(tokens) != len(mask):
            raise RuntimeError("Token and loss-mask lengths diverged")
        return EncodedTrajectory(
            tuple(self._id(token) for token in tokens), tuple(mask), tuple(tokens)
        )

    def decode(self, ids: list[int] | tuple[int, ...]) -> list[str]:
        return [self.itos[index] for index in ids]
