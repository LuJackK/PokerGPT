from __future__ import annotations

from dataclasses import dataclass

from .phh import Decision


RANKS = "23456789TJQKA"
SUITS = "cdhs"
ACTIONS = ("FOLD", "CHECK", "CALL", "BET", "RAISE")
STREETS = ("PREFLOP", "FLOP", "TURN", "RIVER")
RATIO_LABELS = (
    "ZERO",
    "LE_0.25",
    "LE_0.5",
    "LE_0.75",
    "LE_1",
    "LE_1.5",
    "LE_2",
    "LE_3",
    "LE_5",
    "LE_10",
    "LE_20",
    "LE_50",
    "GT_50",
)


def ratio_label(value: object) -> str:
    value = float(value)
    if value <= 0:
        return "ZERO"
    for threshold, label in zip(
        (0.25, 0.5, 0.75, 1, 1.5, 2, 3, 5, 10, 20, 50), RATIO_LABELS[1:-1]
    ):
        if value <= threshold:
            return label
    return "GT_50"


def build_vocabulary() -> list[str]:
    tokens = ["<PAD>", "<BOS>", "<EOS>", "<TARGET>", "<HISTORY>", "<NO_HISTORY>"]
    tokens += [f"STREET_{street}" for street in STREETS]
    tokens += [f"PLAYERS_{count}" for count in range(2, 11)]
    tokens += [f"ACTIVE_{count}" for count in range(1, 11)]
    tokens += [f"HERO_SEAT_{seat}" for seat in range(1, 11)]
    tokens += ["PLAYER_HERO"] + [f"PLAYER_REL_{seat}" for seat in range(1, 10)]
    tokens += ["CARD_UNKNOWN"] + [f"CARD_{rank}{suit}" for rank in RANKS for suit in SUITS]
    tokens += [f"BOARD_COUNT_{count}" for count in range(6)]
    tokens += [f"LEGAL_{action}" for action in ACTIONS]
    tokens += [f"ACTION_{action}" for action in ACTIONS]
    tokens += [f"HIST_STREET_{street}" for street in STREETS]
    for prefix in ("POT_BB", "CALL_BB", "STACK_BB", "EFFECTIVE_BB", "AMOUNT_BB", "AMOUNT_POT"):
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

    def encode_decision(self, decision: Decision, history_limit: int = 32) -> EncodedDecision:
        bb = decision.big_blind
        tokens = [
            "<BOS>",
            f"STREET_{decision.street}",
            f"PLAYERS_{decision.player_count}",
            f"ACTIVE_{decision.active_players}",
            f"HERO_SEAT_{decision.actor + 1}",
            "PLAYER_HERO",
        ]
        hero_cards = list(decision.hero_cards[:2])
        while len(hero_cards) < 2:
            hero_cards.append("UNKNOWN")
        tokens += ["CARD_UNKNOWN" if card == "UNKNOWN" else f"CARD_{card}" for card in hero_cards]
        for relative in range(1, decision.player_count):
            tokens += [f"PLAYER_REL_{relative}", "CARD_UNKNOWN", "CARD_UNKNOWN"]
        tokens += [f"BOARD_COUNT_{len(decision.board)}"]
        tokens += [f"CARD_{card}" for card in decision.board]
        tokens += [
            f"POT_BB_{ratio_label(decision.pot / bb)}",
            f"CALL_BB_{ratio_label(decision.to_call / bb)}",
            f"STACK_BB_{ratio_label(decision.hero_stack / bb)}",
            f"EFFECTIVE_BB_{ratio_label(decision.effective_stack / bb)}",
        ]
        tokens += [f"LEGAL_{action}" for action in decision.legal_actions]
        tokens.append("<HISTORY>")
        history = decision.history[-history_limit:]
        if not history:
            tokens.append("<NO_HISTORY>")
        for item in history:
            relative = (item.player - decision.actor) % decision.player_count
            player_token = "PLAYER_HERO" if relative == 0 else f"PLAYER_REL_{relative}"
            tokens += [
                f"HIST_STREET_{item.street}",
                player_token,
                f"ACTION_{item.action}",
                f"AMOUNT_BB_{ratio_label(item.amount_bb)}",
                f"AMOUNT_POT_{ratio_label(item.amount_pot)}",
            ]
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
