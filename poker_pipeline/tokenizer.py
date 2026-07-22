from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal

from .phh import BoardReveal, Decision, ForcedPost, HeroTrajectory, HistoryAction


RANKS = "23456789TJQKA"
SUITS = "cdhs"
PUBLIC_ACTIONS = ("FOLD", "CHECK", "CALL", "BET", "RAISE")
HERO_ACTIONS = ("PASSIVE", "ALL_IN")
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
    "50_TO_75",
    "75_TO_100",
    "100_TO_150",
    "GT_150",
)
RANGE_THRESHOLDS = tuple(
    Decimal(value)
    for value in (
        "0.25",
        "0.5",
        "0.75",
        "1",
        "1.5",
        "2",
        "3",
        "5",
        "10",
        "20",
        "50",
        "75",
        "100",
        "150",
    )
)
RANGE_TOKENS = tuple(f"RANGE_{label}" for label in RANGE_LABELS if label != "ZERO")
SIZING_OUTPUT_TOKENS = RANGE_TOKENS[:10]
CONTEXT_ONLY_RANGE_TOKENS = RANGE_TOKENS[len(SIZING_OUTPUT_TOKENS) :]
DECISION_TOKENS = (
    "ACTION_FOLD",
    "ACTION_PASSIVE",
    *SIZING_OUTPUT_TOKENS,
    "ACTION_ALL_IN",
)
POSITION_TOKENS = (
    "POSITION_SMALL_BLIND",
    "POSITION_BIG_BLIND",
    "POSITION_UTG",
    "POSITION_HIJACK",
    "POSITION_CUTOFF",
    "POSITION_BUTTON",
)
COUNT_TOKENS = tuple(f"COUNT_{count}" for count in (1, 3, 4, 5))
PLAYER_TOKENS = tuple(f"PLAYER_{number}" for number in range(1, 7))


def range_label(value: object) -> str:
    numeric = value if isinstance(value, Decimal) else Decimal(str(value))
    if numeric <= 0:
        return "ZERO"
    for threshold, label in zip(RANGE_THRESHOLDS, RANGE_LABELS[1:-1]):
        if numeric <= threshold:
            return label
    return RANGE_LABELS[-1]


# Backward-compatible import name for callers that only need bucket assignment.
ratio_label = range_label


def default_range_representatives() -> dict[str, Decimal]:
    """Return deterministic fallbacks for executable sizing-output ranges."""
    representatives: dict[str, Decimal] = {}
    lower = Decimal(0)
    for threshold, token in zip(RANGE_THRESHOLDS, SIZING_OUTPUT_TOKENS):
        representatives[token] = (lower + threshold) / 2
        lower = threshold
    return representatives


def build_vocabulary() -> list[str]:
    tokens = [
        "<PAD>",
        "<BOS>",
        "<EOS>",
        "<PLAYER_1_DECISION>",
        "PLAYER_1_HOLE_CARDS",
        "CURRENT_BOARD",
        "BOARD_REVEAL",
        "POT_SIZE_BB",
        "TO_CALL_BB",
        "STACK_POT",
        "AMOUNT_BB",
        "AMOUNT_POT",
        "STATUS_ACTIVE",
        "STATUS_ALL_IN",
    ]
    tokens += list(POSITION_TOKENS)
    tokens += list(COUNT_TOKENS)
    tokens += list(PLAYER_TOKENS)
    tokens += [f"CARD_{rank}{suit}" for rank in RANKS for suit in SUITS]
    tokens += [f"ACTION_{action}" for action in (*PUBLIC_ACTIONS, *HERO_ACTIONS)]
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

    def hero_decision_token(self, decision: Decision) -> str:
        if decision.target_action == "FOLD":
            return "ACTION_FOLD"
        if decision.target_action in {"CHECK", "CALL"}:
            return "ACTION_PASSIVE"
        if decision.target_action not in {"BET", "RAISE"}:
            raise ValueError(f"Unsupported hero decision: {decision.target_action}")
        if decision.target_amount > 0 and decision.target_amount == decision.hero_stack:
            return "ACTION_ALL_IN"
        token = self._range(decision.target_amount_pot)
        if token == "RANGE_ZERO":
            raise ValueError("RANGE_ZERO is not a valid aggressive hero decision")
        if token not in SIZING_OUTPUT_TOKENS:
            raise ValueError(
                f"Aggressive target {token} is context-only in the fixed Pluribus schema"
            )
        return token

    @staticmethod
    def _hero_position_token(trajectory: HeroTrajectory) -> str:
        if trajectory.player_count != 6:
            raise ValueError(
                "The fixed Pluribus schema requires exactly six players"
            )
        forced_posts = [
            item for item in trajectory.items if isinstance(item, ForcedPost)
        ]
        expected_amounts = (Decimal("0.5"), Decimal(1))
        if (
            len(forced_posts) != 2
            or any(post.kind != "POST_BLIND" for post in forced_posts)
            or tuple(post.amount_bb for post in forced_posts) != expected_amounts
        ):
            raise ValueError(
                "The fixed Pluribus schema requires no antes/straddles and exactly "
                "0.5/1 BB blind posts"
            )
        small_blind, big_blind = (post.player for post in forced_posts)
        if big_blind != (small_blind + 1) % trajectory.player_count:
            raise ValueError("Blind seats are not consecutive clockwise")
        offset = (trajectory.hero - small_blind) % trajectory.player_count
        return POSITION_TOKENS[offset]

    @staticmethod
    def _validate_starting_depth(trajectory: HeroTrajectory) -> None:
        first_decision = next(
            (item for item in trajectory.items if isinstance(item, Decision)), None
        )
        if first_decision is None:
            raise ValueError("A trajectory requires at least one hero decision")
        starting_depths = tuple(
            stack / first_decision.big_blind for stack in first_decision.starting_stacks
        )
        if starting_depths != (Decimal(100),) * 6:
            raise ValueError(
                "The fixed Pluribus schema requires every seat to start at 100 BB"
            )

    def _append_public_event(
        self, tokens: list[str], event: ForcedPost | BoardReveal | HistoryAction, trajectory: HeroTrajectory
    ) -> None:
        if isinstance(event, BoardReveal):
            tokens += ["BOARD_REVEAL", f"COUNT_{event.count}"]
            tokens += [f"CARD_{card}" for card in event.cards]
            return
        if isinstance(event, ForcedPost):
            # The single hero-position token encodes both blind identities.
            return
        player = self._player_token(event.player, trajectory.hero, trajectory.player_count)
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
        if decision.pot <= 0:
            raise ValueError("A decision requires a positive public pot")
        observation_tokens = [
            "<PLAYER_1_DECISION>",
            "PLAYER_1_HOLE_CARDS",
            *(f"CARD_{card}" for card in decision.hero_cards),
        ]
        if decision.board:
            observation_tokens += [
                "CURRENT_BOARD",
                f"COUNT_{len(decision.board)}",
                *(f"CARD_{card}" for card in decision.board),
            ]
        observation_tokens += [
            "POT_SIZE_BB",
            self._range(decision.pot / decision.big_blind),
            "TO_CALL_BB",
            self._range(decision.to_call / decision.big_blind),
        ]
        tokens += observation_tokens
        mask.extend([0] * len(observation_tokens))
        for original_player in range(decision.player_count):
            player = self._player_token(original_player, trajectory.hero, trajectory.player_count)
            status = decision.player_statuses[original_player]
            if status == "FOLDED":
                continue
            state_tokens = [
                player,
                f"STATUS_{status}",
            ]
            if status != "ALL_IN":
                state_tokens += [
                    "STACK_POT",
                    self._range(decision.player_stacks[original_player] / decision.pot),
                ]
            tokens += state_tokens
            mask.extend([0] * len(state_tokens))
        tokens.append(self.hero_decision_token(decision))
        mask.append(1)

    def encode_trajectory(self, trajectory: HeroTrajectory) -> EncodedTrajectory:
        if len(trajectory.hero_cards) != 2 or any(card == "??" for card in trajectory.hero_cards):
            raise ValueError("A trajectory requires two known hero cards")
        position_token = self._hero_position_token(trajectory)
        self._validate_starting_depth(trajectory)
        tokens = [
            "<BOS>",
            position_token,
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
