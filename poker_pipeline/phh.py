from __future__ import annotations

import re
import tomllib
import zipfile
from dataclasses import dataclass, field
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any, Iterator


PLAYER_ACTION = re.compile(r"^p(\d+)\s+(f|cc|cbr)(?:\s+([^\s]+))?$")
HOLE_DEAL = re.compile(r"^d\s+dh\s+p(\d+)\s+([2-9TJQKAcdhs?]+)$")
BOARD_DEAL = re.compile(r"^d\s+db\s+([2-9TJQKAcdhs]+)$")
SHOW_OR_KILL = re.compile(r"^p(\d+)\s+(sm|sd|kc|k)\b")
CARD_PATTERN = re.compile(r"[2-9TJQKA][cdhs]")


class HandParseError(ValueError):
    pass


@dataclass(frozen=True)
class HistoryAction:
    player: int
    street: str
    action: str
    amount: Decimal = Decimal(0)
    amount_bb: Decimal = Decimal(0)
    amount_pot: Decimal = Decimal(0)


@dataclass(frozen=True)
class Decision:
    member: str
    hand_key: str
    actor: int
    street: str
    player_count: int
    active_players: int
    hero_cards: tuple[str, ...]
    board: tuple[str, ...]
    pot: Decimal
    to_call: Decimal
    big_blind: Decimal
    hero_stack: Decimal
    effective_stack: Decimal
    legal_actions: tuple[str, ...]
    history: tuple[HistoryAction, ...]
    target_action: str
    target_amount: Decimal = Decimal(0)
    target_amount_bb: Decimal = Decimal(0)
    target_amount_pot: Decimal = Decimal(0)


@dataclass
class ReplayState:
    member: str
    hand_key: str
    player_count: int
    big_blind: Decimal
    stacks: list[Decimal]
    street_contrib: list[Decimal]
    total_contrib: list[Decimal]
    active: list[bool]
    hole_cards: dict[int, tuple[str, ...]] = field(default_factory=dict)
    board: list[str] = field(default_factory=list)
    street: str = "PREFLOP"
    current_bet: Decimal = Decimal(0)
    history: list[HistoryAction] = field(default_factory=list)

    @property
    def pot(self) -> Decimal:
        return sum(self.total_contrib)

    def to_call(self, actor: int) -> Decimal:
        return max(Decimal(0), self.current_bet - self.street_contrib[actor])

    def legal_actions(self, actor: int) -> tuple[str, ...]:
        if not self.active[actor] or self.stacks[actor] <= 0:
            return ()
        owed = self.to_call(actor)
        legal: list[str] = []
        if owed > 0:
            legal.extend(("FOLD", "CALL"))
            if self.stacks[actor] > owed:
                legal.append("RAISE")
        else:
            legal.append("CHECK")
            if self.current_bet == 0 and self.stacks[actor] > 0:
                legal.append("BET")
            elif self.stacks[actor] > 0:
                legal.append("RAISE")
        return tuple(legal)


def split_cards(value: str) -> tuple[str, ...]:
    if value and set(value) == {"?"} and len(value) % 2 == 0:
        return tuple("??" for _ in range(len(value) // 2))
    cards = tuple(CARD_PATTERN.findall(value))
    if "".join(cards) != value:
        raise HandParseError(f"Invalid card string: {value!r}")
    return cards


def parse_document(text: str, extension: str) -> list[tuple[str, dict[str, Any]]]:
    try:
        document = tomllib.loads(text.lstrip("\ufeff"), parse_float=Decimal)
    except tomllib.TOMLDecodeError as exc:
        raise HandParseError(f"Invalid TOML: {exc}") from exc
    if extension.lower() == "phh":
        return [("1", document)]
    hands = [(str(key), value) for key, value in document.items() if isinstance(value, dict)]
    if not hands:
        raise HandParseError("PHHS document contains no hand tables")
    return hands


def read_member_hands(
    archive: zipfile.ZipFile, member: str, max_member_bytes: int = 64 * 1024 * 1024
) -> list[tuple[str, dict[str, Any]]]:
    info = archive.getinfo(member)
    if info.file_size > max_member_bytes:
        raise HandParseError(
            f"Member is {info.file_size} bytes, above limit {max_member_bytes}: {member}"
        )
    with archive.open(info) as source:
        payload = source.read(max_member_bytes + 1)
    if len(payload) > max_member_bytes:
        raise HandParseError(f"Member exceeded read limit: {member}")
    text = payload.decode("utf-8-sig", errors="strict")
    return parse_document(text, Path(member).suffix.lstrip("."))


def _decimal(value: Any) -> Decimal:
    try:
        result = value if isinstance(value, Decimal) else Decimal(str(value))
    except (InvalidOperation, ValueError) as exc:
        raise HandParseError(f"Invalid numeric value: {value!r}") from exc
    if not result.is_finite() or result < 0:
        raise HandParseError(f"Invalid numeric value: {value!r}")
    return result


def _numbers(hand: dict[str, Any], key: str, count: int) -> list[Decimal]:
    values = hand.get(key)
    if not isinstance(values, list) or len(values) != count:
        raise HandParseError(f"{key} must be a list of length {count}")
    try:
        result = [_decimal(value) for value in values]
    except (TypeError, ValueError) as exc:
        raise HandParseError(f"{key} contains a non-numeric value") from exc
    return result


def _initial_state(member: str, hand_key: str, hand: dict[str, Any]) -> ReplayState:
    if str(hand.get("variant", "")).upper() != "NT":
        raise HandParseError(f"Unsupported variant {hand.get('variant')!r}; expected NT")
    stacks_value = hand.get("starting_stacks")
    if not isinstance(stacks_value, list) or not 2 <= len(stacks_value) <= 10:
        raise HandParseError("starting_stacks must contain 2-10 players")
    count = len(stacks_value)
    starting_stacks = _numbers(hand, "starting_stacks", count)
    antes = _numbers(hand, "antes", count)
    blinds = _numbers(hand, "blinds_or_straddles", count)
    min_bet = _decimal(hand.get("min_bet", 0))
    positive_blinds = [value for value in blinds if value > 0]
    big_blind = min_bet or (max(positive_blinds) if positive_blinds else 0.0)
    if big_blind <= 0:
        raise HandParseError("Could not determine a positive big blind/minimum bet")
    posted = [min(stack, ante + blind) for stack, ante, blind in zip(starting_stacks, antes, blinds)]
    stacks = [stack - contribution for stack, contribution in zip(starting_stacks, posted)]
    street_contrib = [min(blind, posted_value) for blind, posted_value in zip(blinds, posted)]
    return ReplayState(
        member=member,
        hand_key=hand_key,
        player_count=count,
        big_blind=big_blind,
        stacks=stacks,
        street_contrib=street_contrib,
        total_contrib=posted,
        active=[True] * count,
        current_bet=max(street_contrib, default=0.0),
    )


def _decision(
    state: ReplayState, actor: int, target: str, amount_to: Decimal = Decimal(0)
) -> Decision:
    owed = state.to_call(actor)
    pot_before = state.pot
    if target in {"BET", "RAISE"}:
        delta = max(Decimal(0), amount_to - state.street_contrib[actor])
    elif target == "CALL":
        delta = min(owed, state.stacks[actor])
    else:
        delta = Decimal(0)
    opponents = [state.stacks[i] for i in range(state.player_count) if i != actor and state.active[i]]
    effective = min(state.stacks[actor], max(opponents, default=Decimal(0)))
    return Decision(
        member=state.member,
        hand_key=state.hand_key,
        actor=actor,
        street=state.street,
        player_count=state.player_count,
        active_players=sum(state.active),
        hero_cards=state.hole_cards.get(actor, ()),
        board=tuple(state.board),
        pot=pot_before,
        to_call=owed,
        big_blind=state.big_blind,
        hero_stack=state.stacks[actor],
        effective_stack=effective,
        legal_actions=state.legal_actions(actor),
        history=tuple(state.history),
        target_action=target,
        target_amount=delta,
        target_amount_bb=delta / state.big_blind,
        target_amount_pot=delta / max(pot_before, state.big_blind),
    )


def replay_hand(member: str, hand_key: str, hand: dict[str, Any]) -> Iterator[Decision]:
    state = _initial_state(member, hand_key, hand)
    actions = hand.get("actions")
    if not isinstance(actions, list) or not actions:
        raise HandParseError("actions must be a non-empty list")
    seen_player_action = False
    for index, raw in enumerate(actions):
        if not isinstance(raw, str):
            raise HandParseError(f"Action {index} is not a string")
        action = raw.strip()
        hole = HOLE_DEAL.fullmatch(action)
        if hole:
            player = int(hole.group(1)) - 1
            if not 0 <= player < state.player_count:
                raise HandParseError(f"Hole deal has invalid player: {action}")
            state.hole_cards[player] = split_cards(hole.group(2))
            continue
        board = BOARD_DEAL.fullmatch(action)
        if board:
            cards = split_cards(board.group(1))
            state.board.extend(cards)
            state.street = {3: "FLOP", 4: "TURN", 5: "RIVER"}.get(len(state.board), state.street)
            state.street_contrib = [Decimal(0)] * state.player_count
            state.current_bet = Decimal(0)
            continue
        player_action = PLAYER_ACTION.fullmatch(action)
        if player_action:
            seen_player_action = True
            actor = int(player_action.group(1)) - 1
            code = player_action.group(2)
            if not 0 <= actor < state.player_count:
                raise HandParseError(f"Action has invalid player: {action}")
            if not state.active[actor] or state.stacks[actor] <= 0:
                raise HandParseError(f"Inactive/all-in player acts: {action}")
            owed = state.to_call(actor)
            amount_to = Decimal(0)
            if code == "f":
                target = "FOLD"
            elif code == "cc":
                target = "CALL" if owed > 0 else "CHECK"
            else:
                if player_action.group(3) is None:
                    raise HandParseError(f"cbr action is missing amount: {action}")
                try:
                    amount_to = _decimal(player_action.group(3))
                except HandParseError as exc:
                    raise HandParseError(f"Invalid cbr amount: {action}") from exc
                target = "BET" if state.current_bet == 0 else "RAISE"
            decision = _decision(state, actor, target, amount_to)
            if target not in decision.legal_actions:
                raise HandParseError(
                    f"Observed target {target} not legal {decision.legal_actions}: {action}"
                )
            yield decision

            pot_before = state.pot
            contribution = Decimal(0)
            if target == "FOLD":
                state.active[actor] = False
            elif target == "CALL":
                contribution = min(owed, state.stacks[actor])
            elif target in {"BET", "RAISE"}:
                if amount_to <= state.street_contrib[actor]:
                    raise HandParseError(f"Non-increasing cbr amount: {action}")
                contribution = min(amount_to - state.street_contrib[actor], state.stacks[actor])
            state.stacks[actor] -= contribution
            state.street_contrib[actor] += contribution
            state.total_contrib[actor] += contribution
            state.current_bet = max(state.current_bet, state.street_contrib[actor])
            state.history.append(
                HistoryAction(
                    player=actor,
                    street=state.street,
                    action=target,
                    amount=contribution,
                    amount_bb=contribution / state.big_blind,
                    amount_pot=contribution / max(pot_before, state.big_blind),
                )
            )
            continue
        if SHOW_OR_KILL.match(action):
            continue
        raise HandParseError(f"Unsupported action syntax at index {index}: {action!r}")
    if not seen_player_action:
        raise HandParseError("Hand contains no player decisions")


def iter_archive_decisions(
    zip_path: Path,
    members: list[dict[str, Any]],
    max_member_bytes: int = 64 * 1024 * 1024,
) -> Iterator[tuple[dict[str, Any], Decision | None, str | None]]:
    with zipfile.ZipFile(zip_path) as archive:
        archive_names = set(archive.namelist())
        for selection in members:
            member = selection["member"]
            if member not in archive_names:
                yield selection, None, "member_not_found"
                continue
            try:
                hands = read_member_hands(archive, member, max_member_bytes)
                member_decisions: list[Decision] = []
                for hand_key, hand in hands:
                    member_decisions.extend(replay_hand(member, hand_key, hand))
                for decision in member_decisions:
                    yield selection, decision, None
            except (HandParseError, UnicodeDecodeError, KeyError, zipfile.BadZipFile) as exc:
                yield selection, None, f"{type(exc).__name__}:{exc}"
