from __future__ import annotations

import json
import re
import zipfile
from collections import Counter, defaultdict
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any, Callable

from .phh import (
    HandParseError,
    build_hero_trajectories,
    parse_document,
    replay_hand,
    split_cards,
)
from .tokenizer import range_label


AUDIT_SCHEMA_VERSION = 1
HANDHQ_SEGMENT = "/handhq/"
SECTION_PATTERN = re.compile(r"(?m)^[ \t]*\[([^\]]+)\][ \t]*$")
VARIANT_PATTERN = re.compile(
    r"(?m)^[ \t]*variant[ \t]*=[ \t]*['\"]([^'\"]+)['\"]"
)
STACKS_PATTERN = re.compile(
    r"(?m)^[ \t]*starting_stacks[ \t]*=[ \t]*\[([^\r\n]*)\]"
)
ACTIONS_PATTERN = re.compile(r"(?m)^[ \t]*actions[ \t]*=[ \t]*\[([^\r\n]*)\]")
VENUE_PATTERN = re.compile(
    r"(?m)^[ \t]*venue[ \t]*=[ \t]*['\"]([^'\"]*)['\"]"
)
KNOWN_HOLE_PATTERN = re.compile(
    r"'d dh p(?P<player>\d+) (?P<cards>[2-9TJQKA][cdhs][2-9TJQKA][cdhs])'"
)
AUDIT_EVENT_PATTERN = re.compile(
    r"'(?:"
    r"d dh p(?P<hole_player>\d+) (?P<hole_cards>[2-9TJQKA][cdhs][2-9TJQKA][cdhs])"
    r"|d db (?P<board>[2-9TJQKAcdhs]+)"
    r"|p(?P<action_player>\d+) (?P<action_code>f|cc|cbr)(?: [^']+)?"
    r"|p(?P<show_player>\d+) (?P<show_code>sm|sd|kc|k)(?: [^']*)?"
    r")'"
)


def _counter_dict(counter: Counter[Any]) -> dict[str, int]:
    return {
        str(key): value
        for key, value in sorted(counter.items(), key=lambda item: str(item[0]))
    }


def _decimal(value: Any) -> Decimal | None:
    try:
        result = value if isinstance(value, Decimal) else Decimal(str(value))
    except (InvalidOperation, TypeError, ValueError):
        return None
    return result if result.is_finite() else None


def _fast_stacks(block: str) -> tuple[int, tuple[Decimal, ...] | None]:
    match = STACKS_PATTERN.search(block)
    if not match:
        return 0, None
    raw_values = [item.strip() for item in match.group(1).split(",")]
    player_count = len(raw_values)
    stacks = tuple(_decimal(value) for value in raw_values)
    if not 2 <= player_count <= 10 or any(
        value is None or value < 0 for value in stacks
    ):
        return player_count, None
    return player_count, tuple(value for value in stacks if value is not None)


def _fast_actions(block: str) -> str | None:
    match = ACTIONS_PATTERN.search(block)
    return match.group(1) if match else None


def _fast_known_actors(actions: str, player_count: int) -> set[int]:
    known_players = {
        int(match.group("player")) - 1
        for match in KNOWN_HOLE_PATTERN.finditer(actions)
        if 1 <= int(match.group("player")) <= player_count
    }
    return {
        player
        for player in known_players
        if any(
            f"'p{player + 1} {code}" in actions for code in ("f'", "cc'", "cbr ")
        )
    }


def _iter_member_blocks(
    archive: zipfile.ZipFile, info: zipfile.ZipInfo, max_member_bytes: int
):
    if info.file_size > max_member_bytes:
        raise HandParseError(
            f"Member is {info.file_size} bytes, above limit {max_member_bytes}: {info.filename}"
        )
    with archive.open(info) as source:
        payload = source.read(max_member_bytes + 1)
    if len(payload) > max_member_bytes:
        raise HandParseError(f"Member exceeded read limit: {info.filename}")
    text = payload.decode("utf-8-sig", errors="strict")
    if info.filename.lower().endswith(".phh"):
        yield "1", text
        return
    sections = list(SECTION_PATTERN.finditer(text))
    if not sections:
        raise HandParseError("PHHS document contains no hand tables")
    for index, section in enumerate(sections):
        start = section.end()
        end = sections[index + 1].start() if index + 1 < len(sections) else len(text)
        yield section.group(1), text[start:end]


def _parse_candidate_block(hand_key: str, block: str) -> dict[str, Any]:
    document = parse_document(f"[{hand_key}]\n{block}", "phhs")
    return document[0][1]


def _big_blind(hand: dict[str, Any]) -> Decimal | None:
    minimum = _decimal(hand.get("min_bet"))
    if minimum is not None and minimum > 0:
        return minimum
    values = hand.get("blinds_or_straddles")
    if not isinstance(values, list):
        return None
    blinds = [_decimal(value) for value in values]
    positive = [value for value in blinds if value is not None and value > 0]
    return max(positive) if positive else None


def _analyze_actions(
    actions: str, player_count: int
) -> tuple[dict[int, tuple[str, ...]], dict[int, list[tuple[str, str]]], set[int], set[str]]:
    known_cards: dict[int, tuple[str, ...]] = {}
    paths: dict[int, list[tuple[str, str]]] = defaultdict(list)
    showdown_players: set[int] = set()
    reached_streets = {"PREFLOP"}
    street = "PREFLOP"
    board_count = 0
    for event in AUDIT_EVENT_PATTERN.finditer(actions):
        if event.group("hole_player") is not None:
            player = int(event.group("hole_player")) - 1
            if 0 <= player < player_count:
                cards = split_cards(event.group("hole_cards"))
                known_cards[player] = cards
            continue
        if event.group("board") is not None:
            board_count += len(event.group("board")) // 2
            street = {3: "FLOP", 4: "TURN", 5: "RIVER"}.get(board_count, street)
            reached_streets.add(street)
            continue
        player_text = event.group("action_player") or event.group("show_player")
        player = int(player_text) - 1
        if not 0 <= player < player_count:
            continue
        code = event.group("action_code")
        if code is not None:
            paths[player].append((street, code))
        else:
            showdown_players.add(player)
    return known_cards, dict(paths), showdown_players, reached_streets


def _fast_venue(block: str) -> str:
    match = VENUE_PATTERN.search(block)
    return match.group(1).strip() if match and match.group(1).strip() else "unknown"


def _replay_error_category(exc: Exception) -> str:
    message = str(exc)
    categories = (
        ("inactive_or_all_in_player_acts", "Inactive/all-in player acts"),
        ("observed_action_illegal", "Observed target"),
        ("non_increasing_raise", "Non-increasing cbr amount"),
        ("unsupported_action_syntax", "Unsupported action syntax"),
        ("invalid_numeric_value", "Invalid numeric value"),
        ("invalid_starting_stacks", "starting_stacks"),
        ("invalid_antes", "antes"),
        ("invalid_blinds", "blinds_or_straddles"),
        ("invalid_big_blind", "positive big blind"),
        ("invalid_actions", "actions must"),
        ("invalid_hole_deal", "Hole deal"),
        ("invalid_cbr", "cbr"),
    )
    for category, marker in categories:
        if marker in message:
            return category
    return type(exc).__name__


def _rate(numerator: int, denominator: int) -> float | None:
    return round(numerator / denominator, 6) if denominator else None


def _cohort_report(
    metrics: Counter[str], actions: Counter[str], first: Counter[str]
) -> dict[str, Any]:
    perspectives = metrics["perspectives"]
    return {
        "perspectives": perspectives,
        "decisions": metrics["decisions"],
        "action_codes": _counter_dict(actions),
        "first_action_codes": _counter_dict(first),
        "rates": {
            "folded": _rate(metrics["folded"], perspectives),
            "acted_postflop": _rate(metrics["acted_postflop"], perspectives),
            "acted_turn_or_river": _rate(metrics["acted_turn_or_river"], perspectives),
            "showdown_marker": _rate(metrics["showdown_marker"], perspectives),
            "hand_reached_flop": _rate(metrics["hand_reached_flop"], perspectives),
            "hand_reached_river": _rate(metrics["hand_reached_river"], perspectives),
        },
        "counts": {
            key: metrics[key]
            for key in (
                "folded",
                "acted_postflop",
                "acted_turn_or_river",
                "showdown_marker",
                "hand_reached_flop",
                "hand_reached_river",
            )
        },
    }


def _markdown(report: dict[str, Any]) -> str:
    totals = report["totals"]
    eligibility = report["eligibility"]
    known = report["selection_bias"]["known_cards"]
    unknown = report["selection_bias"]["unknown_cards"]

    def percent(value: float | None) -> str:
        return "n/a" if value is None else f"{100 * value:.2f}%"

    eligible = eligibility["replay_valid_trajectories"]
    if eligible == 0:
        conclusion = (
            "No player-perspective trajectory satisfied finite-table stacks, known hero cards, "
            "and complete replay. HandHQ is not usable for the current supervised representation."
        )
    else:
        conclusion = (
            f"The archive contains {eligible:,} replay-valid candidate trajectories. "
            "Their selection-bias measurements below must be considered before admitting them "
            "to training; this audit does not automatically approve the corpus."
        )

    lines = [
        "# HandHQ full per-hand audit",
        "",
        f"Generated: {report['generated_at_utc']}",
        "",
        "## Result",
        "",
        conclusion,
        "",
        "## Coverage and eligibility",
        "",
        "| Measure | Count |",
        "|---|---:|",
        f"| ZIP members audited | {totals['members_audited']:,} |",
        f"| Hands parsed | {totals['hands_parsed']:,} |",
        f"| No-limit Hold'em hands | {totals['nt_hands']:,} |",
        f"| Hands with all finite starting stacks | {totals['finite_stack_hands']:,} |",
        f"| Hands with at least one known-card actor | {totals['known_actor_hands']:,} |",
        f"| Finite-stack hands with a known-card actor | {eligibility['candidate_hands']:,} |",
        f"| Replay-valid candidate hands | {eligibility['replay_valid_hands']:,} |",
        f"| Replay-valid candidate trajectories | {eligible:,} |",
        f"| Supervised decisions in valid trajectories | {eligibility['replay_valid_decisions']:,} |",
        "",
        "## Known-card selection check",
        "",
        "The rates are player-perspective rates within NT hands that contain at least one "
        "known-card actor. This within-hand comparison controls for hand-level and site-level "
        "differences. `cc` and `cbr` remain raw PHH codes so the comparison also covers "
        "infinite-stack hands that the production replay intentionally rejects.",
        "",
        "| Signal | Known cards | Unknown cards |",
        "|---|---:|---:|",
        f"| Acting perspectives | {known['perspectives']:,} | {unknown['perspectives']:,} |",
        f"| Folded | {percent(known['rates']['folded'])} | {percent(unknown['rates']['folded'])} |",
        f"| Acted postflop | {percent(known['rates']['acted_postflop'])} | {percent(unknown['rates']['acted_postflop'])} |",
        f"| Acted on turn or river | {percent(known['rates']['acted_turn_or_river'])} | {percent(unknown['rates']['acted_turn_or_river'])} |",
        f"| Has showdown/muck marker | {percent(known['rates']['showdown_marker'])} | {percent(unknown['rates']['showdown_marker'])} |",
        f"| Hand reached river | {percent(known['rates']['hand_reached_river'])} | {percent(unknown['rates']['hand_reached_river'])} |",
        "",
        "## Candidate distributions",
        "",
        f"Player counts: `{json.dumps(eligibility['trajectories_by_player_count'], sort_keys=True)}`",
        "",
        f"Starting-stack BB ranges: `{json.dumps(eligibility['trajectories_by_starting_stack_bb'], sort_keys=True)}`",
        "",
        f"Venues: `{json.dumps(eligibility['trajectories_by_venue'], sort_keys=True)}`",
        "",
        "## Replay failures",
        "",
        f"`{json.dumps(eligibility['replay_error_categories'], sort_keys=True)}`",
        "",
    ]
    return "\n".join(lines)


def audit_handhq_archive(
    zip_path: Path,
    output_json: Path,
    output_markdown: Path | None = None,
    *,
    max_members: int | None = None,
    max_member_bytes: int = 64 * 1024 * 1024,
    progress_every: int = 250,
    progress: Callable[[str], None] | None = print,
) -> dict[str, Any]:
    """Audit HandHQ one ZIP member at a time without extracting the archive."""
    zip_path = Path(zip_path)
    output_json = Path(output_json)
    output_markdown = (
        Path(output_markdown) if output_markdown else output_json.with_suffix(".md")
    )
    totals: Counter[str] = Counter()
    variants: Counter[str] = Counter()
    player_counts: Counter[int] = Counter()
    matrix: Counter[str] = Counter()
    parse_errors: Counter[str] = Counter()
    replay_errors: Counter[str] = Counter()
    cohort_metrics = {name: Counter() for name in ("known_cards", "unknown_cards")}
    cohort_actions = {name: Counter() for name in ("known_cards", "unknown_cards")}
    cohort_first = {name: Counter() for name in ("known_cards", "unknown_cards")}
    candidate_player_counts: Counter[int] = Counter()
    valid_player_counts: Counter[int] = Counter()
    valid_stack_buckets: Counter[str] = Counter()
    valid_venues: Counter[str] = Counter()
    zip_member_count = 0
    handhq_infos: list[zipfile.ZipInfo] = []

    with zipfile.ZipFile(zip_path) as archive:
        all_infos = archive.infolist()
        for info in all_infos:
            normalized = f"/{info.filename.lower().lstrip('/')}"
            if info.is_dir() or HANDHQ_SEGMENT not in normalized:
                continue
            if not info.filename.lower().endswith((".phh", ".phhs")):
                continue
            handhq_infos.append(info)
        if max_members is not None:
            handhq_infos = handhq_infos[:max_members]
        zip_member_count = len(all_infos)

        for member_index, info in enumerate(handhq_infos, 1):
            totals["members_audited"] += 1
            totals["compressed_bytes"] += info.compress_size
            totals["uncompressed_bytes"] += info.file_size
            if info.file_size == 0:
                totals["empty_members"] += 1
                continue
            try:
                hands = _iter_member_blocks(archive, info, max_member_bytes)
            except (HandParseError, UnicodeDecodeError, KeyError, zipfile.BadZipFile) as exc:
                totals["member_parse_errors"] += 1
                parse_errors[type(exc).__name__] += 1
                continue

            try:
                for hand_key, block in hands:
                    totals["hands_parsed"] += 1
                    variant_match = VARIANT_PATTERN.search(block)
                    variant = (
                        variant_match.group(1).upper() if variant_match else "UNKNOWN"
                    )
                    variants[variant] += 1
                    if variant != "NT":
                        continue
                    totals["nt_hands"] += 1
                    player_count, stacks = _fast_stacks(block)
                    player_counts[player_count] += 1
                    actions = _fast_actions(block)
                    if not 2 <= player_count <= 10 or actions is None:
                        totals["malformed_nt_hands"] += 1
                        continue

                    finite = stacks is not None
                    if finite:
                        totals["finite_stack_hands"] += 1
                    known_actors = _fast_known_actors(actions, player_count)
                    if known_actors:
                        totals["known_actor_hands"] += 1
                    matrix[
                        f"{'finite' if finite else 'nonfinite'}_"
                        f"{'known_actor' if known_actors else 'no_known_actor'}"
                    ] += 1

                    if known_actors:
                        known_cards, paths, showdown_players, reached_streets = (
                            _analyze_actions(actions, player_count)
                        )
                        for player, path in paths.items():
                            cohort = (
                                "known_cards"
                                if player in known_cards
                                else "unknown_cards"
                            )
                            metrics = cohort_metrics[cohort]
                            metrics["perspectives"] += 1
                            metrics["decisions"] += len(path)
                            codes = [code for _, code in path]
                            streets = {street for street, _ in path}
                            cohort_actions[cohort].update(codes)
                            cohort_first[cohort][codes[0]] += 1
                            metrics["folded"] += "f" in codes
                            metrics["acted_postflop"] += bool(
                                streets.intersection({"FLOP", "TURN", "RIVER"})
                            )
                            metrics["acted_turn_or_river"] += bool(
                                streets.intersection({"TURN", "RIVER"})
                            )
                            metrics["showdown_marker"] += player in showdown_players
                            metrics["hand_reached_flop"] += "FLOP" in reached_streets
                            metrics["hand_reached_river"] += "RIVER" in reached_streets

                    if not finite or not known_actors:
                        continue
                    totals["candidate_hands"] += 1
                    totals["candidate_perspectives"] += len(known_actors)
                    candidate_player_counts[player_count] += len(known_actors)
                    try:
                        hand = _parse_candidate_block(hand_key, block)
                        decisions = list(replay_hand(info.filename, hand_key, hand))
                        trajectories = build_hero_trajectories(decisions)
                    except (HandParseError, ArithmeticError, ValueError) as exc:
                        totals["replay_error_hands"] += 1
                        replay_errors[_replay_error_category(exc)] += 1
                        continue
                    if not trajectories:
                        totals["replay_no_trajectory_hands"] += 1
                        continue
                    totals["replay_valid_hands"] += 1
                    totals["replay_valid_trajectories"] += len(trajectories)
                    totals["replay_valid_decisions"] += sum(
                        item.decision_count for item in trajectories
                    )
                    big_blind = _big_blind(hand)
                    for trajectory in trajectories:
                        valid_player_counts[player_count] += 1
                        valid_venues[_fast_venue(block)] += 1
                        if big_blind is not None and big_blind > 0 and stacks is not None:
                            valid_stack_buckets[
                                range_label(stacks[trajectory.hero] / big_blind)
                            ] += 1
            except (HandParseError, UnicodeDecodeError, KeyError, zipfile.BadZipFile) as exc:
                totals["member_parse_errors"] += 1
                parse_errors[type(exc).__name__] += 1
                continue

            if progress and progress_every > 0 and (
                member_index % progress_every == 0
                or member_index == len(handhq_infos)
            ):
                progress(
                    f"HandHQ audit: {member_index:,}/{len(handhq_infos):,} members, "
                    f"{totals['hands_parsed']:,} hands, "
                    f"{totals['candidate_hands']:,} finite+known candidates"
                )

    report = {
        "schema_version": AUDIT_SCHEMA_VERSION,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "archive": {
            "path": str(zip_path.resolve()),
            "size_bytes": zip_path.stat().st_size,
            "zip_member_count": zip_member_count,
            "handhq_member_limit": max_members,
            "max_member_bytes": max_member_bytes,
        },
        "definitions": {
            "candidate_hand": (
                "NT hand with 2-10 players, finite nonnegative starting stacks for every "
                "seat, and at least one acting player with two known dealt cards"
            ),
            "replay_valid_trajectory": (
                "candidate player-perspective produced only after the existing exact replay "
                "accepts the complete hand"
            ),
            "showdown_marker": (
                "player has an sm, sd, kc, or k operation in the PHH action list"
            ),
            "selection_bias_cohort": (
                "acting players within NT hands that contain at least one acting player "
                "with known cards"
            ),
        },
        "totals": {
            key: totals[key]
            for key in (
                "members_audited",
                "empty_members",
                "member_parse_errors",
                "compressed_bytes",
                "uncompressed_bytes",
                "hands_parsed",
                "nt_hands",
                "malformed_nt_hands",
                "finite_stack_hands",
                "known_actor_hands",
            )
        },
        "hand_distributions": {
            "variants": _counter_dict(variants),
            "nt_player_counts": _counter_dict(player_counts),
            "finite_known_matrix": _counter_dict(matrix),
            "member_parse_error_types": _counter_dict(parse_errors),
        },
        "eligibility": {
            "candidate_hands": totals["candidate_hands"],
            "candidate_perspectives": totals["candidate_perspectives"],
            "candidate_perspectives_by_player_count": _counter_dict(
                candidate_player_counts
            ),
            "replay_valid_hands": totals["replay_valid_hands"],
            "replay_valid_trajectories": totals["replay_valid_trajectories"],
            "replay_valid_decisions": totals["replay_valid_decisions"],
            "replay_error_hands": totals["replay_error_hands"],
            "replay_no_trajectory_hands": totals["replay_no_trajectory_hands"],
            "replay_error_categories": _counter_dict(replay_errors),
            "trajectories_by_player_count": _counter_dict(valid_player_counts),
            "trajectories_by_starting_stack_bb": _counter_dict(valid_stack_buckets),
            "trajectories_by_venue": _counter_dict(valid_venues),
        },
        "selection_bias": {
            name: _cohort_report(
                cohort_metrics[name], cohort_actions[name], cohort_first[name]
            )
            for name in ("known_cards", "unknown_cards")
        },
    }
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    output_markdown.parent.mkdir(parents=True, exist_ok=True)
    output_markdown.write_text(_markdown(report), encoding="utf-8")
    return report
