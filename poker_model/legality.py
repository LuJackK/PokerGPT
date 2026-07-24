"""Map raw PokerGPT decision tokens into exact engine legality checks."""

from __future__ import annotations

from typing import Any, Mapping

from poker_pipeline.legality import (
    LegalityResult,
    LegalityState,
    RangeChipBounds,
    _decimal,
    range_token_chip_bounds,
    range_token_ratio_bounds,
)

from .decision import action_group, recover_five_way_action


def check_decision_token(
    token: str,
    state: LegalityState,
    meta: Mapping[str, Any],
) -> LegalityResult:
    """Interpret and check one raw model token without clamping it legal."""

    action = recover_five_way_action(
        token, to_call=state.to_call, current_bet=state.current_bet
    )
    if action_group(token) != "AGGRESSIVE":
        return state.check_action(action)
    if token == "ACTION_ALL_IN":
        contribution = state.remaining_stack
    else:
        try:
            ratio = _decimal(meta["range_representative_ratio"][token])
        except KeyError as exc:
            raise ValueError(f"no executable representative for {token}") from exc
        bounds = range_token_chip_bounds(token, state.pot)
        contribution = state.pot * ratio
        if not bounds.contains(contribution):
            raise ValueError(f"representative for {token} falls outside its range")
    return state.check_action(action, contribution)


__all__ = [
    "LegalityResult",
    "LegalityState",
    "RangeChipBounds",
    "check_decision_token",
    "range_token_chip_bounds",
    "range_token_ratio_bounds",
]
