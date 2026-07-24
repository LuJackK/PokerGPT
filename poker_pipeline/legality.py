"""Exact no-limit Hold'em betting legality independent of model dependencies."""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal, InvalidOperation


def _decimal(value: Decimal | int | float | str) -> Decimal:
    try:
        result = value if isinstance(value, Decimal) else Decimal(str(value))
    except (InvalidOperation, ValueError) as exc:
        raise ValueError(f"invalid chip amount: {value!r}") from exc
    if not result.is_finite() or result < 0:
        raise ValueError(f"invalid chip amount: {value!r}")
    return result


@dataclass(frozen=True)
class RangeChipBounds:
    """Pot-relative token bounds converted to chips.

    RANGE buckets are open on the lower edge and closed on the upper edge,
    matching ``poker_pipeline.tokenizer.range_label``.
    """

    lower: Decimal
    upper: Decimal | None
    lower_inclusive: bool = False
    upper_inclusive: bool = True

    def contains(self, amount: Decimal | int | float | str) -> bool:
        value = _decimal(amount)
        lower_ok = value >= self.lower if self.lower_inclusive else value > self.lower
        upper_ok = (
            True
            if self.upper is None
            else value <= self.upper
            if self.upper_inclusive
            else value < self.upper
        )
        return lower_ok and upper_ok


def range_token_ratio_bounds(token: str) -> RangeChipBounds:
    """Return the exact ratio interval represented by one RANGE token."""

    if not token.startswith("RANGE_"):
        raise ValueError(f"not a range token: {token!r}")
    label = token.removeprefix("RANGE_")
    if label == "GT_150":
        return RangeChipBounds(Decimal(150), None)
    if "_TO_" not in label:
        raise ValueError(f"unsupported range token: {token!r}")
    lower_text, upper_text = label.split("_TO_", 1)
    try:
        lower = Decimal(lower_text)
        upper = Decimal(upper_text)
    except InvalidOperation as exc:
        raise ValueError(f"unsupported range token: {token!r}") from exc
    if lower < 0 or upper <= lower:
        raise ValueError(f"invalid range token bounds: {token!r}")
    return RangeChipBounds(lower, upper)


def range_token_chip_bounds(
    token: str, pot: Decimal | int | float | str
) -> RangeChipBounds:
    """Convert a pot-relative RANGE token interval to exact chip bounds."""

    pot_value = _decimal(pot)
    if pot_value <= 0:
        raise ValueError("pot must be positive")
    ratio = range_token_ratio_bounds(token)
    return RangeChipBounds(
        lower=ratio.lower * pot_value,
        upper=None if ratio.upper is None else ratio.upper * pot_value,
        lower_inclusive=ratio.lower_inclusive,
        upper_inclusive=ratio.upper_inclusive,
    )


@dataclass(frozen=True)
class LegalityState:
    """All exact public betting state needed to check one player's decision."""

    pot: Decimal
    to_call: Decimal
    current_bet: Decimal
    street_contribution: Decimal
    remaining_stack: Decimal
    minimum_bet: Decimal
    minimum_raise_increment: Decimal
    raise_reopened: bool

    def __post_init__(self) -> None:
        for field_name in (
            "pot",
            "to_call",
            "current_bet",
            "street_contribution",
            "remaining_stack",
            "minimum_bet",
            "minimum_raise_increment",
        ):
            object.__setattr__(self, field_name, _decimal(getattr(self, field_name)))
        if self.pot <= 0:
            raise ValueError("pot must be positive")
        if self.minimum_bet <= 0 or self.minimum_raise_increment <= 0:
            raise ValueError("minimum bet and raise increment must be positive")
        expected_to_call = max(Decimal(0), self.current_bet - self.street_contribution)
        if self.to_call != expected_to_call:
            raise ValueError(
                f"to_call {self.to_call} disagrees with current bet/contribution "
                f"({expected_to_call})"
            )

    @property
    def minimum_raise_contribution(self) -> Decimal:
        """Minimum incremental contribution for a full raise by this player."""

        return self.to_call + self.minimum_raise_increment

    def legal_actions(self) -> tuple[str, ...]:
        if self.remaining_stack <= 0:
            return ()
        if self.to_call > 0:
            legal = ["FOLD", "CALL"]
        else:
            legal = ["CHECK"]
        if self.current_bet == 0:
            legal.append("BET")
        elif (
            self.raise_reopened
            and self.remaining_stack > self.to_call
        ):
            legal.append("RAISE")
        return tuple(legal)

    def check_action(
        self, action: str, contribution: Decimal | int | float | str | None = None
    ) -> "LegalityResult":
        """Check a five-way action and its raw incremental chip contribution."""

        normalized = action.upper()
        actions = self.legal_actions()
        action_legal = normalized in actions
        if normalized not in {"BET", "RAISE"}:
            if contribution is not None and _decimal(contribution) != 0:
                return LegalityResult(
                    action=normalized,
                    contribution=_decimal(contribution),
                    action_legal=action_legal,
                    size_legal=False,
                    legal=action_legal,
                    reason="non-aggressive action carries a chip size",
                )
            reason = None if action_legal else f"{normalized} is unavailable"
            return LegalityResult(
                action=normalized,
                contribution=None,
                action_legal=action_legal,
                size_legal=None,
                legal=action_legal,
                reason=reason,
            )

        if contribution is None:
            return LegalityResult(
                action=normalized,
                contribution=None,
                action_legal=action_legal,
                size_legal=False,
                legal=False,
                reason="aggressive action is missing a chip size",
            )
        amount = _decimal(contribution)
        if not action_legal:
            return LegalityResult(
                action=normalized,
                contribution=amount,
                action_legal=False,
                size_legal=None,
                legal=False,
                reason=f"{normalized} is unavailable",
            )
        if amount <= 0:
            size_legal = False
            reason = "aggressive contribution must be positive"
        elif amount > self.remaining_stack:
            size_legal = False
            reason = "contribution exceeds the remaining stack"
        elif normalized == "BET":
            size_legal = amount >= self.minimum_bet or amount == self.remaining_stack
            reason = (
                None
                if size_legal
                else "bet is below the minimum and is not an all-in"
            )
        else:
            if amount <= self.to_call:
                size_legal = False
                reason = "raise does not exceed the call amount"
            else:
                minimum = self.minimum_raise_contribution
                size_legal = amount >= minimum or amount == self.remaining_stack
                reason = (
                    None
                    if size_legal
                    else "raise is below the minimum and is not an all-in"
                )
        return LegalityResult(
            action=normalized,
            contribution=amount,
            action_legal=True,
            size_legal=size_legal,
            legal=size_legal,
            reason=reason,
        )


@dataclass(frozen=True)
class LegalityResult:
    action: str
    contribution: Decimal | None
    action_legal: bool
    size_legal: bool | None
    legal: bool
    reason: str | None
