from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from typing import Any, Mapping

import torch


ACTION_GROUPS = ("FOLD", "PASSIVE", "AGGRESSIVE")


def _decision_tokens(meta: Mapping[str, Any]) -> tuple[str, ...]:
    tokens = tuple(meta.get("decision_tokens", ()))
    required = {"ACTION_FOLD", "ACTION_PASSIVE", "ACTION_ALL_IN"}
    if not tokens or not required.issubset(tokens):
        raise ValueError("meta.pkl does not define the single-token decision vocabulary")
    if "RANGE_ZERO" in tokens:
        raise ValueError("RANGE_ZERO cannot be a decision token")
    return tokens


def action_group(token: str) -> str:
    if token == "ACTION_FOLD":
        return "FOLD"
    if token == "ACTION_PASSIVE":
        return "PASSIVE"
    if token == "ACTION_ALL_IN" or (
        token.startswith("RANGE_") and token != "RANGE_ZERO"
    ):
        return "AGGRESSIVE"
    raise ValueError(f"Not a hero decision token: {token}")


@dataclass(frozen=True)
class DecisionDistribution:
    tokens: tuple[str, ...]
    probabilities: torch.Tensor

    def probability(self, token: str) -> torch.Tensor:
        try:
            index = self.tokens.index(token)
        except ValueError as exc:
            raise KeyError(token) from exc
        return self.probabilities[..., index]

    def group_probabilities(self) -> torch.Tensor:
        grouped = []
        for group in ACTION_GROUPS:
            indexes = [
                index
                for index, token in enumerate(self.tokens)
                if action_group(token) == group
            ]
            grouped.append(self.probabilities[..., indexes].sum(dim=-1))
        return torch.stack(grouped, dim=-1)


def decision_distribution(
    logits: torch.Tensor, meta: Mapping[str, Any]
) -> DecisionDistribution:
    """Normalize only the fixed decision-token logits from one model position."""
    if logits.ndim < 1:
        raise ValueError("logits must have a vocabulary dimension")
    tokens = _decision_tokens(meta)
    stoi = meta["stoi"]
    try:
        ids = [int(stoi[token]) for token in tokens]
    except KeyError as exc:
        raise ValueError(f"Decision token is missing from vocabulary: {exc.args[0]}") from exc
    if max(ids) >= logits.shape[-1]:
        raise ValueError("Decision token ID is outside the logits vocabulary dimension")
    index = torch.tensor(ids, dtype=torch.long, device=logits.device)
    selected = logits.index_select(-1, index)
    return DecisionDistribution(tokens=tokens, probabilities=torch.softmax(selected, dim=-1))


def grouped_greedy_token(logits: torch.Tensor, meta: Mapping[str, Any]) -> str:
    """Choose a group greedily, then the best token inside the winning group."""
    if logits.ndim != 1:
        raise ValueError("grouped_greedy_token expects one vocabulary-logit vector")
    distribution = decision_distribution(logits, meta)
    group = ACTION_GROUPS[int(distribution.group_probabilities().argmax().item())]
    if group == "FOLD":
        return "ACTION_FOLD"
    if group == "PASSIVE":
        return "ACTION_PASSIVE"
    aggressive_indexes = [
        index
        for index, token in enumerate(distribution.tokens)
        if action_group(token) == "AGGRESSIVE"
    ]
    local = distribution.probabilities[aggressive_indexes]
    return distribution.tokens[aggressive_indexes[int(local.argmax().item())]]


def sample_decision_token(
    logits: torch.Tensor,
    meta: Mapping[str, Any],
    *,
    generator: torch.Generator | None = None,
) -> str:
    """Sample one token directly from the normalized decision distribution."""
    if logits.ndim != 1:
        raise ValueError("sample_decision_token expects one vocabulary-logit vector")
    distribution = decision_distribution(logits, meta)
    index = int(torch.multinomial(distribution.probabilities, 1, generator=generator).item())
    return distribution.tokens[index]


def predict_decision_token(
    model: Any,
    idx: torch.Tensor,
    meta: Mapping[str, Any],
    *,
    grouped_greedy: bool = True,
    generator: torch.Generator | None = None,
) -> str:
    """Run one model forward pass and return exactly one hero decision token."""
    if idx.ndim != 2 or idx.shape[0] != 1:
        raise ValueError(
            "predict_decision_token currently expects a batch of one trajectory"
        )
    with torch.no_grad():
        logits, _ = model(idx)
    final_logits = logits[0, -1]
    if grouped_greedy:
        return grouped_greedy_token(final_logits, meta)
    return sample_decision_token(final_logits, meta, generator=generator)


@dataclass(frozen=True)
class InterpretedDecision:
    action: str
    raw_amount: Decimal | None = None


def recover_five_way_action(
    token: str,
    *,
    to_call: Decimal | int | float,
    current_bet: Decimal | int | float,
) -> str:
    """Recover FOLD/CHECK/CALL/BET/RAISE from a compressed hero token."""
    group = action_group(token)
    if group == "FOLD":
        return "FOLD"
    if group == "PASSIVE":
        return "CALL" if Decimal(str(to_call)) > 0 else "CHECK"
    return "RAISE" if Decimal(str(current_bet)) > 0 else "BET"


def interpret_decision_token(
    token: str,
    *,
    pot: Decimal | int | float,
    to_call: Decimal | int | float,
    current_bet: Decimal | int | float,
    remaining_stack: Decimal | int | float,
    meta: Mapping[str, Any],
) -> InterpretedDecision:
    """Convert one raw model token into an executable, unclamped engine decision."""
    action = recover_five_way_action(token, to_call=to_call, current_bet=current_bet)
    if token == "ACTION_ALL_IN":
        return InterpretedDecision(f"{action}_ALL_IN", Decimal(str(remaining_stack)))
    if token.startswith("RANGE_"):
        try:
            ratio = Decimal(str(meta["range_representative_ratio"][token]))
        except KeyError as exc:
            raise ValueError(f"No executable representative for {token}") from exc
        return InterpretedDecision(action, Decimal(str(pot)) * ratio)
    return InterpretedDecision(action)
