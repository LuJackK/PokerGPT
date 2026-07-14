from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from decimal import Decimal
from typing import Any, Mapping

from .decision import (
    ACTION_GROUPS,
    DecisionDistribution,
    action_group,
    recover_five_way_action,
)


@dataclass
class DecisionMetrics:
    """Accumulate token, grouped-action, sizing, legality, and street metrics."""

    total: int = 0
    action_correct: int = 0
    action_top_k_correct: int = 0
    predicted_action_confidence_total: float = 0.0
    truth_action_probability_total: float = 0.0
    aggressive_ground_truth: int = 0
    sizing_correct: int = 0
    ratio_error_total: Decimal = Decimal(0)
    ratio_error_count: int = 0
    illegal_actions: int = 0
    illegal_raw_sizes: int = 0
    action_legality_evaluated: int = 0
    raw_size_legality_evaluated: int = 0
    confusion: Counter[tuple[str, str]] = field(default_factory=Counter)
    street_total: Counter[str] = field(default_factory=Counter)
    street_correct: Counter[str] = field(default_factory=Counter)

    def update(
        self,
        distribution: DecisionDistribution,
        *,
        predicted_token: str,
        truth_token: str,
        to_call: Decimal | int | float,
        current_bet: Decimal | int | float,
        street: str,
        meta: Mapping[str, Any],
        top_k: int = 2,
        raw_action_legal: bool | None = None,
        raw_size_legal: bool | None = None,
    ) -> None:
        truth_group = action_group(truth_token)
        predicted_group = action_group(predicted_token)
        self.total += 1
        self.street_total[street] += 1
        if truth_group == predicted_group:
            self.action_correct += 1
            self.street_correct[street] += 1

        group_probabilities = distribution.group_probabilities()
        if group_probabilities.ndim != 1:
            raise ValueError("DecisionMetrics.update expects one decision distribution")
        predicted_group_index = ACTION_GROUPS.index(predicted_group)
        truth_group_index = ACTION_GROUPS.index(truth_group)
        self.predicted_action_confidence_total += float(
            group_probabilities[predicted_group_index].item()
        )
        self.truth_action_probability_total += float(
            group_probabilities[truth_group_index].item()
        )
        k = min(max(int(top_k), 1), len(ACTION_GROUPS))
        top_groups = {
            ACTION_GROUPS[index]
            for index in group_probabilities.topk(k).indices.tolist()
        }
        if truth_group in top_groups:
            self.action_top_k_correct += 1

        truth_action = recover_five_way_action(
            truth_token, to_call=to_call, current_bet=current_bet
        )
        predicted_action = recover_five_way_action(
            predicted_token, to_call=to_call, current_bet=current_bet
        )
        self.confusion[(truth_action, predicted_action)] += 1

        if truth_group == "AGGRESSIVE":
            self.aggressive_ground_truth += 1
            if predicted_token == truth_token:
                self.sizing_correct += 1
            representatives = meta["range_representative_ratio"]
            if truth_token in representatives and predicted_token in representatives:
                truth_ratio = Decimal(str(representatives[truth_token]))
                predicted_ratio = Decimal(str(representatives[predicted_token]))
                self.ratio_error_total += abs(predicted_ratio - truth_ratio)
                self.ratio_error_count += 1

        if raw_action_legal is not None:
            self.action_legality_evaluated += 1
            if not raw_action_legal:
                self.illegal_actions += 1
        if predicted_group == "AGGRESSIVE" and raw_size_legal is not None:
            self.raw_size_legality_evaluated += 1
            if not raw_size_legal:
                self.illegal_raw_sizes += 1

    def compute(self) -> dict[str, Any]:
        def rate(numerator: int, denominator: int) -> float:
            return numerator / denominator if denominator else 0.0

        return {
            "decisions": self.total,
            "action_accuracy": rate(self.action_correct, self.total),
            "action_top_k_accuracy": rate(self.action_top_k_correct, self.total),
            "mean_predicted_action_confidence": (
                self.predicted_action_confidence_total / self.total if self.total else None
            ),
            "mean_truth_action_probability": (
                self.truth_action_probability_total / self.total if self.total else None
            ),
            "sizing_accuracy": rate(self.sizing_correct, self.aggressive_ground_truth),
            "mean_absolute_ratio_error": (
                float(self.ratio_error_total / self.ratio_error_count)
                if self.ratio_error_count
                else None
            ),
            "illegal_action_rate": (
                rate(self.illegal_actions, self.action_legality_evaluated)
                if self.action_legality_evaluated
                else None
            ),
            "illegal_raw_size_rate": (
                rate(self.illegal_raw_sizes, self.raw_size_legality_evaluated)
                if self.raw_size_legality_evaluated
                else None
            ),
            "five_way_confusion": {
                f"{truth}->{predicted}": count
                for (truth, predicted), count in sorted(self.confusion.items())
            },
            "street_action_accuracy": {
                street: rate(self.street_correct[street], count)
                for street, count in sorted(self.street_total.items())
            },
        }
