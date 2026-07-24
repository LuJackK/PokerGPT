from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from decimal import Decimal
import math
from typing import Any, Mapping

from .decision import (
    ACTION_GROUPS,
    DecisionDistribution,
    action_group,
    recover_five_way_action,
)
from .legality import LegalityState, check_decision_token
from poker_pipeline.legality import range_token_ratio_bounds


FIVE_WAY_ACTIONS = ("FOLD", "CHECK", "CALL", "BET", "RAISE")


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
    illegal_moves: int = 0
    legality_evaluated: int = 0
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
        legality_state: LegalityState | None = None,
    ) -> None:
        if legality_state is not None:
            if raw_action_legal is not None or raw_size_legal is not None:
                raise ValueError(
                    "provide legality_state or precomputed legality flags, not both"
                )
            legality = check_decision_token(predicted_token, legality_state, meta)
            raw_action_legal = legality.action_legal
            raw_size_legal = legality.size_legal
            self.legality_evaluated += 1
            if not legality.legal:
                self.illegal_moves += 1
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
            "illegal_move_rate": (
                rate(self.illegal_moves, self.legality_evaluated)
                if self.legality_evaluated
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


@dataclass
class _MetricSlice:
    decisions: int = 0
    decision_correct: int = 0
    decision_top_k_correct: int = 0
    mapped_action_correct: int = 0
    mapped_action_top_k_correct: int = 0
    illegal_moves: int = 0
    truth_aggressive: int = 0
    sizing_range_errors_overall: int = 0
    aggressive_action_correct: int = 0
    sizing_range_errors_conditional: int = 0
    conditional_ratio_absolute_error_sum: Decimal = Decimal(0)
    conditional_interval_distance_sum: Decimal = Decimal(0)

    def update(
        self,
        *,
        decision_correct: bool,
        decision_top_k_correct: bool,
        action_correct: bool,
        action_top_k_correct: bool,
        illegal: bool,
        truth_aggressive: bool,
        sizing_error: bool,
        aggressive_action_correct: bool,
        ratio_absolute_error: Decimal | None,
        interval_distance: Decimal | None,
    ) -> None:
        self.decisions += 1
        self.decision_correct += int(decision_correct)
        self.decision_top_k_correct += int(decision_top_k_correct)
        self.mapped_action_correct += int(action_correct)
        self.mapped_action_top_k_correct += int(action_top_k_correct)
        self.illegal_moves += int(illegal)
        if truth_aggressive:
            self.truth_aggressive += 1
            self.sizing_range_errors_overall += int(sizing_error)
        if aggressive_action_correct:
            self.aggressive_action_correct += 1
            self.sizing_range_errors_conditional += int(sizing_error)
            assert ratio_absolute_error is not None
            assert interval_distance is not None
            self.conditional_ratio_absolute_error_sum += ratio_absolute_error
            self.conditional_interval_distance_sum += interval_distance

    @staticmethod
    def _rate(numerator: int, denominator: int) -> float | None:
        return numerator / denominator if denominator else None

    def compute(self) -> dict[str, Any]:
        return {
            "decisions": self.decisions,
            "joint_decision_token_accuracy": self._rate(
                self.decision_correct, self.decisions
            ),
            "joint_decision_token_top_k_accuracy": self._rate(
                self.decision_top_k_correct, self.decisions
            ),
            "mapped_action_accuracy": self._rate(
                self.mapped_action_correct, self.decisions
            ),
            "mapped_action_top_k_accuracy": self._rate(
                self.mapped_action_top_k_correct, self.decisions
            ),
            "illegal_move_rate": self._rate(self.illegal_moves, self.decisions),
            "illegal_moves": self.illegal_moves,
            "sizing": {
                "ground_truth_aggressive_decisions": self.truth_aggressive,
                "range_error_rate_overall": self._rate(
                    self.sizing_range_errors_overall, self.truth_aggressive
                ),
                "aggressive_action_correct_decisions": self.aggressive_action_correct,
                "range_error_rate_conditional_on_aggressive_action_correct": self._rate(
                    self.sizing_range_errors_conditional,
                    self.aggressive_action_correct,
                ),
                "representative_ratio_mae_conditional_on_aggressive_action_correct": (
                    float(
                        self.conditional_ratio_absolute_error_sum
                        / self.aggressive_action_correct
                    )
                    if self.aggressive_action_correct
                    else None
                ),
                "range_interval_distance_mae_conditional_on_aggressive_action_correct": (
                    float(
                        self.conditional_interval_distance_sum
                        / self.aggressive_action_correct
                    )
                    if self.aggressive_action_correct
                    else None
                ),
            },
        }


class FrozenEvaluationMetrics:
    """The frozen v1 evaluator metric contract.

    The primary raw prediction is the argmax after normalizing only decision
    tokens. The mapped action is the deterministic five-way interpretation of
    that same raw token. Top-k action probabilities sum all decision-token
    probabilities that map to each five-way action in the exact replay state.
    """

    def __init__(
        self,
        meta: Mapping[str, Any],
        *,
        decision_top_k: int,
        action_top_k: int,
    ) -> None:
        self.meta = meta
        self.tokens = tuple(meta["decision_tokens"])
        if not 1 <= decision_top_k <= len(self.tokens):
            raise ValueError("decision_top_k is outside the decision vocabulary")
        if not 1 <= action_top_k <= len(FIVE_WAY_ACTIONS):
            raise ValueError("action_top_k is outside the five-way action set")
        self.decision_top_k = decision_top_k
        self.action_top_k = action_top_k
        self.overall = _MetricSlice()
        self.by_street: dict[str, _MetricSlice] = {}
        self.action_confusion: Counter[tuple[str, str]] = Counter()
        self.decision_confusion: Counter[tuple[str, str]] = Counter()
        self.illegal_reason_counts: Counter[str] = Counter()
        self.illegal_token_counts: Counter[str] = Counter()
        self.cross_entropy_sum = 0.0

    def _predicted_ratio(self, token: str, state: LegalityState) -> Decimal:
        if token == "ACTION_ALL_IN":
            return state.remaining_stack / state.pot
        return Decimal(str(self.meta["range_representative_ratio"][token]))

    @staticmethod
    def _interval_distance(token: str, truth_ratio: Decimal, predicted_ratio: Decimal) -> Decimal:
        if token == "ACTION_ALL_IN":
            return abs(predicted_ratio - truth_ratio)
        bounds = range_token_ratio_bounds(token)
        if truth_ratio <= bounds.lower:
            return bounds.lower - truth_ratio
        if bounds.upper is not None and truth_ratio > bounds.upper:
            return truth_ratio - bounds.upper
        return Decimal(0)

    def update(
        self,
        distribution: DecisionDistribution,
        *,
        truth_token: str,
        truth_ratio: Decimal | None,
        street: str,
        legality_state: LegalityState,
    ) -> None:
        if distribution.probabilities.ndim != 1:
            raise ValueError("frozen evaluation expects one decision distribution")
        if distribution.tokens != self.tokens:
            raise ValueError("decision distribution vocabulary does not match metadata")
        probabilities = distribution.probabilities.detach().float().cpu()
        predicted_index = int(probabilities.argmax().item())
        predicted_token = self.tokens[predicted_index]
        truth_index = self.tokens.index(truth_token)
        top_tokens = set(
            probabilities.topk(self.decision_top_k).indices.tolist()
        )
        truth_action = recover_five_way_action(
            truth_token,
            to_call=legality_state.to_call,
            current_bet=legality_state.current_bet,
        )
        predicted_action = recover_five_way_action(
            predicted_token,
            to_call=legality_state.to_call,
            current_bet=legality_state.current_bet,
        )
        action_probabilities = {action: 0.0 for action in FIVE_WAY_ACTIONS}
        for token, probability in zip(self.tokens, probabilities.tolist()):
            mapped = recover_five_way_action(
                token,
                to_call=legality_state.to_call,
                current_bet=legality_state.current_bet,
            )
            action_probabilities[mapped] += probability
        top_actions = {
            action
            for action, _ in sorted(
                action_probabilities.items(),
                key=lambda item: (-item[1], FIVE_WAY_ACTIONS.index(item[0])),
            )[: self.action_top_k]
        }
        legality = check_decision_token(predicted_token, legality_state, self.meta)
        if not legality.legal:
            self.illegal_reason_counts[legality.reason or "unspecified"] += 1
            self.illegal_token_counts[predicted_token] += 1
        truth_aggressive = action_group(truth_token) == "AGGRESSIVE"
        sizing_error = truth_aggressive and predicted_token != truth_token
        aggressive_action_correct = (
            truth_aggressive and predicted_action == truth_action
        )
        ratio_error: Decimal | None = None
        interval_distance: Decimal | None = None
        if aggressive_action_correct:
            assert truth_ratio is not None
            predicted_ratio = self._predicted_ratio(predicted_token, legality_state)
            ratio_error = abs(predicted_ratio - truth_ratio)
            interval_distance = self._interval_distance(
                predicted_token, truth_ratio, predicted_ratio
            )
        outcome = {
            "decision_correct": predicted_token == truth_token,
            "decision_top_k_correct": truth_index in top_tokens,
            "action_correct": predicted_action == truth_action,
            "action_top_k_correct": truth_action in top_actions,
            "illegal": not legality.legal,
            "truth_aggressive": truth_aggressive,
            "sizing_error": sizing_error,
            "aggressive_action_correct": aggressive_action_correct,
            "ratio_absolute_error": ratio_error,
            "interval_distance": interval_distance,
        }
        self.overall.update(**outcome)
        self.by_street.setdefault(street, _MetricSlice()).update(**outcome)
        self.action_confusion[(truth_action, predicted_action)] += 1
        self.decision_confusion[(truth_token, predicted_token)] += 1
        probability = max(float(probabilities[truth_index].item()), 1e-45)
        self.cross_entropy_sum += -math.log(probability)

    @staticmethod
    def _matrix(
        labels: tuple[str, ...], counts: Counter[tuple[str, str]]
    ) -> dict[str, dict[str, int]]:
        return {
            truth: {
                predicted: counts[(truth, predicted)] for predicted in labels
            }
            for truth in labels
        }

    def compute(self) -> dict[str, Any]:
        report = self.overall.compute()
        report.update(
            {
                "decision_top_k": self.decision_top_k,
                "mapped_action_top_k": self.action_top_k,
                "decision_cross_entropy": (
                    self.cross_entropy_sum / self.overall.decisions
                    if self.overall.decisions
                    else None
                ),
                "per_street": {
                    street: metrics.compute()
                    for street, metrics in sorted(self.by_street.items())
                },
                "action_confusion_matrix": self._matrix(
                    FIVE_WAY_ACTIONS, self.action_confusion
                ),
                "illegal_move_breakdown": {
                    "by_reason": dict(sorted(self.illegal_reason_counts.items())),
                    "by_predicted_token": dict(
                        sorted(self.illegal_token_counts.items())
                    ),
                },
                "decision_token_confusion_matrix": self._matrix(
                    self.tokens, self.decision_confusion
                ),
            }
        )
        return report
