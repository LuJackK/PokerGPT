from __future__ import annotations

import unittest
from decimal import Decimal

import torch

from poker_model.decision import decision_distribution
from poker_model.evaluation import DecisionMetrics
from poker_pipeline.legality import LegalityState
from poker_pipeline.tokenizer import (
    DECISION_TOKENS,
    PokerTokenizer,
    default_range_representatives,
)


class EvaluationLegalityTests(unittest.TestCase):
    def setUp(self) -> None:
        tokenizer = PokerTokenizer()
        self.meta = {
            "stoi": tokenizer.stoi,
            "decision_tokens": list(DECISION_TOKENS),
            "range_representative_ratio": {
                token: str(value)
                for token, value in default_range_representatives().items()
            },
        }

    def test_illegal_move_rate_combines_action_and_size_legality(self) -> None:
        logits = torch.zeros(len(PokerTokenizer().itos))
        distribution = decision_distribution(logits, self.meta)
        closed = LegalityState(
            pot=Decimal(30),
            to_call=Decimal(10),
            current_bet=Decimal(20),
            street_contribution=Decimal(10),
            remaining_stack=Decimal(90),
            minimum_bet=Decimal(10),
            minimum_raise_increment=Decimal(10),
            raise_reopened=False,
        )
        metrics = DecisionMetrics()
        metrics.update(
            distribution,
            predicted_token="ACTION_ALL_IN",
            truth_token="ACTION_PASSIVE",
            to_call=closed.to_call,
            current_bet=closed.current_bet,
            street="PREFLOP",
            meta=self.meta,
            legality_state=closed,
        )
        report = metrics.compute()
        self.assertEqual(report["illegal_move_rate"], 1.0)
        self.assertEqual(report["illegal_action_rate"], 1.0)
        self.assertIsNone(report["illegal_raw_size_rate"])

    def test_legal_short_all_in_counts_as_legal_move(self) -> None:
        logits = torch.zeros(len(PokerTokenizer().itos))
        distribution = decision_distribution(logits, self.meta)
        short = LegalityState(
            pot=Decimal(30),
            to_call=Decimal(10),
            current_bet=Decimal(20),
            street_contribution=Decimal(10),
            remaining_stack=Decimal(15),
            minimum_bet=Decimal(10),
            minimum_raise_increment=Decimal(10),
            raise_reopened=True,
        )
        metrics = DecisionMetrics()
        metrics.update(
            distribution,
            predicted_token="ACTION_ALL_IN",
            truth_token="ACTION_ALL_IN",
            to_call=short.to_call,
            current_bet=short.current_bet,
            street="PREFLOP",
            meta=self.meta,
            legality_state=short,
        )
        report = metrics.compute()
        self.assertEqual(report["illegal_move_rate"], 0.0)
        self.assertEqual(report["illegal_action_rate"], 0.0)
        self.assertEqual(report["illegal_raw_size_rate"], 0.0)


if __name__ == "__main__":
    unittest.main()
