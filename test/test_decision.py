from __future__ import annotations

import unittest
from decimal import Decimal

try:
    import torch
except ImportError:
    torch = None

from poker_pipeline.tokenizer import (
    DECISION_TOKENS,
    RANGE_TOKENS,
    PokerTokenizer,
    default_range_representatives,
)


@unittest.skipIf(torch is None, "PyTorch is not installed")
class DecisionTests(unittest.TestCase):
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
        self.logits = torch.full((len(tokenizer.itos),), -20.0)

    def test_decision_logits_are_renormalized_and_grouped_for_greedy_play(self) -> None:
        from poker_model.decision import decision_distribution, grouped_greedy_token

        self.logits[self.meta["stoi"]["ACTION_FOLD"]] = 2.0
        self.logits[self.meta["stoi"]["ACTION_PASSIVE"]] = 2.1
        for token in RANGE_TOKENS:
            self.logits[self.meta["stoi"][token]] = 0.0
        self.logits[self.meta["stoi"]["ACTION_ALL_IN"]] = 1.0
        distribution = decision_distribution(self.logits, self.meta)
        self.assertTrue(torch.allclose(distribution.probabilities.sum(), torch.tensor(1.0)))
        self.assertEqual(grouped_greedy_token(self.logits, self.meta), "ACTION_ALL_IN")

    def test_engine_interpretation_uses_current_bet_for_aggression(self) -> None:
        from poker_model.decision import interpret_decision_token, recover_five_way_action

        self.assertEqual(
            recover_five_way_action(
                "RANGE_0.5_TO_0.75", to_call=Decimal(0), current_bet=Decimal(2)
            ),
            "RAISE",
        )
        self.assertEqual(
            recover_five_way_action(
                "ACTION_PASSIVE", to_call=Decimal(0), current_bet=Decimal(2)
            ),
            "CHECK",
        )
        interpreted = interpret_decision_token(
            "RANGE_0.5_TO_0.75",
            pot=Decimal(10),
            to_call=Decimal(0),
            current_bet=Decimal(2),
            remaining_stack=Decimal(50),
            meta=self.meta,
        )
        self.assertEqual(interpreted.action, "RAISE")
        self.assertEqual(interpreted.raw_amount, Decimal("6.25"))


if __name__ == "__main__":
    unittest.main()
