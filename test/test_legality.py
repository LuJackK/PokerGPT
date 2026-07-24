from __future__ import annotations

import unittest
from decimal import Decimal

from poker_model.legality import check_decision_token
from poker_pipeline.legality import LegalityState, range_token_chip_bounds
from poker_pipeline.phh import HandParseError, parse_document, replay_hand
from poker_pipeline.tokenizer import (
    DECISION_TOKENS,
    PokerTokenizer,
    default_range_representatives,
)


def state(
    *,
    pot=30,
    to_call=10,
    current_bet=20,
    street_contribution=10,
    stack=90,
    minimum_bet=10,
    minimum_raise_increment=10,
    raise_reopened=True,
) -> LegalityState:
    return LegalityState(
        pot=Decimal(pot),
        to_call=Decimal(to_call),
        current_bet=Decimal(current_bet),
        street_contribution=Decimal(street_contribution),
        remaining_stack=Decimal(stack),
        minimum_bet=Decimal(minimum_bet),
        minimum_raise_increment=Decimal(minimum_raise_increment),
        raise_reopened=raise_reopened,
    )


class LegalityTests(unittest.TestCase):
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

    def test_check_call_mapping_and_fold_availability(self) -> None:
        facing_bet = state()
        self.assertEqual(
            facing_bet.legal_actions(), ("FOLD", "CALL", "RAISE")
        )
        self.assertTrue(
            check_decision_token("ACTION_PASSIVE", facing_bet, self.meta).legal
        )
        checked_to = state(
            pot=20,
            to_call=0,
            current_bet=0,
            street_contribution=0,
        )
        self.assertEqual(checked_to.legal_actions(), ("CHECK", "BET"))
        self.assertTrue(
            check_decision_token("ACTION_PASSIVE", checked_to, self.meta).legal
        )
        self.assertFalse(
            check_decision_token("ACTION_FOLD", checked_to, self.meta).legal
        )

    def test_bet_versus_raise_and_minimum_sizes(self) -> None:
        unopened = state(
            pot=20,
            to_call=0,
            current_bet=0,
            street_contribution=0,
            minimum_bet=10,
        )
        self.assertTrue(unopened.check_action("BET", 10).legal)
        self.assertFalse(unopened.check_action("BET", 9).legal)
        self.assertTrue(unopened.check_action("BET", 9,).reason is not None)

        raised = state()
        self.assertTrue(raised.check_action("RAISE", 20).legal)
        self.assertFalse(raised.check_action("RAISE", 19).legal)
        self.assertFalse(raised.check_action("BET", 20).legal)

    def test_short_all_in_and_action_reopening(self) -> None:
        short_stack = state(stack=15)
        result = short_stack.check_action("RAISE", 15)
        self.assertTrue(result.legal)
        self.assertTrue(result.size_legal)

        not_all_in = state(stack=90)
        self.assertFalse(not_all_in.check_action("RAISE", 15).legal)

        closed = state(stack=90, raise_reopened=False)
        self.assertEqual(closed.legal_actions(), ("FOLD", "CALL"))
        self.assertFalse(closed.check_action("RAISE", 90).legal)

    def test_all_in_token_is_raw_and_not_clamped_to_a_call(self) -> None:
        covered = state(to_call=30, current_bet=40, street_contribution=10, stack=20)
        result = check_decision_token("ACTION_ALL_IN", covered, self.meta)
        self.assertEqual(result.action, "RAISE")
        self.assertFalse(result.legal)
        self.assertIn("unavailable", result.reason)

    def test_range_token_to_chip_bound_conversion(self) -> None:
        bounds = range_token_chip_bounds("RANGE_0.5_TO_0.75", Decimal(80))
        self.assertEqual(bounds.lower, Decimal(40))
        self.assertEqual(bounds.upper, Decimal(60))
        self.assertFalse(bounds.contains(40))
        self.assertTrue(bounds.contains(Decimal("40.01")))
        self.assertTrue(bounds.contains(60))
        self.assertFalse(bounds.contains(Decimal("60.01")))

    def test_replay_rejects_undersized_non_all_in_raise(self) -> None:
        text = """\
variant = 'NT'
antes = [0, 0, 0, 0]
blinds_or_straddles = [5, 10, 0, 0]
min_bet = 10
starting_stacks = [100, 100, 100, 100]
actions = ['p3 cbr 20', 'p4 cbr 25']
"""
        hand = parse_document(text, "phh")[0][1]
        with self.assertRaisesRegex(HandParseError, "below the minimum"):
            list(replay_hand("fixture.phh", "1", hand))

    def test_short_all_in_does_not_reopen_but_unacted_player_can_raise(self) -> None:
        text = """\
variant = 'NT'
antes = [0, 0, 0, 0]
blinds_or_straddles = [5, 10, 0, 0]
min_bet = 10
starting_stacks = [25, 100, 100, 100]
actions = ['p3 cbr 20', 'p4 cc', 'p1 cbr 25', 'p2 cc', 'p3 cc']
"""
        hand = parse_document(text, "phh")[0][1]
        decisions = list(replay_hand("fixture.phh", "1", hand))
        player_two = decisions[3]
        player_three = decisions[4]
        self.assertIn("RAISE", player_two.legal_actions)
        self.assertTrue(player_two.raise_reopened)
        self.assertNotIn("RAISE", player_three.legal_actions)
        self.assertFalse(player_three.raise_reopened)


if __name__ == "__main__":
    unittest.main()
