from __future__ import annotations

import tempfile
import unittest
import zipfile
from pathlib import Path

from poker_pipeline.handhq_audit import audit_handhq_archive


FINITE = """\
[1]
variant = 'NT'
antes = [0, 0, 0]
blinds_or_straddles = [1, 2, 0]
min_bet = 2
starting_stacks = [40, 100, 300]
actions = ['d dh p1 AsKs', 'd dh p2 ????', 'd dh p3 ????', 'p3 f', 'p1 cbr 6', 'p2 f']
venue = 'Fixture Poker'
"""

NONFINITE = """\
[1]
variant = 'NT'
antes = [0, 0, 0]
blinds_or_straddles = [1, 2, 0]
min_bet = 2
starting_stacks = [inf, inf, inf]
actions = ['d dh p1 AhKh', 'd dh p2 ????', 'd dh p3 QsQd', 'p3 cc', 'p1 cc', 'p2 cc', 'd db 2c3c4c', 'p1 cc', 'p2 f', 'p3 cc', 'd db 5c', 'p1 cc', 'p3 cc', 'd db 6c', 'p1 cc', 'p3 cc', 'p1 sm ????', 'p3 sm ????']
venue = 'Fixture Poker'
"""


class HandHQAuditTests(unittest.TestCase):
    def test_streaming_audit_counts_eligibility_and_bias_cohorts(self) -> None:
        with tempfile.TemporaryDirectory(
            dir=Path(__file__).resolve().parent
        ) as temporary:
            root = Path(temporary)
            archive = root / "fixture.zip"
            with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as target:
                target.writestr("data/handhq/finite.phhs", FINITE)
                target.writestr("data/handhq/nonfinite.phhs", NONFINITE)
                target.writestr("data/pluribus/ignored.phhs", FINITE)
            output = root / "audit.json"
            report = audit_handhq_archive(archive, output, progress=None)

            self.assertEqual(report["totals"]["members_audited"], 2)
            self.assertEqual(report["totals"]["hands_parsed"], 2)
            self.assertEqual(report["totals"]["finite_stack_hands"], 1)
            self.assertEqual(report["eligibility"]["candidate_hands"], 1)
            self.assertEqual(report["eligibility"]["replay_valid_trajectories"], 1)
            self.assertEqual(report["eligibility"]["replay_valid_decisions"], 1)
            self.assertEqual(
                report["eligibility"]["trajectories_by_starting_stack_bb"],
                {"10_TO_20": 1},
            )
            known = report["selection_bias"]["known_cards"]
            unknown = report["selection_bias"]["unknown_cards"]
            self.assertEqual(known["perspectives"], 3)
            self.assertEqual(unknown["perspectives"], 3)
            self.assertGreater(
                known["rates"]["showdown_marker"],
                unknown["rates"]["showdown_marker"],
            )
            self.assertTrue(output.is_file())
            self.assertTrue(output.with_suffix(".md").is_file())


if __name__ == "__main__":
    unittest.main()
