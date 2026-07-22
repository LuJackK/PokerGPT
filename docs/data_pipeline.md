# Data pipeline design and findings

## Scope

The initial dataset is the 10,000-hand Pluribus corpus. Every selected hand is
six-player no-limit Texas Hold'em (`variant = "NT"`). The ACPC corpus is excluded
because it is mostly fixed-limit and contains paired raw/processed histories.
HandHQ is excluded from supervised training: the full per-hand audit found no
hand with both finite table-wide starting stacks and known cards for an acting
player. WSOP contains only 83 files and is too small to complicate the first
baseline.

All 10,000 selected Pluribus hands have six 100-BB starting stacks, zero antes,
and 0.5/1-BB blinds. The chip values in the source happen to be 50/100, but all
encoded amounts are divided by the big blind. The model contract is therefore
six-max 100-BB play, not a literal chip denomination.

The source ZIP is never extracted. Manifesting reads a bounded header from each
member; preprocessing reads and parses one selected member at a time.

## Archive profile (July 2026 scan)

- 276,986 PHH/PHHS files: 10,088 `.phh` and 266,898 `.phhs`.
- 245,116 ACPC files, 21,782 HandHQ files, 10,000 Pluribus files, 83 WSOP files,
  and five standalone examples.
- One zero-byte HandHQ member is rejected as `empty_member`.
- A `.phh` file is one top-level TOML hand. A `.phhs` file contains TOML tables
  (`[1]`, `[2]`, etc.), normally many hands.

The bounded manifest header describes the first hand in a `.phhs` member. It is
adequate for archive reconnaissance, but HandHQ selection must ultimately be
verified per hand because player count can vary inside a file.

## HandHQ full per-hand audit (July 20, 2026)

`audit_handhq.py` streamed and released every HandHQ member and inspected all
21,606,087 NT hands. The eligibility census was conclusive:

- 15,609,342 hands had finite starting stacks for every seat and no known-card
  actor;
- 813,467 hands had at least one known-card actor and nonfinite stacks;
- 5,183,278 hands had neither finite table-wide stacks nor a known-card actor;
- zero hands combined finite table-wide stacks with a known-card actor.

Consequently, HandHQ produces zero replay-valid trajectories for the current
complete player-perspective supervised representation.

Known cards are also extremely selected. Within NT hands containing a known-card
actor, 1,177,710 known-card acting perspectives folded in 0.27% of hands and
acted postflop in 99.83%, compared with 84.73% folds and 20.71% postflop action
for 3,152,499 unknown-card acting perspectives in those same hands. Showdown/muck
markers appeared for 57.12% of known-card actors and 15.23% of unknown-card
actors. These rates rule out treating the known-card subset as a representative
sample even if stacks were reconstructed later without a separately justified
selection correction.

## Action replay

PokerKit PHH actions are state-dependent:

- `cc` becomes `CHECK` when the player owes nothing and `CALL` otherwise.
- `cbr N` is a street-total/raise-to value. Replay converts it to the incremental
  chip contribution, then classifies it as `BET` or `RAISE` from the current bet.
- `sm` and other showdown/muck operations are observed but are not decision
  targets.

Chip arithmetic uses `Decimal`. Amount features are bucketed both as incremental
chips divided by the big blind and incremental chips divided by the pot before
the action.

## Complete hero trajectories

The training unit is one complete causal trajectory for each player who makes a
decision in a hand. That hero is always `PLAYER_1`; other seats are numbered
clockwise. Only the hero's private cards are included. All public events before
the hero's final decision remain in chronological order, and opponent private
cards, future events, and showdown-only information are omitted.

A trajectory may contain several `<PLAYER_1_DECISION>` observations and targets.
Every hero decision produces exactly one supervised token: folds use
`ACTION_FOLD`, checks and calls use `ACTION_PASSIVE`, non-all-in bets and raises
use one pot-relative `RANGE_*`, and aggressive all-ins use `ACTION_ALL_IN`.
`RANGE_ZERO` is invalid as a hero target. Opponent actions remain explicit
`ACTION_CHECK`, `ACTION_CALL`, `ACTION_BET`, and `ACTION_RAISE` context tokens;
state fields and board reveals also have zero loss.

Every trajectory begins with exactly one of `POSITION_SMALL_BLIND`,
`POSITION_BIG_BLIND`, `POSITION_UTG`, `POSITION_HIJACK`, `POSITION_CUTOFF`, or
`POSITION_BUTTON`. Because seats are numbered clockwise relative to the hero,
this one token anchors every player's position. It replaces the constant
`TABLE_SIZE COUNT_6` prefix and both fixed blind-post events. Ante, straddle,
fixed blind-value, unused player-count, and `PLAYER_7` through `PLAYER_10` tokens
are absent from the fixed vocabulary.

There is no `<HISTORY>` or `<EVENT_SEQUENCE>` wrapper: the trajectory itself is
the history. Board cards enter causally as events such as
`BOARD_REVEAL COUNT_3 CARD_Qs CARD_7h CARD_2c`. Poker street names are not tokens.

Before each hero decision, the sequence repeats the hero's hole cards, followed
by the nonempty complete visible board, observable pot, call amount, and
active/all-in player states. Preflop omits the board field entirely; the absence
of any preceding `BOARD_REVEAL` already establishes that the board is empty.
Moving hero cards from the one-time header into each decision observation avoids
retaining a redundant header copy. Players already known to have folded are not
repeated because their fold events remain in the trajectory.

For example:

```text
<PLAYER_1_DECISION>
PLAYER_1_HOLE_CARDS CARD_As CARD_4s
CURRENT_BOARD COUNT_3 CARD_Qs CARD_5h CARD_7s
POT_SIZE_BB RANGE_10_TO_20
TO_CALL_BB RANGE_5_TO_10
PLAYER_1 STATUS_ACTIVE STACK_POT RANGE_3_TO_5
PLAYER_2 STATUS_ACTIVE STACK_POT RANGE_2_TO_3
```

There is no `<PLAYER_STATES>` delimiter. The state block begins deterministically
after the `TO_CALL_BB RANGE_*` pair. Active players carry `STACK_POT`, their exact
remaining stack divided by the current exact pot and then bucketed with the
shared ranges. `STATUS_ALL_IN` already specifies a zero remaining stack, so an
all-in player has no redundant `STACK_POT RANGE_ZERO` pair.

The full trial produced 309,561 active-player stack records. Unlike the former
BB buckets, their stack-to-pot values are distributed across strategically useful
ranges: 46.87% in 50-75, 28.27% in 20-50, 13.02% in 10-20, 7.51% in 5-10,
and 4.33% below 5. This retains player-specific all-in capacity while avoiding
the near-constant 75-100-BB feature.

Numerical values use compositional tokens. For example,
`POT_SIZE_BB RANGE_5_TO_10` reuses the same `RANGE_5_TO_10` token used by stack
fields and aggressive hero targets. This avoids a separate field-by-range
vocabulary entry.
Values above 50 big blinds retain strategically meaningful resolution through
`RANGE_50_TO_75`, `RANGE_75_TO_100`, `RANGE_100_TO_150`, and `RANGE_GT_150`
instead of collapsing every deep value into one bucket. After removing
`COUNT_0` and `<PLAYER_STATES>` and replacing `STACK_BB` with `STACK_POT`, the
fixed vocabulary contains 105 tokens.

For execution, `meta.pkl` stores one representative ratio for every valid sizing
output. The value is the exact median observed ratio in that bucket on the
training split. The full v0.7.0 audit found targets through `RANGE_10_TO_20` but
none in `RANGE_20_TO_50` or any deeper bucket. Those deeper range tokens remain
available as state and history context but are excluded from decision-logit
renormalization. Passive actions become check or call from `to_call`; aggressive
actions become bet or raise from `current_bet`.

Legal actions are deliberately not encoded or stored. Replay checks source action
legality for data integrity, while later evaluation should measure and penalize
raw illegal model predictions with a poker engine.

## Context length and batching

The full v0.8.0 streaming trial covered 58,942 complete hero trajectories and
91,356 supervised decisions. Relative to v0.7.0, position encoding, empty-board
omission, delimiter removal, and redundant all-in stack omission remove 628,088
tokens corpus-wide. The measured v0.8.0 lengths are:

- median: 36 tokens;
- 95th percentile: 144 tokens;
- 99th percentile: 175 tokens;
- maximum: 255 tokens.

The same schema was regenerated into the production `data/processed/` directory
on July 22, 2026 and passed full artifact validation with zero errors and zero
split-group overlap. The versioned transfer bundle
`artifacts/pokergpt-pluribus-v0.8.0.zip` contains the complete training payload in
997,224 bytes, so the raw 20 GB archive is not needed on the training machine.

Every trajectory still fits a 320-token context without truncation. Keep 320 for
the first baseline. Preprocessing fails
if a future trajectory exceeds `block_size`; it never discards early hand history.

`PokerTrajectoryDataset` uses `.idx` boundaries to load complete trajectories.
It creates shifted next-token inputs/targets, shifts the stored loss mask with the
targets, and right-pads within a batch. It never random-crops into the middle of a
hand or joins two hands into one attention context.

## Splitting

Pluribus hand numbers repeat across session folders. Train/validation assignment
groups by session directory, then uses deterministic size-aware group assignment.
All hero perspectives derived from the same hand therefore remain in one split.

## Binary format

- `train.bin` / `val.bin`: little-endian `uint16` token IDs containing concatenated
  complete trajectories.
- `*_loss_mask.bin`: aligned `uint8` masks with multiple hero targets per trajectory.
- `*.idx`: little-endian `uint64` starting token offset for every trajectory.
- `meta.pkl`: vocabulary, decision-token IDs, training-derived range
  representatives, pad ID, block size, and format metadata.
- `stats.json`: corpus summaries plus total, context, and supervised frequency
  for every vocabulary token.
- `parse_errors.jsonl`, `audit_samples.jsonl`, and `preprocessing_manifest.json`:
  QA and reproducibility records.

Use `validate_artifacts.py` after preprocessing to verify framing, token ranges,
exactly one valid supervised token per decision, rejection of `RANGE_ZERO` as a
hero target, full-trajectory lengths, token-frequency agreement, and split-group
isolation.
