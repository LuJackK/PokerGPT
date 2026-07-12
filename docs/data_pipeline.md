# Data pipeline design and findings

## Scope

The initial dataset is the 10,000-hand Pluribus corpus. Every selected hand is
six-player no-limit Texas Hold'em (`variant = "NT"`). The ACPC corpus is excluded
because it is mostly fixed-limit and contains paired raw/processed histories.
HandHQ remains an optional later expansion because files mix player counts and
most private cards are unknown.

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
Loss is applied only to the hero's action token and, for bets and raises, the two
sizing-range tokens. Opponent actions, forced posts, state fields, and board
reveals remain context with zero loss.

There is no `<HISTORY>` or `<EVENT_SEQUENCE>` wrapper: the trajectory itself is
the history. Board cards enter causally as events such as
`BOARD_REVEAL COUNT_3 CARD_Qs CARD_7h CARD_2c`. Poker street names are not tokens.

Before each hero decision, the sequence contains the observable pot, call amount,
and active/all-in player stack states. Players already known to have folded are
not repeated in later state observations because their fold events remain in the
trajectory.

Numerical values use compositional tokens. For example,
`POT_SIZE_BB RANGE_5_TO_10` reuses the same `RANGE_5_TO_10` token used by stack
and action-amount fields. This avoids a separate field-by-range vocabulary entry.
The fixed vocabulary has 111 tokens.

Legal actions are deliberately not encoded or stored. Replay checks source action
legality for data integrity, while later evaluation should measure and penalize
raw illegal model predictions with a poker engine.

## Context length and batching

Across all 10,000 Pluribus hands, preprocessing creates 58,942 complete hero
trajectories containing 91,356 supervised decisions. Length statistics are:

- median: 44 tokens;
- 95th percentile: 128 tokens;
- 99th percentile: 155 tokens;
- maximum: 218 tokens.

Every trajectory fits a 256-token context without truncation. Preprocessing fails
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
- `meta.pkl`: vocabulary, pad ID, block size, and format metadata.
- `stats.json`, `parse_errors.jsonl`, `audit_samples.jsonl`, and
  `preprocessing_manifest.json`: QA and reproducibility records.

Use `validate_artifacts.py` after preprocessing to verify framing, token ranges,
loss-mask placement, full-trajectory lengths, and split-group isolation.
