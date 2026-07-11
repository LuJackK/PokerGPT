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

## Information boundary

Each player action creates a separate example. The acting player is always
`PLAYER_1`; the other seats are numbered clockwise relative to that player. The
example contains only `PLAYER_1`'s private cards. Opponent private-card tokens are
omitted entirely, even though Pluribus source files list all six hands at the
start. Future board cards, future actions, and showdown information are excluded.

Poker street names are not encoded. Current public state is represented by
`BOARD_COUNT_n` followed by the visible board cards. A chronological
`<EVENT_SEQUENCE>` contains forced posts, prior player actions, and
`BOARD_REVEAL_n` boundaries, which preserves when betting rounds changed without
tokens such as `PREFLOP`, `FLOP`, `TURN`, or `RIVER`.

Forced posts identify blind positions through observable events, for example
`PLAYER_3 POST_BLIND BLIND_BB_0.5`. State features distinguish shared state
(`POT_SIZE_BB_*`) from player-specific state (`PLAYER_1_TO_CALL_BB_*` and
`PLAYER_1_STACK_BB_*`). Bucket labels are non-overlapping ranges.

Legal actions are deliberately not encoded or stored: later evaluation should
measure and penalize illegal model predictions with a poker engine. The
preprocessor still checks that every observed source action is legal as a
data-integrity assertion.

## Splitting

Pluribus hand numbers repeat across session folders. Train/validation assignment
therefore groups by session directory, then uses deterministic size-aware group
assignment to approximate the requested validation fraction without putting a
session in both splits.

## Binary format

- `train.bin` / `val.bin`: little-endian `uint16` token IDs.
- `*_loss_mask.bin`: aligned `uint8` masks. Action targets are supervised; bet and
  raise examples also supervise the two amount-bucket tokens.
- `*.idx`: little-endian `uint64` token offsets, one per example.
- `meta.pkl`: fixed vocabulary and format metadata.
- `stats.json`, `parse_errors.jsonl`, `audit_samples.jsonl`, and
  `preprocessing_manifest.json`: QA and reproducibility records.

Use `validate_artifacts.py` after preprocessing to verify framing, lengths, token
ranges, loss-mask placement, and split-group isolation.
