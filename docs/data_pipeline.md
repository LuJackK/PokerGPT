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

Each player action creates a separate example. The example contains the acting
player's two private cards, public cards dealt so far, current stacks/contributions,
legal action types, and observable action history. All opponent private cards are
encoded as `CARD_UNKNOWN`, even though Pluribus source files list all six hands at
the start. Future board cards and showdown information are not included.

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

