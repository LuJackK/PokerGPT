# PokerGPT project journal

This journal records project progress, evidence, decisions, reversals, and open
questions. It is intended to be an append-only source for the final report. When
a design changes, retain the earlier decision and add a new entry explaining why
it was superseded.

## Entry template

```text
## YYYY-MM-DD - Short title

Context:
Decision:
Evidence:
Why:
Outcome:
Open questions:
Related commits/artifacts:
```

## 2026-07-11 - Repository and safe archive strategy

**Context.** The workspace initially contained the project proposal and a
20.29 GB `poker-hand-histories.zip`. The archive expands to approximately
245 GB and contains hundreds of thousands of PHH/PHHS files.

**Decision.** Initialize Git and build a streaming pipeline. Never extract the
archive. Inspect ZIP metadata and bounded member headers, then parse one selected
member at a time. Keep trial outputs under `test/artifacts/` and exclude the
archive and generated binaries from Git.

**Why.** Full extraction would require excessive disk space and make experiments
hard to reproduce or cleanly roll back. Streaming bounds memory and storage use.

**Outcome.** Git repository created on `main`. Pipeline phases were separated
into manifesting, selection, replay/parsing, tokenization, binary writing, and
validation.

**Related commit.** `b965b60`.

## 2026-07-11 - Archive reconnaissance and first corpus selection

**Evidence.** An archive-wide bounded-header scan found:

- 276,986 PHH/PHHS files: 10,088 `.phh` and 266,898 `.phhs`;
- 245,116 ACPC files, 21,782 HandHQ files, 10,000 Pluribus files, 83 WSOP
  files, and five standalone examples;
- one empty HandHQ member;
- ACPC is overwhelmingly fixed-limit and includes duplicated raw/processed
  branches;
- HandHQ mixes player counts within files and usually represents private cards
  as `????`;
- all 10,000 Pluribus hands are clean six-player `NT` no-limit Hold'em hands
  with consistent 50/100 blinds and 100 BB starting stacks.

**Decision.** Use Pluribus as the initial modeling corpus. Exclude ACPC and
HandHQ from the first training run.

**Why.** Pluribus is coherent, small enough for rapid complete validation, and
matches the no-limit six-player target. Mixing fixed-limit and incomplete-card
sources would confound the first experiment.

**Finding.** Pluribus hand numbers repeat across session folders. Train/validation
splits must therefore be grouped by session folder rather than bare hand number.
The final deterministic split contains 9,000 training hands and 1,000 validation
hands with no session overlap.

## 2026-07-11 - Stateful PHH interpretation

**Context.** PHH action codes are intentionally overloaded and cannot be mapped
to model actions by string replacement alone.

**Decision.** Replay every hand with exact `Decimal` chip arithmetic:

- `cc` becomes `CHECK` when nothing is owed and `CALL` otherwise;
- `cbr N` treats `N` as the player's street-total/raise-to amount, computes the
  incremental contribution, and becomes `BET` or `RAISE` according to state;
- `sm` and other showdown/muck operations are parsed but are not betting targets.

**Why.** Correct action labels, pot sizes, call amounts, and normalized sizing
features require the current contributions and betting state.

**Outcome.** All 10,000 Pluribus hands replayed with zero parse or observed-action
legality errors.

## 2026-07-11 - Acting-player information boundary

**Decision.** Generate data from the acting player's perspective. Include only
that player's private cards and board cards revealed so far. Exclude opponent
private cards, future board cards, future actions, names, timestamps, events, and
hand identifiers from model tokens.

**Why.** The original Pluribus files list all six private hands at the beginning.
Passing them through would leak information unavailable to a real player and
produce misleadingly strong results.

**Outcome.** Perspective masking and future-information exclusion became tested
pipeline invariants.

## 2026-07-11 - Initial decision-snapshot tokenizer

**Initial decision.** Create one independent sequence per betting decision. The
first structured representation included street labels, absolute hero seat,
five opponent placeholders with `CARD_UNKNOWN`, explicit legal-action tokens,
current-state buckets, prior-action history, and one target action.

**Why it initially seemed reasonable.** Explicit state markers made examples
easy to inspect, and legality tokens appeared likely to help a small model avoid
impossible actions.

**Validation.** The first complete Pluribus run produced 91,356 decision examples
with zero parse errors. Binary, loss-mask, index, and split-leakage checks passed.

**Limitation discovered.** The format repeated large amounts of context for every
decision and included information that was either redundant or better handled by
evaluation logic.

## 2026-07-11 - Removed legal-action hints

**Changed decision.** Remove all `LEGAL_*` tokens and do not write a legality
sidecar mask. Keep source-action legality checks only as preprocessing assertions.

**Why we changed our mind.** Supplying the legal set lets the model rely on an
external answer rather than learning poker's action grammar. It also makes raw
illegal-move rate less informative. The planned evaluation engine will identify
and penalize illegal predictions.

**Outcome.** Training data contains no legal-action hints. Raw model predictions
remain available for honest illegal-move evaluation.

**Related commit.** `23b867a`.

## 2026-07-11 - Relative players and chronological events

**Changed decision.** Make the acting player `PLAYER_1` and number other seats
clockwise. Remove repeated opponent `CARD_UNKNOWN` blocks and poker street-name
tokens. Replace the history wrapper with chronological forced-post, action, and
board-reveal events. Encode blinds as observable forced contributions.

**Why we changed our mind.** Player names and absolute seats do not generalize.
Opponent unknown-card placeholders conveyed no information. Street labels were
redundant when board reveals already define betting-round boundaries.

**Outcome.** Average decision-sequence length fell from 57.1 to 35.5 tokens and
the maximum fell from 147 to 91, while retaining causal action and board order.

**Related commit.** `759817d`.

## 2026-07-11 - Initial nanoGPT model scaffold

**Decision.** Add a small decoder-only Transformer based on nanoGPT principles:
pre-normalized blocks, causal self-attention, GELU MLPs, weight tying, masked
next-token loss, and AdamW parameter grouping.

**Why.** This matches the proposal's goal of implementing a compact generative
architecture while allowing loss to focus on poker decisions and sizing targets.

**Related commit.** `026aa72`.

## 2026-07-12 - Compositional numeric tokens

**Problem found.** The 188-token vocabulary contained 78 field-by-range numeric
tokens such as `POT_SIZE_BB_1_TO_1.5` and `AMOUNT_BB_1_TO_1.5`. The same thirteen
ranges were duplicated for six fields. Count values were duplicated similarly.

**Changed decision.** Separate fields from shared values:

```text
POT_SIZE_BB RANGE_1_TO_1.5
TO_CALL_BB RANGE_0.75_TO_1
AMOUNT_BB RANGE_2_TO_3
BOARD_COUNT COUNT_3
```

**Why we changed our mind.** Compositional tokens avoid a field-by-value vocabulary
product, make the schema easier to extend, and let range meanings share statistical
strength across contexts. The fixed vocabulary decreased from 188 to 111 tokens.

**Additional change.** Replace the single lossy effective-stack summary with
observable per-player active/all-in stack states. Folded players remain represented
by their earlier fold events and are not needlessly repeated in later state blocks.

## 2026-07-12 - Complete hero trajectories

**Problem found.** Independent decision snapshots repeatedly encoded the same hand
prefix and prevented the model from learning several decisions by the same player
as one continuous causal experience.

**Changed decision.** Use one complete trajectory for every player who makes at
least one decision in a hand. That hero remains `PLAYER_1`. A trajectory can contain
multiple `<PLAYER_1_DECISION>` points and multiple supervised targets. Public
opponent actions and board reveals remain context with zero loss.

**Why we changed our mind.** Complete trajectories preserve causal continuity,
reduce duplicated examples, and align the training unit with the model's intended
use: following one player's observable experience through a hand.

**Validation evidence.** The complete Pluribus run produced:

- 58,942 player-perspective trajectories;
- 91,356 supervised decisions;
- 53,045 training and 5,897 validation trajectories;
- mean 1.55 and maximum 8 decisions per trajectory;
- median length 44, 95th percentile 128, 99th percentile 155, maximum 218;
- zero trajectories above the 256-token block size;
- zero parse errors and zero split-group leakage.

**Batching decision.** Use `.idx` boundaries to load whole trajectories, shift
the stored loss mask with next-token targets, and right-pad batches. Never crop
into the middle of a hand or join separate hands into one attention context.

**Related commit.** `cc67075`.

## 2026-07-12 - Decision-time observations and locality review

**Current implemented status.** Complete chronological hero trajectories remain
intact. Immediately before every `<PLAYER_1_DECISION>`, the tokenizer already
repeats the current public state that cannot be reconstructed exactly from
bucketed history:

- current pot size;
- amount `PLAYER_1` must call;
- every active or all-in player's current status and stack.

Folded players are not repeated because their fold events remain earlier in the
same trajectory. Hero cards currently appear in the trajectory header, and board
cards currently appear at their causal `BOARD_REVEAL` events.

**Proposal under review, not yet implemented.** Repeat the hero's hole cards and
complete currently visible board inside every decision observation, and possibly
repeat compact blind-seat orientation. This would make the most strategically
important cards locally accessible without replacing or summarizing the betting
history. A separate street token remains unnecessary because current board count
already determines the betting phase.

**Measured context impact.** Across all 58,942 trajectories:

- current format: median 44, 99th percentile 155, maximum 218;
- repeating hole cards and current board: median 46, 99th percentile 194,
  maximum 271, with five trajectories above 256;
- also repeating blind-seat orientation: estimated median 50, 99th percentile
  216, maximum approximately 299.

**Implication.** If the local card/board and blind-seat proposal is implemented,
the context should increase from 256 to approximately 320. History must not be
truncated to preserve the old limit. Street-start snapshots, active-player count,
and last-aggressor summaries remain unnecessary because decision-time state and
the chronological trajectory already contain that information.

**Open implementation detail.** Minimum-raise state may be useful to an active
player, but it should wait until full-raise, all-in under-raise, and action-reopen
semantics are represented accurately. It is not currently encoded.

## Current open questions

- Finalize training hyperparameters and compute budget for the first baseline.
- Build the evaluation engine for action accuracy, top-k accuracy, illegal-move
  rate, street-specific confusion, and sizing-bucket error.
- Decide whether raw illegal moves receive only a reported metric or an explicit
  evaluation penalty score.
- Compare action-only loss against joint action-and-sizing loss.
- Determine whether HandHQ hands with known hero cards add useful diversity after
  the Pluribus baseline.
- Revisit amount-bucket boundaries using empirical sizing distributions.
- Measure whether explicit per-player stacks improve validation accuracy relative
  to a smaller summary-state representation.
- Decide whether to implement decision-local hero cards, current board, and blind
  seats with a 320-token context based on the measured locality tradeoff.
