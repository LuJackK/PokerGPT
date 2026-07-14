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

## 2026-07-13 - Decision-local cards and board implemented

**Finalized decision.** Every `<PLAYER_1_DECISION>` observation now begins with
the hero's two hole cards and the complete currently visible board:

```text
<PLAYER_1_DECISION>
PLAYER_1_HOLE_CARDS CARD_As CARD_4s
CURRENT_BOARD COUNT_3 CARD_Qs CARD_5h CARD_7s
POT_SIZE_BB RANGE_10_TO_20
TO_CALL_BB RANGE_5_TO_10
```

`CURRENT_BOARD COUNT_0` represents the preflop state without adding street-name
tokens. Chronological `BOARD_REVEAL` events remain in place because they preserve
when public information became available. Hero cards moved out of the one-time
trajectory header, avoiding a redundant copy while ensuring they are adjacent to
every supervised choice. Blind-seat orientation was not repeated; the forced-post
events and relative player order remain available in the trajectory.

**Schema impact.** Adding `CURRENT_BOARD` raises the fixed vocabulary from 111 to
112 tokens and advances the artifact format to
`complete_player_perspective_trajectories_v2`. The default context increases from
256 to 320 rather than truncating any early history.

**Full Pluribus validation.** A fresh streaming run over all 10,000 selected hands
produced the same 58,942 trajectories and 91,356 supervised decisions, with zero
parse errors and zero train/validation split-group overlap. Updated trajectory
lengths are median 46, 95th percentile 161, 99th percentile 194, and maximum 271.
Five trajectories would exceed the former 256-token context; none exceeds 320.
The artifact validator additionally checks that every decision observation has
two local hero cards and a correctly framed zero-, three-, four-, or five-card
current board.

## 2026-07-14 - Deep-stack range resolution

**Problem found.** The original shared range vocabulary ended at
`RANGE_GT_50`. A full Pluribus audit found that 309,375 of 309,930 active-stack
observations (99.82%) collapsed into that one bucket, making 55 BB, 100 BB, and
200 BB indistinguishable.

**Changed decision.** Replace `RANGE_GT_50` with four compositional ranges:

- `RANGE_50_TO_75`;
- `RANGE_75_TO_100`;
- `RANGE_100_TO_150`;
- `RANGE_GT_150`.

The boundaries remain shared by stacks, pots, calls, and action-sizing fields.
The fixed vocabulary increases from 112 to 115 tokens, the pipeline version
advances to 0.6.0, and existing binary artifacts must be regenerated with their
matching `meta.pkl`. Sequence lengths and the 320-token context limit do not
change.

**Post-change audit.** Across the same 309,930 active-stack observations, 307,291
(99.15%) fall in `RANGE_75_TO_100`, 2,084 (0.67%) fall in
`RANGE_50_TO_75`, and 555 (0.18%) are at or below 50 BB. The Pluribus corpus
starts every hand at 100 BB, so a large high-stack mode is expected; the change
improves strategically meaningful resolution rather than attempting to balance
the observed distribution artificially. The 100-to-150 and above-150 ranges are
available for future deeper-stack corpora.

## Current open questions

- Finalize training hyperparameters and compute budget for the first baseline.
- Build the evaluation engine for action accuracy, top-k accuracy, illegal-move
  rate, street-specific confusion, and sizing-bucket error.
- Decide whether raw illegal moves receive only a reported metric or an explicit
  evaluation penalty score.
- Compare action-only loss against joint action-and-sizing loss.
- Determine whether HandHQ hands with known hero cards add useful diversity after
  the Pluribus baseline.
- Monitor the revised amount buckets with validation accuracy and sizing error.
- Measure whether explicit per-player stacks improve validation accuracy relative
  to a smaller summary-state representation.

## 2026-07-15 - One supervised token per hero decision

**Accepted decision.** Compress every hero decision into exactly one target:

- fold becomes `ACTION_FOLD`;
- check or call becomes `ACTION_PASSIVE`;
- a non-all-in bet or raise becomes one pot-relative nonzero `RANGE_*` token;
- an aggressive contribution that consumes the remaining stack becomes
  `ACTION_ALL_IN`.

`RANGE_ZERO` remains available for zero-valued public state fields but is invalid
as a hero decision. Opponent actions remain explicit action-plus-amount context,
so the compression changes only supervised hero targets. The poker engine recovers
check versus call from `to_call` and bet versus raise from `current_bet`.

**Why.** Every decision now has equal loss weight, action and sizing cannot
contradict one another, and inference ends after one forward pass and one selected
token. The Transformer and masked next-token cross-entropy do not change.

**Implementation.** Pipeline version 0.7.0 uses format
`complete_player_perspective_single_decision_token_v3` and a 117-token vocabulary.
The tokenizer emits one masked token per decision; the validator requires exactly
one valid decision target in each decision span. `meta.pkl` records the fixed
decision-token set and an executable representative for every nonzero range. Each
observed bucket uses its exact training-split median ratio; empty buckets use a
documented deterministic in-bucket fallback. Inference renormalizes only decision
logits, supports grouped greedy decoding, direct one-token sampling, and raw engine
interpretation. Evaluation helpers aggregate three-way action probabilities,
recover five-way actions, and track sizing, street, confusion, and pre-clamp
legality metrics.

**Validation status.** Fixture preprocessing, binary alignment, all-in detection,
metadata, and artifact validation tests pass. Full Pluribus artifacts remain stale
until regenerated and validated with version 0.7.0; retain `block_size = 320` until
that run reports the new maximum complete-trajectory length.

## 2026-07-15 - Version 0.7.0 full Pluribus regeneration

The production pipeline regenerated all 10,000 selected Pluribus NT six-max hands
under `data/processed/` without extracting the archive. Deterministic session-group
splitting assigned 9,000 source hands to training and 1,000 to validation, with 83
training groups, nine validation groups, and zero overlap.

**Validated output.** The run produced:

- 58,942 complete hero trajectories: 53,045 train and 5,897 validation;
- 91,356 supervised decisions and exactly 91,356 supervised tokens;
- 48,271 folds, 24,613 passive decisions, 18,128 non-all-in range decisions,
  and 344 aggressive all-ins;
- zero parse errors and zero trajectories over `block_size = 320`;
- trajectory length median 46, 95th percentile 157, 99th percentile 190, and
  maximum 271.

Artifact validation passed framing, vocabulary bounds, one-target-per-decision,
decision-token validity, complete-trajectory length, binary alignment, and split
isolation checks with no errors. Because the maximum remains 271, keep
`block_size = 320` for the first baseline.

## 2026-07-15 - First-baseline readiness checkpoint

Version 0.7.0 is now the baseline data contract: its 117-token vocabulary,
matching `meta.pkl`, and regenerated Pluribus binaries validated with no errors.
Decision-only decoding and initial action, sizing, street, confusion, and raw
legality metrics are implemented. The seven available pipeline tests pass; seven
PyTorch-dependent tests were skipped in the current environment. The next blocker
is the reproducible trainer and fully resumable checkpointing, followed by the
one-batch overfit and short smoke runs.
