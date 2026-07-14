# PokerGPT agent guide

## Project objective and current phase

PokerGPT is a nanoGPT-style decoder-only Transformer experiment for predicting a
Texas Hold'em player's next action and, for bets and raises, a sizing range. The
model learns from complete causal hand trajectories as observed by one player.

The streaming preprocessing pipeline, trajectory-aware loader, and initial model
architecture are implemented. The current data representation and tokenizer are
functional but remain provisional while the project prepares its first
reproducible training baselines on the Pluribus corpus. Active work includes
finalizing tokenization, regenerating artifacts after schema changes, building
the training loop and checkpointing, experiment logging, and the evaluation
engine.

## Non-negotiable data-safety rules

- Never extract `poker-hand-histories.zip`. It is about 20 GB compressed and
  about 245 GB uncompressed, with hundreds of thousands of `.phh` and `.phhs`
  files.
- Stream the archive with `zipfile.ZipFile`, reading and releasing one member at
  a time.
- Write only compact derived artifacts such as `train.bin`, `val.bin`, aligned
  loss masks, trajectory indexes, `meta.pkl`, statistics, and manifests.
- Keep trial outputs below `test/artifacts/`; production preprocessing belongs
  below `data/processed/`.
- Never truncate a trajectory to fit context. Reject it and require a larger
  `block_size`.

## Current dataset scope

The first baseline uses only the clean 10,000-hand, six-player Pluribus no-limit
Texas Hold'em corpus (`variant = "NT"`). Keep the fixed-limit-heavy ACPC corpus
out of the first run. If ACPC is studied later, deduplicate its paired raw and
processed branches. HandHQ is deferred because files mix player counts and most
private cards are unknown.

Selection is manifest-first:

1. `build_manifest.py` streams bounded headers into JSONL.
2. `select_dataset.py` deterministically selects Pluribus NT six-max hands and
   groups train/validation splits by session folder.
3. `prepare_poker.py` streams only selected members into complete trajectories.
4. `validate_artifacts.py` checks framing, token ranges, multi-target masks,
   context lengths, artifact alignment, and split isolation.

All perspectives from one source hand must remain in the same split.

## Current representation contract

The following describes the implemented schema, not a declaration that the
tokenization design is final. Preserve these invariants while evaluating and
revising token choices; document intentional schema changes and regenerate all
dependent artifacts together.

- The training unit is one complete player-perspective hand trajectory for each
  player who makes a decision, not an independent decision snapshot.
- Rotate the hero to `PLAYER_1`; number other seats clockwise.
- Include only the hero's private cards. Never include opponent private cards.
- Preserve every observable forced post, opponent action, hero decision, and
  board reveal causally through the hero's final decision.
- A trajectory may contain several `<PLAYER_1_DECISION>` targets. Every decision
  has exactly one supervised token and contributes exactly one loss term:
  `ACTION_FOLD`, `ACTION_PASSIVE`, a nonzero pot-relative `RANGE_*`, or
  `ACTION_ALL_IN`. `RANGE_ZERO` is never a hero target.
- Immediately before each hero decision, repeat the hero's two hole cards, the
  complete visible board, public pot, call amount, and active/all-in stacks.
  Encode preflop as `CURRENT_BOARD COUNT_0`.
- Keep causal board events such as
  `BOARD_REVEAL COUNT_3 CARD_Qs CARD_7h CARD_2c` even though the current board is
  repeated at decisions.
- Folded players need not be repeated in later state observations because their
  fold actions remain in the trajectory.
- Do not use `<HISTORY>`, `<EVENT_SEQUENCE>`, street-name tokens, player names,
  tournament names, hand IDs, table names, or timestamps.
- Use compositional numerical tokens, for example
  `POT_SIZE_BB RANGE_5_TO_10`, `STACK_BB RANGE_75_TO_100`, and
  `AMOUNT_POT RANGE_0.5_TO_0.75`.
- Use exact `Decimal` arithmetic during replay. Interpret PHH `cc` as check or
  call from state, `cbr N` as a street-total raise-to amount converted to the
  incremental bet/raise contribution, and `sm` as a non-decision showdown
  operation.
- Do not encode or store legal-action masks. Replay validates recorded actions,
  while generation remains raw so evaluation can measure illegal moves.

The current schema has a 117-token vocabulary. The full Pluribus version 0.7.0
run produced 58,942 complete trajectories and 91,356 supervised decisions, with
trajectory length median 46, 99th percentile 190, and maximum 271. Use
`block_size = 320`; the unchanged maximum does not justify changing the first
baseline context.

The vocabulary includes the revised deep-stack ranges through
`RANGE_50_TO_75`, `RANGE_75_TO_100`, `RANGE_100_TO_150`, and `RANGE_GT_150`.
Opponent actions remain explicit `ACTION_CHECK`, `ACTION_CALL`, `ACTION_BET`,
and `ACTION_RAISE` context tokens. Artifacts created with either earlier schema
are stale; the validated version 0.7.0 binaries must always be used with their
matching current `meta.pkl`.

## Model and batching status

`poker_model/model.py` contains the initial roughly 10.8-million-parameter causal
Transformer: six layers, six attention heads, 384-dimensional embeddings, a
320-token context, masked next-token cross-entropy, and AdamW parameter grouping.
`poker_model/data.py` loads trajectories through `.idx` boundaries and right-pads
within each batch.

Training code must preserve complete trajectories. Never random-crop into a hand
or concatenate separate hands into one attention context. Padding targets and
padding loss positions must remain ignored.

## Target training machine

The intended first-run server is:

- CPU: 13th Gen Intel Core i7-13700K at 3.40 GHz;
- system memory: 32.0 GB (31.6 GB usable);
- GPU: NVIDIA RTX A4000 with 16 GB VRAM.

Design the trainer for CUDA and mixed precision, preferring BF16 when supported
and otherwise FP16. Start conservatively around 64 trajectories per device with
gradient accumulation to an effective batch of 128, then tune from measured
VRAM use. Keep the code device-portable and retain a CPU/small-model smoke-test
path.

## Current priority goals

1. Review and finalize the tokenization schema for the first baseline, preserving
   causal information boundaries and complete player-perspective trajectories.
   Record every accepted schema change in `docs/project_journal.md`.
2. Regenerate and validate the full Pluribus artifacts only after the baseline
   schema is selected; never train with stale binaries or a mismatched
   `meta.pkl`.
3. Implement a reproducible trainer with seeded shuffling, length-aware batching,
   mixed precision, gradient accumulation, clipping, learning-rate scheduling,
   periodic validation, metric logging, and best/latest checkpoints.
4. Make checkpoints fully resumable, including model, optimizer, scheduler,
   precision scaler, step/epoch, configuration, dataset identity, and RNG state.
5. Pass an overfit-one-batch test and a short smoke run before launching the
   first full baseline on the RTX A4000 server.
6. Implement evaluation for action accuracy, top-k accuracy, illegal-move rate,
   internal street-specific accuracy, action confusion, and amount-range error.
7. Run the initial single-token joint action-and-sizing baseline. A hierarchical
   action/size head is a later ablation, not a baseline requirement.

## Deferred research, not current blockers

- Whether illegal moves should receive an aggregate penalty in addition to being
  reported as a metric.
- HandHQ or deduplicated ACPC expansion after the Pluribus baseline.
- Post-baseline representation ablations such as alternative stack summaries;
  tokenization changes needed to define the first baseline remain active work.
- Minimum-raise state, only after full-raise, all-in under-raise, and action-reopen
  semantics can be represented and tested correctly.

Keep `docs/project_journal.md` append-only as the chronological record of
decisions and reversals. Keep this file focused on the current operating contract,
implemented state, and active goals; update it when those materially change.
