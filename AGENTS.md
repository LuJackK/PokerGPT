# PokerGPT agent guide

## Project objective and current phase

PokerGPT is a nanoGPT-style decoder-only Transformer experiment for predicting a
Texas Hold'em player's next action and, for bets and raises, a sizing range. The
model learns from complete causal hand trajectories as observed by one player.

The streaming preprocessing pipeline, trajectory-aware loader, model,
reproducible trainer, checkpointing, and core metric logging are implemented.
The first Pluribus baseline completed on the target RTX A4000. The current data
representation and tokenizer remain provisional while active work shifts to the
evaluation engine, frozen held-out-test reporting, and subsequent replicated
baselines or ablations.

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
processed branches. A full per-hand HandHQ audit found zero hands that combine
finite table-wide starting stacks with known cards for an acting player, so
HandHQ is excluded from supervised training under the current representation.

Selection is manifest-first:

1. `build_manifest.py` streams bounded headers into JSONL.
2. `select_dataset.py` deterministically selects Pluribus NT six-max hands and
   groups train/validation/test splits by session folder.
3. `prepare_poker.py` streams only selected members into complete trajectories.
4. `validate_artifacts.py` checks framing, token ranges, multi-target masks,
   context lengths, artifact alignment, and split isolation.

All perspectives from one source hand must remain in the same split.

## Current representation contract

The following describes the locked first-baseline schema. Document intentional
future schema changes and regenerate all dependent artifacts together.

- The training unit is one complete player-perspective hand trajectory for each
  player who makes a decision, not an independent decision snapshot.
- Rotate the hero to `PLAYER_1`; number other seats clockwise. Begin with one
  hero-position token from small blind, big blind, UTG, hijack, cutoff, or button.
- Include only the hero's private cards. Never include opponent private cards.
- Preserve every opponent action, hero decision, and board reveal causally
  through the hero's final decision. Do not emit the fixed blind posts; the hero
  position token preserves their strategically relevant seat information.
- A trajectory may contain several `<PLAYER_1_DECISION>` targets. Every decision
  has exactly one supervised token and contributes exactly one loss term:
  `ACTION_FOLD`, `ACTION_PASSIVE`, a nonzero pot-relative `RANGE_*`, or
  `ACTION_ALL_IN`. `RANGE_ZERO` is never a hero target.
- Immediately before each hero decision, repeat the hero's two hole cards, the
  nonempty complete visible board, public pot, call amount, and active/all-in
  player states. Omit the board field preflop; preceding board-reveal history
  already establishes an empty board.
- Keep causal board events such as
  `BOARD_REVEAL COUNT_3 CARD_Qs CARD_7h CARD_2c` even though the current board is
  repeated at decisions.
- Folded players need not be repeated in later state observations because their
  fold actions remain in the trajectory.
- Do not use `<HISTORY>`, `<EVENT_SEQUENCE>`, street-name tokens, player names,
  tournament names, hand IDs, table names, or timestamps.
- Use compositional numerical tokens, for example
  `POT_SIZE_BB RANGE_5_TO_10`, `STACK_POT RANGE_3_TO_5`, and
  `AMOUNT_POT RANGE_0.5_TO_0.75`.
- Encode every active player's remaining stack divided by the current exact pot
  as `STACK_POT RANGE_*`. Emit only `PLAYER_* STATUS_ALL_IN` for an all-in player;
  its zero stack is already exact. Do not use a `<PLAYER_STATES>` delimiter.
- Use exact `Decimal` arithmetic during replay. Interpret PHH `cc` as check or
  call from state, `cbr N` as a street-total raise-to amount converted to the
  incremental bet/raise contribution, and `sm` as a non-decision showdown
  operation.
- Do not encode or store legal-action masks. Replay validates recorded actions,
  while generation remains raw so evaluation can measure illegal moves.
- The baseline contract is Pluribus six-max with every seat starting at 100 BB,
  no antes, no straddles, and 0.5/1-BB blinds. Chip denominations are arbitrary
  because all amounts are normalized by the big blind.

Pipeline version 0.8.1 has a 105-token vocabulary and format
`pluribus_6max_100bb_spr_position_single_decision_v5`. It removes the constant
table-size pair, fixed blind/ante/straddle tokens, empty-board marker,
player-state delimiter, unused count tokens, and players 7-10, then adds six
hero-position tokens and replaces `STACK_BB` with `STACK_POT`. The full streaming
trial measured median 36, 99th percentile 175, and maximum 255. Keep
`block_size = 320`.

The vocabulary includes the revised deep-stack ranges through
`RANGE_50_TO_75`, `RANGE_75_TO_100`, `RANGE_100_TO_150`, and `RANGE_GT_150`.
Opponent actions remain explicit `ACTION_CHECK`, `ACTION_CALL`, `ACTION_BET`,
and `ACTION_RAISE` context tokens. Deep ranges from `RANGE_20_TO_50` upward remain
valid context tokens but are not model outputs because the full corpus contains
no such supervised target. The production v0.8.1 binaries under `data/processed/`
are regenerated and validated with their matching `meta.pkl`. They contain
50,112 train, 5,884 validation, and 2,946 held-out test trajectories, with zero
pairwise session overlap. A compact versioned training bundle is stored at
`artifacts/pokergpt-pluribus-v0.8.1.zip`; the raw source ZIP is not required on
the training machine. The v0.8.0 two-way bundle remains a legacy artifact.

## Model and batching status

`poker_model/model.py` contains the initial roughly 10.8-million-parameter causal
Transformer: six layers, six attention heads, 384-dimensional embeddings, a
320-token context, masked next-token cross-entropy, and AdamW parameter grouping.
`poker_model/data.py` loads trajectories through `.idx` boundaries and right-pads
within each batch.

Training code must preserve complete trajectories. Never random-crop into a hand
or concatenate separate hands into one attention context. Padding targets and
padding loss positions must remain ignored.

`poker_model/trainer.py` provides deterministic length-aware batching,
decision-weighted gradient accumulation, CUDA BF16/FP16 support, validation,
metric logging, and atomic best/latest/archive checkpoints. Checkpoints restore
model, optimizer, scheduler, scaler, sampler cursor, counters, configuration,
dataset identity, and CPU/CUDA RNG state. All 45 tests, the one-batch overfit
gate, CPU resume smoke, and production-model CUDA resume smoke pass.

The first immutable run is `runs/baseline-v081-seed1337`. It completed 8,000
optimizer steps with CUDA BF16 and a 64-by-2 trajectory batch. Validation selected
`best.pt` at step 7,750 with loss 0.679179; `latest.pt` is the completed
step-8,000 state.

The replay-aware evaluator is frozen as `pokergpt-replay-evaluator-v1`. It
streams exact PHH state, proves trajectory alignment against the prepared
binaries, and reports joint token, mapped five-way action, legality, per-street,
confusion, and sizing metrics. The one-time held-out evaluation of `best.pt`
completed on all 4,414 test decisions. Joint token accuracy is 0.767331, mapped
action accuracy is 0.784323, mapped-action top-2 accuracy is 0.974626, and the
illegal-move rate is 0.000906 (4 moves). The immutable JSON and Markdown reports
are under `reports/evaluator-v1/`; the access receipt prevents a normal rerun.

## Target training machine

The intended first-run server is:

- CPU: 13th Gen Intel Core i7-13700K at 3.40 GHz;
- system memory: 32.0 GB (31.6 GB usable);
- GPU: NVIDIA RTX A4000 with 16 GB VRAM.

The trainer uses CUDA BF16 on this GPU. The 64-trajectory microbatch with
two-step accumulation completed the first baseline with about 1.64 GiB peak
PyTorch CUDA allocation, so the effective batch of 128 is the established
starting point. Keep the code device-portable and retain the CPU/small-model
smoke-test path.

## Current priority goals

1. Preserve the frozen evaluator, access receipt, candidate identity, and final
   test report without rerunning or revising them.
2. Replicate the unchanged baseline with seed 2027, followed by seed 4099,
   before making multi-seed claims.
3. Confirm whether the held-out postflop action weakness and strong conditional
   sizing behavior replicate across seeds before choosing an action-focused
   ablation.
4. Benchmark 128-by-1 batching separately as a throughput-only change; retain
   64-by-2 for strict baseline replication unless equivalence is established.
5. Maintain the concise baseline report containing configuration, validation
   learning curve, throughput, memory, checkpoint identity, and final metrics.
6. Keep the hierarchical action/size head as a later ablation, not a baseline
   requirement.

## Deferred research, not current blockers

- Whether illegal moves should receive an aggregate penalty in addition to being
  reported as a metric.
- Deduplicated ACPC expansion after the Pluribus baseline. Reconsider HandHQ only
  if a different source or a defensible stack-reconstruction method removes its
  zero-overlap limitation without leaking future information.
- Post-baseline representation ablations such as alternative stack summaries;
  tokenization changes needed to define the first baseline remain active work.

Keep `docs/project_journal.md` append-only as the chronological record of
decisions and reversals. Keep this file focused on the current operating contract,
implemented state, and active goals; update it when those materially change.
