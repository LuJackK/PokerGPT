This project is a PokerGPT / nanoGPT-style Texas Hold'em next-action prediction
experiment based on the project proposal and a very large PHH hand-history
archive. Future agents must treat `poker-hand-histories.zip` as too large to
extract: it is roughly 20 GB compressed and about 245 GB uncompressed, with
hundreds of thousands of `.phh` / `.phhs` files. All dataset work must use a
streaming Python pipeline with `zipfile.ZipFile`, reading and releasing one member
at a time and writing compact artifacts such as `train.bin`, `val.bin`, aligned
loss masks, trajectory indexes, `meta.pkl`, statistics, and manifests.

The initial modeling target is the clean 10,000-hand six-player Pluribus
no-limit Texas Hold'em corpus (`variant = 'NT'`). Keep the ACPC fixed-limit-heavy
corpus out of the initial run; if it is used later, deduplicate its paired raw and
processed branches. HandHQ is optional future data because files mix player
counts and most private cards are unknown.

The training unit is a complete player-perspective hand trajectory, not an
independent snapshot per decision. Generate one trajectory for every player who
makes a decision, rotate that hero to `PLAYER_1`, include only the hero's private
cards, and omit opponent private-card tokens entirely. Preserve every observable
forced post, opponent action, hero decision, and board reveal causally through the
hero's final decision. A trajectory may contain several `<PLAYER_1_DECISION>`
targets; mask loss only onto the hero action and bet/raise sizing-range tokens.
All perspectives derived from one source hand must remain in the same data split.

Do not use `<HISTORY>`, `<EVENT_SEQUENCE>`, poker street-name tokens, player names,
tournament-event names, hand IDs, table names, or timestamps. The trajectory itself is the event
history. Encode board cards when revealed, for example
`BOARD_REVEAL COUNT_3 CARD_Qs CARD_7h CARD_2c`. Before each hero decision, encode
the public pot, call amount, and active/all-in player stacks. Folded players need
not be repeated in later state observations because their fold actions remain in
the trajectory.

Use compositional numerical tokens instead of field-by-range vocabulary entries:
for example `POT_SIZE_BB RANGE_5_TO_10`, `STACK_BB RANGE_GT_50`, and
`AMOUNT_POT RANGE_0.5_TO_0.75`. Continue using exact `Decimal` arithmetic during
replay. Interpret PHH `cc` as check or call from state, `cbr N` as a street-total
raise-to amount that becomes bet or raise plus an incremental contribution, and
`sm` as a non-decision showdown operation.

The measured Pluribus output has 58,942 complete trajectories, 91,356 supervised
decisions, a 111-token vocabulary, median length 44, 99th-percentile length 155,
and maximum length 218. A `block_size` of 256 covers every trajectory. Never
truncate early hand history to fit context; preprocessing and loading must reject
an oversized trajectory and require a larger block size. Training batches must
use `.idx` boundaries, load complete trajectories, right-pad them, and never
random-crop into the middle of a hand or join two hands in one attention context.

Do not encode or preserve legal-action masks. Replay validates recorded source
actions for data integrity, but model generation remains raw so a later poker
engine can measure and penalize illegal moves. Evaluate action accuracy, top-k
accuracy, illegal move rate, internal street-specific accuracy, action confusion,
and amount-range error.

Data selection remains manifest-first. `build_manifest.py` streams bounded
headers into a JSONL manifest; `select_dataset.py` deterministically selects
Pluribus NT 6-max hands and groups splits by session folder; `prepare_poker.py`
streams only selected members into complete trajectories; and
`validate_artifacts.py` verifies framing, ranges, multi-target masks, context
lengths, and split isolation. Test outputs belong below `test/artifacts/` so they
cannot collide with later production preprocessing.
