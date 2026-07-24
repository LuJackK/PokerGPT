# PokerGPT frozen evaluation protocol v1

This protocol evaluates one fixed PokerGPT checkpoint on complete,
player-perspective trajectories. Development uses synthetic fixtures and the
validation split. The held-out test split is sealed behind a one-time access
receipt and is evaluated only after the code, configuration, tests, checkpoint,
artifact identities, and complete validation report are frozen.

## Candidate and prediction policy

The candidate is `baseline-v081-seed1337/best.pt` at optimizer step 7,750.
SHA-256 identities are declared in `configs/evaluator_v1.json`.

At each supervised decision position, logits are restricted to the fixed
13-token decision vocabulary and renormalized with softmax. The primary raw
prediction is the single highest-probability decision token. No legality mask,
clamping, resampling, or post-hoc correction is applied.

## Metrics

**Joint decision-token accuracy** is exact equality between the raw predicted
token and target token. Top-3 accuracy checks whether the target is among the
three highest decision-token probabilities.

**Mapped action accuracy** maps the same raw predicted token and target token to
`FOLD`, `CHECK`, `CALL`, `BET`, or `RAISE` from exact replay state. Passive maps
to check when `to_call` is zero and call otherwise. Aggression maps to bet when
the street has no current wager and raise otherwise. Mapped-action top-2 sums
the probabilities of all decision tokens mapping to each five-way action, then
checks whether the target action is among the two highest action totals.

**Illegal-move rate** is the fraction of primary raw predictions that cannot be
executed unchanged. Exact replay checks fold availability, check/call state,
bet versus raise, stack limits, minimum bets, minimum full-raise increments,
short all-in under-raises, and whether betting has reopened. RANGE tokens
execute at the training-derived representative ratio stored in `meta.pkl`;
`ACTION_ALL_IN` executes the full remaining stack. The aggregate report includes
illegal counts by reason and predicted token.

**Per-street results** report decision count, joint token accuracy and top-3,
mapped action accuracy and top-2, illegal-move rate, and both sizing summaries
for preflop, flop, turn, and river.

**Action confusion** is a complete 5-by-5 count matrix with ground truth on rows
and the mapped primary raw prediction on columns. A complete decision-token
confusion matrix is also retained in JSON.

**Sizing-range error overall** is the non-exact-token rate across every
ground-truth aggressive decision. A passive or fold prediction therefore
counts as a sizing-range error. This exposes the combined action-and-sizing
failure rate.

**Sizing-range error conditional on aggressive-action correctness** uses only
ground-truth aggressive decisions whose primary raw prediction maps to the
correct `BET` or `RAISE` action. The report also gives, on that same subset,
absolute representative-ratio error and distance from the predicted RANGE
interval. All-in predictions are treated as a point at remaining-stack/pot.

## Replay alignment and split isolation

The raw ZIP is streamed one selected member at a time and is never extracted.
Member CRC and sizes must match the selection manifest. Each replayed trajectory
is re-tokenized and must match the prepared split binary exactly before its
decision states are used.

Ordinary dataset construction refuses `test`, and the trainer opens only
`train` and `val`. Final test evaluation requires:

1. a complete validation report;
2. a write-once freeze manifest whose hashes still verify;
3. the exact confirmation phrase `SCORE_UNTOUCHED_TEST_ONCE`;
4. absence of any prior test access receipt or final report.

The test access receipt is created before the first held-out example is read.
Consequently, a failed or interrupted attempt cannot be silently rerun through
the normal command.
