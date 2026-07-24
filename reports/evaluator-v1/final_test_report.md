# PokerGPT test evaluation

- Evaluator: `pokergpt-replay-evaluator-v1`
- Candidate checkpoint: `1d5f7c7d47b9ecfdb67a85be36abef8a0f3013c96d037818083b34c0e76e17ef`
- Optimizer step: 7750
- Seed: 1337
- Decisions: 4,414

## Primary metrics

- Joint decision-token accuracy: 0.767331
- Joint decision-token top-3 accuracy: 0.968962
- Mapped action accuracy: 0.784323
- Mapped action top-2 accuracy: 0.974626
- Illegal-move rate: 0.000906
- Sizing-range error, overall: 0.686343
- Sizing-range error, conditional on aggressive-action correctness: 0.216763

## Per-street results

| Street | Decisions | Token accuracy | Action accuracy | Illegal rate |
|---|---:|---:|---:|---:|
| FLOP | 662 | 0.631420 | 0.648036 | 0.000000 |
| PREFLOP | 3,073 | 0.829157 | 0.849984 | 0.000325 |
| RIVER | 269 | 0.631970 | 0.631970 | 0.011152 |
| TURN | 410 | 0.612195 | 0.612195 | 0.000000 |

## Action confusion matrix

| Truth \ Predicted | FOLD | CHECK | CALL | BET | RAISE |
|---|---:|---:|---:|---:|---:|
| FOLD | 2388 | 0 | 27 | 0 | 14 |
| CHECK | 2 | 628 | 0 | 21 | 15 |
| CALL | 319 | 0 | 100 | 0 | 36 |
| BET | 1 | 295 | 0 | 31 | 0 |
| RAISE | 207 | 2 | 13 | 0 | 315 |

This report contains aggregate results only. The held-out split was not used for evaluator development or checkpoint selection.
