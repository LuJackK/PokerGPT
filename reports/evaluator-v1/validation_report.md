# PokerGPT val evaluation

- Evaluator: `pokergpt-replay-evaluator-v1`
- Candidate checkpoint: `1d5f7c7d47b9ecfdb67a85be36abef8a0f3013c96d037818083b34c0e76e17ef`
- Optimizer step: 7750
- Seed: 1337
- Decisions: 9,180

## Primary metrics

- Joint decision-token accuracy: 0.754357
- Joint decision-token top-3 accuracy: 0.966776
- Mapped action accuracy: 0.773203
- Mapped action top-2 accuracy: 0.974401
- Illegal-move rate: 0.000871
- Sizing-range error, overall: 0.710583
- Sizing-range error, conditional on aggressive-action correctness: 0.244006

## Per-street results

| Street | Decisions | Token accuracy | Action accuracy | Illegal rate |
|---|---:|---:|---:|---:|
| FLOP | 1,403 | 0.657876 | 0.670706 | 0.000000 |
| PREFLOP | 6,161 | 0.817075 | 0.841746 | 0.000000 |
| RIVER | 685 | 0.550365 | 0.553285 | 0.011679 |
| TURN | 931 | 0.634801 | 0.635875 | 0.000000 |

## Action confusion matrix

| Truth \ Predicted | FOLD | CHECK | CALL | BET | RAISE |
|---|---:|---:|---:|---:|---:|
| FOLD | 4713 | 0 | 67 | 0 | 44 |
| CHECK | 8 | 1454 | 0 | 42 | 21 |
| CALL | 700 | 0 | 222 | 0 | 57 |
| BET | 0 | 651 | 0 | 62 | 0 |
| RAISE | 446 | 5 | 41 | 0 | 647 |

This report contains aggregate results only. The held-out split was not used for evaluator development or checkpoint selection.
