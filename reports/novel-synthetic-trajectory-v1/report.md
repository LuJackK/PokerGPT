# PokerGPT novel synthetic trajectory

## Result

The model was evaluated on one newly authored, legal six-max 100-BB trajectory containing 8 hero decisions and 252 encoded tokens.

It matched 3/8 raw decision tokens and 3/8 mapped poker actions. All 8/8 raw predictions were legal.

Exact training-set duplicate search found 0 matches among 50,112 training trajectories. Exact context-prefix matches at the eight decision points were [3, 0, 0, 0, 0, 0, 0, 0]. Opponent hole cards are absent from the encoded model input.

## Synthetic hand

- Hero: hijack with As Qs
- Preflop: UTG folds; hero raises to 2.25 BB; cutoff folds; button calls; small blind folds; big blind raises to 8 BB; hero and button call.
- Flop Qh 7d 2c (24.5 BB): big blind checks; hero checks; button bets 9 BB; big blind folds; hero calls.
- Turn Tc (42.5 BB): hero checks; button bets 22 BB; hero calls.
- River 3s (86.5 BB): hero checks; button bets 43 BB; hero calls.

The authored hero line is raise, call, check, call, check, call, check, call.
Aggressive amounts in the table are incremental contributions, matching the model contract.

## Model predictions

| # | Street | Board | Pot / call | Authored action | Model prediction | P(pred) | Legal | Top 3 raw tokens |
|---:|---|---|---|---|---|---:|---|---|
| 1 | Preflop | - | 1.50 / 1.00 BB | RAISE 2.25 BB | RAISE 2.24 BB | 79.7% | yes | RANGE_1_TO_1.5 79.7%, RANGE_1.5_TO_2 20.1%, ACTION_FOLD 0.2% |
| 2 | Preflop | - | 13.00 / 5.75 BB | CALL | **RAISE 21.67 BB** | 53.9% | yes | RANGE_1.5_TO_2 53.9%, ACTION_PASSIVE 14.5%, RANGE_1_TO_1.5 14.1% |
| 3 | Flop | Qh 7d 2c | 24.50 / 0.00 BB | CHECK | **BET 12.23 BB** | 44.1% | yes | RANGE_0.25_TO_0.5 44.1%, ACTION_PASSIVE 30.3%, RANGE_0.5_TO_0.75 12.6% |
| 4 | Flop | Qh 7d 2c | 33.50 / 9.00 BB | CALL | **FOLD** | 62.1% | yes | ACTION_FOLD 62.1%, ACTION_PASSIVE 29.5%, RANGE_1_TO_1.5 2.7% |
| 5 | Turn | Qh 7d 2c Tc | 42.50 / 0.00 BB | CHECK | CHECK | 81.2% | yes | ACTION_PASSIVE 81.2%, ACTION_FOLD 6.8%, RANGE_0.75_TO_1 3.5% |
| 6 | Turn | Qh 7d 2c Tc | 64.50 / 22.00 BB | CALL | CALL | 89.8% | yes | ACTION_PASSIVE 89.8%, ACTION_FOLD 5.7%, RANGE_1.5_TO_2 1.5% |
| 7 | River | Qh 7d 2c Tc 3s | 86.50 / 0.00 BB | CHECK | **BET 61.00 BB** | 84.0% | yes | ACTION_ALL_IN 84.0%, ACTION_PASSIVE 7.8%, ACTION_FOLD 5.7% |
| 8 | River | Qh 7d 2c Tc 3s | 129.50 / 43.00 BB | CALL | **FOLD** | 50.0% | yes | ACTION_FOLD 50.0%, ACTION_ALL_IN 23.7%, ACTION_PASSIVE 18.5% |

## What it did

The opening was strong: it selected the same raise bucket as the 2.25-BB authored open and decoded to 2.24 BB. Facing the 8-BB 3-bet, however, it preferred another raise rather than a call.

On the flop it preferred a roughly half-pot bet over checking. In the separate teacher-forced state where the button had bet 9 BB, it preferred folding top pair. It matched the authored turn check-call line with high confidence.

On the river it strongly preferred a 61-BB all-in instead of checking. In the alternate teacher-forced state where the button had instead bet 43 BB, it narrowly preferred folding. The fold is not an action after its proposed all-in: those are two independent queries against the authored continuation.

## Interpretation boundary

This proves behavior on a new legal input, but the authored action line is not an optimal-poker label. Agreement measures imitation of this scenario, not expected value. Recorded synthetic actions remain in the context after each decision, so later predictions are teacher-forced and do not form a counterfactual branch.

PokerGPT predicts only hero decisions. It was not asked to invent board cards, opponent actions, or pot transitions, because those tokens were not supervised during training.
