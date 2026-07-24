# PokerGPT long-run validation shadow session

## Protocol

The frozen seed-1337 candidate was run over 1,000 real Pluribus validation hands, represented as 5,884 player-perspective trajectories and 9,180 hero decisions. Cards, boards, opponent actions, pots, and legality state came from exact replay and every trajectory was byte-checked against the prepared binary.

This is a shadow session: the model predicts at every hero decision, but the recorded action advances the hand. It is realistic teacher-forced evaluation, not a counterfactual self-play simulation.

The held-out test split was not opened or rerun.

## Long-run summary

- Joint decision-token accuracy: 75.436%
- Mapped five-way action accuracy: 77.320%
- Illegal moves: 8 / 9,180 (0.087%)
- Entire trajectories with every token prediction correct: 4,528 / 5,884 (76.954%)
- Entire trajectories with every mapped action correct: 4,618 / 5,884 (78.484%)
- Trajectories with at least four hero decisions: 687; token accuracy 57.171%, action accuracy 58.850%, fully token-correct 7.132%
- Confidence calibration ECE: 0.0547
- Longest correct/incorrect token streaks in replay order: 30 / 8
- Rolling 250-decision token accuracy ranged from 66.400% to 83.200%

## What the generated decisions are like

The model is strongest on the opening decision and on short fold-dominated trajectories. First-decision token accuracy is 82.767%, while the 4,090 trajectories of 28-40 tokens score 94.914%. That short-hand strength makes the overall number look better than the long-hand behavior.

In long hands the policy becomes conspicuously conservative. Preflop it folds 83.639% of the time versus 71.628% in the recorded actions. Postflop it strongly prefers checking or folding and rarely initiates aggression: predicted versus recorded bet/raise rates are 6.415% vs 23.521% on the flop, 0.322% vs 24.382% on the turn, and 2.044% vs 30.803% on the river.

The trace pattern is repetitive: many correct checks are followed by folds where the recorded player called, bet, or raised. This is especially visible at later streets. It resembles a cautious baseline with useful preflop pattern recognition, not a balanced multi-street Pluribus imitator.

Sizing is the bright spot once the action is right. Conditional on correctly choosing a bet or raise, the exact range-token success rate is 75.599%, with representative pot-ratio MAE 0.143. The main failure is choosing aggression at all, particularly postflop, rather than choosing a wildly wrong size.

Legality is excellent but not perfect: all 8 illegal outputs are raw folds when checking is available, concentrated on the river. Confidence is somewhat optimistic; 256 wrong token predictions (2.789% of all decisions) still carried at least 90% probability.

## Sequence depth

| Hero decision ordinal | Eligible trajectories | Token accuracy | Action accuracy | All tokens correct through here |
|---:|---:|---:|---:|---:|
| 1 | 5,884 | 82.767% | 85.231% | 82.767% |
| 2 | 1,284 | 67.679% | 69.470% | 24.377% |
| 3 | 977 | 59.468% | 59.672% | 15.148% |
| 4 | 687 | 60.262% | 60.553% | 8.151% |
| 5 | 262 | 56.107% | 56.489% | 2.672% |
| 6 | 67 | 50.746% | 50.746% | 0.000% |
| 7 | 17 | 58.824% | 58.824% | 0.000% |
| 8 | 2 | 0.000% | 0.000% | 0.000% |

## Accuracy by complete trajectory length

| Token length | Trajectories | Decisions | Token accuracy | Action accuracy |
|---|---:|---:|---:|---:|
| 28-40 | 4,090 | 4,090 | 94.914% | 96.039% |
| 41-80 | 798 | 1,090 | 69.817% | 74.862% |
| 81-160 | 877 | 3,329 | 57.615% | 59.658% |
| 161-320 | 119 | 671 | 54.247% | 54.844% |

## Predicted action mix by street

| Street | Fold | Check | Call | Bet | Raise |
|---|---:|---:|---:|---:|---:|
| Preflop | 83.639% | 0.406% | 3.522% | 0.000% | 12.433% |
| Flop | 21.810% | 68.068% | 3.706% | 6.415% | 0.000% |
| Turn | 22.556% | 73.792% | 3.330% | 0.322% | 0.000% |
| River | 28.905% | 64.672% | 4.380% | 1.606% | 0.438% |

## Longest trajectory traces

These are the longest validation trajectories, ordered first by number of hero decisions and then by encoded length. `P(pred)` is the normalized probability of the model's raw decision token.

### Trace 1: Hijack, Qc Jc

- Source: `data/pluribus/106/24.phh::1`; hero seat 4; 228 tokens; 8 decisions

| # | Street | Board | Pot / call | Truth | Prediction | P(pred) | Legal | Top 3 raw tokens |
|---:|---|---|---|---|---|---:|---|---|
| 1 | Preflop | - | 1.50 / 1.00 BB | RAISE 2.00 BB | RAISE 2.24 BB | 74.2% | yes | RANGE_1_TO_1.5 74.2%, RANGE_1.5_TO_2 13.8%, ACTION_FOLD 11.8% |
| 2 | Preflop | - | 10.00 / 4.50 BB | CALL | **FOLD** | 67.2% | yes | ACTION_FOLD 67.2%, ACTION_PASSIVE 23.2%, RANGE_1.5_TO_2 8.3% |
| 3 | Flop | 7s 9c 7c | 14.50 / 0.00 BB | CHECK | CHECK | 98.8% | yes | ACTION_PASSIVE 98.8%, RANGE_0.25_TO_0.5 0.3%, RANGE_2_TO_3 0.2% |
| 4 | Flop | 7s 9c 7c | 21.75 / 7.25 BB | CALL | **FOLD** | 61.7% | yes | ACTION_FOLD 61.7%, ACTION_PASSIVE 30.1%, RANGE_1_TO_1.5 3.1% |
| 5 | Turn | 7s 9c 7c Ac | 29.00 / 0.00 BB | CHECK | CHECK | 73.8% | yes | ACTION_PASSIVE 73.8%, RANGE_0.5_TO_0.75 8.5%, RANGE_0.25_TO_0.5 7.3% |
| 6 | Turn | 7s 9c 7c Ac | 47.00 / 18.00 BB | CALL | CALL | 92.2% | yes | ACTION_PASSIVE 92.2%, ACTION_FOLD 6.9%, RANGE_2_TO_3 0.2% |
| 7 | River | 7s 9c 7c Ac Jh | 65.00 / 0.00 BB | CHECK | CHECK | 98.4% | yes | ACTION_PASSIVE 98.4%, ACTION_FOLD 0.8%, RANGE_0.25_TO_0.5 0.3% |
| 8 | River | 7s 9c 7c Ac Jh | 133.25 / 68.25 BB | CALL | **FOLD** | 78.9% | yes | ACTION_FOLD 78.9%, ACTION_PASSIVE 19.9%, RANGE_1_TO_1.5 0.5% |

### Trace 2: Small Blind, Kh Jc

- Source: `data/pluribus/106/11.phh::1`; hero seat 1; 218 tokens; 8 decisions

| # | Street | Board | Pot / call | Truth | Prediction | P(pred) | Legal | Top 3 raw tokens |
|---:|---|---|---|---|---|---:|---|---|
| 1 | Preflop | - | 1.50 / 0.50 BB | CALL | **RAISE 2.50 BB** | 53.9% | yes | RANGE_1.5_TO_2 53.9%, ACTION_PASSIVE 21.1%, RANGE_1_TO_1.5 19.2% |
| 2 | Preflop | - | 5.00 / 3.00 BB | CALL | CALL | 90.6% | yes | ACTION_PASSIVE 90.6%, ACTION_FOLD 5.6%, RANGE_1.5_TO_2 1.6% |
| 3 | Flop | As Qd Td | 8.00 / 0.00 BB | CHECK | CHECK | 98.8% | yes | ACTION_PASSIVE 98.8%, RANGE_0.25_TO_0.5 0.5%, RANGE_0.5_TO_0.75 0.2% |
| 4 | Flop | As Qd Td | 14.90 / 6.90 BB | CALL | **FOLD** | 58.2% | yes | ACTION_FOLD 58.2%, ACTION_PASSIVE 33.2%, RANGE_1_TO_1.5 3.1% |
| 5 | Turn | As Qd Td 5c | 21.80 / 0.00 BB | CHECK | CHECK | 96.5% | yes | ACTION_PASSIVE 96.5%, RANGE_0.25_TO_0.5 1.3%, RANGE_0.5_TO_0.75 1.1% |
| 6 | Turn | As Qd Td 5c | 39.70 / 17.90 BB | CALL | **FOLD** | 74.2% | yes | ACTION_FOLD 74.2%, ACTION_PASSIVE 19.9%, ACTION_ALL_IN 3.1% |
| 7 | River | As Qd Td 5c 5h | 57.60 / 0.00 BB | CHECK | CHECK | 98.8% | yes | ACTION_PASSIVE 98.8%, RANGE_0.25_TO_0.5 0.3%, RANGE_0.5_TO_0.75 0.2% |
| 8 | River | As Qd Td 5c 5h | 106.48 / 48.88 BB | RAISE 71.20 BB | **FOLD** | 54.3% | yes | ACTION_FOLD 54.3%, RANGE_1_TO_1.5 38.5%, RANGE_1.5_TO_2 7.1% |

### Trace 3: Hijack, Ad Qc

- Source: `data/pluribus/83/13.phh::1`; hero seat 4; 213 tokens; 7 decisions

| # | Street | Board | Pot / call | Truth | Prediction | P(pred) | Legal | Top 3 raw tokens |
|---:|---|---|---|---|---|---:|---|---|
| 1 | Preflop | - | 1.50 / 1.00 BB | RAISE 2.25 BB | RAISE 2.24 BB | 81.6% | yes | RANGE_1_TO_1.5 81.6%, RANGE_1.5_TO_2 18.2%, ACTION_FOLD 0.2% |
| 2 | Flop | 9c Td Qh | 6.00 / 0.00 BB | CHECK | CHECK | 57.0% | yes | ACTION_PASSIVE 57.0%, RANGE_0.25_TO_0.5 20.3%, RANGE_0.5_TO_0.75 15.3% |
| 3 | Flop | 9c Td Qh | 9.00 / 3.00 BB | CALL | **FOLD** | 61.3% | yes | ACTION_FOLD 61.3%, ACTION_PASSIVE 32.8%, RANGE_1_TO_1.5 2.1% |
| 4 | Turn | 9c Td Qh 2h | 12.00 / 0.00 BB | CHECK | CHECK | 78.9% | yes | ACTION_PASSIVE 78.9%, RANGE_0.5_TO_0.75 10.1%, RANGE_0.25_TO_0.5 4.2% |
| 5 | Turn | 9c Td Qh 2h | 21.00 / 9.00 BB | CALL | **FOLD** | 69.1% | yes | ACTION_FOLD 69.1%, ACTION_PASSIVE 27.1%, RANGE_1.5_TO_2 1.3% |
| 6 | River | 9c Td Qh 2h 3h | 30.00 / 0.00 BB | CHECK | **FOLD** | 58.6% | NO: FOLD is unavailable | ACTION_FOLD 58.6%, ACTION_PASSIVE 34.6%, RANGE_1_TO_1.5 2.1% |
| 7 | River | 9c Td Qh 2h 3h | 60.00 / 30.00 BB | FOLD | FOLD | 73.4% | yes | ACTION_FOLD 73.4%, RANGE_1_TO_1.5 18.0%, RANGE_1.5_TO_2 7.0% |

### Trace 4: Hijack, Qc Qh

- Source: `data/pluribus/83/24.phh::1`; hero seat 4; 213 tokens; 7 decisions

| # | Street | Board | Pot / call | Truth | Prediction | P(pred) | Legal | Top 3 raw tokens |
|---:|---|---|---|---|---|---:|---|---|
| 1 | Preflop | - | 1.50 / 1.00 BB | RAISE 2.25 BB | RAISE 2.24 BB | 87.5% | yes | RANGE_1_TO_1.5 87.5%, RANGE_1.5_TO_2 10.2%, ACTION_FOLD 2.1% |
| 2 | Flop | 8d 4c 9s | 6.00 / 0.00 BB | CHECK | CHECK | 70.7% | yes | ACTION_PASSIVE 70.7%, RANGE_0.25_TO_0.5 12.7%, RANGE_0.5_TO_0.75 12.3% |
| 3 | Flop | 8d 4c 9s | 8.50 / 2.50 BB | CALL | **FOLD** | 62.5% | yes | ACTION_FOLD 62.5%, ACTION_PASSIVE 31.4%, RANGE_1_TO_1.5 2.7% |
| 4 | Turn | 8d 4c 9s 2d | 11.00 / 0.00 BB | CHECK | CHECK | 84.0% | yes | ACTION_PASSIVE 84.0%, RANGE_0.5_TO_0.75 6.2%, RANGE_0.25_TO_0.5 2.8% |
| 5 | Turn | 8d 4c 9s 2d | 29.75 / 18.75 BB | CALL | **FOLD** | 73.4% | yes | ACTION_FOLD 73.4%, ACTION_PASSIVE 20.4%, ACTION_ALL_IN 3.5% |
| 6 | River | 8d 4c 9s 2d 4d | 48.50 / 0.00 BB | CHECK | **FOLD** | 65.6% | NO: FOLD is unavailable | ACTION_FOLD 65.6%, ACTION_PASSIVE 29.1%, RANGE_1_TO_1.5 1.7% |
| 7 | River | 8d 4c 9s 2d 4d | 89.52 / 41.02 BB | CALL | **FOLD** | 66.0% | yes | ACTION_FOLD 66.0%, RANGE_1_TO_1.5 28.5%, RANGE_1.5_TO_2 5.2% |

### Trace 5: Hijack, Ad Qd

- Source: `data/pluribus/106/139.phh::1`; hero seat 4; 211 tokens; 7 decisions

| # | Street | Board | Pot / call | Truth | Prediction | P(pred) | Legal | Top 3 raw tokens |
|---:|---|---|---|---|---|---:|---|---|
| 1 | Preflop | - | 1.50 / 1.00 BB | RAISE 2.10 BB | RAISE 2.24 BB | 83.2% | yes | RANGE_1_TO_1.5 83.2%, RANGE_1.5_TO_2 16.4%, ACTION_FOLD 0.3% |
| 2 | Flop | Tc 8d 5c | 5.70 / 0.00 BB | CHECK | CHECK | 66.4% | yes | ACTION_PASSIVE 66.4%, RANGE_0.25_TO_0.5 14.8%, RANGE_0.5_TO_0.75 14.0% |
| 3 | Flop | Tc 8d 5c | 9.97 / 4.27 BB | CALL | **FOLD** | 63.3% | yes | ACTION_FOLD 63.3%, ACTION_PASSIVE 30.9%, RANGE_1_TO_1.5 2.2% |
| 4 | Turn | Tc 8d 5c 7d | 14.24 / 0.00 BB | CHECK | CHECK | 78.1% | yes | ACTION_PASSIVE 78.1%, RANGE_0.5_TO_0.75 10.9%, RANGE_0.25_TO_0.5 4.8% |
| 5 | Turn | Tc 8d 5c 7d | 28.48 / 14.24 BB | CALL | **FOLD** | 71.1% | yes | ACTION_FOLD 71.1%, ACTION_PASSIVE 23.8%, ACTION_ALL_IN 1.9% |
| 6 | River | Tc 8d 5c 7d Ah | 42.72 / 0.00 BB | CHECK | **FOLD** | 63.3% | NO: FOLD is unavailable | ACTION_FOLD 63.3%, ACTION_PASSIVE 29.9%, RANGE_1_TO_1.5 2.0% |
| 7 | River | Tc 8d 5c 7d Ah | 122.11 / 79.39 BB | FOLD | FOLD | 81.6% | yes | ACTION_FOLD 81.6%, ACTION_PASSIVE 14.6%, RANGE_1.5_TO_2 1.4% |

### Trace 6: Utg, Ts Th

- Source: `data/pluribus/106/44.phh::1`; hero seat 3; 211 tokens; 7 decisions

| # | Street | Board | Pot / call | Truth | Prediction | P(pred) | Legal | Top 3 raw tokens |
|---:|---|---|---|---|---|---:|---|---|
| 1 | Preflop | - | 1.50 / 1.00 BB | RAISE 2.10 BB | RAISE 2.24 BB | 87.1% | yes | RANGE_1_TO_1.5 87.1%, ACTION_FOLD 7.4%, RANGE_1.5_TO_2 5.4% |
| 2 | Preflop | - | 10.41 / 4.71 BB | CALL | CALL | 56.6% | yes | ACTION_PASSIVE 56.6%, ACTION_FOLD 40.0%, RANGE_1.5_TO_2 1.8% |
| 3 | Flop | Qd 9h 8c | 15.12 / 0.00 BB | CHECK | CHECK | 95.7% | yes | ACTION_PASSIVE 95.7%, RANGE_0.25_TO_0.5 2.7%, RANGE_0_TO_0.25 0.8% |
| 4 | Flop | Qd 9h 8c | 18.90 / 3.78 BB | CALL | **FOLD** | 69.5% | yes | ACTION_FOLD 69.5%, ACTION_PASSIVE 24.7%, RANGE_1_TO_1.5 2.6% |
| 5 | Turn | Qd 9h 8c 6s | 22.68 / 0.00 BB | BET 6.50 BB | **CHECK** | 59.0% | yes | ACTION_PASSIVE 59.0%, RANGE_0.75_TO_1 12.0%, RANGE_0.5_TO_0.75 9.3% |
| 6 | River | Qd 9h 8c 6s 3c | 35.68 / 0.00 BB | CHECK | **FOLD** | 53.9% | NO: FOLD is unavailable | ACTION_FOLD 53.9%, ACTION_PASSIVE 39.5%, RANGE_1_TO_1.5 2.3% |
| 7 | River | Qd 9h 8c 6s 3c | 62.44 / 26.76 BB | FOLD | **RAISE 82.91 BB** | 54.7% | yes | ACTION_ALL_IN 54.7%, ACTION_FOLD 31.2%, ACTION_PASSIVE 10.8% |

### Trace 7: Utg, Qd Qc

- Source: `data/pluribus/83/104.phh::1`; hero seat 3; 208 tokens; 7 decisions

| # | Street | Board | Pot / call | Truth | Prediction | P(pred) | Legal | Top 3 raw tokens |
|---:|---|---|---|---|---|---:|---|---|
| 1 | Preflop | - | 1.50 / 1.00 BB | RAISE 2.25 BB | RAISE 2.24 BB | 75.8% | yes | RANGE_1_TO_1.5 75.8%, ACTION_FOLD 17.5%, RANGE_1.5_TO_2 6.4% |
| 2 | Preflop | - | 11.25 / 5.25 BB | CALL | **RAISE 18.75 BB** | 44.9% | yes | RANGE_1.5_TO_2 44.9%, ACTION_PASSIVE 38.3%, ACTION_FOLD 11.3% |
| 3 | Flop | 8d Qs Tc | 16.50 / 0.00 BB | CHECK | CHECK | 97.7% | yes | ACTION_PASSIVE 97.7%, RANGE_0.25_TO_0.5 0.9%, RANGE_0.5_TO_0.75 0.4% |
| 4 | Turn | 8d Qs Tc Qh | 16.50 / 0.00 BB | CHECK | CHECK | 42.2% | yes | ACTION_PASSIVE 42.2%, RANGE_0.25_TO_0.5 24.0%, RANGE_0.75_TO_1 12.9% |
| 5 | Turn | 8d Qs Tc Qh | 20.62 / 4.12 BB | CALL | **FOLD** | 70.7% | yes | ACTION_FOLD 70.7%, ACTION_PASSIVE 26.0%, RANGE_1.5_TO_2 1.4% |
| 6 | River | 8d Qs Tc Qh Td | 24.74 / 0.00 BB | CHECK | CHECK | 62.9% | yes | ACTION_PASSIVE 62.9%, ACTION_FOLD 28.9%, RANGE_1_TO_1.5 2.0% |
| 7 | River | 8d Qs Tc Qh Td | 74.22 / 49.48 BB | RAISE 88.38 BB | RAISE 88.38 BB | 42.6% | yes | ACTION_ALL_IN 42.6%, ACTION_FOLD 34.2%, ACTION_PASSIVE 13.9% |

### Trace 8: Big Blind, 4d 6c

- Source: `data/pluribus/83/69.phh::1`; hero seat 2; 207 tokens; 7 decisions

| # | Street | Board | Pot / call | Truth | Prediction | P(pred) | Legal | Top 3 raw tokens |
|---:|---|---|---|---|---|---:|---|---|
| 1 | Preflop | - | 3.75 / 1.25 BB | CALL | **FOLD** | 89.1% | yes | ACTION_FOLD 89.1%, ACTION_PASSIVE 8.8%, RANGE_2_TO_3 1.5% |
| 2 | Flop | Qs 6h 6d | 5.00 / 0.00 BB | CHECK | CHECK | 99.2% | yes | ACTION_PASSIVE 99.2%, ACTION_FOLD 0.2%, RANGE_2_TO_3 0.1% |
| 3 | Flop | Qs 6h 6d | 6.50 / 1.50 BB | RAISE 5.50 BB | **FOLD** | 60.5% | yes | ACTION_FOLD 60.5%, ACTION_PASSIVE 25.2%, RANGE_1_TO_1.5 6.9% |
| 4 | Turn | Qs 6h 6d Ac | 16.00 / 0.00 BB | CHECK | CHECK | 42.8% | yes | ACTION_PASSIVE 42.8%, RANGE_1_TO_1.5 17.3%, RANGE_0.75_TO_1 13.5% |
| 5 | Turn | Qs 6h 6d Ac | 20.00 / 4.00 BB | CALL | **FOLD** | 77.0% | yes | ACTION_FOLD 77.0%, ACTION_PASSIVE 17.7%, RANGE_1_TO_1.5 2.3% |
| 6 | River | Qs 6h 6d Ac Th | 24.00 / 0.00 BB | CHECK | CHECK | 57.0% | yes | ACTION_PASSIVE 57.0%, RANGE_0.5_TO_0.75 13.1%, RANGE_0.75_TO_1 11.6% |
| 7 | River | Qs 6h 6d Ac Th | 42.00 / 18.00 BB | CALL | **FOLD** | 77.0% | yes | ACTION_FOLD 77.0%, RANGE_1_TO_1.5 19.4%, RANGE_1.5_TO_2 3.1% |

### Trace 9: Big Blind, Ad As

- Source: `data/pluribus/106/282.phh::1`; hero seat 2; 205 tokens; 7 decisions

| # | Street | Board | Pot / call | Truth | Prediction | P(pred) | Legal | Top 3 raw tokens |
|---:|---|---|---|---|---|---:|---|---|
| 1 | Preflop | - | 3.60 / 1.10 BB | RAISE 10.00 BB | RAISE 8.80 BB | 68.8% | yes | RANGE_2_TO_3 68.8%, RANGE_3_TO_5 15.8%, ACTION_PASSIVE 7.3% |
| 2 | Flop | 6d 7d 5s | 22.50 / 0.00 BB | CHECK | **BET 11.23 BB** | 43.4% | yes | RANGE_0.25_TO_0.5 43.4%, ACTION_PASSIVE 30.9%, RANGE_0.5_TO_0.75 12.8% |
| 3 | Flop | 6d 7d 5s | 31.50 / 9.00 BB | CALL | **FOLD** | 55.9% | yes | ACTION_FOLD 55.9%, ACTION_PASSIVE 38.5%, RANGE_1_TO_1.5 1.8% |
| 4 | Turn | 6d 7d 5s 8c | 40.50 / 0.00 BB | CHECK | CHECK | 56.2% | yes | ACTION_PASSIVE 56.2%, RANGE_0.5_TO_0.75 20.7%, RANGE_0.25_TO_0.5 13.0% |
| 5 | Turn | 6d 7d 5s 8c | 53.50 / 13.00 BB | CALL | **FOLD** | 65.6% | yes | ACTION_FOLD 65.6%, ACTION_PASSIVE 30.9%, RANGE_1.5_TO_2 1.3% |
| 6 | River | 6d 7d 5s 8c Js | 66.50 / 0.00 BB | CHECK | CHECK | 70.3% | yes | ACTION_PASSIVE 70.3%, RANGE_0.5_TO_0.75 12.6%, RANGE_0.25_TO_0.5 11.1% |
| 7 | River | 6d 7d 5s 8c Js | 133.50 / 67.00 BB | FOLD | FOLD | 77.7% | yes | ACTION_FOLD 77.7%, ACTION_PASSIVE 19.6%, RANGE_1.5_TO_2 1.0% |

### Trace 10: Small Blind, Js As

- Source: `data/pluribus/113b/113.phh::1`; hero seat 1; 204 tokens; 7 decisions

| # | Street | Board | Pot / call | Truth | Prediction | P(pred) | Legal | Top 3 raw tokens |
|---:|---|---|---|---|---|---:|---|---|
| 1 | Preflop | - | 3.75 / 1.75 BB | CALL | **RAISE 9.17 BB** | 71.9% | yes | RANGE_2_TO_3 71.9%, ACTION_FOLD 11.4%, ACTION_PASSIVE 10.7% |
| 2 | Preflop | - | 18.00 / 11.25 BB | CALL | **FOLD** | 68.8% | yes | ACTION_FOLD 68.8%, ACTION_PASSIVE 23.0%, ACTION_ALL_IN 3.8% |
| 3 | Flop | 7d 6h 2c | 29.25 / 0.00 BB | BET 14.62 BB | **CHECK** | 68.4% | yes | ACTION_PASSIVE 68.4%, RANGE_0.25_TO_0.5 20.9%, RANGE_0.5_TO_0.75 5.4% |
| 4 | Turn | 7d 6h 2c 6d | 58.49 / 0.00 BB | CHECK | CHECK | 73.0% | yes | ACTION_PASSIVE 73.0%, RANGE_0.25_TO_0.5 14.8%, RANGE_0.5_TO_0.75 6.0% |
| 5 | Turn | 7d 6h 2c 6d | 67.49 / 9.00 BB | CALL | **FOLD** | 64.8% | yes | ACTION_FOLD 64.8%, ACTION_PASSIVE 22.4%, ACTION_ALL_IN 9.5% |
| 6 | River | 7d 6h 2c 6d 5s | 76.49 / 0.00 BB | CHECK | **BET 62.88 BB** | 49.2% | yes | ACTION_ALL_IN 49.2%, ACTION_FOLD 27.1%, ACTION_PASSIVE 19.9% |
| 7 | River | 7d 6h 2c 6d 5s | 139.37 / 62.88 BB | FOLD | FOLD | 79.3% | yes | ACTION_FOLD 79.3%, ACTION_PASSIVE 18.8%, RANGE_1.5_TO_2 0.7% |

### Trace 11: Hijack, Kh Ah

- Source: `data/pluribus/70/92.phh::1`; hero seat 4; 204 tokens; 7 decisions

| # | Street | Board | Pot / call | Truth | Prediction | P(pred) | Legal | Top 3 raw tokens |
|---:|---|---|---|---|---|---:|---|---|
| 1 | Preflop | - | 1.50 / 1.00 BB | RAISE 2.25 BB | RAISE 2.24 BB | 88.3% | yes | RANGE_1_TO_1.5 88.3%, RANGE_1.5_TO_2 11.6%, ACTION_FOLD 0.1% |
| 2 | Preflop | - | 10.95 / 4.95 BB | CALL | **RAISE 18.25 BB** | 65.6% | yes | RANGE_1.5_TO_2 65.6%, ACTION_PASSIVE 20.7%, ACTION_FOLD 9.0% |
| 3 | Flop | 6d 3s 8s | 15.90 / 0.00 BB | CHECK | CHECK | 98.0% | yes | ACTION_PASSIVE 98.0%, RANGE_2_TO_3 0.6%, RANGE_1.5_TO_2 0.5% |
| 4 | Turn | 6d 3s 8s Jc | 15.90 / 0.00 BB | CHECK | CHECK | 47.1% | yes | ACTION_PASSIVE 47.1%, RANGE_0.5_TO_0.75 21.6%, RANGE_0.25_TO_0.5 19.0% |
| 5 | Turn | 6d 3s 8s Jc | 24.90 / 9.00 BB | CALL | **FOLD** | 61.3% | yes | ACTION_FOLD 61.3%, ACTION_PASSIVE 32.8%, RANGE_1_TO_1.5 1.9% |
| 6 | River | 6d 3s 8s Jc 5s | 33.90 / 0.00 BB | CHECK | CHECK | 77.0% | yes | ACTION_PASSIVE 77.0%, RANGE_0.5_TO_0.75 6.0%, RANGE_0.75_TO_1 5.5% |
| 7 | River | 6d 3s 8s Jc 5s | 54.90 / 21.00 BB | FOLD | **CALL** | 54.7% | yes | ACTION_PASSIVE 54.7%, ACTION_FOLD 40.0%, RANGE_1.5_TO_2 2.5% |

### Trace 12: Big Blind, Qd Kc

- Source: `data/pluribus/106/79.phh::1`; hero seat 2; 201 tokens; 7 decisions

| # | Street | Board | Pot / call | Truth | Prediction | P(pred) | Legal | Top 3 raw tokens |
|---:|---|---|---|---|---|---:|---|---|
| 1 | Preflop | - | 3.75 / 1.25 BB | CALL | **RAISE 9.17 BB** | 43.4% | yes | RANGE_2_TO_3 43.4%, RANGE_3_TO_5 29.9%, ACTION_PASSIVE 19.9% |
| 2 | Flop | 6c Th Js | 5.00 / 0.00 BB | CHECK | CHECK | 99.2% | yes | ACTION_PASSIVE 99.2%, RANGE_0.25_TO_0.5 0.3%, RANGE_0_TO_0.25 0.1% |
| 3 | Flop | 6c Th Js | 8.75 / 3.75 BB | CALL | **FOLD** | 59.4% | yes | ACTION_FOLD 59.4%, ACTION_PASSIVE 27.1%, RANGE_1_TO_1.5 5.9% |
| 4 | Turn | 6c Th Js 6h | 12.50 / 0.00 BB | CHECK | CHECK | 96.1% | yes | ACTION_PASSIVE 96.1%, RANGE_0.25_TO_0.5 2.2%, RANGE_0.5_TO_0.75 0.8% |
| 5 | Turn | 6c Th Js 6h | 21.87 / 9.37 BB | CALL | **FOLD** | 67.2% | yes | ACTION_FOLD 67.2%, ACTION_PASSIVE 24.7%, RANGE_1_TO_1.5 3.4% |
| 6 | River | 6c Th Js 6h 7d | 31.24 / 0.00 BB | CHECK | CHECK | 97.3% | yes | ACTION_PASSIVE 97.3%, RANGE_0.25_TO_0.5 1.0%, RANGE_0.5_TO_0.75 0.8% |
| 7 | River | 6c Th Js 6h 7d | 46.86 / 15.62 BB | FOLD | FOLD | 44.3% | yes | ACTION_FOLD 44.3%, ACTION_PASSIVE 41.6%, ACTION_ALL_IN 10.5% |

## Scope limitation

PokerGPT has supervised loss only on hero decision tokens. Asking it to freely generate board cards, pot updates, or opponent actions would test unsupervised token logits, not poker ability. A genuine counterfactual long-running rollout needs a poker engine plus opponent policies that react to the model's chosen actions.
