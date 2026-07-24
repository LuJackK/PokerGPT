# PokerGPT v0.8.1 first baseline report

## Run identity

- Run: `baseline-v081-seed1337`
- Model: 6 layers, 6 heads, 384 embedding dimensions, approximately 10.8M parameters
- Context: 320 tokens, complete trajectories only
- Training: 8,000 optimizer steps, CUDA BF16, microbatch 64, accumulation 2
- Seed: 1337
- Selected checkpoint: step 7,750
- Checkpoint SHA-256: `1d5f7c7d47b9ecfdb67a85be36abef8a0f3013c96d037818083b34c0e76e17ef`
- Dataset fingerprint: `521316d3f395e4ef078cbb8c1eb3214b898ea5721183ff7b301ef088f0ff0109`
- Artifact bundle SHA-256: `07588049ad6f1f89ac34aff424b10dc72bb93baf2bd53854c76e4be9eaa31bd4`

The original training environment captured Python 3.12.13, PyTorch 2.6.0
CUDA 12.4, NVIDIA RTX A4000, and BF16. It did not capture a Git commit; that
provenance gap is retained explicitly in the run identity.

## Training summary

Training completed in 372.4 recorded seconds. Median logged throughput was
155,691 tokens/s and 4,659 supervised decisions/s. Peak PyTorch CUDA allocation
was 1.642 GiB.

| Step | Validation loss | Token accuracy | Grouped action accuracy |
|---:|---:|---:|---:|
| 250 | 1.316146 | 0.542702 | 0.571786 |
| 1,000 | 0.779081 | 0.718627 | 0.729847 |
| 2,000 | 0.788472 | 0.729630 | 0.742375 |
| 4,000 | 0.688225 | 0.738889 | 0.761874 |
| 6,000 | 0.683263 | 0.750436 | 0.769826 |
| 7,750 | **0.679179** | 0.754575 | 0.771678 |
| 8,000 | 0.699043 | 0.751089 | 0.768301 |

## Frozen held-out result

The `pokergpt-replay-evaluator-v1` procedure was frozen before test access.
The held-out split was then evaluated once on 4,414 decisions.

| Metric | Test |
|---|---:|
| Joint decision-token accuracy | 0.767331 |
| Joint decision-token top-3 accuracy | 0.968962 |
| Mapped five-way action accuracy | 0.784323 |
| Mapped five-way action top-2 accuracy | 0.974626 |
| Illegal-move rate | 0.000906 (4/4,414) |
| Sizing-range error, overall | 0.686343 |
| Sizing-range error conditional on correct aggressive action | 0.216763 |
| Conditional representative-ratio MAE | 0.138751 |
| Conditional range-interval distance MAE | 0.029187 |

| Street | Decisions | Token accuracy | Action accuracy | Illegal rate |
|---|---:|---:|---:|---:|
| Preflop | 3,073 | 0.829157 | 0.849984 | 0.000325 |
| Flop | 662 | 0.631420 | 0.648036 | 0 |
| Turn | 410 | 0.612195 | 0.612195 | 0 |
| River | 269 | 0.631970 | 0.631970 | 0.011152 |

Three illegal predictions were folds when checking was available. One was an
all-in token interpreted as an unavailable raise.

## Error analysis and next experiment

The dominant weakness is postflop action selection. Of 455 true calls, 319 were
predicted as folds. Of 327 true bets, 295 were predicted as checks. Of 537 true
raises, 207 were predicted as folds. Preflop action accuracy was 0.850, while
flop, turn, and river action accuracy ranged from 0.612 to 0.648.

Sizing is not the first intervention: once the model selected the correct
aggressive action, its range-token error fell to 0.217 and its interval-distance
MAE was only 0.029 pot. A hierarchical action/size head therefore remains a
later ablation.

The second experiment should be the exact seed-2027 baseline replication with
the same representation, architecture, schedule, and 64-by-2 batching. This
tests whether the postflop fold/check bias and conditional sizing strength are
stable rather than seed-specific. Seed 4099 should follow before multi-seed
claims. A 128-by-1 throughput benchmark may be run separately, but should not
replace the strict replication configuration unless numerical and sampling
equivalence are established.
