# PokerGPT trainer implementation plan

## Purpose and scope

The first trainer produces a reproducible, fully resumable supervised baseline
using the version 0.8.1 three-way artifact release of the v0.8.0 Pluribus
representation. It trains the existing decoder-only Transformer to predict the
single hero-decision token at every `<PLAYER_1_DECISION>` point in a complete
player-perspective trajectory.

Implementation status as of 2026-07-23: all 31 tests, the one-batch overfit gate,
CPU checkpoint/resume smoke, production-model CUDA checkpoint/resume smoke, and
the 8,000-step seed-1337 baseline have completed. Validation selected the
step-7,750 checkpoint. The final test split remains untouched until the
replay-aware evaluation procedure is frozen.

The baseline is imitation learning. It does not add self-play, reinforcement
learning, a value head, legal-action masking, a hierarchical action/size head,
or a new data representation. Those are separate experiments after the baseline
is working and measured.

## Locked baseline contract

The following choices belong to the first baseline and should not be tuned
inside a run:

- Representation: pipeline version `0.8.1`, format
  `pluribus_6max_100bb_spr_position_single_decision_v5`.
- Dataset release: create a new versioned three-way artifact revision before the
  full baseline, expected to be `v0.8.1` unless another schema change is bundled.
  Do not overwrite the validated two-way `v0.8.0` bundle.
- Split target by source hand: approximately 85% training, 10% validation, and
  5% final test. Session grouping takes priority over exact percentages.
- Vocabulary: 105 tokens, including 13 valid hero-decision tokens.
- Context: complete trajectories only, with `block_size = 320`; never crop a
  hand or concatenate separate hands.
- Objective: full-vocabulary next-token cross-entropy evaluated only where the
  stored loss mask selects a hero decision.
- Model: six layers, six attention heads, 384-dimensional embeddings, 1,536
  MLP width, dropout 0.1, no linear biases, learned positional embeddings, and
  tied token-embedding/output weights. This is approximately 10.8 million
  learned parameters.
- Model selection: lowest decision-weighted validation cross-entropy.

Changing any item above creates a new named experiment. Changing the dataset or
token schema also requires regenerated, versioned, and validated artifacts.

## Planned held-out test split

The current v0.8.0 artifacts contain 53,045 training trajectories with 82,319
supervised decisions and 5,897 validation trajectories with 9,037 supervised
decisions. They do not contain a test split. Before the full baseline, extend
selection, preparation, and validation to create a smaller final test set.

Use these rules:

- Target 5% of the 10,000 source hands for test, 10% for validation, and the
  remaining approximately 85% for training.
- Assign complete Pluribus session folders as indivisible split groups. The
  deterministic size-aware assignment should approach the target hand counts,
  but must never split a session merely to reach an exact percentage.
- Keep every player perspective derived from a source hand in the same split.
- Use a recorded split seed and verify pairwise zero group overlap among train,
  validation, and test.
- Derive sizing-bucket representative ratios only from the training split.
- Write `test.bin`, `test_loss_mask.bin`, and `test.idx` alongside the train and
  validation artifacts, and include them in the manifest and dataset
  fingerprint.
- Record the final hand, group, trajectory, token, and supervised-decision counts
  for all three splits after regeneration.

The test split is not used for learning-rate choice, early stopping, checkpoint
selection, model-size choice, threshold selection, or ablation design. Training
uses train, `best.pt` is selected only on validation, and test is evaluated only
after the candidate configuration and evaluation procedure are frozen. Looking
at test results and then changing the model makes the subsequent result
exploratory; it must not be reported as an untouched final test result.

## Proposed project structure

The implementation should add these focused components:

- `train_poker.py`: small command-line entry point that loads a configuration,
  supports fresh and resumed runs, and starts training.
- `poker_model/trainer.py`: training state, mixed-precision loop, gradient
  accumulation, validation, metric logging, and orchestration.
- `poker_model/checkpoint.py`: atomic checkpoint save/load, compatibility
  checks, RNG capture, and dataset fingerprinting.
- `poker_model/data.py`: extend the existing dataset with trajectory lengths and
  a deterministic length-aware batch sampler.
- `configs/baseline_v0.8.1.json`: immutable three-way-split baseline
  configuration committed to
  the repository.
- `experiments/ablation_registry.csv`: committed index of planned, running,
  completed, failed, and invalid experiments.
- `reports/ablations.md`: generated human-readable comparison of completed
  baseline and ablation runs.
- `test/test_sampler.py`, `test/test_checkpoint.py`, and
  `test/test_trainer.py`: deterministic ordering, exact resume, loss weighting,
  overfit-one-batch, and smoke tests.
- `runs/<run_id>/`: resolved configuration, environment information,
  `metrics.jsonl`, and checkpoints. This directory should be ignored by Git by
  default.

The CLI should allow explicit overrides, but every resolved value must be
written into the run directory before model initialization. An override starts
a new run; it must never silently mutate the configuration of a checkpoint being
resumed.

## Initialization sequence

Initialization must happen in this order:

1. Parse and validate the configuration.
2. Set Python and PyTorch CPU/CUDA seeds before constructing the sampler or
   model.
3. Load `meta.pkl` and verify the pipeline version, format, vocabulary size,
   pad token, block size, and decision-token set.
4. Compute a dataset fingerprint over `meta.pkl`, all three token binaries, all
   three loss masks, and all three trajectory indexes. Store it in the run
   metadata and every checkpoint.
5. Construct train, validation, and final-test datasets. The test loader is not
   invoked by the normal training loop.
6. Construct deterministic samplers and data loaders.
7. Build `GPTConfig` from the locked model configuration and dataset metadata.
8. Initialize the model with the existing GPT-style initialization: normal
   weights with standard deviation 0.02, scaled residual projections, and tied
   token-embedding/output weights. There is no separately pretrained embedder.
9. Create AdamW, the learning-rate scheduler, and an FP16 gradient scaler only
   when FP16 is selected. BF16 does not need gradient scaling.
10. If resuming, restore all saved states and verify configuration and dataset
    compatibility before taking another optimizer step.

The trainer should print the resolved device, precision, model parameter count,
dataset counts, average decisions per trajectory, steps per epoch, and total
planned optimizer updates before training starts.

## Data loading and batching

The existing `PokerTrajectoryDataset` already shifts each complete trajectory
into inputs, next-token targets, and a shifted loss mask. The trainer will retain
that behavior.

The new batch sampler should be deterministic from `seed + epoch`:

1. Generate a seeded permutation of trajectory indexes.
2. Divide the permutation into moderately sized pools.
3. Sort each pool by trajectory length.
4. Form batches inside each pool and deterministically shuffle the batches.

This reduces right-padding without making every epoch use the same globally
sorted order. The sampler must expose enough state to recreate the exact epoch
and batch cursor after a resume.

Initial loader settings are:

| Setting | Baseline value |
|---|---:|
| Trajectories per microbatch | 64 |
| Gradient accumulation | 2 microbatches |
| Effective trajectory batch | 128 |
| Loader workers | 0 initially |
| Pin CUDA host memory | Yes on CUDA |
| Drop final incomplete batch | No |

If 64 trajectories do not fit the RTX A4000, reduce the microbatch to 32 and
increase accumulation to 4. This preserves the effective batch and learning-rate
assumptions. Loader workers may be increased only after profiling shows that data
loading is a bottleneck; that is a performance change rather than a model
experiment.

### Correct loss normalization across accumulation

Trajectories contain different numbers of hero decisions. The optimizer update
must give every selected decision equal weight across the entire effective
batch. It must not average two microbatch means, because that would overweight
the microbatch containing fewer decisions.

For microbatch `i`, let `n_i` be its number of selected targets and `mean_i` its
masked mean cross-entropy. Accumulate gradients from `mean_i * n_i`, then divide
the unscaled accumulated gradients by `sum(n_i)` before clipping and the optimizer
step. Validation must similarly report:

```text
sum(batch_mean_loss * batch_decision_count) / sum(batch_decision_count)
```

Checkpoints will be written only at optimizer-step boundaries, so there is no
partially accumulated gradient state to restore.

## First baseline hyperparameters

These are conservative starting values for the RTX A4000. They are hypotheses
to test with the verification gates below, not values to change during a run.

| Hyperparameter | Baseline value |
|---|---:|
| Seed | 1337 |
| Optimizer | AdamW |
| Peak learning rate | `3e-4` |
| Minimum learning rate | `3e-5` |
| Adam betas | `(0.9, 0.95)` |
| Adam epsilon | `1e-8` |
| Weight decay | `0.1` for matrix weights; `0.0` for vectors |
| Gradient clipping | global norm `1.0` |
| Warmup | 400 optimizer steps |
| Schedule | cosine decay after warmup |
| Maximum training | 8,000 optimizer steps, about 19 epochs |
| Validation interval | 250 optimizer steps |
| Latest checkpoint interval | 250 optimizer steps |
| Archival checkpoint interval | 1,000 optimizer steps |
| Training log interval | 10 optimizer steps |
| Precision | BF16 when supported, otherwise FP16 |
| Label smoothing | `0.0` |
| `torch.compile` | Off for the first baseline |

At an effective batch of 128 trajectories, one optimizer step contains about
199 supervised decisions on average. The exact count varies and is logged.

The trainer should record, at minimum:

- optimizer step, epoch, elapsed time, and current learning rate;
- decision-weighted training and validation cross-entropy;
- exact 13-token decision accuracy;
- fold/passive/aggressive action accuracy and top-k accuracy;
- aggressive sizing-bucket accuracy and representative-ratio error;
- gradient norm, tokens/second, decisions/second, and peak CUDA memory;
- per-class counts and a confusion matrix;
- Python, PyTorch, CUDA, GPU, Git commit, and dirty-worktree information.

Full street-specific and legality metrics require the evaluation engine. They
should use the same saved checkpoints but are not allowed to block the core
trainer implementation.

Routine logs contain training and validation metrics only. Test metrics are
written to a separate final-evaluation record containing the evaluated
checkpoint hash and the frozen run configuration.

## Training-step order

Each optimizer update will:

1. Set the model to training mode and clear gradients.
2. Gather the configured accumulation window and count all selected decisions.
3. Move each microbatch to the device and run it under BF16/FP16 autocast.
4. Backpropagate the summed decision loss.
5. Unscale FP16 gradients when a scaler is active.
6. Divide gradients by the total selected decisions in the effective batch.
7. Clip the global gradient norm to 1.0.
8. Take the AdamW step, update the scaler when present, and advance the
   scheduler exactly once.
9. Update counters and write metrics.
10. At configured intervals, run validation and atomically save latest/best
    checkpoints.

Validation runs in evaluation mode with inference disabled, no dropout, no
gradient construction, and loss accumulated by decision count.

## Fully resumable checkpoint contract

Every checkpoint must contain:

- checkpoint format version;
- model state and complete resolved `GPTConfig`;
- optimizer and scheduler states;
- FP16 scaler state or an explicit `null` for BF16/FP32;
- optimizer step, epoch, next batch cursor, and best validation result;
- trajectories, tokens, and supervised decisions seen;
- Python, PyTorch CPU, and every CUDA RNG state;
- sampler seed/state sufficient to reproduce the next batch;
- resolved training configuration;
- dataset fingerprint and preprocessing identity;
- software/environment metadata.

Save to a temporary file and atomically replace the destination. Maintain
`latest.pt` and `best.pt`; a failed write must not corrupt the previous
checkpoint. Resume must fail clearly when the model contract or dataset
fingerprint differs. A deliberate incompatible start is a new run initialized
from model weights only, not a resume.

## Verification gates

Training must progress through these gates in order:

### 1. Unit tests

- Length-aware batches contain each trajectory exactly once per epoch.
- The same seed and epoch produce the same order; another epoch changes it.
- Loss normalization matches a single combined effective batch.
- Padding and context-only tokens never contribute to loss.
- Validation aggregation is weighted by decision count.
- Checkpoint load restores every required state and rejects another dataset.
- Artifact validation proves pairwise train/validation/test group isolation.
- Four uninterrupted CPU steps match two steps plus checkpoint/resume plus two
  steps under deterministic test settings.

### 2. Overfit one batch

Use a small deterministic model and one fixed batch on CPU or CUDA. Training
must drive masked loss close to zero and exact decision-token accuracy close to
100%. Failure here means the full baseline must not start.

### 3. CPU smoke run

Run a small model for roughly 20 optimizer steps, including validation and a
checkpoint resume. Confirm finite loss, changing weights, valid metrics, and
expected output files.

### 4. Full-model CUDA smoke run

Run the production model for 50-100 optimizer steps on the RTX A4000. Confirm
BF16/FP16 stability, memory headroom, throughput, validation, and resume. Use
this measurement to choose 64x2 or 32x4 batching without changing the effective
batch.

### 5. Baseline run

Run the immutable baseline configuration to completion. Preserve `latest.pt`,
`best.pt`, the resolved configuration, logs, environment report, and a final
evaluation summary. Select `best.pt` using validation only, freeze the candidate,
then run the held-out test evaluation once.

## When and how hyperparameters may change

Only the learning-rate scheduler changes a hyperparameter inside a run. Every
manual change starts a new run with its own configuration and rationale.

Use the following diagnosis rules:

| Observation | First response |
|---|---|
| CUDA out of memory | Lower microbatch and raise accumulation to preserve 128 trajectories |
| NaN/Inf loss or gradients | Verify data and scaler, lower LR to `1e-4`, then test FP32 briefly |
| One-batch test cannot overfit | Treat as an implementation bug before tuning |
| Training loss is flat | Verify gradients/masks, then compare LR `1e-4`, `3e-4`, `6e-4` |
| Training improves but validation worsens | Keep the best checkpoint; test dropout or weight decay in a new run |
| Both losses plateau high | Test learning rate first, then capacity or representation |
| Rare actions perform poorly | Inspect per-class support before testing weighting or focal loss |
| GPU has ample memory | Keep effective batch fixed initially; optimize throughput separately |

The first compact tuning study, only if the baseline shows a clear need, is:

- learning rate: `1e-4`, `3e-4`, `6e-4`;
- dropout: `0.0`, `0.1`, `0.2`, holding the selected LR fixed;
- weight decay: `0.01`, `0.1`, holding LR and dropout fixed.

Tune one family at a time. Do not search architecture, objective, and optimizer
simultaneously. Keep the data split, sampler, seed, effective batch, maximum
steps, evaluation interval, and selection metric fixed.

## Ablation protocol

Hyperparameter tuning asks how to train the same baseline well. An ablation asks
whether a specific modeling or representation choice helps. Keep those results
separate.

For every ablation:

1. Write a one-sentence hypothesis and select one changed component.
2. Keep the same source hands, grouped split, training budget, effective batch,
   evaluation cadence, and checkpoint-selection rule.
3. Use seeds `1337`, `2027`, and `4099` for both baseline and variant when making
   a comparative claim; report every run plus mean and standard deviation.
4. Compare shared downstream metrics. Cross-entropy is not directly comparable
   when variants use different output spaces or objectives.
5. Report parameter count, throughput, peak memory, and convergence step so a
   gain is not presented without its cost.
6. Change the experiment name and configuration. If tokenization changes,
   increment the pipeline/schema version and regenerate all artifacts together.
7. Append the result and accepted/rejected conclusion to
   `docs/project_journal.md`.

Use validation for the exploratory ablation table. Do not inspect test results
after every ablation. Freeze the final baseline and shortlisted variant first,
then evaluate their paired-seed checkpoints on test for the final comparison.

Recommended ablation order after the baseline:

1. Action-only target versus the joint single-token action-and-sizing target.
2. Current `STACK_POT` observations versus a carefully specified alternative or
   removal.
3. Standard categorical sizing loss versus an ordinal or distance-aware sizing
   auxiliary loss.
4. Single-token output versus a hierarchical action head followed by a sizing
   head.
5. Model capacity only after objective and representation results are understood.

The common primary outcomes should be fold/passive/aggressive accuracy,
aggressive sizing error, illegal-action rate, illegal-size rate, and
street-specific accuracy. Exact token loss remains useful for variants sharing
the same output contract.

### Ablation result tracking

Track results at three levels: immutable run artifacts, a committed experiment
registry, and an append-only research conclusion in the project journal. A
charting service may supplement these records but must not be their only copy.

Every run receives a descriptive ID containing the experiment, artifact version,
and seed, for example:

```text
baseline-v080-seed1337
action-only-v1-seed1337
action-only-v1-seed2027
action-only-v1-seed4099
```

Each `runs/<run_id>/` directory must contain:

```text
runs/<run_id>/
|-- config.json
|-- environment.json
|-- dataset_fingerprint.json
|-- metrics.jsonl
|-- evaluation.json
|-- best.pt
`-- latest.pt
```

`config.json` is the fully resolved immutable configuration, not just the values
overridden on the command line. `environment.json` records the Git commit and
dirty state, Python and PyTorch versions, CUDA version, GPU, and relevant
determinism settings. `evaluation.json` contains the selected checkpoint step,
all shared metrics, parameter count, peak memory, throughput, and training time.

The committed `experiments/ablation_registry.csv` has one row per experiment,
not one row per seed. It should include at least:

- ablation ID and descriptive name;
- status: `planned`, `running`, `complete`, `failed`, or `invalid`;
- one-sentence hypothesis;
- the single component changed from the baseline;
- artifact/schema version and baseline run family;
- planned seeds and training budget;
- primary and secondary metrics chosen before the run;
- run IDs, completion date, result summary, and conclusion.

Failed and invalid runs remain in the registry with a reason. They must not be
silently deleted or included in aggregate metrics. A trainer bug, incompatible
artifact, incomplete checkpoint, or nonfinite run makes a result invalid; an
experiment that runs correctly but does not improve the metric is a valid
negative result.

A reporting script will read completed `evaluation.json` files and generate
`reports/ablations.md`. For every baseline/variant comparison it reports:

- every individual seed result;
- mean and standard deviation across the paired seeds;
- the paired difference from the baseline;
- action accuracy, sizing error, legality, and street metrics that are common
  to both variants;
- parameter count, best-checkpoint step, throughput, peak GPU memory, and total
  training time.

Where practical, evaluation should also save predictions keyed by source hand.
This permits paired confidence intervals or bootstrap estimates grouped by hand
or session instead of incorrectly treating related decisions as independent.

Before an ablation begins, its registry entry must state the hypothesis, exact
change, fixed controls, seeds, primary metric, secondary metrics, and decision
criterion. After all planned seeds finish, update the generated report and
append the accepted, rejected, or inconclusive conclusion to
`docs/project_journal.md`. Test-set results are added only after the ablation
suite and selection criteria are frozen; inspecting the locked test set must not
be used to redesign the experiment.

## Definition of done

The trainer is ready for the first full baseline when:

- all unit and exact-resume tests pass;
- the one-batch test overfits;
- CPU and production-model CUDA smoke runs complete and resume successfully;
- decision-weighted accumulation and validation are verified;
- latest and best checkpoints survive interruption and reject incompatible data;
- a single committed configuration reproduces initialization, batch order, and
  scheduler position from a recorded seed;
- logs contain enough data to compare later experiments without reconstructing
  undocumented settings;
- the new versioned artifacts contain an untouched final test split with
  pairwise group-isolation validation, and the normal trainer never evaluates it.
