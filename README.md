# PokerGPT data pipeline

Streaming preprocessing for PokerKit PHH/PHHS hand histories. The source ZIP is
never extracted: archive members are inspected and parsed one at a time.
The selector defaults to the clean six-player Pluribus `NT` corpus.
Preprocessing writes complete player-perspective hand trajectories with multiple
supervised hero decisions rather than independent decision snapshots. Every hero
decision locally repeats the hero's hole cards, nonempty complete visible board, pot, call
amount, and active/all-in stack state, then supervises exactly one compressed
decision token. Each trajectory starts with one hero-position token; fixed
table-size and blind-post tokens are omitted. All chip amounts are normalized in
big blinds, so the baseline is tied to six-max 100-BB play rather than a literal
chip denomination. Decision states express each active player's remaining stack
relative to the current pot; empty preflop boards and redundant state delimiters
are omitted. Preprocessing defaults to a 320-token context without truncation.

## Pipeline

```powershell
python build_manifest.py poker-hand-histories.zip --output data/manifest.jsonl
python select_dataset.py data/manifest.jsonl --output data/selected_nt_6max.jsonl
python prepare_poker.py poker-hand-histories.zip --selection data/selected_nt_6max.jsonl --output-dir data/processed
python validate_artifacts.py data/processed --selection data/selected_nt_6max.jsonl
```

Run the full HandHQ eligibility and known-card selection-bias audit without
extracting the source archive:

```powershell
python audit_handhq.py poker-hand-histories.zip
```

The audit reads and releases one HandHQ member at a time and writes only compact
aggregate JSON and Markdown reports below `data/processed/`.

Use `--max-members` and paths below `test/artifacts/` for trial runs. Run the
standard-library test suite with:

```powershell
python -m unittest discover -s test -v
```

See each command's `--help` for limits and validation options.
Dataset-specific decisions and observed quirks are documented in
`docs/data_pipeline.md`. The chronological decision and progress record is in
`docs/project_journal.md` and is maintained as source material for the final report.

For training on another machine, the versioned package documented in
`artifacts/README.md` contains the complete validated tokenized corpus. Extract it
into `data/processed/`; the original 20 GB source ZIP is not needed for training.

## Model

`poker_model/model.py` contains the nanoGPT-style causal Transformer used by
PokerGPT. It keeps nanoGPT's pre-norm blocks, causal self-attention, GELU MLP,
weight tying, GPT-2 initialization, and AdamW grouping, while adding masked
next-token loss for the pipeline's decision-token masks. Decision decoding
renormalizes only the fixed hero-decision logits and returns exactly one raw token
so a poker engine can measure illegal actions and sizes before clamping.

```python
from poker_model import GPT, GPTConfig, PokerTrajectoryDataset

config = GPTConfig(vocab_size=meta["vocab_size"], block_size=meta["block_size"])
model = GPT(config)
train_data = PokerTrajectoryDataset("data/processed", "train")
```

PyTorch 2.x is recommended so attention uses its optimized scaled-dot-product
implementation.

Install the model/notebook dependencies with `pip install -r requirements-model.txt`,
then open `notebooks/01_model_architecture.ipynb`.

## Training

The trainer requires version 0.8.1 artifacts with isolated train, validation,
and test splits. Normal training commands open and fingerprint only train and
validation; direct test-dataset construction is sealed behind the final
evaluation permit. Run the focused trainer verification first:

```powershell
python -m unittest test.test_sampler test.test_checkpoint test.test_trainer -v
```

After placing the v0.8.1 artifacts under `data/processed/`, run the 20-step CPU
smoke configuration:

```powershell
python train_poker.py configs/smoke_cpu_v0.8.1.json
```

Resume from the last optimizer-step boundary with the same immutable
configuration:

```powershell
python train_poker.py configs/smoke_cpu_v0.8.1.json --resume runs/smoke-cpu-v081-seed1337/latest.pt
```

The full CUDA baseline is defined by `configs/baseline_v0.8.1.json`. To perform
a bounded CUDA smoke without changing its configured 8,000-step schedule, give
the new run an explicit ID and stop at an absolute optimizer step:

```powershell
python train_poker.py configs/baseline_v0.8.1.json --run-id cuda-smoke-v081-seed1337 --stop-after-steps 50
```

Every fresh run writes its resolved configuration, environment, dataset
fingerprint, metrics, and atomic best/latest checkpoints below `runs/<run_id>/`.
Configuration overrides use `--set dotted.path=value`, are recorded in the
resolved configuration, and are rejected during resume.

## Frozen evaluation

The replay-aware evaluator streams selected PHH members without extracting the
source archive and verifies every replayed trajectory against its prepared
binary. Its metric definitions are in `docs/evaluation_protocol_v1.md`.

Validation development uses:

```powershell
python evaluate_poker.py validate --archive C:\path\to\poker-hand-histories.zip
```

The evaluator was frozen and the step-7,750 candidate was scored exactly once
on the held-out test split. The immutable reports and one-time access receipt
are under `reports/evaluator-v1/`. Do not invoke `final-test` again.
