# PokerGPT data pipeline

Streaming preprocessing for PokerKit PHH/PHHS hand histories. The source ZIP is
never extracted: archive members are inspected and parsed one at a time.
The selector defaults to the clean six-player Pluribus `NT` corpus.

## Pipeline

```powershell
python build_manifest.py poker-hand-histories.zip --output data/manifest.jsonl
python select_dataset.py data/manifest.jsonl --output data/selected_nt_6max.jsonl
python prepare_poker.py poker-hand-histories.zip --selection data/selected_nt_6max.jsonl --output-dir data/processed
python validate_artifacts.py data/processed --selection data/selected_nt_6max.jsonl
```

Use `--max-members` and paths below `test/artifacts/` for trial runs. Run the
standard-library test suite with:

```powershell
python -m unittest discover -s test -v
```

See each command's `--help` for limits and validation options.
Dataset-specific decisions and observed quirks are documented in
`docs/data_pipeline.md`.
