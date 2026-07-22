# Versioned training artifacts

`pokergpt-pluribus-v0.8.0.zip` contains the finalized, validated Pluribus
training corpus. It was produced by streaming the 10,000 selected hands from the
source archive; no raw PHH files are included.

- Pipeline version: `0.8.0`
- Artifact format: `pluribus_6max_100bb_spr_position_single_decision_v5`
- Vocabulary: 105 tokens
- Trajectories: 58,942
- Supervised decisions: 91,356
- Maximum trajectory length: 255 (`block_size = 320`)
- ZIP size: 997,224 bytes
- SHA-256: `72C6EFE9D8AA69B26F2FD0640874E30A757CF9ACE05334A6E28E26D16822A00B`

The package contains train/validation token binaries, aligned loss masks,
trajectory indexes, `meta.pkl`, statistics, preprocessing and validation
manifests, parse errors, and audit samples.

On a new training machine, extract the package into `data/processed/`:

```powershell
Expand-Archive artifacts/pokergpt-pluribus-v0.8.0.zip data/processed -Force
python validate_artifacts.py data/processed
```

The raw `poker-hand-histories.zip` is not required for training. Before pushing
this derived dataset to a public remote, confirm that redistribution is permitted
by the source dataset's terms. A private repository or private release asset
avoids unintentionally publishing the corpus.
