# Versioned training artifacts

`pokergpt-pluribus-v0.8.1.zip` contains the current finalized, validated,
three-way Pluribus training corpus. It was produced by streaming the 10,000
selected hands from the source archive; no raw PHH files are included.

- Pipeline version: `0.8.1`
- Artifact format: `pluribus_6max_100bb_spr_position_single_decision_v5`
- Vocabulary: 105 tokens
- Trajectories: 58,942
- Supervised decisions: 91,356
- Maximum trajectory length: 255 (`block_size = 320`)
- Splits: 50,112 train / 5,884 validation / 2,946 held-out test trajectories
- ZIP size: 938,936 bytes
- SHA-256: `07588049AD6F1F89AC34AFF424B10DC72BB93BAF2BD53854C76E4BE9EAA31BD4`

The package contains train/validation/test token binaries, aligned loss masks,
trajectory indexes, `meta.pkl`, statistics, preprocessing and validation
manifests, parse errors, and audit samples. Validation confirms zero pairwise
session-group overlap among the three splits.

On a new training machine, extract the package into `data/processed/`:

```powershell
Expand-Archive artifacts/pokergpt-pluribus-v0.8.1.zip data/processed -Force
python validate_artifacts.py data/processed
```

The raw `poker-hand-histories.zip` is not required for training. Before pushing
this derived dataset to a public remote, confirm that redistribution is permitted
by the source dataset's terms. A private repository or private release asset
avoids unintentionally publishing the corpus.

`pokergpt-pluribus-v0.8.0.zip` is retained as the validated legacy two-way
train/validation release. Its SHA-256 is
`72C6EFE9D8AA69B26F2FD0640874E30A757CF9ACE05334A6E28E26D16822A00B`.

## Pretrained checkpoint

`checkpoints/pokergpt-pluribus-v0.8.1-seed1337-best.pt` is the immutable best
checkpoint from the first 8,000-step baseline. Validation selected it at
optimizer step 7,750 with loss `0.6791787324647758`. Its SHA-256 is
`1D5F7C7D47B9ECFDB67A85BE36ABEF8A0F3013C96D037818083B34C0E76E17EF`.

The checkpoint is stored with Git LFS. The matching configuration, candidate
identity, validation curve, and held-out evaluation results are documented in
`docs/baseline_v081_report.md` and `reports/evaluator-v1/`. Smoke-run and
optimizer/archive checkpoints are intentionally excluded.
