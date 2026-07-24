"""Streaming PHH/PHHS preprocessing for PokerGPT."""

__version__ = "0.8.1"
ARTIFACT_FORMAT = "pluribus_6max_100bb_spr_position_single_decision_v5"
PREPARED_OUTPUT_NAMES = (
    "train.bin",
    "train_loss_mask.bin",
    "train.idx",
    "val.bin",
    "val_loss_mask.bin",
    "val.idx",
    "test.bin",
    "test_loss_mask.bin",
    "test.idx",
    "meta.pkl",
    "stats.json",
    "parse_errors.jsonl",
    "audit_samples.jsonl",
)
