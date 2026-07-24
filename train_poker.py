from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Any

from poker_model.trainer import run_training


def _override(config: dict[str, Any], expression: str) -> None:
    if "=" not in expression:
        raise ValueError(f"override must use dotted.path=value: {expression!r}")
    dotted, raw_value = expression.split("=", 1)
    keys = dotted.split(".")
    target: dict[str, Any] = config
    for key in keys[:-1]:
        if key not in target or not isinstance(target[key], dict):
            raise ValueError(f"override path does not exist: {dotted!r}")
        target = target[key]
    if keys[-1] not in target:
        raise ValueError(f"override key does not exist: {dotted!r}")
    try:
        value = json.loads(raw_value)
    except json.JSONDecodeError:
        value = raw_value
    target[keys[-1]] = value


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the PokerGPT supervised baseline")
    parser.add_argument("config", type=Path, help="complete JSON training configuration")
    parser.add_argument("--resume", type=Path, help="resume an existing latest.pt checkpoint")
    parser.add_argument(
        "--set",
        dest="overrides",
        action="append",
        default=[],
        metavar="PATH=VALUE",
        help="explicit JSON-valued override for a fresh run; may be repeated",
    )
    parser.add_argument("--run-id", help="fresh-run ID override")
    parser.add_argument(
        "--stop-after-steps",
        type=int,
        help="stop at this absolute optimizer step after saving a resumable checkpoint",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.resume is not None and (args.overrides or args.run_id):
        raise ValueError("configuration overrides are not allowed when resuming")
    with args.config.open("r", encoding="utf-8") as handle:
        config = json.load(handle)
    config = copy.deepcopy(config)
    if args.run_id:
        config["run_id"] = args.run_id
    for expression in args.overrides:
        _override(config, expression)
    result = run_training(
        config,
        resume_path=args.resume,
        stop_after_steps=args.stop_after_steps,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
