from __future__ import annotations

import argparse
import json
from pathlib import Path

from poker_model.evaluation_access import (
    FINAL_TEST_CONFIRMATION,
    create_test_access_receipt,
)
from poker_model.evaluation_freeze import (
    build_freeze_manifest,
    verify_freeze_manifest,
    write_freeze_manifest,
)
from poker_model.evaluator import (
    evaluate_split,
    load_evaluator_config,
    report_markdown,
    write_json,
    write_markdown,
)
from poker_model.run_tracking import file_identity


REPO_ROOT = Path(__file__).resolve().parent
DEFAULT_CONFIG = REPO_ROOT / "configs" / "evaluator_v1.json"
DEFAULT_DATA = REPO_ROOT / "data" / "processed"
DEFAULT_SELECTION = REPO_ROOT / "data" / "selected_nt_6max_v0.8.1.jsonl"
DEFAULT_CHECKPOINT = REPO_ROOT / "runs" / "baseline-v081-seed1337" / "best.pt"
DEFAULT_BUNDLE = REPO_ROOT / "artifacts" / "pokergpt-pluribus-v0.8.1.zip"
DEFAULT_RUN_IDENTITY = (
    REPO_ROOT / "runs" / "baseline-v081-seed1337" / "run_identity.json"
)
DEFAULT_REPORT_DIR = REPO_ROOT / "reports" / "evaluator-v1"


def _common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--selection", type=Path, default=DEFAULT_SELECTION)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--report-dir", type=Path, default=DEFAULT_REPORT_DIR)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Frozen replay-aware evaluation for PokerGPT"
    )
    commands = parser.add_subparsers(dest="command", required=True)
    validate = commands.add_parser(
        "validate", help="evaluate the complete validation split only"
    )
    _common(validate)

    freeze = commands.add_parser(
        "freeze", help="freeze code, configuration, candidate, and validation report"
    )
    freeze.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    freeze.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    freeze.add_argument("--artifact-bundle", type=Path, default=DEFAULT_BUNDLE)
    freeze.add_argument("--run-identity", type=Path, default=DEFAULT_RUN_IDENTITY)
    freeze.add_argument("--report-dir", type=Path, default=DEFAULT_REPORT_DIR)

    final_test = commands.add_parser(
        "final-test", help="consume one-time access and score the untouched test split"
    )
    _common(final_test)
    final_test.add_argument("--artifact-bundle", type=Path, default=DEFAULT_BUNDLE)
    final_test.add_argument("--run-identity", type=Path, default=DEFAULT_RUN_IDENTITY)
    final_test.add_argument(
        "--confirmation",
        required=True,
        help=f"must be exactly {FINAL_TEST_CONFIRMATION}",
    )
    return parser.parse_args()


def _validate(args: argparse.Namespace) -> None:
    config = load_evaluator_config(args.config)
    report = evaluate_split(
        config,
        split="val",
        zip_path=args.archive,
        selection_path=args.selection,
        data_dir=args.data_dir,
        checkpoint_path=args.checkpoint,
    )
    json_path = args.report_dir / "validation_report.json"
    markdown_path = args.report_dir / "validation_report.md"
    write_json(json_path, report, exclusive=False)
    write_markdown(markdown_path, report_markdown(report), exclusive=False)
    print(json.dumps({"json": str(json_path), "markdown": str(markdown_path)}, indent=2))


def _freeze(args: argparse.Namespace) -> None:
    validation_report = args.report_dir / "validation_report.json"
    manifest = build_freeze_manifest(
        repo_root=REPO_ROOT,
        config_path=args.config,
        validation_report_path=validation_report,
        checkpoint_path=args.checkpoint,
        artifact_bundle_path=args.artifact_bundle,
        run_identity_path=args.run_identity,
    )
    output = args.report_dir / "freeze_manifest.json"
    write_freeze_manifest(output, manifest)
    print(json.dumps({"freeze_manifest": str(output)}, indent=2))


def _final_test(args: argparse.Namespace) -> None:
    manifest_path = args.report_dir / "freeze_manifest.json"
    manifest = verify_freeze_manifest(
        manifest_path=manifest_path,
        repo_root=REPO_ROOT,
        config_path=args.config,
        checkpoint_path=args.checkpoint,
        artifact_bundle_path=args.artifact_bundle,
        run_identity_path=args.run_identity,
    )
    json_path = args.report_dir / "final_test_report.json"
    markdown_path = args.report_dir / "final_test_report.md"
    completion_path = args.report_dir / "test_completion.json"
    for path in (json_path, markdown_path, completion_path):
        if path.exists():
            raise FileExistsError(f"final evaluation output already exists: {path}")
    manifest_identity = file_identity(manifest_path)
    permit = create_test_access_receipt(
        args.report_dir / "test_access_receipt.json",
        freeze_manifest_sha256=manifest_identity["sha256"],
        confirmation=args.confirmation,
        checkpoint_sha256=manifest["checkpoint"]["sha256"],
    )
    config = load_evaluator_config(args.config)
    report = evaluate_split(
        config,
        split="test",
        zip_path=args.archive,
        selection_path=args.selection,
        data_dir=args.data_dir,
        checkpoint_path=args.checkpoint,
        test_permit=permit,
    )
    write_json(json_path, report, exclusive=True)
    write_markdown(markdown_path, report_markdown(report), exclusive=True)
    completion = {
        "format_version": 1,
        "status": "test_evaluation_complete",
        "freeze_manifest_sha256": manifest_identity["sha256"],
        "json_report": file_identity(json_path),
        "markdown_report": file_identity(markdown_path),
    }
    write_json(completion_path, completion, exclusive=True)
    print(
        json.dumps(
            {
                "json": str(json_path),
                "markdown": str(markdown_path),
                "completion": str(completion_path),
            },
            indent=2,
        )
    )


def main() -> None:
    args = parse_args()
    if args.command == "validate":
        _validate(args)
    elif args.command == "freeze":
        _freeze(args)
    elif args.command == "final-test":
        _final_test(args)
    else:
        raise AssertionError(args.command)


if __name__ == "__main__":
    main()
