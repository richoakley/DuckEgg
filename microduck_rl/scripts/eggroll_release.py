"""Promote and verify an evidence-bound EGGROLL policy derivative."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

from mjlab_microduck.eggroll.release import (
    SUMMARY_ROLES,
    build_release_manifest,
    verify_release_manifest,
)


def _promote(args: argparse.Namespace) -> int:
    summary_paths = {
        role: getattr(args, role.replace("_", "_")) for role in SUMMARY_ROLES
    }
    manifest = build_release_manifest(
        derivative_id=args.derivative_id,
        source_policy=args.source_policy,
        adapted_policy=args.adapted_policy,
        checkpoint=args.checkpoint,
        export_verification=args.export_verification,
        training_dir=args.training_dir,
        summaries=summary_paths,
        evidence_dir=args.evidence_dir,
        output=args.output,
        source_commit=args.source_commit,
        checkpoint_repository=args.checkpoint_repository,
        runtime_verification=args.runtime_verification,
    )
    print(json.dumps(manifest["release_decision"], indent=2, sort_keys=True))
    return 0


def _verify(args: argparse.Namespace) -> int:
    result = verify_release_manifest(
        args.manifest, source_policy_path=args.source_policy
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    promote = subparsers.add_parser("promote")
    promote.add_argument("--derivative-id", required=True)
    promote.add_argument("--source-policy", type=Path, required=True)
    promote.add_argument("--adapted-policy", type=Path, required=True)
    promote.add_argument("--checkpoint", type=Path, required=True)
    promote.add_argument("--export-verification", type=Path, required=True)
    promote.add_argument("--training-dir", type=Path, required=True)
    for role in SUMMARY_ROLES:
        promote.add_argument(f"--{role.replace('_', '-')}", type=Path, required=True)
    promote.add_argument("--evidence-dir", type=Path, required=True)
    promote.add_argument("--output", type=Path, required=True)
    promote.add_argument("--source-commit", required=True)
    promote.add_argument("--checkpoint-repository", required=True)
    promote.add_argument("--runtime-verification", type=Path, required=True)
    promote.set_defaults(func=_promote)

    verify = subparsers.add_parser("verify")
    verify.add_argument("--manifest", type=Path, required=True)
    verify.add_argument("--source-policy", type=Path)
    verify.set_defaults(func=_verify)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
