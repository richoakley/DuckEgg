"""Verify an EGGROLL derivative with MicroDuck's production Rust loader."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

from mjlab_microduck.eggroll.runtime_probe import run_production_loader_probe


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy", type=Path, required=True)
    parser.add_argument("--runtime-repo", type=Path, required=True)
    parser.add_argument("--cargo", type=Path, default=Path("cargo"))
    parser.add_argument("--ort-dylib", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = run_production_loader_probe(
        policy_path=args.policy,
        runtime_repo=args.runtime_repo,
        cargo=args.cargo,
        ort_dylib=args.ort_dylib,
        output=args.output,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
