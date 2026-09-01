"""Render the evidence-bound EGGROLL source-versus-adapted hero video."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

from mjlab_microduck.eggroll.hero import render_hero


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--source-shifted-videos", type=Path, required=True)
    parser.add_argument("--adapted-shifted-videos", type=Path, required=True)
    parser.add_argument("--source-nominal-videos", type=Path, required=True)
    parser.add_argument("--adapted-nominal-videos", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = render_hero(
        manifest_path=args.manifest,
        source_shifted_dir=args.source_shifted_videos,
        adapted_shifted_dir=args.adapted_shifted_videos,
        source_nominal_dir=args.source_nominal_videos,
        adapted_nominal_dir=args.adapted_nominal_videos,
        output=args.output,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
