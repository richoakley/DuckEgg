"""Run the EGGROLL CUDA/JAX preflight without importing mjlab."""

from __future__ import annotations

import argparse
import os
from collections.abc import Sequence

# Must precede importing the EGGROLL package (which imports JAX).
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

from mjlab_microduck.eggroll.preflight import run_cuda_preflight


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args(argv)
    run_cuda_preflight(args.device)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
