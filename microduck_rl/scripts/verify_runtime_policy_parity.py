#!/usr/bin/env python3
"""Verify the Rust production policy contract against Python/ONNX references."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from mjlab_microduck.sim.body_server import HOME_POSE
from mjlab_microduck.sim.runtime_contract import verify_runtime_contract

ROOT = Path(__file__).resolve().parents[1]
WORKSPACE = ROOT.parent
DEFAULT_POLICY = WORKSPACE / "microduck" / "example_policies" / "alpha_stand.onnx"
DEFAULT_PROBE = (
    WORKSPACE / "microduck" / "target" / "debug" / "examples" / "policy_probe"
)


def synthetic_fixtures(count: int, seed: int) -> list[dict]:
    if count < 1:
        raise ValueError("fixture count must be positive")
    rng = np.random.default_rng(seed)
    fixtures = []
    for index in range(count):
        gravity = rng.normal(size=3)
        gravity /= np.linalg.norm(gravity)
        positions = np.asarray(HOME_POSE) + rng.normal(0.0, 0.25, size=15)
        positions[9] = 0.0
        fixtures.append(
            {
                "fixture_id": f"synthetic-{index:03d}",
                "positions": positions.tolist(),
                "velocities": rng.normal(0.0, 1.0, size=15).tolist(),
                "imu": {
                    "gyro": rng.normal(0.0, 0.5, size=3).tolist(),
                    "gravity": gravity.tolist(),
                    "quat": [1.0, 0.0, 0.0, 0.0],
                },
                "previous_action": rng.normal(0.0, 0.3, size=14).tolist(),
                "twist": rng.normal(0.0, 0.2, size=3).tolist(),
                "head": rng.normal(0.0, 0.2, size=4).tolist(),
                "body": {
                    "z": float(rng.normal(0.0, 0.01)),
                    "roll": float(rng.normal(0.0, 0.1)),
                    "pitch": float(rng.normal(0.0, 0.1)),
                },
                "action_scale": 1.0,
            }
        )
    return fixtures


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy", type=Path, default=DEFAULT_POLICY)
    parser.add_argument("--probe", type=Path, default=DEFAULT_PROBE)
    parser.add_argument(
        "--fixtures", type=Path, help="optional raw-sensor JSONL fixtures"
    )
    parser.add_argument("--synthetic-count", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20260831)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    fixtures = (
        load_jsonl(args.fixtures)
        if args.fixtures is not None
        else synthetic_fixtures(args.synthetic_count, args.seed)
    )
    report, dumps = verify_runtime_contract(
        probe=args.probe,
        policy_path=args.policy,
        fixtures=fixtures,
    )
    payload = {
        "schema": "microduck-runtime-policy-parity-v1",
        "policy": str(args.policy.resolve()),
        "rust_probe": str(args.probe.resolve()),
        "seed": args.seed if args.fixtures is None else None,
        "report": report.to_dict(),
        "dumps": dumps,
    }
    rendered = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered)
    print(json.dumps(report.to_dict(), indent=2, sort_keys=True))

    if report.max_observation_error > 1.0e-7:
        raise SystemExit("observation parity exceeded 1e-7")
    if report.max_normalized_observation_error > 1.0e-7:
        raise SystemExit("normalized observation parity exceeded 1e-7")
    if report.max_onnx_action_error >= 1.0e-5:
        raise SystemExit("ONNX action parity did not meet <1e-5")
    if report.max_target_error > 1.0e-7:
        raise SystemExit("proposed action target parity exceeded 1e-7")
    if report.max_applied_target_error > 1.0e-7:
        raise SystemExit("safety-applied target parity exceeded 1e-7")


if __name__ == "__main__":
    main()
