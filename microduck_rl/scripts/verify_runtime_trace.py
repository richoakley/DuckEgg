#!/usr/bin/env python3
"""Verify a production-runtime digital-twin trace step by step."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from mjlab_microduck.sim.runtime_contract import verify_runtime_trace

ROOT = Path(__file__).resolve().parents[1]
WORKSPACE = ROOT.parent
DEFAULT_PROBE = (
    WORKSPACE / "microduck" / "target" / "debug" / "examples" / "policy_probe"
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--policy", type=Path, required=True)
    parser.add_argument("--probe", type=Path, default=DEFAULT_PROBE)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows = [
        json.loads(line) for line in args.trace.read_text().splitlines() if line.strip()
    ]
    payload = verify_runtime_trace(
        probe=args.probe,
        policy_path=args.policy,
        trace_rows=rows,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    summary = {name: value for name, value in payload.items() if name != "dumps"}
    print(json.dumps(summary, indent=2, sort_keys=True))
    if payload["max_sensor_derived_observation_error"] > 1.0e-7:
        raise SystemExit("sensor-derived observation parity exceeded 1e-7")
    if payload["max_runtime_previous_action_chain_error"] > 1.0e-7:
        raise SystemExit("runtime raw previous-action chain parity exceeded 1e-7")
    if payload["max_rust_vs_onnx_action_error"] >= 1.0e-5:
        raise SystemExit("Rust/ONNX action parity did not meet <1e-5")
    if payload["max_rust_proposed_target_formula_error"] > 1.0e-7:
        raise SystemExit("Rust proposed-target formula parity exceeded 1e-7")
    if payload["max_rust_applied_target_formula_error"] > 1.0e-7:
        raise SystemExit("Rust safety-target formula parity exceeded 1e-7")
    if payload["max_runtime_applied_action_vs_rust_error"] >= 1.0e-5:
        raise SystemExit("live applied action diverged from Rust replay")
    if payload["max_runtime_applied_target_vs_rust_error"] > 1.0e-7:
        raise SystemExit("live RobotIo target diverged from Rust safety target")
    if any(payload["initial_previous_action"]):
        raise SystemExit("previous action was not zero at reset")
    if not payload["command_slots_all_zero"]:
        raise SystemExit("deployment command slots were not zero")


if __name__ == "__main__":
    main()
