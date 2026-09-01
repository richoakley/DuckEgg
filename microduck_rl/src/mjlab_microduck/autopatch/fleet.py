"""Deterministic source-policy smokes for the complete production fleet."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from mjlab_microduck.eggroll.policy_io import (
    import_deployed_policy,
    numpy_actions,
    onnx_actions,
)
from mjlab_microduck.sim.body_server import HOME_POSE
from mjlab_microduck.sim.runtime_contract import (
    build_observation,
    verify_runtime_contract,
)

from .registry import AutopatchRegistry


def synthetic_runtime_fixtures(count: int, seed: int) -> tuple[dict[str, Any], ...]:
    """Create in-range sensor/command fixtures shared by every policy smoke."""

    if count <= 0:
        raise ValueError("fixture count must be positive")
    rng = np.random.default_rng(seed)
    fixtures: list[dict[str, Any]] = []
    for index in range(count):
        gravity = rng.normal(size=3)
        gravity /= np.linalg.norm(gravity)
        positions = np.asarray(HOME_POSE) + rng.normal(0.0, 0.15, size=15)
        positions[9] = 0.0
        fixtures.append(
            {
                "fixture_id": f"fleet-{index:03d}",
                "positions": positions.tolist(),
                "velocities": rng.normal(0.0, 0.5, size=15).tolist(),
                "imu": {
                    "gyro": rng.normal(0.0, 0.25, size=3).tolist(),
                    "gravity": gravity.tolist(),
                    "quat": [1.0, 0.0, 0.0, 0.0],
                },
                "previous_action": rng.normal(0.0, 0.2, size=14).tolist(),
                "twist": rng.normal(0.0, 0.1, size=3).tolist(),
                "head": rng.normal(0.0, 0.1, size=4).tolist(),
                "body": {
                    "z": float(rng.normal(0.0, 0.005)),
                    "roll": float(rng.normal(0.0, 0.05)),
                    "pitch": float(rng.normal(0.0, 0.05)),
                },
                "action_scale": 1.0,
            }
        )
    return tuple(fixtures)


def smoke_production_fleet(
    *,
    registry: AutopatchRegistry,
    runtime_repo: Path,
    rust_probe: Path | None,
    seed: int = 20260831,
    fixture_count: int = 8,
) -> dict[str, Any]:
    """Run every sealed policy through ONNX, NumPy, and optionally Rust."""

    inventory = registry.validate_runtime_artifacts(runtime_repo)
    fixtures = synthetic_runtime_fixtures(fixture_count, seed)
    observations = np.stack([build_observation(fixture) for fixture in fixtures])
    rows: list[dict[str, Any]] = []
    for artifact, inventory_row in zip(registry.artifacts, inventory, strict=True):
        path = runtime_repo / "example_policies" / artifact.filename
        policy = import_deployed_policy(path)
        expected = numpy_actions(policy, observations)
        actual = onnx_actions(policy.source_model, observations)
        if actual.shape != (fixture_count, 14) or not np.isfinite(actual).all():
            raise RuntimeError(f"{artifact.filename} failed ONNX inference smoke")
        numpy_error = float(np.max(np.abs(expected - actual)))
        if numpy_error >= 1.0e-5:
            raise RuntimeError(
                f"{artifact.filename} NumPy/ONNX parity {numpy_error:.3g} >= 1e-5"
            )
        row = {
            **inventory_row,
            "onnx_finite": True,
            "numpy_onnx_max_error": numpy_error,
            "max_abs_action": float(np.max(np.abs(actual))),
            "rust": None,
        }
        if rust_probe is not None:
            report, _dumps = verify_runtime_contract(
                probe=rust_probe,
                policy_path=path,
                fixtures=fixtures,
            )
            rust_values = report.to_dict()
            if report.max_observation_error > 1.0e-7:
                raise RuntimeError(
                    f"{artifact.filename} Rust observation parity failed"
                )
            if report.max_onnx_action_error >= 1.0e-5:
                raise RuntimeError(f"{artifact.filename} Rust ONNX parity failed")
            if report.max_numpy_action_error >= 1.0e-5:
                raise RuntimeError(f"{artifact.filename} Rust NumPy parity failed")
            if report.max_target_error > 1.0e-7:
                raise RuntimeError(f"{artifact.filename} Rust target parity failed")
            if report.max_applied_target_error > 1.0e-7:
                raise RuntimeError(f"{artifact.filename} Rust safety boundary failed")
            row["rust"] = rust_values
        rows.append(row)
    return {
        "schema": "microduck-autopatch-fleet-smoke-v1",
        "status": "pass",
        "seed": seed,
        "fixture_count": fixture_count,
        "rust_probe": str(rust_probe.resolve()) if rust_probe is not None else None,
        "artifacts": rows,
    }
