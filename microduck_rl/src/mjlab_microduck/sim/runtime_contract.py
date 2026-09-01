"""Cross-language contract checks for the production MicroDuck policy loop."""

from __future__ import annotations

import json
import subprocess
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from mjlab_microduck.eggroll.policy_io import (
    import_deployed_policy,
    numpy_actions,
    onnx_actions,
)
from mjlab_microduck.sim.body_server import HOME_POSE, MOUTH_INDEX

OBSERVATION_BLOCKS = {
    "gyro": slice(0, 3),
    "projected_gravity": slice(3, 6),
    "joint_position_relative": slice(6, 20),
    "joint_velocity": slice(20, 34),
    "previous_action": slice(34, 48),
    "twist_command": slice(48, 51),
    "head_command": slice(51, 55),
    "body_xy_command": slice(55, 57),
    "body_z_roll_pitch_command": slice(57, 60),
    "body_yaw_command": slice(60, 61),
}
POLICY_JOINTS = tuple(index for index in range(15) if index != MOUTH_INDEX)


def build_observation(fixture: dict[str, Any]) -> np.ndarray:
    """Reference the documented 61D contract from raw robot-unit fields."""

    positions = np.asarray(fixture["positions"], dtype=np.float64)
    velocities = np.asarray(fixture["velocities"], dtype=np.float64)
    previous = np.asarray(
        fixture.get("previous_action", np.zeros(14)), dtype=np.float32
    )
    twist = np.asarray(fixture.get("twist", np.zeros(3)), dtype=np.float64)
    head = np.asarray(fixture.get("head", np.zeros(4)), dtype=np.float64)
    body = fixture.get("body", {})
    if positions.shape != (15,) or velocities.shape != (15,):
        raise ValueError("positions and velocities must be 15D robot-order arrays")
    if previous.shape != (14,) or twist.shape != (3,) or head.shape != (4,):
        raise ValueError("previous_action/twist/head have invalid dimensions")
    home = np.asarray(HOME_POSE, dtype=np.float64)
    observation = np.concatenate(
        (
            np.asarray(fixture["imu"]["gyro"], dtype=np.float64),
            np.asarray(fixture["imu"]["gravity"], dtype=np.float64),
            positions[list(POLICY_JOINTS)] - home[list(POLICY_JOINTS)],
            velocities[list(POLICY_JOINTS)],
            previous,
            twist,
            head,
            np.zeros(2),
            np.asarray(
                [
                    body.get("z", 0.0),
                    body.get("roll", 0.0),
                    body.get("pitch", 0.0),
                ]
            ),
            np.zeros(1),
        )
    ).astype(np.float32)
    if observation.shape != (61,):
        raise AssertionError(f"observation contract produced {observation.shape}")
    return observation


def absolute_targets(action: np.ndarray, action_scale: float = 1.0) -> np.ndarray:
    action = np.asarray(action, dtype=np.float32)
    if action.shape != (14,):
        raise ValueError("action must be 14D")
    offsets = np.zeros(15, dtype=np.float64)
    offsets[list(POLICY_JOINTS)] = action
    return np.asarray(HOME_POSE, dtype=np.float64) + action_scale * offsets


def run_rust_probe(
    *,
    probe: Path,
    policy: Path,
    fixtures: Sequence[dict[str, Any]],
    sequential_previous_action: bool = False,
) -> list[dict[str, Any]]:
    payload = "".join(
        json.dumps(fixture, sort_keys=True) + "\n" for fixture in fixtures
    )
    command = [str(probe), str(policy)]
    if sequential_previous_action:
        command.append("--sequential")
    result = subprocess.run(
        command,
        input=payload,
        text=True,
        capture_output=True,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(f"Rust policy probe failed:\n{result.stderr}")
    rows = [json.loads(line) for line in result.stdout.splitlines() if line.strip()]
    if len(rows) != len(fixtures):
        raise RuntimeError(
            f"Rust probe returned {len(rows)} rows for {len(fixtures)} fixtures"
        )
    return rows


@dataclass(frozen=True)
class ParityReport:
    policy_sha256: str
    fixture_count: int
    max_observation_error: float
    per_block_observation_error: dict[str, float]
    max_normalized_observation_error: float
    max_onnx_action_error: float
    max_numpy_action_error: float
    max_target_error: float
    max_applied_target_error: float
    first_divergence: dict[str, Any] | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "policy_sha256": self.policy_sha256,
            "fixture_count": self.fixture_count,
            "max_observation_error": self.max_observation_error,
            "per_block_observation_error": self.per_block_observation_error,
            "max_normalized_observation_error": self.max_normalized_observation_error,
            "max_onnx_action_error": self.max_onnx_action_error,
            "max_numpy_action_error": self.max_numpy_action_error,
            "max_target_error": self.max_target_error,
            "max_applied_target_error": self.max_applied_target_error,
            "first_divergence": self.first_divergence,
        }


def verify_runtime_contract(
    *,
    probe: Path,
    policy_path: Path,
    fixtures: Sequence[dict[str, Any]],
    sequential_previous_action: bool = False,
) -> tuple[ParityReport, dict[str, Any]]:
    if not fixtures:
        raise ValueError("at least one fixture is required")
    deployed = import_deployed_policy(policy_path)
    rust_rows = run_rust_probe(
        probe=probe,
        policy=policy_path,
        fixtures=fixtures,
        sequential_previous_action=sequential_previous_action,
    )
    if sequential_previous_action:
        observations: list[np.ndarray] = []
        previous = np.zeros(14, dtype=np.float32)
        for fixture in fixtures:
            observation = build_observation(
                {**fixture, "previous_action": previous.tolist()}
            )
            observations.append(observation)
            previous = onnx_actions(deployed.source_model, observation[np.newaxis, :])[
                0
            ]
        expected_obs = np.stack(observations)
    else:
        expected_obs = np.stack([build_observation(fixture) for fixture in fixtures])
    rust_obs = np.asarray([row["observation"] for row in rust_rows], dtype=np.float32)
    obs_error = np.abs(rust_obs - expected_obs)
    normalized_expected = (
        expected_obs - deployed.normalizer_mean
    ) / deployed.normalizer_denominator
    normalized_rust = (
        rust_obs - deployed.normalizer_mean
    ) / deployed.normalizer_denominator
    rust_actions = np.asarray([row["action"] for row in rust_rows], dtype=np.float32)
    onnx_expected = onnx_actions(deployed.source_model, expected_obs)
    numpy_expected = numpy_actions(deployed, expected_obs)
    expected_targets = np.stack(
        [
            absolute_targets(
                action,
                float(fixture.get("action_scale", 1.0)),
            )
            for action, fixture in zip(rust_actions, fixtures, strict=True)
        ]
    )
    rust_targets = np.asarray([row["targets"] for row in rust_rows], dtype=np.float64)
    rust_applied_targets = np.asarray(
        [row["applied_targets"] for row in rust_rows], dtype=np.float64
    )
    expected_applied_targets = np.clip(expected_targets, -np.pi, np.pi)

    tolerance = 1.0e-7
    divergent = np.argwhere(obs_error > tolerance)
    first_divergence = None
    if divergent.size:
        fixture_index, feature_index = map(int, divergent[0])
        block = next(
            name
            for name, indices in OBSERVATION_BLOCKS.items()
            if feature_index in range(indices.start or 0, indices.stop or 61)
        )
        first_divergence = {
            "fixture": fixture_index,
            "feature": feature_index,
            "block": block,
            "python": float(expected_obs[fixture_index, feature_index]),
            "rust": float(rust_obs[fixture_index, feature_index]),
        }

    report = ParityReport(
        policy_sha256=deployed.source_sha256,
        fixture_count=len(fixtures),
        max_observation_error=float(obs_error.max()),
        per_block_observation_error={
            name: float(obs_error[:, indices].max())
            for name, indices in OBSERVATION_BLOCKS.items()
        },
        max_normalized_observation_error=float(
            np.abs(normalized_rust - normalized_expected).max()
        ),
        max_onnx_action_error=float(np.abs(rust_actions - onnx_expected).max()),
        max_numpy_action_error=float(np.abs(rust_actions - numpy_expected).max()),
        max_target_error=float(np.abs(rust_targets - expected_targets).max()),
        max_applied_target_error=float(
            np.abs(rust_applied_targets - expected_applied_targets).max()
        ),
        first_divergence=first_divergence,
    )
    dumps = {
        "raw_observations_python": expected_obs.tolist(),
        "raw_observations_rust": rust_obs.tolist(),
        "normalized_observations_python": normalized_expected.tolist(),
        "normalized_observations_rust": normalized_rust.tolist(),
        "actions_onnx": onnx_expected.tolist(),
        "actions_rust": rust_actions.tolist(),
        "targets_rust": rust_targets.tolist(),
        "applied_targets_rust": rust_applied_targets.tolist(),
    }
    return report, dumps


def verify_runtime_trace(
    *,
    probe: Path,
    policy_path: Path,
    trace_rows: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    """Audit an actual ``robotd`` episode against the task actor stream.

    The live applied action is recovered from the absolute targets that crossed
    ``RobotIo``. The probe independently rebuilds the same step from the captured
    raw sensor fixture using the production Rust observation and policy types, then
    applies the production actuator-range boundary. Raw policy output and applied
    action are deliberately checked as separate stages.
    """

    if not trace_rows:
        raise ValueError("runtime trace cannot be empty")
    fixtures = [row["fixture"] for row in trace_rows]
    report, dumps = verify_runtime_contract(
        probe=probe,
        policy_path=policy_path,
        fixtures=fixtures,
        sequential_previous_action=True,
    )
    actor = np.asarray(
        [
            row.get("task_actor_observation_diagnostic", row.get("actor_observation"))
            for row in trace_rows
        ],
        dtype=np.float32,
    )
    runtime_actions = np.asarray(
        [row["runtime_action"] for row in trace_rows], dtype=np.float32
    )
    rust_observations = np.asarray(dumps["raw_observations_rust"], dtype=np.float32)
    rust_applied_targets = np.asarray(dumps["applied_targets_rust"], dtype=np.float64)
    live_applied_targets = np.asarray(
        [row["absolute_targets"] for row in trace_rows], dtype=np.float64
    )
    home = np.asarray(HOME_POSE, dtype=np.float64)
    rust_applied_actions = (
        rust_applied_targets[:, list(POLICY_JOINTS)]
        - home[np.newaxis, list(POLICY_JOINTS)]
    ).astype(np.float32)
    actor_error = np.abs(actor - rust_observations)
    sensor_actor_error = actor_error.copy()
    sensor_actor_error[:, 34:48] = 0.0
    previous_chain_error = np.abs(
        rust_observations[1:, 34:48]
        - np.asarray(dumps["actions_rust"], dtype=np.float32)[:-1]
    )
    applied_action_error = np.abs(runtime_actions - rust_applied_actions)
    applied_target_error = np.abs(
        live_applied_targets[:, list(POLICY_JOINTS)]
        - rust_applied_targets[:, list(POLICY_JOINTS)]
    )
    mouth_overlay_delta = np.abs(
        live_applied_targets[:, MOUTH_INDEX] - rust_applied_targets[:, MOUTH_INDEX]
    )
    tolerance = 1.0e-7
    candidates: list[dict[str, Any]] = []
    divergent_observation = np.argwhere(sensor_actor_error > tolerance)
    if divergent_observation.size:
        step_index, feature_index = map(int, divergent_observation[0])
        block = next(
            name
            for name, indices in OBSERVATION_BLOCKS.items()
            if feature_index in range(indices.start or 0, indices.stop or 61)
        )
        candidates.append(
            {
                "kind": "observation",
                "step": int(trace_rows[step_index]["step"]),
                "feature": feature_index,
                "block": block,
                "task_actor": float(actor[step_index, feature_index]),
                "rust": float(rust_observations[step_index, feature_index]),
            }
        )
    divergent_action = np.argwhere(applied_action_error >= 1.0e-5)
    if divergent_action.size:
        step_index, action_index = map(int, divergent_action[0])
        candidates.append(
            {
                "kind": "applied_action",
                "step": int(trace_rows[step_index]["step"]),
                "feature": action_index,
                "runtime": float(runtime_actions[step_index, action_index]),
                "rust_replay": float(rust_applied_actions[step_index, action_index]),
            }
        )
    first_divergence = min(candidates, key=lambda row: row["step"], default=None)
    return {
        "schema": "microduck-runtime-trace-parity-v1",
        "policy_sha256": report.policy_sha256,
        "steps": len(trace_rows),
        "max_sensor_derived_observation_error": float(sensor_actor_error.max()),
        "max_task_actor_vs_runtime_observation_error": float(actor_error.max()),
        "max_task_vs_runtime_previous_action_error": float(actor_error[:, 34:48].max()),
        "max_runtime_previous_action_chain_error": float(
            previous_chain_error.max() if previous_chain_error.size else 0.0
        ),
        "per_block_task_actor_vs_runtime_error": {
            name: float(actor_error[:, indices].max())
            for name, indices in OBSERVATION_BLOCKS.items()
        },
        "previous_action_semantics": {
            "runtime": "previous raw policy output",
            "task_internal": "previous safety-applied action sent to physics",
        },
        "max_rust_vs_onnx_action_error": report.max_onnx_action_error,
        "max_rust_proposed_target_formula_error": report.max_target_error,
        "max_rust_applied_target_formula_error": report.max_applied_target_error,
        "max_runtime_applied_action_vs_rust_error": float(applied_action_error.max()),
        "max_runtime_applied_target_vs_rust_error": float(applied_target_error.max()),
        "max_runtime_mouth_overlay_delta": float(mouth_overlay_delta.max()),
        "mouth_semantics": "command overlay outside the 14D standing policy",
        "runtime_safety_clamp_steps": int(
            np.any(
                np.abs(rust_applied_targets - np.asarray(dumps["targets_rust"])) > 0.0,
                axis=1,
            ).sum()
        ),
        "first_divergence": first_divergence,
        "initial_previous_action": actor[0, 34:48].tolist(),
        "command_slots_all_zero": bool(np.all(actor[:, 48:61] == 0.0)),
        "dumps": {
            **dumps,
            "task_actor_observations": actor.tolist(),
            "runtime_actions": runtime_actions.tolist(),
            "runtime_applied_targets": live_applied_targets.tolist(),
            "rust_applied_actions": rust_applied_actions.tolist(),
        },
    }
