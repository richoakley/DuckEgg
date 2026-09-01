"""Generic registered-mjlab task behind the production ``RobotIo`` seam.

The Rust daemon owns observation construction, policy selection, ONNX inference,
previous action, action scaling/filtering, safety, and timing. A task adapter owns
only the actual registered environment reset and capability diagnostics.
"""

from __future__ import annotations

import json
import socketserver
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import numpy as np
import torch

from mjlab_microduck.sim.body_server import (
    COLS,
    HOME_POSE,
    JOINT_NAMES,
    MOUTH_INDEX,
    NOMINAL_TEMP_C,
    NOMINAL_VOLTS,
    ROWS,
    Handler,
    Server,
)

EXPECTED_OBSERVATION_DIM = 61
EXPECTED_ACTION_DIM = 14
POLICY_JOINTS = tuple(
    index for index in range(len(JOINT_NAMES)) if index != MOUTH_INDEX
)


def actor_to_sensor_fixture(
    actor: np.ndarray, quat_wxyz: np.ndarray, *, fixture_id: str
) -> dict[str, Any]:
    """Convert a task actor's sensor blocks back to raw robot-unit fields."""

    actor = np.asarray(actor, dtype=np.float32)
    quat_wxyz = np.asarray(quat_wxyz, dtype=np.float64)
    if actor.shape != (EXPECTED_OBSERVATION_DIM,):
        raise ValueError(f"expected a 61D actor observation, got {actor.shape}")
    if quat_wxyz.shape != (4,):
        raise ValueError(f"expected a 4D wxyz quaternion, got {quat_wxyz.shape}")
    positions = np.asarray(HOME_POSE, dtype=np.float64)
    velocities = np.zeros(len(JOINT_NAMES), dtype=np.float64)
    positions[list(POLICY_JOINTS)] += actor[6:20].astype(np.float64)
    velocities[list(POLICY_JOINTS)] = actor[20:34].astype(np.float64)
    return {
        "fixture_id": fixture_id,
        "positions": positions.tolist(),
        "velocities": velocities.tolist(),
        "imu": {
            "gyro": actor[0:3].astype(np.float64).tolist(),
            "gravity": actor[3:6].astype(np.float64).tolist(),
            "quat": quat_wxyz.tolist(),
        },
        "previous_action": actor[34:48].tolist(),
        "twist": actor[48:51].tolist(),
        "head": actor[51:55].tolist(),
        "body": {
            "z": float(actor[57]),
            "roll": float(actor[58]),
            "pitch": float(actor[59]),
        },
        "action_scale": 1.0,
    }


def absolute_targets_to_action(
    wire_targets: list[float] | np.ndarray, *, action_scale: float = 1.0
) -> np.ndarray:
    """Invert production target construction at the ``RobotIo`` boundary."""

    targets = np.asarray(wire_targets, dtype=np.float64)
    if targets.shape != (len(JOINT_NAMES),):
        raise ValueError(f"expected {len(JOINT_NAMES)} targets, got {targets.shape}")
    if action_scale <= 0.0:
        raise ValueError("action_scale must be positive")
    return (
        (
            targets[list(POLICY_JOINTS)]
            - np.asarray(HOME_POSE, dtype=np.float64)[list(POLICY_JOINTS)]
        )
        / action_scale
    ).astype(np.float32)


class EpisodeMonitor(Protocol):
    """Capability-specific success semantics used by the generic transport."""

    def initial(self) -> None: ...

    def update(
        self,
        *,
        action: torch.Tensor,
        previous_action: torch.Tensor,
        reward: torch.Tensor,
        active: torch.Tensor,
    ) -> None: ...

    def trace_metrics(self) -> dict[str, float | bool | str]: ...

    def finalize(self, *, horizon_steps: int) -> dict[str, float | bool | str]: ...


@dataclass
class PreparedRuntimeTask:
    """Everything the transport needs after an adapter performs the real reset."""

    task: str
    scenario_id: str
    reset_label: str
    seed: int
    profile_name: str
    profile_sha256: str
    action_scale: float
    runtime: Any
    observations: dict[str, Any]
    monitor: EpisodeMonitor
    start_paused: bool = False
    horizon_steps: int | None = None


@dataclass(frozen=True)
class RuntimeTwinResult:
    summary: dict[str, float | bool | str]
    trace: tuple[dict[str, Any], ...]
    timing: dict[str, float | int]


class RegisteredRuntimeBody:
    """Synchronous production body for any prepared registered task."""

    index = 0

    def __init__(self, prepared: PreparedRuntimeTask) -> None:
        if prepared.action_scale <= 0.0:
            raise ValueError("prepared task action_scale must be positive")
        self.prepared = prepared
        self.runtime = prepared.runtime
        self.observations = prepared.observations
        self.monitor = prepared.monitor
        self.monitor.initial()
        self.active = torch.ones(1, device=self.env.device, dtype=torch.bool)
        self.previous_action = torch.zeros(
            (1, EXPECTED_ACTION_DIM), device=self.env.device
        )
        self.horizon = (
            int(self.env.max_episode_length)
            if prepared.horizon_steps is None
            else int(prepared.horizon_steps)
        )
        if self.horizon <= 0 or self.horizon > int(self.env.max_episode_length):
            raise ValueError("runtime horizon must fall inside the registered episode")
        self.step_count = 0
        self.torque_on = False
        self.gain = 0
        self.done = threading.Event()
        self.released = threading.Event()
        if not prepared.start_paused:
            self.released.set()
        self.lock = threading.RLock()
        self.trace: list[dict[str, Any]] = []
        self.video_states: list[tuple[np.ndarray, np.ndarray]] = []
        if self.runtime.render_mode == "rgb_array":
            self.video_states.append(self._sim_state())
        self._write_times: list[float] = []
        self._last_sensor_fixture: dict[str, Any] | None = None

    @property
    def env(self) -> Any:
        return self.runtime.env

    def hello(self) -> dict[str, Any]:
        return {
            "plant": "registered-mjlab-task-v1",
            "task": self.prepared.task,
            "profile": self.prepared.profile_name,
            "profile_sha256": self.prepared.profile_sha256,
            "scenario_id": self.prepared.scenario_id,
            "reset_label": self.prepared.reset_label,
            "seed": self.prepared.seed,
            "action_scale": self.prepared.action_scale,
        }

    def _actor(self) -> np.ndarray:
        actor = np.asarray(
            self.observations["actor"][0].detach().cpu(), dtype=np.float32
        )
        if actor.shape != (EXPECTED_OBSERVATION_DIM,):
            raise RuntimeError(f"expected a 61D actor observation, got {actor.shape}")
        return actor

    def _sensor_fixture(self) -> dict[str, Any]:
        quat = np.asarray(
            self.env.scene["robot"].data.root_link_quat_w[0].detach().cpu(),
            dtype=np.float64,
        )
        return actor_to_sensor_fixture(
            self._actor(),
            quat,
            fixture_id=f"{self.prepared.scenario_id}-step-{self.step_count:04d}",
        )

    def sensors(self) -> dict[str, Any]:
        with self.lock:
            fixture = self._sensor_fixture()
            self._last_sensor_fixture = fixture
            return {
                "positions": fixture["positions"],
                "velocities": fixture["velocities"],
                "currents_ma": [0.0] * len(JOINT_NAMES),
                "imu": fixture["imu"],
                "sim_time": self.step_count * float(self.env.step_dt),
            }

    def slow_sensors(self) -> dict[str, Any]:
        return {
            "volts": NOMINAL_VOLTS,
            "temps_c": [NOMINAL_TEMP_C] * len(JOINT_NAMES),
        }

    def depth(self) -> dict[str, Any]:
        return {
            "rows": ROWS,
            "cols": COLS,
            "distance_mm": [4000] * (ROWS * COLS),
            "status": [255] * (ROWS * COLS),
        }

    def set_gain(self, kp: int) -> None:
        self.gain = int(kp)

    def set_torque(self, on: bool) -> None:
        self.torque_on = bool(on)

    def release(self) -> None:
        """Allow scheduler writes to begin advancing the prepared episode."""

        self.released.set()

    def _sim_state(self) -> tuple[np.ndarray, np.ndarray]:
        return (
            np.asarray(self.env.sim.data.qpos[0].detach().cpu(), dtype=np.float32),
            np.asarray(self.env.sim.data.qvel[0].detach().cpu(), dtype=np.float32),
        )

    def set_targets(self, wire_targets: list[float]) -> None:
        if len(wire_targets) != len(JOINT_NAMES):
            raise ValueError(
                f"expected {len(JOINT_NAMES)} targets, got {len(wire_targets)}"
            )
        with self.lock:
            if self.done.is_set():
                return
            if not self.torque_on:
                raise RuntimeError(
                    "robotd wrote targets before enabling simulator torque"
                )
            if not self.released.is_set():
                return
            actor_before = self._actor()
            fixture = self._last_sensor_fixture or self._sensor_fixture()
            targets = np.asarray(wire_targets, dtype=np.float64)
            action_np = absolute_targets_to_action(
                targets, action_scale=self.prepared.action_scale
            )
            action = torch.as_tensor(
                action_np, device=self.env.device, dtype=torch.float32
            ).unsqueeze(0)
            observations, rewards, terminated, truncated, _ = self.env.step(action)
            done = terminated | truncated
            self.monitor.update(
                action=action,
                previous_action=self.previous_action,
                reward=rewards,
                active=self.active,
            )
            self.previous_action = action
            self.active &= ~done
            self.observations = observations
            self.step_count += 1
            now = time.perf_counter()
            self._write_times.append(now)
            self.trace.append(
                {
                    "step": self.step_count,
                    "fixture": fixture,
                    # Environment diagnostic only. The authoritative deployed
                    # policy input is reconstructed inside robotd and carried
                    # by its sim-eval policy trace.
                    "task_actor_observation_diagnostic": actor_before.tolist(),
                    "absolute_targets": targets.tolist(),
                    "runtime_action": action_np.tolist(),
                    "task_reward": float(rewards[0].item()),
                    "gain_register": self.gain,
                    "wall_time_s": now,
                    **self.monitor.trace_metrics(),
                }
            )
            if self.runtime.render_mode == "rgb_array":
                self.video_states.append(self._sim_state())
            self._last_sensor_fixture = None
            if bool(done[0]) or self.step_count >= self.horizon:
                self.done.set()

    def result(self) -> RuntimeTwinResult:
        if not self.done.is_set():
            raise RuntimeError("episode has not completed")
        summary = self.monitor.finalize(horizon_steps=self.horizon)
        intervals = np.diff(np.asarray(self._write_times, dtype=np.float64))
        timing: dict[str, float | int] = {
            "steps": self.step_count,
            "target_hz": 1.0 / float(self.env.step_dt),
            "wall_duration_s": (
                float(self._write_times[-1] - self._write_times[0])
                if len(self._write_times) > 1
                else 0.0
            ),
            "mean_interval_s": float(intervals.mean()) if intervals.size else 0.0,
            "p95_interval_s": (
                float(np.percentile(intervals, 95)) if intervals.size else 0.0
            ),
            "max_interval_s": float(intervals.max()) if intervals.size else 0.0,
        }
        return RuntimeTwinResult(summary, tuple(self.trace), timing)

    def save(self, output_dir: Path) -> RuntimeTwinResult:
        result = self.result()
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "summary.json").write_text(
            json.dumps(
                {
                    "schema": "microduck-production-runtime-task-episode-v1",
                    "backend": self.hello(),
                    "summary": result.summary,
                    "timing": result.timing,
                },
                indent=2,
                sort_keys=True,
            )
            + "\n"
        )
        with (output_dir / "trace.jsonl").open("w") as stream:
            for row in result.trace:
                stream.write(json.dumps(row, sort_keys=True) + "\n")
        if self.video_states:
            np.savez_compressed(
                output_dir / "render_states.npz",
                qpos=np.stack([state[0] for state in self.video_states]),
                qvel=np.stack([state[1] for state in self.video_states]),
            )
        return result

    def render_trace_video(self, output: Path) -> dict[str, float | int | str]:
        if self.runtime.render_mode != "rgb_array":
            raise RuntimeError("environment was not created with RGB rendering enabled")
        if len(self.video_states) != len(self.trace) + 1:
            raise RuntimeError("render-state count does not match runtime trace")
        frames: list[np.ndarray] = []
        max_state_write_error = 0.0
        for qpos, qvel in self.video_states:
            qpos_tensor = torch.as_tensor(qpos, device=self.env.device)
            qvel_tensor = torch.as_tensor(qvel, device=self.env.device)
            self.env.sim.data.qpos[0].copy_(qpos_tensor)
            self.env.sim.data.qvel[0].copy_(qvel_tensor)
            self.env.sim.forward()
            self.env.scene.update(dt=0.0)
            self.env.sim.sense()
            max_state_write_error = max(
                max_state_write_error,
                float(torch.max(torch.abs(self.env.sim.data.qpos[0] - qpos_tensor))),
                float(torch.max(torch.abs(self.env.sim.data.qvel[0] - qvel_tensor))),
            )
            frame = self.env.render()
            if frame is not None:
                frames.append(np.asarray(frame))
        if max_state_write_error > 0.0:
            raise RuntimeError(
                "video state playback diverged from live runtime trace: "
                f"state={max_state_write_error:.3g}"
            )
        if not frames:
            raise RuntimeError("renderer produced no frames")
        import mediapy

        output.parent.mkdir(parents=True, exist_ok=True)
        mediapy.write_video(output, frames, fps=float(self.env.metadata["render_fps"]))
        return {
            "frames": len(frames),
            "max_state_write_error": max_state_write_error,
            "render_timing": "captured post-integration qpos/qvel with forward",
        }

    def close(self) -> None:
        self.runtime.close()


def start_server(
    body: RegisteredRuntimeBody, host: str = "127.0.0.1", port: int = 0
) -> tuple[socketserver.ThreadingTCPServer, threading.Thread]:
    server = Server((host, port), Handler)
    server.body = body
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server, thread
