"""Production ``robotd`` bundle and capability-driver contracts.

The evaluator replaces exactly one runtime slot and leaves every surrounding
production policy intact.  This matters for transitions: testing a kick by
loading the kick network into every slot does not test the deployed system.
"""

from __future__ import annotations

import json
import socket
import threading
import time
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Self

from .registry import AutopatchRegistry

Mode = Literal["walk", "roller"]


@dataclass(frozen=True)
class RuntimePolicyBundle:
    """The complete policy set loaded around one artifact under test."""

    mode: Mode
    artifact_id: str
    capability_id: str
    under_test: Path
    slots: tuple[tuple[str, Path | None], ...]

    @classmethod
    def for_artifact(
        cls,
        *,
        registry: AutopatchRegistry,
        runtime_repo: Path,
        artifact_id: str,
        replacement_policy: Path | None = None,
        mode: Mode | None = None,
    ) -> RuntimePolicyBundle:
        artifact = registry.artifact(artifact_id)
        if mode is None:
            if len(artifact.runtime_modes) != 1:
                raise ValueError(
                    f"{artifact_id} supports {artifact.runtime_modes}; choose --mode"
                )
            mode = artifact.runtime_modes[0]  # type: ignore[assignment]
        if mode not in artifact.runtime_modes:
            raise ValueError(f"{artifact_id} is not a {mode!r}-mode artifact")
        policy_dir = runtime_repo / "example_policies"
        defaults: dict[str, str | None]
        if mode == "walk":
            defaults = {
                "walk": "alpha_walking.onnx",
                "stand": "alpha_stand.onnx",
                "sitstand": "alpha_sitstand.onnx",
                "ground_pick": "alpha_ground_pick.onnx",
                "kick_left": "ball_kick_left.onnx",
                "kick_right": "ball_kick_right.onnx",
                "roulade": "roulade.onnx",
            }
        else:
            defaults = {
                "walk": "roller.onnx",
                "stand": None,
                "sitstand": "alpha_sitstand.onnx",
                "ground_pick": "roller_crouch.onnx",
                "kick_left": "ball_kick_left.onnx",
                "kick_right": "ball_kick_right.onnx",
                "roulade": "roulade.onnx",
            }
        selected = replacement_policy or (policy_dir / artifact.filename)
        if not selected.is_file():
            raise FileNotFoundError(selected)
        slots: list[tuple[str, Path | None]] = []
        for slot, filename in defaults.items():
            path = None if filename is None else policy_dir / filename
            if slot == artifact.runtime_slot:
                path = selected
            if path is not None and not path.is_file():
                raise FileNotFoundError(path)
            slots.append((slot, path))
        return cls(
            mode=mode,
            artifact_id=artifact.artifact_id,
            capability_id=artifact.capability_id,
            under_test=selected.resolve(),
            slots=tuple(slots),
        )

    @property
    def slot_map(self) -> dict[str, Path | None]:
        return dict(self.slots)

    def render_robotd_toml(self) -> str:
        """Render only real ``robotd`` parameters; no evaluator-only policy path."""

        lines = ["[policy]", "enabled = true", f'mode = "{self.mode}"']
        for slot, path in self.slots:
            value = "none" if path is None else str(path.resolve())
            escaped = value.replace("\\", "\\\\").replace('"', '\\"')
            lines.append(f'{slot} = "{escaped}"')
        lines.extend(
            (
                "voltage_adapt = false",
                "",
                "[safety]",
                "limp_fall = false",
                "battery_empty_shutdown = false",
                "",
                "[audio]",
                "enabled = false",
                "greet = false",
                "",
            )
        )
        return "\n".join(lines)

    def active_action_scale(self) -> float:
        """Production scale while this artifact's network is active."""

        if self.capability_id == "legged-locomotion":
            return 0.9
        if self.capability_id == "roller-locomotion":
            return 0.8
        if self.capability_id == "roller-crouch":
            return 0.8
        return 1.0


@dataclass(frozen=True)
class DriverEvent:
    """One intent synchronized to an environment control step."""

    step: int
    method: str
    params: tuple[tuple[str, str | int | float | bool], ...]
    request: bool

    def params_dict(self) -> dict[str, str | int | float | bool]:
        return dict(self.params)


def capability_events(
    capability_id: str,
    *,
    horizon_steps: int,
    trigger_step: int = 5,
    return_step: int | None = None,
    command: Mapping[str, float] | None = None,
    side: Literal["left", "right"] = "right",
) -> tuple[DriverEvent, ...]:
    """Build the production intent stream for a capability episode."""

    if horizon_steps <= 0:
        raise ValueError("driver horizon must be positive")
    values = dict(command or {})
    if capability_id in {"legged-locomotion", "roller-locomotion"}:
        moving_params = tuple(
            sorted(
                {
                    "vx": float(values.get("vx", 0.1)),
                    "vy": float(values.get("vy", 0.0)),
                    "vyaw": float(values.get("vyaw", 0.0)),
                }.items()
            )
        )
        # Continuous intents expire. Refresh every control step so the real deadman
        # and command EMA see precisely the deployment protocol.
        return tuple(
            DriverEvent(
                step,
                "robot.move",
                (
                    (("vx", 0.0), ("vy", 0.0), ("vyaw", 0.0))
                    if return_step is not None and step >= return_step
                    else moving_params
                ),
                False,
            )
            for step in range(horizon_steps)
        )
    if capability_id == "stationary-body-control":
        if not values:
            return ()
        params = tuple(
            sorted(
                {
                    "z": float(values.get("z", 0.0)),
                    "roll": float(values.get("roll", 0.0)),
                    "pitch": float(values.get("pitch", 0.0)),
                    "active": True,
                }.items()
            )
        )
        return tuple(
            DriverEvent(step, "robot.pose", params, False)
            for step in range(horizon_steps)
        )
    if capability_id == "sit-stand-transition":
        if not 0 <= trigger_step < horizon_steps:
            raise ValueError("trigger step must fall inside the driver horizon")
        return_step = horizon_steps // 2 if return_step is None else return_step
        if not trigger_step < return_step < horizon_steps:
            raise ValueError("sit/stand return step must follow the sit trigger")
        params = (("skill", "sit_toggle"),)
        return (
            DriverEvent(trigger_step, "robot.do", params, True),
            DriverEvent(return_step, "robot.do", params, True),
        )
    skill = {
        "ground-pick": "ground_pick",
        "roller-crouch": "ground_pick",
        "forward-roll": "roulade",
        "ball-kick": f"kick_{side}",
    }.get(capability_id)
    if skill is None:
        raise ValueError(f"no production driver for capability {capability_id!r}")
    if not 0 <= trigger_step < horizon_steps:
        raise ValueError("trigger step must fall inside the driver horizon")
    return (DriverEvent(trigger_step, "robot.do", (("skill", skill),), True),)


class RobotdClient:
    """Minimal strict JSON-RPC client for synchronized evaluator intents."""

    def __init__(self, socket_path: Path, *, timeout: float = 5.0) -> None:
        self.socket_path = socket_path
        self.timeout = timeout
        self._socket: socket.socket | None = None
        self._stream: Any = None
        self._next_id = 1

    def connect(self) -> None:
        deadline = time.monotonic() + self.timeout
        while not self.socket_path.exists():
            if time.monotonic() >= deadline:
                raise TimeoutError(f"robotd socket did not appear: {self.socket_path}")
            time.sleep(0.01)
        connection = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        connection.settimeout(self.timeout)
        connection.connect(str(self.socket_path))
        self._socket = connection
        self._stream = connection.makefile("rwb", buffering=0)
        self.request("hello", {"api_version": 16})

    def close(self) -> None:
        if self._stream is not None:
            self._stream.close()
            self._stream = None
        if self._socket is not None:
            self._socket.close()
            self._socket = None

    def __enter__(self) -> Self:
        self.connect()
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()

    def _write(self, payload: Mapping[str, Any]) -> None:
        if self._stream is None:
            raise RuntimeError("RobotdClient is not connected")
        self._stream.write(
            json.dumps(payload, sort_keys=True, separators=(",", ":")).encode() + b"\n"
        )

    def notify(self, method: str, params: Mapping[str, Any]) -> None:
        self._write({"jsonrpc": "2.0", "method": method, "params": dict(params)})

    def request(self, method: str, params: Mapping[str, Any]) -> Any:
        request_id = self._next_id
        self._next_id += 1
        self._write(
            {
                "jsonrpc": "2.0",
                "id": request_id,
                "method": method,
                "params": dict(params),
            }
        )
        if self._stream is None:
            raise RuntimeError("RobotdClient is not connected")
        raw = self._stream.readline()
        if not raw:
            raise ConnectionError("robotd closed the evaluator socket")
        response = json.loads(raw)
        if response.get("id") != request_id:
            raise RuntimeError(f"unexpected robotd response id: {response}")
        if response.get("error") is not None:
            raise RuntimeError(f"robotd rejected {method}: {response['error']}")
        return response.get("result")

    def send_event(self, event: DriverEvent) -> Any:
        params = event.params_dict()
        if event.request:
            return self.request(event.method, params)
        self.notify(event.method, params)
        return None


class RobotdStateSubscriber:
    """Dedicated subscriber for the production ``robot.state`` stream."""

    def __init__(
        self, socket_path: Path, *, hz: int = 50, timeout: float = 5.0
    ) -> None:
        if hz <= 0 or hz > 50:
            raise ValueError("robotd state subscription must be in [1, 50] Hz")
        self.socket_path = socket_path
        self.hz = hz
        self.timeout = timeout
        self.subscription: Any = None
        self.frames: list[dict[str, Any]] = []
        self._socket: socket.socket | None = None
        self._stream: Any = None
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._error: Exception | None = None

    def _request(self, request_id: int, method: str, params: Mapping[str, Any]) -> Any:
        if self._stream is None:
            raise RuntimeError("state subscriber is not connected")
        payload = {
            "jsonrpc": "2.0",
            "id": request_id,
            "method": method,
            "params": dict(params),
        }
        self._stream.write(
            json.dumps(payload, sort_keys=True, separators=(",", ":")).encode() + b"\n"
        )
        raw = self._stream.readline()
        if not raw:
            raise ConnectionError(f"robotd closed while answering {method}")
        response = json.loads(raw)
        if response.get("id") != request_id or response.get("error") is not None:
            raise RuntimeError(f"robotd rejected state subscription call: {response}")
        return response.get("result")

    def connect(self) -> None:
        deadline = time.monotonic() + self.timeout
        while not self.socket_path.exists():
            if time.monotonic() >= deadline:
                raise TimeoutError(f"robotd socket did not appear: {self.socket_path}")
            time.sleep(0.01)
        connection = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        connection.settimeout(self.timeout)
        connection.connect(str(self.socket_path))
        self._socket = connection
        self._stream = connection.makefile("rwb", buffering=0)
        self._request(1, "hello", {"api_version": 16})
        self.subscription = self._request(2, "robot.subscribe", {"hz": self.hz})
        # A state frame is larger than an intent response. Blocking reads plus
        # shutdown-on-close guarantee one complete JSON line; a socket timeout
        # can surface a partial buffered line as if it were a complete frame.
        connection.settimeout(None)
        self._thread = threading.Thread(target=self._read, daemon=True)
        self._thread.start()

    def _read(self) -> None:
        assert self._stream is not None
        while not self._stop.is_set():
            try:
                raw = self._stream.readline()
            except (OSError, ValueError) as error:
                if not self._stop.is_set():
                    self._error = error
                return
            if not raw:
                if not self._stop.is_set():
                    self._error = ConnectionError("robotd closed the state stream")
                return
            try:
                message = json.loads(raw)
            except json.JSONDecodeError as error:
                if not self._stop.is_set():
                    self._error = error
                return
            if message.get("method") == "robot.state":
                params = message.get("params")
                if not isinstance(params, dict):
                    self._error = TypeError("robot.state params were not an object")
                    return
                self.frames.append(params)

    def close(self) -> tuple[dict[str, Any], ...]:
        self._stop.set()
        if self._socket is not None:
            try:
                self._socket.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
        if self._stream is not None:
            self._stream.close()
            self._stream = None
        if self._socket is not None:
            self._socket.close()
            self._socket = None
        if self._thread is not None:
            self._thread.join(timeout=2)
            self._thread = None
        if self._error is not None:
            raise RuntimeError("robotd state subscriber failed") from self._error
        return tuple(self.frames)


def drive_synchronized(
    client: RobotdClient,
    events: Iterable[DriverEvent],
    *,
    current_step: Callable[[], int],
    is_done: Callable[[], bool],
    timeout: float,
) -> tuple[dict[str, Any], ...]:
    """Send each intent after its preceding physics step and record the response."""

    deadline = time.monotonic() + timeout
    sent: list[dict[str, Any]] = []
    for event in events:
        while current_step() < event.step and not is_done():
            if time.monotonic() >= deadline:
                raise TimeoutError(f"driver stalled before step {event.step}")
            time.sleep(0.001)
        if is_done():
            break
        response = client.send_event(event)
        sent.append(
            {
                "step": event.step,
                "method": event.method,
                "params": event.params_dict(),
                "request": event.request,
                "response": response,
            }
        )
    return tuple(sent)
