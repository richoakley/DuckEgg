"""Run capability episodes through the production Rust scheduler and RobotIo."""

from __future__ import annotations

import hashlib
import json
import os
import signal
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from mjlab_microduck.eggroll.deployment import (
    DeploymentConditionProfile,
    Scenario,
)
from mjlab_microduck.eggroll.policy_io import import_deployed_policy, onnx_actions
from mjlab_microduck.sim.body_server import HOME_POSE, MOUTH_INDEX
from mjlab_microduck.sim.capability_runtime import (
    RegisteredTaskScenario,
    make_ball_kick_runtime_body,
    make_continuous_locomotion_runtime_body,
    make_discrete_skill_runtime_body,
)
from mjlab_microduck.sim.registered_runtime import RegisteredRuntimeBody, start_server
from mjlab_microduck.sim.standup_runtime_twin import StandupRuntimeBody

from .registry import AutopatchRegistry
from .runtime import (
    RobotdClient,
    RobotdStateSubscriber,
    RuntimePolicyBundle,
    capability_events,
    drive_synchronized,
)
from .runtime_trace import audit_robotio_write_coverage


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _runtime_scale(label: str, mode: str) -> float:
    if label == "walk":
        return 0.8 if mode == "roller" else 0.9
    if label == "ground_pick" and mode == "roller":
        return 0.8
    return 1.0


def audit_runtime_policy_trace(
    *,
    states: list[dict[str, Any]],
    bundle: RuntimePolicyBundle,
    body_trace: tuple[dict[str, Any], ...],
    tolerance: float = 1.0e-5,
) -> dict[str, Any]:
    """Validate hidden Rust policy stages against ONNX and ``RobotIo``.

    The state subscription is intentionally lossy, so chain/filter checks only
    compare adjacent captured frames whose Rust tick counters are consecutive.
    Every captured policy frame is still checked independently for dimensions,
    finite values, graph parity, scale semantics, safety clamp and a target that
    actually crossed the simulator's ``RobotIo`` boundary.
    """

    traced = [frame for frame in states if frame.get("policy_trace") is not None]
    if not traced:
        raise RuntimeError("sim-eval produced no Rust policy trace frames")
    slot_map = bundle.slot_map
    body_targets = [
        np.asarray(row["absolute_targets"], dtype=np.float64) for row in body_trace
    ]
    max_action_error = 0.0
    max_scale_error = 0.0
    max_filter_error = 0.0
    max_state_target_error = 0.0
    max_safety_error = 0.0
    max_robotio_error = 0.0
    max_previous_action_error = 0.0
    consecutive_pairs = 0
    policy_frames = 0
    applied_ticks: list[int] = []
    incomplete_ticks: list[int] = []
    loaded: dict[Path, Any] = {}
    previous: dict[str, Any] | None = None
    home = np.asarray(HOME_POSE, dtype=np.float64)
    policy_joints = [index for index in range(home.size) if index != MOUTH_INDEX]
    head_joints = range(5, 9)
    filtered_leg_joints = [
        index
        for index in range(home.size)
        if index not in head_joints and index != MOUTH_INDEX
    ]

    for frame in traced:
        trace = frame["policy_trace"]
        observation = np.asarray(trace["observation"], dtype=np.float32)
        raw_action = np.asarray(trace["raw_action"], dtype=np.float32)
        unfiltered = np.asarray(trace["unfiltered_targets"], dtype=np.float64)
        filtered = np.asarray(trace["filtered_targets"], dtype=np.float64)
        applied = np.asarray(trace["applied_targets"], dtype=np.float64)
        if observation.shape != (61,) or raw_action.shape != (14,):
            raise RuntimeError("Rust policy trace changed the 61D -> 14D API")
        if any(value.shape != (15,) for value in (unfiltered, filtered)):
            raise RuntimeError("Rust policy trace targets must contain all 15 joints")
        if applied.shape not in {(0,), (15,)}:
            raise RuntimeError("Rust applied-target trace must be empty or 15D")
        if not all(
            np.isfinite(value).all()
            for value in (observation, raw_action, unfiltered, filtered, applied)
        ):
            raise RuntimeError("Rust policy trace contains non-finite values")
        slot = frame.get("policy_slot")
        policy_path = slot_map.get(str(slot)) if slot is not None else None
        if policy_path is None:
            raise RuntimeError(
                f"policy frame has no bound runtime slot: {frame['policy']!r}"
            )
        policy_path = policy_path.resolve()
        policy = loaded.get(policy_path)
        if policy is None:
            policy = import_deployed_policy(policy_path)
            loaded[policy_path] = policy
        expected_action = onnx_actions(policy.source_model, observation[None, :])[0]
        max_action_error = max(
            max_action_error,
            float(np.max(np.abs(expected_action - raw_action))),
        )

        scale = _runtime_scale(str(frame["policy"]), bundle.mode)
        expected_unfiltered = home.copy()
        expected_unfiltered[policy_joints] += scale * raw_action.astype(np.float64)
        max_scale_error = max(
            max_scale_error,
            float(np.max(np.abs(expected_unfiltered - unfiltered))),
        )
        max_state_target_error = max(
            max_state_target_error,
            float(
                np.max(
                    np.abs(
                        np.asarray(frame["targets"], dtype=np.float64)[policy_joints]
                        - filtered[policy_joints]
                    )
                )
            ),
        )
        if applied.shape == (15,):
            applied_ticks.append(int(trace["tick"]))
            expected_applied = np.clip(filtered, -np.pi, np.pi)
            max_safety_error = max(
                max_safety_error,
                float(
                    np.max(
                        np.abs(expected_applied[policy_joints] - applied[policy_joints])
                    )
                ),
            )
            if not body_targets:
                raise RuntimeError(
                    "runtime body trace contains no RobotIo target writes"
                )
            max_robotio_error = max(
                max_robotio_error,
                min(float(np.max(np.abs(target - applied))) for target in body_targets),
            )
        else:
            # Preserve every failed write for the frozen coverage classifier.
            # It distinguishes a recovered mid-episode transport gap from the
            # one allowed final tick whose write discovers body closure.
            incomplete_ticks.append(int(trace["tick"]))

        if previous is not None and int(trace["tick"]) == int(previous["tick"]) + 1:
            consecutive_pairs += 1
            previous_action = np.asarray(previous["raw_action"], dtype=np.float32)
            max_previous_action_error = max(
                max_previous_action_error,
                float(np.max(np.abs(observation[34:48] - previous_action))),
            )
            previous_filtered = np.asarray(
                previous["filtered_targets"], dtype=np.float64
            )
            expected_filtered = unfiltered.copy()
            expected_filtered[list(head_joints)] = (
                0.5 * unfiltered[list(head_joints)]
                + 0.5 * previous_filtered[list(head_joints)]
            )
            expected_filtered[filtered_leg_joints] = (
                0.7 * unfiltered[filtered_leg_joints]
                + 0.3 * previous_filtered[filtered_leg_joints]
            )
            max_filter_error = max(
                max_filter_error,
                float(np.max(np.abs(expected_filtered - filtered))),
            )
        previous = trace
        policy_frames += 1

    write_coverage = audit_robotio_write_coverage(
        applied_ticks=applied_ticks,
        unapplied_ticks=incomplete_ticks,
    )

    errors = {
        "onnx_raw_action": max_action_error,
        "action_scale_and_home_offset": max_scale_error,
        "lowpass_filter": max_filter_error,
        "state_filtered_target": max_state_target_error,
        "safety_applied_target": max_safety_error,
        "robotio_applied_target": max_robotio_error,
        "previous_raw_action": max_previous_action_error,
    }
    failed = {name: value for name, value in errors.items() if value > tolerance}
    if failed:
        raise RuntimeError(f"Rust runtime trace parity failed: {failed}")
    return {
        "status": "pass",
        "tolerance": tolerance,
        "captured_policy_frames": policy_frames,
        "robotio_write_coverage": write_coverage,
        "recovered_robotio_write_failures": len(
            write_coverage["recovered_robotio_write_failure_ticks"]
        ),
        "post_episode_write_failures": len(
            write_coverage["post_episode_write_failure_ticks"]
        ),
        "consecutive_tick_pairs": consecutive_pairs,
        "state_subscription_is_lossy": True,
        "max_abs_error": errors,
    }


@dataclass(frozen=True)
class RuntimeEvaluationRequest:
    artifact_id: str
    task: str
    seed: int
    side: str = "right"
    command: tuple[float, ...] = (0.0,) * 13
    device: str = "cpu"
    record_video: bool = False
    timeout_s: float = 30.0
    horizon_steps: int = 250
    reset_label: str = "standing"
    return_step: int | None = None

    def __post_init__(self) -> None:
        if self.side not in {"left", "right"}:
            raise ValueError("side must be left or right")
        if len(self.command) != 13:
            raise ValueError("runtime evaluation command must be 13D")
        if self.timeout_s <= 0.0:
            raise ValueError("timeout must be positive")
        if self.horizon_steps <= 0:
            raise ValueError("runtime evaluation horizon must be positive")
        if self.return_step is not None and not (
            0 < self.return_step < self.horizon_steps
        ):
            raise ValueError("return_step must fall strictly inside the horizon")


def _prepare_body(
    *,
    request: RuntimeEvaluationRequest,
    bundle: RuntimePolicyBundle,
    profile: DeploymentConditionProfile,
) -> RegisteredRuntimeBody:
    scenario = RegisteredTaskScenario(
        scenario_id=f"{request.artifact_id}-{request.task}-{request.seed}",
        task=request.task,
        seed=request.seed,
        profile_name=profile.name,
        profile_sha256=profile.sha256,
        command=request.command,
    )
    if bundle.capability_id == "ball-kick":
        if any(value != 0.0 for value in request.command):
            raise ValueError("BallKick runtime forces all 13 command slots to zero")
        return make_ball_kick_runtime_body(
            scenario=scenario,
            profile=profile,
            action_scale=bundle.active_action_scale(),
            side=request.side,
            device=request.device,
            record_video=request.record_video,
        )
    if (
        bundle.capability_id == "stationary-body-control"
        and request.task == "Mjlab-StandUp-Flat-MicroDuck"
    ):
        standup_scenario = Scenario(
            scenario_id=f"{request.artifact_id}-{request.task}-{request.seed}",
            pose=request.reset_label,
            seed=request.seed,
            profile_name=profile.name,
            profile_sha256=profile.sha256,
            command=request.command,
        )
        return StandupRuntimeBody(
            scenario=standup_scenario,
            profile=profile,
            device=request.device,
            record_video=request.record_video,
            start_paused=True,
            horizon_steps=request.horizon_steps,
        )
    if bundle.capability_id in {"legged-locomotion", "roller-locomotion"}:
        return make_continuous_locomotion_runtime_body(
            scenario=scenario,
            profile=profile,
            capability_id=bundle.capability_id,
            action_scale=bundle.active_action_scale(),
            horizon_steps=request.horizon_steps,
            device=request.device,
            record_video=request.record_video,
        )
    if bundle.capability_id in {
        "sit-stand-transition",
        "ground-pick",
        "roller-crouch",
        "forward-roll",
    }:
        return make_discrete_skill_runtime_body(
            scenario=scenario,
            profile=profile,
            capability_id=bundle.capability_id,
            horizon_steps=request.horizon_steps,
            device=request.device,
            record_video=request.record_video,
        )
    raise NotImplementedError(
        f"actual-runtime adapter is not implemented for {bundle.capability_id!r}"
    )


def run_runtime_evaluation(
    *,
    registry: AutopatchRegistry,
    runtime_repo: Path,
    robotd: Path,
    ort_dylib: Path,
    output_dir: Path,
    request: RuntimeEvaluationRequest,
    profile: DeploymentConditionProfile,
    replacement_policy: Path | None = None,
    mode: str | None = None,
) -> dict[str, Any]:
    """Execute one actual-task episode through the complete production runtime."""

    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"output directory is not empty: {output_dir}")
    for label, path in (
        ("runtime repo", runtime_repo),
        ("robotd", robotd),
        ("ORT", ort_dylib),
    ):
        if not path.exists():
            raise FileNotFoundError(f"{label} not found: {path}")
    bundle = RuntimePolicyBundle.for_artifact(
        registry=registry,
        runtime_repo=runtime_repo,
        artifact_id=request.artifact_id,
        replacement_policy=replacement_policy,
        mode=mode,  # type: ignore[arg-type]
    )
    artifact = registry.artifact(request.artifact_id)
    if request.task not in artifact.task_ids:
        raise ValueError(
            f"{request.task!r} is not registered for {request.artifact_id}"
        )
    body = _prepare_body(request=request, bundle=bundle, profile=profile)
    output_dir.mkdir(parents=True, exist_ok=True)
    server, server_thread = start_server(body)
    host, port = server.server_address
    process: subprocess.Popen[str] | None = None
    driver_trace: tuple[dict[str, Any], ...] = ()
    enable_result: Any = None
    command: list[str] = []
    run_error: Exception | None = None
    subscriber: RobotdStateSubscriber | None = None
    state_frames: tuple[dict[str, Any], ...] = ()
    try:
        with tempfile.TemporaryDirectory(prefix="eggroll-autopatch-runtime-") as temp:
            scratch = Path(temp)
            params = scratch / "robotd.toml"
            socket_path = scratch / "robotd.sock"
            params.write_text(bundle.render_robotd_toml())
            environment = os.environ.copy()
            environment["ORT_DYLIB_PATH"] = str(ort_dylib.resolve())
            environment.setdefault("RUST_LOG", "info")
            command = [
                str(robotd.resolve()),
                "--params",
                str(params),
                "--socket",
                str(socket_path),
                "--sim",
                f"{host}:{port}",
                "--sim-eval",
                "--sim-eval-wait",
            ]
            process = subprocess.Popen(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                env=environment,
            )
            subscriber = RobotdStateSubscriber(
                socket_path, hz=50, timeout=request.timeout_s
            )
            subscriber.connect()
            events = capability_events(
                bundle.capability_id,
                horizon_steps=body.horizon,
                trigger_step=0,
                return_step=request.return_step,
                command={
                    "vx": request.command[0],
                    "vy": request.command[1],
                    "vyaw": request.command[2],
                },
                side=request.side,  # type: ignore[arg-type]
            )
            at_start = tuple(event for event in events if event.step == 0)
            later = tuple(event for event in events if event.step > 0)
            with RobotdClient(socket_path, timeout=request.timeout_s) as client:
                sent: list[dict[str, Any]] = []
                for event in at_start:
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
                enable_result = client.request(
                    "robot.enable", {"on": True, "toggle": False}
                )
                body.release()
                sent.extend(
                    drive_synchronized(
                        client,
                        later,
                        current_step=lambda: body.step_count,
                        is_done=body.done.is_set,
                        timeout=request.timeout_s,
                    )
                )
                driver_trace = tuple(sent)
                if not body.done.wait(request.timeout_s):
                    raise TimeoutError(
                        f"runtime did not finish {body.horizon} steps in "
                        f"{request.timeout_s}s"
                    )
    except Exception as error:  # noqa: BLE001 - close the prepared simulator before reraising
        run_error = error
    finally:
        if subscriber is not None:
            try:
                state_frames = subscriber.close()
            except Exception as error:  # noqa: BLE001 - preserve primary evaluator failure
                if run_error is None:
                    run_error = error
        if process is not None and process.poll() is None:
            process.send_signal(signal.SIGTERM)
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=5)
        runtime_log = ""
        if process is not None and process.stdout is not None:
            runtime_log = process.stdout.read()
        (output_dir / "robotd.log").write_text(runtime_log)
        server.shutdown()
        server.server_close()
        server_thread.join(timeout=2)

    if run_error is not None:
        body.close()
        raise run_error

    try:
        result = body.save(output_dir)
        video_replay = (
            body.render_trace_video(output_dir / "episode.mp4")
            if request.record_video
            else None
        )
        label_to_slot = {
            "walk": "walk",
            "stand": "stand",
            "sit": "sitstand",
            "rise": "sitstand",
            "ground_pick": "ground_pick",
            "kick_left": "kick_left",
            "kick_right": "kick_right",
            "roulade": "roulade",
        }
        slot_map = bundle.slot_map
        enriched_states: list[dict[str, Any]] = []
        policy_sequence: list[dict[str, Any]] = []
        previous_label: str | None = None
        for frame in state_frames:
            enriched = dict(frame)
            label = str(frame.get("policy", "unknown"))
            slot = label_to_slot.get(label)
            policy_path = slot_map.get(slot) if slot is not None else None
            enriched["policy_slot"] = slot
            enriched["policy_sha256"] = (
                sha256_file(policy_path) if policy_path is not None else None
            )
            enriched_states.append(enriched)
            if label != previous_label:
                policy_sequence.append(
                    {
                        "state_index": len(enriched_states) - 1,
                        "t": frame.get("t"),
                        "policy": label,
                        "policy_slot": slot,
                        "policy_sha256": enriched["policy_sha256"],
                    }
                )
                previous_label = label
        with (output_dir / "robotd_state.jsonl").open("w") as stream:
            for frame in enriched_states:
                stream.write(json.dumps(frame, sort_keys=True) + "\n")
        driver_events_path = output_dir / "driver_events.jsonl"
        with driver_events_path.open("w") as stream:
            for event in driver_trace:
                stream.write(json.dumps(event, sort_keys=True) + "\n")
        trace_audit = audit_runtime_policy_trace(
            states=enriched_states,
            bundle=bundle,
            body_trace=result.trace,
        )
        (output_dir / "runtime_trace_audit.json").write_text(
            json.dumps(trace_audit, indent=2, sort_keys=True) + "\n"
        )
        scenario = body.prepared
        manifest = {
            "schema": "eggroll-autopatch-runtime-evaluation-v1",
            "claim_scope": "production-runtime digital twin; no physical robot",
            "artifact": {
                "artifact_id": artifact.artifact_id,
                "source_sha256": artifact.expected_sha256,
                "evaluated_path": str(bundle.under_test),
                "evaluated_sha256": sha256_file(bundle.under_test),
                "runtime_slot": artifact.runtime_slot,
            },
            "capability": {
                "capability_id": bundle.capability_id,
                "spec_sha256": registry.capability(bundle.capability_id).sha256,
            },
            "runtime_bundle": {
                "mode": bundle.mode,
                "slots": {
                    slot: None if path is None else str(path)
                    for slot, path in bundle.slots
                },
            },
            "runtime": {
                "robotd": str(robotd.resolve()),
                "robotd_sha256": sha256_file(robotd),
                "arguments": command[1:],
            },
            "scenario": {
                "scenario_id": scenario.scenario_id,
                "task": scenario.task,
                "reset_label": scenario.reset_label,
                "seed": scenario.seed,
                "command": list(request.command),
            },
            "profile": profile.canonical_dict(),
            "profile_sha256": profile.sha256,
            "driver": {
                "event_count": len(driver_trace),
                "events_sha256": sha256_file(driver_events_path),
                "enable_result": enable_result,
                "state_subscription": None
                if subscriber is None
                else subscriber.subscription,
                "state_frames": len(enriched_states),
                "policy_sequence": policy_sequence,
            },
            "artifacts": {
                "summary": "summary.json",
                "trace": "trace.jsonl",
                "trace_scope": (
                    "raw task sensors, RobotIo-applied targets, and task diagnostics; "
                    "the task actor observation is diagnostic, not the Rust policy input"
                ),
                "robotd_state": "robotd_state.jsonl",
                "robotd_state_scope": (
                    "exact Rust 61D observation, raw ONNX action, pre/post-filter "
                    "targets, safety-applied targets, policy label and resolved SHA"
                ),
                "driver_events": "driver_events.jsonl",
                "runtime_trace_audit": "runtime_trace_audit.json",
                "robotd_log": "robotd.log",
                "render_states": "render_states.npz" if video_replay else None,
                "video": "episode.mp4" if video_replay else None,
            },
            "video_replay": video_replay,
            "runtime_trace_audit": trace_audit,
            "result": result.summary,
            "timing": result.timing,
        }
        (output_dir / "manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n"
        )
        return manifest
    finally:
        body.close()
