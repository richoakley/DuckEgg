"""Vectorized candidate rollouts in the registered MicroDuck locomotion task."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

from mjlab_microduck.eggroll.deployment import (
    DeploymentConditionProfile,
    DeploymentState,
)
from mjlab_microduck.eggroll.rollout import (
    TaskEnvironment,
    environment_state_tensors,
    make_environment,
)
from mjlab_microduck.sim.capability_runtime import (
    RegisteredTaskScenario,
    reset_registered_scenario,
)

from .foot_proof import WalkingCalibrationCase
from .locomotion_objective import (
    LocomotionObjectiveConfig,
    LocomotionTrajectoryAccumulator,
    to_numpy_episode,
)

ActionFunction = Callable[[torch.Tensor], torch.Tensor]
PREVIOUS_ACTION_SLICE = slice(34, 48)
COMMAND_SLICE = slice(48, 61)
PRODUCTION_COMMAND_ALPHA = 0.2
PRODUCTION_WALK_FILTER_ALPHA = (
    0.7,
    0.7,
    0.7,
    0.7,
    0.7,
    0.5,
    0.5,
    0.5,
    0.5,
    0.7,
    0.7,
    0.7,
    0.7,
    0.7,
)
STARTUP_ACTUATOR_FIELDS = (
    "friction_scale",
    "kp_scale",
    "kd_scale",
    "default_kp_scale",
    "default_kd_scale",
    "default_friction_scale",
    "vin_tensor",
    "vin_drop_gain",
    "effort_scale",
)


def _field_tensor(value: Any) -> torch.Tensor | None:
    if isinstance(value, torch.Tensor):
        return value
    tensor = getattr(value, "_tensor", None)
    return tensor if isinstance(tensor, torch.Tensor) else None


def _broadcast_world_tensor(destination: Any, source: torch.Tensor, num_envs: int) -> None:
    tensor = _field_tensor(destination)
    if tensor is None or tensor.ndim == 0 or tensor.shape[0] != num_envs:
        raise RuntimeError("startup world field is not expanded per environment")
    if source.shape[0] != 1 or source.shape[1:] != tensor.shape[1:]:
        raise RuntimeError("startup world field shape changed across runtimes")
    destination[:] = source.to(device=tensor.device, dtype=tensor.dtype).expand_as(tensor)


@dataclass(frozen=True)
class StartupWorldState:
    """A seed-bound physical identity sampled by MJLab startup events."""

    seed: int
    model_fields: tuple[tuple[str, torch.Tensor], ...]
    encoder_bias: torch.Tensor
    actuator_fields: tuple[tuple[tuple[str, torch.Tensor], ...], ...]
    imu_misalign_quat: torch.Tensor | None

    @classmethod
    def capture(cls, env: Any, *, seed: int) -> StartupWorldState:
        if env.num_envs != 1:
            raise ValueError("startup identities must be captured from one world")
        model_fields: list[tuple[str, torch.Tensor]] = []
        for name in env.event_manager.domain_randomization_fields:
            tensor = _field_tensor(getattr(env.sim.model, name))
            if tensor is not None and tensor.ndim > 0 and tensor.shape[0] == 1:
                model_fields.append((name, tensor.clone()))
        robot = env.scene["robot"]
        actuator_fields = []
        for actuator in robot.actuators:
            fields = []
            for name in STARTUP_ACTUATOR_FIELDS:
                value = getattr(actuator, name, None)
                if isinstance(value, torch.Tensor) and value.shape[0] == 1:
                    fields.append((name, value.clone()))
            actuator_fields.append(tuple(fields))
        imu = getattr(env, "_imu_misalign_quat", None)
        return cls(
            seed=seed,
            model_fields=tuple(model_fields),
            encoder_bias=robot.data.encoder_bias.clone(),
            actuator_fields=tuple(actuator_fields),
            imu_misalign_quat=imu.clone() if isinstance(imu, torch.Tensor) else None,
        )

    def apply(self, env: Any) -> None:
        num_envs = int(env.num_envs)
        for name, value in self.model_fields:
            _broadcast_world_tensor(getattr(env.sim.model, name), value, num_envs)
        robot = env.scene["robot"]
        _broadcast_world_tensor(robot.data.encoder_bias, self.encoder_bias, num_envs)
        if len(robot.actuators) != len(self.actuator_fields):
            raise RuntimeError("startup world actuator layout changed")
        for actuator, fields in zip(
            robot.actuators, self.actuator_fields, strict=True
        ):
            for name, value in fields:
                _broadcast_world_tensor(getattr(actuator, name), value, num_envs)
        if self.imu_misalign_quat is not None:
            _broadcast_world_tensor(
                env._imu_misalign_quat,
                self.imu_misalign_quat,
                num_envs,
            )


def capture_startup_world(
    *, task: str, seed: int, device: str, max_actuator_lag_steps: int
) -> StartupWorldState:
    """Construct the same seed-bound startup identity as production playback."""

    runtime = make_environment(
        task=task,
        num_envs=1,
        device=device,
        seed=seed,
        matched_candidates=False,
        max_actuator_lag_steps=max_actuator_lag_steps,
    )
    try:
        return StartupWorldState.capture(runtime.env, seed=seed)
    finally:
        runtime.close()


class ProductionWalkingTransport:
    """Mirror robotd's raw-action history and first-order target filters."""

    def __init__(self, *, num_envs: int, device: torch.device) -> None:
        self._alpha = torch.tensor(
            PRODUCTION_WALK_FILTER_ALPHA, device=device, dtype=torch.float32
        ).expand(num_envs, -1)
        self._previous_filtered: torch.Tensor | None = None
        self._command = torch.zeros((num_envs, 13), device=device)

    def policy_observations(
        self, observations: torch.Tensor, requested_command: tuple[float, ...]
    ) -> torch.Tensor:
        """Apply robotd's per-tick command EMA to the actor command block."""

        if observations.shape != (self._alpha.shape[0], 61):
            raise ValueError("walking actor observations must be [N, 61]")
        if len(requested_command) != 13:
            raise ValueError("walking command must contain 13 values")
        requested = torch.as_tensor(
            requested_command, device=observations.device, dtype=observations.dtype
        ).expand_as(self._command)
        self._command += PRODUCTION_COMMAND_ALPHA * (requested - self._command)
        actor = observations.clone()
        actor[:, COMMAND_SLICE] = self._command
        return actor

    def apply(self, raw_actions: torch.Tensor) -> torch.Tensor:
        if raw_actions.shape != self._alpha.shape:
            raise ValueError(
                f"raw walking actions must be {self._alpha.shape}, "
                f"got {raw_actions.shape}"
            )
        if self._previous_filtered is None:
            filtered = raw_actions
        else:
            filtered = (
                self._alpha * raw_actions
                + (1.0 - self._alpha) * self._previous_filtered
            )
        self._previous_filtered = filtered
        return filtered


def _with_production_previous_action(
    observations: dict[str, Any], raw_actions: torch.Tensor
) -> dict[str, Any]:
    """Replace MJLab's applied-action slot with robotd's previous raw output."""

    actor = observations["actor"].clone()
    actor[:, PREVIOUS_ACTION_SLICE] = raw_actions
    return {**observations, "actor": actor}


def _reset_completed_slots_without_advancing_observations(
    env: Any, done_ids: torch.Tensor
) -> None:
    """Reset inactive vector slots without a second observation/history update."""

    # ManagerBasedRlEnv.reset() recomputes observations for the entire vector,
    # which advances delay/history state a second time for candidates that are
    # still active. The pinned MJLab environment exposes the same per-index
    # reset primitive used by auto-reset; the next regular step writes the
    # reset state to simulation and computes observations exactly once.
    env._reset_idx(done_ids)


def _initial_heading(robot: Any) -> torch.Tensor:
    quat = robot.data.root_link_quat_w
    yaw = torch.atan2(
        2.0 * (quat[:, 0] * quat[:, 3] + quat[:, 1] * quat[:, 2]),
        1.0 - 2.0 * (quat[:, 2] ** 2 + quat[:, 3] ** 2),
    )
    return torch.stack((torch.cos(yaw), torch.sin(yaw)), dim=1)


def rollout_locomotion_episode(
    *,
    runtime: TaskEnvironment,
    case: WalkingCalibrationCase,
    profile: DeploymentConditionProfile,
    action_fn: ActionFunction,
    objective_config: LocomotionObjectiveConfig,
    startup_world: StartupWorldState | None = None,
) -> dict[str, np.ndarray]:
    """Run one command/seed for all candidates using common random numbers."""

    env = runtime.env
    scenario = RegisteredTaskScenario(
        scenario_id=case.case_id,
        task="Mjlab-Velocity-Flat-MicroDuck",
        seed=case.seed,
        profile_name=profile.name,
        profile_sha256=profile.sha256,
        command=case.command,
    )
    if startup_world is not None:
        if startup_world.seed != case.seed:
            raise ValueError("startup world seed does not match the rollout case")
        runtime.deployment_state.restore()
        startup_world.apply(env)
        runtime.deployment_state = DeploymentState.capture(env)
    observations = reset_registered_scenario(runtime, scenario, profile)
    robot = env.scene["robot"]
    accumulator = LocomotionTrajectoryAccumulator(
        num_envs=env.num_envs,
        step_dt=float(env.step_dt),
        horizon_steps=case.horizon_steps,
        command=tuple(float(value) for value in case.command[:3]),
        initial_xy=robot.data.root_link_pos_w[:, :2],
        forward_w=_initial_heading(robot),
        device=torch.device(env.device),
        config=objective_config,
    )
    active = torch.ones(env.num_envs, device=env.device, dtype=torch.bool)
    previous_actions = torch.zeros((env.num_envs, 14), device=env.device)
    transport = ProductionWalkingTransport(
        num_envs=env.num_envs, device=torch.device(env.device)
    )
    for _step in range(case.horizon_steps):
        policy_observations = transport.policy_observations(
            observations["actor"], case.command
        )
        raw_actions = action_fn(policy_observations)
        if raw_actions.shape != (env.num_envs, 14):
            raise RuntimeError(
                f"policy returned {raw_actions.shape}; expected [N, 14]"
            )
        if raw_actions.device != torch.device(env.device):
            raise RuntimeError("policy action device does not match environment")
        if not torch.isfinite(raw_actions).all():
            raise FloatingPointError("policy produced non-finite actions")
        actions = transport.apply(raw_actions)
        actions = torch.where(active[:, None], actions, 0.0)
        observations, rewards, terminated, truncated, _extras = env.step(actions)
        observations = _with_production_previous_action(observations, raw_actions)
        state = environment_state_tensors(env)
        velocity = torch.stack(
            (
                robot.data.root_link_lin_vel_b[:, 0],
                robot.data.root_link_lin_vel_b[:, 1],
                robot.data.root_link_ang_vel_b[:, 2],
            ),
            dim=1,
        )
        accumulator.update(
            root_xy=robot.data.root_link_pos_w[:, :2],
            velocity=velocity,
            trunk_height_m=state["trunk_height_m"],
            upright_cosine=state["upright_cosine"],
            action_rate_l2=torch.mean(
                torch.square(actions - previous_actions), dim=1
            ),
            reward=rewards,
            active=active,
        )
        previous_actions = actions
        done = terminated | truncated
        active &= ~done
        if not bool(torch.any(active)):
            break
        # Reset only completed slots after recording their terminal transition;
        # they remain masked out of every subsequent accumulator update.
        if bool(torch.any(done)):
            done_ids = done.nonzero(as_tuple=False).squeeze(-1)
            _reset_completed_slots_without_advancing_observations(env, done_ids)
    return to_numpy_episode(accumulator.finalize())


def evaluate_locomotion_bank(
    *,
    runtime: TaskEnvironment,
    cases: Sequence[WalkingCalibrationCase],
    profile: DeploymentConditionProfile,
    action_fn: ActionFunction,
    objective_config: LocomotionObjectiveConfig,
    startup_worlds: Sequence[StartupWorldState] | None = None,
) -> list[dict[str, np.ndarray]]:
    if not cases:
        raise ValueError("locomotion bank cannot be empty")
    if startup_worlds is not None and len(startup_worlds) != len(cases):
        raise ValueError("startup worlds must align one-to-one with cases")
    return [
        rollout_locomotion_episode(
            runtime=runtime,
            case=case,
            profile=profile,
            action_fn=action_fn,
            objective_config=objective_config,
            startup_world=(
                None if startup_worlds is None else startup_worlds[index]
            ),
        )
        for index, case in enumerate(cases)
    ]
