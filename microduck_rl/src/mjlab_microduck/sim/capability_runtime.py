"""Capability adapters for the generic production-runtime transport.

Adapters own reset and success semantics for a registered task. They do not
run ONNX, choose a network, scale actions, or emulate scheduler state; those
remain in the Rust daemon.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from mjlab.managers.event_manager import RecomputeLevel

from mjlab_microduck.eggroll.deployment import (
    DeploymentConditionProfile,
    runtime_lag_capacity,
)
from mjlab_microduck.eggroll.rollout import (
    TaskEnvironment,
    _synchronize_candidates,
    environment_state_tensors,
    make_environment,
    set_command_block,
)
from mjlab_microduck.sim.registered_runtime import (
    EpisodeMonitor,
    PreparedRuntimeTask,
    RegisteredRuntimeBody,
)

ZERO_COMMAND = (0.0,) * 13


@dataclass(frozen=True)
class RegisteredTaskScenario:
    """One deterministic draw from a task's real reset distribution."""

    scenario_id: str
    task: str
    seed: int
    profile_name: str
    profile_sha256: str
    command: tuple[float, ...] = ZERO_COMMAND
    reset_label: str = "registered-task-distribution"

    def __post_init__(self) -> None:
        if not self.scenario_id or not self.task or not self.reset_label:
            raise ValueError("scenario id, task, and reset label cannot be empty")
        if len(self.command) != 13 or not np.isfinite(self.command).all():
            raise ValueError("scenario command must contain 13 finite values")

    def to_dict(self) -> dict[str, Any]:
        return {
            "scenario_id": self.scenario_id,
            "task": self.task,
            "seed": self.seed,
            "profile_name": self.profile_name,
            "profile_sha256": self.profile_sha256,
            "command": list(self.command),
            "reset_label": self.reset_label,
        }


def reset_registered_scenario(
    runtime: TaskEnvironment,
    scenario: RegisteredTaskScenario,
    profile: DeploymentConditionProfile,
) -> dict[str, Any]:
    """Reset the actual registered task and then freeze its deployment command."""

    if (
        scenario.profile_name != profile.name
        or scenario.profile_sha256 != profile.sha256
    ):
        raise ValueError("scenario deployment-profile identity does not match")
    env = runtime.env
    env.common_step_counter = 0
    env._sim_step_counter = 0
    env.seed(scenario.seed)
    env.reset()
    if env.num_envs > 1:
        _synchronize_candidates(env)
    set_command_block(env, scenario.command)
    runtime.deployment_state.apply(profile)
    # Geometry/contact fields have derived Warp constants.  Without this step
    # the bridge tensor changes while collision continues to use the compiled
    # nominal transform, producing a dangerously convincing no-op condition.
    env.sim.recompute_constants(RecomputeLevel.set_const)
    env.sim.forward()
    env.scene.update(dt=0.0)
    env.sim.sense()
    env.action_manager.reset()
    env.observation_manager.reset()
    env.observation_manager._obs_buffer = None
    env.obs_buf = env.observation_manager.compute(update_history=True)
    actor = env.obs_buf["actor"]
    if actor.shape != (env.num_envs, 61):
        raise RuntimeError(
            f"expected actor observations [{env.num_envs},61], got {actor.shape}"
        )
    expected = torch.as_tensor(scenario.command, device=env.device, dtype=actor.dtype)
    if not torch.allclose(actor[0, 48:61], expected, atol=1.0e-6, rtol=0.0):
        raise RuntimeError(
            "registered task did not expose the requested deployed command block"
        )
    return env.obs_buf


class BallKickEpisodeMonitor(EpisodeMonitor):
    """Object displacement plus post-skill stability acceptance.

    BallKick has shaped rewards but no registered binary success predicate. The
    acceptance below is therefore explicitly versioned evaluator semantics:
    reach 75% of the task's 1 m/s target, move the ball at least 10 cm forward,
    and finish with at least 0.5 s of continuous supported upright hold.
    """

    TARGET_SPEED_MPS = 1.0
    MIN_SPEED_FRACTION = 0.75
    MIN_DISPLACEMENT_M = 0.10
    MIN_TERMINAL_HOLD_S = 0.50
    MIN_STABLE_HEIGHT_M = 0.10
    MIN_UPRIGHT_COSINE = 0.85
    ACCEPTANCE_ID = "ball-kick-v1-speed75-displacement10cm-terminal-hold500ms"

    def __init__(self, runtime: TaskEnvironment) -> None:
        self.runtime = runtime
        self.env = runtime.env
        self._ball = self.env.scene["ball"]
        self._initial_pos = self._ball.data.root_link_pos_w[0, :2].clone()
        self._direction = self.env._ball_kick_dir_w[0].clone()
        self._return = 0.0
        self._steps = 0
        self._max_forward_speed = 0.0
        self._max_forward_displacement = 0.0
        self._terminal_hold_steps = 0
        self._latest = self._sample()

    def _sample(self) -> dict[str, float | bool]:
        state = environment_state_tensors(self.env)
        velocity = self._ball.data.root_link_lin_vel_w[0, :2]
        displacement = self._ball.data.root_link_pos_w[0, :2] - self._initial_pos
        forward_speed = float(torch.dot(velocity, self._direction).item())
        forward_displacement = float(torch.dot(displacement, self._direction).item())
        height = float(state["trunk_height_m"][0].item())
        upright = float(state["upright_cosine"][0].item())
        stable = (
            height >= self.MIN_STABLE_HEIGHT_M and upright >= self.MIN_UPRIGHT_COSINE
        )
        return {
            "ball_forward_speed_mps": forward_speed,
            "ball_forward_displacement_m": forward_displacement,
            "trunk_height_m": height,
            "upright_cosine": upright,
            "supported_upright": stable,
        }

    def initial(self) -> None:
        self._latest = self._sample()

    def update(
        self,
        *,
        action: torch.Tensor,
        previous_action: torch.Tensor,
        reward: torch.Tensor,
        active: torch.Tensor,
    ) -> None:
        del action, previous_action, active
        self._latest = self._sample()
        self._steps += 1
        self._return += float(reward[0].item())
        self._max_forward_speed = max(
            self._max_forward_speed,
            float(self._latest["ball_forward_speed_mps"]),
        )
        self._max_forward_displacement = max(
            self._max_forward_displacement,
            float(self._latest["ball_forward_displacement_m"]),
        )
        if bool(self._latest["supported_upright"]):
            self._terminal_hold_steps += 1
        else:
            self._terminal_hold_steps = 0

    def trace_metrics(self) -> dict[str, float | bool]:
        return dict(self._latest)

    def finalize(self, *, horizon_steps: int) -> dict[str, float | bool]:
        del horizon_steps
        dt = float(self.env.step_dt)
        hold_s = self._terminal_hold_steps * dt
        speed_pass = self._max_forward_speed >= (
            self.TARGET_SPEED_MPS * self.MIN_SPEED_FRACTION
        )
        displacement_pass = self._max_forward_displacement >= self.MIN_DISPLACEMENT_M
        stable_pass = hold_s >= self.MIN_TERMINAL_HOLD_S
        return {
            "task_has_binary_success": False,
            "acceptance_id": self.ACCEPTANCE_ID,
            "terminal_success": speed_pass and displacement_pass and stable_pass,
            "ball_speed_pass": speed_pass,
            "ball_displacement_pass": displacement_pass,
            "terminal_stability_pass": stable_pass,
            "max_ball_forward_speed_mps": self._max_forward_speed,
            "max_ball_forward_displacement_m": self._max_forward_displacement,
            "final_ball_forward_speed_mps": float(
                self._latest["ball_forward_speed_mps"]
            ),
            "final_ball_forward_displacement_m": float(
                self._latest["ball_forward_displacement_m"]
            ),
            "final_trunk_height_m": float(self._latest["trunk_height_m"]),
            "final_upright_cosine": float(self._latest["upright_cosine"]),
            "terminal_upright_hold_s": hold_s,
            "total_return": self._return,
            "episode_steps": float(self._steps),
        }


class ContinuousLocomotionMonitor(EpisodeMonitor):
    """Task-independent tracking/stability diagnostics for legs or rollers."""

    MIN_UPRIGHT_COSINE = 0.85
    MIN_TRUNK_HEIGHT_M = 0.09
    MIN_UPRIGHT_FRACTION = 0.90
    MIN_PROGRESS_FRACTION = 0.25

    def __init__(
        self,
        runtime: TaskEnvironment,
        *,
        capability_id: str,
        command: tuple[float, ...],
    ) -> None:
        if capability_id not in {"legged-locomotion", "roller-locomotion"}:
            raise ValueError(f"not a continuous locomotion capability: {capability_id}")
        self.runtime = runtime
        self.env = runtime.env
        self.capability_id = capability_id
        self.command = torch.as_tensor(command[:3], device=self.env.device)
        robot = self.env.scene["robot"]
        self._initial_xy = robot.data.root_link_pos_w[0, :2].clone()
        quat = robot.data.root_link_quat_w[0]
        yaw = torch.atan2(
            2.0 * (quat[0] * quat[3] + quat[1] * quat[2]),
            1.0 - 2.0 * (quat[2] * quat[2] + quat[3] * quat[3]),
        )
        self._forward_w = torch.stack((torch.cos(yaw), torch.sin(yaw)))
        self._return = 0.0
        self._steps = 0
        self._upright_steps = 0
        self._sum_velocity = torch.zeros(3, device=self.env.device)
        self._sum_squared_error = torch.zeros(3, device=self.env.device)
        self._latest = self._sample()

    @property
    def acceptance_id(self) -> str:
        return f"{self.capability_id}-v1-upright90-progress25pct-terminal-stable"

    def _sample(self) -> dict[str, float | bool]:
        robot = self.env.scene["robot"]
        velocity = torch.stack(
            (
                robot.data.root_link_lin_vel_b[0, 0],
                robot.data.root_link_lin_vel_b[0, 1],
                robot.data.root_link_ang_vel_b[0, 2],
            )
        )
        state = environment_state_tensors(self.env)
        height = float(state["trunk_height_m"][0].item())
        upright = float(state["upright_cosine"][0].item())
        stable = (
            height >= self.MIN_TRUNK_HEIGHT_M and upright >= self.MIN_UPRIGHT_COSINE
        )
        displacement = robot.data.root_link_pos_w[0, :2] - self._initial_xy
        return {
            "forward_velocity_mps": float(velocity[0].item()),
            "lateral_velocity_mps": float(velocity[1].item()),
            "yaw_rate_rps": float(velocity[2].item()),
            "forward_displacement_m": float(
                torch.dot(displacement, self._forward_w).item()
            ),
            "trunk_height_m": height,
            "upright_cosine": upright,
            "supported_upright": stable,
        }

    def initial(self) -> None:
        self._latest = self._sample()

    def update(
        self,
        *,
        action: torch.Tensor,
        previous_action: torch.Tensor,
        reward: torch.Tensor,
        active: torch.Tensor,
    ) -> None:
        del action, previous_action, active
        self._latest = self._sample()
        velocity = torch.tensor(
            [
                self._latest["forward_velocity_mps"],
                self._latest["lateral_velocity_mps"],
                self._latest["yaw_rate_rps"],
            ],
            device=self.env.device,
        )
        self._steps += 1
        self._return += float(reward[0].item())
        self._sum_velocity += velocity
        self._sum_squared_error += torch.square(velocity - self.command)
        self._upright_steps += int(bool(self._latest["supported_upright"]))

    def trace_metrics(self) -> dict[str, float | bool]:
        return dict(self._latest)

    def finalize(self, *, horizon_steps: int) -> dict[str, float | bool | str]:
        del horizon_steps
        steps = max(self._steps, 1)
        duration = self._steps * float(self.env.step_dt)
        mean_velocity = self._sum_velocity / steps
        rmse = torch.sqrt(self._sum_squared_error / steps)
        upright_fraction = self._upright_steps / steps
        requested_forward = float(self.command[0].item())
        required_progress = max(
            0.0,
            abs(requested_forward) * duration * self.MIN_PROGRESS_FRACTION,
        )
        signed_progress = float(self._latest["forward_displacement_m"])
        if requested_forward < 0.0:
            signed_progress *= -1.0
        progress_pass = (
            True
            if abs(requested_forward) < 1.0e-6
            else signed_progress >= required_progress
        )
        terminal_stable = bool(self._latest["supported_upright"])
        upright_pass = upright_fraction >= self.MIN_UPRIGHT_FRACTION
        return {
            "task_has_binary_success": False,
            "acceptance_id": self.acceptance_id,
            "terminal_success": upright_pass and progress_pass and terminal_stable,
            "upright_fraction_pass": upright_pass,
            "progress_pass": progress_pass,
            "terminal_stability_pass": terminal_stable,
            "upright_fraction": upright_fraction,
            "mean_forward_velocity_mps": float(mean_velocity[0].item()),
            "mean_lateral_velocity_mps": float(mean_velocity[1].item()),
            "mean_yaw_rate_rps": float(mean_velocity[2].item()),
            "forward_velocity_rmse_mps": float(rmse[0].item()),
            "lateral_velocity_rmse_mps": float(rmse[1].item()),
            "yaw_rate_rmse_rps": float(rmse[2].item()),
            "final_forward_displacement_m": float(
                self._latest["forward_displacement_m"]
            ),
            "required_signed_progress_m": required_progress,
            "final_trunk_height_m": float(self._latest["trunk_height_m"]),
            "final_upright_cosine": float(self._latest["upright_cosine"]),
            "total_return": self._return,
            "episode_steps": float(self._steps),
        }


class SkillLifecycleMonitor(EpisodeMonitor):
    """Versioned node acceptance for production-triggered skill policies.

    These predicates use trajectory facts rather than task return. They are
    intentionally explicit because the registered tasks expose shaped rewards,
    not one shared binary success bit, and the production scheduler can run a
    different window from the training task's phase clock.
    """

    MIN_TERMINAL_HEIGHT_M = 0.095
    MIN_TERMINAL_UPRIGHT = 0.80
    MIN_TERMINAL_HOLD_S = 0.40

    def __init__(self, runtime: TaskEnvironment, *, capability_id: str) -> None:
        if capability_id not in {
            "sit-stand-transition",
            "ground-pick",
            "roller-crouch",
            "forward-roll",
        }:
            raise ValueError(f"not a discrete skill capability: {capability_id}")
        self.runtime = runtime
        self.env = runtime.env
        self.capability_id = capability_id
        self._return = 0.0
        self._steps = 0
        self._terminal_hold_steps = 0
        self._initial_height = float(
            environment_state_tensors(self.env)["trunk_height_m"][0].item()
        )
        self._min_height = self._initial_height
        self._max_height = self._initial_height
        self._min_mouth_height = math.inf
        self._max_roll_progress = 0.0
        self._head_latch = False
        self._mouth_site_id: int | None = None
        if capability_id == "ground-pick":
            ids, _names = self.env.scene["robot"].find_sites("mouth_tip")
            if len(ids) != 1:
                raise RuntimeError("GroundPick must expose exactly one mouth_tip site")
            self._mouth_site_id = int(ids[0])
        self._latest = self._sample()

    @property
    def acceptance_id(self) -> str:
        return {
            "sit-stand-transition": (
                "sitstand-v1-seated-height75mm-return-upright-hold400ms"
            ),
            "ground-pick": ("ground-pick-v1-mouth50mm-return-upright-hold400ms"),
            "roller-crouch": (
                "roller-crouch-v1-height-drop25mm-return-upright-hold400ms"
            ),
            "forward-roll": (
                "roulade-v1-supported260deg-head-latch-return-upright-hold400ms"
            ),
        }[self.capability_id]

    def _sample(self) -> dict[str, float | bool]:
        state = environment_state_tensors(self.env)
        height = float(state["trunk_height_m"][0].item())
        upright = float(state["upright_cosine"][0].item())
        stable = (
            height >= self.MIN_TERMINAL_HEIGHT_M
            and upright >= self.MIN_TERMINAL_UPRIGHT
        )
        sample: dict[str, float | bool] = {
            "trunk_height_m": height,
            "upright_cosine": upright,
            "supported_upright": stable,
        }
        if self._mouth_site_id is not None:
            mouth_z = self.env.scene["robot"].data.site_pos_w[0, self._mouth_site_id, 2]
            sample["mouth_height_m"] = float(mouth_z.item())
        if self.capability_id == "forward-roll":
            maximum = getattr(self.env, "_roulade_max", None)
            latch = getattr(self.env, "_roulade_head_latch", None)
            sample["supported_roll_progress_rad"] = (
                0.0 if maximum is None else float(maximum[0].item())
            )
            sample["head_contact_latched"] = (
                False if latch is None else bool(latch[0].item())
            )
        return sample

    def initial(self) -> None:
        self._latest = self._sample()

    def update(
        self,
        *,
        action: torch.Tensor,
        previous_action: torch.Tensor,
        reward: torch.Tensor,
        active: torch.Tensor,
    ) -> None:
        del action, previous_action, active
        self._latest = self._sample()
        self._steps += 1
        self._return += float(reward[0].item())
        height = float(self._latest["trunk_height_m"])
        self._min_height = min(self._min_height, height)
        self._max_height = max(self._max_height, height)
        if "mouth_height_m" in self._latest:
            self._min_mouth_height = min(
                self._min_mouth_height, float(self._latest["mouth_height_m"])
            )
        if "supported_roll_progress_rad" in self._latest:
            self._max_roll_progress = max(
                self._max_roll_progress,
                float(self._latest["supported_roll_progress_rad"]),
            )
            self._head_latch |= bool(self._latest["head_contact_latched"])
        if bool(self._latest["supported_upright"]):
            self._terminal_hold_steps += 1
        else:
            self._terminal_hold_steps = 0

    def trace_metrics(self) -> dict[str, float | bool]:
        return dict(self._latest)

    def finalize(self, *, horizon_steps: int) -> dict[str, float | bool | str]:
        del horizon_steps
        hold_s = self._terminal_hold_steps * float(self.env.step_dt)
        terminal_pass = hold_s >= self.MIN_TERMINAL_HOLD_S
        if self.capability_id == "sit-stand-transition":
            milestone_pass = self._min_height <= 0.075
            milestone = {"seated_height_pass": milestone_pass}
        elif self.capability_id == "ground-pick":
            milestone_pass = self._min_mouth_height <= 0.050
            milestone = {
                "mouth_proximity_pass": milestone_pass,
                "min_mouth_height_m": self._min_mouth_height,
            }
        elif self.capability_id == "roller-crouch":
            height_drop = self._initial_height - self._min_height
            milestone_pass = height_drop >= 0.025
            milestone = {
                "crouch_height_drop_pass": milestone_pass,
                "height_drop_m": height_drop,
            }
        else:
            rotation_pass = self._max_roll_progress >= math.radians(260.0)
            milestone_pass = rotation_pass and self._head_latch
            milestone = {
                "supported_rotation_pass": rotation_pass,
                "head_contact_latch_pass": self._head_latch,
                "max_supported_roll_progress_rad": self._max_roll_progress,
            }
        return {
            "task_has_binary_success": False,
            "acceptance_id": self.acceptance_id,
            "terminal_success": milestone_pass and terminal_pass,
            "milestone_pass": milestone_pass,
            "terminal_stability_pass": terminal_pass,
            "terminal_upright_hold_s": hold_s,
            "initial_trunk_height_m": self._initial_height,
            "min_trunk_height_m": self._min_height,
            "max_trunk_height_m": self._max_height,
            "final_trunk_height_m": float(self._latest["trunk_height_m"]),
            "final_upright_cosine": float(self._latest["upright_cosine"]),
            "total_return": self._return,
            "episode_steps": float(self._steps),
            **milestone,
        }


def make_ball_kick_runtime_body(
    *,
    scenario: RegisteredTaskScenario,
    profile: DeploymentConditionProfile,
    action_scale: float,
    side: str = "right",
    device: str = "cpu",
    record_video: bool = False,
) -> RegisteredRuntimeBody:
    """Prepare BallKick through its registered reset and generic transport."""

    if scenario.task != "Mjlab-BallKick-Flat-MicroDuck":
        raise ValueError(f"BallKick adapter cannot prepare {scenario.task!r}")
    if side not in {"left", "right"}:
        raise ValueError("BallKick side must be left or right")
    runtime = make_environment(
        task=scenario.task,
        num_envs=1,
        device=device,
        seed=scenario.seed,
        matched_candidates=False,
        render_mode="rgb_array" if record_video else None,
        max_actuator_lag_steps=runtime_lag_capacity(profile),
    )
    try:
        observations = reset_registered_scenario(runtime, scenario, profile)
        if side == "left":
            # The repository registers the right-foot task only. Mirror just
            # the ball placement through the task's own reset function; the
            # robot state, DR, terrain and forward objective stay unchanged.
            # This is an explicit initial-state condition, not a second task.
            from mjlab_microduck.tasks import mdp as task_mdp
            from mjlab_microduck.tasks.microduck_ball_kick_env_cfg import (
                BALL_OFFSET_ABS_Y,
                BALL_OFFSET_X,
                BALL_POS_NOISE_XY,
                BALL_RADIUS,
            )

            env_ids = torch.tensor([0], device=runtime.env.device, dtype=torch.long)
            task_mdp.reset_ball_in_front_of_foot(
                runtime.env,
                env_ids,
                offset=(BALL_OFFSET_X, BALL_OFFSET_ABS_Y),
                noise_xy=BALL_POS_NOISE_XY,
                ball_radius=BALL_RADIUS,
                asset_name="ball",
            )
            runtime.env.sim.forward()
            runtime.env.scene.update(dt=0.0)
            runtime.env.sim.sense()
        monitor = BallKickEpisodeMonitor(runtime)
        return RegisteredRuntimeBody(
            PreparedRuntimeTask(
                task=scenario.task,
                scenario_id=scenario.scenario_id,
                reset_label=f"{scenario.reset_label}:{side}-foot-ball",
                seed=scenario.seed,
                profile_name=profile.name,
                profile_sha256=profile.sha256,
                action_scale=action_scale,
                runtime=runtime,
                observations=observations,
                monitor=monitor,
                start_paused=True,
            )
        )
    except Exception:
        runtime.close()
        raise


def make_continuous_locomotion_runtime_body(
    *,
    scenario: RegisteredTaskScenario,
    profile: DeploymentConditionProfile,
    capability_id: str,
    action_scale: float,
    horizon_steps: int,
    device: str = "cpu",
    record_video: bool = False,
) -> RegisteredRuntimeBody:
    """Prepare a registered walking or passive-wheel locomotion task."""

    runtime = make_environment(
        task=scenario.task,
        num_envs=1,
        device=device,
        seed=scenario.seed,
        matched_candidates=False,
        render_mode="rgb_array" if record_video else None,
        max_actuator_lag_steps=runtime_lag_capacity(profile),
    )
    try:
        observations = reset_registered_scenario(runtime, scenario, profile)
        monitor = ContinuousLocomotionMonitor(
            runtime,
            capability_id=capability_id,
            command=scenario.command,
        )
        return RegisteredRuntimeBody(
            PreparedRuntimeTask(
                task=scenario.task,
                scenario_id=scenario.scenario_id,
                reset_label=scenario.reset_label,
                seed=scenario.seed,
                profile_name=profile.name,
                profile_sha256=profile.sha256,
                action_scale=action_scale,
                runtime=runtime,
                observations=observations,
                monitor=monitor,
                start_paused=True,
                horizon_steps=horizon_steps,
            )
        )
    except Exception:
        runtime.close()
        raise


def make_discrete_skill_runtime_body(
    *,
    scenario: RegisteredTaskScenario,
    profile: DeploymentConditionProfile,
    capability_id: str,
    horizon_steps: int,
    device: str = "cpu",
    record_video: bool = False,
) -> RegisteredRuntimeBody:
    """Prepare a registered triggered-skill task behind production ``RobotIo``."""

    expected_tasks = {
        "sit-stand-transition": "Mjlab-SitStand-Flat-MicroDuck",
        "ground-pick": "Mjlab-GroundPick-Flat-MicroDuck",
        "roller-crouch": "Mjlab-RollerCrouch-Flat-MicroDuck",
        "forward-roll": "Mjlab-Roulade-Flat-MicroDuck",
    }
    expected = expected_tasks.get(capability_id)
    if expected is None or scenario.task != expected:
        raise ValueError(
            f"{capability_id!r} adapter requires {expected!r}, got {scenario.task!r}"
        )
    runtime = make_environment(
        task=scenario.task,
        num_envs=1,
        device=device,
        seed=scenario.seed,
        matched_candidates=False,
        render_mode="rgb_array" if record_video else None,
        max_actuator_lag_steps=runtime_lag_capacity(profile),
    )
    try:
        observations = reset_registered_scenario(runtime, scenario, profile)
        monitor = SkillLifecycleMonitor(runtime, capability_id=capability_id)
        return RegisteredRuntimeBody(
            PreparedRuntimeTask(
                task=scenario.task,
                scenario_id=scenario.scenario_id,
                reset_label=scenario.reset_label,
                seed=scenario.seed,
                profile_name=profile.name,
                profile_sha256=profile.sha256,
                # All four registered skill tasks consume unscaled absolute
                # joint offsets; robotd's per-mode skill scale is already in
                # the wire target that this transport inverts.
                action_scale=1.0,
                runtime=runtime,
                observations=observations,
                monitor=monitor,
                start_paused=True,
                horizon_steps=horizon_steps,
            )
        )
    except Exception:
        runtime.close()
        raise
