"""Common-random-number rollouts in the real registered StandUp environment."""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from mjlab.utils.noise.noise_cfg import NoiseCfg, NoiseParam

from .deployment import (
    AsymmetricActuatorProfile,
    DeploymentProfile,
    DeploymentState,
    Scenario,
)
from .objective import (
    StandupObjectiveConfig,
    TrajectoryAccumulator,
    to_numpy_episode,
)

EXPECTED_OBSERVATION_DIM = 61
EXPECTED_ACTION_DIM = 14
POSE_PROBABILITY_KEYS = {
    "face-down": "face_down_prob",
    "face-up": "face_up_prob",
    "sitting": "sitting_prob",
    "standing": "standing_prob",
}
LEG_JOINT_INDICES = (0, 1, 2, 3, 4, 9, 10, 11, 12, 13)
ActionFunction = Callable[[torch.Tensor], torch.Tensor]


@dataclass
class MatchedUniformNoiseCfg(NoiseCfg):
    """Apply the same sensor-noise draw to every EGGROLL candidate."""

    n_min: NoiseParam = -1.0
    n_max: NoiseParam = 1.0

    def __post_init__(self) -> None:
        if (
            isinstance(self.n_min, float)
            and isinstance(self.n_max, float)
            and self.n_min >= self.n_max
        ):
            raise ValueError("n_min must be less than n_max")

    def apply(self, data: torch.Tensor) -> torch.Tensor:
        if data.ndim < 1 or data.shape[0] == 0:
            raise ValueError("matched noise expects a non-empty batch")
        n_min = self._get_cached_tensor("n_min", self.n_min, data.device)
        n_max = self._get_cached_tensor("n_max", self.n_max, data.device)
        sample = torch.rand((1, *data.shape[1:]), dtype=data.dtype, device=data.device)
        noise = (sample * (n_max - n_min) + n_min).expand_as(data)
        if self.operation == "add":
            return data + noise
        if self.operation == "scale":
            return data * noise
        if self.operation == "abs":
            return noise
        raise ValueError(f"Unsupported noise operation: {self.operation}")


@dataclass
class TaskEnvironment:
    """Registered environment plus its restorable deployment baseline."""

    env: Any
    deployment_state: DeploymentState
    render_mode: str | None

    def close(self) -> None:
        self.env.close()


# Compatibility name for the completed StandUp proof. New generic code uses
# TaskEnvironment; this alias avoids rewriting evidence-bound checkpoint paths.
StandupEnvironment = TaskEnvironment


def make_environment(
    *,
    task: str,
    num_envs: int,
    device: str,
    seed: int,
    matched_candidates: bool,
    render_mode: str | None = None,
    max_actuator_lag_steps: int = 6,
) -> TaskEnvironment:
    """Construct the registered task while removing only training curricula.

    Reset events, reward terms, observations, action scaling, physics, domain
    randomization, and termination semantics remain the task's own.  Commands
    are frozen at the deployed zero-command condition and pushes are disabled;
    the experiment isolates one versioned actuator deployment shift.
    """

    if num_envs <= 0:
        raise ValueError("num_envs must be positive")
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.tasks.registry import list_tasks, load_env_cfg

    import mjlab_microduck.tasks  # noqa: F401

    if task not in list_tasks():
        raise ValueError(f"Unknown registered task {task!r}")
    env_cfg = load_env_cfg(task)
    env_cfg.scene.num_envs = num_envs
    env_cfg.seed = seed
    env_cfg.auto_reset = False
    env_cfg.curriculum.clear()
    env_cfg.events.pop("push_robot", None)
    if max_actuator_lag_steps < 6 or max_actuator_lag_steps > 30:
        raise ValueError("max_actuator_lag_steps must be in [6, 30]")
    robot_cfg = env_cfg.scene.entities["robot"]
    for actuator_cfg in robot_cfg.articulation.actuators:
        if hasattr(actuator_cfg, "delay_max_lag"):
            actuator_cfg.delay_max_lag = max_actuator_lag_steps

    for command_cfg in env_cfg.commands.values():
        command_cfg.resampling_time_range = (1.0e9, 1.0e9)

    if matched_candidates:
        for term_cfg in env_cfg.observations["actor"].terms.values():
            noise = term_cfg.noise
            if noise is not None:
                if not hasattr(noise, "n_min") or not hasattr(noise, "n_max"):
                    raise TypeError(
                        "CRN rollouts support the task's uniform actor noise only"
                    )
                term_cfg.noise = MatchedUniformNoiseCfg(
                    n_min=noise.n_min,
                    n_max=noise.n_max,
                    operation=noise.operation,
                )
            if term_cfg.delay_max_lag > 0:
                term_cfg.delay_per_env = False
                term_cfg.delay_per_env_phase = False

    env = ManagerBasedRlEnv(cfg=env_cfg, device=device, render_mode=render_mode)
    if matched_candidates:
        for actuator in env.scene["robot"].actuators:
            delay = getattr(actuator, "_delay_buffer", None)
            if delay is not None:
                delay.per_env = False
                delay.per_env_phase = False
                delay._phase_offsets.zero_()
    return TaskEnvironment(
        env=env,
        deployment_state=DeploymentState.capture(env),
        render_mode=render_mode,
    )


def set_command_block(env: Any, command: Sequence[float] | np.ndarray) -> None:
    """Set the deployed 13D command block without inventing absent terms.

    The shared actor contract is ``twist[3], head[4], body[6]``. Registered
    tasks may deliberately zero-pad head or body rather than owning a command
    term; a non-zero value for such a slot is an error, not a silent no-op.
    Phase-scripted command terms are initialized here but remain responsible
    for their own per-step updates.
    """

    values = np.asarray(command, dtype=np.float32)
    if values.shape != (13,):
        raise ValueError(f"expected a 13D command block, got {values.shape}")
    if not np.isfinite(values).all():
        raise ValueError("command block must be finite")
    active = set(env.command_manager.active_terms)
    if "twist" not in active:
        raise RuntimeError("registered task has no twist command term")
    twist = env.command_manager.get_term("twist")
    if not hasattr(twist, "vel_command_b") or twist.vel_command_b.shape[1] != 3:
        raise RuntimeError("twist term does not expose the deployed 3D command")
    twist.vel_command_b[:] = torch.as_tensor(
        values[:3], device=env.device, dtype=twist.vel_command_b.dtype
    )
    if hasattr(twist, "vel_command_w"):
        twist.vel_command_w[:] = twist.vel_command_b
    for name in (
        "is_heading_env",
        "is_standing_env",
        "is_world_env",
        "is_forward_env",
        "is_turn_in_place_env",
    ):
        value = getattr(twist, name, None)
        if isinstance(value, torch.Tensor):
            value.fill_(False)
    twist.time_left.fill_(1.0e9)
    for name, slot in (("head_pose", values[3:7]), ("body_pose", values[7:13])):
        if name not in active:
            if np.any(slot != 0.0):
                raise ValueError(
                    f"task zero-pads {name}; non-zero deployed values are invalid"
                )
            continue
        term = env.command_manager.get_term(name)
        if not hasattr(term, "_command") or term._command.shape[1] != len(slot):
            raise RuntimeError(
                f"{name} term does not expose the deployed {len(slot)}D command"
            )
        term._command[:] = torch.as_tensor(
            slot, device=env.device, dtype=term._command.dtype
        )
        term.time_left.fill_(1.0e9)


def _set_zero_commands(env: Any) -> None:
    set_command_block(env, np.zeros(13, dtype=np.float32))


def _broadcast_first_environment(value: Any, num_envs: int) -> None:
    tensor = (
        value if isinstance(value, torch.Tensor) else getattr(value, "_tensor", None)
    )
    if (
        isinstance(tensor, torch.Tensor)
        and tensor.ndim > 0
        and tensor.shape[0] == num_envs
    ):
        value[:] = tensor[0:1].expand_as(tensor)


def _synchronize_candidates(env: Any) -> None:
    """Broadcast every exogenous reset draw from candidate zero."""

    from mjlab.managers.event_manager import RecomputeLevel

    num_envs = int(env.num_envs)
    for field in env.event_manager.domain_randomization_fields:
        _broadcast_first_environment(getattr(env.sim.model, field), num_envs)

    qpos = env.sim.data.qpos
    qvel = env.sim.data.qvel
    source = qpos[0].clone()
    origins = env.scene.terrain.env_origins
    relative_root = source[:3] - origins[0]
    qpos[:] = source.unsqueeze(0)
    qpos[:, :3] = origins + relative_root
    qvel[:] = qvel[0:1].expand_as(qvel)

    robot = env.scene["robot"]
    _broadcast_first_environment(robot.data.encoder_bias, num_envs)
    for actuator in robot.actuators:
        for name in (
            "friction_scale",
            "kp_scale",
            "kd_scale",
            "default_kp_scale",
            "default_kd_scale",
            "default_friction_scale",
            "vin_tensor",
            "vin_drop_gain",
            "_prev_motor_torque",
        ):
            value = getattr(actuator, name, None)
            if isinstance(value, torch.Tensor):
                _broadcast_first_environment(value, num_envs)
    imu_quat = getattr(env, "_imu_misalign_quat", None)
    if isinstance(imu_quat, torch.Tensor):
        _broadcast_first_environment(imu_quat, num_envs)

    env.sim.recompute_constants(RecomputeLevel.set_const)
    env.scene.write_data_to_sim()
    for field in ("dof_frictionloss", "dof_damping"):
        _broadcast_first_environment(getattr(env.sim.model, field), num_envs)
    env.sim.forward()
    env.sim.sense()


def reset_scenario(
    runtime: StandupEnvironment,
    scenario: Scenario,
    profile: DeploymentProfile | AsymmetricActuatorProfile,
) -> dict[str, Any]:
    """Reproduce one reset category and seed for all candidate policies."""

    if (
        scenario.profile_name != profile.name
        or scenario.profile_sha256 != profile.sha256
    ):
        raise ValueError("Scenario deployment-profile identity does not match")
    env = runtime.env
    event_cfg = env.event_manager.get_term_cfg("set_ground_state")
    for pose, key in POSE_PROBABILITY_KEYS.items():
        event_cfg.params[key] = float(pose == scenario.pose)
    env.common_step_counter = 0
    env._sim_step_counter = 0
    env.seed(scenario.seed)
    env.reset()
    if env.num_envs > 1:
        _synchronize_candidates(env)
    set_command_block(env, scenario.command)
    runtime.deployment_state.apply(profile)
    from mjlab.managers.event_manager import RecomputeLevel

    env.sim.recompute_constants(RecomputeLevel.set_const)
    env.sim.forward()
    env.scene.update(dt=0.0)
    env.sim.sense()
    env.action_manager.reset()
    env.observation_manager.reset()
    env.observation_manager._obs_buffer = None
    env.obs_buf = env.observation_manager.compute(update_history=True)
    actor = env.obs_buf["actor"]
    if actor.shape != (env.num_envs, EXPECTED_OBSERVATION_DIM):
        raise RuntimeError(f"Expected actor observations [N,61], got {actor.shape}")
    expected_command = torch.as_tensor(
        scenario.command, device=env.device, dtype=actor.dtype
    ).expand(env.num_envs, -1)
    if not torch.allclose(actor[:, 48:61], expected_command):
        raise RuntimeError("Deployment command was not reflected in actor slots")
    return env.obs_buf


def environment_state_tensors(env: Any) -> dict[str, torch.Tensor]:
    """Read only the state needed by the explicit deployment objective."""

    from mjlab_microduck.tasks import mdp as microduck_mdp

    robot = env.scene["robot"]
    origins = env.scene.terrain.env_origins
    trunk_height = robot.data.root_link_pos_w[:, 2] - origins[:, 2]
    quat = robot.data.root_link_quat_w
    qw, qx, qy, qz = quat.unbind(dim=1)
    upright_cosine = 1.0 - 2.0 * (qx * qx + qy * qy)
    joint_pos = microduck_mdp._servo_joint_pos(env, robot)
    default = microduck_mdp._servo_default_joint_pos(env, robot)
    leg_ids = torch.tensor(LEG_JOINT_INDICES, device=env.device, dtype=torch.long)
    leg_error = joint_pos[:, leg_ids] - default[:, leg_ids]
    leg_rms = torch.sqrt(torch.mean(torch.square(leg_error), dim=1))
    found = env.scene.sensors["feet_ground_contact"].data.found
    foot_support = (found.reshape(env.num_envs, -1) > 0).float().mean(dim=1)

    head_error = joint_pos[:, 5:9] - default[:, 5:9]
    head_score = torch.exp(-torch.mean(torch.square(head_error), dim=1) / 0.25)
    sinr_cosp = 2.0 * (qw * qx + qy * qz)
    cosr_cosp = 1.0 - 2.0 * (qx * qx + qy * qy)
    roll = torch.atan2(sinr_cosp, cosr_cosp)
    sinp = torch.clamp(2.0 * (qw * qy - qz * qx), -1.0, 1.0)
    pitch = torch.asin(sinp)
    height_error = trunk_height - 0.115
    angle_error = torch.stack((roll, pitch), dim=1)
    body_score = torch.exp(
        -torch.square(height_error / 0.03)
        - torch.mean(torch.square(angle_error), dim=1) / math.radians(15.0) ** 2
    )
    return {
        "trunk_height_m": trunk_height,
        "upright_cosine": upright_cosine,
        "leg_rms_error_rad": leg_rms,
        "foot_support": foot_support,
        "command_quality": (head_score * body_score).clamp(0.0, 1.0),
        "angular_speed": torch.linalg.vector_norm(
            robot.data.root_link_ang_vel_w, dim=1
        ),
    }


def rollout_episode(
    *,
    runtime: StandupEnvironment,
    scenario: Scenario,
    profile: DeploymentProfile | AsymmetricActuatorProfile,
    action_fn: ActionFunction,
    objective_config: StandupObjectiveConfig,
    max_steps: int | None = None,
    video_path: Path | None = None,
) -> dict[str, np.ndarray]:
    """Run a full real-environment episode and return deployment diagnostics."""

    env = runtime.env
    observations = reset_scenario(runtime, scenario, profile)
    horizon = int(env.max_episode_length if max_steps is None else max_steps)
    accumulator = TrajectoryAccumulator(
        num_envs=env.num_envs,
        step_dt=float(env.step_dt),
        device=torch.device(env.device),
        config=objective_config,
    )
    active = torch.ones(env.num_envs, device=env.device, dtype=torch.bool)
    previous_actions = torch.zeros(
        (env.num_envs, EXPECTED_ACTION_DIM), device=env.device
    )
    accumulator.update(
        **environment_state_tensors(env),
        action_rate_l2=torch.zeros(env.num_envs, device=env.device),
        active=active,
        counts_time=False,
    )
    frames: list[np.ndarray] = []
    if video_path is not None:
        frame = env.render()
        if frame is not None:
            frames.append(np.asarray(frame))

    for _step in range(1, horizon + 1):
        actor = observations["actor"]
        actions = action_fn(actor)
        if actions.shape != (env.num_envs, EXPECTED_ACTION_DIM):
            raise RuntimeError(f"Policy returned actions shaped {actions.shape}")
        if actions.device != torch.device(env.device):
            raise RuntimeError("Policy action device does not match environment")
        if not torch.isfinite(actions).all():
            raise FloatingPointError("Policy produced non-finite actions")
        observations, rewards, terminated, truncated, _ = env.step(actions)
        done = terminated | truncated
        accumulator.update(
            **environment_state_tensors(env),
            action_rate_l2=torch.mean(torch.square(actions - previous_actions), dim=1),
            active=active,
            task_reward=rewards,
        )
        previous_actions = actions
        active &= ~done
        if video_path is not None:
            frame = env.render()
            if frame is not None:
                frames.append(np.asarray(frame))
        if bool(torch.all(done)):
            break
        if bool(torch.any(done)):
            break

    if video_path is not None and frames:
        import mediapy

        video_path.parent.mkdir(parents=True, exist_ok=True)
        mediapy.write_video(
            video_path,
            frames,
            fps=float(env.metadata["render_fps"]),
        )
    return to_numpy_episode(accumulator.finalize(horizon_steps=horizon))


def evaluate_bank(
    *,
    runtime: StandupEnvironment,
    scenarios: Sequence[Scenario],
    profile: DeploymentProfile | AsymmetricActuatorProfile,
    action_fn: ActionFunction,
    objective_config: StandupObjectiveConfig,
    video_dir: Path | None = None,
) -> dict[str, np.ndarray]:
    """Evaluate a base policy one seeded scenario at a time without seed aliasing."""

    if runtime.env.num_envs != 1:
        raise ValueError("Held-out evaluation requires a one-environment runtime")
    if not scenarios:
        raise ValueError("Held-out evaluation bank cannot be empty")
    rows: list[dict[str, np.ndarray]] = []
    for scenario in scenarios:
        video_path = None
        if video_dir is not None:
            video_path = video_dir / f"{scenario.scenario_id}.mp4"
        rows.append(
            rollout_episode(
                runtime=runtime,
                scenario=scenario,
                profile=profile,
                action_fn=action_fn,
                objective_config=objective_config,
                video_path=video_path,
            )
        )
    return {
        name: np.concatenate([row[name] for row in rows], axis=0) for name in rows[0]
    }
