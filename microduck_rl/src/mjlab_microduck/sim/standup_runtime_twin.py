"""StandUp adapter for the generic registered-task production runtime."""

from __future__ import annotations

import numpy as np
import torch

from mjlab_microduck.eggroll.deployment import (
    AsymmetricActuatorProfile,
    DeploymentProfile,
    Scenario,
    runtime_lag_capacity,
)
from mjlab_microduck.eggroll.objective import (
    StandupObjectiveConfig,
    TrajectoryAccumulator,
    to_numpy_episode,
)
from mjlab_microduck.eggroll.rollout import (
    StandupEnvironment,
    environment_state_tensors,
    make_environment,
    reset_scenario,
)
from mjlab_microduck.sim.registered_runtime import (
    POLICY_JOINTS,
    EpisodeMonitor,
    PreparedRuntimeTask,
    RegisteredRuntimeBody,
    RuntimeTwinResult,
    absolute_targets_to_action,
    actor_to_sensor_fixture,
    start_server,
)

TASK = "Mjlab-StandUp-Flat-MicroDuck"


class StandupEpisodeMonitor(EpisodeMonitor):
    """Explicit terminal-standing semantics behind the generic transport."""

    def __init__(
        self,
        *,
        runtime: StandupEnvironment,
        objective_config: StandupObjectiveConfig,
    ) -> None:
        self.runtime = runtime
        self.env = runtime.env
        self.accumulator = TrajectoryAccumulator(
            num_envs=1,
            step_dt=float(self.env.step_dt),
            device=torch.device(self.env.device),
            config=objective_config,
        )
        self._latest = environment_state_tensors(self.env)

    def initial(self) -> None:
        self.accumulator.update(
            **self._latest,
            action_rate_l2=torch.zeros(1, device=self.env.device),
            active=torch.ones(1, device=self.env.device, dtype=torch.bool),
            counts_time=False,
        )

    def update(
        self,
        *,
        action: torch.Tensor,
        previous_action: torch.Tensor,
        reward: torch.Tensor,
        active: torch.Tensor,
    ) -> None:
        self._latest = environment_state_tensors(self.env)
        self.accumulator.update(
            **self._latest,
            action_rate_l2=torch.mean(torch.square(action - previous_action), dim=1),
            active=active,
            task_reward=reward,
        )

    def trace_metrics(self) -> dict[str, float | bool]:
        state = self._latest
        return {
            "trunk_height_m": float(state["trunk_height_m"][0].item()),
            "upright_cosine": float(state["upright_cosine"][0].item()),
            "leg_rms_error_rad": float(state["leg_rms_error_rad"][0].item()),
            "foot_support": float(state["foot_support"][0].item()),
        }

    def finalize(self, *, horizon_steps: int) -> dict[str, float | bool]:
        arrays = to_numpy_episode(
            self.accumulator.finalize(horizon_steps=horizon_steps)
        )
        summary: dict[str, float | bool] = {}
        for name, values in arrays.items():
            value = np.asarray(values)[0]
            summary[name] = (
                bool(value) if np.issubdtype(value.dtype, np.bool_) else float(value)
            )
        return summary


class StandupRuntimeBody(RegisteredRuntimeBody):
    """Backward-compatible StandUp construction over the generic runtime body."""

    def __init__(
        self,
        *,
        scenario: Scenario,
        profile: DeploymentProfile | AsymmetricActuatorProfile,
        device: str = "cpu",
        objective_config: StandupObjectiveConfig | None = None,
        record_video: bool = False,
        start_paused: bool = False,
        horizon_steps: int | None = None,
    ) -> None:
        runtime = make_environment(
            task=TASK,
            num_envs=1,
            device=device,
            seed=scenario.seed,
            matched_candidates=False,
            render_mode="rgb_array" if record_video else None,
            max_actuator_lag_steps=runtime_lag_capacity(profile),
        )
        observations = reset_scenario(runtime, scenario, profile)
        monitor = StandupEpisodeMonitor(
            runtime=runtime,
            objective_config=objective_config or StandupObjectiveConfig(),
        )
        super().__init__(
            PreparedRuntimeTask(
                task=TASK,
                scenario_id=scenario.scenario_id,
                reset_label=scenario.pose,
                seed=scenario.seed,
                profile_name=profile.name,
                profile_sha256=profile.sha256,
                action_scale=1.0,
                runtime=runtime,
                observations=observations,
                monitor=monitor,
                start_paused=start_paused,
                horizon_steps=horizon_steps,
            )
        )


__all__ = [
    "POLICY_JOINTS",
    "RuntimeTwinResult",
    "StandupRuntimeBody",
    "absolute_targets_to_action",
    "actor_to_sensor_fixture",
    "start_server",
]
