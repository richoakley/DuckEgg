"""Evaluation-only locomotion objective for EGGROLL Autopatch.

The objective deliberately ignores the registered shaped task return when
ordering candidates.  It asks, in order, whether the policy finishes upright,
survives, makes signed progress, and tracks the requested velocity.  The hard
predicates match the production-runtime ``ContinuousLocomotionMonitor``.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

from mjlab_microduck.eggroll.objective import lexicographic_ranks


@dataclass(frozen=True)
class LocomotionObjectiveConfig:
    min_upright_cosine: float = 0.85
    min_trunk_height_m: float = 0.09
    min_upright_fraction: float = 0.90
    min_progress_fraction: float = 0.25

    def __post_init__(self) -> None:
        if not -1.0 <= self.min_upright_cosine <= 1.0:
            raise ValueError("min_upright_cosine must be in [-1, 1]")
        if self.min_trunk_height_m <= 0.0:
            raise ValueError("min_trunk_height_m must be positive")
        if not 0.0 < self.min_upright_fraction <= 1.0:
            raise ValueError("min_upright_fraction must be in (0, 1]")
        if not 0.0 <= self.min_progress_fraction <= 1.0:
            raise ValueError("min_progress_fraction must be in [0, 1]")


class LocomotionTrajectoryAccumulator:
    """Online vectorized summaries for one command and one candidate population."""

    def __init__(
        self,
        *,
        num_envs: int,
        step_dt: float,
        horizon_steps: int,
        command: tuple[float, float, float],
        initial_xy: torch.Tensor,
        forward_w: torch.Tensor,
        device: torch.device,
        config: LocomotionObjectiveConfig,
    ) -> None:
        if num_envs <= 0 or step_dt <= 0.0 or horizon_steps <= 0:
            raise ValueError("locomotion accumulator dimensions must be positive")
        if initial_xy.shape != (num_envs, 2) or forward_w.shape != (num_envs, 2):
            raise ValueError("initial_xy and forward_w must be [N, 2]")
        self.num_envs = num_envs
        self.step_dt = step_dt
        self.horizon_steps = horizon_steps
        self.command = torch.as_tensor(command, device=device, dtype=torch.float32)
        self.initial_xy = initial_xy.clone()
        self.forward_w = forward_w.clone()
        self.device = device
        self.config = config
        zeros = torch.zeros(num_envs, device=device, dtype=torch.float32)
        self._steps = torch.zeros(num_envs, device=device, dtype=torch.int64)
        self._upright_steps = torch.zeros_like(self._steps)
        self._velocity_sum = torch.zeros((num_envs, 3), device=device)
        self._squared_error_sum = torch.zeros_like(self._velocity_sum)
        self._action_rate_sum = zeros.clone()
        self._task_return = zeros.clone()
        self._final_displacement = zeros.clone()
        self._final_height = zeros.clone()
        self._final_upright = zeros.clone()
        self._terminal_stable = torch.zeros(
            num_envs, device=device, dtype=torch.bool
        )

    def update(
        self,
        *,
        root_xy: torch.Tensor,
        velocity: torch.Tensor,
        trunk_height_m: torch.Tensor,
        upright_cosine: torch.Tensor,
        action_rate_l2: torch.Tensor,
        reward: torch.Tensor,
        active: torch.Tensor,
    ) -> None:
        expected_vector = (self.num_envs,)
        if root_xy.shape != (self.num_envs, 2):
            raise ValueError("root_xy must be [N, 2]")
        if velocity.shape != (self.num_envs, 3):
            raise ValueError("velocity must be [N, 3]")
        for name, value in (
            ("trunk_height_m", trunk_height_m),
            ("upright_cosine", upright_cosine),
            ("action_rate_l2", action_rate_l2),
            ("reward", reward),
            ("active", active),
        ):
            if value.shape != expected_vector:
                raise ValueError(f"{name} must be [N]")
        if not all(
            torch.isfinite(value).all()
            for value in (
                root_xy,
                velocity,
                trunk_height_m,
                upright_cosine,
                action_rate_l2,
                reward,
            )
        ):
            raise FloatingPointError("locomotion trajectory contains non-finite values")

        stable = (
            (trunk_height_m >= self.config.min_trunk_height_m)
            & (upright_cosine >= self.config.min_upright_cosine)
            & active
        )
        displacement = torch.sum((root_xy - self.initial_xy) * self.forward_w, dim=1)
        self._steps += active.to(torch.int64)
        self._upright_steps += stable.to(torch.int64)
        self._velocity_sum += torch.where(active[:, None], velocity, 0.0)
        self._squared_error_sum += torch.where(
            active[:, None], torch.square(velocity - self.command), 0.0
        )
        self._action_rate_sum += torch.where(active, action_rate_l2, 0.0)
        self._task_return += torch.where(active, reward, 0.0)
        self._final_displacement = torch.where(
            active, displacement, self._final_displacement
        )
        self._final_height = torch.where(active, trunk_height_m, self._final_height)
        self._final_upright = torch.where(active, upright_cosine, self._final_upright)
        self._terminal_stable = torch.where(
            active, stable, self._terminal_stable
        )

    def finalize(self) -> dict[str, torch.Tensor]:
        steps = self._steps.clamp(min=1)
        steps_float = steps.to(torch.float32)
        duration = steps_float * self.step_dt
        mean_velocity = self._velocity_sum / steps_float[:, None]
        rmse = torch.sqrt(self._squared_error_sum / steps_float[:, None])
        upright_fraction = self._upright_steps.to(torch.float32) / steps_float
        requested_forward = float(self.command[0].item())
        signed_progress = self._final_displacement
        if requested_forward < 0.0:
            signed_progress = -signed_progress
        required_progress = (
            abs(requested_forward) * duration * self.config.min_progress_fraction
        )
        if abs(requested_forward) < 1.0e-6:
            progress_pass = torch.ones_like(self._terminal_stable)
            progress_fraction = torch.ones_like(signed_progress)
        else:
            progress_pass = signed_progress >= required_progress
            progress_fraction = signed_progress / required_progress.clamp(min=1.0e-6)
        upright_pass = upright_fraction >= self.config.min_upright_fraction
        terminal_success = upright_pass & progress_pass & self._terminal_stable
        return {
            "terminal_success": terminal_success,
            "terminal_stable": self._terminal_stable,
            "upright_fraction": upright_fraction,
            "survival_fraction": steps_float / float(self.horizon_steps),
            "signed_progress_m": signed_progress,
            "required_progress_m": required_progress,
            "progress_fraction": progress_fraction.clamp(min=-1.0, max=4.0),
            "mean_forward_velocity_mps": mean_velocity[:, 0],
            "mean_lateral_velocity_mps": mean_velocity[:, 1],
            "mean_yaw_rate_rps": mean_velocity[:, 2],
            "forward_velocity_rmse_mps": rmse[:, 0],
            "lateral_velocity_rmse_mps": rmse[:, 1],
            "yaw_rate_rmse_rps": rmse[:, 2],
            "mean_action_rate_l2": self._action_rate_sum / steps_float,
            "final_trunk_height_m": self._final_height,
            "final_upright_cosine": self._final_upright,
            "episode_steps": steps_float,
            "task_return": self._task_return,
        }


def _stack_episode_fields(
    episodes: Sequence[Mapping[str, np.ndarray]],
) -> dict[str, np.ndarray]:
    if not episodes:
        raise ValueError("locomotion episodes cannot be empty")
    population = int(np.asarray(episodes[0]["terminal_success"]).shape[0])
    fields = set(episodes[0])
    if any(set(episode) != fields for episode in episodes):
        raise ValueError("locomotion episode summaries have inconsistent fields")
    stacked: dict[str, np.ndarray] = {}
    for name in fields:
        rows = [np.asarray(episode[name]) for episode in episodes]
        if any(row.shape != (population,) for row in rows):
            raise ValueError(f"episode field {name!r} must be [population]")
        stacked[name] = np.stack(rows, axis=0)
    if not all(np.isfinite(value).all() for value in stacked.values()):
        raise FloatingPointError("locomotion objective contains non-finite values")
    return stacked


def aggregate_candidate_episodes(
    episodes: Sequence[Mapping[str, np.ndarray]],
) -> tuple[np.ndarray, list[tuple[float, ...]], dict[str, float]]:
    """Convert completed black-box rollouts into ordinal EGGROLL fitness."""

    arrays = _stack_episode_fields(episodes)
    population = arrays["terminal_success"].shape[1]
    success_count = arrays["terminal_success"].sum(axis=0)
    stable_count = arrays["terminal_stable"].sum(axis=0)
    keys = [
        (
            float(success_count[index]),
            float(stable_count[index]),
            float(arrays["upright_fraction"][:, index].min()),
            float(arrays["survival_fraction"][:, index].min()),
            float(arrays["progress_fraction"][:, index].min()),
            float(arrays["upright_fraction"][:, index].mean()),
            float(arrays["progress_fraction"][:, index].mean()),
            -float(arrays["forward_velocity_rmse_mps"][:, index].mean()),
            -float(arrays["mean_action_rate_l2"][:, index].mean()),
        )
        for index in range(population)
    ]
    fitness = lexicographic_ranks(keys)
    return fitness, keys, {
        "objective/terminal_success_rate": float(arrays["terminal_success"].mean()),
        "objective/terminal_stability_rate": float(arrays["terminal_stable"].mean()),
        "objective/mean_upright_fraction": float(arrays["upright_fraction"].mean()),
        "objective/mean_survival_fraction": float(arrays["survival_fraction"].mean()),
        "objective/mean_progress_fraction": float(arrays["progress_fraction"].mean()),
        "objective/mean_forward_velocity_rmse_mps": float(
            arrays["forward_velocity_rmse_mps"].mean()
        ),
        "diagnostic/mean_task_return": float(arrays["task_return"].mean()),
        "objective/fitness_unique": float(np.unique(fitness).size),
    }


def summarize_heldout_episodes(
    episodes: Sequence[Mapping[str, np.ndarray]],
    *,
    command_labels: Sequence[str],
) -> tuple[tuple[float, ...], dict[str, float]]:
    """Create the fixed-bank checkpoint key and release diagnostics."""

    if len(episodes) != len(command_labels) or not episodes:
        raise ValueError("held-out episodes and labels must be non-empty and aligned")
    arrays = _stack_episode_fields(episodes)
    if arrays["terminal_success"].shape[1] != 1:
        raise ValueError("held-out evaluation requires one policy per episode")
    values = {name: value[:, 0] for name, value in arrays.items()}
    command_rates: dict[str, float] = {}
    metrics: dict[str, float] = {}
    for label in sorted(set(command_labels)):
        mask = np.asarray([item == label for item in command_labels], dtype=bool)
        rate = float(values["terminal_success"][mask].mean())
        command_rates[label] = rate
        metrics[f"objective/command/{label}/terminal_success_rate"] = rate
    metrics.update(
        {
            "objective/min_command_success_rate": min(command_rates.values()),
            "objective/terminal_success_rate": float(
                values["terminal_success"].mean()
            ),
            "objective/terminal_stability_rate": float(
                values["terminal_stable"].mean()
            ),
            "objective/mean_upright_fraction": float(
                values["upright_fraction"].mean()
            ),
            "objective/worst_upright_fraction": float(
                values["upright_fraction"].min()
            ),
            "objective/mean_survival_fraction": float(
                values["survival_fraction"].mean()
            ),
            "objective/mean_progress_fraction": float(
                values["progress_fraction"].mean()
            ),
            "objective/worst_progress_fraction": float(
                values["progress_fraction"].min()
            ),
            "objective/negative_mean_forward_velocity_rmse": -float(
                values["forward_velocity_rmse_mps"].mean()
            ),
            "diagnostic/mean_task_return": float(values["task_return"].mean()),
        }
    )
    key = (
        metrics["objective/min_command_success_rate"],
        metrics["objective/terminal_success_rate"],
        metrics["objective/terminal_stability_rate"],
        metrics["objective/worst_upright_fraction"],
        metrics["objective/worst_progress_fraction"],
        metrics["objective/mean_upright_fraction"],
        metrics["objective/mean_progress_fraction"],
        metrics["objective/negative_mean_forward_velocity_rmse"],
    )
    return key, metrics


def to_numpy_episode(summary: Mapping[str, torch.Tensor]) -> dict[str, np.ndarray]:
    return {
        name: np.asarray(value.detach().cpu()) for name, value in summary.items()
    }


def objective_definition(config: LocomotionObjectiveConfig) -> dict[str, Any]:
    return {
        "name": "locomotion-terminal-validity-lexicographic-v1",
        "success": {
            "min_upright_cosine": config.min_upright_cosine,
            "min_trunk_height_m": config.min_trunk_height_m,
            "min_upright_fraction": config.min_upright_fraction,
            "min_progress_fraction": config.min_progress_fraction,
        },
        "candidate_order": [
            "terminal_success_count",
            "terminal_stability_count",
            "worst_upright_fraction",
            "worst_survival_fraction",
            "worst_progress_fraction",
            "mean_upright_fraction",
            "mean_progress_fraction",
            "negative_mean_forward_velocity_rmse",
            "negative_mean_action_rate_l2",
        ],
        "registered_task_return": "diagnostic_only",
        "differentiable": False,
    }
