"""Trajectory-level, non-differentiable objectives for EGGROLL StandUp.

The registered mjlab reward remains useful telemetry, but it is deliberately not
the optimization target here.  EGGROLL only needs an ordering over completed
rollouts, so the primary objective is an explicit capability ordering:

1. finish the episode after a continuous genuine standing hold;
2. maximize the continuous standing hold at the end of the episode;
3. maximize smooth terminal standing progress in the worst scenario;
4. only then credit transient recovery and total standing time;
5. improve whole-trajectory progress and recovery time;
6. only then refine post-success command/stability quality.

All functions in this module are simulator-independent and CPU-testable.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch


@dataclass(frozen=True)
class StandupObjectiveConfig:
    """Behavioral success and shaping semantics for StandUp EGGROLL."""

    target_height_m: float = 0.115
    height_tolerance_m: float = 0.015
    max_tilt_deg: float = 20.0
    max_leg_rms_error_rad: float = 0.4
    min_foot_support: float = 0.5
    stable_hold_s: float = 1.0
    progress_window_s: float = 0.5
    height_progress_std_m: float = 0.04
    upright_progress_std: float = 0.40
    leg_progress_std_rad: float = 0.40
    unsupported_progress_floor: float = 0.10

    def __post_init__(self) -> None:
        if self.target_height_m <= 0.0:
            raise ValueError("target_height_m must be positive")
        if self.height_tolerance_m <= 0.0:
            raise ValueError("height_tolerance_m must be positive")
        if not 0.0 < self.max_tilt_deg < 180.0:
            raise ValueError("max_tilt_deg must be between 0 and 180")
        if self.max_leg_rms_error_rad <= 0.0:
            raise ValueError("max_leg_rms_error_rad must be positive")
        if not 0.0 <= self.min_foot_support <= 1.0:
            raise ValueError("min_foot_support must be in [0, 1]")
        if self.stable_hold_s <= 0.0 or self.progress_window_s <= 0.0:
            raise ValueError("hold and progress-window durations must be positive")
        if not 0.0 < self.unsupported_progress_floor <= 1.0:
            raise ValueError("unsupported_progress_floor must be in (0, 1]")


def standing_mask(
    *,
    trunk_height_m: torch.Tensor,
    upright_cosine: torch.Tensor,
    leg_rms_error_rad: torch.Tensor,
    foot_support: torch.Tensor,
    config: StandupObjectiveConfig,
) -> torch.Tensor:
    """Return the task-level standing predicate for every environment."""

    upright_threshold = math.cos(math.radians(config.max_tilt_deg))
    return (
        ((trunk_height_m - config.target_height_m).abs() <= config.height_tolerance_m)
        & (upright_cosine >= upright_threshold)
        & (leg_rms_error_rad <= config.max_leg_rms_error_rad)
        & (foot_support >= config.min_foot_support)
    )


def progress_score(
    *,
    trunk_height_m: torch.Tensor,
    upright_cosine: torch.Tensor,
    leg_rms_error_rad: torch.Tensor,
    foot_support: torch.Tensor,
    config: StandupObjectiveConfig,
) -> torch.Tensor:
    """Bounded multiplicative progress toward a supported stand.

    Multiplication prevents HOME joints while lying on the floor from receiving
    most of the score.  A small unsupported floor keeps a non-zero ordering while
    a policy is still discovering how to place a foot.
    """

    height = torch.exp(
        -torch.square(
            (trunk_height_m - config.target_height_m) / config.height_progress_std_m
        )
    )
    upright_error = torch.clamp(1.0 - upright_cosine, min=0.0)
    upright = torch.exp(
        -upright_error / (config.upright_progress_std * config.upright_progress_std)
    )
    pose = torch.exp(-torch.square(leg_rms_error_rad / config.leg_progress_std_rad))
    support = config.unsupported_progress_floor + (
        1.0 - config.unsupported_progress_floor
    ) * foot_support.clamp(0.0, 1.0)
    return (height * upright * pose * support).clamp(0.0, 1.0)


class TrajectoryAccumulator:
    """Online trajectory summarizer for a vectorized scenario rollout."""

    def __init__(
        self,
        *,
        num_envs: int,
        step_dt: float,
        device: torch.device,
        config: StandupObjectiveConfig,
    ) -> None:
        if num_envs <= 0:
            raise ValueError("num_envs must be positive")
        if step_dt <= 0.0:
            raise ValueError("step_dt must be positive")
        self.num_envs = num_envs
        self.step_dt = step_dt
        self.device = device
        self.config = config
        self.required_hold_steps = max(
            1, math.ceil(config.stable_hold_s / step_dt - 1e-9)
        )
        self.window_steps = max(1, math.ceil(config.progress_window_s / step_dt - 1e-9))

        zeros = torch.zeros(num_envs, device=device, dtype=torch.float32)
        self._initial_progress = zeros.clone()
        self._terminal_progress = zeros.clone()
        self._best_rolling_progress = zeros.clone()
        self._max_trunk_height = torch.full_like(zeros, -torch.inf)
        self._terminal_trunk_height = zeros.clone()
        self._max_upright_cosine = torch.full_like(zeros, -torch.inf)
        self._terminal_upright_cosine = zeros.clone()
        self._current_standing_steps = torch.zeros(
            num_envs, device=device, dtype=torch.int64
        )
        self._max_standing_steps = torch.zeros_like(self._current_standing_steps)
        self._first_stable_step = torch.full(
            (num_envs,), -1, device=device, dtype=torch.int64
        )
        self._standing_steps = torch.zeros_like(self._current_standing_steps)
        self._upright_steps = torch.zeros_like(self._current_standing_steps)
        self._quality_sum = zeros.clone()
        self._quality_steps = torch.zeros_like(self._current_standing_steps)
        self._task_return = zeros.clone()
        self._progress_ring = torch.zeros(
            (self.window_steps, num_envs), device=device, dtype=torch.float32
        )
        self._progress_sum = zeros.clone()
        self._progress_count = 0
        self._ring_index = 0
        self._step = 0
        self._initialized = False

    def update(
        self,
        *,
        trunk_height_m: torch.Tensor,
        upright_cosine: torch.Tensor,
        leg_rms_error_rad: torch.Tensor,
        foot_support: torch.Tensor,
        command_quality: torch.Tensor,
        angular_speed: torch.Tensor,
        action_rate_l2: torch.Tensor,
        active: torch.Tensor,
        task_reward: torch.Tensor | None = None,
        counts_time: bool = True,
    ) -> None:
        """Consume one state sample; the reset sample uses ``counts_time=False``."""

        expected = (self.num_envs,)
        tensors = {
            "trunk_height_m": trunk_height_m,
            "upright_cosine": upright_cosine,
            "leg_rms_error_rad": leg_rms_error_rad,
            "foot_support": foot_support,
            "command_quality": command_quality,
            "angular_speed": angular_speed,
            "action_rate_l2": action_rate_l2,
            "active": active,
        }
        if task_reward is not None:
            tensors["task_reward"] = task_reward
        for name, value in tensors.items():
            if value.shape != expected:
                raise ValueError(f"{name} has shape {value.shape}, expected {expected}")

        progress = progress_score(
            trunk_height_m=trunk_height_m,
            upright_cosine=upright_cosine,
            leg_rms_error_rad=leg_rms_error_rad,
            foot_support=foot_support,
            config=self.config,
        )
        at_goal = (
            standing_mask(
                trunk_height_m=trunk_height_m,
                upright_cosine=upright_cosine,
                leg_rms_error_rad=leg_rms_error_rad,
                foot_support=foot_support,
                config=self.config,
            )
            & active
        )

        if not self._initialized:
            self._initial_progress = progress.clone()
            self._initialized = True

        self._terminal_progress = torch.where(active, progress, self._terminal_progress)
        self._terminal_trunk_height = torch.where(
            active, trunk_height_m, self._terminal_trunk_height
        )
        self._terminal_upright_cosine = torch.where(
            active, upright_cosine, self._terminal_upright_cosine
        )
        self._max_trunk_height = torch.maximum(
            self._max_trunk_height,
            torch.where(active, trunk_height_m, self._max_trunk_height),
        )
        self._max_upright_cosine = torch.maximum(
            self._max_upright_cosine,
            torch.where(active, upright_cosine, self._max_upright_cosine),
        )

        old = self._progress_ring[self._ring_index].clone()
        inserted = torch.where(active, progress, old)
        self._progress_ring[self._ring_index] = inserted
        self._progress_sum += inserted - old
        self._ring_index = (self._ring_index + 1) % self.window_steps
        self._progress_count = min(self.window_steps, self._progress_count + 1)
        rolling = self._progress_sum / float(self._progress_count)
        self._best_rolling_progress = torch.maximum(
            self._best_rolling_progress, rolling
        )

        if not counts_time:
            return

        self._step += 1
        self._current_standing_steps = torch.where(
            at_goal,
            self._current_standing_steps + 1,
            torch.zeros_like(self._current_standing_steps),
        )
        self._max_standing_steps = torch.maximum(
            self._max_standing_steps, self._current_standing_steps
        )
        newly_stable = (self._current_standing_steps >= self.required_hold_steps) & (
            self._first_stable_step < 0
        )
        recovery_start_step = max(1, self._step - self.required_hold_steps + 1)
        self._first_stable_step = torch.where(
            newly_stable,
            torch.full_like(self._first_stable_step, recovery_start_step),
            self._first_stable_step,
        )
        self._standing_steps += at_goal.to(torch.int64)
        upright_threshold = math.cos(math.radians(self.config.max_tilt_deg))
        self._upright_steps += ((upright_cosine >= upright_threshold) & active).to(
            torch.int64
        )

        # Positive, bounded quality.  It cannot compensate for missing success
        # because it is the final lexicographic component and only accumulates
        # while the standing predicate is true.
        stability = torch.exp(-angular_speed.clamp(min=0.0))
        smoothness = torch.exp(-0.05 * action_rate_l2.clamp(min=0.0))
        quality = command_quality.clamp(0.0, 1.0) * stability * smoothness
        after_stable_latch = at_goal & (self._first_stable_step >= 0)
        self._quality_sum += torch.where(
            after_stable_latch, quality, torch.zeros_like(quality)
        )
        self._quality_steps += after_stable_latch.to(torch.int64)
        if task_reward is not None:
            self._task_return += torch.where(
                active, task_reward, torch.zeros_like(task_reward)
            )

    def finalize(self, *, horizon_steps: int) -> dict[str, torch.Tensor]:
        if not self._initialized:
            raise RuntimeError("Cannot finalize an empty trajectory")
        if horizon_steps <= 0:
            raise ValueError("horizon_steps must be positive")
        stable = self._max_standing_steps >= self.required_hold_steps
        terminal = self._current_standing_steps >= self.required_hold_steps
        recovery_step = torch.where(
            self._first_stable_step >= 0,
            self._first_stable_step,
            torch.full_like(self._first_stable_step, horizon_steps + 1),
        )
        quality = self._quality_sum / self._quality_steps.clamp(min=1).to(torch.float32)
        quality = torch.where(stable, quality, torch.zeros_like(quality))
        gain = torch.clamp(
            self._best_rolling_progress - self._initial_progress, min=0.0
        )
        scenario_progress = (
            0.50 * self._best_rolling_progress
            + 0.25 * self._terminal_progress
            + 0.25 * gain
        ).clamp(0.0, 1.0)
        return {
            "stable_success": stable,
            "terminal_success": terminal,
            "stable_hold_s": self._max_standing_steps.to(torch.float32) * self.step_dt,
            "terminal_hold_s": self._current_standing_steps.to(torch.float32)
            * self.step_dt,
            "time_to_recovery_s": recovery_step.to(torch.float32) * self.step_dt,
            "standing_time_s": self._standing_steps.to(torch.float32) * self.step_dt,
            "scenario_progress": scenario_progress,
            "best_rolling_progress": self._best_rolling_progress,
            "terminal_progress": self._terminal_progress,
            "initial_progress": self._initial_progress,
            "max_trunk_height_m": self._max_trunk_height,
            "final_trunk_height_m": self._terminal_trunk_height,
            "max_upright_cosine": self._max_upright_cosine,
            "final_upright_cosine": self._terminal_upright_cosine,
            "time_upright_s": self._upright_steps.to(torch.float32) * self.step_dt,
            "post_success_quality": quality,
            "task_return": self._task_return,
        }


def lexicographic_ranks(keys: Sequence[Sequence[float]]) -> np.ndarray:
    """Map lexicographic tuples to average ranks in ``[0, 1]``.

    The result is ordinal by design.  No arbitrary conversion weights can let a
    large shaping score buy its way past a missing stable recovery.
    """

    if not keys:
        raise ValueError("keys cannot be empty")
    normalized = [tuple(float(value) for value in key) for key in keys]
    width = len(normalized[0])
    if width == 0 or any(len(key) != width for key in normalized):
        raise ValueError("all lexicographic keys must have the same non-zero width")
    if not np.isfinite(np.asarray(normalized, dtype=np.float64)).all():
        raise FloatingPointError("lexicographic keys contain non-finite values")

    order = sorted(range(len(normalized)), key=normalized.__getitem__)
    raw = np.zeros(len(normalized), dtype=np.float32)
    start = 0
    while start < len(order):
        end = start + 1
        while end < len(order) and normalized[order[end]] == normalized[order[start]]:
            end += 1
        average_rank = 0.5 * (start + end - 1)
        for index in order[start:end]:
            raw[index] = average_rank
        start = end
    if len(normalized) == 1:
        return np.ones(1, dtype=np.float32)
    return raw / float(len(normalized) - 1)


def aggregate_candidate_episodes(
    episodes: Sequence[Mapping[str, np.ndarray]],
    *,
    poses: Sequence[str],
) -> tuple[np.ndarray, list[tuple[float, ...]], dict[str, float]]:
    """Aggregate common-scenario episode summaries into EGGROLL fitness."""

    if not episodes or len(episodes) != len(poses):
        raise ValueError("episodes and poses must be non-empty and equally sized")
    population = int(np.asarray(episodes[0]["stable_success"]).shape[0])
    required = {
        "stable_success",
        "terminal_success",
        "stable_hold_s",
        "terminal_hold_s",
        "standing_time_s",
        "time_to_recovery_s",
        "scenario_progress",
        "best_rolling_progress",
        "terminal_progress",
        "post_success_quality",
        "task_return",
    }
    arrays: dict[str, np.ndarray] = {}
    for name in required:
        rows = [np.asarray(episode[name]) for episode in episodes]
        if any(row.shape != (population,) for row in rows):
            raise ValueError(
                f"episode field {name!r} has inconsistent population shape"
            )
        arrays[name] = np.stack(rows, axis=0)
    if not all(np.isfinite(value).all() for value in arrays.values()):
        raise FloatingPointError("episode objective contains non-finite values")

    stable_count = arrays["stable_success"].sum(axis=0)
    terminal_count = arrays["terminal_success"].sum(axis=0)
    hold_total = arrays["stable_hold_s"].sum(axis=0)
    terminal_hold_total = arrays["terminal_hold_s"].sum(axis=0)
    standing_time_total = arrays["standing_time_s"].sum(axis=0)
    progress = arrays["scenario_progress"]
    terminal_progress = arrays["terminal_progress"]
    recovery = arrays["time_to_recovery_s"].mean(axis=0)
    quality = arrays["post_success_quality"].mean(axis=0)
    keys = [
        (
            float(terminal_count[index]),
            float(terminal_hold_total[index]),
            float(terminal_progress[:, index].min()),
            float(terminal_progress[:, index].mean()),
            float(stable_count[index]),
            float(hold_total[index]),
            float(standing_time_total[index]),
            float(progress[:, index].min()),
            float(progress[:, index].mean()),
            -float(recovery[index]),
            float(quality[index]),
        )
        for index in range(population)
    ]
    fitness = lexicographic_ranks(keys)

    metrics: dict[str, float] = {
        "objective/train_success_rate": float(arrays["terminal_success"].mean()),
        "objective/train_terminal_success_rate": float(
            arrays["terminal_success"].mean()
        ),
        "objective/train_transient_success_rate": float(
            arrays["stable_success"].mean()
        ),
        "objective/mean_terminal_success_count": float(terminal_count.mean()),
        "objective/mean_stable_success_count": float(stable_count.mean()),
        "objective/mean_terminal_hold_s": float(arrays["terminal_hold_s"].mean()),
        "objective/mean_terminal_progress": float(terminal_progress.mean()),
        "objective/mean_worst_terminal_progress": float(
            terminal_progress.min(axis=0).mean()
        ),
        "objective/mean_worst_scenario_progress": float(progress.min(axis=0).mean()),
        "objective/mean_scenario_progress": float(progress.mean()),
        "objective/mean_time_to_recovery_s": float(recovery.mean()),
        "objective/mean_post_success_quality": float(quality.mean()),
        "objective/mean_task_return": float(arrays["task_return"].mean()),
        "objective/fitness_unique": float(np.unique(fitness).size),
    }
    for pose in sorted(set(poses)):
        indices = [index for index, value in enumerate(poses) if value == pose]
        metrics[f"objective/pose/{pose}/success_rate"] = float(
            arrays["terminal_success"][indices].mean()
        )
        metrics[f"objective/pose/{pose}/transient_success_rate"] = float(
            arrays["stable_success"][indices].mean()
        )
        metrics[f"objective/pose/{pose}/mean_progress"] = float(
            progress[indices].mean()
        )
        metrics[f"objective/pose/{pose}/mean_terminal_progress"] = float(
            terminal_progress[indices].mean()
        )
    return fitness, keys, metrics


def summarize_heldout_episodes(
    episode: Mapping[str, np.ndarray],
    *,
    poses: Sequence[str],
) -> tuple[tuple[float, ...], dict[str, float], dict[str, float]]:
    """Build checkpoint-selection metrics for one fixed held-out battery."""

    if not poses:
        raise ValueError("poses cannot be empty")
    count = len(poses)
    arrays = {name: np.asarray(value) for name, value in episode.items()}
    if any(value.shape != (count,) for value in arrays.values()):
        bad = {
            name: value.shape
            for name, value in arrays.items()
            if value.shape != (count,)
        }
        raise ValueError(f"held-out episode fields have unexpected shapes: {bad}")

    pose_rates: dict[str, float] = {}
    metrics: dict[str, float] = {}
    for pose in sorted(set(poses)):
        mask = np.asarray([value == pose for value in poses], dtype=bool)
        rate = float(arrays["terminal_success"][mask].mean())
        transient_rate = float(arrays["stable_success"][mask].mean())
        pose_rates[pose] = rate
        metrics[f"eval/objective/pose/{pose}/success_rate"] = rate
        metrics[f"eval/objective/pose/{pose}/terminal_success_rate"] = rate
        metrics[f"eval/objective/pose/{pose}/transient_success_rate"] = transient_rate
        metrics[f"eval/objective/pose/{pose}/mean_progress"] = float(
            arrays["scenario_progress"][mask].mean()
        )

    transient_rate = float(arrays["stable_success"].mean())
    terminal_rate = float(arrays["terminal_success"].mean())
    success_rate = terminal_rate
    min_pose_rate = min(pose_rates.values())
    key = (
        min_pose_rate,
        success_rate,
        float(arrays["terminal_hold_s"].mean()),
        float(arrays["terminal_progress"].min()),
        float(arrays["terminal_progress"].mean()),
        transient_rate,
        float(arrays["stable_hold_s"].mean()),
        float(arrays["scenario_progress"].min()),
        float(arrays["scenario_progress"].mean()),
        -float(arrays["time_to_recovery_s"].mean()),
        float(arrays["post_success_quality"].mean()),
    )
    metrics.update(
        {
            "eval/objective/min_pose_success_rate": min_pose_rate,
            "eval/objective/success_rate": success_rate,
            "eval/objective/terminal_success_rate": terminal_rate,
            "eval/objective/transient_success_rate": transient_rate,
            "eval/objective/mean_terminal_hold_s": float(
                arrays["terminal_hold_s"].mean()
            ),
            "eval/objective/worst_terminal_progress": float(
                arrays["terminal_progress"].min()
            ),
            "eval/objective/mean_terminal_progress": float(
                arrays["terminal_progress"].mean()
            ),
            "eval/objective/mean_hold_s": float(arrays["stable_hold_s"].mean()),
            "eval/objective/worst_progress": float(arrays["scenario_progress"].min()),
            "eval/objective/mean_progress": float(arrays["scenario_progress"].mean()),
            "eval/objective/mean_time_to_recovery_s": float(
                arrays["time_to_recovery_s"].mean()
            ),
            "eval/objective/mean_post_success_quality": float(
                arrays["post_success_quality"].mean()
            ),
            "eval/base_mean_return": float(arrays["task_return"].mean()),
            "eval/base_median_return": float(np.median(arrays["task_return"])),
            "eval/base_std_return": float(arrays["task_return"].std()),
            "eval/base_min_return": float(arrays["task_return"].min()),
            "eval/base_max_return": float(arrays["task_return"].max()),
        }
    )
    return key, metrics, pose_rates


def to_numpy_episode(summary: Mapping[str, torch.Tensor]) -> dict[str, np.ndarray]:
    """Detach a vectorized episode summary without changing its semantics."""

    result: dict[str, np.ndarray] = {}
    for name, value in summary.items():
        array = np.asarray(value.detach().cpu())
        result[name] = array
    return result


def objective_definition(config: StandupObjectiveConfig) -> dict[str, Any]:
    """Serializable description stored in every checkpoint/config artifact."""

    return {
        "name": "posttrain_standup_terminal_success_lexicographic_v1",
        "success": {
            "target_height_m": config.target_height_m,
            "height_tolerance_m": config.height_tolerance_m,
            "max_tilt_deg": config.max_tilt_deg,
            "max_leg_rms_error_rad": config.max_leg_rms_error_rad,
            "min_foot_support": config.min_foot_support,
            "stable_hold_s": config.stable_hold_s,
        },
        "progress": {
            "window_s": config.progress_window_s,
            "formula": (
                "height_gaussian * upright_gaussian * leg_pose_gaussian * "
                "(0.1 + 0.9 * foot_support)"
            ),
            "episode": "0.50 * best_window + 0.25 * terminal + 0.25 * positive_gain",
        },
        "candidate_order": [
            "terminal_success_count",
            "terminal_hold_total",
            "worst_terminal_progress",
            "mean_terminal_progress",
            "transient_recovery_count",
            "stable_hold_total",
            "standing_time_total",
            "worst_scenario_progress",
            "mean_scenario_progress",
            "negative_mean_time_to_recovery",
            "post_success_quality",
        ],
        "selection_gate": "nominal_terminal_success_retention",
        "success_reporting": "terminal_success_only",
        "registered_task_return": "diagnostic_only",
    }
