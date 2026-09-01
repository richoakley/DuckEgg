"""CPU-only tests for the fixed post-training deployment objective."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from mjlab_microduck.eggroll.objective import (
    StandupObjectiveConfig,
    TrajectoryAccumulator,
    aggregate_candidate_episodes,
    lexicographic_ranks,
    progress_score,
    standing_mask,
    summarize_heldout_episodes,
)


def _vectors(**overrides: float) -> dict[str, torch.Tensor]:
    values = {
        "trunk_height_m": 0.115,
        "upright_cosine": 1.0,
        "leg_rms_error_rad": 0.0,
        "foot_support": 1.0,
        "command_quality": 1.0,
        "angular_speed": 0.0,
        "action_rate_l2": 0.0,
    }
    values.update(overrides)
    return {name: torch.tensor([value]) for name, value in values.items()}


def _episode(
    terminal: list[float],
    *,
    transient: list[float] | None = None,
    terminal_progress: list[float] | None = None,
) -> dict[str, np.ndarray]:
    count = len(terminal)
    stable = terminal if transient is None else transient
    progress = terminal if terminal_progress is None else terminal_progress
    return {
        "stable_success": np.asarray(stable, dtype=np.float32),
        "terminal_success": np.asarray(terminal, dtype=np.float32),
        "stable_hold_s": np.asarray(stable, dtype=np.float32),
        "terminal_hold_s": np.asarray(terminal, dtype=np.float32),
        "standing_time_s": np.asarray(stable, dtype=np.float32),
        "time_to_recovery_s": np.asarray([2.0] * count, dtype=np.float32),
        "scenario_progress": np.asarray(progress, dtype=np.float32),
        "best_rolling_progress": np.asarray(progress, dtype=np.float32),
        "terminal_progress": np.asarray(progress, dtype=np.float32),
        "post_success_quality": np.zeros(count, dtype=np.float32),
        "task_return": np.zeros(count, dtype=np.float32),
    }


def test_standing_requires_height_orientation_pose_and_support() -> None:
    config = StandupObjectiveConfig()
    good = _vectors()
    assert bool(
        standing_mask(
            trunk_height_m=good["trunk_height_m"],
            upright_cosine=good["upright_cosine"],
            leg_rms_error_rad=good["leg_rms_error_rad"],
            foot_support=good["foot_support"],
            config=config,
        )[0]
    )
    for changed in (
        {"trunk_height_m": 0.05},
        {"upright_cosine": 0.0},
        {"leg_rms_error_rad": 0.5},
        {"foot_support": 0.0},
    ):
        sample = _vectors(**changed)
        assert not bool(
            standing_mask(
                trunk_height_m=sample["trunk_height_m"],
                upright_cosine=sample["upright_cosine"],
                leg_rms_error_rad=sample["leg_rms_error_rad"],
                foot_support=sample["foot_support"],
                config=config,
            )[0]
        )


def test_home_pose_on_floor_cannot_look_successful() -> None:
    score = progress_score(
        trunk_height_m=torch.tensor([0.025]),
        upright_cosine=torch.tensor([0.0]),
        leg_rms_error_rad=torch.tensor([0.0]),
        foot_support=torch.tensor([0.0]),
        config=StandupObjectiveConfig(),
    )
    assert float(score[0]) < 1.0e-3


def test_success_requires_continuous_terminal_hold() -> None:
    config = StandupObjectiveConfig(stable_hold_s=1.0)
    accumulator = TrajectoryAccumulator(
        num_envs=1, step_dt=0.02, device=torch.device("cpu"), config=config
    )
    fallen = _vectors(trunk_height_m=0.05, upright_cosine=0.0, foot_support=0.0)
    accumulator.update(**fallen, active=torch.tensor([True]), counts_time=False)
    for _ in range(50):
        accumulator.update(**_vectors(), active=torch.tensor([True]))
    summary = accumulator.finalize(horizon_steps=50)
    assert bool(summary["stable_success"][0])
    assert bool(summary["terminal_success"][0])
    assert float(summary["terminal_hold_s"][0]) == pytest.approx(1.0)


def test_terminal_success_dominates_transient_progress_and_task_return() -> None:
    episode = _episode([0.0, 1.0], transient=[1.0, 1.0])
    episode["task_return"] = np.asarray([1_000.0, -1_000.0], dtype=np.float32)
    fitness, keys, _ = aggregate_candidate_episodes([episode], poses=["standing"])
    assert keys[1][0] > keys[0][0]
    assert fitness[1] > fitness[0]


def test_lexicographic_ranks_are_ordinal() -> None:
    ranks = lexicographic_ranks([(0, 1000), (1, -1000), (1, 0)])
    assert ranks[2] > ranks[1] > ranks[0]


def test_heldout_reports_terminal_and_transient_separately() -> None:
    episode = _episode([0.0, 1.0], transient=[1.0, 1.0])
    _, metrics, pose_rates = summarize_heldout_episodes(
        episode, poses=["standing", "standing"]
    )
    assert pose_rates["standing"] == pytest.approx(0.5)
    assert metrics["eval/objective/success_rate"] == pytest.approx(0.5)
    assert metrics["eval/objective/transient_success_rate"] == pytest.approx(1.0)
