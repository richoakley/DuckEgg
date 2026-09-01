from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from mjlab_microduck.autopatch.campaign import build_campaign_plan
from mjlab_microduck.autopatch.contracts import PatchCampaign
from mjlab_microduck.autopatch.foot_proof import (
    make_walking_proof_bank,
    walking_bank_sha256,
)
from mjlab_microduck.autopatch.locomotion_objective import (
    LocomotionObjectiveConfig,
    LocomotionTrajectoryAccumulator,
    aggregate_candidate_episodes,
    summarize_heldout_episodes,
)
from mjlab_microduck.autopatch.locomotion_rollout import (
    ProductionWalkingTransport,
    StartupWorldState,
    rollout_locomotion_episode,
)
from mjlab_microduck.autopatch.locomotion_trainer import (
    _candidate_is_exportable,
    validate_walking_campaign,
)
from mjlab_microduck.autopatch.registry import PRODUCTION_REGISTRY
from mjlab_microduck.eggroll.deployment import NOMINAL_PROFILE
from mjlab_microduck.eggroll.rollout import TaskEnvironment

ROOT = Path(__file__).resolve().parents[1]
CAMPAIGN_PATH = ROOT / "docs/experiments/campaigns/walking_wedge_autopatch_v1.json"


def _summary(
    *,
    success: tuple[bool, ...],
    stable: tuple[bool, ...],
    upright: tuple[float, ...],
    progress: tuple[float, ...],
    rmse: tuple[float, ...],
) -> dict[str, np.ndarray]:
    count = len(success)
    return {
        "terminal_success": np.asarray(success),
        "terminal_stable": np.asarray(stable),
        "upright_fraction": np.asarray(upright, dtype=np.float32),
        "survival_fraction": np.ones(count, dtype=np.float32),
        "signed_progress_m": np.ones(count, dtype=np.float32),
        "required_progress_m": np.ones(count, dtype=np.float32),
        "progress_fraction": np.asarray(progress, dtype=np.float32),
        "mean_forward_velocity_mps": np.zeros(count, dtype=np.float32),
        "mean_lateral_velocity_mps": np.zeros(count, dtype=np.float32),
        "mean_yaw_rate_rps": np.zeros(count, dtype=np.float32),
        "forward_velocity_rmse_mps": np.asarray(rmse, dtype=np.float32),
        "lateral_velocity_rmse_mps": np.zeros(count, dtype=np.float32),
        "yaw_rate_rmse_rps": np.zeros(count, dtype=np.float32),
        "mean_action_rate_l2": np.zeros(count, dtype=np.float32),
        "final_trunk_height_m": np.full(count, 0.11, dtype=np.float32),
        "final_upright_cosine": np.ones(count, dtype=np.float32),
        "episode_steps": np.full(count, 250.0, dtype=np.float32),
        "task_return": np.zeros(count, dtype=np.float32),
    }


def test_locomotion_objective_never_trades_success_for_tracking() -> None:
    episodes = [
        _summary(
            success=(True, False, False, False),
            stable=(True, True, True, True),
            upright=(0.91, 1.0, 1.0, 1.0),
            progress=(1.0, 4.0, 4.0, 4.0),
            rmse=(1.0, 0.0, 0.0, 0.0),
        )
    ]
    fitness, keys, metrics = aggregate_candidate_episodes(episodes)
    assert fitness[0] > max(fitness[1:])
    assert keys[0][0] == 1.0
    assert metrics["objective/terminal_success_rate"] == 0.25


def test_trajectory_success_matches_runtime_locomotion_predicate() -> None:
    accumulator = LocomotionTrajectoryAccumulator(
        num_envs=2,
        step_dt=0.02,
        horizon_steps=100,
        command=(0.4, 0.0, 0.0),
        initial_xy=torch.zeros((2, 2)),
        forward_w=torch.tensor([[1.0, 0.0], [1.0, 0.0]]),
        device=torch.device("cpu"),
        config=LocomotionObjectiveConfig(),
    )
    for step in range(100):
        accumulator.update(
            root_xy=torch.tensor([[0.003 * step, 0.0], [0.003 * step, 0.0]]),
            velocity=torch.tensor([[0.1, 0.0, 0.0], [0.1, 0.0, 0.0]]),
            trunk_height_m=torch.tensor([0.11, 0.08]),
            upright_cosine=torch.tensor([1.0, 1.0]),
            action_rate_l2=torch.zeros(2),
            reward=torch.zeros(2),
            active=torch.ones(2, dtype=torch.bool),
        )
    result = accumulator.finalize()
    assert result["terminal_success"].tolist() == [True, False]
    assert result["terminal_stable"].tolist() == [True, False]


def test_vectorized_rollout_resets_completed_slots_without_reactivating_them(
    monkeypatch,
) -> None:
    class FakeEnv:
        def __init__(self) -> None:
            self.num_envs = 2
            self.device = "cpu"
            self.step_dt = 0.02
            self.step_index = 0
            self.pending = torch.zeros(2, dtype=torch.bool)
            self.reset_calls: list[list[int]] = []
            self.applied_actions: list[torch.Tensor] = []
            self.actor = torch.zeros((2, 61))
            self.robot = SimpleNamespace(
                data=SimpleNamespace(
                    root_link_quat_w=torch.tensor(
                        [[1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]]
                    ),
                    root_link_pos_w=torch.zeros((2, 3)),
                    root_link_lin_vel_b=torch.zeros((2, 3)),
                    root_link_ang_vel_b=torch.zeros((2, 3)),
                )
            )
            self.scene = {"robot": self.robot}

        def step(self, actions):
            if bool(torch.any(self.pending)):
                raise RuntimeError("completed slots must be reset")
            self.applied_actions.append(actions.clone())
            self.step_index += 1
            self.robot.data.root_link_pos_w[:, 0] += 0.01
            terminated = torch.tensor(
                [self.step_index == 1, self.step_index == 3], dtype=torch.bool
            )
            self.pending |= terminated
            return (
                {"actor": self.actor.clone()},
                torch.ones(2),
                terminated,
                torch.zeros(2, dtype=torch.bool),
                {},
            )

        def _reset_idx(self, env_ids):
            ids = env_ids.cpu().tolist()
            self.reset_calls.append(ids)
            self.pending[env_ids] = False

    env = FakeEnv()
    runtime = TaskEnvironment(env=env, deployment_state=None, render_mode=None)
    monkeypatch.setattr(
        "mjlab_microduck.autopatch.locomotion_rollout.reset_registered_scenario",
        lambda *_args, **_kwargs: {"actor": env.actor.clone()},
    )
    monkeypatch.setattr(
        "mjlab_microduck.autopatch.locomotion_rollout.environment_state_tensors",
        lambda _env: {
            "trunk_height_m": torch.full((2,), 0.11),
            "upright_cosine": torch.ones(2),
        },
    )
    case = make_walking_proof_bank(
        base_seed=7, episodes_per_command=1, prefix="mixed-length"
    )[0]
    case = type(case)(
        case_id=case.case_id,
        seed=case.seed,
        command=case.command,
        horizon_steps=3,
    )
    result = rollout_locomotion_episode(
        runtime=runtime,
        case=case,
        profile=NOMINAL_PROFILE,
        action_fn=lambda observations: torch.full((observations.shape[0], 14), 2.0),
        objective_config=LocomotionObjectiveConfig(),
    )

    assert env.reset_calls == [[0]]
    assert result["episode_steps"].tolist() == [1.0, 3.0]
    assert env.applied_actions[1][0].tolist() == [0.0] * 14


def test_production_walking_transport_matches_rust_filter_coefficients() -> None:
    transport = ProductionWalkingTransport(num_envs=1, device=torch.device("cpu"))
    first = transport.apply(torch.ones((1, 14)))
    second = transport.apply(torch.zeros((1, 14)))

    assert first.tolist() == [[1.0] * 14]
    assert torch.allclose(
        second,
        torch.tensor([[0.3] * 5 + [0.5] * 4 + [0.3] * 5]),
    )


def test_production_walking_transport_matches_rust_command_ema() -> None:
    transport = ProductionWalkingTransport(num_envs=1, device=torch.device("cpu"))
    observations = torch.zeros((1, 61))
    command = (0.4, 0.0, 0.0, *(0.0,) * 10)

    first = transport.policy_observations(observations, command)
    second = transport.policy_observations(observations, command)
    third = transport.policy_observations(observations, command)

    assert first[0, 48].item() == pytest.approx(0.08)
    assert second[0, 48].item() == pytest.approx(0.144)
    assert third[0, 48].item() == pytest.approx(0.1952)


def test_evidence_candidate_must_improve_source_and_retain_nominal() -> None:
    assert not _candidate_is_exportable(
        campaign_id="walking-proof",
        nominal_retained=True,
        improves_source=False,
    )
    assert not _candidate_is_exportable(
        campaign_id="walking-proof",
        nominal_retained=False,
        improves_source=True,
    )
    assert _candidate_is_exportable(
        campaign_id="walking-proof",
        nominal_retained=True,
        improves_source=True,
    )
    assert _candidate_is_exportable(
        campaign_id="walking-proof-cuda-smoke",
        nominal_retained=True,
        improves_source=False,
    )


def test_startup_world_state_broadcasts_one_seeded_robot_identity() -> None:
    def fake_env(num_envs: int, fill: float):
        actuator = SimpleNamespace(
            friction_scale=torch.full((num_envs, 2), fill),
            vin_tensor=torch.full((num_envs, 1), fill),
        )
        robot = SimpleNamespace(
            data=SimpleNamespace(
                encoder_bias=torch.full((num_envs, 2), fill),
            ),
            actuators=[actuator],
        )
        return SimpleNamespace(
            num_envs=num_envs,
            event_manager=SimpleNamespace(
                domain_randomization_fields=("body_mass",)
            ),
            sim=SimpleNamespace(
                model=SimpleNamespace(
                    body_mass=torch.full((num_envs, 3), fill),
                )
            ),
            scene={"robot": robot},
            _imu_misalign_quat=torch.full((num_envs, 4), fill),
        )

    source = fake_env(1, 3.0)
    target = fake_env(4, 0.0)
    world = StartupWorldState.capture(source, seed=17)
    world.apply(target)

    assert torch.all(target.sim.model.body_mass == 3.0)
    assert torch.all(target.scene["robot"].data.encoder_bias == 3.0)
    assert torch.all(target.scene["robot"].actuators[0].friction_scale == 3.0)
    assert torch.all(target.scene["robot"].actuators[0].vin_tensor == 3.0)
    assert torch.all(target._imu_misalign_quat == 3.0)


def test_heldout_summary_prioritizes_worst_command() -> None:
    episodes = [
        _summary(
            success=(success,),
            stable=(True,),
            upright=(1.0,),
            progress=(1.0,),
            rmse=(0.1,),
        )
        for success in (True, True, True, False)
    ]
    key, metrics = summarize_heldout_episodes(
        episodes, command_labels=("a", "a", "b", "b")
    )
    assert key[0] == 0.5
    assert metrics["objective/terminal_success_rate"] == 0.75


def test_frozen_walking_campaign_is_eggroll_only_and_hash_bound() -> None:
    campaign = PatchCampaign.from_json(CAMPAIGN_PATH.read_text())
    profile = validate_walking_campaign(
        campaign=campaign, registry=PRODUCTION_REGISTRY
    )
    assert profile.pitch_degrees == 15.0
    bank = make_walking_proof_bank(
        base_seed=20262021,
        episodes_per_command=8,
        prefix="heldout-wedge",
    )
    assert walking_bank_sha256(bank) == campaign.held_out_bank_sha256
    plan = build_campaign_plan(
        campaign=campaign,
        registry=PRODUCTION_REGISTRY,
        runtime_repo=ROOT.parent / "microduck",
    )
    assert plan["execution_protocols"] == ["hyperscalees-eggroll"]
    assert plan["excluded_comparators"] == ["naive-es", "random-search"]
    assert "comparative" in plan["efficiency_claim"]
    json.dumps(plan)
