"""Pure tests for versioned deployment profiles and scenario banks."""

from __future__ import annotations

from types import SimpleNamespace

import torch

from mjlab_microduck.eggroll.deployment import (
    ASYMMETRIC_CALIBRATION_PROFILES,
    ASYMMETRIC_SELECTED_BANK_SHA256,
    ASYMMETRIC_SELECTED_PROFILE,
    ASYMMETRIC_SELECTED_PROFILE_SHA256,
    CALIBRATION_LAG20,
    NOMINAL_PROFILE,
    PRIORITY_FOOT_MATERIAL_CALIBRATION_PROFILES,
    REPLACEMENT_FOOT_CALIBRATION_PROFILES,
    REPLACEMENT_SOLE_CALIBRATION_PROFILES,
    WEDGE_FOOT_CALIBRATION_PROFILES,
    DeploymentState,
    bank_sha256,
    frozen_asymmetric_calibration_bank,
    make_balanced_bank,
    runtime_lag_capacity,
    select_calibrated_profile,
)


class FakeDelay:
    def __init__(self) -> None:
        self.min_lag = 1
        self.max_lag = 6
        self.batch_size = 3
        self.per_env = True
        self.hold_prob = 0.2
        self.update_period = 2
        self.per_env_phase = True
        self._buffer = SimpleNamespace(max_length=7)
        self.current_lags = torch.tensor([1, 2, 3])

    def set_lags(self, lags: torch.Tensor) -> None:
        self.current_lags = lags.clone()


def fake_environment():
    actuator = SimpleNamespace(
        vin_tensor=torch.full((3,), 7.1),
        vin_drop_gain=torch.full((3,), 0.12),
        _delay_buffer=FakeDelay(),
        effort_scale=torch.ones((3, 4)),
        _target_names=("left_hip_pitch", "left_knee", "left_ankle", "right_knee"),
    )
    robot = SimpleNamespace(
        actuators=[actuator],
        indexing=SimpleNamespace(geom_ids=torch.tensor([0, 1])),
        find_geoms=lambda _pattern: (
            [0, 1],
            ["left_foot_collision", "right_foot_collision"],
        ),
    )
    model = SimpleNamespace(
        geom_dataid=torch.tensor([[0, 1]]),
        mesh_vertadr=torch.tensor([0, 2]),
        mesh_vertnum=torch.tensor([2, 2]),
        mesh_vert=torch.tensor(
            [
                [1.0, 2.0, 3.0],
                [-1.0, -2.0, -3.0],
                [4.0, 5.0, 6.0],
                [-4.0, -5.0, -6.0],
            ]
        ),
        geom_aabb=torch.ones((3, 2, 2, 3)),
        geom_rbound=torch.full((3, 2), 2.0),
        geom_friction=torch.full((3, 2, 3), 0.8),
        geom_priority=torch.zeros(2, dtype=torch.int32),
        geom_quat=torch.tensor(
            [[[1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]]] * 3
        ),
    )
    return (
        SimpleNamespace(scene={"robot": robot}, sim=SimpleNamespace(model=model)),
        actuator,
    )


def test_profile_application_is_fixed_and_non_accumulating() -> None:
    env, actuator = fake_environment()
    state = DeploymentState.capture(env)
    actuator._delay_buffer._buffer.max_length = 21
    state.apply(CALIBRATION_LAG20)
    assert torch.all(actuator.vin_tensor == 6.2)
    assert torch.all(actuator.vin_drop_gain == 0.25)
    assert torch.all(actuator._delay_buffer.current_lags == 20)
    state.apply(NOMINAL_PROFILE)
    assert torch.all(actuator.vin_tensor == 7.35)
    assert torch.all(actuator.vin_drop_gain == 0.10)
    assert torch.all(actuator._delay_buffer.current_lags == 4)
    state.restore()
    assert torch.all(actuator.vin_tensor == 7.1)
    assert torch.all(actuator._delay_buffer.current_lags == torch.tensor([1, 2, 3]))


def test_asymmetric_profile_changes_only_named_joints_and_restores() -> None:
    env, actuator = fake_environment()
    state = DeploymentState.capture(env)
    profile = ASYMMETRIC_CALIBRATION_PROFILES[1]
    state.apply(profile)
    assert torch.all(actuator.effort_scale[:, 0] == 1.0)
    assert torch.all(actuator.effort_scale[:, 1] == 0.35)
    assert torch.all(actuator.effort_scale[:, 2] == 0.35)
    assert torch.all(actuator.effort_scale[:, 3] == 1.0)
    state.apply(NOMINAL_PROFILE)
    assert torch.all(actuator.effort_scale == 1.0)
    state.restore()
    assert torch.all(actuator.effort_scale == 1.0)


def test_asymmetric_profile_identity_includes_joint_names_and_effectiveness() -> None:
    first, second = ASYMMETRIC_CALIBRATION_PROFILES[:2]
    assert first.sha256 != second.sha256
    assert first.canonical_dict()["joint_effectiveness"] == {
        "left_knee": 0.45,
        "left_ankle": 0.45,
    }
    assert runtime_lag_capacity(first) == 6


def test_replacement_sole_profile_scales_only_foot_geometry_and_restores() -> None:
    env, actuator = fake_environment()
    del actuator
    model = env.sim.model
    baseline_vertices = model.mesh_vert.clone()
    baseline_aabb = model.geom_aabb.clone()
    baseline_rbound = model.geom_rbound.clone()
    state = DeploymentState.capture(env)
    profile = REPLACEMENT_SOLE_CALIBRATION_PROFILES[-1]
    state.apply(profile)
    assert profile.uniform_scale == 1.45
    assert torch.allclose(model.mesh_vert, baseline_vertices * 1.45)
    assert torch.allclose(model.geom_aabb, baseline_aabb * 1.45)
    assert torch.allclose(model.geom_rbound, baseline_rbound * 1.45)
    state.apply(NOMINAL_PROFILE)
    assert torch.equal(model.mesh_vert, baseline_vertices)
    assert torch.equal(model.geom_aabb, baseline_aabb)
    assert torch.equal(model.geom_rbound, baseline_rbound)


def test_replacement_sole_profile_identity_is_geometry_bound() -> None:
    smaller, larger = REPLACEMENT_SOLE_CALIBRATION_PROFILES[1:3]
    assert smaller.sha256 != larger.sha256
    assert smaller.canonical_dict()["geometry_scope"] == {
        "geoms": ["left_foot_collision", "right_foot_collision"],
        "mass_inertia_changed": False,
    }
    assert runtime_lag_capacity(larger) == 6


def test_replacement_foot_changes_geometry_and_material_nonaccumulatively() -> None:
    env, actuator = fake_environment()
    del actuator
    model = env.sim.model
    vertices = model.mesh_vert.clone()
    friction = model.geom_friction.clone()
    state = DeploymentState.capture(env)
    profile = REPLACEMENT_FOOT_CALIBRATION_PROFILES[-1]
    state.apply(profile)
    assert profile.uniform_scale == 1.30
    assert profile.friction_scale == 0.25
    assert torch.allclose(model.mesh_vert, vertices * 1.30)
    assert torch.allclose(model.geom_friction, friction * 0.25)
    state.apply(profile)
    assert torch.allclose(model.mesh_vert, vertices * 1.30)
    assert torch.allclose(model.geom_friction, friction * 0.25)
    state.restore()
    assert torch.equal(model.mesh_vert, vertices)
    assert torch.equal(model.geom_friction, friction)


def test_priority_foot_material_owns_pair_friction_and_restores() -> None:
    env, actuator = fake_environment()
    del actuator
    model = env.sim.model
    friction = model.geom_friction.clone()
    priority = model.geom_priority.clone()
    state = DeploymentState.capture(env)
    profile = PRIORITY_FOOT_MATERIAL_CALIBRATION_PROFILES[-1]
    state.apply(profile)
    expected = torch.tensor([0.08, 0.0004, 0.000008])
    assert torch.allclose(model.geom_friction, expected.expand_as(model.geom_friction))
    assert torch.equal(model.geom_priority, torch.ones_like(model.geom_priority))
    state.restore()
    assert torch.equal(model.geom_friction, friction)
    assert torch.equal(model.geom_priority, priority)


def test_wedge_foot_rotates_only_contact_frames_and_restores() -> None:
    env, actuator = fake_environment()
    del actuator
    model = env.sim.model
    baseline = model.geom_quat.clone()
    state = DeploymentState.capture(env)
    profile = WEDGE_FOOT_CALIBRATION_PROFILES[1]
    state.apply(profile)
    half_angle = torch.deg2rad(torch.tensor(5.0))
    expected = torch.tensor(
        [torch.cos(half_angle), 0.0, torch.sin(half_angle), 0.0]
    )
    assert torch.allclose(model.geom_quat, expected.expand_as(model.geom_quat))
    state.apply(profile)
    assert torch.allclose(model.geom_quat, expected.expand_as(model.geom_quat))
    state.restore()
    assert torch.equal(model.geom_quat, baseline)


def test_runtime_lag_capacity_preserves_subminimum_fixed_lag() -> None:
    assert NOMINAL_PROFILE.actuator_lag_steps == 4
    assert runtime_lag_capacity(NOMINAL_PROFILE) == 6
    assert runtime_lag_capacity(CALIBRATION_LAG20) == 20


def test_balanced_banks_are_deterministic_and_profile_bound() -> None:
    first = make_balanced_bank(
        profile=CALIBRATION_LAG20,
        base_seed=42,
        episodes_per_pose=3,
        prefix="heldout",
    )
    second = make_balanced_bank(
        profile=CALIBRATION_LAG20,
        base_seed=42,
        episodes_per_pose=3,
        prefix="heldout",
    )
    assert first == second
    assert bank_sha256(first) == bank_sha256(second)
    counts = {
        pose: sum(row.pose == pose for row in first)
        for pose in {row.pose for row in first}
    }
    assert set(counts.values()) == {3}
    assert all(row.profile_sha256 == CALIBRATION_LAG20.sha256 for row in first)


def test_frozen_asymmetric_profile_and_bank_do_not_drift() -> None:
    bank = frozen_asymmetric_calibration_bank()
    assert ASYMMETRIC_SELECTED_PROFILE.name == ("asymmetric-left-knee-ankle-25pct-v1")
    assert ASYMMETRIC_SELECTED_PROFILE.sha256 == ASYMMETRIC_SELECTED_PROFILE_SHA256
    assert len(bank) == 32
    assert bank_sha256(bank) == ASYMMETRIC_SELECTED_BANK_SHA256
    assert {scenario.pose for scenario in bank} == {
        "standing",
        "sitting",
        "face-down",
        "face-up",
    }
    assert all(
        sum(scenario.pose == pose for scenario in bank) == 8
        for pose in {scenario.pose for scenario in bank}
    )


def test_calibration_selects_hardest_noncatastrophic_per_pose_shift() -> None:
    def row(rate: float, minimum: float) -> dict:
        return {
            "metrics": {"eval/objective/success_rate": rate},
            "pose_terminal_success_rates": {
                "standing": rate,
                "sitting": rate,
                "face-down": rate,
                "face-up": minimum,
            },
        }

    nominal = row(1.0, 1.0)
    mild = row(0.75, 0.5)
    hard = row(0.4, 0.125)
    catastrophic = row(0.0, 0.0)
    assert select_calibrated_profile([nominal, mild, hard, catastrophic]) is hard
    assert select_calibrated_profile([nominal, catastrophic]) is None
