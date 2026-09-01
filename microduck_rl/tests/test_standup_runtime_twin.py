"""Pure contract tests for the production-runtime StandUp twin."""

from __future__ import annotations

import numpy as np

from mjlab_microduck.sim.body_server import HOME_POSE, MOUTH_INDEX
from mjlab_microduck.sim.runtime_contract import absolute_targets, build_observation
from mjlab_microduck.sim.standup_runtime_twin import (
    POLICY_JOINTS,
    absolute_targets_to_action,
    actor_to_sensor_fixture,
)


def test_task_actor_round_trips_through_raw_sensor_fixture() -> None:
    rng = np.random.default_rng(20260831)
    actor = rng.normal(size=61).astype(np.float32)
    actor[48:61] = 0.0
    fixture = actor_to_sensor_fixture(
        actor,
        np.asarray([1.0, 0.0, 0.0, 0.0]),
        fixture_id="round-trip",
    )
    np.testing.assert_array_equal(build_observation(fixture), actor)
    assert fixture["positions"][MOUTH_INDEX] == HOME_POSE[MOUTH_INDEX]
    assert fixture["velocities"][MOUTH_INDEX] == 0.0


def test_absolute_runtime_targets_invert_to_raw_policy_action() -> None:
    action = np.linspace(-0.7, 0.7, len(POLICY_JOINTS), dtype=np.float32)
    targets = absolute_targets(action)
    np.testing.assert_allclose(absolute_targets_to_action(targets), action, atol=1e-7)
    assert targets[MOUTH_INDEX] == HOME_POSE[MOUTH_INDEX]


def test_sensor_fixture_rejects_wrong_shapes() -> None:
    try:
        actor_to_sensor_fixture(np.zeros(60), np.zeros(4), fixture_id="bad")
    except ValueError as error:
        assert "61D" in str(error)
    else:
        raise AssertionError("60D actor must be rejected")
