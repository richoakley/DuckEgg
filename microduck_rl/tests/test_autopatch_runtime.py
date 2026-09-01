from __future__ import annotations

from pathlib import Path

import pytest

from mjlab_microduck.autopatch.registry import PRODUCTION_REGISTRY
from mjlab_microduck.autopatch.runtime import (
    RuntimePolicyBundle,
    capability_events,
)

RUNTIME_REPO = Path(__file__).resolve().parents[2] / "microduck"


def test_walk_bundle_replaces_only_the_selected_slot() -> None:
    replacement = RUNTIME_REPO / "example_policies" / "alpha_stand.onnx"
    bundle = RuntimePolicyBundle.for_artifact(
        registry=PRODUCTION_REGISTRY,
        runtime_repo=RUNTIME_REPO,
        artifact_id="alpha-stand",
        replacement_policy=replacement,
    )
    assert bundle.mode == "walk"
    assert bundle.slot_map["stand"] == replacement.resolve()
    assert bundle.slot_map["walk"].name == "alpha_walking.onnx"
    assert bundle.slot_map["ground_pick"].name == "alpha_ground_pick.onnx"
    rendered = bundle.render_robotd_toml()
    assert 'mode = "walk"' in rendered
    assert "action_scale" not in rendered  # use production mode defaults


def test_roller_bundle_uses_real_slot_reuse_and_no_dead_stand_session() -> None:
    bundle = RuntimePolicyBundle.for_artifact(
        registry=PRODUCTION_REGISTRY,
        runtime_repo=RUNTIME_REPO,
        artifact_id="roller-crouch",
    )
    assert bundle.mode == "roller"
    assert bundle.slot_map["walk"].name == "roller.onnx"
    assert bundle.slot_map["stand"] is None
    assert bundle.slot_map["ground_pick"].name == "roller_crouch.onnx"
    assert bundle.active_action_scale() == 0.8


def test_multi_mode_artifact_requires_an_explicit_runtime_mode() -> None:
    with pytest.raises(ValueError, match="choose --mode"):
        RuntimePolicyBundle.for_artifact(
            registry=PRODUCTION_REGISTRY,
            runtime_repo=RUNTIME_REPO,
            artifact_id="roulade",
        )


def test_continuous_driver_refreshes_the_real_runtime_intent() -> None:
    events = capability_events(
        "legged-locomotion",
        horizon_steps=4,
        command={"vx": 0.2, "vyaw": -0.1},
    )
    assert [event.step for event in events] == [0, 1, 2, 3]
    assert {event.method for event in events} == {"robot.move"}
    assert all(not event.request for event in events)
    assert events[0].params_dict() == {"vx": 0.2, "vy": 0.0, "vyaw": -0.1}


def test_sitstand_driver_exercises_sit_then_rise_handoff() -> None:
    events = capability_events(
        "sit-stand-transition",
        horizon_steps=300,
        trigger_step=0,
        return_step=150,
    )
    assert [event.step for event in events] == [0, 150]
    assert all(event.params_dict() == {"skill": "sit_toggle"} for event in events)
    assert all(event.request for event in events)


@pytest.mark.parametrize(
    ("capability", "skill"),
    (
        ("sit-stand-transition", "sit_toggle"),
        ("ground-pick", "ground_pick"),
        ("roller-crouch", "ground_pick"),
        ("forward-roll", "roulade"),
        ("ball-kick", "kick_right"),
    ),
)
def test_discrete_drivers_use_the_production_robot_do_protocol(
    capability: str, skill: str
) -> None:
    event = capability_events(capability, horizon_steps=20, trigger_step=3)[0]
    assert event.step == 3
    assert event.method == "robot.do"
    assert event.request
    assert event.params_dict() == {"skill": skill}
