from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from mjlab_microduck.autopatch.contracts import (
    DeploymentCondition,
    ObjectiveSpec,
    OptimizerSpec,
    PatchCampaign,
    ReleaseGate,
)
from mjlab_microduck.autopatch.fleet import smoke_production_fleet
from mjlab_microduck.autopatch.registry import PRODUCTION_REGISTRY

RUNTIME_REPO = Path(__file__).resolve().parents[2] / "microduck"


def _campaign() -> PatchCampaign:
    artifact = PRODUCTION_REGISTRY.artifact("alpha-stand")
    return PatchCampaign(
        campaign_id="unit-test",
        artifact_id=artifact.artifact_id,
        artifact_sha256=artifact.expected_sha256,
        capability_id=artifact.capability_id,
        condition=DeploymentCondition(
            "left-leg-authority-25pct",
            "asymmetric-actuator",
            (("left_knee", 0.25), ("left_ankle", 0.25)),
            True,
            "Hidden loss of left knee and ankle authority.",
        ),
        objective=ObjectiveSpec(
            "terminal-standing-v1",
            "standup-terminal-success",
            ("terminal_success", "stable_hold_s", "terminal_progress"),
            ("task_return", "max_height_m"),
            "Require a genuine supported terminal stand.",
        ),
        optimizer=OptimizerSpec(
            "eggroll",
            "output-layer-low-rank",
            16,
            256,
            0.01,
            0.01,
            100,
            20260830,
        ),
        gates=(
            ReleaseGate(
                "target-terminal-success",
                "actual-environment",
                "terminal_success_rate",
                ">=",
                0.8,
                "target",
                2,
            ),
        ),
        calibration_bank_sha256="a" * 64,
        held_out_bank_sha256="b" * 64,
    )


def test_registry_covers_exactly_the_nine_production_artifacts() -> None:
    assert len(PRODUCTION_REGISTRY.artifacts) == 9
    assert {item.filename for item in PRODUCTION_REGISTRY.artifacts} == {
        "alpha_ground_pick.onnx",
        "alpha_sitstand.onnx",
        "alpha_stand.onnx",
        "alpha_walking.onnx",
        "ball_kick_left.onnx",
        "ball_kick_right.onnx",
        "roller.onnx",
        "roller_crouch.onnx",
        "roulade.onnx",
    }


def test_every_production_artifact_passes_strict_graph_and_hash_validation() -> None:
    reports = PRODUCTION_REGISTRY.validate_runtime_artifacts(RUNTIME_REPO)
    assert len(reports) == 9
    assert {(row["input_width"], row["output_width"]) for row in reports} == {(61, 14)}


def test_every_production_artifact_runs_finite_forward_only_inference() -> None:
    result = smoke_production_fleet(
        registry=PRODUCTION_REGISTRY,
        runtime_repo=RUNTIME_REPO,
        rust_probe=None,
        fixture_count=3,
    )
    assert result["status"] == "pass"
    assert len(result["artifacts"]) == 9
    assert all(row["onnx_finite"] for row in result["artifacts"])


def test_training_only_gaps_are_explicit_not_fabricated_artifacts() -> None:
    assert PRODUCTION_REGISTRY.training_tasks_without_production_artifact == (
        "Mjlab-Velocity-Swizzle-MicroDuck",
        "Mjlab-RollerSlope-Flat-MicroDuck",
        "Mjlab-RollerStandUp-Flat-MicroDuck",
        "Mjlab-Spin-Flat-MicroDuck",
    )


def test_runtime_slot_reuse_is_mode_specific() -> None:
    assert PRODUCTION_REGISTRY.artifact("alpha-walking").runtime_slot == "walk"
    assert PRODUCTION_REGISTRY.artifact("roller").runtime_slot == "walk"
    assert (
        PRODUCTION_REGISTRY.artifact("alpha-ground-pick").runtime_slot == "ground_pick"
    )
    assert PRODUCTION_REGISTRY.artifact("roller-crouch").runtime_slot == "ground_pick"
    assert PRODUCTION_REGISTRY.artifact("roller").runtime_modes == ("roller",)
    assert (
        PRODUCTION_REGISTRY.artifact("alpha-walking").updater_component
        != PRODUCTION_REGISTRY.artifact("roller").updater_component
    )
    assert (
        PRODUCTION_REGISTRY.artifact("alpha-ground-pick").updater_component
        != PRODUCTION_REGISTRY.artifact("roller-crouch").updater_component
    )


def test_campaign_identity_is_deterministic_and_bound_to_sealed_artifact() -> None:
    campaign = _campaign()
    assert campaign.sha256 == _campaign().sha256
    PRODUCTION_REGISTRY.validate_campaign(campaign)
    with pytest.raises(ValueError, match="sealed production artifact"):
        PRODUCTION_REGISTRY.validate_campaign(
            replace(campaign, artifact_sha256="0" * 64)
        )


def test_transition_graph_references_only_registered_capabilities_and_artifacts() -> (
    None
):
    capability_ids = {item.capability_id for item in PRODUCTION_REGISTRY.capabilities}
    artifact_ids = {item.artifact_id for item in PRODUCTION_REGISTRY.artifacts}
    assert len(PRODUCTION_REGISTRY.transitions) >= 8
    for transition in PRODUCTION_REGISTRY.transitions:
        assert transition.source_capability in capability_ids
        assert transition.target_capability in capability_ids
        assert set(transition.required_artifact_ids) <= artifact_ids


def test_every_patch_requires_its_capability_node_and_relevant_scheduler_edges() -> (
    None
):
    for artifact in PRODUCTION_REGISTRY.artifacts:
        plan = PRODUCTION_REGISTRY.release_test_plan(artifact.artifact_id)
        assert plan["artifact_sha256"] == artifact.expected_sha256
        assert plan["node"]["capability_id"] == artifact.capability_id
        assert plan["edges"]
        assert all(
            artifact.artifact_id in edge["required_artifact_ids"]
            for edge in plan["edges"]
        )
