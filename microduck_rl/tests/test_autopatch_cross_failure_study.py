from __future__ import annotations

import hashlib
import json
from pathlib import Path

from mjlab_microduck.autopatch.contracts import PatchCampaign, ReleaseScope
from mjlab_microduck.autopatch.cross_failure_study import (
    build_trunk_com_study_contracts,
    build_trunk_payload_study_contracts,
    validate_cross_failure_study,
)
from mjlab_microduck.autopatch.efficiency import InteractionCost
from mjlab_microduck.autopatch.foot_proof import (
    make_walking_proof_bank,
    walking_bank_sha256,
)
from mjlab_microduck.autopatch.qualification import QualificationPlan
from mjlab_microduck.autopatch.qualification_command import CommandQualificationSpec
from mjlab_microduck.autopatch.walking_protocol import (
    TRUNK_COM_STUDY_SEEDS,
    TRUNK_PAYLOAD_STUDY_SEEDS,
    walking_campaign_family_sha256,
)
from mjlab_microduck.eggroll.deployment import (
    TRUNK_COM_CALIBRATION_PROFILES,
    TRUNK_PAYLOAD_CALIBRATION_PROFILES,
)

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "docs/experiments/walking_trunk_com_cross_failure_protocol_v2.json"
PAYLOAD_PROTOCOL = (
    ROOT / "docs/experiments/walking_trunk_payload_cross_failure_protocol_v1.json"
)


def test_build_trunk_com_contracts_uses_only_calibrated_predeclared_profile(
    tmp_path: Path,
) -> None:
    profile = TRUNK_COM_CALIBRATION_PROFILES[1]
    calibration = tmp_path / "calibration.json"
    calibration.write_text(
        json.dumps(
            {
                "schema": "eggroll-autopatch-trunk-com-calibration-validation-v1",
                "status": "pass",
                "calibration_status": "condition-frozen",
                "protocol_sha256": hashlib.sha256(PROTOCOL.read_bytes()).hexdigest(),
                "source_policy_sha256": (
                    "e36332d383997d51401897734cd3e79cf5038406feddb18b4d57ecfb141daa6c"
                ),
                "selected_profile_sha256": profile.sha256,
            }
        )
    )

    contracts = build_trunk_com_study_contracts(
        base_campaign_path=(
            ROOT
            / "docs/experiments/campaigns/walking_wedge_autopatch_efficiency_seed4_integrated_v1.json"
        ),
        protocol_path=PROTOCOL,
        calibration_validation_path=calibration,
    )

    campaigns = [PatchCampaign.from_dict(row) for row in contracts["campaigns"]]
    assert [campaign.optimizer.seed for campaign in campaigns] == list(
        TRUNK_COM_STUDY_SEEDS
    )
    assert {campaign.condition.condition_id for campaign in campaigns} == {profile.name}
    assert len({campaign.sha256 for campaign in campaigns}) == 3
    assert (
        len({walking_campaign_family_sha256(campaign) for campaign in campaigns}) == 1
    )
    assert contracts["release_scope"]["profile_sha256s"] == [
        ["shifted", profile.sha256]
    ]
    assert contracts["release_scope"]["unknown_profile_action"] == "retain_source"


def test_build_payload_contracts_uses_only_calibrated_predeclared_profile(
    tmp_path: Path,
) -> None:
    profile = TRUNK_PAYLOAD_CALIBRATION_PROFILES[2]
    calibration = tmp_path / "payload-calibration.json"
    calibration.write_text(
        json.dumps(
            {
                "schema": ("eggroll-autopatch-trunk-payload-calibration-validation-v1"),
                "status": "pass",
                "calibration_status": "condition-frozen",
                "protocol_sha256": hashlib.sha256(
                    PAYLOAD_PROTOCOL.read_bytes()
                ).hexdigest(),
                "source_policy_sha256": (
                    "e36332d383997d51401897734cd3e79cf5038406feddb18b4d57ecfb141daa6c"
                ),
                "selected_profile_sha256": profile.sha256,
            }
        )
    )

    contracts = build_trunk_payload_study_contracts(
        base_campaign_path=(
            ROOT
            / "docs/experiments/campaigns/walking_wedge_autopatch_efficiency_seed4_integrated_v1.json"
        ),
        protocol_path=PAYLOAD_PROTOCOL,
        calibration_validation_path=calibration,
    )

    campaigns = [PatchCampaign.from_dict(row) for row in contracts["campaigns"]]
    assert [campaign.optimizer.seed for campaign in campaigns] == list(
        TRUNK_PAYLOAD_STUDY_SEEDS
    )
    assert {campaign.condition.condition_id for campaign in campaigns} == {profile.name}
    assert {campaign.condition.adapter for campaign in campaigns} == {
        "mjlab-trunk-payload-profile-v1"
    }
    assert contracts["release_scope"]["activation_predicate"] == (
        f"hardware.trunk_payload.profile_sha256 == {profile.sha256}"
    )


def _campaign(seed: int, ordinal: int) -> PatchCampaign:
    document = PatchCampaign.from_json(
        (
            ROOT
            / "docs/experiments/campaigns/walking_wedge_autopatch_efficiency_seed4_integrated_v1.json"
        ).read_text()
    ).canonical_dict()
    profile = TRUNK_COM_CALIBRATION_PROFILES[2]
    document["campaign_id"] = f"alpha-walking-trunk-com-test-seed{ordinal}-v1"
    document["optimizer"]["seed"] = seed
    document["condition"] = {
        "condition_id": profile.name,
        "adapter": "mjlab-trunk-com-shift-profile-v1",
        "parameters": [
            ["profile_name", profile.name],
            ["profile_sha256", profile.sha256],
            ["body", "trunk_base"],
            ["offset_x_m", profile.offset_m[0]],
        ],
        "hidden_from_actor": True,
        "description": "test fixture",
    }
    document["calibration_bank_sha256"] = walking_bank_sha256(
        make_walking_proof_bank(
            base_seed=20293001,
            episodes_per_command=8,
            prefix="calibration-trunk-com",
        )
    )
    document["held_out_bank_sha256"] = walking_bank_sha256(
        make_walking_proof_bank(
            base_seed=20393004,
            episodes_per_command=8,
            prefix="selection-trunk-com",
        )
    )
    return PatchCampaign.from_dict(document)


def test_cross_failure_aggregate_censors_noneligible_seed_above_ceiling(
    tmp_path: Path, monkeypatch
) -> None:
    campaigns = [
        _campaign(seed, ordinal)
        for ordinal, seed in enumerate(TRUNK_COM_STUDY_SEEDS, start=1)
    ]
    profile = TRUNK_COM_CALIBRATION_PROFILES[2]
    scope = ReleaseScope(
        scope_id="trunk-com-test-v1",
        mode="profile_specific",
        profile_sha256s=(("shifted", profile.sha256),),
        required_retention_roles=("shifted",),
        activation_profile_role="shifted",
        activation_predicate="hardware.trunk_com.profile_sha256 == selected",
        source_fallback_sha256=campaigns[0].artifact_sha256,
        unknown_profile_action="retain_source",
    )
    plan = QualificationPlan.from_json(
        (
            ROOT / "docs/experiments/qualification_plans/walking_wedge_release_v1.json"
        ).read_text()
    )
    spec = CommandQualificationSpec.from_json(
        (
            ROOT
            / "docs/experiments/qualification_plans/walking_wedge_release_command_spec_v2.json"
        ).read_text()
    )
    roots = [tmp_path / f"seed-{index}" for index in range(1, 4)]
    for root in roots:
        root.mkdir()
    for root in roots[:2]:
        (root / "integrated_validation.json").write_text("{}")
        manifest = (
            root
            / "run/qualification_evidence/generation-2/production_runtime/manifest.json"
        )
        manifest.parent.mkdir(parents=True)
        manifest.write_text("{}")
        confirmation = (
            root
            / "run/qualification_evidence/generation-2/independent_confirmation/manifest.json"
        )
        confirmation.parent.mkdir(parents=True)
        confirmation.write_text("{}")

    protocol_sha256 = hashlib.sha256(PROTOCOL.read_bytes()).hexdigest()
    calibration = tmp_path / "calibration.json"
    calibration.write_text(
        json.dumps(
            {
                "schema": "eggroll-autopatch-trunk-com-calibration-validation-v1",
                "status": "pass",
                "calibration_status": "condition-frozen",
                "protocol_sha256": protocol_sha256,
                "source_policy_sha256": campaigns[0].artifact_sha256,
                "hf_hardware_flavor": "a10g-large",
                "selected_profile_sha256": profile.sha256,
            }
        )
    )
    source_reference = tmp_path / "source-reference.json"
    source_reference.write_text(
        json.dumps(
            {
                "schema": "eggroll-autopatch-source-behavior-reference-v1",
                "status": "pass",
                "walking_protocol_id": "alpha-walking-trunk-com-cross-failure-v2",
                "source_policy_sha256": campaigns[0].artifact_sha256,
                "activation_profile_sha256": profile.sha256,
                "calibration_validation_sha256": hashlib.sha256(
                    calibration.read_bytes()
                ).hexdigest(),
                "campaign_family_sha256": walking_campaign_family_sha256(campaigns[0]),
                "hf_hardware_flavor": "a10g-large",
            }
        )
    )

    by_root = dict(zip(roots, campaigns, strict=True))

    def fake_contracts(*, root, **_kwargs):
        return (
            by_root[root],
            scope,
            plan,
            spec,
            {"source_commit": "a" * 40},
        )

    eligible_costs = iter((2_000_000, 3_000_000))

    def fake_integrated(**_kwargs):
        cost = next(eligible_costs)
        return {
            "stop_generation": 2,
            "requested_optimization_simulator_steps": cost,
            "failed_qualification_attempts": [],
            "final_candidate_checkpoint_sha256": "b" * 64,
            "output_policy_sha256": "c" * 64,
            "onnx_parity_max_abs_error": 1.0e-6,
            "interaction_cost": {},
        }

    def fake_paired(**_kwargs):
        return (
            {
                "source_behavior_reference_id": "reference:stage",
                "source_successes": 24,
                "adapted_successes": 32,
                "source_success_regressions": 0,
            },
            InteractionCost(
                world_rollouts=64,
                requested_simulator_steps=16_000,
                executed_simulator_steps=15_000,
            ),
        )

    monkeypatch.setattr(
        "mjlab_microduck.autopatch.cross_failure_study._seed_contracts",
        fake_contracts,
    )
    monkeypatch.setattr(
        "mjlab_microduck.autopatch.cross_failure_study.validate_integrated_early_stop_run",
        fake_integrated,
    )
    monkeypatch.setattr(
        "mjlab_microduck.autopatch.cross_failure_study.validate_complete_paired_bank",
        fake_paired,
    )
    monkeypatch.setattr(
        "mjlab_microduck.autopatch.cross_failure_study._validate_noneligible_seed",
        lambda **_kwargs: {
            "status": "complete-noneligible",
            "seed": campaigns[2].optimizer.seed,
            "campaign_id": campaigns[2].campaign_id,
            "campaign_sha256": campaigns[2].sha256,
            "requested_optimization_simulator_steps": 4_617_000,
            "failed_qualification_attempts": [],
            "qualification_cost": InteractionCost().to_dict(),
        },
    )
    # The persisted record must equal the independently recomputed one.
    first = fake_integrated()
    second = fake_integrated()
    (roots[0] / "integrated_validation.json").write_text(json.dumps(first))
    (roots[1] / "integrated_validation.json").write_text(json.dumps(second))
    eligible_costs = iter((2_000_000, 3_000_000))

    result = validate_cross_failure_study(
        protocol_path=PROTOCOL,
        calibration_validation_path=calibration,
        source_behavior_reference_path=source_reference,
        seed_output_dirs=roots,
    )

    assert result["status"] == "pass"
    assert result["release_eligible_seeds"] == 2
    assert (
        result[
            "median_requested_optimization_steps_with_noneligible_censored_above_ceiling"
        ]
        == 3_000_000
    )
    assert result["cross_failure_assessment"] == (
        "bounded-cross-failure-generality-supported"
    )
