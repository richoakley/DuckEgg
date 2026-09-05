from __future__ import annotations

from pathlib import Path

import pytest

from mjlab_microduck.autopatch.contracts import PatchCampaign
from mjlab_microduck.autopatch.foot_proof import (
    make_walking_proof_bank,
    walking_bank_sha256,
)
from mjlab_microduck.autopatch.walking_protocol import (
    resolve_walking_protocol,
    walking_campaign_family_sha256,
)
from mjlab_microduck.eggroll.deployment import (
    TRUNK_COM_CALIBRATION_PROFILES,
    TRUNK_PAYLOAD_CALIBRATION_PROFILES,
)

ROOT = Path(__file__).resolve().parents[1]
WEDGE_CAMPAIGN = (
    ROOT / "docs/experiments/campaigns/walking_wedge_autopatch_release_scope_v2.json"
)


def _trunk_com_campaign() -> PatchCampaign:
    document = PatchCampaign.from_json(WEDGE_CAMPAIGN.read_text()).canonical_dict()
    profile = TRUNK_COM_CALIBRATION_PROFILES[2]
    document["campaign_id"] = "alpha-walking-trunk-com-test-v1"
    document["optimizer"]["seed"] = 21_000_001
    document["optimizer"]["generations"] = 9
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


def _trunk_payload_campaign() -> PatchCampaign:
    document = PatchCampaign.from_json(WEDGE_CAMPAIGN.read_text()).canonical_dict()
    profile = TRUNK_PAYLOAD_CALIBRATION_PROFILES[2]
    document["campaign_id"] = "alpha-walking-trunk-payload-test-v1"
    document["optimizer"]["seed"] = 24_000_001
    document["optimizer"]["generations"] = 9
    document["condition"] = {
        "condition_id": profile.name,
        "adapter": "mjlab-trunk-payload-profile-v1",
        "parameters": [
            ["profile_name", profile.name],
            ["profile_sha256", profile.sha256],
            ["body", "trunk_base"],
            ["added_mass_kg", profile.added_mass_kg],
        ],
        "hidden_from_actor": True,
        "description": "test fixture",
    }
    document["calibration_bank_sha256"] = walking_bank_sha256(
        make_walking_proof_bank(
            base_seed=20794001,
            episodes_per_command=8,
            prefix="calibration-trunk-payload",
        )
    )
    document["held_out_bank_sha256"] = walking_bank_sha256(
        make_walking_proof_bank(
            base_seed=20894004,
            episodes_per_command=8,
            prefix="selection-trunk-payload",
        )
    )
    return PatchCampaign.from_dict(document)


def test_trunk_com_protocol_freezes_disjoint_incident_banks() -> None:
    protocol = resolve_walking_protocol(_trunk_com_campaign())
    assert protocol.condition_family == "trunk_com"
    assert protocol.source_behavior_reference_required is True
    assert protocol.source_behavior_match_mode == (
        "bounded_failure_count_and_paired_casewise"
    )
    assert protocol.source_behavior_failure_count_tolerance == 1
    assert protocol.source_behavior_reference_platform == "external"
    assert walking_bank_sha256(protocol.calibration_bank) == (
        "7c894327677809d2f5e819aa5cbf0104fa90af9d3ffd2907082f9b91826d9802"
    )
    assert walking_bank_sha256(protocol.selection_bank) == (
        "8251685f6aa407d982967eabc52b856ef41ea49726eeca520b7771e5c345a0f7"
    )
    assert walking_bank_sha256(protocol.nominal_selection_bank) == (
        "9a7572aa4198eeeecdf1d18df98dc85400b44f275fba9795da83a7dd9067ccdd"
    )
    production = protocol.release_bank("production_runtime")
    confirmation = protocol.release_bank("independent_confirmation")
    assert production.ab_bank_sha256 == (
        "c71c54790a6e26782a5b27449600648230ae02bc1fba60392d2b13ec56017547"
    )
    assert confirmation.ab_bank_sha256 == (
        "c9e82fa04a658e093db8af8bc6775c04044de0cc5c3154e23fa9a3bbd54fcd33"
    )
    assert production.base_seed != confirmation.base_seed


def test_trunk_payload_protocol_freezes_disjoint_incident_banks() -> None:
    protocol = resolve_walking_protocol(_trunk_payload_campaign())
    assert protocol.protocol_id == "alpha-walking-trunk-payload-cross-failure-v1"
    assert protocol.condition_family == "trunk_payload"
    assert protocol.source_behavior_reference_required is True
    assert protocol.source_behavior_match_mode == (
        "bounded_failure_count_and_paired_casewise"
    )
    assert walking_bank_sha256(protocol.calibration_bank) == (
        "0e6863d389a0788cf68e7e6e64162036681123c7e9c4fae6030aad5089fb1316"
    )
    assert walking_bank_sha256(protocol.selection_bank) == (
        "f0f9c6187f67353c551665262830daacb10e871529abca56da6bd8421b25d874"
    )
    assert walking_bank_sha256(protocol.nominal_selection_bank) == (
        "19f8b87843d26766c563927dbb855fc03f2daebfe1c690d6380f16632e932ffe"
    )
    assert protocol.release_bank("production_runtime").ab_bank_sha256 == (
        "f113436614119516b30c87044ee58469bf549a55a6f89f1b822f1cf3684dee1d"
    )
    assert protocol.release_bank("independent_confirmation").ab_bank_sha256 == (
        "cfb35baf3970909d821d3f5efdd64f702ef486149be20d5f5f6dacf6b9f8b3f4"
    )


def test_versioned_wedge_protocol_uses_paired_source_count_matching() -> None:
    historical = PatchCampaign.from_json(WEDGE_CAMPAIGN.read_text())
    assert resolve_walking_protocol(historical).source_behavior_match_mode == (
        "exact_case_ids"
    )
    assert resolve_walking_protocol(historical).source_behavior_reference_platform == (
        "runtime"
    )

    document = historical.canonical_dict()
    document["condition"]["adapter"] = "mjlab-wedge-foot-paired-source-profile-v2"
    paired = resolve_walking_protocol(PatchCampaign.from_dict(document))
    assert paired.protocol_id == "alpha-walking-wedge-foot-paired-source-v2"
    assert paired.source_behavior_reference_required is False
    assert paired.source_behavior_match_mode == ("failure_count_and_paired_casewise")
    assert paired.source_behavior_failure_count_tolerance == 0
    assert paired.source_behavior_reference_platform == "linux"

    document["condition"]["adapter"] = "mjlab-wedge-foot-paired-source-profile-v3"
    bounded = resolve_walking_protocol(PatchCampaign.from_dict(document))
    assert bounded.protocol_id == "alpha-walking-wedge-foot-paired-source-v3"
    assert bounded.source_behavior_match_mode == (
        "bounded_failure_count_and_paired_casewise"
    )
    assert bounded.source_behavior_failure_count_tolerance == 1
    assert bounded.source_behavior_reference_platform == "linux"


def test_trunk_com_protocol_rejects_parameter_or_bank_drift() -> None:
    campaign = _trunk_com_campaign()
    document = campaign.canonical_dict()
    document["condition"]["parameters"][-1][1] = 0.019
    with pytest.raises(ValueError, match="parameters changed"):
        resolve_walking_protocol(PatchCampaign.from_dict(document))

    document = campaign.canonical_dict()
    document["held_out_bank_sha256"] = "f" * 64
    with pytest.raises(ValueError, match="selection bank"):
        resolve_walking_protocol(PatchCampaign.from_dict(document))


def test_trunk_com_campaign_family_allows_only_predeclared_seed_identity() -> None:
    first = _trunk_com_campaign()
    document = first.canonical_dict()
    document["campaign_id"] = "alpha-walking-trunk-com-test-seed2-v1"
    document["optimizer"]["seed"] = 22_000_003
    second = PatchCampaign.from_dict(document)
    resolve_walking_protocol(second)
    assert walking_campaign_family_sha256(first) == walking_campaign_family_sha256(
        second
    )

    document["optimizer"]["learning_rate"] = 0.004
    changed = PatchCampaign.from_dict(document)
    with pytest.raises(ValueError, match="predeclared optimizer"):
        resolve_walking_protocol(changed)
    assert walking_campaign_family_sha256(first) != walking_campaign_family_sha256(
        changed
    )
