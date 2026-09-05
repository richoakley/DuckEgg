"""Frozen bank and deployment contracts for walking Autopatch incidents."""

from __future__ import annotations

import copy
import hashlib
import json
from dataclasses import dataclass
from typing import Any, Literal

from mjlab_microduck.eggroll.deployment import (
    PROFILES,
    DeploymentConditionProfile,
    TrunkComShiftProfile,
    TrunkPayloadProfile,
    WedgeFootProfile,
)

from .contracts import PatchCampaign
from .foot_proof import (
    WalkingCalibrationCase,
    make_replacement_foot_calibration_bank,
    make_walking_proof_bank,
    walking_bank_sha256,
)

TRUNK_COM_STUDY_SEEDS = (21_000_001, 22_000_003, 23_000_009)
TRUNK_PAYLOAD_STUDY_SEEDS = (24_000_001, 25_000_003, 26_000_009)


@dataclass(frozen=True)
class ReleaseBankContract:
    """One full-episode production-runtime bank and its execution identity."""

    stage: Literal["production_runtime", "independent_confirmation"]
    base_seed: int
    prefix: str
    ab_bank_sha256: str
    device: str


@dataclass(frozen=True)
class WalkingCampaignProtocol:
    """All condition-specific inputs that may differ across walking incidents."""

    protocol_id: str
    condition_family: Literal["wedge_foot", "trunk_com", "trunk_payload"]
    profile: DeploymentConditionProfile
    calibration_bank: tuple[WalkingCalibrationCase, ...]
    selection_bank: tuple[WalkingCalibrationCase, ...]
    nominal_selection_bank: tuple[WalkingCalibrationCase, ...]
    release_banks: tuple[ReleaseBankContract, ReleaseBankContract]
    source_behavior_reference_required: bool
    source_behavior_match_mode: Literal[
        "exact_case_ids",
        "failure_count_and_paired_casewise",
        "bounded_failure_count_and_paired_casewise",
    ]
    source_behavior_failure_count_tolerance: int
    source_behavior_reference_platform: Literal["runtime", "linux", "external"]

    def release_bank(self, stage: str) -> ReleaseBankContract:
        matches = [bank for bank in self.release_banks if bank.stage == stage]
        if len(matches) != 1:
            raise ValueError(f"unknown or duplicate walking release stage {stage!r}")
        return matches[0]


def walking_campaign_family_sha256(campaign: PatchCampaign) -> str:
    """Hash the shared study contract while allowing only ID and seed to vary."""

    document = copy.deepcopy(campaign.canonical_dict())
    document["campaign_id"] = "<study-seed>"
    optimizer = document.get("optimizer")
    if not isinstance(optimizer, dict):
        raise TypeError("campaign optimizer must be an object")
    optimizer["seed"] = 0
    return hashlib.sha256(
        json.dumps(document, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _bound_profile(campaign: PatchCampaign) -> tuple[str, str, dict[str, Any]]:
    if not campaign.condition.hidden_from_actor:
        raise ValueError(
            "walking hardware conditions must remain hidden from the actor"
        )
    parameters = dict(campaign.condition.parameters)
    name = parameters.get("profile_name")
    expected_sha256 = parameters.get("profile_sha256")
    if not isinstance(name, str) or not isinstance(expected_sha256, str):
        raise TypeError("walking condition must bind profile_name and profile_sha256")
    profile = PROFILES.get(name)
    if profile is None or profile.sha256 != expected_sha256:
        raise ValueError("walking condition profile hash does not match its semantics")
    return name, expected_sha256, parameters


def resolve_walking_protocol(campaign: PatchCampaign) -> WalkingCampaignProtocol:
    """Resolve and verify the frozen banks for a supported walking incident."""

    name, expected_sha256, parameters = _bound_profile(campaign)
    profile = PROFILES[name]
    adapter = campaign.condition.adapter
    if adapter in {
        "mjlab-wedge-foot-profile-v1",
        "mjlab-wedge-foot-paired-source-profile-v2",
        "mjlab-wedge-foot-paired-source-profile-v3",
    }:
        if not isinstance(profile, WedgeFootProfile):
            raise TypeError("campaign profile is not a registered wedge-foot condition")
        if parameters != {
            "profile_name": name,
            "profile_sha256": expected_sha256,
            "pitch_degrees": profile.pitch_degrees,
        }:
            raise ValueError("wedge-foot campaign parameters changed")
        calibration = make_replacement_foot_calibration_bank()
        selection = make_walking_proof_bank(
            base_seed=20262021,
            episodes_per_command=8,
            prefix="heldout-wedge",
        )
        nominal_selection = make_walking_proof_bank(
            base_seed=20262022,
            episodes_per_command=8,
            prefix="heldout-nominal",
        )
        release_banks = (
            ReleaseBankContract(
                stage="production_runtime",
                base_seed=20262021,
                prefix="heldout-wedge",
                ab_bank_sha256=(
                    "ba760ae8dcbb6c0b5827ab8c38bcbe6c4f4a5b41bc85864c0447af24f55eff01"
                ),
                device="cpu",
            ),
            ReleaseBankContract(
                stage="independent_confirmation",
                base_seed=20262023,
                prefix="confirmation-wedge",
                ab_bank_sha256=(
                    "106a0c05307852fc6c0b05c383ab658ce2c54fef7d161105cdf4ca97c983d307"
                ),
                device="cuda:0",
            ),
        )
        paired_source = adapter != "mjlab-wedge-foot-profile-v1"
        bounded_source_count = adapter == ("mjlab-wedge-foot-paired-source-profile-v3")
        protocol = WalkingCampaignProtocol(
            protocol_id=(
                "alpha-walking-wedge-foot-paired-source-v3"
                if bounded_source_count
                else "alpha-walking-wedge-foot-paired-source-v2"
                if paired_source
                else "alpha-walking-wedge-foot-v1"
            ),
            condition_family="wedge_foot",
            profile=profile,
            calibration_bank=calibration,
            selection_bank=selection,
            nominal_selection_bank=nominal_selection,
            release_banks=release_banks,
            source_behavior_reference_required=False,
            source_behavior_match_mode=(
                "bounded_failure_count_and_paired_casewise"
                if bounded_source_count
                else "failure_count_and_paired_casewise"
                if paired_source
                else "exact_case_ids"
            ),
            source_behavior_failure_count_tolerance=(1 if bounded_source_count else 0),
            source_behavior_reference_platform=(
                "linux" if paired_source else "runtime"
            ),
        )
    elif adapter == "mjlab-trunk-com-shift-profile-v1":
        if not isinstance(profile, TrunkComShiftProfile):
            raise TypeError("campaign profile is not a registered trunk-CoM condition")
        if campaign.optimizer.worlds_per_candidate % 4:
            raise ValueError(
                "trunk-CoM worlds_per_candidate must balance four commands"
            )
        optimizer = campaign.optimizer
        if (
            optimizer.algorithm != "hyperscalees-eggroll"
            or optimizer.population != 512
            or optimizer.generations != 9
            or optimizer.rank != 4
            or optimizer.worlds_per_candidate != 4
            or optimizer.sigma != 0.015
            or optimizer.learning_rate != 0.003
            or optimizer.seed not in TRUNK_COM_STUDY_SEEDS
            or optimizer.trainable_scope != "final-affine-weight-and-bias"
        ):
            raise ValueError("trunk-CoM campaign changed the predeclared optimizer")
        if parameters != {
            "profile_name": name,
            "profile_sha256": expected_sha256,
            "body": "trunk_base",
            "offset_x_m": profile.offset_m[0],
        }:
            raise ValueError("trunk-CoM campaign parameters changed")
        calibration = make_walking_proof_bank(
            base_seed=20293001,
            episodes_per_command=8,
            prefix="calibration-trunk-com",
        )
        selection = make_walking_proof_bank(
            base_seed=20393004,
            episodes_per_command=8,
            prefix="selection-trunk-com",
        )
        nominal_selection = make_walking_proof_bank(
            base_seed=20493007,
            episodes_per_command=8,
            prefix="selection-nominal-com",
        )
        release_banks = (
            ReleaseBankContract(
                stage="production_runtime",
                base_seed=20593010,
                prefix="production-trunk-com",
                ab_bank_sha256=(
                    "c71c54790a6e26782a5b27449600648230ae02bc1fba60392d2b13ec56017547"
                ),
                device="cuda:0",
            ),
            ReleaseBankContract(
                stage="independent_confirmation",
                base_seed=20693013,
                prefix="confirmation-trunk-com",
                ab_bank_sha256=(
                    "c9e82fa04a658e093db8af8bc6775c04044de0cc5c3154e23fa9a3bbd54fcd33"
                ),
                device="cuda:0",
            ),
        )
        protocol = WalkingCampaignProtocol(
            protocol_id="alpha-walking-trunk-com-cross-failure-v2",
            condition_family="trunk_com",
            profile=profile,
            calibration_bank=calibration,
            selection_bank=selection,
            nominal_selection_bank=nominal_selection,
            release_banks=release_banks,
            source_behavior_reference_required=True,
            source_behavior_match_mode="bounded_failure_count_and_paired_casewise",
            source_behavior_failure_count_tolerance=1,
            source_behavior_reference_platform="external",
        )
    elif adapter == "mjlab-trunk-payload-profile-v1":
        if not isinstance(profile, TrunkPayloadProfile):
            raise TypeError(
                "campaign profile is not a registered trunk-payload condition"
            )
        if campaign.optimizer.worlds_per_candidate % 4:
            raise ValueError(
                "trunk-payload worlds_per_candidate must balance four commands"
            )
        optimizer = campaign.optimizer
        if (
            optimizer.algorithm != "hyperscalees-eggroll"
            or optimizer.population != 512
            or optimizer.generations != 9
            or optimizer.rank != 4
            or optimizer.worlds_per_candidate != 4
            or optimizer.sigma != 0.015
            or optimizer.learning_rate != 0.003
            or optimizer.seed not in TRUNK_PAYLOAD_STUDY_SEEDS
            or optimizer.trainable_scope != "final-affine-weight-and-bias"
        ):
            raise ValueError("trunk-payload campaign changed the predeclared optimizer")
        if parameters != {
            "profile_name": name,
            "profile_sha256": expected_sha256,
            "body": "trunk_base",
            "added_mass_kg": profile.added_mass_kg,
        }:
            raise ValueError("trunk-payload campaign parameters changed")
        calibration = make_walking_proof_bank(
            base_seed=20794001,
            episodes_per_command=8,
            prefix="calibration-trunk-payload",
        )
        selection = make_walking_proof_bank(
            base_seed=20894004,
            episodes_per_command=8,
            prefix="selection-trunk-payload",
        )
        nominal_selection = make_walking_proof_bank(
            base_seed=20994007,
            episodes_per_command=8,
            prefix="selection-nominal-payload",
        )
        release_banks = (
            ReleaseBankContract(
                stage="production_runtime",
                base_seed=21094010,
                prefix="production-trunk-payload",
                ab_bank_sha256=(
                    "f113436614119516b30c87044ee58469bf549a55a6f89f1b822f1cf3684dee1d"
                ),
                device="cuda:0",
            ),
            ReleaseBankContract(
                stage="independent_confirmation",
                base_seed=21194013,
                prefix="confirmation-trunk-payload",
                ab_bank_sha256=(
                    "cfb35baf3970909d821d3f5efdd64f702ef486149be20d5f5f6dacf6b9f8b3f4"
                ),
                device="cuda:0",
            ),
        )
        protocol = WalkingCampaignProtocol(
            protocol_id="alpha-walking-trunk-payload-cross-failure-v1",
            condition_family="trunk_payload",
            profile=profile,
            calibration_bank=calibration,
            selection_bank=selection,
            nominal_selection_bank=nominal_selection,
            release_banks=release_banks,
            source_behavior_reference_required=True,
            source_behavior_match_mode="bounded_failure_count_and_paired_casewise",
            source_behavior_failure_count_tolerance=1,
            source_behavior_reference_platform="external",
        )
    else:
        raise ValueError(f"unsupported walking deployment adapter {adapter!r}")

    if (
        walking_bank_sha256(protocol.calibration_bank)
        != campaign.calibration_bank_sha256
    ):
        raise ValueError("campaign calibration bank does not match the frozen protocol")
    if walking_bank_sha256(protocol.selection_bank) != campaign.held_out_bank_sha256:
        raise ValueError("campaign selection bank does not match the frozen protocol")
    all_banks = [
        protocol.calibration_bank,
        protocol.selection_bank,
        protocol.nominal_selection_bank,
    ]
    if protocol.condition_family in {"trunk_com", "trunk_payload"}:
        all_banks.extend(
            make_walking_proof_bank(
                base_seed=bank.base_seed,
                episodes_per_command=8,
                prefix=bank.prefix,
            )
            for bank in protocol.release_banks
        )
        all_banks.extend(
            make_walking_proof_bank(
                base_seed=campaign.optimizer.seed + generation * 10_007,
                episodes_per_command=campaign.optimizer.worlds_per_candidate // 4,
                prefix=f"train-g{generation:06d}",
            )
            for generation in range(campaign.optimizer.generations)
        )
    fingerprints = [
        {(case.seed, case.command, case.horizon_steps) for case in bank}
        for bank in all_banks
    ]
    for index, first in enumerate(fingerprints):
        for second in fingerprints[index + 1 :]:
            if first & second:
                raise RuntimeError(
                    "walking training, calibration, or evaluation banks overlap"
                )
    return protocol
