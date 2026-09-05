"""Fail-closed aggregation for the predeclared trunk-CoM three-seed study."""

from __future__ import annotations

import hashlib
import json
import statistics
from collections.abc import Mapping, Sequence
from copy import deepcopy
from pathlib import Path
from typing import Any

from mjlab_microduck.eggroll.deployment import (
    TRUNK_COM_CALIBRATION_PROFILES,
    TRUNK_PAYLOAD_CALIBRATION_PROFILES,
)

from .campaign import load_candidate_checkpoint
from .contracts import PatchCampaign, ReleaseScope
from .efficiency import CostLedger, InteractionCost
from .integrated_validation import (
    _validate_stage_evidence,
    validate_integrated_early_stop_run,
)
from .qualification import (
    DEFAULT_QUALIFICATION_STAGES,
    QualificationController,
    QualificationPlan,
)
from .qualification_command import CommandQualificationSpec
from .qualification_runtime import validate_complete_paired_bank
from .walking_protocol import (
    TRUNK_COM_STUDY_SEEDS,
    TRUNK_PAYLOAD_STUDY_SEEDS,
    resolve_walking_protocol,
    walking_campaign_family_sha256,
)

SCHEMA = "eggroll-autopatch-cross-failure-study-validation-v1"
PROTOCOL_SCHEMA = "eggroll-autopatch-cross-failure-protocol-v2"
CALIBRATION_SCHEMA = "eggroll-autopatch-trunk-com-calibration-validation-v1"
PAYLOAD_PROTOCOL_SCHEMA = "eggroll-autopatch-payload-cross-failure-protocol-v1"
PAYLOAD_CALIBRATION_SCHEMA = "eggroll-autopatch-trunk-payload-calibration-validation-v1"
SOURCE_REFERENCE_SCHEMA = "eggroll-autopatch-source-behavior-reference-v1"


def build_trunk_com_study_contracts(
    *,
    base_campaign_path: Path,
    protocol_path: Path,
    calibration_validation_path: Path,
) -> dict[str, Any]:
    """Materialize the three campaigns only after the frozen calibration passes."""

    protocol = _read_object(protocol_path)
    calibration = _read_object(calibration_validation_path)
    protocol_schema = protocol.get("schema")
    if protocol_schema == PAYLOAD_PROTOCOL_SCHEMA:
        calibration_schema = PAYLOAD_CALIBRATION_SCHEMA
        profile_ladder = TRUNK_PAYLOAD_CALIBRATION_PROFILES
        study_seeds = TRUNK_PAYLOAD_STUDY_SEEDS
        condition_label = "trunk-payload"
        adapter = "mjlab-trunk-payload-profile-v1"
    elif protocol_schema == PROTOCOL_SCHEMA:
        calibration_schema = CALIBRATION_SCHEMA
        profile_ladder = TRUNK_COM_CALIBRATION_PROFILES
        study_seeds = TRUNK_COM_STUDY_SEEDS
        condition_label = "trunk-com"
        adapter = "mjlab-trunk-com-shift-profile-v1"
    else:
        raise ValueError("protocol schema mismatch")
    _require(calibration.get("schema"), calibration_schema, name="calibration schema")
    _require(calibration.get("status"), "pass", name="calibration validation")
    _require(
        calibration.get("calibration_status"),
        "condition-frozen",
        name="calibration outcome",
    )
    _require(
        calibration.get("protocol_sha256"),
        _sha256_file(protocol_path),
        name="calibration protocol bytes",
    )
    selected_sha256 = calibration.get("selected_profile_sha256")
    profiles = {profile.sha256: profile for profile in profile_ladder}
    selected = profiles.get(selected_sha256)
    if selected is None:
        raise ValueError(
            "calibration selected a profile outside the predeclared ladder"
        )
    declared_profiles = protocol.get("condition", {}).get("profiles", [])
    matches = [
        row
        for row in declared_profiles
        if isinstance(row, Mapping) and row.get("sha256") == selected.sha256
    ]
    if len(matches) != 1 or matches[0].get("name") != selected.name:
        raise ValueError("calibrated profile does not match the protocol ladder")
    source = protocol.get("source")
    training = protocol.get("training")
    banks = protocol.get("banks")
    if not all(isinstance(value, Mapping) for value in (source, training, banks)):
        raise TypeError("protocol source, training, or banks contract is missing")
    base = PatchCampaign.from_json(base_campaign_path.read_text())
    _require(base.artifact_id, source.get("artifact_id"), name="source artifact")
    _require(
        base.artifact_sha256,
        source.get("policy_sha256"),
        name="source policy",
    )
    _require(
        tuple(training.get("optimizer_seeds", ())),
        study_seeds,
        name="study seeds",
    )
    campaigns: list[PatchCampaign] = []
    for ordinal, seed in enumerate(study_seeds, start=1):
        document = deepcopy(base.canonical_dict())
        document["campaign_id"] = (
            f"alpha-walking-{condition_label}-efficiency-seed{ordinal}-integrated-v1"
        )
        document["optimizer"]["seed"] = seed
        document["optimizer"]["generations"] = int(training["generations_limit"])
        if protocol_schema == PAYLOAD_PROTOCOL_SCHEMA:
            parameters = [
                ["profile_name", selected.name],
                ["profile_sha256", selected.sha256],
                ["body", "trunk_base"],
                ["added_mass_kg", selected.added_mass_kg],
            ]
            description = (
                "A hidden fixed payload added to trunk_base with seeded mass and "
                "pseudo-inertia scaled together; observations and policy API remain "
                "unchanged."
            )
        else:
            parameters = [
                ["profile_name", selected.name],
                ["profile_sha256", selected.sha256],
                ["body", "trunk_base"],
                ["offset_x_m", selected.offset_m[0]],
            ]
            description = (
                "A hidden fixed local-forward shift of trunk_base's inertial "
                "position; body mass, inertia tensor, observations, and policy "
                "API remain unchanged."
            )
        document["condition"] = {
            "condition_id": selected.name,
            "adapter": adapter,
            "parameters": parameters,
            "hidden_from_actor": True,
            "description": description,
        }
        document["calibration_bank_sha256"] = protocol["calibration"][
            "walking_case_bank_sha256"
        ]
        document["held_out_bank_sha256"] = banks["selection"][
            "walking_case_bank_sha256"
        ]
        campaign = PatchCampaign.from_dict(document)
        resolve_walking_protocol(campaign)
        campaigns.append(campaign)
    families = {walking_campaign_family_sha256(campaign) for campaign in campaigns}
    if len(families) != 1:
        raise RuntimeError("predeclared study campaigns do not share one family")
    release_scope = ReleaseScope(
        scope_id=f"alpha-walking-{selected.name}-profile-specific-v1",
        mode="profile_specific",
        profile_sha256s=(("shifted", selected.sha256),),
        required_retention_roles=("shifted",),
        activation_profile_role="shifted",
        activation_predicate=(
            f"hardware.{condition_label.replace('-', '_')}.profile_sha256 == "
            f"{selected.sha256}"
        ),
        source_fallback_sha256=base.artifact_sha256,
        unknown_profile_action="retain_source",
    )
    return {
        "protocol_id": protocol["protocol_id"],
        "protocol_sha256": _sha256_file(protocol_path),
        "calibration_validation_sha256": _sha256_file(calibration_validation_path),
        "selected_profile_name": selected.name,
        "selected_profile_sha256": selected.sha256,
        "campaign_family_sha256": next(iter(families)),
        "campaigns": [campaign.canonical_dict() for campaign in campaigns],
        "campaign_sha256s": [campaign.sha256 for campaign in campaigns],
        "release_scope": release_scope.canonical_dict(),
        "release_scope_sha256": release_scope.sha256,
    }


def build_trunk_payload_study_contracts(
    *,
    base_campaign_path: Path,
    protocol_path: Path,
    calibration_validation_path: Path,
) -> dict[str, Any]:
    """Materialize payload campaigns through the shared fail-closed builder."""

    return build_trunk_com_study_contracts(
        base_campaign_path=base_campaign_path,
        protocol_path=protocol_path,
        calibration_validation_path=calibration_validation_path,
    )


def _reject_nonfinite(value: str) -> None:
    raise ValueError(f"non-finite JSON constant: {value}")


def _read_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(), parse_constant=_reject_nonfinite)
    if not isinstance(value, dict):
        raise TypeError(f"{path} must contain one JSON object")
    return value


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _require(actual: Any, expected: Any, *, name: str) -> None:
    if actual != expected:
        raise ValueError(f"{name} mismatch: expected {expected!r}, got {actual!r}")


def _positive_integer(value: Any, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise TypeError(f"{name} must be a positive integer")
    return value


def _seed_contracts(
    *,
    root: Path,
    protocol_id: str,
    selected_profile_sha256: str,
    family_sha256: str,
    source_reference_path: Path,
    source_reference_filename: str,
    hf_hardware_flavor: str,
) -> tuple[
    PatchCampaign,
    ReleaseScope,
    QualificationPlan,
    CommandQualificationSpec,
    dict[str, Any],
]:
    campaign = PatchCampaign.from_json((root / "campaign.json").read_text())
    walking = resolve_walking_protocol(campaign)
    _require(walking.protocol_id, protocol_id, name="walking protocol")
    _require(
        walking.profile.sha256,
        selected_profile_sha256,
        name="calibrated activation profile",
    )
    _require(
        walking_campaign_family_sha256(campaign),
        family_sha256,
        name="shared campaign family",
    )
    release_scope = ReleaseScope.from_json((root / "release_scope.json").read_text())
    _require(release_scope.mode, "profile_specific", name="release-scope mode")
    _require(
        release_scope.source_fallback_sha256,
        campaign.artifact_sha256,
        name="exact source fallback",
    )
    activation_role = release_scope.activation_profile_role
    if activation_role is None:
        raise ValueError("release scope has no activation role")
    _require(
        dict(release_scope.profile_sha256s).get(activation_role),
        walking.profile.sha256,
        name="release-scope activation profile",
    )
    _require(
        release_scope.unknown_profile_action,
        "retain_source",
        name="unknown-profile action",
    )
    plan = QualificationPlan.from_json((root / "qualification_plan.json").read_text())
    spec = CommandQualificationSpec.from_json(
        (root / "qualification_command_spec.json").read_text()
    )
    spec.validate_plan(plan)
    _require(plan.evaluation_interval, 1, name="qualification interval")
    _require(
        plan.required_stages,
        DEFAULT_QUALIFICATION_STAGES,
        name="six-stage qualification order",
    )
    commands = {command.stage: command for command in spec.commands}
    for stage in ("production_runtime", "independent_confirmation"):
        command = commands[stage]
        _require(
            command.execution_failure_action,
            "abort_campaign",
            name=f"{stage} infrastructure failure action",
        )
        reference_tokens = [
            token for token in command.argv if token.endswith(source_reference_filename)
        ]
        if len(reference_tokens) != 1:
            raise ValueError(f"{stage} does not bind the frozen source reference")
    source_manifest = _read_object(root / "source_manifest.json")
    _require(source_manifest.get("mode"), "train", name="source manifest mode")
    _require(
        source_manifest.get("evidence_role"),
        "frozen campaign",
        name="source manifest role",
    )
    _require(
        source_manifest.get("base_campaign_sha256"),
        campaign.sha256,
        name="source manifest campaign",
    )
    _require(
        source_manifest.get("execution_campaign_sha256"),
        campaign.sha256,
        name="source manifest execution campaign",
    )
    _require(
        source_manifest.get("release_scope_sha256"),
        release_scope.sha256,
        name="source manifest release scope",
    )
    _require(
        source_manifest.get("qualification_plan_sha256"),
        plan.sha256,
        name="source manifest qualification plan",
    )
    _require(
        source_manifest.get("qualification_command_spec_sha256"),
        spec.sha256,
        name="source manifest qualification commands",
    )
    _require(
        source_manifest.get("source_policy_sha256"),
        campaign.artifact_sha256,
        name="source manifest policy",
    )
    _require(
        source_manifest.get("hf_hardware_flavor"),
        hf_hardware_flavor,
        name="source manifest HF hardware flavor",
    )
    source_commit = source_manifest.get("source_commit")
    if not isinstance(source_commit, str) or len(source_commit) != 40:
        raise ValueError("seed output has no full source commit")
    # Resolve now so the caller cannot accidentally validate against a different
    # relative path after the contracts have passed.
    if not source_reference_path.resolve().is_file():
        raise FileNotFoundError(source_reference_path)
    return campaign, release_scope, plan, spec, source_manifest


def _validate_noneligible_seed(
    *,
    root: Path,
    campaign: PatchCampaign,
    plan: QualificationPlan,
    spec: CommandQualificationSpec,
    ceiling: int,
) -> dict[str, Any]:
    """Verify a complete capped campaign that produced no eligible checkpoint."""

    run = root / "run"
    config = _read_object(run / "config.json")
    accounting = _read_object(run / "accounting.json")
    budget = _read_object(run / "budget.json")
    qualification = _read_object(run / "qualification.json")
    _require(config.get("campaign_sha256"), campaign.sha256, name="run campaign")
    _require(
        qualification.get("schema"),
        QualificationController.SCHEMA,
        name="qualification schema",
    )
    _require(qualification.get("plan_sha256"), plan.sha256, name="run plan")
    _require(qualification.get("stop_generation"), None, name="noneligible stop")
    _require(
        budget.get("completed_generations"),
        campaign.optimizer.generations,
        name="completed capped generations",
    )
    _require(
        budget.get("stopped_after_complete_release_qualification"),
        False,
        name="noneligible stop state",
    )
    _require(budget.get("qualification"), qualification, name="budget qualification")
    if accounting.get("executed_steps_complete") is not True:
        raise ValueError("noneligible seed lacks complete executed-step accounting")
    requested = _positive_integer(
        budget.get("requested_optimization_simulator_steps"),
        name="requested optimization steps",
    )
    if requested > ceiling:
        raise ValueError("noneligible seed exceeded the optimization ceiling")
    expected_candidates = campaign.optimizer.generations * campaign.optimizer.population
    _require(
        budget.get("candidate_evaluations"),
        expected_candidates,
        name="noneligible candidate evaluations",
    )
    attempts = qualification.get("attempts")
    if (
        not isinstance(attempts, list)
        or len(attempts) != campaign.optimizer.generations
    ):
        raise ValueError(
            "noneligible seed lacks every predeclared qualification attempt"
        )
    _require(
        [attempt.get("generation") for attempt in attempts],
        list(range(1, campaign.optimizer.generations + 1)),
        name="noneligible attempt order",
    )
    if any(attempt.get("status") != "rejected" for attempt in attempts):
        raise ValueError("noneligible seed contains a pending or eligible attempt")
    reconstructed = CostLedger()
    failures: list[dict[str, Any]] = []
    for attempt in attempts:
        generation = int(attempt["generation"])
        candidate_path = (
            run / "qualification_candidates" / f"generation-{generation:06d}.npz"
        )
        if _sha256_file(candidate_path) != attempt.get("checkpoint_sha256"):
            raise ValueError("noneligible qualification candidate bytes changed")
        load_candidate_checkpoint(candidate_path, campaign=campaign)
        selection = attempt.get("selection")
        stages = attempt.get("stages")
        if not isinstance(selection, Mapping) or not isinstance(stages, list):
            raise TypeError("noneligible qualification attempt is incomplete")
        reconstructed.record(
            "qualification.selection",
            InteractionCost.from_dict(selection["cost"]),
        )
        for stage in stages:
            reconstructed.record(
                f"qualification.{stage['stage']}",
                InteractionCost.from_dict(stage["cost"]),
            )
        _validate_stage_evidence(run_dir=run, attempt=attempt, spec=spec)
        failures.append(
            {
                "generation": generation,
                "rejection_reason": attempt.get("rejection_reason"),
                "stages_executed": len(stages),
            }
        )
    _require(
        qualification.get("cost_ledger"),
        reconstructed.state_dict(),
        name="noneligible qualification costs",
    )
    return {
        "status": "complete-noneligible",
        "seed": campaign.optimizer.seed,
        "campaign_id": campaign.campaign_id,
        "campaign_sha256": campaign.sha256,
        "requested_optimization_simulator_steps": requested,
        "failed_qualification_attempts": failures,
        "qualification_cost": reconstructed.total().to_dict(),
    }


def validate_cross_failure_study(
    *,
    protocol_path: Path,
    calibration_validation_path: Path,
    source_behavior_reference_path: Path,
    seed_output_dirs: Sequence[Path],
    max_requested_optimization_steps: int = 5_120_000,
) -> dict[str, Any]:
    """Verify all seed artifacts and decide the predeclared generality criterion."""

    ceiling = _positive_integer(
        max_requested_optimization_steps,
        name="requested optimization simulator-step ceiling",
    )
    if len(seed_output_dirs) != 3:
        raise ValueError("cross-failure study requires exactly three seed outputs")
    roots = [root.resolve() for root in seed_output_dirs]
    if len(set(roots)) != 3:
        raise ValueError("cross-failure study seed outputs must be distinct")

    protocol = _read_object(protocol_path)
    calibration = _read_object(calibration_validation_path)
    source_reference = _read_object(source_behavior_reference_path)
    protocol_schema = protocol.get("schema")
    if protocol_schema == PAYLOAD_PROTOCOL_SCHEMA:
        calibration_schema = PAYLOAD_CALIBRATION_SCHEMA
        study_seeds = TRUNK_PAYLOAD_STUDY_SEEDS
        source_reference_filename = "walking_trunk_payload_source_behavior_v1.json"
        condition_claim = "trunk-payload"
    elif protocol_schema == PROTOCOL_SCHEMA:
        calibration_schema = CALIBRATION_SCHEMA
        study_seeds = TRUNK_COM_STUDY_SEEDS
        source_reference_filename = "walking_trunk_com_source_behavior_v1.json"
        condition_claim = "trunk-CoM"
    else:
        raise ValueError("protocol schema mismatch")
    protocol_id = protocol.get("protocol_id")
    if not isinstance(protocol_id, str) or not protocol_id:
        raise ValueError("protocol id is missing")
    training = protocol.get("training")
    if not isinstance(training, Mapping):
        raise TypeError("protocol training contract is missing")
    _require(
        tuple(training.get("optimizer_seeds", ())),
        study_seeds,
        name="predeclared optimizer seeds",
    )
    _require(
        training.get("max_requested_optimization_simulator_steps_per_seed"),
        ceiling,
        name="protocol optimization ceiling",
    )
    _require(
        training.get("failure_frontier_branching"),
        "disabled",
        name="failure-frontier branching",
    )

    _require(calibration.get("schema"), calibration_schema, name="calibration schema")
    _require(calibration.get("status"), "pass", name="calibration validation")
    _require(
        calibration.get("calibration_status"),
        "condition-frozen",
        name="calibration outcome",
    )
    _require(
        calibration.get("protocol_sha256"),
        _sha256_file(protocol_path),
        name="calibration protocol bytes",
    )
    selected_profile_sha256 = calibration.get("selected_profile_sha256")
    if (
        not isinstance(selected_profile_sha256, str)
        or len(selected_profile_sha256) != 64
    ):
        raise ValueError("calibration did not select a content-addressed profile")

    _require(
        source_reference.get("schema"),
        SOURCE_REFERENCE_SCHEMA,
        name="source reference schema",
    )
    _require(source_reference.get("status"), "pass", name="source reference")
    _require(
        source_reference.get("walking_protocol_id"),
        protocol_id,
        name="source reference protocol",
    )
    _require(
        source_reference.get("source_policy_sha256"),
        calibration.get("source_policy_sha256"),
        name="source reference policy",
    )
    _require(
        source_reference.get("activation_profile_sha256"),
        selected_profile_sha256,
        name="source reference profile",
    )
    _require(
        source_reference.get("calibration_validation_sha256"),
        _sha256_file(calibration_validation_path),
        name="source reference calibration bytes",
    )
    family_sha256 = source_reference.get("campaign_family_sha256")
    if not isinstance(family_sha256, str) or len(family_sha256) != 64:
        raise ValueError("source reference lacks the campaign family hash")
    hf_hardware_flavor = source_reference.get("hf_hardware_flavor")
    if not isinstance(hf_hardware_flavor, str) or not hf_hardware_flavor:
        raise ValueError("source reference lacks the HF hardware flavor")
    if calibration.get("hf_hardware_flavor") != hf_hardware_flavor:
        raise ValueError("calibration and source reference hardware differ")

    seed_records: list[dict[str, Any]] = []
    observed_seeds: list[int] = []
    source_commits: set[str] = set()
    release_scope_hashes: set[str] = set()
    plan_hashes: set[str] = set()
    command_hashes: set[str] = set()
    for root in roots:
        campaign, release_scope, plan, spec, source_manifest = _seed_contracts(
            root=root,
            protocol_id=protocol_id,
            selected_profile_sha256=selected_profile_sha256,
            family_sha256=family_sha256,
            source_reference_path=source_behavior_reference_path,
            source_reference_filename=source_reference_filename,
            hf_hardware_flavor=hf_hardware_flavor,
        )
        observed_seeds.append(campaign.optimizer.seed)
        source_commits.add(str(source_manifest["source_commit"]))
        release_scope_hashes.add(release_scope.sha256)
        plan_hashes.add(plan.sha256)
        command_hashes.add(spec.sha256)
        integrated_path = root / "integrated_validation.json"
        if not integrated_path.is_file():
            seed_records.append(
                _validate_noneligible_seed(
                    root=root,
                    campaign=campaign,
                    plan=plan,
                    spec=spec,
                    ceiling=ceiling,
                )
            )
            continue
        stored = _read_object(integrated_path)
        fresh = validate_integrated_early_stop_run(
            run_dir=root / "run",
            campaign_path=root / "campaign.json",
            release_scope_path=root / "release_scope.json",
            qualification_plan_path=root / "qualification_plan.json",
            qualification_command_spec_path=root / "qualification_command_spec.json",
            source_manifest_path=root / "source_manifest.json",
            selection_record_path=root / "selection.json",
            output_policy_path=root / "adapted_policy.onnx",
            max_requested_optimization_steps=ceiling,
        )
        _require(stored, fresh, name="persisted integrated validation")
        runtime_reports: dict[str, Any] = {}
        for stage in ("production_runtime", "independent_confirmation"):
            manifest_path = (
                root
                / "run/qualification_evidence"
                / f"generation-{fresh['stop_generation']}"
                / stage
                / "manifest.json"
            )
            report, cost = validate_complete_paired_bank(
                manifest=_read_object(manifest_path),
                manifest_path=manifest_path,
                stage=stage,  # type: ignore[arg-type]
                source_sha256=campaign.artifact_sha256,
                adapted_sha256=str(fresh["output_policy_sha256"]),
                campaign=campaign,
                source_behavior_reference_path=source_behavior_reference_path,
            )
            _require(report.get("adapted_successes"), 32, name=f"{stage} successes")
            _require(
                report.get("source_success_regressions"),
                0,
                name=f"{stage} source regressions",
            )
            runtime_reports[stage] = {
                "source_behavior_reference_id": report.get(
                    "source_behavior_reference_id"
                ),
                "source_successes": report.get("source_successes"),
                "adapted_successes": report.get("adapted_successes"),
                "source_success_regressions": report.get("source_success_regressions"),
                "cost": cost.to_dict(),
                "manifest_sha256": _sha256_file(manifest_path),
            }
        seed_records.append(
            {
                "status": "release-eligible",
                "seed": campaign.optimizer.seed,
                "campaign_id": campaign.campaign_id,
                "campaign_sha256": campaign.sha256,
                "stop_generation": fresh["stop_generation"],
                "requested_optimization_simulator_steps": fresh[
                    "requested_optimization_simulator_steps"
                ],
                "failed_qualification_attempts": fresh["failed_qualification_attempts"],
                "checkpoint_sha256": fresh["final_candidate_checkpoint_sha256"],
                "adapted_policy_sha256": fresh["output_policy_sha256"],
                "onnx_parity_max_abs_error": fresh["onnx_parity_max_abs_error"],
                "interaction_cost": fresh["interaction_cost"],
                "runtime_reports": runtime_reports,
            }
        )

    _require(tuple(sorted(observed_seeds)), study_seeds, name="study seeds")
    if any(
        len(values) != 1
        for values in (
            source_commits,
            release_scope_hashes,
            plan_hashes,
            command_hashes,
        )
    ):
        raise ValueError("seed outputs do not share one code and release contract")
    seed_records.sort(key=lambda record: study_seeds.index(record["seed"]))
    eligible = [
        record for record in seed_records if record["status"] == "release-eligible"
    ]
    censored_costs = [
        int(record["requested_optimization_simulator_steps"])
        if record["status"] == "release-eligible"
        else ceiling + 1
        for record in seed_records
    ]
    median_censored = int(statistics.median(censored_costs))
    success = len(eligible) >= 2 and median_censored <= ceiling
    total_requested = sum(
        int(record["requested_optimization_simulator_steps"]) for record in seed_records
    )
    return {
        "schema": SCHEMA,
        "status": "pass" if success else "negative-result",
        "protocol_id": protocol_id,
        "protocol_sha256": _sha256_file(protocol_path),
        "calibration_validation_sha256": _sha256_file(calibration_validation_path),
        "source_behavior_reference_sha256": _sha256_file(
            source_behavior_reference_path
        ),
        "source_policy_sha256": calibration["source_policy_sha256"],
        "activation_profile_sha256": selected_profile_sha256,
        "campaign_family_sha256": family_sha256,
        "hf_hardware_flavor": hf_hardware_flavor,
        "training_source_commit": next(iter(source_commits)),
        "release_scope_sha256": next(iter(release_scope_hashes)),
        "qualification_plan_sha256": next(iter(plan_hashes)),
        "qualification_command_spec_sha256": next(iter(command_hashes)),
        "independent_seeds": 3,
        "release_eligible_seeds": len(eligible),
        "release_eligible_fraction": len(eligible) / 3,
        "median_requested_optimization_steps_with_noneligible_censored_above_ceiling": median_censored,
        "median_requested_optimization_steps_among_eligible": (
            None
            if not eligible
            else int(
                statistics.median(
                    record["requested_optimization_simulator_steps"]
                    for record in eligible
                )
            )
        ),
        "total_requested_optimization_simulator_steps": total_requested,
        "requested_optimization_simulator_step_ceiling_per_seed": ceiling,
        "success_condition": (
            "at least two of three predeclared seeds release eligible and the "
            "three-seed median, with noneligible runs censored above the ceiling, "
            "is at most 5,120,000 requested optimization simulator steps"
        ),
        "success_condition_met": success,
        "cross_failure_assessment": (
            "bounded-cross-failure-generality-supported"
            if success
            else "cross-failure-generality-not-established"
        ),
        "seeds": seed_records,
        "failure_frontier_branching": "disabled",
        "blind_population_or_rank_grid": "not-run",
        "claim_boundary": (
            f"orthogonal {condition_claim} simulation and production-runtime "
            "digital-twin "
            "evidence under one fixed three-seed protocol; no physical-robot, broad "
            "hardware-failure, transfer, or optimizer-superiority claim"
        ),
    }


def write_cross_failure_validation(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(payload), indent=2, sort_keys=True) + "\n")
