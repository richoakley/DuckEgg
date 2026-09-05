"""Candidate-bound production qualification stages for the walking benchmark.

The trainer invokes this module through :class:`CommandQualificationBackend`.
Each invocation validates the same checkpoint identity, emits one immutable result
manifest, and either passes or rejects that candidate.  Identity and contract drift
remain fatal instead of being converted into an ordinary candidate rejection.
"""

from __future__ import annotations

import hashlib
import json
import math
import numbers
import os
import subprocess
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any, Literal

from mjlab_microduck.eggroll.policy_io import (
    export_adapted_policy,
    import_deployed_policy,
)
from mjlab_microduck.eggroll.release import (
    runtime_parity,
    verify_output_layer_derivative,
)

from .ab import ABCase, run_paired_ab_suite
from .campaign import load_candidate_checkpoint
from .contracts import PatchCampaign, ReleaseScope
from .efficiency import InteractionCost
from .foot_proof import make_walking_proof_bank
from .qualification import DEFAULT_QUALIFICATION_STAGES
from .qualification_command import RESULT_SCHEMA
from .registry import PRODUCTION_REGISTRY
from .release import (
    build_paired_non_regression_envelope,
    evaluate_paired_non_regression,
    validate_routing_evidence,
)
from .walking_protocol import (
    resolve_walking_protocol,
    walking_campaign_family_sha256,
)

FIRST_BANK_BASE_SEED = 20262021
CONFIRMATION_BANK_BASE_SEED = 20262023
BANK_CASES = 32
REQUESTED_STEPS_PER_WORLD = 250
FIRST_BANK_SHA256 = "ba760ae8dcbb6c0b5827ab8c38bcbe6c4f4a5b41bc85864c0447af24f55eff01"
CONFIRMATION_BANK_SHA256 = (
    "106a0c05307852fc6c0b05c383ab658ce2c54fef7d161105cdf4ca97c983d307"
)
EXPECTED_ADAPTED_SUCCESSES = BANK_CASES
HISTORICAL_STAGE_DEVICES = {
    # The first sealed bank was recorded locally through the evaluator's CPU
    # default.  The independent confirmation launch explicitly used cuda:0.
    # Preserve that distinction: seeded MuJoCo execution is not interchangeable
    # across the two devices for an exact historical source-behaviour check.
    "production_runtime": "cpu",
    "independent_confirmation": "cuda:0",
}
SOURCE_BEHAVIOR_FAILURE_REFERENCES = {
    ("production_runtime", "darwin", "cpu"): (
        "heldout-wedge-vx-0.28-000",
        "heldout-wedge-vx-0.28-003",
        "heldout-wedge-vx-0.28-005",
        "heldout-wedge-vx-0.28-007",
        "heldout-wedge-vx-0.32-004",
        "heldout-wedge-vx-0.40-002",
        "heldout-wedge-vx-0.40-006",
        "heldout-wedge-vx-0.40-007",
    ),
    ("production_runtime", "linux", "cpu"): (
        "heldout-wedge-vx-0.28-003",
        "heldout-wedge-vx-0.28-005",
        "heldout-wedge-vx-0.32-002",
        "heldout-wedge-vx-0.32-003",
        "heldout-wedge-vx-0.32-004",
        "heldout-wedge-vx-0.36-001",
        "heldout-wedge-vx-0.36-006",
        "heldout-wedge-vx-0.40-002",
        "heldout-wedge-vx-0.40-004",
        "heldout-wedge-vx-0.40-006",
        "heldout-wedge-vx-0.40-007",
    ),
    ("independent_confirmation", "linux", "cuda:0"): (
        "confirmation-wedge-vx-0.32-002",
        "confirmation-wedge-vx-0.32-004",
        "confirmation-wedge-vx-0.32-007",
        "confirmation-wedge-vx-0.36-000",
        "confirmation-wedge-vx-0.36-001",
        "confirmation-wedge-vx-0.40-001",
        "confirmation-wedge-vx-0.40-002",
        "confirmation-wedge-vx-0.40-005",
        "confirmation-wedge-vx-0.40-006",
    ),
}

Stage = Literal[
    "release_scope_retention",
    "onnx_parity",
    "production_runtime",
    "independent_confirmation",
    "profile_routing",
    "signed_activation_and_rollback",
]


class CandidateRejected(RuntimeError):
    """A valid candidate failed a release gate and training may continue."""

    def __init__(
        self,
        reason: str,
        evidence: dict[str, Any] | None = None,
        cost: InteractionCost | None = None,
    ) -> None:
        super().__init__(reason)
        self.reason = reason
        self.evidence = evidence or {}
        self.cost = cost or InteractionCost()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _walking_ab_bank(*, base_seed: int, prefix: str) -> tuple[ABCase, ...]:
    cases = make_walking_proof_bank(
        base_seed=base_seed,
        episodes_per_command=8,
        prefix=prefix,
    )
    return tuple(
        ABCase(
            case_id=case.case_id,
            seed=case.seed,
            task="Mjlab-Velocity-Flat-MicroDuck",
            reset_label="standing",
            side="right",
            mode="walk",
            command=case.command,
            horizon_steps=REQUESTED_STEPS_PER_WORLD,
        )
        for case in cases
    )


def frozen_release_banks(
    campaign: PatchCampaign | None = None,
) -> dict[str, tuple[ABCase, ...]]:
    """Recreate the two historical banks and reject generator drift."""

    if campaign is None:
        specifications = {
            "production_runtime": (
                FIRST_BANK_BASE_SEED,
                "heldout-wedge",
                FIRST_BANK_SHA256,
            ),
            "independent_confirmation": (
                CONFIRMATION_BANK_BASE_SEED,
                "confirmation-wedge",
                CONFIRMATION_BANK_SHA256,
            ),
        }
    else:
        protocol = resolve_walking_protocol(campaign)
        specifications = {
            bank.stage: (bank.base_seed, bank.prefix, bank.ab_bank_sha256)
            for bank in protocol.release_banks
        }
    banks = {
        stage: _walking_ab_bank(
            base_seed=base_seed,
            prefix=prefix,
        )
        for stage, (base_seed, prefix, _sha256) in specifications.items()
    }
    for stage, cases in banks.items():
        observed = _canonical_sha256([asdict(case) for case in cases])
        expected = specifications[stage][2]
        if observed != expected:
            raise RuntimeError(
                f"frozen {stage} bank drifted: expected {expected}, got {observed}"
            )
    first_seeds = {case.seed for case in banks["production_runtime"]}
    confirmation_seeds = {case.seed for case in banks["independent_confirmation"]}
    if first_seeds & confirmation_seeds:
        raise RuntimeError("release and confirmation banks reuse scenario seeds")
    return banks


def _interaction_cost(manifest: dict[str, Any]) -> InteractionCost:
    rows = manifest.get("rows")
    if not isinstance(rows, list) or len(rows) != BANK_CASES:
        raise RuntimeError("paired runtime manifest does not contain the frozen bank")
    executed = 0
    for row in rows:
        if not isinstance(row, dict):
            raise TypeError("paired runtime row must be an object")
        for policy_role in ("source", "adapted"):
            result = row.get(policy_role, {}).get("result")
            if not isinstance(result, dict):
                raise TypeError("paired runtime row has no policy result")
            steps = result.get("episode_steps")
            if (
                isinstance(steps, bool)
                or not isinstance(steps, numbers.Real)
                or not math.isfinite(float(steps))
                or not float(steps).is_integer()
                or not 0 < int(steps) <= REQUESTED_STEPS_PER_WORLD
            ):
                raise ValueError("paired runtime row has invalid executed steps")
            executed += int(steps)
    rollouts = len(rows) * 2
    return InteractionCost(
        world_rollouts=rollouts,
        requested_simulator_steps=rollouts * REQUESTED_STEPS_PER_WORLD,
        executed_simulator_steps=executed,
        active_interaction_steps=executed,
        policy_forward_rows=executed,
        physics_substeps=executed * 4,
        world_constructions=rollouts,
    )


def source_behavior_reference(
    stage: Literal["production_runtime", "independent_confirmation"],
) -> tuple[str, tuple[str, ...]]:
    device = HISTORICAL_STAGE_DEVICES[stage]
    key = (stage, sys.platform, device)
    try:
        failures = SOURCE_BEHAVIOR_FAILURE_REFERENCES[key]
    except KeyError as error:
        raise RuntimeError(
            f"no sealed source-behavior reference for {stage} on "
            f"{sys.platform}/{device}"
        ) from error
    return f"{stage}-{sys.platform}-{device}", failures


def _campaign_source_behavior_reference(
    *,
    stage: Literal["production_runtime", "independent_confirmation"],
    campaign: PatchCampaign,
    reference_path: Path | None,
) -> tuple[str, tuple[str, ...]]:
    protocol = resolve_walking_protocol(campaign)
    if not protocol.source_behavior_reference_required:
        if reference_path is not None:
            raise ValueError("historical wedge qualification forbids a new reference")
        if protocol.source_behavior_reference_platform == "runtime":
            return source_behavior_reference(stage)
        if protocol.source_behavior_reference_platform != "linux":
            raise RuntimeError("walking source reference platform is invalid")
        bank = protocol.release_bank(stage)
        key = (stage, "linux", bank.device)
        try:
            failures = SOURCE_BEHAVIOR_FAILURE_REFERENCES[key]
        except KeyError as error:  # pragma: no cover - frozen protocol invariant.
            raise RuntimeError(
                f"no sealed Linux source-behavior count for {stage}/{bank.device}"
            ) from error
        return f"{protocol.protocol_id}:{stage}:linux-{bank.device}", failures
    if protocol.source_behavior_reference_platform != "external":
        raise RuntimeError("external source reference protocol is misconfigured")
    if reference_path is None:
        raise ValueError(
            "physical-condition qualification requires a frozen source reference"
        )
    reference = _load_json_object(reference_path)
    if reference.get("schema") != "eggroll-autopatch-source-behavior-reference-v1":
        raise ValueError("unknown source-behavior reference schema")
    if reference.get("status") != "pass":
        raise ValueError("source-behavior reference did not pass")
    reference_flavor = reference.get("hf_hardware_flavor")
    runtime_flavor = os.environ.get("EGGROLL_HF_HARDWARE_FLAVOR")
    if not isinstance(reference_flavor, str) or not reference_flavor:
        raise RuntimeError("source-behavior reference has no HF hardware flavor")
    if runtime_flavor is not None and runtime_flavor != reference_flavor:
        raise RuntimeError(
            "qualification hardware differs from the frozen source-behavior "
            f"environment: expected {reference_flavor!r}, got {runtime_flavor!r}"
        )
    if reference.get("walking_protocol_id") != protocol.protocol_id:
        raise ValueError("source-behavior reference names a different protocol")
    if reference.get("source_policy_sha256") != campaign.artifact_sha256:
        raise ValueError("source-behavior reference used different source bytes")
    if reference.get("campaign_family_sha256") != walking_campaign_family_sha256(
        campaign
    ):
        raise ValueError("source-behavior reference names a different campaign family")
    capture_campaign_sha256 = reference.get("capture_campaign_sha256")
    if (
        not isinstance(capture_campaign_sha256, str)
        or len(capture_campaign_sha256) != 64
    ):
        raise ValueError("source-behavior reference lacks capture-campaign provenance")
    if reference.get("activation_profile_sha256") != protocol.profile.sha256:
        raise ValueError("source-behavior reference used a different hardware profile")
    source_commit = reference.get("source_commit")
    calibration_sha256 = reference.get("calibration_validation_sha256")
    if not isinstance(source_commit, str) or len(source_commit) != 40:
        raise ValueError("source-behavior reference has no full Git identity")
    if not isinstance(calibration_sha256, str) or len(calibration_sha256) != 64:
        raise ValueError("source-behavior reference does not bind its calibration")
    reference_cost = reference.get("cost")
    if not isinstance(reference_cost, dict):
        raise TypeError("source-behavior reference cost is missing")
    if (
        InteractionCost.from_dict(reference_cost).world_rollouts != 64
        or InteractionCost.from_dict(reference_cost).requested_simulator_steps != 16_000
        or reference.get("candidate_optimization_evaluations") != 0
    ):
        raise ValueError("source-behavior reference cost is not exact")
    stages = reference.get("stages")
    if not isinstance(stages, dict) or not isinstance(stages.get(stage), dict):
        raise TypeError("source-behavior reference is missing a release stage")
    stage_reference = stages[stage]
    bank = protocol.release_bank(stage)
    if (
        stage_reference.get("device") != bank.device
        or stage_reference.get("ab_bank_sha256") != bank.ab_bank_sha256
    ):
        raise ValueError("source-behavior reference stage identity changed")
    failures = stage_reference.get("source_failure_case_ids")
    if not isinstance(failures, list) or not all(
        isinstance(case_id, str) for case_id in failures
    ):
        raise TypeError("source-behavior failure ids must be strings")
    if len(failures) != len(set(failures)):
        raise ValueError("source-behavior failure ids are not unique")
    if not 0 < len(failures) < BANK_CASES:
        raise ValueError(
            "source-behavior reference is not a nontrivial failure profile"
        )
    case_order = [case.case_id for case in frozen_release_banks(campaign)[stage]]
    if failures != [case_id for case_id in case_order if case_id in set(failures)]:
        raise ValueError("source-behavior failure ids differ from the frozen bank")
    successes = stage_reference.get("source_successes")
    if successes != BANK_CASES - len(failures):
        raise ValueError("source-behavior reference success count is inconsistent")
    manifest_sha256 = stage_reference.get("source_evidence_manifest_sha256")
    if (
        not isinstance(manifest_sha256, str)
        or len(manifest_sha256) != 64
        or manifest_sha256 != manifest_sha256.lower()
    ):
        raise ValueError("source-behavior reference lacks content-addressed evidence")
    stage_cost = stage_reference.get("cost")
    if not isinstance(stage_cost, dict):
        raise TypeError("source-behavior stage cost is missing")
    parsed_stage_cost = InteractionCost.from_dict(stage_cost)
    if (
        parsed_stage_cost.world_rollouts != 32
        or parsed_stage_cost.requested_simulator_steps != 8_000
        or parsed_stage_cost.candidate_evaluations != 0
    ):
        raise ValueError("source-behavior stage cost is not exact")
    reference_id = reference.get("reference_id")
    if not isinstance(reference_id, str) or not reference_id:
        raise ValueError("source-behavior reference id is missing")
    return f"{reference_id}:{stage}", tuple(failures)


def validate_complete_paired_bank(
    *,
    manifest: dict[str, Any],
    manifest_path: Path,
    stage: Literal["production_runtime", "independent_confirmation"],
    source_sha256: str,
    adapted_sha256: str,
    campaign: PatchCampaign | None = None,
    source_behavior_reference_path: Path | None = None,
) -> tuple[dict[str, Any], InteractionCost]:
    """Require platform-sealed source behavior, repair, retention, and parity."""

    report = evaluate_paired_non_regression(
        manifest,
        artifact_id="alpha-walking",
        source_sha256=source_sha256,
        adapted_sha256=adapted_sha256,
        profile_role="shifted",
    )
    expected_bank = (
        FIRST_BANK_SHA256
        if campaign is None and stage == "production_runtime"
        else CONFIRMATION_BANK_SHA256
        if campaign is None
        else resolve_walking_protocol(campaign).release_bank(stage).ab_bank_sha256
    )
    if report["bank_sha256"] != expected_bank:
        raise RuntimeError(f"{stage} used the wrong frozen bank")
    reference_id, expected_source_failures = (
        source_behavior_reference(stage)
        if campaign is None
        else _campaign_source_behavior_reference(
            stage=stage,
            campaign=campaign,
            reference_path=source_behavior_reference_path,
        )
    )
    rows = manifest.get("rows")
    assert isinstance(rows, list)
    observed_source_failures = tuple(
        row["case"]["case_id"] for row in rows if not row["source"]["terminal_success"]
    )
    match_mode = (
        "exact_case_ids"
        if campaign is None
        else resolve_walking_protocol(campaign).source_behavior_match_mode
    )
    count_tolerance = (
        0
        if campaign is None
        else resolve_walking_protocol(campaign).source_behavior_failure_count_tolerance
    )
    expected_only = tuple(
        case_id
        for case_id in expected_source_failures
        if case_id not in set(observed_source_failures)
    )
    observed_only = tuple(
        case_id
        for case_id in observed_source_failures
        if case_id not in set(expected_source_failures)
    )
    if match_mode == "exact_case_ids":
        if observed_source_failures != expected_source_failures:
            raise RuntimeError(
                f"{stage} source behavior drifted from {reference_id}: expected "
                f"{len(expected_source_failures)} failures, observed "
                f"{len(observed_source_failures)}"
            )
    elif match_mode == "failure_count_and_paired_casewise":
        if len(observed_source_failures) != len(expected_source_failures):
            raise RuntimeError(
                f"{stage} source failure count drifted from {reference_id}: expected "
                f"{len(expected_source_failures)} failures, observed "
                f"{len(observed_source_failures)}"
            )
    elif match_mode == "bounded_failure_count_and_paired_casewise":
        count_drift = abs(len(observed_source_failures) - len(expected_source_failures))
        if count_drift > count_tolerance:
            raise RuntimeError(
                f"{stage} source failure count drift exceeded tolerance for "
                f"{reference_id}: expected {len(expected_source_failures)} +/- "
                f"{count_tolerance} failures, observed "
                f"{len(observed_source_failures)}"
            )
    else:  # pragma: no cover - frozen protocol construction prevents this.
        raise RuntimeError(f"unknown source behavior match mode {match_mode!r}")
    report = {
        **report,
        "source_behavior_reference_id": reference_id,
        "source_behavior_match_mode": match_mode,
        "source_behavior_failure_count_tolerance": count_tolerance,
        "expected_source_failure_count": len(expected_source_failures),
        "expected_source_failure_case_ids": list(expected_source_failures),
        "source_failure_case_ids": list(observed_source_failures),
        "source_failure_case_identity_drift": {
            "expected_only": list(expected_only),
            "observed_only": list(observed_only),
        },
    }
    if report["source_success_regressions"]:
        raise CandidateRejected(
            f"{stage} lost {report['source_success_regressions']} source successes",
            report,
        )
    if report["adapted_successes"] != EXPECTED_ADAPTED_SUCCESSES:
        raise CandidateRejected(
            f"{stage} repaired only {report['adapted_successes']}/32 cases",
            report,
        )

    for row in rows:
        assert isinstance(row, dict)
        for policy_role in ("source", "adapted"):
            relative = row.get(policy_role, {}).get("manifest")
            if not isinstance(relative, str):
                raise TypeError("paired row does not name its runtime evidence")
            episode = json.loads((manifest_path.parent / relative).read_text())
            if episode.get("runtime_trace_audit", {}).get("status") != "pass":
                raise CandidateRejected(
                    f"{stage} {policy_role} runtime trace did not pass"
                )
            expected_policy = (
                source_sha256 if policy_role == "source" else adapted_sha256
            )
            if episode.get("artifact", {}).get("evaluated_sha256") != expected_policy:
                raise RuntimeError("runtime episode evaluated different policy bytes")
    return report, _interaction_cost(manifest)


def _load_json_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise TypeError(f"{path} must contain one JSON object")
    return value


def _run_checked(argv: list[str], *, cwd: Path) -> subprocess.CompletedProcess[str]:
    completed = subprocess.run(
        argv,
        cwd=cwd,
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode:
        raise RuntimeError(
            f"qualification subprocess failed ({completed.returncode}): "
            f"{completed.stderr[-4000:]}"
        )
    return completed


def _ensure_export(
    *,
    candidate_checkpoint: Path,
    campaign: PatchCampaign,
    runtime_repo: Path,
    candidate_root: Path,
) -> tuple[Path, dict[str, Any]]:
    candidate, output_weight, output_bias = load_candidate_checkpoint(
        candidate_checkpoint,
        campaign=campaign,
    )
    source_path = runtime_repo / "example_policies" / "alpha_walking.onnx"
    source = import_deployed_policy(source_path)
    if source.source_sha256 != campaign.artifact_sha256:
        raise RuntimeError("runtime source policy differs from the campaign")
    output_policy = candidate_root / "adapted_policy.onnx"
    export_path = candidate_root / "export.json"
    if not output_policy.exists():
        candidate_root.mkdir(parents=True, exist_ok=True)
        parity_error = export_adapted_policy(
            source,
            output_weight=output_weight,
            output_bias=output_bias,
            output_path=output_policy,
        )
        _write_json(
            export_path,
            {
                "schema": "eggroll-autopatch-qualification-export-v1",
                "generation": candidate.generation,
                "campaign_sha256": campaign.sha256,
                "source_policy_sha256": source.source_sha256,
                "candidate_checkpoint": str(candidate_checkpoint),
                "candidate_checkpoint_sha256": _sha256_file(candidate_checkpoint),
                "adapted_policy": str(output_policy),
                "adapted_policy_sha256": _sha256_file(output_policy),
                "export_onnx_parity_max_abs_error": parity_error,
                "patch_scope": "final-affine-weight-and-bias",
                "trainable_parameters": int(output_weight.size + output_bias.size),
                "selection_metrics": candidate.metric_map(),
            },
        )
    export = _load_json_object(export_path)
    if export.get("candidate_checkpoint_sha256") != _sha256_file(candidate_checkpoint):
        raise RuntimeError("existing export belongs to another checkpoint")
    if export.get("adapted_policy_sha256") != _sha256_file(output_policy):
        raise RuntimeError("existing adapted policy bytes changed")
    return output_policy, export


def _run_paired_stage(
    *,
    stage: Literal["production_runtime", "independent_confirmation"],
    adapted_policy: Path,
    source_sha256: str,
    release_scope: ReleaseScope,
    campaign: PatchCampaign,
    runtime_repo: Path,
    candidate_root: Path,
    source_behavior_reference_path: Path | None,
) -> tuple[dict[str, Any], InteractionCost]:
    protocol = resolve_walking_protocol(campaign)
    banks = frozen_release_banks(campaign)
    profile_name = dict(release_scope.profile_sha256s).get("shifted")
    expected_profile = protocol.profile
    if profile_name != expected_profile.sha256:
        raise RuntimeError("release scope no longer names the frozen walking profile")

    stage_dir = candidate_root / stage
    manifest_path = stage_dir / "manifest.json"
    if not manifest_path.exists():
        robotd = Path(os.environ["EGGROLL_PRODUCTION_ROBOTD"])
        ort_dylib = Path(os.environ["EGGROLL_PRODUCTION_ORT_DYLIB"])
        if not robotd.is_file() or not ort_dylib.is_file():
            raise FileNotFoundError(
                "production runtime binary or ONNX library is absent"
            )
        run_paired_ab_suite(
            registry=PRODUCTION_REGISTRY,
            artifact_id="alpha-walking",
            adapted_policy=adapted_policy,
            runtime_repo=runtime_repo,
            robotd=robotd,
            ort_dylib=ort_dylib,
            profiles=(("shifted", expected_profile),),
            cases=banks[stage],
            output_dir=stage_dir,
            device=protocol.release_bank(stage).device,
            record_video=False,
            timeout_s=120.0,
            max_attempts=(2 if protocol.condition_family == "wedge_foot" else 1),
        )
    manifest = _load_json_object(manifest_path)
    cost = _interaction_cost(manifest)
    try:
        report, validated_cost = validate_complete_paired_bank(
            manifest=manifest,
            manifest_path=manifest_path,
            stage=stage,
            source_sha256=source_sha256,
            adapted_sha256=_sha256_file(adapted_policy),
            campaign=campaign,
            source_behavior_reference_path=source_behavior_reference_path,
        )
    except CandidateRejected as rejection:
        raise CandidateRejected(
            rejection.reason,
            rejection.evidence,
            cost=cost,
        ) from rejection
    if validated_cost != cost:
        raise RuntimeError("paired runtime validation changed interaction accounting")
    if stage == "independent_confirmation":
        first_manifest = candidate_root / "production_runtime" / "manifest.json"
        combined = build_paired_non_regression_envelope(
            manifests=(first_manifest, manifest_path),
            artifact_id="alpha-walking",
            source_sha256=source_sha256,
            adapted_sha256=_sha256_file(adapted_policy),
            release_scope=release_scope,
        )
        if combined["source_success_regressions"] != 0:
            raise CandidateRejected(
                "combined paired banks contain a regression", combined
            )
        report = {**report, "combined_two_bank_non_regression": combined}
    return report, cost


def _run_profile_routing(
    *,
    adapted_policy: Path,
    source_policy: Path,
    release_scope: ReleaseScope,
    candidate_root: Path,
) -> dict[str, Any]:
    harness = Path(os.environ["EGGROLL_PROFILE_ROUTING_HARNESS"])
    output = candidate_root / "profile_routing" / "routing.json"
    if not output.exists():
        output.parent.mkdir(parents=True, exist_ok=True)
        _run_checked(
            [
                str(harness),
                "--root",
                str(output.parent / "updater"),
                "--source",
                str(source_policy),
                "--adapted",
                str(adapted_policy),
                "--component",
                "model-walk",
                "--artifact-id",
                "alpha-walking",
                "--activation-profile-sha256",
                dict(release_scope.profile_sha256s)["shifted"],
                "--release-scope-sha256",
                release_scope.sha256,
                "--output",
                str(output),
            ],
            cwd=harness.parent,
        )
    routing = _load_json_object(output)
    validate_routing_evidence(
        routing,
        artifact_id="alpha-walking",
        adapted_sha256=_sha256_file(adapted_policy),
        release_scope=release_scope,
    )
    proof = routing.get("signed_updater_proof")
    if not isinstance(proof, dict) or proof.get("status") != "pass":
        raise CandidateRejected("profile routing did not pass the signed updater proof")
    return routing


def _run_signed_activation_and_rollback(
    *,
    adapted_policy: Path,
    source_policy: Path,
    candidate_root: Path,
) -> dict[str, Any]:
    harness = Path(os.environ["EGGROLL_SIGNED_UPDATER_HARNESS"])
    root = candidate_root / "signed_activation_and_rollback" / "updater"
    activation_path = root / "activation.json"
    rollback_path = root / "rollback.json"
    if not activation_path.exists():
        root.parent.mkdir(parents=True, exist_ok=True)
        _run_checked(
            [
                str(harness),
                "activate",
                "--root",
                str(root),
                "--source",
                str(source_policy),
                "--adapted",
                str(adapted_policy),
                "--component",
                "model-walk",
            ],
            cwd=harness.parent,
        )
    if not rollback_path.exists():
        _run_checked(
            [str(harness), "rollback", "--root", str(root)],
            cwd=harness.parent,
        )
    activation = _load_json_object(activation_path)
    rollback = _load_json_object(rollback_path)
    source_sha256 = _sha256_file(source_policy)
    adapted_sha256 = _sha256_file(adapted_policy)
    passed = (
        activation.get("schema") == "eggroll-autopatch-updater-activation-v2"
        and activation.get("signature_and_artifact_verification")
        == "passed_by_real_engine"
        and activation.get("source", {}).get("active_sha256") == source_sha256
        and activation.get("adapted", {}).get("active_sha256") == adapted_sha256
        and int(activation.get("health_gate_calls", 0)) >= 2
        and int(activation.get("model_api_calls", 0)) >= 2
        and rollback.get("schema") == "eggroll-autopatch-updater-rollback-v2"
        and rollback.get("before_sha256") == adapted_sha256
        and rollback.get("after_sha256") == source_sha256
        and rollback.get("exact_source_restored") is True
    )
    evidence = {
        "status": "pass" if passed else "fail",
        "activation": activation,
        "rollback": rollback,
        "activation_sha256": _sha256_file(activation_path),
        "rollback_sha256": _sha256_file(rollback_path),
        "fresh_process_rollback": True,
    }
    if not passed:
        raise CandidateRejected(
            "signed activation or exact-source rollback failed", evidence
        )
    return evidence


def run_qualification_stage(
    *,
    stage: Stage,
    candidate_checkpoint: Path,
    checkpoint_sha256: str,
    generation: int,
    evidence_directory: Path,
    result_path: Path,
    campaign_path: Path,
    release_scope_path: Path,
    runtime_repo: Path,
    source_behavior_reference_path: Path | None = None,
) -> dict[str, Any]:
    """Execute one frozen stage and write its command-backend result manifest."""

    if stage not in DEFAULT_QUALIFICATION_STAGES:
        raise ValueError(f"unknown qualification stage {stage!r}")
    if generation <= 0:
        raise ValueError("qualification generation must be positive")
    evidence_directory = evidence_directory.resolve()
    result_path = result_path.resolve()
    if not result_path.is_relative_to(evidence_directory):
        raise ValueError("qualification result escaped the evidence directory")
    if result_path.exists():
        raise FileExistsError(result_path)
    candidate_checkpoint = candidate_checkpoint.resolve()
    if _sha256_file(candidate_checkpoint) != checkpoint_sha256:
        raise RuntimeError("candidate checkpoint hash does not match the controller")
    campaign = PatchCampaign.from_json(campaign_path.read_text())
    release_scope = ReleaseScope.from_json(release_scope_path.read_text())
    if release_scope.source_fallback_sha256 != campaign.artifact_sha256:
        raise RuntimeError("release scope fallback differs from the campaign source")
    candidate_root = result_path.parent
    adapted_policy, export = _ensure_export(
        candidate_checkpoint=candidate_checkpoint,
        campaign=campaign,
        runtime_repo=runtime_repo,
        candidate_root=candidate_root,
    )
    if export.get("generation") != generation:
        raise RuntimeError("candidate generation does not match the command")
    source_policy = runtime_repo / "example_policies" / "alpha_walking.onnx"
    cost = InteractionCost()
    evidence: dict[str, Any]
    reason: str
    status: Literal["pass", "fail"] = "pass"
    try:
        if stage == "release_scope_retention":
            metrics = export.get("selection_metrics")
            if not isinstance(metrics, dict):
                raise RuntimeError("candidate export has no selection metrics")
            retained = float(metrics.get("retained_source_success_rate", -1.0))
            repaired = float(metrics.get("repaired_source_failure_rate", -1.0))
            evidence = {
                "retained_source_success_rate": retained,
                "repaired_source_failure_rate": repaired,
                "release_scope_sha256": release_scope.sha256,
                "candidate_export_sha256": _sha256_file(candidate_root / "export.json"),
            }
            if retained != 1.0:
                raise CandidateRejected(
                    f"campaign-side source-success retention was {retained:.6g}, not 1",
                    evidence,
                )
            if repaired <= 0.0:
                raise CandidateRejected(
                    "campaign-side candidate repaired no source failure", evidence
                )
            reason = "campaign-side release-scope retention and repair screen passed"
        elif stage == "onnx_parity":
            source = import_deployed_policy(source_policy)
            adapted = import_deployed_policy(adapted_policy)
            verify_output_layer_derivative(source=source, adapted=adapted)
            parity_error = runtime_parity(adapted, seed=20262024)
            evidence = {
                "source_policy_sha256": source.source_sha256,
                "adapted_policy_sha256": adapted.source_sha256,
                "max_abs_error": parity_error,
                "threshold_exclusive": 1.0e-5,
                "fixtures": 64,
                "patch_scope": "final-affine-weight-and-bias",
            }
            if parity_error >= 1.0e-5:
                raise CandidateRejected(
                    f"independent ONNX parity error {parity_error:.9g} exceeds 1e-5",
                    evidence,
                )
            reason = "independent ONNX parity and output-layer-only proof passed"
        elif stage in ("production_runtime", "independent_confirmation"):
            evidence, cost = _run_paired_stage(
                stage=stage,
                adapted_policy=adapted_policy,
                source_sha256=campaign.artifact_sha256,
                release_scope=release_scope,
                campaign=campaign,
                runtime_repo=runtime_repo,
                candidate_root=candidate_root,
                source_behavior_reference_path=source_behavior_reference_path,
            )
            reason = (
                f"{stage} passed all 32 candidate cases, retained every source "
                "success, and passed every production Rust trace"
            )
        elif stage == "profile_routing":
            evidence = _run_profile_routing(
                adapted_policy=adapted_policy,
                source_policy=source_policy,
                release_scope=release_scope,
                candidate_root=candidate_root,
            )
            reason = "attested profile activated adapted bytes and unknown profile retained source"
        else:
            evidence = _run_signed_activation_and_rollback(
                adapted_policy=adapted_policy,
                source_policy=source_policy,
                candidate_root=candidate_root,
            )
            reason = "real updater verified signatures, activated adapted bytes, and restored exact source bytes in a fresh rollback process"
    except CandidateRejected as rejection:
        status = "fail"
        reason = rejection.reason
        evidence = rejection.evidence
        cost = rejection.cost

    payload = {
        "schema": RESULT_SCHEMA,
        "stage": stage,
        "generation": generation,
        "checkpoint_sha256": checkpoint_sha256,
        "adapted_policy_sha256": _sha256_file(adapted_policy),
        "status": status,
        "reason": reason,
        "cost": cost.to_dict(),
        "evidence": evidence,
    }
    _write_json(result_path, payload)
    return payload
