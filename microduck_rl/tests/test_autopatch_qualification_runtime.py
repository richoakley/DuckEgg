from __future__ import annotations

import hashlib
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pytest

from mjlab_microduck.autopatch.campaign import save_candidate_checkpoint
from mjlab_microduck.autopatch.contracts import PatchCampaign, ReleaseScope
from mjlab_microduck.autopatch.foot_proof import (
    make_walking_proof_bank,
    walking_bank_sha256,
)
from mjlab_microduck.autopatch.qualification_runtime import (
    BANK_CASES,
    CONFIRMATION_BANK_SHA256,
    FIRST_BANK_SHA256,
    HISTORICAL_STAGE_DEVICES,
    CandidateRejected,
    _campaign_source_behavior_reference,
    _interaction_cost,
    _run_paired_stage,
    frozen_release_banks,
    run_qualification_stage,
    source_behavior_reference,
    validate_complete_paired_bank,
)
from mjlab_microduck.autopatch.walking_protocol import (
    walking_campaign_family_sha256,
)
from mjlab_microduck.eggroll.deployment import TRUNK_COM_CALIBRATION_PROFILES
from mjlab_microduck.eggroll.policy_io import import_deployed_policy

ROOT = Path(__file__).resolve().parents[1]
RUNTIME_REPO = ROOT.parent / "microduck"
CAMPAIGN_PATH = (
    ROOT / "docs/experiments/campaigns/walking_wedge_autopatch_efficiency_seed1_v1.json"
)
RELEASE_SCOPE_PATH = (
    ROOT
    / "docs/experiments/release_scopes/walking_wedge_gen85_profile_specific_v1.json"
)
PROFILE_SHA256 = "3410b59527e069c993212671ce463ac05183777968a1ed8e15872affb46912a2"


def _com_campaign() -> PatchCampaign:
    document = PatchCampaign.from_json(CAMPAIGN_PATH.read_text()).canonical_dict()
    profile = TRUNK_COM_CALIBRATION_PROFILES[2]
    document["campaign_id"] = "alpha-walking-trunk-com-reference-test-v1"
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


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_frozen_release_banks_match_historical_hashes_and_are_disjoint() -> None:
    banks = frozen_release_banks()
    assert len(banks["production_runtime"]) == BANK_CASES
    assert len(banks["independent_confirmation"]) == BANK_CASES
    assert (
        hashlib.sha256(
            json.dumps(
                [asdict(case) for case in banks["production_runtime"]],
                sort_keys=True,
                separators=(",", ":"),
            ).encode()
        ).hexdigest()
        == FIRST_BANK_SHA256
    )
    assert (
        hashlib.sha256(
            json.dumps(
                [asdict(case) for case in banks["independent_confirmation"]],
                sort_keys=True,
                separators=(",", ":"),
            ).encode()
        ).hexdigest()
        == CONFIRMATION_BANK_SHA256
    )
    assert not (
        {case.seed for case in banks["production_runtime"]}
        & {case.seed for case in banks["independent_confirmation"]}
    )
    assert HISTORICAL_STAGE_DEVICES == {
        "production_runtime": "cpu",
        "independent_confirmation": "cuda:0",
    }


def test_com_source_behavior_reference_is_campaign_and_bank_bound(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setenv("EGGROLL_HF_HARDWARE_FLAVOR", "a10g-large")
    campaign = _com_campaign()
    profile = TRUNK_COM_CALIBRATION_PROFILES[2]
    reference = {
        "schema": "eggroll-autopatch-source-behavior-reference-v1",
        "status": "pass",
        "reference_id": "trunk-com-source-behavior-v1",
        "source_commit": "c" * 40,
        "hf_hardware_flavor": "a10g-large",
        "capture_campaign_sha256": campaign.sha256,
        "campaign_family_sha256": walking_campaign_family_sha256(campaign),
        "walking_protocol_id": "alpha-walking-trunk-com-cross-failure-v2",
        "source_policy_sha256": campaign.artifact_sha256,
        "activation_profile_sha256": profile.sha256,
        "calibration_validation_sha256": "d" * 64,
        "candidate_optimization_evaluations": 0,
        "cost": {
            "candidate_evaluations": 0,
            "world_rollouts": 64,
            "requested_simulator_steps": 16_000,
            "executed_simulator_steps": 16_000,
            "active_interaction_steps": 16_000,
            "policy_forward_rows": 16_000,
            "physics_substeps": 64_000,
            "wall_seconds": 0.0,
            "accelerator_seconds": 0.0,
            "world_constructions": 64,
        },
        "stages": {
            "production_runtime": {
                "device": "cuda:0",
                "ab_bank_sha256": (
                    "c71c54790a6e26782a5b27449600648230ae02bc1fba60392d2b13ec56017547"
                ),
                "source_successes": 30,
                "source_failure_case_ids": [
                    "production-trunk-com-vx-0.28-000",
                    "production-trunk-com-vx-0.28-001",
                ],
                "source_evidence_manifest_sha256": "a" * 64,
            },
            "independent_confirmation": {
                "device": "cuda:0",
                "ab_bank_sha256": (
                    "c9e82fa04a658e093db8af8bc6775c04044de0cc5c3154e23fa9a3bbd54fcd33"
                ),
                "source_successes": 31,
                "source_failure_case_ids": ["confirmation-trunk-com-a"],
                "source_evidence_manifest_sha256": "b" * 64,
            },
        },
    }
    reference["stages"]["production_runtime"]["cost"] = {
        **reference["cost"],
        "world_rollouts": 32,
        "requested_simulator_steps": 8_000,
        "executed_simulator_steps": 8_000,
        "active_interaction_steps": 8_000,
        "policy_forward_rows": 8_000,
        "physics_substeps": 32_000,
        "world_constructions": 32,
    }
    path = tmp_path / "reference.json"
    path.write_text(json.dumps(reference))
    reference_id, failures = _campaign_source_behavior_reference(
        stage="production_runtime",
        campaign=campaign,
        reference_path=path,
    )
    assert reference_id.endswith(":production_runtime")
    assert failures == (
        "production-trunk-com-vx-0.28-000",
        "production-trunk-com-vx-0.28-001",
    )

    monkeypatch.setenv("EGGROLL_HF_HARDWARE_FLAVOR", "rtx-pro-6000")
    with pytest.raises(RuntimeError, match="qualification hardware differs"):
        _campaign_source_behavior_reference(
            stage="production_runtime",
            campaign=campaign,
            reference_path=path,
        )
    monkeypatch.setenv("EGGROLL_HF_HARDWARE_FLAVOR", "a10g-large")

    reference["stages"]["production_runtime"]["ab_bank_sha256"] = "f" * 64
    path.write_text(json.dumps(reference))
    with pytest.raises(ValueError, match="identity changed"):
        _campaign_source_behavior_reference(
            stage="production_runtime",
            campaign=campaign,
            reference_path=path,
        )


def test_runtime_accounting_accepts_integral_json_floats() -> None:
    rows = [
        {
            "source": {"result": {"episode_steps": 250.0}},
            "adapted": {"result": {"episode_steps": 125.0}},
        }
        for _ in range(BANK_CASES)
    ]
    cost = _interaction_cost({"rows": rows})
    assert cost.world_rollouts == 64
    assert cost.requested_simulator_steps == 16_000
    assert cost.executed_simulator_steps == 12_000


@pytest.mark.parametrize("invalid", [0.0, -1.0, 1.5, float("nan"), True])
def test_runtime_accounting_rejects_invalid_steps(invalid: object) -> None:
    rows = [
        {
            "source": {"result": {"episode_steps": 250.0}},
            "adapted": {"result": {"episode_steps": 250.0}},
        }
        for _ in range(BANK_CASES)
    ]
    rows[0]["source"]["result"]["episode_steps"] = invalid
    with pytest.raises(ValueError, match="invalid executed steps"):
        _interaction_cost({"rows": rows})


def _paired_manifest(
    tmp_path: Path,
    *,
    stage: str = "production_runtime",
    regress: bool = False,
    source_successes: int | None = None,
    source_sha: str = "a" * 64,
    adapted_sha: str = "b" * 64,
) -> tuple[dict[str, object], Path]:
    cases = frozen_release_banks()[stage]
    _reference_id, expected_failures = source_behavior_reference(stage)
    expected_failure_set = set(expected_failures)
    rows = []
    stage_dir = tmp_path / stage
    for index, case in enumerate(cases):
        source_passed = (
            case.case_id not in expected_failure_set
            if source_successes is None
            else index < source_successes
        )
        adapted_passed = not (regress and index == 1)
        pair: dict[str, object] = {}
        for role, policy_sha, terminal_success in (
            ("source", source_sha, source_passed),
            ("adapted", adapted_sha, adapted_passed),
        ):
            relative = Path("episodes") / case.case_id / f"{role}.json"
            episode = stage_dir / relative
            episode.parent.mkdir(parents=True, exist_ok=True)
            episode.write_text(
                json.dumps(
                    {
                        "artifact": {"evaluated_sha256": policy_sha},
                        "runtime_trace_audit": {"status": "pass"},
                    }
                )
            )
            pair[role] = {
                "manifest": str(relative),
                "policy_sha256": policy_sha,
                "terminal_success": terminal_success,
                "result": {
                    "terminal_success": terminal_success,
                    "episode_steps": 250,
                },
            }
        rows.append(
            {
                "profile_role": "shifted",
                "profile_sha256": PROFILE_SHA256,
                "case": asdict(case),
                **pair,
            }
        )
    manifest: dict[str, object] = {
        "schema": "eggroll-autopatch-paired-ab-v1",
        "artifact_id": "alpha-walking",
        "source_sha256": source_sha,
        "adapted_sha256": adapted_sha,
        "paired_bank": [asdict(case) for case in cases],
        "rows": rows,
    }
    manifest_path = stage_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest))
    return manifest, manifest_path


def test_complete_paired_bank_requires_sealed_source_behavior_and_retention(
    tmp_path: Path,
) -> None:
    manifest, path = _paired_manifest(tmp_path)
    report, cost = validate_complete_paired_bank(
        manifest=manifest,
        manifest_path=path,
        stage="production_runtime",
        source_sha256="a" * 64,
        adapted_sha256="b" * 64,
    )
    _reference_id, expected_failures = source_behavior_reference("production_runtime")
    assert report["source_successes"] == BANK_CASES - len(expected_failures)
    assert report["adapted_successes"] == 32
    assert report["source_success_regressions"] == 0
    assert cost.world_rollouts == 64
    assert cost.requested_simulator_steps == 16_000
    assert cost.executed_simulator_steps == 16_000
    assert cost.physics_substeps == 64_000

    drifted_cases = json.loads(json.dumps(manifest))
    failed_index = next(
        index
        for index, row in enumerate(drifted_cases["rows"])
        if not row["source"]["terminal_success"]
    )
    passed_index = next(
        index
        for index, row in enumerate(drifted_cases["rows"])
        if row["source"]["terminal_success"]
    )
    for index, terminal_success in ((failed_index, True), (passed_index, False)):
        drifted_cases["rows"][index]["source"]["terminal_success"] = terminal_success
        drifted_cases["rows"][index]["source"]["result"]["terminal_success"] = (
            terminal_success
        )
    with pytest.raises(RuntimeError, match="source behavior drifted"):
        validate_complete_paired_bank(
            manifest=drifted_cases,
            manifest_path=path,
            stage="production_runtime",
            source_sha256="a" * 64,
            adapted_sha256="b" * 64,
        )

    regressive, regressive_path = _paired_manifest(
        tmp_path / "regressive", regress=True
    )
    with pytest.raises(CandidateRejected, match="lost 1 source successes") as error:
        validate_complete_paired_bank(
            manifest=regressive,
            manifest_path=regressive_path,
            stage="production_runtime",
            source_sha256="a" * 64,
            adapted_sha256="b" * 64,
        )
    assert error.value.cost.requested_simulator_steps == 0


def test_versioned_wedge_accepts_case_swap_but_not_failure_count_drift(
    tmp_path: Path,
) -> None:
    campaign_document = PatchCampaign.from_json(
        CAMPAIGN_PATH.read_text()
    ).canonical_dict()
    campaign_document["condition"]["adapter"] = (
        "mjlab-wedge-foot-paired-source-profile-v2"
    )
    campaign = PatchCampaign.from_dict(campaign_document)
    manifest, path = _paired_manifest(tmp_path, source_successes=21)
    rows = manifest["rows"]
    assert isinstance(rows, list)
    failed_index = next(
        index for index, row in enumerate(rows) if not row["source"]["terminal_success"]
    )
    passed_index = next(
        index for index, row in enumerate(rows) if row["source"]["terminal_success"]
    )
    for index, terminal_success in ((failed_index, True), (passed_index, False)):
        rows[index]["source"]["terminal_success"] = terminal_success
        rows[index]["source"]["result"]["terminal_success"] = terminal_success

    report, _cost = validate_complete_paired_bank(
        manifest=manifest,
        manifest_path=path,
        stage="production_runtime",
        source_sha256="a" * 64,
        adapted_sha256="b" * 64,
        campaign=campaign,
    )
    assert report["source_behavior_match_mode"] == ("failure_count_and_paired_casewise")
    assert report["source_failure_case_identity_drift"]["expected_only"]
    assert report["source_failure_case_identity_drift"]["observed_only"]

    rows[failed_index]["source"]["terminal_success"] = False
    rows[failed_index]["source"]["result"]["terminal_success"] = False
    with pytest.raises(RuntimeError, match="source failure count drifted"):
        validate_complete_paired_bank(
            manifest=manifest,
            manifest_path=path,
            stage="production_runtime",
            source_sha256="a" * 64,
            adapted_sha256="b" * 64,
            campaign=campaign,
        )


def test_bounded_wedge_accepts_one_count_drift_but_not_two(tmp_path: Path) -> None:
    campaign_document = PatchCampaign.from_json(
        CAMPAIGN_PATH.read_text()
    ).canonical_dict()
    campaign_document["condition"]["adapter"] = (
        "mjlab-wedge-foot-paired-source-profile-v3"
    )
    campaign = PatchCampaign.from_dict(campaign_document)
    manifest, path = _paired_manifest(tmp_path, source_successes=21)
    rows = manifest["rows"]
    assert isinstance(rows, list)
    failed_indices = [
        index for index, row in enumerate(rows) if not row["source"]["terminal_success"]
    ]
    for index in failed_indices[:1]:
        rows[index]["source"]["terminal_success"] = True
        rows[index]["source"]["result"]["terminal_success"] = True

    report, _cost = validate_complete_paired_bank(
        manifest=manifest,
        manifest_path=path,
        stage="production_runtime",
        source_sha256="a" * 64,
        adapted_sha256="b" * 64,
        campaign=campaign,
    )
    assert report["source_behavior_match_mode"] == (
        "bounded_failure_count_and_paired_casewise"
    )
    assert report["source_behavior_failure_count_tolerance"] == 1
    assert report["source_successes"] == 22

    index = failed_indices[1]
    rows[index]["source"]["terminal_success"] = True
    rows[index]["source"]["result"]["terminal_success"] = True
    with pytest.raises(RuntimeError, match="count drift exceeded tolerance"):
        validate_complete_paired_bank(
            manifest=manifest,
            manifest_path=path,
            stage="production_runtime",
            source_sha256="a" * 64,
            adapted_sha256="b" * 64,
            campaign=campaign,
        )


def test_paired_stage_bills_complete_bank_when_candidate_is_rejected(
    tmp_path: Path,
) -> None:
    adapted = tmp_path / "adapted.onnx"
    adapted.write_bytes(b"candidate policy bytes")
    adapted_sha = _sha256(adapted)
    campaign = PatchCampaign.from_json(CAMPAIGN_PATH.read_text())
    candidate_root = tmp_path / "candidate"
    _paired_manifest(
        candidate_root,
        regress=True,
        source_sha=campaign.artifact_sha256,
        adapted_sha=adapted_sha,
    )
    release_scope = ReleaseScope.from_json(RELEASE_SCOPE_PATH.read_text())
    with pytest.raises(CandidateRejected, match="lost 1 source successes") as error:
        _run_paired_stage(
            stage="production_runtime",
            adapted_policy=adapted,
            source_sha256=campaign.artifact_sha256,
            release_scope=release_scope,
            campaign=campaign,
            runtime_repo=RUNTIME_REPO,
            candidate_root=candidate_root,
            source_behavior_reference_path=None,
        )
    assert error.value.cost.world_rollouts == 64
    assert error.value.cost.requested_simulator_steps == 16_000

    drifted, drifted_path = _paired_manifest(tmp_path / "drifted", source_successes=25)
    with pytest.raises(RuntimeError, match="source behavior drifted"):
        validate_complete_paired_bank(
            manifest=drifted,
            manifest_path=drifted_path,
            stage="production_runtime",
            source_sha256="a" * 64,
            adapted_sha256="b" * 64,
        )


def _candidate(tmp_path: Path, *, retained: float = 1.0) -> tuple[Path, str]:
    campaign = PatchCampaign.from_json(CAMPAIGN_PATH.read_text())
    source = import_deployed_policy(
        RUNTIME_REPO / "example_policies" / "alpha_walking.onnx"
    )
    weight = np.array(source.output_weight, copy=True)
    weight[0, 0] += np.float32(1.0e-4)
    path = tmp_path / "generation-000001.npz"
    save_candidate_checkpoint(
        path,
        campaign=campaign,
        generation=1,
        output_weight=weight,
        output_bias=source.output_bias,
        metrics={
            "retained_source_success_rate": retained,
            "repaired_source_failure_rate": 0.5,
            "min_command_success_rate": 0.75,
            "terminal_success_rate": 0.875,
        },
    )
    return path, _sha256(path)


def test_candidate_stage_exports_exact_derivative_and_runs_independent_parity(
    tmp_path: Path,
) -> None:
    checkpoint, checkpoint_sha = _candidate(tmp_path)
    evidence = tmp_path / "evidence"
    retention = run_qualification_stage(
        stage="release_scope_retention",
        candidate_checkpoint=checkpoint,
        checkpoint_sha256=checkpoint_sha,
        generation=1,
        evidence_directory=evidence,
        result_path=evidence / "generation-1/release_scope_retention.json",
        campaign_path=CAMPAIGN_PATH,
        release_scope_path=RELEASE_SCOPE_PATH,
        runtime_repo=RUNTIME_REPO,
    )
    assert retention["status"] == "pass"
    assert retention["checkpoint_sha256"] == checkpoint_sha
    assert retention["evidence"]["retained_source_success_rate"] == 1.0
    adapted = evidence / "generation-1/adapted_policy.onnx"
    assert _sha256(adapted) == retention["adapted_policy_sha256"]

    parity = run_qualification_stage(
        stage="onnx_parity",
        candidate_checkpoint=checkpoint,
        checkpoint_sha256=checkpoint_sha,
        generation=1,
        evidence_directory=evidence,
        result_path=evidence / "generation-1/onnx_parity.json",
        campaign_path=CAMPAIGN_PATH,
        release_scope_path=RELEASE_SCOPE_PATH,
        runtime_repo=RUNTIME_REPO,
    )
    assert parity["status"] == "pass"
    assert parity["evidence"]["max_abs_error"] < 1.0e-5
    assert parity["evidence"]["patch_scope"] == "final-affine-weight-and-bias"


def test_candidate_stage_records_retention_failure_without_losing_identity(
    tmp_path: Path,
) -> None:
    checkpoint, checkpoint_sha = _candidate(tmp_path, retained=0.99)
    evidence = tmp_path / "evidence"
    result = run_qualification_stage(
        stage="release_scope_retention",
        candidate_checkpoint=checkpoint,
        checkpoint_sha256=checkpoint_sha,
        generation=1,
        evidence_directory=evidence,
        result_path=evidence / "generation-1/release_scope_retention.json",
        campaign_path=CAMPAIGN_PATH,
        release_scope_path=RELEASE_SCOPE_PATH,
        runtime_repo=RUNTIME_REPO,
    )
    assert result["status"] == "fail"
    assert result["checkpoint_sha256"] == checkpoint_sha
    assert "not 1" in result["reason"]
