from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest

from mjlab_microduck.autopatch.campaign import (
    CandidateResult,
    save_candidate_checkpoint,
    select_and_export_candidate,
    select_checkpoint,
)
from mjlab_microduck.autopatch.contracts import (
    DeploymentCondition,
    ObjectiveSpec,
    OptimizerSpec,
    PatchCampaign,
    ReleaseGate,
    ReleaseScope,
)
from mjlab_microduck.autopatch.protocols import (
    ProtocolBudget,
    equal_budget_plan,
    run_equal_budget_protocols,
)
from mjlab_microduck.autopatch.registry import PRODUCTION_REGISTRY
from mjlab_microduck.autopatch.release import (
    GateSeries,
    build_paired_non_regression_envelope,
    build_release_envelope,
    evaluate_paired_non_regression,
    evaluate_release_gates,
    validate_routing_evidence,
)
from mjlab_microduck.eggroll.policy_io import (
    export_adapted_policy,
    import_deployed_policy,
)


def campaign() -> PatchCampaign:
    artifact = PRODUCTION_REGISTRY.artifact("alpha-walking")
    return PatchCampaign(
        campaign_id="walk-test",
        artifact_id=artifact.artifact_id,
        artifact_sha256=artifact.expected_sha256,
        capability_id=artifact.capability_id,
        condition=DeploymentCondition(
            "new-foot-v1",
            "model-mutation",
            (("foot_length_scale", 1.2),),
            True,
            "A reversible geometry change.",
        ),
        objective=ObjectiveSpec(
            "tracking-terminal-v1",
            "actual-environment",
            ("terminal_success", "tracking_score"),
            ("task_return",),
            "Terminal validity before tracking quality.",
        ),
        optimizer=OptimizerSpec(
            "eggroll", "output-layer-low-rank", 16, 256, 0.01, 0.01, 100, 7
        ),
        gates=(
            ReleaseGate(
                "target-success",
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


def profile_specific_scope(
    source_sha256: str, *, role: str = "shifted", profile_sha256: str = "f" * 64
) -> ReleaseScope:
    return ReleaseScope(
        scope_id="new-foot-profile-v1",
        mode="profile_specific",
        profile_sha256s=((role, profile_sha256),),
        required_retention_roles=(role,),
        activation_profile_role=role,
        activation_predicate="hardware.feet == wedge-15deg-v1",
        source_fallback_sha256=source_sha256,
        unknown_profile_action="retain_source",
    )


def test_checkpoint_selection_is_objective_lexicographic_not_task_return() -> None:
    selected = select_checkpoint(
        campaign(),
        (
            CandidateResult(
                "high-return.pkl",
                10,
                (
                    ("terminal_success", 0.0),
                    ("tracking_score", 100.0),
                    ("task_return", 999.0),
                ),
            ),
            CandidateResult(
                "valid.pkl",
                20,
                (
                    ("terminal_success", 1.0),
                    ("tracking_score", 0.1),
                    ("task_return", -50.0),
                ),
            ),
        ),
    )
    assert selected.checkpoint == "valid.pkl"


def test_equal_budget_comparison_is_predeclared_and_enforced() -> None:
    budget = ProtocolBudget(256, 100, 4)
    plan = equal_budget_plan(budget=budget, seed=12)
    result = run_equal_budget_protocols(
        plan,
        runner=lambda run: {
            "protocol": run.protocol,
            "evaluations": run.budget.world_rollouts,
        },
    )
    assert result["world_rollouts_per_protocol"] == 102_400
    broken = (plan[0], replace(plan[1], budget=ProtocolBudget(128, 100, 4)), plan[2])
    with pytest.raises(ValueError, match="budget matched"):
        run_equal_budget_protocols(broken, runner=lambda _run: {})


def test_release_gate_requires_final_consecutive_passes() -> None:
    gates = campaign().gates
    failed = evaluate_release_gates(
        gates, (GateSeries("target-success", (0.9, 0.7, 0.9)),)
    )
    assert failed[0]["status"] == "fail"
    passed = evaluate_release_gates(
        gates, (GateSeries("target-success", (0.7, 0.9, 0.8)),)
    )
    assert passed[0]["status"] == "pass"


def test_missing_checkpoint_metric_is_hard_failure() -> None:
    with pytest.raises(ValueError, match="missing metrics"):
        select_checkpoint(
            campaign(),
            (CandidateResult("bad.pkl", 1, (("terminal_success", 1.0),)),),
        )


def test_generic_release_envelope_binds_bytes_nodes_edges_and_gates(
    tmp_path,
) -> None:
    runtime_repo = Path(__file__).resolve().parents[2] / "microduck"
    source_path = runtime_repo / "example_policies" / "alpha_walking.onnx"
    source = import_deployed_policy(source_path)
    adapted_path = tmp_path / "adapted.onnx"
    export_adapted_policy(
        source,
        output_weight=source.output_weight,
        output_bias=source.output_bias + 1.0e-4,
        output_path=adapted_path,
    )
    adapted = import_deployed_policy(adapted_path)
    probe = tmp_path / "runtime_probe.json"
    probe.write_text(
        json.dumps(
            {
                "passed": True,
                "policy_sha256": adapted.source_sha256,
                "production_path": "duck_control::policy::Policy::load",
            }
        )
    )
    node = tmp_path / "node.json"
    node.write_text(
        json.dumps(
            {
                "artifact": {"evaluated_sha256": adapted.source_sha256},
                "runtime_trace_audit": {"status": "pass"},
                "result": {"terminal_success": True},
            }
        )
    )
    plan = PRODUCTION_REGISTRY.release_test_plan("alpha-walking")
    paired_paths = []
    for index in range(2):
        paired = tmp_path / f"paired-{index}.json"
        case = {"case_id": f"nominal-{index:03d}", "seed": index + 1}
        paired.write_text(
            json.dumps(
                {
                    "schema": "eggroll-autopatch-paired-ab-v1",
                    "artifact_id": "alpha-walking",
                    "source_sha256": source.source_sha256,
                    "adapted_sha256": adapted.source_sha256,
                    "paired_bank": [case],
                    "rows": [
                        {
                            "profile_role": "nominal",
                            "profile_sha256": "f" * 64,
                            "case": case,
                            "source": {
                                "policy_sha256": source.source_sha256,
                                "terminal_success": True,
                            },
                            "adapted": {
                                "policy_sha256": adapted.source_sha256,
                                "terminal_success": True,
                            },
                        }
                    ],
                }
            )
        )
        paired_paths.append(paired)
    release_scope = profile_specific_scope(source.source_sha256, role="nominal")
    routing = tmp_path / "routing.json"
    routing.write_text(
        json.dumps(
            {
                "schema": "eggroll-autopatch-routing-attestation-v1",
                "status": "pass",
                "artifact_id": "alpha-walking",
                "adapted_sha256": adapted.source_sha256,
                "release_scope_sha256": release_scope.sha256,
                "source_fallback_sha256": source.source_sha256,
                "unknown_profile_action": "retain_source",
                "production_path": "updaterd::profile_scoped_model_activation",
                "routes": [
                    {
                        "profile_sha256": "f" * 64,
                        "decision": "activate_adapted",
                        "selected_policy_sha256": adapted.source_sha256,
                    },
                    {
                        "profile_sha256": "unknown",
                        "decision": "retain_source",
                        "selected_policy_sha256": source.source_sha256,
                    },
                ],
            }
        )
    )
    envelope = build_release_envelope(
        campaign=campaign(),
        registry=PRODUCTION_REGISTRY,
        source_policy=source_path,
        adapted_policy=adapted_path,
        runtime_probe=probe,
        node_evidence=(node,),
        covered_transition_sha256=tuple(
            edge["transition_sha256"] for edge in plan["edges"]
        ),
        gate_series=(GateSeries("target-success", (0.9, 0.8)),),
        paired_evidence=tuple(paired_paths),
        release_scope=release_scope,
        routing_evidence=routing,
    )
    assert envelope["status"] == "eligible-for-signed-updater-packaging"
    assert envelope["updater_component"] == "model-walk"
    assert envelope["paired_non_regression"]["source_success_regressions"] == 0
    with pytest.raises(ValueError, match="not independent"):
        build_paired_non_regression_envelope(
            manifests=(paired_paths[0], paired_paths[0]),
            artifact_id="alpha-walking",
            source_sha256=source.source_sha256,
            adapted_sha256=adapted.source_sha256,
            release_scope=release_scope,
        )


def test_paired_non_regression_rejects_lost_source_success() -> None:
    source_sha = "a" * 64
    adapted_sha = "b" * 64
    manifest = {
        "schema": "eggroll-autopatch-paired-ab-v1",
        "artifact_id": "alpha-walking",
        "source_sha256": source_sha,
        "adapted_sha256": adapted_sha,
        "paired_bank": [{"case_id": "lost-capability", "seed": 7}],
        "rows": [
            {
                "profile_role": "nominal",
                "profile_sha256": "c" * 64,
                "case": {"case_id": "lost-capability", "seed": 7},
                "source": {
                    "policy_sha256": source_sha,
                    "terminal_success": True,
                },
                "adapted": {
                    "policy_sha256": adapted_sha,
                    "terminal_success": False,
                },
            }
        ],
    }
    report = evaluate_paired_non_regression(
        manifest,
        artifact_id="alpha-walking",
        source_sha256=source_sha,
        adapted_sha256=adapted_sha,
    )
    assert report["source_success_regressions"] == 1
    assert report["regression_case_ids"] == ["lost-capability"]
    assert report["zero_observed_regressions"] is False
    incomplete = {
        **manifest,
        "paired_bank": [
            *manifest["paired_bank"],
            {"case_id": "missing-row", "seed": 8},
        ],
    }
    with pytest.raises(ValueError, match="completely cover"):
        evaluate_paired_non_regression(
            incomplete,
            artifact_id="alpha-walking",
            source_sha256=source_sha,
            adapted_sha256=adapted_sha,
        )


def test_profile_specific_scope_ignores_cross_profile_diagnostic_failure(
    tmp_path: Path,
) -> None:
    source_sha = "a" * 64
    adapted_sha = "b" * 64
    shifted_sha = "c" * 64
    nominal_sha = "d" * 64
    manifests = []
    for index in range(2):
        case = {"case_id": f"case-{index}", "seed": index + 20}
        rows = []
        for role, profile_sha, adapted_success in (
            ("shifted", shifted_sha, True),
            ("nominal", nominal_sha, False),
        ):
            rows.append(
                {
                    "profile_role": role,
                    "profile_sha256": profile_sha,
                    "case": case,
                    "source": {
                        "policy_sha256": source_sha,
                        "terminal_success": True,
                    },
                    "adapted": {
                        "policy_sha256": adapted_sha,
                        "terminal_success": adapted_success,
                    },
                }
            )
        path = tmp_path / f"paired-{index}.json"
        path.write_text(
            json.dumps(
                {
                    "schema": "eggroll-autopatch-paired-ab-v1",
                    "artifact_id": "alpha-walking",
                    "source_sha256": source_sha,
                    "adapted_sha256": adapted_sha,
                    "paired_bank": [case],
                    "rows": rows,
                }
            )
        )
        manifests.append(path)

    scoped = ReleaseScope(
        scope_id="wedge-only",
        mode="profile_specific",
        profile_sha256s=(("shifted", shifted_sha),),
        required_retention_roles=("shifted",),
        activation_profile_role="shifted",
        activation_predicate="hardware.feet == wedge-15deg-v1",
        source_fallback_sha256=source_sha,
        unknown_profile_action="retain_source",
    )
    report = build_paired_non_regression_envelope(
        manifests=tuple(manifests),
        artifact_id="alpha-walking",
        source_sha256=source_sha,
        adapted_sha256=adapted_sha,
        release_scope=scoped,
    )
    assert report["status"] == "pass"
    assert report["retention_profile_roles"] == ["shifted"]
    assert report["evidence"][0]["reports"][0]["adapted_successes"] == 1

    universal = ReleaseScope(
        scope_id="wedge-and-original",
        mode="multi_profile",
        profile_sha256s=(("shifted", shifted_sha), ("nominal", nominal_sha)),
        required_retention_roles=("shifted", "nominal"),
        activation_profile_role=None,
        activation_predicate="profile in signed-release-profile-set",
        source_fallback_sha256=source_sha,
        unknown_profile_action="retain_source",
    )
    with pytest.raises(ValueError, match="source-success regressions"):
        build_paired_non_regression_envelope(
            manifests=tuple(manifests),
            artifact_id="alpha-walking",
            source_sha256=source_sha,
            adapted_sha256=adapted_sha,
            release_scope=universal,
        )


def test_profile_specific_scope_must_fail_closed() -> None:
    with pytest.raises(ValueError, match="retain source"):
        ReleaseScope(
            scope_id="unsafe-wedge",
            mode="profile_specific",
            profile_sha256s=(("shifted", "c" * 64),),
            required_retention_roles=("shifted",),
            activation_profile_role="shifted",
            activation_predicate="hardware.feet == wedge-15deg-v1",
            source_fallback_sha256="a" * 64,
            unknown_profile_action="block_adapted_policy",
        )


def test_routing_attestation_requires_unknown_profile_fallback() -> None:
    scope = profile_specific_scope("a" * 64)
    with pytest.raises(ValueError, match="fail-closed routes"):
        validate_routing_evidence(
            {
                "schema": "eggroll-autopatch-routing-attestation-v1",
                "status": "pass",
                "artifact_id": "alpha-walking",
                "adapted_sha256": "b" * 64,
                "release_scope_sha256": scope.sha256,
                "source_fallback_sha256": "a" * 64,
                "unknown_profile_action": "retain_source",
                "production_path": "updaterd::profile_scoped_model_activation",
                "routes": [
                    {
                        "profile_sha256": "f" * 64,
                        "decision": "activate_adapted",
                        "selected_policy_sha256": "b" * 64,
                    }
                ],
            },
            artifact_id="alpha-walking",
            adapted_sha256="b" * 64,
            release_scope=scope,
        )


def test_campaign_bound_checkpoint_selection_and_export(tmp_path: Path) -> None:
    runtime_repo = Path(__file__).resolve().parents[2] / "microduck"
    source_path = runtime_repo / "example_policies" / "alpha_walking.onnx"
    source = import_deployed_policy(source_path)
    inferior = tmp_path / "inferior.npz"
    eligible = tmp_path / "eligible.npz"
    save_candidate_checkpoint(
        inferior,
        campaign=campaign(),
        generation=10,
        output_weight=source.output_weight,
        output_bias=source.output_bias + 2.0e-4,
        metrics={
            "terminal_success": 0.0,
            "tracking_score": 100.0,
            "task_return": 999.0,
        },
    )
    save_candidate_checkpoint(
        eligible,
        campaign=campaign(),
        generation=20,
        output_weight=source.output_weight,
        output_bias=source.output_bias + 1.0e-4,
        metrics={
            "terminal_success": 1.0,
            "tracking_score": 0.1,
            "task_return": -50.0,
        },
    )
    output = tmp_path / "selected.onnx"
    record = select_and_export_candidate(
        campaign=campaign(),
        registry=PRODUCTION_REGISTRY,
        runtime_repo=runtime_repo,
        checkpoints=(inferior, eligible),
        output_policy=output,
    )
    assert record["selected_generation"] == 20
    assert record["onnx_parity_max_abs_error"] < 1.0e-5
    assert output.is_file()
