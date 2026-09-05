"""Single command-line entry point for MicroDuck EGGROLL Autopatch."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

from .registry import PRODUCTION_REGISTRY
from .runtime import RuntimePolicyBundle


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="eggroll-autopatch",
        description="Inspect and validate policy-agnostic MicroDuck Autopatch contracts.",
    )
    sub = parser.add_subparsers(dest="command", required=True)
    autopatch = sub.add_parser(
        "autopatch",
        help="resolve a sealed campaign into a reproducible, unlaunched execution plan",
    )
    autopatch.add_argument("--campaign", type=Path, required=True)
    autopatch.add_argument("--runtime-repo", type=Path, required=True)
    autopatch.add_argument("--policy-slot", required=True)
    autopatch.add_argument("--capability", required=True)
    autopatch.add_argument("--deployment", required=True)
    autopatch.add_argument("--acceptance", required=True)
    autopatch.add_argument("--output-dir", type=Path, required=True)
    sub.add_parser(
        "registry", help="print the canonical production capability registry"
    )
    validate = sub.add_parser(
        "validate-runtime",
        help="validate all sealed ONNX artifacts in a MicroDuck repo",
    )
    validate.add_argument("--runtime-repo", type=Path, required=True)
    smoke = sub.add_parser(
        "smoke-fleet",
        help="run all nine artifacts through ONNX/NumPy and optionally the Rust loader",
    )
    smoke.add_argument("--runtime-repo", type=Path, required=True)
    smoke.add_argument("--rust-probe", type=Path)
    smoke.add_argument("--seed", type=int, default=20260831)
    smoke.add_argument("--fixtures", type=int, default=8)
    runtime_config = sub.add_parser(
        "runtime-config",
        help="render the real robotd policy configuration for one artifact under test",
    )
    runtime_config.add_argument("--runtime-repo", type=Path, required=True)
    runtime_config.add_argument("--artifact", required=True)
    runtime_config.add_argument("--policy", type=Path)
    runtime_config.add_argument("--mode", choices=("walk", "roller"))
    campaign = sub.add_parser(
        "validate-campaign",
        help="validate and hash one immutable Autopatch campaign JSON",
    )
    campaign.add_argument("campaign", type=Path)
    test_plan = sub.add_parser(
        "test-plan",
        help="print mandatory capability-node and scheduler-edge release tests",
    )
    test_plan.add_argument("--artifact", required=True)
    evaluate = sub.add_parser(
        "evaluate-runtime",
        help="run one artifact in its actual registered task through robotd",
    )
    evaluate.add_argument("--runtime-repo", type=Path, required=True)
    evaluate.add_argument("--robotd", type=Path, required=True)
    evaluate.add_argument("--ort-dylib", type=Path, required=True)
    evaluate.add_argument("--artifact", required=True)
    evaluate.add_argument("--task")
    evaluate.add_argument("--policy", type=Path)
    evaluate.add_argument("--mode", choices=("walk", "roller"))
    evaluate.add_argument("--profile", default="nominal-fixed-median-v1")
    evaluate.add_argument("--seed", type=int, default=20260831)
    evaluate.add_argument("--side", choices=("left", "right"), default="right")
    evaluate.add_argument("--vx", type=float, default=0.0)
    evaluate.add_argument("--vy", type=float, default=0.0)
    evaluate.add_argument("--vyaw", type=float, default=0.0)
    evaluate.add_argument("--steps", type=int, default=250)
    evaluate.add_argument(
        "--return-step",
        type=int,
        help="for continuous policies, switch the real twist intent to zero at this step",
    )
    evaluate.add_argument(
        "--reset-label",
        choices=("standing", "sitting", "face-down", "face-up"),
        default="standing",
    )
    evaluate.add_argument("--device", default="cpu")
    evaluate.add_argument("--video", action="store_true")
    evaluate.add_argument("--timeout", type=float, default=30.0)
    evaluate.add_argument("--output-dir", type=Path, required=True)
    fleet = sub.add_parser(
        "evaluate-source-fleet",
        help="run all nine sealed source artifacts through actual tasks and robotd",
    )
    fleet.add_argument("--runtime-repo", type=Path, required=True)
    fleet.add_argument("--robotd", type=Path, required=True)
    fleet.add_argument("--ort-dylib", type=Path, required=True)
    fleet.add_argument("--profile", default="nominal-fixed-median-v1")
    fleet.add_argument("--seed", type=int, default=20260831)
    fleet.add_argument("--device", default="cpu")
    fleet.add_argument("--video", action="store_true")
    fleet.add_argument("--timeout", type=float, default=30.0)
    fleet.add_argument(
        "--attempts",
        type=int,
        default=2,
        help="strict transport attempts per case; rejected attempts remain as evidence",
    )
    fleet.add_argument("--output-dir", type=Path, required=True)
    foot_calibration = sub.add_parser(
        "calibrate-replacement-sole",
        help=(
            "run the sealed walking source policy over the predeclared replacement-"
            "sole ladder; performs no optimization"
        ),
    )
    foot_calibration.add_argument("--runtime-repo", type=Path, required=True)
    foot_calibration.add_argument("--robotd", type=Path, required=True)
    foot_calibration.add_argument("--ort-dylib", type=Path, required=True)
    foot_calibration.add_argument("--base-seed", type=int, default=20261011)
    foot_calibration.add_argument("--device", default="cpu")
    foot_calibration.add_argument("--timeout", type=float, default=45.0)
    foot_calibration.add_argument("--attempts", type=int, default=2)
    foot_calibration.add_argument("--output-dir", type=Path, required=True)
    replacement_foot = sub.add_parser(
        "calibrate-replacement-foot",
        help=(
            "run the sealed walking source policy over the v2 geometry/material "
            "ladder; performs no optimization"
        ),
    )
    replacement_foot.add_argument("--runtime-repo", type=Path, required=True)
    replacement_foot.add_argument("--robotd", type=Path, required=True)
    replacement_foot.add_argument("--ort-dylib", type=Path, required=True)
    replacement_foot.add_argument("--base-seed", type=int, default=20261021)
    replacement_foot.add_argument("--device", default="cpu")
    replacement_foot.add_argument("--timeout", type=float, default=45.0)
    replacement_foot.add_argument("--attempts", type=int, default=2)
    replacement_foot.add_argument("--output-dir", type=Path, required=True)
    priority_foot = sub.add_parser(
        "calibrate-priority-foot-material",
        help=(
            "run the sealed walking source policy over the contact-priority sole "
            "material ladder; performs no optimization"
        ),
    )
    priority_foot.add_argument("--runtime-repo", type=Path, required=True)
    priority_foot.add_argument("--robotd", type=Path, required=True)
    priority_foot.add_argument("--ort-dylib", type=Path, required=True)
    priority_foot.add_argument("--base-seed", type=int, default=20261021)
    priority_foot.add_argument("--device", default="cpu")
    priority_foot.add_argument("--timeout", type=float, default=45.0)
    priority_foot.add_argument("--attempts", type=int, default=2)
    priority_foot.add_argument("--output-dir", type=Path, required=True)
    wedge_foot = sub.add_parser(
        "calibrate-wedge-foot",
        help="run the sealed walking source policy over the wedge-sole ladder",
    )
    wedge_foot.add_argument("--runtime-repo", type=Path, required=True)
    wedge_foot.add_argument("--robotd", type=Path, required=True)
    wedge_foot.add_argument("--ort-dylib", type=Path, required=True)
    wedge_foot.add_argument("--base-seed", type=int, default=20261021)
    wedge_foot.add_argument("--device", default="cpu")
    wedge_foot.add_argument("--timeout", type=float, default=45.0)
    wedge_foot.add_argument("--attempts", type=int, default=2)
    wedge_foot.add_argument("--output-dir", type=Path, required=True)
    trunk_com = sub.add_parser(
        "calibrate-trunk-com",
        help="calibrate the fixed forward trunk-CoM ladder without optimization",
    )
    trunk_com.add_argument("--runtime-repo", type=Path, required=True)
    trunk_com.add_argument("--robotd", type=Path, required=True)
    trunk_com.add_argument("--ort-dylib", type=Path, required=True)
    trunk_com.add_argument("--base-seed", type=int, default=20293001)
    trunk_com.add_argument("--device", default="cuda:0")
    trunk_com.add_argument("--timeout", type=float, default=45.0)
    trunk_com.add_argument("--attempts", type=int, default=1)
    trunk_com.add_argument("--output-dir", type=Path, required=True)
    trunk_com_validation = sub.add_parser(
        "verify-trunk-com-calibration",
        help="validate exact source-only trunk-CoM calibration evidence and cost",
    )
    trunk_com_validation.add_argument("--manifest", type=Path, required=True)
    trunk_com_validation.add_argument("--protocol", type=Path, required=True)
    trunk_com_validation.add_argument("--source-manifest", type=Path, required=True)
    trunk_com_validation.add_argument("--output", type=Path, required=True)
    trunk_payload = sub.add_parser(
        "calibrate-trunk-payload",
        help="calibrate the fixed trunk-payload ladder without optimization",
    )
    trunk_payload.add_argument("--runtime-repo", type=Path, required=True)
    trunk_payload.add_argument("--robotd", type=Path, required=True)
    trunk_payload.add_argument("--ort-dylib", type=Path, required=True)
    trunk_payload.add_argument("--base-seed", type=int, default=20794001)
    trunk_payload.add_argument("--device", default="cuda:0")
    trunk_payload.add_argument("--timeout", type=float, default=45.0)
    trunk_payload.add_argument("--attempts", type=int, default=1)
    trunk_payload.add_argument("--output-dir", type=Path, required=True)
    trunk_payload_validation = sub.add_parser(
        "verify-trunk-payload-calibration",
        help="validate exact source-only trunk-payload evidence and cost",
    )
    trunk_payload_validation.add_argument("--manifest", type=Path, required=True)
    trunk_payload_validation.add_argument("--protocol", type=Path, required=True)
    trunk_payload_validation.add_argument("--source-manifest", type=Path, required=True)
    trunk_payload_validation.add_argument("--output", type=Path, required=True)
    source_behavior = sub.add_parser(
        "capture-walking-source-behavior",
        help="freeze source behavior on disjoint future release banks",
    )
    source_behavior.add_argument("--campaign", type=Path, required=True)
    source_behavior.add_argument("--calibration-validation", type=Path, required=True)
    source_behavior.add_argument("--source-manifest", type=Path, required=True)
    source_behavior.add_argument("--runtime-repo", type=Path, required=True)
    source_behavior.add_argument("--robotd", type=Path, required=True)
    source_behavior.add_argument("--ort-dylib", type=Path, required=True)
    source_behavior.add_argument("--output-dir", type=Path, required=True)
    cross_failure = sub.add_parser(
        "verify-trunk-com-study",
        help="validate and aggregate the predeclared three-seed trunk-CoM study",
    )
    cross_failure.add_argument("--protocol", type=Path, required=True)
    cross_failure.add_argument("--calibration-validation", type=Path, required=True)
    cross_failure.add_argument("--source-behavior-reference", type=Path, required=True)
    cross_failure.add_argument(
        "--seed-output-dir", type=Path, action="append", required=True
    )
    cross_failure.add_argument(
        "--max-requested-optimization-steps", type=int, default=5_120_000
    )
    cross_failure.add_argument("--output", type=Path, required=True)
    prepare_cross_failure = sub.add_parser(
        "prepare-trunk-com-study",
        help="materialize the calibrated predeclared three-seed contracts",
    )
    prepare_cross_failure.add_argument("--base-campaign", type=Path, required=True)
    prepare_cross_failure.add_argument("--protocol", type=Path, required=True)
    prepare_cross_failure.add_argument(
        "--calibration-validation", type=Path, required=True
    )
    prepare_cross_failure.add_argument("--output", type=Path, required=True)
    payload_study = sub.add_parser(
        "verify-trunk-payload-study",
        help="validate and aggregate the predeclared payload three-seed study",
    )
    payload_study.add_argument("--protocol", type=Path, required=True)
    payload_study.add_argument("--calibration-validation", type=Path, required=True)
    payload_study.add_argument("--source-behavior-reference", type=Path, required=True)
    payload_study.add_argument(
        "--seed-output-dir", type=Path, action="append", required=True
    )
    payload_study.add_argument(
        "--max-requested-optimization-steps", type=int, default=5_120_000
    )
    payload_study.add_argument("--output", type=Path, required=True)
    prepare_payload_study = sub.add_parser(
        "prepare-trunk-payload-study",
        help="materialize the calibrated predeclared payload contracts",
    )
    prepare_payload_study.add_argument("--base-campaign", type=Path, required=True)
    prepare_payload_study.add_argument("--protocol", type=Path, required=True)
    prepare_payload_study.add_argument(
        "--calibration-validation", type=Path, required=True
    )
    prepare_payload_study.add_argument("--output", type=Path, required=True)
    train_walking = sub.add_parser(
        "train-walking-campaign",
        help="run the frozen walking Autopatch campaign with EGGROLL only",
    )
    train_walking.add_argument("--campaign", type=Path, required=True)
    train_walking.add_argument("--runtime-repo", type=Path, required=True)
    train_walking.add_argument("--output-dir", type=Path, required=True)
    train_walking.add_argument("--device", default="cuda:0")
    train_walking.add_argument("--resume", type=Path)
    train_walking.add_argument(
        "--release-scope",
        type=Path,
        help="required by the versioned release-scope-aware objective",
    )
    train_walking.add_argument(
        "--profile-generation",
        action="append",
        type=int,
        default=[],
        help="repeat to collect synchronized phase timings; use 0 for baseline",
    )
    train_walking.add_argument(
        "--qualification-plan",
        type=Path,
        help="predeclared early-stop plan; requires --qualification-command-spec",
    )
    train_walking.add_argument(
        "--qualification-command-spec",
        type=Path,
        help="hashed direct-command backend; requires --qualification-plan",
    )
    efficiency_report = sub.add_parser(
        "efficiency-report",
        help="summarize an existing campaign without conflating gate triggers and release",
    )
    efficiency_report.add_argument("--run-dir", type=Path, required=True)
    efficiency_report.add_argument("--campaign", type=Path, required=True)
    efficiency_report.add_argument(
        "--release-scope",
        type=Path,
        help="apply the declared release-scope retention semantics to gate replay",
    )
    efficiency_report.add_argument(
        "--requested-steps-per-world", type=int, required=True
    )
    efficiency_report.add_argument("--output", type=Path, required=True)
    smoke_validation = sub.add_parser(
        "verify-cuda-smoke-run",
        help="verify exact accounting and optimizer invariants for a non-evidence smoke",
    )
    smoke_validation.add_argument("--run-dir", type=Path, required=True)
    smoke_validation.add_argument("--campaign", type=Path, required=True)
    smoke_validation.add_argument("--release-scope", type=Path)
    smoke_validation.add_argument("--source-manifest", type=Path, required=True)
    smoke_validation.add_argument(
        "--expected-candidate-evaluations", type=int, required=True
    )
    smoke_validation.add_argument(
        "--expected-optimization-world-rollouts", type=int, required=True
    )
    smoke_validation.add_argument(
        "--expected-optimization-requested-steps", type=int, required=True
    )
    smoke_validation.add_argument(
        "--expected-total-world-rollouts", type=int, required=True
    )
    smoke_validation.add_argument(
        "--expected-total-requested-steps", type=int, required=True
    )
    smoke_validation.add_argument("--output", type=Path, required=True)
    integrated_validation = sub.add_parser(
        "verify-integrated-early-stop-run",
        help="prove one campaign trained, fully qualified, stopped, and exported",
    )
    integrated_validation.add_argument("--run-dir", type=Path, required=True)
    integrated_validation.add_argument("--campaign", type=Path, required=True)
    integrated_validation.add_argument("--release-scope", type=Path, required=True)
    integrated_validation.add_argument("--qualification-plan", type=Path, required=True)
    integrated_validation.add_argument(
        "--qualification-command-spec", type=Path, required=True
    )
    integrated_validation.add_argument("--source-manifest", type=Path, required=True)
    integrated_validation.add_argument("--selection-record", type=Path, required=True)
    integrated_validation.add_argument("--output-policy", type=Path, required=True)
    integrated_validation.add_argument(
        "--max-requested-optimization-steps", type=int, required=True
    )
    integrated_validation.add_argument("--output", type=Path, required=True)
    qualification_stage = sub.add_parser(
        "qualify-release-stage",
        help="run one candidate-bound production qualification stage",
    )
    qualification_stage.add_argument("--stage", required=True)
    qualification_stage.add_argument("--candidate-checkpoint", type=Path, required=True)
    qualification_stage.add_argument("--checkpoint-sha256", required=True)
    qualification_stage.add_argument("--generation", type=int, required=True)
    qualification_stage.add_argument("--evidence-directory", type=Path, required=True)
    qualification_stage.add_argument("--result-path", type=Path, required=True)
    qualification_stage.add_argument("--campaign", type=Path, required=True)
    qualification_stage.add_argument("--release-scope", type=Path, required=True)
    qualification_stage.add_argument("--runtime-repo", type=Path, required=True)
    qualification_stage.add_argument("--source-behavior-reference", type=Path)
    paired = sub.add_parser(
        "evaluate-ab",
        help="run a source/derivative paired bank through actual tasks and robotd",
    )
    paired.add_argument("--runtime-repo", type=Path, required=True)
    paired.add_argument("--robotd", type=Path, required=True)
    paired.add_argument("--ort-dylib", type=Path, required=True)
    paired.add_argument("--artifact", required=True)
    paired.add_argument("--adapted-policy", type=Path, required=True)
    paired.add_argument(
        "--profile",
        action="append",
        required=True,
        metavar="ROLE=PROFILE",
        help="repeat for each paired deployment profile",
    )
    paired.add_argument("--bank", type=Path, required=True)
    paired.add_argument("--device", default="cpu")
    paired.add_argument("--video", action="store_true")
    paired.add_argument("--timeout", type=float, default=30.0)
    paired.add_argument("--attempts", type=int, default=2)
    paired.add_argument("--output-dir", type=Path, required=True)
    release = sub.add_parser(
        "release-envelope",
        help="verify candidate bytes, graph evidence and gates before signed packaging",
    )
    release.add_argument("--campaign", type=Path, required=True)
    release.add_argument(
        "--release-scope",
        type=Path,
        required=True,
        help="content-addressed profile-specific or multi-profile release contract",
    )
    release.add_argument("--source-policy", type=Path, required=True)
    release.add_argument("--adapted-policy", type=Path, required=True)
    release.add_argument("--runtime-probe", type=Path, required=True)
    release.add_argument(
        "--routing-evidence",
        type=Path,
        required=True,
        help="production updater attestation for fail-closed profile routing",
    )
    release.add_argument("--node-evidence", type=Path, action="append", required=True)
    release.add_argument(
        "--paired-evidence",
        type=Path,
        action="append",
        required=True,
        help=(
            "paired production-runtime A/B manifest; repeat for independent "
            "release banks"
        ),
    )
    release.add_argument("--covered-transition", action="append", required=True)
    release.add_argument(
        "--gate-series",
        type=Path,
        required=True,
        help="JSON object mapping each campaign gate id to its ordered values",
    )
    release.add_argument("--output", type=Path, required=True)
    non_regression = sub.add_parser(
        "verify-non-regression",
        help="bind two or more paired A/B banks into a strict retention decision",
    )
    non_regression.add_argument("--artifact", required=True)
    non_regression.add_argument("--adapted-policy", type=Path, required=True)
    non_regression.add_argument("--manifest", type=Path, action="append", required=True)
    non_regression.add_argument("--release-scope", type=Path, required=True)
    non_regression.add_argument("--output", type=Path, required=True)
    exporting = sub.add_parser(
        "select-export",
        help="select campaign checkpoints lexicographically and export the derivative",
    )
    exporting.add_argument("--campaign", type=Path, required=True)
    exporting.add_argument("--runtime-repo", type=Path, required=True)
    exporting.add_argument("--checkpoint", type=Path, action="append", required=True)
    exporting.add_argument("--output-policy", type=Path, required=True)
    exporting.add_argument("--output-record", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = _parser().parse_args(argv)
    if args.command == "autopatch":
        from .campaign import build_campaign_plan, write_campaign_plan
        from .contracts import PatchCampaign, ReleaseScope

        campaign = PatchCampaign.from_json(args.campaign.read_text())
        artifact = PRODUCTION_REGISTRY.artifact(campaign.artifact_id)
        selectors = {
            "policy slot": (args.policy_slot, artifact.runtime_slot),
            "capability": (args.capability, campaign.capability_id),
            "deployment": (args.deployment, campaign.condition.condition_id),
            "acceptance": (args.acceptance, campaign.objective.objective_id),
        }
        mismatches = {
            name: {"requested": requested, "campaign": expected}
            for name, (requested, expected) in selectors.items()
            if requested != expected
        }
        if mismatches:
            raise SystemExit(f"CLI selectors disagree with campaign: {mismatches}")
        payload = build_campaign_plan(
            campaign=campaign,
            registry=PRODUCTION_REGISTRY,
            runtime_repo=args.runtime_repo.resolve(),
        )
        output = args.output_dir.resolve() / "campaign_plan.json"
        write_campaign_plan(output, payload)
        payload = {**payload, "plan_path": str(output)}
    elif args.command == "registry":
        payload = PRODUCTION_REGISTRY.to_dict()
    elif args.command == "validate-runtime":
        reports = PRODUCTION_REGISTRY.validate_runtime_artifacts(
            args.runtime_repo.resolve()
        )
        payload = {"status": "pass", "artifacts": list(reports)}
    elif args.command == "smoke-fleet":
        from .fleet import smoke_production_fleet

        payload = smoke_production_fleet(
            registry=PRODUCTION_REGISTRY,
            runtime_repo=args.runtime_repo.resolve(),
            rust_probe=args.rust_probe.resolve() if args.rust_probe else None,
            seed=args.seed,
            fixture_count=args.fixtures,
        )
    elif args.command == "runtime-config":
        bundle = RuntimePolicyBundle.for_artifact(
            registry=PRODUCTION_REGISTRY,
            runtime_repo=args.runtime_repo.resolve(),
            artifact_id=args.artifact,
            replacement_policy=args.policy.resolve() if args.policy else None,
            mode=args.mode,
        )
        print(bundle.render_robotd_toml(), end="")
        return
    elif args.command == "validate-campaign":
        from .contracts import PatchCampaign

        campaign = PatchCampaign.from_json(args.campaign.read_text())
        PRODUCTION_REGISTRY.validate_campaign(campaign)
        payload = {
            "status": "pass",
            "campaign_id": campaign.campaign_id,
            "campaign_sha256": campaign.sha256,
            "artifact_id": campaign.artifact_id,
            "artifact_sha256": campaign.artifact_sha256,
            "capability_id": campaign.capability_id,
        }
    elif args.command == "test-plan":
        payload = PRODUCTION_REGISTRY.release_test_plan(args.artifact)
    elif args.command == "evaluate-runtime":
        from mjlab_microduck.eggroll.deployment import PROFILES

        from .evaluate import RuntimeEvaluationRequest, run_runtime_evaluation

        artifact = PRODUCTION_REGISTRY.artifact(args.artifact)
        task = args.task or artifact.task_ids[0]
        try:
            profile = PROFILES[args.profile]
        except KeyError as error:
            raise SystemExit(
                f"unknown profile {args.profile!r}; choose one of {tuple(PROFILES)}"
            ) from error
        payload = run_runtime_evaluation(
            registry=PRODUCTION_REGISTRY,
            runtime_repo=args.runtime_repo.resolve(),
            robotd=args.robotd.resolve(),
            ort_dylib=args.ort_dylib.resolve(),
            output_dir=args.output_dir.resolve(),
            request=RuntimeEvaluationRequest(
                artifact_id=args.artifact,
                task=task,
                seed=args.seed,
                side=args.side,
                command=(args.vx, args.vy, args.vyaw, *(0.0,) * 10),
                device=args.device,
                record_video=args.video,
                timeout_s=args.timeout,
                horizon_steps=args.steps,
                reset_label=args.reset_label,
                return_step=args.return_step,
            ),
            profile=profile,
            replacement_policy=args.policy.resolve() if args.policy else None,
            mode=args.mode,
        )
    elif args.command == "evaluate-source-fleet":
        from mjlab_microduck.eggroll.deployment import PROFILES

        from .suite import run_source_acceptance_suite

        try:
            profile = PROFILES[args.profile]
        except KeyError as error:
            raise SystemExit(
                f"unknown profile {args.profile!r}; choose one of {tuple(PROFILES)}"
            ) from error
        payload = run_source_acceptance_suite(
            registry=PRODUCTION_REGISTRY,
            runtime_repo=args.runtime_repo.resolve(),
            robotd=args.robotd.resolve(),
            ort_dylib=args.ort_dylib.resolve(),
            output_dir=args.output_dir.resolve(),
            profile=profile,
            seed=args.seed,
            device=args.device,
            record_video=args.video,
            timeout_s=args.timeout,
            max_attempts=args.attempts,
        )
    elif args.command == "calibrate-replacement-sole":
        from .foot_proof import run_replacement_sole_calibration

        payload = run_replacement_sole_calibration(
            registry=PRODUCTION_REGISTRY,
            runtime_repo=args.runtime_repo.resolve(),
            robotd=args.robotd.resolve(),
            ort_dylib=args.ort_dylib.resolve(),
            output_dir=args.output_dir.resolve(),
            base_seed=args.base_seed,
            device=args.device,
            timeout_s=args.timeout,
            max_attempts=args.attempts,
        )
    elif args.command == "calibrate-replacement-foot":
        from .foot_proof import run_replacement_foot_calibration

        payload = run_replacement_foot_calibration(
            registry=PRODUCTION_REGISTRY,
            runtime_repo=args.runtime_repo.resolve(),
            robotd=args.robotd.resolve(),
            ort_dylib=args.ort_dylib.resolve(),
            output_dir=args.output_dir.resolve(),
            base_seed=args.base_seed,
            device=args.device,
            timeout_s=args.timeout,
            max_attempts=args.attempts,
        )
    elif args.command == "calibrate-priority-foot-material":
        from .foot_proof import run_priority_foot_material_calibration

        payload = run_priority_foot_material_calibration(
            registry=PRODUCTION_REGISTRY,
            runtime_repo=args.runtime_repo.resolve(),
            robotd=args.robotd.resolve(),
            ort_dylib=args.ort_dylib.resolve(),
            output_dir=args.output_dir.resolve(),
            base_seed=args.base_seed,
            device=args.device,
            timeout_s=args.timeout,
            max_attempts=args.attempts,
        )
    elif args.command == "calibrate-wedge-foot":
        from .foot_proof import run_wedge_foot_calibration

        payload = run_wedge_foot_calibration(
            registry=PRODUCTION_REGISTRY,
            runtime_repo=args.runtime_repo.resolve(),
            robotd=args.robotd.resolve(),
            ort_dylib=args.ort_dylib.resolve(),
            output_dir=args.output_dir.resolve(),
            base_seed=args.base_seed,
            device=args.device,
            timeout_s=args.timeout,
            max_attempts=args.attempts,
        )
    elif args.command == "calibrate-trunk-com":
        from .foot_proof import run_trunk_com_calibration

        payload = run_trunk_com_calibration(
            registry=PRODUCTION_REGISTRY,
            runtime_repo=args.runtime_repo.resolve(),
            robotd=args.robotd.resolve(),
            ort_dylib=args.ort_dylib.resolve(),
            output_dir=args.output_dir.resolve(),
            base_seed=args.base_seed,
            device=args.device,
            timeout_s=args.timeout,
            max_attempts=args.attempts,
        )
    elif args.command == "verify-trunk-com-calibration":
        from .foot_proof import validate_trunk_com_calibration

        payload = validate_trunk_com_calibration(
            manifest_path=args.manifest.resolve(),
            protocol_path=args.protocol.resolve(),
            source_manifest_path=args.source_manifest.resolve(),
        )
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    elif args.command == "calibrate-trunk-payload":
        from .foot_proof import run_trunk_payload_calibration

        payload = run_trunk_payload_calibration(
            registry=PRODUCTION_REGISTRY,
            runtime_repo=args.runtime_repo.resolve(),
            robotd=args.robotd.resolve(),
            ort_dylib=args.ort_dylib.resolve(),
            output_dir=args.output_dir.resolve(),
            base_seed=args.base_seed,
            device=args.device,
            timeout_s=args.timeout,
            max_attempts=args.attempts,
        )
    elif args.command == "verify-trunk-payload-calibration":
        from .foot_proof import validate_trunk_payload_calibration

        payload = validate_trunk_payload_calibration(
            manifest_path=args.manifest.resolve(),
            protocol_path=args.protocol.resolve(),
            source_manifest_path=args.source_manifest.resolve(),
        )
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    elif args.command == "capture-walking-source-behavior":
        from .source_behavior import capture_walking_source_behavior_reference

        payload = capture_walking_source_behavior_reference(
            campaign_path=args.campaign.resolve(),
            calibration_validation_path=args.calibration_validation.resolve(),
            source_manifest_path=args.source_manifest.resolve(),
            runtime_repo=args.runtime_repo.resolve(),
            robotd=args.robotd.resolve(),
            ort_dylib=args.ort_dylib.resolve(),
            output_dir=args.output_dir.resolve(),
        )
    elif args.command == "verify-trunk-com-study":
        from .cross_failure_study import (
            validate_cross_failure_study,
            write_cross_failure_validation,
        )

        payload = validate_cross_failure_study(
            protocol_path=args.protocol.resolve(),
            calibration_validation_path=args.calibration_validation.resolve(),
            source_behavior_reference_path=args.source_behavior_reference.resolve(),
            seed_output_dirs=tuple(path.resolve() for path in args.seed_output_dir),
            max_requested_optimization_steps=(args.max_requested_optimization_steps),
        )
        write_cross_failure_validation(args.output.resolve(), payload)
    elif args.command == "prepare-trunk-com-study":
        from .cross_failure_study import build_trunk_com_study_contracts

        payload = build_trunk_com_study_contracts(
            base_campaign_path=args.base_campaign.resolve(),
            protocol_path=args.protocol.resolve(),
            calibration_validation_path=args.calibration_validation.resolve(),
        )
        args.output.resolve().parent.mkdir(parents=True, exist_ok=True)
        args.output.resolve().write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n"
        )
    elif args.command == "verify-trunk-payload-study":
        from .cross_failure_study import (
            validate_cross_failure_study,
            write_cross_failure_validation,
        )

        payload = validate_cross_failure_study(
            protocol_path=args.protocol.resolve(),
            calibration_validation_path=args.calibration_validation.resolve(),
            source_behavior_reference_path=args.source_behavior_reference.resolve(),
            seed_output_dirs=tuple(path.resolve() for path in args.seed_output_dir),
            max_requested_optimization_steps=(args.max_requested_optimization_steps),
        )
        write_cross_failure_validation(args.output.resolve(), payload)
    elif args.command == "prepare-trunk-payload-study":
        from .cross_failure_study import build_trunk_payload_study_contracts

        payload = build_trunk_payload_study_contracts(
            base_campaign_path=args.base_campaign.resolve(),
            protocol_path=args.protocol.resolve(),
            calibration_validation_path=args.calibration_validation.resolve(),
        )
        args.output.resolve().parent.mkdir(parents=True, exist_ok=True)
        args.output.resolve().write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n"
        )
    elif args.command == "train-walking-campaign":
        from .contracts import PatchCampaign, ReleaseScope
        from .locomotion_trainer import run_walking_campaign
        from .qualification import QualificationPlan
        from .qualification_command import (
            CommandQualificationBackend,
            CommandQualificationSpec,
        )

        campaign = PatchCampaign.from_json(args.campaign.read_text())
        release_scope = (
            ReleaseScope.from_json(args.release_scope.read_text())
            if args.release_scope is not None
            else None
        )
        if (args.qualification_plan is None) != (
            args.qualification_command_spec is None
        ):
            raise SystemExit(
                "--qualification-plan and --qualification-command-spec are required "
                "together"
            )
        qualification_plan = (
            None
            if args.qualification_plan is None
            else QualificationPlan.from_json(args.qualification_plan.read_text())
        )
        qualification_backend = None
        if qualification_plan is not None:
            command_spec = CommandQualificationSpec.from_json(
                args.qualification_command_spec.read_text()
            )
            qualification_backend = CommandQualificationBackend(
                spec=command_spec,
                plan=qualification_plan,
                candidate_directory=(
                    args.output_dir.resolve() / "qualification_candidates"
                ),
                evidence_directory=(
                    args.output_dir.resolve() / "qualification_evidence"
                ),
                working_directory=Path.cwd(),
            )
        run_dir = run_walking_campaign(
            campaign=campaign,
            registry=PRODUCTION_REGISTRY,
            runtime_repo=args.runtime_repo.resolve(),
            output_dir=args.output_dir.resolve(),
            device=args.device,
            resume=args.resume.resolve() if args.resume else None,
            profile_generations=tuple(args.profile_generation),
            release_scope=release_scope,
            qualification_plan=qualification_plan,
            qualification_backend=qualification_backend,
        )
        payload = {
            "status": "complete",
            "campaign_id": campaign.campaign_id,
            "campaign_sha256": campaign.sha256,
            "run_dir": str(run_dir),
        }
    elif args.command == "efficiency-report":
        from .contracts import PatchCampaign, ReleaseScope
        from .efficiency_study import (
            summarize_efficiency_run,
            write_efficiency_summary,
        )

        campaign = PatchCampaign.from_json(args.campaign.read_text())
        release_scope = (
            None
            if args.release_scope is None
            else ReleaseScope.from_json(args.release_scope.read_text())
        )
        payload = summarize_efficiency_run(
            run_dir=args.run_dir.resolve(),
            campaign=campaign,
            requested_steps_per_world=args.requested_steps_per_world,
            release_scope=release_scope,
        )
        write_efficiency_summary(args.output.resolve(), payload)
    elif args.command == "verify-cuda-smoke-run":
        from .smoke_validation import (
            validate_cuda_smoke_run,
            write_cuda_smoke_validation,
        )

        payload = validate_cuda_smoke_run(
            run_dir=args.run_dir.resolve(),
            campaign_path=args.campaign.resolve(),
            release_scope_path=(
                None if args.release_scope is None else args.release_scope.resolve()
            ),
            source_manifest_path=args.source_manifest.resolve(),
            expected_candidate_evaluations=args.expected_candidate_evaluations,
            expected_optimization_world_rollouts=(
                args.expected_optimization_world_rollouts
            ),
            expected_optimization_requested_steps=(
                args.expected_optimization_requested_steps
            ),
            expected_total_world_rollouts=args.expected_total_world_rollouts,
            expected_total_requested_steps=args.expected_total_requested_steps,
        )
        write_cuda_smoke_validation(args.output.resolve(), payload)
    elif args.command == "verify-integrated-early-stop-run":
        from .integrated_validation import (
            validate_integrated_early_stop_run,
            write_integrated_validation,
        )

        payload = validate_integrated_early_stop_run(
            run_dir=args.run_dir.resolve(),
            campaign_path=args.campaign.resolve(),
            release_scope_path=args.release_scope.resolve(),
            qualification_plan_path=args.qualification_plan.resolve(),
            qualification_command_spec_path=(args.qualification_command_spec.resolve()),
            source_manifest_path=args.source_manifest.resolve(),
            selection_record_path=args.selection_record.resolve(),
            output_policy_path=args.output_policy.resolve(),
            max_requested_optimization_steps=(args.max_requested_optimization_steps),
        )
        write_integrated_validation(args.output.resolve(), payload)
    elif args.command == "qualify-release-stage":
        from .qualification_runtime import run_qualification_stage

        payload = run_qualification_stage(
            stage=args.stage,
            candidate_checkpoint=args.candidate_checkpoint.resolve(),
            checkpoint_sha256=args.checkpoint_sha256,
            generation=args.generation,
            evidence_directory=args.evidence_directory.resolve(),
            result_path=args.result_path.resolve(),
            campaign_path=args.campaign.resolve(),
            release_scope_path=args.release_scope.resolve(),
            runtime_repo=args.runtime_repo.resolve(),
            source_behavior_reference_path=(
                None
                if args.source_behavior_reference is None
                else args.source_behavior_reference.resolve()
            ),
        )
    elif args.command == "evaluate-ab":
        from mjlab_microduck.eggroll.deployment import PROFILES

        from .ab import ABCase, run_paired_ab_suite

        profiles = []
        for value in args.profile:
            try:
                role, name = value.split("=", 1)
                profile = PROFILES[name]
            except (ValueError, KeyError) as error:
                raise SystemExit(
                    f"invalid --profile {value!r}; use ROLE=NAME where NAME is one of "
                    f"{tuple(PROFILES)}"
                ) from error
            profiles.append((role, profile))
        bank_document = json.loads(args.bank.read_text())
        if not isinstance(bank_document, list):
            raise SystemExit("paired bank JSON must contain a list of ABCase objects")
        payload = run_paired_ab_suite(
            registry=PRODUCTION_REGISTRY,
            artifact_id=args.artifact,
            adapted_policy=args.adapted_policy.resolve(),
            runtime_repo=args.runtime_repo.resolve(),
            robotd=args.robotd.resolve(),
            ort_dylib=args.ort_dylib.resolve(),
            profiles=tuple(profiles),
            cases=tuple(ABCase(**row) for row in bank_document),
            output_dir=args.output_dir.resolve(),
            device=args.device,
            record_video=args.video,
            timeout_s=args.timeout,
            max_attempts=args.attempts,
        )
    elif args.command == "release-envelope":
        from .contracts import PatchCampaign, ReleaseScope
        from .release import GateSeries, build_release_envelope

        campaign = PatchCampaign.from_json(args.campaign.read_text())
        release_scope = ReleaseScope.from_json(args.release_scope.read_text())
        gate_document = json.loads(args.gate_series.read_text())
        if not isinstance(gate_document, dict):
            raise SystemExit("gate-series JSON must contain an object")
        payload = build_release_envelope(
            campaign=campaign,
            registry=PRODUCTION_REGISTRY,
            source_policy=args.source_policy.resolve(),
            adapted_policy=args.adapted_policy.resolve(),
            runtime_probe=args.runtime_probe.resolve(),
            node_evidence=tuple(path.resolve() for path in args.node_evidence),
            covered_transition_sha256=tuple(args.covered_transition),
            gate_series=tuple(
                GateSeries(str(gate_id), tuple(float(value) for value in values))
                for gate_id, values in gate_document.items()
            ),
            paired_evidence=tuple(path.resolve() for path in args.paired_evidence),
            release_scope=release_scope,
            routing_evidence=args.routing_evidence.resolve(),
        )
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    elif args.command == "verify-non-regression":
        from mjlab_microduck.eggroll.release import sha256_file

        from .contracts import ReleaseScope
        from .release import build_paired_non_regression_envelope

        artifact = PRODUCTION_REGISTRY.artifact(args.artifact)
        release_scope = ReleaseScope.from_json(args.release_scope.read_text())
        payload = build_paired_non_regression_envelope(
            manifests=tuple(path.resolve() for path in args.manifest),
            artifact_id=artifact.artifact_id,
            source_sha256=artifact.expected_sha256,
            adapted_sha256=sha256_file(args.adapted_policy.resolve()),
            release_scope=release_scope,
        )
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    elif args.command == "select-export":
        from .campaign import select_and_export_candidate
        from .contracts import PatchCampaign

        campaign = PatchCampaign.from_json(args.campaign.read_text())
        payload = select_and_export_candidate(
            campaign=campaign,
            registry=PRODUCTION_REGISTRY,
            runtime_repo=args.runtime_repo.resolve(),
            checkpoints=tuple(path.resolve() for path in args.checkpoint),
            output_policy=args.output_policy.resolve(),
        )
        args.output_record.parent.mkdir(parents=True, exist_ok=True)
        args.output_record.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n"
        )
    else:  # pragma: no cover - argparse enforces the command set
        raise AssertionError(args.command)
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
