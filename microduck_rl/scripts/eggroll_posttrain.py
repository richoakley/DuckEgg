"""Validate, calibrate, post-train, evaluate, and export a deployed PPO actor."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import torch

# Coexist with the simulator instead of letting JAX reserve the whole GPU.
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

from mjlab_microduck.eggroll.checkpoint import load_checkpoint, write_json
from mjlab_microduck.eggroll.deployment import (
    ASYMMETRIC_CALIBRATION_PROFILES,
    CALIBRATION_LADDER,
    PROFILES,
    AsymmetricActuatorProfile,
    DeploymentProfile,
    bank_sha256,
    make_balanced_bank,
    runtime_lag_capacity,
    select_calibrated_profile,
)
from mjlab_microduck.eggroll.hero import render_hero
from mjlab_microduck.eggroll.interop import jax_to_torch, torch_to_jax
from mjlab_microduck.eggroll.objective import (
    StandupObjectiveConfig,
    summarize_heldout_episodes,
)
from mjlab_microduck.eggroll.policy import (
    OutputLayerPolicy,
    PostTrainingPolicyConfig,
)
from mjlab_microduck.eggroll.policy_io import (
    export_adapted_policy,
    import_deployed_policy,
    import_deployed_policy_bytes,
    numpy_actions,
    onnx_actions,
)
from mjlab_microduck.eggroll.preflight import run_cuda_preflight
from mjlab_microduck.eggroll.release import build_release_manifest, sha256_file
from mjlab_microduck.eggroll.rollout import evaluate_bank, make_environment
from mjlab_microduck.eggroll.runtime_probe import run_production_loader_probe
from mjlab_microduck.eggroll.trainer import TrainerConfig, train


def _base_action_fn(policy: OutputLayerPolicy, device: torch.device):
    def action_fn(observations: torch.Tensor) -> torch.Tensor:
        actions = policy.base_actions(torch_to_jax(observations.contiguous()))
        return jax_to_torch(actions, device=device)

    return action_fn


def _numpy_action_fn(policy: OutputLayerPolicy, device: torch.device):
    """Run the exact imported ONNX parameters without requiring JAX or CUDA."""

    if device.type != "cpu":
        raise ValueError("The NumPy playback path is CPU-only")

    def action_fn(observations: torch.Tensor) -> torch.Tensor:
        array = observations.detach().cpu().numpy()
        actions = numpy_actions(policy.deployed, array)
        return torch.from_numpy(actions).to(device=device)

    return action_fn


def _action_fn_for_device(policy: OutputLayerPolicy, device: str):
    torch_device = torch.device(device)
    if torch_device.type == "cpu":
        return _numpy_action_fn(policy, torch_device)
    run_cuda_preflight(device)
    return _base_action_fn(policy, torch_device)


def _policy_from_path(path: Path, *, seed: int = 0) -> OutputLayerPolicy:
    deployed = import_deployed_policy(path)
    return OutputLayerPolicy(
        deployed,
        PostTrainingPolicyConfig(sigma=0.01, learning_rate=0.001, rank=1, seed=seed),
    )


def _evaluate_profile(
    *,
    policy: OutputLayerPolicy,
    policy_path: Path,
    task: str,
    profile: DeploymentProfile | AsymmetricActuatorProfile,
    device: str,
    seed: int,
    episodes_per_pose: int,
    output_dir: Path,
    video: bool,
) -> dict[str, object]:
    action_fn = _action_fn_for_device(policy, device)
    bank = make_balanced_bank(
        profile=profile,
        base_seed=seed,
        episodes_per_pose=episodes_per_pose,
        prefix=profile.name,
    )
    runtime = make_environment(
        task=task,
        num_envs=1,
        device=device,
        seed=seed,
        matched_candidates=False,
        render_mode="rgb_array" if video else None,
        max_actuator_lag_steps=runtime_lag_capacity(profile),
    )
    try:
        episode = evaluate_bank(
            runtime=runtime,
            scenarios=bank,
            profile=profile,
            action_fn=action_fn,
            objective_config=StandupObjectiveConfig(),
            video_dir=output_dir / "videos" if video else None,
        )
    finally:
        runtime.close()
    key, metrics, pose_rates = summarize_heldout_episodes(
        episode, poses=[scenario.pose for scenario in bank]
    )
    result: dict[str, object] = {
        "policy": str(policy_path.resolve()),
        "source_policy_sha256": policy.deployed.source_sha256,
        "task": task,
        "execution_device": device,
        "authoritative_cuda_evaluation": torch.device(device).type == "cuda",
        "profile": profile.canonical_dict(),
        "profile_sha256": profile.sha256,
        "bank_sha256": bank_sha256(bank),
        "bank": [scenario.to_dict() for scenario in bank],
        "selection_key": list(key),
        "metrics": metrics,
        "pose_terminal_success_rates": pose_rates,
        "episodes": {name: value.tolist() for name, value in episode.items()},
    }
    write_json(output_dir / "summary.json", result)
    return result


def _validate(args: argparse.Namespace) -> int:
    deployed = import_deployed_policy(args.policy)
    rng = np.random.default_rng(args.seed)
    observations = rng.normal(size=(64, 61)).astype(np.float32)
    expected = numpy_actions(deployed, observations)
    actual = onnx_actions(deployed.source_model, observations)
    max_error = float(np.max(np.abs(expected - actual)))
    if max_error >= 1.0e-5:
        raise RuntimeError(f"Importer runtime error {max_error:.3g} exceeds 1e-5")
    print(json.dumps({**deployed.metadata(), "maximum_error": max_error}, indent=2))
    return 0


def _evaluate(args: argparse.Namespace) -> int:
    policy = _policy_from_path(args.policy, seed=args.seed)
    result = _evaluate_profile(
        policy=policy,
        policy_path=args.policy,
        task=args.task,
        profile=PROFILES[args.profile],
        device=args.device,
        seed=args.seed,
        episodes_per_pose=args.episodes_per_pose,
        output_dir=args.output_dir,
        video=args.video,
    )
    print(json.dumps(result["metrics"], indent=2, sort_keys=True))
    return 0


def _calibrate(args: argparse.Namespace) -> int:
    """Measure a severity ladder; do not choose a shift from intuition alone."""

    policy = _policy_from_path(args.policy, seed=args.seed)
    run_cuda_preflight(args.device)
    profiles = CALIBRATION_LADDER
    runtime = make_environment(
        task=args.task,
        num_envs=1,
        device=args.device,
        seed=args.seed,
        matched_candidates=False,
        max_actuator_lag_steps=max(profile.actuator_lag_steps for profile in profiles),
    )
    rows: list[dict[str, object]] = []
    try:
        for profile in profiles:
            bank = make_balanced_bank(
                profile=profile,
                base_seed=args.seed,
                episodes_per_pose=args.episodes_per_pose,
                prefix=profile.name,
            )
            episode = evaluate_bank(
                runtime=runtime,
                scenarios=bank,
                profile=profile,
                action_fn=_base_action_fn(policy, torch.device(args.device)),
                objective_config=StandupObjectiveConfig(),
            )
            key, metrics, pose_rates = summarize_heldout_episodes(
                episode, poses=[scenario.pose for scenario in bank]
            )
            rows.append(
                {
                    "profile": profile.canonical_dict(),
                    "profile_sha256": profile.sha256,
                    "bank_sha256": bank_sha256(bank),
                    "selection_key": list(key),
                    "metrics": metrics,
                    "pose_terminal_success_rates": pose_rates,
                }
            )
            print(
                f"{profile.name}: terminal_success="
                f"{metrics['eval/objective/success_rate']:.3f} "
                f"min_pose={metrics['eval/objective/min_pose_success_rate']:.3f}"
            )
    finally:
        runtime.close()
    selected = select_calibrated_profile(rows)
    artifact = {
        "policy": str(args.policy.resolve()),
        "source_policy_sha256": policy.deployed.source_sha256,
        "seed": args.seed,
        "episodes_per_pose": args.episodes_per_pose,
        "selection_rule": (
            "nominal success >=0.75; shifted success 0.10..min(0.90, "
            "nominal-0.10); every pose has at least one terminal success; "
            "select hardest eligible profile"
        ),
        "selected_profile": selected["profile"] if selected is not None else None,
        "selected_profile_sha256": (
            selected["profile_sha256"] if selected is not None else None
        ),
        "profiles": rows,
    }
    write_json(args.output_dir / "calibration.json", artifact)
    if selected is None:
        print("No profile produced a non-catastrophic, per-pose deployment gap.")
        return 0
    print(f"Selected profile: {selected['profile']['name']}")
    return 0


def _calibrate_asymmetric(args: argparse.Namespace) -> int:
    """Calibrate the predeclared left-knee/ankle fault without optimizing."""

    policy = _policy_from_path(args.policy, seed=args.seed)
    action_fn = _action_fn_for_device(policy, args.device)
    profiles = (PROFILES["nominal-fixed-median-v1"], *ASYMMETRIC_CALIBRATION_PROFILES)
    runtime = make_environment(
        task=args.task,
        num_envs=1,
        device=args.device,
        seed=args.seed,
        matched_candidates=False,
        max_actuator_lag_steps=max(
            runtime_lag_capacity(profile) for profile in profiles
        ),
    )
    rows: list[dict[str, object]] = []
    try:
        for profile in profiles:
            bank = make_balanced_bank(
                profile=profile,
                base_seed=args.seed,
                episodes_per_pose=args.episodes_per_pose,
                prefix=profile.name,
            )
            episode = evaluate_bank(
                runtime=runtime,
                scenarios=bank,
                profile=profile,
                action_fn=action_fn,
                objective_config=StandupObjectiveConfig(),
            )
            key, metrics, pose_rates = summarize_heldout_episodes(
                episode, poses=[scenario.pose for scenario in bank]
            )
            rows.append(
                {
                    "profile": profile.canonical_dict(),
                    "profile_sha256": profile.sha256,
                    "bank_sha256": bank_sha256(bank),
                    "selection_key": list(key),
                    "metrics": metrics,
                    "pose_terminal_success_rates": pose_rates,
                    "episodes": {
                        name: values.tolist() for name, values in episode.items()
                    },
                }
            )
            print(
                f"{profile.name}: terminal_success="
                f"{metrics['eval/objective/success_rate']:.3f} "
                f"min_pose={metrics['eval/objective/min_pose_success_rate']:.3f}",
                flush=True,
            )
    finally:
        runtime.close()
    selected = select_calibrated_profile(rows)
    artifact = {
        "schema": "microduck-asymmetric-actuator-calibration-v1",
        "policy": str(args.policy.resolve()),
        "source_policy_sha256": policy.deployed.source_sha256,
        "seed": args.seed,
        "episodes_per_pose": args.episodes_per_pose,
        "selection_rule": (
            "nominal success >=0.75; fault success 0.10..min(0.90, nominal-0.10); "
            "every pose has at least one terminal success; select hardest eligible profile"
        ),
        "selected_profile": selected["profile"] if selected is not None else None,
        "selected_profile_sha256": (
            selected["profile_sha256"] if selected is not None else None
        ),
        "profiles": rows,
    }
    write_json(args.output_dir / "asymmetric_calibration.json", artifact)
    if selected is None:
        print("No asymmetric profile met the predeclared calibration rule.")
    else:
        print(f"Selected profile: {selected['profile']['name']}")
    return 0


def _train(args: argparse.Namespace) -> int:
    config = TrainerConfig.from_toml(args.config)
    run_dir = train(
        config=config,
        source_policy=args.policy,
        calibration=args.calibration,
        resume=args.resume,
    )
    print(run_dir.resolve())
    return 0


def _export(args: argparse.Namespace) -> int:
    checkpoint = load_checkpoint(args.checkpoint)
    deployed = import_deployed_policy_bytes(checkpoint["source_policy_model"])
    config = PostTrainingPolicyConfig(**checkpoint["policy_config"])
    policy = OutputLayerPolicy(deployed, config)
    policy.load_state_dict(checkpoint["policy_state"])
    weight, bias = policy.output_parameters()
    maximum_error = export_adapted_policy(
        deployed,
        output_weight=weight,
        output_bias=bias,
        output_path=args.output,
    )
    write_json(
        args.output.with_suffix(args.output.suffix + ".json"),
        {
            "checkpoint": str(args.checkpoint.resolve()),
            "source_policy_sha256": deployed.source_sha256,
            "exported_policy_sha256": __import__("hashlib")
            .sha256(args.output.read_bytes())
            .hexdigest(),
            "maximum_runtime_error": maximum_error,
            "modified_parameters": 1_806,
        },
    )
    print(args.output.resolve())
    return 0


def _workflow(args: argparse.Namespace) -> int:
    """Run the predeclared adaptation through an evidence-bound runtime bundle."""

    run_cuda_preflight(args.device)
    release_dir = args.release_dir.resolve()
    release_dir.mkdir(parents=True, exist_ok=True)
    calibration = json.loads(args.calibration.read_text())
    selected_profile = calibration.get("selected_profile")
    if not isinstance(selected_profile, dict):
        raise TypeError("Calibration artifact has no selected deployment profile")
    shifted = DeploymentProfile(**selected_profile)
    nominal = PROFILES["nominal-fixed-median-v1"]

    source = import_deployed_policy(args.policy)
    if calibration.get("source_policy_sha256") != source.source_sha256:
        raise ValueError("Calibration artifact belongs to a different source policy")
    run_dir = train(
        config=TrainerConfig.from_toml(args.config),
        source_policy=args.policy,
        calibration=args.calibration,
        resume=args.resume,
    )
    checkpoint_path = run_dir / "best.pkl"
    checkpoint = load_checkpoint(checkpoint_path)
    generation = int(checkpoint["next_generation"])
    if generation != 100:
        raise ValueError(
            f"Release workflow requires the predeclared 100-generation result, got {generation}"
        )

    deployed = import_deployed_policy_bytes(checkpoint["source_policy_model"])
    policy = OutputLayerPolicy(
        deployed, PostTrainingPolicyConfig(**checkpoint["policy_config"])
    )
    policy.load_state_dict(checkpoint["policy_state"])
    weight, bias = policy.output_parameters()
    adapted_path = release_dir / args.policy_filename
    maximum_error = export_adapted_policy(
        deployed,
        output_weight=weight,
        output_bias=bias,
        output_path=adapted_path,
    )
    verification_path = release_dir / "export_verification.json"
    write_json(
        verification_path,
        {
            "checkpoint_generation": generation,
            "checkpoint_sha256": sha256_file(checkpoint_path),
            "source_policy_sha256": deployed.source_sha256,
            "exported_policy_sha256": sha256_file(adapted_path),
            "independent_maximum_runtime_error": maximum_error,
            "modified_parameters": 1_806,
        },
    )

    evaluations = release_dir / "workflow_evaluations"
    summary_paths: dict[str, Path] = {}
    for role, path, profile in (
        ("source_shifted", args.policy, shifted),
        ("adapted_shifted", adapted_path, shifted),
        ("source_nominal", args.policy, nominal),
        ("adapted_nominal", adapted_path, nominal),
    ):
        output_dir = evaluations / role
        evaluated = _policy_from_path(path, seed=args.final_seed)
        _evaluate_profile(
            policy=evaluated,
            policy_path=path,
            task=args.task,
            profile=profile,
            device=args.device,
            seed=args.final_seed,
            episodes_per_pose=args.episodes_per_pose,
            output_dir=output_dir,
            video=args.video,
        )
        summary_paths[role] = output_dir / "summary.json"

    runtime_verification = release_dir / "runtime_verification.json"
    run_production_loader_probe(
        policy_path=adapted_path,
        runtime_repo=args.runtime_repo,
        cargo=args.cargo,
        output=runtime_verification,
        ort_dylib=args.ort_dylib,
    )
    source_runtime_verification = (
        release_dir / "rollback/source_runtime_verification.json"
    )
    run_production_loader_probe(
        policy_path=args.policy,
        runtime_repo=args.runtime_repo,
        cargo=args.cargo,
        output=source_runtime_verification,
        ort_dylib=args.ort_dylib,
    )
    manifest_path = release_dir / "manifest.json"
    manifest = build_release_manifest(
        derivative_id=args.derivative_id,
        source_policy=args.policy,
        adapted_policy=adapted_path,
        checkpoint=checkpoint_path,
        export_verification=verification_path,
        training_dir=run_dir,
        summaries=summary_paths,
        evidence_dir=release_dir / "evidence",
        output=manifest_path,
        source_commit=args.source_commit,
        checkpoint_repository=args.checkpoint_repository,
        runtime_verification=runtime_verification,
    )

    updater_dir = release_dir / "microduck_updater"
    source_updater_dir = release_dir / "rollback/microduck_updater"
    subprocess.run(
        [
            str(args.cargo),
            "run",
            "--locked",
            "-p",
            "xtask",
            "--",
            "package-model",
            "--version",
            args.source_model_version,
            "--channel",
            "model-stand",
            "--policy",
            str(args.policy.resolve()),
            "--runtime-verification",
            str(source_runtime_verification),
            "--out",
            str(source_updater_dir),
            "--revision",
            f"production-source-{source.source_sha256[:12]}",
            "--model-api",
            "1",
        ],
        cwd=args.runtime_repo,
        check=True,
    )
    subprocess.run(
        [
            str(args.cargo),
            "run",
            "--locked",
            "-p",
            "xtask",
            "--",
            "package-model",
            "--version",
            args.model_version,
            "--channel",
            "model-stand",
            "--policy",
            str(adapted_path),
            "--evidence-manifest",
            str(manifest_path),
            "--runtime-verification",
            str(runtime_verification),
            "--out",
            str(updater_dir),
            "--revision",
            args.source_commit,
            "--model-api",
            "1",
        ],
        cwd=args.runtime_repo,
        check=True,
    )

    hero_path: Path | None = None
    if args.video:
        hero_path = release_dir / "eggroll_posttraining_hero.mp4"
        render_hero(
            manifest_path=manifest_path,
            source_shifted_dir=evaluations / "source_shifted/videos",
            adapted_shifted_dir=evaluations / "adapted_shifted/videos",
            source_nominal_dir=evaluations / "source_nominal/videos",
            adapted_nominal_dir=evaluations / "adapted_nominal/videos",
            output=hero_path,
        )

    print(
        json.dumps(
            {
                "status": "release-ready",
                "release_passed": manifest["release_decision"]["passed"],
                "manifest": str(manifest_path),
                "adapted_policy": str(adapted_path),
                "adapted_policy_sha256": hashlib.sha256(
                    adapted_path.read_bytes()
                ).hexdigest(),
                "updater_bundle": str(updater_dir),
                "source_rollback_bundle": str(source_updater_dir),
                "updater_bundle_signed": False,
                "hero": str(hero_path) if hero_path is not None else None,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    validate = subparsers.add_parser("validate-policy")
    validate.add_argument("--policy", type=Path, required=True)
    validate.add_argument("--seed", type=int, default=20260830)
    validate.set_defaults(func=_validate)

    for command, function in (("evaluate", _evaluate), ("calibrate-shift", _calibrate)):
        child = subparsers.add_parser(command)
        child.add_argument("--policy", type=Path, required=True)
        child.add_argument("--task", default="Mjlab-StandUp-Flat-MicroDuck")
        child.add_argument("--device", default="cuda:0")
        child.add_argument("--seed", type=int, default=20260830)
        child.add_argument("--episodes-per-pose", type=int, default=8)
        child.add_argument("--output-dir", type=Path, required=True)
        if command == "evaluate":
            child.add_argument("--profile", choices=sorted(PROFILES), required=True)
            child.add_argument("--video", action="store_true")
        child.set_defaults(func=function)

    asymmetric = subparsers.add_parser("calibrate-asymmetric")
    asymmetric.add_argument("--policy", type=Path, required=True)
    asymmetric.add_argument("--task", default="Mjlab-StandUp-Flat-MicroDuck")
    asymmetric.add_argument("--device", default="cpu")
    asymmetric.add_argument("--seed", type=int, default=20261001)
    asymmetric.add_argument("--episodes-per-pose", type=int, default=8)
    asymmetric.add_argument("--output-dir", type=Path, required=True)
    asymmetric.set_defaults(func=_calibrate_asymmetric)

    training = subparsers.add_parser("train")
    training.add_argument("--policy", type=Path, required=True)
    training.add_argument("--config", type=Path, required=True)
    training.add_argument("--calibration", type=Path, required=True)
    training.add_argument("--resume", type=Path)
    training.set_defaults(func=_train)

    exporting = subparsers.add_parser("export")
    exporting.add_argument("--checkpoint", type=Path, required=True)
    exporting.add_argument("--output", type=Path, required=True)
    exporting.set_defaults(func=_export)

    workflow = subparsers.add_parser(
        "workflow",
        help="train, export, replay, runtime-probe, and package one release",
    )
    workflow.add_argument("--policy", type=Path, required=True)
    workflow.add_argument("--config", type=Path, required=True)
    workflow.add_argument("--calibration", type=Path, required=True)
    workflow.add_argument("--release-dir", type=Path, required=True)
    workflow.add_argument("--runtime-repo", type=Path, required=True)
    workflow.add_argument("--cargo", type=Path, default=Path("cargo"))
    workflow.add_argument("--ort-dylib", type=Path)
    workflow.add_argument("--source-commit", required=True)
    workflow.add_argument("--checkpoint-repository", required=True)
    workflow.add_argument("--derivative-id", required=True)
    workflow.add_argument("--model-version", default="1.0.0")
    workflow.add_argument("--source-model-version", default="0.9.0")
    workflow.add_argument(
        "--policy-filename", default="alpha_stand_eggroll_adapted.onnx"
    )
    workflow.add_argument("--task", default="Mjlab-StandUp-Flat-MicroDuck")
    workflow.add_argument("--device", default="cuda:0")
    workflow.add_argument("--final-seed", type=int, default=20260901)
    workflow.add_argument("--episodes-per-pose", type=int, default=8)
    workflow.add_argument("--resume", type=Path)
    workflow.add_argument("--video", action="store_true")
    workflow.set_defaults(func=_workflow)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
