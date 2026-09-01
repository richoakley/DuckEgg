"""EGGROLL-only Autopatch campaign for the production walking policy."""

from __future__ import annotations

import math
import random
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch

from mjlab_microduck.eggroll.checkpoint import (
    append_jsonl,
    load_checkpoint,
    save_checkpoint,
    write_json,
)
from mjlab_microduck.eggroll.deployment import (
    NOMINAL_PROFILE,
    PROFILES,
    WedgeFootProfile,
    runtime_lag_capacity,
)
from mjlab_microduck.eggroll.interop import jax_to_torch, torch_to_jax
from mjlab_microduck.eggroll.policy import OutputLayerPolicy, PostTrainingPolicyConfig
from mjlab_microduck.eggroll.policy_io import import_deployed_policy
from mjlab_microduck.eggroll.preflight import run_cuda_preflight
from mjlab_microduck.eggroll.rollout import make_environment

from .campaign import save_candidate_checkpoint
from .contracts import PatchCampaign
from .foot_proof import (
    make_replacement_foot_calibration_bank,
    make_walking_proof_bank,
    walking_bank_sha256,
    walking_command_label,
)
from .locomotion_objective import (
    LocomotionObjectiveConfig,
    aggregate_candidate_episodes,
    objective_definition,
    summarize_heldout_episodes,
)
from .locomotion_rollout import (
    StartupWorldState,
    capture_startup_world,
    evaluate_locomotion_bank,
)
from .registry import AutopatchRegistry

TASK_ID = "Mjlab-Velocity-Flat-MicroDuck"
HELDOUT_BASE_SEED = 20262021
NOMINAL_BASE_SEED = 20262022
HELDOUT_EPISODES_PER_COMMAND = 8
EVAL_EVERY = 5
SAVE_EVERY = 5


def _candidate_is_exportable(
    *, campaign_id: str, nominal_retained: bool, improves_source: bool
) -> bool:
    """Keep smoke inspectable while requiring real improvement for evidence runs."""

    return nominal_retained and (
        improves_source or campaign_id.endswith("-cuda-smoke")
    )


def _configure_process(device: str, seed: int) -> None:
    if not device.startswith("cuda"):
        raise ValueError("actual mjlab Autopatch search requires an NVIDIA CUDA device")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.cuda.set_device(torch.device(device))


def _action_fn(
    policy: OutputLayerPolicy,
    *,
    device: torch.device,
    generation: int | None,
):
    def actions(observations: torch.Tensor) -> torch.Tensor:
        converted = torch_to_jax(observations.contiguous())
        result = (
            policy.base_actions(converted)
            if generation is None
            else policy.candidate_actions(converted, generation=generation)
        )
        return jax_to_torch(result, device=device)

    return actions


def resolve_wedge_profile(campaign: PatchCampaign) -> WedgeFootProfile:
    if campaign.condition.adapter != "mjlab-wedge-foot-profile-v1":
        raise ValueError("walking proof requires the frozen wedge-foot adapter")
    parameters = dict(campaign.condition.parameters)
    name = parameters.get("profile_name")
    expected_sha = parameters.get("profile_sha256")
    if not isinstance(name, str) or not isinstance(expected_sha, str):
        raise TypeError("wedge condition must bind profile_name and profile_sha256")
    profile = PROFILES.get(name)
    if not isinstance(profile, WedgeFootProfile):
        raise TypeError("campaign profile is not a registered wedge-foot condition")
    if profile.sha256 != expected_sha:
        raise ValueError("campaign wedge profile hash does not match its semantics")
    return profile


def validate_walking_campaign(
    *,
    campaign: PatchCampaign,
    registry: AutopatchRegistry,
) -> WedgeFootProfile:
    registry.validate_campaign(campaign)
    if campaign.artifact_id != "alpha-walking":
        raise ValueError("this evaluator requires the production alpha-walking actor")
    if campaign.capability_id != "legged-locomotion":
        raise ValueError("walking campaign capability must be legged-locomotion")
    optimizer = campaign.optimizer
    if optimizer.algorithm != "hyperscalees-eggroll":
        raise ValueError("proof campaign must use real HyperscaleES EGGROLL")
    if optimizer.trainable_scope != "final-affine-weight-and-bias":
        raise ValueError("walking proof changes only the final affine layer")
    if optimizer.worlds_per_candidate % 4:
        raise ValueError("worlds_per_candidate must balance all four walking commands")
    calibration = make_replacement_foot_calibration_bank()
    if walking_bank_sha256(calibration) != campaign.calibration_bank_sha256:
        raise ValueError("campaign calibration bank does not match the frozen evidence")
    heldout = make_walking_proof_bank(
        base_seed=HELDOUT_BASE_SEED,
        episodes_per_command=HELDOUT_EPISODES_PER_COMMAND,
        prefix="heldout-wedge",
    )
    if walking_bank_sha256(heldout) != campaign.held_out_bank_sha256:
        raise ValueError("campaign held-out bank does not match the frozen evaluator")
    return resolve_wedge_profile(campaign)


def _evaluate(
    *,
    runtime: Any,
    cases: tuple[Any, ...],
    profile: Any,
    policy: OutputLayerPolicy,
    device: torch.device,
    objective: LocomotionObjectiveConfig,
    startup_worlds: tuple[StartupWorldState, ...],
) -> tuple[tuple[float, ...], dict[str, float], list[dict[str, np.ndarray]]]:
    episodes = evaluate_locomotion_bank(
        runtime=runtime,
        cases=cases,
        profile=profile,
        action_fn=_action_fn(policy, device=device, generation=None),
        objective_config=objective,
        startup_worlds=startup_worlds,
    )
    key, metrics = summarize_heldout_episodes(
        episodes,
        command_labels=[walking_command_label(case) for case in cases],
    )
    return key, metrics, episodes


def _json_episodes(
    cases: tuple[Any, ...], episodes: list[dict[str, np.ndarray]]
) -> list[dict[str, Any]]:
    return [
        {
            "case": {
                "case_id": case.case_id,
                "seed": case.seed,
                "command": list(case.command),
                "horizon_steps": case.horizon_steps,
            },
            "summary": {name: value.tolist() for name, value in episode.items()},
        }
        for case, episode in zip(cases, episodes, strict=True)
    ]


def _minimum_nominal_rate(campaign: PatchCampaign) -> float:
    gates = [gate for gate in campaign.gates if gate.profile_role == "nominal"]
    if len(gates) != 1 or gates[0].metric != "terminal_success_rate":
        raise ValueError("walking campaign requires one nominal success-rate gate")
    if gates[0].comparator not in {">", ">="}:
        raise ValueError("nominal retention gate must be a lower bound")
    return float(gates[0].threshold)


def run_walking_campaign(
    *,
    campaign: PatchCampaign,
    registry: AutopatchRegistry,
    runtime_repo: Path,
    output_dir: Path,
    device: str,
    resume: Path | None = None,
) -> Path:
    """Run one frozen EGGROLL campaign; no baseline optimizer is executed."""

    profile = validate_walking_campaign(campaign=campaign, registry=registry)
    artifact = registry.artifact(campaign.artifact_id)
    source_path = runtime_repo / "example_policies" / artifact.filename
    deployed = import_deployed_policy(source_path)
    if deployed.source_sha256 != campaign.artifact_sha256:
        raise ValueError("runtime source policy bytes do not match the campaign")
    if output_dir.exists() and any(output_dir.iterdir()) and resume is None:
        raise FileExistsError(f"output directory is not empty: {output_dir}")

    _configure_process(device, campaign.optimizer.seed)
    run_cuda_preflight(device)
    policy = OutputLayerPolicy(
        deployed,
        PostTrainingPolicyConfig(
            sigma=campaign.optimizer.sigma,
            learning_rate=campaign.optimizer.learning_rate,
            rank=campaign.optimizer.rank,
            seed=campaign.optimizer.seed,
        ),
    )
    if policy.trainable_parameter_count != 1_806:
        raise RuntimeError("walking patch scope drifted from 1,806 output parameters")

    output_dir.mkdir(parents=True, exist_ok=True)
    objective = LocomotionObjectiveConfig()
    heldout = make_walking_proof_bank(
        base_seed=HELDOUT_BASE_SEED,
        episodes_per_command=HELDOUT_EPISODES_PER_COMMAND,
        prefix="heldout-wedge",
    )
    nominal = make_walking_proof_bank(
        base_seed=NOMINAL_BASE_SEED,
        episodes_per_command=HELDOUT_EPISODES_PER_COMMAND,
        prefix="heldout-nominal",
    )
    config = {
        "schema": "eggroll-autopatch-walking-run-v1",
        "campaign": campaign.canonical_dict(),
        "campaign_sha256": campaign.sha256,
        "source_policy": deployed.metadata(),
        "source_path": str(source_path.resolve()),
        "optimization_profile": profile.canonical_dict(),
        "optimization_profile_sha256": profile.sha256,
        "nominal_profile": NOMINAL_PROFILE.canonical_dict(),
        "nominal_profile_sha256": NOMINAL_PROFILE.sha256,
        "objective": objective_definition(objective),
        "execution_protocols": ["hyperscalees-eggroll"],
        "excluded_comparators": ["naive-es", "random-search"],
        "comparison_claim": "none; report absolute evaluation efficiency only",
        "deployment_transport": {
            "command_ema_alpha": 0.2,
            "leg_action_lowpass_alpha": 0.7,
            "head_action_lowpass_alpha": 0.5,
            "previous_action_observation": "raw-policy-output",
            "startup_world_identity": (
                "fresh-seeded-mjlab-construction-snapshot"
            ),
        },
        "heldout_bank_sha256": walking_bank_sha256(heldout),
        "nominal_bank_sha256": walking_bank_sha256(nominal),
    }
    write_json(output_dir / "config.json", config)

    torch_device = torch.device(device)
    startup_cache: dict[tuple[int, int], StartupWorldState] = {}

    def startup_worlds(
        cases: tuple[Any, ...], *, max_lag: int
    ) -> tuple[StartupWorldState, ...]:
        worlds = []
        for case in cases:
            key = (int(case.seed), max_lag)
            if key not in startup_cache:
                startup_cache[key] = capture_startup_world(
                    task=TASK_ID,
                    seed=case.seed,
                    device=device,
                    max_actuator_lag_steps=max_lag,
                )
            worlds.append(startup_cache[key])
        return tuple(worlds)

    profile_lag = runtime_lag_capacity(profile)
    nominal_lag = runtime_lag_capacity(NOMINAL_PROFILE)
    heldout_startup = startup_worlds(heldout, max_lag=profile_lag)
    nominal_startup = startup_worlds(nominal, max_lag=nominal_lag)
    training_runtime = make_environment(
        task=TASK_ID,
        num_envs=campaign.optimizer.population,
        device=device,
        seed=campaign.optimizer.seed,
        matched_candidates=True,
        max_actuator_lag_steps=runtime_lag_capacity(profile),
    )
    evaluation_runtime = make_environment(
        task=TASK_ID,
        num_envs=1,
        device=device,
        seed=campaign.optimizer.seed + 1,
        matched_candidates=False,
        max_actuator_lag_steps=runtime_lag_capacity(profile),
    )
    metrics_history: list[dict[str, Any]] = []
    start_generation = 0
    best_key: tuple[float, ...]
    baseline: dict[str, Any]
    total_started = time.perf_counter()
    try:
        if resume is None:
            shifted_key, shifted_metrics, shifted_episodes = _evaluate(
                runtime=evaluation_runtime,
                cases=heldout,
                profile=profile,
                policy=policy,
                device=torch_device,
                objective=objective,
                startup_worlds=heldout_startup,
            )
            nominal_key, nominal_metrics, nominal_episodes = _evaluate(
                runtime=evaluation_runtime,
                cases=nominal,
                profile=NOMINAL_PROFILE,
                policy=policy,
                device=torch_device,
                objective=objective,
                startup_worlds=nominal_startup,
            )
            baseline = {
                "shifted_key": list(shifted_key),
                "shifted_metrics": shifted_metrics,
                "shifted_episodes": _json_episodes(heldout, shifted_episodes),
                "nominal_key": list(nominal_key),
                "nominal_metrics": nominal_metrics,
                "nominal_episodes": _json_episodes(nominal, nominal_episodes),
            }
            write_json(output_dir / "source_baseline.json", baseline)
            best_key = shifted_key
        else:
            checkpoint = load_checkpoint(resume)
            if checkpoint.get("campaign_sha256") != campaign.sha256:
                raise ValueError("resume checkpoint belongs to another campaign")
            if checkpoint.get("source_policy_sha256") != deployed.source_sha256:
                raise ValueError("resume checkpoint belongs to another source policy")
            policy.load_state_dict(checkpoint["policy_state"])
            start_generation = int(checkpoint["next_generation"])
            metrics_history = list(checkpoint["metrics_history"])
            baseline = dict(checkpoint["baseline"])
            best_key = tuple(float(value) for value in checkpoint["best_key"])
        if start_generation >= campaign.optimizer.generations:
            raise ValueError("resume checkpoint already completed the campaign")

        nominal_floor = _minimum_nominal_rate(campaign)
        episodes_per_command = campaign.optimizer.worlds_per_candidate // 4
        for generation in range(start_generation, campaign.optimizer.generations):
            generation_started = time.perf_counter()
            train_bank = make_walking_proof_bank(
                base_seed=campaign.optimizer.seed + generation * 10_007,
                episodes_per_command=episodes_per_command,
                prefix=f"train-g{generation:06d}",
            )
            episodes = evaluate_locomotion_bank(
                runtime=training_runtime,
                cases=train_bank,
                profile=profile,
                action_fn=_action_fn(
                    policy, device=torch_device, generation=generation
                ),
                objective_config=objective,
                startup_worlds=startup_worlds(train_bank, max_lag=profile_lag),
            )
            fitness, _keys, objective_metrics = aggregate_candidate_episodes(episodes)
            if np.unique(fitness).size <= 1:
                raise RuntimeError("all candidates tied; refusing a blind EGGROLL update")
            before_weight, before_bias = policy.output_parameters()
            converted = policy.update(fitness, generation=generation)
            after_weight, after_bias = policy.output_parameters()
            delta = math.sqrt(
                float(np.square(after_weight - before_weight).sum())
                + float(np.square(after_bias - before_bias).sum())
            )
            if delta == 0.0:
                raise RuntimeError("non-identical fitness produced a zero update")

            completed = generation + 1
            metric: dict[str, Any] = {
                "generation": generation,
                "completed_generations": completed,
                "timestamp": datetime.now(UTC).isoformat(),
                "generation_wall_seconds": time.perf_counter() - generation_started,
                "train_bank_sha256": walking_bank_sha256(train_bank),
                "candidate_evaluations_cumulative": (
                    completed * campaign.optimizer.population
                ),
                "world_rollouts_cumulative": (
                    completed
                    * campaign.optimizer.population
                    * campaign.optimizer.worlds_per_candidate
                ),
                "startup_world_constructions_cumulative": len(startup_cache),
                "fitness_mean": float(fitness.mean()),
                "fitness_std": float(fitness.std()),
                "fitness_unique": int(np.unique(fitness).size),
                "converted_fitness_norm": float(np.linalg.norm(converted)),
                "parameter_delta_norm": delta,
                **objective_metrics,
            }
            is_best = False
            if completed % EVAL_EVERY == 0 or completed == campaign.optimizer.generations:
                shift_key, shift_metrics, shift_episodes = _evaluate(
                    runtime=evaluation_runtime,
                    cases=heldout,
                    profile=profile,
                    policy=policy,
                    device=torch_device,
                    objective=objective,
                    startup_worlds=heldout_startup,
                )
                nominal_key, nominal_metrics, nominal_episodes = _evaluate(
                    runtime=evaluation_runtime,
                    cases=nominal,
                    profile=NOMINAL_PROFILE,
                    policy=policy,
                    device=torch_device,
                    objective=objective,
                    startup_worlds=nominal_startup,
                )
                metric.update({f"shifted/{key}": value for key, value in shift_metrics.items()})
                metric.update({f"nominal/{key}": value for key, value in nominal_metrics.items()})
                retained = (
                    nominal_metrics["objective/terminal_success_rate"]
                    >= nominal_floor
                )
                metric["selection/nominal_floor"] = nominal_floor
                metric["selection/nominal_retention_passed"] = retained
                metric["selection/shifted_key"] = list(shift_key)
                if retained and shift_key > best_key:
                    best_key = shift_key
                    is_best = True
                    write_json(
                        output_dir / "best_evaluation.json",
                        {
                            "generation": completed,
                            "shifted_metrics": shift_metrics,
                            "shifted_episodes": _json_episodes(
                                heldout, shift_episodes
                            ),
                            "nominal_metrics": nominal_metrics,
                            "nominal_episodes": _json_episodes(
                                nominal, nominal_episodes
                            ),
                        },
                    )
                candidate_metrics = {
                    key.removeprefix("objective/"): value
                    for key, value in shift_metrics.items()
                    if key.startswith("objective/")
                }
                candidate_metrics["nominal_terminal_success_rate"] = nominal_metrics[
                    "objective/terminal_success_rate"
                ]
                candidate_metrics["task_return_diagnostic"] = shift_metrics[
                    "diagnostic/mean_task_return"
                ]
                if _candidate_is_exportable(
                    campaign_id=campaign.campaign_id,
                    nominal_retained=retained,
                    improves_source=is_best,
                ):
                    save_candidate_checkpoint(
                        output_dir
                        / "candidates"
                        / f"generation-{completed:06d}.npz",
                        campaign=campaign,
                        generation=completed,
                        output_weight=after_weight,
                        output_bias=after_bias,
                        metrics=candidate_metrics,
                    )

            metrics_history.append(metric)
            append_jsonl(output_dir / "metrics.jsonl", metric)
            payload = {
                "schema": "eggroll-autopatch-walking-checkpoint-v1",
                "next_generation": completed,
                "campaign_sha256": campaign.sha256,
                "source_policy_sha256": deployed.source_sha256,
                "policy_state": policy.state_dict(),
                "baseline": baseline,
                "best_key": list(best_key),
                "metrics_history": metrics_history,
            }
            if completed % SAVE_EVERY == 0 or completed == campaign.optimizer.generations:
                save_checkpoint(
                    output_dir / "checkpoints" / f"generation-{completed:06d}.pkl",
                    payload,
                )
            save_checkpoint(output_dir / "last.pkl", payload)
            if is_best:
                save_checkpoint(output_dir / "best.pkl", payload)
            print(
                f"generation={completed:04d} fitness_unique={np.unique(fitness).size} "
                f"delta={delta:.6g} wall={metric['generation_wall_seconds']:.1f}s"
            )
    finally:
        evaluation_runtime.close()
        training_runtime.close()

    write_json(
        output_dir / "budget.json",
        {
            "candidate_evaluations": (
                campaign.optimizer.population * campaign.optimizer.generations
            ),
            "optimization_world_rollouts": (
                campaign.optimizer.population
                * campaign.optimizer.generations
                * campaign.optimizer.worlds_per_candidate
            ),
            "requested_optimization_simulator_steps": (
                campaign.optimizer.population
                * campaign.optimizer.generations
                * campaign.optimizer.worlds_per_candidate
                * 250
            ),
            "wall_seconds": time.perf_counter() - total_started,
            "startup_world_constructions": len(startup_cache),
            "comparative_sample_efficiency_claim": False,
            "absolute_efficiency_only": True,
        },
    )
    return output_dir
