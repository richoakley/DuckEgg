"""EGGROLL-only Autopatch campaign for the production walking policy."""

from __future__ import annotations

import hashlib
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
    DeploymentConditionProfile,
    runtime_lag_capacity,
)
from mjlab_microduck.eggroll.interop import jax_to_torch, torch_to_jax
from mjlab_microduck.eggroll.policy import OutputLayerPolicy, PostTrainingPolicyConfig
from mjlab_microduck.eggroll.policy_io import import_deployed_policy
from mjlab_microduck.eggroll.preflight import run_cuda_preflight
from mjlab_microduck.eggroll.rollout import make_environment

from .campaign import save_candidate_checkpoint
from .contracts import PatchCampaign, ReleaseScope
from .efficiency import (
    CostLedger,
    InteractionCost,
    PhaseProfiler,
    episode_interaction_cost,
)
from .foot_proof import (
    make_walking_proof_bank,
    walking_bank_sha256,
    walking_command_label,
)
from .locomotion_objective import (
    LocomotionObjectiveConfig,
    ReleaseScopeEpisodeGroup,
    aggregate_candidate_episodes,
    aggregate_release_scope_candidate_episodes,
    objective_definition,
    release_scope_objective_definition,
    summarize_heldout_episodes,
    summarize_release_scope_heldout_episodes,
)
from .locomotion_rollout import (
    StartupWorldState,
    capture_startup_world,
    evaluate_locomotion_bank,
)
from .qualification import (
    QualificationBackend,
    QualificationCandidate,
    QualificationController,
    QualificationPlan,
    campaign_side_gate_screen,
)
from .registry import AutopatchRegistry
from .walking_protocol import resolve_walking_protocol

TASK_ID = "Mjlab-Velocity-Flat-MicroDuck"
EVAL_EVERY = 5
SAVE_EVERY = 5
RELEASE_SCOPE_OBJECTIVE_ID = "locomotion-release-scope-lexicographic-v2"


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _candidate_is_exportable(
    *, campaign_id: str, release_retained: bool, improves_source: bool
) -> bool:
    """Keep smoke inspectable while requiring real improvement for evidence runs."""

    return release_retained and (improves_source or campaign_id.endswith("-cuda-smoke"))


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
    profiler: PhaseProfiler | None = None,
    immutable_source: bool = False,
):
    def actions(observations: torch.Tensor) -> torch.Tensor:
        profile_timer = profiler or PhaseProfiler(enabled=False)
        with profile_timer.measure("torch_to_jax_transfer"):
            converted = torch_to_jax(observations.contiguous())
        with profile_timer.measure("policy_inference"):
            if immutable_source:
                if generation is not None:
                    raise ValueError("source evaluation cannot name a generation")
                result = policy.source_actions(converted)
            elif generation is None:
                result = policy.base_actions(converted)
            else:
                result = policy.candidate_actions(converted, generation=generation)
            if profile_timer.enabled:
                result.block_until_ready()
        with profile_timer.measure("jax_to_torch_transfer"):
            return jax_to_torch(result, device=device)

    return actions


def validate_walking_campaign(
    *,
    campaign: PatchCampaign,
    registry: AutopatchRegistry,
) -> DeploymentConditionProfile:
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
    return resolve_walking_protocol(campaign).profile


def validate_release_scope_training(
    *,
    campaign: PatchCampaign,
    release_scope: ReleaseScope | None,
    profile: DeploymentConditionProfile,
) -> None:
    """Keep historical v1 runs immutable and bind v2 to explicit routing."""

    is_release_scope_objective = campaign.objective.objective_id == (
        RELEASE_SCOPE_OBJECTIVE_ID
    )
    if is_release_scope_objective != (release_scope is not None):
        raise ValueError(
            "the release-scope objective requires --release-scope, while the "
            "historical v1 objective forbids it"
        )
    if release_scope is None:
        return
    if release_scope.mode != "profile_specific":
        raise ValueError(
            "the walking runner currently supports profile-specific training; "
            "multi-profile aggregation is available but needs explicit profile banks"
        )
    if release_scope.source_fallback_sha256 != campaign.artifact_sha256:
        raise ValueError("release scope fallback is not the frozen source policy")
    role = release_scope.activation_profile_role
    hashes = dict(release_scope.profile_sha256s)
    if role is None or hashes.get(role) != profile.sha256:
        raise ValueError("release scope activation profile does not match the campaign")
    if tuple(campaign.objective.lexicographic_metrics) != (
        "retained_source_success_count",
        "repaired_source_failure_count",
        "worst_profile_command_success_rate",
        "terminal_stability_count",
        "worst_upright_fraction",
        "worst_progress_fraction",
        "mean_upright_fraction",
        "mean_progress_fraction",
        "negative_mean_forward_velocity_rmse",
        "negative_mean_action_rate_l2",
    ):
        raise ValueError("release-scope campaign objective metrics do not match v2")


def _evaluate(
    *,
    runtime: Any,
    cases: tuple[Any, ...],
    profile: Any,
    policy: OutputLayerPolicy,
    device: torch.device,
    objective: LocomotionObjectiveConfig,
    startup_worlds: tuple[StartupWorldState, ...],
    profiler: PhaseProfiler | None = None,
) -> tuple[tuple[float, ...], dict[str, float], list[dict[str, np.ndarray]]]:
    episodes = evaluate_locomotion_bank(
        runtime=runtime,
        cases=cases,
        profile=profile,
        action_fn=_action_fn(
            policy,
            device=device,
            generation=None,
            profiler=profiler,
        ),
        objective_config=objective,
        startup_worlds=startup_worlds,
        profiler=profiler,
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


def _episodes_from_json(rows: list[dict[str, Any]]) -> list[dict[str, np.ndarray]]:
    episodes: list[dict[str, np.ndarray]] = []
    for row in rows:
        summary = row.get("summary")
        if not isinstance(summary, dict):
            raise TypeError("serialized episode row has no summary")
        episodes.append({name: np.asarray(value) for name, value in summary.items()})
    return episodes


def _release_scope_group(
    *,
    release_scope: ReleaseScope,
    cases: tuple[Any, ...],
    source_episodes: list[dict[str, np.ndarray]],
    candidate_episodes: list[dict[str, np.ndarray]],
) -> ReleaseScopeEpisodeGroup:
    role = release_scope.activation_profile_role
    if role is None:
        raise ValueError("profile-specific release scope has no activation role")
    return ReleaseScopeEpisodeGroup(
        profile_role=role,
        command_labels=tuple(walking_command_label(case) for case in cases),
        source_episodes=tuple(source_episodes),
        candidate_episodes=tuple(candidate_episodes),
    )


def _minimum_nominal_rate(campaign: PatchCampaign) -> float:
    gates = [gate for gate in campaign.gates if gate.profile_role == "nominal"]
    if len(gates) != 1 or gates[0].metric != "terminal_success_rate":
        raise ValueError("walking campaign requires one nominal success-rate gate")
    if gates[0].comparator not in {">", ">="}:
        raise ValueError("nominal retention gate must be a lower bound")
    return float(gates[0].threshold)


def _bank_accounting(
    *,
    episodes: list[dict[str, np.ndarray]],
    cases: tuple[Any, ...],
    runtime: Any,
    candidate_evaluations: int = 0,
):
    decimation = getattr(getattr(runtime.env, "cfg", None), "decimation", None)
    return episode_interaction_cost(
        episodes,
        requested_horizon_steps=tuple(int(case.horizon_steps) for case in cases),
        candidate_evaluations=candidate_evaluations,
        physics_decimation=int(decimation) if decimation is not None else None,
        require_simulator_ticks=True,
    )


def run_walking_campaign(
    *,
    campaign: PatchCampaign,
    registry: AutopatchRegistry,
    runtime_repo: Path,
    output_dir: Path,
    device: str,
    resume: Path | None = None,
    profile_generations: tuple[int, ...] = (),
    release_scope: ReleaseScope | None = None,
    qualification_plan: QualificationPlan | None = None,
    qualification_backend: QualificationBackend | None = None,
) -> Path:
    """Run one frozen EGGROLL campaign; no baseline optimizer is executed."""

    process_started = time.perf_counter()
    profile = validate_walking_campaign(campaign=campaign, registry=registry)
    protocol = resolve_walking_protocol(campaign)
    validate_release_scope_training(
        campaign=campaign,
        release_scope=release_scope,
        profile=profile,
    )
    if (qualification_plan is None) != (qualification_backend is None):
        raise ValueError(
            "qualification plan and evidence backend must be supplied together"
        )
    qualification_backend_sha256: str | None = None
    qualification_backend_provenance: dict[str, Any] | None = None
    if qualification_backend is not None:
        identity = getattr(qualification_backend, "identity_sha256", None)
        provenance = getattr(qualification_backend, "provenance", None)
        if not isinstance(identity, str) or len(identity) != 64:
            raise ValueError(
                "qualification backend must expose a content-addressed identity"
            )
        if not isinstance(provenance, dict):
            raise ValueError("qualification backend must expose provenance")
        qualification_backend_sha256 = identity
        qualification_backend_provenance = provenance
    selection_interval = (
        EVAL_EVERY
        if qualification_plan is None
        else qualification_plan.evaluation_interval
    )
    invalid_profile_generations = [
        generation
        for generation in profile_generations
        if generation < 0 or generation > campaign.optimizer.generations
    ]
    if invalid_profile_generations:
        raise ValueError(
            "profile generations must be zero (baseline) or completed generation "
            f"numbers within the campaign: {invalid_profile_generations}"
        )
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
    heldout = protocol.selection_bank
    nominal = protocol.nominal_selection_bank
    config = {
        "schema": "eggroll-autopatch-walking-run-v2",
        "campaign": campaign.canonical_dict(),
        "campaign_sha256": campaign.sha256,
        "source_policy": deployed.metadata(),
        "source_path": str(source_path.resolve()),
        "optimization_profile": profile.canonical_dict(),
        "optimization_profile_sha256": profile.sha256,
        "walking_protocol_id": protocol.protocol_id,
        "nominal_profile": NOMINAL_PROFILE.canonical_dict(),
        "nominal_profile_sha256": NOMINAL_PROFILE.sha256,
        "objective": (
            release_scope_objective_definition(objective)
            if release_scope is not None
            else objective_definition(objective)
        ),
        "release_scope": (
            None if release_scope is None else release_scope.canonical_dict()
        ),
        "release_scope_sha256": (
            None if release_scope is None else release_scope.sha256
        ),
        "execution_protocols": ["hyperscalees-eggroll"],
        "excluded_comparators": ["naive-es", "random-search"],
        "comparison_claim": "none; report absolute evaluation efficiency only",
        "deployment_transport": {
            "command_ema_alpha": 0.2,
            "leg_action_lowpass_alpha": 0.7,
            "head_action_lowpass_alpha": 0.5,
            "previous_action_observation": "raw-policy-output",
            "startup_world_identity": ("fresh-seeded-mjlab-construction-snapshot"),
        },
        "heldout_bank_sha256": walking_bank_sha256(heldout),
        "nominal_bank_sha256": walking_bank_sha256(nominal),
        "accounting_schema": "eggroll-autopatch-interaction-ledger-v1",
        "profile_generations": sorted(set(profile_generations)),
        "qualification_plan": (
            None if qualification_plan is None else qualification_plan.canonical_dict
        ),
        "qualification_plan_sha256": (
            None if qualification_plan is None else qualification_plan.sha256
        ),
        "qualification_backend_sha256": qualification_backend_sha256,
        "qualification_backend": qualification_backend_provenance,
        "campaign_side_selection_interval": selection_interval,
    }
    write_json(output_dir / "config.json", config)

    torch_device = torch.device(device)
    profiler = PhaseProfiler(
        enabled=0 in profile_generations,
        synchronizer=(
            (lambda: torch.cuda.synchronize(torch_device))
            if torch_device.type == "cuda"
            else None
        ),
    )
    ledger = CostLedger()
    qualification = (
        None
        if qualification_plan is None
        else QualificationController(qualification_plan)
    )
    startup_cache: dict[tuple[int, int], StartupWorldState] = {}

    def startup_worlds(
        cases: tuple[Any, ...], *, max_lag: int
    ) -> tuple[StartupWorldState, ...]:
        worlds = []
        for case in cases:
            key = (int(case.seed), max_lag)
            if key not in startup_cache:
                with profiler.measure("startup_world_construction"):
                    startup_cache[key] = capture_startup_world(
                        task=TASK_ID,
                        seed=case.seed,
                        device=device,
                        max_actuator_lag_steps=max_lag,
                    )
                ledger.record(
                    "construction.startup_identity",
                    InteractionCost(world_constructions=1),
                )
            worlds.append(startup_cache[key])
        return tuple(worlds)

    profile_lag = runtime_lag_capacity(profile)
    nominal_lag = runtime_lag_capacity(NOMINAL_PROFILE)
    heldout_startup = startup_worlds(heldout, max_lag=profile_lag)
    nominal_startup = startup_worlds(nominal, max_lag=nominal_lag)
    with profiler.measure("vector_environment_construction"):
        training_runtime = make_environment(
            task=TASK_ID,
            num_envs=campaign.optimizer.population,
            device=device,
            seed=campaign.optimizer.seed,
            matched_candidates=True,
            max_actuator_lag_steps=runtime_lag_capacity(profile),
        )
        ledger.record(
            "construction.training_vector_slots",
            InteractionCost(world_constructions=campaign.optimizer.population),
        )
        evaluation_runtime = make_environment(
            task=TASK_ID,
            num_envs=1,
            device=device,
            seed=campaign.optimizer.seed + 1,
            matched_candidates=False,
            max_actuator_lag_steps=runtime_lag_capacity(profile),
        )
        ledger.record(
            "construction.evaluation_vector_slots",
            InteractionCost(world_constructions=1),
        )
    metrics_history: list[dict[str, Any]] = []
    start_generation = 0
    best_key: tuple[float, ...]
    baseline: dict[str, Any]
    try:
        if resume is None:
            profiler.enabled = 0 in profile_generations
            shifted_key, shifted_metrics, shifted_episodes = _evaluate(
                runtime=evaluation_runtime,
                cases=heldout,
                profile=profile,
                policy=policy,
                device=torch_device,
                objective=objective,
                startup_worlds=heldout_startup,
                profiler=profiler,
            )
            nominal_key, nominal_metrics, nominal_episodes = _evaluate(
                runtime=evaluation_runtime,
                cases=nominal,
                profile=NOMINAL_PROFILE,
                policy=policy,
                device=torch_device,
                objective=objective,
                startup_worlds=nominal_startup,
                profiler=profiler,
            )
            shifted_legacy_key = shifted_key
            shifted_legacy_metrics = shifted_metrics
            if release_scope is not None:
                shifted_key, shifted_metrics = summarize_release_scope_heldout_episodes(
                    (
                        _release_scope_group(
                            release_scope=release_scope,
                            cases=heldout,
                            source_episodes=shifted_episodes,
                            candidate_episodes=shifted_episodes,
                        ),
                    ),
                    required_retention_roles=(release_scope.required_retention_roles),
                )
            ledger.record(
                "source_baseline",
                _bank_accounting(
                    episodes=shifted_episodes,
                    cases=heldout,
                    runtime=evaluation_runtime,
                ),
            )
            ledger.record(
                "source_baseline",
                _bank_accounting(
                    episodes=nominal_episodes,
                    cases=nominal,
                    runtime=evaluation_runtime,
                ),
            )
            baseline = {
                "shifted_key": list(shifted_key),
                "shifted_metrics": shifted_metrics,
                "shifted_legacy_key": list(shifted_legacy_key),
                "shifted_legacy_metrics": shifted_legacy_metrics,
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
            if checkpoint.get("qualification_backend_sha256") != (
                qualification_backend_sha256
            ):
                raise ValueError(
                    "resume checkpoint qualification backend identity changed"
                )
            policy.load_state_dict(checkpoint["policy_state"])
            start_generation = int(checkpoint["next_generation"])
            metrics_history = list(checkpoint["metrics_history"])
            baseline = dict(checkpoint["baseline"])
            best_key = tuple(float(value) for value in checkpoint["best_key"])
            interaction_ledger = checkpoint.get("interaction_ledger")
            if interaction_ledger is None:
                raise ValueError(
                    "resume checkpoint predates exact interaction accounting; "
                    "refusing to undercount prior work"
                )
            resumed_process_construction = ledger.phase("construction")
            ledger = CostLedger.from_state_dict(interaction_ledger)
            ledger.record(
                "construction.resume_process",
                resumed_process_construction,
            )
            phase_profile = checkpoint.get("phase_profile")
            if phase_profile is not None:
                profiler.merge_state_dict(phase_profile)
            qualification_state = checkpoint.get("qualification")
            if qualification is not None:
                if qualification_state is None or qualification_plan is None:
                    raise ValueError(
                        "resume checkpoint predates requested qualification state"
                    )
                qualification = QualificationController.from_state_dict(
                    qualification_state,
                    plan=qualification_plan,
                )
            elif qualification_state is not None:
                raise ValueError(
                    "resume checkpoint has qualification state but no plan was supplied"
                )
        source_shifted_episodes = _episodes_from_json(
            list(baseline["shifted_episodes"])
        )
        if start_generation >= campaign.optimizer.generations:
            raise ValueError("resume checkpoint already completed the campaign")
        if qualification is not None and qualification.should_stop:
            raise ValueError(
                "resume checkpoint already completed release qualification"
            )

        nominal_floor = _minimum_nominal_rate(campaign)
        episodes_per_command = campaign.optimizer.worlds_per_candidate // 4
        for generation in range(start_generation, campaign.optimizer.generations):
            completed = generation + 1
            stop_campaign = False
            profiler.enabled = completed in profile_generations
            generation_started = time.perf_counter()
            train_bank = make_walking_proof_bank(
                base_seed=campaign.optimizer.seed + generation * 10_007,
                episodes_per_command=episodes_per_command,
                prefix=f"train-g{generation:06d}",
            )
            source_train_episodes: list[dict[str, np.ndarray]] | None = None
            if release_scope is not None:
                source_train_episodes = evaluate_locomotion_bank(
                    runtime=evaluation_runtime,
                    cases=train_bank,
                    profile=profile,
                    action_fn=_action_fn(
                        policy,
                        device=torch_device,
                        generation=None,
                        profiler=profiler,
                        immutable_source=True,
                    ),
                    objective_config=objective,
                    startup_worlds=startup_worlds(
                        train_bank,
                        max_lag=profile_lag,
                    ),
                    profiler=profiler,
                )
                ledger.record(
                    "optimization.source_reference",
                    _bank_accounting(
                        episodes=source_train_episodes,
                        cases=train_bank,
                        runtime=evaluation_runtime,
                    ),
                )
            episodes = evaluate_locomotion_bank(
                runtime=training_runtime,
                cases=train_bank,
                profile=profile,
                action_fn=_action_fn(
                    policy,
                    device=torch_device,
                    generation=generation,
                    profiler=profiler,
                ),
                objective_config=objective,
                startup_worlds=startup_worlds(train_bank, max_lag=profile_lag),
                profiler=profiler,
            )
            generation_count = _bank_accounting(
                episodes=episodes,
                cases=train_bank,
                runtime=training_runtime,
                candidate_evaluations=campaign.optimizer.population,
            )
            ledger.record("optimization.candidates", generation_count)
            with profiler.measure("objective_aggregation_and_update"):
                if release_scope is None:
                    fitness, _keys, objective_metrics = aggregate_candidate_episodes(
                        episodes
                    )
                else:
                    assert source_train_episodes is not None
                    fitness, _keys, objective_metrics = (
                        aggregate_release_scope_candidate_episodes(
                            (
                                _release_scope_group(
                                    release_scope=release_scope,
                                    cases=train_bank,
                                    source_episodes=source_train_episodes,
                                    candidate_episodes=episodes,
                                ),
                            ),
                            required_retention_roles=(
                                release_scope.required_retention_roles
                            ),
                        )
                    )
                if np.unique(fitness).size <= 1:
                    raise RuntimeError(
                        "all candidates tied; refusing a blind EGGROLL update"
                    )
                before_weight, before_bias = policy.output_parameters()
                converted = policy.update(fitness, generation=generation)
                after_weight, after_bias = policy.output_parameters()
            delta = math.sqrt(
                float(np.square(after_weight - before_weight).sum())
                + float(np.square(after_bias - before_bias).sum())
            )
            if delta == 0.0:
                raise RuntimeError("non-identical fitness produced a zero update")

            metric: dict[str, Any] = {
                "generation": generation,
                "completed_generations": completed,
                "timestamp": datetime.now(UTC).isoformat(),
                "generation_wall_seconds": time.perf_counter() - generation_started,
                "train_bank_sha256": walking_bank_sha256(train_bank),
                "candidate_evaluations_cumulative": ledger.phase(
                    "optimization"
                ).candidate_evaluations,
                "world_rollouts_cumulative": ledger.phase(
                    "optimization"
                ).world_rollouts,
                "world_constructions_cumulative": ledger.total().world_constructions,
                "interaction_accounting": generation_count.to_dict(),
                "interaction_accounting_cumulative": ledger.phase(
                    "optimization"
                ).to_dict(),
                "fitness_mean": float(fitness.mean()),
                "fitness_std": float(fitness.std()),
                "fitness_unique": int(np.unique(fitness).size),
                "converted_fitness_norm": float(np.linalg.norm(converted)),
                "parameter_delta_norm": delta,
                **objective_metrics,
            }
            is_best = False
            if (
                completed % selection_interval == 0
                or completed == campaign.optimizer.generations
            ):
                with profiler.measure("campaign_side_selection_evaluation"):
                    shift_key, shift_metrics, shift_episodes = _evaluate(
                        runtime=evaluation_runtime,
                        cases=heldout,
                        profile=profile,
                        policy=policy,
                        device=torch_device,
                        objective=objective,
                        startup_worlds=heldout_startup,
                        profiler=profiler,
                    )
                    nominal_key, nominal_metrics, nominal_episodes = _evaluate(
                        runtime=evaluation_runtime,
                        cases=nominal,
                        profile=NOMINAL_PROFILE,
                        policy=policy,
                        device=torch_device,
                        objective=objective,
                        startup_worlds=nominal_startup,
                        profiler=profiler,
                    )
                    if release_scope is not None:
                        shift_key, shift_metrics = (
                            summarize_release_scope_heldout_episodes(
                                (
                                    _release_scope_group(
                                        release_scope=release_scope,
                                        cases=heldout,
                                        source_episodes=source_shifted_episodes,
                                        candidate_episodes=shift_episodes,
                                    ),
                                ),
                                required_retention_roles=(
                                    release_scope.required_retention_roles
                                ),
                            )
                        )
                shifted_selection_cost = _bank_accounting(
                    episodes=shift_episodes,
                    cases=heldout,
                    runtime=evaluation_runtime,
                )
                nominal_selection_cost = _bank_accounting(
                    episodes=nominal_episodes,
                    cases=nominal,
                    runtime=evaluation_runtime,
                )
                ledger.record("selection.shifted", shifted_selection_cost)
                ledger.record("selection.nominal", nominal_selection_cost)
                metric.update(
                    {f"shifted/{key}": value for key, value in shift_metrics.items()}
                )
                metric.update(
                    {f"nominal/{key}": value for key, value in nominal_metrics.items()}
                )
                nominal_retained = (
                    nominal_metrics["objective/terminal_success_rate"] >= nominal_floor
                )
                release_scope_retained = (
                    shift_metrics["objective/retained_source_success_rate"] >= 1.0
                    if release_scope is not None
                    else None
                )
                retained = (
                    nominal_retained
                    if release_scope is None
                    else release_scope_retained is not False
                )
                metric["selection/nominal_floor"] = nominal_floor
                metric["selection/nominal_retention_passed"] = nominal_retained
                metric["selection/nominal_diagnostic_only"] = (
                    release_scope is not None
                    and release_scope.mode == "profile_specific"
                )
                metric["selection/release_scope_retention_passed"] = (
                    release_scope_retained
                )
                metric["selection/shifted_key"] = list(shift_key)
                if retained and shift_key > best_key:
                    best_key = shift_key
                    is_best = True
                    write_json(
                        output_dir / "best_evaluation.json",
                        {
                            "generation": completed,
                            "shifted_metrics": shift_metrics,
                            "shifted_episodes": _json_episodes(heldout, shift_episodes),
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
                candidate_metrics.update(
                    {
                        name: float(value)
                        for name, value in zip(
                            campaign.objective.lexicographic_metrics,
                            shift_key,
                            strict=True,
                        )
                    }
                )
                if _candidate_is_exportable(
                    campaign_id=campaign.campaign_id,
                    release_retained=retained,
                    improves_source=is_best,
                ):
                    save_candidate_checkpoint(
                        output_dir / "candidates" / f"generation-{completed:06d}.npz",
                        campaign=campaign,
                        generation=completed,
                        output_weight=after_weight,
                        output_bias=after_bias,
                        metrics=candidate_metrics,
                    )
                if qualification is not None:
                    assert qualification_backend is not None
                    qualification_path = (
                        output_dir
                        / "qualification_candidates"
                        / f"generation-{completed:06d}.npz"
                    )
                    save_candidate_checkpoint(
                        qualification_path,
                        campaign=campaign,
                        generation=completed,
                        output_weight=after_weight,
                        output_bias=after_bias,
                        metrics=candidate_metrics,
                    )
                    campaign_passed, selection_reason = campaign_side_gate_screen(
                        campaign=campaign,
                        metrics_history=[*metrics_history, metric],
                        release_scope=release_scope,
                    )
                    selection_passed = retained and campaign_passed
                    if not retained:
                        selection_reason = (
                            "release-scope/source-retention screen failed; "
                            + selection_reason
                        )
                    status = qualification.qualify(
                        QualificationCandidate(
                            generation=completed,
                            checkpoint_sha256=_sha256_file(qualification_path),
                            selection_metrics=tuple(
                                sorted(
                                    (name, float(value))
                                    for name, value in candidate_metrics.items()
                                )
                            ),
                            selection_passed=selection_passed,
                            selection_reason=selection_reason,
                            selection_cost=(
                                shifted_selection_cost + nominal_selection_cost
                            ),
                        ),
                        qualification_backend,
                    )
                    for stage in qualification.attempts[-1]["stages"]:
                        ledger.record(
                            f"qualification.{stage['stage']}",
                            InteractionCost.from_dict(stage["cost"]),
                        )
                    metric["qualification/status"] = status
                    metric["qualification/reason"] = selection_reason
                    stop_campaign = qualification.should_stop

            metrics_history.append(metric)
            payload = {
                "schema": "eggroll-autopatch-walking-checkpoint-v2",
                "next_generation": completed,
                "campaign_sha256": campaign.sha256,
                "source_policy_sha256": deployed.source_sha256,
                "policy_state": policy.state_dict(),
                "baseline": baseline,
                "best_key": list(best_key),
                "metrics_history": metrics_history,
                "interaction_ledger": ledger.state_dict(),
                "phase_profile": profiler.to_dict(),
                "qualification": (
                    None if qualification is None else qualification.state_dict()
                ),
                "qualification_backend_sha256": qualification_backend_sha256,
            }
            with profiler.measure("checkpoint_and_artifact_generation"):
                append_jsonl(output_dir / "metrics.jsonl", metric)
                write_json(output_dir / "accounting.json", ledger.report())
                if (
                    completed % SAVE_EVERY == 0
                    or completed == campaign.optimizer.generations
                ):
                    save_checkpoint(
                        output_dir / "checkpoints" / f"generation-{completed:06d}.pkl",
                        payload,
                    )
                save_checkpoint(output_dir / "last.pkl", payload)
                if is_best:
                    save_checkpoint(output_dir / "best.pkl", payload)
                write_json(output_dir / "phase_profile.json", profiler.to_dict())
                if qualification is not None:
                    write_json(
                        output_dir / "qualification.json",
                        qualification.state_dict(),
                    )
            print(
                f"generation={completed:04d} fitness_unique={np.unique(fitness).size} "
                f"delta={delta:.6g} wall={metric['generation_wall_seconds']:.1f}s"
            )
            if stop_campaign:
                break
    finally:
        evaluation_runtime.close()
        training_runtime.close()

    write_json(
        output_dir / "budget.json",
        {
            "schema": "eggroll-autopatch-budget-v2",
            "candidate_evaluations": ledger.phase("optimization").candidate_evaluations,
            "optimization_world_rollouts": ledger.phase("optimization").world_rollouts,
            "requested_optimization_simulator_steps": ledger.phase(
                "optimization"
            ).requested_simulator_steps,
            "executed_optimization_simulator_slot_steps": ledger.phase(
                "optimization"
            ).executed_simulator_steps,
            "active_optimization_interaction_steps": ledger.phase(
                "optimization"
            ).active_interaction_steps,
            "interaction_ledger": ledger.report(),
            "phase_profile": profiler.to_dict(),
            "qualification": (
                None if qualification is None else qualification.state_dict()
            ),
            "completed_generations": len(metrics_history),
            "stopped_after_complete_release_qualification": (
                False if qualification is None else qualification.should_stop
            ),
            "wall_seconds_current_process": time.perf_counter() - process_started,
            "wall_seconds_scope": (
                "current process from campaign entry through final budget write; "
                "do not add across resumed processes without an external job clock"
            ),
            "accelerator_allocation_seconds": None,
            "accelerator_allocation_seconds_reason": (
                "the runner has no authoritative scheduler allocation clock"
            ),
            "world_constructions": ledger.total().world_constructions,
            "startup_identities_cached_this_process": len(startup_cache),
            "comparative_sample_efficiency_claim": False,
            "absolute_efficiency_only": True,
        },
    )
    return output_dir
