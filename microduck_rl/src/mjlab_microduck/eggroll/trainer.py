"""Output-layer EGGROLL post-training from an exact deployed PPO policy."""

from __future__ import annotations

import hashlib
import json
import math
import random
import time
import tomllib
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .checkpoint import append_jsonl, load_checkpoint, save_checkpoint, write_json
from .deployment import (
    NOMINAL_PROFILE,
    DeploymentProfile,
    bank_sha256,
    make_balanced_bank,
)
from .interop import jax_to_torch, torch_to_jax
from .objective import (
    StandupObjectiveConfig,
    aggregate_candidate_episodes,
    objective_definition,
    summarize_heldout_episodes,
)
from .policy import OutputLayerPolicy, PostTrainingPolicyConfig
from .policy_io import DeployedPolicy, import_deployed_policy
from .preflight import run_cuda_preflight
from .rollout import evaluate_bank, make_environment, rollout_episode


@dataclass(frozen=True)
class TrainerConfig:
    """One fixed-objective post-training experiment contract."""

    task: str = "Mjlab-StandUp-Flat-MicroDuck"
    run_name: str = "alpha-stand-low-voltage-latency-v1"
    output_dir: str = "runs/eggroll-posttrain"
    device: str = "cuda:0"
    seed: int = 20260830
    population: int = 512
    generations: int = 100
    train_episodes_per_pose: int = 1
    heldout_episodes_per_pose: int = 16
    nominal_episodes_per_pose: int = 8
    eval_every: int = 5
    save_every: int = 5
    sigma: float = 0.015
    learning_rate: float = 0.003
    rank: int = 4
    noise_reuse: int = 1
    nominal_retention_tolerance: float = 0.05
    success_hold_s: float = 1.0

    def __post_init__(self) -> None:
        if self.population < 2 or self.population % 2:
            raise ValueError("population must be an even integer >= 2")
        for name in (
            "generations",
            "train_episodes_per_pose",
            "heldout_episodes_per_pose",
            "nominal_episodes_per_pose",
            "eval_every",
            "save_every",
            "rank",
            "noise_reuse",
        ):
            if int(getattr(self, name)) <= 0:
                raise ValueError(f"{name} must be positive")
        if not self.device.startswith("cuda"):
            raise ValueError(
                "Actual mjlab post-training requires an NVIDIA CUDA device"
            )
        if self.sigma <= 0.0 or self.learning_rate <= 0.0:
            raise ValueError("sigma and learning_rate must be positive")
        if not 0.0 <= self.nominal_retention_tolerance <= 1.0:
            raise ValueError("nominal_retention_tolerance must be in [0,1]")
        if self.success_hold_s <= 0.0:
            raise ValueError("success_hold_s must be positive")

    @classmethod
    def from_toml(cls, path: Path) -> TrainerConfig:
        with path.open("rb") as stream:
            document = tomllib.load(stream)
        values = document.get("posttrain")
        if not isinstance(values, dict):
            raise TypeError("Config must contain a [posttrain] table")
        unknown = set(values) - set(cls.__dataclass_fields__)
        if unknown:
            raise ValueError(f"Unknown post-training config keys: {sorted(unknown)}")
        return cls(**values)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @property
    def sha256(self) -> str:
        payload = json.dumps(
            self.to_dict(), sort_keys=True, separators=(",", ":")
        ).encode()
        return hashlib.sha256(payload).hexdigest()


def _configure_process(device: str, seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.cuda.set_device(torch.device(device))


def _policy_action_fn(
    policy: OutputLayerPolicy,
    *,
    device: torch.device,
    generation: int | None,
):
    def action_fn(observations: torch.Tensor) -> torch.Tensor:
        jax_observations = torch_to_jax(observations.contiguous())
        if generation is None:
            actions = policy.base_actions(jax_observations)
        else:
            actions = policy.candidate_actions(jax_observations, generation=generation)
        return jax_to_torch(actions, device=device)

    return action_fn


def _evaluate(
    *,
    runtime: Any,
    scenarios: tuple[Any, ...],
    profile: DeploymentProfile,
    policy: OutputLayerPolicy,
    objective: StandupObjectiveConfig,
    device: torch.device,
) -> tuple[tuple[float, ...], dict[str, float], dict[str, np.ndarray]]:
    episode = evaluate_bank(
        runtime=runtime,
        scenarios=scenarios,
        profile=profile,
        action_fn=_policy_action_fn(policy, device=device, generation=None),
        objective_config=objective,
    )
    key, metrics, _pose_rates = summarize_heldout_episodes(
        episode, poses=[scenario.pose for scenario in scenarios]
    )
    return key, metrics, episode


def _json_episode(episode: dict[str, np.ndarray]) -> dict[str, list[Any]]:
    return {name: value.tolist() for name, value in episode.items()}


def _checkpoint_payload(
    *,
    next_generation: int,
    config: TrainerConfig,
    deployed: DeployedPolicy,
    policy: OutputLayerPolicy,
    baseline: dict[str, Any],
    best_key: tuple[float, ...],
    metrics_history: list[dict[str, Any]],
    optimization_profile: DeploymentProfile,
    calibration_sha256: str,
) -> dict[str, Any]:
    return {
        "next_generation": next_generation,
        "trainer_config": config.to_dict(),
        "trainer_config_sha256": config.sha256,
        "source_policy_sha256": deployed.source_sha256,
        "source_policy_model": deployed.source_model,
        "source_policy_metadata": deployed.metadata(),
        "policy_config": policy.config.to_dict(),
        "policy_state": policy.state_dict(),
        "baseline": baseline,
        "best_heldout_key": list(best_key),
        "metrics_history": metrics_history,
        "calibration_sha256": calibration_sha256,
        "deployment_profiles": {
            "optimization": optimization_profile.canonical_dict(),
            "retention": NOMINAL_PROFILE.canonical_dict(),
        },
    }


def _validate_resume(
    checkpoint: dict[str, Any],
    *,
    config: TrainerConfig,
    deployed: DeployedPolicy,
    policy: OutputLayerPolicy,
    optimization_profile: DeploymentProfile,
    calibration_sha256: str,
) -> None:
    if checkpoint.get("trainer_config_sha256") != config.sha256:
        raise ValueError("Resume config does not exactly match the checkpoint")
    if checkpoint.get("source_policy_sha256") != deployed.source_sha256:
        raise ValueError("Resume source policy does not match the checkpoint")
    if checkpoint.get("policy_config") != policy.config.to_dict():
        raise ValueError("Resume EGGROLL settings do not match the checkpoint")
    if checkpoint.get("calibration_sha256") != calibration_sha256:
        raise ValueError("Resume calibration artifact does not match the checkpoint")
    saved_profile = checkpoint.get("deployment_profiles", {}).get("optimization")
    if saved_profile != optimization_profile.canonical_dict():
        raise ValueError("Resume deployment profile does not match the checkpoint")


def load_calibrated_profile(
    path: Path, *, deployed: DeployedPolicy
) -> tuple[DeploymentProfile, str]:
    """Require a measured, source-bound shift before simulator expenditure."""

    raw = path.read_bytes()
    document = json.loads(raw)
    if document.get("source_policy_sha256") != deployed.source_sha256:
        raise ValueError("Calibration artifact belongs to a different source policy")
    if int(document.get("episodes_per_pose", 0)) < 8:
        raise ValueError("Calibration requires at least eight episodes per pose")
    selected = document.get("selected_profile")
    if selected is None:
        raise ValueError(
            "Calibration found no trainable deployment gap; refusing to train"
        )
    if not isinstance(selected, dict):
        raise TypeError("Calibration selected_profile must be an object")
    profile = DeploymentProfile(**selected)
    rows = document.get("profiles", [])
    matches = [row for row in rows if row.get("profile_sha256") == profile.sha256]
    if len(matches) != 1:
        raise ValueError("Selected calibrated profile is not uniquely evidenced")
    return profile, hashlib.sha256(raw).hexdigest()


def train(
    *,
    config: TrainerConfig,
    source_policy: Path,
    calibration: Path,
    resume: Path | None = None,
) -> Path:
    """Post-train and return the self-contained run directory."""

    deployed = import_deployed_policy(source_policy)
    optimization_profile, calibration_sha256 = load_calibrated_profile(
        calibration, deployed=deployed
    )
    _configure_process(config.device, config.seed)
    run_cuda_preflight(config.device)
    policy = OutputLayerPolicy(
        deployed,
        PostTrainingPolicyConfig(
            sigma=config.sigma,
            learning_rate=config.learning_rate,
            rank=config.rank,
            seed=config.seed,
            noise_reuse=config.noise_reuse,
        ),
    )
    if policy.trainable_parameter_count != 1_806:
        raise RuntimeError("The post-training scope is not the expected 1,806 params")

    run_dir = Path(config.output_dir) / config.run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    config_artifact = {
        "trainer": config.to_dict(),
        "trainer_sha256": config.sha256,
        "source_policy": deployed.metadata(),
        "policy": policy.config.to_dict(),
        "objective": objective_definition(
            StandupObjectiveConfig(stable_hold_s=config.success_hold_s)
        ),
        "calibration_artifact": str(calibration.resolve()),
        "calibration_sha256": calibration_sha256,
        "optimization_profile": optimization_profile.canonical_dict(),
        "optimization_profile_sha256": optimization_profile.sha256,
        "retention_profile": NOMINAL_PROFILE.canonical_dict(),
        "retention_profile_sha256": NOMINAL_PROFILE.sha256,
        "search_scope": "final Gemm weight and bias only; 1,806 parameters",
        "frozen_scope": "normalizer and first three Gemm plus ELU layers",
    }
    write_json(run_dir / "config.json", config_artifact)

    objective = StandupObjectiveConfig(stable_hold_s=config.success_hold_s)
    torch_device = torch.device(config.device)
    training_runtime = make_environment(
        task=config.task,
        num_envs=config.population,
        device=config.device,
        seed=config.seed,
        matched_candidates=True,
        max_actuator_lag_steps=optimization_profile.actuator_lag_steps,
    )
    evaluation_runtime = make_environment(
        task=config.task,
        num_envs=1,
        device=config.device,
        seed=config.seed + 1,
        matched_candidates=False,
        max_actuator_lag_steps=optimization_profile.actuator_lag_steps,
    )
    heldout = make_balanced_bank(
        profile=optimization_profile,
        base_seed=config.seed + 1_000_003,
        episodes_per_pose=config.heldout_episodes_per_pose,
        prefix="heldout-shift",
    )
    nominal = make_balanced_bank(
        profile=NOMINAL_PROFILE,
        base_seed=config.seed + 2_000_003,
        episodes_per_pose=config.nominal_episodes_per_pose,
        prefix="heldout-nominal",
    )

    start_generation = 0
    metrics_history: list[dict[str, Any]] = []
    baseline: dict[str, Any]
    best_key: tuple[float, ...]
    try:
        if resume is None:
            shift_key, shift_metrics, shift_episode = _evaluate(
                runtime=evaluation_runtime,
                scenarios=heldout,
                profile=optimization_profile,
                policy=policy,
                objective=objective,
                device=torch_device,
            )
            nominal_key, nominal_metrics, nominal_episode = _evaluate(
                runtime=evaluation_runtime,
                scenarios=nominal,
                profile=NOMINAL_PROFILE,
                policy=policy,
                objective=objective,
                device=torch_device,
            )
            baseline = {
                "shift_key": list(shift_key),
                "shift_metrics": shift_metrics,
                "shift_episode": _json_episode(shift_episode),
                "nominal_key": list(nominal_key),
                "nominal_metrics": nominal_metrics,
                "nominal_episode": _json_episode(nominal_episode),
                "heldout_bank_sha256": bank_sha256(heldout),
                "nominal_bank_sha256": bank_sha256(nominal),
            }
            write_json(run_dir / "source_baseline.json", baseline)
            best_key = shift_key
        else:
            checkpoint = load_checkpoint(resume)
            _validate_resume(
                checkpoint,
                config=config,
                deployed=deployed,
                policy=policy,
                optimization_profile=optimization_profile,
                calibration_sha256=calibration_sha256,
            )
            policy.load_state_dict(checkpoint["policy_state"])
            start_generation = int(checkpoint["next_generation"])
            metrics_history = list(checkpoint["metrics_history"])
            baseline = dict(checkpoint["baseline"])
            best_key = tuple(float(value) for value in checkpoint["best_heldout_key"])
            if start_generation >= config.generations:
                raise ValueError("Resume checkpoint already reached generations")

        nominal_floor = max(
            0.0,
            float(baseline["nominal_metrics"]["eval/objective/success_rate"])
            - config.nominal_retention_tolerance,
        )
        for generation in range(start_generation, config.generations):
            started = time.perf_counter()
            train_bank = make_balanced_bank(
                profile=optimization_profile,
                base_seed=config.seed + generation * 10_007,
                episodes_per_pose=config.train_episodes_per_pose,
                prefix=f"train-g{generation:06d}",
            )
            episodes: list[dict[str, np.ndarray]] = []
            action_fn = _policy_action_fn(
                policy, device=torch_device, generation=generation
            )
            for scenario in train_bank:
                episodes.append(
                    rollout_episode(
                        runtime=training_runtime,
                        scenario=scenario,
                        profile=optimization_profile,
                        action_fn=action_fn,
                        objective_config=objective,
                    )
                )
            fitness, _keys, objective_metrics = aggregate_candidate_episodes(
                episodes, poses=[scenario.pose for scenario in train_bank]
            )
            if np.unique(fitness).size <= 1:
                raise RuntimeError(
                    "All candidates received identical fitness; refusing a blind update"
                )
            before_weight, before_bias = policy.output_parameters()
            converted = policy.update(fitness, generation=generation)
            after_weight, after_bias = policy.output_parameters()
            delta = math.sqrt(
                float(np.square(after_weight - before_weight).sum())
                + float(np.square(after_bias - before_bias).sum())
            )
            if np.unique(fitness).size > 1 and delta == 0.0:
                raise RuntimeError("Non-identical fitness produced a zero update")

            completed = generation + 1
            metric: dict[str, Any] = {
                "generation": generation,
                "completed_generations": completed,
                "timestamp": datetime.now(UTC).isoformat(),
                "wall_time_s": time.perf_counter() - started,
                "train_bank_sha256": bank_sha256(train_bank),
                "fitness_mean": float(fitness.mean()),
                "fitness_std": float(fitness.std()),
                "fitness_unique": int(np.unique(fitness).size),
                "converted_fitness_norm": float(np.linalg.norm(converted)),
                "parameter_delta_norm": delta,
                **objective_metrics,
            }
            is_best = False
            if completed % config.eval_every == 0 or completed == config.generations:
                shift_key, shift_metrics, shift_episode = _evaluate(
                    runtime=evaluation_runtime,
                    scenarios=heldout,
                    profile=optimization_profile,
                    policy=policy,
                    objective=objective,
                    device=torch_device,
                )
                nominal_key, nominal_metrics, nominal_episode = _evaluate(
                    runtime=evaluation_runtime,
                    scenarios=nominal,
                    profile=NOMINAL_PROFILE,
                    policy=policy,
                    objective=objective,
                    device=torch_device,
                )
                metric.update(
                    {f"shift/{key}": value for key, value in shift_metrics.items()}
                )
                metric.update(
                    {f"nominal/{key}": value for key, value in nominal_metrics.items()}
                )
                nominal_rate = float(nominal_metrics["eval/objective/success_rate"])
                retention_passed = nominal_rate >= nominal_floor
                metric["selection/nominal_floor"] = nominal_floor
                metric["selection/nominal_retention_passed"] = retention_passed
                metric["selection/shift_key"] = list(shift_key)
                if retention_passed and shift_key > best_key:
                    best_key = shift_key
                    is_best = True
                    write_json(
                        run_dir / "best_evaluation.json",
                        {
                            "generation": completed,
                            "shift_metrics": shift_metrics,
                            "shift_episode": _json_episode(shift_episode),
                            "nominal_metrics": nominal_metrics,
                            "nominal_episode": _json_episode(nominal_episode),
                        },
                    )

            metrics_history.append(metric)
            append_jsonl(run_dir / "metrics.jsonl", metric)
            print(
                f"generation={completed:04d} fitness_std={fitness.std():.4f} "
                f"delta={delta:.6g} wall={metric['wall_time_s']:.1f}s"
            )
            payload = _checkpoint_payload(
                next_generation=completed,
                config=config,
                deployed=deployed,
                policy=policy,
                baseline=baseline,
                best_key=best_key,
                metrics_history=metrics_history,
                optimization_profile=optimization_profile,
                calibration_sha256=calibration_sha256,
            )
            if completed % config.save_every == 0 or completed == config.generations:
                save_checkpoint(run_dir / f"checkpoint_{completed:06d}.pkl", payload)
            save_checkpoint(run_dir / "last.pkl", payload)
            if is_best:
                save_checkpoint(run_dir / "best.pkl", payload)
    finally:
        evaluation_runtime.close()
        training_runtime.close()
    return run_dir
