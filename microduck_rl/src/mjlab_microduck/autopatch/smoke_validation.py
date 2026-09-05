"""Strict validation for non-evidence Autopatch CUDA smoke runs."""

from __future__ import annotations

import hashlib
import json
import math
import pickle
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np

from .contracts import PatchCampaign, ReleaseScope
from .efficiency import InteractionCost

SCHEMA = "eggroll-autopatch-cuda-smoke-validation-v1"


def _reject_nonfinite_json(value: str) -> None:
    raise ValueError(f"non-finite JSON constant: {value}")


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(), parse_constant=_reject_nonfinite_json)
    if not isinstance(value, dict):
        raise TypeError(f"{path.name} must contain a JSON object")
    return value


def _read_metrics(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text().splitlines(), start=1):
        if not line:
            continue
        value = json.loads(line, parse_constant=_reject_nonfinite_json)
        if not isinstance(value, dict):
            raise TypeError(f"metrics line {line_number} must contain an object")
        rows.append(value)
    if not rows:
        raise ValueError("CUDA smoke requires at least one generation metric")
    return rows


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _integer(value: Any, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    if value < 0:
        raise ValueError(f"{name} cannot be negative")
    return value


def _finite_float(value: Any, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _require_equal(actual: Any, expected: Any, *, name: str) -> None:
    if actual != expected:
        raise ValueError(f"{name} mismatch: expected {expected!r}, got {actual!r}")


def _cost(value: Any, *, name: str) -> InteractionCost:
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must contain an interaction cost")
    return InteractionCost.from_dict(value)


def validate_cuda_smoke_run(
    *,
    run_dir: Path,
    campaign_path: Path,
    release_scope_path: Path | None,
    source_manifest_path: Path,
    expected_candidate_evaluations: int,
    expected_optimization_world_rollouts: int,
    expected_optimization_requested_steps: int,
    expected_total_world_rollouts: int,
    expected_total_requested_steps: int,
) -> dict[str, Any]:
    """Verify CUDA/EGGROLL/accounting invariants without requiring release success.

    A smoke run is deliberately too small to assert repair quality.  It must prove that
    the real accelerator path produced diverse finite fitness, a non-zero EGGROLL update,
    a resumable generation checkpoint, and the exact predeclared interaction ledger.
    Release-retention and export remain mandatory for full campaign jobs only.
    """

    expected = {
        "candidate_evaluations": _integer(
            expected_candidate_evaluations, name="expected candidate evaluations"
        ),
        "optimization_world_rollouts": _integer(
            expected_optimization_world_rollouts,
            name="expected optimization world rollouts",
        ),
        "optimization_requested_steps": _integer(
            expected_optimization_requested_steps,
            name="expected optimization requested steps",
        ),
        "total_world_rollouts": _integer(
            expected_total_world_rollouts, name="expected total world rollouts"
        ),
        "total_requested_steps": _integer(
            expected_total_requested_steps, name="expected total requested steps"
        ),
    }
    if any(value <= 0 for value in expected.values()):
        raise ValueError("all expected smoke costs must be positive")

    run_dir = run_dir.resolve()
    campaign = PatchCampaign.from_json(campaign_path.read_text())
    needs_release_scope = (
        campaign.objective.objective_id == "locomotion-release-scope-lexicographic-v2"
    )
    if needs_release_scope != (release_scope_path is not None):
        raise ValueError(
            "the v2 release-scope objective requires a release scope, while "
            "historical campaign objectives forbid one"
        )
    release_scope = (
        None
        if release_scope_path is None
        else ReleaseScope.from_json(release_scope_path.read_text())
    )
    manifest = _read_json(source_manifest_path)
    config = _read_json(run_dir / "config.json")
    accounting = _read_json(run_dir / "accounting.json")
    budget = _read_json(run_dir / "budget.json")
    metrics = _read_metrics(run_dir / "metrics.jsonl")

    _require_equal(manifest.get("mode"), "smoke", name="manifest mode")
    _require_equal(
        manifest.get("evidence_role"),
        "non-evidence CUDA smoke",
        name="manifest evidence role",
    )
    _require_equal(
        manifest.get("execution_campaign_sha256"),
        campaign.sha256,
        name="execution campaign identity",
    )
    _require_equal(
        manifest.get("release_scope_sha256"),
        None if release_scope is None else release_scope.sha256,
        name="release scope identity",
    )
    if release_scope is not None:
        _require_equal(
            release_scope.source_fallback_sha256,
            campaign.artifact_sha256,
            name="release scope source fallback",
        )
    _require_equal(
        manifest.get("source_policy_sha256"),
        campaign.artifact_sha256,
        name="source policy identity",
    )

    _require_equal(
        config.get("campaign_sha256"), campaign.sha256, name="config campaign"
    )
    _require_equal(
        config.get("release_scope_sha256"),
        None if release_scope is None else release_scope.sha256,
        name="config release scope",
    )
    source_policy = config.get("source_policy")
    if not isinstance(source_policy, Mapping):
        raise TypeError("config source_policy must be an object")
    _require_equal(
        source_policy.get("source_sha256"),
        campaign.artifact_sha256,
        name="config source policy",
    )
    _require_equal(
        source_policy.get("widths"), [61, 512, 256, 128, 14], name="policy widths"
    )
    _require_equal(
        source_policy.get("trainable_scope"),
        "output-layer",
        name="policy trainable scope",
    )
    _require_equal(
        campaign.optimizer.trainable_scope,
        "final-affine-weight-and-bias",
        name="campaign trainable scope",
    )
    _require_equal(
        source_policy.get("trainable_parameters"), 1806, name="trainable parameters"
    )

    completed = _integer(
        budget.get("completed_generations"), name="completed generations"
    )
    _require_equal(
        completed, campaign.optimizer.generations, name="completed generations"
    )
    _require_equal(len(metrics), completed, name="metrics generation count")
    _require_equal(
        budget.get("candidate_evaluations"),
        expected["candidate_evaluations"],
        name="candidate evaluations",
    )
    _require_equal(
        budget.get("optimization_world_rollouts"),
        expected["optimization_world_rollouts"],
        name="optimization world rollouts",
    )
    _require_equal(
        budget.get("requested_optimization_simulator_steps"),
        expected["optimization_requested_steps"],
        name="optimization requested steps",
    )
    if budget.get("comparative_sample_efficiency_claim") is not False:
        raise ValueError("CUDA smoke must not make a comparative efficiency claim")

    if accounting.get("executed_steps_complete") is not True:
        raise ValueError("CUDA smoke requires complete executed-step accounting")
    total = _cost(accounting.get("total"), name="accounting total")
    _require_equal(
        total.candidate_evaluations,
        expected["candidate_evaluations"],
        name="ledger candidate evaluations",
    )
    _require_equal(
        total.world_rollouts,
        expected["total_world_rollouts"],
        name="ledger world rollouts",
    )
    _require_equal(
        total.requested_simulator_steps,
        expected["total_requested_steps"],
        name="ledger requested steps",
    )
    phase_totals = accounting.get("phase_totals")
    if not isinstance(phase_totals, Mapping):
        raise TypeError("accounting phase_totals must be an object")
    candidate_cost = _cost(
        phase_totals.get("optimization.candidates"), name="candidate optimization phase"
    )
    source_cost = _cost(
        phase_totals.get("optimization.source_reference"),
        name="source-reference optimization phase",
    )
    optimization_cost = candidate_cost + source_cost
    _require_equal(
        optimization_cost.world_rollouts,
        expected["optimization_world_rollouts"],
        name="phase optimization world rollouts",
    )
    _require_equal(
        optimization_cost.requested_simulator_steps,
        expected["optimization_requested_steps"],
        name="phase optimization requested steps",
    )
    _require_equal(
        budget.get("executed_optimization_simulator_slot_steps"),
        optimization_cost.executed_simulator_steps,
        name="executed optimization slot steps",
    )
    _require_equal(
        budget.get("active_optimization_interaction_steps"),
        optimization_cost.active_interaction_steps,
        name="active optimization interaction steps",
    )

    final_metric = metrics[-1]
    _require_equal(
        final_metric.get("completed_generations"), completed, name="metric generation"
    )
    _require_equal(
        final_metric.get("candidate_evaluations_cumulative"),
        expected["candidate_evaluations"],
        name="metric candidate evaluations",
    )
    fitness_unique = _integer(final_metric.get("fitness_unique"), name="unique fitness")
    if not 2 <= fitness_unique <= expected["candidate_evaluations"]:
        raise ValueError("candidate fitness must be non-identical")
    fitness_mean = _finite_float(final_metric.get("fitness_mean"), name="fitness mean")
    fitness_std = _finite_float(final_metric.get("fitness_std"), name="fitness std")
    if fitness_std <= 0.0:
        raise ValueError("candidate fitness standard deviation must be positive")
    delta_norm = _finite_float(
        final_metric.get("parameter_delta_norm"), name="parameter delta norm"
    )
    if delta_norm <= 0.0:
        raise ValueError("non-identical fitness produced a zero EGGROLL update")

    checkpoint = run_dir / "checkpoints" / f"generation-{completed:06d}.pkl"
    if not checkpoint.is_file() or checkpoint.stat().st_size <= 0:
        raise FileNotFoundError("CUDA smoke generation checkpoint is missing")
    with checkpoint.open("rb") as stream:
        checkpoint_state = pickle.load(stream)
    if not isinstance(checkpoint_state, Mapping):
        raise TypeError("CUDA smoke checkpoint must contain a mapping")
    _require_equal(
        checkpoint_state.get("schema"),
        "eggroll-autopatch-walking-checkpoint-v2",
        name="checkpoint schema",
    )
    _require_equal(
        checkpoint_state.get("campaign_sha256"),
        campaign.sha256,
        name="checkpoint campaign",
    )
    _require_equal(
        checkpoint_state.get("source_policy_sha256"),
        campaign.artifact_sha256,
        name="checkpoint source policy",
    )
    _require_equal(
        checkpoint_state.get("next_generation"), completed, name="checkpoint generation"
    )
    policy_state = checkpoint_state.get("policy_state")
    if not isinstance(policy_state, Mapping) or not isinstance(
        policy_state.get("params"), Mapping
    ):
        raise TypeError("checkpoint policy_state.params must be an object")
    params = policy_state["params"]
    weight = np.asarray(params.get("weight"))
    bias = np.asarray(params.get("bias"))
    if weight.shape != (14, 128) or bias.shape != (14,):
        raise ValueError("checkpoint changed the final-affine parameter shapes")
    if not np.isfinite(weight).all() or not np.isfinite(bias).all():
        raise ValueError("checkpoint final-affine parameters must be finite")

    last_checkpoint = run_dir / "last.pkl"
    if not last_checkpoint.is_file():
        raise FileNotFoundError("CUDA smoke last.pkl is missing")
    checkpoint_sha256 = _sha256_file(checkpoint)
    _require_equal(
        _sha256_file(last_checkpoint), checkpoint_sha256, name="last checkpoint bytes"
    )

    return {
        "schema": SCHEMA,
        "status": "pass",
        "evidence_role": "non-evidence CUDA smoke",
        "release_eligible": False,
        "source_commit": manifest.get("source_commit"),
        "source_policy_sha256": campaign.artifact_sha256,
        "campaign_sha256": campaign.sha256,
        "release_scope_sha256": None if release_scope is None else release_scope.sha256,
        "completed_generations": completed,
        "fitness_unique": fitness_unique,
        "fitness_mean": fitness_mean,
        "fitness_std": fitness_std,
        "parameter_delta_norm": delta_norm,
        "selection": {
            "nominal_retention_passed": final_metric.get(
                "selection/nominal_retention_passed"
            ),
            "release_scope_retention_passed": final_metric.get(
                "selection/release_scope_retention_passed"
            ),
        },
        "interaction_cost": {
            "optimization": optimization_cost.to_dict(),
            "total": total.to_dict(),
        },
        "world_constructions": budget.get("world_constructions"),
        "wall_seconds_current_process": budget.get("wall_seconds_current_process"),
        "checkpoint_sha256": checkpoint_sha256,
        "claim_boundary": (
            "validates the real CUDA path, EGGROLL update, checkpoint, and exact "
            "accounting only; it does not assert repair quality or release eligibility"
        ),
    }


def write_cuda_smoke_validation(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(payload), indent=2, sort_keys=True) + "\n")
