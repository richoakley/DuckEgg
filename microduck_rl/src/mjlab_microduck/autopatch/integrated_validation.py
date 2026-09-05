"""Fail-closed validation for an integrated Autopatch early-stop campaign."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from .campaign import load_candidate_checkpoint
from .contracts import PatchCampaign, ReleaseScope
from .efficiency import CostLedger, InteractionCost
from .qualification import QualificationController, QualificationPlan
from .qualification_command import (
    RESULT_SCHEMA,
    TRANSCRIPT_SCHEMA,
    CommandQualificationSpec,
)

SCHEMA = "eggroll-autopatch-integrated-early-stop-validation-v1"


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
        raise ValueError("integrated validation requires generation metrics")
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


def _validate_stage_evidence(
    *,
    run_dir: Path,
    attempt: Mapping[str, Any],
    spec: CommandQualificationSpec,
) -> None:
    generation = _integer(attempt.get("generation"), name="attempt generation")
    checkpoint_sha256 = attempt.get("checkpoint_sha256")
    commands = {command.stage: command for command in spec.commands}
    stages = attempt.get("stages")
    if not isinstance(stages, list):
        raise TypeError("qualification attempt stages must be a list")
    for row in stages:
        if not isinstance(row, Mapping):
            raise TypeError("qualification stage must be an object")
        stage = row.get("stage")
        command = commands.get(stage)
        if command is None:
            raise ValueError(f"qualification stage is not predeclared: {stage!r}")
        evidence_sha256 = row.get("evidence_sha256")
        if not isinstance(evidence_sha256, str) or len(evidence_sha256) != 64:
            raise ValueError("qualification stage has no valid evidence hash")
        relative = command.result_path.format(
            candidate_checkpoint="unused",
            checkpoint_sha256=checkpoint_sha256,
            evidence_directory="unused",
            generation=generation,
            stage=stage,
        )
        result_path = run_dir / "qualification_evidence" / relative
        transcript_path = (
            run_dir
            / "qualification_evidence"
            / "transcripts"
            / f"generation-{generation:06d}-{stage}.json"
        )
        if not result_path.is_file() or _sha256_file(result_path) != evidence_sha256:
            raise ValueError(
                "qualification stage is not backed by its result manifest for "
                f"generation {generation} stage {stage}"
            )
        if not transcript_path.is_file():
            raise FileNotFoundError(
                "qualification command transcript is missing for "
                f"generation {generation} stage {stage}"
            )

        value = _read_json(result_path)
        _require_equal(value.get("schema"), RESULT_SCHEMA, name="evidence schema")
        _require_equal(value.get("stage"), stage, name="evidence stage")
        _require_equal(
            value.get("checkpoint_sha256"),
            checkpoint_sha256,
            name="evidence checkpoint",
        )
        _require_equal(value.get("generation"), generation, name="evidence generation")
        _require_equal(value.get("status"), row.get("status"), name="stage status")
        _require_equal(value.get("reason"), row.get("reason"), name="stage reason")

        transcript = _read_json(transcript_path)
        _require_equal(
            transcript.get("schema"), TRANSCRIPT_SCHEMA, name="transcript schema"
        )
        _require_equal(
            transcript.get("generation"), generation, name="transcript generation"
        )
        _require_equal(transcript.get("stage"), stage, name="transcript stage")
        _require_equal(
            transcript.get("checkpoint_sha256"),
            checkpoint_sha256,
            name="transcript checkpoint",
        )
        _require_equal(transcript.get("returncode"), 0, name="transcript return code")
        _require_equal(transcript.get("failure"), None, name="transcript failure")
        _finite_float(
            transcript.get("elapsed_wall_seconds"), name="transcript wall seconds"
        )
        if transcript["elapsed_wall_seconds"] < 0:
            raise ValueError("transcript wall seconds cannot be negative")
        if not isinstance(transcript.get("argv"), list) or not transcript["argv"]:
            raise TypeError("qualification transcript argv must be a non-empty list")
        for field in ("stdout", "stderr"):
            if not isinstance(transcript.get(field), str):
                raise TypeError(f"qualification transcript {field} must be text")


def validate_integrated_early_stop_run(
    *,
    run_dir: Path,
    campaign_path: Path,
    release_scope_path: Path,
    qualification_plan_path: Path,
    qualification_command_spec_path: Path,
    source_manifest_path: Path,
    selection_record_path: Path,
    output_policy_path: Path,
    max_requested_optimization_steps: int,
) -> dict[str, Any]:
    """Prove that one campaign trained, qualified, stopped, and exported in one job."""

    ceiling = _integer(
        max_requested_optimization_steps,
        name="requested optimization simulator-step ceiling",
    )
    if ceiling <= 0:
        raise ValueError(
            "requested optimization simulator-step ceiling must be positive"
        )

    run_dir = run_dir.resolve()
    campaign = PatchCampaign.from_json(campaign_path.read_text())
    release_scope = ReleaseScope.from_json(release_scope_path.read_text())
    plan = QualificationPlan.from_json(qualification_plan_path.read_text())
    spec = CommandQualificationSpec.from_json(
        qualification_command_spec_path.read_text()
    )
    spec.validate_plan(plan)

    manifest = _read_json(source_manifest_path)
    config = _read_json(run_dir / "config.json")
    accounting = _read_json(run_dir / "accounting.json")
    budget = _read_json(run_dir / "budget.json")
    qualification = _read_json(run_dir / "qualification.json")
    selection = _read_json(selection_record_path)
    metrics = _read_metrics(run_dir / "metrics.jsonl")

    _require_equal(manifest.get("mode"), "train", name="manifest mode")
    _require_equal(
        manifest.get("evidence_role"), "frozen campaign", name="evidence role"
    )
    for field in ("base_campaign_sha256", "execution_campaign_sha256"):
        _require_equal(manifest.get(field), campaign.sha256, name=field)
    _require_equal(
        manifest.get("source_policy_sha256"),
        campaign.artifact_sha256,
        name="manifest source policy",
    )
    _require_equal(
        manifest.get("release_scope_sha256"),
        release_scope.sha256,
        name="manifest release scope",
    )
    _require_equal(
        manifest.get("qualification_plan_sha256"),
        plan.sha256,
        name="manifest qualification plan",
    )
    _require_equal(
        manifest.get("qualification_command_spec_sha256"),
        spec.sha256,
        name="manifest qualification command spec",
    )
    _require_equal(
        manifest.get("requested_optimization_simulator_step_ceiling"),
        ceiling,
        name="manifest optimization ceiling",
    )
    _require_equal(
        release_scope.source_fallback_sha256,
        campaign.artifact_sha256,
        name="exact source fallback",
    )

    _require_equal(
        config.get("campaign_sha256"), campaign.sha256, name="config campaign"
    )
    _require_equal(
        config.get("release_scope_sha256"),
        release_scope.sha256,
        name="config release scope",
    )
    _require_equal(
        config.get("qualification_plan_sha256"),
        plan.sha256,
        name="config qualification plan",
    )
    backend = config.get("qualification_backend")
    if not isinstance(backend, Mapping):
        raise TypeError("config qualification_backend must be an object")
    _require_equal(
        backend.get("spec_sha256"), spec.sha256, name="qualification backend spec"
    )
    _require_equal(
        config.get("campaign_side_selection_interval"),
        plan.evaluation_interval,
        name="qualification evaluation interval",
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
        source_policy.get("trainable_scope"), "output-layer", name="policy scope"
    )
    _require_equal(
        source_policy.get("trainable_parameters"), 1806, name="trainable parameters"
    )
    _require_equal(
        campaign.optimizer.trainable_scope,
        "final-affine-weight-and-bias",
        name="campaign trainable scope",
    )
    _require_equal(
        campaign.optimizer.algorithm,
        "hyperscalees-eggroll",
        name="optimizer protocol",
    )

    _require_equal(
        qualification.get("schema"),
        QualificationController.SCHEMA,
        name="qualification schema",
    )
    _require_equal(
        qualification.get("plan_sha256"), plan.sha256, name="qualification plan"
    )
    stop_generation = _integer(
        qualification.get("stop_generation"), name="qualification stop generation"
    )
    completed = _integer(
        budget.get("completed_generations"), name="completed generations"
    )
    _require_equal(completed, stop_generation, name="integrated stop generation")
    if completed >= campaign.optimizer.generations:
        raise ValueError("campaign did not demonstrate early stopping before its limit")
    _require_equal(len(metrics), completed, name="metrics generation count")
    for expected, metric in enumerate(metrics, start=1):
        _require_equal(
            metric.get("completed_generations"), expected, name="metric order"
        )
    _require_equal(
        metrics[-1].get("qualification/status"),
        "eligible",
        name="terminal metric status",
    )
    _require_equal(
        budget.get("stopped_after_complete_release_qualification"),
        True,
        name="budget early-stop state",
    )
    _require_equal(
        budget.get("qualification"),
        qualification,
        name="budget qualification state",
    )

    attempts = qualification.get("attempts")
    if not isinstance(attempts, list) or not attempts:
        raise ValueError("qualification attempt history is absent")
    expected_generations = list(
        range(plan.evaluation_interval, completed + 1, plan.evaluation_interval)
    )
    _require_equal(
        [attempt.get("generation") for attempt in attempts],
        expected_generations,
        name="qualification attempt generations",
    )
    for attempt in attempts:
        if not isinstance(attempt, Mapping):
            raise TypeError("qualification attempt must be an object")
        generation = _integer(attempt.get("generation"), name="attempt generation")
        candidate_path = (
            run_dir / "qualification_candidates" / f"generation-{generation:06d}.npz"
        )
        if not candidate_path.is_file():
            raise FileNotFoundError(
                f"qualification candidate is missing: {candidate_path}"
            )
        _require_equal(
            _sha256_file(candidate_path),
            attempt.get("checkpoint_sha256"),
            name="qualification candidate hash",
        )
        candidate, weight, bias = load_candidate_checkpoint(
            candidate_path, campaign=campaign
        )
        _require_equal(candidate.generation, generation, name="candidate generation")
        _require_equal(weight.shape, (14, 128), name="candidate weight shape")
        _require_equal(bias.shape, (14,), name="candidate bias shape")
        stages = attempt.get("stages")
        if not isinstance(stages, list):
            raise TypeError("qualification attempt stages must be a list")
        _require_equal(
            [stage.get("stage") for stage in stages],
            list(plan.required_stages[: len(stages)]),
            name="qualification stage prefix",
        )
        _validate_stage_evidence(run_dir=run_dir, attempt=attempt, spec=spec)
    if any(attempt.get("status") == "eligible" for attempt in attempts[:-1]):
        raise ValueError("an earlier candidate was eligible but training continued")
    final_attempt = attempts[-1]
    _require_equal(final_attempt.get("status"), "eligible", name="final attempt status")
    final_stages = final_attempt.get("stages")
    if not isinstance(final_stages, list):
        raise TypeError("final qualification stages must be a list")
    _require_equal(
        [stage.get("stage") for stage in final_stages],
        list(plan.required_stages),
        name="final qualification stage order",
    )
    if any(stage.get("status") != "pass" for stage in final_stages):
        raise ValueError("the final candidate did not pass every release stage")

    if accounting.get("executed_steps_complete") is not True:
        raise ValueError("integrated run requires complete executed-step accounting")
    phase_totals = accounting.get("phase_totals")
    if not isinstance(phase_totals, Mapping):
        raise TypeError("accounting phase_totals must be an object")
    candidate_cost = _cost(
        phase_totals.get("optimization.candidates"), name="candidate optimization"
    )
    source_cost = _cost(
        phase_totals.get("optimization.source_reference"),
        name="source-reference optimization",
    )
    optimization_cost = candidate_cost + source_cost
    expected_candidates = completed * campaign.optimizer.population
    _require_equal(
        candidate_cost.candidate_evaluations,
        expected_candidates,
        name="candidate evaluations",
    )
    _require_equal(
        budget.get("candidate_evaluations"),
        expected_candidates,
        name="budget candidate evaluations",
    )
    _require_equal(
        budget.get("optimization_world_rollouts"),
        optimization_cost.world_rollouts,
        name="optimization world rollouts",
    )
    requested_steps = _integer(
        budget.get("requested_optimization_simulator_steps"),
        name="requested optimization simulator steps",
    )
    _require_equal(
        requested_steps,
        optimization_cost.requested_simulator_steps,
        name="optimization requested steps",
    )
    if requested_steps > ceiling:
        raise ValueError(
            "requested optimization simulator steps exceed ceiling: "
            f"{requested_steps} > {ceiling}"
        )
    _require_equal(
        budget.get("executed_optimization_simulator_slot_steps"),
        optimization_cost.executed_simulator_steps,
        name="executed optimization simulator steps",
    )
    _require_equal(
        metrics[-1].get("candidate_evaluations_cumulative"),
        expected_candidates,
        name="terminal metric candidate evaluations",
    )

    _require_equal(
        selection.get("schema"),
        "eggroll-autopatch-selection-export-v1",
        name="selection schema",
    )
    _require_equal(
        selection.get("campaign_sha256"),
        campaign.sha256,
        name="selection campaign",
    )
    _require_equal(
        selection.get("source_policy_sha256"),
        campaign.artifact_sha256,
        name="selection source policy",
    )
    _require_equal(
        selection.get("selected_generation"), completed, name="selected generation"
    )
    final_candidate = (
        run_dir / "qualification_candidates" / f"generation-{completed:06d}.npz"
    )
    selected_checkpoint = selection.get("selected_checkpoint")
    if not isinstance(selected_checkpoint, str):
        raise TypeError("selection record has no checkpoint path")
    recorded_checkpoint = Path(selected_checkpoint)
    expected_bundle_relative = final_candidate.relative_to(run_dir.parent)
    recorded_parts = recorded_checkpoint.parts
    expected_parts = expected_bundle_relative.parts
    path_matches = recorded_checkpoint.resolve() == final_candidate.resolve() or (
        len(recorded_parts) >= len(expected_parts)
        and recorded_parts[-len(expected_parts) :] == expected_parts
    )
    if not path_matches:
        raise ValueError(
            "selected checkpoint path mismatch: expected the current bundle path "
            f"{final_candidate.resolve()!r} or suffix {expected_bundle_relative!r}, "
            f"got {recorded_checkpoint!r}"
        )
    _require_equal(
        _sha256_file(final_candidate),
        final_attempt.get("checkpoint_sha256"),
        name="selected candidate bytes",
    )
    if not output_policy_path.is_file():
        raise FileNotFoundError("exported adapted policy is missing")
    output_policy_sha256 = _sha256_file(output_policy_path)
    _require_equal(
        selection.get("output_policy_sha256"),
        output_policy_sha256,
        name="exported policy hash",
    )
    parity = _finite_float(
        selection.get("onnx_parity_max_abs_error"), name="ONNX parity error"
    )
    parity_threshold = next(
        (
            float(gate.threshold)
            for gate in campaign.gates
            if gate.gate_id == "onnx-parity"
        ),
        None,
    )
    if parity_threshold is None or parity > parity_threshold:
        raise ValueError("exported policy does not satisfy the frozen ONNX parity gate")

    qualification_ledger = qualification.get("cost_ledger")
    if not isinstance(qualification_ledger, Mapping):
        raise TypeError("qualification cost ledger is missing")
    reconstructed_qualification_ledger = CostLedger()
    for attempt in attempts:
        selection_row = attempt.get("selection")
        if not isinstance(selection_row, Mapping):
            raise TypeError("qualification selection cost is missing")
        reconstructed_qualification_ledger.record(
            "qualification.selection",
            _cost(selection_row.get("cost"), name="qualification selection"),
        )
        for stage in attempt["stages"]:
            reconstructed_qualification_ledger.record(
                f"qualification.{stage['stage']}",
                _cost(stage.get("cost"), name="qualification stage"),
            )
    _require_equal(
        qualification_ledger,
        reconstructed_qualification_ledger.state_dict(),
        name="qualification cost ledger",
    )
    qualification_cost = CostLedger.from_state_dict(qualification_ledger).total()
    total_cost = _cost(accounting.get("total"), name="accounting total")
    failed_attempts = [
        {
            "generation": attempt["generation"],
            "status": attempt["status"],
            "rejection_reason": attempt.get("rejection_reason"),
            "stages_executed": len(attempt["stages"]),
        }
        for attempt in attempts[:-1]
    ]
    return {
        "schema": SCHEMA,
        "status": "pass",
        "evidence_role": "integrated single-job early-stop proof",
        "source_commit": manifest.get("source_commit"),
        "source_policy_sha256": campaign.artifact_sha256,
        "campaign_id": campaign.campaign_id,
        "campaign_sha256": campaign.sha256,
        "release_scope_sha256": release_scope.sha256,
        "qualification_plan_sha256": plan.sha256,
        "qualification_command_spec_sha256": spec.sha256,
        "stop_generation": completed,
        "campaign_generation_limit": campaign.optimizer.generations,
        "all_prior_attempts_noneligible": True,
        "failed_qualification_attempts": failed_attempts,
        "final_candidate_checkpoint_sha256": final_attempt.get("checkpoint_sha256"),
        "output_policy_sha256": output_policy_sha256,
        "onnx_parity_max_abs_error": parity,
        "interaction_cost": {
            "optimization": optimization_cost.to_dict(),
            "qualification_controller": qualification_cost.to_dict(),
            "total": total_cost.to_dict(),
        },
        "requested_optimization_simulator_step_ceiling": ceiling,
        "requested_optimization_simulator_steps": requested_steps,
        "claim_boundary": (
            "proves one integrated CUDA campaign stopped at its first fully qualified "
            "candidate in the production-runtime digital twin; this is not physical-"
            "robot evidence or a general optimizer-superiority claim"
        ),
    }


def write_integrated_validation(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(payload), indent=2, sort_keys=True) + "\n")
