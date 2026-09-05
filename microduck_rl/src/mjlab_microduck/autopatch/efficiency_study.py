"""Reproducible summaries for Autopatch interaction-efficiency runs."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from .contracts import PatchCampaign, ReleaseScope
from .qualification import campaign_side_gate_screen


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_metrics(path: Path) -> list[dict[str, Any]]:
    rows = [json.loads(line) for line in path.read_text().splitlines() if line]
    if not rows:
        raise ValueError("efficiency report requires at least one generation")
    for expected, row in enumerate(rows, start=1):
        if int(row.get("completed_generations", -1)) != expected:
            raise ValueError("generation metrics are incomplete or out of order")
    return rows


def summarize_efficiency_run(
    *,
    run_dir: Path,
    campaign: PatchCampaign,
    requested_steps_per_world: int,
    release_scope: ReleaseScope | None = None,
) -> dict[str, Any]:
    """Separate plausible campaign-side gates from complete release qualification."""

    if requested_steps_per_world <= 0:
        raise ValueError("requested steps per world must be positive")
    metrics_path = run_dir / "metrics.jsonl"
    rows = _read_metrics(metrics_path)
    first_campaign_side_pass: int | None = None
    screen_reason: str | None = None
    plausible_candidates: list[dict[str, Any]] = []
    for index, row in enumerate(rows, start=1):
        if not any(name.startswith("shifted/objective/") for name in row):
            continue
        passed, reason = campaign_side_gate_screen(
            campaign=campaign,
            metrics_history=rows[:index],
            release_scope=release_scope,
        )
        release_retained = release_scope is None or (
            row.get("selection/release_scope_retention_passed") is True
        )
        if passed and release_retained:
            candidate_path = (
                run_dir / "qualification_candidates" / f"generation-{index:06d}.npz"
            )
            plausible_candidates.append(
                {
                    "generation": index,
                    "checkpoint_sha256": (
                        _sha256_file(candidate_path)
                        if candidate_path.is_file()
                        else None
                    ),
                    "screen_reason": reason,
                }
            )
        if passed and release_retained and first_campaign_side_pass is None:
            first_campaign_side_pass = index
            screen_reason = reason

    qualification_path = run_dir / "qualification.json"
    qualification: dict[str, Any] | None = None
    release_eligible_generation: int | None = None
    if qualification_path.exists():
        qualification = json.loads(qualification_path.read_text())
        stop_generation = qualification.get("stop_generation")
        release_eligible_generation = (
            None if stop_generation is None else int(stop_generation)
        )

    completed = len(rows)
    candidate_evaluations = completed * campaign.optimizer.population
    optimization_world_rollouts = (
        candidate_evaluations * campaign.optimizer.worlds_per_candidate
    )
    requested_steps = optimization_world_rollouts * requested_steps_per_world
    if (run_dir / "budget.json").exists():
        budget = json.loads((run_dir / "budget.json").read_text())
        candidate_evaluations = int(
            budget.get("candidate_evaluations", candidate_evaluations)
        )
        optimization_world_rollouts = int(
            budget.get("optimization_world_rollouts", optimization_world_rollouts)
        )
        requested_steps = int(
            budget.get("requested_optimization_simulator_steps", requested_steps)
        )
    else:
        budget = None

    trigger_cost = None
    if first_campaign_side_pass is not None:
        trigger_candidates = first_campaign_side_pass * campaign.optimizer.population
        trigger_worlds = trigger_candidates * campaign.optimizer.worlds_per_candidate
        trigger_row = rows[first_campaign_side_pass - 1]
        cumulative = trigger_row.get("interaction_accounting_cumulative")
        if isinstance(cumulative, dict):
            trigger_candidates = int(
                cumulative.get("candidate_evaluations", trigger_candidates)
            )
            trigger_worlds = int(cumulative.get("world_rollouts", trigger_worlds))
            trigger_requested_steps = int(
                cumulative.get(
                    "requested_simulator_steps",
                    trigger_worlds * requested_steps_per_world,
                )
            )
            trigger_executed_steps = cumulative.get("executed_simulator_steps")
        else:
            trigger_requested_steps = trigger_worlds * requested_steps_per_world
            trigger_executed_steps = None
        trigger_cost = {
            "candidate_evaluations": trigger_candidates,
            "optimization_world_rollouts": trigger_worlds,
            "requested_optimization_simulator_steps": trigger_requested_steps,
            "executed_optimization_simulator_steps": trigger_executed_steps,
            "summed_generation_wall_seconds": sum(
                float(row.get("generation_wall_seconds", row.get("wall_time_s", 0.0)))
                for row in rows[:first_campaign_side_pass]
            ),
        }

    return {
        "schema": "eggroll-autopatch-efficiency-run-summary-v1",
        "run_dir": str(run_dir.resolve()),
        "campaign_id": campaign.campaign_id,
        "campaign_sha256": campaign.sha256,
        "release_scope_sha256": (
            None if release_scope is None else release_scope.sha256
        ),
        "metrics_sha256": _sha256_file(metrics_path),
        "completed_generations": completed,
        "training_cost": {
            "candidate_evaluations": candidate_evaluations,
            "optimization_world_rollouts": optimization_world_rollouts,
            "requested_optimization_simulator_steps": requested_steps,
            "executed_optimization_simulator_steps": (
                None
                if budget is None
                else budget.get("executed_optimization_simulator_slot_steps")
            ),
        },
        "first_campaign_side_gate_pass_generation": first_campaign_side_pass,
        "first_campaign_side_gate_pass_reason": screen_reason,
        "plausible_qualification_candidates": plausible_candidates,
        "cost_to_first_campaign_side_gate_pass": trigger_cost,
        "release_eligible_generation": release_eligible_generation,
        "qualification_state_sha256": (
            None if qualification is None else _sha256_file(qualification_path)
        ),
        "claim_boundary": (
            "campaign-side gate passage is only a qualification trigger; release "
            "eligibility requires every predeclared stage in qualification.json"
        ),
    }


def write_efficiency_summary(path: Path, summary: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
