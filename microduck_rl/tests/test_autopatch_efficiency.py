from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from mjlab_microduck.autopatch import cli
from mjlab_microduck.autopatch.contracts import PatchCampaign, ReleaseScope
from mjlab_microduck.autopatch.efficiency import (
    CostLedger,
    InteractionCost,
    PhaseProfiler,
    episode_interaction_cost,
)
from mjlab_microduck.autopatch.efficiency_study import summarize_efficiency_run
from mjlab_microduck.autopatch.qualification import (
    DEFAULT_QUALIFICATION_STAGES,
    QualificationCandidate,
    QualificationController,
    QualificationPlan,
    QualificationStageResult,
    campaign_side_gate_screen,
    stage_backend,
)

ROOT = Path(__file__).resolve().parents[1]
REFERENCE = ROOT / "docs/experiments/eggroll_autopatch_efficiency_reference_v1.json"
RELEASE_SCOPE_CAMPAIGN = (
    ROOT / "docs/experiments/campaigns/walking_wedge_autopatch_release_scope_v2.json"
)
RELEASE_SCOPE = (
    ROOT
    / "docs/experiments/release_scopes/walking_wedge_gen85_profile_specific_v1.json"
)
QUALIFICATION_PLAN = (
    ROOT / "docs/experiments/qualification_plans/walking_wedge_release_v1.json"
)
REFERENCE_CAMPAIGN = ROOT / "docs/experiments/campaigns/walking_wedge_autopatch_v1.json"


def test_training_cli_loads_and_passes_explicit_release_scope(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    observed: dict[str, object] = {}
    command_spec = tmp_path / "qualification_commands.json"
    command_spec.write_text(
        json.dumps(
            {
                "spec_id": "cli-fixture-v1",
                "commands": [
                    {
                        "stage": stage,
                        "argv": ["qualification-tool"],
                        "result_path": f"generation-{{generation}}/{stage}.json",
                        "timeout_seconds": 30.0,
                    }
                    for stage in DEFAULT_QUALIFICATION_STAGES
                ],
            }
        )
    )

    def fake_runner(**kwargs: object) -> Path:
        observed.update(kwargs)
        return tmp_path / "run"

    monkeypatch.setattr(
        "mjlab_microduck.autopatch.locomotion_trainer.run_walking_campaign",
        fake_runner,
    )
    cli.main(
        [
            "train-walking-campaign",
            "--campaign",
            str(RELEASE_SCOPE_CAMPAIGN),
            "--release-scope",
            str(RELEASE_SCOPE),
            "--qualification-plan",
            str(QUALIFICATION_PLAN),
            "--qualification-command-spec",
            str(command_spec),
            "--runtime-repo",
            str(tmp_path),
            "--output-dir",
            str(tmp_path / "run"),
        ]
    )
    assert isinstance(observed["release_scope"], ReleaseScope)
    assert isinstance(observed["qualification_plan"], QualificationPlan)
    backend = observed["qualification_backend"]
    assert isinstance(getattr(backend, "identity_sha256", None), str)
    assert '"status": "complete"' in capsys.readouterr().out


def _plan() -> QualificationPlan:
    return QualificationPlan(
        plan_id="walking-release-v1",
        evaluation_interval=5,
        selection_metrics=(
            "retained_source_success_rate",
            "repaired_source_failure_rate",
        ),
    )


def _candidate(*, generation: int = 5, passed: bool = True) -> QualificationCandidate:
    return QualificationCandidate(
        generation=generation,
        checkpoint_sha256="a" * 64,
        selection_metrics=(
            ("retained_source_success_rate", 1.0),
            ("repaired_source_failure_rate", 1.0),
            ("task_return_diagnostic", 9999.0),
        ),
        selection_passed=passed,
        selection_reason="complete campaign-side release-scope screen",
        selection_cost=InteractionCost(
            world_rollouts=32,
            requested_simulator_steps=8_000,
            executed_simulator_steps=7_900,
        ),
    )


def _stage(
    index: int, *, passed: bool = True, executed_steps: int = 500
) -> QualificationStageResult:
    return QualificationStageResult(
        stage=DEFAULT_QUALIFICATION_STAGES[index],
        status="pass" if passed else "fail",
        reason="gate passed" if passed else "gate rejected candidate",
        evidence_sha256=f"{index + 1:x}" * 64,
        cost=InteractionCost(
            world_rollouts=2,
            requested_simulator_steps=500,
            executed_simulator_steps=executed_steps,
        ),
    )


def test_frozen_reference_ties_original_interaction_budget() -> None:
    reference = json.loads(REFERENCE.read_text())
    cost = reference["original_training_cost"]
    assert cost["candidate_evaluations"] == 512 * 100
    assert cost["optimization_world_rollouts"] == 512 * 100 * 4
    assert cost["requested_optimization_simulator_steps"] == 512 * 100 * 4 * 250
    assert cost["executed_optimization_simulator_steps"] is None
    assert (
        reference["ten_x_target"]["maximum_requested_optimization_simulator_steps"]
        == 5_120_000
    )


def test_versioned_campaign_binds_release_scope_without_mutating_v1() -> None:
    campaign = PatchCampaign.from_json(RELEASE_SCOPE_CAMPAIGN.read_text())
    scope = ReleaseScope.from_json(RELEASE_SCOPE.read_text())
    assert campaign.objective.objective_id.endswith("-v2")
    assert campaign.artifact_sha256 == scope.source_fallback_sha256
    assert scope.mode == "profile_specific"
    assert scope.unknown_profile_action == "retain_source"
    plan = QualificationPlan.from_json(QUALIFICATION_PLAN.read_text())
    assert plan.evaluation_interval == 1
    assert "production_runtime" in plan.required_stages


def test_profile_specific_screen_keeps_nominal_as_diagnostic() -> None:
    campaign = PatchCampaign.from_json(RELEASE_SCOPE_CAMPAIGN.read_text())
    scope = ReleaseScope.from_json(RELEASE_SCOPE.read_text())
    rows = [
        {
            "shifted/objective/terminal_success_rate": 1.0,
            "shifted/objective/min_command_success_rate": 1.0,
            "nominal/objective/terminal_success_rate": 0.0,
        },
        {
            "shifted/objective/terminal_success_rate": 1.0,
            "shifted/objective/min_command_success_rate": 1.0,
            "nominal/objective/terminal_success_rate": 0.0,
        },
    ]

    passed, reason = campaign_side_gate_screen(
        campaign=campaign,
        metrics_history=rows,
        release_scope=scope,
    )
    historical_passed, historical_reason = campaign_side_gate_screen(
        campaign=campaign,
        metrics_history=rows,
    )

    assert passed is True
    assert "shifted-terminal-success=pass" in reason
    assert "nominal-terminal-retention" not in reason
    assert historical_passed is False
    assert "nominal-terminal-retention=fail" in historical_reason


def test_episode_accounting_separates_worlds_requested_and_executed_steps() -> None:
    episodes = [
        {"episode_steps": np.asarray([3.0, 5.0])},
        {"episode_steps": np.asarray([2.0, 5.0])},
    ]
    cost = episode_interaction_cost(
        episodes,
        requested_horizon_steps=(5, 5),
        candidate_evaluations=2,
    )
    assert cost.candidate_evaluations == 2
    assert cost.world_rollouts == 4
    assert cost.requested_simulator_steps == 20
    assert cost.executed_simulator_steps == 15


def test_vector_ticks_distinguish_simulated_slots_from_active_interactions() -> None:
    episodes = [
        {
            "episode_steps": np.asarray([2.0, 5.0]),
            "simulator_ticks": np.asarray([5.0, 5.0]),
        }
    ]
    cost = episode_interaction_cost(
        episodes,
        requested_horizon_steps=(5,),
        candidate_evaluations=2,
        physics_decimation=4,
        require_simulator_ticks=True,
    )
    assert cost.requested_simulator_steps == 10
    assert cost.executed_simulator_steps == 10
    assert cost.active_interaction_steps == 7
    assert cost.policy_forward_rows == 10
    assert cost.physics_substeps == 40


def test_interaction_cost_rejects_fractional_or_non_finite_accounting() -> None:
    with pytest.raises(TypeError, match="counts must be integers"):
        InteractionCost(candidate_evaluations=1.5)  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="executed simulator steps"):
        InteractionCost(executed_simulator_steps=1.5)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="finite and non-negative"):
        InteractionCost(wall_seconds=float("nan"))


def test_efficiency_report_never_promotes_campaign_screen_to_release(
    tmp_path: Path,
) -> None:
    campaign = PatchCampaign.from_json(REFERENCE_CAMPAIGN.read_text())
    rows = []
    for completed in range(1, 11):
        row = {
            "completed_generations": completed,
            "generation_wall_seconds": 1.0,
        }
        if completed in (5, 10):
            row.update(
                {
                    "shifted/objective/terminal_success_rate": 1.0,
                    "shifted/objective/min_command_success_rate": 1.0,
                    "nominal/objective/terminal_success_rate": 1.0,
                }
            )
        rows.append(row)
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "metrics.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rows)
    )
    report = summarize_efficiency_run(
        run_dir=run_dir,
        campaign=campaign,
        requested_steps_per_world=250,
    )
    assert report["first_campaign_side_gate_pass_generation"] == 10
    assert report["release_eligible_generation"] is None
    assert (
        report["cost_to_first_campaign_side_gate_pass"][
            "requested_optimization_simulator_steps"
        ]
        == 5_120_000
    )


def test_efficiency_report_replays_profile_specific_candidates(
    tmp_path: Path,
) -> None:
    campaign = PatchCampaign.from_json(RELEASE_SCOPE_CAMPAIGN.read_text())
    release_scope = ReleaseScope.from_json(RELEASE_SCOPE.read_text())
    rows = [
        {
            "completed_generations": completed,
            "generation_wall_seconds": 1.0,
            "interaction_accounting_cumulative": {
                "candidate_evaluations": completed * campaign.optimizer.population,
                "world_rollouts": completed * 2_052,
                "requested_simulator_steps": completed * 513_000,
                "executed_simulator_steps": completed * 512_900,
            },
            "shifted/objective/terminal_success_rate": 1.0,
            "shifted/objective/min_command_success_rate": 1.0,
            "nominal/objective/terminal_success_rate": 0.0,
            "selection/release_scope_retention_passed": True,
        }
        for completed in (1, 2)
    ]
    run_dir = tmp_path / "run"
    candidate_dir = run_dir / "qualification_candidates"
    candidate_dir.mkdir(parents=True)
    (run_dir / "metrics.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rows)
    )
    candidate = candidate_dir / "generation-000002.npz"
    candidate.write_bytes(b"candidate")

    historical = summarize_efficiency_run(
        run_dir=run_dir,
        campaign=campaign,
        requested_steps_per_world=250,
    )
    scoped = summarize_efficiency_run(
        run_dir=run_dir,
        campaign=campaign,
        requested_steps_per_world=250,
        release_scope=release_scope,
    )

    assert historical["first_campaign_side_gate_pass_generation"] is None
    assert scoped["first_campaign_side_gate_pass_generation"] == 2
    assert scoped["release_scope_sha256"] == release_scope.sha256
    assert (
        scoped["cost_to_first_campaign_side_gate_pass"][
            "requested_optimization_simulator_steps"
        ]
        == 1_026_000
    )
    assert (
        scoped["cost_to_first_campaign_side_gate_pass"][
            "executed_optimization_simulator_steps"
        ]
        == 1_025_800
    )
    assert scoped["plausible_qualification_candidates"] == [
        {
            "generation": 2,
            "checkpoint_sha256": (
                "dda18a0e21ae47c53b4309434cbc02ae8bf764fa83a6defbb719431242722aa7"
            ),
            "screen_reason": (
                "shifted-terminal-success=pass values=[1.0, 1.0]; "
                "shifted-command-coverage=pass values=[1.0, 1.0]"
            ),
        }
    ]

    output = tmp_path / "scoped-summary.json"
    cli.main(
        [
            "efficiency-report",
            "--run-dir",
            str(run_dir),
            "--campaign",
            str(RELEASE_SCOPE_CAMPAIGN),
            "--release-scope",
            str(RELEASE_SCOPE),
            "--requested-steps-per-world",
            "250",
            "--output",
            str(output),
        ]
    )
    assert (
        json.loads(output.read_text())["plausible_qualification_candidates"]
        == (scoped["plausible_qualification_candidates"])
    )


def test_cost_ledger_and_profile_resume_without_losing_state() -> None:
    ledger = CostLedger()
    ledger.record(
        "optimization.candidates",
        InteractionCost(2, 4, 20, 15, wall_seconds=1.25),
    )
    ledger.record(
        "qualification.production_runtime",
        InteractionCost(0, 2, 10, 10, wall_seconds=2.0),
    )
    restored = CostLedger.from_state_dict(ledger.state_dict())
    assert restored.total().world_rollouts == 6
    assert restored.total().requested_simulator_steps == 30
    assert restored.total().executed_simulator_steps == 25

    profiler = PhaseProfiler()
    profiler.add("physics", 1.5, calls=3)
    resumed = PhaseProfiler.from_state_dict(profiler.state_dict())
    assert resumed.state_dict() == profiler.state_dict()


def test_superficially_strong_but_regressive_checkpoint_does_not_stop() -> None:
    controller = QualificationController(_plan())
    stages = [
        _stage(0, passed=False),
    ]
    status = controller.qualify(_candidate(), stage_backend(stages))
    assert status == "rejected"
    assert not controller.should_stop
    assert "release_scope_retention" in controller.attempts[0]["rejection_reason"]


def test_failed_campaign_screen_never_invokes_expensive_backend() -> None:
    controller = QualificationController(_plan())
    invoked = False

    def backend(_candidate: QualificationCandidate, _stage_name: str):
        nonlocal invoked
        invoked = True

    status = controller.qualify(_candidate(passed=False), backend)
    assert status == "rejected"
    assert invoked is False
    assert controller.last_backend_cost == InteractionCost()


def test_campaign_pass_that_fails_production_runtime_is_rejected_and_billed() -> None:
    controller = QualificationController(_plan())
    stages = [_stage(0), _stage(1), _stage(2, passed=False, executed_steps=400)]
    called: list[str] = []

    def backend(_candidate: QualificationCandidate, stage_name: str):
        called.append(stage_name)
        return next((stage for stage in stages if stage.stage == stage_name), None)

    status = controller.qualify(_candidate(), backend)
    assert status == "rejected"
    assert not controller.should_stop
    # Selection plus every attempted stage is retained, including the failure.
    assert controller.cost.world_rollouts == 38
    assert controller.cost.requested_simulator_steps == 9_500
    assert controller.cost.executed_simulator_steps == 9_300
    assert called == list(DEFAULT_QUALIFICATION_STAGES[:3])


def test_complete_release_evidence_stops_cleanly_and_reproducibly() -> None:
    controller = QualificationController(_plan())
    status = controller.qualify(
        _candidate(),
        stage_backend([_stage(index) for index in range(6)]),
    )
    assert status == "eligible"
    assert controller.should_stop
    assert controller.stop_generation == 5
    with pytest.raises(RuntimeError, match="already established"):
        controller.qualify(_candidate(generation=10), stage_backend(()))


def test_qualification_resume_preserves_attempts_cost_and_stop_semantics() -> None:
    controller = QualificationController(_plan())
    controller.qualify(
        _candidate(generation=5),
        stage_backend([_stage(0), _stage(1), _stage(2, passed=False)]),
    )
    restored = QualificationController.from_state_dict(
        controller.state_dict(), plan=_plan()
    )
    assert restored.attempts == controller.attempts
    assert restored.cost == controller.cost
    status = restored.qualify(
        _candidate(generation=10),
        stage_backend([_stage(index) for index in range(6)]),
    )
    assert status == "eligible"
    assert restored.stop_generation == 10


def test_task_return_cannot_be_a_qualification_selection_metric() -> None:
    with pytest.raises(ValueError, match="task return"):
        QualificationPlan(
            plan_id="invalid",
            evaluation_interval=5,
            selection_metrics=("task_return_diagnostic",),
        )
