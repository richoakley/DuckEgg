"""Machine-verifiable proof that early stopping happened inside one campaign."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from mjlab_microduck.autopatch.campaign import save_candidate_checkpoint
from mjlab_microduck.autopatch.contracts import PatchCampaign, ReleaseScope
from mjlab_microduck.autopatch.efficiency import CostLedger, InteractionCost
from mjlab_microduck.autopatch.integrated_validation import (
    SCHEMA,
    validate_integrated_early_stop_run,
)
from mjlab_microduck.autopatch.qualification import QualificationPlan
from mjlab_microduck.autopatch.qualification_command import (
    RESULT_SCHEMA,
    TRANSCRIPT_SCHEMA,
    CommandQualificationSpec,
)

ROOT = Path(__file__).parents[1]
CAMPAIGN = (
    ROOT / "docs/experiments/campaigns/"
    "walking_wedge_autopatch_efficiency_seed4_integrated_v1.json"
)
SCOPE = (
    ROOT / "docs/experiments/release_scopes/"
    "walking_wedge_gen85_profile_specific_v1.json"
)
PLAN = ROOT / "docs/experiments/qualification_plans/walking_wedge_release_v1.json"
SPEC = (
    ROOT / "docs/experiments/qualification_plans/"
    "walking_wedge_release_command_spec_v1.json"
)


def _write(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _fixture(tmp_path: Path) -> dict[str, Path]:
    campaign = PatchCampaign.from_json(CAMPAIGN.read_text())
    scope = ReleaseScope.from_json(SCOPE.read_text())
    plan = QualificationPlan.from_json(PLAN.read_text())
    spec = CommandQualificationSpec.from_json(SPEC.read_text())
    run = tmp_path / "run"
    candidates = run / "qualification_candidates"
    checkpoints: list[Path] = []
    for generation in (1, 2):
        checkpoint = candidates / f"generation-{generation:06d}.npz"
        save_candidate_checkpoint(
            checkpoint,
            campaign=campaign,
            generation=generation,
            output_weight=np.full((14, 128), generation, dtype=np.float32),
            output_bias=np.full((14,), generation, dtype=np.float32),
            metrics={name: 1.0 for name in campaign.objective.lexicographic_metrics},
        )
        checkpoints.append(checkpoint)

    zero = InteractionCost().to_dict()
    attempts: list[dict[str, object]] = [
        {
            "generation": 1,
            "checkpoint_sha256": _sha256(checkpoints[0]),
            "selection_metrics": {name: 0.0 for name in plan.selection_metrics},
            "selection": {"status": "fail", "reason": "not plausible", "cost": zero},
            "stages": [],
            "status": "rejected",
            "rejection_reason": "not plausible",
        }
    ]
    final_stages: list[dict[str, object]] = []
    for command in spec.commands:
        result = (
            run
            / "qualification_evidence"
            / command.result_path.format(
                generation=2,
                stage=command.stage,
                candidate_checkpoint="unused",
                checkpoint_sha256=_sha256(checkpoints[1]),
                evidence_directory="unused",
            )
        )
        _write(
            result,
            {
                "schema": RESULT_SCHEMA,
                "stage": command.stage,
                "generation": 2,
                "checkpoint_sha256": _sha256(checkpoints[1]),
                "adapted_policy_sha256": "a" * 64,
                "status": "pass",
                "reason": "fixture passed",
                "cost": zero,
                "evidence": {},
            },
        )
        transcript = (
            run
            / "qualification_evidence"
            / "transcripts"
            / f"generation-000002-{command.stage}.json"
        )
        _write(
            transcript,
            {
                "schema": TRANSCRIPT_SCHEMA,
                "generation": 2,
                "checkpoint_sha256": _sha256(checkpoints[1]),
                "stage": command.stage,
                "argv": ["fixture-command", command.stage],
                "returncode": 0,
                "elapsed_wall_seconds": 1.0,
                "stdout": "",
                "stderr": "",
                "failure": None,
            },
        )
        final_stages.append(
            {
                "stage": command.stage,
                "status": "pass",
                "reason": "fixture passed",
                "evidence_sha256": _sha256(result),
                "cost": zero,
            }
        )
    attempts.append(
        {
            "generation": 2,
            "checkpoint_sha256": _sha256(checkpoints[1]),
            "selection_metrics": {name: 1.0 for name in plan.selection_metrics},
            "selection": {"status": "pass", "reason": "plausible", "cost": zero},
            "stages": final_stages,
            "status": "eligible",
        }
    )
    qualification_ledger = CostLedger()
    qualification_ledger.record("qualification.selection", InteractionCost())
    qualification_ledger.record("qualification.selection", InteractionCost())
    for stage in plan.required_stages:
        qualification_ledger.record(f"qualification.{stage}", InteractionCost())
    qualification = {
        "schema": "eggroll-autopatch-qualification-state-v1",
        "plan": plan.canonical_dict,
        "plan_sha256": plan.sha256,
        "attempts": attempts,
        "stop_generation": 2,
        "cost_ledger": qualification_ledger.state_dict(),
    }

    candidate_cost = InteractionCost(
        candidate_evaluations=1024,
        world_rollouts=4096,
        requested_simulator_steps=1_024_000,
        executed_simulator_steps=1_000_000,
        active_interaction_steps=900_000,
    )
    source_cost = InteractionCost(
        world_rollouts=8,
        requested_simulator_steps=2_000,
        executed_simulator_steps=1_900,
        active_interaction_steps=1_800,
    )
    ledger = CostLedger()
    ledger.record("optimization.candidates", candidate_cost)
    ledger.record("optimization.source_reference", source_cost)
    _write(run / "accounting.json", ledger.report())
    _write(run / "qualification.json", qualification)
    _write(
        run / "budget.json",
        {
            "candidate_evaluations": 1024,
            "optimization_world_rollouts": 4104,
            "requested_optimization_simulator_steps": 1_026_000,
            "executed_optimization_simulator_slot_steps": 1_001_900,
            "completed_generations": 2,
            "stopped_after_complete_release_qualification": True,
            "qualification": qualification,
        },
    )
    (run / "metrics.jsonl").write_text(
        json.dumps(
            {"completed_generations": 1, "candidate_evaluations_cumulative": 512}
        )
        + "\n"
        + json.dumps(
            {
                "completed_generations": 2,
                "candidate_evaluations_cumulative": 1024,
                "qualification/status": "eligible",
            }
        )
        + "\n"
    )
    _write(
        run / "config.json",
        {
            "campaign_sha256": campaign.sha256,
            "release_scope_sha256": scope.sha256,
            "qualification_plan_sha256": plan.sha256,
            "qualification_backend": {"spec_sha256": spec.sha256},
            "campaign_side_selection_interval": plan.evaluation_interval,
            "source_policy": {
                "source_sha256": campaign.artifact_sha256,
                "widths": [61, 512, 256, 128, 14],
                "trainable_scope": "output-layer",
                "trainable_parameters": 1806,
            },
        },
    )
    manifest = tmp_path / "source_manifest.json"
    _write(
        manifest,
        {
            "mode": "train",
            "evidence_role": "frozen campaign",
            "source_commit": "f" * 40,
            "source_policy_sha256": campaign.artifact_sha256,
            "base_campaign_sha256": campaign.sha256,
            "execution_campaign_sha256": campaign.sha256,
            "release_scope_sha256": scope.sha256,
            "qualification_plan_sha256": plan.sha256,
            "qualification_command_spec_sha256": spec.sha256,
            "requested_optimization_simulator_step_ceiling": 5_120_000,
        },
    )
    output_policy = tmp_path / "adapted_policy.onnx"
    output_policy.write_bytes(b"fixture-adapted-policy")
    selection = tmp_path / "selection.json"
    _write(
        selection,
        {
            "schema": "eggroll-autopatch-selection-export-v1",
            "campaign_sha256": campaign.sha256,
            "source_policy_sha256": campaign.artifact_sha256,
            "selected_generation": 2,
            "selected_checkpoint": str(checkpoints[1]),
            "output_policy_sha256": _sha256(output_policy),
            "onnx_parity_max_abs_error": 1e-7,
        },
    )
    return {
        "run": run,
        "manifest": manifest,
        "selection": selection,
        "output_policy": output_policy,
    }


def _validate(paths: dict[str, Path]) -> dict[str, object]:
    return validate_integrated_early_stop_run(
        run_dir=paths["run"],
        campaign_path=CAMPAIGN,
        release_scope_path=SCOPE,
        qualification_plan_path=PLAN,
        qualification_command_spec_path=SPEC,
        source_manifest_path=paths["manifest"],
        selection_record_path=paths["selection"],
        output_policy_path=paths["output_policy"],
        max_requested_optimization_steps=5_120_000,
    )


def test_integrated_validation_binds_stop_qualification_export_and_cost(
    tmp_path: Path,
) -> None:
    payload = _validate(_fixture(tmp_path))

    assert payload["schema"] == SCHEMA
    assert payload["status"] == "pass"
    assert payload["stop_generation"] == 2
    assert payload["all_prior_attempts_noneligible"] is True
    assert payload["requested_optimization_simulator_steps"] == 1_026_000
    assert payload["interaction_cost"]["optimization"]["world_rollouts"] == 4104


def test_integrated_validation_accepts_relocated_evidence_bundle(
    tmp_path: Path,
) -> None:
    paths = _fixture(tmp_path)
    selection = json.loads(paths["selection"].read_text())
    selection["selected_checkpoint"] = (
        "/work/run/qualification_candidates/generation-000002.npz"
    )
    _write(paths["selection"], selection)

    assert _validate(paths)["status"] == "pass"


def test_integrated_validation_rejects_wrong_relocated_checkpoint_suffix(
    tmp_path: Path,
) -> None:
    paths = _fixture(tmp_path)
    selection = json.loads(paths["selection"].read_text())
    selection["selected_checkpoint"] = (
        "/work/run/qualification_candidates/generation-000001.npz"
    )
    _write(paths["selection"], selection)

    with pytest.raises(ValueError, match="selected checkpoint path mismatch"):
        _validate(paths)


def test_integrated_validation_rejects_posthoc_or_over_budget_result(
    tmp_path: Path,
) -> None:
    paths = _fixture(tmp_path)
    selection = json.loads(paths["selection"].read_text())
    selection["selected_generation"] = 1
    _write(paths["selection"], selection)
    with pytest.raises(ValueError, match="selected generation mismatch"):
        _validate(paths)

    paths = _fixture(tmp_path / "fresh")
    budget = json.loads((paths["run"] / "budget.json").read_text())
    budget["requested_optimization_simulator_steps"] = 5_120_001
    _write(paths["run"] / "budget.json", budget)
    with pytest.raises(ValueError, match="optimization requested steps mismatch"):
        _validate(paths)


def test_integrated_validation_rejects_transcript_only_stage_failure(
    tmp_path: Path,
) -> None:
    paths = _fixture(tmp_path)
    qualification_path = paths["run"] / "qualification.json"
    qualification = json.loads(qualification_path.read_text())
    stage = qualification["attempts"][-1]["stages"][0]
    result = next(
        path
        for path in (paths["run"] / "qualification_evidence").rglob("*.json")
        if path.parent.name != "transcripts"
    )
    transcript = (
        paths["run"]
        / "qualification_evidence"
        / "transcripts"
        / f"generation-000002-{stage['stage']}.json"
    )
    stage["status"] = "fail"
    stage["reason"] = "command timed out after 1800 seconds"
    stage["evidence_sha256"] = _sha256(transcript)
    result.unlink()
    _write(qualification_path, qualification)

    budget = json.loads((paths["run"] / "budget.json").read_text())
    budget["qualification"] = qualification
    _write(paths["run"] / "budget.json", budget)

    with pytest.raises(ValueError, match="not backed by its result manifest"):
        _validate(paths)
