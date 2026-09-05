"""Regression tests for the CUDA smoke success boundary."""

from __future__ import annotations

import json
import pickle
from pathlib import Path

import numpy as np
import pytest

from mjlab_microduck.autopatch.contracts import PatchCampaign, ReleaseScope
from mjlab_microduck.autopatch.efficiency import CostLedger, InteractionCost
from mjlab_microduck.autopatch.smoke_validation import validate_cuda_smoke_run

ROOT = Path(__file__).parents[1]


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def _fixture(tmp_path: Path) -> tuple[Path, Path, Path, Path]:
    document = json.loads(
        (
            ROOT
            / "docs/experiments/campaigns/"
            / "walking_wedge_autopatch_efficiency_seed1_v1.json"
        ).read_text()
    )
    document["campaign_id"] += "-cuda-smoke"
    document["optimizer"]["population"] = 16
    document["optimizer"]["generations"] = 1
    campaign = PatchCampaign.from_dict(document)
    campaign_path = tmp_path / "campaign.json"
    _write_json(campaign_path, campaign.canonical_dict())

    release_scope = ReleaseScope.from_json(
        (
            ROOT
            / "docs/experiments/release_scopes/"
            / "walking_wedge_gen85_profile_specific_v1.json"
        ).read_text()
    )
    release_scope_path = tmp_path / "release_scope.json"
    _write_json(release_scope_path, release_scope.canonical_dict())

    manifest_path = tmp_path / "source_manifest.json"
    _write_json(
        manifest_path,
        {
            "mode": "smoke",
            "evidence_role": "non-evidence CUDA smoke",
            "source_commit": "f" * 40,
            "source_policy_sha256": campaign.artifact_sha256,
            "execution_campaign_sha256": campaign.sha256,
            "release_scope_sha256": release_scope.sha256,
        },
    )

    run_dir = tmp_path / "run"
    _write_json(
        run_dir / "config.json",
        {
            "campaign_sha256": campaign.sha256,
            "release_scope_sha256": release_scope.sha256,
            "source_policy": {
                "source_sha256": campaign.artifact_sha256,
                "widths": [61, 512, 256, 128, 14],
                "trainable_scope": "output-layer",
                "trainable_parameters": 1806,
            },
        },
    )

    ledger = CostLedger()
    ledger.record(
        "construction.startup_identity", InteractionCost(world_constructions=68)
    )
    ledger.record(
        "construction.training_vector_slots",
        InteractionCost(world_constructions=16),
    )
    ledger.record(
        "construction.evaluation_vector_slots",
        InteractionCost(world_constructions=1),
    )
    ledger.record(
        "source_baseline",
        InteractionCost(
            world_rollouts=64,
            requested_simulator_steps=16000,
            executed_simulator_steps=13982,
            active_interaction_steps=13982,
            policy_forward_rows=13982,
        ),
    )
    ledger.record(
        "optimization.source_reference",
        InteractionCost(
            world_rollouts=4,
            requested_simulator_steps=1000,
            executed_simulator_steps=786,
            active_interaction_steps=786,
            policy_forward_rows=786,
        ),
    )
    ledger.record(
        "optimization.candidates",
        InteractionCost(
            candidate_evaluations=16,
            world_rollouts=64,
            requested_simulator_steps=16000,
            executed_simulator_steps=16000,
            active_interaction_steps=6179,
            policy_forward_rows=16000,
        ),
    )
    ledger.record(
        "selection.nominal",
        InteractionCost(
            world_rollouts=32,
            requested_simulator_steps=8000,
            executed_simulator_steps=8000,
            active_interaction_steps=8000,
            policy_forward_rows=8000,
        ),
    )
    ledger.record(
        "selection.shifted",
        InteractionCost(
            world_rollouts=32,
            requested_simulator_steps=8000,
            executed_simulator_steps=6684,
            active_interaction_steps=6684,
            policy_forward_rows=6684,
        ),
    )
    _write_json(run_dir / "accounting.json", ledger.report())
    _write_json(
        run_dir / "budget.json",
        {
            "completed_generations": 1,
            "candidate_evaluations": 16,
            "optimization_world_rollouts": 68,
            "requested_optimization_simulator_steps": 17000,
            "executed_optimization_simulator_slot_steps": 16786,
            "active_optimization_interaction_steps": 6965,
            "comparative_sample_efficiency_claim": False,
            "world_constructions": 85,
            "wall_seconds_current_process": 930.0,
        },
    )
    metric = {
        "completed_generations": 1,
        "candidate_evaluations_cumulative": 16,
        "fitness_unique": 16,
        "fitness_mean": 0.5,
        "fitness_std": 0.307,
        "parameter_delta_norm": 0.127,
        "selection/nominal_retention_passed": True,
        "selection/release_scope_retention_passed": False,
    }
    (run_dir / "metrics.jsonl").write_text(json.dumps(metric) + "\n")
    checkpoint = {
        "schema": "eggroll-autopatch-walking-checkpoint-v2",
        "campaign_sha256": campaign.sha256,
        "source_policy_sha256": campaign.artifact_sha256,
        "next_generation": 1,
        "policy_state": {
            "params": {
                "weight": np.zeros((14, 128), dtype=np.float32),
                "bias": np.zeros(14, dtype=np.float32),
            }
        },
    }
    checkpoint_path = run_dir / "checkpoints/generation-000001.pkl"
    checkpoint_path.parent.mkdir(parents=True)
    checkpoint_bytes = pickle.dumps(checkpoint)
    checkpoint_path.write_bytes(checkpoint_bytes)
    (run_dir / "last.pkl").write_bytes(checkpoint_bytes)
    return run_dir, campaign_path, release_scope_path, manifest_path


def _validate(paths: tuple[Path, Path, Path, Path]) -> dict[str, object]:
    run_dir, campaign, release_scope, manifest = paths
    return validate_cuda_smoke_run(
        run_dir=run_dir,
        campaign_path=campaign,
        release_scope_path=release_scope,
        source_manifest_path=manifest,
        expected_candidate_evaluations=16,
        expected_optimization_world_rollouts=68,
        expected_optimization_requested_steps=17000,
        expected_total_world_rollouts=196,
        expected_total_requested_steps=49000,
    )


def test_smoke_passes_without_a_release_retained_candidate(tmp_path: Path) -> None:
    paths = _fixture(tmp_path)
    payload = _validate(paths)

    assert payload["status"] == "pass"
    assert payload["release_eligible"] is False
    assert payload["fitness_unique"] == 16
    assert payload["parameter_delta_norm"] == pytest.approx(0.127)
    assert not (paths[0] / "candidates").exists()


def test_historical_v1_smoke_preserves_scope_omission(tmp_path: Path) -> None:
    paths = _fixture(tmp_path)
    run_dir, campaign_path, _, manifest_path = paths
    document = json.loads(
        (
            ROOT / "docs/experiments/campaigns/walking_wedge_autopatch_v1.json"
        ).read_text()
    )
    document["campaign_id"] += "-cuda-smoke"
    document["optimizer"]["population"] = 16
    document["optimizer"]["generations"] = 1
    campaign = PatchCampaign.from_dict(document)
    _write_json(campaign_path, campaign.canonical_dict())

    manifest = json.loads(manifest_path.read_text())
    manifest["execution_campaign_sha256"] = campaign.sha256
    manifest["release_scope_sha256"] = None
    _write_json(manifest_path, manifest)
    config_path = run_dir / "config.json"
    config = json.loads(config_path.read_text())
    config["campaign_sha256"] = campaign.sha256
    config["release_scope_sha256"] = None
    _write_json(config_path, config)
    checkpoint_path = run_dir / "checkpoints/generation-000001.pkl"
    checkpoint = pickle.loads(checkpoint_path.read_bytes())
    checkpoint["campaign_sha256"] = campaign.sha256
    checkpoint_bytes = pickle.dumps(checkpoint)
    checkpoint_path.write_bytes(checkpoint_bytes)
    (run_dir / "last.pkl").write_bytes(checkpoint_bytes)

    payload = validate_cuda_smoke_run(
        run_dir=run_dir,
        campaign_path=campaign_path,
        release_scope_path=None,
        source_manifest_path=manifest_path,
        expected_candidate_evaluations=16,
        expected_optimization_world_rollouts=68,
        expected_optimization_requested_steps=17000,
        expected_total_world_rollouts=196,
        expected_total_requested_steps=49000,
    )

    assert payload["status"] == "pass"
    assert payload["release_scope_sha256"] is None


def test_v2_smoke_requires_release_scope(tmp_path: Path) -> None:
    paths = _fixture(tmp_path)

    with pytest.raises(ValueError, match="v2 release-scope objective requires"):
        validate_cuda_smoke_run(
            run_dir=paths[0],
            campaign_path=paths[1],
            release_scope_path=None,
            source_manifest_path=paths[3],
            expected_candidate_evaluations=16,
            expected_optimization_world_rollouts=68,
            expected_optimization_requested_steps=17000,
            expected_total_world_rollouts=196,
            expected_total_requested_steps=49000,
        )


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("fitness_unique", 1, "fitness must be non-identical"),
        ("parameter_delta_norm", 0.0, "zero EGGROLL update"),
    ],
)
def test_smoke_rejects_optimizer_invariant_failure(
    tmp_path: Path, field: str, value: float, message: str
) -> None:
    paths = _fixture(tmp_path)
    metrics_path = paths[0] / "metrics.jsonl"
    metric = json.loads(metrics_path.read_text())
    metric[field] = value
    metrics_path.write_text(json.dumps(metric) + "\n")

    with pytest.raises(ValueError, match=message):
        _validate(paths)


def test_smoke_rejects_accounting_drift(tmp_path: Path) -> None:
    paths = _fixture(tmp_path)
    accounting_path = paths[0] / "accounting.json"
    accounting = json.loads(accounting_path.read_text())
    accounting["total"]["requested_simulator_steps"] = 49001
    _write_json(accounting_path, accounting)

    with pytest.raises(ValueError, match="ledger requested steps mismatch"):
        _validate(paths)
