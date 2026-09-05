from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from types import ModuleType

from mjlab_microduck.autopatch.contracts import PatchCampaign
from mjlab_microduck.autopatch.foot_proof import (
    make_walking_proof_bank,
    walking_bank_sha256,
)
from mjlab_microduck.autopatch.source_behavior import (
    capture_walking_source_behavior_reference,
)
from mjlab_microduck.autopatch.walking_protocol import (
    walking_campaign_family_sha256,
)
from mjlab_microduck.eggroll.deployment import TRUNK_COM_CALIBRATION_PROFILES

ROOT = Path(__file__).resolve().parents[1]


def _campaign(source_sha256: str) -> PatchCampaign:
    document = PatchCampaign.from_json(
        (
            ROOT
            / "docs/experiments/campaigns/walking_wedge_autopatch_release_scope_v2.json"
        ).read_text()
    ).canonical_dict()
    profile = TRUNK_COM_CALIBRATION_PROFILES[2]
    document["campaign_id"] = "alpha-walking-trunk-com-source-test-v1"
    document["optimizer"]["seed"] = 21_000_001
    document["optimizer"]["generations"] = 9
    document["artifact_sha256"] = source_sha256
    document["condition"] = {
        "condition_id": profile.name,
        "adapter": "mjlab-trunk-com-shift-profile-v1",
        "parameters": [
            ["profile_name", profile.name],
            ["profile_sha256", profile.sha256],
            ["body", "trunk_base"],
            ["offset_x_m", profile.offset_m[0]],
        ],
        "hidden_from_actor": True,
        "description": "test fixture",
    }
    document["calibration_bank_sha256"] = walking_bank_sha256(
        make_walking_proof_bank(
            base_seed=20293001,
            episodes_per_command=8,
            prefix="calibration-trunk-com",
        )
    )
    document["held_out_bank_sha256"] = walking_bank_sha256(
        make_walking_proof_bank(
            base_seed=20393004,
            episodes_per_command=8,
            prefix="selection-trunk-com",
        )
    )
    return PatchCampaign.from_dict(document)


def test_source_behavior_preflight_runs_exact_disjoint_source_banks(
    tmp_path: Path, monkeypatch
) -> None:
    runtime_repo = tmp_path / "microduck"
    policy = runtime_repo / "example_policies/alpha_walking.onnx"
    policy.parent.mkdir(parents=True)
    policy.write_bytes(b"sealed-source")
    source_sha256 = hashlib.sha256(policy.read_bytes()).hexdigest()
    campaign = _campaign(source_sha256)
    campaign_path = tmp_path / "campaign.json"
    campaign_path.write_text(json.dumps(campaign.canonical_dict()))
    profile = TRUNK_COM_CALIBRATION_PROFILES[2]
    calibration_path = tmp_path / "calibration.json"
    calibration_path.write_text(
        json.dumps(
            {
                "status": "pass",
                "selected_profile_sha256": profile.sha256,
                "source_policy_sha256": source_sha256,
                "hf_hardware_flavor": "a10g-large",
            }
        )
    )
    source_manifest_path = tmp_path / "source_manifest.json"
    source_manifest_path.write_text(
        json.dumps(
            {
                "schema": "eggroll-autopatch-source-behavior-source-v1",
                "source_commit": "a" * 40,
                "hf_hardware_flavor": "a10g-large",
                "capture_campaign_sha256": campaign.sha256,
                "campaign_family_sha256": walking_campaign_family_sha256(campaign),
                "source_policy_sha256": source_sha256,
                "calibration_validation_sha256": hashlib.sha256(
                    calibration_path.read_bytes()
                ).hexdigest(),
                "policy_episode_evaluation_ceiling": 64,
                "requested_simulator_step_ceiling": 16_000,
                "candidate_optimization_evaluation_ceiling": 0,
            }
        )
    )

    calls = []

    class Request:
        def __init__(self, **values):
            self.__dict__.update(values)

    def fake_runtime_evaluation(*, output_dir, request, profile, **_kwargs):
        calls.append((request.seed, request.device, request.command))
        output_dir.mkdir(parents=True)
        success = request.seed % 7 != 0
        result = {
            "terminal_success": success,
            "episode_steps": 250,
            "upright_fraction": 1.0 if success else 0.5,
        }
        record = {
            "schema": "eggroll-autopatch-runtime-evaluation-v1",
            "artifact": {"evaluated_sha256": source_sha256},
            "profile_sha256": profile.sha256,
            "runtime_trace_audit": {"status": "pass"},
            "result": result,
        }
        (output_dir / "manifest.json").write_text(json.dumps(record))
        return record

    evaluate = ModuleType("mjlab_microduck.autopatch.evaluate")
    evaluate.RuntimeEvaluationRequest = Request
    evaluate.run_runtime_evaluation = fake_runtime_evaluation
    monkeypatch.setitem(sys.modules, "mjlab_microduck.autopatch.evaluate", evaluate)

    output = tmp_path / "reference"
    reference = capture_walking_source_behavior_reference(
        campaign_path=campaign_path,
        calibration_validation_path=calibration_path,
        source_manifest_path=source_manifest_path,
        runtime_repo=runtime_repo,
        robotd=tmp_path / "robotd",
        ort_dylib=tmp_path / "libonnxruntime.so",
        output_dir=output,
    )

    assert len(calls) == 64
    assert {device for _seed, device, _command in calls} == {"cuda:0"}
    assert len({seed for seed, _device, _command in calls}) == 64
    assert reference["status"] == "pass"
    assert reference["cost"]["world_rollouts"] == 64
    assert reference["cost"]["requested_simulator_steps"] == 16_000
    assert reference["candidate_optimization_evaluations"] == 0
    assert reference["hf_hardware_flavor"] == "a10g-large"
    assert set(reference["stages"]) == {
        "production_runtime",
        "independent_confirmation",
    }
    assert (output / "source_behavior_reference.json").is_file()
