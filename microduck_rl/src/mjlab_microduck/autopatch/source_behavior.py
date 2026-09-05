"""Freeze source behavior on disjoint release banks before Autopatch training."""

from __future__ import annotations

import hashlib
import json
import math
import numbers
from dataclasses import asdict
from pathlib import Path
from typing import Any

from .contracts import PatchCampaign
from .efficiency import InteractionCost
from .registry import PRODUCTION_REGISTRY
from .walking_protocol import (
    resolve_walking_protocol,
    walking_campaign_family_sha256,
)

REFERENCE_SCHEMA = "eggroll-autopatch-source-behavior-reference-v1"
SOURCE_SCHEMA = "eggroll-autopatch-source-behavior-source-v1"


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _load_object(path: Path) -> dict[str, Any]:
    def reject_nonfinite(value: str) -> None:
        raise ValueError(f"non-finite JSON constant: {value}")

    payload = json.loads(path.read_text(), parse_constant=reject_nonfinite)
    if not isinstance(payload, dict):
        raise TypeError(f"{path} must contain one JSON object")
    return payload


def capture_walking_source_behavior_reference(
    *,
    campaign_path: Path,
    calibration_validation_path: Path,
    source_manifest_path: Path,
    runtime_repo: Path,
    robotd: Path,
    ort_dylib: Path,
    output_dir: Path,
) -> dict[str, Any]:
    """Run the sealed source once on each future release-bank case."""

    from .evaluate import RuntimeEvaluationRequest, run_runtime_evaluation
    from .qualification_runtime import _walking_ab_bank

    campaign = PatchCampaign.from_json(campaign_path.read_text())
    protocol = resolve_walking_protocol(campaign)
    if not protocol.source_behavior_reference_required:
        raise ValueError(
            "source behavior capture is only for the new incident protocol"
        )
    calibration = _load_object(calibration_validation_path)
    source_manifest = _load_object(source_manifest_path)
    if calibration.get("status") != "pass":
        raise ValueError("physical-condition calibration validation did not pass")
    if calibration.get("selected_profile_sha256") != protocol.profile.sha256:
        raise ValueError("campaign profile differs from the calibrated condition")
    if calibration.get("source_policy_sha256") != campaign.artifact_sha256:
        raise ValueError("calibration and campaign source policies differ")
    if source_manifest.get("schema") != SOURCE_SCHEMA:
        raise ValueError("unknown source-behavior bundle manifest")
    if source_manifest.get("capture_campaign_sha256") != campaign.sha256:
        raise ValueError("source-behavior bundle contains a different capture campaign")
    family_sha256 = walking_campaign_family_sha256(campaign)
    if source_manifest.get("campaign_family_sha256") != family_sha256:
        raise ValueError("source-behavior bundle changed the shared campaign contract")
    if source_manifest.get("source_policy_sha256") != campaign.artifact_sha256:
        raise ValueError("source-behavior bundle contains a different source policy")
    if source_manifest.get("calibration_validation_sha256") != _sha256_file(
        calibration_validation_path
    ):
        raise ValueError("source-behavior bundle contains different calibration bytes")
    if (
        source_manifest.get("policy_episode_evaluation_ceiling") != 64
        or source_manifest.get("requested_simulator_step_ceiling") != 16_000
        or source_manifest.get("candidate_optimization_evaluation_ceiling") != 0
    ):
        raise ValueError("source-behavior bundle changed the preflight cost ceiling")
    source_commit = source_manifest.get("source_commit")
    if not isinstance(source_commit, str) or len(source_commit) != 40:
        raise ValueError("source-behavior bundle has no full Git identity")
    hf_hardware_flavor = source_manifest.get("hf_hardware_flavor")
    if not isinstance(hf_hardware_flavor, str) or not hf_hardware_flavor:
        raise ValueError("source-behavior bundle has no HF hardware flavor")
    if calibration.get("hf_hardware_flavor") != hf_hardware_flavor:
        raise ValueError("source behavior and calibration hardware differ")
    source_path = runtime_repo / "example_policies" / "alpha_walking.onnx"
    if _sha256_file(source_path) != campaign.artifact_sha256:
        raise ValueError("runtime source policy differs from the campaign")
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"output directory is not empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    stage_records: dict[str, Any] = {}
    total_cost = InteractionCost()
    for release_bank in protocol.release_banks:
        stage = release_bank.stage
        cases = _walking_ab_bank(
            base_seed=release_bank.base_seed,
            prefix=release_bank.prefix,
        )
        if _canonical_sha256([asdict(case) for case in cases]) != (
            release_bank.ab_bank_sha256
        ):
            raise RuntimeError(f"{stage} bank generator drifted")
        rows: list[dict[str, Any]] = []
        executed_steps = 0
        for case in cases:
            case_dir = output_dir / stage / case.case_id
            record = run_runtime_evaluation(
                registry=PRODUCTION_REGISTRY,
                runtime_repo=runtime_repo,
                robotd=robotd,
                ort_dylib=ort_dylib,
                output_dir=case_dir,
                request=RuntimeEvaluationRequest(
                    artifact_id="alpha-walking",
                    task=case.task,
                    seed=case.seed,
                    side=case.side,
                    command=case.command,
                    device=release_bank.device,
                    record_video=False,
                    timeout_s=120.0,
                    horizon_steps=case.horizon_steps,
                    reset_label=case.reset_label,
                ),
                profile=protocol.profile,
                mode=case.mode,
            )
            if record.get("schema") != "eggroll-autopatch-runtime-evaluation-v1":
                raise RuntimeError("source preflight emitted the wrong runtime schema")
            if record.get("artifact", {}).get("evaluated_sha256") != (
                campaign.artifact_sha256
            ):
                raise RuntimeError("source preflight evaluated different policy bytes")
            if record.get("profile_sha256") != protocol.profile.sha256:
                raise RuntimeError("source preflight evaluated a different condition")
            if record.get("runtime_trace_audit", {}).get("status") != "pass":
                raise RuntimeError("source preflight runtime trace did not pass")
            result = record.get("result")
            if not isinstance(result, dict):
                raise TypeError("source preflight result is missing")
            steps = result.get("episode_steps")
            if (
                isinstance(steps, bool)
                or not isinstance(steps, numbers.Real)
                or not math.isfinite(float(steps))
                or not float(steps).is_integer()
                or not 0 < int(steps) <= case.horizon_steps
            ):
                raise ValueError("source preflight has invalid executed steps")
            executed_steps += int(steps)
            runtime_manifest = case_dir / "manifest.json"
            rows.append(
                {
                    "case": asdict(case),
                    "terminal_success": bool(result["terminal_success"]),
                    "result": result,
                    "runtime_manifest": str(runtime_manifest.relative_to(output_dir)),
                    "runtime_manifest_sha256": _sha256_file(runtime_manifest),
                }
            )
        failures = tuple(
            row["case"]["case_id"] for row in rows if not row["terminal_success"]
        )
        if not 0 < len(failures) < len(rows):
            raise RuntimeError(
                f"{stage} does not expose a nontrivial source failure profile"
            )
        stage_cost = InteractionCost(
            world_rollouts=32,
            requested_simulator_steps=8_000,
            executed_simulator_steps=executed_steps,
            active_interaction_steps=executed_steps,
            policy_forward_rows=executed_steps,
            physics_substeps=executed_steps * 4,
            world_constructions=32,
        )
        total_cost += stage_cost
        stage_manifest = {
            "schema": "eggroll-autopatch-source-behavior-stage-v1",
            "stage": stage,
            "source_policy_sha256": campaign.artifact_sha256,
            "activation_profile_sha256": protocol.profile.sha256,
            "device": release_bank.device,
            "ab_bank_sha256": release_bank.ab_bank_sha256,
            "rows": rows,
            "source_successes": 32 - len(failures),
            "source_failure_case_ids": list(failures),
            "cost": stage_cost.to_dict(),
        }
        stage_manifest_path = output_dir / stage / "source_behavior_manifest.json"
        _write_json(stage_manifest_path, stage_manifest)
        stage_records[stage] = {
            "device": release_bank.device,
            "ab_bank_sha256": release_bank.ab_bank_sha256,
            "source_successes": 32 - len(failures),
            "source_failure_case_ids": list(failures),
            "source_evidence_manifest": str(
                stage_manifest_path.relative_to(output_dir)
            ),
            "source_evidence_manifest_sha256": _sha256_file(stage_manifest_path),
            "cost": stage_cost.to_dict(),
        }

    if (
        total_cost.world_rollouts != 64
        or total_cost.requested_simulator_steps != 16_000
    ):
        raise RuntimeError("source preflight cost ledger is not exact")
    reference = {
        "schema": REFERENCE_SCHEMA,
        "status": "pass",
        "reference_id": f"{campaign.condition.condition_id}-source-behavior-v1",
        "source_commit": source_commit,
        "hf_hardware_flavor": hf_hardware_flavor,
        "capture_campaign_sha256": campaign.sha256,
        "campaign_family_sha256": family_sha256,
        "walking_protocol_id": protocol.protocol_id,
        "source_policy_sha256": campaign.artifact_sha256,
        "activation_profile_sha256": protocol.profile.sha256,
        "calibration_validation_sha256": _sha256_file(calibration_validation_path),
        "stages": stage_records,
        "cost": total_cost.to_dict(),
        "candidate_optimization_evaluations": 0,
        "claim_boundary": (
            "source-only production-runtime digital-twin preflight on future release "
            "banks; no candidate, training, physical-robot, or optimizer evidence"
        ),
    }
    _write_json(output_dir / "source_behavior_reference.json", reference)
    return reference
