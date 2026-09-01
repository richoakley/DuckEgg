"""Immutable planning and checkpoint selection for generic Autopatch campaigns."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from mjlab_microduck.eggroll.policy_io import (
    export_adapted_policy,
    import_deployed_policy,
)

from .contracts import PatchCampaign
from .registry import AutopatchRegistry


@dataclass(frozen=True)
class CandidateResult:
    """One checkpoint evaluated on the campaign's fixed held-out bank."""

    checkpoint: str
    generation: int
    metrics: tuple[tuple[str, float], ...]

    def metric_map(self) -> dict[str, float]:
        return dict(self.metrics)


def select_checkpoint(
    campaign: PatchCampaign, candidates: tuple[CandidateResult, ...]
) -> CandidateResult:
    """Select lexicographically by objective metrics, never task return by default."""

    if not candidates:
        raise ValueError("checkpoint selection requires at least one candidate")
    ordered = campaign.objective.lexicographic_metrics

    def key(candidate: CandidateResult) -> tuple[float, ...]:
        values = candidate.metric_map()
        missing = [name for name in ordered if name not in values]
        if missing:
            raise ValueError(
                f"checkpoint {candidate.checkpoint!r} is missing metrics {missing}"
            )
        return tuple(values[name] for name in ordered)

    return max(candidates, key=key)


def build_campaign_plan(
    *,
    campaign: PatchCampaign,
    registry: AutopatchRegistry,
    runtime_repo: Path,
) -> dict[str, Any]:
    """Resolve every campaign selector without launching candidate evaluations."""

    registry.validate_campaign(campaign)
    artifact = registry.artifact(campaign.artifact_id)
    policy_path = runtime_repo / "example_policies" / artifact.filename
    runtime_rows = {
        row["artifact_id"]: row
        for row in registry.validate_runtime_artifacts(runtime_repo)
    }
    return {
        "schema": "eggroll-autopatch-campaign-plan-v1",
        "status": "ready-unlaunched",
        "claim_scope": "production-runtime digital twin; no physical robot",
        "campaign_id": campaign.campaign_id,
        "campaign_sha256": campaign.sha256,
        "source": runtime_rows[artifact.artifact_id],
        "source_policy": str(policy_path.resolve()),
        "capability": registry.capability(artifact.capability_id).canonical_dict(),
        "deployment_condition": campaign.condition.canonical_dict(),
        "objective": campaign.objective.canonical_dict(),
        "optimizer": campaign.optimizer.canonical_dict(),
        "release_test_plan": registry.release_test_plan(artifact.artifact_id),
        "banks": {
            "calibration_sha256": campaign.calibration_bank_sha256,
            "held_out_sha256": campaign.held_out_bank_sha256,
            "disjoint": True,
        },
        "execution_protocols": [campaign.optimizer.algorithm],
        "excluded_comparators": ["naive-es", "random-search"],
        "efficiency_claim": (
            "absolute candidate-evaluation and simulator-step accounting only; "
            "no comparative sample-efficiency claim"
        ),
        "search_authorization": (
            "not granted by planning; launching substantial training requires "
            "separate user approval"
        ),
    }


def write_campaign_plan(path: Path, plan: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(plan, indent=2, sort_keys=True) + "\n")


def save_candidate_checkpoint(
    path: Path,
    *,
    campaign: PatchCampaign,
    generation: int,
    output_weight: np.ndarray,
    output_bias: np.ndarray,
    metrics: dict[str, float],
) -> None:
    """Atomically store one campaign-bound, output-layer-only candidate."""

    if generation < 0 or not metrics:
        raise ValueError("candidate checkpoint requires generation and metrics")
    weight = np.asarray(output_weight, dtype=np.float32)
    bias = np.asarray(output_bias, dtype=np.float32)
    if weight.ndim != 2 or bias.shape != (weight.shape[0],):
        raise ValueError("candidate checkpoint output tensors have incompatible shapes")
    if not np.isfinite(weight).all() or not np.isfinite(bias).all():
        raise FloatingPointError("candidate checkpoint contains non-finite parameters")
    if not all(np.isfinite(float(value)) for value in metrics.values()):
        raise FloatingPointError("candidate checkpoint contains non-finite metrics")
    metadata = {
        "schema": "eggroll-autopatch-candidate-v1",
        "campaign_id": campaign.campaign_id,
        "campaign_sha256": campaign.sha256,
        "source_policy_sha256": campaign.artifact_sha256,
        "generation": generation,
        "metrics": {name: float(value) for name, value in sorted(metrics.items())},
        "patch_scope": "output-layer-only",
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as stream:
        np.savez_compressed(
            stream,
            metadata=np.asarray(json.dumps(metadata, sort_keys=True)),
            output_weight=weight,
            output_bias=bias,
        )
    temporary.replace(path)


def load_candidate_checkpoint(
    path: Path, *, campaign: PatchCampaign
) -> tuple[CandidateResult, np.ndarray, np.ndarray]:
    """Load a non-pickle candidate and prove its campaign/source identity."""

    with np.load(path, allow_pickle=False) as archive:
        if set(archive.files) != {"metadata", "output_weight", "output_bias"}:
            raise ValueError(f"unknown candidate checkpoint members in {path}")
        metadata = json.loads(str(archive["metadata"].item()))
        weight = np.asarray(archive["output_weight"], dtype=np.float32)
        bias = np.asarray(archive["output_bias"], dtype=np.float32)
    if metadata.get("schema") != "eggroll-autopatch-candidate-v1":
        raise ValueError("unknown Autopatch candidate schema")
    if metadata.get("campaign_sha256") != campaign.sha256:
        raise ValueError("candidate checkpoint belongs to a different campaign")
    if metadata.get("source_policy_sha256") != campaign.artifact_sha256:
        raise ValueError("candidate checkpoint belongs to different source bytes")
    if metadata.get("patch_scope") != "output-layer-only":
        raise ValueError("candidate checkpoint changed the declared patch scope")
    metrics = metadata.get("metrics")
    if not isinstance(metrics, dict):
        raise TypeError("candidate checkpoint metrics must be an object")
    candidate = CandidateResult(
        checkpoint=str(path.resolve()),
        generation=int(metadata["generation"]),
        metrics=tuple((str(name), float(value)) for name, value in metrics.items()),
    )
    return candidate, weight, bias


def select_and_export_candidate(
    *,
    campaign: PatchCampaign,
    registry: AutopatchRegistry,
    runtime_repo: Path,
    checkpoints: tuple[Path, ...],
    output_policy: Path,
) -> dict[str, Any]:
    """Select by held-out objective and export exact derivative ONNX bytes."""

    registry.validate_campaign(campaign)
    if not checkpoints:
        raise ValueError("select/export requires candidate checkpoints")
    loaded = [
        load_candidate_checkpoint(path, campaign=campaign) for path in checkpoints
    ]
    selected = select_checkpoint(campaign, tuple(row[0] for row in loaded))
    selected_row = next(row for row in loaded if row[0] == selected)
    artifact = registry.artifact(campaign.artifact_id)
    source_path = runtime_repo / "example_policies" / artifact.filename
    source = import_deployed_policy(source_path)
    if source.source_sha256 != campaign.artifact_sha256:
        raise ValueError(
            "runtime repository does not contain the campaign source bytes"
        )
    weight, bias = selected_row[1], selected_row[2]
    parity_error = export_adapted_policy(
        source,
        output_weight=weight,
        output_bias=bias,
        output_path=output_policy,
    )
    return {
        "schema": "eggroll-autopatch-selection-export-v1",
        "campaign_id": campaign.campaign_id,
        "campaign_sha256": campaign.sha256,
        "source_policy_sha256": source.source_sha256,
        "selected_checkpoint": selected.checkpoint,
        "selected_generation": selected.generation,
        "selected_metrics": selected.metric_map(),
        "selection_order": list(campaign.objective.lexicographic_metrics),
        "task_return_role": (
            "selection metric only if explicitly present in objective.lexicographic_metrics"
        ),
        "output_policy": str(output_policy.resolve()),
        "output_policy_sha256": hashlib.sha256(output_policy.read_bytes()).hexdigest(),
        "onnx_parity_max_abs_error": parity_error,
        "patch_scope": "output-layer-only",
    }
