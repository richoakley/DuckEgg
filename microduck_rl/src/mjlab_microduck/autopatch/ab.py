"""Generic paired source/derivative evaluation through the production runtime."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from mjlab_microduck.eggroll.deployment import (
    AsymmetricActuatorProfile,
    DeploymentProfile,
)

from .evaluate import RuntimeEvaluationRequest, run_runtime_evaluation, sha256_file
from .registry import AutopatchRegistry


@dataclass(frozen=True)
class ABCase:
    case_id: str
    seed: int
    task: str
    reset_label: str = "standing"
    side: str = "right"
    mode: str | None = None
    command: tuple[float, ...] = (0.0,) * 13
    horizon_steps: int = 300


def run_paired_ab_suite(
    *,
    registry: AutopatchRegistry,
    artifact_id: str,
    adapted_policy: Path,
    runtime_repo: Path,
    robotd: Path,
    ort_dylib: Path,
    profiles: tuple[tuple[str, DeploymentProfile | AsymmetricActuatorProfile], ...],
    cases: tuple[ABCase, ...],
    output_dir: Path,
    device: str = "cpu",
    record_video: bool = False,
    timeout_s: float = 30.0,
    max_attempts: int = 2,
) -> dict[str, Any]:
    """Evaluate exact paired banks; source/adapted differ only in one runtime slot."""

    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"output directory is not empty: {output_dir}")
    if not profiles or not cases:
        raise ValueError("paired evaluation requires profiles and cases")
    if max_attempts <= 0:
        raise ValueError("max_attempts must be positive")
    output_dir.mkdir(parents=True, exist_ok=True)
    artifact = registry.artifact(artifact_id)
    rows = []
    for profile_role, profile in profiles:
        for case in cases:
            pair: dict[str, Any] = {}
            for policy_role, replacement in (
                ("source", None),
                ("adapted", adapted_policy.resolve()),
            ):
                failures: list[dict[str, str | int]] = []
                record = None
                run_dir = output_dir / profile_role / case.case_id / policy_role
                for attempt in range(1, max_attempts + 1):
                    attempt_dir = run_dir.with_name(f"{policy_role}.attempt-{attempt}")
                    try:
                        record = run_runtime_evaluation(
                            registry=registry,
                            runtime_repo=runtime_repo,
                            robotd=robotd,
                            ort_dylib=ort_dylib,
                            output_dir=attempt_dir,
                            request=RuntimeEvaluationRequest(
                                artifact_id=artifact_id,
                                task=case.task,
                                seed=case.seed,
                                side=case.side,
                                command=case.command,
                                device=device,
                                record_video=record_video,
                                timeout_s=timeout_s,
                                horizon_steps=case.horizon_steps,
                                reset_label=case.reset_label,
                            ),
                            profile=profile,
                            replacement_policy=replacement,
                            mode=case.mode,
                        )
                        run_dir = attempt_dir
                        break
                    except (ConnectionError, TimeoutError, RuntimeError) as error:
                        failures.append(
                            {
                                "attempt": attempt,
                                "error_type": type(error).__name__,
                                "error": str(error),
                            }
                        )
                if record is None:
                    raise RuntimeError(
                        f"{profile_role}/{case.case_id}/{policy_role} exhausted "
                        f"{max_attempts} strict runtime attempts: {failures}"
                    )
                pair[policy_role] = {
                    "manifest": str(
                        (run_dir / "manifest.json").relative_to(output_dir)
                    ),
                    "manifest_sha256": sha256_file(run_dir / "manifest.json"),
                    "policy_sha256": record["artifact"]["evaluated_sha256"],
                    "terminal_success": bool(record["result"]["terminal_success"]),
                    "result": record["result"],
                    "rejected_runtime_attempts": failures,
                }
            if pair["source"]["policy_sha256"] != artifact.expected_sha256:
                raise ValueError("paired source run did not use the sealed artifact")
            rows.append(
                {
                    "profile_role": profile_role,
                    "profile_sha256": profile.sha256,
                    "case": asdict(case),
                    **pair,
                }
            )
    totals = {
        role: {
            policy_role: sum(
                int(row[policy_role]["terminal_success"])
                for row in rows
                if row["profile_role"] == role
            )
            for policy_role in ("source", "adapted")
        }
        for role, _profile in profiles
    }
    manifest = {
        "schema": "eggroll-autopatch-paired-ab-v1",
        "claim_scope": "production-runtime digital twin; no physical robot",
        "artifact_id": artifact_id,
        "source_sha256": artifact.expected_sha256,
        "adapted_sha256": sha256_file(adapted_policy),
        "paired_bank": [asdict(case) for case in cases],
        "profiles": [
            {
                "role": role,
                "sha256": profile.sha256,
                "profile": profile.canonical_dict(),
            }
            for role, profile in profiles
        ],
        "totals": totals,
        "task_return_role": "diagnostic-only",
        "rows": rows,
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    return manifest
