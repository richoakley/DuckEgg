"""Source-only calibration for the walking replacement-foot proof.

This phase deliberately performs no optimization.  It freezes a deterministic
command/seed bank and selects one predeclared geometry change that is difficult
but not catastrophic for the sealed production walking policy.
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from mjlab_microduck.eggroll.deployment import (
    NOMINAL_PROFILE,
    PRIORITY_FOOT_MATERIAL_CALIBRATION_PROFILES,
    REPLACEMENT_FOOT_CALIBRATION_PROFILES,
    REPLACEMENT_SOLE_CALIBRATION_PROFILES,
    WEDGE_FOOT_CALIBRATION_PROFILES,
)

from .evaluate import RuntimeEvaluationRequest, run_runtime_evaluation, sha256_file
from .registry import AutopatchRegistry

ARTIFACT_ID = "alpha-walking"
TASK_ID = "Mjlab-Velocity-Flat-MicroDuck"
CALIBRATION_PROTOCOL_ID = "replacement-sole-source-calibration-v1"
REPLACEMENT_FOOT_PROTOCOL_ID = "replacement-foot-source-calibration-v1"
PRIORITY_FOOT_PROTOCOL_ID = "priority-foot-material-source-calibration-v1"
WEDGE_FOOT_PROTOCOL_ID = "wedge-foot-source-calibration-v1"


@dataclass(frozen=True)
class WalkingCalibrationCase:
    """One production command and real task-reset draw."""

    case_id: str
    seed: int
    command: tuple[float, ...]
    horizon_steps: int = 250

    def __post_init__(self) -> None:
        if not self.case_id or len(self.command) != 13:
            raise ValueError("walking calibration cases require id and 13D command")
        if self.horizon_steps <= 0 or not np.isfinite(self.command).all():
            raise ValueError("walking calibration case must be finite and non-empty")


def make_walking_calibration_bank(
    base_seed: int = 20261011,
) -> tuple[WalkingCalibrationCase, ...]:
    """Build the predeclared four-command source calibration bank."""

    rng = np.random.default_rng(base_seed)
    commands = (
        (0.12, 0.00, 0.00),
        (0.20, 0.00, 0.00),
        (0.28, 0.00, 0.00),
        (0.18, 0.00, 0.30),
    )
    return tuple(
        WalkingCalibrationCase(
            case_id=f"walk-{index:02d}-{label}",
            seed=int(rng.integers(0, np.iinfo(np.int32).max)),
            command=(*command, *(0.0,) * 10),
        )
        for index, (label, command) in enumerate(
            zip(("slow", "nominal", "fast", "turn"), commands, strict=True)
        )
    )


def walking_bank_sha256(bank: tuple[WalkingCalibrationCase, ...]) -> str:
    payload = json.dumps(
        [asdict(case) for case in bank], sort_keys=True, separators=(",", ":")
    ).encode()
    return hashlib.sha256(payload).hexdigest()


def make_replacement_foot_calibration_bank(
    base_seed: int = 20261021,
) -> tuple[WalkingCalibrationCase, ...]:
    """Build v2 from commands the rejected v1 nominal run showed are active.

    The v1 result is not discarded: it established that 0.12--0.20 m/s mostly
    selects a stationary behavior in this production policy.  V2 therefore
    stays inside the task's trained range but at or above the observed 0.28 m/s
    gait onset.
    """

    rng = np.random.default_rng(base_seed)
    commands = (0.28, 0.32, 0.36, 0.40)
    return tuple(
        WalkingCalibrationCase(
            case_id=f"active-walk-{index:02d}-{speed:.2f}",
            seed=int(rng.integers(0, np.iinfo(np.int32).max)),
            command=(speed, 0.0, 0.0, *(0.0,) * 10),
        )
        for index, speed in enumerate(commands)
    )


def make_walking_proof_bank(
    *,
    base_seed: int,
    episodes_per_command: int,
    prefix: str,
) -> tuple[WalkingCalibrationCase, ...]:
    """Build a balanced active-command bank for search or held-out proof."""

    if episodes_per_command <= 0 or not prefix:
        raise ValueError("proof bank count and prefix must be non-empty")
    rng = np.random.default_rng(base_seed)
    cases: list[WalkingCalibrationCase] = []
    for speed in (0.28, 0.32, 0.36, 0.40):
        label = f"vx-{speed:.2f}"
        for index in range(episodes_per_command):
            cases.append(
                WalkingCalibrationCase(
                    case_id=f"{prefix}-{label}-{index:03d}",
                    seed=int(rng.integers(0, np.iinfo(np.int32).max)),
                    command=(speed, 0.0, 0.0, *(0.0,) * 10),
                )
            )
    return tuple(cases)


def walking_command_label(case: WalkingCalibrationCase) -> str:
    return f"vx-{case.command[0]:.2f}"


def select_replacement_sole_result(
    rows: list[dict[str, Any]],
) -> dict[str, Any] | None:
    """Select the hardest predeclared partial failure after nominal passes.

    With four cases, eligible means one to three successes.  Zero-success
    conditions are rejected as catastrophic; four-success conditions do not
    expose a binary deployment gap.  Ties prefer larger tracking error and then
    the declared profile order, never task return.
    """

    if len(rows) != 1 + len(REPLACEMENT_SOLE_CALIBRATION_PROFILES):
        raise ValueError("calibration rows do not match the predeclared profile ladder")
    nominal = rows[0]
    if nominal["profile_sha256"] != NOMINAL_PROFILE.sha256:
        raise ValueError("first calibration row must be the nominal profile")
    if int(nominal["success_count"]) < 3:
        return None
    eligible = [
        row
        for row in rows[1:]
        if 0 < int(row["success_count"]) < int(row["case_count"])
    ]
    return min(
        eligible,
        key=lambda row: (
            int(row["success_count"]),
            -float(row["mean_forward_velocity_rmse_mps"]),
        ),
        default=None,
    )


def select_replacement_foot_result(
    rows: list[dict[str, Any]],
) -> dict[str, Any] | None:
    """Apply the same partial-failure rule to the v2 hardware ladder."""

    if len(rows) != 1 + len(REPLACEMENT_FOOT_CALIBRATION_PROFILES):
        raise ValueError("calibration rows do not match the replacement-foot ladder")
    nominal = rows[0]
    if nominal["profile_sha256"] != NOMINAL_PROFILE.sha256:
        raise ValueError("first calibration row must be the nominal profile")
    if int(nominal["success_count"]) < 3:
        return None
    eligible = [
        row
        for row in rows[1:]
        if 0 < int(row["success_count"]) < int(row["case_count"])
    ]
    return min(
        eligible,
        key=lambda row: (
            int(row["success_count"]),
            -float(row["mean_forward_velocity_rmse_mps"]),
        ),
        default=None,
    )


def select_priority_foot_material_result(
    rows: list[dict[str, Any]],
) -> dict[str, Any] | None:
    """Select a partial failure only after the contact material truly changes."""

    if len(rows) != 1 + len(PRIORITY_FOOT_MATERIAL_CALIBRATION_PROFILES):
        raise ValueError("calibration rows do not match the priority-foot ladder")
    nominal = rows[0]
    if nominal["profile_sha256"] != NOMINAL_PROFILE.sha256:
        raise ValueError("first calibration row must be the nominal profile")
    if int(nominal["success_count"]) < 3:
        return None
    eligible = [
        row
        for row in rows[1:]
        if 0 < int(row["success_count"]) < int(row["case_count"])
    ]
    return min(
        eligible,
        key=lambda row: (
            int(row["success_count"]),
            -float(row["mean_forward_velocity_rmse_mps"]),
        ),
        default=None,
    )


def select_wedge_foot_result(rows: list[dict[str, Any]]) -> dict[str, Any] | None:
    """Select the hardest non-catastrophic wedge condition."""

    if len(rows) != 1 + len(WEDGE_FOOT_CALIBRATION_PROFILES):
        raise ValueError("calibration rows do not match the wedge-foot ladder")
    nominal = rows[0]
    if nominal["profile_sha256"] != NOMINAL_PROFILE.sha256:
        raise ValueError("first calibration row must be the nominal profile")
    if int(nominal["success_count"]) < 3:
        return None
    eligible = [
        row
        for row in rows[1:]
        if 0 < int(row["success_count"]) < int(row["case_count"])
    ]
    return min(
        eligible,
        key=lambda row: (
            int(row["success_count"]),
            -float(row["mean_forward_velocity_rmse_mps"]),
        ),
        default=None,
    )


def _aggregate(profile: Any, cases: list[dict[str, Any]]) -> dict[str, Any]:
    results = [row["result"] for row in cases]
    successes = sum(bool(result["terminal_success"]) for result in results)
    return {
        "profile": profile.canonical_dict(),
        "profile_sha256": profile.sha256,
        "case_count": len(cases),
        "success_count": successes,
        "terminal_success_rate": successes / len(cases),
        "mean_upright_fraction": float(
            np.mean([float(result["upright_fraction"]) for result in results])
        ),
        "mean_forward_velocity_rmse_mps": float(
            np.mean(
                [float(result["forward_velocity_rmse_mps"]) for result in results]
            )
        ),
        "mean_yaw_rate_rmse_rps": float(
            np.mean([float(result["yaw_rate_rmse_rps"]) for result in results])
        ),
        "mean_task_return_diagnostic": float(
            np.mean([float(result["total_return"]) for result in results])
        ),
        "cases": cases,
    }


def _run_source_calibration(
    *,
    registry: AutopatchRegistry,
    runtime_repo: Path,
    robotd: Path,
    ort_dylib: Path,
    output_dir: Path,
    bank: tuple[WalkingCalibrationCase, ...],
    profiles: tuple[Any, ...],
    protocol_id: str,
    selection_rule: str,
    selector: Any,
    device: str = "cpu",
    timeout_s: float = 45.0,
    max_attempts: int = 2,
) -> dict[str, Any]:
    """Evaluate only the sealed source policy and freeze a usable condition."""

    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"output directory is not empty: {output_dir}")
    if max_attempts <= 0:
        raise ValueError("max_attempts must be positive")
    output_dir.mkdir(parents=True, exist_ok=True)
    artifact = registry.artifact(ARTIFACT_ID)
    started = time.monotonic()
    rows: list[dict[str, Any]] = []
    rejected_attempts = 0
    for profile in profiles:
        case_rows: list[dict[str, Any]] = []
        for case in bank:
            failures: list[dict[str, str | int]] = []
            record = None
            run_dir = output_dir / profile.name / case.case_id
            for attempt in range(1, max_attempts + 1):
                attempt_dir = run_dir.with_name(f"{case.case_id}.attempt-{attempt}")
                try:
                    record = run_runtime_evaluation(
                        registry=registry,
                        runtime_repo=runtime_repo,
                        robotd=robotd,
                        ort_dylib=ort_dylib,
                        output_dir=attempt_dir,
                        request=RuntimeEvaluationRequest(
                            artifact_id=ARTIFACT_ID,
                            task=TASK_ID,
                            seed=case.seed,
                            command=case.command,
                            device=device,
                            timeout_s=timeout_s,
                            horizon_steps=case.horizon_steps,
                        ),
                        profile=profile,
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
                    f"{profile.name}/{case.case_id} exhausted strict runtime attempts: "
                    f"{failures}"
                )
            rejected_attempts += len(failures)
            case_rows.append(
                {
                    "case": asdict(case),
                    "manifest": str(
                        (run_dir / "manifest.json").relative_to(output_dir)
                    ),
                    "manifest_sha256": sha256_file(run_dir / "manifest.json"),
                    "runtime_trace_status": record["runtime_trace_audit"]["status"],
                    "result": record["result"],
                    "rejected_runtime_attempts": failures,
                }
            )
        rows.append(_aggregate(profile, case_rows))
    selected = selector(rows)
    requested_steps = sum(case.horizon_steps for case in bank) * len(profiles)
    manifest = {
        "schema": protocol_id,
        "status": "condition-frozen" if selected is not None else "no-eligible-condition",
        "claim_scope": "production-runtime digital twin; no physical robot",
        "phase": "source-only calibration; no optimization or training",
        "artifact_id": ARTIFACT_ID,
        "source_sha256": artifact.expected_sha256,
        "task": TASK_ID,
        "bank": [asdict(case) for case in bank],
        "bank_sha256": walking_bank_sha256(bank),
        "selection_rule": selection_rule,
        "profiles": rows,
        "selected_profile": None if selected is None else selected["profile"],
        "selected_profile_sha256": (
            None if selected is None else selected["profile_sha256"]
        ),
        "absolute_evaluation_budget": {
            "policy_episode_evaluations": len(bank) * len(profiles),
            "requested_simulator_steps": requested_steps,
            "requested_simulated_seconds": requested_steps * 0.02,
            "wall_seconds": time.monotonic() - started,
            "rejected_runtime_attempts": rejected_attempts,
            "candidate_optimization_evaluations": 0,
        },
        "task_return_role": "diagnostic-only",
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    return manifest


def run_replacement_sole_calibration(
    *,
    registry: AutopatchRegistry,
    runtime_repo: Path,
    robotd: Path,
    ort_dylib: Path,
    output_dir: Path,
    base_seed: int = 20261011,
    device: str = "cpu",
    timeout_s: float = 45.0,
    max_attempts: int = 2,
) -> dict[str, Any]:
    """Reproduce the preserved v1 geometry-only source calibration."""

    return _run_source_calibration(
        registry=registry,
        runtime_repo=runtime_repo,
        robotd=robotd,
        ort_dylib=ort_dylib,
        output_dir=output_dir,
        bank=make_walking_calibration_bank(base_seed),
        profiles=(NOMINAL_PROFILE, *REPLACEMENT_SOLE_CALIBRATION_PROFILES),
        protocol_id=CALIBRATION_PROTOCOL_ID,
        selection_rule=(
            "nominal >=3/4; choose fewest nonzero successes below 4/4; "
            "break ties by larger forward-velocity RMSE; task return excluded"
        ),
        selector=select_replacement_sole_result,
        device=device,
        timeout_s=timeout_s,
        max_attempts=max_attempts,
    )


def run_replacement_foot_calibration(
    *,
    registry: AutopatchRegistry,
    runtime_repo: Path,
    robotd: Path,
    ort_dylib: Path,
    output_dir: Path,
    base_seed: int = 20261021,
    device: str = "cpu",
    timeout_s: float = 45.0,
    max_attempts: int = 2,
) -> dict[str, Any]:
    """Calibrate a 130% lightweight foot across a predeclared grip ladder."""

    return _run_source_calibration(
        registry=registry,
        runtime_repo=runtime_repo,
        robotd=robotd,
        ort_dylib=ort_dylib,
        output_dir=output_dir,
        bank=make_replacement_foot_calibration_bank(base_seed),
        profiles=(NOMINAL_PROFILE, *REPLACEMENT_FOOT_CALIBRATION_PROFILES),
        protocol_id=REPLACEMENT_FOOT_PROTOCOL_ID,
        selection_rule=(
            "v1-validated active command range; nominal >=3/4; choose fewest "
            "nonzero successes below 4/4; break ties by larger forward-velocity "
            "RMSE; task return excluded"
        ),
        selector=select_replacement_foot_result,
        device=device,
        timeout_s=timeout_s,
        max_attempts=max_attempts,
    )


def run_priority_foot_material_calibration(
    *,
    registry: AutopatchRegistry,
    runtime_repo: Path,
    robotd: Path,
    ort_dylib: Path,
    output_dir: Path,
    base_seed: int = 20261021,
    device: str = "cpu",
    timeout_s: float = 45.0,
    max_attempts: int = 2,
) -> dict[str, Any]:
    """Calibrate the physically effective replacement-foot material ladder."""

    return _run_source_calibration(
        registry=registry,
        runtime_repo=runtime_repo,
        robotd=robotd,
        ort_dylib=ort_dylib,
        output_dir=output_dir,
        bank=make_replacement_foot_calibration_bank(base_seed),
        profiles=(NOMINAL_PROFILE, *PRIORITY_FOOT_MATERIAL_CALIBRATION_PROFILES),
        protocol_id=PRIORITY_FOOT_PROTOCOL_ID,
        selection_rule=(
            "v1-validated active command range; replacement material has contact "
            "priority; nominal >=3/4; choose fewest nonzero successes below 4/4; "
            "break ties by larger forward-velocity RMSE; task return excluded"
        ),
        selector=select_priority_foot_material_result,
        device=device,
        timeout_s=timeout_s,
        max_attempts=max_attempts,
    )


def run_wedge_foot_calibration(
    *,
    registry: AutopatchRegistry,
    runtime_repo: Path,
    robotd: Path,
    ort_dylib: Path,
    output_dir: Path,
    base_seed: int = 20261021,
    device: str = "cpu",
    timeout_s: float = 45.0,
    max_attempts: int = 2,
) -> dict[str, Any]:
    """Calibrate a fixed wedge sole without changing the task contract."""

    return _run_source_calibration(
        registry=registry,
        runtime_repo=runtime_repo,
        robotd=robotd,
        ort_dylib=ort_dylib,
        output_dir=output_dir,
        bank=make_replacement_foot_calibration_bank(base_seed),
        profiles=(NOMINAL_PROFILE, *WEDGE_FOOT_CALIBRATION_PROFILES),
        protocol_id=WEDGE_FOOT_PROTOCOL_ID,
        selection_rule=(
            "v1-validated active command range; nominal >=3/4; choose fewest "
            "nonzero successes below 4/4; break ties by larger forward-velocity "
            "RMSE; task return excluded"
        ),
        selector=select_wedge_foot_result,
        device=device,
        timeout_s=timeout_s,
        max_attempts=max_attempts,
    )
