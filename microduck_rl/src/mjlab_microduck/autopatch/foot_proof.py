"""Source-only calibration for the walking replacement-foot proof.

This phase deliberately performs no optimization.  It freezes a deterministic
command/seed bank and selects one predeclared geometry change that is difficult
but not catastrophic for the sealed production walking policy.
"""

from __future__ import annotations

import hashlib
import json
import math
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
    TRUNK_COM_CALIBRATION_PROFILES,
    TRUNK_PAYLOAD_CALIBRATION_PROFILES,
    WEDGE_FOOT_CALIBRATION_PROFILES,
)

from .registry import AutopatchRegistry
from .runtime_trace import robotio_write_coverage_contract

ARTIFACT_ID = "alpha-walking"
TASK_ID = "Mjlab-Velocity-Flat-MicroDuck"
CALIBRATION_PROTOCOL_ID = "replacement-sole-source-calibration-v1"
REPLACEMENT_FOOT_PROTOCOL_ID = "replacement-foot-source-calibration-v1"
PRIORITY_FOOT_PROTOCOL_ID = "priority-foot-material-source-calibration-v1"
WEDGE_FOOT_PROTOCOL_ID = "wedge-foot-source-calibration-v1"
TRUNK_COM_PROTOCOL_ID = "trunk-com-source-calibration-v1"
TRUNK_PAYLOAD_PROTOCOL_ID = "trunk-payload-source-calibration-v1"


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


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


def select_trunk_com_result(rows: list[dict[str, Any]]) -> dict[str, Any] | None:
    """Select the hardest broad, recoverable shift from the fixed 32-case ladder."""

    if len(rows) != 1 + len(TRUNK_COM_CALIBRATION_PROFILES):
        raise ValueError("calibration rows do not match the trunk-CoM ladder")
    nominal = rows[0]
    if nominal["profile_sha256"] != NOMINAL_PROFILE.sha256:
        raise ValueError("first calibration row must be the nominal profile")
    if int(nominal["success_count"]) < 31 or int(nominal["case_count"]) != 32:
        return None
    eligible = [
        row
        for row in rows[1:]
        if 16 <= int(row["success_count"]) <= 28
        and set(row.get("command_success_counts", {}).values())
        and min(row["command_success_counts"].values()) >= 2
    ]
    return min(
        eligible,
        key=lambda row: (
            int(row["success_count"]),
            -float(row["mean_forward_velocity_rmse_mps"]),
        ),
        default=None,
    )


def select_trunk_payload_result(rows: list[dict[str, Any]]) -> dict[str, Any] | None:
    """Select the hardest broad, recoverable payload from the fixed ladder."""

    if len(rows) != 1 + len(TRUNK_PAYLOAD_CALIBRATION_PROFILES):
        raise ValueError("calibration rows do not match the trunk-payload ladder")
    nominal = rows[0]
    if nominal["profile_sha256"] != NOMINAL_PROFILE.sha256:
        raise ValueError("first calibration row must be the nominal profile")
    if int(nominal["success_count"]) < 31 or int(nominal["case_count"]) != 32:
        return None
    eligible = [
        row
        for row in rows[1:]
        if 16 <= int(row["success_count"]) <= 28
        and set(row.get("command_success_counts", {}).values())
        and min(row["command_success_counts"].values()) >= 2
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
    command_success_counts: dict[str, int] = {}
    for row in cases:
        command = row["case"]["command"]
        label = f"vx-{float(command[0]):.2f}"
        command_success_counts[label] = command_success_counts.get(label, 0) + int(
            bool(row["result"]["terminal_success"])
        )
    return {
        "profile": profile.canonical_dict(),
        "profile_sha256": profile.sha256,
        "case_count": len(cases),
        "success_count": successes,
        "terminal_success_rate": successes / len(cases),
        "command_success_counts": command_success_counts,
        "mean_upright_fraction": float(
            np.mean([float(result["upright_fraction"]) for result in results])
        ),
        "mean_forward_velocity_rmse_mps": float(
            np.mean([float(result["forward_velocity_rmse_mps"]) for result in results])
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

    from .evaluate import RuntimeEvaluationRequest, run_runtime_evaluation

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
                    "manifest_sha256": _sha256_file(run_dir / "manifest.json"),
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
        "status": "condition-frozen"
        if selected is not None
        else "no-eligible-condition",
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
        "runtime_trace_gate": robotio_write_coverage_contract(),
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


def run_trunk_com_calibration(
    *,
    registry: AutopatchRegistry,
    runtime_repo: Path,
    robotd: Path,
    ort_dylib: Path,
    output_dir: Path,
    base_seed: int = 20293001,
    device: str = "cuda:0",
    timeout_s: float = 45.0,
    max_attempts: int = 1,
) -> dict[str, Any]:
    """Calibrate the fixed forward trunk-CoM ladder on 32 source-only cases."""

    return _run_source_calibration(
        registry=registry,
        runtime_repo=runtime_repo,
        robotd=robotd,
        ort_dylib=ort_dylib,
        output_dir=output_dir,
        bank=make_walking_proof_bank(
            base_seed=base_seed,
            episodes_per_command=8,
            prefix="calibration-trunk-com",
        ),
        profiles=(NOMINAL_PROFILE, *TRUNK_COM_CALIBRATION_PROFILES),
        protocol_id=TRUNK_COM_PROTOCOL_ID,
        selection_rule=(
            "fixed 32-case active-command bank; nominal >=31/32; selected shift "
            "must score 16..28/32 with >=2/8 successes at every command; choose "
            "fewest successes then larger velocity RMSE; task return excluded"
        ),
        selector=select_trunk_com_result,
        device=device,
        timeout_s=timeout_s,
        max_attempts=max_attempts,
    )


def run_trunk_payload_calibration(
    *,
    registry: AutopatchRegistry,
    runtime_repo: Path,
    robotd: Path,
    ort_dylib: Path,
    output_dir: Path,
    base_seed: int = 20794001,
    device: str = "cuda:0",
    timeout_s: float = 45.0,
    max_attempts: int = 1,
) -> dict[str, Any]:
    """Calibrate the frozen trunk-payload ladder on 32 source-only cases."""

    return _run_source_calibration(
        registry=registry,
        runtime_repo=runtime_repo,
        robotd=robotd,
        ort_dylib=ort_dylib,
        output_dir=output_dir,
        bank=make_walking_proof_bank(
            base_seed=base_seed,
            episodes_per_command=8,
            prefix="calibration-trunk-payload",
        ),
        profiles=(NOMINAL_PROFILE, *TRUNK_PAYLOAD_CALIBRATION_PROFILES),
        protocol_id=TRUNK_PAYLOAD_PROTOCOL_ID,
        selection_rule=(
            "fixed 32-case active-command bank; nominal >=31/32; selected payload "
            "must score 16..28/32 with >=2/8 successes at every command; choose "
            "fewest successes then larger velocity RMSE; task return excluded"
        ),
        selector=select_trunk_payload_result,
        device=device,
        timeout_s=timeout_s,
        max_attempts=max_attempts,
    )


def validate_trunk_com_calibration(
    *, manifest_path: Path, protocol_path: Path, source_manifest_path: Path
) -> dict[str, Any]:
    """Validate source-only calibration identities, evidence, cost, and selection."""

    def reject_nonfinite(value: str) -> None:
        raise ValueError(f"non-finite JSON constant: {value}")

    manifest = json.loads(manifest_path.read_text(), parse_constant=reject_nonfinite)
    protocol = json.loads(protocol_path.read_text(), parse_constant=reject_nonfinite)
    source_manifest = json.loads(
        source_manifest_path.read_text(), parse_constant=reject_nonfinite
    )
    if not all(
        isinstance(value, dict) for value in (manifest, protocol, source_manifest)
    ):
        raise TypeError("calibration, protocol, and source manifests must be objects")
    protocol_schema = protocol.get("schema")
    is_payload = protocol_schema == (
        "eggroll-autopatch-payload-cross-failure-protocol-v1"
    )
    if is_payload:
        source_manifest_schema = "eggroll-autopatch-trunk-payload-calibration-source-v1"
        manifest_schema = TRUNK_PAYLOAD_PROTOCOL_ID
        expected_protocol_id = "alpha-walking-trunk-payload-cross-failure-v1"
        expected_protocol_status = "predeclared-before-calibration"
        expected_profiles = (NOMINAL_PROFILE, *TRUNK_PAYLOAD_CALIBRATION_PROFILES)
        calibration_base_seed = 20794001
        calibration_prefix = "calibration-trunk-payload"
        selection_rule = (
            "fixed 32-case active-command bank; nominal >=31/32; selected payload "
            "must score 16..28/32 with >=2/8 successes at every command; choose "
            "fewest successes then larger velocity RMSE; task return excluded"
        )
        expected_condition = {
            "adapter": "mjlab-trunk-payload-profile-v1",
            "body": "trunk_base",
            "operation": "add_to_seeded_startup_mass_and_scale_inertia",
            "payload_inertia_model": (
                "uniform-density-scale-of-seeded-trunk-pseudo-inertia"
            ),
            "physical_basis": {
                "nominal_robot_mass_kg_approx": 0.8,
                "nominal_trunk_mass_kg": 0.199224,
                "largest_added_payload_fraction_of_robot_approx": 0.25,
                "largest_added_payload_fraction_of_nominal_trunk": (1.0038951150463793),
            },
            "hidden_from_actor": True,
            "body_mass_kg_changed": True,
            "body_inertia_tensor_changed": True,
            "body_inertial_position_changed": False,
            "profiles": [
                {
                    "name": profile.name,
                    "added_mass_kg": profile.added_mass_kg,
                    "sha256": profile.sha256,
                }
                for profile in TRUNK_PAYLOAD_CALIBRATION_PROFILES
            ],
        }
        selector = select_trunk_payload_result
        validation_schema = "eggroll-autopatch-trunk-payload-calibration-validation-v1"
        claim_condition = "trunk-payload"
    elif protocol_schema == "eggroll-autopatch-cross-failure-protocol-v2":
        source_manifest_schema = "eggroll-autopatch-trunk-com-calibration-source-v1"
        manifest_schema = TRUNK_COM_PROTOCOL_ID
        expected_protocol_id = "alpha-walking-trunk-com-cross-failure-v2"
        expected_protocol_status = "predeclared-before-restarted-calibration"
        expected_profiles = (NOMINAL_PROFILE, *TRUNK_COM_CALIBRATION_PROFILES)
        calibration_base_seed = 20293001
        calibration_prefix = "calibration-trunk-com"
        selection_rule = (
            "fixed 32-case active-command bank; nominal >=31/32; selected shift "
            "must score 16..28/32 with >=2/8 successes at every command; choose "
            "fewest successes then larger velocity RMSE; task return excluded"
        )
        expected_condition = {
            "adapter": "mjlab-trunk-com-shift-profile-v1",
            "body": "trunk_base",
            "axis": "local-forward-x",
            "operation": "add_to_seeded_startup_body_ipos",
            "hidden_from_actor": True,
            "body_mass_kg_changed": False,
            "body_inertia_tensor_changed": False,
            "body_inertial_position_changed": True,
            "profiles": [
                {
                    "name": profile.name,
                    "offset_m": list(profile.offset_m),
                    "sha256": profile.sha256,
                }
                for profile in TRUNK_COM_CALIBRATION_PROFILES
            ],
        }
        selector = select_trunk_com_result
        validation_schema = "eggroll-autopatch-trunk-com-calibration-validation-v1"
        claim_condition = "trunk-CoM"
    else:
        raise ValueError("unknown cross-failure protocol schema")
    protocol_canonical_sha256 = hashlib.sha256(
        json.dumps(protocol, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    if source_manifest.get("schema") != source_manifest_schema:
        raise ValueError("unknown calibration source manifest schema")
    if source_manifest.get("protocol_sha256") != _sha256_file(protocol_path):
        raise ValueError("calibration source bundle changed the raw protocol bytes")
    if source_manifest.get("protocol_canonical_sha256") != protocol_canonical_sha256:
        raise ValueError("calibration source bundle used a different protocol")
    if source_manifest.get("source_policy_sha256") != manifest.get("source_sha256"):
        raise ValueError("calibration source bundle used different policy bytes")
    if (
        source_manifest.get("policy_episode_evaluation_ceiling") != 160
        or source_manifest.get("requested_simulator_step_ceiling") != 40_000
        or source_manifest.get("candidate_optimization_evaluation_ceiling") != 0
    ):
        raise ValueError("calibration source bundle changed the absolute cost ceiling")
    source_commit = source_manifest.get("source_commit")
    if not isinstance(source_commit, str) or len(source_commit) != 40:
        raise ValueError("calibration source commit is not a full Git identity")
    hf_hardware_flavor = source_manifest.get("hf_hardware_flavor")
    if not isinstance(hf_hardware_flavor, str) or not hf_hardware_flavor:
        raise ValueError("calibration source manifest has no HF hardware flavor")
    if manifest.get("schema") != manifest_schema:
        raise ValueError("unknown physical-condition calibration schema")
    if manifest.get("phase") != "source-only calibration; no optimization or training":
        raise ValueError("calibration phase changed")
    if manifest.get("task_return_role") != "diagnostic-only":
        raise ValueError("calibration used task return as a decision metric")
    if manifest.get("selection_rule") != selection_rule:
        raise ValueError("calibration selection rule changed")
    if protocol.get("protocol_id") != expected_protocol_id:
        raise ValueError("unexpected cross-failure protocol id")
    if protocol.get("status") != expected_protocol_status:
        raise ValueError("cross-failure protocol was not predeclared")
    if not is_payload:
        amendment = protocol.get("amendment")
        if not isinstance(amendment, dict):
            raise TypeError("cross-failure protocol amendment must be an object")
        if amendment.get("prior_protocol_sha256") != (
            "ac4261f24253bbf54b5cc62cca1a7ac574553ef9c7ceb42fea409e13dacfa2f3"
        ):
            raise ValueError("cross-failure protocol does not bind the failed v1 bytes")
        if amendment.get("change_scope") != (
            "runtime-trace transport coverage classification only"
        ):
            raise ValueError("cross-failure protocol amendment scope changed")
        if amendment.get("restart_rule") != (
            "discard all partial v1 calibration rows and rerun all 160 source-only "
            "episodes under v2"
        ):
            raise ValueError("cross-failure calibration restart rule changed")
    trace_contract = robotio_write_coverage_contract()
    if protocol.get("runtime_trace_gate") != trace_contract:
        raise ValueError("cross-failure runtime-trace gate changed")
    if manifest.get("runtime_trace_gate") != trace_contract:
        raise ValueError("calibration manifest used a different runtime-trace gate")
    source = protocol.get("source")
    if not isinstance(source, dict):
        raise TypeError("cross-failure protocol source must be an object")
    expected_source = {
        "artifact_id": ARTIFACT_ID,
        "policy_sha256": source.get("policy_sha256"),
        "actor_widths": [61, 512, 256, 128, 14],
        "trainable_scope": "final-affine-weight-and-bias",
        "trainable_parameters": 1806,
    }
    if source != expected_source:
        raise ValueError("cross-failure source contract changed")
    if manifest.get("source_sha256") != source["policy_sha256"]:
        raise ValueError("calibration source policy differs from the protocol")
    if manifest.get("artifact_id") != ARTIFACT_ID or manifest.get("task") != TASK_ID:
        raise ValueError("calibration artifact or task identity changed")

    condition = protocol.get("condition")
    if not isinstance(condition, dict):
        raise TypeError("cross-failure condition must be an object")
    if condition != expected_condition:
        raise ValueError("cross-failure condition ladder changed")

    calibration = protocol.get("calibration")
    if not isinstance(calibration, dict):
        raise TypeError("cross-failure calibration must be an object")
    expected_bank_rows = [
        {
            "case_id": case.case_id,
            "seed": case.seed,
            "command": list(case.command),
            "horizon_steps": case.horizon_steps,
        }
        for case in make_walking_proof_bank(
            base_seed=calibration_base_seed,
            episodes_per_command=8,
            prefix=calibration_prefix,
        )
    ]
    expected_bank_sha256 = hashlib.sha256(
        json.dumps(expected_bank_rows, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    if calibration != {
        "source_only": True,
        "base_seed": calibration_base_seed,
        "prefix": calibration_prefix,
        "cases": 32,
        "episodes_per_command": 8,
        "walking_case_bank_sha256": expected_bank_sha256,
        "selection_rule": {
            "nominal_minimum_successes": 31,
            "shifted_minimum_successes": 16,
            "shifted_maximum_successes": 28,
            "minimum_successes_per_command": 2,
            "order": [
                "fewest_terminal_successes",
                "largest_forward_velocity_rmse",
            ],
            "task_return_role": "diagnostic-only",
        },
        "no_eligible_condition_action": "stop-and-record-negative-result",
    }:
        raise ValueError("cross-failure calibration contract changed")
    bank = manifest.get("bank")
    if bank != expected_bank_rows:
        raise ValueError("trunk-CoM calibration cases differ from the predeclaration")
    bank_sha256 = hashlib.sha256(
        json.dumps(bank, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    if (
        manifest.get("bank_sha256") != bank_sha256
        or bank_sha256 != expected_bank_sha256
    ):
        raise ValueError("calibration bank bytes differ from the predeclaration")

    rows = manifest.get("profiles")
    if not isinstance(rows, list):
        raise TypeError("calibration profile rows must be a list")
    if not all(isinstance(row, dict) for row in rows):
        raise TypeError("calibration profile rows must be objects")
    if [row["profile_sha256"] for row in rows] != [
        profile.sha256 for profile in expected_profiles
    ]:
        raise ValueError("calibration profile ladder differs from the predeclaration")
    for row, profile in zip(rows, expected_profiles, strict=True):
        if row.get("profile") != profile.canonical_dict():
            raise ValueError("calibration profile bytes differ from the implementation")
        cases = row.get("cases")
        if not isinstance(cases, list) or not all(
            isinstance(case, dict) for case in cases
        ):
            raise TypeError("calibration cases must be objects")
        if [case["case"] for case in cases] != bank:
            raise ValueError("calibration profiles did not use identical case bytes")
        successes = 0
        command_success_counts = {
            f"vx-{speed:.2f}": 0 for speed in (0.28, 0.32, 0.36, 0.40)
        }
        aggregate_values = {
            "mean_upright_fraction": [],
            "mean_forward_velocity_rmse_mps": [],
            "mean_yaw_rate_rmse_rps": [],
            "mean_task_return_diagnostic": [],
        }
        for case in cases:
            if case.get("runtime_trace_status") != "pass":
                raise ValueError(
                    "calibration contains a failed production-runtime trace"
                )
            relative = case.get("manifest")
            expected_sha256 = case.get("manifest_sha256")
            if not isinstance(relative, str) or not isinstance(expected_sha256, str):
                raise TypeError("calibration case does not bind runtime evidence")
            evidence = (manifest_path.parent / relative).resolve()
            if not evidence.is_relative_to(manifest_path.parent.resolve()):
                raise ValueError("calibration evidence path escapes its artifact root")
            if not evidence.is_file() or _sha256_file(evidence) != expected_sha256:
                raise ValueError(
                    "calibration runtime evidence bytes are missing or changed"
                )
            evidence_payload = json.loads(
                evidence.read_text(), parse_constant=reject_nonfinite
            )
            if not isinstance(evidence_payload, dict):
                raise TypeError("runtime evidence manifest must be an object")
            result = case.get("result")
            if not isinstance(result, dict) or not isinstance(
                result.get("terminal_success"), bool
            ):
                raise TypeError("calibration result must contain a boolean outcome")
            case_spec = case["case"]
            if (
                evidence_payload.get("schema")
                != "eggroll-autopatch-runtime-evaluation-v1"
            ):
                raise ValueError("calibration evidence has the wrong runtime schema")
            if (
                evidence_payload.get("artifact", {}).get("evaluated_sha256")
                != manifest["source_sha256"]
            ):
                raise ValueError(
                    "calibration evidence evaluated different policy bytes"
                )
            if evidence_payload.get("profile_sha256") != profile.sha256:
                raise ValueError("calibration evidence used a different profile")
            scenario = evidence_payload.get("scenario")
            if not isinstance(scenario, dict) or (
                scenario.get("task") != TASK_ID
                or scenario.get("seed") != case_spec["seed"]
                or scenario.get("command") != case_spec["command"]
            ):
                raise ValueError("calibration evidence used a different scenario")
            trace_audit = evidence_payload.get("runtime_trace_audit")
            if not isinstance(trace_audit, dict) or trace_audit.get("status") != "pass":
                raise ValueError("runtime evidence trace audit did not pass")
            coverage = trace_audit.get("robotio_write_coverage")
            if not isinstance(coverage, dict):
                raise TypeError("runtime evidence lacks RobotIo write coverage")
            if any(coverage.get(key) != value for key, value in trace_contract.items()):
                raise ValueError(
                    "runtime evidence used a different write-coverage rule"
                )
            applied_frames = coverage.get("applied_target_frames")
            unapplied_frames = coverage.get("unapplied_target_frames")
            captured_frames = trace_audit.get("captured_policy_frames")
            observed_coverage = coverage.get("applied_target_coverage")
            if any(
                isinstance(value, bool) or not isinstance(value, int)
                for value in (applied_frames, unapplied_frames, captured_frames)
            ):
                raise TypeError(
                    "runtime evidence write-coverage counts must be integers"
                )
            if (
                applied_frames < 1
                or unapplied_frames < 0
                or captured_frames != applied_frames + unapplied_frames
            ):
                raise ValueError(
                    "runtime evidence write-coverage counts do not tie out"
                )
            if not isinstance(observed_coverage, int | float) or not math.isclose(
                float(observed_coverage),
                applied_frames / captured_frames,
                abs_tol=1.0e-12,
            ):
                raise ValueError(
                    "runtime evidence write-coverage ratio does not tie out"
                )
            if (
                float(observed_coverage)
                < trace_contract["minimum_applied_target_coverage"]
            ):
                raise ValueError("runtime evidence is below the frozen coverage floor")
            recovered_ticks = coverage.get("recovered_robotio_write_failure_ticks")
            closing_ticks = coverage.get("post_episode_write_failure_ticks")
            if not all(
                isinstance(ticks, list) and all(isinstance(tick, int) for tick in ticks)
                for ticks in (recovered_ticks, closing_ticks)
            ):
                raise TypeError(
                    "runtime evidence write-failure ticks must be integer lists"
                )
            if len(recovered_ticks) + len(closing_ticks) != unapplied_frames:
                raise ValueError("runtime evidence write-failure ticks do not tie out")
            if trace_audit.get("recovered_robotio_write_failures") != len(
                recovered_ticks
            ) or trace_audit.get("post_episode_write_failures") != len(closing_ticks):
                raise ValueError("runtime evidence write-failure counts do not tie out")
            if evidence_payload.get("result") != result:
                raise ValueError(
                    "calibration aggregate result differs from its evidence"
                )

            success = result["terminal_success"]
            successes += int(success)
            command = f"vx-{float(case_spec['command'][0]):.2f}"
            command_success_counts[command] += int(success)
            for aggregate, result_key in (
                ("mean_upright_fraction", "upright_fraction"),
                ("mean_forward_velocity_rmse_mps", "forward_velocity_rmse_mps"),
                ("mean_yaw_rate_rmse_rps", "yaw_rate_rmse_rps"),
                ("mean_task_return_diagnostic", "total_return"),
            ):
                value = result.get(result_key)
                if not isinstance(value, int | float) or not math.isfinite(value):
                    raise TypeError("calibration result metrics must be finite numbers")
                aggregate_values[aggregate].append(float(value))
        if (
            row.get("case_count") != 32
            or row.get("success_count") != successes
            or row.get("command_success_counts") != command_success_counts
            or not math.isclose(
                float(row.get("terminal_success_rate", math.nan)),
                successes / 32,
                abs_tol=1.0e-12,
            )
        ):
            raise ValueError(
                "calibration outcome aggregates do not match case evidence"
            )
        for aggregate, values in aggregate_values.items():
            if not math.isclose(
                float(row.get(aggregate, math.nan)),
                float(np.mean(values)),
                abs_tol=1.0e-12,
            ):
                raise ValueError(
                    "calibration metric aggregate differs from case evidence"
                )

    selected = selector(rows)
    expected_status = (
        "condition-frozen" if selected is not None else "no-eligible-condition"
    )
    if manifest.get("status") != expected_status:
        raise ValueError("calibration status disagrees with the frozen selection rule")
    selected_sha256 = None if selected is None else selected["profile_sha256"]
    if manifest.get("selected_profile_sha256") != selected_sha256:
        raise ValueError("calibration selected a different profile")
    budget = manifest.get("absolute_evaluation_budget")
    if not isinstance(budget, dict):
        raise TypeError("calibration evaluation budget is missing")
    if budget.get("policy_episode_evaluations") != 160:
        raise ValueError("calibration did not execute the exact 160 source episodes")
    if budget.get("requested_simulator_steps") != 40_000:
        raise ValueError("calibration requested-step total is not exact")
    if budget.get("candidate_optimization_evaluations") != 0:
        raise ValueError("calibration unexpectedly performed candidate optimization")
    if budget.get("rejected_runtime_attempts") != 0:
        raise ValueError("calibration contains unaccounted rejected runtime attempts")
    return {
        "schema": validation_schema,
        "status": "pass",
        "calibration_status": expected_status,
        "source_policy_sha256": manifest["source_sha256"],
        "protocol_sha256": _sha256_file(protocol_path),
        "protocol_canonical_sha256": protocol_canonical_sha256,
        "source_commit": source_commit,
        "hf_hardware_flavor": hf_hardware_flavor,
        "bank_sha256": bank_sha256,
        "profile_sha256s": [profile.sha256 for profile in expected_profiles],
        "selected_profile_sha256": selected_sha256,
        "policy_episode_evaluations": 160,
        "requested_simulator_steps": 40_000,
        "candidate_optimization_evaluations": 0,
        "claim_boundary": (
            f"source-only {claim_condition} production-runtime digital-twin "
            "calibration; no training, "
            "candidate result, physical-robot evidence, or optimizer comparison"
        ),
    }


def validate_trunk_payload_calibration(
    *, manifest_path: Path, protocol_path: Path, source_manifest_path: Path
) -> dict[str, Any]:
    """Validate the payload calibration through the shared physical-condition gate."""

    return validate_trunk_com_calibration(
        manifest_path=manifest_path,
        protocol_path=protocol_path,
        source_manifest_path=source_manifest_path,
    )
