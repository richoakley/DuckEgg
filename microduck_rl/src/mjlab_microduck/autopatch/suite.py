"""One production-runtime source acceptance suite for all deployed policies."""

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
class FleetCase:
    artifact_id: str
    task: str
    steps: int
    mode: str | None = None
    side: str = "right"
    reset_label: str = "standing"
    command: tuple[float, ...] = (0.0,) * 13
    return_step: int | None = None

    def __post_init__(self) -> None:
        if self.steps <= 0 or len(self.command) != 13:
            raise ValueError("fleet cases require positive steps and a 13D command")
        if self.return_step is not None and not (0 < self.return_step < self.steps):
            raise ValueError("fleet return_step must fall strictly inside the horizon")


SOURCE_ACCEPTANCE_CASES = (
    FleetCase(
        "alpha-walking",
        "Mjlab-Velocity-Flat-MicroDuck",
        250,
        command=(0.2, 0.0, 0.0, *(0.0,) * 10),
        return_step=205,
    ),
    FleetCase(
        "alpha-stand",
        "Mjlab-StandUp-Flat-MicroDuck",
        300,
        reset_label="face-down",
    ),
    FleetCase(
        "alpha-sitstand",
        "Mjlab-SitStand-Flat-MicroDuck",
        300,
        mode="walk",
    ),
    FleetCase(
        "alpha-ground-pick",
        "Mjlab-GroundPick-Flat-MicroDuck",
        250,
    ),
    FleetCase(
        "ball-kick-left",
        "Mjlab-BallKick-Flat-MicroDuck",
        250,
        mode="walk",
        side="left",
    ),
    FleetCase(
        "ball-kick-right",
        "Mjlab-BallKick-Flat-MicroDuck",
        250,
        mode="walk",
        side="right",
    ),
    FleetCase(
        "roller",
        "Mjlab-Velocity-Flat-MicroDuck-Rollers",
        250,
        command=(0.2, 0.0, 0.0, *(0.0,) * 10),
    ),
    FleetCase(
        "roller-crouch",
        "Mjlab-RollerCrouch-Flat-MicroDuck",
        250,
    ),
    FleetCase(
        "roulade",
        "Mjlab-Roulade-Flat-MicroDuck",
        250,
        mode="walk",
    ),
)


def run_source_acceptance_suite(
    *,
    registry: AutopatchRegistry,
    runtime_repo: Path,
    robotd: Path,
    ort_dylib: Path,
    output_dir: Path,
    profile: DeploymentProfile | AsymmetricActuatorProfile,
    seed: int,
    device: str = "cpu",
    record_video: bool = False,
    timeout_s: float = 30.0,
    max_attempts: int = 2,
) -> dict[str, Any]:
    """Run all nine sealed source artifacts through one actual-task workflow."""

    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"output directory is not empty: {output_dir}")
    if max_attempts <= 0:
        raise ValueError("max_attempts must be positive")
    output_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    for case in SOURCE_ACCEPTANCE_CASES:
        failures: list[dict[str, str | int]] = []
        manifest = None
        case_dir = output_dir / case.artifact_id
        for attempt in range(1, max_attempts + 1):
            attempt_dir = (
                case_dir
                if max_attempts == 1
                else output_dir / f"{case.artifact_id}.attempt-{attempt}"
            )
            try:
                manifest = run_runtime_evaluation(
                    registry=registry,
                    runtime_repo=runtime_repo,
                    robotd=robotd,
                    ort_dylib=ort_dylib,
                    output_dir=attempt_dir,
                    request=RuntimeEvaluationRequest(
                        artifact_id=case.artifact_id,
                        task=case.task,
                        seed=seed,
                        side=case.side,
                        command=case.command,
                        device=device,
                        record_video=record_video,
                        timeout_s=timeout_s,
                        horizon_steps=case.steps,
                        reset_label=case.reset_label,
                        return_step=case.return_step,
                    ),
                    profile=profile,
                    mode=case.mode,
                )
                case_dir = attempt_dir
                break
            except (ConnectionError, TimeoutError, RuntimeError) as error:
                failures.append(
                    {
                        "attempt": attempt,
                        "error_type": type(error).__name__,
                        "error": str(error),
                    }
                )
        if manifest is None:
            raise RuntimeError(
                f"{case.artifact_id} exhausted {max_attempts} strict runtime attempts: "
                f"{failures}"
            )
        rows.append(
            {
                "case": asdict(case),
                "manifest": str((case_dir / "manifest.json").relative_to(output_dir)),
                "manifest_sha256": sha256_file(case_dir / "manifest.json"),
                "evaluated_sha256": manifest["artifact"]["evaluated_sha256"],
                "terminal_success": bool(manifest["result"]["terminal_success"]),
                "acceptance_id": manifest["result"].get(
                    "acceptance_id", "standup-task-terminal-stable-success-v1"
                ),
                "runtime_trace_status": manifest["runtime_trace_audit"]["status"],
                "policy_sequence": manifest["driver"]["policy_sequence"],
                "rejected_runtime_attempts": failures,
            }
        )
    passed = all(
        row["terminal_success"] and row["runtime_trace_status"] == "pass"
        for row in rows
    )
    graph = _graph_coverage(registry=registry, rows=rows)
    passed = passed and graph["status"] == "pass"
    manifest = {
        "schema": "eggroll-autopatch-source-fleet-v1",
        "claim_scope": "production-runtime digital twin; no physical robot",
        "suite_role": (
            "one deterministic source acceptance smoke per artifact; not statistical "
            "robustness and not evidence of EGGROLL adaptation"
        ),
        "seed": seed,
        "profile_sha256": profile.sha256,
        "status": "pass" if passed else "fail",
        "artifacts_passed": sum(bool(row["terminal_success"]) for row in rows),
        "artifacts_total": len(rows),
        "cases": rows,
        "capability_graph": graph,
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    return manifest


def _has_subsequence(sequence: list[str], expected: tuple[str, ...]) -> bool:
    cursor = iter(sequence)
    return all(any(value == wanted for value in cursor) for wanted in expected)


def _graph_coverage(
    *, registry: AutopatchRegistry, rows: list[dict[str, Any]]
) -> dict[str, Any]:
    """Bind node and scheduler-edge release coverage to fleet evidence.

    Triggered skills are selected before the simulator is released, so their
    incoming edge is proven by the recorded production intent followed by the
    exact target network/SHA. Return edges are observed directly in robotd's
    policy sequence. Continuous stand/walk edges are both observed directly.
    """

    by_artifact = {row["case"]["artifact_id"]: row for row in rows}
    node_rows = []
    for artifact in registry.artifacts:
        row = by_artifact[artifact.artifact_id]
        passed = bool(row["terminal_success"]) and row["runtime_trace_status"] == "pass"
        node_rows.append(
            {
                "artifact_id": artifact.artifact_id,
                "capability_id": artifact.capability_id,
                "status": "pass" if passed else "fail",
                "manifest": row["manifest"],
                "acceptance_id": row["acceptance_id"],
            }
        )

    patterns = {
        ("stationary-body-control", "legged-locomotion"): ("stand", "walk"),
        ("legged-locomotion", "stationary-body-control"): ("walk", "stand"),
        ("sit-stand-transition", "stationary-body-control"): ("sit", "rise", "stand"),
    }
    target_label = {
        "sit-stand-transition": "sit",
        "ground-pick": "ground_pick",
        "ball-kick": "kick_",
        "forward-roll": "roulade",
        "roller-crouch": "ground_pick",
    }
    return_label = {
        "sit-stand-transition": "stand",
        "ground-pick": "stand",
        "ball-kick": "stand",
        "forward-roll": "stand",
        "roller-crouch": "walk",
    }
    edge_rows = []
    for transition in registry.transitions:
        candidates = [
            by_artifact[artifact_id]
            for artifact_id in transition.required_artifact_ids
            if artifact_id in by_artifact
        ]
        direct = patterns.get(
            (transition.source_capability, transition.target_capability)
        )
        passed = False
        evidence: dict[str, Any] = {}
        if direct is not None:
            for row in candidates:
                labels = [item["policy"] for item in row["policy_sequence"]]
                if _has_subsequence(labels, direct) and row["terminal_success"]:
                    passed = True
                    evidence = {
                        "manifest": row["manifest"],
                        "policy_subsequence": list(direct),
                    }
                    break
        elif transition.source_capability == "stationary-body-control":
            wanted = target_label[transition.target_capability]
            for row in candidates:
                labels = [item["policy"] for item in row["policy_sequence"]]
                selected = any(
                    label == wanted
                    or (wanted.endswith("_") and label.startswith(wanted))
                    for label in labels
                )
                returned = return_label[transition.target_capability] in labels
                if selected and returned and row["terminal_success"]:
                    passed = True
                    evidence = {
                        "manifest": row["manifest"],
                        "trigger_intent": transition.trigger,
                        "selected_policy": wanted,
                        "return_policy": return_label[transition.target_capability],
                    }
                    break
        elif (
            transition.source_capability == "roller-locomotion"
            and transition.target_capability == "roller-crouch"
        ):
            row = by_artifact["roller-crouch"]
            labels = [item["policy"] for item in row["policy_sequence"]]
            passed = _has_subsequence(labels, ("ground_pick", "walk")) and bool(
                row["terminal_success"]
            )
            evidence = {
                "manifest": row["manifest"],
                "trigger_intent": transition.trigger,
                "policy_subsequence": ["ground_pick", "walk"],
            }
        edge_rows.append(
            {
                "transition_sha256": transition.sha256,
                "source_capability": transition.source_capability,
                "target_capability": transition.target_capability,
                "status": "pass" if passed else "gap",
                "evidence": evidence,
            }
        )
    gaps = [row for row in edge_rows if row["status"] != "pass"]
    return {
        "status": "pass" if not gaps else "gap",
        "node_tests": node_rows,
        "edge_tests": edge_rows,
        "gaps": gaps,
    }
