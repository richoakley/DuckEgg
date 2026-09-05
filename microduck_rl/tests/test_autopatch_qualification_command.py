"""Direct-command qualification backend contract tests."""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pytest

from mjlab_microduck.autopatch.efficiency import InteractionCost
from mjlab_microduck.autopatch.qualification import (
    QualificationCandidate,
    QualificationPlan,
)
from mjlab_microduck.autopatch.qualification_command import (
    RESULT_SCHEMA,
    CommandQualificationBackend,
    CommandQualificationSpec,
    QualificationStageCommand,
)


def _plan() -> QualificationPlan:
    return QualificationPlan(
        plan_id="command-test-v1",
        evaluation_interval=1,
        selection_metrics=("retained_source_success_rate",),
        required_stages=("production_runtime",),
    )


def _candidate(candidate_directory: Path) -> QualificationCandidate:
    path = candidate_directory / "generation-000001.npz"
    path.parent.mkdir(parents=True)
    path.write_bytes(b"candidate-bytes")
    return QualificationCandidate(
        generation=1,
        checkpoint_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        selection_metrics=(("retained_source_success_rate", 1.0),),
        selection_passed=True,
        selection_reason="cheap gates passed",
        selection_cost=InteractionCost(),
    )


def _backend(
    tmp_path: Path, command: QualificationStageCommand
) -> CommandQualificationBackend:
    spec = CommandQualificationSpec(spec_id="command-test-v1", commands=(command,))
    return CommandQualificationBackend(
        spec=spec,
        plan=_plan(),
        candidate_directory=tmp_path / "candidates",
        evidence_directory=tmp_path / "evidence",
        working_directory=tmp_path,
    )


def test_command_backend_hashes_candidate_bound_result_and_cost(tmp_path: Path) -> None:
    writer = tmp_path / "write_result.py"
    writer.write_text(
        """import json
import sys
from pathlib import Path

result, stage, checkpoint_sha256, candidate = sys.argv[1:]
if not Path(candidate).is_file():
    raise SystemExit(2)
Path(result).write_text(json.dumps({
    "schema": "eggroll-autopatch-qualification-command-result-v1",
    "stage": stage,
    "checkpoint_sha256": checkpoint_sha256,
    "status": "pass",
    "reason": "production trace matched",
    "cost": {
        "world_rollouts": 2,
        "requested_simulator_steps": 500,
        "executed_simulator_steps": 480,
        "active_interaction_steps": 470
    }
}))
"""
    )
    command = QualificationStageCommand(
        stage="production_runtime",
        argv=(
            sys.executable,
            str(writer),
            "{result_path}",
            "{stage}",
            "{checkpoint_sha256}",
            "{candidate_checkpoint}",
        ),
        result_path="generation-{generation}/{stage}.json",
        timeout_seconds=10.0,
    )
    backend = _backend(tmp_path, command)
    result = backend(_candidate(tmp_path / "candidates"), "production_runtime")

    assert result.status == "pass"
    assert result.evidence_sha256 is not None
    assert len(result.evidence_sha256) == 64
    assert result.cost.world_rollouts == 2
    assert result.cost.requested_simulator_steps == 500
    assert result.cost.executed_simulator_steps == 480
    assert result.cost.wall_seconds > 0.0
    assert len(backend.identity_sha256) == 64
    transcript = (
        tmp_path / "evidence/transcripts/generation-000001-production_runtime.json"
    )
    assert json.loads(transcript.read_text())["returncode"] == 0


def test_failed_command_is_rejected_and_still_billed(tmp_path: Path) -> None:
    command = QualificationStageCommand(
        stage="production_runtime",
        argv=(sys.executable, "-c", "raise SystemExit(7)"),
        result_path="generation-{generation}/{stage}.json",
        timeout_seconds=10.0,
    )
    backend = _backend(tmp_path, command)
    result = backend(_candidate(tmp_path / "candidates"), "production_runtime")

    assert result.status == "fail"
    assert result.reason == "command exited with status 7"
    assert result.evidence_sha256 is not None
    assert result.cost.wall_seconds > 0.0


def test_timed_out_command_persists_binary_partial_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    command = QualificationStageCommand(
        stage="production_runtime",
        argv=(sys.executable, "-c", "pass"),
        result_path="generation-{generation}/{stage}.json",
        timeout_seconds=1.0,
    )
    backend = _backend(tmp_path, command)

    def time_out(*_args, **_kwargs):
        raise subprocess.TimeoutExpired(
            cmd=(sys.executable, "-c", "pass"),
            timeout=1.0,
            output=b"partial stdout \xff",
            stderr=b"partial stderr \xfe",
        )

    monkeypatch.setattr(subprocess, "run", time_out)
    result = backend(_candidate(tmp_path / "candidates"), "production_runtime")

    assert result.status == "fail"
    assert result.reason == "command timed out after 1 seconds"
    transcript = json.loads(
        (
            tmp_path / "evidence/transcripts/generation-000001-production_runtime.json"
        ).read_text()
    )
    assert transcript["stdout"] == "partial stdout �"
    assert transcript["stderr"] == "partial stderr �"


def test_fail_closed_command_aborts_campaign_on_infrastructure_failure(
    tmp_path: Path,
) -> None:
    command = QualificationStageCommand(
        stage="production_runtime",
        argv=(sys.executable, "-c", "raise SystemExit(7)"),
        result_path="generation-{generation}/{stage}.json",
        timeout_seconds=10.0,
        execution_failure_action="abort_campaign",
    )
    backend = _backend(tmp_path, command)

    with pytest.raises(RuntimeError, match="infrastructure failed"):
        backend(_candidate(tmp_path / "candidates"), "production_runtime")
    assert (
        tmp_path / "evidence/transcripts/generation-000001-production_runtime.json"
    ).is_file()


def test_command_backend_rejects_candidate_orchestration_drift(tmp_path: Path) -> None:
    command = QualificationStageCommand(
        stage="production_runtime",
        argv=(sys.executable, "-c", "raise SystemExit(0)"),
        result_path="generation-{generation}/{stage}.json",
        timeout_seconds=10.0,
    )
    backend = _backend(tmp_path, command)
    candidate = _candidate(tmp_path / "candidates")
    (tmp_path / "candidates/generation-000001.npz").write_bytes(b"changed")
    with pytest.raises(ValueError, match="bytes do not match"):
        backend(candidate, "production_runtime")

    with pytest.raises(ValueError, match="unknown placeholders"):
        QualificationStageCommand(
            stage="production_runtime",
            argv=("tool", "{secret}"),
            result_path="result.json",
            timeout_seconds=1.0,
        )
    with pytest.raises(ValueError, match="cannot reference itself"):
        QualificationStageCommand(
            stage="production_runtime",
            argv=("tool",),
            result_path="{result_path}",
            timeout_seconds=1.0,
        )


def test_command_spec_must_match_qualification_plan_order() -> None:
    spec = CommandQualificationSpec(
        spec_id="wrong-order",
        commands=(
            QualificationStageCommand(
                stage="onnx_parity",
                argv=("tool",),
                result_path="parity.json",
                timeout_seconds=1.0,
            ),
        ),
    )
    with pytest.raises(ValueError, match="exactly match"):
        spec.validate_plan(_plan())


def test_result_schema_constant_is_stable() -> None:
    assert RESULT_SCHEMA == "eggroll-autopatch-qualification-command-result-v1"
