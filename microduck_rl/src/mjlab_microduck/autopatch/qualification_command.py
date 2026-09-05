"""Content-addressed direct-command backend for release qualification.

The backend never invokes a shell. Each predeclared stage receives the exact
candidate identity, writes one machine-readable result, and is billed even when
the command fails or times out. The result is evidence *about* a gate; the
backend does not reinterpret campaign-side simulation as production evidence.
"""

from __future__ import annotations

import hashlib
import json
import math
import subprocess
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Literal

from .efficiency import InteractionCost
from .qualification import (
    QualificationCandidate,
    QualificationPlan,
    QualificationStageResult,
)

RESULT_SCHEMA = "eggroll-autopatch-qualification-command-result-v1"
TRANSCRIPT_SCHEMA = "eggroll-autopatch-qualification-command-transcript-v1"
ALLOWED_PLACEHOLDERS = {
    "candidate_checkpoint",
    "checkpoint_sha256",
    "evidence_directory",
    "generation",
    "result_path",
    "stage",
}


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _captured_text(value: str | bytes | None) -> str:
    """Normalize TimeoutExpired output, which may be bytes despite text mode."""

    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    if not isinstance(value, str):
        raise TypeError("captured subprocess output must be text or bytes")
    return value


def _field_names(template: str) -> set[str]:
    import string

    names = set()
    for _literal, field, _format_spec, _conversion in string.Formatter().parse(
        template
    ):
        if field is not None:
            names.add(field)
    return names


@dataclass(frozen=True)
class QualificationStageCommand:
    """One exact argv and result location for one predeclared release stage."""

    stage: str
    argv: tuple[str, ...]
    result_path: str
    timeout_seconds: float
    execution_failure_action: Literal["reject_candidate", "abort_campaign"] = (
        "reject_candidate"
    )

    def __post_init__(self) -> None:
        if (
            not self.stage
            or not self.argv
            or not math.isfinite(self.timeout_seconds)
            or self.timeout_seconds <= 0.0
        ):
            raise ValueError("qualification command stage is incomplete")
        if self.execution_failure_action not in {
            "reject_candidate",
            "abort_campaign",
        }:
            raise ValueError("unknown qualification execution-failure action")
        path = Path(self.result_path)
        if path.is_absolute() or ".." in path.parts or not self.result_path:
            raise ValueError("qualification result path must stay below evidence root")
        result_fields = _field_names(self.result_path)
        if "result_path" in result_fields:
            raise ValueError("qualification result path cannot reference itself")
        templates = (*self.argv, self.result_path)
        unknown = set().union(*(_field_names(value) for value in templates)) - (
            ALLOWED_PLACEHOLDERS
        )
        if unknown:
            raise ValueError(
                f"qualification command has unknown placeholders {unknown}"
            )

    @property
    def canonical_dict(self) -> dict[str, Any]:
        value = {
            "stage": self.stage,
            "argv": list(self.argv),
            "result_path": self.result_path,
            "timeout_seconds": self.timeout_seconds,
        }
        # Preserve the historical v1 command hashes and semantics when the new
        # fail-closed action is absent from a document.
        if self.execution_failure_action != "reject_candidate":
            value["execution_failure_action"] = self.execution_failure_action
        return value

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> QualificationStageCommand:
        return cls(
            stage=str(value["stage"]),
            argv=tuple(str(token) for token in value["argv"]),
            result_path=str(value["result_path"]),
            timeout_seconds=float(value["timeout_seconds"]),
            execution_failure_action=str(
                value.get("execution_failure_action", "reject_candidate")
            ),  # type: ignore[arg-type]
        )


@dataclass(frozen=True)
class CommandQualificationSpec:
    """Hashed orchestration contract for all expensive qualification stages."""

    spec_id: str
    commands: tuple[QualificationStageCommand, ...]

    def __post_init__(self) -> None:
        if not self.spec_id or not self.commands:
            raise ValueError("qualification command spec cannot be empty")
        stages = tuple(command.stage for command in self.commands)
        if len(stages) != len(set(stages)):
            raise ValueError("qualification command stages must be unique")

    @property
    def canonical_dict(self) -> dict[str, Any]:
        return {
            "spec_id": self.spec_id,
            "commands": [command.canonical_dict for command in self.commands],
        }

    @property
    def sha256(self) -> str:
        return hashlib.sha256(_canonical_json(self.canonical_dict).encode()).hexdigest()

    def validate_plan(self, plan: QualificationPlan) -> None:
        stages = tuple(command.stage for command in self.commands)
        if stages != plan.required_stages:
            raise ValueError(
                "qualification commands must exactly match the plan stage order"
            )

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> CommandQualificationSpec:
        commands = value.get("commands")
        if not isinstance(commands, Sequence) or isinstance(commands, (str, bytes)):
            raise TypeError("qualification command spec requires a command list")
        return cls(
            spec_id=str(value["spec_id"]),
            commands=tuple(
                QualificationStageCommand.from_dict(command) for command in commands
            ),
        )

    @classmethod
    def from_json(cls, payload: str) -> CommandQualificationSpec:
        value = json.loads(payload)
        if not isinstance(value, dict):
            raise TypeError("qualification command spec JSON must contain one object")
        return cls.from_dict(value)


class CommandQualificationBackend:
    """Execute hashed stage commands and validate their emitted evidence records."""

    def __init__(
        self,
        *,
        spec: CommandQualificationSpec,
        plan: QualificationPlan,
        candidate_directory: Path,
        evidence_directory: Path,
        working_directory: Path,
    ) -> None:
        spec.validate_plan(plan)
        self.spec = spec
        self.plan = plan
        self.candidate_directory = candidate_directory.resolve()
        self.evidence_directory = evidence_directory.resolve()
        self.working_directory = working_directory.resolve()
        self._commands = {command.stage: command for command in spec.commands}

    @property
    def identity_sha256(self) -> str:
        return hashlib.sha256(
            _canonical_json(
                {
                    "kind": "direct-command",
                    "spec_sha256": self.spec.sha256,
                    "working_directory": str(self.working_directory),
                }
            ).encode()
        ).hexdigest()

    @property
    def provenance(self) -> dict[str, Any]:
        return {
            "kind": "direct-command",
            "spec": self.spec.canonical_dict,
            "spec_sha256": self.spec.sha256,
            "working_directory": str(self.working_directory),
            "identity_sha256": self.identity_sha256,
        }

    def _candidate_path(self, candidate: QualificationCandidate) -> Path:
        path = (
            self.candidate_directory / f"generation-{candidate.generation:06d}.npz"
        ).resolve()
        if not path.is_relative_to(self.candidate_directory) or not path.is_file():
            raise FileNotFoundError(f"qualification candidate is missing: {path}")
        if _sha256_file(path) != candidate.checkpoint_sha256:
            raise ValueError("qualification candidate bytes do not match controller")
        return path

    def _paths_and_argv(
        self,
        *,
        candidate: QualificationCandidate,
        command: QualificationStageCommand,
        candidate_path: Path,
    ) -> tuple[Path, tuple[str, ...]]:
        initial = {
            "candidate_checkpoint": str(candidate_path),
            "checkpoint_sha256": candidate.checkpoint_sha256,
            "evidence_directory": str(self.evidence_directory),
            "generation": str(candidate.generation),
            "stage": command.stage,
        }
        relative_result = Path(command.result_path.format_map(initial))
        result_path = (self.evidence_directory / relative_result).resolve()
        if not result_path.is_relative_to(self.evidence_directory):
            raise ValueError("qualification result escaped the evidence directory")
        values = {**initial, "result_path": str(result_path)}
        argv = tuple(token.format_map(values) for token in command.argv)
        return result_path, argv

    def _write_transcript(
        self,
        *,
        candidate: QualificationCandidate,
        stage: str,
        argv: tuple[str, ...],
        elapsed: float,
        returncode: int | None,
        stdout: str,
        stderr: str,
        failure: str | None,
    ) -> Path:
        transcript = (
            self.evidence_directory
            / "transcripts"
            / f"generation-{candidate.generation:06d}-{stage}.json"
        )
        if transcript.exists():
            raise FileExistsError(
                f"qualification transcript already exists: {transcript}"
            )
        transcript.parent.mkdir(parents=True, exist_ok=True)
        transcript.write_text(
            json.dumps(
                {
                    "schema": TRANSCRIPT_SCHEMA,
                    "generation": candidate.generation,
                    "checkpoint_sha256": candidate.checkpoint_sha256,
                    "stage": stage,
                    "argv": list(argv),
                    "returncode": returncode,
                    "elapsed_wall_seconds": elapsed,
                    "stdout": stdout,
                    "stderr": stderr,
                    "failure": failure,
                },
                indent=2,
                sort_keys=True,
            )
            + "\n"
        )
        return transcript

    def __call__(
        self, candidate: QualificationCandidate, stage: str
    ) -> QualificationStageResult:
        command = self._commands.get(stage)
        if command is None:
            raise ValueError(f"qualification stage is not predeclared: {stage}")
        candidate_path = self._candidate_path(candidate)
        result_path, argv = self._paths_and_argv(
            candidate=candidate,
            command=command,
            candidate_path=candidate_path,
        )
        if result_path.exists():
            raise FileExistsError(f"qualification result already exists: {result_path}")
        result_path.parent.mkdir(parents=True, exist_ok=True)

        started = time.perf_counter()
        completed: subprocess.CompletedProcess[str] | None = None
        failure: str | None = None
        try:
            completed = subprocess.run(
                argv,
                cwd=self.working_directory,
                check=False,
                capture_output=True,
                text=True,
                timeout=command.timeout_seconds,
            )
        except subprocess.TimeoutExpired as error:
            failure = f"command timed out after {command.timeout_seconds:g} seconds"
            stdout = _captured_text(error.stdout)
            stderr = _captured_text(error.stderr)
        except OSError as error:
            failure = f"command could not start: {error}"
            stdout = ""
            stderr = ""
        else:
            stdout = completed.stdout
            stderr = completed.stderr
            if completed.returncode != 0:
                failure = f"command exited with status {completed.returncode}"
        elapsed = time.perf_counter() - started
        transcript = self._write_transcript(
            candidate=candidate,
            stage=stage,
            argv=argv,
            elapsed=elapsed,
            returncode=None if completed is None else completed.returncode,
            stdout=stdout,
            stderr=stderr,
            failure=failure,
        )
        if failure is not None and command.execution_failure_action == "abort_campaign":
            raise RuntimeError(
                f"qualification infrastructure failed at {stage}: {failure}; "
                f"transcript={transcript}"
            )
        if failure is not None:
            return QualificationStageResult(
                stage=stage,
                status="fail",
                reason=failure,
                evidence_sha256=_sha256_file(transcript),
                cost=InteractionCost(wall_seconds=elapsed),
            )
        if not result_path.is_file():
            return QualificationStageResult(
                stage=stage,
                status="fail",
                reason="qualification command emitted no result manifest",
                evidence_sha256=_sha256_file(transcript),
                cost=InteractionCost(wall_seconds=elapsed),
            )

        value = json.loads(result_path.read_text())
        if not isinstance(value, dict) or value.get("schema") != RESULT_SCHEMA:
            raise ValueError("qualification command emitted an unknown result schema")
        if value.get("stage") != stage:
            raise ValueError("qualification result stage does not match command")
        if value.get("checkpoint_sha256") != candidate.checkpoint_sha256:
            raise ValueError("qualification result belongs to another candidate")
        status = value.get("status")
        reason = value.get("reason")
        cost_value = value.get("cost")
        if status not in ("pass", "fail") or not isinstance(reason, str) or not reason:
            raise ValueError("qualification result status or reason is invalid")
        if not isinstance(cost_value, dict):
            raise TypeError("qualification result has no interaction cost")
        cost = InteractionCost.from_dict(cost_value)
        cost = replace(cost, wall_seconds=cost.wall_seconds + elapsed)
        return QualificationStageResult(
            stage=stage,
            status=status,
            reason=reason,
            evidence_sha256=_sha256_file(result_path),
            cost=cost,
        )
