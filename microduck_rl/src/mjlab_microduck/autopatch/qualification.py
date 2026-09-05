"""Evidence-preserving qualification controller for Autopatch early stopping."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal, Protocol

from .contracts import PatchCampaign, ReleaseScope
from .efficiency import CostLedger, InteractionCost

QualificationStatus = Literal["pending", "rejected", "eligible"]
StageStatus = Literal["pass", "fail"]


DEFAULT_QUALIFICATION_STAGES = (
    "release_scope_retention",
    "onnx_parity",
    "production_runtime",
    "independent_confirmation",
    "profile_routing",
    "signed_activation_and_rollback",
)


def _gate_passes(value: float, comparator: str, threshold: float) -> bool:
    return {
        ">": value > threshold,
        ">=": value >= threshold,
        "==": value == threshold,
        "<=": value <= threshold,
        "<": value < threshold,
    }[comparator]


def campaign_side_gate_screen(
    *,
    campaign: PatchCampaign,
    metrics_history: Sequence[Mapping[str, Any]],
    release_scope: ReleaseScope | None = None,
) -> tuple[bool, str]:
    """Apply deployment-relevant cheap gates before expensive qualification.

    Historical and multi-profile campaigns retain target and nominal gates.  A
    profile-specific derivative is never activated on nominal or unknown profiles,
    so its adapted-policy nominal result remains a recorded diagnostic while exact
    source fallback is proved by the routing and activation stages.
    """

    reasons: list[str] = []
    for gate in campaign.gates:
        if gate.profile_role not in ("target", "nominal"):
            continue
        if (
            release_scope is not None
            and release_scope.mode == "profile_specific"
            and gate.profile_role == "nominal"
        ):
            continue
        prefix = "shifted" if gate.profile_role == "target" else "nominal"
        metric_name = f"{prefix}/objective/{gate.metric}"
        observed: list[float] = []
        for row in reversed(metrics_history):
            if metric_name not in row:
                continue
            observed.append(float(row[metric_name]))
            if len(observed) == gate.consecutive_passes:
                break
        passed = len(observed) == gate.consecutive_passes and all(
            _gate_passes(value, gate.comparator, gate.threshold) for value in observed
        )
        reasons.append(
            f"{gate.gate_id}={'pass' if passed else 'fail'} "
            f"values={list(reversed(observed))}"
        )
        if not passed:
            return False, "; ".join(reasons)
    return True, "; ".join(reasons) or "no campaign-side gates"


@dataclass(frozen=True)
class QualificationPlan:
    """Predeclared stopping contract; never inferred from observed results."""

    plan_id: str
    evaluation_interval: int
    selection_metrics: tuple[str, ...]
    required_stages: tuple[str, ...] = DEFAULT_QUALIFICATION_STAGES

    def __post_init__(self) -> None:
        if not self.plan_id or self.evaluation_interval <= 0:
            raise ValueError("qualification plan id and interval must be valid")
        if not self.selection_metrics or not self.required_stages:
            raise ValueError("qualification plan requires selection metrics and stages")
        if len(self.required_stages) != len(set(self.required_stages)):
            raise ValueError("qualification stages must be unique")
        if any("task_return" in name for name in self.selection_metrics):
            raise ValueError("registered task return cannot select a qualification")

    @property
    def canonical_dict(self) -> dict[str, Any]:
        return {
            "plan_id": self.plan_id,
            "evaluation_interval": self.evaluation_interval,
            "selection_metrics": list(self.selection_metrics),
            "required_stages": list(self.required_stages),
        }

    @property
    def sha256(self) -> str:
        payload = json.dumps(
            self.canonical_dict, sort_keys=True, separators=(",", ":")
        ).encode()
        return hashlib.sha256(payload).hexdigest()

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> QualificationPlan:
        return cls(
            plan_id=str(value["plan_id"]),
            evaluation_interval=int(value["evaluation_interval"]),
            selection_metrics=tuple(str(name) for name in value["selection_metrics"]),
            required_stages=tuple(str(name) for name in value["required_stages"]),
        )

    @classmethod
    def from_json(cls, payload: str) -> QualificationPlan:
        value = json.loads(payload)
        if not isinstance(value, dict):
            raise TypeError("qualification plan JSON must contain one object")
        return cls.from_dict(value)


@dataclass(frozen=True)
class QualificationCandidate:
    generation: int
    checkpoint_sha256: str
    selection_metrics: tuple[tuple[str, float], ...]
    selection_passed: bool
    selection_reason: str
    selection_cost: InteractionCost

    def __post_init__(self) -> None:
        if self.generation <= 0 or len(self.checkpoint_sha256) != 64:
            raise ValueError("qualification candidate identity is invalid")
        if not self.selection_reason:
            raise ValueError("qualification selection must record a reason")


@dataclass(frozen=True)
class QualificationStageResult:
    stage: str
    status: StageStatus
    reason: str
    evidence_sha256: str | None
    cost: InteractionCost

    def __post_init__(self) -> None:
        if not self.stage or self.status not in ("pass", "fail") or not self.reason:
            raise ValueError("qualification stage result is incomplete")
        if self.status == "pass" and self.evidence_sha256 is None:
            raise ValueError("a passing qualification stage requires evidence bytes")
        if self.evidence_sha256 is not None and len(self.evidence_sha256) != 64:
            raise ValueError("qualification evidence identity must be a SHA-256")


class QualificationBackend(Protocol):
    """Production implementation supplies expensive stage evidence in order."""

    def __call__(
        self, candidate: QualificationCandidate, stage: str
    ) -> QualificationStageResult | None: ...


class QualificationController:
    """Record attempts and stop only after every required release stage passes."""

    SCHEMA = "eggroll-autopatch-qualification-state-v1"

    def __init__(self, plan: QualificationPlan) -> None:
        self.plan = plan
        self._attempts: list[dict[str, Any]] = []
        self._ledger = CostLedger()
        self._stop_generation: int | None = None

    @property
    def should_stop(self) -> bool:
        return self._stop_generation is not None

    @property
    def stop_generation(self) -> int | None:
        return self._stop_generation

    @property
    def attempts(self) -> tuple[dict[str, Any], ...]:
        return tuple(self._attempts)

    @property
    def cost(self) -> InteractionCost:
        return self._ledger.total()

    @property
    def last_backend_cost(self) -> InteractionCost:
        """Cost of only the most recent expensive stages, excluding selection."""

        if not self._attempts:
            return InteractionCost()
        total = InteractionCost()
        for row in self._attempts[-1]["stages"]:
            total = total + InteractionCost.from_dict(row["cost"])
        return total

    def qualify(
        self,
        candidate: QualificationCandidate,
        backend: QualificationBackend,
    ) -> QualificationStatus:
        if self.should_stop:
            raise RuntimeError("qualification already established campaign success")
        if candidate.generation % self.plan.evaluation_interval:
            raise ValueError("candidate generation is outside the declared interval")
        if any(
            int(attempt["generation"]) == candidate.generation
            for attempt in self._attempts
        ):
            raise ValueError("qualification generation was already attempted")
        metrics = dict(candidate.selection_metrics)
        missing = [name for name in self.plan.selection_metrics if name not in metrics]
        if missing:
            raise ValueError(f"qualification selection is missing metrics {missing}")
        if any(
            "task_return" in name
            for name in metrics
            if name in self.plan.selection_metrics
        ):
            raise ValueError("task return cannot be a qualification selection metric")

        attempt: dict[str, Any] = {
            "generation": candidate.generation,
            "checkpoint_sha256": candidate.checkpoint_sha256,
            "selection_metrics": {
                name: float(metrics[name]) for name in self.plan.selection_metrics
            },
            "selection": {
                "status": "pass" if candidate.selection_passed else "fail",
                "reason": candidate.selection_reason,
                "cost": candidate.selection_cost.to_dict(),
            },
            "stages": [],
            "status": "pending",
        }
        self._ledger.record("qualification.selection", candidate.selection_cost)
        self._attempts.append(attempt)
        if not candidate.selection_passed:
            attempt["status"] = "rejected"
            attempt["rejection_reason"] = candidate.selection_reason
            return "rejected"

        for stage in self.plan.required_stages:
            result = backend(candidate, stage)
            if result is None:
                attempt["status"] = "rejected"
                attempt["rejection_reason"] = (
                    f"qualification evidence was incomplete at {stage}"
                )
                return "rejected"
            if result.stage != stage:
                raise ValueError(
                    "qualification backend stages must follow the predeclared order"
                )
            stage_row = {
                "stage": result.stage,
                "status": result.status,
                "reason": result.reason,
                "evidence_sha256": result.evidence_sha256,
                "cost": result.cost.to_dict(),
            }
            attempt["stages"].append(stage_row)
            self._ledger.record(f"qualification.{result.stage}", result.cost)
            if result.status == "fail":
                attempt["status"] = "rejected"
                attempt["rejection_reason"] = f"{result.stage}: {result.reason}"
                return "rejected"
        attempt["status"] = "eligible"
        self._stop_generation = candidate.generation
        return "eligible"

    def state_dict(self) -> dict[str, Any]:
        return {
            "schema": self.SCHEMA,
            "plan": self.plan.canonical_dict,
            "plan_sha256": self.plan.sha256,
            "attempts": self._attempts,
            "stop_generation": self._stop_generation,
            "cost_ledger": self._ledger.state_dict(),
        }

    @classmethod
    def from_state_dict(
        cls, value: Mapping[str, Any], *, plan: QualificationPlan
    ) -> QualificationController:
        if value.get("schema") != cls.SCHEMA:
            raise ValueError("unknown qualification-state schema")
        if value.get("plan_sha256") != plan.sha256:
            raise ValueError("resume qualification plan does not match checkpoint")
        attempts = value.get("attempts")
        if not isinstance(attempts, list):
            raise TypeError("qualification attempts must be a list")
        controller = cls(plan)
        controller._attempts = json.loads(json.dumps(attempts))
        controller._stop_generation = (
            None
            if value.get("stop_generation") is None
            else int(value["stop_generation"])
        )
        ledger = value.get("cost_ledger")
        if not isinstance(ledger, dict):
            raise TypeError("qualification resume state has no cost ledger")
        controller._ledger = CostLedger.from_state_dict(ledger)
        eligible = [a for a in controller._attempts if a.get("status") == "eligible"]
        if (controller._stop_generation is None) != (len(eligible) == 0):
            raise ValueError("qualification stop state disagrees with attempt history")
        if len(eligible) > 1:
            raise ValueError("qualification history contains multiple stop attempts")
        return controller


def stage_backend(
    results: Sequence[QualificationStageResult],
) -> Callable[[QualificationCandidate, str], QualificationStageResult | None]:
    """Small explicit adapter useful for deterministic replay and tests."""

    frozen = tuple(results)

    def backend(
        _candidate: QualificationCandidate, stage: str
    ) -> QualificationStageResult | None:
        return next((result for result in frozen if result.stage == stage), None)

    return backend
