"""Experimental, training-only failure-frontier snapshot contracts.

This module deliberately does not enable branching in the production trainer.
An adapter must first prove exact continuation replay for its complete state.
Selection, confirmation, and release evaluation always remain from-reset.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Protocol

from .efficiency import CostLedger, InteractionCost

REQUIRED_RUNTIME_STATE = (
    "physics_state",
    "environment_state",
    "command_state",
    "previous_raw_policy_action",
    "action_filter_state",
    "scheduler_state",
    "deployment_profile",
    "random_number_state",
    "episode_counters",
    "objective_state",
)


def _canonical_json(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"))
    # JSON's permissive NaN spelling is not a deterministic runtime state.
    if any(token in payload for token in ("NaN", "Infinity", "-Infinity")):
        raise ValueError("frontier state contains non-finite values")
    return payload


def locate_failure_frontier(
    discriminating_failure: Sequence[bool], *, lead_steps: int
) -> tuple[int, int]:
    """Return ``(snapshot_step, failure_step)`` from a training-only trace."""

    if not discriminating_failure or lead_steps < 0:
        raise ValueError("failure trace must be non-empty and lead steps non-negative")
    failures = [index for index, value in enumerate(discriminating_failure) if value]
    if not failures:
        raise ValueError("training incident trace contains no discriminating failure")
    failure_step = failures[0]
    return max(0, failure_step - lead_steps), failure_step


@dataclass(frozen=True)
class FailureFrontierSnapshot:
    """Content-addressed state just before a training-only source failure."""

    bank_id: str
    bank_role: str
    source_policy_sha256: str
    deployment_profile_sha256: str
    horizon_steps: int
    snapshot_step: int
    failure_step: int
    state: Mapping[str, Any]

    def __post_init__(self) -> None:
        if self.bank_role != "training_incident":
            raise ValueError("failure-frontier snapshots are training-only")
        lowered = self.bank_id.lower()
        if any(
            term in lowered
            for term in ("heldout", "held-out", "confirmation", "release")
        ):
            raise ValueError("protected evaluation banks cannot supply snapshots")
        if not self.bank_id:
            raise ValueError("frontier snapshot bank id cannot be empty")
        if len(self.source_policy_sha256) != 64:
            raise ValueError("frontier snapshot requires exact source-policy SHA-256")
        if len(self.deployment_profile_sha256) != 64:
            raise ValueError("frontier snapshot requires exact profile SHA-256")
        if not (0 <= self.snapshot_step < self.failure_step < self.horizon_steps):
            raise ValueError("frontier steps must precede failure within the episode")
        missing = [name for name in REQUIRED_RUNTIME_STATE if name not in self.state]
        if missing:
            raise ValueError(f"frontier snapshot is missing runtime state {missing}")
        counters = self.state["episode_counters"]
        if not isinstance(counters, Mapping):
            raise TypeError("frontier episode counters must be a mapping")
        if int(counters.get("step", -1)) != self.snapshot_step:
            raise ValueError("snapshot state step disagrees with frontier identity")
        if self.state["deployment_profile"] != self.deployment_profile_sha256:
            raise ValueError("snapshot state profile disagrees with attested profile")
        _canonical_json(self.canonical_dict())

    def canonical_dict(self) -> dict[str, Any]:
        return {
            "schema": "eggroll-failure-frontier-snapshot-v1",
            "bank_id": self.bank_id,
            "bank_role": self.bank_role,
            "source_policy_sha256": self.source_policy_sha256,
            "deployment_profile_sha256": self.deployment_profile_sha256,
            "horizon_steps": self.horizon_steps,
            "snapshot_step": self.snapshot_step,
            "failure_step": self.failure_step,
            "state": dict(self.state),
        }

    @property
    def sha256(self) -> str:
        return hashlib.sha256(
            _canonical_json(self.canonical_dict()).encode()
        ).hexdigest()


class FrontierReplayRuntime(Protocol):
    """Adapter boundary that an actual simulator/runtime must implement."""

    def restore_frontier_state(self, state: Mapping[str, Any]) -> None: ...

    def continue_from_frontier(self, actions: Sequence[float]) -> Sequence[Any]: ...


def verify_deterministic_replay(
    *,
    snapshot: FailureFrontierSnapshot,
    runtime_factory: Callable[[], FrontierReplayRuntime],
    actions: Sequence[float],
    uninterrupted_suffix: Sequence[Any],
) -> dict[str, Any]:
    """Require two restores and the uninterrupted continuation to match exactly."""

    observed: list[list[Any]] = []
    for _attempt in range(2):
        runtime = runtime_factory()
        runtime.restore_frontier_state(snapshot.state)
        observed.append(list(runtime.continue_from_frontier(actions)))
    expected = list(uninterrupted_suffix)
    if observed[0] != expected or observed[1] != expected:
        raise RuntimeError(
            "failure-frontier continuation is not an exact deterministic replay"
        )
    trace_sha256 = hashlib.sha256(_canonical_json(expected).encode()).hexdigest()
    return {
        "schema": "eggroll-failure-frontier-replay-v1",
        "snapshot_sha256": snapshot.sha256,
        "attempts": 2,
        "exact_match": True,
        "continuation_trace_sha256": trace_sha256,
    }


@dataclass(frozen=True)
class AntitheticCandidateIdentity:
    generation: int
    thread_index: int
    pair_index: int
    sign: int


def eggroll_antithetic_identities(
    *, generation: int, population: int
) -> tuple[AntitheticCandidateIdentity, ...]:
    """Expose the exact even/odd identity rule used by HyperscaleES EGGROLL."""

    if generation < 0 or population <= 0 or population % 2:
        raise ValueError("EGGROLL branching requires a positive even population")
    return tuple(
        AntitheticCandidateIdentity(
            generation=generation,
            thread_index=index,
            pair_index=index // 2,
            sign=1 if index % 2 == 0 else -1,
        )
        for index in range(population)
    )


def branch_accounting(
    *,
    snapshot: FailureFrontierSnapshot,
    population: int,
    selection_world_rollouts: int,
) -> CostLedger:
    """Account suffix optimization and mandatory full-episode selection separately."""

    if population <= 0 or population % 2:
        raise ValueError("branch population must be positive and even")
    if selection_world_rollouts < 0:
        raise ValueError("selection worlds cannot be negative")
    suffix_steps = snapshot.horizon_steps - snapshot.snapshot_step
    ledger = CostLedger()
    ledger.record(
        "optimization.failure_frontier_suffix",
        InteractionCost(
            candidate_evaluations=population,
            world_rollouts=population,
            requested_simulator_steps=population * suffix_steps,
            executed_simulator_steps=population * suffix_steps,
            active_interaction_steps=population * suffix_steps,
            policy_forward_rows=population * suffix_steps,
        ),
    )
    ledger.record(
        "selection.full_from_reset",
        InteractionCost(
            world_rollouts=selection_world_rollouts,
            requested_simulator_steps=(
                selection_world_rollouts * snapshot.horizon_steps
            ),
            executed_simulator_steps=(
                selection_world_rollouts * snapshot.horizon_steps
            ),
            active_interaction_steps=(
                selection_world_rollouts * snapshot.horizon_steps
            ),
            policy_forward_rows=(selection_world_rollouts * snapshot.horizon_steps),
        ),
    )
    full_reset_optimization = population * snapshot.horizon_steps
    saving = full_reset_optimization - population * suffix_steps
    if saving < 0 or not math.isfinite(float(saving)):
        raise RuntimeError("failure-frontier accounting produced an invalid saving")
    return ledger
