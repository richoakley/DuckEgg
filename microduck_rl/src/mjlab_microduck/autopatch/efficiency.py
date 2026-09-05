"""Deterministic interaction accounting and phase profiling for Autopatch.

Candidate evaluations, world rollouts, requested simulator steps, executed
simulator steps, and wall time are deliberately independent quantities.  This
module keeps them independent all the way to the machine-readable result.
"""

from __future__ import annotations

import math
import time
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class InteractionCost:
    """One additive unit of Autopatch work.

    ``executed_simulator_steps`` may be ``None`` only for imported historical
    evidence that did not retain enough information to reconstruct it.  New
    work must always provide an integer.
    """

    candidate_evaluations: int = 0
    world_rollouts: int = 0
    requested_simulator_steps: int = 0
    executed_simulator_steps: int | None = 0
    active_interaction_steps: int | None = 0
    policy_forward_rows: int | None = 0
    physics_substeps: int | None = None
    wall_seconds: float = 0.0
    accelerator_seconds: float = 0.0
    world_constructions: int = 0

    def __post_init__(self) -> None:
        integers = (
            self.candidate_evaluations,
            self.world_rollouts,
            self.requested_simulator_steps,
            self.world_constructions,
        )
        if any(
            isinstance(value, bool) or not isinstance(value, int) for value in integers
        ):
            raise TypeError("interaction counts must be integers")
        if any(value < 0 for value in integers):
            raise ValueError("interaction counts cannot be negative")
        if self.executed_simulator_steps is not None:
            if isinstance(self.executed_simulator_steps, bool) or not isinstance(
                self.executed_simulator_steps, int
            ):
                raise TypeError("executed simulator steps must be an integer")
            if self.executed_simulator_steps < 0:
                raise ValueError("executed simulator steps cannot be negative")
            if self.executed_simulator_steps > self.requested_simulator_steps:
                raise ValueError("executed simulator steps exceed requested steps")
        if self.active_interaction_steps is not None:
            if isinstance(self.active_interaction_steps, bool) or not isinstance(
                self.active_interaction_steps, int
            ):
                raise TypeError("active interaction steps must be an integer")
            if self.active_interaction_steps < 0:
                raise ValueError("active interaction steps cannot be negative")
            if (
                self.executed_simulator_steps is not None
                and self.active_interaction_steps > self.executed_simulator_steps
            ):
                raise ValueError(
                    "active interaction steps exceed executed simulator slots"
                )
        for name, value in (
            ("policy forward rows", self.policy_forward_rows),
            ("physics substeps", self.physics_substeps),
        ):
            if value is not None:
                if isinstance(value, bool) or not isinstance(value, int):
                    raise TypeError(f"{name} must be an integer")
                if value < 0:
                    raise ValueError(f"{name} cannot be negative")
        if (
            not math.isfinite(self.wall_seconds)
            or not math.isfinite(self.accelerator_seconds)
            or self.wall_seconds < 0.0
            or self.accelerator_seconds < 0.0
        ):
            raise ValueError("time costs must be finite and non-negative")

    def __add__(self, other: InteractionCost) -> InteractionCost:
        executed = (
            None
            if self.executed_simulator_steps is None
            or other.executed_simulator_steps is None
            else self.executed_simulator_steps + other.executed_simulator_steps
        )
        active = (
            None
            if self.active_interaction_steps is None
            or other.active_interaction_steps is None
            else self.active_interaction_steps + other.active_interaction_steps
        )
        forwards = (
            None
            if self.policy_forward_rows is None or other.policy_forward_rows is None
            else self.policy_forward_rows + other.policy_forward_rows
        )
        physics = (
            None
            if self.physics_substeps is None or other.physics_substeps is None
            else self.physics_substeps + other.physics_substeps
        )
        return InteractionCost(
            candidate_evaluations=(
                self.candidate_evaluations + other.candidate_evaluations
            ),
            world_rollouts=self.world_rollouts + other.world_rollouts,
            requested_simulator_steps=(
                self.requested_simulator_steps + other.requested_simulator_steps
            ),
            executed_simulator_steps=executed,
            active_interaction_steps=active,
            policy_forward_rows=forwards,
            physics_substeps=physics,
            wall_seconds=self.wall_seconds + other.wall_seconds,
            accelerator_seconds=(self.accelerator_seconds + other.accelerator_seconds),
            world_constructions=self.world_constructions + other.world_constructions,
        )

    def to_dict(self) -> dict[str, int | float | None]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> InteractionCost:
        return cls(**dict(value))


def episode_interaction_cost(
    episodes: Sequence[Mapping[str, np.ndarray]],
    *,
    requested_horizon_steps: Sequence[int],
    candidate_evaluations: int = 0,
    wall_seconds: float = 0.0,
    accelerator_seconds: float = 0.0,
    physics_decimation: int | None = None,
    require_simulator_ticks: bool = False,
) -> InteractionCost:
    """Account a bank from active steps and vector ticks actually executed.

    Historical episode summaries may omit ``simulator_ticks``; in that compatibility
    mode executed steps equal active steps. New efficiency runs set
    ``require_simulator_ticks`` so masked vector slots cannot be undercounted.
    """

    if not episodes or len(episodes) != len(requested_horizon_steps):
        raise ValueError(
            "episodes and requested horizons must be non-empty and aligned"
        )
    world_rollouts = 0
    requested_steps = 0
    active_steps = 0
    executed_steps = 0
    population: int | None = None
    for episode, horizon in zip(episodes, requested_horizon_steps, strict=True):
        if horizon <= 0:
            raise ValueError("requested horizons must be positive")
        if "episode_steps" not in episode:
            raise ValueError("episode accounting requires episode_steps")
        steps = np.asarray(episode["episode_steps"])
        if steps.ndim != 1 or steps.size == 0:
            raise ValueError("episode_steps must be a non-empty vector")
        if population is None:
            population = int(steps.size)
        elif steps.size != population:
            raise ValueError("all cases in an episode bank must use one population")
        if not np.isfinite(steps).all() or np.any(steps < 0):
            raise ValueError("episode_steps must be finite and non-negative")
        rounded = np.rint(steps).astype(np.int64)
        if not np.allclose(steps, rounded, rtol=0.0, atol=1.0e-6):
            raise ValueError("executed episode steps must be integral")
        if np.any(rounded > horizon):
            raise ValueError("executed episode steps exceed the requested horizon")
        world_rollouts += int(steps.size)
        requested_steps += int(steps.size) * int(horizon)
        active_steps += int(rounded.sum())
        ticks_value = episode.get("simulator_ticks")
        if ticks_value is None:
            if require_simulator_ticks:
                raise ValueError("exact accounting requires simulator_ticks")
            executed_steps += int(rounded.sum())
        else:
            ticks = np.asarray(ticks_value)
            if ticks.shape != steps.shape or not np.isfinite(ticks).all():
                raise ValueError("simulator_ticks must align with episode_steps")
            rounded_ticks = np.rint(ticks).astype(np.int64)
            if not np.allclose(ticks, rounded_ticks, rtol=0.0, atol=1.0e-6):
                raise ValueError("simulator ticks must be integral")
            if np.any(rounded_ticks < rounded) or np.any(rounded_ticks > horizon):
                raise ValueError("simulator ticks violate active steps or horizon")
            executed_steps += int(rounded_ticks.sum())
    if candidate_evaluations < 0:
        raise ValueError("candidate evaluations cannot be negative")
    if population is not None and candidate_evaluations not in (0, population):
        raise ValueError(
            "candidate evaluations must be zero or the bank population; worlds are "
            "counted separately"
        )
    if physics_decimation is not None and physics_decimation <= 0:
        raise ValueError("physics decimation must be positive")
    return InteractionCost(
        candidate_evaluations=candidate_evaluations,
        world_rollouts=world_rollouts,
        requested_simulator_steps=requested_steps,
        executed_simulator_steps=executed_steps,
        active_interaction_steps=active_steps,
        policy_forward_rows=executed_steps,
        physics_substeps=(
            None if physics_decimation is None else executed_steps * physics_decimation
        ),
        wall_seconds=wall_seconds,
        accelerator_seconds=accelerator_seconds,
    )


class CostLedger:
    """Append-only named costs with deterministic checkpoint state."""

    SCHEMA = "eggroll-autopatch-cost-ledger-v1"

    def __init__(self) -> None:
        self._entries: list[tuple[str, InteractionCost]] = []

    def record(self, phase: str, cost: InteractionCost) -> None:
        if not phase or not phase.strip():
            raise ValueError("cost phase cannot be empty")
        self._entries.append((phase, cost))

    @property
    def entries(self) -> tuple[tuple[str, InteractionCost], ...]:
        return tuple(self._entries)

    def total(self, *, phase_prefix: str | None = None) -> InteractionCost:
        result = InteractionCost()
        for phase, cost in self._entries:
            if phase_prefix is None or phase.startswith(phase_prefix):
                result = result + cost
        return result

    def phase(self, name: str) -> InteractionCost:
        return self.total(phase_prefix=name)

    def phase_totals(self) -> dict[str, dict[str, int | float | None]]:
        totals: dict[str, InteractionCost] = {}
        for phase, cost in self._entries:
            totals[phase] = totals.get(phase, InteractionCost()) + cost
        return {phase: totals[phase].to_dict() for phase in sorted(totals)}

    def state_dict(self) -> dict[str, Any]:
        return {
            "schema": self.SCHEMA,
            "entries": [
                {"phase": phase, "cost": cost.to_dict()}
                for phase, cost in self._entries
            ],
        }

    @classmethod
    def from_state_dict(cls, value: Mapping[str, Any]) -> CostLedger:
        if value.get("schema") != cls.SCHEMA:
            raise ValueError("unknown Autopatch cost-ledger schema")
        entries = value.get("entries")
        if not isinstance(entries, list):
            raise TypeError("cost ledger entries must be a list")
        ledger = cls()
        for entry in entries:
            if not isinstance(entry, dict) or not isinstance(entry.get("cost"), dict):
                raise TypeError("invalid cost-ledger entry")
            ledger.record(str(entry["phase"]), InteractionCost.from_dict(entry["cost"]))
        return ledger

    def report(self) -> dict[str, Any]:
        return {
            "schema": self.SCHEMA,
            "phase_totals": self.phase_totals(),
            "total": self.total().to_dict(),
            "executed_steps_complete": (
                self.total().executed_simulator_steps is not None
            ),
        }


class PhaseProfiler:
    """Low-overhead host-wall phase profiler with resumable totals."""

    SCHEMA = "eggroll-autopatch-phase-profile-v1"

    def __init__(
        self,
        *,
        enabled: bool = True,
        synchronizer: Callable[[], None] | None = None,
    ) -> None:
        self.enabled = enabled
        self._synchronizer = synchronizer
        self._seconds: dict[str, float] = {}
        self._calls: dict[str, int] = {}

    @contextmanager
    def measure(self, phase: str) -> Iterator[None]:
        if not self.enabled:
            yield
            return
        if not phase:
            raise ValueError("profile phase cannot be empty")
        if self._synchronizer is not None:
            self._synchronizer()
        started = time.perf_counter()
        try:
            yield
        finally:
            if self._synchronizer is not None:
                self._synchronizer()
            self.add(phase, time.perf_counter() - started)

    def add(self, phase: str, seconds: float, *, calls: int = 1) -> None:
        if not phase or not math.isfinite(seconds) or seconds < 0.0 or calls <= 0:
            raise ValueError("invalid phase-profile sample")
        self._seconds[phase] = self._seconds.get(phase, 0.0) + float(seconds)
        self._calls[phase] = self._calls.get(phase, 0) + int(calls)

    def state_dict(self) -> dict[str, Any]:
        return {
            "schema": self.SCHEMA,
            "enabled": self.enabled,
            "phases": {
                phase: {"seconds": self._seconds[phase], "calls": self._calls[phase]}
                for phase in sorted(self._seconds)
            },
        }

    @classmethod
    def from_state_dict(cls, value: Mapping[str, Any]) -> PhaseProfiler:
        if value.get("schema") != cls.SCHEMA:
            raise ValueError("unknown Autopatch phase-profile schema")
        phases = value.get("phases")
        if not isinstance(phases, dict):
            raise TypeError("phase profile must contain a phases object")
        profiler = cls(enabled=bool(value.get("enabled", True)))
        for phase, row in phases.items():
            if not isinstance(row, dict):
                raise TypeError("invalid phase-profile row")
            profiler.add(str(phase), float(row["seconds"]), calls=int(row["calls"]))
        return profiler

    def restore(self, value: Mapping[str, Any]) -> None:
        restored = self.from_state_dict(value)
        self._seconds = restored._seconds
        self._calls = restored._calls

    def merge_state_dict(self, value: Mapping[str, Any]) -> None:
        """Add persisted phase totals to work already done in this process."""

        restored = self.from_state_dict(value)
        for phase, seconds in restored._seconds.items():
            self.add(phase, seconds, calls=restored._calls[phase])

    def to_dict(self) -> dict[str, Any]:
        result = self.state_dict()
        result["sum_inclusive_phase_seconds"] = sum(self._seconds.values())
        return result
