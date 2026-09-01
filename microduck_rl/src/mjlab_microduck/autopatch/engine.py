"""Forward-only, evaluator-driven EGGROLL search orchestration.

This module intentionally knows nothing about standing, locomotion, balls,
rollers, rewards, or MuJoCo.  A capability evaluator executes candidates and
returns one finite score per policy.  EGGROLL updates from those evaluations;
there is no backward pass and no differentiability requirement on the scorer.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol

import numpy as np


class SearchPolicy(Protocol):
    """The perturb/update surface supplied by the EGGROLL policy representation."""

    def update(self, raw_fitness: np.ndarray, *, generation: int) -> np.ndarray: ...


class CandidateEvaluator(Protocol):
    """Black-box capability evaluation; simulator and objective live behind it."""

    def __call__(self, policy: SearchPolicy, generation: int) -> np.ndarray: ...


@dataclass(frozen=True)
class GenerationResult:
    generation: int
    raw_min: float
    raw_mean: float
    raw_max: float
    converted_min: float
    converted_mean: float
    converted_max: float


class EggrollSearchEngine:
    """Generic finite-evaluation loop with no task- or gradient-specific behavior."""

    def __init__(
        self,
        *,
        policy: SearchPolicy,
        evaluator: CandidateEvaluator,
        population: int,
        generations: int,
        start_generation: int = 0,
    ) -> None:
        if population < 2 or population % 2:
            raise ValueError("population must be an even integer >= 2")
        if generations <= 0 or start_generation < 0:
            raise ValueError(
                "generations must be positive and start_generation non-negative"
            )
        self.policy = policy
        self.evaluator = evaluator
        self.population = population
        self.generations = generations
        self.start_generation = start_generation

    def run(
        self,
        *,
        on_generation: Callable[[GenerationResult], None] | None = None,
    ) -> tuple[GenerationResult, ...]:
        history: list[GenerationResult] = []
        stop = self.start_generation + self.generations
        for generation in range(self.start_generation, stop):
            raw = np.asarray(self.evaluator(self.policy, generation), dtype=np.float32)
            if raw.shape != (self.population,):
                raise ValueError(
                    f"evaluator returned {raw.shape}; expected {(self.population,)}"
                )
            if not np.isfinite(raw).all():
                raise FloatingPointError("evaluator returned non-finite fitness")
            converted = np.asarray(
                self.policy.update(raw, generation=generation), dtype=np.float32
            )
            if converted.shape != raw.shape or not np.isfinite(converted).all():
                raise FloatingPointError("EGGROLL returned invalid converted fitness")
            result = GenerationResult(
                generation=generation,
                raw_min=float(raw.min()),
                raw_mean=float(raw.mean()),
                raw_max=float(raw.max()),
                converted_min=float(converted.min()),
                converted_mean=float(converted.mean()),
                converted_max=float(converted.max()),
            )
            history.append(result)
            if on_generation is not None:
                on_generation(result)
        return tuple(history)
