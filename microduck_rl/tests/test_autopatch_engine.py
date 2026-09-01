from __future__ import annotations

import numpy as np
import pytest

from mjlab_microduck.autopatch.engine import EggrollSearchEngine


class FakePolicy:
    def __init__(self) -> None:
        self.updates: list[tuple[int, np.ndarray]] = []

    def update(self, raw_fitness: np.ndarray, *, generation: int) -> np.ndarray:
        self.updates.append((generation, raw_fitness.copy()))
        return np.argsort(np.argsort(raw_fitness)).astype(np.float32)


def test_generic_engine_only_requires_finite_completed_evaluations() -> None:
    policy = FakePolicy()
    calls: list[int] = []

    def nondifferentiable_evaluator(_policy: FakePolicy, generation: int) -> np.ndarray:
        calls.append(generation)
        # Deliberately discontinuous and ordinal: exactly the use case that should
        # not leak a gradient or task reward into the search engine.
        return np.array([0.0, 1.0, 1.0, 0.0]) if generation == 7 else np.arange(4)

    engine = EggrollSearchEngine(
        policy=policy,
        evaluator=nondifferentiable_evaluator,
        population=4,
        generations=2,
        start_generation=7,
    )
    history = engine.run()
    assert calls == [7, 8]
    assert [result.generation for result in history] == [7, 8]
    assert [generation for generation, _fitness in policy.updates] == [7, 8]


def test_generic_engine_rejects_wrong_population_and_nonfinite_scores() -> None:
    policy = FakePolicy()
    with pytest.raises(ValueError, match="expected"):
        EggrollSearchEngine(
            policy=policy,
            evaluator=lambda _policy, _generation: np.zeros(2),
            population=4,
            generations=1,
        ).run()
    with pytest.raises(FloatingPointError, match="non-finite"):
        EggrollSearchEngine(
            policy=policy,
            evaluator=lambda _policy, _generation: np.array([0.0, 1.0, np.nan, 2.0]),
            population=4,
            generations=1,
        ).run()
