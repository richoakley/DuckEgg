"""Equal-budget optimizer comparisons for evaluator-only Autopatch campaigns."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass
from typing import Literal

ProtocolName = Literal["eggroll", "naive-es", "random-search"]


@dataclass(frozen=True)
class ProtocolBudget:
    """One immutable black-box evaluation budget shared by every protocol."""

    population: int
    generations: int
    worlds_per_candidate: int

    def __post_init__(self) -> None:
        if self.population < 2 or self.population % 2:
            raise ValueError("population must be an even integer >= 2")
        if self.generations <= 0 or self.worlds_per_candidate <= 0:
            raise ValueError("generations and worlds_per_candidate must be positive")

    @property
    def candidate_evaluations(self) -> int:
        return self.population * self.generations

    @property
    def world_rollouts(self) -> int:
        return self.candidate_evaluations * self.worlds_per_candidate


@dataclass(frozen=True)
class ProtocolRun:
    protocol: ProtocolName
    budget: ProtocolBudget
    seed: int


def equal_budget_plan(*, budget: ProtocolBudget, seed: int) -> tuple[ProtocolRun, ...]:
    """Predeclare the comparison without consuming a simulator evaluation."""

    return tuple(
        ProtocolRun(protocol=name, budget=budget, seed=seed)
        for name in ("eggroll", "naive-es", "random-search")
    )


def run_equal_budget_protocols(
    plan: tuple[ProtocolRun, ...],
    *,
    runner: Callable[[ProtocolRun], Mapping[str, object]],
) -> dict[str, object]:
    """Execute injected protocol runners while enforcing identical budgets.

    The optimizer implementations stay behind ``runner`` so this orchestration
    cannot accidentally substitute a generic ES implementation for EGGROLL.
    """

    if {run.protocol for run in plan} != {"eggroll", "naive-es", "random-search"}:
        raise ValueError("comparison must contain EGGROLL, naive ES, and random search")
    budget_hashes = {
        (
            run.budget.population,
            run.budget.generations,
            run.budget.worlds_per_candidate,
        )
        for run in plan
    }
    if len(budget_hashes) != 1:
        raise ValueError("optimizer comparison is not evaluation-budget matched")
    results = {run.protocol: dict(runner(run)) for run in plan}
    return {
        "schema": "eggroll-autopatch-equal-budget-comparison-v1",
        "budget": asdict(plan[0].budget),
        "candidate_evaluations_per_protocol": plan[0].budget.candidate_evaluations,
        "world_rollouts_per_protocol": plan[0].budget.world_rollouts,
        "results": results,
    }
