from __future__ import annotations

from copy import deepcopy
from typing import Any

import pytest

from mjlab_microduck.autopatch.failure_frontier import (
    REQUIRED_RUNTIME_STATE,
    FailureFrontierSnapshot,
    branch_accounting,
    eggroll_antithetic_identities,
    locate_failure_frontier,
    verify_deterministic_replay,
)


class _ToyRuntime:
    """Stateful filter/previous-action runtime used to prove exact replay."""

    def __init__(self) -> None:
        self.state: dict[str, Any] = {
            "physics_state": {"position": 0.0},
            "environment_state": {"alive": True},
            "command_state": {"target": 1.0},
            "previous_raw_policy_action": 0.0,
            "action_filter_state": 0.0,
            "scheduler_state": {"active_policy": "walking"},
            "deployment_profile": "b" * 64,
            "random_number_state": {"counter": 11},
            "episode_counters": {"step": 0},
            "objective_state": {"return": 0.0},
        }

    def restore_frontier_state(self, state: dict[str, Any]) -> None:
        self.state = deepcopy(state)

    def continue_from_frontier(self, actions: list[float]) -> list[dict[str, float]]:
        trace = []
        for action in actions:
            previous = float(self.state["previous_raw_policy_action"])
            filtered = 0.5 * action + 0.5 * float(self.state["action_filter_state"])
            position = float(self.state["physics_state"]["position"])
            position += filtered + 0.1 * previous
            self.state["previous_raw_policy_action"] = action
            self.state["action_filter_state"] = filtered
            self.state["physics_state"]["position"] = position
            self.state["episode_counters"]["step"] += 1
            self.state["objective_state"]["return"] += position
            trace.append(
                {
                    "position": round(position, 12),
                    "filtered": round(filtered, 12),
                    "previous": round(action, 12),
                }
            )
        return trace


def _snapshot(state: dict[str, Any] | None = None) -> FailureFrontierSnapshot:
    payload = deepcopy(state if state is not None else _ToyRuntime().state)
    payload["episode_counters"]["step"] = 3
    return FailureFrontierSnapshot(
        bank_id="training-incident-seed-7",
        bank_role="training_incident",
        source_policy_sha256="a" * 64,
        deployment_profile_sha256="b" * 64,
        horizon_steps=8,
        snapshot_step=3,
        failure_step=5,
        state=payload,
    )


def test_restored_source_continuation_matches_uninterrupted_continuation() -> None:
    uninterrupted = _ToyRuntime()
    uninterrupted.continue_from_frontier([1.0, -0.5, 0.2])
    state = deepcopy(uninterrupted.state)
    snapshot = _snapshot(state)
    suffix_actions = [0.3, -0.1, 0.8, 0.0, -0.2]
    expected = uninterrupted.continue_from_frontier(suffix_actions)

    result = verify_deterministic_replay(
        snapshot=snapshot,
        runtime_factory=_ToyRuntime,
        actions=suffix_actions,
        uninterrupted_suffix=expected,
    )
    assert result["exact_match"] is True
    assert result["attempts"] == 2


def test_snapshot_requires_previous_action_filter_scheduler_and_all_state() -> None:
    assert {
        "previous_raw_policy_action",
        "action_filter_state",
        "scheduler_state",
    }.issubset(REQUIRED_RUNTIME_STATE)
    state = deepcopy(_ToyRuntime().state)
    state["episode_counters"]["step"] = 3
    del state["previous_raw_policy_action"]
    with pytest.raises(ValueError, match="missing runtime state"):
        _snapshot(state)


@pytest.mark.parametrize("bank_id", ["heldout-1", "confirmation-a", "release-bank"])
def test_protected_evaluation_banks_cannot_supply_training_snapshots(
    bank_id: str,
) -> None:
    with pytest.raises(ValueError, match="cannot supply snapshots"):
        FailureFrontierSnapshot(
            bank_id=bank_id,
            bank_role="training_incident",
            source_policy_sha256="a" * 64,
            deployment_profile_sha256="b" * 64,
            horizon_steps=8,
            snapshot_step=3,
            failure_step=5,
            state=_snapshot().state,
        )


def test_frontier_detection_and_antithetic_candidate_identities_are_stable() -> None:
    assert locate_failure_frontier(
        [False, False, False, False, False, True, True], lead_steps=2
    ) == (3, 5)
    identities = eggroll_antithetic_identities(generation=9, population=6)
    assert [(row.pair_index, row.sign) for row in identities] == [
        (0, 1),
        (0, -1),
        (1, 1),
        (1, -1),
        (2, 1),
        (2, -1),
    ]
    assert all(row.generation == 9 for row in identities)


def test_suffix_savings_do_not_hide_full_episode_selection_cost() -> None:
    ledger = branch_accounting(
        snapshot=_snapshot(),
        population=64,
        selection_world_rollouts=8,
    )
    suffix = ledger.phase("optimization")
    selection = ledger.phase("selection")
    assert suffix.requested_simulator_steps == 64 * (8 - 3)
    assert suffix.requested_simulator_steps < 64 * 8
    assert selection.requested_simulator_steps == 8 * 8
    assert selection.world_rollouts == 8


def test_suffix_candidate_must_still_pass_from_reset_episode() -> None:
    # The candidate learns a consequential suffix action, but eligibility is
    # checked by a complete episode, including its unchanged prefix.
    source_actions = [0.0] * 8
    candidate_actions = [0.0] * 3 + [0.6] * 5
    source = _ToyRuntime().continue_from_frontier(source_actions)
    candidate = _ToyRuntime().continue_from_frontier(candidate_actions)
    assert source[-1]["position"] == 0.0
    assert candidate[-1]["position"] > 1.0
    assert len(candidate) == 8
