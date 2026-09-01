from __future__ import annotations

from mjlab_microduck.autopatch.registry import PRODUCTION_REGISTRY
from mjlab_microduck.autopatch.suite import SOURCE_ACCEPTANCE_CASES, _has_subsequence


def test_source_acceptance_suite_covers_each_sealed_artifact_once() -> None:
    expected = {artifact.artifact_id for artifact in PRODUCTION_REGISTRY.artifacts}
    actual = [case.artifact_id for case in SOURCE_ACCEPTANCE_CASES]
    assert set(actual) == expected
    assert len(actual) == len(set(actual)) == 9


def test_source_acceptance_cases_only_use_registered_artifact_tasks() -> None:
    for case in SOURCE_ACCEPTANCE_CASES:
        artifact = PRODUCTION_REGISTRY.artifact(case.artifact_id)
        assert case.task in artifact.task_ids
        assert len(case.command) == 13


def test_walking_case_exercises_both_stand_walk_edges() -> None:
    walking = next(
        case for case in SOURCE_ACCEPTANCE_CASES if case.artifact_id == "alpha-walking"
    )
    assert walking.return_step is not None


def test_subsequence_matcher_preserves_order() -> None:
    assert _has_subsequence(["stand", "walk", "stand"], ("stand", "walk"))
    assert _has_subsequence(["stand", "walk", "stand"], ("walk", "stand"))
    assert not _has_subsequence(["walk", "stand"], ("stand", "walk"))
