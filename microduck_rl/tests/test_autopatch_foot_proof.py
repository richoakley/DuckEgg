from __future__ import annotations

from mjlab_microduck.autopatch.foot_proof import (
    make_replacement_foot_calibration_bank,
    make_walking_calibration_bank,
    select_priority_foot_material_result,
    select_replacement_foot_result,
    select_replacement_sole_result,
    select_wedge_foot_result,
    walking_bank_sha256,
)
from mjlab_microduck.eggroll.deployment import (
    NOMINAL_PROFILE,
    PRIORITY_FOOT_MATERIAL_CALIBRATION_PROFILES,
    REPLACEMENT_FOOT_CALIBRATION_PROFILES,
    REPLACEMENT_SOLE_CALIBRATION_PROFILES,
    WEDGE_FOOT_CALIBRATION_PROFILES,
)


def _row(profile, successes: int, rmse: float) -> dict:
    return {
        "profile_sha256": profile.sha256,
        "case_count": 4,
        "success_count": successes,
        "mean_forward_velocity_rmse_mps": rmse,
    }


def test_walking_calibration_bank_is_deterministic_and_command_diverse() -> None:
    first = make_walking_calibration_bank()
    second = make_walking_calibration_bank()
    held_out = make_walking_calibration_bank(20261012)
    assert first == second
    assert walking_bank_sha256(first) == walking_bank_sha256(second)
    assert walking_bank_sha256(first) != walking_bank_sha256(held_out)
    assert len(first) == 4
    assert len({case.seed for case in first}) == 4
    assert {case.command[:3] for case in first} == {
        (0.12, 0.0, 0.0),
        (0.20, 0.0, 0.0),
        (0.28, 0.0, 0.0),
        (0.18, 0.0, 0.30),
    }


def test_sole_selection_requires_nominal_and_rejects_catastrophic_shift() -> None:
    profiles = REPLACEMENT_SOLE_CALIBRATION_PROFILES
    rows = [
        _row(NOMINAL_PROFILE, 4, 0.1),
        _row(profiles[0], 0, 1.0),
        _row(profiles[1], 3, 0.2),
        _row(profiles[2], 2, 0.3),
        _row(profiles[3], 1, 0.4),
        _row(profiles[4], 0, 2.0),
    ]
    assert select_replacement_sole_result(rows) is rows[4]
    rows[0] = _row(NOMINAL_PROFILE, 2, 0.1)
    assert select_replacement_sole_result(rows) is None


def test_sole_selection_tie_breaks_on_tracking_not_task_return() -> None:
    profiles = REPLACEMENT_SOLE_CALIBRATION_PROFILES
    rows = [_row(NOMINAL_PROFILE, 4, 0.1)] + [
        _row(profile, 4, 0.1) for profile in profiles
    ]
    rows[2] = {**_row(profiles[1], 2, 0.3), "mean_task_return": 999.0}
    rows[4] = {**_row(profiles[3], 2, 0.8), "mean_task_return": -999.0}
    assert select_replacement_sole_result(rows) is rows[4]


def test_v2_bank_uses_nominally_active_trained_command_range() -> None:
    bank = make_replacement_foot_calibration_bank()
    assert [case.command[0] for case in bank] == [0.28, 0.32, 0.36, 0.40]
    assert len({case.seed for case in bank}) == 4
    assert walking_bank_sha256(bank) != walking_bank_sha256(
        make_walking_calibration_bank()
    )


def test_replacement_foot_selection_uses_same_partial_failure_guardrail() -> None:
    profiles = REPLACEMENT_FOOT_CALIBRATION_PROFILES
    rows = [
        _row(NOMINAL_PROFILE, 4, 0.1),
        _row(profiles[0], 4, 0.2),
        _row(profiles[1], 3, 0.3),
        _row(profiles[2], 2, 0.4),
        _row(profiles[3], 0, 2.0),
    ]
    assert select_replacement_foot_result(rows) is rows[3]


def test_priority_material_selection_rejects_zero_success_ice_case() -> None:
    profiles = PRIORITY_FOOT_MATERIAL_CALIBRATION_PROFILES
    rows = [
        _row(NOMINAL_PROFILE, 4, 0.1),
        _row(profiles[0], 4, 0.2),
        _row(profiles[1], 3, 0.3),
        _row(profiles[2], 1, 0.7),
        _row(profiles[3], 0, 1.5),
    ]
    assert select_priority_foot_material_result(rows) is rows[3]


def test_wedge_selection_uses_partial_failure_guardrail() -> None:
    profiles = WEDGE_FOOT_CALIBRATION_PROFILES
    rows = [
        _row(NOMINAL_PROFILE, 4, 0.1),
        _row(profiles[0], 4, 0.2),
        _row(profiles[1], 3, 0.3),
        _row(profiles[2], 2, 0.6),
        _row(profiles[3], 0, 1.2),
    ]
    assert select_wedge_foot_result(rows) is rows[3]
