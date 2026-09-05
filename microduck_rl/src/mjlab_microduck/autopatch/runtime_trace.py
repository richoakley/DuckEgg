"""Frozen RobotIo write-coverage contract for production-runtime traces."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

ROBOTIO_WRITE_COVERAGE_RULE_ID = "robotio-applied-target-coverage-v2"
ROBOTIO_MINIMUM_APPLIED_TARGET_COVERAGE = 0.99
ROBOTIO_MAXIMUM_POST_EPISODE_CLOSE_TICKS = 1


def robotio_write_coverage_contract() -> dict[str, Any]:
    """Return the protocol-visible, immutable write-coverage rule."""

    return {
        "rule_id": ROBOTIO_WRITE_COVERAGE_RULE_ID,
        "minimum_applied_target_coverage": (ROBOTIO_MINIMUM_APPLIED_TARGET_COVERAGE),
        "recovered_mid_episode_write_gaps": (
            "allowed-only-when-a-later-frame-has-applied-targets"
        ),
        "maximum_post_episode_close_ticks": (ROBOTIO_MAXIMUM_POST_EPISODE_CLOSE_TICKS),
    }


def audit_robotio_write_coverage(
    *,
    applied_ticks: Sequence[int],
    unapplied_ticks: Sequence[int],
) -> dict[str, Any]:
    """Classify captured policy frames without hiding RobotIo write loss.

    A missing write in the middle of an episode is accepted only when a later
    captured frame proves that RobotIo recovered, and only while at least 99%
    of all captured policy frames contain an applied target.  The existing
    allowance for one final computed tick whose write discovers that the body
    has closed is retained.  Any larger unrecovered tail fails closed.
    """

    applied = [int(tick) for tick in applied_ticks]
    unapplied = [int(tick) for tick in unapplied_ticks]
    frame_count = len(applied) + len(unapplied)
    if frame_count == 0:
        raise RuntimeError("sim-eval produced no Rust policy trace frames")
    if not applied:
        raise RuntimeError("RobotIo received no captured policy target writes")

    last_applied_tick = max(applied)
    recovered = [tick for tick in unapplied if tick < last_applied_tick]
    unrecovered = [tick for tick in unapplied if tick >= last_applied_tick]
    final_tick = max((*applied, *unapplied))
    if unrecovered and (
        len(unrecovered) > ROBOTIO_MAXIMUM_POST_EPISODE_CLOSE_TICKS
        or unrecovered != [final_tick]
    ):
        raise RuntimeError(
            f"RobotIo write loss contains an unrecovered terminal tail: {unrecovered}"
        )

    coverage = len(applied) / frame_count
    if coverage < ROBOTIO_MINIMUM_APPLIED_TARGET_COVERAGE:
        raise RuntimeError(
            "RobotIo applied-target frame coverage fell below the frozen "
            f"{ROBOTIO_MINIMUM_APPLIED_TARGET_COVERAGE:.2%} minimum: "
            f"{len(applied)}/{frame_count} ({coverage:.6%}); "
            f"unapplied ticks={unapplied}"
        )

    return {
        **robotio_write_coverage_contract(),
        "applied_target_frames": len(applied),
        "unapplied_target_frames": len(unapplied),
        "applied_target_coverage": coverage,
        "recovered_robotio_write_failure_ticks": recovered,
        "post_episode_write_failure_ticks": unrecovered,
    }
