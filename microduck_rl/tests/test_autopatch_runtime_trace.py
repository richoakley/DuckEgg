from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from mjlab_microduck.autopatch.runtime_trace import (
    audit_robotio_write_coverage,
    robotio_write_coverage_contract,
)

ROOT = Path(__file__).resolve().parents[1]


def test_robotio_write_coverage_accepts_reproduced_recovered_gap() -> None:
    result = audit_robotio_write_coverage(
        applied_ticks=[*range(234), *range(236, 249)],
        unapplied_ticks=[234, 235],
    )

    assert result["rule_id"] == "robotio-applied-target-coverage-v2"
    assert result["applied_target_frames"] == 247
    assert result["unapplied_target_frames"] == 2
    assert result["applied_target_coverage"] == pytest.approx(247 / 249)
    assert result["recovered_robotio_write_failure_ticks"] == [234, 235]
    assert result["post_episode_write_failure_ticks"] == []


def test_robotio_write_coverage_retains_single_body_close_tick_allowance() -> None:
    result = audit_robotio_write_coverage(
        applied_ticks=range(249),
        unapplied_ticks=[249],
    )

    assert result["recovered_robotio_write_failure_ticks"] == []
    assert result["post_episode_write_failure_ticks"] == [249]


def test_robotio_write_coverage_rejects_gap_below_frozen_floor() -> None:
    with pytest.raises(RuntimeError, match="fell below the frozen 99.00% minimum"):
        audit_robotio_write_coverage(
            applied_ticks=[*range(230), *range(236, 249)],
            unapplied_ticks=list(range(230, 236)),
        )


def test_robotio_write_coverage_rejects_unrecovered_terminal_tail() -> None:
    with pytest.raises(RuntimeError, match="unrecovered terminal tail"):
        audit_robotio_write_coverage(
            applied_ticks=range(247),
            unapplied_ticks=[247, 248],
        )


def test_trunk_com_v2_amends_only_trace_classification_and_identity() -> None:
    v1_path = ROOT / "docs/experiments/walking_trunk_com_cross_failure_protocol_v1.json"
    v2_path = ROOT / "docs/experiments/walking_trunk_com_cross_failure_protocol_v2.json"
    v1 = json.loads(v1_path.read_text())
    v2 = json.loads(v2_path.read_text())

    assert hashlib.sha256(v1_path.read_bytes()).hexdigest() == (
        "ac4261f24253bbf54b5cc62cca1a7ac574553ef9c7ceb42fea409e13dacfa2f3"
    )
    assert (
        v2["amendment"]["prior_protocol_sha256"]
        == hashlib.sha256(v1_path.read_bytes()).hexdigest()
    )
    assert v2["runtime_trace_gate"] == robotio_write_coverage_contract()
    for frozen_section in (
        "prerequisite",
        "source",
        "condition",
        "calibration",
        "banks",
        "training",
        "campaign_side_gates",
        "release_gates",
        "claim_boundaries",
    ):
        assert v2[frozen_section] == v1[frozen_section]
