from __future__ import annotations

import hashlib
import json
from dataclasses import asdict
from pathlib import Path

import pytest

from mjlab_microduck.autopatch.foot_proof import (
    make_replacement_foot_calibration_bank,
    make_walking_calibration_bank,
    make_walking_proof_bank,
    select_priority_foot_material_result,
    select_replacement_foot_result,
    select_replacement_sole_result,
    select_trunk_com_result,
    select_trunk_payload_result,
    select_wedge_foot_result,
    validate_trunk_com_calibration,
    validate_trunk_payload_calibration,
    walking_bank_sha256,
)
from mjlab_microduck.autopatch.runtime_trace import robotio_write_coverage_contract
from mjlab_microduck.eggroll.deployment import (
    NOMINAL_PROFILE,
    PRIORITY_FOOT_MATERIAL_CALIBRATION_PROFILES,
    REPLACEMENT_FOOT_CALIBRATION_PROFILES,
    REPLACEMENT_SOLE_CALIBRATION_PROFILES,
    TRUNK_COM_CALIBRATION_PROFILES,
    TRUNK_PAYLOAD_CALIBRATION_PROFILES,
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


def test_trunk_com_bank_and_selection_require_broad_recoverable_failure() -> None:
    bank = make_walking_proof_bank(
        base_seed=20293001,
        episodes_per_command=8,
        prefix="calibration-trunk-com",
    )
    assert len(bank) == 32
    assert len({case.seed for case in bank}) == 32
    assert {case.command[0] for case in bank} == {0.28, 0.32, 0.36, 0.40}

    def row(profile, successes: int, per_command: tuple[int, int, int, int]) -> dict:
        return {
            "profile_sha256": profile.sha256,
            "case_count": 32,
            "success_count": successes,
            "mean_forward_velocity_rmse_mps": 0.2,
            "command_success_counts": dict(
                zip(("vx-0.28", "vx-0.32", "vx-0.36", "vx-0.40"), per_command)
            ),
        }

    profiles = TRUNK_COM_CALIBRATION_PROFILES
    rows = [
        row(NOMINAL_PROFILE, 32, (8, 8, 8, 8)),
        row(profiles[0], 30, (8, 8, 7, 7)),
        row(profiles[1], 24, (6, 6, 6, 6)),
        row(profiles[2], 16, (5, 5, 4, 2)),
        row(profiles[3], 15, (8, 5, 2, 0)),
    ]
    assert select_trunk_com_result(rows) is rows[3]
    rows[0] = row(NOMINAL_PROFILE, 30, (8, 8, 7, 7))
    assert select_trunk_com_result(rows) is None


def test_trunk_payload_selection_uses_the_same_frozen_broad_failure_rule() -> None:
    def row(profile, successes: int, per_command: tuple[int, int, int, int]) -> dict:
        return {
            "profile_sha256": profile.sha256,
            "case_count": 32,
            "success_count": successes,
            "mean_forward_velocity_rmse_mps": 0.2,
            "command_success_counts": dict(
                zip(("vx-0.28", "vx-0.32", "vx-0.36", "vx-0.40"), per_command)
            ),
        }

    profiles = TRUNK_PAYLOAD_CALIBRATION_PROFILES
    rows = [
        row(NOMINAL_PROFILE, 32, (8, 8, 8, 8)),
        row(profiles[0], 30, (8, 8, 7, 7)),
        row(profiles[1], 27, (7, 7, 7, 6)),
        row(profiles[2], 22, (6, 6, 5, 5)),
        row(profiles[3], 15, (8, 5, 2, 0)),
    ]
    assert select_trunk_payload_result(rows) is rows[3]
    rows[0] = row(NOMINAL_PROFILE, 30, (8, 8, 7, 7))
    assert select_trunk_payload_result(rows) is None


@pytest.mark.parametrize(
    (
        "protocol_name",
        "profiles",
        "base_seed",
        "prefix",
        "manifest_schema",
        "source_schema",
        "selected_noun",
        "validator",
    ),
    (
        (
            "walking_trunk_com_cross_failure_protocol_v2.json",
            (NOMINAL_PROFILE, *TRUNK_COM_CALIBRATION_PROFILES),
            20293001,
            "calibration-trunk-com",
            "trunk-com-source-calibration-v1",
            "eggroll-autopatch-trunk-com-calibration-source-v1",
            "shift",
            validate_trunk_com_calibration,
        ),
        (
            "walking_trunk_payload_cross_failure_protocol_v1.json",
            (NOMINAL_PROFILE, *TRUNK_PAYLOAD_CALIBRATION_PROFILES),
            20794001,
            "calibration-trunk-payload",
            "trunk-payload-source-calibration-v1",
            "eggroll-autopatch-trunk-payload-calibration-source-v1",
            "payload",
            validate_trunk_payload_calibration,
        ),
    ),
)
def test_physical_condition_validator_binds_runtime_evidence_and_exact_cost(
    tmp_path: Path,
    protocol_name: str,
    profiles: tuple,
    base_seed: int,
    prefix: str,
    manifest_schema: str,
    source_schema: str,
    selected_noun: str,
    validator,
) -> None:
    root = Path(__file__).resolve().parents[1]
    protocol_path = root / "docs/experiments" / protocol_name
    protocol = json.loads(protocol_path.read_text())
    bank = make_walking_proof_bank(
        base_seed=base_seed,
        episodes_per_command=8,
        prefix=prefix,
    )
    bank_rows = [json.loads(json.dumps(asdict(case))) for case in bank]
    source_sha256 = protocol["source"]["policy_sha256"]
    per_command_successes = (
        (8, 8, 8, 8),
        (8, 8, 7, 7),
        (6, 6, 6, 6),
        (4, 4, 4, 4),
        (2, 2, 2, 2),
    )
    profile_rows = []
    for profile_index, (profile, counts) in enumerate(
        zip(profiles, per_command_successes, strict=True)
    ):
        cases = []
        for case_index, case in enumerate(bank_rows):
            command_index = case_index // 8
            within_command = case_index % 8
            success = within_command < counts[command_index]
            result = {
                "terminal_success": success,
                "upright_fraction": 1.0 if success else 0.5,
                "forward_velocity_rmse_mps": 0.1 + profile_index * 0.1,
                "yaw_rate_rmse_rps": 0.01,
                "total_return": 100.0 - profile_index,
            }
            evidence = (
                tmp_path
                / "evidence"
                / f"profile-{profile_index}"
                / f"case-{case_index}"
                / "manifest.json"
            )
            evidence.parent.mkdir(parents=True)
            evidence.write_text(
                json.dumps(
                    {
                        "schema": "eggroll-autopatch-runtime-evaluation-v1",
                        "artifact": {"evaluated_sha256": source_sha256},
                        "profile_sha256": profile.sha256,
                        "scenario": {
                            "task": "Mjlab-Velocity-Flat-MicroDuck",
                            "seed": case["seed"],
                            "command": case["command"],
                        },
                        "runtime_trace_audit": {
                            "status": "pass",
                            "captured_policy_frames": 250,
                            "robotio_write_coverage": {
                                **robotio_write_coverage_contract(),
                                "applied_target_frames": 250,
                                "unapplied_target_frames": 0,
                                "applied_target_coverage": 1.0,
                                "recovered_robotio_write_failure_ticks": [],
                                "post_episode_write_failure_ticks": [],
                            },
                            "recovered_robotio_write_failures": 0,
                            "post_episode_write_failures": 0,
                        },
                        "result": result,
                    },
                    sort_keys=True,
                )
            )
            relative = evidence.relative_to(tmp_path)
            cases.append(
                {
                    "case": case,
                    "manifest": str(relative),
                    "manifest_sha256": hashlib.sha256(
                        evidence.read_bytes()
                    ).hexdigest(),
                    "runtime_trace_status": "pass",
                    "result": result,
                    "rejected_runtime_attempts": [],
                }
            )
        successes = sum(counts)
        profile_rows.append(
            {
                "profile": profile.canonical_dict(),
                "profile_sha256": profile.sha256,
                "case_count": 32,
                "success_count": successes,
                "terminal_success_rate": successes / 32,
                "command_success_counts": dict(
                    zip(
                        ("vx-0.28", "vx-0.32", "vx-0.36", "vx-0.40"),
                        counts,
                        strict=True,
                    )
                ),
                "mean_upright_fraction": sum(
                    row["result"]["upright_fraction"] for row in cases
                )
                / 32,
                "mean_forward_velocity_rmse_mps": 0.1 + profile_index * 0.1,
                "mean_yaw_rate_rmse_rps": 0.01,
                "mean_task_return_diagnostic": 100.0 - profile_index,
                "cases": cases,
            }
        )
    manifest = {
        "schema": manifest_schema,
        "status": "condition-frozen",
        "claim_scope": "production-runtime digital twin; no physical robot",
        "phase": "source-only calibration; no optimization or training",
        "artifact_id": "alpha-walking",
        "source_sha256": source_sha256,
        "task": "Mjlab-Velocity-Flat-MicroDuck",
        "bank": bank_rows,
        "bank_sha256": walking_bank_sha256(bank),
        "selection_rule": (
            f"fixed 32-case active-command bank; nominal >=31/32; selected {selected_noun} "
            "must score 16..28/32 with >=2/8 successes at every command; choose "
            "fewest successes then larger velocity RMSE; task return excluded"
        ),
        "profiles": profile_rows,
        "selected_profile": profiles[3].canonical_dict(),
        "selected_profile_sha256": profiles[3].sha256,
        "absolute_evaluation_budget": {
            "policy_episode_evaluations": 160,
            "requested_simulator_steps": 40_000,
            "requested_simulated_seconds": 800.0,
            "wall_seconds": 1.0,
            "rejected_runtime_attempts": 0,
            "candidate_optimization_evaluations": 0,
        },
        "runtime_trace_gate": robotio_write_coverage_contract(),
        "task_return_role": "diagnostic-only",
    }
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True))
    protocol_canonical_sha256 = hashlib.sha256(
        json.dumps(protocol, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    source_manifest_path = tmp_path / "source_manifest.json"
    source_manifest_path.write_text(
        json.dumps(
            {
                "schema": source_schema,
                "source_commit": "a" * 40,
                "hf_hardware_flavor": "a10g-large",
                "source_policy_sha256": source_sha256,
                "protocol_sha256": hashlib.sha256(
                    protocol_path.read_bytes()
                ).hexdigest(),
                "protocol_canonical_sha256": protocol_canonical_sha256,
                "policy_episode_evaluation_ceiling": 160,
                "requested_simulator_step_ceiling": 40_000,
                "candidate_optimization_evaluation_ceiling": 0,
            }
        )
    )

    validation = validator(
        manifest_path=manifest_path,
        protocol_path=protocol_path,
        source_manifest_path=source_manifest_path,
    )
    assert validation["status"] == "pass"
    assert validation["hf_hardware_flavor"] == "a10g-large"
    assert validation["selected_profile_sha256"] == profiles[3].sha256

    source_manifest = json.loads(source_manifest_path.read_text())
    source_manifest["protocol_sha256"] = "0" * 64
    source_manifest_path.write_text(json.dumps(source_manifest))
    with pytest.raises(ValueError, match="raw protocol bytes"):
        validator(
            manifest_path=manifest_path,
            protocol_path=protocol_path,
            source_manifest_path=source_manifest_path,
        )
    source_manifest["protocol_sha256"] = hashlib.sha256(
        protocol_path.read_bytes()
    ).hexdigest()
    source_manifest_path.write_text(json.dumps(source_manifest))

    evidence_path = tmp_path / profile_rows[3]["cases"][0]["manifest"]
    evidence_path.write_text("{}")
    with pytest.raises(ValueError, match="missing or changed"):
        validator(
            manifest_path=manifest_path,
            protocol_path=protocol_path,
            source_manifest_path=source_manifest_path,
        )
