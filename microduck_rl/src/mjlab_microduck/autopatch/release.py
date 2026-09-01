"""Policy-agnostic Autopatch release gates and evidence envelopes."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from mjlab_microduck.eggroll.policy_io import import_deployed_policy
from mjlab_microduck.eggroll.release import (
    runtime_parity,
    sha256_file,
    verify_output_layer_derivative,
)

from .contracts import PatchCampaign, ReleaseGate, ReleaseScope
from .registry import AutopatchRegistry


@dataclass(frozen=True)
class GateSeries:
    gate_id: str
    values: tuple[float, ...]


def evaluate_paired_non_regression(
    manifest: dict[str, Any],
    *,
    artifact_id: str,
    source_sha256: str,
    adapted_sha256: str,
    profile_role: str = "nominal",
) -> dict[str, Any]:
    """Prove case-by-case retention of every source success in one paired bank.

    Zero regression is deliberately narrower than equality of every diagnostic.  It
    means that a derivative may not fail the capability's binary terminal semantics
    on any case where the sealed source passed.  Improvements are allowed, task return
    remains diagnostic, and an empty or mismatched bank is a hard failure.
    """

    if manifest.get("schema") != "eggroll-autopatch-paired-ab-v1":
        raise ValueError("unknown paired A/B evidence schema")
    if manifest.get("artifact_id") != artifact_id:
        raise ValueError("paired evidence belongs to a different artifact")
    if manifest.get("source_sha256") != source_sha256:
        raise ValueError("paired evidence used different source bytes")
    if manifest.get("adapted_sha256") != adapted_sha256:
        raise ValueError("paired evidence used different adapted bytes")
    bank = manifest.get("paired_bank")
    rows = manifest.get("rows")
    if not isinstance(bank, list) or not bank:
        raise TypeError("paired evidence bank must be a non-empty list")
    if not isinstance(rows, list):
        raise TypeError("paired evidence rows must be a list")
    bank_by_id = {
        case.get("case_id"): case for case in bank if isinstance(case, dict)
    }
    if len(bank_by_id) != len(bank) or not all(
        isinstance(case_id, str) and case_id for case_id in bank_by_id
    ):
        raise ValueError("paired evidence bank needs unique non-empty case ids")
    seeds = [case.get("seed") for case in bank]
    if not all(isinstance(seed, int) for seed in seeds) or len(set(seeds)) != len(seeds):
        raise ValueError("paired evidence bank needs unique integer seeds")
    bank_sha256 = hashlib.sha256(
        json.dumps(bank, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()

    selected = [row for row in rows if row.get("profile_role") == profile_role]
    if not selected:
        raise ValueError(
            f"paired evidence has no {profile_role!r} source-retention rows"
        )
    seen: set[tuple[str, str]] = set()
    seen_case_ids: set[str] = set()
    regressions: list[str] = []
    improvements: list[str] = []
    source_successes = 0
    adapted_successes = 0
    profile_sha256s: set[str] = set()
    for row in selected:
        case = row.get("case")
        source = row.get("source")
        adapted = row.get("adapted")
        if not (
            isinstance(case, dict)
            and isinstance(source, dict)
            and isinstance(adapted, dict)
        ):
            raise TypeError("paired evidence row has malformed case or policy result")
        case_id = case.get("case_id")
        profile_sha256 = row.get("profile_sha256")
        if (
            not isinstance(case_id, str)
            or not case_id
            or not isinstance(profile_sha256, str)
        ):
            raise ValueError("paired evidence row needs a case id and profile hash")
        if bank_by_id.get(case_id) != case:
            raise ValueError(f"paired row {case_id!r} is not bound to the sealed bank")
        identity = (profile_sha256, case_id)
        if identity in seen:
            raise ValueError(f"duplicate paired evidence case {case_id!r}")
        seen.add(identity)
        seen_case_ids.add(case_id)
        profile_sha256s.add(profile_sha256)
        if source.get("policy_sha256") != source_sha256:
            raise ValueError(f"paired source row {case_id!r} used different bytes")
        if adapted.get("policy_sha256") != adapted_sha256:
            raise ValueError(f"paired adapted row {case_id!r} used different bytes")
        source_passed = source.get("terminal_success")
        adapted_passed = adapted.get("terminal_success")
        if not isinstance(source_passed, bool) or not isinstance(adapted_passed, bool):
            raise TypeError("paired terminal_success values must be booleans")
        source_successes += int(source_passed)
        adapted_successes += int(adapted_passed)
        if source_passed and not adapted_passed:
            regressions.append(case_id)
        elif adapted_passed and not source_passed:
            improvements.append(case_id)

    if seen_case_ids != set(bank_by_id):
        raise ValueError("paired profile rows do not completely cover the sealed bank")
    if len(profile_sha256s) != 1:
        raise ValueError("one paired profile role must resolve to one profile hash")
    return {
        "profile_role": profile_role,
        "profile_sha256": next(iter(profile_sha256s)),
        "bank_sha256": bank_sha256,
        "seeds": seeds,
        "paired_cases": len(selected),
        "source_successes": source_successes,
        "adapted_successes": adapted_successes,
        "source_success_regressions": len(regressions),
        "regression_case_ids": regressions,
        "improvement_case_ids": improvements,
        "zero_observed_regressions": not regressions,
        "claim_scope": (
            "zero observed regressions under the capability's terminal-success "
            "semantics on this sealed paired bank"
        ),
    }


def build_paired_non_regression_envelope(
    *,
    manifests: tuple[Path, ...],
    artifact_id: str,
    source_sha256: str,
    adapted_sha256: str,
    release_scope: ReleaseScope,
    minimum_independent_banks: int = 2,
) -> dict[str, Any]:
    """Bind independent paired banks to a declared deployment scope.

    ``profile_specific`` patches prove retention only on the attested activation
    profile. ``multi_profile`` patches prove retention on every declared profile.
    Cross-profile diagnostics may still be recorded in the input manifests, but do not
    silently expand or narrow the release claim.
    """

    if len(manifests) < minimum_independent_banks:
        raise ValueError(
            "release requires at least two independent paired non-regression banks"
        )
    if release_scope.source_fallback_sha256 != source_sha256:
        raise ValueError("release scope fallback does not match the sealed source bytes")
    expected_profiles = dict(release_scope.profile_sha256s)
    records = []
    for path in manifests:
        record = json.loads(path.read_text())
        if not isinstance(record, dict):
            raise TypeError(f"{path} must contain a JSON object")
        reports = []
        for profile_role in release_scope.required_retention_roles:
            report = evaluate_paired_non_regression(
                record,
                artifact_id=artifact_id,
                source_sha256=source_sha256,
                adapted_sha256=adapted_sha256,
                profile_role=profile_role,
            )
            if report["profile_sha256"] != expected_profiles[profile_role]:
                raise ValueError(
                    f"paired {profile_role!r} evidence used an undeclared profile"
                )
            reports.append(report)
        records.append(
            {"path": str(path), "sha256": sha256_file(path), "reports": reports}
        )
    if any(
        not report["zero_observed_regressions"]
        for row in records
        for report in row["reports"]
    ):
        raise ValueError("paired release evidence contains source-success regressions")
    bank_sha256s = {row["reports"][0]["bank_sha256"] for row in records}
    if len(bank_sha256s) != len(records):
        raise ValueError("paired non-regression evidence banks are not independent")
    seen_seeds: set[int] = set()
    for row in records:
        report_bank_sha256s = {
            report["bank_sha256"] for report in row["reports"]
        }
        if len(report_bank_sha256s) != 1:
            raise ValueError("retention profiles in one manifest use different banks")
        seeds = set(row["reports"][0]["seeds"])
        if seen_seeds & seeds:
            raise ValueError("paired non-regression banks reuse scenario seeds")
        seen_seeds.update(seeds)
    return {
        "schema": "eggroll-autopatch-paired-non-regression-v2",
        "status": "pass",
        "artifact_id": artifact_id,
        "source_sha256": source_sha256,
        "adapted_sha256": adapted_sha256,
        "release_scope": release_scope.canonical_dict(),
        "release_scope_sha256": release_scope.sha256,
        "retention_profile_roles": list(release_scope.required_retention_roles),
        "independent_banks": len(records),
        "source_success_regressions": 0,
        "claim_scope": (
            "zero observed regressions under terminal-success semantics on the "
            "declared deployment profiles across the sealed paired banks; not a "
            "universal guarantee"
        ),
        "evidence": records,
    }


def validate_routing_evidence(
    evidence: dict[str, Any],
    *,
    artifact_id: str,
    adapted_sha256: str,
    release_scope: ReleaseScope,
) -> None:
    """Require fail-closed runtime/updater enforcement for scoped activation."""

    if evidence.get("schema") != "eggroll-autopatch-routing-attestation-v1":
        raise ValueError("unknown routing attestation schema")
    expected = {
        "status": "pass",
        "artifact_id": artifact_id,
        "adapted_sha256": adapted_sha256,
        "release_scope_sha256": release_scope.sha256,
        "source_fallback_sha256": release_scope.source_fallback_sha256,
        "unknown_profile_action": release_scope.unknown_profile_action,
    }
    mismatches = {
        key: {"expected": value, "observed": evidence.get(key)}
        for key, value in expected.items()
        if evidence.get(key) != value
    }
    if mismatches:
        raise ValueError(f"routing attestation does not enforce release scope: {mismatches}")
    if evidence.get("production_path") != "updaterd::profile_scoped_model_activation":
        raise ValueError("routing attestation did not use the production updater path")
    routes = evidence.get("routes")
    if not isinstance(routes, list):
        raise TypeError("routing attestation routes must be a list")
    observed = {
        (
            route.get("profile_sha256"),
            route.get("decision"),
            route.get("selected_policy_sha256"),
        )
        for route in routes
        if isinstance(route, dict)
    }
    required = {
        (profile_sha256, "activate_adapted", adapted_sha256)
        for _role, profile_sha256 in release_scope.profile_sha256s
    }
    unknown_policy = (
        release_scope.source_fallback_sha256
        if release_scope.unknown_profile_action == "retain_source"
        else None
    )
    required.add(
        (
            "unknown",
            release_scope.unknown_profile_action,
            unknown_policy,
        )
    )
    missing = required - observed
    if missing:
        raise ValueError(f"routing attestation is missing fail-closed routes: {missing}")


def _compare(value: float, gate: ReleaseGate) -> bool:
    operations = {
        ">": lambda: value > gate.threshold,
        ">=": lambda: value >= gate.threshold,
        "==": lambda: value == gate.threshold,
        "<=": lambda: value <= gate.threshold,
        "<": lambda: value < gate.threshold,
    }
    return operations[gate.comparator]()


def evaluate_release_gates(
    gates: tuple[ReleaseGate, ...], series: tuple[GateSeries, ...]
) -> tuple[dict[str, Any], ...]:
    """Evaluate consecutive hard gates; missing evidence is a release failure."""

    by_id = {item.gate_id: item.values for item in series}
    if len(by_id) != len(series):
        raise ValueError("duplicate gate evidence")
    rows = []
    for gate in gates:
        values = by_id.get(gate.gate_id)
        if values is None:
            raise ValueError(f"missing evidence for release gate {gate.gate_id!r}")
        if not values or not all(math.isfinite(value) for value in values):
            raise ValueError(f"gate {gate.gate_id!r} has invalid measurements")
        passes = tuple(_compare(value, gate) for value in values)
        consecutive = 0
        for passed in passes:
            consecutive = consecutive + 1 if passed else 0
        rows.append(
            {
                **gate.canonical_dict(),
                "values": list(values),
                "passes": list(passes),
                "final_consecutive_passes": consecutive,
                "status": "pass" if consecutive >= gate.consecutive_passes else "fail",
            }
        )
    return tuple(rows)


def build_release_envelope(
    *,
    campaign: PatchCampaign,
    registry: AutopatchRegistry,
    source_policy: Path,
    adapted_policy: Path,
    runtime_probe: Path,
    node_evidence: tuple[Path, ...],
    covered_transition_sha256: tuple[str, ...],
    gate_series: tuple[GateSeries, ...],
    paired_evidence: tuple[Path, ...],
    release_scope: ReleaseScope,
    routing_evidence: Path,
) -> dict[str, Any]:
    """Verify bytes, parity, graph coverage and gates before updater packaging."""

    registry.validate_campaign(campaign)
    artifact = registry.artifact(campaign.artifact_id)
    source = import_deployed_policy(source_policy)
    adapted = import_deployed_policy(adapted_policy)
    if source.source_sha256 != artifact.expected_sha256:
        raise ValueError("release source is not the sealed registry artifact")
    if release_scope.source_fallback_sha256 != source.source_sha256:
        raise ValueError("release scope fallback is not the sealed source policy")
    verify_output_layer_derivative(source=source, adapted=adapted)
    parity_error = runtime_parity(adapted)
    if parity_error >= 1.0e-5:
        raise ValueError("adapted ONNX parity exceeds 1e-5")

    probe = json.loads(runtime_probe.read_text())
    if not isinstance(probe, dict) or not (
        probe.get("passed") is True or probe.get("status") == "pass"
    ):
        raise ValueError("production Rust loader probe did not pass")
    if probe.get("production_path") != "duck_control::policy::Policy::load":
        raise ValueError("runtime probe did not use the production Rust loader")
    if probe.get("policy_sha256") != adapted.source_sha256:
        raise ValueError("production loader probe names different adapted bytes")

    node_records = []
    for path in node_evidence:
        record = json.loads(path.read_text())
        if not isinstance(record, dict):
            raise TypeError(f"{path} must contain a JSON object")
        evaluated = record.get("artifact", {}).get("evaluated_sha256")
        if evaluated != adapted.source_sha256:
            raise ValueError(f"node evidence {path} evaluated different policy bytes")
        if record.get("runtime_trace_audit", {}).get("status") != "pass":
            raise ValueError(f"node evidence {path} failed runtime trace parity")
        if record.get("result", {}).get("terminal_success") is not True:
            raise ValueError(f"node evidence {path} failed capability acceptance")
        node_records.append({"path": str(path), "sha256": sha256_file(path)})

    plan = registry.release_test_plan(artifact.artifact_id)
    required_edges = {edge["transition_sha256"] for edge in plan["edges"]}
    covered_edges = set(covered_transition_sha256)
    missing_edges = required_edges - covered_edges
    if missing_edges:
        raise ValueError(
            f"release is missing scheduler-edge evidence: {sorted(missing_edges)}"
        )
    paired_envelope = build_paired_non_regression_envelope(
        manifests=paired_evidence,
        artifact_id=artifact.artifact_id,
        source_sha256=source.source_sha256,
        adapted_sha256=adapted.source_sha256,
        release_scope=release_scope,
    )
    routing = json.loads(routing_evidence.read_text())
    if not isinstance(routing, dict):
        raise TypeError("routing evidence must contain a JSON object")
    validate_routing_evidence(
        routing,
        artifact_id=artifact.artifact_id,
        adapted_sha256=adapted.source_sha256,
        release_scope=release_scope,
    )
    gate_rows = evaluate_release_gates(campaign.gates, gate_series)
    passed = bool(node_records) and all(row["status"] == "pass" for row in gate_rows)
    if not passed:
        raise ValueError("one or more hard release gates failed")
    return {
        "schema": "eggroll-autopatch-release-envelope-v3",
        "status": "eligible-for-signed-updater-packaging",
        "campaign_id": campaign.campaign_id,
        "campaign_sha256": campaign.sha256,
        "artifact_id": artifact.artifact_id,
        "runtime_slot": artifact.runtime_slot,
        "updater_component": artifact.updater_component,
        "source_policy": {
            "path": str(source_policy),
            "sha256": source.source_sha256,
        },
        "adapted_policy": {
            "path": str(adapted_policy),
            "sha256": adapted.source_sha256,
            "parity_max_abs_error": parity_error,
            "patch_scope": "output-layer-only",
        },
        "production_loader_probe": {
            "path": str(runtime_probe),
            "sha256": sha256_file(runtime_probe),
        },
        "node_evidence": node_records,
        "paired_non_regression": paired_envelope,
        "release_scope": release_scope.canonical_dict(),
        "release_scope_sha256": release_scope.sha256,
        "routing_attestation": {
            "path": str(routing_evidence),
            "sha256": sha256_file(routing_evidence),
        },
        "covered_transition_sha256": sorted(covered_edges),
        "release_gates": list(gate_rows),
        "rollback_target_sha256": source.source_sha256,
        "next_step": "package and sign with cargo xtask package-model/sign",
    }
