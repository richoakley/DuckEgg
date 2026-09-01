"""Evidence-bound release manifests for EGGROLL policy derivatives.

The training checkpoint is not the deployable product.  A released derivative is
the exact ONNX bytes plus the black-box evaluation evidence that made those bytes
eligible to replace a production policy.  This module builds and verifies that
contract without requiring CUDA or a simulator.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

import numpy as np

from .checkpoint import load_checkpoint, write_json
from .deployment import DeploymentProfile, Scenario, bank_sha256, make_balanced_bank
from .policy_io import (
    DeployedPolicy,
    import_deployed_policy,
    numpy_actions,
    onnx_actions,
)

RELEASE_SCHEMA = "mjlab-microduck-eggroll-policy-derivative"
RELEASE_SCHEMA_VERSION = 1
TASK_ID = "Mjlab-StandUp-Flat-MicroDuck"
POSES = ("standing", "sitting", "face-down", "face-up")
SUMMARY_ROLES = (
    "source_shifted",
    "adapted_shifted",
    "source_nominal",
    "adapted_nominal",
)


def sha256_file(path: Path) -> str:
    """Return the lowercase SHA-256 of one local artifact."""

    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_sha256(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise TypeError(f"{path} must contain a JSON object")
    return value


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text().splitlines(), start=1):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise TypeError(f"{path}:{line_number} must contain a JSON object")
        rows.append(value)
    return rows


def _policy_fixed_state(policy: DeployedPolicy) -> tuple[np.ndarray, ...]:
    values: list[np.ndarray] = [
        policy.normalizer_mean,
        policy.normalizer_denominator,
    ]
    for layer in policy.layers[:-1]:
        values.extend((layer.weight, layer.bias))
    return tuple(values)


def _tensor_sha256(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    return hashlib.sha256(array.tobytes()).hexdigest()


def _policy_tensor_hashes(policy: DeployedPolicy) -> dict[str, Any]:
    return {
        "frozen": [_tensor_sha256(value) for value in _policy_fixed_state(policy)],
        "output_weight": _tensor_sha256(policy.output_weight),
        "output_bias": _tensor_sha256(policy.output_bias),
    }


def verify_output_layer_derivative(
    *, source: DeployedPolicy, adapted: DeployedPolicy
) -> None:
    """Prove that only the final weight and bias differ from the base actor."""

    if source.input_name != adapted.input_name or source.output_name != adapted.output_name:
        raise ValueError("Adapted policy changed the ONNX input/output names")
    for index, (before, after) in enumerate(
        zip(_policy_fixed_state(source), _policy_fixed_state(adapted), strict=True)
    ):
        if not np.array_equal(before, after):
            raise ValueError(f"Adapted policy changed frozen tensor {index}")
    if np.array_equal(source.output_weight, adapted.output_weight) and np.array_equal(
        source.output_bias, adapted.output_bias
    ):
        raise ValueError("Adapted policy did not change the output layer")


def runtime_parity(policy: DeployedPolicy, *, seed: int = 20260830) -> float:
    """Independently compare the imported graph with ONNX Runtime."""

    observations = np.random.default_rng(seed).normal(size=(64, 61)).astype(np.float32)
    expected = numpy_actions(policy, observations)
    actual = onnx_actions(policy.source_model, observations)
    return float(np.max(np.abs(expected - actual)))


def _summary_worlds(summary: Mapping[str, Any]) -> int:
    bank = summary.get("bank")
    if not isinstance(bank, list) or not bank:
        raise ValueError("Evaluation summary has no scenario bank")
    return len(bank)


def _summary_successes(summary: Mapping[str, Any]) -> int:
    values = summary.get("episodes", {}).get("terminal_success")
    if not isinstance(values, list):
        raise TypeError("Evaluation summary has no per-episode terminal_success")
    return sum(bool(value) for value in values)


def _summary_pose_successes(summary: Mapping[str, Any]) -> dict[str, int]:
    bank = summary.get("bank")
    values = summary.get("episodes", {}).get("terminal_success")
    if not isinstance(bank, list) or not isinstance(values, list) or len(bank) != len(values):
        raise ValueError("Evaluation summary bank and terminal results do not align")
    result = {pose: 0 for pose in POSES}
    totals = {pose: 0 for pose in POSES}
    for scenario, success in zip(bank, values, strict=True):
        pose = str(scenario["pose"])
        if pose not in result:
            raise ValueError(f"Evaluation summary contains unknown pose {pose!r}")
        totals[pose] += 1
        result[pose] += int(bool(success))
    if len(set(totals.values())) != 1 or min(totals.values()) <= 0:
        raise ValueError("Final evaluation bank is not balanced across reset poses")
    return result


def _scenario_fingerprints(scenarios: Iterable[Scenario | Mapping[str, Any]]) -> set[tuple[Any, ...]]:
    result: set[tuple[Any, ...]] = set()
    for scenario in scenarios:
        if isinstance(scenario, Scenario):
            pose = scenario.pose
            seed = scenario.seed
            profile = scenario.profile_sha256
            command = scenario.command
        else:
            pose = str(scenario["pose"])
            seed = int(scenario["seed"])
            profile = str(scenario["profile_sha256"])
            command = tuple(float(value) for value in scenario["command"])
        result.add((pose, seed, profile, tuple(command)))
    return result


def _assert_disjoint(roles: Mapping[str, set[tuple[Any, ...]]]) -> None:
    names = list(roles)
    for index, left in enumerate(names):
        for right in names[index + 1 :]:
            overlap = roles[left] & roles[right]
            if overlap:
                raise ValueError(
                    f"Episode-bank leakage between {left} and {right}: "
                    f"{next(iter(overlap))}"
                )


def _summary_record(path: Path, summary: Mapping[str, Any]) -> dict[str, Any]:
    episodes = summary["episodes"]
    return {
        "path": path.name,
        "sha256": sha256_file(path),
        "policy_sha256": summary["source_policy_sha256"],
        "profile": summary["profile"],
        "profile_sha256": summary["profile_sha256"],
        "bank_sha256": summary["bank_sha256"],
        "worlds": _summary_worlds(summary),
        "terminal_successes": _summary_successes(summary),
        "per_pose_terminal_successes": _summary_pose_successes(summary),
        "metrics": {
            "mean_stable_hold_s": episodes_mean(episodes, "stable_hold_s"),
            "mean_max_trunk_height_m": episodes_mean(
                episodes, "max_trunk_height_m"
            ),
            "mean_final_trunk_height_m": episodes_mean(
                episodes, "final_trunk_height_m"
            ),
            "mean_final_upright_cosine": episodes_mean(
                episodes, "final_upright_cosine"
            ),
            "mean_time_upright_s": episodes_mean(episodes, "time_upright_s"),
            "mean_task_return": episodes_mean(episodes, "task_return"),
        },
    }


def episodes_mean(episodes: Mapping[str, Any], key: str) -> float:
    values = episodes.get(key)
    if not isinstance(values, list) or not values:
        raise ValueError(f"Evaluation summary has no episode metric {key!r}")
    return float(np.mean(np.asarray(values, dtype=np.float64)))


def _validate_summary(
    summary: Mapping[str, Any],
    *,
    expected_policy_sha256: str,
    expected_profile_sha256: str,
) -> None:
    if summary.get("task") != TASK_ID:
        raise ValueError("Evaluation did not use the actual registered StandUp task")
    if summary.get("authoritative_cuda_evaluation") is not True:
        raise ValueError("Final release evidence must be authoritative CUDA evaluation")
    if summary.get("source_policy_sha256") != expected_policy_sha256:
        raise ValueError("Evaluation policy hash does not match the manifest")
    if summary.get("profile_sha256") != expected_profile_sha256:
        raise ValueError("Evaluation profile hash does not match the manifest")
    bank = tuple(Scenario(**row) for row in summary["bank"])
    if bank_sha256(bank) != summary.get("bank_sha256"):
        raise ValueError("Evaluation bank payload does not match its recorded hash")
    terminal = summary["episodes"].get("terminal_success")
    if not isinstance(terminal, list) or len(terminal) != len(bank):
        raise ValueError("Evaluation episode payload does not match its bank")
    _summary_pose_successes(summary)


def _release_decision(
    summaries: Mapping[str, Mapping[str, Any]], *, nominal_tolerance: float
) -> dict[str, Any]:
    source_shifted = summaries["source_shifted"]
    adapted_shifted = summaries["adapted_shifted"]
    source_nominal = summaries["source_nominal"]
    adapted_nominal = summaries["adapted_nominal"]

    if source_shifted["bank_sha256"] != adapted_shifted["bank_sha256"]:
        raise ValueError("Shifted source/adapted evaluations are not paired")
    if source_nominal["bank_sha256"] != adapted_nominal["bank_sha256"]:
        raise ValueError("Nominal source/adapted evaluations are not paired")
    shifted_source = _summary_successes(source_shifted)
    shifted_adapted = _summary_successes(adapted_shifted)
    nominal_source = _summary_successes(source_nominal)
    nominal_adapted = _summary_successes(adapted_nominal)
    nominal_worlds = _summary_worlds(source_nominal)
    nominal_floor = nominal_source / nominal_worlds - nominal_tolerance
    source_pose = _summary_pose_successes(source_shifted)
    adapted_pose = _summary_pose_successes(adapted_shifted)
    weak_pose_improved = any(
        adapted_pose[pose] > source_pose[pose] for pose in POSES
    )
    passed = (
        shifted_adapted > shifted_source
        and weak_pose_improved
        and nominal_adapted / nominal_worlds >= nominal_floor
    )
    return {
        "status": "released" if passed else "rejected",
        "primary_gate": "terminal_stable_success",
        "task_return_role": "diagnostic_only",
        "shifted": {
            "source_successes": shifted_source,
            "adapted_successes": shifted_adapted,
            "worlds": _summary_worlds(source_shifted),
            "source_per_pose": source_pose,
            "adapted_per_pose": adapted_pose,
        },
        "nominal_retention": {
            "source_successes": nominal_source,
            "adapted_successes": nominal_adapted,
            "worlds": nominal_worlds,
            "tolerance": nominal_tolerance,
            "minimum_rate": nominal_floor,
            "passed": nominal_adapted / nominal_worlds >= nominal_floor,
        },
        "weak_pose_improved": weak_pose_improved,
        "passed": passed,
    }


def build_release_manifest(
    *,
    derivative_id: str,
    source_policy: Path,
    adapted_policy: Path,
    checkpoint: Path,
    export_verification: Path,
    training_dir: Path,
    summaries: Mapping[str, Path],
    evidence_dir: Path,
    output: Path,
    source_commit: str,
    checkpoint_repository: str,
    runtime_verification: Path,
    nominal_tolerance: float = 0.05,
) -> dict[str, Any]:
    """Build a self-verifying manifest from one completed post-training run."""

    missing = set(SUMMARY_ROLES) - set(summaries)
    if missing:
        raise ValueError(f"Missing evaluation summaries: {sorted(missing)}")
    source = import_deployed_policy(source_policy)
    adapted = import_deployed_policy(adapted_policy)
    verify_output_layer_derivative(source=source, adapted=adapted)

    checkpoint_payload = load_checkpoint(checkpoint)
    generation = int(checkpoint_payload["next_generation"])
    if generation != 100:
        raise ValueError(f"Canonical release requires generation 100, got {generation}")
    embedded_source = hashlib.sha256(checkpoint_payload["source_policy_model"]).hexdigest()
    if embedded_source != source.source_sha256:
        raise ValueError("Checkpoint embeds a different source policy")

    verification = _load_json(export_verification)
    if int(verification.get("checkpoint_generation", -1)) != generation:
        raise ValueError("Export verification names a different checkpoint generation")
    expected_hashes = {
        "checkpoint_sha256": sha256_file(checkpoint),
        "source_policy_sha256": source.source_sha256,
        "exported_policy_sha256": adapted.source_sha256,
    }
    for key, value in expected_hashes.items():
        if verification.get(key) != value:
            raise ValueError(f"Export verification {key} does not match local bytes")
    independent_error = runtime_parity(adapted)
    if independent_error >= 1.0e-5:
        raise ValueError(f"Adapted ONNX runtime parity {independent_error:.3g} failed")
    runtime_result = _load_json(runtime_verification)
    if runtime_result.get("passed") is not True:
        raise ValueError("MicroDuck production loader verification did not pass")
    if runtime_result.get("policy_sha256") != adapted.source_sha256:
        raise ValueError("Runtime verification belongs to a different policy")
    if runtime_result.get("production_path") != "duck_control::policy::Policy::load":
        raise ValueError("Runtime verification did not exercise the production loader")

    config = _load_json(training_dir / "config.json")
    baseline = _load_json(training_dir / "source_baseline.json")
    metrics = _load_jsonl(training_dir / "metrics.jsonl")
    trainer = config["trainer"]
    profile = DeploymentProfile(**config["optimization_profile"])
    retention_profile = DeploymentProfile(**config["retention_profile"])
    if len(metrics) != int(trainer["generations"]) or len(metrics) != generation:
        raise ValueError("Training metrics do not contain all completed generations")

    optimization_banks: list[str] = []
    optimization_scenarios: list[Scenario] = []
    for generation_index, row in enumerate(metrics):
        bank = make_balanced_bank(
            profile=profile,
            base_seed=int(trainer["seed"]) + generation_index * 10_007,
            episodes_per_pose=int(trainer["train_episodes_per_pose"]),
            prefix=f"train-g{generation_index:06d}",
        )
        digest = bank_sha256(bank)
        if row.get("train_bank_sha256") != digest:
            raise ValueError(f"Training bank {generation_index} hash mismatch")
        optimization_banks.append(digest)
        optimization_scenarios.extend(bank)

    selection_bank = make_balanced_bank(
        profile=profile,
        base_seed=int(trainer["seed"]) + 1_000_003,
        episodes_per_pose=int(trainer["heldout_episodes_per_pose"]),
        prefix="heldout-shift",
    )
    nominal_bank = make_balanced_bank(
        profile=retention_profile,
        base_seed=int(trainer["seed"]) + 2_000_003,
        episodes_per_pose=int(trainer["nominal_episodes_per_pose"]),
        prefix="heldout-nominal",
    )
    if bank_sha256(selection_bank) != baseline["heldout_bank_sha256"]:
        raise ValueError("Selection bank does not reproduce from the training contract")
    if bank_sha256(nominal_bank) != baseline["nominal_bank_sha256"]:
        raise ValueError("Retention bank does not reproduce from the training contract")

    summary_documents = {role: _load_json(path) for role, path in summaries.items()}
    _validate_summary(
        summary_documents["source_shifted"],
        expected_policy_sha256=source.source_sha256,
        expected_profile_sha256=profile.sha256,
    )
    _validate_summary(
        summary_documents["adapted_shifted"],
        expected_policy_sha256=adapted.source_sha256,
        expected_profile_sha256=profile.sha256,
    )
    _validate_summary(
        summary_documents["source_nominal"],
        expected_policy_sha256=source.source_sha256,
        expected_profile_sha256=retention_profile.sha256,
    )
    _validate_summary(
        summary_documents["adapted_nominal"],
        expected_policy_sha256=adapted.source_sha256,
        expected_profile_sha256=retention_profile.sha256,
    )

    final_shifted = tuple(
        Scenario(**row) for row in summary_documents["source_shifted"]["bank"]
    )
    final_nominal = tuple(
        Scenario(**row) for row in summary_documents["source_nominal"]["bank"]
    )
    bank_roles = {
        "optimization": _scenario_fingerprints(optimization_scenarios),
        "checkpoint_selection": _scenario_fingerprints(selection_bank),
        "nominal_retention": _scenario_fingerprints(nominal_bank),
        "final_shifted": _scenario_fingerprints(final_shifted),
        "final_nominal": _scenario_fingerprints(final_nominal),
    }
    _assert_disjoint(bank_roles)
    decision = _release_decision(
        summary_documents, nominal_tolerance=nominal_tolerance
    )
    if not decision["passed"]:
        raise ValueError("Release gates rejected the adapted policy")

    output.parent.mkdir(parents=True, exist_ok=True)
    evidence_dir.mkdir(parents=True, exist_ok=True)
    summary_records: dict[str, Any] = {}
    for role, original in summaries.items():
        destination = evidence_dir / f"{role}.summary.json"
        if original.resolve() != destination.resolve():
            destination.write_bytes(original.read_bytes())
        summary_records[role] = _summary_record(
            destination, summary_documents[role]
        )
        summary_records[role]["path"] = str(destination.relative_to(output.parent))

    manifest: dict[str, Any] = {
        "schema": RELEASE_SCHEMA,
        "schema_version": RELEASE_SCHEMA_VERSION,
        "derivative": {
            "id": derivative_id,
            "kind": "deployment-condition-adapter",
            "runtime_slot": "stand",
            "generation": generation,
        },
        "base_policy": {
            **source.metadata(),
            "filename": source_policy.name,
            "sha256": source.source_sha256,
            "tensor_sha256": _policy_tensor_hashes(source),
            "bundled": False,
        },
        "adapted_policy": {
            **adapted.metadata(),
            "filename": adapted_policy.name,
            "path": str(adapted_policy.relative_to(output.parent)),
            "sha256": adapted.source_sha256,
            "tensor_sha256": _policy_tensor_hashes(adapted),
            "independent_maximum_runtime_error": independent_error,
        },
        "adaptation": {
            "method": "EGGROLL output-layer post-training",
            "frozen_scope": config["frozen_scope"],
            "trainable_scope": config["search_scope"],
            "modified_parameters": 1_806,
            "configuration": trainer,
            "objective": config["objective"],
            "deployment_profile": config["optimization_profile"],
            "deployment_profile_sha256": profile.sha256,
        },
        "runtime_contract": {
            "model_api": 1,
            "observation": {
                "shape": [1, 61],
                "dtype": "float32",
                "normalizer": "baked into ONNX and frozen",
            },
            "action": {
                "shape": [1, 14],
                "dtype": "float32",
                "standing_action_scale": 1.0,
            },
            "graph": "Sub, Div, (Gemm, ELU) x3, Gemm; IR 8; opset 18",
            "production_loader_verification": {
                "path": str(runtime_verification.relative_to(output.parent)),
                "sha256": sha256_file(runtime_verification),
                "passed": True,
                "loader": "duck_control::policy::Policy::load",
            },
        },
        "episode_banks": {
            "optimization": {
                "count": len(optimization_banks),
                "worlds_per_bank": int(trainer["train_episodes_per_pose"]) * 4,
                "ordered_hashes_sha256": canonical_sha256(optimization_banks),
                "hashes": optimization_banks,
                "seed_rule": "trainer_seed + generation_index * 10007",
            },
            "checkpoint_selection": {
                "sha256": baseline["heldout_bank_sha256"],
                "worlds": len(selection_bank),
                "seed_rule": "trainer_seed + 1000003",
            },
            "nominal_retention": {
                "sha256": baseline["nominal_bank_sha256"],
                "worlds": len(nominal_bank),
                "seed_rule": "trainer_seed + 2000003",
            },
            "final_shifted": {
                "sha256": bank_sha256(final_shifted),
                "worlds": len(final_shifted),
            },
            "final_nominal": {
                "sha256": bank_sha256(final_nominal),
                "worlds": len(final_nominal),
            },
            "pairwise_disjoint": True,
        },
        "evaluation": summary_records,
        "release_decision": decision,
        "provenance": {
            "source_commit": source_commit,
            "checkpoint_repository": checkpoint_repository,
            "checkpoint_path": "run/alpha-stand-low-voltage-latency-v1/checkpoint_000100.pkl",
            "checkpoint_sha256": sha256_file(checkpoint),
            "export_verification_sha256": sha256_file(export_verification),
        },
        "rollback": {
            "target_runtime_slot": "stand",
            "target_filename": source_policy.name,
            "target_sha256": source.source_sha256,
            "mechanism": "robotctl update rollback model-stand",
        },
        "limitations": [
            "Simulation evidence only; no physical MicroDuck deployment is claimed.",
            "Nominal terminal capability was retained, not nominal task return or style.",
            "The result is specific to the declared lag/voltage/sag profile.",
        ],
    }
    write_json(output, manifest)
    return manifest


def verify_release_manifest(
    manifest_path: Path, *, source_policy_path: Path | None = None
) -> dict[str, Any]:
    """Verify local bytes, evidence, release gates, and optional base equivalence."""

    manifest = _load_json(manifest_path)
    if manifest.get("schema") != RELEASE_SCHEMA:
        raise ValueError("Unknown policy derivative manifest schema")
    if manifest.get("schema_version") != RELEASE_SCHEMA_VERSION:
        raise ValueError("Unsupported policy derivative manifest version")
    root = manifest_path.parent
    adapted_path = root / manifest["adapted_policy"]["path"]
    adapted = import_deployed_policy(adapted_path)
    if adapted.source_sha256 != manifest["adapted_policy"]["sha256"]:
        raise ValueError("Adapted policy bytes do not match the manifest")
    maximum_error = runtime_parity(adapted)
    if maximum_error >= 1.0e-5:
        raise ValueError(f"Adapted policy runtime parity {maximum_error:.3g} failed")
    adapted_hashes = _policy_tensor_hashes(adapted)
    if adapted_hashes != manifest["adapted_policy"]["tensor_sha256"]:
        raise ValueError("Adapted policy tensor hashes do not match the manifest")
    if adapted_hashes["frozen"] != manifest["base_policy"]["tensor_sha256"]["frozen"]:
        raise ValueError("Adapted policy changed a frozen base-policy tensor")
    if (
        adapted_hashes["output_weight"]
        == manifest["base_policy"]["tensor_sha256"]["output_weight"]
        and adapted_hashes["output_bias"]
        == manifest["base_policy"]["tensor_sha256"]["output_bias"]
    ):
        raise ValueError("Adapted policy output layer is identical to the base policy")
    runtime_record = manifest["runtime_contract"]["production_loader_verification"]
    runtime_path = root / runtime_record["path"]
    if sha256_file(runtime_path) != runtime_record["sha256"]:
        raise ValueError("Production loader verification hash mismatch")
    runtime_result = _load_json(runtime_path)
    if (
        runtime_result.get("passed") is not True
        or runtime_result.get("policy_sha256") != adapted.source_sha256
        or runtime_result.get("production_path")
        != "duck_control::policy::Policy::load"
    ):
        raise ValueError("Production loader verification does not match the policy")
    if source_policy_path is not None:
        source = import_deployed_policy(source_policy_path)
        if source.source_sha256 != manifest["base_policy"]["sha256"]:
            raise ValueError("Base policy bytes do not match the manifest")
        verify_output_layer_derivative(source=source, adapted=adapted)

    summaries: dict[str, dict[str, Any]] = {}
    for role in SUMMARY_ROLES:
        record = manifest["evaluation"][role]
        path = root / record["path"]
        if sha256_file(path) != record["sha256"]:
            raise ValueError(f"{role} evaluation evidence hash mismatch")
        summary = _load_json(path)
        if summary["bank_sha256"] != record["bank_sha256"]:
            raise ValueError(f"{role} bank hash does not match the manifest")
        if _summary_successes(summary) != record["terminal_successes"]:
            raise ValueError(f"{role} terminal result does not match the manifest")
        summaries[role] = summary
    decision = _release_decision(
        summaries,
        nominal_tolerance=float(
            manifest["release_decision"]["nominal_retention"]["tolerance"]
        ),
    )
    if decision != manifest["release_decision"] or not decision["passed"]:
        raise ValueError("Recorded release decision does not reproduce")
    return {
        "derivative_id": manifest["derivative"]["id"],
        "adapted_policy_sha256": adapted.source_sha256,
        "maximum_runtime_error": maximum_error,
        "output_layer_only": True,
        "production_loader_passed": True,
        "release_passed": True,
        "episode_banks_pairwise_disjoint": bool(
            manifest["episode_banks"]["pairwise_disjoint"]
        ),
    }
