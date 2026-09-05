"""Canonical contracts for a policy-agnostic EGGROLL patch campaign.

These types deliberately separate four things that earlier experiments coupled:

* the sealed production artifact;
* the capability and its real runtime command protocol;
* the deployment condition and black-box objective;
* the search and release gates.

Every contract has deterministic JSON and a content hash.  Metrics, checkpoints,
videos, and updater manifests can therefore bind to the exact campaign semantics
without trusting a directory name or an informal experiment description.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from typing import Any, Literal


def _canonical_value(value: Any) -> Any:
    if hasattr(value, "canonical_dict"):
        return value.canonical_dict()
    if isinstance(value, dict):
        return {str(key): _canonical_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_canonical_value(item) for item in value]
    return value


class CanonicalContract:
    """Mixin providing one deterministic serialization and identity rule."""

    def canonical_dict(self) -> dict[str, Any]:
        return _canonical_value(asdict(self))

    @property
    def canonical_json(self) -> str:
        return json.dumps(self.canonical_dict(), sort_keys=True, separators=(",", ":"))

    @property
    def sha256(self) -> str:
        return hashlib.sha256(self.canonical_json.encode()).hexdigest()


@dataclass(frozen=True)
class PolicyArtifactSpec(CanonicalContract):
    """One sealed ONNX artifact and the production slot it is allowed to fill."""

    artifact_id: str
    filename: str
    expected_sha256: str
    capability_id: str
    runtime_slot: str
    updater_component: str
    runtime_net: str
    runtime_modes: tuple[str, ...]
    task_ids: tuple[str, ...]
    source_provenance: str
    input_width: int = 61
    output_width: int = 14

    def __post_init__(self) -> None:
        if not self.artifact_id or not self.capability_id:
            raise ValueError("artifact_id and capability_id cannot be empty")
        if not self.filename.endswith(".onnx") or "/" in self.filename:
            raise ValueError("filename must be one local ONNX basename")
        if len(self.expected_sha256) != 64:
            raise ValueError("expected_sha256 must be a lowercase SHA-256")
        if self.expected_sha256 != self.expected_sha256.lower():
            raise ValueError("expected_sha256 must be lowercase")
        if self.input_width != 61 or self.output_width != 14:
            raise ValueError("MicroDuck production artifacts must remain 61D -> 14D")
        if not self.updater_component.startswith("model-"):
            raise ValueError("updater_component must be one model-<artifact> component")
        if not self.runtime_modes or not self.task_ids:
            raise ValueError("artifact runtime modes and task ids cannot be empty")


@dataclass(frozen=True)
class CapabilitySpec(CanonicalContract):
    """Behavioral contract independent of a particular artifact or fault."""

    capability_id: str
    policy_class: str
    driver: str
    artifact_ids: tuple[str, ...]
    task_ids: tuple[str, ...]
    command_protocol: str
    initial_state_protocol: str
    success_semantics: tuple[str, ...]
    terminal_state: str

    def __post_init__(self) -> None:
        if not self.capability_id or not self.driver:
            raise ValueError("capability_id and driver cannot be empty")
        if not self.artifact_ids or not self.task_ids:
            raise ValueError("capability artifacts and task ids cannot be empty")
        if not self.success_semantics:
            raise ValueError("capability success semantics cannot be empty")


@dataclass(frozen=True)
class DeploymentCondition(CanonicalContract):
    """One observable or hidden deployment change applied during evaluation."""

    condition_id: str
    adapter: str
    parameters: tuple[tuple[str, str | int | float | bool], ...]
    hidden_from_actor: bool
    description: str

    def __post_init__(self) -> None:
        names = [name for name, _value in self.parameters]
        if not self.condition_id or not self.adapter:
            raise ValueError("condition_id and adapter cannot be empty")
        if len(names) != len(set(names)):
            raise ValueError("deployment condition parameter names must be unique")


@dataclass(frozen=True)
class ObjectiveSpec(CanonicalContract):
    """Trajectory ordering evaluated without differentiation."""

    objective_id: str
    evaluator: str
    lexicographic_metrics: tuple[str, ...]
    diagnostics: tuple[str, ...]
    description: str
    differentiable: bool = False

    def __post_init__(self) -> None:
        if not self.objective_id or not self.evaluator:
            raise ValueError("objective_id and evaluator cannot be empty")
        if not self.lexicographic_metrics:
            raise ValueError("an objective needs at least one ordered metric")
        if self.differentiable:
            raise ValueError("Autopatch objectives must be evaluation-only")


@dataclass(frozen=True)
class ReleaseGate(CanonicalContract):
    """A hard gate that must pass before a derivative can be packaged."""

    gate_id: str
    evaluator: str
    metric: str
    comparator: Literal[">", ">=", "==", "<=", "<"]
    threshold: float
    profile_role: Literal["target", "nominal", "parity", "runtime"]
    consecutive_passes: int = 1

    def __post_init__(self) -> None:
        if not self.gate_id or not self.metric:
            raise ValueError("gate_id and metric cannot be empty")
        if self.consecutive_passes <= 0:
            raise ValueError("consecutive_passes must be positive")


@dataclass(frozen=True)
class ReleaseScope(CanonicalContract):
    """Deployment applicability and retention semantics for one derivative.

    This is deliberately separate from :class:`PatchCampaign`: selecting where a
    finished derivative may run must not rewrite the immutable training campaign or
    invalidate its checkpoints.  The scope is content-addressed and is bound into the
    release envelope alongside the exact source and adapted policy bytes.
    """

    scope_id: str
    mode: Literal["profile_specific", "multi_profile"]
    profile_sha256s: tuple[tuple[str, str], ...]
    required_retention_roles: tuple[str, ...]
    activation_profile_role: str | None
    activation_predicate: str
    source_fallback_sha256: str
    unknown_profile_action: Literal["retain_source", "block_adapted_policy"]

    def __post_init__(self) -> None:
        if self.mode not in ("profile_specific", "multi_profile"):
            raise ValueError(
                "release scope mode must be profile_specific or multi_profile"
            )
        if self.unknown_profile_action not in (
            "retain_source",
            "block_adapted_policy",
        ):
            raise ValueError("unknown profile action must fail closed")
        if not self.scope_id or not self.activation_predicate:
            raise ValueError(
                "release scope id and activation predicate cannot be empty"
            )
        if len(self.source_fallback_sha256) != 64:
            raise ValueError("release scope source fallback must be a SHA-256")
        if self.source_fallback_sha256 != self.source_fallback_sha256.lower():
            raise ValueError("release scope source fallback must be lowercase")
        roles = [role for role, _sha256 in self.profile_sha256s]
        if not roles or len(roles) != len(set(roles)):
            raise ValueError("release scope profile roles must be non-empty and unique")
        if not self.required_retention_roles or len(
            self.required_retention_roles
        ) != len(set(self.required_retention_roles)):
            raise ValueError(
                "release scope retention roles must be non-empty and unique"
            )
        unknown_roles = set(self.required_retention_roles) - set(roles)
        if unknown_roles:
            raise ValueError(
                f"release scope has unknown retention roles: {unknown_roles}"
            )
        for role, sha256 in self.profile_sha256s:
            if not role or len(sha256) != 64 or sha256 != sha256.lower():
                raise ValueError(
                    "release scope profiles need names and lowercase SHA-256s"
                )

        if self.mode == "profile_specific":
            if self.activation_profile_role is None:
                raise ValueError("profile-specific scope needs one activation profile")
            if self.required_retention_roles != (self.activation_profile_role,):
                raise ValueError(
                    "profile-specific scope retains exactly its activation profile"
                )
            if self.unknown_profile_action != "retain_source":
                raise ValueError(
                    "profile-specific scope must retain source on an unknown profile"
                )
        else:
            if self.activation_profile_role is not None:
                raise ValueError(
                    "multi-profile scope cannot name one activation profile"
                )
            if set(self.required_retention_roles) != set(roles):
                raise ValueError(
                    "multi-profile scope must retain every declared profile"
                )

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> ReleaseScope:
        document = dict(value)
        document["profile_sha256s"] = tuple(
            (str(role), str(sha256)) for role, sha256 in document["profile_sha256s"]
        )
        document["required_retention_roles"] = tuple(
            str(role) for role in document["required_retention_roles"]
        )
        return cls(**document)

    @classmethod
    def from_json(cls, payload: str) -> ReleaseScope:
        value = json.loads(payload)
        if not isinstance(value, dict):
            raise TypeError("release scope JSON must contain one object")
        return cls.from_dict(value)


@dataclass(frozen=True)
class OptimizerSpec(CanonicalContract):
    """Forward-only search settings, separate from capability semantics."""

    algorithm: str
    trainable_scope: str
    rank: int
    population: int
    sigma: float
    learning_rate: float
    generations: int
    seed: int
    worlds_per_candidate: int = 1

    def __post_init__(self) -> None:
        if not self.algorithm or not self.trainable_scope:
            raise ValueError("optimizer algorithm and trainable scope cannot be empty")
        if (
            self.rank <= 0
            or self.population <= 1
            or self.generations <= 0
            or self.worlds_per_candidate <= 0
        ):
            raise ValueError(
                "rank, population, generations, and worlds_per_candidate must be positive"
            )
        if self.sigma <= 0.0 or self.learning_rate <= 0.0:
            raise ValueError("sigma and learning_rate must be positive")


@dataclass(frozen=True)
class PatchCampaign(CanonicalContract):
    """Complete immutable input to one Autopatch search and release decision."""

    campaign_id: str
    artifact_id: str
    artifact_sha256: str
    capability_id: str
    condition: DeploymentCondition
    objective: ObjectiveSpec
    optimizer: OptimizerSpec
    gates: tuple[ReleaseGate, ...]
    calibration_bank_sha256: str
    held_out_bank_sha256: str

    def __post_init__(self) -> None:
        if not self.campaign_id or not self.artifact_id or not self.capability_id:
            raise ValueError("campaign, artifact, and capability ids cannot be empty")
        if len(self.artifact_sha256) != 64:
            raise ValueError("campaign artifact_sha256 must be a SHA-256")
        if not self.gates:
            raise ValueError("campaign needs release gates")
        if self.calibration_bank_sha256 == self.held_out_bank_sha256:
            raise ValueError("calibration and held-out banks must be disjoint")
        for name, value in (
            ("calibration_bank_sha256", self.calibration_bank_sha256),
            ("held_out_bank_sha256", self.held_out_bank_sha256),
        ):
            if len(value) != 64:
                raise ValueError(f"{name} must be a SHA-256")

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> PatchCampaign:
        """Load a strict campaign document; unknown fields remain errors."""

        document = dict(value)
        try:
            document["condition"] = DeploymentCondition(**document["condition"])
            document["objective"] = ObjectiveSpec(**document["objective"])
            document["optimizer"] = OptimizerSpec(**document["optimizer"])
            document["gates"] = tuple(ReleaseGate(**gate) for gate in document["gates"])
        except KeyError as error:
            raise ValueError(f"campaign is missing {error.args[0]!r}") from error
        return cls(**document)

    @classmethod
    def from_json(cls, payload: str) -> PatchCampaign:
        value = json.loads(payload)
        if not isinstance(value, dict):
            raise TypeError("campaign JSON must contain one object")
        return cls.from_dict(value)


@dataclass(frozen=True)
class TransitionSpec(CanonicalContract):
    """A runtime transition that must remain valid after patching one capability."""

    source_capability: str
    target_capability: str
    trigger: str
    runtime_priority: int
    required_artifact_ids: tuple[str, ...]
    acceptance: tuple[str, ...]

    def __post_init__(self) -> None:
        if not self.source_capability or not self.target_capability:
            raise ValueError("transition endpoints cannot be empty")
        if self.runtime_priority < 0:
            raise ValueError("runtime_priority cannot be negative")
        if not self.required_artifact_ids or not self.acceptance:
            raise ValueError("transition artifacts and acceptance cannot be empty")
