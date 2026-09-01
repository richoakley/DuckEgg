"""Canonical inventory of production MicroDuck policies and capabilities."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from mjlab_microduck.eggroll.policy_io import import_deployed_policy

from .contracts import CapabilitySpec, PatchCampaign, PolicyArtifactSpec, TransitionSpec

PROVENANCE = (
    "pollen-robotics/microduck example_policies; copied from "
    "apirrone/microduck_runtime at 5f3b314 (roulade first at 7e4ab6d)"
)


def _artifact(
    artifact_id: str,
    filename: str,
    sha256: str,
    capability_id: str,
    runtime_slot: str,
    updater_component: str,
    runtime_net: str,
    runtime_modes: tuple[str, ...],
    task_ids: tuple[str, ...],
) -> PolicyArtifactSpec:
    return PolicyArtifactSpec(
        artifact_id=artifact_id,
        filename=filename,
        expected_sha256=sha256,
        capability_id=capability_id,
        runtime_slot=runtime_slot,
        updater_component=updater_component,
        runtime_net=runtime_net,
        runtime_modes=runtime_modes,
        task_ids=task_ids,
        source_provenance=PROVENANCE,
    )


ARTIFACTS = (
    _artifact(
        "alpha-walking",
        "alpha_walking.onnx",
        "e36332d383997d51401897734cd3e79cf5038406feddb18b4d57ecfb141daa6c",
        "legged-locomotion",
        "walk",
        "model-walk",
        "Walk",
        ("walk",),
        (
            "Mjlab-Velocity-Flat-MicroDuck",
            "Mjlab-Velocity-Rough-MicroDuck",
            "Mjlab-VelStand-Flat-MicroDuck",
            "Mjlab-VelStand-Rough-MicroDuck",
        ),
    ),
    _artifact(
        "alpha-stand",
        "alpha_stand.onnx",
        "1569268713e40deea795dd2922dba50d3621e15a872855408b6b1b125b1c094b",
        "stationary-body-control",
        "stand",
        "model-stand",
        "Stand",
        ("walk",),
        (
            "Mjlab-VelStand-Flat-MicroDuck",
            "Mjlab-VelStand-Rough-MicroDuck",
            "Mjlab-StandUp-Flat-MicroDuck",
            "Mjlab-StandUp-Rough-MicroDuck",
        ),
    ),
    _artifact(
        "alpha-sitstand",
        "alpha_sitstand.onnx",
        "c6c40e35e726eabd803d633e090d112994f469921152448367953fbaf9799bc8",
        "sit-stand-transition",
        "sitstand",
        "model-sitstand",
        "SitStand",
        ("walk", "roller"),
        ("Mjlab-SitStand-Flat-MicroDuck", "Mjlab-SitStand-Rough-MicroDuck"),
    ),
    _artifact(
        "alpha-ground-pick",
        "alpha_ground_pick.onnx",
        "ffbf5109982ff999b0ba53afe86b9ae731bbec679d67fb7f8ab4c52152c88872",
        "ground-pick",
        "ground_pick",
        "model-ground-pick",
        "GroundPick",
        ("walk",),
        ("Mjlab-GroundPick-Flat-MicroDuck", "Mjlab-GroundPick-Rough-MicroDuck"),
    ),
    _artifact(
        "ball-kick-left",
        "ball_kick_left.onnx",
        "d6928284dccd3dd61e08bf2f760effa74309fbefd97b2b31afb2a60f526d196a",
        "ball-kick",
        "kick_left",
        "model-kick-left",
        "KickLeft",
        ("walk", "roller"),
        ("Mjlab-BallKick-Flat-MicroDuck",),
    ),
    _artifact(
        "ball-kick-right",
        "ball_kick_right.onnx",
        "147a32c388c6b19111b3ac3b550a9a6dc8b8bf267118af4d8c3712522eedb5af",
        "ball-kick",
        "kick_right",
        "model-kick-right",
        "KickRight",
        ("walk", "roller"),
        ("Mjlab-BallKick-Flat-MicroDuck",),
    ),
    _artifact(
        "roller",
        "roller.onnx",
        "cf05651d2708a2f9364212e86b866c97a70ace8131c492500105e8f28bf99afd",
        "roller-locomotion",
        "walk",
        "model-roller",
        "Walk",
        ("roller",),
        ("Mjlab-Velocity-Flat-MicroDuck-Rollers",),
    ),
    _artifact(
        "roller-crouch",
        "roller_crouch.onnx",
        "a1a084be240469c76ac9d3fa44d4792f16d4b1da60398b3ecd3cfc5e2244d990",
        "roller-crouch",
        "ground_pick",
        "model-roller-crouch",
        "GroundPick",
        ("roller",),
        ("Mjlab-RollerCrouch-Flat-MicroDuck",),
    ),
    _artifact(
        "roulade",
        "roulade.onnx",
        "3d60da08fc13f29c1b57f41977aa898132c0d60042100149d8e775affcbca32b",
        "forward-roll",
        "roulade",
        "model-roulade",
        "Roulade",
        ("walk", "roller"),
        ("Mjlab-Roulade-Flat-MicroDuck",),
    ),
)


CAPABILITIES = (
    CapabilitySpec(
        "legged-locomotion",
        "continuous",
        "twist-command",
        ("alpha-walking",),
        ARTIFACTS[0].task_ids,
        "twist[0:3]=vx,vy,yaw-rate; head[3:7] and body[7:13] remain live",
        "task reset distribution plus commanded zero and non-zero velocity buckets",
        ("velocity tracking", "upright locomotion", "terminal stability"),
        "upright-moving-or-stationary",
    ),
    CapabilitySpec(
        "stationary-body-control",
        "recovery-and-pose",
        "stand-selection",
        ("alpha-stand",),
        ARTIFACTS[1].task_ids,
        "zero twist selects Stand; head[3:7] and body-pose[7:13] remain live",
        "registered VelStand/StandUp reset distributions, including fallen poses",
        ("terminal supported upright hold", "body/head pose tracking", "fall recovery"),
        "supported-standing",
    ),
    CapabilitySpec(
        "sit-stand-transition",
        "commanded-transition",
        "posture-toggle",
        ("alpha-sitstand",),
        ARTIFACTS[2].task_ids,
        "twist vx is posture flag: 1=sit, 0=rise/stand; head/body slots remain live",
        "standing and sitting task states with scheduler transition timing",
        ("reaches commanded posture", "low-impact transition", "stable terminal hold"),
        "commanded-sitting-or-standing",
    ),
    CapabilitySpec(
        "ground-pick",
        "phase-scripted",
        "ground-pick-trigger",
        ("alpha-ground-pick",),
        ARTIFACTS[3].task_ids,
        "twist=[cos(phi),sin(phi),0]; head/body zero; runtime phi advances over period",
        "standing task reset followed by the production phase scheduler",
        ("mouth reaches target", "support maintained", "returns to stable stand"),
        "supported-standing",
    ),
    CapabilitySpec(
        "ball-kick",
        "event-triggered",
        "kick-trigger",
        ("ball-kick-left", "ball-kick-right"),
        ARTIFACTS[4].task_ids,
        "all 13 command slots zero while the selected kick network runs",
        "standing robot plus registered ball placement/randomization",
        ("correct foot contact", "ball displacement/velocity", "post-kick stability"),
        "supported-standing",
    ),
    CapabilitySpec(
        "roller-locomotion",
        "continuous",
        "twist-command-roller-mode",
        ("roller",),
        ARTIFACTS[6].task_ids,
        "walk slot in roller mode; twist/head/body use the shared 13D command block",
        "registered passive-wheel robot and roller locomotion reset distribution",
        ("roller velocity tracking", "uprightness", "wheel/foot support"),
        "upright-rolling-or-stationary",
    ),
    CapabilitySpec(
        "roller-crouch",
        "event-triggered",
        "ground-pick-trigger-roller-mode",
        ("roller-crouch",),
        ARTIFACTS[7].task_ids,
        "ground_pick slot in roller mode; production scheduler phase command and period",
        "upright passive-wheel robot using the registered RollerCrouch task",
        ("reaches crouch", "maintains roller support", "returns upright"),
        "upright-on-rollers",
    ),
    CapabilitySpec(
        "forward-roll",
        "event-triggered-dynamic",
        "roulade-trigger",
        ("roulade",),
        ARTIFACTS[8].task_ids,
        "all 13 command slots zero; selecting Roulade initiates motion immediately",
        "registered standing start and production one-second scheduler window",
        (
            "completes sagittal roll",
            "avoids disallowed support",
            "lands stably upright",
        ),
        "supported-standing",
    ),
)


TRANSITIONS = (
    TransitionSpec(
        "stationary-body-control",
        "legged-locomotion",
        "non-zero twist",
        6,
        ("alpha-stand", "alpha-walking"),
        ("no target discontinuity", "upright after handoff"),
    ),
    TransitionSpec(
        "legged-locomotion",
        "stationary-body-control",
        "twist magnitude <= standing threshold",
        6,
        ("alpha-walking", "alpha-stand"),
        ("settles without fall", "body command becomes active"),
    ),
    TransitionSpec(
        "stationary-body-control",
        "sit-stand-transition",
        "sit toggle",
        4,
        ("alpha-stand", "alpha-sitstand"),
        ("reaches seat", "scheduler state is Sitting"),
    ),
    TransitionSpec(
        "sit-stand-transition",
        "stationary-body-control",
        "rise timer expiry",
        4,
        ("alpha-sitstand", "alpha-stand"),
        ("upright at handoff", "no action jump failure"),
    ),
    TransitionSpec(
        "stationary-body-control",
        "ground-pick",
        "ground-pick trigger",
        3,
        ("alpha-stand", "alpha-ground-pick"),
        ("production phase command used", "returns to stand"),
    ),
    TransitionSpec(
        "stationary-body-control",
        "ball-kick",
        "left/right kick trigger",
        2,
        ("alpha-stand", "ball-kick-left", "ball-kick-right"),
        ("zero command during kick", "returns to stand"),
    ),
    TransitionSpec(
        "stationary-body-control",
        "forward-roll",
        "roulade trigger",
        1,
        ("alpha-stand", "roulade"),
        ("zero command during roll", "returns to stand"),
    ),
    TransitionSpec(
        "roller-locomotion",
        "roller-crouch",
        "ground-pick trigger in roller mode",
        3,
        ("roller", "roller-crouch"),
        ("roller tuning used", "returns to roller locomotion"),
    ),
)

TRAINING_TASKS_WITHOUT_PRODUCTION_ARTIFACT = (
    "Mjlab-Velocity-Swizzle-MicroDuck",
    "Mjlab-RollerSlope-Flat-MicroDuck",
    "Mjlab-RollerStandUp-Flat-MicroDuck",
    "Mjlab-Spin-Flat-MicroDuck",
)


@dataclass(frozen=True)
class AutopatchRegistry:
    artifacts: tuple[PolicyArtifactSpec, ...]
    capabilities: tuple[CapabilitySpec, ...]
    transitions: tuple[TransitionSpec, ...]
    training_tasks_without_production_artifact: tuple[str, ...]

    def __post_init__(self) -> None:
        self._unique("artifact", [item.artifact_id for item in self.artifacts])
        self._unique("filename", [item.filename for item in self.artifacts])
        self._unique(
            "updater component", [item.updater_component for item in self.artifacts]
        )
        self._unique("capability", [item.capability_id for item in self.capabilities])
        artifacts = {item.artifact_id: item for item in self.artifacts}
        capabilities = {item.capability_id: item for item in self.capabilities}
        for artifact in self.artifacts:
            if artifact.capability_id not in capabilities:
                raise ValueError(f"Unknown capability {artifact.capability_id!r}")
        for capability in self.capabilities:
            for artifact_id in capability.artifact_ids:
                if artifact_id not in artifacts:
                    raise ValueError(f"Unknown artifact {artifact_id!r}")
                if artifacts[artifact_id].capability_id != capability.capability_id:
                    raise ValueError(
                        f"Artifact {artifact_id!r} is assigned inconsistently"
                    )
        for transition in self.transitions:
            if transition.source_capability not in capabilities:
                raise ValueError(f"Unknown source {transition.source_capability!r}")
            if transition.target_capability not in capabilities:
                raise ValueError(f"Unknown target {transition.target_capability!r}")
            for artifact_id in transition.required_artifact_ids:
                if artifact_id not in artifacts:
                    raise ValueError(f"Unknown transition artifact {artifact_id!r}")

    @staticmethod
    def _unique(label: str, values: list[str]) -> None:
        if len(values) != len(set(values)):
            raise ValueError(f"Duplicate {label} in Autopatch registry")

    def artifact(self, artifact_id: str) -> PolicyArtifactSpec:
        try:
            return next(
                item for item in self.artifacts if item.artifact_id == artifact_id
            )
        except StopIteration as error:
            raise KeyError(f"unknown production artifact {artifact_id!r}") from error

    def capability(self, capability_id: str) -> CapabilitySpec:
        try:
            return next(
                item
                for item in self.capabilities
                if item.capability_id == capability_id
            )
        except StopIteration as error:
            raise KeyError(f"unknown capability {capability_id!r}") from error

    def release_test_plan(self, artifact_id: str) -> dict[str, Any]:
        """Return the mandatory capability node and scheduler edges for a patch."""

        artifact = self.artifact(artifact_id)
        capability = self.capability(artifact.capability_id)
        edges = tuple(
            transition
            for transition in self.transitions
            if artifact_id in transition.required_artifact_ids
        )
        if not edges:
            raise ValueError(f"{artifact_id!r} has no scheduler-edge release coverage")
        return {
            "artifact_id": artifact_id,
            "artifact_sha256": artifact.expected_sha256,
            "node": {
                "capability_id": capability.capability_id,
                "capability_sha256": capability.sha256,
                "task_ids": list(capability.task_ids),
                "acceptance": list(capability.success_semantics),
            },
            "edges": [
                {
                    **transition.canonical_dict(),
                    "transition_sha256": transition.sha256,
                }
                for transition in edges
            ],
        }

    def validate_campaign(self, campaign: PatchCampaign) -> None:
        artifact = self.artifact(campaign.artifact_id)
        if campaign.artifact_sha256 != artifact.expected_sha256:
            raise ValueError("Campaign is not bound to the sealed production artifact")
        if campaign.capability_id != artifact.capability_id:
            raise ValueError("Campaign capability does not own its artifact")

    def validate_runtime_artifacts(
        self, runtime_repo: Path
    ) -> tuple[dict[str, Any], ...]:
        policy_dir = runtime_repo / "example_policies"
        expected_names = {artifact.filename for artifact in self.artifacts}
        actual_names = {path.name for path in policy_dir.glob("*.onnx")}
        if actual_names != expected_names:
            missing = sorted(expected_names - actual_names)
            unexpected = sorted(actual_names - expected_names)
            raise ValueError(
                f"Production policy inventory drifted; missing={missing}, unexpected={unexpected}"
            )
        reports: list[dict[str, Any]] = []
        for artifact in self.artifacts:
            path = policy_dir / artifact.filename
            policy = import_deployed_policy(path)
            if policy.source_sha256 != artifact.expected_sha256:
                raise ValueError(
                    f"{artifact.filename} hash changed: {policy.source_sha256}; "
                    f"expected {artifact.expected_sha256}"
                )
            reports.append(
                {
                    "artifact_id": artifact.artifact_id,
                    "capability_id": artifact.capability_id,
                    "filename": artifact.filename,
                    "sha256": policy.source_sha256,
                    "input_name": policy.input_name,
                    "output_name": policy.output_name,
                    "input_width": artifact.input_width,
                    "output_width": artifact.output_width,
                    "runtime_slot": artifact.runtime_slot,
                    "updater_component": artifact.updater_component,
                    "runtime_net": artifact.runtime_net,
                    "runtime_modes": list(artifact.runtime_modes),
                }
            )
        return tuple(reports)

    def to_dict(self) -> dict[str, Any]:
        return {
            "artifacts": [item.canonical_dict() for item in self.artifacts],
            "capabilities": [item.canonical_dict() for item in self.capabilities],
            "transitions": [item.canonical_dict() for item in self.transitions],
            "training_tasks_without_production_artifact": list(
                self.training_tasks_without_production_artifact
            ),
        }


PRODUCTION_REGISTRY = AutopatchRegistry(
    artifacts=ARTIFACTS,
    capabilities=CAPABILITIES,
    transitions=TRANSITIONS,
    training_tasks_without_production_artifact=TRAINING_TASKS_WITHOUT_PRODUCTION_ARTIFACT,
)
