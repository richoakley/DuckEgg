"""Versioned simulated-deployment shifts and deterministic scenario banks."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import torch

POSES = ("standing", "sitting", "face-down", "face-up")
ZERO_COMMAND = (0.0,) * 13


@dataclass(frozen=True)
class DeploymentProfile:
    """Hidden actuator conditions applied without changing actor observations."""

    name: str
    battery_voltage: float
    voltage_sag_gain: float
    actuator_lag_steps: int

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("Deployment profile name cannot be empty")
        if not 6.0 <= self.battery_voltage <= 8.2:
            raise ValueError("battery_voltage must remain within BAM's safe range")
        if not 0.0 <= self.voltage_sag_gain <= 0.5:
            raise ValueError("voltage_sag_gain must be in [0, 0.5]")
        if not 0 <= self.actuator_lag_steps <= 30:
            raise ValueError("actuator_lag_steps must be in the compiled buffer range")

    def canonical_dict(self) -> dict[str, Any]:
        return asdict(self)

    @property
    def sha256(self) -> str:
        payload = json.dumps(
            self.canonical_dict(), sort_keys=True, separators=(",", ":")
        ).encode()
        return hashlib.sha256(payload).hexdigest()


@dataclass(frozen=True)
class AsymmetricActuatorProfile:
    """A hidden, named per-joint torque-effectiveness deployment fault."""

    name: str
    base_profile: DeploymentProfile
    joint_effectiveness: tuple[tuple[str, float], ...]

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("Asymmetric profile name cannot be empty")
        names = [name for name, _value in self.joint_effectiveness]
        if not names or len(names) != len(set(names)):
            raise ValueError("Affected joint names must be non-empty and unique")
        if any(not 0.0 < value <= 1.0 for _name, value in self.joint_effectiveness):
            raise ValueError("Joint effectiveness must be in (0, 1]")

    def canonical_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "base_profile": self.base_profile.canonical_dict(),
            "joint_effectiveness": {
                name: value for name, value in self.joint_effectiveness
            },
        }

    @property
    def sha256(self) -> str:
        payload = json.dumps(
            self.canonical_dict(), sort_keys=True, separators=(",", ":")
        ).encode()
        return hashlib.sha256(payload).hexdigest()

    @property
    def actuator_lag_steps(self) -> int:
        return self.base_profile.actuator_lag_steps


@dataclass(frozen=True)
class FootGeometryProfile:
    """A hidden replacement-sole geometry applied to both collision meshes.

    The scale is uniform so the compiled convex-mesh normals remain valid.  It
    intentionally changes contact geometry only: mass and inertia stay fixed,
    matching a lightweight 3D-printed shoe or replacement sole.
    """

    name: str
    base_profile: DeploymentProfile
    uniform_scale: float

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("Foot geometry profile name cannot be empty")
        if not 0.5 <= self.uniform_scale <= 1.75:
            raise ValueError("uniform_scale must be in [0.5, 1.75]")

    def canonical_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "base_profile": self.base_profile.canonical_dict(),
            "uniform_scale": self.uniform_scale,
            "geometry_scope": {
                "geoms": ["left_foot_collision", "right_foot_collision"],
                "mass_inertia_changed": False,
            },
        }

    @property
    def sha256(self) -> str:
        payload = json.dumps(
            self.canonical_dict(), sort_keys=True, separators=(",", ":")
        ).encode()
        return hashlib.sha256(payload).hexdigest()

    @property
    def actuator_lag_steps(self) -> int:
        return self.base_profile.actuator_lag_steps


@dataclass(frozen=True)
class ReplacementFootProfile:
    """A replacement foot with changed geometry and sole material."""

    name: str
    base_profile: DeploymentProfile
    uniform_scale: float
    friction_scale: float

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("Replacement foot profile name cannot be empty")
        if not 0.5 <= self.uniform_scale <= 1.75:
            raise ValueError("uniform_scale must be in [0.5, 1.75]")
        if not 0.1 <= self.friction_scale <= 1.5:
            raise ValueError("friction_scale must be in [0.1, 1.5]")

    def canonical_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "base_profile": self.base_profile.canonical_dict(),
            "uniform_scale": self.uniform_scale,
            "friction_scale": self.friction_scale,
            "geometry_scope": {
                "geoms": ["left_foot_collision", "right_foot_collision"],
                "mass_inertia_changed": False,
            },
            "material_scope": {
                "field": "geom_friction",
                "operation": "scale_from_compiled_startup_baseline",
            },
        }

    @property
    def sha256(self) -> str:
        payload = json.dumps(
            self.canonical_dict(), sort_keys=True, separators=(",", ":")
        ).encode()
        return hashlib.sha256(payload).hexdigest()

    @property
    def actuator_lag_steps(self) -> int:
        return self.base_profile.actuator_lag_steps


@dataclass(frozen=True)
class PriorityFootMaterialProfile:
    """Replacement foot whose material owns MuJoCo contact-pair friction."""

    name: str
    base_profile: DeploymentProfile
    uniform_scale: float
    slide_friction: float
    contact_priority: int = 1

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("Priority foot material profile name cannot be empty")
        if not 0.5 <= self.uniform_scale <= 1.75:
            raise ValueError("uniform_scale must be in [0.5, 1.75]")
        if not 0.02 <= self.slide_friction <= 1.5:
            raise ValueError("slide_friction must be in [0.02, 1.5]")
        if self.contact_priority <= 0:
            raise ValueError("replacement material must outrank the floor")

    def canonical_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "base_profile": self.base_profile.canonical_dict(),
            "uniform_scale": self.uniform_scale,
            "contact_material": {
                "friction": [
                    self.slide_friction,
                    self.slide_friction * 0.005,
                    self.slide_friction * 0.0001,
                ],
                "geom_priority": self.contact_priority,
                "pair_semantics": "higher-priority replacement foot owns friction",
            },
            "geometry_scope": {
                "geoms": ["left_foot_collision", "right_foot_collision"],
                "mass_inertia_changed": False,
            },
        }

    @property
    def sha256(self) -> str:
        payload = json.dumps(
            self.canonical_dict(), sort_keys=True, separators=(",", ":")
        ).encode()
        return hashlib.sha256(payload).hexdigest()

    @property
    def actuator_lag_steps(self) -> int:
        return self.base_profile.actuator_lag_steps


@dataclass(frozen=True)
class WedgeFootProfile:
    """A symmetric replacement sole with a fixed sagittal wedge angle."""

    name: str
    base_profile: DeploymentProfile
    pitch_degrees: float

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("Wedge foot profile name cannot be empty")
        if not 0.0 < self.pitch_degrees <= 30.0:
            raise ValueError("pitch_degrees must be in (0, 30]")

    def canonical_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "base_profile": self.base_profile.canonical_dict(),
            "pitch_degrees": self.pitch_degrees,
            "geometry_scope": {
                "geoms": ["left_foot_collision", "right_foot_collision"],
                "axis": "local-y",
                "mass_inertia_changed": False,
            },
        }

    @property
    def sha256(self) -> str:
        payload = json.dumps(
            self.canonical_dict(), sort_keys=True, separators=(",", ":")
        ).encode()
        return hashlib.sha256(payload).hexdigest()

    @property
    def actuator_lag_steps(self) -> int:
        return self.base_profile.actuator_lag_steps


type DeploymentConditionProfile = (
    DeploymentProfile
    | AsymmetricActuatorProfile
    | FootGeometryProfile
    | ReplacementFootProfile
    | PriorityFootMaterialProfile
    | WedgeFootProfile
)


NOMINAL_PROFILE = DeploymentProfile(
    name="nominal-fixed-median-v1",
    battery_voltage=7.35,
    voltage_sag_gain=0.10,
    actuator_lag_steps=4,
)

CALIBRATION_LAG12 = DeploymentProfile(
    name="calibration-6p8V-lag12",
    battery_voltage=6.8,
    voltage_sag_gain=0.15,
    actuator_lag_steps=12,
)

CALIBRATION_LAG14 = DeploymentProfile(
    name="calibration-6p5V-lag14",
    battery_voltage=6.5,
    voltage_sag_gain=0.20,
    actuator_lag_steps=14,
)

CALIBRATION_LAG16 = DeploymentProfile(
    name="calibration-6p5V-lag16",
    battery_voltage=6.5,
    voltage_sag_gain=0.20,
    actuator_lag_steps=16,
)

CALIBRATION_LAG20 = DeploymentProfile(
    name="calibration-6p2V-lag20",
    battery_voltage=6.2,
    voltage_sag_gain=0.25,
    actuator_lag_steps=20,
)

CALIBRATION_LADDER = (
    NOMINAL_PROFILE,
    CALIBRATION_LAG12,
    CALIBRATION_LAG14,
    CALIBRATION_LAG16,
    CALIBRATION_LAG20,
)

PROFILES = {profile.name: profile for profile in CALIBRATION_LADDER}


def replacement_sole_profile(scale: float) -> FootGeometryProfile:
    """Build one predeclared, symmetric replacement-sole condition."""

    percentage = round(scale * 100)
    return FootGeometryProfile(
        name=f"replacement-sole-uniform-{percentage}pct-v1",
        base_profile=NOMINAL_PROFILE,
        uniform_scale=scale,
    )


# This bounded ladder is declared before source-policy calibration.  Calibration
# may select one member; it must not invent an easier condition after seeing the
# policy's results.
REPLACEMENT_SOLE_CALIBRATION_PROFILES = tuple(
    replacement_sole_profile(value) for value in (0.70, 0.85, 1.15, 1.30, 1.45)
)
PROFILES.update(
    {profile.name: profile for profile in REPLACEMENT_SOLE_CALIBRATION_PROFILES}
)


def replacement_foot_profile(friction_scale: float) -> ReplacementFootProfile:
    """Build the second-stage 130% geometry plus changed-material condition."""

    percentage = round(friction_scale * 100)
    return ReplacementFootProfile(
        name=f"replacement-foot-130pct-grip-{percentage}pct-v1",
        base_profile=NOMINAL_PROFILE,
        uniform_scale=1.30,
        friction_scale=friction_scale,
    )


# Declared only after the geometry-only v1 ladder showed that the source policy
# was insensitive to size but the original command bank did not pass nominal.
REPLACEMENT_FOOT_CALIBRATION_PROFILES = tuple(
    replacement_foot_profile(value) for value in (0.80, 0.60, 0.40, 0.25)
)
PROFILES.update(
    {profile.name: profile for profile in REPLACEMENT_FOOT_CALIBRATION_PROFILES}
)


def priority_foot_material_profile(slide_friction: float) -> PriorityFootMaterialProfile:
    """Build a physically effective fixed-material replacement foot."""

    label = str(slide_friction).replace(".", "p")
    return PriorityFootMaterialProfile(
        name=f"replacement-foot-130pct-mu-{label}-priority1-v1",
        base_profile=NOMINAL_PROFILE,
        uniform_scale=1.30,
        slide_friction=slide_friction,
    )


# The preceding friction-scale ladder was intentionally preserved after proving
# that equal-priority MuJoCo contact pairs selected the floor's larger friction.
# This ladder changes the sole material with the correct contact-pair semantics.
PRIORITY_FOOT_MATERIAL_CALIBRATION_PROFILES = tuple(
    priority_foot_material_profile(value) for value in (0.60, 0.35, 0.18, 0.08)
)
PROFILES.update(
    {
        profile.name: profile
        for profile in PRIORITY_FOOT_MATERIAL_CALIBRATION_PROFILES
    }
)


def wedge_foot_profile(pitch_degrees: float) -> WedgeFootProfile:
    label = str(round(pitch_degrees)).replace(".", "p")
    return WedgeFootProfile(
        name=f"replacement-wedge-foot-pitch-{label}deg-v1",
        base_profile=NOMINAL_PROFILE,
        pitch_degrees=pitch_degrees,
    )


WEDGE_FOOT_CALIBRATION_PROFILES = tuple(
    wedge_foot_profile(value) for value in (5.0, 10.0, 15.0, 20.0)
)
PROFILES.update({profile.name: profile for profile in WEDGE_FOOT_CALIBRATION_PROFILES})


def asymmetric_left_leg_profile(effectiveness: float) -> AsymmetricActuatorProfile:
    label = str(round(effectiveness * 100))
    return AsymmetricActuatorProfile(
        name=f"asymmetric-left-knee-ankle-{label}pct-v1",
        base_profile=NOMINAL_PROFILE,
        joint_effectiveness=(
            ("left_knee", effectiveness),
            ("left_ankle", effectiveness),
        ),
    )


ASYMMETRIC_INITIAL_CALIBRATION_PROFILES = tuple(
    asymmetric_left_leg_profile(value) for value in (0.85, 0.75, 0.65, 0.55)
)

# Predeclared only after the initial ladder saturated at 4/4 smoke success.
ASYMMETRIC_CALIBRATION_PROFILES = tuple(
    asymmetric_left_leg_profile(value) for value in (0.45, 0.35, 0.25, 0.15)
)

# Frozen by the 2026-08-31 source-policy calibration.  The profile is hard
# enough to expose a useful deployment gap without making any reset category
# unrecoverable.  These identifiers are release evidence, not tunable defaults.
ASYMMETRIC_SELECTED_PROFILE = asymmetric_left_leg_profile(0.25)
ASYMMETRIC_SELECTED_PROFILE_SHA256 = (
    "4ff1208b44ac154772939fb07c2902b4c902e47b0b51def94fb5d163fa8a925a"
)
ASYMMETRIC_CALIBRATION_SEED = 20261001
ASYMMETRIC_CALIBRATION_EPISODES_PER_POSE = 8
ASYMMETRIC_SELECTED_BANK_SHA256 = (
    "6df62d7ef310e06d7d44437c8e3ac0a9c423d3b12d5c4bd97cdf6e0274dbe4fb"
)

if ASYMMETRIC_SELECTED_PROFILE.sha256 != ASYMMETRIC_SELECTED_PROFILE_SHA256:
    raise RuntimeError("Frozen asymmetric deployment profile has drifted")


def runtime_lag_capacity(
    profile: DeploymentConditionProfile,
) -> int:
    """Return the compiled delay-buffer capacity required for a profile.

    The actuator delay implementation requires a buffer capacity of at least six
    steps even when the fixed deployment lag is smaller.  This only sizes the
    buffer; :class:`DeploymentState` still applies the profile's exact lag.
    """

    return max(6, profile.actuator_lag_steps)


@dataclass(frozen=True)
class Scenario:
    """One reproducible episode specification."""

    scenario_id: str
    pose: str
    seed: int
    profile_name: str
    profile_sha256: str
    command: tuple[float, ...] = ZERO_COMMAND

    def __post_init__(self) -> None:
        if self.pose not in POSES:
            raise ValueError(f"Unknown pose {self.pose!r}")
        if len(self.command) != 13:
            raise ValueError("Scenario command must contain 13 values")

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["command"] = list(self.command)
        return result


def make_balanced_bank(
    *,
    profile: DeploymentProfile | AsymmetricActuatorProfile,
    base_seed: int,
    episodes_per_pose: int,
    prefix: str,
) -> tuple[Scenario, ...]:
    """Build an equal-pose deterministic bank with independently mixed seeds."""

    if episodes_per_pose <= 0:
        raise ValueError("episodes_per_pose must be positive")
    rng = np.random.default_rng(base_seed)
    scenarios: list[Scenario] = []
    for pose in POSES:
        for pose_index in range(episodes_per_pose):
            seed = int(rng.integers(0, np.iinfo(np.int32).max))
            scenarios.append(
                Scenario(
                    scenario_id=f"{prefix}-{pose}-{pose_index:03d}",
                    pose=pose,
                    seed=seed,
                    profile_name=profile.name,
                    profile_sha256=profile.sha256,
                )
            )
    return tuple(scenarios)


@dataclass
class _ActuatorSnapshot:
    actuator: Any
    vin_tensor: torch.Tensor
    vin_drop_gain: torch.Tensor | None
    min_lag: int | None
    max_lag: int | None
    per_env: bool | None
    hold_prob: float | None
    update_period: int | None
    per_env_phase: bool | None
    current_lags: torch.Tensor | None
    effort_scale: torch.Tensor | None


@dataclass
class _FootGeometrySnapshot:
    model: Any
    geom_ids: tuple[int, int]
    mesh_ranges: tuple[tuple[int, int], tuple[int, int]]
    mesh_vertices: tuple[torch.Tensor, torch.Tensor]
    geom_aabb: torch.Tensor
    geom_rbound: torch.Tensor
    geom_friction: torch.Tensor
    geom_priority: torch.Tensor
    geom_quat: torch.Tensor


class DeploymentState:
    """Restorable baseline for non-accumulating deployment-profile application."""

    def __init__(
        self,
        snapshots: list[_ActuatorSnapshot],
        foot_geometry: _FootGeometrySnapshot,
    ) -> None:
        self._snapshots = snapshots
        self._foot_geometry = foot_geometry

    @classmethod
    def capture(cls, env: Any) -> DeploymentState:
        snapshots: list[_ActuatorSnapshot] = []
        for actuator in env.scene["robot"].actuators:
            vin = getattr(actuator, "vin_tensor", None)
            if not isinstance(vin, torch.Tensor):
                continue
            sag = getattr(actuator, "vin_drop_gain", None)
            delay = getattr(actuator, "_delay_buffer", None)
            snapshots.append(
                _ActuatorSnapshot(
                    actuator=actuator,
                    vin_tensor=vin.clone(),
                    vin_drop_gain=sag.clone()
                    if isinstance(sag, torch.Tensor)
                    else None,
                    min_lag=int(delay.min_lag) if delay is not None else None,
                    max_lag=int(delay.max_lag) if delay is not None else None,
                    per_env=bool(delay.per_env) if delay is not None else None,
                    hold_prob=float(delay.hold_prob) if delay is not None else None,
                    update_period=(
                        int(delay.update_period) if delay is not None else None
                    ),
                    per_env_phase=(
                        bool(delay.per_env_phase) if delay is not None else None
                    ),
                    current_lags=(
                        delay.current_lags.clone() if delay is not None else None
                    ),
                    effort_scale=(
                        actuator.effort_scale.clone()
                        if isinstance(
                            getattr(actuator, "effort_scale", None), torch.Tensor
                        )
                        else None
                    ),
                )
            )
        if not snapshots:
            raise RuntimeError("Deployment profile found no BAM actuator state")
        robot = env.scene["robot"]
        geom_ids, geom_names = robot.find_geoms(
            "left_foot_collision|right_foot_collision"
        )
        if tuple(geom_names) != ("left_foot_collision", "right_foot_collision"):
            raise RuntimeError(
                "Deployment geometry requires exactly the two named foot collisions"
            )
        model = env.sim.model
        local_geom_ids = torch.tensor(geom_ids, dtype=torch.long)
        global_geom_ids = robot.indexing.geom_ids[local_geom_ids]
        geom_ids_tuple = (
            int(global_geom_ids[0].item()),
            int(global_geom_ids[1].item()),
        )
        mesh_ids = model.geom_dataid[0, list(geom_ids_tuple)].to(dtype=torch.long)
        if mesh_ids.shape != (2,) or bool(torch.any(mesh_ids < 0)):
            raise RuntimeError("Foot collision geoms must reference two mesh assets")
        if int(mesh_ids[0].item()) == int(mesh_ids[1].item()):
            raise RuntimeError("Left and right foot collisions must use distinct meshes")
        mesh_ranges: list[tuple[int, int]] = []
        mesh_vertices: list[torch.Tensor] = []
        for mesh_id_tensor in mesh_ids:
            mesh_id = int(mesh_id_tensor.item())
            start = int(model.mesh_vertadr[mesh_id].item())
            count = int(model.mesh_vertnum[mesh_id].item())
            if count <= 0:
                raise RuntimeError("Foot collision mesh has no vertices")
            stop = start + count
            mesh_ranges.append((start, stop))
            mesh_vertices.append(model.mesh_vert[start:stop].clone())
        foot_geometry = _FootGeometrySnapshot(
            model=model,
            geom_ids=geom_ids_tuple,
            mesh_ranges=(mesh_ranges[0], mesh_ranges[1]),
            mesh_vertices=(mesh_vertices[0], mesh_vertices[1]),
            geom_aabb=model.geom_aabb[:, list(geom_ids_tuple)].clone(),
            geom_rbound=model.geom_rbound[:, list(geom_ids_tuple)].clone(),
            geom_friction=model.geom_friction[:, list(geom_ids_tuple)].clone(),
            geom_priority=model.geom_priority[list(geom_ids_tuple)].clone(),
            geom_quat=model.geom_quat[:, list(geom_ids_tuple)].clone(),
        )
        return cls(snapshots, foot_geometry)

    def restore(self) -> None:
        for snapshot in self._snapshots:
            actuator = snapshot.actuator
            actuator.vin_tensor.copy_(snapshot.vin_tensor)
            if snapshot.vin_drop_gain is not None:
                actuator.vin_drop_gain.copy_(snapshot.vin_drop_gain)
            delay = getattr(actuator, "_delay_buffer", None)
            if delay is not None and snapshot.min_lag is not None:
                delay.min_lag = snapshot.min_lag
                delay.max_lag = snapshot.max_lag
                delay.per_env = snapshot.per_env
                delay.hold_prob = snapshot.hold_prob
                delay.update_period = snapshot.update_period
                delay.per_env_phase = snapshot.per_env_phase
                if snapshot.current_lags is not None:
                    delay.set_lags(snapshot.current_lags)

            if snapshot.effort_scale is not None:
                actuator.effort_scale.copy_(snapshot.effort_scale)

        geometry = self._foot_geometry
        for (start, stop), vertices in zip(
            geometry.mesh_ranges, geometry.mesh_vertices, strict=True
        ):
            geometry.model.mesh_vert[start:stop].copy_(vertices)
        geometry.model.geom_aabb[:, list(geometry.geom_ids)] = geometry.geom_aabb
        geometry.model.geom_rbound[:, list(geometry.geom_ids)] = geometry.geom_rbound
        geometry.model.geom_friction[:, list(geometry.geom_ids)] = (
            geometry.geom_friction
        )
        geometry.model.geom_priority[list(geometry.geom_ids)] = geometry.geom_priority
        geometry.model.geom_quat[:, list(geometry.geom_ids)] = geometry.geom_quat

    def apply(
        self,
        profile: (
            DeploymentProfile
            | AsymmetricActuatorProfile
            | FootGeometryProfile
            | ReplacementFootProfile
            | PriorityFootMaterialProfile
            | WedgeFootProfile
        ),
    ) -> None:
        """Restore the baseline and apply one fixed hidden hardware condition."""

        self.restore()
        base_profile = (
            profile.base_profile
            if isinstance(
                profile,
                (
                    AsymmetricActuatorProfile,
                    FootGeometryProfile,
                    ReplacementFootProfile,
                    PriorityFootMaterialProfile,
                    WedgeFootProfile,
                ),
            )
            else profile
        )
        for snapshot in self._snapshots:
            actuator = snapshot.actuator
            actuator.vin_tensor.fill_(base_profile.battery_voltage)
            if actuator.vin_drop_gain is None:
                raise RuntimeError("BAM actuator has no voltage-sag state")
            actuator.vin_drop_gain.fill_(base_profile.voltage_sag_gain)
            delay = getattr(actuator, "_delay_buffer", None)
            if delay is None:
                raise RuntimeError("BAM actuator has no configured delay buffer")
            if base_profile.actuator_lag_steps > delay._buffer.max_length - 1:
                raise ValueError("Profile lag exceeds the compiled delay history")
            delay.min_lag = base_profile.actuator_lag_steps
            delay.max_lag = base_profile.actuator_lag_steps
            delay.per_env = False
            delay.hold_prob = 1.0
            delay.update_period = 0
            delay.per_env_phase = False
            lags = torch.full(
                (delay.batch_size,),
                base_profile.actuator_lag_steps,
                device=actuator.vin_tensor.device,
                dtype=torch.long,
            )
            delay.set_lags(lags)

            if isinstance(profile, AsymmetricActuatorProfile):
                effort = getattr(actuator, "effort_scale", None)
                names = list(getattr(actuator, "_target_names", ()))
                if not isinstance(effort, torch.Tensor):
                    raise TypeError("BAM actuator has no effort-scale state")
                for joint_name, effectiveness in profile.joint_effectiveness:
                    if joint_name not in names:
                        raise ValueError(f"Actuator has no target joint {joint_name!r}")
                    effort[:, names.index(joint_name)] = effectiveness

        geometry = self._foot_geometry
        if isinstance(
            profile,
            (
                FootGeometryProfile,
                ReplacementFootProfile,
                PriorityFootMaterialProfile,
            ),
        ):
            for (start, stop), vertices in zip(
                geometry.mesh_ranges, geometry.mesh_vertices, strict=True
            ):
                geometry.model.mesh_vert[start:stop].copy_(
                    vertices * profile.uniform_scale
                )
            geometry.model.geom_aabb[:, list(geometry.geom_ids)] = (
                geometry.geom_aabb * profile.uniform_scale
            )
            geometry.model.geom_rbound[:, list(geometry.geom_ids)] = (
                geometry.geom_rbound * profile.uniform_scale
            )
            if isinstance(profile, ReplacementFootProfile):
                geometry.model.geom_friction[:, list(geometry.geom_ids)] = (
                    geometry.geom_friction * profile.friction_scale
                )
            if isinstance(profile, PriorityFootMaterialProfile):
                friction = torch.tensor(
                    [
                        profile.slide_friction,
                        profile.slide_friction * 0.005,
                        profile.slide_friction * 0.0001,
                    ],
                    device=geometry.geom_friction.device,
                    dtype=geometry.geom_friction.dtype,
                )
                geometry.model.geom_friction[:, list(geometry.geom_ids)] = friction
                geometry.model.geom_priority[list(geometry.geom_ids)] = (
                    profile.contact_priority
                )

        if isinstance(profile, WedgeFootProfile):
            angle = math.radians(profile.pitch_degrees) * 0.5
            delta = torch.tensor(
                [math.cos(angle), 0.0, math.sin(angle), 0.0],
                device=geometry.geom_quat.device,
                dtype=geometry.geom_quat.dtype,
            )
            baseline = geometry.geom_quat
            w1, x1, y1, z1 = baseline.unbind(dim=-1)
            w2, x2, y2, z2 = delta.unbind(dim=-1)
            rotated = torch.stack(
                (
                    w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
                    w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
                    w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
                    w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
                ),
                dim=-1,
            )
            geometry.model.geom_quat[:, list(geometry.geom_ids)] = rotated


def bank_sha256(scenarios: tuple[Scenario, ...]) -> str:
    payload = json.dumps(
        [scenario.to_dict() for scenario in scenarios],
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    return hashlib.sha256(payload).hexdigest()


def frozen_asymmetric_calibration_bank() -> tuple[Scenario, ...]:
    """Rebuild the exact selected 32-world asymmetric calibration bank."""

    bank = make_balanced_bank(
        profile=ASYMMETRIC_SELECTED_PROFILE,
        base_seed=ASYMMETRIC_CALIBRATION_SEED,
        episodes_per_pose=ASYMMETRIC_CALIBRATION_EPISODES_PER_POSE,
        prefix=ASYMMETRIC_SELECTED_PROFILE.name,
    )
    actual_sha256 = bank_sha256(bank)
    if actual_sha256 != ASYMMETRIC_SELECTED_BANK_SHA256:
        raise RuntimeError(
            "Frozen asymmetric calibration bank has drifted: "
            f"expected {ASYMMETRIC_SELECTED_BANK_SHA256}, got {actual_sha256}"
        )
    return bank


def select_calibrated_profile(rows: list[dict[str, Any]]) -> dict[str, Any] | None:
    """Select the hardest measured shift that is neither trivial nor catastrophic."""

    if len(rows) < 2:
        raise ValueError("Calibration requires nominal plus at least one shift")
    nominal_rate = float(rows[0]["metrics"]["eval/objective/success_rate"])
    eligible: list[dict[str, Any]] = []
    if nominal_rate >= 0.75:
        for row in rows[1:]:
            rate = float(row["metrics"]["eval/objective/success_rate"])
            pose_rates = row["pose_terminal_success_rates"]
            if (
                0.10 <= rate <= min(0.90, nominal_rate - 0.10)
                and min(float(value) for value in pose_rates.values()) > 0.0
            ):
                eligible.append(row)
    return min(
        eligible,
        key=lambda row: float(row["metrics"]["eval/objective/success_rate"]),
        default=None,
    )
