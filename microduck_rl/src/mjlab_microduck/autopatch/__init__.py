"""Policy-agnostic EGGROLL patch contracts for the MicroDuck policy fleet."""

from .contracts import (
    CapabilitySpec,
    DeploymentCondition,
    ObjectiveSpec,
    OptimizerSpec,
    PatchCampaign,
    PolicyArtifactSpec,
    ReleaseGate,
    TransitionSpec,
)
from .registry import PRODUCTION_REGISTRY, AutopatchRegistry

__all__ = [
    "PRODUCTION_REGISTRY",
    "AutopatchRegistry",
    "CapabilitySpec",
    "DeploymentCondition",
    "ObjectiveSpec",
    "OptimizerSpec",
    "PatchCampaign",
    "PolicyArtifactSpec",
    "ReleaseGate",
    "TransitionSpec",
]
