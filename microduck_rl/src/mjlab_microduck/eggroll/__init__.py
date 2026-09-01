"""EGGROLL training support for Microduck.

The simulator and PyTorch own most of the GPU memory in this process.  Disable
JAX's default up-front allocation before importing JAX/HyperscaleES so the two
frameworks can coexist on one device.
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING, Any

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

if TYPE_CHECKING:
    from .policy import OutputLayerPolicy, PostTrainingPolicyConfig

__all__ = ["OutputLayerPolicy", "PostTrainingPolicyConfig"]


def __getattr__(name: str) -> Any:
    """Keep the public imports lazy so CUDA preflight loads only JAX/Torch."""

    if name in __all__:
        from .policy import OutputLayerPolicy, PostTrainingPolicyConfig

        return {
            "OutputLayerPolicy": OutputLayerPolicy,
            "PostTrainingPolicyConfig": PostTrainingPolicyConfig,
        }[name]
    raise AttributeError(name)
