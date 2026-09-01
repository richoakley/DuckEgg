"""Zero-copy Torch/JAX conversion helpers using the direct DLPack protocol."""

from __future__ import annotations

import jax
import torch


def torch_to_jax(tensor: torch.Tensor) -> jax.Array:
    """Expose a contiguous Torch tensor to JAX without a host round-trip.

    ``contiguous()`` is a no-op for mjlab's normal observation/reward buffers.
    If a caller supplies a strided view it may make a same-device copy; it
    never moves the value through CPU.
    """
    if tensor.dtype != torch.float32:
        raise TypeError(f"Expected a float32 Torch tensor, got {tensor.dtype}")
    if not tensor.is_contiguous():
        tensor = tensor.contiguous()
    array = jax.dlpack.from_dlpack(tensor)
    _validate_same_device(tensor.device, array)
    return array


def jax_to_torch(array: jax.Array, *, device: torch.device) -> torch.Tensor:
    """Expose a JAX array to Torch without a host round-trip."""
    if str(array.dtype) != "float32":
        raise TypeError(f"Expected a float32 JAX array, got {array.dtype}")
    tensor = torch.from_dlpack(array)
    if tensor.device != device:
        raise RuntimeError(
            f"DLPack changed devices: JAX output became {tensor.device}, expected {device}"
        )
    if not tensor.is_contiguous():
        tensor = tensor.contiguous()
    return tensor


def _validate_same_device(torch_device: torch.device, array: jax.Array) -> None:
    devices = array.devices()
    if len(devices) != 1:
        raise RuntimeError(f"Expected one JAX device for a DLPack array, got {devices}")
    jax_device = next(iter(devices))

    if torch_device.type == "cuda":
        if jax_device.platform not in {"cuda", "gpu"}:
            raise RuntimeError(
                f"Torch observations are on {torch_device}, but JAX used {jax_device}"
            )
        torch_index = (
            torch_device.index
            if torch_device.index is not None
            else torch.cuda.current_device()
        )
        if jax_device.id != torch_index:
            raise RuntimeError(
                f"Torch/JAX CUDA device mismatch: {torch_device} vs {jax_device}"
            )
    elif torch_device.type == "cpu" and jax_device.platform != "cpu":
        raise RuntimeError(f"Torch observations are on CPU, but JAX used {jax_device}")
