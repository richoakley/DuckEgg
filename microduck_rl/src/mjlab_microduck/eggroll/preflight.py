"""Fail-fast CUDA and Torch/JAX DLPack validation for EGGROLL jobs."""

from __future__ import annotations

import os
import sys
from dataclasses import asdict, dataclass

import jax
import jaxlib
import torch

from .interop import jax_to_torch, torch_to_jax


@dataclass(frozen=True)
class CudaPreflightResult:
    """Machine-readable summary of a successful on-device preflight."""

    python_version: str
    jax_version: str
    jaxlib_version: str
    torch_version: str
    torch_cuda_version: str
    torch_device: str
    torch_device_name: str
    jax_backend: str
    jax_devices: str
    dlpack_shape: tuple[int, ...]

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def run_cuda_preflight(device: str = "cuda:0") -> CudaPreflightResult:
    """Validate CUDA device zero and a same-device Torch/JAX round trip.

    This function intentionally does not import mjlab or create a simulator.
    It is used both by the standalone preflight job and by the trainer before
    environment construction.
    """

    requested = torch.device(device)
    if requested.type != "cuda" or requested.index not in (None, 0):
        raise RuntimeError(
            f"The first EGGROLL smoke test requires device cuda:0, got {device!r}"
        )

    print("=" * 60, flush=True)
    print("EGGROLL CUDA PREFLIGHT", flush=True)
    print("=" * 60, flush=True)
    print(f"Python: {sys.version.split()[0]}", flush=True)
    print(f"jax: {jax.__version__}", flush=True)
    print(f"jaxlib: {jaxlib.__version__}", flush=True)
    print(f"torch: {torch.__version__}", flush=True)
    print(f"Torch built with CUDA: {torch.version.cuda}", flush=True)
    print(
        f"CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES', '<unset>')}",
        flush=True,
    )
    print(f"Torch CUDA available: {torch.cuda.is_available()}", flush=True)

    if not torch.cuda.is_available():
        raise RuntimeError("Torch cannot see CUDA; refusing to start mjlab")
    if torch.cuda.device_count() < 1:
        raise RuntimeError("Torch reports CUDA available but no CUDA devices")

    torch.cuda.set_device(0)
    torch_index = torch.cuda.current_device()
    torch_name = torch.cuda.get_device_name(torch_index)
    print(f"Torch CUDA device count: {torch.cuda.device_count()}", flush=True)
    print(f"Torch current device: cuda:{torch_index}", flush=True)
    print(f"Torch device: {torch_name}", flush=True)
    if torch_index != 0:
        raise RuntimeError(f"Torch selected cuda:{torch_index}; expected cuda:0")

    jax_backend = jax.default_backend()
    jax_devices = jax.devices()
    print(f"JAX backend: {jax_backend}", flush=True)
    print(f"JAX devices: {jax_devices}", flush=True)
    if jax_backend != "gpu":
        raise RuntimeError(
            f"JAX backend is {jax_backend!r}, not 'gpu'; CPU fallback is forbidden"
        )
    gpu_devices = [
        candidate for candidate in jax_devices if candidate.platform == "gpu"
    ]
    if not any(candidate.id == 0 for candidate in gpu_devices):
        raise RuntimeError(f"JAX cannot see GPU device 0; devices are {jax_devices}")

    source = torch.arange(8, device="cuda:0", dtype=torch.float32).reshape(2, 4)
    jax_source = torch_to_jax(source)
    jax_result = jax_source + 1.0
    restored = jax_to_torch(jax_result, device=torch.device("cuda:0"))
    torch.cuda.synchronize(0)

    if restored.device != torch.device("cuda:0"):
        raise RuntimeError(f"DLPack round trip moved data to {restored.device}")
    if restored.dtype != torch.float32:
        raise RuntimeError(f"DLPack round trip changed dtype to {restored.dtype}")
    if restored.shape != source.shape:
        raise RuntimeError(
            f"DLPack round trip changed shape from {source.shape} to {restored.shape}"
        )
    if not torch.equal(restored, source + 1.0):
        raise RuntimeError("DLPack round trip produced incorrect values")

    print(
        "DLPack Torch->JAX->op->Torch: PASS "
        f"(device={restored.device}, dtype={restored.dtype}, "
        f"shape={tuple(restored.shape)}, values=correct)",
        flush=True,
    )
    print("CUDA preflight: PASS", flush=True)
    print("=" * 60, flush=True)

    return CudaPreflightResult(
        python_version=sys.version.split()[0],
        jax_version=jax.__version__,
        jaxlib_version=jaxlib.__version__,
        torch_version=torch.__version__,
        torch_cuda_version=str(torch.version.cuda),
        torch_device="cuda:0",
        torch_device_name=torch_name,
        jax_backend=jax_backend,
        jax_devices=str(jax_devices),
        dlpack_shape=tuple(restored.shape),
    )
