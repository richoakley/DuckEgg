"""Run a policy through the actual MicroDuck Rust production loader."""

from __future__ import annotations

import os
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .checkpoint import write_json
from .policy_io import import_deployed_policy
from .release import sha256_file


def _git_revision(repo: Path) -> tuple[str | None, bool | None]:
    if not (repo / ".git").exists():
        return None, None
    revision = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    dirty = bool(
        subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=repo,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    )
    return revision, dirty


def run_production_loader_probe(
    *,
    policy_path: Path,
    runtime_repo: Path,
    cargo: Path,
    output: Path,
    ort_dylib: Path | None = None,
) -> dict[str, Any]:
    """Compile and execute `duck-control::Policy::load` against one ONNX file."""

    deployed = import_deployed_policy(policy_path)
    loader = runtime_repo / "duck-control/src/policy.rs"
    observation = runtime_repo / "duck-control/src/obs.rs"
    probe = runtime_repo / "duck-control/examples/policy_probe.rs"
    lock = runtime_repo / "Cargo.lock"
    for required in (loader, observation, probe, lock):
        if not required.is_file():
            raise FileNotFoundError(f"MicroDuck runtime source is missing {required}")

    environment = os.environ.copy()
    if ort_dylib is not None:
        environment["ORT_DYLIB_PATH"] = str(ort_dylib.resolve())
    command = [
        str(cargo),
        "run",
        "--locked",
        "-p",
        "duck-control",
        "--example",
        "policy_probe",
        "--",
        str(policy_path.resolve()),
    ]
    completed = subprocess.run(
        command,
        cwd=runtime_repo,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
        timeout=600,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            "MicroDuck production policy loader rejected the derivative:\n"
            f"{completed.stdout}\n{completed.stderr}"
        )
    marker = "production-policy-load=passed"
    if marker not in completed.stdout:
        raise RuntimeError("Production loader probe returned no success marker")
    revision, dirty = _git_revision(runtime_repo)
    result: dict[str, Any] = {
        "format": "microduck-production-policy-loader-verification-v1",
        "verified_at": datetime.now(UTC).isoformat(),
        "passed": True,
        "policy_filename": policy_path.name,
        "policy_sha256": deployed.source_sha256,
        "production_path": "duck_control::policy::Policy::load",
        "warmup_inference": True,
        "observation_width": 61,
        "action_count": 14,
        "model_api": 1,
        "runtime_source": {
            "revision": revision,
            "dirty": dirty,
            "snapshot_without_git_metadata": revision is None,
            "policy_loader_sha256": sha256_file(loader),
            "observation_builder_sha256": sha256_file(observation),
            "probe_sha256": sha256_file(probe),
            "cargo_lock_sha256": sha256_file(lock),
        },
        "onnx_runtime": (
            {
                "filename": ort_dylib.name,
                "sha256": sha256_file(ort_dylib),
            }
            if ort_dylib is not None
            else {"source": "system ORT_DYLIB_PATH/default loader path"}
        ),
        "stdout": completed.stdout.strip(),
    }
    write_json(output, result)
    return result
