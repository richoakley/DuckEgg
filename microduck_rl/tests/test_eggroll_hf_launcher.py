"""Local safety and provenance tests for the private HF job launcher."""

from __future__ import annotations

import json
import runpy
import subprocess
import tarfile
from pathlib import Path

import pytest

_LAUNCHER = Path(__file__).parents[1] / "scripts/hf/eggroll_posttrain_hf.py"
_LAUNCHER_GLOBALS = runpy.run_path(str(_LAUNCHER))
_build_bundle = _LAUNCHER_GLOBALS["_build_bundle"]
_BOOTSTRAP = _LAUNCHER_GLOBALS["BOOTSTRAP"]


def _git(repo: Path, *arguments: str) -> None:
    subprocess.check_call(
        [
            "git",
            "-c",
            "user.name=Eggroll Test",
            "-c",
            "user.email=eggroll-test@example.invalid",
            *arguments,
        ],
        cwd=repo,
    )


def test_bundle_requires_clean_commit_and_records_exact_inputs(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init", "-q")
    tracked = repo / "tracked.txt"
    tracked.write_text("committed\n")
    _git(repo, "add", "tracked.txt")
    _git(repo, "commit", "-q", "-m", "fixture")

    policy = tmp_path / "policy.onnx"
    policy.write_bytes(b"exact-policy-bytes")
    calibration = tmp_path / "calibration.json"
    calibration.write_text('{"selected_profile": "fixture"}\n')
    output = tmp_path / "source.tar.gz"

    leak = repo / "untracked-secret.txt"
    leak.write_text("must not upload\n")
    with pytest.raises(RuntimeError, match="uncommitted source snapshot"):
        _build_bundle(
            repo=repo,
            output=output,
            policy=policy,
            calibration=calibration,
        )

    leak.unlink()
    _build_bundle(
        repo=repo,
        output=output,
        policy=policy,
        calibration=calibration,
    )
    with tarfile.open(output, "r:gz") as archive:
        names = set(archive.getnames())
        assert names == {
            "microduck_rl/tracked.txt",
            "microduck_rl/.artifacts/input/source_policy.onnx",
            "microduck_rl/.artifacts/input/calibration.json",
            "microduck_rl/.artifacts/input/source_manifest.json",
        }
        manifest_file = archive.extractfile(
            "microduck_rl/.artifacts/input/source_manifest.json"
        )
        assert manifest_file is not None
        manifest = json.load(manifest_file)

    expected_branch = subprocess.check_output(
        ["git", "branch", "--show-current"], cwd=repo, text=True
    ).strip()
    assert manifest["source_branch"] == expected_branch
    assert len(manifest["source_commit"]) == 40
    assert manifest["source_policy_sha256"] == (
        "f5c409fdacad6b3205148ca17a900b84599571a53a8cb0233919a7ef90b01217"
    )
    assert manifest["calibration_sha256"] == (
        "670febb37fc579916164d024c8dccd5ebd009a7c9bc67a9c9f992c75e74d3a2e"
    )


def test_remote_collection_failure_propagates_after_artifact_upload() -> None:
    assert 'eval "$COLLECT_COMMAND" || COLLECT_RC=$?' in _BOOTSTRAP
    assert 'exit "$COLLECT_RC"' in _BOOTSTRAP
    assert 'eval "$COLLECT_COMMAND" || true' not in _BOOTSTRAP
