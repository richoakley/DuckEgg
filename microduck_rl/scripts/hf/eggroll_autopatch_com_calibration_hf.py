"""Submit a predeclared source-only walking physical-condition calibration."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import tarfile
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from eggroll_autopatch_hf import (
    BOOTSTRAP,
    PRODUCTION_BUCKET,
    PRODUCTION_RUNTIME_VOLUME_PATH,
    _git_output,
    _require_clean_repo,
    _tracked_files,
)
from huggingface_hub import HfApi, Volume, get_token

from mjlab_microduck.hf_jobs import DEFAULT_IMAGE, _pick_namespace, _repo_root


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _canonical_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _calibration_bootstrap(
    *,
    calibrate_command: str = "calibrate-trunk-com",
    verify_command: str = "verify-trunk-com-calibration",
    base_seed: int = 20293001,
) -> str:
    setup, marker, _tail = BOOTSTRAP.partition("mkdir -p /work/run /work/output")
    if not marker:
        raise RuntimeError("HF bootstrap setup boundary changed")
    reference_copy = (
        "  mkdir -p /work/microduck/example_policies\n"
        "  cp -a /inputs/reference_policies/example_policies/. "
        "/work/microduck/example_policies/\n"
    )
    if reference_copy not in setup:
        raise RuntimeError("HF reference-policy copy boundary changed")
    setup = setup.replace(
        reference_copy,
        (
            "  mkdir -p /work/microduck/example_policies\n"
            "  # Calibration bundles their exact source policy, so the shared "
            "reference-policy bucket is not required.\n"
        ),
    )
    body = r"""
mkdir -p /work/calibration /work/output
cp .artifacts/input/source_manifest.json /work/output/source_manifest.json
cp .artifacts/input/calibration_protocol.json /work/output/calibration_protocol.json
set +e
uv run --no-sync eggroll-autopatch __CALIBRATE_COMMAND__ \
  --runtime-repo /work/microduck \
  --robotd "$EGGROLL_PRODUCTION_ROBOTD" \
  --ort-dylib "$EGGROLL_PRODUCTION_ORT_DYLIB" \
  --base-seed __BASE_SEED__ \
  --device cuda:0 \
  --timeout 120 \
  --attempts 1 \
  --output-dir /work/calibration
RUN_RC=$?
set -e
if [ "$RUN_RC" -eq 0 ]; then
  cp /work/calibration/manifest.json /work/output/calibration_manifest.json
  uv run --no-sync eggroll-autopatch __VERIFY_COMMAND__ \
    --manifest /work/calibration/manifest.json \
    --protocol .artifacts/input/calibration_protocol.json \
    --source-manifest .artifacts/input/source_manifest.json \
    --output /work/output/calibration_validation.json || RUN_RC=$?
fi
uv run --no-sync hf upload "$OUTPUT_REPO" /work/calibration calibration \
  --repo-type model --private --token "$HF_TOKEN" || true
uv run --no-sync hf upload "$OUTPUT_REPO" /work/output . \
  --repo-type model --private --token "$HF_TOKEN" || true
exit "$RUN_RC"
"""
    body = body.replace("__CALIBRATE_COMMAND__", calibrate_command)
    body = body.replace("__VERIFY_COMMAND__", verify_command)
    body = body.replace("__BASE_SEED__", str(base_seed))
    return setup + body


def _build_calibration_bundle(
    *,
    repo: Path,
    output: Path,
    policy: Path,
    protocol_path: Path,
    hf_hardware_flavor: str = "unbound-local-test",
) -> dict[str, Any]:
    _require_clean_repo(repo)
    if not hf_hardware_flavor:
        raise ValueError("HF hardware flavor cannot be empty")
    protocol = json.loads(protocol_path.read_text())
    if not isinstance(protocol, dict):
        raise TypeError("cross-failure calibration protocol must be an object")
    source_schema = {
        "eggroll-autopatch-cross-failure-protocol-v2": (
            "eggroll-autopatch-trunk-com-calibration-source-v1"
        ),
        "eggroll-autopatch-payload-cross-failure-protocol-v1": (
            "eggroll-autopatch-trunk-payload-calibration-source-v1"
        ),
    }.get(protocol.get("schema"))
    if source_schema is None:
        raise ValueError("unknown cross-failure calibration protocol")
    source_sha256 = _sha256(policy)
    if protocol.get("source", {}).get("policy_sha256") != source_sha256:
        raise ValueError("calibration policy differs from the predeclared source")
    if protocol.get("calibration", {}).get("base_seed") not in {
        20293001,
        20794001,
    }:
        raise ValueError("calibration base seed differs from the launcher contract")
    manifest = {
        "schema": source_schema,
        "evidence_role": "source-only condition calibration; no optimization",
        "source_commit": _git_output(repo, "rev-parse", "HEAD"),
        "source_branch": _git_output(repo, "rev-parse", "--abbrev-ref", "HEAD"),
        "hf_hardware_flavor": hf_hardware_flavor,
        "source_policy_sha256": source_sha256,
        "protocol_sha256": _sha256(protocol_path),
        "protocol_canonical_sha256": _canonical_sha256(protocol),
        "policy_episode_evaluation_ceiling": 160,
        "requested_simulator_step_ceiling": 40_000,
        "candidate_optimization_evaluation_ceiling": 0,
    }
    with tarfile.open(output, "w:gz") as archive:
        for relative in _tracked_files(repo):
            path = repo / relative
            if path.is_file() and ".DS_Store" not in path.parts:
                archive.add(path, arcname=str(Path("microduck_rl") / relative))
        archive.add(
            policy,
            arcname="microduck/example_policies/alpha_walking.onnx",
        )
        for name, value in (("source_manifest.json", manifest),):
            raw = (json.dumps(value, indent=2, sort_keys=True) + "\n").encode()
            info = tarfile.TarInfo(f"microduck_rl/.artifacts/input/{name}")
            info.size = len(raw)
            info.mode = 0o644
            archive.addfile(info, io.BytesIO(raw))
        protocol_raw = protocol_path.read_bytes()
        protocol_info = tarfile.TarInfo(
            "microduck_rl/.artifacts/input/calibration_protocol.json"
        )
        protocol_info.size = len(protocol_raw)
        protocol_info.mode = 0o644
        archive.addfile(protocol_info, io.BytesIO(protocol_raw))
    return manifest


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--namespace")
    parser.add_argument("--output-repo", required=True)
    parser.add_argument("--flavor", default="l4x1")
    parser.add_argument("--timeout", default="4h")
    parser.add_argument("--image", default=DEFAULT_IMAGE)
    return parser


def main() -> int:
    args = _parser().parse_args()
    policy = args.policy.resolve()
    protocol = args.protocol.resolve()
    for path in (policy, protocol):
        if not path.is_file():
            raise FileNotFoundError(path)
    token = get_token()
    if not token:
        raise RuntimeError("no cached Hugging Face token")
    api = HfApi(token=token)
    namespace = _pick_namespace(api, args.namespace)
    output_repo = args.output_repo
    if "/" not in output_repo:
        output_repo = f"{namespace}/{output_repo}"
    if api.repo_exists(output_repo, repo_type="model"):
        raise RuntimeError(f"refusing to reuse output repository {output_repo}")

    repo = _repo_root()
    stamp = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
    protocol_document = json.loads(protocol.read_text())
    if protocol_document.get("schema") == (
        "eggroll-autopatch-payload-cross-failure-protocol-v1"
    ):
        source_label = "payload"
        bootstrap = _calibration_bootstrap(
            calibrate_command="calibrate-trunk-payload",
            verify_command="verify-trunk-payload-calibration",
            base_seed=20794001,
        )
    else:
        source_label = "com"
        bootstrap = _calibration_bootstrap()
    source_repo = (
        f"{namespace}/eggroll-autopatch-{source_label}-calibration-source-{stamp}"
    )
    with tempfile.TemporaryDirectory() as temporary:
        bundle = Path(temporary) / f"source-{stamp}.tar.gz"
        manifest = _build_calibration_bundle(
            repo=repo,
            output=bundle,
            policy=policy,
            protocol_path=protocol,
            hf_hardware_flavor=args.flavor,
        )
        bundle_sha256 = _sha256(bundle)
        api.create_repo(source_repo, repo_type="dataset", private=True, exist_ok=False)
        api.upload_file(
            path_or_fileobj=str(bundle),
            path_in_repo=bundle.name,
            repo_id=source_repo,
            repo_type="dataset",
        )
        api.create_repo(output_repo, repo_type="model", private=True, exist_ok=False)
        job = api.run_job(
            image=args.image,
            command=["bash", "-c", bootstrap],
            env={
                "SRC_TARBALL": bundle.name,
                "OUTPUT_REPO": output_repo,
                "EGGROLL_HF_HARDWARE_FLAVOR": args.flavor,
            },
            secrets={"HF_TOKEN": token},
            flavor=args.flavor,
            timeout=args.timeout,
            volumes=[
                Volume(
                    type="dataset",
                    source=source_repo,
                    mount_path="/src",
                    read_only=True,
                ),
                Volume(
                    type="bucket",
                    source=PRODUCTION_BUCKET,
                    path=PRODUCTION_RUNTIME_VOLUME_PATH,
                    mount_path="/inputs/microduck",
                    read_only=True,
                ),
            ],
            namespace=namespace,
        )
    print(f"[job] {job.id}")
    print(f"[source] {source_repo}")
    print(f"[artifacts] https://huggingface.co/{output_repo}")
    print(
        json.dumps(
            {**manifest, "source_bundle_sha256": bundle_sha256},
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
