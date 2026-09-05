"""Submit the pre-training trunk-CoM source-behavior release-bank preflight."""

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

from mjlab_microduck.autopatch.contracts import PatchCampaign
from mjlab_microduck.autopatch.walking_protocol import (
    resolve_walking_protocol,
    walking_campaign_family_sha256,
)
from mjlab_microduck.hf_jobs import DEFAULT_IMAGE, _pick_namespace, _repo_root


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _source_behavior_bootstrap() -> str:
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
            "  # Source-behavior capture bundles its exact source policy, so the shared "
            "reference-policy bucket is not required.\n"
        ),
    )
    return (
        setup
        + r"""
mkdir -p /work/reference /work/output
cp .artifacts/input/source_manifest.json /work/output/source_manifest.json
cp .artifacts/input/campaign.json /work/output/campaign.json
cp .artifacts/input/calibration_validation.json \
  /work/output/calibration_validation.json
set +e
uv run --no-sync eggroll-autopatch capture-walking-source-behavior \
  --campaign .artifacts/input/campaign.json \
  --calibration-validation .artifacts/input/calibration_validation.json \
  --source-manifest .artifacts/input/source_manifest.json \
  --runtime-repo /work/microduck \
  --robotd "$EGGROLL_PRODUCTION_ROBOTD" \
  --ort-dylib "$EGGROLL_PRODUCTION_ORT_DYLIB" \
  --output-dir /work/reference
RUN_RC=$?
set -e
if [ -f /work/reference/source_behavior_reference.json ]; then
  cp /work/reference/source_behavior_reference.json \
    /work/output/source_behavior_reference.json
fi
uv run --no-sync hf upload "$OUTPUT_REPO" /work/reference reference \
  --repo-type model --private --token "$HF_TOKEN" || true
uv run --no-sync hf upload "$OUTPUT_REPO" /work/output . \
  --repo-type model --private --token "$HF_TOKEN" || true
exit "$RUN_RC"
"""
    )


def _build_source_behavior_bundle(
    *,
    repo: Path,
    output: Path,
    policy: Path,
    campaign_path: Path,
    calibration_validation_path: Path,
    hf_hardware_flavor: str = "unbound-local-test",
) -> dict[str, Any]:
    _require_clean_repo(repo)
    if not hf_hardware_flavor:
        raise ValueError("HF hardware flavor cannot be empty")
    campaign = PatchCampaign.from_json(campaign_path.read_text())
    protocol = resolve_walking_protocol(campaign)
    if not protocol.source_behavior_reference_required:
        raise ValueError("source-behavior preflight requires the new incident protocol")
    source_sha256 = _sha256(policy)
    if source_sha256 != campaign.artifact_sha256:
        raise ValueError("source-behavior policy differs from the campaign")
    calibration = json.loads(calibration_validation_path.read_text())
    if not isinstance(calibration, dict) or calibration.get("status") != "pass":
        raise ValueError("calibration validation did not pass")
    if calibration.get("selected_profile_sha256") != protocol.profile.sha256:
        raise ValueError("campaign profile differs from the calibrated condition")
    if calibration.get("source_policy_sha256") != campaign.artifact_sha256:
        raise ValueError("campaign source differs from the calibrated policy")
    if calibration.get("hf_hardware_flavor") != hf_hardware_flavor:
        raise ValueError("source-behavior preflight must use the calibration hardware")
    campaign_raw = (
        json.dumps(campaign.canonical_dict(), indent=2, sort_keys=True) + "\n"
    ).encode()
    calibration_raw = (
        json.dumps(calibration, indent=2, sort_keys=True) + "\n"
    ).encode()
    manifest = {
        "schema": "eggroll-autopatch-source-behavior-source-v1",
        "evidence_role": "source-only release-bank preflight; no optimization",
        "source_commit": _git_output(repo, "rev-parse", "HEAD"),
        "source_branch": _git_output(repo, "rev-parse", "--abbrev-ref", "HEAD"),
        "hf_hardware_flavor": hf_hardware_flavor,
        "source_policy_sha256": source_sha256,
        "capture_campaign_sha256": campaign.sha256,
        "campaign_family_sha256": walking_campaign_family_sha256(campaign),
        "walking_protocol_id": protocol.protocol_id,
        "activation_profile_sha256": protocol.profile.sha256,
        "calibration_validation_sha256": hashlib.sha256(calibration_raw).hexdigest(),
        "policy_episode_evaluation_ceiling": 64,
        "requested_simulator_step_ceiling": 16_000,
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
        documents = (
            (
                "source_manifest.json",
                (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode(),
            ),
            ("campaign.json", campaign_raw),
            ("calibration_validation.json", calibration_raw),
        )
        for name, raw in documents:
            info = tarfile.TarInfo(f"microduck_rl/.artifacts/input/{name}")
            info.size = len(raw)
            info.mode = 0o644
            archive.addfile(info, io.BytesIO(raw))
    return manifest


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy", type=Path, required=True)
    parser.add_argument("--campaign", type=Path, required=True)
    parser.add_argument("--calibration-validation", type=Path, required=True)
    parser.add_argument("--namespace")
    parser.add_argument("--output-repo", required=True)
    parser.add_argument("--flavor", default="a10g-small")
    parser.add_argument("--timeout", default="2h")
    parser.add_argument("--image", default=DEFAULT_IMAGE)
    return parser


def main() -> int:
    args = _parser().parse_args()
    policy = args.policy.resolve()
    campaign = args.campaign.resolve()
    calibration = args.calibration_validation.resolve()
    for path in (policy, campaign, calibration):
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
    source_repo = f"{namespace}/eggroll-autopatch-source-behavior-{stamp}"
    bootstrap = _source_behavior_bootstrap()
    with tempfile.TemporaryDirectory() as temporary:
        bundle = Path(temporary) / f"source-{stamp}.tar.gz"
        manifest = _build_source_behavior_bundle(
            repo=repo,
            output=bundle,
            policy=policy,
            campaign_path=campaign,
            calibration_validation_path=calibration,
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
