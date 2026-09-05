"""Submit one checkpoint for exact remote Autopatch qualification."""

from __future__ import annotations

import argparse
import hashlib
import json
import py_compile
import tempfile
from datetime import UTC, datetime
from pathlib import Path

from eggroll_autopatch_hf import (
    BOOTSTRAP,
    PRODUCTION_BUCKET,
    PRODUCTION_RUNTIME_VOLUME_PATH,
    REFERENCE_POLICIES_VOLUME_PATH,
    _build_bundle,
)
from huggingface_hub import HfApi, Volume, get_token

from mjlab_microduck.hf_jobs import DEFAULT_IMAGE, _pick_namespace, _repo_root


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _driver_source(
    generation: int, evidence_role: str = "environment-preflight"
) -> str:
    if evidence_role not in ("environment-preflight", "candidate-qualification"):
        raise ValueError(f"unknown qualification evidence role {evidence_role!r}")
    is_preflight = evidence_role == "environment-preflight"
    output_name = (
        "qualification_environment_validation.json"
        if is_preflight
        else "qualification_result.json"
    )
    schema = (
        "eggroll-autopatch-qualification-environment-validation-v1"
        if is_preflight
        else "eggroll-autopatch-candidate-qualification-v1"
    )
    candidate_origin = (
        "historical generation-85 output weights rebound to the frozen v2 campaign "
        "solely for environment validation"
        if is_preflight
        else "mechanically selected checkpoint from a frozen efficiency campaign"
    )
    selection_reason = (
        "historical known-good derivative used only for environment preflight"
        if is_preflight
        else "profile-specific campaign-side gates passed; nominal adapted-policy "
        "evaluation remains diagnostic because the release scope routes exact source "
        "bytes outside the attested activation profile"
    )
    claim_boundary = (
        "remote production-runtime environment and six-stage orchestration validation "
        "only; not a new training or efficiency result and no physical robot"
        if is_preflight
        else "candidate-bound six-stage simulation and production-runtime digital-twin "
        "qualification; not physical-robot evidence"
    )
    return f"""from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

from mjlab_microduck.autopatch.campaign import load_candidate_checkpoint
from mjlab_microduck.autopatch.contracts import PatchCampaign
from mjlab_microduck.autopatch.efficiency import InteractionCost
from mjlab_microduck.autopatch.qualification import (
    QualificationCandidate,
    QualificationController,
    QualificationPlan,
)
from mjlab_microduck.autopatch.qualification_command import (
    CommandQualificationBackend,
    CommandQualificationSpec,
)

ROOT = Path("/work/microduck_rl")
GENERATION = {generation}
CANDIDATE = Path(f"/work/input/generation-{{GENERATION:06d}}.npz")
EVIDENCE = Path("/work/qualification_evidence")
OUTPUT = Path("/work/output/{output_name}")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


campaign = PatchCampaign.from_json((ROOT / ".artifacts/input/campaign.json").read_text())
plan = QualificationPlan.from_json(
    (ROOT / ".artifacts/input/qualification_plan.json").read_text()
)
spec = CommandQualificationSpec.from_json(
    (ROOT / ".artifacts/input/qualification_command_spec.json").read_text()
)
checkpoint, _weight, _bias = load_candidate_checkpoint(CANDIDATE, campaign=campaign)
if checkpoint.generation != GENERATION:
    raise RuntimeError("preflight candidate generation does not match the declaration")
if sha256(CANDIDATE) != os.environ["CANDIDATE_SHA256"]:
    raise RuntimeError("preflight candidate bytes changed")
backend = CommandQualificationBackend(
    spec=spec,
    plan=plan,
    candidate_directory=CANDIDATE.parent,
    evidence_directory=EVIDENCE,
    working_directory=ROOT,
)
controller = QualificationController(plan)
status = controller.qualify(
    QualificationCandidate(
        generation=checkpoint.generation,
        checkpoint_sha256=sha256(CANDIDATE),
        selection_metrics=tuple(sorted(checkpoint.metric_map().items())),
        selection_passed=True,
        selection_reason={selection_reason!r},
        selection_cost=InteractionCost(),
    ),
    backend,
)
adapted = EVIDENCE / f"generation-{{GENERATION}}/adapted_policy.onnx"
files = sorted(path for path in EVIDENCE.rglob("*") if path.is_file())
payload = {{
    "schema": {schema!r},
    "status": "pass" if status == "eligible" else "fail",
    "qualification_status": status,
    "source_commit": os.environ["SOURCE_COMMIT"],
    "source_bundle_sha256": os.environ["SOURCE_BUNDLE_SHA256"],
    "preflight_bootstrap_sha256": os.environ["PREFLIGHT_BOOTSTRAP_SHA256"],
    "candidate_checkpoint_sha256": sha256(CANDIDATE),
    "candidate_generation_label": GENERATION,
    "candidate_origin": {candidate_origin!r},
    "expected_historical_adapted_policy_sha256": (
        os.environ.get("EXPECTED_ADAPTED_SHA256") or None
    ),
    "exported_adapted_policy_sha256": sha256(adapted) if adapted.is_file() else None,
    "plan_sha256": plan.sha256,
    "command_spec_sha256": spec.sha256,
    "controller": controller.state_dict(),
    "evidence_files": [
        {{"path": str(path.relative_to(EVIDENCE)), "sha256": sha256(path)}}
        for path in files
    ],
    "claim_boundary": {claim_boundary!r},
}}
OUTPUT.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\\n")
if payload["status"] != "pass":
    raise SystemExit("complete qualification preflight did not pass")
if payload["expected_historical_adapted_policy_sha256"] is not None and (
    payload["exported_adapted_policy_sha256"]
    != payload["expected_historical_adapted_policy_sha256"]
):
    raise SystemExit("preflight export differs from the historical adapted policy")
"""


def _qualification_bootstrap() -> str:
    setup, marker, _tail = BOOTSTRAP.partition("mkdir -p /work/run /work/output")
    if not marker:
        raise RuntimeError("HF bootstrap setup boundary changed")
    return (
        setup
        + r"""
mkdir -p /work/input /work/output /work/qualification_evidence
cp "/src/$CANDIDATE_FILENAME" "/work/input/$CANDIDATE_FILENAME"
cp .artifacts/input/source_manifest.json /work/output/source_manifest.json
cp .artifacts/input/campaign.json /work/output/campaign.json
cp .artifacts/input/release_scope.json /work/output/release_scope.json
cp .artifacts/input/qualification_plan.json /work/output/qualification_plan.json
cp .artifacts/input/qualification_command_spec.json \
  /work/output/qualification_command_spec.json
cp /src/preflight_driver.py /work/output/preflight_driver.py
set +e
uv run --no-sync python /src/preflight_driver.py
RUN_RC=$?
set -e
uv run --no-sync hf upload "$OUTPUT_REPO" /work/qualification_evidence \
  qualification_evidence --repo-type model --private --token "$HF_TOKEN" || true
uv run --no-sync hf upload "$OUTPUT_REPO" /work/output . \
  --repo-type model --private --token "$HF_TOKEN" || true
exit "$RUN_RC"
"""
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--generation", type=int, required=True)
    parser.add_argument("--expected-adapted-sha256")
    parser.add_argument(
        "--evidence-role",
        choices=("environment-preflight", "candidate-qualification"),
        default="environment-preflight",
    )
    parser.add_argument("--policy", type=Path, required=True)
    parser.add_argument("--campaign", type=Path, required=True)
    parser.add_argument("--release-scope", type=Path, required=True)
    parser.add_argument("--qualification-plan", type=Path, required=True)
    parser.add_argument("--qualification-command-spec", type=Path, required=True)
    parser.add_argument("--namespace")
    parser.add_argument("--output-repo", required=True)
    parser.add_argument("--flavor", default="a10g-small")
    parser.add_argument("--timeout", default="1h")
    parser.add_argument("--image", default=DEFAULT_IMAGE)
    return parser


def main() -> int:
    args = _parser().parse_args()
    if args.generation <= 0:
        raise ValueError("generation must be positive")
    paths = (
        args.candidate,
        args.policy,
        args.campaign,
        args.release_scope,
        args.qualification_plan,
        args.qualification_command_spec,
    )
    resolved = tuple(path.resolve() for path in paths)
    if any(not path.is_file() for path in resolved):
        missing = [str(path) for path in resolved if not path.is_file()]
        raise FileNotFoundError(f"preflight inputs are absent: {missing}")
    candidate, policy, campaign, release_scope, plan, spec = resolved
    if args.evidence_role == "environment-preflight":
        if (
            args.expected_adapted_sha256 is None
            or len(args.expected_adapted_sha256) != 64
        ):
            raise ValueError("environment preflight needs the expected adapted SHA-256")
    elif args.expected_adapted_sha256 is not None:
        raise ValueError("candidate qualification must discover its adapted SHA-256")
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
    source_repo = f"{namespace}/eggroll-autopatch-qualification-source-{stamp}"
    candidate_filename = f"generation-{args.generation:06d}.npz"
    candidate_sha256 = _sha256(candidate)
    bootstrap = _qualification_bootstrap()
    bootstrap_sha256 = hashlib.sha256(bootstrap.encode()).hexdigest()
    driver = _driver_source(args.generation, args.evidence_role)

    with tempfile.TemporaryDirectory() as temporary:
        temporary_root = Path(temporary)
        bundle = temporary_root / f"source-{stamp}.tar.gz"
        manifest = _build_bundle(
            repo=repo,
            output=bundle,
            policy=policy,
            campaign_path=campaign,
            release_scope_path=release_scope,
            mode="qualification-preflight",
            qualification_plan_path=plan,
            qualification_command_spec_path=spec,
            hf_hardware_flavor=args.flavor,
        )
        bundle_sha256 = _sha256(bundle)
        driver_path = temporary_root / "preflight_driver.py"
        driver_path.write_text(driver)
        py_compile.compile(str(driver_path), doraise=True)
        api.create_repo(source_repo, repo_type="dataset", private=True, exist_ok=False)
        for local, remote in (
            (bundle, bundle.name),
            (candidate, candidate_filename),
            (driver_path, "preflight_driver.py"),
        ):
            api.upload_file(
                path_or_fileobj=str(local),
                path_in_repo=remote,
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
                "RUN_MODE": "qualification-preflight",
                "CANDIDATE_FILENAME": candidate_filename,
                "CANDIDATE_SHA256": candidate_sha256,
                "EXPECTED_ADAPTED_SHA256": args.expected_adapted_sha256 or "",
                "SOURCE_COMMIT": manifest["source_commit"],
                "SOURCE_BUNDLE_SHA256": bundle_sha256,
                "PREFLIGHT_BOOTSTRAP_SHA256": bootstrap_sha256,
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
                Volume(
                    type="bucket",
                    source=PRODUCTION_BUCKET,
                    path=REFERENCE_POLICIES_VOLUME_PATH,
                    mount_path="/inputs/reference_policies",
                    read_only=True,
                ),
            ],
            namespace=namespace,
        )
    print(f"[job] {job.id}")
    print(f"[job-url] https://huggingface.co/jobs/{namespace}/{job.id}")
    print(f"[source] {source_repo}")
    print(f"[artifacts] https://huggingface.co/{output_repo}")
    print(
        json.dumps(
            {
                "source_commit": manifest["source_commit"],
                "source_bundle_sha256": bundle_sha256,
                "candidate_sha256": candidate_sha256,
                "preflight_bootstrap_sha256": bootstrap_sha256,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
